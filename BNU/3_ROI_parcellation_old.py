#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import argparse
from sklearn.cluster import KMeans
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import simlr  # 导入实现的 simlr 模块

def spectral_clustering(adjacency_matrix, num_clusters):
    """
    Python 版本的基于 sc3 的谱聚类函数，模仿 MATLAB 中:
      function index = sc3(k, W)
        ...
      end

    参数:
    ----------
    num_clusters : int
        目标聚类数 k
    adjacency_matrix : (n, n) ndarray
        相似度矩阵 (如由 matrix @ matrix.T 得到)，形状 (n, n)

    返回:
    ----------
    cluster_labels : (n,) ndarray
        每个点的聚类标签 (0..k-1)
    """

    # 1) 计算 L_sym = D^-1/2 * (D - adjacency_matrix) * D^-1/2
    num_points = adjacency_matrix.shape[0]
    degree_array = np.sum(adjacency_matrix, axis=1)        # degs
    degree_matrix = np.diag(degree_array)                  # D
    laplacian_matrix = degree_matrix - adjacency_matrix    # L = D - W

    # 避免除以零
    degree_array[degree_array == 0] = 1e-12
    inv_sqrt_degree = 1.0 / np.sqrt(degree_array)          # 1/(degs^0.5)
    inv_sqrt_degree_matrix = np.diag(inv_sqrt_degree)      # D_sqrt

    laplacian_sym = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix

    # 2) 求 (num_clusters+5) 个最小特征值/特征向量
    #    MATLAB sc3 里是 eigs(L_sym, k+5, eps)
    #    Python 用 eigsh(which='SM') 求最小特征值
    kplus = min(num_clusters + 5, num_points)
    laplacian_sym_sp = sp.csr_matrix(laplacian_sym)
    eigen_values_all, eigen_vectors_all = spla.eigsh(laplacian_sym_sp, k=kplus, which='SM')
    # eigen_values_all, eigen_vectors_all 分别是特征值、特征向量

    # 3) 与 MATLAB 中 find(diag(d)) 类似，这里先对特征值升序排序
    index_sorted = np.argsort(eigen_values_all)
    sorted_values = eigen_values_all[index_sorted]

    # 找非零特征值下标
    nonzero_indices = np.where(np.abs(sorted_values) > 1e-12)[0]
    if len(nonzero_indices) < num_clusters:
        # 若非零特征值不足 num_clusters 个，则直接取前 num_clusters 个
        chosen_indices = index_sorted[:num_clusters]
    else:
        # MATLAB 中 starting = idx(1), U=U(:,starting:starting+k-1)
        starting_idx = nonzero_indices[0]
        chosen_indices = index_sorted[starting_idx : starting_idx + num_clusters]

    # 提取对应特征向量
    eigen_vectors = eigen_vectors_all[:, chosen_indices]  # (n, num_clusters)

    # 4) 行归一化
    row_norms = np.linalg.norm(eigen_vectors, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-12
    normalized_vectors = eigen_vectors / row_norms

    # 5) kmeans
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=300, random_state=0)
    cluster_labels = kmeans_model.fit_predict(normalized_vectors)

    return cluster_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="被试数据根目录，应包含 data/probtrack_old/ 和 data/seeds_txt_all/ 等子目录")
    parser.add_argument("--method", default="sc", choices=["sc","kmeans","simlr"],
                        help="聚类方法：sc（谱聚类）、kmeans 或 simlr，默认=sc")
    parser.add_argument("--max_cl_num", type=int, default=12,
                        help="最大聚类数（默认12）")
    parser.add_argument("--start_seed", type=int, default=1,
                        help="起始 ROI 编号（默认1）")
    parser.add_argument("--end_seed", type=int, default=50,
                        help="结束 ROI 编号（默认50）")
    args = parser.parse_args()

    data_path  = args.data_path
    method     = args.method
    max_cl_num = args.max_cl_num
    start_seed = args.start_seed
    end_seed   = args.end_seed

    # 修改后的目录：
    # ROI 坐标文件：data/seeds_txt_all/seed_region_{i}.txt （保持不变）
    roi_coord_folder = os.path.join(data_path, "data", "seeds_txt_all")
    # 4D 文件：data/probtrack_old/ROI_{i}_merged_fdt_paths.nii.gz
    fourD_folder = os.path.join(data_path, "data", "probtrack_old")
    # 连接矩阵（con_cor）存放目录
    conn_folder = os.path.join(data_path, "data", "probtrack_old", "con_cor")
    # 聚类结果输出目录：data/probtrack_old/parcellation_{method}
    outdir = os.path.join(data_path, "data", "probtrack_old", f"parcellation_{method}")
    os.makedirs(outdir, exist_ok=True)

    # 对每个 ROI（种子）
    for i in range(start_seed, end_seed+1):
        # 1) 读取 ROI 坐标文件
        coord_file = os.path.join(roi_coord_folder, f"seed_region_{i}.txt")
        if not os.path.isfile(coord_file):
            print(f"[WARNING] {coord_file} 不存在，跳过 ROI {i}")
            continue
        coords = np.loadtxt(coord_file, dtype=int)
        n_voxels = coords.shape[0]
        if n_voxels == 0:
            print(f"[WARNING] {coord_file} 为空，跳过 ROI {i}")
            continue

        # 2) 读取连接矩阵文件 (con_matrix_seed_{i}.npy)，保存在 con_cor 文件夹中
        con_mat_path = os.path.join(conn_folder, f"con_matrix_seed_{i}.npy")
        if not os.path.isfile(con_mat_path):
            print(f"[WARNING] {con_mat_path} 不存在，跳过 ROI {i}")
            continue
        con_matrix = np.load(con_mat_path)  # 假设形状为 (n_voxels, M)

        # 3) 读取 4D 文件，用于获取空间信息（体素坐标与 affine）
        fourD_file = os.path.join(fourD_folder, f"ROI_{i}_merged_fdt_paths.nii.gz")
        if not os.path.isfile(fourD_file):
            print(f"[WARNING] {fourD_file} 不存在，跳过 ROI {i}")
            continue
        ref_nii = nib.load(fourD_file)
        vol_shape = ref_nii.shape[:3]  # (X, Y, Z)
        affine = ref_nii.affine

        # 4) 针对不同聚类数（从2到 max_cl_num）进行聚类
        for k in range(2, max_cl_num+1):
            out_nii_name = f"seed_{i}_{k}.nii.gz"
            out_nii_path = os.path.join(outdir, out_nii_name)
            if os.path.isfile(out_nii_path):
                print(f"[INFO] {out_nii_path} 已存在，跳过。")
                continue

            print(f"[INFO] ROI {i} 聚类：簇数 = {k}，方法 = {method}")
            # 根据方法选择聚类：谱聚类（sc）、K-means 或 simlr（占位）
            if method == "sc":
                # 构造相似矩阵：这里用 con_matrix @ con_matrix.T
                sim_mat = con_matrix @ con_matrix.T
                np.fill_diagonal(sim_mat, 0)
                labels = spectral_clustering(sim_mat, k)
            elif method == "kmeans":
                km = KMeans(n_clusters=k, n_init=10, random_state=0)
                labels = km.fit_predict(con_matrix)
            elif method == "simlr":
                labels = simlr.simlr_cluster(con_matrix, k)
            else:
                print(f"[ERROR] 未知方法 {method}，跳过 ROI {i}")
                continue

            # 5) 根据 ROI 坐标和聚类标签生成分割结果图（3D NIfTI）
            cluster_img = np.zeros(vol_shape, dtype=np.int16)
            # 将每个体素赋上标签（加1保证标签从1开始）
            for idx_vox in range(n_voxels):
                x, y, z = coords[idx_vox]
                cluster_img[x, y, z] = labels[idx_vox] + 1

            out_nii = nib.Nifti1Image(cluster_img, affine)
            out_nii.to_filename(out_nii_path)
            print(f"[INFO] 保存聚类结果：{out_nii_path}")

    print("[INFO] ROI 分割完成。")

if __name__ == "__main__":
    main()
