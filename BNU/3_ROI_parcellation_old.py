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
    基于谱聚类实现，类似于 MATLAB 中的 sc3 算法。
    计算对称归一化拉普拉斯矩阵，再提取最小特征向量，最后利用 K-means 聚类。
    
    参数:
      num_clusters: 目标聚类数 k
      adjacency_matrix: (n, n) 相似度矩阵
    
    返回:
      cluster_labels: (n,) 每个点的聚类标签 (0...k-1)
    """
    num_points = adjacency_matrix.shape[0]
    degree = np.sum(adjacency_matrix, axis=1)
    D = np.diag(degree)
    L = D - adjacency_matrix  # 拉普拉斯矩阵

    # 避免除以零
    degree[degree == 0] = 1e-12
    inv_sqrt = 1.0 / np.sqrt(degree)
    D_inv_sqrt = np.diag(inv_sqrt)

    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # 求最小 (num_clusters+5) 个特征值/向量
    kplus = min(num_clusters + 5, num_points)
    L_sym_sp = sp.csr_matrix(L_sym)
    eigvals, eigvecs = spla.eigsh(L_sym_sp, k=kplus, which='SM')
    sorted_idx = np.argsort(eigvals)
    sorted_vals = eigvals[sorted_idx]
    nonzero = np.where(np.abs(sorted_vals) > 1e-12)[0]
    if len(nonzero) < num_clusters:
        chosen = sorted_idx[:num_clusters]
    else:
        start = nonzero[0]
        chosen = sorted_idx[start:start+num_clusters]
    U = eigvecs[:, chosen]

    # 行归一化
    row_norm = np.linalg.norm(U, axis=1, keepdims=True)
    row_norm[row_norm==0] = 1e-12
    U_norm = U / row_norm

    kmeans_model = KMeans(n_clusters=num_clusters, n_init=300, random_state=0)
    labels = kmeans_model.fit_predict(U_norm)
    return labels

def simlr_cluster(matrix, k):
    """
    占位函数，返回随机标签
    """
    n = matrix.shape[0]
    return np.random.randint(0, k, size=n)

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
