#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def spectral_clustering(num_clusters, adjacency_matrix):
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


def simlr_cluster(matrix, k):
    """
    占位函数: simlr_cluster
    """
    print("[WARNING] 'simlr' method not implemented in Python. Return random labels.")
    n = matrix.shape[0]
    labels = np.random.randint(0, k, size=n)
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Subject's data directory containing con_cor, seeds_all, seeds_list_all")
    parser.add_argument("--method", default="sc", choices=["sc","kmeans","simlr"],
                        help="Clustering method: sc (spectral), kmeans, or simlr. Default=sc")
    parser.add_argument("--max_cl_num", type=int, default=12,
                        help="Maximum cluster number (default=5)")
    parser.add_argument("--start_seed", type=int, default=1,
                        help="Start seed index (default=1)")
    parser.add_argument("--end_seed", type=int, default=50,
                        help="End seed index (default=50)")
    args = parser.parse_args()

    data_path   = args.data_path
    method      = args.method
    max_cl_num  = args.max_cl_num
    start_seed  = args.start_seed
    end_seed    = args.end_seed

    # 在 data_path 下新建一个输出目录 e.g. parcellation_sc / parcellation_kmeans / parcellation_simlr
    outdir = os.path.join(data_path, f"data/parcellation_{method}")
    os.makedirs(outdir, exist_ok=True)


    # 对每个 i= start_seed..end_seed
    for i in range(start_seed, end_seed+1):
        # 1) 读取 coords: data_path/data/seeds_list_all/seed_region_{i}.txt
        coord_file = os.path.join(data_path, "data/seeds_list_all", f"seed_region_{i}.txt")
        if not os.path.isfile(coord_file):
            print(f"[WARNING] {coord_file} not found, skip seed {i}")
            continue
        coords = np.loadtxt(coord_file, dtype=int)
        nVox = coords.shape[0]
        if nVox == 0:
            print(f"[WARNING] seed_region_{i}.txt is empty, skip.")
            continue

        # 2) 读取 con_matrix_seed_{i}.npy: data_path/con_cor/con_matrix_seed_{i}.npy
        con_mat_path = os.path.join(data_path, "data/con_cor", f"con_matrix_seed_{i}.npy")
        if not os.path.isfile(con_mat_path):
            print(f"[WARNING] {con_mat_path} not found, skip seed {i}")
            continue
        matrix = np.load(con_mat_path)  # shape (nVox x M)
        # 过滤全零行? 这里不做, assume it was done or no need

        # 3) 读取 seeds_{i}_to_seed_region_all.nii.gz: data_path/seeds_all/seeds_{i}_to_seed_region_all.nii.gz
        #    用它来获取体素空间 & affine
        fourD_nii_path = os.path.join(data_path, "data/seeds_all", f"seed_{i}_to_seeds_all.nii.gz")
        if not os.path.isfile(fourD_nii_path):
            print(f"[WARNING] {fourD_nii_path} not found, skip seed {i}")
            continue
        ref_nii = nib.load(fourD_nii_path)
        shape_3D = ref_nii.shape[:3]  # (X,Y,Z)
        affine   = ref_nii.affine

        # 4) 循环 k=2..max_cl_num  clusters
        for k in range(2, max_cl_num+1):
            cluster_num = k
            out_nii_name = f"seed_{i}_{cluster_num}.nii.gz"
            out_nii_path = os.path.join(outdir, out_nii_name)
            if os.path.isfile(out_nii_path):
                print(f"[INFO] {out_nii_path} already exists, skip.")
                continue

            print(f"[INFO] Clustering seed={i}, cluster_num={cluster_num}, method={method}")
            # 执行聚类
            if method == "sc":
                # matrix1 = matrix @ matrix.T => shape (nVox x nVox)
                matrix1 = matrix @ matrix.T
                # diag=0
                np.fill_diagonal(matrix1, 0)
                labels = spectral_clustering(cluster_num, matrix1)
            elif method == "kmeans":
                km = KMeans(n_clusters=cluster_num, n_init=10, random_state=0)
                labels = km.fit_predict(matrix)
            elif method == "simlr":
                labels = simlr_cluster(matrix, cluster_num)
            else:
                print(f"[ERROR] Unknown method={method}, skip.")
                continue

            # 5) 写入新体素图
            #    先创建一个空图, shape=ref_nii.shape[:3]
            cluster_img = np.zeros(shape_3D, dtype=np.int16)

            # coords 是 (nVox, 3), 0-based
            # labels[i] 在 0..(cluster_num-1)
            # 这里 +1 避免出现 0 label
            for idx_vox in range(nVox):
                x, y, z = coords[idx_vox]
                cluster_img[x, y, z] = labels[idx_vox] + 1

            out_nii = nib.Nifti1Image(cluster_img, affine)
            out_nii.to_filename(out_nii_path)
            print(f"[INFO] Saved {out_nii_path}")

    print("[INFO] ROI_parcellation done for all seeds.")


if __name__ == "__main__":
    main()
