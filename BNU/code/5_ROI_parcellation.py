#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from sklearn.cluster import SpectralClustering, KMeans
import argparse
import scipy.sparse as sp
import simlr  # 假定 simlr 模块已经实现

def process_roi(roi_name, roi_coord_folder, img_folder, conn_folder, outdir, method, max_cl_num):
    """
    对单个 ROI（由 roi_name 索引指定）进行处理：
      1) 从 roi_coord_folder 中读取 ROI 坐标文件（每行 "x y z"），
         顺序决定连接矩阵的行顺序。
      2) 从 conn_folder 中加载对应的稀疏连接矩阵（存储为 npz 格式），
         用于构造相似性矩阵。
      3) 从 img_folder 中，根据 ROI 坐标文件中第一行构造参考文件（fdt_paths_x_y_z.nii.gz），
         以获得空间信息（体积尺寸与 affine）。
      4) 对不同聚类数（从2到 max_cl_num），利用指定的聚类方法对相似性矩阵进行聚类，
         并根据 ROI 坐标生成分割结果（仅在 ROI 内赋值），保存为 3D NIfTI 文件。
    """
    # 1) 读取 ROI 坐标文件
    coord_file = os.path.join(roi_coord_folder, f"seed_region_{roi_name}.txt")
    if not os.path.isfile(coord_file):
        print(f"[WARNING] {coord_file} 不存在，跳过 ROI {roi_name}。")
        return
    coords = np.loadtxt(coord_file, dtype=int)
    n_voxels = coords.shape[0]
    if n_voxels == 0:
        print(f"[WARNING] {coord_file} 为空，跳过 ROI {roi_name}。")
        return

    # 2) 加载连接矩阵（稀疏格式，npz）
    con_mat_path = os.path.join(conn_folder, f"con_matrix_seed_{roi_name}.npz")
    if not os.path.isfile(con_mat_path):
        print(f"[WARNING] {con_mat_path} 不存在，跳过 ROI {roi_name}。")
        return
    con_matrix = sp.load_npz(con_mat_path)  # shape: (n_voxels, M)
    print(f"[INFO] 加载连接矩阵：形状 {con_matrix.shape}, 非零元素 {con_matrix.nnz}")

    # 3) 获得空间信息：从 ROI 坐标文件中的第一个坐标构造参考 3D 文件路径
    first_coord = coords[0]
    ref_file = os.path.join(img_folder, f"fdt_paths_{first_coord[0]}_{first_coord[1]}_{first_coord[2]}.nii.gz")
    if not os.path.isfile(ref_file):
        print(f"[WARNING] 参考文件 {ref_file} 不存在，跳过 ROI {roi_name}。")
        return
    ref_nii = nib.load(ref_file)
    vol_shape = ref_nii.shape  # (X, Y, Z)
    affine = ref_nii.affine

    # 4) 对不同聚类数进行聚类
    for k in range(2, max_cl_num + 1):
        out_nii_name = f"seed_{roi_name}_{k}.nii.gz"
        out_nii_path = os.path.join(outdir, out_nii_name)
        if os.path.isfile(out_nii_path):
            print(f"[INFO] {out_nii_path} 已存在，跳过。")
            continue

        print(f"[INFO] ROI {roi_name} 聚类：聚类数 = {k}, 方法 = {method}")
        if method == "sc":
            # 构造相似性矩阵：sim_mat = con_matrix * con_matrix.T
            sim_mat = con_matrix.dot(con_matrix.T)
            # 转为 dense 数组以供聚类使用
            sim_mat = sim_mat.toarray()
            np.fill_diagonal(sim_mat, 0)
            sc = SpectralClustering(
                n_clusters=k,
                n_init=300,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=0,
                eigen_solver='arpack'
            )
            labels = sc.fit_predict(sim_mat)
            labels = labels + 1
        elif method == "kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(con_matrix.toarray())
        elif method == "simlr":
            labels = simlr.simlr_cluster(con_matrix.toarray(), k)
        else:
            print(f"[ERROR] 未知聚类方法 {method}，跳过 ROI {roi_name}。")
            continue

        # 根据 ROI 坐标和聚类标签构造分割结果：创建一个与参考图像大小相同的 3D 数组，
        # 对 ROI 坐标处赋予聚类标签，其它位置保持 0。
        seg_img = np.zeros(vol_shape, dtype=np.int16)
        for idx in range(n_voxels):
            x, y, z = coords[idx]
            seg_img[x, y, z] = labels[idx]
        out_nii = nib.Nifti1Image(seg_img, affine)
        out_nii.to_filename(out_nii_path)
        print(f"[INFO] 聚类结果已保存：{out_nii_path}")

def main():
    """
    Main 程序：
      --data_path: 被试数据根目录（应包含 data/seeds_txt_all/ 和 data/probtrack_old/）
      --method: 聚类方法 (默认: sc)
      --max_cl_num: 最大聚类数 (默认: 12)
    
    该脚本读取 data/seeds_txt_all/ 中的 ROI 坐标文件，
    并从 data/probtrack_old/merged_fdt_paths/ 中加载对应的 3D 文件以获得空间信息；
    同时加载存储在 data/probtrack_old/con_cor/ 中的稀疏连接矩阵（npz 格式），
    然后对连接矩阵构造相似性矩阵并执行聚类，
    最后生成 3D 分割结果并保存到 data/probtrack_old/parcellation_{method}/ 目录下。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="被试数据根目录；应包含子目录 data/seeds_txt_all/ 和 data/probtrack_old/")
    parser.add_argument("--roi_dir",default="/data2/mayupeng/BNU/ROI",help="标准空间 ROI 文件夹，文件名 *.nii 或 *.nii.gz")
    parser.add_argument("--roi_name",required=True,help="要处理的 ROI 名称（不含扩展名），如 MCP 或 FA_L")
    parser.add_argument("--method", default="sc", choices=["sc", "kmeans", "simlr"],
                        help="聚类方法 (默认: sc)")
    parser.add_argument("--max_cl_num", type=int, default=12,
                        help="最大聚类数 (默认: 12)")
    args = parser.parse_args()
 
    data_path = args.data_path
    method = args.method
    max_cl_num = args.max_cl_num
    roi_name = args.roi_name
    os.chdir(data_path)

    roi_coord_folder = os.path.join(data_path, "data", "seeds_txt_all")
    # 这里的 img_folder 存放单个 3D 文件，文件名格式为 fdt_paths_x_y_z.nii.gz
    img_folder = os.path.join(args.data_path, "data", "probtrack_old", f"ROI_{roi_name}")
    # 连接矩阵（以及相关矩阵）存放在 con_cor 目录中，格式 npz
    conn_folder = os.path.join(data_path, "data", "probtrack_old", "con_cor")
    outdir = os.path.join(data_path, "data", "probtrack_old", f"parcellation_{method}")
    os.makedirs(outdir, exist_ok=True)

    process_roi(roi_name, roi_coord_folder, img_folder, conn_folder, outdir, method, max_cl_num)
    print("[INFO] ROI 分割处理完成.")

if __name__ == "__main__":
    main()
