#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
import argparse
import simlr  # 假定 simlr 模块已经实现

def sc3(k, W):
    """
    Python 版 sc3：将任何 np.matrix 转为 ndarray，并用 numpy.linalg.eigh 做全谱分解。
    
    参数:
      k: 聚类簇数
      W: 相似度矩阵，支持 np.ndarray 或 np.matrix
    
    返回:
      labels: shape = (n,), 值为 0..k-1
    """
    # 0) 如果是 np.matrix，就转成 np.ndarray
    W = np.asarray(W, dtype=np.float64)
    
    # 1) 打印输入形状
    n, m = W.shape
    
    # 2) 计算度矩阵和拉普拉斯
    degs = W.sum(axis=1)
    D = np.diag(degs)
    L = D - W
    
    # 3) 处理零度，避免除零
    eps = np.finfo(float).eps
    degs[degs == 0] = eps
    
    # 4) 构造对称归一化拉普拉斯 L_sym
    inv_sqrt = 1.0 / np.sqrt(degs)
    D_sqrt = np.diag(inv_sqrt)
    L_sym = D_sqrt @ L @ D_sqrt
    L_sym = np.asarray(L_sym)
    
    # 5) 全谱分解（eigh 只接受 ndarray）
    vals, vecs = np.linalg.eigh(L_sym)
    # 6) 选取第一个大于阈值 tol_eig 的特征值开始的 k 列
    tol_eig = 1e-8
    idx = np.where(vals > tol_eig)[0]
    start = int(idx[0]) if idx.size > 0 else 0
    print(vals[start:start+k])
    U = vecs[:, start:start+k]
    
    # 7) 行归一化
    row_norms = np.linalg.norm(U, axis=1)
    U_norm = U / row_norms[:, np.newaxis]
    
    # 8) 最后做 KMeans
    labels = KMeans(n_clusters=k, n_init=1000, random_state=0).fit_predict(U_norm)
    
    return labels


def process_roi(roi_name, roi_coord_folder, img_folder, con_folder, outdir, method, max_cl_num):
    """
    对单个 ROI（由 roi_name 索引指定）进行处理：
      1) 从 roi_coord_folder 中读取 ROI 坐标文件（每行 "x y z"），
         顺序决定聚类时标签的先后。
      2) 从 con_folder 中加载对应的【稠密】相关矩阵 con_matrix_seed_{roi}.npy，
         直接用它作为相似性/特征矩阵。
      3) 从 img_folder 中，根据 ROI 坐标文件中第一行构造参考文件（fdt_paths_x_y_z.nii.gz），
         以获得空间信息（体积尺寸与 affine）。
      4) 对不同聚类数（从2到 max_cl_num），在 SC/KMeans/SimLR 下分别用 con_matrix
         执行聚类，并根据 ROI 坐标生成分割结果（仅在 ROI 内赋值），保存为 3D NIfTI 文件。
    """
    # 1) 读取 ROI 坐标
    coord_file = os.path.join(roi_coord_folder, f"seed_region_{roi_name}.txt")
    if not os.path.isfile(coord_file):
        print(f"[WARNING] 找不到 ROI 坐标文件 {coord_file}，跳过 ROI {roi_name}。")
        return
    coords = np.loadtxt(coord_file, dtype=int)
    n_voxels = coords.shape[0]
    if n_voxels == 0:
        print(f"[WARNING] ROI 坐标文件 {coord_file} 为空，跳过 ROI {roi_name}。")
        return

    # 2) 加载稠密相关矩阵
    con_path = os.path.join(con_folder, f"con_matrix_seed_{roi_name}.npy")
    if not os.path.isfile(con_path):
        print(f"[WARNING] 找不到相关矩阵文件 {con_path}，跳过 ROI {roi_name}。")
        return
    con_mat = np.load(con_path)  # shape: (n_voxels, n_voxels)
    print(f"[INFO] 已加载相关矩阵：shape={con_mat.shape}")
    
    # ---- 删除全零行：剔除那些在所有体素间无任何连接的行/列 ----
    valid_mask = np.any(con_mat != 0, axis=1)
    num_removed = np.count_nonzero(~valid_mask)
    if num_removed > 0:
        print(f"[INFO] 删除全零行：共剔除 {num_removed} / {len(valid_mask)} 个体素")
        con_mat = con_mat[valid_mask][:, valid_mask]
        coords  = coords[valid_mask]
        print(f"[INFO] 剔除后相关矩阵 shape={con_mat.shape}, 坐标数={coords.shape[0]}")
    # ---- 删除全零行结束 ----
    
    # 3) 获得空间信息：用第一个 ROI 坐标对应的 fdt_paths 文件
    x0, y0, z0 = coords[0]
    ref_file = os.path.join(img_folder, f"{roi_name}_{x0}_{y0}_{z0}.nii.gz")
    if not os.path.isfile(ref_file):
        print(f"[WARNING] 参考文件 {ref_file} 不存在，跳过 ROI {roi_name}。")
        return
    ref_nii = nib.load(ref_file)
    vol_shape = ref_nii.shape  # (X, Y, Z)
    affine = ref_nii.affine

    # 4) 循环不同聚类数
    for k in range(2, max_cl_num + 1):
        out_nii = os.path.join(outdir, f"seed_{roi_name}_{k}.nii.gz")
        if os.path.isfile(out_nii):
            print(f"[INFO] {out_nii} 已存在，跳过。")
            continue

        print(f"[INFO] ROI {roi_name} 聚类：k={k}, method={method}")
        if method == "sc":
            sim_mat = con_mat @ con_mat.T
            np.fill_diagonal(sim_mat, 0)
            # 2) 调用我们刚写的 sc3
            labels = sc3(k,sim_mat)
            labels = labels + 1
        elif method == "kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(con_mat) + 1
        elif method == "simlr":
            labels = simlr.simlr_cluster(con_mat, k) + 1
        else:
            print(f"[ERROR] 未知聚类方法 {method}，跳过 ROI {roi_name}。")
            continue

        # 5) 生成 3D 分割结果
        seg = np.zeros(vol_shape, dtype=np.int16)
        for idx in range(n_voxels):
            x, y, z = coords[idx]
            seg[x, y, z] = labels[idx]
        nib.Nifti1Image(seg, affine).to_filename(out_nii)
        print(f"[INFO] 保存聚类结果: {out_nii}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--roi_dir",default="/data2/mayupeng/BNU/ROI")
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--method", default="sc", choices=["sc","kmeans","simlr"])
    parser.add_argument("--max_cl_num", type=int, default=6)
    args = parser.parse_args()

    dp = args.data_path
    rn = args.roi_name
    roi_coord_folder = os.path.join(dp, "data", "seeds_txt_all")
    img_folder       = os.path.join(dp, "data", "probtrack_old", f"ROI_{rn}")
    con_folder       = os.path.join(dp, "data", "probtrack_old", "con_cor")
    outdir           = os.path.join(dp, "data", "probtrack_old", f"parcellation_{args.method}")
    os.makedirs(outdir, exist_ok=True)

    process_roi(rn, roi_coord_folder, img_folder, con_folder, outdir, args.method, args.max_cl_num)
    print("[INFO] ROI 分割完成。")

if __name__ == "__main__":
    main()
