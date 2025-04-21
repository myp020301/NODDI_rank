#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import argparse
from scipy import sparse
    
def calc_matrix_for_seed(roi_coord_file, img_folder, threshold, output_folder, roi_name):
    """
    构造稀疏连接矩阵和相关矩阵，基于 3D 文件：
      - roi_coord_file: 每行 x y z 的 ROI 坐标
      - img_folder: 存放 fdt_paths_x_y_z.nii.gz 的目录
      - threshold: 阈值，小于该值的元素置零
      - output_folder: 保存结果的目录
      - seed_index: ROI 索引（用于文件命名）
    """
    # 1) 读取 ROI 坐标
    coords = np.loadtxt(roi_coord_file, dtype=int)
    n_voxels = coords.shape[0]
    if n_voxels == 0:
        raise ValueError(f"[ERROR] ROI 坐标文件 {roi_coord_file} 中没有坐标！")
    
    # 2) 根据第一个坐标确定图像尺寸
    x0, y0, z0 = coords[0]
    ref_file = os.path.join(img_folder, f"fdt_paths_{x0}_{y0}_{z0}.nii.gz")
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"[ERROR] 找不到参考文件: {ref_file}")
    ref_nii = nib.load(ref_file)
    img_shape = ref_nii.get_fdata().shape  # (X, Y, Z)
    n_vox = np.prod(img_shape)
    print(f"[INFO] 每个 3D 文件尺寸：{img_shape}, 展平后长度：{n_vox}")
    
    # 3) 构造稀疏连接矩阵
    rows, cols, vals = [], [], []
    for i, (x, y, z) in enumerate(coords):
        fpath = os.path.join(img_folder, f"fdt_paths_{x}_{y}_{z}.nii.gz")
        if not os.path.isfile(fpath):
            print(f"[WARNING] 文件不存在: {fpath}，对应行置零")
            continue
        data = nib.load(fpath).get_fdata().reshape(-1)
        nz = np.nonzero(data)[0]
        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(data[nz].tolist())
    con_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_voxels, n_vox))
    print(f"[INFO] 原始稀疏连接矩阵：shape={con_mat.shape}, nnz={con_mat.nnz}")
    
    # 4) 删除全零列
    col_sum = np.array(con_mat.sum(axis=0)).ravel()
    keep = np.where(col_sum != 0)[0]
    con_mat = con_mat[:, keep]
    print(f"[INFO] 删除全零列后：shape={con_mat.shape}, nnz={con_mat.nnz}")
    
    # 5) 阈值化
    mask = con_mat.data < threshold
    con_mat.data[mask] = 0
    con_mat.eliminate_zeros()
    print(f"[INFO] 阈值化后：shape={con_mat.shape}, nnz={con_mat.nnz}")
    
    # 6) 计算相关矩阵
    cor_mat = con_mat.dot(con_mat.T)
    print(f"[INFO] 稀疏相关矩阵：shape={cor_mat.shape}, nnz={cor_mat.nnz}")
    
    # 7) 保存结果
    os.makedirs(output_folder, exist_ok=True)
    con_out = os.path.join(output_folder, f"con_matrix_seed_{roi_name}.npz")
    cor_out = os.path.join(output_folder, f"cor_matrix_seed_{roi_name}.npz")
    sparse.save_npz(con_out, con_mat)
    sparse.save_npz(cor_out, cor_mat)
    print(f"[INFO] ROI {roi_name} 处理完成，结果保存在 {output_folder}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="根目录，应包含 data/seeds_txt_all/ 和 data/probtrack_old/")
    parser.add_argument("--roi_dir",default="/data2/mayupeng/BNU/ROI",help="标准空间 ROI 文件夹，文件名 *.nii 或 *.nii.gz")
    parser.add_argument("--roi_name",required=True,help="要处理的 ROI 名称（不含扩展名），如 MCP 或 FA_L")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="阈值 (default: 10)")
    
    args = parser.parse_args()

    data_path = args.data_path
    threshold = args.threshold
    roi_name = args.roi_name
    os.chdir(data_path)

    roi_coord_folder = os.path.join(data_path, "data", "seeds_txt_all")
    probtrack_folder = os.path.join(data_path, "data", "probtrack_old")
    img_folder = os.path.join(probtrack_folder, f"ROI_{roi_name}")
    output_folder = os.path.join(probtrack_folder, "con_cor")

    roi_coord_file = os.path.join(roi_coord_folder, f"seed_region_{roi_name}.txt")
    if not os.path.isfile(roi_coord_file):
        print(f"[WARNING] ROI 坐标文件不存在: {roi_coord_file}, 中止。")
        return
    if not os.path.isdir(img_folder):
        print(f"[WARNING] 3D 文件夹不存在: {img_folder}, 中止。")
        return

    calc_matrix_for_seed(
        roi_coord_file=roi_coord_file,
        img_folder=img_folder,
        threshold=threshold,
        output_folder=output_folder,
        roi_name=roi_name
    )

if __name__ == "__main__":
    main()
