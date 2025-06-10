#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import argparse
from scipy import sparse
from nibabel import processing as nbproc  

VOX_SIZE = (5, 5, 5)     

def calc_matrix_for_seed(
    roi_coord_file: str,
    img_folder: str,
    threshold: float,
    output_folder: str,
    roi_name: str
): 
    """
    构造连接矩阵和相关矩阵（稠密形式），严格复刻 MATLAB 实现：
      1) 从多个 3D 文件读取数据，构造稀疏 con_mat
      2) 删除全零列
      3) 阈值化
      4) 将 con_mat 转为稠密矩阵 con_dense
      5) 计算 cor_mat = con_dense @ con_dense.T
      6) 保存 con_dense 和 cor_mat（均为稠密）
    """
    coords = np.loadtxt(roi_coord_file, dtype=int)
    n_vox = coords.shape[0]
    if n_vox == 0:
        raise ValueError(f"[ERROR] ROI 坐标文件 {roi_coord_file} 中没有坐标！")
    
    # 用首个 seed 文件确定整体体素数
    x0, y0, z0 = coords[0]
    ref_file = os.path.join(img_folder, f"{roi_name}_{x0}_{y0}_{z0}.nii.gz")
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"[ERROR] 找不到参考文件: {ref_file}")
    ref_img = nib.load(ref_file)
    ref_img5 = nbproc.resample_to_output(ref_img, VOX_SIZE, order=1)  ### <- NEW
    img_shape = ref_img5.shape                                        ### <- NEW
    n_total = np.prod(img_shape)
    print(f"[INFO] 下采样后体积尺寸：{img_shape}  → 展平长度 {n_total}")

    # 构造稀疏连接矩阵
    rows, cols, vals = [], [], []
    for i, (x, y, z) in enumerate(coords):
        fpath = os.path.join(img_folder, f"{roi_name}_{x}_{y}_{z}.nii.gz")
        print(fpath)
        if not os.path.isfile(fpath):
            print(f"[WARNING] 文件不存在: {fpath}，对应行置零")
            continue
        img = nib.load(fpath)
        img5 = nbproc.resample_to_output(img, VOX_SIZE, order=1)      ### <- NEW
        data = img5.get_fdata().reshape(-1)                           ### <- NEW
        
        nz = np.nonzero(data)[0]
        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(data[nz].tolist())
    con_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_vox, n_total))

    # 删除全零列
    col_sum = np.array(con_mat.sum(axis=0)).ravel()
    keep = np.where(col_sum != 0)[0]
    con_mat = con_mat[:, keep]
    print(f"[INFO] 删除全零列后：shape={con_mat.shape}, nnz={con_mat.nnz}")
    # 转为稠密矩阵
    con_dense = con_mat.toarray()
    
    # 保存
    os.makedirs(output_folder, exist_ok=True)
    con_out = os.path.join(output_folder, f"con_matrix_seed_{roi_name}.npy")
    np.save(con_out, con_dense)
    print(f"[INFO] 未阈值化稠密连接矩阵已保存至 {con_out}")
    
    # 阈值化
    mask = con_dense < threshold
    con_dense[mask] = 0
    print(f"[INFO] 阈值化后：shape={con_dense.shape}")

    # 计算稠密相关矩阵
    cor_mat = con_dense @ con_dense.T
    print(f"[INFO] 稠密相关矩阵 cor_mat: shape={cor_mat.shape}")

    cor_out = os.path.join(output_folder, f"cor_matrix_seed_{roi_name}.npy")

    np.save(cor_out, cor_mat)
    print(f"[INFO] ROI {roi_name} 处理完成，结果保存在 {output_folder}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--roi_dir",default="/data2/mayupeng/BNU/ROI")
    parser.add_argument("--roi_name",required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    
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

