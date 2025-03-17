#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import argparse

def calc_matrix_for_seed(
    roi_coord_file: str,
    fourD_file: str,
    threshold: float,
    output_folder: str,
    seed_index: int
):
    """
    针对单个 ROI 区域（seed_index）对应的 4D 文件，
    构建连接矩阵 con_matrix (nVox x T)，阈值化并移除全 0 列，
    然后计算相关矩阵 cor_matrix = con_matrix @ con_matrix.T。
    
    最终将 con_matrix_seed_{i}.npy, cor_matrix_seed_{i}.npy, 
    coords_seed_{i}.txt 输出到 output_folder 中。
    """
    # 1) 读取 ROI 坐标
    coords = np.loadtxt(roi_coord_file, dtype=int)  # shape: (nVox, 3)
    n_voxels = coords.shape[0]
    if n_voxels == 0:
        raise ValueError(f"[ERROR] 文件 {roi_coord_file} 中没有坐标！")
    
    # 2) 加载 4D NIfTI 数据
    print(f"[INFO] 加载 4D 文件（ROI {seed_index}）：{fourD_file}")
    nii4D = nib.load(fourD_file)
    vol4D = nii4D.get_fdata()  # shape: (X, Y, Z, T)
    if vol4D.ndim != 4:
        raise ValueError(f"[ERROR] 文件 {fourD_file} 不是4D数据！")
    X_dim, Y_dim, Z_dim, T = vol4D.shape
    print(f"[INFO] 4D 数据尺寸：({X_dim}, {Y_dim}, {Z_dim}, {T})，ROI 中体素数：{n_voxels}")
    
    # 3) 构建 con_matrix: 每个 ROI 内的体素提取对应 4D 时间序列
    con_matrix = np.zeros((n_voxels, T), dtype=np.float32)
    for i in range(n_voxels):
        x, y, z = coords[i]
        con_matrix[i, :] = vol4D[x, y, z, :]
    
    # 4) 阈值化处理
    con_matrix[con_matrix < threshold] = 0
    # 移除全 0 列（时间点）
    col_max = con_matrix.max(axis=0)
    col_min = con_matrix.min(axis=0)
    keep_cols = ~((col_max == 0) & (col_min == 0))
    con_matrix = con_matrix[:, keep_cols]
    print(f"[INFO] 阈值化后，con_matrix 尺寸：{con_matrix.shape}")
    
    # 5) 计算相关矩阵（或称连接矩阵）：cor_matrix = con_matrix @ con_matrix.T
    cor_matrix = con_matrix @ con_matrix.T  # shape: (n_voxels, n_voxels)
    
    # 6) 保存结果到 output_folder
    os.makedirs(output_folder, exist_ok=True)
    con_out = os.path.join(output_folder, f"con_matrix_seed_{seed_index}.npy")
    cor_out = os.path.join(output_folder, f"cor_matrix_seed_{seed_index}.npy")
    coords_out = os.path.join(output_folder, f"coords_seed_{seed_index}.txt")
    np.save(con_out, con_matrix)
    np.save(cor_out, cor_matrix)
    np.savetxt(coords_out, coords, fmt="%d")
    print(f"[INFO] ROI {seed_index} 处理完成，结果保存在：{output_folder}")

def main():
    """
    主程序：
      --data_path: 被试工作目录（包含 data/seeds_txt_all 和 data/probtrack_old 子目录）
      --threshold: 阈值，默认10
      --start_seed, --end_seed: ROI 范围（默认1到50）
    
    本脚本将基于 data/seeds_txt_all 中的坐标文件，
    以及 data/probtrack_old 中的 4D 文件（文件名格式为 ROI_{i}_merged_fdt_paths.nii.gz），
    生成连接矩阵和相关矩阵，并保存在 data/probtrack_old/con_cor/ 目录中。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="被试数据根目录，应包含 data/seeds_txt_all/ 和 data/probtrack_old/ 等子目录")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="阈值（默认10）")
    parser.add_argument("--start_seed", type=int, default=1,
                        help="起始 ROI 编号（默认1）")
    parser.add_argument("--end_seed", type=int, default=50,
                        help="结束 ROI 编号（默认50）")
    args = parser.parse_args()

    data_path = args.data_path
    threshold = args.threshold
    start_seed = args.start_seed
    end_seed = args.end_seed

    # 拼接目录路径
    roi_coord_folder = os.path.join(data_path, "data", "seeds_txt_all")
    # 修改4D文件路径：probtrack_old 文件夹下的 ROI_{i}_merged_fdt_paths.nii.gz
    probtrack_folder = os.path.join(data_path, "data", "probtrack_old")
    output_folder = os.path.join(probtrack_folder, "con_cor")

    for roi_idx in range(start_seed, end_seed + 1):
        roi_coord_file = os.path.join(roi_coord_folder, f"seed_region_{roi_idx}.txt")
        fourD_file = os.path.join(probtrack_folder, f"ROI_{roi_idx}_merged_fdt_paths.nii.gz")
        if not os.path.isfile(roi_coord_file):
            print(f"[WARNING] 未找到 ROI 坐标文件：{roi_coord_file}，跳过 ROI {roi_idx}.")
            continue
        if not os.path.isfile(fourD_file):
            print(f"[WARNING] 未找到 4D 文件：{fourD_file}，跳过 ROI {roi_idx}.")
            continue

        calc_matrix_for_seed(
            roi_coord_file=roi_coord_file,
            fourD_file=fourD_file,
            threshold=threshold,
            output_folder=output_folder,
            seed_index=roi_idx
        )

if __name__ == "__main__":
    main()
