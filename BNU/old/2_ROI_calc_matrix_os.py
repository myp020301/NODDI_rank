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
    针对单个种子 seed_index 的 4D 文件 seed_{i}_to_seeds_all.nii.gz，
    构建 con_matrix (nVox x T)，阈值化并移除全 0 列，然后计算 cor_matrix。

    最终将 con_matrix_seed_{i}.npy, cor_matrix_seed_{i}.npy, coords_seed_{i}.txt 输出到指定文件夹。
    """

    # 1) 读取 ROI 坐标
    coords = np.loadtxt(roi_coord_file, dtype=int)  # shape: (nVox, 3)
    nVox = coords.shape[0]
    if nVox == 0:
        raise ValueError(f"[ERROR] roi_coord_file={roi_coord_file} 中无坐标！")

    # 2) 加载 4D NIfTI
    print(f"[INFO] Loading 4D file for seed={seed_index}: {fourD_file}")
    nii4D = nib.load(fourD_file)
    vol4D = nii4D.get_fdata()  # float64, shape: (X, Y, Z, T)
    shape4D = vol4D.shape
    if len(shape4D) != 4:
        raise ValueError(f"[ERROR] {fourD_file} 不是 4D 数据！ shape={shape4D}")
    X, Y, Z, T = shape4D
    print(f"[INFO] 4D shape: (X={X}, Y={Y}, Z={Z}, T={T}), nVox={nVox}")

    # 3) 构建 con_matrix (nVox x T)
    con_matrix = np.zeros((nVox, T), dtype=np.float32)
    for i in range(nVox):
        x, y, z = coords[i]
        con_matrix[i, :] = vol4D[x, y, z, :]

    # 4) 阈值化
    con_matrix[con_matrix < threshold] = 0

    # 移除全 0 列
    col_max = con_matrix.max(axis=0)
    col_min = con_matrix.min(axis=0)
    keep_cols = ~((col_max == 0) & (col_min == 0))
    con_matrix = con_matrix[:, keep_cols]
    print(f"[INFO] After removing zero-columns, con_matrix shape: {con_matrix.shape}")

    # 5) 计算 cor_matrix = con_matrix @ con_matrix.T
    cor_matrix = con_matrix @ con_matrix.T  # shape: (nVox, nVox)
    
    # 6) 保存
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, f"con_matrix_seed_{seed_index}.npy"), con_matrix)
    np.save(os.path.join(output_folder, f"cor_matrix_seed_{seed_index}.npy"), cor_matrix)
    np.savetxt(os.path.join(output_folder, f"coords_seed_{seed_index}.txt"), coords, fmt="%d")

    print(f"[INFO] Done seed={seed_index}, saved to {output_folder}")


def main():
    """
    主程序：
      - data_path: 被试工作目录 (其中包含 seeds_txt_all, seeds_result_all, con_cor 等子目录)
      - threshold: 阈值, 默认10
      - start_seed, end_seed: 种子索引范围
      - 本脚本会在 data_path 下拼出 seeds_txt_all, seeds_result_all, con_cor 目录。
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Base path for the subject's data, containing seeds_txt_all/ seeds_result_all/ etc.")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Threshold value (default=10)")
    parser.add_argument("--start_seed", type=int, default=1,
                        help="Start seed index (default=1)")
    parser.add_argument("--end_seed", type=int, default=50,
                        help="End seed index (default=50)")
    args = parser.parse_args()
    
    data_path   = args.data_path
    threshold   = args.threshold
    start_seed  = args.start_seed
    end_seed    = args.end_seed
    
    # 根据 data_path 拼接:
    roi_coord_folder = os.path.join(data_path, "data/seeds_txt_all")
    input_folder     = os.path.join(data_path, "data/seeds_result_all")
    output_folder    = os.path.join(data_path, "data/con_cor_v2")
    
    # 循环处理每个种子 i
    for i in range(start_seed, end_seed + 1):
        roi_coord_file = os.path.join(roi_coord_folder, f"seed_region_{i}.txt")
        fourD_file     = os.path.join(input_folder,     f"seed_{i}_to_targets_all.nii.gz")

        if not os.path.isfile(roi_coord_file):
            print(f"[WARNING] ROI coord file not found: {roi_coord_file}, skipping.")
            continue
        if not os.path.isfile(fourD_file):
            print(f"[WARNING] 4D file not found: {fourD_file}, skipping.")
            continue

        calc_matrix_for_seed(
            roi_coord_file=roi_coord_file,
            fourD_file=fourD_file,
            threshold=threshold,
            output_folder=output_folder,
            seed_index=i
        )


if __name__ == "__main__":
    main()
