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
    针对单个种子 seed_index 的 4D 文件 seeds_{i}_to_seed_region_all.nii.gz，
    构建 con_matrix (nVox x T)，阈值化并移除全 0 列，然后计算 cor_matrix。

    最终将 con_matrix_seed_{i}.npy, cor_matrix_seed_{i}.npy, coords_seed_{i}.txt 输出到指定文件夹。

    参数:
    ----------
    roi_coord_file : str
        某个 ROI 坐标文件 (x y z)，形状 (nVox, 3)，例如 seed_region_{i}.txt
    fourD_file : str
        4D NIfTI 文件的路径，如 seeds_{i}_to_seed_region_all.nii.gz
    threshold : float
        小于该阈值的元素置 0
    output_folder : str
        保存输出矩阵的文件夹
    seed_index : int
        种子序号 (仅用于输出文件命名等)
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

    # 对每个 voxel i, 从 4D 文件中取 vol4D[x, y, z, :]
    for i in range(nVox):
        x, y, z = coords[i]
        # 若需安全检查，可判断 x,y,z 是否在 [0, X), [0, Y), [0, Z) 范围内
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
    - roi_coord_folder: 存放 50 个 txt 文件, 每个对应 seed_region_{i}.txt
    - input_folder: 存放 50 个 4D 文件 seeds_{i}_to_seed_region_all.nii.gz
    - output_folder: 用于保存计算得到的 con_matrix, cor_matrix
    - threshold: 阈值, 默认 10
    - start_seed, end_seed: 种子索引范围
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_coord_folder", default="data/seeds_list_all",
                        help="Folder with seed_region_{i}.txt files (default: data/seeds_list_all)")
    parser.add_argument("--input_folder", default="data/seeds_all",
                        help="Folder with seeds_{i}_to_seed_region_all.nii.gz (default: data/seeds_all)")
    parser.add_argument("--output_folder", default="data/con_cor",
                        help="Folder to save the results (default: data/con_cor)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Threshold value (default=10)")
    parser.add_argument("--start_seed", type=int, default=1, help="Start seed index (default=1)")
    parser.add_argument("--end_seed", type=int, default=50, help="End seed index (default=50)")
    args = parser.parse_args()
    
    roi_coord_folder = args.roi_coord_folder
    input_folder = args.input_folder
    output_folder = args.output_folder
    threshold = args.threshold
    start_seed = args.start_seed
    end_seed = args.end_seed

    # 循环处理每个种子 i
    for i in range(start_seed, end_seed + 1):
        # (1) 对应的坐标文件
        roi_coord_file = os.path.join(roi_coord_folder, f"seed_region_{i}.txt")
        # (2) 对应的 4D 文件
        fourD_file = os.path.join(input_folder, f"seeds_{i}_to_seed_region_all.nii.gz")

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
