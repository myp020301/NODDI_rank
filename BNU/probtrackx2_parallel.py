#!/usr/bin/env python3
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

# =====================
# 配置区
# =====================
SEED_REGION = "/data/mayupeng/BNU/dwi/data/seed_region_1.nii.gz"  # ROI 区域(整体)
MASK_FILE   = "/data/mayupeng/BNU/dwi/data/nodif_brain_mask.nii.gz"  # 脑掩膜
SAMPLES     = "merged"                                           # bedpostx 输出的samples
NSAMPLES    = 5000                                               # probtrackx2 样本数
COORDS_FILE = "/data/mayupeng/BNU/dwi/data/coords.txt"           # 存储ROI内每个体素(x,y,z)坐标
OUTPUT_DIR  = "/data/mayupeng/BNU/dwi/data/probtrack_python"      # 输出目录
N_JOBS      = 8                                                  # 并行进程数(可根据CPU核数调整)

# 如果你想避免内部多线程，可将 OMP_NUM_THREADS 设置为1
os.environ["OMP_NUM_THREADS"] = "1"

# =====================
# 函数：对单个体素执行 probtrackx2
# =====================
def run_probtrackx2(idx, x, y, z):
    """
    idx: 该体素的序号(1开始)
    x,y,z: 体素坐标
    """
    # 1) 为该体素生成单体素seed
    single_voxel_seed = f"{OUTPUT_DIR}/voxel_seed_{idx}.nii.gz"
    cmd_fslmaths = [
        "fslmaths",
        SEED_REGION,
        "-mul", "0",  # 将原图置为0
        "-add", "1",  # 加1
        "-roi", str(x), "1", str(y), "1", str(z), "1", "0", "1",
        single_voxel_seed
    ]
    subprocess.run(cmd_fslmaths, check=True)

    # 2) 创建输出目录
    single_voxel_out = f"{OUTPUT_DIR}/voxel_{idx}"
    os.makedirs(single_voxel_out, exist_ok=True)

    # 3) 执行 probtrackx2
    cmd_probtrack = [
        "probtrackx2",
        "--seed", single_voxel_seed,
        "--mask", MASK_FILE,
        "--samples", SAMPLES,
        "--nsamples", str(NSAMPLES),
        "--dir", single_voxel_out,
        "--forcedir",
        "--opd"  # 其他参数可自行添加
    ]
    subprocess.run(cmd_probtrack, check=True)

    # 打印日志
    print(f"[INFO] voxel #{idx} at ({x}, {y}, {z}) 完成.")


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取坐标文件 coords.txt
    # 假设 coords.txt 每行是 "x y z"
    with open(COORDS_FILE, "r") as f:
        coords = [line.strip().split() for line in f]
    # coords 形如 [["10","20","30"], ["10","21","30"], ...]

    # 把字符串坐标转为整数
    coords = [(int(x), int(y), int(z)) for (x, y, z) in coords]

    # 使用多进程并行
    print(f"[INFO] 并行处理开始，总体素数={len(coords)}，进程数={N_JOBS}")
    start_time = subprocess.getoutput("date +%s")  # 记录开始时间(秒)

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = []
        for i, (x, y, z) in enumerate(coords, start=1):
            futures.append(executor.submit(run_probtrackx2, i, x, y, z))

        # 等待所有任务完成
        for fut in futures:
            # 如果出错，会在这里抛出异常
            fut.result()

    end_time = subprocess.getoutput("date +%s")
    elapsed = int(end_time) - int(start_time)
    print(f"[INFO] 全部处理完成，耗时 {elapsed} 秒")


if __name__ == "__main__":
    main()
