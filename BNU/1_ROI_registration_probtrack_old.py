#!/usr/bin/env python3
import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# =====================
# 配置区
# =====================
SEED_REGION = "../data/JHU50_native_nn.nii.gz"       # ROI 区域(整体)
MASK_FILE   = "../data/nodif_brain_mask.nii.gz"      # 脑掩膜
SAMPLES     = "merged"                               # bedpostx 输出的samples
NSAMPLES    = 5000                                   # probtrackx2_gpu 样本数
OUTPUT_DIR  = "../data/probtrack_old"                # 输出目录
N_JOBS      = 30                                     # 并行进程数(可根据CPU核数调整)

# 如果你想避免内部多线程，可将 OMP_NUM_THREADS 设置为1
os.environ["OMP_NUM_THREADS"] = "1"

def run_probtrackx2(roi_idx, voxel_idx, x, y, z):
    """
    roi_idx: 当前ROI区域编号 (1-50)
    voxel_idx: 该ROI区域的某个体素的索引
    x, y, z: 体素坐标
    """
    roi_folder = f"{OUTPUT_DIR}/ROI_{roi_idx}"
    os.makedirs(roi_folder, exist_ok=True)

    # 1) 为该体素生成单体素seed
    single_voxel_seed = f"{roi_folder}/seed_voxel_{voxel_idx}.nii.gz"
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
    single_voxel_out = f"{roi_folder}/voxel_{voxel_idx}"
    os.makedirs(single_voxel_out, exist_ok=True)

    # 3) 执行 probtrackx2
    cmd_probtrack = [
        "probtrackx2",
        f"--seed={single_voxel_seed}",
        f"--mask={MASK_FILE}",
        f"--samples={SAMPLES}",
        f"--nsamples={NSAMPLES}",
        f"--dir={single_voxel_out}",
        "--forcedir",
        "--opd"
    ]
    subprocess.run(cmd_probtrack, check=True)

    print(f"[INFO] ROI {roi_idx} - voxel #{voxel_idx} at ({x}, {y}, {z}) 处理完成.")

def save_nonzero_coordinates(nifti_path, output_txt):
    """
    读取 nifti_path 中的非零体素坐标(0-based)，保存到 output_txt 文件。
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()  # float64
    nonzero_coords = np.argwhere(data != 0)  # shape (N, 3)

    # 将坐标写入txt
    with open(output_txt, 'w') as f:
        for x, y, z in nonzero_coords:
            # 如果需要1-based，可改为 x+1, y+1, z+1
            f.write(f"{int(x)} {int(y)} {int(z)}\n")

    print(f"[INFO] Saved nonzero coordinates of {nifti_path} to {output_txt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path where data/ subfolder already prepared (with data.nii.gz, nodif_brain_mask.nii.gz, bvals, bvecs)")
    parser.add_argument("--sub_id", required=True)
    parser.add_argument("--FMRIB_FA_TEMPLATE", required=True)
    parser.add_argument("--JHU50_SEED", required=True)
    args = parser.parse_args()

    data_path = args.data_path
    sub_id    = args.sub_id
    FMRIB_FA_TEMPLATE = args.FMRIB_FA_TEMPLATE
    JHU50_SEED = args.JHU50_SEED

    print(f"[INFO] ROI_registration_probtrack for subject={sub_id}, data_path={data_path}")
    os.chdir(data_path)

    # bedpostx
    if os.path.isdir("data.bedpostX"):
        print(f"[WARNING] data.bedpostX already exists, skipping bedpostx")
    else:
        subprocess.run("bedpostx_gpu data -n 3", shell=True, check=True)
    
    # dtifit
    subprocess.run("dtifit -k data/data.nii.gz -o data/dtifit -m data/nodif_brain_mask.nii.gz -r data/bvecs -b data/bvals", shell=True, check=True)

    # FLIRT
    subprocess.run(f"flirt -in data/dtifit_FA.nii.gz -ref {FMRIB_FA_TEMPLATE} -out data/fa_standard_space.nii.gz -omat data/fa2standard.mat", shell=True, check=True)

    # FNIRT
    subprocess.run(f"fnirt --in=data/dtifit_FA.nii.gz --aff=data/fa2standard.mat --ref={FMRIB_FA_TEMPLATE} --iout=data/fa_standard_space_nonlin.nii.gz --cout=data/fa2standard_warp.nii.gz", shell=True, check=True)

    # Inverse warp
    subprocess.run("invwarp -w data/fa2standard_warp.nii.gz -r data/dtifit_FA.nii.gz -o data/JHU50_inverse_warp.nii.gz", shell=True, check=True)
    subprocess.run(f"applywarp -i {JHU50_SEED} -r data/dtifit_FA.nii.gz -w data/JHU50_inverse_warp.nii.gz -o data/JHU50_native_nn.nii.gz --interp=nn", shell=True, check=True)

    os.makedirs("data/seeds_txt_all", exist_ok=True)
    os.makedirs("data/seeds_region_all", exist_ok=True)

    # 创建每个种子区域的文件
    for i in range(1, 51):
        seed_out = f"data/seeds_region_all/seed_region_{i}.nii.gz"
        subprocess.run(f"fslmaths data/JHU50_native_nn.nii.gz -thr {i} -uthr {i} -bin {seed_out}", shell=True, check=True)

        # 保存非零坐标到 seed_region_{i}.txt
        seed_txt = f"data/seeds_txt_all/seed_region_{i}.txt"
        save_nonzero_coordinates(seed_out, seed_txt)

    with open("data/seed_list.txt", "w") as f:
        for i in range(1, 51):
            f.write(f"../data/seeds_region_all/seed_region_{i}.nii.gz\n")
    
    os.chdir("data.bedpostX")
    # 并行处理每个 ROI 区域的每个 voxel
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = []
        for roi_idx in range(1, 51):
            seed_txt = f"../data/seeds_txt_all/seed_region_{roi_idx}.txt"
            # 读取每个 ROI 的体素坐标
            with open(seed_txt, 'r') as f:
                coords = [tuple(map(int, line.split())) for line in f]
            
            # 提交每个 voxel 的处理任务
            for voxel_idx, (x, y, z) in enumerate(coords, start=1):
                futures.append(executor.submit(run_probtrackx2, roi_idx, voxel_idx, x, y, z))
        
        # 等待所有任务完成
        for fut in futures:
            fut.result()

    # 合并每个 ROI 下所有 voxel 的 fdt_paths.nii.gz 文件为一个 4D 文件
    # 由于当前工作目录在 data.bedpostX，因此ROI文件夹位于 "../data/probtrack_old/ROI_{roi_idx}"
    for roi_idx in range(1, 51):
        roi_folder = f"../data/probtrack_old/ROI_{roi_idx}"
        merged_file = f"../data/probtrack_old/ROI_{roi_idx}_merged_fdt_paths.nii.gz"
        fdt_paths_list = []
        # 遍历 ROI 文件夹下的所有 voxel 文件夹
        for voxel_dir in sorted(os.listdir(roi_folder)):
            voxel_path = os.path.join(roi_folder, voxel_dir)
            if os.path.isdir(voxel_path) and voxel_dir.startswith("voxel_"):
                fdt_file = os.path.join(voxel_path, "fdt_paths.nii.gz")
                if os.path.exists(fdt_file):
                    fdt_paths_list.append(fdt_file)
        if fdt_paths_list:
            cmd_merge = ["fslmerge", "-t", merged_file] + fdt_paths_list
            subprocess.run(cmd_merge, check=True)
            print(f"[INFO] ROI {roi_idx}: 合并了 {len(fdt_paths_list)} 个 fdt_paths 文件，生成 {merged_file}.")
        else:
            print(f"[WARNING] ROI {roi_idx}: 未找到 fdt_paths 文件，跳过合并。")

    print(f"[INFO] ROI_registration_probtrack completed for subject={sub_id}.")

if __name__ == "__main__":
    main()
