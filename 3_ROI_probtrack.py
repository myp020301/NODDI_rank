#!/usr/bin/env python3
import subprocess
import argparse
import nibabel as nib
import numpy as np
import glob, os

def run_command(cmd):
    """Wrapper for subprocess.run with shell=True, check=True."""
    subprocess.run(cmd, shell=True, check=True)
    
def save_nonzero_coordinates(nifti_path, output_txt):
    """
    Read nonzero voxel coordinates (0-based) from the NIfTI file at nifti_path,
    and save them into the output_txt file.
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    nonzero_coords = np.argwhere(data != 0)
    with open(output_txt, 'w') as f:
        for x, y, z in nonzero_coords:
            f.write(f"{int(x)} {int(y)} {int(z)}\n")
    print(f"[INFO] Saved nonzero coordinates of {nifti_path} to {output_txt}")
    
def run_probtrack_for_roi(roi_name):
    """
    针对单个 ROI 区域运行 probtrackx2 命令，
    使用 --simple 参数一次处理 ROI 中所有 seed voxel。
    """
    seed_file = f"../data/seeds_txt_all/seed_region_{roi_name}.txt"
    mask_file = "../data/nodif_brain_mask.nii.gz"
    samples = "merged"
    output_dir = f"../data/probtrack_old/ROI_{roi_name}"
    seedref = "../data/nodif_brain_mask.nii.gz"
    
    cmd = (
        f"probtrackx2 --simple --seedref={seedref} --out={roi_name} --seed={seed_file} -l "
        f"--pd  --cthr=0.2 --nsteps=2000 --steplength=0.5 --nsamples=5000 --forcedir --opd "
        f"--opd  --samples={samples} --mask={mask_file} --dir={output_dir} "
        
    )
    print(f"[INFO] Running probtrackx2 for ROI {roi_name}")
    run_command(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,help="Path where the data/ subfolder is prepared")
    parser.add_argument("--roi_dir",default="/data2/mayupeng/BNU/ROI",help="标准空间 ROI 文件夹，文件名 *.nii 或 *.nii.gz")
    parser.add_argument("--roi_name",required=True,help="要处理的 ROI 名称（不含扩展名），如 MCP 或 FA_L")
    args = parser.parse_args()
    data_path = args.data_path
    roi_name = args.roi_name
    
    os.chdir(data_path)
    
    # 创建必要的目录（如果不存在）
    os.makedirs("data/seeds_txt_all", exist_ok=True)
    
    # 生成 seed region 文件，并保存指定 ROI 的非零坐标
    seed = f"data/{args.roi_name}_mask.nii.gz"
    seed_txt = f"data/seeds_txt_all/seed_region_{roi_name}.txt"
    save_nonzero_coordinates(seed, seed_txt)
    
    # 切换到 data.bedpostX 目录下运行 probtrackx2
    os.chdir("data.bedpostX")
    run_probtrack_for_roi(roi_name)
    
    print("[INFO] ROI processing completed.")

if __name__ == "__main__":
    main()
