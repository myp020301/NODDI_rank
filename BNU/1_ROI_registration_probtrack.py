#!/usr/bin/env python3
import os
import subprocess
import glob
import re
import argparse
import nibabel as nib
import numpy as np

def run_command(cmd):
    subprocess.run(cmd, shell=True, check=True)
    
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
        return
    else:
        run_command("bedpostx_gpu data -n 3")

    # dtifit
    run_command("dtifit -k data/data.nii.gz -o data/dtifit -m data/nodif_brain_mask.nii.gz -r data/bvecs -b data/bvals")

    # FLIRT
    run_command(f"flirt -in data/dtifit_FA.nii.gz -ref {FMRIB_FA_TEMPLATE} -out data/fa_standard_space.nii.gz -omat data/fa2standard.mat")

    # FNIRT
    run_command(f"fnirt --in=data/dtifit_FA.nii.gz --aff=data/fa2standard.mat --ref={FMRIB_FA_TEMPLATE} --iout=data/fa_standard_space_nonlin.nii.gz --cout=data/fa2standard_warp.nii.gz")

    # Inverse warp
    run_command("invwarp -w data/fa2standard_warp.nii.gz -r data/dtifit_FA.nii.gz -o data/JHU50_inverse_warp.nii.gz")
    run_command(f"applywarp -i {JHU50_SEED} -r data/dtifit_FA.nii.gz -w data/JHU50_inverse_warp.nii.gz -o data/JHU50_native_nn.nii.gz --interp=nn")

    os.makedirs("data/seeds_txt_all", exist_ok=True)
    os.makedirs("data/seeds_region_all", exist_ok=True)
    # Create individual seed region files (1 to 50)
    for i in range(1, 51):
        seed_out = f"data/seeds_region_all/seed_region_{i}.nii.gz"
        run_command(f"fslmaths data/JHU50_native_nn.nii.gz -thr {i} -uthr {i} -bin {seed_out}")

        # 这里新增：保存非0坐标到 seed_region_{i}.txt
        seed_txt = f"data/seeds_txt_all/seed_region_{i}.txt"
        save_nonzero_coordinates(seed_out, seed_txt)
        
    with open("data/seed_list.txt", "w") as f:
        for i in range(1, 51):
            f.write(f"../data/seeds_region_all/seed_region_{i}.nii.gz\n")

    # probtrackx2
    if os.path.isdir("data.bedpostX"):
        os.chdir("data.bedpostX")
        run_command("probtrackx2_gpu --seed=../data/seed_list.txt --mask=../data/nodif_brain_mask.nii.gz --samples=merged --targetmasks=../data/seed_list.txt --dir=../data/probtrack_os2t/ --forcedir --os2t --opd")

        # Merge seeds
        os.makedirs("../data/seeds_result_all", exist_ok=True)
        for X in range(0, 50):
            input_files = []
            for Y in range(1, 51):
                in_path = f"../data/probtrack_os2t/seeds_{X}_to_seed_region_{Y}.nii.gz"
                input_files.append(in_path)
            out_4d = f"../data/seeds_result_all/seed_{X + 1}_to_seeds_all.nii.gz"
            run_command(f"fslmerge -t {out_4d} {' '.join(input_files)}")
            print(f"[INFO] Merged seed_{X + 1}_to_seeds_all.nii.gz into a 4D file.")
    else:
        print("[ERROR] bedpostx output folder 'data.bedpostX' not found.")

    print(f"[INFO] ROI_registration_probtrack completed for subject={sub_id}.")

if __name__ == "__main__":
    main()
