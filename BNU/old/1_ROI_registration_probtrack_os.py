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
    如果只是做 targetmasks 并不一定要坐标，但留着函数以备后续需要。
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()  # float64
    nonzero_coords = np.argwhere(data != 0)  # shape (N, 3)

    # 将坐标写入txt
    with open(output_txt, 'w') as f:
        for x, y, z in nonzero_coords:
            f.write(f"{int(x)} {int(y)} {int(z)}\n")

    print(f"[INFO] Saved nonzero coordinates of {nifti_path} to {output_txt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path where data/ subfolder already prepared (with data.nii.gz, nodif_brain_mask.nii.gz, bvals, bvecs)")
    parser.add_argument("--sub_id", required=True)
    parser.add_argument("--FMRIB_FA_TEMPLATE", required=True, default="$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz")
    parser.add_argument("--JHU50_SEED", required=True, default="$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz")
    # 新增: BN_Atlas_246_1mm 的路径（假设你放在固定位置，或你也可改成固定写死）
    parser.add_argument("--BN246_Atlas", default="/data2/mayupeng/BNU/BN_Atlas_246_1mm.nii.gz",
                        help="Path to BN_Atlas_246_1mm.nii.gz (default /data2/mayupeng/BNU/BN_Atlas_246_1mm.nii.gz)")
    args = parser.parse_args()

    data_path = args.data_path
    sub_id    = args.sub_id
    FMRIB_FA_TEMPLATE = args.FMRIB_FA_TEMPLATE
    JHU50_SEED = args.JHU50_SEED
    BN246_Atlas = args.BN246_Atlas

    print(f"[INFO] ROI_registration_probtrack for subject={sub_id}, data_path={data_path}")
    os.chdir(data_path)

    # 1) bedpostx
    if os.path.isdir("data.bedpostX"):
        print(f"[WARNING] data.bedpostX already exists, skipping bedpostx")
        return
    else:
        run_command("bedpostx_gpu data -n 3")

    # 2) dtifit
    run_command("dtifit -k data/data.nii.gz -o data/dtifit -m data/nodif_brain_mask.nii.gz -r data/bvecs -b data/bvals")

    # 3) FLIRT
    run_command(f"flirt -in data/dtifit_FA.nii.gz -ref {FMRIB_FA_TEMPLATE} -out data/fa_standard_space.nii.gz -omat data/fa2standard.mat")

    # 4) FNIRT
    run_command(f"fnirt --in=data/dtifit_FA.nii.gz --aff=data/fa2standard.mat --ref={FMRIB_FA_TEMPLATE} --iout=data/fa_standard_space_nonlin.nii.gz --cout=data/fa2standard_warp.nii.gz")

    # 5) Inverse warp for JHU50
    run_command("invwarp -w data/fa2standard_warp.nii.gz -r data/dtifit_FA.nii.gz -o data/JHU50_inverse_warp.nii.gz")
    run_command(f"applywarp -i {JHU50_SEED} -r data/dtifit_FA.nii.gz -w data/JHU50_inverse_warp.nii.gz -o data/JHU50_native_nn.nii.gz --interp=nn")

    # 5.1) Inverse warp for BN_Atlas_246_1mm
    run_command(f"applywarp -i {BN246_Atlas} -r data/dtifit_FA.nii.gz -w data/JHU50_inverse_warp.nii.gz -o data/BN246_native_nn.nii.gz --interp=nn")

    # 创建一些文件夹
    os.makedirs("data/seeds_region_all", exist_ok=True)  # 存 JHU 50 seeds
    os.makedirs("data/bn246_region_all", exist_ok=True)  # 存 BN246 seeds
    os.makedirs("data/seeds_txt_all", exist_ok=True)     # 如果需要坐标txt
    
    # 6) Create 50 JHU seeds
    for i in range(1, 51):
        seed_out = f"data/seeds_region_all/seed_region_{i}.nii.gz"
        run_command(f"fslmaths data/JHU50_native_nn.nii.gz -thr {i} -uthr {i} -bin {seed_out}")

        # 保存坐标
        seed_txt = f"data/seeds_txt_all/seed_region_{i}.txt"
        save_nonzero_coordinates(seed_out, seed_txt)

    # 7) Create 246 BN seeds
    # BN_Atlas_246_1mm 通常标号 1..246
    for i in range(1, 247):
        bn_out = f"data/bn246_region_all/bn246_region_{i}.nii.gz"
        run_command(f"fslmaths data/BN246_native_nn.nii.gz -thr {i} -uthr {i} -bin {bn_out}")


    # 8) 生成 seeds_list.txt (依然 50 行, 只包含 JHU seeds)
    with open("data/seed_list.txt", "w") as f:
        for i in range(1, 51):
            f.write(f"../data/seeds_region_all/seed_region_{i}.nii.gz\n")

    # 9) 生成 target_list.txt (融合 50 seeds + 246 BN)
    # 先写 50 seeds, 再写 246 BN
    with open("data/target_list.txt", "w") as f:
        # seeds(1..50)
        for i in range(1, 51):
            f.write(f"../data/seeds_region_all/seed_region_{i}.nii.gz\n")
        # BN246(1..246)
        for i in range(1, 247):
            f.write(f"../data/bn246_region_all/bn246_region_{i}.nii.gz\n")

    # 10) probtrackx2
    if os.path.isdir("data.bedpostX"):
        os.chdir("data.bedpostX")
        # seed=seed_list.txt (50 seeds), targetmasks=target_list.txt(50+246)
        run_command("probtrackx2_gpu --seed=../data/seed_list.txt "
                    "--mask=../data/nodif_brain_mask.nii.gz "
                    "--samples=merged "
                    "--targetmasks=../data/target_list.txt "
                    "--dir=../data/probtrack_os2t_246/ --forcedir --os2t --opd")

        # Merge seeds
        os.makedirs("../data/seeds_result_all", exist_ok=True)
        # seeds_{X}_to_seed_region_{Y}.nii.gz => 这里X=0..(50-1?), 
        # 可能现在X的上限=50+BN(??) => 其实probtrackx2会产生 seeds_X_to_seed_{Y}, 
        # X in [0..49], Y in [1..(50+246)] if there's code in FSL
        # 这里还是按 0..49 遍历?
        # 先保留原逻辑:
        for X in range(0, 50):
            input_files = []
            for Y in range(1, 51):
                # JHU seeds 1..50
                in_path = f"../data/probtrack_os2t_246/seeds_{X}_to_seed_region_{Y}.nii.gz"
                if os.path.isfile(in_path):
                    input_files.append(in_path)
            for Y in range(1, 247):
                # BN seeds 1..246
                in_path = f"../data/probtrack_os2t_246/seeds_{X}_to_bn246_region_{Y}.nii.gz"
                if os.path.isfile(in_path):
                    input_files.append(in_path)
            out_4d = f"../data/seeds_result_all/seed_{X + 1}_to_targets_all.nii.gz"
            if input_files:
                run_command(f"fslmerge -t {out_4d} {' '.join(input_files)}")
                print(f"[INFO] Merged seed_{X + 1}_to_targets_all.nii.gz into a 4D file.")
    else:
        print("[ERROR] bedpostx output folder 'data.bedpostX' not found.")

    print(f"[INFO] ROI_registration_probtrack completed for subject={sub_id}.")


if __name__ == "__main__":
    main()
