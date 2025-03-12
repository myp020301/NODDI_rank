#!/usr/bin/env python3
import os
import subprocess
import glob
import re
import argparse
import nibabel as nib
import numpy as np

def run_command(command):
    """Executes a shell command using subprocess."""
    subprocess.run(command, shell=True, check=True)

def create_4d_image(input_files, output_file):
    """Merge multiple 3D NIfTI files into a single 4D NIfTI file using fslmerge."""
    # 确保输出文件所在的目录已创建

    cmd = f"fslmerge -t {output_file} {' '.join(input_files)}"
    run_command(cmd)
    print(f"[INFO] Merged files into 4D image: {output_file}")

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

def process_subject(dataset_path, sub_id, FMRIB_FA_TEMPLATE, JHU50_SEED):
    """
    Process each subject and handle DWI directory search and further processing.
    """
    print(f"[INFO] Processing subject: {sub_id}")

    # 1) Search for possible DWI paths (ses-*/dwi or directly dwi)
    possible_paths = glob.glob(f"{dataset_path}/{sub_id}/ses-*/dwi") + [f"{dataset_path}/{sub_id}/dwi"]
    
    data_path = ""
    found = False
    for path in possible_paths:
        if os.path.isdir(path):
            data_path = path
            found = True
            break

    if not found or not data_path:
        print(f"[WARNING] No DWI directory found for {sub_id}, skipping...")
        return

    print(f"[INFO] Found DWI path: {data_path}")
    os.chdir(data_path)
    
    # Locate necessary files (bval, bvec, dwi, mask)
    bval_file = next((f for f in os.listdir() if f.endswith("desc-preproc_dwi.bval")), None)
    bvec_file = next((f for f in os.listdir() if f.endswith("desc-preproc_dwi.bvec")), None)

    # Use regex to match .nii or .nii.gz
    dwi_file = next((f for f in os.listdir() if re.search(r"desc-preproc_dwi\.nii(\.gz)?$", f)), None)
    mask_file = next((f for f in os.listdir() if re.search(r"desc-brain_mask\.nii(\.gz)?$", f)), None)

    if not all([bval_file, bvec_file, dwi_file, mask_file]):
        print(f"[WARNING] Missing required files for {sub_id}, skipping...")
        return

    # Prepare the data folder
    os.makedirs("data", exist_ok=True)
    os.system(f"cp {dwi_file} data/data.nii.gz")
    os.system(f"cp {mask_file} data/nodif_brain_mask.nii.gz")
    os.system(f"cp {bval_file} data/bvals")
    os.system(f"cp {bvec_file} data/bvecs")

    print(f"[INFO] Files prepared in data/ for subject {sub_id}")

    # Check if bedpostx has already been run
    if os.path.isdir("data.bedpostX"):
        print(f"[WARNING] data.bedpostX already exists for {sub_id}. Skipping bedpostx processing.")
        return
    else:
        # Run bedpostx_gpu
        run_command("bedpostx_gpu data -n 3")

    # Compute FA map using dtifit
    run_command("dtifit -k data/data.nii.gz -o data/dtifit -m data/nodif_brain_mask.nii.gz -r data/bvecs -b data/bvals")

    # Register individual FA map to the FMRIB FA template using FLIRT
    run_command(f"flirt -in data/dtifit_FA.nii.gz -ref {FMRIB_FA_TEMPLATE} -out data/fa_standard_space.nii.gz -omat data/fa2standard.mat")

    # Perform FNIRT nonlinear registration
    run_command(f"fnirt --in=data/dtifit_FA.nii.gz --aff=data/fa2standard.mat --ref={FMRIB_FA_TEMPLATE} --iout=data/fa_standard_space_nonlin.nii.gz --cout=data/fa2standard_warp.nii.gz")

    # Inverse transform the JHU50 seed region from standard space to native space
    run_command(f"invwarp -w data/fa2standard_warp.nii.gz -r data/dtifit_FA.nii.gz -o data/JHU50_inverse_warp.nii.gz")
    run_command(f"applywarp -i {JHU50_SEED} -r data/dtifit_FA.nii.gz -w data/JHU50_inverse_warp.nii.gz -o data/JHU50_native_nn.nii.gz --interp=nn")
    
    os.makedirs("/data/seeds_list_all", exist_ok=True)
    # Create individual seed region files (1 to 50)
    for i in range(1, 51):
        seed_out = f"data/seed_region_{i}.nii.gz"
        run_command(f"fslmaths data/JHU50_native_nn.nii.gz -thr {i} -uthr {i} -bin {seed_out}")

        # 这里新增：保存非0坐标到 seed_region_{i}.txt
        seed_txt = f"data/seeds_list_all/seed_region_{i}.txt"
        save_nonzero_coordinates(seed_out, seed_txt)
        
    # Prepare seed list file
    with open("data/seed_list.txt", "w") as f:
        for i in range(1, 51):
            f.write(f"../data/seed_region_{i}.nii.gz\n")

    # Run probtrackx2_gpu for each seed region (1 to 50)
    if os.path.isdir("data.bedpostX"):
        os.chdir("data.bedpostX")
        run_command("probtrackx2_gpu --seed=../data/seed_list.txt --mask=../data/nodif_brain_mask.nii.gz --samples=merged --targetmasks=../data/seed_list.txt --dir=../data/probtrack_py/ --forcedir --os2t --opd")

        # Loop over X and create 4D images for each X value (0-49)
        # 在此确保 ../data/seeds_all 存在
        os.makedirs("../data/seeds_all", exist_ok=True)

        for X in range(0, 50):
            input_files = []
            for Y in range(1, 51):
                in_path = f"../data/probtrack_py/seeds_{X}_to_seed_region_{Y}.nii.gz"
                input_files.append(in_path)
            # Merge the 3D images into a 4D file
            out_4d = f"../data/seeds_all/seeds_{X + 1}_to_seed_all.nii.gz"
            create_4d_image(input_files, out_4d)
            print(f"[INFO] Merged seeds_{X + 1}_to_seed_region_all.nii.gz into a 4D file.")
    else:
        print("[ERROR] bedpostx output folder 'data.bedpostX' not found.")
    
    print(f"[INFO] Finished processing {sub_id}")

def main():
    """Main function to process all datasets and subjects."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Base directory path")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="List of dataset names, e.g. --datasets Agility_sample BBP_sample MASIVAR_sample")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    DATASETS = args.datasets

    FMRIB_FA_TEMPLATE = "$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz"
    JHU50_SEED = "$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz"

    for dataset in DATASETS:
        print(f"[INFO] Processing dataset: {dataset}")

        dataset_path = os.path.join(BASE_DIR, dataset)
        if not os.path.isdir(dataset_path):
            print(f"[ERROR] Dataset directory not found: {dataset_path}")
            continue

        os.chdir(dataset_path)

        # Debugging: Print out the files and folders in the dataset directory
        print(f"[INFO] Files and directories in {dataset_path}: {os.listdir(dataset_path)}")

        # Traverse all subjects (sub-*) in this dataset
        for sub_id in os.listdir(dataset_path):
            # Check if it's a directory that starts with "sub-"
            if os.path.isdir(os.path.join(dataset_path, sub_id)) and sub_id.startswith("sub-"):
                process_subject(dataset_path, sub_id, FMRIB_FA_TEMPLATE, JHU50_SEED)

if __name__ == "__main__":
    main()
