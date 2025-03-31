#!/usr/bin/env python3
import os
import glob
import re
import argparse
import subprocess

def run_command(cmd):
    """Wrapper for subprocess.run with shell=True, check=True."""
    subprocess.run(cmd, shell=True, check=True)

def search_and_prepare_data(dataset_path, sub_id):
    """
    Search for DWI files (desc-preproc_dwi.bval/bvec/nii[.gz], desc-brain_mask.nii[.gz])
    in the sub_id directory. If found, create a 'data' folder in the DWI path and copy these files there.
    Returns (data_path, True) on success; otherwise returns (None, False).
    """
    # 1) Search for possible DWI directories (either in ses-*/dwi or directly in dwi)
    possible_paths = glob.glob(f"{dataset_path}/{sub_id}/ses-*/dwi") + [f"{dataset_path}/{sub_id}/dwi"]

    data_path = ""
    found = False
    for path in possible_paths:
        if os.path.isdir(path):
            data_path = path
            found = True
            break

    if not found or not data_path:
        print(f"[WARNING] No DWI directory found for {sub_id}, skipping.")
        return None, False

    files_in_dwi = os.listdir(data_path)
    bval_file = next((f for f in files_in_dwi if f.endswith("desc-preproc_dwi.bval")), None)
    bvec_file = next((f for f in files_in_dwi if f.endswith("desc-preproc_dwi.bvec")), None)
    dwi_file  = next((f for f in files_in_dwi if re.search(r"desc-preproc_dwi\.nii(\.gz)?$", f)), None)
    mask_file = next((f for f in files_in_dwi if re.search(r"desc-brain_mask\.nii(\.gz)?$", f)), None)

    if not all([bval_file, bvec_file, dwi_file, mask_file]):
        print(f"[WARNING] Missing bval/bvec/dwi/mask for {sub_id}, skipping.")
        return None, False

    bval_file = os.path.join(data_path, bval_file)
    bvec_file = os.path.join(data_path, bvec_file)
    dwi_file  = os.path.join(data_path, dwi_file)
    mask_file = os.path.join(data_path, mask_file)

    print(f"[INFO] Found DWI path for {sub_id}: {data_path}")
    print(f"       bval={bval_file}\n       bvec={bvec_file}\n       dwi={dwi_file}\n       mask={mask_file}")

    data_dir = os.path.join(data_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    run_command(f"cp {bval_file} {data_dir}/bvals")
    run_command(f"cp {bvec_file} {data_dir}/bvecs")
    run_command(f"cp {dwi_file}  {data_dir}/data.nii.gz")
    run_command(f"cp {mask_file} {data_dir}/nodif_brain_mask.nii.gz")

    return data_path, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", 
                        default="/data2/mayupeng/BNU",
                        help="Base directory path (default: /data2/mayupeng/BNU)")
    parser.add_argument("--datasets", 
                        nargs="+", 
                        default=["Agility_sample", "BBP_sample", "MASIVAR_sample"],
                        help="List of dataset names (default: Agility_sample BBP_sample MASIVAR_sample)")
    parser.add_argument("--bedpostx_script", 
                        default="/data2/mayupeng/BNU/1_bedpostx.py",
                        help="Path to the bedpostx processing script")
    parser.add_argument("--registration_script", 
                        default="/data2/mayupeng/BNU/2_registration.py",
                        help="Path to the registration processing script")
    parser.add_argument("--roi_probtrack_script", 
                        default="/data2/mayupeng/BNU/3_ROI_probtrack.py",
                        help="Path to the probtrack processing script")
    parser.add_argument("--roi_calc_matrix_script", 
                        default="/data2/mayupeng/BNU/4_ROI_calc_matrix.py",
                        help="Path to calc matrix processing script")
    parser.add_argument("--roi_parcellation_script", 
                        default="/data2/mayupeng/BNU/5_ROI_parcellation.py",
                        help="Path to parcellation processing script")
    parser.add_argument("--roi_toMNI_script", 
                        default="/data2/mayupeng/BNU/6_ROI_toMNI.py",
                        help="Path to ROI-to-MNI processing script")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    DATASETS = args.datasets
    bedpostx_script = args.bedpostx_script
    registration_script = args.registration_script
    roi_probtrack_script = args.roi_probtrack_script
    roi_calc_matrix_script = args.roi_calc_matrix_script
    roi_parcellation_script = args.roi_parcellation_script
    roi_toMNI_script = args.roi_toMNI_script

    # Open a file to record each subject's data_path for group analysis
    data_paths_file = os.path.join(BASE_DIR, "group_data_paths.txt")

    for dataset in DATASETS:
        dataset_path = os.path.join(BASE_DIR, dataset)
        if not os.path.isdir(dataset_path):
            print(f"[ERROR] Dataset directory not found: {dataset_path}")
            continue

        subs = [d for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith("sub-")]

        for sub_id in subs:
            data_path, ok = search_and_prepare_data(dataset_path, sub_id)
            if not ok:
                continue 

            # Record the subject's data_path into the txt file
            with open(data_paths_file, "a") as f_out:
                f_out.write(f"{data_path}\n")

            # 1) Call bedpostx processing script
            cmd_bedpostx = [
                "python", bedpostx_script,
                "--data_path", data_path
            ]
            print(f"[INFO] Calling bedpostx processing for {sub_id}")
            subprocess.run(cmd_bedpostx, check=True)

            # 2) Call registration processing script
            cmd_reg = [
                "python", registration_script,
                "--data_path", data_path,
            ]
            print(f"[INFO] Calling registration processing for {sub_id}")
            subprocess.run(cmd_reg, check=True)

            # 3) Call probtrack processing script
            cmd_probtrack = [
                "python", roi_probtrack_script,
                "--data_path", data_path,
            ]
            print(f"[INFO] Calling probtrack processing for {sub_id}")
            subprocess.run(cmd_probtrack, check=True)

            # 4) Call calc_matrix processing script
            cmd_calc = [
                "python", roi_calc_matrix_script,
                "--data_path", data_path
            ]
            print(f"[INFO] Calling calc_matrix processing for {sub_id}")
            subprocess.run(cmd_calc, check=True)
            
            # 5) Call parcellation processing script with default segmentation method 'sc'
            cmd_parcellation = [
                "python", roi_parcellation_script,
                "--data_path", data_path,
                "--method", "sc"
            ]
            print(f"[INFO] Calling parcellation processing for {sub_id}")
            subprocess.run(cmd_parcellation, check=True)
            
            # 6) Call ROI-to-MNI processing script
            cmd_toMNI = [
                "python", roi_toMNI_script,
                "--data_path", data_path,
                "--method", "sc"
            ]
            print(f"[INFO] Calling ROI-to-MNI processing for {sub_id}")
            subprocess.run(cmd_toMNI, check=True)

            print(f"[INFO] Done for subject {sub_id}.\n")

    print(f"[INFO] Group data paths recorded in {data_paths_file}")

if __name__ == "__main__":
    main()
