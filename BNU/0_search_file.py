#!/usr/bin/env python3
import os
import glob
import re
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_command(cmd):
    """Wrapper for subprocess.run with shell=True, check=True."""
    subprocess.run(cmd, shell=True, check=True)

def search_and_prepare_data(dataset_path, sub_id):
    """
    Search for DWI files (desc-preproc_dwi.bval/bvec/nii[.gz], desc-brain_mask.nii[.gz])
    in the sub_id directory. If found, create a 'data' folder in the DWI path and copy these files there.
    Returns (data_path, True) on success; otherwise returns (None, False).
    """
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

def run_subject_level_steps(data_path, bedpostx_script, registration_script, sub_id):
    """
    Run subject-level processing steps that do not depend on ROI.
    These include:
      1. Bedpostx processing.
      2. Registration processing.
    """
    cmd_bedpostx = [
        "python", bedpostx_script,
        "--data_path", data_path
    ]
    print(f"[INFO] Calling bedpostx processing for {sub_id}")
    subprocess.run(cmd_bedpostx, check=True)

    cmd_reg = [
        "python", registration_script,
        "--data_path", data_path,
    ]
    print(f"[INFO] Calling registration processing for {sub_id}")
    subprocess.run(cmd_reg, check=True)

def run_probtrack_step(data_path, roi, roi_probtrack_script):
    cmd = [
        "python", roi_probtrack_script,
        "--data_path", data_path,
        "--roi", str(roi)
    ]
    print(f"[INFO] Running probtrack for ROI {roi} at {data_path}")
    subprocess.run(cmd, check=True)

def run_calc_matrix_step(data_path, roi, roi_calc_matrix_script):
    cmd = [
        "python", roi_calc_matrix_script,
        "--data_path", data_path,
        "--roi", str(roi)
    ]
    print(f"[INFO] Running calc_matrix for ROI {roi} at {data_path}")
    subprocess.run(cmd, check=True)

def run_parcellation_step(data_path, roi, roi_parcellation_script):
    cmd = [
        "python", roi_parcellation_script,
        "--data_path", data_path,
        "--method", "sc",
        "--roi", str(roi)
    ]
    print(f"[INFO] Running parcellation for ROI {roi} at {data_path}")
    subprocess.run(cmd, check=True)

def run_toMNI_step(data_path, roi, roi_toMNI_script):
    cmd = [
        "python", roi_toMNI_script,
        "--data_path", data_path,
        "--method", "sc",
        "--roi", str(roi)
    ]
    print(f"[INFO] Running ROI-to-MNI for ROI {roi} at {data_path}")
    subprocess.run(cmd, check=True)

def run_roi_steps_for_all_subjects(roi, subject_data, roi_probtrack_script,
                                   roi_calc_matrix_script, roi_parcellation_script, roi_toMNI_script):
    """
    For a given ROI, run the ROI-dependent steps for all subjects in sequence:
      Step 3: Probabilistic tractography.
      Step 4: Cross-correlation matrix calculation.
      Step 5: Clustering/parcellation.
      Step 6: ROI-to-MNI transformation.
    Each step is executed in parallel across subjects, and the next step is only started once all subjects have completed the previous one.
    """
    steps = [
        ("Probtrack", run_probtrack_step, roi_probtrack_script),
        ("CalcMatrix", run_calc_matrix_step, roi_calc_matrix_script),
        ("Parcellation", run_parcellation_step, roi_parcellation_script),
        ("ROI_to_MNI", run_toMNI_step, roi_toMNI_script)
    ]
    for step_name, step_func, script in steps:
        print(f"[INFO] Starting {step_name} for ROI {roi} across all subjects")
        with ProcessPoolExecutor() as executor:
            futures = []
            for data_path in subject_data:
                futures.append(executor.submit(step_func, data_path, roi, script))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] An error occurred during {step_name} for ROI {roi}: {e}")
        print(f"[INFO] Completed {step_name} for ROI {roi}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", 
                        default="/data2/mayupeng/BNU",
                        help="Base directory path (default: /data2/mayupeng/BNU)")
    parser.add_argument("--datasets", 
                        nargs="+", 
                        default=["BBP_sample"],
                        help="List of dataset names (default: Agility_sample BBP_sample MASIVAR_sample)")
    parser.add_argument("--bedpostx_script", 
                        default="/data2/mayupeng/BNU/1_bedpostx.py",
                        help="Path to the bedpostx processing script")
    parser.add_argument("--registration_script", 
                        default="/data2/mayupeng/BNU/2_registration.py",
                        help="Path to the registration processing script")
    parser.add_argument("--roi_probtrack_script", 
                        default="/data2/mayupeng/BNU/3_modi.py",
                        help="Path to the probtrack processing script")
    parser.add_argument("--roi_calc_matrix_script", 
                        default="/data2/mayupeng/BNU/4_modi.py",
                        help="Path to calc matrix processing script")
    parser.add_argument("--roi_parcellation_script", 
                        default="/data2/mayupeng/BNU/5_modi.py",
                        help="Path to parcellation processing script")
    parser.add_argument("--roi_toMNI_script", 
                        default="/data2/mayupeng/BNU/6_modi.py",
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

    data_paths_file = os.path.join(BASE_DIR, "group_data_paths.txt")
    # Create/overwrite the file without a header.
    with open(data_paths_file, "w") as f_out:
        pass

    # Loop over datasets and subjects to record data paths and run subject-level steps.
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
            # Record subject's data_path.
            with open(data_paths_file, "a") as f_out:
                f_out.write(f"{data_path}\n")
            print(f"[INFO] Processing subject {sub_id} (subject-level steps)")
            run_subject_level_steps(data_path, bedpostx_script, registration_script, sub_id)

    print(f"[INFO] Group data paths recorded in {data_paths_file}")

    # Now run ROI-dependent steps by reading the data paths from the file.
    with open(data_paths_file, "r") as f_in:
        subject_data = [line.strip() for line in f_in if line.strip()]

    num_rois = 50
    # For ROI-dependent processing, the outer loop is over ROI.
    for roi in range(1, num_rois + 1):
        print(f"[INFO] Starting ROI-level processing for ROI {roi}")
        run_roi_steps_for_all_subjects(roi, subject_data, roi_probtrack_script, 
                                       roi_calc_matrix_script, roi_parcellation_script, roi_toMNI_script)
        print(f"[INFO] Completed ROI-level processing for ROI {roi}")

    print("[INFO] All ROI-level processing completed.")


if __name__ == "__main__":
    main()
