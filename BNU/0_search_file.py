#!/usr/bin/env python3
import os
import glob
import re
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

NJOBS=10

###############################################################################
# Utility Functions
###############################################################################
def run_command(cmd):
    """Wrapper for subprocess.run with shell=True, check=True."""
    subprocess.run(cmd, shell=True, check=True)

def search_and_prepare_data_BNU(dataset_path, sub_id):
    """
    (Step 1) 搜索每个 subject 下的 DWI 文件（bval/bvec/dwi/mask），
    如果找到则在对应 dwi 目录下创建 data 文件夹并复制文件。
    返回 (data_path, True) 表示成功，否则返回 (None, False)。
    """
    possible_paths = glob.glob(f"{dataset_path}/{sub_id}/ses-*/dwi") + [f"{dataset_path}/{sub_id}/dwi"]
    data_path = ""
    for path in possible_paths:
        if os.path.isdir(path):
            data_path = path
            break

    if not data_path:
        print(f"[WARNING] No DWI directory found for {sub_id}, skipping.")
        return None, False

    files_in_dwi = os.listdir(data_path)
    bval_file = next((f for f in files_in_dwi if f.endswith("desc-preproc_dwi.bval")), None)
    bvec_file = next((f for f in files_in_dwi if f.endswith("desc-preproc_dwi.bvec")), None)
    dwi_file  = next((f for f in files_in_dwi if re.search(r"desc-preproc_dwi\.nii(\.gz)?$", f)), None)
    mask_file = next((f for f in files_in_dwi if re.search(r"desc-brain_mask\.nii(\.gz)?$", f)), None)

    if not all([bval_file, bvec_file, dwi_file, mask_file]):
        print(f"[WARNING] Missing required files for {sub_id}, skipping.")
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

def search_and_prepare_data_HCP(dataset_path, sub_id):
    """
    针对 HCP_25 数据集，每个 subject 文件夹下直接存在以下文件：
      - data.nii.gz
      - nodif_brain_mask.nii.gz
      - dti.bval
      - dti.bvec
    在 subject 文件夹下建立 data 文件夹，并将上述文件复制进去。
    返回 (subject_folder, True) 表示成功，否则返回 (None, False)。
    """
    subject_dir = os.path.join(dataset_path, sub_id)
    if not os.path.isdir(subject_dir):
        print(f"[WARNING] Subject directory not found: {subject_dir}, skipping.")
        return None, False

    # 检查必需的文件是否存在
    data_file = os.path.join(subject_dir, "data.nii.gz")
    mask_file = os.path.join(subject_dir, "nodif_brain_mask.nii.gz")
    bval_file = os.path.join(subject_dir, "dti.bval")
    bvec_file = os.path.join(subject_dir, "dti.bvec")

    if not all([os.path.exists(data_file), os.path.exists(mask_file), os.path.exists(bval_file), os.path.exists(bvec_file)]):
        print(f"[WARNING] Missing required files in {subject_dir}, skipping.")
        return None, False

    print(f"[INFO] Found subject folder: {subject_dir}")
    print(f"       data={data_file}\n       mask={mask_file}\n       bval={bval_file}\n       bvec={bvec_file}")

    # 在 subject 文件夹下创建 data 文件夹
    data_dir = os.path.join(subject_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 复制文件到 data 文件夹中
    run_command(f"cp {data_file} {data_dir}/data.nii.gz")
    run_command(f"cp {mask_file} {data_dir}/nodif_brain_mask.nii.gz")
    run_command(f"cp {bval_file} {data_dir}/bvals")
    run_command(f"cp {bvec_file} {data_dir}/bvecs")

    return subject_dir, True

###############################################################################
# Subject-Level Processing (Step 1,2: 串行处理每个 subject)
###############################################################################
def process_subjects(BASE_DIR, DATASETS, bedpostx_script, registration_script, data_paths_file):
    """
    遍历每个 dataset 和其下的 subject：
      - 查找并准备数据
      - 执行 subject 级处理（bedpostx、registration）
      - 将每个 subject 的 data_path 记录到 data_paths_file 中
    """
    with open(data_paths_file, "w") as f_out:
        pass  # 清空文件

    for dataset in DATASETS:
        dataset_path = os.path.join(BASE_DIR, dataset)
        if not os.path.isdir(dataset_path):
            print(f"[ERROR] Dataset directory not found: {dataset_path}")
            continue

        subs = [d for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))]
        for sub_id in subs:
            data_path, ok = search_and_prepare_data_HCP(dataset_path, sub_id)
            if not ok:
                continue
            # 记录 subject 数据路径
            with open(data_paths_file, "a") as f_out:
                f_out.write(f"{data_path}\n")
            print(f"[INFO] Processing subject {sub_id} (subject-level steps)")
            run_subject_level_steps(data_path, bedpostx_script, registration_script, sub_id)

def run_subject_level_steps(data_path, bedpostx_script, registration_script, sub_id):
    """
    执行 subject 级处理：调用 bedpostx 和 registration 脚本
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

###############################################################################
# ROI-Level Processing (Step 3-6: ROI 串行处理，每步对 subject 并行)
###############################################################################
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

def run_roi_steps_for_all_subjects(subject_data, roi_probtrack_script,
                                   roi_calc_matrix_script, roi_parcellation_script, roi_toMNI_script, num_rois):
    """
    针对给定 ROI，依次执行以下步骤：
      Step 3: Probabilistic tractography
      Step 4: Cross-correlation matrix calculation
      Step 5: Clustering/parcellation
      Step 6: ROI-to-MNI transformation
    每一步在所有 subject 内并行执行，ROI 之间串行。
    """
    for roi in range(1, num_rois + 1):
        print(f"[INFO] Processing ROI {roi}...")
        steps = [
            ("Probtrack", run_probtrack_step, roi_probtrack_script),
            ("CalcMatrix", run_calc_matrix_step, roi_calc_matrix_script),
            ("Parcellation", run_parcellation_step, roi_parcellation_script),
            ("ROI_to_MNI", run_toMNI_step, roi_toMNI_script)
        ]
        for step_name, step_func, script in steps:
            print(f"[INFO] Starting {step_name} for ROI {roi} across all subjects")
            with ProcessPoolExecutor(max_workers=NJOBS) as executor:
                futures = [executor.submit(step_func, data_path, roi, script) for data_path in subject_data]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] {step_name} error for ROI {roi}: {e}")
            print(f"[INFO] Completed {step_name} for ROI {roi}")
        print(f"[INFO] ROI {roi} processing completed.")  

###############################################################################
# Group-Level Processing (Step 7: ROI 串行，内部 subject 并行调用外部 group_refer 脚本)
###############################################################################
def run_group_level_analysis(group_refer_script, BASE_DIR, subject_data, num_rois):
    """
    封装的函数：对于每个 ROI（1到 num_rois），依次调用 group_refer 脚本完成群体统计分析。
    """
    subject_str = ",".join(subject_data)
    for roi in range(1, num_rois + 1):
        print(f"[INFO] Starting group-level analysis for ROI index {roi}")
        cmd = [
            "python", group_refer_script,
            "--base_dir", BASE_DIR,
            "--roi", str(roi),
            "--subject_data", subject_str
        ]
        subprocess.run(cmd, check=True)
        print(f"[INFO] Completed group-level analysis for ROI index {roi}")

###############################################################################
# Main Function
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    # Subject-level and ROI-level parameters
    parser.add_argument("--base_dir", default="/data2/mayupeng/BNU",
                        help="Base directory path (default: /data2/mayupeng/BNU)")
    parser.add_argument("--datasets", 
                        nargs="+", 
                        default=["Agility_sample","BBP_sample","MASIVAR_sample"],
                        help="List of dataset names (default: Agility_sample BBP_sample MASIVAR_sample)")
    parser.add_argument("--bedpostx_script", default="/data2/mayupeng/BNU/1_bedpostx.py",
                        help="Path to the bedpostx processing script")
    parser.add_argument("--registration_script", default="/data2/mayupeng/BNU/2_registration.py",
                        help="Path to the registration processing script")
    parser.add_argument("--roi_probtrack_script", default="/data2/mayupeng/BNU/3_ROI_probtrack.py",
                        help="Path to the probtrack processing script")
    parser.add_argument("--roi_calc_matrix_script", default="/data2/mayupeng/BNU/4_ROI_calc_matrix.py",
                        help="Path to calc matrix processing script")
    parser.add_argument("--roi_parcellation_script", default="/data2/mayupeng/BNU/5_ROI_parcellation.py",
                        help="Path to parcellation processing script")
    parser.add_argument("--roi_toMNI_script", default="/data2/mayupeng/BNU/6_ROI_toMNI.py",
                        help="Path to ROI-to-MNI processing script")
    # Group-level parameter：group_refer 脚本路径
    parser.add_argument("--group_refer_script", default="7_group_refer.py",
                        help="Path to the group_refer script")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    DATASETS = args.datasets

    # 定义保存所有 subject data path 的文件
    data_paths_file = os.path.join(BASE_DIR, "group_data_paths.txt")
    
    # ------------------------------
    # 1. Subject-level Processing (串行)
    # ------------------------------
    print("[INFO] Starting subject-level processing...")
    process_subjects(BASE_DIR, DATASETS, args.bedpostx_script, args.registration_script, data_paths_file)
    print(f"[INFO] Subject-level processing completed. Data paths recorded in {data_paths_file}")

    # 读取所有 subject data path
    with open(data_paths_file, "r") as f:
        subject_data = [line.strip() for line in f if line.strip()]
    
    num_rois = 50
    # ------------------------------
    # 2. ROI-level Processing (ROI 串行，内部 subject 并行)
    # ------------------------------
    print("[INFO] Starting ROI-level processing...")
    run_roi_steps_for_all_subjects(subject_data, args.roi_probtrack_script, args.roi_calc_matrix_script, args.roi_parcellation_script, args.roi_toMNI_script, num_rois)
    print("[INFO] All ROI-level processing completed.")  
    # ------------------------------
    # 3. Group-level Processing (ROI 串行调用外部脚本)
    # ------------------------------
    print("[INFO] Starting group-level processing...")
    run_group_level_analysis(args.group_refer_script, BASE_DIR, subject_data, num_rois)
    print("[INFO] Group-level processing completed.")

    # 若后续需要其他群体级分析（例如：利用匈牙利算法返回配准的 label 到个体空间），可在此处增加相应的函数调用。
    
if __name__ == "__main__":
    main()
