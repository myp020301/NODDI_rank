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
    在 sub_id 目录下搜索DWI文件(desc-preproc_dwi.bval/bvec/nii[.gz], desc-brain_mask.nii[.gz])。
    如果找到，则在 data_path 下创建 data/ 文件夹并复制这些文件进去。
    返回 (data_path, True) 表示成功；否则返回 (None, False)。
    """
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

    # 拼出绝对路径
    bval_file = os.path.join(data_path, bval_file)
    bvec_file = os.path.join(data_path, bvec_file)
    dwi_file  = os.path.join(data_path, dwi_file)
    mask_file = os.path.join(data_path, mask_file)

    print(f"[INFO] Found DWI path for {sub_id}: {data_path}")
    print(f"       bval={bval_file}\n       bvec={bvec_file}\n       dwi={dwi_file}\n       mask={mask_file}")

    # 在 data_path 下创建 data/ 文件夹并复制
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
    parser.add_argument("--roi_registration_script", default="/data2/mayupeng/BNU/1_ROI_registration_probtrack_old.py",
                        help="Path to ROI_registration_probtrack.py")
    parser.add_argument("--roi_calc_matrix_script", default="/data2/mayupeng/BNU/2_ROI_calc_matrix.py",
                        help="Path to ROI_calc_matrix.py")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    DATASETS = args.datasets
    roi_registration_script = args.roi_registration_script
    roi_calc_matrix_script = args.roi_calc_matrix_script
    

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
                continue  # skip this subject

            # 2) 调用 ROI_registration_probtrack.py
            #    需要 data_path, sub_id, FMRIB_FA_TEMPLATE, JHU50_SEED
            cmd_reg = [
                "python", roi_registration_script,
                "--data_path", data_path,
                "--sub_id", sub_id,
            ]
            print(f"[INFO] Calling ROI_registration_probtrack.py for {sub_id}")
            subprocess.run(cmd_reg, check=True)

            # 3) 调用 ROI_calc_matrix.py
            #    假设其默认参数即可 (roi_coord_folder="data/seeds_list_all",
            #    input_folder="data/seeds_all", output_folder="data/con_cor"),
            #    需要先切换到 data_path 执行(或也可直接在 search_dwi_files 这里加参数).
            os.chdir(data_path)
            cmd_calc = [
                "python", roi_calc_matrix_script,
                # 如果需要自定义参数可加:
                "--data_path", data_path
                # "--roi_coord_folder", "data/seeds_list_all",
                # "--input_folder",    "data/seeds_all",
                # "--output_folder",   "data/con_cor",
                # ...
            ]
            print(f"[INFO] Calling ROI_calc_matrix.py for {sub_id}")
            subprocess.run(cmd_calc, check=True)

            print(f"[INFO] Done for subject {sub_id}.\n")

if __name__ == "__main__":
    main()
