#!/usr/bin/env python3
import os, glob, re, argparse, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

NJOBS = 20

def run_command(command: str):
    """Run a shell command and exit on failure."""
    print(f"[CMD] {command}")
    subprocess.run(command, shell=True, check=True)

def prepare_subject_data_BNU(dataset_directory: str, subject_id: str, use_t1: bool=False):
    candidate = glob.glob(f"{dataset_directory}/{subject_id}/ses-*/dwi") + [f"{dataset_directory}/{subject_id}/dwi"]
    dwi_dir = next((p for p in candidate if os.path.isdir(p)), None)
    if dwi_dir is None:
        print(f"[WARN] {subject_id} 未找到 DWI 目录，跳过。")
        return None, False

    files_in_dwi = os.listdir(dwi_dir)
    bval = next((f for f in files_in_dwi if f.endswith("desc-preproc_dwi.bval")), None)
    bvec = next((f for f in files_in_dwi if f.endswith("desc-preproc_dwi.bvec")), None)
    dwi  = next((f for f in files_in_dwi if re.search(r"desc-preproc_dwi\.nii(\.gz)?$", f)), None)
    mask = next((f for f in files_in_dwi if re.search(r"desc-brain_mask\.nii(\.gz)?$", f)), None)
    if not all([bval, bvec, dwi, mask]):
        print(f"[WARN] {subject_id} 缺少必需文件，跳过。")
        return None, False

    data_dir = os.path.join(dwi_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    cmd = (
        f"cp {os.path.join(dwi_dir, bval)} {data_dir}/bvals && "
        f"cp {os.path.join(dwi_dir, bvec)} {data_dir}/bvecs && "
        f"cp {os.path.join(dwi_dir, dwi)} {data_dir}/data.nii.gz && "
        f"cp {os.path.join(dwi_dir, mask)} {data_dir}/nodif_brain_mask.nii.gz"
    )
    if use_t1:
        t1_src = os.path.join(dwi_dir, "T1.nii.gz")
        if os.path.isfile(t1_src):
            cmd += f" && cp {t1_src} {data_dir}/T1_brain.nii.gz"
        else:
            print(f"[WARN] {subject_id} 未找到 T1.nii.gz，跳过 T1 复制。")

    run_command(cmd)
    return dwi_dir, True

def prepare_subject_data_xuanwu(dataset_directory: str,
                                subject_id: str,
                                use_t1: bool = False):
    """
    搜索 Xuanwu 目录下 subject_id 子目录并准备 data 子目录：
      - 复制 data.nii.gz → data/data.nii.gz
      - 复制 dti.bval   → data/bvals
      - 复制 dti.bvec   → data/bvecs
      - 复制 nodif_brain.nii.gz      → data/nodif_brain.nii.gz
      - 复制 nodif_brain_mask.nii.gz → data/nodif_brain_mask.nii.gz
    如果 use_t1=True，会在 subject_dir 下寻找 T1.nii.gz，
    并复制为 data/T1.nii.gz。
    返回 (subject_dir, True) 或 (None, False)。
    """
    subject_dir = os.path.join(dataset_directory, subject_id)
    if not os.path.isdir(subject_dir):
        print(f"[WARN] 未找到被试目录：{subject_dir}，跳过。")
        return None, False

    # 列出当前目录下所有文件
    files = os.listdir(subject_dir)
    # 必需文件
    data_file = "data.nii.gz" if "data.nii.gz" in files else None
    bval      = "dti.bval"    if "dti.bval"    in files else None
    bvec      = "dti.bvec"    if "dti.bvec"    in files else None
    nodif     = "nodif_brain.nii.gz"      if "nodif_brain.nii.gz"      in files else None
    mask      = "nodif_brain_mask.nii.gz" if "nodif_brain_mask.nii.gz" in files else None

    if not all([data_file, bval, bvec, nodif, mask]):
        print(f"[WARN] {subject_id} 缺少必需文件，跳过。")
        return None, False

    # 创建 data 子目录
    data_dir = os.path.join(subject_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 构造复制命令
    cmd = (
        f"cp {os.path.join(subject_dir, data_file)} {data_dir}/data.nii.gz && "
        f"cp {os.path.join(subject_dir, bval)}      {data_dir}/bvals     && "
        f"cp {os.path.join(subject_dir, bvec)}      {data_dir}/bvecs     && "
        f"cp {os.path.join(subject_dir, nodif)}     {data_dir}/nodif_brain.nii.gz      && "
        f"cp {os.path.join(subject_dir, mask)}      {data_dir}/nodif_brain_mask.nii.gz"
    )

    # 可选 T1
    if use_t1:
        t1_src = os.path.join(subject_dir, "T1.nii.gz")
        if os.path.isfile(t1_src):
            cmd += f" && cp {t1_src} {data_dir}/T1_brain.nii.gz"
        else:
            print(f"[WARN] {subject_id} 未找到 T1.nii.gz，跳过 T1 复制。")

    run_command(cmd)
    return subject_dir, True


def run_bedpostx(data_directory: str, bedpostx_script: str):
    """对单个被试调用 bedpostx 脚本。"""
    run_command(f"python {bedpostx_script} --data_path {data_directory}")

def run_roi_step(
    data_directory: str,
    script_path: str,
    roi_dir: str=None,
    roi_name: str=None,
    use_t1: bool=False
):
    
    cmd = f"python {script_path} --data_path {data_directory} --roi_dir {roi_dir} --roi_name {roi_name}"
    if use_t1 and roi_name.startswith(('2_','6_')):
        cmd += " --use_t1"
    run_command(cmd)


def run_group_analysis(subject_paths: list, group_refer_script: str, base_directory: str, roi_name: str=None):
    subjects = ",".join(subject_paths)
    run_command(f"python {group_refer_script} --base_dir {base_directory}  --subject_data {subjects} --roi_name {roi_name}")


def process_single_roi(
    base_directory: str,
    use_t1: bool,
    roi_name: str,
    roi_dir: str,
    subject_paths: list,
    registration_script: str,
    roi_probtrack_script: str,
    roi_calc_matrix_script: str,
    roi_parcellation_script: str,
    roi_toMNI_script: str,
    group_refer_script: str,
):
    """处理单个 ROI：步骤2–7，步骤6保留不动。"""
    print(f"[INFO] ==== 开始处理 ROI {roi_name} ====")

    with ProcessPoolExecutor(max_workers=NJOBS) as ex:
        # 步骤 2: Registration
        
        print(f"[INFO] Step 2: Registration for ROI {roi_name}")
        futures = [ex.submit(
            run_roi_step, p, registration_script, roi_dir, roi_name, use_t1
        ) for p in subject_paths]
        for f in as_completed(futures):
            try: f.result()
            except Exception as e: print(f"[ERROR] Registration ROI {roi_name}: {e}")

        # 步骤 3: Probtrack
        print(f"[INFO] Step 3: Probtrack for ROI {roi_name}")
        futures = [ex.submit(run_roi_step, p, roi_probtrack_script, roi_dir, roi_name) for p in subject_paths]
        for f in as_completed(futures):
            try: f.result()
            except Exception as e: print(f"[ERROR] Probtrack ROI {roi_name}: {e}")
        
        # 步骤 4: CalcMatrix
        print(f"[INFO] Step 4: CalcMatrix for ROI {roi_name}")
        futures = [ex.submit(run_roi_step, p,  roi_calc_matrix_script, roi_dir, roi_name) for p in subject_paths]
        for f in as_completed(futures):
            try: f.result()
            except Exception as e: print(f"[ERROR] CalcMatrix ROI {roi_name}: {e}")

        # 步骤 5: Parcellation
        print(f"[INFO] Step 5: Parcellation for ROI {roi_name}")
        futures = [ex.submit(run_roi_step, p,  roi_parcellation_script, roi_dir, roi_name) for p in subject_paths]
        for f in as_completed(futures):
            try: f.result()
            except Exception as e: print(f"[ERROR] Parcellation ROI {roi_name}: {e}")

        # 步骤 6: ROI-to-MNI（保留原逻辑）
        print(f"[INFO] Step 6: ROI-to-MNI for ROI {roi_name}")
        futures = [ex.submit(run_roi_step, p, roi_toMNI_script, roi_dir, roi_name, use_t1) for p in subject_paths]
        for f in as_completed(futures):
            try: f.result()
            except Exception as e: print(f"[ERROR] ROI-to-MNI ROI {e}")

    # 步骤 7: Group-level analysis
    print(f"[INFO] Step 7: Group-level analysis for ROI")
    run_group_analysis(subject_paths, group_refer_script, base_directory, roi_name)

    print(f"[INFO] ==== 完成处理 ROI {roi_name} ====")


def main():
    parser = argparse.ArgumentParser(description="批量执行 DWI/ROI 处理和群体分析")
    parser.add_argument("--base_dir", default="/data2/mayupeng/BNU", help="Base directory path")
    parser.add_argument("--datasets", nargs="+", default=["Agility_sample"], help="数据集列表")
    parser.add_argument("--roi_dir", default="/data2/mayupeng/BNU/ROI", help="标准空间 ROI 文件夹")
    parser.add_argument("--roi_list_file", default="/data2/mayupeng/BNU/JHU50.txt", help="包含 50 个 ROI 名称的文件，每行一个名称")
    parser.add_argument("--bedpostx_script", default="/data2/mayupeng/BNU/1_bedpostx.py", help="bedpostx 脚本路径")
    parser.add_argument("--registration_script", default="/data2/mayupeng/BNU/2_registration.py", help="步骤二脚本路径")
    parser.add_argument("--roi_probtrack_script", default="/data2/mayupeng/BNU/3_ROI_probtrack.py", help="Probtrack 脚本路径")
    parser.add_argument("--roi_calc_matrix_script", default="/data2/mayupeng/BNU/4_ROI_calc_matrix.py", help="CalcMatrix 脚本路径")
    parser.add_argument("--roi_parcellation_script", default="/data2/mayupeng/BNU/5_ROI_parcellation.py", help="Parcellation 脚本路径")
    parser.add_argument("--roi_toMNI_script", default="/data2/mayupeng/BNU/6_ROI_toMNI.py", help="ROI-to-MNI 脚本路径")
    parser.add_argument("--group_refer_script", default="/data2/mayupeng/BNU/7_group_refer.py", help="Group-level analysis 脚本路径")
    parser.add_argument("--use_t1", action="store_true", help="启用 T1 参与 ROI-to-MNI 注册")
    args = parser.parse_args()
    
    data_paths_file = os.path.join(args.base_dir, "group_data_paths.txt")
    with open(data_paths_file, "w") as f_out:
        pass
    
    for ds in args.datasets:
        base = os.path.join(args.base_dir, ds)
        if not os.path.isdir(base):
            print(f"[ERROR] 找不到目录: {base}")
            continue
        for subj in os.listdir(base):
            data_path, ok = prepare_subject_data_xuanwu(base, subj, args.use_t1)
            if not ok:
                continue
            # 记录 subject 数据路径
            with open(data_paths_file, "a") as f_out:
                f_out.write(f"{data_path}\n")
            print(f"[INFO] Running bedpostx for subject {subj}")
            run_bedpostx(data_path, args.bedpostx_script)

    # 读取 ROI 名称列表
    with open(args.roi_list_file) as f:
        roi_names = [l.strip() for l in f if l.strip()]
    
    with open(data_paths_file, "r") as f:
        subject_data = [line.strip() for line in f if line.strip()]
        
    for idx, roi_name in enumerate(roi_names, start=1):
        process_single_roi(
            args.base_dir,
            args.use_t1,
            roi_name, args.roi_dir, subject_data,
            args.registration_script, 
            args.roi_probtrack_script,
            args.roi_calc_matrix_script,
            args.roi_parcellation_script,
            args.roi_toMNI_script,
            args.group_refer_script
        )

if __name__ == "__main__":
    main()
