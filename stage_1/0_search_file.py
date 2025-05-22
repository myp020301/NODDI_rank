#!/usr/bin/env python3
import os, glob, re, argparse, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed


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

def prepare_subject_data(dataset_directory: str,
                                subject_id: str,
                                use_t1: bool = False):

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

# 顶层定义，保证它是 module-level
def _worker(subject_path, bedpostx_script):
    # … 具体执行逻辑 …
    run_command(f"python {bedpostx_script} --data_path {subject_path} ")

def run_bedpostx(subject_paths, bedpostx_script, njobs):
    with ProcessPoolExecutor(max_workers=njobs) as ex:
        futures = [
            ex.submit(_worker, p, bedpostx_script)
            for p in subject_paths
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[ERROR] bedpostx 失败 ({p}): {e}")


def run_roi_step(data_directory,
                 script_path,
                 roi_dir=None,
                 roi_name=None,
                 use_t1=False,
                 max_clusters=None):
    """封装单个被试脚本的 CLI 调用"""
    cmd = f"python {script_path} --data_path {data_directory}"

    # 2‑6 步统一需要 ROI 信息
    if roi_dir:
        cmd += f" --roi_dir {roi_dir}"
    if roi_name:
        cmd += f" --roi_name {roi_name}"

    step_tag = os.path.basename(script_path).split("_", 1)[0]   # '2','3','4',...
    if step_tag in {"5", "6"} and max_clusters is not None:
        cmd += f" --max_clusters {max_clusters}"

    # registration / ROI‑to‑MNI 可加 --use_t1
    if use_t1 and step_tag in {"2", "6"}:
        cmd += " --use_t1"

    run_command(cmd)


def run_group_analysis(subject_paths,
                       group_refer_script,
                       cluster_relabel_script,
                       calc_mpm_script,
                       postprocess_mpm_script,
                       validation_script,
                       indices_plot_script,
                       base_directory,
                       roi_name,
                       njobs,
                       max_clusters):

    subjects_arg = ",".join(subject_paths)
    scripts = [
        #group_refer_script,      # 7
        #cluster_relabel_script,  # 8
        #calc_mpm_script,         # 9
        #postprocess_mpm_script,  # 10
        validation_script,       # 11
        indices_plot_script      # 12
    ]

    for script in scripts:
        cmd = (
            f"python {script} "
            f"--base_dir {base_directory} "
            f"--roi_name {roi_name} "
            f"--subject_data {subjects_arg} "   
            f"--njobs {njobs} "
            f"--max_clusters {max_clusters}"    
        )
        print(f"[INFO] Run {os.path.basename(script)} for ROI {roi_name}")
        run_command(cmd)

    print(f"[INFO] Steps 7‑12 finished for ROI {roi_name}")

def process_single_roi(base_directory,
                       use_t1,
                       roi_name,
                       roi_dir,
                       subject_paths,
                       registration_script,
                       roi_probtrack_script,
                       roi_calc_matrix_script,
                       roi_parcellation_script,
                       roi_toMNI_script,
                       group_refer_script,
                       cluster_relabel_script,
                       calc_mpm_script,
                       postprocess_mpm_script,
                       validation_script,
                       indices_plot_script,
                       njobs,
                       max_clusters):

    print(f"\n[INFO] ==== 开始处理 ROI {roi_name} ====")

    # ───────────── 2‑6 步：按脚本顺序，脚本内对被试并行 ─────────────
    
    per_subj_scripts = [
        registration_script,     # 2
        roi_probtrack_script,    # 3
        roi_calc_matrix_script,  # 4
        roi_parcellation_script, # 5
        roi_toMNI_script         # 6
    ]

    for script in per_subj_scripts:
        step_tag = os.path.basename(script).split("_", 1)[0]
        print(f"[INFO] Step {step_tag}: {os.path.basename(script)}  (ROI {roi_name})")

        with ProcessPoolExecutor(max_workers=njobs) as ex:
            futures = [
                ex.submit(
                    run_roi_step,
                    subj_path,
                    script,
                    roi_dir,
                    roi_name,
                    use_t1,          
                    max_clusters
                )
                for subj_path in subject_paths
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"[ERROR] {os.path.basename(script)} ROI {roi_name}: {e}")
    
    # ───────────── 7‑12 步：群组流程 ─────────────
    run_group_analysis(
        subject_paths,
        group_refer_script,
        cluster_relabel_script,
        calc_mpm_script,
        postprocess_mpm_script,
        validation_script,
        indices_plot_script,
        base_directory,
        roi_name,
        njobs,
        max_clusters
    )
    
    print(f"[INFO] ==== 完成 ROI {roi_name} ====")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量执行 DWI / ROI 全流程，包括 12 步指标绘图"
    )
    parser.add_argument("--base_dir", default="/data2/mayupeng/BNU")
    parser.add_argument("--datasets", nargs="+", default=["Agility_sample"])
    parser.add_argument("--roi_dir", default="/data2/mayupeng/BNU/ROI")
    parser.add_argument("--roi_list_file", default="/data2/mayupeng/BNU/JHU50.txt")
    parser.add_argument("--max_clusters", type=int, default=5)          
    parser.add_argument("--use_t1", action="store_true")
    parser.add_argument("--njobs", type=int, default=4)

    # 脚本路径
    parser.add_argument("--bedpostx_script",        default="/data2/mayupeng/BNU/1_bedpostx.py")
    parser.add_argument("--registration_script",    default="/data2/mayupeng/BNU/2_registration.py")
    parser.add_argument("--roi_probtrack_script",   default="/data2/mayupeng/BNU/3_ROI_probtrack.py")
    parser.add_argument("--roi_calc_matrix_script", default="/data2/mayupeng/BNU/4_ROI_calc_matrix.py")
    parser.add_argument("--roi_parcellation_script",default="/data2/mayupeng/BNU/5_ROI_parcellation.py")
    parser.add_argument("--roi_toMNI_script",       default="/data2/mayupeng/BNU/6_ROI_toMNI.py")
    parser.add_argument("--group_refer_script",     default="/data2/mayupeng/BNU/7_group_refer.py")
    parser.add_argument("--cluster_relabel_script", default="/data2/mayupeng/BNU/8_cluster_relabel.py")
    parser.add_argument("--calc_mpm_script",        default="/data2/mayupeng/BNU/9_calc_mpm.py")
    parser.add_argument("--postprocess_mpm_script", default="/data2/mayupeng/BNU/10_postprocess_mpm.py")
    parser.add_argument("--validation_script",      default="/data2/mayupeng/BNU/11_validation.py")
    parser.add_argument("--indices_plot_script",    default="/data2/mayupeng/BNU/12_indices_plot.py")
    args = parser.parse_args()
    
    # ── 预处理：收集所有被试路径 ───────────────────────────────
    data_paths_file = os.path.join(args.base_dir, "group_data_paths.txt")
    
    open(data_paths_file, "w").close()
    
    all_subject_paths = []
    for dataset in args.datasets:
        ds_dir = os.path.join(args.base_dir, dataset)
        if not os.path.isdir(ds_dir):
            print(f"[WARN] 跳过不存在的数据集 {ds_dir}")
            continue
        for subj in os.listdir(ds_dir):
            path, ok = prepare_subject_data(ds_dir, subj, args.use_t1)
            if ok:
                all_subject_paths.append(path)
                with open(data_paths_file, "a") as f_out:
                    f_out.write(path + "\n")
                    
    with open(data_paths_file) as f:
        all_subject_paths = [line.strip() for line in f if line.strip()]
    run_bedpostx(all_subject_paths, args.bedpostx_script, args.njobs)

    # 读取 ROI 列表
    with open(args.roi_list_file) as f:
        roi_names = [l.strip() for l in f if l.strip()]

    # 主循环：逐 ROI
    for roi in roi_names[0:1]:
        process_single_roi(
            args.base_dir,
            args.use_t1,
            roi,
            args.roi_dir,
            all_subject_paths,
            args.registration_script,
            args.roi_probtrack_script,
            args.roi_calc_matrix_script,
            args.roi_parcellation_script,
            args.roi_toMNI_script,
            args.group_refer_script,
            args.cluster_relabel_script,
            args.calc_mpm_script,
            args.postprocess_mpm_script,
            args.validation_script,
            args.indices_plot_script,      
            args.njobs,
            args.max_clusters              
        )