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

def run_bedpostx(data_directories, bedpostx_script, njobs):
    def _worker(data_dir):
        run_command(f"python {bedpostx_script} --data_path {data_dir}")

    print(f"[INFO] 开始并行运行 bedpostx：{len(data_directories)} 个被试，进程数={njobs}")
    with ProcessPoolExecutor(max_workers=njobs) as ex:
        futures = {ex.submit(_worker, d): d for d in data_directories}
        for future in as_completed(futures):
            data_dir = futures[future]
            try:
                future.result()
                print(f"[INFO] 完成 bedpostx: {data_dir}")
            except Exception as e:
                print(f"[ERROR] bedpostx 失败 ({data_dir}): {e}")

# Steps 2-6: per-ROI per-subject processing
def run_roi_step(data_directory, script_path, roi_dir=None, roi_name=None, use_t1=False):
    cmd = f"python {script_path} --data_path {data_directory} --roi_dir {roi_dir} --roi_name {roi_name}"
    script_filename = os.path.basename(script_path)
    if use_t1 and script_filename.startswith(('2_','6_')):
        cmd += " --use_t1"
    run_command(cmd)

def run_group_analysis(subject_paths, group_refer_script, cluster_relabel_script, calc_mpm_script,
                       postprocess_mpm_script, base_directory, roi_name, njobs):
    subjects = ",".join(subject_paths)

    # Step 7: Group refer
    '''
    run_command(
        f"python {group_refer_script} --base_dir {base_directory} --subject_data {subjects} --roi_name {roi_name}"
    )
    
    # Step 8: Cluster relabel
    
    run_command(
        f"python {cluster_relabel_script} --base_dir {base_directory} --subject_data {subjects} --roi_name {roi_name} --njobs {njobs}"
    )
    
    # Step 9: Calc MPM, using defaults for other parameters
    run_command(
        f"python {calc_mpm_script} --base_dir {base_directory} --roi_name {roi_name} --subject_data {subjects} --njobs {njobs}"
    )
    '''
    # Step 10: Post-process MPM
    print(f"[INFO] Step 10: Post-process MPM for ROI {roi_name}")
    run_command(
        f"python {postprocess_mpm_script} --base_dir {base_directory} --roi_name {roi_name} --subject_data {subjects} --njobs {njobs}"
    )


def process_single_roi(base_directory, use_t1, roi_name, roi_dir, subject_paths,
                       registration_script, roi_probtrack_script, roi_calc_matrix_script,
                       roi_parcellation_script, roi_toMNI_script,
                       group_refer_script, cluster_relabel_script, calc_mpm_script,
                       postprocess_mpm_script, njobs):
    print(f"[INFO] ==== 开始处理 ROI {roi_name} ====")
    
    # Steps 7,8 & 9: Group analysis, relabel, and MPM
    print(f"[INFO] Step 7-10: Group analysis, relabel & MPM for ROI {roi_name}")
    run_group_analysis(subject_paths,
                       group_refer_script,
                       cluster_relabel_script,
                       calc_mpm_script,
                       postprocess_mpm_script,
                       base_directory,
                       roi_name,
                       njobs)
    print(f"[INFO] ==== 完成处理 ROI {roi_name} ====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量执行 DWI/ROI 处理及群组分析，包括 MPM 计算")
    parser.add_argument("--base_dir", default="/data2/mayupeng/BNU", help="Base directory path")
    parser.add_argument("--datasets", nargs='+', default=["Agility_sample"], help="数据集列表")
    parser.add_argument("--roi_dir", default="/data2/mayupeng/BNU/ROI", help="标准空间 ROI 文件夹")
    parser.add_argument("--roi_list_file", default="/data2/mayupeng/BNU/JHU50.txt", help="包含 ROI 名称的文件，每行一个名称")
    parser.add_argument("--bedpostx_script", default="/data2/mayupeng/BNU/1_bedpostx.py")
    parser.add_argument("--registration_script", default="/data2/mayupeng/BNU/2_registration.py")
    parser.add_argument("--roi_probtrack_script", default="/data2/mayupeng/BNU/3_ROI_probtrack.py")
    parser.add_argument("--roi_calc_matrix_script", default="/data2/mayupeng/BNU/4_ROI_calc_matrix.py")
    parser.add_argument("--roi_parcellation_script", default="/data2/mayupeng/BNU/5_ROI_parcellation.py")
    parser.add_argument("--roi_toMNI_script", default="/data2/mayupeng/BNU/6_ROI_toMNI.py")
    parser.add_argument("--group_refer_script", default="/data2/mayupeng/BNU/7_group_refer.py")
    parser.add_argument("--cluster_relabel_script", default="/data2/mayupeng/BNU/8_cluster_relabel.py")
    parser.add_argument("--calc_mpm_script", default="/data2/mayupeng/BNU/9_calc_mpm.py",
                        help="MPM 计算脚本路径")
    parser.add_argument("--postprocess_mpm_script", default="/data2/mayupeng/BNU/10_postprocess_mpm.py",
                        help="MPM后处理脚本路径")
    parser.add_argument("--use_t1", action="store_true", help="启用 T1 注册")
    parser.add_argument("--njobs", type=int, default=5, help="并行作业数量 (default:5)")
    args = parser.parse_args()

    # 准备数据路径文件
    data_paths_file = os.path.join(args.base_dir, "group_data_paths.txt")
    open(data_paths_file, 'w').close()

    # 收集被试路径并执行 BedpostX
    all_subject_paths = []
    for ds in args.datasets:
        ds_dir = os.path.join(args.base_dir, ds)
        if not os.path.isdir(ds_dir):
            print(f"[ERROR] 找不到目录: {ds_dir}")
            continue
        for subj in os.listdir(ds_dir):
            path, ok = prepare_subject_data_xuanwu(ds_dir, subj, args.use_t1)
            if not ok:
                continue
            all_subject_paths.append(path)
    
    # 读取 ROI 列表
    with open(args.roi_list_file) as f:
        roi_names = [l.strip() for l in f if l.strip()]

    # 针对每个 ROI 执行流水线
    for roi in roi_names:
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
            args.njobs
        )
