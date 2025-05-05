#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
import numpy as np
from munkres import Munkres
from concurrent.futures import ProcessPoolExecutor, as_completed

def relabel_subject(base_dir, region, subj, clus, method, group_dir, thresh_int):
    """
    将群体模板下的第 clus 个簇标签，映射到单个被试的第 clus 个分割结果上，并保存。
    """
    tpl_path = os.path.join(
        group_dir,
        f"{region}_{clus}_{thresh_int}_group.nii.gz"
    )
    if not os.path.exists(tpl_path):
        print(f"[WARNING] 模板不存在: {tpl_path}")
        return
    tpl_nii  = nib.load(tpl_path)
    tpl_data = np.nan_to_num(tpl_nii.get_fdata(), nan=0).astype(int)

    indiv_path = os.path.join(
        subj, "data", "probtrack_old",
        f"parcellation_{method}_MNI", region,
        f"seed_{clus}.nii.gz"
    )
    if not os.path.exists(indiv_path):
        print(f"[WARNING] 被试文件不存在: {indiv_path}")
        return
    indiv_nii  = nib.load(indiv_path)
    indiv_data = np.nan_to_num(indiv_nii.get_fdata(), nan=0).astype(int)

    # 计算重叠矩阵
    overlap = np.zeros((clus, clus), dtype=int)
    for i in range(1, clus+1):
        mask_i = (tpl_data == i)
        for j in range(1, clus+1):
            overlap[i-1, j-1] = np.sum(mask_i & (indiv_data == j))

    # 用 Munkres 算法对 -overlap 做最小化匹配，等价于最大化重叠
    m = Munkres()
    cost = (-overlap).tolist()
    assignment = m.compute(cost)
    mapping = { col+1: row+1 for row, col in assignment }

    # 应用映射并保存
    new_data = np.zeros_like(indiv_data)
    for orig, new in mapping.items():
        new_data[indiv_data == orig] = new

    out_path = indiv_path.replace(".nii.gz", "_relabel_group.nii.gz")
    os.rename(indiv_path, indiv_path + ".old")
    nib.save(
        nib.Nifti1Image(new_data, indiv_nii.affine, indiv_nii.header),
        out_path
    )
    print(f"[INFO] 保存重标号结果: {out_path}")


def cluster_relabel(base_dir, region, subject_paths, max_clusters, method, group_threshold, njobs):
    """
    把群体模板中每个簇的标签，映射回每个被试的对应簇上（并行版）。
    参数：
      base_dir       – BNU 根目录
      region         – ROI 名称
      subject_paths       – 被试列表
      max_clusters   – 最大簇数
      method         – 分割方法
      group_threshold– 群体阈值（0–1）
      njobs          – 并行作业数量
    """
    if isinstance(subject_paths, str):
        subject_paths = subject_paths.split(",")

    group_dir  = os.path.join(base_dir, "Group_xuanwu", region)
    thresh_int = int(group_threshold * 100)

    # 收集所有要处理的任务
    tasks = []
    for clus in range(2, max_clusters+1):
        tpl_file = os.path.join(group_dir, f"{region}_{clus}_{thresh_int}_group.nii.gz")
        if not os.path.exists(tpl_file):
            print(f"[WARNING] 跳过簇 {clus}，找不到模板文件 {tpl_file}")
            continue
        for subj in subject_paths:
            tasks.append((base_dir, region, subj, clus, method, group_dir, thresh_int))

    if not tasks:
        print("[INFO] 未找到任何需要重标记的任务，退出。")
        return

    # 并行执行所有 relabel 操作
    with ProcessPoolExecutor(max_workers=njobs) as executor:
        future_to_task = {executor.submit(relabel_subject, *t): t for t in tasks}
        for future in as_completed(future_to_task):
            base_dir, region, subj, clus, method, group_dir, thresh_int = future_to_task[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] 处理 {region} 簇{clus} 被试{subj} 失败: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",        required=True)
    parser.add_argument("--roi_name",        required=True)
    parser.add_argument("--subject_data",    required=True)
    parser.add_argument("--max_clusters",   type=int,   default=6)
    parser.add_argument("--method",         default="sc")
    parser.add_argument("--group_threshold",type=float, default=0.25)
    parser.add_argument("--njobs",          type=int,   default=3)
    args = parser.parse_args()

    if os.path.isfile(args.subject_data):
        with open(args.subject_data) as f:
            subject_paths = [l.strip() for l in f if l.strip()]
    else:
        subject_paths = args.subject_data.split(",")

    cluster_relabel(
        base_dir        = args.base_dir,
        region          = args.roi_name,
        subject_paths        = subject_paths,
        max_clusters    = args.max_clusters,
        method          = args.method,
        group_threshold = args.group_threshold,
        njobs           = args.njobs
    )

if __name__ == "__main__":
    main()
