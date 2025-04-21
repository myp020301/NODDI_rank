#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
import numpy as np
from munkres import Munkres

def relabel_subject(base_dir, region, subj, clus, method, group_dir, thresh_int):
    """
    将群体模板下的第 clus 个簇标签，映射到单个被试的第 clus 个分割结果上，并保存。
    """
    # 群体模板文件
    tpl_path = os.path.join(
        group_dir,
        f"{region}_{clus}_{thresh_int}_group.nii.gz"
    )
    if not os.path.exists(tpl_path):
        print(f"[WARNING] 模板不存在: {tpl_path}")
        return
    tpl_nii  = nib.load(tpl_path)
    tpl_data = np.nan_to_num(tpl_nii.get_fdata(), nan=0).astype(int)

    # 个体分割结果
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
    # 构建标签映射：原标签 j -> 模板标签 i
    mapping = { col+1: row+1 for row, col in assignment }

    # 应用映射
    new_data = np.zeros_like(indiv_data)
    for orig, new in mapping.items():
        new_data[indiv_data == orig] = new

    # 备份并保存
    out_path = indiv_path.replace(".nii.gz", "_relabel_group.nii.gz")
    os.rename(indiv_path, indiv_path + ".old")
    nib.save(
        nib.Nifti1Image(new_data, indiv_nii.affine, indiv_nii.header),
        out_path
    )
    print(f"[INFO] 保存重标号结果: {out_path}")

def cluster_relabel(base_dir, region, subjects, max_clusters, method, group_threshold):
    """
    把群体模板中每个簇的标签，映射回每个被试的对应簇上。
    
    参数：
      base_dir       – BNU 根目录
      region         – ROI 名称（如 "MCP" 或 "FA_L"）
      subjects       – 被试列表，路径列表或逗号分隔字符串
      max_clusters   – 最大簇数
      method         – 分割方法（如 "sc"）
      group_threshold– 群体阈值（0–1）
    """
    # 支持逗号分隔或文件
    if isinstance(subjects, str):
        subjects = subjects.split(",")

    # 群体结果目录
    group_dir = os.path.join(base_dir, "Group_xuanwu", region)
    thresh_int = int(group_threshold * 100)

    for clus in range(2, max_clusters+1):
        tpl_file = os.path.join(group_dir, f"{region}_{clus}_{thresh_int}_group.nii.gz")
        if not os.path.exists(tpl_file):
            print(f"[WARNING] 跳过簇 {clus}，找不到模板文件 {tpl_file}")
            continue
        for subj in subjects:
            relabel_subject(base_dir, region, subj, clus, method, group_dir, thresh_int)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",       required=True,
                        help="BNU 根目录")
    parser.add_argument("--roi_name",       required=True,
                        help="要处理的 ROI 名称，例如 MCP 或 FA_L")
    parser.add_argument("--subject_data",    required=True,
                        help="被试列表文件（每行一个路径）或逗号分隔路径字符串")
    parser.add_argument("--max_clusters",   type=int,   default=12,
                        help="最大簇数 (default: 12)")
    parser.add_argument("--method",         default="sc",
                        help="分割方法, 如 sc (default: sc)")
    parser.add_argument("--group_threshold",type=float, default=0.25,
                        help="群体阈值 (default: 0.25)")
    args = parser.parse_args()

    # 读取被试列表
    if os.path.isfile(args.subject_data):
        with open(args.subject_data) as f:
            subjects = [l.strip() for l in f if l.strip()]
    else:
        subjects = args.subject_data.split(",")

    cluster_relabel(
        base_dir       = args.base_dir,
        region         = args.roi_name,
        subjects       = subjects,
        max_clusters   = args.max_clusters,
        method         = args.method,
        group_threshold= args.group_threshold
    )

if __name__ == "__main__":
    main()
