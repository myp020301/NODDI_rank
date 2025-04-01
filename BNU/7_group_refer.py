#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from sklearn.cluster import SpectralClustering
import argparse

BRAIN_REGIONS = [
    "MCP", "PCT", "GCC", "BCC", "SCC", "Fx",
    "CST_R", "CST_L", "ML_R", "ML_L", "ICP_R", "ICP_L",
    "SCP_R", "SCP_L", "CP_R", "CP_L", "ALIC_R", "ALIC_L",
    "PLIC_R", "PLIC_L", "RLIC_R", "RLIC_L", "ACR_R", "ACR_L",
    "SCR_R", "SCR_L", "PCR_R", "PCR_L", "PTR_R", "PTR_L",
    "SS_R", "SS_L", "EC_R", "EC_L", "CGC_R", "CGC_L",
    "CGH_R", "CGH_L", "Fx/ST_R", "Fx/ST_L", "SLF_R", "SLF_L",
    "SFOF_R", "SFOF_L", "IFOF_R", "IFOF_L", "UF_R", "UF_L",
    "TAP_R", "TAP_L"
]

def group_refer(base_dir, region, subject_paths, max_clusters, method, group_threshold):
    """
    对指定ROI进行群体统计分析，处理所有subject在标准空间下的分割结果。
    
    参数：
      - base_dir: 基础目录（例如 BNU 根目录）
      - region: 当前处理的ROI（从BRAIN_REGIONS中索引得到的名称）
      - subject_paths: 所有subject数据路径列表
      - max_clusters: 最大聚类数
      - method: 使用的分割方法（例如"sc"）
      - group_threshold: 群体统计阈值
    """
    if isinstance(subject_paths, str):
       subject_paths = subject_paths.split(",")

    num_subjects = len(subject_paths)
    real_threshold = np.finfo(float).eps if group_threshold == 0 else group_threshold
    
    
    # 使用第一个subject的seed_2结果作为模板
    first_sub = subject_paths[0]
    template_file = os.path.join(first_sub, "data", "probtrack_old",
                                 f"parcellation_{method}_MNI", region, "seed_2.nii.gz")
    
    if not os.path.exists(template_file):
        print(f"[ERROR] File not found: {template_file}")
        return
    nii_template = nib.load(template_file)
    template_img = nii_template.get_fdata()
    sum_img = np.zeros(template_img.shape, dtype=np.float64)

    for sub in subject_paths:
        file_path = os.path.join(sub, "data", "probtrack_old",
                                 f"parcellation_{method}_MNI", region, "seed_2.nii.gz")
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found for subject {sub}: {file_path}")
            continue
        nii_sub = nib.load(file_path)
        data = nii_sub.get_fdata()
        data = np.nan_to_num(data, nan=0)
        data = (data > 0).astype(np.float64)
        sum_img += data

    # 构建群体ROI mask
    group_mask = sum_img.copy()
    threshold_val = real_threshold * num_subjects
    group_mask[group_mask < threshold_val] = 0
    group_mask[group_mask >= threshold_val] = 1

    group_roi_dir = os.path.join(base_dir, "Group", region)
    os.makedirs(group_roi_dir, exist_ok=True)
    group_mask_file = os.path.join(group_roi_dir, f"{region}_roimask_thr{int(real_threshold*100)}.nii.gz")
    group_mask_nii = nib.Nifti1Image(group_mask, affine=nii_template.affine, header=nii_template.header)
    nib.save(group_mask_nii, group_mask_file)
    print(f"[INFO] Saved group ROI mask to {group_mask_file}")

    flat_img = sum_img.ravel()
    roi_idx = np.where(flat_img >= threshold_val)[0]
    num_roi_voxels = len(roi_idx)
    if num_roi_voxels == 0:
        print(f"[WARNING] No voxels meet the threshold in ROI {region}")
        return

    # 对聚类数从2到max_clusters进行群体分群
    for cluster_num in range(2, max_clusters + 1):
        print(f"[INFO] ROI {region} cluster number {cluster_num} is running...")
        co_occurrence = np.zeros((num_roi_voxels, num_roi_voxels), dtype=np.float64)
        for sub in subject_paths:
            file_path = os.path.join(sub, "data", "probtrack_old",
                                     f"parcellation_{method}_MNI", region, f"seed_{cluster_num}.nii.gz")
            if not os.path.exists(file_path):
                print(f"[WARNING] File not found for subject {sub}: {file_path}")
                continue
            nii_sub = nib.load(file_path)
            data = nii_sub.get_fdata()
            data = np.nan_to_num(data, nan=0).astype(np.float64)
            flat_data = data.ravel()
            for label in range(1, cluster_num + 1):
                cluster_vox = np.where(flat_data == label)[0]
                common_idx = np.where(np.isin(roi_idx, cluster_vox))[0]
                if common_idx.size > 0:
                    clust_mat = np.zeros((num_roi_voxels, num_roi_voxels), dtype=np.float64)
                    clust_mat[np.ix_(common_idx, common_idx)] = 1
                    co_occurrence += clust_mat
        np.fill_diagonal(co_occurrence, 0)
        try:
            sc = SpectralClustering(
                n_clusters=cluster_num,
                n_init=300,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=0,
                eigen_solver='arpack'
            )
            labels = sc.fit_predict(co_occurrence)
            labels = labels + 1
            
        except Exception as e:
            print(f"[ERROR] Clustering failed for ROI {region} cluster {cluster_num}: {e}")
            continue
        cluster_img_flat = np.zeros(flat_img.shape, dtype=np.int32)
        if labels.shape[0] != num_roi_voxels:
            print(f"[WARNING] Voxel number mismatch: labels {labels.shape[0]} vs roi_idx {num_roi_voxels}")
        else:
            cluster_img_flat[roi_idx] = labels
        cluster_img = cluster_img_flat.reshape(template_img.shape)
        output_file = os.path.join(group_roi_dir, f"{region}_{cluster_num}_{int(real_threshold*100)}_group.nii.gz")
        group_nii = nib.Nifti1Image(cluster_img, affine=nii_template.affine, header=nii_template.header)
        nib.save(group_nii, output_file)
        print(f"[INFO] Saved clustering result to {output_file}")
        print(f"[INFO] ROI {region} cluster {cluster_num} Done !!")
    
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Base directory (e.g. BNU root directory)")
    parser.add_argument("--roi", type=int, required=True,
                        help="ROI index (1-50) corresponding to BRAIN_REGIONS")
    parser.add_argument("--subject_data", required=True,
                        help="List of subject data paths (one per line)")
    parser.add_argument("--max_clusters", type=int, default=12,
                        help="Maximum number of clusters (default: 12)")
    parser.add_argument("--method", default="sc", choices=["sc", "kmeans", "simlr"],
                        help="Segmentation method (default: sc)")
    parser.add_argument("--group_threshold", type=float, default=0.25,
                        help="Group threshold (default: 0.25)")
    args = parser.parse_args()

    
    region = BRAIN_REGIONS[args.roi - 1]
    print(args.subject_data)
    group_refer(args.base_dir, region, args.subject_data, args.max_clusters, args.method, args.group_threshold)
