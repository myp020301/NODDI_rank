#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import argparse
from munkres import Munkres
from sklearn.cluster import KMeans
def sc3(k, W):
    """
    Python 版 sc3：将任何 np.matrix 转为 ndarray，并用 numpy.linalg.eigh 做全谱分解。
    
    参数:
      k: 聚类簇数
      W: 相似度矩阵，支持 np.ndarray 或 np.matrix
    
    返回:
      labels: shape = (n,), 值为 0..k-1
    """
    # 0) 如果是 np.matrix，就转成 np.ndarray
    W = np.asarray(W, dtype=np.float64)
    
    # 1) 打印输入形状
    n, m = W.shape
    
    # 2) 计算度矩阵和拉普拉斯
    degs = W.sum(axis=1)
    D = np.diag(degs)
    L = D - W
    
    # 3) 处理零度，避免除零
    eps = np.finfo(float).eps
    degs[degs == 0] = eps
    
    # 4) 构造对称归一化拉普拉斯 L_sym
    inv_sqrt = 1.0 / np.sqrt(degs)
    D_sqrt = np.diag(inv_sqrt)
    L_sym = D_sqrt @ L @ D_sqrt
    L_sym = np.asarray(L_sym)
    
    # 5) 全谱分解（eigh 只接受 ndarray）
    vals, vecs = np.linalg.eigh(L_sym)
    # 6) 选取第一个大于阈值 tol_eig 的特征值开始的 k 列
    tol_eig = 1e-8
    idx = np.where(vals > tol_eig)[0]
    start = int(idx[0]) if idx.size > 0 else 0
    print(vals[start:start+k])
    U = vecs[:, start:start+k]
    
    # 7) 行归一化
    row_norms = np.linalg.norm(U, axis=1)
    U_norm = U / row_norms[:, np.newaxis]
    
    # 8) 最后做 KMeans
    labels = KMeans(n_clusters=k, n_init=1000, random_state=0).fit_predict(U_norm)
    
    return labels


def group_refer(base_dir, roi, subject_paths, max_clusters, method, group_threshold):
    """
    对指定ROI进行群体统计分析，处理所有 subject 在标准空间下的分割结果。
    """
    if isinstance(subject_paths, str):
        subject_paths = subject_paths.split(",")
    
    num_subjects = len(subject_paths)
    real_thresh = np.finfo(float).eps if group_threshold == 0 else group_threshold

    # 使用第一个 subject 的 seed_2 结果作为模板
    first_sub = subject_paths[0]
    template_file = os.path.join(first_sub, "data", "probtrack_old",
                                 f"parcellation_{method}_MNI", roi, "seed_2.nii.gz")
    if not os.path.exists(template_file):
        print(f"[ERROR] File not found: {template_file}")
        return
    nii_template = nib.load(template_file)
    template_img = nii_template.get_fdata()

    sum_img = np.zeros(template_img.shape, dtype=np.float64)

    for sub in subject_paths:
        file_path = os.path.join(sub, "data", "probtrack_old",
                                 f"parcellation_{method}_MNI", roi, "seed_2.nii.gz")
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found for subject {sub}: {file_path}")
            continue
        nii_sub = nib.load(file_path)
        data = nii_sub.get_fdata()
        data = np.nan_to_num(data, nan=0)
        data = (data > 0).astype(np.float64)
        sum_img += data

    # 打印sum_img信息
    print(f"Size of sum_img: {sum_img.shape}")
    print(f"First few values of sum_img: {sum_img.ravel()[:10]}")

    # 创建群体掩码
    group_mask = sum_img.copy()
    thresh_val = real_thresh * num_subjects
    group_mask[group_mask < thresh_val] = 0
    group_mask[group_mask >= thresh_val] = 1


    group_roi_dir = os.path.join(base_dir, "Group_xuanwu", roi)
    os.makedirs(group_roi_dir, exist_ok=True)
    mask_file = os.path.join(group_roi_dir, f"{roi}_roimask_thr{int(real_thresh*100)}.nii.gz")
    mask_nii = nib.Nifti1Image(group_mask, affine=nii_template.affine, header=nii_template.header)
    nib.save(mask_nii, mask_file)
    print(f"[INFO] Saved group ROI mask to {mask_file}")

    flat_img = sum_img.ravel()
    roi_idx = np.where(flat_img >= thresh_val)[0]
    num_voxels = len(roi_idx)
    if num_voxels == 0:
        print(f"[WARNING] No voxels meet the threshold in ROI {roi}")
        return

    # 对聚类数从2到 max_clusters 进行群体分群
    for clus in range(2, max_clusters + 1):
        print(f"[INFO] ROI {roi} cluster {clus} is running...")
        co_occur = np.zeros((num_voxels, num_voxels), dtype=np.float64)
        for sub in subject_paths:
            file_path = os.path.join(sub, "data", "probtrack_old",
                                     f"parcellation_{method}_MNI", roi, f"seed_{clus}.nii.gz")
            if not os.path.exists(file_path):
                print(f"[WARNING] File not found for subject {sub}: {file_path}")
                continue
            nii_sub = nib.load(file_path)
            data = nii_sub.get_fdata()
            data = np.nan_to_num(data, nan=0).astype(np.float64)
            flat_data = data.ravel()
            
            for lbl in range(1, clus + 1):
                cluster_vox = np.where(flat_data == lbl)[0]
                common_idx = np.where(np.isin(roi_idx, cluster_vox))[0]
                if common_idx.size > 0:
                    temp_mat = np.zeros((num_voxels, num_voxels), dtype=np.float64)
                    temp_mat[np.ix_(common_idx, common_idx)] = 1
                    co_occur += temp_mat

        np.fill_diagonal(co_occur, 0)

        # 聚类
        try:
            labels = sc3(clus, co_occur)
            labels = labels + 1
        except Exception as e:
            print(f"[ERROR] Clustering failed for ROI {roi} cluster {clus}: {e}")
            continue
        cluster_flat = np.zeros(flat_img.shape, dtype=np.int32)
        if labels.shape[0] != num_voxels:
            print(f"[WARNING] Voxel number mismatch: labels {labels.shape[0]} vs roi_idx {num_voxels}")
        else:
            cluster_flat[roi_idx] = labels
        cluster_img = cluster_flat.reshape(template_img.shape)

        # 保存聚类结果
        output_file = os.path.join(group_roi_dir, f"{roi}_{clus}_{int(real_thresh*100)}_group.nii.gz")
        group_nii = nib.Nifti1Image(cluster_img, affine=nii_template.affine, header=nii_template.header)
        nib.save(group_nii, output_file)
        print(f"[INFO] Saved clustering result to {output_file}")
        print(f"[INFO] ROI {roi} cluster {clus} Done !!")




def symmetry_group(base_dir, region_l, region_r, max_clusters, group_thresh):
    """
    对于 bilateral ROI（左侧 region_l 和右侧 region_r），使用匈牙利算法对右侧的分割结果进行对称修正，
    使得左右标签统一。
    
    参数：
      - base_dir: 基础目录
      - region_l: 左侧 ROI 名称（应包含 "L"）
      - region_r: 右侧 ROI 名称（将 "L" 替换为 "R"）
      - max_clusters: 最大聚类数（与 group_refer 中相同）
      - group_thresh: 已使用的群体阈值（real_thresh）
      - num_subjects: subject 数量，用于文件命名（可选）
      
    实现思路：
      对每个聚类数，从 group 目录中加载对应的左侧和右侧结果，
      对右侧结果进行镜像处理（沿 x 轴翻转），计算左右之间的重叠矩阵，
      利用 Munkres 算法得到最佳匹配，将右侧结果的标签重新映射，
      最后保存修正后的右侧结果（先将原文件重命名为 .old）。
    """
    group_dir_l = os.path.join(base_dir, "Group_xuanwu", region_l)
    group_dir_r = os.path.join(base_dir, "Group_xuanwu", region_r)
    # 若右侧目录不存在，则不处理
    if not os.path.isdir(group_dir_r):
        print(f"[WARNING] Right-side group directory not found: {group_dir_r}")
        return

    for clus in range(2, max_clusters + 1):
        file_l = os.path.join(group_dir_l, f"{region_l}_{clus}_{int(group_thresh*100)}_group.nii.gz")
        print(file_l)
        file_r = os.path.join(group_dir_r, f"{region_r}_{clus}_{int(group_thresh*100)}_group.nii.gz")
        if not os.path.exists(file_l) or not os.path.exists(file_r):
            print(f"[WARNING] Symmetry files for cluster {clus} not found (L or R).")
            continue

        nii_l = nib.load(file_l)
        img_l = nii_l.get_fdata()
        img_l = np.nan_to_num(img_l, nan=0)
        nii_r = nib.load(file_r)
        img_r = nii_r.get_fdata()
        img_r = np.nan_to_num(img_r, nan=0)

        # 对右侧图像进行镜像（沿 x 轴翻转）
        xr, yr, zr = img_r.shape
        img_r_mirror = np.zeros_like(img_r)
        for x in range(xr):
            img_r_mirror[x, :, :] = img_r[xr - x - 1, :, :]

        # 构造 overlay 矩阵
        overlay = np.zeros((clus, clus), dtype=np.float64)
        for ki in range(1, clus+1):
            for kj in range(1, clus+1):
                mask_l = (img_l == ki)
                mask_r_mirror = (img_r_mirror == kj)
                overlay[ki-1, kj-1] = np.sum(mask_l & mask_r_mirror)
        
        # 使用 Munkres 算法对 -overlay 求最小匹配，即最大重叠
        m = Munkres()
        cost_matrix = (-overlay).tolist()  # 先对 overlay 取负，再转换为列表
        # 取负后求最小匹配
        indexes = m.compute(cost_matrix)  # 返回 [(row, col), ...]
        # 根据结果建立映射：右侧原标签 -> 左侧标签
        mapping = {}
        for row, col in indexes:
            mapping[col + 1] = row + 1

        # 根据 mapping 重新分配右侧标签
        img_r_new = img_r.copy()
        for r_label, l_label in mapping.items():
            img_r_new[img_r == r_label] = l_label

        # 重命名原右侧文件并保存新结果
        old_file_r = file_r + ".old"
        os.rename(file_r, old_file_r)
        nii_r_new = nib.Nifti1Image(img_r_new, affine=nii_r.affine, header=nii_r.header)
        nib.save(nii_r_new, file_r)
        print(f"[INFO] Symmetrized cluster {clus} for ROI {region_r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--subject_data", required=True)
    parser.add_argument("--max_clusters", type=int, default=6)
    parser.add_argument("--method", default="sc", choices=["sc", "kmeans", "simlr"])
    parser.add_argument("--group_threshold", type=float, default=0.25)
    args = parser.parse_args()

    # 读取 subject 数据列表（支持逗号分隔的字符串或文件内容）
    if os.path.isfile(args.subject_data):
        with open(args.subject_data, "r") as f:
            subjects = [line.strip() for line in f if line.strip()]
    else:
        subjects = args.subject_data.split(",")

    region_name = args.roi_name
    group_refer(args.base_dir, region_name, subjects, args.max_clusters, args.method, args.group_threshold)
    
    # 如果 roi 名称中含有 "L"，则需要对左右标签进行对称统一 
    if region_name.endswith("_L"):
        # 此处假定对称文件的命名规则为：
        # 左侧文件： f"{roi}_L_{clus}_{int(real_thresh*100)}_group.nii.gz"
        # 右侧文件： f"{region_R}_{clus}_{int(real_thresh*100)}_group.nii.gz"
        # 其中右侧 ROI 名称由 roi 中的 "L" 替换为 "R"
        region_r = region_name[:-1] + "R"
        symmetry_group(args.base_dir, region_name, region_r, args.max_clusters, args.group_threshold)
