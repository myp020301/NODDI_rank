#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

def connect6mean(img, i, j, k):
    return (
        img[i-1, j, k] + img[i+1, j, k] +
        img[i, j-1, k] + img[i, j+1, k] +
        img[i, j, k-1] + img[i, j, k+1]
    ) / 6.0

def process_cluster(base_dir, roi, subjects, cl_num, method, mpm_thres, img_shape, affine, header, prob_dir):
    prob_cluster = np.zeros((*img_shape, cl_num), dtype=float)
    sumimg = np.zeros(img_shape, dtype=float)

    # -------- 3.1 汇总所有被试 --------
    for sub in subjects:
        in_path = os.path.join(
            base_dir, sub,
            "data", "probtrack_old",
            f"parcellation_{method}_MNI", roi,
            f"seed_{cl_num}_relabel_group.nii.gz"
        )
        if not os.path.exists(in_path):
            continue

        data = np.nan_to_num(nib.load(in_path).get_fdata(), nan=0).astype(int)
        mask = (data > 0).astype(int)
        sumimg += mask

        for ki in range(1, cl_num + 1):
            prob_cluster[..., ki-1] += (data == ki).astype(float)

    # -------- 3.2 构建 index_mask --------
    thresh = mpm_thres * len(subjects)
    index_mask = sumimg >= thresh

    # -------- 3.3 输出概率图 --------
    for ki in range(1, cl_num + 1):
        prob = np.zeros(img_shape, dtype=float)
        prob[index_mask] = (
            prob_cluster[..., ki-1][index_mask]
            / sumimg[index_mask] * 100
        )
            
        nib.save(
            nib.Nifti1Image(prob, affine, header),
            os.path.join(prob_dir, f"{roi}_{cl_num}_{ki}_prob.nii.gz")
        )

    print(f" <{roi}_{cl_num}> probabilistic maps done!")

    # -------- 3.4 生成 MPM --------
    mpm = np.zeros(img_shape, dtype=int)
    xs, ys, zs = np.where(index_mask)
    internal = (xs > 0) & (xs < img_shape[0]-1) & \
               (ys > 0) & (ys < img_shape[1]-1) & \
               (zs > 0) & (zs < img_shape[2]-1)
    xs, ys, zs = xs[internal], ys[internal], zs[internal]

    for x, y, z in zip(xs, ys, zs):
        total = sumimg[x, y, z]
        if total == 0:
            continue

        prob = prob_cluster[x, y, z, :] / total * 100
        order = np.argsort(-prob)
        a, b = order[0], order[1]

        if prob[a] - prob[b] > 0:  # dominant 分支
            mpm[x, y, z] = a + 1
        else: 
            mean1 = connect6mean(prob_cluster[..., a], x, y, z)
            mean2 = connect6mean(prob_cluster[..., b], x, y, z)

            if mean1 >= mean2:
                mpm[x, y, z] = a + 1
            else:
                mpm[x, y, z] = b + 1

    nib.save(
        nib.Nifti1Image(mpm, affine, header),
        os.path.join(
            prob_dir, f"{roi}_{cl_num}_MPM_thr{int(mpm_thres * 100)}_group.nii.gz"
        )
    )

def calc_mpm_group_xmm(base_dir, roi, subject_paths, max_cl_num, method, mpm_thres, njobs):
    
    if os.path.isfile(subject_paths):
        with open(subject_paths) as f:
            subjects = [s.strip() for s in f if s.strip()]
    else:
        subjects = subject_paths.split(',')
    n_sub = len(subjects)

    prob_dir = os.path.join(base_dir, f"MPM_{n_sub}")
    os.makedirs(prob_dir, exist_ok=True)
    ref_path = os.path.join(
        base_dir, subjects[0],
        "data", "probtrack_old",
        f"parcellation_{method}_MNI", roi,
        "seed_2_relabel_group.nii.gz"
    )
    ref_nii = nib.load(ref_path)
    ref_data = np.nan_to_num(ref_nii.get_fdata(), nan=0).astype(int)
    img_shape = ref_data.shape
    affine, header = ref_nii.affine, ref_nii.header

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        futures = []
        for cl_num in range(2, max_cl_num + 1):
            futures.append(executor.submit(process_cluster, base_dir, roi, subjects, cl_num, method, mpm_thres, img_shape, affine, header, prob_dir))
        
        # 等待并获取所有任务结果
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] 处理簇失败: {e}")
                
    print("所有任务已完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group-level MPM (Python, MATLAB-equivalent)"
    )
    parser.add_argument("--base_dir",     required=True)
    parser.add_argument("--roi_name",     required=True)
    parser.add_argument("--subject_data", required=True)
    parser.add_argument("--max_clusters", type=int, default=6)
    parser.add_argument("--method",       default="sc")
    parser.add_argument("--mpm_thres",    type=float, default=0.25)
    parser.add_argument("--njobs",        type=int,   default=3)
    args = parser.parse_args()

    calc_mpm_group_xmm(
        base_dir   = args.base_dir,
        roi        = args.roi_name,
        subject_paths = args.subject_data,
        max_cl_num = args.max_clusters,
        method     = args.method,
        mpm_thres  = args.mpm_thres,
        njobs      = args.njobs
    )
