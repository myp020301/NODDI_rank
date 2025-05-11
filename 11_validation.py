#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed
from validation_metrics import v_dice, v_nmi, v_cramerv


def connect6mean(img, i, j, k):
    return (
        img[i-1, j, k] + img[i+1, j, k] +
        img[i, j-1, k] + img[i, j+1, k] +
        img[i, j, k-1] + img[i, j, k+1]
    ) / 6.0


def cluster_mpm_validation(base_dir, roi, subjects, method, kc, mpm_thres):
    """
    Generate group MPM for given subjects and cluster count kc,
    with detailed printing of intermediate statistics.
    """
    ref_path = os.path.join(
        base_dir, subjects[0],
        f"data/probtrack_old/parcellation_{method}_MNI/{roi}/seed_{kc}_relabel_group.nii.gz"
    )
    img0 = nib.load(ref_path)
    shape = img0.get_fdata().shape

    sumimg = np.zeros(shape, dtype=float)
    prob_cluster = np.zeros((*shape, kc), dtype=float)

    # Iterate subjects: load, print per-subject counts, accumulate
    for sub in subjects:
        sub_path = os.path.join(
            base_dir, sub,
            f"data/probtrack_old/parcellation_{method}_MNI/{roi}/seed_{kc}_relabel_group.nii.gz"
        )
        data = np.nan_to_num(nib.load(sub_path).get_fdata(), nan=0).astype(int)

        # Accumulate binary mask
        mask_vol = (data > 0).astype(float)
        sumimg += mask_vol

        # Accumulate raw counts for each cluster
        for ki in range(1, kc+1):
            prob_cluster[..., ki-1] += (data == ki).astype(float)


    # Build threshold mask
    thresh = mpm_thres * len(subjects)
    index_mask = sumimg >= thresh
    print(f"Threshold = {thresh:.2f} ({mpm_thres}×{len(subjects)}), voxels after thresh: {index_mask.sum()}")

    # Zero-out probabilities outside mask
    prob_cluster *= index_mask[..., None]

    # Allocate MPM array and perform assignment
    coords = np.column_stack(np.where(index_mask))
    mpm_cluster = np.zeros(shape, dtype=int)

    for x, y, z in coords:
        total = sumimg[x, y, z]
        if total == 0:
            continue

        prob = prob_cluster[x, y, z, :] / total * 100
        order = np.argsort(-prob)
        a, b = order[0], order[1]

        if prob[a] - prob[b] > 0:
            mpm_cluster[x, y, z] = a + 1
        else:
            m1 = connect6mean(prob_cluster[..., a], x, y, z)
            m2 = connect6mean(prob_cluster[..., b], x, y, z)
            chosen = a if m1 >= m2 else b
            mpm_cluster[x, y, z] = chosen + 1

    return mpm_cluster


def _one_split_half(idx, subjects, half, base_dir, roi, method,
                    max_clusters, mpm_thres, mask):
    np.random.seed()
    perm = np.random.permutation(len(subjects))
    list1 = [subjects[i] for i in perm[:half]]
    list2 = [subjects[i] for i in perm[half:]]
    td = np.full(max_clusters+1, np.nan)
    tn = np.full(max_clusters+1, np.nan)
    tv = np.full(max_clusters+1, np.nan)
    tc = np.full(max_clusters+1, np.nan)
    for kc in range(2, max_clusters+1):
        print(f"split_half: {roi} kc={kc} iter={idx+1}")
        m1 = cluster_mpm_validation(base_dir, roi, list1, method, kc, mpm_thres)
        m2 = cluster_mpm_validation(base_dir, roi, list2, method, kc, mpm_thres)
        m1 *= mask; m2 *= mask
        td[kc] = v_dice(m1, m2)
        tn[kc], tv[kc] = v_nmi(m1, m2)
        tc[kc] = v_cramerv(m1, m2)
    return idx, td, tn, tv, tc



def validation_split_half(base_dir, roi, subjects, method,
                           max_clusters, n_iter, njobs,
                           group_threshold, mpm_thres):
    sub_num = len(subjects)
    half = sub_num // 2
    # load mask
    thr = np.finfo(float).eps if group_threshold == 0 else group_threshold
    mask_file = os.path.join(base_dir, "Group_xuanwu", roi,
                             f"{roi}_roimask_thr{int(thr*100)}.nii.gz")
    mask = np.nan_to_num(nib.load(mask_file).get_fdata(), nan=0) > 0
    out_dir = os.path.join(base_dir, f"validation_{sub_num}")
    os.makedirs(out_dir, exist_ok=True)

    dice_arr = np.full((n_iter, max_clusters+1), np.nan)
    nmi_arr  = np.full((n_iter, max_clusters+1), np.nan)
    vi_arr   = np.full((n_iter, max_clusters+1), np.nan)
    cv_arr   = np.full((n_iter, max_clusters+1), np.nan)

    params = (subjects, half, base_dir, roi, method,
              max_clusters, mpm_thres, mask)
    with ProcessPoolExecutor(max_workers=njobs) as exe:
        futures = {exe.submit(_one_split_half, i, *params): i for i in range(n_iter)}
        for fut in as_completed(futures):
            idx, td, tn, tv, tc = fut.result()
            dice_arr[idx] = td
            nmi_arr[idx]  = tn
            vi_arr[idx]   = tv
            cv_arr[idx]   = tc

    np.savez(os.path.join(out_dir, f"{roi}_index_split_half.npz"),
             dice=dice_arr, nmi=nmi_arr, vi=vi_arr, cv=cv_arr)
    with open(os.path.join(out_dir, f"{roi}_index_split_half.txt"), 'w') as fp:
        for kc in range(2, max_clusters+1):
            fp.write(f"cluster num = {kc}\n")
            fp.write(f"  dice: mean = {np.nanmean(dice_arr[:,kc])}, std = {np.nanstd(dice_arr[:,kc])}\n")
            fp.write(f"  nmi: mean = {np.nanmean(nmi_arr[:,kc])}, std = {np.nanstd(nmi_arr[:,kc])}\n")
            fp.write(f"  vi: mean = {np.nanmean(vi_arr[:,kc])}, std = {np.nanstd(vi_arr[:,kc])}\n")
            fp.write(f"  cv: mean = {np.nanmean(cv_arr[:,kc])}, std = {np.nanstd(cv_arr[:,kc])}\n\n")
    print(f"Split-half validation done! Results in {out_dir}") 

def _one_pairwise(kc, subjects, base_dir, roi, method, mask):
    sub_num = len(subjects)
    dice_k = np.zeros((sub_num, sub_num))
    nmi_k  = np.zeros((sub_num, sub_num))
    vi_k   = np.zeros((sub_num, sub_num))
    cv_k   = np.zeros((sub_num, sub_num))
    print(f"pairwise: {roi} kc={kc}")
    for i in range(sub_num-1):
        path1 = os.path.join(base_dir, subjects[i],
                             f"data/probtrack_old/parcellation_{method}_MNI/{roi}/seed_{kc}_relabel_group.nii.gz")
        img1 = np.nan_to_num(nib.load(path1).get_fdata(), nan=0).astype(int)
        m1 = img1 * mask
        for j in range(i+1, sub_num):
            path2 = os.path.join(base_dir, subjects[j],
                                 f"data/probtrack_old/parcellation_{method}_MNI/{roi}/seed_{kc}_relabel_group.nii.gz")
            img2 = np.nan_to_num(nib.load(path2).get_fdata(), nan=0).astype(int)
            m2 = img2 * mask
            dice_k[i,j] = v_dice(m1, m2)
            nmi_k[i,j], vi_k[i,j] = v_nmi(m1, m2)
            cv_k[i,j] = v_cramerv(m1, m2)
    return kc, dice_k, nmi_k, vi_k, cv_k


def validation_pairwise(base_dir, roi, subjects, method,
                        max_clusters, njobs, group_threshold):
    sub_num = len(subjects)
    thr = np.finfo(float).eps if group_threshold == 0 else group_threshold
    mask = np.nan_to_num(nib.load(os.path.join(
        base_dir, "Group_xuanwu", roi,
        f"{roi}_roimask_thr{int(thr*100)}.nii.gz"
    )).get_fdata(), nan=0) > 0
    out_dir = os.path.join(base_dir, f"validation_{sub_num}")
    os.makedirs(out_dir, exist_ok=True)
    dice = np.zeros((sub_num, sub_num, max_clusters+1))
    nmi  = np.zeros((sub_num, sub_num, max_clusters+1))
    vi   = np.zeros((sub_num, sub_num, max_clusters+1))
    cv   = np.zeros((sub_num, sub_num, max_clusters+1))

    params = (subjects, base_dir, roi, method, mask)
    with ProcessPoolExecutor(max_workers=njobs) as exe:
        futures = {exe.submit(_one_pairwise, kc, *params): kc for kc in range(2, max_clusters+1)}
        for fut in as_completed(futures):
            kc, dk, nk, vk, ck = fut.result()
            dice[:,:,kc] = dk; nmi[:,:,kc] = nk; vi[:,:,kc] = vk; cv[:,:,kc] = ck
    np.savez(os.path.join(out_dir, f"{roi}_index_pairwise.npz"),
             dice=dice, nmi=nmi, vi=vi, cv=cv)
    with open(os.path.join(out_dir, f"{roi}_index_pairwise.txt"), 'w') as fp:
        for kc in range(2, max_clusters+1):
            col_cv = cv[:,:,kc][cv[:,:,kc]!=0]
            col_dice = dice[:,:,kc][dice[:,:,kc]!=0]
            col_nmi = nmi[:,:,kc][nmi[:,:,kc]!=0]
            col_vi = vi[:,:,kc][vi[:,:,kc]!=0]
            fp.write(f"cluster_num: {kc}\n")
            fp.write(f"mcv: {np.nanmean(col_cv)}, std_cv: {np.nanstd(col_cv)}\n")
            fp.write(f"mdice: {np.nanmean(col_dice)}, std_dice: {np.nanstd(col_dice)}\n")
            fp.write(f"nminfo: {np.nanmean(col_nmi)}, std_nminfo: {np.nanstd(col_nmi)}\n")
            fp.write(f"mvi: {np.nanmean(col_vi)}, std_vi: {np.nanstd(col_vi)}\n\n")
    print(f"Pairwise validation done! Results in {out_dir}")


def _one_leave(idx, subjects, base_dir, roi, method,
               max_clusters, mpm_thres, mask):
    sub_num = len(subjects)
    dice_row = np.full(max_clusters+1, np.nan)
    nmi_row  = np.full(max_clusters+1, np.nan)
    vi_row   = np.full(max_clusters+1, np.nan)
    cv_row   = np.full(max_clusters+1, np.nan)
    leave_out = subjects[idx]
    others = subjects[:idx] + subjects[idx+1:]
    for kc in range(2, max_clusters+1):
        print(f"leave_one_out: {roi} kc={kc} leave_index={idx+1}/{sub_num}")
        # load left-out subject's MPM
        p1 = os.path.join(
            base_dir, leave_out,
            f"data/probtrack_old/parcellation_{method}_MNI/{roi}/seed_{kc}_relabel_group.nii.gz"
        )
        img1 = np.nan_to_num(nib.load(p1).get_fdata(), nan=0).astype(int)
        m1 = img1 * mask
        # compute group MPM of others
        m2 = cluster_mpm_validation(base_dir, roi, others, method, kc, mpm_thres)
        m2 *= mask
        dice_row[kc] = v_dice(m1, m2)
        nmi_row[kc], vi_row[kc] = v_nmi(m1, m2)
        cv_row[kc] = v_cramerv(m1, m2)
    return idx, dice_row, nmi_row, vi_row, cv_row


def validation_leave_one_out(base_dir, roi, subjects, method,
                              max_clusters, njobs, group_threshold, mpm_thres):
    sub_num = len(subjects)
    # load mask
    thr = np.finfo(float).eps if group_threshold == 0 else group_threshold
    mask = np.nan_to_num(nib.load(os.path.join(
        base_dir, "Group_xuanwu", roi,
        f"{roi}_roimask_thr{int(thr*100)}.nii.gz"
    )).get_fdata(), nan=0) > 0
    out_dir = os.path.join(base_dir, f"validation_{sub_num}")
    os.makedirs(out_dir, exist_ok=True)

    dice_arr = np.full((sub_num, max_clusters+1), np.nan)
    nmi_arr  = np.full((sub_num, max_clusters+1), np.nan)
    vi_arr   = np.full((sub_num, max_clusters+1), np.nan)
    cv_arr   = np.full((sub_num, max_clusters+1), np.nan)

    params = (subjects, base_dir, roi, method, max_clusters, mpm_thres, mask)
    with ProcessPoolExecutor(max_workers=njobs) as exe:
        futures = {exe.submit(_one_leave, i, *params): i for i in range(sub_num)}
        for fut in as_completed(futures):
            idx, dr, nr, vr, cr = fut.result()
            dice_arr[idx] = dr
            nmi_arr[idx]  = nr
            vi_arr[idx]   = vr
            cv_arr[idx]   = cr

    np.savez(os.path.join(out_dir, f"{roi}_index_leave_one_out.npz"),
             dice=dice_arr, nmi=nmi_arr, vi=vi_arr, cv=cv_arr)
    with open(os.path.join(out_dir, f"{roi}_index_leave_one_out.txt"), 'w') as fp:
        for kc in range(2, max_clusters+1):
            fp.write(f"cluster_num: {kc}\n")
            fp.write(
                f"mcv: {np.nanmean(cv_arr[:,kc])}, std_cv: {np.nanstd(cv_arr[:,kc])}\n"
            )
            fp.write(
                f"mdice: {np.nanmean(dice_arr[:,kc])}, std_dice: {np.nanstd(dice_arr[:,kc])}\n"
            )
            fp.write(
                f"nmi: {np.nanmean(nmi_arr[:,kc])}, std_nmi: {np.nanstd(nmi_arr[:,kc])}\n"
            )
            fp.write(
                f"mvi: {np.nanmean(vi_arr[:,kc])}, std_vi: {np.nanstd(vi_arr[:,kc])}\n\n"
            )
    print(f"Leave-one-out validation done! Results in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--subject_data", required=True)
    parser.add_argument("--method", default="sc")
    parser.add_argument("--max_clusters", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--njobs", type=int, default=3)
    parser.add_argument("--group_threshold", type=float, default=0.25)
    parser.add_argument("--mpm_thres", type=float, default=0.25)
    args = parser.parse_args()

    # load subjects
    if os.path.isfile(args.subject_data):
        with open(args.subject_data) as f:
            subjects = [s.strip() for s in f if s.strip()]
    else:
        subjects = args.subject_data.split(',')

    # split-half validation
    validation_split_half(
        base_dir=args.base_dir,
        roi=args.roi_name,
        subjects=subjects,
        method=args.method,
        max_clusters=args.max_clusters,
        n_iter=args.n_iter,
        njobs=args.njobs,
        group_threshold=args.group_threshold,
        mpm_thres=args.mpm_thres
    )

    # pairwise validation with parallel kc
    validation_pairwise(
        base_dir=args.base_dir,
        roi=args.roi_name,
        subjects=subjects,
        method=args.method,
        max_clusters=args.max_clusters,
        njobs=args.njobs,
        group_threshold=args.group_threshold
    )

    # leave-one-out validation
    validation_leave_one_out(
        base_dir=args.base_dir,
        roi=args.roi_name,
        subjects=subjects,
        method=args.method,
        max_clusters=args.max_clusters,
        njobs=args.njobs,
        group_threshold=args.group_threshold,
        mpm_thres=args.mpm_thres
    )