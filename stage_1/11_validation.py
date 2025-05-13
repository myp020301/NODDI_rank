#!/usr/bin/env python3
import os, warnings
import argparse
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed
from validation_metrics import v_dice, v_nmi, v_cramerv
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances, silhouette_samples
from typing import List, Tuple
from scipy.ndimage import label as bwlabel

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

        mask_vol = (data > 0).astype(float)
        sumimg += mask_vol

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
    mask_file = os.path.join(base_dir, "Group", roi,
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
        base_dir, "Group", roi,
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
        base_dir, "Group", roi,
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

def validation_group_hi_vi(base_dir, roi, subjects, method,
                            max_clusters, group_threshold, mpm_thres, njobs):
    """
    Compute group Hierarchy Index (HI) and Variation of Information (VI)
    between successive K values for group MPMs.
    """
    n_sub = len(subjects)
    thr = group_threshold or np.finfo(float).eps
    vox_dir = os.path.join(base_dir, f"MPM_{n_sub}")
    out_dir = os.path.join(base_dir, f"validation_{n_sub}")
    os.makedirs(out_dir, exist_ok=True)

    group_hi = np.zeros(max_clusters+1)
    group_vi = np.zeros(max_clusters+1)

    for kc in range(3, max_clusters+1):
        # load MPM for kc-1 and kc
        f1 = os.path.join(vox_dir, f"{roi}_{kc-1}_MPM_thr{int(mpm_thres*100)}_group.nii.gz")
        f2 = os.path.join(vox_dir, f"{roi}_{kc}_MPM_thr{int(mpm_thres*100)}_group.nii.gz")
        m1 = np.nan_to_num(nib.load(f1).get_fdata(), nan=0).astype(int)
        m2 = np.nan_to_num(nib.load(f2).get_fdata(), nan=0).astype(int)

        xi = []
        for i in range(1, kc+1):
            mask_i = (m2 == i)
            counts = []
            for j in range(1, kc):
                counts.append(np.sum((m1[mask_i] == j)))
            total = np.sum(counts)
            if total > 0:
                xi.append(np.max(counts) / total)
        group_hi[kc] = np.nanmean(xi) if xi else np.nan

        # compute VI between m1 and m2
        _, group_vi[kc] = v_nmi(m1, m2)

    # save results
    np.savez(os.path.join(out_dir, f"{roi}_index_group_hi_vi.npz"),
             group_hi=group_hi, group_vi=group_vi)
    with open(os.path.join(out_dir, f"{roi}_index_group_hi_vi.txt"), 'w') as fp:
        for kc in range(3, max_clusters+1):
            fp.write(f"cluster: {kc-1}->{kc} hi={group_hi[kc]:.4f} vi={group_vi[kc]:.4f}\n")
    print("Group HI/VI validation done!")


def _one_indi_hi_vi(idx, subjects, base_dir, roi, method,
                    max_clusters, group_threshold, mpm_thres):
    """
    并行计算单个被试的 HI/VI。
    """
    sub = subjects[idx]
    # 加载 Mask
    thr = group_threshold if group_threshold != 0 else np.finfo(float).eps
    mask_dir = os.path.join(base_dir, "Group", roi)
    mask = np.nan_to_num(
        nib.load(os.path.join(mask_dir, f"{roi}_roimask_thr{int(thr*100)}.nii.gz")).get_fdata(),
        nan=0
    ).astype(bool)

    hi_row = np.full(max_clusters+1, np.nan)
    vi_row = np.full(max_clusters+1, np.nan)

    for kc in range(3, max_clusters+1):
        # load mpm (kc-1 vs kc)
        fn1 = os.path.join(
            base_dir, sub,
            f"data/probtrack_old/parcellation_{method}_MNI/{roi}/seed_{kc-1}_relabel_group.nii.gz"
        )
        fn2 = os.path.join(
            base_dir, sub,
            f"data/probtrack_old/parcellation_{method}_MNI/{roi}/seed_{kc}_relabel_group.nii.gz"
        )
        img1 = np.nan_to_num(nib.load(fn1).get_fdata(), nan=0).astype(int) * mask
        img2 = np.nan_to_num(nib.load(fn2).get_fdata(), nan=0).astype(int) * mask

        # 构建 xmatrix 并计算 xi
        xmat = np.zeros((kc, kc-1), dtype=int)
        xi   = np.zeros(kc, dtype=float)
        for i in range(1, kc+1):
            idx_vox = (img2 == i)
            for j in range(1, kc):
                xmat[i-1, j-1] = np.count_nonzero(img1[idx_vox] == j)
            row = xmat[i-1]
            xi[i-1] = row.max() / row.sum() if row.sum()>0 else np.nan

        hi_row[kc] = np.nanmean(xi)
        vi_row[kc] = v_nmi(img1, img2)[1]

    return idx, hi_row, vi_row


# 2. 在 validation_indi_hi_vi 中直接引用顶层函数
def validation_indi_hi_vi(base_dir, roi, subjects, method,
                          max_clusters, njobs, group_threshold, mpm_thres):
    sub_num = len(subjects)
    out_dir = os.path.join(base_dir, f"validation_{sub_num}")
    os.makedirs(out_dir, exist_ok=True)

    indi_hi = np.full((sub_num, max_clusters+1), np.nan)
    indi_vi = np.full((sub_num, max_clusters+1), np.nan)

    # 并行提交
    with ProcessPoolExecutor(max_workers=njobs) as exe:
        futures = {
            exe.submit(
                _one_indi_hi_vi,
                i, subjects, base_dir, roi, method,
                max_clusters, group_threshold, mpm_thres
            ): i
            for i in range(sub_num)
        }
        for fut in as_completed(futures):
            idx, hi_row, vi_row = fut.result()
            indi_hi[idx] = hi_row
            indi_vi[idx] = vi_row

    # 保存 .npz 和 .txt ...
    np.savez(os.path.join(out_dir, f"{roi}_index_indi_hi_vi.npz"),
             indi_hi=indi_hi, indi_vi=indi_vi)
    with open(os.path.join(out_dir, f"{roi}_index_indi_hi_vi.txt"), 'w') as fp:
        for kc in range(3, max_clusters+1):
            vals_hi = indi_hi[:,kc]
            vals_vi = indi_vi[:,kc]
            fp.write(
                f"cluster_num: {kc-1}->{kc}\n"
                f"avg_indi_hi: {np.nanmean(vals_hi)}, std: {np.nanstd(vals_hi)}, "
                f"median: {np.nanmedian(vals_hi)}\n"
                f"avg_indi_vi: {np.nanmean(vals_vi)}, std: {np.nanstd(vals_vi)}, "
                f"median: {np.nanmedian(vals_vi)}\n\n"
            )
    print(f"Individual HI/VI done! Results in {out_dir}")   

# ──────────────────────── 辅助：邻接矩阵 ────────────────────────
def _adjacency(img: np.ndarray, kc: int, struct: np.ndarray):
    """计算 kc×kc 的邻接计数矩阵，并按行归一化（除非 kc==2）。"""
    mat = np.zeros((kc, kc), dtype=float)
    for i in range(1, kc + 1):
        mask_i = img == i
        dil_i = binary_dilation(mask_i, structure=struct)
        for j in range(1, kc + 1):
            if i != j:
                mat[j - 1, i - 1] = np.count_nonzero(dil_i & (img == j))
    if kc != 2:
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sum, where=row_sum != 0)
    return mat

# ──────────────────────── group‑level TPD ─────────────────────
def validation_group_tpd(base_dir: str, roi: str, subjects: List[str],
                         max_clusters: int, mpm_thres: float):
    """
    只在 roi 以 '_L' 结尾时才计算 TPD；否则直接跳过。
    依次对 kc=2..max_clusters 计算左右群组 MPM 的余弦距离。
    """
    if not roi.endswith("_L"):
        print(f"[TPD‑group] ROI {roi} 没有 '_L' 后缀 —— 跳过 TPD 计算。")
        return

    roi_base = roi[:-2]
    n_sub = len(subjects)
    mpm_dir = os.path.join(base_dir, f"MPM_{n_sub}")

    group_tpd = np.full(max_clusters + 1, np.nan)
    struct = np.ones((3, 3, 3), dtype=bool)

    for kc in range(2, max_clusters + 1):
        fn_l = f"{roi_base}_L_{kc}_MPM_thr{int(mpm_thres * 100)}_group.nii.gz"
        fn_r = f"{roi_base}_R_{kc}_MPM_thr{int(mpm_thres * 100)}_group.nii.gz"
        path_l = os.path.join(mpm_dir, fn_l)
        path_r = os.path.join(mpm_dir, fn_r)

        if not (os.path.exists(path_l) and os.path.exists(path_r)):
            raise FileNotFoundError(
                f"[TPD‑group] 缺少配对文件：{path_l if not os.path.exists(path_l) else ''} "
                f"{path_r if not os.path.exists(path_r) else ''}"
            )

        img_l = np.nan_to_num(nib.load(path_l).get_fdata(), nan=0).astype(int)
        img_r = np.nan_to_num(nib.load(path_r).get_fdata(), nan=0).astype(int)

        con_l = _adjacency(img_l, kc, struct)
        con_r = _adjacency(img_r, kc, struct)
        group_tpd[kc] = cosine(con_l.T.ravel(), con_r.T.ravel())
        print(f"[TPD‑group] {roi_base} kc={kc} tpd={group_tpd[kc]:.4f}")

    # 保存
    out_dir = os.path.join(base_dir, f"validation_{n_sub}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{roi_base}_index_group_tpd.npz"),
             group_tpd=group_tpd)
    with open(os.path.join(out_dir, f"{roi_base}_index_group_tpd.txt"), "w") as fp:
        for kc in range(2, max_clusters + 1):
            fp.write(f"cluster_num: {kc}\ngroup_tpd: {group_tpd[kc]}\n\n")
    print(f"[TPD‑group] 结果已保存至 {out_dir}")


# ────────── individual‑level TPD 并行任务 (顶层函数) ──────────
def _calc_indi_tpd(idx: int, subjects: List[str], base_dir: str, roi_base: str,
                   method: str, max_clusters: int,
                   mask_l: np.ndarray, mask_r: np.ndarray,
                   struct: np.ndarray) -> Tuple[int, np.ndarray]:
    """并行：计算单个被试的 TPD 序列。"""
    sub = subjects[idx]
    res = np.full(max_clusters + 1, np.nan)

    for kc in range(2, max_clusters + 1):
        f_l = os.path.join(
            base_dir, sub,
            "data", "probtrack_old",
            f"parcellation_{method}_MNI/{roi_base}_L",
            f"seed_{kc}_relabel_group.nii.gz"
        )
        f_r = os.path.join(
            base_dir, sub,
            "data", "probtrack_old",
            f"parcellation_{method}_MNI/{roi_base}_R",
            f"seed_{kc}_relabel_group.nii.gz"
        )
        if not (os.path.exists(f_l) and os.path.exists(f_r)):
            raise FileNotFoundError(f"缺少 {f_l} 或 {f_r}")

        img_l = (np.nan_to_num(nib.load(f_l).get_fdata(), nan=0)
                 .astype(int) * mask_l)
        img_r = (np.nan_to_num(nib.load(f_r).get_fdata(), nan=0)
                 .astype(int) * mask_r)

        con_l = _adjacency(img_l, kc, struct)
        con_r = _adjacency(img_r, kc, struct)
        res[kc] = cosine(con_l.T.ravel(), con_r.T.ravel())

    return idx, res


# ──────────────────────── individual‑level TPD ────────────────────────
def validation_indi_tpd(base_dir: str, roi: str, subjects: List[str], method: str,
                        max_clusters: int, njobs: int,
                        group_threshold: float, mpm_thres: float) -> None:
    """
    当 roi 以 '_L' 结尾时才执行；否则直接跳过。
    对每个受试者并行计算 indi‑TPD。
    """
    if not roi.endswith("_L"):
        print(f"[TPD‑indi] ROI {roi} 没有 '_L' 后缀 —— 跳过 TPD 计算。")
        return

    roi_base = roi[:-2]
    n_sub = len(subjects)

    # 加载左右半球 group mask
    thr = group_threshold if group_threshold != 0 else np.finfo(float).eps
    thr_i = int(thr * 100)
    mask_dir_l = os.path.join(base_dir, "Group", roi)
    mask_dir_r = os.path.join(base_dir, "Group", f"{roi_base}_R")
    mask_l = np.nan_to_num(
        nib.load(os.path.join(mask_dir_l, f"{roi_base}_L_roimask_thr{thr_i}.nii.gz")
                 ).get_fdata(), nan=0
    ).astype(bool)
    mask_r = np.nan_to_num(
        nib.load(os.path.join(mask_dir_r, f"{roi_base}_R_roimask_thr{thr_i}.nii.gz")
                 ).get_fdata(), nan=0
    ).astype(bool)

    struct = np.ones((3, 3, 3), dtype=bool)
    indi_tpd = np.full((n_sub, max_clusters + 1), np.nan)

    with ProcessPoolExecutor(max_workers=njobs) as exe:
        futures = {
            exe.submit(_calc_indi_tpd, i, subjects, base_dir, roi_base,
                       method, max_clusters, mask_l, mask_r, struct): i
            for i in range(n_sub)
        }
        for fut in as_completed(futures):
            idx, row = fut.result()
            indi_tpd[idx] = row
            print(f"[TPD‑indi] 完成 subj {idx+1}/{n_sub}")

    # 保存
    out_dir = os.path.join(base_dir, f"validation_{n_sub}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{roi_base}_index_indi_tpd.npz"),
             indi_tpd=indi_tpd)
    with open(os.path.join(out_dir, f"{roi_base}_index_indi_tpd.txt"), "w") as fp:
        for kc in range(2, max_clusters + 1):
            vals = indi_tpd[:, kc]
            fp.write(
                f"cluster_num: {kc}\n"
                f"avg_indi_tpd: {np.nanmean(vals)}\n"
                f"std_indi_tpd: {np.nanstd(vals)}\n"
                f"median_indi_tpd: {np.nanmedian(vals)}\n\n"
            )
    print(f"[TPD‑indi] 结果已保存至 {out_dir}")

def validation_group_silhouette(
        base_dir: str, roi: str, subjects: List[str],
        max_clusters: int, mpm_thres: float) -> None:

    if roi.endswith(("_L", "_R")):
        hemi      = roi[-1]
        roi_base  = roi[:-2]
        prefix    = f"{roi_base}_{hemi}_"
    else:
        roi_base  = roi
        prefix    = f"{roi_base}_"

    n_sub   = len(subjects)
    mpm_dir = os.path.join(base_dir, f"MPM_{n_sub}")
    group_sil = np.full(max_clusters + 1, np.nan)

    for kc in range(2, max_clusters + 1):
        fn   = f"{prefix}{kc}_MPM_thr{int(mpm_thres*100)}_group.nii.gz"
        path = os.path.join(mpm_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        img    = np.nan_to_num(nib.load(path).get_fdata(), nan=0).astype(int)
        xs, ys, zs = np.where(img > 0)
        labels = img[xs, ys, zs]

        # ---------- 打印每个簇的体素数 ----------
        uniq, cnt = np.unique(labels, return_counts=True)
        print(f"[SIL-group] {roi} kc={kc} label_counts: {dict(zip(uniq, cnt))}")
        # --------------------------------------

        # --- 与 MATLAB 一致：4 维特征 + sqEuclidean ---
        X_feat = np.column_stack((xs, ys, zs, labels.astype(float)))
        # 先算欧氏距离再平方 → 得到 sqEuclidean
        D = pairwise_distances(X_feat, metric="euclidean") ** 2
        sil_vals = silhouette_samples(D, labels, metric="precomputed")
        # ----------------------------------------------

        group_sil[kc] = np.nanmean(sil_vals)
        print(f"[SIL-group] {roi} kc={kc} silhouette={group_sil[kc]:.4f}")

    # ───── 保存结果 ─────
    out_dir = os.path.join(base_dir, f"validation_{n_sub}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{roi}_index_group_silhouette.npz"),
             group_silhouette=group_sil)
    with open(os.path.join(out_dir, f"{roi}_index_group_silhouette.txt"), "w") as fp:
        for kc in range(2, max_clusters + 1):
            fp.write(f"cluster_num: {kc}\naverage_group_silhouette: {group_sil[kc]}\n\n")

# ────────── individual-silhouette 并行任务 ──────────
def _calc_indi_sil(idx: int, subjects: List[str], base_dir: str, roi: str,
                   method: str, max_clusters: int) -> Tuple[int, np.ndarray]:
    sub     = subjects[idx]
    sub_dir = os.path.join(base_dir, sub)

    # ① 加载坐标 & 连接矩阵 --------------------------------------------------
    coord_file = os.path.join(sub_dir, "data", "seeds_txt_all",
                              f"seed_region_{roi}.txt")
    con_file   = os.path.join(sub_dir, "data", "probtrack_old", "con_cor",
                              f"con_matrix_seed_{roi}.npy")
    if not (os.path.exists(coord_file) and os.path.exists(con_file)):
        raise FileNotFoundError(f"{coord_file} 或 {con_file} 缺失")

    xyz   = np.loadtxt(coord_file, dtype=int)          # (N,3)
    conn  = np.load(con_file).astype(np.float64)       # (N,N)
    conn /= conn.sum(1, keepdims=True) + 1e-12         # 归一化防 0
    dist_mat = squareform(pdist(conn, metric="cosine"))  # (N,N)

    # ② 逐 kc 计算 silhouette -------------------------------------------------
    sil_row = np.full(max_clusters + 1, np.nan)

    for kc in range(2, max_clusters + 1):
        seg_path = os.path.join(
            sub_dir, "data", "probtrack_old",
            f"parcellation_{method}",
            f"seed_{roi}_{kc}.nii.gz"
        )
        if not os.path.exists(seg_path):
            warnings.warn(f"{seg_path} 缺失，跳 subj={sub} kc={kc}")
            continue
        lab_img = np.nan_to_num(nib.load(seg_path).get_fdata(), nan=0).astype(int)
        u, c = np.unique(lab_img, return_counts=True)
        label_counts_full = dict(zip(u.tolist(), c.tolist()))      
        print(f"[SIL-indi] seg_path={seg_path} seg_label_counts={label_counts_full}")    
              

        labels  = lab_img[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
        print(f"[SIL-indi] subj={sub} kc={kc}  label_counts={label_counts}")

        # —— 关键改动：簇数不足时直接给 NaN，不抛错也不强行置 0
        if np.unique(labels).size < 2:
            continue

        sil_vals     = silhouette_samples(dist_mat, labels, metric="precomputed")
        sil_row[kc]  = np.nanmean(sil_vals)

    return idx, sil_row


# ────────── individual‑silhouette 主调度 ──────────
def validation_indi_silhouette(base_dir: str, roi: str, subjects: List[str],
                               method: str, max_clusters: int, njobs: int):
    """
    读取新的路径结构并并行计算 silhouette。
    """
    n_sub = len(subjects)
    indi_sil = np.full((n_sub, max_clusters + 1), np.nan)

    with ProcessPoolExecutor(max_workers=njobs) as exe:
        futs = {exe.submit(_calc_indi_sil, i, subjects, base_dir, roi,
                           method, max_clusters): i
                for i in range(n_sub)}
        for fut in as_completed(futs):
            idx, row = fut.result()
            indi_sil[idx] = row
            print(f"[SIL‑indi] subj {idx+1}/{n_sub} done")

    out_dir = os.path.join(base_dir, f"validation_{n_sub}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{roi}_index_indi_silhouette.npz"),
             indi_silhouette=indi_sil)
    with open(os.path.join(out_dir, f"{roi}_index_indi_silhouette.txt"), "w") as fp:
        for kc in range(2, max_clusters + 1):
            vals = indi_sil[:, kc]
            fp.write(
                f"cluster_num: {kc}\n"
                f"avg_indi_silhouette: {np.nanmean(vals)}\n"
                f"std_indi_silhouette: {np.nanstd(vals)}\n"
                f"median_indi_silhouette: {np.nanmedian(vals)}\n\n"
            )
    print(f"[SIL‑indi] 结果保存于 {out_dir}")

# ────────── continuity 基函数 ──────────
def _continuity_index(img: np.ndarray, kc: int) -> float:
    """
    img: 3D int array, 取值 0..kc
    返回: 平均 continuity = 每个簇 (最大连通分量 / 全簇体素) 再对 kc 取均值
    """
    struct = np.zeros((3, 3, 3), dtype=bool)        # 6-neigh
    struct[1, 1, [0, 2]] = True
    struct[1, [0, 2], 1] = True
    struct[[0, 2], 1, 1] = True

    cont_sum = 0
    for i in range(1, kc + 1):
        sub = (img == i)
        if not sub.any():
            continue
        labeled, n = bwlabel(sub, structure=struct)
        if n == 0:
            continue
        sizes = np.bincount(labeled.ravel())[1:]     # 忽略 0
        cont_sum += sizes.max() / sizes.sum()
    return cont_sum / kc


# ────────── group‑level continuity ──────────
def validation_group_cont(base_dir: str, roi: str, subjects: List[str],
                          max_clusters: int, mpm_thres: float) -> None:
    n_sub = len(subjects)
    mpm_dir = os.path.join(base_dir, f"MPM_{n_sub}")

    # ROI 文件前缀
    if roi.endswith(("_L", "_R")):
        prefix = f"{roi}_"
    else:
        prefix = f"{roi}_"

    group_cont = np.full(max_clusters + 1, np.nan)
    for kc in range(2, max_clusters + 1):
        fn = f"{prefix}{kc}_MPM_thr{int(mpm_thres*100)}_group.nii.gz"
        path = os.path.join(mpm_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        img = np.nan_to_num(nib.load(path).get_fdata(), nan=0).astype(int)
        group_cont[kc] = _continuity_index(img, kc)
        print(f"[CONT‑group] {roi} kc={kc} cont={group_cont[kc]:.4f}")

    out_dir = os.path.join(base_dir, f"validation_{n_sub}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{roi}_index_group_continuity.npz"),
             group_continuity=group_cont)
    with open(os.path.join(out_dir, f"{roi}_index_group_continuity.txt"), "w") as fp:
        for kc in range(2, max_clusters + 1):
            fp.write(
                f"cluster_num: {kc}\ngroup_continuity: {group_cont[kc]}\n\n"
            )
    print(f"[CONT‑group] 结果保存于 {out_dir}")


# ────────── individual‑level 并行任务 ──────────
def _calc_indi_cont(idx: int, subjects: List[str], base_dir: str, roi: str,
                    method: str, max_clusters: int,
                    mask: np.ndarray) -> Tuple[int, np.ndarray]:
    sub = subjects[idx]
    sub_dir = os.path.join(base_dir, sub)
    res = np.full(max_clusters + 1, np.nan)

    for kc in range(2, max_clusters + 1):
        seg_path = os.path.join(
            sub_dir, "data", "probtrack_old",
            f"parcellation_{method}_MNI/{roi}",
            f"seed_{kc}_relabel_group.nii.gz"
        )
        if not os.path.exists(seg_path):
            warnings.warn(f"{seg_path} 缺失，跳 subj={sub} kc={kc}")
            continue
        img = (np.nan_to_num(nib.load(seg_path).get_fdata(), nan=0)
               .astype(int) * mask)
        res[kc] = _continuity_index(img, kc)
    return idx, res


# ────────── individual‑level continuity ──────────
def validation_indi_cont(base_dir: str, roi: str, subjects: List[str], method: str,
                         max_clusters: int, njobs: int,
                         group_threshold: float) -> None:
    n_sub = len(subjects)

    # 加载 group mask (若不存在，使用全 1)
    thr = group_threshold if group_threshold != 0 else np.finfo(float).eps
    thr_i = int(thr * 100)
    mask_dir = os.path.join(base_dir, "Group", roi)
    mask_path = os.path.join(mask_dir, f"{roi}_roimask_thr{thr_i}.nii.gz")
    if os.path.exists(mask_path):
        mask = np.nan_to_num(nib.load(mask_path).get_fdata(), nan=0).astype(bool)
    else:
        warnings.warn(f"mask {mask_path} 缺失，使用全 1 掩码")
        # 从任一分割文件取体积形状
        sample_img = nib.load(
            os.path.join(base_dir, subjects[0],
                         "data", "probtrack_old",
                         f"parcellation_{method}_MNI/{roi}",
                         f"seed_2_relabel_group.nii.gz")
        )
        mask = np.ones(sample_img.shape, dtype=bool)

    indi_cont = np.full((n_sub, max_clusters + 1), np.nan)
    with ProcessPoolExecutor(max_workers=njobs) as exe:
        futs = {exe.submit(_calc_indi_cont, i, subjects, base_dir, roi, method,
                           max_clusters, mask): i
                for i in range(n_sub)}
        for fut in as_completed(futs):
            idx, row = fut.result()
            indi_cont[idx] = row
            print(f"[CONT‑indi] subj {idx+1}/{n_sub} done")

    out_dir = os.path.join(base_dir, f"validation_{n_sub}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{roi}_index_indi_continuity.npz"),
             indi_continuity=indi_cont)
    with open(os.path.join(out_dir, f"{roi}_index_indi_continuity.txt"), "w") as fp:
        for kc in range(2, max_clusters + 1):
            vals = indi_cont[:, kc]
            fp.write(
                f"cluster_num: {kc}\n"
                f"avg_indi_continuity: {np.nanmean(vals)}\n"
                f"std_indi_continuity: {np.nanstd(vals)}\n"
                f"median_indi_continuity: {np.nanmedian(vals)}\n\n"
            )
    print(f"[CONT‑indi] 结果保存于 {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--subject_data", required=True)
    parser.add_argument("--max_clusters", type=int, default=5)
    parser.add_argument("--method", default="sc")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--njobs", type=int, default=3)
    parser.add_argument("--group_threshold", type=float, default=0.25)
    parser.add_argument("--mpm_thres", type=float, default=0.25)
    parser.add_argument("--no_split_half", dest="split_half", action="store_false", default=True)
    parser.add_argument("--no_pairwise",  dest="pairwise",  action="store_false", default=True)
    parser.add_argument("--no_leave_one_out", dest="leave_one_out", action="store_false", default=True)
    parser.add_argument("--no_group_hi_vi", dest="group_hi_vi", action="store_false", default=True)
    parser.add_argument("--no_indi_hi_vi",  dest="indi_hi_vi",  action="store_false", default=True)
    parser.add_argument("--no_group_tpd", dest="group_tpd", action="store_false", default=True)
    parser.add_argument("--no_indi_tpd", dest="indi_tpd", action="store_false", default=True)
    parser.add_argument("--no_group_silhouette", dest="group_sil", action="store_false", default=True)
    parser.add_argument("--no_indi_silhouette", dest="indi_sil", action="store_false", default=True)
    parser.add_argument("--no_group_continuity", dest="group_cont", action="store_false",default=True)
    parser.add_argument("--no_indi_continuity",  dest="indi_cont",  action="store_false",default=True)
    args = parser.parse_args()

    # load subjects
    if os.path.isfile(args.subject_data):
        with open(args.subject_data) as f:
            subjects = [s.strip() for s in f if s.strip()]
    else:
        subjects = args.subject_data.split(',')
    if args.split_half:
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
    if args.pairwise:
        validation_pairwise(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            method=args.method,
            max_clusters=args.max_clusters,
            njobs=args.njobs,
            group_threshold=args.group_threshold
        )
    if args.leave_one_out:
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
    
    if args.group_hi_vi:
        validation_group_hi_vi(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            method=args.method,
            max_clusters=args.max_clusters,
            group_threshold=args.group_threshold,
            mpm_thres=args.mpm_thres,
            njobs=args.njobs
        )
    if args.indi_hi_vi:
        validation_indi_hi_vi(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            method=args.method,
            max_clusters=args.max_clusters,
            group_threshold=args.group_threshold,
            mpm_thres=args.mpm_thres,
            njobs=args.njobs
        )
    if args.group_tpd:
        validation_group_tpd(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            max_clusters=args.max_clusters,
            mpm_thres=args.mpm_thres
        )
    if args.indi_tpd:
        validation_indi_tpd(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            method=args.method,
            max_clusters=args.max_clusters,
            njobs=args.njobs,
            group_threshold=args.group_threshold,
            mpm_thres=args.mpm_thres
        )
     
    if args.group_sil:
        validation_group_silhouette(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            max_clusters=args.max_clusters,
            mpm_thres=args.mpm_thres
        )
        
    if args.indi_sil:
        validation_indi_silhouette(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            method=args.method,
            max_clusters=args.max_clusters,
            njobs=args.njobs
        )
    if args.group_cont:
        validation_group_cont(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            max_clusters=args.max_clusters,
            mpm_thres=args.mpm_thres
        )
    
    if args.indi_cont:
        validation_indi_cont(
            base_dir=args.base_dir,
            roi=args.roi_name,
            subjects=subjects,
            method=args.method,
            max_clusters=args.max_clusters,
            njobs=args.njobs,
            group_threshold=args.group_threshold
        )
