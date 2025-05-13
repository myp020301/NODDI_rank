#!/usr/bin/env python3
# 12_indices_plot.py
# =========================================================
# 绘制 8 个评估指标（split-half、pairwise、leave-one-out、
# continuity、HI、VI、silhouette、TPD）的折线/误差条图。
# ---------------------------------------------------------
import os, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # 后台绘图
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ──────────────────────── 工具函数 ────────────────────────
def _mkdir(path:str):
    os.makedirs(path, exist_ok=True)
    return path

def _load_npz(path:str, key:str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)[key]

def _errorbar(ax, x, y_mean, y_std, **kw):
    ax.errorbar(x, y_mean, yerr=y_std,
                marker='*', capsize=4, lw=1.2, **kw)

def _save(fig, out_path:str):
    fig.set_facecolor("w")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ──────────────────────── 1. split-half ────────────────────────
def plot_split_half(out_dir, roi, x_max):
    npz = os.path.join(out_dir, f"{roi}_index_split_half.npz")
    dice = _load_npz(npz, "dice")
    nmi  = _load_npz(npz, "nmi")
    cv   = _load_npz(npz, "cv")
    vi   = _load_npz(npz, "vi")

    x = np.arange(2, x_max + 1)
    fig, ax = plt.subplots(figsize=(6,4))
    _errorbar(ax, x, np.nanmean(dice,0)[2:], np.nanstd(dice,0)[2:], color='r', label="Dice")
    _errorbar(ax, x, np.nanmean(nmi ,0)[2:], np.nanstd(nmi ,0)[2:], color='b', label="NMI")
    _errorbar(ax, x, np.nanmean(cv  ,0)[2:], np.nanstd(cv  ,0)[2:], color='g', label="CV")
    ax.set_title(f"{roi}  split-half")
    ax.set_xlabel("Number of clusters"); ax.set_ylabel("Index")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{roi}_split_half.png"))

    # 额外：单独画 VI（做相邻 k t-test，与 MATLAB 基本一致）
    fig, ax = plt.subplots(figsize=(6,4))
    _errorbar(ax, x, np.nanmean(vi,0)[2:], np.nanstd(vi,0)[2:], color='r')
    for k in range(2, x_max):
        h = ttest_ind(vi[:,k], vi[:,k+1], nan_policy='omit', alternative='less').pvalue
        if h < .05:  ax.text(k+0.5, np.nanmean(vi[:,k:k+2]), "*", ha='center', va='bottom')
    ax.set_title(f"{roi}  split-half VI")
    ax.set_xlabel("Number of clusters"); ax.set_ylabel("VI")
    ax.set_xticks(x)
    _save(fig, os.path.join(out_dir, f"{roi}_split_half_vi.png"))

# ──────────────────────── 2. pairwise ─────────────────────────
def plot_pairwise(out_dir, roi, x_max):
    npz = os.path.join(out_dir, f"{roi}_index_pairwise.npz")
    dice = _load_npz(npz, "dice"); nmi  = _load_npz(npz, "nmi")
    cv   = _load_npz(npz, "cv");   vi   = _load_npz(npz, "vi")
    x = np.arange(2, x_max + 1)

    fig, ax = plt.subplots(figsize=(6,4))
    _errorbar(ax, x, np.nanmean(dice,(0,1))[2:], np.nanstd(dice,(0,1))[2:], color='r', label="Dice")
    _errorbar(ax, x, np.nanmean(nmi ,(0,1))[2:], np.nanstd(nmi ,(0,1))[2:], color='b', label="NMI")
    _errorbar(ax, x, np.nanmean(cv  ,(0,1))[2:], np.nanstd(cv  ,(0,1))[2:], color='g', label="CV")
    ax.set_title(f"{roi}  pairwise"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("Index")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{roi}_pairwise.png"))

    # VI
    fig, ax = plt.subplots(figsize=(6,4))
    _errorbar(ax, x, np.nanmean(vi,(0,1))[2:], np.nanstd(vi,(0,1))[2:], color='r')
    ax.set_title(f"{roi}  pairwise VI"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("VI")
    ax.set_xticks(x)
    _save(fig, os.path.join(out_dir, f"{roi}_pairwise_vi.png"))

# ──────────────────────── 3. leave-one-out ────────────────────
def plot_leave_one_out(out_dir, roi, x_max):
    npz = os.path.join(out_dir, f"{roi}_index_leave_one_out.npz")
    dice = _load_npz(npz, "dice"); nmi=_load_npz(npz,"nmi")
    cv=_load_npz(npz,"cv"); vi=_load_npz(npz,"vi")
    x = np.arange(2, x_max + 1)

    fig, ax = plt.subplots(figsize=(6,4))
    _errorbar(ax, x, np.nanmean(dice,0)[2:], np.nanstd(dice,0)[2:], color='r', label="Dice")
    _errorbar(ax, x, np.nanmean(nmi ,0)[2:], np.nanstd(nmi ,0)[2:], color='b', label="NMI")
    _errorbar(ax, x, np.nanmean(cv  ,0)[2:], np.nanstd(cv  ,0)[2:], color='g', label="CV")
    ax.set_title(f"{roi}  leave-one-out"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("Index")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{roi}_leave_one_out.png"))

    # VI
    fig, ax = plt.subplots(figsize=(6,4))
    _errorbar(ax, x, np.nanmean(vi,0)[2:], np.nanstd(vi,0)[2:], color='r')
    for k in range(2, x_max):
        h = ttest_ind(vi[:,k], vi[:,k+1], nan_policy='omit', alternative='less').pvalue
        if h < .05: ax.text(k+0.5, np.nanmean(vi[:,k:k+2]), "*", ha='center', va='bottom')
    ax.set_title(f"{roi}  leave-one-out VI"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("VI")
    ax.set_xticks(x)
    _save(fig, os.path.join(out_dir, f"{roi}_leave_one_out_vi.png"))

# ──────────────────────── 4. continuity ───────────────────────
def plot_cont(out_dir, roi, x_max):
    g_npz = os.path.join(out_dir, f"{roi}_index_group_continuity.npz")
    i_npz = os.path.join(out_dir, f"{roi}_index_indi_continuity.npz")
    g = _load_npz(g_npz, "group_continuity")
    indi = _load_npz(i_npz, "indi_continuity")
    x = np.arange(2, x_max+1)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, g[2:], '-r', marker='*', label='group cont')
    _errorbar(ax, x, np.nanmean(indi,0)[2:], np.nanstd(indi,0)[2:], color='b', label='indi cont')
    ax.set_title(f"{roi}  continuity"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("Index")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{roi}_continuity.png"))

# ──────────────────────── 5. HI & 6. VI (hierarchy / info) ────
def plot_hi_vi(out_dir, roi, x_max):
    g_npz = os.path.join(out_dir, f"{roi}_index_group_hi_vi.npz")
    i_npz = os.path.join(out_dir, f"{roi}_index_indi_hi_vi.npz")
    g_hi = _load_npz(g_npz,"group_hi"); g_vi = _load_npz(g_npz,"group_vi")
    i_hi = _load_npz(i_npz,"indi_hi");  i_vi = _load_npz(i_npz,"indi_vi")
    x = np.arange(3, x_max+1)

    # HI
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, g_hi[3:], '-r', marker='*', label='group HI')
    _errorbar(ax, x, np.nanmean(i_hi,0)[3:], np.nanstd(i_hi,0)[3:], color='b', label='indi HI')
    ax.set_title(f"{roi}  hierarchy index"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("Index")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{roi}_hi.png"))

    # VI
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, g_vi[3:], '-r', marker='*', label='group VI')
    _errorbar(ax, x, np.nanmean(i_vi,0)[3:], np.nanstd(i_vi,0)[3:], color='b', label='indi VI')
    # 简易显著性：左右星号（可选）
    for k in range(3, x_max):
        p1 = ttest_ind(i_vi[:,k], i_vi[:,k+1], nan_policy='omit', alternative='less').pvalue
        if p1 < .05: ax.text(k+0.5, np.nanmean(i_vi[:,k:k+2]), "*", ha='center', va='bottom')
    ax.set_title(f"{roi}  variation of information"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("VI")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{roi}_vi.png"))

# ──────────────────────── 7. silhouette ───────────────────────
def plot_sil(out_dir, roi, x_max):
    g_npz = os.path.join(out_dir, f"{roi}_index_group_silhouette.npz")
    i_npz = os.path.join(out_dir, f"{roi}_index_indi_silhouette.npz")
    g = _load_npz(g_npz, "group_silhouette")
    indi = _load_npz(i_npz, "indi_silhouette")
    x = np.arange(2, x_max+1)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, g[2:], '-r', marker='*', label='group silhouette')
    _errorbar(ax, x, np.nanmean(indi,0)[2:], np.nanstd(indi,0)[2:], color='b', label='indi silhouette')
    ax.set_title(f"{roi}  silhouette"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("Index")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{roi}_silhouette.png"))

# ──────────────────────── 8. TPD (需左右都有) ─────────────────
def plot_tpd(out_dir, roi_L, roi_R, x_max):
    """roi_L 形如 'M1_L', roi_R 对应右侧"""
    base = roi_L[:-2]            # 去掉 _L
    g_npz = os.path.join(out_dir, f"{base}_index_group_tpd.npz")
    i_npz = os.path.join(out_dir, f"{base}_index_indi_tpd.npz")
    if not (os.path.exists(g_npz) and os.path.exists(i_npz)):
        warnings.warn("TPD npz 不全，跳过绘图"); return
    g = _load_npz(g_npz,"group_tpd")
    indi = _load_npz(i_npz,"indi_tpd")
    x = np.arange(2, x_max+1)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, g[2:], '-r', marker='*', label='group TPD')
    _errorbar(ax, x, np.nanmean(indi,0)[2:], np.nanstd(indi,0)[2:], color='b', label='indi TPD')
    ax.set_title(f"{base}  TPD (L vs R)"); ax.set_xlabel("Number of clusters"); ax.set_ylabel("Index")
    ax.set_xticks(x); ax.legend()
    _save(fig, os.path.join(out_dir, f"{base}_tpd.png"))

# ──────────────────────── 主调度 ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--subject_data", required=True)       # 与 11_validation 相同
    parser.add_argument("--max_clusters", type=int, default=5)
    parser.add_argument("--njobs",        type=int, default=3)
    parser.add_argument("--no_split_half", dest="split_half", action="store_false", default=True)
    parser.add_argument("--no_pairwise",  dest="pairwise",     action="store_false", default=True)
    parser.add_argument("--no_leave_one_out", dest="leave_one_out", action="store_false", default=True)
    parser.add_argument("--no_cont", dest="cont", action="store_false", default=True)
    parser.add_argument("--no_hi_vi", dest="hi_vi", action="store_false", default=True)
    parser.add_argument("--no_sil", dest="sil", action="store_false", default=True)
    parser.add_argument("--no_tpd", dest="tpd", action="store_false", default=True)
    args = parser.parse_args()

    # 解析被试列表
    if os.path.isfile(args.subject_data):
        with open(args.subject_data) as f:
            subjects = [s.strip() for s in f if s.strip()]
    else:
        subjects = args.subject_data.split(',')
    sub_num = len(subjects)

    # 输出目录
    out_dir = _mkdir(os.path.join(args.base_dir, f"validation_{sub_num}"))

    roi = args.roi_name

    #—————— 绘图 ——————
    if args.split_half:
        plot_split_half(out_dir, roi, args.max_clusters)
    if args.pairwise:
        plot_pairwise(out_dir, roi, args.max_clusters)
    if args.leave_one_out:
        plot_leave_one_out(out_dir, roi, args.max_clusters)
    if args.cont:
        plot_cont(out_dir, roi, args.max_clusters)
    if args.hi_vi:
        plot_hi_vi(out_dir, roi, args.max_clusters)
    if args.sil:
        plot_sil(out_dir, roi, args.max_clusters)
    if args.tpd:
        if roi.endswith("_L"):
            roi_L, roi_R = roi, roi[:-2] + "_R"
            plot_tpd(out_dir, roi_L, roi_R, args.max_clusters)
        else:
            warnings.warn("TPD 需要 ROI_L 和 ROI_R，两侧均存在才绘制")

if __name__ == "__main__":
    main()
