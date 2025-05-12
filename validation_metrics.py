# validation_metrics.py
import numpy as np
import math

def v_dice(xf: np.ndarray, yf: np.ndarray) -> float:
    """严格复刻 MATLAB v_dice 行为（两标签向量均展平成 1D）。"""
    xf = xf.flatten()
    yf = yf.flatten()

    kc = int(np.max(xf))       # 假设簇标签连续 1…kc
    dice_vals = []
    for k in range(1, kc + 1):
        m1 = (xf == k)
        m2 = (yf == k)
        inter = np.logical_and(m1, m2).sum()
        denom = m1.sum() + m2.sum()
        dice_vals.append(2 * inter / denom if denom > 0 else np.nan)
    return np.nanmean(dice_vals)


def v_nmi(xf: np.ndarray, yf: np.ndarray) -> (float, float):
    """
    Compute normalized mutual information and variation of information.
    """
    xf = xf.flatten()
    yf = yf.flatten()
    # remove NaNs
    mask = ~np.isnan(xf)
    xf = xf[mask]
    yf = yf[mask]
    n = len(xf)

    # marginal entropies
    ux, cx = np.unique(xf, return_counts=True)
    px = cx / n
    ex = -np.sum(px * np.log(px + 1e-12))

    uy, cy = np.unique(yf, return_counts=True)
    py = cy / n
    ey = -np.sum(py * np.log(py + 1e-12))

    # joint entropy
    hist, _, _ = np.histogram2d(xf, yf, bins=[np.concatenate((ux, [ux.max()+1])),
                                              np.concatenate((uy, [uy.max()+1]))])
    pxy = hist / n
    exy = -np.nansum(pxy * np.log(pxy + 1e-12))

    mi = ex + ey - exy
    nmi = 2 * mi / (ex + ey + 1e-12)
    vi = ex + ey - 2 * mi
    return nmi, vi


def v_cramerv(xf: np.ndarray, yf: np.ndarray) -> float:
    """
    Compute Cramer's V for two flat label arrays.
    """
    xf = xf.flatten()
    yf = yf.flatten()
    mask = ~np.isnan(xf)
    xf = xf[mask]
    yf = yf[mask]
    n = len(xf)

    ux = np.unique(xf)
    uy = np.unique(yf)
    # contingency counts
    hist, _, _ = np.histogram2d(xf, yf, bins=[np.concatenate((ux, [ux.max()+1])),
                                              np.concatenate((uy, [uy.max()+1]))])
    counts = hist
    row = counts.sum(axis=1)
    col = counts.sum(axis=0)
    expected = np.outer(row, col) / n

    chi2 = np.nansum((counts - expected)**2 / (expected + 1e-12))
    k = min(counts.shape)
    return math.sqrt(chi2 / (n * (k - 1) + 1e-12))