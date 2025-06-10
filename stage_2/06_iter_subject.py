#!/usr/bin/env python3
"""
06_iter_subject.py  ——  dMRI-GIAGA 个体迭代（修复 6-邻域平滑索引）
--------------------------------------
主要更改：
  • 从 seed_lin 构建 “seed-only” 的 6-邻域列表，而不再用 build_adj(jhu_mask)
  • smooth() 内确保对 vec_vox 行数索引合法
  • 其余逻辑与之前一致：每轮保存子文件夹中带迭代号的标签/置信度/ROI 图
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from utils import pearson_z, overlap_ratio, CFG, WORKDIR

TH_CONF, ALPHA, EPS = 1.1, 0.9, 1e-12

def smooth(z, neigh):
    """
    对 z 进行 6-邻域均值平滑。
    - z.shape = (N_seed, 50)
    - neigh: 长度为 N_seed 的列表，每个元素是一个 numpy 数组，包含该 seed 的邻居的 seed 索引
    """
    out = z.copy()
    for v, nb in enumerate(neigh):
        if nb.size > 0:
            out[v] = z[nb].mean(axis=0)
    return out

def build_seed_adjacency(seed_lin, mask_shape):
    """
    根据 seed 在原始 3D 体积（mask_shape）中的线性索引 seed_lin，
    构建 “seed-only” 的 6-邻域。返回列表 neigh，长度 = N_seed；
    neigh[i] 是一个 np.ndarray，包含与 seed i 相邻的那些 seed 在 vec_vox 中的行索引。
    """
    # 1. 将 seed_lin 转为 3D 坐标
    #    mask_shape = (nx, ny, nz)
    coords = np.vstack(np.unravel_index(seed_lin, mask_shape)).T  # (N_seed, 3)

    # 2. 建立从 (x,y,z) → seed_index 的映射表；先初始化为 -1
    total_vox = mask_shape[0] * mask_shape[1] * mask_shape[2]
    lin2seed = -np.ones(total_vox, dtype=np.int64)
    for idx, lin in enumerate(seed_lin):
        lin2seed[int(lin)] = idx

    # 3. 对每个 seed，寻找它的 6 个邻居（±x, ±y, ±z），若邻居也是 seed，则加入邻接列表
    neigh = []
    nx, ny, nz = mask_shape
    for i, (x, y, z) in enumerate(coords):
        neighbors = []
        # 定义 6 种偏移
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            xx, yy, zz = x+dx, y+dy, z+dz
            # 检查边界
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                lin_nb = (xx * ny + yy) * nz + zz
                seed_nb = lin2seed[int(lin_nb)]
                if seed_nb >= 0:
                    neighbors.append(seed_nb)
        neigh.append(np.array(neighbors, dtype=np.int64))
    return neigh

def iter_once(vec_vox, vec_roi, neigh):
    z = pearson_z(vec_vox, vec_roi)
    z = smooth(z, neigh)
    idx = z.argsort(axis=1)[:, ::-1][:, :2]  # 前两大相关值下标
    lbl = idx[:,0] + 1
    conf = z[np.arange(len(lbl)), idx[:,0]] / (z[np.arange(len(lbl)), idx[:,1]] + EPS)
    return lbl, conf

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--sub", required=True)
    args = pa.parse_args()
    sdir = WORKDIR / args.sub

    # 加载 vec_vox, seed_lin
    vec_vox  = np.load(sdir / "vec_vox.npy")   # (N_seed × V)
    seed_lin = np.load(sdir / "seed_lin.npy")  # (N_seed,)
    N_seed   = vec_vox.shape[0]
    assert len(seed_lin) == N_seed, "seed_lin 与 vec_vox 行数不符"

    # —— 初始参考标签（群体模板→个体；若无则 JHU50） —— 
    atlas_indiv = sdir / "atlas_iter.nii.gz"
    if not atlas_indiv.exists():
        atlas_indiv = sdir / "atlas" / "JHU50_sub.nii.gz"
    lbl_img = nib.load(str(atlas_indiv))
    lbl_vol = lbl_img.get_fdata().astype(int)  # 原始 2 mm 空间标签 (X×Y×Z)
    jhu_mask = lbl_vol > 0

    # 在原始 2 mm 体积里，每个 seed 的初始 ROI 编号
    vox2roi = lbl_vol.reshape(-1)[seed_lin]    # (N_seed,)
    V = vec_vox.shape[1]

    # 计算初始 vec_roi：每个 ROI 的平均向量
    vec_roi = np.zeros((50, V), dtype=np.float32)
    for k in range(1, 51):
        idx = (vox2roi == k)
        if idx.any():
            vec_roi[k-1] = vec_vox[idx].mean(axis=0)

    # 破解 smooth 的索引越界：为 seed 单独构建 6-邻域
    mask_shape = lbl_vol.shape  # e.g. (125, 125, 75)
    neigh = build_seed_adjacency(seed_lin, mask_shape)

    # 准备保存子目录
    labels_dir    = sdir / "labels";          labels_dir.mkdir(exist_ok=True)
    confs_dir     = sdir / "confs";           confs_dir.mkdir(exist_ok=True)
    networks_dir  = sdir / "networks";        networks_dir.mkdir(exist_ok=True)
    net_confs_dir = sdir / "network_confs";   net_confs_dir.mkdir(exist_ok=True)

    lbl_prev = vox2roi.copy()

    for it in range(CFG["iter_max_sub"]):
        lbl, conf = iter_once(vec_vox, vec_roi, neigh)

        # -------- 存储整图标签 label_iter{it+1}.nii.gz --------
        full_lbl = np.zeros(jhu_mask.size, dtype=np.int16)
        full_lbl[seed_lin] = lbl
        label_img = nib.Nifti1Image(full_lbl.reshape(jhu_mask.shape), lbl_img.affine)
        nib.save(label_img, str(labels_dir / f"label_iter{it+1}.nii.gz"))

        # -------- 存储整图置信度 conf_iter{it+1}.nii.gz --------
        full_conf = np.zeros(jhu_mask.size, dtype=np.float32)
        full_conf[seed_lin] = conf
        conf_img = nib.Nifti1Image(full_conf.reshape(jhu_mask.shape), lbl_img.affine)
        nib.save(conf_img, str(confs_dir / f"conf_iter{it+1}.nii.gz"))

        # -------- 存储逐 ROI 的二值图 & 置信度图 --------
        for k in range(1, 51):
            roi_mask = (lbl == k)
            if roi_mask.any():
                # 二值网络图 network_{k:02d}_iter{it+1}.nii.gz
                vol_bin = np.zeros_like(full_lbl)
                vol_bin[seed_lin] = roi_mask.astype(np.uint8)
                net_img = nib.Nifti1Image(vol_bin.reshape(jhu_mask.shape), lbl_img.affine)
                nib.save(net_img, str(networks_dir / f"network_{k:02d}_iter{it+1}.nii.gz"))

                # ROI 置信度图 conf_{k:02d}_iter{it+1}.nii.gz
                roi_conf = (lbl == k) * conf
                vol_conf = np.zeros_like(full_conf)
                vol_conf[seed_lin] = roi_conf
                net_conf_img = nib.Nifti1Image(vol_conf.reshape(jhu_mask.shape), lbl_img.affine)
                nib.save(net_conf_img, str(net_confs_dir / f"conf_{k:02d}_iter{it+1}.nii.gz"))

        # -------- 收敛检测 --------
        ov = overlap_ratio(lbl_prev, lbl)
        print(f"[{args.sub}] iter {it+1}: overlap = {ov:.4f}")
        if ov >= 0.98:
            print(f"[{args.sub}] 收敛，停止个体迭代。")
            break
        lbl_prev = lbl

        # -------- 更新参考轮廓 --------
        for k in range(1, 51):
            idx = (lbl == k) & (conf > TH_CONF)
            if idx.any():
                vec_roi[k-1] = ALPHA * vec_roi[k-1] + (1 - ALPHA) * vec_vox[idx].mean(axis=0)

    # -------- 保存最终结果 --------
    full_lbl[seed_lin] = lbl_prev
    final_img = nib.Nifti1Image(full_lbl.reshape(jhu_mask.shape), lbl_img.affine)
    nib.save(final_img, str(labels_dir / "label_final.nii.gz"))
    np.save(sdir / "vec_roi_final.npy", vec_roi)

    print(f"[OK] {args.sub}: 个体迭代完成，共 {it+1} 轮，结果已保存。")

if __name__ == "__main__":
    main()
