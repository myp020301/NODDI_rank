#!/usr/bin/env python3
"""
05_conn_vectors.py  —— 仅下采样 fdt_paths，seed 坐标保留原始 2 mm 空间
-------------------------------------------------------------
1. 对每个 fdt_paths_*：
     • 将值 < 2 的体素置 0 → 去噪
     • 下采样到 5×5×5 mm³ → 获得一行向量（长度 V）
     • 如果向量 sum < 2，则丢弃该 seed
2. “seed_lin.npy” 存储每个保留 seed 在 原始 2 mm 体积 中的线性索引，用于后续回写标签
3. 输出：
     vec_vox.npy   → 形状 (N_seed × V)
     seed_lin.npy  → 形状 (N_seed,)
"""
import argparse
import numpy as np
import nibabel as nib
import nibabel.processing as nbproc
from pathlib import Path
from utils import WORKDIR

VOX_SZ   = (5, 5, 5)   # 下采样体素大小 (mm)
THR_VOX  = 2          # 体素阈值：<2 归零
THR_SEED = 2          # seed 过滤阈值：sum <2 丢弃

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--sub", required=True, help="Subject ID (e.g. HC_001)")
    args = pa.parse_args()
    sdir     = WORKDIR / args.sub
    prob_dir = sdir / "probtrack"
    if not prob_dir.exists():
        print(f"[WARN] {args.sub}: probtrack 目录不存在，跳过")
        return

    # 用任意一条 fdt_paths_* 确定下采样后向量长度 V
    sample = next(prob_dir.rglob("fdt_paths_*"), None)
    if sample is None:
        print(f"[WARN] {args.sub}: 未找到任何 fdt_paths_* 文件，跳过")
        return
    shape_ds = nbproc.resample_to_output(
        nib.load(str(sample)), voxel_sizes=VOX_SZ, order=1
    ).shape
    V = int(np.prod(shape_ds))
    print(f"[INFO] 下采样后体积尺寸 {shape_ds}，向量维度 {V}")

    vec_vox, seed_lin = [], []

    for roi in range(1, 51):
        roi_dir = prob_dir / f"ROI_{roi}"
        if not roi_dir.exists():
            continue
        paths = sorted(roi_dir.glob("fdt_paths_*"))
        if not paths:
            continue

        for p in paths:
            img  = nib.load(str(p))
            data = img.get_fdata()
            # （1）体素阈值：<2 归零
            data[data < THR_VOX] = 0

            # （2）下采样到 5 mm³
            img_thr = nib.Nifti1Image(data, img.affine)
            img_ds  = nbproc.resample_to_output(
                          img_thr, voxel_sizes=VOX_SZ, order=1)
            v = img_ds.get_fdata().reshape(-1).astype(np.float32)

            # （3）seed 过滤：sum <2 丢弃
            if v.sum() < THR_SEED:
                continue

            vec_vox.append(v)

            # —— 计算 seed 在原始 2 mm 体积 中的线性索引 seed_lin —— 
            # 文件名形如 "fdt_paths_59_56_14.nii.gz"
            fname = p.name
            if not fname.endswith(".nii.gz"):
                raise RuntimeError(f"文件名格式不符：{fname}")
            core = fname[:-7]  # 去掉 ".nii.gz"，得到 "fdt_paths_59_56_14"
            parts = core.split("_")
            try:
                x = int(parts[-3])
                y = int(parts[-2])
                z = int(parts[-1])
            except ValueError:
                raise RuntimeError(f"无法从文件名解析坐标：{fname}")
            # 原始 2 mm 体积尺寸
            orig_shape = img.shape   # 例如 (125, 125, 75)
            lin = (x * orig_shape[1] + y) * orig_shape[2] + z
            seed_lin.append(lin)

    if not vec_vox:
        print(f"[WARN] {args.sub}: 所有 seed 都被过滤，vec_vox 为空")
        return

    # 保存结果
    np.save(sdir / "vec_vox.npy",  np.stack(vec_vox, 0))
    np.save(sdir / "seed_lin.npy", np.array(seed_lin, np.int64))
    print(f"[OK] {args.sub}: vec_vox 大小 {len(vec_vox)}×{V}，seed_lin 已保存")

if __name__ == "__main__":
    main()
