#!/usr/bin/env python3
"""
07_giaga_driver_stream.py
———————————
• 将每轮 group-atlas 同时保存为
    ─ group_iter#.nii.gz  （体素 × int16）
    ─ group_iter#.txt     （flatten 一维文本）
"""

from pathlib import Path
import numpy as np, nibabel as nib
from tqdm import tqdm

from utils import WORKDIR, CFG, run, overlap_ratio   # 与旧脚本相同

# ------------------------------------------------------------
# 高斯年龄权重
# ------------------------------------------------------------
def compute_age_weights(ages, atlas_order):
    A_b = np.array([-0.1667,-0.125,-0.0833,-0.0417,0,0.0833,0.25,0.5,0.75,1,
                    1.5,2,4,6,8,10,12,14,16,18,20,30,40,50,60,70])
    A_m = np.array([-0.125,-0.0833,-0.0417,0,0.0833,0.25,0.5,0.75,1,1.5,2,4,
                     6,8,10,12,14,16,18,20,30,40,50,60,70,80])
    A_a = np.array([-0.0833,-0.0417,0,0.0417,0.1667,0.5,0.75,1,1.25,2,2.5,6,
                     8,10,12,14,16,18,20,22,40,50,60,70,80,80])
    c = A_m[atlas_order-1]
    scale = np.where(ages < c,
                     1.96/(c-A_b[atlas_order-1]),
                     1.96/(A_a[atlas_order-1]-c))
    norm  = (ages - c) * scale
    pdf   = np.exp(-0.5*norm**2)/np.sqrt(2*np.pi)
    return (pdf/pdf.sum()).astype(np.float32)

# ------------------------------------------------------------
def main():

    # ---------- 受试者列表 ----------
    subs = sorted([d.name for d in WORKDIR.iterdir()
                   if (d/'label_final.nii.gz').exists()])
    if not subs:
        print('[GIAGA] no subject labels found'); return
    N = len(subs)

    # ---------- 年龄权重 ----------
    age_list = WORKDIR/'Final_sublist'/f'Atlas{CFG["atlas_order"]}_subage.list'
    ages  = np.loadtxt(age_list)
    w_vec = compute_age_weights(ages, CFG['atlas_order'])
    print(f'[INFO] {N} subjects, Σweight={w_vec.sum():.3f}')

    # ---------- 目录 ----------
    atlas_dir = WORKDIR/'atlas'; atlas_dir.mkdir(exist_ok=True)

    # ---------- warp 每个 label_final → 标准空间 (一次性/增量皆可) ----------
    print('[Step] Warping individual labels to standard space')
    for s in tqdm(subs):
        lab = WORKDIR/s/'label_final.nii.gz'
        out = WORKDIR/s/'label_std.nii.gz'
        if out.exists():   # 若早已 warp 过可跳过
            continue
        run(f"applywarp --ref={CFG['fa_std']} --in={lab} "
            f"--warp={WORKDIR/s/'reg/fa2std_warpcoef.nii.gz'} "
            f"--out={out} --interp=nn")

    # ---------- 获取标准空间尺寸 ----------
    ref_hdr  = nib.load(str(CFG['fa_std']))
    H,W,D    = ref_hdr.shape
    img_aff  = ref_hdr.affine
    V        = H*W*D
    print(f'[INFO] standard grid = {H}×{W}×{D}  ({V:,} voxels)')

    overlap_thr   = CFG['overlap_thr']
    max_group_it  = CFG['group_iter_max']
    grp_prev_flat = None

    for gi in range(1, max_group_it+1):
        print(f'\n=== GROUP ITER {gi}/{max_group_it} ===')
    
        # -------- ① 流式加权投票 --------
        score = np.memmap('tmp_grp_score.bin', mode='w+',
                          dtype=np.float32, shape=(50, H, W, D))
        score[:] = 0.0
    
        for s_idx, s in enumerate(tqdm(subs, desc='Accum votes')):
            w = w_vec[s_idx]
            data = nib.load(str(WORKDIR/s/'label_std.nii.gz')).get_fdata().astype(np.int8)
            for k in range(1, 51):
                mask = (data == k)
                if mask.any():
                    score[k-1][mask] += w    # in-place 累加
    
        # -------- ② 取 argmax → 新 atlas（不再设阈值） --------
        grp_lab = np.argmax(score, axis=0) + 1  # 直接将得票最高的索引+1 作为网络标签
        del score  # 及时释放 mmap 文件
    
        # -------- ③ 保存 NIfTI + txt --------
        out_std = atlas_dir/f'group_iter{gi}.nii.gz'
        nib.save(nib.Nifti1Image(grp_lab.astype(np.int16), img_aff), out_std)
    
        txt_flat = grp_lab.reshape(-1)
        np.savetxt(atlas_dir/f'group_iter{gi}.txt', txt_flat, fmt='%d')
    
        # -------- ④ 收敛检测 --------
        if grp_prev_flat is not None:
            ov = overlap_ratio(grp_prev_flat, txt_flat, valid_mask=txt_flat>0)
            print(f'   overlap with prev = {ov:.4f}')
            if ov >= overlap_thr:
                print('<<< Converged. GIAGA finished >>>')
                break
        grp_prev_flat = txt_flat.copy()
    
        # -------- ⑤ 把新 atlas 回写个体空间 --------
        print('[Step] Warping atlas back to each subject')
        for s in tqdm(subs):
            run(f"applywarp --ref={WORKDIR/s/'dtifit/dtifit_FA.nii.gz'} "
                f"--in={out_std} --warp={WORKDIR/s/'reg/std2fa_warp.nii.gz'} "
                f"--out={WORKDIR/s/'atlas_iter.nii.gz'} --interp=nn")
    
        # -------- ⑥ 触发下一轮个体迭代 --------
        print('[Step] Trigger subject-level iteration')
        for s in tqdm(subs):
            run(f"python 06_iter_subject.py --sub {s}")
    
        print('[GIAGA] Done.')

if __name__ == '__main__':
    main()
