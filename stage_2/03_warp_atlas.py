#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path
from utils import run, CFG ,WORKDIR
import nibabel as nib
import numpy as np

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--sub", required=True)
    args = pa.parse_args()
    subdir = WORKDIR / args.sub
    regdir = subdir / "reg"; regdir.mkdir(exist_ok=True)

    fa = subdir / "dtifit/dtifit_FA.nii.gz"
    aff = regdir / "fa2std_aff.mat"
    warp= regdir / "fa2std_warpcoef.nii.gz"
    inv = regdir / "std2fa_warp.nii.gz"

    if not warp.exists():
        run(f"flirt -in {fa} -ref {CFG['fa_std']} -omat {aff} "
            f"-out {regdir/'fa2std_aff.nii.gz'}")
        run(f"fnirt --in={fa} --aff={aff} --ref={CFG['fa_std']} "
            f"--cout={warp}")
        run(f"invwarp -w {warp} -r {fa} -o {inv}")

    out_atlas = subdir / "atlas" / "JHU50_sub.nii.gz"
    out_atlas.parent.mkdir(exist_ok=True)
    run(f"applywarp --ref={fa} --in={CFG['atlas_std']} "
        f"--warp={inv} --out={out_atlas} --interp=nn")

    # 分割出 50 ROI mask
    roi_mask_dir = subdir / "roi_masks"
    roi_mask_dir.mkdir(exist_ok=True)
    lab_img = nib.load(out_atlas)
    lab = lab_img.get_fdata()
    affine = lab_img.affine
    for k in range(1, 51):
        m = (lab == k).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, affine), roi_mask_dir / f"roi_{k}.nii.gz")
    print("[OK] atlas warp & masks", args.sub)

if __name__ == "__main__":
    main()