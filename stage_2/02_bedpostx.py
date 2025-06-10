#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path
from utils import run, CFG,WORKDIR
import os 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub", required=True)
    args = p.parse_args()

    subdir = WORKDIR / args.sub
    bp_dir = subdir / "bedpostx"
    bp_in  = bp_dir / "data"
    bp_in.mkdir(parents=True, exist_ok=True)
    # 复制并重命名
    file_map = {
        "dwi.nii.gz": "data.nii.gz",
        "bvecs": "bvecs",
        "bvals": "bvals",
        "nodif_brain_mask.nii.gz": "nodif_brain_mask.nii.gz"
    }
    for src_name, dst_name in file_map.items():
        src = subdir / "dtifit" / src_name
        dst = bp_in / dst_name
        if not dst.exists():
            shutil.copy(src, dst)
    if os.path.isdir(os.path.join(bp_dir,"data.bedpostX")):
        print("[WARNING] data.bedpostX already exists, skipping bedpostx")
    else:
        run(f"bedpostx {bp_in} -n 3")
        print("[OK] bedpostx", args.sub)
    

if __name__ == "__main__":
    main()