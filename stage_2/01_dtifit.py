#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from utils import run, WORKDIR, DATASET_DIR

def main():
    parser = argparse.ArgumentParser(description="Run dtifit on DWI data with fixed filenames")
    parser.add_argument("--sub", required=True, help="Subject ID (e.g. HC_001)")
    args = parser.parse_args()

    raw_dir = DATASET_DIR / args.sub
    out_dir = WORKDIR  / args.sub / "dtifit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 映射原始文件到 dtifit 所需命名
    file_map = {
        raw_dir / "data.nii.gz":            out_dir / "dwi.nii.gz",
        raw_dir / "dti.bvec":               out_dir / "bvecs",
        raw_dir / "dti.bval":              out_dir / "bvals",
        raw_dir / "nodif_brain_mask.nii.gz": out_dir / "nodif_brain_mask.nii.gz",
    }

    # 复制并重命名
    for src, dst in file_map.items():
        if not src.exists():
            raise FileNotFoundError(f"[ERROR] Required file not found: {src}")
        shutil.copy(src, dst)

    # 运行 dtifit
    run(
        f"dtifit -k {out_dir/'dwi.nii.gz'} "
        f"-o {out_dir/'dtifit'} "
        f"-m {out_dir/'nodif_brain_mask.nii.gz'} "
        f"-r {out_dir/'bvecs'} -b {out_dir/'bvals'}"
    )

    print(f"[OK] dtifit completed for {args.sub}")

if __name__ == "__main__":
    main()
