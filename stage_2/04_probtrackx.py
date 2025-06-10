#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from glob import glob
from utils import run, CFG,WORKDIR

def save_seed_txt(mask: Path, out: Path):
    """Save nonzero voxel coordinates of mask into a text file."""
    data = nib.load(str(mask)).get_fdata()
    idx = np.argwhere(data > 0)
    np.savetxt(str(out), idx, fmt="%d")

def main():
    parser = argparse.ArgumentParser(description="Run probtrackx2 for each ROI with --opd, skipping if already complete")
    parser.add_argument("--sub", required=True, help="Subject ID (sub-xxxx)")
    args = parser.parse_args()

    sdir = WORKDIR / args.sub
    mask = sdir / "bedpostx/data" / "nodif_brain_mask.nii.gz"
    bpx_dir = sdir / "bedpostx" / "data.bedpostX" / "merged"

    for k in range(1, 51):
        roi_mask = sdir / "roi_masks" / f"roi_{k}.nii.gz"
        if not roi_mask.exists():
            continue

        # 1) prepare seed TXT
        seed_dir = sdir / "seeds"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_txt = seed_dir / f"seed_{k}.txt"
        save_seed_txt(roi_mask, seed_txt)
        n_seeds = sum(1 for _ in open(seed_txt, 'r'))

        # 2) check existing fdt_paths files
        outdir = sdir / "probtrack" / f"ROI_{k}"
        outdir.mkdir(parents=True, exist_ok=True)
        existing = glob(str(outdir / "fdt_paths_*"))
        if len(existing) == n_seeds:
            print(f"[SKIP] ROI {k}: found {len(existing)} fdt_paths files (matches {n_seeds} seeds)")
            continue
        # if partial results exist, remove them
        for f in existing:
            Path(f).unlink()

        # 3) run probtrackx2
        cmd = (
            f"probtrackx2 --seed={seed_txt} --seedref={mask} "
            f"-l --simple --opd --pd "
            f"--nsamples={CFG['nsamples']} --nsteps={CFG['nsteps']} "
            f"--steplength={CFG['steplength']} --cthr={CFG['cthr']} "
            f"--mask={mask} --dir={outdir} --forcedir --samples={bpx_dir}"
        )
        run(cmd)
        print(f"[DONE] ROI {k}: probtrackx2 finished")

    print("[OK] probtrackx for subject", args.sub)

if __name__ == "__main__":
    main()
