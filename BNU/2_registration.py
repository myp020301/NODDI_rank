#!/usr/bin/env python3
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path to the data directory (contains the 'data' subfolder)")
    parser.add_argument("--FMRIB_FA_TEMPLATE", default="$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz",
                        help="Path to the FMRIB_FA_TEMPLATE")
    parser.add_argument("--JHU50_SEED", default="$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz",
                        help="Path to the JHU50_SEED")
    args = parser.parse_args()

    data_path = args.data_path
    FMRIB_FA_TEMPLATE = args.FMRIB_FA_TEMPLATE
    JHU50_SEED = args.JHU50_SEED

    os.chdir(data_path)

    # Run dtifit
    subprocess.run("dtifit -k data/data.nii.gz -o data/dtifit -m data/nodif_brain_mask.nii.gz -r data/bvecs -b data/bvals",
                   shell=True, check=True)
    # Run FLIRT
    subprocess.run(f"flirt -in data/dtifit_FA.nii.gz -ref {FMRIB_FA_TEMPLATE} -out data/fa_standard_space.nii.gz -omat data/fa2standard.mat",
                   shell=True, check=True)
    # Run FNIRT
    subprocess.run(f"fnirt --in=data/dtifit_FA.nii.gz --aff=data/fa2standard.mat --ref={FMRIB_FA_TEMPLATE} --iout=data/fa_standard_space_nonlin.nii.gz --cout=data/fa2standard_warp.nii.gz",
                   shell=True, check=True)
    # Inverse warp and apply warp
    subprocess.run("invwarp -w data/fa2standard_warp.nii.gz -r data/dtifit_FA.nii.gz -o data/JHU50_inverse_warp.nii.gz",
                   shell=True, check=True)
    subprocess.run(f"applywarp -i {JHU50_SEED} -r data/dtifit_FA.nii.gz -w data/JHU50_inverse_warp.nii.gz -o data/JHU50_native_nn.nii.gz --interp=nn",
                   shell=True, check=True)
    print("[INFO] Registration step completed.")

if __name__ == "__main__":
    main()
