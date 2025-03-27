#!/usr/bin/env python3
import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

N_JOBS = 30  # Number of parallel processes (adjust based on CPU cores)

# Avoid internal multi-threading by setting OMP_NUM_THREADS to 1
os.environ["OMP_NUM_THREADS"] = "1"

def save_nonzero_coordinates(nifti_path, output_txt):
    """
    Read nonzero voxel coordinates (0-based) from the nifti file at nifti_path,
    and save them into the output_txt file.
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()  # float64
    nonzero_coords = np.argwhere(data != 0)  # shape (N, 3)

    # Write coordinates to txt file
    with open(output_txt, 'w') as f:
        for x, y, z in nonzero_coords:
            # Change to x+1, y+1, z+1 for 1-based indexing if needed
            f.write(f"{int(x)} {int(y)} {int(z)}\n")

    print(f"[INFO] Saved nonzero coordinates of {nifti_path} to {output_txt}")

def run_probtrack_for_roi(roi_idx):
    """
    Run the probtrackx2 command for a single ROI region,
    processing all seed voxels in the ROI at once with the --simple parameter.
    
    Parameters:
      roi_idx: int, ROI index (e.g., from 1 to 50)
    """
    seed_file = f"../data/seeds_txt_all/seed_region_{roi_idx}.txt"
    mask_file = "../data/nodif_brain_mask.nii.gz"
    samples = "merged"
    output_dir = f"../data/probtrack_old/ROI_{roi_idx}"
    seedref = "../data/nodif_brain_mask.nii.gz"
    
    cmd = (
        f"probtrackx2 --seed={seed_file} --mask={mask_file} "
        f"--samples={samples} --simple --dir={output_dir} --forcedir "
        f"--seedref={seedref} --opd"
    )
    print(f"[INFO] Running probtrackx2 for ROI {roi_idx}")
    subprocess.run(cmd, shell=True, check=True)

def merge_roi_fdt_paths(roi_idx):
    """
    Merge all nii.gz files starting with 'fdt_paths_' in the specified ROI folder
    into a single 4D file named ROI_{roi_idx}_merged_fdt_paths.nii.gz.
    """
    roi_folder = f"../data/probtrack_old/ROI_{roi_idx}"
    merged_file = f"../data/probtrack_old/ROI_{roi_idx}_merged_fdt_paths.nii.gz"

    fdt_paths_list = [os.path.join(roi_folder, fname) for fname in os.listdir(roi_folder)
                      if fname.startswith("fdt_paths_") and fname.endswith(".nii.gz")]
    if fdt_paths_list:
        cmd_merge = "fslmerge -t " + merged_file + " " + " ".join(fdt_paths_list)
        print(f"[INFO] Merging {len(fdt_paths_list)} fdt_paths files for ROI {roi_idx}")
        subprocess.run(cmd_merge, shell=True, check=True)
        print(f"[INFO] Merged file created: {merged_file}")
    else:
        print(f"[WARNING] ROI {roi_idx}: No fdt_paths files found, skipping merge.")

def process_roi(roi_idx):
    """
    For a single ROI, first run probtrackx2 and then merge the fdt_paths files.
    """
    try:
        run_probtrack_for_roi(roi_idx)
        merge_roi_fdt_paths(roi_idx)
        print(f"[INFO] ROI {roi_idx} processing completed.")
    except Exception as e:
        print(f"[ERROR] ROI {roi_idx} processing failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path where the data/ subfolder is already prepared (with data.nii.gz, nodif_brain_mask.nii.gz, bvals, bvecs)")
    parser.add_argument("--FMRIB_FA_TEMPLATE", default="$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz")
    parser.add_argument("--JHU50_SEED", default="$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz")
    args = parser.parse_args()

    data_path = args.data_path
    FMRIB_FA_TEMPLATE = args.FMRIB_FA_TEMPLATE
    JHU50_SEED = args.JHU50_SEED

    os.chdir(data_path)

    # Run bedpostx
    if os.path.isdir("data.bedpostX"):
        print(f"[WARNING] data.bedpostX already exists, skipping bedpostx")
    else:
        subprocess.run("bedpostx_gpu data -n 3", shell=True, check=True)
    
    # Run dtifit
    subprocess.run("dtifit -k data/data.nii.gz -o data/dtifit -m data/nodif_brain_mask.nii.gz -r data/bvecs -b data/bvals", shell=True, check=True)

    # Run FLIRT
    subprocess.run(f"flirt -in data/dtifit_FA.nii.gz -ref {FMRIB_FA_TEMPLATE} -out data/fa_standard_space.nii.gz -omat data/fa2standard.mat", shell=True, check=True)

    # Run FNIRT
    subprocess.run(f"fnirt --in=data/dtifit_FA.nii.gz --aff=data/fa2standard.mat --ref={FMRIB_FA_TEMPLATE} --iout=data/fa_standard_space_nonlin.nii.gz --cout=data/fa2standard_warp.nii.gz", shell=True, check=True)

    # Inverse warp
    subprocess.run("invwarp -w data/fa2standard_warp.nii.gz -r data/dtifit_FA.nii.gz -o data/JHU50_inverse_warp.nii.gz", shell=True, check=True)
    subprocess.run(f"applywarp -i {JHU50_SEED} -r data/dtifit_FA.nii.gz -w data/JHU50_inverse_warp.nii.gz -o data/JHU50_native_nn.nii.gz --interp=nn", shell=True, check=True)

    os.makedirs("data/seeds_txt_all", exist_ok=True)
    os.makedirs("data/seeds_region_all", exist_ok=True)

    # Create files for each seed region
    for i in range(1, 51):
        seed_out = f"data/seeds_region_all/seed_region_{i}.nii.gz"
        subprocess.run(f"fslmaths data/JHU50_native_nn.nii.gz -thr {i} -uthr {i} -bin {seed_out}", shell=True, check=True)

        # Save nonzero coordinates to seed_region_{i}.txt
        seed_txt = f"data/seeds_txt_all/seed_region_{i}.txt"
        save_nonzero_coordinates(seed_out, seed_txt)

    os.chdir("data.bedpostX")
    # Process each ROI voxel in parallel
    roi_indices = range(1, 51)
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = [executor.submit(process_roi, roi_idx) for roi_idx in roi_indices]
        # Wait for all tasks to complete
        for fut in futures:
            fut.result()
    print("[INFO] All ROI processing completed.")

if __name__ == "__main__":
    main()
