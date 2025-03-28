#!/usr/bin/env python3
import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

N_JOBS = 30  # Number of parallel processes (adjust based on CPU cores)
os.environ["OMP_NUM_THREADS"] = "1"  # Avoid internal multi-threading

def save_nonzero_coordinates(nifti_path, output_txt):
    """
    Read nonzero voxel coordinates (0-based) from the nifti file at nifti_path,
    and save them into the output_txt file.
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    nonzero_coords = np.argwhere(data != 0)
    with open(output_txt, 'w') as f:
        for x, y, z in nonzero_coords:
            f.write(f"{int(x)} {int(y)} {int(z)}\n")
    print(f"[INFO] Saved nonzero coordinates of {nifti_path} to {output_txt}")

def run_probtrack_for_roi(roi_idx):
    """
    Run the probtrackx2 command for a single ROI region,
    processing all seed voxels in the ROI at once with the --simple parameter.
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
    into a single 4D file.
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
    For a single ROI, run probtrackx2 and then merge the fdt_paths files.
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
                        help="Path where the data/ subfolder is prepared")
    args = parser.parse_args()
    data_path = args.data_path
    os.chdir(data_path)
    
    os.makedirs("data/seeds_txt_all", exist_ok=True)
    os.makedirs("data/seeds_region_all", exist_ok=True)
    
    # Create seed region files and save nonzero coordinates
    for i in range(1, 51):
        seed_out = f"data/seeds_region_all/seed_region_{i}.nii.gz"
        subprocess.run(f"fslmaths data/JHU50_native_nn.nii.gz -thr {i} -uthr {i} -bin {seed_out}",
                       shell=True, check=True)
        seed_txt = f"data/seeds_txt_all/seed_region_{i}.txt"
        save_nonzero_coordinates(seed_out, seed_txt)
    
    os.chdir("data.bedpostX")
    roi_indices = range(1, 51)
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = [executor.submit(process_roi, roi_idx) for roi_idx in roi_indices]
        for fut in futures:
            fut.result()
    print("[INFO] All ROI processing completed.")

if __name__ == "__main__":
    main()
