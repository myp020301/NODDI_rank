#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import argparse
import concurrent.futures

# Set the number of parallel processes; adjust as needed, e.g., os.cpu_count()
N_JOBS = 30

def calc_matrix_for_seed(
    roi_coord_file: str,
    fourD_file: str,
    threshold: float,
    output_folder: str,
    seed_index: int
):
    """
    For a single ROI region (seed_index) corresponding to the 4D file,
    construct a connectivity matrix (con_matrix) of size (nVox x T),
    apply thresholding, remove all-zero columns, and compute the correlation matrix 
    (cor_matrix = con_matrix @ con_matrix.T). The results are then saved in output_folder.
    """
    # 1) Read ROI coordinates
    coords = np.loadtxt(roi_coord_file, dtype=int)  # shape: (nVox, 3)
    n_voxels = coords.shape[0]
    if n_voxels == 0:
        raise ValueError(f"[ERROR] No coordinates found in file {roi_coord_file}!")
    
    # 2) Load the 4D NIfTI data
    print(f"[INFO] Loading 4D file (ROI {seed_index}): {fourD_file}")
    nii4D = nib.load(fourD_file)
    vol4D = nii4D.get_fdata()  # shape: (X, Y, Z, T)
    if vol4D.ndim != 4:
        raise ValueError(f"[ERROR] File {fourD_file} is not 4D data!")
    X_dim, Y_dim, Z_dim, T = vol4D.shape
    print(f"[INFO] 4D data dimensions: ({X_dim}, {Y_dim}, {Z_dim}, {T}), Number of voxels in ROI: {n_voxels}")
    
    # 3) Construct connectivity matrix: extract the 4D time series for each voxel in the ROI
    con_matrix = np.zeros((n_voxels, T), dtype=np.float32)
    for i in range(n_voxels):
        x, y, z = coords[i]
        con_matrix[i, :] = vol4D[x, y, z, :]
    
    # 4) Apply thresholding: set values below threshold to 0 and remove columns that are entirely 0
    con_matrix[con_matrix < threshold] = 0
    col_max = con_matrix.max(axis=0)
    col_min = con_matrix.min(axis=0)
    keep_cols = ~((col_max == 0) & (col_min == 0))
    con_matrix = con_matrix[:, keep_cols]
    print(f"[INFO] After thresholding, con_matrix dimensions: {con_matrix.shape}")
    
    # 5) Compute the correlation matrix: cor_matrix = con_matrix @ con_matrix.T
    cor_matrix = con_matrix @ con_matrix.T  # shape: (n_voxels, n_voxels)
    
    # 6) Save the results to output_folder
    os.makedirs(output_folder, exist_ok=True)
    con_out = os.path.join(output_folder, f"con_matrix_seed_{seed_index}.npy")
    cor_out = os.path.join(output_folder, f"cor_matrix_seed_{seed_index}.npy")
    np.save(con_out, con_matrix)
    np.save(cor_out, cor_matrix)
    print(f"[INFO] ROI {seed_index} processing completed, results saved in: {output_folder}")

def main():
    """
    Main program:
      --data_path: Subject's working directory (should contain subdirectories such as data/seeds_txt_all and data/probtrack_old)
      --threshold: Threshold value (default: 10)
      --start_seed, --end_seed: ROI index range (default: 1 to 50)
    
    The script processes the coordinate files in data/seeds_txt_all and the 4D files in data/probtrack_old,
    generates connectivity and correlation matrices, and saves the results in the data/probtrack_old/con_cor/ directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Root directory of the subject's data; should contain subdirectories like data/seeds_txt_all/ and data/probtrack_old/")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Threshold value (default: 10)")
    parser.add_argument("--start_seed", type=int, default=1,
                        help="Starting ROI index (default: 1)")
    parser.add_argument("--end_seed", type=int, default=50,
                        help="Ending ROI index (default: 50)")
    args = parser.parse_args()
 
    data_path = args.data_path
    threshold = args.threshold
    start_seed = args.start_seed
    end_seed = args.end_seed
    os.chdir(data_path)

    roi_coord_folder = os.path.join(data_path, "data", "seeds_txt_all")
    probtrack_folder = os.path.join(data_path, "data", "probtrack_old")
    output_folder = os.path.join(probtrack_folder, "con_cor")

    # Use ProcessPoolExecutor to process each ROI's task in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = []
        for roi_idx in range(start_seed, end_seed + 1):
            roi_coord_file = os.path.join(roi_coord_folder, f"seed_region_{roi_idx}.txt")
            fourD_file = os.path.join(probtrack_folder, f"ROI_{roi_idx}_merged_fdt_paths.nii.gz")
            if not os.path.isfile(roi_coord_file):
                print(f"[WARNING] ROI coordinate file not found: {roi_coord_file}, skipping ROI {roi_idx}.")
                continue
            if not os.path.isfile(fourD_file):
                print(f"[WARNING] 4D file not found: {fourD_file}, skipping ROI {roi_idx}.")
                continue

            futures.append(executor.submit(
                calc_matrix_for_seed,
                roi_coord_file=roi_coord_file,
                fourD_file=fourD_file,
                threshold=threshold,
                output_folder=output_folder,
                seed_index=roi_idx
            ))
        
        # Wait for all tasks to complete and capture exceptions if any
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()
