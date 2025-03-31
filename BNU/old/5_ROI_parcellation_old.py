#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import argparse
from sklearn.cluster import KMeans
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import simlr  # Import the implemented simlr module
import concurrent.futures

N_JOBS = 30

def spectral_clustering(adjacency_matrix, num_clusters):
    """
    A Python version of the SC3-based spectral clustering function, mimicking the MATLAB function.

    Parameters:
    ----------
    num_clusters : int
        The target number of clusters (k)
    adjacency_matrix : (n, n) ndarray
        The similarity matrix (e.g., obtained from matrix @ matrix.T) with shape (n, n)

    Returns:
    ----------
    cluster_labels : (n,) ndarray
        Cluster labels for each point (0..k-1)
    """
    # 1) Compute L_sym = D^-1/2 * (D - adjacency_matrix) * D^-1/2
    num_points = adjacency_matrix.shape[0]
    degree_array = np.sum(adjacency_matrix, axis=1)
    degree_matrix = np.diag(degree_array)
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Avoid division by zero
    degree_array[degree_array == 0] = 1e-12
    inv_sqrt_degree = 1.0 / np.sqrt(degree_array)
    inv_sqrt_degree_matrix = np.diag(inv_sqrt_degree)

    laplacian_sym = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix

    # 2) Compute the smallest (num_clusters+5) eigenvalues/eigenvectors
    kplus = min(num_clusters + 5, num_points)
    laplacian_sym_sp = sp.csr_matrix(laplacian_sym)
    eigen_values_all, eigen_vectors_all = spla.eigsh(laplacian_sym_sp, k=kplus, which='SM')

    # 3) Sort eigenvalues and select the appropriate eigenvectors
    index_sorted = np.argsort(eigen_values_all)
    sorted_values = eigen_values_all[index_sorted]

    nonzero_indices = np.where(np.abs(sorted_values) > 1e-12)[0]
    if len(nonzero_indices) < num_clusters:
        chosen_indices = index_sorted[:num_clusters]
    else:
        starting_idx = nonzero_indices[0]
        chosen_indices = index_sorted[starting_idx: starting_idx + num_clusters]

    eigen_vectors = eigen_vectors_all[:, chosen_indices]

    # 4) Normalize rows
    row_norms = np.linalg.norm(eigen_vectors, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-12
    normalized_vectors = eigen_vectors / row_norms

    # 5) Perform k-means clustering
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=300, random_state=0)
    cluster_labels = kmeans_model.fit_predict(normalized_vectors)

    return cluster_labels

def process_roi(i, roi_coord_folder, fourD_folder, conn_folder, outdir, method, max_cl_num):
    """
    Process a single ROI (seed index i). For each ROI, it:
      1) Loads the ROI coordinate file, connectivity matrix, and 4D file.
      2) For different numbers of clusters (from 2 to max_cl_num), it performs clustering
         using the specified method (spectral clustering, k-means, or simlr) and saves the result.
    """
    # Read the ROI coordinate file
    coord_file = os.path.join(roi_coord_folder, f"seed_region_{i}.txt")
    if not os.path.isfile(coord_file):
        print(f"[WARNING] {coord_file} does not exist, skipping ROI {i}")
        return
    coords = np.loadtxt(coord_file, dtype=int)
    n_voxels = coords.shape[0]
    if n_voxels == 0:
        print(f"[WARNING] {coord_file} is empty, skipping ROI {i}")
        return

    # Load the connectivity matrix file from the con_cor folder
    con_mat_path = os.path.join(conn_folder, f"con_matrix_seed_{i}.npy")
    if not os.path.isfile(con_mat_path):
        print(f"[WARNING] {con_mat_path} does not exist, skipping ROI {i}")
        return
    con_matrix = np.load(con_mat_path)  # Assumed shape: (n_voxels, M)

    # Read the 4D file to obtain spatial information (voxel coordinates and affine)
    fourD_file = os.path.join(fourD_folder, f"ROI_{i}_merged_fdt_paths.nii.gz")
    if not os.path.isfile(fourD_file):
        print(f"[WARNING] {fourD_file} does not exist, skipping ROI {i}")
        return
    ref_nii = nib.load(fourD_file)
    vol_shape = ref_nii.shape[:3]  # (X, Y, Z)
    affine = ref_nii.affine

    # For different numbers of clusters, perform clustering sequentially
    for k in range(2, max_cl_num + 1):
        out_nii_name = f"seed_{i}_{k}.nii.gz"
        out_nii_path = os.path.join(outdir, out_nii_name)
        if os.path.isfile(out_nii_path):
            print(f"[INFO] {out_nii_path} already exists, skipping.")
            continue

        print(f"[INFO] ROI {i} clustering: number of clusters = {k}, method = {method}")
        if method == "sc":
            # Construct the similarity matrix
            sim_mat = con_matrix @ con_matrix.T
            np.fill_diagonal(sim_mat, 0)
            labels = spectral_clustering(sim_mat, k)
        elif method == "kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(con_matrix)
        elif method == "simlr":
            labels = simlr.simlr_cluster(con_matrix, k)
        else:
            print(f"[ERROR] Unknown method {method}, skipping ROI {i}")
            continue

        # Generate the segmentation result (3D NIfTI) based on ROI coordinates and cluster labels
        cluster_img = np.zeros(vol_shape, dtype=np.int16)
        for idx_vox in range(n_voxels):
            x, y, z = coords[idx_vox]
            cluster_img[x, y, z] = labels[idx_vox] + 1  # Labels start from 0, so add 1

        out_nii = nib.Nifti1Image(cluster_img, affine)
        out_nii.to_filename(out_nii_path)
        print(f"[INFO] Clustering result saved: {out_nii_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Subject data root directory; should contain subdirectories like data/probtrack_old/ and data/seeds_txt_all/")
    parser.add_argument("--method", default="sc", choices=["sc", "kmeans", "simlr"],
                        help="Clustering method: sc (spectral clustering), kmeans, or simlr (placeholder), default=sc")
    parser.add_argument("--max_cl_num", type=int, default=12,
                        help="Maximum number of clusters (default: 12)")
    parser.add_argument("--start_seed", type=int, default=1,
                        help="Starting ROI index (default: 1)")
    parser.add_argument("--end_seed", type=int, default=50,
                        help="Ending ROI index (default: 50)")
    args = parser.parse_args()

    data_path  = args.data_path
    method     = args.method
    max_cl_num = args.max_cl_num
    start_seed = args.start_seed
    end_seed   = args.end_seed

    os.chdir(data_path)

    roi_coord_folder = os.path.join(data_path, "data", "seeds_txt_all")
    fourD_folder = os.path.join(data_path, "data", "probtrack_old")
    conn_folder = os.path.join(data_path, "data", "probtrack_old", "con_cor")
    outdir = os.path.join(data_path, "data", "probtrack_old", f"parcellation_{method}")
    os.makedirs(outdir, exist_ok=True)

    # Parallelize over ROIs (outer loop) only
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = []
        for i in range(start_seed, end_seed + 1):
            futures.append(
                executor.submit(process_roi, i, roi_coord_folder, fourD_folder, conn_folder, outdir, method, max_cl_num)
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] An error occurred: {e}")

    print("[INFO] ROI parcellation completed.")

if __name__ == "__main__":
    main()
