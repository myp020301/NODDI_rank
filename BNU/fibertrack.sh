#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Description:
#   This script traverses three datasets (BBP, Agility, MASIVAR) within BASE_DIR,
#   locates each subject's DWI folder, creates a "data" subfolder with the files
#   renamed for bedpostx, then runs bedpostx_gpu followed by probtrackx2_gpu.
#   
# Folder structure under BASE_DIR:
#
# BNU_dataset
# ├── BBP_sample
# │   └── sub-BBPxxxx
# │       └── dwi
# │           ├── sub-BBPxxxx_space-T1w_desc-preproc_dwi.nii.gz
# │           ├── sub-BBPxxxx_space-T1w_desc-brain_mask.nii.gz
# │           ├── sub-BBPxxxx_space-T1w_desc-preproc_dwi.bval
# │           └── sub-BBPxxxx_space-T1w_desc-preproc_dwi.bvec
# ├── Agility_sample
# │   └── sub-001
# │       └── ses-baseline
# │           └── dwi
# │               ├── sub-001_ses-baseline_dir-AP_space-T1w_desc-preproc_dwi.nii.gz
# │               ├── sub-001_ses-baseline_dir-AP_space-T1w_desc-brain_mask.nii.gz
# │               ├── sub-001_ses-baseline_dir-AP_space-T1w_desc-preproc_dwi.bval
# │               └── sub-001_ses-baseline_dir-AP_space-T1w_desc-preproc_dwi.bvec
# └── MASIVAR_sample
#     └── sub-cIVs010
#         └── ses-xxx
#             └── dwi
#                 ├── sub-cIVs010_ses-s1Bx3_space-T1w_desc-preproc_dwi.nii.gz
#                 ├── sub-cIVs010_ses-s1Bx3_space-T1w_desc-brain_mask.nii.gz
#                 ├── sub-cIVs010_ses-s1Bx3_space-T1w_desc-preproc_dwi.bval
#                 └── sub-cIVs010_ses-s1Bx3_space-T1w_desc-preproc_dwi.bvec
#
# Usage:
#   1. Ensure bedpostx_gpu and probtrackx2_gpu are installed and in your PATH.
#   2. Adjust BASE_DIR to point to the parent directory containing BBP_sample, Agility_sample,
#      MASIVAR_sample folders.
#   3. Run: ./fibertract.sh
# -----------------------------------------------------------------------------

# Exit script on any error
set -e

# Set your data root directory
BASE_DIR="/data2/mayupeng/BNU"
datasets=( "Agility_sample")

for dataset in "${datasets[@]}"; do
    echo "[INFO] Processing dataset: ${dataset}"

    dataset_path="${BASE_DIR}/${dataset}"
    if [ ! -d "${dataset_path}" ]; then
        echo "[ERROR] Dataset directory not found: ${dataset_path}"
        continue
    fi
    cd "${dataset_path}"

    # Traverse all subjects (sub-*) in this dataset
    for sub_id in sub-*; do
        # Skip non-directory entries
        [ -d "${sub_id}" ] || { 
            echo "[WARNING] Not a directory: ${sub_id}. Skipping..."; 
            continue; 
        }

        echo "[INFO] Processing subject: ${sub_id}"

        # 1) Search for possible DWI paths (ses-*/dwi or directly dwi)
        possible_paths=( "${sub_id}/ses-"*/dwi "${sub_id}/dwi" )
        data_path=""
        found=false
        for path in "${possible_paths[@]}"; do
            if [ -d "${path}" ]; then
                data_path="${path}"
                found=true
                break
            fi
        done
        if [ "${found}" = false ] || [ -z "${data_path}" ]; then
            echo "[WARNING] No DWI directory found for ${sub_id}, skipping..."
            continue
        fi

        # 2) Enter the DWI directory
        cd "${dataset_path}/${data_path}" || {
            echo "[ERROR] Cannot enter DWI directory: ${data_path}"
            continue
        }

        # 3) Locate required files
        bval_file=$(ls *desc-preproc_dwi.bval 2>/dev/null | head -n1)
        bvec_file=$(ls *desc-preproc_dwi.bvec 2>/dev/null | head -n1)
        dwi_file=$( ls *desc-preproc_dwi.nii* 2>/dev/null | head -n1)
        mask_file=$(ls *desc-brain_mask.nii*  2>/dev/null | head -n1)

        # If any file is missing, skip
        if [ -z "${bval_file}" ] || [ -z "${bvec_file}" ] || \
           [ -z "${dwi_file}"  ] || [ -z "${mask_file}" ]; then
            echo "[WARNING] Missing bval/bvec/dwi/mask in $(pwd), skipping..."
            cd "${dataset_path}"
            continue
        fi

        # 4) Create a data folder
        mkdir -p data
        
        # Copy files required by bedpostx into the data folder and rename them:
        # data.nii.gz, nodif_brain_mask.nii.gz, bvals, bvecs
        cp "${dwi_file}"   data/data.nii.gz
        cp "${mask_file}"  data/nodif_brain_mask.nii.gz
        cp "${bval_file}"  data/bvals
        cp "${bvec_file}"  data/bvecs

        echo "[INFO] Files prepared in data/ for subject ${sub_id}"

        # 5) Run bedpostx_gpu data in the current DWI directory
        #    This will generate data.bedpostX at the same level as data
        echo "[INFO] Running bedpostx_gpu for ${sub_id}..."
        bedpostx_gpu data

        # 6) Enter the sibling directory data.bedpostX and run probtrackx2_gpu
        if [ -d "data.bedpostX" ]; then
            cd data.bedpostX
            echo "[INFO] Starting probtrackx2_gpu in $(pwd)..."

            # Example probtrackx2_gpu command, modify parameters as needed
            probtrackx2_gpu \
                --seed=nodif_brain_mask.nii.gz \
                --mask=nodif_brain_mask.nii.gz \
                --samples=merged \
                --out=fdt_paths \
                --opd

            echo "[INFO] Finished probtrackx2_gpu for subject ${sub_id}"
            # Return to the DWI directory
            cd ..
        else
            echo "[ERROR] bedpostx output folder 'data.bedpostX' not found for ${sub_id}."
        fi

        # Return to the root of the dataset, preparing to process the next subject
        cd "${dataset_path}"
    done

    echo "[INFO] Finished processing dataset: ${dataset}"
done

echo "All tasks completed for all datasets!"
