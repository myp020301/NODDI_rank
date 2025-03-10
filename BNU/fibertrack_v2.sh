#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Description:
#   This script processes datasets by:
#   1) Running bedpostx to compute fiber orientations.
#   2) Computing FA maps using dtifit.
#   3) Registering the individual FA map to the standard FMRIB FA template 
#      using FLIRT and then FNIRT to generate a nonlinear warp (fa2standard_warp.nii.gz).
#   4) Inverse transforming the JHU48 seed region (in standard space) to each subject's native space.
#   5) Running probtrackx2_gpu with the native-space JHU48 seed as seed.
#
# Note: Further steps (e.g., group template creation) are not included.
# -----------------------------------------------------------------------------

# Exit script on any error
set -e

# Set your data root directory
BASE_DIR="/data2/mayupeng/BNU"
datasets=( "BBP_sample" "Agility_sample" "MASIVAR_sample" )

# FMRIB FA Template (standard space)
FMRIB_FA_TEMPLATE="$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz"

# JHU48 Seed regions in standard space (adjust the path if necessary)
JHU48_SEED="$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz"

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

        # 5) Check if bedpostx has already been run
        if [ -d "data.bedpostX" ]; then
            cd "${dataset_path}"
            echo "[WARNING] data.bedpostX already exists for ${sub_id}. Skipping bedpostx processing."
            continue
        fi

        # 6) Run bedpostx_gpu on the data folder to compute fiber orientations
        echo "[INFO] Running bedpostx_gpu for ${sub_id}..."
        bedpostx_gpu data

        # 7) Compute FA map using dtifit
        echo "[INFO] Running dtifit to compute FA map for ${sub_id}..."
        dtifit -k data/data.nii.gz \
               -o data/dtifit \
               -m data/nodif_brain_mask.nii.gz \
               -r data/bvecs \
               -b data/bvals
        # dtifit outputs include data/dtifit_FA.nii.gz

        # 8) Register individual FA map to the FMRIB FA template using FLIRT
        echo "[INFO] Running FLIRT registration for ${sub_id}..."
        flirt -in data/dtifit_FA.nii.gz \
              -ref $FMRIB_FA_TEMPLATE \
              -out data/fa_standard_space.nii.gz \
              -omat data/fa2standard.mat

        # 9) Perform nonlinear registration using FNIRT to generate warp field
        echo "[INFO] Running FNIRT nonlinear registration for ${sub_id}..."
        fnirt --in=data/dtifit_FA.nii.gz \
              --aff=data/fa2standard.mat \
              --ref=$FMRIB_FA_TEMPLATE \
              --iout=data/fa_standard_space_nonlin.nii.gz \
              --cout=data/fa2standard_warp.nii.gz

        # 10) Inverse transform the JHU48 seed region from standard space to native space.
        echo "[INFO] Inverse transforming JHU48 seed region to native space for ${sub_id}..."
        invwarp -w data/fa2standard_warp.nii.gz \
                -r data/dtifit_FA.nii.gz \
                -o data/jhu48_inverse_warp.nii.gz

        # Apply the inverse warp to JHU48 seed region
        applywarp -i $JHU48_SEED \
                  -r data/dtifit_FA.nii.gz \
                  -w data/jhu48_inverse_warp.nii.gz \
                  -o data/jhu48_native.nii.gz

        # 11) Enter the bedpostx output directory and run probtrackx2_gpu
        if [ -d "data.bedpostX" ]; then
            cd data.bedpostX
            echo "[INFO] Starting probtrackx2_gpu in $(pwd) for ${sub_id}..."
            probtrackx2_gpu \
                --seed=../data/jhu48_native.nii.gz \
                --mask=../data/nodif_brain_mask.nii.gz \
                --samples=merged \
                --dir=../data/probtrack_output \
                --forcedir \
                --opd
            cd ..
        else
            echo "[ERROR] bedpostx output folder 'data.bedpostX' not found for ${sub_id}."
        fi

        echo "[INFO] Subject ${sub_id} processing complete for bedpostx, dtifit, registration, seed transformation, and probtrackx2_gpu."

        # Return to the root of the dataset, preparing to process the next subject
        cd "${dataset_path}"
    done

    echo "[INFO] Finished processing dataset: ${dataset}"
done

echo "All tasks completed for all datasets!"
