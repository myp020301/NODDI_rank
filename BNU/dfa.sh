#!/usr/bin/env bash

BASE_DIR="/data2/mayupeng/BNU"
datasets=( "BBP_sample" "Agility_sample" "MASIVAR_sample" )

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: ${dataset}"
    cd "${BASE_DIR}/${dataset}" || { echo "Error: Cannot access ${BASE_DIR}/${dataset}"; continue; }
    
    for sub_id in sub-*; do
        [ -d "${sub_id}" ] || continue

        #  DWI 路径
        possible_paths=( ${sub_id}/ses-*/dwi  "${sub_id}/dwi" )
        found=false
        for path in "${possible_paths[@]}"; do
            if [ -d "${path}" ]; then
                data_path="${path}"
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            echo "[Warning] No DWI directory found for ${sub_id}, skipping..."
            continue
        fi

        cd "${data_path}" || { echo "[Error] Cannot cd into ${data_path}"; continue; }

        # 查找文件
        bval_file=$(ls *desc-preproc_dwi.bval 2>/dev/null | head -n1)
        bvec_file=$(ls *desc-preproc_dwi.bvec 2>/dev/null | head -n1)
        dwi_file=$(ls *desc-preproc_dwi.nii* 2>/dev/null | head -n1)
        mask_file=$(ls *desc-brain_mask.nii* 2>/dev/null | head -n1)

        if [ -z "${bval_file}" ] || [ -z "${bvec_file}" ] || \
           [ -z "${dwi_file}" ] || [ -z "${mask_file}" ]; then
            echo "[Warning] Missing required files in $(pwd), skipping ${sub_id}..."
            cd "${BASE_DIR}/${dataset}"
            continue
        fi

        mkdir -p DFA
        cd DFA || { echo "[Error] Cannot enter DFA directory."; cd "${BASE_DIR}/${dataset}"; continue; }

        # ======== 以下为 dfa 的计算流程 ========
        dt TextFileOperator "../${bval_file}" --transpose -o b.txt
        dt TextFileOperator "../${bvec_file}" --transpose -o grad.txt

        echo "b.txt grad.txt ../${dwi_file}" > data.txt

        dt DWIToTensor data.txt dti.nii.gz --mask "../${mask_file}"

        dt TensorToFeature dti.nii.gz --v1 dti_v1.nii.gz --fathreshold 0.2 

        dt PeakToLocalFrame dti_v1.nii.gz dti_frame.nii.gz --type XYZ

        dt LocalFrameToFeature dti_frame.nii.gz \
            --splay dti_frame_splay.nii.gz \
            --bend dti_frame_bend.nii.gz \
            --twist dti_frame_twist.nii.gz \
            --distortion dti_frame_distortion.nii.gz

        echo "[DFA Done]: $(pwd)"

        # 返回到当前 dataset 目录准备下一个被试
        cd "${BASE_DIR}/${dataset}"
    done
done

echo "All subjects done!"

