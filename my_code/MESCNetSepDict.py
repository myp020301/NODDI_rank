#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import nibabel as nib
import numpy as np
from input_parser import input_parser
from data_loader import load_training_matrix, load_test_matrix, data_combine_matrix
from models import mesc_sep_dict
import nibabel.processing

import time
from tqdm import trange
from keras.models import load_model
import matplotlib.pyplot as plt

def train_generator():
    train_size = int(0.9 * len(dwiTraining))  # 计算训练数据集的大小
    while True:
        for i in range(0, train_size, 128):  # 以128为批次大小
            yield dwiTraining[i:i+128, :, :], featurePatchTraining[i:i+128]

def validation_generator():
    validation_size = len(dwiTraining) - int(0.9 * len(dwiTraining))  # 计算验证数据集的大小
    start_index = int(0.9 * len(dwiTraining))  # 验证数据集的起始索引
    while True:
        for i in range(start_index, len(dwiTraining), 128):
            end_index = i + 128 if i + 128 < len(dwiTraining) else len(dwiTraining)
            yield dwiTraining[i:end_index, :, :], featurePatchTraining[i:end_index]



# %%
(
    dwinames,
    masknames,
    featurenumbers,
    featurenames,
    testdwinames,
    testmasknames,
    patch_size_low,
    patch_size_high,
    upsample,
    nDictQ,
    nDictS,
    directory,
) = input_parser(sys.argv)

# %%
if os.path.exists(directory) == False:
    os.mkdir(directory)

start = time.time()
print("Loading")

with open(dwinames) as f:
    allDwiNames = f.readlines()
with open(masknames) as f:
    allMaskNames = f.readlines()
allFeatureNames = []
for feature_index in range(featurenumbers):
    tempFeatureNames = None
    with open(featurenames[feature_index]) as f:
        tempFeatureNames = f.readlines()
    allFeatureNames.append(tempFeatureNames)
allDwiNames = [x.strip("\n") for x in allDwiNames]
allMaskNames = [x.strip("\n") for x in allMaskNames]
for feature_index in range(featurenumbers):
    allFeatureNames[feature_index] = [
        x.strip("\n") for x in allFeatureNames[feature_index]
    ]
    
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

# %%
# input ouput
dwiTraining, featurePatchTraining, scales = load_training_matrix(
    allDwiNames,
    allMaskNames,
    allFeatureNames,
    featurenumbers,
    patch_size_high,
    patch_size_low,
    upsample,
)
# %%
regressor_model_name = os.path.join(directory, "mesc_sep_dict_regressor.h5")
scales_txt_name = os.path.join(directory, "scales.txt")
np.savetxt(scales_txt_name, scales)

if os.path.exists(regressor_model_name) == False:
    regressor = mesc_sep_dict(
        dwiTraining.shape[1:], featurePatchTraining.shape[1:], nDictQ, nDictS
    )
    epoch = 10
    

    # 计算每个时期结束前从生成器中产生的步骤（样本批次）总数
    steps_per_epoch = int(0.9 * len(dwiTraining)) // 128
    
    # 计算从验证生成器产生的步骤（样本批次）总数
    validation_steps = (len(dwiTraining) - int(0.9 * len(dwiTraining))) // 128
    
    # 使用 fit_generator
    hist = regressor.fit_generator(
        generator=train_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        verbose=1,
        validation_data=validation_generator(),
        validation_steps=validation_steps
        #callbacks=[PrintWeightsCallback()]
    )
    
    print(hist.history)
    end = time.time()
    print("Training took ", (end - start))
     # 绘制loss图像
    plt.plot(hist.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
    regressor.save(regressor_model_name)
else:
    regressor = load_model(regressor_model_name)
    


# %%###### Test #######
print("Test Phase")

start = time.time()
with open(testdwinames) as f:
    allTestDwiNames = f.readlines()
with open(testmasknames) as f:
    allTestMaskNames = f.readlines()

allTestDwiNames = [x.strip("\n") for x in allTestDwiNames]
allTestMaskNames = [x.strip("\n") for x in allTestMaskNames]



# for iMask in progressbar.progressbar(range(len(allTestDwiNames))):
for iMask in trange(len(allTestDwiNames)):
    
    dwi_nii = nib.load(allTestDwiNames[iMask])
    dwi = dwi_nii.get_fdata()
    mask_nii = nib.load(allTestMaskNames[iMask])
    mask = mask_nii.get_fdata()

    dwiTest, patchCornerList = load_test_matrix(
        dwi, mask, patch_size_high, patch_size_low, upsample
    )
    featureList = regressor.predict(dwiTest)
    
    features = data_combine_matrix(
        featureList, mask.shape, upsample, patch_size_high, patchCornerList, scales
    )
    
    mask_upsampled_nii = nibabel.processing.resample_to_output(
        mask_nii,
        (
            mask_nii.header.get_zooms()[0] / upsample,
            mask_nii.header.get_zooms()[1] / upsample,
            mask_nii.header.get_zooms()[2] / upsample,
        ),
    )
    
    hdr = dwi_nii.header
    hdr.set_qform(mask_upsampled_nii.header.get_qform())
    for feature_index in range(featurenumbers):
        feature_nii = nib.Nifti1Image(
            features[:, :, :, feature_index], hdr.get_base_affine(), hdr
        )
        feature_name = os.path.join(
            directory,
            "MESC_sep_dict_feature_"
            + "%02d" % feature_index
            + "_sub_"
            + "%02d" % iMask
            + ".nii.gz",
        )
        feature_nii.to_filename(feature_name)


end = time.time()
print("Test took ", (end - start))
