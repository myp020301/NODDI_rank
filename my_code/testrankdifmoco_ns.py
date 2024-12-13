import sys
import os
import nibabel as nib
import numpy as np
from input_parser import input_parser
from data_loader import load_training_matrix, load_test_matrix, data_combine_matrix
import nibabel.processing
from torch.utils.data import random_split
import time
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from rankmoco_loss import rank_moco_loss
from differentmoco_loss import different_moco_loss
from testmodel_linear_v2 import Mymodel

import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def weights_init(w):
    classname = w.__class__.__name__

    # if classname.find('Conv') != -1:
    # ? ? if hasattr(w, 'weight'):
    # ? ? ? ? # nn.init.kaiming_normal_(w.weight, mode='fan_out', nonlinearity='relu')
    # ? ? ? ? nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='leaky_relu')
    # ? ? if hasattr(w, 'bias') and w.bias is not None:
    # ? ? ? ? ? ? nn.init.constant_(w.bias, 0)
    if classname.find("Linear") != -1:
        if hasattr(w, "weight"):
            #torch.nn.init.normal_(w.weight, mean=0, std=0.02)
            torch.nn.init.xavier_normal_(w.weight)
        if hasattr(w, "bias") and w.bias is not None:
            nn.init.constant_(w.bias, 0)
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
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
print(dwiTraining.shape)
print(featurePatchTraining.shape)

model_name  = os.path.join(directory, "mesc_sep_dict_regressor.pth")

scales_txt_name =  os.path.join(directory , "scales.txt")
np.savetxt(scales_txt_name, scales)  # 保存 scales

coe_rank_moco = 1
coe_dif = 1

print("coe_rank_moco",coe_rank_moco)
print("coe_dif_moco",coe_dif)

if not os.path.exists(model_name):
    model = Mymodel().to(device)
    model.apply(weights_init)

    # 数据加载
    X_train_tensor = torch.from_numpy(dwiTraining).float()
    y_train_tensor = torch.from_numpy(featurePatchTraining).float()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # 分割数据集为训练集和验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    criterion = torch.nn.MSELoss().to(device)
    criterion_rank_moco = rank_moco_loss(
        sorter_checkpoint_path="/data2/mayupeng/Tied_rank_best_lstmla_slen_128.pth.tar"
    ).to(device)
    criterion_dif = different_moco_loss().to(device)
    epochs = 10
    max_norm = 1.0
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = 0.0
        for step, (batch_x, batch_y) in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if batch_x.size(0) != 64:
                continue
            output = model(batch_x)

            optimizer.zero_grad()
            loss1 = criterion(output, batch_y)
            loss2 = criterion_rank_moco(output[:, 0, :], batch_y[:, 0, :]) + criterion_rank_moco(output[:, 1, :], batch_y[:, 1, :]) +criterion_rank_moco(output[:, 2, :], batch_y[:, 2, :])
            loss3 = criterion_dif(output, batch_y)
            loss = loss1  + coe_rank_moco * loss2  + coe_dif * loss3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            running_loss += loss.item()
            # 更新信息
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(
                loss=running_loss / (step + 1),
            )

        model.eval()
        val_loss = 0.0
        loop2 = tqdm(enumerate(val_loader), total=len(val_loader))
        with torch.no_grad():
            for step, (batch_x, batch_y) in loop2:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                if batch_x.size(0) != 64:
                   continue
                output = model(batch_x)

                loss1 = criterion(output, batch_y)
                loss2 = criterion_rank_moco(output[:, 0, :], batch_y[:, 0, :]) + criterion_rank_moco(output[:, 1, :], batch_y[:, 1, :]) +criterion_rank_moco(output[:, 2, :], batch_y[:, 2, :])
                loss3 = criterion_dif(output, batch_y)
                loss = loss1  + coe_rank_moco * loss2  + coe_dif * loss3
                
                val_loss += loss.item()
                # 更新信息
                loop2.set_description(f"Epoch [{epoch}/{epochs}]")
                loop2.set_postfix(
                    loss=val_loss / (step + 1),
                )
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        model_save_path = os.path.join(directory, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)           
    end = time.time()
    print("Training took ", (end - start))

    # 保存模型
    torch.save(model.state_dict(), model_name)
else:
    # 加载模型
    model = Mymodel().to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

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
    dwiTest_tensor = torch.from_numpy(dwiTest).float()

    test_dataset = TensorDataset(dwiTest_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model.eval()
    featureList_tensor = None
    
    loop2 = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for step, (x) in loop2:
            output = model(x[0].to(device))
            if featureList_tensor is None:
                featureList_tensor = output
            else:
                featureList_tensor = torch.vstack([featureList_tensor, output])

    # 将预测结果转换回 NumPy 数组
    featureList = featureList_tensor.cpu().numpy()

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
            features[:, :, :, feature_index], dwi_nii.affine, hdr
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
