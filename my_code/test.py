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
from testmodel import Mymodel
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(w):
    classname = w.__class__.__name__

    if classname.find('Conv') != -1:
        if hasattr(w, 'weight'):
            # nn.init.kaiming_normal_(w.weight, mode='fan_out', nonlinearity='relu')
          nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)
    if classname.find("Linear") != -1:
        if hasattr(w, "weight"):
            #torch.nn.init.normal_(w.weight, mean=0, std=0.02)
            torch.nn.init.xavier_normal_(w.weight)
        if hasattr(w, "bias") and w.bias is not None:
            nn.init.constant_(w.bias, 0)
            
            
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
'''
model_name  = os.path.join(directory, "model_epoch_12.pth")
scales_txt_name =  os.path.join(directory , "scales.txt")
scales = np.loadtxt(scales_txt_name)
'''
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    epochs = 50
    early_stop_patience = 5   # 允许验证损失未改善的最大轮次数
    min_delta = 0.001         # 验证损失的最小改善值
    start = time.time()
    best_val_loss = float('inf')  # 初始化最优验证损失为正无穷
    epochs_no_improve = 0         # 未改善的轮次数
    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = 0.0
        for step, (batch_x, batch_y) in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            output = model(batch_x)

            optimizer.zero_grad()
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                # 更新信息
                loop2.set_description(f"Epoch [{epoch}/{epochs}]")
                loop2.set_postfix(
                    loss=val_loss / (step + 1),
                )
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        model_save_path = os.path.join(directory, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        
        # 如果验证损失在连续 early_stop_patience 个轮次中未显著改善，则停止训练
        if epochs_no_improve >= early_stop_patience:
            print(f"验证损失在连续 {early_stop_patience} 个轮次中未改善，提前停止训练。")
            break

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
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


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
