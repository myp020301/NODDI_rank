"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2019 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2019, June).
    SoDeep: A Sorting Deep Net to Learn Ranking Loss Surrogates.
    In Proceedings of CVPR

Author: Martin Engilberge
"""
import argparse
from input_parser import input_parser
import os
import time
import torch
import torch.nn as nn
from data_loader import load_training_matrix
from dataset import SeqDataset
from model import model_loader
from torch.utils.data import DataLoader, SubsetRandomSampler,Dataset,ConcatDataset
from torch.optim.lr_scheduler import StepLR
from scipy import stats
from tensorboardX import SummaryWriter
from utils import AverageMeter, save_checkpoint, log_epoch, count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_tiedrank(batch_score, dim=0):
    batch_score = batch_score
    rank = stats.rankdata(batch_score, method='average')
    rank = (rank - 1) * -1 + batch_score.size
    rank = torch.from_numpy(rank).float()
    rank = rank / batch_score.size
    return rank

class FeaturePatchDataset(Dataset):
    def __init__(self, features, seq_len):
        self.features = features
        self.seq_len = seq_len
        self.num_sequences = min(len(features) // seq_len, 400000)  # 计算有多少个序列，取较小值

    def __getitem__(self, index):
        start_idx = index * self.seq_len
        end_idx = start_idx + self.seq_len

        if end_idx > len(self.features):
            # Skip this sample if it would result in an incomplete sequence
            print(f"Skipping index {index} due to insufficient data.")
            return torch.FloatTensor([]), torch.FloatTensor([])  # 返回空张量，这会导致 DataLoader 跳过该样本

        feature_seq = self.features[start_idx:end_idx]
        feature_seq = feature_seq.squeeze(-1)  # 去掉最后一维，使其形状为 [seq_len]
        rank = get_tiedrank(feature_seq)
        return torch.FloatTensor(feature_seq), torch.FloatTensor(rank)

    def __len__(self):
        return self.num_sequences

    
def train(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (s, r) in enumerate(train_loader):

        seq_in, rank_in = s.float().to(device, non_blocking=True), r.float().to(device, non_blocking=True)
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        rank_hat = model(seq_in)
        loss = criterion(rank_hat, rank_in)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), seq_in.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses), end="\r")

    print('Train: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i + 1, len(train_loader), batch_time=batch_time,
              data_time=data_time, loss=losses), end="\n")

    return losses.avg, batch_time.avg, data_time.avg


def validate(val_loader, model, criterion, print_freq=1):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (s, r) in enumerate(val_loader):

        seq_in, rank_in = s.float().to(device, non_blocking=True), r.float().to(device, non_blocking=True)
        data_time.update(time.time() - end)

        with torch.set_grad_enabled(False):
            rank_hat = model(seq_in)
            loss = criterion(rank_hat, rank_in)

        losses.update(loss.item(), seq_in.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i + 1, len(val_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses), end="\r")

    print('Val: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              i + 1, len(val_loader), batch_time=batch_time,
              data_time=data_time, loss=losses), end="\n")

    return losses.avg, batch_time.avg, data_time.avg


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--dwinames', help='File containing the names of DWI files')
    parser.add_argument('--masknames', help='File containing the names of mask files')
    parser.add_argument('--featurenumbers', type=int, help='Number of feature files')
    parser.add_argument('--featurenames', nargs='+', help='Files containing the names of feature files')
    parser.add_argument('--testdwinames', help='File containing the names of test DWI files')
    parser.add_argument('--testmasknames', help='File containing the names of test mask files')
    parser.add_argument('--patch_size_low', type=int, help='Lower bound of patch size')
    parser.add_argument('--patch_size_high', type=int, help='Upper bound of patch size')
    parser.add_argument('--upsample', type=int, help='Upsample factor')

    parser.add_argument("-n", '--name', default="model", help='Name of the model')
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=256)
    parser.add_argument("-lr", dest="lr", help="Initialization of the learning rate", type=float, default=0.001)
    parser.add_argument("-lrs", dest="lr_steps", help="Number of epochs to step down LR", type=int, default=70)
    parser.add_argument("-mepoch", dest="mepoch", help="Max epoch", type=int, default=400)
    parser.add_argument("-pf", dest="print_frequency", help="Number of element processed between print", type=int, default=100)
    parser.add_argument("-slen", dest="seq_len", help="lenght of the sequence process by the ranker", type=int, default=128)
    parser.add_argument("-d", dest="dist", help="index of a single distribution for dataset if None all the distribution will be used.", default=None)
    parser.add_argument('-m', dest="model_type", help="Specify which model to use. (lstm, grus, gruc, grup, exa, lstmla, lstme, mlp, cnn) ", default='lstmla')

    args = parser.parse_args()
    # Create the output directory if it doesn't exist
    
    start = time.time()
    print("Loading")
    
    # Load the file names
    with open(args.dwinames) as f:
        allDwiNames = f.readlines()
    with open(args.masknames) as f:
        allMaskNames = f.readlines()
    allFeatureNames = []
    for feature_file in args.featurenames:
        with open(feature_file) as f:
            tempFeatureNames = f.readlines()
        allFeatureNames.append(tempFeatureNames)
    
    # Strip newline characters from file names
    allDwiNames = [x.strip("\n") for x in allDwiNames]
    allMaskNames = [x.strip("\n") for x in allMaskNames]
    allFeatureNames = [[x.strip("\n") for x in feature_list] for feature_list in allFeatureNames]
    
    dwiTraining, featurePatchTraining, scales = load_training_matrix(
    allDwiNames,
    allMaskNames,
    allFeatureNames,
    args.featurenumbers,
    args.patch_size_high,
    args.patch_size_low,
    args.upsample,
    )
    # 分离三个指标，创建三个单独的 FeaturePatchDataset 数据集
    datasets = []
    for i in range(3):
        single_feature_data = featurePatchTraining[:, i, :]  # 提取第 i 个指标
        dataset = FeaturePatchDataset(single_feature_data, args.seq_len)
        datasets.append(dataset)

    # 合并所有指标的数据集
    combined_real_dataset = ConcatDataset(datasets)

    print("Using GPUs: ", os.environ['CUDA_VISIBLE_DEVICES'])

    writer = SummaryWriter(os.path.join("./logs/", args.name))

    simulated_dataset = SeqDataset(seq_len=args.seq_len)
    
    # 模拟数据的训练集和验证集采样器
    train_indices_simulated = list(range(int(len(simulated_dataset) * 0.1), len(simulated_dataset)))
    val_indices_simulated = list(range(int(len(simulated_dataset) * 0.1)))
    
    # 真实数据的训练集和验证集采样器
    train_size_real = int(0.8 * len(combined_real_dataset))
    val_size_real = len(combined_real_dataset) - train_size_real
    train_dataset_real, val_dataset_real = torch.utils.data.random_split(combined_real_dataset, [train_size_real, val_size_real])
    
    # 合并真实数据和模拟数据的训练集
    train_combined_dataset = ConcatDataset([train_dataset_real, torch.utils.data.Subset(simulated_dataset, train_indices_simulated)])
    val_combined_dataset = ConcatDataset([val_dataset_real, torch.utils.data.Subset(simulated_dataset, val_indices_simulated)])
    
    # 创建数据加载器
    train_loader = DataLoader(train_combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_combined_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = model_loader(args.model_type, args.seq_len)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, args.lr_steps, 0.5)
    
    criterion = nn.L1Loss()

    print("Nb parameters:", count_parameters(model))

    start_epoch = 0
    best_rec = 10000
    for epoch in range(start_epoch, args.mepoch):
        print("epoch: ",epoch)
        is_best = False
        
        train_loss, batch_train, data_train = train(train_loader, model, criterion, optimizer, epoch, print_freq=args.print_frequency)
        val_loss, batch_val, data_val = validate(val_loader, model, criterion, print_freq=args.print_frequency)
        lr_scheduler.step()
        if(val_loss < best_rec):
            best_rec = val_loss
            is_best = True

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_rec': best_rec,
            'args_dict': args
        }

        log_epoch(writer, epoch, train_loss, val_loss, optimizer.param_groups[0]['lr'], batch_train, batch_val, data_train, data_val)
        save_checkpoint(state, is_best, args.name, epoch)

    print('Finished Training')
    print(best_rec)
