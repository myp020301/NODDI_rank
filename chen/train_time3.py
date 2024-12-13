#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Mar 24 09:59:10 2021

@author: Jiquan

Graph-based on pytorch geometric
'''
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

from dataset import DmriDataset
from model import *
from utils import *

from rankmoco_loss import rank_moco_loss
from differentmoco_loss import different_moco_loss


def build_model(model_cfg):
    if model_cfg['backbone']['type'] == 'CNN':
        model = CNN(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                    model_cfg['backbone']['out_channel'])
    elif model_cfg['backbone']['type'] == 'MLP':
        model = MLP(model_cfg['backbone']['in_channel'], model_cfg['backbone']['features'],
                    model_cfg['backbone']['out_channel'], model_cfg['backbone']['drop_out'])
    elif model_cfg['backbone']['type'] == 'GCNN':
        model = GCNN(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                     model_cfg['backbone']['out_channel'], model_cfg['backbone']['K'],
                     model_cfg['backbone']['hidden'])
    elif model_cfg['backbone']['type'] == 'UNet':
        model = UNet(model_cfg['backbone']['out_channel'], model_cfg['backbone']['in_channel'])
    elif model_cfg['backbone']['type'] == 'NestedUNet':
        model = NestedUNet(model_cfg['backbone']['out_channel'], model_cfg['backbone']['in_channel'])
    elif model_cfg['backbone']['type'] == 'HGT':
        model = HGT(in_channel=model_cfg['backbone']['in_channel'], embed_dims=model_cfg['backbone']['embed_dims'],
                    num_heads=model_cfg['backbone']['num_heads'], mlp_ratios=model_cfg['backbone']['mlp_ratios'],
                    qkv_bias=model_cfg['backbone']['qkv_bias'], depths=model_cfg['backbone']['depths'],
                    sr_ratios=model_cfg['backbone']['sr_ratios'], drop_rate=model_cfg['backbone']['drop_rate'],
                    attn_drop_rate=model_cfg['backbone']['attn_drop_rate'],
                    drop_path_rate=model_cfg['backbone']['drop_path_rate'],
                    num_stages=model_cfg['backbone']['num_stages'],
                    gradient_direction_number=model_cfg['backbone']['gradient_direction_number'],
                    gnn_dim=model_cfg['backbone']['gnn_dim'],
                    gnn_out=model_cfg['backbone']['gnn_out'], K=model_cfg['backbone']['K'])
    else:
        print('No {} model'.format(model_cfg['backbone']['type']))
    return model

def build_optimizer(model, optimizer_cfg):
    if optimizer_cfg['type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg['lr'])
    elif optimizer_cfg['type'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=optimizer_cfg['lr'],
                                    momentum=optimizer_cfg['momentum'],
                                    weight_decay=optimizer_cfg['weight_decay'])
    else:
        print('No use {} optimizer'.format(optimizer_cfg['type']))
    return optimizer

def build_scheduler(lr_config, optimizer):
    if lr_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    else:
        print('No use {} scheduler'.format(lr_config['type']))
    return scheduler

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="predictive microstructure model")
    parser.add_argument("--config", default='/data2/mayupeng/chen/config/hgt_config.py',
                        type=str, help="Run the model config file address")
    parser.add_argument("--train_subject_id", default=[    "100206"
    ],
                        type=list, help="Training file name")
    parser.add_argument("--valid_subject_id", default=["102816"],
                        type=list, help="Validing file name")
    parser.add_argument("--data_path", default="/data2/mayupeng/HCP_25_dataset_60",
                        type=str, help="Name of the data path")
    parser.add_argument("--mask_name", default="nodif_brain_mask.nii.gz",
                        type=str, help="Name of the mask file for the data")
    parser.add_argument("--train_data_name", default="train_ght_data_60_NODDI.npy",
                        type=str, help="Name of the training file")
    parser.add_argument("--train_gt_data_name", default="train_gt_ght_data_60_NODDI.npy",
                        type=str, help="Name of the training gold standard file")
    parser.add_argument("--test_data_name", default="train_ght_data_60_NODDI.npy",
                        type=str, help="Name of the testing file")
    parser.add_argument("--test_gt_data_name", default="train_gt_ght_data_60_NODDI.npy",
                        type=str, help="Name of the testing gold standard file")
    parser.add_argument("--is_train", default=True,
                        type=bool, help="Whether to train the model")
    parser.add_argument("--is_generate_image", default=True,
                        type=bool, help="Whether to generate prediction")
    parser.add_argument("--generate_image_save_path", default='./rank_image',
                        type=str, help="Path to save generated images")
    parser.add_argument("--save_parameter_path", default="./parameter",
                        type=str, help="Path to save model parameters")
    parser.add_argument("--microstructure_name", default="NODDI", choices=['NODDI', 'DKI'],
                        type=str, help="Name of the microstructure model")
    parser.add_argument("--brain_max_lenght", default="/data2/mayupeng/chen/hcp_brain_max_lenght.npy",
                        type=str, help="Storage file for the maximum length and width of the model")
    parser.add_argument("--edge_save_name", default="edge",
                        type=str, help="Storage name of the edge")
    parser.add_argument("--gpu_id", default="3",
                        type=str, help="Which GPU to be used")
    args = parser.parse_args()
    return args

def test(test_loader, edge_index, model, test_start_time, subject_id, device, is_voxel, model_name, kwargs):
    if edge_index is not None:
        edge_index = edge_index[1].to(device)
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test'):
        model.eval()
        if model_name == 'GCNN':
            test_input = data.to(device)
            test_label = data.y.to(device)
        else:
            test_input, test_label = data
            test_input = test_input.type(torch.FloatTensor).to(device)
            test_label = test_label.type(torch.FloatTensor).to(device)

        if edge_index is not None:
            out = model(test_input, edge_index)
        else:
            out = model(test_input)

        if i == 0:
            prediction = out.cpu().detach().numpy()
            gt = test_label.cpu().numpy()
        else:
            temp_out = out.cpu().detach().numpy()
            temp_gt = test_label.cpu().numpy()
            prediction = np.concatenate((prediction, temp_out), axis=0)
            gt = np.concatenate((gt, temp_gt), axis=0)
    test_time_cost = time.time() - test_start_time
    print(' took {:.2f} seconds'.format(test_time_cost))
    return restore_img(subject_id, gt, prediction, is_voxel, kwargs)



def train(train_loader, model, optimizer, scheduler, edge_index, device, is_voxel, model_name, criterion_rank_moco, criterion_dif_moco):
    epoch_start_time = time.time()
    total_loss = 0.0
    total_index1_loss = 0.0
    total_index2_loss = 0.0
    total_index3_loss = 0.0
    batch_number = 0

    # 初始化各个时间总和变量
    total_data_prep_time = 0.0
    total_model_forward_time = 0.0
    total_mse_time = 0.0
    total_flat_time = 0.0
    total_sampling_time = 0.0
    total_batch_creation_time = 0.0
    total_ranking_loss_time = 0.0
    total_diff_loss_time = 0.0
    total_ranking_diff_time = 0.0
    total_backward_time = 0.0

    if edge_index is not None:
        edge_index = edge_index[0].to(device)

    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='Training'):
        batch_start_time = time.time()  # 每个批次开始时间
        model.train()
        batch_number += 1
        optimizer.zero_grad()

        # 数据准备时间
        data_prep_start = time.time()
        if model_name == 'GCNN':
            input = data.to(device)
            label = data.y.to(device)
        else:
            input, label = data
            label = label.type(torch.FloatTensor).to(device)
            input = input.type(torch.FloatTensor).to(device)
        data_prep_time = time.time() - data_prep_start
        total_data_prep_time += data_prep_time
        print(f"Batch {i+1}/{len(train_loader)}: Data Preparation Time: {data_prep_time:.6f} s")

        # 模型前向传播时间
        model_forward_start = time.time()
        if edge_index is not None:
            out = model(input, edge_index)
        else:
            out = model(input)
        model_forward_time = time.time() - model_forward_start
        total_model_forward_time += model_forward_time
        print(f"Batch {i+1}/{len(train_loader)}: Model Forward Time: {model_forward_time:.6f} s")

        # 损失计算时间
        mse_loss_start = time.time()
        loss = F.mse_loss(out, label)
        mse_loss_time = time.time() - mse_loss_start
        total_mse_time += mse_loss_time
        print(f"Batch {i+1}/{len(train_loader)}: MSE Loss Calculation Time: {mse_loss_time:.6f} s")

        # 根据是否为 Voxel 模式计算不同的损失
        if is_voxel:
            # Voxel 特定的损失计算时间
            voxel_loss_start = time.time()
            index1_loss = F.mse_loss(out[:, 0], label[:, 0])
            index2_loss = F.mse_loss(out[:, 1], label[:, 1])
            index3_loss = F.mse_loss(out[:, 2], label[:, 2])
            voxel_loss_time = time.time() - voxel_loss_start
            total_ranking_loss_time += voxel_loss_time  # 假设 voxel loss 包含 ranking loss
            total_diff_loss_time += 0.0  # Voxel 模式下没有 Diff Loss
            print(f"Batch {i+1}/{len(train_loader)}: Voxel Loss Calculation Time: {voxel_loss_time:.6f} s")
        else:
            # Ranking 和 Diff 损失计算时间
            

            index1_loss = F.mse_loss(out[:, :, :, 0], label[:, :, :, 0])
            index2_loss = F.mse_loss(out[:, :, :, 1], label[:, :, :, 1])
            index3_loss = F.mse_loss(out[:, :, :, 2], label[:, :, :, 2])

            # 展开输出和标签的时间
            start_flat_time = time.time()
            out_flat = out.view(-1, 3)  # 形状：[N, 3]
            label_flat = label.view(-1, 3)
            flat_time = time.time() - start_flat_time
            total_flat_time += flat_time
            print(f"Batch {i+1}/{len(train_loader)}: Flattening Time: {flat_time:.6f} s")

            # 随机采样子批次索引的时间
            num_samples = out_flat.size(0)
            sub_batch_size = 64
            num_sub_batches = 25  # 减少子批次数量以优化性能
            start_sampling_time = time.time()
            indices = torch.randperm(num_samples)[:sub_batch_size * num_sub_batches]
            sampling_time = time.time() - start_sampling_time
            total_sampling_time += sampling_time
            print(f"Batch {i+1}/{len(train_loader)}: Sampling Time: {sampling_time:.6f} s")

            # 计算子批次创建的时间
            start_batch_time = time.time()
            out_sub_batches = out_flat[indices].view(num_sub_batches, sub_batch_size, 3)
            label_sub_batches = label_flat[indices].view(num_sub_batches, sub_batch_size, 3)
            batch_creation_time = time.time() - start_batch_time
            total_batch_creation_time += batch_creation_time
            print(f"Batch {i+1}/{len(train_loader)}: Batch Creation Time: {batch_creation_time:.6f} s")
            
            ranking_diff_start = time.time()
            # 遍历每个子批次，计算 rankingloss 和 diffloss
            for sub_batch_idx in range(num_sub_batches):
                out_sub = out_sub_batches[sub_batch_idx]
                label_sub = label_sub_batches[sub_batch_idx]

                # 计算 rankingloss 时间
                ranking_loss_start = time.time()
                rank_losses = [criterion_rank_moco(out_sub[:, idx].unsqueeze(1), label_sub[:, idx].unsqueeze(1)) for idx in range(3)]
                total_ranking_loss = sum(rank_losses)
                ranking_loss_time = time.time() - ranking_loss_start
                total_ranking_loss_time += ranking_loss_time
                print(f"Batch {i+1}/{len(train_loader)}, Sub-batch {sub_batch_idx+1}/{num_sub_batches}: Ranking Loss Time: {ranking_loss_time:.6f} s")

                # 计算 diffloss 时间
                diff_loss_start = time.time()
                diff_loss = criterion_dif_moco(out_sub.unsqueeze(2), label_sub.unsqueeze(2))
                total_diff_loss = diff_loss.item()
                diff_loss_time = time.time() - diff_loss_start
                total_diff_loss_time += diff_loss_time
                print(f"Batch {i+1}/{len(train_loader)}, Sub-batch {sub_batch_idx+1}/{num_sub_batches}: Diff Loss Time: {diff_loss_time:.6f} s")

            ranking_diff_end = time.time()
            ranking_diff_time = ranking_diff_end - ranking_diff_start
            total_ranking_diff_time += ranking_diff_time
            print(f"Batch {i+1}/{len(train_loader)}: Total Ranking & Diff Loss Time: {ranking_diff_time:.6f} s")

        # 反向传播和优化时间
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        total_backward_time += backward_time
        print(f"Batch {i+1}/{len(train_loader)}: Backward Time: {backward_time:.6f} s")

        # 累加损失
        total_loss += loss.item()
        total_index1_loss += index1_loss.item()
        total_index2_loss += index2_loss.item()
        total_index3_loss += index3_loss.item()


    # 计算并返回平均损失和时间
    total_loss_avg = total_loss / batch_number
    total_index1_loss_avg = total_index1_loss / batch_number
    total_index2_loss_avg = total_index2_loss / batch_number
    total_index3_loss_avg = total_index3_loss / batch_number
    epoch_time_cost = time.time() - epoch_start_time

    # 计算各个时间的平均值
    avg_data_prep_time = total_data_prep_time / batch_number
    avg_model_forward_time = total_model_forward_time / batch_number
    avg_mse_time = total_mse_time / batch_number
    avg_flat_time = total_flat_time / batch_number
    avg_sampling_time = total_sampling_time / batch_number
    avg_batch_creation_time = total_batch_creation_time / batch_number
    avg_ranking_loss_time = total_ranking_loss_time / batch_number
    avg_diff_loss_time = total_diff_loss_time / batch_number
    avg_ranking_diff_time = total_ranking_diff_time / batch_number
    avg_backward_time = total_backward_time / batch_number

    # 打印平均时间
    print("\n=== Epoch Summary ===")
    print(f"Epoch Time: {epoch_time_cost:.2f} seconds")
    print(f"Average Data Preparation Time: {avg_data_prep_time:.6f} s")
    print(f"Average Model Forward Time: {avg_model_forward_time:.6f} s")
    print(f"Average MSE Calculation Time: {avg_mse_time:.6f} s")
    if is_voxel:
        print(f"Average Voxel Loss Calculation Time: {avg_ranking_loss_time:.6f} s")
    else:
        print(f"Average Flattening Time: {avg_flat_time:.6f} s")
        print(f"Average Sampling Time: {avg_sampling_time:.6f} s")
        print(f"Average Batch Creation Time: {avg_batch_creation_time:.6f} s")
        print(f"Average Ranking Loss Time: {avg_ranking_loss_time:.6f} s")
        print(f"Average Diff Loss Time: {avg_diff_loss_time:.6f} s")
        print(f"Average Total Ranking & Diff Loss Time: {avg_ranking_diff_time:.6f} s")
    print(f"Average Backward Time: {avg_backward_time:.6f} s")
    print(f"Average Total Loss: {total_loss_avg:.6f}")
    print(f"Average Index1 Loss: {total_index1_loss_avg:.6f}")
    print(f"Average Index2 Loss: {total_index2_loss_avg:.6f}")
    print(f"Average Index3 Loss: {total_index3_loss_avg:.6f}")

    if scheduler is not None:
        scheduler.step()

    return total_loss_avg, total_index1_loss_avg, total_index2_loss_avg, total_index3_loss_avg, epoch_time_cost


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    if not os.path.exists(args.save_parameter_path + '/' + model_cfg['backbone']['type']):
        os.makedirs(args.save_parameter_path + '/' + model_cfg['backbone']['type'])

    if data_cfg['train']['pretrained_flag'] == True:
        args.save_parameter_path = args.save_parameter_path + '/' + model_cfg['backbone']['type']  + '_pre_best_parameter.pth'
    else:
        args.save_parameter_path = args.save_parameter_path + '/' + model_cfg['backbone']['type']  + '_best_parameter.pth'

    InputPatch, TargetPatch = get_concatenate_data(args.data_path, args.train_subject_id, args.train_data_name,
                                                   args.train_gt_data_name)
    if model_cfg['backbone']['type'] == 'GCNN':
        InputPatch = torch.from_numpy(InputPatch)
        TargetPatch = torch.from_numpy(TargetPatch)
        graph_data_list = []
        voxel_number = InputPatch.shape[0]
        edge_index = make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                               data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'], data_cfg['edge']['image_shape'],
                               data_cfg['batch_size'], None, model_cfg['backbone']['type'])
        for j in range(0, voxel_number, 1):
            graph_data_list.append(Data(x=InputPatch[j, :].view(-1, 1).type(torch.FloatTensor), edge_index=edge_index,
                                        y=TargetPatch[j, :].view(1, -1).type(torch.FloatTensor)))
        train_loader = GraphDataLoader(graph_data_list, batch_size=data_cfg['batch_size'], shuffle=True, drop_last=True,
                                       num_workers=data_cfg['num_workers'])
    else:
        input_list = list(InputPatch)
        targets_list = list(TargetPatch)

        train_dataset = DmriDataset(input_list, targets_list)
        train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True,
                                  drop_last=True, num_workers=data_cfg['num_workers'])

    model = build_model(model_cfg)
    model.to(device)

    optimizer = build_optimizer(model, optimizer_cfg)

    edge_index_list = []
    if data_cfg['need_edge'] == True:
        train_edge_path = './train_' + args.edge_save_name + '.npy'
        test_edge_path = './test_' + args.edge_save_name + '.npy'
        if os.path.isfile(train_edge_path) and os.path.isfile(test_edge_path):
            train_edge_index = torch.load(train_edge_path)
            train_edge_index = torch.LongTensor(train_edge_index)
            edge_index_list.append(train_edge_index)
            
            test_edge_index = torch.load(test_edge_path)
            test_edge_index = torch.LongTensor(test_edge_index)
            edge_index_list.append(test_edge_index)
        else:
            make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                      data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'], data_cfg['edge']['image_shape'],
                      data_cfg['batch_size'], train_edge_path, model_cfg['backbone']['type'])
            train_edge_index = torch.load(train_edge_path)
            train_edge_index = torch.LongTensor(train_edge_index)
            edge_index_list.append(train_edge_index)

            make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                      data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'], data_cfg['edge']['image_shape'],
                      data_cfg['test']['batch_size'], test_edge_path, model_cfg['backbone']['type'])
            test_edge_index = torch.load(test_edge_path)
            test_edge_index = torch.LongTensor(test_edge_index)
            edge_index_list.append(test_edge_index)
    else:
        edge_index_list = None

    if data_cfg['train']['pretrained_flag'] == True:
        if os.path.isfile(data_cfg['train']['pretrained_weights']):
            model.load_state_dict(torch.load(data_cfg['train']['pretrained_weights']))
            print('Load parameters success!')
        else:
            print('Load parameters unsuccess!')

    if lr_config['is_scheduler'] == True:
        scheduler = build_scheduler(lr_config, optimizer)
    else:
        scheduler = None

    train_start_time = time.time()
    min_loss = float('inf')
    all_psnr_max = 0
    early_stop_number = 0
    criterion_rank_moco =  rank_moco_loss(sorter_checkpoint_path="/data2/mayupeng/Loss/weights/real_model.pth.tar").to(device)
    criterion_dif_moco = different_moco_loss().to(device)
    for epoch in range(0, data_cfg['train']['epoches']):
        total_loss, total_index1_loss, total_index2_loss, total_index3_loss, epoch_time_cost = train(train_loader,
                                                                                                     model,
                                                                                                     optimizer,
                                                                                                     scheduler,
                                                                                                     edge_index_list,
                                                                                                     device,
                                                                                                     data_cfg['is_voxel'],
                                                                                                     model_cfg['backbone']['type'], criterion_rank_moco,criterion_dif_moco)
        if args.microstructure_name == 'NODDI':
            print('Epoch {} time {:.2f} seconds total_loss is: {:.6f} icvf_loss is: {:.6f} isovf_loss is: {:.6f} od_loss is: {:.6f}'.format(
                epoch + 1, epoch_time_cost, total_loss, total_index1_loss, total_index2_loss, total_index3_loss))
        if args.microstructure_name == 'DKI':
            print('Epoch {} time {:.2f} seconds total_loss is: {:.6f} ak_loss is: {:.6f} mk_loss is: {:.6f} rk_loss is: {:.6f}'.format(
                epoch + 1, epoch_time_cost, total_loss, total_index1_loss, total_index2_loss, total_index3_loss))
        if ((epoch + 1) % 1 == 0) and (total_loss < min_loss):  # % 1
            min_loss = total_loss
            print('min loss is {:.6f}'.format(min_loss))
            index1_psnr, index2_psnr, index3_psnr, all_psnr = 0, 0, 0, 0
            subject_number = len(args.valid_subject_id)
            for subject_id in args.valid_subject_id:
                test_start_time = time.time()
                subject_list = []
                subject_list.append(subject_id)
                Test_InputPatch, Test_TargetPatch = get_concatenate_data(args.data_path, subject_list,
                                                                         args.test_data_name, args.test_gt_data_name)
                if model_cfg['backbone']['type'] == 'GCNN':
                    Test_InputPatch = torch.from_numpy(Test_InputPatch.astype('float32'))
                    Test_TargetPatch = torch.from_numpy(Test_TargetPatch.astype('float32'))
                    graph_data_list = []
                    voxel_number = Test_InputPatch.shape[0]
                    test_edge_index = make_edge(args.data_path, data_cfg['edge']['angle'],
                                                model_cfg['backbone']['in_channel'],
                                                data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'],
                                                data_cfg['edge']['image_shape'],
                                                data_cfg['batch_size'], None,
                                                model_cfg['backbone']['type'])
                    for j in range(0, voxel_number, 1):
                        graph_data_list.append(
                            Data(x=Test_InputPatch[j, :].view(-1, 1).type(torch.FloatTensor),
                                 edge_index=test_edge_index,
                                 y=Test_TargetPatch[j, :].view(1, -1).type(torch.FloatTensor)))
                    test_loader = GraphDataLoader(graph_data_list, batch_size=data_cfg['test']['batch_size'],
                                                  shuffle=False, drop_last=False, num_workers=data_cfg['num_workers'])
                    edge_index = None
                else:
                    test_input = list(Test_InputPatch.astype('float32'))
                    test_targets = list(Test_TargetPatch.astype('float32'))
                    test_dataset = DmriDataset(test_input, test_targets)
                    test_loader = DataLoader(test_dataset, batch_size=data_cfg['test']['batch_size'], shuffle=False,
                                             drop_last=False, num_workers=data_cfg['num_workers'])
                if args.is_train:
                    index1_psnr_value, index2_psnr_value, index3_psnr_value, all_psnr_value = test(test_loader,
                                                                edge_index_list, model, test_start_time, subject_id,
                                                                device, data_cfg['is_voxel'],
                                                                model_cfg['backbone']['type'], args)
                else:
                    print('Not training')
                index1_psnr += index1_psnr_value
                index2_psnr += index2_psnr_value
                index3_psnr += index3_psnr_value
                all_psnr += all_psnr_value
            index1_psnr = index1_psnr / subject_number
            index2_psnr = index2_psnr / subject_number
            index3_psnr = index3_psnr / subject_number
            all_psnr = all_psnr / subject_number
            if args.microstructure_name == 'NODDI':
                print('Validation subject mean PSNR icvf: {:.6f}, isovf: {:.6f}, od: {:.6f}, all_noddi: {:.6f}'.format(
                    index1_psnr, index2_psnr, index3_psnr, all_psnr))
            if args.microstructure_name == 'DKI':
                print('Validation subject mean PSNR ak: {:.6f}, mk: {:.6f}, rk: {:.6f}, all_dki: {:.6f}'.format(
                    index1_psnr, index2_psnr, index3_psnr, all_psnr))
            if all_psnr_max < all_psnr:
                all_psnr_max = all_psnr
                torch.save(model.state_dict(), args.save_parameter_path)
                early_stop_number = 0
                print('Model parameters have been stored at epoch {}'.format(epoch + 1))
            else:
                early_stop_number += 1
                print('Early stopping counter: {}'.format(early_stop_number))
        else:
            early_stop_number += 1
            print('Early stopping counter: {}'.format(early_stop_number))
        if early_stop_number > 9:
            break
    train_time_cost = time.time() - train_start_time
    print('Training took {:.2f} seconds'.format(train_time_cost))

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists("./parameter"):
        os.makedirs("./parameter")
    main(args)
