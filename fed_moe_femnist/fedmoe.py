import datetime
import json
import random
import math
import os
import argparse
from typing import List, Tuple, Dict, Any

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import write_result, moe_test, l2loss, augment, count_dataset, mlp_test, cnn_test, moe_cnn_test
from fed_moe_femnist.train import fedmoe_train, server_gate, client_logits
from fed_moe_femnist.fedmoe_dataset import split_and_return_dataset
from fed_moe_femnist.moe import MLP, MoE, CNNModel


# ========================= 配置参数 =========================
class Config:
    """配置类：集中管理所有超参数和路径配置"""
    
    # 数据集配置
    DATASET = 'femnist'
    INPUT_SIZE = 784
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 62
    BATCH_SIZE = 64
    
    # 训练配置
    NUM_ACTIVATED_CLIENTS = 3
    CLIENT_LR = 3e-4
    SERVER_GATE_LR = 1e-4
    NUM_CLIENTS = 100
    USERS_PER_SERVER = 10
    USERS_PER_CLIENT = 30
    DEVICE_INDEX = "cuda:0"
    SERVER_EPOCH = 2
    CLIENT_EPOCH = 5
    ITERATION_EPOCH = 1
    TOTAL_EPOCHS = 100

    
    # 损失权重配置
    AUX_WEIGHT = 1
    L2_WEIGHT = 1e-7
    
    # 其他配置
    NOISY_GATING = True
    RANDOM_GATE = False
    ENTROPY_LOSS_FLAG = True
    
    # 数据路径
    RESULT_PATH = "../result"
    DATA_PATH = '../leaf/data/femnist/data'
    RESULT_PATH_TEMPLATE = '../result/femnist/fedmoe_clients_{}_users_{}_lr_{}_a_{}_entro_{}'


# ========================= 工具函数 =========================
def setup_environment():
    """设置运行环境"""
    device = torch.device(Config.DEVICE_INDEX)
    datetime_now = datetime.datetime.now().strftime('%m-%d-%H-%M')
    writer = SummaryWriter(Config.RESULT_PATH)
    return device, datetime_now, writer


def load_and_split_dataset():
    """加载并分割数据集"""
    print("=" * 50)
    print("加载数据集...")
    print("=" * 50)
    
    train_subset, test_subset, server_set, test_set, validation_set, fedavg_set = \
        split_and_return_dataset(
            path=Config.DATA_PATH,
            num_files=35,
            users_per_client=Config.USERS_PER_CLIENT,
            users_per_server=Config.USERS_PER_SERVER,
            num_clients=Config.NUM_CLIENTS
        )
    
    # 打印数据集统计信息
    for i in range(Config.NUM_CLIENTS):
        train_count = count_dataset(train_subset[i], Config.OUTPUT_SIZE)
        test_count = count_dataset(test_subset[i], Config.OUTPUT_SIZE)
        print(f'Client {i + 1} - Train: {train_count}, Test: {test_count}')
    
    server_train_count = count_dataset(server_set, Config.OUTPUT_SIZE)
    valid_count = count_dataset(validation_set, Config.OUTPUT_SIZE)
    fedavg_count = count_dataset(fedavg_set, Config.OUTPUT_SIZE)
    test_count = count_dataset(test_set, Config.OUTPUT_SIZE)
    
    print(f'Server - Train: {server_train_count}, Valid: {valid_count}')
    print(f'FedAvg: {fedavg_count}, Test: {test_count}')
    print(f'Server set size: {len(server_set)}, Test set size: {len(test_set)}')
    
    return train_subset, test_subset, server_set, test_set, validation_set, fedavg_set


def create_data_loaders(server_set, validation_set, fedavg_set):
    """创建数据加载器"""
    server_trainloader = torch.utils.data.DataLoader(
        server_set, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    server_testloader = torch.utils.data.DataLoader(
        validation_set, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    fedavgloader = torch.utils.data.DataLoader(
        fedavg_set, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    return server_trainloader, server_testloader, fedavgloader


def initialize_server_model(device):
    """初始化服务器模型"""
    print("=" * 50)
    print("初始化服务器模型...")
    print("=" * 50)
    
    server_net = MoE(
        input_size=Config.INPUT_SIZE,
        output_size=Config.OUTPUT_SIZE,
        num_experts=5,
        hidden_size=Config.HIDDEN_SIZE,
        noisy_gating=Config.NOISY_GATING,
        k=1,
        random_gate=Config.RANDOM_GATE,
        alpha=0.5
    )
    
    # 配置优化器参数组
    param_groups = [
        {'params': server_net.w_gate.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': server_net.w_noise.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': [server_net.alpha], 'lr': Config.SERVER_GATE_LR},
    ]
    
    server_optimizer = optim.Adam(param_groups)
    scheduler = lr_scheduler.StepLR(server_optimizer, step_size=10, gamma=0.7)
    
    server_net.to(device)
    return server_net, server_optimizer, scheduler


def initialize_client_models():
    """初始化客户端模型"""
    print("=" * 50)
    print("初始化客户端模型...")
    print("=" * 50)
    
    client_nets = []
    client_optimizers = []
    
    for i in range(Config.NUM_CLIENTS):
        client_model = CNNModel(Config.OUTPUT_SIZE)
        optimizer = optim.Adam(client_model.parameters(), lr=Config.CLIENT_LR)
        
        client_nets.append(client_model)
        client_optimizers.append(optimizer)
    
    return client_nets, client_optimizers


def create_client_groups():
    """创建客户端分组"""
    client_ids_dict = {i: [] for i in range(5)}
    for i in range(Config.NUM_CLIENTS):
        client_ids_dict[i % 5].append(i)
    return client_ids_dict


def train_clients(activated_client_ids, client_nets, client_optimizers, 
                 train_subset, validation_set, device, epoch):
    """训练客户端模型"""
    print('-' * 60)
    print('客户端训练阶段')
    print('-' * 60)
    
    client_models = []
    criterion = nn.CrossEntropyLoss()
    
    for client_id in activated_client_ids:
        client_model = client_nets[client_id]
        client_model.to(device)
        optimizer = client_optimizers[client_id]
        
        print(f'Epoch: {epoch + 1}, Client: {client_id + 1}')
        
        # 创建数据加载器
        trainloader = torch.utils.data.DataLoader(
            train_subset[client_id], batch_size=Config.BATCH_SIZE, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            validation_set, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        
        # 客户端训练
        for e in range(Config.CLIENT_EPOCH):
            client_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)
                client_loss += loss.item()
                loss.backward()
                optimizer.step()
        
        # 评估客户端模型
        _, client_acc = cnn_test(client_model, testloader, Config.DEVICE_INDEX)
        print(f'Client {client_id + 1} - Loss: {client_loss / len(trainloader):.4f}, Acc: {client_acc:.4f}')
        
        client_models.append(client_model)
    
    return client_models


def server_iteration_training(server_net, client_models, fedavgloader, 
                            server_testloader, device, epoch, writer, best_model):
    """服务器迭代训练"""
    print('-' * 60)
    print('服务器迭代训练阶段')
    print('-' * 60)
    
    client_logits_combine = client_logits(client_models, fedavgloader, Config.DEVICE_INDEX)
    server_gate_new = []
    c_to_s_logits = None
    
    for e in range(Config.ITERATION_EPOCH):
        print(f'Epoch {epoch + 1}, Iteration {e + 1}')
        
        # Q步骤：获取服务器门控逻辑
        if e == 0:
            print('Q步骤:')
            server_gate_logits, _ = server_gate(server_net, fedavgloader, Config.DEVICE_INDEX)
        else:
            server_gate_logits = server_gate_new
        
        server_gate_logits_combine = torch.cat(server_gate_logits, dim=0)
        
        # FedAvg: 客户端到服务器
        avg = torch.matmul(server_gate_logits_combine.T, client_logits_combine.T)
        c_to_s_logits = F.softmax(avg, dim=1)
        print(f'w1(c to s): {c_to_s_logits.to(torch.float)}')
        
        # 更新服务器专家模型
        update_server_experts(server_net, client_models, c_to_s_logits)
        
        # Q_new步骤
        print('Q_new步骤:')
        new_server_gate_logits, tensor = server_gate(server_net, fedavgloader, Config.DEVICE_INDEX)
        print(f'专家数据分布: {tensor}')
        server_gate_new = new_server_gate_logits
        
        # 评估服务器模型
        cur_loss, cur_acc = moe_cnn_test(server_net, server_testloader, Config.DEVICE_INDEX)
        writer.add_scalar('valid_loss', cur_loss, epoch + 1)
        writer.add_scalar('alpha', server_net.alpha, epoch + 1)
        
        # 更新最佳模型
        update_best_model(server_net, best_model, cur_loss, cur_acc, epoch, c_to_s_logits)
    
    return server_gate_new, client_logits_combine, c_to_s_logits


def update_server_experts(server_net, client_models, c_to_s_logits):
    """更新服务器专家模型参数"""
    # 更新专家网络
    for i in range(server_net.num_experts):
        state_dict = server_net.experts[i].state_dict()
        new_key = {key: 0.0 for key in state_dict.keys()}
        
        for key in state_dict.keys():
            for c in range(5):
                weight = c_to_s_logits[i, c].item()
                new_key[key] += weight * client_models[c].state_dict()[key].clone()
        
        server_net.experts[i].load_state_dict(new_key)
    # 更新固定专家网络
    fix_state_dict = server_net.expert_fix.state_dict()
    new_key = {key: 0.0 for key in fix_state_dict.keys()}
    
    for key in fix_state_dict.keys():
        for c in range(5):
            new_key[key] += 0.2 * client_models[c].state_dict()[key].clone()
    
    server_net.expert_fix.load_state_dict(new_key)


def update_best_model(server_net, best_model, cur_loss, cur_acc, epoch, c_to_s_logits):
    """更新最佳模型记录"""
    if best_model['min_val_loss'] > cur_loss:
        best_model['min_val_loss'] = cur_loss
        best_model['max_val_acc'] = cur_acc
        best_model['epoch'] = epoch + 1
        
        print(f'*** 新的最佳模型 ***')
        print(f'Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}, Alpha: {server_net.alpha:.4f}')
        
        # 保存最佳模型
        save_path = os.path.join(Config.RESULT_PATH, 'best_val.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': server_net.state_dict(),
            'c_to_s': c_to_s_logits,
            'acc': cur_acc
        }, save_path)
    else:
        print(f'当前模型 - Epoch: {epoch + 1}, Loss: {cur_loss:.4f}, Acc: {cur_acc:.4f}')
        print(f'最佳模型 - Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}')


def fedavg_server_to_client(server_net, client_nets, activated_client_ids, 
                           server_gate_new, client_logits_combine):
    """FedAvg: 服务器到客户端"""
    new_server_gate_logits_combine = torch.cat(server_gate_new, dim=0)
    new_avg = torch.matmul(new_server_gate_logits_combine.T, client_logits_combine.T)
    s_to_c_logits = F.softmax(new_avg, dim=0)
    
    # 添加固定专家的logits
    fix_logits = torch.ones(1, s_to_c_logits.size(1), device=s_to_c_logits.device) * (1 - server_net.alpha)
    s_to_c_logits = F.softmax(torch.cat((s_to_c_logits * server_net.alpha, fix_logits), dim=0), dim=0)
    
    print(f'w2(s to c): {s_to_c_logits.to(torch.float)}')
    
    # 更新客户端模型参数
    for i in range(Config.NUM_ACTIVATED_CLIENTS):
        client_state_dict = client_nets[activated_client_ids[i]].state_dict()
        new_key = {key: 0.0 for key in client_state_dict.keys()}
        
        for key in client_state_dict.keys():
            # 从专家网络聚合
            for c in range(server_net.num_experts):
                weight = s_to_c_logits[c, i].item()
                new_key[key] += weight * server_net.experts[c].state_dict()[key].clone()
            
            # 从固定专家网络聚合
            weight = s_to_c_logits[server_net.num_experts, i].item()
            new_key[key] += weight * server_net.expert_fix.state_dict()[key].clone()
        
        client_nets[activated_client_ids[i]].load_state_dict(new_key)
        client_nets[activated_client_ids[i]].to('cpu')


def evaluate_final_model(server_net, test_set):
    """评估最终模型性能"""
    print("=" * 50)
    print("最终模型评估")
    print("=" * 50)
    
    testloader = torch.utils.data.DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    checkpoint = torch.load(os.path.join(Config.RESULT_PATH, 'best_val.pth'))
    server_net.load_state_dict(checkpoint['model_state_dict'])
    
    _, acc = moe_cnn_test(server_net, testloader, Config.DEVICE_INDEX)
    print(f'最终测试准确率: {acc:.4f}')
    
    return acc


# ========================= 主训练流程 =========================
def main(num_clients: int, users_per_client: int, users_per_server: int, 
         lr: float, device_index: int, a: float, entro_weight: float) -> List[Tuple]:
    """主训练流程"""
    print("=" * 60)
    print("联邦学习MoE训练 - FEMNIST数据集")
    print("=" * 60)
    
    # 更新配置
    Config.NUM_CLIENTS = num_clients
    Config.NUM_ACTIVATED_CLIENTS = min(5, num_clients)
    Config.USERS_PER_CLIENT = users_per_client
    Config.USERS_PER_SERVER = users_per_server
    Config.CLIENT_LR = lr
    Config.SERVER_GATE_LR = lr * 0.1  # 门控网络使用较小的学习率
    Config.SERVER_EXPERT_LR = lr * 0.05  # 专家网络使用更小的学习率
    Config.A = a
    Config.ENTROPY_LOSS_WEIGHT = entro_weight
    Config.DEVICE_INDEX = f'cuda:{device_index}'
    
    # 设置结果保存路径
    result_path = Config.RESULT_PATH_TEMPLATE.format(
        num_clients, users_per_client, lr, a, entro_weight
    )
    Config.RESULT_PATH = result_path
    
    # 环境设置
    device = torch.device(Config.DEVICE_INDEX)
    datetime_now = datetime.datetime.now().strftime('%m-%d-%H-%M')
    writer = SummaryWriter(f'{result_path}/tensorboard_{datetime_now}')
    
    # 数据准备
    print("=" * 50)
    print("加载数据集...")
    print("=" * 50)
    
    train_subset, test_subset, server_set, test_set, validation_set, fedavg_set = \
        split_and_return_dataset(
            path=Config.DATA_PATH,
            num_files=35,
            users_per_client=Config.USERS_PER_CLIENT,
            users_per_server=Config.USERS_PER_SERVER,
            num_clients=Config.NUM_CLIENTS
        )
    
    # 打印数据集统计信息
    for i in range(min(10, Config.NUM_CLIENTS)):  # 只打印前10个客户端的信息
        train_count = count_dataset(train_subset[i], Config.OUTPUT_SIZE)
        test_count = count_dataset(test_subset[i], Config.OUTPUT_SIZE)
        print(f'Client {i + 1} - Train: {train_count}, Test: {test_count}')
    
    if Config.NUM_CLIENTS > 10:
        print(f'... 还有 {Config.NUM_CLIENTS - 10} 个客户端')
    
    server_train_count = count_dataset(server_set, Config.OUTPUT_SIZE)
    valid_count = count_dataset(validation_set, Config.OUTPUT_SIZE)
    fedavg_count = count_dataset(fedavg_set, Config.OUTPUT_SIZE)
    test_count = count_dataset(test_set, Config.OUTPUT_SIZE)
    
    print(f'Server - Train: {server_train_count}, Valid: {valid_count}')
    print(f'FedAvg: {fedavg_count}, Test: {test_count}')
    print(f'Server set size: {len(server_set)}, Test set size: {len(test_set)}')
    
    # 创建数据加载器
    server_trainloader = torch.utils.data.DataLoader(
        server_set, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    server_testloader = torch.utils.data.DataLoader(
        validation_set, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    fedavgloader = torch.utils.data.DataLoader(
        fedavg_set, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    # 模型初始化
    print("=" * 50)
    print("初始化服务器模型...")
    print("=" * 50)
    
    server_net = MoE(
        input_size=Config.INPUT_SIZE,
        output_size=Config.OUTPUT_SIZE,
        num_experts=5,
        hidden_size=Config.HIDDEN_SIZE,
        noisy_gating=Config.NOISY_GATING,
        k=1,
        random_gate=Config.RANDOM_GATE,
        alpha=0.5
    )
    
    # 配置优化器参数组
    param_groups = [
        {'params': server_net.w_gate.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': server_net.w_noise.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': [server_net.alpha], 'lr': Config.SERVER_GATE_LR},
    ]
    
    server_optimizer = optim.Adam(param_groups)
    scheduler = lr_scheduler.StepLR(server_optimizer, step_size=10, gamma=0.7)
    
    server_net.to(device)
    
    print("=" * 50)
    print("初始化客户端模型...")
    print("=" * 50)
    
    client_nets = []
    client_optimizers = []
    
    for i in range(Config.NUM_CLIENTS):
        client_model = CNNModel(Config.OUTPUT_SIZE)
        optimizer = optim.Adam(client_model.parameters(), lr=Config.CLIENT_LR)
        
        client_nets.append(client_model)
        client_optimizers.append(optimizer)
    
    print(f"✓ {Config.NUM_CLIENTS} 个客户端模型初始化完成")
    
    # 客户端分组
    client_ids_dict = {i: [] for i in range(Config.NUM_ACTIVATED_CLIENTS)}
    for i in range(Config.NUM_CLIENTS):
        client_ids_dict[i % Config.NUM_ACTIVATED_CLIENTS].append(i)
    
    # 训练状态跟踪
    activated = {i: 0 for i in range(Config.NUM_CLIENTS)}
    best_model = {'epoch': -1, 'min_val_loss': 1e6, 'max_val_acc': 0}
    
    # 服务器预训练
    print("=" * 50)
    print("服务器预训练")
    print("=" * 50)
    fedmoe_train(server_net, server_set, Config.BATCH_SIZE, server_optimizer, 1, 
                Config.ENTROPY_LOSS_WEIGHT, Config.AUX_WEIGHT, Config.L2_WEIGHT, 
                Config.DEVICE_INDEX, entropy_loss_flag=Config.ENTROPY_LOSS_FLAG)
    
    # 主训练循环
    print("=" * 50)
    print("开始联邦学习训练")
    print("=" * 50)
    
    results = []
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(Config.TOTAL_EPOCHS):
        print(f"\n{'='*20} EPOCH {epoch + 1}/{Config.TOTAL_EPOCHS} {'='*20}")
        
        scheduler.step()
        
        # 随机选择激活的客户端
        activated_client_ids = []
        for i in range(Config.NUM_ACTIVATED_CLIENTS):
            selected_client = random.choice(client_ids_dict[i])
            activated_client_ids.append(selected_client)
            activated[selected_client] += 1
        
        # 客户端训练
        client_models = train_clients(activated_client_ids, client_nets, client_optimizers,
                                    train_subset, validation_set, device, epoch)
        
        # 服务器迭代训练
        server_gate_new, client_logits_combine, c_to_s_logits = server_iteration_training(
            server_net, client_models, fedavgloader, server_testloader, 
            device, epoch, writer, best_model)
        
        # FedAvg: 服务器到客户端
        fedavg_server_to_client(server_net, client_nets, activated_client_ids,
                               server_gate_new, client_logits_combine)
    
    # 最终评估
    print("=" * 50)
    print("最终模型评估")
    print("=" * 50)
    
    testloader = torch.utils.data.DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    checkpoint = torch.load(os.path.join(Config.RESULT_PATH, 'best_val.pth'))
    server_net.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc = moe_cnn_test(server_net, testloader, Config.DEVICE_INDEX)
    final_alpha = server_net.alpha.item() if hasattr(server_net, 'alpha') else 0.5
    
    # 记录结果
    results.append((Config.SERVER_GATE_LR, final_acc, final_alpha, best_model['epoch']))
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("训练完成统计")
    print("=" * 50)
    print(f"客户端激活次数: {activated}")
    print(f"最终测试准确率: {final_acc:.4f}")
    print(f"Alpha值: {final_alpha:.4f}")
    print(f"最佳Epoch: {best_model['epoch']}")
    
    writer.close()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedMoE FEMNIST 训练参数')
    parser.add_argument('--nc', dest='num_clients', type=int, 
                       help='客户端数量', default=50)
    parser.add_argument('--upc', dest='users_per_client', type=int,
                       help='每个客户端的用户数', default=30)
    parser.add_argument('--ups', dest='users_per_server', type=int,
                       help='服务器的用户数', default=10)
    parser.add_argument('--lr', dest='lr', type=float, 
                       help='学习率', default=3e-4)
    parser.add_argument('--cuda', dest='cuda', type=int, 
                       help='CUDA设备编号', default=0)
    parser.add_argument('--a', dest='a', type=float, 
                       help='FedAvg权重参数', default=0.1)
    parser.add_argument('--entro', dest='entro_w', type=float, 
                       help='熵损失权重', default=1e-1)
    
    args = parser.parse_args()
    
    # 运行主函数
    results = main(
        num_clients=args.num_clients,
        users_per_client=args.users_per_client,
        users_per_server=args.users_per_server,
        lr=args.lr,
        device_index=args.cuda,
        a=args.a,
        entro_weight=args.entro_w
    )
    
    print("\n" + "=" * 50)
    print("训练结果汇总")
    print("=" * 50)
    for i, (gate_lr, acc, alpha, epoch) in enumerate(results):
        print(f"结果 {i+1}: Gate_LR={gate_lr}, Acc={acc:.4f}, Alpha={alpha:.4f}, Best_Epoch={epoch}")
