import json
import math
import os
import pickle as pkl
import argparse
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils import (write_result, moe_test, l2loss, augment, count_dataset, 
                   mlp_test, bert_test, convertToTorchInt, split_data_by_majority_class,
                   create_balanced_dataset)
from train import fedmoe_train, server_gate, client_logits
from moe_bert import MLP, MoE
from model_bert.bert import BertNet
import torch.nn.functional as F


# ========================= 配置参数 =========================
class Config:
    """配置类：集中管理所有超参数和路径配置"""
    
    # 数据集配置
    N_POSITIVE = 500
    N_NEGATIVE = 500
    DATASET = 'sent140'
    INPUT_SIZE = 784  # 兼容性保留
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 2  # 二分类：正面/负面情感
    BATCH_SIZE = 16
    CLIENT_BATCH_SIZE = 4
    
    # 数据分布配置
    MAJORITY_RATIO = 0.7  # 客户端数据不均衡比例
    
    # 训练配置
    NUM_ACTIVATED_CLIENTS = 3
    TOTAL_EPOCHS = 100
    SERVER_EPOCH = 2
    CLIENT_EPOCH = 1
    ITERATION_EPOCH = 1
    
    # 学习率配置
    BASE_LR = 1e-3
    MOMENTUM = 0.9
    ALPHA_LR = 1e-5
    
    # 损失权重配置
    SERVER_ENTROPY_LOSS_WEIGHT = 1e-3
    AUX_WEIGHT = 1
    L2_WEIGHT = 1e-7
    FEDAVG_WEIGHT_A = 0.1

    # 模型配置
    CLIENT_LR = 1e-3
    SERVER_EXPERT_LR = 1e-3
    K = 1
    SERVER_GATE_LR = 1e-4
    SERVER_NUM_EXPERTS = 3
    NUM_CLIENTS = 100
    NOISY_GATING = True
    RANDOM_GATE = True
    
    # 设备配置
    DEVICE_0 = 'cuda:0'
    DEVICE_1 = 'cuda:1'
    CPU_DEVICE = 'cpu'
    
    # 数据文件路径
    TRAIN_FILE = 'sent140/train_input_ids.pkl'
    VAL_FILE = 'sent140/val_input_ids.pkl'
    TEST_FILE = 'sent140/test_input_ids.pkl'
    
    # 结果保存路径模板
    RESULT_PATH_TEMPLATE = '../result/moe_bert/fedmoe_main_expert_{}_non-iid_{}-{}_client_{}_lr_{}_a_{}_entro_{}'
    
    @classmethod
    def get_client_device(cls, client_id: int) -> str:
        """获取客户端对应的设备"""
        return cls.DEVICE_0 if client_id % 2 == 0 else cls.DEVICE_1
    
    @classmethod
    def get_result_path(cls, n_main_experts: int, n_positive: int, n_negative: int, 
                       num_clients: int, lr: float, a: float, entro_weight: float) -> str:
        """获取结果保存路径"""
        return cls.RESULT_PATH_TEMPLATE.format(
            n_main_experts, n_positive, n_negative, num_clients, lr, a, entro_weight
        )


# ========================= 数据加载和处理 =========================
def load_sent140_data() -> Tuple[List, List, List, List, List, List]:
    """加载SENT140数据集"""
    print("=" * 50)
    print("加载SENT140数据集...")
    print("=" * 50)
    
    # 加载训练数据
    with open(Config.TRAIN_FILE, 'rb') as f:
        Xtrain, Ytrain = pkl.load(f)
        print(f"训练数据: {len(Xtrain)} 样本")
    
    # 加载验证数据
    with open(Config.VAL_FILE, 'rb') as f:
        Xval, Yval = pkl.load(f)
        print(f"验证数据: {len(Xval)} 样本")
    
    # 加载测试数据
    with open(Config.TEST_FILE, 'rb') as f:
        Xtest, Ytest = pkl.load(f)
        print(f"测试数据: {len(Xtest)} 样本")
    
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


def create_datasets(Xtrain: List, Ytrain: List, Xval: List, Yval: List, 
                   Xtest: List, Ytest: List) -> Tuple[TensorDataset, Dict, Dict, TensorDataset, TensorDataset, TensorDataset]:
    """创建训练和测试数据集"""
    print("=" * 50)
    print("创建数据集...")
    print("=" * 50)
    
    # 创建服务器平衡数据集
    server_set = create_balanced_dataset(
        convertToTorchInt(Xtrain[16000:20000], torch.device(Config.DEVICE_0)),
        convertToTorchInt(Ytrain[16000:20000], torch.device(Config.DEVICE_0)),
        n_positive=Config.N_POSITIVE,
        n_negative=Config.N_NEGATIVE,
        device=torch.device(Config.DEVICE_0)
    )
    
    # 按主要类别分割客户端数据
    train_subset, test_subset = split_data_by_majority_class(
        Xtrain[:19000], Ytrain[:19000], Xtest, Ytest,
        num_clients=Config.NUM_CLIENTS,
        device=torch.device(Config.DEVICE_0),
        majority_ratio=Config.MAJORITY_RATIO
    )
    
    # 创建联邦平均数据集
    fedavg_set = TensorDataset(
        convertToTorchInt(Xtest[:200], torch.device(Config.DEVICE_1)), 
        convertToTorchInt(Ytest[:200], torch.device(Config.DEVICE_1))
    )
    
    # 创建验证和测试数据集
    validation_set = TensorDataset(
        convertToTorchInt(Xtest, torch.device(Config.DEVICE_1)), 
        convertToTorchInt(Ytest, torch.device(Config.DEVICE_1))
    )
    
    test_set = TensorDataset(
        convertToTorchInt(Xval, torch.device(Config.DEVICE_1)), 
        convertToTorchInt(Yval, torch.device(Config.DEVICE_1))
    )
    
    # 打印数据集统计信息
    print_dataset_stats(train_subset, test_subset, server_set, validation_set, fedavg_set)
    
    return server_set, train_subset, test_subset, fedavg_set, validation_set, test_set


def print_dataset_stats(train_subset: Dict, test_subset: Dict, server_set: TensorDataset,
                       validation_set: TensorDataset, fedavg_set: TensorDataset):
    """打印数据集统计信息"""
    for i in range(Config.NUM_CLIENTS):
        print(f'Client {i + 1} - Train: {len(train_subset[i])}, Test: {len(test_subset[i])}')
    
    print(f'Server - Train: {len(server_set)}, Valid: {len(validation_set)}, FedAvg: {len(fedavg_set)}')


def create_data_loaders(server_set: TensorDataset, validation_set: TensorDataset, 
                       fedavg_set: TensorDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    server_trainloader = DataLoader(server_set, batch_size=Config.BATCH_SIZE, shuffle=True)
    server_testloader = DataLoader(validation_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    fedavgloader = DataLoader(fedavg_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return server_trainloader, server_testloader, fedavgloader


# ========================= 模型初始化 =========================
def initialize_server_model(n_main_experts: int) -> Tuple[MoE, optim.Optimizer]:
    """初始化服务器MoE模型"""
    print("=" * 50)
    print("初始化服务器MoE模型...")
    print("=" * 50)
    
    server_net = MoE(
        input_size=Config.INPUT_SIZE,
        output_size=Config.OUTPUT_SIZE,
        num_experts=Config.SERVER_NUM_EXPERTS,
        hidden_size=Config.HIDDEN_SIZE,
        noisy_gating=Config.NOISY_GATING,
        k=Config.K,
        random_gate=Config.RANDOM_GATE,
        n_main_experts=n_main_experts
    )
    
    print(f"主专家数量: {server_net.n_main_experts}")
    
    # 配置优化器参数组
    if n_main_experts == 1:
        param_groups = [
            {'params': server_net.w_gate.parameters(), 'lr': Config.SERVER_GATE_LR},
            {'params': server_net.experts.parameters(), 'lr': Config.SERVER_EXPERT_LR},
            {'params': server_net.expert_fix.parameters(), 'lr': Config.SERVER_EXPERT_LR},
            {'params': [server_net.alpha], 'lr': Config.ALPHA_LR}
        ]
    else:
        param_groups = [
            {'params': server_net.w_gate.parameters(), 'lr': Config.SERVER_GATE_LR},
            {'params': server_net.experts.parameters(), 'lr': Config.SERVER_EXPERT_LR},
            {'params': server_net.expert_fix.parameters(), 'lr': Config.SERVER_EXPERT_LR}
        ]
    
    optimizer = optim.SGD(param_groups, momentum=Config.MOMENTUM)
    server_net.to(Config.DEVICE_1)
    
    return server_net, optimizer


def initialize_client_models() -> Tuple[List[BertNet], List[optim.Optimizer]]:
    """初始化客户端模型"""
    print("=" * 50)
    print("初始化客户端模型...")
    print("=" * 50)
    
    client_nets = []
    client_optimizers = []
    
    for i in range(Config.NUM_CLIENTS):
        client_model = BertNet(num_classes=Config.OUTPUT_SIZE)
        optimizer = optim.SGD(client_model.parameters(), lr=Config.CLIENT_LR, momentum=Config.MOMENTUM)
        
        client_nets.append(client_model)
        client_optimizers.append(optimizer)
    
    print(f"✓ {Config.NUM_CLIENTS} 个客户端模型初始化完成")
    return client_nets, client_optimizers


# ========================= 训练辅助函数 =========================
def move_optimizer_to_device(optimizer: optim.Optimizer, device: str):
    """将优化器状态移动到指定设备"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def clear_gpu_memory():
    """清理GPU内存"""
    torch.cuda.empty_cache()
    with torch.cuda.device(Config.DEVICE_0):
        torch.cuda.empty_cache()
    with torch.cuda.device(Config.DEVICE_1):
        torch.cuda.empty_cache()


# ========================= 训练函数 =========================
def train_clients(activated_client_ids: List[int], client_nets: List[BertNet],
                 client_optimizers: List[optim.Optimizer], train_subset: Dict,
                 test_subset: Dict, epoch: int) -> List[BertNet]:
    """训练激活的客户端模型"""
    print('-' * 60)
    print('客户端训练阶段')
    print('-' * 60)
    
    client_models = []
    criterion = nn.CrossEntropyLoss()
    
    for client_id in activated_client_ids:
        print(f'Epoch: {epoch + 1}, Client: {client_id + 1}')
        
        # 设置设备和模型
        device = Config.get_client_device(client_id)
        client_model = client_nets[client_id]
        client_model.to(device)
        optimizer = client_optimizers[client_id]
        
        # 移动优化器状态到设备
        move_optimizer_to_device(optimizer, device)
        
        # 创建数据加载器
        trainloader = DataLoader(train_subset[client_id], batch_size=Config.CLIENT_BATCH_SIZE, shuffle=True)
        testloader = DataLoader(test_subset[client_id], batch_size=Config.CLIENT_BATCH_SIZE, shuffle=False)
        
        # 客户端训练
        total_loss = 0.0
        for e in range(Config.CLIENT_EPOCH):
            epoch_loss = 0.0
            for batch_idx, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                inputs = inputs.view(inputs.shape[0], -1)
                outputs, prob = client_model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            total_loss += epoch_loss
        
        # 评估客户端模型
        _, client_acc = bert_test(client_model, testloader, device)
        avg_loss = total_loss / (Config.CLIENT_EPOCH * len(trainloader))
        
        print(f'Client {client_id + 1} - Loss: {avg_loss:.4f}, Acc: {client_acc:.4f}')
        
        # 移回CPU并清理
        client_model.to('cpu')
        client_models.append(client_model)
        move_optimizer_to_device(optimizer, 'cpu')
        clear_gpu_memory()
    
    return client_models


def main(num_clients: int, n_main_experts: int, lr: float, device_index: int,
         n_positive: int, n_negative: int, a: float, entro_weight: float) -> List[Tuple]:
    """主训练流程"""
    print("=" * 60)
    print("联邦学习MoE训练 - SENT140数据集")
    print("=" * 60)
    
    # 更新配置
    Config.NUM_CLIENTS = num_clients
    Config.NUM_ACTIVATED_CLIENTS = min(3, num_clients)
    Config.N_MAIN_EXPERTS = n_main_experts
    Config.N_POSITIVE = n_positive
    Config.N_NEGATIVE = n_negative
    Config.SERVER_GATE_LR = lr * 10  # 门控网络使用更大的学习率
    Config.SERVER_EXPERT_LR = lr     # 专家网络学习率
    Config.CLIENT_LR = lr            # 客户端学习率
    Config.FEDAVG_WEIGHT_A = a
    Config.SERVER_ENTROPY_LOSS_WEIGHT = entro_weight
    Config.CLIENT_ENTROPY_LOSS_WEIGHT = entro_weight
    
    # 设置设备配置
    if device_index >= 0:
        Config.DEVICE_0 = f'cuda:{device_index}'
        Config.DEVICE_1 = f'cuda:{min(device_index + 1, torch.cuda.device_count() - 1)}'
    
    # 设置结果保存路径
    result_path = Config.get_result_path(
        n_main_experts, n_positive, n_negative, num_clients, lr, a, entro_weight
    )
    
    # 数据准备
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = load_sent140_data()
    server_set, train_subset, test_subset, fedavg_set, validation_set, test_set = create_datasets(
        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
    server_trainloader, server_testloader, fedavgloader = create_data_loaders(
        server_set, validation_set, fedavg_set)
    
    # 模型初始化
    server_net, server_optimizer = initialize_server_model(n_main_experts)
    client_nets, client_optimizers = initialize_client_models()
    
    # 训练状态跟踪
    best_model = {'epoch': -1, 'min_val_loss': 1e6, 'max_val_acc': 0}
    results = []
    
    # 主训练循环
    print("=" * 50)
    print("开始联邦学习训练")
    print("=" * 50)
    
    for epoch in range(Config.TOTAL_EPOCHS):
        print(f"\n{'='*20} EPOCH {epoch + 1}/{Config.TOTAL_EPOCHS} {'='*20}")
        
        # 随机选择激活的客户端
        activated_client_ids = np.random.choice(
            Config.NUM_CLIENTS, Config.NUM_ACTIVATED_CLIENTS, replace=False
        ).tolist()
        
        print(f"激活的客户端: {activated_client_ids}")
        
        # 客户端训练
        client_models = train_clients(activated_client_ids, client_nets, client_optimizers,
                                    train_subset, test_subset, epoch)
        
        # 服务器迭代训练
        server_gate_new, c_to_s_logits = server_iteration_training(
            server_net, client_models, fedavgloader, server_testloader,
            server_set, server_optimizer, epoch, best_model, n_main_experts, result_path)
        
        # FedAvg: 服务器到客户端
        client_logits_combine = client_logits(client_models, fedavgloader, 'cpu')
        fedavg_server_to_client(server_net, client_nets, activated_client_ids,
                               server_gate_new, client_logits_combine)
        
        # 清理内存和移动模型到CPU
        server_net.to('cpu')
        for client_id in activated_client_ids:
            client_nets[client_id].to('cpu')
        for model in client_models:
            model.to('cpu')
        clear_gpu_memory()
    
    # 最终评估
    final_acc = evaluate_final_model(server_net, test_set, n_main_experts, result_path)
    final_alpha = server_net.alpha.item() if hasattr(server_net, 'alpha') else 0.0
    
    # 记录结果
    results.append((Config.SERVER_GATE_LR, final_acc, final_alpha, best_model['epoch']))
    
    print("\n" + "=" * 50)
    print("训练完成统计")
    print("=" * 50)
    print(f"最终测试准确率: {final_acc:.4f}")
    print(f"Alpha值: {final_alpha:.4f}")
    print(f"最佳Epoch: {best_model['epoch']}")
    
    return results


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
    for i in range(5):
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


def server_iteration_training(server_net: MoE, client_models: List[BertNet],
                            fedavgloader: DataLoader, server_testloader: DataLoader,
                            server_set: TensorDataset, server_optimizer: optim.Optimizer,
                            epoch: int, best_model: Dict, n_main_experts: int,
                            result_path: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """服务器迭代训练过程"""
    print('-' * 60)
    print('服务器迭代训练阶段')
    print('-' * 60)
    
    # 获取客户端logits
    client_logits_combine = client_logits(client_models, fedavgloader, 'cpu')
    server_gate_new = []
    c_to_s_logits = None
    criterion = nn.CrossEntropyLoss()
    
    for iteration in range(Config.ITERATION_EPOCH):
        print(f'Epoch {epoch + 1}, Iteration {iteration + 1}')
        
        # Q步骤：获取服务器门控
        if iteration == 0:
            print('Q步骤:')
            server_gate_logits, _ = server_gate(server_net, fedavgloader, Config.DEVICE_1)
        else:
            server_gate_logits = server_gate_new
        
        server_gate_logits_combine = torch.cat(server_gate_logits, dim=0)
        
        # FedAvg: 客户端到服务器聚合
        server_gate_logits_combine = server_gate_logits_combine.to('cpu')
        client_logits_combine = client_logits_combine.to('cpu')
        avg = torch.matmul(server_gate_logits_combine.T, client_logits_combine.T)
        c_to_s_logits = F.softmax(avg, dim=1)
        
        print(f'Client to Server权重: {c_to_s_logits.to(torch.float)}')
        
        # 更新服务器专家参数
        update_server_experts(server_net, client_models, c_to_s_logits, n_main_experts)
        
        # 服务器训练
        print(f'服务器训练 - Epoch {epoch + 1}, Iteration {iteration + 1}')
        fedmoe_train(server_net, server_set, Config.BATCH_SIZE, server_optimizer, 
                    Config.SERVER_EPOCH, Config.SERVER_ENTROPY_LOSS_WEIGHT,
                    Config.AUX_WEIGHT, Config.L2_WEIGHT, Config.DEVICE_1, 
                    entropy_loss_flag=True)
        
        # 清理内存
        server_net.to('cpu')
        clear_gpu_memory()
        
        # Q_new步骤
        print('Q_new步骤:')
        new_server_gate_logits, expert_distribution = server_gate(server_net, fedavgloader, 'cpu')
        print(f'专家数据分布: {expert_distribution}')
        server_gate_new = new_server_gate_logits
        
        # 评估服务器模型
        evaluate_and_save_server_model(server_net, server_testloader, criterion, 
                                     epoch, best_model, c_to_s_logits, n_main_experts, result_path)
    
    return server_gate_new, c_to_s_logits


def update_server_experts(server_net: MoE, client_models: List[BertNet], 
                         c_to_s_logits: torch.Tensor, n_main_experts: int):
    """更新服务器专家网络参数"""
    server_net.to('cpu')
    
    # 更新专家网络
    for expert_idx in range(server_net.num_experts):
        expert_state = server_net.experts[expert_idx].state_dict()
        
        for key in expert_state.keys():
            weighted_sum = torch.zeros_like(expert_state[key])
            for client_idx in range(Config.NUM_ACTIVATED_CLIENTS):
                client_models[client_idx].to('cpu')
                weight = c_to_s_logits[expert_idx, client_idx].item()
                client_param = client_models[client_idx].state_dict()[key]
                weighted_sum += weight * client_param
            
            # 应用动量更新
            expert_state[key] = ((1 - Config.FEDAVG_WEIGHT_A) * expert_state[key] + 
                               Config.FEDAVG_WEIGHT_A * weighted_sum).clone()
        
        server_net.experts[expert_idx].load_state_dict(expert_state)
    
    # 更新主专家网络（如果存在）
    if n_main_experts == 1:
        fix_state = server_net.expert_fix[0].state_dict()
        
        for key in fix_state.keys():
            weighted_sum = torch.zeros_like(fix_state[key])
            for client_idx in range(Config.NUM_ACTIVATED_CLIENTS):
                client_param = client_models[client_idx].state_dict()[key]
                weighted_sum += 0.2 * client_param
            
            fix_state[key] = ((1 - Config.FEDAVG_WEIGHT_A) * fix_state[key] + 
                            Config.FEDAVG_WEIGHT_A * weighted_sum).clone()
        
        server_net.expert_fix[0].load_state_dict(fix_state)


def evaluate_and_save_server_model(server_net: MoE, server_testloader: DataLoader,
                                  criterion: nn.CrossEntropyLoss, epoch: int,
                                  best_model: Dict, c_to_s_logits: torch.Tensor,
                                  n_main_experts: int, result_path: str):
    """评估并保存服务器模型"""
    # 评估模型
    correct = 0
    total = 0
    val_loss = 0.0
    
    server_net.eval()
    server_net.to(Config.DEVICE_1)
    
    with torch.no_grad():
        for data in server_testloader:
            images, labels = data
            images, labels = images.to(Config.DEVICE_1), labels.to(Config.DEVICE_1)
            outputs, _, _, _ = server_net(images.view(images.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    cur_acc = correct / total
    cur_loss = val_loss / len(server_testloader)
    
    # 更新最佳模型
    if best_model['min_val_loss'] > cur_loss:
        best_model['min_val_loss'] = cur_loss
        best_model['max_val_acc'] = cur_acc
        best_model['epoch'] = epoch + 1
        
        print(f'*** 新的最佳模型 ***')
        print(f'Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}')
        
        # 保存模型
        save_models(server_net, c_to_s_logits, result_path)
    else:
        print(f'当前模型 - Epoch: {epoch + 1}, Loss: {cur_loss:.4f}, Acc: {cur_acc:.4f}')
        print(f'最佳模型 - Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}')


def save_models(server_net: MoE, c_to_s_logits: torch.Tensor, result_path: str):
    """保存服务器模型"""
    # 保存服务器模型
    save_path = f'{result_path}/best_val.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': server_net.state_dict(),
        'c_to_s': c_to_s_logits
    }, save_path)


def evaluate_final_model(server_net: MoE, test_set: TensorDataset, 
                        n_main_experts: int, result_path: str) -> float:
    """评估最终模型性能"""
    print("=" * 50)
    print("最终模型评估")
    print("=" * 50)
    
    testloader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    checkpoint = torch.load(f'{result_path}/best_val.pth')
    server_net.load_state_dict(checkpoint['model_state_dict'])
    
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    server_net.eval()
    server_net.to(Config.DEVICE_1)
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(Config.DEVICE_1), labels.to(Config.DEVICE_1)
            outputs, _, _, _ = server_net(images.view(images.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    final_acc = correct / total
    final_loss = val_loss / len(testloader)
    
    print(f'Alpha: {server_net.alpha if hasattr(server_net, "alpha") else "N/A"}')
    print(f'最终测试准确率: {final_acc:.4f}')
    print(f'主专家数量: {n_main_experts}')
    
    # 计算门控网络参数量
    gate_params = sum(p.numel() for p in server_net.w_gate.parameters())
    print(f'服务器门控网络参数量: {gate_params / 1_000_000:.2f}M')
    
    return final_acc


# ========================= 主训练流程 =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedMoE SENT140 训练参数')
    parser.add_argument('--nc', dest='num_clients', type=int, 
                       help='客户端数量', default=50)
    parser.add_argument('--nme', dest='n_main_experts', type=int,
                       help='主专家数量', default=0)
    parser.add_argument('--lr', dest='lr', type=float, 
                       help='学习率', default=5e-4)
    parser.add_argument('--cuda', dest='cuda', type=int, 
                       help='CUDA设备编号', default=0)
    parser.add_argument('--npos', dest='n_positive', type=int,
                       help='服务器正样本数', default=500)
    parser.add_argument('--nneg', dest='n_negative', type=int,
                       help='服务器负样本数', default=500)
    parser.add_argument('--a', dest='a', type=float, 
                       help='FedAvg权重参数', default=0.1)
    parser.add_argument('--entro', dest='entro_w', type=float, 
                       help='熵损失权重', default=1e-1)
    
    args = parser.parse_args()
    
    # 运行主函数
    results = main(
        num_clients=args.num_clients,
        n_main_experts=args.n_main_experts,
        lr=args.lr,
        device_index=args.cuda,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        a=args.a,
        entro_weight=args.entro_w
    )
    
    print("\n" + "=" * 50)
    print("训练结果汇总")
    print("=" * 50)
    for i, (gate_lr, acc, alpha, epoch) in enumerate(results):
        print(f"结果 {i+1}: Gate_LR={gate_lr}, Acc={acc:.4f}, Alpha={alpha:.4f}, Best_Epoch={epoch}")

