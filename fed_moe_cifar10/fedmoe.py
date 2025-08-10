"""
基于CIFAR-10数据集的联邦学习MoE (Mixture of Experts) 实现

使用ResNet-18模型进行图像分类任务的联邦学习框架
- Server: MoE结构，包含多个专家网络和固定专家
- Clients: ResNet-18模型，通过FedAvg进行参数聚合
"""

import os
import torch
from torch import nn, optim
import argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cifar_utils.utils import moe_test, l2loss, count_dataset, mlp_test
from train import fedmoe_train, server_gate, client_logits
from cifar_utils.fedmoe_dataset import generate_cifar_dataset
from moe import MoE
from model.resnet import ResNet18


# ========================= 配置参数 =========================
class Config:
    """配置类：集中管理所有超参数和路径配置"""
    
    # 数据集配置
    DATASET = 'cifar10'
    INPUT_CHANNEL = 3
    INPUT_WIDTH = 32
    INPUT_HEIGHT = 32
    OUTPUT_SIZE = 10
    HIDDEN_SIZE = 64
    
    # 模型配置
    NUM_EXPERTS = 2
    CLIENT_NUM_EXPERTS = 2
    SERVER_NUM_EXPERTS = 5
    K = 1  # Top-k专家
    
    # 联邦学习配置
    NUM_ACTIVATED_CLIENTS = 5
    BATCH_SIZE = 32
    
    # 训练配置
    TOTAL_EPOCHS = 100
    SERVER_EPOCH = 5
    CLIENT_EPOCH = 2
    ITERATION_EPOCH = 5
    FIX_EXPERT_EPOCH = 20
    
    # 损失权重配置
    AUX_WEIGHT = 1e-1
    L2_WEIGHT = 1e-7
    
    # 模型配置标志
    NOISY_GATING = True
    CLIENT_RANDOM_GATE = False
    SERVER_RANDOM_GATE = False
    
    # 优化器配置
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    EPS = 1e-6
    
    # 学习率配置
    SERVER_GATE_LR = 1e-4
    SERVER_ALPHA_LR = 5e-7
    FIX_EXPERT_LR_FACTOR = 1.0
    
    # FedAvg配置
    FEDAVG_PER_CLASS = 20
    
    # Alpha参数
    DEFAULT_ALPHA = 0.9
    
    # 调度器配置
    MILESTONES = [40, 80]
    GAMMA = 0.1
    
    @classmethod
    def get_input_config(cls, dataset: str) -> Dict[str, int]:
        """获取输入配置"""
        if dataset == 'mnist':
            return {
                'input_channel': 1,
                'input_w': 28,
                'input_h': 28,
                'hidden_size': 32
            }
        elif dataset == 'cifar10':
            return {
                'input_channel': cls.INPUT_CHANNEL,
                'input_w': cls.INPUT_WIDTH,
                'input_h': cls.INPUT_HEIGHT,
                'hidden_size': cls.HIDDEN_SIZE
            }
        else:
            raise ValueError(f"不支持的数据集: {dataset}")


# ========================= 工具函数 =========================
def setup_environment(device_index: int) -> Tuple[torch.device, torch.device]:
    """设置运行环境"""
    device = torch.device(f'cuda:{device_index}')
    device_cpu = torch.device('cpu')
    return device, device_cpu


def load_datasets(num_clients: int, server_per_class: int) -> Tuple[List, List, Any, Any, Any, Any]:
    """加载CIFAR-10数据集"""
    print("=" * 50)
    print("加载CIFAR-10数据集...")
    print("=" * 50)
    
    train_subset, test_subset, test_set, validation_set, server_set, fedavg_set = generate_cifar_dataset(
        n_clients=num_clients, server_per_class=server_per_class
    )
    
    print(f"数据集加载完成:")
    print(f"- 客户端数量: {num_clients}")
    print(f"- 服务器每类样本数: {server_per_class}")
    
    return train_subset, test_subset, test_set, validation_set, server_set, fedavg_set


def create_data_loaders(server_set: Any, validation_set: Any, fedavg_set: Any, 
                       batch_size: int) -> Tuple[Any, Any, Any]:
    """创建数据加载器"""
    fedavgloader = torch.utils.data.DataLoader(fedavg_set, batch_size=batch_size, shuffle=False)
    server_testloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    pretrain_loader = torch.utils.data.DataLoader(server_set, batch_size=256, shuffle=True)
    
    return fedavgloader, server_testloader, pretrain_loader


def load_pretrained_model(server_net: MoE, entro_weight: float, num_clients: int, 
                         server_per_class: int) -> bool:
    """加载预训练模型"""
    checkpoint_path = f'./entro_exp_result/entro_{entro_weight}_{num_clients}_nums_clients_server({server_per_class})/best_val.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"加载预训练模型: {checkpoint_path}")
        moe_checkpoint = torch.load(checkpoint_path)
        server_net.load_state_dict(moe_checkpoint['model_state_dict'])
        return True
    else:
        print("未找到预训练模型，使用随机初始化")
        return False


def initialize_server_model(num_clients: int, server_per_class: int, entro_weight: float,
                           lr: float, device: torch.device) -> Tuple[MoE, optim.Optimizer, Any, SummaryWriter]:
    """初始化服务器MoE模型"""
    print("=" * 50)
    print("初始化服务器MoE模型...")
    print("=" * 50)
    
    # 获取输入配置
    input_config = Config.get_input_config(Config.DATASET)
    
    # 创建MoE模型
    server_net = MoE(
        input_channel=input_config['input_channel'],
        input_w=input_config['input_w'],
        input_h=input_config['input_h'],
        output_size=Config.OUTPUT_SIZE,
        num_experts=Config.SERVER_NUM_EXPERTS,
        hidden_size=input_config['hidden_size'],
        noisy_gating=Config.NOISY_GATING,
        k=Config.K,
        random_gate=Config.SERVER_RANDOM_GATE,
        dataset=Config.DATASET
    )
    
    # 加载预训练模型
    load_pretrained_model(server_net, entro_weight, num_clients, server_per_class)
    
    # 设置alpha参数
    server_net.alpha = nn.Parameter(torch.tensor(Config.DEFAULT_ALPHA))
    
    # 配置优化器参数组
    param_groups = [
        {
            'params': server_net.w_gate.parameters(), 
            'lr': Config.SERVER_GATE_LR, 
            'eps': Config.EPS, 
            'weight_decay': Config.WEIGHT_DECAY
        },
        {
            'params': server_net.experts.parameters(), 
            'lr': lr, 
            'eps': Config.EPS, 
            'weight_decay': Config.WEIGHT_DECAY
        },
        {
            'params': server_net.expert_fix.parameters(), 
            'lr': 5e-5
        },
        {
            'params': [server_net.alpha], 
            'lr': Config.SERVER_ALPHA_LR
        }
    ]
    
    server_optimizer = optim.Adam(param_groups)
    
    # 配置学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=server_optimizer,
        T_max=Config.TOTAL_EPOCHS * Config.ITERATION_EPOCH
    )
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(
        f"../tensorboard_new_cifar/entro_{Config.DATASET}_resnet_a_{Config.FEDAVG_WEIGHT_A}_logs_{num_clients}_{server_per_class}_{entro_weight}_{Config.SERVER_ALPHA_LR}"
    )
    
    server_net.to(device)
    
    print(f"✓ 服务器MoE模型初始化完成，Alpha: {server_net.alpha.item():.4f}")
    
    return server_net, server_optimizer, scheduler, writer


def initialize_client_models(num_clients: int, lr: float, device_cpu: torch.device) -> Tuple[List[ResNet18], List[optim.Optimizer]]:
    """初始化客户端模型"""
    print("=" * 50)
    print("初始化客户端模型...")
    print("=" * 50)
    
    client_nets = []
    client_optimizers = []
    
    for i in range(num_clients):
        client_model = ResNet18(num_classes=Config.OUTPUT_SIZE)
        optimizer = optim.Adam(
            client_model.parameters(), 
            lr=lr, 
            eps=Config.EPS, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        client_model.to(device_cpu)
        client_nets.append(client_model)
        client_optimizers.append(optimizer)
    
    print(f"✓ {num_clients} 个客户端模型初始化完成")
    return client_nets, client_optimizers


def select_activated_clients(num_clients: int) -> List[int]:
    """选择激活的客户端"""
    activated_client_ids_step = []
    activated_client_ids = [0] * Config.NUM_ACTIVATED_CLIENTS
    
    for i in range(Config.NUM_ACTIVATED_CLIENTS):
        activated_client_ids_step.append(
            np.random.choice(int(num_clients / Config.NUM_ACTIVATED_CLIENTS), replace=False)
        )
        activated_client_ids[i] = activated_client_ids_step[i] * Config.NUM_ACTIVATED_CLIENTS + i
    
    return activated_client_ids


def train_clients(activated_client_ids: List[int], client_nets: List[ResNet18],
                 client_optimizers: List[optim.Optimizer], train_subset: List,
                 test_subset: List, device: torch.device, device_cpu: torch.device,
                 epoch: int, activated: Dict[int, int]) -> List[ResNet18]:
    """训练激活的客户端模型"""
    print('-' * 60)
    print('客户端训练阶段')
    print('-' * 60)
    
    client_models = []
    criterion = nn.CrossEntropyLoss()
    
    print(f'激活的客户端: {activated_client_ids}')
    
    for client_id in activated_client_ids:
        activated[client_id] += 1
        
        print(f'Epoch: {epoch + 1}, Client: {client_id + 1}')
        
        # 移动模型到GPU
        client_model = client_nets[client_id]
        client_model.to(device)
        optimizer = client_optimizers[client_id]
        
        # 创建数据加载器
        trainloader = torch.utils.data.DataLoader(
            train_subset[client_id], batch_size=Config.BATCH_SIZE, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            test_subset[client_id], batch_size=Config.BATCH_SIZE, shuffle=False
        )
        
        # 客户端训练
        for e in range(Config.CLIENT_EPOCH):
            for batch_idx, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # 评估客户端模型
        with torch.no_grad():
            _, client_acc = mlp_test(client_model, testloader, device)
            print(f'Client {client_id + 1} - Acc: {client_acc:.4f}')
        
        client_models.append(client_model)
    
    return client_models


def train_fix_expert(server_net: MoE, pretrain_loader: Any, server_testloader: Any,
                    lr: float, device: torch.device, epoch: int, iteration: int):
    """训练固定专家网络"""
    print(f'固定专家训练 - Epoch {epoch + 1}, Iteration {iteration + 1}')
    
    fix_expert_optimizer = optim.Adam(server_net.expert_fix.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for fix_epoch in range(Config.FIX_EXPERT_EPOCH):
        server_net.expert_fix.train()
        total_loss = 0.0
        
        for batch_idx, data in enumerate(pretrain_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            fix_expert_optimizer.zero_grad()
            outputs = server_net.expert_fix(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            fix_expert_optimizer.step()
            
            total_loss += loss.item()
        
        if (fix_epoch + 1) % 5 == 0:
            print(f"Fix Expert Epoch: {fix_epoch + 1}, Loss: {total_loss / len(pretrain_loader):.4f}")
    
    # 评估固定专家
    _, fix_acc = mlp_test(server_net.expert_fix, server_testloader, device)
    print(f"固定专家训练准确率: {fix_acc:.4f}")


def server_iteration_training(server_net: MoE, client_models: List[ResNet18],
                            fedavgloader: Any, server_testloader: Any, pretrain_loader: Any,
                            server_set: Any, server_optimizer: optim.Optimizer,
                            scheduler: Any, writer: SummaryWriter, epoch: int,
                            best_model: Dict, entro_weight: float, lr: float,
                            device: torch.device, a: float, num_clients: int,
                            server_per_class: int, activated_client_ids: List[int],
                            client_nets: List[ResNet18]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """服务器迭代训练过程"""
    print('-' * 60)
    print('服务器迭代训练阶段')
    print('-' * 60)
    
    # 获取客户端logits
    client_logits_combine = client_logits(client_models, fedavgloader, device)
    server_gate_new = []
    c_to_s_logits = None
    
    for iteration in range(Config.ITERATION_EPOCH):
        print(f'Epoch {epoch + 1}, Iteration {iteration + 1}')
        
        # Q步骤：获取服务器门控
        if iteration == 0:
            print('Q步骤:')
            server_gate_logits, _ = server_gate(server_net, fedavgloader, device)
        else:
            server_gate_logits = server_gate_new
        
        server_gate_logits_combine = torch.cat(server_gate_logits, dim=0)
        
        # FedAvg: 客户端到服务器聚合
        avg = torch.matmul(server_gate_logits_combine.T, client_logits_combine.T)
        c_to_s_logits = F.softmax(avg, dim=1)
        print(f'Client to Server权重: {c_to_s_logits.to(torch.float)}')
        
        # 更新服务器专家参数
        update_server_experts(server_net, client_models, c_to_s_logits, a)
        
        # 更新固定专家参数
        update_fix_expert(server_net, client_models)
        
        # 训练固定专家
        train_fix_expert(server_net, pretrain_loader, server_testloader, lr, device, epoch, iteration)
        
        # 服务器训练
        print(f'服务器训练 - Epoch {epoch + 1}, Iteration {iteration + 1}')
        fedmoe_train(
            server_net, server_set, Config.BATCH_SIZE, server_optimizer,
            Config.SERVER_EPOCH, entro_weight, Config.AUX_WEIGHT, Config.L2_WEIGHT,
            device, entropy_loss_flag=True, writer=writer,
            cur_epoch=epoch * Config.ITERATION_EPOCH + iteration, scheduler=scheduler
        )
        
        # Q_new步骤
        print('Q_new步骤:')
        new_server_gate_logits, expert_distribution = server_gate(server_net, fedavgloader, device)
        print(f'专家数据分布: {expert_distribution.to(torch.int)}')
        server_gate_new = new_server_gate_logits
        
        # 评估和保存模型
        evaluate_and_save_model(server_net, server_testloader, writer, epoch, iteration,
                               best_model, c_to_s_logits, entro_weight, num_clients,
                               server_per_class, activated_client_ids, client_nets, device)
    
    return server_gate_new, c_to_s_logits


def update_server_experts(server_net: MoE, client_models: List[ResNet18], 
                         c_to_s_logits: torch.Tensor, a: float):
    """更新服务器专家网络参数"""
    for expert_idx in range(server_net.num_experts):
        expert_state = server_net.experts[expert_idx].state_dict()
        
        for key in expert_state.keys():
            weighted_sum = torch.zeros_like(expert_state[key])
            for client_idx in range(Config.SERVER_NUM_EXPERTS):
                weight = c_to_s_logits[expert_idx, client_idx].item()
                client_param = client_models[client_idx].state_dict()[key]
                weighted_sum += weight * client_param
            
            # 应用FedAvg更新
            expert_state[key] = ((1 - a) * expert_state[key] + a * weighted_sum).clone()
        
        server_net.experts[expert_idx].load_state_dict(expert_state)


def update_fix_expert(server_net: MoE, client_models: List[ResNet18]):
    """更新固定专家网络参数"""
    fix_state = server_net.expert_fix.state_dict()
    new_state = {}
    
    for key in fix_state.keys():
        weighted_sum = torch.zeros_like(fix_state[key])
        for client_idx in range(Config.SERVER_NUM_EXPERTS):
            client_param = client_models[client_idx].state_dict()[key]
            weighted_sum += (1.0 / Config.SERVER_NUM_EXPERTS) * client_param
        new_state[key] = weighted_sum.clone()
    
    server_net.expert_fix.load_state_dict(new_state)


def evaluate_and_save_model(server_net: MoE, server_testloader: Any, writer: SummaryWriter,
                           epoch: int, iteration: int, best_model: Dict, c_to_s_logits: torch.Tensor,
                           entro_weight: float, num_clients: int, server_per_class: int,
                           activated_client_ids: List[int], client_nets: List[ResNet18],
                           device: torch.device):
    """评估并保存模型"""
    # 验证数据集上的评估
    cur_loss, cur_acc, kl_mean, alpha = moe_test(server_net, server_testloader, device)
    
    # 记录到TensorBoard
    global_step = epoch * Config.ITERATION_EPOCH + iteration
    writer.add_scalar("validate/validate_loss", cur_loss, global_step)
    writer.add_scalar("validate/validate_accuracy", cur_acc, global_step)
    writer.add_scalar("validate/kl_mean", kl_mean, global_step)
    writer.add_scalar('validate/alpha', alpha, global_step)
    
    # 更新最佳模型
    if best_model['min_val_loss'] > cur_loss:
        best_model['min_val_loss'] = cur_loss
        best_model['max_val_acc'] = cur_acc
        best_model['epoch'] = epoch + 1
        best_model['state_dict'] = server_net.state_dict()
        
        print(f'*** 新的最佳模型 ***')
        print(f'Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}, Alpha: {server_net.alpha.item():.4f}')
        
        # 保存服务器模型
        save_path = f'./entro_exp_result/entro_{entro_weight}_{num_clients}_nums_clients_server({server_per_class})/best_val.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': server_net.state_dict(),
            'c_to_s': c_to_s_logits
        }, save_path)
        
        # 保存客户端模型
        for client_id in activated_client_ids:
            client_save_path = f'./entro_exp_result/entro_{entro_weight}_{num_clients}_nums_clients_server({server_per_class})/client_{client_id + 1}_best_val.pth'
            os.makedirs(os.path.dirname(client_save_path), exist_ok=True)
            torch.save({'model_state_dict': client_nets[client_id].state_dict()}, client_save_path)
    else:
        print(f'当前模型 - Epoch: {epoch + 1}, Loss: {cur_loss:.4f}, Acc: {cur_acc:.4f}')
        print(f'最佳模型 - Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}, Alpha: {server_net.alpha.item():.4f}')


def fedavg_server_to_client(server_net: MoE, client_nets: List[ResNet18],
                           activated_client_ids: List[int], server_gate_new: List[torch.Tensor],
                           client_logits_combine: torch.Tensor, device_cpu: torch.device,
                           a: float):
    """FedAvg: 服务器到客户端参数聚合"""
    # 计算服务器到客户端的权重
    new_server_gate_logits_combine = torch.cat(server_gate_new, dim=0)
    new_avg = torch.matmul(new_server_gate_logits_combine.T, client_logits_combine.T)
    s_to_c_logits = F.softmax(new_avg, dim=0)
    
    # 添加固定专家权重
    fix_logits = torch.ones(1, s_to_c_logits.size(1), device=s_to_c_logits.device) * (1 - server_net.alpha)
    s_to_c_logits = F.softmax(torch.cat((s_to_c_logits * server_net.alpha, fix_logits), dim=0), dim=0)
    
    print(f'Server to Client权重: {s_to_c_logits.to(torch.float)}')
    
    # 更新客户端模型参数
    for i in range(Config.NUM_ACTIVATED_CLIENTS):
        client_state = client_nets[activated_client_ids[i]].state_dict()
        new_client_state = {}
        
        for key in client_state.keys():
            weighted_sum = torch.zeros_like(client_state[key])
            
            # 从专家网络聚合
            for expert_idx in range(server_net.num_experts):
                weight = s_to_c_logits[expert_idx, i].item()
                expert_param = server_net.experts[expert_idx].state_dict()[key]
                weighted_sum += weight * expert_param
            
            # 从固定专家聚合
            weight = s_to_c_logits[Config.SERVER_NUM_EXPERTS, i].item()
            fix_param = server_net.expert_fix.state_dict()[key]
            weighted_sum += weight * fix_param
            
            # 应用FedAvg更新
            new_client_state[key] = (a * client_state[key] + (1 - a) * weighted_sum).clone()
        
        client_nets[activated_client_ids[i]].load_state_dict(new_client_state)
    
    # 将客户端模型移回CPU
    for client_id in activated_client_ids:
        client_nets[client_id].to(device_cpu)


def evaluate_final_model(server_net: MoE, test_set: Any, best_model: Dict,
                        device: torch.device) -> Tuple[float, float]:
    """评估最终模型性能"""
    print("=" * 50)
    print("最终模型评估")
    print("=" * 50)
    
    server_net.load_state_dict(best_model['state_dict'])
    testloader = torch.utils.data.DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    _, acc, _, alpha = moe_test(server_net, testloader, device)
    
    print(f'最终测试准确率: {acc:.4f}')
    print(f'Alpha值: {alpha:.4f}')
    
    return acc, alpha


# ========================= 主函数 =========================
def main(num_clients: int, lr: float, device_index: int, server_per_class: int, 
         a: float, entro_weight: float) -> List[Tuple]:
    """主训练流程"""
    print("=" * 60)
    print("联邦学习MoE训练 - CIFAR-10数据集")
    print("=" * 60)
    
    # 环境设置
    device, device_cpu = setup_environment(device_index)
    
    # 数据准备
    train_subset, test_subset, test_set, validation_set, server_set, fedavg_set = load_datasets(
        num_clients, server_per_class)
    fedavgloader, server_testloader, pretrain_loader = create_data_loaders(
        server_set, validation_set, fedavg_set, Config.BATCH_SIZE)
    
    # 训练状态跟踪
    activated = {i: 0 for i in range(num_clients)}
    
    # 学习率网格搜索（这里简化为单个学习率）
    results = []
    for current_lr in [lr]:  # 可以扩展为学习率列表
        print(f"\n{'='*30} 学习率: {current_lr} {'='*30}")
        
        # 模型初始化
        server_net, server_optimizer, scheduler, writer = initialize_server_model(
            num_clients, server_per_class, entro_weight, current_lr, device)
        client_nets, client_optimizers = initialize_client_models(num_clients, current_lr, device_cpu)
        
        # 训练状态
        best_model = {'epoch': -1, 'min_val_loss': 1e6, 'max_val_acc': 0, 'state_dict': None}
        
        # 主训练循环
        print("=" * 50)
        print("开始联邦学习训练")
        print("=" * 50)
        
        for epoch in range(Config.TOTAL_EPOCHS):
            print(f"\n{'='*20} EPOCH {epoch + 1}/{Config.TOTAL_EPOCHS} {'='*20}")
            
            # 选择激活的客户端
            activated_client_ids = select_activated_clients(num_clients)
            
            # 客户端训练
            client_models = train_clients(activated_client_ids, client_nets, client_optimizers,
                                        train_subset, test_subset, device, device_cpu, epoch, activated)
            
            # 服务器迭代训练
            server_gate_new, c_to_s_logits = server_iteration_training(
                server_net, client_models, fedavgloader, server_testloader, pretrain_loader,
                server_set, server_optimizer, scheduler, writer, epoch, best_model,
                entro_weight, current_lr, device, a, num_clients, server_per_class,
                activated_client_ids, client_nets)
            
            # FedAvg: 服务器到客户端
            client_logits_combine = client_logits(client_models, fedavgloader, device)
            fedavg_server_to_client(server_net, client_nets, activated_client_ids,
                                   server_gate_new, client_logits_combine, device_cpu, a)
        
        # 关闭写入器
        writer.close()
        
        # 最终评估
        final_acc, final_alpha = evaluate_final_model(server_net, test_set, best_model, device)
        
        # 记录结果
        results.append((Config.SERVER_ALPHA_LR, final_acc, final_alpha, best_model['epoch']))
        
        print(f"\n学习率 {current_lr} 的训练完成")
        print(f"最佳Epoch: {best_model['epoch']}")
        print(f"最终准确率: {final_acc:.4f}")
        print(f"Alpha值: {final_alpha:.4f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedMoE CIFAR-10 训练参数')
    parser.add_argument('--nc', dest='num_clients', type=int, 
                       help='客户端数量', default=50)
    parser.add_argument('--lr', dest='lr', type=float, 
                       help='学习率', default=1e-4)
    parser.add_argument('--cuda', dest='cuda', type=int, 
                       help='CUDA设备编号', default=0)
    parser.add_argument('--spc', dest='server_per_class', type=int, 
                       help='服务器每类样本数', default=1000)
    parser.add_argument('--a', dest='a', type=float, 
                       help='FedAvg权重参数', default=0.1)
    parser.add_argument('--entro', dest='entro_w', type=float, 
                       help='熵损失权重', default=1e-3)
    
    args = parser.parse_args()
    
    # 运行主函数
    results = main(
        num_clients=args.num_clients,
        lr=args.lr,
        device_index=args.cuda,
        server_per_class=args.server_per_class,
        a=args.a,
        entro_weight=args.entro_w
    )
    
    print("\n" + "=" * 50)
    print("训练结果汇总")
    print("=" * 50)
    for i, (alpha_lr, acc, alpha, epoch) in enumerate(results):
        print(f"结果 {i+1}: Alpha_LR={alpha_lr}, Acc={acc:.4f}, Alpha={alpha:.4f}, Best_Epoch={epoch}")
