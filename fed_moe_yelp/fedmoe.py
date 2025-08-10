import datetime
import gc
import os
import random
import argparse
from typing import List, Dict, Tuple, Any

import pandas as pd
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
import numpy as np
import torch.nn.functional as F

from utils import l2loss, gpt_test, moe_gpt_test
from fed_moe_yelp.model import GPT2
from fed_moe_yelp.moe import MoE
from fed_moe_yelp.train import fedmoe_train, client_logits, server_gate


# ========================= 配置参数 =========================
class Config:
    """配置类：集中管理所有超参数和路径配置"""
    
    # 数据集配置
    FEDAVG_WEIGHT_A = 0.1
    CLIENT_LR = 5e-4
    SERVER_GATE_LR = 1e-4
    NUM_CLIENTS = 100
    DATASET = 'yelp'
    INPUT_SIZE = 784  # 兼容性保留
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 5  # Yelp评分分类数量
    MAX_SEQUENCE_LENGTH = 64
    BATCH_SIZE = 1
    
    # 模型配置
    SERVER_NUM_EXPERTS = 3
    ALPHA = 0.5  # MoE混合系数
    
    # 训练配置
    TOTAL_EPOCHS = 5
    SERVER_EPOCH = 2
    CLIENT_EPOCH = 5
    ITERATION_EPOCH = 1
    
    # 损失权重配置
    AUX_WEIGHT = 1
    L2_WEIGHT = 1e-7
    
    # 模型配置
    NOISY_GATING = True
    RANDOM_GATE = True
    ENTROPY_LOSS_FLAG = True
    
    # 设备配置
    SERVER_DEVICE = 'cuda:3'
    CLIENT_DEVICES = ['cuda:0', 'cuda:1', 'cuda:2']
    
    # 路径配置
    DATA_ROOT = '../data/yelp'
    MODEL_ROOT = '../result/yelp'
    GPT2_MODEL_PATH = "../gpt2-medium"
    RESULT_PATH_TEMPLATE = '../result/yelp/fedmoe_clients_{}_lr_{}_a_{}_entro_{}'


# ========================= 数据处理类 =========================
class YelpDataset(Dataset):
    """Yelp数据集处理类"""
    
    def __init__(self, tokenizer: GPT2Tokenizer, data: pd.DataFrame, max_length: int = Config.MAX_SEQUENCE_LENGTH):
        """
        初始化Yelp数据集
        
        Args:
            tokenizer: GPT2分词器
            data: 包含text和label列的DataFrame
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self.data):
            raise IndexError(f"索引 {idx} 超出数据集范围 {len(self.data)}")
        
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ========================= 工具函数 =========================
def setup_environment() -> Tuple[str, SummaryWriter]:
    """设置运行环境"""
    datetime_now = datetime.datetime.now().strftime('%m-%d-%H-%M')
    writer = SummaryWriter(f'{Config.MODEL_ROOT}/fedmoe/{datetime_now}')
    return datetime_now, writer


def load_tokenizer() -> GPT2Tokenizer:
    """加载并配置GPT2分词器"""
    print("=" * 50)
    print("加载GPT2分词器...")
    print("=" * 50)
    
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_datasets(tokenizer: GPT2Tokenizer) -> Tuple[List[YelpDataset], YelpDataset, YelpDataset, YelpDataset, YelpDataset]:
    """加载所有数据集"""
    print("=" * 50)
    print("加载Yelp数据集...")
    print("=" * 50)
    
    # 服务器端数据集
    server_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/server_train.csv'))
    validation_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/validation.csv'))
    fedavg_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/fedavg.csv'))
    test_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/test.csv'))
    
    # 客户端数据集
    train_subset = []
    for i in range(Config.NUM_CLIENTS):
        client_data = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/client_{i + 1}_train.csv'))
        train_subset.append(client_data)
    
    print(f"数据集加载完成:")
    print(f"- 服务器训练集: {len(server_set)} 样本")
    print(f"- 验证集: {len(validation_set)} 样本")
    print(f"- FedAvg集: {len(fedavg_set)} 样本")
    print(f"- 测试集: {len(test_set)} 样本")
    print(f"- 客户端数据集: {Config.NUM_CLIENTS} 个")
    
    return train_subset, server_set, validation_set, fedavg_set, test_set


def create_data_loaders(server_set: YelpDataset, validation_set: YelpDataset, 
                       fedavg_set: YelpDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    server_trainloader = DataLoader(server_set, batch_size=Config.BATCH_SIZE, shuffle=True)
    server_testloader = DataLoader(validation_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    fedavgloader = DataLoader(fedavg_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return server_trainloader, server_testloader, fedavgloader


def load_pretrained_checkpoints() -> Dict[str, Any]:
    """加载预训练检查点"""
    print("=" * 50)
    print("加载预训练模型检查点...")
    print("=" * 50)
    
    checkpoints = {}
    
    # 加载各种预训练模型
    try:
        checkpoints['expert'] = torch.load(f'{Config.MODEL_ROOT}/expert/best_val.pth')
        checkpoints['gate'] = torch.load(f'{Config.MODEL_ROOT}/fedmoe_gate/best_val.pth')
        
        # 加载各个专家的检查点
        checkpoints['experts'] = []
        for i in range(Config.SERVER_NUM_EXPERTS):
            expert_checkpoint = torch.load(f'{Config.MODEL_ROOT}/fedmoe_expert_{i}/best_val.pth')
            checkpoints['experts'].append(expert_checkpoint)
            
        print("所有检查点加载成功")
    except FileNotFoundError as e:
        print(f"警告: 部分检查点文件未找到: {e}")
    
    return checkpoints


def initialize_server_model(checkpoints: Dict[str, Any]) -> Tuple[MoE, optim.Optimizer, lr_scheduler.StepLR]:
    """初始化服务器MoE模型"""
    print("=" * 50)
    print("初始化服务器MoE模型...")
    print("=" * 50)
    
    # 创建MoE模型
    server_net = MoE(
        input_size=Config.INPUT_SIZE,
        output_size=Config.OUTPUT_SIZE,
        num_experts=Config.SERVER_NUM_EXPERTS,
        hidden_size=Config.HIDDEN_SIZE,
        noisy_gating=Config.NOISY_GATING,
        k=1,
        random_gate=Config.RANDOM_GATE,
        alpha=Config.ALPHA
    )
    
    # 加载预训练权重
    if 'expert' in checkpoints:
        server_net.expert_fix.load_state_dict(checkpoints['expert']['model_state_dict'])
        print("✓ 固定专家权重加载完成")
    
    if 'gate' in checkpoints:
        server_net.w_gate.load_state_dict(checkpoints['gate']['model_state_dict'])
        server_net.w_noise.load_state_dict(checkpoints['gate']['model_state_dict'])
        print("✓ 门控网络权重加载完成")
    
    if 'experts' in checkpoints:
        for i, expert_checkpoint in enumerate(checkpoints['experts']):
            server_net.experts[i].load_state_dict(expert_checkpoint['model_state_dict'])
        print(f"✓ {len(checkpoints['experts'])} 个专家权重加载完成")
    
    # 移动到指定设备
    server_net.to(Config.SERVER_DEVICE)
    
    # 配置优化器
    param_groups = [
        {'params': server_net.w_gate.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': server_net.w_noise.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': [server_net.alpha], 'lr': 1e-5},
    ]
    
    optimizer = optim.SGD(param_groups)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    return server_net, optimizer, scheduler


def initialize_client_models(checkpoints: Dict[str, Any]) -> Tuple[List[GPT2], List[optim.Optimizer]]:
    """初始化客户端模型"""
    print("=" * 50)
    print("初始化客户端模型...")
    print("=" * 50)
    
    client_nets = []
    client_optimizers = []
    
    for i in range(Config.NUM_CLIENTS):
        # 创建客户端模型
        client_model = GPT2(num_classes=Config.OUTPUT_SIZE)
        
        # 加载预训练权重
        if 'expert' in checkpoints:
            client_model.load_state_dict(checkpoints['expert']['model_state_dict'])
        
        # 创建优化器
        optimizer = optim.SGD(client_model.parameters(), lr=Config.CLIENT_LR)
        
        client_nets.append(client_model)
        client_optimizers.append(optimizer)
    
    print(f"✓ {Config.NUM_CLIENTS} 个客户端模型初始化完成")
    return client_nets, client_optimizers


def train_clients(activated_client_ids: List[int], client_nets: List[GPT2], 
                 client_optimizers: List[optim.Optimizer], train_subset: List[YelpDataset],
                 validation_set: YelpDataset, epoch: int) -> List[GPT2]:
    """训练激活的客户端模型"""
    print('-' * 60)
    print('客户端训练阶段')
    print('-' * 60)
    
    client_models = []
    criterion = nn.CrossEntropyLoss()
    
    for client_id in activated_client_ids:
        print(f'Epoch: {epoch + 1}, Client: {client_id + 1}')
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_model = client_nets[client_id]
        client_model.to(device)
        optimizer = client_optimizers[client_id]
        
        # 创建数据加载器
        trainloader = DataLoader(train_subset[client_id], batch_size=Config.BATCH_SIZE, shuffle=True)
        testloader = DataLoader(validation_set, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        # 客户端训练循环
        total_loss = 0.0
        for e in range(Config.CLIENT_EPOCH):
            epoch_loss = 0.0
            for batch_idx, data in enumerate(trainloader):
                # 数据准备
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                labels = data["labels"].to(device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = client_model(input_ids)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
        
        # 评估客户端模型
        _, client_acc = gpt_test(client_model, testloader, device)
        avg_loss = total_loss / (Config.CLIENT_EPOCH * len(trainloader))
        
        print(f'Client {client_id + 1} - Loss: {avg_loss:.4f}, Acc: {client_acc:.4f}')
        
        # 移回CPU并添加到列表
        client_model.to('cpu')
        client_models.append(client_model)
    
    return client_models


def main(num_clients: int, lr: float, device_index: int, a: float, 
         entro_weight: float) -> List[Tuple]:
    """主训练流程"""
    print("=" * 60)
    print("联邦学习MoE训练 - Yelp数据集")
    print("=" * 60)
    
    # 更新配置
    Config.NUM_CLIENTS = num_clients
    Config.NUM_ACTIVATED_CLIENTS = min(3, num_clients)
    Config.CLIENT_LR = lr
    Config.SERVER_GATE_LR = lr * 0.01  # 门控网络使用更小的学习率
    Config.SERVER_EXPERT_LR = lr * 10   # 专家网络使用更大的学习率
    Config.FEDAVG_WEIGHT_A = a
    Config.FEDAVG_WEIGHT_B = a
    Config.ENTROPY_LOSS_WEIGHT = entro_weight
    
    # 设置设备配置
    Config.SERVER_DEVICE = f'cuda:{device_index}'
    Config.CLIENT_DEVICES = [f'cuda:{i}' for i in range(max(1, device_index))]
    if not Config.CLIENT_DEVICES:
        Config.CLIENT_DEVICES = [Config.SERVER_DEVICE]
    
    # 设置结果保存路径
    result_path = Config.RESULT_PATH_TEMPLATE.format(
        num_clients, lr, a, entro_weight
    )
    
    # 环境设置
    datetime_now = datetime.datetime.now().strftime('%m-%d-%H-%M')
    writer = SummaryWriter(f'{result_path}/tensorboard_{datetime_now}')
    
    # 数据准备
    print("=" * 50)
    print("加载GPT2分词器...")
    print("=" * 50)
    
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("=" * 50)
    print("加载Yelp数据集...")
    print("=" * 50)
    
    # 服务器端数据集
    server_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/server_train.csv'))
    validation_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/validation.csv'))
    fedavg_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/fedavg.csv'))
    test_set = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/test.csv'))
    
    # 客户端数据集
    train_subset = []
    for i in range(Config.NUM_CLIENTS):
        client_data = YelpDataset(tokenizer, pd.read_csv(f'{Config.DATA_ROOT}/client_{i + 1}_train.csv'))
        train_subset.append(client_data)
    
    print(f"数据集加载完成:")
    print(f"- 服务器训练集: {len(server_set)} 样本")
    print(f"- 验证集: {len(validation_set)} 样本")
    print(f"- FedAvg集: {len(fedavg_set)} 样本")
    print(f"- 测试集: {len(test_set)} 样本")
    print(f"- 客户端数据集: {Config.NUM_CLIENTS} 个")
    
    # 创建数据加载器
    server_trainloader = DataLoader(server_set, batch_size=Config.BATCH_SIZE, shuffle=True)
    server_testloader = DataLoader(validation_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    fedavgloader = DataLoader(fedavg_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 模型初始化
    print("=" * 50)
    print("加载预训练模型检查点...")
    print("=" * 50)
    
    checkpoints = {}
    
    # 加载各种预训练模型
    try:
        checkpoints['expert'] = torch.load(f'{Config.MODEL_ROOT}/expert/best_val.pth')
        checkpoints['gate'] = torch.load(f'{Config.MODEL_ROOT}/fedmoe_gate/best_val.pth')
        
        # 加载各个专家的检查点
        checkpoints['experts'] = []
        for i in range(Config.SERVER_NUM_EXPERTS):
            expert_checkpoint = torch.load(f'{Config.MODEL_ROOT}/fedmoe_expert_{i}/best_val.pth')
            checkpoints['experts'].append(expert_checkpoint)
            
        print("所有检查点加载成功")
    except FileNotFoundError as e:
        print(f"警告: 部分检查点文件未找到: {e}")
    
    print("=" * 50)
    print("初始化服务器MoE模型...")
    print("=" * 50)
    
    # 创建MoE模型
    server_net = MoE(
        input_size=Config.INPUT_SIZE,
        output_size=Config.OUTPUT_SIZE,
        num_experts=Config.SERVER_NUM_EXPERTS,
        hidden_size=Config.HIDDEN_SIZE,
        noisy_gating=Config.NOISY_GATING,
        k=1,
        random_gate=Config.RANDOM_GATE,
        alpha=Config.ALPHA
    )
    
    # 加载预训练权重
    if 'expert' in checkpoints:
        server_net.expert_fix.load_state_dict(checkpoints['expert']['model_state_dict'])
        print("✓ 固定专家权重加载完成")
    
    if 'gate' in checkpoints:
        server_net.w_gate.load_state_dict(checkpoints['gate']['model_state_dict'])
        server_net.w_noise.load_state_dict(checkpoints['gate']['model_state_dict'])
        print("✓ 门控网络权重加载完成")
    
    if 'experts' in checkpoints:
        for i, expert_checkpoint in enumerate(checkpoints['experts']):
            server_net.experts[i].load_state_dict(expert_checkpoint['model_state_dict'])
        print(f"✓ {len(checkpoints['experts'])} 个专家权重加载完成")
    
    # 移动到指定设备
    server_net.to(Config.SERVER_DEVICE)
    
    # 配置优化器
    param_groups = [
        {'params': server_net.w_gate.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': server_net.w_noise.parameters(), 'lr': Config.SERVER_GATE_LR},
        {'params': [server_net.alpha], 'lr': 1e-5},
    ]
    
    server_optimizer = optim.SGD(param_groups)
    scheduler = lr_scheduler.StepLR(server_optimizer, step_size=10, gamma=0.7)
    
    print("=" * 50)
    print("初始化客户端模型...")
    print("=" * 50)
    
    client_nets = []
    client_optimizers = []
    
    for i in range(Config.NUM_CLIENTS):
        # 创建客户端模型
        client_model = GPT2(num_classes=Config.OUTPUT_SIZE)
        
        # 加载预训练权重
        if 'expert' in checkpoints:
            client_model.load_state_dict(checkpoints['expert']['model_state_dict'])
        
        # 创建优化器
        optimizer = optim.SGD(client_model.parameters(), lr=Config.CLIENT_LR)
        
        client_nets.append(client_model)
        client_optimizers.append(optimizer)
    
    print(f"✓ {Config.NUM_CLIENTS} 个客户端模型初始化完成")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # 训练状态跟踪
    activated = {i: 0 for i in range(Config.NUM_CLIENTS)}
    best_model = {'epoch': -1, 'min_val_loss': 1e6, 'max_val_acc': 0}
    
    # 服务器预训练
    print("=" * 50)
    print("服务器预训练")
    print("=" * 50)
    fedmoe_train(server_net, scaler, server_set, Config.BATCH_SIZE, server_optimizer, 1, 
                Config.ENTROPY_LOSS_WEIGHT, Config.AUX_WEIGHT, Config.L2_WEIGHT, 
                Config.SERVER_DEVICE, entropy_loss_flag=Config.ENTROPY_LOSS_FLAG)
    server_net = server_net.to('cpu')
    
    # 主训练循环
    print("=" * 50)
    print("开始联邦学习训练")
    print("=" * 50)
    
    results = []
    
    for epoch in range(Config.TOTAL_EPOCHS):
        print(f"\n{'='*20} EPOCH {epoch + 1}/{Config.TOTAL_EPOCHS} {'='*20}")
        
        scheduler.step()
        
        # 随机选择激活的客户端
        activated_client_ids = np.random.choice(
            Config.NUM_CLIENTS, Config.NUM_ACTIVATED_CLIENTS, replace=False
        ).tolist()
        
        for client_id in activated_client_ids:
            activated[client_id] += 1
        
        # 客户端训练
        client_models = train_clients(activated_client_ids, client_nets, client_optimizers,
                                    train_subset, validation_set, epoch)
        
        # 服务器迭代训练
        server_gate_new, c_to_s_logits = server_iteration_training(
            server_net, client_models, fedavgloader, server_testloader,
            epoch, writer, best_model, result_path)
        
        # 保存客户端模型
        save_client_models(client_nets, activated_client_ids, epoch, result_path)
        
        # FedAvg: 服务器到客户端
        client_logits_combine = client_logits(client_models, fedavgloader, 'cpu')
        fedavg_server_to_client(server_net, client_nets, activated_client_ids,
                               server_gate_new, client_logits_combine)
        
        # 清理内存
        del client_models
        cleanup_memory()
    
    # 最终评估
    print("=" * 50)
    print("最终模型评估")
    print("=" * 50)
    
    testloader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    checkpoint = torch.load(f'{result_path}/best_val.pth')
    server_net.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc = moe_gpt_test(server_net, testloader, Config.SERVER_DEVICE)
    final_alpha = server_net.alpha.item() if hasattr(server_net, 'alpha') else Config.ALPHA
    
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


def server_iteration_training(server_net: MoE, client_models: List[GPT2], 
                            fedavgloader: DataLoader, server_testloader: DataLoader,
                            epoch: int, writer: SummaryWriter, 
                            best_model: Dict[str, Any], result_path: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """服务器迭代训练过程"""
    print('-' * 60)
    print('服务器迭代训练阶段')
    print('-' * 60)
    
    # 获取客户端logits
    client_logits_combine = client_logits(client_models, fedavgloader, 'cpu')
    server_gate_new = []
    c_to_s_logits = None
    
    for iteration in range(Config.ITERATION_EPOCH):
        print(f'Epoch {epoch + 1}, Iteration {iteration + 1}')
        
        # Q步骤：获取服务器门控
        if iteration == 0:
            print('Q步骤:')
            server_gate_logits, _ = server_gate(server_net, fedavgloader, 'cpu')
        else:
            server_gate_logits = server_gate_new
        
        server_gate_logits_combine = torch.cat(server_gate_logits, dim=0)
        
        # FedAvg: 客户端到服务器聚合
        avg = torch.matmul(server_gate_logits_combine.T, client_logits_combine.T)
        c_to_s_logits = F.softmax(avg, dim=1)
        print(f'Client to Server权重: {c_to_s_logits.to(torch.float)}')
        
        # 更新服务器专家参数
        update_server_experts(server_net, client_models, c_to_s_logits)
        
        # Q_new步骤
        print('Q_new步骤:')
        new_server_gate_logits, expert_distribution = server_gate(server_net, fedavgloader, 'cpu')
        print(f'专家数据分布: {expert_distribution}')
        server_gate_new = new_server_gate_logits
        
        # 评估服务器模型
        evaluate_and_save_server_model(server_net, server_testloader, epoch, writer, 
                                     best_model, c_to_s_logits, result_path)
    
    return server_gate_new, c_to_s_logits


def update_server_experts(server_net: MoE, client_models: List[GPT2], c_to_s_logits: torch.Tensor):
    """更新服务器专家网络参数"""
    # 更新专家网络
    for expert_idx in range(server_net.num_experts):
        expert_state = server_net.experts[expert_idx].state_dict()
        new_state = {}
        
        for key in expert_state.keys():
            weighted_sum = torch.zeros_like(expert_state[key])
            for client_idx in range(Config.SERVER_NUM_EXPERTS):
                weight = c_to_s_logits[expert_idx, client_idx].item()
                client_param = client_models[client_idx].state_dict()[key]
                weighted_sum += weight * client_param
            
            # 应用动量更新
            new_state[key] = ((1 - Config.FEDAVG_WEIGHT_A) * expert_state[key] + 
                             Config.FEDAVG_WEIGHT_A * weighted_sum).clone()
        
        server_net.experts[expert_idx].load_state_dict(new_state)
    
    # 更新固定专家网络
    fix_state = server_net.expert_fix.state_dict()
    new_fix_state = {}
    
    for key in fix_state.keys():
        weighted_sum = torch.zeros_like(fix_state[key])
        for client_idx in range(Config.SERVER_NUM_EXPERTS):
            client_param = client_models[client_idx].state_dict()[key]
            weighted_sum += (1.0 / Config.SERVER_NUM_EXPERTS) * client_param
        
        new_fix_state[key] = ((1 - Config.FEDAVG_WEIGHT_A) * fix_state[key] + 
                             Config.FEDAVG_WEIGHT_A * weighted_sum).clone()
    
    server_net.expert_fix.load_state_dict(new_fix_state)


def evaluate_and_save_server_model(server_net: MoE, server_testloader: DataLoader, 
                                  epoch: int, writer: SummaryWriter,
                                  best_model: Dict[str, Any], c_to_s_logits: torch.Tensor,
                                  result_path: str):
    """评估并保存服务器模型"""
    cur_loss, cur_acc = moe_gpt_test(server_net, server_testloader, Config.SERVER_DEVICE)
    
    # 记录到tensorboard
    writer.add_scalar('valid_loss', cur_loss, epoch + 1)
    writer.add_scalar('alpha', server_net.alpha, epoch + 1)
    
    # 更新最佳模型
    if best_model['min_val_loss'] > cur_loss:
        best_model['min_val_loss'] = cur_loss
        best_model['max_val_acc'] = cur_acc
        best_model['epoch'] = epoch + 1
        
        print(f'*** 新的最佳模型 ***')
        print(f'Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}, Alpha: {server_net.alpha:.4f}')
        
        # 保存模型组件
        save_server_model_components(server_net, c_to_s_logits, result_path)
    else:
        print(f'当前模型 - Epoch: {epoch + 1}, Loss: {cur_loss:.4f}, Acc: {cur_acc:.4f}')
        print(f'最佳模型 - Epoch: {best_model["epoch"]}, Loss: {best_model["min_val_loss"]:.4f}, '
              f'Acc: {best_model["max_val_acc"]:.4f}')


def save_server_model_components(server_net: MoE, c_to_s_logits: torch.Tensor, result_path: str):
    """保存服务器模型的各个组件"""
    save_info = {
        'c_to_s': c_to_s_logits,
        'alpha': server_net.alpha
    }
    
    # 保存完整模型
    save_path = f'{result_path}/best_val.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({**save_info, 'model_state_dict': server_net.state_dict()}, save_path)
    
    # 保存各个组件
    components = [
        ('expert_fix', server_net.expert_fix),
        ('gate', server_net.w_gate),
        ('noise', server_net.w_noise)
    ]
    
    for name, component in components:
        save_path = f'{result_path}/fedmoe_{name}/best_val.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({**save_info, 'model_state_dict': component.state_dict()}, save_path)
    
    # 保存各个专家
    for i in range(server_net.num_experts):
        save_path = f'{result_path}/fedmoe_expert_{i}/best_val.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({**save_info, 'model_state_dict': server_net.experts[i].state_dict()}, save_path)


def save_client_models(client_nets: List[GPT2], activated_client_ids: List[int], 
                      epoch: int, result_path: str):
    """保存客户端模型"""
    for client_id in activated_client_ids:
        save_path = f'{result_path}/client_{client_id + 1}_ground_{epoch + 1}_best_val.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model_state_dict': client_nets[client_id].state_dict()}, save_path)


def fedavg_server_to_client(server_net: MoE, client_nets: List[GPT2], 
                           activated_client_ids: List[int], server_gate_new: List[torch.Tensor],
                           client_logits_combine: torch.Tensor):
    """FedAvg: 服务器到客户端参数聚合"""
    # 计算服务器到客户端的权重
    server_net.to('cpu')
    new_server_gate_logits_combine = torch.cat(server_gate_new, dim=0)
    new_avg = torch.matmul(new_server_gate_logits_combine.T, client_logits_combine.T)
    s_to_c_logits = F.softmax(new_avg, dim=0)
    
    # 添加固定专家的权重
    fix_logits = torch.ones(1, s_to_c_logits.size(1)) * (1 - server_net.alpha)
    s_to_c_logits = F.softmax(torch.cat((s_to_c_logits * server_net.alpha, fix_logits), dim=0), dim=0)
    
    print(f'Server to Client权重: {s_to_c_logits.to(torch.float)}')
    
    # 更新客户端模型参数
    for i in range(Config.SERVER_NUM_EXPERTS):
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
            fix_weight = s_to_c_logits[server_net.num_experts, i].item()
            fix_param = server_net.expert_fix.state_dict()[key]
            weighted_sum += fix_weight * fix_param
            
            # 应用动量更新
            new_client_state[key] = (Config.FEDAVG_WEIGHT_A * client_state[key] + 
                                   (1 - Config.FEDAVG_WEIGHT_A) * weighted_sum).clone()
        
        client_nets[activated_client_ids[i]].load_state_dict(new_client_state)
        client_nets[activated_client_ids[i]].to('cpu')


def cleanup_memory():
    """清理GPU内存"""
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_final_model(server_net: MoE, test_set: YelpDataset) -> float:
    """评估最终模型性能"""
    print("=" * 50)
    print("最终模型评估")
    print("=" * 50)
    
    testloader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    checkpoint = torch.load(f'{Config.MODEL_ROOT}/fedmoe/best_val.pth')
    server_net.load_state_dict(checkpoint['model_state_dict'])
    
    _, acc = moe_gpt_test(server_net, testloader, Config.SERVER_DEVICE)
    print(f'最终测试准确率: {acc:.4f}')
    
    return acc


# ========================= 主训练流程 =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedMoE Yelp 训练参数')
    parser.add_argument('--nc', dest='num_clients', type=int, 
                       help='客户端数量', default=10)
    parser.add_argument('--lr', dest='lr', type=float, 
                       help='学习率', default=3e-4)
    parser.add_argument('--cuda', dest='cuda', type=int, 
                       help='CUDA设备编号', default=3)
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
        a=args.a,
        entro_weight=args.entro_w
    )
    
    print("\n" + "=" * 50)
    print("训练结果汇总")
    print("=" * 50)
    for i, (gate_lr, acc, alpha, epoch) in enumerate(results):
        print(f"结果 {i+1}: Gate_LR={gate_lr}, Acc={acc:.4f}, Alpha={alpha:.4f}, Best_Epoch={epoch}")
