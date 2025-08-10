# Ester Hlav
# Oct 6, 2019
# utils.py 

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import torch
import torch.nn.functional as F

from torch import nn
from torchvision import datasets, transforms

criterion = nn.CrossEntropyLoss()


def moe_test(model, testloader, device_index='cuda:1'):
    device = torch.device(device_index)
    correct = 0
    total = 0
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, probs, _, gates = model(images.view(images.shape[0], -1))
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def mlp_test(model, testloader, device_index='cuda:1'):
    device = torch.device(device_index)
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, probs = model(images.view(images.shape[0], -1))
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def bert_test(model, testloader, device_index='cuda:1'):
    device = torch.device(device_index)
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, probs = model(images.view(images.shape[0], -1))
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def write_result(path, text):
    with open(path, 'w') as file:
        file.write(text)


class l2loss:
    def __init__(self):
        pass

    def __call__(self, model):
        regularization_loss = 0

        for name, param in model.named_parameters():
            regularization_loss += torch.sum(param ** 2)

        return regularization_loss


# 数据增广，对图片进行缩放裁剪
def augment(data_list, dataset):
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    augmented_data = []
    for i in range(len(data_list)):
        image, label = data_list[i]
        pil_image = transforms.ToPILImage()(image)
        augmented_image = transform_train(pil_image)
        augmented_data.append((augmented_image, label))

    return augmented_data


def cv_squared(x):
    eps = 1e-10

    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)


# 信息熵，使每个专家处理的数据分布更尖锐，即让专家专一
def infor_entropy(gates, labels, output_size):
    sharp_tensor = torch.zeros(gates.size(1), output_size, device=labels.device)
    for _, (gate, label) in enumerate(zip(gates, labels)):
        sharp_tensor[:, label.item()] += gate.squeeze()

    # 计算熵
    entropy = -torch.sum(F.softmax(sharp_tensor, dim=-1) * F.log_softmax(sharp_tensor, dim=-1), dim=-1)
    loss = torch.mean(entropy)

    return loss


# 使专家专一的同时让每个专家专一的数据不同（前面那个已经足够）
def infor_entropy_nb(gates, labels, output_size):
    sharp_tensor = torch.zeros(gates.size(1), output_size, device=labels.device)
    # sharp_tensor_1 = torch.zeros(gates.size(1), output_size, device=labels.device)
    for _, (gate, label) in enumerate(zip(gates, labels)):
        sharp_tensor[:, label.item()] += gate.squeeze()
        # gate = torch.where(gate != 0, torch.tensor(1, device=gate.device), gate)
        # sharp_tensor_1[:, label.item()] += gate.squeeze()
    # 计算熵
    entropy = -torch.sum(F.softmax(sharp_tensor, dim=-1) * F.log_softmax(sharp_tensor, dim=-1), dim=-1)
    loss = torch.mean(entropy) + cv_squared(entropy.sum(1))

    return loss


def count_dataset(dataset):
    dataset_dict = {i: 0 for i in range(10)}
    for image, label in dataset:
        dataset_dict[label] += 1
    return dataset_dict
def convertToTorchFloat(x, device):
    '''
        Converting np.array to torch.tensor (float) and, if availabe, convert to cuda.
        
        
        inputs:
            - x (np.array):            tensor
            
        return:
            - tensor (torch.tensor):   converted float tensor
    '''
    x = torch.tensor(x).float().to(device)
    return x

def convertToTorchInt(x, device):
    '''
        Converting np.array to torch.tensor (int64) and, if availabe, convert to cuda.
        
        
        inputs:
            - x (np.array):            tensor
            
        return:
            - tensor (torch.tensor):   converted int64 tensor
    '''
    x = np.array(x)
    print(x.shape)
    x = torch.tensor(x).to(torch.int64).to(device)
    return x


def plot_perf(history, final_perf):
    '''
        Function to plot performance plots, i.e. evolution of metrics during training
        on train, val and test sets.
    '''
    epochs = range(1, len(history['loss'])+1)
    for key in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history[key], '+-b', label=key)
        plt.plot(epochs, history['val_'+key], '+-g', label='val_'+key)
        plt.axhline(y=final_perf[key], color='r', linestyle='--', label='test_'+key)
        plt.legend()
        plt.title('Evolution of {} during training'.format(key))
        plt.plot()


def split_data_by_majority_class(Xtrain, Ytrain, Xtest, Ytest, num_clients, device, majority_ratio=0.8):
    """
    将数据集划分为多个客户端，每个客户端以某一类为主。
    """
    # 确保是二分类问题
    unique_classes = np.unique(Ytrain)
    assert len(unique_classes) == 2, "仅支持二分类任务"

    class0_indices = np.where(Ytrain == unique_classes[0])[0]
    class1_indices = np.where(Ytrain == unique_classes[1])[0]

    np.random.shuffle(class0_indices)
    np.random.shuffle(class1_indices)

    train_subset = {}
    test_subset = {}

    total_samples_per_client = len(Xtrain) // num_clients

    class0_count = int(total_samples_per_client * majority_ratio)
    class1_count = total_samples_per_client - class0_count

    for i in range(num_clients):
        if i % 2 == 0:
            major_class_indices = class0_indices[i * class0_count: (i + 1) * class0_count]
            minor_class_indices = class1_indices[i * class1_count: (i + 1) * class1_count]
        else:
            major_class_indices = class1_indices[i * class0_count: (i + 1) * class0_count]
            minor_class_indices = class0_indices[i * class1_count: (i + 1) * class1_count]

        selected_indices = np.concatenate([major_class_indices, minor_class_indices])
        np.random.shuffle(selected_indices)

        client_Xtrain = convertToTorchInt(Xtrain[selected_indices], device)
        client_Ytrain = convertToTorchInt(Ytrain[selected_indices], device)

        train_subset[i] = TensorDataset(client_Xtrain, client_Ytrain)

    client_Xtest = convertToTorchInt(Xtest, device)
    client_Ytest = convertToTorchInt(Ytest, device)
    test_data = TensorDataset(client_Xtest, client_Ytest)
    for i in range(num_clients):
        test_subset[i] = test_data

    return train_subset, test_subset


def create_balanced_dataset(X, y, n_positive, n_negative, device):
    """
    创建一个类别平衡的数据集，从原始数据中抽取指定数量的正类和负类样本。
    """
    # 获取正类和负类的索引
    positive_indices = (y == 1).nonzero(as_tuple=True)[0]
    negative_indices = (y == 0).nonzero(as_tuple=True)[0]

    # 检查是否有足够的样本
    assert len(positive_indices) >= n_positive, f"正类样本不足 {n_positive} 个，只有 {len(positive_indices)}"
    assert len(negative_indices) >= n_negative, f"负类样本不足 {n_negative} 个，只有 {len(negative_indices)}"

    # 随机选择指定数量的样本
    selected_positive = positive_indices[torch.randperm(len(positive_indices))[:n_positive]]
    selected_negative = negative_indices[torch.randperm(len(negative_indices))[:n_negative]]

    # 合并索引并打乱顺序
    selected_indices = torch.cat([selected_positive, selected_negative], dim=0)
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]

    # 提取对应的样本
    X_balanced = X[selected_indices].to(device)
    y_balanced = y[selected_indices].to(device)

    return TensorDataset(X_balanced, y_balanced)