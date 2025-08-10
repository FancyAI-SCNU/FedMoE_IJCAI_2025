import os

import json
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
    kl_mean = 0.0
    total_alpha = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _, gates, alpha = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_alpha += alpha.item()

            one_hot_labels = F.one_hot(labels, num_classes=outputs.shape[-1])
            kl_mean += F.kl_div(outputs.log_softmax(dim=1), one_hot_labels.float().softmax(dim=1), reduction='batchmean')

    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    kl_mean = kl_mean / len(testloader)
    cur_alpha = total_alpha / len(testloader)
    return cur_loss, cur_acc, kl_mean, cur_alpha


def generate_dataset(client_train_num, server_train_num):
    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_data = datasets.MNIST(
        root="./cifar_utils/",
        train=True,
        transform=transform,
        download=True)

    test_data = datasets.MNIST(
        root="./cifar_utils/",
        train=False,
        transform=transform,
        download=True)

    train_sorted_data = {i: [] for i in range(10)}
    test_sorted_data = {i: [] for i in range(10)}

    # 将训练集中的数据按标签分组
    for image, label in train_data:
        train_sorted_data[label].append((image, label))

    # 将测试集中的数据按标签分组
    for image, label in test_data:
        test_sorted_data[label].append((image, label))

    client_train_num = client_train_num / 10
    server_train_num = server_train_num / 10
    train_subset = {}
    test_subset = {}
    test_set = []
    server_set = []
    validation_set = []
    for i in range(5):
        if client_train_num == 5000:
            train_subset[i] = train_sorted_data[2 * i][:len(train_sorted_data[2 * i]) - 1000] + \
                              train_sorted_data[2 * i + 1][:len(train_sorted_data[2 * i + 1]) - 1000]
        else:
            train_subset[i] = train_sorted_data[2 * i][:client_train_num] + \
                              train_sorted_data[2 * i + 1][:client_train_num]
        test_subset[i] = test_sorted_data[2 * i][:100] + test_sorted_data[2 * i + 1][:100]
        test_set += test_sorted_data[2 * i] + test_sorted_data[2 * i + 1]
        validation_set += train_sorted_data[2 * i][
                          len(train_sorted_data[2 * i]) - 1000:len(train_sorted_data[2 * i]) - 500] + \
                          train_sorted_data[2 * i + 1][
                          len(train_sorted_data[2 * i]) - 1000:len(train_sorted_data[2 * i]) - 500]
        server_set += train_sorted_data[2 * i][len(train_sorted_data[2 * i]) - server_train_num:] + train_sorted_data[2 * i + 1][len(
            train_sorted_data[2 * i + 1]) - server_train_num:]
    return train_subset, test_subset, test_set, validation_set, server_set


def test(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def validate(model, val_loader, device, criterion):
    model.eval()  # 将模型设置为评估模式

    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()
            preds = outputs.max(1)[1]
            correct += preds.eq(y).sum().item()

    val_acc = correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader)
    return val_loss, val_acc


def validate_moe(model, val_loader, device, criterion):
    model.eval()  # 将模型设置为评估模式

    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs, _, _ = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()
            preds = outputs.max(1)[1]
            correct += preds.eq(y).sum().item()

    val_acc = correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader)
    return val_loss, val_acc


def mlp_test(model, testloader, device_index='cuda:1'):
    device = torch.device(device_index)
    correct = 0
    total = 0
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def write_result(path, text):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w') as file:
            result = []
            result.append(text)
            json.dump(result, file, indent=4)
    else:
        with open(path, 'r') as file:
            data = json.load(file)
            data.append(text)
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class l2loss:
    def __init__(self):
        pass

    def __call__(self, model):
        regularization_loss = 0

        for name, param in model.named_parameters():
            regularization_loss += torch.sum(param ** 2)

        return regularization_loss


# 数据增广，对图片进行缩放裁剪. (使用BILINEAR进行图片插值 https://zhuanlan.zhihu.com/p/110754637)
def augment(data_list, dataset):
    augmented_data = []
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
            Cutout(n_holes=1, length=16),
        ])
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
    # sharp_tensor_1 = torch.zeros(gates.size(1), output_size, device=labels.device)
    for _, (gate, label) in enumerate(zip(gates, labels)):
        sharp_tensor[:, label.item()] += gate.squeeze()
        # gate = torch.where(gate != 0, torch.tensor(1, device=gate.device), gate)
        # sharp_tensor_1[:, label.item()] += gate.squeeze()
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
    from torch.utils.data.dataset import Subset
    if isinstance(dataset, Subset):
        # 获取原始数据集和子集的索引
        original_dataset = dataset.dataset
        indices = dataset.indices
        dataset_dict = {i: 0 for i in range(10)}
        for idx in indices:
            _, label = original_dataset[idx]
            dataset_dict[label] += 1
    else:
        # 直接统计普通数据集
        dataset_dict = {i: 0 for i in range(10)}
        for _, label in dataset[0]:
            dataset_dict[label] += 1
    return dataset_dict
