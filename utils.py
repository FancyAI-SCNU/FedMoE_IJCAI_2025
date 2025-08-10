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
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _, gates = model(images.view(images.shape[0], -1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def moe_cnn_test(model, testloader, device_index='cuda:1'):
    device = torch.device(device_index)
    model.to(device)
    correct = 0
    total = 0
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _, gates = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def moe_gpt_test(model, testloader, device):
    # device = torch.device(device_index)
    model.to(device)
    correct = 0
    total = 0
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)
            outputs, _, gates = model(input_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
            outputs = model(images.view(images.shape[0], -1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def cnn_test(model, testloader, device_index='cuda:1'):
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
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cur_loss = val_loss / len(testloader)
    cur_acc = correct / total
    return cur_loss, cur_acc


def gpt_test(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
            outputs, prob = model(images.view(images.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
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


def count_dataset(dataset, output_size=10):
    dataset_dict = {i: 0 for i in range(output_size)}
    for image, label in dataset:
        dataset_dict[label] += 1
    return dataset_dict