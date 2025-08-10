"""
    Dataset Loading and Preprocessing
"""
from torchvision import datasets, transforms
import numpy as np
from cifar_utils.cutout import Cutout
# from cutout import Cutout
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset


def generate_dataset(dataset, num_clients, server_per_calss):
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_data = datasets.MNIST(
            root="../data/",
            train=True,
            transform=transform_train,
            download=True)

        test_data = datasets.MNIST(
            root="./data/",
            train=False,
            transform=transform_test,
            download=True)

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
            Cutout(n_holes=1, length=16),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_data = datasets.CIFAR10(root='../cifar_exp/cifar_utils/CIFAR10/',
                                      train=True,
                                      download=False,
                                      transform=transform_train
                                      )
        valid_data = datasets.CIFAR10(root="../cifar_exp/cifar_utils/CIFAR10/",
                                      train=True,
                                      transform=transform_test,
                                      download=True)
        test_data = datasets.CIFAR10(root='../cifar_exp/cifar_utils/CIFAR10/',
                                     train=False,
                                     download=False,
                                     transform=transform_test
                                     )
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_sorted_data = {i: [] for i in range(10)}
    valid_sorted_data = {i: [] for i in range(10)}
    test_sorted_data = {i: [] for i in range(10)}

    print(len(train_data))
    print(len(test_data))
    # 将训练集中的数据按标签分组
    for image, label in train_data:
        train_sorted_data[label].append((image, label))
    print(len(train_sorted_data[0]))

    for image, label in valid_data:
        valid_sorted_data[label].append((image, label))

    # 将测试集中的数据按标签分组
    for image, label in test_data:
        test_sorted_data[label].append((image, label))

    # train_subset:{1:数字0 & 数字1, 2:数字2 & 数字3 ... 4:数字8 & 数字9}
    train_subset = {i: [] for i in range(num_clients)}
    test_subset = {i: [] for i in range(num_clients)}
    test_set = []
    server_set = []
    fedavg_set = []
    validation_set = []
    # 20用于fedavg, 5000用于validate, sever_perclass用于Server_Training
    server_set_flag = 20 + 500 + server_per_calss

    client_set = []
    server_train_dataset = {i: [] for i in range(10)}
    for i in range(10):
        server_train_dataset[i] = train_sorted_data[i][4000:]
        train_sorted_data[i] = train_sorted_data[i][:3480]

        client_set.append(split_list(train_sorted_data[i], num_clients / 5))

    for i in range(10):
        test_set += test_sorted_data[i]
        count = 0
        # 每轮的 data_list 为单个数字数据集
        data_list = server_train_dataset[i]
        valid_data_list = valid_sorted_data[i][3480:4000]
        for j in range(len(data_list)+len(valid_data_list)):
            if count < 20:
                fedavg_set.append(valid_data_list[j])
                count = count + 1
            elif 20 <= count < 520:
                validation_set.append(valid_data_list[j])
                # server_set.append(data_list[j])
                count = count + 1
            elif 500 <= count < server_set_flag:
                server_set.append(data_list[j-520])
                count = count + 1

    for i, data in enumerate(choice_dataset(client_set)):
        train_subset[i] = data
        test_subset[i] = validation_set

    return train_subset, test_subset, test_set, validation_set, server_set, fedavg_set

def split_list(data, num_parts):
    avg = len(data) / float(num_parts)
    out = []
    last = 0.0

    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg

    return out


def choice_dataset(dataset):
    temp = []
    for i in range(len(dataset[0])):
        for j in range(10):
            temp.append(dataset[j][i])
    # random.shuffle(temp)
    result = []
    for i in range(0, len(temp), 2):
        result.append(temp[i] + temp[i+1])
    return result


def dirichlet_split_noniid(train_labels, alpha, n_clients, n_server_samples_per_class):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    同时服务器端先拿每个类n_server_samples_per_class个训练样本，50 testset，20 fedavgset
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    server_idcs = []
    val_idcs = []
    fedavg_idcs = []

    # assign to server
    for k_idcs in class_idcs:
        server_idcs.extend(k_idcs[-n_server_samples_per_class:])
        val_idcs.extend(k_idcs[-1500:-1000])
        fedavg_idcs.extend(k_idcs[-1520:-1500])
    server_idcs = np.array(server_idcs)

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # 只使用剩余的数据进行分配
        k_idcs = np.setdiff1d(k_idcs, server_idcs)
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]



    return client_idcs, val_idcs, server_idcs, fedavg_idcs


def augment(data_list):
    augmented_data = []
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])
    if isinstance(data_list, Subset):
        # 获取原始数据集和子集的索引
        original_dataset = data_list.dataset
        indices = data_list.indices
        for idx in indices:
            image, label = original_dataset[idx]
            pil_image = transforms.ToPILImage()(image)
            augmented_image = transform_train(pil_image)
            augmented_data.append((augmented_image, label))
    else:
        for i in range(len(data_list)):
            image, label = data_list[i]
            pil_image = transforms.ToPILImage()(image)
            augmented_image = transform_train(pil_image)
            augmented_data.append((augmented_image, label))

    return augmented_data


def generate_cifar_dataset(n_clients, server_per_class):
    '''per client has all of the label of data'''
    dirichlet_alpha = 1.0
    seed = 42
    np.random.seed(seed)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # CIFAR10 Data Read
    train_data = datasets.CIFAR10(
        root='/home/yanzhao/tmp/pycharm_project_fedmoe/mixture-of-experts-master/cifar_exp/cifar_utils/CIFAR10',
        train=True,
        download=False,
        transform=transform_train
        )
    valid_data = datasets.CIFAR10(
        root='/home/yanzhao/tmp/pycharm_project_fedmoe/mixture-of-experts-master/cifar_exp/cifar_utils/CIFAR10',
        train=True,
        download=False,
        transform=transform_test
    )
    test_data = datasets.CIFAR10(
        root='/home/yanzhao/tmp/pycharm_project_fedmoe/mixture-of-experts-master/cifar_exp/cifar_utils/CIFAR10',
        train=False,
        download=False,
        transform=transform_test
        )
    labels = np.array(train_data.targets)

    # server端先拿每个类1520，总共15200条数据
    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    # 随后返回client_idcs，以及server_idcs
    client_idcs, val_idcs, server_idcs, fedavg_idcs = dirichlet_split_noniid(
        labels, alpha=dirichlet_alpha, n_clients=n_clients, n_server_samples_per_class=server_per_class)

    server_set = Subset(train_data, server_idcs)
    train_subset = [Subset(train_data, idcs) for idcs in client_idcs]
    test_set = test_data
    validation_set = Subset(valid_data, val_idcs)
    test_subset = [validation_set for _ in range(n_clients)]
    fedavg_set = Subset(valid_data, fedavg_idcs)

    # 进行数据增强
    # server_set = augment(server_set)

    return (train_subset, test_subset, test_set, server_set, validation_set, fedavg_set)


if __name__ == '__main__':
    # 使用方法
    n_clients = 10
    dirichlet_alpha = 1.0
    seed = 42
    np.random.seed(seed)
    n_classes = 10
    server_per_class = 250  # 留1000 * 10 + 5000 + 200 = 15200 数据
    #
    train_subset, test_subset, test_set, server_set, validation_set, fedavg_set = generate_cifar_dataset(
        n_clients=n_clients, server_per_class=server_per_class)

    