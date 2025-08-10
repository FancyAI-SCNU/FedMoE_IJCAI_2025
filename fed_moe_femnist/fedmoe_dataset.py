import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import numpy as np
import random
import matplotlib.pyplot as plt

'''
first of all, run 'git clone https://github.com/TalwalkarLab/leaf.git'
assumes that the script is run in the femnist folder
Then run './preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.9'.
Next, run the code below for testing.
'''

# 对训练数据进行数据增强
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class FemnistDataset(Dataset):
    def __init__(self, data, transform_custom=None):
        self.data = data
        if transform_custom is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform_custom

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        image = np.array(self.data['x'][idx], dtype=np.float32).reshape(28, 28)
        pil_image = transforms.ToPILImage()(image)
        label = self.data['y'][idx]

        if self.transform:
            image = self.transform(pil_image)

        return image, label


def count_dataset(dataset):
    dataset_dict = {i: 0 for i in range(62)}
    for image, label in dataset:
        dataset_dict[label] += 1
    return dataset_dict


def split_train_data(train_data, val_size=0.1):

    random.seed(42)

    train_split = []
    val_split = []

    for user_data in train_data:
        data_length = len(user_data['x'])
        val_length = int(data_length * val_size)

        # 随机抽取验证集数据索引
        val_indices = random.sample(range(data_length), val_length)
        train_indices = list(set(range(data_length)) - set(val_indices))

        # 划分训练集和验证集
        user_train_data = {'x': [user_data['x'][i] for i in train_indices],
                           'y': [user_data['y'][i] for i in train_indices]}
        user_val_data = {'x': [user_data['x'][i] for i in val_indices], 'y': [user_data['y'][i] for i in val_indices]}

        train_split.append(user_train_data)
        val_split.append(user_val_data)

    return train_split, val_split


def load_all_data(file_pattern, num_files=2, sample_fragment_spatio='0', keep=64, train_spatio='9'):
    # 定义训练集和测试集文件路径
    train_files = [
        file_pattern + '/train/all_data_{}_niid_{}_keep_{}_train_{}.json'.format(i, sample_fragment_spatio, keep,
                                                                                 train_spatio) for i in
        range(num_files)]
    test_files = [
        file_pattern + '/test/all_data_{}_niid_{}_keep_{}_test_{}.json'.format(i, sample_fragment_spatio, keep,
                                                                               train_spatio) for i in range(num_files)]
    # print("Train files:", train_files)
    # print("Test files:", test_files)

    train_data = []
    test_data = []
    num_users = 0
    for train_file, test_file in zip(train_files, test_files):
        with open(train_file, 'r') as f:
            train_json = json.load(f)
        with open(test_file, 'r') as f:
            test_json = json.load(f)

        num_users += len(train_json['users'])

        # 将用户数据添加到二维列表中
        for user in train_json['users']:
            train_data.append(train_json['user_data'][user])

        for user in test_json['users']:
            test_data.append(test_json['user_data'][user])

    return train_data, test_data, num_users


def iid_generate(temp_set, num=10):
    x = np.array(temp_set['x'])
    y = np.array(temp_set['y'])
    required_samples_per_class = num  # 例如每个类别需要10个样本
    unique_labels = np.unique(y)
    label_to_indices = {label: np.where(y == label)[0] for label in unique_labels}
    iid_dataset = {'x': [], 'y': []}
    for label in unique_labels:
        indices = label_to_indices[label]
        num_samples = len(indices)
        if num_samples > required_samples_per_class:  # 如果样本数超过所需数量，随机选择所需数量的样本
            selected_indices = np.random.choice(indices, required_samples_per_class, replace=False)
        elif num_samples < required_samples_per_class:  # 如果样本数少于所需数量，进行重采样直到达到所需数量
            selected_indices = np.random.choice(indices, required_samples_per_class, replace=True)
        else:
            selected_indices = indices
        iid_dataset['x'].extend([x[i] for i in selected_indices])
        iid_dataset['y'].extend([y[i] for i in selected_indices])
    return FemnistDataset(iid_dataset, transform_train)


def split_and_return_dataset(path, num_files, num_clients, users_per_client, users_per_server,
                             test_data_total_len=10000, iid=False):
    # train_data's type is list, index represents user.
    train_data, test_data, num_users = load_all_data(path, num_files=num_files, sample_fragment_spatio='0', keep=0,
                                                     train_spatio='9')
    print("the number of users:", num_users)
    train_data, val_data = split_train_data(train_data, val_size=0.1)
    dataset = FemnistDataset
    # server
    server_set = {'x': [], 'y': []}
    test_set = {'x': [], 'y': []}
    all_test_set = {'x': [], 'y': []}
    val_set = {'x': [], 'y': []}
    # clients
    train_subset = []
    val_subset = []
    # user_index = np.random.choice(num_users, num_clients * users_per_client + users_per_server, replace=False)
    user_index = list(range(num_users))
    # for u in user_index:
    # client set
    for i in range(num_clients):
        client_train = {'x': [], 'y': []}
        client_val = {'x': [], 'y': []}
        for user in user_index[users_per_client * i:users_per_client * (i+1)]:
            client_train['x'].extend(train_data[user]['x'])
            client_train['y'].extend(train_data[user]['y'])
            client_val['x'].extend(val_data[user]['x'])
            client_val['y'].extend(val_data[user]['y'])
            val_set['x'].extend(val_data[user]['x'])
            val_set['y'].extend(val_data[user]['y'])
            test_set['x'].extend(test_data[user]['x'])
            test_set['y'].extend(test_data[user]['y'])
        train_subset.append(dataset(client_train, transform_train))
        val_subset.append(dataset(client_val))
    # server set
    if iid:
        for user in user_index[-51:-2]:
            server_set['x'].extend(train_data[user]['x'])
            server_set['y'].extend(train_data[user]['y'])
        if users_per_server == 10:
            server_set = iid_generate(server_set, 20)
        elif users_per_server == 20:
            server_set = iid_generate(server_set, 45)
        else:
            server_set = iid_generate(server_set, 65)
    else:
        for user in user_index[-(users_per_server + 1):-2]:
            server_set['x'].extend(train_data[user]['x'])
            server_set['y'].extend(train_data[user]['y'])
        server_set = dataset(server_set, transform_train)

    # for user in user_index[-51:-2]:
    #     val_set['x'].extend(val_data[user]['x'])
    #     val_set['y'].extend(val_data[user]['y'])
    # test set
    # user_count = 0
    # current_size = 0
    # for user in user_index[::2]:  # 间隔取
    #     test_set['x'].extend(test_data[user]['x'])
    #     test_set['y'].extend(test_data[user]['y'])
    #     user_count += 1
    #     current_size += len(test_data[user]['x'])
    #     if current_size >= test_data_total_len:
    #         break
    test_set = dataset(test_set)
    val_set = dataset(val_set)
    # fedavg set: last one user in user_index
    temp_set = {'x': [], 'y': []}
    temp_set['x'].extend(train_data[user_index[-1]]['x'])
    temp_set['y'].extend(train_data[user_index[-1]]['y'])
    temp_set['x'].extend(test_data[user_index[-1]]['x'])
    temp_set['y'].extend(test_data[user_index[-1]]['y'])
    temp_set['x'].extend(val_data[user_index[-1]]['x'])
    temp_set['y'].extend(val_data[user_index[-1]]['y'])
    # x = np.array(temp_set['x'])
    # y = np.array(temp_set['y'])
    # required_samples_per_class = 10  # 例如每个类别需要10个样本
    # unique_labels = np.unique(y)
    # label_to_indices = {label: np.where(y == label)[0] for label in unique_labels}
    # fedavg_set = {'x': [], 'y': []}
    # for label in unique_labels:
    #     indices = label_to_indices[label]
    #     num_samples = len(indices)
    #     if num_samples > required_samples_per_class:  # 如果样本数超过所需数量，随机选择所需数量的样本
    #         selected_indices = np.random.choice(indices, required_samples_per_class, replace=False)
    #     elif num_samples < required_samples_per_class:  # 如果样本数少于所需数量，进行重采样直到达到所需数量
    #         selected_indices = np.random.choice(indices, required_samples_per_class, replace=True)
    #     else:
    #         selected_indices = indices
    #     fedavg_set['x'].extend([x[i] for i in selected_indices])
    #     fedavg_set['y'].extend([y[i] for i in selected_indices])
    fedavg_set = dataset(temp_set)
    return train_subset, val_subset, server_set, test_set, val_set, fedavg_set


def plot_client_data_distribution(client1_data, client2_data, client1_id, client2_id):
    client1_dist = count_dataset(client1_data)
    client2_dist = count_dataset(client2_data)

    labels = list(client1_dist.keys())
    client1_counts = list(client1_dist.values())
    client2_counts = list(client2_dist.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, client1_counts, width, label=f'Client {client1_id}')
    rects2 = ax.bar(x + width / 2, client2_counts, width, label=f'Client {client2_id}')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_title('Data distribution of two clients')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    num_clients = 100
    users_per_client = 30
    users_per_server = 10
    path = '../leaf/data/femnist/data'
    train_subset, test_subset, server_set, test_set, validation_set, fedavg_set =\
    split_and_return_dataset(
        path=path,
        num_files=35,
        users_per_client=users_per_client,
        users_per_server=users_per_server,
        num_clients=num_clients
    )

    # 保存到data下femnist下
    for i in range(num_clients):
        torch.save(train_subset[i], f'../data/femnist/client{i}.pt')
        torch.save(test_subset[i], f'../data/femnist/test_client{i}.pt')
    torch.save(server_set, '../data/femnist/server.pt')
    torch.save(test_set, '../data/femnist/test.pt')
    torch.save(validation_set, '../data/femnist/val.pt')
    torch.save(fedavg_set, '../data/femnist/fedavg.pt')
        

    # Plot Distribution
    # client1_id = random.randint(0, num_clients - 1)
    # client2_id = random.randint(0, num_clients - 1)
    # while client2_id == client1_id:
    #     client2_id = random.randint(0, num_clients - 1)
    # plot_client_data_distribution(train_subset[client1_id], train_subset[client2_id], client1_id + 1, client2_id + 1)
