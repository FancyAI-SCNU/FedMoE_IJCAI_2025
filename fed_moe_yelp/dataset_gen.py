import random

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

data_files = {
    "train": "../data/yelp/train-00000-of-00001.parquet",
    "test": "../data/yelp/test-00000-of-00001.parquet"
}

# 加载数据集
dataset = load_dataset("parquet", data_files=data_files)

df = pd.DataFrame(dataset['train'])


# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# test = df.iloc[:10000]
# test.to_csv('../data/yelp/fedavg.csv', index=False)
# fedavg = df.iloc[10000:10200]
# fedavg.to_csv('../data/yelp/fedavg.csv', index=False)

#
# 分割数据集
def split_data(df, num_clients=10, client_size=5000, server_size=5000, val_size=500):
    # 按类别分割数据集
    grouped = df.groupby('label')

    # 创建server数据集和验证集
    server_df = pd.concat([group.sample(1000) for _, group in grouped])
    remaining_df = df.drop(server_df.index.intersection(df.index))

    val_df = pd.concat([group.sample(100) for _, group in grouped])
    remaining_df = remaining_df.drop(val_df.index.intersection(remaining_df.index))

    client_dfs = []
    for i in range(num_clients):
        # 随机选择一个类，使其占比最大
        p = random.uniform(0.55, 0.65)
        major_class = i % 5
        major_class_df = remaining_df[remaining_df['label'] == major_class].sample(int(p * client_size))
        remaining_df = remaining_df.drop(major_class_df.index.intersection(remaining_df.index))

        other_class_df = remaining_df.sample(client_size - len(major_class_df))
        remaining_df = remaining_df.drop(other_class_df.index.intersection(remaining_df.index))

        client_df = pd.concat([major_class_df, other_class_df]).sample(frac=1).reset_index(drop=True)
        client_dfs.append(client_df)

    return client_dfs, server_df, val_df


client_dfs, server_df, val_df = split_data(df)

# 保存数据集到文件
# for i, client_df in enumerate(client_dfs):
#     client_df.to_csv(f'../data/yelp/client_{i + 1}_train.csv', index=False)
server_df.to_csv('../data/yelp/server_train.csv', index=False)
# val_df.to_csv('../data/yelp/validation.csv', index=False)
#
# client_dfs = []
# for i in range(1, 11):  # 假设有10个client数据集
#     client_df = pd.read_csv(f'../data/yelp/client_{i}_train.csv')
#     client_dfs.append(client_df)
#
# server_df = pd.read_csv('../data/yelp/server_train.csv')
# val_df = pd.read_csv('../data/yelp/validation.csv')
#
# # 绘制柱状图表示每个client的数据分布
# client_labels = [f'Client {i + 1}' for i in range(len(client_dfs))]
# label_counts = [client_df['label'].value_counts() for client_df in client_dfs]
# label_counts_df = pd.DataFrame(label_counts).fillna(0).astype(int)
#
# label_counts_df.plot(kind='bar', stacked=True, colormap='tab20', figsize=(10, 7))
# plt.xlabel('Clients')
# plt.ylabel('Number of Samples')
# plt.title('Data Distribution Across Clients')
# plt.legend(title='Class Label')
# plt.show()
