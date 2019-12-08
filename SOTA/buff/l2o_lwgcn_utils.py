import torch
import torch.utils.data as d

import json
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler
import random


class feeder(d.Dataset):

    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        data = self.feat[index]
        label = self.label[index]
        return data, label


# dataset loader should return:
#   1. feature
#   2. label
#   3. Adj_hat matrix
#   4. dataset split: [train, val, test]


def amazon_670k_loader():

    node_num = 643474
    edge_num = -1
    class_num = 32

    dataset_dir = '/home/sjyjya/project/dataset/amazon_670k/amazon_670k'
    A_file = dataset_dir + '/Adj_hat.npz'
    feat_file = dataset_dir + '/feat_truncatedSVD.npy'
    label_file = dataset_dir + '/label.npy'
    dataset_split_file = dataset_dir + '/dataset_split.json'

    # load feature
    feat = np.load(feat_file)
    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    # load label
    label = np.load(label_file)
    label = label.astype(np.int64)
    y = np.zeros((node_num, class_num))
    y[range(node_num), label] = 1

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(dataset_split_file, 'r') as f:
        dataset_split = json.load(f)

    train_mask = np.zeros(node_num)
    train_mask[dataset_split['train']] = 1
    val_mask = np.zeros(node_num)
    test_mask = np.zeros(node_num)
    test_mask[dataset_split['test']] = 1

    # return feat, label, Adj_hat, dataset_split
    return Adj_hat, sps.lil_matrix(feat), y, y, y, train_mask, val_mask, test_mask


def amazon_3m_loader():

    node_num = 2460406
    edge_num = -1
    class_num = 38

    dataset_dir = '/home/sjyjya/project/dataset/amazon_3m/amazon_3m'
    A_file = dataset_dir + '/Adj_hat.npz'
    feat_file = dataset_dir + '/feat_truncatedSVD.npy'
    label_file = dataset_dir + '/label.npy'
    dataset_split_file = dataset_dir + '/dataset_split.json'

    # load feature
    feat = np.load(feat_file)
    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    # load label
    label = np.load(label_file)
    label = label.astype(np.int64)
    y = np.zeros((node_num, class_num))
    y[range(node_num), label] = 1

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(dataset_split_file, 'r') as f:
        dataset_split = json.load(f)

    train_mask = np.zeros(node_num)
    train_mask[dataset_split['train']] = 1
    val_mask = np.zeros(node_num)
    test_mask = np.zeros(node_num)
    test_mask[dataset_split['test']] = 1

    # return feat, label, Adj_hat, dataset_split
    return Adj_hat, sps.lil_matrix(feat), y, y, y, train_mask, val_mask, test_mask

