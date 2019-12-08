import torch
import torch.utils.data as d

import json
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler
import random
import networkx as nx

from utils import *
from train_batch_multiRank_inductive_reddit_Mixlayers_sampleA import *


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


class feeder_sample(d.Dataset):

    def __init__(self, feat, label, Adj_hat, Adj_eye, train, total_round, sample_node_num):
        self.feat = feat
        self.label = label
        self.Adj_hat = Adj_hat
        self.Adj_eye = Adj_eye
        self.train = train
        self.total_round = total_round
        self.sample_node_num = sample_node_num

    def __len__(self):
        return self.total_round

    def __getitem__(self, index):

        train_sample = random.sample(list(self.train), self.sample_node_num)
        data = self.feat[train_sample]
        label = self.label[train_sample]

        return data, label, train_sample, index


# dataset loader should return:
#   1. feature
#   2. label
#   3. Adj_hat matrix
#   4. dataset split: [train, val, test]


def cora_loader():

    dataset_str = 'cora'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    feat_data = features.todense()
    Adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) + sps.eye(2708).tocsr()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.where(labels)[1].astype(np.int64)
    # assert False

    dataset_split = {}
    dataset_split['test'] = np.array(test_idx_range.tolist())
    dataset_split['train'] = np.array(range(1208))
    dataset_split['val'] = np.array(range(1208, 1708))

    return feat_data, labels, Adj, dataset_split


'''
def cora_loader():

    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty(num_nodes, dtype=np.int64)
    node_map = {}
    label_map = {}
    with open('./dataset/cora/cora.content') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    Adj = sps.eye(num_nodes).tolil()
    with open('./dataset/cora/cora.cites') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            Adj[paper1, paper2] = 1
            Adj[paper2, paper1] = 1
    Adj = Adj.tocsr()

    dataset_split = {}
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    dataset_split['test'] = test
    val = rand_indices[1000:1500]
    dataset_split['val'] = val
    train = rand_indices[1500:]
    dataset_split['train'] = train

    return feat_data, labels, Adj, dataset_split
'''


def pubmed_loader():

    dataset_str = 'pubmed'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    feat_data = features.todense()
    Adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) + sps.eye(19717).tocsr()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.where(labels)[1].astype(np.int64)
    # assert False

    dataset_split = {}
    dataset_split['test'] = np.array(test_idx_range.tolist())
    dataset_split['train'] = np.array(range(18217))
    dataset_split['val'] = np.array(range(18217, 18717))

    return feat_data, labels, Adj, dataset_split


'''
def pubmed_loader():

    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty(num_nodes, dtype=np.int64)
    node_map = {}
    with open('./dataset/pubmed-data/Pubmed-Diabetes.NODE.paper.tab') as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])

    Adj = sps.eye(num_nodes).tolil()
    with open('./dataset/pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab') as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            Adj[paper1, paper2] = 1
            Adj[paper2, paper1] = 1
    Adj = Adj.tocsr()

    dataset_split = {}
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    dataset_split['test'] = test
    val = rand_indices[1000:1500]
    dataset_split['val'] = val
    train = rand_indices[1500:]
    dataset_split['train'] = train

    return feat_data, labels, Adj, dataset_split
'''

def ppi_loader():

    node_num = 56944
    feat_num = 50
    class_num = 121

    dataset_dir = '/home/sjyjya/project/dataset/ppi/ppi'
    id_map_file = dataset_dir + '/ppi-id_map.json'
    node_file = dataset_dir + '/ppi-G.json'
    A_file = dataset_dir + '/Adj_hat.npz'
    feat_file = dataset_dir + '/ppi-feats.npy'
    label_file = dataset_dir + '/label.npy'

    # load feature
    feat = np.load(feat_file)
    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    # load id map
    with open(id_map_file, 'r') as f:
        id_map = json.load(f)

    # load label
    label = np.load(label_file)
    label = label.astype(np.float32)

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(node_file, 'r') as f:
        data_node = json.load(f)['nodes']
    dataset_split = {'train': [], 'val': [], 'test': []}
    for dn in data_node:
        if dn['val']:
            dataset_split['val'].append(dn['id'])
        elif dn['test']:
            dataset_split['test'].append(dn['id'])
        else:
            dataset_split['train'].append(dn['id'])

    return feat, label, Adj_hat, dataset_split


def reddit_loader():

    node_num = 232965
    feat_num = 100
    class_num = 41

    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("./")
    Adj_hat = adj + adj.T

    # load feature
    features = sps.lil_matrix(features)
    normADJ = nontuple_preprocess_adj(adj)
    feat = normADJ.dot(features).todense()

    dataset_dir = '/home/sjyjya/project/L2O_LWGCN/dataset/reddit/reddit'
    id_map_file = dataset_dir + '/reddit-id_map.json'
    node_file = dataset_dir + '/reddit-G.json'
    A_file = dataset_dir + '/Adj_hat.npz'
    feat_file = dataset_dir + '/reddit-feats.npy'
    label_file = dataset_dir + '/reddit-class_map.json'

    # load id map
    with open(id_map_file, 'r') as f:
        id_map = json.load(f)

    # load label
    with open(label_file, 'r') as f:
        data_label = json.load(f)
    label = np.ones(node_num, dtype=np.int64) * -1
    for dl in data_label.items():
        label[id_map[dl[0]]] = dl[1]

    # load dataset split: [train, val, test]
    dataset_split = {}
    dataset_split['train'] = train_index
    dataset_split['val'] = val_index
    dataset_split['test'] = test_index

    return feat, label, Adj_hat, dataset_split


def amazon_670k_loader():

    node_num = 643474
    feat_num = 100
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

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(dataset_split_file, 'r') as f:
        dataset_split = json.load(f)

    return feat, label, Adj_hat, dataset_split


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

    # load Adj hat matrix
    Adj_hat = sps.load_npz(A_file)

    # load dataset split: [train, val, test]
    with open(dataset_split_file, 'r') as f:
        dataset_split = json.load(f)

    return feat, label, Adj_hat, dataset_split

