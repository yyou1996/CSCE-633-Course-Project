import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
import graphsage.l2o_lwgcn_utils as utils

import os

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

'''
def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists
'''

def load_cora():
    num_nodes = 2708
    num_feats = 1433

    feat_data, labels, Adj_hat, dataset_split = utils.cora_loader()

    adj_lists = defaultdict(set)
    A = Adj_hat.tocoo()
    for i, j in zip(A.row, A.col):
        # a = str(a).split(')')[0].split('(')[1].split(', ')
        # i = int(a[0])
        # j = int(a[1])
        adj_lists[i].add(j)
        adj_lists[j].add(i)
    for n in range(num_nodes):
        adj_lists[n].add(n)
    return feat_data, labels, adj_lists, dataset_split


def run_cora(seed):
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = 2708
    feat_data, labels, adj_lists, dataset_split = load_cora()

    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 16, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 16, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 25
    # enc1.num_samples = None
    # enc2.num_samples = None

    graphsage = SupervisedGraphSage(7, enc2)
    graphsage.cuda()
    '''
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    '''

    test = dataset_split['test']
    val = dataset_split['val']
    train = dataset_split['train']

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []

    for epoch in range(100):
        for batch in range(int(len(train) / 256)):
            batch_nodes = train[:256]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])).cuda())
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
        print(epoch + 1, loss.data)

    print('')
    os.system("nvidia-smi")

    val_output = graphsage.forward(test) 
    print("Validation F1:", f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Total time:", np.sum(times))
    print("Average batch time:", np.sum(times) / (epoch + 1))
    print('')

    return f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"), np.sum(times)


def load_pubmed():
    num_nodes = 19717
    num_feats = 500

    feat_data, labels, Adj_hat, dataset_split = utils.pubmed_loader()

    adj_lists = defaultdict(set)
    A = Adj_hat.tocoo()
    for i, j in zip(A.row, A.col):
        # a = str(a).split(')')[0].split('(')[1].split(', ')
        # i = int(a[0])
        # j = int(a[1])
        adj_lists[i].add(j)
        adj_lists[j].add(i)
    for n in range(num_nodes):
        adj_lists[n].add(n)
    return feat_data, labels, adj_lists, dataset_split


'''
def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists
'''


def run_pubmed(seed):
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = 19717
    feat_data, labels, adj_lists, dataset_split = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 16, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 16, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 25
    # enc1.num_samples = None
    # enc2.num_samples = None

    graphsage = SupervisedGraphSage(3, enc2)
    graphsage.cuda()
 
    test = dataset_split['test']
    val = dataset_split['val']
    train = dataset_split['train']

    '''
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    '''

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []

    for epoch in range(200):
        for batch in range(int(len(train) / 1024)):
            batch_nodes = train[:1024]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])).cuda())
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
        print(epoch + 1, loss.data)

    print('')
    os.system("nvidia-smi")

    val_output = graphsage.forward(test) 
    print("Validation F1:", f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Total time:", np.sum(times))
    print("Average batch time:", np.sum(times) / (epoch + 1))

    print('')
    return f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"), np.sum(times)


def load_reddit():
    num_nodes = 232965
    num_feats = 602

    feat_data, labels, Adj_hat, dataset_split = utils.reddit_loader()

    adj_lists = defaultdict(set)
    A = Adj_hat.tocoo()
    for i, j in zip(A.row, A.col):
        # a = str(a).split(')')[0].split('(')[1].split(', ')
        # i = int(a[0])
        # j = int(a[1])
        adj_lists[i].add(j)
        adj_lists[j].add(i)
    for n in range(num_nodes):
        adj_lists[n].add(n)
    return feat_data, labels, adj_lists, dataset_split

def run_reddit(seed):
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = 232965
    feat_data, labels, adj_lists, dataset_split = load_reddit()

    features = nn.Embedding(232965, 602)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 602, 128, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 25
    # enc1.num_samples = None
    # enc2.num_samples = None

    graphsage = SupervisedGraphSage(41, enc2)
    graphsage.cuda()

    test = dataset_split['test']
    train = dataset_split['train']

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.003)
    times = []
    epoch_num = 10
    for epoch in range(epoch_num):
        for batch in range(int(len(train) / 1024)):
            batch_nodes = train[:1024]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])).cuda())
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
        print(epoch + 1, loss.data)

    print('')
    os.system("nvidia-smi")

    agg1.cuda = False
    enc1.cuda = False
    agg2.cuda = False
    enc2.cuda = False

    val_output = graphsage.cpu().forward(test)
    print("Validation F1:", f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Total time:", np.sum(times))
    print("Average batch time:", np.sum(times) / epoch_num)
    print('')

    return f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"), np.sum(times)

def load_amazon_670k():
    num_nodes = 643474
    num_feats = 100

    feat_data, labels, Adj_hat, dataset_split = utils.amazon_670k_loader()

    adj_lists = defaultdict(set)
    for a in Adj_hat:
        a = str(a).split(')')[0].split('(')[1].split(', ')
        i = int(a[0])
        j = int(a[1])
        adj_lists[i].add(j)
        adj_lists[j].add(i)
    for n in range(num_nodes):
        adj_lists[n].add(n)

    return feat_data, labels, adj_lists, dataset_split

def run_amazon_670k():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 643474
    feat_data, labels, adj_lists, dataset_split = load_amazon_670k()

    features = nn.Embedding(643474, 100)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 100, 128, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 25
    # enc1.num_samples = None
    # enc2.num_samples = None


    graphsage = SupervisedGraphSage(32, enc2)
    graphsage.cuda()

    test = dataset_split['test']
    train = dataset_split['train']

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.0001)
    times = []
    epoch_num = 100
    for epoch in range(epoch_num):
        for batch in range(int(len(train) / 1024)):
            batch_nodes = train[:1024]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])).cuda())
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
        print(epoch + 1, loss.data)

    print('')
    os.system("nvidia-smi")

    agg1.cuda = False
    enc1.cuda = False
    agg2.cuda = False
    enc2.cuda = False

    val_output = graphsage.cpu().forward(test)
    print("Validation F1:", f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Total time:", np.sum(times))
    print("Average batch time:", np.sum(times) / epoch_num)
    print('')

def load_amazon_3m():
    num_nodes = 2460406
    num_feats = 100

    feat_data, labels, Adj_hat, dataset_split = utils.amazon_3m_loader()

    adj_lists = defaultdict(set)
    for a in Adj_hat:
        a = str(a).split(')')[0].split('(')[1].split(', ')
        i = int(a[0])
        j = int(a[1])
        adj_lists[i].add(j)
        adj_lists[j].add(i)
    for n in range(num_nodes):
        adj_lists[n].add(n)

    return feat_data, labels, adj_lists, dataset_split

def run_amazon_3m():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2460406
    feat_data, labels, adj_lists, dataset_split = load_amazon_3m()

    features = nn.Embedding(2460406, 100)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 100, 128, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 25
    # enc1.num_samples = None
    # enc2.num_samples = None

    graphsage = SupervisedGraphSage(38, enc2)
    graphsage.cuda()

    test = dataset_split['test']
    train = dataset_split['train']

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.0001)
    times = []
    epoch_num = 100
    for epoch in range(epoch_num):
        for batch in range(int(len(train) / 1024)):
            batch_nodes = train[:1024]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])).cuda())
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
        print(epoch + 1, loss.data)

    print('')
    os.system("nvidia-smi")

    agg1.cuda = False
    enc1.cuda = False
    agg2.cuda = False
    enc2.cuda = False

    val_output = graphsage.cpu().forward(test)
    print("Validation F1:", f1_score(labels[test], val_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Total time:", np.sum(times))
    print("Average batch time:", np.sum(times) / epoch_num)
    print('')

