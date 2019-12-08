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
        self.xent = nn.BCELoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        scores = self.sigmoid(scores)
        return self.xent(scores, labels)

def load_ppi():
    num_nodes = 56944
    num_feats = 50

    feat_data, labels, Adj_hat, dataset_split = utils.ppi_loader()

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

def run_ppi():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 56944
    feat_data, labels, adj_lists, dataset_split = load_ppi()

    features = nn.Embedding(56944, 50)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 50, 512, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 512, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=True)
    enc1.num_samples = 10
    enc2.num_samples = 25
    # enc1.num_samples = None
    # enc2.num_samples = None

    graphsage = SupervisedGraphSage(121, enc2)
    graphsage.cuda()

    test = dataset_split['test']
    train = dataset_split['train']

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.0001)
    times = []
    epoch_num = 400
    for epoch in range(epoch_num):
        for batch in range(int(len(train) / 1024)):
            batch_nodes = train[:1024]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.FloatTensor(labels[np.array(batch_nodes)])).cuda())
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
    val_output[val_output>0] = 1
    val_output[val_output<=0] = 1
    print("Validation F1:", f1_score(labels[test], val_output.cpu().data.numpy(), average="micro"))
    print("Total time:", np.sum(times))
    print("Average batch time:", np.sum(times) / epoch_num)
    print('')


if __name__ == "__main__":
    run_ppi()
