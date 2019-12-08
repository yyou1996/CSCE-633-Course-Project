import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask_value = torch.ones(len(row_indices))
        n_start = 0
        n_end = 0
        r_count = 0
        for n in range(len(row_indices)):
            # print(r_count, row_indices[n])
            if row_indices[n] == r_count:
                n_end = n_end + 1
            else:
                mask_value[n_start:n_end] = 1 / (n_end - n_start)
                n_start = n_end
                n_end = n_end + 1
                r_count = r_count + 1

        indices = [row_indices, column_indices]
        # mask[row_indices, column_indices] = 1
        mask = torch.sparse.FloatTensor(torch.tensor(indices), mask_value, (len(samp_neighs),len(unique_nodes)))
        if self.cuda:
            mask = mask.cuda()
        # num_neigh = mask.sum(1, keepdim=True)
        # mask = mask.div(num_neigh)
        nodes = list(nodes)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            node_feats = self.features(torch.LongTensor(nodes).cuda()) ### GCN
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            node_feats = self.features(torch.LongTensor(nodes)) ### GCN
        to_feats = mask.mm(embed_matrix)
        to_feats = node_feats + mask.mm(embed_matrix) ### GCN
        return to_feats
