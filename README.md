# CSCE-633-Course-Project

Layer-wise GCN: cd L2O-LWGCN

python -m l2o_lwgcn.main --dataset cora --config-file cora.yaml --layer-num 2 --epoch-num 80 80

python -m l2o_lwgcn.main --dataset pubmed --config-file pubmed.yaml --layer-num 2 --epoch-num 80 80

Layer-wise GCN with learning to optimize controller: cd L2O-LWGCN

python -m l2o_lwgcn.main_l2o --dataset cora --config-file cora.yaml --layer-num 2

python -m l2o_lwgcn.main_l2o --dataset pubmed --config-file pubmed.yaml --layer-num 2

For SOTA:

graphsage forked from https://github.com/williamleif/graphsage-simple.git with slight modification

fastgcn downloaded from https://github.com/matenure/FastGCN.git

vrgcn downloaded from https://github.com/thu-ml/stochastic_gcn.git
