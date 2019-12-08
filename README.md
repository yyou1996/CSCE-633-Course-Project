# CSCE-633-Course-Project

1.

Layer-wise GCN: cd ./L2O-LWGCN/

python -m l2o_lwgcn.main --dataset cora --config-file cora.yaml --layer-num 2 --epoch-num 80 80

python -m l2o_lwgcn.main --dataset pubmed --config-file pubmed.yaml --layer-num 2 --epoch-num 80 80

ppi data: https://pan.baidu.com/s/1l0cvtk1a-Gt1S0JoEhnASw

python -m l2o_lwgcn.main_ppi --dataset ppi --config-file ppi.yaml --layer-num 3 --epoch-num 400 400 400

reddit data: https://pan.baidu.com/s/1kFpyTrpwmO5MXjhg3xcpow

python -m l2o_lwgcn.main --dataset reddit --config-file reddit.yaml --layer-num 3 --epoch-num 80 80 80

2.

Layer-wise GCN with learning to optimize controller: cd ./L2O-LWGCN/

python -m l2o_lwgcn.main_l2o --dataset cora --config-file cora.yaml --layer-num 2

python -m l2o_lwgcn.main_l2o --dataset pubmed --config-file pubmed.yaml --layer-num 2

python -m l2o_lwgcn.main_l2o_ppi --dataset ppi --config-file ppi.yaml --layer-num 3 --controller-len 30

python -m l2o_lwgcn.main_l2o --dataset reddit --config-file reddit.yaml --layer-num 3 --controller-len 30

3.

For SOTA:

graphsage forked from https://github.com/williamleif/graphsage-simple.git with slight modification

fastgcn downloaded from https://github.com/matenure/FastGCN.git

vrgcn downloaded from https://github.com/thu-ml/stochastic_gcn.git
