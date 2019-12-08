# CSCE-633-Course-Project

Layer-wise GCN:
python -m l2o_lwgcn.main --dataset cora --config-file cora.yaml --layer-num 2 --epoch-num 80 80
python -m l2o_lwgcn.main --dataset pubmed --config-file pubmed.yaml --layer-num 2 --epoch-num 80 80
Layer-wise GCN with learning to optimize controller:
python -m l2o_lwgcn.main_l2o --dataset cora --config-file cora.yaml --layer-num 2
python -m l2o_lwgcn.main_l2o --dataset pubmed --config-file pubmed.yaml --layer-num 2
