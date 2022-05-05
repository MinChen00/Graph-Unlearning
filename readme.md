## Introduction

This code is the implementation of  graph unlearning.

#### Code Strcuture

```
.
├── config.py
├── exp
├── lib_aggregator
├── lib_dataset
├── lib_gnn_model
├── lib_graph_partition
├── lib_node_embedding
├── lib_utils
├── main.py
├── parameter_parser.py
└── readme.md
```

#### Environment prepare

```bash
conda create --name graph_unlearning python=3.6.10
conda activate graph_unlearning 
pip install sklearn ogb infomap seaborn munkres gensim fastdtw leidenalg cvxpy pymetis mysqlclient pymetis MulticoreTSNE cupy-cuda111 tensorflow
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
TORCH="1.9.0"
CUDA="cu111"
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

#### GraphEraser Framework

###### Graph Partition

See more parameters settings in parameter_parser.py at ***##graph partition parameters##***.

```bash
$ --partition true --partition_method lpa --is_constrained true

$ --partition true --partition_method sage_km --is_constrained true
```

###### Aggregation

See more parameters settings in parameter_parser.py at ***##training parameters##***.

```bash
Use '--aggregator' choose the desired aggregation method, choose from ['mean', 'majority', 'optimal'].

```

###### Unlearning

See more parameters settings in parameter_parser.py at ***##unlearning parameters##***.

```bash
Use '--repartition' to decide whether unlearning the graph partition.

Use '--unlearning_request' to choose the unlearning request distributions from ['random', 'adaptive', 'dependant', 'top1', 'last5'].
```
