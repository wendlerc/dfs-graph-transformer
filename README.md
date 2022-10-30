# Graph transformers based on depth first search codes

We represent graphs as sequences of edges (so called DFS codes) and process them using transformers.
DFS codes correspond to depth first search (DFS) traversals of graphs. In particular, they record edges 
in the order in which they are encoundered using DFS. By defining a total order on the space of all such 
sequences it is possible to associate graphs with minimal DFS codes which are unique up to graph isomorphy.
That is, isomorphic graphs have the same minimal DFS codes. 

For very symmetrical molecules the computation of the minimal DFS codes can become extremely slow. 
In our preprocessing scripts we omitted those.

# Project structure

### Code structure
```
.
├── config <-- yaml files containing configs for the scripts in ./exp
├── datasets <-- molecular and graph datasets
├── exp <-- scripts for the pretraining and evaluation
├── notebooks <-- some jupyter notebooks
├── preprocessed <-- precomputed DFS code representations (you need to download them first see preprocessed/README.md)
├── results <-- scripts for the pretraining and evaluation
├── scripts <-- scripts for preprocessing data and submitting jobs to the cluster (you hopefully won't need that)
└── src <-- nn architectures, dataset classes and trainers

```

# Installation

```bash
poetry install
poetry shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html

git clone git@github.com:ElizaWszola/dfs-code-representation.git
cd dfs-code-representation
pip install . 
```

Also go to preprocessed README.md and download the preprocessed DFS codes and update the configs in ./config accordingly.

# Baselines
```bash
pip install git+https://github.com/bp-kelley/descriptastorus
pip install chemprop
```

# Example usage

All scripts are parametrized by config files and command line arguments. Please consult the respective files for more details. Scripts for running
experiments are in ./exp and configuration files are in ./config. 

# Molecular data

## Pretraining

For the pretraining to work, make sure to download https://www.icloud.com/iclouddrive/0d7bts2-v_f4d7GvCV03HrV5Q#pubchem and update the config files in ./config/selfattn/data accordingly. 

The parametrization of the training loop in the pretraining script is a bit unconventional. This is because the pretraining dataset with 10 million 
molecules (not provided) does not fit into memory on my machines. I store large datasets by splitting them into several parts of equal size (number of molecules). 
Then in the outer loop that does n_epochs repetitions a subset of the splits is complsed into a torch dataset. The inner loop then performs n_iter_per_split 
passes over this dataset and so on. If all splits fit into memory I recommend using n_epochs=1 and n_iter_per_split=<desired_number_of_epochs>, e.g., 
n_iter_per_split=10. If at least es_period (default=1000) batches have been processed the final checkpoint is uploaded as a wandb artifact for later use.
Please make sure to set --wandb_entity, --wandb_project accordingly. --overwrite is used to overwrite parameters that are set via config files.

```bash
python exp/pretrain/selfattn/pubchem_plus_properties.py --wandb_entity dfstransformer --wandb_project pubchem_pretrain --name bert-10K --yaml_data './config/selfattn/data/pubchem10K.yaml' --overwrite '{"training" : {"n_epochs" : 1}, "data" : {"n_iter_per_split" : 10}}'
```

## Evaluation

For the evaluation to work, make sure to download https://www.icloud.com/iclouddrive/0b5IUU6Yzd4QmU5jt3IrHJQ_Q#mymoleculenet%5Fplus%5Ffeatures and update ./config/selfattn/moleculenet.yaml and ./config/selfattn/finetune_moleculenet.yaml accordingly.

### Pretrained models

Here are some of my pretrained models for molecular data: https://wandb.ai/dfstransformer/pubchem_newencoding 

### Use pretrained features 

The evaluation script is parametrized by the config file ./config/selfattn/moleculenet.yaml.

```bash
python exp/evaluate/selfattn/moleculenet_plus_properties.py --wandb_entity dfstransformer --wandb_project moleculenet_eval --overwrite '{"pretrained_model":"r2r-30"}'
```

### Finetune

The finetuning script is parametrized by the config file ./config/finetune_moleculenet.yaml. Importantly, this file points to the wandb project containing the pretrained models.
The pretrained model is then selected by setting pretrained_model to the name of the run containing the checkpoint artifact. 

```bash
python exp/evaluate/selfattn/finetune_moleculenet.py --wandb_entity dfstransformer --wandb_project moleculenet_finetune --overwrite '{"pretrained_model":"r2r-30"}'

```

# Karateclub

For this dataset to work, download https://www.icloud.com/iclouddrive/09fslwInJA2i6grbE2Dm9e9mw#karateclub and store it in ./datasets.

DFS codes can be also computed for other graphs. Here I considered two karateclub datasets: reddit_threads and twitch_egos. Each of these datasets consists of two files
reddit_edges.json and reddit_target.csv, twitch_edges.json and twitch_target.csv respectively. Paths to these files can be supplied via --graph_file which expects a path 
to the json and --label_file which expects a path to the csv. For performance reasons (the DFS code transformer's attention matrix is quadratic in the number of edges), 
we only consider graphs with less or equal 200 edges --max_edges 200.

## Run torch geometric baselines

Toch geometric models can be supplied via the --model argument. In the file I import torch_geometric.nn as tnn. 
Suitable choices are, e.g., tnn.models.GCN or tnn.models.GIN.

```bash
python exp/evaluate/gnn/karateclub.py --wandb_entity dfstransformer --wandb_project karateclub --model nn.models.GCN
```

For prototyping the --n_samples parameter is useful as it allows to run on a subset of the dataset

```bash
python exp/evaluate/gnn/karateclub.py --wandb_entity dfstransformer --wandb_project karateclub --model nn.models.GIN --n_samples 10000
```

## Run DFS code transformer 

```bash
python exp/evaluate/selfattn/karateclub.py 
```




