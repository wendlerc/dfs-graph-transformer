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
├── config <-- main directory
├── datasets
├── preprocessed
├── exp
└── src

```

# Installation

```bash
poetry install
poetry shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html

git clone git@github.com:ElizaWszola/dfs-code-representation.git
cd dfs-code-representation
git checkout vertexids
pip install . 
```

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

For the pretraining to work, make sure to download ... and update the config files in ./config/selfattn/data accordingly. 

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

For the evaluation to work, make sure to download ... and update ./config/selfattn/finetune_moleculenet.yaml accordingly.

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

# Other graphs



