# TODO: 
bold means important, stuff that people are working on contains names

### Show that the new representation can solve an actual problem
- [ ] show that DFS codes work well on other types of graphs 
- [x] try ogb-mag node prediction task 

### Evaluate the following baselines on our moleculenet splits
- [x] DMPNN
- [ ] **GROVER**, [git](https://github.com/tencent-ailab/grover) [paper](https://arxiv.org/abs/2007.02835)
- [x] Hugo **ChemBERTa**, [git](https://github.com/seyonechithrananda/bert-loves-chemistry) [paper](https://arxiv.org/abs/2010.09885) [huggingface](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- [x] Hugo SMILES transformer, [git](https://github.com/DSPsleeporg/smiles-transformer) [paper](https://arxiv.org/abs/1911.04738)

#### general 
- [x] implement a better metric than accuracy to assess the quality of the fit of the pretraining 
#### improve the results
- [x] Chris **implement a version that utilizes self-loops**
- [ ] make larger model converge, ADAM beta2=0.95
- [x] check whether having multiple cls tokens is helpful
- [x] tweak the way the gradients are propagated through the cls token 
- [x] add a term to the pretraining objective that depends on the features  
- [ ] a transformer decoder that has the hidden states of a GNN as memory
#### pretraining 
- [x] Chris **write cluster ready pretraining script**
- [x] Chris **write cluster ready finetuning script** 
- [ ] **find good hyperparameters**  
- [ ] investigate the behavior when the pretraining dataset size is increased


# Installation:

```bash
poetry install
poetry shell
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-geometric
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html

pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html

git clone git@gitlab.inf.ethz.ch:ewszola/dfs-code-representation.git
cd dfs-code-representation
git checkout vertexids
pip install . 
```

Baseline:

```bash
pip install git+https://github.com/bp-kelley/descriptastorus

pip install chemprop
```

Cluster 

This needs to be active.
```bash
env2lmod
module load gcc/8.2.0 python_gpu/3.9.9
module load eth_proxy
```
This python comes with support for torch 1.10.0+cu113 and torch-geometric 2.0.3, for some reason the poetry always installs 1.9. (maybe fix this...)
https://scicomp.ethz.ch/wiki/Python_on_Euler#python_gpu.2F3.8.5_2
```bash
poetry install
poetry shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html

git clone git@gitlab.inf.ethz.ch:ewszola/dfs-code-representation.git
cd dfs-code-representation
git checkout vertexids
pip install . 
```

Examples:

Overwrite run parameters example:

```bash
python exp/pretrain/selfattn/pubchem.py --name bert-10K --wandb_mode offline --overwrite '{"training" : {"n_epochs" : 1}, "data" : {"n_iter_per_split" : 2}}'
```
