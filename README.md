# TODO: 
bold means important, stuff that people are working on contains names

### Show that the new representation can solve an actual problem
- [ ] show that there are advantages of using DFS codes over smiles 
- [ ] show that DFS codes can capture stereochemical properties 
- [ ] show that DFS codes work well on other types of graphs (ogb-mag seems to be a good fit)

### Evaluate the following baselines on our moleculenet splits
- [x] DMPNN
- [ ] Hugo **GROVER**, [git](https://github.com/tencent-ailab/grover) [paper](https://arxiv.org/abs/2007.02835)
- [ ] Hugo **ChemBERTa**, [git](https://github.com/seyonechithrananda/bert-loves-chemistry) [paper](https://arxiv.org/abs/2010.09885) [huggingface](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- [x] Hugo SMILES transformer, [git](https://github.com/DSPsleeporg/smiles-transformer) [paper](https://arxiv.org/abs/1911.04738)
- [ ] **SchNet** or a successor, [git](https://github.com/atomistic-machine-learning/schnetpack)
- [ ] N-GRAM, see Table 1 in [GROVER paper](https://arxiv.org/abs/2007.02835)

#### general 
- [ ] **rewrite smiles2graph such that also the positions of the atoms are computed as features**
- [ ] implement a better metric than accuracy to assess the quality of the fit of the pretraining 
#### improve the results
- [ ] Chris **implement a version that utilizes self-loops**
- [ ] make larger model converge
- [x] check whether having multiple cls tokens is helpful
- [ ] tweak the way the gradients are propagated through the cls token 
- [ ] come up with a way to make the rnd2min pretraining converge to 0 loss
- [x] add a term to the pretraining objective that depends on the features  
- [ ] a transformer decoder that has the hidden states of a GNN as memory
#### random DFS code to minimal DFS code pretraining 
- [x] Chris **write cluster ready pretraining script**
- [ ] Chris **write cluster ready finetuning script** 
- [ ] **find good hyperparameters**  
- [ ] investigate the behavior when the pretraining dataset size is increased
#### BERT pretraining 
- [x] **write cluster ready pretraining script**
- [ ] Chris **write cluster ready finetuning script** 
- [ ] find good hyperparameters (actually I think this is semi-done, our selfattn converges pretty well)
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

Examples:

Overwrite run parameters example:

```bash
python exp/pretrain/selfattn/pubchem.py --name bert-10K --wandb_mode offline --overwrite '{"training" : {"n_epochs" : 1}, "data" : {"n_iter_per_split" : 2}}'
```
