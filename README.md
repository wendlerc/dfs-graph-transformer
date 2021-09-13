# TODO bold means important: 
### Evaluate the following baselines on our moleculenet splits
- [x] DMPNN
- [ ] Hugo **GROVER**, [git](https://github.com/tencent-ailab/grover) [paper](https://arxiv.org/abs/2007.02835)
- [ ] Hugo **ChemBERTa**, [git](https://github.com/seyonechithrananda/bert-loves-chemistry) [paper](https://arxiv.org/abs/2010.09885) [huggingface](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- [ ] Hugo SMILES transformer, [git](https://github.com/DSPsleeporg/smiles-transformer) [paper](https://arxiv.org/abs/1911.04738)
- [ ] **SchNet** or a successor, [git](https://github.com/atomistic-machine-learning/schnetpack)
- [ ] N-GRAM, see Table 1 in [GROVER paper](https://arxiv.org/abs/2007.02835)

#### general 
- [ ] **rewrite smiles2graph such that also the positions of the atoms are computed as features**
- [ ] implement a better metric than accuracy to assess the quality of the fit of the pretraining 
#### improve the results
- [ ] **implement a version that utilizes self-loops**
- [ ] make larger model converge
- [ ] check whether having multiple cls tokens is helpful
- [ ] tweak the way the gradients are propagated through the cls token 
- [ ] come up with a way to make the rnd2min pretraining converge to 0 loss
- [ ] add a term to the pretraining objective that depends on the features  
- [ ] a transformer decoder that has the hidden states of a GNN as memory
#### random DFS code to minimal DFS code pretraining 
- [ ] **write cluster ready pretraining script**
- [ ] **write cluster ready finetuning script** 
- [ ] **find good hyperparameters**  
- [ ] investigate the behavior when the pretraining dataset size is increased
#### BERT pretraining 
- [x] **write cluster ready pretraining script**
- [ ] **write cluster ready finetuning script** 
- [ ] find good hyperparameters  
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

git pull https://gitlab.inf.ethz.ch/ewszola/dfs-code-representation
cd dfs-code-representation
git checkout vertexids
pip install . 
pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html
```

Baseline:

pip install git+https://github.com/bp-kelley/descriptastorus

pip install chemprop

