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

Examples:

Overwrite run parameters example:

```bash
python exp/pretrain/selfattn/pubchem_bert.py --name bert-10K --wandb_mode offline --overwrite '{"training" : {"n_epochs" : 1}, "data" : {"n_iter_per_split" : 2}}'
```