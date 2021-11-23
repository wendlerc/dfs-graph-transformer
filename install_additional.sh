CUDA=cu110
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+$CUDA.html --upgrade
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+$CUDA.html --upgrade
pip install torch-geometric --upgrade
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+$CUDA.html --upgrade
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+$CUDA.html --upgrade
pip install dgl-$CUDA -f https://data.dgl.ai/wheels/repo.html --upgrade
