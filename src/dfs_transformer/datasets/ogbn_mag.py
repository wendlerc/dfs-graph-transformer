import pickle
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import glob
import torch
import tqdm
from .utils import get_n_files
from ml_collections import ConfigDict



class OgbnMag(Dataset):
    def __init__(self, path, pattern = "data_split%d.pkl1s.pkl", max_nodes=np.inf, max_edges=np.inf, indices = None,
                 require_min_dfs_code = False):
        self.path = path
        self.pattern = pattern
        self.data = []
        self.path = path
        self.n_splits = get_n_files(path)
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.indices = indices
        self.require_min_dfs_code = require_min_dfs_code
        self.prepare()
        
        
    def prepare(self):
        d_all = {}
        for i in tqdm.tqdm(range(self.n_splits)):
            dname = glob.glob(self.path+"/%d/"%(i+1) + self.pattern%i)[0]
            with open(dname, 'rb') as f:
                d = pickle.load(f)
                for key, val in d.items():
                    d_all[key] = ConfigDict(val)
        
        for idx, d in tqdm.tqdm(d_all.items()):
            if self.indices is None or idx in self.indices:
                if self.require_min_dfs_code:
                    if "min_dfs_code" in d and d.min_dfs_code is not None and len(d.min_dfs_code) > 1:
                        if len(d.node_labels) > self.max_nodes:
                            continue
                        if len(d.edge_labels) > 2*self.max_edges:
                            continue
                                        
                        data_ = Data(idx=idx,
                                     edge_index = d.edge_index,
                                     node_labels = torch.tensor(d.node_labels),
                                     edge_labels = torch.tensor(d.edge_labels),
                                     graph_features = torch.tensor(d.graph_features),
                                     min_dfs_code = torch.tensor(d.min_dfs_code),
                                     min_dfs_index  = torch.tensor(d.dfs_index, dtype=torch.long),
                                     y = d.y)
                        self.data += [data_]   
                
                else:
                    data_ = Data(idx=idx,
                                edge_index = d.edge_index,
                                node_labels = torch.tensor(d.node_labels),
                                edge_labels = torch.tensor(d.edge_labels),
                                graph_features = torch.tensor(d.graph_features),
                                y = d.y)
                    self.data += [data_]   
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]