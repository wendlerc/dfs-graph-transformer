import pickle
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import glob
import torch
import tqdm
from .utils import get_n_files


class PubChem(Dataset):
    """PubChem dataset of molecules and minimal DFS codes."""
    def __init__(self, path, n_used = None, n_splits = None, max_nodes=np.inf,
                 max_edges=np.inf, useHs=False, addLoops=False, memoryEfficient=False,
                 transform=None, exclude=[]):
        """
        Parameters
        ----------
        path : str, path to the preprocessed minimal dfs codes and features. 
        For the largest dataset I used 64 splits.
        n_used : int, how many splits of the dataset to concurrently load. The default is 8.
        n_splits : int, how many splits there are in total. If it is set to None 
        the number of splits is automatically determined. The default is None.
        max_nodes : The default is np.inf.
        max_edges : The default is np.inf.
        n_mols_per_dataset : int, number of molecules to process per split. The default is np.inf.

        """
        self.path = path
        self.data = []
        self.smiles = []
        self.path = path
        if n_splits is None:
            n_splits = get_n_files(path)
        self.n_splits = n_splits
        if n_used is None:
            self.n_used = self.n_splits
        else:
            self.n_used = n_used
        self.useHs = useHs
        self.addLoops = addLoops
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.exclude = set(exclude)
        self.prepare()
        
        
    def prepare(self):
        codes_all = {}
        d_all = {}
        perm = np.random.permutation(self.n_splits)
        for i in tqdm.tqdm(perm[:self.n_used]):
            dname = glob.glob(self.path+"/%d/min_dfs_codes_split*.pkl"%(i+1))[0]
            didx = int(dname.split("split")[-1][:-4])
            dname2 = self.path+"/%d/data_split%d.pkl"%(i+1, didx)
            with open(dname, 'rb') as f:
                codes = pickle.load(f)
                for key, val in codes.items():
                    if key not in self.exclude:
                        codes_all[key] = val
                    
            with open(dname2, 'rb') as f:
                d_dict = pickle.load(f)
                for key, val in d_dict.items():
                    if key not in self.exclude:
                        d_all[key] = val
        
        for smiles, code in tqdm.tqdm(codes_all.items()):
            if code['min_dfs_code'] is not None and len(code['min_dfs_code']) > 1:
                d = d_all[smiles]
                if len(d['z']) > self.max_nodes:
                    continue
                if len(d['edge_attr']) > 2*self.max_edges:
                    continue
                
                z = torch.tensor(d['z'], dtype=torch.long)
                
                data_ = Data(z=z,
                             edge_attr=torch.tensor(d['edge_attr']),
                             edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
                             min_dfs_code=torch.tensor(code['min_dfs_code']),
                             min_dfs_index=torch.tensor(code['dfs_index'], dtype=torch.long),
                             smiles=smiles,
                             node_features=torch.tensor(d['atom_features'], dtype=torch.float32),
                             edge_features=torch.tensor(d['bond_features'], dtype=torch.float32))
                self.data += [data_]   
                self.smiles += [smiles]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]