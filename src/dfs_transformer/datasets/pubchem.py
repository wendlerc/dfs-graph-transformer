import json
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import glob
import torch
import tqdm

class PubChem(Dataset):
    """PubChem dataset of molecules and minimal DFS codes."""
    def __init__(self, path, n_used = 8, n_splits = None, max_nodes=np.inf,
                 max_edges=np.inf, useHs=False, addLoops=False, memoryEfficient=False,
                 transform=None, n_mols_per_dataset=np.inf):
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
        self.path = path
        self.n_used = n_used
        if n_splits = None:
            nums = []
            for name in glob.glob('%s/*'%path):
                try:
                    nums += [int(name.split('/')[-1])]
                except ValueError:
                    continue
            n_splits = max(nums)
        self.n_splits = n_splits
        self.useHs = useHs
        self.addLoops = addLoops
        self.n_mols_per_dataset = n_mols_per_dataset
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.prepare()
        
        
    def prepare(self):
        codes_all = {}
        d_all = {}
        perm = np.random.permutation(self.n_splits)
        for i in tqdm.tqdm(perm[:self.n_used]):
            dname = glob.glob(self.path+"/%d/min_dfs_codes_split*.json"%(i+1))[0]
            didx = int(dname.split("split")[-1][:-5])
            dname2 = self.path+"/%d/data_split%d.%s"%(i+1, didx, self.dformat)
            with open(dname, 'r') as f:
                codes = json.load(f)
                for key, val in codes.items():
                    codes_all[key] = val
                    
            with open(dname2, 'r') as f:
                d_dict = json.load(f)
                for key, val in d_dict.items():
                    d_all[key] = val
        
        for smiles, code in tqdm.tqdm(codes_all.items()):
            if len(self.data) > self.n_mols_per_dataset:
                break
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
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]