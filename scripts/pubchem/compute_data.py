import pickle
import numpy as np
import torch
import pandas as pd
import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import dfs_code
from torch_geometric.data import InMemoryDataset, Data
import sys
sys.path = ['./src'] + sys.path
from dfs_transformer import EarlyStopping, DFSCodeSeq2SeqFC, smiles2graph
import logging
from sacred import Experiment

exp = Experiment('compute data')

@exp.config
def cfg(_log):
    nr = 0
    total = 10
    n_splits = 64
    max_nodes = np.inf
    max_edges = np.inf
    time_limit = 60
    log_level = logging.INFO
    use_Hs = False
    add_loops = False
    path = "../../results/pubchem/amd-server/noH/timeout60_4"

@exp.automain
def main(path, n_splits, nr, total, max_nodes, max_edges, time_limit, log_level, use_Hs, add_loops, _run, _log):
    logging.basicConfig(level=log_level)
    for i in range(n_splits):
        if i % total == nr:
            dname = glob.glob(path+"/%d/min_dfs_codes_split*.pkl"%(i+1))[0]
            didx = int(dname.split("split")[-1][:-4])
            with open(path+"/%d/min_dfs_codes_split%d.pkl"%(i+1, didx), 'rb') as f:
                codes = pickle.load(f)
            d_dict = {}
            for smiles, code in tqdm.tqdm(codes.items()):
                if len(code["min_dfs_code"]) <= max_edges and len(code["dfs_index"]) <= max_nodes:
                    d = smiles2graph(smiles, useHs=use_Hs, addLoops=add_loops)
                    data = {}
                    data['x'] = d.x.detach().cpu().numpy().tolist()
                    data['z'] = d.z.detach().cpu().numpy().tolist()
                    data['edge_attr'] = d.edge_attr.detach().cpu().numpy().tolist()
                    data['edge_index'] = d.edge_index.detach().cpu().numpy().tolist()
                    data['atom_features'] = d.atom_features.detach().cpu().numpy().tolist()
                    data['bond_features'] = d.bond_features.detach().cpu().numpy().tolist()
                    d_dict[smiles] = data
            with open(path+"/%d/data_split%d.pkl"%(i+1, didx), 'wb') as f:
                pickle.dump(d_dict, f)
            

        

