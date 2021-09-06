import json
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
    dont_trim = True
    path = "../../results/pubchem/amd-server/noH/timeout60_4"

@exp.automain
def main(path, n_splits, nr, total, max_nodes, max_edges, time_limit, log_level, use_Hs, add_loops, dont_trim, _run, _log):
    logging.basicConfig(level=log_level)
    for i in range(n_splits):
        if i % total == nr:
            dname = glob.glob(path+"/%d/min_dfs_codes_split*.json"%(i+1))[0]
            didx = int(dname.split("split")[-1][:-5])
            with open(path+"/%d/min_dfs_codes_split%d.json"%(i+1, didx), 'r') as f:
                codes = json.load(f)
            d_dict = {}
            for smiles, code in tqdm.tqdm(codes.items()):
                d = smiles2graph(smiles, useHs=use_Hs, addLoops=add_loops, dontTrimEdges=dont_trim, skipCliqueCheck=False)
                data = {}
                data['x'] = d.x.detach().cpu().numpy().tolist()
                data['z'] = d.z.detach().cpu().numpy().tolist()
                data['edge_attr'] = d.edge_attr.detach().cpu().numpy().tolist()
                data['edge_index'] = d.edge_index.detach().cpu().numpy().tolist()
                data['atom_features'] = d.atom_features.detach().cpu().numpy().tolist()
                data['bond_features'] = d.bond_features.detach().cpu().numpy().tolist()
                d_dict[smiles] = data
            with open(path+"/%d/data_split%d.json"%(i+1, didx), 'w') as f:
                json.dump(d_dict, f)
            

        

