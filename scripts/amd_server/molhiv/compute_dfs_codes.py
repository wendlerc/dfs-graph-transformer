import dfs_code
import json
import tqdm
import numpy as np
import func_timeout
import time
import logging
from sacred import Experiment
from tempfile import NamedTemporaryFile
from torch_geometric.data import InMemoryDataset
import torch
import sys
sys.path = ['./src'] + sys.path
from dfs_transformer import smiles2graph
import deepchem as dc
import traceback
import pandas as pd

exp = Experiment('compute minimum dfs codes')

@exp.config
def cfg(_log):
    max_nodes = np.inf
    max_edges = np.inf
    time_limit = 1
    log_level = logging.INFO
    use_Hs = False
    add_loops = False

@exp.automain
def main(max_nodes, max_edges, time_limit, log_level, use_Hs, add_loops, _run, _log):
    logging.basicConfig(level=log_level)
    dfs_codes = {}
    d_dict = {}
    
    train = pd.read_csv("datasets/ogbg-molhiv/molhiv/0/train.csv")
    valid = pd.read_csv("datasets/ogbg-molhiv/molhiv/0/valid.csv")
    test = pd.read_csv("datasets/ogbg-molhiv/molhiv/0/test.csv")
    smiles_list = train['smiles'].tolist() + valid['smiles'].tolist() + test['smiles'].tolist()
    
    
    for idx, smiles in tqdm.tqdm(enumerate(smiles_list)):
        try:
            time1 = time.time()
            d = smiles2graph(smiles, use_Hs, add_loops, max_nodes, max_edges)
            code, dfs_index = dfs_code.min_dfs_code_from_torch_geometric(d, 
                                                                         d.z.numpy().tolist(), 
                                                                         np.argmax(d.edge_attr.numpy(), axis=1),
                                                                         timeout=time_limit)          
            time2 = time.time()
            exp.log_scalar('time %s'%smiles, time2-time1)
            dfs_codes[smiles] = {'min_dfs_code':code, 'dfs_index':dfs_index}
            data = {}
            data['x'] = d.x.detach().cpu().numpy().tolist()
            data['z'] = d.z.detach().cpu().numpy().tolist()
            data['edge_attr'] = d.edge_attr.detach().cpu().numpy().tolist()
            data['edge_index'] = d.edge_index.detach().cpu().numpy().tolist()
            data['atom_features'] = d.atom_features.detach().cpu().numpy().tolist()
            data['bond_features'] = d.bond_features.detach().cpu().numpy().tolist()
            d_dict[smiles] = data
        except:
            logging.warning('%s failed'%smiles)
            exp.log_scalar('%s failed with'%smiles, sys.exc_info()[0])
            logging.warning(sys.exc_info()[0])
            logging.warning(sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
            continue
        
    with NamedTemporaryFile(suffix='.json', delete=True) as f:
        with open(f.name, 'w') as ff:
            json.dump(dfs_codes, ff)
        _run.add_artifact(f.name, 'min_dfs_codes.json')
        
    with NamedTemporaryFile(suffix='.json', delete=True) as f:
        with open(f.name, 'w') as ff:
            json.dump(d_dict, ff)
        _run.add_artifact(f.name, 'data.json')
        

