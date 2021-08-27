import dfs_code
import json
from torch_geometric.datasets.qm9 import QM9
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
sys.path = ['/home/chrisw/Documents/projects/2021/graph-transformer/src'] + sys.path
from dfs_transformer import smiles2graph

exp = Experiment('compute minimum dfs codes')

@exp.config
def cfg(_log):
    nr = 0
    total = 10
    max_nodes = np.inf
    max_edges = np.inf
    time_limit = 60
    log_level = logging.INFO
    use_Hs = False
    add_loops = False
    smiles_file = "/mnt/ssd/datasets/pubchem_10m.txt/pubchem-10m.txt"

@exp.automain
def main(smiles_file, nr, total, max_nodes, max_edges, time_limit, log_level, use_Hs, add_loops, _run, _log):
    logging.basicConfig(level=log_level)
    dfs_codes = {}
    with open(smiles_file, "r") as f:
        for idx, smiles in tqdm.tqdm(enumerate(f.readlines())):
            if idx % total == nr:
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
                except:
                    logging.warning('%s failed'%smiles)
                    exp.log_scalar('%s failed with', sys.exc_info()[0])
                    continue
        
        
    with NamedTemporaryFile(suffix='.json', delete=True) as f:
        with open(f.name, 'w') as ff:
            json.dump(dfs_codes, ff)
        _run.add_artifact(f.name, 'min_dfs_codes_split%d.json'%(nr))
        

