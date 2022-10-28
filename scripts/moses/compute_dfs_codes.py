import dfs_code
import pickle
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
from dfs_transformer import smiles2mindfscode
import pandas as pd

exp = Experiment('compute minimum dfs codes')

@exp.config
def cfg(_log):
    nr = 0
    total = 10
    max_nodes = np.inf
    max_edges = np.inf
    time_limit = 60
    log_level = logging.DEBUG
    use_Hs = False
    dataset_file = "/home/chrisw/Documents/projects/2021/moses/data/dataset_v1.csv"
    add_loops = False
    max_lines = np.inf#100000
    start_idx = 0

@exp.automain
def main(dataset_file, nr, total, max_nodes, max_edges, time_limit, log_level, use_Hs, add_loops, start_idx, max_lines, _run, _log):
    logging.basicConfig(level=log_level)
    dfs_codes = {}
    
    df = pd.read_csv(dataset_file)
    
    for idx, smiles in tqdm.tqdm(enumerate(df["SMILES"])):
        if idx < start_idx:
            continue
        if idx >= start_idx + max_lines:
            break
        if idx % total == nr:
            try:
                time1 = time.time()
                logging.debug("processing %s"%smiles)
                code, dfs_index = smiles2mindfscode(smiles, 
                                                    useHs=use_Hs, 
                                                    addLoops=add_loops, 
                                                    max_nodes=max_nodes, 
                                                    max_edges=max_edges) 
                time2 = time.time()
                exp.log_scalar('time %s'%smiles, time2-time1)
                dfs_codes[smiles] = {'min_dfs_code':code, 'dfs_index':dfs_index}
                
            except:
                logging.warning('%s failed'%smiles)
                logging.warning(sys.exc_info())
                exp.log_scalar('%s failed with'%smiles, sys.exc_info()[0])
                continue
        
    with NamedTemporaryFile(suffix='.pkl', delete=True) as f:
        with open(f.name, 'wb') as ff:
            pickle.dump(dfs_codes, ff)
        _run.add_artifact(f.name, 'min_dfs_codes_split%d.pkl'%(nr))
    

        

