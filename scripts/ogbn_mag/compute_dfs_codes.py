import dfs_code
import pickle
import torch
import tqdm
import numpy as np
import func_timeout
import time
import logging
from sacred import Experiment
from tempfile import NamedTemporaryFile
import sys
from collections import defaultdict
from ml_collections import ConfigDict
import glob

exp = Experiment('compute minimum dfs codes')

@exp.config
def cfg(_log):
    nr = 0
    total = 10
    max_nodes = np.inf
    max_edges = np.inf
    time_limit = 60
    log_level = logging.INFO
    start_idx = 0
    max_lines = np.inf#100000
    path = None    
    
@exp.automain
def main(nr, total, max_nodes, max_edges, time_limit, log_level, start_idx, max_lines, path, _run, _log):
    logging.basicConfig(level=log_level)
    for idx, fname in enumerate(glob.glob("%s/*/data_split*.pkl"%path)):
        if idx % total != nr:
            continue
        with open(fname, "rb") as f:
            ddict = pickle.load(f)
        for i, d in tqdm.tqdm(ddict.items()):
            try:
                d = ConfigDict(d)
                time1 = time.time()
                code, dfs_index = dfs_code.min_dfs_code_from_torch_geometric(d, 
                                                                             d.node_labels.tolist(), 
                                                                             d.edge_labels.tolist(),
                                                                             timeout=time_limit)
                d.min_dfs_code = code
                d.dfs_index = dfs_index
                time2 = time.time()
                exp.log_scalar('time %d'%i, time2-time1)
                ddict[i] = d.to_dict()
            except:
                logging.warning('%d failed with %s'%(i, sys.exc_info()))
                exp.log_scalar('%d failed with'%i, sys.exc_info()[0])
                continue
        with open(fname+"%ds.pkl"%time_limit, 'wb') as ff:
            pickle.dump(ddict, ff)
         
        

