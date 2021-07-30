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

class ChEMBL(InMemoryDataset):
    def __init__(self, path):
        super().__init__()
        self.data, self.slices = torch.load(path)

exp = Experiment('compute minimum dfs codes')

@exp.config
def cfg(_log):
    nr = 0
    time_limit = 60
    log_level = logging.WARN

@exp.automain
def main(nr, time_limit, log_level, _run, _log):
    logging.basicConfig(level=log_level)
    path = 'datasets/ChEMBL/preprocessedPlusHs_split%d.pt'%nr
    dataset = ChEMBL(path)
    
    dfs_codes = {}
    
    for data in tqdm.tqdm(dataset):
        vertex_labels = data.z.detach().cpu().numpy().tolist()
        edge_features = data.edge_attr.detach().cpu().numpy()
        edge_labels = np.argmax(edge_features, axis=1).tolist()
        try:
            time1 = time.time()
            code, dfs_index = func_timeout.func_timeout(time_limit, 
                                                     dfs_code.min_dfs_code_from_torch_geometric,
                                                     args=[data, vertex_labels, edge_labels])            
            time2 = time.time()
            exp.log_scalar('time %s'%data.name, time2-time1)
        except func_timeout.FunctionTimedOut:
            exp.log_scalar('timeout_index', data.name)
            logging.warning('Computing the minimal DFS code of %s timed out with a timelimit of %d seconds.'%(data.name, time_limit))
        dfs_codes[data.name] = {'min_dfs_code':code, 'dfs_index':dfs_index}
        
    with NamedTemporaryFile(suffix='.json', delete=True) as f:
        with open(f.name, 'w') as ff:
            json.dump(dfs_codes, ff)
        _run.add_artifact(f.name, 'min_dfs_codes_split%d.json'%(nr))
        
    return dfs_codes

