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


"""
atoms
{'C': 0,
 'O': 1,
 'N': 2,
 'Cl': 3,
 'S': 4,
 'F': 5,
 'P': 6,
 'Se': 7,
 'Br': 8,
 'I': 9,
 'Na': 10,
 'B': 11,
 'K': 12,
 'Li': 13,
 'H': 14,
 'Si': 15,
 'Ca': 16,
 'Rb': 17,
 'Te': 18,
 'Zn': 19,
 'Mg': 20,
 'As': 21,
 'Al': 22,
 'Ba': 23,
 'Be': 24,
 'Sr': 25,
 'Ag': 26,
 'Bi': 27,
 'Ra': 28,
 'Kr': 29,
 'Cs': 30,
 'Xe': 31,
 'He': 32}

bounds
{rdkit.Chem.rdchem.BondType.SINGLE: 0,
 rdkit.Chem.rdchem.BondType.DOUBLE: 1,
 rdkit.Chem.rdchem.BondType.AROMATIC: 2,
 rdkit.Chem.rdchem.BondType.TRIPLE: 3}
"""

class ChEMBL(InMemoryDataset):
    def __init__(self, path):
        super().__init__()
        self.data, self.slices = torch.load(path)

exp = Experiment('compute minimum dfs codes')

@exp.config
def cfg(_log):
    nr = 0
    max_nodes = np.inf
    time_limit = 60
    log_level = logging.INFO
    use_Hs = True

@exp.automain
def main(nr, max_nodes, time_limit, log_level, use_Hs, _run, _log):
    logging.basicConfig(level=log_level)
    if use_Hs:
        path = 'datasets/ChEMBL/preprocessedPlusHs_split%d.pt'%nr
    else:
        path = 'datasets/ChEMBL/preprocessedNoHs_split%d.pt'%nr
    dataset = ChEMBL(path)
    
    dfs_codes = {}
    
    for data in tqdm.tqdm(dataset):
        vertex_labels = data.z.detach().cpu().numpy().tolist()
        if len(vertex_labels) > max_nodes:
            exp.log_scalar('skipped', data.name)
            logging.info('skipped %s'%data.name)
            continue
        edge_features = data.edge_attr.detach().cpu().numpy()
        edge_labels = np.argmax(edge_features, axis=1).tolist()
        try:
            time1 = time.time()
            code, dfs_index = func_timeout.func_timeout(time_limit, 
                                                     dfs_code.min_dfs_code_from_torch_geometric,
                                                     args=[data, vertex_labels, edge_labels])            
            time2 = time.time()
            exp.log_scalar('time %s'%data.name, time2-time1)
            dfs_codes[data.name] = {'min_dfs_code':code, 'dfs_index':dfs_index}
        except func_timeout.FunctionTimedOut:
            exp.log_scalar('timeout_index', data.name)
            logging.warning('Computing the minimal DFS code of %s timed out with a timelimit of %d seconds.'%(data.name, time_limit))
            continue
        except:
            logging.warning('%s failed'%data.name)
            exp.log_scalar('failed', data.name)
            continue
        
        
    with NamedTemporaryFile(suffix='.json', delete=True) as f:
        with open(f.name, 'w') as ff:
            json.dump(dfs_codes, ff)
        _run.add_artifact(f.name, 'min_dfs_codes_split%d.json'%(nr))
        
    return dfs_codes

