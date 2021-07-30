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

exp = Experiment('compute minimum dfs codes')

@exp.config
def cfg(_log):
    path = 'datasets/qm9_torch_geometric/'
    start = 0
    n_graphs = 5000
    time_limit = 60
    log_level = logging.WARNING

@exp.automain
def main(path, start, n_graphs, time_limit, log_level, _run, _log):
    logging.basicConfig(level=log_level)
    dataset = QM9(path)
    end = min(len(dataset), start+n_graphs)
    
    dfs_codes = {}
    
    for i in tqdm.tqdm(range(start, end)):
        data = dataset[i]
        vertex_features = data.x.detach().cpu().numpy()
        edge_features = data.edge_attr.detach().cpu().numpy()
        vertex_labels = np.argmax(vertex_features[:, :5], axis=1).tolist()
        edge_labels = np.argmax(edge_features, axis=1).tolist()
        try:
            time1 = time.time()
            code, id2row = func_timeout.func_timeout(time_limit, 
                                                     dfs_code.min_dfs_code_from_torch_geometric,
                                                     args=[data, vertex_labels, edge_labels])            
            time2 = time.time()
            logging.info('Computed the minimal DFS code of %s in %2.4f seconds.'%(data.name, time2-time1))
            exp.log_scalar('time %s'%data.name, time2-time1)
        except func_timeout.FunctionTimedOut:
            exp.log_scalar('timeout_index', i)
            logging.warning('Computing the minimal DFS code of %s timed out with a timelimit of %d seconds.'%(data.name, time_limit))
        dfs_codes[data.name] = {'min_dfs_code':code, 'edge_id_2_edge_number':id2row}
        
    with NamedTemporaryFile(suffix='.json', delete=True) as f:
        json.dump(dfs_codes, f)
        _run.add_artifact(f.name, 'min_dfs_codes_%d_to_%d.json'%(start, end))
        
    return dfs_codes
