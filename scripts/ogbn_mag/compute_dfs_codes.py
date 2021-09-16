import dfs_code
import pickle
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
sys.path = ['./src'] + sys.path
from dfs_transformer import smiles2graph
from ogb.nodeproppred import PygNodePropPredDataset
from collections import defaultdict
import networkx as nx
from networkx.generators.ego import ego_graph
from ml_collections import ConfigDict

def ego2data(idx, ego, node_types, edge_type_dict, graph):
    d = {}
    d['idx'] = idx
    i2i = {iold: inew for inew, iold in enumerate(ego.nodes)}
    edges = []
    elabels = []
    for e in ego.edges:
        edges += [[i2i[e[0]], i2i[e[1]]], [i2i[e[1]], i2i[e[0]]]]
        elabels += 2*[edge_type_dict[tuple(e)]]
    d['edge_index'] = np.asarray(edges) 
    d['node_labels'] = node_types[np.asarray(ego.nodes)]
    d['edge_labels'] = np.asarray(elabels)
    d['graph_features'] = graph['x_dict']['paper'][idx].numpy()
    d['y'] = graph['y_dict']['paper'][idx].item()
    return ConfigDict(d)

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

@exp.automain
def main(nr, total, max_nodes, max_edges, time_limit, log_level, start_idx, max_lines, _run, _log):
    logging.basicConfig(level=log_level)
    
    dataset = PygNodePropPredDataset(name = "ogbn-mag") 

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]
    
    nodes = np.arange(sum([val for val in graph['num_nodes_dict'].values()]))
    node_types = []
    type_order = ["paper", "author", "field_of_study", "institution"]
    
    old2newidx = defaultdict(dict)
    curr = 0
    for tidx, key in enumerate(type_order):
        for idx_old in range(graph['num_nodes_dict'][key]):
            old2newidx[key][idx_old] = curr
            curr += 1
            node_types += [tidx]
    node_types = np.asarray(node_types)
    
    edge_index = graph['edge_index_dict'] 
    edge_reltype= graph['edge_reltype']
    edges = []
    edge_types = []
    edge_type_dict = {}
    for key in edge_index.keys():
        curr_edges = edge_index[key].detach().numpy().T
        curr_type = edge_reltype[key].squeeze().detach().numpy().tolist()
        for edge in curr_edges:
            e0 = old2newidx[key[0]][edge[0]]
            e1 = old2newidx[key[2]][edge[1]]
            edges += [[e0, e1], [e1, e0]]
            edge_type_dict[(e0, e1)] = curr_type[0]
            edge_type_dict[(e1, e0)] = curr_type[0]
        edge_types += 2*curr_type
    
    g = nx.Graph()
    g.add_nodes_from(nodes.tolist())
    g.add_edges_from(edges)
    
    
    dfs_codes = {}
    d_dict = {}
    
    for idx in tqdm.tqdm(nodes[node_types == 0]):
        if idx < start_idx:
            continue
        if idx >= start_idx + max_lines:
            break
        if idx % total == nr:
            try:
                time1 = time.time()
                d = ego2data(idx, ego_graph(g, idx, radius=1), node_types, edge_type_dict, graph)
                code, dfs_index = dfs_code.min_dfs_code_from_torch_geometric(d, 
                                                                             d['node_labels'].tolist(), 
                                                                             d['edge_labels'].tolist(),
                                                                             timeout=time_limit)    
                time2 = time.time()
                exp.log_scalar('time %d'%idx, time2-time1)
                dfs_codes[idx] = {'min_dfs_code':code, 'dfs_index':dfs_index}
                d_dict[idx] = d.to_dict()
            except:
                logging.warning('%d failed'%idx)
                exp.log_scalar('%d failed with'%idx, sys.exc_info()[0])
                continue
        
    with NamedTemporaryFile(suffix='.pkl', delete=True) as f:
        with open(f.name, 'wb') as ff:
            pickle.dump(dfs_codes, ff)
        _run.add_artifact(f.name, 'min_dfs_codes_split%d.pkl'%(nr))
    
    with NamedTemporaryFile(suffix='.pkl', delete=True) as f:
        with open(f.name, 'wb') as ff:
            pickle.dump(d_dict, ff)
        _run.add_artifact(f.name, 'data_split%d.pkl'%nr)
        

