#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:18:21 2021

@author: chrisw
"""

import pickle
import numpy as np
import logging
from sacred import Experiment
from tempfile import NamedTemporaryFile 
from joblib import Parallel, delayed
import functools
#import dfs_code
import sys 
sys.path.append('src')
from sPickle import *

exp = Experiment('compute graphs')

@exp.config
def cfg():
    log_level = logging.INFO
    n_jobs = 10
    path = "/mnt/ssd/datasets/enzyme/full_nodfs.pkl"
    node_limit = np.inf
    edge_limit = np.inf
    node_types = 'aminoacid' # choose between aminoacid and secondary
    nmax = np.inf
    
    
def work_(edge_index, node_types, edge_types, node_limit, edge_limit):
    import dfs_code
    if len(node_types) > node_limit or len(edge_types) > 2*edge_limit:
        return None
    min_dfs_code, min_dfs_index = dfs_code.min_dfs_code_from_edgeindex(edge_index, 
                                                                       node_types.tolist(), 
                                                                       edge_types.tolist())
    return min_dfs_code, min_dfs_index
    


@exp.automain
def main(log_level, n_jobs, path, node_limit, edge_limit, node_types, nmax, _run, _log):
    logging.basicConfig(level=log_level)
    with open(path, 'rb') as f:
        graphs = pickle.load(f) # this is a ddict
        """
        d = graphs[name] 
        d['edge_index'] 
        d['edge_features'] 
        d['edge_types'] 
        d['node_features']
        d['node_types_aminoacid'] 
        d['node_types_secondary'] 
        """
    
    work = functools.partial(work_, node_limit=node_limit, edge_limit=edge_limit)
    keys = list(graphs.keys())   
    if nmax is not None:
        keys = keys[:nmax]
        graphs = {key: graphs[key] for key in keys}
    
    
    #dfs_code.min_dfs_code_from_edgeindex(edge_index, node_types.tolist(), edge_types.tolist())
    #exit()
    dfs = Parallel(n_jobs=n_jobs)(delayed(work)(graphs[key]['edge_index'].copy(),
                                                graphs[key]['node_types_'+node_types].copy(),
                                                graphs[key]['edge_types'].copy()) for key in keys)
    dfs = {key:{'min_dfs_code': d[0], 'min_dfs_index': d[1]} for key, d in zip(keys, dfs) if d is not None}
    
    with NamedTemporaryFile(suffix='.spkl', delete=True) as f:
        with open(f.name, 'wb') as ff:
            s_dump(dfs, ff)
        _run.add_artifact(f.name, 'min_dfs_codes.spkl')

