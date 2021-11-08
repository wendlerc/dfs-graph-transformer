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
import dfs_code
import glob
import os
import tqdm

exp = Experiment('compute min dfs codes')

@exp.config
def cfg():
    log_level = logging.INFO
    path = '/mnt/ssd/datasets/enzyme/preprocessed/graphs'
    result_dir = '/mnt/ssd/datasets/enzyme/preprocessed/min_dfs'
    start_idx = 0
    end_idx = 100
    node_limit = np.inf
    edge_limit = np.inf
    node_type_level = 'aminoacid' # choose between aminoacid and secondary


@exp.automain
def main(log_level, path, result_dir, start_idx, end_idx, node_limit, edge_limit, node_type_level, _run, _log):
    logging.basicConfig(level=log_level)
    os.makedirs(result_dir, exist_ok=True)
    flist = sorted(glob.glob(path+'/*.pkl'))
    for fpath in tqdm.tqdm(flist[start_idx:end_idx]):
        fname = fpath.split('/')[-1]
        with open(fpath, 'rb') as f:
            ddict = pickle.load(f)
            edge_index = ddict['edge_index']
            node_types = ddict['node_types_'+node_type_level]
            edge_types = ddict['edge_types']
            min_dfs_code, min_dfs_index = dfs_code.min_dfs_code_from_edgeindex(edge_index, 
                                                                           node_types.tolist(), 
                                                                           edge_types.tolist())
        with open(result_dir+"/%s"%fname, 'wb') as f:
            pickle.dump({'min_dfs_code': min_dfs_code, 
                         'min_dfs_index': min_dfs_index}, f)

