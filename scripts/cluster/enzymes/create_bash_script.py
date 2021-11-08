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
import tqdm

exp = Experiment('compute min dfs codes')

@exp.config
def cfg():
    n_total = 37423
    n_job = 100
    path = '/mnt/ssd/datasets/enzyme/preprocessed/graphs'
    result_dir = '/mnt/ssd/datasets/enzyme/preprocessed/min_dfs'
    target = "scripts/compute_enzyme_dfs_codes.sh"
    log_level = logging.INFO
    run_time = "04:00"
    memory_per_core = "5000"
    number_of_cores = 1

    
@exp.automain
def main(n_total, n_job, path, result_dir, target, log_level, run_time, memory_per_core, number_of_cores, _run, _log):
    logging.basicConfig(level=log_level)
    intervals = np.arange(0, n_total+n_job, n_job)
    with open(target, "w+") as file:
        for i, start_idx in tqdm.tqdm(enumerate(intervals[:-1])):
            end_idx = intervals[i+1]
            command = "bsub -o /cluster/scratch/wendlerc/lsf/ -G ls_krausea -W {} -R 'rusage[mem={}]' -n {} python scripts/cluster/enzymes/compute_min_dfs_codes.py"\
                      " with path={} result_dir={} start_idx={} end_idx={}\n".format(run_time, memory_per_core, number_of_cores, 
                                                                                    path, result_dir, start_idx, end_idx)
            file.write(command)
