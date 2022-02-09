#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:02:08 2022

@author: chrisw
"""

import json
import numpy as np
import glob
import wandb
import os
import torch.optim as optimizers
import pandas as pd
from copy import deepcopy
from ml_collections import ConfigDict 
import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import defaultdict
import yaml
from sklearn.metrics import roc_auc_score
import networkx as nx

import sys
sys.path.append("src")
import dfs_code 
from dfs_transformer import collate_downstream, DFSCodeSeq2SeqFC, TrainerGNN, KarateClubDataset
import argparse
#torch.multiprocessing.set_sharing_strategy('file_system')

import torch_geometric.nn as tnn
import networkx as nx
from torch_geometric.loader import DataLoader

def loss_pretrain(pred, y, l=nn.MSELoss()):
    return l(pred, y)

def loss(pred, y, ce=nn.CrossEntropyLoss()):
    return ce(pred, y.squeeze())

def auc(pred, y):
    try:
        return roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy()[:, 1])
    except ValueError:
        return torch.tensor(0.)
    
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
parser.add_argument('--wandb_project', type=str, default="karateclub-rep100")
parser.add_argument('--wandb_mode', type=str, default="online")
parser.add_argument('--wandb_group', type=str, default=None)
parser.add_argument('--wandb_dir', type=str, default="./wandb")
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--graph_file', type=str, default="/mnt/ssd/datasets/graphs/reddit_threads/reddit_edges.json")
parser.add_argument('--label_file', type=str, default="/mnt/ssd/datasets/graphs/reddit_threads/reddit_target.csv")
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--n_repetitions', type=int, default=100)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--max_edges', type=int, default=200)
parser.add_argument('--n_samples', type=int, default=None)
parser.add_argument('--n_channels', type=int, default=32)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--model', type=str, default="tnn.models.GCN")
parser.add_argument('--readout', type=str, default="tnn.global_mean_pool")
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--pretrain_flag', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
args = parser.parse_args()

config = wandb.config
config.graph_file = args.graph_file
config.label_file = args.label_file
#config.graph_file = "/mnt/ssd/datasets/graphs/twitch_egos/twitch_edges.json"
#config.label_file = "/mnt/ssd/datasets/graphs/twitch_egos/twitch_target.csv"
config.batch_size = args.batch_size
config.n_epochs = args.n_epochs
config.learning_rate = args.learning_rate
config.n_repetitions = max(args.n_repetitions, args.end)
config.start = args.start
config.end = args.end
config.rep = args.rep
config.max_edges = args.max_edges
config.n_samples = args.n_samples
config.n_channels = args.n_channels
config.n_layers = args.n_layers
config.num_workers = args.num_workers
config.model = args.model
config.readout = args.readout
config.seed = args.seed
config.training = {}
config.pretrain_flag = args.pretrain_flag

run = wandb.init(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity, 
                 config=config, job_type="evaluation", name=args.name, settings=wandb.Settings(start_method="fork"),
                 group=args.wandb_group)

dataset = KarateClubDataset(config.graph_file, config.label_file, max_n=config.n_samples, max_edges=config.max_edges)
dim_input = dataset[0].x.shape[1]


np.random.seed(config.seed)
n = len(dataset)
n_train = int(0.8*n)
n_valid = 0
n_test = n - n_train - n_valid
perms = [np.random.permutation(n) for _ in range(config.n_repetitions)]
scores = defaultdict(list)
for perm in perms[config.start:config.end]:
    train_idx = torch.tensor(perm[:n_train], dtype=torch.long)
    valid_idx = torch.tensor(perm[n_train:n_train+n_valid].tolist(), dtype=torch.long)
    test_idx = torch.tensor(perm[n_train+n_valid:].tolist(), dtype=torch.long)
    ce = nn.CrossEntropyLoss()
    trainloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(train_idx), 
                             batch_size=config.batch_size, num_workers=config.num_workers)
    validloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(valid_idx), 
                             batch_size=config.batch_size, num_workers=config.num_workers)
    testloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(test_idx), 
                            batch_size=config.batch_size, num_workers=config.num_workers)
    
    
    if config.pretrain_flag:
        gnn = eval(config.model)(dim_input, config.n_channels, config.n_layers, 600)
        premodel = tnn.Sequential('x, edge_index, batch', [
            (gnn, 'x, edge_index -> x'),
            (eval(config.readout), 'x, batch -> x'),
        ])
        
        model = tnn.Sequential('x, edge_index, batch', [
            (gnn, 'x, edge_index -> x'),
            (eval(config.readout), 'x, batch -> x'),
            (nn.Linear(600, 2),'x -> x')
        ])
    else:
        gnn = eval(config.model)(dim_input, config.n_channels, config.n_layers)
        model = tnn.Sequential('x, edge_index, batch', [
            (gnn, 'x, edge_index -> x'),
            (eval(config.readout), 'x, batch -> x'),
            (nn.Linear(config.n_channels, 2),'x -> x')
        ])
    
    if config.pretrain_flag:
        pretrainer = TrainerGNN(premodel, trainloader, loss_pretrain, target='nystroem', n_epochs=config.n_epochs, lr=config.learning_rate, 
                      es_period=n_train//config.batch_size, lr_adjustment_period=10*n_train//config.batch_size//4, wandb_run=run,
                      clip_gradient_norm=None, lr_patience=5, es_patience=20)
    
    trainer = TrainerGNN(model, trainloader, loss, metrics={'auc': auc}, n_epochs=config.n_epochs, lr=config.learning_rate, 
                  es_period=n_train//config.batch_size, lr_adjustment_period=10*n_train//config.batch_size//4, wandb_run=run,
                  clip_gradient_norm=None, lr_patience=5, es_patience=20)
    
    if config.pretrain_flag:
        pretrainer.fit()
    trainer.fit()
    
    rocauc = 0.
    with torch.no_grad():
        model.eval()
        pbar_valid = tqdm.tqdm(testloader)
        for i, data in enumerate(pbar_valid):
            data = data.cuda()
            pred = model(data.x, data.edge_index, data.batch)
            rocauc = (rocauc*i + auc(pred, data.y).item())/(i+1)

    scores['test_roc_auc'] += [rocauc]
    # Calculate accuracy of classification.
    wandb.log({'test_roc_auc': rocauc})
for key, value in scores.items():
    wandb.log({'mean_'+key: np.mean(value),
               'std_'+key: np.std(value)})
