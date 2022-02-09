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
from torch.utils.data import DataLoader
from collections import defaultdict
import yaml
from sklearn.metrics import roc_auc_score
import networkx as nx
import functools
import sys
sys.path.append("src")
import dfs_code 
from dfs_transformer import collate_downstream, DFSCodeSeq2SeqFC, Trainer, KarateClubDataset
from dfs_transformer import to_cuda as to_cuda_
import argparse
#torch.multiprocessing.set_sharing_strategy('file_system')

import networkx as nx

def collate_fn(dlist, use_min=False, rep=1):
    dfs_codes = defaultdict(list)
    node_batch = [] 
    edge_batch = []
    y_batch = []
    rnd_code_batch = []
    nystroem_batch = []
    
    for d in dlist:
        for r in range(rep):
            edge_features = d.edge_features.clone()
            if use_min:
                code = d.min_dfs_code.clone()
                index = d.min_dfs_index.clone()
            else:
                code, index = dfs_code.rnd_dfs_code_from_edgeindex(d.edge_index.numpy(), 
                                                                   d.node_labels.numpy().tolist(), 
                                                                   d.edge_labels.numpy().tolist())

                code = torch.tensor(np.asarray(code), dtype=torch.long)
                index = torch.tensor(np.asarray(index), dtype=torch.long)


            rnd_code_batch += [code]
            node_batch += [d.node_features.clone()]
            edge_batch += [edge_features]
            y_batch += [d.y.unsqueeze(0).clone()]
            nystroem_batch += [d.nystroem.clone()]

    y = torch.cat(y_batch).unsqueeze(1)
    nystroem = torch.cat(nystroem_batch)

    
    for inp, nfeats, efeats in zip(rnd_code_batch, node_batch, edge_batch):
        dfs_codes['dfs_from'] += [inp[:, 0]]
        dfs_codes['dfs_to'] += [inp[:, 1]]
        atm_from_feats = nfeats[inp[:, -3]]
        atm_to_feats = nfeats[inp[:, -1]]
        bnd_feats = efeats[inp[:, -2]]
        dfs_codes['atm_from'] += [atm_from_feats]
        dfs_codes['atm_to'] += [atm_to_feats]
        dfs_codes['bnd'] += [bnd_feats]

    dfs_codes = {key: nn.utils.rnn.pad_sequence(values, padding_value=-1000).clone()
                 for key, values in dfs_codes.items()}
    return dfs_codes, y, nystroem

def collate_train(dlist, use_min=False, rep=1):
    dfs_codes, y, _ = collate_fn(dlist, use_min=use_min, rep=rep)
    return dfs_codes, y

def collate_pretrain(dlist, use_min=False, rep=1):
    dfs_codes, _, nystroem = collate_fn(dlist, use_min=use_min, rep=rep)
    return dfs_codes, nystroem

class TransformerCLS(nn.Module):
    def __init__(self, encoder, fingerprint='cls'):
        super(TransformerCLS, self).__init__()
        self.encoder = encoder
        self.fingerprint = fingerprint
    
    def forward(self, dfs_code):
        features = self.encoder.encode(dfs_code, method=self.fingerprint)
        return features

class TransformerPlusHead(nn.Module):
    def __init__(self, encoder, n_classes, fingerprint='cls', linear=True):
        super(TransformerPlusHead, self).__init__()
        self.encoder = encoder
        if linear:
            self.head = nn.Linear(encoder.get_n_encoding(fingerprint), n_classes)
        else:
            self.head = nn.Sequential(nn.Linear(encoder.get_n_encoding(fingerprint), 128),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(128, 128),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(128, n_classes))
        self.fingerprint = fingerprint
    
    def forward(self, dfs_code):
        features = self.encoder.encode(dfs_code, method=self.fingerprint)
        output = self.head(features)
        return output

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
parser.add_argument('--wandb_dir', type=str, default="./wandb")
parser.add_argument('--wandb_group', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--graph_file', type=str, default="/mnt/ssd/datasets/graphs/reddit_threads/reddit_edges.json")
parser.add_argument('--label_file', type=str, default="/mnt/ssd/datasets/graphs/reddit_threads/reddit_target.csv")
parser.add_argument('--model_yaml', type=str, default="./config/selfattn/model/bert.yaml")
parser.add_argument('--nonlinear', action='store_true')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=5*1e-5)
parser.add_argument('--clip_gradient_norm', type=float, default=0.5)
parser.add_argument('--n_repetitions', type=int, default=100)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--max_edges', type=int, default=200)
parser.add_argument('--n_samples', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--pretrain_flag', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--overwrite', type=json.loads, default="{}")
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--use_min', action='store_true')
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
config.model_yaml = args.model_yaml
config.nonlinear = args.nonlinear
config.clip_gradient_norm = args.clip_gradient_norm

config.num_workers = args.num_workers
config.seed = args.seed
config.training = {}
config.pretrain_flag = args.pretrain_flag
config.device=args.device

dataset = KarateClubDataset(config.graph_file, config.label_file, max_n=config.n_samples, max_edges=config.max_edges)

with open(args.model_yaml) as file:
    m = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    
m["n_atoms"] = int(dataset.maxdegree)+1
m["n_bonds"] = 2
m["n_edge_features"] = 2
m["max_edges"] = config.max_edges
m["max_nodes"] = config.max_edges
m["use_min"] = args.use_min


dim_input = dataset[0].x.shape[1]
collate_train_ = functools.partial(collate_train, use_min=m['use_min'], rep=config.rep)
collate_pretrain_ = functools.partial(collate_pretrain, use_min=m['use_min'], rep=config.rep)

loader = DataLoader(dataset,  
                    batch_size=config.batch_size, collate_fn=collate_train_,
                    num_workers=config.num_workers)
d = next(iter(loader))
m["n_node_features"] = d[0]['atm_from'].shape[2]

config.model = m.to_dict()

for key,value in args.overwrite.items():
    if type(value) is dict:
        for key1,value1 in value.items():
            config[key][key1] = value1
    else:
        config[key] = value


run = wandb.init(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity, 
                 config=config, job_type="evaluation", name=args.name, settings=wandb.Settings(start_method="fork"),
                 group=args.wandb_group)

device = torch.device(config.device)
to_cuda = functools.partial(to_cuda_, device=device)


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
                             batch_size=config.batch_size, collate_fn=collate_train_,
                             num_workers=config.num_workers)
    validloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(valid_idx), 
                             batch_size=config.batch_size, collate_fn=collate_train_,
                             num_workers=config.num_workers)
    testloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(test_idx), 
                            batch_size=config.batch_size, collate_fn=collate_train_,
                             num_workers=config.num_workers)
    pretrainloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(train_idx), 
                             batch_size=config.batch_size, collate_fn=collate_pretrain_,
                             num_workers=config.num_workers)
    prevalidloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(valid_idx), 
                             batch_size=config.batch_size, collate_fn=collate_pretrain_,
                             num_workers=config.num_workers)
    pretestloader = DataLoader(dataset, sampler=torch.utils.data.SubsetRandomSampler(test_idx), 
                            batch_size=config.batch_size, collate_fn=collate_pretrain_,
                             num_workers=config.num_workers)
    
    if config.pretrain_flag:
        encoder = DFSCodeSeq2SeqFC(**m)
        premodel = TransformerCLS(encoder)
        model = TransformerPlusHead(encoder, 2, linear=not config.nonlinear)
    else:
        encoder = DFSCodeSeq2SeqFC(**m)
        model = TransformerPlusHead(encoder, 2, linear=not config.nonlinear)
    
    if config.pretrain_flag:
        pretrainer = Trainer(premodel, pretrainloader, loss_pretrain, lr=config.learning_rate, validloader=pretestloader, 
                  es_period=1*n_train//config.batch_size, lr_adjustment_period=10*n_train//config.batch_size//4, wandb_run=run,
                  clip_gradient_norm=config.clip_gradient_norm, n_epochs=config.n_epochs, device=device)
    
    trainer = Trainer(model, trainloader, loss, metrics={'auc': auc}, lr=config.learning_rate, validloader=testloader, 
                  es_period=1*n_train//config.batch_size, lr_adjustment_period=10*n_train//config.batch_size//4, wandb_run=run,
                  clip_gradient_norm=config.clip_gradient_norm, n_epochs=config.n_epochs, device=device)
    
    if config.pretrain_flag:
        pretrainer.fit()
    trainer.fit()
    
    rocauc = 0.
    with torch.no_grad():
        model.eval()
        pbar_valid = tqdm.tqdm(testloader)
        for i, data in enumerate(pbar_valid):
            pred = model(to_cuda(data[0]))
            rocauc = (rocauc*i + auc(pred, to_cuda(data[1])).item())/(i+1)

    scores['test_roc_auc'] += [rocauc]
    # Calculate accuracy of classification.
    wandb.log({'test_roc_auc': rocauc})
for key, value in scores.items():
    wandb.log({'mean_'+key: np.mean(value),
               'std_'+key: np.std(value)})
