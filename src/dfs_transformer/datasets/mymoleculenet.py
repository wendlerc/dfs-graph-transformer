#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:27:25 2021

@author: chrisw
"""
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import torch.nn as nn
import numpy as np
import json
import dfs_code
from .utils import smiles2graph

class Deepchem2TorchGeometric(Dataset):
    def __init__(self, smiles, labels, features="chemprop", loaddir=None,
                 max_edges=np.inf, max_nodes=np.inf, onlyRandom=False,
                 useHs=False, addLoops=False):
        """
        Parameters
        ----------
        smiles : iterable, smiles strings
        labels : iterable, labels 
        features : str, which features to use, admissible options "old", "chemprop" and "none". 
        The default is "chemprop".
        loaddir : str, directory containing precomputed dfs codes and features. 
        If loaddir is set then the subsequent options have no effect. The default is None.
        max_edges : int, The default is np.inf.
        max_nodes : int, The default is np.inf.
        onlyRandom : bool, if this is true we don't compute minimal dfs codes. The default is False.
        useHs : bool, whether to model H atoms as vertices or not. The default is False.
        addLoops : bool, whether to add loops to the atoms of the molecular graphs. The default is False.
        """
        self.smiles = smiles
        self.labels = labels[:, np.newaxis]
        self.useHs = useHs
        self.addLoops = addLoops
        self.loaddir = loaddir
        self.data = []
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.onlyRandom = onlyRandom
        self.features = features
        self.prepare()
  
    
    def prepare(self):
        if self.loaddir is not None:
            with open(self.loaddir+'data.json', 'r') as f:
                data = json.load(f)
            with open(self.loaddir+'min_dfs_codes.json', 'r') as f:
                dfs = json.load(f)
        
        for idx in range(len(self.smiles)):
            smiles = self.smiles[idx]
            if self.loaddir is not None:
                if smiles in dfs:
                    d = data[smiles]
                    d = {key:torch.tensor(value) for key, value in d.items()}
                    min_code = dfs[smiles]['min_dfs_code']
                    min_index = dfs[smiles]['dfs_index']
                else:
                    continue
            else:
                d = smiles2graph(smiles, self.useHs, self.addLoops, not self.trimEdges, self.max_nodes, self.max_edges)
                if d is None:
                    continue
                if self.onlyRandom:
                    min_code, min_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                             d.z.numpy().tolist(), 
                                                                             np.argmax(d.edge_attr.numpy(), axis=1))
                else:
                    min_code, min_index = dfs_code.min_dfs_code_from_torch_geometric(d, 
                                                                             d.z.numpy().tolist(), 
                                                                             np.argmax(d.edge_attr.numpy(), axis=1))
                
                
            z = d['z']
            x = d['x']
            if self.features == "old":
                h = x[:, -1].clone().detach().long()
                z_ind = nn.functional.one_hot(z, num_classes=118).float()
                h_ind = nn.functional.one_hot(h, num_classes=5).float()
                node_features = torch.cat((z_ind, x[:, 1:-1], h_ind), dim=1)
                edge_features = d['edge_attr']
            elif self.features == "chemprop":
                node_features = d['atom_features']
                edge_features = d['bond_features']
            else:#no features
                node_features = nn.functional.one_hot(z, num_classes=118).float()
                edge_features = d['edge_attr']
                
                
            self.data += [Data(x=d['x'], z=d['z'], pos=None, edge_index=d['edge_index'],
                            edge_attr=d['edge_attr'], y=torch.tensor(self.labels[idx], dtype=torch.float32),
                            min_dfs_code=torch.tensor(min_code), min_dfs_index=torch.tensor(min_index), 
                            smiles=smiles, node_features=node_features, edge_features=edge_features)]
            
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        return self.data[idx]