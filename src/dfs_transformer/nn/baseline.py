#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:54:36 2021

@author: chrisw
"""

from deepchem.models.torch_models import MPNN
import torch
import torch.nn as nn
import dgl


def collate_dgl(dlist, node_feats, edge_feats, target):
    dglbatch = []
    targetbatch = []
    for d in dlist:
        edge_index = d.edge_index.clone()
        g = dgl.graph((edge_index[0], edge_index[1]))
        g.ndata['x'] = d[node_feats].clone()
        g.edata['edge_attr'] = d[edge_feats].clone()
        targetbatch.append(d[target])
        dglbatch += [g]
    dglbatch = dgl.batch(dglbatch)
    return dglbatch, torch.tensor(targetbatch, dtype=torch.long)


class Gilmer(nn.Module):     
    def __init__(self, n_tasks = 1, 
                 node_out_feats = 64,
                 edge_hidden_feats = 128,
                 num_step_message_passing = 3,
                 num_step_set2set = 6,
                 num_layer_set2set = 3,
                 mode = 'regression',
                 number_atom_features = 30,
                 number_bond_features = 11,
                 n_classes = 2,
                 nfeat_name = 'x',
                 efeat_name = 'edge_attr',
                 **kwargs):
        super().__init__()
        self.mpnn = MPNN(n_tasks=n_tasks,
                         node_out_feats=node_out_feats,
                         edge_hidden_feats=edge_hidden_feats,
                         num_step_message_passing=num_step_message_passing,
                         num_step_set2set=num_step_set2set,
                         num_layer_set2set=num_layer_set2set,
                         mode=mode,
                         number_atom_features=number_atom_features,
                         number_bond_features=number_bond_features,
                         n_classes=n_classes,
                         nfeat_name=nfeat_name,
                         efeat_name=efeat_name)
        
    def forward(self, g):
        if self.mpnn.mode == "classification":
            return self.mpnn(g)[1]
        else:
            return self.mpnn(g)
            
        
    
        