#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:54:35 2021

@author: chrisw
"""

import torch
import torch.nn as nn
import numpy as np
import dfs_code
from chemprop.features.featurization import atom_features, bond_features
from copy import deepcopy
from collections import defaultdict


def to_cuda(T, device):
    if type(T) is dict:
        return {key: value.to(device) for key, value in T.items()}
    elif type(T) is list:
        return [t.to(device) for t in T]
    else:
        return T.to(device)

def seq_loss(pred, target, m, ce=nn.CrossEntropyLoss(ignore_index=-1)):
    """
    
    Parameters
    ----------
    pred : prediction sequence [seq_len, batch, 5*emb_dim]
    target : target sequence [seq_len, batch, 5] (or 8 instead of 5 cause of the indices)
    m : model config
    ce : loss to apply for each component. The default is nn.CrossEntropyLoss(ignore_index=-1).

    Returns
    -------
    loss : loss for the batch

    """
    dfs1, dfs2, atm1, atm2, bnd = pred
    pred_dfs1 = torch.reshape(dfs1, (-1, m.max_nodes))
    pred_dfs2 = torch.reshape(dfs2, (-1, m.max_nodes))
    pred_atm1 = torch.reshape(atm1, (-1, m.n_atoms))
    pred_atm2 = torch.reshape(atm2, (-1, m.n_atoms))
    pred_bnd = torch.reshape(bnd, (-1, m.n_bonds))
    tgt_dfs1 = target[:, :, 0].view(-1)
    tgt_dfs2 = target[:, :, 1].view(-1)
    tgt_atm1 = target[:, :, 2].view(-1)
    tgt_atm2 = target[:, :, 4].view(-1)
    tgt_bnd = target[:, :, 3].view(-1)
    loss = ce(pred_dfs1, tgt_dfs1) 
    loss += ce(pred_dfs2, tgt_dfs2)
    loss += ce(pred_atm1, tgt_atm1)
    loss += ce(pred_bnd, tgt_bnd)
    loss += ce(pred_atm2, tgt_atm2)
    return loss 

def seq_acc(pred, target, idx=0):
    with torch.no_grad():
        tgt_idx = {0:0, 1:1, 2:2, 3:4, 4:3}
        tgt = target[:, :, tgt_idx[idx]].view(-1)
        prd = pred[idx].reshape(tgt.shape[0], -1)
        mask = tgt != -1
        n_tgts = torch.sum(mask)
        acc = (torch.argmax(prd[mask], axis=1) == tgt[mask]).sum()/n_tgts
        return acc


def BERTize(codes, fraction_missing=0.15):
    inputs = []
    targets = []
    for code in codes:
        n = len(code)
        perm = np.random.permutation(n)
        target_idx = perm[:int(fraction_missing*n)]
        input_idx = perm[int(fraction_missing*n):]
        inp = code.clone()
        target = code.clone()
        target[input_idx] = -1
        inp[target_idx] = -1
        inputs += [inp]
        targets += [target]
    return inputs, targets


def collate_BERT(dlist, mode="min2min", fraction_missing=0.1, use_loops=False):
        node_batch = [] 
        edge_batch = []
        code_batch = []
        if "properties" in dlist[0].keys:
            prop_batch = defaultdict(list)
        if use_loops:
            loop = torch.tensor(bond_features(None)).unsqueeze(0)
        for d in dlist:
            
            edge_features = d.edge_features.clone()
            
            if mode == "min2min":
                code = d.min_dfs_code.clone()
                index = d.min_dfs_index.clone()
            elif mode == "rnd2rnd":
                rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                         d.z.numpy().tolist(), 
                                                                         np.argmax(d.edge_attr.numpy(), axis=1))                
                code = torch.tensor(rnd_code)
                index = torch.tensor(rnd_index)
            else:
                raise ValueError("unknown config.training.mode %s"%mode)
                
            if use_loops:
                edge_features = torch.cat((edge_features, loop), dim=0)
                vids = torch.argsort(index).unsqueeze(1)
                eids = torch.ones_like(vids)*(edge_features.shape[0] - 1)
                nattr = d.z[vids]
                eattr = torch.ones_like(vids)*4 # 4 stands for loop
                arange = index[vids]
                loops = torch.cat((arange, arange, nattr, eattr, nattr, vids, eids, vids), dim=1)
                code = torch.cat((code, loops), dim=0)
                
            node_batch += [d.node_features.clone()]
            edge_batch += [edge_features]
            code_batch += [code]
            if "properties" in dlist[0].keys:
                for name, prop in d.properties.items():
                    prop_batch[name] += [prop]
                    
        inputs, outputs = BERTize(code_batch, fraction_missing=fraction_missing)
        targets = nn.utils.rnn.pad_sequence(outputs, padding_value=-1)
        if "properties" in dlist[0].keys:
            prop_batch = {name: torch.tensor(deepcopy(plist)) for name, plist in prop_batch.items()}
            return inputs, node_batch, edge_batch, targets, prop_batch
        return inputs, node_batch, edge_batch, targets 
    
    
def collate_rnd2min(dlist, use_loops=False):
        node_batch = [] 
        edge_batch = []
        min_code_batch = []
        rnd_code_batch = []
        if use_loops:
            loop = torch.tensor(bond_features(None)).unsqueeze(0)
        for d in dlist:
            edge_features = d.edge_features.clone()
            min_code = d.min_dfs_code.clone()
            rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                     d.z.numpy().tolist(), 
                                                                     np.argmax(d.edge_attr.numpy(), axis=1))
            rnd_code = torch.tensor(rnd_code)
            if use_loops:
                edge_features = torch.cat((edge_features, loop), dim=0)
                min_vids = torch.argsort(d.min_dfs_index).unsqueeze(1)
                rnd_vids = torch.argsort(torch.tensor(rnd_index, dtype=torch.long)).unsqueeze(1)
                eids = torch.ones_like(min_vids)*(edge_features.shape[0] - 1)
                min_nattr = d.z[min_vids]
                rnd_nattr = d.z[rnd_vids]
                eattr = torch.ones_like(min_vids)*4 # 4 stands for loop
                arange = d.min_dfs_index[min_vids]
                
                min_loops = torch.cat((arange, arange, min_nattr, eattr, min_nattr, min_vids, eids, min_vids), dim=1)
                rnd_loops = torch.cat((arange, arange, rnd_nattr, eattr, rnd_nattr, rnd_vids, eids, rnd_vids), dim=1)
                min_code = torch.cat((min_code, min_loops), dim=0)
                rnd_code = torch.cat((rnd_code, rnd_loops), dim=0)
                
            node_batch += [d.node_features.clone()]
            edge_batch += [edge_features]
            min_code_batch += [min_code]
            rnd_code_batch += [rnd_code]
        targets = nn.utils.rnn.pad_sequence(min_code_batch, padding_value=-1)
        return rnd_code_batch, node_batch, edge_batch, targets 
    
    
def collate_downstream(dlist, alpha=0, use_loops=False, use_min=False):
    smiles = []
    node_batch = [] 
    edge_batch = []
    y_batch = []
    rnd_code_batch = []
    if use_loops:
        loop = torch.tensor(bond_features(None)).unsqueeze(0)
    for d in dlist:
        edge_features = d.edge_features.clone()
        if use_min:
            code = d.min_dfs_code.clone()
            index = d.min_dfs_index.clone()
        else:
            code, index = dfs_code.rnd_dfs_code_from_torch_geometric(d, d.z.numpy().tolist(), 
                                                                     np.argmax(d.edge_attr.numpy(), axis=1).tolist())
            
            code = torch.tensor(np.asarray(code), dtype=torch.long)
            index = torch.tensor(np.asarray(index), dtype=torch.long)
        
        if use_loops:
            edge_features = torch.cat((edge_features, loop), dim=0)
            vids = torch.argsort(index).unsqueeze(1)
            eids = torch.ones_like(vids)*(edge_features.shape[0] - 1)
            nattr = d.z[vids]
            eattr = torch.ones_like(vids)*4 # 4 stands for loop
            arange = index[vids]
            loops = torch.cat((arange, arange, nattr, eattr, nattr, vids, eids, vids), dim=1)
            code = torch.cat((code, loops), dim=0).clone()
        
        rnd_code_batch += [code]
        node_batch += [d.node_features.clone()]
        edge_batch += [edge_features]
        y_batch += [d.y.clone()]
        smiles += [deepcopy(d.smiles)]
    y = torch.cat(y_batch).unsqueeze(1)
    y = (1-alpha)*y + alpha/2
    return smiles, rnd_code_batch, node_batch, edge_batch, y