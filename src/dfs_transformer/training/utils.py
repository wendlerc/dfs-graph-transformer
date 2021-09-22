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

def to_cuda(T, device):
    if type(T) is list:
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


def collate_BERT(dlist, mode="min2min", fraction_missing=0.1):
        node_batch = [] 
        edge_batch = []
        min_code_batch = []
        for d in dlist:
            node_batch += [d.node_features]
            edge_batch += [d.edge_features]
            if mode == "min2min":
                min_code_batch += [d.min_dfs_code]
            elif mode == "rnd2rnd":
                rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                         d.z.numpy().tolist(), 
                                                                         np.argmax(d.edge_attr.numpy(), axis=1))
                min_code_batch += [rnd_code]
            else:
                raise ValueError("unknown config.training.mode %s"%mode)
        inputs, outputs = BERTize(min_code_batch, fraction_missing=fraction_missing)
        targets = nn.utils.rnn.pad_sequence(outputs, padding_value=-1)
        return inputs, node_batch, edge_batch, targets 
    
    
def collate_rnd2min(dlist):
        node_batch = [] 
        edge_batch = []
        min_code_batch = []
        rnd_code_batch = []
        for d in dlist:
            node_batch += [d.node_features]
            edge_batch += [d.edge_features]
            min_code_batch += [d.min_dfs_code]
            rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                     d.z.numpy().tolist(), 
                                                                     np.argmax(d.edge_attr.numpy(), axis=1))
            rnd_code_batch += [torch.tensor(rnd_code)]
        targets = nn.utils.rnn.pad_sequence(min_code_batch, padding_value=-1)
        return rnd_code_batch, node_batch, edge_batch, targets 