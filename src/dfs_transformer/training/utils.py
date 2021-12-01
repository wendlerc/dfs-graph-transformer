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


def BERTize(codes, fraction_missing=0.15, fraction_mask=0.8, fraction_rand=0.1):
    """
    Training the language model in BERT is done by predicting 15% of the tokens in the input, that were randomly picked. 
    These tokens are pre-processed as follows — 80% are replaced with a “[MASK]” token, 10% with a random word, and 10% 
    use the original word. 
    
    to get a random word we use the following strategy: we copy an random entry to that position 
    TODO: in the future one could also in addition implement a flip of the edge direction with a 50% chance
    
    returns preprocessed input sequences, target sequences and a mask indicating which inputs are part of the 15% masked, orig, rnd
    """
    fraction_orig = 1. - fraction_mask - fraction_rand
    fo = fraction_orig
    fm = fraction_mask
    fr = fraction_rand
    inputs = []
    targets = []
    masks = []
    for code in codes:
        n = len(code)
        perm = np.random.permutation(n)
        perm2 = np.random.permutation(n)
        mask = torch.zeros(n, dtype=bool)
        mask[perm[:int(fraction_missing*n)]] = True
        delete_target_idx = perm[int(fraction_missing*n):]
        delete_input_idx = perm[:int(fraction_missing*fm*n)]
        input_rnd_idx = perm[int(fraction_missing*fm*n):int(fraction_missing*(fm+fr)*n)]
        target_rnd_idx = perm2[int(fraction_missing*fm*n):int(fraction_missing*(fm+fr)*n)] 
        inp = code.clone()
        target = code.clone()
        inp[input_rnd_idx] = target[target_rnd_idx]
        target[delete_target_idx] = -1
        inp[delete_input_idx] = -1
        #print(inp)
        #print(target)
        inputs += [inp]
        targets += [target]
        masks += [mask]
    return inputs, targets, masks


def collate_BERT(dlist, mode="min2min", fraction_missing=0.1, use_loops=False):
        node_batch = [] 
        edge_batch = []
        code_batch = []
        dfs_codes = defaultdict(list)
        if "properties" in dlist[0].keys():
            prop_batch = defaultdict(list)
        if use_loops:
            loop = torch.tensor(bond_features(None)).unsqueeze(0)
        for d in dlist:
            
            edge_features = d.edge_features
            
            if mode == "min2min":
                code = d.min_dfs_code
                index = d.min_dfs_index
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
                
            node_batch += [d.node_features]
            edge_batch += [edge_features]
            code_batch += [code]
            if "properties" in dlist[0].keys():
                for name, prop in d.properties.items():
                    prop_batch[name.replace('_', '.')] += [prop]
                    
        inputs, outputs, masks = BERTize(code_batch, fraction_missing=fraction_missing)
        
        for bertmask, inp, nfeats, efeats in zip(masks, inputs, node_batch, edge_batch):
            #print(inp[bertmask])
            #print('----')
            #print(output[bertmask])
            mask = ~bertmask # the mask returned by BERT indicates which inputs will be masked away
            dfs_codes['dfs_from'] += [inp[:, 0]]
            dfs_codes['dfs_to'] += [inp[:, 1]]
            
            atm_from_feats = torch.ones((inp.shape[0], nfeats.shape[1]))
            atm_from_feats[mask] *= nfeats[inp[mask][:, -3]]
            atm_from_feats[~mask] *= -1
            
            atm_to_feats = torch.ones((inp.shape[0], nfeats.shape[1]))
            atm_to_feats[mask] *= nfeats[inp[mask][:, -1]]
            atm_to_feats[~mask] *= -1
            
            bnd_feats = torch.ones((inp.shape[0], efeats.shape[1]))
            bnd_feats[mask] *= efeats[inp[mask][:, -2]]
            bnd_feats[~mask] *= -1
            
            
            dfs_codes['atm_from'] += [atm_from_feats]
            dfs_codes['atm_to'] += [atm_to_feats]
            dfs_codes['bnd'] += [bnd_feats]
            
        dfs_codes = {key: nn.utils.rnn.pad_sequence(values, padding_value=-1000).clone()
                     for key, values in dfs_codes.items()}
        
        targets = nn.utils.rnn.pad_sequence(outputs, padding_value=-1).clone()
        if "properties" in dlist[0].keys():
            prop_batch = {name: torch.tensor(plist).clone() for name, plist in prop_batch.items()}
            return dfs_codes, targets, prop_batch
        return dfs_codes, targets 
    
    
def collate_rnd2min(dlist, use_loops=False):
        node_batch = [] 
        edge_batch = []
        min_code_batch = []
        rnd_code_batch = []
        if "properties" in dlist[0].keys:
            prop_batch = defaultdict(list)
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
            if "properties" in dlist[0].keys:
                for name, prop in d.properties.items():
                    prop_batch[name] += [prop]
        targets = nn.utils.rnn.pad_sequence(min_code_batch, padding_value=-1)
        if "properties" in dlist[0].keys:
            prop_batch = {name: torch.tensor(deepcopy(plist)) for name, plist in prop_batch.items()}
            return rnd_code_batch, node_batch, edge_batch, targets, prop_batch
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