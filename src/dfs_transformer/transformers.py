#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:13:40 2021

@author: chrisw, hugop
"""

import torch
import torch.nn as nn
import math
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        

class DFSCodeEncoder(nn.Module):
    def __init__(self, emb_dim, nhead, nlayers, dim_feedforward=2048, max_nodes=400, max_edges=400, 
                 atom_encoder=None, bond_encoder=None):
        super().__init__()
        self.ninp = emb_dim * 5
        self.emb_dfs = PositionalEncoding(emb_dim, dropout=0, max_len=max_nodes)
        dfs_emb = self.emb_dfs(torch.zeros((max_nodes, 1, emb_dim)))
        dfs_emb = torch.squeeze(dfs_emb)
        self.register_buffer('dfs_emb', dfs_emb)
        self.emb_seq = PositionalEncoding(self.ninp, max_len=max_edges)
        
        if atom_encoder is None:
            self.emb_atom = AtomEncoder(emb_dim=emb_dim)
        else:
            self.emb_atom = atom_encoder
            
        if bond_encoder is None:
            self.emb_bond = BondEncoder(emb_dim=emb_dim)
        else:
            self.emb_bond = bond_encoder
        
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.ninp, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward), nlayers)
    #for hugos dataloader:
    #C = batch of codes
    #N = batch of node features
    #E = batch of edge features
    def prepare_tokens(self, C, N, E):
        src = []
        for code, n_feats, e_feats in zip(C, N, E):
            atom_emb, bond_emb = self.emb_atom(n_feats), self.emb_bond(e_feats)
            src.append(torch.cat((self.dfs_emb[code[:, 0]], self.dfs_emb[code[:, 1]], 
                                  atom_emb[code[:, -3]], bond_emb[code[:, -2]], atom_emb[code[:,-1]]), dim=1))
        batch = self.emb_seq(nn.utils.rnn.pad_sequence(src) * math.sqrt(self.ninp))
        return batch

    def forward(self, C, N, E, class_token=None):
        if class_token is None:
            self_attn = self.enc(self.prepare_tokens(C, N, E))
        else:
            batch = self.prepare_tokens(C, N, E)
            self_attn = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0))
        return self_attn


class DFSCodeClassifier(nn.Module):
    def __init__(self, n_classes, emb_dim, nhead, nlayers, dim_feedforward=2048, max_nodes=400, max_edges=400, 
                 atom_encoder=None, bond_encoder=None):
        super().__init__()
        self.ninp = emb_dim * 5
        self.encoder = DFSCodeEncoder(emb_dim, nhead, nlayers, dim_feedforward=dim_feedforward,
                                      max_nodes=max_nodes, max_edges=max_edges, 
                                      atom_encoder=atom_encoder, bond_encoder=bond_encoder)
        self.fc_out = nn.Linear(self.ninp, n_classes)
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.ninp), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=.0, std=.5)
        

    def forward(self, C, N, E):
        self_attn = self.encoder(C, N, E, class_token=self.cls_token)
        return self.fc_out(self_attn[0])
