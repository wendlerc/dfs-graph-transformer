#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:13:40 2021

@author: chrisw, hugop
"""

import torch
import torch.nn as nn
import math
from .utils import PositionalEncoding

class DFSCodeEncoder(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 emb_dim=120, nhead=12, nlayers=6, dim_feedforward=2048, 
                 max_nodes=400, max_edges=400, dropout=0.1):
        super().__init__()
        self.ninp = emb_dim * 5
        self.emb_dfs = PositionalEncoding(emb_dim, dropout=0, max_len=max_nodes)
        dfs_emb = self.emb_dfs(torch.zeros((max_nodes, 1, emb_dim)))
        dfs_emb = torch.squeeze(dfs_emb)
        self.register_buffer('dfs_emb', dfs_emb)
        self.emb_seq = PositionalEncoding(self.ninp, max_len=max_edges, dropout=dropout)
        self.emb_atom = atom_embedding
        self.emb_bond = bond_embedding
            
        self.mixer = nn.Linear(self.ninp, self.ninp)
        
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.ninp, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward,
                                                                    dropout=dropout), nlayers)

    def prepare_tokens(self, C, N, E):
        src = []
        for code, n_feats, e_feats in zip(C, N, E):
            atom_emb, bond_emb = self.emb_atom(n_feats), self.emb_bond(e_feats)
            code_emb = torch.cat((self.dfs_emb[code[:, 0]], self.dfs_emb[code[:, 1]], 
                                  atom_emb[code[:, -3]], bond_emb[code[:, -2]], atom_emb[code[:,-1]]), dim=1)
            code_emb = self.mixer(code_emb) # allow the transformer to reshuffle features before the pe is added
            src.append(code_emb)
        batch = nn.utils.rnn.pad_sequence(src, padding_value=-1000)
        src_key_padding_mask = (batch[:, :, 0] == -1000).T 
        batch = self.emb_seq(batch * math.sqrt(self.ninp))
        return batch, src_key_padding_mask


    def forward(self, C, N, E, class_token=None, eos=None):
        batch, src_key_padding_mask = self.prepare_tokens(C, N, E, eos=eos)
        # batch is of shape (sequence length, batch, d_model)
        if class_token is None:
            self_attn = self.enc(batch, src_key_padding_mask=src_key_padding_mask)
        else:
            src_key_padding_mask = torch.cat((torch.zeros((batch.shape[1], 1), dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x n_seq
            self_attn = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0),
                                 src_key_padding_mask=src_key_padding_mask)
        return self_attn, src_key_padding_mask