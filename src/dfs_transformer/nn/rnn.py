#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:00:27 2021

@author: chrisw
"""
import torch
import torch.nn as nn
import math
from .utils import PositionalEncoding
from einops import rearrange
from torchdistill.core.forward_hook import ForwardHookManager


class DFSCodeEncoderLSTM(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 emb_dim=120, nlayers=6, bidirectional=True, 
                 max_nodes=250, max_edges=500, dropout=0.0, missing_value=None,
                 rescale_flag=True, **kwargs):
        super().__init__()
        self.ninp = emb_dim * 5
        self.emb_dfs = PositionalEncoding(emb_dim, dropout=0, max_len=max_nodes)
        dfs_emb = self.emb_dfs(torch.zeros((max_nodes, 1, emb_dim)))
        dfs_emb = torch.squeeze(dfs_emb)
        self.register_buffer('dfs_emb', dfs_emb)
        self.emb_atom = atom_embedding
        self.emb_bond = bond_embedding
            
        self.hidden_size = self.ninp // 2 if bidirectional else self.ninp
        self.enc = nn.LSTM(input_size=self.ninp, hidden_size=self.hidden_size, 
                           num_layers=nlayers, dropout=dropout, 
                           bidirectional=bidirectional)
        self.missing_value = missing_value
        if missing_value is not None:
            self.missing_token = nn.Parameter(torch.empty(1, self.ninp), requires_grad=True)
            nn.init.normal_(self.missing_token, mean=.0, std=.5)
            
        
    def prepare_tokens_BERT(self, C, N, E):
        src = []
        for code, n_feats, e_feats in zip(C, N, E):
            mask = code[:, 0] != self.missing_value
            atom_emb, bond_emb = self.emb_atom(n_feats), self.emb_bond(e_feats)
            # embed all the non-missing values
            code_input = torch.cat((self.dfs_emb[code[mask][:, 0]], 
                                    self.dfs_emb[code[mask][:, 1]], 
                                    atom_emb[code[mask][:, -3]], 
                                    bond_emb[code[mask][:, -2]], 
                                    atom_emb[code[mask][:,-1]]), dim=1)
            # create missing tokens
            code_missing = self.missing_token.expand(torch.sum(~mask),-1)
            code_emb = torch.ones((code.shape[0], self.ninp), device=code_input.device)
            code_emb[mask] *= code_input
            code_emb[~mask] *= code_missing
            src.append(code_emb)
        batch = nn.utils.rnn.pad_sequence(src, padding_value=0)
        return batch

    def prepare_tokens(self, C, N, E):
        src = []
        for code, n_feats, e_feats in zip(C, N, E):
            atom_emb, bond_emb = self.emb_atom(n_feats), self.emb_bond(e_feats)
            code_emb = torch.cat((self.dfs_emb[code[:, 0]], self.dfs_emb[code[:, 1]], 
                                  atom_emb[code[:, -3]], bond_emb[code[:, -2]], atom_emb[code[:,-1]]), dim=1)
            src.append(code_emb)
        batch = nn.utils.rnn.pad_sequence(src, padding_value=0)
        return batch

    def forward(self, C, N, E, class_token=None):
        if self.missing_value is None:
            batch = self.prepare_tokens(C, N, E)
        else:
            batch = self.prepare_tokens_BERT(C, N, E)
        # batch is of shape (sequence length, batch, d_model)
        if class_token is None:
            output, _ = self.enc(batch)
        else:
            output, _ = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0))
        return output, None