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
        self.emb_seq = PositionalEncoding(self.ninp, max_len=max_edges, dropout=dropout)
        self.mixer = nn.Linear(self.ninp, self.ninp)
        
        self.hidden_size = self.ninp // 2 if bidirectional else self.ninp
        self.enc = nn.LSTM(input_size=self.ninp, hidden_size=self.hidden_size, 
                           num_layers=nlayers, dropout=dropout, 
                           bidirectional=bidirectional)
        self.missing_value = missing_value
        if missing_value is not None:
            self.missing_token = nn.Parameter(torch.empty(1, self.ninp), requires_grad=True)
            nn.init.normal_(self.missing_token, mean=.0, std=.5)
        self.rescale_factor = math.sqrt(self.ninp) if rescale_flag else 1.
            
    
    def forward(self, dfs_codes, class_token=None):
        if self.missing_value is not None:
            src_key_padding_mask = (dfs_codes['dfs_from'] == -1000).T
            n_seq = dfs_codes['dfs_from'].shape[0]
            n_batch = dfs_codes['dfs_from'].shape[1]
            mask = dfs_codes['dfs_from'] >= 0
            missing = dfs_codes['dfs_from'] == -1
            atm_from = dfs_codes['atm_from'][mask]
            atm1 = self.emb_atom(atm_from)
            atm_to = dfs_codes['atm_to'][mask]
            atm2 = self.emb_atom(atm_to)
            bnd_inp = dfs_codes['bnd'][mask]
            bnd = self.emb_bond(bnd_inp)
            code_input = torch.cat((self.dfs_emb[dfs_codes['dfs_from'][mask]],
                                    self.dfs_emb[dfs_codes['dfs_to'][mask]],
                                    atm1, atm2, bnd), dim=1)
            
            code_emb = torch.ones((n_seq, 
                                   n_batch, 
                                   self.ninp), device=dfs_codes['dfs_from'].device)
            code_emb[mask] *= code_input
            code_emb[missing] *= self.missing_token
            code_emb[~src_key_padding_mask.T] = self.mixer(code_emb[~src_key_padding_mask.T])
            code_emb[src_key_padding_mask.T] = 0 # for transformer we don't need this cuz it takes the mask
            batch = self.emb_seq(code_emb * self.rescale_factor)
        else:
            raise NotImplemented('not implemented yet')
            
            
        # batch is of shape (sequence length, batch, d_model)
        if class_token is None:
            self_attn, _ = self.enc(batch)
        else:
            src_key_padding_mask = torch.cat((torch.zeros((batch.shape[1], class_token.shape[0]), dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x n_seq
            self_attn, _ = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0))
        return self_attn, src_key_padding_mask
    