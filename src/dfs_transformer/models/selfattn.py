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
                 max_nodes=250, max_edges=500, dropout=0.1, missing_value=None):
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
            # put everything into the right place
            code_emb = self.mixer(code_emb) # allow the transformer to reshuffle features before the pe is added
            src.append(code_emb)
        batch = nn.utils.rnn.pad_sequence(src, padding_value=-1000)
        src_key_padding_mask = (batch[:, :, 0] == -1000).T 
        batch = self.emb_seq(batch * math.sqrt(self.ninp))
        return batch, src_key_padding_mask

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

    def forward(self, C, N, E, class_token=None):
        if self.missing_value is None:
            batch, src_key_padding_mask = self.prepare_tokens(C, N, E)
        else:
            batch, src_key_padding_mask = self.prepare_tokens_BERT(C, N, E)
        # batch is of shape (sequence length, batch, d_model)
        if class_token is None:
            self_attn = self.enc(batch, src_key_padding_mask=src_key_padding_mask)
        else:
            src_key_padding_mask = torch.cat((torch.zeros((batch.shape[1], class_token.shape[0]), dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x n_seq
            self_attn = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0),
                                 src_key_padding_mask=src_key_padding_mask)
        return self_attn, src_key_padding_mask
    

class DFSCodeSeq2SeqFC(nn.Module):
    def __init__(self, n_node_features, n_edge_features, 
                 n_atoms, n_bonds, emb_dim=120, nhead=12, 
                 nlayers=6, n_class_tokens=1, dim_feedforward=2048, 
                 max_nodes=250, max_edges=500, dropout=0.1, 
                 missing_value=None, **kwargs):
        super().__init__()
        self.ninp = emb_dim * 5
        self.n_class_tokens = n_class_tokens
        atom_embedding = nn.Linear(n_node_features, emb_dim)
        bond_embedding = nn.Linear(n_edge_features, emb_dim)
        self.encoder = DFSCodeEncoder(atom_embedding, bond_embedding, 
                                      emb_dim=emb_dim, nhead=nhead, nlayers=nlayers, 
                                      dim_feedforward=dim_feedforward,
                                      max_nodes=max_nodes, max_edges=max_edges, 
                                      dropout=dropout, missing_value=missing_value)
        self.fc_dfs_idx1 = nn.Linear(self.ninp + n_class_tokens*self.ninp, max_nodes)
        self.fc_dfs_idx2 = nn.Linear(self.ninp + n_class_tokens*self.ninp, max_nodes)
        self.fc_atom1 = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_atoms)
        self.fc_atom2 = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_atoms)
        self.fc_bond = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_bonds)
        
        self.cls_token = nn.Parameter(torch.empty(n_class_tokens, 1, self.ninp), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=.0, std=.5)
        
    def forward(self, C, N, E):
        self_attn, _ = self.encoder(C, N, E, class_token=self.cls_token) # seq x batch x feat
        
        batch_feats = self_attn[self.n_class_tokens:] 
        batch_global = self_attn[:self.n_class_tokens]
        batch_global = batch_global.view(1, -1, self.n_class_tokens * self.ninp)
        batch_global = batch_global.expand(batch_feats.shape[0], -1, -1) # batch x seq x ncls * feats
        batch = torch.cat((batch_feats, batch_global), dim=2) # each of the prediction heads gets cls token as additional input
        
        dfs_idx1_logits = self.fc_dfs_idx1(batch) #seq x batch x feat
        dfs_idx2_logits = self.fc_dfs_idx2(batch)
        atom1_logits = self.fc_atom1(batch)
        atom2_logits = self.fc_atom2(batch)
        bond_logits = self.fc_bond(batch)
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits
    
    def encode(self, C, N, E, method="cls"):
        ncls = self.n_class_tokens
        self_attn, _ = self.encoder(C, N, E, class_token=self.cls_token) # seq x batch x feat
        if method == "cls":
            features = self_attn[:ncls]
        elif method == "sum":
            features = torch.sum(self_attn[ncls:], dim=0)
        elif method == "mean":
            features = torch.mean(self_attn[ncls:], dim=0)
        elif method == "max":
            features = torch.max(self_attn[ncls:], dim=0)[0]
        elif method == "cls-mean-max":
            fcls = self_attn[0]
            fmean = torch.mean(self_attn[ncls:], dim=0)
            fmax = torch.max(self_attn[ncls:], dim=0)[0]
            features = torch.cat((fcls, fmean, fmax), axis=1)
        else:
            raise ValueError("unsupported method")
        return features.view(self_attn.shape[1], -1)

