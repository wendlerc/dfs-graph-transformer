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
                 atom_encoder=None, bond_encoder=None, dropout=0.1):
        super().__init__()
        self.ninp = emb_dim * 5
        self.emb_dfs = PositionalEncoding(emb_dim, dropout=0, max_len=max_nodes)
        dfs_emb = self.emb_dfs(torch.zeros((max_nodes, 1, emb_dim)))
        dfs_emb = torch.squeeze(dfs_emb)
        self.register_buffer('dfs_emb', dfs_emb)
        self.emb_seq = PositionalEncoding(self.ninp, max_len=max_edges, dropout=dropout)
        
        if atom_encoder is None:
            self.emb_atom = AtomEncoder(emb_dim=emb_dim)
        else:
            self.emb_atom = atom_encoder
            
        if bond_encoder is None:
            self.emb_bond = BondEncoder(emb_dim=emb_dim)
        else:
            self.emb_bond = bond_encoder
            
        self.mixer = nn.Linear(self.ninp, self.ninp)
        
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.ninp, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward,
                                                                    dropout=dropout), nlayers)
    #for hugos dataloader:
    #C = batch of codes
    #N = batch of node features
    #E = batch of edge features
    def prepare_tokens(self, C, N, E, eos=None):
        """
        TODO: refactor
        """
        src = []
        if eos is not None:
            eos_idx = []
        else:
            eos_idx = None
            
        for code, n_feats, e_feats in zip(C, N, E):
            atom_emb, bond_emb = self.emb_atom(n_feats), self.emb_bond(e_feats)
            code_emb = torch.cat((self.dfs_emb[code[:, 0]], self.dfs_emb[code[:, 1]], 
                                  atom_emb[code[:, -3]], bond_emb[code[:, -2]], atom_emb[code[:,-1]]), dim=1)
            code_emb = self.mixer(code_emb) # allow the transformer to reshuffle features before the pe is added
            if eos is not None:
                eos_idx.append(code_emb.shape[0])
                code_emb = torch.cat((code_emb, eos.view(1,-1)), dim=0)
            src.append(code_emb)
        batch = nn.utils.rnn.pad_sequence(src, padding_value=-1000)
        src_key_padding_mask = (batch[:, :, 0] == -1000).T 
        batch = self.emb_seq(batch * math.sqrt(self.ninp))
        if eos is not None:
            eos_idx = torch.tensor(eos_idx, device=C[0].device, dtype=torch.long)
        return batch, eos_idx, src_key_padding_mask


    def forward(self, C, N, E, class_token=None, eos=None):
        batch, eos_idx, src_key_padding_mask = self.prepare_tokens(C, N, E, eos=eos)
        # batch is of shape (sequence length, batch, d_model)
        if class_token is None:
            self_attn = self.enc(batch, src_key_padding_mask=src_key_padding_mask)
        else:
            src_key_padding_mask = torch.cat((torch.zeros((batch.shape[1], 1), dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x n_seq
            self_attn = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0),
                                 src_key_padding_mask=src_key_padding_mask)
        
        if eos_idx is not None and class_token is not None:
            eos_idx += 1
            
        return self_attn, eos_idx, src_key_padding_mask


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
        self_attn, _, _ = self.encoder(C, N, E, class_token=self.cls_token)
        return self.fc_out(self_attn[0])
    
    
class DFSCodeSeq2SeqFC(nn.Module):
    def __init__(self, n_atoms, n_bonds, emb_dim, nhead, nlayers, dim_feedforward=2048, max_nodes=400, max_edges=400, 
                 atom_encoder=None, bond_encoder=None, dropout=0.1):
        super().__init__()
        self.ninp = emb_dim * 5
        self.encoder = DFSCodeEncoder(emb_dim, nhead, nlayers, dim_feedforward=dim_feedforward,
                                      max_nodes=max_nodes, max_edges=max_edges, 
                                      atom_encoder=atom_encoder, bond_encoder=bond_encoder, dropout=dropout)
        self.fc_dfs_idx1 = nn.Linear(2*self.ninp, max_nodes)
        self.fc_dfs_idx2 = nn.Linear(2*self.ninp, max_nodes)
        self.fc_atom1 = nn.Linear(2*self.ninp, n_atoms)
        self.fc_atom2 = nn.Linear(2*self.ninp, n_atoms)
        self.fc_bond = nn.Linear(2*self.ninp, n_bonds)
        self.fc_eos = nn.Linear(2*self.ninp, 1)
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.ninp), requires_grad=True)
        self.eos = nn.Parameter(torch.empty(self.ninp), requires_grad=True)
        nn.init.normal_(self.eos, mean=.0, std=.5)
        nn.init.normal_(self.cls_token, mean=.0, std=.5)
        

    def forward(self, C, N, E):
        self_attn, eos_idx, _ = self.encoder(C, N, E, class_token=self.cls_token, eos=self.eos) # seq x batch x feat
        batch_feats = self_attn[1:].permute(1,0,2) # batch x seq x feat
        batch_global = torch.unsqueeze(self_attn[0], 0).permute(1, 0, 2) # batch x 1 x feat
        batch_global = torch.tile(batch_global, [1, batch_feats.shape[1], 1]) # batch x seq x feat
        batch = torch.cat((batch_feats, batch_global), dim=2) # each of the prediction heads gets cls token as additional input
        dfs_idx1_logits = self.fc_dfs_idx1(batch).permute(1,0,2) #seq x batch x feat
        dfs_idx2_logits = self.fc_dfs_idx2(batch).permute(1,0,2)
        atom1_logits = self.fc_atom1(batch).permute(1,0,2)
        atom2_logits = self.fc_atom2(batch).permute(1,0,2)
        bond_logits = self.fc_bond(batch).permute(1,0,2)
        eos_logits = self.fc_eos(batch).permute(1,0,2)
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits, eos_logits
    
    
    def encode(self, C, N, E):
        """
        

        Parameters
        ----------
        C : TYPE
            DESCRIPTION.
        N : TYPE
            DESCRIPTION.
        E : TYPE
            DESCRIPTION.

        Returns
        -------
        self_attn : all hidden states
        cls token
        eos token
        """
        self_attn, eos_idx = self.encoder(C, N, E, class_token=self.cls_token, eos=self.eos) # seq x batch x feat
        idx = torch.arange(len(eos_idx), device=C[0].device)
        return self_attn, self_attn[0], self_attn[eos_idx, idx]




        
    
