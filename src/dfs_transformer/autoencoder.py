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
from .transformers import PositionalEncoding, DFSCodeEncoder


class DFSCodeAutoencoder(nn.Module):
    def __init__(self, n_atoms, n_bonds, emb_dim, nhead, nlayers, n_memory_blocks=6, 
                 dim_feedforward=2048, max_nodes=400, max_edges=400, 
                 atom_encoder=None, bond_encoder=None, dropout=0.1):
        super().__init__()
        self.ninp = emb_dim * 5
        self.n_memory_blocks = n_memory_blocks # length of the bottleneck sequence
        self.encoder = DFSCodeEncoder(emb_dim, nhead, nlayers, dim_feedforward=dim_feedforward,
                                      max_nodes=max_nodes, max_edges=max_edges, 
                                      atom_encoder=atom_encoder, bond_encoder=bond_encoder, dropout=dropout)
        self.bottleneck = nn.MultiheadAttention(self.ninp, nhead)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.ninp, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward,
                                                                    dropout=dropout), nlayers)
        self.fc_dfs_idx1 = nn.Linear(self.ninp, max_nodes)
        self.fc_dfs_idx2 = nn.Linear(self.ninp, max_nodes)
        self.fc_atom1 = nn.Linear(self.ninp, n_atoms)
        self.fc_atom2 = nn.Linear(self.ninp, n_atoms)
        self.fc_bond = nn.Linear(self.ninp, n_bonds)
        
        self.memory_query = nn.Parameter(torch.empty(n_memory_blocks, 1, self.ninp), requires_grad=True)
        pos_emb = PositionalEncoding(self.ninp, dropout=0, max_len=max_edges)(torch.zeros((max_edges, 1, self.ninp)))
        pos_emb = torch.squeeze(pos_emb)
        self.register_buffer('output_query', pos_emb)
        nn.init.normal_(self.memory_query, mean=.0, std=.5)
        

    def forward(self, C, N, E):
        self_attn, _, src_key_padding_mask = self.encoder(C, N, E) # seq x batch x feat
        query = torch.tile(self.memory_query, [1, self_attn.shape[1], 1]) # n_memory_blocks x batch x feat
        attn_output, attn_output_weights = self.bottleneck(query, self_attn, self_attn,
                                                           key_padding_mask=src_key_padding_mask) # n_memory_blocks x batch x feat
        memory = attn_output
        tgt = torch.tile(self.output_query[:self_attn.shape[0]].unsqueeze(1), [1, self_attn.shape[1], 1]) # seq x batch x feat
        cross_attn = self.decoder(tgt, memory, tgt_key_padding_mask=src_key_padding_mask) # seq x batch x feat
        batch = cross_attn.permute(1,0,2) # batch x seq x feat
        dfs_idx1_logits = self.fc_dfs_idx1(batch).permute(1,0,2) #seq x batch x feat
        dfs_idx2_logits = self.fc_dfs_idx2(batch).permute(1,0,2)
        atom1_logits = self.fc_atom1(batch).permute(1,0,2)
        atom2_logits = self.fc_atom2(batch).permute(1,0,2)
        bond_logits = self.fc_bond(batch).permute(1,0,2)
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits
    
    
    def encode(self, C, N, E):
  
        self_attn, _, src_key_padding_mask = self.encoder(C, N, E) # seq x batch x feat
        query = torch.tile(self.memory_query, [1, self_attn.shape[1], 1]) # n_memory_blocks x batch x feat
        memory, _ = self.bottleneck(query, self_attn, self_attn,
                                    key_padding_mask=src_key_padding_mask) # n_memory_blocks x batch x feat
        return memory
    
    
class DFSCodeSeq2SeqDecoder(nn.Module):
    def __init__(self, n_atoms, n_bonds, emb_dim, nhead, nlayers, n_class_tokens=2, 
                 dim_feedforward=2048, max_nodes=400, max_edges=400, 
                 atom_encoder=None, bond_encoder=None, dropout=0.1):
        super().__init__()
        self.ninp = emb_dim * 5
        self.n_class_tokens = n_class_tokens # length of the bottleneck sequence
        self.encoder = DFSCodeEncoder(emb_dim, nhead, nlayers, dim_feedforward=dim_feedforward,
                                      max_nodes=max_nodes, max_edges=max_edges, 
                                      atom_encoder=atom_encoder, bond_encoder=bond_encoder, dropout=dropout)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.ninp, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward,
                                                                    dropout=dropout), nlayers)
        self.fc_dfs_idx1 = nn.Linear(self.ninp + n_class_tokens*self.ninp, max_nodes)
        self.fc_dfs_idx2 = nn.Linear(self.ninp + n_class_tokens*self.ninp, max_nodes)
        self.fc_atom1 = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_atoms)
        self.fc_atom2 = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_atoms)
        self.fc_bond = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_bonds)
        
        self.class_tokens = nn.Parameter(torch.empty(n_class_tokens, 1, self.ninp), requires_grad=True)
        pos_emb = PositionalEncoding(self.ninp, dropout=0, max_len=max_edges)(torch.zeros((max_edges, 1, self.ninp)))
        pos_emb = torch.squeeze(pos_emb)
        self.register_buffer('output_query', pos_emb)
        nn.init.normal_(self.class_tokens, mean=.0, std=.5)
        

    def forward(self, C, N, E):
        self_attn, _, src_key_padding_mask = self.encoder(C, N, E) # seq x batch x feat
        memory = self_attn
        tgt = torch.tile(self.output_query[:self_attn.shape[0]].unsqueeze(1), [1, self_attn.shape[1], 1]) # seq x batch x feat
        tgt = torch.cat((self.class_tokens.expand(-1, tgt.shape[1], -1), tgt), dim=0)
        zeros = torch.zeros((tgt.shape[1], self.n_class_tokens), dtype=torch.bool, device=tgt.device)
        tgt_key_padding_mask = torch.cat((zeros, src_key_padding_mask), dim=1) # batch x (ncls + seq) 
        cross_attn = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask) # (ncls + seq) x batch x feat
        
        batch_feats = cross_attn[self.n_class_tokens:].permute(1,0,2) # batch x seq x feat
        if self.n_class_tokens == 1:
            batch_global = torch.unsqueeze(cross_attn[:self.n_class_tokens], 0).permute(1, 0, 2) # batch x ncls x feat
        else:
            batch_global = cross_attn[:self.n_class_tokens].permute(1, 0, 2) # batch x ncls x feat
        batch_global = torch.reshape(batch_global, (-1, 1, self.n_class_tokens * self.ninp))
        batch_global = batch_global.expand(-1, batch_feats.shape[1], -1) # batch x seq x ncls * feats
        batch = torch.cat((batch_feats, batch_global), dim=2) # each of the prediction heads gets cls token as additional input
        dfs_idx1_logits = self.fc_dfs_idx1(batch).permute(1,0,2) #seq x batch x feat
        dfs_idx2_logits = self.fc_dfs_idx2(batch).permute(1,0,2)
        atom1_logits = self.fc_atom1(batch).permute(1,0,2)
        atom2_logits = self.fc_atom2(batch).permute(1,0,2)
        bond_logits = self.fc_bond(batch).permute(1,0,2)
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits
    
    
    def encode(self, C, N, E):
        self_attn, _, src_key_padding_mask = self.encoder(C, N, E) # seq x batch x feat
        memory = self_attn
        tgt = torch.tile(self.output_query[:self_attn.shape[0]].unsqueeze(1), [1, self_attn.shape[1], 1]) # seq x batch x feat
        tgt = torch.cat((self.class_tokens.expand(-1, tgt.shape[1], -1), tgt), dim=0)
        zeros = torch.zeros((tgt.shape[1], self.n_class_tokens), dtype=torch.bool, device=tgt.device)
        tgt_key_padding_mask = torch.cat((zeros, src_key_padding_mask), dim=1) # batch x (ncls + seq) 
        cross_attn = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask) # (ncls + seq) x batch x feat
        
        return cross_attn[:self.n_class_tokens].permute(1, 0, 2).view(-1, self.ninp*self.n_class_tokens)
