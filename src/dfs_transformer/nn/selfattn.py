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
from einops import rearrange
from torchdistill.core.forward_hook import ForwardHookManager


class TransformerPlusHeads(nn.Module):
    def __init__(self, encoder, head_specs):
        """head_specs a dict specifying the heads"""
        super(TransformerPlusHeads, self).__init__()
        self.encoder = encoder
        self.ninp = self.encoder.ninp 
        self.encoder.return_features = True
        self.head_specs = head_specs
        self.head_dict = nn.ModuleDict({name.replace(".", "_"): nn.Linear(self.ninp* self.encoder.n_class_tokens, n_output) for name, n_output in head_specs.items()})
            
    def forward(self, C, N, E):
        dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits, features = self.encoder(C, N, E)
        property_predictions = {name.replace("_", "."): head(features) for name, head in self.head_dict.items()}
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits, property_predictions
    

class DFSCodeEncoder(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 emb_dim=120, nhead=12, nlayers=6, dim_feedforward=2048, 
                 max_nodes=250, max_edges=500, dropout=0.1, missing_value=None, **kwargs):
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
                 missing_value=None, return_features=False, **kwargs):
        super().__init__()
        self.nlayers = nlayers
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
        self.return_features = return_features
        
        
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
        if self.return_features:
            features = self_attn[:self.n_class_tokens]
            features = rearrange(features, 'd0 d1 d2 -> d1 (d0 d2)')
            return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits, features
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits
    
    def encode(self, C, N, E, method="cls"):
        if method == "cls3":
            forward_hook_manager = ForwardHookManager(C[0].device)
            for lid in range(self.nlayers): 
                forward_hook_manager.add_hook(self.encoder, 'enc.layers.%d'%lid, requires_input=False, requires_output=True)
        ncls = self.n_class_tokens
        self_attn, _ = self.encoder(C, N, E, class_token=self.cls_token) # seq x batch x feat
        if method == "cls":
            #features = self_attn[:ncls].permute(1, 0, 2).reshape(self_attn.shape[1], -1)
            features = self_attn[:ncls]
            features = rearrange(features, 'd0 d1 d2 -> d1 (d0 d2)')
        elif method == "cls3":
            io_dict = forward_hook_manager.pop_io_dict()
            fs = []
            for lid in range(self.nlayers):
                out = io_dict['enc.layers.%d'%lid]['output']
                f = out[:ncls]
                f = rearrange(f, 'd0 d1 d2 -> d1 (d0 d2)')
                fs += [f]
            features =  torch.cat(fs, dim=1)
        elif method == "sum":
            features = torch.sum(self_attn[ncls:], dim=0)
        elif method == "mean":
            features = torch.mean(self_attn[ncls:], dim=0)
        elif method == "max":
            features = torch.max(self_attn[ncls:], dim=0)[0]
        elif method == "cls-mmm":
            fmin = torch.min(self_attn, dim=0)[0]
            fmean = torch.mean(self_attn, dim=0)
            fmax = torch.max(self_attn, dim=0)[0]
            features1 = torch.cat((fmin, fmean, fmax), dim=1)
            features2 = self_attn[:ncls]
            features2 = rearrange(features2, 'd0 d1 d2 -> d1 (d0 d2)')
            features = torch.cat((features1, features2), dim=1)
        elif method == "min-mean-max":
            fmin = torch.min(self_attn, dim=0)[0]
            fmean = torch.mean(self_attn, dim=0)
            fmax = torch.max(self_attn, dim=0)[0]
            features = torch.cat((fmin, fmean, fmax), dim=1)
        elif method == "min-mean-max-std":
            fmin = torch.min(self_attn, dim=0)[0]
            fmean = torch.mean(self_attn, dim=0)
            fmax = torch.max(self_attn, dim=0)[0]
            fstd = torch.std(self_attn, dim=0)
            features = torch.cat((fmin, fmean, fmax, fstd), dim=1)
        elif method == "max-of-cls":
            features = torch.max(self_attn[:ncls], dim=0)[0]
        elif method == "mean-of-cls":
            features = torch.mean(self_attn[:ncls], dim=0)
        elif method == "max-mean-of-cls":
            features = torch.cat((torch.max(self_attn[:ncls], dim=0)[0], torch.mean(self_attn[:ncls], dim=0)), dim=1)
        elif int(method) < ncls:
            features = self_attn[int(method)]
        else:
            raise ValueError("unsupported method")
        return features.view(self_attn.shape[1], -1)
    
    def get_n_encoding(self, method="cls"):
        ncls = self.n_class_tokens
        if method == "cls":
            return ncls * self.ninp
        elif method == "cls3":
            return self.nlayers * ncls * self.ninp
        elif method in {"sum", "mean", "max", "max-of-cls", "mean-of-cls"}:
            return self.ninp
        elif method == "cls-mmm":
            return ncls*self.ninp + 3*self.ninp
        elif method == "min-mean-max":
            return 3*self.ninp
        elif method == "min-mean-max-std":
            return 4*self.ninp
        elif method == "max-mean-of-cls":
            return 2*self.ninp
        elif int(method) < ncls:
            return self.ninp
        else:
            raise ValueError("unsupported method")
    
    
class DFSCodeSeq2SeqFCFeatures(nn.Module):
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
        self.fc_feats = nn.Linear(self.ninp + n_class_tokens*self.ninp, 2*n_node_features+n_edge_features)
        
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
        feature_logits = self.fc_feats(batch)
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits, \
            feature_logits
    
    def encode(self, C, N, E, method="cls"):
        ncls = self.n_class_tokens
        self_attn, _ = self.encoder(C, N, E, class_token=self.cls_token) # seq x batch x feat
        if method == "cls":
            #features = self_attn[:ncls].permute(1, 0, 2).reshape(self_attn.shape[1], -1)
            features = self_attn[:ncls]
            features = rearrange(features, 'd0 d1 d2 -> d1 (d0 d2)')
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
            features = torch.cat((fcls, fmean, fmax), dim=1)
        elif method == "min-mean-max":
            fmin = torch.min(self_attn, dim=0)[0]
            fmean = torch.mean(self_attn, dim=0)
            fmax = torch.max(self_attn, dim=0)[0]
            features = torch.cat((fmin, fmean, fmax), dim=1)
        elif method == "min-mean-max-std":
            fmin = torch.min(self_attn, dim=0)[0]
            fmean = torch.mean(self_attn, dim=0)
            fmax = torch.max(self_attn, dim=0)[0]
            fstd = torch.std(self_attn, dim=0)
            features = torch.cat((fmin, fmean, fmax, fstd), dim=1)
        elif method == "max-of-cls":
            features = torch.max(self_attn[:ncls], dim=0)[0]
        elif method == "mean-of-cls":
            features = torch.mean(self_attn[:ncls], dim=0)
        elif method == "max-mean-of-cls":
            features = torch.cat((torch.max(self_attn[:ncls], dim=0)[0], torch.mean(self_attn[:ncls], dim=0)), dim=1)
        elif int(method) < ncls:
            features = self_attn[int(method)]
        else:
            raise ValueError("unsupported method")
        return features.view(self_attn.shape[1], -1)
    
    def get_n_encoding(self, method="cls"):
        ncls = self.n_class_tokens
        if method == "cls":
            return ncls * self.ninp
        elif method in {"sum", "mean", "max", "max-of-cls", "mean-of-cls"}:
            return self.ninp
        elif method == "cls-mean-max":
            return ncls * self.ninp + 2*self.ninp
        elif method == "min-mean-max":
            return 3*self.ninp
        elif method == "min-mean-max-std":
            return 4*self.ninp
        elif method == "max-mean-of-cls":
            return 2*self.ninp
        elif int(method) < ncls:
            return self.ninp
        else:
            raise ValueError("unsupported method")
            

        

