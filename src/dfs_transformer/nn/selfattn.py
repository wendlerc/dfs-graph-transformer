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
from .rnn import DFSCodeEncoderLSTM
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
            
    def forward(self, dfs_code):
        dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits, features = self.encoder(dfs_code)
        property_predictions = {name.replace("_", "."): head(features) for name, head in self.head_dict.items()}
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits, property_predictions
    

class DFSCodeEncoder(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 emb_dim=120, nhead=12, nlayers=6, dim_feedforward=2048, 
                 activation = 'gelu',
                 max_nodes=250, max_edges=500, dropout=0.1, missing_value=None,
                 rescale_flag=True, **kwargs):
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
                                                                    dropout=dropout, 
                                                                    activation=activation), nlayers)
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
            batch = self.emb_seq(code_emb * self.rescale_factor)
        else:
            raise NotImplemented('not implemented yet')
            
            
        # batch is of shape (sequence length, batch, d_model)
        if class_token is None:
            self_attn = self.enc(batch, src_key_padding_mask=src_key_padding_mask)
        else:
            src_key_padding_mask = torch.cat((torch.zeros((batch.shape[1], class_token.shape[0]), dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x n_seq
            self_attn = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0),
                                 src_key_padding_mask=src_key_padding_mask)
        return self_attn, src_key_padding_mask
    
    
class DFSCodeEncoderEntryBERT(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 emb_dim=120, nhead=12, nlayers=6, dim_feedforward=2048, 
                 activation = 'gelu',
                 max_nodes=250, max_edges=500, dropout=0.1, missing_value=None,
                 rescale_flag=True, **kwargs):
        super().__init__()
        self.ninp = emb_dim * 5
        self.emb_dim = emb_dim
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
                                                                    dropout=dropout, 
                                                                    activation=activation), nlayers)
        self.missing_value = missing_value
        if missing_value is not None:
            self.missing_dfs1 = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
            self.missing_dfs2 = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
            self.missing_atm1 = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
            self.missing_atm2 = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
            self.missing_bnd = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
            nn.init.normal_(self.missing_dfs1, mean=.0, std=.5)
            nn.init.normal_(self.missing_dfs2, mean=.0, std=.5)
            nn.init.normal_(self.missing_atm1, mean=.0, std=.5)
            nn.init.normal_(self.missing_atm2, mean=.0, std=.5)
            nn.init.normal_(self.missing_bnd, mean=.0, std=.5)
            
        self.rescale_factor = math.sqrt(self.ninp) if rescale_flag else 1.

    def forward(self, dfs_codes, class_token=None):
        if self.missing_value is not None:
            src_key_padding_mask = (dfs_codes['dfs_from'] == -1000).T #THIS MIGHT CAUSE PROBLEMS BUT I THINK IT SHOULD NOT
            n_seq = dfs_codes['dfs_from'].shape[0]
            n_batch = dfs_codes['dfs_from'].shape[1]
            mask_dfs1 = dfs_codes['dfs_from'] >= 0
            mask_dfs2 = dfs_codes['dfs_to'] >= 0
            mask_atm1 = torch.all(dfs_codes['atm_from'] >= 0, dim=2)
            mask_atm2 = torch.all(dfs_codes['atm_to'] >= 0, dim=2)
            mask_bnd = torch.all(dfs_codes['bnd'] >= 0, dim=2)
            missing_dfs1 = ~mask_dfs1 
            missing_dfs2 = ~mask_dfs2 
            missing_atm1 = ~mask_atm1 
            missing_atm2 = ~mask_atm2 
            missing_bnd =  ~mask_bnd 
            atm_from = dfs_codes['atm_from'][mask_atm1]
            atm1 = self.emb_atom(atm_from)
            atm_to = dfs_codes['atm_to'][mask_atm2]
            atm2 = self.emb_atom(atm_to)
            bnd_inp = dfs_codes['bnd'][mask_bnd]
            bnd = self.emb_bond(bnd_inp)
            dfs1 = self.dfs_emb[dfs_codes['dfs_from'][mask_dfs1]]
            dfs2 = self.dfs_emb[dfs_codes['dfs_to'][mask_dfs2]]
            
            emb_dfs1 = torch.ones((n_seq, 
                                   n_batch, 
                                   self.emb_dim), device=dfs_codes['dfs_from'].device)
            emb_dfs2 = torch.ones((n_seq, 
                                   n_batch, 
                                   self.emb_dim), device=dfs_codes['dfs_from'].device)
            emb_atm1 = torch.ones((n_seq, 
                                   n_batch, 
                                   self.emb_dim), device=dfs_codes['dfs_from'].device)
            emb_atm2 = torch.ones((n_seq, 
                                   n_batch, 
                                   self.emb_dim), device=dfs_codes['dfs_from'].device)
            emb_bnd = torch.ones((n_seq, 
                                  n_batch, 
                                  self.emb_dim), device=dfs_codes['dfs_from'].device)

            emb_dfs1[mask_dfs1] *= dfs1
            emb_dfs2[mask_dfs2] *= dfs2
            emb_atm1[mask_atm1] *= atm1
            emb_atm2[mask_atm2] *= atm2
            emb_bnd[mask_bnd] *= bnd
            emb_dfs1[missing_dfs1] *= self.missing_dfs1
            emb_dfs2[missing_dfs2] *= self.missing_dfs2
            emb_atm1[missing_atm1] *= self.missing_atm1
            emb_atm2[missing_atm2] *= self.missing_atm2
            emb_bnd[missing_bnd] *= self.missing_bnd
            code_emb = torch.cat((emb_dfs1, emb_dfs2, emb_atm1, emb_atm2, emb_bnd), dim=2)
            code_emb[~src_key_padding_mask.T] = self.mixer(code_emb[~src_key_padding_mask.T])
            batch = self.emb_seq(code_emb * self.rescale_factor)
        else:
            raise NotImplemented('not implemented yet')
            
            
        # batch is of shape (sequence length, batch, d_model)
        if class_token is None:
            self_attn = self.enc(batch, src_key_padding_mask=src_key_padding_mask)
        else:
            src_key_padding_mask = torch.cat((torch.zeros((batch.shape[1], class_token.shape[0]), dtype=torch.bool, device=batch.device), 
                                              src_key_padding_mask), dim=1) # n_batch x n_seq
            self_attn = self.enc(torch.cat((class_token.expand(-1, batch.shape[1], -1), batch), dim=0),
                                 src_key_padding_mask=src_key_padding_mask)
        return self_attn, src_key_padding_mask
    
    
class DFSCodeEncoderJoint(nn.Module):
    """
    embeds the whole 5-tuple using one 2-layer fully connected.
    """
    def __init__(self, n_node_features, n_edge_features, 
                 emb_dim=120, nhead=12, nlayers=6, dim_feedforward=2048, 
                 activation = 'gelu',
                 max_nodes=250, max_edges=500, dropout=0.1, missing_value=None,
                 rescale_flag=True, **kwargs):
        super().__init__()
        self.ninp = emb_dim 
        self.emb_seq = PositionalEncoding(self.ninp, max_len=max_edges, dropout=dropout)
        self.n_input_features = 2*n_node_features + n_edge_features + 2*max_nodes
        self.max_nodes = max_nodes
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.max_edges = max_edges
        self.emb_5tuple = nn.Sequential(nn.Linear(self.n_input_features, self.ninp), # or
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.ninp, self.ninp)) # and
        
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.ninp, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward,
                                                                    dropout=dropout, 
                                                                    activation=activation), nlayers)
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
            atm_to = dfs_codes['atm_to'][mask]
            bnd_inp = dfs_codes['bnd'][mask]
            dfs_from = nn.functional.one_hot(dfs_codes['dfs_from'][mask], self.max_nodes)
            dfs_to = nn.functional.one_hot(dfs_codes['dfs_to'][mask], self.max_nodes)
            tuple5 = torch.cat((dfs_from, dfs_to, atm_from, atm_to, bnd_inp), dim=1)
            code_input = self.emb_5tuple(tuple5)
            
            code_emb = torch.ones((n_seq, 
                                   n_batch, 
                                   self.ninp), device=dfs_codes['dfs_from'].device)
            code_emb[mask] *= code_input
            code_emb[missing] *= self.missing_token
            batch = self.emb_seq(code_emb * self.rescale_factor)
        else:
            raise NotImplemented('not implemented yet')
            
            
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
                 missing_value=None, return_features=False, 
                 encoder_class="DFSCodeEncoder", cls_for_seq=True, **kwargs):
        super().__init__()
        self.nlayers = nlayers
        self.n_class_tokens = n_class_tokens
        if encoder_class == "DFSCodeEncoderV1":
            emb_dim = 5*emb_dim
            atom_embedding = nn.Linear(n_node_features, emb_dim//2)
            bond_embedding = nn.Linear(n_edge_features, emb_dim)
            self.encoder = eval(encoder_class)(atom_embedding, bond_embedding, 
                                          emb_dim=emb_dim, nhead=nhead, nlayers=nlayers, 
                                          dim_feedforward=dim_feedforward,
                                          max_nodes=max_nodes, max_edges=max_edges, 
                                          dropout=dropout, missing_value=missing_value)
        elif encoder_class == "DFSCodeEncoderJoint":
            emb_dim = 5*emb_dim
            self.encoder = eval(encoder_class)(n_node_features, n_edge_features, 
                                          emb_dim=emb_dim, nhead=nhead, nlayers=nlayers, 
                                          dim_feedforward=dim_feedforward,
                                          max_nodes=max_nodes, max_edges=max_edges, 
                                          dropout=dropout, missing_value=missing_value)
        else:
            atom_embedding = nn.Linear(n_node_features, emb_dim)
            bond_embedding = nn.Linear(n_edge_features, emb_dim)
            self.encoder = eval(encoder_class)(atom_embedding, bond_embedding, 
                                          emb_dim=emb_dim, nhead=nhead, nlayers=nlayers, 
                                          dim_feedforward=dim_feedforward,
                                          max_nodes=max_nodes, max_edges=max_edges, 
                                          dropout=dropout, missing_value=missing_value)
        self.ninp = self.encoder.ninp
        self.cls_for_seq = cls_for_seq
        self.return_features = return_features
        
        if not cls_for_seq:
            self.fc_dfs_idx1 = nn.Linear(self.ninp, max_nodes)
            self.fc_dfs_idx2 = nn.Linear(self.ninp, max_nodes)
            self.fc_atom1 = nn.Linear(self.ninp, n_atoms)
            self.fc_atom2 = nn.Linear(self.ninp, n_atoms)
            self.fc_bond = nn.Linear(self.ninp, n_bonds)
        else:
            self.fc_dfs_idx1 = nn.Linear(self.ninp + n_class_tokens*self.ninp, max_nodes)
            self.fc_dfs_idx2 = nn.Linear(self.ninp + n_class_tokens*self.ninp, max_nodes)
            self.fc_atom1 = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_atoms)
            self.fc_atom2 = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_atoms)
            self.fc_bond = nn.Linear(self.ninp + n_class_tokens*self.ninp, n_bonds)
        
        self.cls_token = nn.Parameter(torch.empty(n_class_tokens, 1, self.ninp), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=.0, std=.5)
        
    def forward(self, dfs_codes):
        self_attn, _ = self.encoder(dfs_codes, class_token=self.cls_token) # seq x batch x feat
        
        batch_feats = self_attn[self.n_class_tokens:] 
        if self.cls_for_seq:
            #TODO: check if this batch global thingy is correct and does the same thing einsum would do
            batch_global = self_attn[:self.n_class_tokens]
            batch_global = batch_global.view(1, -1, self.n_class_tokens * self.ninp)
            batch_global = batch_global.expand(batch_feats.shape[0], -1, -1) # batch x seq x ncls * feats
            batch = torch.cat((batch_feats, batch_global), dim=2) # each of the prediction heads gets cls token as additional input
        else:
            batch = batch_feats
        
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
    
    def encode(self, dfs_codes, method="cls"):
        if method == "cls3":
            forward_hook_manager = ForwardHookManager(C[0].device)
            for lid in range(self.nlayers): 
                forward_hook_manager.add_hook(self.encoder, 'enc.layers.%d'%lid, requires_input=False, requires_output=True)
        ncls = self.n_class_tokens
        self_attn, _ = self.encoder(dfs_codes, class_token=self.cls_token) # seq x batch x feat
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
    
    def fwd_code(self, dfs_codes, targets, features=False):
        """
        only fills in the masked inputs
        """
        dfs1 = dfs_codes['dfs_from'].clone()
        dfs2 = dfs_codes['dfs_to'].clone()
        if features:
            atm1 = torch.argmax(dfs_codes['atm_from'][:, :, :100], dim=2)+1
            atm2 = torch.argmax(dfs_codes['atm_to'][:, :, :100], dim=2)+1
            bnd = torch.argmax(dfs_codes['bnd'][:, :, 1:5], dim=2) #single doble triple aromatic is used in chemprop
            tmp = bnd.clone()
            bnd[tmp==2] = 3
            bnd[tmp==3] = 2
        else:
            atm1 = torch.argmax(dfs_codes['atm_from'], dim=2)+1
            atm2 = torch.argmax(dfs_codes['atm_to'], dim=2)+1
            bnd = torch.argmax(dfs_codes['bnd'], dim=2)

        mask = dfs1 == -1000
        
        missing_dfs1 = (targets[:, :, 0] != -1)*(~mask)
        missing_dfs2 = (targets[:, :, 1] != -1)*(~mask)
        missing_atm1 = (targets[:, :, 2] != -1)*(~mask)
        missing_atm2 = (targets[:, :, 4] != -1)*(~mask)
        missing_bnd = (targets[:, :, 3] != -1)*(~mask)
        dfs1_logits, dfs2_logits, atm1_logits, atm2_logits, bnd_logits = self.forward(dfs_codes)
        
        dfs1[missing_dfs1] = torch.argmax(dfs1_logits, dim=2)[missing_dfs1]
        dfs2[missing_dfs2] = torch.argmax(dfs2_logits, dim=2)[missing_dfs2]
        # here we don't need +1 cuz it is trained without the +1...
        atm1[missing_atm1] = torch.argmax(atm1_logits, dim=2)[missing_atm1]
        atm2[missing_atm2] = torch.argmax(atm2_logits, dim=2)[missing_atm2]
        bnd[missing_bnd] = torch.argmax(bnd_logits, dim=2)[missing_bnd]
        
        dfs1[mask] = -1000
        dfs2[mask] = -1000
        atm1[mask] = -1000
        atm2[mask] = -1000
        bnd[mask] = -1000
        dfs_code_list = []
        for d1, d2, a1, a2, b in zip(dfs1.T, dfs2.T, atm1.T, atm2.T, bnd.T):
            mask = d1 != -1000
            code = torch.cat((d1[mask].unsqueeze(1), 
                               d2[mask].unsqueeze(1), 
                               a1[mask].unsqueeze(1), 
                               b[mask].unsqueeze(1), 
                               a2[mask].unsqueeze(1)), dim=1)
            dfs_code_list += [code.detach().cpu().numpy().tolist()]
        return dfs_code_list
    
    
    def fwd_code_sample(self, dfs_codes, targets, features=False):
        """
        only fills in the masked inputs
        """
        dfs1 = dfs_codes['dfs_from'].clone()
        dfs2 = dfs_codes['dfs_to'].clone()
        if features:
            atm1 = torch.argmax(dfs_codes['atm_from'][:, :, :100], dim=2)+1
            atm2 = torch.argmax(dfs_codes['atm_to'][:, :, :100], dim=2)+1
            bnd = torch.argmax(dfs_codes['bnd'][:, :, 1:5], dim=2) #single doble triple aromatic is used in chemprop
            tmp = bnd.clone()
            bnd[tmp==2] = 3
            bnd[tmp==3] = 2
        else:
            atm1 = torch.argmax(dfs_codes['atm_from'], dim=2)+1
            atm2 = torch.argmax(dfs_codes['atm_to'], dim=2)+1
            bnd = torch.argmax(dfs_codes['bnd'], dim=2)

        mask = dfs1 == -1000
        
        missing_dfs1 = (targets[:, :, 0] != -1)*(~mask)
        missing_dfs2 = (targets[:, :, 1] != -1)*(~mask)
        missing_atm1 = (targets[:, :, 2] != -1)*(~mask)
        missing_atm2 = (targets[:, :, 4] != -1)*(~mask)
        missing_bnd = (targets[:, :, 3] != -1)*(~mask)
        dfs1_logits, dfs2_logits, atm1_logits, atm2_logits, bnd_logits = self.forward(dfs_codes)
        sm = nn.Softmax(dim=1)
        d0 = dfs1_logits.shape[0]
        d1 = dfs1_logits.shape[1]
        dfs1_logits = rearrange(dfs1_logits, 'd0 d1 d2 -> (d0 d1) d2')
        dfs2_logits = rearrange(dfs2_logits, 'd0 d1 d2 -> (d0 d1) d2')
        atm1_logits = rearrange(atm1_logits, 'd0 d1 d2 -> (d0 d1) d2')
        atm2_logits = rearrange(atm2_logits, 'd0 d1 d2 -> (d0 d1) d2')
        bnd_logits = rearrange(bnd_logits, 'd0 d1 d2 -> (d0 d1) d2')
        dfs1s = torch.multinomial(sm(dfs1_logits), 1)
        dfs2s = torch.multinomial(sm(dfs2_logits), 1)
        atm1s = torch.multinomial(sm(atm1_logits), 1)
        atm2s = torch.multinomial(sm(atm2_logits), 1)
        bnds= torch.multinomial(sm(bnd_logits), 1)
        dfs1s = rearrange(dfs1s, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze()
        dfs2s = rearrange(dfs2s, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze()
        atm1s = rearrange(atm1s, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze() 
        atm2s = rearrange(atm2s, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze() 
        bnds = rearrange(bnds, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze() 
        
        
        dfs1[missing_dfs1] = dfs1s[missing_dfs1]
        dfs2[missing_dfs2] = dfs2s[missing_dfs2]
        # here we don't need +1 cuz it is trained without the +1...
        atm1[missing_atm1] = atm1s[missing_atm1]
        atm2[missing_atm2] = atm2s[missing_atm2]
        bnd[missing_bnd] = bnds[missing_bnd]
        
        dfs1[mask] = -1000
        dfs2[mask] = -1000
        atm1[mask] = -1000
        atm2[mask] = -1000
        bnd[mask] = -1000
        dfs_code_list = []
        for d1, d2, a1, a2, b in zip(dfs1.T, dfs2.T, atm1.T, atm2.T, bnd.T):
            mask = d1 != -1000
            code = torch.cat((d1[mask].unsqueeze(1), 
                               d2[mask].unsqueeze(1), 
                               a1[mask].unsqueeze(1), 
                               b[mask].unsqueeze(1), 
                               a2[mask].unsqueeze(1)), dim=1)
            dfs_code_list += [code.detach().cpu().numpy().tolist()]
        return dfs_code_list
    
    
    def fwd_code_all(self, dfs_codes):
        mask = dfs_codes['dfs_from'] == -1000
        dfs1_logits, dfs2_logits, atm1_logits, atm2_logits, bnd_logits = self.forward(dfs_codes)
        dfs1 = torch.argmax(dfs1_logits, dim=2)
        dfs2 = torch.argmax(dfs2_logits, dim=2)
        atm1 = torch.argmax(atm1_logits, dim=2)
        atm2 = torch.argmax(atm2_logits, dim=2)
        bnd = torch.argmax(bnd_logits, dim=2)

        dfs1[mask] = -1000
        dfs2[mask] = -1000
        atm1[mask] = -1000
        atm2[mask] = -1000
        bnd[mask] = -1000
        dfs_code_list = []
        for d1, d2, a1, a2, b in zip(dfs1.T, dfs2.T, atm1.T, atm2.T, bnd.T):
            mask = d1 != -1000
            code = torch.cat((d1[mask].unsqueeze(1), 
                               d2[mask].unsqueeze(1), 
                               a1[mask].unsqueeze(1), 
                               b[mask].unsqueeze(1), 
                               a2[mask].unsqueeze(1)), dim=1)
            dfs_code_list += [code.detach().cpu().numpy().tolist()]
        return dfs_code_list
    
    
    def fwd_code_all_sample(self, dfs_codes):
        mask = dfs_codes['dfs_from'] == -1000
        dfs1_logits, dfs2_logits, atm1_logits, atm2_logits, bnd_logits = self.forward(dfs_codes)
        sm = nn.Softmax(dim=1)
        d0 = dfs1_logits.shape[0]
        d1 = dfs1_logits.shape[1]
        dfs1_logits = rearrange(dfs1_logits, 'd0 d1 d2 -> (d0 d1) d2')
        dfs2_logits = rearrange(dfs2_logits, 'd0 d1 d2 -> (d0 d1) d2')
        atm1_logits = rearrange(atm1_logits, 'd0 d1 d2 -> (d0 d1) d2')
        atm2_logits = rearrange(atm2_logits, 'd0 d1 d2 -> (d0 d1) d2')
        bnd_logits = rearrange(bnd_logits, 'd0 d1 d2 -> (d0 d1) d2')
        dfs1 = torch.multinomial(sm(dfs1_logits), 1)
        dfs2 = torch.multinomial(sm(dfs2_logits), 1)
        atm1 = torch.multinomial(sm(atm1_logits), 1)
        atm2 = torch.multinomial(sm(atm2_logits), 1)
        bnd = torch.multinomial(sm(bnd_logits), 1)
        dfs1 = rearrange(dfs1, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze()
        dfs2 = rearrange(dfs2, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze()
        atm1 = rearrange(atm1, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze() 
        atm2 = rearrange(atm2, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze() 
        bnd = rearrange(bnd, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze() 
       
        dfs1[mask] = -1000
        dfs2[mask] = -1000
        atm1[mask] = -1000
        atm2[mask] = -1000
        bnd[mask] = -1000
        
        dfs_code_list = []
        for d1, d2, a1, a2, b in zip(dfs1.T, dfs2.T, atm1.T, atm2.T, bnd.T):
            mask = d1 != -1000
            code = torch.cat((d1[mask].unsqueeze(1), 
                               d2[mask].unsqueeze(1), 
                               a1[mask].unsqueeze(1), 
                               b[mask].unsqueeze(1), 
                               a2[mask].unsqueeze(1)), dim=1)
            dfs_code_list += [code.detach().cpu().numpy().tolist()]
        return dfs_code_list
        
    
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
            

        

