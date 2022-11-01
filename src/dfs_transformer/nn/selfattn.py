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
from ..utils.rdkit import ATOM_FEATURES_SHAPES, BOND_FEATURES_SHAPES, parseChempropAtomFeatures, parseChempropBondFeatures
from ..utils import DFSCodesDict2Nx, FeaturizedDFSCodes2Dict
from copy import deepcopy
from collections import defaultdict

NFEAT_SHAPES = deepcopy(ATOM_FEATURES_SHAPES)
#we don't want to predict degree and mass
del NFEAT_SHAPES['degree']
del NFEAT_SHAPES['mass']
EFEAT_SHAPES = deepcopy(BOND_FEATURES_SHAPES)

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
        sequence_predictions, features = self.encoder(dfs_code)
        property_predictions = {name.replace("_", "."): head(features) for name, head in self.head_dict.items()}
        return sequence_predictions, property_predictions
    

class DFSCodeEncoder(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 emb_dim=120, nhead=12, nlayers=6, dim_feedforward=2048, 
                 activation='gelu',
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
    

class DFSCodeEncoderM3(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 emb_dim=120, nhead=12, nlayers=6, dim_feedforward=2048, 
                 activation='gelu',
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
            
        self.mixer = nn.Sequential(nn.Linear(self.ninp, self.ninp),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.ninp, 4*self.ninp),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(4*self.ninp, self.ninp))
        
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
                 encoder_class="DFSCodeEncoder", cls_for_seq=True, 
                 nfeat_shapes=NFEAT_SHAPES, 
                 efeat_shapes=EFEAT_SHAPES,
                 **kwargs): 
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
        
        # initialize all the prediction heads for the pretraining task
        if not cls_for_seq:
            dim_in = self.ninp
        else:
            dim_in = self.ninp + n_class_tokens*self.ninp
            
        fcs = {'dfs_from': nn.Linear(dim_in, max_nodes), 
               'dfs_to': nn.Linear(dim_in, max_nodes)}
        for key, dim in nfeat_shapes.items():
            if dim == 1:
                dim_out = dim+1
            else:
                dim_out = dim
            fcs[key+'_from'] = nn.Linear(dim_in, dim_out)
            fcs[key+'_to'] = nn.Linear(dim_in, dim_out)
            
        for key, dim in efeat_shapes.items():
            if dim == 1:
                dim_out = dim+1
            else:
                dim_out = dim
            fcs[key] = nn.Linear(dim_in, dim_out)
        
        self.fcs = nn.ModuleDict(fcs)
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
        
        # dictionary of key: tensor with shape nseq x nbatch x dim_of_predicted_feat
        predicted_sequences = {key: fc(batch) for key, fc in self.fcs.items()} 
        
        if self.return_features:
            features = self_attn[:self.n_class_tokens]
            features = rearrange(features, 'd0 d1 d2 -> d1 (d0 d2)')
            return predicted_sequences, features
        return predicted_sequences
    
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
    
    
    def fwd_smart(self, dfs_codes, targets):
        mask = dfs_codes['dfs_from'] == -1000
        dfs_codes_dict = FeaturizedDFSCodes2Dict(dfs_codes)
        pred_codes = self.forward(dfs_codes)
        # parse the known atom features from the input into a dict to reuse them later
        known_atom_features = defaultdict(dict)
        dfs_from = dfs_codes_dict['dfs_from']
        dfs_to = dfs_codes_dict['dfs_to']
        for key, feats in dfs_codes_dict.items():
            if key != 'dfs_from' and '_from' in key:
                fkey = key[:-5]
                for eid, (dfs_batch, feat_batch) in enumerate(zip(dfs_from, feats)):
                    for bid, (dfs_idx, feat) in enumerate(zip(dfs_batch, feat_batch)):
                        if targets[key][eid, bid].item() != -1:
                            continue
                        entry_key = (bid, dfs_idx.item()) 
                        if feat not in [-1, -1000]  and entry_key not in known_atom_features[fkey]:
                            known_atom_features[fkey][entry_key] = feat.item()
            elif key != 'dfs_to' and '_to' in key:
                fkey = key[:-3]
                for eid, (dfs_batch, feat_batch) in enumerate(zip(dfs_to, feats)):
                    for bid, (dfs_idx, feat) in enumerate(zip(dfs_batch, feat_batch)):
                        if targets[key][eid, bid].item() != -1:
                            continue
                        entry_key = (bid, dfs_idx.item()) 
                        if feat not in [-1, -1000] and entry_key not in known_atom_features[fkey]:
                            known_atom_features[fkey][entry_key] = feat.item()
        
        for (pkey, pred), (tkey, tgt) in zip(pred_codes.items(), targets.items()):
            tmask = tgt != -1
            new = dfs_codes_dict[pkey].clone()
            new[tmask] = torch.argmax(pred.detach().cpu(), dim=2)[tmask]
            new[mask] = -1000
            pred_codes[pkey] = new
            
        # overwrite atom features for the entries where we already know them from the inputs
        dfs_from = pred_codes['dfs_from']
        dfs_to = pred_codes['dfs_to']
        for key, feats in pred_codes.items():
            if key != 'dfs_from' and '_from' in key:
                fkey = key[:-5]
                for eid, (dfs_batch, feat_batch) in enumerate(zip(dfs_from, feats)):
                    for bid, (dfs_idx, feat) in enumerate(zip(dfs_batch, feat_batch)):
                        entry_key = (bid, dfs_idx.item()) 
                        if entry_key in known_atom_features[fkey]:
                            if pred_codes[key][eid, bid] != known_atom_features[fkey][entry_key]:
                                #print('%s dfs_id %d: %d -> %d'%(key, dfs_idx.item(), pred_codes[key][eid, bid], known_atom_features[fkey][entry_key]))
                                pred_codes[key][eid, bid] = known_atom_features[fkey][entry_key]
                                
            elif key != 'dfs_to' and '_to' in key:
                fkey = key[:-3]
                for eid, (dfs_batch, feat_batch) in enumerate(zip(dfs_to, feats)):
                    for bid, (dfs_idx, feat) in enumerate(zip(dfs_batch, feat_batch)):
                        entry_key = (bid, dfs_idx.item()) 
                        if entry_key in known_atom_features[fkey]:
                            if pred_codes[key][eid, bid] != known_atom_features[fkey][entry_key]:
                                #print('%s dfs_id %d: %d -> %d'%(key, dfs_idx.item(), pred_codes[key][eid, bid], known_atom_features[fkey][entry_key]))
                                pred_codes[key][eid, bid] = known_atom_features[fkey][entry_key]
        
        return DFSCodesDict2Nx(pred_codes, padding_value=-1000)
    
    
    def fwd_graph(self, dfs_codes, targets, pred_codes=None):
        mask = dfs_codes['dfs_from'] == -1000
        dfs_codes_dict = FeaturizedDFSCodes2Dict(dfs_codes)
        if pred_codes is None:
            pred_codes = self.forward(dfs_codes)
        for (pkey, pred), (tkey, tgt) in zip(pred_codes.items(), targets.items()):
            tmask = tgt != -1
            new = dfs_codes_dict[pkey].clone()
            new[tmask] = torch.argmax(pred.detach().cpu(), dim=2)[tmask]
            new[mask] = -1000
            pred_codes[pkey] = new
        return DFSCodesDict2Nx(pred_codes, padding_value=-1000)
    
    
    def fwd_graph_sample(self, dfs_codes, targets):
        mask = dfs_codes['dfs_from'] == -1000
        dfs_codes_dict = FeaturizedDFSCodes2Dict(dfs_codes)
        pred_codes = self.forward(dfs_codes)
        sm = nn.Softmax(dim=1)
        d0 = dfs_codes['dfs_from'].shape[0]
        d1 = dfs_codes['dfs_from'].shape[1]
        for (pkey, pred), (tkey, tgt) in zip(pred_codes.items(), targets.items()):
            tmask = tgt != -1
            new = dfs_codes_dict[pkey].clone()
            probas = sm(rearrange(pred.detach().cpu(), 'd0 d1 d2 -> (d0 d1) d2'))
            sample = torch.multinomial(probas, 1)
            sample = rearrange(sample, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze()
            new[tmask] = sample[tmask]
            new[mask] = -1000
            pred_codes[pkey] = new
        return DFSCodesDict2Nx(pred_codes, padding_value=-1000)
    
    
    def fwd_graph_all(self, dfs_codes):
        mask = dfs_codes['dfs_from'] == -1000
        pred_codes = self.forward(dfs_codes)
        for key, pred in pred_codes.items():
            pred_codes[key] = torch.argmax(pred.detach().cpu(), dim=2)
            pred_codes[key][mask] = -1000
        return DFSCodesDict2Nx(pred_codes, padding_value=-1000)

    
    def fwd_graph_all_sample(self, dfs_codes):
        mask = dfs_codes['dfs_from'] == -1000
        pred_codes = self.forward(dfs_codes)
        sm = nn.Softmax(dim=1)
        d0 = dfs_codes['dfs_from'].shape[0]
        d1 = dfs_codes['dfs_from'].shape[1]
        for key, pred in pred_codes.items():
            probas = sm(rearrange(pred.detach().cpu(), 'd0 d1 d2 -> (d0 d1) d2'))
            sample = torch.multinomial(probas, 1)
            sample = rearrange(sample, '(d0 d1) d2 -> d0 d1 d2', d0=d0, d1=d1).squeeze()
            pred_codes[key] = sample
            pred_codes[key][mask] = -1000
        return DFSCodesDict2Nx(pred_codes, padding_value=-1000)
        
    