#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:48:40 2021

@author: chrisw
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets.qm9 import QM9
import torch_geometric.datasets.qm9 as qm9
from torch_geometric.data import DataLoader as GDataLoader
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn.models.schnet import GaussianSmearing

import torch_geometric.nn as tgnn
from torch_scatter import scatter
import tqdm
import numpy as np
import wandb
import json
import os
import argparse
import yaml
from ml_collections import ConfigDict

import sys
sys.path = ['./src'] + sys.path
from dfs_transformer import EarlyStopping, PositionalEncoding
import math
from copy import deepcopy

# [0] Reports MAE in eV / Chemical Accuracy of the target variable U0. 
# The chemical accuracy of U0 is 0.043 see [1, Table 5].

# Reproduced table [0]
# MXMNet: 0.00590/0.043 = 0.13720930232558143
# HMGNN:  0.00592/0.043 = 0.13767441860465118
# MPNN:   0.01935/0.043 = 0.45
# KRR:    0.0251 /0.043 = 0.5837209302325582
# [0] https://paperswithcode.com/sota/formation-energy-on-qm9
# [1] Neural Message Passing for Quantum Chemistry, https://arxiv.org/pdf/1704.01212v2.pdf
# MXMNet https://arxiv.org/pdf/2011.07457v1.pdf
# HMGNN https://arxiv.org/pdf/2009.12710v1.pdf
# MPNN https://arxiv.org/pdf/1704.01212v2.pdf
# KRR HDAD kernel ridge regression https://arxiv.org/pdf/1702.05532.pdf
# HDAD means HDAD (Histogram of distances, anglesand dihedral angles)

# [2] Reports the average value of MAE / Chemical Accuracy of over all targets
# [2] https://paperswithcode.com/sota/drug-discovery-on-qm9
target_dict = {0: 'mu, D, Dipole moment', 
               1: 'alpha, {a_0}^3, Isotropic polarizability', 
               2: 'epsilon_{HOMO}, eV, Highest occupied molecular orbital energy',
               3: 'epsilon_{LUMO}, eV, Lowest unoccupied molecular orbital energy',
               4: 'Delta, eV, Gap between HOMO and LUMO',
               5: '< R^2 >, {a_0}^2, Electronic spatial extent',
               6: 'ZPVE, eV, Zero point vibrational energy', 
               7: 'U_0, eV, Internal energy at 0K',
               8: 'U, eV, Internal energy at 298.15K', 
               9: 'H, eV, Enthalpy at 298.15K',
               10: 'G, eV, Free energy at 298.15K',  
               11: 'c_{v}, cal\(mol K), Heat capacity at 298.15K'}

chemical_accuracy = {idx:0.043 for idx in range(12)}
chemical_accuracy[0] = 0.1
chemical_accuracy[1] = 0.1
chemical_accuracy[5] = 1.2
chemical_accuracy[6] = 0.0012
chemical_accuracy[11] = 0.050

def preprocess_vertex_features(features):
    feat_np = features.detach().cpu().numpy()
    # make atomic number a one hot
    atomic_number = nn.functional.one_hot(features[:, 5].long(), 100)
    # make num_h a one hot
    num_h = nn.functional.one_hot(features[:, -4].long(), 9)
    return torch.cat((features[:, :5], features[:, 6:-4], features[:, -3:], atomic_number, num_h), axis=1)

def preprocess_vertex_features_nopos(features):
    feat_np = features.detach().cpu().numpy()
    # make atomic number a one hot
    atomic_number = nn.functional.one_hot(features[:, 5].long(), 100)
    # make num_h a one hot
    num_h = nn.functional.one_hot(features[:, -1].long(), 9)
    return torch.cat((features[:, :5], features[:, 6:-1], atomic_number, num_h), axis=1)

class QM9DFSCodes(Dataset):
    def __init__(self, dfs_codes, qm9_dataset, 
                 target_idx, vert_feats = ['x', 'pos'],
                 vertex_transform=None, edge_transform=None):
        self.dfs_codes = dfs_codes
        self.qm9_dataset = qm9_dataset
        self.target_idx = target_idx
        self.vertex_features = vert_feats
        self.max_vert = 29
        self.max_len = np.max([len(d['min_dfs_code']) for d in self.dfs_codes.values()])
        self.feat_dim = 2 + qm9_dataset[0].edge_attr.shape[1] # 2 for the dfs indices
        for feat in vert_feats: 
            self.feat_dim += 2*qm9_dataset[0][feat].shape[1]
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.dist_emb = GaussianSmearing(0., 4., 50)

    def __len__(self):
        return len(self.qm9_dataset)

    def __getitem__(self, idx):
        """
        graph_repr: [batch_size, max_edges, 2 + n_vert_feat + n_edge_feat + n_vert_feat]
        return vertex_features, graph_repr, target
        """
        data = self.qm9_dataset[idx]
        code_dict = self.dfs_codes[data.name]
        code = code_dict['min_dfs_code']
        
        vert_feats = [data[k].detach().cpu().numpy() for k in self.vertex_features]
        vert_feats = np.concatenate(vert_feats, axis=1)
        edge_feats = data.edge_attr.detach().cpu().numpy()
        dists = []
        edge_index = data.edge_index.detach().cpu().numpy().T
        pos = data.pos.detach().cpu().numpy()
        for e in edge_index:
            dists += [np.linalg.norm(pos[e[0]] - pos[e[1]])]
        dist_feats = self.dist_emb(torch.tensor(dists)).detach().cpu().numpy()
        edge_feats = np.concatenate((edge_feats, dist_feats), axis=1)
        
        
        d = {'dfs_from': np.zeros(self.max_len), 
             'dfs_to': np.zeros(self.max_len),
             'label_from': np.zeros(self.max_len),
             'label_to': np.zeros(self.max_len),
             'label_edge': np.zeros(self.max_len),
             'feat_from': np.zeros((self.max_len, vert_feats.shape[1])), 
             'feat_to': np.zeros((self.max_len, vert_feats.shape[1])), 
             'feat_edge':np.zeros((self.max_len, edge_feats.shape[1])),
             'n_edges':len(code)*np.ones((1,), dtype=np.int32),
             'z':np.zeros(self.max_vert)}
        
        for idx, e_tuple in enumerate(code):
            d['dfs_from'][idx] = e_tuple[0]+1
            d['dfs_to'][idx] = e_tuple[1]+1
            d['label_from'][idx] = e_tuple[2]+1
            d['label_to'][idx] = e_tuple[4]+1
            d['label_edge'][idx] = e_tuple[3]+1
            d['feat_from'][idx] = vert_feats[e_tuple[-3]]
            d['feat_to'][idx] = vert_feats[e_tuple[-1]]
            d['feat_edge'][idx] = edge_feats[e_tuple[-2]]      
        
        d['z'][:len(data.z)] = data.z.detach().cpu().numpy()
        
        d_tensors = {}
        d_tensors['dfs_from'] = torch.LongTensor(d['dfs_from']).clone()
        d_tensors['dfs_to'] = torch.LongTensor(d['dfs_to']).clone()
        d_tensors['label_from'] = torch.LongTensor(d['label_from']).clone()
        d_tensors['label_to'] = torch.LongTensor(d['label_to']).clone()
        d_tensors['label_edge'] = torch.LongTensor(d['label_edge']).clone()
        d_tensors['feat_from'] = torch.Tensor(d['feat_from']).clone()
        d_tensors['feat_to'] = torch.Tensor(d['feat_to']).clone()
        d_tensors['feat_edge'] = torch.Tensor(d['feat_edge']).clone()
        d_tensors['z'] = torch.LongTensor(d['z']).clone()
        
        if self.vertex_transform:
            d_tensors['feat_from'] = self.vertex_transform(d_tensors['feat_from'])
            d_tensors['feat_to'] = self.vertex_transform(d_tensors['feat_to'])
        if self.edge_transform:
            d_tensors['feat_edge'] = self.edge_transform(d_tensors['feat_edge'])
        
        d_tensors['target'] = data.y[:, self.target_idx].clone()
        return d_tensors
    
    def shuffle(self):
        self.qm9_dataset = self.qm9_dataset.shuffle()
        return self
    
class MoleculeTransformer(nn.Module):
    def __init__(self, vert_dim, edge_dim, d_model=600, nhead=8, nlayers=4, dim_feedforward=2048, mean=None, std=None, atomref=None,
                 max_vertices=29, max_edges=28):
        """
        transfomer model is some type of transformer that 
        """
        super(MoleculeTransformer, self).__init__()
        # atomic masses could be used as additional features
        # see https://github.com/rusty1s/pytorch_geometric/blob/97d3177dc43858f66c07bb66d7dc12506b986199/torch_geometric/nn/models/schnet.py#L113
        self.vert_dim = vert_dim
        self.edge_dim = edge_dim
        self.d_model = d_model
        assert d_model % 6 == 0
        self.emb_dim = d_model//6
        self.nhead = nhead
        self.nlayers = nlayers
        self.dim_feedforward = dim_feedforward
        self.max_vertices = max_vertices
        self.max_edges = max_edges
        
        # pos/topo encoding embeddings
        # position in sequence
        self.emb_seq = PositionalEncoding(self.emb_dim, max_len=max_edges, dropout=0.1)
        seq_emb = self.emb_seq(torch.zeros((max_edges, 1, self.emb_dim)))
        seq_emb = torch.squeeze(seq_emb)
        self.register_buffer('seq_emb', seq_emb)
        # dfs indices
        self.emb_dfs = PositionalEncoding(self.emb_dim, max_len=max_vertices+1, dropout=0)
        dfs_emb = self.emb_dfs(torch.zeros((max_vertices+1, 1, self.emb_dim)))
        dfs_emb = torch.squeeze(dfs_emb)
        self.register_buffer('dfs_emb', dfs_emb)
        # labels
        self.emb_vlabel = nn.Embedding(118+1, self.emb_dim)
        self.emb_elabel = nn.Embedding(5+1, self.emb_dim)
        # mix before adding to undirected features
        self.mixer = nn.Linear(self.d_model, self.d_model)
        # feature embeddings
        self.emb_vertex = nn.Linear(self.vert_dim, self.d_model)
        self.emb_edge = nn.Linear(self.edge_dim, self.d_model)      
        
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.d_model), requires_grad=True)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward), self.nlayers)
        
        self.fc_out = nn.Linear(self.d_model, 1)
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)
            
        nn.init.normal_(self.cls_token, mean=.0, std=.5)
    
    def forward(self, data):
        z = data['z']
        mask = data['dfs_from'] == 0
        # topological encoding
        dfs_from_emb = self.dfs_emb[data['dfs_from']]
        dfs_to_emb = self.dfs_emb[data['dfs_to']]
        dfs_emb = torch.cat((dfs_from_emb, dfs_to_emb), -1)
        dfs_emb[mask] = 0
        vlabel_emb = torch.cat((self.emb_vlabel(data['label_from']), 
                                self.emb_vlabel(data['label_to'])), -1)
        vlabel_emb[mask] = 0
        elabel_emb = self.emb_elabel(data['label_edge'])
        elabel_emb[mask] = 0
        pos_emb = self.seq_emb.unsqueeze(0).expand(data['dfs_from'].shape[0], -1, -1).clone()
        pos_emb[mask] = 0
        topo_enc = torch.cat((dfs_emb, vlabel_emb, elabel_emb, pos_emb), dim=-1)# batch_dim x seq_dim x n_model
        # features
        from_emb = self.emb_vertex(data['feat_from'])
        to_emb = self.emb_vertex(data['feat_to'])
        edge_emb = self.emb_edge(data['feat_edge'])
        feat_enc = from_emb + to_emb + edge_emb
        batch = self.mixer(topo_enc) + math.sqrt(self.d_model)*feat_enc
        
        batch = batch.permute(1, 0, 2) # seq_dim x batch_dim x n_model
        batch = torch.cat((self.cls_token.expand(-1, batch.shape[1], -1), batch), dim=0)
        transformer_out = self.enc(batch)
        out = self.fc_out(transformer_out[0]) 
        
        # tricks from Schnet
        if self.mean is not None and self.std is not None:
            out = out * self.std + self.mean
        
        if self.atomref is not None:
            out = out + torch.sum(self.atomref(z), axis=1)
        
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    args = parser.parse_args()
    
    if args.path is None:
        run = wandb.init(project='QM9-transformer', entity='chrisxx',
                         settings=wandb.Settings(start_method='fork'))
        config = wandb.config
        config.n_epochs = 9000
        config.minimal_lr = 6e-8
        config.target_idx = 7
        config.batch_size = 1000
        config.n_train = 110000
        config.n_valid = 10000
        config.target_ratio = 0.1
        config.store_starting_from_ratio = 1
        config.required_improvement = 0.8
        config.nlayers=6
        config.nhead=12
        config.d_model=600
        config.dim_feedforward=4*config.d_model
        config.model_dir = './models/qm9/gtransformer_6tuple_%d/'%(np.random.randint(10000000))
        config.dfs_codes = './datasets/qm9_torch_geometric/min_dfs_codes.json'
        config.num_workers = 4
        config.feats = ['x']
        config.no_tricks = False
        dset = QM9('./datasets/qm9_torch_geometric/')
        dset = dset.shuffle()
    else:
        artifact_path = args.path
        with open(artifact_path + 'config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = ConfigDict(config)
        run = wandb.init(project='QM9-transformer', entity='chrisxx',
                         settings=wandb.Settings(start_method='fork'),
                         config=config.to_dict())
        config = wandb.config
        #config.update({"n_epochs": 0, "batch_size":100}, allow_val_change=True)
        dset = QM9('./datasets/qm9_torch_geometric/')
        dataset_indices = torch.load(artifact_path+'dataset_indices.pt')
        dset = dset[dataset_indices]
    
    if 'feats' not in config.as_dict():
        config.update({"feats": ['x', 'pos']})
    if 'no_tricks' not in config.as_dict():
        config.update({"no_tricks": False})
    if 'pos' not in config.feats:
        preprocess_vertex_features = preprocess_vertex_features_nopos
        
    target_idx = config.target_idx
    
    with open(config.dfs_codes, 'r') as f:
        dfs_codes = json.load(f)
        
    train_qm9 = GDataLoader(dset[:config.n_train], batch_size=config.batch_size)
    train_dataset = QM9DFSCodes(dfs_codes, dset[:config.n_train], target_idx, vert_feats = config.feats, vertex_transform=preprocess_vertex_features)
    valid_dataset = QM9DFSCodes(dfs_codes, dset[config.n_train:config.n_train+config.n_valid], target_idx, vert_feats = config.feats, vertex_transform=preprocess_vertex_features) 
    test_dataset = QM9DFSCodes(dfs_codes, dset[config.n_train+config.n_valid:], target_idx, vert_feats = config.feats, vertex_transform=preprocess_vertex_features) 
    config.n_test = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    
    os.makedirs(config.model_dir, exist_ok=True)
    torch.save(dset.indices(), config.model_dir+'dataset_indices.pt')
    with open(config.model_dir+'config.yaml', 'w') as f:
        yaml.dump(config.as_dict(), f, default_flow_style=False)
    
    ngpu=1
    device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
        
    target_vec = []
    
    # based on https://schnetpack.readthedocs.io/en/stable/tutorials/tutorial_02_qm9.html
    # and https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html#SchNet
    for data in train_qm9:
        data = data.to(device)
        atomU0s = torch.tensor(qm9.atomrefs[target_idx], device=device)[torch.argmax(data.x[:, :5], axis=1)]
        target_modular = scatter(atomU0s, data.batch, dim=-1, reduce='sum')
        target_vec += [(data.y[:, target_idx] - target_modular).detach().cpu().numpy()]
    target_vec = np.concatenate(target_vec, axis=0)
    
    target_mean = torch.tensor(np.mean(target_vec))
    target_std = torch.tensor(np.std(target_vec))
    
    d = next(iter(train_loader))
    vert_dim = d['feat_from'].shape[-1]
    edge_dim = d['feat_edge'].shape[-1]
    if not config.no_tricks:
        model = MoleculeTransformer(vert_dim, edge_dim, nlayers=config.nlayers, nhead=config.nhead, 
                                    d_model=config.d_model, dim_feedforward=config.dim_feedforward, 
                                    atomref=dset.atomref(target_idx), mean=target_mean, std=target_std)
    else:
        model = MoleculeTransformer(vert_dim, edge_dim, nlayers=config.nlayers, nhead=config.nhead, 
                                    d_model=config.d_model, dim_feedforward=config.dim_feedforward)
    
    if args.path is not None:
        model.load_state_dict(torch.load(artifact_path+'checkpoint.pt', map_location=device), strict=False)
    
    loss = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1/(2*(512**0.5)), betas=(0.9,0.98), eps=1e-9)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda t: min(1/((t+1)**0.5), (t+1)*(1/(4000**1.5))), 
                                                                  verbose = False)
    
    model = model.to(device)
        
    
    loss_hist = []
    min_mae = config.store_starting_from_ratio
    try:
        # For each epoch
        for epoch in range(config.n_epochs):
            # For each batch in the dataloader
            model.train()
            pbar = tqdm.tqdm(enumerate(train_loader, 0))
            epoch_loss = 0
            for i, data in pbar:
                model.zero_grad()
                data = {key:d.to(device) for key, d in data.items()}
                target = data['target']
                prediction = model(data)
                output = loss(prediction.view(-1), target.view(-1))
                mae = (prediction.view(-1) - target.view(-1)).abs().mean()
                epoch_loss = (epoch_loss*i + mae.item())/(i+1)
                
                pbar.set_description('Epoch %d: MAE/CA %2.6f'%(epoch+1, epoch_loss/chemical_accuracy[target_idx]))
                output.backward()
                optimizer.step()
                lr_scheduler.step()
                curr_lr = list(optimizer.param_groups)[0]['lr']
                wandb.log({'MSE': output.item(), 'learning rate':curr_lr})
            
            wandb.log({'MAE':epoch_loss, 
                       'MAE/CA':epoch_loss/chemical_accuracy[target_idx]})
            
            loss_hist += [epoch_loss] 
    
            if epoch_loss/chemical_accuracy[target_idx] < min_mae*config.required_improvement:
                min_mae = epoch_loss/chemical_accuracy[target_idx]
                torch.save(model.state_dict(), config.model_dir+'checkpoint%d.pt'%epoch)
            if curr_lr < config.minimal_lr:
                break
            if epoch_loss/chemical_accuracy[target_idx] < config.target_ratio:
                break
            
            with torch.no_grad():
                model.eval()
                pbar = tqdm.tqdm(enumerate(valid_loader, 0))
                epoch_loss = 0
                maes = []
                for i, data in pbar:
                    data = {key:d.to(device) for key, d in data.items()}
                    target = data['target']
                    prediction = model(data)
                    mae = (prediction.view(-1) - target.view(-1)).abs()
                    maes += [mae.detach().cpu()]
                maes = torch.cat(maes, dim=0)
                mae = maes.mean().item()
                print(mae, mae/chemical_accuracy[target_idx])
                wandb.log({'VALID MAE':mae, 'VALID MAE/CA':mae/chemical_accuracy[target_idx]})
    
        torch.save(model.state_dict(), config.model_dir+'checkpoint_final.pt')
        
        
    except KeyboardInterrupt:
        print('keyboard interrupt caught')
        torch.save(model.state_dict(), config.model_dir+'checkpoint_interrupt.pt')
    finally:
        print("uploading model...")
        #store config and model
        with open(config.model_dir+'config.yaml', 'w') as f:
            yaml.dump(config.as_dict(), f, default_flow_style=False)
        trained_model_artifact = wandb.Artifact(run.name, type="model", description="trained selfattn model")
        trained_model_artifact.add_dir(config.model_dir)
        run.log_artifact(trained_model_artifact)
        
        with torch.no_grad():
            model.eval()
            pbar = tqdm.tqdm(enumerate(test_loader, 0))
            epoch_loss = 0
            maes = []
            for i, data in pbar:
                data = {key:d.to(device) for key, d in data.items()}
                target = data['target']
                prediction = model(data)
                mae = (prediction.view(-1) - target.view(-1)).abs()
                maes += [mae.detach().cpu()]
            maes = torch.cat(maes, dim=0)
            mae = maes.mean().item()
            print(mae, mae/chemical_accuracy[target_idx])
            wandb.log({'TEST MAE':mae, 'TEST MAE/CA':mae/chemical_accuracy[target_idx]})
        
    
    
    
