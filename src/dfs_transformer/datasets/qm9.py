#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:27:25 2021

@author: chrisw
"""
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import torch.nn as nn
import numpy as np
import dfs_code
from .utils import smiles2graph

import os
import pickle
import tqdm
import deepchem as dc

from pathlib import Path



HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


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

conversion_dir = {7: HAR2EV}


class QM9(Dataset):
    def __init__(self, target_idx = 7, split = "train", onlyRandom=False, useHs=False, 
                 addLoops=False, features="chemprop", path = None):
        """
        

        Parameters
        ----------
        target_idx : index of target variable. The default is 7.
        split : train, valid or test. The default is "train".

        """
        if target_idx not in conversion_dir:
            raise NotImplementedError("target not implemented yet")
            
        self.target_idx = target_idx
        self.split = split
        self.onlyRandom = onlyRandom 
        self.useHs = useHs 
        self.addLoops = addLoops 
        self.features = features 
        self.data = []
        tasks, dsets, transformers = dc.molnet.load_qm9(reload=False, featurizer=dc.feat.RawFeaturizer(True), splitter=None)
        self.smiles = dsets[0].X
        self.y = dsets[0].y[:, target_idx]*conversion_dir[target_idx]
        self.n_train = 110000
        self.n_valid = 10000
        if split == "train":
            self.smiles = self.smiles[:self.n_train]
            self.y = self.y[:self.n_train]
        elif split == "valid":
            self.smiles = self.smiles[self.n_train: self.n_train + self.n_valid]
            self.y = self.y[self.n_train: self.n_train + self.n_valid]
        else:
            self.smiles = self.smiles[self.n_train + self.n_valid:]
            self.y = self.y[self.n_train + self.n_valid:]
        
        self.path = path
            
        self.prepare()
    
    def prepare(self):
        if self.path is not None:
            dfs_file = Path(self.path + "/" + self.split + "/dfs_codes.pkl")
        
            if dfs_file.is_file():
                with open(self.path + "/" + self.split + "/dfs_codes.pkl", "rb") as f:
                    c_dict = pickle.load(f) 
                with open(self.path + "/" + self.split + "/features.pkl", "rb") as f:
                    d_dict = pickle.load(f)
            else:
                d_dict = {}
                c_dict = {}
        else:
            d_dict = {}
            c_dict = {}
        
        for smiles, y in tqdm.tqdm(zip(self.smiles, self.y)):
            if self.path is not None and dfs_file.is_file():
                if smiles not in c_dict:
                    continue
                d = d_dict[smiles]
                min_code = c_dict[smiles]['min_dfs_code']
                min_index = c_dict[smiles]['min_dfs_index']
            else:
                d = smiles2graph(smiles, self.useHs, self.addLoops)
                if d is None:
                    continue
                
                if self.onlyRandom:
                    min_code, min_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                             d.z.numpy().tolist(), 
                                                                             np.argmax(d.edge_attr.numpy(), axis=1))
                else:
                    min_code, min_index = dfs_code.min_dfs_code_from_torch_geometric(d, 
                                                                             d.z.numpy().tolist(), 
                                                                                 np.argmax(d.edge_attr.numpy(), axis=1))
                
                min_code = np.asarray(min_code)
                min_index = np.asarray(min_index)
                d_dict[smiles] = d
                c_dict[smiles] = {'min_dfs_code': min_code, 'min_dfs_index': min_index}
            z = torch.zeros(30, dtype=torch.long) 
            z[:len(d['z'])] = d['z']
            x = d['x']
            if self.features == "old":
                h = x[:, -1].clone().detach().long()
                z_ind = nn.functional.one_hot(z, num_classes=118).float()
                h_ind = nn.functional.one_hot(h, num_classes=5).float()
                node_features = torch.cat((z_ind, x[:, 1:-1], h_ind), dim=1)
                edge_features = d['edge_attr']
            elif self.features == "chemprop":
                node_features = d['atom_features']
                edge_features = d['bond_features']
            else:#no features
                node_features = nn.functional.one_hot(z-1, num_classes=118).float()
                edge_features = d['edge_attr']
                
            self.data += [Data(x=d['x'], z=z, pos=None, edge_index=d['edge_index'],
                            edge_attr=d['edge_attr'], y=torch.tensor(np.asarray([y]), dtype=torch.float32),
                            min_dfs_code=torch.tensor(min_code), min_dfs_index=torch.tensor(min_index), 
                            smiles=smiles, node_features=node_features, edge_features=edge_features)]
        if self.path is not None and not dfs_file.is_file():
            os.makedirs(self.path + "/" + self.split, exist_ok = True)
            with open(self.path + "/" + self.split + "/" + "dfs_codes.pkl", "wb") as f:
                pickle.dump(c_dict, f)
            with open(self.path + "/" + self.split + "/" + "features.pkl", "wb") as f:
                pickle.dump(d_dict, f)
            
            
            
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


