#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:27:25 2021

@author: chrisw
"""
import deepchem as dc
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np

import dfs_code


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import networkx as nx


types = {'C': 0,
 'O': 1,
 'N': 2,
 'Cl': 3,
 'S': 4,
 'F': 5,
 'P': 6,
 'Se': 7,
 'Br': 8,
 'I': 9,
 'Na': 10,
 'B': 11,
 'K': 12,
 'Li': 13,
 'H': 14,
 'Si': 15,
 'Ca': 16,
 'Rb': 17,
 'Te': 18,
 'Zn': 19,
 'Mg': 20,
 'As': 21,
 'Al': 22,
 'Ba': 23,
 'Be': 24,
 'Sr': 25,
 'Ag': 26,
 'Bi': 27,
 'Ra': 28,
 'Kr': 29,
 'Cs': 30,
 'Xe': 31,
 'He': 32,#ChEMBL end
 'Au': 33,
 'Sn': 34,
 'Hg': 35,
 'Ge': 36,
 'Sb': 37,
 'Pb': 38,
 'Cu': 39,
 }
bonds =  {rdkit.Chem.rdchem.BondType.SINGLE: 0,
 rdkit.Chem.rdchem.BondType.DOUBLE: 1,
 rdkit.Chem.rdchem.BondType.AROMATIC: 2,
 rdkit.Chem.rdchem.BondType.TRIPLE: 3}

def collate_minc_rndc_y(dlist):
    z_batch = []
    y_batch = []
    edge_attr_batch = []
    rnd_code_batch = []
    min_code_batch = []
    for d in dlist:
        rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                         d.z.numpy().tolist(), 
                                                                         np.argmax(d.edge_attr.numpy(), axis=1))
        z_batch += [d.z]#[nn.functional.one_hot(d.z, 118)]#118 elements in periodic table
        edge_attr_batch += [d.edge_attr]
        rnd_code_batch += [torch.tensor(rnd_code)]
        min_code_batch += [d.min_dfs_code]
        y_batch += [d.y]
    return rnd_code_batch, min_code_batch, z_batch, edge_attr_batch, torch.cat(y_batch)

class Deepchem2TorchGeometric(Dataset):
    def __init__(self, deepchem_smiles_dataset, taskid=0, useHs=False, precompute_min_dfs=True):
        self.deepchem = deepchem_smiles_dataset
        self.smiles = deepchem_smiles_dataset.X
        self.labels = deepchem_smiles_dataset.y[:, taskid][:, np.newaxis]
        self.w = deepchem_smiles_dataset.w
        self.useHs = useHs
        self.precompute_min_dfs=precompute_min_dfs
        self.data = []
        self.prepare()
  
    
    def prepare(self):
        for idx in range(len(self.smiles)):
            smiles = self.smiles[idx]
            mol = Chem.MolFromSmiles(smiles)
            if self.useHs:
                mol = Chem.rdmolops.AddHs(mol)        
            N = mol.GetNumAtoms()

            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            atomic_number = np.asarray(atomic_number)
            aromatic = np.asarray(atomic_number)
            sp = np.asarray(sp)
            sp2 = np.asarray(sp2)
            sp3 = np.asarray(sp3)
            num_hs = np.asarray(num_hs)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()

            # only keep largest connected component
            edges_coo = edge_index.detach().cpu().numpy().T
            g = nx.Graph()
            g.add_nodes_from(np.arange(len(z)))
            g.add_edges_from(edges_coo.tolist())

            ccs = list(nx.connected_components(g))
            largest_cc = ccs[np.argmax([len(cc) for cc in ccs])]
            node_ids = np.asarray(list(largest_cc))

            x = x[node_ids]
            z = z[node_ids]
            edges_cc = []
            edge_feats = []
            old2new = {old:new for new, old in enumerate(node_ids)}
            for idx2, (u, v) in enumerate(edges_coo):
                if u in node_ids and v in node_ids:
                    edges_cc += [[old2new[u], old2new[v]]]
                    edge_feats += [edge_attr[idx2].numpy().tolist()]
            edge_index = torch.tensor(edges_cc, dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
            
            d = Data(x=x, z=z, pos=None, edge_index=edge_index.T,
                            edge_attr=edge_attr, y=torch.tensor(self.labels[idx]))
            if len(edge_attr) == 0:
                continue 
            
            min_code, min_index = dfs_code.min_dfs_code_from_torch_geometric(d, 
                                                                         d.z.numpy().tolist(), 
                                                                         np.argmax(d.edge_attr.numpy(), axis=1))
            self.data += [Data(x=x, z=z, pos=None, edge_index=edge_index.T,
                            edge_attr=edge_attr, y=torch.tensor(self.labels[idx]),
                            min_dfs_code=torch.tensor(min_code), min_dfs_index=torch.tensor(min_index))]
            
    

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        return self.data[idx]
