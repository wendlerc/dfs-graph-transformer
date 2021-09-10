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
import json
import dfs_code


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import networkx as nx
from chemprop.features.featurization import atom_features, bond_features



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
 rdkit.Chem.rdchem.BondType.TRIPLE: 3,
 "loop": 4}


def smiles2graph(smiles, useHs=False, addLoops=False, dontTrimEdges=True, 
                 max_nodes=np.inf, max_edges=np.inf, skipCliqueCheck=False):
    """
    Parameters
    ----------
    smiles : string
        smiles representation of molecule.
    useHs : bool
        whether to model H atoms as nodes. The default is False.
    addLoops : bool
        whether to add loops to the atoms. The default is False.
    max_nodes : int, optional
        the maximum number of atoms. The default is 100.
    max_edges : int, optional
        the maximum number of edges. The default is 200.

    Returns
    -------
    d :  torch geometric data object containing the graph or None in case of failure
    """
    mol = Chem.MolFromSmiles(smiles)
    if useHs:
        mol = Chem.rdmolops.AddHs(mol)        
    N = mol.GetNumAtoms()

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    atom_chemprop = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        atom_chemprop.append(atom_features(atom))

    if len(atomic_number) > max_nodes:
        return None
    z = torch.tensor(atomic_number, dtype=torch.long)
    

    atomic_number = np.asarray(atomic_number)
    aromatic = np.asarray(aromatic)
    sp = np.asarray(sp)
    sp2 = np.asarray(sp2)
    sp3 = np.asarray(sp3)
    num_hs = np.asarray(num_hs)
    atom_chemprop = torch.tensor(atom_chemprop, dtype=torch.float)

    row, col, edge_type = [], [], []
    bond_chemprop = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bond_chemprop += 2*[bond_features(bond)]
        
    if addLoops or len(edge_type) == 0:
        for vidx, anr in enumerate(z.numpy()):
            row += [vidx, vidx]
            col += [vidx, vidx]
            edge_type += 2*[bonds["loop"]]
            bond_chemprop += 2*[bond_features(None)]
            
    if len(edge_type) > 2*max_edges:
        return None
    
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type,
                          num_classes=len(bonds)).to(torch.float)
    bond_chemprop = torch.tensor(bond_chemprop, dtype=torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]
    bond_chemprop = bond_chemprop[perm]
    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                      dtype=torch.float).t().contiguous()

    # only keep largest connected component
    if not skipCliqueCheck:
        edges_coo = edge_index.detach().cpu().numpy().T
        g = nx.Graph()
        g.add_nodes_from(np.arange(len(z)))
        g.add_edges_from(edges_coo.tolist())
    
        ccs = list(nx.connected_components(g))
        largest_cc = ccs[np.argmax([len(cc) for cc in ccs])]
        node_ids = np.asarray(list(largest_cc))
    
        x = x[node_ids]
        z = z[node_ids]
        atom_chemprop = atom_chemprop[node_ids]
        edges_cc = []
        edge_feats = []
        bond_chemprop_new = []
        old2new = {old:new for new, old in enumerate(node_ids)}
        for idx2, (u, v) in enumerate(edges_coo):
            if u in node_ids and v in node_ids:
                edges_cc += [[old2new[u], old2new[v]]]
                edge_feats += [edge_attr[idx2].numpy().tolist()]
                bond_chemprop_new += [bond_chemprop[idx2].numpy().tolist()]
        
        edge_index = torch.tensor(edges_cc, dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        bond_chemprop = torch.tensor(bond_chemprop_new, dtype=torch.float)
        
    
    if not addLoops and not dontTrimEdges:
        edge_attr = edge_attr[:, :4]
    
    
    d = Data(x=x, z=z, pos=None, edge_index=edge_index.T,
                    edge_attr=edge_attr, atom_features=atom_chemprop, bond_features=bond_chemprop)
    return d
    
    

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


def collate_minc_rndc_features_y(dlist):
    node_batch = []
    y_batch = []
    edge_batch = []
    rnd_code_batch = []
    min_code_batch = []
    for d in dlist:
        rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                         d.z.numpy().tolist(), 
                                                                         np.argmax(d.edge_attr.numpy(), axis=1))
        node_batch += [d.node_features]
        edge_batch += [d.edge_features]
        rnd_code_batch += [torch.tensor(rnd_code)]
        min_code_batch += [d.min_dfs_code]
        y_batch += [d.y]
    return rnd_code_batch, min_code_batch, node_batch, edge_batch, torch.cat(y_batch)


def collate_smiles_y(dlist):
    z_batch = []
    y_batch = []
    edge_attr_batch = []
    smiles_batch = []
    for d in dlist:
        smiles_batch += [d.smiles]
        z_batch += [nn.functional.one_hot(d.z, 118).numpy()]#118 elements in periodic table
        edge_attr_batch += [d.edge_attr.numpy()]
        y_batch += [d.y]
    return smiles_batch, z_batch, edge_attr_batch, torch.cat(y_batch)


class Deepchem2TorchGeometric(Dataset):
    def __init__(self, smiles, labels, loaddir=None,
                 max_edges=np.inf, max_nodes=np.inf, onlyRandom=False,
                 useHs=False, addLoops=False, trimEdges=True, precompute_min_dfs=True, 
                 features="chemprop"):
        self.smiles = smiles
        self.labels = labels[:, np.newaxis]
        self.useHs = useHs
        self.addLoops = addLoops
        self.loaddir = loaddir
        self.data = []
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.onlyRandom = onlyRandom
        self.trimEdges = trimEdges
        self.features = features
        self.prepare()
  
    
    def prepare(self):
        if self.loaddir is not None:
            with open(self.loaddir+'data.json', 'r') as f:
                data = json.load(f)
            with open(self.loaddir+'min_dfs_codes.json', 'r') as f:
                dfs = json.load(f)
        
        for idx in range(len(self.smiles)):
            smiles = self.smiles[idx]
            if self.loaddir is not None:
                if smiles in dfs:
                    d = data[smiles]
                    d = {key:torch.tensor(value) for key, value in d.items()}
                    min_code = dfs[smiles]['min_dfs_code']
                    min_index = dfs[smiles]['dfs_index']
                else:
                    continue
            else:
                d = smiles2graph(smiles, self.useHs, self.addLoops, not self.trimEdges, self.max_nodes, self.max_edges)
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
                
                
            z = d['z']
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
                node_features = nn.functional.one_hot(z, num_classes=118).float()
                edge_features = d['edge_attr']
                
                
            self.data += [Data(x=d['x'], z=d['z'], pos=None, edge_index=d['edge_index'],
                            edge_attr=d['edge_attr'], y=torch.tensor(self.labels[idx], dtype=torch.float32),
                            min_dfs_code=torch.tensor(min_code), min_dfs_index=torch.tensor(min_index), 
                            smiles=smiles, node_features=node_features, edge_features=edge_features)]
            
    

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        return self.data[idx]
