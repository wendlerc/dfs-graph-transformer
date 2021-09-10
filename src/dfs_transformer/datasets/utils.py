#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:27:25 2021

@author: chrisw
"""
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import networkx as nx
from chemprop.features.featurization import atom_features, bond_features


bonds =  {BT.SINGLE: 0, BT.DOUBLE: 1, BT.AROMATIC: 2, BT.TRIPLE: 3, "loop": 4}


def smiles2graph(smiles, useHs=False, addLoops=False, max_nodes=np.inf, max_edges=np.inf):
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
    
    d = Data(x=x, z=z, pos=None, edge_index=edge_index.T,
                    edge_attr=edge_attr, atom_features=atom_chemprop, bond_features=bond_chemprop)
    return d