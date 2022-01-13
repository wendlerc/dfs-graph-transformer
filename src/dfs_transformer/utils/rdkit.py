#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:57:29 2022

@author: chrisw
"""

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import dfs_code
import numpy as np
import networkx as nx 
import torch

bond_types =  {0: BT.SINGLE, 1: BT.DOUBLE, 2: BT.AROMATIC, 3: BT.TRIPLE}
bonds =  {BT.SINGLE: 0, BT.DOUBLE: 1, BT.AROMATIC: 2, BT.TRIPLE: 3, "loop": 4}


def DFSCode2Graph(dfs_code):
    # TODO: maybe check whether code is valid
    edge_list = []
    edge_labels = []
    node_dict = {}
    for etuple in dfs_code:
        edge_list += [(etuple[0], etuple[1])]
        edge_labels += [etuple[3]]
        node_dict[etuple[0]] = etuple[2]
        node_dict[etuple[1]] = etuple[4]
    node_labels = [node_dict[idx] for idx in range(len(node_dict))]
    return edge_list, node_labels, edge_labels


def Graph2Mol(edge_list, node_labels, edge_labels):
    # create empty editable mol object
    mol = Chem.RWMol()
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for node, atomic_num in enumerate(node_labels):
        a=Chem.Atom(atomic_num)
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx
    
    for (node_from, node_to), elabel in zip(edge_list, edge_labels):
        if elabel in bond_types:
            mol.AddBond(node_to_idx[node_from], node_to_idx[node_to], bond_types[elabel])
            
    # Convert RWMol to Mol object
    mol = mol.GetMol()            
    return mol


def isValid(mol, verbose=False):
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception as e:
        if verbose:
            print(e)
        return False
    

def Mol2Smiles(mol):
    return Chem.MolToSmiles(mol)


def DFSCode2Smiles(dfs_code):
    return Chem.MolToSmiles(Graph2Mol(*DFSCode2Graph(dfs_code)))


def isValidMoleculeDFSCode(dfs_code):
    return isValid(Graph2Mol(*DFSCode2Graph(dfs_code)))


def Smiles2DFSCode(smiles, useMin=False, useHs=False, addLoops=False, max_nodes=np.inf, max_edges=np.inf):
    mol = Chem.MolFromSmiles(smiles)
    if useHs:
        mol = Chem.rdmolops.AddHs(mol)        
    N = mol.GetNumAtoms()

    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())

    if len(atomic_number) > max_nodes:
        return None
    z = np.asarray(atomic_number, dtype=np.int64)
    

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        
    if addLoops or len(edge_type) == 0:
        for vidx, anr in enumerate(z.numpy()):
            row += [vidx, vidx]
            col += [vidx, vidx]
            edge_type += 2*[bonds["loop"]]
            
    if len(edge_type) > 2*max_edges:
        return None
    
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    
    row, col = edge_index

    # only keep largest connected component
    edges_coo = edge_index.detach().cpu().numpy().T
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(z)))
    g.add_edges_from(edges_coo.tolist())

    ccs = list(nx.connected_components(g))
    largest_cc = ccs[np.argmax([len(cc) for cc in ccs])]
    node_ids = np.asarray(list(largest_cc))

    z = z[node_ids]
    edges_cc = []
    edge_labels = []
    old2new = {old:new for new, old in enumerate(node_ids)}
    for idx2, (u, v) in enumerate(edges_coo):
        if u in node_ids and v in node_ids:
            edges_cc += [[old2new[u], old2new[v]]]
            edge_labels += [edge_type[idx2].item()]
        
    edge_index = np.asarray(edges_cc).T
    if useMin:
        return dfs_code.min_dfs_code_from_edgeindex(edge_index, z.tolist(), edge_labels)
    else:
        return dfs_code.rnd_dfs_code_from_edgeindex(edge_index, z.tolist(), edge_labels)
    

def computeChemicalValidity(dfs_codes):
    valid_list = []
    for code in dfs_codes:
        try:
            valid_list += [isValidMoleculeDFSCode(code)]
        except:
            valid_list += [False]
    valid = np.asarray(valid_list)
    return valid.sum()/len(valid)


def computeChemicalValidityAndNovelty(smiles, dfs_codes):
    valid_list = []
    same_list = []
    for code, sml in zip(dfs_codes, smiles):
        try:
            valid_list += [isValidMoleculeDFSCode(code)]
        except:
            valid_list += [False]
        try:
            if valid_list[-1]:
                smiles_orig = Mol2Smiles(Chem.MolFromSmiles(sml))
                smiles_rec = DFSCode2Smiles(code)
                same_list += [smiles_orig == smiles_rec]
        except:
            continue
    valid = np.asarray(valid_list)
    same = np.asarray(same_list)
    return valid.sum()/len(valid), same.sum()/len(same)
    

    
