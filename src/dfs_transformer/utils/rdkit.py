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

# chemprop feature specification
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(1, MAX_ATOMIC_NUM+1)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                   Chem.rdchem.ChiralType.CHI_OTHER],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}
# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FEATURES_SHAPES = {key:len(choices)+1 for key, choices in ATOM_FEATURES.items()}
ATOM_FEATURES_SHAPES['is_aromatic'] = 1
ATOM_FEATURES_SHAPES['mass'] = 1 

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FEATURES = {
    'bond_type': [-1, BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
    }
BOND_FEATURES_SHAPES = {key:len(choices)+1 for key, choices in BOND_FEATURES.items()}


def parseChempropAtomFeatures(features, true_values=False, missing_value=-1, padding_value=-1000):
    """    
    TODO: missing value -1 can conflict with formal charge value
    
    for a batch of sequences of atoms (nseqxnbatchxnfeat) parse the features 
    into a format that can be more easily be used for training / converting 
    the dfs code.
    
    feature dimension is assumed to be the last dimension

    Parameters
    ----------
    features : tensor, batch of sequences of atoms
    true_values: bool, whether to return true feature values or 0, ..., n_options

    Returns
    -------
    feature_dict : dict, all the parsed features
    """
    if len(features.shape) != 3:
        raise NotImplemented("currently only implemented for batches of sequences")
    features = features.clone()
    feature_dict = {}
    pos = 0
    mask = features[:, :, -1] == padding_value
    mask_missing = features[:, :, -1] == missing_value
    for fkey, foptions in ATOM_FEATURES.items():
        feature_dict[fkey] = torch.argmax(features[:, :, pos:pos+len(foptions)+1], dim=2).numpy()
        if true_values:
            feature_dict[fkey] = np.array(ATOM_FEATURES[fkey], dtype=object)[feature_dict[fkey]]
        feature_dict[fkey][mask] = padding_value
        feature_dict[fkey][mask_missing] = missing_value
        pos += len(foptions)+1
    feature_dict['is_aromatic'] = features[:, :, pos].numpy()
    feature_dict['mass'] = features[:, :, pos+1].numpy()
    return feature_dict


def parseChempropBondFeatures(features, true_values=False, missing_value=-1, padding_value=-1000):
    if len(features.shape) != 3:
        raise NotImplemented("currently only implemented for batches of sequences")
    features = features.clone()
    feature_dict = {}
    pos = 0
    mask = features[:, :, -1] == padding_value
    mask_missing = features[:, :, -1] == missing_value
    for fkey, foptions in BOND_FEATURES.items():
        feature_dict[fkey] = torch.argmax(features[:, :, pos:pos+len(foptions)+1], dim=2).numpy()
        if true_values:
            feature_dict[fkey] = np.array(BOND_FEATURES[fkey], dtype=object)[feature_dict[fkey]]
        feature_dict[fkey][mask] = padding_value
        feature_dict[fkey][mask_missing] = missing_value
        pos += len(foptions)+1
    return feature_dict


def FeaturizedDFSCodes2Dict(dfs_code, missing_value=-1, padding_value=-1000):
    dfs1_batch = dfs_code["dfs_from"]
    dfs2_batch = dfs_code["dfs_to"]
    atm1_batch = parseChempropAtomFeatures(dfs_code["atm_from"], true_values=False, missing_value=missing_value, padding_value=padding_value)
    atm2_batch = parseChempropAtomFeatures(dfs_code["atm_to"], true_values=False, missing_value=missing_value, padding_value=padding_value)
    bnd_batch = parseChempropBondFeatures(dfs_code["bnd"], true_values=False, missing_value=missing_value, padding_value=padding_value)
    bnd_dict = {key: torch.tensor(val, dtype=torch.long) for key, val in bnd_batch.items()}
    atm1_dict = {key+'_from': torch.tensor(val, dtype=torch.long) for key, val in atm1_batch.items()}
    atm2_dict = {key+'_to': torch.tensor(val, dtype=torch.long) for key, val in atm2_batch.items()}
    d = {"dfs_from": dfs1_batch,
         "dfs_to": dfs2_batch}
    d.update(atm1_dict)
    d.update(atm2_dict)
    d.update(bnd_dict)
    return d
    
    
def FeaturizedDFSCodes2Nx(dfs_code, padding_value=-1000):
    if torch.any(dfs_code["dfs_from"] == -1):
        raise NotImplemented("does not account for missing values yet")
    dfs1_batch = dfs_code["dfs_from"]
    dfs2_batch = dfs_code["dfs_to"]
    atm1_batch = parseChempropAtomFeatures(dfs_code["atm_from"], true_values=True)
    atm2_batch = parseChempropAtomFeatures(dfs_code["atm_to"], true_values=True)
    bnd_batch = parseChempropBondFeatures(dfs_code["bnd"], true_values=True)
    
    graphs = []
    for batch_id, (dfs1, dfs2) in enumerate(zip(dfs1_batch.T, dfs2_batch.T)):
        G = nx.Graph()
        nodes_added = set()
        for edge_id, (d1, d2) in enumerate(zip(dfs1, dfs2)):
            if d1 == padding_value or d2 == padding_value:
                break
            if d1 not in nodes_added:
                G.add_node(d1.item(),
                           atomic_num=atm1_batch['atomic_num'][edge_id, batch_id],
                           formal_charge=atm1_batch['formal_charge'][edge_id, batch_id],
                           chiral_tag=atm1_batch['chiral_tag'][edge_id, batch_id],
                           hybridization=atm1_batch['hybridization'][edge_id, batch_id],
                           num_total_hs=atm1_batch['num_Hs'][edge_id, batch_id],
                           is_aromatic=bool(atm1_batch['is_aromatic'][edge_id, batch_id] == 1.))
                nodes_added.add(d1)
            if d2 not in nodes_added:
                G.add_node(d2.item(),
                           atomic_num=atm2_batch['atomic_num'][edge_id, batch_id],
                           formal_charge=atm2_batch['formal_charge'][edge_id, batch_id],
                           chiral_tag=atm2_batch['chiral_tag'][edge_id, batch_id],
                           hybridization=atm2_batch['hybridization'][edge_id, batch_id],
                           num_total_hs=atm2_batch['num_Hs'][edge_id, batch_id],
                           is_aromatic=bool(atm2_batch['is_aromatic'][edge_id, batch_id] == 1.))
                nodes_added.add(d2)
                
            G.add_edge(d1.item(), d2.item(), 
                       bond_type=bnd_batch["bond_type"][edge_id, batch_id])
        graphs += [G]
    return graphs
            

def Mol2Nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_total_hs=atom.GetTotalNumHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def Nx2Mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_total_hs = nx.get_node_attributes(G, 'num_total_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)
    
    # this is a workaround because we precomputed num_total_hs in the existing pipeline.
    # so using this we avoid touching the feature extraction 
    mol.UpdatePropertyCache()
    for node in G.nodes():
        a = mol.GetAtomWithIdx(node_to_idx[node])
        a.SetNumExplicitHs(num_total_hs[node] - a.GetNumImplicitHs())

    Chem.SanitizeMol(mol)
    return mol


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
        a=Chem.Atom(int(atomic_num))
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
    
    
def Smiles2Mol(smiles):
    return Chem.MolFromSmiles(smiles)


def Mol2Smiles(mol):
    return Chem.MolToSmiles(mol)


def DFSCode2Smiles(dfs_code):
    return Chem.MolToSmiles(Graph2Mol(*DFSCode2Graph(dfs_code)))


def isValidMoleculeDFSCode(dfs_code, verbose=False):
    return isValid(Graph2Mol(*DFSCode2Graph(dfs_code)), verbose=verbose)


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
    return valid.sum()/len(valid), valid


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

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

    

    
