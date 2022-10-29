#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:36:51 2022

@author: chrisw
"""

from torch_geometric.data import Data, Dataset
import torch
import tqdm
import networkx as nx
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import json
import pandas as pd
from joblib import Parallel, delayed
import dfs_code


def graph2labelledgraph(graph, use_dummy=False, use_degree=True):
    graph = deepcopy(graph)
    node_ids = np.unique(graph).tolist()
    dummy = max(node_ids) + 1
    
    edge_labels = len(graph)*[0]
    edgeindex = []
    for e in graph:
        edgeindex += [e]
        edgeindex += [[e[1], e[0]]]
        edge_labels += [0]
    
    if use_dummy:
        for idx in node_ids:
            edgeindex += [[idx, dummy], [dummy, idx]]
            edge_labels += [1, 1]
        node_ids += [dummy]
    
    edgeindex = np.asarray(edgeindex).T
    
    node_labels = []
    for idx in node_ids:
        if use_degree:
            node_labels += [(edgeindex[0] == idx).sum()]
        else:
            node_labels += [1]
    return edgeindex, node_labels, edge_labels


class KarateClubDataset(Dataset):
    def __init__(self, graph_file, label_file, max_n = None, max_edges=np.inf, 
                 features=['deg', 'triangles', 'eccentricity'], min_dfs_flag=False):
        super().__init__()
        self.graph_file = graph_file
        self.label_file = label_file
        with open(graph_file, 'r') as f:
            self.graph_dict = json.load(f)
        self.label_df = pd.read_csv(label_file)
        self.data = []
        self.maxn = max_n
        self.max_edges = max_edges
        self.features = features
        self.min_dfs_flag = min_dfs_flag
        self.preprocess()
    
    def preprocess(self):
        use_degree = 'deg' in self.features
        use_triangles = 'triangles' in self.features
        use_eccentricity = 'eccentricity' in self.features
        maxn = self.maxn
        maxdegree = 0
        max_edges = 0
        edgeindex_list = []
        vlabels_list = []
        elabels_list = []
        label_list = []
        #res_list = Parallel(n_jobs=self.n_jobs, prefer='threads')(
        #    delayed(graph2labelledgraph)(graph, use_degree=use_degree) 
        #    for graph in list(self.graph_dict.values()))
        
        for idx, graph in tqdm.tqdm(enumerate(self.graph_dict.values())):
            res = graph2labelledgraph(graph, use_degree=use_degree) 
            if len(res[2])//2 > self.max_edges:
                continue
            if maxn is not None:
                if idx >= maxn:
                    break
            edgeindex_list += [res[0]]
            vlabels_list += [res[1]]
            elabels_list += [res[2]]
            label_list += [self.label_df['target'][idx]]
            maxdegree = max(maxdegree, max(res[1]))
            max_edges = max(max_edges, len(res[2])//2)
        self.maxdegree = maxdegree
        self.max_edges = max_edges
        
        def compute_datapoint(edgeindex, vlabels, elabels, label):
            node_features = F.one_hot(torch.tensor(vlabels), num_classes=maxdegree+1).float()
            feats = [node_features]
            if use_triangles or use_eccentricity:
                g = nx.Graph()
                g.add_edges_from(edgeindex.T.tolist())
                if use_triangles:
                    clustering_coefs = nx.triangles(g)
                    clustering_coefs = torch.tensor([clustering_coefs[idx] for idx in range(len(clustering_coefs))])
                    clust_feats = F.one_hot(clustering_coefs, num_classes=self.max_edges).float()
                    feats += [clust_feats]
                if use_eccentricity:
                    eccentricity = nx.eccentricity(g)
                    eccentricity = torch.tensor([eccentricity[idx] for idx in range(len(eccentricity))], dtype=torch.long)
                    ecc_feats = F.one_hot(eccentricity, num_classes=self.max_edges).float()
                    feats += [ecc_feats]
                node_features = torch.cat(feats, dim=1)
            
            edge_features = F.one_hot(torch.tensor(elabels), num_classes=2).float()
            if self.min_dfs_flag:
                #vlabels = eccentricity.numpy().tolist()
                #vlabels = clustering_coefs.numpy().tolist()
                vlabels2 = []
                for deg, n_tri, ecc in zip(vlabels, clustering_coefs, eccentricity):
                    vlabels2 += [deg.item()]# + self.max_edges*n_tri.item() + self.max_edges**2*ecc.item()]
                
                elabels2 = [vlabels[edge[0]]+vlabels[edge[1]] - 2 for edge in edgeindex.T]
                #elabels2 = elabels
                #print(len(elabels2), edgeindex.shape)
                print(edgeindex)
                print(vlabels2)
                print(elabels2)
                code, index = dfs_code.min_dfs_code_from_edgeindex(edgeindex, vlabels2, elabels2)
                print(code)
                print(len(code))
                return Data(**{"edge_index": torch.tensor(edgeindex, dtype=torch.long),
                           "edge_features": edge_features,
                           "node_features": node_features,
                           "node_labels": torch.tensor(vlabels, dtype=torch.long),
                           "edge_labels": torch.tensor(elabels, dtype=torch.long),
                           "x": node_features, 
                           "y": torch.tensor(label, dtype=torch.long),
                           "num_nodes": len(node_features),
                           "min_dfs_code": torch.tensor(np.asarray(code)),
                           "min_dfs_index": torch.tensor(np.asarray(index), dtype=torch.long)})
            
            return Data(**{"edge_index": torch.tensor(edgeindex, dtype=torch.long),
                           "edge_features": edge_features,
                           "node_features": node_features,
                           "node_labels": torch.tensor(vlabels, dtype=torch.long),
                           "edge_labels": torch.tensor(elabels, dtype=torch.long),
                           "x": node_features, 
                           "y": torch.tensor(label, dtype=torch.long),
                           "num_nodes": len(node_features)})
        
        for edgeindex, vlabels, elabels, label in tqdm.tqdm(zip(edgeindex_list, vlabels_list, elabels_list, label_list)):
            self.data += [compute_datapoint(edgeindex, vlabels, elabels, label)]
        #self.data = Parallel(n_jobs=self.n_jobs, prefer='threads')(
        #    delayed(compute_datapoint)(edgeindex, vlabels, elabels, label)
        #    for edgeindex, vlabels, elabels, label in zip(edgeindex_list, vlabels_list, elabels_list, label_list))
        

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
