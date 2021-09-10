#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:10:20 2021

@author: chrisw
"""
import os
import warnings
import os.path as osp
from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
import torch.nn as nn
import numpy as np

from torch_scatter import scatter
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn import radius_graph, MessagePassing, GlobalAttention
import torch_geometric.nn as tgnn

from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock,\
    CFConv, ShiftedSoftplus


qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class MySchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """


    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50, cutoff=10.0,
                 readout='add', dipole=False, mean=None, std=None,
                 atomref=None, use_dfs_codes=False, dfs_before_interaction=False):
        super(MySchNet, self).__init__()

        import ase

        self.dfs_before_interaction = dfs_before_interaction
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)
            
        if self.readout == 'attention':
            self.gate = Sequential(Linear(hidden_channels, hidden_channels // 2),
                                   ShiftedSoftplus(),
                                   Linear(hidden_channels // 2, 1))
            self.fc_out = Sequential(self.lin1, self.act, self.lin2)
            self.attention = GlobalAttention(self.gate, None)
        
        self.dfs_emb = None
        if use_dfs_codes:
            self.dfs_emb = Linear(29, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    
    def forward(self, z, pos, batch=None, dfs=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        if self.dfs_before_interaction:
            if dfs is not None and self.dfs_emb is not None:
                h += self.dfs_emb(dfs)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        
        if not self.dfs_before_interaction:
            if dfs is not None and self.dfs_emb is not None:
                h += self.dfs_emb(dfs)
        
        if self.readout == 'attention':
            h = self.attention(h, batch)
            out = self.fc_out(h)
            
            if self.mean is not None and self.std is not None:
                out = out * self.std + self.mean
    
            if self.atomref is not None:
                out = out + tgnn.global_add_pool(self.atomref(z), batch)
    
            if self.scale is not None:
                out = self.scale * out
            
        else:
            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)
    
            if self.dipole:
                # Get center of mass.
                mass = self.atomic_mass[z].view(-1, 1)
                c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
                h = h * (pos - c[batch])
    
            if not self.dipole and self.mean is not None and self.std is not None:
                h = h * self.std + self.mean
    
            if not self.dipole and self.atomref is not None:
                h = h + self.atomref(z)
    
            out = scatter(h, batch, dim=0, reduce=self.readout)
    
            if self.dipole:
                out = torch.norm(out, dim=-1, keepdim=True)
    
            if self.scale is not None:
                out = self.scale * out

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')
