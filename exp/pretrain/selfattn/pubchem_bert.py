#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import wandb
import os
import torch.optim as optimizers


# In[3]:


import dfs_code
from torch_geometric.data import InMemoryDataset, Data
import pickle
import torch
import torch.nn as nn
import tqdm
import copy
import pandas as pd
#torch.multiprocessing.set_sharing_strategy('file_system') # this is important on local machine
#def set_worker_sharing_strategy(worker_id: int) -> None:
#    torch.multiprocessing.set_sharing_strategy('file_system')


# In[4]:


import sys
sys.path = ['../../../src_old'] + sys.path
from dfs_transformer import EarlyStopping, DFSCodeSeq2SeqFC, smiles2graph


# In[5]:


wandb.init(project='pubchem-bert', entity='chrisxx', name='bert10k')

config = wandb.config
config.mode = "min2min" #rnd2rnd
config.fraction_missing = 0.15
config.n_atoms = 118
config.n_bonds = 5
config.emb_dim = 120
config.nhead = 12
config.nlayers = 6
config.max_nodes = 250
config.max_edges = 500
config.dim_feedforward = 2048
config.n_files = 10
config.n_splits = 1
config.n_iter_per_split = 10000
config.lr = 0.00003
config.n_epochs = 10000
config.lr_adjustment_period = 500
config.patience = 5
config.factor = 0.5
config.minimal_lr = 6e-8
config.batch_size = 50
config.accumulate_grads = 2
config.valid_patience = 100
config.valid_minimal_improvement=0.00
config.model_dir = "../../../models/pubchem/mini10k/features_selfattention/medium/"
#config.data_dir = "/mnt/project/pubchem_noH/"
config.data_dir = "../../../results/pubchem/mini10k/"
config.pretrained_dir = None# "../../../models/pubchem_mini/features_selfattention/medium/"#"../../models/chembl/better_transformer/medium/"
config.num_workers = 0
config.prefetch_factor = 2
config.persistent_workers = False
config.load_last = True
config.dformat = "json"
config.gpu_id = 0


# In[6]:


os.makedirs(config.model_dir, exist_ok=True)


# In[7]:


path = config.data_dir


# In[8]:


class PubChem(Dataset):
    """PubChem dataset of molecules and minimal DFS codes."""
    def __init__(self, path=path, n_used = 8, n_splits = 64, max_nodes=config.max_nodes,
                 max_edges=config.max_edges, useHs=False, addLoops=False, memoryEfficient=False,
                 transform=None, n_mols_per_dataset=np.inf, dformat='json'):
        self.path = path
        self.data = []
        self.path = path
        self.n_used = n_used
        self.n_splits = n_splits
        self.useHs = useHs
        self.addLoops = addLoops
        self.n_mols_per_dataset = n_mols_per_dataset
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.dformat = dformat
        self.prepare()
        
        
    def prepare(self):
        codes_all = {}
        d_all = {}
        i2didx = {}
        perm = np.random.permutation(self.n_splits)
        for i in tqdm.tqdm(perm[:self.n_used]):
            dname = glob.glob(self.path+"/%d/min_dfs_codes_split*.json"%(i+1))[0]
            didx = int(dname.split("split")[-1][:-5])
            dname2 = self.path+"/%d/data_split%d.%s"%(i+1, didx, self.dformat)
            with open(dname, 'r') as f:
                codes = json.load(f)
                for key, val in codes.items():
                    codes_all[key] = val
            if self.dformat == "json":
                with open(dname2, 'r') as f:
                    d_dict = json.load(f)
                    for key, val in d_dict.items():
                        d_all[key] = val
            elif self.dformat == "pkl":
                with open(dname2, 'rb') as f:
                    d_dict = pickle.load(f)
                    for key, val in d_dict.items():
                        d_all[key] = val
            else:
                raise ValueError("unsupported dformat")
        
        for smiles, code in tqdm.tqdm(codes_all.items()):
            if len(self.data) > self.n_mols_per_dataset:
                break
            if code['min_dfs_code'] is not None and len(code['min_dfs_code']) > 1:
                d = d_all[smiles]
                if len(d['z']) > self.max_nodes:
                    continue
                if len(d['edge_attr']) > 2*self.max_edges:
                    continue
                
                
                z = torch.tensor(d['z'], dtype=torch.long)
                
                data_ = Data(z=z,
                             edge_attr=torch.tensor(d['edge_attr']),
                             edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
                             min_dfs_code=torch.tensor(code['min_dfs_code']),
                             min_dfs_index=torch.tensor(code['dfs_index'], dtype=torch.long),
                             smiles=smiles,
                             node_features=torch.tensor(d['atom_features'], dtype=torch.float32),
                             edge_features=torch.tensor(d['bond_features'], dtype=torch.float32))
                self.data += [data_]   
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# In[9]:


def collate_fn_features(dlist):
    node_batch = [] 
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
    return rnd_code_batch, node_batch, edge_batch, min_code_batch


# In[10]:


ngpu=1
device = torch.device('cuda:%d'%config.gpu_id if (torch.cuda.is_available() and ngpu > 0) else 'cpu')


# In[11]:


to_cuda = lambda T: [t.to(device) for t in T]


# In[12]:


dataset = PubChem(n_used = 1, n_splits=config.n_files, dformat=config.dformat)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False, collate_fn=collate_fn_features,
                   num_workers=config.num_workers, prefetch_factor=config.prefetch_factor, 
                    persistent_workers=config.persistent_workers)


# In[13]:


data = next(iter(loader))


# In[14]:


n_node_features = data[1][0].shape[1]
n_edge_features = data[2][0].shape[1]


# In[15]:


print(n_node_features, n_edge_features)


# In[16]:


codes = data[-1]


# In[17]:


codes


# In[18]:


def BERTize(codes):
    inputs = []
    targets = []
    for code in codes:
        n = len(code)
        perm = np.random.permutation(n)
        target_idx = perm[:int(config.fraction_missing*n)]
        input_idx = perm[int(config.fraction_missing*n):]
        inp = code.clone()
        target = code.clone()
        target[input_idx] = -1
        inp[target_idx] = -1
        inputs += [inp]
        targets += [target]
    return inputs, targets


# In[19]:


inputs, targets = BERTize(codes)


# In[20]:


inputs


# In[21]:


targets


# In[22]:


model = DFSCodeSeq2SeqFC(n_atoms=config.n_atoms,
                         n_bonds=config.n_bonds, 
                         emb_dim=config.emb_dim, 
                         nhead=config.nhead, 
                         nlayers=config.nlayers, 
                         max_nodes=config.max_nodes, 
                         max_edges=config.max_edges,
                         atom_encoder=nn.Linear(n_node_features, config.emb_dim), 
                         bond_encoder=nn.Linear(n_edge_features, config.emb_dim))


# In[23]:


if config.pretrained_dir is not None:
    model.load_state_dict(torch.load(config.pretrained_dir+'checkpoint.pt'))


# In[24]:


if config.load_last:
    model.load_state_dict(torch.load(config.model_dir+'checkpoint.pt'))


# In[25]:


optim = optimizers.Adam(model.parameters(), lr=config.lr)

lr_scheduler = optimizers.lr_scheduler.ReduceLROnPlateau(optim, mode='min', verbose=True, patience=config.patience, factor=config.factor)
#lr_scheduler = optimizers.lr_scheduler.ExponentialLR(optim, gamma=config.factor)

early_stopping = EarlyStopping(patience=config.valid_patience, delta=config.valid_minimal_improvement,
                              path=config.model_dir+'checkpoint.pt')
bce = torch.nn.BCEWithLogitsLoss()
ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
softmax = nn.Softmax(dim=2)


# In[26]:


model.to(device)


# In[ ]:


try:
    for epoch in range(config.n_epochs):
        epoch_loss = 0
        for split in range(config.n_splits):
            
            n_ids = config.n_files//config.n_splits
            dataset = PubChem(n_used = n_ids, n_splits = config.n_files, dformat=config.dformat)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False, 
                                collate_fn=collate_fn_features,
                   num_workers=config.num_workers, prefetch_factor=config.prefetch_factor, 
                    persistent_workers=config.persistent_workers)
            for j in range(config.n_iter_per_split):
                pbar = tqdm.tqdm(loader)
                for i, data in enumerate(pbar):
                    if i % config.accumulate_grads == 0: #bei 0 wollen wir das
                        optim.zero_grad()
                    rndc, nattr, eattr, minc = data
                    if config.mode == "min2min":
                        code = to_cuda(minc)
                    elif config.mode == "rnd2rnd":
                        code = to_cuda(rndc)
                    else:
                        raise ValueError("unrecognized config.mode")
                    nattr = to_cuda(nattr)
                    eattr = to_cuda(eattr)
                    #prepare inputs
                    inputs, targets = BERTize(code)
                    #prepare labels
                    targetc = [torch.cat((c, (-1)*torch.ones((1, 8), dtype=torch.long, device=device)), dim=0) for c in targets]
                    targetc_seq = nn.utils.rnn.pad_sequence(targetc, padding_value=-1)
                    #prediction
                    dfs1, dfs2, atm1, atm2, bnd, eos = model(inputs, nattr, eattr)
                    pred_dfs1 = torch.reshape(dfs1, (-1, config.max_nodes))
                    pred_dfs2 = torch.reshape(dfs2, (-1, config.max_nodes))
                    pred_atm1 = torch.reshape(atm1, (-1, config.n_atoms))
                    pred_atm2 = torch.reshape(atm2, (-1, config.n_atoms))
                    pred_bnd = torch.reshape(bnd, (-1, config.n_bonds))
                    tgt_dfs1 = targetc_seq[:, :, 0].view(-1)
                    tgt_dfs2 = targetc_seq[:, :, 1].view(-1)
                    tgt_atm1 = targetc_seq[:, :, 2].view(-1)
                    tgt_atm2 = targetc_seq[:, :, 4].view(-1)
                    tgt_bnd = targetc_seq[:, :, 3].view(-1)
                    loss = ce(pred_dfs1, tgt_dfs1) 
                    loss += ce(pred_dfs2, tgt_dfs2)
                    loss += ce(pred_atm1, tgt_atm1)
                    loss += ce(pred_bnd, tgt_bnd)
                    loss += ce(pred_atm2, tgt_atm2)
                    loss.backward()
                    if (i+1) % config.accumulate_grads == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optim.step() # bei 2 wollen wir das
                    epoch_loss = (epoch_loss*i + loss.item())/(i+1)
                    mask = tgt_dfs1 != -1
                    n_tgts = torch.sum(mask)
                    acc_dfs1 = (torch.argmax(pred_dfs1[mask], axis=1) == tgt_dfs1[mask]).sum()/n_tgts
                    acc_dfs2 = (torch.argmax(pred_dfs2[mask], axis=1) == tgt_dfs2[mask]).sum()/n_tgts
                    acc_atm1 = (torch.argmax(pred_atm1[mask], axis=1) == tgt_atm1[mask]).sum()/n_tgts
                    acc_atm2 = (torch.argmax(pred_atm2[mask], axis=1) == tgt_atm2[mask]).sum()/n_tgts
                    acc_bnd = (torch.argmax(pred_bnd[mask], axis=1) == tgt_bnd[mask]).sum()/n_tgts
                    curr_lr = list(optim.param_groups)[0]['lr']
                    wandb.log({'loss':epoch_loss, 'learning rate':curr_lr,
                               'acc-dfs1':acc_dfs1, 'acc-dfs2':acc_dfs2, 
                               'acc-atm1':acc_atm1, 'acc-atm2':acc_atm2,
                               'acc-bnd':acc_bnd})
                    pbar.set_description('Epoch %d: CE %2.6f accs: %2.2f %2.2f %2.2f %2.2f %2.2f'%(epoch+1, 
                                                                                                   epoch_loss, 
                                                                                                   100*acc_dfs1,
                                                                                                   100*acc_dfs2,
                                                                                                   100*acc_atm1,
                                                                                                   100*acc_bnd,
                                                                                                   100*acc_atm2))

                    if i % config.lr_adjustment_period == 0:
                        early_stopping(epoch_loss, model)
                        lr_scheduler.step(epoch_loss)
                        if early_stopping.early_stop:
                            break

                        if curr_lr < config.minimal_lr:
                            break
                            
                            
            if early_stopping.early_stop:
                break

            if curr_lr < config.minimal_lr:
                break
                
                
            del dataset
            del loader
        

except KeyboardInterrupt:
    torch.save(model.state_dict(), config.model_dir+'_keyboardinterrupt.pt')
    print('keyboard interrupt caught')


# In[ ]:




