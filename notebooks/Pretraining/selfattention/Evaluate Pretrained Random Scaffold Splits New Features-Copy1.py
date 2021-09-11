#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import deepchem as dc
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
import random
import tqdm
import copy
import pandas as pd
#torch.multiprocessing.set_sharing_strategy('file_system') # this is important
# ulimit -n 500000
#def set_worker_sharing_strategy(worker_id: int) -> None:
#    torch.multiprocessing.set_sharing_strategy('file_system')


# In[4]:


import sys
sys.path = ['/home/chrisw/Documents/projects/2021/graph-transformer/src_old'] + sys.path
from dfs_transformer import EarlyStopping, DFSCodeSeq2SeqFC, Deepchem2TorchGeometric, collate_minc_rndc_features_y, DFSCodeSeq2SeqDecoder


# In[5]:


wandb.init(project='moleculenet10', entity='chrisxx', name='bert-10M-wdecay')

config = wandb.config
config.n_atoms = 118
config.n_bonds = 5
config.emb_dim = 120
config.nhead = 12
config.nlayers = 6
config.use_Hs = False
config.max_nodes = 250
config.max_edges = 500
config.features = "chemprop" # chemprop 133 node features 14 edge featuers, old 127 5, none 118 5
config.n_node_features = 133
config.n_edge_features = 14
config.max_nodes_data = np.inf
config.max_edges_data = np.inf
config.onlyRandom = False
config.use_min = True
config.dim_feedforward = 2048
config.lr = 0.00003 # 10x smaller than the learning rate of the pretraining seems to be good
config.pretrained_weight_decay = 0.1
config.clip_gradient = 0.5
config.n_epochs = 25
config.patience = 3
config.factor = 0.5
config.minimal_lr = 6e-8
config.batch_size = 50
config.accumulate_grads = 1
config.valid_patience = 25
config.valid_minimal_improvement = 0.00
config.pretrained_dir = "../../../models/pubchem10M/features_selfattention/medium/converged/"
config.num_workers = 0
config.load_last = True
config.dataset = 'bbbp' # supported 'bbbp', 'clintox', 'hiv', 'tox21'
config.model_dir = '../../../models/moleculenet/%s/pubchem_mini/features_selfattention/medium/2/'%config.dataset
config.data_dir = "../../../datasets/mymoleculenet/%s/"%config.dataset
config.loaddir = "../../../results/mymoleculenet_plus_features/%s/1/"%config.dataset
config.seed = 123


# In[6]:


random.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)


# In[7]:


os.makedirs(config.model_dir, exist_ok=True)


# In[8]:


ngpu=1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')


# In[9]:


to_cuda = lambda T: [t.cuda() for t in T]


# In[10]:


def get_random_scaffold_split(dataset):
    splitter = dc.splits.ScaffoldSplitter()
    shuffled = dataset.complete_shuffle()
    trainidx, valididx, testidx = splitter.split(shuffled)
    train = shuffled.select(trainidx)
    valid = shuffled.select(valididx)
    test = shuffled.select(testidx)
    return train, valid, test


# In[11]:


if config.data_dir is None:
    if config.dataset == 'clintox':
        tasks, datasets, transformers = dc.molnet.load_clintox(reload=False, featurizer=dc.feat.RawFeaturizer(True), splitter=None)
    elif config.dataset == 'tox21':
        tasks, datasets, transformers = dc.molnet.load_tox21(reload=False, featurizer=dc.feat.RawFeaturizer(True), splitter=None)
    elif config.dataset == 'hiv':
        tasks, datasets, transformers = dc.molnet.load_hiv(reload=False, featurizer=dc.feat.RawFeaturizer(True), splitter=None)
    elif config.dataset == 'bbbp':
        tasks, datasets, transformers = dc.molnet.load_bbbp(reload=False, featurizer=dc.feat.RawFeaturizer(True), splitter=None)


# In[12]:


from sklearn.metrics import roc_auc_score, average_precision_score

def score(loader, model, model_head, use_min=False):
    val_roc = 0
    with torch.no_grad():
        full_preds, target = [], []
        for batch in tqdm.tqdm(loader):
            rndc, minc, z, eattr, y = batch
            if use_min:
                code = to_cuda(minc)
            else:
                code = to_cuda(rndc)
            y = y.to(device)
            features = model.encode(code, to_cuda(z), to_cuda(eattr)) # not clear whether to use minc or randc
            pred = sigmoid(model_head(features)).squeeze()
            
            pred_np = pred.detach().cpu().numpy().tolist()
            y_np = y.detach().cpu().numpy().tolist()
            full_preds += pred_np
            target += y_np

        target = np.asarray(target)
        full_preds = np.asarray(full_preds)
        roc = roc_auc_score(target, full_preds)
        prc = average_precision_score(target, full_preds)
    return roc, prc


# In[13]:


roc_avgs = []
prc_avgs = []
for rep in range(10):
    model = DFSCodeSeq2SeqFC(n_atoms=config.n_atoms,
                         n_bonds=config.n_bonds, 
                         emb_dim=config.emb_dim, 
                         nhead=config.nhead, 
                         nlayers=config.nlayers, 
                         max_nodes=config.max_nodes, 
                         max_edges=config.max_edges,
                         atom_encoder=nn.Linear(config.n_node_features, config.emb_dim), 
                         bond_encoder=nn.Linear(config.n_edge_features, config.emb_dim))
    if config.load_last:
        model.load_state_dict(torch.load(config.pretrained_dir+'checkpoint.pt', map_location=device))
        print('loaded pretrained model')
    
    model_head = nn.Linear(5*config.emb_dim, 1)
    model_head.to(device)
    model.to(device)
    w_init = nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    
    if config.data_dir is None:
        trainset, validset, testset = get_random_scaffold_split(datasets[0])
        train_X, train_y = trainset.X, trainset.y[:, -1]
        valid_X, valid_y = validset.X, validset.y[:, -1]
        test_X, text_y = testset.X, testset.y[:, -1]
    else:
        trainset = pd.read_csv(config.data_dir+"%d/train.csv"%rep)
        validset = pd.read_csv(config.data_dir+"%d/valid.csv"%rep)
        testset = pd.read_csv(config.data_dir+"%d/test.csv"%rep)
        train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
        valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
        test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
        
    
    collate_fn = collate_minc_rndc_features_y

    traindata = Deepchem2TorchGeometric(train_X, train_y, loaddir=config.loaddir, features=config.features, trimEdges=False, useHs=config.use_Hs, onlyRandom=config.onlyRandom, max_nodes=config.max_nodes_data, max_edges=config.max_edges_data)
    validdata = Deepchem2TorchGeometric(valid_X, valid_y, loaddir=config.loaddir, features=config.features, trimEdges=False, useHs=config.use_Hs, onlyRandom=config.onlyRandom, max_nodes=config.max_nodes_data, max_edges=config.max_edges_data)
    testdata = Deepchem2TorchGeometric(test_X, test_y, loaddir=config.loaddir, features=config.features, trimEdges=False, useHs=config.use_Hs, onlyRandom=config.onlyRandom, max_nodes=config.max_nodes_data, max_edges=config.max_edges_data)
    trainloader = DataLoader(traindata, batch_size=config.batch_size, shuffle=True, pin_memory=False, collate_fn=collate_fn, num_workers=config.num_workers)
    validloader = DataLoader(validdata, batch_size=config.batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn, num_workers=config.num_workers)
    testloader = DataLoader(testdata, batch_size=config.batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn, num_workers=config.num_workers)
    optim = optimizers.Adam(list(model.parameters()) + list(model_head.parameters()), lr=config.lr)
    lr_scheduler = optimizers.lr_scheduler.ReduceLROnPlateau(optim, mode='min', verbose=True, patience=config.patience, factor=config.factor)
    early_stopping_model = EarlyStopping(patience=config.valid_patience, delta=config.valid_minimal_improvement,
                                  path=config.model_dir+'checkpoint_model.pt')
    early_stopping_head = EarlyStopping(patience=config.valid_patience, delta=config.valid_minimal_improvement,
                                  path=config.model_dir+'checkpoint_head.pt')
    bce = torch.nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    
    valid_scores = []
    try:
        for epoch in range(config.n_epochs):  
            epoch_loss = 0
            pbar = tqdm.tqdm(trainloader)
            model.train()
            for i, data in enumerate(pbar):
                if i % config.accumulate_grads == 0: 
                    optim.zero_grad()
                rndc, minc, z, eattr, y = data
                if config.use_min:
                    code = to_cuda(minc)
                else:
                    code = to_cuda(rndc)
                z = to_cuda(z)
                eattr = to_cuda(eattr)
                y = y.to(device)
                #prediction
                features = model.encode(code, z, eattr)
                prediction = model_head(features).squeeze()
                loss = bce(prediction, y)
                if config.pretrained_weight_decay > 0:
                    w_curr = nn.utils.parameters_to_vector(model.parameters())
                    loss += config.pretrained_weight_decay * 0.5 * torch.sum((w_curr - w_init)**2)
                loss.backward()
                
                if (i + 1) % config.accumulate_grads == 0: 
                    if config.clip_gradient is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradient)
                    optim.step()
                epoch_loss = (epoch_loss*i + loss.item())/(i+1)
                pbar.set_description('Epoch %d: CE %2.6f'%(epoch+1, epoch_loss))
            model.eval()
            roc_auc_valid, prc_auc_valid = score(validloader, model, model_head, config.use_min) 
            valid_scores += [roc_auc_valid]
            lr_scheduler.step(epoch_loss)
            early_stopping_model(-roc_auc_valid, model)
            early_stopping_head(-roc_auc_valid, model_head)
            curr_lr = list(optim.param_groups)[0]['lr']
            wandb.log({'loss':epoch_loss, 'roc_valid':roc_auc_valid, 'prc_valid':prc_auc_valid, 'learning rate':curr_lr}, 
                      step=config.n_epochs*rep + epoch)
            print('ROCAUC',roc_auc_valid, 'PRCAUC', prc_auc_valid)

            if early_stopping_model.early_stop:
                break

            if curr_lr < config.minimal_lr:
                break

    except KeyboardInterrupt:
        torch.save(model.state_dict(), config.model_dir+'_keyboardinterrupt.pt')
        print('keyboard interrupt caught')
        
    model.load_state_dict(torch.load(config.model_dir+'checkpoint_model.pt'))
    model_head.load_state_dict(torch.load(config.model_dir+'checkpoint_head.pt'))
    model.eval()
    roc_auc_valid, prc_auc_valid = score(testloader, model, model_head, config.use_min) 
    wandb.log({'roc_test':roc_auc_valid, 'prc_test':prc_auc_valid}, step=config.n_epochs*(rep+1))
    
    roc, prc = score(testloader, model, model_head, True)
    wandb.log({'roc_test_min':roc, 'prc_test_min':prc}, step=config.n_epochs*(rep+1))
    print('ROC, PRC VALID', score(validloader, model, model_head, True))
    print('ROC, PRC TEST', score(testloader, model, model_head, True))
    
    avg_roc = 0
    avg_prc = 0
    for i in range(20):
        roc, prc = score(testloader, model, model_head, config.use_min)
        avg_roc += roc
        avg_prc += prc
    avg_roc /= 20
    avg_prc /= 20 
    roc_avgs += [avg_roc]
    prc_avgs += [avg_prc]
    wandb.log({'roc_test_avg20':avg_roc, 'prc_test_avg20':avg_prc}, step=config.n_epochs*(rep+1))
    print('ROC, PRC TEST', avg_roc, avg_prc)
    del model
    del model_head
wandb.log({'roc_test_avgavg':np.mean(roc_avgs), 'prc_test_avgavg':np.mean(prc_avgs)}, step=config.n_epochs*(rep+1))
wandb.log({'roc_test_avgstd':np.std(roc_avgs), 'prc_test_avgstd':np.std(prc_avgs)}, step=config.n_epochs*(rep+1))
wandb.run.summary["roc_test_mean"] = np.mean(roc_avgs) 
wandb.run.summary["roc_test_std"] = np.std(roc_avgs)
wandb.run.summary["prc_test_mean"] = np.mean(prc_avgs)
wandb.run.summary["prc_test_std"] = np.std(prc_avgs)


# In[ ]:




