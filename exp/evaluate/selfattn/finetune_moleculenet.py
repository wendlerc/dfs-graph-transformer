#!/usr/bin/env python
# coding: utf-8
import numpy as np
from torch.utils.data import DataLoader
import wandb
import os
import torch.optim as optimizers
import torch
import torch.nn as nn
import random
import tqdm
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path = ['./src'] + sys.path
from dfs_transformer import EarlyStopping, DFSCodeSeq2SeqFC, Deepchem2TorchGeometric, collate_minc_rndc_features_y

# wandb.save("config.yaml", base_path=".hydra/")

def score(loader, model, model_head, use_min=False):
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

datasets = ['bbbp', 'clintox', 'tox21', 'hiv']

for dataset in datasets:
    wandb.init(project='moleculenet10-clean', entity='chrisxx', name='bert-10M')
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
    config.lr = 0.000003 # 10x smaller than the learning rate of the pretraining seems to be good
    config.pretrained_weight_decay = 0#0.1
    config.clip_gradient = 0.5
    config.n_epochs = 25
    config.patience = 3
    config.factor = 0.5
    config.minimal_lr = 6e-8
    config.batch_size = 50
    config.accumulate_grads = 1
    config.valid_patience = 25
    config.valid_minimal_improvement = 0.00
    config.pretrained_dir = "./models/pubchem10M/features_selfattention/medium/converged/"
    config.num_workers = 0
    config.load_last = dataset
    config.dataset = 'bbbp' # supported 'bbbp', 'clintox', 'hiv', 'tox21'
    config.model_dir = './models/moleculenet/%s/pubchem_mini/features_selfattention/medium/1/'%config.dataset
    config.data_dir = "./datasets/mymoleculenet/%s/"%config.dataset
    config.loaddir = "./results/mymoleculenet_plus_features/%s/1/"%config.dataset
    config.seed = 123
    config.finetune = True
    config.missing_value = -1

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)


    os.makedirs(config.model_dir, exist_ok=True)
    
    ngpu=1
    device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    
    
    
    to_cuda = lambda T: [t.cuda() for t in T]
    
    roc_avgs = []
    prc_avgs = []
    for rep in range(10):
        model = DFSCodeSeq2SeqFC(nn.Linear(config.n_node_features, config.emb_dim),
                                 nn.Linear(config.n_edge_features, config.emb_dim),
                             n_atoms=config.n_atoms,
                             n_bonds=config.n_bonds, 
                             emb_dim=config.emb_dim, 
                             nhead=config.nhead, 
                             nlayers=config.nlayers, 
                             max_nodes=config.max_nodes, 
                             max_edges=config.max_edges,
                             missing_value=config.missing_value)
        if config.load_last:
            model.load_state_dict(torch.load(config.pretrained_dir+'checkpoint.pt', map_location=device))
            print('loaded pretrained model')
        
        model_head = nn.Linear(5*config.emb_dim, 1)
        model_head.to(device)
        model.to(device)
        w_init = nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        
        trainset = pd.read_csv(config.data_dir+"%d/train.csv"%rep)
        validset = pd.read_csv(config.data_dir+"%d/valid.csv"%rep)
        testset = pd.read_csv(config.data_dir+"%d/test.csv"%rep)
        train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
        valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
        test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
            
        
        collate_fn = collate_minc_rndc_features_y
    
        traindata = Deepchem2TorchGeometric(train_X, train_y, loaddir=config.loaddir, features=config.features)
        validdata = Deepchem2TorchGeometric(valid_X, valid_y, loaddir=config.loaddir, features=config.features)
        testdata = Deepchem2TorchGeometric(test_X, test_y, loaddir=config.loaddir, features=config.features)
        trainloader = DataLoader(traindata, batch_size=config.batch_size, shuffle=True, pin_memory=False, collate_fn=collate_fn)
        validloader = DataLoader(validdata, batch_size=config.batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn)
        testloader = DataLoader(testdata, batch_size=config.batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn)
        
        params = list(model_head.parameters())
        if config.finetune:
            params += list(model.parameters())
        optim = optimizers.Adam(params, lr=config.lr)
        lr_scheduler = optimizers.lr_scheduler.ReduceLROnPlateau(optim, mode='min', verbose=True, patience=config.patience, factor=config.factor)
        early_stopping_model = EarlyStopping(patience=config.valid_patience, delta=config.valid_minimal_improvement,
                                      path=config.model_dir+'checkpoint_model.pt')
        early_stopping_head = EarlyStopping(patience=config.valid_patience, delta=config.valid_minimal_improvement,
                                      path=config.model_dir+'checkpoint_head.pt')
        
        bce = torch.nn.BCEWithLogitsLoss()
        
        valid_scores = []
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
    
            
        model.load_state_dict(torch.load(config.model_dir+'checkpoint_model.pt'))
        model_head.load_state_dict(torch.load(config.model_dir+'checkpoint_head.pt'))
        model.eval()
        if config.use_min:
            roc_auc_valid, prc_auc_valid = score(testloader, model, model_head, config.use_min) 
            wandb.log({'roc_test':roc_auc_valid, 'prc_test':prc_auc_valid}, step=config.n_epochs*(rep+1))
            roc_avgs += [roc_auc_valid]
            prc_avgs += [prc_auc_valid]
        else:        
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
            wandb.log({'roc_test':avg_roc, 'prc_test':avg_prc}, step=config.n_epochs*(rep+1))
            print('ROC, PRC TEST', avg_roc, avg_prc)
        del model
        del model_head
    wandb.log({'roc_test_mean':np.mean(roc_avgs), 'prc_test_mean':np.mean(prc_avgs)}, step=config.n_epochs*(rep+1))
    wandb.log({'roc_test_std':np.std(roc_avgs), 'prc_test_std':np.std(prc_avgs)}, step=config.n_epochs*(rep+1))
    wandb.run.summary["roc_test_mean"] = np.mean(roc_avgs) 
    wandb.run.summary["roc_test_std"] = np.std(roc_avgs)
    wandb.run.summary["prc_test_mean"] = np.mean(prc_avgs)
    wandb.run.summary["prc_test_std"] = np.std(prc_avgs)
    



