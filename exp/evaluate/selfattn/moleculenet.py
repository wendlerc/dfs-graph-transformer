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
from dfs_transformer import EarlyStopping, DFSCodeSeq2SeqFC, Deepchem2TorchGeometric, FeaturesDataset, collate_smiles_minc_rndc_features_y
import argparse

# TODO: make pretraining script that utilizes wandb.save("config.yaml", base_path=".hydra/") and artifact

parser = argparse.ArgumentParser()
# transformer params
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--n_atoms', type=int, default=118)
parser.add_argument('--n_bonds', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=120)
parser.add_argument('--nhead', type=int, default=12)
parser.add_argument('--nlayers', type=int, default=6)
parser.add_argument('--use_Hs', type=bool, default=False)
parser.add_argument('--max_nodes', type=int, default=250)
parser.add_argument('--max_edges', type=int, default=500)
parser.add_argument('--features', type=str, default="chemprop")
parser.add_argument('--n_node_features', type=int, default=133)
parser.add_argument('--n_edge_features', type=int, default=14)
parser.add_argument('--dim_feedforward', type=int, default=2048)
parser.add_argument('--missing_value', type=int, default=-1)
parser.add_argument('--use_min', type=int, default=1)
parser.add_argument('--fingerprint', type=str, default='cls')
# training params
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--n_epochs', type=int, default=25)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--factor', type=float, default=0.5)
parser.add_argument('--minimal_lr', type=float, default=6e-8)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--batch_size_preprocessing', type=int, default=10)
parser.add_argument('--accumulate_grads', type=int, default=1)
parser.add_argument('--valid_patience', type=int, default=25)
parser.add_argument('--valid_minimal_improvement', type=float, default=0)
parser.add_argument('--pretrained_dir', type=str, default="./models/pubchem10M/features_selfattention/medium/converged/")
parser.add_argument('--data_dir_pattern', type=str, default="./datasets/mymoleculenet/%s/")
parser.add_argument('--load_dir_pattern', type=str, default="./results/mymoleculenet_plus_features/%s/1/")
parser.add_argument('--model_dir_pattern', type=str, default="./models/mymoleculenet/%s/")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--strict', type=int, default=1)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

ngpu = args.ngpu
device = torch.device('cuda:%d'%args.gpu_id if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
to_cuda = lambda T: [t.to(device) for t in T]

def score(loader, model):
    with torch.no_grad():
        full_preds, target = [], []
        for batch in tqdm.tqdm(loader):
            features, y = batch
            y = y.to(device).squeeze()
            pred = model_head(features.to(device)).squeeze()
            pred_np = pred.detach().cpu().numpy().tolist()
            y_np = y.detach().cpu().numpy().tolist()
            full_preds += pred_np
            target += y_np

        target = np.asarray(target)
        full_preds = np.asarray(full_preds)
        roc = roc_auc_score(target, full_preds)
        prc = average_precision_score(target, full_preds)
    return roc, prc


model = DFSCodeSeq2SeqFC(nn.Linear(args.n_node_features, args.emb_dim),
                         nn.Linear(args.n_edge_features, args.emb_dim),
                         n_atoms=args.n_atoms,
                         n_bonds=args.n_bonds, 
                         emb_dim=args.emb_dim, 
                         nhead=args.nhead, 
                         nlayers=args.nlayers, 
                         max_nodes=args.max_nodes, 
                         max_edges=args.max_edges,
                         missing_value=args.missing_value)

model.load_state_dict(torch.load(args.pretrained_dir+'checkpoint.pt', map_location=device), strict=args.strict)
model.to(device)
print('loaded pretrained model')

datasets = ['bbbp', 'clintox', 'tox21', 'hiv']

for dataset in datasets:
    run = wandb.init(project='moleculenet10-clean', entity='chrisxx', name=args.name, reinit=True)
    config = wandb.config
    config.dataset = dataset
    wandb.config.update(args)
    # 1. compute all feature vectors
    rep = 0
    trainset = pd.read_csv(config.data_dir_pattern%dataset+"%d/train.csv"%rep)
    validset = pd.read_csv(config.data_dir_pattern%dataset+"%d/valid.csv"%rep)
    testset = pd.read_csv(config.data_dir_pattern%dataset+"%d/test.csv"%rep)
    train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
    valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
    test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
    X = np.concatenate((train_X, valid_X, test_X), axis=0)
    y = np.concatenate((train_y, valid_y, test_y), axis=0)
    dset = Deepchem2TorchGeometric(X, y, loaddir=config.load_dir_pattern%dataset, features=config.features)
    loader = DataLoader(dset, batch_size=config.batch_size_preprocessing, shuffle=False, pin_memory=False, 
                        collate_fn=collate_smiles_minc_rndc_features_y)
    smiles = []
    features = []
    for i, data in enumerate(tqdm.tqdm(loader)):
        smls, rndc, minc, nfeat, efeat, y = data
        if config.use_min:
            code = to_cuda(minc)
        else:
            code = to_cuda(rndc)
        feats = model.encode(code, to_cuda(nfeat), to_cuda(efeat), method=config.fingerprint)
        smiles += smls
        features += [feats.detach().cpu()]
    features = torch.cat(features, dim=0)
    features_dict = {smile:feature for smile, feature in zip(smiles, features)}
        
    model_dir = config.model_dir_pattern%dataset+'%d/'%np.random.randint(100000)
    os.makedirs(model_dir, exist_ok=True)
    # 2. run evaluation
    roc_avgs = []
    prc_avgs = []
    for rep in range(10):
        model_head = nn.Linear(5*config.emb_dim, 1)
        model_head.to(device)
           
        trainset = pd.read_csv(config.data_dir_pattern%dataset+"%d/train.csv"%rep)
        validset = pd.read_csv(config.data_dir_pattern%dataset+"%d/valid.csv"%rep)
        testset = pd.read_csv(config.data_dir_pattern%dataset+"%d/test.csv"%rep)
        train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
        valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
        test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
            
        traindata = FeaturesDataset(train_X, train_y, features_dict)
        validdata = FeaturesDataset(valid_X, valid_y, features_dict)
        testdata = FeaturesDataset(test_X, test_y, features_dict)
        trainloader = DataLoader(traindata, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        validloader = DataLoader(validdata, batch_size=config.batch_size, shuffle=False, pin_memory=True)
        testloader = DataLoader(testdata, batch_size=config.batch_size, shuffle=False, pin_memory=True)
        
        params = list(model_head.parameters())
        
        optim = optimizers.Adam(params, lr=config.lr)
        lr_scheduler = optimizers.lr_scheduler.ReduceLROnPlateau(optim, mode='min', verbose=True, patience=config.patience, factor=config.factor)
        early_stopping_head = EarlyStopping(patience=config.valid_patience, delta=config.valid_minimal_improvement,
                                      path=model_dir+'checkpoint_head.pt')
        
        bce = torch.nn.BCEWithLogitsLoss()
        
        valid_scores = []
        for epoch in range(config.n_epochs):  
            epoch_loss = 0
            pbar = tqdm.tqdm(trainloader)
            for i, data in enumerate(pbar):
                if i % config.accumulate_grads == 0: 
                    optim.zero_grad()
                features, y = data
                prediction = model_head(features.to(device))
                loss = bce(prediction, y.to(device))
                loss.backward()
                
                if (i + 1) % config.accumulate_grads == 0: 
                    optim.step()
                epoch_loss = (epoch_loss*i + loss.item())/(i+1)
                pbar.set_description('Epoch %d: CE %2.6f'%(epoch+1, epoch_loss))
            roc_auc_valid, prc_auc_valid = score(validloader, model_head) 
            valid_scores += [roc_auc_valid]
            lr_scheduler.step(epoch_loss)
            early_stopping_head(-roc_auc_valid, model_head)
            curr_lr = list(optim.param_groups)[0]['lr']
            wandb.log({'loss':epoch_loss, 'roc_valid':roc_auc_valid, 'prc_valid':prc_auc_valid, 'learning rate':curr_lr}, 
                      step=config.n_epochs*rep + epoch)
            print('ROCAUC',roc_auc_valid, 'PRCAUC', prc_auc_valid)

            if early_stopping_head.early_stop:
                break

            if curr_lr < config.minimal_lr:
                break
    
        # test set
        model_head.load_state_dict(torch.load(model_dir+'checkpoint_head.pt'))
        
        roc_auc_valid, prc_auc_valid = score(testloader, model_head) 
        wandb.log({'roc_test':roc_auc_valid, 'prc_test':prc_auc_valid}, step=config.n_epochs*(rep+1))
        roc_avgs += [roc_auc_valid]
        prc_avgs += [prc_auc_valid]
        
        del model_head
    wandb.log({'roc_test_mean':np.mean(roc_avgs), 'prc_test_mean':np.mean(prc_avgs)}, step=config.n_epochs*(rep+1))
    wandb.log({'roc_test_std':np.std(roc_avgs), 'prc_test_std':np.std(prc_avgs)}, step=config.n_epochs*(rep+1))
    wandb.run.summary["roc_test_mean"] = np.mean(roc_avgs) 
    wandb.run.summary["roc_test_std"] = np.std(roc_avgs)
    wandb.run.summary["prc_test_mean"] = np.mean(prc_avgs)
    wandb.run.summary["prc_test_std"] = np.std(prc_avgs)
    run.finish()
    



