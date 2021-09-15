#!/usr/bin/env python
# coding: utf-8
import json
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
import yaml
from ml_collections import ConfigDict

def load_selfattn(t, device):
    model_dir = None
    model_yaml = None
    if t.pretrained_model is not None:
        # download pretrained model
        run = wandb.init(mode=args.wandb_mode, 
                         project=t.pretrained_project, 
                         entity=t.pretrained_entity, 
                         job_type="inference")
        model_at = run.use_artifact(t.pretrained_model + ":latest")
        model_dir = model_at.download()
        run.finish()
    elif t.pretrained_dir is not None:
        model_dir = t.pretrained_dir
        
    model_yaml = model_dir+"/config.yaml"
    if not os.path.isfile(model_yaml) or t.use_local_yaml:
        model_yaml = t.pretrained_yaml
    
    with open(model_yaml) as file:
        config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    m = config.model
        
    model = DFSCodeSeq2SeqFC(**m)
    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir+'/checkpoint.pt', map_location=device), strict=t.strict)
        
    return model

def score(loader, model):
    with torch.no_grad():
        full_preds, target = [], []
        for batch in tqdm.tqdm(loader):
            features, y = batch
            y = y.to(device).squeeze()
            pred = model(features.to(device)).squeeze()
            pred_np = pred.detach().cpu().numpy().tolist()
            y_np = y.detach().cpu().numpy().tolist()
            full_preds += pred_np
            target += y_np

        target = np.asarray(target)
        full_preds = np.asarray(full_preds)
        roc = roc_auc_score(target, full_preds)
        prc = average_precision_score(target, full_preds)
    return roc, prc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
    parser.add_argument('--wandb_project', type=str, default="moleculenet10")
    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--yaml', type=str, default="./config/selfattn/finetune_bert.yaml") 
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--overwrite', type=json.loads, default="{}")
    args = parser.parse_args()
    
    with open(args.yaml) as file:
        config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    
    for key,value in args.overwrite.items():
        if type(value) is dict:
            for key1,value1 in value.items():
                config[key][key1] = value1
        else:
            config[key] = value
    
    t = config
    print(t)

    random.seed(t.seed)
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)
    
    device = torch.device('cuda:%d'%t.gpu_id if torch.cuda.is_available()  else 'cpu')
    to_cuda = lambda T: [t.to(device) for t in T]
    
    model = load_selfattn(t, device)
    model.to(device)
    print('loaded pretrained model')
    
    datasets = ['bbbp', 'clintox', 'tox21', 'hiv']
    
    for idx, dataset in enumerate(datasets):
        run = wandb.init(args.wandb_mode, 
                 project=args.wandb_project, 
                 entity=args.wandb_entity, 
                 name=args.name, config=config.to_dict(),
                 reinit=True)
        wandb.config.update({'dataset': dataset}, allow_val_change=True)
        # 1. compute all feature vectors
        rep = 0
        trainset = pd.read_csv(t.data_dir_pattern%dataset+"%d/train.csv"%rep)
        validset = pd.read_csv(t.data_dir_pattern%dataset+"%d/valid.csv"%rep)
        testset = pd.read_csv(t.data_dir_pattern%dataset+"%d/test.csv"%rep)
        train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
        valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
        test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
        X = np.concatenate((train_X, valid_X, test_X), axis=0)
        y = np.concatenate((train_y, valid_y, test_y), axis=0)
        dset = Deepchem2TorchGeometric(X, y, loaddir=t.load_dir_pattern%dataset, features="chemprop")
        loader = DataLoader(dset, batch_size=t.batch_size, shuffle=False, pin_memory=False, 
                            collate_fn=collate_smiles_minc_rndc_features_y)
        smiles = []
        features = []
        for i, data in enumerate(tqdm.tqdm(loader)):
            smls, rndc, minc, nfeat, efeat, y = data
            if t.use_min:
                code = to_cuda(minc)
            else:
                code = to_cuda(rndc)
            feats = model.encode(code, to_cuda(nfeat), to_cuda(efeat), method=t.fingerprint)
            smiles += smls
            features += [feats.detach().cpu()]
        features = torch.cat(features, dim=0)
        features_dict = {smile:feature for smile, feature in zip(smiles, features)}
        
        model_dir = t.model_dir_pattern%dataset+'%d/'%np.random.randint(100000)
        os.makedirs(model_dir, exist_ok=True)
        # 2. run evaluation
        roc_avgs = []
        prc_avgs = []
        for rep in range(10):
            model_head = nn.Linear(features.shape[1], 1)
            model_head.to(device)
               
            trainset = pd.read_csv(t.data_dir_pattern%dataset+"%d/train.csv"%rep)
            validset = pd.read_csv(t.data_dir_pattern%dataset+"%d/valid.csv"%rep)
            testset = pd.read_csv(t.data_dir_pattern%dataset+"%d/test.csv"%rep)
            train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
            valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
            test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
                
            traindata = FeaturesDataset(train_X, train_y, features_dict)
            validdata = FeaturesDataset(valid_X, valid_y, features_dict)
            testdata = FeaturesDataset(test_X, test_y, features_dict)
            trainloader = DataLoader(traindata, batch_size=t.batch_size, shuffle=True, pin_memory=True)
            validloader = DataLoader(validdata, batch_size=t.batch_size, shuffle=False, pin_memory=True)
            testloader = DataLoader(testdata, batch_size=t.batch_size, shuffle=False, pin_memory=True)
            
            params = list(model_head.parameters())
            
            optim = optimizers.Adam(params, lr=t.lr)
            lr_scheduler = optimizers.lr_scheduler.ReduceLROnPlateau(optim, mode='min', verbose=True, patience=t.lr_patience, factor=t.decay_factor)
            early_stopping_head = EarlyStopping(patience=t.es_patience, delta=t.es_improvement,
                                          path=model_dir+'checkpoint_head.pt')
            
            bce = torch.nn.BCEWithLogitsLoss()
            
            valid_scores = []
            for epoch in range(t.n_epochs):  
                epoch_loss = 0
                pbar = tqdm.tqdm(trainloader)
                for i, data in enumerate(pbar):
                    if i % t.accumulate_grads == 0: 
                        optim.zero_grad()
                    features, y = data
                    prediction = model_head(features.to(device))
                    loss = bce(prediction, y.to(device))
                    loss.backward()
                    
                    if (i + 1) % t.accumulate_grads == 0: 
                        optim.step()
                    epoch_loss = (epoch_loss*i + loss.item())/(i+1)
                    pbar.set_description('Epoch %d: CE %2.6f'%(epoch+1, epoch_loss))
                roc_auc_valid, prc_auc_valid = score(validloader, model_head) 
                valid_scores += [roc_auc_valid]
                lr_scheduler.step(epoch_loss)
                early_stopping_head(-roc_auc_valid, model_head)
                curr_lr = list(optim.param_groups)[0]['lr']
                wandb.log({'loss':epoch_loss, 'roc_valid':roc_auc_valid, 'prc_valid':prc_auc_valid, 'learning rate':curr_lr}, 
                          step=t.n_epochs*rep + epoch)
                print('ROCAUC',roc_auc_valid, 'PRCAUC', prc_auc_valid)
    
                if early_stopping_head.early_stop:
                    break
    
                if curr_lr < t.minimal_lr:
                    break
        
            # test set
            model_head.load_state_dict(torch.load(model_dir+'checkpoint_head.pt'))
            
            roc_auc_valid, prc_auc_valid = score(testloader, model_head) 
            wandb.log({'roc_test':roc_auc_valid, 'prc_test':prc_auc_valid}, step=t.n_epochs*(rep+1))
            roc_avgs += [roc_auc_valid]
            prc_avgs += [prc_auc_valid]
            
            del model_head
        wandb.log({'roc_test_mean':np.mean(roc_avgs), 'prc_test_mean':np.mean(prc_avgs)}, step=t.n_epochs*(rep+1))
        wandb.log({'roc_test_std':np.std(roc_avgs), 'prc_test_std':np.std(prc_avgs)}, step=t.n_epochs*(rep+1))
        wandb.run.summary["roc_test_mean"] = np.mean(roc_avgs) 
        wandb.run.summary["roc_test_std"] = np.std(roc_avgs)
        wandb.run.summary["prc_test_mean"] = np.mean(prc_avgs)
        wandb.run.summary["prc_test_std"] = np.std(prc_avgs)
        run.finish()
    



