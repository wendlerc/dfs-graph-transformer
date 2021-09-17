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
from dfs_transformer import EarlyStopping, DFSCodeSeq2SeqFC, Deepchem2TorchGeometric, FeaturesDataset, collate_smiles_minc_rndc_features_y, \
DFSCodeSeq2SeqFCFeatures
import argparse
import yaml
from ml_collections import ConfigDict
import copy
import dfs_code

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
        
    model = eval(t.pretrained_class)(**m)
    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir+'/checkpoint.pt', map_location=device), strict=t.strict)
        
    return model

def score(loader, model, model_head):
    with torch.no_grad():
        full_preds, target = [], []
        for data in tqdm.tqdm(loader):
            code, nfeat, efeat, y = data
            features = model.encode(to_cuda(code), to_cuda(nfeat), to_cuda(efeat), method=t.fingerprint)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
    parser.add_argument('--wandb_project', type=str, default="moleculenet10")
    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--yaml', type=str, default="./config/selfattn/finetune_bert.yaml") 
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--n_hidden', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="bbbp")
    parser.add_argument('--overwrite', type=json.loads, default="{}")
    args = parser.parse_args()
    
    with open(args.yaml) as file:
        config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
           
    config.n_hidden = args.n_hidden
    config.dataset = args.dataset
    
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
    
    # make collate fn
    if t.use_min:
        def collate_fn(dlist):
            node_batch = []
            y_batch = []
            edge_batch = []
            min_code_batch = []
            for d in dlist:
                node_batch += [d.node_features]
                edge_batch += [d.edge_features]
                min_code_batch += [d.min_dfs_code]
                y_batch += [d.y]
            return min_code_batch, node_batch, edge_batch, torch.cat(y_batch).unsqueeze(1)
    else:
        def collate_fn(dlist):
            node_batch = []
            y_batch = []
            edge_batch = []
            rnd_code_batch = []
            for d in dlist:
                rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                                 d.z.numpy().tolist(), 
                                                                                 np.argmax(d.edge_attr.numpy(), axis=1))
                node_batch += [d.node_features]
                edge_batch += [d.edge_features]
                rnd_code_batch += [torch.tensor(rnd_code)]
                y_batch += [d.y]
            return rnd_code_batch, node_batch, edge_batch, torch.cat(y_batch).unsqueeze(1)
    
    pretrained = load_selfattn(t, device)
    
    print('loaded pretrained model')
    dataset = t.dataset
    run = wandb.init(args.wandb_mode, 
             project=args.wandb_project, 
             entity=args.wandb_entity, 
             name=args.name, config=config.to_dict(),
             reinit=True)
    
    # push one batch through the nextwork just to get the feature shape...
    rep = 0
    trainset = pd.read_csv(t.data_dir_pattern%dataset+"%d/train.csv"%rep)
    train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
    
    dset = Deepchem2TorchGeometric(train_X, train_y, loaddir=t.load_dir_pattern%dataset, features="chemprop")
    loader = DataLoader(dset, batch_size=t.batch_size, shuffle=False, pin_memory=False, 
                        collate_fn=collate_fn)
    code, nfeat, efeat, y = next(iter(loader))
    features = pretrained.encode(code, nfeat, efeat, method=t.fingerprint)
    input_dim = features.shape[1]
    del dset
    del loader
    del features

    # 2. run evaluation
    roc_avgs = []
    prc_avgs = []
    for rep in range(10):
        model_dir = t.model_dir_pattern%dataset+'%d/%d/'%(rep, np.random.randint(100000))
        os.makedirs(model_dir, exist_ok=True)
        # create head
        if t.n_hidden > 0:
            layers = []
            for j in range(t.n_hidden):
                layers += [nn.Linear(input_dim, input_dim//2), nn.ReLU(inplace=True)]
                input_dim = input_dim // 2
            layers += [nn.Linear(input_dim, 1)]
            model_head = nn.Sequential(*layers)
        else:
            model_head = nn.Linear(input_dim, 1)
        model_head.to(device)
        model = copy.deepcopy(pretrained)
        model.to(device)
        
        if t.weight_decay is not None:
            w_init = nn.utils.parameters_to_vector(model.parameters()).detach().clone()
           
        trainset = pd.read_csv(t.data_dir_pattern%dataset+"%d/train.csv"%rep)
        validset = pd.read_csv(t.data_dir_pattern%dataset+"%d/valid.csv"%rep)
        testset = pd.read_csv(t.data_dir_pattern%dataset+"%d/test.csv"%rep)
        train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
        valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
        test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
        traindata = Deepchem2TorchGeometric(train_X, train_y, loaddir=t.load_dir_pattern%dataset, features="chemprop")
        validdata = Deepchem2TorchGeometric(valid_X, valid_y, loaddir=t.load_dir_pattern%dataset, features="chemprop")
        testdata = Deepchem2TorchGeometric(test_X, test_y, loaddir=t.load_dir_pattern%dataset, features="chemprop")

        trainloader = DataLoader(traindata, batch_size=t.batch_size, shuffle=True, pin_memory=t.use_min, collate_fn=collate_fn)
        validloader = DataLoader(validdata, batch_size=t.batch_size, shuffle=False, pin_memory=t.use_min, collate_fn=collate_fn)
        testloader = DataLoader(testdata, batch_size=t.batch_size, shuffle=False, pin_memory=t.use_min, collate_fn=collate_fn)
        
        optim = optimizers.Adam(model_head.parameters(), lr = t.lr_head)
        
        lr_scheduler = optimizers.lr_scheduler.ReduceLROnPlateau(optim, mode='min', verbose=True, patience=t.lr_patience, factor=t.decay_factor)
        early_stopping_model = EarlyStopping(patience=t.es_patience, delta=t.es_improvement,
                                      path=model_dir+'checkpoint_model.pt')
        early_stopping_head = EarlyStopping(patience=t.es_patience, delta=t.es_improvement,
                                      path=model_dir+'checkpoint_head.pt')
        
        bce = torch.nn.BCEWithLogitsLoss()
        
        valid_scores = []
        for epoch in range(t.n_epochs):  
            if epoch == t.n_frozen:
                optim.add_param_group({"params": model.parameters(), "lr": t.lr_pretrained})
                lr_scheduler = optimizers.lr_scheduler.ReduceLROnPlateau(optim, mode='min', verbose=True, patience=t.lr_patience, factor=t.decay_factor)
            epoch_loss = 0
            pbar = tqdm.tqdm(trainloader)
            model.train()
            for i, data in enumerate(pbar):
                if i % t.accumulate_grads == 0: 
                    optim.zero_grad()
                
                code, nfeat, efeat, y = data
                features = model.encode(to_cuda(code), to_cuda(nfeat), to_cuda(efeat), method=t.fingerprint)
                prediction = model_head(features.to(device))
                loss = bce(prediction, y.to(device))
                if t.weight_decay > 0:
                    w_curr = nn.utils.parameters_to_vector(model.parameters())
                    loss += 0.5*t.weight_decay*torch.sum((w_init - w_curr)**2)
                loss.backward()
                
                if (i + 1) % t.accumulate_grads == 0: 
                    if t.clip_gradient > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), t.clip_gradient)
                    optim.step()
                epoch_loss = (epoch_loss*i + loss.item())/(i+1)
                pbar.set_description('Epoch %d: CE %2.6f'%(epoch+1, epoch_loss))
                #run.log({'batch-loss':loss, 'loss':epoch_loss})
            model.eval()
            roc_auc_valid, prc_auc_valid = score(validloader, model, model_head) 
            valid_scores += [roc_auc_valid]
            lr_scheduler.step(epoch_loss)
            early_stopping_head(-roc_auc_valid, model_head)
            early_stopping_model(-roc_auc_valid, model)
            curr_lr = list(optim.param_groups)[0]['lr']
            run.log({'loss':epoch_loss, 'roc_valid':roc_auc_valid, 'prc_valid':prc_auc_valid, 'learning rate':curr_lr}, 
                      step=t.n_epochs*rep + epoch)
            print('ROCAUC',roc_auc_valid, 'PRCAUC', prc_auc_valid)

            if early_stopping_head.early_stop:
                break

            if curr_lr < t.minimal_lr:
                break
    
        # test set
        model.load_state_dict(torch.load(model_dir+'checkpoint_model.pt'))
        model_head.load_state_dict(torch.load(model_dir+'checkpoint_head.pt'))
        model.eval()
        roc_auc_valid, prc_auc_valid = score(testloader, model, model_head) 
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




