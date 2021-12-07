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
from dfs_transformer import QM9, DFSCodeSeq2SeqFC, TransformerPlusHeads, chemical_accuracy, atomrefs
from dfs_transformer import to_cuda, TrainerNew
from dfs_transformer import collate_downstream_qm9 as collate_downstream
import argparse
import yaml
from ml_collections import ConfigDict
import functools
from copy import deepcopy
import pickle
import glob

torch.multiprocessing.set_sharing_strategy('file_system')

def load_selfattn(t, device):
    # download pretrained model
    run = wandb.init(mode=args.wandb_mode, 
                     project=t.pretrained_project, 
                     entity=t.pretrained_entity, 
                     job_type="inference",
                     dir=t.wandb_dir)
    model_at = run.use_artifact(t.pretrained_model + ":latest")
    model_dir = model_at.download(root=t.wandb_dir+'/artifacts/%s/'%t.pretrained_model)
    run.finish()

    model_yaml = model_dir+"/config.yaml"
    if not os.path.isfile(model_yaml) or t.use_local_yaml:
        model_yaml = t.pretrained_yaml
    
    with open(model_yaml) as file:
        cfg = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    
    m = cfg.model
    
    encoder = DFSCodeSeq2SeqFC(**m)
    head_specs = {}
    
    with open(t.data.path+"/properties_aggr.pkl", "rb") as f:
        prop_aggr = pickle.load(f)
    if config.data.molecular_properties is None:
        cfg.molecular_properties = list(prop_aggr.keys())
    
    for name in cfg.data.molecular_properties:
        aggr = prop_aggr[name]
        if aggr['is_int']:
            head_specs[name] = aggr['max'] - aggr['min'] + 2 # +1 because we have class 0, ..., k, +1 for outliers 
        else:
            head_specs[name] = 1
    
    model = TransformerPlusHeads(encoder, head_specs)
    
    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir+'/checkpoint.pt', map_location=device), strict=t.strict)
    model = model.encoder
        
    return model, m

def loss(pred, y, l):
    return l(pred, y[0])


def mae_ca(pred, y, target_idx):
    mae = (pred.view(-1) - y[0].view(-1)).abs().mean()
    return mae/chemical_accuracy[target_idx]


def score(model, loader, device, target_idx):
    with torch.no_grad():
        maes = []
        for data in tqdm.tqdm(loader):
            smiles, codes, z, y = data
            codes = to_cuda(codes, device)
            z = to_cuda(z, device)
            y = to_cuda(y, device)
            pred = model(codes, z).squeeze()
            mae = (pred.view(-1) - y.view(-1)).abs()
            maes += [mae]
        
        maes = torch.cat(maes, dim=0)
        mae = maes.mean()
        return mae/chemical_accuracy[target_idx]


class TransformerPlusHead(nn.Module):
    def __init__(self, encoder, n_encoding, n_classes, n_hidden=0, fingerprint='cls',
                 mean = 0., std = 1., atomref = None):
        super(TransformerPlusHead, self).__init__()
        self.encoder = encoder
        self.ninp = self.encoder.ninp
        if n_hidden > 0:
            layers = []
            input_dim = n_encoding
            for j in range(n_hidden):
                layers += [nn.Linear(input_dim, input_dim//2), nn.ReLU(inplace=True)]
                input_dim = input_dim // 2
            layers += [nn.Linear(input_dim, n_classes)]
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Linear(n_encoding, n_classes)
        self.fingerprint = fingerprint
        self.mean = mean
        self.std = std
        self.register_buffer('initial_atomref', atomref)
        self.atomref = nn.Embedding(119, 1) # 119 better for batch processing because then we don't have -1 in the missing spots
        if atomref is not None:
            self.atomref.weight.data.copy_(atomref)
    
    def forward(self, dfs_codes, z):
        features = self.encoder.encode(dfs_codes, method=self.fingerprint)
        out = self.head(features)
        if self.mean is not None and self.std is not None:
            out = out * self.std + self.mean
        
        if self.atomref is not None:
            #print(self.atomref(z))
            #print(self.atomref(z).shape)
            out = out + torch.sum(self.atomref(z), axis=1) # does not make sense if the data has no Hs
        
        return out
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
    parser.add_argument('--wandb_project', type=str, default="qm9-finetune")
    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--wandb_dir', type=str, default="./wandb")
    parser.add_argument('--yaml', type=str, default="./config/selfattn/finetune_qm9.yaml") 
    parser.add_argument('--yaml_data', type=str, default="./config/selfattn/data/pubchem1M.yaml")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--n_hidden', type=int, default=0)
    parser.add_argument('--overwrite', type=json.loads, default="{}")
    args = parser.parse_args()
        
    with open(args.yaml) as file:
        config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
        
    with open(args.yaml_data) as file:
        d = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    
    for key,value in args.overwrite.items():
        if type(value) is dict:
            for key1,value1 in value.items():
                config[key][key1] = value1
        else:
            config[key] = value
    
    config.data = d
    
    if args.model is not None:
        config.pretrained_model = args.model
        if args.name is None:
            args.name = args.model
    
    if "n_hidden" not in args.overwrite.keys():
        config.n_hidden = args.n_hidden
    config.wandb_dir = args.wandb_dir
    t = config
    print(t)

    random.seed(t.seed)
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)
    
    device = torch.device('cuda:%d'%t.gpu_id if torch.cuda.is_available()  else 'cpu')
    
    encoder, m = load_selfattn(t, device)
    print('loaded pretrained model')
    if "use_min" not in args.overwrite.keys():
        config["use_min"] = m.missing_value == -1
    
    if "use_loops" not in m:
        m.use_loops = False
    
    collate_fn = collate_downstream
    
    coll_train = functools.partial(collate_fn, use_loops=m.use_loops, use_min=t.use_min, alpha=t.label_smoothing_alpha)
    coll_val = functools.partial(collate_fn, use_loops=m.use_loops, use_min=t.use_min, alpha=0)
    
    
    dataset = "qm9"
    run = wandb.init(args.wandb_mode, 
             project=args.wandb_project, 
             entity=args.wandb_entity, 
             name=args.name, config=config.to_dict(),
             reinit=True,
             dir=args.wandb_dir)
    wandb.config.update({'dataset': "qm9"}, allow_val_change=True)
    
    roc_avgs = []
    prc_avgs = []
    
    roc_avgs_valid = []
    prc_avgs_valid = []
    
    mname = "".join(x for x in t.pretrained_model + t.fingerprint if x.isalnum())
    model_dir = t.model_dir_pattern%dataset+'/%s/%d/'%(mname, np.random.randint(100000))
    os.makedirs(model_dir, exist_ok=True)
    
    wandb.config.update({'es_path': model_dir}, allow_val_change=True)
    t.es_path = model_dir

    n_encoding = encoder.get_n_encoding(t.fingerprint)
    
    
    traindata = QM9(split="train", path="datasets/myqm9/noH/") 
    validdata = QM9(split="valid", path="datasets/myqm9/noH/") 
    testdata = QM9(split="test", path="datasets/myqm9/noH/")
    #traindata = validdata 
    #testdata = validdata
    target_idx = traindata.target_idx
    
    trainloader = DataLoader(traindata, batch_size=t.batch_size, shuffle=True, pin_memory=True, 
                    collate_fn=coll_train, num_workers=t.num_workers)
    validloader = DataLoader(validdata, batch_size=t.batch_size, shuffle=False, pin_memory=True, 
                    collate_fn=coll_val, num_workers=t.num_workers)
    testloader = DataLoader(testdata, batch_size=t.batch_size, shuffle=False, pin_memory=True, 
                    collate_fn=coll_val, num_workers=t.num_workers)
    

    
    target_mean, target_std = traindata.compute_mean_and_std()
    
    model = TransformerPlusHead(deepcopy(encoder), n_encoding, 1, n_hidden=t.n_hidden, fingerprint=t.fingerprint, 
                                mean = target_mean, std = target_std, atomref = traindata.atomref())
    model.to(device)
    
    param_groups = [
        {'amsgrad': False,
         'betas': (0.9,0.98),
         'eps': 1e-09,
         'lr': t.lr_encoder,
         'params': model.encoder.parameters(),
         'weight_decay': 0},
        {'amsgrad': False,
         'betas': (0.9, 0.999),
         'eps': 1e-08,
         'lr': t.lr_head,
         'params': model.head.parameters(),
         'weight_decay': 0}
    ]
    
    
    l = functools.partial(loss, l=torch.nn.MSELoss(reduction='mean'))
    mae_ca = functools.partial(mae_ca, target_idx = traindata.target_idx)
    scorer = functools.partial(score, target_idx = traindata.target_idx, loader=validloader, device=device)
    
    t.es_period = (len(traindata)//t.batch_size)#//2
    t.lr_adjustment_period = (len(traindata)//t.batch_size)#//2
    wandb.config.update({'es_period': t.es_period}, allow_val_change=True)
    wandb.config.update({'lr_adjustment_period': t.lr_adjustment_period}, allow_val_change=True)
    
    
    # create trainer
    trainer = TrainerNew(model, trainloader, l, input_idxs = [1, 2], output_idxs = [3], validloader=validloader, metrics={'MAE/CA': mae_ca}, scorer=scorer, wandb_run = run, param_groups=param_groups, lr_decay_type=None, **t)
    trainer.fit()
    # valid and test acc of best model
    model.load_state_dict(torch.load(trainer.es_path+'checkpoint.pt'))
    
    maeca = score(model, testloader, device, traindata.target_idx)
    if not t.use_min:
        for i in range(19):
            maeca += score(model, testloader, device, traindata.target_idx)
        maeca /= 20
    wandb.log({'MAE/CA test':maeca})

    run.finish()
    



