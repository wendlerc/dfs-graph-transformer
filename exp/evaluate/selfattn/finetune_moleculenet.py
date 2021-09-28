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
from dfs_transformer import Deepchem2TorchGeometric, collate_downstream, DFSCodeSeq2SeqFC
from dfs_transformer import to_cuda, Trainer
import argparse
import yaml
from ml_collections import ConfigDict
import functools
from copy import deepcopy
import dfs_code
from chemprop.features.featurization import bond_features

torch.multiprocessing.set_sharing_strategy('file_system')

def load_selfattn(t, device):
    model_dir = None
    model_yaml = None
    if t.pretrained_model is not None:
        # download pretrained model
        run = wandb.init(mode=args.wandb_mode, 
                         project=t.pretrained_project, 
                         entity=t.pretrained_entity, 
                         job_type="inference",
                         dir=t.wandb_dir)
        model_at = run.use_artifact(t.pretrained_model + ":latest")
        model_dir = model_at.download(root=t.wandb_dir+'/artifacts/%s/'%t.pretrained_model)
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
        
    return model, m

def loss(pred, y, ce):
    return ce(pred, y)


def acc(pred, y):
    y_pred = (pred > 0.5).squeeze()
    y = (y > 0.5)
    return (y_pred == y.squeeze()).sum()/len(y)

def score(model, loader, device, onlyroc=False):
    with torch.no_grad():
        full_preds, target = [], []
        for data in tqdm.tqdm(loader):
            data = [to_cuda(d, device) for d in data]
            pred = model(*data[:-1]).squeeze()
            y = data[-1].detach().cpu().squeeze()
            pred_np = pred.detach().cpu().numpy().tolist()
            y_np = y.numpy().tolist()
            full_preds += pred_np
            target += y_np

        target = np.asarray(target)
        full_preds = np.asarray(full_preds)
        roc = roc_auc_score(target, full_preds)
        prc = average_precision_score(target, full_preds)
    if onlyroc:
        return roc
    return roc, prc

def collate_fn(dlist, alpha=0, use_loops=True, use_min=False):
    node_batch = [] 
    edge_batch = []
    y_batch = []
    code_batch = []
    if use_loops:
        loop = torch.tensor(bond_features(None)).unsqueeze(0)
    for d in dlist:
        edge_features = d.edge_features.clone()
        if use_min:
            code = d.min_dfs_code.clone()
            index = d.min_dfs_index.clone()
        else:
            code, index = dfs_code.rnd_dfs_code_from_torch_geometric(d, d.z.numpy().tolist(), 
                                                                     np.argmax(d.edge_attr.numpy(), axis=1).tolist())
            
            code = torch.tensor(code, dtype=torch.long)
            index = torch.tensor(index, dtype=torch.long)
        
        if use_loops:
            edge_features = torch.cat((edge_features, loop), dim=0)
            vids = torch.argsort(index).unsqueeze(1)
            eids = torch.ones_like(vids)*(edge_features.shape[0] - 1)
            nattr = d.z[vids]
            eattr = torch.ones_like(vids)*4 # 4 stands for loop
            arange = index[vids]
            loops = torch.cat((arange, arange, nattr, eattr, nattr, vids, eids, vids), dim=1)
            code = torch.cat((code, loops), dim=0)
        
        code_batch += [code]
        node_batch += [d.node_features.clone()]
        edge_batch += [edge_features]
        y_batch += [d.y.clone()]
    y = torch.cat(y_batch).unsqueeze(1)
    y = (1-alpha)*y + alpha/2
    return code_batch, node_batch, edge_batch, y


class TransformerPlusHead(nn.Module):
    def __init__(self, encoder, n_encoding, n_classes, n_hidden=0, fingerprint='cls'):
        super(TransformerPlusHead, self).__init__()
        self.encoder = encoder
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
    
    def forward(self, C, N, E):
        features = self.encoder.encode(C, N, E, method=self.fingerprint)
        output = self.head(features)
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
    parser.add_argument('--wandb_project', type=str, default="moleculenet10-finetune")
    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--wandb_dir', type=str, default="./wandb")
    parser.add_argument('--yaml', type=str, default="./config/selfattn/finetune_moleculenet.yaml") 
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--n_hidden', type=int, default=0)
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
    
    if args.model is not None:
        config.pretrained_model = args.model
        if args.name is None:
            args.name = args.model
            
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
    
    coll_train = functools.partial(collate_fn, use_loops=m.use_loops, use_min=t.use_min, alpha=t.label_smoothing_alpha)
    coll_val = functools.partial(collate_fn, use_loops=m.use_loops, use_min=t.use_min, alpha=0)
    
    datasets = ['bbbp', 'clintox', 'tox21', 'hiv']
    
    for idx, dataset in enumerate(datasets):
        run = wandb.init(args.wandb_mode, 
                 project=args.wandb_project, 
                 entity=args.wandb_entity, 
                 name=args.name, config=config.to_dict(),
                 reinit=True,
                 dir=args.wandb_dir)
        wandb.config.update({'dataset': dataset}, allow_val_change=True)
        
        roc_avgs = []
        prc_avgs = []
        
        roc_avgs_valid = []
        prc_avgs_valid = []
        
        for rep in range(10):
            n_encoding = encoder.get_n_encoding(t.fingerprint)
            model = TransformerPlusHead(deepcopy(encoder), n_encoding, 1, n_hidden=t.n_hidden, fingerprint=t.fingerprint)
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
            
            #make dataset
            trainset = pd.read_csv(t.data_dir_pattern%dataset+"%d/train.csv"%rep)
            validset = pd.read_csv(t.data_dir_pattern%dataset+"%d/valid.csv"%rep)
            testset = pd.read_csv(t.data_dir_pattern%dataset+"%d/test.csv"%rep)
            train_X, train_y = trainset["smiles"].to_numpy(), trainset["target"].to_numpy()
            valid_X, valid_y = validset["smiles"].to_numpy(), validset["target"].to_numpy()
            test_X, test_y = testset["smiles"].to_numpy(), testset["target"].to_numpy()
            
        
            
            traindata = Deepchem2TorchGeometric(train_X, train_y, loaddir=t.load_dir_pattern%dataset, features="chemprop")
            validdata = Deepchem2TorchGeometric(valid_X, valid_y, loaddir=t.load_dir_pattern%dataset, features="chemprop")
            testdata = Deepchem2TorchGeometric(test_X, test_y, loaddir=t.load_dir_pattern%dataset, features="chemprop")
            
            trainloader = DataLoader(traindata, batch_size=t.batch_size, shuffle=False, pin_memory=False, 
                            collate_fn=coll_train, num_workers=t.num_workers)
            validloader = DataLoader(validdata, batch_size=t.batch_size, shuffle=False, pin_memory=False, 
                            collate_fn=coll_val, num_workers=t.num_workers)
            testloader = DataLoader(testdata, batch_size=t.batch_size, shuffle=False, pin_memory=False, 
                            collate_fn=coll_val, num_workers=t.num_workers)
            
            
            
            bce = torch.nn.BCEWithLogitsLoss()
            l = functools.partial(loss, ce=bce)
            scorer = functools.partial(score, loader=validloader, device=device, onlyroc=True)
            
            t.es_period = (len(trainset)//t.batch_size)//2
            t.lr_adjustment_period = (len(trainset)//t.batch_size)//2
            wandb.config.update({'es_period': t.es_period}, allow_val_change=True)
            wandb.config.update({'lr_adjustment_period': t.lr_adjustment_period}, allow_val_change=True)
            
            # create trainer
            trainer = Trainer(model, trainloader, l, validloader=validloader, metrics={'acc': acc}, scorer=scorer, wandb_run = run, param_groups=param_groups, **t)
            trainer.fit()
            # valid and test acc of best model
            model.load_state_dict(torch.load(trainer.es_path+'checkpoint.pt'))
            
            roc, prc = score(model, testloader, device, onlyroc=False)
            if not t.use_min:
                for i in range(19):
                    roc2, prc2 = score(model, testloader, device, onlyroc=False)
                    roc += roc2
                    prc += prc2
                roc /= 20
                prc /= 20
            wandb.log({'roc_test':roc, 'prc_test':prc})
            roc_avgs += [roc]
            prc_avgs += [prc]
            
            roc, prc = score(model, validloader, device, onlyroc=False)
            if not t.use_min:
                for i in range(19):
                    roc2, prc2 = score(model, validloader, device, onlyroc=False)
                    roc += roc2
                    prc += prc2
                roc /= 20
                prc /= 20
            wandb.log({'roc_valid':roc, 'prc_valid':prc})
            roc_avgs_valid += [roc]
            prc_avgs_valid += [prc]
            
            del model
        wandb.log({'roc_test_mean':np.mean(roc_avgs), 'prc_test_mean':np.mean(prc_avgs)})
        wandb.log({'roc_test_std':np.std(roc_avgs), 'prc_test_std':np.std(prc_avgs)})
        wandb.log({'roc_valid_mean':np.mean(roc_avgs_valid), 'prc_valid_mean':np.mean(prc_avgs_valid)})
        wandb.log({'roc_valid_std':np.std(roc_avgs_valid), 'prc_valid_std':np.std(prc_avgs_valid)})
        wandb.run.summary["roc_test_mean"] = np.mean(roc_avgs) 
        wandb.run.summary["roc_test_std"] = np.std(roc_avgs)
        wandb.run.summary["prc_test_mean"] = np.mean(prc_avgs)
        wandb.run.summary["prc_test_std"] = np.std(prc_avgs)
        wandb.run.summary["roc_valid_mean"] = np.mean(roc_avgs_valid) 
        wandb.run.summary["roc_valid_std"] = np.std(roc_avgs_valid)
        wandb.run.summary["prc_valid_mean"] = np.mean(prc_avgs_valid)
        wandb.run.summary["prc_valid_std"] = np.std(prc_avgs_valid)
        run.finish()
    



