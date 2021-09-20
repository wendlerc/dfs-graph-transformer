import numpy as np
import json
import wandb
import dfs_code
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path = ['./src'] + sys.path
from dfs_transformer import DFSCodeSeq2SeqFC, Trainer, PubChem, get_n_files
from dfs_transformer.training.utils import seq_loss, seq_acc, collate_BERT, collate_rnd2min
import argparse
import yaml
import functools
from ml_collections import ConfigDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
    parser.add_argument('--wandb_project', type=str, default="pubchem")
    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--yaml', type=str, default="./config/selfattn/bert.yaml") 
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--overwrite', type=json.loads, default="{}")
    args = parser.parse_args()
    
    with open(args.yaml) as file:
        config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    
    for key,value in args.overwrite.items():
        for key1,value1 in value.items():
            config[key][key1] = value1
    
    run = wandb.init(mode=args.wandb_mode, 
                     project=args.wandb_project, 
                     entity=args.wandb_entity, 
                     name=args.name, config=config.to_dict())
    
    m = config.model
    t = config.training
    d = config.data
    print(config)
    device = torch.device('cuda:%d'%config.training.gpu_id if torch.cuda.is_available() else 'cpu')
    
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    loss = functools.partial(seq_loss, ce=ce, m=m)
    
    if config.training.mode == "min2min":
        collate_fn = functools.partial(collate_BERT, mode="min2min", fraction_missing = config.training.fraction_missing)
    elif config.training.mode == "rnd2rnd":
        collate_fn = functools.partial(collate_BERT, mode="rnd2rnd", fraction_missing = config.training.fraction_missing)
    elif config.training.mode == "rnd2min":
        collate_fn = collate_rnd2min
    else:
        raise ValueError("unknown config.training.mode %s"%config.training.mode)        
    
    fields = ['acc-dfs1', 'acc-dfs2', 'acc-atm1', 'acc-atm2', 'acc-bnd']
    metrics = {field:functools.partial(seq_acc, idx=idx) for idx, field in enumerate(fields)}
    model = DFSCodeSeq2SeqFC(**m)
    
    if t.load_last and t.es_path is not None:
        model.load_state_dict(torch.load(t.es_path+'checkpoint.pt', map_location=device))
    elif t.pretrained_dir is not None:
        model.load_state_dict(torch.load(t.pretrained_dir+'checkpoint.pt', map_location=device))
    
    validloader = None
    if d.valid_path is not None:
        validset = PubChem(d.valid_path, max_nodes=m.max_nodes, max_edges=m.max_edges)
        validloader = DataLoader(validset, batch_size=t.batch_size, shuffle=True, 
                                 pin_memory=False, collate_fn=collate_fn)
        exclude = validset.smiles
    
    trainer = Trainer(model, None, loss, validloader=validloader, metrics=metrics, 
                      wandb_run = run, **t)
    trainer.n_epochs = d.n_iter_per_split
    
    n_files = get_n_files(d.path)
    if d.n_used is None:
        n_splits = 1
    else:
        n_splits = n_files // d.n_used
        
    for epoch in range(t.n_epochs):
        print('starting epoch %d'%(epoch+1))
        for split in range(n_splits):
            dataset = PubChem(d.path, n_used = d.n_used, max_nodes=m.max_nodes, 
                              max_edges=m.max_edges, exclude=exclude)
            loader = DataLoader(dataset, batch_size=t.batch_size, shuffle=True, 
                                pin_memory=False, collate_fn=collate_fn)
            trainer.loader = loader
            trainer.fit()
            if trainer.stop_training:
                break
        if trainer.stop_training:
            break
    
    #store config and model
    with open(trainer.es_path+'config.yaml', 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    if args.name is not None and args.wandb_mode != "offline":
        trained_model_artifact = wandb.Artifact(args.name, type="model", description="trained selfattn model")
        trained_model_artifact.add_dir(trainer.es_path)
        run.log_artifact(trained_model_artifact)
        
        
    
