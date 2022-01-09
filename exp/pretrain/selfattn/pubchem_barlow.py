import numpy as np
import json
import wandb
import dfs_code
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path = ['./src'] + sys.path
from dfs_transformer import DFSCodeSeq2SeqFC, TrainerBarlow, PubChem, get_n_files
from dfs_transformer.training.utils import seq_loss, seq_acc, collate_Barlow
import argparse
import yaml
import functools
from ml_collections import ConfigDict
from copy import deepcopy
import pickle
from sklearn.metrics import r2_score
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2*2048, rlimit[1]))
#torch.multiprocessing.set_sharing_strategy('file_system')

class EncoderModel(nn.Module):
    def __init__(self, encoder, method='cls'):
        super().__init__()
        self.encoder = encoder
        self.method = method
    
    def forward(self, inputs):
        return self.encoder.encode(inputs, method=self.method)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
    parser.add_argument('--wandb_project', type=str, default="pubchem_newdataloader")
    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--yaml_model', type=str, default="./config/selfattn/model/bert.yaml") 
    parser.add_argument('--yaml_data', type=str, default="./config/selfattn/data/pubchem10K.yaml")
    parser.add_argument('--yaml_training', type=str, default="./config/selfattn/training/barlow.yaml")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--loops', dest='loops', action='store_true')
    parser.add_argument('--no_features', dest='no_features', action='store_true')
    parser.add_argument('--overwrite', type=json.loads, default="{}")
    parser.set_defaults(loops=False, no_features=False)
    args = parser.parse_args()
    
    
    
    config = ConfigDict({'model':{}, 'data':{}, 'training':{}})
    with open(args.yaml_model) as file:
        config.model = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    with open(args.yaml_data) as file:
        config.data = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    with open(args.yaml_training) as file:
        config.training = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    
    #config.data.molecular_properties = None #["qed", "rdMolDescriptors.CalcNumHeteroatoms"]
    
    config.model.use_loops = args.loops

    for key,value in args.overwrite.items():
        for key1,value1 in value.items():
            config[key][key1] = value1
    
    if config.model.use_loops:
        config.model.max_edges += config.model.max_nodes
        
    if args.no_features:
        config.model.n_node_features = 118
        config.model.n_edge_features = 5
        config.data["no_features"] = True
    else:
        config.data["no_features"] = False
        
    if config.data.useDists:
        config.model.n_edge_features += 50 #TODO: clean this up...
    
    wandb.login(key="5c53eb61039d99e4467ef1fccd1d035bb84c1c21")
    run = wandb.init(mode=args.wandb_mode, 
                     project=args.wandb_project, 
                     entity=args.wandb_entity, 
                     name=args.name, config=config.to_dict())
    
    m = deepcopy(config.model)
    t = deepcopy(config.training)
    d = deepcopy(config.data)
    
    print(config)
    device = torch.device('cuda:%d'%config.training.gpu_id if torch.cuda.is_available() else 'cpu')
        
    encoder = DFSCodeSeq2SeqFC(**m)
    model = EncoderModel(encoder)
    
    
    if m.use_loops: # undo the thing for the dataloader
        m.max_edges -= m.max_nodes
    
    if t.load_last and t.es_path is not None:
        model.load_state_dict(torch.load(t.es_path+'checkpoint.pt', map_location=device))
    elif t.pretrained_dir is not None:
        model.load_state_dict(torch.load(t.pretrained_dir+'checkpoint.pt', map_location=device))
    
    collate_fn = collate_Barlow
    validloader = None
    if d.valid_path is not None:
        validset = PubChem(d.valid_path, max_nodes=d.max_nodes, max_edges=d.max_edges, noFeatures=d.no_features,
                           molecular_properties=d.molecular_properties, useDists=d.useDists, useHs=d.useHs)
        validloader = DataLoader(validset, batch_size=t.batch_size, shuffle=True, 
                                 pin_memory=t.pin_memory, collate_fn=collate_fn, num_workers=t.num_workers,
                                 prefetch_factor=t.prefetch_factor)
        exclude = validset.smiles
    
    trainer = TrainerBarlow(model, None, encoder.get_n_encoding(), validloader=validloader, wandb_run = run, **t)
    trainer.n_epochs = d.n_iter_per_split
    
    n_files = get_n_files(d.path)
    if d.n_used is None:
        n_splits = 1
    else:
        n_splits = n_files // d.n_used
    try:    
        for epoch in range(t.n_epochs):
            print('starting epoch %d'%(epoch+1))
            for split in range(n_splits):
                dataset = PubChem(d.path, n_used = d.n_used, max_nodes=d.max_nodes, 
                                  max_edges=d.max_edges, exclude=exclude, noFeatures=d.no_features,
                                  molecular_properties=d.molecular_properties, useDists=d.useDists, useHs=d.useHs)
                loader = DataLoader(dataset, batch_size=t.batch_size, shuffle=True, 
                                    pin_memory=t.pin_memory, collate_fn=collate_fn, num_workers=t.num_workers,
                                    prefetch_factor=t.prefetch_factor)
                trainer.loader = loader
                trainer.fit()
                if trainer.stop_training:
                    break
            if trainer.stop_training:
                break
    finally:
        print("uploading model...")
        #store config and model
        with open(trainer.es_path+'config.yaml', 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        if args.name is not None and args.wandb_mode != "offline":
            trained_model_artifact = wandb.Artifact(args.name, type="model", description="trained selfattn model")
            trained_model_artifact.add_dir(trainer.es_path)
            run.log_artifact(trained_model_artifact)
        
        
    
