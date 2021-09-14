import numpy as np
import json
import wandb
import dfs_code
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path = ['./src'] + sys.path
from dfs_transformer import DFSCodeSeq2SeqFC, Trainer, PubChem
from dfs_transformer.training.utils import seq_loss, seq_acc
import argparse
import yaml
import functools
from ml_collections import ConfigDict

def BERTize(codes, fraction_missing=0.15):
    inputs = []
    targets = []
    for code in codes:
        n = len(code)
        perm = np.random.permutation(n)
        target_idx = perm[:int(fraction_missing*n)]
        input_idx = perm[int(fraction_missing*n):]
        inp = code.clone()
        target = code.clone()
        target[input_idx] = -1
        inp[target_idx] = -1
        inputs += [inp]
        targets += [target]
    return inputs, targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default="chrisxx")
    parser.add_argument('--wandb_project', type=str, default="pubchem-bert")
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
                     name=args.name, config=config)
    
    m = config.model
    t = config.training
    d = config.data
    print(config)
    device = torch.device('cuda:%d'%config.training.gpu_id if torch.cuda.is_available() else 'cpu')
    
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    loss = functools.partial(seq_loss, ce=ce, m=m)
    
    def collate_fn(dlist):
        node_batch = [] 
        edge_batch = []
        min_code_batch = []
        for d in dlist:
            node_batch += [d.node_features]
            edge_batch += [d.edge_features]
            if config.training.mode == "min2min":
                min_code_batch += [d.min_dfs_code]
            elif config.training.mode == "rnd2rnd":
                rnd_code, rnd_index = dfs_code.rnd_dfs_code_from_torch_geometric(d, 
                                                                         d.z.numpy().tolist(), 
                                                                         np.argmax(d.edge_attr.numpy(), axis=1))
                min_code_batch += [rnd_code]
            else:
                raise ValueError("unknown config.training.mode %s"%config.training.mode)
        inputs, outputs = BERTize(min_code_batch, config.training.fraction_missing)
        targets = nn.utils.rnn.pad_sequence(outputs, padding_value=-1)
        return inputs, node_batch, edge_batch, targets 
    
    fields = ['acc-dfs1', 'acc-dfs2', 'acc-atm1', 'acc-atm2', 'acc-bnd']
    metrics = {field:functools.partial(seq_acc, idx=idx) for idx, field in enumerate(fields)}
    model = DFSCodeSeq2SeqFC(nn.Linear(m.n_node_features, m.emb_dim),
                             nn.Linear(m.n_edge_features, m.emb_dim),
                             n_atoms=m.n_atoms,
                             n_bonds=m.n_bonds, 
                             emb_dim=m.emb_dim, 
                             nhead=m.nhead, 
                             nlayers=m.nlayers, 
                             max_nodes=m.max_nodes, 
                             max_edges=m.max_edges,
                             missing_value=m.missing_value)
    
    if t.load_last and t.es_path is not None:
        model.load_state_dict(torch.load(t.es_path, map_location=device))
    elif t.pretrained_dir is not None:
        model.load_state_dict(torch.load(t.pretrained_dir, map_location=device))
    
    validloader = None
    if d.valid_path is not None:
        validset = PubChem(d.valid_path, max_nodes=m.max_nodes, max_edges=m.max_edges)
        validloader = DataLoader(validset, batch_size=d.batch_size, shuffle=True, 
                                 pin_memory=False, collate_fn=collate_fn)
        exclude = validset.smiles
    
    trainer = Trainer(model, None, loss, validloader=validloader, metrics=metrics, 
                      wandb_run = run, **t)
    trainer.n_epochs = d.n_iter_per_split
    
    for epoch in range(t.n_epochs):
        for split in range(d.n_splits):
            n_ids = d.n_files//d.n_splits
            dataset = PubChem(d.path, n_used = n_ids, max_nodes=m.max_nodes, 
                              max_edges=m.max_edges, exclude=exclude)
            loader = DataLoader(dataset, batch_size=d.batch_size, shuffle=True, 
                                pin_memory=False, collate_fn=collate_fn)
            trainer.loader = loader
            trainer.fit()
            if trainer.stop_training:
                break
        if trainer.stop_training:
            break
    trained_model_artifact = wandb.Artifact(args.name, type="model", description="trained selfattn model")
    trained_model_artifact.add_dir(trainer.es_path)
    run.log_artifact(trained_model_artifact)
        
        
    
