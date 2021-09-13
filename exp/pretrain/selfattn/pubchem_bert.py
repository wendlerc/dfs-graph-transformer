import numpy as np
import wandb
import dfs_code
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path = ['./src'] + sys.path
from dfs_transformer import DFSCodeSeq2SeqFC, Trainer, PubChem
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
    parser.add_argument('--yaml', type=str, default="./config/selfattn/bert.yaml") 
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.yaml) as file:
        config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

    m = config.model
    t = config.training
    d = config.data
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.name, config=config)

    print(config)
    device = torch.device('cuda:%d'%config.training.gpu_id if torch.cuda.is_available() else 'cpu')
    
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    
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
    
    def loss(pred, target, ce=ce):
        dfs1, dfs2, atm1, atm2, bnd = pred
        pred_dfs1 = torch.reshape(dfs1, (-1, m.max_nodes))
        pred_dfs2 = torch.reshape(dfs2, (-1, m.max_nodes))
        pred_atm1 = torch.reshape(atm1, (-1, m.n_atoms))
        pred_atm2 = torch.reshape(atm2, (-1, m.n_atoms))
        pred_bnd = torch.reshape(bnd, (-1, m.n_bonds))
        tgt_dfs1 = target[:, :, 0].view(-1)
        tgt_dfs2 = target[:, :, 1].view(-1)
        tgt_atm1 = target[:, :, 2].view(-1)
        tgt_atm2 = target[:, :, 4].view(-1)
        tgt_bnd = target[:, :, 3].view(-1)
        loss = ce(pred_dfs1, tgt_dfs1) 
        loss += ce(pred_dfs2, tgt_dfs2)
        loss += ce(pred_atm1, tgt_atm1)
        loss += ce(pred_bnd, tgt_bnd)
        loss += ce(pred_atm2, tgt_atm2)
        return loss 
    
    def acc(pred, target, idx=0):
        dfs1, dfs2, atm1, atm2, bnd = pred
        tgt = target[:, :, idx].view(-1)
        prd = pred[idx].reshape(tgt.shape[0], -1)
        mask = tgt != -1
        n_tgts = torch.sum(mask)
        acc = (torch.argmax(prd[mask], axis=1) == tgt[mask]).sum()/n_tgts
        return acc
    
    fields = ['dfs1-acc', 'dfs2-acc', 'atm1-acc', 'atm2-acc', 'bnd-acc']
    metrics = {field:functools.partial(acc, idx=idx) for idx, field in enumerate(fields)}
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
    
    trainer = Trainer(model, None, loss, metrics=metrics, wandb_run = run, **t)
    trainer.n_epochs = d.n_iter_per_split
    
    for epoch in range(t.n_epochs):
        for split in range(d.n_splits):
            n_ids = d.n_files//d.n_splits
            dataset = PubChem(d.path, n_used = n_ids, max_nodes=m.max_nodes, 
                              max_edges=m.max_edges)
            loader = DataLoader(dataset, batch_size=d.batch_size, shuffle=True, 
                                pin_memory=False, collate_fn=collate_fn)
            trainer.loader = loader
            trainer.fit()
        
        
        
    
