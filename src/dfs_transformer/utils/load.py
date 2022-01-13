#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:12:20 2022

@author: chrisw
"""

import wandb
import yaml
import torch
from collections import OrderedDict
from ml_collections import ConfigDict
from .. import DFSCodeSeq2SeqFC

def load_selfattn_wandb(pretrained_model,
                        pretrained_entity="dfstransformer", 
                        pretrained_project="pubchem_newdataloader",
                        wandb_dir="./wandb",
                        strict=False,
                        device="cpu"):
    # download pretrained model
    run = wandb.init(mode="online",
                     project=pretrained_project, 
                     entity=pretrained_entity, 
                     job_type="inference",
                     dir=wandb_dir,
                     settings=wandb.Settings(start_method='fork'))
    model_at = run.use_artifact(pretrained_model + ":latest")
    model_dir = model_at.download(root=wandb_dir+'/artifacts/%s/'%pretrained_model)
    run.finish()
    return load_DFSCodeSeq2SeqFC(model_dir, device, strict=strict)


def load_DFSCodeSeq2SeqFC(model_dir, device="cpu", strict=False):
    with open(model_dir+"/config.yaml") as file:
        cfg = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))
    model = DFSCodeSeq2SeqFC(**cfg.model)
    weights = torch.load(model_dir+"/checkpoint.pt", map_location=device)
    params = OrderedDict()
    for target_key in model.state_dict().keys():
        for key, value in weights.items():
            if target_key in key:
                params[target_key] = value
    model.load_state_dict(params, strict=strict)
    return model, cfg


def load_selfattn_local(model_dir, device="cpu", strict=False):
    return load_DFSCodeSeq2SeqFC(model_dir, device=device, strict=strict)