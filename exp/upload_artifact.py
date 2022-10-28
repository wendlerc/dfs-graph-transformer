#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:03:37 2021

@author: chrisw
"""

import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
parser.add_argument('--wandb_project', type=str, default="pubchem")
parser.add_argument('--wandb_mode', type=str, default="online")
parser.add_argument('name', type=str)
parser.add_argument('path', type=str) 
parser.set_defaults(loops=False)
args = parser.parse_args()
    


wandb.login(key="5c53eb61039d99e4467ef1fccd1d035bb84c1c21")
run = wandb.init(mode=args.wandb_mode, 
                 project=args.wandb_project, 
                 entity=args.wandb_entity, 
                 name=args.name, job_type="upload artefact")

if args.name is not None and args.wandb_mode != "offline":
    trained_model_artifact = wandb.Artifact(args.name, type="model", description="trained selfattn model")
    trained_model_artifact.add_dir(args.path)
    run.log_artifact(trained_model_artifact)