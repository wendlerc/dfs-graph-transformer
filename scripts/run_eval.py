#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:27:33 2021

@author: chrisw


usage example: 
    
    python scripts/run_eval.py "bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 1 -W 4:00 -R 'rusage[mem=30000, ngpus_excl_p=1]' -R 'select[gpu_mtotal0>=10000]' python exp/evaluate/selfattn/finetune_moleculenet.py --wandb_project moleculenet10-finetune-cluster --model"
    
    python scripts/run_eval.py --pretrained_project dfstransformer/pubchem_newencoding "bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 1 -W 4:00 -R 'rusage[mem=30000, ngpus_excl_p=1]' -R 'select[gpu_mtotal0>=10000]' python exp/evaluate/selfattn/finetune_moleculenet.py --wandb_project moleculenet10-finetune-newencoding --model"

"""

import wandb 
import subprocess
import argparse
import shlex

print("starting evaluation script...")
parser = argparse.ArgumentParser()
parser.add_argument('bashcommand', type=str, help="bash \"bashcommand <pretrainedmodel>; will be run")
parser.add_argument('--pretrained_project', type=str, default="dfstransformer/pubchem_newencoding")
parser.add_argument('--target_project', type=str, default="dfstransformer/moleculenet10-finetune-newencoding")
args = parser.parse_args()

print("logging into wandb...")
try:
    wandb.login(key="5c53eb61039d99e4467ef1fccd1d035bb84c1c21")
except:
    print('already logged in')

print("running experiments...")
api = wandb.Api()

exclude = [run.name for run in api.runs(args.target_project)]

for run in api.runs(args.pretrained_project):
    try:
        if run.name in exclude:
            continue
        artifact = api.artifact(args.pretrained_project+"/%s:latest"%run.name)
        bashCommand = args.bashcommand+' %s'
        bashCommand = bashCommand%(run.name)
        print('running: %s'%bashCommand)
        process = subprocess.Popen(shlex.split(bashCommand), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
    except Exception as e:
        print("Exception: {0}".format(e))
        continue
