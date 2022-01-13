#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:27:33 2021

@author: chrisw
"""

import wandb 
import subprocess
import argparse
import shlex

print("starting evaluation script...")
parser = argparse.ArgumentParser()
parser.add_argument('bashcommand', type=str, help="bash \"bashcommand <pretrainedmodel>; will be run")
parser.add_argument('--pretrained_project', type=str, default="pubchem_newloader")
args = parser.parse_args()

print("logging into wandb...")
try:
    wandb.login(key="5c53eb61039d99e4467ef1fccd1d035bb84c1c21")
except:
    print('already logged in')

print("running experiments...")
api = wandb.Api()
for run in api.runs(args.pretrained_project):
    try:
        artifact = api.artifact(args.pretrained_project+"%s:latest"%run.name)
        bashCommand = args.bashcommand+' %s'
        bashCommand = bashCommand%(run.name)
        print('running: %s'%bashCommand)
        print(shlex.split(bashCommand))
        process = subprocess.Popen(shlex.split(bashCommand), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
    except:
        continue
