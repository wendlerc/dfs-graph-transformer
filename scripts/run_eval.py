#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:27:33 2021

@author: chrisw
"""

import wandb 
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('bashcommand', type=str, help="bash \"bashcommand <pretrainedmodel>; will be run")
args = parser.parse_args()

try:
    wandb.login(key="5c53eb61039d99e4467ef1fccd1d035bb84c1c21")
except:
    print('already logged in')

api = wandb.Api()
for run in api.runs("dfstransformer/pubchem_plus_properties"):
    try:
        artifact = api.artifact('dfstransformer/pubchem_plus_properties/%s:latest'%run.name)
        bashCommand = args.bashcommand+' %s'
        bashCommand = bashCommand%(run.name)
        print('running: %s'%bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
    except:
        continue
