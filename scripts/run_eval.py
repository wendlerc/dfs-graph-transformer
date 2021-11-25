#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:27:33 2021

@author: chrisw
"""

import wandb 
import subprocess

wandb.login(key="5c53eb61039d99e4467ef1fccd1d035bb84c1c21")

api = wandb.Api()
for run in api.runs("dfstransformer/pubchem_plus_properties"):
    try:
        artifact = api.artifact('dfstransformer/pubchem_plus_properties/%s:latest'%run.name)
        bashCommand = 'python exp/evaluate/selfattn/moleculenet_plus_properties.py --name %s --model %s'
        bashCommand = bashCommand%(run.name, run.name)
        print('running: %s'%bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
    except:
        continue