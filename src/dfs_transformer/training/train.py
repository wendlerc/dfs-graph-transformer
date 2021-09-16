#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:10:09 2021

@author: chrisw
"""

import torch
import tqdm
from .early_stopping import EarlyStopping
import numpy as np
import os
from collections import defaultdict


class Trainer():
    def __init__(self, model, loader, loss, validloader=None, metrics={}, optimizer=torch.optim.Adam,
                 n_epochs=1000, accumulate_grads=1, lr=0.0003, lr_patience=5, 
                 lr_adjustment_period=500, decay_factor=0.8, minimal_lr=6e-8, 
                 lr_argument = lambda log: log['loss'], gpu_id=0, es_improvement=0.0, 
                 es_patience=100, es_path=None, es_period=1000, wandb_run = None, **kwargs):
        """
        data = next(iter(loader)),
        loss and metrics will be computed on model(data[:-1]), data[-1] 
        """
        self.model = model
        self.loader = loader
        self.validloader = validloader
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.accumulate_grads = accumulate_grads
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_adjustment_period = lr_adjustment_period
        self.lr_argument = lr_argument
        self.decay_factor = decay_factor
        self.minimal_lr = minimal_lr
        self.gpu_id = gpu_id
        self.device = torch.device('cuda:%d'%self.gpu_id if torch.cuda.is_available() else 'cpu')
        self.es_improvement = es_improvement
        self.es_patience = es_patience
        self.es_period = es_period
        if es_path is None:
            self.es_path = "./models/tmp/%d/"%np.random.randint(100000)
        else:
            self.es_path = es_path
        self.wandb = wandb_run 
        
        self.optim = self.optimizer(model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 
                                                                       mode='min', 
                                                                       verbose=True, 
                                                                       patience=lr_patience, 
                                                                       factor=decay_factor)
        self.model = self.model.to(self.device)
        os.makedirs(self.es_path, exist_ok=True)
        self.early_stopping = EarlyStopping(patience=es_patience, delta=es_improvement, path=self.es_path+'checkpoint.pt')
        self.stop_training = False
        
        
    def fit(self):
        model = self.model
        optim = self.optim
        lr_scheduler = self.lr_scheduler
        def to_cuda(T):
            if type(T) is list:
                return [t.to(self.device) for t in T]
            else:
                return T.to(self.device)
            
        try:
            step = 0
            for epoch in range(self.n_epochs):
                if self.stop_training:
                    break
                epoch_loss = 0
                epoch_metric = defaultdict(float)
                pbar = tqdm.tqdm(self.loader)
                for i, data in enumerate(pbar):
                    self.model.train()
                    log = {}
                    if step % self.accumulate_grads == 0: #bei 0 wollen wir das
                        optim.zero_grad()
                    inputs = [to_cuda(d) for d in data[:-1]]
                    output = to_cuda(data[-1])
                    pred = self.model(*inputs)
                    loss = self.loss(pred, output)
                    loss.backward()
                    if (step+1) % self.accumulate_grads == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optim.step() 
                    epoch_loss = (epoch_loss*i + loss.item())/(i+1)
                    log['batch-loss'] = loss
                    log['loss'] = epoch_loss
                    
                    pbar_string = "Epoch %d: loss %2.6f"%(epoch+1, epoch_loss)
                    for name, metric in self.metrics.items():
                        res = metric(pred, output)
                        epoch_metric[name] = (epoch_metric[name]*i + res.item())/(i+1)
                        log['batch-'+name] = res
                        log[name] = epoch_metric[name]
                        pbar_string += " %2.4f"%res
                    
                    curr_lr = list(optim.param_groups)[0]['lr']
                    log['learning rate'] = curr_lr
                    pbar.set_description(pbar_string)
                    self.wandb.log(log)
                    if step % self.lr_adjustment_period == 0:
                        
                        if self.lr_argument is not None:
                            lr_scheduler.step(self.lr_argument(log))
                        else:
                            lr_scheduler.step()
                            
                        if curr_lr < self.minimal_lr:
                            self.stop_training = True
                            break
                            
                    if (step + 1) % self.es_period == 0:
                        self.model.eval()
                        if self.validloader is not None:
                            valid_loss = 0
                            valid_metric = defaultdict(float)
                            with torch.no_grad():
                                pbar_valid = tqdm.tqdm(self.validloader)
                                for i, data in enumerate(pbar_valid):
                                    log = {}
                                    inputs = [to_cuda(d) for d in data[:-1]]
                                    output = to_cuda(data[-1])
                                    pred = self.model(*inputs)
                                    loss = self.loss(pred, output)
                                    valid_loss = (valid_loss*i + loss.item())/(i+1)
                                    log['valid-loss'] = valid_loss
                                    pbar_string = "Valid %d: loss %2.6f"%(epoch+1, epoch_loss)
                                    for name, metric in self.metrics.items():
                                        res = metric(pred, output)
                                        valid_metric[name] = (valid_metric[name]*i + res.item())/(i+1)
                                        log['valid-'+name] = valid_metric[name]
                                        pbar_string += " %2.4f"%res
                                    pbar_valid.set_description(pbar_string)
                                self.wandb.log(log)
                            self.early_stopping(valid_loss, model)
                        else:
                            self.early_stopping(epoch_loss, model)
                        
                        if self.early_stopping.early_stop:
                            self.stop_training = True
                            break
                            
                    step += 1
                self.wandb.log({'train-'+key: value for key, value in log.items() if 'batch' not in key})
                    
                
                    
            
        except KeyboardInterrupt:
            torch.save(model.state_dict(), self.es_path+'checkpoint_keyboardinterrupt.pt')
            self.stop_training = True
        
        
        