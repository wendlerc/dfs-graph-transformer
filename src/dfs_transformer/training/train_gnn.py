#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:10:09 2021

@author: chrisw
"""

import torch
import tqdm
from .early_stopping import EarlyStopping
from .utils import to_cuda as utils_to_cuda
import numpy as np
import os
from collections import defaultdict
import functools
from ml_collections import ConfigDict

class WandbDummy():
    def __init__(self):
        self.config = ConfigDict({"training":{}})
    def log(*args, **kwargs):
        return
    
    

class TrainerGNN():
    #TODO: refactor such that this method takes an optimizer object instead of all these params...
    def __init__(self, model, loader, loss, validloader=None, metrics={}, 
                 scorer=None, optimizer=torch.optim.Adam, target='y',
                 n_epochs=1000, accumulate_grads=1, lr=0.0003, lr_patience=5, 
                 lr_adjustment_period=500, decay_factor=0.8, minimal_lr=6e-8, 
                 lr_argument = lambda log: log['loss'],
                 es_argument = None,
                 gpu_id=0, es_improvement=0.0, 
                 es_patience=100, es_path=None, es_period=1000, wandb_run = None, 
                 adam_betas=(0.9,0.98), adam_eps=1e-9, param_groups=None,
                 clip_gradient_norm=0.5, **kwargs):
        """
        scorer should return a real number, not a tensor
        
        data = next(iter(loader)),
        loss and metrics will be computed on model(data[:-1]), data[-1] 
        """
        if wandb_run is None:
            self.wandb = WandbDummy()
        else:
            self.wandb = wandb_run 
        self.target = target
        self.model = model
        self.loader = loader
        self.validloader = validloader
        self.loss = loss
        self.metrics = metrics
        self.scorer = scorer
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.accumulate_grads = accumulate_grads
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_adjustment_period = lr_adjustment_period
        self.lr_argument = lr_argument
        self.decay_factor = decay_factor
        self.minimal_lr = minimal_lr
        self.clip_gradient_norm = clip_gradient_norm
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            self.device = torch.device('cuda:%d'%self.gpu_id if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.es_improvement = es_improvement
        self.es_patience = es_patience
        self.es_period = es_period
        if es_path is None:
            self.es_path = "./models/tmp/%d/"%np.random.randint(100000)
            t = self.wandb.config.training
            t['es_path'] = self.es_path
            self.wandb.config.update({'training':t}, allow_val_change=True)
        else:
            self.es_path = es_path
        self.es_argument = es_argument
        
        
        if param_groups is not None:
            self.optim = self.optimizer(param_groups)
        else:
            self.optim = self.optimizer(model.parameters(), betas=adam_betas, eps=adam_eps, lr=self.lr)
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
        to_cuda = functools.partial(utils_to_cuda, device=self.device)
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
                    data = data.cuda()
                    pred = self.model(data.x, data.edge_index, data.batch)
                    loss = self.loss(pred, data[self.target])
                    loss.backward()
                    if (step+1) % self.accumulate_grads == 0:
                        if self.clip_gradient_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_gradient_norm)
                        optim.step() 
                    epoch_loss = (epoch_loss*i + loss.item())/(i+1)
                    log['batch-loss'] = loss.item() 
                    log['loss'] = epoch_loss
                    pbar_string = "Epoch %d: loss %2.6f"%(epoch+1, epoch_loss)
                    
                    self.model.eval()
                    with torch.no_grad():
                        for name, metric in self.metrics.items():
                            res = metric(pred, data[self.target])
                            epoch_metric[name] = (epoch_metric[name]*i + res.item())/(i+1)
                            log['batch-'+name] = res.item()
                            log[name] = epoch_metric[name]
                            pbar_string += " %2.4f"%epoch_metric[name]
                    
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
                        if self.scorer is not None:
                            with torch.no_grad():
                                score = self.scorer(model).item()
                            self.early_stopping(-score, model)
                            self.wandb.log({"valid-score": score})
                            
                        elif self.validloader is not None:
                            valid_loss = 0
                            valid_metric = defaultdict(float)
                            with torch.no_grad():
                                pbar_valid = tqdm.tqdm(self.validloader)
                                for i, data in enumerate(pbar_valid):
                                    valid_log = {}
                                    data = data.cuda()
                                    pred = self.model(data.x, data.edge_index, data.batch)
                                    loss = self.loss(pred, data[self.target])
                                    valid_loss = (valid_loss*i + loss.item())/(i+1)
                                    valid_log['valid-loss'] = valid_loss
                                    pbar_string = "Valid %d: loss %2.6f"%(epoch+1, valid_loss)
                                    for name, metric in self.metrics.items():
                                        res = metric(pred, data[self.target])
                                        valid_metric[name] = (valid_metric[name]*i + res.item())/(i+1)
                                        valid_log['valid-'+name] = valid_metric[name]
                                        pbar_string += " %2.4f"%res.item()
                                    pbar_valid.set_description(pbar_string)
                                self.wandb.log(valid_log)
                            if self.es_argument is None:
                                self.early_stopping(valid_loss, model)
                            else: 
                                self.early_stopping(self.es_argument(valid_log), model)
                        else:
                            if self.es_argument is None:
                                self.early_stopping(epoch_loss, model)
                            else: 
                                self.early_stopping(self.es_argument(log), model)
                        
                        if self.early_stopping.early_stop:
                            self.stop_training = True
                            break
                            
                    step += 1
                self.wandb.log({'train-'+key: value for key, value in log.items() if (key == 'loss' or key in self.metrics.keys())})
                    
        except KeyboardInterrupt:
            torch.save(model.state_dict(), self.es_path+'checkpoint_keyboardinterrupt.pt')
            self.stop_training = True
        
        
        
