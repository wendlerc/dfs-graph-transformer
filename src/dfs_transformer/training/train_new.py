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

class TrainerNew():
    def __init__(self, model, loader, loss, validloader=None, metrics={}, metric_pbar_keys=None,
                 input_idxs = [0],
                 output_idxs = [1, 2],
                 scorer=None, optimizer=torch.optim.Adam,
                 n_epochs=1000, accumulate_grads=1, lr=None, lr_warmup=4000,
                 minimal_lr=6e-8, lr_factor=1.,
                 es_argument = None,
                 gpu_id=0, es_improvement=0.0, 
                 es_patience=100, es_path=None, es_period=1000, wandb_run = None, 
                 adam_betas=(0.9,0.98), adam_eps=1e-9, param_groups=None, **kwargs):
        """
        data = next(iter(loader)),
        loss and metrics are mappings from (preds, outputs) -> real number 
        """
        self.input_idxs = input_idxs
        self.output_idxs = output_idxs 
        self.model = model
        self.loader = loader
        self.validloader = validloader
        self.loss = loss
        self.metrics = metrics
        if metric_pbar_keys is None:
            self.metric_pbar_keys = list(metrics.keys())
        else:
            self.metric_pbar_keys = metric_pbar_keys
        self.scorer = scorer
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.accumulate_grads = accumulate_grads
        if lr is None:
            self.lr = 1/(model.ninp**0.5)
        else:
            self.lr = lr
        self.lr *= lr_factor
        self.lr_warmup = lr_warmup
        self.minimal_lr = minimal_lr
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
        else:
            self.es_path = es_path
        self.es_argument = es_argument
        self.wandb = wandb_run 
        
        if param_groups is not None:
            self.optim = self.optimizer(param_groups)
        else:
            self.optim = self.optimizer(model.parameters(), betas=adam_betas, eps=adam_eps, lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, 
                                                              lr_lambda = lambda t: min(1/((t+1)**0.5), (t+1)*(1/(self.lr_warmup**1.5))), 
                                                              verbose = False)
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
                    
                    data = [to_cuda(d) for d in data]
                    inputs = [data[i] for i in self.input_idxs]
                    outputs = [data[i] for i in self.output_idxs]

                    pred = self.model(*inputs)
                    loss = self.loss(pred, outputs)
                    loss.backward()
                    if (step+1) % self.accumulate_grads == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optim.step() 
                        lr_scheduler.step()
                    loss = loss.item()
                    epoch_loss = (epoch_loss*i + loss)/(i+1)
                    log['batch-loss'] = loss
                    log['loss'] = epoch_loss
                    
                    pbar_string = "Epoch %d: loss %2.6f"%(epoch+1, epoch_loss)
                    for name, metric in self.metrics.items():
                        res = metric(pred, outputs).item()
                        epoch_metric[name] = (epoch_metric[name]*i + res)/(i+1)
                        log['batch-'+name] = res
                        log[name] = epoch_metric[name]
                        if name in self.metric_pbar_keys:
                            pbar_string += " %2.4f"%res
                    
                    curr_lr = list(optim.param_groups)[0]['lr']
                    log['learning rate'] = curr_lr
                    pbar.set_description(pbar_string)
                    self.wandb.log(log)
                    del inputs
                    del outputs
                    del data
                            
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
                                    data = [to_cuda(d) for d in data]
                                    inputs = [data[i] for i in self.input_idxs]
                                    outputs = [data[i] for i in self.output_idxs]
                                    pred = self.model(*inputs)
                                    loss = self.loss(pred, outputs).item()
                                    valid_loss = (valid_loss*i + loss)/(i+1)
                                    valid_log['valid-loss'] = valid_loss
                                    pbar_string = "Valid %d: loss %2.6f"%(epoch+1, valid_loss)
                                    for name, metric in self.metrics.items():
                                        res = metric(pred, outputs).item()
                                        valid_metric[name] = (valid_metric[name]*i + res)/(i+1)
                                        valid_log['valid-'+name] = valid_metric[name]
                                        if name in self.metric_pbar_keys:
                                            pbar_string += " %2.4f"%res
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
                        del valid_log
                        del inputs
                        del outputs
                        del data
                        
                        # END CONDITIONS
                        if self.early_stopping.early_stop or curr_lr < self.minimal_lr:
                            self.stop_training = True
                            break
                        
                    step += 1
                self.wandb.log({'train-'+key: value for key, value in log.items() if (key == 'loss' or key in self.metrics.keys())})
                del log
                    
        except KeyboardInterrupt:
            torch.save(model.state_dict(), self.es_path+'checkpoint_keyboardinterrupt.pt')
            self.stop_training = True
        
        
        
