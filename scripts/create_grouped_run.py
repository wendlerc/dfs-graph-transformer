#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:11:12 2022

@author: chrisw
"""

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('wandb_group', type=str)
parser.add_argument('--wandb_entity', type=str, default="dfstransformer")
parser.add_argument('--wandb_project', type=str, default="karateclub-rep100")
parser.add_argument('--wandb_mode', type=str, default="online")
parser.add_argument('--wandb_dir', type=str, default="./wandb")
parser.add_argument('--share', type=str, default="ls_krausea")
parser.add_argument('--logdir', type=str, default="/cluster/scratch/wendlerc/lsf_gtrans/")
parser.add_argument('--n_cpus', type=int, default=5)
parser.add_argument('--cpu_memory', type=int, default=30000)
parser.add_argument('--n_gpus', type=int, default=1)
parser.add_argument('--gpu_memory', type=int, default=16000)
parser.add_argument('--time', type=str, default="4:00")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=1)
parser.add_argument('--command', type=str, default="python exp/evaluate/selfattn/karateclub.py")
args = parser.parse_args()

assert (args.end - args.start) % args.stepsize == 0

for i in range(args.start, args.end, args.stepsize):
    print('bsub -G %s -o %s -n %d -W %s -R ' \
          '"rusage[mem=%d, ngpus_excl_p=%d]" -R "select[gpu_mtotal0>=%d]" '\
          '%s --start %d --end %d'%(args.share,
                                    args.logdir,
                                    args.n_cpus,
                                    args.time,
                                    args.cpu_memory,
                                    args.n_gpus, 
                                    args.gpu_memory,
                                    args.command, 
                                    i, i+args.stepsize))
    