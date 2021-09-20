#!/bin/bash

MEMORY=40960
GPU=GeForceRTX2080Ti

for size in 10K 100K 1M
do 
    bsub -o bert$size.log -W 48:00 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" -R "select[gpu_model0==$GPU]" python exp/pretrain/selfattn/pubchem.py --yaml config/selfattn/bert$size.yaml --name bert-$size-euler
done

for size in 10K 100K 1M 10M
do
    bsub -o rnd2min$size.log -W 48:00 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" -R "select[gpu_model0==$GPU]" python exp/pretrain/selfattn/pubchem.py --yaml config/selfattn/rnd2min$size.yaml --name rnd2min-$size-euler
done
