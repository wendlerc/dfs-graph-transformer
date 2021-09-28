#!/bin/bash

echo "Be careful with the cluster file system... the evaluation script stores and loads the head checkpoints"

MEMORY=40960
GPU=GeForceRTX2080Ti
FINGERPRINT=$1

for frac in 0.15 0.3 0.5 
do
for size in 10M
do 
    bsub -o bert_finetune_$frac$size.log -n 8 -W 24:00 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" -R "select[gpu_model0==$GPU]" python exp/evaluate/selfattn/finetune_moleculenet.py --overwrite '{"fingerprint" : "'$FINGERPRINT'"}' --model bert2$frac-$size-euler 
done
done 

for size in 10M
do 
    bsub -o rnd2min_finetune_$size.log -n 8 -W 24:00 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" -R "select[gpu_model0==$GPU]" python exp/evaluate/selfattn/finetune_moleculenet.py --overwrite '{"fingerprint" : "'$FINGERPRINT'"}' --model rnd2min2-$size-euler 
done

