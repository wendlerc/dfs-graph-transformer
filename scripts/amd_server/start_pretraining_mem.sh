#!/bin/bash

MEMORY=40960
GPU=16240

for frac in 0.15 0.3 0.5 
do
for size in 10M
do 
    bsub -o bert2$frac$size.log -n 8 -W 48:00 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=$GPU]" python exp/pretrain/selfattn/pubchem.py --yaml config/selfattn/bert$size.yaml --name bertloops$frac-$size-euler --overwrite '{"training" : {"fraction_missing" : '$frac', "es_patience" : 100, "lr_patience" : 50, "es_path" : "./models/pubchem/loops/bert'$frac$size'/"}}' --loops
done
done 
#for size in 10K 100K 1M 10M
#do
#    bsub -o rnd2min$size.log -W 48:00 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" -R "select[gpu_model0==$GPU]" python exp/pretrain/selfattn/pubchem.py --yaml config/selfattn/rnd2min$size.yaml --name rnd2min-$size-euler
#done

  
bsub -o rnd2min2$size.log -n 8 -W 48:00 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=$GPU]" python exp/pretrain/selfattn/pubchem.py --yaml config/selfattn/rnd2min$size.yaml --name rnd2minloops-$size-euler --overwrite '{"training" : {"fraction_missing" : '$frac', "es_patience" : 100, "lr_patience" : 50, "es_path" : "./models/pubchem/loops/rnd2min'$size'/"}}' --loops
  
  
