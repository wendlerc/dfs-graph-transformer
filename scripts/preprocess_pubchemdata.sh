#!/bin/bash
DIR=$1
TIME_LIMIT=$2

for i in {0..63}
do
    python scripts/pubchem/compute_data.py with use_Hs=False add_loops=False nr=$i total=64 path=$DIR & 
done
