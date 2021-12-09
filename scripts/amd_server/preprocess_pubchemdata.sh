#!/bin/bash
DIR=$1

for i in {0..9}
do
    python scripts/amd_server/pubchem/compute_data.py with use_Hs=True add_loops=False nr=$i total=10 path=$DIR & 
done
