#!/bin/bash
DIR=$1

for i in {0..63}
do
    python scripts/amd_server/pubchem/compute_positions_and_properties.py with use_Hs=False nr=$i total=12 path=$DIR & 
done
