#!/bin/bash

for i in {0..63}
do
if [ $(((($i+1)) % 12)) -eq 0 ]
then 
    python scripts/chembl/compute_dfs_codes.py with nr=$i time_limit=600 -F results/chembl &
else
    python scripts/chembl/compute_dfs_codes.py with nr=$i time_limit=600 -F results/chembl & 
fi
sleep 1;
done
