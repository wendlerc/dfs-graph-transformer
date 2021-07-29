#!/bin/bash

for i in {0..64}
do
if [ $(((($i+1)) % 12)) -eq 0 ]
then 
    python scripts/chembl/compute_dfs_codes.py with nr=$i -F results/chembl
else
    python scripts/chembl/compute_dfs_codes.py with nr=$i -F results/chembl &
fi
done
