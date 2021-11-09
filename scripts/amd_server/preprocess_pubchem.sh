#!/bin/bash
DIR=$1
TIME_LIMIT=$2

for i in {0..63}
do
    python scripts/pubchem/compute_dfs_codes.py with use_Hs=False add_loops=True nr=$i total=64 time_limit=$TIME_LIMIT -F $DIR &> $DIR$i.log &
sleep 1;
done