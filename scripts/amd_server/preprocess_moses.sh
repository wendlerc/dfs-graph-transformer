#!/bin/bash
DIR=$1
TIME_LIMIT=$2

for i in {0..63}
do
    python scripts/amd_server/moses/compute_dfs_codes.py with use_Hs=True add_loops=False nr=$i total=64 time_limit=$TIME_LIMIT dataset_file="~/moses.csv" -F $DIR &> $DIR$i.log &
sleep 1;
done
