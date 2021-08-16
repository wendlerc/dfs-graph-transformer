#!/bin/bash
DIR=$1
N_CORES=$2
MAX_NODES=$4
TIME_LIMIT=$3

for i in {0..63}
do
    python scripts/chembl/compute_dfs_codes.py with nr=$i time_limit=$TIME_LIMIT max_nodes=$MAX_NODES -F $DIR &
sleep 1;
done
