#!/bin/bash
DIR=$1
TIME_LIMIT=$2

for i in {0..9}
do
    python scripts/pubchem/compute_dfs_codes.py with use_Hs=True nr=$i total=10 time_limit=$TIME_LIMIT -F $DIR &
sleep 1;
done
