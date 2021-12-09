#!/bin/bash
DIR=$1
TIME_LIMIT=$2

for i in {0..9}
do
    python scripts/amd_server/pubchem/compute_dfs_codes.py with use_Hs=True add_loops=False nr=$i total=10 start_idx=10000 max_lines=10000 time_limit=$TIME_LIMIT -F $DIR &> $DIR$i.log &
sleep 1;
done
