#!/bin/bash
DIR=$1
TIME_LIMIT=$2
N=$3

mkdir $DIR;
for i in {0..9}
do
    python scripts/pubchem/compute_dfs_codes.py with use_Hs=False add_loops=False nr=$i total=10 time_limit=$TIME_LIMIT max_lines=$N -F $DIR &> $DIR$i.log &
    sleep 1;
done
