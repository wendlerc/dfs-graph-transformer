#!/bin/bash

for i in {0..64}
do
    start=$(expr $i \* 2060)
    python scripts/qm9/compute_dfs_codes.py with start=$start -F results &
done
