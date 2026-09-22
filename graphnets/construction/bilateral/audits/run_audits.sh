#!/bin/bash
# Nine-criterion audit for the 8,000-node construction: parent and quotient.
# Paper claims parent 7/9 (failing the two path criteria) and quotient 9/9.
set -u
M=/home/vahid/python_programs/tensormind/graphnets/graphmetrics
B=/home/vahid/python_programs/tensormind/graphnets/construction/bilateral
cd "$M"
echo "[$(date +%T)] QUOTIENT (1,015 / 64,760)"
python3 connectome_audit_gold.py --graph "$B/graphs/cnew_coarse.npz" \
    --n-null 100 --n-null-rc 1000 --seed 42 \
    --json "$B/audits/audit_8k_quotient_100.json" > "$B/audits/audit_8k_quotient_100.log" 2>&1
echo "[$(date +%T)] quotient rc=$?"
echo "[$(date +%T)] PARENT (8,000 / 477,584)"
python3 connectome_audit_gold.py --graph "$B/graphs/cnew_parent_supernode.npz" \
    --n-null 100 --n-null-rc 1000 --seed 42 \
    --json "$B/audits/audit_8k_parent_100.json" > "$B/audits/audit_8k_parent_100.log" 2>&1
echo "[$(date +%T)] parent rc=$?"
echo "[$(date +%T)] ALL DONE"
