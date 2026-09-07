#!/bin/bash
# Reproduce graph_brain_mild.npz — the paper's canonical pipeline.
#
# Parameters (all from paper Section 2.1):
#   k=4 tensor steps -> n=16,807 nodes
#   Edge budget: n*log2(n) per tensor step
#   Inter-community quota: rho=0.35
#   Triangle closing: 1.2n = 20,168 edges
#   Box-boundary bridges: 5n = 84,000
#   Sparsification: remove low-triangle-score edges until d_B <= 3.8
#
# Requirements: Python 3.8+, numpy, scipy, networkx

set -e

# Run from this script's own directory so outputs land beside it,
# where quotient/save_v146_npz.py and quotient/lift/ expect them.
cd "$(dirname "$0")"

echo "Stage 1: Tensor product + community-aware pruning + triangle closing"
python3 build_graph_conscious.py \
    --steps 4 \
    --inter-ratio 0.35 \
    --add 20168 \
    --out step1.npz

echo "Stage 2: Box-boundary bridges (5n = 84,000)"
python3 graph_bridge_paper.py \
    --graph step1.npz \
    --out step2.npz \
    --max-add 84000

echo "Stage 3: Fractal-targeted sparsification (d_B <= 3.8)"
python3 graph_sparsify_memb.py \
    --graph step2.npz \
    --out graph_brain_mild.npz \
    --target-db 3.8 \
    --batch 20000 \
    --min-edges 100000 \
    --min-degree 2

echo "Done. Output: graph_brain_mild.npz"
