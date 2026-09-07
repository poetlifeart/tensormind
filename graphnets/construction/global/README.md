# Building the mild parent graph (16,807 nodes / 337,983 edges)

> Copied verbatim from `emergentgraph/graph_topology/README.md` (2026-05-11),
> lines 13-88. That package reproduced this graph bit-for-bit.
> The runnable driver is `build_mild_graph.sh` in this folder.


The construction pipeline has three stages. All use seed=42 and are
fully deterministic — running the same commands produces a bit-for-bit
identical graph.

**Seed graph (hardcoded in build_graph_conscious.py):**
7 nodes, 9 edges, 3-colorable (tripartite), layer sizes (2, 2, 3).
Colors: [0, 0, 1, 1, 2, 2, 2].
Edges: (0,2), (0,5), (1,3), (1,4), (1,6), (2,5), (2,6), (3,4), (3,6).

### Stage 1: Tensor product + community-aware pruning + triangle closing

```bash
python3 build_graph_conscious.py \
    --steps 4 \
    --inter-ratio 0.35 \
    --add 20168 \
    --out step1.npz
```

Builds n = 7^(k+1) = 16,807 nodes (layers 4802/4802/7203) via k=4
iterations of tensor product with the seed graph. At each iteration:
- Compute G = G_prev tensor G_seed
- Detect Louvain communities
- Prune to budget n*log2(n) edges, reserving 35% (rho=0.35) for
  inter-community edges, rest for intra-community, ranked by triangle score
- Repair minimum degree to 3
- Repair connectivity (add cross-color edges if disconnected)

After all 4 tensor steps, add 20,168 triangle-closing edges (1.2n).

Output: 273,695 edges.

### Stage 2: Box-boundary bridges

```bash
python3 graph_bridge_paper.py \
    --graph step1.npz \
    --out step2.npz \
    --max-add 84000
```

Computes all-pairs shortest paths, runs greedy MEMB box-covering at
scales lB=3,4,5,6,7, then adds up to 84,000 bridge edges (5n) between
nodes that share the same box at all scales but belong to different
Louvain communities. This shortens paths without destroying fractal
structure.

Output: 357,695 edges (2,355,878 candidates, 84,000 sampled).

### Stage 3: Fractal-targeted sparsification

```bash
python3 graph_sparsify_memb.py \
    --graph step2.npz \
    --out graph_brain_mild.npz \
    --target-db 3.8 \
    --batch 20000 \
    --min-edges 100000 \
    --min-degree 2
```

Removes low-triangle-score edges in batches of 20,000 until the
box-covering dimension dB drops to 3.8 or below. Never removes an edge
if it would drop a node below degree 2, and stops if edge count falls
below 100,000.

Output: **graph_brain_mild.npz** (16,807 nodes, 337,983 edges — one pass, removed 19,712).

### All three stages at once

```bash
bash build_mild_graph.sh
```

