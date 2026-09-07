#!/usr/bin/env python3
"""
build_graph_conscious.py — Build brain-like graph with community-aware pruning at each tensor step.

Key change from build_graph_layers.py: instead of pruning only at the end by
triangle score (which kills inter-community edges), we prune after EACH tensor
step with a quota that preserves inter-community edges.

This bakes cross-community connectivity into the fractal structure at every scale,
rather than bolting it on after the fact.

Pipeline (this script):
  For each tensor step:
    1. Tensor product → edge explosion
    2. Detect communities (Louvain)
    3. Split edges into intra-community and inter-community
    4. Keep top edges from each pool (by triangle score) with inter-community quota
    5. Budget = n * log2(n) at each step
  Then:
    6. Add triangle-closing edges
    7. Save graph

  Optional follow-up (separate scripts, not run by this file):
    - graph_bridge.py: box-boundary bridges
    - graph_add_triangles.py: additional triangle-closing

Usage:
    python build_graph_conscious.py --out graph_conscious.npz --inter-ratio 0.3
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import numpy as np
import networkx as nx

from graph_singletrack import make_seeds, tensor_product, project_colors, save_graph


def community_aware_prune(n, edges, colors, target, inter_ratio=0.3):
    """Prune edges while preserving a quota of inter-community edges."""
    t0 = time.time()
    current = len(edges)
    print(f"  Community-aware pruning: {current:,} edges -> target {target:,}")

    if current <= target:
        print(f"  Already at/under budget, no pruning needed")
        return edges

    # Detect communities
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(current):
        G.add_edge(int(edges[i, 0]), int(edges[i, 1]))

    from networkx.algorithms.community import louvain_communities
    comms = louvain_communities(G, seed=42)
    partition = {}
    for ci, comm in enumerate(comms):
        for node in comm:
            partition[node] = ci
    print(f"  Communities: {len(comms)}")

    # Build adjacency for triangle counting
    adj = defaultdict(set)
    for i in range(current):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        adj[u].add(v)
        adj[v].add(u)

    # Score all edges and split into intra/inter
    intra_idx = []
    inter_idx = []
    scores = np.zeros(current, dtype=np.float64)

    for i in range(current):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        tri = len(adj[u] & adj[v])
        deg_sum = len(adj[u]) + len(adj[v])
        scores[i] = tri + deg_sum * 1e-7

        if partition[u] == partition[v]:
            intra_idx.append(i)
        else:
            inter_idx.append(i)

    intra_idx = np.array(intra_idx)
    inter_idx = np.array(inter_idx)
    print(f"  Intra-community: {len(intra_idx):,}, Inter-community: {len(inter_idx):,}")

    # Allocate budget with rebalancing
    inter_budget = min(int(target * inter_ratio), len(inter_idx))
    intra_budget = min(target - inter_budget, len(intra_idx))
    # If intra pool was scarce, redirect surplus back to inter
    if intra_budget < target - inter_budget:
        inter_budget = min(target - intra_budget, len(inter_idx))

    print(f"  Budget: {intra_budget:,} intra + {inter_budget:,} inter = {intra_budget + inter_budget:,}")

    # Select top edges from each pool
    kept_indices = []

    if len(intra_idx) > 0 and intra_budget > 0:
        intra_scores = scores[intra_idx]
        if len(intra_idx) > intra_budget:
            top = np.argpartition(intra_scores, -intra_budget)[-intra_budget:]
            kept_indices.append(intra_idx[top])
        else:
            kept_indices.append(intra_idx)

    if len(inter_idx) > 0 and inter_budget > 0:
        inter_scores = scores[inter_idx]
        if len(inter_idx) > inter_budget:
            top = np.argpartition(inter_scores, -inter_budget)[-inter_budget:]
            kept_indices.append(inter_idx[top])
        else:
            kept_indices.append(inter_idx)

    kept_indices = np.concatenate(kept_indices)
    kept_edges = edges[kept_indices]
    order = np.lexsort((kept_edges[:, 1], kept_edges[:, 0]))
    kept_edges = kept_edges[order]
    print(f"  Kept {len(kept_edges):,} edges")

    # Repair min degree >= 3
    edge_set = set()
    adj_kept = defaultdict(set)
    for i in range(len(kept_edges)):
        u, v = int(kept_edges[i, 0]), int(kept_edges[i, 1])
        edge_set.add((u, v))
        adj_kept[u].add(v)
        adj_kept[v].add(u)

    degrees = np.array([len(adj_kept[i]) for i in range(n)], dtype=np.int32)
    nodes_by_color = {c: np.where(colors == c)[0] for c in range(3)}
    rng = np.random.default_rng(42)
    repair_added = 0
    under = np.where(degrees < 3)[0]
    for node in under:
        node = int(node)
        my_color = int(colors[node])
        pool = np.concatenate([nodes_by_color[c] for c in range(3) if c != my_color])
        att = 0
        while degrees[node] < 3 and att < 150:
            v = int(pool[rng.integers(len(pool))])
            e = (min(node, v), max(node, v))
            if e not in edge_set:
                edge_set.add(e)
                adj_kept[node].add(v)
                adj_kept[v].add(node)
                degrees[node] += 1
                degrees[v] += 1
                repair_added += 1
            att += 1
        # Fallback: exhaustive search if random sampling failed
        if degrees[node] < 3:
            perm = rng.permutation(pool)
            for v in perm:
                if degrees[node] >= 3:
                    break
                v = int(v)
                e = (min(node, v), max(node, v))
                if e not in edge_set:
                    edge_set.add(e)
                    adj_kept[node].add(v)
                    adj_kept[v].add(node)
                    degrees[node] += 1
                    degrees[v] += 1
                    repair_added += 1
            if degrees[node] < 3:
                print(f"  WARNING: node {node} still has degree {degrees[node]} < 3 after exhaustive search")

    # Repair connectivity
    G2 = nx.Graph()
    G2.add_nodes_from(range(n))
    G2.add_edges_from(edge_set)
    components = list(nx.connected_components(G2))
    bridge_added = 0
    if len(components) > 1:
        components.sort(key=len, reverse=True)
        main_comp = components[0]
        for comp in components[1:]:
            comp_list = list(comp)
            rng.shuffle(comp_list)
            connected = False
            for u in comp_list:
                if connected:
                    break
                my_color = int(colors[u])
                candidates = [v for v in main_comp if colors[v] != my_color]
                sample = candidates[:200]
                rng.shuffle(sample)
                for v in sample[:20]:
                    e = (min(u, v), max(u, v))
                    if e not in edge_set:
                        edge_set.add(e)
                        bridge_added += 1
                        connected = True
                        break
            if connected:
                main_comp = main_comp | comp
            else:
                print(f"  WARNING: failed to connect component of size {len(comp)}")

    final_edges = np.array(sorted(edge_set), dtype=np.int64)
    print(f"  Final: {len(final_edges):,} edges "
          f"(+{repair_added} degree repair, +{bridge_added} bridge)")
    print(f"  ({time.time()-t0:.1f}s)")
    return final_edges


def close_triangles(n, edge_set, colors, max_add, seed=42):
    """Close triangles respecting 3-coloring."""
    import random
    rng = random.Random(seed)

    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    nodes = list(range(n))
    rng.shuffle(nodes)

    candidates = set()
    for v in nodes:
        nbrs = list(adj[v])
        if len(nbrs) < 2:
            continue
        sample = nbrs if len(nbrs) <= 50 else rng.sample(nbrs, 50)
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                u, w = sample[i], sample[j]
                if w not in adj[u] and colors[u] != colors[w]:
                    candidates.add((min(u, w), max(u, w)))
        if len(candidates) > max_add * 10:
            break

    candidates -= edge_set
    candidates = list(candidates)
    rng.shuffle(candidates)

    added = 0
    for e in candidates[:max_add]:
        edge_set.add(e)
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
        added += 1

    print(f"  Added {added:,} triangle-closing edges (total: {len(edge_set):,})")
    return edge_set


def main():
    parser = argparse.ArgumentParser(description="Build brain-like graph with community-aware pruning")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--inter-ratio", type=float, default=0.3,
                        help="Fraction of edge budget reserved for inter-community edges (default 0.3)")
    parser.add_argument("--prune-mult", type=float, nargs='+', default=None,
                        help="Budget multiplier per step (e.g. --prune-mult 3 2 1.5 1). "
                             "Length must match --steps. Default: 1.0 at every step.")
    parser.add_argument("--add", type=int, default=10000, help="Triangle-closing edges to add")
    parser.add_argument("--out", type=str, default="graph_conscious.npz")
    args = parser.parse_args()

    t_total = time.time()

    (n_seed, _e1, c1), (_, e2, _c2) = make_seeds()
    seed_edges = e2

    n = n_seed
    edges = seed_edges.copy()
    colors = c1.copy()

    # Budget multipliers per step
    if args.prune_mult is not None:
        assert len(args.prune_mult) == args.steps, \
            f"--prune-mult needs {args.steps} values, got {len(args.prune_mult)}"
        prune_mult = args.prune_mult
    else:
        prune_mult = [1.0] * args.steps

    print(f"Seed G2: {n_seed} nodes, {len(seed_edges)} edges")
    print(f"Inter-community ratio: {args.inter_ratio}")
    print(f"Prune multipliers: {prune_mult}")
    print(f"Step 0: n={n:,}, edges={len(edges):,}")

    for step in range(1, args.steps + 1):
        print(f"\n{'=' * 64}")
        print(f"Tensor step {step}: {n:,} x {n_seed} -> {n * n_seed:,} nodes")
        print(f"{'=' * 64}")

        n, edges = tensor_product(n, edges, n_seed, seed_edges)
        colors = project_colors(colors, n_seed)

        print(f"  Edges after product: {len(edges):,}")

        bad = sum(1 for i in range(len(edges)) if colors[int(edges[i, 0])] == colors[int(edges[i, 1])])
        assert bad == 0, f"Step {step}: {bad} same-color edges!"

        # Prune at each step with community awareness
        mult = prune_mult[step - 1]
        budget = int(round(n * math.log2(n) * mult))
        if len(edges) > budget:
            print(f"  Budget (n * log2(n) * {mult}): {budget:,}")
            edges = community_aware_prune(n, edges, colors, budget,
                                          inter_ratio=args.inter_ratio)

    # Triangle closing
    print(f"\nAdding triangle-closing edges: up to {args.add:,}")
    edge_set = {(int(u), int(v)) for u, v in np.asarray(edges, dtype=np.int64)}
    edge_set = close_triangles(n, edge_set, colors, max_add=args.add)

    # Save
    n1 = int((colors == 0).sum())
    n2 = int((colors == 1).sum())
    n3 = int((colors == 2).sum())

    from graph_add_triangles import save_graph as save_tri_graph
    save_tri_graph(edge_set, n1, n2, n3, args.out)

    print(f"\nDone in {time.time()-t_total:.1f}s")
    print(f"  Nodes: {n:,}")
    print(f"  Edges: {len(edge_set):,}")
    print(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
