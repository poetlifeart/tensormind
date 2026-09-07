#!/usr/bin/env python3
"""
graph_bridge.py — Box-boundary bridge insertion.

Adds inter-community edges ONLY between nodes that fall in the SAME
box at every scale of the greedy box-covering. These edges are invisible
to box-counting at the scales where they share a box, approximately
preserving d_B. (New edges alter shortest paths, so exact preservation
is not guaranteed.)

Algorithm:
  1. Build graph, compute full APSP
  2. Run greedy box-covering at each l_B in [3,4,5,6,7]
  3. For each l_B, record which box each node belongs to
  4. Find node pairs that are:
     a. In DIFFERENT Louvain communities
     b. In the SAME box at ALL tested scales
     c. Different colors (tripartite constraint)
     d. Not already connected
  5. Add edges from this candidate pool

References:
  Goh, Kahng et al. (PRL 2006) — skeleton theory
  Gallos, Makse, Sigman (PNAS 2012) — weak ties in fractal brain networks

Usage:
    python graph_bridge.py --graph graph_layers.npz --out graph_bridged.npz
    python graph_bridge.py --graph graph_layers.npz --out graph_bridged.npz --max-add 500
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def load_graph(path: str):
    data = np.load(path)
    n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
    n = n1 + n2 + n3
    off2, off3 = n1, n1 + n2

    edges: Set[Tuple[int, int]] = set()
    for mask_key, o1, o2 in [('mask_12', 0, off2), ('mask_13', 0, off3), ('mask_23', off2, off3)]:
        rows, cols = np.where(data[mask_key] > 0)
        for r, c in zip(rows, cols):
            edges.add((min(o1 + int(c), o2 + int(r)), max(o1 + int(c), o2 + int(r))))

    colors = np.empty(n, dtype=np.int32)
    colors[:n1] = 0
    colors[n1:off3] = 1
    colors[off3:] = 2

    return n, edges, colors, n1, n2, n3


def save_graph(edges, n1, n2, n3, out_path):
    mask_12 = np.zeros((n2, n1), dtype=np.float32)
    mask_13 = np.zeros((n3, n1), dtype=np.float32)
    mask_23 = np.zeros((n3, n2), dtype=np.float32)
    off3 = n1 + n2

    for u, v in edges:
        if u < n1:
            lu, pu = 0, u
        elif u < off3:
            lu, pu = 1, u - n1
        else:
            lu, pu = 2, u - off3
        if v < n1:
            lv, pv = 0, v
        elif v < off3:
            lv, pv = 1, v - n1
        else:
            lv, pv = 2, v - off3
        if lu == lv:
            continue
        if lu > lv:
            lu, lv = lv, lu
            pu, pv = pv, pu
        if lu == 0 and lv == 1:
            mask_12[pv, pu] = 1.0
        elif lu == 0 and lv == 2:
            mask_13[pv, pu] = 1.0
        elif lu == 1 and lv == 2:
            mask_23[pv, pu] = 1.0

    total = int(mask_12.sum() + mask_13.sum() + mask_23.sum())
    print(f"Saved: {total:,} edges -> {out_path}")
    np.savez(out_path, mask_12=mask_12, mask_13=mask_13, mask_23=mask_23,
             n1=np.int64(n1), n2=np.int64(n2), n3=np.int64(n3))


# ═══════════════════════════════════════════════════════════════════════════
# Community detection
# ═══════════════════════════════════════════════════════════════════════════


def find_communities(n: int, edges: Set[Tuple[int, int]]) -> Dict[int, int]:
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    from networkx.algorithms.community import louvain_communities
    comms = louvain_communities(G, seed=42)
    partition = {}
    for i, comm in enumerate(comms):
        for node in comm:
            partition[node] = i
    print(f"Communities: {len(comms)}")
    return partition


# ═══════════════════════════════════════════════════════════════════════════
# APSP + Greedy box-covering
# ═══════════════════════════════════════════════════════════════════════════


def compute_apsp(n: int, edges: Set[Tuple[int, int]]) -> np.ndarray:
    """Full all-pairs shortest paths via scipy sparse Dijkstra."""
    rows, cols = [], []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
    adj = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

    import time
    print("  Computing APSP...")
    t0 = time.time()
    dist = scipy.sparse.csgraph.shortest_path(adj, method='D', directed=False, unweighted=True)
    dist[~np.isfinite(dist)] = 32767
    dist = dist.astype(np.int16)
    print(f"  APSP done ({time.time()-t0:.1f}s, {dist.nbytes / 1e6:.0f} MB)")
    return dist


def memb_box_assignment(n: int, dist: np.ndarray, l_B: int,
                        n_trials: int = 50, seed: int = 42) -> np.ndarray:
    """Randomized greedy box-covering (Song et al. burning convention).
    Returns box assignment using fewest boxes across n_trials random orderings.
    Note: center selection is random permutation, not true MEMB (max excluded mass)."""
    rng = np.random.RandomState(seed)
    best_n_boxes = n + 1
    best_assignment = None

    for _ in range(n_trials):
        perm = rng.permutation(n)
        assignment = np.full(n, -1, dtype=np.int32)
        box_id = 0
        for center in perm:
            if assignment[center] >= 0:
                continue
            members = np.where(dist[center] < l_B)[0]
            unassigned = members[assignment[members] < 0]
            assignment[unassigned] = box_id
            box_id += 1
        if box_id < best_n_boxes:
            best_n_boxes = box_id
            best_assignment = assignment.copy()

    return best_assignment


# ═══════════════════════════════════════════════════════════════════════════
# Find box-boundary bridge candidates
# ═══════════════════════════════════════════════════════════════════════════


def find_candidates(n: int, edges: Set[Tuple[int, int]], colors: np.ndarray,
                    partition: Dict[int, int], dist: np.ndarray,
                    l_values: List[int], n_trials: int = 50,
                    seed: int = 42) -> List[Tuple[int, int]]:
    """Find node pairs in different communities but same box at all scales."""

    # Get box assignments at each scale
    print(f"\n  Running box-covering at {len(l_values)} scales...")
    assignments = {}
    for l_B in l_values:
        assign = memb_box_assignment(n, dist, l_B, n_trials=n_trials, seed=seed)
        n_boxes = len(set(assign))
        assignments[l_B] = assign
        print(f"    l_B={l_B}: {n_boxes} boxes")

    # Build a composite key for each node: (box_at_l2, box_at_l3, ..., box_at_l6)
    # Nodes with identical keys are in the same box at every scale
    print("  Building composite box keys...")
    node_keys = {}
    for i in range(n):
        key = tuple(int(assignments[l_B][i]) for l_B in l_values)
        node_keys[i] = key

    # Group nodes by composite key
    key_to_nodes: Dict[tuple, List[int]] = defaultdict(list)
    for node, key in node_keys.items():
        key_to_nodes[key].append(node)

    # Find cross-community pairs within each box group
    print("  Scanning for cross-community same-box pairs...")
    candidates = []
    for key, nodes in key_to_nodes.items():
        if len(nodes) < 2:
            continue
        # Group by community
        by_comm: Dict[int, List[int]] = defaultdict(list)
        for nd in nodes:
            by_comm[partition[nd]].append(nd)
        if len(by_comm) < 2:
            continue  # all same community
        # Generate cross-community pairs
        comms = list(by_comm.keys())
        for i in range(len(comms)):
            for j in range(i + 1, len(comms)):
                for u in by_comm[comms[i]]:
                    for v in by_comm[comms[j]]:
                        if colors[u] == colors[v]:
                            continue
                        e = (min(u, v), max(u, v))
                        if e not in edges:
                            candidates.append(e)

    print(f"  Found {len(candidates):,} candidate bridge edges")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Box-boundary bridge insertion")
    parser.add_argument('--graph', type=str, default='graph_layers.npz')
    parser.add_argument('--out', type=str, default='graph_bridged.npz')
    parser.add_argument('--max-add', type=int, default=500,
                        help='Maximum bridge edges to add (default 500)')
    parser.add_argument('--l-values', type=str, default='3,4,5,6,7',
                        help='Box linear sizes for box-covering (comma-separated)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Box-covering trials per scale (default 50)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    l_values = [int(x) for x in args.l_values.split(',')]

    print(f"Loading {args.graph}...")
    n, edges, colors, n1, n2, n3 = load_graph(args.graph)
    print(f"  {n:,} nodes, {len(edges):,} edges")

    bad = sum(1 for u, v in edges if colors[u] == colors[v])
    assert bad == 0, f"Input has {bad} same-color edges!"

    # Communities
    partition = find_communities(n, edges)

    # APSP
    dist = compute_apsp(n, edges)

    # Find candidates
    candidates = find_candidates(n, edges, colors, partition, dist,
                                 l_values=l_values, n_trials=args.n_trials,
                                 seed=args.seed)

    if not candidates:
        print("\nNo valid candidates found! Try relaxing l_values.")
        return

    # Shuffle and add up to max_add
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    n_add = min(args.max_add, len(candidates))
    n_before = len(edges)

    for e in candidates[:n_add]:
        edges.add(e)

    print(f"\nAdded {n_add} box-boundary bridge edges ({n_before:,} -> {len(edges):,})")

    # Verify tripartite
    bad = sum(1 for u, v in edges if colors[u] == colors[v])
    assert bad == 0, f"Output has {bad} same-color edges!"
    print("Tripartite coloring preserved.")

    # Stats on bridge nodes
    bridge_nodes = set()
    for u, v in candidates[:n_add]:
        bridge_nodes.add(u)
        bridge_nodes.add(v)
    comms_touched = set(partition[nd] for nd in bridge_nodes)
    print(f"Bridge nodes: {len(bridge_nodes)} across {len(comms_touched)} communities")

    save_graph(edges, n1, n2, n3, args.out)


if __name__ == '__main__':
    main()
