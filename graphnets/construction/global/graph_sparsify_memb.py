#!/usr/bin/env python3
"""
graph_sparsify.py — Progressive edge removal to increase diameter and lower d_B.

Strategy: remove edges that provide redundant short paths (low triangle score,
high-degree endpoints). These are the edges that compress the diameter.
After each batch, remeasure diameter and d_B. Stop when d_B hits target.

Preserves: chi=3, connectivity, high-triangle-score edges (clustering).
"""

from __future__ import annotations
import argparse
import time
from collections import defaultdict

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


def load_graph(path: str):
    data = np.load(path)
    n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
    n = n1 + n2 + n3
    off2, off3 = n1, n1 + n2

    edges = set()
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


def compute_apsp(n, edges):
    rows, cols = [], []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
    adj = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    dist = scipy.sparse.csgraph.shortest_path(adj, method='D', directed=False, unweighted=True)
    dist[~np.isfinite(dist)] = 32767
    return dist.astype(np.int16)


def measure_db(n, dist, n_trials=30):
    finite = dist[dist < 32767].astype(np.float64)
    diam = int(finite.max())
    max_lb = min(diam, 12)
    l_values = sorted(l for l in [2, 3, 4, 5, 6, 8, 10, 12] if l <= max_lb)

    rng = np.random.RandomState(42)
    box_counts = []

    for l_B in l_values:
        best = n + 1
        for _ in range(n_trials):
            perm = rng.permutation(n)
            covered = np.zeros(n, dtype=bool)
            nb = 0
            for center in perm:
                if covered[center]:
                    continue
                covered[dist[center] <= l_B] = True  # <= not <
                nb += 1
                if covered.all():
                    break
            best = min(best, nb)
        box_counts.append(best)

    # Filter out floor points
    valid = [(l, b) for l, b in zip(l_values, box_counts) if b > 1]
    if len(valid) < 3:
        return None, None, diam, l_values, box_counts

    log_l = np.log(np.array([v[0] for v in valid], dtype=np.float64))
    log_n = np.log(np.array([v[1] for v in valid], dtype=np.float64))
    slope, intercept = np.polyfit(log_l, log_n, 1)
    d_B = -slope

    ss_res = np.sum((log_n - (slope * log_l + intercept)) ** 2)
    ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return d_B, r2, diam, l_values, box_counts


def build_nx_graph(n, edges):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def find_safe_removals(n, edges):
    """Score edges for removal, excluding bridges (cut edges).
    Returns list of (score, u, v) sorted ascending — lowest score = remove first.
    """
    import networkx as nx
    G = build_nx_graph(n, edges)

    # Find all bridges — these CANNOT be removed
    bridge_set = set()
    for u, v in nx.bridges(G):
        bridge_set.add((min(u, v), max(u, v)))
    print(f"  Bridges (protected): {len(bridge_set):,}")

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    scored = []
    for u, v in edges:
        e = (min(u, v), max(u, v))
        if e in bridge_set:
            continue  # skip bridges entirely
        tri = len(adj[u] & adj[v])
        deg_u = len(adj[u])
        deg_v = len(adj[v])
        score = tri * 100 + 1.0 / (deg_u + deg_v + 1)
        scored.append((score, u, v))

    scored.sort()
    print(f"  Removable edges: {len(scored):,}")
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, required=True)
    parser.add_argument('--out', type=str, default='graph_sparsified.npz')
    parser.add_argument('--target-db', type=float, default=3.3)
    parser.add_argument('--batch', type=int, default=10000)
    parser.add_argument('--max-remove', type=int, default=200000)
    parser.add_argument('--min-edges', type=int, default=60000,
                        help='Stop if edges drop below this')
    parser.add_argument('--min-degree', type=int, default=2,
                        help='Never remove edge if it would drop a node below this degree')
    args = parser.parse_args()

    print(f"Loading {args.graph}...")
    n, edges, colors, n1, n2, n3 = load_graph(args.graph)
    print(f"  {n:,} nodes, {len(edges):,} edges")

    # Initial measurement
    print("Computing APSP...")
    dist = compute_apsp(n, edges)
    d_B, r2, diam, l_vals, box_counts = measure_db(n, dist)
    print(f"Initial: {len(edges):,} edges, diameter={diam}, d_B={d_B:.3f}, R²={r2:.4f}")
    for l, b in zip(l_vals, box_counts):
        print(f"  l_B={l}: N_B={b}")

    if d_B is not None and d_B <= args.target_db:
        print(f"Already at target d_B <= {args.target_db}")
        return

    # Build degree tracker
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    total_removed = 0

    while total_removed < args.max_remove and len(edges) > args.min_edges:
        # Find safe edges to remove (excluding bridges), rescored each round
        print(f"\nFinding safe removals (bridges protected)...")
        scored = find_safe_removals(n, edges)

        if not scored:
            print("No more removable edges (all remaining are bridges)!")
            break

        # Remove a batch from the scored list
        batch_removed = 0
        batch_target = min(args.batch, args.max_remove - total_removed, len(scored))

        for score, u, v in scored[:batch_target]:
            e = (min(u, v), max(u, v))
            if e not in edges:
                continue
            if len(adj[u]) <= args.min_degree or len(adj[v]) <= args.min_degree:
                continue
            edges.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            batch_removed += 1

        total_removed += batch_removed

        if len(edges) < args.min_edges:
            print(f"\nHit minimum edge count ({args.min_edges:,})")
            break

        # Remeasure
        print(f"\n--- Removed {batch_removed:,} edges (total removed: {total_removed:,}, remaining: {len(edges):,}) ---")
        t0 = time.time()
        dist = compute_apsp(n, edges)
        d_B, r2, diam, l_vals, box_counts = measure_db(n, dist)
        elapsed = time.time() - t0

        if d_B is not None:
            print(f"  diameter={diam}, d_B={d_B:.3f}, R²={r2:.4f} ({elapsed:.1f}s)")
            for l, b in zip(l_vals, box_counts):
                print(f"    l_B={l}: N_B={b}")

            if d_B <= args.target_db:
                print(f"\n  TARGET REACHED: d_B={d_B:.3f} <= {args.target_db}")
                break
        else:
            print(f"  diameter={diam}, d_B=N/A (too few scaling points) ({elapsed:.1f}s)")
            for l, b in zip(l_vals, box_counts):
                print(f"    l_B={l}: N_B={b}")

    # Verify connectivity
    G_final = build_nx_graph(n, edges)
    import networkx as nx
    connected = nx.is_connected(G_final)
    print(f"\nConnected: {connected}")

    # Final summary
    print(f"\nFinal: {len(edges):,} edges (removed {total_removed:,})")
    if d_B is not None:
        print(f"  d_B={d_B:.3f}, R²={r2:.4f}, diameter={diam}")

    save_graph(edges, n1, n2, n3, args.out)


if __name__ == '__main__':
    main()
