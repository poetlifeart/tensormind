#!/usr/bin/env python3
"""
Add triangle-closing edges on top of an existing graph.
No removal, just add. Respects 3-coloring.
"""

import argparse
import random
import numpy as np
from collections import defaultdict


def load_graph(path):
    data = np.load(path)
    n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
    n = n1 + n2 + n3
    off1, off2, off3 = 0, n1, n1 + n2

    edges = set()
    for mask_key, o1, o2 in [('mask_12', off1, off2), ('mask_13', off1, off3), ('mask_23', off2, off3)]:
        rows, cols = np.where(data[mask_key] > 0)
        for r, c in zip(rows, cols):
            edges.add((min(o1 + c, o2 + r), max(o1 + c, o2 + r)))

    colors = np.empty(n, dtype=np.int32)
    colors[:n1] = 0
    colors[n1:n1+n2] = 1
    colors[n1+n2:] = 2

    print(f"Loaded: {n:,} nodes, {len(edges):,} edges")
    return n, edges, colors, n1, n2, n3


def close_triangles(n, edges, colors, max_add, seed=42):
    rng = random.Random(seed)

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Find open wedges (u-v-w where u,w not connected, different colors)
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

    candidates -= edges
    candidates = list(candidates)
    rng.shuffle(candidates)

    added = 0
    for e in candidates[:max_add]:
        edges.add(e)
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
        added += 1

    print(f"Added {added:,} triangle-closing edges")
    print(f"Total: {len(edges):,} edges")
    return edges


def save_graph(edges, n1, n2, n3, out_path):
    mask_12 = np.zeros((n2, n1), dtype=np.float32)
    mask_13 = np.zeros((n3, n1), dtype=np.float32)
    mask_23 = np.zeros((n3, n2), dtype=np.float32)
    off2, off3 = n1, n1 + n2

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--add', type=int, default=20000, help='Max triangles to add')
    args = parser.parse_args()

    n, edges, colors, n1, n2, n3 = load_graph(args.graph)
    edges = close_triangles(n, edges, colors, args.add)
    save_graph(edges, n1, n2, n3, args.out)
