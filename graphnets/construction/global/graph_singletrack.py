#!/usr/bin/env python3
"""
graph_singletrack.py -- Single-track tensor product with triangle pruning.

Uses only G1 (hub seed). Pure self-multiplication: M_k = M_{k-1} ⊗ G1.
Same triangle pruning as the dual-track version.

Usage:
    python graph_singletrack.py --seed hub --out graph_single_hub.npz
    python graph_singletrack.py --seed modular --out graph_single_modular.npz
"""

import argparse
import math
import numpy as np
import time
from collections import defaultdict
import networkx as nx


def make_seeds():
    colors = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)

    # G1: hub (node 0 connects to 5 others)
    e1 = np.array([
        [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
        [1, 2], [1, 3], [1, 4],
        [2, 4], [2, 5], [3, 6],
    ], dtype=np.int64)

    # G2: modular (balanced degree ~2-3)
    e2 = np.array([
        [0, 2], [0, 5],
        [1, 3], [1, 4], [1, 6],
        [2, 5], [2, 6],
        [3, 4], [3, 6],
    ], dtype=np.int64)

    for name, e in [("G1", e1), ("G2", e2)]:
        for u, v in e:
            assert colors[u] != colors[v], f"{name}: edge ({u},{v}) same color!"

    return (7, e1, colors), (7, e2, colors)


def tensor_product(n_m, edges_m, n_s, edges_s):
    n = n_m * n_s
    ne_m, ne_s = len(edges_m), len(edges_s)
    if ne_m == 0 or ne_s == 0:
        return n, np.empty((0, 2), dtype=np.int64)

    um = np.repeat(edges_m[:, 0], ne_s)
    vm = np.repeat(edges_m[:, 1], ne_s)
    us = np.tile(edges_s[:, 0], ne_m)
    vs = np.tile(edges_s[:, 1], ne_m)

    x1 = um * n_s + us;  y1 = vm * n_s + vs
    x2 = um * n_s + vs;  y2 = vm * n_s + us

    e1 = np.stack([np.minimum(x1, y1), np.maximum(x1, y1)], axis=1)
    e2 = np.stack([np.minimum(x2, y2), np.maximum(x2, y2)], axis=1)

    mask = e2[:, 0] != e2[:, 1]
    e2 = e2[mask]

    all_edges = np.vstack([e1, e2])
    edges = np.unique(all_edges, axis=0)
    return n, edges


def project_colors(colors_m, n_s):
    n = len(colors_m) * n_s
    colors = np.empty(n, dtype=np.int32)
    for i in range(len(colors_m)):
        colors[i * n_s:(i + 1) * n_s] = colors_m[i]
    return colors


def edge_budget(n, mode='gmean'):
    if mode == 'gmean':
        # Geometric mean between cycle (n) and complete (n(n-1)/2)
        return int(round(n * math.sqrt((n - 1) / 2)))
    else:
        return int(round(n * math.log2(n)))


def triangle_prune(n, edges, colors, target):
    t0 = time.time()
    current = len(edges)
    print(f"  Triangle pruning: {current:,} edges -> target {target:,}")

    if current <= target:
        print(f"  Already at/under budget, no pruning needed")
        return edges

    adj = defaultdict(set)
    for i in range(current):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        adj[u].add(v)
        adj[v].add(u)

    scores = np.zeros(current, dtype=np.float64)
    for i in range(current):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        tri = len(adj[u] & adj[v])
        deg_sum = len(adj[u]) + len(adj[v])
        scores[i] = tri + deg_sum * 1e-7

    if current > target:
        idx = np.argpartition(scores, -target)[-target:]
        kept_edges = edges[idx]
        order = np.lexsort((kept_edges[:, 1], kept_edges[:, 0]))
        kept_edges = kept_edges[order]
    else:
        kept_edges = edges
    print(f"  Kept {len(kept_edges):,} edges (top by triangle score)")

    edge_set = set()
    adj_kept = defaultdict(set)
    for i in range(len(kept_edges)):
        u, v = int(kept_edges[i, 0]), int(kept_edges[i, 1])
        edge_set.add((u, v))
        adj_kept[u].add(v)
        adj_kept[v].add(u)

    # Repair min degree >= 3
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
        if degrees[node] < 3:
            print(f"  WARNING: node {node} still has degree {degrees[node]} < 3 after 150 attempts")

    # Repair connectivity
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_set)
    components = list(nx.connected_components(G))
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

    final_edges = np.array(sorted(edge_set), dtype=np.int64)
    print(f"  Final: {len(final_edges):,} edges "
          f"(+{repair_added} degree repair, +{bridge_added} bridge)")
    print(f"  ({time.time()-t0:.1f}s)")
    return final_edges


def save_graph(n, edges, colors, out_path):
    layer1 = np.where(colors == 0)[0]
    layer2 = np.where(colors == 1)[0]
    layer3 = np.where(colors == 2)[0]
    n1, n2, n3 = len(layer1), len(layer2), len(layer3)

    node_layer = np.empty(n, dtype=np.int32)
    node_pos = np.empty(n, dtype=np.int32)
    for pos, node in enumerate(layer1):
        node_layer[node] = 0; node_pos[node] = pos
    for pos, node in enumerate(layer2):
        node_layer[node] = 1; node_pos[node] = pos
    for pos, node in enumerate(layer3):
        node_layer[node] = 2; node_pos[node] = pos

    mask_12 = np.zeros((n2, n1), dtype=np.float32)
    mask_13 = np.zeros((n3, n1), dtype=np.float32)
    mask_23 = np.zeros((n3, n2), dtype=np.float32)

    for i in range(len(edges)):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        lu, lv = node_layer[u], node_layer[v]
        pu, pv = node_pos[u], node_pos[v]
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
    print(f"\nSaved: {total:,} edges")
    print(f"  L1-L2: {int(mask_12.sum()):,}")
    print(f"  L1-L3: {int(mask_13.sum()):,}")
    print(f"  L2-L3: {int(mask_23.sum()):,}")

    np.savez(out_path,
             mask_12=mask_12, mask_13=mask_13, mask_23=mask_23,
             n1=np.int64(n1), n2=np.int64(n2), n3=np.int64(n3))
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=str, default='hub', choices=['hub', 'modular'],
                        help='Which seed to use (hub=G1, modular=G2)')
    parser.add_argument('--out', type=str, default='graph_single.npz')
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()

    t_total = time.time()

    (n_seed, e1, c1), (_, e2, c2) = make_seeds()

    if args.seed == 'hub':
        seed_edges = e1
        print(f"Using G1 (hub): 7 nodes, {len(e1)} edges")
    else:
        seed_edges = e2
        print(f"Using G2 (modular): 7 nodes, {len(e2)} edges")

    n = n_seed
    edges = seed_edges.copy()
    colors = c1.copy()
    print(f"\nStep 0: n={n}, edges={len(edges)}")

    # Single-track: M_k = M_{k-1} ⊗ G_seed
    for step in range(1, args.steps + 1):
        ts = time.time()
        new_n = n * n_seed
        print(f"\n{'='*65}")
        print(f"Step {step}: {n:,} x {n_seed} -> {new_n:,} nodes")
        print(f"{'='*65}")

        n, edges = tensor_product(n, edges, n_seed, seed_edges)
        colors = project_colors(colors, n_seed)

        print(f"  Edges: {len(edges):,}")
        print(f"  Step time: {time.time()-ts:.1f}s")

        bad = sum(1 for i in range(len(edges))
                  if colors[int(edges[i, 0])] == colors[int(edges[i, 1])])
        assert bad == 0, f"Step {step}: {bad} same-color edges!"

    budget = edge_budget(n)
    print(f"\nBudget: {budget:,} edges")
    edges = triangle_prune(n, edges, colors, budget)

    print(f"\n{'='*65}")
    print(f"FINAL GRAPH")
    print(f"{'='*65}")
    print(f"  Nodes: {n:,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Density: {2*len(edges)/(n*(n-1)):.6f}")

    degrees = np.zeros(n, dtype=np.int32)
    for i in range(len(edges)):
        degrees[int(edges[i, 0])] += 1
        degrees[int(edges[i, 1])] += 1
    print(f"  Avg degree: {degrees.mean():.1f}")
    print(f"  Degree range: {degrees.min()}-{degrees.max()}")

    for c in range(3):
        cnt = (colors == c).sum()
        print(f"  Layer {c+1} (color {c}): {cnt:,} nodes")

    print(f"  Total time: {time.time()-t_total:.1f}s")

    save_graph(n, edges, colors, args.out)


if __name__ == '__main__':
    main()
