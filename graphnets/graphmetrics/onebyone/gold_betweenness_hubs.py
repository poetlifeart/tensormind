#!/usr/bin/env python3
"""
gold_betweenness_hubs.py — the 'betweenness' block of connectome_audit_gold.py
(descriptive, not scored), on its own.

_betweenness_hubs(G, k=200, seed=seed) on the WHOLE graph (not its GCC), as in
the audit:

    bc     = nx.betweenness_centrality(G, k=min(200, n), seed=seed)
             shortest-path node betweenness (Freeman 1977; Brandes 2001),
             normalized by (n-1)(n-2)/2 for an undirected graph, endpoints
             excluded; estimated from k sampled source nodes when n > 200
             (Brandes & Pich 2007), EXACT when n <= 200 (every node is a source)
    mean, max, std (ddof=0) of bc over all nodes,  max_over_mean = max / mean,
    n_hubs = number of nodes with bc > 2 * mean

No null model.  Self-contained: needs numpy and networkx only, and does not
import connectome_audit_gold.py.  The loading functions and _betweenness_hubs
below are copied from it unchanged, so with the same --graph and --seed the
result is the same as the 'betweenness' entry of the full audit.  The driver
also lists the top nodes (from the same nx.betweenness_centrality call, which
the audit does not keep).

Usage:
    python gold_betweenness_hubs.py --graph G.npz
    python gold_betweenness_hubs.py --graph G.npz --seed 42 --json betweenness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
# Loading  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LayeredGraph:
    G: nx.Graph
    masks: Dict[str, np.ndarray]
    sizes: Tuple[int, int, int]
    has_edge_list: bool = False


def load_layered_graph(path: str) -> LayeredGraph:
    if path.endswith('.gml'):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
        n = G.number_of_nodes()
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
        masks = {k: np.zeros((1, 1), dtype=np.uint8) for k in ['12', '13', '23']}
        print(f"  [Loaded GML: {n} nodes, {G.number_of_edges()} edges]")
        return LayeredGraph(G=G, masks=masks, sizes=(n1, n2, n3), has_edge_list=True)

    data = np.load(path)

    # Check for edge-list format FIRST (χ>3 graphs may lack masks)
    if 'edges' in data:
        n_total = int(data['n_total']) if 'n_total' in data else (
            int(data['n1']) + int(data['n2']) + int(data['n3']))
        edges = data['edges']
        G = nx.Graph()
        G.add_nodes_from(range(n_total))
        G.add_edges_from((int(u), int(v)) for u, v in edges)
        print(f"  [Loaded from edge list: {G.number_of_edges()} edges]")
        if 'n1' in data:
            n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        else:
            n1 = n_total // 3
            n2 = n_total // 3
            n3 = n_total - 2 * (n_total // 3)
        masks = {}
        for key in ['mask_12', 'mask_13', 'mask_23']:
            if key in data:
                masks[key[5:]] = (data[key] > 0).astype(np.uint8)
            else:
                masks[key[5:]] = np.zeros((1, 1), dtype=np.uint8)
        return LayeredGraph(G=G, masks=masks,
                            sizes=(n1, n2, n3), has_edge_list=True)

    # Standard mask-based loading (χ=3 tripartite graphs)
    n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
    mask_12 = (data['mask_12'] > 0).astype(np.uint8)
    mask_13 = (data['mask_13'] > 0).astype(np.uint8)
    mask_23 = (data['mask_23'] > 0).astype(np.uint8)
    G = _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23)
    return LayeredGraph(G=G, masks={'12': mask_12, '13': mask_13, '23': mask_23},
                        sizes=(n1, n2, n3), has_edge_list=False)


def _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23):
    G = nx.Graph()
    off1, off2, off3 = 0, n1, n1 + n2
    G.add_nodes_from(range(n1 + n2 + n3))
    rows, cols = np.where(mask_12)
    G.add_edges_from((off1 + int(c), off2 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_13)
    G.add_edges_from((off1 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_23)
    G.add_edges_from((off2 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Betweenness centrality hubs  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _betweenness_hubs(G: nx.Graph, k: int = 200, seed: int = 42) -> Dict[str, float]:
    bc = nx.betweenness_centrality(G, k=min(k, G.number_of_nodes()), seed=seed)
    vals = np.array(list(bc.values()), dtype=float)
    if vals.size == 0 or vals.mean() == 0:
        return {'mean': 0, 'max': 0, 'std': 0, 'max_over_mean': float('nan'), 'n_hubs': 0}
    thresh = 2.0 * vals.mean()
    return {
        'mean': float(vals.mean()), 'max': float(vals.max()),
        'std': float(vals.std()),
        'max_over_mean': float(vals.max() / vals.mean()),
        'n_hubs': int(np.sum(vals > thresh)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Betweenness hubs of the gold audit, standalone")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--k', type=int, default=200,
                    help='Sampled source nodes (the audit uses 200; exact when n <= k)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--top', type=int, default=10, help='Top nodes to list (not in the audit)')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    G = layered.G
    n = G.number_of_nodes()
    k_used = min(args.k, n)
    print(f"  Nodes: {n:,}   Edges: {G.number_of_edges():,}   sources: {k_used} of {n} "
          f"({'exact' if k_used == n else 'sampled'}; whole graph, as in the audit)")
    bc = _betweenness_hubs(G, k=args.k, seed=args.seed)

    # The same call _betweenness_hubs makes, to list the top nodes.
    per_node = nx.betweenness_centrality(G, k=min(args.k, n), seed=args.seed)
    top = sorted(per_node.items(), key=lambda kv: (-kv[1], kv[0]))[:args.top]

    print("\n" + "=" * 72)
    print("BETWEENNESS HUBS")
    print("=" * 72)
    for key in ['mean', 'max', 'std', 'max_over_mean', 'n_hubs']:
        v = bc[key]
        print(f"  {key:>16s}: {v:.6g}" if isinstance(v, float) else f"  {key:>16s}: {v}")
    print(f"  top {len(top)} nodes (node index: normalized betweenness):")
    for node, v in top:
        print(f"    {node:>8}: {v:.6f}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'graph': {'n_nodes': n, 'n_edges': G.number_of_edges()},
                       'k_sources': k_used, 'seed': args.seed, 'betweenness': bc,
                       'top_nodes': [[int(a), float(b)] for a, b in top],
                       'per_node': {int(a): float(b) for a, b in per_node.items()}}, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
