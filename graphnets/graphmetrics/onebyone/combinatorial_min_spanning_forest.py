#!/usr/bin/env python3
"""
combinatorial_min_spanning_forest.py — the 'MinSpanningForest' measure of
combinatorial_metrics_npz.py, on its own.

    T      = networkx.minimum_spanning_tree(G)   (Kruskal; a spanning forest,
             one tree per connected component)
    weight = number of edges of T = n - c        (c = number of components)

MinSpanningForest of Szalkai, Varga & Grolmusz (arXiv:1603.00904;
arXiv:1709.04974) is the weight of a minimum spanning forest.  The source
reports T.number_of_edges(), not the sum of edge weights, so it equals the MSF
weight only for unweighted graphs (unit edge weights, which is every .npz the
loader reads); for a weighted .gml/.graphml it would still return n - c.

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph and min_spanning_forest below are
copied from it unchanged, so with the same --graph the result is the
'min_spanning_forest' entry of its JSON.

Usage:
    python combinatorial_min_spanning_forest.py --graph G.npz
    python combinatorial_min_spanning_forest.py --graph G.npz --json msf.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════
# Loading  (copied from combinatorial_metrics_npz.py)
# ═══════════════════════════════════════════════════════════════════════════


def load_graph(path: str) -> nx.Graph:
    if path.endswith('.gml') or path.endswith('.graphml'):
        G = nx.read_graphml(path) if path.endswith('.graphml') else nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
    elif path.endswith('.npz'):
        d = np.load(path)
        if 'n_total' in d:
            n = int(d['n_total'])
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from((int(u), int(v)) for u, v in d['edges'])
        else:
            n1, n2, n3 = int(d['n1']), int(d['n2']), int(d['n3'])
            n = n1 + n2 + n3
            if 'edges' in d:
                G = nx.Graph()
                G.add_nodes_from(range(n))
                G.add_edges_from((int(u), int(v)) for u, v in d['edges'])
            else:
                G = nx.Graph()
                G.add_nodes_from(range(n))
                # mask_12 is written as (n2, n1): row = layer-2 index, col =
                # layer-1 index (see save_graph in graph_sparsify_memb.py).
                # It is square (n1 == n2), so a transposed read does not raise --
                # it silently builds a different graph.
                rs, cs = np.where(d['mask_12'])
                G.add_edges_from(zip(cs.astype(int), (rs + n1).astype(int)))
                rs, cs = np.where(d['mask_13'])
                G.add_edges_from(zip((rs + n1 + n2).astype(int), cs.astype(int)))
                rs, cs = np.where(d['mask_23'])
                G.add_edges_from(zip((rs + n1 + n2).astype(int), (cs + n1).astype(int)))
    else:
        raise ValueError(f"Unknown format: {path}")
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# The measure  (copied from combinatorial_metrics_npz.py)
# ═══════════════════════════════════════════════════════════════════════════


def min_spanning_forest(G: nx.Graph) -> dict:
    """Minimum spanning tree weight. For unweighted graphs = n - components."""
    t0 = time.time()
    T = nx.minimum_spanning_tree(G)
    weight = T.number_of_edges()
    return {
        'weight': weight,
        'n_components': nx.number_connected_components(G),
        'time': time.time() - t0
    }


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════


def run_metric(G: nx.Graph, label: str) -> dict:
    """As run_all in combinatorial_metrics_npz.py, for this measure only."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"\n{'='*65}")
    print(f"{label}: {n} nodes, {m:,} edges, density {2*m/(n*(n-1)):.4f}")
    print(f"{'='*65}")

    results = {'n': n, 'm': m, 'density': 2 * m / (n * (n - 1))}

    print("10. MinSpanningForest (Kruskal)...")
    results['min_spanning_forest'] = min_spanning_forest(G)
    r = results['min_spanning_forest']
    print(f"    weight={r['weight']} [{r['time']:.1f}s]")
    print(f"    n_components={r['n_components']}  (n - components = {n - r['n_components']})")
    return results


def main():
    ap = argparse.ArgumentParser(description="MinSpanningForest of combinatorial_metrics_npz.py")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz, .gml or .graphml)')
    ap.add_argument('--json', default='', help='Output JSON (optional)')
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.graph))[0]
    G = load_graph(args.graph)
    res = run_metric(G, name)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
