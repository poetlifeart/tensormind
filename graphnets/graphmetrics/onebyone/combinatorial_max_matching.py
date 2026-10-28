#!/usr/bin/env python3
"""
combinatorial_max_matching.py — the 'MaxMatching' and 'MaxFracMatching'
measures of combinatorial_metrics_npz.py, on their own.

  max_matching       maximum-cardinality matching nu(G):
                     networkx.max_weight_matching(G, maxcardinality=True)
                     (Edmonds' blossom algorithm; unweighted graphs, so every
                     edge has weight 1).  'size' = nu(G), 'normalized' = 2 nu / n
                     (fraction of nodes matched).
  max_frac_matching  LP relaxation (scipy linprog, HiGHS):
                         max sum_e x_e  s.t.  sum_{e ~ v} x_e <= 1 for every node v,
                                              0 <= x_e <= 1
                     'size' = fractional matching number nu_f(G),
                     'normalized' = 2 nu_f / n.

MaxMatching / MaxFracMatching of Szalkai, Varga & Grolmusz (arXiv:1603.00904;
arXiv:1709.04974).  nu <= nu_f <= tau (minimum vertex cover); nu_f equals the
LP relaxation of vertex cover (LP duality), and for bipartite graphs
nu = nu_f = tau (Konig-Egervary; the incidence matrix is totally unimodular).

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph, max_matching and max_frac_matching
below are copied from it unchanged, so with the same --graph the result is the
'max_matching' and 'max_frac_matching' entries of its JSON.

Usage:
    python combinatorial_max_matching.py --graph G.npz
    python combinatorial_max_matching.py --graph G.npz --json matching.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse
from scipy.optimize import linprog
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
# The measures  (copied from combinatorial_metrics_npz.py)
# ═══════════════════════════════════════════════════════════════════════════


def max_matching(G: nx.Graph) -> dict:
    """Maximum cardinality matching."""
    t0 = time.time()
    matching = nx.max_weight_matching(G, maxcardinality=True)
    n = G.number_of_nodes()
    return {
        'size': len(matching),
        'normalized': 2 * len(matching) / n,
        'time': time.time() - t0
    }


def max_frac_matching(G: nx.Graph) -> dict:
    """Fractional maximum matching via LP.
    max sum(x_e) s.t. sum(x_e : e incident to v) <= 1, 0 <= x_e <= 1."""
    t0 = time.time()
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    c = -np.ones(m)
    row_idx = np.array([u for u, v in edges] + [v for u, v in edges])
    col_idx = np.concatenate([np.arange(m), np.arange(m)])
    data = np.ones(2 * m)
    A_ub = scipy.sparse.csc_matrix((data, (row_idx, col_idx)), shape=(n, m))
    b_ub = np.ones(n)
    bounds = [(0, 1)] * m
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    val = -res.fun if res.success else -1
    return {
        'size': float(val),
        'normalized': 2 * float(val) / n if val >= 0 else -1,
        'time': time.time() - t0
    }


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════


def run_metric(G: nx.Graph, label: str) -> dict:
    """As run_all in combinatorial_metrics_npz.py, for these two measures only."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"\n{'='*65}")
    print(f"{label}: {n} nodes, {m:,} edges, density {2*m/(n*(n-1)):.4f}")
    print(f"{'='*65}")

    results = {'n': n, 'm': m, 'density': 2 * m / (n * (n - 1))}

    print("8. MaxMatching...")
    results['max_matching'] = max_matching(G)
    r = results['max_matching']
    print(f"   size={r['size']} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")

    print("9. MaxFracMatching (LP)...")
    results['max_frac_matching'] = max_frac_matching(G)
    r = results['max_frac_matching']
    print(f"   size={r['size']:.2f} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")
    print(f"   full precision: matching size={results['max_matching']['size']!r}  "
          f"fractional size={results['max_frac_matching']['size']!r}")
    return results


def main():
    ap = argparse.ArgumentParser(description="MaxMatching and MaxFracMatching of combinatorial_metrics_npz.py")
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
