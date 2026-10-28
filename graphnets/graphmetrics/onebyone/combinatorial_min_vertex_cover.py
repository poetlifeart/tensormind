#!/usr/bin/env python3
"""
combinatorial_min_vertex_cover.py — the two 'MinVertexCover' measures of
combinatorial_metrics_npz.py, on their own.

  min_vc_frac    LP relaxation of minimum vertex cover (scipy linprog, HiGHS):
                     min sum_i x_i   s.t.  x_u + x_v >= 1 for every edge,
                                           0 <= x_i <= 1
                 'size' = LP optimum (a LOWER bound on the minimum vertex
                 cover tau; equal to it for bipartite graphs), 'normalized' =
                 size / n.
  min_vc_approx  networkx.approximation.min_weighted_vertex_cover (local-ratio
                 algorithm of Bar-Yehuda & Even 1985, unit weights): a vertex
                 cover of size <= 2 tau (an UPPER bound), 'normalized' = size / n.

MinVertexCover of Szalkai, Varga & Grolmusz (arXiv:1603.00904;
arXiv:1709.04974) is the exact minimum; these are the two bounds on it.
By LP duality min_vc_frac equals max_frac_matching (combinatorial_max_matching.py).

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph, min_vertex_cover_frac and
min_vertex_cover_approx below are copied from it unchanged, so with the same
--graph the result is the 'min_vc_frac' and 'min_vc_approx' entries of its JSON.

Usage:
    python combinatorial_min_vertex_cover.py --graph G.npz
    python combinatorial_min_vertex_cover.py --graph G.npz --json vc.json
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


def min_vertex_cover_frac(G: nx.Graph) -> dict:
    """Fractional minimum vertex cover via LP relaxation.
    min sum(x_i) s.t. x_u + x_v >= 1 for each edge, 0 <= x_i <= 1."""
    t0 = time.time()
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    c = np.ones(n)
    row_idx = np.concatenate([np.arange(m), np.arange(m)])
    col_idx = np.array([u for u, v in edges] + [v for u, v in edges])
    data = -np.ones(2 * m)
    A_ub = scipy.sparse.csc_matrix((data, (row_idx, col_idx)), shape=(m, n))
    b_ub = -np.ones(m)
    bounds = [(0, 1)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return {
        'size': float(res.fun) if res.success else -1,
        'normalized': float(res.fun) / n if res.success else -1,
        'time': time.time() - t0
    }


def min_vertex_cover_approx(G: nx.Graph) -> dict:
    """2-approximation minimum vertex cover (networkx greedy matching)."""
    t0 = time.time()
    vc = nx.approximation.min_weighted_vertex_cover(G)
    n = G.number_of_nodes()
    return {
        'size': len(vc),
        'normalized': len(vc) / n,
        'note': '2-approx upper bound',
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

    print("5. MinVertexCover (fractional LP)...")
    results['min_vc_frac'] = min_vertex_cover_frac(G)
    r = results['min_vc_frac']
    print(f"   size={r['size']:.2f} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")

    print("6. MinVertexCover (2-approx)...")
    results['min_vc_approx'] = min_vertex_cover_approx(G)
    r = results['min_vc_approx']
    print(f"   size={r['size']} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")
    print(f"   full precision: frac size={results['min_vc_frac']['size']!r}  "
          f"approx size={results['min_vc_approx']['size']!r}")
    return results


def main():
    ap = argparse.ArgumentParser(description="MinVertexCover (LP relaxation and 2-approximation) of combinatorial_metrics_npz.py")
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
