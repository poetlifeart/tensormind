#!/usr/bin/env python3
"""
combinatorial_hoffman_bound.py — the 'HoffmanBound' measure of
combinatorial_metrics_npz.py, on its own.

    HoffmanBound = 1 + lambda_max(A) / |lambda_min(A)|
    lambda_max, lambda_min = largest / smallest adjacency eigenvalue
                 (scipy.sparse.linalg.eigsh, k=1, which='LA' / which='SA')

Hoffman (1970) lower bound on the chromatic number, chi(G) >= 1 + lambda_max /
|lambda_min|; HoffmanBound of Szalkai, Varga & Grolmusz (arXiv:1603.00904;
arXiv:1709.04974).  It is 2 for every bipartite graph.

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph and hoffman_bound below are copied
from it unchanged, so with the same --graph the result is the 'hoffman_bound'
entry of its JSON.  The last one or two digits can differ: ARPACK's start
vector depends on how many ARPACK calls came before in the same process, and
hoffman_bound makes the third and fourth such calls in
combinatorial_metrics_npz.py but the first and second here.

Usage:
    python combinatorial_hoffman_bound.py --graph G.npz
    python combinatorial_hoffman_bound.py --graph G.npz --json hoffman.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
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


def hoffman_bound(G: nx.Graph) -> dict:
    """Hoffman chromatic number lower bound: 1 + lambda_max / |lambda_min|."""
    t0 = time.time()
    A = nx.adjacency_matrix(G, dtype=float)
    lam_max = scipy.sparse.linalg.eigsh(A, k=1, which='LA',
                                         return_eigenvectors=False)[0]
    lam_min = scipy.sparse.linalg.eigsh(A, k=1, which='SA',
                                         return_eigenvectors=False)[0]
    bound = 1 + lam_max / abs(lam_min) if abs(lam_min) > 1e-10 else float('inf')
    return {
        'hoffman_bound': float(bound),
        'lambda_max': float(lam_max),
        'lambda_min': float(lam_min),
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

    print("3. HoffmanBound...")
    results['hoffman_bound'] = hoffman_bound(G)
    r = results['hoffman_bound']
    print(f"   1 + lam_max/|lam_min| = {r['hoffman_bound']:.4f}")
    print(f"   lam_max={r['lambda_max']:.4f}, lam_min={r['lambda_min']:.4f} [{r['time']:.1f}s]")
    print(f"   full precision: hoffman_bound={r['hoffman_bound']!r}  lambda_max={r['lambda_max']!r}  "
          f"lambda_min={r['lambda_min']!r}")
    return results


def main():
    ap = argparse.ArgumentParser(description="HoffmanBound (1 + lambda_max/|lambda_min|) of combinatorial_metrics_npz.py")
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
