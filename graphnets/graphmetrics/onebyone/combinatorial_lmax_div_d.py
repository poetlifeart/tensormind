#!/usr/bin/env python3
"""
combinatorial_lmax_div_d.py — the 'AdjLMaxDivD' measure of
combinatorial_metrics_npz.py (lambda_max / <k>), on its own.

    value      = lambda_max(A) / <k>
    lambda_max = largest eigenvalue of the adjacency matrix A
                 (scipy.sparse.linalg.eigsh, k=1, which='LA')
    <k>        = 2m / n   (mean degree, every node counted)

AdjLMaxDivD of Szalkai, Varga & Grolmusz (arXiv:1603.00904; arXiv:1709.04974).
It is 1 for a regular graph and grows with degree heterogeneity.  For a
connected graph lambda_max is the spectral norm ||A||_2 (Perron-Frobenius).

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph and adj_lmax_div_d below are copied
from it unchanged, so with the same --graph the result is the 'adj_lmax_div_d'
entry of its JSON.  (adj_lmax_div_d is the first ARPACK call in both scripts,
so even the last digit agrees; ARPACK's start vector depends on how many
ARPACK calls came before in the same process.)

Usage:
    python combinatorial_lmax_div_d.py --graph G.npz
    python combinatorial_lmax_div_d.py --graph G.npz --json lmax.json
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


def adj_lmax_div_d(G: nx.Graph) -> dict:
    """Largest adjacency eigenvalue / average degree."""
    t0 = time.time()
    A = nx.adjacency_matrix(G, dtype=float)
    n = G.number_of_nodes()
    deg_avg = 2 * G.number_of_edges() / n
    lam_max = scipy.sparse.linalg.eigsh(A, k=1, which='LA',
                                         return_eigenvectors=False)[0]
    return {
        'lambda_max': float(lam_max),
        'avg_degree': float(deg_avg),
        'value': float(lam_max) / deg_avg if deg_avg > 0 else 0,
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

    print("\n1. AdjLMaxDivD...")
    results['adj_lmax_div_d'] = adj_lmax_div_d(G)
    r = results['adj_lmax_div_d']
    print(f"   lambda_max={r['lambda_max']:.4f}, avg_deg={r['avg_degree']:.1f}, "
          f"AdjLMaxDivD={r['value']:.4f} [{r['time']:.1f}s]")
    print(f"   full precision: lambda_max={r['lambda_max']!r}  avg_degree={r['avg_degree']!r}  "
          f"value={r['value']!r}")
    return results


def main():
    ap = argparse.ArgumentParser(description="AdjLMaxDivD (lambda_max / <k>) of combinatorial_metrics_npz.py")
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
