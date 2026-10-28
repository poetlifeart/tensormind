#!/usr/bin/env python3
"""
combinatorial_rw_eigengap.py — the 'PGEigengap' measure of
combinatorial_metrics_npz.py (random-walk eigengap), on its own.

    P        = D^{-1} A                 (random-walk transition matrix)
    eigengap = lambda_1(P) - lambda_2(P)  (the two largest real parts among the
               min(6, n-2) eigenvalues of largest real part found by
               scipy.sparse.linalg.eigs, which='LR')

PGEigengap of Szalkai, Varga & Grolmusz (arXiv:1603.00904; arXiv:1709.04974).
For a connected graph lambda_1(P) = 1, and P is similar to
D^{-1/2} A D^{-1/2}, so the eigengap equals the second-smallest eigenvalue of
the normalized Laplacian I - D^{-1/2} A D^{-1/2} (Chung 1997).  It is 0 for a
disconnected graph.

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph and pg_eigengap below are copied
from it unchanged, so with the same --graph the result is the 'pg_eigengap'
entry of its JSON.  The last one or two digits can differ: ARPACK's start
vector depends on how many ARPACK calls came before in the same process, and
pg_eigengap is the second such call in combinatorial_metrics_npz.py but the
first here.

Usage:
    python combinatorial_rw_eigengap.py --graph G.npz
    python combinatorial_rw_eigengap.py --graph G.npz --json eigengap.json
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


def pg_eigengap(G: nx.Graph) -> dict:
    """Eigengap of transition matrix P = D^{-1}A.
    PGEigengap = lambda_1(P) - lambda_2(P).
    lambda_1(P) = 1 always for connected graphs."""
    t0 = time.time()
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, dtype=float)
    D_inv = scipy.sparse.diags([1.0 / G.degree(i) if G.degree(i) > 0 else 0
                                 for i in range(n)])
    P = D_inv @ A
    eigs_P = scipy.sparse.linalg.eigs(P, k=min(6, n - 2), which='LR',
                                       return_eigenvectors=False)
    eigs_P = np.sort(np.real(eigs_P))[::-1]
    gap = float(eigs_P[0] - eigs_P[1]) if len(eigs_P) > 1 else 0
    return {
        'top_eigenvalues': eigs_P[:5].tolist(),
        'eigengap': gap,
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

    print("2. PGEigengap (transition matrix D^-1 A)...")
    results['pg_eigengap'] = pg_eigengap(G)
    r = results['pg_eigengap']
    print(f"   top eigs(P): {[f'{v:.4f}' for v in r['top_eigenvalues'][:4]]}")
    print(f"   PGEigengap={r['eigengap']:.6f} [{r['time']:.1f}s]")
    print(f"   full precision: eigengap={r['eigengap']!r}  top={r['top_eigenvalues']!r}")
    return results


def main():
    ap = argparse.ArgumentParser(description="PGEigengap (random-walk eigengap) of combinatorial_metrics_npz.py")
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
