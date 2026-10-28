#!/usr/bin/env python3
"""
combinatorial_spanning_tree_entropy.py — the 'LogAbsSpanningForestN' measure
of combinatorial_metrics_npz.py (spanning-tree entropy), on its own.

    log_spanning_tree_count  = sum_i log(mu_i) - log(n)   over Laplacian
                               eigenvalues mu_i > 1e-10
    log_spanning_tree_per_n  = log_spanning_tree_count / n      (natural log)

Kirchhoff's matrix-tree theorem: a connected graph has
tau = (1/n) * prod_{i>=2} mu_i spanning trees (mu_i = Laplacian eigenvalues);
log(tau)/n is the spanning-tree entropy (Lyons 2005).  LogAbsSpanningForestN
of Szalkai, Varga & Grolmusz (arXiv:1603.00904; arXiv:1709.04974).

As in the source: for n <= 2000 all eigenvalues are computed (dense
numpy.linalg.eigvalsh, note 'exact'); for n > 2000 only the min(n-1, 500)
smallest are computed (eigsh, which='SM') and summed (note 'partial') — that
is not the log spanning-tree count.  For a disconnected graph the formula
gives log(prod_c n_c * tau_c / n), not the log of the number of spanning
forests (prod_c tau_c).

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph and log_spanning_forest below are
copied from it unchanged, so with the same --graph the result is the
'log_spanning_forest' entry of its JSON.

Usage:
    python combinatorial_spanning_tree_entropy.py --graph G.npz
    python combinatorial_spanning_tree_entropy.py --graph G.npz --json st.json
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


def log_spanning_forest(G: nx.Graph) -> dict:
    """Log of spanning tree count / n, via Kirchhoff's theorem."""
    t0 = time.time()
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).astype(float)
    if n <= 2000:
        L_dense = L.toarray() if scipy.sparse.issparse(L) else L
        eigs = np.linalg.eigvalsh(L_dense)
    else:
        eigs = scipy.sparse.linalg.eigsh(L, k=min(n - 1, 500),
                                          which='SM',
                                          return_eigenvectors=False)
    eigs = np.sort(eigs)
    nonzero = eigs[eigs > 1e-10]
    if len(nonzero) > 0:
        log_count = np.sum(np.log(nonzero)) - np.log(n)
        log_count_per_n = log_count / n
    else:
        log_count = 0
        log_count_per_n = 0
    return {
        'log_spanning_tree_count': float(log_count),
        'log_spanning_tree_per_n': float(log_count_per_n),
        'n_nonzero_eigs': len(nonzero),
        'note': 'partial' if n > 2000 else 'exact',
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

    print("4. LogAbsSpanningForestN (Kirchhoff)...")
    results['log_spanning_forest'] = log_spanning_forest(G)
    r = results['log_spanning_forest']
    print(f"   log(count)/n = {r['log_spanning_tree_per_n']:.4f} ({r['note']}) [{r['time']:.1f}s]")
    print(f"   full precision: log_spanning_tree_count={r['log_spanning_tree_count']!r}  "
          f"log_spanning_tree_per_n={r['log_spanning_tree_per_n']!r}  "
          f"n_nonzero_eigs={r['n_nonzero_eigs']}  count=exp(log)={np.exp(r['log_spanning_tree_count']):.10g}")
    return results


def main():
    ap = argparse.ArgumentParser(description="LogAbsSpanningForestN (spanning-tree entropy) of combinatorial_metrics_npz.py")
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
