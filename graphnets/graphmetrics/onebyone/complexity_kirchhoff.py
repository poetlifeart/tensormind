#!/usr/bin/env python3
"""
complexity_kirchhoff.py — measure 4 of run_complexity.py (Kirchhoff index),
on its own.

Kirchhoff index of the whole loaded graph (every node of the file, not only
its giant component), from the full spectrum of the dense Laplacian
L = D - A (np.linalg.eigvalsh, as in run_complexity.py's shared
eigendecomposition):

    Kf(G) = n * sum_{mu_i > 1e-10} 1/mu_i       kirchhoff_index
          (= sum over unordered vertex pairs of the resistance distance,
             for a connected graph)
    Kf / n^2                                    kirchhoff_index_per_n2

Definition: Klein & Randic (1993), "Resistance distance", J. Math. Chem.
12:81-95; the spectral formula is Gutman & Mohar (1996), J. Chem. Inf. Comput.
Sci. 36:982, and Zhu, Klein & Lukovits (1996), ibid. 36:420.  Same quantity as
the "effective graph resistance" R_G of Ellens et al. (2011), Linear Algebra
Appl. 435:2491.

Note: the formula drops every Laplacian eigenvalue <= 1e-10.  On a connected
graph that is the single zero eigenvalue and the result is Kf.  On a
disconnected graph Kf is infinite (some pair has no path), but this code drops
all c zero eigenvalues and returns a finite number.

Self-contained: needs numpy and networkx only, and does not import
run_complexity.py.  load_graph_npz() and compute_kirchhoff() below are copied
from it unchanged; load_graph() and laplacian_eigenvalues() are the
graph-loading branch and the Laplacian part of the shared eigendecomposition
of its main(), as functions (if that dense eigendecomposition fails, main()
passes None and compute_kirchhoff falls back to
nx.effective_graph_resistance — kept here).  With the same --graph the
numbers are the same as the 'kirchhoff' block of run_complexity.py's JSON.
run_complexity.py's --skip-orbits / --skip-eigen only switch measures off, so
they are not carried over.

Usage:
    python complexity_kirchhoff.py --graph G.npz
    python complexity_kirchhoff.py --graph G.npz --json kirchhoff.json
"""

import argparse
import json
import time
import sys
import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════
# Graph loading  (copied from run_complexity.py)
# ═══════════════════════════════════════════════════════════════════

def load_graph_npz(path):
    """Load graph from .npz — supports both mask-based and edge-list formats."""
    data = np.load(path)
    if 'n1' in data:
        n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        n = n1 + n2 + n3
    else:
        n = int(data['n_total'])
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    if 'edges' in data:
        G.add_edges_from(data['edges'].tolist())
    else:
        off2, off3 = n1, n1 + n2
        for key, o1, o2 in [('mask_12', 0, off2), ('mask_13', 0, off3), ('mask_23', off2, off3)]:
            rs, cs = np.where(data[key] > 0)
            for r, c in zip(rs, cs):
                G.add_edge(o1 + c, o2 + r)

    return G, n1, n2, n3


def load_graph(path):
    """The graph-loading branch of run_complexity.py main(), as a function
    (.gml read by networkx, anything else by load_graph_npz)."""
    if path.endswith('.gml'):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
        n = G.number_of_nodes()
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    else:
        G, n1, n2, n3 = load_graph_npz(path)
    return G, n1, n2, n3


# ═══════════════════════════════════════════════════════════════════
# Shared eigendecomposition — Laplacian part
# (the code of run_complexity.py main(), as a function)
# ═══════════════════════════════════════════════════════════════════

def laplacian_eigenvalues(G):
    """Dense Laplacian, all n eigenvalues by np.linalg.eigvalsh, exactly as
    run_complexity.py main() computes them; None if that fails."""
    lap_eigenvalues = None
    print("  Computing Laplacian eigenvalues (dense, all n)...")
    t0 = time.time()
    try:
        L = nx.laplacian_matrix(G).toarray().astype(np.float64)
        lap_eigenvalues = np.linalg.eigvalsh(L)
        print(f"  Laplacian eigenvalues: done ({time.time()-t0:.1f}s)")
        print(f"    range: [{lap_eigenvalues.min():.6f}, {lap_eigenvalues.max():.4f}]")
        del L  # free memory
    except Exception as e:
        print(f"  FAILED: {e}")
    return lap_eigenvalues


# ═══════════════════════════════════════════════════════════════════
# 4. Kirchhoff index — nx.effective_graph_resistance()
#    (copied from run_complexity.py)
# ═══════════════════════════════════════════════════════════════════

def compute_kirchhoff(G, lap_eigenvalues=None):
    """Kirchhoff index = n * sum(1/lambda_i) for nonzero Laplacian eigenvalues."""
    n = G.number_of_nodes()
    if lap_eigenvalues is None:
        R = nx.effective_graph_resistance(G)
    else:
        nonzero = lap_eigenvalues[lap_eigenvalues > 1e-10]
        R = n * np.sum(1.0 / nonzero)
    return {
        'kirchhoff_index': float(R),
        'kirchhoff_index_per_n2': float(R / (n * n)),
    }


# ═══════════════════════════════════════════════════════════════════
# Main  (the measure-4 part of run_complexity.py main())
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Kirchhoff index (measure 4 of run_complexity.py)")
    ap.add_argument('--graph', type=str, required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--json', type=str, default=None, help='Save results to JSON')
    args = ap.parse_args()

    print(f"\nLoading {args.graph}...")
    t0 = time.time()
    G, n1, n2, n3 = load_graph(args.graph)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  {n:,} nodes, {m:,} edges ({time.time()-t0:.1f}s)")
    n_comp = nx.number_connected_components(G) if n > 0 else 0
    if n_comp > 1:
        print(f"  WARNING: {n_comp} connected components — the Kirchhoff index is infinite; "
              f"the value below drops all {n_comp} zero eigenvalues")

    results = {'graph': args.graph, 'nodes': n, 'edges': m}

    print(f"\n{'─'*70}")
    print("EIGENDECOMPOSITION (Laplacian)")
    print(f"{'─'*70}")
    lap_eigenvalues = laplacian_eigenvalues(G)

    print(f"\n{'─'*70}")
    print("4. KIRCHHOFF INDEX (Klein & Randic 1993)")
    print(f"{'─'*70}")
    t0 = time.time()
    try:
        kirch = compute_kirchhoff(G, lap_eigenvalues=lap_eigenvalues)
        results['kirchhoff'] = kirch
        print(f"  Kirchhoff index:     {kirch['kirchhoff_index']:.4f}")
        print(f"  R / n^2:             {kirch['kirchhoff_index_per_n2']:.8f}")
        print(f"  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['kirchhoff'] = {'error': str(e)}

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
