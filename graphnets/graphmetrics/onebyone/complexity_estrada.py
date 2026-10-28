#!/usr/bin/env python3
"""
complexity_estrada.py — measure 2 of run_complexity.py (Estrada index), on
its own.

Estrada index of the whole loaded graph (every node of the file, not only its
giant component), from the full spectrum of the dense adjacency matrix
(np.linalg.eigvalsh, as in run_complexity.py's shared eigendecomposition):

    EE(G) = sum_i exp(lambda_i)  = trace(exp(A))    estrada_index
    log10 EE                                        estrada_index_log10
    EE / n   (= mean subgraph centrality)           estrada_index_per_node

Definition: Estrada (2000), "Characterization of 3D molecular structure",
Chem. Phys. Lett. 319:713-718; named the Estrada index by de la Pena, Gutman &
Rada (2007), Linear Algebra Appl. 427:70-76.  EE/n is the mean subgraph
centrality of Estrada & Rodriguez-Velazquez (2005), Phys. Rev. E 71:056103.

Self-contained: needs numpy and networkx only, and does not import
run_complexity.py.  load_graph_npz() and compute_estrada() below are copied
from it unchanged; load_graph() and adjacency_eigenvalues() are the
graph-loading branch and the adjacency part of the shared eigendecomposition
of its main(), as functions (if that dense eigendecomposition fails, main()
passes None and compute_estrada falls back to nx.estrada_index — kept here).
With the same --graph the numbers are the same as the 'estrada' block of
run_complexity.py's JSON.  run_complexity.py's --skip-orbits / --skip-eigen
only switch measures off, so they are not carried over.

Usage:
    python complexity_estrada.py --graph G.npz
    python complexity_estrada.py --graph G.npz --json estrada.json
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
# Shared eigendecomposition — adjacency part
# (the code of run_complexity.py main(), as a function)
# ═══════════════════════════════════════════════════════════════════

def adjacency_eigenvalues(G):
    """Dense adjacency matrix, all n eigenvalues by np.linalg.eigvalsh, exactly
    as run_complexity.py main() computes them; None if that fails."""
    adj_eigenvalues = None
    print("  Computing adjacency eigenvalues (dense, all n)...")
    t0 = time.time()
    try:
        adj_dense = nx.to_numpy_array(G, dtype=np.float64)
        adj_eigenvalues = np.linalg.eigvalsh(adj_dense)
        print(f"  Adjacency eigenvalues: done ({time.time()-t0:.1f}s)")
        print(f"    range: [{adj_eigenvalues.min():.4f}, {adj_eigenvalues.max():.4f}]")
    except Exception as e:
        print(f"  FAILED: {e}")
    return adj_eigenvalues


# ═══════════════════════════════════════════════════════════════════
# 2. Estrada index — NetworkX built-in
#    (copied from run_complexity.py)
# ═══════════════════════════════════════════════════════════════════

def compute_estrada(G, eigenvalues=None):
    """Estrada index = sum(exp(lambda_i)) for adjacency eigenvalues.
    Uses pre-computed eigenvalues if available, otherwise nx built-in."""
    if eigenvalues is not None:
        EE = np.sum(np.exp(eigenvalues))
    else:
        EE = nx.estrada_index(G)
    return {
        'estrada_index': float(EE),
        'estrada_index_log10': float(np.log10(EE)) if EE > 0 else float('-inf'),
        'estrada_index_per_node': float(EE / G.number_of_nodes()),
    }


# ═══════════════════════════════════════════════════════════════════
# Main  (the measure-2 part of run_complexity.py main())
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Estrada index (measure 2 of run_complexity.py)")
    ap.add_argument('--graph', type=str, required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--json', type=str, default=None, help='Save results to JSON')
    args = ap.parse_args()

    print(f"\nLoading {args.graph}...")
    t0 = time.time()
    G, n1, n2, n3 = load_graph(args.graph)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  {n:,} nodes, {m:,} edges ({time.time()-t0:.1f}s)")

    results = {'graph': args.graph, 'nodes': n, 'edges': m}

    print(f"\n{'─'*70}")
    print("EIGENDECOMPOSITION (adjacency)")
    print(f"{'─'*70}")
    adj_eigenvalues = adjacency_eigenvalues(G)

    print(f"\n{'─'*70}")
    print("2. ESTRADA INDEX (Estrada 2000)")
    print(f"{'─'*70}")
    t0 = time.time()
    try:
        est = compute_estrada(G, eigenvalues=adj_eigenvalues)
        results['estrada'] = est
        print(f"  Estrada index:       {est['estrada_index']:.6e}")
        print(f"  log10(EE):           {est['estrada_index_log10']:.4f}")
        print(f"  EE / n:              {est['estrada_index_per_node']:.6e}")
        print(f"  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['estrada'] = {'error': str(e)}

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
