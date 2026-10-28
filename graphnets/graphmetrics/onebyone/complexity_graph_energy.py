#!/usr/bin/env python3
"""
complexity_graph_energy.py — measure 3 of run_complexity.py (graph energy),
on its own.

Graph energy of the whole loaded graph (every node of the file, not only its
giant component), from the full spectrum of the dense adjacency matrix
(np.linalg.eigvalsh, as in run_complexity.py's shared eigendecomposition):

    E(G)  = sum_i |lambda_i|          graph_energy
    E / n                             graph_energy_per_node

Definition: Gutman (1978), "The energy of a graph", Ber. Math.-Statist. Sekt.
Forschungsz. Graz 103:1-22 (the total pi-electron energy of Hueckel theory).

Self-contained: needs numpy and networkx only, and does not import
run_complexity.py.  load_graph_npz() and compute_graph_energy() below are
copied from it unchanged; load_graph() and adjacency_eigenvalues() are the
graph-loading branch and the adjacency part of the shared eigendecomposition
of its main(), as functions (if that dense eigendecomposition fails, main()
passes None and compute_graph_energy falls back to nx.adjacency_spectrum —
kept here).  With the same --graph the numbers are the same as the
'graph_energy' block of run_complexity.py's JSON.  run_complexity.py's
--skip-orbits / --skip-eigen only switch measures off, so they are not
carried over.

Usage:
    python complexity_graph_energy.py --graph G.npz
    python complexity_graph_energy.py --graph G.npz --json energy.json
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
# 3. Graph energy (Gutman 1978) — nx.adjacency_spectrum()
#    (copied from run_complexity.py)
# ═══════════════════════════════════════════════════════════════════

def compute_graph_energy(G, eigenvalues=None):
    """Graph energy = sum(|lambda_i|) for adjacency eigenvalues."""
    if eigenvalues is None:
        eigenvalues = np.real(nx.adjacency_spectrum(G))
    energy = np.sum(np.abs(eigenvalues))
    return {
        'graph_energy': float(energy),
        'graph_energy_per_node': float(energy / G.number_of_nodes()),
    }


# ═══════════════════════════════════════════════════════════════════
# Main  (the measure-3 part of run_complexity.py main())
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Graph energy (measure 3 of run_complexity.py)")
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
    print("3. GRAPH ENERGY (Gutman 1978)")
    print(f"{'─'*70}")
    t0 = time.time()
    try:
        ener = compute_graph_energy(G, eigenvalues=adj_eigenvalues)
        results['graph_energy'] = ener
        print(f"  Graph energy:        {ener['graph_energy']:.4f}")
        print(f"  Energy / n:          {ener['graph_energy_per_node']:.4f}")
        print(f"  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['graph_energy'] = {'error': str(e)}

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
