#!/usr/bin/env python3
"""
Network controllability analysis (Gu et al. 2015, Nature Communications 6:8414).

Computes average and modal controllability for each node, then reports
degree--controllability correlations.  The brain-like pattern is:
  - high-degree hubs  → high average controllability  (r ≈ +0.9)
  - low-degree nodes  → high modal controllability    (r ≈ -1.0)

Usage:
  python run_controllability.py --graph ../graph_brain_mild.npz
  python run_controllability.py --graph path/to/graph.npz --json results.json
  python run_controllability.py --gml path/to/graph.gml
"""

import argparse
import json
import time
import sys
import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


def load_graph_npz(path):
    data = np.load(path)
    if 'n_total' in data:
        n = int(data['n_total'])
    else:
        n = int(data['n1']) + int(data['n2']) + int(data['n3'])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if 'edges' in data:
        G.add_edges_from(data['edges'].tolist())
    else:
        n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        off2, off3 = n1, n1 + n2
        for key, o1, o2 in [('mask_12', 0, off2), ('mask_13', 0, off3), ('mask_23', off2, off3)]:
            rs, cs = np.where(data[key] > 0)
            for r, c in zip(rs, cs):
                G.add_edge(o1 + c, o2 + r)
    return G


def controllability(A):
    """Compute average and modal controllability for each node.

    Following Gu et al. 2015:
      - Normalize A by (1 + |lambda_max|) so all eigenvalues lie in (-1, 1).
      - Average controllability of node i = sum_j V[i,j]^2 / (1 - lam[j]^2)
        (trace of the controllability Gramian for a single input at node i).
      - Modal controllability of node i = sum_j V[i,j]^2 * (1 - lam[j]^2)
        (weights hard-to-excite eigenmodes).
    """
    evals_raw = np.linalg.eigvalsh(A)
    A_norm = A / (1.0 + np.max(np.abs(evals_raw)))
    lam, V = np.linalg.eigh(A_norm)

    avg_ctrl = np.sum(V**2 / (1.0 - lam**2 + 1e-10), axis=1)
    modal_ctrl = np.sum(V**2 * (1.0 - lam**2), axis=1)

    return avg_ctrl, modal_ctrl


def analyze(name, G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2.0 * m / (n * (n - 1))
    print(f'\n{"="*60}')
    print(f'{name}: n={n:,} m={m:,} density={density:.4f}')
    print(f'{"="*60}')

    A = nx.to_numpy_array(G, dtype=np.float64)
    degrees = np.array([G.degree(i) for i in range(n)])

    t0 = time.time()
    avg_ctrl, modal_ctrl = controllability(A)
    dt = time.time() - t0
    print(f'  Computed in {dt:.1f}s')

    corr_avg = np.corrcoef(degrees, avg_ctrl)[0, 1]
    corr_modal = np.corrcoef(degrees, modal_ctrl)[0, 1]

    results = {
        'name': name,
        'n': n,
        'm': m,
        'density': round(density, 4),
        'avg_ctrl_mean': round(float(np.mean(avg_ctrl)), 4),
        'avg_ctrl_std': round(float(np.std(avg_ctrl)), 4),
        'avg_ctrl_cv': round(float(np.std(avg_ctrl) / np.mean(avg_ctrl)), 4),
        'modal_ctrl_mean': round(float(np.mean(modal_ctrl)), 4),
        'modal_ctrl_std': round(float(np.std(modal_ctrl)), 4),
        'modal_ctrl_cv': round(float(np.std(modal_ctrl) / np.mean(modal_ctrl)), 4),
        'corr_degree_avg_ctrl': round(float(corr_avg), 4),
        'corr_degree_modal_ctrl': round(float(corr_modal), 4),
    }

    print(f'  Avg ctrl:   mean={results["avg_ctrl_mean"]}, CV={results["avg_ctrl_cv"]}')
    print(f'  Modal ctrl: mean={results["modal_ctrl_mean"]}, CV={results["modal_ctrl_cv"]}')
    print(f'  Corr(degree, avg_ctrl):   {results["corr_degree_avg_ctrl"]}')
    print(f'  Corr(degree, modal_ctrl): {results["corr_degree_modal_ctrl"]}')

    return results


def main():
    parser = argparse.ArgumentParser(description='Network controllability (Gu et al. 2015)')
    parser.add_argument('--graph', type=str, required=True,
                        help='Path to the graph (.npz or .gml)')
    parser.add_argument('--gml', type=str, default=None,
                        help=argparse.SUPPRESS)   # legacy alias; --graph now accepts .gml
    parser.add_argument('--json', type=str, default=None, help='Save results to JSON')
    args = parser.parse_args()

    src = args.graph or args.gml
    if src and src.endswith('.gml'):
        G = nx.read_gml(src)
        G = nx.convert_node_labels_to_integers(G)
        name = src
    elif args.graph:
        G = load_graph_npz(args.graph)
        name = args.graph
    elif args.gml:
        G = nx.read_gml(args.gml)
        G = nx.convert_node_labels_to_integers(G)
        name = args.gml
    else:
        parser.error('Provide --graph or --gml')

    results = analyze(name, G)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved to {args.json}')


if __name__ == '__main__':
    main()
