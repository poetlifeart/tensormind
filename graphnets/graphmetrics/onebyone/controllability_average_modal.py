#!/usr/bin/env python3
"""
controllability_average_modal.py — average and modal controllability of
run_controllability.py, on their own.

Linear discrete-time network dynamics x(t+1) = A_n x(t) + e_i u(t) with input
at one node i (Gu et al. 2015, Nat. Commun. 6:8414):

    A        binary symmetric adjacency matrix of the graph (nx.to_numpy_array)
    c        1 + max |lambda(A)|   (= 1 + largest singular value, A symmetric)
    A_n      A / c = V diag(lambda) V^T          (all |lambda| < 1: stable)
    average controllability   AC_i  = sum_j V_ij^2 / (1 - lambda_j^2 + 1e-10)
                              = trace of the infinite-horizon controllability
                                Gramian sum_t A_n^t e_i e_i^T A_n^t
                              = [(I - A_n^2)^-1]_ii
    modal controllability     phi_i = sum_j V_ij^2 (1 - lambda_j^2)
                              = 1 - (A_n^2)_ii  = 1 - k_i / c^2 for a binary graph

Reported (all rounded to 4 decimals by analyze): mean, std (ddof 0) and CV of
AC and of phi over all nodes, and the Pearson correlation of each with degree.
Because phi_i = 1 - k_i / c^2 exactly on a binary graph, corr(degree, phi) is
-1 by construction.

Self-contained: needs numpy and networkx only, and does not import
run_controllability.py.  The loader, controllability() and analyze() below are
copied from it unchanged, so with the same --graph the result is the same as
run_controllability.py's JSON.  Only the driver (main) is new.  Isolated nodes
are kept (load_graph_npz adds all n_total nodes); a directed or multi-edge
.gml would be passed to the eigensolver as is, as in the original.

Usage:
    python controllability_average_modal.py --graph G.npz
    python controllability_average_modal.py --graph G.npz --json controllability.json
"""

import argparse
import json
import time
import sys
import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
# Loader, metric and summary  (copied from run_controllability.py, lines 26-105,
# unchanged)
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# Driver (new): loads the graph as run_controllability.py's main does
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description='Average and modal controllability (Gu et al. 2015), '
                                             'as run_controllability.py computes them')
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--json', default='', help='Save the results to JSON')
    args = ap.parse_args()

    if args.graph.endswith('.gml'):
        G = nx.read_gml(args.graph)
        G = nx.convert_node_labels_to_integers(G)
    else:
        G = load_graph_npz(args.graph)

    res = analyze(args.graph, G)

    print("\n" + "=" * 72)
    print("AVERAGE AND MODAL CONTROLLABILITY  (A / (1 + max|lambda|))")
    print("=" * 72)
    print(f"  mean average controllability   {res['avg_ctrl_mean']}   (std {res['avg_ctrl_std']}, CV {res['avg_ctrl_cv']})")
    print(f"  mean modal controllability     {res['modal_ctrl_mean']}   (std {res['modal_ctrl_std']}, CV {res['modal_ctrl_cv']})")
    print(f"  r(degree, average)             {res['corr_degree_avg_ctrl']}")
    print(f"  r(degree, modal)               {res['corr_degree_modal_ctrl']}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
