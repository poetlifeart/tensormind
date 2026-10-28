#!/usr/bin/env python3
"""
complexity_offdiagonal.py — measure 5 of run_complexity.py (off-diagonal
complexity, degree-distribution entropy, unique degree values: one function,
compute_offdiagonal), on its own.

On the whole loaded graph (every node of the file, not only its giant
component), with l(i) the degree of node i:

  off-diagonal complexity (Claussen 2007, Physica A 375:365-373, Eqs. 2-5)
    c_mn = number of edges whose end degrees are m <= n  (each edge once)
    a~_k = sum_i c_{i,i+k}        (k-th diagonal of c, k = 0 .. k_max)
    a_k  = a~_k / sum_k a~_k      (= fraction of all m edges with |l(u)-l(v)| = k)
    OdC  = -sum_k a_k ln a_k      offdiagonal_complexity   (nats; k = 0 included)

  degree-distribution entropy
    p_d  = fraction of nodes with degree d
    H    = -sum_d p_d ln p_d      degree_distribution_entropy   (nats)

  unique degree values            unique_degrees   (number of distinct degrees)

  and, from the same pass: full_degree_pair_entropy (entropy, in nats, of all
  entries of c_mn), n_offdiag_edges, n_diag_edges, unique_degree_diffs,
  unique_degree_pairs, density 2m/(n(n-1)).

Self-contained: needs numpy and networkx only, and does not import
run_complexity.py.  load_graph_npz() and compute_offdiagonal() below are
copied from it unchanged; load_graph() is the graph-loading branch of its
main().  With the same --graph the numbers are the same as the 'offdiagonal'
block of run_complexity.py's JSON.  run_complexity.py's --skip-orbits /
--skip-eigen only switch other measures off, so they are not carried over.

Usage:
    python complexity_offdiagonal.py --graph G.npz
    python complexity_offdiagonal.py --graph G.npz --json offdiagonal.json
"""

import argparse
import json
import time
import sys
import numpy as np
import networkx as nx
from collections import Counter

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
# 5. Offdiagonal complexity (Claussen 2007)
#    (copied from run_complexity.py)
# ═══════════════════════════════════════════════════════════════════

def compute_offdiagonal(G):
    """Claussen (2007) Physica A 375:365-373: off-diagonal complexity.

    OdC = -sum_{k=0}^{k_max} a_k * ln(a_k)
    where a_k is the fraction of ALL edges connecting nodes whose degrees
    differ by exactly k.  Sum and normalization include k=0 (Eq. 5).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2.0 * m / (n * (n - 1))

    diff_counts = Counter()
    for u, v in G.edges():
        k = abs(G.degree(u) - G.degree(v))
        diff_counts[k] += 1

    probs = np.array(list(diff_counts.values()), dtype=float) / m
    OdC = float(-np.sum(probs * np.log(probs + 1e-300)))

    degree_pairs = Counter()
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        degree_pairs[(min(du, dv), max(du, dv))] += 1
    total = sum(degree_pairs.values())
    all_probs = np.array(list(degree_pairs.values()), dtype=float) / total
    H_all = float(-np.sum(all_probs * np.log(all_probs)))

    deg_counts = Counter(dict(G.degree()).values())
    deg_total = sum(deg_counts.values())
    deg_probs = np.array(list(deg_counts.values()), dtype=float) / deg_total
    H_deg = float(-np.sum(deg_probs * np.log(deg_probs)))

    n_diag_edges = sum(1 for u, v in G.edges() if G.degree(u) == G.degree(v))

    return {
        'offdiagonal_complexity': OdC,
        'full_degree_pair_entropy': H_all,
        'degree_distribution_entropy': H_deg,
        'n_offdiag_edges': m - diff_counts.get(0, 0),
        'n_diag_edges': n_diag_edges,
        'unique_degree_diffs': len(diff_counts),
        'unique_degree_pairs': len(degree_pairs),
        'unique_degrees': len(deg_counts),
        'density': float(density),
    }


# ═══════════════════════════════════════════════════════════════════
# Main  (the measure-5 part of run_complexity.py main())
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Off-diagonal complexity, degree entropy, unique degrees "
                                             "(measure 5 of run_complexity.py)")
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
    print("5. OFFDIAGONAL COMPLEXITY (Claussen 2007)")
    print(f"{'─'*70}")
    t0 = time.time()
    try:
        odc = compute_offdiagonal(G)
        results['offdiagonal'] = odc
        print(f"  Offdiag complexity:  {odc['offdiagonal_complexity']:.4f}")
        print(f"  Degree-pair entropy: {odc['full_degree_pair_entropy']:.4f} nats")
        print(f"  Degree entropy:      {odc['degree_distribution_entropy']:.4f} nats")
        print(f"  Unique degree pairs: {odc['unique_degree_pairs']:,}")
        print(f"  Unique degrees:      {odc['unique_degrees']}")
        print(f"  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['offdiagonal'] = {'error': str(e)}

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
