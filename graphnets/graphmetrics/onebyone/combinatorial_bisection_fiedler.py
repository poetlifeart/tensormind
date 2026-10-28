#!/usr/bin/env python3
"""
combinatorial_bisection_fiedler.py — the 'MinCutBalDivSum' measure of
combinatorial_metrics_npz.py (spectral balanced bisection + Fiedler value), on
its own.

    L          = D - A  (graph Laplacian)
    fiedler    = eigenvector of the second-smallest eigenvalue of L
                 (scipy.sparse.linalg.eigsh, k=2, which='SM', columns re-sorted)
    fiedler_value = that eigenvalue, lambda_2(L) (algebraic connectivity, Fiedler 1973)
    bisection  = nodes ranked by their Fiedler-vector entry (stable argsort);
                 the first floor(n/2) form one side.  For odd n the split with
                 ceil(n/2) on that side is also tried and the smaller cut kept.
    cut_size   = edges between the two sides;  normalized = cut_size / m

MinCutBalDivSum of Szalkai, Varga & Grolmusz (arXiv:1603.00904;
arXiv:1709.04974) is the minimum perfectly balanced bisection divided by m.
This is a spectral HEURISTIC for it (Fiedler 1975; Pothen, Simon & Liou 1990),
so cut_size is an UPPER bound on the minimum bisection.

Self-contained: needs numpy, scipy and networkx only, and does not import
combinatorial_metrics_npz.py.  load_graph and min_cut_bal below are copied
from it unchanged, so with the same --graph the result is the 'min_cut_bal'
entry of its JSON.  The last digits of fiedler_value (and, in principle, the
order of nodes whose Fiedler entries tie) can differ: ARPACK's start vector
depends on how many ARPACK calls came before in the same process, and
min_cut_bal is the fifth or sixth such call in combinatorial_metrics_npz.py
but the first here.

Usage:
    python combinatorial_bisection_fiedler.py --graph G.npz
    python combinatorial_bisection_fiedler.py --graph G.npz --json bisection.json
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


def min_cut_bal(G: nx.Graph) -> dict:
    """Approximate balanced bisection via Fiedler vector (spectral).
    MinCutBalDivSum = cut_size / num_edges."""
    t0 = time.time()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    L = nx.laplacian_matrix(G).astype(float)
    # eigsh(which='SM') does NOT return eigenvalues in ascending order -- it
    # returns [lambda_2, lambda_1] here -- so the columns must be reordered
    # explicitly.  Taking vecs[:, 1] unsorted picks the constant lambda_1
    # eigenvector and bisects on numerical noise.
    vals, vecs = scipy.sparse.linalg.eigsh(L, k=2, which='SM')
    order = np.argsort(vals)
    fiedler = vecs[:, order[1]]

    # eigsh fixes neither the eigenvalue order nor the eigenvector sign, and for
    # odd n the median is an element of the vector, so a '<= median' split keeps
    # the tied node on the same side under both signs -- the two splits are then
    # not complements and the cut is not reproducible.  Split by rank instead,
    # and since this is a *minimum* balanced cut, take the better of the two
    # admissible balanced splits when n is odd.
    nodes = list(G.nodes())
    rank = np.argsort(fiedler, kind='stable')

    def _cut(size):
        part_a = set(nodes[i] for i in rank[:size])
        return sum(1 for u, v in G.edges() if (u in part_a) != (v in part_a)), part_a

    cut, part_a = _cut(n // 2)
    if n % 2:
        cut_alt, part_alt = _cut(n // 2 + 1)
        if cut_alt < cut:
            cut, part_a = cut_alt, part_alt
    part_b = set(nodes) - part_a
    fiedler_val = float(vals[order[1]])

    return {
        'cut_size': cut,
        'normalized': cut / m if m > 0 else 0,
        'partition_sizes': [len(part_a), len(part_b)],
        'fiedler_value': fiedler_val,
        'note': 'spectral approx (Fiedler bisection)',
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

    print("7. MinCutBalDivSum (spectral bisection)...")
    results['min_cut_bal'] = min_cut_bal(G)
    r = results['min_cut_bal']
    print(f"   cut={r['cut_size']} ({r['normalized']:.4f} norm)")
    print(f"   Fiedler={r['fiedler_value']:.6f}")
    print(f"   partition={r['partition_sizes']} [{r['time']:.1f}s]")
    print(f"   full precision: cut_size={r['cut_size']}  normalized={r['normalized']!r}  "
          f"fiedler_value={r['fiedler_value']!r}")
    return results


def main():
    ap = argparse.ArgumentParser(description="MinCutBalDivSum (spectral bisection + Fiedler value) of combinatorial_metrics_npz.py")
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
