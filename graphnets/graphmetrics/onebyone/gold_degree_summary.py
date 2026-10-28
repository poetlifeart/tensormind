#!/usr/bin/env python3
"""
gold_degree_summary.py — the 'degree' block of connectome_audit_gold.py
(descriptive, not scored), on its own.

_degree_summary(G) on the WHOLE graph (not its GCC), as in the audit:

    min, max, mean, median, std (population, ddof=0), cv = std/mean,
    gini        = 2 sum_i i x_(i) / (n sum x) - (n+1)/n     (x sorted ascending, i = 1..n)
    dispersion  = var/mean,  p90, p99 (np.percentile, linear),  max_over_mean,
    hill_tail   = Hill (1975) tail-index estimate on the positive degrees sorted
                  descending, x_1 >= x_2 >= ...:
                      k   = min(max(10, n // 10), 200), then min(k, n - 1)
                      hill_tail = 1 / mean_{i=1..k} ln(x_i / x_(k+1))
                  i.e. the exponent of the complementary CDF, P(X > x) ~ x^-hill_tail,
                  from the k largest degrees above the (k+1)-th largest
                  (nan if fewer than 10 positive degrees or k < 5).
                  A density exponent alpha (p(x) ~ x^-alpha) is hill_tail + 1.

No null model.  Self-contained: needs numpy and networkx only, and does not
import connectome_audit_gold.py.  The loading functions, _gini, _hill and
_degree_summary below are copied from it unchanged, so with the same --graph
the result is the same as the 'degree' entry of the full audit.  The driver
also prints the Hill k and threshold degree it used (not in the audit).

Usage:
    python gold_degree_summary.py --graph G.npz
    python gold_degree_summary.py --graph G.npz --json degree.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
# Loading  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LayeredGraph:
    G: nx.Graph
    masks: Dict[str, np.ndarray]
    sizes: Tuple[int, int, int]
    has_edge_list: bool = False


def load_layered_graph(path: str) -> LayeredGraph:
    if path.endswith('.gml'):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
        n = G.number_of_nodes()
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
        masks = {k: np.zeros((1, 1), dtype=np.uint8) for k in ['12', '13', '23']}
        print(f"  [Loaded GML: {n} nodes, {G.number_of_edges()} edges]")
        return LayeredGraph(G=G, masks=masks, sizes=(n1, n2, n3), has_edge_list=True)

    data = np.load(path)

    # Check for edge-list format FIRST (χ>3 graphs may lack masks)
    if 'edges' in data:
        n_total = int(data['n_total']) if 'n_total' in data else (
            int(data['n1']) + int(data['n2']) + int(data['n3']))
        edges = data['edges']
        G = nx.Graph()
        G.add_nodes_from(range(n_total))
        G.add_edges_from((int(u), int(v)) for u, v in edges)
        print(f"  [Loaded from edge list: {G.number_of_edges()} edges]")
        if 'n1' in data:
            n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        else:
            n1 = n_total // 3
            n2 = n_total // 3
            n3 = n_total - 2 * (n_total // 3)
        masks = {}
        for key in ['mask_12', 'mask_13', 'mask_23']:
            if key in data:
                masks[key[5:]] = (data[key] > 0).astype(np.uint8)
            else:
                masks[key[5:]] = np.zeros((1, 1), dtype=np.uint8)
        return LayeredGraph(G=G, masks=masks,
                            sizes=(n1, n2, n3), has_edge_list=True)

    # Standard mask-based loading (χ=3 tripartite graphs)
    n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
    mask_12 = (data['mask_12'] > 0).astype(np.uint8)
    mask_13 = (data['mask_13'] > 0).astype(np.uint8)
    mask_23 = (data['mask_23'] > 0).astype(np.uint8)
    G = _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23)
    return LayeredGraph(G=G, masks={'12': mask_12, '13': mask_13, '23': mask_23},
                        sizes=(n1, n2, n3), has_edge_list=False)


def _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23):
    G = nx.Graph()
    off1, off2, off3 = 0, n1, n1 + n2
    G.add_nodes_from(range(n1 + n2 + n3))
    rows, cols = np.where(mask_12)
    G.add_edges_from((off1 + int(c), off2 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_13)
    G.add_edges_from((off1 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_23)
    G.add_edges_from((off2 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Degree distribution  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or np.all(x == 0):
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * x)) / (n * np.sum(x)) - (n + 1) / n)


def _hill(x: np.ndarray, k: int | None = None) -> float:
    x = np.sort(np.asarray(x, dtype=float)[np.asarray(x, dtype=float) > 0])[::-1]
    if x.size < 10:
        return float('nan')
    if k is None:
        k = min(max(10, x.size // 10), 200)
    k = min(k, x.size - 1)
    if k < 5 or x[k] <= 0:
        return float('nan')
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def _degree_summary(G: nx.Graph) -> Dict[str, float]:
    deg = np.array([d for _, d in G.degree()], dtype=float)
    if deg.size == 0:
        return {}
    mu = float(deg.mean())
    return {
        'min': float(deg.min()), 'max': float(deg.max()),
        'mean': mu, 'median': float(np.median(deg)),
        'std': float(deg.std()), 'cv': float(deg.std() / mu) if mu > 0 else float('nan'),
        'gini': _gini(deg),
        'dispersion': float(deg.var() / mu) if mu > 0 else float('nan'),
        'p90': float(np.percentile(deg, 90)),
        'p99': float(np.percentile(deg, 99)),
        'max_over_mean': float(deg.max() / mu) if mu > 0 else float('nan'),
        'hill_tail': _hill(deg),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════


def _hill_detail(deg: np.ndarray) -> Dict[str, float]:
    """The k and threshold _hill uses with its default k (not in the audit)."""
    x = np.sort(deg[deg > 0])[::-1]
    if x.size < 10:
        return {'n_positive': int(x.size), 'k': None, 'threshold_degree': None}
    k = min(min(max(10, x.size // 10), 200), x.size - 1)
    return {'n_positive': int(x.size), 'k': int(k),
            'tail_degrees_from': float(x[k - 1]), 'threshold_degree': float(x[k])}


def main():
    ap = argparse.ArgumentParser(description="Degree summary (Gini, Hill tail) of the gold audit, standalone")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    G = layered.G
    print(f"  Nodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}   "
          f"(degree summary on the whole graph, as in the audit)")
    deg = _degree_summary(G)
    hd = _hill_detail(np.array([d for _, d in G.degree()], dtype=float))

    print("\n" + "=" * 72)
    print("DEGREE DISTRIBUTION")
    print("=" * 72)
    for k in ['min', 'max', 'mean', 'median', 'std', 'cv', 'gini',
              'dispersion', 'p90', 'p99', 'max_over_mean', 'hill_tail']:
        v = deg.get(k, float('nan'))
        print(f"  {k:>16s}: {v:.6g}" if np.isfinite(v) else f"  {k:>16s}: nan")
    if hd['k'] is not None:
        print(f"  Hill: top k = {hd['k']} of {hd['n_positive']} positive degrees "
              f"(degrees {hd['tail_degrees_from']:g} and up), threshold x_(k+1) = {hd['threshold_degree']:g};"
              f"  density exponent hill_tail + 1 = {deg['hill_tail'] + 1:.4g}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'graph': {'n_nodes': G.number_of_nodes(), 'n_edges': G.number_of_edges()},
                       'degree': deg, 'hill_detail': hd}, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
