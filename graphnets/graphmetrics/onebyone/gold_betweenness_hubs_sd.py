#!/usr/bin/env python3
"""
gold_betweenness_hubs_sd.py — betweenness hubs by the mean + 1 SD cutoff,
with the audit's 2 x mean count kept alongside as the legacy result.

A companion to gold_betweenness_hubs.py (same folder), whose loader it reuses
unchanged.  Descriptive, not scored, and no null model: the cutoff is relative
to the other nodes of the same graph.

    bc        shortest-path node betweenness (Freeman 1977; Brandes 2001),
              normalized by (n-1)(n-2)/2, endpoints excluded, on the WHOLE
              graph (not its GCC), as in the audit
                --k 200 (default)  nx.betweenness_centrality(G, k=200, seed),
                                   the audit's 200-source estimate
                                   (Brandes & Pich 2007)
                --k 0              exact: every node a source, the sources
                                   split across --workers processes
    mean, std mean and population SD (ddof=0) of bc over all nodes
    hubs      nodes with bc > mean + std          (this metric)
    legacy    nodes with bc > 2 * mean            (the audit's n_hubs, unchanged)

Cutoff: one SD above the mean, applied to betweenness alone.
    Oldham & Fornito 2019 (Dev. Cogn. Neurosci. 36:100607), Fig. 2: hubs are
    nodes ">= 1 standard deviation above the mean on at least one centrality
    measure (degree, nodal efficiency, or betweenness)".  That is a combined
    rule, so betweenness alone is an adaptation of it.
    Lord et al. 2016 (Front. Syst. Neurosci. 10:85), sec. 2.3.4: hubs are
    nodes "with a metric value larger than 1 standard deviation from the
    mean", for each measure separately, betweenness included.
The cutoff is a convention, not a significance test.  It marks unusually high
shortest-path intermediaries, not necessarily cross-module connectors (that is
gold_connector_hubs.py).

Usage:
    python gold_betweenness_hubs_sd.py --graph G.npz --json out.json            # audit estimate
    python gold_betweenness_hubs_sd.py --graph G.npz --k 0 --workers 8 --json out.json   # exact
    python gold_betweenness_hubs_sd.py --self-test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Dict, List

import numpy as np
import networkx as nx

from gold_betweenness_hubs import load_layered_graph

sys.stdout.reconfigure(line_buffering=True)


def hub_summary(vals: np.ndarray) -> Dict[str, float]:
    """Mean + 1 SD hubs and the legacy 2 x mean count, from per-node bc."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0 or vals.mean() == 0:
        return {'mean': 0, 'max': 0, 'std': 0, 'max_over_mean': float('nan'),
                'cutoff': 0, 'n_hubs': 0, 'cutoff_legacy': 0, 'n_hubs_legacy': 0}
    mean, std = float(vals.mean()), float(vals.std(ddof=0))
    cutoff, cutoff_legacy = mean + std, 2.0 * mean
    return {
        'mean': mean, 'max': float(vals.max()), 'std': std,
        'max_over_mean': float(vals.max() / mean),
        'cutoff': cutoff, 'n_hubs': int(np.sum(vals > cutoff)),
        'cutoff_legacy': cutoff_legacy, 'n_hubs_legacy': int(np.sum(vals > cutoff_legacy)),
    }


_G = None  # the graph, shared with forked workers


def _subset_sum(sources: List[int]) -> Dict[int, float]:
    # All targets: sum over these sources of the pair dependencies, halved for
    # an undirected graph (networkx's unnormalized scale).
    return nx.betweenness_centrality_subset(_G, sources=sources, targets=list(_G), normalized=False)


def exact_betweenness(G: nx.Graph, workers: int = 1) -> Dict[int, float]:
    """Exact normalized betweenness, equal to nx.betweenness_centrality(G):
    the sources are split into chunks and the chunk sums added."""
    global _G
    _G = G
    nodes = list(G)
    n = len(nodes)
    chunks = [nodes[i::max(1, workers) * 4] for i in range(max(1, workers) * 4)]
    chunks = [c for c in chunks if c]
    total = dict.fromkeys(nodes, 0.0)
    if workers <= 1:
        parts = map(_subset_sum, chunks)
    else:
        pool = mp.get_context('fork').Pool(workers)
        parts = pool.imap_unordered(_subset_sum, chunks)
    for done, part in enumerate(parts, 1):
        for v, x in part.items():
            total[v] += x
        if done % max(1, len(chunks) // 10) == 0 or done == len(chunks):
            print(f"    {done}/{len(chunks)} source chunks", flush=True)
    if workers > 1:
        pool.close(); pool.join()
    scale = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    return {v: x * scale for v, x in total.items()}


def _self_test() -> None:
    # Strict >: [0, 0, 1, 1] has mean 0.5, SD 0.5, cutoff exactly 1.0, so no hub.
    s = hub_summary(np.array([0.0, 0.0, 1.0, 1.0]))
    assert s['cutoff'] == 1.0 and s['n_hubs'] == 0 and s['n_hubs_legacy'] == 0, s
    # ddof=0: 0.47 lies above mean + SD(ddof=0) = 0.4640 and below mean + SD(ddof=1) = 0.4811.
    v = np.array([0.0] * 8 + [0.47, 1.0])
    s = hub_summary(v)
    assert v.mean() + v.std(ddof=1) > 0.47 > s['cutoff'] and s['n_hubs'] == 2, s
    # Legacy factor 2: mean 0.25, 2 x mean = 0.5 -> 0 hubs (1 x mean would give 3).
    s = hub_summary(np.array([0.0, 0.3, 0.3, 0.4]))
    assert s['n_hubs_legacy'] == 0 and abs(s['cutoff_legacy'] - 0.5) < 1e-12, s
    # Empty or all-zero input.
    assert hub_summary(np.array([]))['n_hubs'] == 0
    assert hub_summary(np.zeros(5))['n_hubs'] == 0
    # The legacy fields equal the audit's own function, on a real graph.
    from gold_betweenness_hubs import _betweenness_hubs
    G = nx.karate_club_graph()
    for k in (10, 34):
        audit = _betweenness_hubs(G, k=k, seed=42)
        mine = hub_summary(np.array(list(nx.betweenness_centrality(G, k=k, seed=42).values())))
        assert mine['n_hubs_legacy'] == audit['n_hubs'], (k, mine, audit)
        for key in ('mean', 'max', 'std', 'max_over_mean'):
            assert mine[key] == audit[key], (k, key)
    # Exact mode equals networkx's exact betweenness, serial and parallel.
    ref = nx.betweenness_centrality(G)
    for workers in (1, 3):
        ex = exact_betweenness(G, workers=workers)
        assert set(ex) == set(ref) and max(abs(ex[u] - ref[u]) for u in G) < 1e-12, workers
    print("Betweenness-hub self-tests passed.")


def main():
    ap = argparse.ArgumentParser(description="Betweenness hubs, mean + 1 SD cutoff (legacy 2 x mean alongside)")
    ap.add_argument('--graph', help='Path to the graph (.npz or .gml)')
    ap.add_argument('--k', type=int, default=200,
                    help='Sampled source nodes (the audit uses 200); 0 = exact (every node a source)')
    ap.add_argument('--seed', type=int, default=42, help='Source sample seed (sampled mode only)')
    ap.add_argument('--workers', type=int, default=1, help='Processes for the exact mode')
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--json', default='')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.graph:
        ap.error('--graph is required')
    if args.json and os.path.exists(args.json):
        ap.error(f'{args.json} exists; choose a new path')

    G = load_layered_graph(args.graph).G
    n = G.number_of_nodes()
    exact = args.k <= 0 or args.k >= n
    print(f"  Nodes: {n:,}   Edges: {G.number_of_edges():,}   "
          + (f"exact (all {n} sources, {args.workers} workers)" if exact
             else f"sampled: {args.k} of {n} sources, seed {args.seed} (as in the audit)")
          + "; whole graph")
    if exact:
        per_node = exact_betweenness(G, workers=args.workers)
    else:
        per_node = nx.betweenness_centrality(G, k=args.k, seed=args.seed)
    s = hub_summary(np.array(list(per_node.values())))
    ranked = sorted(per_node.items(), key=lambda kv: (-kv[1], kv[0]))
    hubs = [int(u) for u, x in ranked if x > s['cutoff']]
    legacy = [int(u) for u, x in ranked if x > s['cutoff_legacy']]

    print("\n" + "=" * 72)
    print("BETWEENNESS HUBS  (mean + 1 SD; legacy 2 x mean alongside)")
    print("=" * 72)
    print(f"  mean {s['mean']:.6g}   SD (ddof=0) {s['std']:.6g}   max {s['max']:.6g}   "
          f"max/mean {s['max_over_mean']:.4g}")
    print(f"  hubs   bc > mean + SD = {s['cutoff']:.6g}:  {s['n_hubs']} of {n} nodes ({100 * s['n_hubs'] / n:.1f}%)")
    print(f"  legacy bc > 2 x mean  = {s['cutoff_legacy']:.6g}:  {s['n_hubs_legacy']} of {n} nodes")
    print(f"  top {min(args.top, n)} nodes (node index: normalized betweenness):")
    for u, x in ranked[:args.top]:
        print(f"    {u:>8}: {x:.6f}")

    if args.json:
        with open(args.json, 'x') as f:
            json.dump({'graph': {'path': args.graph, 'n_nodes': n, 'n_edges': G.number_of_edges()},
                       'mode': 'exact' if exact else 'sampled',
                       'k_sources': n if exact else args.k,
                       'seed': None if exact else args.seed,
                       'cutoff_rule': 'bc > mean + std (ddof=0)',
                       'legacy_rule': 'bc > 2 * mean',
                       'betweenness': s, 'hubs': hubs, 'hubs_legacy': legacy,
                       'top_nodes': [[int(u), float(x)] for u, x in ranked[:args.top]],
                       'per_node': {int(u): float(x) for u, x in per_node.items()}}, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
