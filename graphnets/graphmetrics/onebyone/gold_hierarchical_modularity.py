#!/usr/bin/env python3
"""
gold_hierarchical_modularity.py — the descriptive 'hierarchical' entry of
connectome_audit_gold.py (legacy hierarchical modularity), on its own.

Louvain at six resolutions on the WHOLE graph G (not its GCC), exactly as the
gold audit does, repeated for n_seeds = 10 Louvain seeds (seed .. seed + 9):

    gammas     = 0.5 0.8 1.0 1.5 2.0 3.0
    per gamma  Q (= modularity at that gamma, Q_gamma), Q_standard (= modularity
               of the same partition at gamma = 1), n_comm
    n_modular  = #{gammas with Q_standard > 0.3}
    nesting    = for each consecutive pair (coarse = gamma_i, fine = gamma_i+1):
               mean over fine communities of the fraction of their nodes lying
               in the single most common coarse community; mean over the 5 pairs
    reported   'sweep', 'n_modular', 'nesting_scores' of the first seed that
               reaches the highest n_modular (seed 42 unless a later seed has
               more); 'mean_nesting', 'std_nesting' (ddof 1) over all seeds
    'pass'     = n_modular > 1  (descriptive; NOT in the audit's evidence
               scorecard; run_audit's _legacy_hier_verdict is defined but unused)

There is no null ensemble, so no --n-null / --workers; the result is the same
with or without the audit's --no-null.  It runs serially, like the audit.

Self-contained: needs numpy and networkx only, and does not import
connectome_audit_gold.py.  The loading, Louvain and _hierarchical_modularity
functions below are copied from it unchanged, so with the same --graph and
--seed the result is the same as the full audit's 'hierarchical' entry.

Usage:
    python gold_hierarchical_modularity.py --graph G.npz
    python gold_hierarchical_modularity.py --graph G.npz --seed 42 --json hier.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

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
# Community structure and hierarchical modularity
# (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _louvain(G: nx.Graph, seed: int = 42, resolution: float = 1.0):
    try:
        from networkx.algorithms.community import louvain_communities, modularity
        comms = louvain_communities(G, seed=seed, resolution=resolution)
        Q = float(modularity(G, comms, resolution=resolution))
    except Exception:
        import community as community_louvain
        part = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
        by_c = {}
        for node, c in part.items():
            by_c.setdefault(c, set()).add(node)
        comms = list(by_c.values())
        Q = float(community_louvain.modularity(part, G))
    partition = {}
    for i, comm in enumerate(comms):
        for node in comm:
            partition[node] = i
    return Q, comms, partition


def _hierarchical_modularity(G: nx.Graph, seed: int = 42, n_seeds: int = 10) -> Dict[str, object]:
    """Legacy hierarchical modularity (kept for backward compat, not scored)."""
    from collections import Counter
    gammas = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    def _run_one_seed(s):
        partitions = []
        sweep = []
        for g in gammas:
            Q, comms, part = _louvain(G, seed=s, resolution=g)
            # FIX 2026-10-23, same as in _propermulti: the Q > 0.3 threshold is
            # for STANDARD modularity, but _louvain returns Q_gamma here (or
            # standard Q under the python-louvain fallback), which rises as gamma
            # falls.  Threshold the standard Q of the same partition; 'Q' stays
            # Q_gamma so earlier JSONs remain comparable.
            Q1 = float(nx.community.modularity(G, comms))
            sweep.append({'gamma': g, 'Q': Q, 'Q_standard': Q1, 'n_comm': len(comms)})
            partitions.append(part)
        n_mod = sum(1 for r in sweep if r['Q_standard'] > 0.3)
        nesting_scores = []
        for i in range(len(gammas) - 1):
            coarse = partitions[i]
            fine = partitions[i + 1]
            fine_comms: Dict[int, set] = {}
            for node, fc in fine.items():
                fine_comms.setdefault(fc, set()).add(node)
            containment = []
            for fc_nodes in fine_comms.values():
                coarse_labels = [coarse[n] for n in fc_nodes]
                most_common_count = max(Counter(coarse_labels).values())
                containment.append(most_common_count / len(fc_nodes))
            nesting_scores.append(float(np.mean(containment)))
        mean_nesting = float(np.mean(nesting_scores)) if nesting_scores else 0.0
        return sweep, partitions, n_mod, nesting_scores, mean_nesting

    all_nestings = []
    best_sweep = best_partitions = best_nesting_scores = None
    best_n_mod = 0
    for i in range(n_seeds):
        s = seed + i
        sweep, partitions, n_mod, nesting_scores, mn = _run_one_seed(s)
        all_nestings.append(mn)
        if i == 0 or n_mod > best_n_mod:
            best_sweep, best_partitions, best_n_mod = sweep, partitions, n_mod
            best_nesting_scores = nesting_scores

    mean_nesting = float(np.mean(all_nestings))
    std_nesting = float(np.std(all_nestings, ddof=1)) if len(all_nestings) > 1 else 0.0
    return {
        'sweep': best_sweep, 'n_modular': best_n_mod, 'total': len(gammas),
        'nesting_scores': best_nesting_scores, 'mean_nesting': mean_nesting,
        'std_nesting': std_nesting, 'n_seeds': n_seeds,
        'all_nestings': [round(x, 4) for x in all_nestings],
        'pass': best_n_mod > 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# The descriptive entry  (run_audit: hier = _hierarchical_modularity(G, seed=seed))
# ═══════════════════════════════════════════════════════════════════════════


def hierarchical_modularity(layered: LayeredGraph, seed: int = 42,
                            n_seeds: int = 10) -> Dict[str, object]:
    G = layered.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Nodes: {n:,}   Edges: {m:,}   (run on the whole graph, not its GCC, as in the audit)")
    print(f"  Louvain seeds {seed}..{seed + n_seeds - 1}")
    hier = _hierarchical_modularity(G, seed=seed, n_seeds=n_seeds)
    return {'graph': {'n_nodes': n, 'n_edges': m}, 'seed': seed,
            'hierarchical': hier}


def main():
    ap = argparse.ArgumentParser(description="hierarchical modularity, a descriptive entry of the gold audit")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-seeds', type=int, default=10,
                    help='Louvain seeds (the audit uses the default, 10)')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    res = hierarchical_modularity(layered, seed=args.seed, n_seeds=args.n_seeds)

    h = res['hierarchical']
    print("\n" + "=" * 72)
    print("HIERARCHICAL MODULARITY (legacy, descriptive)")
    print("=" * 72)
    print(f"  {'gamma':>6}  {'Q_gamma':>9}  {'Q_standard':>10}  {'n_comm':>6}")
    print("  " + "-" * 38)
    for r in h['sweep']:
        print(f"  {r['gamma']:6.2f}  {r['Q']:9.4f}  {r['Q_standard']:10.4f}  {r['n_comm']:6d}")
    print(f"  n_modular (Q_standard > 0.3): {h['n_modular']}/{h['total']}")
    print(f"  nesting scores (best seed): {[round(x, 4) for x in h['nesting_scores']]}")
    print(f"  mean nesting over {h['n_seeds']} seeds: {h['mean_nesting']:.4f} +- {h['std_nesting']:.4f}")
    print(f"  pass (descriptive, n_modular > 1): {h['pass']}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
