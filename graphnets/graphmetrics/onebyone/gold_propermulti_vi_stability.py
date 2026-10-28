#!/usr/bin/env python3
"""
gold_propermulti_vi_stability.py — the 'propermulti_vi_stability' criterion of
connectome_audit_gold.py, on its own.

Multi-resolution stability of Louvain partitions (after Betzel et al. 2017),
run on the WHOLE graph G (not its GCC), exactly as the gold audit does:

    gammas    = 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5 2.0 2.5 3.0
    at each gamma (index gi): --repeats (20) Louvain runs,
              louvain_communities(G, resolution=gamma, seed=seed + 1000*gi + r)
    mean_VI   = mean variation of information over all 190 pairs of the runs
              VI(X, Y) = H(X) + H(Y) - 2 I(X; Y)   (natural log; Meila 2007)
    minima    = local minima of mean_VI over gamma (_find_local_minima: a
              flat-bottomed valley counts once; the ends of the sweep never count)
    Q at a minimum = mean STANDARD modularity (gamma = 1) of that gamma's runs
    verdict   = 'pass' if at least one minimum has Q > 0.3 else 'fail'

There is no null ensemble in this criterion, so no --n-null / --workers; the
result is the same with or without the audit's --no-null.  It runs serially,
like the audit.

Self-contained: needs numpy and networkx only, and does not import
connectome_audit_gold.py.  The loading, VI, minimum-finding and _propermulti
functions below are copied from it unchanged, so with the same --graph and
--seed the result is the same as the full audit's 'propermulti' entry and its
'propermulti_vi_stability' verdict.

Usage:
    python gold_propermulti_vi_stability.py --graph G.npz
    python gold_propermulti_vi_stability.py --graph G.npz --seed 42 --json vi.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
# Propermulti — Betzel-Bassett gamma-sweep + VI stability
# (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _partition_to_labels(nodes, partition):
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    labels = np.full(len(nodes), -1, dtype=int)
    for cid, community in enumerate(partition):
        for node in community:
            labels[node_to_idx[node]] = cid
    return labels


def _entropy_from_counts(counts):
    probs = counts[counts > 0].astype(float)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def variation_of_information(labels_x, labels_y):
    labels_x = np.asarray(labels_x)
    labels_y = np.asarray(labels_y)
    n = labels_x.size
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    _, x = np.unique(labels_x, return_inverse=True)
    _, y = np.unique(labels_y, return_inverse=True)
    kx, ky = int(x.max()) + 1, int(y.max()) + 1
    contingency = np.zeros((kx, ky), dtype=np.int64)
    np.add.at(contingency, (x, y), 1)
    counts_x = contingency.sum(axis=1)
    counts_y = contingency.sum(axis=0)
    hx = _entropy_from_counts(counts_x)
    hy = _entropy_from_counts(counts_y)
    pxy = contingency.astype(float) / n
    px = counts_x.astype(float) / n
    py = counts_y.astype(float) / n
    mutual_info = 0.0
    rows, cols = np.where(pxy > 0)
    for i, j in zip(rows, cols):
        mutual_info += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return max(0.0, hx + hy - 2.0 * mutual_info)


#: tolerance for treating two mean-VI values as equal when detecting a plateau.
#: VI values of interest span 0.02-2.5, so this only ever merges values that are
#: numerically identical (exact 0.0, or float noise such as 2.2e-16 around it).
_VI_PLATEAU_TOL = 1e-12


def _find_local_minima(xs, ys):
    """Local minima of ys, treating a FLAT-BOTTOMED valley as a single minimum.

    The strict two-sided test -- ys[i] < ys[i-1] and ys[i] < ys[i+1] -- misses a
    plateau: a run of equal values lying below the points on either side.  That
    is not a pathological case here.  Mean VI is exactly 0.0 when the partition
    is perfectly reproducible across all repeats, which is MAXIMAL stability, and
    several consecutive resolutions can share it.  The strict test scored such a
    run as no minimum at all -- i.e. as unstable -- so on this criterion it could
    only ever produce FALSE NEGATIVES, never false positives.

    Observed: a degree-corrected SBM with mean VI exactly 0.0 at gamma =
    1.0, 1.2, 1.5 and 2.0 was scored 0 stable scales and failed, while a graph
    with a single noisy dip passed.

    A plateau spanning indices [i..j] counts as one minimum when ys[i] lies
    strictly below both ys[i-1] and ys[j+1]; the plateau's midpoint is reported.
    A single-point valley (i == j) reduces to the original strict test exactly,
    which is why this cannot change a verdict that did not involve a plateau --
    verified against every audit in the repository by
    test_find_local_minima.py.  Plateaus touching either end of the sweep are
    still not minima: a comparison on both sides is required.
    """
    out = []
    n = len(ys)
    i = 1
    while i < n - 1:
        j = i
        while j + 1 < n and abs(ys[j + 1] - ys[i]) <= _VI_PLATEAU_TOL:
            j += 1
        if (j + 1 < n
                and ys[i] < ys[i - 1] - _VI_PLATEAU_TOL
                and ys[i] < ys[j + 1] - _VI_PLATEAU_TOL):
            k = (i + j) // 2
            out.append((xs[k], ys[k]))
        i = j + 1
    return out


def _propermulti(G: nx.Graph, seed: int = 42, repeats: int = 20) -> Dict[str, object]:
    """Betzel-Bassett multi-resolution VI stability (Betzel 2017)."""
    from networkx.algorithms.community import louvain_communities
    nodes = list(G.nodes())
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    mean_vis, std_vis, all_Q, all_Q1, all_ncomm = [], [], [], [], []

    for gi, gamma in enumerate(gammas):
        t0 = time.time()
        labelings, qs, q1s, ncomms = [], [], [], []
        for r in range(repeats):
            comms = list(louvain_communities(G, resolution=float(gamma),
                                            seed=seed + 1000 * gi + r))
            labels = _partition_to_labels(nodes, comms)
            labelings.append(labels)
            qs.append(float(nx.community.modularity(G, comms, resolution=float(gamma))))
            q1s.append(float(nx.community.modularity(G, comms)))  # standard Q (gamma=1)
            ncomms.append(len(comms))

        n_rep = len(labelings)
        vi_vals = []
        for i in range(n_rep):
            for j in range(i + 1, n_rep):
                vi_vals.append(variation_of_information(labelings[i], labelings[j]))
        mv = float(np.mean(vi_vals))
        sv = float(np.std(vi_vals, ddof=1)) if len(vi_vals) > 1 else 0.0
        mean_vis.append(mv)
        std_vis.append(sv)
        all_Q.append(qs)
        all_Q1.append(q1s)
        all_ncomm.append(ncomms)
        print(f"    gamma={gamma:.2f}  mean_VI={mv:.4f}  Q={np.mean(qs):.4f}  "
              f"n_comm={np.mean(ncomms):.1f}  ({time.time()-t0:.1f}s)")

    minima = _find_local_minima(gammas, mean_vis)
    Q_at_minima = []
    for g_min, _ in minima:
        idx = gammas.index(g_min)
        # FIX: the Q > 0.3 threshold (Newman & Girvan 2004) is for STANDARD
        # modularity.  Q_gamma rises as gamma falls (one community gives
        # Q_gamma = 1 - gamma), so thresholding Q_gamma passed weak partitions
        # at low gamma.  Threshold the standard Q of the same partitions.
        Q_at_minima.append(float(np.mean(all_Q1[idx])))

    n_stable_modular = sum(1 for q in Q_at_minima if q > 0.3)

    return {
        'gammas': gammas,
        'mean_vi': mean_vis, 'std_vi': std_vis,
        'local_minima': minima,
        'Q_at_minima': Q_at_minima,
        'n_stable_modular': n_stable_modular,
        'Q_per_gamma': {g: {'mean': float(np.mean(q)), 'std': float(np.std(q))}
                        for g, q in zip(gammas, all_Q)},
        'ncomm_per_gamma': {g: {'mean': float(np.mean(nc)), 'std': float(np.std(nc))}
                            for g, nc in zip(gammas, all_ncomm)},
        'repeats': repeats, 'seed': seed,
        'pass': n_stable_modular >= 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# The criterion  (run_audit's '7. Multi-scale VI stability' and the verdict)
# ═══════════════════════════════════════════════════════════════════════════


def _propermulti_verdict_of(propermulti: Dict[str, object]) -> str:
    """run_audit's nested _propermulti_verdict, with its closure variable as an argument."""
    # Betzel 2017: VI local minimum with Q > 0.3 (Newman & Girvan 2004)
    if propermulti.get('n_stable_modular', 0) >= 1:
        return 'pass'
    return 'fail'


def propermulti_vi_stability(layered: LayeredGraph, seed: int = 42,
                             repeats: int = 20) -> Dict[str, object]:
    G = layered.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Nodes: {n:,}   Edges: {m:,}   (run on the whole graph, not its GCC, as in the audit)")
    print(f"  {repeats} Louvain runs per gamma, seeds seed + 1000*gi + r, seed = {seed}")
    propermulti = _propermulti(G, seed=seed, repeats=repeats)
    if propermulti['local_minima']:
        for g_min, vi_min in propermulti['local_minima']:
            idx = propermulti['gammas'].index(g_min)
            q_at = propermulti['Q_at_minima'][propermulti['local_minima'].index((g_min, vi_min))]
            print(f"  VI local minimum at gamma={g_min}: VI={vi_min:.4f}, Q={q_at:.4f}")
    else:
        print("  No VI local minima found")
    print(f"  Stable scales with Q>0.3: {propermulti['n_stable_modular']}")
    return {'graph': {'n_nodes': n, 'n_edges': m},
            'propermulti': propermulti,
            'propermulti_vi_stability': _propermulti_verdict_of(propermulti)}


def main():
    ap = argparse.ArgumentParser(description="propermulti_vi_stability, the ninth of the gold audit's nine criteria")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--repeats', type=int, default=20,
                    help='Louvain runs per gamma (the audit uses the default, 20)')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    res = propermulti_vi_stability(layered, seed=args.seed, repeats=args.repeats)

    pm = res['propermulti']
    print("\n" + "=" * 72)
    print("MULTI-SCALE VI STABILITY (propermulti, Betzel 2017)")
    print("=" * 72)
    print(f"  {'gamma':>6}  {'mean_VI':>8}  {'Q':>8}  {'n_comm':>8}")
    print("  " + "-" * 38)
    for i, g in enumerate(pm['gammas']):
        qm = pm['Q_per_gamma'][g]['mean']
        nc = pm['ncomm_per_gamma'][g]['mean']
        print(f"  {g:6.2f}  {pm['mean_vi'][i]:8.4f}  {qm:8.4f}  {nc:8.1f}")
    if pm['local_minima']:
        for idx, (g_min, vi_min) in enumerate(pm['local_minima']):
            q_at = pm['Q_at_minima'][idx]
            tag = " *" if q_at > 0.3 else ""
            print(f"  VI minimum at gamma={g_min}: Q={q_at:.4f}{tag}")
    else:
        print("  No VI local minima")
    print(f"  Stable scales with Q>0.3: {pm['n_stable_modular']} -> "
          f"{'PASS' if pm['pass'] else 'FAIL'}")
    print(f"  propermulti_vi_stability: {res['propermulti_vi_stability'].upper()}"
          f"   (pass if a mean-VI local minimum has standard Q > 0.3)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
