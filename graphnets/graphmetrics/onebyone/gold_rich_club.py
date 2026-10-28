#!/usr/bin/env python3
"""
gold_rich_club.py — the 'rich_club' criterion of connectome_audit_gold.py,
on its own.

Unnormalized rich-club coefficient (Zhou & Mondragon 2004; Colizza et al. 2006)

    phi(k) = 2 E_{>k} / (N_{>k} (N_{>k} - 1))    (nx.rich_club_coefficient, normalized=False)

where N_{>k} is the number of nodes of degree > k and E_{>k} the number of
edges among them, computed on the WHOLE graph (not its GCC), at every degree
threshold k, and normalised by a degree-preserving null ensemble, as the gold
audit does it:

    curves  = the n_null main nulls  _generate_null(layered, seed + 1000 + i), i < n_null,
              then n_null_rc - n_null extra nulls  _generate_null(layered, seed + 5000 + j)
              -> max(n_null, n_null_rc) curves, each on the whole null graph
    k grid  = thresholds present in the real curve AND in every null curve
    rho(k)  = phi(k) / mean(phi_null(k))
    p(k)    = (#{phi_null(k) >= phi(k)} + 1) / (n_curves + 1)
    k is significant if rho(k) > 1 and p(k) < 0.05
    verdict = 'pass' if the longest run of CONSECUTIVE significant k is >= 3
              (van den Heuvel & Sporns 2011), else 'fail'
              ('not_tested' with --no-null, as in the audit)

This is the audit's UNFLOORED count: every k in the grid is tested, however few
nodes are left above it.  richclub_floored.py is the floored count
(>= 20 nodes above k) that richclub_other_graphs.py reports.

Self-contained: needs numpy and networkx only, and does not import
connectome_audit_gold.py.  The loading, null-model and _rich_club functions
below are copied from it unchanged, so with the same --graph, --n-null,
--n-null-rc and --seed the result is the same as the 'rich_club' entry of the
full audit and its 'rich_club' verdict.  The nulls are generated in parallel
here (each has its own seed, so the graphs are identical to the serial ones,
and the curves are kept in the audit's order); the full audit generates them
serially.  _rich_club_table is NOT in the audit: it repeats _rich_club's per-k
loop to list WHICH k are significant (the audit keeps only the counts), and its
count is checked against _rich_club's.

Usage:
    python gold_rich_club.py --graph G.npz
    python gold_rich_club.py --graph G.npz --n-null 100 --n-null-rc 1000 --seed 42 --json rc.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
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
# Null model — tripartite-aware degree-preserving rewiring
# (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _randomize_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degree-preserving 2×2 swap randomization for a layer-pair mask."""
    mask = (mask > 0).astype(np.uint8)
    n_rows, n_cols = mask.shape
    rows, cols = np.where(mask)
    m = len(rows)
    if m < 2:
        return mask.copy()

    edges = [[int(r), int(c)] for r, c in zip(rows, cols)]
    row_to_cols: List[set] = [set() for _ in range(n_rows)]
    for r, c in edges:
        row_to_cols[r].add(c)

    target = max(10, m * 10)
    max_tries = target * 25
    success = tries = 0
    while success < target and tries < max_tries:
        tries += 1
        i, j = int(rng.integers(0, m)), int(rng.integers(0, m))
        if i == j:
            continue
        r1, c1 = edges[i]
        r2, c2 = edges[j]
        if r1 == r2 or c1 == c2:
            continue
        if c2 in row_to_cols[r1] or c1 in row_to_cols[r2]:
            continue
        row_to_cols[r1].remove(c1); row_to_cols[r2].remove(c2)
        row_to_cols[r1].add(c2);    row_to_cols[r2].add(c1)
        edges[i] = [r1, c2]; edges[j] = [r2, c1]
        success += 1

    out = np.zeros_like(mask, dtype=np.uint8)
    for r, c in edges:
        out[r, c] = 1
    return out


def _generate_null_edgelist(G_real: nx.Graph, rng: np.random.Generator,
                            n_swaps_factor: int = 10) -> nx.Graph:
    """Degree-preserving rewiring on a plain graph (for χ>3 or edge-list graphs)."""
    G = G_real.copy()
    edges = list(G.edges())
    m = len(edges)
    target = max(10, m * n_swaps_factor)
    max_tries = target * 25
    success = tries = 0
    while success < target and tries < max_tries:
        tries += 1
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m))
        if i == j:
            continue
        u1, v1 = edges[i]
        u2, v2 = edges[j]
        if len({u1, v1, u2, v2}) < 4:
            continue
        # Try both swap orientations (Maslov-Sneppen)
        if rng.random() < 0.5:
            a1, b1, a2, b2 = u1, u2, v1, v2
        else:
            a1, b1, a2, b2 = u1, v2, v1, u2
        if not G.has_edge(a1, b1) and not G.has_edge(a2, b2):
            G.remove_edge(u1, v1); G.remove_edge(u2, v2)
            G.add_edge(a1, b1); G.add_edge(a2, b2)
            edges[i] = (a1, b1); edges[j] = (a2, b2)
            success += 1
    return G


def _use_generic_null(layered: LayeredGraph) -> bool:
    """Decide null model based on graph semantics, not file format.
    Use tripartite-aware rewiring only if masks carry real structure
    (non-trivial, matching the graph's edges). Otherwise use generic.

    FIX: previously returned True for every edge-list file, so the SAME graph
    got a different null (and different verdicts) depending on file format.
    The mask/edge match promised above is now actually checked (once, cached)."""
    cached = getattr(layered, '_generic_null', None)
    if cached is not None:
        return cached
    generic = False
    # Check if masks are trivially small (placeholder)
    for m in layered.masks.values():
        if m.shape[0] <= 1 or m.shape[1] <= 1:
            generic = True
    if not generic:
        n1, n2, n3 = layered.sizes
        Gm = _graph_from_masks(n1, n2, n3, layered.masks['12'],
                               layered.masks['13'], layered.masks['23'])
        same = (Gm.number_of_nodes() == layered.G.number_of_nodes() and
                {frozenset(e) for e in Gm.edges()} ==
                {frozenset(e) for e in layered.G.edges()})
        generic = not same
    layered._generic_null = generic
    return generic


def _generate_null(layered: LayeredGraph, seed: int, retries: int = 8) -> nx.Graph:
    n1, n2, n3 = layered.sizes
    rng = np.random.default_rng(seed)
    best_G, best_gcc = None, -1

    if _use_generic_null(layered):
        # Generic degree-preserving rewiring (Maslov-Sneppen)
        for _ in range(retries):
            G = _generate_null_edgelist(layered.G, rng)
            if nx.is_connected(G):
                return G
            gcc = len(max(nx.connected_components(G), key=len))
            if gcc > best_gcc:
                best_G, best_gcc = G, gcc
        return best_G

    for _ in range(retries):
        m12 = _randomize_mask(layered.masks['12'], rng)
        m13 = _randomize_mask(layered.masks['13'], rng)
        m23 = _randomize_mask(layered.masks['23'], rng)
        G = _graph_from_masks(n1, n2, n3, m12, m13, m23)
        if nx.is_connected(G):
            return G
        gcc = len(max(nx.connected_components(G), key=len))
        if gcc > best_gcc:
            best_G, best_gcc = G, gcc
    return best_G


# ═══════════════════════════════════════════════════════════════════════════
# Rich-club, ensemble-normalized  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _rich_club(G: nx.Graph, nulls: Sequence[nx.Graph] = None,
               quantile: float = 0.75,
               null_curves: list = None) -> Dict[str, object]:
    real = nx.rich_club_coefficient(G, normalized=False)
    if null_curves is None:
        null_curves = [nx.rich_club_coefficient(R, normalized=False) for R in nulls]

    common_k = set(real.keys())
    for c in null_curves:
        common_k &= set(c.keys())
    common_k = sorted(common_k)
    if not common_k:
        return {'available': False}

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    k0 = int(np.percentile(degrees, 100 * quantile))

    normalized = {}
    high_vals = []
    high_pvals = []
    # Per-k rich-club test: Phi_norm(k) > 1 with p < 0.05
    # (Colizza et al. 2006, van den Heuvel & Sporns 2011, Riedel et al. 2022)
    n_k_significant = 0
    sig_k_set = set()
    for k in common_k:
        nv = [c[k] for c in null_curves if k in c and np.isfinite(c[k])]
        if not nv:
            continue
        mu = float(np.mean(nv))
        if mu <= 0:
            continue
        val = float(real[k] / mu)
        normalized[int(k)] = val
        pval = float((np.sum(np.array(nv) >= real[k]) + 1) / (len(nv) + 1))
        if val > 1.0 and pval < 0.05:
            n_k_significant += 1
            sig_k_set.add(int(k))
        if k >= k0:
            high_vals.append(val)
            high_pvals.append(pval)

    # Longest consecutive run of significant k-values
    longest_run = 0
    if sig_k_set:
        sorted_sig = sorted(sig_k_set)
        run = 1
        for i in range(1, len(sorted_sig)):
            if sorted_sig[i] == sorted_sig[i - 1] + 1:
                run += 1
            else:
                longest_run = max(longest_run, run)
                run = 1
        longest_run = max(longest_run, run)

    if not normalized:
        return {'available': False}
    if not high_vals:
        tail = sorted(normalized.keys())[-min(5, len(normalized)):]
        high_vals = [normalized[k] for k in tail]
        high_pvals = [0.5] * len(high_vals)  # no meaningful p-value

    # Summary stats for reporting
    n_sig = sum(1 for p in high_pvals if p < 0.05)

    return {
        'available': True,
        'n_k_significant': n_k_significant,
        'longest_consecutive_run': longest_run,
        'high_k_mean_norm': float(np.mean(high_vals)),
        'high_k_max_norm': float(np.max(high_vals)),
        'high_k_n_tested': len(high_vals),
        'high_k_n_significant': n_sig,
        'high_k_frac_significant': float(n_sig / len(high_vals)) if high_vals else 0.0,
        'high_k_max_pval': float(max(high_pvals)) if high_pvals else float('nan'),
    }


# ═══════════════════════════════════════════════════════════════════════════
# The criterion
# ═══════════════════════════════════════════════════════════════════════════


_LAYERED = None  # set before Pool creation; workers inherit via fork


def _null_rc_curve(args):
    """Worker: build null `null_seed` exactly as the gold audit does and return
    its unnormalized rich-club curve, taken on the whole null graph (as in the audit)."""
    i, null_seed = args
    t_n = time.time()
    R = _generate_null(_LAYERED, seed=null_seed)
    return i, nx.rich_club_coefficient(R, normalized=False), nx.is_connected(R), time.time() - t_n


def _rich_club_table(G: nx.Graph, null_curves: list) -> List[Dict[str, object]]:
    """Per-k detail (not in the audit).  The same k grid, null mean, p-value and
    test as the loop in _rich_club; the audit keeps only the counts."""
    real = nx.rich_club_coefficient(G, normalized=False)
    common_k = set(real.keys())
    for c in null_curves:
        common_k &= set(c.keys())
    deg = np.array([d for _, d in G.degree()])
    rows = []
    for k in sorted(common_k):
        nv = [c[k] for c in null_curves if k in c and np.isfinite(c[k])]
        if not nv:
            continue
        mu = float(np.mean(nv))
        if mu <= 0:
            continue
        val = float(real[k] / mu)
        pval = float((np.sum(np.array(nv) >= real[k]) + 1) / (len(nv) + 1))
        rows.append({'k': int(k), 'n_above_k': int((deg > k).sum()),
                     'phi': float(real[k]), 'phi_null_mean': mu,
                     'phi_null_std': float(np.std(nv, ddof=1)) if len(nv) > 1 else 0.0,
                     'rho': val, 'p': pval,
                     'significant': bool(val > 1.0 and pval < 0.05)})
    return rows


def _runs(ks: List[int]) -> List[Tuple[int, int]]:
    """Consecutive runs [(first, last), ...] of a sorted list of integers."""
    out = []
    for k in ks:
        if out and k == out[-1][1] + 1:
            out[-1] = (out[-1][0], k)
        else:
            out.append((k, k))
    return out


def rich_club(layered: LayeredGraph, n_null: int = 100, n_null_rc: int = 1000,
              seed: int = 42, workers: int = 0) -> Dict[str, object]:
    global _LAYERED
    G = layered.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])
    print(f"  Nodes: {n:,}   Edges: {m:,}   max degree: {int(deg.max())}   "
          f"(rich club measured on the whole graph, as in the audit)")

    # The audit's ensemble: one curve per main null (seed+1000+i), then extra
    # nulls (seed+5000+j) up to n_null_rc curves in all.
    seeds = [seed + 1000 + i for i in range(n_null)] if n_null > 0 else []
    if n_null_rc > len(seeds):
        seeds += [seed + 5000 + j for j in range(n_null_rc - len(seeds))]

    if not seeds:
        # No-null mode, as in the gold audit: unnormalized phi(k) above the
        # 75th degree percentile; no test, verdict 'not_tested'.
        real_rc = nx.rich_club_coefficient(G, normalized=False)
        degrees = np.array([d for _, d in G.degree()], dtype=float)
        k0 = int(np.percentile(degrees, 75))
        high_vals = [v for k, v in real_rc.items() if k >= k0 and np.isfinite(v)]
        rc = {
            'available': True,
            'n_null_rc': 0,
            'high_k_mean_norm': float(np.mean(high_vals)) if high_vals else 0.0,
            'high_k_max_norm': float(max(high_vals)) if high_vals else 0.0,
            'note': 'unnormalized (no null ensemble)',
        }
        return {'graph': {'n_nodes': n, 'n_edges': m}, 'n_null': 0, 'n_null_rc': 0,
                'null_model_type': 'none', 'seed': seed, 'rich_club': rc,
                'per_k': [], 'significant_k': [], 'significant_runs': [],
                'rich_club_verdict': 'not_tested'}

    null_model_type = ("generic_degree_preserving" if _use_generic_null(layered)
                       else "tripartite_degree_preserving")  # cached before the fork
    n_curves = len(seeds)
    if workers <= 0:
        workers = min(6, n_curves, max(1, os.cpu_count() - 2))
    n_main = min(n_null, n_curves) if n_null > 0 else 0
    print(f"\n  {n_curves} null curves ({null_model_type}), {workers} workers:")
    if n_main:
        print(f"    {n_main} main nulls, seeds {seed + 1000}..{seed + 1000 + n_main - 1}")
    if n_curves > n_main:
        print(f"    {n_curves - n_main} extra nulls, seeds {seed + 5000}..{seed + 5000 + n_curves - n_main - 1}")

    _LAYERED = layered
    curves = [None] * n_curves
    n_disconnected = 0
    done = 0
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        for i, c, is_conn, elapsed in pool.imap_unordered(
                _null_rc_curve, list(enumerate(seeds)), chunksize=4):
            curves[i] = c
            n_disconnected += (not is_conn)
            done += 1
            if done % 50 == 0 or done == n_curves:
                print(f"    {done:>5d}/{n_curves} curves  ({time.time() - t0:.0f}s, last {elapsed:.2f}s)")
    _LAYERED = None
    print(f"  null ensemble done ({time.time() - t0:.1f}s)")
    if n_disconnected > 0:
        print(f"  note: {n_disconnected}/{n_curves} nulls disconnected (curves are on the whole graph, as in the audit)")

    rc = _rich_club(G, null_curves=curves)
    rc['n_null_rc'] = len(curves)
    verdict = 'not_tested' if rc.get('n_null_rc', 0) == 0 else (
        'pass' if (rc.get('available', False) and
                   rc.get('longest_consecutive_run', 0) >= 3) else 'fail')

    table = _rich_club_table(G, curves)
    sig_k = [r['k'] for r in table if r['significant']]
    assert len(sig_k) == rc.get('n_k_significant', 0), "per-k table disagrees with _rich_club"
    runs = _runs(sig_k)
    assert max((b - a + 1 for a, b in runs), default=0) == rc.get('longest_consecutive_run', 0)
    return {'graph': {'n_nodes': n, 'n_edges': m, 'max_degree': int(deg.max())},
            'n_null': n_null, 'n_null_rc': n_null_rc, 'n_curves': n_curves,
            'null_model_type': null_model_type, 'seed': seed,
            'n_null_disconnected': n_disconnected,
            'rich_club': rc, 'per_k': table,
            'significant_k': sig_k, 'significant_runs': runs,
            'rich_club_verdict': verdict}


def main():
    ap = argparse.ArgumentParser(description="rich_club, the eighth of the gold audit's nine criteria")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--n-null', type=int, default=100,
                    help='Main nulls of the audit (their curves are the first ones used)')
    ap.add_argument('--n-null-rc', type=int, default=1000,
                    help='Null curves for rich-club in all (vdH 2011 uses 1000)')
    ap.add_argument('--no-null', action='store_true', help='Skip the null ensemble (verdict is not_tested)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0,
                    help='Parallel null workers (default: min(6, cpus-2))')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    res = rich_club(layered,
                    n_null=0 if args.no_null else args.n_null,
                    n_null_rc=0 if args.no_null else args.n_null_rc,
                    seed=args.seed, workers=args.workers)

    rc = res['rich_club']
    print("\n" + "=" * 72)
    print("RICH_CLUB (ensemble-normalized, unfloored)")
    print("=" * 72)
    if res.get('n_curves', 0):
        print(f"  {'n_null_rc':>24s}: {rc['n_null_rc']}   (p resolution 1/{rc['n_null_rc'] + 1})")
        for key in ['n_k_significant', 'longest_consecutive_run', 'high_k_mean_norm',
                    'high_k_max_norm', 'high_k_n_tested', 'high_k_n_significant',
                    'high_k_frac_significant', 'high_k_max_pval']:
            if key in rc:
                v = rc[key]
                print(f"  {key:>24s}: {v:.6g}" if isinstance(v, float) else f"  {key:>24s}: {v}")
        print(f"  {'k grid':>24s}: {res['per_k'][0]['k'] if res['per_k'] else '-'}"
              f"..{res['per_k'][-1]['k'] if res['per_k'] else '-'}  ({len(res['per_k'])} thresholds tested)")
        print(f"  {'significant k (runs)':>24s}: "
              + (", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in res['significant_runs']) or 'none'))
    else:
        print(f"  {'high_k_mean (unnorm.)':>24s}: {rc['high_k_mean_norm']:.6g}")
        print(f"  {'high_k_max (unnorm.)':>24s}: {rc['high_k_max_norm']:.6g}")
    print(f"  rich_club: {res['rich_club_verdict'].upper()}   (pass if longest run of consecutive significant k >= 3)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
