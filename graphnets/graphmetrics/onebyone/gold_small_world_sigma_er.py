#!/usr/bin/env python3
"""
gold_small_world_sigma_er.py — the 'small_world_sigma_er' criterion of
connectome_audit_gold.py, on its own, with the unscored sigma_dp beside it.

Humphries & Gurney (2008) small-world-ness S^Delta: the TRANSITIVITY ratio
over the path-length ratio, both against random graphs.  Real values are
measured on the graph's GCC: C = nx.transitivity, L = _exact_apl (all pairs,
scipy sparse Dijkstra, unweighted).

  sigma_er (scored).  Null = n_null Erdos-Renyi G(n,m) graphs with the GCC's
  n and m, seeds drawn in order from np.random.default_rng(seed + 7777);
  each measured on its own GCC.
      sigma_er = (C / mean C_ER) / (L / mean L_ER)
      verdict  = 'pass' if sigma_er > 1 else 'fail'

  sigma_dp (not scored).  The same definition with the audit's
  degree-preserving null ensemble, null i = _generate_null(layered,
  seed = seed + 1000 + i), each measured on its own GCC.
      sigma_dp = comp['transitivity']['ratio'] / comp['apl']['ratio']

With --no-null both are replaced, as in the gold audit, by one analytic
estimate: C_rand = p = 2m/(n(n-1)) and the Fronczak, Fronczak & Holyst
(2004) path length l = (ln n - 0.5772)/ln<k> + 1/2, <k> = n p, with n, m of
the whole graph; sigma_er = sigma_dp.

Self-contained: needs numpy, scipy and networkx only, and does not import
connectome_audit_gold.py.  The loading, null-model, statistics and APL
functions below are copied from it unchanged, so with the same --graph,
--n-null and --seed the result is the same as the full audit's
'transitivity' and 'apl' entries of 'comparisons', its 'sigma' (= sigma_dp),
its 'sigma_er' and its 'small_world_sigma_er' verdict.  Both null ensembles
are generated in parallel here (each graph has its own seed, and results are
reduced in index order, so the numbers are identical to the serial ones);
the full audit generates them serially.

Usage:
    python gold_small_world_sigma_er.py --graph G.npz
    python gold_small_world_sigma_er.py --graph G.npz --n-null 100 --seed 42 --json sw.json
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
import scipy.sparse
import scipy.sparse.csgraph
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
    (non-trivial, matching the graph's edges). Otherwise use generic."""
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
# Utilities  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _gcc(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan')
    return (float(a.mean()), float(a.std(ddof=1))) if a.size > 1 else (float(a[0]), 0.0)


def _z(real: float, nulls: Sequence[float]) -> float:
    mu, sd = _mean_std(nulls)
    return float((real - mu) / sd) if np.isfinite(mu) and sd > 0 else float('nan')


def _pval(real: float, nulls: Sequence[float], higher: bool = True) -> float:
    a = np.asarray(nulls, dtype=float)
    if a.size == 0:
        return float('nan')
    count = np.sum(a >= real) if higher else np.sum(a <= real)
    return float((count + 1) / (a.size + 1))


def _exact_apl(G: nx.Graph) -> float:
    """Exact average path length on GCC via scipy sparse APSP."""
    n = G.number_of_nodes()
    if n <= 1:
        return float('nan')
    nodes = sorted(G.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    rs, cs = [], []
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        rs.extend([i, j]); cs.extend([j, i])
    adj = scipy.sparse.csr_matrix(
        (np.ones(len(rs), dtype=np.float32), (rs, cs)), shape=(n, n))
    dist = scipy.sparse.csgraph.shortest_path(adj, method='D', directed=False, unweighted=True)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    return float(finite.mean()) if len(finite) > 0 else float('nan')


# ═══════════════════════════════════════════════════════════════════════════
# The criterion
# ═══════════════════════════════════════════════════════════════════════════


_LAYERED = None  # set before Pool creation; workers inherit via fork


def _null_trans_apl(args):
    """Worker: build degree-preserving null i exactly as the gold audit does,
    return the transitivity and exact APL of its GCC."""
    i, null_seed = args
    t_n = time.time()
    R = _generate_null(_LAYERED, seed=null_seed)
    is_conn = nx.is_connected(R)
    Rcc = _gcc(R)
    return (i, float(nx.transitivity(Rcc)), _exact_apl(Rcc), is_conn,
            Rcc.number_of_nodes(), time.time() - t_n)


def _er_trans_apl(args):
    """Worker: ER G(n,m) graph i, with the seed the gold audit draws for it,
    return the transitivity and exact APL of its GCC."""
    i, n_gcc, m_gcc, er_seed = args
    t_n = time.time()
    R_er = nx.gnm_random_graph(n_gcc, m_gcc, seed=er_seed)
    R_er_gcc = _gcc(R_er)
    return (i, float(nx.transitivity(R_er_gcc)), _exact_apl(R_er_gcc),
            R_er_gcc.number_of_nodes(), time.time() - t_n)


def _comparison(real: float, vals: Sequence[float], higher: bool) -> Dict[str, float]:
    """One entry of the gold audit's 'comparisons' table."""
    mu, sd = _mean_std(vals)
    return {'real': real, 'null_mean': mu, 'null_std': sd,
            'ratio': float(real / mu) if mu != 0 else float('nan'),
            'z': _z(real, vals),
            'p': _pval(real, vals, higher=higher)}


def _sigma(c_r: float, l_r: float) -> float:
    return float(c_r / l_r) if np.isfinite(c_r) and np.isfinite(l_r) and l_r != 0 else float('nan')


def _sw_verdict(s: float) -> str:
    return 'pass' if (np.isfinite(s) and s > 1.0) else 'fail'


def small_world_sigma_er(layered: LayeredGraph, n_null: int = 100, seed: int = 42,
                         workers: int = 0) -> Dict[str, object]:
    global _LAYERED
    G = layered.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    Gcc = _gcc(G)
    n_gcc = Gcc.number_of_nodes()
    m_gcc = Gcc.number_of_edges()
    print(f"  Nodes: {n:,}   Edges: {m:,}   GCC: {n_gcc:,} nodes, {m_gcc:,} edges")

    real_t = float(nx.transitivity(Gcc))
    real_l = _exact_apl(Gcc)
    print(f"  real transitivity (GCC) = {real_t:.6g}")
    print(f"  real APL (exact, all pairs, GCC) = {real_l:.6g}")

    graph_info = {'n_nodes': n, 'n_edges': m, 'gcc': n_gcc, 'gcc_edges': m_gcc}

    if n_null <= 0:
        # No-null mode, as in the gold audit: one ER analytic estimate for both sigmas.
        p = 2.0 * m / (n * (n - 1))
        c_rand = p
        l_rand = ((np.log(n) - 0.5772) / np.log(n * p) + 0.5) if n * p > 1 else float('inf')
        comp_t = {'real': real_t, 'null_mean': c_rand, 'null_std': 0.0,
                  'ratio': float(real_t / c_rand) if c_rand != 0 else float('nan'),
                  'z': float('nan'), 'p': float('nan')}
        comp_l = {'real': real_l, 'null_mean': l_rand, 'null_std': 0.0,
                  'ratio': float(real_l / l_rand) if l_rand != 0 else float('nan'),
                  'z': float('nan'), 'p': float('nan')}
        sigma = _sigma(comp_t['ratio'], comp_l['ratio'])
        return {'graph': graph_info, 'n_null': 0, 'null_model_type': 'none', 'seed': seed,
                'transitivity': comp_t, 'apl': comp_l,
                'sigma': sigma, 'sigma_er': sigma, 'er': None,
                'small_world_sigma_er': _sw_verdict(sigma),
                'small_world_sigma_dp_unscored': _sw_verdict(sigma)}

    if workers <= 0:
        workers = min(6, n_null, max(1, os.cpu_count() - 2))

    # ── degree-preserving null ensemble (sigma_dp) ──
    null_model_type = ("generic_degree_preserving" if _use_generic_null(layered)
                       else "tripartite_degree_preserving")  # cached before the fork
    print(f"\n  {n_null} degree-preserving nulls ({null_model_type}), seeds "
          f"{seed + 1000}..{seed + 1000 + n_null - 1}, {workers} workers")
    _LAYERED = layered
    t_vals = [None] * n_null
    l_vals = [None] * n_null
    n_disconnected = 0
    done = 0
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        for i, t, L, is_conn, rn, elapsed in pool.imap_unordered(
                _null_trans_apl, [(i, seed + 1000 + i) for i in range(n_null)]):
            t_vals[i] = t
            l_vals[i] = L
            n_disconnected += (not is_conn)
            done += 1
            print(f"    null {i+1:>4d}  connected={is_conn}  GCC={rn:,}  transitivity={t:.6g}"
                  f"  APL={L:.6g}   [{done}/{n_null}, {elapsed:.1f}s]")
    _LAYERED = None
    print(f"  degree-preserving ensemble done ({time.time() - t0:.1f}s)")
    if n_disconnected > 0:
        print(f"  WARNING: {n_disconnected}/{n_null} nulls disconnected — metrics computed on GCC")

    comp_t = _comparison(real_t, t_vals, higher=True)
    comp_l = _comparison(real_l, l_vals, higher=True)
    sigma = _sigma(comp_t['ratio'], comp_l['ratio'])

    # ── ER G(n,m) ensemble (sigma_er, H&G S^Delta) ──
    er_rng = np.random.default_rng(seed + 7777)
    er_seeds = [int(er_rng.integers(0, 2**31)) for _ in range(n_null)]
    print(f"\n  {n_null} ER G(n={n_gcc:,}, m={m_gcc:,}) graphs, seeds from "
          f"default_rng({seed + 7777}), {workers} workers")
    er_trans = [None] * n_null
    er_apl = [None] * n_null
    er_gcc_sizes = [None] * n_null
    done = 0
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        for i, t, L, rn, elapsed in pool.imap_unordered(
                _er_trans_apl, [(i, n_gcc, m_gcc, er_seeds[i]) for i in range(n_null)]):
            er_trans[i] = t
            er_apl[i] = L
            er_gcc_sizes[i] = rn
            done += 1
            print(f"    ER {i+1:>4d}  GCC={rn:,}  transitivity={t:.6g}  APL={L:.6g}"
                  f"   [{done}/{n_null}, {elapsed:.1f}s]")
    print(f"  ER ensemble done ({time.time() - t0:.1f}s)")
    n_er_disconnected = sum(1 for rn in er_gcc_sizes if rn < n_gcc)
    if n_er_disconnected > 0:
        print(f"  NOTE: {n_er_disconnected}/{n_null} ER graphs disconnected — measured on their GCC")

    er_trans_mean = float(np.mean(er_trans))
    er_apl_mean = float(np.mean(er_apl))
    c_r_er = float(real_t / er_trans_mean) if er_trans_mean > 0 else float('nan')
    l_r_er = float(real_l / er_apl_mean) if er_apl_mean > 0 else float('nan')
    sigma_er = _sigma(c_r_er, l_r_er)

    er = {'n': n_gcc, 'm': m_gcc, 'trans_mean': er_trans_mean, 'apl_mean': er_apl_mean,
          'trans_ratio': c_r_er, 'apl_ratio': l_r_er,
          'n_disconnected': n_er_disconnected,
          'trans_values': er_trans, 'apl_values': er_apl, 'gcc_sizes': er_gcc_sizes}
    return {'graph': graph_info, 'n_null': n_null, 'null_model_type': null_model_type,
            'seed': seed, 'n_null_disconnected': n_disconnected,
            'transitivity': comp_t, 'apl': comp_l,
            'null_trans_values': t_vals, 'null_apl_values': l_vals,
            'sigma': sigma, 'sigma_er': sigma_er, 'er': er,
            'small_world_sigma_er': _sw_verdict(sigma_er),
            'small_world_sigma_dp_unscored': _sw_verdict(sigma)}


def main():
    ap = argparse.ArgumentParser(description="small_world_sigma_er, the third of the gold audit's "
                                             "nine criteria (and the unscored sigma_dp)")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--n-null', type=int, default=100,
                    help='Degree-preserving nulls AND ER graphs (the audit uses the same count for both)')
    ap.add_argument('--no-null', action='store_true',
                    help='Skip both ensembles (analytic ER estimates, as the audit does)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0,
                    help='Parallel workers (default: as the gold audit, min(6, cpus-2))')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    res = small_world_sigma_er(layered, n_null=0 if args.no_null else args.n_null,
                               seed=args.seed, workers=args.workers)

    print("\n" + "=" * 72)
    print("SMALL_WORLD_SIGMA_ER  (+ unscored sigma_dp)")
    print("=" * 72)
    for name in ['transitivity', 'apl']:
        d = res[name]
        print(f"  {name:>16s} │ real={d['real']:.6g}  null={d['null_mean']:.6g}±{d['null_std']:.3g}"
              f"  ratio={d['ratio']:.4g}  z={d['z']:.3g}  p={d['p']:.3g}")
    er = res['er']
    if er is not None:
        print(f"  {'ER G(n,m)':>16s} │ transitivity={er['trans_mean']:.6g}  APL={er['apl_mean']:.6g}"
              f"  C/C_ER={er['trans_ratio']:.4g}  L/L_ER={er['apl_ratio']:.4g}")
    print(f"  {'sigma (dp)':>16s} │ {res['sigma']:.4g}")
    print(f"  {'sigma (ER)':>16s} │ {res['sigma_er']:.4g}")
    print(f"  small_world_sigma_er: {res['small_world_sigma_er'].upper()}   (pass if sigma_er > 1)")
    print(f"  sigma_dp > 1 (not scored): {res['small_world_sigma_dp_unscored'].upper()}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
