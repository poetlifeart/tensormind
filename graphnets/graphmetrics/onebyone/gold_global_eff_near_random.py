#!/usr/bin/env python3
"""
gold_global_eff_near_random.py — the 'global_eff_near_random' criterion of
connectome_audit_gold.py, on its own.

Exact global efficiency (_exact_geff: E_glob = sum_{i!=j} 1/d_ij / (N(N-1)),
all pairs, scipy sparse Dijkstra, unweighted; unreachable pairs contribute 0;
Latora & Marchiori 2001, Eq. 1) of the graph's GCC, against the same
degree-preserving null ensemble the gold audit uses:

    null i  = _generate_null(layered, seed=seed + 1000 + i),  i = 0 .. n_null-1
    ratio   = E_real / mean(E_null)          (each null measured on its own GCC)
    p       = (#{null <= real} + 1) / (n_null + 1)      (reported, not used)
    verdict = 'pass' if 0.8 <= ratio < 1.0 else 'fail'

With --no-null the null mean is replaced, as in the gold audit, by
1 / l_rand, where l_rand is the Fronczak, Fronczak & Holyst (2004) path
length of an Erdos-Renyi graph with the same n and m:
l = (ln n - 0.5772) / ln<k> + 1/2,  <k> = n p,  p = 2m/(n(n-1))
(and by p itself if l_rand is not finite and positive).

Self-contained: needs numpy, scipy and networkx only, and does not import
connectome_audit_gold.py.  The loading, null-model, statistics and
efficiency functions below are copied from it unchanged, so with the same
--graph, --n-null and --seed the result is the same as the 'global_eff'
entry of the full audit's 'comparisons' and its 'global_eff_near_random'
verdict.  The nulls are generated in parallel here (each has its own seed,
so the graphs are identical to the serial ones); the full audit generates
them serially.

Usage:
    python gold_global_eff_near_random.py --graph G.npz
    python gold_global_eff_near_random.py --graph G.npz --n-null 100 --seed 42 --json geff.json
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


def _exact_geff(G: nx.Graph) -> float:
    """Exact global efficiency: E_glob = (1/(N(N-1))) * sum(1/d_ij).
    Unreachable pairs contribute 0 (Latora & Marchiori 2001)."""
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
    with np.errstate(divide='ignore'):
        inv_dist = 1.0 / dist
    inv_dist[~np.isfinite(inv_dist)] = 0.0
    np.fill_diagonal(inv_dist, 0.0)
    return float(inv_dist.sum() / (n * (n - 1)))


# ═══════════════════════════════════════════════════════════════════════════
# The criterion
# ═══════════════════════════════════════════════════════════════════════════


_LAYERED = None  # set before Pool creation; workers inherit via fork


def _null_geff(args):
    """Worker: build null i exactly as the gold audit does, return its GCC global efficiency."""
    i, null_seed = args
    t_n = time.time()
    R = _generate_null(_LAYERED, seed=null_seed)
    is_conn = nx.is_connected(R)
    Rcc = _gcc(R)
    return i, _exact_geff(Rcc), is_conn, Rcc.number_of_nodes(), time.time() - t_n


def global_eff_near_random(layered: LayeredGraph, n_null: int = 100, seed: int = 42,
                           workers: int = 0) -> Dict[str, object]:
    global _LAYERED
    G = layered.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    Gcc = _gcc(G)
    print(f"  Nodes: {n:,}   Edges: {m:,}   GCC: {Gcc.number_of_nodes():,} nodes")

    real = _exact_geff(Gcc)
    print(f"  real global efficiency (exact, all pairs, GCC) = {real:.6g}")

    def verdict(ratio):
        # Iturria-Medina et al. 2008: real/random global efficiency just under 1.
        return 'pass' if 0.8 <= ratio < 1.0 else 'fail'

    if n_null <= 0:
        # No-null mode, as in the gold audit: 1 / Fronczak-Fronczak-Holyst ER path length.
        p = 2.0 * m / (n * (n - 1))
        l_rand = ((np.log(n) - 0.5772) / np.log(n * p) + 0.5) if n * p > 1 else float('inf')
        e_rand = 1.0 / l_rand if l_rand > 0 and np.isfinite(l_rand) else p
        ratio = float(real / e_rand) if e_rand != 0 else float('nan')
        comp = {'real': real, 'null_mean': e_rand, 'null_std': 0.0,
                'ratio': ratio, 'z': float('nan'), 'p': float('nan')}
        return {'graph': {'n_nodes': n, 'n_edges': m, 'gcc': Gcc.number_of_nodes()},
                'n_null': 0, 'null_model_type': 'none', 'seed': seed,
                'global_eff': comp, 'null_values': [],
                'global_eff_near_random': verdict(ratio)}

    null_model_type = ("generic_degree_preserving" if _use_generic_null(layered)
                       else "tripartite_degree_preserving")  # cached before the fork
    if workers <= 0:
        workers = min(6, n_null, max(1, os.cpu_count() - 2))
    print(f"\n  {n_null} nulls ({null_model_type}), seeds {seed + 1000}..{seed + 1000 + n_null - 1}, "
          f"{workers} workers")

    _LAYERED = layered
    vals = [None] * n_null
    n_disconnected = 0
    done = 0
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        for i, E, is_conn, rn, elapsed in pool.imap_unordered(
                _null_geff, [(i, seed + 1000 + i) for i in range(n_null)]):
            vals[i] = E
            n_disconnected += (not is_conn)
            done += 1
            print(f"    null {i+1:>4d}  connected={is_conn}  GCC={rn:,}  E_glob={E:.6g}"
                  f"   [{done}/{n_null}, {elapsed:.1f}s]")
    _LAYERED = None
    print(f"  null ensemble done ({time.time() - t0:.1f}s)")
    if n_disconnected > 0:
        print(f"  WARNING: {n_disconnected}/{n_null} nulls disconnected — efficiency computed on GCC")

    mu, sd = _mean_std(vals)
    ratio = float(real / mu) if mu != 0 else float('nan')
    comp = {'real': real, 'null_mean': mu, 'null_std': sd,
            'ratio': ratio,
            'z': _z(real, vals),
            'p': _pval(real, vals, higher=False)}
    return {'graph': {'n_nodes': n, 'n_edges': m, 'gcc': Gcc.number_of_nodes()},
            'n_null': n_null, 'null_model_type': null_model_type, 'seed': seed,
            'n_null_disconnected': n_disconnected,
            'global_eff': comp, 'null_values': vals,
            'global_eff_near_random': verdict(ratio)}


def main():
    ap = argparse.ArgumentParser(description="global_eff_near_random, the fifth of the gold audit's nine criteria")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--n-null', type=int, default=100)
    ap.add_argument('--no-null', action='store_true',
                    help='Skip the null ensemble (ratio against 1 / the Fronczak ER path length)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0,
                    help='Parallel null workers (default: as the gold audit, min(6, cpus-2))')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    res = global_eff_near_random(layered, n_null=0 if args.no_null else args.n_null,
                                 seed=args.seed, workers=args.workers)

    d = res['global_eff']
    print("\n" + "=" * 72)
    print("GLOBAL_EFF_NEAR_RANDOM")
    print("=" * 72)
    print(f"  {'global_eff':>16s} │ real={d['real']:.6g}  null={d['null_mean']:.6g}±{d['null_std']:.3g}"
          f"  ratio={d['ratio']:.4g}  z={d['z']:.3g}  p={d['p']:.3g}")
    print(f"  global_eff_near_random: {res['global_eff_near_random'].upper()}   (pass if 0.8 <= ratio < 1.0)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
