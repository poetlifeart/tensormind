#!/usr/bin/env python3
"""
richclub_floored.py — the rich-club threshold counts of richclub_other_graphs.py
(floored at MINNODES = 20 nodes), on their own.

Unnormalized rich-club coefficient (Zhou & Mondragon 2004; Colizza et al. 2006)

    phi(k) = 2 E_{>k} / (N_{>k} (N_{>k} - 1))    (nx.rich_club_coefficient, normalized=False)

of the whole graph at every degree threshold k of the real graph, normalised by
a degree-preserving null ensemble, exactly as richclub_other_graphs.py does it:

    nulls   = _generate_null(layered, SEED + 5000 + i), i = 0 .. N_NULL-1   (SEED = 42, N_NULL = 1000)
    k grid  = every k of the REAL curve; nulls that do not reach k are NaN there
    rho(k)  = phi(k) / nanmean(phi_null(k))
    p(k)    = (nansum(phi_null(k) >= phi(k)) + 1) / (#finite phi_null(k) + 1)
    sig(k)  = rho(k) > 1 and p(k) < 0.05
    floor   = keep k only if at least MINNODES (20) nodes have degree > k
    reports n_sig_full, run_full (all k) and n_sig_floor, run_floor (floored k),
    run = longest run of consecutive significant k

The paper's supplementary rich-club counts are the floored ones (n_sig_floor,
run_floor).  The scored audit criterion is gold_rich_club.py (unfloored, a
different k grid and a partly different null ensemble: the audit's first 100
curves come from seeds 1042.., this script's from 5042.. only).

Self-contained: needs numpy and networkx only, and imports neither
connectome_audit_gold.py nor richclub_other_graphs.py.  load_layered_graph and
the null-model functions are copied unchanged from connectome_audit_gold.py
(which richclub_other_graphs.py imports them from); _null_phi, longest_run,
compute and the module constants are copied unchanged from
richclub_other_graphs.py.  As there, the nulls are built in a Pool of NPROC
workers (pool.map keeps their order); --workers sets NPROC (the source fixes
it at 8), which changes no number.

Usage:
    python richclub_floored.py --graph G.npz
    python richclub_floored.py --graph G.npz --n-null 1000 --minnodes 20 --workers 3 --json rc_floor.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple

import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
# Constants  (copied from richclub_other_graphs.py)
# ═══════════════════════════════════════════════════════════════════════════

N_NULL   = 1000
SEED     = 42
NPROC    = 8
MINNODES = 20          # same publication floor as richclub_rho_pub.png


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
# Rich-club curve, nulls, floor  (copied from richclub_other_graphs.py)
# ═══════════════════════════════════════════════════════════════════════════


def _null_phi(args):
    layered, seed = args
    R = _generate_null(layered, seed)
    return nx.rich_club_coefficient(R, normalized=False)

def longest_run(mask):
    best = run = 0
    for x in mask:
        run = run + 1 if x else 0; best = max(best, run)
    return best

def compute(layered, label):
    t0 = time.time()
    G = layered.G
    null_kind = 'generic_degree_preserving' if _use_generic_null(layered) else 'tripartite_degree_preserving'
    phi = nx.rich_club_coefficient(G, normalized=False)
    with Pool(NPROC) as pool:
        nulls = pool.map(_null_phi, [(layered, SEED + 5000 + i) for i in range(N_NULL)], chunksize=2)
    ks = np.array(sorted(phi))
    real = np.array([phi[k] for k in ks], float)
    null = np.array([[c.get(int(k), np.nan) for k in ks] for c in nulls], float)
    mean = np.nanmean(null, 0); sd = np.nanstd(null, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = real / mean
    pval = (np.nansum(null >= real, 0) + 1) / (np.sum(np.isfinite(null), 0) + 1)
    sig = (rho > 1) & (pval < 0.05)

    # >=20-node floor: keep thresholds where at least MINNODES nodes have degree > k
    deg = np.array([dd for _, dd in G.degree()])
    n_above = np.array([(deg > k).sum() for k in ks])
    keep = n_above >= MINNODES
    kmax = int(ks[keep].max()) if keep.any() else 0
    sig_floor = sig & keep

    res = dict(
        n=G.number_of_nodes(), m=G.number_of_edges(), n_null=N_NULL, null_model=null_kind,
        ks=[int(k) for k in ks], rho=rho.tolist(),
        null_mean=mean.tolist(), null_std=sd.tolist(),
        sig=[bool(x) for x in sig], degrees=[int(x) for x in deg],
        n_sig_full=int(sig.sum()), run_full=longest_run(sig),
        n_sig_floor=int(sig_floor.sum()), run_floor=longest_run(sig_floor),
        floor_kmax=kmax, minnodes=MINNODES)
    print(f"[{label}] n={res['n']} m={res['m']} null={null_kind}\n"
          f"    FULL      : {res['n_sig_full']} sig, longest run {res['run_full']}\n"
          f"    FLOOR>=20 : {res['n_sig_floor']} sig, longest run {res['run_floor']}  (k<={kmax})\n"
          f"    {time.time()-t0:.0f}s", flush=True)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════


def _runs(ks: List[int]) -> List[Tuple[int, int]]:
    """Consecutive runs [(first, last), ...] of a sorted list of integers."""
    out = []
    for k in ks:
        if out and k == out[-1][1] + 1:
            out[-1] = (out[-1][0], k)
        else:
            out.append((k, k))
    return out


def main():
    global N_NULL, MINNODES, NPROC
    ap = argparse.ArgumentParser(
        description="Rich-club curve of one graph vs degree-preserving nulls, "
                    "with the >=20-node floor (richclub_other_graphs.py, standalone).")
    ap.add_argument('--graph', required=True, metavar='PATH',
                    help='Graph to analyse (.npz or .gml)')
    ap.add_argument('--json', default='', metavar='PATH',
                    help='Output JSON for this graph')
    ap.add_argument('--n-null', type=int, default=N_NULL,
                    help=f'Number of degree-preserving nulls (default: {N_NULL})')
    ap.add_argument('--minnodes', type=int, default=MINNODES,
                    help=f'Floor: only count k with >= this many nodes above k '
                         f'(default: {MINNODES}). The paper reports floored values.')
    ap.add_argument('--workers', type=int, default=NPROC,
                    help=f'Pool size for the nulls (default: {NPROC}, as the source)')
    args = ap.parse_args()

    N_NULL   = args.n_null
    MINNODES = args.minnodes
    NPROC    = max(1, args.workers)
    name     = os.path.splitext(os.path.basename(args.graph))[0]

    layered = load_layered_graph(args.graph)
    result  = compute(layered, name)

    ks = result['ks']
    deg = np.array(result['degrees'])
    sig_full = [k for k, s in zip(ks, result['sig']) if s]
    sig_floor = [k for k in sig_full if (deg > k).sum() >= MINNODES]
    result['significant_k_full'] = sig_full
    result['significant_k_floor'] = sig_floor
    print("\n" + "=" * 72)
    print(f"RICH-CLUB COUNTS (richclub_other_graphs.py), {N_NULL} nulls, floor {MINNODES} nodes")
    print("=" * 72)
    print(f"  k grid (real curve): {ks[0]}..{ks[-1]}  ({len(ks)} thresholds);  floor keeps k <= {result['floor_kmax']}")
    print(f"  FULL      : {result['n_sig_full']} significant k, longest run {result['run_full']}")
    print(f"              " + (", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in _runs(sig_full)) or 'none'))
    print(f"  FLOOR>={MINNODES:<3d}: {result['n_sig_floor']} significant k, longest run {result['run_floor']}")
    print(f"              " + (", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in _runs(sig_floor)) or 'none'))

    if args.json:
        d = os.path.dirname(os.path.abspath(args.json))
        if d:
            os.makedirs(d, exist_ok=True)
        json.dump(result, open(args.json, 'w'), indent=2)
        print(f"  -> {args.json}", flush=True)


if __name__ == '__main__':
    main()
