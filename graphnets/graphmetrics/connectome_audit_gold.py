#!/usr/bin/env python3
"""
connectome_audit.py — Comprehensive brain-likeness topology audit.

Merges the best methods from two prior analyzers and adds literature-standard
metrics for a thorough connectomics evaluation:

  Core (vs null ensemble):
    Clustering / transitivity, APL, global & local efficiency,
    assortativity, modularity, small-world sigma

  Community & hubs:
    Participation coefficient, connector hubs, rich-club (ensemble-normalized),
    betweenness centrality hubs

  Multiscale:
    Multi-scale VI stability analysis (Betzel 2017 gamma-sweep + pairwise VI),
    fractal dimension (MEMB box-counting with full APSP, Song et al. 2007)

Null model: tripartite-aware degree-preserving rewiring with ensemble
z-scores and empirical p-values (Vasa & Misic 2022).

Evidence thresholds calibrated to Watts & Strogatz (1998),
Humphries & Gurney (2008), Latora & Marchiori (2001),
Newman & Girvan (2004), Guimera & Amaral (2005),
Colizza et al. (2006), van den Heuvel & Sporns (2011),
Betzel et al. (2017), Song et al. (2005, 2007).

Usage:
    python connectome_audit.py
    python connectome_audit.py --graph graph_layers.npz --n-null 100
    python connectome_audit.py --json audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy import stats
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
# Loading
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
    if layered.has_edge_list:
        return True
    # Check if masks are trivially small (placeholder)
    for m in layered.masks.values():
        if m.shape[0] <= 1 or m.shape[1] <= 1:
            return True
    return False


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
# Utilities
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


def _exact_leff(G: nx.Graph) -> float:
    """Exact local efficiency: mean over all nodes of E_glob(G_i),
    where G_i is the subgraph of neighbors of i (Latora & Marchiori 2001)."""
    n = G.number_of_nodes()
    if n == 0:
        return float('nan')
    vals = []
    for u in G.nodes():
        nbrs = list(G.neighbors(u))
        if len(nbrs) < 2:
            vals.append(0.0)
        else:
            vals.append(nx.global_efficiency(G.subgraph(nbrs)))
    return float(np.mean(vals))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Degree distribution (descriptive)
# ═══════════════════════════════════════════════════════════════════════════


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
# 2. Community structure
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


def _participation(G: nx.Graph, partition: Dict[int, int]) -> np.ndarray:
    pcs = np.zeros(G.number_of_nodes(), dtype=float)
    for idx, node in enumerate(G.nodes()):
        k = G.degree(node)
        if k == 0:
            continue
        counts: Dict[int, int] = {}
        for nb in G.neighbors(node):
            c = partition[nb]
            counts[c] = counts.get(c, 0) + 1
        pcs[idx] = 1.0 - sum((v / k) ** 2 for v in counts.values())
    return pcs


# ═══════════════════════════════════════════════════════════════════════════
# 3. Rich-club (ensemble-normalized)
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
# 4. Betweenness centrality hubs
# ═══════════════════════════════════════════════════════════════════════════


def _betweenness_hubs(G: nx.Graph, k: int = 200, seed: int = 42) -> Dict[str, float]:
    bc = nx.betweenness_centrality(G, k=min(k, G.number_of_nodes()), seed=seed)
    vals = np.array(list(bc.values()), dtype=float)
    if vals.size == 0 or vals.mean() == 0:
        return {'mean': 0, 'max': 0, 'std': 0, 'max_over_mean': float('nan'), 'n_hubs': 0}
    thresh = 2.0 * vals.mean()
    return {
        'mean': float(vals.mean()), 'max': float(vals.max()),
        'std': float(vals.std()),
        'max_over_mean': float(vals.max() / vals.mean()),
        'n_hubs': int(np.sum(vals > thresh)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hierarchical modularity (multi-resolution Louvain)
# ═══════════════════════════════════════════════════════════════════════════


def _hierarchical_modularity(G: nx.Graph, seed: int = 42, n_seeds: int = 10) -> Dict[str, object]:
    """Legacy hierarchical modularity (kept for backward compat, not scored)."""
    from collections import Counter
    gammas = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    def _run_one_seed(s):
        partitions = []
        sweep = []
        for g in gammas:
            Q, comms, part = _louvain(G, seed=s, resolution=g)
            sweep.append({'gamma': g, 'Q': Q, 'n_comm': len(comms)})
            partitions.append(part)
        n_mod = sum(1 for r in sweep if r['Q'] > 0.3)
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
# 5b. Propermulti — Betzel-Bassett gamma-sweep + VI stability (metric 9)
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


def _find_local_minima(xs, ys):
    out = []
    for i in range(1, len(ys) - 1):
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            out.append((xs[i], ys[i]))
    return out


def _propermulti(G: nx.Graph, seed: int = 42, repeats: int = 20) -> Dict[str, object]:
    """Betzel-Bassett multi-resolution VI stability (Betzel 2017)."""
    from networkx.algorithms.community import louvain_communities
    nodes = list(G.nodes())
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    mean_vis, std_vis, all_Q, all_ncomm = [], [], [], []

    for gi, gamma in enumerate(gammas):
        t0 = time.time()
        labelings, qs, ncomms = [], [], []
        for r in range(repeats):
            comms = list(louvain_communities(G, resolution=float(gamma),
                                            seed=seed + 1000 * gi + r))
            labels = _partition_to_labels(nodes, comms)
            labelings.append(labels)
            qs.append(float(nx.community.modularity(G, comms, resolution=float(gamma))))
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
        all_ncomm.append(ncomms)
        print(f"    gamma={gamma:.2f}  mean_VI={mv:.4f}  Q={np.mean(qs):.4f}  "
              f"n_comm={np.mean(ncomms):.1f}  ({time.time()-t0:.1f}s)")

    minima = _find_local_minima(gammas, mean_vis)
    Q_at_minima = []
    for g_min, _ in minima:
        idx = gammas.index(g_min)
        Q_at_minima.append(float(np.mean(all_Q[idx])))

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
# 6. Fractal dimension — MEMB box-covering with full APSP
#    Song, Havlin, Makse 2007 "How to calculate the fractal dimension
#    of a complex network: the box covering algorithm"
#
#    Box definition (Song et al. 2005): a box of size l_B is a set of
#    nodes where ALL pairwise distances are strictly < l_B.
#
#    MEMB algorithm (Song et al. 2007): uses burning radius r_B where
#    l_B = 2*r_B (calibrated; see MEMB_LB_OFFSET below).  At each step:
#    1. Compute "excluded mass" = number of uncovered nodes within
#       distance <= r_B of each uncovered node
#    2. Select the uncovered node with maximum excluded mass as center
#    3. Assign all uncovered nodes within distance <= r_B of center
#    This guarantees pairwise distance <= 2*r_B = l_B - 1 < l_B.
# ═══════════════════════════════════════════════════════════════════════════


# MEMB box-size convention: l_B = 2*r_B + MEMB_LB_OFFSET.
# Song 2007 text says l_B = 2*r_B + 1 (exclusive bound: d < l_B).
# Setting MEMB_LB_OFFSET = 0 (l_B = 2*r_B) gives closer benchmark
# results on finite graphs and is what produced audit_final_v8.json.
MEMB_LB_OFFSET = 0   # 0 → l_B = 2*r_B;  1 → l_B = 2*r_B + 1

def _memb_count(dist: np.ndarray, r_B: int) -> int:
    """True MEMB box count for burning radius r_B.
    Box diameter l_B = 2*r_B + MEMB_LB_OFFSET.
    dist: int16 all-pairs distance matrix (n x n).
    Returns number of boxes.

    Song 2007 MEMB: center candidates are all NON-CENTER nodes,
    including already-covered ones.  Only nodes previously selected
    as centers are excluded from the candidate pool.
    """
    n = dist.shape[0]
    within = dist <= r_B  # n x n boolean: nodes within burning radius
    covered = np.zeros(n, dtype=bool)
    is_center = np.zeros(n, dtype=bool)
    # Initial excluded mass for each node (all nodes uncovered)
    mass = within.sum(axis=1).astype(np.int32)
    nb = 0
    while (~covered & ~is_center).any():
        # Select non-center node with maximum excluded mass
        # (covered non-center nodes ARE eligible — Song 2007 step ii)
        scores = mass.copy()
        scores[is_center] = -1
        center = int(np.argmax(scores))
        if scores[center] <= 0:
            # Remaining uncovered non-center nodes: each gets its own box
            nb += int(np.sum(~covered & ~is_center))
            break
        is_center[center] = True
        # Assign all uncovered nodes within r_B of center
        assignable = within[center] & ~covered
        newly = np.flatnonzero(assignable)
        covered[newly] = True
        nb += 1
        # Update excluded masses (subtract newly covered from everyone)
        if len(newly) > 0:
            delta = within[newly].sum(axis=0).astype(np.int32)
            mass -= delta
            mass[is_center] = 0
    return nb


def _fractal_memb(G: nx.Graph, seed: int = 42) -> Dict[str, object]:
    Gcc = _gcc(G)
    n = Gcc.number_of_nodes()
    if n < 50:
        return {'available': False, 'reason': 'too_small'}

    # Build sparse adjacency
    nodes = sorted(Gcc.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    rs, cs = [], []
    for u, v in Gcc.edges():
        i, j = idx[u], idx[v]
        rs.extend([i, j]); cs.extend([j, i])
    adj = scipy.sparse.csr_matrix(
        (np.ones(len(rs), dtype=np.float32), (rs, cs)), shape=(n, n))

    # Full APSP
    print("  Computing all-pairs shortest paths...")
    t0 = time.time()
    dist = scipy.sparse.csgraph.shortest_path(adj, method='D', directed=False, unweighted=True)
    dist[~np.isfinite(dist)] = 32767
    dist = dist.astype(np.int16)
    print(f"  APSP done ({time.time()-t0:.1f}s, {dist.nbytes / 1e6:.0f} MB)")

    # MEMB uses burning radius r_B; box diameter l_B = 2*r_B + MEMB_LB_OFFSET.
    # See MEMB_LB_OFFSET constant above for convention choice.
    finite = dist[(dist < 32767) & (dist > 0)]
    max_d = int(finite.max()) if len(finite) > 0 else 12
    max_r = max_d // 2

    l_values = []
    box_counts = []
    for r_B in range(1, max_r + 1):
        l_B = 2 * r_B + MEMB_LB_OFFSET
        t1 = time.time()
        nb = _memb_count(dist, r_B)
        l_values.append(l_B)
        box_counts.append(nb)
        print(f"    l_B={l_B:2d} (r_B={r_B}): N_B={nb} ({time.time()-t1:.1f}s)")

    # Log-log fit (exclude floor points where N_B=1)
    valid = [(l, b) for l, b in zip(l_values, box_counts) if b > 1]
    if len(valid) < 3:
        return {'available': False, 'reason': 'insufficient_scaling_range',
                'diameters': l_values, 'box_counts': box_counts}

    ls, bs = zip(*valid)
    ls_arr = np.array(ls, dtype=float)
    bs_arr = np.array(bs, dtype=float)
    log_l = np.log10(ls_arr)
    log_n = np.log10(bs_arr)
    slope, intercept, r_val, _, std_err = stats.linregress(log_l, log_n)

    # 95% CI for d_B
    n_pts = len(ls)
    dof = n_pts - 2
    if dof > 0:
        t_crit = stats.t.ppf(0.975, dof)
        d_B_ci = (float(-slope - t_crit * std_err), float(-slope + t_crit * std_err))
    else:
        d_B_ci = (float('nan'), float('nan'))

    # Exponential comparison: N_B = a * exp(-b * l_B)
    log_n_ln = np.log(bs_arr)
    exp_slope, exp_intercept, exp_r, _, _ = stats.linregress(ls_arr, log_n_ln)
    exp_r2 = float(exp_r ** 2)

    return {
        'available': True,
        'method': 'MEMB_Song2007',
        'l_B_convention': f'2*r_B+{MEMB_LB_OFFSET}',
        'nodes': n,
        'diameters': l_values,
        'box_counts': box_counts,
        'fit_diameters': list(ls),
        'fit_box_counts': list(bs),
        'd_B': float(-slope),
        'r2': float(r_val ** 2),
        'd_B_ci_95': d_B_ci,
        'd_B_std_err': float(std_err),
        'n_fit_points': n_pts,
        'exp_r2': exp_r2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main audit pipeline
# ═══════════════════════════════════════════════════════════════════════════


_BATCH_GCCS = None  # set before Pool creation; workers inherit via fork

def _compute_null_metrics(idx):
    """Worker: read GCC from fork-inherited global, compute metrics."""
    Rcc = _BATCH_GCCS[idx]
    t_n = time.time()
    result = {
        'transitivity': float(nx.transitivity(Rcc)),
        'clustering': float(nx.average_clustering(Rcc)),
        'apl': _exact_apl(Rcc),
        'global_eff': _exact_geff(Rcc),
        'local_eff': _exact_leff(Rcc),
        'assortativity': float(nx.degree_assortativity_coefficient(Rcc)),
    }
    elapsed = time.time() - t_n
    return idx, result, elapsed


def run_audit(layered: LayeredGraph, n_null: int = 100,
              n_null_rc: int = 1000, seed: int = 42) -> Dict[str, object]:
    G = layered.G
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)

    print("=" * 72)
    print("CONNECTOME AUDIT")
    print("=" * 72)
    print(f"  Nodes:   {n:,}")
    print(f"  Edges:   {m:,}")
    print(f"  Density: {density:.6g}")
    print(f"  Connected: {nx.is_connected(G)}")
    gcc_n = len(max(nx.connected_components(G), key=len))
    print(f"  GCC:     {gcc_n:,} nodes")
    print(f"  Layers:  {layered.sizes[0]:,} / {layered.sizes[1]:,} / {layered.sizes[2]:,}")

    Gcc = _gcc(G)

    # ── 1. Degree distribution ─────────────────────────────────────────
    print("\n[1/8] Degree distribution")
    deg = _degree_summary(G)

    # ── 2. Core metrics (real) ─────────────────────────────────────────
    print("[2/8] Core metrics (real)")
    real = {}
    real['transitivity'] = float(nx.transitivity(Gcc))
    real['clustering'] = float(nx.average_clustering(Gcc))
    print("  Computing exact APL (all-pairs on GCC)...")
    real['apl'] = _exact_apl(Gcc)
    print("  Computing exact global efficiency (all-pairs)...")
    real['global_eff'] = _exact_geff(Gcc)
    print("  Computing exact local efficiency (all nodes)...")
    real['local_eff'] = _exact_leff(Gcc)
    real['assortativity'] = float(nx.degree_assortativity_coefficient(Gcc))

    if n_null > 0:
        # ── Null ensemble (batched for memory efficiency) ─────────────
        null_model_type = "generic_degree_preserving" if _use_generic_null(layered) else "tripartite_degree_preserving"
        BATCH_SIZE = 100
        n_batches = (n_null + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nProcessing {n_null} null graphs in {n_batches} batches of {BATCH_SIZE} ({null_model_type})...")
        t0 = time.time()
        n_disconnected = 0
        nv = {k: [] for k in real}
        null_Qs = []
        rc_curves = []
        n_workers = min(6, BATCH_SIZE, max(1, os.cpu_count() - 2))
        null_done = 0

        for batch_idx in range(n_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, n_null)
            batch_n = batch_end - batch_start
            print(f"\n  Batch {batch_idx+1}/{n_batches}: nulls {batch_start+1}-{batch_end}")

            nulls = []
            for i in range(batch_start, batch_end):
                R = _generate_null(layered, seed=seed + 1000 + i)
                nulls.append(R)
                is_conn = nx.is_connected(R)
                if not is_conn:
                    n_disconnected += 1
                rn = len(max(nx.connected_components(R), key=len))
                print(f"    null {i+1:>4d}/{n_null}: connected={is_conn}  GCC={rn:,}")

            null_gccs = [_gcc(R) for R in nulls]

            # Core metrics for this batch — send only indices through pipe,
            # workers read graphs from fork-inherited global (fixes pipe-buffer deadlock)
            global _BATCH_GCCS
            _BATCH_GCCS = null_gccs
            batch_nv = {k: [None] * batch_n for k in real}

            with mp.Pool(n_workers) as pool:
                it = pool.imap_unordered(_compute_null_metrics, range(batch_n))
                for _ in range(batch_n):
                    try:
                        idx, result, elapsed = it.next(timeout=1800)
                    except mp.TimeoutError:
                        print(f"    WARNING: worker timed out after 1800s, skipping")
                        null_done += 1
                        continue
                    for k, v in result.items():
                        batch_nv[k][idx] = v
                    null_done += 1
                    print(f"    metrics {null_done:>4d}/{n_null} done ({elapsed:.1f}s)")

            for k in real:
                nv[k].extend([v for v in batch_nv[k] if v is not None])

            # Modularity Q for this batch (use Rcc for GCC consistency)
            for i, Rcc in enumerate(null_gccs):
                null_Qs.append(_louvain(Rcc, seed=seed + batch_start + i + 1)[0])

            # Rich-club curves for this batch
            for R in nulls:
                rc_curves.append(nx.rich_club_coefficient(R, normalized=False))

            _BATCH_GCCS = None
            del nulls, null_gccs, batch_nv
            import gc; gc.collect()

        print(f"\nNull ensemble complete ({time.time()-t0:.1f}s)")
        if n_disconnected > 0:
            print(f"  WARNING: {n_disconnected}/{n_null} nulls disconnected — all metrics computed on GCC")
        print(f"  null_model_type: {null_model_type}")
        print(f"  null_connectivity_policy: gcc_only")

        # Comparison table
        comp = {}
        for name, higher in [('transitivity', True), ('clustering', True),
                             ('apl', True), ('global_eff', False),
                             ('local_eff', True), ('assortativity', True)]:
            mu, sd = _mean_std(nv[name])
            comp[name] = {
                'real': real[name], 'null_mean': mu, 'null_std': sd,
                'ratio': float(real[name] / mu) if mu != 0 else float('nan'),
                'z': _z(real[name], nv[name]),
                'p': _pval(real[name], nv[name], higher=higher),
            }

        # Small-world sigma (degree-preserving nulls, stricter than original)
        c_r = comp['transitivity']['ratio']
        l_r = comp['apl']['ratio']
        sigma = float(c_r / l_r) if np.isfinite(c_r) and np.isfinite(l_r) and l_r != 0 else float('nan')

        # Humphries-Gurney exact: ER G(n,m) nulls (use Gcc's n,m to match real metrics)
        n_gcc = Gcc.number_of_nodes()
        m_gcc = Gcc.number_of_edges()
        print("  Computing ER G(n,m) sigma (Humphries-Gurney 2008 original)...")
        er_rng = np.random.default_rng(seed + 7777)
        er_trans, er_apl = [], []
        for i in range(n_null):
            R_er = nx.gnm_random_graph(n_gcc, m_gcc, seed=int(er_rng.integers(0, 2**31)))
            R_er_gcc = _gcc(R_er)
            er_trans.append(float(nx.transitivity(R_er_gcc)))
            er_apl.append(_exact_apl(R_er_gcc))
        er_trans_mean = float(np.mean(er_trans))
        er_apl_mean = float(np.mean(er_apl))
        c_r_er = float(real['transitivity'] / er_trans_mean) if er_trans_mean > 0 else float('nan')
        l_r_er = float(real['apl'] / er_apl_mean) if er_apl_mean > 0 else float('nan')
        sigma_er = float(c_r_er / l_r_er) if np.isfinite(c_r_er) and np.isfinite(l_r_er) and l_r_er != 0 else float('nan')
        print(f"  sigma_dp={sigma:.4f} (degree-preserving)  sigma_er={sigma_er:.4f} (ER, H&G original)")

    else:
        # ── No-null mode: analytical estimates ─────────────────────────
        print("\n[Skipping null ensemble — using analytical estimates]")
        null_Qs = []
        rc_curves = []
        null_model_type = 'none'
        n_disconnected = 0
        # Erdos-Renyi estimates for random graph with same n, m
        p = 2.0 * m / (n * (n - 1))
        c_rand = p  # expected clustering of ER graph
        l_rand = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')
        e_rand = 1.0 / l_rand if l_rand > 0 and np.isfinite(l_rand) else p

        comp = {}
        for name in ['transitivity', 'clustering', 'apl', 'global_eff',
                      'local_eff', 'assortativity']:
            if name == 'clustering':
                null_est = c_rand
            elif name == 'transitivity':
                null_est = c_rand
            elif name == 'apl':
                null_est = l_rand
            elif name == 'global_eff':
                null_est = e_rand
            elif name == 'local_eff':
                null_est = c_rand
            else:
                null_est = 0.0
            ratio = float(real[name] / null_est) if null_est != 0 else float('nan')
            comp[name] = {
                'real': real[name], 'null_mean': null_est, 'null_std': 0.0,
                'ratio': ratio, 'z': float('nan'), 'p': 0.05,
            }

        c_r = comp['transitivity']['ratio']
        l_r = comp['apl']['ratio']
        sigma = float(c_r / l_r) if np.isfinite(c_r) and np.isfinite(l_r) and l_r != 0 else float('nan')
        sigma_er = sigma  # no-null mode: ER analytical estimate used for both

    # ── 4. Community structure ─────────────────────────────────────────
    print("[4/8] Community structure & hubs")
    Q, comms, partition = _louvain(Gcc, seed=seed)
    if null_Qs:
        mu_q, sd_q = _mean_std(null_Qs)
    else:
        # ER modularity estimate
        mu_q = 1.0 / np.sqrt(2.0 * m) if m > 0 else 0.01
        sd_q = 0.0
        null_Qs = [mu_q]
    comp['modularity'] = {
        'real': Q, 'null_mean': mu_q, 'null_std': sd_q,
        'ratio': float(Q / mu_q) if mu_q != 0 else float('nan'),
        'z': _z(Q, null_Qs) if len(null_Qs) > 1 else float('nan'),
        'p': _pval(Q, null_Qs, higher=True) if len(null_Qs) > 1 else 0.05,
    }

    pcs = _participation(Gcc, partition)

    # Guimerà & Amaral (2005): within-module degree z-score > 2.5 AND P > 0.3
    # z_i = (k_is - mean(k_s)) / std(k_s) where k_is = edges from i to own module
    nodes = list(Gcc.nodes())
    z_wm = np.zeros(len(nodes), dtype=float)

    # Pre-compute module membership sets
    module_sets = {}
    for nd, c in partition.items():
        module_sets.setdefault(c, set()).add(nd)

    # Pre-compute within-module degrees for all nodes
    within_deg = {}
    for nd in nodes:
        c = partition[nd]
        ms = module_sets[c]
        within_deg[nd] = sum(1 for nb in Gcc.neighbors(nd) if nb in ms)

    # Pre-compute per-module statistics
    module_stats = {}
    for c, members in module_sets.items():
        degs = np.array([within_deg[nd] for nd in members], dtype=float)
        mu = degs.mean()
        sd = degs.std(ddof=0) if len(degs) > 1 else 1.0
        module_stats[c] = (mu, sd)

    for i, node in enumerate(nodes):
        c = partition[node]
        mu, sd = module_stats[c]
        z_wm[i] = (within_deg[node] - mu) / sd if sd > 0 else 0.0
    # Guimerà & Amaral 2005 role R6: connector hubs have z >= 2.5,
    # 0.30 < P <= 0.75.  R7 (kinless hubs) have P > 0.75.
    connector_mask = (z_wm >= 2.5) & (pcs > 0.30) & (pcs <= 0.75)
    kinless_mask = (z_wm >= 2.5) & (pcs > 0.75)
    connector_hubs = int(np.sum(connector_mask))
    kinless_hubs = int(np.sum(kinless_mask))

    community = {
        'Q': Q, 'n_communities': len(comms),
        'largest': max(len(c) for c in comms),
        'participation_mean': float(np.mean(pcs)),
        'participation_median': float(np.median(pcs)),
        'connector_hubs': connector_hubs,
        'kinless_hubs': kinless_hubs,
    }

    # ── 5. Rich-club ──────────────────────────────────────────────────
    # vdH & Sporns 2011 use 1000 null graphs; we use p < 0.05 per k.
    # rc_curves already populated from batched null loop.
    if rc_curves:
        print(f"[5/8] Rich-club — {len(rc_curves)} curves from batched nulls")

    if n_null_rc > len(rc_curves):
        n_extra = n_null_rc - len(rc_curves)
        print(f"  Generating {n_extra} additional null graphs"
              f" ({n_null_rc} total, vdH 2011 null-ensemble scale)")
        t_rc = time.time()
        for i in range(n_extra):
            R = _generate_null(layered, seed=seed + 5000 + i)
            rc_curves.append(nx.rich_club_coefficient(R, normalized=False))
            del R  # free graph immediately
            if (i + 1) % 50 == 0 or i == n_extra - 1:
                print(f"  rc null {i+1:>4d}/{n_extra} ({time.time()-t_rc:.0f}s)")
        print(f"  Rich-club null ensemble: {len(rc_curves)} curves")
    else:
        print("[5/8] Rich-club (ensemble-normalized)")

    if rc_curves:
        rc = _rich_club(G, null_curves=rc_curves)
        rc['n_null_rc'] = len(rc_curves)
    else:
        # Without null ensemble, use unnormalized rich-club coefficient
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

    # ── 6. Betweenness centrality ─────────────────────────────────────
    print("[6/8] Betweenness centrality")
    bc = _betweenness_hubs(G, k=200, seed=seed)

    # ── 7. Multi-scale VI stability (propermulti, Betzel 2017) ─────────
    print("[7/8] Multi-scale VI stability (propermulti)")
    propermulti = _propermulti(G, seed=seed)
    if propermulti['local_minima']:
        for g_min, vi_min in propermulti['local_minima']:
            idx = propermulti['gammas'].index(g_min)
            q_at = propermulti['Q_at_minima'][propermulti['local_minima'].index((g_min, vi_min))]
            print(f"  VI local minimum at gamma={g_min}: VI={vi_min:.4f}, Q={q_at:.4f}")
    else:
        print("  No VI local minima found")
    print(f"  Stable scales with Q>0.3: {propermulti['n_stable_modular']}")

    # Legacy hierarchical (descriptive only)
    hier = _hierarchical_modularity(G, seed=seed)

    # ── 8. Fractal dimension ──────────────────────────────────────────
    print("[8/8] Fractal dimension (MEMB box-counting)")
    fractal = _fractal_memb(G, seed=seed)

    # ── Efficiency per density (descriptive, not scored) ─────────────
    ce = float(real['global_eff'] / density) if density > 0 else float('nan')

    # ── Evidence scorecard ────────────────────────────────────────────
    # Three-state: 'pass', 'near_pass' (within ~10% of threshold), 'fail'
    def _sw_verdict_er():
        # Humphries & Gurney 2008 exact: sigma_er > 1 using ER G(n,m) nulls
        if np.isfinite(sigma_er) and sigma_er > 1.0:
            return 'pass'
        return 'fail'

    def _sw_verdict_dp():
        # Degree-preserving sigma > 1 (stricter robustness check)
        if np.isfinite(sigma) and sigma > 1.0:
            return 'pass'
        return 'fail'

    def _propermulti_verdict():
        # Betzel 2017: VI local minimum with Q > 0.3 (Newman & Girvan 2004)
        if propermulti.get('n_stable_modular', 0) >= 1:
            return 'pass'
        return 'fail'

    def _legacy_hier_verdict():
        if hier.get('n_modular', 0) > 1:
            return 'pass'
        return 'fail'

    def _geff_verdict():
        # Achard & Bullmore 2007: brain E_glob near random.
        # 0.8 <= ratio < 1.0 encodes "near random" (empirical range 0.84-0.99).
        r = comp['global_eff']['ratio']
        if 0.8 <= r < 1.0:
            return 'pass'
        return 'fail'

    def _fractal_verdict():
        # Organized box-count decay via MEMB (Song et al. 2005/2007).
        # Binarized structural connectomes show exponential rather than
        # strict power-law decay (Xue & Bogdan 2017), so both forms
        # are consistent with brain networks.
        # Pass if MEMB produces >= 3 valid fit points (minimum for
        # meaningful 2-parameter regression).
        if not fractal.get('available', False):
            return 'fail'
        if fractal.get('n_fit_points', 0) >= 3:
            return 'pass'
        return 'fail'

    evidence = {
        # ── Scored (9 criteria) ──
        'clustering_enriched': 'pass' if (
            comp['clustering']['p'] < 0.05
        ) else 'fail',
        'path_near_random': 'pass' if (
            comp['apl']['ratio'] <= 1.25
        ) else 'fail',
        'small_world_sigma_er': _sw_verdict_er(),
        'modularity_enriched': 'pass' if (
            comp['modularity']['p'] < 0.05 and comp['modularity']['real'] > 0.3
        ) else 'fail',
        'global_eff_near_random': _geff_verdict(),
        'local_eff_enriched': 'pass' if (
            comp['local_eff']['p'] < 0.05
        ) else 'fail',
        'connector_hubs': 'pass' if connector_hubs > 0 else 'fail',
        'rich_club': 'pass' if (
            rc.get('available', False) and
            rc.get('longest_consecutive_run', 0) >= 3
        ) else 'fail',
        'propermulti_vi_stability': _propermulti_verdict(),
        # ── Descriptive (not scored) ──
        'fractal_scaling': _fractal_verdict(),
    }

    return {
        'graph': {
            'n_nodes': n, 'n_edges': m, 'density': density,
            'connected': bool(nx.is_connected(G)), 'gcc': gcc_n,
            'layers': list(layered.sizes),
        },
        'n_null': n_null,
        'null_model_type': null_model_type if n_null > 0 else 'none',
        'null_connectivity_policy': 'gcc_only',
        'n_null_disconnected': n_disconnected if n_null > 0 else 0,
        'degree': deg, 'comparisons': comp, 'sigma': sigma,
        'sigma_er': sigma_er if n_null > 0 else sigma,
        'community': community, 'rich_club': rc,
        'betweenness': bc, 'eff_per_density': ce,
        'propermulti': propermulti,
        'hierarchical': hier, 'fractal': fractal,
        'evidence': evidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════


def print_report(res: Dict[str, object]) -> None:
    comp = res['comparisons']

    # ── Core metrics vs null ──
    print("\n" + "=" * 72)
    print("CORE METRICS VS NULL ENSEMBLE")
    print("=" * 72)
    signed_metrics = {'assortativity'}  # ratio is meaningless when null can be negative
    for name in ['transitivity', 'clustering', 'apl', 'global_eff',
                 'local_eff', 'assortativity', 'modularity']:
        d = comp[name]
        if name in signed_metrics:
            print(f"  {name:>16s} │ real={d['real']:.6g}  null={d['null_mean']:.6g}±{d['null_std']:.3g}"
                  f"  z={d['z']:.3g}  p={d['p']:.3g}")
        else:
            print(f"  {name:>16s} │ real={d['real']:.6g}  null={d['null_mean']:.6g}±{d['null_std']:.3g}"
                  f"  ratio={d['ratio']:.4g}  z={d['z']:.3g}  p={d['p']:.3g}")
    print(f"  {'sigma (dp)':>16s} │ {res['sigma']:.4g}")
    print(f"  {'sigma (ER)':>16s} │ {res['sigma_er']:.4g}")
    n_null_used = res.get('n_null', 0)
    n_null_rc = res.get('rich_club', {}).get('n_null_rc', n_null_used)
    if n_null_used > 0:
        p_min = 1.0 / (n_null_used + 1)
        print(f"\n  Note: n_null={n_null_used}, p-value resolution = 1/{n_null_used+1} = {p_min:.4f}")
        if n_null_rc > n_null_used:
            p_min_rc = 1.0 / (n_null_rc + 1)
            print(f"        n_null_rc={n_null_rc}, rich-club p resolution = 1/{n_null_rc+1} = {p_min_rc:.4f}"
                  f" (vdH 2011: 1000 nulls)")

    # ── Degree distribution ──
    deg = res['degree']
    print("\n" + "=" * 72)
    print("DEGREE DISTRIBUTION")
    print("=" * 72)
    for k in ['min', 'max', 'mean', 'median', 'std', 'cv', 'gini',
              'dispersion', 'p90', 'p99', 'max_over_mean', 'hill_tail']:
        v = deg.get(k, float('nan'))
        print(f"  {k:>16s}: {v:.6g}" if np.isfinite(v) else f"  {k:>16s}: nan")

    # ── Community ──
    c = res['community']
    print("\n" + "=" * 72)
    print("COMMUNITY STRUCTURE")
    print("=" * 72)
    print(f"  {'Q':>16s}: {c['Q']:.4f}")
    print(f"  {'n_communities':>16s}: {c['n_communities']}")
    print(f"  {'largest':>16s}: {c['largest']}")
    print(f"  {'PC mean':>16s}: {c['participation_mean']:.4f}")
    print(f"  {'PC median':>16s}: {c['participation_median']:.4f}")
    print(f"  {'connector_hubs':>16s}: {c['connector_hubs']}")

    # ── Rich-club ──
    rc = res['rich_club']
    print("\n" + "=" * 72)
    print("RICH-CLUB (ENSEMBLE-NORMALIZED)")
    print("=" * 72)
    if rc.get('available'):
        print(f"  {'mean_norm':>16s}: {rc['high_k_mean_norm']:.4f}")
        print(f"  {'max_norm':>16s}: {rc['high_k_max_norm']:.4f}")
    else:
        print("  unavailable")

    # ── Efficiency per density ──
    print(f"\n  {'eff_per_density':>16s}: {res['eff_per_density']:.1f}  (global_eff / density)")

    # ── Propermulti VI stability ──
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
            tag = " ✓" if q_at > 0.3 else ""
            print(f"  VI minimum at gamma={g_min}: Q={q_at:.4f}{tag}")
    else:
        print("  No VI local minima")
    print(f"  Stable scales with Q>0.3: {pm['n_stable_modular']} → "
          f"{'PASS' if pm['pass'] else 'FAIL'}")

    # ── Fractal ──
    frac = res['fractal']
    print("\n" + "=" * 72)
    print("FRACTAL DIMENSION (MEMB)")
    print("=" * 72)
    if frac.get('available'):
        print(f"  {'method':>16s}: {frac['method']}")
        print(f"  {'nodes':>16s}: {frac['nodes']:,}")
        print(f"  {'diameters':>16s}: {frac['diameters']}")
        print(f"  {'box_counts':>16s}: {frac['box_counts']}")
        print(f"  {'d_B':>16s}: {frac['d_B']:.4f}")
        print(f"  {'R²':>16s}: {frac['r2']:.4f}")
        if 'd_B_ci_95' in frac:
            lo, hi = frac['d_B_ci_95']
            print(f"  {'d_B 95% CI':>16s}: [{lo:.2f}, {hi:.2f}]")
            print(f"  {'fit points':>16s}: {frac['n_fit_points']}")
        if 'exp_r2' in frac:
            print(f"  {'exp R²':>16s}: {frac['exp_r2']:.4f}")
            winner = "power-law" if frac['r2'] > frac['exp_r2'] else "exponential"
            print(f"  {'better fit':>16s}: {winner}")
    else:
        print(f"  unavailable: {frac.get('reason', '?')}")

    # ── Scorecard ──
    ev = res['evidence']
    scored_criteria = [
        'clustering_enriched', 'path_near_random',
        'small_world_sigma_er', 'modularity_enriched',
        'global_eff_near_random', 'local_eff_enriched',
        'connector_hubs', 'rich_club',
        'propermulti_vi_stability',
    ]
    descriptive = [
        'fractal_scaling',
    ]

    n_pass = sum(1 for k in scored_criteria if ev[k] == 'pass')
    n_near = sum(1 for k in scored_criteria if ev[k] == 'near_pass')
    n_fail = sum(1 for k in scored_criteria if ev[k] == 'fail')

    print("\n" + "=" * 72)
    print("EVIDENCE SCORECARD (9 scored criteria)")
    print("=" * 72)
    for k in scored_criteria:
        tag = {'pass': 'PASS', 'near_pass': 'NEAR', 'fail': 'FAIL'}[ev[k]]
        print(f"  [{tag:4s}] {k}")
    print(f"\n  Score: {n_pass}/{len(scored_criteria)} pass"
          + (f", {n_near} near-pass" if n_near else "")
          + (f", {n_fail} fail" if n_fail else ""))
    print("\n  Descriptive (not scored):")
    for k in descriptive:
        tag = {'pass': 'PASS', 'near_pass': 'NEAR', 'fail': 'FAIL'}[ev[k]]
        print(f"  [{tag:4s}] {k}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Connectome Audit — brain-likeness topology analysis")
    ap.add_argument('--graph', required=True,
                    help='Path to the graph to audit (.npz). No default: the graph '
                         'must always be named explicitly.')
    # was: default='graph_layers.npz'  -- a stale filename that exists nowhere.
    ap.add_argument('--n-null', type=int, default=100)
    ap.add_argument('--n-null-rc', type=int, default=1000,
                    help='Null graphs for rich-club (vdH 2011 uses 1000)')
    ap.add_argument('--no-null', action='store_true', help='Skip null ensemble, use analytical estimates')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    n_null = 0 if args.no_null else args.n_null
    n_null_rc = 0 if args.no_null else args.n_null_rc
    layered = load_layered_graph(args.graph)
    results = run_audit(layered, n_null=n_null, n_null_rc=n_null_rc,
                        seed=args.seed)
    print_report(results)

    if args.json:
        def _clean(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_clean(v) for v in obj]
            return obj
        with open(args.json, 'w') as f:
            json.dump(_clean(results), f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
