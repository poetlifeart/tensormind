#!/usr/bin/env python3
"""
clique_census.py — Sizemore et al. 2018 style clique analysis.

Computes:
  1. Maximal clique enumeration via Bron-Kerbosch (NetworkX)
  2. h(k): histogram of maximal clique sizes
  3. P_k(v): node participation in maximal k-cliques
  4. P(v): total node participation = sum_k P_k(v)
  5. Comparison against degree-preserving random nulls
  6. Bimodality assessment of h(k)

Reference: Sizemore AE et al. (2018) "Cliques and cavities in the
human connectome." J Comput Neurosci 44:115-145.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════
# Core: maximal clique enumeration + participation
# ═══════════════════════════════════════════════════════════════════

def clique_census(G, timeout=None, verbose=True):
    """Enumerate all maximal cliques, compute h(k) and P_k(v).

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (no self-loops).
    timeout : float or None
        Max seconds for enumeration. None = no limit.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        'omega': int — clique number (largest maximal clique)
        'n_maximal': int — total number of maximal cliques
        'h': dict[int, int] — h(k) = count of maximal cliques of size k
        'P_k': dict[int, dict[int, int]] — P_k[k][v] = # maximal k-cliques containing v
        'P': dict[int, int] — P[v] = total participation = sum_k P_k[k][v]
        'timed_out': bool
    """
    h = Counter()            # h[k] = count of maximal k-cliques
    P_k = defaultdict(lambda: defaultdict(int))  # P_k[k][v]
    P = defaultdict(int)     # P[v] = total participation
    omega = 0
    n_maximal = 0
    timed_out = False
    t0 = time.time()

    for clique in nx.find_cliques(G):
        k = len(clique)
        h[k] += 1
        n_maximal += 1
        if k > omega:
            omega = k
            if verbose:
                print(f"  New max clique size: {omega}")

        for v in clique:
            P_k[k][v] += 1
            P[v] += 1

        if verbose and n_maximal % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"  ... {n_maximal:,} cliques enumerated ({elapsed:.0f}s, omega={omega})")

        if timeout and (time.time() - t0) > timeout:
            timed_out = True
            if verbose:
                print(f"  TIMEOUT after {n_maximal:,} cliques ({timeout}s)")
            break

    elapsed = time.time() - t0
    if verbose and not timed_out:
        print(f"  Complete: {n_maximal:,} maximal cliques ({elapsed:.1f}s)")

    return {
        'omega': omega,
        'n_maximal': n_maximal,
        'h': dict(h),
        'P_k': {k: dict(v) for k, v in P_k.items()},
        'P': dict(P),
        'timed_out': timed_out,
        'elapsed': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════
# Participation statistics (Sizemore Fig. 2-3)
# ═══════════════════════════════════════════════════════════════════

def participation_stats(census, G):
    """Compute participation summary statistics.

    Returns dict with:
        'P_array': ndarray of P(v) for all nodes (sorted by node id)
        'P_mean', 'P_std', 'P_max'
        'high_participation_nodes': nodes with P(v) > mean + 2*std
        'participation_degree_corr': Pearson r between P(v) and degree
        'participation_strength_corr': Pearson r between P(v) and strength (if weighted)
    """
    nodes = sorted(G.nodes())
    P = census['P']
    P_arr = np.array([P.get(v, 0) for v in nodes], dtype=np.float64)
    deg_arr = np.array([G.degree(v) for v in nodes], dtype=np.float64)

    result = {
        'P_mean': float(P_arr.mean()),
        'P_std': float(P_arr.std()),
        'P_max': float(P_arr.max()),
        'P_median': float(np.median(P_arr)),
    }

    # High-participation nodes
    threshold = result['P_mean'] + 2 * result['P_std']
    high_nodes = [int(nodes[i]) for i in range(len(nodes)) if P_arr[i] > threshold]
    result['n_high_participation'] = len(high_nodes)
    result['high_participation_frac'] = len(high_nodes) / len(nodes)

    # Correlation with degree (Sizemore Fig. 3a: r=0.957)
    if P_arr.std() > 0 and deg_arr.std() > 0:
        r = np.corrcoef(P_arr, deg_arr)[0, 1]
        result['participation_degree_corr'] = float(r)
    else:
        result['participation_degree_corr'] = 0.0

    # Participation Gini coefficient
    sorted_P = np.sort(P_arr)
    n = len(sorted_P)
    if sorted_P.sum() > 0:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_P) - (n + 1) * np.sum(sorted_P)) / (n * np.sum(sorted_P))
        result['participation_gini'] = float(gini)
    else:
        result['participation_gini'] = 0.0

    return result


# ═══════════════════════════════════════════════════════════════════
# Histogram analysis: bimodality (Sizemore Fig. 2a)
# ═══════════════════════════════════════════════════════════════════

def histogram_analysis(h):
    """Analyze the maximal clique histogram h(k).

    Returns dict with:
        'k_range': (min_k, max_k)
        'peak_k': k with most maximal cliques
        'distribution': list of (k, count) sorted by k
        'bimodality_ratio': ratio of valley to peaks (< 1 = bimodal)
    """
    if not h:
        return {'k_range': (0, 0), 'peak_k': 0, 'distribution': [], 'bimodality_ratio': None}

    ks = sorted(h.keys())
    counts = [h[k] for k in ks]

    result = {
        'k_range': (min(ks), max(ks)),
        'peak_k': ks[np.argmax(counts)],
        'distribution': [(k, h[k]) for k in ks],
        'total_cliques': sum(counts),
    }

    # Simple bimodality check: fill in gaps, find local maxima manually
    if len(ks) >= 3:
        # Build contiguous array from min_k to max_k
        k_min, k_max = min(ks), max(ks)
        full_ks = list(range(k_min, k_max + 1))
        arr = np.array([h.get(k, 0) for k in full_ks], dtype=float)

        # Find local maxima (higher than both neighbors)
        local_max = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                local_max.append(i)
        # Also check endpoints
        if len(arr) >= 2:
            if arr[0] > arr[1]:
                local_max.insert(0, 0)
            if arr[-1] > arr[-2]:
                local_max.append(len(arr) - 1)

        if len(local_max) >= 2:
            # Sort by height, take top 2
            local_max.sort(key=lambda i: arr[i], reverse=True)
            p1, p2 = sorted(local_max[:2])
            valley = arr[p1:p2+1].min()
            smaller_peak = min(arr[p1], arr[p2])
            result['bimodality_ratio'] = float(valley / smaller_peak) if smaller_peak > 0 else None
            result['peak1_k'] = int(full_ks[p1])
            result['peak2_k'] = int(full_ks[p2])
            result['is_bimodal'] = result['bimodality_ratio'] < 0.5 if result['bimodality_ratio'] is not None else False
        else:
            result['bimodality_ratio'] = None
            result['is_bimodal'] = False
    else:
        result['bimodality_ratio'] = None
        result['is_bimodal'] = False

    return result


# ═══════════════════════════════════════════════════════════════════
# Null model comparison (degree-preserving randomization)
# ═══════════════════════════════════════════════════════════════════

def null_clique_census(G, n_null=10, timeout_per=60, verbose=True):
    """Run clique census on degree-preserving random rewirings.

    Uses nx.double_edge_swap to create degree-preserving nulls.

    Returns dict with:
        'null_omega': list of omega values
        'null_h_mean': dict[k] -> mean count across nulls
        'null_h_std': dict[k] -> std across nulls
        'null_P_mean': mean total participation across nulls
    """
    n_edges = G.number_of_edges()
    n_swaps = 10 * n_edges  # standard: 10x edges

    null_omegas = []
    null_h_all = []

    for i in range(n_null):
        if verbose:
            print(f"  Null {i+1}/{n_null}...", end=" ", flush=True)
        G_null = G.copy()
        try:
            nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 10)
        except nx.NetworkXAlgorithmError:
            pass  # couldn't complete all swaps

        census = clique_census(G_null, timeout=timeout_per, verbose=False)
        null_omegas.append(census['omega'])
        null_h_all.append(census['h'])
        if verbose:
            print(f"omega={census['omega']}, {census['n_maximal']:,} cliques")

    # Aggregate histograms
    all_ks = set()
    for h in null_h_all:
        all_ks.update(h.keys())

    null_h_mean = {}
    null_h_std = {}
    for k in sorted(all_ks):
        vals = [h.get(k, 0) for h in null_h_all]
        null_h_mean[k] = float(np.mean(vals))
        null_h_std[k] = float(np.std(vals))

    return {
        'null_omega_mean': float(np.mean(null_omegas)),
        'null_omega_std': float(np.std(null_omegas)),
        'null_omega_all': null_omegas,
        'null_h_mean': null_h_mean,
        'null_h_std': null_h_std,
        'n_null': n_null,
    }


# ═══════════════════════════════════════════════════════════════════
# Enrichment: compare real h(k) to null h(k)
# ═══════════════════════════════════════════════════════════════════

def enrichment(real_h, null_h_mean, null_h_std):
    """Compute per-k enrichment ratios and z-scores.

    Returns list of dicts, one per k.
    """
    all_ks = sorted(set(list(real_h.keys()) + list(null_h_mean.keys())))
    results = []
    for k in all_ks:
        real = real_h.get(k, 0)
        null_m = null_h_mean.get(k, 0)
        null_s = null_h_std.get(k, 0)
        ratio = real / null_m if null_m > 0 else float('inf') if real > 0 else 1.0
        z = (real - null_m) / null_s if null_s > 0 else float('nan')
        results.append({
            'k': k,
            'real': real,
            'null_mean': null_m,
            'null_std': null_s,
            'ratio': ratio,
            'z': z,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# I/O: load graph
# ═══════════════════════════════════════════════════════════════════

def load_graph(path):
    """Load graph from .npz or .gml file."""
    if path.endswith('.gml'):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
    else:
        d = np.load(path, allow_pickle=True)
        if 'edges' in d:
            edges = d['edges']
        elif 'edge_list' in d:
            edges = d['edge_list']
        else:
            raise ValueError(f"No edges found in {path}. Keys: {list(d.keys())}")

        if 'n_total' in d:
            n = int(d['n_total'])
        else:
            n = int(edges.max()) + 1
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges.tolist())

    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        print(f"  Removed {len(isolates)} isolates")

    return G


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sizemore-style clique census")
    parser.add_argument("--graph", required=True, help="Path to .npz graph file")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout for clique enumeration (seconds)")
    parser.add_argument("--n-null", type=int, default=10, help="Number of degree-preserving nulls")
    parser.add_argument("--null-timeout", type=float, default=60, help="Timeout per null model")
    parser.add_argument("--skip-nulls", action="store_true", help="Skip null model comparison")
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--density-threshold", type=float, default=None,
                        help="Threshold to target density (remove weakest edges)")
    args = parser.parse_args()

    print(f"Loading graph: {args.graph}")
    G = load_graph(args.graph)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    print(f"  N={n:,}, E={m:,}, density={density:.4f}")

    # Optional density thresholding (for fair comparison with Sizemore's rho=0.25)
    if args.density_threshold and density > args.density_threshold:
        print(f"  Thresholding to density {args.density_threshold}...")
        # For unweighted graphs, random edge removal to target density
        target_edges = int(args.density_threshold * n * (n - 1) / 2)
        if target_edges < m:
            edges = list(G.edges())
            np.random.seed(42)
            remove_idx = np.random.choice(len(edges), size=m - target_edges, replace=False)
            for i in remove_idx:
                G.remove_edge(*edges[i])
            # Remove new isolates
            isolates = list(nx.isolates(G))
            G.remove_nodes_from(isolates)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            density = nx.density(G)
            print(f"  After threshold: N={n:,}, E={m:,}, density={density:.4f}")

    # 1. Clique census
    print(f"\n{'='*60}")
    print(f"CLIQUE CENSUS")
    print(f"{'='*60}")
    census = clique_census(G, timeout=args.timeout, verbose=True)
    print(f"\n  Clique number (omega): {census['omega']}")
    print(f"  Total maximal cliques: {census['n_maximal']:,}")
    if census['timed_out']:
        print(f"  WARNING: enumeration timed out, results are partial")

    # 2. Histogram
    print(f"\n  Maximal clique size distribution h(k):")
    print(f"  {'k':>5s}  {'count':>12s}  {'cumulative':>12s}")
    print(f"  {'---':>5s}  {'-------':>12s}  {'----------':>12s}")
    h_analysis = histogram_analysis(census['h'])
    cum = 0
    for k, count in h_analysis['distribution']:
        cum += count
        print(f"  {k:5d}  {count:12,}  {cum:12,}")

    print(f"\n  Peak at k={h_analysis['peak_k']}")
    if h_analysis.get('is_bimodal'):
        print(f"  BIMODAL: peaks at k={h_analysis['peak1_k']} and k={h_analysis['peak2_k']}")
        print(f"  Valley/peak ratio: {h_analysis['bimodality_ratio']:.3f}")
    else:
        print(f"  Unimodal (or insufficient data for bimodality test)")

    # 3. Participation
    print(f"\n  Node participation statistics:")
    p_stats = participation_stats(census, G)
    print(f"    P(v) mean:   {p_stats['P_mean']:.1f}")
    print(f"    P(v) median: {p_stats['P_median']:.1f}")
    print(f"    P(v) max:    {p_stats['P_max']:.0f}")
    print(f"    P(v) std:    {p_stats['P_std']:.1f}")
    print(f"    P(v) Gini:   {p_stats['participation_gini']:.3f}")
    print(f"    P-degree corr (r): {p_stats['participation_degree_corr']:.3f}")
    print(f"    High-participation nodes (>mean+2std): {p_stats['n_high_participation']}"
          f" ({p_stats['high_participation_frac']:.1%})")

    # 4. Null models
    null_results = None
    enrich = None
    if not args.skip_nulls:
        print(f"\n{'='*60}")
        print(f"NULL MODEL COMPARISON ({args.n_null} degree-preserving rewirings)")
        print(f"{'='*60}")
        null_results = null_clique_census(G, n_null=args.n_null,
                                          timeout_per=args.null_timeout, verbose=True)
        print(f"\n  Null omega: {null_results['null_omega_mean']:.1f} +/- {null_results['null_omega_std']:.1f}")
        print(f"  Real omega: {census['omega']}")

        # Enrichment
        enrich = enrichment(census['h'], null_results['null_h_mean'], null_results['null_h_std'])
        print(f"\n  Per-k enrichment (real / null_mean):")
        print(f"  {'k':>5s}  {'real':>10s}  {'null_mean':>10s}  {'ratio':>8s}  {'z':>8s}")
        for e in enrich:
            z_str = f"{e['z']:.1f}" if not np.isnan(e['z']) else "NaN"
            print(f"  {e['k']:5d}  {e['real']:10,}  {e['null_mean']:10.0f}"
                  f"  {e['ratio']:8.2f}  {z_str:>8s}")

    # 5. Save JSON
    if args.json:
        output = {
            'graph': args.graph,
            'n_nodes': n,
            'n_edges': m,
            'density': density,
            'omega': census['omega'],
            'n_maximal_cliques': census['n_maximal'],
            'timed_out': census['timed_out'],
            'elapsed': census['elapsed'],
            'h': {str(k): v for k, v in census['h'].items()},
            'histogram_analysis': {
                'k_range': h_analysis['k_range'],
                'peak_k': h_analysis['peak_k'],
                'is_bimodal': h_analysis.get('is_bimodal', False),
                'bimodality_ratio': h_analysis.get('bimodality_ratio'),
            },
            'participation': p_stats,
        }
        if null_results:
            output['null'] = {
                'n_null': null_results['n_null'],
                'omega_mean': null_results['null_omega_mean'],
                'omega_std': null_results['null_omega_std'],
                'omega_all': null_results['null_omega_all'],
            }
        if enrich:
            output['enrichment'] = enrich

        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
