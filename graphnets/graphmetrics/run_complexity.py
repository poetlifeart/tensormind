#!/usr/bin/env python3
"""
Graph complexity measures on graph_brain_mild.npz.
All core algorithms from external libraries (NetworkX, pynauty, scipy).
This script is glue code only.

Measures:
  1. Edge orbit entropy (Mowshowitz 1968) — pynauty for automorphisms
  2. Estrada index — nx.estrada_index()
  3. Graph energy (Gutman 1978) — nx.adjacency_spectrum()
  4. Kirchhoff index (Klein & Randic 1993) — nx.effective_graph_resistance()
  5. Offdiagonal complexity (Claussen 2007) — degree-pair entropy

Usage:
  python run_complexity.py --graph ../graph_brain_mild.npz
  python run_complexity.py --graph ../graph_brain_mild.npz --skip-orbits   # skip slow orbit computation
  python run_complexity.py --graph ../graph_brain_mild.npz --skip-eigen    # skip dense eigendecomposition
"""

import argparse
import json
import time
import sys
import numpy as np
import scipy.sparse
import networkx as nx
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════
# Graph loading (from our .npz tripartite format)
# ═══════════════════════════════════════════════════════════════════

def load_graph_npz(path):
    """Load graph from .npz — supports both mask-based and edge-list formats."""
    data = np.load(path)
    if 'n1' in data:
        n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        n = n1 + n2 + n3
    else:
        n = int(data['n_total'])
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    if 'edges' in data:
        G.add_edges_from(data['edges'].tolist())
    else:
        off2, off3 = n1, n1 + n2
        for key, o1, o2 in [('mask_12', 0, off2), ('mask_13', 0, off3), ('mask_23', off2, off3)]:
            rs, cs = np.where(data[key] > 0)
            for r, c in zip(rs, cs):
                G.add_edge(o1 + c, o2 + r)

    return G, n1, n2, n3


# ═══════════════════════════════════════════════════════════════════
# 1. Edge orbit entropy (Mowshowitz 1968) via pynauty
# ═══════════════════════════════════════════════════════════════════

def edge_orbit_entropy(G):
    """Compute edge orbit entropy using pynauty automorphism generators.

    Steps (all from pynauty/nauty):
      1. pynauty.autgrp() → vertex orbits + generators
      2. Apply generators to edge set → edge orbits
      3. Shannon entropy of edge orbit sizes
    """
    from pynauty import Graph as NautyGraph, autgrp

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Build pynauty graph — pynauty needs integer-indexed adjacency dict
    # Remap nodes to 0..n-1 if not already
    node_list = sorted(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(node_list)}

    adj_dict = {}
    for v in node_list:
        adj_dict[node_to_idx[v]] = [node_to_idx[u] for u in G.neighbors(v)]

    ng = NautyGraph(n, adjacency_dict=adj_dict)

    print("  Computing automorphism group (pynauty/nauty)...")
    t0 = time.time()
    generators, grpsize1, grpsize2, orbits, num_orbits = autgrp(ng)
    dt = time.time() - t0
    print(f"  Vertex orbits: {num_orbits} ({dt:.1f}s)")
    print(f"  |Aut(G)| = {grpsize1} x 10^{grpsize2}")

    # Edge orbits: two edges are in the same orbit if some generator maps one to the other.
    # Use union-find on edges, applying each generator.
    edges = list(G.edges())
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        a, b = node_to_idx[u], node_to_idx[v]
        key = (min(a, b), max(a, b))
        edge_to_idx[key] = i

    # Union-find
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    print("  Computing edge orbits from generators...")
    t0 = time.time()
    for gen in generators:
        for i, (u, v) in enumerate(edges):
            a, b = node_to_idx[u], node_to_idx[v]
            # Apply generator: maps vertex a → gen[a], b → gen[b]
            ga, gb = gen[a], gen[b]
            mapped_key = (min(ga, gb), max(ga, gb))
            if mapped_key in edge_to_idx:
                j = edge_to_idx[mapped_key]
                union(i, j)

    # Count orbit sizes
    orbit_map = Counter(find(i) for i in range(m))
    n_edge_orbits = len(orbit_map)
    orbit_sizes = np.array(list(orbit_map.values()), dtype=float)
    dt = time.time() - t0
    print(f"  Edge orbits: {n_edge_orbits} ({dt:.1f}s)")

    # Shannon entropy
    probs = orbit_sizes / orbit_sizes.sum()
    H = -np.sum(probs * np.log2(probs))
    H_max = np.log2(m)
    H_norm = H / H_max if H_max > 0 else 0

    return {
        'vertex_orbits': num_orbits,
        'edge_orbits': n_edge_orbits,
        'aut_group_size': f"{grpsize1}e{grpsize2}",
        'edge_orbit_entropy_bits': float(H),
        'edge_orbit_entropy_max': float(H_max),
        'edge_orbit_entropy_normalized': float(H_norm),
        'largest_orbit': int(orbit_sizes.max()),
        'smallest_orbit': int(orbit_sizes.min()),
    }


# ═══════════════════════════════════════════════════════════════════
# 2. Estrada index — NetworkX built-in
# ═══════════════════════════════════════════════════════════════════

def compute_estrada(G, eigenvalues=None):
    """Estrada index = sum(exp(lambda_i)) for adjacency eigenvalues.
    Uses pre-computed eigenvalues if available, otherwise nx built-in."""
    if eigenvalues is not None:
        EE = np.sum(np.exp(eigenvalues))
    else:
        EE = nx.estrada_index(G)
    return {
        'estrada_index': float(EE),
        'estrada_index_log10': float(np.log10(EE)) if EE > 0 else float('-inf'),
        'estrada_index_per_node': float(EE / G.number_of_nodes()),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Graph energy (Gutman 1978) — nx.adjacency_spectrum()
# ═══════════════════════════════════════════════════════════════════

def compute_graph_energy(G, eigenvalues=None):
    """Graph energy = sum(|lambda_i|) for adjacency eigenvalues."""
    if eigenvalues is None:
        eigenvalues = np.real(nx.adjacency_spectrum(G))
    energy = np.sum(np.abs(eigenvalues))
    return {
        'graph_energy': float(energy),
        'graph_energy_per_node': float(energy / G.number_of_nodes()),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Kirchhoff index — nx.effective_graph_resistance()
# ═══════════════════════════════════════════════════════════════════

def compute_kirchhoff(G, lap_eigenvalues=None):
    """Kirchhoff index = n * sum(1/lambda_i) for nonzero Laplacian eigenvalues."""
    n = G.number_of_nodes()
    if lap_eigenvalues is None:
        R = nx.effective_graph_resistance(G)
    else:
        nonzero = lap_eigenvalues[lap_eigenvalues > 1e-10]
        R = n * np.sum(1.0 / nonzero)
    return {
        'kirchhoff_index': float(R),
        'kirchhoff_index_per_n2': float(R / (n * n)),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Offdiagonal complexity (Claussen 2007)
# ═══════════════════════════════════════════════════════════════════

def compute_offdiagonal(G):
    """Claussen (2007) Physica A 375:365-373: off-diagonal complexity.

    OdC = -sum_{k=0}^{k_max} a_k * ln(a_k)
    where a_k is the fraction of ALL edges connecting nodes whose degrees
    differ by exactly k.  Sum and normalization include k=0 (Eq. 5).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2.0 * m / (n * (n - 1))

    diff_counts = Counter()
    for u, v in G.edges():
        k = abs(G.degree(u) - G.degree(v))
        diff_counts[k] += 1

    probs = np.array(list(diff_counts.values()), dtype=float) / m
    OdC = float(-np.sum(probs * np.log(probs + 1e-300)))

    degree_pairs = Counter()
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        degree_pairs[(min(du, dv), max(du, dv))] += 1
    total = sum(degree_pairs.values())
    all_probs = np.array(list(degree_pairs.values()), dtype=float) / total
    H_all = float(-np.sum(all_probs * np.log(all_probs)))

    deg_counts = Counter(dict(G.degree()).values())
    deg_total = sum(deg_counts.values())
    deg_probs = np.array(list(deg_counts.values()), dtype=float) / deg_total
    H_deg = float(-np.sum(deg_probs * np.log(deg_probs)))

    n_diag_edges = sum(1 for u, v in G.edges() if G.degree(u) == G.degree(v))

    return {
        'offdiagonal_complexity': OdC,
        'full_degree_pair_entropy': H_all,
        'degree_distribution_entropy': H_deg,
        'n_offdiag_edges': m - diff_counts.get(0, 0),
        'n_diag_edges': n_diag_edges,
        'unique_degree_diffs': len(diff_counts),
        'unique_degree_pairs': len(degree_pairs),
        'unique_degrees': len(deg_counts),
        'density': float(density),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Graph complexity measures')
    # was: default='../graph_brain_mild.npz' -- relative to the CWD, resolved nowhere useful.
    parser.add_argument('--graph', type=str, required=True)
    parser.add_argument('--skip-orbits', action='store_true', help='Skip orbit entropy (pynauty)')
    parser.add_argument('--skip-eigen', action='store_true', help='Skip dense eigendecomposition')
    parser.add_argument('--json', type=str, default=None, help='Save results to JSON')
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 70)
    print("GRAPH COMPLEXITY MEASURES")
    print("=" * 70)

    # Load graph
    print(f"\nLoading {args.graph}...")
    t0 = time.time()
    if args.graph.endswith('.gml'):
        G = nx.read_gml(args.graph)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
        n = G.number_of_nodes()
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    else:
        G, n1, n2, n3 = load_graph_npz(args.graph)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  {n:,} nodes, {m:,} edges ({time.time()-t0:.1f}s)")
    print(f"  Density: {2*m/(n*(n-1)):.6f}")
    print(f"  Avg degree: {2*m/n:.1f}")

    results = {'graph': args.graph, 'nodes': n, 'edges': m}

    # ── 1. Edge orbit entropy ──
    if not args.skip_orbits:
        print(f"\n{'─'*70}")
        print("1. EDGE ORBIT ENTROPY (vertex orbits: Mowshowitz 1968; edge extension, via pynauty/nauty)")
        print(f"{'─'*70}")
        t0 = time.time()
        try:
            orb = edge_orbit_entropy(G)
            results['orbit_entropy'] = orb
            print(f"\n  Vertex orbits:       {orb['vertex_orbits']:,}")
            print(f"  Edge orbits:         {orb['edge_orbits']:,}")
            print(f"  |Aut(G)|:            {orb['aut_group_size']}")
            print(f"  Orbit entropy:       {orb['edge_orbit_entropy_bits']:.4f} bits")
            print(f"  Max possible:        {orb['edge_orbit_entropy_max']:.4f} bits")
            print(f"  Normalized:          {orb['edge_orbit_entropy_normalized']:.6f}")
            print(f"  Largest orbit:       {orb['largest_orbit']:,} edges")
            print(f"  Smallest orbit:      {orb['smallest_orbit']:,} edges")
            print(f"  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results['orbit_entropy'] = {'error': str(e)}
    else:
        print("\n[Skipping orbit entropy]")

    # ── Shared eigendecomposition for measures 2-4 ──
    adj_eigenvalues = None
    lap_eigenvalues = None

    if not args.skip_eigen:
        print(f"\n{'─'*70}")
        print("EIGENDECOMPOSITION (shared by Estrada, Energy, Kirchhoff)")
        print(f"{'─'*70}")

        # Adjacency eigenvalues (for Estrada + Energy)
        print("  Computing adjacency eigenvalues (dense, all n)...")
        t0 = time.time()
        try:
            adj_dense = nx.to_numpy_array(G, dtype=np.float64)
            adj_eigenvalues = np.linalg.eigvalsh(adj_dense)
            print(f"  Adjacency eigenvalues: done ({time.time()-t0:.1f}s)")
            print(f"    range: [{adj_eigenvalues.min():.4f}, {adj_eigenvalues.max():.4f}]")
        except Exception as e:
            print(f"  FAILED: {e}")

        # Laplacian eigenvalues (for Kirchhoff)
        print("  Computing Laplacian eigenvalues (dense, all n)...")
        t0 = time.time()
        try:
            L = nx.laplacian_matrix(G).toarray().astype(np.float64)
            lap_eigenvalues = np.linalg.eigvalsh(L)
            print(f"  Laplacian eigenvalues: done ({time.time()-t0:.1f}s)")
            print(f"    range: [{lap_eigenvalues.min():.6f}, {lap_eigenvalues.max():.4f}]")
            del L  # free memory
        except Exception as e:
            print(f"  FAILED: {e}")

        # Free dense adjacency
        try:
            del adj_dense
        except NameError:
            pass

        # ── 2. Estrada index ──
        print(f"\n{'─'*70}")
        print("2. ESTRADA INDEX (Estrada 2000)")
        print(f"{'─'*70}")
        t0 = time.time()
        try:
            est = compute_estrada(G, eigenvalues=adj_eigenvalues)
            results['estrada'] = est
            print(f"  Estrada index:       {est['estrada_index']:.6e}")
            print(f"  log10(EE):           {est['estrada_index_log10']:.4f}")
            print(f"  EE / n:              {est['estrada_index_per_node']:.6e}")
            print(f"  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results['estrada'] = {'error': str(e)}

        # ── 3. Graph energy ──
        print(f"\n{'─'*70}")
        print("3. GRAPH ENERGY (Gutman 1978)")
        print(f"{'─'*70}")
        t0 = time.time()
        try:
            ener = compute_graph_energy(G, eigenvalues=adj_eigenvalues)
            results['graph_energy'] = ener
            print(f"  Graph energy:        {ener['graph_energy']:.4f}")
            print(f"  Energy / n:          {ener['graph_energy_per_node']:.4f}")
            print(f"  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results['graph_energy'] = {'error': str(e)}

        # ── 4. Kirchhoff index ──
        print(f"\n{'─'*70}")
        print("4. KIRCHHOFF INDEX (Klein & Randic 1993)")
        print(f"{'─'*70}")
        t0 = time.time()
        try:
            kirch = compute_kirchhoff(G, lap_eigenvalues=lap_eigenvalues)
            results['kirchhoff'] = kirch
            print(f"  Kirchhoff index:     {kirch['kirchhoff_index']:.4f}")
            print(f"  R / n^2:             {kirch['kirchhoff_index_per_n2']:.8f}")
            print(f"  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results['kirchhoff'] = {'error': str(e)}
    else:
        print("\n[Skipping eigendecomposition-based measures]")

    # ── 5. Offdiagonal complexity ──
    print(f"\n{'─'*70}")
    print("5. OFFDIAGONAL COMPLEXITY (Claussen 2007)")
    print(f"{'─'*70}")
    t0 = time.time()
    try:
        odc = compute_offdiagonal(G)
        results['offdiagonal'] = odc
        print(f"  Offdiag complexity:  {odc['offdiagonal_complexity']:.4f}")
        print(f"  Degree-pair entropy: {odc['full_degree_pair_entropy']:.4f} nats")
        print(f"  Degree entropy:      {odc['degree_distribution_entropy']:.4f} nats")
        print(f"  Unique degree pairs: {odc['unique_degree_pairs']:,}")
        print(f"  Unique degrees:      {odc['unique_degrees']}")
        print(f"  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        results['offdiagonal'] = {'error': str(e)}

    # ── Summary ──
    dt_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"DONE ({dt_total:.1f}s total)")
    print(f"{'='*70}")

    if args.json:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"Results saved to {args.json}")


if __name__ == '__main__':
    main()
