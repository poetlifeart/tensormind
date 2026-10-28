#!/usr/bin/env python3
"""
complexity_orbit_entropy.py — measure 1 of run_complexity.py (automorphism
group and edge-orbit entropy), on its own.

Automorphism group Aut(G) of the whole loaded graph (every node of the file,
not only its giant component), computed by nauty through pynauty.autgrp():
its order |Aut(G)| = grpsize1 x 10^grpsize2, its generators and its vertex
orbits.  The edge orbits are the classes of the edge set under the generators
(union-find: edge {a,b} is joined to {g(a),g(b)} for every generator g), and
the edge-orbit entropy is the Shannon entropy of the orbit sizes, in bits:

    p_i  = |E_i| / m              (E_i = i-th edge orbit, m = number of edges)
    H    = -sum_i p_i log2 p_i    edge_orbit_entropy_bits
    Hmax = log2 m                 (every edge in its own orbit)
    H / Hmax                      edge_orbit_entropy_normalized

The orbit entropy (topological information content) is Mowshowitz (1968),
Bull. Math. Biophys. 30:175, on vertex orbits; the edge-orbit form is Trucco
(1956), Bull. Math. Biophys. 18:129.  |Aut(G)| of real networks via nauty as
in MacArthur, Sanchez-Garcia & Anderson (2008), Discrete Appl. Math. 156:3525.

Self-contained: needs numpy, networkx and pynauty, and does not import
run_complexity.py.  load_graph_npz() and edge_orbit_entropy() below are copied
from it unchanged; load_graph() is the graph-loading branch of its main().
With the same --graph the numbers are the same as the 'orbit_entropy' block of
run_complexity.py's JSON.  run_complexity.py's --skip-orbits / --skip-eigen
only switch measures off, so they are not carried over.

Usage:
    python complexity_orbit_entropy.py --graph G.npz
    python complexity_orbit_entropy.py --graph G.npz --json orbits.json
"""

import argparse
import json
import time
import sys
import numpy as np
import networkx as nx
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════
# Graph loading  (copied from run_complexity.py)
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


def load_graph(path):
    """The graph-loading branch of run_complexity.py main(), as a function
    (.gml read by networkx, anything else by load_graph_npz)."""
    if path.endswith('.gml'):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
        n = G.number_of_nodes()
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    else:
        G, n1, n2, n3 = load_graph_npz(path)
    return G, n1, n2, n3


# ═══════════════════════════════════════════════════════════════════
# 1. Edge orbit entropy (Mowshowitz 1968) via pynauty
#    (copied from run_complexity.py)
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
# Main  (the measure-1 part of run_complexity.py main())
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Automorphism group and edge-orbit entropy (measure 1 of run_complexity.py)")
    ap.add_argument('--graph', type=str, required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--json', type=str, default=None, help='Save results to JSON')
    args = ap.parse_args()

    print(f"\nLoading {args.graph}...")
    t0 = time.time()
    G, n1, n2, n3 = load_graph(args.graph)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  {n:,} nodes, {m:,} edges ({time.time()-t0:.1f}s)")

    results = {'graph': args.graph, 'nodes': n, 'edges': m}

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

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
