#!/usr/bin/env python3
"""
Combinatorial graph metrics (Grolmusz-style), matching the exact definitions
from Szalkai, Varga & Grolmusz (2017, 2021).

Metrics computed:
  AdjLMaxDivD        — lambda_max(A) / avg_degree
  PGEigengap         — lambda_1(P) - lambda_2(P), P = D^{-1}A (transition matrix)
  HoffmanBound       — 1 + lambda_max / |lambda_min|
  LogAbsSpanningForestN — log(spanning tree count) / n via Kirchhoff
  MinVertexCover     — fractional (LP relaxation) + 2-approx (networkx)
  MinCutBalDivSum    — spectral balanced bisection (Fiedler) + Fiedler value
  MaxMatching        — maximum cardinality matching
  MaxFracMatching    — fractional matching (LP relaxation)
  MinSpanningForest  — MST weight (Kruskal, unweighted = n-1)

Supports:
  - .gml / .graphml files
  - .npz with n_total + edges (quotient graphs)
  - .npz with n1, n2, n3 + masks (tripartite parent graphs)

References:
  Szalkai, Varga, Grolmusz (2021) "The Graph of Our Mind" arXiv:1603.00904
  Szalkai, Varga, Grolmusz (2018) "Comparing Advanced Graph-Theoretical
    Parameters of the Connectomes of the Lobes" arXiv:1709.04974
"""

from __future__ import annotations
import argparse
import json, os
import time

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import linprog
import networkx as nx


def load_graph(path: str) -> nx.Graph:
    if path.endswith('.gml') or path.endswith('.graphml'):
        G = nx.read_graphml(path) if path.endswith('.graphml') else nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
    elif path.endswith('.npz'):
        d = np.load(path)
        if 'n_total' in d:
            n = int(d['n_total'])
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from((int(u), int(v)) for u, v in d['edges'])
        else:
            n1, n2, n3 = int(d['n1']), int(d['n2']), int(d['n3'])
            n = n1 + n2 + n3
            if 'edges' in d:
                G = nx.Graph()
                G.add_nodes_from(range(n))
                G.add_edges_from((int(u), int(v)) for u, v in d['edges'])
            else:
                G = nx.Graph()
                G.add_nodes_from(range(n))
                rs, cs = np.where(d['mask_12'])
                G.add_edges_from(zip(rs.astype(int), (cs + n1).astype(int)))
                rs, cs = np.where(d['mask_13'])
                G.add_edges_from(zip((rs + n1 + n2).astype(int), cs.astype(int)))
                rs, cs = np.where(d['mask_23'])
                G.add_edges_from(zip((rs + n1 + n2).astype(int), (cs + n1).astype(int)))
    else:
        raise ValueError(f"Unknown format: {path}")
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# ── 1. AdjLMaxDivD ──────────────────────────────────────────────────────

def adj_lmax_div_d(G: nx.Graph) -> dict:
    """Largest adjacency eigenvalue / average degree."""
    t0 = time.time()
    A = nx.adjacency_matrix(G, dtype=float)
    n = G.number_of_nodes()
    deg_avg = 2 * G.number_of_edges() / n
    lam_max = scipy.sparse.linalg.eigsh(A, k=1, which='LA',
                                         return_eigenvectors=False)[0]
    return {
        'lambda_max': float(lam_max),
        'avg_degree': float(deg_avg),
        'value': float(lam_max) / deg_avg if deg_avg > 0 else 0,
        'time': time.time() - t0
    }


# ── 2. PGEigengap (transition matrix) ──────────────────────────────────

def pg_eigengap(G: nx.Graph) -> dict:
    """Eigengap of transition matrix P = D^{-1}A.
    PGEigengap = lambda_1(P) - lambda_2(P).
    lambda_1(P) = 1 always for connected graphs."""
    t0 = time.time()
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, dtype=float)
    D_inv = scipy.sparse.diags([1.0 / G.degree(i) if G.degree(i) > 0 else 0
                                 for i in range(n)])
    P = D_inv @ A
    eigs_P = scipy.sparse.linalg.eigs(P, k=min(6, n - 2), which='LR',
                                       return_eigenvectors=False)
    eigs_P = np.sort(np.real(eigs_P))[::-1]
    gap = float(eigs_P[0] - eigs_P[1]) if len(eigs_P) > 1 else 0
    return {
        'top_eigenvalues': eigs_P[:5].tolist(),
        'eigengap': gap,
        'time': time.time() - t0
    }


# ── 3. HoffmanBound ────────────────────────────────────────────────────

def hoffman_bound(G: nx.Graph) -> dict:
    """Hoffman chromatic number lower bound: 1 + lambda_max / |lambda_min|."""
    t0 = time.time()
    A = nx.adjacency_matrix(G, dtype=float)
    lam_max = scipy.sparse.linalg.eigsh(A, k=1, which='LA',
                                         return_eigenvectors=False)[0]
    lam_min = scipy.sparse.linalg.eigsh(A, k=1, which='SA',
                                         return_eigenvectors=False)[0]
    bound = 1 + lam_max / abs(lam_min) if abs(lam_min) > 1e-10 else float('inf')
    return {
        'hoffman_bound': float(bound),
        'lambda_max': float(lam_max),
        'lambda_min': float(lam_min),
        'time': time.time() - t0
    }


# ── 4. LogAbsSpanningForestN (Kirchhoff) ───────────────────────────────

def log_spanning_forest(G: nx.Graph) -> dict:
    """Log of spanning tree count / n, via Kirchhoff's theorem."""
    t0 = time.time()
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).astype(float)
    if n <= 2000:
        L_dense = L.toarray() if scipy.sparse.issparse(L) else L
        eigs = np.linalg.eigvalsh(L_dense)
    else:
        eigs = scipy.sparse.linalg.eigsh(L, k=min(n - 1, 500),
                                          which='SM',
                                          return_eigenvectors=False)
    eigs = np.sort(eigs)
    nonzero = eigs[eigs > 1e-10]
    if len(nonzero) > 0:
        log_count = np.sum(np.log(nonzero)) - np.log(n)
        log_count_per_n = log_count / n
    else:
        log_count = 0
        log_count_per_n = 0
    return {
        'log_spanning_tree_count': float(log_count),
        'log_spanning_tree_per_n': float(log_count_per_n),
        'n_nonzero_eigs': len(nonzero),
        'note': 'partial' if n > 2000 else 'exact',
        'time': time.time() - t0
    }


# ── 5. MinVertexCover (fractional LP) ──────────────────────────────────

def min_vertex_cover_frac(G: nx.Graph) -> dict:
    """Fractional minimum vertex cover via LP relaxation.
    min sum(x_i) s.t. x_u + x_v >= 1 for each edge, 0 <= x_i <= 1."""
    t0 = time.time()
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    c = np.ones(n)
    row_idx = np.concatenate([np.arange(m), np.arange(m)])
    col_idx = np.array([u for u, v in edges] + [v for u, v in edges])
    data = -np.ones(2 * m)
    A_ub = scipy.sparse.csc_matrix((data, (row_idx, col_idx)), shape=(m, n))
    b_ub = -np.ones(m)
    bounds = [(0, 1)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return {
        'size': float(res.fun) if res.success else -1,
        'normalized': float(res.fun) / n if res.success else -1,
        'time': time.time() - t0
    }


# ── 6. MinVertexCover (2-approximation) ───────────────────────────────

def min_vertex_cover_approx(G: nx.Graph) -> dict:
    """2-approximation minimum vertex cover (networkx greedy matching)."""
    t0 = time.time()
    vc = nx.approximation.min_weighted_vertex_cover(G)
    n = G.number_of_nodes()
    return {
        'size': len(vc),
        'normalized': len(vc) / n,
        'note': '2-approx upper bound',
        'time': time.time() - t0
    }


# ── 7. MinCutBalDivSum (spectral balanced bisection) ──────────────────

def min_cut_bal(G: nx.Graph) -> dict:
    """Approximate balanced bisection via Fiedler vector (spectral).
    MinCutBalDivSum = cut_size / num_edges."""
    t0 = time.time()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    L = nx.laplacian_matrix(G).astype(float)
    _, vecs = scipy.sparse.linalg.eigsh(L, k=2, which='SM')
    fiedler = vecs[:, 1]

    median_val = np.median(fiedler)
    part_a = set(i for i in range(n) if fiedler[i] <= median_val)
    part_b = set(range(n)) - part_a

    cut = sum(1 for u, v in G.edges() if (u in part_a) != (v in part_a))
    fiedler_val = float(np.sort(scipy.sparse.linalg.eigsh(
        L, k=2, which='SM', return_eigenvectors=False))[1])

    return {
        'cut_size': cut,
        'normalized': cut / m if m > 0 else 0,
        'partition_sizes': [len(part_a), len(part_b)],
        'fiedler_value': fiedler_val,
        'note': 'spectral approx (Fiedler bisection)',
        'time': time.time() - t0
    }


# ── 8. MaxMatching ─────────────────────────────────────────────────────

def max_matching(G: nx.Graph) -> dict:
    """Maximum cardinality matching."""
    t0 = time.time()
    matching = nx.max_weight_matching(G, maxcardinality=True)
    n = G.number_of_nodes()
    return {
        'size': len(matching),
        'normalized': 2 * len(matching) / n,
        'time': time.time() - t0
    }


# ── 9. MaxFracMatching (LP) ────────────────────────────────────────────

def max_frac_matching(G: nx.Graph) -> dict:
    """Fractional maximum matching via LP.
    max sum(x_e) s.t. sum(x_e : e incident to v) <= 1, 0 <= x_e <= 1."""
    t0 = time.time()
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    c = -np.ones(m)
    row_idx = np.array([u for u, v in edges] + [v for u, v in edges])
    col_idx = np.concatenate([np.arange(m), np.arange(m)])
    data = np.ones(2 * m)
    A_ub = scipy.sparse.csc_matrix((data, (row_idx, col_idx)), shape=(n, m))
    b_ub = np.ones(n)
    bounds = [(0, 1)] * m
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    val = -res.fun if res.success else -1
    return {
        'size': float(val),
        'normalized': 2 * float(val) / n if val >= 0 else -1,
        'time': time.time() - t0
    }


# ── 10. MinSpanningForest ──────────────────────────────────────────────

def min_spanning_forest(G: nx.Graph) -> dict:
    """Minimum spanning tree weight. For unweighted graphs = n - components."""
    t0 = time.time()
    T = nx.minimum_spanning_tree(G)
    weight = T.number_of_edges()
    return {
        'weight': weight,
        'n_components': nx.number_connected_components(G),
        'time': time.time() - t0
    }


# ── Run all ────────────────────────────────────────────────────────────

def run_all(G: nx.Graph, label: str) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"\n{'='*65}")
    print(f"{label}: {n} nodes, {m:,} edges, density {2*m/(n*(n-1)):.4f}")
    print(f"{'='*65}")

    results = {'n': n, 'm': m, 'density': 2 * m / (n * (n - 1))}

    print("\n1. AdjLMaxDivD...")
    results['adj_lmax_div_d'] = adj_lmax_div_d(G)
    r = results['adj_lmax_div_d']
    print(f"   lambda_max={r['lambda_max']:.4f}, avg_deg={r['avg_degree']:.1f}, "
          f"AdjLMaxDivD={r['value']:.4f} [{r['time']:.1f}s]")

    print("2. PGEigengap (transition matrix D^-1 A)...")
    results['pg_eigengap'] = pg_eigengap(G)
    r = results['pg_eigengap']
    print(f"   top eigs(P): {[f'{v:.4f}' for v in r['top_eigenvalues'][:4]]}")
    print(f"   PGEigengap={r['eigengap']:.6f} [{r['time']:.1f}s]")

    print("3. HoffmanBound...")
    results['hoffman_bound'] = hoffman_bound(G)
    r = results['hoffman_bound']
    print(f"   1 + lam_max/|lam_min| = {r['hoffman_bound']:.4f}")
    print(f"   lam_max={r['lambda_max']:.4f}, lam_min={r['lambda_min']:.4f} [{r['time']:.1f}s]")

    print("4. LogAbsSpanningForestN (Kirchhoff)...")
    results['log_spanning_forest'] = log_spanning_forest(G)
    r = results['log_spanning_forest']
    print(f"   log(count)/n = {r['log_spanning_tree_per_n']:.4f} ({r['note']}) [{r['time']:.1f}s]")

    print("5. MinVertexCover (fractional LP)...")
    results['min_vc_frac'] = min_vertex_cover_frac(G)
    r = results['min_vc_frac']
    print(f"   size={r['size']:.2f} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")

    print("6. MinVertexCover (2-approx)...")
    results['min_vc_approx'] = min_vertex_cover_approx(G)
    r = results['min_vc_approx']
    print(f"   size={r['size']} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")

    print("7. MinCutBalDivSum (spectral bisection)...")
    results['min_cut_bal'] = min_cut_bal(G)
    r = results['min_cut_bal']
    print(f"   cut={r['cut_size']} ({r['normalized']:.4f} norm)")
    print(f"   Fiedler={r['fiedler_value']:.6f}")
    print(f"   partition={r['partition_sizes']} [{r['time']:.1f}s]")

    print("8. MaxMatching...")
    results['max_matching'] = max_matching(G)
    r = results['max_matching']
    print(f"   size={r['size']} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")

    print("9. MaxFracMatching (LP)...")
    results['max_frac_matching'] = max_frac_matching(G)
    r = results['max_frac_matching']
    print(f"   size={r['size']:.2f} ({r['normalized']:.4f} norm) [{r['time']:.1f}s]")

    print("10. MinSpanningForest (Kruskal)...")
    results['min_spanning_forest'] = min_spanning_forest(G)
    r = results['min_spanning_forest']
    print(f"    weight={r['weight']} [{r['time']:.1f}s]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Combinatorial graph metrics (Grolmusz-style)")
    parser.add_argument("--graph", required=True, help="Path to the graph (.npz or .gml)")
    parser.add_argument("--json", dest="out", required=True, help="Output JSON for this graph")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.graph))[0]
    G = load_graph(args.graph)
    results = run_all(G, name)

    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
