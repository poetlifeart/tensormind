#!/usr/bin/env python3
"""
static_test_effective_diameter.py -- test T2 ('T2 eff. diameter') of
construction/bilateral/static_tests.py, on its own.

Statistic: the interpolated 90% effective diameter (Leskovec, Kleinberg &
Faloutsos 2005, KDD; the definition used in Leskovec, Chakrabarti, Kleinberg,
Faloutsos & Ghahramani 2010, JMLR 11:985, Sec. 2).  Let g(h) be the fraction
of connected node pairs whose shortest path has at most h hops; join the points
(h, g(h)) linearly; the effective diameter is the x at which g(x) reaches 0.9.

How static_tests.py computes it (measure() and effective_diameter()):
    A      = adjacency of the simple undirected graph (self-loops dropped)
    D      = exact all-pairs shortest paths (scipy Dijkstra, unweighted)
    finite = D[(D > 0) & isfinite(D)]       ordered pairs u != v, connected only
    eff    = first h in 1..39 with mean(finite <= h) >= 0.9, then
             h - 1 + (0.9 - g(h-1)) / (g(h) - g(h-1))
             (NaN if 90% is not reached within 39 hops)
Comparison error (static_tests.scores()):
    T2 = |eff_graph - eff_ref| / eff_ref

Self-contained: needs numpy, scipy, networkx and pandas only (pandas for the
text-file branch of load_edges), and does not import static_tests.py.
load_edges, build_graph and effective_diameter are copied from it unchanged;
measure_effective_diameter() and score_effective_diameter() hold the lines of
measure() and scores() that this test uses, copied verbatim (one context field,
connected_ordered_pairs, is added).  The diameter is printed as context (it
comes from the same distance array in measure()); it is not part of the score.

Usage:
    python static_test_effective_diameter.py --graph G.npz
    python static_test_effective_diameter.py --graph G.npz --reference budapest.edgelist
    python static_test_effective_diameter.py --graph G.npz --json eff.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import shortest_path


# ═══════════════════════════════════════════════════════════════════════════
# Loading  (copied from static_tests.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_edges(path: str) -> np.ndarray:
    """Return an (m,2) int array of edges from .npz or a two-column text file."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        for key in ("edges", "edge_list", "E"):
            if key in d:
                return np.asarray(d[key], dtype=np.int64)[:, :2]
        raise KeyError(f"{path}: no 'edges' array (found {list(d.keys())})")
    # Budapest exports are comma-separated with '#' comment lines and no header.
    df = pd.read_csv(path, comment="#", header=None, sep=None, engine="python")
    return df.iloc[:, :2].to_numpy(dtype=np.int64)


def build_graph(edges: np.ndarray, n_nodes: int | None = None) -> nx.Graph:
    """Simple undirected graph; self-loops and duplicates dropped."""
    edges = edges[edges[:, 0] != edges[:, 1]]
    n = int(edges.max()) + 1 if n_nodes is None else n_nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Measure  (copied from static_tests.py)
# ═══════════════════════════════════════════════════════════════════════════

def effective_diameter(dists: np.ndarray, q: float = 0.9) -> float:
    """Interpolated q-effective diameter: the x at which g(x) reaches q."""
    for h in range(1, 40):
        if (dists <= h).mean() >= q:
            prev = (dists <= h - 1).mean()
            cur = (dists <= h).mean()
            return h - 1 + (q - prev) / max(cur - prev, 1e-12)
    return float("nan")


# ═══════════════════════════════════════════════════════════════════════════
# The test  (lines of static_tests.measure() / scores(), copied verbatim)
# ═══════════════════════════════════════════════════════════════════════════

def measure_effective_diameter(G: nx.Graph) -> dict:
    """The part of static_tests.measure() that T2 uses."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="csr").astype(float)

    D = shortest_path(A, unweighted=True, method="D", directed=False)
    finite = D[(D > 0) & np.isfinite(D)]

    return dict(
        n=n,
        m=G.number_of_edges(),
        diameter=int(finite.max()),
        eff_diam=effective_diameter(finite),
        connected_ordered_pairs=int(finite.size),   # added, context only
    )


def score_effective_diameter(model: dict, ref: dict) -> float:
    """The 'T2 eff. diameter' entry of static_tests.scores()."""
    t = {
        "T2 eff. diameter": abs(model["eff_diam"] - ref["eff_diam"]) / ref["eff_diam"],
    }
    return t["T2 eff. diameter"]


def show(M: dict, name: str) -> None:
    print(f"{name}")
    print(f"  nodes {M['n']:,}   edges {M['m']:,}   connected ordered pairs (u != v) "
          f"{M['connected_ordered_pairs']:,}")
    print(f"  effective diameter (90%, interpolated) = {float(M['eff_diam'])!r}")
    print(f"  diameter (context, not scored)          = {M['diameter']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, help="graph (.npz with 'edges', or edge list), as static_tests.py --graph")
    p.add_argument("--reference", default=None, help="optional second graph, as static_tests.py --budapest")
    p.add_argument("--json", default=None, help="write the statistic (and the error) to this JSON file")
    args = p.parse_args()
    for path in [args.graph] + ([args.reference] if args.reference else []):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    M = measure_effective_diameter(build_graph(load_edges(args.graph)))
    show(M, args.graph)
    out = {"graph": args.graph, "stat": M}
    if args.reference:
        R = measure_effective_diameter(build_graph(load_edges(args.reference)))
        show(R, args.reference)
        err = score_effective_diameter(M, R)
        print(f"\nT2 eff. diameter   |eff_graph - eff_ref| / eff_ref = {float(err)!r}")
        out.update(reference=args.reference, ref_stat=R, error=float(err))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
