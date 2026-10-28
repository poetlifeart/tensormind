#!/usr/bin/env python3
"""
static_test_hop_plot.py -- test T3 ('T3 hop plot') of
construction/bilateral/static_tests.py, on its own.

Statistic: the hop plot (Palmer, Gibbons & Faloutsos 2002, KDD, "ANF"; as used
in Leskovec, Chakrabarti, Kleinberg, Faloutsos & Ghahramani 2010, JMLR 11:985,
Sec. 2): the number of node pairs g(h) reachable within h hops, against h.

How static_tests.py computes it (measure()):
    A          = adjacency of the simple undirected graph (self-loops dropped)
    D          = exact all-pairs shortest paths (scipy Dijkstra, unweighted)
    finite     = D[(D > 0) & isfinite(D)]   ordered pairs u != v, connected only
    hop_count  = [#(finite <= h)    for h = 1..6]   (ordered pairs: 2x unordered)
    hop_frac   = [mean(finite <= h) for h = 1..6]   (fraction of CONNECTED pairs)
Comparison error (static_tests.scores()) scores one point of the curve, the
fraction of pairs within 2 hops:
    T3 = |hop_frac_graph[1] - hop_frac_ref[1]| / hop_frac_ref[1]
(static_tests.py's table calls hop_frac[1] "g(2)/total pairs"; the denominator
is the number of connected ordered pairs, which equals all pairs only when the
graph is connected.)

Self-contained: needs numpy, scipy, networkx and pandas only (pandas for the
text-file branch of load_edges), and does not import static_tests.py.
load_edges and build_graph are copied from it unchanged; measure_hop_plot()
and score_hop_plot() hold the lines of measure() and scores() that this test
uses, copied verbatim (one context field, connected_ordered_pairs, is added).

Usage:
    python static_test_hop_plot.py --graph G.npz
    python static_test_hop_plot.py --graph G.npz --reference budapest.edgelist
    python static_test_hop_plot.py --graph G.npz --json hop.json
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
# The test  (lines of static_tests.measure() / scores(), copied verbatim)
# ═══════════════════════════════════════════════════════════════════════════

def measure_hop_plot(G: nx.Graph) -> dict:
    """The part of static_tests.measure() that T3 uses."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="csr").astype(float)

    D = shortest_path(A, unweighted=True, method="D", directed=False)
    finite = D[(D > 0) & np.isfinite(D)]

    return dict(
        n=n,
        m=G.number_of_edges(),
        hop_count=np.array([int((finite <= h).sum()) for h in range(1, 7)]),
        hop_frac=np.array([float((finite <= h).mean()) for h in range(1, 7)]),
        connected_ordered_pairs=int(finite.size),   # added, context only
    )


def score_hop_plot(model: dict, ref: dict) -> float:
    """The 'T3 hop plot' entry of static_tests.scores()."""
    t = {
        "T3 hop plot": abs(model["hop_frac"][1] - ref["hop_frac"][1]) / ref["hop_frac"][1],
    }
    return t["T3 hop plot"]


def show(M: dict, name: str) -> None:
    print(f"{name}")
    print(f"  nodes {M['n']:,}   edges {M['m']:,}   connected ordered pairs (u != v) "
          f"{M['connected_ordered_pairs']:,}")
    print(f"  {'h':>3} {'hop_count g(h)':>16} {'hop_frac':>22}")
    for h in range(1, 7):
        print(f"  {h:>3} {int(M['hop_count'][h - 1]):>16,} {float(M['hop_frac'][h - 1])!r:>22}")
    print(f"  scored point hop_frac[1] (within 2 hops) = {float(M['hop_frac'][1])!r}")


def jsonable(M: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in M.items()}


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

    M = measure_hop_plot(build_graph(load_edges(args.graph)))
    show(M, args.graph)
    out = {"graph": args.graph, "stat": jsonable(M)}
    if args.reference:
        R = measure_hop_plot(build_graph(load_edges(args.reference)))
        show(R, args.reference)
        err = score_hop_plot(M, R)
        print(f"\nT3 hop plot   |frac2_graph - frac2_ref| / frac2_ref = {float(err)!r}")
        out.update(reference=args.reference, ref_stat=jsonable(R), error=float(err))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
