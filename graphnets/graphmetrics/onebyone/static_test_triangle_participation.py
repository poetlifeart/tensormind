#!/usr/bin/env python3
"""
static_test_triangle_participation.py -- test T6 ('T6 triangles') of
construction/bilateral/static_tests.py, on its own.

Statistic: node triangle participation (Tsourakakis 2008, ICDM; as used in
Leskovec, Chakrabarti, Kleinberg, Faloutsos & Ghahramani 2010, JMLR 11:985,
Sec. 2): for every node, the number of triangles it belongs to (the number of
links among its neighbours).  The graph's total triangle count is
sum(triangles) / 3.

How static_tests.py computes it (measure()):
    triangles = list(nx.triangles(G).values())   (simple undirected graph)
Comparison error (static_tests.scores()): the two-sample Kolmogorov-Smirnov
statistic between the two per-node triangle counts,
    T6 = scipy.stats.ks_2samp(triangles_graph, triangles_ref).statistic

Self-contained: needs numpy, scipy, networkx and pandas only (pandas for the
text-file branch of load_edges), and does not import static_tests.py.
load_edges and build_graph are copied from it unchanged;
measure_triangles() and score_triangles() hold the lines of measure() and
scores() that this test uses, copied verbatim.  Printed summaries (total,
mean, max, nodes in no triangle) are added for reading the result; they are
not part of the score.

Usage:
    python static_test_triangle_participation.py --graph G.npz
    python static_test_triangle_participation.py --graph G.npz --reference budapest.edgelist
    python static_test_triangle_participation.py --graph G.npz --json tri.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import ks_2samp


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

def measure_triangles(G: nx.Graph) -> dict:
    """The part of static_tests.measure() that T6 uses."""
    n = G.number_of_nodes()
    return dict(
        n=n,
        m=G.number_of_edges(),
        triangles=np.array(list(nx.triangles(G).values())),
    )


def score_triangles(model: dict, ref: dict) -> float:
    """The 'T6 triangles' entry of static_tests.scores()."""
    t = {
        "T6 triangles": ks_2samp(model["triangles"], ref["triangles"]).statistic,
    }
    return t["T6 triangles"]


def show(M: dict, name: str) -> None:
    t = M["triangles"]
    print(f"{name}")
    print(f"  nodes {M['n']:,}   edges {M['m']:,}")
    print(f"  total triangles sum/3 = {int(t.sum()) // 3:,}   (sum over nodes {int(t.sum()):,})")
    print(f"  per node: mean {float(t.mean())!r}   min {int(t.min())}   max {int(t.max())}   "
          f"nodes in no triangle {int((t == 0).sum()):,}")


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

    M = measure_triangles(build_graph(load_edges(args.graph)))
    show(M, args.graph)
    out = {"graph": args.graph, "n": M["n"], "m": M["m"], "triangles": M["triangles"].tolist()}
    if args.reference:
        R = measure_triangles(build_graph(load_edges(args.reference)))
        show(R, args.reference)
        err = score_triangles(M, R)
        print(f"\nT6 triangles   KS statistic between the per-node triangle counts = {float(err)!r}")
        out.update(reference=args.reference, ref_n=R["n"], ref_m=R["m"],
                   ref_triangles=R["triangles"].tolist(), error=float(err))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
