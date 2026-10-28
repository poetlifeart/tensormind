#!/usr/bin/env python3
"""
static_test_degree.py -- test T1 ('T1 degree') of
construction/bilateral/static_tests.py, on its own.

Statistic: the degree distribution (the first static pattern in Leskovec,
Chakrabarti, Kleinberg, Faloutsos & Ghahramani 2010, JMLR 11:985, Sec. 2;
heavy tails after Faloutsos, Faloutsos & Faloutsos 1999): the degree of every
node of the simple undirected graph.

How static_tests.py computes it (measure()):
    degree = [deg(v) for v in 0..n-1]   (self-loops dropped, isolated nodes kept)
Comparison error (static_tests.scores()): the two-sample Kolmogorov-Smirnov
statistic between the two degree sequences,
    T1 = scipy.stats.ks_2samp(degree_graph, degree_ref).statistic
i.e. the largest vertical gap between the two empirical degree CDFs.

Self-contained: needs numpy, scipy, networkx and pandas only (pandas for the
text-file branch of load_edges), and does not import static_tests.py.
load_edges and build_graph are copied from it unchanged; measure_degree() and
score_degree() hold the lines of measure() and scores() that this test uses,
copied verbatim.  Printed summaries (mean, sd, min, max, histogram) are added
for reading the result; they are not part of the score.

Usage:
    python static_test_degree.py --graph G.npz
    python static_test_degree.py --graph G.npz --reference budapest.edgelist
    python static_test_degree.py --graph G.npz --json degree.json
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

def measure_degree(G: nx.Graph) -> dict:
    """The part of static_tests.measure() that T1 uses."""
    n = G.number_of_nodes()
    return dict(
        n=n,
        m=G.number_of_edges(),
        degree=np.array([G.degree(v) for v in range(n)]),
    )


def score_degree(model: dict, ref: dict) -> float:
    """The 'T1 degree' entry of static_tests.scores()."""
    t = {
        "T1 degree": ks_2samp(model["degree"], ref["degree"]).statistic,
    }
    return t["T1 degree"]


def show(M: dict, name: str) -> None:
    d = M["degree"]
    print(f"{name}")
    print(f"  nodes {M['n']:,}   edges {M['m']:,}   sum of degrees {int(d.sum()):,}")
    print(f"  mean degree {float(d.mean())!r}   sd {float(d.std())!r}   min {int(d.min())}   max {int(d.max())}")
    ks, counts = np.unique(d, return_counts=True)
    pairs = [f"{int(k)}:{int(c)}" for k, c in zip(ks, counts)]
    print("  degree:count  " + "  ".join(pairs[:60]) + ("  ..." if len(pairs) > 60 else ""))


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

    M = measure_degree(build_graph(load_edges(args.graph)))
    show(M, args.graph)
    out = {"graph": args.graph, "n": M["n"], "m": M["m"], "degree": M["degree"].tolist()}
    if args.reference:
        R = measure_degree(build_graph(load_edges(args.reference)))
        show(R, args.reference)
        err = score_degree(M, R)
        print(f"\nT1 degree   KS statistic between the degree sequences = {float(err)!r}")
        out.update(reference=args.reference, ref_n=R["n"], ref_m=R["m"],
                   ref_degree=R["degree"].tolist(), error=float(err))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
