#!/usr/bin/env python3
"""
static_test_scree.py -- test T4 ('T4 scree') of
construction/bilateral/static_tests.py, on its own.

Statistic: the scree plot (Chakrabarti, Zhan & Faloutsos 2004; Farkas et al.
2001; as used in Leskovec, Chakrabarti, Kleinberg, Faloutsos & Ghahramani
2010, JMLR 11:985, Sec. 2): the singular values of the adjacency matrix in
decreasing order.  For a symmetric adjacency matrix the singular values are the
absolute eigenvalues.

How static_tests.py computes it (measure()):
    dense = adjacency of the simple undirected graph, as a dense array
    sv    = sort(|eigvalsh(dense)|), descending   (full spectrum, exact)
The score uses two scalars of that curve:
    lambda_1/<d> = sv[0] / mean degree          (spectral norm over mean degree)
    gap          = sv[0] - sv[1]                (top two singular values)
Comparison error (static_tests.scores()):
    T4 = 0.5 * |l1_g - l1_r| / l1_r  +  0.5 * min(|gap_g - gap_r| / gap_r, 2.0)

Self-contained: needs numpy, scipy, networkx and pandas only (pandas for the
text-file branch of load_edges), and does not import static_tests.py.
load_edges and build_graph are copied from it unchanged; measure_scree() and
score_scree() hold the lines of measure() and scores() that this test uses,
copied verbatim.

Usage:
    python static_test_scree.py --graph G.npz
    python static_test_scree.py --graph G.npz --reference budapest.edgelist
    python static_test_scree.py --graph G.npz --json scree.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import networkx as nx
import pandas as pd


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

def measure_scree(G: nx.Graph) -> dict:
    """The part of static_tests.measure() that T4 uses."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="csr").astype(float)

    dense = A.toarray()
    eigvals = np.linalg.eigvalsh(dense)
    sv = np.sort(np.abs(eigvals))[::-1]

    return dict(
        n=n,
        m=G.number_of_edges(),
        degree=np.array([G.degree(v) for v in range(n)]),
        sv=sv,
    )


def score_scree(model: dict, ref: dict) -> float:
    """The 'T4 scree' entry of static_tests.scores()."""
    l1_m = model["sv"][0] / model["degree"].mean()
    l1_r = ref["sv"][0] / ref["degree"].mean()
    gap_m = model["sv"][0] - model["sv"][1]
    gap_r = ref["sv"][0] - ref["sv"][1]
    t = {
        "T4 scree": 0.5 * abs(l1_m - l1_r) / l1_r
                    + 0.5 * min(abs(gap_m - gap_r) / gap_r, 2.0),
    }
    return t["T4 scree"]


def show(M: dict, name: str) -> None:
    sv = M["sv"]
    print(f"{name}")
    print(f"  nodes {M['n']:,}   edges {M['m']:,}   mean degree {float(M['degree'].mean())!r}")
    print(f"  sv[0] = lambda_1 (spectral norm) = {float(sv[0])!r}")
    print(f"  sv[1] = lambda_2                 = {float(sv[1])!r}")
    print(f"  lambda_1/<d>                     = {float(sv[0] / M['degree'].mean())!r}")
    print(f"  gap sv[0] - sv[1]                = {float(sv[0] - sv[1])!r}")
    print(f"  sv[0]/sv[1]                      = {float(sv[0] / sv[1])!r}")
    print("  sv[0:10] = " + "  ".join(f"{float(x):.6f}" for x in sv[:10]))


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

    M = measure_scree(build_graph(load_edges(args.graph)))
    show(M, args.graph)
    out = {"graph": args.graph, "n": M["n"], "m": M["m"],
           "mean_degree": float(M["degree"].mean()), "sv": M["sv"].tolist()}
    if args.reference:
        R = measure_scree(build_graph(load_edges(args.reference)))
        show(R, args.reference)
        err = score_scree(M, R)
        print(f"\nT4 scree   0.5*|l1 rel. err| + 0.5*min(|gap rel. err|, 2) = {float(err)!r}")
        out.update(reference=args.reference, ref_n=R["n"], ref_m=R["m"],
                   ref_mean_degree=float(R["degree"].mean()), ref_sv=R["sv"].tolist(),
                   error=float(err))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
