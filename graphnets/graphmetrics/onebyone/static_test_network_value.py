#!/usr/bin/env python3
"""
static_test_network_value.py -- test T5 ('T5 network value') of
construction/bilateral/static_tests.py, on its own.

Statistic: the "network values" (Chakrabarti, Zhan & Faloutsos 2004; as used
in Leskovec, Chakrabarti, Kleinberg, Faloutsos & Ghahramani 2010, JMLR 11:985,
Sec. 2): the components of the principal eigenvector of the adjacency matrix,
in absolute value, sorted in decreasing order.

How static_tests.py computes it (measure()):
    dense   = adjacency of the simple undirected graph, as a dense array
    _, vecs = np.linalg.eigh(dense)            (eigenvalues ascending)
    nv      = sort(|vecs[:, -1]|), descending  (unit Euclidean norm; the
              eigenvector of the largest eigenvalue, i.e. the Perron vector)
Comparison error (static_tests.scores()): the two-sample Kolmogorov-Smirnov
statistic between the two sets of components,
    T5 = scipy.stats.ks_2samp(nv_graph, nv_ref).statistic
(static_tests.py's docstring notes that unit-norm components scale like
1/sqrt(n), so graphs of different sizes differ for that reason alone.)

Self-contained: needs numpy, scipy, networkx and pandas only (pandas for the
text-file branch of load_edges), and does not import static_tests.py.
load_edges and build_graph are copied from it unchanged;
measure_network_value() and score_network_value() hold the lines of measure()
and scores() that this test uses, copied verbatim.

Usage:
    python static_test_network_value.py --graph G.npz
    python static_test_network_value.py --graph G.npz --reference budapest.edgelist
    python static_test_network_value.py --graph G.npz --json nv.json
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

def measure_network_value(G: nx.Graph) -> dict:
    """The part of static_tests.measure() that T5 uses."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="csr").astype(float)

    dense = A.toarray()
    _, vecs = np.linalg.eigh(dense)
    nv = np.sort(np.abs(vecs[:, -1]))[::-1]

    return dict(
        n=n,
        m=G.number_of_edges(),
        nv=nv,
        principal_vector=np.abs(vecs[:, -1]),   # added, context only (node order)
    )


def score_network_value(model: dict, ref: dict) -> float:
    """The 'T5 network value' entry of static_tests.scores()."""
    t = {
        "T5 network value": ks_2samp(model["nv"], ref["nv"]).statistic,
    }
    return t["T5 network value"]


def show(M: dict, name: str) -> None:
    nv, pv = M["nv"], M["principal_vector"]
    top = np.argsort(-pv, kind="stable")[:5]
    print(f"{name}")
    print(f"  nodes {M['n']:,}   edges {M['m']:,}   ||nv||_2 = {float(np.sqrt((nv ** 2).sum()))!r}")
    print("  nv[0:10] = " + "  ".join(f"{float(x):.6f}" for x in nv[:10]))
    print(f"  nv min {float(nv[-1])!r}   median {float(np.median(nv))!r}   "
          f"nv[0]/median {float(nv[0] / np.median(nv))!r}")
    print("  largest components at nodes (0-based id: value) " +
          "  ".join(f"{int(i)}: {float(pv[i]):.6f}" for i in top))


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

    M = measure_network_value(build_graph(load_edges(args.graph)))
    show(M, args.graph)
    out = {"graph": args.graph, "n": M["n"], "m": M["m"], "nv": M["nv"].tolist(),
           "principal_vector": M["principal_vector"].tolist()}
    if args.reference:
        R = measure_network_value(build_graph(load_edges(args.reference)))
        show(R, args.reference)
        err = score_network_value(M, R)
        print(f"\nT5 network value   KS statistic between the sorted components = {float(err)!r}")
        out.update(reference=args.reference, ref_n=R["n"], ref_m=R["m"],
                   ref_nv=R["nv"].tolist(), error=float(err))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
