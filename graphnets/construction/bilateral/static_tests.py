#!/usr/bin/env python3
"""
static_tests.py -- Six static graph tests comparing a constructed quotient
against the Budapest Reference Connectome.

The six tests are: degree distribution, effective diameter, hop plot,
scree plot (singular-value spectrum), network value (principal eigenvector
components), and node triangle participation.

Produces a 4x3 figure: rows 1-2 use the log-log convention standard in the
generative-network-model literature; rows 3-4 use cumulative distributions,
which weight by the mass of the distribution rather than by its tails.
Both are shown because log-log plots give a handful of extreme-degree nodes
the same visual weight as the hundreds of nodes carrying the bulk.

Also prints a table of the underlying scalars and the KS statistics.

Usage:
    python static_tests.py --graph graph_v146_6500.npz --budapest edges.csv
    python static_tests.py --graph q.npz --budapest edges.csv \
        --label "global quotient" --out fig.png --csv stats.csv

Inputs:
    --graph      .npz with an 'edges' array of shape (m,2), or a two-column
                 CSV/TSV edge list.  Node ids must be 0..n-1.
    --budapest   Budapest edge file (braingraph.org CSV export, or any
                 two-column edge list).  Self-loops are dropped.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from scipy.stats import ks_2samp
from networkx.algorithms.community import louvain_communities, modularity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- loading

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


# ---------------------------------------------------------------- measures

def effective_diameter(dists: np.ndarray, q: float = 0.9) -> float:
    """Interpolated q-effective diameter: the x at which g(x) reaches q."""
    for h in range(1, 40):
        if (dists <= h).mean() >= q:
            prev = (dists <= h - 1).mean()
            cur = (dists <= h).mean()
            return h - 1 + (q - prev) / max(cur - prev, 1e-12)
    return float("nan")


def measure(G: nx.Graph) -> dict:
    """All quantities the six tests need.  Exact: full APSP and full spectrum."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="csr").astype(float)

    D = shortest_path(A, unweighted=True, method="D", directed=False)
    finite = D[(D > 0) & np.isfinite(D)]

    dense = A.toarray()
    eigvals = np.linalg.eigvalsh(dense)
    sv = np.sort(np.abs(eigvals))[::-1]
    _, vecs = np.linalg.eigh(dense)
    nv = np.sort(np.abs(vecs[:, -1]))[::-1]

    parts = louvain_communities(G, seed=1)

    return dict(
        n=n,
        m=G.number_of_edges(),
        density=nx.density(G),
        degree=np.array([G.degree(v) for v in range(n)]),
        triangles=np.array(list(nx.triangles(G).values())),
        sv=sv,
        nv=nv,
        apl=float(finite.mean()),
        diameter=int(finite.max()),
        eff_diam=effective_diameter(finite),
        hop_count=np.array([int((finite <= h).sum()) for h in range(1, 7)]),
        hop_frac=np.array([float((finite <= h).mean()) for h in range(1, 7)]),
        clustering=nx.average_clustering(G),
        transitivity=nx.transitivity(G),
        modularity=modularity(G, parts),
        n_communities=len(parts),
    )


def scores(model: dict, ref: dict) -> dict:
    """Normalised error per test; 0 means an exact match to the reference."""
    l1_m = model["sv"][0] / model["degree"].mean()
    l1_r = ref["sv"][0] / ref["degree"].mean()
    gap_m = model["sv"][0] - model["sv"][1]
    gap_r = ref["sv"][0] - ref["sv"][1]
    t = {
        "T1 degree": ks_2samp(model["degree"], ref["degree"]).statistic,
        "T2 eff. diameter": abs(model["eff_diam"] - ref["eff_diam"]) / ref["eff_diam"],
        "T3 hop plot": abs(model["hop_frac"][1] - ref["hop_frac"][1]) / ref["hop_frac"][1],
        "T4 scree": 0.5 * abs(l1_m - l1_r) / l1_r
                    + 0.5 * min(abs(gap_m - gap_r) / gap_r, 2.0),
        "T5 network value": ks_2samp(model["nv"], ref["nv"]).statistic,
        "T6 triangles": ks_2samp(model["triangles"], ref["triangles"]).statistic,
    }
    t["SCORE (mean)"] = float(np.mean(list(t.values())))
    return t


# ---------------------------------------------------------------- plotting

def logbin(x: np.ndarray, nbins: int = 20):
    """Exponential binning, as used for log-log degree plots."""
    x = np.asarray(x)
    x = x[x > 0]
    if x.size == 0:
        return np.array([1]), np.array([0.0])
    edges = np.unique(
        np.round(np.logspace(np.log10(x.min()), np.log10(x.max() + 1), nbins)).astype(int)
    )
    counts, _ = np.histogram(x, bins=np.append(edges, edges[-1] + 1))
    widths = np.diff(np.append(edges, edges[-1] + 1))
    return edges, counts / np.maximum(widths, 1)


def make_figure(M: dict, R: dict, label: str, out: str,
                ref_label: str = "Budapest") -> None:
    col = {label: "crimson", ref_label: "seagreen"}
    S = {label: M, ref_label: R}
    ks_deg = ks_2samp(M["degree"], R["degree"]).statistic
    ks_nv = ks_2samp(M["nv"], R["nv"]).statistic
    ks_tri = ks_2samp(M["triangles"], R["triangles"]).statistic

    fig, ax = plt.subplots(4, 3, figsize=(15.5, 17.5))

    # --- rows 1-2: log-log convention -----------------------------------
    for nm in S:
        e, c = logbin(S[nm]["degree"])
        ax[0, 0].loglog(e, c, "o-", ms=4, color=col[nm], label=nm)
    ax[0, 0].set(xlabel="degree $k$", ylabel="count")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f"(a) degree distribution\nKS $= {ks_deg:.3f}$", fontsize=10)

    for nm in S:
        ax[0, 1].semilogy(range(1, 7), S[nm]["hop_count"], "o-", color=col[nm], label=nm)
    ax[0, 1].set(xlabel="hops $h$", ylabel="reachable pairs $g(h)$")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("(b) hop plot", fontsize=10)

    for nm in S:
        ax[0, 2].loglog(range(1, 101), S[nm]["sv"][:100], "o-", ms=3,
                        color=col[nm], label=nm)
    ax[0, 2].set(xlabel="rank", ylabel="singular value")
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].set_title(
        f"(c) scree plot\n$\\lambda_1$: {M['sv'][0]:.0f} vs {R['sv'][0]:.0f}", fontsize=10)

    for nm in S:
        ax[1, 0].loglog(range(1, S[nm]["n"] + 1), S[nm]["nv"], "-", lw=2,
                        color=col[nm], label=nm)
    ax[1, 0].set(xlabel="rank", ylabel="network value")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title(f"(d) network value\nKS $= {ks_nv:.3f}$", fontsize=10)

    for nm in S:
        e, c = logbin(S[nm]["triangles"])
        ax[1, 1].loglog(e, c, "o-", ms=4, color=col[nm], label=nm)
    ax[1, 1].set(xlabel="triangles per node", ylabel="count")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title(f"(e) triangle participation\nKS $= {ks_tri:.3f}$", fontsize=10)

    names = [label, ref_label]
    ax[1, 2].bar(range(2), [S[n]["eff_diam"] for n in names],
                 color=[col[n] for n in names], width=0.55)
    for i, n in enumerate(names):
        ax[1, 2].text(i, S[n]["eff_diam"] + 0.05, f"{S[n]['eff_diam']:.3f}",
                      ha="center", fontsize=9)
    ax[1, 2].set_xticks(range(2))
    ax[1, 2].set_xticklabels([label.replace(" ", "\n"), ref_label], fontsize=8)
    ax[1, 2].set(ylabel="effective diameter (90%)", ylim=(0, 3.4))
    ax[1, 2].set_title("(f) effective diameter", fontsize=10)

    # --- rows 3-4: cumulative convention --------------------------------
    def cdf(a, key, xlabel, title):
        for nm in [ref_label, label]:
            x = np.sort(S[nm][key])
            a.plot(x, np.arange(1, len(x) + 1) / len(x), color=col[nm], lw=2.2, label=nm)
        a.set(xlabel=xlabel, ylabel="cumulative fraction")
        a.legend(fontsize=8, loc="lower right")
        a.grid(alpha=0.25)
        a.set_title(title, fontsize=10)

    cdf(ax[2, 0], "degree", "node degree", "(a$'$) degree — cumulative")

    for nm in [ref_label, label]:
        ax[2, 1].plot(range(1, 7), S[nm]["hop_frac"], "o-", color=col[nm], lw=2.2, label=nm)
    ax[2, 1].set(xlabel="hops $h$", ylabel="fraction of pairs within $h$", ylim=(0, 1.05))
    ax[2, 1].legend(fontsize=8, loc="lower right")
    ax[2, 1].grid(alpha=0.25)
    ax[2, 1].set_title("(b$'$) hop plot — fraction reached", fontsize=10)

    for nm in [ref_label, label]:
        s = S[nm]["sv"][:40]
        ax[2, 2].plot(range(1, 41), s / s[0], "o-", ms=3.5, color=col[nm], lw=2, label=nm)
    ax[2, 2].set(xlabel="rank", ylabel="singular value / $\\lambda_1$", ylim=(0, 1.05))
    ax[2, 2].legend(fontsize=8)
    ax[2, 2].grid(alpha=0.25)
    ax[2, 2].set_title("(c$'$) scree — normalised to $\\lambda_1$", fontsize=10)

    cdf(ax[3, 0], "nv", "principal eigenvector component",
        "(d$'$) network value — cumulative")
    cdf(ax[3, 1], "triangles", "triangles per node",
        "(e$'$) triangle participation — cumulative")

    keys = ["degree sd", "clustering", "modularity", "APL", "eff. diam", "density"]
    mv = [M["degree"].std(), M["clustering"], M["modularity"],
          M["apl"], M["eff_diam"], M["density"]]
    rv = [R["degree"].std(), R["clustering"], R["modularity"],
          R["apl"], R["eff_diam"], R["density"]]
    x = np.arange(len(keys))
    w = 0.36
    ax[3, 2].bar(x - w / 2, [a / b for a, b in zip(mv, rv)], w,
                 color=col[label], label=label)
    ax[3, 2].bar(x + w / 2, [1.0] * len(keys), w, color=col[ref_label], label=ref_label)
    ax[3, 2].axhline(1.0, color="k", ls="--", lw=1)
    ax[3, 2].set_xticks(x)
    ax[3, 2].set_xticklabels([k.replace(" ", "\n") for k in keys], fontsize=8)
    ax[3, 2].set(ylabel=f"ratio to {ref_label}", ylim=(0, 1.4))
    ax[3, 2].legend(fontsize=8)
    ax[3, 2].grid(alpha=0.25, axis="y")
    ax[3, 2].set_title(f"(f$'$) summary — relative to {ref_label}", fontsize=10)

    fig.suptitle(
        f"Six static graph tests: {label} ($n={M['n']}$, $m={M['m']:,}$) "
        f"versus the {ref_label} Reference Connectome.\n"
        f"Rows 1–2: log-log convention.   Rows 3–4: cumulative convention.",
        fontsize=12.5, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"figure -> {out}")


# ---------------------------------------------------------------- report

def report(M: dict, R: dict, label: str, ref_label: str, csv_path: str | None):
    rows = [
        ("nodes", M["n"], R["n"]),
        ("edges", M["m"], R["m"]),
        ("density", M["density"], R["density"]),
        ("mean degree", M["degree"].mean(), R["degree"].mean()),
        ("degree sd", M["degree"].std(), R["degree"].std()),
        ("min degree", M["degree"].min(), R["degree"].min()),
        ("max degree", M["degree"].max(), R["degree"].max()),
        ("clustering", M["clustering"], R["clustering"]),
        ("transitivity", M["transitivity"], R["transitivity"]),
        ("modularity", M["modularity"], R["modularity"]),
        ("n communities", M["n_communities"], R["n_communities"]),
        ("avg path length", M["apl"], R["apl"]),
        ("diameter", M["diameter"], R["diameter"]),
        ("effective diameter", M["eff_diam"], R["eff_diam"]),
        ("g(2)/total pairs", M["hop_frac"][1], R["hop_frac"][1]),
        ("lambda_1", M["sv"][0], R["sv"][0]),
        ("lambda_2", M["sv"][1], R["sv"][1]),
        ("lambda_1/<d>", M["sv"][0] / M["degree"].mean(), R["sv"][0] / R["degree"].mean()),
        ("spectral gap", M["sv"][0] - M["sv"][1], R["sv"][0] - R["sv"][1]),
        ("mean triangles", M["triangles"].mean(), R["triangles"].mean()),
        ("min triangles", M["triangles"].min(), R["triangles"].min()),
    ]
    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'':<{w}}{label:>18}{ref_label:>14}")
    print("-" * (w + 32))
    for name, a, b in rows:
        fa = f"{a:.4f}" if isinstance(a, float) else f"{a:,}"
        fb = f"{b:.4f}" if isinstance(b, float) else f"{b:,}"
        print(f"{name:<{w}}{fa:>18}{fb:>14}")

    print(f"\nNormalised error per test (0 = exact match to {ref_label}):")
    sc = scores(M, R)
    for k, v in sc.items():
        print(f"  {k:<22}{v:8.4f}")

    if csv_path:
        pd.DataFrame(
            [{"measure": n, label: a, ref_label: b} for n, a, b in rows]
            + [{"measure": k, label: v, ref_label: 0.0} for k, v in sc.items()]
        ).to_csv(csv_path, index=False)
        print(f"\nstats -> {csv_path}")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, help="constructed quotient (.npz or edge list)")
    p.add_argument("--budapest", required=True, help="Budapest edge file")
    p.add_argument("--label", default="global quotient", help="legend label for --graph")
    p.add_argument("--ref-label", default="Budapest", help="legend label for --budapest")
    p.add_argument("--out", default="static_tests.png", help="output figure path")
    p.add_argument("--csv", default=None, help="optional path for the statistics table")
    args = p.parse_args()

    for path in (args.graph, args.budapest):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    Gm = build_graph(load_edges(args.graph))
    Gr = build_graph(load_edges(args.budapest))
    if Gm.number_of_nodes() != Gr.number_of_nodes():
        print(f"note: node counts differ ({Gm.number_of_nodes()} vs "
              f"{Gr.number_of_nodes()}); tests are still computed, but "
              f"comparisons assume matched scale.")

    print(f"measuring {args.label} ...")
    M = measure(Gm)
    print(f"measuring {args.ref_label} ...")
    R = measure(Gr)

    report(M, R, args.label, args.ref_label, args.csv)
    make_figure(M, R, args.label, args.out, args.ref_label)


if __name__ == "__main__":
    main()
