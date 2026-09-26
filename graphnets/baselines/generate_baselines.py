#!/usr/bin/env python3
"""
generate_baselines.py -- standard generative models at the reference's scale, for
comparison against the construction.

    python3 generate_baselines.py                 # write all baselines to graphs/
    python3 generate_baselines.py --list          # show what would be built
    python3 generate_baselines.py --only dcsbm_construction

WHY.  The paper compares the construction to the Budapest reference connectome but
to no other generator, so a reader cannot tell whether nine passed criteria and a
six-test score of 0.2009 are hard to reach or easy.  These baselines answer that.

THREE TIERS, AND THE TIER IS THE POINT.  A single "fair baseline" invites an
argument about whether it was fair; a ladder shows which constraints carry the
result.

  TIER 1  n and density matched only.  The weakest comparison: any model that
          reproduces the connectome here is doing so from scale alone.

  TIER 2  degree sequence matched as well, to Budapest's exact sequence.  This is
          what a referee will assume, because most connectome statistics are
          partly determined by the degree sequence and the audit's own null model
          is degree-preserving.

  TIER 3  degree sequence AND community structure matched.  Nearly the
          construction by design, so passing here is close to uninformative about
          the construction's value -- but FAILING here would be informative, and
          the tier bounds how much of the result is block structure plus degrees.

  THE TIER-3 FITTING CHOICE, stated because it changes what the tier means.
  A degree-corrected SBM needs a block assignment from somewhere:

    from BUDAPEST's own Louvain partition  ->  asks "can a block model carrying
      the connectome's community structure reproduce the connectome?"  Nearly
      circular; the answer is close to yes by construction.

    from the CONSTRUCTION's 307/306/201/201 ->  asks "does the construction add
      anything beyond its own block structure and Budapest's degrees?"  This is
      the question worth answering, and it is the primary tier-3 baseline here.

  Both are built.  `dcsbm_construction` is the one to read; `dcsbm_budapest` is
  included so the near-circular version is visible rather than omitted.

MATCHING IS REPORTED, NOT ASSUMED.  Several models cannot hit m exactly at fixed
n -- Watts-Strogatz needs an even degree, a random geometric graph reaches a given
m only through a tuned radius, Kronecker is defined on powers of two.  Every
baseline's achieved n, m and density are written into the manifest next to the
targets, so a reader sees the mismatch rather than trusting a claim of parity.

KRONECKER IS DELIBERATELY ABSENT, and the reason is a result rather than a
shortfall.  A 2x2 stochastic Kronecker initiator over L levels has expected
directed edge count S**L, where S is the sum of the four entries.  Matching
2m = 141,308 at L = 10 (N = 1,024) requires

    S = 141308 ** (1/10) = 3.274,   against a maximum of 4.000

so the initiator must average 0.818 per entry.  Every heavy-tailed shape
[[a,b],[b,c]] with a > b > c then forces a > 1, which is not a probability:

    c = 0.1  ->  a = 1.322        c = 0.3  ->  a = 1.239
    c = 0.2  ->  a = 1.281        c = 0.4  ->  a = 1.197

Only a near-uniform initiator reaches this density, and a near-uniform Kronecker
graph is an Erdos-Renyi graph with extra steps -- already in the ladder as `er`.
Kronecker graphs are built for sparse heavy-tailed networks; a 1,015-node graph at
density 0.137 is outside that regime.  The builder is retained below, unused, so
the arithmetic can be checked.

Deterministic: every generator takes an explicit seed, default 42.
Requires numpy, scipy, networkx.  No GPU, no downloads.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', 'construction', 'bilateral', 'reference',
                   'budapest_1015_70654.edgelist')
CONSTRUCTION = os.path.join(HERE, '..', 'construction', 'bilateral', 'graphs',
                            'cnew_coarse.npz')
OUT = os.path.join(HERE, 'graphs')


# ─────────────────────────────────────────────────────────── reference targets

def reference():
    B = nx.read_edgelist(REF, nodetype=int)
    B = nx.convert_node_labels_to_integers(B, ordering='sorted')
    return B


def budapest_partition(B, seeds=20):
    """Budapest's own best-of-N Louvain partition, as block labels."""
    from networkx.algorithms.community import louvain_communities, modularity
    best = None
    for s in range(seeds):
        cs = louvain_communities(B, seed=s)
        q = modularity(B, cs)
        if best is None or q > best[0]:
            best = (q, cs)
    lab = np.empty(B.number_of_nodes(), int)
    for i, c in enumerate(sorted(best[1], key=len, reverse=True)):
        for v in c:
            lab[v] = i
    return lab, best[0]


def construction_blocks(n=1015, targets=(307, 306, 201, 201)):
    """The construction's own block assignment, which is contiguous by
    construction: build_coarse numbers supernodes block by block in the order of
    BLOCK_TARGETS."""
    lab = np.empty(n, int)
    o = 0
    for b, t in enumerate(targets):
        lab[o:o + t] = b
        o += t
    return lab


# ───────────────────────────────────────────────────────────── tier 1 builders

def er(n, m, seed):
    """Erdos-Renyi G(n,m).  Exact on n and m."""
    return nx.gnm_random_graph(n, m, seed=seed)


def watts_strogatz(n, m, seed, p=0.1):
    """Watts-Strogatz.  k must be even, so m is matched only to the nearest
    n*k/2; the achieved value is reported."""
    k = int(round(2 * m / n))
    if k % 2:
        k -= 1
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def random_geometric(n, m, seed, tol=0.002, iters=40):
    """Random geometric graph in the unit square, radius bisected to hit m."""
    lo, hi = 0.0, 1.5
    G = None
    for _ in range(iters):
        r = 0.5 * (lo + hi)
        G = nx.random_geometric_graph(n, r, seed=seed)
        got = G.number_of_edges()
        if abs(got - m) / m < tol:
            break
        if got < m:
            lo = r
        else:
            hi = r
    return G


def modular_random(n, m, seed, blocks=(307, 306, 201, 201), ratio=None):
    """Four-block random graph.  Within- and between-block densities are fitted
    so the total edge count matches m while the within:between density ratio
    matches Budapest's, measured under Budapest's own partition."""
    rng = np.random.default_rng(seed)
    lab = construction_blocks(n, blocks)
    B = reference()
    blab, _ = budapest_partition(B)
    # measured within/between density ratio in the reference
    win = bet = wpairs = bpairs = 0
    for u, v in B.edges():
        if blab[u] == blab[v]:
            win += 1
        else:
            bet += 1
    for i in range(len(blocks)):
        ni = int((blab == i).sum())
        wpairs += ni * (ni - 1) / 2
        for j in range(i + 1, len(blocks)):
            bpairs += ni * int((blab == j).sum())
    dw, db = win / wpairs, bet / bpairs
    ratio = dw / db if ratio is None else ratio
    # scale so total edges = m
    tw = bw = 0
    for i in range(len(blocks)):
        ni = blocks[i]
        tw += ni * (ni - 1) / 2
        for j in range(i + 1, len(blocks)):
            bw += ni * blocks[j]
    p_b = m / (ratio * tw + bw)
    p_w = ratio * p_b
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            p = p_w if lab[u] == lab[v] else p_b
            if rng.random() < p:
                G.add_edge(u, v)
    return G


def kronecker(n, m, seed, levels=10, iters=14):
    """NOT IN THE LADDER -- see the module docstring.  A 2x2 Kronecker initiator
    cannot reach density 0.137 at N = 1,024 with any heavy-tailed shape, so the
    only fit is near-uniform, which duplicates `er`.  Retained so the arithmetic
    is checkable and so a later attempt at a larger initiator has a starting point.

    Stochastic Kronecker with a 2x2 initiator, on 2**levels nodes, trimmed to n
    by keeping the highest-degree vertices.

    THE FIT MUST BE DONE AFTER THE TRIM.  A first version scaled the initiator so
    the EXPECTED edge count at 2**levels matched the target and then trimmed --
    which overshot by 102%, because keeping the top-n vertices by degree keeps the
    densest part of the graph.  The scale is therefore bisected on the ACHIEVED
    post-trim edge count, which costs one generation per bisection step.

    Kronecker graphs are defined on powers of two, so n = 1,015 is reached by a
    trim from 1,024 either way; that is reported in the manifest.
    """
    base = np.array([[0.9, 0.5], [0.5, 0.2]])
    N = 2 ** levels
    ub = ((np.arange(N)[:, None] >> np.arange(levels)) & 1).astype(int)

    def sample(scale):
        rng = np.random.default_rng(seed)
        P = np.clip(base * scale, 0.0, 1.0)
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for u in range(N):
            pr = np.prod(P[ub[u][None, :], ub], axis=1)
            pr[:u + 1] = 0.0
            hit = np.flatnonzero(rng.random(N) < pr)
            G.add_edges_from((u, int(v)) for v in hit)
        keep = sorted(G.nodes(), key=lambda v: -G.degree(v))[:n]
        H = G.subgraph(keep).copy()
        H.remove_edges_from(nx.selfloop_edges(H))
        return nx.convert_node_labels_to_integers(H, ordering='sorted')

    lo, hi = 0.2, 1.2
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        H = sample(mid)
        got = H.number_of_edges()
        if best is None or abs(got - m) < abs(best[1] - m):
            best = (H, got)
        if abs(got - m) / m < 0.01:
            return H
        if got < m:
            lo = mid
        else:
            hi = mid
    return best[0]


# ───────────────────────────────────────────────────────────── tier 2 builders

def configuration(n, m, seed, degseq=None):
    """Configuration model on Budapest's exact degree sequence, simplified.
    Simplifying removes self-loops and parallel edges, so m falls slightly short;
    the achieved value is reported."""
    if degseq is None:
        degseq = sorted((d for _, d in reference().degree()), reverse=True)
    G = nx.configuration_model(list(degseq), seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def chung_lu(n, m, seed, degseq=None):
    """Expected-degree (Chung-Lu) graph on Budapest's degree sequence."""
    if degseq is None:
        degseq = [d for _, d in reference().degree()]
    G = nx.expected_degree_graph(list(degseq), seed=seed, selfloops=False)
    return nx.Graph(G)


def rewired_budapest(n, m, seed, swaps=20):
    """Degree-preserving rewiring of Budapest itself: the exact degree sequence
    and nothing else.  This is the audit's own null model, and the strongest
    tier-2 comparison available."""
    B = reference()
    G = B.copy()
    nx.double_edge_swap(G, nswap=swaps * G.number_of_edges(),
                        max_tries=swaps * G.number_of_edges() * 20, seed=seed)
    return G


# ───────────────────────────────────────────────────────────── tier 3 builders

def dcsbm(n, m, seed, labels, degseq=None):
    """Degree-corrected SBM.  Block mixing matrix taken from Budapest under the
    SAME labels, degrees from Budapest, so the model carries both the degree
    sequence and the block structure and differs from the reference only in
    which specific pairs are joined."""
    rng = np.random.default_rng(seed)
    B = reference()
    if degseq is None:
        degseq = np.array([d for _, d in B.degree()], float)
    else:
        degseq = np.asarray(degseq, float)
    K = int(labels.max()) + 1
    # observed block edge counts under these labels, measured on the reference
    e = np.zeros((K, K))
    blab, _ = budapest_partition(B)
    for u, v in B.edges():
        a, b = blab[u], blab[v]
        e[a, b] += 1
        e[b, a] += 1
    # per-block degree totals under the TARGET labels
    kap = np.array([degseq[labels == k].sum() for k in range(K)])
    kap[kap == 0] = 1.0
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # expected edge probability between u,v: theta_u theta_v e[ru,rv] / (kap_ru kap_rv)
    for u in range(n):
        ru = labels[u]
        pr = degseq[u] * degseq * e[ru, labels] / (kap[ru] * kap[labels])
        pr[:u + 1] = 0.0
        pr = np.clip(pr, 0, 1)
        hit = np.flatnonzero(rng.random(n) < pr)
        G.add_edges_from((u, int(v)) for v in hit)
    return G


# ───────────────────────────────────────────────────────────────── the registry

def build_registry():
    B = reference()
    n, m = B.number_of_nodes(), B.number_of_edges()
    bud_lab, bud_q = budapest_partition(B)
    con_lab = construction_blocks(n)
    return n, m, [
        # name                     tier  builder
        ('er',                     1, lambda s: er(n, m, s),
         'Erdos-Renyi G(n,m); n and m exact'),
        ('watts_strogatz',         1, lambda s: watts_strogatz(n, m, s),
         'Watts-Strogatz, p=0.1; k forced even so m is approximate'),
        ('random_geometric',       1, lambda s: random_geometric(n, m, s),
         'random geometric in the unit square; radius bisected to hit m'),
        ('modular_random',         1, lambda s: modular_random(n, m, s),
         'four blocks at the construction sizes; within:between density ratio '
         'taken from Budapest, scaled to hit m'),
        ('configuration',          2, lambda s: configuration(n, m, s),
         "configuration model on Budapest's exact degree sequence, simplified"),
        ('chung_lu',               2, lambda s: chung_lu(n, m, s),
         "Chung-Lu expected-degree graph on Budapest's degree sequence"),
        ('rewired_budapest',       2, lambda s: rewired_budapest(n, m, s),
         "degree-preserving rewiring of Budapest itself; the audit's own null"),
        ('dcsbm_construction',     3, lambda s: dcsbm(n, m, s, con_lab),
         "degree-corrected SBM: Budapest's degrees, the CONSTRUCTION's "
         "307/306/201/201 blocks. THE PRIMARY TIER-3 BASELINE"),
        ('dcsbm_budapest',         3, lambda s: dcsbm(n, m, s, bud_lab),
         "degree-corrected SBM: Budapest's degrees AND Budapest's own Louvain "
         "blocks. Near-circular; included so that is visible"),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--only', default=None, help='build just this baseline')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    n, m, reg = build_registry()
    if a.list:
        for name, tier, _, desc in reg:
            print(f"  tier {tier}  {name:22} {desc}")
        return

    os.makedirs(a.out, exist_ok=True)
    print(f"targets from the reference: n={n}  m={m:,}  density={2*m/(n*(n-1)):.4f}\n")
    manifest = []
    for name, tier, fn, desc in reg:
        if a.only and name != a.only:
            continue
        print(f"  building {name} (tier {tier}) ...", flush=True)
        G = fn(a.seed)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        gn, gm = G.number_of_nodes(), G.number_of_edges()
        deg = np.array([d for _, d in G.degree()], float)
        E = np.array(sorted((min(u, v), max(u, v)) for u, v in G.edges()), dtype=np.int64)
        p = os.path.join(a.out, f'{name}.npz')
        np.savez_compressed(p, edges=E, n_total=gn)
        row = dict(name=name, tier=tier, description=desc, seed=a.seed,
                   n=gn, m=gm, density=float(2 * gm / (gn * (gn - 1))),
                   deg_mean=float(deg.mean()), deg_sd=float(deg.std()),
                   deg_min=int(deg.min()), deg_max=int(deg.max()),
                   connected=bool(nx.is_connected(G)),
                   target_n=n, target_m=m,
                   m_error_pct=float(100.0 * (gm - m) / m))
        manifest.append(row)
        print(f"    n={gn} m={gm:,} ({row['m_error_pct']:+.2f}% vs target)  "
              f"deg {deg.mean():.1f}+-{deg.std():.1f} [{int(deg.min())},{int(deg.max())}]  "
              f"connected={row['connected']}")
    mp = os.path.join(a.out, 'manifest.json')
    json.dump(manifest, open(mp, 'w'), indent=1)
    print(f"\nwrote {len(manifest)} baselines and {mp}")
    print("Reference for comparison: n=1015 m=70,654 deg 139.22+-71.55 [7,466]")


if __name__ == '__main__':
    main()
