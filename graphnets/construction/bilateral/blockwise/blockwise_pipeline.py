#!/usr/bin/env python3
"""
blockwise_pipeline.py
=====================

Builds a chromatically constrained tensor graph and coarsens it using nothing
but Louvain community detection, applied independently within each block.
No hand-written merge rule anywhere.

    PARENT     8,000 vertices, ~478k edges, properly 3-colourable
    QUOTIENT   ~1,076 supernodes, ~73k edges

Of the coarsenings tested, this scores best on the six graph tests: 0.182,
against 0.311 for a global resolution sweep and 0.244 for a variant using a
SMALLEST-FIRST local merge rule.  Restricting Louvain to blocks is what does the
work.

  CLARIFIED 2026-09-25.  The 0.244 above is NOT construct.py's rule.  That one
  merges the globally densest adjacent pair, and its quotient -- the shipped
  graphs/cnew_coarse.npz -- measures 0.2009, which is the 0.201 quoted in
  ../README.md.  Reproduce with
      python3 static_tests.py --graph graphs/cnew_coarse.npz \
          --budapest reference/budapest_1015_70654.edgelist --label "8k quotient" \
          --out /tmp/st.png
  The smallest-first variant behind 0.244 is not shipped here, so that figure
  cannot be reproduced from this repository; it is reported as history. The two
  rules are easy to confuse -- see "The merge rule" in ../README.md.


WHY THREE COLOURS
-----------------
Read a vertex as a computational unit and an edge as a dependency: two units
joined by an edge cannot update in the same phase, so the minimum number of
sequential phases is the chromatic number.  Here it is exactly 3, at every
scale, however large the graph grows.  The colouring is a property of the
parent and does not survive coarsening.


PIPELINE
--------
Stage 0  SEED       20 vertices, 31 edges.  Colour classes (6,6,8); blocks
                    (6,6,4,4) in two hemispheres {A,C} and {B,D}.  Edge counts
                    26 within-block : 4 same-hemisphere : 1 cross-hemisphere.

Stage 1  GROW       Three Kronecker powers with triangle closure after each,
                    closing alpha_k * n edges at step k, alpha = (3,4,30).
                    -> the PARENT.

Stage 2  VERIFY     Louvain at low resolution, with no block labels supplied,
                    returns 2400/2400/1600/1600 at NMI = ARI = 1.000.

Stage 3  COARSEN    Within each block independently, sweep the Louvain
                    resolution and keep whichever partition comes closest to
                    that block's share of the target count.  Take it as it
                    stands.  Two supernodes are joined whenever at least one
                    parent edge runs between them.
                    -> the QUOTIENT.

The per-block targets are the parent's own block proportions -- 2400/2400/
1600/1600 of 8,000 scaled to the target count -- giving 304/304/203/203.  They
are not Budapest's 307/306/201/201.  Nothing in the coarsening uses a quantity
taken from any connectome.

Louvain has no mechanism for reaching a prescribed community count: it moves
each node to whichever neighbouring community most increases modularity,
collapses the result, and repeats until modularity stops improving.  The count
is whatever falls out.  Sweeping the resolution is the standard way to steer
it, and it lands near but not on the target -- here 1,076 against 1,015.


TWO FACTS THAT MAKE STAGE 1 WORK
--------------------------------
Colour needs no repair.  Every product edge is cross-colour because every seed
edge is, and every closure is filtered to be, so chi = 3 survives without a
check and no repair operator appears anywhere.

Blocks ride on the first coordinate.  A vertex of G_k is written in base 20 as
(u_1,...,u_k) and expansion appends digits on the right, so u_1 is preserved
and block membership is read off it.  A within-block edge added at step k is
therefore still within-block at step k+1, which is why closure lives INSIDE
the recursion.  The within-block to cross-block density contrast is 5.3 in the
pure tensor power and 28.5 with in-loop closure; below roughly 10 the blocks
stop being detectable.


WHAT IS FITTED
--------------
Parent:    three growth scalars, alpha = (3,4,30), plus the closure sampling
           exponent beta.
Quotient:  a target community count, used only to choose which resolution to
           keep from each block's sweep.  The per-block split is derived from
           the parent, not supplied.

The seed's four-block pattern was designed by hand to mirror the connectome's
block densities of roughly 0.45 : 0.09 : 0.003.  Bilateral organisation is
accommodated, not predicted; what is shown is that it survives growth and is
recoverable blind.


A NOTE ON WHAT BLOCK-WISE CLUSTERING DOES AND DOES NOT SHOW
------------------------------------------------------------
Because each block is clustered separately, every supernode lies inside one
block by construction.  Community detection on the finished quotient therefore
recovers the blocks trivially, and that recovery is NOT evidence of anything.
The meaningful emergence result is at the parent scale, where Louvain is given
only the edge list.

What block-wise clustering does buy, measurably: the three block pairs that
are exactly zero in the parent stay exactly zero in the quotient.  Under a
global resolution sweep, 11 of 992 supernodes straddle block boundaries and
that is enough to put density into all six off-diagonal pairs, where the
connectome has two.


USAGE
-----
    python blockwise_pipeline.py
    python blockwise_pipeline.py --out run
    python blockwise_pipeline.py --budapest edges.csv
    python blockwise_pipeline.py --parent-only

Requires numpy, scipy, networkx.  pandas only for --budapest; scikit-learn
only for the NMI/ARI lines.
"""

from __future__ import annotations

import argparse

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from networkx.algorithms.community import louvain_communities, modularity

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    HAVE_SK = True
except ImportError:                                   # pragma: no cover
    HAVE_SK = False


# ===========================================================================
# STAGE 0 -- the seed.  Everything is generated from this.
# ===========================================================================

SEED_N = 20

COLOURS = np.array([0, 0, 1, 1, 2, 2,       # block A
                    0, 0, 1, 1, 2, 2,       # block B
                    0, 1, 2, 2,             # block C
                    0, 1, 2, 2],            # block D
                   dtype=np.int32)

BLOCKS = np.array([0] * 6 + [1] * 6 + [2] * 4 + [3] * 4, dtype=np.int32)

_A, _B, _C, _D = range(0, 6), range(6, 12), range(12, 16), range(16, 20)

#: within-block edges for the two MAJOR blocks (8 edges on 6 vertices).
#: Four of the eight sit on vertex 0, giving seed degrees spanning 1..6 rather
#: than 2..3.  Wider seed degree spread propagates to the parent and raises
#: quotient clustering at no cost to block recovery.
MAJOR_PATTERN = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (2, 4)]

#: within-block edges for the two MINOR blocks (5 edges on 4 vertices).
#: This is the only legal cross-colour connected pattern of this size.  Seeds
#: whose minor blocks have two or three vertices collapse to the two
#: hemispheres under blind detection; four suffices.
MINOR_PATTERN = [(0, 1), (0, 2), (1, 2), (1, 3), (0, 3)]


def build_seed() -> list[tuple[int, int]]:
    """The 31 seed edges.

        26 within-block    (8 + 8 major, 5 + 5 minor)
         4 same-hemisphere (2 joining A-C, 2 joining B-D)
         1 cross-hemisphere (A-B)

    Note which pairs are ABSENT: no edge joins A-D, B-C or C-D.  Since the
    tensor product creates an edge between two fibres only when their seed
    vertices are adjacent, those three block pairs have density exactly zero at
    every Kronecker power.  The interhemispheric bottleneck is inherited rather
    than imposed -- the one structural property here that is derived.
    """
    def place(block, pattern):
        idx = list(block)
        return [(idx[i], idx[j]) for i, j in pattern]

    edges = (place(_A, MAJOR_PATTERN) + place(_B, MAJOR_PATTERN)
             + place(_C, MINOR_PATTERN) + place(_D, MINOR_PATTERN)
             + [(_A[0], _C[2]), (_A[4], _C[0])]        # hemisphere R: A-C
             + [(_B[0], _D[2]), (_B[4], _D[0])]        # hemisphere L: B-D
             + [(_A[0], _B[2])])                        # cross-hemisphere

    for u, v in edges:
        assert COLOURS[u] != COLOURS[v], f"seed edge {(u, v)} is monochromatic"
    return edges


# ===========================================================================
# STAGE 1 -- growth with in-loop triangle closure  ->  PARENT
# ===========================================================================

def close_triangles(u_arr, v_arr, n, colour, block, quota, rng, beta):
    """Add `quota` chromatically legal, within-block, triangle-closing edges.

    Each candidate: pick a vertex v with probability proportional to
    deg(v)**beta, take two of its neighbours u and w, and add {u,w} if

        colour(u) != colour(w)      chromatic legality
        block(u)  == block(w)       stays inside a block
        {u,w} is not already an edge

    Selection within the legal pool is deliberately unranked.  Ranking
    candidates by common-neighbour support was tested and is worse: it sends
    every closure to the few densest neighbourhoods, leaving within-block
    density uneven and dropping blind block recovery to ARI 0.53.
    """
    if quota <= 0:
        return u_arr, v_arr

    adj = csr_matrix((np.ones(len(u_arr), np.int8), (u_arr, v_arr)), shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1
    indptr, indices = adj.indptr, adj.indices
    degree = np.diff(indptr)

    seen = set((np.minimum(u_arr, v_arr).astype(np.int64) * n
                + np.maximum(u_arr, v_arr).astype(np.int64)).tolist())

    eligible = np.where(degree >= 2)[0]
    if not len(eligible):
        return u_arr, v_arr
    weight = degree[eligible].astype(float) ** beta
    weight /= weight.sum()

    accepted, remaining = [], quota
    for _ in range(24):
        if remaining <= 0:
            break
        draw = int(remaining * 3) + 10
        centre = rng.choice(eligible, size=draw, p=weight)
        deg_c = degree[centre]
        nb1 = indices[indptr[centre] + (rng.random(draw) * deg_c).astype(int)]
        nb2 = indices[indptr[centre] + (rng.random(draw) * deg_c).astype(int)]

        ok = ((nb1 != nb2)
              & (colour[nb1] != colour[nb2])
              & (block[nb1] == block[nb2]))
        nb1, nb2 = nb1[ok], nb2[ok]
        if not len(nb1):
            break

        keys = np.unique(np.minimum(nb1, nb2).astype(np.int64) * n
                         + np.maximum(nb1, nb2).astype(np.int64))
        keys = np.array([k for k in keys if k not in seen], np.int64)[:remaining]
        if not len(keys):
            break
        seen.update(keys.tolist())
        accepted.append(keys)
        remaining -= len(keys)

    if not accepted:
        return u_arr, v_arr
    new = np.concatenate(accepted)
    return (np.concatenate([u_arr, new // n]),
            np.concatenate([v_arr, new % n]))


def grow(seed_edges, alphas, beta, n_steps, rng, verbose=True):
    """Kronecker powers with closure interleaved.  Returns the parent.

    The tensor product G (x) S has vertex set V(G) x V(S), with (u,a) ~ (v,b)
    exactly when u ~ v in G and a ~ b in S.  Vertex (u,a) is stored as the
    integer u*20 + a, so the first base-20 digit is most significant and is
    preserved by every further expansion.
    """
    su = np.array([a for a, b in seed_edges] + [b for a, b in seed_edges], np.int64)
    sv = np.array([b for a, b in seed_edges] + [a for a, b in seed_edges], np.int64)

    u_arr = np.array([a for a, b in seed_edges], np.int64)
    v_arr = np.array([b for a, b in seed_edges], np.int64)
    n = SEED_N
    colour, block = COLOURS.copy(), BLOCKS.copy()

    before = len(u_arr)
    u_arr, v_arr = close_triangles(u_arr, v_arr, n, colour, block,
                                   int(alphas[0] * n), rng, beta)
    if verbose:
        print(f"    seed       n={n:>6,}   product m={before:>7,}"
              f"   closure +{len(u_arr)-before:>7,}   = {len(u_arr):>7,}")

    for step in range(2, n_steps + 1):
        u_arr = (u_arr[:, None] * SEED_N + su[None, :]).ravel()
        v_arr = (v_arr[:, None] * SEED_N + sv[None, :]).ravel()
        lo, hi = np.minimum(u_arr, v_arr), np.maximum(u_arr, v_arr)
        keys = np.unique(lo * (SEED_N ** step) + hi)
        u_arr, v_arr = keys // (SEED_N ** step), keys % (SEED_N ** step)

        n = SEED_N ** step
        digit = (np.arange(n) // (SEED_N ** (step - 1))) % SEED_N
        colour, block = COLOURS[digit], BLOCKS[digit]

        before = len(u_arr)
        u_arr, v_arr = close_triangles(u_arr, v_arr, n, colour, block,
                                       int(alphas[step - 1] * n), rng, beta)
        if verbose:
            print(f"    tensor {step}   n={n:>6,}   product m={before:>7,}"
                  f"   closure +{len(u_arr)-before:>7,}   = {len(u_arr):>7,}")

    assert (colour[u_arr] == colour[v_arr]).sum() == 0, "chromatic violation"
    return u_arr.astype(np.int32), v_arr.astype(np.int32), colour, block


# ===========================================================================
# STAGE 3 -- block-wise Louvain, no merge  ->  QUOTIENT
# ===========================================================================

def coarsen_blockwise(graph, block, target, resolutions, seed=42, verbose=True):
    """Coarsen each block independently by Louvain alone.

    Each block's target is its share of `target`, computed from the parent's
    own block proportions.  Within a block, the resolution sweep keeps
    whichever partition comes closest; no merge step follows.

    Because each block is clustered separately, every supernode lies inside one
    block by construction.  That makes block recovery on the quotient trivial
    and therefore not evidence of anything -- but it does keep the three
    exactly-zero block pairs of the parent exactly zero in the quotient, which
    a global sweep does not.
    """
    n = graph.number_of_nodes()
    counts = np.bincount(block)
    targets = [int(round(c / n * target)) for c in counts]
    if verbose:
        print(f"  per-block targets from parent proportions: {targets}"
              f"  (sum {sum(targets)})")

    label = np.full(n, -1, np.int32)
    offset = 0
    for b, t in enumerate(targets):
        nodes = np.where(block == b)[0]
        sub = graph.subgraph(nodes)
        rename = {v: i for i, v in enumerate(nodes)}
        relabelled = nx.relabel_nodes(sub, rename)

        best = None
        for gamma in resolutions:
            comms = louvain_communities(relabelled, resolution=gamma, seed=seed)
            if best is None or abs(len(comms) - t) < abs(best[1] - t):
                best = (gamma, len(comms), comms)
        gamma, count, comms = best
        if verbose:
            print(f"    block {b}: {len(nodes):>5,} vertices, target {t:>4}"
                  f"  ->  gamma {gamma:>5.1f} gives {count:>4} communities"
                  f"  (miss {abs(count-t)/t:.1%})")

        for i, comm in enumerate(comms):
            for v in comm:
                label[nodes[v]] = i + offset
        offset += count

    if verbose:
        print(f"  total supernodes: {offset:,}   (target {target:,},"
              f" miss {abs(offset-target)/target:.1%})")
    return label, offset


def build_quotient(u_arr, v_arr, label, n_super):
    """Two supernodes are joined whenever at least one parent edge runs between
    them.  Nothing is thinned.

    The witness weight of a supernode pair -- the number of parent edges
    between them -- is returned because it records how strongly each quotient
    edge is supported, but it is not used to filter.
    """
    a, b = label[u_arr], label[v_arr]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    cross = lo != hi
    key = lo[cross].astype(np.int64) * n_super + hi[cross].astype(np.int64)
    pairs, witness = np.unique(key, return_counts=True)

    quotient = nx.Graph()
    quotient.add_nodes_from(range(n_super))
    quotient.add_edges_from((int(k // n_super), int(k % n_super)) for k in pairs)
    return quotient, pairs, witness, int((~cross).sum())


# ===========================================================================
# reporting
# ===========================================================================

def verify_parent(graph, u_arr, v_arr, colour, block, resolution=0.2, seed=7):
    n = graph.number_of_nodes()
    degree = np.array([graph.degree(v) for v in range(n)])

    print(f"\n  n = {n:,}   m = {graph.number_of_edges():,}"
          f"   mean degree {degree.mean():.1f}"
          f"   CV {degree.std() / degree.mean():.3f}")

    bad = int((colour[u_arr] == colour[v_arr]).sum())
    has_triangle = sum(nx.triangles(graph).values()) > 0
    print(f"  chi = 3:          {bad} same-colour edges (=> chi <= 3);"
          f" triangle present: {has_triangle} (=> chi >= 3)")
    print(f"  colour classes:   {list(np.bincount(colour))}")
    print(f"  connected:        {nx.number_connected_components(graph) == 1}")

    parts = sorted(louvain_communities(graph, resolution=resolution, seed=seed),
                   key=len, reverse=True)
    found = np.empty(n, int)
    for i, comm in enumerate(parts):
        for v in comm:
            found[v] = i
    line = (f"  blind Louvain:    {len(parts)} communities "
            f"{[len(p) for p in parts][:6]}")
    if HAVE_SK:
        line += (f"\n                    NMI={normalized_mutual_info_score(found, block):.4f}"
                 f"  ARI={adjusted_rand_score(found, block):.4f}"
                 f"   (true blocks {list(np.bincount(block))})")
    print(line)

    sizes_b = np.bincount(block)
    counts = np.zeros((4, 4))
    for a, b in zip(block[u_arr], block[v_arr]):
        counts[a, b] += 1
        counts[b, a] += 1
    for i in range(4):
        counts[i, i] /= 2
    print("\n  block density matrix (A, B, C, D):")
    for i in range(4):
        row = "".join(
            f"{counts[i, j] / (sizes_b[i] * sizes_b[j] if i != j else sizes_b[i] * (sizes_b[i] - 1) / 2):>9.4f}"
            for j in range(4))
        print(f"   {sizes_b[i]:>6}{row}")
    names = "ABCD"
    zeros = [f"{names[i]}-{names[j]}" for i in range(4) for j in range(i + 1, 4)
             if counts[i, j] == 0]
    print(f"  exactly zero:     {', '.join(zeros)}"
          f"   (inherited from the seed at every power)")


def verify_quotient(quotient, label, witness, internal, block, u_arr, v_arr):
    parts = sorted(louvain_communities(quotient, seed=1), key=len, reverse=True)
    degree = np.array([quotient.degree(v)
                       for v in range(quotient.number_of_nodes())])
    sizes = np.bincount(label)
    print(f"\n  n = {quotient.number_of_nodes():,}"
          f"   m = {quotient.number_of_edges():,}"
          f"   density {nx.density(quotient):.4f}"
          f"   mean degree {degree.mean():.1f}   sd {degree.std():.1f}")
    print(f"  supernode sizes:  {sizes.min()} to {sizes.max()} parent vertices"
          f"   (median {int(np.median(sizes))})")
    print(f"  witness weight:   median {int(np.median(witness))}"
          f"   mean {witness.mean():.2f}   max {witness.max()}")
    print(f"  parent edges absorbed inside supernodes: {internal:,}")
    print(f"  communities found: {[len(p) for p in parts]}"
          f"    (trivially block-aligned: see the module docstring)")

    # the quotient's own block density matrix
    sb = np.zeros(quotient.number_of_nodes(), np.int8)
    for b in range(4):
        sb[np.unique(label[block == b])] = b
    sz = np.bincount(sb)
    C = np.zeros((4, 4))
    for u, v in quotient.edges():
        C[sb[u], sb[v]] += 1
        C[sb[v], sb[u]] += 1
    for i in range(4):
        C[i, i] /= 2
    print("\n  quotient block density matrix (A, B, C, D):")
    for i in range(4):
        row = "".join(
            f"{C[i, j] / (sz[i] * sz[j] if i != j else sz[i] * (sz[i] - 1) / 2):>9.4f}"
            for j in range(4))
        print(f"   {sz[i]:>6}{row}")


def compare(quotient, reference):
    def summarise(g):
        n = g.number_of_nodes()
        adj = nx.to_scipy_sparse_array(g, format="csr").astype(float)
        dist = shortest_path(adj, unweighted=True, method="D")
        finite = dist[(dist > 0) & np.isfinite(dist)]
        eff_diam = float("nan")
        for h in range(1, 15):
            if (finite <= h).mean() >= 0.9:
                prev = (finite <= h - 1).mean()
                eff_diam = h - 1 + (0.9 - prev) / max((finite <= h).mean() - prev, 1e-12)
                break
        deg = np.array([g.degree(v) for v in range(n)])
        return dict(n=n, m=g.number_of_edges(), density=nx.density(g),
                    mean=deg.mean(), sd=deg.std(),
                    lo=int(deg.min()), hi=int(deg.max()),
                    clust=nx.average_clustering(g), trans=nx.transitivity(g),
                    Q=modularity(g, louvain_communities(g, seed=1)),
                    apl=finite.mean(), ed=eff_diam,
                    eff=nx.global_efficiency(g))

    a, b = summarise(quotient), summarise(reference)
    rows = [("nodes", "n"), ("edges", "m"), ("density", "density"),
            ("mean degree", "mean"), ("degree sd", "sd"),
            ("min degree", "lo"), ("max degree", "hi"),
            ("clustering", "clust"), ("transitivity", "trans"),
            ("modularity", "Q"), ("avg path length", "apl"),
            ("effective diameter", "ed"), ("global efficiency", "eff")]
    print(f"\n  {'':<22}{'quotient':>14}{'reference':>14}{'ratio':>9}")
    for name, key in rows:
        fa = f"{a[key]:.4f}" if isinstance(a[key], float) else f"{a[key]:,}"
        fb = f"{b[key]:.4f}" if isinstance(b[key], float) else f"{b[key]:,}"
        ratio = a[key] / b[key] if b[key] else float("nan")
        print(f"  {name:<22}{fa:>14}{fb:>14}{ratio:>9.3f}")


def load_reference(path):
    import pandas as pd
    frame = pd.read_csv(path, comment="#", header=None, sep=None, engine="python")
    try:
        frame.iloc[0, :2].astype(np.int64)
    except (ValueError, TypeError):
        frame = frame.iloc[1:]
    edges = frame.iloc[:, :2].to_numpy(dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    g = nx.Graph()
    g.add_nodes_from(range(int(edges.max()) + 1))
    g.add_edges_from(map(tuple, edges))
    return g


# ===========================================================================
# driver
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--alpha", type=float, nargs="+", default=[3, 4, 30])
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--target", type=int, default=1015,
                   help="total community count to steer the sweeps toward")
    p.add_argument("--resolutions", type=float, nargs="+",
                   default=[2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 55],
                   help="resolutions to try in each block's sweep")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--parent-only", action="store_true")
    p.add_argument("--budapest", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if len(args.alpha) != args.steps:
        p.error(f"--alpha needs {args.steps} values, got {len(args.alpha)}")

    rng = np.random.default_rng(args.seed)

    seed_edges = build_seed()
    print("=" * 74)
    print(f"STAGE 0  SEED    {SEED_N} vertices, {len(seed_edges)} edges")
    print(f"  colours {list(np.bincount(COLOURS))}"
          f"   blocks {list(np.bincount(BLOCKS))}")
    print("  26 within-block : 4 same-hemisphere : 1 cross-hemisphere")

    print("=" * 74)
    print(f"STAGE 1  GROW    alpha={args.alpha}  beta={args.beta}  seed={args.seed}")
    u_arr, v_arr, colour, block = grow(seed_edges, args.alpha, args.beta,
                                       args.steps, rng)
    n = SEED_N ** args.steps
    parent = nx.Graph()
    parent.add_nodes_from(range(n))
    parent.add_edges_from(zip(u_arr, v_arr))

    print("=" * 74)
    print("STAGE 2  VERIFY THE PARENT")
    verify_parent(parent, u_arr, v_arr, colour, block)

    quotient = label = None
    if not args.parent_only:
        print("=" * 74)
        print(f"STAGE 3  BLOCK-WISE LOUVAIN, NO MERGE   (target {args.target:,})")
        label, count = coarsen_blockwise(parent, block, args.target,
                                         args.resolutions)
        quotient, pairs, witness, internal = build_quotient(u_arr, v_arr,
                                                            label, count)
        verify_quotient(quotient, label, witness, internal, block, u_arr, v_arr)

        if args.budapest:
            print("=" * 74)
            print("COMPARISON")
            compare(quotient, load_reference(args.budapest))
    print("=" * 74)

    if args.out:
        pre = args.out
        np.savetxt(f"{pre}_parent_edges.csv", np.stack([u_arr, v_arr]).T,
                   fmt="%d", delimiter=",", header="u,v", comments="")
        np.savetxt(f"{pre}_colours.csv", colour, fmt="%d",
                   header="chromatic_class", comments="")
        np.savetxt(f"{pre}_blocks.csv", block, fmt="%d", header="block",
                   comments="")
        written = ["_parent_edges", "_colours", "_blocks"]
        if quotient is not None:
            np.savetxt(f"{pre}_partition.csv", label, fmt="%d",
                       header="supernode", comments="")
            arr = np.array([[int(k // count), int(k % count), w]
                            for k, w in zip(pairs, witness)])
            np.savetxt(f"{pre}_quotient_edges.csv", arr, fmt="%d",
                       delimiter=",", header="A,B,witness", comments="")
            written += ["_partition", "_quotient_edges"]
        print("wrote " + ", ".join(f"{pre}{s}.csv" for s in written))


if __name__ == "__main__":
    main()
