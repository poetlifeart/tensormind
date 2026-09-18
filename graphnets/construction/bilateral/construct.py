#!/usr/bin/env python3
"""
construct.py
============

Complete, self-contained.  Builds both graphs from nothing but the seed
defined in this file:

    PARENT GRAPH    8,000 vertices, ~478k edges, properly 3-colourable
    COARSE GRAPH    1,015 supernodes, ~65k edges, the community quotient

No input files are needed.  A Budapest edge list is optional and only
enables the comparison table at the end.


WHAT IS BEING CLAIMED, AND AT WHICH SCALE
-----------------------------------------
The parent graph is the object the construction produces.  The coarse graph
is a *view* of it -- a grouping of its vertices into supernodes -- so
different groupings give different coarse graphs while the parent is
unchanged.  Claims about the parent are claims about the construction;
claims about the coarse graph are claims about the construction plus a
choice of view.  The chromatic property lives at the parent scale and does
not survive coarsening: the coarse graph contains a 55-clique, so its
chromatic number is at least 55.

Read a parent vertex as a computational unit and an edge as a dependency:
two units joined by an edge cannot update in the same phase, so the minimum
number of sequential phases is the chromatic number.  Here it is exactly 3
at every scale, however large the graph grows.


PIPELINE
--------
Stage 0  SEED       20 vertices, 31 edges.  Colour classes (6,6,8); blocks
                    (6,6,4,4) grouped into hemispheres {A,C} and {B,D}.
                    Edge counts 26 within-block : 4 same-hemisphere :
                    1 cross-hemisphere.

Stage 1  GROW       Three Kronecker powers with triangle closure after each:
                    G1 = S;  G2 = G1 (x) S;  G3 = G2 (x) S, closing
                    alpha_k * n edges at step k, alpha = (3,4,30).
                    -> the PARENT GRAPH.

Stage 2  VERIFY     Louvain on the parent with no block labels supplied
                    returns 2400/2400/1600/1600 at NMI = ARI = 1.000.

Stage 3  COARSEN    Group the parent's vertices into 1,015 supernodes:
                    Louvain within each block, then merge the densest
                    adjacent pair repeatedly until each block hits its
                    target count.  Two supernodes are joined whenever at
                    least one parent edge runs between them.
                    -> the COARSE GRAPH.

There is no fourth stage.  Earlier versions thinned the coarse graph to a
target density matrix read from Budapest; measurement showed that stage was
discarding only 3% of candidate pairs, almost all of them weak
cross-hemisphere ones, and that keeping everything is closer to the target
on modularity, path length, effective diameter and the degree
distribution.  Dropping it removes ten fitted numbers and one tuned
parameter, so only the four block sizes remain.


TWO FACTS THAT MAKE STAGE 1 WORK
--------------------------------
Colour needs no repair.  Every product edge is cross-colour because every
seed edge is, and every closure is filtered to be.  So chi = 3 survives
without a check and no repair operator appears anywhere.

Blocks ride on the first coordinate.  A vertex of G_k is written in base 20
as (u_1,...,u_k) and expansion appends digits on the right, so u_1 is
preserved and block membership is read off it.  A within-block edge added at
step k is therefore still within-block at step k+1, which is why closure can
live INSIDE the recursion rather than being applied afterwards.  That
ordering is not cosmetic: early closures are tensor-multiplied by later
steps, and the block-density contrast rises from 5.3 (pure product) to 28.5,
which is what makes the blocks detectable at all.


WHAT IS FITTED
--------------
Parent graph:  three growth scalars alpha = (3,4,30), plus beta.  Nothing
               taken from any connectome.
Coarse graph:  four more numbers, the block sizes 307/306/201/201, read from
               Budapest's own Louvain partition.

The seed's four-block pattern was designed by hand to mirror the
connectome's block densities of roughly 0.45 : 0.09 : 0.003.  So bilateral
organisation is accommodated, not predicted; what is shown is that it
survives growth and is recoverable blind.

NOT fitted, and therefore consequences rather than targets: modularity, path
length, effective diameter, global efficiency, clustering, transitivity,
betweenness, the degree distribution, and the spectrum.


WHAT IS NOT REPRODUCED
----------------------
Density is about 8% short of Budapest and the degree distribution is too
narrow -- sd ~61 against 71.5, minimum degree ~33 against 7.  Diagnosis:
coarse degree tracks the mean parent degree of a supernode's members
(r = 0.887), and averaging ~8 parent vertices per supernode compresses the
spread.  Several mechanisms were tried against this; each either failed to
move it or destroyed the emergent partition.  The density shortfall is
upstream: most block pairs are support-limited, meaning the parent does not
generate enough distinct supernode-to-supernode connections, so adding
parent edges would only help if they joined new supernode pairs rather than
thickening existing ones.


USAGE
-----
    python construct.py                        # build both, verify, print
    python construct.py --out run              # also write run_*.csv
    python construct.py --budapest edges.csv   # add the comparison table
    python construct.py --parent-only          # stop after the parent

Requires: numpy, scipy, networkx.  scikit-learn optional (NMI/ARI).
"""

from __future__ import annotations

import argparse
import heapq
from collections import defaultdict

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
# STAGE 0 -- the seed.  Everything below is generated from this.
# ===========================================================================

SEED_N = 20

#: chromatic class of each seed vertex; classes have sizes (6, 6, 8).
#: Every block spans all three classes, so no block is internally bipartite.
COLOURS = np.array([0, 0, 1, 1, 2, 2,       # block A
                    0, 0, 1, 1, 2, 2,       # block B
                    0, 1, 2, 2,             # block C
                    0, 1, 2, 2],            # block D
                   dtype=np.int32)

#: block membership; blocks have sizes (6, 6, 4, 4).
BLOCKS = np.array([0] * 6 + [1] * 6 + [2] * 4 + [3] * 4, dtype=np.int32)

_A, _B, _C, _D = range(0, 6), range(6, 12), range(12, 16), range(16, 20)

#: within-block edges for the two MAJOR blocks (8 edges on 6 vertices).
#: Four of the eight sit on vertex 0, giving seed degrees spanning 1..6
#: rather than 2..3.  Chosen from the 489 legal cross-colour connected
#: patterns of this size: wider seed degree spread propagates to the parent
#: (CV 0.500 -> 0.580) and raises coarse clustering, at no cost to block
#: recovery.  The commented line is the lower-spread original.
MAJOR_PATTERN = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (2, 4)]
# MAJOR_PATTERN = [(0,2),(0,4),(2,4),(1,3),(1,5),(3,5),(0,3),(2,5)]

#: within-block edges for the two MINOR blocks (5 edges on 4 vertices).
#: This is the ONLY legal cross-colour connected pattern of this size, so the
#: minor blocks admit no variation at all -- a real constraint, since they
#: supply 402 of the 1,015 supernodes and are the tighter-degreed half of the
#: coarse graph.
MINOR_PATTERN = [(0, 1), (0, 2), (1, 2), (1, 3), (0, 3)]

#: the only numbers taken from Budapest: its Louvain block sizes.
BLOCK_TARGETS = [307, 306, 201, 201]


def build_seed() -> list[tuple[int, int]]:
    """The 31 seed edges.

        26 within-block    (8 + 8 major, 5 + 5 minor)
         4 same-hemisphere (2 joining A-C, 2 joining B-D)
         1 cross-hemisphere (A-B)

    Note which pairs are ABSENT: no edge joins A-D, B-C or C-D.  Since the
    tensor product can only create an edge between fibres whose seed vertices
    are adjacent, those three block pairs have density exactly zero at every
    Kronecker power.  The interhemispheric bottleneck is therefore inherited
    rather than imposed -- the one structural property here that is derived.
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
# STAGE 1 -- growth with in-loop triangle closure  ->  PARENT GRAPH
# ===========================================================================

def close_triangles(u_arr, v_arr, n, colour, block, quota, rng, beta):
    """Add `quota` chromatically legal, within-block, triangle-closing edges.

    Each candidate: pick a vertex v with probability proportional to
    deg(v)**beta, take two of its neighbours u and w, and add {u,w} if

        colour(u) != colour(w)      chromatic legality
        block(u)  == block(w)       stay inside a block
        {u,w} not already an edge

    Sampling proportional to degree concentrates closures where the graph is
    already locally dense, which is what produces clustering; beta > 1
    concentrates further and slightly widens the degree spread.

    Selection within the legal pool is deliberately UNRANKED.  Ranking
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

    eligible = np.where(degree >= 2)[0]        # need two neighbours for a wedge
    if not len(eligible):
        return u_arr, v_arr
    weight = degree[eligible].astype(float) ** beta
    weight /= weight.sum()

    accepted, remaining = [], quota
    for _ in range(24):                        # bounded retry; duplicates common
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
    """Kronecker powers with closure interleaved.  Returns the parent graph.

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
        keys = np.unique(lo * (SEED_N ** step) + hi)          # dedupe
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
# STAGE 3 -- coarsening  ->  COARSE GRAPH
# ===========================================================================

def merge_densest(edges_2xm, communities, n, target):
    """Merge communities to `target` by repeatedly combining the pair with the
    largest normalised connection density w(X,Y)/(|X|*|Y|).

    Contrast with merging the SMALLEST community first, the more common
    choice: that equalises supernode sizes, and since coarse degree tracks
    supernode size it flattens the degree distribution.  Densest-first keeps
    more of Louvain's natural size spread (1-46 parent vertices per supernode
    rather than 4-19).

    Note what this does NOT do: no individual vertex is ever moved between
    supernodes.  A vertex is placed once by Louvain and thereafter travels
    with whichever community it was put in.  The coarsening is purely
    agglomerative.  A variant with 30 rounds of pairwise vertex exchange fits
    Budapest markedly better (KS 0.037 against 0.10) but the resulting
    partition is chosen rather than found, and fails the blind-recovery test
    at the supernode scale.
    """
    label = np.empty(n, np.int32)
    for i, comm in enumerate(communities):
        for v in comm:
            label[v] = i

    n_comm = len(communities)
    size = np.array([len(c) for c in communities], np.int64)

    a, b = label[edges_2xm[0]], label[edges_2xm[1]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    cross = lo != hi
    pair_key = lo[cross].astype(np.int64) * n_comm + hi[cross].astype(np.int64)
    uniq, count = np.unique(pair_key, return_counts=True)

    weight, adjacency = defaultdict(int), defaultdict(set)
    for key, c in zip(uniq, count):
        i, j = int(key // n_comm), int(key % n_comm)
        weight[(i, j)] = int(c)
        adjacency[i].add(j)
        adjacency[j].add(i)

    def w(i, j):
        return weight.get((i, j) if i < j else (j, i), 0)

    alive = np.ones(n_comm, bool)
    remaining = n_comm
    heap = [(-c / (size[i] * size[j]), i, j) for (i, j), c in weight.items()]
    heapq.heapify(heap)

    while remaining > target and heap:
        neg, i, j = heapq.heappop(heap)
        if not alive[i] or not alive[j]:
            continue
        current = w(i, j) / (size[i] * size[j]) if size[i] and size[j] else 0.0
        if abs(-neg - current) > 1e-12:            # stale entry, reinsert
            if current > 0:
                heapq.heappush(heap, (-current, i, j))
            continue

        for k in list(adjacency[i]):               # fold i into j
            if k == j or not alive[k]:
                continue
            key = (j, k) if j < k else (k, j)
            weight[key] = weight.get(key, 0) + w(i, k)
            adjacency[j].add(k)
            adjacency[k].add(j)
            adjacency[k].discard(i)
        adjacency[j].discard(i)
        weight.pop((i, j) if i < j else (j, i), None)
        for k in list(adjacency[i]):
            weight.pop((i, k) if i < k else (k, i), None)

        alive[i] = False
        size[j] += size[i]
        size[i] = 0
        label[label == i] = j
        for k in adjacency[j]:
            if alive[k] and size[k]:
                heapq.heappush(heap, (-w(j, k) / (size[j] * size[k]),
                                      min(j, k), max(j, k)))
        remaining -= 1

    while remaining > target:                      # fallback if the heap empties
        ids = np.where(alive)[0]
        i = ids[np.argmin(size[ids])]
        j = min((x for x in ids if x != i), key=lambda x: size[x])
        alive[i] = False
        size[j] += size[i]
        label[label == i] = j
        remaining -= 1

    ids = np.where(alive)[0]
    remap = -np.ones(n_comm, np.int32)
    remap[ids] = np.arange(len(ids))
    return remap[label]


def coarsen(graph, block, targets, seed=42):
    """Group the parent's vertices into supernodes, one block at a time.

    Clustering per block rather than globally guarantees that every supernode
    lies wholly inside one block, so the coarse graph inherits the block
    structure exactly.

    Louvain is run at rising resolution until it over-partitions; in practice
    the first rung (30) already returns roughly five times the target, so the
    ladder never advances.
    """
    n = graph.number_of_nodes()
    label = np.full(n, -1, np.int32)
    offset = 0
    for b, target in enumerate(targets):
        nodes = np.where(block == b)[0]
        sub = graph.subgraph(nodes)
        rename = {v: i for i, v in enumerate(nodes)}
        relabelled = nx.relabel_nodes(sub, rename)

        communities = None
        for resolution in (30, 80, 160, 300, 600, 1200, 2400):
            communities = louvain_communities(relabelled, resolution=resolution,
                                              seed=seed)
            if len(communities) > target:
                break

        sub_edges = np.array([[rename[x], rename[y]] for x, y in sub.edges()],
                             dtype=np.int32).T
        label[nodes] = merge_densest(sub_edges, communities,
                                     len(nodes), target) + offset
        offset += target
    return label


def build_coarse(u_arr, v_arr, label, block, n_super):
    """Two supernodes are joined whenever at least one parent edge runs
    between them.  Nothing is thinned.

    The WITNESS WEIGHT of a supernode pair -- the number of parent edges
    between them -- is computed and returned, since it records how strongly
    each coarse edge is supported, but it is not used to filter.
    """
    super_block = np.zeros(n_super, np.int8)
    for b in range(4):
        super_block[np.unique(label[block == b])] = b

    a, b = label[u_arr], label[v_arr]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    cross = lo != hi
    key = lo[cross].astype(np.int64) * n_super + hi[cross].astype(np.int64)
    pairs, witness = np.unique(key, return_counts=True)

    coarse = nx.Graph()
    coarse.add_nodes_from(range(n_super))
    coarse.add_edges_from((int(k // n_super), int(k % n_super)) for k in pairs)
    return coarse, super_block, pairs, witness


# ===========================================================================
# verification and reporting
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


def verify_coarse(coarse, super_block, label, witness):
    """Whether community detection recovers the partition that was built."""
    parts = sorted(louvain_communities(coarse, seed=1), key=len, reverse=True)
    degree = np.array([coarse.degree(v) for v in range(coarse.number_of_nodes())])
    sizes = np.bincount(label)
    print(f"\n  n = {coarse.number_of_nodes():,}   m = {coarse.number_of_edges():,}"
          f"   density {nx.density(coarse):.4f}"
          f"   mean degree {degree.mean():.1f}   sd {degree.std():.1f}")
    print(f"  supernode sizes:  {sizes.min()} to {sizes.max()} parent vertices"
          f"   (median {int(np.median(sizes))})")
    print(f"  witness weight:   median {int(np.median(witness))}"
          f"   mean {witness.mean():.1f}   max {witness.max()}")
    line = f"  detected blocks:  {[len(p) for p in parts]}"
    if HAVE_SK:
        detected = np.empty(len(super_block), int)
        for i, p in enumerate(parts):
            for v in p:
                detected[v] = i
        line += (f"   ARI vs built = {adjusted_rand_score(detected, super_block):.4f}"
                 f"\n                    (built: {list(np.bincount(super_block))})")
    print(line)


def compare(coarse, reference):
    def summarise(g):
        adj = nx.to_scipy_sparse_array(g, format="csr").astype(float)
        dist = shortest_path(adj, unweighted=True, method="D")
        finite = dist[(dist > 0) & np.isfinite(dist)]
        eff_diam = float("nan")
        for h in range(1, 15):
            if (finite <= h).mean() >= 0.9:
                prev = (finite <= h - 1).mean()
                eff_diam = h - 1 + (0.9 - prev) / max((finite <= h).mean() - prev, 1e-12)
                break
        deg = np.array([g.degree(v) for v in range(g.number_of_nodes())])
        return dict(m=g.number_of_edges(), density=nx.density(g),
                    mean=deg.mean(), sd=deg.std(),
                    lo=int(deg.min()), hi=int(deg.max()),
                    clust=nx.average_clustering(g), trans=nx.transitivity(g),
                    Q=modularity(g, louvain_communities(g, seed=1)),
                    apl=finite.mean(), ed=eff_diam,
                    eff=nx.global_efficiency(g))

    a, b = summarise(coarse), summarise(reference)
    rows = [("edges", "m"), ("density", "density"), ("mean degree", "mean"),
            ("degree sd", "sd"), ("min degree", "lo"), ("max degree", "hi"),
            ("clustering", "clust"), ("transitivity", "trans"),
            ("modularity", "Q"), ("avg path length", "apl"),
            ("effective diameter", "ed"), ("global efficiency", "eff")]
    print(f"\n  {'':<22}{'coarse graph':>16}{'reference':>16}{'ratio':>9}")
    for name, key in rows:
        fa = f"{a[key]:.4f}" if isinstance(a[key], float) else f"{a[key]:,}"
        fb = f"{b[key]:.4f}" if isinstance(b[key], float) else f"{b[key]:,}"
        ratio = a[key] / b[key] if b[key] else float("nan")
        print(f"  {name:<22}{fa:>16}{fb:>16}{ratio:>9.3f}")


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
    p.add_argument("--steps", type=int, default=3,
                   help="Kronecker powers; 3 gives 20^3 = 8,000 parent vertices")
    p.add_argument("--alpha", type=float, nargs="+", default=[3, 4, 30],
                   help="closure edges per vertex at each step (default 3 4 30)")
    p.add_argument("--beta", type=float, default=1.5,
                   help="closure sampling exponent, deg(v)**beta (default 1.5)")
    p.add_argument("--seed", type=int, default=99, help="random seed (default 99)")
    p.add_argument("--parent-only", action="store_true",
                   help="stop after the parent graph")
    p.add_argument("--budapest", default=None,
                   help="reference edge list; enables the comparison table")
    p.add_argument("--out", default=None, help="write <out>_*.csv")
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

    coarse = super_block = label = None
    if not args.parent_only:
        print("=" * 74)
        print(f"STAGE 3  COARSEN    targets {BLOCK_TARGETS}   (no thinning)")
        label = coarsen(parent, block, BLOCK_TARGETS, seed=42)
        coarse, super_block, pairs, witness = build_coarse(
            u_arr, v_arr, label, block, sum(BLOCK_TARGETS))
        verify_coarse(coarse, super_block, label, witness)

        if args.budapest:
            print("=" * 74)
            print("COMPARISON")
            compare(coarse, load_reference(args.budapest))
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
        if coarse is not None:
            np.savetxt(f"{pre}_partition.csv", label, fmt="%d",
                       header="supernode", comments="")
            arr = np.array([[int(k // 1015), int(k % 1015), w]
                            for k, w in zip(pairs, witness)])
            np.savetxt(f"{pre}_coarse_edges.csv", arr, fmt="%d", delimiter=",",
                       header="A,B,witness", comments="")
            written += ["_partition", "_coarse_edges"]
        print("wrote " + ", ".join(f"{pre}{s}.csv" for s in written))


if __name__ == "__main__":
    main()
