#!/usr/bin/env python3
"""
louvain_family_sweep.py -- Coarsen the parent with several Louvain-family
community-detection methods, each at the resolution that brings its community
count CLOSEST TO BUDAPEST'S 1,015 NODES, then compare the resulting quotients.

WHY THIS EXISTS.  The deposited coarsening fixes the node count by fiat: it
runs Louvain at resolution 30 inside each block, which over-partitions about
fivefold, and then merges the densest adjacent community pair until each block
reaches a target read from Budapest (307/306/201/201).  Two things are therefore
not yet separated:

    (i)  how much of the agreement is the MERGE RULE, and
    (ii) how much survives if a different algorithm picks the partition.

This script removes the merge rule entirely.  Every method is asked to land on
1,015 communities BY RESOLUTION ALONE, so the node count is matched without any
Budapest-derived merge target, and the only remaining input from Budapest is the
number 1,015 itself.  Each resulting partition is then quotiented strictly --
two supernodes joined iff at least one parent edge runs between them, no
thinning, exactly as in construct.build_coarse -- and scored with the paper's
own static_tests.py, unmodified.

METHODS.  Four Louvain-family objectives, each run two ways:

    nx_louvain   networkx.louvain_communities            modularity, gamma
    pylouvain    python-louvain best_partition           modularity, gamma
    leiden_mod   leidenalg RBConfigurationVertexPartition modularity, gamma
    leiden_cpm   leidenalg CPMVertexPartition            CPM, gamma

    global    one run on the whole 8,000-vertex parent.  Communities may
              straddle the four blocks, so the coarse graph does NOT inherit
              the block structure by construction.
    perblock  one run inside each block at the same gamma, counts summed.
              Every supernode lies wholly inside one block, as in the
              deposited construction.

Two reference rows are added: the deposited coarsening (Louvain@30 per block +
merge_densest, exactly 1,015 by construction) and the raw parent.

Usage:
    python louvain_family_sweep.py                        # full sweep
    python louvain_family_sweep.py --methods leiden_mod    # one method
    python louvain_family_sweep.py --evals 10              # cheaper sweep
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
BILATERAL = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))
TARGET_N = 1015                       # Budapest's node count
SEED = 42                             # one seed for every method


# ------------------------------------------------------------------ imports

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cc = _load('construct', os.path.join(BILATERAL, 'construct.py'))
st = _load('static_tests', os.path.join(BILATERAL, 'static_tests.py'))


# ------------------------------------------------------- partition backends
# Each backend takes an (n, edges) graph and a resolution, and returns an
# int label array of length n.  Labels need not be contiguous.

def _labels_from_communities(communities, n):
    lab = np.empty(n, np.int32)
    for i, comm in enumerate(communities):
        for v in comm:
            lab[v] = i
    return lab


def part_nx_louvain(n, edges, gamma):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges))
    from networkx.algorithms.community import louvain_communities
    return _labels_from_communities(
        louvain_communities(G, resolution=gamma, seed=SEED), n)


def part_pylouvain(n, edges, gamma):
    import community as community_louvain
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges))
    d = community_louvain.best_partition(G, resolution=gamma,
                                         random_state=SEED)
    lab = np.empty(n, np.int32)
    for v, c in d.items():
        lab[v] = c
    return lab


def _igraph(n, edges):
    import igraph as ig
    return ig.Graph(n=n, edges=[(int(a), int(b)) for a, b in edges],
                    directed=False)


def part_leiden_mod(n, edges, gamma):
    import leidenalg as la
    g = _igraph(n, edges)
    p = la.find_partition(g, la.RBConfigurationVertexPartition,
                          resolution_parameter=gamma, seed=SEED, n_iterations=2)
    return np.asarray(p.membership, np.int32)


def part_leiden_cpm(n, edges, gamma):
    import leidenalg as la
    g = _igraph(n, edges)
    p = la.find_partition(g, la.CPMVertexPartition,
                          resolution_parameter=gamma, seed=SEED, n_iterations=2)
    return np.asarray(p.membership, np.int32)


BACKENDS = {
    'nx_louvain': part_nx_louvain,
    'pylouvain':  part_pylouvain,
    'leiden_mod': part_leiden_mod,
    'leiden_cpm': part_leiden_cpm,
}

#: CPM's resolution is an edge-density threshold, so it needs a much smaller
#: starting bracket than the modularity methods' gamma.
GAMMA0 = {'leiden_cpm': 0.01}


# ------------------------------------------------------------- partitioning

def partition(method, mode, u, v, block, gamma):
    """Return a contiguous label array over all 8,000 parent vertices."""
    fn = BACKENDS[method]
    n = len(block)
    if mode == 'global':
        lab = fn(n, np.stack([u, v], 1), gamma)
    else:
        lab = np.full(n, -1, np.int64)
        offset = 0
        for b in range(int(block.max()) + 1):
            nodes = np.flatnonzero(block == b)
            rename = -np.ones(n, np.int64)
            rename[nodes] = np.arange(len(nodes))
            keep = (block[u] == b) & (block[v] == b)
            sub = np.stack([rename[u[keep]], rename[v[keep]]], 1)
            sl = fn(len(nodes), sub, gamma)
            _, sl = np.unique(sl, return_inverse=True)
            lab[nodes] = sl + offset
            offset += int(sl.max()) + 1
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(np.int32)


def sweep_resolution(method, mode, u, v, block, evals, log):
    """Find the resolution whose community count is closest to 1,015.

    Bracket by doubling gamma until the count exceeds the target, then bisect
    on log(gamma).  The count is monotone in gamma only on average, so the best
    partition SEEN is returned rather than the last one evaluated.
    """
    g0 = GAMMA0.get(method, 1.0)
    seen = {}

    def count(gamma):
        if gamma in seen:
            return seen[gamma][0]
        t = time.time()
        lab = partition(method, mode, u, v, block, gamma)
        k = int(lab.max()) + 1
        seen[gamma] = (k, lab)
        log(f'      gamma {gamma:>12.5g} -> {k:>5,} communities'
            f'   ({time.time() - t:.1f}s)')
        return k

    lo, hi = g0, g0
    k = count(g0)
    used = 1
    if k < TARGET_N:                                  # climb
        while k < TARGET_N and used < evals and hi < 1e7:
            lo, hi = hi, hi * 4
            k = count(hi)
            used += 1
    else:                                             # descend
        while k > TARGET_N and used < evals and lo > 1e-6:
            hi, lo = lo, lo / 4
            k = count(lo)
            used += 1

    while used < evals:                               # bisect on log gamma
        mid = float(np.sqrt(lo * hi))
        if mid in seen or hi / lo < 1.0005:
            break
        k = count(mid)
        used += 1
        if k < TARGET_N:
            lo = mid
        else:
            hi = mid

    gamma = min(seen, key=lambda g: (abs(seen[g][0] - TARGET_N), g))
    return gamma, seen[gamma][1], len(seen)


# ---------------------------------------------------------------- quotient

def quotient(u, v, lab):
    """Strict quotient: supernodes joined iff >=1 parent edge runs between
    them.  Nothing thinned.  Identical in effect to construct.build_coarse,
    minus its per-block bookkeeping (which assumes supernodes respect blocks).
    """
    k = int(lab.max()) + 1
    a, b = lab[u], lab[v]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    cross = lo != hi
    key = lo[cross].astype(np.int64) * k + hi[cross]
    pairs, witness = np.unique(key, return_counts=True)
    G = nx.Graph()
    G.add_nodes_from(range(k))
    G.add_edges_from((int(x // k), int(x % k)) for x in pairs)
    return G, witness, int(cross.sum())


# ------------------------------------------------------------------ scoring

def evaluate(name, lab, u, v, block, deposited, ref, log):
    t = time.time()
    G, witness, n_cross = quotient(u, v, lab)
    if G.number_of_edges() == 0:
        log(f'    {name}: empty quotient, skipped')
        return None
    if not nx.is_connected(G):
        comp = max(nx.connected_components(G), key=len)
        log(f'    {name}: quotient DISCONNECTED '
            f'({nx.number_connected_components(G)} components, '
            f'largest {len(comp)}); scored on the largest component')
        G = nx.convert_node_labels_to_integers(G.subgraph(comp).copy())
    M = st.measure(G)
    sc = st.scores(M, ref)

    try:
        from sklearn.metrics import adjusted_rand_score as ari, \
            normalized_mutual_info_score as nmi
        ari_dep, nmi_dep = ari(lab, deposited), nmi(lab, deposited)
        ari_blk = ari(lab, block)
    except Exception:
        ari_dep = nmi_dep = ari_blk = float('nan')

    sizes = np.bincount(lab)
    # how many supernodes straddle a block boundary
    straddle = sum(1 for c in range(int(lab.max()) + 1)
                   if len(np.unique(block[lab == c])) > 1)

    row = dict(
        name=name,
        n=M['n'], m=M['m'], density=M['density'],
        n_partitions=int(lab.max()) + 1,
        parent_cross_edges=n_cross,
        witness_median=float(np.median(witness)),
        size_min=int(sizes.min()), size_max=int(sizes.max()),
        size_median=float(np.median(sizes)),
        straddling_supernodes=straddle,
        clustering=M['clustering'], transitivity=M['transitivity'],
        modularity=M['modularity'], apl=M['apl'], diameter=M['diameter'],
        lambda1=float(M['sv'][0]), lambda2=float(M['sv'][1]),
        gap=float(M['sv'][0] - M['sv'][1]),
        deg_mean=float(M['degree'].mean()), deg_sd=float(M['degree'].std()),
        ari_vs_deposited=float(ari_dep), nmi_vs_deposited=float(nmi_dep),
        ari_vs_blocks=float(ari_blk),
        **{k: float(x) for k, x in sc.items()},
    )
    log(f'    {name:<26} n={M["n"]:>5,} m={M["m"]:>7,}  '
        f'SCORE {sc["SCORE (mean)"]:.4f}   ({time.time() - t:.0f}s)')
    return row


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=list(BACKENDS),
                   choices=list(BACKENDS))
    p.add_argument('--modes', nargs='+', default=['global', 'perblock'],
                   choices=['global', 'perblock'])
    p.add_argument('--evals', type=int, default=14,
                   help='resolution evaluations per configuration (default 14)')
    p.add_argument('--budapest',
                   default=os.path.join(BILATERAL, 'reference',
                                        'budapest_1015_70654.edgelist'))
    p.add_argument('--out', default=os.path.join(HERE, 'sweep'))
    a = p.parse_args()

    logf = open(a.out + '.log', 'w')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log('=' * 78)
    log('LOUVAIN-FAMILY COARSENING SWEEP')
    log(f'target node count {TARGET_N} (Budapest); seed {SEED}; '
        f'{a.evals} resolution evaluations per configuration')
    log('=' * 78)

    log('\nrebuilding the parent from construct.py (rng 99, alpha 3/4/30, '
        'beta 1.5, 3 steps)')
    rng = np.random.default_rng(99)
    u, v, colour, block = cc.grow(cc.build_seed(), [3, 4, 30], 1.5, 3, rng,
                                  verbose=False)
    n = cc.SEED_N ** 3
    log(f'  parent  n={n:,}  m={len(u):,}  '
        f'blocks {np.bincount(block).tolist()}')

    parent = nx.Graph()
    parent.add_nodes_from(range(n))
    parent.add_edges_from(zip(u.tolist(), v.tolist()))
    deposited = cc.coarsen(parent, block, cc.BLOCK_TARGETS, seed=42)
    log(f'  deposited partition: {deposited.max() + 1} supernodes '
        f'(Louvain@30 per block + merge_densest)')

    log('\nreference: Budapest')
    ref = st.measure(st.build_graph(st.load_edges(a.budapest)))
    log(f'  n={ref["n"]:,}  m={ref["m"]:,}  clustering {ref["clustering"]:.4f}'
        f'  modularity {ref["modularity"]:.4f}'
        f'  lambda1 {ref["sv"][0]:.2f}  gap {ref["sv"][0] - ref["sv"][1]:.2f}')

    rows = []
    log('\n--- reference row: the deposited coarsening ---')
    r = evaluate('deposited (merge_densest)', deposited, u, v, block,
                 deposited, ref, log)
    if r:
        r['method'], r['mode'], r['gamma'] = 'deposited', 'perblock', 30.0
        rows.append(r)

    for method in a.methods:
        for mode in a.modes:
            tag = f'{method}/{mode}'
            log(f'\n--- {tag} ---')
            t0 = time.time()
            try:
                gamma, lab, n_eval = sweep_resolution(
                    method, mode, u, v, block, a.evals, log)
            except Exception as e:
                log(f'    FAILED: {type(e).__name__}: {e}')
                continue
            k = int(lab.max()) + 1
            log(f'    chosen gamma {gamma:.6g} -> {k:,} communities '
                f'(target {TARGET_N}, off by {k - TARGET_N:+,}) '
                f'after {n_eval} evaluations, {time.time() - t0:.0f}s')
            r = evaluate(tag, lab, u, v, block, deposited, ref, log)
            if r:
                r['method'], r['mode'], r['gamma'] = method, mode, float(gamma)
                rows.append(r)

    # ------------------------------------------------------------- output
    if not rows:
        log('\nno rows produced')
        return
    rows.sort(key=lambda r: r['SCORE (mean)'])

    import csv
    keys = ['name', 'method', 'mode', 'gamma', 'n_partitions', 'n', 'm',
            'SCORE (mean)', 'T1 degree', 'T2 eff. diameter', 'T3 hop plot',
            'T4 scree', 'T5 network value', 'T6 triangles',
            'clustering', 'transitivity', 'modularity', 'apl', 'diameter',
            'lambda1', 'lambda2', 'gap', 'deg_mean', 'deg_sd', 'density',
            'size_min', 'size_median', 'size_max', 'straddling_supernodes',
            'parent_cross_edges', 'witness_median',
            'ari_vs_deposited', 'nmi_vs_deposited', 'ari_vs_blocks']
    with open(a.out + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with open(a.out + '.json', 'w') as f:
        json.dump(dict(target_n=TARGET_N, seed=SEED, evals=a.evals,
                       budapest=dict(n=ref['n'], m=ref['m'],
                                     clustering=ref['clustering'],
                                     transitivity=ref['transitivity'],
                                     modularity=ref['modularity'],
                                     lambda1=float(ref['sv'][0]),
                                     gap=float(ref['sv'][0] - ref['sv'][1])),
                       rows=rows), f, indent=2)

    log('\n' + '=' * 78)
    log('RESULT -- six-test mean against Budapest, lower is better')
    log('=' * 78)
    log(f'{"configuration":<27}{"gamma":>10}{"n":>7}{"m":>9}'
        f'{"SCORE":>8}{"clust":>8}{"gap":>8}{"ARIdep":>8}')
    log(f'{"Budapest (reference)":<27}{"":>10}{ref["n"]:>7,}{ref["m"]:>9,}'
        f'{0.0:>8.4f}{ref["clustering"]:>8.4f}'
        f'{ref["sv"][0] - ref["sv"][1]:>8.2f}{"":>8}')
    for r in rows:
        log(f'{r["name"]:<27}{r["gamma"]:>10.4g}{r["n"]:>7,}{r["m"]:>9,}'
            f'{r["SCORE (mean)"]:>8.4f}{r["clustering"]:>8.4f}'
            f'{r["gap"]:>8.2f}{r["ari_vs_deposited"]:>8.3f}')
    log(f'\nwrote {a.out}.csv  {a.out}.json  {a.out}.log')
    logf.close()


if __name__ == '__main__':
    main()
