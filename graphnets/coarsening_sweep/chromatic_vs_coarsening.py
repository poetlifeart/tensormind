#!/usr/bin/env python3
"""
chromatic_vs_coarsening.py -- Does the chromatic constraint still win once the
merge rule is replaced by resolution-only per-block Louvain?

The chromatic filter in close_triangles was previously shown to raise quotient
density: an edge whose endpoints differ in colour is more likely to cross a
supernode boundary and therefore survive coarsening, rather than being absorbed
into a supernode's interior and vanishing.  That was measured against the
DEPOSITED coarsening (Louvain@30 per block + merge_densest).

Resolution-only per-block Louvain produces a different partition (ARI ~0.18 to
the deposited one) and a denser quotient, so the chromatic advantage has to be
re-measured there.  This script grows the parent with the chromatic filter ON
and OFF -- everything else identical, same quota alpha*n at every step, same
within-block filter, same seed -- and coarsens each parent BOTH ways.

The seed is cross-colour by construction and the tensor product cannot create a
monochromatic edge (colour rides on the leading digit, exactly as block does),
so closure is the only source of monochromatic edges.  Relaxing the closure
filter is therefore the clean ablation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
BILATERAL = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))
TARGET_N = 1015
SEED = 42


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cc = _load('construct', os.path.join(BILATERAL, 'construct.py'))
st = _load('static_tests', os.path.join(BILATERAL, 'static_tests.py'))


# ------------------------------------------------------------------ growth
# close_triangles and grow, copied from construct.py with ONE change: the
# chromatic condition is switched by a flag.  Copied rather than patched so
# construct.py stays untouched and this file is a standalone record.

def close_triangles(u_arr, v_arr, n, colour, block, quota, rng, beta,
                    chromatic=True):
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

        ok = (nb1 != nb2) & (block[nb1] == block[nb2])
        if chromatic:                                  # <-- the ablation
            ok &= (colour[nb1] != colour[nb2])
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


def grow(seed_edges, alphas, beta, n_steps, rng, chromatic=True):
    N = cc.SEED_N
    su = np.array([a for a, b in seed_edges] + [b for a, b in seed_edges], np.int64)
    sv = np.array([b for a, b in seed_edges] + [a for a, b in seed_edges], np.int64)
    u_arr = np.array([a for a, b in seed_edges], np.int64)
    v_arr = np.array([b for a, b in seed_edges], np.int64)
    n = N
    colour, block = cc.COLOURS.copy(), cc.BLOCKS.copy()
    u_arr, v_arr = close_triangles(u_arr, v_arr, n, colour, block,
                                   int(alphas[0] * n), rng, beta, chromatic)
    for step in range(2, n_steps + 1):
        u_arr = (u_arr[:, None] * N + su[None, :]).ravel()
        v_arr = (v_arr[:, None] * N + sv[None, :]).ravel()
        lo, hi = np.minimum(u_arr, v_arr), np.maximum(u_arr, v_arr)
        keys = np.unique(lo * (N ** step) + hi)
        u_arr, v_arr = keys // (N ** step), keys % (N ** step)
        n = N ** step
        digit = (np.arange(n) // (N ** (step - 1))) % N
        colour, block = cc.COLOURS[digit], cc.BLOCKS[digit]
        u_arr, v_arr = close_triangles(u_arr, v_arr, n, colour, block,
                                       int(alphas[step - 1] * n), rng, beta,
                                       chromatic)
    return u_arr.astype(np.int32), v_arr.astype(np.int32), colour, block


# --------------------------------------------------------------- coarsening

def quotient(u, v, lab):
    k = int(lab.max()) + 1
    a, b = lab[u], lab[v]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    cross = lo != hi
    key = lo[cross].astype(np.int64) * k + hi[cross]
    pairs = np.unique(key)
    G = nx.Graph()
    G.add_nodes_from(range(k))
    G.add_edges_from((int(x // k), int(x % k)) for x in pairs)
    return G, int(cross.sum())


def perblock_louvain(u, v, block, gamma):
    from networkx.algorithms.community import louvain_communities
    n = len(block)
    lab = np.full(n, -1, np.int64)
    offset = 0
    for b in range(int(block.max()) + 1):
        nodes = np.flatnonzero(block == b)
        rename = -np.ones(n, np.int64)
        rename[nodes] = np.arange(len(nodes))
        keep = (block[u] == b) & (block[v] == b)
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(zip(rename[u[keep]].tolist(), rename[v[keep]].tolist()))
        comms = louvain_communities(G, resolution=gamma, seed=SEED)
        sl = np.empty(len(nodes), np.int64)
        for i, c in enumerate(comms):
            for x in c:
                sl[x] = i
        lab[nodes] = sl + offset
        offset += len(comms)
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(np.int32)


def sweep(u, v, block, evals, log):
    """Resolution closest to TARGET_N communities, per-block networkx Louvain."""
    seen = {}

    def count(g):
        if g in seen:
            return seen[g][0]
        lab = perblock_louvain(u, v, block, g)
        k = int(lab.max()) + 1
        seen[g] = (k, lab)
        log(f'        gamma {g:>10.5g} -> {k:>5,}')
        return k

    lo = hi = 1.0
    k = count(1.0)
    used = 1
    while k < TARGET_N and used < evals and hi < 1e6:
        lo, hi = hi, hi * 4
        k = count(hi)
        used += 1
    while used < evals:
        mid = float(np.sqrt(lo * hi))
        if mid in seen or hi / lo < 1.0005:
            break
        k = count(mid)
        used += 1
        if k < TARGET_N:
            lo = mid
        else:
            hi = mid
    g = min(seen, key=lambda x: (abs(seen[x][0] - TARGET_N), x))
    return g, seen[g][1]


# ------------------------------------------------------------------- report

def main():
    log = print
    log('=' * 78)
    log('CHROMATIC CONSTRAINT vs COARSENING METHOD')
    log('alpha (3,4,30)  beta 1.5  3 steps  rng 99  block filter ON in both arms')
    log('=' * 78)

    ref = st.measure(st.build_graph(st.load_edges(
        os.path.join(BILATERAL, 'reference', 'budapest_1015_70654.edgelist'))))
    bd = 2 * ref['m'] / (ref['n'] * (ref['n'] - 1))
    log(f"\nBUDAPEST  n={ref['n']:,}  m={ref['m']:,}  density={bd:.5f}"
        f"  clustering={ref['clustering']:.4f}")

    rows = []
    for chromatic in (True, False):
        arm = 'chi=3 ENFORCED' if chromatic else 'chromatic filter OFF'
        log(f'\n{"=" * 78}\n{arm}\n{"=" * 78}')
        u, v, colour, block = grow(cc.build_seed(), [3, 4, 30], 1.5, 3,
                                   np.random.default_rng(99), chromatic)
        mono = int((colour[u] == colour[v]).sum())
        log(f'  parent  n=8,000  m={len(u):,}  monochromatic edges={mono:,}'
            f'  ({100 * mono / len(u):.1f}%)')

        parent = nx.Graph()
        parent.add_nodes_from(range(8000))
        parent.add_edges_from(zip(u.tolist(), v.tolist()))

        for method in ('deposited merge_densest', 'resolution-only per block'):
            t = time.time()
            if method.startswith('deposited'):
                lab = cc.coarsen(parent, block, cc.BLOCK_TARGETS, seed=42)
                gamma = 30.0
            else:
                log(f'      sweeping resolution ({arm})')
                gamma, lab = sweep(u, v, block, 16, log)
            G, n_cross = quotient(u, v, lab)
            M = st.measure(G)
            sc = st.scores(M, ref)
            den = 2 * M['m'] / (M['n'] * (M['n'] - 1))
            row = dict(arm=arm, chromatic=chromatic, coarsening=method,
                       gamma=float(gamma), parent_m=len(u), mono=mono,
                       n=M['n'], m=M['m'], density=den,
                       survive=n_cross / len(u),
                       clustering=M['clustering'],
                       transitivity=M['transitivity'],
                       modularity=M['modularity'],
                       gap=float(M['sv'][0] - M['sv'][1]),
                       deg_mean=float(M['degree'].mean()),
                       deg_sd=float(M['degree'].std()),
                       score=float(sc['SCORE (mean)']),
                       **{k: float(x) for k, x in sc.items()})
            rows.append(row)
            log(f'    {method:<28} gamma={gamma:<8.4g} n={M["n"]:>5,}'
                f' m={M["m"]:>7,} density={den:.5f}'
                f'  SCORE {sc["SCORE (mean)"]:.4f}  ({time.time() - t:.0f}s)')

    with open(os.path.join(HERE, 'chromatic_vs_coarsening.json'), 'w') as f:
        json.dump(dict(budapest=dict(n=ref['n'], m=ref['m'], density=bd,
                                     clustering=ref['clustering'],
                                     deg_sd=float(ref['degree'].std())),
                       rows=rows), f, indent=2)

    log('\n' + '=' * 78)
    log('SUMMARY -- quotient edges and density, and what the chromatic filter buys')
    log('=' * 78)
    log(f'{"coarsening":<26}{"chromatic":>11}{"n":>7}{"m":>8}{"dm%":>8}'
        f'{"density":>9}{"dd%":>8}{"surv%":>7}{"clust":>7}{"SCORE":>8}')
    log(f'{"Budapest":<26}{"":>11}{ref["n"]:>7,}{ref["m"]:>8,}{"":>8}'
        f'{bd:>9.5f}{"":>8}{"":>7}{ref["clustering"]:>7.4f}{"":>8}')
    for meth in ('deposited merge_densest', 'resolution-only per block'):
        for r in rows:
            if r['coarsening'] != meth:
                continue
            log(f'{meth:<26}{"ON" if r["chromatic"] else "OFF":>11}'
                f'{r["n"]:>7,}{r["m"]:>8,}{100 * (r["m"] - ref["m"]) / ref["m"]:>+8.2f}'
                f'{r["density"]:>9.5f}{100 * (r["density"] - bd) / bd:>+8.2f}'
                f'{100 * r["survive"]:>7.1f}{r["clustering"]:>7.4f}'
                f'{r["score"]:>8.4f}')
    for meth in ('deposited merge_densest', 'resolution-only per block'):
        on = next(r for r in rows if r['coarsening'] == meth and r['chromatic'])
        off = next(r for r in rows if r['coarsening'] == meth and not r['chromatic'])
        log(f'\n  {meth}: chromatic filter changes quotient edges by '
            f'{100 * (on["m"] - off["m"]) / off["m"]:+.2f}%, '
            f'density {100 * (on["density"] - off["density"]) / off["density"]:+.2f}%, '
            f'score {on["score"] - off["score"]:+.4f}')


if __name__ == '__main__':
    main()
