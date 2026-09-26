#!/usr/bin/env python3
"""
chromatic_null_controls.py -- Is the chromatic constraint's benefit about COLOUR,
or would rejecting any comparable fraction of closures do the same?

THE REFEREE QUESTION.  chromatic_matched_control.py showed that enforcing chi=3
on closure improves the six-test agreement under both coarsenings at matched
parent size.  But the filter also THINS: it rejects a substantial fraction of
otherwise legal closure candidates.  If rejecting a comparable fraction at
random did the same, the effect would be about thinning, not colour.

FOUR ARMS.  Everything identical -- alpha (3,4,30) unless retuned, beta 1.5,
three steps, rng 99, within-block filter ON -- except the extra condition
imposed on a closure candidate {nb1, nb2}:

  chromatic   colour[nb1] != colour[nb2]          the deposited construction
  random      rng.random() < p                    p = the chromatic arm's own
                                                  measured conditional
                                                  acceptance rate.  Rejects a
                                                  comparable fraction, ignoring
                                                  colour entirely.
  permuted    pcolour[nb1] != pcolour[nb2]        the SAME rule against a random
                                                  permutation of the colour
                                                  array (class sizes 2400/2400/
                                                  3200 preserved).  Rejects at a
                                                  similar rate with the same
                                                  "must differ" structure, but
                                                  the labels no longer track the
                                                  tensor fibres.
  off         (nothing)                           the unconstrained arm

Each arm's step-3 alpha is bisected so its PARENT has the same edge count as the
chromatic arm, because the filter starves close_triangles' bounded retry loop
and unmatched arms differ in size by ~7%.  Then each parent is coarsened both
ways -- deposited merge_densest, and per-block Louvain reaching 1,015 by
resolution alone -- and scored with static_tests.py.

READING THE RESULT.  If `random` and `permuted` stay far from the reference while
`chromatic` is close, the colour rule specifically matters.  If they close most
of the gap, the effect is thinning and the paper must say so.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
BILATERAL = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))

_s = importlib.util.spec_from_file_location(
    'cvc', os.path.join(HERE, 'chromatic_vs_coarsening.py'))
_m = importlib.util.module_from_spec(_s); sys.modules['cvc'] = _m
_s.loader.exec_module(_m)
cc, st = _m.cc, _m.st
TARGET_N = _m.TARGET_N

STATS = {}          # arm -> [candidates after block filter, accepted by the rule]


def close_triangles(u_arr, v_arr, n, colour, block, quota, rng, beta,
                    mode, p_accept, pcolour, arm):
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

        base = (nb1 != nb2) & (block[nb1] == block[nb2])
        if mode == 'chromatic':
            rule = colour[nb1] != colour[nb2]
        elif mode == 'permuted':
            rule = pcolour[nb1] != pcolour[nb2]
        elif mode == 'random':
            rule = rng.random(draw) < p_accept
        else:
            rule = np.ones(draw, bool)
        s = STATS.setdefault(arm, [0, 0])
        s[0] += int(base.sum())
        s[1] += int((base & rule).sum())

        ok = base & rule
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


def grow(alphas, mode, p_accept=1.0, arm='', perm_seed=7):
    N = cc.SEED_N
    seed_edges = cc.build_seed()
    su = np.array([a for a, b in seed_edges] + [b for a, b in seed_edges], np.int64)
    sv = np.array([b for a, b in seed_edges] + [a for a, b in seed_edges], np.int64)
    u_arr = np.array([a for a, b in seed_edges], np.int64)
    v_arr = np.array([b for a, b in seed_edges], np.int64)
    rng = np.random.default_rng(99)
    prng = np.random.default_rng(perm_seed)          # separate stream
    n = N
    colour, block = cc.COLOURS.copy(), cc.BLOCKS.copy()
    pcolour = prng.permutation(colour)
    u_arr, v_arr = close_triangles(u_arr, v_arr, n, colour, block,
                                   int(alphas[0] * n), rng, 1.5, mode,
                                   p_accept, pcolour, arm)
    for step in range(2, 4):
        u_arr = (u_arr[:, None] * N + su[None, :]).ravel()
        v_arr = (v_arr[:, None] * N + sv[None, :]).ravel()
        lo, hi = np.minimum(u_arr, v_arr), np.maximum(u_arr, v_arr)
        keys = np.unique(lo * (N ** step) + hi)
        u_arr, v_arr = keys // (N ** step), keys % (N ** step)
        n = N ** step
        digit = (np.arange(n) // (N ** (step - 1))) % N
        colour, block = cc.COLOURS[digit], cc.BLOCKS[digit]
        pcolour = prng.permutation(colour)           # class sizes preserved
        u_arr, v_arr = close_triangles(u_arr, v_arr, n, colour, block,
                                       int(alphas[step - 1] * n), rng, 1.5,
                                       mode, p_accept, pcolour, arm)
    return u_arr.astype(np.int32), v_arr.astype(np.int32), colour, block, pcolour


def main():
    ref = st.measure(st.build_graph(st.load_edges(
        os.path.join(BILATERAL, 'reference', 'budapest_1015_70654.edgelist'))))
    bd = 2 * ref['m'] / (ref['n'] * (ref['n'] - 1))
    print('=' * 78)
    print('NULL CONTROLS FOR THE CHROMATIC CONSTRAINT')
    print('=' * 78)
    print(f'\nreference  n={ref["n"]:,}  m={ref["m"]:,}  density={bd:.5f}'
          f'  clustering={ref["clustering"]:.4f}  gap={ref["sv"][0]-ref["sv"][1]:.2f}'
          f'  deg sd={ref["degree"].std():.2f}')

    # --- the chromatic arm, and its measured acceptance rate -----------------
    STATS.clear()
    u, v, colour, block, _ = grow([3, 4, 30], 'chromatic', arm='chromatic')
    target = len(u)
    cand, acc = STATS['chromatic']
    p = acc / cand
    print(f'\nchi=3 arm:  parent m = {target:,}   (the size all arms match)')
    print(f'  closure candidates surviving the block filter: {cand:,}')
    print(f'  of those, accepted by the chromatic rule:      {acc:,}')
    print(f'  => CONDITIONAL ACCEPTANCE RATE p = {p:.4f} '
          f'(rejects {100*(1-p):.1f}%)')
    print(f'  the random arm uses exactly this p, ignoring colour.')

    arms = [('chromatic', 'chromatic', 30.0, 1.0),
            ('random', 'random', None, p),
            ('permuted', 'permuted', None, 1.0),
            ('off', 'off', None, 1.0)]

    rows, parents = [], {}
    for arm, mode, fixed_a3, pa in arms:
        if fixed_a3 is not None:
            a3 = fixed_a3
            uu, vv, cl, bl, pc = u, v, colour, block, None
        else:
            print(f'\ntuning {arm}: bisecting alpha3 to match {target:,} parent edges')
            lo, hi, best = 5.0, 60.0, None
            for _ in range(13):
                mid = 0.5 * (lo + hi)
                STATS.clear()
                uu, vv, cl, bl, pc = grow([3, 4, mid], mode, pa, arm)
                mm = len(uu)
                print(f'    alpha3 {mid:7.3f} -> parent m {mm:>8,}  ({mm-target:+,})')
                if best is None or abs(mm - target) < abs(best[1] - target):
                    best = (mid, mm)
                if mm > target:
                    hi = mid
                else:
                    lo = mid
                if abs(mm - target) <= 200:
                    break
            a3 = best[0]
            STATS.clear()
            uu, vv, cl, bl, pc = grow([3, 4, a3], mode, pa, arm)
        cand_a, acc_a = STATS.get(arm, [1, 1])
        mono = int((cl[uu] == cl[vv]).sum())
        parents[arm] = (len(uu), mono, acc_a / cand_a if cand_a else float('nan'))
        print(f'\n{arm.upper()}:  alpha3={a3:.4f}  parent m={len(uu):,} '
              f'({100*(len(uu)-target)/target:+.2f}%)  monochromatic={mono:,} '
              f'({100*mono/len(uu):.1f}%)  accept rate={parents[arm][2]:.4f}')

        parent = nx.Graph(); parent.add_nodes_from(range(8000))
        parent.add_edges_from(zip(uu.tolist(), vv.tolist()))
        for meth in ('deposited merge_densest', 'resolution-only per block'):
            if meth.startswith('deposited'):
                lab, gamma = cc.coarsen(parent, bl, cc.BLOCK_TARGETS, seed=42), 30.0
            else:
                gamma, lab = _m.sweep(uu, vv, bl, 16, lambda s: None)
            G, n_cross = _m.quotient(uu, vv, lab)
            M = st.measure(G); sc = st.scores(M, ref)
            den = 2 * M['m'] / (M['n'] * (M['n'] - 1))
            rows.append(dict(arm=arm, alpha3=float(a3), coarsening=meth,
                             gamma=float(gamma), parent_m=len(uu), mono=mono,
                             accept_rate=parents[arm][2], n=M['n'], m=M['m'],
                             density=den, density_err=100*(den-bd)/bd,
                             survive=n_cross/len(uu),
                             clustering=M['clustering'],
                             transitivity=M['transitivity'],
                             gap=float(M['sv'][0]-M['sv'][1]),
                             deg_sd=float(M['degree'].std()),
                             score=float(sc['SCORE (mean)']),
                             **{k: float(x) for k, x in sc.items()}))
            print(f'    {meth:<28} gamma={gamma:<8.4g} n={M["n"]:>5,} '
                  f'm={M["m"]:>7,} density={den:.5f} ({100*(den-bd)/bd:+.2f}%) '
                  f'SCORE {sc["SCORE (mean)"]:.4f}')

    json.dump(dict(reference=dict(n=ref['n'], m=ref['m'], density=bd,
                                  clustering=ref['clustering'],
                                  deg_sd=float(ref['degree'].std()),
                                  gap=float(ref['sv'][0]-ref['sv'][1])),
                   chromatic_accept_rate=p, target_parent_m=target, rows=rows),
              open(os.path.join(HERE, 'chromatic_null_controls.json'), 'w'), indent=2)

    print('\n' + '=' * 78)
    print('RESULT -- all arms at matched parent size')
    print('=' * 78)
    for meth in ('deposited merge_densest', 'resolution-only per block'):
        print(f'\n{meth}')
        print(f'  {"arm":<12}{"accept":>8}{"parent m":>10}{"n":>7}{"m":>8}'
              f'{"density":>9}{"dd%":>8}{"clust":>7}{"gap":>8}{"SCORE":>8}')
        print(f'  {"reference":<12}{"":>8}{"":>10}{ref["n"]:>7,}{ref["m"]:>8,}'
              f'{bd:>9.5f}{"":>8}{ref["clustering"]:>7.4f}'
              f'{ref["sv"][0]-ref["sv"][1]:>8.2f}{"":>8}')
        for r in sorted((x for x in rows if x['coarsening'] == meth),
                        key=lambda x: x['score']):
            print(f'  {r["arm"]:<12}{r["accept_rate"]:>8.4f}{r["parent_m"]:>10,}'
                  f'{r["n"]:>7,}{r["m"]:>8,}{r["density"]:>9.5f}'
                  f'{r["density_err"]:>+8.2f}{r["clustering"]:>7.4f}'
                  f'{r["gap"]:>8.2f}{r["score"]:>8.4f}')

    print('\n' + '=' * 78)
    print('THE QUESTION: does thinning alone reproduce the chromatic effect?')
    print('=' * 78)
    for meth in ('deposited merge_densest', 'resolution-only per block'):
        g = {r['arm']: r for r in rows if r['coarsening'] == meth}
        ch, off = g['chromatic'], g['off']
        span = off['score'] - ch['score']
        print(f'\n  {meth}:   chi=3 {ch["score"]:.4f}   off {off["score"]:.4f}'
              f'   (span {span:+.4f})')
        for a in ('random', 'permuted'):
            r = g[a]
            frac = (off['score'] - r['score']) / span if span else float('nan')
            print(f'    {a:<10} {r["score"]:.4f}   closes {100*frac:>6.1f}% of the '
                  f'gap   density {r["density_err"]:+.2f}% vs chi=3 '
                  f'{ch["density_err"]:+.2f}%')


if __name__ == '__main__':
    main()
