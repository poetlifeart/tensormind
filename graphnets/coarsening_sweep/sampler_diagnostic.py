#!/usr/bin/env python3
"""
sampler_diagnostic.py -- What does the closure sampler's round structure actually
do, and does the filter change it?

GPT's observation: a random null that drops each proposal with probability 0.51,
independently of the pair, should at matched parent size be statistically
identical to no filter at all -- yet it scores very differently (0.1549 vs
0.2904).  So the difference cannot be "thinning" in the naive sense and must come
from the sampler's round structure.  GPT's proposed mechanism: rejection changes
how many candidates survive per round, which changes how strongly the sampler
favours high-degree centres, i.e. it acts as a change in beta.

WHAT THE CODE ACTUALLY DOES, which is not what GPT assumed.  The acceptance step is

    keys = np.unique(min(nb1,nb2)*n + max(nb1,nb2))      # SORTED
    keys = np.array([k for k in keys if k not in seen])[:remaining]

np.unique returns keys in ASCENDING order and the comprehension preserves it, so
[:remaining] keeps the LOWEST-KEYED survivors -- a deterministic truncation toward
low vertex indices, NOT a uniform random subset.  So there are two candidate
mechanisms, not one:

    (i)  DEGREE BIAS   fewer survivors per round -> different effective beta
    (ii) INDEX BIAS    truncation keeps low-index pairs; how hard it bites
                       depends on how many survive the filter

Both are only live if the truncation actually bites, i.e. if len(keys) > remaining
in some round.  This script measures that first, then quantifies both biases.

Reports, per arm and per growth step:
    draw size, candidates passing the block filter, candidates passing the rule,
    new unique keys, remaining quota, how many were taken, WHETHER TRUNCATED,
    and for the edges actually accepted: mean endpoint degree (the degree-bias
    probe) and mean endpoint index (the index-bias probe).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
BILATERAL = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))
_s = importlib.util.spec_from_file_location('construct',
                                            os.path.join(BILATERAL, 'construct.py'))
cc = importlib.util.module_from_spec(_s); sys.modules['construct'] = cc
_s.loader.exec_module(cc)

ROUNDS = []          # one dict per round


def close_triangles(u_arr, v_arr, n, colour, block, quota, rng, beta,
                    mode, p_accept, arm, step):
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
    for rnd in range(24):
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
        elif mode == 'random':
            rule = rng.random(draw) < p_accept
        else:
            rule = np.ones(draw, bool)
        n_base, n_rule = int(base.sum()), int((base & rule).sum())

        ok = base & rule
        a, b = nb1[ok], nb2[ok]
        if not len(a):
            break
        allkeys = np.unique(np.minimum(a, b).astype(np.int64) * n
                            + np.maximum(a, b).astype(np.int64))
        fresh = np.array([k for k in allkeys if k not in seen], np.int64)
        n_fresh = len(fresh)
        keys = fresh[:remaining]
        truncated = n_fresh > remaining
        if not len(keys):
            break
        eu, ev = keys // n, keys % n
        ROUNDS.append(dict(arm=arm, step=step, rnd=rnd, draw=draw,
                           n_base=n_base, n_rule=n_rule, n_fresh=n_fresh,
                           remaining=remaining, n_taken=len(keys),
                           truncated=bool(truncated),
                           # degree-bias probe: degree of accepted endpoints
                           deg_mean=float(degree[eu].mean() + degree[ev].mean()) / 2,
                           # what was available, for contrast
                           deg_mean_available=float(
                               (degree[fresh // n].mean()
                                + degree[fresh % n].mean()) / 2),
                           # index-bias probe
                           idx_mean=float((eu.mean() + ev.mean()) / 2),
                           idx_mean_available=float(
                               ((fresh // n).mean() + (fresh % n).mean()) / 2),
                           n_vertices=n))
        seen.update(keys.tolist())
        accepted.append(keys)
        remaining -= len(keys)

    if not accepted:
        return u_arr, v_arr
    new = np.concatenate(accepted)
    return (np.concatenate([u_arr, new // n]),
            np.concatenate([v_arr, new % n]))


def grow(alphas, mode, p_accept=1.0, beta=1.5, arm=''):
    N = cc.SEED_N
    se = cc.build_seed()
    su = np.array([a for a, b in se] + [b for a, b in se], np.int64)
    sv = np.array([b for a, b in se] + [a for a, b in se], np.int64)
    u = np.array([a for a, b in se], np.int64)
    v = np.array([b for a, b in se], np.int64)
    rng = np.random.default_rng(99)
    n = N
    colour, block = cc.COLOURS.copy(), cc.BLOCKS.copy()
    u, v = close_triangles(u, v, n, colour, block, int(alphas[0] * n), rng,
                           beta, mode, p_accept, arm, 1)
    for step in range(2, 4):
        u = (u[:, None] * N + su[None, :]).ravel()
        v = (v[:, None] * N + sv[None, :]).ravel()
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = np.unique(lo * (N ** step) + hi)
        u, v = keys // (N ** step), keys % (N ** step)
        n = N ** step
        digit = (np.arange(n) // (N ** (step - 1))) % N
        colour, block = cc.COLOURS[digit], cc.BLOCKS[digit]
        u, v = close_triangles(u, v, n, colour, block,
                               int(alphas[step - 1] * n), rng, beta, mode,
                               p_accept, arm, step)
    return u.astype(np.int32), v.astype(np.int32), colour, block


def main():
    print('=' * 78)
    print('SAMPLER DIAGNOSTIC -- does the [:remaining] truncation ever bite?')
    print('=' * 78)
    for arm, mode, p, a3 in (('chromatic', 'chromatic', 1.0, 30.0),
                             ('random', 'random', 0.4904, 25.6787),
                             ('off', 'off', 1.0, 25.6787)):
        u, v, cl, bl = grow([3, 4, a3], mode, p, 1.5, arm)
        print(f'\n{arm.upper()}  (alpha3={a3})  parent m={len(u):,}')
        rs = [r for r in ROUNDS if r['arm'] == arm]
        for step in (1, 2, 3):
            sr = [r for r in rs if r['step'] == step]
            if not sr:
                continue
            nt = sum(r['truncated'] for r in sr)
            print(f'  step {step}: {len(sr)} rounds, {nt} TRUNCATED, '
                  f'quota filled {sum(r["n_taken"] for r in sr):,}')
            for r in sr[:4]:
                flag = 'TRUNCATED' if r['truncated'] else ''
                print(f'      rnd {r["rnd"]}: draw {r["draw"]:>7,} '
                      f'block-pass {r["n_base"]:>7,} rule-pass {r["n_rule"]:>7,} '
                      f'fresh {r["n_fresh"]:>7,} remaining {r["remaining"]:>7,} '
                      f'took {r["n_taken"]:>7,}  {flag}')
                if r['truncated']:
                    print(f'          degree of taken {r["deg_mean"]:.1f} vs '
                          f'available {r["deg_mean_available"]:.1f}   |   '
                          f'index of taken {r["idx_mean"]:,.0f} vs available '
                          f'{r["idx_mean_available"]:,.0f}  (n={r["n_vertices"]:,})')
            if len(sr) > 4:
                print(f'      ... {len(sr)-4} more rounds')

    json.dump(ROUNDS, open(os.path.join(HERE, 'sampler_diagnostic.json'), 'w'),
              indent=1)
    print('\n' + '=' * 78)
    print('SUMMARY: truncation events per arm (only these can create bias)')
    print('=' * 78)
    for arm in ('chromatic', 'random', 'off'):
        rs = [r for r in ROUNDS if r['arm'] == arm]
        tr = [r for r in rs if r['truncated']]
        tot = sum(r['n_taken'] for r in rs)
        intr = sum(r['n_taken'] for r in tr)
        print(f'  {arm:<11} {len(tr):>2} of {len(rs):>2} rounds truncated; '
              f'{intr:,} of {tot:,} closure edges ({100*intr/max(tot,1):.1f}%) '
              f'came from a truncated round')
        if tr:
            dg = np.mean([r['deg_mean'] - r['deg_mean_available'] for r in tr])
            ix = np.mean([r['idx_mean'] - r['idx_mean_available'] for r in tr])
            print(f'              mean degree shift {dg:+.2f}   '
                  f'mean index shift {ix:+,.0f}')


if __name__ == '__main__':
    main()
