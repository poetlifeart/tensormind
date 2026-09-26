#!/usr/bin/env python3
"""
r2_truncation_fix.py -- Repair the sorted-truncation bias in the experiment
harness, and measure what it was doing.

BACKGROUND.  construct.py's close_triangles accepts candidates as

    keys = np.unique(...)                        # ASCENDING by min(u,v)*n+max(u,v)
    keys = np.array([k for k in keys if k not in seen])[:remaining]

np.unique sorts, so [:remaining] keeps the LOWEST-KEYED survivors -- a
deterministic cut toward low vertex indices.  Because colour AND block are both
read off the LEADING base-20 digit of the index, that cut is correlated with both
labels: it preferentially discards whole high-index blocks.

THIS WAS ALREADY KNOWN AND ALREADY FIXED in graphnets/ablation/
chromatic_ablation.py, which does rng.shuffle(keys) before truncating and
documents exactly this reasoning.  The paper's Table 5 and the 20-seed stage
ablation use that shuffled path and are NOT affected.  The coarsening_sweep
scripts copied close_triangles from construct.py -- the unshuffled version -- and
so reintroduced it.  This script is the repair and the measurement.

HOW BADLY IT BIT, from sampler_diagnostic.py (share of closure edges coming from a
round where the truncation actually applied):

    chromatic      3 of 241,605   (0.0%)
    random        14 of 207,043   (0.0%)
    off      207,029 of 207,043 (100.0%)

Only the unconstrained arm is affected, because only it has candidates to spare:
it fills the whole step-3 quota in ONE round (356,444 fresh, 205,429 taken) and
the cut moves accepted endpoints from mean index 2,963 to 1,695 out of 8,000, and
mean degree from 99.5 to 110.1.

THE FIX, and why it is done this way.  A uniform random subset replaces the
sorted cut, drawn from a SEPARATE generator seeded independently of growth.  Two
reasons: construct.py must not be touched at all, since adding a shuffle there
would advance the random stream and break exact reproduction of the deposited
graph; and using a separate stream here means an arm that never truncates is
BIT-IDENTICAL with and without the fix.  That identity is the control -- it proves
the fix changes only the truncation and nothing else.

ARMS.  chromatic is run both ways as that control; off is run both ways as the
measurement; random_fixed is the clean null to compare off_fixed against.

READING IT.
  off_fixed ~= random_fixed  => the "thinning" effect was entirely the truncation
                               artifact, and every published or drafted
                               comparison against a no-filter arm must be
                               restated against the random-rejection null.
  off_fixed still far        => something about removing the filter survives
                               beyond the truncation.
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

TRUNC_SEED = 20260926
STATS = {}


def close_triangles(u, v, n, colour, block, quota, rng, beta, mode, p_accept,
                    arm, fix, trng, pcolour, q_extra):
    if quota <= 0:
        return u, v
    adj = csr_matrix((np.ones(len(u), np.int8), (u, v)), shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1
    indptr, indices = adj.indptr, adj.indices
    degree = np.diff(indptr)
    seen = set((np.minimum(u, v).astype(np.int64) * n
                + np.maximum(u, v).astype(np.int64)).tolist())
    eligible = np.where(degree >= 2)[0]
    if not len(eligible):
        return u, v
    w = degree[eligible].astype(float) ** beta
    w /= w.sum()

    acc_list, remaining = [], quota
    for _ in range(24):
        if remaining <= 0:
            break
        draw = int(remaining * 3) + 10
        centre = rng.choice(eligible, size=draw, p=w)
        dc = degree[centre]
        nb1 = indices[indptr[centre] + (rng.random(draw) * dc).astype(int)]
        nb2 = indices[indptr[centre] + (rng.random(draw) * dc).astype(int)]

        base = (nb1 != nb2) & (block[nb1] == block[nb2])
        if mode == 'chromatic':
            rule = colour[nb1] != colour[nb2]
        elif mode == 'permuted':
            rule = pcolour[nb1] != pcolour[nb2]
        elif mode == 'permuted_rate':
            rule = (pcolour[nb1] != pcolour[nb2]) & (rng.random(draw) < q_extra)
        elif mode == 'random':
            rule = rng.random(draw) < p_accept
        else:
            rule = np.ones(draw, bool)

        s = STATS.setdefault(arm, [0, 0, 0, 0])   # cand, accepted, trunc_rounds, trunc_edges
        s[0] += int(base.sum())
        s[1] += int((base & rule).sum())

        ok = base & rule
        a, b = nb1[ok], nb2[ok]
        if not len(a):
            break
        allk = np.unique(np.minimum(a, b).astype(np.int64) * n
                         + np.maximum(a, b).astype(np.int64))
        fresh = np.array([k for k in allk if k not in seen], np.int64)
        if not len(fresh):
            break
        if len(fresh) > remaining:
            s[2] += 1
            s[3] += remaining
            # separate generator: the growth stream `rng` is never advanced here
            keys = (trng.choice(fresh, size=remaining, replace=False) if fix
                    else fresh[:remaining])
        else:
            keys = fresh
        seen.update(keys.tolist())
        acc_list.append(keys)
        remaining -= len(keys)

    if not acc_list:
        return u, v
    new = np.concatenate(acc_list)
    return np.concatenate([u, new // n]), np.concatenate([v, new % n])


def grow(alphas, mode, p_accept, arm, fix, q_extra=1.0, perm_seed=7):
    N = cc.SEED_N
    se = cc.build_seed()
    su = np.array([a for a, b in se] + [b for a, b in se], np.int64)
    sv = np.array([b for a, b in se] + [a for a, b in se], np.int64)
    u = np.array([a for a, b in se], np.int64)
    v = np.array([b for a, b in se], np.int64)
    rng = np.random.default_rng(99)
    trng = np.random.default_rng(TRUNC_SEED)
    prng = np.random.default_rng(perm_seed)
    n = N
    colour, block = cc.COLOURS.copy(), cc.BLOCKS.copy()
    pc = prng.permutation(colour)
    u, v = close_triangles(u, v, n, colour, block, int(alphas[0] * n), rng, 1.5,
                           mode, p_accept, arm, fix, trng, pc, q_extra)
    for step in range(2, 4):
        u = (u[:, None] * N + su[None, :]).ravel()
        v = (v[:, None] * N + sv[None, :]).ravel()
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        k = np.unique(lo * (N ** step) + hi)
        u, v = k // (N ** step), k % (N ** step)
        n = N ** step
        digit = (np.arange(n) // (N ** (step - 1))) % N
        colour, block = cc.COLOURS[digit], cc.BLOCKS[digit]
        pc = prng.permutation(colour)
        u, v = close_triangles(u, v, n, colour, block, int(alphas[step - 1] * n),
                              rng, 1.5, mode, p_accept, arm, fix, trng, pc,
                              q_extra)
    return u.astype(np.int32), v.astype(np.int32), colour, block


def tune(mode, p, target, arm, fix, q, log):
    lo, hi, best = 5.0, 60.0, None
    for _ in range(13):
        mid = 0.5 * (lo + hi)
        STATS.clear()
        uu, vv, _, _ = grow([3, 4, mid], mode, p, arm, fix, q)
        mm = len(uu)
        log(f'      alpha3 {mid:7.3f} -> {mm:>8,} ({mm - target:+,})')
        if best is None or abs(mm - target) < abs(best[1] - target):
            best = (mid, mm)
        if mm > target:
            hi = mid
        else:
            lo = mid
        if abs(mm - target) <= 200:
            break
    return best[0]


def main():
    log = print
    ref = st.measure(st.build_graph(st.load_edges(
        os.path.join(BILATERAL, 'reference', 'budapest_1015_70654.edgelist'))))
    bd = 2 * ref['m'] / (ref['n'] * (ref['n'] - 1))
    log('=' * 78)
    log('R2 -- SORTED-TRUNCATION REPAIR, AND WHAT IT WAS DOING')
    log('=' * 78)

    STATS.clear()
    u0, v0, _, _ = grow([3, 4, 30], 'chromatic', 1.0, 'probe', False)
    target = len(u0)
    P = STATS['probe'][1] / STATS['probe'][0]
    log(f'\nchromatic parent m = {target:,}   accept rate p = {P:.4f}')

    STATS.clear()
    grow([3, 4, 30], 'permuted', 1.0, 'pprobe', False)
    Ppm = STATS['pprobe'][1] / STATS['pprobe'][0]
    Q = P / Ppm
    log(f'permuted unmatched rate {Ppm:.4f} -> extra rejection q = {Q:.4f}')

    arms = [('chromatic',          'chromatic',     1.0, False, 1.0, 30.0),
            ('chromatic_fixed',    'chromatic',     1.0, True,  1.0, 30.0),
            ('random_fixed',       'random',        P,   True,  1.0, None),
            ('permuted_fixed',     'permuted',      1.0, True,  1.0, None),
            ('permuted_rate_fixed','permuted_rate', 1.0, True,  Q,   None),
            ('off',                'off',           1.0, False, 1.0, None),
            ('off_fixed',          'off',           1.0, True,  1.0, None)]

    rows = []
    for arm, mode, p, fix, q, a3fix in arms:
        log(f'\n--- {arm}{"   [FIXED]" if fix else ""} ---')
        a3 = a3fix if a3fix is not None else tune(mode, p, target, arm, fix, q, log)
        STATS.clear()
        uu, vv, cl, bl = grow([3, 4, a3], mode, p, arm, fix, q)
        cand, accn, tr_r, tr_e = STATS[arm]
        mono = int((cl[uu] == cl[vv]).sum())
        log(f'    alpha3 {a3:.4f}  parent {len(uu):,} '
            f'({100 * (len(uu) - target) / target:+.2f}%)  accept {accn/cand:.4f}'
            f'  truncated rounds {tr_r}, edges from them {tr_e:,}  mono {mono:,}')
        par = nx.Graph(); par.add_nodes_from(range(8000))
        par.add_edges_from(zip(uu.tolist(), vv.tolist()))
        for meth in ('deposited merge_densest', 'resolution-only per block'):
            if meth.startswith('deposited'):
                lab, g = cc.coarsen(par, bl, cc.BLOCK_TARGETS, seed=42), 30.0
            else:
                g, lab = _m.sweep(uu, vv, bl, 16, lambda s: None)
            G, ncross = _m.quotient(uu, vv, lab)
            M = st.measure(G); sc = st.scores(M, ref)
            den = 2 * M['m'] / (M['n'] * (M['n'] - 1))
            rows.append(dict(arm=arm, mode=mode, fixed=fix, coarsening=meth,
                             alpha3=float(a3), accept_rate=accn / cand,
                             trunc_rounds=tr_r, trunc_edges=tr_e,
                             parent_m=len(uu), mono=mono, n=M['n'], m=M['m'],
                             density=den, density_err=100 * (den - bd) / bd,
                             survive=ncross / len(uu),
                             clustering=M['clustering'],
                             gap=float(M['sv'][0] - M['sv'][1]),
                             deg_sd=float(M['degree'].std()),
                             score=float(sc['SCORE (mean)']),
                             **{k: float(x) for k, x in sc.items()}))
            log(f'      {meth:<28} m={M["m"]:>7,} dens {100*(den-bd)/bd:+7.2f}%  '
                f'T1 {sc["T1 degree"]:.4f} T6 {sc["T6 triangles"]:.4f} '
                f'gap {M["sv"][0]-M["sv"][1]:6.2f} SCORE {sc["SCORE (mean)"]:.4f}')

    json.dump(dict(reference=dict(n=ref['n'], m=ref['m'], density=bd,
                                  clustering=ref['clustering'],
                                  deg_sd=float(ref['degree'].std())),
                   chromatic_rate=P, permuted_rate_unmatched=Ppm, q_extra=Q,
                   trunc_seed=TRUNC_SEED, target_parent_m=target, rows=rows),
              open(os.path.join(HERE, 'r2_truncation_fix.json'), 'w'), indent=2)

    order = ['chromatic', 'chromatic_fixed', 'random_fixed', 'permuted_fixed',
             'permuted_rate_fixed', 'off', 'off_fixed']
    log('\n' + '=' * 78)
    log('R2 RESULT -- all arms matched on parent size')
    log('=' * 78)
    for meth in ('resolution-only per block', 'deposited merge_densest'):
        gg = {r['arm']: r for r in rows if r['coarsening'] == meth}
        log(f'\n{meth}')
        log(f'  {"arm":<21}{"accept":>8}{"trunc.e":>9}{"m":>8}{"dens%":>8}'
            f'{"T1":>8}{"T6":>8}{"T5":>8}{"gap":>8}{"SCORE":>8}')
        for a in order:
            if a not in gg:
                continue
            r = gg[a]
            log(f'  {a:<21}{r["accept_rate"]:>8.4f}{r["trunc_edges"]:>9,}'
                f'{r["m"]:>8,}{r["density_err"]:>+8.2f}{r["T1 degree"]:>8.4f}'
                f'{r["T6 triangles"]:>8.4f}{r["T5 network value"]:>8.4f}'
                f'{r["gap"]:>8.2f}{r["score"]:>8.4f}')

    log('\n' + '=' * 78)
    log('VERDICTS')
    log('=' * 78)
    for meth in ('resolution-only per block', 'deposited merge_densest'):
        gg = {r['arm']: r for r in rows if r['coarsening'] == meth}
        ch, chf = gg['chromatic'], gg['chromatic_fixed']
        o, of, rf = gg['off'], gg['off_fixed'], gg['random_fixed']
        log(f'\n  {meth}')
        log(f'    CONTROL, chromatic must not move: {ch["score"]:.4f} -> '
            f'{chf["score"]:.4f} ({chf["score"]-ch["score"]:+.4f}), '
            f'm {ch["m"]:,} -> {chf["m"]:,}')
        log(f'    off repaired:                     {o["score"]:.4f} -> '
            f'{of["score"]:.4f} ({of["score"]-o["score"]:+.4f}), '
            f'density {o["density_err"]:+.2f}% -> {of["density_err"]:+.2f}%')
        log(f'    off_fixed vs random_fixed:        {of["score"]:.4f} vs '
            f'{rf["score"]:.4f} (|d| {abs(of["score"]-rf["score"]):.4f}), '
            f'density {of["density_err"]:+.2f}% vs {rf["density_err"]:+.2f}%')
        log(f'    chromatic vs a CLEAN no-filter arm: relations '
            f'{ch["m"]:,} vs {of["m"]:,} '
            f'({100*(ch["m"]-of["m"])/of["m"]:+.1f}%), '
            f'density {ch["density_err"]:+.2f}% vs {of["density_err"]:+.2f}%')
        prf = gg.get('permuted_rate_fixed')
        if prf:
            log(f'    R1 rerun, chromatic vs permuted_rate (both fixed): '
                f'density {chf["density_err"]:+.2f}% vs {prf["density_err"]:+.2f}%, '
                f'T1 {chf["T1 degree"]:.4f} vs {prf["T1 degree"]:.4f}, '
                f'T6 {chf["T6 triangles"]:.4f} vs {prf["T6 triangles"]:.4f}')


if __name__ == '__main__':
    main()
