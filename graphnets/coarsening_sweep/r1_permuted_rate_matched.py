#!/usr/bin/env python3
"""
r1_permuted_rate_matched.py -- R1, the decisive colour control.

THE QUESTION.  chromatic_null_controls.py established that the chromatic rule
beats a RATE-MATCHED random null on density, the degree distribution and triangle
participation, under both coarsenings.  So something about the RULE matters, not
just the thinning.  But which?

    (a) THE SPECIFIC COLOURING -- the classes track the tensor fibres, because
        colour is read off the leading base-20 digit exactly as block is; or
    (b) ANY "ENDPOINTS MUST DIFFER" RULE over three classes of sizes
        (2400, 2400, 3200), with the tensor correspondence incidental.

The `permuted` arm tests (b): the same must-differ rule against a RANDOM
PERMUTATION of the colour array, class sizes preserved, so the labels no longer
track the fibres.  It reached density -4.71% under the per-block cut, essentially
matching chromatic's -4.98%, which points at (b).

BUT IT PROVES NOTHING AS RUN, because permuted rejects only 33.9% where chromatic
rejects 51.0%.  Rule and rejection rate are confounded: permuted thins less, and
thinning by itself moves density.

WHAT THIS SCRIPT ADDS.  A `permuted_rate` arm: the permuted must-differ rule AND
extra random rejection, so its conditional acceptance rate matches chromatic's
0.4904.  Then all four arms below share a parent edge count AND a rejection rate,
and the only thing varying is the rule.

    chromatic       colour[nb1] != colour[nb2]                        p ~ 0.4904
    permuted_rate   pcolour[nb1] != pcolour[nb2]  AND  rng() < q      p ~ 0.4904
    random          rng() < 0.4904                                    p ~ 0.4904
    permuted        pcolour[nb1] != pcolour[nb2]   (unmatched, for reference)

READING IT.  Compare chromatic against permuted_rate on density, T1 and T6 --
the three measures that survived every earlier control:
    * chromatic still ahead  => (a).  The specific colouring does work, the
      tensor correspondence matters, and the paper's chromatic language stands.
    * roughly equal          => (b).  Only the partition's shape matters, and the
      chromatic language must soften to "a three-class must-differ constraint".
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

STATS = {}


def close_triangles(u_arr, v_arr, n, colour, block, quota, rng, beta,
                    mode, p_accept, q_extra, pcolour, arm):
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
        elif mode == 'permuted_rate':
            rule = (pcolour[nb1] != pcolour[nb2]) & (rng.random(draw) < q_extra)
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


def grow(alphas, mode, p_accept=1.0, q_extra=1.0, arm='', perm_seed=7):
    N = cc.SEED_N
    se = cc.build_seed()
    su = np.array([a for a, b in se] + [b for a, b in se], np.int64)
    sv = np.array([b for a, b in se] + [a for a, b in se], np.int64)
    u = np.array([a for a, b in se], np.int64)
    v = np.array([b for a, b in se], np.int64)
    rng = np.random.default_rng(99)
    prng = np.random.default_rng(perm_seed)
    n = N
    colour, block = cc.COLOURS.copy(), cc.BLOCKS.copy()
    pc = prng.permutation(colour)
    u, v = close_triangles(u, v, n, colour, block, int(alphas[0] * n), rng, 1.5,
                           mode, p_accept, q_extra, pc, arm)
    for step in range(2, 4):
        u = (u[:, None] * N + su[None, :]).ravel()
        v = (v[:, None] * N + sv[None, :]).ravel()
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = np.unique(lo * (N ** step) + hi)
        u, v = keys // (N ** step), keys % (N ** step)
        n = N ** step
        digit = (np.arange(n) // (N ** (step - 1))) % N
        colour, block = cc.COLOURS[digit], cc.BLOCKS[digit]
        pc = prng.permutation(colour)
        u, v = close_triangles(u, v, n, colour, block, int(alphas[step - 1] * n),
                              rng, 1.5, mode, p_accept, q_extra, pc, arm)
    return u.astype(np.int32), v.astype(np.int32), colour, block


def tune_alpha(mode, target, p, q, arm, log):
    lo, hi, best = 5.0, 60.0, None
    for _ in range(13):
        mid = 0.5 * (lo + hi)
        STATS.clear()
        uu, vv, cl, bl = grow([3, 4, mid], mode, p, q, arm)
        mm = len(uu)
        log(f'      alpha3 {mid:7.3f} -> parent m {mm:>8,} ({mm-target:+,})')
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
    log('R1 -- IS IT THIS COLOURING, OR ANY THREE-CLASS MUST-DIFFER RULE?')
    log('=' * 78)

    # chromatic reference: its parent size and its own acceptance rate
    STATS.clear()
    u, v, colour, block = grow([3, 4, 30], 'chromatic', arm='chromatic')
    target = len(u)
    c_cand, c_acc = STATS['chromatic']
    P = c_acc / c_cand
    log(f'\nchromatic: parent m = {target:,}   accept rate p = {P:.4f} '
        f'({100*(1-P):.1f}% rejected)')

    # permuted's own (unmatched) rate, to derive the extra rejection q
    STATS.clear()
    grow([3, 4, 30], 'permuted', arm='permuted_probe')
    pm_cand, pm_acc = STATS['permuted_probe']
    Ppm = pm_acc / pm_cand
    Q = P / Ppm
    log(f'permuted:  accept rate {Ppm:.4f} unmatched -> extra rejection '
        f'q = {P:.4f}/{Ppm:.4f} = {Q:.4f}')
    log(f'           so permuted_rate should land at ~{Ppm*Q:.4f}')

    arms = [('chromatic',     'chromatic',     30.0, 1.0, 1.0),
            ('permuted_rate', 'permuted_rate', None, 1.0, Q),
            ('random',        'random',        None, P,   1.0),
            ('permuted',      'permuted',      None, 1.0, 1.0)]

    rows = []
    for arm, mode, a3fix, p, q in arms:
        log(f'\n--- {arm} ---')
        if a3fix is not None:
            a3 = a3fix
            STATS.clear()
            uu, vv, cl, bl = grow([3, 4, a3], mode, p, q, arm)
        else:
            a3 = tune_alpha(mode, target, p, q, arm, log)
            STATS.clear()
            uu, vv, cl, bl = grow([3, 4, a3], mode, p, q, arm)
        cand, acc = STATS[arm]
        rate = acc / cand
        mono = int((cl[uu] == cl[vv]).sum())
        log(f'    alpha3 {a3:.4f}  parent m {len(uu):,} '
            f'({100*(len(uu)-target)/target:+.2f}%)  accept {rate:.4f}  '
            f'mono {mono:,} ({100*mono/len(uu):.1f}%)')

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
            rows.append(dict(arm=arm, mode=mode, alpha3=float(a3),
                             accept_rate=rate, q_extra=float(q),
                             coarsening=meth, gamma=float(gamma),
                             parent_m=len(uu), mono=mono,
                             n=M['n'], m=M['m'], density=den,
                             density_err=100*(den-bd)/bd,
                             clustering=M['clustering'],
                             transitivity=M['transitivity'],
                             gap=float(M['sv'][0]-M['sv'][1]),
                             deg_sd=float(M['degree'].std()),
                             score=float(sc['SCORE (mean)']),
                             **{k: float(x) for k, x in sc.items()}))
            log(f'      {meth:<28} n={M["n"]:>5,} m={M["m"]:>7,} '
                f'dens {100*(den-bd)/bd:+.2f}%  T1 {sc["T1 degree"]:.4f}  '
                f'T6 {sc["T6 triangles"]:.4f}  SCORE {sc["SCORE (mean)"]:.4f}')

    json.dump(dict(reference=dict(n=ref['n'], m=ref['m'], density=bd,
                                  clustering=ref['clustering'],
                                  deg_sd=float(ref['degree'].std())),
                   chromatic_rate=P, permuted_rate_unmatched=Ppm, q_extra=Q,
                   target_parent_m=target, rows=rows),
              open(os.path.join(HERE, 'r1_permuted_rate_matched.json'), 'w'),
              indent=2)

    log('\n' + '=' * 78)
    log('R1 RESULT.  All arms matched on parent size; first three on rate too.')
    log('=' * 78)
    order = ['chromatic', 'permuted_rate', 'random', 'permuted']
    for meth in ('resolution-only per block', 'deposited merge_densest'):
        g = {r['arm']: r for r in rows if r['coarsening'] == meth}
        log(f'\n{meth}')
        log(f'  {"arm":<15}{"accept":>8}{"dens%":>9}{"T1 deg":>9}{"T6 tri":>9}'
            f'{"T5 nv":>9}{"clust":>8}{"gap":>8}{"SCORE":>8}')
        log(f'  {"reference":<15}{"":>8}{0.0:>9.2f}{"":>9}{"":>9}{"":>9}'
            f'{ref["clustering"]:>8.4f}{"":>8}{"":>8}')
        for a in order:
            if a not in g:
                continue
            r = g[a]
            log(f'  {a:<15}{r["accept_rate"]:>8.4f}{r["density_err"]:>+9.2f}'
                f'{r["T1 degree"]:>9.4f}{r["T6 triangles"]:>9.4f}'
                f'{r["T5 network value"]:>9.4f}{r["clustering"]:>8.4f}'
                f'{r["gap"]:>8.2f}{r["score"]:>8.4f}')

    log('\n' + '=' * 78)
    log('THE VERDICT: chromatic vs permuted_rate on the three surviving measures')
    log('=' * 78)
    for meth in ('resolution-only per block', 'deposited merge_densest'):
        g = {r['arm']: r for r in rows if r['coarsening'] == meth}
        ch, pr = g['chromatic'], g.get('permuted_rate')
        if pr is None:
            continue
        log(f'\n  {meth}')
        for k, lab, lower_better in (('density_err', 'density |error| %', True),
                                     ('T1 degree', 'T1 degree', True),
                                     ('T6 triangles', 'T6 triangles', True),
                                     ('T5 network value', 'T5 network value', True)):
            a = abs(ch[k]) if k == 'density_err' else ch[k]
            b = abs(pr[k]) if k == 'density_err' else pr[k]
            who = 'chromatic' if a < b else ('permuted_rate' if b < a else 'tie')
            log(f'    {lab:<20} chromatic {a:>8.4f}   permuted_rate {b:>8.4f}'
                f'   -> {who}')


if __name__ == '__main__':
    main()
