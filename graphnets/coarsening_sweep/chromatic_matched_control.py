#!/usr/bin/env python3
"""
chromatic_matched_control.py -- The matched control for chromatic_vs_coarsening.py.

THE CONFOUND THIS FIXES.  chromatic_vs_coarsening.py compared chi=3 enforced
against the chromatic filter removed at the same nominal closure quota alpha*n.
That is NOT a matched comparison: the filter rejects candidates, the retry loop
in close_triangles is bounded at 24 rounds, so the constrained arm never fills
its quota.  Parent edge counts came out 477,584 (enforced) against 512,180
(unconstrained) -- the unconstrained arm is 7.2% larger, and part of the
measured effect is therefore size, not colour.

THE FIX.  Tune the unconstrained arm's step-3 alpha down until its parent has
the same number of edges as the enforced arm, then coarsen both the same two
ways.  Same procedure, same seed, same block filter; only the closure
composition and the quota differ.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
BILATERAL = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))
cvc = importlib.util.spec_from_file_location(
    'cvc', os.path.join(HERE, 'chromatic_vs_coarsening.py'))
m = importlib.util.module_from_spec(cvc); sys.modules['cvc'] = m
cvc.loader.exec_module.__self__ if False else cvc.loader.exec_module(m)
cc, st = m.cc, m.st
TARGET_N = m.TARGET_N


def parent_m(alpha3, chromatic):
    u, v, _, _ = m.grow(cc.build_seed(), [3, 4, alpha3], 1.5, 3,
                        np.random.default_rng(99), chromatic)
    return len(u)


def main():
    ref = st.measure(st.build_graph(st.load_edges(
        os.path.join(BILATERAL, 'reference', 'budapest_1015_70654.edgelist'))))
    bd = 2 * ref['m'] / (ref['n'] * (ref['n'] - 1))
    print('=' * 78)
    print('MATCHED CONTROL -- parent edge count held equal across the two arms')
    print('=' * 78)

    target = parent_m(30, True)
    print(f'\n  chi=3 enforced, alpha3=30  ->  parent m = {target:,}   (the target)')

    # bisect alpha3 for the unconstrained arm
    lo, hi = 20.0, 30.0
    print(f'  tuning the unconstrained arm\'s alpha3 to match {target:,}:')
    best = None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        mm = parent_m(mid, False)
        print(f'      alpha3 {mid:7.4f} -> parent m {mm:>8,}  ({mm - target:+,})')
        if best is None or abs(mm - target) < abs(best[1] - target):
            best = (mid, mm)
        if mm > target:
            hi = mid
        else:
            lo = mid
        if abs(mm - target) <= 200:
            break
    a3, mm = best
    print(f'  chosen alpha3 = {a3:.4f}  ->  parent m = {mm:,} '
          f'({100 * (mm - target) / target:+.2f}% vs the enforced arm)')

    rows = []
    for chromatic, alpha3, tag in ((True, 30.0, 'chi=3 ENFORCED'),
                                   (False, a3, 'chromatic OFF, size-matched')):
        u, v, colour, block = m.grow(cc.build_seed(), [3, 4, alpha3], 1.5, 3,
                                     np.random.default_rng(99), chromatic)
        mono = int((colour[u] == colour[v]).sum())
        print(f'\n{tag}:  parent m={len(u):,}  monochromatic={mono:,} '
              f'({100 * mono / len(u):.1f}%)')
        parent = nx.Graph(); parent.add_nodes_from(range(8000))
        parent.add_edges_from(zip(u.tolist(), v.tolist()))
        for meth in ('deposited merge_densest', 'resolution-only per block'):
            if meth.startswith('deposited'):
                lab, gamma = cc.coarsen(parent, block, cc.BLOCK_TARGETS, seed=42), 30.0
            else:
                gamma, lab = m.sweep(u, v, block, 16, lambda s: None)
            G, n_cross = m.quotient(u, v, lab)
            M = st.measure(G); sc = st.scores(M, ref)
            den = 2 * M['m'] / (M['n'] * (M['n'] - 1))
            rows.append(dict(tag=tag, chromatic=chromatic, alpha3=float(alpha3),
                             coarsening=meth, gamma=float(gamma),
                             parent_m=len(u), mono=mono, n=M['n'], m=M['m'],
                             density=den, survive=n_cross / len(u),
                             clustering=M['clustering'],
                             transitivity=M['transitivity'],
                             gap=float(M['sv'][0] - M['sv'][1]),
                             deg_sd=float(M['degree'].std()),
                             score=float(sc['SCORE (mean)']),
                             **{k: float(x) for k, x in sc.items()}))
            print(f'    {meth:<28} gamma={gamma:<8.4g} n={M["n"]:>5,} '
                  f'm={M["m"]:>7,} density={den:.5f} SCORE {sc["SCORE (mean)"]:.4f}')

    json.dump(dict(budapest=dict(n=ref['n'], m=ref['m'], density=bd,
                                 clustering=ref['clustering']), rows=rows),
              open(os.path.join(HERE, 'chromatic_matched_control.json'), 'w'),
              indent=2)

    print('\n' + '=' * 78)
    print('MATCHED RESULT')
    print('=' * 78)
    print(f'{"coarsening":<26}{"chrom":>7}{"parent m":>10}{"n":>7}{"m":>8}'
          f'{"dm%":>8}{"density":>9}{"dd%":>8}{"clust":>7}{"SCORE":>8}')
    print(f'{"Budapest":<26}{"":>7}{"":>10}{ref["n"]:>7,}{ref["m"]:>8,}{"":>8}'
          f'{bd:>9.5f}{"":>8}{ref["clustering"]:>7.4f}{"":>8}')
    for meth in ('deposited merge_densest', 'resolution-only per block'):
        for r in rows:
            if r['coarsening'] != meth: continue
            print(f'{meth:<26}{"ON" if r["chromatic"] else "OFF":>7}'
                  f'{r["parent_m"]:>10,}{r["n"]:>7,}{r["m"]:>8,}'
                  f'{100*(r["m"]-ref["m"])/ref["m"]:>+8.2f}{r["density"]:>9.5f}'
                  f'{100*(r["density"]-bd)/bd:>+8.2f}{r["clustering"]:>7.4f}'
                  f'{r["score"]:>8.4f}')
    for meth in ('deposited merge_densest', 'resolution-only per block'):
        on = next(r for r in rows if r['coarsening'] == meth and r['chromatic'])
        off = next(r for r in rows if r['coarsening'] == meth and not r['chromatic'])
        print(f'\n  {meth}: at matched parent size, chi=3 changes score by '
              f'{on["score"]-off["score"]:+.4f} '
              f'(density error {100*(on["density"]-bd)/bd:+.2f}% vs '
              f'{100*(off["density"]-bd)/bd:+.2f}%)')


if __name__ == '__main__':
    main()
