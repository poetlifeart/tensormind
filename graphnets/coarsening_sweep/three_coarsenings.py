#!/usr/bin/env python3
"""
three_coarsenings.py -- The three per-block coarsenings we measure on, all at
Budapest's vertex count, and the numbers each of them borrows from Budapest.

  A  density merge, Budapest targets   Louvain@30 per block, then merge_densest
                                       to 307/306/201/201.  Borrows FOUR numbers
                                       -- Budapest's own Louvain block sizes.
                                       This is the deposited construction.

  B  pure Louvain                      resolution swept per block, same gamma in
                                       all four blocks, to the total closest to
                                       1,015.  No merge step.  Borrows ONE
                                       number, the total.

  C  density merge, own proportions    Louvain@30 per block, then merge_densest
                                       to targets taken from the PARENT's own
                                       block sizes (2400/2400/1600/1600, i.e.
                                       0.3/0.3/0.2/0.2) scaled to 1,015 by
                                       largest remainder -> 305/304/203/203.
                                       Borrows ONE number, the total -- matched
                                       to B.

WHY C EXISTS.  A and B are not matched on how much they borrow: A takes four
Budapest-derived numbers and B takes one.  So A beating B on degree spread (61.3
against 46.0, target 71.55) and clustering (0.6119 against 0.5488, target 0.6696)
could be the density merge rule OR could be Budapest's block split.  C keeps the
merge rule and drops the split, separating the two.

Every graph is quotiented strictly -- supernodes joined iff at least one parent
edge runs between them, nothing thinned -- and scored with the paper's own
static_tests.py, unmodified.
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

_s = importlib.util.spec_from_file_location(
    'cvc', os.path.join(HERE, 'chromatic_vs_coarsening.py'))
_m = importlib.util.module_from_spec(_s); sys.modules['cvc'] = _m
_s.loader.exec_module(_m)
cc, st = _m.cc, _m.st
TOTAL = 1015


def targets_from_parent(block, total):
    """Scale the parent's own block sizes to `total` by largest remainder.

    Deterministic, and ties go to the lower block index.
    """
    sizes = np.bincount(block).astype(float)
    exact = sizes / sizes.sum() * total
    base = np.floor(exact).astype(int)
    short = total - base.sum()
    if short:
        order = np.lexsort((np.arange(len(base)), -(exact - base)))
        base[order[:short]] += 1
    assert base.sum() == total
    return base.tolist()


def coarsen_to(parent, block, u, v, targets, seed=42):
    """construct.coarsen with arbitrary per-block targets (it hard-codes none;
    BLOCK_TARGETS is passed in), then the strict quotient."""
    lab = cc.coarsen(parent, block, targets, seed=seed)
    return lab


def main():
    ref = st.measure(st.build_graph(st.load_edges(
        os.path.join(BILATERAL, 'reference', 'budapest_1015_70654.edgelist'))))
    bd = 2 * ref['m'] / (ref['n'] * (ref['n'] - 1))
    print('=' * 78)
    print('THREE PER-BLOCK COARSENINGS AT BUDAPEST\'S VERTEX COUNT')
    print('=' * 78)
    print(f'\nBUDAPEST  n={ref["n"]:,}  m={ref["m"]:,}  density={bd:.5f}'
          f'  C={ref["clustering"]:.4f}  T={ref["transitivity"]:.4f}'
          f'  Q={ref["modularity"]:.4f}')
    print(f'          deg mean={ref["degree"].mean():.2f}  sd={ref["degree"].std():.2f}'
          f'  gap={ref["sv"][0]-ref["sv"][1]:.2f}  APL={ref["apl"]:.3f}')

    u, v, colour, block = _m.grow(cc.build_seed(), [3, 4, 30], 1.5, 3,
                                  np.random.default_rng(99), True)
    parent = nx.Graph(); parent.add_nodes_from(range(8000))
    parent.add_edges_from(zip(u.tolist(), v.tolist()))
    print(f'\nparent  n=8,000  m={len(u):,}  block sizes {np.bincount(block).tolist()}')

    own = targets_from_parent(block, TOTAL)
    print(f'\ntargets')
    print(f'  A  Budapest block sizes      {cc.BLOCK_TARGETS}  sum {sum(cc.BLOCK_TARGETS)}'
          f'   <- borrows 4 numbers')
    print(f'  C  parent block proportions  {own}  sum {sum(own)}'
          f'   <- borrows 1 number (the total)')

    rows, labs = [], {}
    specs = [('A  density merge, Budapest targets', 'merge', cc.BLOCK_TARGETS),
             ('B  pure Louvain, resolution only', 'louvain', None),
             ('C  density merge, own proportions', 'merge', own)]
    for name, kind, targ in specs:
        if kind == 'merge':
            lab, gamma = cc.coarsen(parent, block, targ, seed=42), 30.0
        else:
            gamma, lab = _m.sweep(u, v, block, 18, lambda s: None)
        labs[name[0]] = lab
        G, n_cross = _m.quotient(u, v, lab)
        M = st.measure(G); sc = st.scores(M, ref)
        den = 2 * M['m'] / (M['n'] * (M['n'] - 1))
        sizes = np.bincount(lab)
        rows.append(dict(name=name, kind=kind, gamma=float(gamma),
                         targets=list(map(int, targ)) if targ else None,
                         borrows=(4 if targ == cc.BLOCK_TARGETS else 1),
                         n=M['n'], m=M['m'], density=den,
                         density_err=100*(den-bd)/bd,
                         survive=n_cross/len(u),
                         clustering=M['clustering'],
                         transitivity=M['transitivity'],
                         modularity=M['modularity'], apl=M['apl'],
                         diameter=M['diameter'], eff_diam=M['eff_diam'],
                         assort=float(nx.degree_assortativity_coefficient(G)),
                         lambda1=float(M['sv'][0]), lambda2=float(M['sv'][1]),
                         gap=float(M['sv'][0]-M['sv'][1]),
                         deg_mean=float(M['degree'].mean()),
                         deg_sd=float(M['degree'].std()),
                         deg_min=int(M['degree'].min()),
                         deg_max=int(M['degree'].max()),
                         super_min=int(sizes.min()), super_max=int(sizes.max()),
                         super_median=float(np.median(sizes)),
                         score=float(sc['SCORE (mean)']),
                         **{k: float(x) for k, x in sc.items()}))
        print(f'\n{name}')
        print(f'    n={M["n"]:,}  m={M["m"]:,}  density={den:.5f} ({100*(den-bd)/bd:+.2f}%)'
              f'  SCORE {sc["SCORE (mean)"]:.4f}')
        print(f'    C={M["clustering"]:.4f}  T={M["transitivity"]:.4f}'
              f'  Q={M["modularity"]:.4f}  deg sd={M["degree"].std():.1f}'
              f'  gap={M["sv"][0]-M["sv"][1]:.2f}')

    # partition agreement between the three (labels cached above, not recomputed)
    from sklearn.metrics import adjusted_rand_score as ari

    json.dump(dict(budapest=dict(n=ref['n'], m=ref['m'], density=bd,
                                 clustering=ref['clustering'],
                                 transitivity=ref['transitivity'],
                                 modularity=ref['modularity'],
                                 deg_sd=float(ref['degree'].std()),
                                 deg_min=int(ref['degree'].min()),
                                 deg_max=int(ref['degree'].max()),
                                 apl=ref['apl'],
                                 gap=float(ref['sv'][0]-ref['sv'][1])),
                   parent_targets=own, rows=rows),
              open(os.path.join(HERE, 'three_coarsenings.json'), 'w'), indent=2)

    print('\n' + '=' * 78)
    print('COMPARISON.  Budapest row first; "borrow" = numbers taken from Budapest.')
    print('=' * 78)
    h = (f'{"graph":<34}{"brw":>4}{"n":>7}{"m":>8}{"dens%":>8}{"C":>8}{"T":>8}'
         f'{"Q":>8}{"degsd":>7}{"gap":>8}{"SCORE":>8}')
    print(h)
    print(f'{"BUDAPEST":<34}{"-":>4}{ref["n"]:>7,}{ref["m"]:>8,}{"-":>8}'
          f'{ref["clustering"]:>8.4f}{ref["transitivity"]:>8.4f}'
          f'{ref["modularity"]:>8.4f}{ref["degree"].std():>7.1f}'
          f'{ref["sv"][0]-ref["sv"][1]:>8.2f}{"-":>8}')
    for r in rows:
        print(f'{r["name"]:<34}{r["borrows"]:>4}{r["n"]:>7,}{r["m"]:>8,}'
              f'{r["density_err"]:>+8.2f}{r["clustering"]:>8.4f}'
              f'{r["transitivity"]:>8.4f}{r["modularity"]:>8.4f}'
              f'{r["deg_sd"]:>7.1f}{r["gap"]:>8.2f}{r["score"]:>8.4f}')

    print('\npartition agreement (ARI)')
    ks = sorted(labs)
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            print(f'  {ks[i]} vs {ks[j]}:  {ari(labs[ks[i]], labs[ks[j]]):.4f}')

    print('\nDOES A\'S ADVANTAGE COME FROM THE MERGE RULE OR FROM BUDAPEST\'S SPLIT?')
    A, B, C = rows[0], rows[1], rows[2]
    for k, tgt, lab in (('deg_sd', ref['degree'].std(), 'degree sd'),
                        ('clustering', ref['clustering'], 'clustering'),
                        ('density_err', 0.0, 'density error %'),
                        ('score', 0.0, 'six-test mean')):
        print(f'  {lab:<16} A {A[k]:>9.4f}   C {C[k]:>9.4f}   B {B[k]:>9.4f}'
              f'    (target {tgt:.4f})')
    print('\n  C shares B\'s single borrowed number and A\'s merge rule, so C-vs-B'
          '\n  isolates the merge rule and C-vs-A isolates Budapest\'s block split.')


if __name__ == '__main__':
    main()
