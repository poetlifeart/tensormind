#!/usr/bin/env python3
"""
connector_hub_count.py -- Count Guimera-Amaral connector hubs without running the
five-hour audit.

WHY THIS IS POSSIBLE.  The connector_hubs criterion needs NO null ensemble.  From
connectome_audit_gold.py: the partition is one Louvain run at seed 42, the
within-module degree z-score is z_wm, the participation coefficient is
P = 1 - sum_c (k_ic/k_i)^2, and

    connector hub  <=>  z_wm >= 2.5  AND  0.30 < P <= 0.75
    criterion passes <=>  connector_hubs > 0

So the whole criterion is seconds of work.  Everything else in the audit needs
100-1000 nulls; this does not.

WHAT IT IS FOR.  Graph B -- one gamma shared across all four blocks, swept to the
total nearest 1,015, no merge step -- audited 8/9, failing connector_hubs with
ZERO of them.  But B is NOT the Louvain-only coarsening the paper reports.  The
paper's version (blockwise/blockwise_pipeline.py) sweeps a SEPARATE gamma inside
each block and lands at 1,076 supernodes, not 1,015.  A different gamma per block
could give a different spread of community sizes, and community-size spread is
exactly what B lacks.  So B's failure does not automatically carry over.

VALIDATION FIRST.  The counter is run on A, B and C, whose audited hub counts are
known (9, 0, 10).  If it does not reproduce those three it is not trusted on the
1,076-node graph.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
BIL = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))
GM = os.path.normpath(os.path.join(HERE, '..', 'graphmetrics'))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


audit = _load('audit', os.path.join(GM, 'connectome_audit_gold.py'))
_m = _load('cvc', os.path.join(HERE, 'chromatic_vs_coarsening.py'))
cc = _m.cc


def hub_count(G, seed=42):
    """Exactly connectome_audit_gold's connector-hub computation, no nulls."""
    Gcc = G if nx.is_connected(G) else nx.subgraph(
        G, max(nx.connected_components(G), key=len)).copy()
    Q, comms, partition = audit._louvain(Gcc, seed=seed)
    pcs = audit._participation(Gcc, partition)

    nodes = list(Gcc.nodes())
    module_sets = {}
    for nd, c in partition.items():
        module_sets.setdefault(c, set()).add(nd)
    within = {nd: sum(1 for nb in Gcc.neighbors(nd) if nb in module_sets[partition[nd]])
              for nd in nodes}
    stats = {}
    for c, mem in module_sets.items():
        d = np.array([within[nd] for nd in mem], float)
        stats[c] = (d.mean(), d.std(ddof=0) if len(d) > 1 else 1.0)
    z = np.zeros(len(nodes))
    for i, nd in enumerate(nodes):
        mu, sd = stats[partition[nd]]
        z[i] = (within[nd] - mu) / sd if sd > 0 else 0.0

    conn = int(np.sum((z >= 2.5) & (pcs > 0.30) & (pcs <= 0.75)))
    kinless = int(np.sum((z >= 2.5) & (pcs > 0.75)))
    deg = np.array([d for _, d in Gcc.degree()], float)
    return dict(n=Gcc.number_of_nodes(), m=Gcc.number_of_edges(),
                Q=float(Q), n_comm=len(comms),
                connector_hubs=conn, kinless_hubs=kinless,
                participation_mean=float(pcs.mean()),
                n_z_above_2p5=int(np.sum(z >= 2.5)),
                max_z=float(z.max()), max_P_among_z=float(
                    pcs[z >= 2.5].max()) if np.any(z >= 2.5) else float('nan'),
                deg_mean=float(deg.mean()), deg_sd=float(deg.std()),
                deg_max=int(deg.max()))


def targets_from_parent(block, total):
    sizes = np.bincount(block).astype(float)
    exact = sizes / sizes.sum() * total
    base = np.floor(exact).astype(int)
    short = total - base.sum()
    if short:
        order = np.lexsort((np.arange(len(base)), -(exact - base)))
        base[order[:short]] += 1
    return base.tolist()


def main():
    print('=' * 78)
    print('CONNECTOR HUBS WITHOUT THE AUDIT   (z_wm >= 2.5 and 0.30 < P <= 0.75)')
    print('=' * 78)

    u, v, colour, block = _m.grow(cc.build_seed(), [3, 4, 30], 1.5, 3,
                                  np.random.default_rng(99), True)
    parent = nx.Graph(); parent.add_nodes_from(range(8000))
    parent.add_edges_from(zip(u.tolist(), v.tolist()))
    print(f'\nparent n=8,000 m={len(u):,}')

    graphs = {}

    print('\nbuilding A (deposited, merge to 307/306/201/201) ...', flush=True)
    graphs['A deposited'] = _m.quotient(
        u, v, cc.coarsen(parent, block, cc.BLOCK_TARGETS, seed=42))[0]

    print('building C (merge to own proportions) ...', flush=True)
    own = targets_from_parent(block, 1015)
    graphs['C own proportions'] = _m.quotient(
        u, v, cc.coarsen(parent, block, own, seed=42))[0]

    print('building B (one shared gamma, no merge) ...', flush=True)
    gamma, labB = _m.sweep(u, v, block, 18, lambda s: None)
    graphs['B shared gamma'] = _m.quotient(u, v, labB)[0]
    print(f'   gamma {gamma:.6g}')

    print('building the PAPER\'S Louvain-only quotient '
          '(per-block gamma sweep, blockwise_pipeline) ...', flush=True)
    bw = _load('bw', os.path.join(BIL, 'blockwise', 'blockwise_pipeline.py'))
    # blockwise_pipeline.main's own default resolution ladder, verbatim
    RES = [2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 55]
    # coarsen_blockwise returns (label, offset); offset is the supernode count
    labP, nsup = bw.coarsen_blockwise(parent, block, 1015, RES, seed=42,
                                      verbose=False)
    # build_quotient returns (quotient, pairs, witness, n_intra)
    GP = bw.build_quotient(u, v, labP, nsup)[0]
    graphs[f'PAPER per-block gamma ({nsup})'] = GP
    print(f'   -> {nsup:,} supernodes')

    print('\n' + '=' * 78)
    print(f'{"graph":<30}{"n":>7}{"m":>8}{"hubs":>6}{"z>=2.5":>8}'
          f'{"maxP|z":>8}{"degmax":>8}{"degsd":>7}{"Pmean":>8}{"verdict":>9}')
    rows = {}
    for name, G in graphs.items():
        r = hub_count(G)
        rows[name] = r
        v_ = 'PASS' if r['connector_hubs'] > 0 else 'FAIL'
        print(f'{name:<30}{r["n"]:>7,}{r["m"]:>8,}{r["connector_hubs"]:>6}'
              f'{r["n_z_above_2p5"]:>8}{r["max_P_among_z"]:>8.3f}'
              f'{r["deg_max"]:>8}{r["deg_sd"]:>7.1f}'
              f'{r["participation_mean"]:>8.3f}{v_:>9}')

    json.dump(rows, open(os.path.join(HERE, 'connector_hub_count.json'), 'w'),
              indent=2)

    print('\n' + '=' * 78)
    print('VALIDATION against the audited counts (A 9, B 0, C 10)')
    print('=' * 78)
    exp = {'A deposited': 9, 'B shared gamma': 0, 'C own proportions': 10}
    ok = True
    for k, want in exp.items():
        got = rows[k]['connector_hubs']
        good = got == want
        ok &= good
        print(f'  {k:<24} audited {want:>3}   recomputed {got:>3}   '
              f'{"ok" if good else "MISMATCH"}')
    if not ok:
        print('\n  COUNTER DOES NOT REPRODUCE THE AUDIT -- do not trust the '
              'paper-version row above.')
        return
    print('\n  counter reproduces all three -> the paper-version row is trustworthy')

    pk = [k for k in rows if k.startswith('PAPER')][0]
    p = rows[pk]
    print('\n' + '=' * 78)
    print('ANSWER FOR THE PAPER\'S LOUVAIN-ONLY COARSENING')
    print('=' * 78)
    print(f'  {pk}:  {p["connector_hubs"]} connector hubs  '
          f'-> criterion would {"PASS" if p["connector_hubs"] > 0 else "FAIL"}')
    if p['connector_hubs'] > 0:
        print('  => run the full nine-criterion audit on it.  If 9/9, the paper\'s')
        print('     original sentence stands and can add "and it too passes all')
        print('     nine criteria".')
    else:
        print('  => the merge step is genuinely required, and the softened')
        print('     sentence is the correct one.')


if __name__ == '__main__':
    main()
