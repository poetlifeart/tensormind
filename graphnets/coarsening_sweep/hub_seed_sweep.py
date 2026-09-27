#!/usr/bin/env python3
"""
hub_seed_sweep.py -- Is the connector-hub verdict stable across Louvain seeds?

WHY.  connector_hubs is decided by ONE Louvain run (seed 42 in the audit), and the
criterion is simply connector_hubs > 0, where a connector hub needs within-module
degree z >= 2.5 AND participation 0.30 < P <= 0.75.  The paper's Louvain-only
quotient (per-block gamma, 1,076 supernodes) passes with exactly ONE hub, whose
participation is 0.327 against the 0.30 threshold -- a margin of 0.027.  Graph B
(shared gamma, 1,015) fails with zero, its three high-z nodes all sitting at
P <= 0.251.  The two graphs are otherwise alike: degree sd 48.0 vs 46.0, max
degree 287 vs 288.

So the difference between PASS and FAIL may be one node's participation relative
to a fixed cut, under one partition.  This session already measured that the
per-block pipeline amplifies a 3-edge parent perturbation into a 0.011 score
change because Louvain flips, so a single-seed verdict on a 0.027 margin is not
something the paper should lean on.

WHAT THIS DOES.  Recompute the hub count over many Louvain seeds for each graph,
holding the graph fixed, and report the DISTRIBUTION of verdicts rather than one
draw.  Reporting a distribution instead of a best-of is the same correction
already applied to the Louvain block-size selection (76 of 100 seeds give the
reported partition).

READING IT.  For the paper's variant:
  * hubs > 0 at nearly every seed  -> the pass is real; restore the original
    sentence and add that it passes the criterion.
  * hubs = 0 at a material fraction -> the verdict is seed-dependent, and the
    softened sentence is right, for a subtler reason than "no hubs".
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
GM = os.path.normpath(os.path.join(HERE, '..', 'graphmetrics'))
N_SEEDS = 20


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m)
    return m


audit = _load('audit', os.path.join(GM, 'connectome_audit_gold.py'))


def hubs_at_seed(G, seed):
    Gcc = G if nx.is_connected(G) else nx.subgraph(
        G, max(nx.connected_components(G), key=len)).copy()
    Q, comms, partition = audit._louvain(Gcc, seed=seed)
    pcs = audit._participation(Gcc, partition)
    nodes = list(Gcc.nodes())
    msets = {}
    for nd, c in partition.items():
        msets.setdefault(c, set()).add(nd)
    within = {nd: sum(1 for nb in Gcc.neighbors(nd) if nb in msets[partition[nd]])
              for nd in nodes}
    stats = {}
    for c, mem in msets.items():
        d = np.array([within[nd] for nd in mem], float)
        stats[c] = (d.mean(), d.std(ddof=0) if len(d) > 1 else 1.0)
    z = np.array([((within[nd] - stats[partition[nd]][0]) / stats[partition[nd]][1])
                  if stats[partition[nd]][1] > 0 else 0.0 for nd in nodes])
    hi = z >= 2.5
    conn = int(np.sum(hi & (pcs > 0.30) & (pcs <= 0.75)))
    return dict(seed=seed, hubs=conn, n_hi_z=int(hi.sum()),
                max_P_among_hi_z=float(pcs[hi].max()) if hi.any() else float('nan'),
                Q=float(Q), n_comm=len(comms))


def main():
    print('=' * 78)
    print(f'CONNECTOR-HUB VERDICT OVER {N_SEEDS} LOUVAIN SEEDS')
    print('criterion: at least one node with z_wm >= 2.5 and 0.30 < P <= 0.75')
    print('=' * 78)

    graphs = {
        'A deposited (1,015)':      'graphs/quot_A_is_cnew_coarse',
        'C own proportions (1,015)': 'graphs/quot_C_merge_own_proportions.npz',
        'B shared gamma (1,015)':    'graphs/quot_B_pure_louvain.npz',
        'PAPER per-block g (1,076)': 'graphs/quot_P_paper_louvain_perblock.npz',
    }
    # A is the deposited quotient, already on disk in the bilateral graphs dir
    graphs['A deposited (1,015)'] = os.path.normpath(os.path.join(
        HERE, '..', 'construction', 'bilateral', 'graphs', 'cnew_coarse.npz'))

    out = {}
    for name, path in graphs.items():
        p = path if os.path.isabs(path) else os.path.join(HERE, path)
        d = np.load(p)
        E = d['edges']
        G = nx.Graph(); G.add_nodes_from(range(int(d['n_total'])))
        G.add_edges_from(map(tuple, E))
        rows = [hubs_at_seed(G, s) for s in range(1, N_SEEDS + 1)]
        hubs = [r['hubs'] for r in rows]
        passes = sum(1 for h in hubs if h > 0)
        out[name] = dict(n=G.number_of_nodes(), m=G.number_of_edges(), rows=rows,
                         pass_rate=passes / len(rows),
                         hub_min=min(hubs), hub_max=max(hubs),
                         hub_median=float(np.median(hubs)))
        print(f'\n{name}   n={G.number_of_nodes():,} m={G.number_of_edges():,}')
        print(f'  hubs per seed: {hubs}')
        print(f'  PASSES {passes}/{len(rows)} seeds   '
              f'hubs min/median/max {min(hubs)}/{np.median(hubs):.1f}/{max(hubs)}')
        mp = [r['max_P_among_hi_z'] for r in rows if not np.isnan(r['max_P_among_hi_z'])]
        if mp:
            print(f'  max participation among high-z nodes: '
                  f'{min(mp):.3f} .. {max(mp):.3f}   (threshold 0.30)')
        print(f'  n high-z nodes per seed: {[r["n_hi_z"] for r in rows]}')

    json.dump(out, open(os.path.join(HERE, 'hub_seed_sweep.json'), 'w'), indent=2)

    print('\n' + '=' * 78)
    print('VERDICT STABILITY')
    print('=' * 78)
    print(f'  {"graph":<28}{"passes":>10}{"hubs median":>13}{"reading":>28}')
    for name, r in out.items():
        rate = r['pass_rate']
        if rate == 1.0:
            read = 'stable PASS'
        elif rate == 0.0:
            read = 'stable FAIL'
        else:
            read = 'SEED-DEPENDENT'
        print(f'  {name:<28}{f"{int(rate*N_SEEDS)}/{N_SEEDS}":>10}'
              f'{r["hub_median"]:>13.1f}{read:>28}')


if __name__ == '__main__':
    main()
