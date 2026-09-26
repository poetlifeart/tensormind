#!/usr/bin/env python3
"""
export_graphs.py -- write the four graph files the rest of the repository reads.

construct.py builds the 8,000-node parent and its 1,015-supernode quotient and
prints their verification, but it only writes CSVs, and only when given --out.
The .npz/.npy files that stats.py, bundles.py, make_figures.py and the
inpainting models read are not produced by anything in the repository and are
.gitignored, so a fresh checkout does not have them.  This script produces them.

    python3 export_graphs.py                 # writes into ./graphs/
    python3 export_graphs.py --out DIR       # elsewhere
    python3 export_graphs.py --check         # verify against files already present

Outputs, all deterministic from construct.py's hard-coded seed:

    graphs/cnew_parent.npz             8,000 vertices, three tripartite masks
    graphs/cnew_parent_supernode.npz   the same, plus the supernode assignment
    graphs/cnew_coarse.npz             1,015-node strict quotient, edge list
    graphs/cnew_coarse_w.npy           one fibre-bundle weight per quotient edge

`cnew_parent_supernode.npz` is the file the recurrent models load as
`graphdynamics/graph_8k_parent_supernode.npz`; copy it there to run them on this
substrate.  --check compares ARRAY CONTENTS, not file bytes: a .npz is a zip and
records a timestamp in every entry, so two archives holding identical arrays
never have the same checksum.

FORMAT.  The parent is stored as three inter-class blocks rather than an edge
list, because a proper 3-colouring forbids intra-class edges and so there is no
mask_11.  Vertices are renumbered from construct.py's tensor order into
per-class contiguous blocks -- colour 1 occupies 0..n1-1, colour 2 occupies
n1..n1+n2-1, colour 3 the rest -- preserving ascending order inside each class.
mask_12 has shape (n2, n1): its row indexes the colour-2 endpoint and its column
the colour-1 endpoint, so an edge is read as (col, row + n1).  Every loader in
the repository uses that convention; see stats.py:load_masks.

Requires numpy and networkx.  About three minutes and ~3 GB at peak.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_construct():
    spec = importlib.util.spec_from_file_location(
        'construct', os.path.join(HERE, 'construct.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['construct'] = m
    spec.loader.exec_module(m)
    return m


def build():
    """Run construct.py's own grow() and coarsen(); return everything needed."""
    cc = _load_construct()
    rng = np.random.default_rng(99)                    # construct.py's default
    u, v, colour, block = cc.grow(cc.build_seed(), [3, 4, 30], 1.5, 3, rng,
                                  verbose=True)
    n = cc.SEED_N ** 3
    parent = nx.Graph()
    parent.add_nodes_from(range(n))
    parent.add_edges_from(zip(u.tolist(), v.tolist()))
    label = cc.coarsen(parent, block, cc.BLOCK_TARGETS, seed=42)
    coarse, super_block, pairs, witness = cc.build_coarse(
        u, v, label, block, sum(cc.BLOCK_TARGETS))
    return u, v, colour, label, coarse, pairs, witness, n


def to_masks(u, v, colour, n):
    """Renumber into per-class contiguous blocks and build the three masks."""
    sizes = [int((colour == c).sum()) for c in (0, 1, 2)]
    n1, n2, n3 = sizes
    # position of each vertex inside its own colour class, ascending by index
    pos = np.empty(n, np.int64)
    for c in (0, 1, 2):
        idx = np.flatnonzero(colour == c)
        pos[idx] = np.arange(len(idx))

    masks = {'mask_12': np.zeros((n2, n1), np.uint8),
             'mask_13': np.zeros((n3, n1), np.uint8),
             'mask_23': np.zeros((n3, n2), np.uint8)}
    cu, cv = colour[u], colour[v]
    for lo, hi, key in ((0, 1, 'mask_12'), (0, 2, 'mask_13'), (1, 2, 'mask_23')):
        s = ((cu == lo) & (cv == hi))
        t = ((cu == hi) & (cv == lo))
        rows = np.concatenate([pos[v[s]], pos[u[t]]])      # the higher colour
        cols = np.concatenate([pos[u[s]], pos[v[t]]])      # the lower colour
        masks[key][rows, cols] = 1
    assert int(sum(m.sum() for m in masks.values())) == len(u), \
        'mask edge count does not match the parent'
    return (n1, n2, n3), masks, pos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'graphs'),
                    help='directory to write into (default: ./graphs)')
    ap.add_argument('--check', action='store_true',
                    help='compare against files already in --out instead of '
                         'overwriting them')
    a = ap.parse_args()

    print('building the 8,000-node parent and its quotient ...', flush=True)
    u, v, colour, label, coarse, pairs, witness, n = build()
    (n1, n2, n3), masks, pos = to_masks(u, v, colour, n)
    n_super = int(label.max()) + 1

    # supernode id per vertex, in per-class order
    sid = {}
    for c, key in ((0, 'supernode_id_1'), (1, 'supernode_id_2'), (2, 'supernode_id_3')):
        idx = np.flatnonzero(colour == c)
        sid[key] = label[idx].astype(np.int32)

    edges = np.array(sorted((min(x, y), max(x, y)) for x, y in coarse.edges()),
                     dtype=np.int64)
    # pairs came from np.unique on lo*n_super+hi, so it is already in the same
    # lexicographic order as `edges`; assert rather than assume.
    from_pairs = np.stack([pairs // n_super, pairs % n_super], 1).astype(np.int64)
    assert np.array_equal(from_pairs, edges), 'weight order does not match edges'

    print(f'  parent   {n:,} vertices, {len(u):,} edges, classes {n1}/{n2}/{n3}')
    print(f'  quotient {coarse.number_of_nodes():,} supernodes, '
          f'{coarse.number_of_edges():,} edges')

    os.makedirs(a.out, exist_ok=True)
    targets = {
        'cnew_parent.npz': dict(kind='npz', data=dict(
            n1=n1, n2=n2, n3=n3, **masks)),
        'cnew_parent_supernode.npz': dict(kind='npz', data=dict(
            n1=n1, n2=n2, n3=n3, **masks, **sid, n_supernodes=n_super)),
        'cnew_coarse.npz': dict(kind='npz', data=dict(
            n_total=coarse.number_of_nodes(), edges=edges)),
        'cnew_coarse_w.npy': dict(kind='npy', data=witness.astype(np.float64)),
    }

    ok = True
    for name, spec in targets.items():
        p = os.path.join(a.out, name)
        if a.check:
            if not os.path.exists(p):
                print(f'  [MISSING] {name}'); ok = False; continue
            same, detail = _same_content(p, spec)
            ok &= same
            print(f'  [{"MATCH" if same else "DIFFERS"}] {name}  {detail}')
        else:
            _write(p, spec)
            print(f'  wrote {name}')

    if a.check:
        print('\nALL FOUR MATCH -- the rebuild reproduces the deposited files'
              if ok else '\nMISMATCH -- see above')
        raise SystemExit(0 if ok else 1)

    print(f'\nwrote four files into {a.out}')
    print('To run the recurrent models on this substrate, copy the supernode '
          'file across:\n'
          f'  cp {os.path.join(a.out, "cnew_parent_supernode.npz")} '
          '../../../graphdynamics/graph_8k_parent_supernode.npz')


def _same_content(path, spec):
    """Compare ARRAY CONTENTS, not file bytes.  A .npz is a zip and embeds a
    timestamp, so two archives of identical arrays never have the same md5."""
    if spec['kind'] == 'npy':
        have = np.load(path)
        want = spec['data']
        same = have.shape == want.shape and np.array_equal(have, want)
        return same, f'{have.shape} {"equal" if same else "NOT equal"}'
    have = np.load(path)
    want = spec['data']
    hk, wk = sorted(have.files), sorted(want)
    if hk != wk:
        return False, f'key sets differ: {hk} vs {wk}'
    bad = [k for k in wk if not np.array_equal(np.asarray(have[k]), np.asarray(want[k]))]
    return (not bad), (f'{len(wk)} arrays equal' if not bad
                       else f'differing arrays: {bad}')


def _write(path, spec):
    if spec['kind'] == 'npz':
        np.savez_compressed(path, **spec['data'])
    else:
        np.save(path, spec['data'])


if __name__ == '__main__':
    main()
