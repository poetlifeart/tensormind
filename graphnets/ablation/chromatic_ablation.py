#!/usr/bin/env python3
"""DOES BREAKING THE CHROMATIC CONSTRAINT HELP OR HURT THE QUOTIENT?

The paper's claim is existential: a graph held to chi=3 can still sit under a
connectome-like quotient.  This asks the adjacent empirical question -- if we
RELAX that constraint during triangle closure and let a fraction f of
same-colour edges in, does the 1,015-node quotient move TOWARD Budapest or
AWAY from it?

Everything else is construct.py unchanged: same seed, same alphas (3,4,30),
same beta 1.5, same block filter, same coarsening (Louvain per block at
resolution 30, densest-pair merge to 307/306/201/201), same strict quotient.
Only the colour filter in close_triangles is relaxed.

f = 0.0 reproduces the deposited construction exactly (checked: 477,584 edges,
0 monochromatic).
"""
import sys, os, json, time
import numpy as np, networkx as nx
from scipy.sparse import csr_matrix
import importlib.util

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'construction', 'bilateral')
spec = importlib.util.spec_from_file_location("cc", os.path.join(B, 'construct.py'))
cc = importlib.util.module_from_spec(spec); sys.modules["cc"] = cc; spec.loader.exec_module(cc)
SEED_N, COLOURS, BLOCKS, TARGETS = cc.SEED_N, cc.COLOURS, cc.BLOCKS, cc.BLOCK_TARGETS

BUD = nx.read_edgelist(os.path.join(B, 'reference/budapest_1015_70654.edgelist'), nodetype=int)


def close_relaxed(u_arr, v_arr, n, colour, block, quota, rng, beta, f):
    """construct.py's close_triangles, with the colour filter relaxed.

    A candidate whose endpoints share a colour is normally rejected.  Here it
    is accepted with probability f.  f=0 is the original rule exactly.
    """
    if quota <= 0:
        return u_arr, v_arr
    adj = csr_matrix((np.ones(len(u_arr), np.int8), (u_arr, v_arr)), shape=(n, n))
    adj = adj + adj.T; adj.data[:] = 1
    indptr, indices = adj.indptr, adj.indices
    degree = np.diff(indptr)
    seen = set((np.minimum(u_arr, v_arr).astype(np.int64) * n
                + np.maximum(u_arr, v_arr).astype(np.int64)).tolist())
    eligible = np.where(degree >= 2)[0]
    if not len(eligible):
        return u_arr, v_arr
    weight = degree[eligible].astype(float) ** beta; weight /= weight.sum()
    accepted, remaining = [], quota
    for _ in range(24):
        if remaining <= 0: break
        draw = int(remaining * 3) + 10
        centre = rng.choice(eligible, size=draw, p=weight)
        deg_c = degree[centre]
        nb1 = indices[indptr[centre] + (rng.random(draw) * deg_c).astype(int)]
        nb2 = indices[indptr[centre] + (rng.random(draw) * deg_c).astype(int)]
        cross = colour[nb1] != colour[nb2]
        if f > 0:                                          # <-- the violation
            keep_mono = (~cross) & (rng.random(draw) < f)  # (drawn only when
            allow = cross | keep_mono                      #  f>0, so f=0 leaves
        else:                                              #  the RNG stream
            allow = cross                                  #  untouched)
        ok = (nb1 != nb2) & (block[nb1] == block[nb2]) & allow
        nb1, nb2 = nb1[ok], nb2[ok]
        if not len(nb1): break
        keys = np.unique(np.minimum(nb1, nb2).astype(np.int64) * n
                         + np.maximum(nb1, nb2).astype(np.int64))
        keys = np.array([k for k in keys if k not in seen], np.int64)[:remaining]
        if not len(keys): break
        seen.update(keys.tolist()); accepted.append(keys); remaining -= len(keys)
    if not accepted:
        return u_arr, v_arr
    new = np.concatenate(accepted)
    return np.concatenate([u_arr, new // n]), np.concatenate([v_arr, new % n])


def grow(f, rng, alphas=(3, 4, 30), beta=1.5, n_steps=3):
    se = cc.build_seed()
    su = np.array([a for a, b in se] + [b for a, b in se], np.int64)
    sv = np.array([b for a, b in se] + [a for a, b in se], np.int64)
    u = np.array([a for a, b in se], np.int64); v = np.array([b for a, b in se], np.int64)
    n = SEED_N; colour, block = COLOURS.copy(), BLOCKS.copy()
    u, v = close_relaxed(u, v, n, colour, block, int(alphas[0] * n), rng, beta, f)
    for step in range(2, n_steps + 1):
        u = (u[:, None] * SEED_N + su[None, :]).ravel()
        v = (v[:, None] * SEED_N + sv[None, :]).ravel()
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = np.unique(lo * (SEED_N ** step) + hi)
        u, v = keys // (SEED_N ** step), keys % (SEED_N ** step)
        n = SEED_N ** step
        digit = (np.arange(n) // (SEED_N ** (step - 1))) % SEED_N
        colour, block = COLOURS[digit], BLOCKS[digit]
        u, v = close_relaxed(u, v, n, colour, block, int(alphas[step - 1] * n), rng, beta, f)
    return u.astype(np.int32), v.astype(np.int32), colour, block


def quotient_stats(Q):
    deg = np.array([d for _, d in Q.degree()], float)
    from networkx.algorithms.community import louvain_communities, modularity
    comm = louvain_communities(Q, seed=7)
    return dict(n=Q.number_of_nodes(), m=Q.number_of_edges(),
                density=nx.density(Q), meandeg=float(deg.mean()), degsd=float(deg.std(ddof=1)),
                dmin=int(deg.min()), dmax=int(deg.max()),
                clust=nx.average_clustering(Q), trans=nx.transitivity(Q),
                Q=modularity(Q, comm), ncomm=len(comm),
                apl=nx.average_shortest_path_length(Q), diam=nx.diameter(Q))

BUDSTAT = quotient_stats(BUD)

# FULL: the nine measures originally used.  NOTE that at fixed n = 1,015 the
# first three are algebraically the same quantity -- density = 2m/(n(n-1)) and
# <k> = 2m/n -- so a mean over all nine counts ONE density effect THREE TIMES.
# Conclusions drawn from `dist` alone overstate density-driven differences.
KEYS = ['m', 'density', 'meandeg', 'degsd', 'clust', 'trans', 'Q', 'apl', 'diam']

# SHAPE: the six that are not density restatements.  Two caveats: this DROPS
# density rather than counting it once, and `diam` is 5 (the Budapest value) in
# 48 of 50 runs, so it contributes a hard zero and compresses every contrast by
# about a sixth.  Report both columns, never one alone.
SHAPE_KEYS = ['degsd', 'clust', 'trans', 'Q', 'apl', 'diam']


def _mard(s, keys):
    return float(np.mean([abs(s[k] - BUDSTAT[k]) / abs(BUDSTAT[k]) for k in keys]))


def distance_to_budapest(s):
    """Mean absolute relative deviation over the nine measures (see KEYS)."""
    return _mard(s, KEYS)


def shape_distance_to_budapest(s):
    """Same, over the six non-density measures (see SHAPE_KEYS)."""
    return _mard(s, SHAPE_KEYS)


def run(f, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    u, v, colour, block = grow(f, rng)
    mono = int((colour[u] == colour[v]).sum())
    P = nx.Graph(); P.add_nodes_from(range(len(block)))
    P.add_edges_from(zip(u.tolist(), v.tolist()))
    label = cc.coarsen(P, block, TARGETS, seed=42)
    coarse, super_block, pairs, witness = cc.build_coarse(u, v, label, block, sum(TARGETS))
    s = quotient_stats(coarse)
    # true chromatic damage: how many colour classes would be needed?
    s.update(f=f, seed=seed, parent_m=P.number_of_edges(), mono_parent=mono,
             mono_frac=mono / P.number_of_edges(),
             dist=distance_to_budapest(s),
             shape_dist=shape_distance_to_budapest(s),
             secs=round(time.time() - t0, 1))
    return s


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--control', action='store_true',
                    help='run only f=0 seed=99 and assert it reproduces the '
                         'deposited construction (about 25 seconds)')
    ap.add_argument('--f', type=float, nargs='+', default=None,
                    help='violation probabilities to sweep (default: 10 values)')
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='random seeds (default: 99 1 2 3 4)')
    ap.add_argument('--out', default='results_chromatic_ablation.json')
    args = ap.parse_args()

    if args.control:
        r = run(0.0, 99)
        ok = (r['parent_m'] == 477584 and r['mono_parent'] == 0
              and r['n'] == 1015 and r['m'] == 64760)
        print(f"f=0 seed=99:  parent {r['parent_m']:,} edges, "
              f"{r['mono_parent']} monochromatic, quotient {r['n']}/{r['m']:,}")
        print(f"distance to reference: full {r['dist']:.4f}  "
              f"shape-only {r['shape_dist']:.4f}")
        print("REPRODUCES THE DEPOSITED CONSTRUCTION:", ok)
        raise SystemExit(0 if ok else 1)

    FS = args.f if args.f is not None else [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    SEEDS = args.seeds if args.seeds is not None else [99, 1, 2, 3, 4]
    print("BUDAPEST reference:", {k: round(BUDSTAT[k], 4) for k in KEYS}, flush=True)
    print(f"\n{'f':>5s} {'seed':>5s} {'mono':>8s} {'mono%':>6s} {'parent m':>9s} "
          f"{'quot m':>7s} {'clust':>6s} {'trans':>6s} {'Q':>6s} {'apl':>6s} "
          f"{'diam':>4s} {'DIST':>7s} {'SHAPE':>7s} {'s':>5s}", flush=True)
    rows = []
    for f in FS:
        for sd in SEEDS:
            r = run(f, sd); rows.append(r)
            print(f"{f:5.2f} {sd:5d} {r['mono_parent']:8,} {100*r['mono_frac']:5.1f}% "
                  f"{r['parent_m']:9,} {r['m']:7,} {r['clust']:6.4f} {r['trans']:6.4f} "
                  f"{r['Q']:6.4f} {r['apl']:6.4f} {r['diam']:4d} {r['dist']:7.4f} "
                  f"{r['shape_dist']:7.4f} {r['secs']:5.0f}",
                  flush=True)
            json.dump(rows, open(args.out, 'w'), indent=1)
    print(f"\nwrote {args.out}")
