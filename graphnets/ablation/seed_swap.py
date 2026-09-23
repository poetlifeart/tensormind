#!/usr/bin/env python3
"""SEED SWAP: does the 8k pipeline still reach the regime from a DIFFERENT seed?

The paper contains two constructions with two seeds, but they also differ in
pipeline (the local one prunes and repairs; the 8k one only closes), so they
cannot separate "the seed does not matter" from "the pipeline does not matter".

This holds the PIPELINE fixed -- construct.py's tensor-plus-legal-closure, no
pruning, no repair, then per-block Louvain with densest-pair merge to 1,015
supernodes, then the strict quotient -- and swaps only the SEED:

    20-vertex seed : 31 edges, colours (6,6,8), FOUR BLOCKS   -> 20^3 = 8,000
     7-vertex seed :  9 edges, colours (2,2,3), NO BLOCKS     -> 7^5 = 16,807

The 7-vertex seed is the local construction's G2, taken verbatim from
construction/global/graph_singletrack.py:make_seeds().

TWO HONEST DIFFERENCES THE SWAP CANNOT AVOID:
  1. The 7-vertex seed carries no block labels, so the closure rule's
     block(u)==block(w) filter is vacuous (one block) and the coarsening runs
     globally rather than per block.  That is a property of the seed, not a
     change of pipeline: there are no blocks to respect.
  2. Reaching a comparable size needs five tensor steps rather than three, so
     the closure schedule alpha has five entries rather than three.  It is
     chosen to land near the 8k parent's mean degree (119.4); the value is
     reported, not tuned against the audit.
"""
import sys, os, json, time, argparse
import numpy as np, networkx as nx, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, '..', 'construction', 'bilateral')
spec = importlib.util.spec_from_file_location("cc", os.path.join(B, 'construct.py'))
cc = importlib.util.module_from_spec(spec); sys.modules["cc"] = cc; spec.loader.exec_module(cc)

# the local construction's G2, verbatim from global/graph_singletrack.py
SEED7_COLOURS = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
SEED7_EDGES = [(0, 2), (0, 5), (1, 3), (1, 4), (1, 6), (2, 5), (2, 6), (3, 4), (3, 6)]


def grow7(alphas, beta, rng, n_steps=5, verbose=True):
    """construct.py's grow(), with the 7-vertex seed and a single block."""
    se = SEED7_EDGES
    su = np.array([a for a, b in se] + [b for a, b in se], np.int64)
    sv = np.array([b for a, b in se] + [a for a, b in se], np.int64)
    u = np.array([a for a, b in se], np.int64); v = np.array([b for a, b in se], np.int64)
    S = 7
    n = S
    colour = SEED7_COLOURS.copy(); block = np.zeros(n, np.int32)
    before = len(u)
    u, v = cc.close_triangles(u, v, n, colour, block, int(alphas[0] * n), rng, beta)
    if verbose:
        print(f"    seed       n={n:>7,}   product m={before:>9,}"
              f"   closure +{len(u)-before:>9,}   = {len(u):>9,}", flush=True)
    for step in range(2, n_steps + 1):
        u = (u[:, None] * S + su[None, :]).ravel()
        v = (v[:, None] * S + sv[None, :]).ravel()
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = np.unique(lo * (S ** step) + hi)
        u, v = keys // (S ** step), keys % (S ** step)
        n = S ** step
        digit = (np.arange(n) // (S ** (step - 1))) % S
        colour = SEED7_COLOURS[digit]; block = np.zeros(n, np.int32)
        before = len(u)
        u, v = cc.close_triangles(u, v, n, colour, block, int(alphas[step - 1] * n), rng, beta)
        if verbose:
            print(f"    tensor {step}   n={n:>7,}   product m={before:>9,}"
                  f"   closure +{len(u)-before:>9,}   = {len(u):>9,}", flush=True)
    assert (colour[u] == colour[v]).sum() == 0, "CHROMATIC VIOLATION"
    return u.astype(np.int32), v.astype(np.int32), colour, block


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--alpha', type=float, nargs='+', default=[3, 4, 4, 4, 3.5])
    ap.add_argument('--beta', type=float, default=1.5)
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--target', type=int, default=1015)
    ap.add_argument('--out', default=os.path.join(HERE, 'quot_seed7.npz'))
    ap.add_argument('--json', default=os.path.join(HERE, 'seed_swap_stats.json'))
    a = ap.parse_args()

    t0 = time.time()
    print(f"SEED SWAP: 7-vertex seed, {a.steps} steps -> 7^{a.steps} = {7**a.steps:,} vertices")
    print(f"  alpha={a.alpha}  beta={a.beta}  rng seed={a.seed}\n")
    rng = np.random.default_rng(a.seed)
    u, v, colour, block = grow7(a.alpha, a.beta, rng, a.steps)
    n = 7 ** a.steps
    P = nx.Graph(); P.add_nodes_from(range(n)); P.add_edges_from(zip(u.tolist(), v.tolist()))
    deg = np.array([d for _, d in P.degree()], float)
    print(f"\n  PARENT  n={n:,}  m={P.number_of_edges():,}  <k>={deg.mean():.2f}"
          f"  monochromatic={int((colour[u]==colour[v]).sum())}"
          f"  connected={nx.is_connected(P)}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"  colour class sizes {list(np.bincount(colour))}")

    print(f"\n  coarsening to {a.target} supernodes (single block, global Louvain)...", flush=True)
    label = cc.coarsen(P, block, [a.target], seed=42)
    coarse, super_block, pairs, witness = cc.build_coarse(u, v, label, block, a.target)
    Q = coarse
    qd = np.array([d for _, d in Q.degree()], float)
    from networkx.algorithms.community import louvain_communities, modularity
    comm = louvain_communities(Q, seed=7)
    st = dict(seed_vertices=7, steps=a.steps, alpha=a.alpha, beta=a.beta, rng_seed=a.seed,
              parent_n=n, parent_m=P.number_of_edges(), parent_meandeg=float(deg.mean()),
              monochromatic=int((colour[u]==colour[v]).sum()),
              quot_n=Q.number_of_nodes(), quot_m=Q.number_of_edges(),
              quot_density=nx.density(Q), quot_meandeg=float(qd.mean()),
              quot_degsd=float(qd.std(ddof=1)), quot_dmin=int(qd.min()), quot_dmax=int(qd.max()),
              quot_clust=nx.average_clustering(Q), quot_trans=nx.transitivity(Q),
              quot_Q=modularity(Q, comm), quot_ncomm=len(comm),
              quot_apl=nx.average_shortest_path_length(Q), quot_diam=nx.diameter(Q),
              secs=round(time.time()-t0, 1))
    print(f"\n  QUOTIENT  n={st['quot_n']:,}  m={st['quot_m']:,}  density={st['quot_density']:.4f}")
    print(f"    <k>={st['quot_meandeg']:.2f} sd={st['quot_degsd']:.2f} "
          f"min={st['quot_dmin']} max={st['quot_dmax']}")
    print(f"    clustering={st['quot_clust']:.4f}  transitivity={st['quot_trans']:.4f}")
    print(f"    modularity={st['quot_Q']:.4f} ({st['quot_ncomm']} communities)")
    print(f"    APL={st['quot_apl']:.4f}  diameter={st['quot_diam']}")
    print(f"\n  BUDAPEST for comparison: n=1,015 m=70,654 density=0.1373 <k>=139.22")
    print(f"    sd=71.58 min=7 max=466 clustering=0.6696 transitivity=0.5578")
    print(f"    modularity=0.5574 APL=2.2190 diameter=5")
    E = np.array(sorted((min(x,y),max(x,y)) for x,y in Q.edges()), dtype=np.int64)
    np.savez_compressed(a.out, edges=E, n_total=Q.number_of_nodes())
    json.dump(st, open(a.json,'w'), indent=1)
    print(f"\n  wrote {a.out} and {a.json}   [{st['secs']:.0f}s total]")
