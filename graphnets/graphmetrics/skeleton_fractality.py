#!/usr/bin/env python3
"""
skeleton_fractality.py -- box-covering fractality of a graph and of its
SKELETON.

A fractal network, in the sense of Goh, Salvi, Kahng & Kim (PRL 96, 018701,
2006), IS a skeleton -- the spanning tree of highest edge betweenness --
dressed with local shortcuts.  The skeleton carries the fractality; the
shortcuts make the full graph small-world.  Measuring the full graph measures
the dressed object, which is why dense low-diameter graphs come out
non-fractal even when their backbone is not.

This computes both, on the same covering, so they can be compared directly.

    box covering     greedy ball cover: repeatedly claim the ball of radius r
                     around an uncovered vertex.  N_B(r) is the count.
    fractal?         fit log N_B on log r, and an exponential on the same
                     points; report which wins.  A power law winning is the
                     signature of fractality; an exponential is the signature
                     of a small-world graph (Song, Havlin & Makse 2005).
    skeleton         maximum spanning tree weighted by edge betweenness.

Betweenness is exact by default.  It is O(nm), so use --pivots on large
graphs to sample it; the skeleton is then approximate.  Above ~20,000 nodes
sampling is effectively required.

READ THE FIT COUNT.  A box-covering fit on fewer than about 10 points is not
evidence either way -- that is how graphs with provably bounded diameter have
been reported as fractal.  The `npts` field is printed and written for exactly
this reason.
"""
import argparse, json, os
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components


def load_graph(path):
    if path.endswith('.gml'):
        G = nx.Graph(nx.read_gml(path))
    elif path.endswith('.npz'):
        d = np.load(path)
        G = nx.Graph()
        if 'edges' in d:
            G.add_edges_from(map(tuple, d['edges'].tolist()))
        else:                                   # tripartite colour-class masks
            n1, n2, n3 = int(d['n1']), int(d['n2']), int(d['n3'])
            G.add_nodes_from(range(n1 + n2 + n3))
            # FIXED 2026-09-25.  The offsets were applied the wrong way round:
            # this read (row + lower_offset, col + higher_offset), while every
            # other loader in the repository reads (col + lower, row + higher)
            # -- see connectome_audit_gold._graph_from_masks, run_complexity,
            # run_controllability and combinatorial_metrics_npz, and the writers
            # graph_sparsify_memb / graph_bridge_paper, which build mask_12 as
            # (n2, n1) with mask_12[higher, lower] = 1.
            #   Measured on the 8,000-node parent before the fix: 7,200 nodes
            #   instead of 8,000 (800 fell out as isolates), only 233,906 of
            #   477,584 edges correct, and 21,795 edges monochromatic under the
            #   true colouring against 0 in the real graph.  mask_13 is (n3, n1),
            #   so the swap did not merely transpose -- it used row indices up to
            #   n3-1 as colour-1 vertex ids, which that class does not have.
            # Only the mask branch was affected; 'edges' .npz and .gml were not.
            for k, (r_off, c_off) in (('mask_12', (n1, 0)),
                                      ('mask_13', (n1 + n2, 0)),
                                      ('mask_23', (n1 + n2, n1))):
                if k not in d:
                    continue
                M = d[k]
                r, c = (np.nonzero(M) if M.ndim == 2 else (M[:, 0], M[:, 1]))
                # row indexes the HIGHER colour class, col the LOWER
                G.add_edges_from(zip((c + c_off).tolist(), (r + r_off).tolist()))
    else:
        raise SystemExit(f"unsupported graph file: {path}")
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def _csr(G):
    idx = {v: i for i, v in enumerate(G.nodes())}
    e = np.array([[idx[u], idx[v]] for u, v in G.edges()], np.int64)
    n = G.number_of_nodes()
    A = csr_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    return (A + A.T).astype(np.int8).tocsr()


def ball_cover(A, n, r, rng):
    """Greedy: claim the ball of radius r around an uncovered vertex, repeat.
    The frontier expands THROUGH covered vertices; only uncovered ones are
    claimed.  Filtering the frontier truncates balls at box boundaries and
    drives the fitted slope toward zero."""
    covered = np.zeros(n, bool); boxes = 0
    indptr, indices = A.indptr, A.indices
    for s in rng.permutation(n):
        if covered[s]:
            continue
        boxes += 1
        seen = np.zeros(n, bool); seen[s] = True
        fr = np.array([s], dtype=np.int64); covered[s] = True
        for _ in range(r):
            if len(fr) == 0:
                break
            nb = np.unique(np.concatenate(
                [indices[indptr[x]:indptr[x + 1]] for x in fr]))
            nb = nb[~seen[nb]]
            if len(nb) == 0:
                break
            seen[nb] = True; covered[nb] = True; fr = nb
    return boxes


def box_fit(G, rng, trials=3, nsrc=300):
    A = _csr(G); n = A.shape[0]
    ncomp, lab = connected_components(A, directed=False)
    sz = np.bincount(lab)
    keep = np.flatnonzero(lab == sz.argmax())
    Ag = A[keep][:, keep].tocsr(); ng = Ag.shape[0]
    src = rng.choice(ng, min(nsrc, ng), replace=False)
    D = shortest_path(Ag, method='D', unweighted=True, directed=False, indices=src)
    d = D[np.isfinite(D)]; d = d[d > 0]
    diam = int(d.max())
    rs = list(range(1, max(2, diam // 2) + 1))
    counts = [min(ball_cover(Ag, ng, r, rng) for _ in range(trials)) for r in rs]
    s = np.array([r for r, c in zip(rs, counts) if c > 1], float)
    c = np.array([x for x in counts if x > 1], float)
    out = dict(n=int(ng), m=int(G.number_of_edges()), diameter=diam,
               giant_fraction=float(sz.max() / n),
               radii=[int(x) for x in s], counts=[int(x) for x in c],
               npts=int(len(s)))
    if len(s) < 4:
        out.update(d_B=None, r2_power=None, r2_exp=None, gap=None,
                   verdict='too few points')
        return out
    x, y = np.log(s), np.log(c)
    m1, b1 = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    r2 = 1 - ((y - (m1 * x + b1)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    m2, b2 = np.linalg.lstsq(np.vstack([s, np.ones_like(s)]).T, y, rcond=None)[0]
    e2 = 1 - ((y - (m2 * s + b2)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out.update(d_B=float(-m1), r2_power=float(r2), r2_exp=float(e2),
               gap=float(r2 - e2),
               verdict=('power law' if r2 > e2 else 'exponential'))
    return out


def skeleton(G, pivots=None):
    """Maximum spanning tree by edge betweenness (Goh et al.)."""
    eb = (nx.edge_betweenness_centrality(G) if not pivots else
          nx.edge_betweenness_centrality(G, k=min(pivots, G.number_of_nodes()),
                                         seed=0))
    H = nx.Graph(); H.add_nodes_from(G.nodes())
    for (u, v), w in eb.items():
        H.add_edge(u, v, w=-w)
    return nx.minimum_spanning_tree(H, weight='w')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--graph', required=True, help='.npz or .gml')
    ap.add_argument('--json', required=True, help='output file')
    ap.add_argument('--pivots', type=int, default=0,
                    help='sample betweenness with this many pivots '
                         '(0 = exact; required above ~20k nodes)')
    ap.add_argument('--trials', type=int, default=3,
                    help='ball-cover restarts per radius (min is kept)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    G = load_graph(a.graph)
    print(f"{os.path.basename(a.graph)}: n={G.number_of_nodes()} "
          f"m={G.number_of_edges()}")

    full = box_fit(G, rng, a.trials)
    T = skeleton(G, a.pivots or None)
    skel = box_fit(T, rng, a.trials)

    def show(tag, r):
        gap = "   n/a" if r['gap'] is None else f"{r['gap']:+.4f}"
        dB = "  n/a" if r['d_B'] is None else f"{r['d_B']:5.2f}"
        print(f"  {tag:10s} diam={r['diameter']:4d}  npts={r['npts']:3d}  "
              f"d_B={dB}  gap={gap}  {r['verdict']}")
    show('full', full)
    show('skeleton', skel)
    if skel['npts'] < 10:
        print("  NOTE: skeleton fit rests on fewer than 10 points -- "
              "suggestive, not conclusive.")

    json.dump(dict(graph=a.graph, betweenness=('exact' if not a.pivots
                                               else f'sampled/{a.pivots}'),
                   trials=a.trials, seed=a.seed, full=full, skeleton=skel),
              open(a.json, 'w'), indent=1)
    print(f"  wrote {a.json}")


if __name__ == '__main__':
    main()
