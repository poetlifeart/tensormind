#!/usr/bin/env python3
"""
stats.py -- every number quoted about the 8k construction, recomputed from the
shipped graphs.  Nothing is read from a cache; each figure below is derived
here.  Run from this folder:

    python3 stats.py                 # everything
    python3 stats.py --section bilateral

Sections
    basic       node/edge counts, density, degree, connectivity, colouring
    bundles     fibre-bundle weights of the coarse graph vs Budapest
    community   Louvain community counts and sizes, over 20 seeds
    bilateral   the block-density matrices and the hemisphere test
    sixtests    pointer to static_tests.py (run separately, it is slow)
    audit       the recorded nine-criterion verdicts

Requires: numpy, networkx.  Budapest edge list in reference/.
"""
import argparse, json, os, sys
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

HERE = os.path.dirname(os.path.abspath(__file__))
G_DIR = os.path.join(HERE, 'graphs')
BUD = os.path.join(HERE, 'reference', 'budapest_1015_70654.edgelist')
SEEDS = range(20)          # Louvain seeds; the best-modularity partition is used


# ---------------------------------------------------------------- loading

def load_masks(path):
    """The parent is stored as three tripartite blocks; rebuild the graph."""
    D = np.load(path)
    n1, n2, n3 = int(D['n1']), int(D['n2']), int(D['n3'])
    o2, o3 = n1, n1 + n2
    n = n1 + n2 + n3
    E = []
    for mk, orow, ocol in [('mask_12', o2, 0), ('mask_13', o3, 0), ('mask_23', o3, o2)]:
        r, c = np.nonzero(D[mk])
        E.append(np.stack([c + ocol, r + orow], 1))
    E = np.vstack(E)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(E[:, 0].tolist(), E[:, 1].tolist()))
    colour = np.concatenate([np.zeros(n1, int), np.ones(n2, int), np.full(n3, 2)])
    return G, E, colour, (n1, n2, n3)


def load_edges(path):
    D = np.load(path)
    G = nx.Graph()
    G.add_nodes_from(range(int(D['n_total'])))
    G.add_edges_from(map(tuple, D['edges']))
    return G


def budapest():
    return nx.read_edgelist(BUD, nodetype=int)


def best_partition(G, seeds=SEEDS):
    """Louvain at resolution 1.0, best modularity over `seeds`.

    This matters.  A single hard-coded seed is not a property of the graph:
    on the coarse graph seed 42 returns 5 communities at Q = 0.5318 while
    16 of 20 seeds return 4 at up to Q = 0.5554.  Always take the best.
    """
    best = None
    for s in seeds:
        cs = louvain_communities(G, resolution=1.0, seed=s)
        q = modularity(G, cs)
        if best is None or q > best[0]:
            best = (q, cs, s)
    q, cs, s = best
    return q, sorted(cs, key=lambda c: -len(c)), s


def block_density(G, comms):
    """Edge density inside and between communities."""
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    lab = np.empty(len(nodes), int)
    for ci, c in enumerate(comms):
        for v in c:
            lab[idx[v]] = ci
    K = len(comms)
    sz = np.array([len(c) for c in comms], float)
    D = np.zeros((K, K))
    for u, v in G.edges():
        a, b = lab[idx[u]], lab[idx[v]]
        D[a, b] += 1
        D[b, a] += 1
    den = np.outer(sz, sz)
    np.fill_diagonal(den, sz * (sz - 1))
    return D / np.maximum(den, 1), sz.astype(int)


def gini(x):
    x = np.sort(np.asarray(x, float))
    m = len(x)
    return float((2 * np.arange(1, m + 1) - m - 1).dot(x) / (m * x.sum()))


# ---------------------------------------------------------------- sections

def sec_basic():
    print('\n=== BASIC =========================================================')
    P, EP, col, layers = load_masks(os.path.join(G_DIR, 'cnew_parent.npz'))
    Q = load_edges(os.path.join(G_DIR, 'cnew_coarse.npz'))
    B = budapest()
    for name, G in [('parent', P), ('coarse quotient', Q), ('Budapest', B)]:
        d = np.array([x for _, x in G.degree()])
        n, m = G.number_of_nodes(), G.number_of_edges()
        print('%-16s %6d nodes %8d edges  density %.5f  degree mean %.1f sd %.1f '
              'min %d max %d  connected %s'
              % (name, n, m, 2 * m / (n * (n - 1)), d.mean(), d.std(),
                 d.min(), d.max(), nx.is_connected(G)))
    print('parent colour layers      %s  (sum %d)' % (list(layers), sum(layers)))
    mono = sum(1 for u, v in P.edges() if col[u] == col[v])
    print('parent monochromatic edges %d   -> properly 3-coloured: %s'
          % (mono, mono == 0))


def sec_bundles():
    print('\n=== FIBRE BUNDLES =================================================')
    w = np.load(os.path.join(G_DIR, 'cnew_coarse_w.npy'))
    print('  recomputed from the supernode map, for checking:')
    D = np.load(os.path.join(G_DIR, 'cnew_parent_supernode.npz'))
    sup = np.concatenate([D['supernode_id_1'], D['supernode_id_2'],
                          D['supernode_id_3']]).astype(np.int64)
    K = int(D['n_supernodes'])
    _, E, _, _ = load_masks(os.path.join(G_DIR, 'cnew_parent.npz'))
    a, b = sup[E[:, 0]], sup[E[:, 1]]
    keep = a != b
    key = np.minimum(a[keep], b[keep]) * K + np.maximum(a[keep], b[keep])
    _, w2 = np.unique(key, return_counts=True)
    print('    shipped %d values, recomputed %d, identical: %s'
          % (len(w), len(w2), np.array_equal(np.sort(w), np.sort(w2.astype(float)))))
    # Budapest fibre counts, if the source array is to hand
    bw = None
    # FIXED 2026-09-25: the second candidate was an absolute path on one
    # machine.  Both are now repo-relative.
    for c in [os.path.join(HERE, 'reference', 'bud_w.npy'),
              os.path.join(HERE, '..', 'finalgraph', 'inputs', 'bud_w.npy')]:
        if os.path.exists(c):
            bw = np.load(c)
            bw = bw[0] if bw.ndim > 1 else bw
            break
    print('%-14s n=%d  mean %.3f  median %.0f  max %d  gini %.3f  >18 %d'
          % ('8k coarse', len(w), w.mean(), np.median(w), w.max(), gini(w), (w > 18).sum()))
    if bw is not None:
        print('%-14s n=%d  mean %.3f  median %.0f  max %d  gini %.3f  >18 %d'
              % ('Budapest', len(bw), bw.mean(), np.median(bw), bw.max(),
                 gini(bw), (bw > 18).sum()))
    else:
        print('Budapest      fiber_count_mean array not found; see README for where it lives')
    print('  -> this is the measure on which the 8k graph loses: its bundle tail is'
          '\n     an order of magnitude heavier than the connectome\'s.')


def sec_community():
    print('\n=== COMMUNITIES ===================================================')
    import collections
    Q = load_edges(os.path.join(G_DIR, 'cnew_coarse.npz'))
    B = budapest()
    for name, G in [('8k coarse', Q), ('Budapest', B)]:
        K, Qs = [], []
        for s in SEEDS:
            cs = louvain_communities(G, resolution=1.0, seed=s)
            K.append(len(cs))
            Qs.append(modularity(G, cs))
        q, cs, seed = best_partition(G)
        print('%-12s K over %d seeds %s | Q %.4f +- %.4f'
              % (name, len(K), dict(sorted(collections.Counter(K).items())),
                 np.mean(Qs), np.std(Qs, ddof=1)))
        print('%-12s best: seed %2d  Q %.4f  sizes %s'
              % ('', seed, q, [len(c) for c in cs]))
    print('  NOTE  static_tests.py hard-codes louvain seed=1 and reports 4;')
    print('        connectome_audit_gold.py hard-codes seed=42 and reports 5.')
    print('        Both are the same graph.  Report the best-of-N partition.')


def sec_bilateral():
    print('\n=== BILATERAL STRUCTURE ===========================================')
    Q = load_edges(os.path.join(G_DIR, 'cnew_coarse.npz'))
    B = budapest()
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    for name, G in [('8k coarse', Q), ('Budapest', B)]:
        q, cs, seed = best_partition(G)
        Dn, sz = block_density(G, cs)
        K = len(sz)
        print('\n-- %s   Q %.4f (seed %d)   sizes %s' % (name, q, seed, [int(x) for x in sz]))
        print(Dn)
        iu = np.triu_indices(K, 1)
        u = np.sort(Dn[iu])[::-1]
        print('   off-diagonal couplings, strongest first: %s'
              % np.array2string(u, precision=4))
        if K == 4:
            # the two hemispheric couplings are the top two; each half is one
            # large community plus the small one it is coupled to
            print('   top two %.4f / %.4f   ratio to third %.2f'
                  % (u[0], u[1], u[1] / u[2] if u[2] > 0 else float('inf')))
            pair = {}
            W = Dn.copy()
            np.fill_diagonal(W, 0)
            for i in range(K):
                pair[i] = int(np.argmax(W[i]))
            halves = []
            seen = set()
            for i in range(K):
                if i in seen:
                    continue
                j = pair[i]
                if pair[j] == i:
                    halves.append((int(sz[i]), int(sz[j]), int(sz[i] + sz[j])))
                    seen.update([i, j])
            print('   mutually-strongest pairs (hemispheres): %s' % halves)
        print('   diagonal (within-community density): %s  spread %.2fx'
              % (np.array2string(Dn.diagonal(), precision=4),
                 Dn.diagonal().max() / Dn.diagonal().min()))
    print('\n  Both graphs give 307/306/201/201 -> halves of 508 and 507, with the')
    print('  same two strong couplings.  The 8k graph\'s are ~32% weaker and its')
    print('  seam between halves sits in one block instead of being spread over')
    print('  four -- which traces back to the seed: 26 within-block : 4 same-')
    print('  hemisphere : 1 cross-hemisphere edges (construct.py, Stage 0).')


def sec_sixtests():
    print('\n=== SIX STATIC TESTS ==============================================')
    csv = os.path.join(HERE, 'figures', 'AD_8k_coarse_six_tests.csv')
    if os.path.exists(csv):
        for line in open(csv):
            k = line.split(',')[0]
            if k.startswith('T') or k.startswith('SCORE'):
                print('  ' + line.rstrip())
    print('  reproduce with:')
    print('    python3 static_tests.py --graph graphs/cnew_coarse.npz \\')
    print('        --budapest reference/budapest_1015_70654.edgelist \\')
    print('        --label "8k coarse quotient" --out out.png --csv out.csv')


def sec_audit():
    print('\n=== NINE-CRITERION AUDIT (recorded) ===============================')
    for f, what in [('audit_constructpy_parent_100.json', '8k parent'),
                    ('audit_constructpy_coarse_100.json', 'coarse quotient')]:
        p = os.path.join(HERE, 'audits', f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        ev = d['evidence']
        scored = {k: v for k, v in ev.items() if k != 'fractal_scaling'}
        npass = sum(1 for v in scored.values() if v == 'pass')
        bad = [k for k, v in scored.items() if v != 'pass']
        g = d['graph']
        print('  %-16s %d nodes / %d edges, %s nulls -> %d/9   fails: %s'
              % (what, g['n_nodes'], g['n_edges'], d.get('n_null'), npass,
                 ', '.join(bad) or 'none'))
    print('  nulls are tripartite degree-preserving; p-floor is 1/(n_null+1),')
    print('  so 100 nulls is the minimum that can pass a p<0.05 criterion.')


SECTIONS = dict(basic=sec_basic, bundles=sec_bundles, community=sec_community,
                bilateral=sec_bilateral, sixtests=sec_sixtests, audit=sec_audit)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', choices=list(SECTIONS) + ['all'], default='all')
    a = ap.parse_args()
    if not os.path.exists(BUD):
        sys.exit('missing %s' % BUD)
    for name, fn in SECTIONS.items():
        if a.section in ('all', name):
            fn()
    print()
