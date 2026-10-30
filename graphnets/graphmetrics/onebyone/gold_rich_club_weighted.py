#!/usr/bin/env python3
"""
gold_rich_club_weighted.py -- weighted (and binary) rich club with BH-FDR and the
Riedel et al. (2022) regime rule, on the 1,015-node quotient or the Budapest copy.

Put this file next to gold_rich_club.py and gold_rich_club_fdr.py
(graphnets/graphmetrics/onebyone/ on the richclub-fdr-literature-test branch).
Run export_graphs.py in graphnets/construction/bilateral first, so
bilateral/graphs/ exists.

    python3 gold_rich_club_weighted.py --self-test
    python3 gold_rich_club_weighted.py --graph quotient --json wrc_quotient.json
    python3 gold_rich_club_weighted.py --graph quotient --weights axons --json wrc_quotient_axons.json
    python3 gold_rich_club_weighted.py --graph budapest --json wrc_budapest.json

Weights (--weights runs a subset; binary phi is always added)
  quotient  raw              = bundle weight, cnew_coarse_w.npy (parent edges between two supernodes)
            size_normalized  = raw / (|A| |B|), supernode sizes from cnew_parent_supernode.npz
            axons            = the repo's axon law (bundles.py), f(W) = round(W^0.05 (1 + (W-1)/11)^0.95),
                               min 1, applied to each colour-pair bundle and summed per quotient edge;
                               equal to bundles.load_bundles() + n_axons summed per supernode pair
  budapest  fibre_count_mean = bud_w.npy row 0, in budapest_connectome.gml edge order

Coefficient (Opsahl et al. 2008; BCT rich_club_wu)
  phi_w(k) = W_{>k} / (sum of the E_{>k} largest weights in the whole graph),
  where E_{>k}, W_{>k} = number and total weight of edges among nodes of degree > k.
  Binary phi(k) = 2 E_{>k} / (N_{>k} (N_{>k} - 1)) is computed on the same nulls.

Nulls
  Maslov-Sneppen double-edge swaps: 10 m successful swaps, at most 25 times as
  many attempts (the swap of gold_rich_club._generate_null_edgelist).  Each
  weight stays with its edge (as BCT randmio_und).  Null s uses seed
  seed + 1000 + s.
  --null-mode audit (default): up to 8 rewirings per seed from one seeded
      stream, the first connected one kept, else the one with the largest
      component (as gold_rich_club._generate_null).
  --null-mode unconstrained: the first rewiring, kept as drawn.
  n_first_draw_disconnected counts seeds whose first rewiring is disconnected;
  when it is 0 the two modes give the same nulls.  n_null_disconnected counts
  the kept nulls that are disconnected.
  These are not the nulls of the gold_rich_club_fdr runs, which take seeds
  seed + 1000 + i for i < n_null and then seed + 5000 + j.

Statistics (gold_rich_club_fdr.add_fdr_results, unchanged)
  Tested: every k in the real curve and in every null curve, null mean > 0.
  rho = phi / mean(phi_null);  p = (#{phi_null >= phi} + 1) / (M + 1);
  Benjamini-Hochberg over all tested k; significant if adjusted p < alpha and
  rho > 1; regime by the Riedel et al. (2022) rule; p resolution reported.
  rho_max_regime is the largest rho inside the regime; rho_max_all_k is the
  largest over all tested k, significant or not.

Output
  --json must name a file that does not exist yet; results are never overwritten.
"""
import argparse, json, os, re, sys, time
import multiprocessing as mp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gold_rich_club_fdr                         # noqa: E402
from gold_rich_club_fdr import add_fdr_results    # noqa: E402


# ── weights ──────────────────────────────────────────────────────────────────

def n_axons(W, p=0.05, c=11.0):
    """The repo's axon law (bundles.py n_axons / BundleSparseLinear._n_axons)."""
    W = np.asarray(W, dtype=float)
    return np.clip(np.round(W**p * (1.0 + (W - 1.0) / c)**(1.0 - p)), 1.0, None)


def axon_weights(d, E, n):
    """Per quotient edge: f(W) summed over its colour-pair bundles.  A bundle is
    (ordered supernode pair, colour pair), as in bundles.py load_bundles();
    bundles inside one supernode are dropped."""
    sid = [d['supernode_id_1'], d['supernode_id_2'], d['supernode_id_3']]
    out = np.zeros(n * n)
    for k, (i, j) in (('mask_12', (0, 1)), ('mask_13', (0, 2)), ('mask_23', (1, 2))):
        row, col = np.nonzero(d[k])                     # row: colour j, col: colour i
        key = sid[i][col].astype(np.int64) * n + sid[j][row]
        uk, cnt = np.unique(key, return_counts=True)
        a, b = uk // n, uk % n
        keep = a != b
        np.add.at(out, np.minimum(a, b)[keep] * n + np.maximum(a, b)[keep], n_axons(cnt[keep]))
    lo, hi = np.minimum(E[:, 0], E[:, 1]), np.maximum(E[:, 0], E[:, 1])
    ax = out[lo * n + hi]
    assert (ax > 0).all() and np.count_nonzero(out) == len(E)
    return ax


def load(graph, root):
    if graph == 'quotient':
        g = os.path.join(root, 'construction', 'bilateral', 'graphs')
        d = np.load(os.path.join(g, 'cnew_coarse.npz'))
        E, n = d['edges'].astype(np.int64), int(d['n_total'])
        w = np.load(os.path.join(g, 'cnew_coarse_w.npy')).astype(float)
        s = np.load(os.path.join(g, 'cnew_parent_supernode.npz'))
        sid = np.concatenate([s['supernode_id_1'], s['supernode_id_2'], s['supernode_id_3']])
        size = np.bincount(sid, minlength=n).astype(float)
        assert len(E) == len(w) and size.sum() == len(sid) and (size > 0).all()
        return E, n, {'raw': w, 'size_normalized': w / (size[E[:, 0]] * size[E[:, 1]]),
                      'axons': axon_weights(s, E, n)}
    if graph == 'budapest':
        with open(os.path.join(root, 'Budapest', 'budapest_connectome.gml')) as fh:
            t = fh.read()
        E = np.array(re.findall(r'source (\d+)\s+target (\d+)', t), dtype=np.int64)
        w = np.load(os.path.join(root, 'construction', 'bilateral', 'reference', 'bud_w.npy'))[0]
        assert len(E) == len(w) == 70654
        return E, 1015, {'fibre_count_mean': w.astype(float)}
    raise ValueError(graph)


# ── coefficients ─────────────────────────────────────────────────────────────

def phi_w(E, w, n):
    deg = np.bincount(E.ravel(), minlength=n)
    mind = np.minimum(deg[E[:, 0]], deg[E[:, 1]])
    top = np.cumsum(np.sort(w)[::-1])                 # sum of the r largest weights
    o = np.argsort(mind, kind='stable'); md = mind[o]
    suff = np.cumsum(w[o][::-1])[::-1]                # weight of edges with min degree >= md[i]
    res = {}
    for k in range(deg.max() + 1):
        if (deg > k).sum() < 2:
            break
        i = np.searchsorted(md, k, side='right'); ne = len(md) - i
        res[k] = 0.0 if ne == 0 else float(suff[i] / top[ne - 1])
    return res


def phi_b(E, n):
    deg = np.bincount(E.ravel(), minlength=n)
    md = np.sort(np.minimum(deg[E[:, 0]], deg[E[:, 1]]))
    res = {}
    for k in range(deg.max() + 1):
        N = (deg > k).sum()
        if N < 2:
            break
        e = len(md) - np.searchsorted(md, k, side='right')
        res[k] = float(2 * e / (N * (N - 1)))
    return res


def curves(E, ws, n):
    out = {name: phi_w(E, w, n) for name, w in ws.items()}
    out['binary'] = phi_b(E, n)
    return out


# ── nulls ────────────────────────────────────────────────────────────────────

def swap(E, n, rng):
    """One rewiring of E (a copy): the swap of gold_rich_club._generate_null_edgelist.
    Row i keeps weight i, so each weight moves with its edge."""
    E = E.copy(); m = len(E)
    adj = [set() for _ in range(n)]
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    target = max(10, 10 * m); tries = succ = 0
    while succ < target and tries < 25 * target:
        tries += 1
        i = int(rng.integers(m)); j = int(rng.integers(m))
        if i == j:
            continue
        u1, v1 = E[i]; u2, v2 = E[j]
        if len({u1, v1, u2, v2}) < 4:
            continue
        if rng.random() < 0.5:
            a1, b1, a2, b2 = u1, u2, v1, v2
        else:
            a1, b1, a2, b2 = u1, v2, v1, u2
        if b1 in adj[a1] or b2 in adj[a2]:
            continue
        adj[u1].discard(v1); adj[v1].discard(u1); adj[u2].discard(v2); adj[v2].discard(u2)
        adj[a1].add(b1); adj[b1].add(a1); adj[a2].add(b2); adj[b2].add(a2)
        E[i] = (a1, b1); E[j] = (a2, b2); succ += 1
    return E


def components(E, n):
    """(connected?, size of the largest component), counting all n nodes."""
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    return nc == 1, int(np.bincount(lab).max())


def make_null(E, n, seed, retries):
    """As gold_rich_club._generate_null: up to `retries` rewirings from one seeded
    stream; the first connected one, else the one with the largest component.
    Returns (edges, first draw connected?, kept null connected?)."""
    rng = np.random.default_rng(seed)
    best, best_gcc, first_ok = None, -1, None
    for t in range(retries):
        En = swap(E, n, rng)
        ok, gcc = components(En, n)
        if t == 0:
            first_ok = ok
        if ok:
            return En, first_ok, True
        if gcc > best_gcc:
            best, best_gcc = En, gcc
    return best, first_ok, False


_JOB = None   # (E, ws, n, retries); set before the Pool starts, inherited through fork


def _null_worker(seed):
    E, ws, n, retries = _JOB
    En, first_ok, ok = make_null(E, n, seed, retries)
    return curves(En, ws, n), first_ok, ok


# ── statistics ───────────────────────────────────────────────────────────────

def analyse(real, nulls, deg, info, alpha):
    common = set(real)
    for c in nulls:
        common &= set(c)
    rows = []
    for k in sorted(common):
        nv = np.array([c[k] for c in nulls])
        mu = float(nv.mean())
        if mu <= 0:
            continue
        rows.append(dict(k=int(k), n_above_k=int((deg > k).sum()), phi=float(real[k]),
                         phi_null_mean=mu,
                         phi_null_std=float(nv.std(ddof=1)) if len(nv) > 1 else 0.0,
                         rho=float(real[k] / mu),
                         p=float((np.sum(nv >= real[k]) + 1) / (len(nv) + 1))))
    res = dict(per_k=rows, **info)
    add_fdr_results(res, alpha=alpha)
    if rows:
        top = max(rows, key=lambda r: r['rho'])
        res.update(rho_max_all_k=top['rho'], k_rho_max_all_k=top['k'])
    reg = res['fdr_bh']['riedel2022_regime']
    if reg:
        inside = [r for r in rows if reg['k_lower'] <= r['k'] <= reg['k_upper']]
        top = max(inside, key=lambda r: r['rho'])
        res.update(rho_max_regime=top['rho'], k_rho_max_regime=top['k'])
    return res


def report(name, res):
    f = res['fdr_bh']; reg = f['riedel2022_regime']
    print(f"{name:16s} FDR-significant {f['n_fdr_significant']}/{f['n_tests']}", end='')
    if reg:
        print(f"  regime k {reg['k_lower']}..{reg['k_upper']}, nodes {reg['n_above_k_lower']} -> "
              f"{reg['n_above_k_upper']}, {reg['n_distinct_node_sets']} node sets, "
              f"{reg['n_nonsignificant_inside']} non-significant inside, "
              f"rho max {res['rho_max_regime']:.3f} at k {res['k_rho_max_regime']}")
    else:
        print('  no regime')
    print(f"{'':16s} {f['regime_verdict']} ({f['regime_rule']}); "
          f"rho max over all k {res['rho_max_all_k']:.3f} at k {res['k_rho_max_all_k']}; "
          f"p resolution {f['p_resolution']:.6f}, {f['n_k_at_p_resolution']} k there, "
          f"BH needs {f['n_k_at_p_resolution_needed']}", flush=True)


# ── self-test ────────────────────────────────────────────────────────────────

def _self_test():
    import networkx as nx
    gold_rich_club_fdr._self_test()                    # BH, regime rule, p resolution
    n = 80
    G = nx.gnm_random_graph(n, 600, seed=1)
    E = np.array(G.edges(), dtype=np.int64)
    w = np.random.default_rng(0).lognormal(0.0, 1.0, len(E))
    # binary phi against networkx
    ref, got = nx.rich_club_coefficient(G, normalized=False), phi_b(E, n)
    assert set(ref) == set(got) and all(abs(ref[k] - got[k]) < 1e-12 for k in ref)
    # weighted phi against its definition, on the dense matrix
    A = np.zeros((n, n)); A[E[:, 0], E[:, 1]] = w; A[E[:, 1], E[:, 0]] = w
    deg, top = (A > 0).sum(0), np.sort(w)[::-1]
    got = phi_w(E, w, n)
    assert set(got) == set(ref)
    for k, v in got.items():
        sub = A[np.ix_(deg > k, deg > k)]
        e = int((sub > 0).sum()) // 2
        want = 0.0 if e == 0 else sub.sum() / 2 / top[:e].sum()
        assert abs(v - want) < 1e-12, (k, v, want)
    # the swap keeps every degree and a simple graph, is seeded, and moves most edges
    E2 = swap(E, n, np.random.default_rng(5))
    assert (np.bincount(E2.ravel(), minlength=n) == np.bincount(E.ravel(), minlength=n)).all()
    assert (E2[:, 0] != E2[:, 1]).all() and len({frozenset(e) for e in E2.tolist()}) == len(E2)
    assert (swap(E, n, np.random.default_rng(5)) == E2).all()
    assert len({frozenset(e) for e in E2.tolist()} & {frozenset(e) for e in E.tolist()}) < len(E) // 2
    # audit retries: two 4-cycles; any rewiring is a union of cycles, connected iff one 8-cycle
    C = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]])
    redrawn = 0
    for s in range(40):
        _, f1, ok1 = make_null(C, 8, s, 1)
        _, f8, ok8 = make_null(C, 8, s, 8)
        assert f1 == f8 == ok1                          # same first draw in both modes
        assert ok8 or not f8
        redrawn += (not f8) and ok8
    assert redrawn > 0
    # the axon law: values of bundles.n_axons
    assert n_axons([1, 2, 5, 6, 10, 20, 50, 100, 200, 978]).tolist() == \
        [1, 1, 1, 2, 2, 3, 6, 11, 21, 101]
    # axon weights: supernodes A=0, B=1, C=2.  Six colour-1 nodes of A to one
    # colour-2 node of B (W=6, f=2); one colour-1 node of B to a colour-2 node
    # of A (a separate ordered bundle, W=1, f=1); one 1-3 edge A-C (f=1); one
    # edge inside A (dropped).
    d = {'supernode_id_1': np.array([0, 0, 0, 0, 0, 0, 1]),
         'supernode_id_2': np.array([1, 0]), 'supernode_id_3': np.array([2])}
    m12 = np.zeros((2, 7), int); m12[0, :6] = 1; m12[1, 6] = 1; m12[1, 0] = 1
    m13 = np.zeros((1, 7), int); m13[0, 0] = 1
    d.update(mask_12=m12, mask_13=m13, mask_23=np.zeros((1, 2), int))
    assert axon_weights(d, np.array([[0, 1], [0, 2]]), 3).tolist() == [3.0, 1.0]
    print('Weighted rich-club self-tests passed.')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global _JOB
    ap = argparse.ArgumentParser(description='Weighted and binary rich club with BH-FDR.')
    ap.add_argument('--graph', choices=['quotient', 'budapest'])
    ap.add_argument('--root', default=os.path.abspath(os.path.join(HERE, '..', '..')),
                    help='the graphnets/ directory (default: two levels up)')
    ap.add_argument('--n-null', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--null-mode', choices=['audit', 'unconstrained'], default='audit')
    ap.add_argument('--weights', default=None,
                    help='comma list to run a subset, e.g. axons (default: all)')
    ap.add_argument('--json', default=None, help='output path; must not exist')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()

    if a.self_test:
        _self_test()
        return
    if not a.graph:
        ap.error('--graph is required unless --self-test is used')
    if a.n_null < 1:
        ap.error('--n-null must be at least 1')
    if a.json and os.path.exists(a.json):
        ap.error(f'{a.json} exists; this script never overwrites a result file')

    t = time.time()
    E, n, ws = load(a.graph, a.root)
    if a.weights:
        names = a.weights.split(',')
        bad = [k for k in names if k not in ws]
        if bad:
            ap.error(f'unknown weights {bad}; {a.graph} has {list(ws)}')
        ws = {k: ws[k] for k in names}
    deg = np.bincount(E.ravel(), minlength=n)
    print(f'{a.graph}: {n} nodes, {len(E)} edges, max degree {deg.max()}, '
          f'weights {list(ws)}, {a.n_null} nulls ({a.null_mode})', flush=True)
    real = curves(E, ws, n)
    _JOB = (E, ws, n, 8 if a.null_mode == 'audit' else 1)
    nulls, first_bad, kept_bad = [], 0, 0
    seeds = [a.seed + 1000 + s for s in range(a.n_null)]
    with mp.get_context('fork').Pool(a.workers) as pool:
        for i, (c, first_ok, ok) in enumerate(pool.imap(_null_worker, seeds)):
            nulls.append(c); first_bad += not first_ok; kept_bad += not ok
            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{a.n_null} nulls  {time.time() - t:.0f}s', flush=True)
    print(f'  first rewiring disconnected for {first_bad} seeds; '
          f'{kept_bad} kept nulls disconnected', flush=True)
    info = dict(n_curves=a.n_null, seed=a.seed, null_mode=a.null_mode,
                null_model_type='Maslov-Sneppen double-edge swaps, weights kept on their edges',
                n_null_disconnected=kept_bad)
    out = dict(graph=a.graph, n_nodes=n, n_edges=len(E), n_null=a.n_null, seed=a.seed,
               alpha=a.alpha, null_mode=a.null_mode,
               n_first_draw_disconnected=first_bad, n_null_disconnected=kept_bad)
    for name in real:
        out[name] = analyse(real[name], [c[name] for c in nulls], deg, info, a.alpha)
        report(name, out[name])
    if a.json:
        with open(a.json, 'x') as fh:
            json.dump(out, fh, indent=1,
                      default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        print(f'Wrote {a.json}')


if __name__ == '__main__':
    main()
