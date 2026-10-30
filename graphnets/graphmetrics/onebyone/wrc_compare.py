#!/usr/bin/env python3
"""
wrc_compare.py -- summary tables for the gold_rich_club_weighted.py runs on the
quotient and on Budapest.

Put it next to gold_rich_club_weighted.py (it uses that script's load() and
curves()).

    python3 wrc_compare.py --quotient wrc_quotient_1000_v2.json \
                           --budapest wrc_budapest_1000_v2.json

Prints
  1. the runs: nulls, seeds, null mode, disconnected nulls, p resolution, and
     whether the real curves recomputed from the graphs under --root equal the
     ones in the JSONs (if they do, the graphs are the ones the JSONs came from);
  2. per weighting: FDR-significant k, the Riedel et al. (2022) regime, and the
     largest rho inside the regime and over all tested k, with the club size;
  3. rho at matched club sizes (the k whose club size is closest to N);
  4. the clubs of 2 nodes: one edge, and the spread of its null value;
  5. where the heavy edges sit (read from the graphs):
     hubs   = nodes with degree >= d*, d* = degree of the ceil(0.1 n)-th
              highest-degree node (ties at d* included);
     heavy  = edges with weight >= w*, w* = the ceil(0.01 m)-th largest weight
              (ties at w* included);
     share of hub-hub edges among all edges, among heavy edges, and of the
     total weight; Spearman correlation of weight with deg(u) * deg(v);
  6. quotient supernode sizes against degree and weight.
"""
import argparse, json, math, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gold_rich_club_weighted import load, curves   # noqa: E402

SIZES = (500, 250, 150, 100, 50, 20)


def weightings(d):
    return [k for k, v in d.items() if isinstance(v, dict) and 'per_k' in v]


def regime(res):
    return res['fdr_bh']['riedel2022_regime']


def at_size(res, N):
    return min(res['per_k'], key=lambda r: (abs(r['n_above_k'] - N), r['k']))


def mark(res, r):
    """* significant inside the regime, + significant outside it, blank otherwise."""
    if not r['significant_fdr_bh']:
        return ' '
    reg = regime(res)
    return '*' if reg and reg['k_lower'] <= r['k'] <= reg['k_upper'] else '+'


def section_runs(runs, graphs):
    print('== 1. Runs ==')
    for g, d in runs:
        f = d[weightings(d)[0]]['fdr_bh']
        print(f"{g}: {d['n_nodes']} nodes, {d['n_edges']} edges; {d['n_null']} nulls, "
              f"seeds {d['seed'] + 1000}..{d['seed'] + 999 + d['n_null']}, null mode {d['null_mode']}; "
              f"first rewiring disconnected {d['n_first_draw_disconnected']}, "
              f"kept nulls disconnected {d['n_null_disconnected']}")
        print(f"  {f['n_tests']} k tested; p resolution {f['p_resolution']:.6f}; "
              f"BH at q = {f['q']} needs {f['n_k_at_p_resolution_needed']} k at that p")
        E, n, ws = graphs[g]
        real = curves(E, ws, n)
        for w in weightings(d):
            diff = max(abs(r['phi'] - real[w][r['k']]) for r in d[w]['per_k'])
            print(f"  real curve recomputed from the graph, {w}: "
                  f"{'equals the JSON' if diff < 1e-12 else 'DIFFERS from the JSON'} "
                  f"at all {len(d[w]['per_k'])} k (max difference {diff:.1e})")
    print()


def section_regimes(runs):
    print('== 2. Rich club per weighting ==')
    print(f"{'graph':9s} {'weighting':17s} {'FDR-sig k':10s} {'regime k':9s} {'nodes':12s} "
          f"{'sets':>4s}  {'largest rho in regime':26s} largest rho over all tested k")
    for g, d in runs:
        for w in weightings(d):
            res = d[w]; f = res['fdr_bh']; reg = regime(res); byk = {r['k']: r for r in res['per_k']}
            ka, kr = res['k_rho_max_all_k'], res.get('k_rho_max_regime')
            ra = byk[ka]
            sig_all = 'FDR-sig' if ra['significant_fdr_bh'] else 'not sig'
            if reg:
                rr = byk[kr]
                in_reg = f"{res['rho_max_regime']:.3f} (k {kr}, {rr['n_above_k']} nodes)"
                print(f"{g:9s} {w:17s} {f['n_fdr_significant']:>3d}/{f['n_tests']:<6d} "
                      f"{reg['k_lower']:>3d}-{reg['k_upper']:<5d} "
                      f"{reg['n_above_k_lower']:>4d} -> {reg['n_above_k_upper']:<4d} "
                      f"{reg['n_distinct_node_sets']:>4d}  {in_reg:26s} "
                      f"{res['rho_max_all_k']:.3f} (k {ka}, {ra['n_above_k']} nodes, {sig_all})")
            else:
                print(f"{g:9s} {w:17s} {f['n_fdr_significant']:>3d}/{f['n_tests']:<6d} no regime")
        for w in weightings(d):
            f = d[w]['fdr_bh']
            ranges = ', '.join(f"{r['k_first']}-{r['k_last']}" for r in f['fdr_significant_ranges'])
            print(f"{'':9s} {w:17s} significant k: {ranges}")
            print(f"{'':9s} {'':17s} regime rule: {f['regime_rule']}")
    print()


def section_sizes(runs):
    print('== 3. rho at matched club sizes ==')
    print('   (the k whose club size is closest to N; * FDR-significant inside the regime,')
    print('    + FDR-significant outside it, no mark = not significant)')
    head = ''.join(f'{"N=" + str(N):>12s}' for N in SIZES)
    print(f"{'':27s}{head}")
    for g, d in runs:
        ws = weightings(d)
        cells = []
        for N in SIZES:
            r = at_size(d[ws[0]], N)
            cells.append(f"k {r['k']} ({r['n_above_k']})")
        print(f"{g:9s} {'k (nodes)':17s} " + ''.join(f'{c:>12s}' for c in cells))
        for w in ws:
            res = d[w]
            cells = []
            for N in SIZES:
                r = at_size(res, N)
                cells.append(f"{r['rho']:.3f}{mark(res, r)}")
            print(f"{g:9s} {w:17s} " + ''.join(f'{c:>12s}' for c in cells))
    print()


def section_two_node(runs):
    print('== 4. Clubs of 2 nodes (one edge) ==')
    for g, d in runs:
        M = d['n_null']
        for w in weightings(d):
            two = [r for r in d[w]['per_k'] if r['n_above_k'] == 2]
            if not two:
                continue
            r = two[0]
            se = r['phi_null_std'] / math.sqrt(M) / r['phi_null_mean']
            print(f"{g:9s} {w:17s} k {two[0]['k']}..{two[-1]['k']}: phi {r['phi']:.4g}, "
                  f"null mean {r['phi_null_mean']:.4g}, null sd {r['phi_null_std']:.4g} "
                  f"({r['phi_null_std'] / r['phi_null_mean']:.2f} x mean); "
                  f"SE of null mean {100 * se:.1f}% -> rho {r['rho']:.3f} +/- {r['rho'] * se:.2f}; "
                  f"p {r['p']:.4f}, {'FDR-sig' if r['significant_fdr_bh'] else 'not sig'}")
    print()


def section_hubs(graphs):
    print('== 5. Where the heavy edges sit ==')
    print(f"{'':39s}{'share that is hub-hub':^41s}".rstrip())
    print(f"{'graph':9s} {'weighting':17s} {'hubs':>4s} {'d*':>4s}  "
          f"{'all edges':>10s} {'heavy edges (n)':>17s} {'total weight':>12s}  Spearman(w, deg*deg)")
    for g, (E, n, ws) in graphs.items():
        m = len(E); deg = np.bincount(E.ravel(), minlength=n)
        dstar = np.sort(deg)[::-1][math.ceil(0.1 * n) - 1]
        hub = deg >= dstar
        hh = hub[E[:, 0]] & hub[E[:, 1]]
        dd = deg[E[:, 0]].astype(float) * deg[E[:, 1]]
        for w, x in ws.items():
            wstar = np.sort(x)[::-1][math.ceil(0.01 * m) - 1]
            heavy = x >= wstar
            heavy_s = f"{100 * hh[heavy].mean():.2f}% ({heavy.sum()})"
            print(f"{g:9s} {w:17s} {hub.sum():>4d} {dstar:>4d}  "
                  f"{100 * hh.mean():>9.2f}% {heavy_s:>17s} {100 * x[hh].sum() / x.sum():>11.2f}%  "
                  f"{spearmanr(x, dd)[0]:>6.3f}")
    print()


def section_supernodes(E, n, ws, root):
    print('== 6. Quotient supernode sizes ==')
    s = np.load(os.path.join(root, 'construction', 'bilateral', 'graphs', 'cnew_parent_supernode.npz'))
    sid = np.concatenate([s['supernode_id_1'], s['supernode_id_2'], s['supernode_id_3']])
    size = np.bincount(sid, minlength=n).astype(float)
    deg = np.bincount(E.ravel(), minlength=n)
    ab = size[E[:, 0]] * size[E[:, 1]]
    dd = deg[E[:, 0]].astype(float) * deg[E[:, 1]]
    print(f"parent vertices per supernode: {size.min():.0f}-{size.max():.0f}, median {np.median(size):.0f}")
    print(f"Spearman over nodes: degree vs size {spearmanr(deg, size)[0]:.3f}")
    print(f"Spearman over edges: |A||B| vs deg*deg {spearmanr(ab, dd)[0]:.3f}; "
          + '; '.join(f"{w} vs |A||B| {spearmanr(ws[w], ab)[0]:.3f}" for w in ws))
    print()


def main():
    ap = argparse.ArgumentParser(description='Summary tables for the gold_rich_club_weighted.py runs.')
    ap.add_argument('--quotient', required=True, help='JSON from --graph quotient')
    ap.add_argument('--budapest', required=True, help='JSON from --graph budapest')
    ap.add_argument('--root', default=os.path.abspath(os.path.join(HERE, '..', '..')),
                    help='the graphnets/ directory (default: two levels up)')
    a = ap.parse_args()
    runs = []
    for g, path in (('quotient', a.quotient), ('budapest', a.budapest)):
        with open(path) as fh:
            d = json.load(fh)
        assert d['graph'] == g, (path, d['graph'])
        runs.append((g, d))
    graphs = {g: load(g, a.root) for g, _ in runs}
    section_runs(runs, graphs)
    section_regimes(runs)
    section_sizes(runs)
    section_two_node(runs)
    section_hubs(graphs)
    section_supernodes(*graphs['quotient'], a.root)


if __name__ == '__main__':
    main()
