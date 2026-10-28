#!/usr/bin/env python3
"""
gold_fractal_scaling.py — the descriptive 'fractal_scaling' entry of
connectome_audit_gold.py, on its own.

MEMB box covering (Song, Gallos, Havlin & Makse 2007, J. Stat. Mech. P03006)
of the graph's GCC, with full all-pairs shortest paths:

    for r_B = 1 .. floor(max_distance / 2):
        l_B   = 2*r_B + MEMB_LB_OFFSET          (MEMB_LB_OFFSET = 0, as in the audit)
        N_B   = _memb_count(dist, r_B)          (maximum-excluded-mass burning)
    fit     log10 N_B = -d_B log10 l_B + a      over points with N_B > 1
            (scipy.stats.linregress; also a 95% CI on d_B and the R^2 of an
             exponential ln N_B = -b l_B + c on the same points)
    verdict = 'pass' if at least 3 points have N_B > 1, else 'fail'
              (descriptive; not one of the nine scored criteria).  The verdict
              does not look at d_B, R^2 or the exponential fit.

Self-contained: needs numpy, scipy and networkx only, and does not import
connectome_audit_gold.py.  The loading, utility and MEMB functions and the
MEMB_LB_OFFSET constant below are copied from it unchanged; _fractal_verdict
is copied unchanged too, nested exactly as in run_audit (it reads `fractal`
from the enclosing function).  With the same --graph the result is the same as
the 'fractal' block of the full audit and its 'fractal_scaling' verdict.  The
audit passes its --seed to _fractal_memb, which does not use it (the MEMB
cover is deterministic: ties go to the lowest node index); --seed is kept so
the call is the same.

Usage:
    python gold_fractal_scaling.py --graph G.npz
    python gold_fractal_scaling.py --graph G.npz --json fractal.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy import stats
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
# Loading  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LayeredGraph:
    G: nx.Graph
    masks: Dict[str, np.ndarray]
    sizes: Tuple[int, int, int]
    has_edge_list: bool = False


def load_layered_graph(path: str) -> LayeredGraph:
    if path.endswith('.gml'):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
        n = G.number_of_nodes()
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
        masks = {k: np.zeros((1, 1), dtype=np.uint8) for k in ['12', '13', '23']}
        print(f"  [Loaded GML: {n} nodes, {G.number_of_edges()} edges]")
        return LayeredGraph(G=G, masks=masks, sizes=(n1, n2, n3), has_edge_list=True)

    data = np.load(path)

    # Check for edge-list format FIRST (χ>3 graphs may lack masks)
    if 'edges' in data:
        n_total = int(data['n_total']) if 'n_total' in data else (
            int(data['n1']) + int(data['n2']) + int(data['n3']))
        edges = data['edges']
        G = nx.Graph()
        G.add_nodes_from(range(n_total))
        G.add_edges_from((int(u), int(v)) for u, v in edges)
        print(f"  [Loaded from edge list: {G.number_of_edges()} edges]")
        if 'n1' in data:
            n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        else:
            n1 = n_total // 3
            n2 = n_total // 3
            n3 = n_total - 2 * (n_total // 3)
        masks = {}
        for key in ['mask_12', 'mask_13', 'mask_23']:
            if key in data:
                masks[key[5:]] = (data[key] > 0).astype(np.uint8)
            else:
                masks[key[5:]] = np.zeros((1, 1), dtype=np.uint8)
        return LayeredGraph(G=G, masks=masks,
                            sizes=(n1, n2, n3), has_edge_list=True)

    # Standard mask-based loading (χ=3 tripartite graphs)
    n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
    mask_12 = (data['mask_12'] > 0).astype(np.uint8)
    mask_13 = (data['mask_13'] > 0).astype(np.uint8)
    mask_23 = (data['mask_23'] > 0).astype(np.uint8)
    G = _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23)
    return LayeredGraph(G=G, masks={'12': mask_12, '13': mask_13, '23': mask_23},
                        sizes=(n1, n2, n3), has_edge_list=False)


def _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23):
    G = nx.Graph()
    off1, off2, off3 = 0, n1, n1 + n2
    G.add_nodes_from(range(n1 + n2 + n3))
    rows, cols = np.where(mask_12)
    G.add_edges_from((off1 + int(c), off2 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_13)
    G.add_edges_from((off1 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_23)
    G.add_edges_from((off2 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Utilities  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _gcc(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


# ═══════════════════════════════════════════════════════════════════════════
# 6. Fractal dimension — MEMB box-covering with full APSP
#    (copied from connectome_audit_gold.py, with its comments)
#    Song, Havlin, Makse 2007 "How to calculate the fractal dimension
#    of a complex network: the box covering algorithm"
#
#    Box definition (Song et al. 2005): a box of size l_B is a set of
#    nodes where ALL pairwise distances are strictly < l_B.
#
#    MEMB algorithm (Song et al. 2007): uses burning radius r_B where
#    l_B = 2*r_B (calibrated; see MEMB_LB_OFFSET below).  At each step:
#    1. Compute "excluded mass" = number of uncovered nodes within
#       distance <= r_B of each uncovered node
#    2. Select the uncovered node with maximum excluded mass as center
#    3. Assign all uncovered nodes within distance <= r_B of center
#    This guarantees pairwise distance <= 2*r_B = l_B - 1 < l_B.
# ═══════════════════════════════════════════════════════════════════════════


# MEMB box-size convention: l_B = 2*r_B + MEMB_LB_OFFSET.
# Song 2007 text says l_B = 2*r_B + 1 (exclusive bound: d < l_B).
# Setting MEMB_LB_OFFSET = 0 (l_B = 2*r_B) gives closer benchmark
# results on finite graphs and is what produced audit_final_v8.json.
MEMB_LB_OFFSET = 0   # 0 → l_B = 2*r_B;  1 → l_B = 2*r_B + 1

def _memb_count(dist: np.ndarray, r_B: int) -> int:
    """True MEMB box count for burning radius r_B.
    Box diameter l_B = 2*r_B + MEMB_LB_OFFSET.
    dist: int16 all-pairs distance matrix (n x n).
    Returns number of boxes.

    Song 2007 MEMB: center candidates are all NON-CENTER nodes,
    including already-covered ones.  Only nodes previously selected
    as centers are excluded from the candidate pool.
    """
    n = dist.shape[0]
    within = dist <= r_B  # n x n boolean: nodes within burning radius
    covered = np.zeros(n, dtype=bool)
    is_center = np.zeros(n, dtype=bool)
    # Initial excluded mass for each node (all nodes uncovered)
    mass = within.sum(axis=1).astype(np.int32)
    nb = 0
    while (~covered & ~is_center).any():
        # Select non-center node with maximum excluded mass
        # (covered non-center nodes ARE eligible — Song 2007 step ii)
        scores = mass.copy()
        scores[is_center] = -1
        center = int(np.argmax(scores))
        if scores[center] <= 0:
            # Remaining uncovered non-center nodes: each gets its own box
            nb += int(np.sum(~covered & ~is_center))
            break
        is_center[center] = True
        # Assign all uncovered nodes within r_B of center
        assignable = within[center] & ~covered
        newly = np.flatnonzero(assignable)
        covered[newly] = True
        nb += 1
        # Update excluded masses (subtract newly covered from everyone)
        if len(newly) > 0:
            delta = within[newly].sum(axis=0).astype(np.int32)
            mass -= delta
            mass[is_center] = 0
    return nb


def _fractal_memb(G: nx.Graph, seed: int = 42) -> Dict[str, object]:
    Gcc = _gcc(G)
    n = Gcc.number_of_nodes()
    if n < 50:
        return {'available': False, 'reason': 'too_small'}

    # Build sparse adjacency
    nodes = sorted(Gcc.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    rs, cs = [], []
    for u, v in Gcc.edges():
        i, j = idx[u], idx[v]
        rs.extend([i, j]); cs.extend([j, i])
    adj = scipy.sparse.csr_matrix(
        (np.ones(len(rs), dtype=np.float32), (rs, cs)), shape=(n, n))

    # Full APSP
    print("  Computing all-pairs shortest paths...")
    t0 = time.time()
    dist = scipy.sparse.csgraph.shortest_path(adj, method='D', directed=False, unweighted=True)
    dist[~np.isfinite(dist)] = 32767
    dist = dist.astype(np.int16)
    print(f"  APSP done ({time.time()-t0:.1f}s, {dist.nbytes / 1e6:.0f} MB)")

    # MEMB uses burning radius r_B; box diameter l_B = 2*r_B + MEMB_LB_OFFSET.
    # See MEMB_LB_OFFSET constant above for convention choice.
    finite = dist[(dist < 32767) & (dist > 0)]
    max_d = int(finite.max()) if len(finite) > 0 else 12
    max_r = max_d // 2

    l_values = []
    box_counts = []
    for r_B in range(1, max_r + 1):
        l_B = 2 * r_B + MEMB_LB_OFFSET
        t1 = time.time()
        nb = _memb_count(dist, r_B)
        l_values.append(l_B)
        box_counts.append(nb)
        print(f"    l_B={l_B:2d} (r_B={r_B}): N_B={nb} ({time.time()-t1:.1f}s)")

    # Log-log fit (exclude floor points where N_B=1)
    valid = [(l, b) for l, b in zip(l_values, box_counts) if b > 1]
    if len(valid) < 3:
        return {'available': False, 'reason': 'insufficient_scaling_range',
                'diameters': l_values, 'box_counts': box_counts}

    ls, bs = zip(*valid)
    ls_arr = np.array(ls, dtype=float)
    bs_arr = np.array(bs, dtype=float)
    log_l = np.log10(ls_arr)
    log_n = np.log10(bs_arr)
    slope, intercept, r_val, _, std_err = stats.linregress(log_l, log_n)

    # 95% CI for d_B
    n_pts = len(ls)
    dof = n_pts - 2
    if dof > 0:
        t_crit = stats.t.ppf(0.975, dof)
        d_B_ci = (float(-slope - t_crit * std_err), float(-slope + t_crit * std_err))
    else:
        d_B_ci = (float('nan'), float('nan'))

    # Exponential comparison: N_B = a * exp(-b * l_B)
    log_n_ln = np.log(bs_arr)
    exp_slope, exp_intercept, exp_r, _, _ = stats.linregress(ls_arr, log_n_ln)
    exp_r2 = float(exp_r ** 2)

    return {
        'available': True,
        'method': 'MEMB_Song2007',
        'l_B_convention': f'2*r_B+{MEMB_LB_OFFSET}',
        'nodes': n,
        'diameters': l_values,
        'box_counts': box_counts,
        'fit_diameters': list(ls),
        'fit_box_counts': list(bs),
        'd_B': float(-slope),
        'r2': float(r_val ** 2),
        'd_B_ci_95': d_B_ci,
        'd_B_std_err': float(std_err),
        'n_fit_points': n_pts,
        'exp_r2': exp_r2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# The descriptive entry
# ═══════════════════════════════════════════════════════════════════════════


def fractal_scaling(layered: LayeredGraph, seed: int = 42) -> Dict[str, object]:
    """As in run_audit: '[8/8]' call of _fractal_memb on layered.G, then the
    nested _fractal_verdict (copied unchanged, including its comments)."""
    G = layered.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Nodes: {n:,}   Edges: {m:,}   GCC: {_gcc(G).number_of_nodes():,} nodes")

    print("[8/8] Fractal dimension (MEMB box-counting)")
    fractal = _fractal_memb(G, seed=seed)

    def _fractal_verdict():
        # Organized box-count decay via MEMB (Song et al. 2005/2007).
        # Binarized structural connectomes show exponential rather than
        # strict power-law decay (Xue & Bogdan 2017), so both forms
        # are consistent with brain networks.
        # Pass if MEMB produces >= 3 valid fit points (minimum for
        # meaningful 2-parameter regression).
        if not fractal.get('available', False):
            return 'fail'
        if fractal.get('n_fit_points', 0) >= 3:
            return 'pass'
        return 'fail'

    return {'graph': {'n_nodes': n, 'n_edges': m},
            'seed': seed,
            'fractal': fractal,
            'fractal_scaling': _fractal_verdict()}


def main():
    ap = argparse.ArgumentParser(description="fractal_scaling, the descriptive MEMB entry of the gold audit")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Passed to _fractal_memb as in the audit (unused there)')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    res = fractal_scaling(layered, seed=args.seed)

    # Printed exactly as the 'FRACTAL DIMENSION (MEMB)' block of print_report.
    frac = res['fractal']
    print("\n" + "=" * 72)
    print("FRACTAL DIMENSION (MEMB)")
    print("=" * 72)
    if frac.get('available'):
        print(f"  {'method':>16s}: {frac['method']}")
        print(f"  {'nodes':>16s}: {frac['nodes']:,}")
        print(f"  {'diameters':>16s}: {frac['diameters']}")
        print(f"  {'box_counts':>16s}: {frac['box_counts']}")
        print(f"  {'d_B':>16s}: {frac['d_B']:.4f}")
        print(f"  {'R²':>16s}: {frac['r2']:.4f}")
        if 'd_B_ci_95' in frac:
            lo, hi = frac['d_B_ci_95']
            print(f"  {'d_B 95% CI':>16s}: [{lo:.2f}, {hi:.2f}]")
            print(f"  {'fit points':>16s}: {frac['n_fit_points']}")
        if 'exp_r2' in frac:
            print(f"  {'exp R²':>16s}: {frac['exp_r2']:.4f}")
            winner = "power-law" if frac['r2'] > frac['exp_r2'] else "exponential"
            print(f"  {'better fit':>16s}: {winner}")
    else:
        print(f"  unavailable: {frac.get('reason', '?')}")
    print(f"\n  fractal_scaling: {res['fractal_scaling'].upper()}   "
          f"(descriptive; pass if MEMB gives >= 3 points with N_B > 1)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
