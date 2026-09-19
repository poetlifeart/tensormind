#!/usr/bin/env python3
"""
bundles.py -- post-processing of the 8,000-node construction's fibre bundles.

A coarse relation's multiplicity W_AB is the number of fine edges crossing
between two supernodes.  The reference connectome's edge weight is a mean
FIBRE count.  Those are different quantities, and this script measures how
they relate.

    python3 bundles.py                  # everything
    python3 bundles.py --section fan

Sections
    dist     the raw distribution against the reference
    fan      how many distinct vertices actually participate at each end
    tract    counting rules and fitted transforms
    floor    the irreducible KS floor, and why it exists

Reads ../graphs/ and ../reference/, and rebuilds the parent with ../construct.py
for the fan measurement.  Requires numpy, scipy, networkx.
"""
import argparse, os, sys, importlib.util
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(HERE)

def _load_construct():
    spec = importlib.util.spec_from_file_location("construct", os.path.join(PKG, "construct.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())

def load_weights():
    """Quotient bundle weights, and the reference's mean fibre counts."""
    W = np.load(os.path.join(PKG, 'graphs', 'cnew_coarse_w.npy'))
    B = np.load(os.path.join(PKG, 'reference', 'bud_w.npy'))[0]   # row 0 = mean fibre count
    return W.astype(float), B.astype(float)

def build_relations():
    """Rebuild the parent and its partition; return per-relation (W, a, b)."""
    import networkx as nx
    c = _load_construct()
    rng = np.random.default_rng(99)
    u, v, col, blk = c.grow(c.build_seed(), [3, 4, 30], 1.5, 3, rng, verbose=False)
    G = nx.Graph(); G.add_nodes_from(range(8000)); G.add_edges_from(zip(u, v))
    lab = c.coarsen(G, blk, c.BLOCK_TARGETS)
    a, b = lab[u], lab[v]; cross = a != b
    rel = defaultdict(lambda: [set(), set(), 0])
    for x, y, su, sv in zip(u[cross], v[cross], a[cross], b[cross]):
        if su > sv: x, y, su, sv = y, x, sv, su
        r = rel[(su, sv)]; r[0].add(int(x)); r[1].add(int(y)); r[2] += 1
    Wv = np.array([r[2] for r in rel.values()], float)
    A  = np.array([len(r[0]) for r in rel.values()], float)
    Bv = np.array([len(r[1]) for r in rel.values()], float)
    return Wv, A, Bv, int(cross.sum()), int((~cross).sum()), len(u)

# ------------------------------------------------------------------ sections

def sec_dist(**_):
    from scipy.stats import ks_2samp
    W, B = load_weights()
    print("\n=== DISTRIBUTION =================================================")
    print("  %-22s %8s %8s %8s %9s %9s"%("", "n", "mean", "median", "max", "Gini"))
    for nm, x in (("8k quotient", W), ("reference", B)):
        print("  %-22s %8d %8.3f %8.2f %9.1f %9.4f" % (nm, len(x), x.mean(), np.median(x), x.max(), gini(x)))
    print("\n  mass carried by relations of weight <= 5")
    for nm, x in (("8k quotient", W), ("reference", B)):
        print("    %-20s %5.1f%%" % (nm, 100 * x[x <= 5].sum() / x.sum()))
    print("\n  quantiles")
    print("    %-6s %10s %10s %8s" % ("", "8k", "reference", "ratio"))
    for p in (50, 75, 90, 95, 99, 99.9):
        a, b = np.percentile(W, p), np.percentile(B, p)
        print("    p%-5s %10.1f %10.2f %8.1fx" % (p, a, b, a / b))
    print("\n  KS (means matched): %.4f" % ks_2samp(W * (B.mean() / W.mean()), B).statistic)

def sec_fan(**_):
    print("\n=== FAN-IN / FAN-OUT =============================================")
    print("  rebuilding the parent ...", flush=True)
    W, A, Bv, cross, intra, m = build_relations()
    print("  parent %s edges = %s crossing (%.1f%%) + %s absorbed in supernodes"
          % (f"{m:,}", f"{cross:,}", 100 * cross / m, f"{intra:,}"))
    print("\n  a, b = distinct vertices participating at each end of a relation")
    print("  %-12s %8s %8s %8s %9s %10s %12s" %
          ("W band", "n rel", "mean a", "mean b", "mean W", "a*b / W", "(a+b) / W"))
    for lo, hi in [(1,1),(2,2),(3,5),(6,10),(11,20),(21,50),(51,100),(101,300),(301,10**9)]:
        s = (W >= lo) & (W <= hi)
        if not s.sum(): continue
        lbl = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**8 else f"{lo}+")
        print("  %-12s %8d %8.2f %8.2f %8.1f %9.3f %12.3f" %
              (lbl, s.sum(), A[s].mean(), Bv[s].mean(), W[s].mean(),
               (A[s]*Bv[s]/W[s]).mean(), ((A[s]+Bv[s])/W[s]).mean()))
    print("\n  compression is SIZE-DEPENDENT: light relations expand (a+b > W),")
    print("  heavy ones compress ~5x.  The crossover is near W = 15.")
    print("\n  the five heaviest relations, as converge-diverge tracts:")
    for k in np.argsort(-W)[:5]:
        print("    W=%4d  a=%3d b=%3d  ->  %3d fibres  (%.1fx compression, %.0f%% saturated)"
              % (W[k], A[k], Bv[k], A[k]+Bv[k], W[k]/(A[k]+Bv[k]), 100*W[k]/(A[k]*Bv[k])))

def sec_tract(**_):
    from scipy.stats import ks_2samp
    from scipy.optimize import brentq
    W, B = load_weights(); TGT = B.mean()
    print("\n=== TRACT READINGS ===============================================")
    print("  A relation of multiplicity W is realised by fewer fibres if the")
    print("  vertices converge locally, cross, and fan out.  Every rule below")
    print("  satisfies f(1) = 1, and its free parameter is fixed by the mean.")
    p_pure   = brentq(lambda q: (W**q).mean() - TGT, 0.05, 1.0)
    smooth   = lambda W, p, c: W**p * (1.0 + (W - 1.0) / c)**(1.0 - p)
    C = 50.0
    p_smooth = brentq(lambda q: smooth(W, q, C).mean() - TGT, -1.0, 1.0)
    rows = [("relations W (rescaled)", W * (TGT / W.mean())),
            ("pure power  W^%.3f" % p_pure, W**p_pure),
            ("smooth  W^%.2f (1+(W-1)/%d)^%.2f" % (p_smooth, int(C), 1-p_smooth), smooth(W, p_smooth, C)),
            ("reference", B)]
    print("\n  %-38s %7s %7s %8s %8s %8s" % ("", "median", "max", "Gini", "mass<=5", "KS"))
    for nm, x in rows:
        ks = 0.0 if nm == "reference" else ks_2samp(x, B).statistic
        print("  %-38s %7.2f %7.1f %8.4f %7.1f%% %8.4f" %
              (nm, np.median(x), x.max(), gini(x), 100*x[x<=5].sum()/x.sum(), ks))
    print("\n  The pure exponent %.3f is fixed by the mean alone -- one constraint," % p_pure)
    print("  one unknown, nothing about the shape enters.  The convergence argument")
    print("  predicts 1/2 independently: relations go as a*b, fibres as a+b, so")
    print("  fibres ~ sqrt(W) when the two ends are comparable in size.")
    print("  Gini and body mass then follow without further fitting.")
    print("\n  The smooth form adds one parameter and buys the tail:")
    print("    maximum  %6.1f (pure power)  %6.1f (smooth)  %6.1f (reference)"
          % ((W**p_pure).max(), smooth(W, p_smooth, C).max(), B.max()))
    print("    the measured fan for that same relation is 104 fibres (45 x 59 vertices)")

def sec_floor(**_):
    W, B = load_weights()
    a, b = (W == 1).mean(), (B <= 1.0).mean()
    print("\n=== KS FLOOR =====================================================")
    print("  Any rule with f(1) = 1 sends every weight-1 relation to exactly 1.")
    print("    8k relations at W = 1          %5.1f%%" % (100 * a))
    print("    reference connections at 1.0   %5.1f%%" % (100 * b))
    print("    irreducible CDF gap            %.4f   <- KS cannot go below this" % abs(a - b))
    print("\n  It exists because our weights are integer edge counts and the")
    print("  reference's are MEAN fibre counts, which take fractional values.")
    print("  That is a difference between the two measurements, not the graphs.")

SECTIONS = dict(dist=sec_dist, fan=sec_fan, tract=sec_tract, floor=sec_floor)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--section', choices=list(SECTIONS) + ['all'], default='all')
    args = ap.parse_args()
    todo = SECTIONS if args.section == 'all' else {args.section: SECTIONS[args.section]}
    for name, fn in todo.items():
        fn()
    print()

if __name__ == '__main__':
    main()
