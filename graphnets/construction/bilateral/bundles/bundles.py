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
    law      the model's axon law f(W), on the colour-to-colour bundles it
             actually acts on -- a finer object than the coarse relations above
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

# ------------------------------------------------------------------- the law
P_EXP, C_KNEE = 0.05, 11.0

def n_axons(W, p=P_EXP, c=C_KNEE):
    """Axons carrying a bundle of W fine edges.

        f(W) = round( W^p * (1 + (W-1)/c)^(1-p) ),  minimum 1

    f(1) = 1 identically, for every p and c -- a single edge is a single axon
    by construction of the form, not by a choice of constants.  For large W,
    f -> W / c^(1-p): the growth is asymptotically LINEAR with a constant
    compression factor c^(1-p) = 9.76, not sub-linear.  What is sub-linear is
    f/W, which falls monotonically from 1 to 1/c^(1-p) = 0.102, so the heaviest
    bundles are compressed hardest.  The log-log slope runs from
    p + (1-p)/c = 0.136 at W = 1 to 1 as W grows; p alone is not the low-W
    exponent.  Neither limit fixes p or c -- the asymptote gives one equation in
    two unknowns.  Both constants are fitted, and nothing in the construction
    supplies them.

    This is the same function the recurrent model applies, in
    graphdynamics/v1004bundle/model_v1004bundle.py as
    BundleSparseLinear._n_axons.  If one changes, change both.
    """
    W = np.asarray(W, dtype=float)
    return np.clip(np.round(W**p * (1.0 + (W - 1.0) / c)**(1.0 - p)), 1.0, None)


def load_bundles():
    """The model's colour-to-colour bundles, read from the model's own input.

    A bundle is (ordered supernode pair, colour pair).  Inside one layer the
    colour pair is fixed, so BundleSparseLinear's key
    `sid_in[source] * n_super + sid_out[target]` is exactly that.  The three
    undirected colour pairs give three independent partitions of the parent's
    edges, and nothing is pooled across colour classes.

    `graphs/cnew_parent_supernode.npz` is byte-identical to the model's
    `graph_8k_parent_supernode.npz`, so this reproduces the model's bundles
    exactly, without rebuilding the parent.

    Returns the per-bundle weights; per bundle, the unordered supernode pair it
    belongs to (-1 where the two supernodes coincide); and which colour pair it
    came from (0 = 1-2, 1 = 1-3, 2 = 2-3).
    """
    d = np.load(os.path.join(PKG, 'graphs', 'cnew_parent_supernode.npz'))
    sid = [d['supernode_id_1'], d['supernode_id_2'], d['supernode_id_3']]
    ns = int(d['n_supernodes'])
    W, pair, lay = [], [], []
    for t, (k, (i, j)) in enumerate((('mask_12', (0, 1)), ('mask_13', (0, 2)),
                                     ('mask_23', (1, 2)))):
        row, col = np.nonzero(d[k])          # row indexes colour j, col colour i
        key = sid[i][col].astype(np.int64) * ns + sid[j][row]
        uk, cnt = np.unique(key, return_counts=True)
        a, b = uk // ns, uk % ns
        W.append(cnt.astype(float))
        pair.append(np.where(a == b, -1, np.minimum(a, b) * ns + np.maximum(a, b)))
        lay.append(np.full(len(uk), t))
    return np.concatenate(W), np.concatenate(pair), np.concatenate(lay)


def sec_law(**_):
    Wp, B = load_weights()
    W, pair, lay = load_bundles()
    F = n_axons(W)
    print("\n=== THE LAW ======================================================")
    print("  f(W) = round( W^%.2f * (1 + (W-1)/%.1f)^%.2f ),  minimum 1"
          % (P_EXP, C_KNEE, 1 - P_EXP))
    print("\n     W    f(W)     f/W")
    for w in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        f = float(n_axons(w))
        print("  %4d  %6.0f  %6.3f" % (w, f, f / w))
    print("\n  f/W falls monotonically from 1 to 1/%.1f^%.2f = %.3f, so the heaviest"
          % (C_KNEE, 1 - P_EXP, C_KNEE**-(1 - P_EXP)))
    print("  bundles are compressed hardest.  The smooth form is strictly concave;")
    print("  the rounded f is only monotone, since its steps are 0 or exactly 1.")
    print("\n  THE OBJECT.  The law acts on COLOUR-TO-COLOUR bundles -- one per")
    print("  (supernode pair, colour pair) -- not on the pooled quotient weights.")
    print("  The three colour pairs partition the parent's %s edges into %s"
          % (f"{int(W.sum()):,}", f"{len(W):,}"))
    print("  bundles.  Pooling them by supernode pair alone gives the %s coarse"
          % f"{len(Wp):,}")
    print("  relations used by the other sections, and is a DIFFERENT object.")
    print()
    for tag, x in (("bundles W", W), ("axons f(W)", F),
                   ("pooled quotient W", Wp), ("reference fibres", B)):
        x = np.asarray(x, float); x = x[x > 0]
        srt = np.sort(x); cum = np.cumsum(srt)
        gini = float((2*np.arange(1, len(srt)+1) - len(srt) - 1).dot(srt)
                     / (len(srt) * cum[-1]))
        print("    %-20s n=%-7d median=%6.2f  mean=%6.2f  max=%7.1f  "
              "Gini=%.3f  mass<=5=%5.1f%%"
              % (tag, len(x), np.median(x), x.mean(), x.max(), gini,
                 100 * x[x <= 5].sum() / x.sum()))
    m = pair >= 0
    tot = np.zeros(int(pair[m].max()) + 1)
    np.add.at(tot, pair[m].astype(np.int64), F[m])
    tot = tot[tot > 0]
    dup = lay > 0                            # the 1-3 and 2-3 partitions
    print("\n  Axons per bundle %.3f.  The model instantiates five layers, two of"
          % F.mean())
    print("  them transposes, so the 1-3 and 2-3 partitions are built twice:")
    print("    %s bundles, %s axons -- the model's graph-weight count."
          % (f"{len(W) + int(dup.sum()):,}", f"{int(F.sum() + F[dup].sum()):,}"))
    print("\n  Summing axons over the colour pairs of one supernode pair gives the")
    print("  only quantity commensurable with the reference's per-connection")
    print("  fibre counts: n=%d  median=%.2f  mean=%.2f  max=%.0f"
          % (len(tot), np.median(tot), tot.mean(), tot.max()))
    print("  against the reference's %d / %.2f / %.2f / %.1f.  It is not a match,"
          % (len(B), np.median(B), B.mean(), B.max()))
    print("  and the law is not what produces the paper's fibre-bundle agreement:")
    print("  that comes from the degree-matched relocation quotient, not from any")
    print("  transform of the weights.  p and c are fitted; see the README.")


SECTIONS = dict(dist=sec_dist, fan=sec_fan, tract=sec_tract,
                law=sec_law, floor=sec_floor)

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
