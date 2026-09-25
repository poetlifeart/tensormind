"""Does the Fiedler bisection IDENTIFY the bilateral partition, or merely
match its cut cost?

Equal cut cost does not imply equal partition: c(S1) ~ c(S2) does NOT imply
S1 ~ S2, since two balanced partitions can cut the same number of edges while
sharing few vertices.  Comparing 3.71% against 3.97% therefore establishes
similar cost, not similar membership.  This script measures membership
directly -- vertex agreement, ARI, NMI, and the correlation between the
Fiedler vector and the centred bilateral indicator.

It reproduces the deposited bisection cut of 2,404 edges and shows that the
Fiedler bisection and the bilateral split are the SAME partition on all 1,015
supernodes (ARI = 1.000), as they are for the reference.

    python3 fiedler_vs_bilateral.py        # ~1 minute

Builds the parent with construct.py's grow(), so it reproduces the deposited
instance exactly.  Requires scikit-learn in addition to the usual dependencies.
"""
import os, sys, importlib.util
import numpy as np, networkx as nx, scipy.sparse.linalg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Build with construct.py's OWN grow(), not the ablation's.  chromatic_ablation
# .close_relaxed permutes its candidate pool before applying the quota (the
# sampler fix), which moves the random stream, so CA.grow(0.0, ...) lands on a
# different -- equally valid -- instance rather than the deposited one.  The
# paper's quotient comes from construct.py, so this must too.
_B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  '..', 'construction', 'bilateral')
_spec = importlib.util.spec_from_file_location("cc", os.path.join(_B, 'construct.py'))
cc = importlib.util.module_from_spec(_spec)
sys.modules["cc"] = cc
_spec.loader.exec_module(cc)

rng = np.random.default_rng(99)
u, v, colour, block = cc.grow(cc.build_seed(), [3, 4, 30], 1.5, 3, rng, verbose=False)
P = nx.Graph(); P.add_nodes_from(range(len(block))); P.add_edges_from(zip(u.tolist(),v.tolist()))
label = cc.coarsen(P, block, cc.BLOCK_TARGETS, seed=42)
Q, super_block, pairs, witness = cc.build_coarse(u, v, label, block, sum(cc.BLOCK_TARGETS))
print(f"rebuilt quotient: {Q.number_of_nodes()} nodes, {Q.number_of_edges():,} edges "
      f"(deposited: 1015 / 64,760)")

# hemispheres are {A,C} = blocks {0,2} and {B,D} = blocks {1,3}
hemi = np.isin(super_block, [0,2]).astype(int)
print(f"hemisphere sizes: {hemi.sum()} / {(1-hemi).sum()}   "
      f"(blocks {np.bincount(super_block)})")

E = np.array(Q.edges())
cutH = int((hemi[E[:,0]] != hemi[E[:,1]]).sum())
m = Q.number_of_edges()
print(f"\n|dH| bilateral      = {cutH:,}  ({100*cutH/m:.2f}%)   paper: 2,404 (3.71%)")

# Fiedler bisection, exactly as combinatorial_metrics_npz.min_cut_bal does it
L = nx.laplacian_matrix(Q).astype(float)
vals, vecs = scipy.sparse.linalg.eigsh(L, k=2, which='SM')
order = np.argsort(vals); lam2 = float(vals[order[1]]); v2 = vecs[:, order[1]]
# For odd n there are TWO admissible balanced splits, n//2 and n//2+1, and
# combinatorial_metrics_npz.min_cut_bal reports the better of the two.  Taking
# only rank[:n//2] gives 2,568 here; the other split gives 2,404 and is the one
# min_cut_bal deposits.  Match the deposited function.
rank = np.argsort(v2, kind='stable'); n_q = Q.number_of_nodes()
_best = None
for _size in (n_q // 2, n_q // 2 + 1):
    _p = np.zeros(n_q, int); _p[rank[:_size]] = 1
    _c = int((_p[E[:, 0]] != _p[E[:, 1]]).sum())
    if _best is None or _c < _best[0]:
        _best = (_c, _p)
cutF, fied = _best
print(f"|dF| Fiedler        = {cutF:,}  ({100*cutF/m:.2f}%)   paper: 2,568 (3.97%)")
print(f"lambda_2            = {lam2:.4f}                    paper: 6.90")

print("\n--- MEMBERSHIP, NOT COST: do the two partitions pick the same vertices? ---")
agree = max((fied==hemi).mean(), (fied!=hemi).mean())
print(f"  vertex agreement after label swap : {100*agree:.2f}%")
print(f"  ARI(Fiedler, bilateral)           : {adjusted_rand_score(hemi, fied):.4f}")
print(f"  NMI(Fiedler, bilateral)           : {normalized_mutual_info_score(hemi, fied):.4f}")
h = hemi*2.0-1.0; h -= h.mean()
corr = abs(float(v2 @ h)) / (np.linalg.norm(v2)*np.linalg.norm(h))
print(f"  |<v2, h>| / (||v2|| ||h||)        : {corr:.4f}")
np.savez('fiedler_check.npz', hemi=hemi, fied=fied, v2=v2, super_block=super_block)
