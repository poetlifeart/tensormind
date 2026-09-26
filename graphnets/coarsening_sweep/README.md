# Coarsening independence, and the chromatic constraint under two coarsenings

Three scripts, run 2026-09-26.  All scoring uses the paper's own
`construction/bilateral/static_tests.py`, unmodified.  The reference throughout
is `construction/bilateral/reference/budapest_1015_70654.edgelist`:
n = 1,015, m = 70,654, density 0.13730, clustering 0.6696, degree sd 71.55,
spectral gap 7.51.

Every script rebuilds the parent from `construct.py` itself (rng 99,
alpha = (3,4,30), beta = 1.5, three steps).  Harness check: the rebuilt parent
reproduces the deposited 1,015 / 64,760 quotient exactly, and the deposited row
reproduces its published six-test score of 0.2009 exactly.

## 1. `louvain_family_sweep.py` — does the merge rule carry the result?

The deposited coarsening fixes its node count by fiat: Louvain at resolution 30
inside each block over-partitions roughly fivefold, then `merge_densest` merges
the densest adjacent community pair until each block reaches a target read off
Budapest (307/306/201/201).  That confounds the merge rule with the algorithm
that chose the partition.

This script removes the merge rule and asks each method to reach 1,015
communities **by resolution alone**, so the node count is matched with no
Budapest-derived merge target — the only remaining input from Budapest is the
number 1,015.  Four Louvain-family objectives (networkx Louvain, python-louvain,
leidenalg RBConfiguration, leidenalg CPM), each run `global` (one run on the
whole parent, communities may straddle blocks) and `perblock` (one run inside
each block at the same gamma).

Resolution is bracketed by quadrupling then bisected on log gamma, 18
evaluations per configuration, seed 42 throughout.  The community count is
monotone in gamma only on average — 7.7442 gives 1,015 while 7.7390 and 7.7364
give 1,019 and 1,003 — so the best evaluation *seen* is taken, not the last.

    python louvain_family_sweep.py --evals 18 --out sweep

Results: `sweep.csv`, `sweep.json`, `sweep.log`.

| configuration | gamma | n | m | SCORE | clust | gap | ARI vs deposited |
|---|---|---|---|---|---|---|---|
| nx_louvain/perblock | 7.744 | 1,015 | 67,133 | 0.1744 | 0.5488 | 15.27 | 0.179 |
| leiden_mod/perblock | 7.497 | 1,016 | 64,832 | 0.1750 | 0.5458 | 14.49 | 0.173 |
| pylouvain/perblock | 7.797 | 1,015 | 67,483 | 0.1929 | 0.5503 | 16.73 | 0.179 |
| **deposited** (merge_densest) | 30 | 1,015 | 64,760 | 0.2009 | 0.6119 | 19.93 | 1.000 |
| leiden_mod/global | 24.36 | 1,012 | 80,894 | 0.2995 | 0.5217 | 27.97 | 0.150 |
| pylouvain/global | 25.35 | 1,014 | 83,108 | 0.3120 | 0.5341 | 27.03 | 0.151 |
| nx_louvain/global | 25.27 | 1,013 | 83,208 | 0.3139 | 0.5357 | 25.47 | 0.151 |
| leiden_cpm/global | 0.403 | 1,015 | 15,734 | 0.4717 | 0.7070 | 8.83 | 0.156 |
| leiden_cpm/perblock | 0.404 | 1,013 | 16,119 | 0.4721 | 0.7000 | 9.48 | 0.156 |

Four readings:

* **The score is robust to the algorithm; the partition is not.**  Three
  implementations of modularity Louvain land within 0.019 of each other, yet each
  agrees with the deposited partition at only ARI ≈ 0.18.  Different cuts, same
  statistics — which is the substance of a coarsening-independence claim and also
  its limit.
* **Per-block versus global is the dominant choice.**  0.174–0.193 against
  0.300–0.314 at matched node count.  Global runs also overshoot edges (~83,000
  against 70,654), because supernodes spanning two blocks manufacture coarse
  relations the block structure should forbid.
* **Density: resolution alone roughly halves the documented shortfall,** from
  −8.34% to −4.49% (pylouvain/perblock).  The trade runs the other way on degree
  spread — deposited 61.3 against resolution-only 46.0–48.0, target 71.55 — and
  on clustering, 0.6119 against 0.5458–0.5503, target 0.6696.  `merge_densest`
  merges the *densest* rather than the smallest pair specifically to preserve
  community-size spread, and the measurement confirms it does that job.  It is
  simply not the source of the connectome agreement.
* **The spectral gap is not intrinsically broken.**  It is the construction's
  largest documented residual at 19.93 against 7.51.  CPM reaches 8.83 and
  clustering 0.7070, both better than the deposited graph, but at 15,734 edges,
  so it loses overall.  The gap error is a property of how the quotient is cut,
  not something the parent forces.

Infomap is not installed on this machine, so that half of the
coarsening-independence question is still open.  `leidenalg` and `pymetis` are
available.

## 2. `chromatic_vs_coarsening.py` — the chromatic constraint, first pass

Colour rides on the leading base-20 digit exactly as block does, and every seed
edge is cross-colour, so the tensor product cannot create a monochromatic edge.
Closure is the only source of one, and relaxing the closure filter is the clean
ablation.  `close_triangles` and `grow` are **copied** from `construct.py` with
one change — the chromatic condition behind a flag — so `construct.py` stays
untouched and this file is a standalone record.

**This pass is confounded and is superseded by (3).**  It is kept because the
confound is worth recording: the two arms were *not* matched on parent edges,
despite the nominal quota alpha·n being equal in both.  The filter rejects
candidates, `close_triangles`' retry loop is bounded at 24 rounds, so the
constrained arm never fills its quota — parents came out 477,584 (enforced)
against 512,180 (relaxed), the relaxed arm 7.2% larger.

Results: `chromatic_vs_coarsening.json`.

## 3. `chromatic_matched_control.py` — the matched comparison

The relaxed arm's step-3 alpha is bisected down to 25.6641, giving a parent of
477,492 edges against the enforced arm's 477,584 — a 0.02% difference.  27.9% of
the relaxed arm's parent edges are monochromatic.  Both parents are then
coarsened both ways.

    python chromatic_matched_control.py

Results: `chromatic_matched_control.json`.

| coarsening | chi=3 | parent m | n | m | density | vs Budapest | clust | SCORE |
|---|---|---|---|---|---|---|---|---|
| deposited merge_densest | ON | 477,584 | 1,015 | 64,760 | 0.12584 | −8.34% | 0.6119 | 0.2009 |
| deposited merge_densest | OFF | 477,492 | 1,015 | 48,787 | 0.09480 | −30.95% | 0.6035 | 0.3731 |
| resolution-only per block | ON | 477,584 | 1,015 | 67,133 | 0.13046 | −4.98% | 0.5488 | 0.1744 |
| resolution-only per block | OFF | 477,492 | 1,015 | 82,377 | 0.16008 | +16.59% | 0.5817 | 0.2898 |

**The direction of the density effect depends on the coarsening, and this
survives matching.**  Under `merge_densest` the constraint yields 32.7% *more*
coarse relations; under resolution-only per-block Louvain it yields 22.7%
*fewer*.  What holds under both is proximity: density error falls from −30.95%
to −8.34% under one cut and from +16.59% to −4.98% under the other.  So the
constraint does not raise density — it moves density toward the reference, from
opposite sides.

**The six-test score improves under both, matched:** 0.2009 against 0.3731, and
0.1744 against 0.2898.

**Where the benefit actually sits — per test, resolution-only per block:**

| test | chi=3 ON | OFF | winner |
|---|---|---|---|
| T1 degree | 0.1251 | 0.2108 | chi=3 |
| T2 eff. diameter | 0.0551 | 0.0041 | OFF |
| T3 hop plot | 0.0627 | 0.0278 | OFF |
| T4 scree | 0.5543 | 1.0052 | chi=3, by 0.45 |
| T5 network value | 0.0640 | 0.3064 | chi=3, by 0.24 |
| T6 triangles | 0.1852 | 0.1842 | OFF, by 0.001 |
| mean | **0.1744** | 0.2898 | chi=3 |

The two spectral tests account for 0.69 of the 0.115 × 6 total.  Without the
constraint the spectral gap goes to 68.97 against the reference's 7.51,
destroying the near-degenerate leading pair, and T4 saturates at 1.0052 — the
value every structureless baseline reaches.  The constraint loses density,
clustering, effective diameter and hop plot.

One detail worth keeping: the unconstrained arm gets the degree *spread* nearly
right — sd 69.2 against the reference's 71.55, where chi=3 manages 46.0 — and
still loses the degree test, 0.2108 to 0.1251.  Right spread, wrong shape.

### A superseded mechanism, recorded rather than deleted

An earlier account held that chromatically legal closures join distinct fibres
and are therefore likelier to cross a supernode boundary and survive
quotienting, rather than being absorbed into a supernode.  **Measurement does
not support it.**  Survival rates are near-equal: 93.4% against 90.2% under
`merge_densest`, and 81.8% against 82.4% under resolution-only — where the
constrained arm survives slightly *less*.  Whatever the constraint does, it is
not making individual edges likelier to survive coarsening.  The mechanism is
currently unexplained.

## Not established

* No nine-criterion audit has been run on any quotient here.  This matters for
  the resolution-only variant in particular: it wins the six static tests but has
  clustering 0.5488 against the deposited 0.6119, and clustering enrichment is a
  scored audit criterion the six static tests do not cover.
* One seed graph, one RNG seed per arm.  Nothing here speaks to seed-class
  robustness.
* Two coarsenings.  Claims should read "under both coarsenings tested".
