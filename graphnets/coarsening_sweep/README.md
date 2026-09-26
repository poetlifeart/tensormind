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

The two spectral tests account for 0.69 of the 0.115 × 6 total against the
unconstrained arm.  **An earlier version of this README read that as the
constraint's benefit being spectral.  Section 4 shows it is not** — the
unconstrained arm does no thinning at all, and a rate-matched random null
recovers a *better* spectral gap than the colour rule does.  The spectral
improvement over `off` belongs to the thinning, not to colour.

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

## 4. `chromatic_null_controls.py` — is it colour, or is it thinning?

The chromatic filter also *thins*: measured here, it rejects **51.0%** of the
closure candidates that survive the block filter (758,472 candidates, 371,941
accepted, conditional acceptance rate p = 0.4904).  If rejecting a comparable
fraction at random did the same thing, the effect would be about thinning rather
than colour.  Four arms, each with its step-3 alpha bisected to the chromatic
arm's parent size of 477,584 edges:

| arm | extra condition on a candidate | accept rate |
|---|---|---|
| `chromatic` | `colour[nb1] != colour[nb2]` | 0.4904 |
| `random` | `rng.random() < 0.4904` — ignores colour | 0.4898 |
| `permuted` | same rule against a random permutation of the colour array (class sizes 2400/2400/3200 preserved) | 0.6612 |
| `off` | none | 1.0000 |

    python chromatic_null_controls.py

Results: `chromatic_null_controls.json`.

**resolution-only per block**

| | chromatic | random | permuted | off |
|---|---|---|---|---|
| accept rate | 0.4904 | 0.4898 | 0.6612 | 1.0000 |
| six-test mean | 0.1744 | **0.1549** | 0.2798 | 0.2904 |
| density error | **−4.98%** | −10.43% | −4.71% | +17.33% |
| spectral gap (target 7.51) | 15.27 | **12.29** | 58.36 | 69.15 |
| T1 degree | **0.1251** | 0.1448 | 0.1067 | 0.2182 |
| T4 scree | 0.5543 | **0.3566** | 1.0093 | 1.0035 |
| T5 network value | **0.0640** | 0.0749 | 0.3005 | 0.2944 |
| T6 triangles | **0.1852** | 0.2227 | 0.1685 | 0.1769 |

**deposited merge_densest**

| | chromatic | random | permuted | off |
|---|---|---|---|---|
| six-test mean | **0.2009** | 0.2228 | 0.3014 | 0.3724 |
| density error | **−8.34%** | −24.58% | −22.30% | −31.56% |
| spectral gap | 19.93 | **17.47** | 27.52 | 46.80 |
| T1 degree | **0.0985** | 0.2414 | 0.2158 | 0.3064 |
| T4 scree | 0.8304 | **0.6633** | 1.0285 | 1.1152 |
| T5 network value | **0.0394** | 0.0512 | 0.2335 | 0.4039 |
| T6 triangles | **0.1517** | 0.3369 | 0.3005 | 0.3803 |

### Thinning, not colour, carries the six-test mean

Random rejection at the same rate closes **87.3%** of the score gap under the
deposited coarsening and **116.8%** under the per-block one — where it beats the
colour rule outright, 0.1549 against 0.1744.  The six-test mean therefore has no
consistent sign for colour: chromatic wins under one coarsening, random under the
other.

### A SECOND RETRACTED MECHANISM

An earlier reading of section 3 held that the constraint's benefit was almost
entirely spectral.  **That is backwards.**  At matched rejection rate the colour
rule is *worse* on both spectral measures under both coarsenings: T4 scree 0.5543
against 0.3566 and 0.8304 against 0.6633; spectral gap 15.27 against 12.29 and
19.93 against 17.47.  The error was comparing against `off`, which does no
thinning at all, so the gap moving from 15.27 to 69.15 was a thinning effect
credited to colour.  The gap is in fact monotone in the accept rate — 12.29,
15.27, 58.36, 69.15 at rates 0.49, 0.49, 0.66, 1.00 — essentially independent of
which rule does the rejecting.

### What the colour rule does buy, at matched rate

Four measures, and these hold under **both** coarsenings, which the six-test mean
does not:

* density proximity: −4.98% vs −10.43%, and −8.34% vs −24.58%
* T1 degree distribution: 0.1251 vs 0.1448, and 0.0985 vs 0.2414
* T6 triangle participation: 0.1852 vs 0.2227, and 0.1517 vs 0.3369
* T5 network value, narrowly: 0.0640 vs 0.0749, and 0.0394 vs 0.0512

So the constraint is a degree-and-triangle effect and a density effect, not a
spectral one.

### The control that is still missing

`permuted` reaches density −4.71% under the per-block cut, essentially matching
chromatic's −4.98%, while rate-matched `random` reaches only −10.43%.  That hints
the density effect comes from the *must-differ-on-a-three-class-partition*
structure rather than from the specific identity of the colour classes.  But
`permuted` rejects only 33.9% against chromatic's 51.0%, so rule and rate are
confounded there and the hint is not a result.

The decisive arm would be `permuted` **plus** additional random rejection tuned
to p = 0.4904, making all arms rate-matched and size-matched.  Not run.

## Not established

* No nine-criterion audit has been run on any quotient here.  This matters for
  the resolution-only variant in particular: it wins the six static tests but has
  clustering 0.5488 against the deposited 0.6119, and clustering enrichment is a
  scored audit criterion the six static tests do not cover.
* One seed graph, one RNG seed per arm.  Nothing here speaks to seed-class
  robustness.
* Two coarsenings.  Claims should read "under both coarsenings tested".
* The `permuted` arm is not rate-matched, so nothing there separates the rule's
  structure from its rejection rate.
* Nothing here explains WHY the colour rule improves density and the degree and
  triangle distributions.  Two candidate mechanisms have now been tested and
  refuted: edge survival under quotienting (section 3), and a spectral account
  (section 4).  The mechanism is open.
