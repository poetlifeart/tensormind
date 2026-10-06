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

> **CORRECTED 2026-10-06 — read sections 5 and 6 before 3 and 4.** Sections 3
> and 4 were written before commit b04671b found a sampler truncation bug that
> corrupted every arm with candidates to spare: 100% of the closure edges in the
> unconstrained `off` arm, and 239,804 edges of the `permuted` arm. Their
> conclusions — that thinning rather than colour carries the effect, that the
> spectral gap is monotone in the accept rate, and that a rate-matched permuted
> control had not been run — do not survive the repair. Section 5 gives the
> repaired results, which are the ones the main paper reports. Section 6 adds
> the nine-criterion audits and the connector-hub seed sweep that were run
> afterwards. Sections 1–4 are kept as the record of what was measured at the
> time.

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

> **Superseded by section 5.** The OFF rows below come from the unrepaired
> sampler, whose truncation step kept the lowest-indexed candidates; in the
> unconstrained arm that cut applied to every closure edge. The direction claim
> ("32.7% more under one cut, 22.7% fewer under the other") does not hold after
> the repair: the constraint gives MORE coarse relations under both coarsenings.

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

> **Superseded by section 5.** The `off` and `permuted` arms here were corrupted
> by the truncation bug (207,029 and 239,804 truncated edges). Three readings
> below are withdrawn: "thinning, not colour" (the corrupted `off` arm was what
> made thinning look sufficient), "the gap is monotone in the accept rate" (the
> repaired `off` arm, at rate 1.0, has gap 12.36 against chromatic's 16.08), and
> "the control that is still missing" (it was run; see section 5).

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
to p = 0.4904, making all arms rate-matched and size-matched.  (Since run: see
section 5.)

## 5. `sampler_diagnostic.py`, `r1_permuted_rate_matched.py`, `r2_truncation_fix.py` — the repaired controls

**These are the results the main paper reports.**

**The bug.** `close_triangles`, copied here from `construct.py`, accepts
candidates as `np.unique(...)[:remaining]`. `np.unique` sorts, so whenever a
round yields more candidates than the remaining quota, the cut keeps the
lowest-keyed ones — a deterministic bias toward low vertex indices, which carry
both colour and block in their leading base-20 digit. `sampler_diagnostic.py`
found it. It barely touches the constrained arm, which never has candidates to
spare, but it decided every closure in the unconstrained arm:

| arm | closure edges from a truncated round |
|---|---|
| chromatic | 3 of 241,605 |
| random | 14 of 207,043 |
| off | 207,029 of 207,043 |

`graphnets/ablation/chromatic_ablation.py` already shuffled before truncating, so
the paper's ablation table and the stage ablation were never affected. These
scripts had copied the unshuffled version.

**The repair** (`r2_truncation_fix.py`) takes a uniform random subset from a
separate generator (`TRUNC_SEED = 20260926`), so the growth stream is never
advanced and `construct.py` stays untouched. An arm that never truncates is
therefore bit-identical with and without the fix, which is the built-in control.

**R1** (`r1_permuted_rate_matched.py`) is the control section 4 said was missing:
the must-differ rule on a random relabelling of the colours (class sizes
2400/2400/3200 kept), plus extra random rejection so that its accept rate
matches chromatic's.

    python r2_truncation_fix.py        # -> r2_truncation_fix.json

All four arms parent-matched at ~477,600 edges, repaired sampler:

**deposited merge_densest**

| | chromatic | random | permuted, rate-matched | off |
|---|---|---|---|---|
| accept rate | 0.4904 | 0.4899 | 0.4908 | 1.0000 |
| quotient relations | **64,766** | 51,896 | 53,607 | 55,719 |
| density error | **−8.33%** | −26.55% | −24.13% | −21.14% |
| spectral gap (target 7.51) | 19.93 | **16.01** | 26.32 | 16.43 |
| T1 degree | **0.0995** | 0.2690 | 0.2552 | 0.2158 |
| T4 scree | 0.8302 | **0.5684** | 1.0173 | 0.5957 |
| T5 network value | **0.0384** | 0.0552 | 0.2453 | 0.0493 |
| T6 triangles | **0.1537** | 0.3695 | 0.3537 | 0.3084 |
| six-test mean | 0.2012 | 0.2216 | 0.3196 | **0.1977** |

**resolution-only per block**

| | chromatic | random | permuted, rate-matched | off |
|---|---|---|---|---|
| quotient relations | **66,916** | 62,324 | 65,013 | 64,797 |
| density error | **−5.29%** | −11.79% | −7.98% | −8.29% |
| spectral gap | 16.08 | **11.79** | 42.88 | 12.36 |
| T1 degree | 0.1222 | 0.1507 | 0.1507 | **0.1192** |
| T4 scree | 0.6040 | **0.3229** | 1.0040 | 0.3610 |
| T5 network value | 0.0808 | **0.0768** | 0.3320 | 0.0975 |
| T6 triangles | **0.1833** | 0.2335 | 0.2108 | 0.1970 |
| six-test mean | 0.1855 | 0.1532 | 0.3137 | **0.1490** |

**What holds under both coarsenings, against every null:**

* **More coarse relations.** Chromatic over random +24.8% / +7.4%, over
  rate-matched permuted +20.8% / +2.9%, over off +16.2% / +3.3%
  (merge / per-block). The "about 10% more coarse relations" reading is not
  dead; the sign flip claimed in section 3 came from the corrupted `off` arm.
* **Density closer to the reference**: −8.33% and −5.29%, against −26.55% and
  −11.79% for random, −24.13% and −7.98% for permuted, −21.14% and −8.29% for off.
* **T6 triangle participation**: chromatic is best of the four under both.

**It is this colouring, not any three-class rule.** At matched rate the
relabelled rule loses to chromatic on density, T1 and T6 under both
coarsenings. The apparent tie in section 4 came from its lower rejection rate
(33.9% against 51.0%) plus its own truncation contamination.

**What does not hold.** The six-test mean favours the unconstrained arm under
both coarsenings, and random rejection under the per-block one. The spectral
gap and T4 scree favour random and off under both. T1 is chromatic's under
merge_densest but off's by 0.003 under per-block; T5 is chromatic's under
merge_densest but random's by 0.004 under per-block.

**Noise floor for the per-block pipeline.** Three parent edges out of 477,584
(the chromatic arm with and without the repair) move the per-block six-test
mean by 0.0111, because the pipeline re-sweeps resolution and Louvain flips.
Per-block differences below ~0.011 are not meaningful, which is why off against
random (0.0042) reads as zero.

## 6. `three_coarsenings.py`, the audits, and the connector-hub seed sweep

Four coarsenings of the deposited parent:

| | coarsening | nodes / edges | audit | connector hubs | Q |
|---|---|---|---|---|---|
| A | density merge, Budapest's 307/306/201/201 (deposited) | 1,015 / 64,760 | 9/9 | 9 | 0.5318 |
| C | density merge, the parent's own 305/304/203/203 | 1,015 / 64,541 | 9/9 | 10 | 0.5571 |
| P | per-block resolution sweep, no merge (the paper's Louvain-only) | 1,076 / 73,152 | 9/9 | 1 | 0.5867 |
| B | one shared resolution, no merge | 1,015 / 67,133 | 8/9 | 0 | 0.5862 |

100 degree-preserving nulls, 1,000 rich-club nulls, seed 42. A's audit is
`../construction/bilateral/audits/audit_8k_quotient_100.json`; B, C and P are in
`audits/`. Only B fails, on connector hubs, and B is not in the paper.

`hub_seed_sweep.py` repeats the hub count over 20 Louvain seeds
(`hub_seed_sweep.json`):

| | criterion passes | hubs per seed | max participation among high-z nodes |
|---|---|---|---|
| A | 20/20 | 7–10 | 0.487–0.624 |
| C | 20/20 | 5–10 | 0.488–0.628 |
| P | 19/20 | 0–1 | 0.289–0.327 |
| B | 0/20 | 0 | 0.251 every seed |

P clears the criterion on a single hub whose participation straddles the 0.30
threshold, and fails at one seed (seed 3). Without the merge step the
connector-hub result is fragile in both resolution and Louvain seed; with it,
robust in both. B and P are the same coarsening at two nearby resolutions (all
four blocks select gamma = 8.0 in P) and share the same flat degree
distribution — sd 46.0 and 48.0, maximum degree 288 and 287 — against A and
C's 61.3 / 383 and 61.2 / 382.

## Not established

* One seed graph, one RNG seed per arm.  Nothing here speaks to seed-class
  robustness.
* Two coarsenings for the chromatic controls.  Claims should read "under both
  coarsenings tested".
* Nothing here explains WHY the colour rule improves coarse-relation count,
  density and triangle participation.  Two candidate mechanisms have been tested
  and refuted: edge survival under quotienting (section 3; the repaired survival
  rates are 93.4% against 90.4% for random under merge_densest and 81.7% against
  81.8% under per-block), and a spectral account (section 4).  The mechanism is
  open.
