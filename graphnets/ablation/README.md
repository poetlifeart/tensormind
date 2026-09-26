# Ablation: is the chromatic constraint doing anything?

Three questions about the 8,000-node construction in
`../construction/bilateral/construct.py`, and the code that answers them.

1. **What is random in the construction, and how much does it matter?**
2. **Does the chromatic constraint ever bind, or is it preserved vacuously?**
3. **What happens to the quotient if you break it?**

All of it builds on `construct.py` unchanged. Two scripts carry hand-copies
of its closure loop; see *On code duplication* at the foot of this file.

## Two things called "seed"

Do not confuse them.

| | what it is | varies? |
|---|---|---|
| **seed graph** | the fixed 20-vertex, 31-edge graph in `construct.py` | never |
| **random seed** | `np.random.default_rng(args.seed)`, default **99** | yes, that is what these scripts vary |

The random seed drives **one** thing: which triangle closures get sampled. At
each attempt a vertex `v` is drawn with probability proportional to
`deg(v)**1.5`, two of its neighbours are drawn uniformly, and the edge between
them is added if it passes three deterministic filters. The seed decides
*which* legal closures are made; it cannot admit an illegal one.

It does **not** affect the coarsening *algorithm*: `coarsen(..., seed=42)` is
hard-coded, so every run uses the same Louvain seed and the same densest-pair
merge rule. The resulting partitions still differ between runs, because Louvain
is applied to a different parent graph each time — which is why the quotient
edge counts differ (64,760 / 63,413 / 62,901 / 63,628 / 62,584).

The parent's 477,584 edges decompose three ways. Only the first is
deterministic; the tensor product *copies* earlier random closures forward, so
its output is not all forced.

| source | edges | |
|---|---:|---|
| pure tensor images of the 31 seed edges, `2^2 * 31^3` | 119,164 | deterministic |
| tensor images of the 5 seed-stage closures, `5 * 62^2` | 19,220 | chance-placed |
| tensor images of the 1,600 scale-2 closures, `1600 * 62` | 99,200 | chance-placed |
| direct scale-3 closures | 240,000 | chance-placed |
| **total** | **477,584** | |

So **75%** of the parent is placed by chance inside the set the constraint
defines, not "roughly half". The figure `product m = 237,584` printed by
`construct.py` at scale 3 is the product of the scale-2 graph, which already
contained 1,605 random closures — it is not a count of deterministic edges.

> Note this is the **opposite** of the local construction in
> `../construction/local/`, where the fine parent is deterministic (277,319
> edges every run) and the Louvain coarsening is the only stochastic step.

## Results

### Seed robustness — five seeds, all 9/9

| | seed 99 (deposited) | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| parent edges | 477,584 | 481,428 | 481,366 | 481,428 | 481,366 |
| monochromatic | 0 | 0 | 0 | 0 | 0 |
| quotient | 1015 / 64,760 | 1015 / 63,413 | 1015 / 62,901 | 1015 / 63,628 | 1015 / 62,584 |
| clustering | 0.6119 | 0.6139 | 0.6111 | 0.6096 | 0.6097 |
| nine-criterion audit | **9/9** | **9/9** | **9/9** | **9/9** | **9/9** |

All five pass every criterion, under the same protocol as the deposited run
(100 degree-preserving nulls, 1,000 rich-club nulls). Audit JSONs are in this
folder as `audit_quot_seed*.json`.

> **These four cannot be regenerated from the current code, and that is by
> design.** `build_seed_quotients.py` calls `chromatic_ablation.py`'s `grow()`,
> which now permutes candidates before applying the quota. `rng.shuffle` consumes
> draws, so the random stream moved: at `f = 0` seed 99 the quotient is now
> 1,015 / 64,491, and the five f=0 seeds span 61,489–64,491 (which is what
> `--control` asserts). The shipped `quot_seed*.npz` and their audits predate the
> fix. Seed 99's 64,760 is still reproducible — that graph comes from
> `construct.py`, which is untouched — but seeds 1–4 are not. Rerunning
> `build_seed_quotients.py` gives four different, equally valid graphs that would
> need re-auditing.

The deposited run is the most favourable of the five — it has the largest
quotient, which puts its density and mean degree closest to the reference.
Reproducing under another seed should give about 63,000 quotient edges and the
same 9/9.

The unscored tenth row, `fractal_scaling`, shows **`fail` in all five
scorecards, but nothing was measured.** The JSONs record
`{"available": false, "reason": "insufficient_scaling_range",
"diameters": [2,4], "box_counts": [13,2]}` — at diameter 5 there are only two
usable box sizes, which cannot support a scaling fit. The audit tool prints
that as `fail`. Read it as *not measurable at this scale*, not as a measured
negative. (Separate work on box-counting fractality finds the full graph is not
fractal while its maximum-edge-betweenness skeleton is; that analysis is not
part of this audit.)

### The constraint binds on half of all candidates

Of the closure **proposals** whose endpoints were distinct and in the same
block — the adjacency test happens *after* the colour filter, so it is not part
of this denominator, and proposals carry multiplicity because the sampler draws
with replacement:

```
stage               otherwise legal   killed by colour   bind rate
seed   (n=20)                   155                 54      34.8%
tensor 2 (n=400)              6,373              3,135      49.2%
tensor 3 (n=8,000)          751,944            383,342      51.0%
ALL STAGES                  758,472            386,531      51.0%
```

**51.0%** of those proposals are refused for colour. The constraint is not
slack.

The quota is **not** met at the seed stage. The 20-vertex seed admits exactly
**six** legal closures in total — `{2,5} {3,4} {3,5} {8,11} {9,10} {9,11}`,
confirmed by brute force — against a quota of `alpha[0] * n = 60`. Seed 99
accepts 5 of the 6 and seeds 1–4 accept all 6; seed 99 stops one short not
because the pool is empty but because its retry loop hits
`if not len(keys): break` on a round that redraws only already-seen pairs.

Nor is the quota always met at scale 2. Per-stage accept counts:

| seed | seed stage (q=60) | scale 2 (q=1,600) | scale 3 (q=240,000) | parent m |
|---|---:|---:|---:|---:|
| 99 | 5 | 1,600 | 240,000 | 477,584 |
| 1 | 6 | 1,600 | 240,000 | 481,428 |
| 2 | 6 | **1,599** | 240,000 | 481,366 |
| 3 | 6 | 1,600 | 240,000 | 481,428 |
| 4 | 6 | **1,599** | 240,000 | 481,366 |

Seeds 2 and 4 fall one closure short at scale 2 as well, which is why they give
481,366 rather than 481,428: `119,164 + 6*3,844 + 1,599*62 + 240,000`. Only
scale 3 fills for every seed.

Caveat worth knowing: ~50% is close to what the wedge geometry forces. Both
endpoints are neighbours of a common vertex, so both lie outside that vertex's
colour class — two draws from two classes collide about half the time. The
constraint is provably active on every second candidate, but this is not
evidence of unusual pressure.

### Breaking it costs coarse-relation coverage

Admitting a monochromatic closure with probability `f`, changing nothing else.
Five random seeds per row. **Relations** means distinct supernode pairs joined
by at least one parent edge, i.e. edges of the strict quotient.

> **CORRECTED 2026-09-25.** The table that stood here was the PRE-FIX sweep,
> `results_chromatic_ablation.json` — the file this README's own file list
> describes as "not used by the paper". It was never updated when the sampler bug
> was found and the sweep rerun on 2026-09-23. Four conclusions drawn from it
> were wrong; all four are corrected below, and the paper already carries the
> corrected numbers. Regenerate this table with
> `python3 aggregate_sweep.py results_chromatic_ablation_shuffled.json`.

| f | monochromatic | parent edges | relations | full distance | shape distance |
|---|---:|---:|---:|---:|---:|
| 0.00 | 0 | 480,560 | 63,019 | 0.076 | 0.060 ± 0.004 |
| 0.01 | 8,152 | 484,503 | 59,424 | 0.096 | 0.065 ± 0.004 |
| 0.02 | 15,995 | 488,347 | 57,370 | 0.109 | 0.070 ± 0.004 |
| 0.05 | 33,215 | 495,266 | 55,392 | 0.125 | 0.080 ± 0.005 |
| 0.10 | 52,044 | 501,417 | **54,538** | 0.132 | 0.085 ± 0.005 |
| 0.20 | 76,644 | 506,798 | 55,312 | 0.129 | 0.084 ± 0.002 |
| 0.35 | 100,258 | 508,336 | 54,905 | 0.132 | 0.087 ± 0.004 |
| 0.50 | 118,788 | 510,642 | 55,470 | 0.129 | 0.085 ± 0.004 |
| 0.75 | 140,731 | 512,180 | 56,292 | 0.125 | 0.085 ± 0.004 |
| 1.00 | 156,717 | 512,180 | 56,812 | 0.122 | 0.085 ± 0.006 |

**Not monotone.** The quotient reaches its minimum near `f = 0.10` (54,538) and
then partly recovers, to 56,812 at `f = 1`. The previous claim — "strictly
decreasing at every one of the ten steps", separating the endpoints by "18.6
standard deviations" — was an artifact of the sampler bug and is withdrawn.

**The loss is 9.8%, not 22%.** 63,019 relations constrained against 56,812 fully
relaxed.

**It is still not a density effect, and the direction still proves it.**
Relaxing the constraint ADDS 31,620 parent edges (480,560 → 512,180) while the
quotient LOSES 6,207 relations. A graph that grew cannot be sparser by accident:
the extra edges thicken coarse edges that already exist instead of opening new
ones.

**Shape agreement is better when constrained, by about four standard
deviations.** 0.060 ± 0.004 constrained against 0.085 ± 0.006 fully relaxed, on
the six non-density measures. Under the corrected sampler all five shape
measures are closer to the reference when the constraint holds, including the
two that previously looked like counterexamples: degree s.d. 60.92 → 53.88
against a reference 71.58, and transitivity 0.5037 → 0.4990 against 0.5578 —
both FARTHER without the constraint, not closer.

> **Three earlier readings of this data are withdrawn.**
> (1) "Unconstrained closure is 1.9x further from the reference" — an artifact of
> a distance counting edge count, density and mean degree as three measures when
> at fixed n = 1,015 they are one.
> (2) "On shape metrics the endpoints are indistinguishable, so the constraint
> has no demonstrated topological payoff" — false under the corrected sampler;
> the constraint HELPS shape, by ~4 sd.
> (3) The monotonicity and the 18.6-sd separation, as above.

**What the sampler bug was.** `close_relaxed` ended each round with
`np.unique(...)[:remaining]`. `np.unique` returns SORTED keys, the key is
`min*n + max`, and in this construction both colour and block are read off the
leading base-20 digit of the vertex index — so truncating to the quota kept the
lowest-indexed endpoints and discarded whole high-index blocks, which are the
monochromatic-rich ones by arithmetic. At `f = 0` the chromatic filter holds the
pool at or under quota, so the truncation fires only on end-of-quota remainders
worth 2–3 edges in 477,584; the overflow, and therefore the error, grows with
`f`. The fix permutes the candidates before truncating. Files produced after it
carry a `stage_closures` field, which is how to tell them apart.

> **What this does not show.** Not that the constraint is necessary or
> sufficient for connectome-like topology. The paper's claim is compatibility
> plus a measured association: a graph held to chi = 3 throughout growth, with
> the constraint demonstrably active, can still sit under a connectome-like
> quotient, and relaxing it costs coarse relations and shape agreement.

## How to run

Requires the same environment as `construct.py` (numpy, scipy, networkx).
Pin `networkx==3.5` — Louvain drifts across versions even at a fixed seed.

**Check the control reproduces the deposited graph** (~25 s). Do this first;
everything else is meaningless if it fails.

```bash
python3 chromatic_ablation.py --control
```

Expected, and asserted by the script:

```
f=0 seed=99:  parent 477,584 edges, 0 monochromatic, quotient 1015/64,760
distance to reference 0.0644
REPRODUCES THE DEPOSITED CONSTRUCTION: True
```

**Measure the bind rate** (~25 s):

```bash
python3 bind_rate.py
```

**Run the ablation sweep** (50 runs, ~25 s each, ~21 min):

```bash
python3 chromatic_ablation.py            # full grid -> results_chromatic_ablation_shuffled.json
python3 chromatic_ablation.py --f 0 0.05 1.0 --seeds 99 1   # a quick subset
```

**Rebuild the quotients and audit them** (~25 s each to build; the audits run
in parallel):

```bash
python3 build_seed_quotients.py                # writes quot_seed{99,1,2,3,4}.npz
cd ../graphmetrics
for s in 99 1 2 3 4; do                        # 99 included: it is the deposited case
  python3 connectome_audit_gold.py --graph ../ablation/quot_seed$s.npz \
      --n-null 100 --n-null-rc 1000 --seed 42 \
      --json ../ablation/audit_quot_seed$s.json \
      > ../ablation/audit_quot_seed$s.log 2>&1 &   # stdout must be redirected,
done                                               # or the grep below finds nothing
wait
grep -A11 "EVIDENCE SCORECARD" ../ablation/audit_quot_seed*.log
```

About 1 h 40 m each; five in parallel finish in about that. The **parent**
audit is four times the work (25,303 s for 8,000 nodes / 477,584 edges), so do
not launch one casually.

## Files

| file | what |
|---|---|
| `chromatic_ablation.py` | the sweep; `--control` reproduces the deposited parent. Reports BOTH the nine-measure distance and the six-measure shape distance |
| `fiedler_vs_bilateral.py` | does the Fiedler bisection *identify* the bilateral split, or only match its cut cost? Reports ARI/NMI and the Fiedler-vector correlation |
| `bind_rate.py` | instruments closure and counts colour rejections |
| `build_seed_quotients.py` | builds `quot_seed*.npz` for auditing |
| `results_chromatic_ablation_shuffled.json` | the 50-run sweep — **the source of the paper's ablation table** |
| `results_chromatic_ablation.json` | the same sweep before the sampler fix in `close_relaxed` (candidates are now permuted before the quota truncation). Kept as a record; not used by the paper. `--out` no longer defaults to this path, and the script refuses to overwrite an existing output without `--force` |
| `audit_quot_seed{99,1,2,3,4}.json` | nine-criterion audits, all 9/9 |

**On code duplication, stated plainly.** `build_seed_quotients.py` does exec
`chromatic_ablation.py` and shares its `grow()`. `bind_rate.py` does **not** —
it carries its own instrumented copy of the closure loop, so there are two
hand-copies of `construct.py`'s `close_triangles` in this folder, and all three
scripts reimplement `grow()`. Only `build_seed`, `coarsen`, `build_coarse`,
`COLOURS`, `BLOCKS` and `BLOCK_TARGETS` are imported from `construct.py`.

The copies are therefore verified empirically rather than guaranteed
structurally: `chromatic_ablation.py --control` asserts the deposited parent
and quotient, and `bind_rate.py` prints `n=8,000 m=477,584 monochromatic=0`
with per-stage accepts 5 / 1,600 / 240,000, matching a direct trace of
`construct.py`. If `close_triangles` is ever changed, both copies must be
updated by hand and both checks rerun.
