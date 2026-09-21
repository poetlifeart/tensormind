# Graph metrics

The six scripts that produced the paper's graph-topology numbers. Each was
identified by matching the JSON keys it emits against the paper's result files —
not by filename or date. Several had multiple versions on disk that differ in
substance; these are the ones that made the published numbers.

| script | computes | feeds |
|---|---|---|
| `connectome_audit_gold.py` | nine-criterion audit vs degree-preserving nulls | all nine-criteria tables |
| `run_complexity.py` | graph energy, Estrada, Kirchhoff, off-diagonal complexity | complexity tables |
| `run_controllability.py` | average + modal controllability (Gu et al. 2015) | controllability tables |
| `clique_census.py` | clique number ω, histogram peak, maximal cliques | clique tables |
| `combinatorial_metrics_npz.py` | vertex cover, matching, Hoffman bound, spanning forest, balanced bisection | both combinatorial tables |
| `richclub_other_graphs.py` | rich-club curves, 1,000 nulls, `MINNODES=20` | rich-club figure and counts |
| `skeleton_fractality.py` | box-covering fractality of the graph **and of its skeleton** | fractality discussion |

## Usage — one graph in, one JSON out

All six behave identically and independently. Give one graph, get one JSON for
that graph. No defaults, no accumulation, no shared state.

```bash
python3 connectome_audit_gold.py     --graph G.npz --json audit.json --n-null 100 --n-null-rc 1000
python3 run_complexity.py            --graph G.npz --json complexity.json
python3 run_controllability.py       --graph G.npz --json controllability.json
python3 clique_census.py             --graph G.npz --json cliques.json
python3 combinatorial_metrics_npz.py --graph G.npz --json combinatorial.json
python3 richclub_other_graphs.py     --graph G.npz --json richclub.json
python3 skeleton_fractality.py       --graph G.npz --json skeleton.json
```

`--graph` accepts `.npz` or `.gml`. Give each run its own `--json`: a metric
writes the whole file, so pointing two runs at one path loses the first.

The paper's own result files are named `<metric>_<graph>.json` — for example
`complexity_budapest.json`, `clique_v14.json`, `audit_v146_1000.json`.

**Note on output shape.** Two of the archived paper JSONs are keyed by graph
(`combinatorial_v14_v146_budapest.json` holds `v14` / `v146_6500` / `Budapest`;
`richclub_other_graphs.json` holds all six). Those were produced by earlier batch
versions of these two scripts. The scripts here write one graph per file with the
metrics at the top level, so the numbers match but the nesting differs.

## The null model is chosen for you

`connectome_audit_gold.py` does not need to be told which null ensemble to use.
`_use_generic_null()` decides from the graph itself — its docstring: *"Decide null
model based on graph semantics, not file format."* A graph carrying real
tripartite masks gets tripartite-aware rewiring; an edge-list graph, or one whose
masks are placeholders, gets generic degree-preserving rewiring.

This is why `construction/local/` writes a third file,
`localrepair_quotient_audit.npz`, with `n1=nq, n2=0, n3=0` — colourless layers, so
the audit selects the generic null. **Feed the audit
`localrepair_quotient.npz` instead and you silently get a different null model
and different ratios.** Use the `_audit` file for auditing the local quotient.


## Rich club: which script gives the published numbers

> **In one line:** the paper reports the **floored** counts; the audit reports
> **unfloored** ones as an internal diagnostic; for the numbers as published, run
> `richclub_other_graphs.py`.

**The floor.** A degree threshold *k* is only counted if at least 20 nodes still
have degree > *k*. Past that point the "rich club" is a handful of nodes — eleven
at k=294 on the parent graph, two by k=302 — and the coefficient is not a
meaningful test there. The paper states this rule in its own caption: *"Threshold
counts are restricted to degree thresholds k for which at least 20 nodes have
degree >k, with significance assessed at p<0.05."*

Flooring can only remove thresholds, so the published counts are the smaller,
more conservative ones — 294 where 330 was available, 288 where 295 was.

**Every rich-club count printed in the paper is a floored value from
`richclub_other_graphs.py`** (`n_sig_floor` / `run_floor`), verified across all
six sites:

| graph | paper prints | unfloored | floored |
|---|---|---|---|
| local-repair quotient | 294 / run 177 | 330 / 177 | **294 / 177** |
| Budapest | 212 / run 212 | 271 / 212 | **212 / 212** |
| parent global | 288 / run 288 | 295 / 290 | **288 / 288** |
| feeder (runtime) | 291 / run 289 | 297 / 289 | **291 / 289** |
| post-swap quotient | 237 / run 199 | 279 / 199 | **237 / 199** |

**The audit is still correct and still sufficient for the verdicts.** Its scored
criterion is `longest_consecutive_run >= 3`, and the run is identical in both
scripts (199, 288, 289, 177, 212, 248). So `connectome_audit_gold.py` reproduces
all nine PASS/FAIL verdicts and every continuous metric on its own. Only the
threshold **counts** need the standalone.

Do not compare the audit's `n_k_significant` to the paper — it is unfloored and
will not match. The two scripts also differ slightly even before flooring (289 vs
279 on the post-swap quotient). They use the same significance test (rho > 1,
permutation p < 0.05) but build their k-grids differently: the audit tests only
thresholds present in the real graph AND in *every* null curve, while the
standalone tests every threshold in the real graph with a `nanmean` over whatever
nulls reach it. That difference has not been fully traced, and it affects no
published number.

**An older audit scored rich club differently.** The criterion is
`longest_consecutive_run >= 3` — a run of *consecutive* significant thresholds,
per van den Heuvel & Sporns 2011. A pre-May-2026 version scored
`n_k_significant > 0` (any single k). Results from it lack the
`longest_consecutive_run` field entirely, which is how to spot them.

## Off-diagonal complexity

`run_complexity.py` implements Claussen (2007), *Physica A* 375:365–373:
`OdC = -Σ a_k ln a_k`. An earlier version used an ad-hoc
`H × (1 - |2ρ - 1|)` in bits and emitted `degree_pair_entropy_bits`; if you see
that key, the file came from the wrong version.

## Dependencies beyond numpy/scipy/networkx

- `pynauty` — `run_complexity.py`, for orbit entropy. Needs the nauty C library.
- `python-louvain` (`import community`) — a fallback in `connectome_audit_gold.py`
  after `networkx.algorithms.community`; may not be needed.

Pin `networkx==3.5`: anything calling Louvain drifts across versions at fixed seed.

## Runtimes

The audit dominates. On the 16,807-node parent, 1,000 nulls is days; 100 nulls is
hours. On a 1,015-node quotient it is roughly 1.5–2 hours with 100 nulls plus
1,000 rich-club nulls.

## skeleton_fractality.py

Box-covering fractality measured twice: on the graph, and on its **skeleton** —
the maximum spanning tree by edge betweenness.

A fractal network in the sense of Goh, Salvi, Kahng & Kim (PRL 96, 018701,
2006) *is* a skeleton dressed with local shortcuts. The skeleton carries the
fractality; the shortcuts make the full graph small-world. Measuring the full
graph measures the dressed object, which is why dense low-diameter graphs come
out non-fractal even when their backbone is not.

Input:  `--graph` (`.npz` or `.gml`; `.npz` accepts either an `edges` array or
        the tripartite `mask_12/13/23` form).
Output: one JSON with a `full` and a `skeleton` block, each carrying `n`, `m`,
        `d_B`, `r2_power`, `r2_exp`, `gap`, `npts`, `radii`, `counts`,
        `diameter`, `giant_fraction`, and a `verdict` of `power law` /
        `exponential` / `too few points`.

    python3 skeleton_fractality.py --graph G.npz --json skeleton.json
    python3 skeleton_fractality.py --graph G.npz --json skeleton.json --pivots 256

`n` is the giant component, which is what everything is measured on; `m` is the
edge count of the whole input graph, so the two do not correspond when
`giant_fraction < 1`. `diameter` is the largest distance seen from 300 sampled
sources, hence a lower bound on the true diameter, and it is what sets the radius
ladder (`r = 1 … diameter/2`).

### The two fits

Box covering claims, for each radius `r`, the ball of that radius around an
uncovered vertex, and counts the balls needed. The fractal hypothesis is a power
law in the box size, and the non-fractal alternative is an exponential — the two
forms the script fits:

```
fractal        N_B(r)  ~  r^(-d_B)         fitted as   log N_B = -d_B log r + a
non-fractal    N_B(r)  ~  exp(-beta r)     fitted as   log N_B = -beta r   + b
```

Both are ordinary least squares on the same response `log N_B` over the same
radii, with the same number of parameters, so their `R²` values are directly
comparable. `gap = r2_power - r2_exp`: positive means the power law wins, which
is the fractal signature; negative means the exponential wins, which is the
small-world signature. `d_B` is `-1` times the power-law slope; `beta` is fitted
but not written to the JSON.

**One honest caveat on the abscissa.** Song, Havlin & Makse define the relation
on the box *size* `l_B` — every pair of nodes inside a box is closer than `l_B`.
This script fits on the ball *radius* `r`, and a ball of radius `r` has
`l_B ≈ 2r + 1`. The exponential fit is unaffected, since
`exp(-beta(2r+1))` differs from `exp(-2 beta r)` only by a constant factor, but
the power-law fit is **not** invariant under that affine shift, and the reported
`d_B` is a radius exponent. The two agree in the large-`r` limit; at the small
radii actually available on a low-diameter graph they need not, and the
power-versus-exponential comparison inherits that bias.

**Read `npts` before reading `gap`.** A box-covering fit on fewer than about
ten points is not evidence in either direction — that is how graphs with
provably bounded diameter get reported as fractal. The script prints a warning
below ten and refuses to fit below four.

`--pivots N` samples edge betweenness instead of computing it exactly. Exact is
O(nm) and is the default; above roughly 20,000 nodes sampling is required, and
the skeleton is then approximate. On a 1,015-node graph the two agree closely.

`--trials` sets how many ball-cover restarts are run per radius (the minimum is
kept, since the cover is a greedy heuristic). `--seed` fixes the RNG.

### References

- Song, Havlin & Makse, "Self-similarity of complex networks", *Nature*
  433:392–395 (2005). doi:10.1038/nature03248 — box covering, the relation
  `N_B(l_B) ~ l_B^(-d_B)` (their Eq. 3), and the pure exponential
  `N_B(l_B) ~ exp(-l_B/l_e)` reported for the non-fractal networks, which is the
  alternative `gap` scores against.
- Song, Gallos, Havlin & Makse, "How to calculate the fractal dimension of a
  complex network: the box covering algorithm", *J. Stat. Mech.* 2007:P03006
  (2007). doi:10.1088/1742-5468/2007/03/P03006 — the covering algorithms, and
  the mapping of minimum box covering to graph colouring. The greedy ball cover
  used here is a burning heuristic in that family, not a minimum cover, which is
  why `--trials` restarts and keeps the smallest count.
- Goh, Salvi, Kahng & Kim, "Skeleton and fractal scaling in complex networks",
  *Phys. Rev. Lett.* 96:018701 (2006). doi:10.1103/PhysRevLett.96.018701 — the
  skeleton as the spanning tree formed by the edges of highest betweenness
  centrality, and fractal networks as "a fractal skeleton dressed with local
  shortcuts", which is what measuring both blocks is for.
