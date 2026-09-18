# The local-repair construction

`emergence_localrepair_standalone.py` builds the paper's *local* graph end to end
in a single file — fine graph, coarsening, and self-verification. Its own header
says: "No imports from any other project file."

| output | nodes | edges |
|---|---|---|
| `localrepair_parent.npz` | 16,807 | fine parent |
| `localrepair_quotient.npz` | 1,015 | 67,094 |
| `localrepair_quotient_audit.npz` | 1,015 | same, colourless layers, ready for the audit |

## Recipe (from the script's own docstring)

```
seed        modular (G2, degrees 2-3)
steps       4                    parent = 7^5 = 16,807 nodes
prune-frac  0.10                 delete to 0.10 * gmean, keeping top-triangle edges
add-frac    0.18                 add chromatic triangle-closing edges to 0.18 * gmean
repair      LOCAL 2-hop          attach low-degree nodes inside their own
                                 neighbourhood -> keeps modularity and a low tail
```

Then coarsen to 1,015 super-nodes (Louvain + density-merge) and verify both
levels against recorded numbers — parent edge count plus seven quotient statistics.

Unlike the global pipeline, the parameters are **in the file**, not in the
invocation, so running it bare reproduces the paper's graph.

## Inputs

**None.** The graph is grown from a hard-coded seed; nothing is read to build it.

`../../Budapest/budapest_connectome.gml` is read once, at line 174, *after* the
graph is complete, solely to compute the Kolmogorov-Smirnov distance between the
quotient's degree distribution and Budapest's. That value appears only in the
console summary — none of the three saved `.npz` files depend on it. Remove those
two lines and the graphs come out byte-identical, minus the KS number.

## Don't confuse this with the sibling variant

`emergencesimply/modular_run/REPRODUCE.txt` documents a *different* graph from the
same lineage: `build_emergence_simple.py` + `coarsen_to_edges.py`, prune-frac 0.12,
**global random** repair, giving **954 nodes / 59,579 edges**. That is not the
paper's local graph. The paper's is 1,015 / 67,094, prune-frac 0.10, **local
2-hop** repair, from the script in this folder.

## Before you run it

Pin `networkx==3.5` — the coarsening uses `louvain_communities`, which drifts
across versions even at fixed seed 42.

## Seed sensitivity (measured 2026-09-16)

The fine parent is deterministic — 277,319 edges on every run. The only
stochastic step is `louvain_communities(..., resolution=40, seed=42)` in
`coarsen()`. Repeating just that step under eleven seeds (42, plus 0–9), with the
parent rebuilt each time:

| measure | seed 42 (this file) | seeds 0–9: mean ± sd (range) |
|---|---|---|
| nodes | 1,015 | 1,015 ± 0 (fixed by `TARGET`) |
| edges | 67,094 | 66,700 ± 629 (65,358–67,755) |
| density | 0.1304 | 0.1296 ± 0.0012 |
| degree sd | 68.13 | 69.58 ± 0.46 |
| min degree | 3 | 2 in five of ten |
| max degree | 502 | 482.1 ± 12.4 (453–496) |
| KS vs Budapest | 0.1113 | 0.1295 ± 0.0075 (0.1182–0.1419) |
| modularity Q | 0.5245 | 0.5087 ± 0.0205 (0.4508–0.5198) |
| six-test score | 0.3919 | 0.3980 ± 0.0028 (0.3930–0.4022) |
| bundle mean / max | 2.870 / 918 | 2.881 ± 0.029 / 897 ± 141 (698–1,200) |
| bundle Gini | 0.5434 | 0.5458 ± 0.0032 |

Structure is stable to under 1%. **Seed 42 is the best of the eleven on all three
statistics the paper quotes** (KS, Q, six-test score), so quote the ensemble and
name seed 42 as the deposited instance, not a typical one. `EXPECT` in this file
records seed 42's values and is a reproduction check for *this* seed — it will
fail under any other, by design.

`dmin=3` and the bundle maximum are seed-specific and are not properties of the
method.

Sweep script and per-seed results: `~/Downloads/paper_data/sweeps/seed_sensitivity/`
(`local_seedsweep.py` imports this file rather than editing it).
