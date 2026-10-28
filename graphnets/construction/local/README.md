# The local-repair construction

`emergence_localrepair_standalone.py` builds the paper's *local* graph end to end
in a single file — fine graph, coarsening, and self-verification. Its only
project import is `../odd_cycle_witness.py`, the odd-cycle invariant shared with
the global construction.

| output | nodes | edges |
|---|---|---|
| `localrepair_parent.npz` | 16,807 | 277,319, with its three colour masks, so the audit uses the colour-respecting null |
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

`../../Budapest/budapest_connectome.gml` is read once, in `main()`, *after* the
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

The fine parent is identical on every run (277,319 edges), because its repair
and closure steps use a fixed pseudorandom order in the code. That order is
not incidental: different orders give a family of parents with the same
guarantees but different coarse statistics (see "The repair/closure order"
below, and the main paper, local quotient section). The table below varies
only the Louvain seed, `louvain_communities(..., resolution=40, seed=42)` in
`coarsen()`. Repeating just that step under eleven seeds (42, plus 0–9), with
the parent rebuilt each time:

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

Across Louvain seeds, structure is stable to under 1%. **Seed 42 is the best of the eleven on all three
statistics the paper quotes** (KS, Q, six-test score), so quote the ensemble and
name seed 42 as the deposited instance, not a typical one. `EXPECT` in this file
records seed 42's values and is a reproduction check for *this* seed — it will
fail under any other, by design.

`dmin=3` and the bundle maximum are seed-specific and are not properties of the
method.

The sweep script (`local_seedsweep.py`, which imports this file rather than
editing it) and its per-seed results are not shipped in this repository.

### The repair/closure order (measured 2026-10-28)

The table above varies only the Louvain seed. The growth also uses a fixed
pseudorandom order: `triangle_prune_local` (degree repair) and
`add_chromatic_to_budget` (closure) each call `np.random.default_rng(42)`,
hard-coded. That order sets which repair attachments are tried first and the
order in which wedge centres are scanned in closure. Different orders give a
family of parents with the same guarantees but different coarse statistics,
and the order changes the graph far more than the Louvain seed does. Changing
only the order (the value passed to `default_rng`), with the Louvain seed held
at 42:

| order (`default_rng` value) | parent edges shared with order 42 | quotient edges | degree sd | max degree | KS vs Budapest | Q |
|---|---|---|---|---|---|---|
| 42 (this file) | 100% | 67,094 | 68.1 | 502 | 0.111 | 0.524 |
| 0    | 6.2%   | 55,763 | 48.3 | 346 | 0.242 | 0.579 |
| 1    | 6.2%   | 61,094 | 50.6 | 335 | 0.221 | 0.594 |
| 2    | 3.6%   | 53,902 | 45.3 | 309 | 0.267 | 0.589 |
| 3    | 2.4%   | 53,864 | 45.3 | 303 | 0.279 | 0.616 |
| 4    | 5.2%   | 44,094 | 27.0 | 226 | 0.433 | 0.682 |
| 5    | 8.7%   | 52,595 | 38.7 | 290 | 0.300 | 0.628 |
| 6    | 8.9%   | 54,307 | 43.6 | 315 | 0.260 | 0.621 |
| 7    | 6.3%   | 57,491 | 54.9 | 433 | 0.229 | 0.533 |
| 8    | 6.2%   | 50,020 | 37.6 | 248 | 0.321 | 0.660 |
| 9    | 6.4%   | 59,962 | 56.4 | 390 | 0.210 | 0.510 |
| 10   | 7.2%   | 54,055 | 53.6 | 278 | 0.215 | 0.584 |
| 11   | 8.6%   | 63,826 | 68.3 | 443 | 0.137 | 0.499 |
| 100  | 7.8%   | 54,953 | 38.6 | 266 | 0.285 | 0.644 |
| 1000 | 5.2%   | 49,006 | 35.7 | 236 | 0.350 | 0.643 |

Every parent has exactly 277,319 edges, is connected and non-bipartite, has no
monochromatic edge and has minimum degree 3. Of the fifteen, order 42 is the
closest to Budapest. Its KS of 0.111 and its degree spread are properties of
this member of the family, not of the construction.

Nine-criterion audits (100 degree-preserving nulls, 1,000 rich-club nulls) of
five other orders, chosen to span the KS range:

| order (`default_rng` value) | KS vs Budapest | audit | connector hubs |
|---|---|---|---|
| 42 (this file) | 0.111 | 9/9 | 4 |
| 11 | 0.137 | 9/9 | 5 |
| 9 | 0.210 | 8/9 | 0 |
| 1 | 0.221 | 8/9 | 0 |
| 3 | 0.279 | 8/9 | 0 |
| 4 | 0.433 | 9/9 | 1 |

The three 8/9 results fail only `connector_hubs`; the other eight criteria pass
under every order audited. Order 3's failure does not depend on the Louvain
run: under 50 Louvain seeds no node reaches z >= 2.5 at all. The connector
hubs, like the degree fit, are a property of the deposited member of the
family.
