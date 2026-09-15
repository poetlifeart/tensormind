# finalgraph — the inverse relocation construction

Takes the global parent and produces **the graph pair the paper reports**: a
16,807-node chromatic substrate and the 1,015-node quotient derived from it.

```
graph_brain_mild.npz          16,807 / 337,983     the parent, built by ../global
        |
        |  step1_match_bundles.py         relocation + Budapest bundle match
        v
parent_bigswap3_masks.npz     16,807 / 337,980
        |
        |  step2_symmetric_swap_lift.py   6,500-edge swap + Feeder inversion
        v
parent_final_masks.npz        16,807 / 337,980
        |
        |  step4_degree_match.py          degree-sequence refinement
        v
parent_degmatch_masks.npz     16,807 / 337,980     <-- FINAL PARENT
graph_degmatch.npz            1,015  /  71,098     <-- FINAL QUOTIENT
degmatch_w.npy                bundle weight per quotient edge
```

Run them in order from this directory:

```bash
python3 step1_match_bundles.py      # ~5 min
python3 step2_symmetric_swap_lift.py
python3 step4_degree_match.py       # ~45 s
```

Each writes into this directory and the next reads from it. `step1` needs
`../global/graph_brain_mild.npz`, so build the parent first
(`cd ../global && bash build_mild_graph.sh`).

There is no `step3`. The numbering is historical: a bridge-protected variant
occupied that slot, was measured, and was dropped. See *Discarded variant* below.

---

## The idea

The published quotient displays 70,538 supernode pairs, but collapsing the
parent honestly gives 143,679 pairs holding at least one edge. The difference
was not shown. Because a quotient edge exists only where parent edges lie
beneath it, **not showing a pair means deleting its parent edges**, and at this
rate the substrate does not survive: deleting all 101,816 of them disconnects
the parent into 3 components and drops its minimum degree to 1.

So the edges are **relocated, not removed**. An edge may be invisible in the
quotient for exactly one reason — both endpoints lie in the same supernode, so
there is no pair to draw. That is geometry, not suppression. Everything else is
displayed.

This inverts the problem. Rather than coarsening a fixed substrate and reporting
what emerges, the substrate becomes the unknown: Budapest supplies the target
distributions, fine-edge placement is changed to realise them, and **the
quotient is always derived afterwards, never authored**:

```python
for u, v in parent.edges():
    a, b = sid[u], sid[v]
    if a != b:
        W[(min(a,b), max(a,b))] += 1
```

Verified both directions on the final pair: recomputing from
`parent_degmatch_masks.npz` reproduces `graph_degmatch.npz` edge-for-edge and
`degmatch_w.npy` weight-for-weight, and
147,217 cross-supernode + 190,763 intra-supernode = 337,980 = the whole parent.

---

## What each step does

### step1 — relocation and bundle match
* Every hidden pair's parent edges are removed; the count is re-spent inside
  supernodes at the end, so the parent keeps its edge count.
* Displayed pairs are rank-matched to Budapest's `fiber_count_mean`: pairs
  ordered by current weight, an equal-size sample of Budapest weights ordered
  the same way, paired rank for rank, integer targets with a floor of 1.
* **Endpoint rule.** One quantity drives every choice,
  `SUPPORT(u,v) = |N(u) ∩ N(v)|`, the triangles the edge closes.
  Trimming removes the *lowest*-support edges; densifying adds the
  *highest*-support non-edges; the surplus fills supernode interiors
  highest-support first. Every added edge satisfies `col[u] != col[v]`.
* Repairs: vertices below degree 3 gain their highest-support two-hop
  neighbour, falling back to a uniform random cross-colour vertex; disconnected
  components are rejoined.

**Counts for this step** (reproduced from a clean checkout 2026-09-14):
hidden pairs 73,141 → 101,816 parent edges relocated; trimmed 59,126;
densified +14,209; degree repair +5,170 over 4,220 vertices; connectivity
repair +0; surplus into interiors 141,563; parent 337,983 → 337,983,
monochromatic 0, connected, non-bipartite.

### step2 — symmetric swap and Feeder inversion
The published 6,500-edge swap, applied on top of the bundle-matched graph.
6,500 weight-1 coarse edges are removed and 6,500 new ones added, each backed by
exactly one chromatically legal parent edge. **Net effect on the weight multiset
is zero** — it is symmetric, so the removed edges lose their parent support too.
This is why the bundle match established in step1 survives to the end.

### step4 — degree-sequence refinement
Budapest's quotient degree sequence is rank-matched onto the supernodes.
Supernodes above target shed their weakest-bundle partners (never starving a
partner already at its own target); those below gain partners; the remainder
goes inside supernodes. About 2,146 edges, 0.6% of the parent.

---

## Results

| | final quotient | Budapest |
|---|---|---|
| nodes / edges | 1,015 / 71,098 | 1,015 / 70,654 |
| modularity | 0.4908 | 0.5574 |
| clustering | 0.4724 | 0.6696 |
| average path length | 1.9811 | 2.2190 |
| **bundle mean** | **2.071** | 2.033 |
| **bundle max** | **155** | 154.87 |
| **bundle Gini** | **0.306** | 0.305 |
| **bundle edges > 18** | **41** | 40 |
| nine-criterion audit | **9/9**, 1,000 nulls | — |
| six static tests | 0.3033 (best degree match, T1 KS 0.0187) | — |

| | final parent | original parent |
|---|---|---|
| clustering | 0.2281 (14.6x) | 0.0366 (2.29x) |
| modularity | 0.8104 (7.45x) | 0.4605 (4.23x) |
| sigma_dp | 15.37 | 1.68 |
| APL | 4.5114 (ratio **1.44**) | 3.7871 (ratio 1.217) |
| global efficiency | 0.2439 (ratio **0.729**) | 0.2844 (0.845) |
| fractal scaling | pass | pass |
| audit | **7/9**, 100 nulls | 9/9, 1,000 nulls |

The substrate trades the two near-random path criteria for a large gain in every
local and mesoscale property. It remains short-path in absolute terms: 4.49
hops, 93% of node pairs within six, against 210.6 for an equally sparse ring
lattice. It shares 50.3% of its edges with the original parent.

---

## Input files (`inputs/`)

| file | what |
|---|---|
| `bud_w.npy` | (4 × 70,654) Budapest edge attributes in `budapest_connectome.gml` edge order. **Row 0 is `fiber_count_mean` — the bundle target.** Rows 1–3 are `occurences`, `electrical_connectivity_median`, `fiber_length_mean`. Verified against the Netzschleuder source to 0 / 3e-18 / 1e-14. |
| `v146_weights.pkl` | `witness` (143,679 pair → weight), `v14` (the 70,538 displayed pairs), `v146`, `removed` (the 6,500 swapped). Produced by the tiered selection in `../global/quotient/save_v146_npz.py`. |

**Do not use row 1 of `bud_w.npy`.** `occurences` is the number of subjects
carrying the edge (max 418 of 477), not a fibre count, and its statistics
(mean 23.5, max 418) superficially resemble one.

Budapest source: `https://networks.skewed.de/net/budapest_connectome`,
Reference Connectome 3.0, `all_20k` variant, consensus over 477 HCP subjects.
Cite Szalkai et al., *Neurosci. Lett.* 595:60–62 (2015) and
*Cogn. Neurodyn.* 11(1):113–116 (2017).

---

## Choices that are arbitrary

State these as arbitrary rather than derived:

1. Supernode interiors are filled **largest-first**. The 309-member supernode
   has ~47,000 internal slots and fills before an 8-member one gets any.
   Proportional allocation is more defensible and untested.
2. `SUPPORT` ties break by vertex index, an artefact of sorting tuples.
3. step4 draws 60 uniform random candidate supernodes and takes the first
   admissible one. The least principled rule in the construction.
4. The degree-repair fallback places a uniform random cross-colour edge.

Everything else follows from one principle — prefer high triangle support when
adding, low when removing — under the chromatic constraint.

## Discarded variant

`step3` was a bridge-protected variant: the 12,717 zero-triangle-support
cross-supernode edges were exempted from relocation. It reads the original
parent directly and is **not** part of this chain. Result: the parent's APL
ratio improves from 1.435 to 1.288 — recovering 79% of the gap to the 1.25
threshold, the only lever found that helps — but it still fails, and the
quotient degrades badly (six-test 0.3936 against 0.3033, and 189 bundle edges
above weight 18 against 41). Kept only for that one measurement.

Its counts — 12,717 bridges protected, 53,750 trimmed, 697 repair edges,
127,943 into interiors — belong to **that variant only**. Do not quote them for
step1, whose figures are above.

## Determinism

`step1` and `step2` seed `default_rng(42)`, `step4` seeds `default_rng(11)`.
The partition is Louvain resolution 20.0, seed 42, merged smallest-into-
strongest-neighbour to 1,015 groups. **Pin `networkx==3.5`** — `louvain_communities`
gives different partitions across versions even at fixed seed.

---

## Reproduction check

Run from a clean checkout 2026-09-14, `networkx==3.5`, python 3.13.5 — the three
steps reproduce the reported graph exactly:

```
step1   trimmed 59,126   densified +14,209   repair +5,170 (4,220 vertices)
        surplus 141,563  parent 337,983      mono=0 connected non-bipartite
step2   KS 0.198  Gini 0.308  mean 2.06  max 155  >18: 41  Q 0.495  C 0.466
        parent 337,980   chi = 3 verified (triangle witness 0-9156-14093)
step4   shed 763 pairs (1,073 edges)  gained 159 (814 edges)  +259 interior
        quotient 71,098   bundle mean 2.071 max 155 gini 0.306 >18: 41
        Q 0.4908  C 0.4724  APL 1.9811  diam 4
```

Outputs total ~280 MB uncompressed and are `.gitignore`d — rebuild them by
running the three steps. `step1` takes ~25 s, `step2` ~25 s, `step4` ~45 s,
plus a few minutes for the Louvain partition each script recomputes.

**Known defect:** the final parent has minimum degree 1. The trim in step1 can
take a vertex below the degree-3 floor, and the repair runs before the trim.
The *original* parent also has minimum degree 2 with 638 vertices below 3, so
this is not a regression, but the degree-3 floor is a repair target in the code
rather than a satisfied invariant of either graph.
