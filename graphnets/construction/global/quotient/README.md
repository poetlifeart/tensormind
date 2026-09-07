# Coarsening the parent to the Budapest scale (1,015 nodes)

`save_v146_npz.py` takes the mild parent graph (16,807 / 337,983, built one level
up) and produces **both** quotients used by the paper:

| output | nodes | edges | what it is |
|---|---|---|---|
| `graph_v14.npz` | 1,015 | 70,537–70,538 | pre-swap quotient |
| `graph_v146_6500.npz` | 1,015 | 70,538 | post-swap — most quotient tables use this |

Run it after the parent exists. It is one self-contained script; there is no
separate coarsening step to run first.

## What the coarsening does

1. **Group.** Louvain on the parent at `resolution=20.0`, seed 42 → 1,054
   communities. Merge the smallest into its densest neighbour until 1,015 remain
   (39 merges). Each surviving community is one super-node (sizes 3–309, mean 16.6).
2. **Witness graph.** For each super-node pair, `W(a,b)` = the number of parent
   edges crossing between them → 143,679 candidate pairs, mean weight ≈2.4.
   This is the honesty constraint: a quotient edge is legal only if real parent
   edges support it. No edges are invented.
3. **Hierarchy.** L0 partition of the witness graph at `resolution=0.95` → 4
   blocks (389 / 265 / 239 / 122). Each block then gets L1/L2 sub-communities at
   its own resolutions (B0 0.9/1.0, B1 0.7/1.0, B2 1.0/2.0, B3 1.0/2.0).
4. **Tiered selection.** 143,679 witness pairs would give mean degree ~283;
   Budapest is 139, so roughly half must be dropped. Edges are tiered by how far
   apart their endpoints sit in the hierarchy — T1/T2 within an L1 group, T3 same
   block across L1 groups (31,302, all kept), T4 cross-block (96 pairs, only the
   top 10 by witness weight). Fill to mean degree 139.

Result: **v14**, 70,537 edges, 83.2% intra-block, 16.8% cross-block, modularity 0.501.

Steps 1–4 are documented in full, with parameter tables and source, in
`emergentgraph/graph_topology/quotient_reproduction/reproduction.tex`.

## The swap (v14 → v146) — the least documented step in the pipeline

`save_v146_npz.py` lines 136–183. It is documented **nowhere else**: no README,
paper appendix, or note in emergentgraph describes it.

1. Take the T3 edges (same L0 block, different L1 group) and sort by witness
   weight ascending — weakest first.
2. Build a pool of **within-L1 non-edges**, scored by number of common
   neighbours (triangle-closing potential), sorted descending.
3. Remove the 6,500 weakest T3 edges; add the 6,500 best within-L1 candidates.
   Edge count is preserved by construction.
4. **Connectivity repair.** If the result is disconnected, add back the
   *strongest* removed T3 edges (highest witness weight first) until connected.

The effect is to trade long-range within-block edges for local triangle-closing
ones, concentrating connectivity near the diagonal to match Budapest's texture.

**Caveat:** because of step 4 the operation is not exactly "6,500 out, 6,500 in".
That, together with Louvain version drift, is the likely source of the
70,537-vs-70,538 discrepancy seen between recorded runs of the same graph.

## Before you run it

- **Pin `networkx==3.5`.** Every step here calls `louvain_communities`, which
  drifts across networkx versions even at fixed seed 42. An unpinned run gives a
  near-identical but not bit-identical graph.
- The script has **hard-coded absolute paths** for its input parent and its two
  outputs. Edit them for your machine.
