# tensormind

Code to build the graphs in *Causal Event Networks* and measure them.

Everything is grown from a hard-coded seed. The only input file in this
repository is the Budapest reference connectome, used for comparison.

```
graphnets/                      build the graphs and measure them
  Budapest/                     budapest_connectome.gml   1,015 nodes / 70,654 edges
  construction/
    global/                     the parent, and the earlier quotient/lift path
    finalgraph/                 THE PAPER'S GRAPH: parent -> matched quotient
    local/                      the local-repair construction
  graphmetrics/                 six metric scripts

graphdynamics/                  the inpainting models
  v1003/                        Graph+attn.  <- the reported arm
  feeder1004/                   Graph  (v1003 minus supernode attention)
  unet/                         U-Net baseline
```

The two halves are independent. `graphnets/` needs nothing but Python;
`graphdynamics/` ships its graph already built and needs CelebA-HQ and a GPU.
See `graphdynamics/README.md` for that side.

## Requirements

```bash
pip install numpy scipy networkx==3.5 matplotlib
pip install pynauty        # needed only for orbit entropy in run_complexity.py
```

**Pin `networkx==3.5`.** Every coarsening step calls `louvain_communities`, which
gives slightly different partitions on different networkx versions even at fixed
seed 42. Other versions produce a near-identical but not bit-identical graph.

Reference environment: python 3.13.5, numpy 2.3.4, networkx 3.5, scipy 1.16.2.

---

# Part 1 — Build the graphs

## Step 1. The global parent  (required first)

```bash
cd graphnets/construction/global
bash build_mild_graph.sh
```

Three stages, ~seed 42 throughout, deterministic:

| stage | script | edges |
|---|---|---|
| 1 | `build_graph_conscious.py --steps 4 --inter-ratio 0.35 --add 20168` | 273,695 |
| 2 | `graph_bridge_paper.py --max-add 84000` | 357,695 |
| 3 | `graph_sparsify_memb.py --target-db 3.8 --batch 20000 --min-edges 100000 --min-degree 2` | **337,983** |

Output: `construction/global/graph_brain_mild.npz` — **16,807 nodes / 337,983 edges**.

The flags matter. The scripts' own defaults are different (`--inter-ratio`
defaults to 0.3, `--add` to 10000); running them bare gives a different graph.
`build_mild_graph.sh` has the right values, so use it.

See `construction/global/README.md` for the full stage-by-stage description.

## Step 2. The final graph pair  (required — this is what the paper reports)

```bash
cd graphnets/construction/finalgraph
python3 step1_match_bundles.py        # relocation + Budapest bundle match
python3 step2_symmetric_swap_lift.py  # 6,500-edge swap + Feeder inversion
python3 step4_degree_match.py         # degree-sequence refinement
```

Outputs, in that directory:
- `parent_degmatch_masks.npz` — **16,807 / 337,980**, the final parent, carries
  the 1,015-supernode map
- `graph_degmatch.npz` + `degmatch_w.npy` — **1,015 / 71,098**, the final
  quotient and its bundle weights

The quotient is *derived* from the parent by counting cross-supernode edges,
never authored. Full description, endpoint rules, results and caveats in
`finalgraph/README.md`.

(There is no `step3`; the numbering is historical. A bridge-protected variant
occupied that slot and was dropped — see `finalgraph/README.md`.)

## Step 3. The earlier coarsening  (no longer how the quotient is obtained)

**The global parent is still required — it is the input to step 2 — but it is no
longer coarsened to Budapest this way.** These scripts applied a tiered selection
that displayed 70,538 of the 143,679 supernode pairs holding at least one parent
edge. That is what `finalgraph/` replaces: not showing a pair means deleting its
parent edges, and deleting all 101,816 of them disconnects the parent into three
components. `finalgraph/` relocates them instead.

One thing here is still needed. `save_v146_npz.py` produces the witness weights
and the displayed-pair list that `finalgraph/inputs/v146_weights.pkl` holds, and
step1 reads them as its starting point. The shipped `.pkl` means you do not have
to rerun it, but this is where it came from.

```bash
python3 graphnets/construction/global/quotient/save_v146_npz.py
python3 graphnets/construction/global/quotient/lift/invert146_feeder.py
```
- `quotient/graph_v14.npz` — 1,015 / 70,537 (pre-swap)
- `quotient/graph_v146_6500.npz` — 1,015 / 70,538 (post-swap)
- `quotient/lift/graph_brain_mild_v146_feeder.npz` — 16,807 / 344,483 (feeder lift)

The feeder lift was the substrate for the earlier inpainting runs. **The models
now train on the final parent** — shipped as
`graphdynamics/graph_degmatch_parent_supernode.npz`, identical to
`finalgraph/parent_degmatch_masks.npz` with the supernode map attached.

## Step 4. The local construction  (independent of steps 1-3)

```bash
python3 graphnets/construction/local/emergence_localrepair_standalone.py
```

One self-contained script: builds its own 16,807-node parent, coarsens it, and
verifies both against recorded numbers. Outputs:
- `local/localrepair_parent.npz` — 16,807 / 277,319
- `local/localrepair_quotient.npz` — 1,015 / 67,094
- `local/localrepair_quotient_audit.npz` — same graph, colourless layers

Needs nothing from the global pipeline. Can run first, last, or alongside.

## What you end up with

| graph | nodes | edges | built by | status |
|---|---|---|---|---|
| global parent | 16,807 | 337,983 | step 1 | input to everything |
| **final parent** | **16,807** | **337,980** | **step 2** | **reported** |
| **final quotient** | **1,015** | **71,098** | **step 2** | **reported** |
| pre-swap quotient (v14) | 1,015 | 70,537 | step 3 | superseded |
| post-swap quotient (v146) | 1,015 | 70,538 | step 3 | superseded |
| feeder lift | 16,807 | 344,483 | step 3 | superseded |
| local parent | 16,807 | 277,319 | step 4 | comparison |
| local quotient | 1,015 | 67,094 | step 4 | comparison |
| Budapest (shipped) | 1,015 | 70,654 | — | reference |

---

# Part 2 — Measure them

Six scripts, one interface: **one graph in, one JSON out.**

```bash
python3 graphnets/graphmetrics/connectome_audit_gold.py \
    --graph GRAPH --json audit.json --n-null 100 --n-null-rc 1000

python3 graphnets/graphmetrics/run_complexity.py        --graph GRAPH --json complexity.json
python3 graphnets/graphmetrics/run_controllability.py   --graph GRAPH --json controllability.json
python3 graphnets/graphmetrics/clique_census.py         --graph GRAPH --json cliques.json
python3 graphnets/graphmetrics/combinatorial_metrics_npz.py --graph GRAPH --json combinatorial.json
python3 graphnets/graphmetrics/richclub_other_graphs.py --graph GRAPH --json richclub.json
```

`--graph` accepts `.npz` or `.gml` and is required in all six. Give each run its
own `--json`.

| script | measures |
|---|---|
| `connectome_audit_gold.py` | the nine criteria vs degree-preserving nulls |
| `run_complexity.py` | graph energy, Estrada, Kirchhoff, off-diagonal complexity |
| `run_controllability.py` | average and modal controllability |
| `clique_census.py` | clique number, histogram peak, maximal cliques |
| `combinatorial_metrics_npz.py` | vertex cover, matching, Hoffman bound, bisection |
| `richclub_other_graphs.py` | rich-club curve, floored at 20 nodes |

Two things to know before running the audit, both in
`graphnets/graphmetrics/README.md`:

- Audit the local quotient with **`localrepair_quotient_audit.npz`**, not
  `localrepair_quotient.npz`. The audit picks its null model from the file, and
  the `_audit` file is the one with colourless layers.
- The paper's rich-club **counts** come from `richclub_other_graphs.py` (floored).
  The audit's own rich-club count is unfloored and will not match. The audit's
  PASS/FAIL verdicts are correct and reproducible on their own.

## Runtimes

The audit dominates everything else.

| graph | nulls | time |
|---|---|---|
| 1,015-node quotient | 100 + 1,000 rich-club | ~2 hours |
| 16,807-node parent | 100 | hours |
| 16,807-node parent | 1,000 | days |

The other five metrics are minutes at 1,015 nodes.

on CPU   Intel Core i9-9940X — 14 cores / 28 threads, 3.3 GHz (4.5 boost), 19.7 MB L3
RAM   62 GB (33 GB available). 

