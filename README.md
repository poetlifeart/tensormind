# tensormind

Code to build the graphs in *Causal Event Networks* and measure them.

Everything is grown from a hard-coded seed. The only input file in this
repository is the Budapest reference connectome, used for comparison.

## Supporting material

The paper has no separate Supporting Information file; it points here instead.
The full construction procedures, parameter accounting, null-model definitions,
extended audits and model specifications are in:

- **[`docs/repository_notes.tex`](docs/repository_notes.tex)** — *Repository
  notes for "Chromatically Constrained Tensor Graphs: Concurrency and Connectome
  Topology"*. Figures it uses are in `docs/figures/`.

To build the PDF:

```bash
cd docs && pdflatex repository_notes.tex && pdflatex repository_notes.tex
```

Needs `tcolorbox` (Debian/Ubuntu: `sudo apt install texlive-latex-extra`).

```
graphnets/                      build the graphs and measure them
  Budapest/                     budapest_connectome.gml   1,015 nodes / 70,654 edges
  construction/
    bilateral/                  THE 8,000-NODE SEED-ENCODED CONSTRUCTION
                                  -- one of the two the main article emphasises
      blockwise/                  the same parent, coarsened by Louvain alone
      bundles/                    fibre-bundle post-processing
    local/                      the local prune-repair construction
                                  -- the other one the main article emphasises
    global/                     the integrated parent, and the earlier
                                  quotient/lift path
    finalgraph/                 parent -> degree-matched quotient. The optional
                                  fibre-bundle refinement; the supplementary
                                  notes mark it exploratory
  graphmetrics/                 seven metric scripts
  ablation/                     does the chromatic constraint do anything?
  expressiveness/               grammar sweep over seeds, chi and closure

graphdynamics/                  the inpainting models
  v1003/                        Graph + supernode attention  <- the reported arm
  feeder1004/                   Graph, attention removed (the ablation arm)
  unet/                         U-Net baseline (takes no graph)
  v1004bundle/                  Graph with bundle-map message passing
```

**Which construction the article reports.** The main text emphasises the
`local/` prune-repair construction and the `bilateral/` 8,000-node seed-encoded
construction. The `finalgraph/` relocation pipeline is retained as an optional
refinement, for the case where fibre-bundle multiplicity is targeted explicitly;
the supplementary notes head those sections "Exploratory; not part of the main
article's reported results". This README described `finalgraph/` as "THE PAPER'S
GRAPH" and omitted `bilateral/`, `ablation/` and `expressiveness/` entirely until
2026-09-25, having been written before the article was reorganised.

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

## Step 0. The 8,000-node construction  (independent of everything else)

```bash
cd graphnets/construction/bilateral
python3 construct.py             # build and verify parent + quotient  (~3 min)
python3 export_graphs.py         # write graphs/ -- needed by everything downstream
```

Self-contained: it reads no file, and grows both graphs from the 20-vertex seed
in `construct.py`. `construct.py` verifies as it goes but writes nothing;
`export_graphs.py` writes the four files that `stats.py`, `bundles/bundles.py`,
`make_figures.py` and the inpainting models read. They are `.gitignored` because
they are regenerable, so a fresh checkout must run this first.

Outputs, in `bilateral/graphs/`:
- `cnew_parent.npz` — **8,000 / 477,584**, three tripartite masks
- `cnew_parent_supernode.npz` — the same plus the 1,015-supernode map. This is
  the substrate the recurrent models use; copy it to
  `graphdynamics/graph_8k_parent_supernode.npz`
- `cnew_coarse.npz` + `cnew_coarse_w.npy` — **1,015 / 64,760** and its bundle
  weights

See `bilateral/README.md`.

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
| **8k parent** | **8,000** | **477,584** | `bilateral/construct.py` | **reported** |
| **8k quotient** | **1,015** | **64,760** | `bilateral/construct.py` | **reported** |
| block-wise quotient | 1,076 | 73,152 | `bilateral/blockwise/` | comparison |
| global parent | 16,807 | 337,983 | step 1 | input to everything |
| final parent | 16,807 | 337,980 | step 2 | optional refinement |
| final quotient | 1,015 | 71,098 | step 2 | optional refinement |
| pre-swap quotient (v14) | 1,015 | 70,537 | step 3 | superseded |
| post-swap quotient (v146) | 1,015 | 70,538 | step 3 | superseded |
| feeder lift | 16,807 | 344,483 | step 3 | superseded |
| local parent | 16,807 | 277,319 | step 4 | comparison |
| local quotient | 1,015 | 67,094 | step 4 | comparison |
| Budapest (shipped) | 1,015 | 70,654 | — | reference |

---

# Part 2 — Measure them

Seven scripts, one interface: **one graph in, one JSON out.**

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

