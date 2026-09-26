# bilateral — the 8,000-node seed-encoded construction

A chromatically constrained tensor graph whose seed carries a four-block
bilateral organisation, and its two coarsenings to connectome scale.

The name is the point of the construction. The 20-vertex seed carries two
independent labellings: a proper 3-colouring, and a four-block partition
arranged as two hemispheres `{A,C}` and `{B,D}`. Both survive three Kronecker
powers. The colouring survives exactly — the parent is 3-chromatic with zero
monochromatic edges — and the blocks survive well enough to be recovered
blindly from the edge list alone at NMI = ARI = 1.

This folder is self-contained. It does not import from the rest of the
repository, and the only external input is the Budapest reference connectome
in `reference/`.

---

## Quick start

```bash
cd graphnets/construction/bilateral

python3 construct.py                 # build and verify the parent + quotient   (~3 min)
python3 export_graphs.py             # write graphs/ -- REQUIRED BEFORE THE NEXT TWO
python3 stats.py                     # recompute every quoted number            (~4 min)
python3 make_figures.py              # regenerate all six figures               (~8 min)
```

Nothing takes arguments to do the standard thing.

**Run `export_graphs.py` before `stats.py` or `make_figures.py`.** `construct.py`
prints its verification but writes nothing unless given `--out`, and then only
CSVs. The four files in `graphs/` are `.gitignored` because they are
regenerable, so a fresh checkout does not have them and anything that reads them
fails. `export_graphs.py` calls `construct.py`'s own `grow()` and `coarsen()` and
writes all four: the parent, the parent plus its supernode assignment, the
1,015-node quotient, and the quotient's bundle weights. It is deterministic, and
`--check` verifies a rebuild against files already present.

`graphs/cnew_parent_supernode.npz` is also the substrate the recurrent
inpainting models use; copy it to
`graphdynamics/graph_8k_parent_supernode.npz` to run them on it. See
`graphdynamics/README.md`. `construct.py` grows the
graph from a hard-coded seed and prints its own verification; it does not read
any file unless you ask it to.

## Requirements

```
numpy  scipy  networkx  matplotlib
pandas  scikit-learn        # only for the NMI/ARI and --budapest comparison lines
```

Verified on python 3.13.5, numpy 2.3.4, scipy 1.16.2, networkx 3.5,
matplotlib 3.10.8, pandas 3.0.2, scikit-learn 1.8.0.

No GPU. No compilation. Peak memory is about 3 GB, in the parent's adjacency.

---

## What is here

```
construct.py              the construction: seed -> parent -> quotient (1,015 nodes)
export_graphs.py          writes graphs/ from construct.py; run it first
stats.py                  recomputes every number quoted about this graph
static_tests.py           the six static generative tests against Budapest
display_quotient.py       community-ordered adjacency rendering
make_figures.py           writes the six figures into figures/

graphs/                   the parent and quotient
reference/                Budapest: 1,015 nodes, 70,654 edges, plus edge weights
blockwise/                the alternative coarsening (see below)
```

### The graphs

| file | what |
|---|---|
| `graphs/cnew_parent.npz` | the 8,000-node parent, stored as three tripartite masks |
| `graphs/cnew_parent_supernode.npz` | the same parent plus the supernode assignment |
| `graphs/cnew_coarse.npz` | the 1,015-node strict quotient |
| `graphs/cnew_coarse_w.npy` | per-quotient-edge fibre-bundle weights (64,760 values) |

The parent is stored as masks rather than an edge list because the three
colour classes are the natural block structure: `mask_12`, `mask_13`, `mask_23`
are the three inter-class blocks, and there is no `mask_11` because a proper
colouring forbids it. `stats.py:load_masks()` shows the reconstruction.

`reference/bud_w.npy` has shape `(4, 70654)`. Row 0 is the mean fibre count per
connection — the channel used everywhere in the paper (mean 2.033, max 154.9).
Rows 1–3 are fibre count, fibre density and mean fibre length. Use row 0 unless
you have a reason not to.

---

## What `construct.py` does

Four stages, printed as it goes.

**Seed.** 20 vertices, 31 edges, hard-coded. Colour classes of size 6/6/8;
blocks of size 6/6/4/4 in two hemispheres. The edge counts are 26 within-block,
4 same-hemisphere, 1 cross-hemisphere — a ratio chosen to mirror Budapest's
block densities of roughly 0.45 : 0.09 : 0.003.

**Growth.** Three Kronecker powers, `G_{k+1} = G_k ⊗ S`, with chromatically
legal within-block triangle closure after each, `α = (3,4,30)`, `β = 1.5`.
No edge is ever deleted.

```
seed       n=    20   product m=     31   closure +      5   =      36
tensor 2   n=   400   product m=  2,232   closure +  1,600   =   3,832
tensor 3   n= 8,000   product m=237,584   closure +240,000   = 477,584
```

Closure runs *inside* the recursion rather than as a final overlay. A vertex of
`G_k` is a base-20 numeral with the most significant digit first, and expansion
appends digits on the right, so the leading digit — which carries both colour
and block — is never disturbed. An edge added within a block at step k is still
within that block at step k+1, and early closures get tensor-multiplied by the
later steps. The within-block to cross-block density contrast is 5.3 for the
pure tensor power and 28.5 with in-loop closure; below about 10 the blocks stop
being detectable at all.

**Verify.** Not asserted — measured. χ = 3 is established from both sides: zero
same-colour edges gives χ ≤ 3, and an explicit triangle gives χ ≥ 3. Blind
Louvain, given only the edge list, returns the four planted blocks exactly.

**Coarsen.** Within each inherited block, high-resolution Louvain, then greedy
agglomeration to the block's share of 1,015. Strict quotient: two supernodes
are joined iff at least one parent edge runs between them. No thinning.

### The merge rule — read this before citing it

`merge_densest()` builds a heap over **every** adjacent pair of communities in
the block, keyed by `-w(i,j)/(size_i · size_j)`, and repeatedly pops the
**globally densest pair**.

It is not smallest-first. Size enters only through the normalisation; no
community is ever selected for being small, and the rule compares the whole
partition at every step. This is greedy agglomerative clustering under a
size-normalised linkage.

The choice is deliberate and the docstring records why: smallest-first
equalises supernode sizes, and since coarse degree tracks supernode size, that
flattens the degree distribution. Densest-first keeps more of Louvain's natural
size spread.

Two further things the rule does **not** do. No individual vertex is ever moved
between supernodes — a vertex is placed once by Louvain and travels with
whichever community it landed in, so the coarsening is purely agglomerative. And
no modularity is computed during agglomeration; modularity maximisation supplies
the initial communities only.

Note that the *local* construction in `../local/` uses a genuinely different,
genuinely local rule — smallest community first, then its densest neighbour.
The two are easy to confuse. They are not the same rule.

### Community detection is the library routine

Everywhere Louvain appears, it is
`networkx.algorithms.community.louvain_communities` — the standard
implementation of Blondel et al. 2008 — called with the resolution passed
through unmodified. It is not reimplemented anywhere in this folder.

---

## Expected output

`construct.py` should end with exactly this. If it does not, something is
wrong with your environment, not with the construction — there is no
randomness in the parent.

```
  n = 8,000   m = 477,584   mean degree 119.4   CV 0.580
  chi = 3:          0 same-colour edges (=> chi <= 3); triangle present: True
  colour classes:   [2400, 2400, 3200]
  connected:        True
  blind Louvain:    4 communities [2400, 2400, 1600, 1600]
                    NMI=1.0000  ARI=1.0000

  block density matrix (A, B, C, D):
     2400   0.0609   0.0007   0.0020   0.0000
     2400   0.0007   0.0550   0.0000   0.0020
     1600   0.0020   0.0000   0.0501   0.0000
     1600   0.0000   0.0020   0.0000   0.0473
  exactly zero:     A-D, B-C, C-D

  n = 1,015   m = 64,760   density 0.1258   mean degree 127.6   sd 61.3
  supernode sizes:  2 to 59 parent vertices   (median 6)
  witness weight:   median 2   mean 6.9   max 978
  detected blocks:  [307, 306, 201, 201]   ARI vs built = 1.0000
```

The three exact zeros are the one structural property here that is derived
rather than designed or fitted. No seed edge joins A–D, B–C or C–D, and the
tensor product creates an edge between two fibres only when their seed vertices
are adjacent, so those pairs stay at exactly zero density at every Kronecker
power — not approximately, exactly.

---

## `stats.py`

Recomputes every number quoted about this graph from the shipped graphs. No
cached results are read.

```bash
python3 stats.py                       # all sections
python3 stats.py --section bilateral   # one
```

| section | what |
|---|---|
| `basic` | node/edge counts, density, degree, connectivity, colouring |
| `bundles` | fibre-bundle weights against Budapest |
| `community` | Louvain community counts and sizes over 20 seeds |
| `bilateral` | block-density matrices and the hemisphere test |
| `sixtests` | pointer to `static_tests.py`, which is slow and runs separately |
| `audit` | reads recorded audit JSON from `audits/`; that folder is not shipped, so this section prints its header only |

### The seed trap

Louvain is stochastic, and on this graph a single seed will mislead you.

Seed 42 returns 5 communities at Q = 0.5318 and shows no bilateral pairing.
Best-of-20-seeds returns 307/306/201/201 at Q = 0.5554, with both hemispheric
couplings present. The bilateral structure is there; one unlucky seed hides it.

So `stats.py` and `make_figures.py` both take the best-modularity partition
over 20 seeds. Do not substitute a single fixed seed for speed.

---

## `blockwise/` — the second coarsening

The same parent, coarsened by Louvain alone: each block clustered
independently over a resolution sweep, keeping whichever partition lands
closest to that block's target. No merge rule at any point.

```bash
cd blockwise
python3 blockwise_pipeline.py                  # build and verify
python3 blockwise_pipeline.py --parent-only    # stop after the parent
python3 blockwise_pipeline.py --out run        # rewrite the CSVs
```

It reproduces the identical parent — same 477,584 edges, same block density
matrix — and then diverges at the coarsening.

|  | merge rule | block-wise only |
|---|---|---|
| nodes | 1,015 (exact) | 1,076 (+6.0%) |
| edges | 64,760 | 73,152 |
| six-test mean | 0.201 | 0.182 |
| resolution | γ = 30 | γ = 8, all four blocks |

Each has one honest advantage and they are different advantages. The merge rule
reaches the target node count exactly and fits the degree distribution better.
The block-wise variant introduces no bespoke rule at all — its only two
operations are the tensor product and library Louvain — and it is better on the
spectrum.

**The two partitions are not similar.** Measured over the same 8,000 vertices,
they agree at ARI 0.1829 (NMI 0.7270), and neither is a refinement of the other:
48.0% of block-wise supernodes sit wholly inside one merge-rule supernode, but
only 7.3% the other way. They are substantially different cuts that score within
0.02 of each other on the six tests.

That is worth stating plainly, because the natural assumption is the opposite.
The connectome-like statistics here are largely a property of the parent, not of
the particular cut taken through it. Neither coarsening should be preferred on
topological grounds.

No CSVs are shipped; `--out run` writes all of them in about a minute.

---

## `static_tests.py` and `display_quotient.py`

```bash
python3 static_tests.py --graph graphs/cnew_coarse.npz \
                        --budapest reference/budapest_1015_70654.edgelist \
                        --label "8k quotient" --out fig.png --csv stats.csv

python3 display_quotient.py                       # default 3-panel
python3 display_quotient.py g1.npz g2.npz         # custom
```

`static_tests.py` runs the six tests of the generative-network-modelling
literature: degree distribution, effective diameter, hop plot, scree plot,
network value, and node triangle participation. It plots each under both
conventions — log-log, standard in that literature, and cumulative. Both are
shown deliberately: a log-log plot gives a handful of extreme-degree nodes the
same visual weight as the hundreds of nodes carrying the bulk of the
distribution, which is exactly the misreading that matters here.

`display_quotient.py` orders nodes by two nested Louvain levels (γ = 1.0 for
blocks, γ = 3.0 for sub-communities) and then by degree within each. Use it
rather than a Fiedler ordering — Fiedler ordering on a bilateral graph shows you
the bisection and hides everything else.

---

## What is fitted, and what is not

**Fitted.** `α = (3,4,30)`, the closure budget per vertex at each step;
`β = 1.5`, the closure sampling exponent; and the total target count of 1,015,
which is Budapest's node count. The per-block split of that total is derived
from the parent's own proportions, not from the reference.

**Designed.** The seed's four-block hemisphere pattern, chosen after measuring
the connectome's block densities. Bilateral organisation is accommodated here,
not predicted.

**Measured, not targeted.** χ = 3 exactly, with no repair operator anywhere in
the code. Blind block recovery at NMI = ARI = 1, across seeds. Three block pairs
at exactly zero density at every Kronecker power. Modularity, path length,
effective diameter, global efficiency, clustering, transitivity, the degree
distribution, and the spectrum.

## What is not reproduced

The nine-criterion connectome audit returns 7 of 9, failing
`path_near_random` and `global_eff_near_random` — the parent's mean shortest
path is longer than its degree-preserving nulls, not near-random. The audit
itself is `../../graphmetrics/connectome_audit_gold.py`.

Fibre-bundle weights are the clearest miss and the paper understates it if read
quickly. Each quotient edge stands for 6.88 parent edges on average against
2.03 fibres per connection in Budapest, and the divergence is in the body of
the distribution, not the tail: relations of weight ≤ 5 carry 21.9% of total
fine-edge support here against 88.2% in Budapest.

Degree spread is too narrow, minimum degree far too high (33 against 7), and
the spectral gap remains the largest single error in both coarsenings.

---

## Reproducibility

The parent is fully deterministic; the coarsening's Louvain step is not, which
is what the 20-seed rule above is for. `construct.py`, `stats.py --section basic`, `make_figures.py --only AI` and
`blockwise_pipeline.py --parent-only` were each run from this folder and
verified; `make_figures.py` creates `figures/` on first run. `static_tests.py`
and `display_quotient.py` were carried over unchanged from the package they
were developed in and were not re-run here.
