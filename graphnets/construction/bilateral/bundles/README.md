# bundles — post-processing of the 8,000-node construction's fibre bundles

A coarse relation's **multiplicity** `W_AB` is the number of fine edges crossing
between two supernodes. The reference connectome's edge weight is a **mean fibre
count**. These are different quantities. This script measures how they relate.

```bash
python3 bundles.py                 # all five sections
python3 bundles.py --section fan   # one
```

## Input

Nothing is passed on the command line. Paths are resolved relative to the
script.

| file | what it supplies |
|---|---|
| `../graphs/cnew_coarse_w.npy` | one weight per quotient relation, pooled over colour pairs |
| `../graphs/cnew_parent_supernode.npz` | the parent's three colour masks and its supernode assignment; read by `--section law` to rebuild the colour-to-colour bundles |
| `../reference/bud_w.npy` | reference edge weights; **row 0** is the mean fibre count, the channel used throughout |
| `../construct.py` | rebuilt by `--section fan` to recover the parent and its partition |

`cnew_parent_supernode.npz` is byte-identical to the model's
`graphdynamics/graph_8k_parent_supernode.npz`, so `--section law` sees exactly
the bundles `BundleSparseLinear` builds.

`--section fan` regenerates the 8,000-vertex parent and re-runs the coarsening,
so it takes a few minutes and needs about 2 GB. The other sections read the
stored arrays and return in under a second; `law` derives its bundles from the
stored masks rather than rebuilding the parent.

Requires `numpy`, `scipy`, `networkx`. No matplotlib, no GPU, no downloads.

## Output

Plain text on stdout. Nothing is written to disk.

**`dist`** — a table of count, mean, median, max and Gini for the quotient and
the reference; the fraction of total fine-edge support carried by relations of
weight ≤ 5 for each; a quantile ladder from p50 to p99.9 with the ratio between
them; and the KS statistic with the means matched.

**`fan`** — the parent's edge count split into crossing and absorbed. Then, per
band of `W`: how many relations fall in it, the mean number of distinct vertices
participating at each end (`a` and `b`), and two ratios — `a·b/W`, which says how
saturated the relation is, and `(a+b)/W`, which says how much a converge-diverge
reading compresses it. Ends with the five heaviest relations individually, each
with its `a`, `b`, resulting fibre count, compression factor and saturation.

**`tract`** — a table comparing the raw weights against two transforms, each
constrained so `f(1) = 1` and with its free parameter fixed by matching the
mean. Columns are median, max, Gini, mass at weight ≤ 5, and KS against the
reference. Followed by the exponent the mean-matching produces and the value the
convergence argument predicts independently.

**`law`** — the axon law `f(W)` the recurrent model applies: the formula with
its constants, a short ladder of `W`, `f(W)` and `f/W`, and the statistics of
`W` and `f(W)` over the colour-to-colour bundles, with the pooled quotient
weights and the reference fibre counts printed alongside so the three objects
cannot be confused. Ends with the axon total, which reproduces the model's
graph-weight count, and with the one quantity that is commensurable with the
reference. See [The bundle law](#the-bundle-law).

**`floor`** — the fraction of quotient relations at weight exactly 1, the
fraction of reference connections at exactly 1.0, and the gap between them,
which is a lower bound on KS for any rule preserving `f(1) = 1`.

## Why the sections are separated

Each answers a different question, and they are independent:

- `dist` asks **how far apart** the two distributions are.
- `fan` asks **what a relation actually is** — how many vertices take part, so
  whether multiplicity should be read as a count of links at all.
- `tract` asks **what transform relates the two units**, and whether its
  parameter has to be fitted or falls out of a constraint.
- `law` asks **what the model actually does**, which is a different question
  again: it acts on colour-split bundles, not on the pooled relations the other
  three sections compare against the reference.
- `floor` asks **how much of the residual is reachable**, given that one side is
  an integer edge count and the other a fractional average.

## The bundle law

### What a bundle is

A **bundle** is a *(supernode pair, colour pair)*. The parent is tripartite, so
every edge joins two different colour classes, and the three undirected colour
pairs 1–2, 1–3 and 2–3 partition the parent's edges into three independent
families. Within one family, a bundle collects every fine edge whose
lower-coloured endpoint lies in supernode *A* and whose higher-coloured endpoint
lies in supernode *B*.

This is exactly the key `BundleSparseLinear` builds. In the model each layer is
one ordered colour pair, and inside a layer the key is
`sid_in[source] * n_super + sid_out[target]` — the supernode pair alone, because
the layer has already fixed the colour pair. Supernodes here are **not**
monochromatic (every one of the 1,015 carries all three colours), so splitting
by colour pair is a genuine refinement and not a relabelling.

`W` is the number of fine edges collapsing onto that one coarse relation — the
bundle's multiplicity. `f(W)` is the number of axons the model gives it: one
learnable weight each, with the bundle's edges dealt round-robin among them.

The three colour pairs give **168,922** bundles over the parent's 477,584 edges.
Pooling them by supernode pair alone gives the **64,760** coarse relations that
`dist`, `tract` and `floor` compare against the reference. **These are different
objects**, and the law is defined on the first.

### The law

```
f(W) = round( W^p * (1 + (W - 1)/c)^(1 - p) ),   minimum 1

       p = 0.05        the exponent on the raw multiplicity
       c = 11.0        the knee, in fine edges
```

Identical to `BundleSparseLinear._n_axons` in
`graphdynamics/v1004bundle/model_v1004bundle.py`, term for term: same exponents,
same knee, same rounding, same clamp, same order of operations. The two agree
exactly for every integer `W` the construction produces.

### The two limits

**Small `W`.** `f(1) = 1`: one edge is one axon. This holds *identically, for
every `p` and `c`* — it is a property of the functional form, since `1^p = 1`
and `(1 + 0/c)^(1-p) = 1`. It therefore constrains neither constant. Nor is `p`
by itself the low-`W` exponent: the log-log slope at `W = 1` is
`p + (1-p)/c = 0.136`, not `p`.

**Large `W`.** `(1 + (W-1)/c) → W/c`, so

```
f(W)  ->  W / c^(1-p)  =  W / 9.757
```

The growth is asymptotically **linear**, with a constant compression factor
`c^(1-p) = 9.757`. What is sub-linear is the *ratio*: `f/W` falls monotonically
from 1 at `W = 1` to `c^-(1-p) = 0.102`, which is the sense in which the heaviest
bundles are compressed hardest. This limit gives one equation in two unknowns,
so it does not fix `p` and `c` either. **Both constants are fitted. Neither is
derived from the construction, and neither follows from the limits.**

### Monotonicity and concavity

Write `g(W) = W^p (1 + (W-1)/c)^(1-p)` for the unrounded function.

- `g` is **strictly increasing**: `g'/g = p/W + (1-p)/(W + c - 1) > 0`.
- `g` is **strictly concave**, and this is exact rather than numerical:

```
g'' = -p(1-p) * g(W) * [ 1/W - 1/(W + c - 1) ]^2   <=  0
```

  which is negative for all `W > 1` whenever `0 < p < 1` and `c > 1`.

- `f = max(round(g), 1)` is **monotone non-decreasing** — verified over
  `W = 1 … 100000` — but it is **not concave**. Rounding makes it a step
  function whose successive increments are 0 or exactly 1, so the second
  difference changes sign repeatedly; the first violation is at `W = 4`. Read
  "concave" as a statement about `g`, not about `f`. Likewise each extra edge
  adds **at most** one axon, not strictly less than one: `g'` never exceeds
  `g'(1) = 0.136`, but a rounding step is a jump of exactly 1.

### Two parameterisations in one file, fitted to different things

The same functional form appears twice in `bundles.py` with different constants,
and they answer different questions:

| where | constants | acts on | fitted to |
|---|---|---|---|
| `sec_tract`, "smooth" row | `c = 50` fixed by hand, `p = 0.4128` fitted | the 64,760 **pooled** quotient relations | the reference fibre-count **mean** (2.0326), matched exactly by construction of the fit |
| `sec_law`, `P_EXP`/`C_KNEE` | `p = 0.05`, `c = 11.0`, both hard-coded | the 168,922 **colour-split** bundles | not matched to any statistic this script computes |

The `tract` fit is reproducible from this repository: one constraint, one
unknown, solved by `brentq`. The model's `p = 0.05`, `c = 11.0` are not — they
are carried over as the model's constants, and applying them to the bundles
gives axon counts (median 1.00, mean 1.14, max 33, Gini 0.114) that are not
close to the reference's fibre counts (1.67 / 2.03 / 154.9 / 0.305). Summing
axons over the colour pairs of each supernode pair — the only quantity
commensurable with a per-connection fibre count — gives median 2.00, mean 2.89,
max 105, and is still not a match.

**The law is not what produces the paper's fibre-bundle agreement.** That
agreement comes from the degree-matched relocation quotient, whose weights are
raw edge multiplicities and need no transform. The law is the model's
weight-sharing rule, and the two claims should not be run together.

## References

The bundle law is **fitted in this work and has no source**. `p` and `c` are
free constants chosen against the reference fibre-count distribution; nothing in
the literature, and nothing in the construction, supplies them. Do not cite the
references below for the law itself — they are for the objects it acts on.

- Szalkai, Kerepesi, Varga & Grolmusz, "The Budapest Reference Connectome Server
  v2.0", *Neurosci. Lett.* 595:60–62 (2015). doi:10.1016/j.neulet.2015.03.071
- Szalkai, Kerepesi, Varga & Grolmusz, "Parameterizable consensus connectomes
  from the Human Connectome Project: the Budapest Reference Connectome Server
  v3.0", *Cogn. Neurodyn.* 11(1):113–116 (2017). doi:10.1007/s11571-016-9407-z

The reference weights in `../reference/bud_w.npy` row 0 are the `fiber_count_mean`
edge attribute of the Budapest Reference Connectome 3.0, `all_20k` variant — the
streamline count between two regions, averaged over the subjects in which the
connection is present.
