# bundles — post-processing of the 8,000-node construction's fibre bundles

A coarse relation's **multiplicity** `W_AB` is the number of fine edges crossing
between two supernodes. The reference connectome's edge weight is a **mean fibre
count**. These are different quantities. This script measures how they relate.

```bash
python3 bundles.py                 # all four sections
python3 bundles.py --section fan   # one
```

## Input

Nothing is passed on the command line. Paths are resolved relative to the
script.

| file | what it supplies |
|---|---|
| `../graphs/cnew_coarse_w.npy` | one bundle weight per quotient relation |
| `../reference/bud_w.npy` | reference edge weights; **row 0** is the mean fibre count, the channel used throughout |
| `../construct.py` | rebuilt by `--section fan` to recover the parent and its partition |

`--section fan` regenerates the 8,000-vertex parent and re-runs the coarsening,
so it takes a few minutes and needs about 2 GB. The other sections read the two
arrays and return immediately.

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
- `floor` asks **how much of the residual is reachable**, given that one side is
  an integer edge count and the other a fractional average.
