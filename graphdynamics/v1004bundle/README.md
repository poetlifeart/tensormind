# v1004bundle — the recurrent model with bundle-map message passing

Same architecture as `v1003/` in every respect but one: how signal crosses the
graph.

## The difference

`v1003` uses `MultiChannelSparseLinear` — **one learnable weight per fine edge**.
A pair of supernodes joined by 342 fine edges is 342 independent channels.

`v1004bundle` uses `BundleSparseLinear` — signal travels along the **bundle
map**:

```
converge   an axon averages the distinct source vertices dealt to it
transmit   one learnable weight per axon
fan out    every distinct target vertex dealt to it receives that value
```

A **bundle** is a colour-to-colour relation between two supernodes: keyed by
(source supernode, target supernode) inside a layer, and each layer is one
ordered colour pair. Nothing is averaged across colour classes — a 0→1 relation
and a 0→2 relation carry different timings and never share an axon.

A bundle of `W` fine edges reduces to

```
n(W) = round( W^0.05 * (1 + (W-1)/11)^0.95 )        at least 1
```

axons, its edges dealt round-robin among them. That rule is measured, not
chosen: it is the reduction that brings the construction's coarse-relation
multiplicity onto the reference connectome's fibre-bundle distribution —
median 1.67 against 1.67, Gini 0.371 against 0.305. See
`../../graphnets/construction/bilateral/bundles/`.

## What it costs

```
                     graph weights   total params
  v1003 per-edge           806,551     74,911,834
  v1004bundle              323,901     74,429,184
```

2.5× fewer graph weights; the model as a whole is 0.65% smaller, since the
encoder, decoder, collector, twin path and supernode attention are untouched.

Axons per bundle averages 1.14 — most bundles hold a single edge and reduce to
one axon, while the heaviest (342 edges) gets 36.

## Running it

```bash
cd graphdynamics
python3 -u v1004bundle/train_v1004bundle.py \
    --seed 0 --gpu cuda:0 --from-scratch \
    --data /path/to/celebahq256 \
    --save-dir /path/to/ckpt_bundlemap_s0
```

`--graph` defaults to `graph_8k_parent_supernode.npz` in this directory — the
8,000-vertex parent with its 1,015-supernode assignment. The layer reads the
three tripartite masks and the three `supernode_id_*` arrays from it; the
bundles and their axon counts are derived at construction time.

Benchmark the result exactly as for `v1003`:

```bash
python3 eval_benchmark_v2.py --checkpoint <ckpt> \
    --graph graph_8k_parent_supernode.npz --v1004bundle \
    --mask-dist benchmark --save bench.json
```

## What is not settled

**Edge dealing.** A bundle is a symmetric set of edges between two supernodes in one colour pair; the edges are dealt round-robin among the bundle's axons. There is no source/target asymmetry to respect.
index order, not by structure. Grouping edges that share a source vertex would
make each axon a genuine fan rather than an arbitrary slice; untested.

**The bottleneck is mild.** At 1.14 axons per bundle most bundles are unchanged
from a one-axon-per-bundle version. The reduction is real against the per-edge
baseline but sits much nearer the one-axon model.

**Not a clean capacity control.** The bundle model has 483k fewer parameters
than `v1003`, so a difference in results is not purely attributable to the
message-passing change.
