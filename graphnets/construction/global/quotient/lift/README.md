# Lifting the quotient back to parent scale (the feeder graph)

`invert146_feeder.py` maps the v146 quotient's **non-witness** edges back down to
individual parent nodes, producing the graph the inpainting models were trained on.

| output | nodes | edges |
|---|---|---|
| `graph_brain_mild_v146_feeder.npz` | 16,807 | 344,483 |

That is the mild parent (337,983) plus the 6,500 edges the swap introduced —
the swap added within-L1 quotient edges that no parent edge witnessed, and this
script gives each of them a concrete parent-level realisation.

## Method

For each non-witness quotient edge between super-nodes A and B:

1. Compute the local 80th-percentile degree within each super-node.
2. Connect nodes at or above that percentile in their own super-node.

So the lifted edges attach to locally high-degree nodes rather than arbitrary
members — the quotient edge is realised by the sub-graph's own hubs.

---

## ⚠ WARNING — this script re-implements the ENTIRE quotient pipeline

Logically this step is downstream of the **post-swap** graph: what it lifts are
the non-witness edges the 6,500 swap introduced. But it does **not** read
`graph_v146_6500.npz`. It takes the mild parent and rebuilds everything itself:

```
line  44   coarsen(Gp, target=1015)     louvain_communities(resolution=20.0, seed=42)
line  72   build_v14(Q, witness, n)     L0/L1/L2 hierarchy + tiered selection
line 174   "Reproduce v146 swap=6500"   SWAP_N = 6500, remove weakest T3, add within-L1
line 244   non_witness_added            the edges the swap introduced
line 270   invert them onto the parent  the actual lift
```

Lines 44–216 duplicate `../save_v146_npz.py` in full — coarsening, hierarchy,
tiered selection **and the swap**. The complete quotient pipeline exists twice in
this repository, independently implemented.

**They agree today.** But there is no shared module and no consistency check, so:

- editing the coarsening, the tiered selection or the swap in one file and not the
  other makes this script lift from a v146 that is *not* the one in `../`,
  silently and with no error;
- running the two under different networkx versions has the same effect, because
  Louvain drifts (see the note in `../README.md`);
- the duplication is invisible from the outside — both scripts produce plausible
  output either way.

The consequence is that **two graphs in this repository both claim to be "the"
post-swap quotient**, produced by independent code, with nothing checking that
they match. `invert146_feeder.py` carries its own `SWAP_N = 6500` and raises if it
cannot complete the swaps, so it will build a feeder graph on its own v146 without
ever consulting `graph_v146_6500.npz`.

### The fix

Have this script **load** the post-swap graph and the witness set instead of
recomputing them:

```
current:  parent -> coarsen -> build_v14 -> swap -> non-witness edges -> lift
better:   load graph_v146_6500.npz + witness  ->  non-witness edges  -> lift
```

That deletes lines 44-216 (the duplicated pipeline), removes the hazard entirely,
and makes the dependency explicit — the lift becomes what it logically is, a step
downstream of `../save_v146_npz.py` rather than a parallel reimplementation of it.
It also makes the script much faster, since the coarsening and swap are the
expensive parts.

Until that change is made: if you touch anything in `../save_v146_npz.py`, make
the same change here, and pin `networkx==3.5` for both runs.

## Also note

- The script has hard-coded absolute paths for its input parent and output directory.
- It does `from connectome_audit_gold import main as audit_main` (line 460). That
  file lives in `graphnets/graphmetrics/`, so the import will fail from here until
  the path is fixed.
