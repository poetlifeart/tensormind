# The local-repair construction

`emergence_localrepair_standalone.py` builds the paper's *local* graph end to end
in a single file — fine graph, coarsening, and self-verification. Its own header
says: "No imports from any other project file."

| output | nodes | edges |
|---|---|---|
| `localrepair_parent.npz` | 16,807 | fine parent |
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

`../../Budapest/budapest_connectome.gml` is read once, at line 174, *after* the
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
