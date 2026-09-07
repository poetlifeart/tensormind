# Budapest Reference Connectome

`budapest_connectome.gml` — **1,015 nodes / 70,654 edges**, verified on load.

The only input in this repository that is not constructed by it. Everything else
is grown deterministically from a hard-coded seed.

## Provenance

Third-party data, not produced by this project. Cite the original work:

- Szalkai, Kerepesi, Varga & Grolmusz, "The Budapest Reference Connectome Server
  v2.0", *Neuroscience Letters* 595:60–62, 2015.
- Szalkai, Kerepesi, Varga & Grolmusz, "Parameterizable consensus connectomes
  from the Human Connectome Project: the Budapest Reference Connectome Server
  v3.0", *Cognitive Neurodynamics* 11(1):113–116, 2017.
- Varga & Grolmusz, "The braingraph.org database with more than 1000 robust human
  connectomes in five resolutions", *Cognitive Neurodynamics* 15:915–919, 2021.

Source: braingraph.org

## Which snapshot this is

The 1,015-node `all_20k` graph. Note the difference from the currently hosted
version, which the paper documents explicitly: the hosted `all_20k` has 71,604
edges; this copy omits the weakest 950, preserving the same node set, giving
70,654.

## Used by

`clique_census.py`, `combinatorial_metrics_npz.py`, `run_complexity.py`,
`run_controllability.py` and `richclub_other_graphs.py` as the comparison
reference, and by `construction/local/` for its degree-distribution check.

`clique_census.py` is the only metric script that reads `.gml` directly — the
others may need an edge-list conversion depending on how they are invoked.
