"""The five four-vertex graphs of Ellens & Kooij, "Graph measures and network
robustness", arXiv:1311.5064 (2013), Figure 4 / Table 1:
  (a) complete graph K4, (b) cycle C4, (c) star S4 (K_{1,3}), (d) path P4,
  (e) empty graph O4 (4 isolated vertices).
Built with networkx (complete_graph(4), cycle_graph(4), star_graph(3),
path_graph(4), empty_graph(4)).  Writes K4.npz, C4.npz, S4.npz, P4.npz, O4.npz
(n_total + edges) next to this script."""
import os
import numpy as np
import networkx as nx

here = os.path.dirname(os.path.abspath(__file__))
graphs = {'K4': nx.complete_graph(4), 'C4': nx.cycle_graph(4), 'S4': nx.star_graph(3),
          'P4': nx.path_graph(4), 'O4': nx.empty_graph(4)}
for name, G in graphs.items():
    assert G.number_of_nodes() == 4
    edges = np.array(sorted(G.edges()), dtype=np.int64).reshape(-1, 2)
    np.savez(os.path.join(here, f'{name}.npz'), n_total=4, edges=edges)
    print(f"{name}: 4 nodes, {G.number_of_edges()} edges, degrees {sorted(d for _, d in G.degree())}, "
          f"components {nx.number_connected_components(G)}")
