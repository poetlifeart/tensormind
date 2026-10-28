"""Petersen graph, exactly as printed in Alikhani & Ghanbari, "More on energy and
Randic energy of specific graphs", arXiv:1412.8137 (2014), proof of Theorem 2(i),
p. 7: the 10x10 adjacency matrix A(P) (vertex labels 1..10 of their Figure 2).
Checks it is symmetric, 3-regular, and isomorphic to networkx's petersen_graph().
Writes petersen.npz (n_total + edges, 0-based) next to this script."""
import os
import numpy as np
import networkx as nx

A = np.array([
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
])
assert (A == A.T).all() and (np.diag(A) == 0).all()
assert (A.sum(1) == 3).all()
G = nx.from_numpy_array(A)
assert nx.is_isomorphic(G, nx.petersen_graph())
edges = np.array(sorted(G.edges()), dtype=np.int64)
print(f"Petersen graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"3-regular, isomorphic to nx.petersen_graph(): True")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'petersen.npz')
np.savez(out, n_total=G.number_of_nodes(), edges=edges)
print("wrote", out)
