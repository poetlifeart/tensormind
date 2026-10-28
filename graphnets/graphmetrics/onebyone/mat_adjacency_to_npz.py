"""MATLAB .mat adjacency matrix -> undirected, unweighted, no self-loops, GCC edge-list npz the audit's loader reads.
Usage: python3 -I mat_adjacency_to_npz.py IN.mat VARIABLE OUT.npz
Any nonzero entry is an edge; a directed matrix is symmetrised (A or A^T), and the script says how many entries were one-way."""
import sys, numpy as np, scipy.io, networkx as nx
path, var, out = sys.argv[1], sys.argv[2], sys.argv[3]
M = scipy.io.loadmat(path)[var]
M = M.toarray() if hasattr(M, 'toarray') else np.asarray(M)
A = (M != 0)
n0 = A.shape[0]
print(f"matrix {var}: {A.shape}, nonzero entries {int(A.sum())}, diagonal {int(np.trace(A))}, "
      f"symmetric {bool((A == A.T).all())}, one-way entries {int((A & ~A.T).sum())}")
U = A | A.T
np.fill_diagonal(U, False)
G = nx.Graph(); G.add_nodes_from(range(n0))
G.add_edges_from((int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(U, 1))))
cc = max(nx.connected_components(G), key=len)
Gc = nx.convert_node_labels_to_integers(G.subgraph(cc).copy())
n, m = Gc.number_of_nodes(), Gc.number_of_edges()
print(f"undirected: {G.number_of_nodes()} nodes {G.number_of_edges()} edges; components "
      f"{nx.number_connected_components(G)}; GCC {n} / {m}; mean degree {2*m/n:.3f}")
np.savez(out, n_total=n, edges=np.array(list(Gc.edges()), dtype=np.int64))
print("wrote", out)
