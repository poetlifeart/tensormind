"""MatrixMarket (pattern, symmetric) -> undirected, unweighted, no self-loops; ALL nodes kept
(no GCC extraction).  Writes an edge-list npz (n_total + edges, 0-based)."""
import sys, numpy as np, scipy.io, scipy.sparse, networkx as nx
A = scipy.sparse.coo_matrix(scipy.io.mmread(sys.argv[1]))
n = A.shape[0]; assert A.shape[0] == A.shape[1]
G = nx.Graph(); G.add_nodes_from(range(n))
G.add_edges_from((int(i), int(j)) for i, j in zip(A.row, A.col) if i != j)
print(f"matrix {A.shape}, {A.nnz} stored entries; self-loops dropped {int(np.sum(A.row == A.col))}")
cc = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
print(f"graph: {n} nodes, {G.number_of_edges()} edges (nnz of symmetric A = {2*G.number_of_edges()}); "
      f"components {len(cc)}, largest {cc[:5]}")
np.savez(sys.argv[2], n_total=n, edges=np.array(sorted(G.edges()), dtype=np.int64))
print("wrote", sys.argv[2])
