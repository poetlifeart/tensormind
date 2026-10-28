"""Newman-format GML (node [ id N ] ... edge [ source A target B ]) -> undirected, unweighted, no self-loops, GCC.
Writes an edge-list npz the audit's loader reads (n_total + edges; no masks -> generic null)."""
import re, sys, numpy as np, networkx as nx
txt = open(sys.argv[1]).read()
ids = [int(x) for x in re.findall(r'node\s*\[\s*id\s+(\d+)', txt)]
pairs = [(int(a), int(b)) for a, b in re.findall(r'source\s+(\d+)\s+target\s+(\d+)', txt)]
print(f"file: {len(ids)} nodes, {len(pairs)} directed edge records")
G = nx.Graph(); G.add_nodes_from(ids); G.add_edges_from(pairs); G.remove_edges_from(nx.selfloop_edges(G))
cc = max(nx.connected_components(G), key=len); Gc = nx.convert_node_labels_to_integers(G.subgraph(cc).copy())
n, m = Gc.number_of_nodes(), Gc.number_of_edges(); k = 2*m/n
print(f"undirected: {G.number_of_nodes()} nodes {G.number_of_edges()} edges; components {nx.number_connected_components(G)}; GCC {n} / {m}; mean degree {k:.2f}")
print(f"C (avg clustering) = {nx.average_clustering(Gc):.4f}   L = {nx.average_shortest_path_length(Gc):.4f}   C_rand = k/n = {k/n:.4f}")
np.savez(sys.argv[2], n_total=n, edges=np.array(list(Gc.edges()), dtype=np.int64))
print("wrote", sys.argv[2])
