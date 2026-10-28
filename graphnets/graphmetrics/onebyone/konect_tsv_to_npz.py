"""KONECT out.* TSV (lines 'u v [w ...]', '%' comments) -> undirected, unweighted, no self-loops, GCC.
Nodes relabelled 0..n-1 in sorted order of the KONECT ids. Writes n_total + edges (+ konect_ids)."""
import sys, numpy as np, networkx as nx
pairs = []
for line in open(sys.argv[1]):
    if line.startswith('%') or not line.strip():
        continue
    a, b = line.split()[:2]
    pairs.append((int(a), int(b)))
G = nx.Graph(); G.add_edges_from(pairs); G.remove_edges_from(list(nx.selfloop_edges(G)))
print(f"file: {len(pairs)} edge records; undirected {G.number_of_nodes()} nodes {G.number_of_edges()} edges; components {nx.number_connected_components(G)}")
cc = max(nx.connected_components(G), key=len)
ids = sorted(cc); idx = {u: i for i, u in enumerate(ids)}
edges = np.array(sorted((min(idx[u], idx[v]), max(idx[u], idx[v])) for u, v in G.subgraph(cc).edges()), dtype=np.int64)
print(f"GCC {len(ids)} nodes / {len(edges)} edges")
np.savez(sys.argv[2], n_total=len(ids), edges=edges, konect_ids=np.array(ids, dtype=np.int64))
print("wrote", sys.argv[2])
