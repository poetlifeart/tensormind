"""Varshney et al. 2011 NeuronConnect (CSV export of WormAtlas NeuronConnect.xls: N1,N2,Type,Nbr)
-> undirected binary graph of chemical (S, Sp, R, Rp) + electrical (EJ) connections, NMJ records dropped,
self-loops dropped, giant component kept (as Towlson et al. 2013: N = 279, M = 2287).
Nodes numbered 0..n-1 in sorted name order; names saved in 'names'."""
import csv, sys, numpy as np, networkx as nx
rows = list(csv.reader(open(sys.argv[1])))[1:]
G = nx.Graph()
for a, b, t, n in rows:
    if t == 'NMJ':
        continue
    G.add_node(a); G.add_node(b)
    if a != b:
        G.add_edge(a, b)
cs = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"{G.number_of_nodes()} neurons, {G.number_of_edges()} undirected edges, components {[len(c) for c in cs]}; dropped: {[sorted(c) for c in cs[1:]]}")
names = sorted(cs[0]); idx = {u: i for i, u in enumerate(names)}
edges = np.array(sorted((min(idx[u], idx[v]), max(idx[u], idx[v])) for u, v in G.subgraph(cs[0]).edges()), dtype=np.int64)
print(f"GCC {len(names)} nodes / {len(edges)} edges")
np.savez(sys.argv[2], n_total=len(names), edges=edges, names=np.array(names))
print("wrote", sys.argv[2])
