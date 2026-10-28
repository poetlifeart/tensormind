"""Network Repository MatrixMarket graph (.mtx, coordinate, 1-indexed) -> edge-list npz (n_total + edges).

Usage: python3 -I mtx_to_npz.py IN.mtx OUT.npz   (also KONECT-style .edges: "% sym ...", "% m n n")

Undirected, unweighted: any weight column is dropped, (i, j) and (j, i) are one edge,
self-loops are dropped.  The whole graph is kept (n_total = rows in the size line);
components and isolates are reported, not removed.
"""
import sys

import numpy as np
import networkx as nx

lines = open(sys.argv[1]).read().splitlines()
header = lines[0]
body = [l for l in lines[1:] if l.strip() and not l.startswith('%')]
if 'MatrixMarket' not in header:                 # KONECT-style .edges: '% sym unweighted', then '% m n n'
    m_, nr, nc = (int(x) for x in lines[1].lstrip('% ').split()[:3])
    size, entries = f"{nr} {nc} {m_}", body
elif len(body[0].split()) == 3 and int(body[0].split()[2]) == len(body) - 1:
    size, entries = body[0], body[1:]            # standard: size line, then entries
else:                                            # some files carry the size line as a comment: '% n n nnz'
    size = [l.lstrip('% ') for l in lines[1:] if l.startswith('%')
            and len(l.lstrip('% ').split()) == 3 and all(t.isdigit() for t in l.lstrip('% ').split())][-1]
    entries = body
nr, nc, nnz = (int(x) for x in size.split()[:3])
pairs = [tuple(int(x) for x in l.split()[:2]) for l in entries]
assert len(pairs) == nnz, f"{len(pairs)} entries, size line says {nnz}"
G = nx.Graph()
G.add_nodes_from(range(max(nr, nc)))
G.add_edges_from((i - 1, j - 1) for i, j in pairs)
loops = nx.number_of_selfloops(G)
G.remove_edges_from(list(nx.selfloop_edges(G)))
n, m = G.number_of_nodes(), G.number_of_edges()
comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
print(f"{sys.argv[1]}: header '{header}'; size line {nr} {nc} {nnz}; "
      f"undirected {n} nodes {m} edges; self-loops dropped {loops}; "
      f"components {len(comps)} (largest {comps[0]}); isolates {nx.number_of_isolates(G)}")
np.savez(sys.argv[2], n_total=n, edges=np.array(sorted(G.edges()), dtype=np.int64))
print("wrote", sys.argv[2])
