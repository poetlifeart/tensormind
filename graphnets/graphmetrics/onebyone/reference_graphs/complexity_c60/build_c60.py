"""Buckminsterfullerene C60 (the icosahedral, isolated-pentagon C60 cage) as the
truncated icosahedron: start from networkx's icosahedral_graph(), take its
(unique, 3-connected) planar embedding, replace every vertex u by a 5-cycle of
new vertices (u, v) ordered as u's neighbours v go round u, and join (u, v) to
(v, u) for every edge uv.  Checks: 60 vertices, 90 edges, 3-regular, planar,
12 pentagonal + 20 hexagonal faces, no two pentagons share an edge (isolated
pentagon rule), Wiener index 8340 (the value Fowler 2002, Croat. Chem. Acta
75:401, abstract, gives for this isomer).  Writes c60.npz (n_total + edges)."""
import os
import numpy as np
import networkx as nx

I = nx.icosahedral_graph()
ok, emb = nx.check_planarity(I)
assert ok
T = nx.Graph()
for u in I.nodes():
    ring = list(emb.neighbors_cw_order(u))
    assert len(ring) == 5
    for i in range(5):
        T.add_edge((u, ring[i]), (u, ring[(i + 1) % 5]))
for u, v in I.edges():
    T.add_edge((u, v), (v, u))
G = nx.convert_node_labels_to_integers(T, ordering='sorted')

assert G.number_of_nodes() == 60 and G.number_of_edges() == 90
assert all(d == 3 for _, d in G.degree())
ok, embG = nx.check_planarity(G)
assert ok
faces, seen = [], set()
for u, v in embG.edges():
    if (u, v) not in seen:
        faces.append(embG.traverse_face(u, v, mark_half_edges=seen))
sizes = sorted(len(f) for f in faces)
assert sizes.count(5) == 12 and sizes.count(6) == 20 and len(faces) == 32
pent_edges = [frozenset((f[i], f[(i + 1) % 5])) for f in faces if len(f) == 5 for i in range(5)]
assert len(pent_edges) == len(set(pent_edges)) == 60      # no edge on two pentagons
W = int(nx.wiener_index(G))
assert W == 8340, W
print(f"C60: 60 vertices, 90 edges, cubic, planar, 12 pentagons + 20 hexagons, "
      f"isolated pentagons, Wiener index {W}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c60.npz')
np.savez(out, n_total=60, edges=np.array(sorted(G.edges()), dtype=np.int64))
print("wrote", out)
