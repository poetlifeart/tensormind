"""Wolfram Data Repository graph in plain WL text, Graph[{v1, v2, ...}, {DirectedEdge[a, b], ...}]
-> undirected, unweighted, no self-loops, largest component.  Writes the edge-list npz the
audit's loader reads (n_total + edges; no masks -> generic null).
Usage: python -I wolfram_graph_to_npz.py IN.wl OUT.npz"""
import re, sys, numpy as np, networkx as nx
txt = open(sys.argv[1]).read()
m = re.fullmatch(r'\s*Graph\[\{([0-9,\s]*)\},\s*\{(.*)\}\]\s*', txt, re.S)
if not m:
    raise SystemExit("not of the form Graph[{integers}, {edges}] -- refusing to guess")
verts = [int(x) for x in m.group(1).split(',') if x.strip()]
body = m.group(2)
pairs = [(int(a), int(b)) for a, b in re.findall(r'(?:Directed|Undirected)Edge\[\s*(\d+)\s*,\s*(\d+)\s*\]', body)]
n_dir = body.count('DirectedEdge['); n_und = body.count('UndirectedEdge[')
leftover = re.sub(r'(?:Directed|Undirected)Edge\[\s*\d+\s*,\s*\d+\s*\]', '', body).replace(',', '').strip()
print(f"file: {len(verts)} vertices ({len(set(verts))} distinct), {len(pairs)} edge records "
      f"({n_dir} DirectedEdge, {n_und} UndirectedEdge); unparsed text after edges: {len(leftover)} chars")
if leftover:
    raise SystemExit("edge list contains something other than Directed/UndirectedEdge[int, int]")
D = nx.DiGraph(); D.add_nodes_from(verts); D.add_edges_from(pairs)
print(f"directed: {D.number_of_nodes()} nodes, {D.number_of_edges()} distinct arcs, "
      f"{nx.number_of_selfloops(D)} self-loops, {sum(1 for u, v in D.edges() if D.has_edge(v, u) and u < v)} reciprocal pairs")
G = nx.Graph(D); G.remove_edges_from(nx.selfloop_edges(G))
cc = max(nx.connected_components(G), key=len)
Gc = nx.convert_node_labels_to_integers(G.subgraph(cc).copy())
n, e = Gc.number_of_nodes(), Gc.number_of_edges()
print(f"undirected: {G.number_of_nodes()} nodes {G.number_of_edges()} edges; components "
      f"{nx.number_connected_components(G)}; GCC {n} / {e}; mean degree {2*e/n:.3f}; diameter {nx.diameter(Gc)}")
np.savez(sys.argv[2], n_total=n, edges=np.array(list(Gc.edges()), dtype=np.int64))
print("wrote", sys.argv[2])
