"""Usage: python3 -I edgelist_to_npz.py IN OUT.npz [--gcc]

Two-column integer edge list (SNAP .txt / .txt.gz, KONECT out.* file; lines starting
with '#' or '%' are comments; extra columns ignored) -> undirected, unweighted, no
self-loops, duplicates merged, node ids relabelled 0..n-1 in sorted order of the
original ids.  Whole graph by default (every node that appears in the file is kept,
including nodes left isolated once self-loops are dropped); --gcc keeps only the
largest connected component.  Writes n_total + edges, the format static_tests.py
load_edges() and the gold-audit loader read."""
import gzip, sys
import numpy as np, networkx as nx

src, out = sys.argv[1], sys.argv[2]
gcc_only = '--gcc' in sys.argv[3:]
opener = gzip.open if src.endswith('.gz') else open
rows = []
with opener(src, 'rt') as f:
    for line in f:
        s = line.strip()
        if not s or s[0] in '#%':
            continue
        a, b = s.split()[:2]
        rows.append((int(a), int(b)))
rows = np.array(rows, dtype=np.int64)
ids = np.unique(rows)
loops = int((rows[:, 0] == rows[:, 1]).sum())
print(f"file: {len(rows)} edge records, {len(ids)} distinct node ids, {loops} self-loop records")
idx = {v: i for i, v in enumerate(ids.tolist())}
G = nx.Graph()
G.add_nodes_from(range(len(ids)))
G.add_edges_from((idx[a], idx[b]) for a, b in rows.tolist() if a != b)
print(f"undirected simple graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
      f"(+{loops} self-loops dropped); isolated nodes {nx.number_of_isolates(G)}; "
      f"components {nx.number_connected_components(G)}")
cc = max(nx.connected_components(G), key=len)
print(f"largest component: {len(cc)} nodes, {G.subgraph(cc).number_of_edges()} edges")
if gcc_only:
    G = nx.convert_node_labels_to_integers(G.subgraph(cc).copy())
    print(f"kept largest component only: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
edges = np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)
np.savez(out, n_total=G.number_of_nodes(), edges=edges)
print(f"max node id in edges = {int(edges.max())} (n_total = {G.number_of_nodes()})")
print("wrote", out)
