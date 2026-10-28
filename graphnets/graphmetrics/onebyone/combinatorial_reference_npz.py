"""Reference graphs for the combinatorial_*.py checks -> edge-list npz (n_total + edges).

Usage:
    python3 -I combinatorial_reference_npz.py mtx IN.mtx OUT.npz [--gcc]
    python3 -I combinatorial_reference_npz.py named petersen OUT.npz
    python3 -I combinatorial_reference_npz.py konect-bip out.NAME OUT.npz
    python3 -I combinatorial_reference_npz.py hog GRAPH.json OUT.npz [--same-as petersen]
    python3 -I combinatorial_reference_npz.py newman-gml IN.gml OUT.npz

mtx    MatrixMarket coordinate file (Network Repository / SuiteSparse; 1-indexed;
       any value column is dropped).  Undirected, unweighted: (i, j) and (j, i) are
       one edge; self-loops are dropped.  Whole graph by default (n_total = the size
       line); --gcc keeps only the largest connected component, relabelled 0..N-1 in
       sorted order of the original ids.
named  a graph built by networkx: 'petersen' = nx.petersen_graph() (10 nodes, 15 edges).
konect-bip  KONECT bipartite TSV ('% bip ...' then '% m n1 n2', lines 'left right'):
       left ids 1..n1 -> 0..n1-1, right ids 1..n2 -> n1..n1+n2-1; whole graph kept
       (n_total = n1 + n2, isolated nodes included).
hog    House of Graphs graph record (https://houseofgraphs.org/api/graphs/ID, JSON):
       its 'adjacencyList' (0-indexed) -> npz.  --same-as NAME also reports whether
       the graph is isomorphic to the networkx-built named graph NAME.
newman-gml  Newman-format GML (node [ id N ] ... edge [ source A target B ]), parsed
       as reference_graphs/newman_gml_to_npz.py does, but the WHOLE graph is kept
       (all components; node ids relabelled 0..n-1 in sorted order).
"""
import sys

import numpy as np
import networkx as nx


def from_mtx(path):
    lines = open(path).read().splitlines()
    body = [l for l in lines[1:] if l.strip() and not l.startswith('%')]
    nr, nc, nnz = (int(x) for x in body[0].split()[:3])
    pairs = [tuple(int(x) for x in l.split()[:2]) for l in body[1:]]
    assert len(pairs) == nnz, f"{len(pairs)} entries, size line says {nnz}"
    G = nx.Graph()
    G.add_nodes_from(range(max(nr, nc)))
    G.add_edges_from((i - 1, j - 1) for i, j in pairs)
    print(f"{path}: header '{lines[0].strip()}'; size line {nr} {nc} {nnz}")
    return G


def from_konect_bip(path):
    lines = open(path).read().splitlines()
    assert lines[0].startswith('% bip'), lines[0]
    m, n1, n2 = (int(x) for x in lines[1].lstrip('% ').split()[:3])
    pairs = [tuple(int(x) for x in l.split()[:2]) for l in lines[2:] if l.strip() and not l.startswith('%')]
    assert len(pairs) == m, f"{len(pairs)} edge lines, header says {m}"
    G = nx.Graph()
    G.add_nodes_from(range(n1 + n2))
    G.add_edges_from((a - 1, n1 + b - 1) for a, b in pairs)
    print(f"{path}: header '{lines[0]}' '{lines[1]}'; {m} edge lines; n1={n1} n2={n2}; "
          f"bipartite={nx.is_bipartite(G)}")
    return G


def from_hog(path):
    import json
    rec = json.load(open(path))
    adj = rec['adjacencyList']
    G = nx.Graph()
    G.add_nodes_from(range(len(adj)))
    G.add_edges_from((u, v) for u, nbrs in enumerate(adj) for v in nbrs)
    e = rec['entity']
    print(f"{path}: House of Graphs id {e['graphId']} '{e.get('graphName')}', canonical form {e['canonicalForm']}")
    return G


def from_newman_gml(path):
    import re
    txt = open(path).read()
    ids = [int(x) for x in re.findall(r'node\s*\[\s*id\s+(\d+)', txt)]
    pairs = [(int(a), int(b)) for a, b in re.findall(r'source\s+(\d+)\s+target\s+(\d+)', txt)]
    print(f"{path}: {len(ids)} node records, {len(pairs)} edge records")
    G = nx.Graph()
    G.add_nodes_from(sorted(ids))
    G.add_edges_from(pairs)
    return nx.convert_node_labels_to_integers(G, ordering='sorted')


NAMED = {'petersen': nx.petersen_graph}


def main():
    mode = sys.argv[1] if len(sys.argv) > 3 else ''
    if mode == 'newman-gml':
        G = from_newman_gml(sys.argv[2])
        out, gcc = sys.argv[3], False
    elif mode == 'hog':
        G = from_hog(sys.argv[2])
        out, gcc = sys.argv[3], False
        if '--same-as' in sys.argv:
            name = sys.argv[sys.argv.index('--same-as') + 1]
            print(f"isomorphic to networkx {name}: {nx.is_isomorphic(G, NAMED[name]())}")
    elif mode == 'konect-bip':
        G = from_konect_bip(sys.argv[2])
        out, gcc = sys.argv[3], False
    elif mode == 'mtx':
        G = from_mtx(sys.argv[2])
        out = sys.argv[3]
        gcc = '--gcc' in sys.argv[4:]
    elif mode == 'named':
        name, out, gcc = sys.argv[2], sys.argv[3], False
        G = NAMED[name]()
    else:
        raise SystemExit(__doc__)
    loops = nx.number_of_selfloops(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    print(f"undirected simple graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"self-loops dropped {loops}; components {len(comps)} (largest {comps[0]}); "
          f"isolates {nx.number_of_isolates(G)}")
    if gcc:
        cc = max(nx.connected_components(G), key=len)
        G = nx.convert_node_labels_to_integers(G.subgraph(cc).copy(), ordering='sorted')
        print(f"kept largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    edges = np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)
    np.savez(out, n_total=G.number_of_nodes(), edges=edges)
    print("wrote", out)


if __name__ == '__main__':
    main()
