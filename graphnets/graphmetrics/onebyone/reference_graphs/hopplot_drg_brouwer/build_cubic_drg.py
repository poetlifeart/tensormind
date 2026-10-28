"""Build cubic distance-regular graphs (Brouwer's table), verify their
intersection arrays from the graph itself, and save edge-list .npz files.
Usage: python build_cubic_drg.py OUTDIR
"""
import itertools, json, os, sys
import numpy as np
import networkx as nx

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

# Published intersection arrays: Brouwer, "The cubic distance-regular graphs",
# https://www.win.tue.nl/~aeb/graphs/cubic_drg.html
PUBLISHED = {
    "petersen":       ([3, 2], [1, 1]),
    "heawood":        ([3, 2, 2], [1, 1, 3]),
    "coxeter":        ([3, 2, 2, 1], [1, 1, 1, 2]),
    "dodecahedron":   ([3, 2, 1, 1, 1], [1, 1, 1, 2, 3]),
    "desargues":      ([3, 2, 2, 1, 1], [1, 1, 2, 2, 3]),
    "foster":         ([3, 2, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 3]),
    "tutte_12cage":   ([3, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 3]),
}

def coxeter_from_fano():
    # Brouwer, Coxeter.html: "the graph on the triangles in the Fano plane,
    # where triangles are adjacent when they are disjoint."
    lines = [frozenset(l) for l in ([0,1,2],[0,3,4],[0,5,6],[1,3,5],[1,4,6],[2,3,6],[2,4,5])]
    tri = [frozenset(t) for t in itertools.combinations(range(7), 3) if frozenset(t) not in lines]
    G = nx.Graph(); G.add_nodes_from(range(len(tri)))
    for i, j in itertools.combinations(range(len(tri)), 2):
        if not (tri[i] & tri[j]):
            G.add_edge(i, j)
    return G

def build(name):
    if name == "petersen":     return nx.petersen_graph()
    if name == "heawood":      return nx.heawood_graph()
    if name == "dodecahedron": return nx.dodecahedral_graph()
    if name == "desargues":    return nx.desargues_graph()
    if name == "coxeter":      return coxeter_from_fano()
    if name == "foster":       return nx.LCF_graph(90, [17, -9, 37, -37, 9, -17], 15)
    if name == "tutte_12cage": return nx.LCF_graph(126, [17, 27, -13, -59, -35, 35, -11, 13, -53, 53, -27, 21, 57, 11, -21, -57, 59, -17], 7)
    raise KeyError(name)

def intersection_array(G):
    """Return (b, c) if G is distance-regular, else None."""
    D = dict(nx.all_pairs_shortest_path_length(G))
    diam = max(max(r.values()) for r in D.values())
    b = [None] * diam; c = [None] * (diam + 1)
    for u in G:
        for v, i in D[u].items():
            nb = [D[w][u] for w in G[v]]   # distances from u of v's neighbours
            bi = sum(1 for x in nb if x == i + 1); ci = sum(1 for x in nb if x == i - 1)
            if i < diam:
                if b[i] is None: b[i] = bi
                elif b[i] != bi: return None
            if i > 0:
                if c[i] is None: c[i] = ci
                elif c[i] != ci: return None
    return b, c[1:]

summary = {}
for name, (bpub, cpub) in PUBLISHED.items():
    G = build(name)
    G = nx.convert_node_labels_to_integers(G)
    ia = intersection_array(G)
    ok = ia is not None and list(ia[0]) == bpub and list(ia[1]) == cpub
    # k_i from the published array: k_0 = 1, k_{i+1} = k_i * b_i / c_{i+1}
    k = [1]
    for i in range(len(bpub)):
        k.append(k[-1] * bpub[i] // cpub[i])
    n = G.number_of_nodes(); m = G.number_of_edges()
    assert sum(k) == n, (name, k, n)
    cum = np.cumsum(k[1:])
    exp_frac = [float(cum[min(h, len(cum)) - 1]) / (n - 1) for h in range(1, 7)]
    edges = np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)
    np.savez(os.path.join(out, f"{name}.npz"), edges=edges)
    summary[name] = dict(n=n, m=m, published_b=bpub, published_c=cpub, measured_array=ia,
                         array_matches=ok, k_from_published=k,
                         expected_hop_frac_h1_6=exp_frac,
                         expected_hop_frac_h2_fraction=f"{int(cum[min(2,len(cum))-1])}/{n-1}")
    print(f"{name:13s} n={n:4d} m={m:4d} array_matches={ok} k={k} expected hop_frac[h=2]={int(cum[min(2,len(cum))-1])}/{n-1}={exp_frac[1]!r}")
with open(os.path.join(out, "drg_expected.json"), "w") as f:
    json.dump(summary, f, indent=1, default=str)
