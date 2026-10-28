"""Odd graph O4, Johnson J(8,3), Hamming H(4,3): build from Brouwer's
definitions, check distance counts, save .npz.  Usage: python build_more_drg.py OUTDIR"""
import itertools, json, os, sys
from math import comb
import numpy as np
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cubic_drg_lib import intersection_array

out = sys.argv[1]
def subsets_graph(n, m, adj):
    V = [frozenset(s) for s in itertools.combinations(range(n), m)]
    G = nx.Graph(); G.add_nodes_from(range(len(V)))
    for i, j in itertools.combinations(range(len(V)), 2):
        if adj(V[i], V[j]): G.add_edge(i, j)
    return G
graphs = {
  # Odd.html: O_{m+1} = m-subsets of a (2m+1)-set, adjacent when disjoint; O4 array {4,3,3;1,1,2}
  "odd_O4": (subsets_graph(7, 3, lambda a, b: not (a & b)), "array", ([4,3,3],[1,1,2])),
  # Johnson.html: J(n,m) = m-subsets, adjacent when Johnson distance |A^B|/2 = 1; k_i = C(m,i)C(n-m,i)
  "johnson_J8_3": (subsets_graph(8, 3, lambda a, b: len(a ^ b) == 2), "formula", [comb(3,i)*comb(5,i) for i in range(4)]),
  # Hamming.html: H(d,q) = product of d complete graphs K_q; k_i = C(d,i)(q-1)^i
  "hamming_H4_3": (nx.convert_node_labels_to_integers(nx.cartesian_product(nx.cartesian_product(nx.complete_graph(3), nx.complete_graph(3)), nx.cartesian_product(nx.complete_graph(3), nx.complete_graph(3)))), "formula", [comb(4,i)*2**i for i in range(5)]),
}
summary = {}
for name, (G, kind, pub) in graphs.items():
    G = nx.convert_node_labels_to_integers(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    ia = intersection_array(G)
    if kind == "array":
        b, c = pub; k = [1]
        for i in range(len(b)): k.append(k[-1] * b[i] // c[i])
        ok = ia is not None and list(ia[0]) == b and list(ia[1]) == c
    else:
        k = pub
        b, c = ia; kk = [1]
        for i in range(len(b)): kk.append(kk[-1] * b[i] // c[i])
        ok = kk == k
    assert sum(k) == n
    cum = np.cumsum(k[1:])
    exp = [float(cum[min(h, len(cum)) - 1]) / (n - 1) for h in range(1, 7)]
    edges = np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)
    np.savez(os.path.join(out, f"{name}.npz"), edges=edges)
    summary[name] = dict(n=n, m=m, kind=kind, published=pub, measured_array=ia, matches=ok, k=k, expected_hop_frac_h1_6=exp)
    print(f"{name:13s} n={n:3d} m={m:4d} {kind} matches={ok} k={k} expected hop_frac[h=2]={int(cum[1])}/{n-1}={exp[1]!r}")
with open(os.path.join(out, "drg_expected_more.json"), "w") as f:
    json.dump(summary, f, indent=1, default=str)
