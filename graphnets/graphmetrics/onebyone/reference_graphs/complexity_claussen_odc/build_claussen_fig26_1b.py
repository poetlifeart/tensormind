"""A graph with exactly the node-node link correlation matrix c_mn printed in
Claussen, "Offdiagonal complexity: a computationally quick network complexity
measure. Application to protein networks and cell division", in Mathematical
Modeling of Biological Systems, Vol. II (Birkhauser, Boston 2007), ch. 26, pp. 291-299,
arXiv:0712.4216, Fig. 26.1(b) ("Same network, links partly randomized").

Published there:  #k (nodes of degree k = 1..8) = 8 7 8 5 2 1 0 0
                  c_mn (m <= n, rows m = 1..8):
                    m=1: 0 1 4 0 2 1 0 0
                    m=2:   0 7 5 1 0 0 0
                    m=3:     2 4 4 1 0 0
                    m=4:       3 2 3 0 0
                    m=5:         0 1 0 0
                    m=6..8: all 0
                  diagonal sums (5,15,16,2,2,1,0,0);  OdC = 1.376939.

OdC depends on the graph only through c_mn, so any simple graph with this c_mn
has the published OdC.  The original network (a Sakaguchi self-organized
spatial network) is not available; this script builds one realization with
networkx.joint_degree_graph (Stanton & Pinar 2012), seed 0, and checks that its
degree counts and c_mn equal the published ones.  Writes
claussen_fig26_1b.npz (n_total + edges) next to this script.

(The companion matrix of Fig. 26.1(a) cannot be realized: its row sums give 6
edge ends at degree 5 and 9 at degree 8, but its #k gives 5 and 8, and its
diagonal sums total 41 while its #k gives 40 edges.)"""
import os
from collections import Counter
import numpy as np
import networkx as nx

nk = {1: 8, 2: 7, 3: 8, 4: 5, 5: 2, 6: 1, 7: 0, 8: 0}
rows = {1: [0, 1, 4, 0, 2, 1, 0, 0],
        2: [0, 7, 5, 1, 0, 0, 0],
        3: [2, 4, 4, 1, 0, 0],
        4: [3, 2, 3, 0, 0],
        5: [0, 1, 0, 0],
        6: [0, 0, 0], 7: [0, 0], 8: [0]}
c = {}
for m, r in rows.items():
    for j, v in enumerate(r):
        if v:
            c[(m, m + j)] = v

# networkx joint degree dict: jd[k][l] = number of edges between degree-k and
# degree-l nodes, symmetric, with jd[k][k] = 2 x (edges among degree-k nodes).
jd = {}
for (m, n), v in c.items():
    jd.setdefault(m, {}); jd.setdefault(n, {})
    if m == n:
        jd[m][m] = 2 * v
    else:
        jd[m][n] = v
        jd[n][m] = v
assert nx.is_valid_joint_degree(jd)
G = nx.joint_degree_graph(jd, seed=0)

deg = dict(G.degree())
got_nk = Counter(deg.values())
assert all(got_nk.get(k, 0) == v for k, v in nk.items()), got_nk
got_c = Counter((min(deg[u], deg[v]), max(deg[u], deg[v])) for u, v in G.edges())
assert dict(got_c) == c, (dict(got_c), c)
diag = [sum(v for (m, n), v in c.items() if n - m == k) for k in range(8)]
assert diag == [5, 15, 16, 2, 2, 1, 0, 0], diag
print(f"realization: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"components {nx.number_connected_components(G)}; #k and c_mn equal the published ones; "
      f"diagonal sums {diag}")
G = nx.convert_node_labels_to_integers(G)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'claussen_fig26_1b.npz')
np.savez(out, n_total=G.number_of_nodes(), edges=np.array(sorted(G.edges()), dtype=np.int64))
print("wrote", out)
