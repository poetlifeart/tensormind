#!/usr/bin/env python3
"""Save v14 and v146 (swap=6500) as .npz for use with combinatorial_metrics.py"""
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import defaultdict
import sys, time, os

# Paths are resolved relative to THIS file so the script works from any cwd.
# HERE = .../construction/global/quotient ; the mild parent is one level up,
# where build_mild_graph.sh writes it.
HERE = os.path.dirname(os.path.abspath(__file__))

sys.stdout.reconfigure(line_buffering=True)

PARENT_PATH = os.path.join(HERE, '..', 'graph_brain_mild.npz')
# original (pre-tensormind): '/home/vahid/braingptgraph/graph_brain_mild.npz'
SEED = 42

def load_brain(path):
    d = np.load(path)
    n1, n2, n3 = int(d['n1']), int(d['n2']), int(d['n3'])
    off2, off3 = n1, n1 + n2
    G = nx.Graph(); G.add_nodes_from(range(n1+n2+n3))
    for mk, o_r, o_c in [('mask_12',off2,0),('mask_13',off3,0),('mask_23',off3,off2)]:
        rows, cols = np.where(d[mk] > 0)
        G.add_edges_from((int(c)+o_c, int(r)+o_r) for r,c in zip(rows, cols))
    return G

def coarsen(Gp, target=1015):
    comms = louvain_communities(Gp, resolution=20.0, seed=SEED)
    comm_list = [set(c) for c in sorted(comms, key=lambda c: -len(c))]
    n2c = {}
    for ci, c in enumerate(comm_list):
        for node in c: n2c[node] = ci
    while len(comm_list) > target:
        mi = min(range(len(comm_list)), key=lambda i: len(comm_list[i]))
        mc = comm_list[mi]
        ne = defaultdict(int)
        for node in mc:
            for nb in Gp.neighbors(node):
                nc = n2c[nb]
                if nc != mi: ne[nc] += 1
        if not ne:
            sizes = [(len(comm_list[i]),i) for i in range(len(comm_list)) if i!=mi]
            bt = min(sizes)[1]
        else:
            bt = max(ne, key=lambda c: ne[c]/(len(mc)*len(comm_list[c])))
        comm_list[bt] = comm_list[bt] | mc
        for node in mc: n2c[node] = bt
        comm_list.pop(mi)
        for ci, c in enumerate(comm_list):
            for node in c: n2c[node] = ci
    return comm_list, n2c

def build_v14(Q, witness, n):
    L0 = sorted(louvain_communities(Q, resolution=0.95, seed=SEED, weight=None), key=lambda c: -len(c))
    n2L0, n2L1, n2L2 = {}, {}, {}
    for bi, bl in enumerate(L0):
        for nd in bl: n2L0[nd] = bi
    bp = {0:(0.9,1.0), 1:(0.7,1.0), 2:(1.0,2.0), 3:(1.0,2.0)}
    for bi, bl in enumerate(L0):
        g1, g2 = bp.get(bi, (1.0, 2.0))
        sub = Q.subgraph(bl)
        L1c = sorted(louvain_communities(sub, resolution=g1, seed=SEED, weight='weight'), key=lambda c: -len(c))
        for gi, gr in enumerate(L1c):
            for nd in gr: n2L1[nd] = (bi, gi)
            sub2 = Q.subgraph(gr)
            L2c = sorted(louvain_communities(sub2, resolution=g2, seed=SEED, weight='weight'), key=lambda c: -len(c))
            for si, sc in enumerate(L2c):
                for nd in sc: n2L2[nd] = (bi, gi, si)
    t1, t2, t3, t4 = [], [], [], []
    for (u, v), w in witness.items():
        if n2L0[u] != n2L0[v]: t4.append((u, v, w))
        elif n2L1[u] != n2L1[v]: t3.append((u, v, w))
        elif n2L2[u] != n2L2[v]: t2.append((u, v, w))
        else: t1.append((u, v, w))
    target_e = int(139 * n / 2)
    final = set()
    for u, v, w in t1 + t2 + t3: final.add((min(u, v), max(u, v)))
    needed = target_e - len(final)
    if needed > 0:
        pe = defaultdict(list)
        for u, v, w in t4:
            L1u, L1v = n2L1[u], n2L1[v]
            key = (L1u, L1v) if L1u < L1v else (L1v, L1u)
            pe[key].append((u, v, w))
        pw = {k: sum(w for _, _, w in v) for k, v in pe.items()}
        ranked = sorted(pw, key=lambda k: -pw[k])[:10]
        tw = sum(pw[p] for p in ranked) or 1
        for p in ranked: pe[p] = sorted(pe[p], key=lambda x: -x[2])
        alloc = {p: max(1, int(needed * pw[p] / tw)) for p in ranked}
        added = 0
        for p in ranked:
            if added >= needed: break
            b = min(alloc[p], needed - added)
            for i, (u, v, w) in enumerate(pe[p]):
                if i >= b: break
                e = (min(u, v), max(u, v))
                if e not in final: final.add(e); added += 1
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(final)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        cmap = {}
        for ci, c in enumerate(comps):
            for nd in c: cmap[nd] = ci
        for u, v, w in sorted(t4, key=lambda x: -x[2]):
            if cmap[u] != cmap[v]:
                final.add((min(u, v), max(u, v))); G.add_edge(u, v)
                comps = list(nx.connected_components(G))
                cmap = {}
                for ci, c in enumerate(comps):
                    for nd in c: cmap[nd] = ci
                if len(comps) == 1: break
    return final, n2L0, n2L1, n2L2, L0

def save_npz(edges_set, n, path):
    edges = np.array(sorted(edges_set), dtype=np.int32)
    n3 = n // 3
    np.savez(path, n1=n3, n2=n3, n3=n - 2*n3, edges=edges)
    print(f"Saved {path}: {n} nodes, {len(edges)} edges")

def main():
    t0 = time.time()
    print("Loading parent...")
    Gp = load_brain(PARENT_PATH)
    print("Coarsening...")
    cl, n2c = coarsen(Gp)
    n = len(cl)
    witness = defaultdict(int)
    for u, v in Gp.edges():
        cu, cv = n2c[u], n2c[v]
        if cu != cv: witness[(min(cu, cv), max(cu, cv))] += 1
    Q = nx.Graph(); Q.add_nodes_from(range(n))
    for (u, v), w in witness.items(): Q.add_edge(u, v, weight=w)

    print("Building v14...")
    final14, n2L0, n2L1, n2L2, L0 = build_v14(Q, witness, n)
    G14 = nx.Graph(); G14.add_nodes_from(range(n)); G14.add_edges_from(final14)
    save_npz(final14, n, os.path.join(HERE, 'graph_v14.npz'))

    # v146 swap=6500
    print("Building v146 swap=6500...")
    edges_v14 = set((min(u,v), max(u,v)) for u,v in G14.edges())
    witness_w = {}
    for (u,v), w in witness.items():
        witness_w[(min(u,v), max(u,v))] = w
    adj = defaultdict(set)
    for u, v in edges_v14:
        adj[u].add(v); adj[v].add(u)
    t3_edges = [e for e in edges_v14 if n2L0[e[0]]==n2L0[e[1]] and n2L1[e[0]]!=n2L1[e[1]]]
    t3_sorted = sorted(t3_edges, key=lambda e: witness_w.get(e, 0))
    L1_groups = defaultdict(list)
    for nd in range(n): L1_groups[n2L1[nd]].append(nd)
    pool = []
    for key, nodes in L1_groups.items():
        ns = sorted(nodes)
        for i in range(len(ns)):
            for j in range(i+1, len(ns)):
                e = (ns[i], ns[j])
                if e not in edges_v14:
                    pool.append((e, len(adj[ns[i]] & adj[ns[j]])))
    pool.sort(key=lambda x: -x[1])

    swap_n = 6500
    edges_new = set(edges_v14)
    removed = []
    for i in range(swap_n):
        edges_new.discard(t3_sorted[i]); removed.append(t3_sorted[i])
    added = 0
    for e, sc in pool:
        if added >= swap_n: break
        if e not in edges_new: edges_new.add(e); added += 1
    Gnew = nx.Graph(); Gnew.add_nodes_from(range(n)); Gnew.add_edges_from(edges_new)
    if not nx.is_connected(Gnew):
        comps = list(nx.connected_components(Gnew))
        cmap = {}
        for ci, comp in enumerate(comps):
            for nd in comp: cmap[nd] = ci
        for e in sorted(removed, key=lambda e: -witness_w.get(e, 0)):
            u, v = e
            if cmap.get(u,-1) != cmap.get(v,-2):
                edges_new.add(e); Gnew.add_edge(u, v)
                comps = list(nx.connected_components(Gnew))
                cmap = {}
                for ci, comp in enumerate(comps):
                    for nd in comp: cmap[nd] = ci
                if len(comps) == 1: break
    save_npz(edges_new, n, os.path.join(HERE, 'graph_v146_6500.npz'))
    print(f"Done in {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
