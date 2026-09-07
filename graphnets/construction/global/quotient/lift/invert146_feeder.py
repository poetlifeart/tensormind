#!/usr/bin/env python3
"""
Invert v146: map non-witness quotient edges back to parent-level edges
using feeder-lift rule (one local hub + one non-hub per edge).

For each non-witness edge between supernodes A and B:
  1. Compute local 80th percentile degree within each supernode
  2. Find cross-color pairs where exactly one endpoint is a local hub
     (degree >= 80th percentile of its supernode)
  3. Pick the feeder pair minimizing:
       key = (load(a) + load(b), -(deg(a) + deg(b)), a, b)
  4. Fallback: relax to 70th percentile, then to any min-load pair

Hub-mediated but not hub-saturated: every new edge touches one locally
important node and one satellite, preserving routing without excessive
hub-hub shortcutting.

Runs connectome_audit_gold with 100 nulls on the result.
"""
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import defaultdict
import sys, os, time

sys.stdout.reconfigure(line_buffering=True)

# Paths resolved relative to THIS file. HERE = .../quotient/lift ;
# the mild parent is two levels up, where build_mild_graph.sh writes it.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = os.path.join(HERE, '..', '..', 'graph_brain_mild.npz')
# original (pre-tensormind): '/home/vahid/braingptgraph/graph_brain_mild.npz'
OUT_DIR = HERE
# original (pre-tensormind): '/home/vahid/Desktop/v146results'
SEED = 42


def load_brain(path):
    d = np.load(path)
    n1, n2, n3 = int(d['n1']), int(d['n2']), int(d['n3'])
    off2, off3 = n1, n1 + n2
    G = nx.Graph(); G.add_nodes_from(range(n1+n2+n3))
    for mk, o_r, o_c in [('mask_12',off2,0),('mask_13',off3,0),('mask_23',off3,off2)]:
        rows, cols = np.where(d[mk] > 0)
        G.add_edges_from((int(c)+o_c, int(r)+o_r) for r,c in zip(rows, cols))
    return G, n1, n2, n3


def coarsen(Gp, target=1015):
    comms = louvain_communities(Gp, resolution=20.0, seed=SEED)
    comm_list = [set(c) for c in sorted(comms, key=lambda c: -len(c))]
    n2c = {}
    for ci, c in enumerate(comm_list):
        for node in c: n2c[node] = ci
    print(f"  Louvain initial: {len(comm_list)} clusters")
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
    L0 = sorted(louvain_communities(Q, resolution=0.95, seed=SEED, weight=None),
                key=lambda c: -len(c))
    n2L0, n2L1, n2L2 = {}, {}, {}
    for bi, bl in enumerate(L0):
        for nd in bl: n2L0[nd] = bi
    bp = {0:(0.9,1.0), 1:(0.7,1.0), 2:(1.0,2.0), 3:(1.0,2.0)}
    for bi, bl in enumerate(L0):
        g1, g2 = bp.get(bi, (1.0, 2.0))
        sub = Q.subgraph(bl)
        L1c = sorted(louvain_communities(sub, resolution=g1, seed=SEED, weight='weight'),
                      key=lambda c: -len(c))
        for gi, gr in enumerate(L1c):
            for nd in gr: n2L1[nd] = (bi, gi)
            sub2 = Q.subgraph(gr)
            L2c = sorted(louvain_communities(sub2, resolution=g2, seed=SEED, weight='weight'),
                          key=lambda c: -len(c))
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


t0 = time.time()

# === Load parent with colors ===
print("Loading parent...")
Gp, n1, n2, n3 = load_brain(PARENT_PATH)
print(f"  Parent: {Gp.number_of_nodes()} nodes, {Gp.number_of_edges()} edges")

color_of = {}
for i in range(n1): color_of[i] = 0
for i in range(n1, n1+n2): color_of[i] = 1
for i in range(n1+n2, n1+n2+n3): color_of[i] = 2

degree = dict(Gp.degree())

# === Coarsen and build v14 ===
print("Coarsening...")
cl, n2c = coarsen(Gp)
n = len(cl)
print(f"  Supernodes: {n}")

witness = defaultdict(int)
for u, v in Gp.edges():
    cu, cv = n2c[u], n2c[v]
    if cu != cv: witness[(min(cu, cv), max(cu, cv))] += 1

Q = nx.Graph(); Q.add_nodes_from(range(n))
for (u, v), w in witness.items(): Q.add_edge(u, v, weight=w)

print("Building v14...")
final14, n2L0, n2L1, n2L2, L0 = build_v14(Q, witness, n)
G14 = nx.Graph(); G14.add_nodes_from(range(n)); G14.add_edges_from(final14)
print(f"  v14: {len(final14)} edges")

# === Reproduce v146 swap=6500 ===
print("\nReproducing v146 swap=6500...")
edges_v14 = set((min(u,v), max(u,v)) for u, v in G14.edges())
witness_w = {(min(u,v), max(u,v)): w for (u,v), w in witness.items()}
witness_set = set(witness_w.keys())

t3_edges = [e for e in edges_v14 if n2L0[e[0]]==n2L0[e[1]] and n2L1[e[0]]!=n2L1[e[1]]]
t3_sorted = sorted(t3_edges, key=lambda e: witness_w.get(e, 0))

L1_groups = defaultdict(list)
for nd in range(n): L1_groups[n2L1[nd]].append(nd)

adj = defaultdict(set)
for u, v in edges_v14:
    adj[u].add(v); adj[v].add(u)

pool = []
for key, nodes in L1_groups.items():
    nodes_sorted = sorted(nodes)
    for i in range(len(nodes_sorted)):
        for j in range(i+1, len(nodes_sorted)):
            u, v = nodes_sorted[i], nodes_sorted[j]
            e = (u, v)
            if e not in edges_v14:
                common = len(adj[u] & adj[v])
                pool.append((e, common))
pool.sort(key=lambda x: -x[1])

SWAP_N = 6500
actual_swap = min(SWAP_N, len(t3_sorted), len(pool))
if actual_swap < SWAP_N:
    raise RuntimeError(f"Did not complete requested {SWAP_N} swaps (t3={len(t3_sorted)}, pool={len(pool)})")

edges_v146 = set(edges_v14)
removed = []
for i in range(actual_swap):
    edges_v146.discard(t3_sorted[i])
    removed.append(t3_sorted[i])

added_edges = []
added_count = 0
for e, score in pool:
    if added_count >= actual_swap: break
    if e not in edges_v146:
        edges_v146.add(e)
        added_edges.append(e)
        added_count += 1

# Connectivity fix
repair_count = 0
Gv146 = nx.Graph(); Gv146.add_nodes_from(range(n)); Gv146.add_edges_from(edges_v146)
if not nx.is_connected(Gv146):
    comps = list(nx.connected_components(Gv146))
    cmap = {}
    for ci, comp in enumerate(comps):
        for nd in comp: cmap[nd] = ci
    for e in sorted(removed, key=lambda e: -witness_w.get(e, 0)):
        u, v = e
        if cmap.get(u, -1) != cmap.get(v, -2):
            edges_v146.add(e); Gv146.add_edge(u, v)
            repair_count += 1
            comps = list(nx.connected_components(Gv146))
            cmap = {}
            for ci, comp in enumerate(comps):
                for nd in comp: cmap[nd] = ci
            if len(comps) == 1: break

if repair_count > 0:
    raise RuntimeError(f"Connectivity repair changed quotient edge count: re-added {repair_count} edges")

non_witness_added = [e for e in added_edges if e not in witness_set]
print(f"  Final v146 quotient edges: {len(edges_v146)}")
print(f"  Added edges: {len(added_edges)} ({len(non_witness_added)} non-witness)")

# === Precompute local hub thresholds per supernode ===
print(f"\n=== Computing local hub thresholds ===")
local_p80 = {}
local_p70 = {}
for si in range(n):
    nodes_s = sorted(cl[si])
    if len(nodes_s) == 0:
        local_p80[si] = 0
        local_p70[si] = 0
        continue
    degs = sorted([degree[nd] for nd in nodes_s])
    local_p80[si] = degs[int(0.8 * (len(degs) - 1))]
    local_p70[si] = degs[int(0.7 * (len(degs) - 1))]

hub_sizes = []
for si in range(n):
    h = sum(1 for nd in cl[si] if degree[nd] >= local_p80[si])
    hub_sizes.append(h)
print(f"  Local hub (p80) nodes per supernode: mean={np.mean(hub_sizes):.1f}, "
      f"min={min(hub_sizes)}, max={max(hub_sizes)}")

# === Invert: feeder-lift rule ===
print(f"\n=== Inverting {len(non_witness_added)} non-witness edges (feeder-lift) ===")

new_load = defaultdict(int)
new_parent_edges = []
failed = 0
feeder_p80 = 0
feeder_p70 = 0
fallback_minload = 0

for idx, (sA, sB) in enumerate(non_witness_added):
    if (idx+1) % 500 == 0:
        print(f"  {idx+1}/{len(non_witness_added)}...")

    nodes_A = sorted(cl[sA])
    nodes_B = sorted(cl[sB])
    thresh_A_80 = local_p80[sA]
    thresh_B_80 = local_p80[sB]
    thresh_A_70 = local_p70[sA]
    thresh_B_70 = local_p70[sB]

    # Try p80 feeder pairs first
    best_pair = None
    best_key = None

    for a in nodes_A:
        hub_a_80 = degree[a] >= thresh_A_80
        for b in nodes_B:
            if color_of[a] == color_of[b]:
                continue
            if Gp.has_edge(a, b):
                continue
            hub_b_80 = degree[b] >= thresh_B_80
            if hub_a_80 == hub_b_80:
                continue
            load = new_load[a] + new_load[b]
            degree_score = degree[a] + degree[b]
            key = (load, -degree_score, a, b)
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (a, b)

    if best_pair is not None:
        feeder_p80 += 1
    else:
        # Fallback: try p70 threshold
        for a in nodes_A:
            hub_a_70 = degree[a] >= thresh_A_70
            for b in nodes_B:
                if color_of[a] == color_of[b]:
                    continue
                if Gp.has_edge(a, b):
                    continue
                hub_b_70 = degree[b] >= thresh_B_70
                if hub_a_70 == hub_b_70:
                    continue
                load = new_load[a] + new_load[b]
                degree_score = degree[a] + degree[b]
                key = (load, -degree_score, a, b)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (a, b)

        if best_pair is not None:
            feeder_p70 += 1
        else:
            # Final fallback: pure min-load (any cross-color pair)
            for a in nodes_A:
                for b in nodes_B:
                    if color_of[a] == color_of[b]:
                        continue
                    if Gp.has_edge(a, b):
                        continue
                    load = new_load[a] + new_load[b]
                    degree_score = degree[a] + degree[b]
                    key = (load, -degree_score, a, b)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pair = (a, b)

            if best_pair is not None:
                fallback_minload += 1

    if best_pair is None:
        failed += 1
        continue

    a, b = best_pair
    new_load[a] += 1
    new_load[b] += 1
    new_parent_edges.append((a, b, best_key[0], sA, sB))

print(f"\n  Mapped: {len(new_parent_edges)} edges")
print(f"  Failed: {failed}")
print(f"  Feeder p80: {feeder_p80}")
print(f"  Feeder p70 fallback: {feeder_p70}")
print(f"  Min-load fallback: {fallback_minload}")

# Load distribution
loads = [v for v in new_load.values() if v > 0]
if loads:
    print(f"  Load distribution: max={max(loads)}, mean={np.mean(loads):.1f}, "
          f"nodes_used={len(loads)}")

# Hub vs non-hub endpoint stats
if new_parent_edges:
    degs_a = [degree[a] for a, _, _, _, _ in new_parent_edges]
    degs_b = [degree[b] for _, b, _, _, _ in new_parent_edges]
    hub_endpoints = 0
    nonhub_endpoints = 0
    for a, b, _, sA, sB in new_parent_edges:
        if degree[a] >= local_p80[sA]: hub_endpoints += 1
        else: nonhub_endpoints += 1
        if degree[b] >= local_p80[sB]: hub_endpoints += 1
        else: nonhub_endpoints += 1
    print(f"  Hub endpoints: {hub_endpoints}, Non-hub: {nonhub_endpoints}")
    print(f"  Selected node degrees: A mean={np.mean(degs_a):.1f}, B mean={np.mean(degs_b):.1f}")
    print(f"  Parent mean degree: {2*Gp.number_of_edges()/Gp.number_of_nodes():.1f}")

# === Build modified parent ===
print("\n=== Building modified parent ===")
Gp_new = Gp.copy()
actually_added = 0
for a, b, _, sA, sB in new_parent_edges:
    if not Gp_new.has_edge(a, b):
        Gp_new.add_edge(a, b)
        actually_added += 1

print(f"  Original parent: {Gp.number_of_edges()} edges")
print(f"  New parent edges: {actually_added}")
print(f"  Modified parent: {Gp_new.number_of_edges()} edges")
print(f"  Change: +{actually_added} ({100*actually_added/Gp.number_of_edges():.2f}%)")

# Verify: all v146 edges now witnessed
print("\n=== Verification ===")
witness_new = defaultdict(int)
for u, v in Gp_new.edges():
    cu, cv = n2c[u], n2c[v]
    if cu != cv:
        witness_new[(min(cu, cv), max(cu, cv))] += 1

unwitnessed = 0
for e in edges_v146:
    key = (min(e[0], e[1]), max(e[0], e[1]))
    if key not in witness_new:
        unwitnessed += 1

print(f"  v146 edges not witnessed by modified parent: {unwitnessed}")
if unwitnessed == 0:
    print(f"  ALL v146 edges are now witness-backed!")

if failed > 0 or unwitnessed > 0:
    print(f"\n  ABORT: failed={failed}, unwitnessed={unwitnessed}")
    print(f"  Inverted parent does not fully realize v146. Not saving.")
    sys.exit(1)

# === Save ===
out_npz = os.path.join(OUT_DIR, 'graph_brain_mild_v146_feeder.npz')

mask_12 = np.zeros((n2, n1), dtype=np.float32)
mask_13 = np.zeros((n3, n1), dtype=np.float32)
mask_23 = np.zeros((n3, n2), dtype=np.float32)

for u, v in Gp_new.edges():
    a, b = min(u, v), max(u, v)
    if a < n1 and n1 <= b < n1+n2:
        mask_12[b - n1, a] = 1.0
    elif a < n1 and b >= n1+n2:
        mask_13[b - n1 - n2, a] = 1.0
    elif n1 <= a < n1+n2 and b >= n1+n2:
        mask_23[b - n1 - n2, a - n1] = 1.0

np.savez(out_npz, n1=n1, n2=n2, n3=n3,
         mask_12=mask_12, mask_13=mask_13, mask_23=mask_23)
print(f"\nSaved {out_npz}")

out_edges = os.path.join(OUT_DIR, 'graph_brain_mild_v146_feeder_edges.npz')
edges_new = np.array(sorted(Gp_new.edges()), dtype=np.int64)
np.savez(out_edges,
         n1=n1, n2=n2, n3=n3, n_total=n1+n2+n3,
         edges=edges_new,
         added_edges=np.array([(a, b) for a, b, _, _, _ in new_parent_edges], dtype=np.int64),
         added_loads=np.array([l for _, _, l, _, _ in new_parent_edges], dtype=np.int64))
print(f"Saved {out_edges}")

# === Run audit with 100 nulls ===
print(f"\n{'='*60}")
print("RUNNING CONNECTOME AUDIT (100 nulls)")
print(f"{'='*60}")

sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'graphmetrics'))
# original (pre-tensormind): '/home/vahid/Desktop/emergentgraph/graph_topology'
from connectome_audit_gold import main as audit_main

audit_json = os.path.join(OUT_DIR, 'audit_v146_feeder_100.json')
sys.argv = [
    'connectome_audit_gold.py',
    '--graph', out_npz,
    '--n-null', '100',
    '--json', audit_json,
]
audit_main()

print(f"\nAudit saved: {audit_json}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
