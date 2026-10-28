#!/usr/bin/env python3
"""
gold_connector_hubs.py — the 'connector_hubs' criterion of
connectome_audit_gold.py, on its own.

Guimera & Amaral (2005) node roles on the Louvain partition of the graph's GCC
(_louvain: networkx louvain_communities, gamma = 1, seed = --seed):

    P_i  = 1 - sum_s (k_is / k_i)^2               participation coefficient
           (k_is = links of i to module s, k_i = degree of i)
    z_i  = (k_i,own - mean_own) / std_own          within-module degree z-score
           (k_i,own = links of i inside its own module; mean and std over the
           members of that module, std with ddof = 0; std = 1 for a one-node
           module; z = 0 when std = 0)

    connector hub (role R6)  z >= 2.5  and  0.30 < P <= 0.75
    kinless hub   (role R7)  z >= 2.5  and  P > 0.75
    verdict = 'pass' if #connector hubs > 0 else 'fail'

There is no null ensemble in this criterion, so no --n-null / --workers; the
result is the same with or without the audit's --no-null.

Self-contained: needs numpy and networkx only, and does not import
connectome_audit_gold.py.  The loading, Louvain and participation functions
below are copied from it unchanged; the within-module z-score and the role
masks are run_audit's inline code ('4. Community structure') moved, line for
line, into _within_module_z and connector_hubs.  So with the same --graph and
--seed the result is the same as the full audit's 'community' entry and its
'connector_hubs' verdict.

Usage:
    python gold_connector_hubs.py --graph G.npz
    python gold_connector_hubs.py --graph G.npz --seed 42 --json hubs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════════
# Loading  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LayeredGraph:
    G: nx.Graph
    masks: Dict[str, np.ndarray]
    sizes: Tuple[int, int, int]
    has_edge_list: bool = False


def load_layered_graph(path: str) -> LayeredGraph:
    if path.endswith('.gml'):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.Graph(G)
        n = G.number_of_nodes()
        n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
        masks = {k: np.zeros((1, 1), dtype=np.uint8) for k in ['12', '13', '23']}
        print(f"  [Loaded GML: {n} nodes, {G.number_of_edges()} edges]")
        return LayeredGraph(G=G, masks=masks, sizes=(n1, n2, n3), has_edge_list=True)

    data = np.load(path)

    # Check for edge-list format FIRST (χ>3 graphs may lack masks)
    if 'edges' in data:
        n_total = int(data['n_total']) if 'n_total' in data else (
            int(data['n1']) + int(data['n2']) + int(data['n3']))
        edges = data['edges']
        G = nx.Graph()
        G.add_nodes_from(range(n_total))
        G.add_edges_from((int(u), int(v)) for u, v in edges)
        print(f"  [Loaded from edge list: {G.number_of_edges()} edges]")
        if 'n1' in data:
            n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        else:
            n1 = n_total // 3
            n2 = n_total // 3
            n3 = n_total - 2 * (n_total // 3)
        masks = {}
        for key in ['mask_12', 'mask_13', 'mask_23']:
            if key in data:
                masks[key[5:]] = (data[key] > 0).astype(np.uint8)
            else:
                masks[key[5:]] = np.zeros((1, 1), dtype=np.uint8)
        return LayeredGraph(G=G, masks=masks,
                            sizes=(n1, n2, n3), has_edge_list=True)

    # Standard mask-based loading (χ=3 tripartite graphs)
    n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
    mask_12 = (data['mask_12'] > 0).astype(np.uint8)
    mask_13 = (data['mask_13'] > 0).astype(np.uint8)
    mask_23 = (data['mask_23'] > 0).astype(np.uint8)
    G = _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23)
    return LayeredGraph(G=G, masks={'12': mask_12, '13': mask_13, '23': mask_23},
                        sizes=(n1, n2, n3), has_edge_list=False)


def _graph_from_masks(n1, n2, n3, mask_12, mask_13, mask_23):
    G = nx.Graph()
    off1, off2, off3 = 0, n1, n1 + n2
    G.add_nodes_from(range(n1 + n2 + n3))
    rows, cols = np.where(mask_12)
    G.add_edges_from((off1 + int(c), off2 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_13)
    G.add_edges_from((off1 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    rows, cols = np.where(mask_23)
    G.add_edges_from((off2 + int(c), off3 + int(r)) for r, c in zip(rows, cols))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Utilities and community structure  (copied from connectome_audit_gold.py)
# ═══════════════════════════════════════════════════════════════════════════


def _gcc(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def _louvain(G: nx.Graph, seed: int = 42, resolution: float = 1.0):
    try:
        from networkx.algorithms.community import louvain_communities, modularity
        comms = louvain_communities(G, seed=seed, resolution=resolution)
        Q = float(modularity(G, comms, resolution=resolution))
    except Exception:
        import community as community_louvain
        part = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
        by_c = {}
        for node, c in part.items():
            by_c.setdefault(c, set()).add(node)
        comms = list(by_c.values())
        Q = float(community_louvain.modularity(part, G))
    partition = {}
    for i, comm in enumerate(comms):
        for node in comm:
            partition[node] = i
    return Q, comms, partition


def _participation(G: nx.Graph, partition: Dict[int, int]) -> np.ndarray:
    pcs = np.zeros(G.number_of_nodes(), dtype=float)
    for idx, node in enumerate(G.nodes()):
        k = G.degree(node)
        if k == 0:
            continue
        counts: Dict[int, int] = {}
        for nb in G.neighbors(node):
            c = partition[nb]
            counts[c] = counts.get(c, 0) + 1
        pcs[idx] = 1.0 - sum((v / k) ** 2 for v in counts.values())
    return pcs


# ═══════════════════════════════════════════════════════════════════════════
# The criterion  (run_audit's inline code, '4. Community structure')
# ═══════════════════════════════════════════════════════════════════════════


def _within_module_z(Gcc: nx.Graph, partition: Dict[int, int]) -> np.ndarray:
    """run_audit lines 'z_wm = ...' to 'z_wm[i] = ...', unchanged."""
    # Guimerà & Amaral (2005): within-module degree z-score > 2.5 AND P > 0.3
    # z_i = (k_is - mean(k_s)) / std(k_s) where k_is = edges from i to own module
    nodes = list(Gcc.nodes())
    z_wm = np.zeros(len(nodes), dtype=float)

    # Pre-compute module membership sets
    module_sets = {}
    for nd, c in partition.items():
        module_sets.setdefault(c, set()).add(nd)

    # Pre-compute within-module degrees for all nodes
    within_deg = {}
    for nd in nodes:
        c = partition[nd]
        ms = module_sets[c]
        within_deg[nd] = sum(1 for nb in Gcc.neighbors(nd) if nb in ms)

    # Pre-compute per-module statistics
    module_stats = {}
    for c, members in module_sets.items():
        degs = np.array([within_deg[nd] for nd in members], dtype=float)
        mu = degs.mean()
        sd = degs.std(ddof=0) if len(degs) > 1 else 1.0
        module_stats[c] = (mu, sd)

    for i, node in enumerate(nodes):
        c = partition[node]
        mu, sd = module_stats[c]
        z_wm[i] = (within_deg[node] - mu) / sd if sd > 0 else 0.0
    return z_wm


def connector_hubs(layered: LayeredGraph, seed: int = 42) -> Dict[str, object]:
    G = layered.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    Gcc = _gcc(G)
    print(f"  Nodes: {n:,}   Edges: {m:,}   GCC: {Gcc.number_of_nodes():,} nodes")

    Q, comms, partition = _louvain(Gcc, seed=seed)
    print(f"  Louvain (gamma=1, GCC, seed {seed}): Q = {Q:.6g}, {len(comms)} communities")

    pcs = _participation(Gcc, partition)
    z_wm = _within_module_z(Gcc, partition)
    nodes = list(Gcc.nodes())

    # Guimerà & Amaral 2005 role R6: connector hubs have z >= 2.5,
    # 0.30 < P <= 0.75.  R7 (kinless hubs) have P > 0.75.
    connector_mask = (z_wm >= 2.5) & (pcs > 0.30) & (pcs <= 0.75)
    kinless_mask = (z_wm >= 2.5) & (pcs > 0.75)
    connector_hubs = int(np.sum(connector_mask))
    kinless_hubs = int(np.sum(kinless_mask))

    community = {
        'Q': Q, 'n_communities': len(comms),
        'largest': max(len(c) for c in comms),
        'participation_mean': float(np.mean(pcs)),
        'participation_median': float(np.median(pcs)),
        'connector_hubs': connector_hubs,
        'kinless_hubs': kinless_hubs,
    }
    per_node = [{'node': int(nd), 'degree': int(Gcc.degree(nd)), 'module': int(partition[nd]),
                 'P': float(pcs[i]), 'z': float(z_wm[i])} for i, nd in enumerate(nodes)]
    return {'graph': {'n_nodes': n, 'n_edges': m, 'gcc': Gcc.number_of_nodes()},
            'seed': seed, 'community': community,
            'connector_hub_nodes': [int(nodes[i]) for i in np.where(connector_mask)[0]],
            'kinless_hub_nodes': [int(nodes[i]) for i in np.where(kinless_mask)[0]],
            'per_node': per_node,
            'connector_hubs': 'pass' if connector_hubs > 0 else 'fail'}


def main():
    ap = argparse.ArgumentParser(description="connector_hubs, the seventh of the gold audit's nine criteria")
    ap.add_argument('--graph', required=True, help='Path to the graph (.npz or .gml)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    layered = load_layered_graph(args.graph)
    res = connector_hubs(layered, seed=args.seed)

    c = res['community']
    print("\n" + "=" * 72)
    print("CONNECTOR_HUBS")
    print("=" * 72)
    print(f"  {'Q':>16s}: {c['Q']:.4f}")
    print(f"  {'n_communities':>16s}: {c['n_communities']}")
    print(f"  {'largest':>16s}: {c['largest']}")
    print(f"  {'PC mean':>16s}: {c['participation_mean']:.4f}")
    print(f"  {'PC median':>16s}: {c['participation_median']:.4f}")
    print(f"  {'connector_hubs':>16s}: {c['connector_hubs']}   nodes {res['connector_hub_nodes']}")
    print(f"  {'kinless_hubs':>16s}: {c['kinless_hubs']}   nodes {res['kinless_hub_nodes']}")
    print(f"  connector_hubs: {res['connector_hubs'].upper()}   (pass if >= 1 node with z >= 2.5 and 0.30 < P <= 0.75)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote JSON to {args.json}")


if __name__ == '__main__':
    main()
