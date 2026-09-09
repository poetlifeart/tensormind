#!/usr/bin/env python3
"""
odd_cycle_witness.py — non-bipartiteness invariant (A3) for the chromatic
tensor construction.

The paper's growth rule requires, at every scale k, that

    (A1) every added edge is chromatically legal,
    (A2) G_{k+1} is connected,
    (A3) G_{k+1} contains at least one odd cycle,

and states that one odd cycle is recorded as a witness immediately after
tensor expansion, then restored if pruning has made the graph bipartite.
(A1) and (A2) are enforced inside the pruning/repair routines; this module
supplies (A3).

Usage in a construction loop:

    from odd_cycle_witness import find_odd_cycle, enforce_odd_cycle

    n, edges = tensor_product(...)
    witness = find_odd_cycle(n, edges)          # record before pruning
    edges = prune_and_repair(...)               # may delete witness edges
    edges = enforce_odd_cycle(n, edges, witness)  # restore if bipartite

Restoring the witness is chromatically safe: every witness edge is an edge of
the tensor-expanded graph, which is cross-colour under the projected colouring,
so (A1) is preserved by the restoration.
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["find_odd_cycle", "is_bipartite", "enforce_odd_cycle"]


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    return adj


def find_odd_cycle(n, edges):
    """Return one odd cycle as a list of undirected edges, or None if the
    graph is bipartite.

    BFS two-colouring.  The first edge joining two vertices of equal parity
    closes an odd cycle; the cycle is recovered by walking both endpoints up
    their BFS parent chains to their meeting point.

    Runs in O(n + |E|).
    """
    adj = _adjacency(n, edges)
    parity = np.full(n, -1, dtype=np.int8)
    parent = np.full(n, -1, dtype=np.int64)

    for root in range(n):
        if parity[root] != -1:
            continue
        parity[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if parity[v] == -1:
                    parity[v] = 1 - parity[u]
                    parent[v] = u
                    queue.append(v)
                elif parity[v] == parity[u]:
                    return _cycle_through(u, v, parent)
    return None


def _cycle_through(u, v, parent):
    """Edges of the odd cycle closed by the equal-parity edge (u, v)."""
    # Ancestors of u, in order, up to the root.
    seen = {}
    x, depth = u, 0
    while x != -1:
        seen[x] = depth
        x = int(parent[x])
        depth += 1

    # Walk v upward until it meets that chain.
    path_v = [v]
    y = int(parent[v])
    while y != -1 and y not in seen:
        path_v.append(y)
        y = int(parent[y])
    meet = y if y != -1 else u

    path_u = []
    x = u
    while x != meet and x != -1:
        path_u.append(x)
        x = int(parent[x])
    path_u.append(meet)

    walk = path_u + path_v[::-1]
    cycle = [(min(walk[i], walk[i + 1]), max(walk[i], walk[i + 1]))
             for i in range(len(walk) - 1)]
    cycle.append((min(u, v), max(u, v)))
    return cycle


def is_bipartite(n, edges):
    """True when the graph contains no odd cycle."""
    return find_odd_cycle(n, edges) is None


def enforce_odd_cycle(n, edges, witness, verbose=True):
    """Guarantee (A3).

    If the graph already contains an odd cycle, return `edges` unchanged.
    Otherwise restore the edges of `witness` that are missing, which
    reinstates that odd cycle.

    `edges` may be an (m, 2) array or an iterable of pairs; the return type
    matches an (m, 2) int64 array.  `witness` is the list returned by
    find_odd_cycle, or None.
    """
    edge_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}

    if find_odd_cycle(n, edge_set) is not None:
        if verbose:
            print("  (A3) odd cycle present; no restoration needed")
        return np.array(sorted(edge_set), dtype=np.int64)

    if not witness:
        raise RuntimeError(
            "graph is bipartite and no odd-cycle witness was recorded; "
            "invariant (A3) cannot be restored")

    missing = [e for e in witness if e not in edge_set]
    edge_set.update(missing)

    if verbose:
        print(f"  (A3) graph was bipartite; restored {len(missing)} "
              f"witness edge(s) of a {len(witness)}-cycle")

    if find_odd_cycle(n, edge_set) is None:
        raise RuntimeError("witness restoration failed to reinstate an odd cycle")

    return np.array(sorted(edge_set), dtype=np.int64)
