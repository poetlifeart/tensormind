import networkx as nx

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

