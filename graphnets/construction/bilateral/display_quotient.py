#!/usr/bin/env python3
"""
Display quotient graphs with two-level hierarchical ordering.

Each graph gets its own Louvain community detection at two resolutions:
  L1 (γ=1.0): large blocks → thick boundary lines
  L2 (γ=3.0): sub-communities → thin boundary lines
Within each L1 block, nodes are grouped by L2 sub-community,
then sorted by degree descending.

Usage:
  python display_quotient.py                          # default 3-panel
  python display_quotient.py graph1.npz graph2.npz    # custom graphs
"""

import sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from networkx.algorithms.community import louvain_communities

# All display is black and white — no colored boundaries, no colored edges.


def make_adj(G, order):
    n = len(order)
    idx = {node: i for i, node in enumerate(order)}
    A = np.zeros((n, n), dtype=np.uint8)
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1
    return A


def hierarchical_order(G):
    comms_L0 = louvain_communities(G, resolution=1.0, seed=42)
    comms_L0 = sorted(comms_L0, key=lambda c: -len(c))

    order, bounds_L0, bounds_L1, bounds_L2 = [], [], [], []
    for block in comms_L0:
        sub_G = G.subgraph(block)
        mid_comms = louvain_communities(sub_G, resolution=1.0, seed=42)
        mid_comms = sorted(mid_comms, key=lambda c: -len(c))

        for mc in mid_comms:
            sub_G2 = G.subgraph(mc)
            fine_comms = louvain_communities(sub_G2, resolution=1.5, seed=42)
            fine_comms = sorted(fine_comms, key=lambda c: -len(c))
            for fc in fine_comms:
                fc_sorted = sorted(fc, key=lambda v: -G.degree(v))
                order.extend(fc_sorted)
                bounds_L2.append(len(order))
            bounds_L1.append(len(order))
        bounds_L0.append(len(order))
    return order, bounds_L0, bounds_L1, bounds_L2


def load_npz_graph(path):
    d = np.load(path)
    n = int(d['n_total'])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(d['edges'].tolist())
    return G


NICE_NAMES = {
    'budapest_connectome': 'Budapest Connectome',
    'graph_cap30_fiedler_qonly_nocomp': 'Original Quotient',
    'graph_heterogeneous_quotient_v3': 'Heterogeneous Quotient (v3)',
}


def display_graphs(graphs, titles, outpath):
    cmap = LinearSegmentedColormap.from_list('bw', ['#ffffff', '#000000'], N=2)
    n_panels = len(graphs)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    for ax, G, title in zip(axes, graphs, titles):
        order, bounds_L0, bounds_L1, bounds_L2 = hierarchical_order(G)
        A = make_adj(G, order)
        # Scale to ~1000x1000 for consistent visual comparison
        nn = A.shape[0]
        if nn > 1200:
            factor = max(1, nn // 1000)
            new_n = nn // factor
            A_scaled = np.zeros((new_n, new_n), dtype=np.uint8)
            for i in range(new_n):
                for j in range(new_n):
                    A_scaled[i, j] = A[i*factor:(i+1)*factor, j*factor:(j+1)*factor].max()
            A = A_scaled
            scale = factor
        else:
            scale = 1
        ax.imshow(A, cmap=cmap, interpolation='none', aspect='equal')
        b_L0 = set(bounds_L0)
        b_L1 = set(bounds_L1)
        for b in bounds_L0[:-1]:
            ax.axhline(b/scale - 0.5, color='gray', lw=1.2, alpha=0.6)
            ax.axvline(b/scale - 0.5, color='gray', lw=1.2, alpha=0.6)
        for b in bounds_L1:
            if b not in b_L0:
                ax.axhline(b/scale - 0.5, color='gray', lw=0.5, alpha=0.4)
                ax.axvline(b/scale - 0.5, color='gray', lw=0.5, alpha=0.4)
        for b in bounds_L2:
            if b not in b_L0 and b not in b_L1:
                ax.axhline(b/scale - 0.5, color='gray', lw=0.15, alpha=0.25)
                ax.axvline(b/scale - 0.5, color='gray', lw=0.15, alpha=0.25)
        ne = G.number_of_edges()
        nn = G.number_of_nodes()
        ax.set_title(f'{title}\n{nn} nodes, {ne:,} edges',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f'Saved {outpath}')
    plt.close()


if __name__ == '__main__':
    BUDAPEST = '/home/vahid/Desktop/emergentgraph/graph_topology/validation_graphs/budapest_connectome.gml'
    ORIGINAL = '/home/vahid/Desktop/emergentgraph/graph_topology/graph_cap30_fiedler_qonly_nocomp.npz'
    V3 = '/home/vahid/Desktop/experiments/graph_heterogeneous_quotient_v3.npz'

    if len(sys.argv) > 1:
        # Last arg ending in .png is the output path; otherwise auto-generate
        args = sys.argv[1:]
        if args[-1].endswith('.png'):
            outpath = args.pop()
        else:
            stems = [a.split('/')[-1].replace('.npz','').replace('.gml','') for a in args]
            outpath = '/home/vahid/Desktop/experiments/connectivity_' + '_vs_'.join(stems) + '.png'
        paths = args
        graphs, titles = [], []
        for p in paths:
            if p.endswith('.gml'):
                G = nx.read_gml(p)
                G = nx.convert_node_labels_to_integers(G)
            else:
                G = load_npz_graph(p)
            graphs.append(G)
            stem = p.split('/')[-1].replace('.npz', '').replace('.gml', '')
            titles.append(NICE_NAMES.get(stem, stem))
    else:
        G_bud = nx.read_gml(BUDAPEST)
        G_bud = nx.convert_node_labels_to_integers(G_bud)
        G_orig = load_npz_graph(ORIGINAL)
        G_v3 = load_npz_graph(V3)

        graphs = [G_bud, G_orig, G_v3]
        titles = ['Budapest Connectome', 'Original Quotient', 'Heterogeneous Quotient (v3)']
        outpath = '/home/vahid/Desktop/experiments/connectivity_comparison.png'

    display_graphs(graphs, titles, outpath)
