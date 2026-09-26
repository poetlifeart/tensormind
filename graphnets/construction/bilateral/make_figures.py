#!/usr/bin/env python3
"""
make_figures.py -- regenerates every figure in figures/ from the shipped graphs.

    python3 make_figures.py                # all
    python3 make_figures.py --only AI      # one

Figures
  AA  parent | coarse quotient | Budapest, hierarchically community-ordered
      adjacency (uses display_quotient.py: three nested Louvain levels)
  AB  C(k) crescent, degree histogram, fibre-bundle histogram, coarse vs Budapest
  AD  the six static generative tests            (delegates to static_tests.py)
  AE  five display types side by side: force-directed, spectral embedding,
      community block densities, circular, degree-sorted adjacency
  AH  bilateral comparison, communities ordered so each half is contiguous,
      couplings annotated
  AI  the plain version of AH: communities by size descending, no annotation

All community detection is Louvain at resolution 1.0, best modularity over
20 seeds.  Do not substitute a single fixed seed -- see README, "The seed trap".
"""
import argparse, os, subprocess, sys
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from networkx.algorithms.community import louvain_communities, modularity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
G_DIR = os.path.join(HERE, 'graphs')
BUD = os.path.join(HERE, 'reference', 'budapest_1015_70654.edgelist')
BW = LinearSegmentedColormap.from_list('bw', ['#ffffff', '#000000'], N=2)
SEEDS = range(20)

sys.path.insert(0, HERE)
from stats import (load_masks, load_edges, budapest, best_partition,
                   block_density, gini)                      # noqa: E402


def coarse():
    return load_edges(os.path.join(G_DIR, 'cnew_coarse.npz'))


def parent():
    return load_masks(os.path.join(G_DIR, 'cnew_parent.npz'))[0]


def order_by_size(G):
    """Communities largest-first; nodes by degree inside each."""
    q, cs, seed = best_partition(G)
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    deg = dict(G.degree())
    perm, bounds = [], []
    for c in cs:
        perm += sorted((idx[v] for v in c), key=lambda i: -deg[nodes[i]])
        bounds.append(len(perm))
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()])
    return np.array(perm), bounds, [len(c) for c in cs], E, len(nodes), q, seed, cs


def adj_from(perm, E, n):
    r = np.empty(n, int)
    r[perm] = np.arange(n)
    A = np.zeros((n, n), np.uint8)
    A[r[E[:, 0]], r[E[:, 1]]] = 1
    A[r[E[:, 1]], r[E[:, 0]]] = 1
    return A


# ------------------------------------------------------------------ AI / AH

def _sub(G):
    """Subtitle derived from the graph, not typed in.  The counts used to be
    literals ('1015 nodes, 64,760 edges'), which would have gone quietly wrong
    the moment either graph changed.  Fixed 2026-09-25."""
    return '%d nodes, %s edges' % (G.number_of_nodes(),
                                   format(G.number_of_edges(), ','))


def fig_AI():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.3))
    _Q, _B = coarse(), budapest()
    for ax, (name, G, sub) in zip(axes, [
            ('8k coarse quotient', _Q, _sub(_Q)),
            ('Budapest Connectome', _B, _sub(_B))]):
        perm, bounds, sizes, E, n, q, seed, _ = order_by_size(G)
        ax.imshow(adj_from(perm, E, n), cmap=BW, interpolation='none', aspect='equal')
        for b in bounds[:-1]:
            ax.axhline(b - .5, color='gray', lw=.8, alpha=.5)
            ax.axvline(b - .5, color='gray', lw=.8, alpha=.5)
        starts = [0] + bounds[:-1]
        mid = [(s + b) / 2 for s, b in zip(starts, bounds)]
        ax.set_xticks(mid); ax.set_xticklabels(sizes, fontsize=12)
        ax.set_yticks(mid); ax.set_yticklabels(sizes, fontsize=12)
        ax.set_title('%s — %s\nQ = %.4f' % (name, sub, q), fontsize=12, fontweight='bold')
    fig.suptitle('Both graphs partition into 307 / 306 / 201 / 201',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, .945])
    fig.savefig(os.path.join(OUT, 'AI_bilateral_plain.png'), dpi=120)
    plt.close(fig)


def fig_AH():
    """As AI, but communities reordered so each hemisphere is contiguous,
    with the two hemispheric couplings and the empty blocks marked."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.6))
    _Q, _B = coarse(), budapest()
    for ax, (name, G, sub) in zip(axes, [
            ('8k coarse quotient', _Q, _sub(_Q)),
            ('Budapest Connectome', _B, _sub(_B))]):
        q, cs, seed = best_partition(G)
        Dn, sz = block_density(G, cs)
        K = len(sz)
        W = Dn.copy(); np.fill_diagonal(W, 0)
        used, order = set(), []
        for i in np.argsort(-sz):
            if i in used:
                continue
            j = int(np.argmax([W[i, k] if k not in used and k != i else -1
                               for k in range(K)]))
            order += [i, j] if j not in used else [i]
            used.update([i, j])
        nodes = sorted(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
        lab = np.empty(len(nodes), int)
        for ci, c in enumerate(cs):
            for v in c:
                lab[idx[v]] = ci
        deg = dict(G.degree()); perm, bounds = [], []
        for ci in order:
            mem = [i for i in range(len(nodes)) if lab[i] == ci]
            mem.sort(key=lambda i: -deg[nodes[i]])
            perm += mem; bounds.append(len(perm))
        E = np.array([(idx[u], idx[v]) for u, v in G.edges()])
        n = len(nodes)
        Dn = Dn[np.ix_(order, order)]; sizes = [int(sz[c]) for c in order]
        ax.imshow(adj_from(np.array(perm), E, n), cmap=BW,
                  interpolation='none', aspect='equal')
        starts = [0] + bounds[:-1]
        for b in bounds[:-1]:
            ax.axhline(b - .5, color='#0088cc', lw=1.2, alpha=.85)
            ax.axvline(b - .5, color='#0088cc', lw=1.2, alpha=.85)
        hb = bounds[1] - .5
        ax.axhline(hb, color='#ff6a00', lw=3.4); ax.axvline(hb, color='#ff6a00', lw=3.4)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                v = Dn[i, j]
                xy = ((starts[j] + bounds[j]) / 2, (starts[i] + bounds[i]) / 2)
                if v == 0:
                    ax.text(*xy, '0', ha='center', va='center', fontsize=16,
                            color='#b00020', fontweight='bold')
                else:
                    hemi = (i < 2) == (j < 2)
                    ax.text(*xy, '%.4f' % v, ha='center', va='center', fontsize=11,
                            color='#0066aa' if hemi else '#666666',
                            fontweight='bold' if hemi else 'normal',
                            bbox=dict(fc='white', ec='#0088cc' if hemi else '#bbbbbb',
                                      alpha=.85, pad=1.6, lw=.9))
        mid = [(s + b) / 2 for s, b in zip(starts, bounds)]
        ax.set_xticks(mid); ax.set_xticklabels(sizes, fontsize=12)
        ax.set_yticks(mid); ax.set_yticklabels(sizes, fontsize=12)
        ax.set_xlabel('orange = split between the halves: %d | %d'
                      % (sizes[0] + sizes[1], sizes[2] + sizes[3]), fontsize=10)
        ax.set_title('%s — %s\nbest-of-20-seed Louvain: Q = %.4f (seed %d)'
                     % (name, sub, q, seed), fontsize=12, fontweight='bold')
    fig.suptitle('Both graphs partition into 307 / 306 / 201 / 201 — halves of 508 and 507',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, .945])
    fig.savefig(os.path.join(OUT, 'AH_bilateral_bestQ.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- AB

def fig_AB():
    Q = coarse(); B = budapest()
    wC = np.load(os.path.join(G_DIR, 'cnew_coarse_w.npy'))
    wB = None
    # FIXED 2026-09-25: the second candidate was an absolute path on one
    # machine.  Both are now repo-relative.
    for c in [os.path.join(HERE, 'reference', 'bud_w.npy'),
              os.path.join(HERE, '..', 'finalgraph', 'inputs', 'bud_w.npy')]:
        if os.path.exists(c):
            wB = np.load(c)
            wB = wB[0] if wB.ndim > 1 else wB
            break
    if wB is None:
        print('AB skipped: Budapest fiber_count_mean array not found'); return

    def crescent(ax, Gr, t, c):
        d = dict(Gr.degree()); cl = nx.clustering(Gr)
        ks = [d[i] for i in Gr]; cs = [cl[i] for i in Gr]
        ax.scatter(ks, cs, s=4, alpha=.4, c=c)
        bins = list(range(0, max(ks) + 30, 30)); bk, bc = [], []
        for i in range(len(bins) - 1):
            ib = [(k, x) for k, x in zip(ks, cs) if bins[i] <= k < bins[i + 1]]
            if len(ib) >= 3:
                bk.append(np.mean([q[0] for q in ib])); bc.append(np.mean([q[1] for q in ib]))
        ax.plot(bk, bc, color='red', lw=2, alpha=.8)
        ax.set_xlabel('Degree k', fontsize=9); ax.set_ylabel('C(k)', fontsize=9)
        ax.set_title('%s\nC=%.3f' % (t, np.mean(cs)), fontsize=9, fontweight='bold')
        ax.set_xlim(0, 500); ax.set_ylim(0, 1.05)

    fig, ax = plt.subplots(3, 2, figsize=(11, 13))
    for j, (Gr, w, t, c) in enumerate([(Q, wC, 'construct.py coarse', '#c1121f'),
                                       (B, wB, 'Budapest', '#228B22')]):
        crescent(ax[0, j], Gr, t, c)
        dg = [Gr.degree(i) for i in Gr]
        ax[1, j].hist(dg, bins=40, color=c, alpha=.7, edgecolor='k', lw=.3)
        ax[1, j].axvline(np.mean(dg), color='red', lw=1, ls='--')
        ax[1, j].set_xlabel('Degree', fontsize=9); ax[1, j].set_ylabel('Count', fontsize=9)
        ax[1, j].set_title('%s degree\nmu=%.0f sd=%.1f min=%d max=%d'
                           % (t, np.mean(dg), np.std(dg), min(dg), max(dg)), fontsize=8)
        ax[2, j].hist(w, bins=np.logspace(0, np.log10(1200), 50), color=c,
                      alpha=.7, edgecolor='k', lw=.3)
        ax[2, j].set_xscale('log'); ax[2, j].set_yscale('log'); ax[2, j].set_xlim(1, 1200)
        ax[2, j].axvline(18, color='red', lw=1, ls='--')
        ax[2, j].set_xlabel('bundle weight W_ab  (red = 18)', fontsize=9)
        ax[2, j].set_ylabel('Count', fontsize=9)
        ax[2, j].set_title('%s fibre bundles\nmean=%.2f max=%d  >18: %d  gini=%.3f'
                           % (t, w.mean(), int(w.max()), (w > 18).sum(), gini(w)), fontsize=8)
    fig.suptitle('construct.py coarse vs Budapest — C(k) crescent, degree, fibre bundles',
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, .975])
    fig.savefig(os.path.join(OUT, 'AB_cnew_coarse_panels.png'), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- AE

def fig_AE():
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    rng = np.random.default_rng(0)
    pal = plt.get_cmap('tab10')
    fig, axes = plt.subplots(5, 2, figsize=(16, 38))
    for j, (name, G) in enumerate([('8k coarse quotient', coarse()),
                                   ('Budapest', budapest())]):
        q, cs, seed = best_partition(G)
        nodes = sorted(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
        lab = np.empty(len(nodes), int)
        for ci, c in enumerate(cs):
            for v in c:
                lab[idx[v]] = ci
        n = len(nodes); K = len(cs)
        colors = [pal(lab[idx[v]] % 10) for v in nodes]
        E = np.array([(idx[u], idx[v]) for u, v in G.edges()])

        ax = axes[0, j]
        pos = nx.spring_layout(G, seed=42, iterations=40, k=1.6 / np.sqrt(n))
        sub = E[rng.choice(len(E), min(6000, len(E)), replace=False)]
        ax.add_collection(LineCollection(
            np.array([[pos[nodes[a]], pos[nodes[b]]] for a, b in sub]),
            colors='0.65', linewidths=.14, alpha=.35))
        xy = np.array([pos[v] for v in nodes])
        ax.scatter(xy[:, 0], xy[:, 1], s=9, c=colors, linewidths=0)
        ax.set_title('%s — force-directed, coloured by community\n%d communities'
                     % (name, K), fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]); ax.autoscale()

        ax = axes[1, j]
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr').astype(float)
        dg = np.asarray(A.sum(1)).ravel()
        dinv = sp.diags(1 / np.sqrt(np.maximum(dg, 1)))
        vals, vecs = sla.eigsh(sp.eye(n) - dinv @ A @ dinv, k=4, sigma=-1e-3, which='LM')
        o = np.argsort(vals)
        ax.scatter(vecs[:, o[1]], vecs[:, o[2]], s=11, c=colors, linewidths=0, alpha=.85)
        ax.set_xlabel('Fiedler vector $u_2$', fontsize=9); ax.set_ylabel('$u_3$', fontsize=9)
        ax.set_title('%s — spectral embedding (normalised Laplacian)' % name,
                     fontsize=10, fontweight='bold'); ax.grid(alpha=.25)

        ax = axes[2, j]
        Dn, sz = block_density(G, cs)
        im = ax.imshow(Dn, cmap='magma_r',
                       norm=LogNorm(vmin=max(Dn[Dn > 0].min(), 1e-4), vmax=Dn.max()))
        plt.colorbar(im, ax=ax, fraction=.046, label='edge density')
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(sz, fontsize=8, rotation=45); ax.set_yticklabels(sz, fontsize=8)
        ax.set_title('%s — community block densities (labels = sizes)' % name,
                     fontsize=10, fontweight='bold')

        ax = axes[3, j]
        order = sorted(range(n), key=lambda i: (lab[i], -G.degree(nodes[i])))
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        p = np.zeros((n, 2)); p[order, 0] = np.cos(ang); p[order, 1] = np.sin(ang)
        sub = E[rng.choice(len(E), min(4000, len(E)), replace=False)]
        ax.add_collection(LineCollection(np.stack([p[sub[:, 0]], p[sub[:, 1]]], 1),
                                         colors='0.55', linewidths=.1, alpha=.3))
        ax.scatter(p[:, 0], p[:, 1], s=8, c=[pal(lab[i] % 10) for i in range(n)], linewidths=0)
        ax.set_aspect('equal'); ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('%s — circular, grouped by community' % name, fontsize=10, fontweight='bold')

        ax = axes[4, j]
        dorder = sorted(range(n), key=lambda i: -G.degree(nodes[i]))
        ax.imshow(adj_from(np.array(dorder), E, n), cmap=BW,
                  interpolation='none', aspect='equal')
        ax.set_title('%s — adjacency sorted by DEGREE (rich club at top-left)' % name,
                     fontsize=10, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('The audited 8k coarse quotient against Budapest — five display types',
                 fontsize=15, fontweight='bold', y=.998)
    fig.tight_layout(rect=[0, 0, 1, .99])
    fig.savefig(os.path.join(OUT, 'AE_8k_coarse_displays.png'), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------- AA

def fig_AA():
    from importlib.machinery import SourceFileLoader
    dq = SourceFileLoader('dq', os.path.join(HERE, 'display_quotient.py')).load_module()
    P, Q, B = parent(), coarse(), budapest()
    fig, axes = plt.subplots(1, 3, figsize=(23, 7.8))
    for ax, G, t in zip(axes, [P, Q, B], [
            '8k parent (construct.py)\n%d nodes, %s edges'
            % (P.number_of_nodes(), format(P.number_of_edges(), ',')),
            'its quotient (coarse)\n%d nodes, %s edges'
            % (Q.number_of_nodes(), format(Q.number_of_edges(), ',')),
            'Budapest Connectome\n%s' % _sub(B)]):
        order, b0, b1, b2 = dq.hierarchical_order(G)
        ax.imshow(dq.make_adj(G, order), cmap=BW, interpolation='none', aspect='equal')
        for b in b0[:-1]:
            ax.axhline(b - .5, color='gray', lw=.7, alpha=.45)
            ax.axvline(b - .5, color='gray', lw=.7, alpha=.45)
        ax.set_title(t, fontsize=12, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('construct.py 8k substrate and its quotient, against Budapest — '
                 'community-ordered adjacency', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, .955])
    fig.savefig(os.path.join(OUT, 'AA_cnew_parent_quotient_budapest.png'), dpi=115)
    plt.close(fig)


def fig_AD():
    subprocess.run([sys.executable, os.path.join(HERE, 'static_tests.py'),
                    '--graph', os.path.join(G_DIR, 'cnew_coarse.npz'),
                    '--budapest', BUD,
                    '--label', '8k coarse quotient (audited, 9/9)',
                    '--out', os.path.join(OUT, 'AD_8k_coarse_six_tests.png'),
                    '--csv', os.path.join(OUT, 'AD_8k_coarse_six_tests.csv')], check=True)


FIGS = dict(AA=fig_AA, AB=fig_AB, AD=fig_AD, AE=fig_AE, AH=fig_AH, AI=fig_AI)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(FIGS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for k, fn in FIGS.items():
        if a.only in (None, k):
            print('building %s ...' % k, flush=True)
            fn()
    print('done ->', OUT)
