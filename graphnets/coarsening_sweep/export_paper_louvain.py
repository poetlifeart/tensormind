#!/usr/bin/env python3
"""
export_paper_louvain.py -- Write the PAPER'S Louvain-only quotient to .npz for the
nine-criterion audit.

This is NOT graph B.  B shares one gamma across all four blocks and is swept to
the total nearest 1,015; it audited 8/9, failing connector_hubs with zero of them.
The paper's Louvain-only coarsening (construction/bilateral/blockwise/
blockwise_pipeline.py) sweeps a SEPARATE gamma inside each block against that
block's own-proportion target, and lands at ~1,076 supernodes rather than 1,015.
Different gamma per block can give a different spread of community sizes, and
community-size spread is exactly what B lacks -- so B's failure does not carry
over automatically.

Format matches cnew_coarse.npz ('edges', 'n_total'), which is what
connectome_audit_gold.py reads.
"""
import importlib.util, os, sys
import numpy as np, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
BIL = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))

def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m)
    return m

_m = _load('cvc', os.path.join(HERE, 'chromatic_vs_coarsening.py'))
bw = _load('bw', os.path.join(BIL, 'blockwise', 'blockwise_pipeline.py'))
cc = _m.cc

u, v, colour, block = _m.grow(cc.build_seed(), [3, 4, 30], 1.5, 3,
                              np.random.default_rng(99), True)
parent = nx.Graph(); parent.add_nodes_from(range(8000))
parent.add_edges_from(zip(u.tolist(), v.tolist()))
print(f'parent n=8,000 m={len(u):,}', flush=True)

RES = [2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 55]   # blockwise_pipeline's own ladder
lab, nsup = bw.coarsen_blockwise(parent, block, 1015, RES, seed=42, verbose=True)
G = bw.build_quotient(u, v, lab, nsup)[0]
print(f'quotient n={G.number_of_nodes():,} m={G.number_of_edges():,} '
      f'connected={nx.is_connected(G)}')

out = os.path.join(HERE, 'graphs', 'quot_P_paper_louvain_perblock.npz')
os.makedirs(os.path.dirname(out), exist_ok=True)
E = np.array(sorted((min(a, b), max(a, b)) for a, b in G.edges()), np.int64)
np.savez_compressed(out, edges=E, n_total=G.number_of_nodes())
print(f'wrote {out}')
