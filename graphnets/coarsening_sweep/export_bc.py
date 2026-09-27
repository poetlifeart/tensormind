#!/usr/bin/env python3
"""
export_bc.py -- Write graphs B and C to .npz so the nine-criterion audit can run
on them.

  B  pure Louvain per block, one gamma in all four blocks swept to the total
     nearest 1,015, no merge step.  Borrows ONE number from the reference.
  C  Louvain@30 per block then merge_densest to targets from the PARENT's own
     block sizes (2400/2400/1600/1600 -> 305/304/203/203).  Borrows ONE number.

A (the deposited quotient) is already audited: ablation/audit_quot_seed99.json,
1,015 / 64,760, 9/9.  So only B and C are needed.

Format matches construction/bilateral/graphs/cnew_coarse.npz -- 'edges' (m,2) and
'n_total' -- which is what connectome_audit_gold.py reads.
"""
import importlib.util, os, sys
import numpy as np, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    'cvc', os.path.join(HERE, 'chromatic_vs_coarsening.py'))
_m = importlib.util.module_from_spec(_s); sys.modules['cvc'] = _m
_s.loader.exec_module(_m)
cc = _m.cc

def targets_from_parent(block, total):
    sizes = np.bincount(block).astype(float)
    exact = sizes / sizes.sum() * total
    base = np.floor(exact).astype(int)
    short = total - base.sum()
    if short:
        order = np.lexsort((np.arange(len(base)), -(exact - base)))
        base[order[:short]] += 1
    return base.tolist()

u, v, colour, block = _m.grow(cc.build_seed(), [3, 4, 30], 1.5, 3,
                             np.random.default_rng(99), True)
parent = nx.Graph(); parent.add_nodes_from(range(8000))
parent.add_edges_from(zip(u.tolist(), v.tolist()))
print(f'parent n=8,000 m={len(u):,}', flush=True)

out = os.path.join(HERE, 'graphs'); os.makedirs(out, exist_ok=True)

print('B: sweeping resolution per block to the total nearest 1,015 ...', flush=True)
gamma, labB = _m.sweep(u, v, block, 18, lambda s: None)
GB, _ = _m.quotient(u, v, labB)
print(f'   gamma {gamma:.6g}  n={GB.number_of_nodes():,} m={GB.number_of_edges():,}')

own = targets_from_parent(block, 1015)
print(f'C: merge_densest to the parent\'s own proportions {own} ...', flush=True)
labC = cc.coarsen(parent, block, own, seed=42)
GC, _ = _m.quotient(u, v, labC)
print(f'   n={GC.number_of_nodes():,} m={GC.number_of_edges():,}')

for tag, G in (('B_pure_louvain', GB), ('C_merge_own_proportions', GC)):
    E = np.array(sorted((min(a, b), max(a, b)) for a, b in G.edges()), np.int64)
    p = os.path.join(out, f'quot_{tag}.npz')
    np.savez_compressed(p, edges=E, n_total=G.number_of_nodes())
    assert nx.is_connected(G), f'{tag} is not connected'
    print(f'wrote {p}  n={G.number_of_nodes():,} m={len(E):,} connected=True')
