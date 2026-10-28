#!/usr/bin/env python3
"""Build the 8k quotient under alternative RNG seeds, for a seed-sensitivity
audit.  Seed 99 is the deposited instance (construct.py's default).
Grows with construct.py's own grow(), so the five outputs are the audited
graphs (1015 / 64,760, 63,413, 62,901, 63,628, 62,584); seed 99 is
cnew_coarse.npz edge for edge.  chromatic_ablation.py's grow() is NOT used:
it shuffles candidates and so draws a different random stream."""
import sys, os, numpy as np, networkx as nx, importlib.util
V=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chromatic_ablation.py')
exec(open(V).read().split("if __name__")[0])          # gives cc (construct.py), TARGETS
for sd in [99,1,2,3,4]:
    rng=np.random.default_rng(sd)
    u,v,colour,block=cc.grow(cc.build_seed(),[3,4,30],1.5,3,rng,verbose=False)   # construct.py's grow()
    P=nx.Graph(); P.add_nodes_from(range(len(block))); P.add_edges_from(zip(u.tolist(),v.tolist()))
    label=cc.coarsen(P,block,TARGETS,seed=42)
    coarse,super_block,pairs,witness=cc.build_coarse(u,v,label,block,sum(TARGETS))
    E=np.array(sorted((min(a,b),max(a,b)) for a,b in coarse.edges()),dtype=np.int64)
    out=f'quot_seed{sd}.npz'
    np.savez_compressed(out, edges=E, n_total=coarse.number_of_nodes())
    mono=int((colour[u]==colour[v]).sum())
    print(f"seed {sd:3d}  parent {P.number_of_edges():,}  mono {mono}  "
          f"quotient {coarse.number_of_nodes()}/{coarse.number_of_edges():,}  -> {out}", flush=True)
