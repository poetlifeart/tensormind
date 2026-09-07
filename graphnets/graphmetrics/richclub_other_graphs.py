#!/usr/bin/env python3
"""Rich-club rho(k) for the FOUR remaining paper graphs, at the SAME >=20-node floor
used by richclub_rho_pub.png, so the counts match the paper convention.

Null model is auto-selected by the audit's own loader, exactly as in the paper:
  edge-list graphs (the two 1015-node quotients) -> generic degree-preserving null
  mask graphs      (the two 16807-node graphs)   -> TRIPARTITE degree-preserving null
Significance per k: rho>1 AND one-sided p<0.05  (same as rich_club_curve.py, whose
237/212 the author already verified against the paper).

Graphs:
  preswap_quotient     threefinal/graph_v14.npz                        (1015, 70538)  old table cell 283
  reduced_localrepair  modular_run/repro_local_repair_quotient.npz     (1015, 67094)  old 195/320
  global_parent        experimentbrain/graph_brain_mild.npz            (16807, 337983) old 295 / run 290
  feeder_lifted        experimentbrain/graph_brain_mild_v146_feeder.npz(16807, 344483) old run 289

Reports full-range AND >=20-floor: n_sig, longest run, kmax. Saves curves incrementally.
1000 nulls. New file, nothing overwritten. Written 2026-08-14."""
import sys, os, json, time
import numpy as np, networkx as nx
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # connectome_audit_gold.py is a sibling
# original (pre-tensormind): '/home/vahid/Desktop/emergentgraph/graph_topology'
from connectome_audit_gold import load_layered_graph, _generate_null, _use_generic_null

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'results', 'richclub_other_graphs.json')
N_NULL   = 1000
SEED     = 42
NPROC    = 8
MINNODES = 20          # same publication floor as richclub_rho_pub.png

# Paths resolved relative to THIS file: graphnets/graphmetrics/ -> graphnets/
_G = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_CG = os.path.join(_G, 'construction', 'global')
# Reference: the six graphs the paper reports. No longer used by main() --
# pass --graph explicitly. Kept so the paper's set is discoverable.
PAPER_GRAPHS = [
    ('preswap_quotient',    os.path.join(_CG, 'quotient', 'graph_v14.npz')),
    ('postswap_quotient',   os.path.join(_CG, 'quotient', 'graph_v146_6500.npz')),
    ('reduced_localrepair', os.path.join(_G, 'construction', 'local', 'localrepair_quotient.npz')),
    ('global_parent',       os.path.join(_CG, 'graph_brain_mild.npz')),
    ('feeder_lifted',       os.path.join(_CG, 'quotient', 'lift', 'graph_brain_mild_v146_feeder.npz')),
    ('budapest',            os.path.join(_G, 'Budapest', 'budapest_connectome.gml')),
]
# originals (pre-tensormind):
#   ~/Desktop/threefinal/graph_v14.npz
#   ~/Desktop/threefinal/graph_v146_6500.npz
#   ~/Desktop/emergencesimply/modular_run/repro_local_repair_quotient.npz   <- same graph as
#       construction/local/localrepair_quotient.npz (edge-set fingerprint verified identical)
#   ~/experimentbrain/graph_brain_mild.npz
#   ~/experimentbrain/graph_brain_mild_v146_feeder.npz
#   ~/Desktop/finalthree/budapest_connectome.gml

def _null_phi(args):
    layered, seed = args
    R = _generate_null(layered, seed)
    return nx.rich_club_coefficient(R, normalized=False)

def longest_run(mask):
    best = run = 0
    for x in mask:
        run = run + 1 if x else 0; best = max(best, run)
    return best

def compute(layered, label):
    t0 = time.time()
    G = layered.G
    null_kind = 'generic_degree_preserving' if _use_generic_null(layered) else 'tripartite_degree_preserving'
    phi = nx.rich_club_coefficient(G, normalized=False)
    with Pool(NPROC) as pool:
        nulls = pool.map(_null_phi, [(layered, SEED + 5000 + i) for i in range(N_NULL)], chunksize=2)
    ks = np.array(sorted(phi))
    real = np.array([phi[k] for k in ks], float)
    null = np.array([[c.get(int(k), np.nan) for k in ks] for c in nulls], float)
    mean = np.nanmean(null, 0); sd = np.nanstd(null, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = real / mean
    pval = (np.nansum(null >= real, 0) + 1) / (np.sum(np.isfinite(null), 0) + 1)
    sig = (rho > 1) & (pval < 0.05)

    # >=20-node floor: keep thresholds where at least MINNODES nodes have degree > k
    deg = np.array([dd for _, dd in G.degree()])
    n_above = np.array([(deg > k).sum() for k in ks])
    keep = n_above >= MINNODES
    kmax = int(ks[keep].max()) if keep.any() else 0
    sig_floor = sig & keep

    res = dict(
        n=G.number_of_nodes(), m=G.number_of_edges(), n_null=N_NULL, null_model=null_kind,
        ks=[int(k) for k in ks], rho=rho.tolist(),
        null_mean=mean.tolist(), null_std=sd.tolist(),
        sig=[bool(x) for x in sig], degrees=[int(x) for x in deg],
        n_sig_full=int(sig.sum()), run_full=longest_run(sig),
        n_sig_floor=int(sig_floor.sum()), run_floor=longest_run(sig_floor),
        floor_kmax=kmax, minnodes=MINNODES)
    print(f"[{label}] n={res['n']} m={res['m']} null={null_kind}\n"
          f"    FULL      : {res['n_sig_full']} sig, longest run {res['run_full']}\n"
          f"    FLOOR>=20 : {res['n_sig_floor']} sig, longest run {res['run_floor']}  (k<={kmax})\n"
          f"    {time.time()-t0:.0f}s", flush=True)
    return res

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Rich-club curve of one graph vs degree-preserving nulls, "
                    "with the >=20-node floor.")
    ap.add_argument('--graph', required=True, metavar='PATH',
                    help='Graph to analyse (.npz or .gml)')
    ap.add_argument('--json', required=True, metavar='PATH',
                    help='Output JSON for this graph')
    ap.add_argument('--n-null', type=int, default=N_NULL,
                    help=f'Number of degree-preserving nulls (default: {N_NULL})')
    ap.add_argument('--minnodes', type=int, default=MINNODES,
                    help=f'Floor: only count k with >= this many nodes above k '
                         f'(default: {MINNODES}). The paper reports floored values.')
    args = ap.parse_args()

    N_NULL   = args.n_null
    MINNODES = args.minnodes
    name     = os.path.splitext(os.path.basename(args.graph))[0]

    layered = load_layered_graph(args.graph)
    result  = compute(layered, name)

    d = os.path.dirname(os.path.abspath(args.json))
    if d:
        os.makedirs(d, exist_ok=True)
    json.dump(result, open(args.json, 'w'), indent=2)
    print(f"  -> {args.json}", flush=True)
