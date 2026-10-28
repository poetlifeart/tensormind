"""Threshold the USF undirected word-association graph at f* = 0.67 (Palla et al.
2005: f* = fraction of links stronger than w*), save edge-list .npz.
Usage: python -I usf_threshold.py DIR"""
import sys, os, pickle
import numpy as np
d = sys.argv[1]
und = pickle.load(open(os.path.join(d, "usf_undirected_weights.pkl"), "rb"))
L = len(und)
ws = np.array(sorted(und.values(), reverse=True))
# candidate thresholds: distinct weights; f(w*) = #(w > w*)/L
uniq = np.unique(ws)
best = None
for t in uniq:
    f = float((ws > t).sum()) / L
    if best is None or abs(f - 0.67) < abs(best[1] - 0.67):
        best = (float(t), f)
wstar, fstar = best
print(f"links {L}; chosen w* = {wstar:.4f}: f* = {fstar:.5f}  (neighbours:",
      [(round(float(t),4), round(float((ws > t).sum())/L,5)) for t in uniq[max(0,np.searchsorted(uniq,wstar)-2):np.searchsorted(uniq,wstar)+3]], ")")
kept = [(a, b) for (a, b), x in und.items() if x > wstar]
nodes = sorted({v for e in kept for v in e})
idx = {v: i for i, v in enumerate(nodes)}
E = np.array(sorted((min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in kept), dtype=np.int64)
np.savez(os.path.join(d, "usf_fstar067.npz"), edges=E)
print(f"kept links {len(E)}, nodes {len(nodes)} -> usf_fstar067.npz")
