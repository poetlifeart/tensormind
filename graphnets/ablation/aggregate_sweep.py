"""Five-seed means for the rerun ablation sweep."""
import json, sys, numpy as np
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1
                      else 'results_chromatic_ablation_shuffled.json'))
FS = sorted({r['f'] for r in rows})
def mean(g, k): return float(np.mean([r[k] for r in g]))
print(f"{len(rows)} runs, {len(FS)} values of f\n")
print("MAIN TABLE  (five-seed means)")
print(f"{'f':>5s} {'mono':>9s} {'parent m':>9s} {'quot m':>8s} {'full':>7s} {'shape':>15s}")
for f in FS:
    g = [r for r in rows if r['f'] == f]
    sd = np.std([r['shape_dist'] for r in g], ddof=1)
    print(f"{f:5.2f} {mean(g,'mono_parent'):9,.0f} {mean(g,'parent_m'):9,.0f} "
          f"{mean(g,'m'):8,.0f} {mean(g,'dist'):7.3f} "
          f"{mean(g,'shape_dist'):7.3f}+-{sd:.3f}   (n={len(g)})")
print("\nSHAPE VALUES QUOTED IN THE TEXT  (five-seed means)")
print(f"{'f':>5s} {'degree s.d.':>12s} {'transitivity':>13s} {'APL':>8s} {'clust':>8s} {'Q':>8s}")
for f in FS:
    g = [r for r in rows if r['f'] == f]
    print(f"{f:5.2f} {mean(g,'degsd'):12.2f} {mean(g,'trans'):13.4f} "
          f"{mean(g,'apl'):8.4f} {mean(g,'clust'):8.4f} {mean(g,'Q'):8.4f}")
print("\nSEED-STAGE CLOSURES PER RUN  (for the caption)")
print(f"{'f':>5s}  " + "  ".join(f"seed {r['seed']:<3d}" for r in rows if r['f'] == FS[0]))
for f in FS:
    g = sorted([r for r in rows if r['f'] == f], key=lambda r: [99,1,2,3,4].index(r['seed']))
    cells = "  ".join(f"{r.get('stage_closures',['?'])[0]:>8}" for r in g)
    same = len({r.get('stage_closures',[None])[0] for r in g}) == 1
    print(f"{f:5.2f}  {cells}   {'all equal' if same else 'VARIES'}")
print("\nPARENT m PER RUN  (is it deterministic at high f?)")
for f in FS:
    g = [r for r in rows if r['f'] == f]
    vals = sorted({r['parent_m'] for r in g})
    print(f"  f={f:4.2f}: {', '.join(f'{v:,}' for v in vals)}"
          + ("   <- identical in every run" if len(vals) == 1 else ""))
