#!/usr/bin/env python3
"""Aggregate the stage-specific ablation: per-corner means, and the three key
comparisons PAIRED BY RNG SEED.

    python3 aggregate_stage.py [results_stage_ablation.json]

WHY PAIRED, AND NOT JUST CORNER MEANS.  The matched-parent-size property does not
follow from sharing f1 alone.  At f1 = 0 the ACHIEVED seed-stage closure count
differs between RNG seeds -- seed 99's retry loop stops at five closures while
seeds 1-4 reach six -- and each closure is worth 62**2 = 3,844 parent edges, so
parent_m varies by that much across the ensemble at fixed f.  Measured on the
published scalar sweep: f = 0 alone spans three distinct parent_m values
(477,584 / 480,932 / 481,428).

What IS exact is the pair: same RNG seed and same f1 gives the same seed-stage
draw sequence, hence the same k1.  So every comparison below is computed within
seed and then averaged over seeds, and parent_m equality is CHECKED for each pair
rather than assumed.  Pairs whose parent sizes differ are reported separately
instead of being silently averaged in -- scale-2 and scale-3 quotas do not always
fill (one run in fifty of the published sweep fell eight closures short at
scale 2, which is 496 parent edges).

Columns: relations = quotient edges = distinct supernode pairs carrying at least
one parent edge.  shape = mean absolute relative deviation from the reference over
the six non-density measures.  full = the same over all nine, which counts one
density effect three times at fixed n and is reported only for continuity.
"""
import json
import sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else 'results_stage_ablation.json'
rows = json.load(open(PATH))


def corner(r):
    return ''.join(str(int(v)) for v in r['f_stages'])


by = defaultdict(dict)                     # corner -> seed -> row
for r in rows:
    by[corner(r)][r['seed']] = r

order = ['000', '001', '010', '011', '100', '101', '110', '111']
present = [c for c in order if c in by] + [c for c in sorted(by) if c not in order]

print(f"{len(rows)} runs, {len(by)} corners\n")

print("PER-CORNER MEANS")
print(f"  {'f1f2f3':>7} {'n':>2} {'mono':>9} {'parent m':>10} {'relations':>10} "
      f"{'shape':>15} {'full':>7} {'Q':>7} {'apl':>7}")
for c in present:
    g = list(by[c].values())
    m = lambda k: float(np.mean([x[k] for x in g]))
    sd = np.std([x['shape_dist'] for x in g], ddof=1) if len(g) > 1 else 0.0
    pm = sorted({x['parent_m'] for x in g})
    pm_s = f"{pm[0]:,}" if len(pm) == 1 else f"{pm[0]:,}..{pm[-1]:,}"
    print(f"  {c:>7} {len(g):>2} {m('mono_parent'):>9,.0f} {pm_s:>10} "
          f"{m('m'):>10,.0f} {m('shape_dist'):>8.4f}+-{sd:.4f} "
          f"{m('dist'):>7.4f} {m('Q'):>7.4f} {m('apl'):>7.4f}")

COMPARISONS = [
    ('011', '000', 'A  PLACEMENT at matched parent size  (the mechanism claim)'),
    ('100', '000', 'B  SCAFFOLD  (seed-stage pool widened, later scales intact)'),
    ('111', '100', 'C  PLACEMENT again, at the larger matched size  (replication)'),
    ('001', '000', 'D  scale 3 only'),
    ('010', '000', 'E  scale 2 only'),
    ('111', '000', 'F  all stages  (= the published scalar f=1 vs f=0)'),
]

print("\nPAIRED COMPARISONS  (within RNG seed, then averaged)")
for a, b, label in COMPARISONS:
    if a not in by or b not in by:
        continue
    seeds = sorted(set(by[a]) & set(by[b]))
    if not seeds:
        continue
    matched, unmatched = [], []
    for s in seeds:
        ra, rb = by[a][s], by[b][s]
        (matched if ra['parent_m'] == rb['parent_m'] else unmatched).append((s, ra, rb))
    print(f"\n{label}")
    print(f"  {a} vs {b}   seeds {seeds}")
    if unmatched:
        print(f"  [!] {len(unmatched)} pair(s) NOT matched on parent_m -- excluded "
              f"from the means below:")
        for s, ra, rb in unmatched:
            print(f"      seed {s}: parent {rb['parent_m']:,} -> {ra['parent_m']:,} "
                  f"(stages {rb['stage_closures']} -> {ra['stage_closures']})")
    if not matched:
        print("  no parent-matched pairs; nothing to report")
        continue
    d_rel = [ra['m'] - rb['m'] for _, ra, rb in matched]
    p_rel = [100.0 * (ra['m'] - rb['m']) / rb['m'] for _, ra, rb in matched]
    d_shape = [ra['shape_dist'] - rb['shape_dist'] for _, ra, rb in matched]
    d_mono = [ra['mono_parent'] - rb['mono_parent'] for _, ra, rb in matched]
    pm = {rb['parent_m'] for _, ra, rb in matched} | {ra['parent_m'] for _, ra, rb in matched}
    print(f"  parent-matched pairs: {len(matched)}   parent m held at "
          f"{', '.join(f'{x:,}' for x in sorted(pm))}")
    print(f"  relations      {np.mean([rb['m'] for _,ra,rb in matched]):,.0f} -> "
          f"{np.mean([ra['m'] for _,ra,rb in matched]):,.0f}"
          f"   change {np.mean(d_rel):+,.0f} ({np.mean(p_rel):+.2f}%)")
    print(f"  shape distance {np.mean([rb['shape_dist'] for _,ra,rb in matched]):.4f} -> "
          f"{np.mean([ra['shape_dist'] for _,ra,rb in matched]):.4f}"
          f"   change {np.mean(d_shape):+.4f}")
    print(f"  monochromatic  {np.mean([rb['mono_parent'] for _,ra,rb in matched]):,.0f} -> "
          f"{np.mean([ra['mono_parent'] for _,ra,rb in matched]):,.0f}"
          f"   ({np.mean(d_mono):+,.0f})")
    if len(matched) > 1:
        sd = np.std(d_rel, ddof=1)
        print(f"  per-seed change in relations: "
              f"{[f'{x:+,}' for x in d_rel]}   sd {sd:,.0f}"
              + (f"   |mean|/sd = {abs(np.mean(d_rel))/sd:.1f}" if sd > 0 else ""))

print("\nREAD IN THIS ORDER: A, then B, then C.  A is the comparison the paper's")
print("mechanism claim rests on: it relaxes chromatic legality ONLY at scales 2")
print("and 3, with the seed stage and the parent edge count held fixed.")
