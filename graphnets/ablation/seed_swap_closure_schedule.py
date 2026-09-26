#!/usr/bin/env python3
"""
seed_swap_closure_schedule.py -- Was seed_swap's alpha=[0,0,0,0,3.5] a handicap,
or is it forced by the 7-vertex seed?

THE QUESTION.  seed_swap.log records the 7-vertex arm run with closure ONLY at the
final step: 944,784 of its 1,003,608 parent edges are pure tensor product and only
58,824 (5.9%) are closure.  The deposited 20-vertex construction closes at EVERY
step, and this session measured that in-loop closure is what makes the block
structure findable at all (contrast 8.3x -> 34.3x, blind recovery ARI 0.17 ->
1.00).  So the 7-vertex result (Q 0.2047 against the reference's 0.5574) might be
a property of the schedule rather than of the seed -- which matters, because that
number is cited as evidence that the 20-vertex seed is special.

WHAT seed_swap.py ITSELF SAYS, and it is fair.  Its docstring discloses two
differences it cannot avoid: the 7-vertex seed carries NO BLOCK LABELS, so global
coarsening is forced rather than chosen; and five tensor steps are needed, so
alpha has five entries, "chosen to land near the 8k parent's mean degree (119.4)".
The run hits 119.43.  So alpha was set by a MATCHING CRITERION, not carelessly.

THE TEST.  Sweep the closure schedule and ask whether ANY schedule with early
closure can match the 8k parent's mean degree of 119.43.  The answer decides the
reading:

  * if early closure overshoots the degree budget no matter how small, then
    alpha=[0,0,0,0,3.5] is FORCED, the shipped result is the best available test
    of the seed, and my criticism of it is wrong;
  * if some schedule closes early and still matches degree, that schedule is the
    fair comparison and the shipped number should be replaced by it.

WHY IT IS PLAUSIBLY FORCED.  The pure product alone reaches 944,784 edges on
16,807 vertices -- mean degree 112.4 against the target 119.4 -- so only ~59,000
edges of headroom exist at step 5.  And closure added at step k is tensor-
multiplied by every later step, so an edge closed early costs far more than one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.normpath(os.path.join(HERE, '..', 'construction', 'bilateral'))
_s = importlib.util.spec_from_file_location('ss', os.path.join(HERE, 'seed_swap.py'))
ss = importlib.util.module_from_spec(_s); sys.modules['ss'] = ss; _s.loader.exec_module(ss)
cc = ss.cc

TARGET_K = 119.43          # the 8k parent's mean degree, seed_swap's own criterion
N7 = 7 ** 5                # 16,807


def run(alphas, verbose=False):
    rng = np.random.default_rng(99)
    out = ss.grow7(list(alphas), 1.5, rng, n_steps=5, verbose=verbose)
    u, v = out[0], out[1]
    return len(u), 2 * len(u) / N7


def main():
    print('=' * 78)
    print('IS seed_swap\'s CLOSURE SCHEDULE FORCED BY THE 7-VERTEX SEED?')
    print('=' * 78)
    print(f'\ntarget mean degree = {TARGET_K} (the 8k parent; seed_swap\'s own '
          f'matching criterion)')

    print('\npure product, no closure anywhere:')
    m0, k0 = run([0, 0, 0, 0, 0])
    print(f'    m={m0:>10,}  <k>={k0:>7.2f}   headroom to target: '
          f'{int((TARGET_K - k0) * N7 / 2):+,} edges')

    schedules = [
        ('SHIPPED      [0,0,0,0,3.5]',      [0, 0, 0, 0, 3.5]),
        ('script deflt [3,4,4,4,3.5]',      [3, 4, 4, 4, 3.5]),
        ('step5 only   [0,0,0,0,7.0]',      [0, 0, 0, 0, 7.0]),
        ('last two     [0,0,0,1,3.5]',      [0, 0, 0, 1, 3.5]),
        ('last two sml [0,0,0,0.1,3.5]',    [0, 0, 0, 0.1, 3.5]),
        ('step3 on     [0,0,0.1,0,3.5]',    [0, 0, 0.1, 0, 3.5]),
        ('step2 on     [0,0.1,0,0,3.5]',    [0, 0.1, 0, 0, 3.5]),
        ('step1 on     [0.5,0,0,0,3.5]',    [0.5, 0, 0, 0, 3.5]),
        ('step1 tiny   [0.15,0,0,0,3.5]',   [0.15, 0, 0, 0, 3.5]),
        ('every step   [0.1,0.1,0.1,0.1,3.5]', [0.1, 0.1, 0.1, 0.1, 3.5]),
    ]
    rows = []
    print(f'\n{"schedule":<32}{"parent m":>11}{"<k>":>9}{"vs target":>11}'
          f'{"closure":>11}{"feasible":>10}')
    for name, a in schedules:
        m, k = run(a)
        d = 100 * (k - TARGET_K) / TARGET_K
        closure = m - m0
        feas = 'yes' if abs(d) <= 10 else 'NO'
        rows.append(dict(name=name, alphas=list(map(float, a)), m=m, mean_deg=k,
                         pct_vs_target=d, closure_edges=closure,
                         feasible=abs(d) <= 10))
        print(f'  {name:<30}{m:>11,}{k:>9.2f}{d:>+10.1f}%{closure:>11,}{feas:>10}')

    # how much does one closure edge at step k cost by step 5?
    print('\nCOST OF CLOSING EARLY -- extra parent edges per unit of alpha at step k')
    print('  (alpha*n edges requested at step k; measured as the parent-size delta)')
    base = m0
    for k_step in range(1, 6):
        a = [0.0] * 5
        a[k_step - 1] = 1.0
        m, _ = run(a)
        n_at_step = 7 ** k_step
        requested = int(1.0 * n_at_step)
        print(f'    step {k_step}: requested {requested:>6,} closure edges at '
              f'n={n_at_step:>6,}  ->  parent grew by {m - base:>10,}  '
              f'(x{(m - base) / max(requested,1):>8,.0f} amplification)')

    json.dump(dict(target_k=TARGET_K, pure_product=dict(m=m0, mean_deg=k0),
                   schedules=rows),
              open(os.path.join(HERE, 'seed_swap_closure_schedule.json'), 'w'),
              indent=2)

    print('\n' + '=' * 78)
    print('VERDICT')
    print('=' * 78)
    feas = [r for r in rows if r['feasible']]
    early = [r for r in feas if any(x > 0 for x in r['alphas'][:4])]
    print(f'  schedules matching mean degree within 10%: {len(feas)} of {len(rows)}')
    for r in feas:
        print(f'      {r["name"]}  <k>={r["mean_deg"]:.2f} ({r["pct_vs_target"]:+.1f}%)')
    if early:
        print('\n  AT LEAST ONE FEASIBLE SCHEDULE CLOSES EARLY:')
        for r in early:
            print(f'      {r["name"]}')
        print('  => the shipped alpha was NOT forced; the fair comparison exists'
              '\n     and the 7-vertex arm should be re-run with it.')
    else:
        print('\n  NO FEASIBLE SCHEDULE CLOSES BEFORE THE FINAL STEP.')
        print('  => alpha=[0,0,0,0,3.5] IS FORCED by the 7-vertex seed at five'
              '\n     steps: the pure product already consumes the degree budget,'
              '\n     and early closure is amplified by every later step.  The'
              '\n     shipped result is the best available test of this seed, and'
              '\n     the criticism that it was handicapped does not hold.')


if __name__ == '__main__':
    main()
