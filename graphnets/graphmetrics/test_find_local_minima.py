#!/usr/bin/env python3
"""
test_find_local_minima.py -- regression test for the propermulti plateau fix.

WHY THIS EXISTS.  connectome_audit_gold._find_local_minima used a strict
two-sided test, which misses a FLAT-BOTTOMED valley -- a run of equal mean-VI
values lying below both neighbours.  Mean VI is exactly 0.0 when the partition is
perfectly reproducible across all repeats, i.e. MAXIMAL stability, and several
consecutive resolutions can share it.  The strict test scored that as zero stable
scales, so on this criterion it could only ever produce FALSE NEGATIVES.

The criterion is SCORED (propermulti, one of the nine), so the fix must be shown
not to change any verdict that did not involve a plateau.  This test does that by
replaying both implementations over the stored mean-VI curve of every audit JSON
in the repository and in the paper's data directory, and checking the outcome
against the pass/fail each audit actually recorded.

    python3 test_find_local_minima.py

Two guarantees are asserted:
  1. the OLD implementation reproduces every stored pass/fail exactly -- without
     this the replay would prove nothing;
  2. the NEW implementation changes no verdict except where a plateau is present.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))

_s = importlib.util.spec_from_file_location(
    'audit', os.path.join(HERE, 'connectome_audit_gold.py'))
audit = importlib.util.module_from_spec(_s)
sys.modules['audit'] = audit
_s.loader.exec_module(audit)
new_minima = audit._find_local_minima
TOL = audit._VI_PLATEAU_TOL


def old_minima(xs, ys):
    """The implementation before the fix, kept here as the comparison baseline."""
    return [(xs[i], ys[i]) for i in range(1, len(ys) - 1)
            if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]]


def unit_cases():
    """Behaviour the fix must have, independent of any stored data."""
    cases = [
        ('single dip reduces to the strict test',
         [1, 2, 3], [0.5, 0.1, 0.4], [(2, 0.1)]),
        ('flat-bottomed valley is one minimum at its midpoint',
         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
         [0, 0, 0.1324, 0.0902, 0.0671, 0.0, 0.0, 0.0, 0.0, 1.4906, 2.4847],
         [(1.2, 0.0)]),
        ('a plateau at the end is not a minimum',
         [1, 2, 3, 4], [0.5, 0.2, 0.0, 0.0], []),
        ('a plateau at the start is not a minimum',
         [1, 2, 3, 4], [0.0, 0.0, 0.2, 0.5], []),
        ('a monotone curve has no minimum',
         [1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], []),
        ('a flat curve throughout has no minimum',
         [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], []),
    ]
    print('UNIT CASES')
    for name, xs, ys, want in cases:
        got = new_minima(xs, ys)
        ok = got == want
        print(f'  {"ok " if ok else "FAIL"} {name}')
        assert ok, f'{name}: got {got}, want {want}'
    print()


def replay():
    paths = sorted(set(
        glob.glob(os.path.join(REPO, 'graphnets', '**', 'audit*.json'),
                  recursive=True)
        + glob.glob(os.path.expanduser('~/Downloads/audit_8k_*.json'))
        + glob.glob(os.path.expanduser(
            '~/Downloads/paper_data/comparison/audit_*.json'))))
    print(f'REPLAY over {len(paths)} audit files')
    print(f'  {"audit":<34}{"old":>5}{"new":>5}{"old pass":>10}'
          f'{"new pass":>10}   verdict')
    flips, checked = [], 0
    for p in paths:
        d = json.load(open(p))
        pm = d.get('propermulti')
        if not isinstance(pm, dict) or 'mean_vi' not in pm:
            continue
        xs, ys, Qpg = pm['gammas'], pm['mean_vi'], pm['Q_per_gamma']

        def stable(mins):
            return sum(1 for g, _ in mins if Qpg[str(g)]['mean'] > 0.3)

        o, n = old_minima(xs, ys), new_minima(xs, ys)
        po, pn = stable(o) >= 1, stable(n) >= 1

        # GUARANTEE 1: the old implementation must reproduce what was recorded.
        assert po == pm['pass'], (
            f'{p}: replay of the OLD test gives pass={po} but the file records '
            f"pass={pm['pass']} -- the replay is not faithful, so this test "
            'proves nothing')
        checked += 1

        plateau = any(abs(ys[i + 1] - ys[i]) <= TOL for i in range(len(ys) - 1))
        verdict = ('unchanged' if po == pn else
                   ('fail -> PASS' if pn else 'pass -> FAIL'))
        if po != pn:
            flips.append((os.path.basename(p), verdict, plateau))
            # GUARANTEE 2: a verdict may only change where a plateau exists.
            assert plateau, (
                f'{p}: verdict changed with no plateau in the VI curve -- the '
                'fix is doing more than it should')
            assert pn, (
                f'{p}: the fix turned a PASS into a FAIL, which it must never '
                'do -- it can only repair false negatives')
        print(f'  {os.path.basename(p):<34}{len(o):>5}{len(n):>5}'
              f'{str(po):>10}{str(pn):>10}   {verdict}')

    print(f'\n  {checked} audits replayed; OLD implementation reproduced every '
          'recorded pass/fail')
    if flips:
        print('  verdicts changed by the fix:')
        for name, v, pl in flips:
            print(f'    {name}: {v}  (plateau present: {pl})')
    else:
        print('  no verdict changed')
    return checked, flips


if __name__ == '__main__':
    unit_cases()
    checked, flips = replay()
    assert checked >= 6, f'only {checked} audits replayed; expected at least 6'
    print('\nPASS')
