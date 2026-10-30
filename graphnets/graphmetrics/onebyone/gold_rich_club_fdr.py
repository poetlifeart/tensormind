#!/usr/bin/env python3
"""Benjamini-Hochberg FDR analysis of rich-club thresholds.

This is a standalone companion to gold_rich_club.py. It reuses that script's
loading, null-graph generation, rich-club curve, and empirical per-k p-values,
without changing its original three-consecutive-k verdict.

Family and correction
    Every k in gold_rich_club.py's per_k table is tested (k present in the
    real curve and in every null curve, null mean > 0); nothing is filtered
    on unadjusted significance or on degree before the correction.
    BH-adjusted p-values (Benjamini & Hochberg 1995) are computed over that
    whole family; a k is FDR-significant if adjusted p < q and rho > 1.

Outputs (in result["fdr_bh"])
    fdr_significant_k       every k surviving the correction
    fdr_significant_ranges  consecutive runs of those k (same run definition
                            as gold_rich_club.py: integer k, k_next = k + 1),
                            each with its node counts and the number of
                            distinct node sets it contains.  The club at k is
                            the nodes of degree > k, so two thresholds select
                            the same node set exactly when n_above_k is equal;
                            a long run of k over a gap in the degree sequence
                            is not a long run of distinct clubs.
    riedel2022_regime       the rich-club regime by the rule of Riedel, van den
                            Heuvel & Markett 2022 (Hum. Brain Mapp. 43:4239,
                            doi:10.1002/hbm.25950), or None.
    regime_verdict          "valid_regime" (length > 0), "single_threshold_regime"
                            (the rule returned one k, length 0: a significant
                            threshold, not a continuous range; the paper's
                            text does not settle this case) or
                            "no_valid_regime".
    p_resolution etc.       the smallest attainable p, 1/(n_curves + 1); the
                            BH cutoff at rank 1, q/n_tests; how many k sit at
                            that smallest p; and how many such k BH needs
                            before it can reject any (ceil(n_tests * p_min / q)).

Null selection (--null-mode)
    audit           gold_rich_club.py's generator, unchanged: up to 8 rewirings
                    per null seed, the first connected one kept (else the one
                    with the largest component).
    unconstrained   one rewiring per null seed, kept whether connected or not.
    Both draw the first rewiring from the same seeded stream, so the two
    ensembles differ only in nulls whose first rewiring is disconnected;
    n_null_disconnected of an unconstrained run counts exactly those.
    legacy_three_consecutive_k_verdict
                            gold_rich_club.py's verdict, unchanged.

Regime rule (Riedel et al. 2022, Methods, quoted)
    "We first selected the lowest and highest k-level with p < .05. If all
    interjacent p-values were also <.05, we defined the range between the two
    k-values as the rich club regime. If only one single or two
    non-neighboring p-values within this range were >.05, we still considered
    the range as rich club regime (thus considering these data points as
    outliers). In case that two or more neighboring p-values exceeded the .05
    threshold (i.e., cutting the range between the lowest and highest k-level
    with p < .05 in two or more), we assessed whether any of the ranges'
    length exceeded the other ones by a factor of 1.5. If no range was
    1.5-times larger, we assumed the range with larger k values (i.e., at the
    upper end of the normalized rich club curve) as the rich club regime. If
    none of the criteria applied, we did not assume a valid continuous rich
    club regime for this participant."  Length = upper end - lower end.
    Their p is FDR-corrected (BH 1995, all k).  The authors' released code
    (osf.io/wf4tv) only reads the finished regime (P_RC_Range_adj); the code
    that computes it is not released, so the rule is implemented from the
    text.  Readings taken where the text is not explicit:
      - p < .05 is BH q < q_alpha and rho > 1 (their p is the upper tail);
      - "neighboring" means adjacent in the tested k grid;
      - range length is upper - lower k, also for the 1.5 comparison, and a
        range is "1.5-times larger" if its length is > 0 and >= 1.5 times
        every other range's length;
      - in the cut case, isolated non-significant k stay inside their range;
      - three or more non-significant k, none neighboring, match no criterion,
        so there is no valid regime.
    Differences from their analysis that are kept: binary phi (theirs is
    weighted, BCT rich_club_wu), 1,000 null curves by default (theirs 2,500,
    BCT randmio_und, 10 rewires per edge), and the family starts at k = 0
    (their k = 1..n; at k = 0 every null equals the graph, so p = 1).

Dependence
    Adjacent thresholds share most of their nodes, so their p-values are
    dependent.  BH controls the FDR under independence or positive regression
    dependence (Benjamini & Yekutieli 2001); control under arbitrary
    dependence is NOT claimed here.  This is an established multiple-testing
    procedure, not an exact reproduction of one paper's study.

Run from the repository root:
    python graphnets/graphmetrics/onebyone/gold_rich_club_fdr.py --graph path/to/G.npz
    python graphnets/graphmetrics/onebyone/gold_rich_club_fdr.py --graph path/to/G.gml --json result.json
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import gold_rich_club
from gold_rich_club import _runs, load_layered_graph, rich_club


def bh_adjust(p_values: List[float]) -> List[float]:
    """BH-adjusted p-values in original order, monotonized and capped at one."""
    n = len(p_values)
    if n == 0:
        return []
    if any(not (0.0 <= float(p) <= 1.0) for p in p_values):
        raise ValueError("Each p-value must be between 0 and 1.")
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):
        index = order[rank - 1]
        running_min = min(running_min, p_values[index] * n / rank)
        adjusted[index] = running_min
    return adjusted


def riedel_regime(ks: List[int], sig: List[bool]) -> Tuple[Optional[Tuple[int, int]], str]:
    """Riedel et al. (2022) rich-club regime over the tested grid ks (sorted)
    with per-k significance sig.  Returns ((k_lower, k_upper), rule) or
    (None, reason); see the module docstring for the rule and its readings."""
    pos = [i for i, s in enumerate(sig) if s]
    if not pos:
        return None, "no significant k"
    lo, hi = pos[0], pos[-1]
    gaps = []  # runs of non-significant positions inside [lo, hi]
    for i in range(lo, hi + 1):
        if not sig[i]:
            if gaps and gaps[-1][1] == i - 1:
                gaps[-1][1] = i
            else:
                gaps.append([i, i])
    if not gaps:
        return (ks[lo], ks[hi]), "all k between lowest and highest significant k significant"
    if all(a == b for a, b in gaps):
        if len(gaps) <= 2:
            return (ks[lo], ks[hi]), f"{len(gaps)} isolated non-significant k treated as outliers"
        return None, f"{len(gaps)} isolated non-significant k: no criterion applies"
    pieces, start = [], lo
    for a, b in gaps:
        if b > a:  # two or more neighboring non-significant k cut the range
            pieces.append((start, a - 1))
            start = b + 1
    pieces.append((start, hi))
    lengths = [ks[b] - ks[a] for a, b in pieces]
    for i, li in enumerate(lengths):
        others = lengths[:i] + lengths[i + 1:]
        if li > 0 and li >= 1.5 * max(others):
            a, b = pieces[i]
            return (ks[a], ks[b]), f"cut into {len(pieces)} ranges; one is >= 1.5x every other"
    a, b = pieces[-1]
    return (ks[a], ks[b]), f"cut into {len(pieces)} ranges; none 1.5x longer, uppermost taken"


def add_fdr_results(result: Dict[str, object], alpha: float = 0.05) -> Dict[str, object]:
    """Annotate each tested k and return a separate BH-FDR report.

    Family: every valid k in the existing per_k table, without filtering
    on unadjusted significance or on degree before multiple-testing correction.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    rows = result.get("per_k", [])
    report = {
        "method": "Benjamini-Hochberg", "q": alpha,
        "null_model_type": result.get("null_model_type"),
        "null_mode": result.get("null_mode", "audit"),
        "n_null_curves": result.get("n_curves"), "seed": result.get("seed"),
        "n_null_disconnected": result.get("n_null_disconnected"),
        "n_tests": len(rows),
        "k_tested": [int(row["k"]) for row in rows],
        "legacy_three_consecutive_k_verdict": result.get("rich_club_verdict"),
    }
    if not rows:
        report.update(fdr_significant_k=[], n_fdr_significant=0,
                      fdr_significant_ranges=[], n_distinct_node_sets=0,
                      riedel2022_regime=None, regime_rule="no k tested",
                      regime_verdict="no_valid_regime")
        result["fdr_bh"] = report
        return result

    if result.get("n_curves"):
        p_min = 1.0 / (result["n_curves"] + 1)
        report.update(p_resolution=p_min, bh_rank1_cutoff=alpha / len(rows),
                      n_k_at_p_resolution=sum(math.isclose(float(r["p"]), p_min) for r in rows),
                      n_k_at_p_resolution_needed=math.ceil(len(rows) * p_min / alpha - 1e-12))

    adjusted = bh_adjust([float(row["p"]) for row in rows])
    significant = []
    for row, q in zip(rows, adjusted):
        row["p_fdr_bh"] = q
        # The direction check retains the original enrichment definition.
        passed = q < alpha and float(row["rho"]) > 1.0
        row["significant_fdr_bh"] = passed
        if passed:
            significant.append(int(row["k"]))

    n_above = {int(row["k"]): int(row["n_above_k"]) for row in rows}
    ranges = []
    for first, last in _runs(significant):
        sets = {n_above[k] for k in range(first, last + 1)}
        ranges.append({"k_first": first, "k_last": last,
                       "n_thresholds": last - first + 1,
                       "n_above_k_first": n_above[first],
                       "n_above_k_last": n_above[last],
                       "n_distinct_node_sets": len(sets)})
    ks = [int(row["k"]) for row in rows]
    span, rule = riedel_regime(ks, [row["significant_fdr_bh"] for row in rows])
    regime = None
    if span is not None:
        inside = [row for row in rows if span[0] <= row["k"] <= span[1]]
        regime = {"k_lower": span[0], "k_upper": span[1],
                  "length": span[1] - span[0],
                  "n_thresholds": len(inside),
                  "n_nonsignificant_inside": sum(not r["significant_fdr_bh"] for r in inside),
                  "n_above_k_lower": n_above[span[0]],
                  "n_above_k_upper": n_above[span[1]],
                  "n_distinct_node_sets": len({r["n_above_k"] for r in inside}),
                  "single_threshold": span[0] == span[1]}
    if regime is None:
        verdict = "no_valid_regime"
    elif regime["single_threshold"]:
        verdict = "single_threshold_regime"
    else:
        verdict = "valid_regime"
    report.update(
        fdr_significant_k=significant, n_fdr_significant=len(significant),
        fdr_significant_ranges=ranges,
        n_distinct_node_sets=len({n_above[k] for k in significant}),
        riedel2022_regime=regime, regime_rule=rule, regime_verdict=verdict,
    )
    result["fdr_bh"] = report
    return result


def _self_test() -> None:
    # Standard BH example, plus empty and monotonic adjusted p-values.
    p = [0.01, 0.04, 0.03, 0.20]
    q = bh_adjust(p)
    assert all(abs(a - b) < 1e-12 for a, b in zip(q, [0.04, 0.05333333333333334, 0.05333333333333334, 0.20]))
    assert bh_adjust([]) == []
    # Ties get equal adjusted values; p = 0 stays 0; p = 1 stays 1.
    assert bh_adjust([0.02, 0.02, 0.5]) == [0.03, 0.03, 0.5]
    assert bh_adjust([0.0, 1.0]) == [0.0, 1.0]
    # Against scipy's independent implementation.
    from scipy.stats import false_discovery_control
    for vals in (p, [0.02, 0.02, 0.5], [0.0, 1.0], [0.001, 0.2, 0.004, 0.03, 0.03, 0.9]):
        ref = false_discovery_control(vals, method="bh")
        assert all(abs(a - b) < 1e-12 for a, b in zip(bh_adjust(vals), ref)), vals
    # Isolated significant k, two consecutive runs, shared node sets.
    sample = {
        "per_k": [
            {"k": 2, "p": 0.001, "rho": 1.2, "n_above_k": 50},
            {"k": 3, "p": 0.001, "rho": 1.3, "n_above_k": 40},
            {"k": 4, "p": 0.001, "rho": 1.3, "n_above_k": 40},
            {"k": 5, "p": 0.50, "rho": 1.1, "n_above_k": 30},
            {"k": 6, "p": 0.001, "rho": 1.4, "n_above_k": 20},
            {"k": 7, "p": 0.001, "rho": 0.9, "n_above_k": 10},
        ],
        "rich_club_verdict": "pass",
    }
    out = add_fdr_results(sample)["fdr_bh"]
    assert out["fdr_significant_k"] == [2, 3, 4, 6]
    assert [(r["k_first"], r["k_last"], r["n_distinct_node_sets"])
            for r in out["fdr_significant_ranges"]] == [(2, 4, 2), (6, 6, 1)]
    assert out["n_distinct_node_sets"] == 3
    assert out["riedel2022_regime"]["k_lower"] == 2 and out["riedel2022_regime"]["k_upper"] == 6
    assert out["riedel2022_regime"]["n_distinct_node_sets"] == 4
    assert out["regime_verdict"] == "valid_regime"
    assert out["legacy_three_consecutive_k_verdict"] == "pass"
    # Each branch of the Riedel et al. (2022) rule, k = 1, 2, ...
    def regime(pattern):
        return riedel_regime(list(range(1, len(pattern) + 1)), [c == "T" for c in pattern])[0]
    assert regime("TTTTT") == (1, 5)               # all significant
    assert regime("FTTF") == (2, 3)                # outside the extremes ignored
    assert regime("TTFTT") == (1, 5)               # one outlier
    assert regime("TFTFT") == (1, 5)               # two non-neighboring outliers
    assert regime("TFTFTFT") is None               # three isolated: no criterion
    assert regime("TTTTTFFTTT") == (1, 5)          # cut; length 4 >= 1.5 x 2
    assert regime("TTTTFFTTT") == (1, 4)           # cut; length 3 >= 1.5 x 2 (counts 4, 3 would not be)
    assert regime("TTTFFTTT") == (6, 8)            # cut; equal lengths -> uppermost
    assert regime("TTTTFFT") == (1, 4)             # cut; length 3 vs 0
    assert regime("TFFTFFT") == (7, 7)             # cut; all lengths 0 -> uppermost
    assert regime("TTTFTFFTT") == (1, 5)           # outlier stays inside its range
    assert regime("FTF") == (2, 2)                 # one significant k: length 0
    assert regime("FFF") is None                   # nothing significant
    empty = add_fdr_results({"per_k": [], "rich_club_verdict": "not_tested"})["fdr_bh"]
    assert empty["n_tests"] == 0 and empty["regime_verdict"] == "no_valid_regime"
    none = add_fdr_results({"per_k": [{"k": k, "p": 0.5, "rho": 1.1, "n_above_k": 9 - k}
                                      for k in range(4)]})["fdr_bh"]
    assert none["riedel2022_regime"] is None and none["regime_verdict"] == "no_valid_regime"
    # One significant k: the rule returns it, flagged as a single threshold.
    one = add_fdr_results({"per_k": [{"k": k, "p": 0.001 if k == 2 else 0.5, "rho": 1.1,
                                      "n_above_k": 9 - k} for k in range(4)]})["fdr_bh"]
    assert one["riedel2022_regime"]["single_threshold"] is True
    assert one["regime_verdict"] == "single_threshold_regime"
    # p resolution: 1,000 curves, 500 tests -> p_min 1/1001, rank-1 cutoff 1e-4,
    # BH needs 10 k at p_min (10 x 0.05 / 500 = 0.001 >= 1/1001; 9 would not do).
    res = add_fdr_results({"n_curves": 1000, "per_k": [
        {"k": k, "p": 1 / 1001 if k < 10 else 0.5, "rho": 1.1, "n_above_k": 600 - k}
        for k in range(500)]})["fdr_bh"]
    assert res["n_k_at_p_resolution"] == 10 and res["n_k_at_p_resolution_needed"] == 10
    assert abs(res["bh_rank1_cutoff"] - 1e-4) < 1e-15 and res["n_fdr_significant"] == 10
    res9 = add_fdr_results({"n_curves": 1000, "per_k": [
        {"k": k, "p": 1 / 1001 if k < 9 else 0.5, "rho": 1.1, "n_above_k": 600 - k}
        for k in range(500)]})["fdr_bh"]
    assert res9["n_fdr_significant"] == 0
    print("BH-FDR self-tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rich-club per-k significance with Benjamini-Hochberg FDR."
    )
    parser.add_argument("--graph", help="Input graph (.gml or .npz)")
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--n-null-rc", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null-mode", choices=["audit", "unconstrained"], default="audit",
                        help="audit: gold_rich_club.py's connected-preferring nulls (default); "
                             "unconstrained: one rewiring per seed, kept as drawn")
    parser.add_argument("--json", default="", help="Optional output JSON path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.graph:
        parser.error("--graph is required unless --self-test is used")
    if args.n_null < 0 or args.n_null_rc < 0:
        parser.error("Null counts cannot be negative")
    if args.n_null == 0 and args.n_null_rc == 0:
        parser.error("FDR significance needs null graphs; supply --n-null-rc > 0")
    if args.json and os.path.exists(args.json):
        parser.error(f"{args.json} exists; this script never overwrites a result file")

    if args.null_mode == "unconstrained":
        # The workers look _generate_null up in gold_rich_club at call time and
        # inherit this module state through fork; retries=1 keeps the first draw.
        gold_rich_club._generate_null = functools.partial(gold_rich_club._generate_null, retries=1)
    layered = load_layered_graph(args.graph)
    result = rich_club(
        layered, n_null=args.n_null, n_null_rc=args.n_null_rc,
        seed=args.seed, workers=args.workers,
    )
    result["null_mode"] = args.null_mode
    add_fdr_results(result, alpha=args.alpha)
    summary = result["fdr_bh"]
    print("\nRich-club significance (BH-FDR)")
    print(f"  {'k':>5s} {'nodes > k':>9s} {'phi':>8s} {'phi_null':>8s} {'rho':>7s}"
          f" {'emp. p':>8s} {'BH p':>8s}  sig")
    for row in result["per_k"]:
        print(f"  {row['k']:>5d} {row['n_above_k']:>9d} {row['phi']:>8.4f}"
              f" {row['phi_null_mean']:>8.4f} {row['rho']:>7.4f} {row['p']:>8.4f}"
              f" {row['p_fdr_bh']:>8.4f}  {'yes' if row['significant_fdr_bh'] else '-'}")
    ks = summary["k_tested"]
    print(f"\n  Null model: {summary['null_model_type']} ({summary['null_mode']} selection),"
          f" {summary['n_null_curves']} curves, seed {summary['seed']},"
          f" {summary['n_null_disconnected']} disconnected")
    print(f"  Tested family: {summary['n_tests']} thresholds"
          + (f", k {ks[0]}..{ks[-1]}" if ks else ""))
    print(f"  FDR q: {summary['q']}")
    if "p_resolution" in summary:
        print(f"  p resolution {summary['p_resolution']:.6f}; BH rank-1 cutoff {summary['bh_rank1_cutoff']:.6f};"
              f" {summary['n_k_at_p_resolution']} k at the resolution, BH needs"
              f" {summary['n_k_at_p_resolution_needed']} there to reject any")
    print(f"  FDR-significant thresholds: {summary['n_fdr_significant']}"
          f" ({summary['n_distinct_node_sets']} distinct node sets)")
    print("  Consecutive ranges:")
    for r in summary["fdr_significant_ranges"]:
        print(f"    k {r['k_first']}..{r['k_last']}: {r['n_thresholds']} thresholds,"
              f" nodes > k {r['n_above_k_first']} -> {r['n_above_k_last']},"
              f" {r['n_distinct_node_sets']} distinct node sets")
    reg = summary["riedel2022_regime"]
    print(f"  Riedel et al. (2022) regime: {summary['regime_verdict']} ({summary['regime_rule']})")
    if reg:
        print(f"    k {reg['k_lower']}..{reg['k_upper']}, length {reg['length']},"
              f" {reg['n_nonsignificant_inside']} non-significant k inside,"
              f" nodes > k {reg['n_above_k_lower']} -> {reg['n_above_k_upper']},"
              f" {reg['n_distinct_node_sets']} distinct node sets"
              + (" (single threshold)" if reg["single_threshold"] else ""))
    print(f"  Legacy 3-consecutive-k verdict (unchanged): {summary['legacy_three_consecutive_k_verdict']}")
    if args.json:
        with open(args.json, "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"  Wrote {args.json}")


if __name__ == "__main__":
    main()
