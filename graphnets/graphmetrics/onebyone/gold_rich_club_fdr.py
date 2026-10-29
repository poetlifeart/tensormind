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
    regime_verdict          "not_evaluated".  No published rule turning
                            FDR-significant k into a rich-club regime has been
                            verified and implemented here, so none is applied.
    legacy_three_consecutive_k_verdict
                            gold_rich_club.py's verdict, unchanged.

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
import json
from typing import Dict, List

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
        "n_null_curves": result.get("n_curves"), "seed": result.get("seed"),
        "n_tests": len(rows),
        "k_tested": [int(row["k"]) for row in rows],
        "regime_verdict": "not_evaluated",
        "legacy_three_consecutive_k_verdict": result.get("rich_club_verdict"),
    }
    if not rows:
        report.update(fdr_significant_k=[], n_fdr_significant=0,
                      fdr_significant_ranges=[], n_distinct_node_sets=0)
        result["fdr_bh"] = report
        return result

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
    report.update(
        fdr_significant_k=significant, n_fdr_significant=len(significant),
        fdr_significant_ranges=ranges,
        n_distinct_node_sets=len({n_above[k] for k in significant}),
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
    assert out["regime_verdict"] == "not_evaluated"
    assert out["legacy_three_consecutive_k_verdict"] == "pass"
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

    layered = load_layered_graph(args.graph)
    result = rich_club(
        layered, n_null=args.n_null, n_null_rc=args.n_null_rc,
        seed=args.seed, workers=args.workers,
    )
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
    print(f"\n  Null model: {summary['null_model_type']}, {summary['n_null_curves']} curves, seed {summary['seed']}")
    print(f"  Tested family: {summary['n_tests']} thresholds"
          + (f", k {ks[0]}..{ks[-1]}" if ks else ""))
    print(f"  FDR q: {summary['q']}")
    print(f"  FDR-significant thresholds: {summary['n_fdr_significant']}"
          f" ({summary['n_distinct_node_sets']} distinct node sets)")
    print("  Consecutive ranges:")
    for r in summary["fdr_significant_ranges"]:
        print(f"    k {r['k_first']}..{r['k_last']}: {r['n_thresholds']} thresholds,"
              f" nodes > k {r['n_above_k_first']} -> {r['n_above_k_last']},"
              f" {r['n_distinct_node_sets']} distinct node sets")
    print(f"  Regime verdict: {summary['regime_verdict']}")
    print(f"  Legacy 3-consecutive-k verdict (unchanged): {summary['legacy_three_consecutive_k_verdict']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"  Wrote {args.json}")


if __name__ == "__main__":
    main()
