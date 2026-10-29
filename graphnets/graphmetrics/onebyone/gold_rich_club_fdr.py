#!/usr/bin/env python3
"""Benjamini-Hochberg FDR analysis of rich-club thresholds.

This is a standalone companion to gold_rich_club.py. It reuses that script's
loading, null-graph generation, rich-club curve, and empirical per-k p-values,
without changing its original three-consecutive-k verdict.

Benjamini-Hochberg (1995) controls the false discovery rate under its usual
dependence assumptions. Adjacent rich-club thresholds are dependent, and
arbitrary-dependence FDR guarantees are NOT claimed here. This is an established
multiple-testing procedure, not an exact reproduction of one paper's study.

Run from the repository root:
    python graphnets/graphmetrics/onebyone/gold_rich_club_fdr.py --graph path/to/G.npz
    python graphnets/graphmetrics/onebyone/gold_rich_club_fdr.py --graph path/to/G.gml --json result.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from gold_rich_club import load_layered_graph, rich_club


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
    """Annotate each tested k and return a separate BH-FDR decision.

    Family: every valid k in the existing per_k table, without filtering
    on unadjusted significance or on degree before multiple-testing correction.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    rows = result.get("per_k", [])
    if not rows:
        result["fdr_bh"] = {
            "method": "Benjamini-Hochberg", "alpha": alpha,
            "n_tests": 0, "significant_k": [], "verdict": "not_tested"
        }
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
    result["fdr_bh"] = {
        "method": "Benjamini-Hochberg",
        "alpha": alpha, "n_tests": len(rows),
        "significant_k": significant, "n_significant": len(significant),
        "verdict": "pass" if significant else "fail"
    }
    return result


def _self_test() -> None:
    # Standard BH example, plus empty and monotonic adjusted p-values.
    p = [0.01, 0.04, 0.03, 0.20]
    q = bh_adjust(p)
    assert all(abs(a - b) < 1e-12 for a, b in zip(q, [0.04, 0.05333333333333334, 0.05333333333333334, 0.20]))
    assert bh_adjust([]) == []
    sample = {
        "per_k": [
            {"k": 2, "p": 0.01, "rho": 1.2},
            {"k": 3, "p": 0.04, "rho": 1.3},
            {"k": 4, "p": 0.03, "rho": 0.9},
            {"k": 5, "p": 0.20, "rho": 1.1},
        ],
        "rich_club_verdict": "fail",
    }
    out = add_fdr_results(sample)
    assert out["fdr_bh"]["significant_k"] == [2]
    assert out["rich_club_verdict"] == "fail"
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
    print(f"  Tested degree thresholds: {summary['n_tests']}")
    print(f"  FDR alpha: {summary['alpha']}")
    print(f"  BH-significant thresholds: {summary['significant_k']}")
    print(f"  BH-FDR verdict: {summary['verdict']}")
    print(f"  Existing 3-consecutive-k verdict: {result['rich_club_verdict']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"  Wrote {args.json}")


if __name__ == "__main__":
    main()
