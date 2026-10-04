#!/usr/bin/env python3
"""Compare ROOT and converted-NPZ burst-gap outputs from the same entry range.

This is a data-equivalence check, not a physics comparison or fitted-model test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import zip_longest
from pathlib import Path


TABLES = ("bursts.csv", "gaps.csv", "t0_group_gaps.csv", "alignment_candidates.csv")
COUNT_PAIRS = (
    ("quality_good_windows", "converted_quality_good_windows"),
    ("selected_windows", "selected_windows"),
    ("windows_with_bursts", "windows_with_bursts"),
    ("windows_with_multiple_bursts", "windows_with_multiple_bursts"),
)


def equal_cell(left: str, right: str, tolerance_ns: float) -> bool:
    if left == right:
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance_ns)
    except ValueError:
        return False


def compare(root_dir: Path, npz_dir: Path, *, tolerance_ns: float = 1e-6,
            max_examples: int = 20) -> dict:
    if tolerance_ns < 0 or max_examples < 1:
        raise ValueError("Tolerance must be nonnegative and max_examples positive")
    root = json.loads((root_dir / "summary.json").read_text(encoding="utf-8"))
    npz = json.loads((npz_dir / "summary.json").read_text(encoding="utf-8"))
    settings_equal = all(root.get(name) == npz.get(name)
                         for name in ("selection", "tdc_time_mode"))
    root_counts = root.get("counts", {})
    npz_counts = npz.get("counts", {})
    count_comparisons = {
        root_key: {"root": root_counts.get(root_key, 0), "npz": npz_counts.get(npz_key, 0)}
        for root_key, npz_key in COUNT_PAIRS
    }
    count_comparisons["t0_group_gap_count"] = {
        "root": root.get("t0_group_gap_count", 0), "npz": npz.get("t0_group_gap_count", 0)}
    tables: dict[str, dict] = {}
    all_equal = settings_equal and all(item["root"] == item["npz"]
                                       for item in count_comparisons.values())
    for filename in TABLES:
        mismatches = []
        mismatch_count = 0
        n_rows = 0
        with (root_dir / filename).open(newline="", encoding="utf-8") as root_handle, \
             (npz_dir / filename).open(newline="", encoding="utf-8") as npz_handle:
            left = csv.DictReader(root_handle)
            right = csv.DictReader(npz_handle)
            same_columns = left.fieldnames == right.fieldnames
            for row_number, (a, b) in enumerate(zip_longest(left, right), 1):
                n_rows += 1
                if a is None or b is None or not same_columns or any(
                        not equal_cell(a[key], b[key], tolerance_ns) for key in a):
                    mismatch_count += 1
                    if len(mismatches) < max_examples:
                        mismatches.append({"row": row_number,
                                           "root_entry": (a or b).get("root_entry"),
                                           "root": a, "npz": b})
            tables[filename] = {"rows_compared": n_rows, "same_columns": same_columns,
                                "mismatch_examples": mismatches,
                                "mismatch_count": mismatch_count}
            if mismatch_count or not same_columns:
                all_equal = False
    return {"equivalent_within_tolerance": all_equal,
            "numeric_tolerance": tolerance_ns,
            "same_selection_and_tdc_mode": settings_equal,
            "count_comparisons": count_comparisons, "tables": tables,
            "note": "NPZ conversion can omit PMTs absent from the WCSim map; mismatches must be inspected, not silently accepted."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_output_dir", type=Path)
    parser.add_argument("npz_output_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tolerance-ns", type=float, default=1e-6)
    args = parser.parse_args()
    result = compare(args.root_output_dir, args.npz_output_dir,
                     tolerance_ns=args.tolerance_ns)
    output = args.output or args.npz_output_dir / "root_npz_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"ROOT/NPZ equivalent within tolerance: {result['equivalent_within_tolerance']}")
    print(f"Comparison: {output}")
    return 0 if result["equivalent_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
