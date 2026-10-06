#!/usr/bin/env python3
"""Compare PMT burst structure after applying a historical LED calibration.

This diagnostic reads raw ``hit_pmt_times`` from a WCTE ROOT file and applies
per-channel offsets from a historical Mohit LED JSON file. It deliberately
does not modify the ROOT file or the production-calibrated branch. Burst
finding and beamline selection are shared with ``study_wcte_burst_gaps.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

try:
    from scripts.study_wcte_burst_gaps import REQUIRED, beamline_status, find_bursts
except ModuleNotFoundError:
    from study_wcte_burst_gaps import REQUIRED, beamline_status, find_bursts


RAW_REQUIRED = tuple(name if name != "hit_pmt_calibrated_times"
                     else "hit_pmt_times" for name in REQUIRED)


def load_offsets(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("calibration_name") != "timing_offsets":
        raise ValueError("Calibration JSON is not a timing_offsets export")
    offsets = {int(row["channel_id"]): float(row["timing_offset"])
               for row in payload.get("data", [])}
    if not offsets:
        raise ValueError("Calibration JSON contains no channel offsets")
    return offsets


def raw_hits(event, offsets: dict[int, float]):
    mask = np.asarray(ak.to_numpy(event["hit_pmt_readout_mask"])) == 0
    raw = np.asarray(ak.to_numpy(event["hit_pmt_times"]), dtype=float)[mask]
    charges = np.asarray(ak.to_numpy(event["hit_pmt_charges"]), dtype=float)[mask]
    slots = np.asarray(ak.to_numpy(event["hit_mpmt_slot_ids"]), dtype=np.int64)[mask]
    positions = np.asarray(ak.to_numpy(event["hit_pmt_position_ids"]), dtype=np.int64)[mask]
    channels = 100 * slots + positions
    correction = np.asarray([offsets.get(int(channel), np.nan) for channel in channels])
    valid = np.isfinite(raw) & np.isfinite(charges) & np.isfinite(correction)
    return raw[valid] - correction[valid], charges[valid], channels[valid]


def run(args: argparse.Namespace) -> dict:
    offsets = load_offsets(args.calibration_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(0, args.max_ns + args.hist_bin_ns, args.hist_bin_ns)
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    counts = {"raw_windows": 0, "quality_good_windows": 0,
              "selected_windows": 0, "windows_with_multiple_bursts": 0,
              "hits_without_historical_offset": 0, "bursts": 0}
    with uproot.open(args.input_root) as root:
        tree = root["WCTEReadoutWindows"]
        missing = set(RAW_REQUIRED) - set(tree.keys())
        if missing:
            raise ValueError("Missing ROOT branches: " + ", ".join(sorted(missing)))
        stop = min(tree.num_entries, args.entry_start + args.scan_windows)
        with (args.output_dir / "bursts.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("root_entry", "run_id", "event_number",
                "readout_number", "burst_index", "strongest", "center_ns", "start_ns",
                "end_ns", "n_hits", "n_pmts", "charge"))
            writer.writeheader()
            for start in range(args.entry_start, stop, args.batch_windows):
                batch = tree.arrays(RAW_REQUIRED, entry_start=start,
                                    entry_stop=min(start + args.batch_windows, stop), library="ak")
                for offset, event in enumerate(batch):
                    counts["raw_windows"] += 1
                    if int(event["window_data_quality_mask"]) != 0:
                        continue
                    counts["quality_good_windows"] += 1
                    beam_ok, tagged, _, _ = beamline_status(event, tdc_time_mode="reference")
                    if args.selection == "beam" and not beam_ok:
                        continue
                    if args.selection == "tagged" and not tagged:
                        continue
                    counts["selected_windows"] += 1
                    times, charges, channels = raw_hits(event, offsets)
                    total_hits = int(np.asarray(ak.to_numpy(event["hit_pmt_times"])).size)
                    counts["hits_without_historical_offset"] += total_hits - len(times)
                    bursts = find_bursts(times, charges, channels, min_ns=args.min_ns,
                        max_ns=args.max_ns, bin_ns=args.bin_ns, width_ns=args.width_ns,
                        min_pmts=args.min_pmts, min_separation_ns=args.min_separation_ns,
                        max_bursts=args.max_bursts)
                    counts["bursts"] += len(bursts)
                    if len(bursts) > 1:
                        counts["windows_with_multiple_bursts"] += 1
                    strongest = max(range(len(bursts)), key=lambda i: bursts[i].n_pmts) if bursts else -1
                    for i, burst in enumerate(bursts):
                        writer.writerow({"root_entry": start + offset, "run_id": int(event["run_id"]),
                            "event_number": int(event["event_number"]), "readout_number": int(event["readout_number"]),
                            "burst_index": i, "strongest": int(i == strongest), **burst.__dict__})
                    for i in range(1, len(bursts)):
                        histogram += np.histogram([bursts[i].center_ns - bursts[i-1].center_ns], bins=edges)[0]
    manifest = {"input_root": str(args.input_root), "source_branch": "hit_pmt_times",
        "calibration_file": str(args.calibration_json),
        "calibration_sha256": hashlib.sha256(args.calibration_json.read_bytes()).hexdigest(),
        "correction": "corrected_time_ns = raw_time_ns - timing_offset",
        "selection": args.selection, "counts": counts,
        "warning": "Comparison diagnostic only; production-calibrated PMT times were not used."}
    (args.output_dir / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].stairs(histogram, edges)
    axes[0].set_xlim(0, min(args.max_ns, 1500))
    axes[1].stairs(histogram, edges)
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set(xlabel="Consecutive PMT burst gap (ns)", ylabel="Gaps / bin")
        axis.grid(alpha=0.2)
    fig.suptitle("ROOT burst gaps after historical Mohit LED timing correction")
    fig.savefig(args.output_dir / "burst_gap_histograms.png", dpi=170)
    plt.close(fig)
    print(json.dumps(counts, indent=2))
    print(f"Results: {args.output_dir}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("calibration_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", choices=("all", "beam", "tagged"), default="tagged")
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--scan-windows", type=int, default=10_000)
    parser.add_argument("--batch-windows", type=int, default=500)
    parser.add_argument("--min-ns", type=float, default=0)
    parser.add_argument("--max-ns", type=float, default=10_000)
    parser.add_argument("--bin-ns", type=float, default=10)
    parser.add_argument("--width-ns", type=float, default=50)
    parser.add_argument("--min-pmts", type=int, default=10)
    parser.add_argument("--min-separation-ns", type=float, default=100)
    parser.add_argument("--max-bursts", type=int, default=20)
    parser.add_argument("--hist-bin-ns", type=float, default=50)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
