#!/usr/bin/env python3
"""Run the ROOT burst-gap study on provenance-preserving converted WCTE NPZ.

PMT times remain on the calibrated readout-window axis. Beamline TDC hits
are corrected only within their own clock; no TDC-to-PMT offset is inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.study_wcte_burst_gaps import (
        Burst, HD_IDS, beamline_status, corrected_beamline_hits, find_bursts,
        later_hodoscope_match_counts, plot_histograms,
        plot_t0_comparison,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from study_wcte_burst_gaps import (
        Burst, HD_IDS, beamline_status, corrected_beamline_hits, find_bursts,
        later_hodoscope_match_counts, plot_histograms,
        plot_t0_comparison,
    )


REQUIRED_NPZ = (
    "root_entry", "run_id", "event_id", "readout_number",
    "digi_hit_time", "digi_hit_charge", "digi_hit_pmt",
    "beamline_pmt_tdc_ids", "beamline_pmt_tdc_times",
)


def plot_hodoscope_times(path: Path, times_by_element: dict[int, list[float]],
                        *, tdc_time_mode: str) -> None:
    """Plot HD hit times relative to the earliest T0 group for each readout."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    edges = np.arange(-500, 10_000 + 50, 50)
    for element, axis in enumerate(axes.flat):
        values = times_by_element[element]
        if values:
            axis.hist(values, bins=edges, histtype="step", color="tab:purple", linewidth=1.2)
        else:
            axis.text(0.5, 0.5, "No hits", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(f"HD{element} (n={len(values):,})", fontsize=9)
        axis.grid(alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel("HD time − earliest T0 group (ns)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Hits / 50 ns")
    fig.suptitle(f"Beamline hodoscope timing by element ({tdc_time_mode} TDC mode; "
                 "no additional inter-bank offset)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def conversion_parts(directory: Path) -> tuple[list[Path], dict]:
    manifests = list(directory.glob("*_conversion_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Expected exactly one conversion manifest in {directory}; found {len(manifests)}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version", 0) < 2 or not manifest.get("root_entry_preserved"):
        raise ValueError("NPZ lacks original ROOT entries; regenerate with converter schema v2")
    if not manifest.get("beamline_tdc_fields_preserved"):
        raise ValueError("NPZ lacks beamline TDC hits; regenerate from a ROOT file with those branches")
    if manifest.get("timing", {}).get("additional_offsets_applied_by_converter") != []:
        raise ValueError("Expected converter to preserve calibrated PMT time without extra offsets")
    cuts = manifest.get("applied_quality_cuts", {})
    if cuts.get("t5") or cuts.get("vme"):
        raise ValueError("This study expects the default window/hit cuts without additional T5 or VME cuts")
    parts = [directory / item["file"] for item in manifest.get("output_parts", [])]
    if not parts or any(not part.is_file() for part in parts):
        raise FileNotFoundError("One or more NPZ parts listed in the manifest are missing")
    return parts, manifest


def run(args: argparse.Namespace) -> dict:
    if (args.entry_start < 0 or (args.entry_stop is not None and args.entry_stop <= args.entry_start)
            or args.max_ns <= args.min_ns or args.min_ns < 0 or args.bin_ns <= 0
            or args.width_ns <= 0 or args.min_pmts < 1 or args.min_separation_ns <= 0
            or args.max_bursts < 1 or args.hist_bin_ns <= 0
            or args.prompt_max_ns <= args.prompt_min_ns):
        raise ValueError("Invalid entry range or burst-finding settings")
    parts, manifest = conversion_parts(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(0, args.max_ns - args.min_ns + args.hist_bin_ns, args.hist_bin_ns)
    histograms = {(kind, group): np.zeros(len(edges) - 1, dtype=np.int64)
                  for kind in ("consecutive", "later_from_strongest")
                  for group in ("one_t0_group", "multiple_t0_groups", "no_t0")}
    t0_hist = np.zeros(len(edges) - 1, dtype=np.int64)
    counts: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    alignment_candidates: list[float] = []
    hd_times_by_element = {element: [] for element in range(15)}
    previous_entry = -1
    with (args.output_dir / "bursts.csv").open("w", newline="", encoding="utf-8") as burst_file, \
         (args.output_dir / "gaps.csv").open("w", newline="", encoding="utf-8") as gap_file, \
         (args.output_dir / "t0_group_gaps.csv").open("w", newline="", encoding="utf-8") as t0_file, \
         (args.output_dir / "alignment_candidates.csv").open("w", newline="", encoding="utf-8") as align_file, \
         (args.output_dir / "hodoscope_hits.csv").open("w", newline="", encoding="utf-8") as hd_file:
        identity_fields = ("root_entry", "run_id", "event_number", "readout_number")
        burst_writer = csv.DictWriter(burst_file, fieldnames=(
            *identity_fields, "n_t0_groups", "n_late_hd_hits", "n_late_hd_groups",
            "burst_index", "strongest", *Burst.__dataclass_fields__))
        gap_writer = csv.DictWriter(gap_file, fieldnames=(
            *identity_fields, "n_t0_groups", "kind", "from_burst", "to_burst", "gap_ns"))
        t0_writer = csv.DictWriter(t0_file, fieldnames=(
            *identity_fields, "from_t0_group", "to_t0_group", "from_time_ns", "to_time_ns", "gap_ns"))
        align_writer = csv.DictWriter(align_file, fieldnames=(
            *identity_fields, "t0_ns", "pmt_burst_ns", "pmt_minus_t0_ns", "n_pmt_bursts"))
        hd_writer = csv.DictWriter(hd_file, fieldnames=(
            *identity_fields, "channel_id", "hd_element", "corrected_time_ns",
            "time_from_earliest_t0_ns"))
        for writer in (burst_writer, gap_writer, t0_writer, align_writer, hd_writer):
            writer.writeheader()
        for part in parts:
            with np.load(part, allow_pickle=True) as archive:
                missing = set(REQUIRED_NPZ) - set(archive.files)
                if missing:
                    raise ValueError(f"{part} missing NPZ fields: {', '.join(sorted(missing))}")
                fields = {name: archive[name] for name in REQUIRED_NPZ}
                n = len(fields["root_entry"])
                if any(len(values) != n for values in fields.values()):
                    raise ValueError(f"{part} has unequal event-array lengths")
                for i in range(n):
                    entry = int(fields["root_entry"][i])
                    if entry <= previous_entry:
                        raise ValueError("NPZ parts are not strictly ordered by ROOT entry")
                    previous_entry = entry
                    if entry < args.entry_start or (args.entry_stop is not None and entry >= args.entry_stop):
                        continue
                    counts["converted_quality_good_windows"] += 1
                    event = {name: values[i] for name, values in fields.items()}
                    beam_ok, tagged, t0_centers, refs = beamline_status(
                        event, tdc_time_mode=args.tdc_time_mode)
                    if refs[0] is None:
                        counts["windows_missing_tdc_ref31"] += 1
                    if refs[1] is None:
                        counts["windows_missing_tdc_ref46"] += 1
                    if args.selection == "beam" and not beam_ok:
                        continue
                    if args.selection == "tagged" and not tagged:
                        continue
                    counts["selected_windows"] += 1
                    n_late_hd_hits, n_late_hd_groups = later_hodoscope_match_counts(
                        event, t0_centers, tdc_time_mode=args.tdc_time_mode)
                    corrected_hits, _ = corrected_beamline_hits(
                        event, tdc_time_mode=args.tdc_time_mode)
                    n_t0 = len(t0_centers)
                    identity = {"root_entry": entry, "run_id": int(event["run_id"]),
                                "event_number": int(event["event_id"]),
                                "readout_number": int(event["readout_number"]),
                                "n_t0_groups": n_t0}
                    if t0_centers:
                        for channel_id, hd_time in corrected_hits:
                            if channel_id not in HD_IDS:
                                continue
                            element = HD_IDS[channel_id]
                            dt = hd_time - t0_centers[0]
                            hd_times_by_element[element].append(dt)
                            hd_writer.writerow({**{key: identity[key] for key in identity_fields},
                                                "channel_id": channel_id,
                                                "hd_element": element,
                                                "corrected_time_ns": hd_time,
                                                "time_from_earliest_t0_ns": dt})
                    for j in range(1, n_t0):
                        gap = t0_centers[j] - t0_centers[j - 1]
                        t0_writer.writerow({**{key: identity[key] for key in identity_fields},
                                            "from_t0_group": j - 1, "to_t0_group": j,
                                            "from_time_ns": t0_centers[j - 1],
                                            "to_time_ns": t0_centers[j], "gap_ns": gap})
                        t0_hist += np.histogram([gap], bins=edges)[0]
                    times = np.asarray(event["digi_hit_time"], dtype=float).reshape(-1)
                    charges = np.asarray(event["digi_hit_charge"], dtype=float).reshape(-1)
                    pmts = np.asarray(event["digi_hit_pmt"], dtype=int).reshape(-1)
                    if not (len(times) == len(charges) == len(pmts)):
                        raise ValueError(f"Unequal PMT-hit arrays in {part}, ROOT entry {entry}")
                    finite = np.isfinite(times) & np.isfinite(charges)
                    bursts = find_bursts(times[finite], charges[finite], pmts[finite],
                                         min_ns=args.min_ns, max_ns=args.max_ns,
                                         bin_ns=args.bin_ns, width_ns=args.width_ns,
                                         min_pmts=args.min_pmts,
                                         min_separation_ns=args.min_separation_ns,
                                         max_bursts=args.max_bursts)
                    multiplicities[len(bursts)] += 1
                    if not bursts:
                        continue
                    counts["windows_with_bursts"] += 1
                    if len(bursts) > 1:
                        counts["windows_with_multiple_bursts"] += 1
                    prompt = [burst for burst in bursts
                              if args.prompt_min_ns <= burst.center_ns < args.prompt_max_ns]
                    if args.tdc_time_mode == "reference" and n_t0 == 1 and len(prompt) == 1:
                        candidate = prompt[0].center_ns - t0_centers[0]
                        alignment_candidates.append(candidate)
                        align_writer.writerow({**{key: identity[key] for key in identity_fields},
                                               "t0_ns": t0_centers[0],
                                               "pmt_burst_ns": prompt[0].center_ns,
                                               "pmt_minus_t0_ns": candidate,
                                               "n_pmt_bursts": len(bursts)})
                    strongest = max(range(len(bursts)), key=lambda j: bursts[j].n_pmts)
                    group = "no_t0" if n_t0 == 0 else "one_t0_group" if n_t0 == 1 else "multiple_t0_groups"
                    for j, burst in enumerate(bursts):
                        burst_writer.writerow({**identity, "burst_index": j,
                                               "n_late_hd_hits": n_late_hd_hits,
                                               "n_late_hd_groups": n_late_hd_groups,
                                               "strongest": int(j == strongest), **asdict(burst)})
                    for j in range(1, len(bursts)):
                        gap = bursts[j].center_ns - bursts[j - 1].center_ns
                        gap_writer.writerow({**identity, "kind": "consecutive",
                                             "from_burst": j - 1, "to_burst": j, "gap_ns": gap})
                        histograms[("consecutive", group)] += np.histogram([gap], bins=edges)[0]
                    for j in range(strongest + 1, len(bursts)):
                        gap = bursts[j].center_ns - bursts[strongest].center_ns
                        gap_writer.writerow({**identity, "kind": "later_from_strongest",
                                             "from_burst": strongest, "to_burst": j, "gap_ns": gap})
                        histograms[("later_from_strongest", group)] += np.histogram([gap], bins=edges)[0]
            print(f"{part.name}: {n} converted windows read", flush=True)
    summary = {
        "input_manifest": str(next(args.input_dir.glob("*_conversion_manifest.json"))),
        "input_root": manifest["input_root"], "source_kind": "converted_npz",
        "counts": dict(counts),
        "burst_multiplicity_per_selected_window": {str(k): v for k, v in sorted(multiplicities.items())},
        "selection": args.selection, "tdc_time_mode": args.tdc_time_mode,
        "alignment_candidates": {
            "count": len(alignment_candidates),
            "prompt_pmt_range_ns": [args.prompt_min_ns, args.prompt_max_ns],
            "median_pmt_minus_t0_ns": (float(np.median(alignment_candidates))
                                        if alignment_candidates else None),
            "mad_ns": (float(np.median(np.abs(np.asarray(alignment_candidates)
                                             - np.median(alignment_candidates))))
                       if alignment_candidates else None),
            "note": "Empirical offset includes particle/light propagation; not applied to PMT hits."},
        "settings": {key: getattr(args, key) for key in (
            "entry_start", "entry_stop", "min_ns", "max_ns", "bin_ns", "width_ns",
            "min_pmts", "min_separation_ns", "max_bursts", "hist_bin_ns")},
        "interpretation": "PMT burst definitions match the ROOT study. Converted NPZ omits bad windows and unmapped PMTs; compare by root_entry. No cross-system clock alignment or particle ID is inferred.",
        "t0_group_gap_count": int(t0_hist.sum()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_histograms(args.output_dir / "burst_gap_histograms.png", histograms, edges,
                    counts["selected_windows"], counts["windows_with_multiple_bursts"])
    pmt_consecutive = sum((histograms[("consecutive", group)] for group in (
        "one_t0_group", "multiple_t0_groups", "no_t0")), np.zeros(len(edges) - 1, dtype=np.int64))
    plot_t0_comparison(args.output_dir / "t0_vs_pmt_gap_histograms.png",
                       t0_hist, pmt_consecutive, edges)
    plot_hodoscope_times(args.output_dir / "hodoscope_time_histograms.png", hd_times_by_element,
                         tdc_time_mode=args.tdc_time_mode)
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    if alignment_candidates:
        axis.hist(alignment_candidates, bins=100, histtype="step", color="tab:green")
    else:
        axis.text(0.5, 0.5, "No qualifying alignment candidates",
                  transform=axis.transAxes, ha="center", va="center")
    axis.set(xlabel="Prompt PMT burst − reference-corrected T0 (ns)",
             ylabel="Windows / bin", title="Empirical cross-system alignment candidates")
    axis.grid(alpha=0.2)
    fig.savefig(args.output_dir / "alignment_candidates.png", dpi=170)
    plt.close(fig)
    print(f"Read {counts['converted_quality_good_windows']:,} quality-selected NPZ windows; "
          f"selected {counts['selected_windows']:,}; "
          f"{counts['windows_with_multiple_bursts']:,} have 2+ PMT bursts")
    print(f"Results: {args.output_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="converted_npz directory with schema-v2 manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", choices=("all", "beam", "tagged"), default="tagged")
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--entry-stop", type=int, default=None,
                        help="Exclusive original ROOT entry; omit for every converted window")
    parser.add_argument("--min-ns", type=float, default=0)
    parser.add_argument("--max-ns", type=float, default=10_000)
    parser.add_argument("--bin-ns", type=float, default=10)
    parser.add_argument("--width-ns", type=float, default=50)
    parser.add_argument("--min-pmts", type=int, default=10)
    parser.add_argument("--min-separation-ns", type=float, default=100)
    parser.add_argument("--max-bursts", type=int, default=20)
    parser.add_argument("--hist-bin-ns", type=float, default=50)
    parser.add_argument("--tdc-time-mode", choices=("reference", "raw"), default="reference")
    parser.add_argument("--prompt-min-ns", type=float, default=1500)
    parser.add_argument("--prompt-max-ns", type=float, default=1900)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
