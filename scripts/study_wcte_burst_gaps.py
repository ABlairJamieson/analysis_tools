#!/usr/bin/env python3
"""Histogram time gaps between PMT hit bursts over many WCTE ROOT windows.

This is an exploratory within-window timing study, not a particle or Michel tag.
The beamline TDC is used only for window selection and a T0-multiplicity split;
no offset between its clock and the WCTE PMT clock is assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

try:
    from scripts.beamline_timing import REFERENCE_IDS, correct_tdc_hits
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from beamline_timing import REFERENCE_IDS, correct_tdc_hits


T0_IDS = {0, 1, 2, 3}
T2_ID = 8
HC2_ID = 11
HD_IDS = {**{32 + i: i for i in range(7)}, 16: 7,
          **{17 + i: i + 8 for i in range(7)}}
REQUIRED = ("run_id", "event_number", "readout_number", "window_data_quality_mask",
            "hit_pmt_calibrated_times", "hit_pmt_charges", "hit_pmt_readout_mask",
            "hit_mpmt_slot_ids", "hit_pmt_position_ids",
            "beamline_pmt_tdc_ids", "beamline_pmt_tdc_times")


@dataclass(frozen=True)
class Burst:
    center_ns: float
    start_ns: float
    end_ns: float
    n_hits: int
    n_pmts: int
    charge: float


def find_bursts(times: np.ndarray, charges: np.ndarray, pmts: np.ndarray,
                *, min_ns: float, max_ns: float, bin_ns: float, width_ns: float,
                min_pmts: int, min_separation_ns: float, max_bursts: int) -> list[Burst]:
    """Find dense 50-ns-scale windows, suppressing duplicate nearby maxima."""
    if len(times) == 0:
        return []
    edges = np.arange(min_ns, max_ns + bin_ns, bin_ns)
    counts, _ = np.histogram(times, bins=edges)
    width_bins = max(1, int(round(width_ns / bin_ns)))
    smoothed = np.convolve(counts, np.ones(width_bins, dtype=int), mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    candidates = np.flatnonzero((smoothed >= min_pmts)
                                & (smoothed >= np.r_[0, smoothed[:-1]])
                                & (smoothed > np.r_[smoothed[1:], 0]))
    chosen: list[Burst] = []
    for index in sorted(candidates, key=lambda i: smoothed[i], reverse=True):
        center = float(centers[index])
        if any(abs(center - burst.center_ns) < min_separation_ns for burst in chosen):
            continue
        start, end = center - width_ns / 2, center + width_ns / 2
        inside = (times >= start) & (times < end)
        if len(np.unique(pmts[inside])) < min_pmts:
            continue
        chosen.append(Burst(center, start, end, int(inside.sum()),
                            int(len(np.unique(pmts[inside]))), float(charges[inside].sum())))
        if len(chosen) >= max_bursts:
            break
    return sorted(chosen, key=lambda burst: burst.center_ns)


def t0_group_centers(t0_times: list[float], separation_ns: float = 50) -> list[float]:
    """Robust center of each T0 time group; four PMTs do not mean four bunches."""
    if not t0_times:
        return []
    ordered = np.sort(t0_times)
    split_at = np.flatnonzero(np.diff(ordered) > separation_ns) + 1
    return [float(np.median(group)) for group in np.split(ordered, split_at)]


def corrected_beamline_hits(event, *, tdc_time_mode: str = "reference"
                            ) -> tuple[list[tuple[int, float]], tuple[float | None, float | None]]:
    ids = ak.to_list(event["beamline_pmt_tdc_ids"])
    times = ak.to_list(event["beamline_pmt_tdc_times"])
    if len(ids) != len(times):
        raise ValueError("Beamline TDC ID/time lengths differ")
    raw_hits = [(int(cid), float(time)) for cid, time in zip(ids, times)
                if time is not None and np.isfinite(float(time))]
    corrected, refs = correct_tdc_hits(raw_hits, mode=tdc_time_mode)
    hits = [(hit.channel_id, float(hit.corrected_ns)) for hit in corrected
            if hit.corrected_ns is not None and hit.channel_id not in REFERENCE_IDS]
    return hits, refs


def beamline_status(event, *, tdc_time_mode: str = "reference") -> tuple[bool, bool, list[float], tuple]:
    hits, refs = corrected_beamline_hits(event, tdc_time_mode=tdc_time_mode)
    channels = {cid for cid, _ in hits}
    t0 = [time for cid, time in hits if cid in T0_IDS]
    beam_ok = bool(channels & T0_IDS) and T2_ID in channels and HC2_ID not in channels
    tagged = False
    if beam_ok:
        for cid, hd_time in hits:
            if cid not in HD_IDS:
                continue
            hd = HD_IDS[cid]
            lo, hi = (-50, 100) if hd in (0, 1, 2, 3, 8, 9, 10, 11) else (80, 200)
            if any(lo < hd_time - t0_time < hi for t0_time in t0):
                tagged = True
                break
    return beam_ok, tagged, t0_group_centers(t0), refs


def later_hodoscope_match_counts(event, t0_centers: list[float], *,
                                tdc_time_mode: str = "reference") -> tuple[int, int]:
    """Count HD hits and later T0 groups with a compatible HD hit.

    Compatibility uses the same channel-dependent T0-relative timing windows
    as ``beamline_status``. The first T0 group is treated as the triggered
    bunch; later groups with matching HD hits flag additional charged beam
    activity in that readout. This is not a gamma tag.
    """
    if len(t0_centers) < 2:
        return 0, 0
    hits, _ = corrected_beamline_hits(event, tdc_time_mode=tdc_time_mode)
    hits = [(cid, time) for cid, time in hits if cid in HD_IDS]
    matched_hit_indices: set[int] = set()
    matched_group_indices: set[int] = set()
    for group_index, t0_time in enumerate(t0_centers[1:], start=1):
        for hit_index, (channel_id, hd_time) in enumerate(hits):
            hd = HD_IDS[channel_id]
            lo, hi = (-50, 100) if hd in (0, 1, 2, 3, 8, 9, 10, 11) else (80, 200)
            if lo < hd_time - t0_time < hi:
                matched_hit_indices.add(hit_index)
                matched_group_indices.add(group_index)
    return len(matched_hit_indices), len(matched_group_indices)


def _hits(event) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(ak.to_numpy(event["hit_pmt_readout_mask"])) == 0
    times = np.asarray(ak.to_numpy(event["hit_pmt_calibrated_times"]), dtype=float)[mask]
    charges = np.asarray(ak.to_numpy(event["hit_pmt_charges"]), dtype=float)[mask]
    slots = np.asarray(ak.to_numpy(event["hit_mpmt_slot_ids"]), dtype=np.int64)[mask]
    positions = np.asarray(ak.to_numpy(event["hit_pmt_position_ids"]), dtype=np.int64)[mask]
    valid = np.isfinite(times) & np.isfinite(charges)
    return times[valid], charges[valid], (100 * slots + positions)[valid]


def plot_histograms(path: Path, histograms: dict[tuple[str, str], np.ndarray],
                    edges: np.ndarray, selected: int, multi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    groups = (("one_t0_group", "1 T0 time group"),
              ("multiple_t0_groups", "2+ T0 time groups"),
              ("no_t0", "No T0 hit"))
    colors = {"one_t0_group": "#1f77b4", "multiple_t0_groups": "#d95f02", "no_t0": "#777777"}
    for row, (kind, title) in enumerate((("consecutive", "Consecutive bursts"),
                                          ("later_from_strongest", "Later than strongest burst"))):
        for col, log_scale in enumerate((False, True)):
            axis = axes[row, col]
            for group, label in groups:
                values = histograms[(kind, group)]
                if values.sum():
                    axis.stairs(values, edges, label=f"{label} (gaps={values.sum():,})",
                                color=colors[group], linewidth=1.5)
            if log_scale:
                axis.set_yscale("log")
            else:
                axis.set_xlim(0, min(float(edges[-1]), 1500))
            axis.set(xlabel="Time gap between burst centers (ns)", ylabel="Gaps / bin",
                     title=title + (" — full range, log count" if log_scale else " — 0–1500 ns"))
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    fig.suptitle(f"Within-window PMT burst gaps: {multi:,}/{selected:,} selected windows have 2+ bursts")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_t0_comparison(path: Path, t0_hist: np.ndarray, pmt_hist: np.ndarray,
                       edges: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].stairs(t0_hist, edges, linewidth=1.7, color="tab:purple")
    axes[0].set(xlabel="Gap between T0 time-group centers (ns)", ylabel="T0 group gaps / bin",
                title=f"Beamline T0 groups, 0–1500 ns (n={t0_hist.sum():,} gaps)")
    axes[0].set_xlim(0, min(float(edges[-1]), 1500))
    for values, label, color in ((t0_hist, "T0 group gaps", "tab:purple"),
                                 (pmt_hist, "PMT consecutive-burst gaps", "tab:blue")):
        if values.sum():
            axes[1].stairs(values / values.sum(), edges, linewidth=1.5,
                            label=f"{label} (n={values.sum():,})", color=color)
    axes[1].set(xlabel="Within-window gap (ns)", ylabel="Fraction of gaps / bin",
                title="Shape comparison only — no clock alignment")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    tdc_mode = getattr(args, "tdc_time_mode", "reference")
    prompt_min = getattr(args, "prompt_min_ns", 1500)
    prompt_max = getattr(args, "prompt_max_ns", 1900)
    if prompt_max <= prompt_min:
        raise ValueError("Prompt maximum must exceed prompt minimum")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(0, args.max_ns - args.min_ns + args.hist_bin_ns, args.hist_bin_ns)
    histograms = {(kind, group): np.zeros(len(edges) - 1, dtype=np.int64)
                  for kind in ("consecutive", "later_from_strongest")
                  for group in ("one_t0_group", "multiple_t0_groups", "no_t0")}
    t0_hist = np.zeros(len(edges) - 1, dtype=np.int64)
    counts: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    alignment_candidates: list[float] = []
    with uproot.open(args.input_root) as root:
        tree = root["WCTEReadoutWindows"]
        missing = set(REQUIRED) - set(tree.keys())
        if missing:
            raise ValueError("Missing ROOT branches: " + ", ".join(sorted(missing)))
        stop = min(tree.num_entries, args.entry_start + args.scan_windows)
        with (args.output_dir / "bursts.csv").open("w", newline="", encoding="utf-8") as burst_file, \
             (args.output_dir / "gaps.csv").open("w", newline="", encoding="utf-8") as gap_file, \
             (args.output_dir / "t0_group_gaps.csv").open("w", newline="", encoding="utf-8") as t0_file, \
             (args.output_dir / "alignment_candidates.csv").open("w", newline="", encoding="utf-8") as align_file:
            burst_writer = csv.DictWriter(burst_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "n_t0_groups",
                "n_late_hd_hits", "n_late_hd_groups", "burst_index", "strongest",
                *Burst.__dataclass_fields__))
            gap_writer = csv.DictWriter(gap_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "n_t0_groups",
                "kind", "from_burst", "to_burst", "gap_ns"))
            t0_writer = csv.DictWriter(t0_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number",
                "from_t0_group", "to_t0_group", "from_time_ns", "to_time_ns", "gap_ns"))
            align_writer = csv.DictWriter(align_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "t0_ns",
                "pmt_burst_ns", "pmt_minus_t0_ns", "n_pmt_bursts"))
            burst_writer.writeheader()
            gap_writer.writeheader()
            t0_writer.writeheader()
            align_writer.writeheader()
            for start in range(args.entry_start, stop, args.batch_windows):
                batch = tree.arrays(REQUIRED, entry_start=start,
                                    entry_stop=min(start + args.batch_windows, stop), library="ak")
                for offset, event in enumerate(batch):
                    counts["raw_windows"] += 1
                    if int(event["window_data_quality_mask"]) != 0:
                        continue
                    counts["quality_good_windows"] += 1
                    beam_ok, tagged, t0_centers, refs = beamline_status(event, tdc_time_mode=tdc_mode)
                    if refs[0] is None:
                        counts["windows_missing_tdc_ref31"] += 1
                    if refs[1] is None:
                        counts["windows_missing_tdc_ref46"] += 1
                    if args.selection == "beam" and not beam_ok:
                        continue
                    if args.selection == "tagged" and not tagged:
                        continue
                    counts["selected_windows"] += 1
                    n_t0 = len(t0_centers)
                    n_late_hd_hits, n_late_hd_groups = later_hodoscope_match_counts(
                        event, t0_centers, tdc_time_mode=tdc_mode)
                    entry = start + offset
                    identity = {"root_entry": entry, "run_id": int(event["run_id"]),
                                "event_number": int(event["event_number"]),
                                "readout_number": int(event["readout_number"]),
                                "n_t0_groups": n_t0}
                    for i in range(1, n_t0):
                        gap = t0_centers[i] - t0_centers[i - 1]
                        t0_writer.writerow({**{key: identity[key] for key in (
                            "root_entry", "run_id", "event_number", "readout_number")},
                            "from_t0_group": i - 1, "to_t0_group": i,
                            "from_time_ns": t0_centers[i - 1], "to_time_ns": t0_centers[i],
                            "gap_ns": gap})
                        t0_hist += np.histogram([gap], bins=edges)[0]
                    times, charges, pmts = _hits(event)
                    bursts = find_bursts(times, charges, pmts, min_ns=args.min_ns,
                                         max_ns=args.max_ns, bin_ns=args.bin_ns,
                                         width_ns=args.width_ns, min_pmts=args.min_pmts,
                                         min_separation_ns=args.min_separation_ns,
                                         max_bursts=args.max_bursts)
                    multiplicities[len(bursts)] += 1
                    if not bursts:
                        continue
                    counts["windows_with_bursts"] += 1
                    if len(bursts) > 1:
                        counts["windows_with_multiple_bursts"] += 1
                    prompt_bursts = [burst for burst in bursts
                                     if prompt_min <= burst.center_ns < prompt_max]
                    if tdc_mode == "reference" and n_t0 == 1 and len(prompt_bursts) == 1:
                        candidate = prompt_bursts[0].center_ns - t0_centers[0]
                        alignment_candidates.append(candidate)
                        align_writer.writerow({**{key: identity[key] for key in (
                            "root_entry", "run_id", "event_number", "readout_number")},
                            "t0_ns": t0_centers[0], "pmt_burst_ns": prompt_bursts[0].center_ns,
                            "pmt_minus_t0_ns": candidate, "n_pmt_bursts": len(bursts)})
                    strongest = max(range(len(bursts)), key=lambda i: bursts[i].n_pmts)
                    group = "no_t0" if n_t0 == 0 else "one_t0_group" if n_t0 == 1 else "multiple_t0_groups"
                    for i, burst in enumerate(bursts):
                        burst_writer.writerow({**identity, "burst_index": i,
                                               "n_late_hd_hits": n_late_hd_hits,
                                               "n_late_hd_groups": n_late_hd_groups,
                                               "strongest": int(i == strongest), **asdict(burst)})
                    for i in range(1, len(bursts)):
                        gap = bursts[i].center_ns - bursts[i - 1].center_ns
                        gap_writer.writerow({**identity, "kind": "consecutive",
                                             "from_burst": i - 1, "to_burst": i, "gap_ns": gap})
                        histograms[("consecutive", group)] += np.histogram([gap], bins=edges)[0]
                    for i in range(strongest + 1, len(bursts)):
                        gap = bursts[i].center_ns - bursts[strongest].center_ns
                        gap_writer.writerow({**identity, "kind": "later_from_strongest",
                                             "from_burst": strongest, "to_burst": i, "gap_ns": gap})
                        histograms[("later_from_strongest", group)] += np.histogram([gap], bins=edges)[0]
            summary = {
                "counts": dict(counts),
                "burst_multiplicity_per_selected_window": {str(k): v for k, v in sorted(multiplicities.items())},
                "selection": args.selection,
                "tdc_time_mode": tdc_mode,
                "alignment_candidates": {
                    "count": len(alignment_candidates),
                    "prompt_pmt_range_ns": [prompt_min, prompt_max],
                    "median_pmt_minus_t0_ns": (float(np.median(alignment_candidates))
                                                if alignment_candidates else None),
                    "mad_ns": (float(np.median(np.abs(np.asarray(alignment_candidates)
                                                     - np.median(alignment_candidates))))
                               if alignment_candidates else None),
                    "note": "Empirical prompt-light offset includes particle/light propagation; inspect the distribution before applying it.",
                },
                "settings": {key: getattr(args, key) for key in (
                    "entry_start", "scan_windows", "min_ns", "max_ns", "bin_ns", "width_ns",
                    "min_pmts", "min_separation_ns", "max_bursts", "hist_bin_ns")},
                "interpretation": "Bursts are time-density candidates, not particle IDs. Gaps from one window are correlated; TDC references correct only within the beamline clock, not against WCTE PMT time.",
                "t0_group_gap_count": int(t0_hist.sum()),
            }
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_histograms(args.output_dir / "burst_gap_histograms.png", histograms, edges,
                    counts["selected_windows"], counts["windows_with_multiple_bursts"])
    pmt_consecutive = sum((histograms[("consecutive", group)] for group in (
        "one_t0_group", "multiple_t0_groups", "no_t0")), np.zeros(len(edges) - 1, dtype=np.int64))
    plot_t0_comparison(args.output_dir / "t0_vs_pmt_gap_histograms.png",
                       t0_hist, pmt_consecutive, edges)
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
    print(f"Scanned {counts['raw_windows']:,} ROOT windows; selected {counts['selected_windows']:,}; "
          f"{counts['windows_with_multiple_bursts']:,} have 2+ PMT bursts")
    if tdc_mode == "reference":
        print(f"Missing TDC references among quality-good windows: "
              f"31={counts['windows_missing_tdc_ref31']:,}, "
              f"46={counts['windows_missing_tdc_ref46']:,}")
        print(f"Empirical alignment candidates: {len(alignment_candidates):,}; "
              "inspect alignment_candidates.png before using an offset")
    print(f"Results: {args.output_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", choices=("all", "beam", "tagged"), default="tagged",
                        help="Tagged is a window-level T0+T2, no HC2, in-time HD selection")
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--scan-windows", type=int, default=100_000)
    parser.add_argument("--batch-windows", type=int, default=500)
    parser.add_argument("--min-ns", type=float, default=0)
    parser.add_argument("--max-ns", type=float, default=10_000)
    parser.add_argument("--bin-ns", type=float, default=10)
    parser.add_argument("--width-ns", type=float, default=50)
    parser.add_argument("--min-pmts", type=int, default=10)
    parser.add_argument("--min-separation-ns", type=float, default=100)
    parser.add_argument("--max-bursts", type=int, default=20)
    parser.add_argument("--hist-bin-ns", type=float, default=50)
    parser.add_argument("--tdc-time-mode", choices=("reference", "raw"), default="reference",
                        help="Subtract TDC references 31/46; raw is for comparison only")
    parser.add_argument("--prompt-min-ns", type=float, default=1500)
    parser.add_argument("--prompt-max-ns", type=float, default=1900)
    args = parser.parse_args()
    if (args.entry_start < 0 or args.scan_windows < 1 or args.batch_windows < 1
            or args.max_ns <= args.min_ns or args.min_ns < 0 or args.bin_ns <= 0
            or args.width_ns <= 0 or args.min_pmts < 1 or args.min_separation_ns <= 0
            or args.max_bursts < 1 or args.hist_bin_ns <= 0):
        parser.error("Require nonnegative entry/time starts and positive window, bin, threshold and scan settings")
    if args.prompt_max_ns <= args.prompt_min_ns:
        parser.error("Prompt maximum must exceed prompt minimum")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
