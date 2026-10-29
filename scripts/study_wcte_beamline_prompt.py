#!/usr/bin/env python3
"""Prompt-gated beam-monitor timing and multiplicity diagnostics for WCTE ROOT data.

Beamline TDC channels are corrected using the first hit on reference channel 31
for IDs <=31 and channel 46 for IDs >31. This is the production beam-monitor
convention; it does not align the beamline clock to WCTE PMT time. A fixed
per-detector prompt gate is found once from the main peak and then applied to
all readouts. WCTE PMT branches and WCTE PMT quality masks are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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


T0_CHANNELS = (0, 1, 2, 3)
DEFAULT_T1_CHANNELS = (4, 5)
T5_BARS = tuple((48 + bar, 56 + bar) for bar in range(8))
HD_CHANNELS = {**{32 + i: i for i in range(7)}, 16: 7,
               **{17 + i: i + 8 for i in range(7)}}
REQUIRED = ("beamline_pmt_tdc_ids", "beamline_pmt_tdc_times")


def main_peak(values: list[float], *, bin_ns: float = 2.0) -> tuple[float, float, int]:
    """Return modal-bin center, modal-bin count and number of finite entries."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan"), 0.0, 0
    lo, hi = np.percentile(data, [0.1, 99.9])
    if hi <= lo:
        return float(data[0]), float(len(data)), len(data)
    edges = np.arange(lo, hi + bin_ns, bin_ns)
    counts, edges = np.histogram(data, bins=edges)
    index = int(np.argmax(counts))
    return float((edges[index] + edges[index + 1]) / 2), float(counts[index]), len(data)


def robust_group_mean(channel_times: dict[int, float], channels: tuple[int, ...],
                      tolerance_ns: float = 20.0) -> float | None:
    """Production-style four-channel average, rejecting hits >20 ns from median."""
    if not all(channel in channel_times for channel in channels):
        return None
    values = np.asarray([channel_times[ch] for ch in channels], dtype=float)
    median = float(np.median(values))
    kept = values[np.abs(values - median) < tolerance_ns]
    return float(np.mean(kept)) if len(kept) else None


def first_hits(event_ids, event_times) -> tuple[dict[int, float], tuple[float | None, float | None]]:
    pairs = [(int(ch), float(t)) for ch, t in zip(ak.to_list(event_ids), ak.to_list(event_times))
             if t is not None and np.isfinite(float(t))]
    corrected, refs = correct_tdc_hits(pairs, mode="reference")
    by_channel: dict[int, float] = {}
    for hit in corrected:
        if hit.channel_id in REFERENCE_IDS or hit.corrected_ns is None:
            continue
        by_channel.setdefault(hit.channel_id, float(hit.corrected_ns))
    return by_channel, refs


def save_data(fig, path: Path, run_number: int | str | None = None):
    run_tag = f" | Run {run_number}" if run_number is not None else ""
    fig.text(0.99, 0.995, f"data{run_tag}", ha="right", va="top", fontsize=10,
             color="#7f1d1d", weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_peak_distributions(samples: dict[str, list[float]], gates: dict[str, dict],
                            out: Path, run_number: int | str | None = None,
                            t0_label: str = "T0 four-PMT average",
                            t1_label: str = "T1 average"):
    """Write separate, readable pages for T0/T1, T5 bars, and HD channels."""
    def draw(axis, label, values):
        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)]
        if len(data):
            axis.hist(data, bins=160, histtype="step", color="tab:blue")
        info = gates[label]
        if np.isfinite(info["center_ns"]):
            axis.axvspan(info["center_ns"] - info["half_width_ns"],
                         info["center_ns"] + info["half_width_ns"],
                         alpha=0.18, color="tab:orange", label="fixed ±40 ns gate")
            axis.axvline(info["center_ns"], color="tab:orange", linestyle="--",
                         label=f"main peak={info['center_ns']:.1f} ns")
            axis.legend(fontsize=8)
        axis.set_ylabel("Readouts / bin")
        axis.set_title(f"{label} (n={len(data):,})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, label in zip(axes, (t0_label, t1_label)):
        draw(axis, label, samples[label])
        axis.set_xlabel("Reference-corrected time (ns)")
    fig.suptitle("T0/T1 main prompt peaks and fixed gates")
    fig.tight_layout()
    save_data(fig, out.with_name("beamline_prompt_peaks_t0_t1.png"), run_number)

    for page, first_bar in enumerate((0, 4), start=1):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
        for bar, axis in zip(range(first_bar, first_bar + 4), axes.flat):
            for end in range(2):
                label = f"T5 bar {bar} end {end}"
                values = np.asarray(samples[label], dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    axis.hist(values, bins=120, histtype="step", label=f"end {end} (n={len(values):,})")
                center = gates[label]["center_ns"]
                if np.isfinite(center):
                    axis.axvspan(center - gates[label]["half_width_ns"],
                                 center + gates[label]["half_width_ns"], alpha=0.12)
                    axis.axvline(center, linestyle="--", linewidth=0.9)
            axis.set(title=f"T5 bar {bar}", xlabel="Reference-corrected time (ns)", ylabel="Hits / bin")
            axis.legend(fontsize=8)
        fig.suptitle(f"T5 prompt peaks, bars {first_bar}-{first_bar + 3}; per-end main peaks")
        fig.tight_layout()
        save_data(fig, out.with_name(f"beamline_prompt_peaks_t5_bars_{first_bar}-{first_bar + 3}.png"), run_number)

    for first_hd in range(0, 15, 4):
        last_hd = min(first_hd + 3, 14)
        labels = list(range(first_hd, last_hd + 1))
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
        for hd, axis in zip(labels, axes.flat):
            label = f"HD{hd}"
            draw(axis, label, samples[label])
            axis.set_xlabel("Reference-corrected time (ns)")
        for axis in axes.flat[len(labels):]:
            axis.set_visible(False)
        fig.suptitle(f"HD element prompt peaks, HD{first_hd}-HD{last_hd}")
        fig.tight_layout()
        save_data(fig, out.with_name(f"beamline_prompt_peaks_hd_{first_hd}-{last_hd}.png"), run_number)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0_channels = tuple(getattr(args, "t0_channels", T0_CHANNELS))
    t1_channels = tuple(getattr(args, "t1_channels", DEFAULT_T1_CHANNELS))
    with uproot.open(args.input_root) as root:
        tree = root[args.tree]
        missing = set(REQUIRED) - set(tree.keys())
        if missing:
            raise ValueError("Missing ROOT branches: " + ", ".join(sorted(missing)))
        available = set(tree.keys())
        has_quality = "vme_digi_issues_bitmask" in available
        optional = [name for name in ("vme_digi_issues_bitmask", "run_id")
                    if name in available]
        start = args.entry_start
        stop = min(tree.num_entries, start + args.max_events)
        arrays = tree.arrays([*REQUIRED, *optional], entry_start=start, entry_stop=stop,
                             library="ak")
        run_number = None
        if "run_id" in optional and stop > start:
            try:
                run_number = int(ak.to_list(arrays["run_id"])[0])
            except (TypeError, ValueError):
                run_number = str(ak.to_list(arrays["run_id"])[0])
        if run_number is None:
            match = re.search(r"(?:^|[^A-Za-z])R?(\d{4,})(?:[^0-9]|$)", args.input_root.name)
            run_number = int(match.group(1)) if match else None

    rows = []
    reference_good = 0
    missing_reference = {"31": 0, "46": 0}
    for i in range(stop - start):
        times, refs = first_hits(arrays[REQUIRED[0]][i], arrays[REQUIRED[1]][i])
        if refs[0] is None:
            missing_reference["31"] += 1
        if refs[1] is None:
            missing_reference["46"] += 1
        refs_good = refs[0] is not None and refs[1] is not None
        if refs_good:
            reference_good += 1
        # Beamline-only data quality: both reference channels must exist and
        # the beam-monitor digitiser issue flag must be clear when available.
        digi_good = True
        if has_quality:
            digi_good = int(arrays["vme_digi_issues_bitmask"][i]) == 0
        quality_good = refs_good and digi_good
        if not quality_good:
            continue
        t0 = robust_group_mean(times, t0_channels)
        t1 = robust_group_mean(times, t1_channels)
        t5_bars = []
        for bar, (left, right) in enumerate(T5_BARS):
            if left in times and right in times:
                end_delta = times[left] - times[right]
                mean = (times[left] + times[right]) / 2
                t5_bars.append((bar, mean, end_delta))
        hd_times = {HD_CHANNELS[ch]: times[ch] for ch in HD_CHANNELS if ch in times}
        rows.append({"entry": start + i, "times": times, "t0": t0, "t1": t1,
                     "t5_bars": t5_bars, "hd": hd_times})

    # Determine every fixed peak once from beamline-quality readouts. T5 gates
    # are per bar because cable/geometry offsets can differ between bars.
    t0_residuals = {ch: [] for ch in t0_channels}
    for row in rows:
        if row["t0"] is None:
            continue
        event_median = float(np.median([row["times"][ch] for ch in t0_channels]))
        for channel in t0_channels:
            t0_residuals[channel].append(row["times"][channel] - event_median)
    # This equalization is estimated from this same run solely as a diagnostic;
    # it is not an official calibration and is not applied to the TOF selection.
    t0_relative_offsets = {ch: float(np.median(values)) if values else 0.0
                           for ch, values in t0_residuals.items()}
    t0_comparison = {"reference_only": [], "reference_plus_data_derived_relative_offsets": []}
    t0_residuals_for_plot = {ch: [] for ch in t0_channels}
    for row in rows:
        if row["t0"] is None:
            continue
        vals = [row["times"][ch] for ch in t0_channels]
        median = float(np.median(vals))
        for channel in t0_channels:
            t0_residuals_for_plot[channel].append(row["times"][channel] - median)
        t0_comparison["reference_only"].append(row["t0"])
        t0_comparison["reference_plus_data_derived_relative_offsets"].append(
            float(np.mean([row["times"][ch] - t0_relative_offsets[ch] for ch in t0_channels])))
    t0_label = f"T0 average ({len(t0_channels)} channels)"
    t1_label = f"T1 average ({len(t1_channels)} channels)"
    samples = {t0_label: t0_comparison["reference_only"],
               t1_label: [r["t1"] for r in rows if r["t1"] is not None]}
    for bar in range(8):
        samples[f"T5 bar {bar} paired-end mean"] = [mean for r in rows for b, mean, _ in r["t5_bars"] if b == bar]
        for end, channel in enumerate(T5_BARS[bar]):
            samples[f"T5 bar {bar} end {end}"] = [r["times"][channel] for r in rows
                                                  if channel in r["times"]]
    for hd in range(15):
        samples[f"HD{hd}"] = [r["hd"][hd] for r in rows if hd in r["hd"]]
    gates = {}
    for label, values in samples.items():
        center, peak_count, n = main_peak(values, bin_ns=args.peak_bin_ns)
        gates[label] = {"center_ns": center, "half_width_ns": args.prompt_half_width_ns,
                        "peak_bin_count": int(peak_count), "entries_used": n}
    plot_peak_distributions(samples, gates, args.output_dir / "beamline_prompt_peaks.png",
                            run_number, t0_label, t1_label)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    baseline = np.asarray(t0_comparison["reference_only"], dtype=float)
    equalized = np.asarray(t0_comparison["reference_plus_data_derived_relative_offsets"], dtype=float)
    if len(baseline):
        edges = np.histogram_bin_edges(np.concatenate([baseline, equalized]), bins=100)
        axes[0].hist(baseline, bins=edges, histtype="step", label="reference subtraction only")
        axes[0].hist(equalized, bins=edges, histtype="step",
                     label="+ run-derived relative channel offsets")
        axes[0].legend(fontsize=8)
    axes[0].set(xlabel="T0 average time (ns)", ylabel="Readouts / bin",
                title="T0 timing: no per-channel offsets vs diagnostic equalization")
    for channel, values in t0_residuals_for_plot.items():
        if values:
            axes[1].hist(values, bins=100, histtype="step", label=f"T0 channel {channel}")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Channel time − same-event T0 median (ns)", ylabel="Hits / bin",
                title="T0 per-channel residuals before equalization")
    axes[1].legend(fontsize=8)
    fig.suptitle("Timing-correction comparison (run-derived offsets are diagnostic only)")
    fig.tight_layout()
    save_data(fig, args.output_dir / "t0_channel_correction_comparison.png", run_number)

    def in_gate(label, value):
        center = gates[label]["center_ns"]
        return value is not None and np.isfinite(center) and abs(value - center) <= args.prompt_half_width_ns

    multiplicities = np.zeros((8, 3), dtype=np.int64)
    end_differences = [[] for _ in range(8)]
    tofs = {"T0-T1": [], "T0-T5": [], "T0-HD0": [], "T0-HD5": [], "T0-HD8": []}
    ungated_t0_t1 = []
    gate_counts = {"t0_in_gate": 0, "t1_in_gate": 0, "both_in_gate": 0,
                   "both_groups_present_before_gates": 0}
    t5_bar_tofs = [[] for _ in range(8)]
    for row in rows:
        t0_ok = in_gate(t0_label, row["t0"])
        t1_ok = in_gate(t1_label, row["t1"])
        gate_counts["t0_in_gate"] += int(t0_ok)
        gate_counts["t1_in_gate"] += int(t1_ok)
        if row["t0"] is not None and row["t1"] is not None:
            gate_counts["both_groups_present_before_gates"] += 1
            ungated_t0_t1.append(row["t1"] - row["t0"])
        if t0_ok and t1_ok:
            gate_counts["both_in_gate"] += 1
            tofs["T0-T1"].append(row["t1"] - row["t0"])
        if t0_ok:
            for hd_id in (0, 5, 8):
                label = f"HD{hd_id}"
                if hd_id in row["hd"] and in_gate(label, row["hd"][hd_id]):
                    tofs[f"T0-HD{hd_id}"].append(row["hd"][hd_id] - row["t0"])
            for bar, mean, delta in row["t5_bars"]:
                label = f"T5 bar {bar} paired-end mean"
                if in_gate(label, mean):
                    t5_bar_tofs[bar].append(mean - row["t0"])
        for bar, (left, right) in enumerate(T5_BARS):
            left_gate = gates[f"T5 bar {bar} end 0"]["center_ns"]
            right_gate = gates[f"T5 bar {bar} end 1"]["center_ns"]
            left_ok = (left in row["times"] and np.isfinite(left_gate)
                       and abs(row["times"][left] - left_gate) <= args.prompt_half_width_ns)
            right_ok = (right in row["times"] and np.isfinite(right_gate)
                        and abs(row["times"][right] - right_gate) <= args.prompt_half_width_ns)
            multiplicities[bar, int(left_ok) + int(right_ok)] += 1
            if left_ok and right_ok:
                end_differences[bar].append(row["times"][left] - row["times"][right])
    tofs["T0-T5"] = [x for values in t5_bar_tofs for x in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.25
    x = np.arange(8)
    for mult, color, label in ((0, "tab:gray", "0 ends"), (1, "tab:orange", "1 end"), (2, "tab:blue", "2 ends")):
        axes[0].bar(x + (mult - 1) * width, multiplicities[:, mult], width=width,
                    color=color, label=label)
    axes[0].set(xticks=x, xlabel="T5 bar", ylabel="Readouts", title="Prompt-gated T5 end multiplicity")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.5)
    axes[0].legend()
    for bar, vals in enumerate(end_differences):
        if vals:
            axes[1].hist(vals, bins=80, range=(-args.end_diff_range_ns, args.end_diff_range_ns),
                         histtype="step", label=f"bar {bar}")
    axes[1].set(xlabel="T5 left − right end time (ns)", ylabel="Paired hits / bin",
                title="Prompt-gated T5 end-time differences")
    axes[1].set_xlim(-args.end_diff_range_ns, args.end_diff_range_ns)
    axes[1].legend(ncol=2, fontsize=8)
    save_data(fig, args.output_dir / "beamline_prompt_t5_diagnostics.png", run_number)

    # TOF panels are split into pages of no more than four plots.
    tof_items = list(tofs.items())
    for page_start in range(0, len(tof_items), 4):
        page_items = tof_items[page_start:page_start + 4]
        rows_n = 2
        cols_n = 2
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(12, 8))
        for axis, (label, vals) in zip(axes.flat, page_items):
            if vals:
                axis.hist(vals, bins=120, histtype="step", color="tab:blue")
            if label == "T0-T1" and ungated_t0_t1:
                common_edges = np.histogram_bin_edges(ungated_t0_t1, bins=120)
                axis.hist(ungated_t0_t1, bins=common_edges, histtype="step", color="tab:orange",
                          label=f"all T0/T1 pairs (n={len(ungated_t0_t1):,})")
                if vals:
                    axis.hist(vals, bins=common_edges, histtype="step", color="tab:blue",
                              label=f"both prompt gates (n={len(vals):,})")
                axis.legend(fontsize=8)
            axis.set(xlabel="Corrected detector-time difference (ns)", ylabel="Readouts / bin",
                     title=f"{label} (prompt-gated n={len(vals):,})")
            if label in ("T0-HD0", "T0-HD5", "T0-HD8"):
                hd_id = int(label.split("HD", 1)[1])
                channel = next(ch for ch, element in HD_CHANNELS.items() if element == hd_id)
                bank = ("different ref banks (T0=31, HD=46); uncalibrated"
                        if channel > 31 else "same ref 31; detector offsets remain")
                axis.set_title(f"{label}: {bank}\nprompt-gated n={len(vals):,}", fontsize=9)
        for axis in axes.flat[len(page_items):]:
            axis.set_visible(False)
        fig.suptitle("Beamline prompt-gated timing (no WCTE PMT quality selection)")
        fig.tight_layout()
        save_data(fig, args.output_dir / f"beamline_prompt_tof_page_{page_start // 4 + 1}.png", run_number)

    with (args.output_dir / "prompt_tof_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tof_name", "tof_ns"])
        for label, values in tofs.items():
            writer.writerows((label, value) for value in values)
    with (args.output_dir / "t5_prompt_multiplicity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bar", "end_multiplicity", "readouts"])
        for bar in range(8):
            for multiplicity in range(3):
                writer.writerow([bar, multiplicity, int(multiplicities[bar, multiplicity])])
    summary = {"input": str(args.input_root), "tree": args.tree,
               "run_number": run_number,
               "processed_entries": stop - start, "beamline_quality_readouts": len(rows),
               "readouts_with_both_reference_channels": reference_good,
               "missing_reference_counts": dict(missing_reference),
               "tdc_reference_correction": "first channel 31 for IDs <=31; first channel 46 for IDs >31",
               "prompt_gate_half_width_ns": args.prompt_half_width_ns,
               "prompt_peak_bin_width_ns": args.peak_bin_ns,
               "prompt_gate_centers": gates,
               "t0_channels": list(t0_channels), "t1_channels": list(t1_channels),
               "t1_channel_presence_counts": {str(ch): sum(ch in row["times"] for row in rows)
                                              for ch in (4, 5, 6, 7)},
               "t0_data_derived_relative_channel_offsets_ns": {
                   str(ch): offset for ch, offset in t0_relative_offsets.items()},
               "t0_time_comparison_widths_ns": {
                   mode: {"median": float(np.median(values)),
                          "q16": float(np.percentile(values, 16)),
                          "q84": float(np.percentile(values, 84)),
                          "q84_minus_q16": float(np.percentile(values, 84) - np.percentile(values, 16))}
                   for mode, values in t0_comparison.items() if values},
               "tof_counts": {name: len(values) for name, values in tofs.items()},
               "t0_t1_prompt_gate_counts": gate_counts,
               "t5_end_multiplicity": multiplicities.tolist(),
               "selection_note": "Beamline references required; vme_digi_issues_bitmask==0 when present. No WCTE PMT quality mask. T5 bars are not vetoed for multiplicity.",
               "t0_offset_comparison_note": "The reference-only series is the analysis default. Relative channel offsets are estimated from this run's T0 residuals solely for comparison and are not applied to TOFs or prompt gates."}
    summary["clock_interpretation_note"] = (
        "TOF histograms are corrected detector-time differences, not yet calibrated physical TOFs. "
        "T0 uses reference 31; HD0/HD5 use reference 46 while HD8 uses 31. "
        "Even same-reference channels retain detector/electronics offsets.")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {stop-start:,} readouts; {len(rows):,} passed beamline reference/digitiser quality")
    print(f"Prompt gates: ±{args.prompt_half_width_ns:g} ns around fixed main-peak centers")
    print("TOF counts: " + ", ".join(f"{name}={len(values):,}" for name, values in tofs.items()))
    print(f"Results: {args.output_dir}")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("input_root", type=Path)
    result.add_argument("--tree", default="WCTEReadoutWindows")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--entry-start", type=int, default=0)
    result.add_argument("--max-events", type=int, default=1_000_000)
    result.add_argument("--t0-channels", type=int, nargs=4, default=T0_CHANNELS,
                        help="Four T0 TDC channels used for the event average")
    result.add_argument("--t1-channels", type=int, nargs="+", default=DEFAULT_T1_CHANNELS,
                        help="T1 TDC channels to average (default 4 5; choose 4 5 6 7 if four are present)")
    result.add_argument("--prompt-half-width-ns", type=float, default=40.0)
    result.add_argument("--peak-bin-ns", type=float, default=2.0)
    result.add_argument("--end-diff-range-ns", type=float, default=20.0)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
