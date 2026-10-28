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
T1_CHANNELS = (4, 5, 6, 7)
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


def save_data(fig, path: Path):
    fig.text(0.99, 0.995, "data", ha="right", va="top", fontsize=10,
             color="#7f1d1d", weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_peak_distributions(samples: dict[str, list[float]], gates: dict[str, dict], out: Path):
    labels = list(samples)
    fig, axes = plt.subplots(len(labels), 1, figsize=(11, max(3, 2.4 * len(labels))))
    if len(labels) == 1:
        axes = [axes]
    for axis, label in zip(axes, labels):
        data = np.asarray(samples[label], dtype=float)
        data = data[np.isfinite(data)]
        if len(data):
            axis.hist(data, bins=160, histtype="step", color="tab:blue")
        info = gates[label]
        if np.isfinite(info["center_ns"]):
            axis.axvspan(info["center_ns"] - info["half_width_ns"],
                         info["center_ns"] + info["half_width_ns"],
                         alpha=0.18, color="tab:orange", label="fixed prompt gate")
            axis.axvline(info["center_ns"], color="tab:orange", linestyle="--",
                         label=f"peak={info['center_ns']:.1f} ns")
            axis.legend()
        axis.set_ylabel("Readouts / bin")
        axis.set_title(f"{label}: n={len(data):,}")
    axes[-1].set_xlabel("Reference-corrected beamline time (ns)")
    fig.suptitle("Beamline prompt peaks and fixed ±40 ns gates")
    fig.tight_layout()
    save_data(fig, out)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with uproot.open(args.input_root) as root:
        tree = root[args.tree]
        missing = set(REQUIRED) - set(tree.keys())
        if missing:
            raise ValueError("Missing ROOT branches: " + ", ".join(sorted(missing)))
        available = set(tree.keys())
        has_quality = "vme_digi_issues_bitmask" in available
        optional = [name for name in ("vme_digi_issues_bitmask",)
                    if name in available]
        start = args.entry_start
        stop = min(tree.num_entries, start + args.max_events)
        arrays = tree.arrays([*REQUIRED, *optional], entry_start=start, entry_stop=stop,
                             library="ak")

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
        t0 = robust_group_mean(times, T0_CHANNELS)
        t1 = robust_group_mean(times, T1_CHANNELS)
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
    samples = {"T0 four-PMT average": [r["t0"] for r in rows if r["t0"] is not None],
               "T1 four-PMT average": [r["t1"] for r in rows if r["t1"] is not None]}
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
    plot_peak_distributions(samples, gates, args.output_dir / "beamline_prompt_peaks.png")

    def in_gate(label, value):
        center = gates[label]["center_ns"]
        return value is not None and np.isfinite(center) and abs(value - center) <= args.prompt_half_width_ns

    multiplicities = np.zeros((8, 3), dtype=np.int64)
    end_differences = [[] for _ in range(8)]
    tofs = {"T0-T1": [], "T0-T5": [], "T0-HD0": [], "T0-HD8": []}
    t5_bar_tofs = [[] for _ in range(8)]
    for row in rows:
        t0_ok = in_gate("T0 four-PMT average", row["t0"])
        t1_ok = in_gate("T1 four-PMT average", row["t1"])
        if t0_ok and t1_ok:
            tofs["T0-T1"].append(row["t1"] - row["t0"])
        if t0_ok:
            for hd_id in (0, 8):
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

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    width = 0.25
    x = np.arange(8)
    for mult, color, label in ((0, "tab:gray", "0 ends"), (1, "tab:orange", "1 end"), (2, "tab:blue", "2 ends")):
        axes[0, 0].bar(x + (mult - 1) * width, multiplicities[:, mult], width=width,
                       color=color, label=label)
    axes[0, 0].set(xticks=x, xlabel="T5 bar", ylabel="Readouts", title="Prompt-gated T5 end multiplicity")
    axes[0, 0].legend()
    for bar, vals in enumerate(end_differences):
        if vals:
            axes[0, 1].hist(vals, bins=80, range=(-args.end_diff_range_ns, args.end_diff_range_ns),
                            histtype="step", label=f"bar {bar}")
    axes[0, 1].set(xlabel="T5 left − right end time (ns)", ylabel="Paired hits / bin",
                   title="Prompt-gated T5 end-time differences")
    axes[0, 1].legend(ncol=2, fontsize=8)
    tof_axes = (axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])
    # Draw each TOF on its own scale; the detector separations and offsets differ.
    for axis, label in zip(tof_axes, tofs):
        vals = tofs[label]
        if vals:
            axis.hist(vals, bins=120, histtype="step", color="tab:blue")
        axis.set(xlabel="Reference-corrected TOF (ns)", ylabel="Readouts / bin",
                 title=f"{label} (n={len(vals):,})")
    fig.suptitle("Beamline prompt-gated diagnostics — no WCTE PMT quality selection")
    fig.tight_layout()
    save_data(fig, args.output_dir / "beamline_prompt_tof_and_t5.png")

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
               "processed_entries": stop - start, "beamline_quality_readouts": len(rows),
               "readouts_with_both_reference_channels": reference_good,
               "missing_reference_counts": dict(missing_reference),
               "tdc_reference_correction": "first channel 31 for IDs <=31; first channel 46 for IDs >31",
               "prompt_gate_half_width_ns": args.prompt_half_width_ns,
               "prompt_peak_bin_width_ns": args.peak_bin_ns,
               "prompt_gate_centers": gates,
               "tof_counts": {name: len(values) for name, values in tofs.items()},
               "t5_end_multiplicity": multiplicities.tolist(),
               "selection_note": "Beamline references required; vme_digi_issues_bitmask==0 when present. No WCTE PMT quality mask. T5 bars are not vetoed for multiplicity."}
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
    result.add_argument("--prompt-half-width-ns", type=float, default=40.0)
    result.add_argument("--peak-bin-ns", type=float, default=2.0)
    result.add_argument("--end-diff-range-ns", type=float, default=40.0)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
