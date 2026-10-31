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
T2_CHANNELS = (8,)
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


def first_hits(event_ids, event_times) -> tuple[
        dict[int, float], dict[int, float], tuple[float | None, float | None]]:
    pairs = [(int(ch), float(t)) for ch, t in zip(ak.to_list(event_ids), ak.to_list(event_times))
             if t is not None and np.isfinite(float(t))]
    corrected, refs = correct_tdc_hits(pairs, mode="reference")
    raw_by_channel: dict[int, float] = {}
    by_channel: dict[int, float] = {}
    for channel, raw_time in pairs:
        if channel not in REFERENCE_IDS:
            raw_by_channel.setdefault(channel, raw_time)
    for hit in corrected:
        if hit.channel_id in REFERENCE_IDS or hit.corrected_ns is None:
            continue
        by_channel.setdefault(hit.channel_id, float(hit.corrected_ns))
    return by_channel, raw_by_channel, refs


def save_data(fig, path: Path, run_number: int | str | None = None):
    run_tag = f" | Run {run_number}" if run_number is not None else ""
    fig.text(0.99, 0.995, f"data{run_tag}", ha="right", va="top", fontsize=10,
             color="#7f1d1d", weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_peak_distributions(samples: dict[str, list[float]], gates: dict[str, dict],
                            out: Path, run_number: int | str | None = None,
                            t0_label: str = "T0 four-PMT average",
                            t2_label: str = "T2"):
    """Write separate, readable pages for T0/T2, T5 bars, and HD channels."""
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
    for axis, label in zip(axes, (t0_label, t2_label)):
        draw(axis, label, samples[label])
        axis.set_xlabel("Reference-corrected time (ns)")
    fig.suptitle("T0/T2 main prompt peaks and fixed gates")
    fig.tight_layout()
    save_data(fig, out.with_name("beamline_prompt_peaks_t0_t2.png"), run_number)

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
    t2_channels = tuple(getattr(args, "t2_channels", T2_CHANNELS))
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
        times, raw_times, refs = first_hits(arrays[REQUIRED[0]][i], arrays[REQUIRED[1]][i])
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
        t0_raw = robust_group_mean(raw_times, t0_channels)
        t2 = robust_group_mean(times, t2_channels)
        t5_bars = []
        for bar, (left, right) in enumerate(T5_BARS):
            if left in times and right in times:
                end_delta = times[left] - times[right]
                mean = (times[left] + times[right]) / 2
                t5_bars.append((bar, mean, end_delta))
        hd_times = {HD_CHANNELS[ch]: times[ch] for ch in HD_CHANNELS if ch in times}
        rows.append({"entry": start + i, "times": times, "raw_times": raw_times,
                     "reference31_raw": refs[0], "t0": t0, "t0_raw": t0_raw, "t2": t2,
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
    t0_residuals_after_equalization = {ch: [] for ch in t0_channels}
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
        equalized_times = {ch: row["times"][ch] - t0_relative_offsets[ch]
                           for ch in t0_channels}
        equalized_median = float(np.median(list(equalized_times.values())))
        for channel in t0_channels:
            t0_residuals_after_equalization[channel].append(
                equalized_times[channel] - equalized_median)
    t0_label = f"T0 average ({len(t0_channels)} channels)"
    t2_label = f"T2 average ({len(t2_channels)} channels)"
    samples = {t0_label: t0_comparison["reference_only"],
               t2_label: [r["t2"] for r in rows if r["t2"] is not None]}
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
                            run_number, t0_label, t2_label)

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
            axes[1].hist(values, bins=100,
                         range=(-args.t0_residual_range_ns, args.t0_residual_range_ns),
                         histtype="step", label=f"T0 channel {channel}")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlim=(-args.t0_residual_range_ns, args.t0_residual_range_ns),
                xlabel="Channel time − same-event T0 median (ns)", ylabel="Hits / bin",
                title=f"Before equalization; common event time removed (±{args.t0_residual_range_ns:g} ns)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Absolute T0 average retains event-to-event jitter; right panel removes it")
    fig.tight_layout()
    save_data(fig, args.output_dir / "t0_channel_correction_comparison.png", run_number)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for channel in t0_channels:
        before = t0_residuals_for_plot[channel]
        after = t0_residuals_after_equalization[channel]
        plot_range = (-args.t0_residual_range_ns, args.t0_residual_range_ns)
        if before:
            axes[0].hist(before, bins=100, range=plot_range, histtype="step",
                         label=f"T0 channel {channel}")
        if after:
            axes[1].hist(after, bins=100, range=plot_range, histtype="step",
                         label=f"T0 channel {channel}")
    for axis, title in zip(axes, ("Before relative-offset correction",
                                  "After relative-offset correction")):
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(xlim=(-args.t0_residual_range_ns, args.t0_residual_range_ns),
                 xlabel="Channel time − same-event T0 median (ns)",
                 ylabel="Readouts / bin", title=title)
        axis.legend(fontsize=8)
    fig.suptitle("Relative channel alignment (same-event timing jitter removed in both panels)")
    fig.tight_layout()
    save_data(fig, args.output_dir / "t0_channel_residuals_before_after_equalization.png", run_number)

    # Show every T0 channel against the raw channel-31 reference separately.
    raw_t0_by_channel = {}
    corrected_t0_by_channel = {}
    direct_corrected_t0_by_channel = {}
    subtraction_checks = {}
    for channel in t0_channels:
        diagnostic_rows = [r for r in rows if channel in r["raw_times"]
                           and channel in r["times"] and r["reference31_raw"] is not None]
        raw = np.asarray([r["raw_times"][channel] for r in diagnostic_rows], dtype=float)
        ref = np.asarray([r["reference31_raw"] for r in diagnostic_rows], dtype=float)
        corrected = np.asarray([r["times"][channel] for r in diagnostic_rows], dtype=float)
        direct = raw - ref
        raw_t0_by_channel[channel] = raw
        corrected_t0_by_channel[channel] = corrected
        direct_corrected_t0_by_channel[channel] = direct
        subtraction_checks[str(channel)] = {
            "readouts": len(diagnostic_rows),
            "raw_channel_median_ns": float(np.median(raw)) if len(raw) else None,
            "raw_channel31_median_ns": float(np.median(ref)) if len(ref) else None,
            "corrected_channel_median_ns": float(np.median(corrected)) if len(corrected) else None,
            "direct_raw_minus_channel31_median_ns": float(np.median(direct)) if len(direct) else None,
            "max_abs_difference_current_vs_direct_ns": (
                float(np.max(np.abs(corrected - direct))) if len(direct) else None),
        }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    raw_reference = np.asarray([r["reference31_raw"] for r in rows
                                if r["reference31_raw"] is not None], dtype=float)
    raw_all = [values for values in raw_t0_by_channel.values() if len(values)]
    if len(raw_reference) and raw_all:
        raw_edges = np.histogram_bin_edges(np.concatenate([raw_reference, *raw_all]), bins=120)
        for channel, values in raw_t0_by_channel.items():
            if len(values):
                axes[0].hist(values, bins=raw_edges, histtype="step", linewidth=1.2,
                             label=f"raw T0 channel {channel} (n={len(values):,})")
        axes[0].hist(raw_reference, bins=raw_edges, histtype="step", linewidth=1.5,
                     color="black", label=f"raw channel-31 reference (n={len(raw_reference):,})")
        axes[0].legend(fontsize=8)
    axes[0].set(xlabel="Unsubtracted TDC time (ns; common beamline clock)",
                ylabel="Readouts / bin", title="Each raw T0 channel and raw reference")
    corrected_all = [values for values in corrected_t0_by_channel.values() if len(values)]
    if corrected_all:
        zoom_values = np.concatenate([*corrected_all, *direct_corrected_t0_by_channel.values()])
        zoom_lo, zoom_hi = np.percentile(zoom_values, [0.2, 99.8])
        padding = max((zoom_hi - zoom_lo) * 0.08, 1.0)
        zoom_limits = (zoom_lo - padding, zoom_hi + padding)
        # Build the bins from the visible window; using the full outlier range
        # here leaves only a few very wide bins after the x-axis is zoomed.
        corrected_edges = np.linspace(zoom_limits[0], zoom_limits[1], 121)
        for channel, corrected in corrected_t0_by_channel.items():
            if len(corrected):
                direct = direct_corrected_t0_by_channel[channel]
                axes[1].hist(corrected, bins=corrected_edges, histtype="step", linewidth=1.3,
                             label=f"channel {channel}: corrected")
                axes[1].hist(direct, bins=corrected_edges, histtype="step", linestyle="--",
                             linewidth=1.2, label=f"channel {channel}: raw − ref")
        axes[1].legend(fontsize=8)
    axes[1].set(xlabel="T0 time after channel-31 subtraction (ns)",
                ylabel="Readouts / bin", title="Per-channel single-subtraction check")
    if corrected_all:
        axes[1].set_xlim(*zoom_limits)
    fig.suptitle("T0 reference-subtraction diagnostic")
    fig.tight_layout()
    save_data(fig, args.output_dir / "t0_raw_vs_reference_subtraction.png", run_number)

    raw_t0_reference_pairs = {}
    for channel in t0_channels:
        channel_rows = [r for r in rows if channel in r["raw_times"]
                        and r["reference31_raw"] is not None]
        raw_t0_reference_pairs[channel] = (
            np.asarray([r["reference31_raw"] for r in channel_rows], dtype=float),
            np.asarray([r["raw_times"][channel] for r in channel_rows], dtype=float))
    nonempty_pairs = [(x, y) for x, y in raw_t0_reference_pairs.values() if len(x)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    mesh = None
    if nonempty_pairs:
        all_reference = np.concatenate([x for x, _ in nonempty_pairs])
        all_channel_times = np.concatenate([y for _, y in nonempty_pairs])
        xlim = tuple(np.percentile(all_reference, [0.5, 99.5]))
        ylim = tuple(np.percentile(all_channel_times, [0.5, 99.5]))
        xpad = max((xlim[1] - xlim[0]) * 0.05, 0.5)
        ypad = max((ylim[1] - ylim[0]) * 0.05, 0.5)
        display_range = ((xlim[0] - xpad, xlim[1] + xpad),
                         (ylim[0] - ypad, ylim[1] + ypad))
    else:
        display_range = None
    for channel, axis in zip(t0_channels, axes.flat):
        reference_values, channel_values = raw_t0_reference_pairs[channel]
        if len(channel_values):
            x_range, y_range = display_range
            in_view = ((reference_values >= x_range[0]) & (reference_values <= x_range[1])
                       & (channel_values >= y_range[0]) & (channel_values <= y_range[1]))
            mesh = axis.hist2d(reference_values, channel_values, bins=80, range=display_range,
                               norm=matplotlib.colors.LogNorm(), cmap="viridis")[-1]
            shown = int(np.count_nonzero(in_view))
        else:
            shown = 0
        axis.set(title=f"T0 channel {channel} (shown {shown:,}/{len(channel_values):,})",
                 xlabel="Raw channel-31 reference time (ns)",
                 ylabel="Raw T0 channel time (ns)")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Readouts / bin", shrink=0.9, pad=0.02)
    fig.suptitle("Raw T0 channel time versus raw channel-31 reference (central 99% view)")
    save_data(fig, args.output_dir / "t0_channels_vs_reference_2d.png", run_number)

    # T0 channel geometry: 0/1 are the upper pair and 2/3 the lower pair;
    # 0/2 are left and 1/3 right. The common channel-31 subtraction cancels
    # algebraically in both contrasts. Keep channel offsets unadjusted so the
    # observed centroid is retained, but label the axes as uncalibrated.
    beam_spot_events = [r for r in rows if all(ch in r["times"] for ch in t0_channels)]
    vertical_asymmetry = np.asarray([
        (r["times"][t0_channels[0]] + r["times"][t0_channels[1]]
         - r["times"][t0_channels[2]] - r["times"][t0_channels[3]]) / 2
        for r in beam_spot_events], dtype=float)
    horizontal_asymmetry = np.asarray([
        (r["times"][t0_channels[0]] + r["times"][t0_channels[2]]
         - r["times"][t0_channels[1]] - r["times"][t0_channels[3]]) / 2
        for r in beam_spot_events], dtype=float)
    beam_spot_plot_summary = {"events_with_all_four_t0_channels": len(beam_spot_events),
                              "channel_times": "reference-subtracted once; no per-channel offset equalization",
                              "vertical_top_minus_bottom_median_ns": (
                                  float(np.median(vertical_asymmetry)) if len(vertical_asymmetry) else None),
                              "horizontal_left_minus_right_median_ns": (
                                  float(np.median(horizontal_asymmetry)) if len(horizontal_asymmetry) else None)}
    if len(vertical_asymmetry):
        x_low, x_high = np.percentile(horizontal_asymmetry, [0.5, 99.5])
        y_low, y_high = np.percentile(vertical_asymmetry, [0.5, 99.5])
        x_pad = max((x_high - x_low) * 0.08, 0.5)
        y_pad = max((y_high - y_low) * 0.08, 0.5)
        beam_spot_range = ((float(x_low - x_pad), float(x_high + x_pad)),
                           (float(y_low - y_pad), float(y_high + y_pad)))
        in_view = ((horizontal_asymmetry >= beam_spot_range[0][0])
                   & (horizontal_asymmetry <= beam_spot_range[0][1])
                   & (vertical_asymmetry >= beam_spot_range[1][0])
                   & (vertical_asymmetry <= beam_spot_range[1][1]))
        fig, axis = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
        mesh = axis.hist2d(horizontal_asymmetry, vertical_asymmetry, bins=100,
                           range=beam_spot_range, norm=matplotlib.colors.LogNorm(),
                           cmap="viridis")[3]
        fig.colorbar(mesh, ax=axis, label="Readouts / bin")
        axis.set(xlabel="[(T00 + T02) − (T01 + T03)] / 2 (ns; left − right)",
                 ylabel="[(T00 + T01) − (T02 + T03)] / 2 (ns; top − bottom)",
                 title=f"T0 timing-asymmetry proxy; central 99% view, shown {np.count_nonzero(in_view):,}/{len(in_view):,}")
        fig.suptitle("T0 beam-spot timing proxy (not position-calibrated; no channel equalization)")
        save_data(fig, args.output_dir / "t0_beam_spot_timing_proxy.png", run_number)
        beam_spot_plot_summary["plot"] = "t0_beam_spot_timing_proxy.png"
        beam_spot_plot_summary["events_shown_in_central_view"] = int(np.count_nonzero(in_view))
        beam_spot_plot_summary["display_range_ns"] = [list(beam_spot_range[0]), list(beam_spot_range[1])]

    else:
        beam_spot_plot_summary["plot"] = None

    def in_gate(label, value):
        center = gates[label]["center_ns"]
        return value is not None and np.isfinite(center) and abs(value - center) <= args.prompt_half_width_ns

    multiplicities = np.zeros((8, 3), dtype=np.int64)
    end_differences = [[] for _ in range(8)]
    tofs = {"T2-T0": [], "T0-T5": [], "T0-HD0": [], "T0-HD5": [], "T0-HD8": []}
    ungated_t2_t0 = []
    gate_counts = {"t0_in_gate": 0, "t2_in_gate": 0, "both_in_gate": 0,
                   "both_groups_present_before_gates": 0}
    t5_bar_tofs = [[] for _ in range(8)]
    for row in rows:
        t0_ok = in_gate(t0_label, row["t0"])
        t2_ok = in_gate(t2_label, row["t2"])
        gate_counts["t0_in_gate"] += int(t0_ok)
        gate_counts["t2_in_gate"] += int(t2_ok)
        if row["t0"] is not None and row["t2"] is not None:
            gate_counts["both_groups_present_before_gates"] += 1
            ungated_t2_t0.append(row["t2"] - row["t0"])
        if t0_ok and t2_ok:
            gate_counts["both_in_gate"] += 1
            tofs["T2-T0"].append(row["t2"] - row["t0"])
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

    # A first-look T5 beam-profile proxy: bar number gives the vertical
    # detector row (bar 0 at the bottom through bar 7 at the top), while the
    # paired-end time difference carries information along each horizontal
    # bar. Do not label this as a position in mm until
    # the bar layout, end orientation, channel offsets, and propagation speed
    # have been calibrated.
    t5_bar_ids = np.concatenate([
        np.full(len(values), bar, dtype=float)
        for bar, values in enumerate(end_differences) if values
    ]) if any(end_differences) else np.asarray([], dtype=float)
    t5_end_dt = np.concatenate([
        np.asarray(values, dtype=float)
        for values in end_differences if values
    ]) if any(end_differences) else np.asarray([], dtype=float)
    t5_beamspot_summary = {
        "plot": None,
        "paired_prompt_gated_entries": int(len(t5_end_dt)),
        "coordinates": "lower-channel minus higher-channel time (ns) horizontally; T5 bar row vertically (0 bottom to 7 top; channel pairs 48/56 through 55/63)",
        "note": "Exploratory profile only; not spatially calibrated. Bar positions/orientation, per-channel timing offsets, and effective light propagation speed are not applied.",
    }
    if len(t5_end_dt):
        fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
        mesh = axis.hist2d(
            t5_end_dt, t5_bar_ids,
            bins=[100, np.arange(-0.5, 8.5, 1.0)],
            range=[[-args.end_diff_range_ns, args.end_diff_range_ns], [-0.5, 7.5]],
            norm=matplotlib.colors.LogNorm(), cmap="viridis",
        )[3]
        fig.colorbar(mesh, ax=axis, label="Prompt-gated paired readouts / bin")
        axis.set(
            xticks=np.linspace(-args.end_diff_range_ns, args.end_diff_range_ns, 5),
            xlabel="Lower-numbered end time − higher-numbered end time (ns; along-bar proxy)",
            yticks=np.arange(8),
            yticklabels=[f"{bar}  ({48 + bar}/{56 + bar})" for bar in range(8)],
            ylabel="T5 bar row (bottom to top; bar number / TOF channel pair)",
            title="T5 beam-profile proxy (end-time asymmetry × bar row)",
            xlim=(-args.end_diff_range_ns, args.end_diff_range_ns), ylim=(-0.5, 7.5),
        )
        fig.suptitle("Not a calibrated spatial beam spot")
        save_data(fig, args.output_dir / "beamline_prompt_t5_beam_profile_proxy.png", run_number)
        t5_beamspot_summary["plot"] = "beamline_prompt_t5_beam_profile_proxy.png"

    # TOF panels are split into pages of no more than four plots.
    tof_items = list(tofs.items())
    t2_t0_zoom_summary = None
    t0_hd_zoom_summaries = {}
    for page_start in range(0, len(tof_items), 4):
        page_items = tof_items[page_start:page_start + 4]
        rows_n = 2
        cols_n = 2
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(12, 8))
        for axis, (label, vals) in zip(axes.flat, page_items):
            if vals and label not in ("T2-T0", "T0-HD0", "T0-HD8"):
                axis.hist(vals, bins=120, histtype="step", color="tab:blue")
            if label == "T2-T0" and ungated_t2_t0:
                all_values = np.asarray(ungated_t2_t0, dtype=float)
                zoom_low, zoom_high = np.percentile(all_values, [0.5, 99.5])
                padding = max((zoom_high - zoom_low) * 0.08, 0.5)
                zoom_limits = (float(zoom_low - padding), float(zoom_high + padding))
                common_edges = np.linspace(zoom_limits[0], zoom_limits[1], 121)
                all_in_view = int(np.count_nonzero((all_values >= zoom_limits[0])
                                                   & (all_values <= zoom_limits[1])))
                gated_values = np.asarray(vals, dtype=float)
                gated_in_view = int(np.count_nonzero((gated_values >= zoom_limits[0])
                                                     & (gated_values <= zoom_limits[1])))
                axis.hist(all_values, bins=common_edges, histtype="step", color="tab:orange",
                          label=f"all T2−T0 pairs (n={len(ungated_t2_t0):,})")
                if vals:
                    axis.hist(gated_values, bins=common_edges, histtype="step", color="tab:blue",
                              label=f"both prompt gates (n={len(vals):,})")
                axis.set_xlim(*zoom_limits)
                t2_t0_zoom_summary = {
                    "display_range_ns": list(zoom_limits),
                    "all_pairs_total": len(all_values), "all_pairs_in_view": all_in_view,
                    "gated_pairs_total": len(gated_values), "gated_pairs_in_view": gated_in_view,
                    "quantile_range_percent": [0.5, 99.5],
                }
                axis.legend(fontsize=8)
            if label in ("T0-HD0", "T0-HD8") and vals:
                hd_values = np.asarray(vals, dtype=float)
                zoom_low, zoom_high = np.percentile(hd_values, [0.5, 99.5])
                padding = max((zoom_high - zoom_low) * 0.08, 0.5)
                hd_limits = (float(zoom_low - padding), float(zoom_high + padding))
                hd_edges = np.linspace(hd_limits[0], hd_limits[1], 121)
                hd_in_view = int(np.count_nonzero((hd_values >= hd_limits[0])
                                                  & (hd_values <= hd_limits[1])))
                axis.hist(hd_values, bins=hd_edges, histtype="step", color="tab:blue")
                axis.set_xlim(*hd_limits)
                t0_hd_zoom_summaries[label] = {
                    "display_range_ns": list(hd_limits), "total": len(hd_values),
                    "in_view": hd_in_view, "quantile_range_percent": [0.5, 99.5],
                }
            axis.set(xlabel="Reference-corrected detector-time difference (ns)", ylabel="Readouts / bin",
                     title=(f"{label} (central 99% view; out of view: "
                            f"all {len(ungated_t2_t0) - all_in_view:,}, "
                            f"gated {len(vals) - gated_in_view:,})"
                            if label == "T2-T0" and ungated_t2_t0
                            else f"{label} (prompt-gated n={len(vals):,})"))
            if label in ("T0-HD0", "T0-HD5", "T0-HD8"):
                hd_id = int(label.split("HD", 1)[1])
                channel = next(ch for ch, element in HD_CHANNELS.items() if element == hd_id)
                bank = ("different ref banks (T0=31, HD=46); uncalibrated"
                        if channel > 31 else "same ref 31; detector offsets remain")
                if label in t0_hd_zoom_summaries:
                    hd_zoom = t0_hd_zoom_summaries[label]
                    outside = hd_zoom["total"] - hd_zoom["in_view"]
                    detail = f"central 99% view; out of view {outside:,}/{len(vals):,}"
                else:
                    detail = f"prompt-gated n={len(vals):,}"
                axis.set_title(f"{label}: {bank}\n{detail}", fontsize=9)
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
               "t0_channels": list(t0_channels), "t2_channels": list(t2_channels),
               "t0_data_derived_relative_channel_offsets_ns": {
                   str(ch): offset for ch, offset in t0_relative_offsets.items()},
               "t0_time_comparison_widths_ns": {
                   mode: {"median": float(np.median(values)),
                          "q16": float(np.percentile(values, 16)),
                          "q84": float(np.percentile(values, 84)),
                          "q84_minus_q16": float(np.percentile(values, 84) - np.percentile(values, 16))}
                   for mode, values in t0_comparison.items() if values},
               "tof_counts": {name: len(values) for name, values in tofs.items()},
               "t2_t0_histogram_zoom": t2_t0_zoom_summary,
               "t0_hd_histogram_zooms": t0_hd_zoom_summaries,
               "t0_t2_prompt_gate_counts": gate_counts,
               "t0_beam_spot_timing_proxy": beam_spot_plot_summary,
               "t0_channel_residual_display_range_ns": [-args.t0_residual_range_ns,
                                                        args.t0_residual_range_ns],
               "t0_channel_residuals_outside_display_range": {
                   str(ch): sum(abs(value) > args.t0_residual_range_ns for value in values)
                   for ch, values in t0_residuals_for_plot.items()},
               "t0_channel_residuals_after_equalization_outside_display_range": {
                   str(ch): sum(abs(value) > args.t0_residual_range_ns for value in values)
                   for ch, values in t0_residuals_after_equalization.items()},
               "t5_end_multiplicity": multiplicities.tolist(),
               "t5_beam_profile_proxy": t5_beamspot_summary,
               "selection_note": "Beamline references required; vme_digi_issues_bitmask==0 when present. No WCTE PMT quality mask. T5 bars are not vetoed for multiplicity.",
               "t0_offset_comparison_note": "The reference-only series is the analysis default. Relative channel offsets are estimated from this run's T0 residuals solely for comparison and are not applied to TOFs or prompt gates. The absolute four-channel average retains event-to-event common-mode timing spread; the per-channel residual plot subtracts the same-event median and therefore does not show that common-mode spread."}
    summary["reference_subtraction_diagnostic"] = {
        "plot": "t0_raw_vs_reference_subtraction.png",
        "correlation_plot": "t0_channels_vs_reference_2d.png",
        "per_channel": subtraction_checks,
        "note": "Each channel is checked against its own raw time minus raw channel 31. This tests the script's subtraction arithmetic but cannot infer whether an undocumented correction was applied upstream of the preserved ROOT branch."
    }
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
    result.add_argument("--t2-channels", type=int, nargs="+", default=T2_CHANNELS,
                        help="T2 TDC channel(s) to average (default: channel 8)")
    result.add_argument("--t0-residual-range-ns", type=float, default=5.0,
                        help="Half-range for the zoomed T0 channel-residual comparison plot")
    result.add_argument("--prompt-half-width-ns", type=float, default=40.0)
    result.add_argument("--peak-bin-ns", type=float, default=2.0)
    result.add_argument("--end-diff-range-ns", type=float, default=5.0,
                        help="Half-range for the T5 left-right end-time-difference plot")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
