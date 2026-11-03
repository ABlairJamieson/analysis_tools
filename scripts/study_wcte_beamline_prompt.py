#!/usr/bin/env python3
"""Make exploratory beamline and WCTE charge plots from merged WCTE ROOT data.

How the script is organized
---------------------------
1. Read the beamline TDC/QDC branches and optional WCTE PMT branches.
2. Keep readouts with both beamline reference channels and, when available,
   a clear beamline digitiser-quality flag. Build one small Python record per
   retained readout (see ``rows`` in :func:`run`).
3. Find fixed prompt peaks from the retained sample, then calculate T0/T2/T5/HD
   timing diagnostics and the T2-T0 positron-enriched first-peak window.
4. Write plots and ``summary.json``. Optional charge plots are diagnostics;
   they do not add an event-quality or WCTE-time cut to the original samples.

Timing conventions and limitations
-----------------------------------
Beamline channels use the first hit on reference channel 31 for channel IDs
<=31 and channel 46 for IDs >31. This corrects times within each beamline TDC
bank; it does not align the beamline clock with the WCTE PMT clock. WCTE PMT
times are handled separately. Their optional prompt gate is centered on the
dominant accepted PMT-time peak, not on T0. Detector-to-detector differences
are therefore timing diagnostics, not calibrated physical TOFs.

The fixed prompt gates, ``first significant`` T2-T0 peak, and HD/T5 cuts are
exploratory selections. Read the plot labels and ``summary.json`` before
interpreting them as particle identification or applying them to physics data.
See the README section “Reading and modifying the WCTE studies” for a code map.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
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
QDC_BRANCHES = ("beamline_pmt_qdc_ids", "beamline_pmt_qdc_charges")
HD_CHARGE_DISPLAY_RANGE = (0.0, 3000.0)
HD_CHARGE_HIST_BINS = 100
HC2_CHANNEL = 11
WCTE_CHARGE_BRANCHES = ("hit_pmt_charges", "hit_pmt_readout_mask")
WCTE_TIME_BRANCH = "hit_pmt_calibrated_times"
WCTE_EVENT_QUALITY_BRANCH = "window_data_quality_mask"
T5_PROMPT_BRANCH = "T5_HasInTimeWindow"


def main_peak(values: list[float], *, bin_ns: float = 2.0) -> tuple[float, float, int]:
    """Return the modal-bin center/count and number of finite input values.

    The histogram range excludes the outer 0.1% tails at each end so an
    isolated outlier does not set the bin width/range for a prompt peak.
    """
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


def first_significant_peak(values: list[float], *, bin_ns: float = 0.2,
                           min_fraction: float = 0.10) -> tuple[float, int, int]:
    """Find the earliest locally maximal TOF bin above a fraction of the tallest peak.

    A short moving average damps single-bin noise. The returned center is a
    data-driven gate anchor, not a particle identification by itself.
    ``min_fraction`` is relative to the tallest smoothed peak. Among local
    maxima passing that threshold, the earliest is returned. This is intended
    to enrich the positron population relative to slower particles; it is not
    by itself a particle-identification result.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan"), 0, 0
    lo, hi = np.percentile(data, [0.1, 99.9])
    if hi <= lo:
        return float(data[0]), int(len(data)), int(len(data))
    edges = np.arange(lo, hi + bin_ns, bin_ns)
    if len(edges) < 2:
        edges = np.asarray([lo, lo + bin_ns])
    counts, edges = np.histogram(data, bins=edges)
    smooth = (np.convolve(counts.astype(float), np.asarray([1, 2, 3, 2, 1]) / 9.0,
                          mode="same") if len(counts) >= 5 else counts.astype(float))
    threshold = float(np.max(smooth)) * min_fraction
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]
             and smooth[i] >= threshold]
    index = peaks[0] if peaks else int(np.argmax(smooth))
    return float((edges[index] + edges[index + 1]) / 2), int(counts[index]), int(len(data))


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
    """Correct beamline TDC hits and keep the first hit per detector channel.

    The TDC reference correction itself lives in ``beamline_timing.py``. This
    wrapper returns both raw and reference-corrected first-hit maps because
    some plots compare the two. Repeated detector hits are intentionally not
    used by the prompt-gate analysis.
    """
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


def plot_wcte_charge_diagnostics(rows: list[dict], prompt_center: float | None,
                                 prompt_half_width: float, low_charge_max: float,
                                 out: Path, run_number: int | str | None,
                                 has_times: bool) -> dict:
    """Write follow-up plots and a row-level CSV for zero/low WCTE charge.

    ``rows`` are the beamline-quality readouts from :func:`run`. This function
    only visualizes all-time and independently prompt-gated PMT charge; it
    does not remove readouts from any existing cut-flow sample.
    """
    hd_rows = list({row["entry"]: row for row in rows if row["hd"]}.values())
    final_rows = [row for row in hd_rows if row.get("t0_prompt", False) and not row.get("hc2_hit", False)
                  and row.get("first_tof", False) and row.get("t5_veto_pass", True)]
    stage_sets = [("HD-tagged", hd_rows), ("T0 prompt", [r for r in hd_rows if r.get("t0_prompt")]),
                  ("T0 + no HC2", [r for r in hd_rows if r.get("t0_prompt") and not r.get("hc2_hit")]),
                  ("Final cuts", final_rows)]

    def charge_counts(sample):
        values = np.asarray([r["wcte_charge"] for r in sample if r.get("wcte_charge") is not None], dtype=float)
        return {"readouts": int(len(values)), "exact_zero": int(np.count_nonzero(values == 0)),
                "positive_le_low_max": int(np.count_nonzero((values > 0) & (values <= low_charge_max))),
                "above_low_max": int(np.count_nonzero(values > low_charge_max))}

    summary = {
        "plots": [], "prompt_time_available": bool(has_times),
        "prompt_center_ns": prompt_center, "prompt_half_width_ns": float(prompt_half_width),
        "low_charge_max_input_units": float(low_charge_max),
        "counts_by_sample": {name: charge_counts(sample) for name, sample in stage_sets},
        "event_quality_available": any(r.get("wcte_event_quality") is not None for r in rows),
        "event_quality_zero_by_sample": {
            name: sum(r.get("wcte_event_quality") == 0 for r in sample) for name, sample in stage_sets},
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axis = axes[0, 0]
    for name, sample in stage_sets:
        values = [r["wcte_charge"] for r in sample if r.get("wcte_charge") is not None
                  and r["wcte_charge"] > 0]
        if values:
            axis.hist(values, bins=np.linspace(0, low_charge_max, 81), histtype="step",
                      label=f"{name} (n={len(values):,})")
    axis.set(xlim=(0, low_charge_max), xlabel="All-time accepted PMT charge (input units)",
             ylabel="Readouts / bin", title="Positive-charge zoom; exact zero excluded")
    zero_note = "Exact zero counts: " + ", ".join(
        f"{name} {charge_counts(sample)['exact_zero']}" for name, sample in stage_sets)
    axis.text(0.02, 0.98, zero_note, transform=axis.transAxes, va="top", fontsize=7)
    axis.legend(fontsize=7)

    axis = axes[0, 1]
    categories = ["exact zero", f"0 < Q ≤ {low_charge_max:g}", f"Q > {low_charge_max:g}"]
    x = np.arange(len(stage_sets))
    width = 0.23
    for j, (label, pred, color) in enumerate((
            (categories[0], lambda q: q == 0, "0.4"),
            (categories[1], lambda q: 0 < q <= low_charge_max, "tab:orange"),
            (categories[2], lambda q: q > low_charge_max, "tab:blue"))):
        vals = [sum(pred(r["wcte_charge"]) for r in sample if r.get("wcte_charge") is not None)
                for _, sample in stage_sets]
        axis.bar(x + (j - 1) * width, vals, width=width, color=color, label=label)
    axis.set(xticks=x, xticklabels=[name for name, _ in stage_sets], ylabel="Readouts",
             title="Exact zero separated from positive low charge")
    axis.tick_params(axis="x", labelrotation=15)
    axis.set_yscale("log"); axis.set_ylim(bottom=0.5); axis.legend(fontsize=7)

    axis = axes[1, 0]
    paired = [(r["wcte_accepted_hits"], r["wcte_charge"]) for r in hd_rows
              if r.get("wcte_accepted_hits") is not None and r.get("wcte_charge") is not None]
    accepted = np.asarray([p[0] for p in paired], dtype=float)
    charges = np.asarray([p[1] for p in paired], dtype=float)
    if len(accepted):
        axis.hist2d(accepted, charges, bins=[np.arange(-0.5, max(accepted.max()+1.5, 2), 1),
                                              np.linspace(0, max(low_charge_max, np.percentile(charges, 99)), 81)],
                    norm=matplotlib.colors.LogNorm(), cmap="viridis")
    axis.set(xlabel="Accepted PMT hits (readout mask = 0)", ylabel="All-time PMT charge (input units)",
             title="Is low charge associated with few accepted hits?")

    axis = axes[1, 1]
    if has_times and prompt_center is not None:
        qvals = np.asarray([r["wcte_prompt_charge"] for r in hd_rows if r.get("wcte_prompt_charge") is not None])
        qhits = np.asarray([r["wcte_prompt_hits"] for r in hd_rows if r.get("wcte_prompt_hits") is not None])
        if len(qvals):
            axis.hist2d(qhits, qvals, bins=[np.arange(-0.5, max(qhits.max()+1.5, 2), 1),
                                             np.linspace(0, max(low_charge_max, np.percentile(qvals, 99)), 81)],
                        norm=matplotlib.colors.LogNorm(), cmap="viridis")
        axis.set(xlabel=f"Accepted PMT hits in ±{prompt_half_width:g} ns prompt gate",
                 ylabel="Prompt-window PMT charge (input units)", title="Prompt-window charge vs hit count")
    else:
        axis.text(0.5, 0.5, "PMT calibrated-time branch unavailable\n(prompt diagnostics omitted)",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    fig.suptitle("WCTE charge investigation — HD-tagged readouts")
    fig.tight_layout()
    filename = "beamline_wcte_charge_low_investigation.png"
    save_data(fig, out / filename, run_number); summary["plots"].append(filename)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    quality_groups = [("quality mask = 0", [r for r in hd_rows if r.get("wcte_event_quality") == 0]),
                      ("quality mask ≠ 0", [r for r in hd_rows if r.get("wcte_event_quality") not in (None, 0)]),
                      ("quality unavailable", [r for r in hd_rows if r.get("wcte_event_quality") is None])]
    x = np.arange(len(quality_groups)); width = 0.24
    for j, (label, predicate, color) in enumerate((
            ("exact zero", lambda q: q == 0, "0.4"),
            (f"0 < Q ≤ {low_charge_max:g}", lambda q: 0 < q <= low_charge_max, "tab:orange"),
            (f"Q > {low_charge_max:g}", lambda q: q > low_charge_max, "tab:blue"))):
        counts = [sum(predicate(r["wcte_charge"]) for r in group
                      if r.get("wcte_charge") is not None) for _, group in quality_groups]
        axes[0].bar(x + (j - 1) * width, counts, width=width, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=[name for name, _ in quality_groups], ylabel="Readouts",
                title="Charge class versus event-level WCTE quality mask")
    axes[0].tick_params(axis="x", labelrotation=12)
    axes[0].set_yscale("log"); axes[0].set_ylim(bottom=0.5); axes[0].legend(fontsize=8)

    accepted_counts = [sum(r.get("wcte_accepted_hits") == n for r in hd_rows) for n in range(5)]
    masked_counts = [sum(r.get("wcte_masked_hits") == n for r in hd_rows) for n in range(5)]
    x = np.arange(5)
    axes[1].bar(x - 0.18, accepted_counts, width=0.36, label="accepted (mask=0)")
    axes[1].bar(x + 0.18, masked_counts, width=0.36, label="masked (mask≠0)")
    axes[1].set(xticks=x, xlabel="PMT-hit count per readout (shown 0–4)", ylabel="HD-tagged readouts",
                title="Accepted and masked hit multiplicities")
    axes[1].set_yscale("log"); axes[1].set_ylim(bottom=0.5); axes[1].legend(fontsize=8)
    fig.tight_layout()
    filename = "beamline_wcte_charge_quality_mask_checks.png"
    save_data(fig, out / filename, run_number); summary["plots"].append(filename)

    if has_times and prompt_center is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        paired_charge = [(r["wcte_charge"], r["wcte_prompt_charge"]) for r in hd_rows
                         if r.get("wcte_charge") is not None
                         and r.get("wcte_prompt_charge") is not None]
        all_values = np.asarray([pair[0] for pair in paired_charge], dtype=float)
        prompt_values = np.asarray([pair[1] for pair in paired_charge], dtype=float)
        edges = np.linspace(0, low_charge_max, 81)
        axes[0].hist(all_values[all_values > 0], bins=edges, histtype="step",
                     label=f"all-time positive charge (n={np.count_nonzero(all_values > 0):,})")
        axes[0].hist(prompt_values[prompt_values > 0], bins=edges, histtype="step",
                     label=f"prompt-window positive charge (n={np.count_nonzero(prompt_values > 0):,})")
        axes[0].text(0.02, 0.98,
                     f"exact zero: all-time {np.count_nonzero(all_values == 0):,}; "
                     f"prompt {np.count_nonzero(prompt_values == 0):,}",
                     transform=axes[0].transAxes, va="top", fontsize=8)
        axes[0].set(xlim=(0, low_charge_max), xlabel="Accepted PMT charge (input units)",
                    ylabel="HD-tagged readouts / bin", title="Does a prompt-time gate change the charge spectrum?")
        axes[0].legend(fontsize=8)
        if len(all_values):
            axes[1].hist2d(all_values, prompt_values,
                           bins=[np.linspace(0, max(low_charge_max, np.percentile(all_values, 99)), 81),
                                 np.linspace(0, max(low_charge_max, np.percentile(prompt_values, 99)), 81)],
                           norm=matplotlib.colors.LogNorm(), cmap="viridis")
        axes[1].set(xlabel="All-time accepted PMT charge", ylabel="Prompt-window accepted PMT charge",
                    title="Readout-by-readout charge comparison")
        fig.tight_layout()
        filename = "beamline_wcte_all_time_vs_prompt_charge.png"
        save_data(fig, out / filename, run_number); summary["plots"].append(filename)

    # Per-event rows make it possible to inspect anomalous zero/low-charge cases.
    csv_name = "wcte_low_charge_event_diagnostics.csv"
    with (out / csv_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["root_entry", "event_number", "readout_number", "hd_elements",
                         "all_time_charge", "all_time_accepted_hits", "masked_hits",
                         "total_pmt_hits", "event_quality_mask", "prompt_charge",
                         "prompt_accepted_hits", "in_final_beamline_selection"])
        for row in hd_rows:
            q = row.get("wcte_charge")
            pq = row.get("wcte_prompt_charge")
            if q is None or (q > low_charge_max and (pq is None or pq > low_charge_max)):
                continue
            writer.writerow([row["entry"], row.get("event_number"), row.get("readout_number"),
                             ";".join(map(str, sorted(row["hd"]))), q, row.get("wcte_accepted_hits"),
                             row.get("wcte_masked_hits"), row.get("wcte_total_hits"),
                             row.get("wcte_event_quality"), pq, row.get("wcte_prompt_hits"),
                             row in final_rows])
    summary["event_csv"] = csv_name
    return summary


def plot_peak_distributions(samples: dict[str, list[float]], gates: dict[str, dict],
                            out: Path, run_number: int | str | None = None,
                            t0_label: str = "T0 four-PMT average",
                            t2_label: str = "T2"):
    """Write separate peak/gate pages for T0/T2, T5 bars, and HD channels.

    Each fixed detector gate is calculated once in :func:`run` and passed in
    ``gates``. T5 is shown per end/bar because cable and detector offsets differ.
    """
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
    """Run the analysis and write figures, a CSV detail table, and summary JSON.

    ``rows`` below is the hand-off between ROOT reading and plotting. Each
    record represents one beamline-quality readout and contains raw/corrected
    beamline times plus optional WCTE diagnostics. Keeping this intermediate
    representation makes the selection logic inspectable without repeatedly
    decoding ROOT branches in every plot block.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0_channels = tuple(getattr(args, "t0_channels", T0_CHANNELS))
    t2_channels = tuple(getattr(args, "t2_channels", T2_CHANNELS))
    tof_peak_bin_ns = getattr(args, "tof_peak_bin_ns", 0.2)
    tof_first_peak_half_width_ns = getattr(args, "tof_first_peak_half_width_ns", 0.5)
    tof_first_peak_min_fraction = getattr(args, "tof_first_peak_min_fraction", 0.10)
    wcte_prompt_half_width = getattr(args, "wcte_prompt_half_width_ns", 40.0)
    wcte_prompt_center_override = getattr(args, "wcte_prompt_center_ns", None)
    wcte_prompt_bin = getattr(args, "wcte_prompt_peak_bin_ns", 2.0)
    wcte_low_charge_max = getattr(args, "wcte_low_charge_max", 20000.0)
    if (tof_peak_bin_ns <= 0 or tof_first_peak_half_width_ns <= 0
            or not 0 < tof_first_peak_min_fraction <= 1):
        raise ValueError("TOF peak bin/half-width must be positive and minimum peak fraction in (0, 1]")
    if wcte_prompt_half_width <= 0 or wcte_prompt_bin <= 0 or wcte_low_charge_max <= 0:
        raise ValueError("WCTE prompt half-width, peak bin, and low-charge threshold must be positive")
    # 1) Read only branches this study uses. Optional branches add diagnostics;
    #    only REQUIRED beamline TDC fields are necessary to run the core study.
    with uproot.open(args.input_root) as root:
        tree = root[args.tree]
        missing = set(REQUIRED) - set(tree.keys())
        if missing:
            raise ValueError("Missing ROOT branches: " + ", ".join(sorted(missing)))
        available = set(tree.keys())
        has_quality = "vme_digi_issues_bitmask" in available
        has_qdc = all(name in available for name in QDC_BRANCHES)
        has_wcte_charge = all(name in available for name in WCTE_CHARGE_BRANCHES)
        has_wcte_times = WCTE_TIME_BRANCH in available
        has_wcte_event_quality = WCTE_EVENT_QUALITY_BRANCH in available
        has_t5_prompt_flag = T5_PROMPT_BRANCH in available
        partial_qdc = any(name in available for name in QDC_BRANCHES) and not has_qdc
        if partial_qdc:
            raise ValueError("ROOT file must contain both QDC branches: " + ", ".join(QDC_BRANCHES))
        optional = [name for name in ("vme_digi_issues_bitmask", "run_id", "event_number",
                                      "readout_number", *QDC_BRANCHES, *WCTE_CHARGE_BRANCHES,
                                      WCTE_TIME_BRANCH, WCTE_EVENT_QUALITY_BRANCH, T5_PROMPT_BRANCH)
                    if name in available]
        start = args.entry_start
        stop = min(tree.num_entries, start + args.max_events)
        print(f"Reading ROOT entries {start:,}–{stop:,} from {args.input_root} "
              f"({len(REQUIRED) + len(optional)} branches)...", flush=True)
        read_started = time.perf_counter()
        arrays = tree.arrays([*REQUIRED, *optional], entry_start=start, entry_stop=stop,
                             library="ak")
        print(f"Read {stop - start:,} entries in {time.perf_counter() - read_started:.1f} s. "
              "Building beamline readout records...", flush=True)
        run_number = None
        if "run_id" in optional and stop > start:
            try:
                run_number = int(ak.to_list(arrays["run_id"])[0])
            except (TypeError, ValueError):
                run_number = str(ak.to_list(arrays["run_id"])[0])
        if run_number is None:
            match = re.search(r"(?:^|[^A-Za-z])R?(\d{4,})(?:[^0-9]|$)", args.input_root.name)
            run_number = int(match.group(1)) if match else None

    # 2) Convert each readout into a compact record. Reject bad beamline
    #    references/digitiser windows here; WCTE PMT quality is NOT an event cut.
    rows = []
    qdc_charge_matrix = (np.full((stop - start, 15), np.nan, dtype=np.float32)
                         if has_qdc else None)
    reference_good = 0
    missing_reference = {"31": 0, "46": 0}
    process_started = time.perf_counter()

    def report_progress(processed: int) -> None:
        """Print a compact status line at the configured readout interval."""
        if processed % args.progress_every == 0 or processed == stop - start:
            elapsed = time.perf_counter() - process_started
            rate = processed / elapsed if elapsed else 0.0
            print(f"Processed {processed:,}/{stop - start:,} readouts; "
                  f"{len(rows):,} passed beamline quality ({rate:,.0f} readouts/s)",
                  flush=True)

    for i in range(stop - start):
        if has_qdc:
            qdc_ids = ak.to_list(arrays[QDC_BRANCHES[0]][i])
            qdc_values = ak.to_list(arrays[QDC_BRANCHES[1]][i])
            if qdc_ids is not None and qdc_values is not None:
                for channel, charge in zip(qdc_ids, qdc_values):
                    if channel is None or charge is None:
                        continue
                    charge = float(charge)
                    channel = int(channel)
                    if channel in HD_CHANNELS and np.isfinite(charge):
                        element = HD_CHANNELS[channel]
                        # Match the existing beam-monitor loader's first-hit
                        # behavior when a channel is repeated in one readout.
                        if np.isnan(qdc_charge_matrix[i, element]):
                            qdc_charge_matrix[i, element] = charge
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
            report_progress(i + 1)
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
        wcte_charge = None
        if has_wcte_charge:
            charges = ak.to_list(arrays["hit_pmt_charges"][i])
            masks = ak.to_list(arrays["hit_pmt_readout_mask"][i])
            if charges is not None and masks is not None and len(charges) == len(masks):
                good_charges = [float(charge) for charge, mask in zip(charges, masks)
                                if charge is not None and mask is not None and int(mask) == 0
                                and np.isfinite(float(charge))]
                wcte_charge = float(np.sum(good_charges))
            accepted_hits = (len(good_charges) if charges is not None and masks is not None
                             and len(charges) == len(masks) else None)
            masked_hits = (sum(mask is not None and int(mask) != 0 for mask in masks)
                           if masks is not None else None)
            total_hits = len(charges) if charges is not None else None
        else:
            accepted_hits = masked_hits = total_hits = None
        t5_in_time_window = None
        if has_t5_prompt_flag:
            value = ak.to_list(arrays[T5_PROMPT_BRANCH][i])
            if value is not None:
                t5_in_time_window = bool(value)
        event_quality = (int(ak.to_list(arrays[WCTE_EVENT_QUALITY_BRANCH][i]))
                         if has_wcte_event_quality else None)
        event_number = (ak.to_list(arrays["event_number"][i]) if "event_number" in optional else None)
        readout_number = (ak.to_list(arrays["readout_number"][i]) if "readout_number" in optional else None)
        rows.append({"entry": start + i, "times": times, "raw_times": raw_times,
                     "reference31_raw": refs[0], "t0": t0, "t0_raw": t0_raw, "t2": t2,
                     "t5_bars": t5_bars, "hd": hd_times,
                     "wcte_charge": wcte_charge,
                     "wcte_accepted_hits": accepted_hits, "wcte_masked_hits": masked_hits,
                     "wcte_total_hits": total_hits, "wcte_event_quality": event_quality,
                     "event_number": event_number, "readout_number": readout_number,
                     "t5_in_time_window": t5_in_time_window})
        report_progress(i + 1)

    # 3) WCTE PMT time is its own clock. Find its dominant hit-time peak from
    # beamline-quality readouts, then count accepted hits/charge in a fixed gate.
    print("Building prompt peaks and diagnostic plots; large WCTE hit arrays may take a while...",
          flush=True)
    wcte_prompt_peak_count = 0
    if has_wcte_charge and has_wcte_times:
        time_edges = np.arange(-1000.0, 12000.0 + wcte_prompt_bin, wcte_prompt_bin)
        time_counts = np.zeros(len(time_edges) - 1, dtype=np.int64)
        for row in rows:
            index = row["entry"] - start
            charges = ak.to_list(arrays["hit_pmt_charges"][index])
            masks = ak.to_list(arrays["hit_pmt_readout_mask"][index])
            times_values = ak.to_list(arrays[WCTE_TIME_BRANCH][index])
            if charges is None or masks is None or times_values is None:
                continue
            if len(charges) != len(masks) or len(charges) != len(times_values):
                continue
            good_times = [float(t) for t, q, mask in zip(times_values, charges, masks)
                          if t is not None and q is not None and mask is not None
                          and int(mask) == 0 and np.isfinite(float(t)) and np.isfinite(float(q))]
            if good_times:
                time_counts += np.histogram(good_times, bins=time_edges)[0]
        if wcte_prompt_center_override is None and np.any(time_counts):
            peak_index = int(np.argmax(time_counts))
            wcte_prompt_center = float((time_edges[peak_index] + time_edges[peak_index + 1]) / 2)
            wcte_prompt_peak_count = int(time_counts[peak_index])
        elif wcte_prompt_center_override is not None:
            wcte_prompt_center = float(wcte_prompt_center_override)
            index = int(np.searchsorted(time_edges, wcte_prompt_center, side="right") - 1)
            if 0 <= index < len(time_counts):
                wcte_prompt_peak_count = int(time_counts[index])
        else:
            wcte_prompt_center = None
        if wcte_prompt_center is not None:
            for row in rows:
                index = row["entry"] - start
                charges = ak.to_list(arrays["hit_pmt_charges"][index])
                masks = ak.to_list(arrays["hit_pmt_readout_mask"][index])
                times_values = ak.to_list(arrays[WCTE_TIME_BRANCH][index])
                selected = []
                if (charges is not None and masks is not None and times_values is not None
                        and len(charges) == len(masks) == len(times_values)):
                    selected = [(float(q), float(t)) for q, mask, t in zip(charges, masks, times_values)
                                if q is not None and mask is not None and t is not None
                                and int(mask) == 0 and np.isfinite(float(q)) and np.isfinite(float(t))
                                and abs(float(t) - wcte_prompt_center) <= wcte_prompt_half_width]
                row["wcte_prompt_charge"] = float(sum(q for q, _ in selected))
                row["wcte_prompt_hits"] = len(selected)
    else:
        wcte_prompt_center = None
        for row in rows:
            row["wcte_prompt_charge"] = None
            row["wcte_prompt_hits"] = None

    if has_wcte_times and wcte_prompt_center is not None:
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.stairs(time_counts, time_edges, color="tab:blue")
        axis.axvspan(wcte_prompt_center - wcte_prompt_half_width,
                     wcte_prompt_center + wcte_prompt_half_width, color="tab:orange", alpha=0.2,
                     label=f"prompt gate ±{wcte_prompt_half_width:g} ns")
        axis.axvline(wcte_prompt_center, color="tab:orange", linestyle="--",
                     label=f"peak center {wcte_prompt_center:g} ns")
        axis.set(xlabel="Calibrated WCTE PMT hit time (ns; WCTE clock)",
                 ylabel="Accepted PMT hits / bin", title="WCTE PMT times; beamline-quality readouts")
        axis.legend()
        fig.tight_layout()
        save_data(fig, args.output_dir / "beamline_wcte_pmt_time_prompt_gate.png", run_number)

    # 4) Determine every fixed beamline prompt peak once from retained rows. T5 gates
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

    # 5) T0 correction checks. The run-derived channel equalization below is
    #    for comparison only; selection gates continue to use reference-only times.
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

    # 6) Reference subtraction diagnostics: show every raw T0 channel against
    #    raw reference 31, then verify the corrected values against direct
    #    single subtraction. These plots cannot establish upstream processing.
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

    # 7) T0 timing-asymmetry proxies. T0 channel geometry: 0/1 are the upper
    #    pair and 2/3 the lower pair;
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

    # 8) Build the first significant T2-T0 peak window, then use it for the
    #    beamline TOF and cumulative HD-charge diagnostic samples.
    t0_prompt_tof_values = [row["t2"] - row["t0"] for row in rows
                            if row["t2"] is not None and in_gate(t0_label, row["t0"])]
    first_tof_center, first_tof_peak_bin_count, first_tof_entries = first_significant_peak(
        t0_prompt_tof_values,
        bin_ns=tof_peak_bin_ns,
        min_fraction=tof_first_peak_min_fraction,
    )
    first_tof_half_width = tof_first_peak_half_width_ns
    first_tof_bounds = (first_tof_center - first_tof_half_width,
                        first_tof_center + first_tof_half_width)

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

    # 9) HD QDC charge cut-flow. This is plotted as a diagnostic progression;
    #    no QDC charge threshold is imposed by this script.
    hd_charge_progression = {"available": bool(has_qdc), "plot": None,
                             "qdc_to_tdc_channel_mapping": "provisional identity mapping, per user direction"}
    if has_qdc:
        quality_indices = np.asarray([row["entry"] - start for row in rows], dtype=int)
        t0_indices = np.asarray([row["entry"] - start for row in rows
                                 if in_gate(t0_label, row["t0"])], dtype=int)
        t0_tof_indices = np.asarray([row["entry"] - start for row in rows
                                     if row["t2"] is not None and in_gate(t0_label, row["t0"])
                                     and np.isfinite(first_tof_center)
                                     and first_tof_bounds[0] <= row["t2"] - row["t0"]
                                     <= first_tof_bounds[1]], dtype=int)
        stage_indices = {
            "Raw": np.arange(stop - start, dtype=int),
            "Beamline quality": quality_indices,
            "Prompt T0": t0_indices,
            "Prompt T0 + first T2-T0 peak": t0_tof_indices,
        }
        stage_colors = {"Raw": "0.4", "Beamline quality": "tab:blue",
                        "Prompt T0": "tab:orange", "Prompt T0 + first T2-T0 peak": "tab:green"}
        stage_legend_labels = {"Raw": "Raw", "Beamline quality": "Beamline quality",
                               "Prompt T0": "Prompt T0",
                               "Prompt T0 + first T2-T0 peak": "T0 + first TOF peak"}
        charge_counts = {}
        charge_out_of_range = {}
        charge_edges = np.linspace(*HD_CHARGE_DISPLAY_RANGE, HD_CHARGE_HIST_BINS + 1)
        plot_files = []
        element_groups = (range(0, 4), range(4, 8), range(8, 12), range(12, 15))
        for page, elements in enumerate(element_groups, start=1):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
            legend_handles = {}
            elements = list(elements)
            for element, axis in zip(elements, axes.flat):
                tdc_channel = next(ch for ch, hd_element in HD_CHANNELS.items()
                                   if hd_element == element)
                by_stage = {
                    stage: qdc_charge_matrix[indices, element][
                        np.isfinite(qdc_charge_matrix[indices, element])].astype(float)
                    for stage, indices in stage_indices.items()
                }
                all_values = by_stage["Raw"]
                charge_counts[str(element)] = {stage: int(len(values))
                                               for stage, values in by_stage.items()}
                charge_out_of_range[str(element)] = {
                    stage: {"below_range": int(np.count_nonzero(values < HD_CHARGE_DISPLAY_RANGE[0])),
                            "above_range": int(np.count_nonzero(values > HD_CHARGE_DISPLAY_RANGE[1]))}
                    for stage, values in by_stage.items()
                }
                if not len(all_values):
                    axis.text(0.5, 0.5, "no QDC entries", ha="center", va="center",
                              transform=axis.transAxes)
                    axis.set_title(f"HD{element} (QDC/TDC ch {tdc_channel})")
                    axis.set_xlim(*HD_CHARGE_DISPLAY_RANGE)
                    continue
                for stage, values in by_stage.items():
                    if len(values):
                        _, _, artists = axis.hist(values, bins=charge_edges, histtype="step",
                                                  linewidth=1.2, color=stage_colors[stage],
                                                  label=stage_legend_labels[stage])
                        if artists:
                            legend_handles.setdefault(stage, artists[0])
                axis.set_yscale("log")
                axis.set_ylim(bottom=0.5)
                counts_label = "/".join(f"{len(by_stage[stage]):,}" for stage in stage_indices)
                axis.set_title(f"HD{element} (QDC/TDC ch {tdc_channel})\n"
                               f"entries raw/quality/T0/T0+TOF: {counts_label}", fontsize=9)
                axis.set_xlabel("QDC charge (input units)")
                axis.set_ylabel("Readouts / bin")
                axis.set_xlim(*HD_CHARGE_DISPLAY_RANGE)
            for axis in axes.flat[len(elements):]:
                axis.set_visible(False)
            if legend_handles:
                fig.legend(list(legend_handles.values()), list(legend_handles),
                           loc="upper center", ncol=4, fontsize=9,
                           bbox_to_anchor=(0.5, 0.98))
            first_element, last_element = elements[0], elements[-1]
            fig.suptitle(f"HD charge cut progression — HD{first_element}–HD{last_element}\n"
                         f"Final: T0 prompt + first T2−T0 peak ({first_tof_center:.2f} ± "
                         f"{first_tof_half_width:.2f} ns); provisional QDC/TDC channel match; no charge cut",
                         y=1.02)
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            filename = f"beamline_hd_charge_cut_progression_hd_{first_element}-{last_element}.png"
            save_data(fig, args.output_dir / filename, run_number)
            plot_files.append(filename)
        hd_charge_progression.update({"plots": plot_files,
                                      "display_range_input_units": list(HD_CHARGE_DISPLAY_RANGE),
                                      "histogram_bins": HD_CHARGE_HIST_BINS,
                                      "counts_by_element_and_stage": charge_counts,
                                      "out_of_display_range_counts_by_element_and_stage": charge_out_of_range,
                                      "stages": ["Raw", "Beamline quality", "Prompt T0",
                                                 "Prompt T0 + first T2-T0 peak"],
                                      "stage_definitions": {
                                          "Beamline quality": "both reference channels present and digitiser issue flag clear, when available",
                                          "Prompt T0": "beamline-quality readouts within the fixed T0 prompt gate",
                                          "Prompt T0 + first T2-T0 peak": "within the fixed T0 prompt gate and a configurable window around the earliest significant T2-T0 peak; no independent T2 gate",
                                      },
                                      "first_t2_t0_peak": {
                                          "center_ns": (float(first_tof_center)
                                                        if np.isfinite(first_tof_center) else None),
                                          "half_width_ns": float(first_tof_half_width),
                                          "window_ns": (list(map(float, first_tof_bounds))
                                                        if np.isfinite(first_tof_center) else None),
                                          "entries_considered_after_t0_gate": int(first_tof_entries),
                                          "peak_bin_count": int(first_tof_peak_bin_count),
                                          "histogram_bin_ns": float(tof_peak_bin_ns),
                                          "minimum_peak_fraction": float(tof_first_peak_min_fraction),
                                      }})
    else:
        hd_charge_progression["note"] = "QDC branches are absent; no HD charge plots were produced."

    # 10) WCTE event charge through the same beamline selection stages, plus
    #     independent PMT-time/quality-mask checks. The PMT prompt gate uses
    #     WCTE times only and is deliberately not applied to the legacy plots.
    wcte_charge_progression = {
        "available": bool(has_wcte_charge),
        "plots": [],
        "charge_definition": "sum of finite hit_pmt_charges where hit_pmt_readout_mask == 0; no WCTE event-quality selection",
        "t5_prompt_veto_available": bool(has_t5_prompt_flag),
        "hc_tdc_channel": HC2_CHANNEL,
    }
    if has_wcte_charge:
        for row in rows:
            row["t0_prompt"] = row["t0"] is not None and in_gate(t0_label, row["t0"])
            row["hc2_hit"] = HC2_CHANNEL in row["times"]
            row["first_tof"] = (row["t2"] is not None and row["t0"] is not None
                                and np.isfinite(first_tof_center)
                                and first_tof_bounds[0] <= row["t2"] - row["t0"] <= first_tof_bounds[1])
            row["t5_veto_pass"] = (row["t5_in_time_window"] is False
                                   if has_t5_prompt_flag else True)
        stage_names = ["HD TDC hit", "+ T0 prompt", "+ no HC hit", "+ first T2-T0 peak"]
        stage_colors = ["0.35", "tab:blue", "tab:orange", "tab:green", "tab:red"]
        if has_t5_prompt_flag:
            stage_names.append("+ no T5 prompt hit")
        selected_rows: dict[int, dict[str, list[dict]]] = {}
        raw_charge_values = []
        charge_counts_by_hd: dict[str, dict[str, int]] = {}
        readout_counts_by_hd: dict[str, dict[str, int]] = {}
        for element in range(15):
            hd_rows = [row for row in rows if element in row["hd"]]
            t0_rows = [row for row in hd_rows if in_gate(t0_label, row["t0"])]
            no_hc_rows = [row for row in t0_rows if HC2_CHANNEL not in row["times"]]
            tof_rows = [row for row in no_hc_rows
                        if row["t2"] is not None and np.isfinite(first_tof_center)
                        and first_tof_bounds[0] <= row["t2"] - row["t0"] <= first_tof_bounds[1]]
            stages = {"HD TDC hit": hd_rows,
                      "+ T0 prompt": t0_rows,
                      "+ no HC hit": no_hc_rows,
                      "+ first T2-T0 peak": tof_rows}
            if has_t5_prompt_flag:
                stages["+ no T5 prompt hit"] = [row for row in tof_rows
                                                  if row["t5_in_time_window"] is False]
            selected_rows[element] = stages
            for row in hd_rows:
                if row["wcte_charge"] is not None:
                    raw_charge_values.append(row["wcte_charge"])
            readout_counts_by_hd[str(element)] = {
                stage: len(stage_rows) for stage, stage_rows in stages.items()}
            charge_counts_by_hd[str(element)] = {
                stage: sum(row["wcte_charge"] is not None for row in stage_rows)
                for stage, stage_rows in stages.items()}

        if raw_charge_values:
            charge_low = min(0.0, float(np.percentile(raw_charge_values, 0.1)))
            charge_high = float(np.percentile(raw_charge_values, 99.5))
            if not np.isfinite(charge_high) or charge_high <= charge_low:
                charge_high = max(charge_low + 1.0, float(np.max(raw_charge_values)))
            charge_edges = np.linspace(charge_low, charge_high, 101)
        else:
            charge_low, charge_high = 0.0, 1.0
            charge_edges = np.linspace(charge_low, charge_high, 101)

        overflow_by_hd: dict[str, dict[str, dict[str, int]]] = {}
        for page, elements in enumerate((range(0, 4), range(4, 8),
                                         range(8, 12), range(12, 15)), start=1):
            elements = list(elements)
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
            for element, axis in zip(elements, axes.flat):
                stages = selected_rows[element]
                overflow_by_hd[str(element)] = {}
                for color, stage in zip(stage_colors, stage_names):
                    stage_rows = stages.get(stage, [])
                    values = np.asarray([row["wcte_charge"] for row in stage_rows
                                         if row["wcte_charge"] is not None], dtype=float)
                    overflow_by_hd[str(element)][stage] = {
                        "below_range": int(np.count_nonzero(values < charge_low)),
                        "above_range": int(np.count_nonzero(values > charge_high)),
                    }
                    if len(values):
                        axis.hist(values, bins=charge_edges, histtype="step", linewidth=1.3,
                                  color=color, label=f"{stage} (n={len(values):,})")
                axis.set_yscale("log")
                axis.set_ylim(bottom=0.5)
                channel = next(ch for ch, hd_element in HD_CHANNELS.items()
                               if hd_element == element)
                axis.set(title=f"HD{element} TDC ch {channel}",
                         xlabel="WCTE total PMT charge (input units)",
                         ylabel="Readouts / bin", xlim=(charge_low, charge_high))
                axis.legend(fontsize=7)
            for axis in axes.flat[len(elements):]:
                axis.set_visible(False)
            first_element, last_element = elements[0], elements[-1]
            fig.suptitle(f"WCTE charge by HD TDC tag — HD{first_element}–HD{last_element}\n"
                         "Cumulative cuts: T0 prompt → no HC2 TDC hit → first T2−T0 peak"
                         + (" → no T5 prompt-window hit" if has_t5_prompt_flag else ""), y=1.02)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            filename = f"beamline_wcte_charge_cut_progression_hd_{first_element}-{last_element}.png"
            save_data(fig, args.output_dir / filename, run_number)
            wcte_charge_progression["plots"].append(filename)

        wcte_charge_progression.update({
            "display_range_input_units": [charge_low, charge_high],
            "display_range_upper_percentile": 99.5,
            "histogram_bins": 100,
            "readout_counts_by_hd_and_stage": readout_counts_by_hd,
            "charge_counts_by_hd_and_stage": charge_counts_by_hd,
            "out_of_display_range_by_hd_and_stage": overflow_by_hd,
            "stages": stage_names,
            "stage_definitions": {
                "HD TDC hit": "beamline-quality readouts containing that HD element's mapped TDC channel",
                "+ T0 prompt": "HD-tagged readout also falls within the fixed four-channel T0 prompt gate",
                "+ no HC hit": f"T0-prompt sample with no HC2 TDC channel {HC2_CHANNEL} hit",
                "+ first T2-T0 peak": "HC-vetoed sample with T2−T0 inside the first significant peak gate",
                "+ no T5 prompt hit": ("TOF-selected sample with T5_HasInTimeWindow == false" if has_t5_prompt_flag
                                         else "unavailable: T5_HasInTimeWindow branch is absent"),
            },
            "first_t2_t0_peak": {
                "center_ns": float(first_tof_center) if np.isfinite(first_tof_center) else None,
                "half_width_ns": float(first_tof_half_width),
            },
            "notes": [
                "The first stage is already restricted to beamline-quality readouts (both reference channels and clear VME digitiser issue flag when present).",
                "WCTE charge is summed over finite PMT hits whose per-hit readout mask is zero; no event-level WCTE quality cut is applied.",
                "The displayed charge range is common across HD elements and clips only the highest 0.5% of the initial HD-tagged charge distribution; clipped counts are reported.",
            ],
        })
        if not has_t5_prompt_flag:
            wcte_charge_progression["note"] = (
                "T5_HasInTimeWindow is absent; plots end at the TOF stage, so the requested T5 veto stage was not applied.")
        low_diag = plot_wcte_charge_diagnostics(
            rows, wcte_prompt_center, wcte_prompt_half_width, wcte_low_charge_max,
            args.output_dir, run_number, has_wcte_times)
        low_diag.update({
            "available": True,
            "prompt_peak_bin_count": int(wcte_prompt_peak_count),
            "prompt_peak_bin_width_ns": float(wcte_prompt_bin),
            "prompt_center_source": ("user override" if wcte_prompt_center_override is not None
                                      else "mode of accepted WCTE PMT times in beamline-quality readouts"),
            "event_quality_mask_definition": "recorded only; no WCTE event-level quality cut applied",
            "hit_mask_definition": "accepted = per-hit readout mask == 0; masked are all nonzero mask values",
        })
    else:
        wcte_charge_progression["note"] = (
            "hit_pmt_charges and/or hit_pmt_readout_mask branches are absent; no WCTE charge plots were produced.")
        low_diag = {"available": False, "note": wcte_charge_progression["note"]}

    # 11) Final compact T5 multiplicity/profile plots and paginated TOF figures.
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
                if t0_prompt_tof_values:
                    axis.hist(t0_prompt_tof_values, bins=common_edges, histtype="step",
                              color="tab:green", label=f"T0 prompt only (n={len(t0_prompt_tof_values):,})")
                if np.isfinite(first_tof_center):
                    axis.axvspan(first_tof_bounds[0], first_tof_bounds[1], color="tab:green",
                                 alpha=0.16,
                                 label=f"first-peak gate: {first_tof_center:.2f} ± {first_tof_half_width:.2f} ns")
                    axis.axvline(first_tof_center, color="tab:green", linestyle="--", linewidth=1)
                axis.set_xlim(*zoom_limits)
                t2_t0_zoom_summary = {
                    "display_range_ns": list(zoom_limits),
                    "all_pairs_total": len(all_values), "all_pairs_in_view": all_in_view,
                    "gated_pairs_total": len(gated_values), "gated_pairs_in_view": gated_in_view,
                    "t0_prompt_pairs_total": len(t0_prompt_tof_values),
                    "first_peak_center_ns": (float(first_tof_center)
                                              if np.isfinite(first_tof_center) else None),
                    "first_peak_window_ns": (list(map(float, first_tof_bounds))
                                             if np.isfinite(first_tof_center) else None),
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
    # 12) Machine-readable provenance and counts; update this summary whenever
    #     adding/changing a selection or output plot.
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
               "hd_charge_cut_progression": hd_charge_progression,
               "wcte_charge_cut_progression_by_hd_tdc": wcte_charge_progression,
               "wcte_charge_low_diagnostics": low_diag,
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
    if np.isfinite(first_tof_center):
        print(f"First significant T2-T0 peak: {first_tof_center:.2f} ns; "
              f"final charge-stage TOF gate ±{first_tof_half_width:g} ns")
    else:
        print("No significant T2-T0 peak found after the T0 prompt gate")
    print("TOF counts: " + ", ".join(f"{name}={len(values):,}" for name, values in tofs.items()))
    print(f"Results: {args.output_dir}")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    io = result.add_argument_group("Input and output")
    io.add_argument("input_root", type=Path, help="Merged production ROOT file")
    io.add_argument("--tree", default="WCTEReadoutWindows", help="ROOT tree to analyze")
    io.add_argument("--output-dir", type=Path, required=True, help="Directory for plots, CSV, and summary")
    io.add_argument("--entry-start", type=int, default=0, help="First ROOT entry to read (inclusive)")
    io.add_argument("--max-events", type=int, default=1_000_000,
                    help="Maximum ROOT entries to read; use a larger value for a full run")
    io.add_argument("--progress-every", type=int, default=50000,
                    help="Print progress after this many readouts (default: 50000)")

    beamline = result.add_argument_group("Beamline gates and plots")
    beamline.add_argument("--t0-channels", type=int, nargs=4, default=T0_CHANNELS,
                          help="Four T0 TDC channels used for the event average")
    beamline.add_argument("--t2-channels", type=int, nargs="+", default=T2_CHANNELS,
                          help="T2 TDC channel(s) to average (default: channel 8)")
    beamline.add_argument("--t0-residual-range-ns", type=float, default=5.0,
                          help="Half-range for the zoomed T0 channel-residual plot")
    beamline.add_argument("--prompt-half-width-ns", type=float, default=40.0,
                          help="Half-width of each beamline detector's fixed prompt gate")
    beamline.add_argument("--peak-bin-ns", type=float, default=2.0,
                          help="Bin width for locating fixed beamline prompt peaks")
    beamline.add_argument("--tof-first-peak-half-width-ns", type=float, default=0.5,
                          help="Half-width of the T2-T0 gate around its earliest significant peak")
    beamline.add_argument("--tof-peak-bin-ns", type=float, default=0.2,
                          help="Histogram bin width used to locate the first significant T2-T0 peak")
    beamline.add_argument("--tof-first-peak-min-fraction", type=float, default=0.10,
                          help="Minimum smoothed peak height relative to the tallest TOF peak")
    beamline.add_argument("--end-diff-range-ns", type=float, default=5.0,
                          help="Half-range for the T5 left-right end-time-difference plot")

    wcte = result.add_argument_group("WCTE PMT charge diagnostics")
    wcte.add_argument("--wcte-prompt-half-width-ns", type=float, default=40.0,
                      help="WCTE-time gate half-width around its dominant PMT hit-time peak")
    wcte.add_argument("--wcte-prompt-center-ns", type=float, default=None,
                      help="Override WCTE prompt center (independent of beamline clock)")
    wcte.add_argument("--wcte-prompt-peak-bin-ns", type=float, default=2.0,
                      help="Bin width for locating the dominant WCTE PMT hit-time peak")
    wcte.add_argument("--wcte-low-charge-max", type=float, default=20000.0,
                      help="Upper edge for low-charge diagnostic views, in input charge units")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.progress_every < 1:
        parser().error("--progress-every must be at least 1")
    run(arguments)
