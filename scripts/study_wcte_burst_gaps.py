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
from matplotlib.colors import LogNorm
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
HD_DT_WINDOWS = {hd: ((-50.0, 100.0) if hd in (0, 1, 2, 3, 8, 9, 10, 11)
                       else (80.0, 200.0)) for hd in range(15)}
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
            lo, hi = HD_DT_WINDOWS[hd]
            if any(lo < hd_time - t0_time < hi for t0_time in t0):
                tagged = True
                break
    return beam_ok, tagged, t0_group_centers(t0), refs


def hd_matches_to_group(hits: list[tuple[int, float]], t0_time: float
                        ) -> list[tuple[int, int, float, float]]:
    """HD hits inside the existing element-specific timing window of one T0 group."""
    matches = []
    for channel_id, hd_time in hits:
        if channel_id not in HD_IDS:
            continue
        hd_id = HD_IDS[channel_id]
        low, high = HD_DT_WINDOWS[hd_id]
        dt = hd_time - t0_time
        if low < dt < high:
            matches.append((channel_id, hd_id, hd_time, dt))
    return matches


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


def plot_hodoscope_times(path: Path, histograms: dict[int, np.ndarray],
                         edges: np.ndarray, tdc_time_mode: str) -> None:
    """Plot each HD element's TDC hits relative to the earliest T0 group."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for hd_id, axis in enumerate(axes.flat):
        counts = histograms[hd_id]
        axis.stairs(counts, edges, color="tab:purple", linewidth=1.25)
        axis.set_title(f"HD{hd_id} (n={counts.sum():,})", fontsize=10)
        axis.grid(alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel("Corrected TDC time − earliest T0 group (ns)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Hits / bin")
    fig.suptitle(f"Tagged-gamma hodoscope hits by element — {tdc_time_mode} TDC correction\n"
                 "Relative T0 timing only; HD element indicates remaining positron energy")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_hd_residuals(path: Path, values_by_hd: dict[int, list[float]], *,
                      title: str, xlabel: str, limits: tuple[float, float]) -> None:
    """Per-element residual histograms; all values remain available in CSV."""
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    edges = np.linspace(limits[0], limits[1], 151)
    for hd_id, axis in enumerate(axes.flat):
        values = np.asarray(values_by_hd[hd_id], dtype=float)
        axis.hist(values, bins=edges, histtype="step", color="tab:blue", linewidth=1.2)
        axis.set_title(f"HD{hd_id} (n={len(values):,})", fontsize=10)
        axis.grid(alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel(xlabel)
    for axis in axes[:, 0]:
        axis.set_ylabel("Pairs / bin")
    fig.suptitle(title + " — per HD element")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_delayed_match_map(path: Path, t0_residuals: list[float],
                           hd_residuals: list[float]) -> None:
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if t0_residuals:
        hist = axis.hist2d(t0_residuals, hd_residuals, bins=(100, 100),
                           range=((-1000, 1000), (-1000, 1000)), cmap="viridis",
                           norm=LogNorm(vmin=1))
        fig.colorbar(hist[3], ax=axis, label="Pairs / bin")
    else:
        axis.text(0.5, 0.5, "No delayed PMT–later-HD pairs",
                  transform=axis.transAxes, ha="center", va="center")
    axis.axvline(0, color="white", linewidth=0.8, alpha=0.8)
    axis.axhline(0, color="white", linewidth=0.8, alpha=0.8)
    axis.set(xlabel="PMT delay after prompt − later-T0 group gap (ns)",
             ylabel="(delayed PMT − later HD) − (prompt PMT − prompt HD) (ns)",
             title="Later-bunch / HD timing consistency (diagnostic only)")
    axis.grid(alpha=0.15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_t5_native_times(path: Path, t5_times: list[float], available: bool) -> None:
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if t5_times:
        axis.hist(t5_times, bins=150, histtype="step", color="tab:red")
    else:
        message = "No T5 times in selected windows" if available else "ROOT tree has no T5_hit_time branch"
        axis.text(0.5, 0.5, message, transform=axis.transAxes,
                  ha="center", va="center")
    axis.set(xlabel="T5_hit_time (native branch units; no alignment applied)",
             ylabel="Hits / bin", title="T5 hit-time distribution — native values only")
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_diagnostic_veto_comparison(output_dir: Path, prompt_residuals: dict[int, list[float]],
                                    eligible_prompt_windows: int, *,
                                    t0_window_ns: float, pmt_hd_window_ns: float,
                                    bin_ns: float, delay_min_ns: float,
                                    delay_max_ns: float) -> dict:
    """Compare delayed candidates with a configurable later-T0+HD diagnostic veto."""
    prompt_medians = {hd_id: float(np.median(values))
                      for hd_id, values in prompt_residuals.items() if values}
    vetoed: set[tuple[str, str, str, str]] = set()
    pair_count = 0
    with (output_dir / "pmt_hd_timing_pairs.csv").open(
            newline="", encoding="utf-8") as pair_file:
        for row in csv.DictReader(pair_file):
            if row["pair_kind"] != "delayed_candidate_later_t0_hd":
                continue
            hd_id = int(row["hd_element"])
            if hd_id not in prompt_medians:
                continue
            bunch_residual = float(row["pmt_delay_minus_t0_gap_ns"])
            hd_residual = float(row["pmt_minus_hd_ns"]) - prompt_medians[hd_id]
            if abs(bunch_residual) <= t0_window_ns and abs(hd_residual) <= pmt_hd_window_ns:
                key = (row["root_entry"], row["readout_number"],
                       row["event_number"], row["burst_index"])
                if key not in vetoed:
                    vetoed.add(key)
                    pair_count += 1

    base_path = output_dir / "delayed_candidates_no_veto.csv"
    all_delays: list[float] = []
    vetoed_delays: list[float] = []
    with base_path.open(newline="", encoding="utf-8") as source, \
            (output_dir / "delayed_candidates_veto_comparison.csv").open(
                "w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=(*reader.fieldnames, "veto_by_later_t0_hd"))
        writer.writeheader()
        for row in reader:
            delay = float(row["delay_after_prompt_ns"])
            key = (row["root_entry"], row["readout_number"],
                   row["event_number"], row["burst_index"])
            is_vetoed = key in vetoed
            row["veto_by_later_t0_hd"] = int(is_vetoed)
            writer.writerow(row)
            all_delays.append(delay)
            if is_vetoed:
                vetoed_delays.append(delay)
    edges = np.arange(delay_min_ns, delay_max_ns + bin_ns, bin_ns)
    total = np.histogram(all_delays, bins=edges)[0]
    removed = np.histogram(vetoed_delays, bins=edges)[0]
    retained = total - removed
    denominator = max(eligible_prompt_windows, 1)
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for values, label, color in (
            (total, f"No veto ({len(all_delays):,} candidates)", "black"),
            (removed, f"HD+T0 matched ({len(vetoed_delays):,})", "tab:red"),
            (retained, f"Remaining ({len(all_delays) - len(vetoed_delays):,})", "tab:blue")):
        axis.stairs(values / denominator, edges, label=label, color=color, linewidth=1.6)
    axis.set(xlabel="Delayed PMT burst time − unique prompt burst time (ns)",
             ylabel="Candidates / prompt-tagged readout / bin",
             title=("No-veto vs diagnostic candidate-level later-T0+HD veto\n"
                    f"Both timing residuals within ±{t0_window_ns:g}/±{pmt_hd_window_ns:g} ns"))
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    fig.savefig(output_dir / "delayed_candidate_no_veto_vs_late_hd_veto.png", dpi=170)
    plt.close(fig)
    return {"eligible_prompt_tagged_readouts": eligible_prompt_windows,
            "no_veto_delayed_candidates": len(all_delays),
            "vetoed_candidates": len(vetoed_delays),
            "remaining_candidates": len(all_delays) - len(vetoed_delays),
            "candidate_veto_fraction": len(vetoed_delays) / len(all_delays) if all_delays else None,
            "matching_later_t0_hd_pairs_that_vetoed_candidates": pair_count,
            "t0_gap_match_window_ns": t0_window_ns,
            "pmt_hd_residual_match_window_ns": pmt_hd_window_ns,
            "prompt_median_pmt_minus_hd_ns_by_element": {
                str(hd_id): median for hd_id, median in prompt_medians.items()},
            "definition": "Diagnostic only: a delayed PMT burst is marked if any later T0+HD match has PMT-delay minus T0-gap and prompt-median-corrected PMT-HD residuals inside the configured windows. No cut is applied to the source sample."}


def run(args: argparse.Namespace) -> dict:
    tdc_mode = getattr(args, "tdc_time_mode", "reference")
    prompt_min = getattr(args, "prompt_min_ns", 1500)
    prompt_max = getattr(args, "prompt_max_ns", 1900)
    delayed_min = getattr(args, "delayed_min_ns", 500)
    delayed_max = getattr(args, "delayed_max_ns", 7500)
    veto_t0_window = getattr(args, "veto_t0_window_ns", 100)
    veto_pmt_hd_window = getattr(args, "veto_pmt_hd_window_ns", 100)
    veto_bin = getattr(args, "veto_bin_ns", 25)
    if prompt_max <= prompt_min:
        raise ValueError("Prompt maximum must exceed prompt minimum")
    if delayed_max <= delayed_min or min(veto_t0_window, veto_pmt_hd_window, veto_bin) <= 0:
        raise ValueError("Require positive diagnostic windows and delayed range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(0, args.max_ns - args.min_ns + args.hist_bin_ns, args.hist_bin_ns)
    histograms = {(kind, group): np.zeros(len(edges) - 1, dtype=np.int64)
                  for kind in ("consecutive", "later_from_strongest")
                  for group in ("one_t0_group", "multiple_t0_groups", "no_t0")}
    t0_hist = np.zeros(len(edges) - 1, dtype=np.int64)
    hd_edges = np.arange(-500, max(args.max_ns, delayed_max) + 501,
                         max(args.hist_bin_ns, 10))
    hd_histograms = {hd_id: np.zeros(len(hd_edges) - 1, dtype=np.int64) for hd_id in range(15)}
    prompt_pmt_hd_by_element = {hd_id: [] for hd_id in range(15)}
    delayed_pmt_hd_by_element = {hd_id: [] for hd_id in range(15)}
    delayed_relative_by_element = {hd_id: [] for hd_id in range(15)}
    delayed_t0_residuals: list[float] = []
    delayed_hd_residuals: list[float] = []
    t5_native_times: list[float] = []
    eligible_prompt_windows = 0
    counts: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    alignment_candidates: list[float] = []
    with uproot.open(args.input_root) as root:
        tree = root["WCTEReadoutWindows"]
        missing = set(REQUIRED) - set(tree.keys())
        if missing:
            raise ValueError("Missing ROOT branches: " + ", ".join(sorted(missing)))
        stop = min(tree.num_entries, args.entry_start + args.scan_windows)
        has_t5 = "T5_hit_time" in tree.keys()
        branches = list(REQUIRED) + (["T5_hit_time"] if has_t5 else [])
        with (args.output_dir / "bursts.csv").open("w", newline="", encoding="utf-8") as burst_file, \
             (args.output_dir / "gaps.csv").open("w", newline="", encoding="utf-8") as gap_file, \
             (args.output_dir / "t0_group_gaps.csv").open("w", newline="", encoding="utf-8") as t0_file, \
             (args.output_dir / "alignment_candidates.csv").open("w", newline="", encoding="utf-8") as align_file, \
             (args.output_dir / "hodoscope_hits.csv").open("w", newline="", encoding="utf-8") as hd_file, \
             (args.output_dir / "pmt_hd_timing_pairs.csv").open("w", newline="", encoding="utf-8") as pair_file, \
             (args.output_dir / "delayed_candidates_no_veto.csv").open("w", newline="", encoding="utf-8") as candidate_file, \
             (args.output_dir / "t5_hit_times.csv").open("w", newline="", encoding="utf-8") as t5_file:
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
            hd_writer = csv.DictWriter(hd_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "channel_id",
                "hd_element", "corrected_tdc_time_ns", "earliest_t0_group_ns",
                "time_from_earliest_t0_ns"))
            pair_writer = csv.DictWriter(pair_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "pair_kind",
                "burst_index", "hd_element", "t0_group_index", "pmt_burst_ns", "hd_time_ns",
                "pmt_minus_hd_ns", "prompt_t0_ns", "matched_t0_ns",
                "pmt_delay_from_prompt_ns", "t0_gap_from_prompt_ns",
                "pmt_delay_minus_t0_gap_ns", "residual_relative_to_prompt_same_hd_ns"))
            candidate_writer = csv.DictWriter(candidate_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "burst_index",
                "prompt_burst_ns", "candidate_burst_ns", "delay_after_prompt_ns",
                "candidate_n_hits", "candidate_n_pmts", "candidate_charge"))
            t5_writer = csv.DictWriter(t5_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "t5_hit_index",
                "t5_hit_time_native_units"))
            burst_writer.writeheader()
            gap_writer.writeheader()
            t0_writer.writeheader()
            align_writer.writeheader()
            hd_writer.writeheader()
            pair_writer.writeheader()
            candidate_writer.writeheader()
            t5_writer.writeheader()
            for start in range(args.entry_start, stop, args.batch_windows):
                batch = tree.arrays(branches, entry_start=start,
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
                    if has_t5:
                        for t5_index, value in enumerate(ak.to_list(event["T5_hit_time"])):
                            if value is None or not np.isfinite(float(value)):
                                continue
                            value = float(value)
                            t5_native_times.append(value)
                            t5_writer.writerow({**{key: identity[key] for key in (
                                "root_entry", "run_id", "event_number", "readout_number")},
                                "t5_hit_index": t5_index, "t5_hit_time_native_units": value})
                    beam_hits, _ = corrected_beamline_hits(event, tdc_time_mode=tdc_mode)
                    earliest_t0 = t0_centers[0] if t0_centers else None
                    for channel_id, hit_time in beam_hits:
                        if channel_id not in HD_IDS:
                            continue
                        hd_id = HD_IDS[channel_id]
                        relative_time = hit_time - earliest_t0 if earliest_t0 is not None else None
                        hd_writer.writerow({**{key: identity[key] for key in (
                            "root_entry", "run_id", "event_number", "readout_number")},
                            "channel_id": channel_id, "hd_element": hd_id,
                            "corrected_tdc_time_ns": hit_time,
                            "earliest_t0_group_ns": earliest_t0,
                            "time_from_earliest_t0_ns": relative_time})
                        if relative_time is not None:
                            hd_histograms[hd_id] += np.histogram([relative_time], bins=hd_edges)[0]
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
                    if len(prompt_bursts) == 1 and n_t0:
                        prompt_burst = prompt_bursts[0]
                        prompt_t0 = t0_centers[0]
                        prompt_hd_by_id = {}
                        for match in hd_matches_to_group(beam_hits, prompt_t0):
                            channel_id, hd_id, hd_time, dt = match
                            if (hd_id not in prompt_hd_by_id
                                    or abs(dt) < abs(prompt_hd_by_id[hd_id][3])):
                                prompt_hd_by_id[hd_id] = match
                        if not prompt_hd_by_id:
                            counts["unique_prompt_pmt_without_earliest_t0_hd_tag"] += 1
                        for hd_id, (channel_id, _, hd_time, _) in prompt_hd_by_id.items():
                            prompt_residual = prompt_burst.center_ns - hd_time
                            prompt_pmt_hd_by_element[hd_id].append(prompt_residual)
                            pair_writer.writerow({**{key: identity[key] for key in (
                                "root_entry", "run_id", "event_number", "readout_number")},
                                "pair_kind": "prompt_reference", "burst_index": bursts.index(prompt_burst),
                                "hd_element": hd_id, "t0_group_index": 0,
                                "pmt_burst_ns": prompt_burst.center_ns, "hd_time_ns": hd_time,
                                "pmt_minus_hd_ns": prompt_residual, "prompt_t0_ns": prompt_t0,
                                "matched_t0_ns": prompt_t0, "pmt_delay_from_prompt_ns": 0,
                                "t0_gap_from_prompt_ns": 0, "pmt_delay_minus_t0_gap_ns": 0,
                                "residual_relative_to_prompt_same_hd_ns": 0})
                        later_groups = []
                        if prompt_hd_by_id:
                            eligible_prompt_windows += 1
                            for burst_index, burst in enumerate(bursts):
                                delay = burst.center_ns - prompt_burst.center_ns
                                if not delayed_min <= delay <= delayed_max:
                                    continue
                                candidate_writer.writerow({**{key: identity[key] for key in (
                                    "root_entry", "run_id", "event_number", "readout_number")},
                                    "burst_index": burst_index,
                                    "prompt_burst_ns": prompt_burst.center_ns,
                                    "candidate_burst_ns": burst.center_ns,
                                    "delay_after_prompt_ns": delay,
                                    "candidate_n_hits": burst.n_hits,
                                    "candidate_n_pmts": burst.n_pmts,
                                    "candidate_charge": burst.charge})
                            for group_index, group_t0 in enumerate(t0_centers[1:], start=1):
                                later_groups.append((group_index, group_t0,
                                                     hd_matches_to_group(beam_hits, group_t0)))
                        for burst in bursts:
                            delay = burst.center_ns - prompt_burst.center_ns
                            if not delayed_min <= delay <= delayed_max:
                                continue
                            for group_index, group_t0, late_matches in later_groups:
                                t0_gap = group_t0 - prompt_t0
                                delay_minus_t0 = delay - t0_gap
                                for _, hd_id, hd_time, _ in late_matches:
                                    direct_residual = burst.center_ns - hd_time
                                    delayed_pmt_hd_by_element[hd_id].append(direct_residual)
                                    delayed_hd_residual = None
                                    if hd_id in prompt_hd_by_id:
                                        prompt_hd = prompt_hd_by_id[hd_id][2]
                                        delayed_hd_residual = direct_residual - (
                                            prompt_burst.center_ns - prompt_hd)
                                        delayed_relative_by_element[hd_id].append(delayed_hd_residual)
                                        delayed_t0_residuals.append(delay_minus_t0)
                                        delayed_hd_residuals.append(delayed_hd_residual)
                                    pair_writer.writerow({**{key: identity[key] for key in (
                                        "root_entry", "run_id", "event_number", "readout_number")},
                                        "pair_kind": "delayed_candidate_later_t0_hd",
                                        "burst_index": bursts.index(burst), "hd_element": hd_id,
                                        "t0_group_index": group_index,
                                        "pmt_burst_ns": burst.center_ns, "hd_time_ns": hd_time,
                                        "pmt_minus_hd_ns": direct_residual,
                                        "prompt_t0_ns": prompt_t0, "matched_t0_ns": group_t0,
                                        "pmt_delay_from_prompt_ns": delay,
                                        "t0_gap_from_prompt_ns": t0_gap,
                                        "pmt_delay_minus_t0_gap_ns": delay_minus_t0,
                                        "residual_relative_to_prompt_same_hd_ns":
                                            delayed_hd_residual if delayed_hd_residual is not None else ""})
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
                "hodoscope_hit_counts_by_element": {
                    str(hd_id): int(values.sum()) for hd_id, values in hd_histograms.items()},
                "hodoscope_time_origin": "Corrected HD times are relative to the earliest T0 group in each readout; this is not an alignment to the WCTE PMT clock.",
                "prompt_pmt_hd_pair_count": int(sum(map(len, prompt_pmt_hd_by_element.values()))),
                "delayed_pmt_later_hd_pair_count": int(sum(map(len, delayed_pmt_hd_by_element.values()))),
                "delayed_same_hd_prompt_relative_pair_count": int(sum(map(len, delayed_relative_by_element.values()))),
                "t5_branch_present": has_t5,
                "t5_hit_count": len(t5_native_times),
                "t5_time_units": "native ROOT branch units; not aligned or assumed to be ns",
                "delayed_candidate_window_ns": [delayed_min, delayed_max],
                "diagnostic_veto_comparison": None,
            }
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["diagnostic_veto_comparison"] = make_diagnostic_veto_comparison(
        args.output_dir, prompt_pmt_hd_by_element, eligible_prompt_windows,
        t0_window_ns=veto_t0_window, pmt_hd_window_ns=veto_pmt_hd_window,
        bin_ns=veto_bin, delay_min_ns=delayed_min, delay_max_ns=delayed_max)
    summary["settings"].update({
        "delayed_min_ns": delayed_min, "delayed_max_ns": delayed_max,
        "veto_t0_window_ns": veto_t0_window,
        "veto_pmt_hd_window_ns": veto_pmt_hd_window,
        "veto_bin_ns": veto_bin})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_histograms(args.output_dir / "burst_gap_histograms.png", histograms, edges,
                    counts["selected_windows"], counts["windows_with_multiple_bursts"])
    pmt_consecutive = sum((histograms[("consecutive", group)] for group in (
        "one_t0_group", "multiple_t0_groups", "no_t0")), np.zeros(len(edges) - 1, dtype=np.int64))
    plot_t0_comparison(args.output_dir / "t0_vs_pmt_gap_histograms.png",
                       t0_hist, pmt_consecutive, edges)
    plot_hodoscope_times(args.output_dir / "hodoscope_hit_times.png", hd_histograms,
                         hd_edges, tdc_mode)
    plot_hd_residuals(args.output_dir / "prompt_pmt_hd_residuals_by_element.png",
                      prompt_pmt_hd_by_element,
                      title="Prompt PMT burst time − earliest-T0-matched prompt HD time",
                      xlabel="Prompt PMT − prompt HD time (ns)",
                      limits=(-500, max(args.max_ns, prompt_max) + 500))
    plot_hd_residuals(args.output_dir / "delayed_pmt_later_hd_residuals_by_element.png",
                      delayed_pmt_hd_by_element,
                      title="Delayed PMT burst time − later-T0-matched HD time",
                      xlabel="Delayed PMT − later HD time (ns)",
                      limits=(-500, max(args.max_ns, delayed_max) + 500))
    plot_hd_residuals(args.output_dir / "delayed_pmt_hd_residuals_relative_prompt.png",
                      delayed_relative_by_element,
                      title="Delayed PMT–HD residual minus same-event prompt residual",
                      xlabel="Delayed-minus-prompt PMT–HD residual (ns)",
                      limits=(-1500, 1500))
    plot_delayed_match_map(args.output_dir / "delayed_pmt_t0_hd_residual_map.png",
                           delayed_t0_residuals, delayed_hd_residuals)
    plot_t5_native_times(args.output_dir / "t5_hit_times_native.png",
                         t5_native_times, has_t5)
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
    veto = summary["diagnostic_veto_comparison"]
    print("HD/WCTE timing diagnostics:")
    for filename in (
            "prompt_pmt_hd_residuals_by_element.png",
            "delayed_pmt_later_hd_residuals_by_element.png",
            "delayed_pmt_hd_residuals_relative_prompt.png",
            "delayed_pmt_t0_hd_residual_map.png",
            "delayed_candidate_no_veto_vs_late_hd_veto.png"):
        print(f"  {args.output_dir / filename}")
    print("Diagnostic later-T0+HD comparison (not an applied/validated cut): "
          f"{veto['no_veto_delayed_candidates']:,} no-veto candidates; "
          f"{veto['vetoed_candidates']:,} matched; "
          f"{veto['remaining_candidates']:,} retained; "
          f"{veto['eligible_prompt_tagged_readouts']:,} prompt-tagged readouts")
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
    parser.add_argument("--delayed-min-ns", type=float, default=500,
                        help="Minimum delayed PMT burst delay after the unique prompt burst")
    parser.add_argument("--delayed-max-ns", type=float, default=7500,
                        help="Maximum delayed PMT burst delay after the unique prompt burst")
    parser.add_argument("--veto-t0-window-ns", type=float, default=100,
                        help="Exploratory tolerance for matching PMT delay to a later T0-group gap")
    parser.add_argument("--veto-pmt-hd-window-ns", type=float, default=100,
                        help="Exploratory tolerance around the prompt PMT-HD residual for that HD element")
    parser.add_argument("--veto-bin-ns", type=float, default=25,
                        help="Histogram bin width for no-veto/vetoed/retained comparison")
    args = parser.parse_args()
    if (args.entry_start < 0 or args.scan_windows < 1 or args.batch_windows < 1
            or args.max_ns <= args.min_ns or args.min_ns < 0 or args.bin_ns <= 0
            or args.width_ns <= 0 or args.min_pmts < 1 or args.min_separation_ns <= 0
            or args.max_bursts < 1 or args.hist_bin_ns <= 0):
        parser.error("Require nonnegative entry/time starts and positive window, bin, threshold and scan settings")
    if args.prompt_max_ns <= args.prompt_min_ns:
        parser.error("Prompt maximum must exceed prompt minimum")
    if args.delayed_max_ns <= args.delayed_min_ns:
        parser.error("Delayed maximum must exceed delayed minimum")
    if min(args.veto_t0_window_ns, args.veto_pmt_hd_window_ns, args.veto_bin_ns) <= 0:
        parser.error("Diagnostic veto windows and histogram bin width must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
