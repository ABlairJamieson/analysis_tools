#!/usr/bin/env python3
"""Inspect WCTE PMT and tagged-gamma beamline timing in ROOT readout windows.

Beamline TDC, WCTE PMT, and T5 times are shown on separate native time axes.
No cross-system clock offset or one-particle-per-window assumption is made.
All repeated beamline TDC hits are retained in beam_hits.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


T0_IDS = {0, 1, 2, 3}
T2_ID = 8
HC2_ID = 11
HD_NAMES = {**{32 + i: f"HD{i}" for i in range(7)}, 16: "HD7",
            **{17 + i: f"HD{i + 8}" for i in range(7)}}
REQUIRED = ("run_id", "event_number", "readout_number", "window_data_quality_mask",
            "hit_pmt_calibrated_times", "hit_pmt_charges", "hit_pmt_readout_mask",
            "beamline_pmt_tdc_ids", "beamline_pmt_tdc_times")


def hd_window(channel_id: int) -> tuple[float, float]:
    """T0-relative HD windows from compute_tagging_efficiency_TG.py."""
    name = HD_NAMES[channel_id]
    number = int(name[2:])
    return (-50.0, 100.0) if number in (0, 1, 2, 3, 8, 9, 10, 11) else (80.0, 200.0)


def beam_hits(event) -> list[tuple[int, float]]:
    ids = ak.to_list(event["beamline_pmt_tdc_ids"])
    times = ak.to_list(event["beamline_pmt_tdc_times"])
    if len(ids) != len(times):
        raise ValueError("Beamline TDC ID/time lengths differ")
    return [(int(channel), float(time)) for channel, time in zip(ids, times)
            if time is not None and np.isfinite(float(time))]


def tag_matches(hits: list[tuple[int, float]]) -> list[tuple[int, float, float]]:
    """Every in-time (HD channel, HD time, T0 time) combination, including repeats."""
    t0_times = [time for channel, time in hits if channel in T0_IDS]
    matches = []
    for channel, hd_time in hits:
        if channel not in HD_NAMES:
            continue
        low, high = hd_window(channel)
        matches.extend((channel, hd_time, t0) for t0 in t0_times
                       if low < hd_time - t0 < high)
    return matches


def peak_bins(times: np.ndarray, low: float, high: float, width: float,
              min_hits: int) -> list[tuple[float, int]]:
    """Local high-count PMT time bins; visual aids, not reconstructed particles."""
    edges = np.arange(low, high + width, width)
    counts, _ = np.histogram(times, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    candidates = np.flatnonzero((counts >= min_hits)
                                & (counts >= np.r_[0, counts[:-1]])
                                & (counts > np.r_[counts[1:], 0]))
    chosen = []
    for index in sorted(candidates, key=lambda i: counts[i], reverse=True):
        if all(abs(centers[index] - centers[other]) >= 50 for other in chosen):
            chosen.append(index)
    return sorted(((float(centers[i]), int(counts[i])) for i in chosen), key=lambda item: item[0])


def draw_window(path: Path, pmt_times: np.ndarray, pmt_charges: np.ndarray,
                hits: list[tuple[int, float]], t5_times: list[float],
                peaks: list[tuple[float, int]], *, pmt_min: float, pmt_max: float,
                bin_ns: float, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    edges = np.arange(pmt_min, pmt_max + bin_ns, bin_ns)
    axes[0].hist(pmt_times, bins=edges, histtype="step", color="navy", label="PMT hits")
    for time, count in peaks:
        axes[0].axvline(time, color="tab:orange", alpha=0.45, linewidth=0.8)
        axes[0].annotate(f"{time:g}", (time, count), rotation=90, fontsize=7)
    axes[0].set(xlabel="Calibrated WCTE PMT time from readout-window start (ns)",
                ylabel="Hits / bin", xlim=(pmt_min, pmt_max))
    axes[0].set_title(f"{len(pmt_times)} quality-kept PMT hits; charge sum {pmt_charges.sum():.0f} input units", loc="left")

    t0_times = [time for channel, time in hits if channel in T0_IDS]
    t0_ref = min(t0_times) if t0_times else None
    labels = ["T0", "T2", "HC2", *[f"HD{i}" for i in range(15)], "Other"]
    colors = {"T0": "tab:blue", "T2": "tab:green", "HC2": "tab:red", "Other": "gray"}
    for channel, time in hits:
        label = ("T0" if channel in T0_IDS else "T2" if channel == T2_ID else
                 "HC2" if channel == HC2_ID else HD_NAMES.get(channel, "Other"))
        axes[1].scatter(time - t0_ref if t0_ref is not None else time,
                        labels.index(label), color=colors.get(label, "tab:purple"), s=35)
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set(xlabel="Beamline TDC time relative to earliest T0 hit (ns)"
                if t0_ref is not None else "Beamline TDC native time (ns)",
                ylabel="Beamline channel")
    axes[1].grid(axis="x", alpha=0.25)
    if t0_ref is not None:
        axes[1].axvline(0, color="black", linewidth=0.8)

    if t5_times:
        axes[2].eventplot(t5_times, lineoffsets=0, linelengths=0.8, color="tab:red")
        axes[2].set_ylim(-1, 1)
    else:
        axes[2].text(0.5, 0.5, "No T5 hit times", transform=axes[2].transAxes,
                     ha="center", va="center")
    axes[2].set(xlabel="T5 hit time (native branch units; NOT aligned to panels above)",
                yticks=[])
    fig.suptitle(title + " | panels have independent time origins", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[int, int]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with uproot.open(args.input_root) as root:
        tree = root["WCTEReadoutWindows"]
        available = set(tree.keys())
        missing = set(REQUIRED) - available
        if missing:
            raise ValueError("Missing ROOT branches: " + ", ".join(sorted(missing)))
        branches = [*REQUIRED, *(name for name in ("T5_hit_time",) if name in available)]
        stop = min(tree.num_entries, args.entry_start + args.scan_windows)
        wanted = set(args.readout_numbers or ())
        scanned = plotted = 0
        with (args.output_dir / "window_summary.csv").open("w", newline="", encoding="utf-8") as summary_file, \
             (args.output_dir / "beam_hits.csv").open("w", newline="", encoding="utf-8") as hits_file, \
             (args.output_dir / "pmt_peaks.csv").open("w", newline="", encoding="utf-8") as peaks_file:
            summary = csv.DictWriter(summary_file, fieldnames=(
                "root_entry", "run_id", "event_number", "readout_number", "n_pmt_hits",
                "n_pmt_peaks", "n_t0_hits", "n_t2_hits", "n_hd_t0_matches", "n_t5_hits", "plot"))
            beam = csv.DictWriter(hits_file, fieldnames=(
                "root_entry", "readout_number", "channel_id", "channel", "tdc_time",
                "dt_earliest_t0_ns", "matched_t0_times"))
            peak_writer = csv.DictWriter(peaks_file, fieldnames=(
                "root_entry", "readout_number", "peak_bin_center_ns", "hits_in_bin"))
            for writer in (summary, beam, peak_writer):
                writer.writeheader()
            for start in range(args.entry_start, stop, args.batch_windows):
                batch = tree.arrays(branches, entry_start=start,
                                    entry_stop=min(start + args.batch_windows, stop), library="ak")
                for offset, event in enumerate(batch):
                    scanned += 1
                    entry = start + offset
                    readout = int(event["readout_number"])
                    if wanted and readout not in wanted:
                        continue
                    if int(event["window_data_quality_mask"]) != 0:
                        continue
                    hits = beam_hits(event)
                    ids = [channel for channel, _ in hits]
                    matches = tag_matches(hits)
                    beam_ok = any(cid in T0_IDS for cid in ids) and T2_ID in ids and HC2_ID not in ids
                    if args.selection != "all" and not beam_ok:
                        continue
                    if args.selection == "tagged" and not matches:
                        continue
                    mask = np.asarray(ak.to_numpy(event["hit_pmt_readout_mask"])) == 0
                    times = np.asarray(ak.to_numpy(event["hit_pmt_calibrated_times"]), dtype=float)[mask]
                    charges = np.asarray(ak.to_numpy(event["hit_pmt_charges"]), dtype=float)[mask]
                    valid = np.isfinite(times) & np.isfinite(charges)
                    times, charges = times[valid], charges[valid]
                    peaks = peak_bins(times, args.pmt_min_ns, args.pmt_max_ns,
                                      args.bin_ns, args.min_peak_hits)
                    t5 = ([float(t) for t in ak.to_list(event["T5_hit_time"])
                           if t is not None and np.isfinite(float(t))]
                          if "T5_hit_time" in available else [])
                    image_name = f"entry{entry:08d}_readout{readout}.png"
                    title = f"Run {int(event['run_id'])}, ROOT entry {entry}, event {int(event['event_number'])}, readout {readout}"
                    draw_window(args.output_dir / image_name, times, charges, hits, t5, peaks,
                                pmt_min=args.pmt_min_ns, pmt_max=args.pmt_max_ns,
                                bin_ns=args.bin_ns, title=title)
                    summary.writerow({"root_entry": entry, "run_id": int(event["run_id"]),
                                     "event_number": int(event["event_number"]), "readout_number": readout,
                                     "n_pmt_hits": len(times), "n_pmt_peaks": len(peaks),
                                     "n_t0_hits": sum(cid in T0_IDS for cid in ids),
                                     "n_t2_hits": ids.count(T2_ID), "n_hd_t0_matches": len(matches),
                                     "n_t5_hits": len(t5), "plot": image_name})
                    t0_times = [time for cid, time in hits if cid in T0_IDS]
                    t0_ref = min(t0_times) if t0_times else None
                    for cid, time in hits:
                        label = ("T0" if cid in T0_IDS else "T2" if cid == T2_ID else
                                 "HC2" if cid == HC2_ID else HD_NAMES.get(cid, "Other"))
                        matched = [t0 for hd_id, hd_time, t0 in matches if hd_id == cid and hd_time == time]
                        beam.writerow({"root_entry": entry, "readout_number": readout,
                                       "channel_id": cid, "channel": label, "tdc_time": time,
                                       "dt_earliest_t0_ns": time - t0_ref if t0_ref is not None else "",
                                       "matched_t0_times": ";".join(map(str, matched))})
                    for time, count in peaks:
                        peak_writer.writerow({"root_entry": entry, "readout_number": readout,
                                              "peak_bin_center_ns": time, "hits_in_bin": count})
                    plotted += 1
                    if plotted >= args.max_plots:
                        print(f"Scanned {scanned} ROOT windows; plotted {plotted}: {args.output_dir}")
                        return scanned, plotted
    print(f"Scanned {scanned} ROOT windows; plotted {plotted}: {args.output_dir}")
    return scanned, plotted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--scan-windows", type=int, default=10_000)
    parser.add_argument("--batch-windows", type=int, default=200)
    parser.add_argument("--max-plots", type=int, default=20)
    parser.add_argument("--readout-numbers", type=int, nargs="*", help="Specific readout numbers")
    parser.add_argument("--selection", choices=("all", "beam", "tagged"), default="tagged",
                        help="tagged = T0 + T2, no HC2, and an HD hit in-time to a T0 hit")
    parser.add_argument("--pmt-min-ns", type=float, default=0)
    parser.add_argument("--pmt-max-ns", type=float, default=10_000)
    parser.add_argument("--bin-ns", type=float, default=10)
    parser.add_argument("--min-peak-hits", type=int, default=10)
    args = parser.parse_args()
    if (args.entry_start < 0 or args.scan_windows < 1 or args.batch_windows < 1
            or args.max_plots < 1 or args.min_peak_hits < 1 or args.bin_ns <= 0
            or args.pmt_max_ns <= args.pmt_min_ns):
        parser.error("Require positive scan/plot/bin settings and PMT max > min")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
