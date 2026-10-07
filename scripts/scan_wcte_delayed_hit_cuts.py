#!/usr/bin/env python3
"""Scan delayed PMT burst hit-count ceilings in an existing WCTE burst CSV.

The output compares prompt-relative delayed-burst rates for several inclusive
upper hit-count cuts. It is an exploratory beam-bunch/Michel study, not a
positron tag: ``n_hits`` is the digit-hit count in the burst finder's local
time window (50 ns by default), not necessarily the total hits in a reconstructed
physics cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.fit_wcte_delayed_bursts import fit_models, phase_template
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from fit_wcte_delayed_bursts import fit_models, phase_template


REQUIRED = {"root_entry", "run_id", "readout_number", "center_ns", "n_hits", "n_pmts"}


def read_prompt_delayed(path: Path, *, prompt_min_ns: float, prompt_max_ns: float,
                        prompt_min_pmts: int, delay_min_ns: float, delay_max_ns: float,
                        delayed_min_pmts: int, delayed_min_hits: int,
                        root_entry_start: int | None, root_entry_stop: int | None
                        ) -> tuple[list[dict], dict]:
    """Return delayed bursts and event-level later-hodoscope match annotations."""
    delayed = []
    counts = {"input_windows_with_bursts": 0, "windows_without_unique_prompt": 0,
              "unique_prompt_windows": 0, "windows_with_delayed_candidates": 0,
              "delayed_candidates_before_upper_hit_cut": 0}

    def process_window(bursts: list[dict]) -> None:
        if not bursts:
            return
        counts["input_windows_with_bursts"] += 1
        prompts = [b for b in bursts if prompt_min_ns <= b["center_ns"] < prompt_max_ns
                   and b["n_pmts"] >= prompt_min_pmts]
        if len(prompts) != 1:
            counts["windows_without_unique_prompt"] += 1
            return
        counts["unique_prompt_windows"] += 1
        prompt = prompts[0]
        found = 0
        for burst in bursts:
            delay = burst["center_ns"] - prompt["center_ns"]
            if (delay_min_ns <= delay < delay_max_ns
                    and burst["n_pmts"] >= delayed_min_pmts
                    and burst["n_hits"] >= delayed_min_hits):
                delayed.append({**burst, "prompt_ns": prompt["center_ns"],
                                "delay_ns": delay})
                found += 1
        if found:
            counts["windows_with_delayed_candidates"] += 1

    current_key = None
    current_bursts: list[dict] = []
    previous_entry = -1
    has_late_hd_info = False
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"{path} is not a study_wcte_burst_gaps.py bursts.csv")
        has_late_hd_info = {"n_late_hd_hits", "n_late_hd_groups"}.issubset(reader.fieldnames)
        for row in reader:
            entry = int(row["root_entry"])
            if entry < previous_entry:
                raise ValueError(f"{path} must be ordered by ROOT entry")
            previous_entry = entry
            key = (row["run_id"], row["root_entry"], row["readout_number"])
            if current_key is not None and key != current_key:
                process_window(current_bursts)
                current_bursts = []
            current_key = key
            if root_entry_start is not None and entry < root_entry_start:
                continue
            if root_entry_stop is not None and entry >= root_entry_stop:
                continue
            center = float(row["center_ns"])
            if not np.isfinite(center):
                continue
            current_bursts.append({
                "root_entry": entry,
                "run_id": int(row["run_id"]),
                "readout_number": int(row["readout_number"]),
                "center_ns": center,
                "n_hits": int(row["n_hits"]),
                "n_pmts": int(row["n_pmts"]),
                "n_late_hd_hits": (int(row["n_late_hd_hits"])
                                    if has_late_hd_info else None),
                "n_late_hd_groups": (int(row["n_late_hd_groups"])
                                     if has_late_hd_info else None),
            })
    process_window(current_bursts)
    counts["delayed_candidates_before_upper_hit_cut"] = len(delayed)
    counts["later_hodoscope_columns_available"] = has_late_hd_info
    return delayed, counts


def run(args: argparse.Namespace) -> dict:
    if not (0 < args.prompt_min_ns < args.prompt_max_ns
            and 0 <= args.min_delay_ns < args.max_delay_ns
            and args.bin_ns > 0 and args.period_ns > 0 and args.lifetime_ns > 0
            and args.phase_bins >= 3 and args.min_prompt_pmts > 0
            and args.min_delayed_pmts > 0 and args.min_delayed_hits >= 0):
        raise ValueError("Invalid timing, bin, period, lifetime, or hit-count settings")
    cuts = sorted(set(args.max_delayed_hits))
    if not cuts or any(cut < 1 for cut in cuts):
        raise ValueError("At least one positive --max-delayed-hits value is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bursts, counts = read_prompt_delayed(
        args.bursts_csv, prompt_min_ns=args.prompt_min_ns,
        prompt_max_ns=args.prompt_max_ns, prompt_min_pmts=args.min_prompt_pmts,
        delay_min_ns=args.min_delay_ns, delay_max_ns=args.max_delay_ns,
        delayed_min_pmts=args.min_delayed_pmts, delayed_min_hits=args.min_delayed_hits,
        root_entry_start=args.root_entry_start, root_entry_stop=args.root_entry_stop)
    if not bursts:
        raise ValueError("No delayed candidates; check prompt/delay and minimum-hit settings")

    edges = np.arange(args.min_delay_ns, args.max_delay_ns + args.bin_ns / 2,
                      args.bin_ns)
    if len(edges) < 4 or not np.isclose(edges[-1], args.max_delay_ns):
        raise ValueError("Delay range must contain at least 3 whole, equal-width bins")
    results = []
    histograms: dict[int, np.ndarray] = {}
    for cut in cuts:
        selected = [b for b in bursts if b["n_hits"] <= cut]
        delays = np.asarray([b["delay_ns"] for b in selected])
        histograms[cut] = np.histogram(delays, bins=edges)[0]
        fit = None
        fit_error = None
        sideband = delays[(delays >= args.template_min_ns)
                          & (delays < args.template_max_ns)]
        if len(delays) and len(sideband) >= args.min_template_bursts:
            try:
                phase = phase_template(sideband, period_ns=args.period_ns,
                                       phase_bins=args.phase_bins)
                fit = fit_models(histograms[cut], edges, phase,
                                 period_ns=args.period_ns, lifetime_ns=args.lifetime_ns)
            except (ValueError, RuntimeError) as exc:
                fit_error = str(exc)
        elif len(delays):
            fit_error = (f"late-sideband template has {len(sideband)} bursts; "
                         f"need {args.min_template_bursts}")
        late_hd = ([b for b in selected if b["n_late_hd_hits"] > 0]
                   if counts["later_hodoscope_columns_available"] else None)
        no_late_hd = ([b for b in selected if b["n_late_hd_hits"] == 0]
                      if counts["later_hodoscope_columns_available"] else None)
        result = {
            "max_delayed_burst_hits_inclusive": cut,
            "delayed_candidates": len(selected),
            "candidate_windows": len({(b["run_id"], b["root_entry"]) for b in selected}),
            "candidate_rate_per_unique_prompt_window": (
                len(selected) / counts["unique_prompt_windows"]
                if counts["unique_prompt_windows"] else None),
            "fraction_of_uncut_candidates": len(selected) / len(bursts),
            "candidates_with_later_hd_t0_match": len(late_hd) if late_hd is not None else None,
            "candidates_without_later_hd_t0_match": len(no_late_hd) if no_late_hd is not None else None,
            "fraction_rejected_by_later_hd_veto": (
                len(late_hd) / len(selected) if late_hd is not None and selected else None),
            "late_template_bursts": len(sideband),
            "fit": fit,
            "fit_note": fit_error,
        }
        results.append(result)

    # Rates are divided by the same unique-prompt-window count for direct cut comparison.
    prompt_denominator = max(counts["unique_prompt_windows"], 1)
    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis", len(cuts))
    centers = (edges[:-1] + edges[1:]) / 2
    for i, cut in enumerate(cuts):
        axis.stairs(histograms[cut] / prompt_denominator, edges,
                    color=cmap(i), linewidth=1.6,
                    label=f"≤{cut} digit hits ({histograms[cut].sum():,} candidates)")
    axis.set(xlabel="Time after unique prompt PMT burst (ns)",
             ylabel="Delayed candidates / prompt window / bin",
             title="WCTE delayed-burst timing under inclusive hit-count ceilings")
    axis.grid(alpha=0.2)
    axis.legend(title=f"Prompt windows: {counts['unique_prompt_windows']:,}", fontsize=9)
    fig.savefig(args.output_dir / "hit_cut_timing_overlay.png", dpi=170)
    plt.close(fig)

    all_delays = np.asarray([b["delay_ns"] for b in bursts])
    all_nhits = np.asarray([b["n_hits"] for b in bursts])
    nhit_bin = max(1, args.nhits_bin)
    nhit_edges = np.arange(0, max(nhit_bin, int(all_nhits.max()) + nhit_bin), nhit_bin)
    if nhit_edges[-1] <= all_nhits.max():
        nhit_edges = np.append(nhit_edges, nhit_edges[-1] + nhit_bin)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    nhit_hist, _ = np.histogram(all_nhits, bins=nhit_edges)
    axes[0].stairs(nhit_hist, nhit_edges, color="black", linewidth=1.5)
    for cut in cuts:
        axes[0].axvline(cut, color="tab:blue", alpha=0.5, linestyle="--")
    axes[0].set(xlabel="Digit hits in delayed burst (local burst window)",
                ylabel="Delayed bursts / bin", title="Delayed-burst hit-count distribution")
    hist2d = axes[1].hist2d(all_delays, all_nhits, bins=[edges, nhit_edges],
                            norm=LogNorm(vmin=1), cmap="viridis")
    for cut in cuts:
        axes[1].axhline(cut, color="white", alpha=0.65, linestyle="--", linewidth=0.9)
    axes[1].set(xlabel="Delay after unique prompt burst (ns)",
                ylabel="Digit hits in delayed burst", title="Delayed-burst hits versus time")
    fig.colorbar(hist2d[3], ax=axes[1], label="Delayed bursts / bin (log scale)")
    fig.savefig(args.output_dir / "hit_cut_metrics.png", dpi=170)
    plt.close(fig)

    if counts["later_hodoscope_columns_available"]:
        compare_cut = args.hd_compare_cut
        base = [b for b in bursts if b["n_hits"] <= compare_cut]
        no_hd = [b for b in base if b["n_late_hd_hits"] == 0]
        has_hd = [b for b in base if b["n_late_hd_hits"] > 0]
        fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        for sample, label, color in (
                (base, "All candidates", "black"),
                (no_hd, "No later HD–T0 match", "tab:blue"),
                (has_hd, "Later HD–T0 match present", "tab:orange")):
            hist = np.histogram([b["delay_ns"] for b in sample], bins=edges)[0]
            axis.stairs(hist / prompt_denominator, edges, label=f"{label} (n={len(sample):,})",
                        color=color, linewidth=1.5)
        axis.set(xlabel="Delay after unique prompt PMT burst (ns)",
                 ylabel="Delayed candidates / prompt window / bin",
                 title=f"Effect of later hodoscope activity (≤{compare_cut} hits)")
        axis.grid(alpha=0.2)
        axis.legend()
        fig.savefig(args.output_dir / "late_hd_veto_comparison.png", dpi=170)
        plt.close(fig)

    with (args.output_dir / "hit_cut_histograms.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delay_start_ns", "delay_end_ns", *[f"hits_le_{c}" for c in cuts]])
        for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            writer.writerow([low, high, *[int(histograms[c][i]) for c in cuts]])
    with (args.output_dir / "hit_cut_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ("max_delayed_burst_hits_inclusive", "delayed_candidates", "candidate_windows",
                  "candidate_rate_per_unique_prompt_window", "fraction_of_uncut_candidates",
                  "candidates_with_later_hd_t0_match",
                  "candidates_without_later_hd_t0_match",
                  "fraction_rejected_by_later_hd_veto", "late_template_bursts")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: result[key] for key in fields} for result in results)

    summary = {
        "input_bursts_csv": str(args.bursts_csv), "counts": counts,
        "settings": {
            "prompt_range_ns": [args.prompt_min_ns, args.prompt_max_ns],
            "min_prompt_pmts": args.min_prompt_pmts,
            "delay_range_ns": [args.min_delay_ns, args.max_delay_ns],
            "min_delayed_pmts": args.min_delayed_pmts,
            "min_delayed_burst_hits": args.min_delayed_hits,
            "inclusive_max_delayed_burst_hits": cuts,
            "histogram_bin_ns": args.bin_ns, "comb_period_ns": args.period_ns,
            "fixed_lifetime_ns": args.lifetime_ns,
            "nhits_histogram_bin": args.nhits_bin,
            "hd_comparison_inclusive_hit_cut": args.hd_compare_cut,
            "phase_bins": args.phase_bins,
            "late_template_range_ns": [args.template_min_ns, args.template_max_ns],
            "root_entry_range": [args.root_entry_start, args.root_entry_stop],
        },
        "results_by_hit_cut": results,
        "interpretation": (
            "Exploratory only. The hit ceiling is applied to each delayed burst's local "
            "digit-hit count; it does not establish that the burst is a Michel positron. "
            "Fixed-lifetime fit improvements are descriptive, not significance estimates. "
            "This burst count may not match simulation's "
            "total delayed-cluster hit definition. A later HD–T0 match flags a readout "
            "with hodoscope activity compatible with a later T0 group; it indicates "
            "charged beam activity, not a gamma. It is an exploratory event-level veto, "
            "not a candidate-to-bunch assignment. End-of-spill beam-free selection "
            "requires a validated spill/time selection and is not inferred here."),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Unique prompt windows: {counts['unique_prompt_windows']:,}; "
          f"uncut delayed candidates: {len(bursts):,}")
    for result in results:
        print(f"≤{result['max_delayed_burst_hits_inclusive']} hits: "
              f"{result['delayed_candidates']:,} candidates")
    if not counts["later_hodoscope_columns_available"]:
        print("No later-HD columns in input CSV; rerun the updated NPZ burst study "
              "to create the late_hd_veto_comparison.png")
    print(f"Results: {args.output_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bursts_csv", type=Path,
                        help="bursts.csv from study_wcte_burst_gaps.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-delayed-hits", type=int, nargs="+",
                        default=[100, 150, 200, 300, 400],
                        help="inclusive hit ceilings for delayed bursts")
    parser.add_argument("--prompt-min-ns", type=float, default=1500)
    parser.add_argument("--prompt-max-ns", type=float, default=1900)
    parser.add_argument("--min-prompt-pmts", type=int, default=10)
    parser.add_argument("--min-delay-ns", type=float, default=500)
    parser.add_argument("--max-delay-ns", type=float, default=7500)
    parser.add_argument("--min-delayed-pmts", type=int, default=10)
    parser.add_argument("--min-delayed-hits", type=int, default=1)
    parser.add_argument("--nhits-bin", type=int, default=10,
                        help="digit-hit width for the multiplicity histogram and density plot")
    parser.add_argument("--hd-compare-cut", type=int, default=300,
                        help="hit ceiling used for the later-hodoscope comparison plot")
    parser.add_argument("--bin-ns", type=float, default=25)
    parser.add_argument("--period-ns", type=float, default=330)
    parser.add_argument("--lifetime-ns", type=float, default=2196.9811)
    parser.add_argument("--phase-bins", type=int, default=12)
    parser.add_argument("--template-min-ns", type=float, default=5000)
    parser.add_argument("--template-max-ns", type=float, default=7500)
    parser.add_argument("--min-template-bursts", type=int, default=50)
    parser.add_argument("--root-entry-start", type=int)
    parser.add_argument("--root-entry-stop", type=int,
                        help="exclusive stop entry, for a pre-identified run segment")
    args = parser.parse_args()
    if args.nhits_bin < 1 or args.hd_compare_cut < 1:
        parser.error("--nhits-bin and --hd-compare-cut must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

