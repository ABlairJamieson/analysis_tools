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
    """Return all delayed bursts in windows with exactly one accepted prompt."""
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
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"{path} is not a study_wcte_burst_gaps.py bursts.csv")
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
            })
    process_window(current_bursts)
    counts["delayed_candidates_before_upper_hit_cut"] = len(delayed)
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
    delays_all = np.asarray([b["delay_ns"] for b in bursts])
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
        phase_profile = None
        if len(delays):
            phase_profile, _ = np.histogram(np.mod(delays, args.period_ns),
                                             bins=np.linspace(0, args.period_ns,
                                                              args.phase_bins + 1))
        phase_mean = float(phase_profile.mean()) if phase_profile is not None else 0.0
        phase_contrast = (float((phase_profile.max() - phase_profile.min()) / phase_mean)
                          if phase_mean > 0 else None)
        result = {
            "max_delayed_burst_hits_inclusive": cut,
            "delayed_candidates": len(selected),
            "candidate_windows": len({(b["run_id"], b["root_entry"]) for b in selected}),
            "candidate_rate_per_unique_prompt_window": (
                len(selected) / counts["unique_prompt_windows"]
                if counts["unique_prompt_windows"] else None),
            "fraction_of_uncut_candidates": len(selected) / len(bursts),
            "phase_peak_to_trough_over_mean": phase_contrast,
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    x = np.arange(len(results))
    axes[0].bar(x, [r["delayed_candidates"] for r in results], color="tab:blue")
    axes[0].set(ylabel="Delayed candidates", title="Candidates retained")
    axes[1].plot(x, [r["phase_peak_to_trough_over_mean"] or 0 for r in results],
                 "o-", color="tab:purple")
    axes[1].set(ylabel="(phase max − min) / phase mean",
                title=f"Bunch-phase contrast (period={args.period_ns:g} ns)")
    for axis in axes:
        axis.set_xticks(x, [str(cut) for cut in cuts])
        axis.set_xlabel("Inclusive maximum digit hits per delayed burst")
        axis.grid(alpha=0.2)
    fig.savefig(args.output_dir / "hit_cut_metrics.png", dpi=170)
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
                  "phase_peak_to_trough_over_mean", "late_template_bursts")
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
            "phase_bins": args.phase_bins,
            "late_template_range_ns": [args.template_min_ns, args.template_max_ns],
            "root_entry_range": [args.root_entry_start, args.root_entry_stop],
        },
        "results_by_hit_cut": results,
        "interpretation": (
            "Exploratory only. The hit ceiling is applied to each delayed burst's local "
            "digit-hit count; it does not establish that the burst is a Michel positron. "
            "Comb phase contrast and fixed-lifetime fit improvements are descriptive, "
            "not significance estimates. This burst count may not match simulation's "
            "total delayed-cluster hit definition. End-of-spill beam-free selection "
            "requires a validated spill/time selection and is not inferred here."),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Unique prompt windows: {counts['unique_prompt_windows']:,}; "
          f"uncut delayed candidates: {len(bursts):,}")
    for result in results:
        print(f"≤{result['max_delayed_burst_hits_inclusive']} hits: "
              f"{result['delayed_candidates']:,} candidates, "
              f"phase contrast={result['phase_peak_to_trough_over_mean']}")
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
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

