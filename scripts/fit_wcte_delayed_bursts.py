#!/usr/bin/env python3
"""Exploratory prompt-relative PMT burst fit: periodic comb versus comb + muon lifetime.

Input is bursts.csv from study_wcte_burst_gaps.py. This is a population-level
timing check, not a Michel/e+ tag or a calibrated significance measurement.
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
from scipy.optimize import minimize


REQUIRED = {"root_entry", "run_id", "readout_number", "center_ns", "n_pmts"}


def load_delays(path: Path, *, prompt_min_ns: float, prompt_max_ns: float,
                min_delay_ns: float, max_delay_ns: float,
                min_prompt_pmts: int, min_delayed_pmts: int) -> tuple[np.ndarray, dict]:
    """Use exactly one prompt burst per readout; retain all later burst centers.

    The exporter writes consecutive rows for each ROOT entry. Only one window's
    burst records are held at a time, making a full-run CSV practical on lxplus.
    """
    delays = []
    counts = {"input_windows": 0, "windows_without_unique_prompt": 0,
              "selected_windows": 0, "windows_with_delayed_bursts": 0,
              "delayed_bursts": 0}

    def process_window(bursts):
        if not bursts:
            return
        counts["input_windows"] += 1
        prompt = [(t, n) for t, n in bursts
                  if prompt_min_ns <= t < prompt_max_ns and n >= min_prompt_pmts]
        if len(prompt) != 1:
            counts["windows_without_unique_prompt"] += 1
            return
        counts["selected_windows"] += 1
        prompt_time = prompt[0][0]
        found = 0
        for center, n_pmts in bursts:
            delay = center - prompt_time
            if min_delay_ns <= delay < max_delay_ns and n_pmts >= min_delayed_pmts:
                delays.append(delay)
                found += 1
        if found:
            counts["windows_with_delayed_bursts"] += 1

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"{path} is not a study_wcte_burst_gaps.py bursts.csv")
        current_key = None
        current_bursts = []
        previous_entry = -1
        for row in reader:
            key = (row["run_id"], row["root_entry"], row["readout_number"])
            entry = int(row["root_entry"])
            if entry < previous_entry:
                raise ValueError(f"{path} must be ordered by ROOT entry")
            if current_key is not None and key != current_key:
                process_window(current_bursts)
                current_bursts = []
            current_key = key
            previous_entry = entry
            center = float(row["center_ns"])
            n_pmts = int(row["n_pmts"])
            if np.isfinite(center):
                current_bursts.append((center, n_pmts))
        process_window(current_bursts)
    counts["delayed_bursts"] = len(delays)
    return np.asarray(delays, dtype=float), counts


def phase_template(delays: np.ndarray, *, period_ns: float, phase_bins: int,
                   smoothing_bins: int = 1) -> np.ndarray:
    """Circularly smoothed phase density, normalized to mean one."""
    phase = np.mod(delays, period_ns)
    profile, _ = np.histogram(phase, bins=np.linspace(0, period_ns, phase_bins + 1))
    profile = profile.astype(float)
    if not profile.sum():
        raise ValueError("No bursts available to estimate the comb phase")
    for _ in range(smoothing_bins):
        profile = (np.roll(profile, 1) + 2 * profile + np.roll(profile, -1)) / 4
    return profile / profile.mean()


def fit_models(counts: np.ndarray, edges: np.ndarray, phase: np.ndarray,
               *, period_ns: float, lifetime_ns: float) -> dict:
    """Fit binned Poisson counts with nonnegative flat, comb, and decay terms."""
    centers = (edges[:-1] + edges[1:]) / 2
    phase_index = np.floor(np.mod(centers, period_ns) / period_ns * len(phase)).astype(int)
    comb = phase[phase_index]
    # Bin-integrated exponential shape, normalized to one at t=0 for equal bins.
    decay = (np.exp(-edges[:-1] / lifetime_ns) - np.exp(-edges[1:] / lifetime_ns))
    decay /= 1 - np.exp(-(edges[1] - edges[0]) / lifetime_ns)
    if np.count_nonzero(counts) < 3:
        raise ValueError("Too few occupied time bins for a fit")

    def solve(include_decay: bool) -> dict:
        columns = [np.ones_like(comb), comb]
        if include_decay:
            columns.append(decay)
        design = np.column_stack(columns)
        initial = np.full(design.shape[1], max(float(counts.mean()) / design.shape[1], 0.1))

        def objective(coefficients):
            expected = np.clip(design @ coefficients, 1e-12, None)
            return float(np.sum(expected - counts * np.log(expected)))

        result = minimize(objective, initial, method="L-BFGS-B",
                          bounds=[(0, None)] * design.shape[1])
        if not result.success:
            raise RuntimeError(f"Poisson fit failed: {result.message}")
        return {"coefficients": [float(x) for x in result.x],
                "expected": [float(x) for x in design @ result.x],
                "negative_log_likelihood_without_constant": float(result.fun)}

    null = solve(False)
    signal = solve(True)
    return {"comb_only": null, "comb_plus_fixed_lifetime": signal,
            "twice_log_likelihood_improvement_descriptive":
                max(0.0, 2 * (null["negative_log_likelihood_without_constant"]
                              - signal["negative_log_likelihood_without_constant"]))}


def run(args: argparse.Namespace) -> dict:
    if not (0 < args.prompt_min_ns < args.prompt_max_ns and
            0 <= args.min_delay_ns < args.max_delay_ns and args.bin_ns > 0 and
            args.period_ns > 0 and args.lifetime_ns > 0 and args.phase_bins >= 3 and
            args.min_prompt_pmts > 0 and args.min_delayed_pmts > 0):
        raise ValueError("Invalid prompt, delay, bin, period, lifetime, or PMT threshold")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = dict(prompt_min_ns=args.prompt_min_ns, prompt_max_ns=args.prompt_max_ns,
                   min_delay_ns=args.min_delay_ns, max_delay_ns=args.max_delay_ns,
                   min_prompt_pmts=args.min_prompt_pmts,
                   min_delayed_pmts=args.min_delayed_pmts)
    delays, counts = load_delays(args.bursts_csv, **options)
    if not len(delays):
        raise ValueError("No delayed bursts; check the prompt range and selection")
    source = "same_sample_late_sideband"
    if args.control_bursts_csv:
        template_delays, control_counts = load_delays(args.control_bursts_csv, **options)
        source = "control_bursts_csv"
    else:
        template_delays = delays[(delays >= args.template_min_ns)
                                 & (delays < args.template_max_ns)]
        control_counts = None
    if len(template_delays) < args.min_template_bursts:
        raise ValueError(f"Only {len(template_delays)} comb-template bursts; need at least "
                         f"{args.min_template_bursts}. Increase scan size or use --control-bursts-csv")
    phase = phase_template(template_delays, period_ns=args.period_ns,
                           phase_bins=args.phase_bins)
    edges = np.arange(args.min_delay_ns, args.max_delay_ns + args.bin_ns / 2,
                      args.bin_ns)
    if len(edges) < 4 or not np.isclose(edges[-1], args.max_delay_ns):
        raise ValueError("Fit range must contain at least 3 whole, equal-width bins")
    hist, _ = np.histogram(delays, bins=edges)
    fit = fit_models(hist, edges, phase, period_ns=args.period_ns,
                     lifetime_ns=args.lifetime_ns)
    summary = {
        "input_bursts_csv": str(args.bursts_csv), "control_bursts_csv":
            str(args.control_bursts_csv) if args.control_bursts_csv else None,
        "counts": counts, "control_counts": control_counts,
        "settings": {"prompt_range_ns": [args.prompt_min_ns, args.prompt_max_ns],
                     "fit_delay_range_ns": [args.min_delay_ns, args.max_delay_ns],
                     "bin_ns": args.bin_ns, "period_ns": args.period_ns,
                     "lifetime_ns_fixed": args.lifetime_ns,
                     "min_prompt_pmts": args.min_prompt_pmts,
                     "min_delayed_pmts": args.min_delayed_pmts,
                     "phase_bins": args.phase_bins,
                     "template_range_ns": [args.template_min_ns, args.template_max_ns]},
        "template_source": source, "template_bursts": len(template_delays),
        "phase_template_mean_one": [float(x) for x in phase], "fit": fit,
        "caveat": ("Exploratory only: no calibrated p-value. Bursts within a window are "
                   "correlated; same-sample phase templates reuse data; and time-varying "
                   "beam background/readout acceptance can imitate a lifetime. This single-prompt "
                   "exponential does not model muons stopping in later beam bunches. A muon-enriched "
                   "sample, parent-time information, and independent control are needed for a "
                   "Michel interpretation.")}
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("delay_start_ns", "delay_end_ns", "bursts", "comb_only_expected",
                         "comb_plus_lifetime_expected"))
        for low, high, observed, null, alternative in zip(
                edges[:-1], edges[1:], hist, fit["comb_only"]["expected"],
                fit["comb_plus_fixed_lifetime"]["expected"]):
            writer.writerow((low, high, observed, null, alternative))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    centers = (edges[:-1] + edges[1:]) / 2
    axes[0].stairs(hist, edges, color="black", label="Observed delayed bursts")
    axes[0].plot(centers, fit["comb_only"]["expected"], label="Comb + flat")
    axes[0].plot(centers, fit["comb_plus_fixed_lifetime"]["expected"],
                 label=f"Comb + flat + {args.lifetime_ns / 1000:g} µs exponential")
    axes[0].set(ylabel=f"Bursts / {args.bin_ns:g} ns bin")
    axes[0].legend()
    axes[0].set_title("Exploratory prompt-relative PMT timing (not a Michel tag)")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].plot(centers, hist - np.asarray(fit["comb_only"]["expected"]),
                 color="tab:blue", lw=1)
    axes[1].set(xlabel="Time after selected prompt PMT burst (ns)",
                ylabel="Data − comb")
    fig.savefig(args.output_dir / "delayed_burst_lifetime_check.png", dpi=170)
    plt.close(fig)
    print(f"Selected {counts['selected_windows']:,} windows; {len(delays):,} delayed bursts")
    print(f"Comb template: {source}, {len(template_delays):,} bursts")
    print("Descriptive 2ΔlogL (not a significance): "
          f"{fit['twice_log_likelihood_improvement_descriptive']:.2f}")
    print(f"Results: {args.output_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bursts_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-bursts-csv", type=Path,
                        help="Independent non-muon control export for comb phase; recommended")
    parser.add_argument("--prompt-min-ns", type=float, default=1500)
    parser.add_argument("--prompt-max-ns", type=float, default=1900)
    parser.add_argument("--min-delay-ns", type=float, default=500)
    parser.add_argument("--max-delay-ns", type=float, default=7500)
    parser.add_argument("--bin-ns", type=float, default=25)
    parser.add_argument("--period-ns", type=float, default=330)
    parser.add_argument("--lifetime-ns", type=float, default=2196.9811)
    parser.add_argument("--min-prompt-pmts", type=int, default=10)
    parser.add_argument("--min-delayed-pmts", type=int, default=10)
    parser.add_argument("--phase-bins", type=int, default=12)
    parser.add_argument("--template-min-ns", type=float, default=5000)
    parser.add_argument("--template-max-ns", type=float, default=7500)
    parser.add_argument("--min-template-bursts", type=int, default=50)
    args = parser.parse_args()
    if args.template_min_ns >= args.template_max_ns or args.min_template_bursts < 1:
        parser.error("Invalid comb-template range or minimum count")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
