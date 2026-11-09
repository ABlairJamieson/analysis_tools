"""Initial framework plugin for delayed PMT bursts in real WCTE data.

No beam-bunch veto is applied. The purpose of this first framework version is
to reproduce and inspect the observed roughly 300 ns structure before
designing any exclusion around it.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from studies.wcte_delayed_bursts.burst_tools import Burst, find_bursts, t0_group_centers


T0_CHANNELS = {0, 1, 2, 3}
T2_CHANNEL = 8
HC2_CHANNEL = 11
HD_CHANNEL_TO_ELEMENT = {
    **{32 + index: index for index in range(7)},
    16: 7,
    **{17 + index: index + 8 for index in range(7)},
}
HD_DT_WINDOWS_NS = {
    hd: ((-50.0, 100.0) if hd in (0, 1, 2, 3, 8, 9, 10, 11) else (80.0, 200.0))
    for hd in range(15)
}


def _data_label(input_path, observed_runs):
    """Build the requested data/run stamp for figures."""
    match = re.search(r"R(\d+)", input_path.name)
    run = str(next(iter(observed_runs))) if observed_runs else (match.group(1) if match else "unknown")
    return f"data | Run {run}"


def _value(batch, name, index, default=None):
    """Safely extract one scalar or jagged event branch value."""
    if name not in batch.fields:
        return default
    result = ak.to_list(batch[name][index])
    return default if result is None else result


def _finite_hits(batch, index):
    """Return good calibrated hit times, charges, and packed PMT IDs."""
    times = np.asarray(_value(batch, "hit_pmt_calibrated_times", index, []), dtype=float)
    charges = np.asarray(_value(batch, "hit_pmt_charges", index, []), dtype=float)
    masks = np.asarray(_value(batch, "hit_pmt_readout_mask", index, []), dtype=np.int64)
    slots = np.asarray(_value(batch, "hit_mpmt_slot_ids", index, []), dtype=np.int64)
    positions = np.asarray(_value(batch, "hit_pmt_position_ids", index, []), dtype=np.int64)
    if len({len(times), len(charges), len(masks), len(slots), len(positions)}) != 1:
        raise ValueError("PMT hit time/charge/mask/ID arrays have different lengths")
    keep = (masks == 0) & np.isfinite(times) & np.isfinite(charges)
    return times[keep], charges[keep], 100 * slots[keep] + positions[keep]


def _t0_and_selection(corrected_hits):
    """Return T0 group times and the existing diagnostic beam/tag predicates."""
    valid_hits = [(hit.channel_id, float(hit.corrected_ns)) for hit in corrected_hits
                  if hit.corrected_ns is not None and np.isfinite(hit.corrected_ns)]
    channels = {channel for channel, _ in valid_hits}
    t0_hits = [time for channel, time in valid_hits if channel in T0_CHANNELS]
    groups = t0_group_centers(t0_hits)
    beam = bool(channels & T0_CHANNELS) and T2_CHANNEL in channels and HC2_CHANNEL not in channels
    tagged = False
    if beam:
        for channel, hd_time in valid_hits:
            if channel not in HD_CHANNEL_TO_ELEMENT:
                continue
            low, high = HD_DT_WINDOWS_NS[HD_CHANNEL_TO_ELEMENT[channel]]
            if any(low < hd_time - t0_time < high for t0_time in t0_hits):
                tagged = True
                break
    return groups, beam, tagged


def _save_data_plot(fig, path, stamp):
    fig.text(0.99, 0.995, stamp, ha="right", va="top", fontsize=10,
             color="#7f1d1d", weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


class Study:
    """DataLoader plugin that records burst timing and pre-clustering basics."""

    required_branches = (
        "run_id", "event_number", "readout_number", "window_data_quality_mask",
        "hit_pmt_calibrated_times", "hit_pmt_charges", "hit_pmt_readout_mask",
        "hit_mpmt_slot_ids", "hit_pmt_position_ids",
    )
    optional_branches = ("T5_hit_time",)

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("Delayed-burst search")
        group.add_argument("--selection", choices=("all", "beam", "tagged"), default="tagged",
                           help="Use good readouts; tagged requires T0+T2, no HC2, and an in-time HD")
        group.add_argument("--min-ns", type=float, default=0)
        group.add_argument("--max-ns", type=float, default=10_000)
        group.add_argument("--bin-ns", type=float, default=10)
        group.add_argument("--width-ns", type=float, default=50)
        group.add_argument("--min-pmts", type=int, default=10)
        group.add_argument("--min-separation-ns", type=float, default=100)
        group.add_argument("--max-bursts", type=int, default=20)
        group.add_argument("--hist-bin-ns", type=float, default=50)
        group.add_argument("--prompt-min-ns", type=float, default=1500)
        group.add_argument("--prompt-max-ns", type=float, default=1900)
        group.add_argument("--delayed-min-ns", type=float, default=500)
        group.add_argument("--delayed-max-ns", type=float, default=7500)

    def __init__(self, args):
        self.args = args
        self.output = args.output_dir
        self.output.mkdir(parents=True, exist_ok=True)
        self.burst_file = (self.output / "bursts.csv").open("w", newline="", encoding="utf-8")
        self.gap_file = (self.output / "gaps.csv").open("w", newline="", encoding="utf-8")
        self.event_file = (self.output / "events.csv").open("w", newline="", encoding="utf-8")
        self.candidate_file = (self.output / "delayed_candidates_no_veto.csv").open(
            "w", newline="", encoding="utf-8")
        self.burst_writer = csv.DictWriter(self.burst_file, fieldnames=(
            "root_entry", "run_id", "event_number", "readout_number", "n_t0_groups",
            "burst_index", "strongest", *Burst.__dataclass_fields__))
        self.gap_writer = csv.DictWriter(self.gap_file, fieldnames=(
            "root_entry", "run_id", "event_number", "readout_number", "kind",
            "from_burst", "to_burst", "gap_ns"))
        self.event_writer = csv.DictWriter(self.event_file, fieldnames=(
            "root_entry", "run_id", "event_number", "readout_number", "selection",
            "n_t0_groups", "n_bursts", "n_unique_prompt_bursts", "n_delayed_candidates"))
        self.candidate_writer = csv.DictWriter(self.candidate_file, fieldnames=(
            "root_entry", "run_id", "event_number", "readout_number", "burst_index",
            "prompt_burst_ns", "candidate_burst_ns", "delay_after_prompt_ns",
            "candidate_n_hits", "candidate_n_pmts", "candidate_charge"))
        for writer in (self.burst_writer, self.gap_writer, self.event_writer, self.candidate_writer):
            writer.writeheader()

        self.counts = Counter()
        self.burst_multiplicity = Counter()
        self.cluster_hits = Counter()
        self.time_edges = np.arange(args.min_ns, args.max_ns + args.hist_bin_ns,
                                    args.hist_bin_ns)
        self.charge_edges = np.linspace(0, 500, 101)
        self.raw_time_hist = np.zeros(len(self.time_edges) - 1, dtype=np.int64)
        self.raw_charge_hist = np.zeros(len(self.charge_edges) - 1, dtype=np.int64)
        self.raw_time_charge_hist = np.zeros((len(self.time_edges) - 1,
                                               len(self.charge_edges) - 1), dtype=np.int64)
        self.all_burst_times = np.zeros(len(self.time_edges) - 1, dtype=np.int64)
        self.prompt_times = np.zeros(len(self.time_edges) - 1, dtype=np.int64)
        self.delayed_absolute_times = np.zeros(len(self.time_edges) - 1, dtype=np.int64)
        self.delayed_relative_edges = np.linspace(args.delayed_min_ns, args.delayed_max_ns, 101)
        self.delayed_relative_times = np.zeros(len(self.delayed_relative_edges) - 1, dtype=np.int64)
        self.delayed_candidate_count = 0
        self.gap_t0_categories = {name: np.zeros(200, dtype=np.int64)
                                  for name in ("no T0", "1 T0 group", "2+ T0 groups")}
        self.gap_edges = np.linspace(0, 10_000, 201)
        self.consecutive_gaps = {name: np.zeros(200, dtype=np.int64)
                                 for name in self.gap_t0_categories}
        self.strongest_gaps = {name: np.zeros(200, dtype=np.int64)
                               for name in self.gap_t0_categories}
        self.run_ids = set()
        self.t5_hits = 0

    def process_batch(self, batch, context):
        """Apply the requested readout selection and accumulate event products."""
        quality_good = []
        beam_selected = []
        tagged_selected = []
        for index in range(len(batch)):
            quality = int(_value(batch, "window_data_quality_mask", index, -1)) == 0
            corrected = context.corrected_tdc_hits[index] if context.corrected_tdc_hits else []
            groups, beam, tagged = _t0_and_selection(corrected)
            quality_good.append(quality)
            beam_selected.append(quality and beam)
            tagged_selected.append(quality and tagged)
            self.counts["raw_readouts"] += 1
            if not quality:
                self.counts["rejected_window_quality"] += 1
                continue
            self.counts["quality_good_readouts"] += 1
            if self.args.selection == "beam" and not beam:
                continue
            if self.args.selection == "tagged" and not tagged:
                continue

            run_id = int(_value(batch, "run_id", index, -1))
            event_number = int(_value(batch, "event_number", index, -1))
            readout_number = int(_value(batch, "readout_number", index, -1))
            self.run_ids.add(run_id)
            entry = context.entry_start + index
            identity = {"root_entry": entry, "run_id": run_id,
                        "event_number": event_number, "readout_number": readout_number}
            self.counts["selected_readouts"] += 1
            self.counts[f"selected_{self.args.selection}_readouts"] += 1
            times, charges, pmts = _finite_hits(batch, index)
            self.raw_time_hist += np.histogram(times, bins=self.time_edges)[0]
            self.raw_charge_hist += np.histogram(charges, bins=self.charge_edges)[0]
            in_2d = (np.isfinite(times) & np.isfinite(charges) & (charges >= 0)
                     & (charges < 500) & (times >= self.args.min_ns)
                     & (times < self.args.max_ns))
            self.raw_time_charge_hist += np.histogram2d(
                times[in_2d], charges[in_2d], bins=(self.time_edges, self.charge_edges))[0].astype(np.int64)
            t5 = _value(batch, "T5_hit_time", index, [])
            self.t5_hits += len(t5) if t5 is not None else 0

            bursts = find_bursts(
                times, charges, pmts, min_ns=self.args.min_ns, max_ns=self.args.max_ns,
                bin_ns=self.args.bin_ns, width_ns=self.args.width_ns, min_pmts=self.args.min_pmts,
                min_separation_ns=self.args.min_separation_ns, max_bursts=self.args.max_bursts)
            self.burst_multiplicity[len(bursts)] += 1
            self.counts["readouts_with_bursts"] += bool(bursts)
            self.counts["readouts_with_multiple_bursts"] += len(bursts) > 1
            self.all_burst_times += np.histogram([burst.center_ns for burst in bursts], bins=self.time_edges)[0]
            self.cluster_hits.update(burst.n_hits for burst in bursts)
            category = ("no T0" if len(groups) == 0 else
                        "1 T0 group" if len(groups) == 1 else "2+ T0 groups")
            strongest = max(range(len(bursts)), key=lambda i: bursts[i].n_pmts) if bursts else None
            prompt_bursts = [burst for burst in bursts
                             if self.args.prompt_min_ns <= burst.center_ns < self.args.prompt_max_ns]
            unique_prompt = prompt_bursts[0] if len(prompt_bursts) == 1 else None
            delayed = []
            if unique_prompt is not None:
                self.prompt_times += np.histogram([unique_prompt.center_ns], bins=self.time_edges)[0]
                for burst_index, burst in enumerate(bursts):
                    delay = burst.center_ns - unique_prompt.center_ns
                    if self.args.delayed_min_ns <= delay <= self.args.delayed_max_ns:
                        delayed.append((burst_index, burst, delay))
                        self.delayed_relative_times += np.histogram([delay], bins=self.delayed_relative_edges)[0]
                        self.delayed_absolute_times += np.histogram([burst.center_ns], bins=self.time_edges)[0]
                        self.delayed_candidate_count += 1
                        self.candidate_writer.writerow({**identity, "burst_index": burst_index,
                            "prompt_burst_ns": unique_prompt.center_ns,
                            "candidate_burst_ns": burst.center_ns,
                            "delay_after_prompt_ns": delay,
                            "candidate_n_hits": burst.n_hits,
                            "candidate_n_pmts": burst.n_pmts,
                            "candidate_charge": burst.charge})
            self.event_writer.writerow({**identity, "selection": self.args.selection,
                "n_t0_groups": len(groups), "n_bursts": len(bursts),
                "n_unique_prompt_bursts": int(unique_prompt is not None),
                "n_delayed_candidates": len(delayed)})

            for burst_index, burst in enumerate(bursts):
                self.burst_writer.writerow({**identity, "n_t0_groups": len(groups),
                    "burst_index": burst_index, "strongest": int(burst_index == strongest),
                    **burst.__dict__})
            for burst_index in range(1, len(bursts)):
                gap = bursts[burst_index].center_ns - bursts[burst_index - 1].center_ns
                self.consecutive_gaps[category] += np.histogram([gap], bins=self.gap_edges)[0]
                self.gap_writer.writerow({**identity, "kind": "consecutive",
                    "from_burst": burst_index - 1, "to_burst": burst_index, "gap_ns": gap})
            if strongest is not None:
                for burst_index in range(strongest + 1, len(bursts)):
                    gap = bursts[burst_index].center_ns - bursts[strongest].center_ns
                    self.strongest_gaps[category] += np.histogram([gap], bins=self.gap_edges)[0]
                    self.gap_writer.writerow({**identity, "kind": "later_from_strongest",
                        "from_burst": strongest, "to_burst": burst_index, "gap_ns": gap})
        context.selections.add("readout_quality_good", quality_good)
        context.selections.add("beam_selection", beam_selected, parent="readout_quality_good")
        context.selections.add("tagged_selection", tagged_selected, parent="readout_quality_good")

    def finish(self):
        """Close event tables and write the first-pass data diagnostics."""
        self._close_files()
        self._write_plots()
        summary = {
            "source_type": "real WCTE data ROOT via analysis_tools.DataLoader",
            "selection": self.args.selection,
            "run_ids": sorted(self.run_ids),
            "counts": dict(self.counts),
            "burst_multiplicity_per_selected_readout": {
                str(key): value for key, value in sorted(self.burst_multiplicity.items())},
            "n_bursts": int(sum(self.cluster_hits.values())),
            "n_delayed_candidates": self.delayed_candidate_count,
            "t5_hit_count_in_selected_readouts": self.t5_hits,
            "settings": {key: getattr(self.args, key) for key in (
                "min_ns", "max_ns", "bin_ns", "width_ns", "min_pmts",
                "min_separation_ns", "max_bursts", "hist_bin_ns", "prompt_min_ns",
                "prompt_max_ns", "delayed_min_ns", "delayed_max_ns")},
            "bunch_structure_note": (
                "No beam-bunch veto or removal around the observed ~300 ns structure is applied. "
                "This is an initial rewrite to reproduce and inspect the structure."),
            "timing_note": (
                "PMT hit times remain on the calibrated WCTE readout-window clock. "
                "Beamline TDC references are corrected once by the shared runner for optional "
                "readout selection; no absolute cross-clock timing comparison is made here."),
            "outputs": ["events.csv", "bursts.csv", "gaps.csv",
                        "delayed_candidates_no_veto.csv", "burst_gap_histograms.png",
                        "clusters_per_event.png", "hits_per_cluster.png",
                        "prompt_delayed_cluster_times.png", "all_cluster_times_readout.png",
                        "raw_pmt_hit_times_preclustering.png",
                        "raw_pmt_hit_charges_preclustering.png",
                        "raw_pmt_charge_vs_time_preclustering.png"],
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                   encoding="utf-8")

    def abort(self):
        self._close_files()

    def _close_files(self):
        for handle in (self.burst_file, self.gap_file, self.event_file, self.candidate_file):
            if not handle.closed:
                handle.close()

    def _write_plots(self):
        stamp = _data_label(self.args.input_root, self.run_ids)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        colors = {"no T0": "tab:gray", "1 T0 group": "tab:blue", "2+ T0 groups": "tab:orange"}
        for axis, distributions, title, limits, log in (
            (axes[0, 0], self.consecutive_gaps, "Consecutive burst gaps — 0–1500 ns", (0, 1500), False),
            (axes[0, 1], self.consecutive_gaps, "Consecutive burst gaps — full range", None, True),
            (axes[1, 0], self.strongest_gaps, "Later than strongest burst — 0–1500 ns", (0, 1500), False),
            (axes[1, 1], self.strongest_gaps, "Later than strongest burst — full range", None, True),
        ):
            for category, counts in distributions.items():
                axis.stairs(counts, self.gap_edges, color=colors[category],
                            label=f"{category} (n={int(counts.sum()):,})")
            if log:
                axis.set_yscale("log")
            elif limits:
                axis.set_xlim(*limits)
            axis.set(xlabel="Time gap between burst centers (ns)", ylabel="Gaps / bin", title=title)
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
        fig.suptitle("Within-readout PMT burst gaps — no bunch veto")
        _save_data_plot(fig, self.output / "burst_gap_histograms.png", stamp)

        fig, axis = plt.subplots(figsize=(9, 5))
        multiplicities = sorted(self.burst_multiplicity)
        axis.bar(multiplicities, [self.burst_multiplicity[key] for key in multiplicities], color="tab:blue")
        axis.set(xlabel="Found bursts per selected readout", ylabel="Readouts",
                 title="Burst multiplicity per readout")
        axis.grid(axis="y", alpha=0.25)
        _save_data_plot(fig, self.output / "clusters_per_event.png", stamp)

        fig, axis = plt.subplots(figsize=(9, 5))
        if self.cluster_hits:
            axis.bar(list(self.cluster_hits), list(self.cluster_hits.values()), color="tab:blue")
        axis.set(xlabel=f"Digit hits in a {self.args.width_ns:g}-ns burst window",
                 ylabel="Bursts", title="Hits per found burst")
        axis.grid(axis="y", alpha=0.25)
        _save_data_plot(fig, self.output / "hits_per_cluster.png", stamp)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False, constrained_layout=True)
        axes[0].stairs(self.prompt_times, self.time_edges, color="tab:green")
        axes[0].set(title="Unique prompt-burst times", xlabel="Readout time (ns)", ylabel="Readouts / bin")
        axes[1].stairs(self.delayed_relative_times, self.delayed_relative_edges, color="tab:orange")
        axes[1].set(title="Delayed-burst times relative to unique prompt", xlabel="Delay (ns)",
                    ylabel="Candidates / bin")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle("Basic prompt and delayed burst timing")
        _save_data_plot(fig, self.output / "prompt_delayed_cluster_times.png", stamp)

        fig, axis = plt.subplots(figsize=(10, 5))
        for values, label, color in (
            (self.raw_time_hist, "All calibrated PMT hits", "tab:blue"),
            (self.all_burst_times, "All found bursts", "tab:green"),
            (self.prompt_times, "Unique prompt bursts", "tab:red"),
            (self.delayed_absolute_times, "Delayed candidates", "tab:orange"),
        ):
            axis.stairs(values, self.time_edges, linewidth=1.2,
                        label=f"{label} (n={int(np.sum(values)):,})", color=color)
        axis.set(xlabel="Calibrated readout time (ns; common unshifted axis)",
                 ylabel="Hits or bursts / bin", title="All hit and burst times on one axis")
        axis.legend()
        axis.grid(alpha=0.25)
        _save_data_plot(fig, self.output / "all_cluster_times_readout.png", stamp)

        fig, axis = plt.subplots(figsize=(10, 5))
        axis.stairs(self.raw_time_hist, self.time_edges, color="tab:blue")
        axis.set(xlabel="Calibrated PMT hit time in readout (ns)", ylabel="PMT hits / bin",
                 title="Raw PMT hit times before burst finding")
        axis.grid(alpha=0.25)
        _save_data_plot(fig, self.output / "raw_pmt_hit_times_preclustering.png", stamp)

        fig, axis = plt.subplots(figsize=(9, 5))
        axis.stairs(self.raw_charge_hist, self.charge_edges, color="tab:purple")
        axis.set(xlabel="PMT hit charge (input units)", ylabel="PMT hits / bin",
                 title="Raw PMT hit charges before burst finding")
        axis.grid(alpha=0.25)
        _save_data_plot(fig, self.output / "raw_pmt_hit_charges_preclustering.png", stamp)

        fig, axis = plt.subplots(figsize=(10, 5))
        if np.any(self.raw_time_charge_hist):
            hist = axis.pcolormesh(self.time_edges, self.charge_edges,
                                   np.ma.masked_less_equal(self.raw_time_charge_hist.T, 0),
                                   norm=LogNorm(), cmap="viridis", shading="auto")
            fig.colorbar(hist, ax=axis, label="PMT hits / bin (log scale)")
        axis.set(xlabel="Calibrated PMT hit time in readout (ns)", ylabel="PMT hit charge (input units)",
                 title="Raw PMT hit charge versus time before burst finding")
        _save_data_plot(fig, self.output / "raw_pmt_charge_vs_time_preclustering.png", stamp)
