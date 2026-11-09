"""Streamed port of the core WCTE beamline-prompt analysis.

This plugin carries the main timing selections, TOF distributions, T5
multiplicity/end-time checks, HD charge progression, and WCTE charge-by-HD
progression from ``scripts/study_wcte_beamline_prompt.py`` into the new
DataLoader study interface. The older script remains unchanged as the full
reference while less central diagnostics are migrated and validated.

Selections intentionally remain exploratory: channel-31/channel-46 reference
subtraction is intra-beamline only, and prompt/TOF windows are data-derived.
No WCTE PMT quality selection is applied.
"""

from __future__ import annotations

import csv
import json
import re

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.study_wcte_beamline_prompt import (
    first_significant_peak,
    main_peak,
    robust_group_mean,
)


T0_CHANNELS = (0, 1, 2, 3)
T2_CHANNELS = (8,)
T5_BARS = tuple((48 + bar, 56 + bar) for bar in range(8))
HD_CHANNEL_TO_ELEMENT = {**{32 + i: i for i in range(7)}, 16: 7,
                         **{17 + i: i + 8 for i in range(7)}}
ELEMENT_TO_HD_CHANNEL = {element: channel
                         for channel, element in HD_CHANNEL_TO_ELEMENT.items()}
HC2_CHANNEL = 11


def _save(fig, path, run_number):
    """Save one plot with the same data/run stamp as the reference analysis."""
    tag = f"data | Run {run_number}" if run_number is not None else "data"
    fig.text(0.99, 0.995, tag, ha="right", va="top", fontsize=10,
             color="#7f1d1d", weight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class Study:
    """DataLoader plugin for the core beamline prompt study."""

    required_branches = ("beamline_pmt_tdc_ids", "beamline_pmt_tdc_times")
    optional_branches = (
        "vme_digi_issues_bitmask", "beamline_pmt_qdc_ids",
        "beamline_pmt_qdc_charges", "hit_pmt_charges",
        "hit_pmt_readout_mask", "hit_pmt_calibrated_times",
        "window_data_quality_mask", "event_number", "readout_number",
        "T5_HasInTimeWindow",
    )

    @staticmethod
    def add_arguments(parser):
        """Register the settings specific to this beamline analysis."""
        gates = parser.add_argument_group("Beamline prompt study settings")
        gates.add_argument("--prompt-half-width-ns", type=float, default=40.0,
                           help="Half-width of each detector's prompt gate")
        gates.add_argument("--peak-bin-ns", type=float, default=2.0,
                           help="Bin width for locating detector prompt peaks")
        gates.add_argument("--tof-first-peak-half-width-ns", type=float, default=0.5,
                           help="Half-width around the first T2-T0 peak")
        gates.add_argument("--tof-peak-bin-ns", type=float, default=0.2,
                           help="Histogram bin width used to locate the first T2-T0 peak")
        gates.add_argument("--tof-first-peak-min-fraction", type=float, default=0.10,
                           help="Peak threshold as a fraction of the tallest smoothed peak")
        gates.add_argument("--end-diff-range-ns", type=float, default=5.0,
                           help="Half-range for the T5 end-time-difference plot")

    def __init__(self, args):
        self.args = args
        self.out = args.output_dir
        self.rows = []
        self.qdc_raw = []
        self.has_qdc = False
        self.entries_seen = 0
        self.selection_counts = {"all": 0, "beamline_quality": 0, "T0 present": 0}
        self.references_seen = {"31": 0, "46": 0}
        self.references_missing = {"31": 0, "46": 0}
        self._run_number = None

    def process_batch(self, batch, context):
        """Reduce this batch to compact event records before retaining it."""
        size = len(batch)
        self.entries_seen += size
        self.selection_counts["all"] += size
        ids = ak.to_list(batch["beamline_pmt_tdc_ids"])
        raw_times = ak.to_list(batch["beamline_pmt_tdc_times"])
        quality = context.selections.mask("beamline_quality")
        tdc_issue = (ak.to_list(batch["vme_digi_issues_bitmask"])
                     if "vme_digi_issues_bitmask" in batch.fields else None)
        qdc_ids = (ak.to_list(batch["beamline_pmt_qdc_ids"])
                   if "beamline_pmt_qdc_ids" in batch.fields else None)
        qdc_values = (ak.to_list(batch["beamline_pmt_qdc_charges"])
                      if "beamline_pmt_qdc_charges" in batch.fields else None)
        self.has_qdc = self.has_qdc or (qdc_ids is not None and qdc_values is not None)
        pmt_charges = (ak.to_list(batch["hit_pmt_charges"])
                       if "hit_pmt_charges" in batch.fields else None)
        pmt_masks = (ak.to_list(batch["hit_pmt_readout_mask"])
                     if "hit_pmt_readout_mask" in batch.fields else None)
        event_numbers = (ak.to_list(batch["event_number"])
                         if "event_number" in batch.fields else None)
        readout_numbers = (ak.to_list(batch["readout_number"])
                           if "readout_number" in batch.fields else None)
        t5_flags = (ak.to_list(batch["T5_HasInTimeWindow"])
                    if "T5_HasInTimeWindow" in batch.fields else None)
        wcte_quality = (ak.to_list(batch["window_data_quality_mask"])
                        if "window_data_quality_mask" in batch.fields else None)

        hd_charge_rows = np.full((size, 15), np.nan, dtype=np.float32)
        if qdc_ids is not None and qdc_values is not None:
            for index, (channels, charges) in enumerate(zip(qdc_ids, qdc_values)):
                if channels is None or charges is None:
                    continue
                for channel, charge in zip(channels, charges):
                    if channel is None or charge is None:
                        continue
                    channel = int(channel)
                    charge = float(charge)
                    element = HD_CHANNEL_TO_ELEMENT.get(channel)
                    if element is not None and np.isfinite(charge) and np.isnan(hd_charge_rows[index, element]):
                        hd_charge_rows[index, element] = charge
        self.qdc_raw.append(hd_charge_rows)

        has_t0 = []
        for local_index, (event_ids, event_raw_times, event_hits) in enumerate(
                zip(ids, raw_times, context.corrected_tdc_hits)):
            corrected = {}
            raw = {}
            for hit in event_hits:
                if hit.channel_id in (31, 46):
                    continue
                corrected.setdefault(hit.channel_id, hit.corrected_ns)
            if event_ids is not None and event_raw_times is not None:
                for channel, value in zip(event_ids, event_raw_times):
                    if channel is not None and value is not None and np.isfinite(float(value)):
                        raw.setdefault(int(channel), float(value))

            t0 = robust_group_mean(corrected, T0_CHANNELS)
            t0_raw = robust_group_mean(raw, T0_CHANNELS)
            t2 = robust_group_mean(corrected, T2_CHANNELS)
            has_t0.append(t0 is not None)
            for reference_id, value in zip((31, 46), context.reference_times_ns[local_index]):
                key = str(reference_id)
                self.references_seen[key] += value is not None
                self.references_missing[key] += value is None
            if not quality[local_index]:
                continue

            t5_bars = []
            for bar, (left, right) in enumerate(T5_BARS):
                if left in corrected and right in corrected:
                    t5_bars.append((bar, (corrected[left] + corrected[right]) / 2,
                                    corrected[left] - corrected[right]))
            hd = {HD_CHANNEL_TO_ELEMENT[ch]: corrected[ch]
                  for ch in HD_CHANNEL_TO_ELEMENT if ch in corrected}
            charge = accepted = masked = total = None
            if pmt_charges is not None and pmt_masks is not None:
                charges, masks = pmt_charges[local_index], pmt_masks[local_index]
                if charges is not None and masks is not None and len(charges) == len(masks):
                    good_charges = [float(value) for value, mask in zip(charges, masks)
                                    if value is not None and mask is not None and int(mask) == 0
                                    and np.isfinite(float(value))]
                    charge = float(sum(good_charges))
                    accepted = len(good_charges)
                    masked = sum(mask is not None and int(mask) != 0 for mask in masks)
                    total = len(charges)
            row = {
                "entry": context.entry_start + local_index,
                "times": corrected, "raw_times": raw, "t0": t0, "t0_raw": t0_raw,
                "t2": t2, "t5_bars": t5_bars, "hd": hd,
                "wcte_charge": charge, "wcte_accepted_hits": accepted,
                "wcte_masked_hits": masked, "wcte_total_hits": total,
                "wcte_event_quality": (int(wcte_quality[local_index])
                                        if wcte_quality is not None else None),
                "event_number": event_numbers[local_index] if event_numbers is not None else None,
                "readout_number": readout_numbers[local_index] if readout_numbers is not None else None,
                "t5_in_time_window": (bool(t5_flags[local_index]) if t5_flags is not None
                                       and t5_flags[local_index] is not None else None),
            }
            self.rows.append(row)

        selection = context.selections.add("beamline_quality_and_T0", has_t0,
                                           parent="beamline_quality")
        self.selection_counts["beamline_quality"] += int(quality.sum())
        self.selection_counts["T0 present"] += int(selection.sum())

    def finish(self):
        """Build the study's run-wide prompt gates, plots, CSVs, and summary."""
        args = self.args
        run_match = re.search(r"(?:^|[^A-Za-z])R?(\d{4,})(?:[^0-9]|$)",
                              self.args.input_root.name)
        self._run_number = int(run_match.group(1)) if run_match else None
        if not self.rows:
            raise RuntimeError("No readouts passed beamline reference/digitiser quality")

        t0_label = f"T0 average ({len(T0_CHANNELS)} channels)"
        t2_label = f"T2 average ({len(T2_CHANNELS)} channels)"
        samples = {
            t0_label: [row["t0"] for row in self.rows if row["t0"] is not None],
            t2_label: [row["t2"] for row in self.rows if row["t2"] is not None],
        }
        for bar in range(8):
            samples[f"T5 bar {bar} paired-end mean"] = [mean for row in self.rows
                                                         for b, mean, _ in row["t5_bars"] if b == bar]
            for end, channel in enumerate(T5_BARS[bar]):
                samples[f"T5 bar {bar} end {end}"] = [row["times"][channel] for row in self.rows
                                                       if channel in row["times"]]
        for element in range(15):
            samples[f"HD{element}"] = [row["hd"][element] for row in self.rows
                                        if element in row["hd"]]
        gates = {}
        for name, values in samples.items():
            center, peak_count, n = main_peak(values, bin_ns=args.peak_bin_ns)
            gates[name] = {"center_ns": center, "half_width_ns": args.prompt_half_width_ns,
                           "peak_bin_count": int(peak_count), "entries_used": n}
        self._plot_peak_distributions(samples, gates, t0_label, t2_label)

        def in_gate(label, value):
            center = gates[label]["center_ns"]
            return value is not None and np.isfinite(center) and abs(value - center) <= args.prompt_half_width_ns

        t0_prompt_tof = [row["t2"] - row["t0"] for row in self.rows
                         if row["t2"] is not None and in_gate(t0_label, row["t0"])]
        tof_center, tof_peak_count, tof_entries = first_significant_peak(
            t0_prompt_tof, bin_ns=args.tof_peak_bin_ns,
            min_fraction=args.tof_first_peak_min_fraction)
        tof_bounds = (tof_center - args.tof_first_peak_half_width_ns,
                      tof_center + args.tof_first_peak_half_width_ns)
        tof_values = {"T2-T0": [], "T0-T5": [], "T0-HD0": [], "T0-HD5": [], "T0-HD8": []}
        t5_multiplicity = np.zeros((8, 3), dtype=np.int64)
        t5_end_differences = [[] for _ in range(8)]
        for row in self.rows:
            t0_ok, t2_ok = in_gate(t0_label, row["t0"]), in_gate(t2_label, row["t2"])
            if t0_ok and t2_ok:
                tof_values["T2-T0"].append(row["t2"] - row["t0"])
            if t0_ok:
                for element in (0, 5, 8):
                    label = f"HD{element}"
                    if element in row["hd"] and in_gate(label, row["hd"][element]):
                        tof_values[f"T0-HD{element}"].append(row["hd"][element] - row["t0"])
                for bar, mean, _ in row["t5_bars"]:
                    label = f"T5 bar {bar} paired-end mean"
                    if in_gate(label, mean):
                        tof_values["T0-T5"].append(mean - row["t0"])
            for bar, (left, right) in enumerate(T5_BARS):
                left_ok = left in row["times"] and in_gate(f"T5 bar {bar} end 0", row["times"].get(left))
                right_ok = right in row["times"] and in_gate(f"T5 bar {bar} end 1", row["times"].get(right))
                multiplicity = int(left_ok) + int(right_ok)
                t5_multiplicity[bar, multiplicity] += 1
                if left_ok and right_ok:
                    t5_end_differences[bar].append(row["times"][left] - row["times"][right])

        self._plot_tof(tof_values)
        self._plot_t5(t5_multiplicity, t5_end_differences)
        hd_charge_summary = self._plot_hd_charge_progression(gates, in_gate, tof_center, tof_bounds)
        wcte_summary = self._plot_wcte_charge_progression(gates, in_gate, tof_center, tof_bounds)

        with (self.out / "prompt_tof_values.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tof_name", "tof_ns"])
            for label, values in tof_values.items():
                writer.writerows((label, value) for value in values)
        with (self.out / "t5_prompt_multiplicity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bar", "end_multiplicity", "readouts"])
            for bar in range(8):
                for multiplicity in range(3):
                    writer.writerow([bar, multiplicity, int(t5_multiplicity[bar, multiplicity])])

        summary = {
            "input": str(args.input_root), "tree": args.tree,
            "run_number": self._run_number, "processed_entries": self.entries_seen,
            "beamline_quality_readouts": len(self.rows),
            "readouts_with_both_reference_channels": self.references_seen["31"]
                - self.references_missing["31"],
            "missing_reference_counts": self.references_missing,
            "tdc_reference_correction": "first channel 31 for IDs <=31; first channel 46 for IDs >31",
            "prompt_gate_half_width_ns": args.prompt_half_width_ns,
            "prompt_peak_bin_width_ns": args.peak_bin_ns,
            "prompt_gate_centers": gates,
            "t0_channels": list(T0_CHANNELS), "t2_channels": list(T2_CHANNELS),
            "tof_counts": {name: len(values) for name, values in tof_values.items()},
            "t0_t2_prompt_gate_counts": {
                "t0_in_gate": sum(in_gate(t0_label, row["t0"]) for row in self.rows),
                "t2_in_gate": sum(in_gate(t2_label, row["t2"]) for row in self.rows),
                "both_in_gate": len(tof_values["T2-T0"]),
            },
            "first_t2_t0_peak": {
                "center_ns": float(tof_center) if np.isfinite(tof_center) else None,
                "half_width_ns": args.tof_first_peak_half_width_ns,
                "window_ns": list(map(float, tof_bounds)) if np.isfinite(tof_center) else None,
                "entries_considered_after_t0_gate": int(tof_entries),
                "peak_bin_count": int(tof_peak_count),
                "histogram_bin_ns": args.tof_peak_bin_ns,
                "minimum_peak_fraction": args.tof_first_peak_min_fraction,
            },
            "t5_end_multiplicity": t5_multiplicity.tolist(),
            "hd_charge_cut_progression": hd_charge_summary,
            "wcte_charge_cut_progression_by_hd_tdc": wcte_summary,
            "selection_counts": self.selection_counts,
            "selection_note": "Beamline references required; VME digitiser issue flag clear when present. No WCTE PMT quality mask. T5 bars are not vetoed for multiplicity.",
            "clock_interpretation_note": "TOF histograms are corrected detector-time differences, not calibrated physical TOFs or an alignment to the WCTE PMT clock.",
            "migration_note": "This framework port contains the main timing/gate and charge-progression products. See scripts/study_wcte_beamline_prompt.py for the complete reference set of ancillary T0 correction, low-charge, and detailed per-event diagnostics while the migration is validated.",
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def _plot_peak_distributions(self, samples, gates, t0_label, t2_label):
        def draw(axis, label):
            values = np.asarray(samples[label], dtype=float)
            if len(values):
                axis.hist(values, bins=160, histtype="step", color="tab:blue")
            center = gates[label]["center_ns"]
            if np.isfinite(center):
                axis.axvspan(center - self.args.prompt_half_width_ns,
                             center + self.args.prompt_half_width_ns,
                             alpha=0.18, color="tab:orange")
                axis.axvline(center, color="tab:orange", linestyle="--",
                             label=f"main peak={center:.1f} ns")
                axis.legend(fontsize=8)
            axis.set_ylabel("Readouts / bin")
            axis.set_title(f"{label} (n={len(values):,})")
            axis.set_xlabel("Reference-corrected time (ns)")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for axis, label in zip(axes, (t0_label, t2_label)):
            draw(axis, label)
        fig.suptitle("T0/T2 main prompt peaks and fixed gates")
        fig.tight_layout()
        _save(fig, self.out / "beamline_prompt_peaks_t0_t2.png", self._run_number)
        for first_bar in (0, 4):
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for bar, axis in zip(range(first_bar, first_bar + 4), axes.flat):
                for end in range(2):
                    label = f"T5 bar {bar} end {end}"
                    values = samples[label]
                    if values:
                        axis.hist(values, bins=120, histtype="step", label=f"end {end} (n={len(values):,})")
                    center = gates[label]["center_ns"]
                    if np.isfinite(center):
                        axis.axvspan(center - self.args.prompt_half_width_ns,
                                     center + self.args.prompt_half_width_ns, alpha=0.12)
                        axis.axvline(center, linestyle="--", linewidth=0.8)
                axis.set(title=f"T5 bar {bar}", xlabel="Reference-corrected time (ns)", ylabel="Hits / bin")
                axis.legend(fontsize=8)
            fig.suptitle(f"T5 prompt peaks, bars {first_bar}-{first_bar + 3}")
            fig.tight_layout()
            _save(fig, self.out / f"beamline_prompt_peaks_t5_bars_{first_bar}-{first_bar + 3}.png", self._run_number)
        for first_hd in range(0, 15, 4):
            end_hd = min(first_hd + 3, 14)
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for element, axis in zip(range(first_hd, end_hd + 1), axes.flat):
                draw(axis, f"HD{element}")
            for axis in axes.flat[end_hd - first_hd + 1:]:
                axis.set_visible(False)
            fig.suptitle(f"HD element prompt peaks, HD{first_hd}-HD{end_hd}")
            fig.tight_layout()
            _save(fig, self.out / f"beamline_prompt_peaks_hd_{first_hd}-{end_hd}.png", self._run_number)

    def _plot_tof(self, values):
        labels = list(values)
        for page, start in enumerate(range(0, len(labels), 4), start=1):
            page_labels = labels[start:start + 4]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for label, axis in zip(page_labels, axes.flat):
                data = values[label]
                if data:
                    axis.hist(data, bins=120, histtype="step", color="tab:blue")
                axis.set(title=f"{label} (prompt-gated n={len(data):,})",
                         xlabel="Reference-corrected detector-time difference (ns)",
                         ylabel="Readouts / bin")
            for axis in axes.flat[len(page_labels):]:
                axis.set_visible(False)
            fig.suptitle("Beamline prompt-gated timing (no WCTE PMT quality selection)")
            fig.tight_layout()
            _save(fig, self.out / f"beamline_prompt_tof_page_{page}.png", self._run_number)

    def _plot_t5(self, multiplicities, end_differences):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        width = 0.25
        x = np.arange(8)
        for mult, color, label in ((0, "tab:gray", "0 ends"),
                                   (1, "tab:orange", "1 end"),
                                   (2, "tab:blue", "2 ends")):
            axes[0].bar(x + (mult - 1) * width, multiplicities[:, mult],
                        width=width, color=color, label=label)
        axes[0].set(xticks=x, xlabel="T5 bar", ylabel="Readouts",
                    title="Prompt-gated T5 end multiplicity")
        axes[0].set_yscale("log")
        axes[0].set_ylim(bottom=0.5)
        axes[0].legend()
        for bar, data in enumerate(end_differences):
            if data:
                axes[1].hist(data, bins=80,
                             range=(-self.args.end_diff_range_ns, self.args.end_diff_range_ns),
                             histtype="step", label=f"bar {bar}")
        axes[1].set(xlabel="T5 lower-numbered − higher-numbered end time (ns)",
                    ylabel="Paired hits / bin", title="Prompt-gated T5 end-time differences",
                    xlim=(-self.args.end_diff_range_ns, self.args.end_diff_range_ns))
        axes[1].legend(ncol=2, fontsize=8)
        fig.tight_layout()
        _save(fig, self.out / "beamline_prompt_t5_diagnostics.png", self._run_number)

    def _plot_hd_charge_progression(self, gates, in_gate, tof_center, tof_bounds):
        if not self.has_qdc:
            return {"available": False, "note": "HD QDC branches unavailable"}
        charge_matrix = np.concatenate(self.qdc_raw, axis=0)
        # qdc_raw is one row for every input readout. Root entry -> local row.
        start = self.args.entry_start
        good_rows = {row["entry"]: row for row in self.rows}
        stage_masks = {
            "Raw": np.ones(self.entries_seen, dtype=bool),
            "Beamline quality": np.zeros(self.entries_seen, dtype=bool),
            "Prompt T0": np.zeros(self.entries_seen, dtype=bool),
            "Prompt T0 + first T2-T0 peak": np.zeros(self.entries_seen, dtype=bool),
        }
        for entry, row in good_rows.items():
            index = entry - start
            if 0 <= index < self.entries_seen:
                stage_masks["Beamline quality"][index] = True
                is_prompt = in_gate(f"T0 average ({len(T0_CHANNELS)} channels)", row["t0"])
                stage_masks["Prompt T0"][index] = is_prompt
                stage_masks["Prompt T0 + first T2-T0 peak"][index] = (
                    is_prompt and row["t2"] is not None and np.isfinite(tof_center)
                    and tof_bounds[0] <= row["t2"] - row["t0"] <= tof_bounds[1])
        edges = np.linspace(0, 3000, 101)
        counts = {}
        files = []
        for first in range(0, 15, 4):
            elements = list(range(first, min(first + 4, 15)))
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for element, axis in zip(elements, axes.flat):
                counts[str(element)] = {}
                for stage, mask in stage_masks.items():
                    vals = charge_matrix[mask, element]
                    vals = vals[np.isfinite(vals)]
                    counts[str(element)][stage] = int(len(vals))
                    if len(vals):
                        axis.hist(vals, bins=edges, histtype="step", label=f"{stage} (n={len(vals):,})")
                axis.set_yscale("log")
                axis.set_ylim(bottom=0.5)
                axis.set(xlim=(0, 3000), xlabel="HD QDC charge (input units)",
                         ylabel="Readouts / bin", title=f"HD{element} QDC/TDC ch {ELEMENT_TO_HD_CHANNEL[element]}")
                axis.legend(fontsize=7)
            for axis in axes.flat[len(elements):]:
                axis.set_visible(False)
            fig.suptitle(f"HD charge cut progression — HD{elements[0]}–HD{elements[-1]}\n"
                         "Raw → beamline quality → T0 prompt → first T2−T0 peak")
            fig.tight_layout()
            name = f"beamline_hd_charge_cut_progression_hd_{elements[0]}-{elements[-1]}.png"
            _save(fig, self.out / name, self._run_number)
            files.append(name)
        return {"available": True, "plots": files, "counts_by_element_and_stage": counts,
                "display_range_input_units": [0, 3000], "histogram_bins": 100}

    def _plot_wcte_charge_progression(self, gates, in_gate, tof_center, tof_bounds):
        if not self.rows or all(row["wcte_charge"] is None for row in self.rows):
            return {"available": False, "note": "WCTE hit charge/mask branches unavailable"}
        stage_names = ["HD TDC hit", "+ T0 prompt", "+ no HC hit", "+ first T2-T0 peak"]
        has_t5 = any(row["t5_in_time_window"] is not None for row in self.rows)
        if has_t5:
            stage_names.append("+ no T5 prompt hit")
        counts = {}
        files = []
        per_hd = {}
        t0_label = f"T0 average ({len(T0_CHANNELS)} channels)"
        for element in range(15):
            tagged = [r for r in self.rows if element in r["hd"]]
            prompt = [r for r in tagged if in_gate(t0_label, r["t0"])]
            no_hc = [r for r in prompt if HC2_CHANNEL not in r["times"]]
            tof = [r for r in no_hc if r["t2"] is not None and np.isfinite(tof_center)
                   and tof_bounds[0] <= r["t2"] - r["t0"] <= tof_bounds[1]]
            stages = [tagged, prompt, no_hc, tof]
            if has_t5:
                stages.append([r for r in tof if r["t5_in_time_window"] is False])
            per_hd[str(element)] = {name: len(items) for name, items in zip(stage_names, stages)}
        all_charges = [row["wcte_charge"] for row in self.rows if row["wcte_charge"] is not None]
        upper = max(float(np.percentile(all_charges, 99.5)), 1.0)
        edges = np.linspace(min(0.0, min(all_charges)), upper, 101)
        for first in range(0, 15, 4):
            elements = list(range(first, min(first + 4, 15)))
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for element, axis in zip(elements, axes.flat):
                tagged = [r for r in self.rows if element in r["hd"]]
                prompt = [r for r in tagged if in_gate(t0_label, r["t0"])]
                no_hc = [r for r in prompt if HC2_CHANNEL not in r["times"]]
                tof = [r for r in no_hc if r["t2"] is not None and np.isfinite(tof_center)
                       and tof_bounds[0] <= r["t2"] - r["t0"] <= tof_bounds[1]]
                stages = [tagged, prompt, no_hc, tof]
                if has_t5:
                    stages.append([r for r in tof if r["t5_in_time_window"] is False])
                counts[str(element)] = {}
                for name, selected, color in zip(stage_names, stages,
                                                  ("0.35", "tab:blue", "tab:orange", "tab:green", "tab:red")):
                    values = [r["wcte_charge"] for r in selected if r["wcte_charge"] is not None]
                    counts[str(element)][name] = len(values)
                    if values:
                        axis.hist(values, bins=edges, histtype="step", label=f"{name} (n={len(values):,})", color=color)
                axis.set_yscale("log")
                axis.set_ylim(bottom=0.5)
                axis.set(title=f"HD{element} TDC ch {ELEMENT_TO_HD_CHANNEL[element]}",
                         xlabel="WCTE total PMT charge (input units)", ylabel="Readouts / bin",
                         xlim=(edges[0], edges[-1]))
                axis.legend(fontsize=7)
            for axis in axes.flat[len(elements):]:
                axis.set_visible(False)
            fig.suptitle(f"WCTE charge by HD TDC tag — HD{elements[0]}–HD{elements[-1]}\n"
                         "Cumulative cuts: T0 prompt → no HC2 → first T2−T0 peak"
                         + (" → no T5 prompt hit" if has_t5 else ""))
            fig.tight_layout()
            name = f"beamline_wcte_charge_cut_progression_hd_{elements[0]}-{elements[-1]}.png"
            _save(fig, self.out / name, self._run_number)
            files.append(name)
        return {"available": True, "plots": files, "stages": stage_names,
                "readout_counts_by_hd_and_stage": per_hd,
                "charge_counts_by_hd_and_stage": counts,
                "charge_definition": "sum finite PMT charges with per-hit readout mask == 0"}
