"""Small example plugin: compare corrected T0 and T2 times.

Copy this file to make a new study. Add ROOT inputs to ``required_branches``;
implement event calculations in ``process_batch`` and write final products in
``finish``. The runner passes the original Awkward batch plus common corrected
TDC hits and an AND-only selection book.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class Study:
    """Example accumulation study with an explicit T0-present child sample."""

    required_branches = ()
    t0_channels = (0, 1, 2, 3)
    t2_channels = (8,)

    def __init__(self, args):
        self.output_dir = args.output_dir
        self.t0_times = []
        self.t2_minus_t0 = []
        self.selection_counts = {}

    def process_batch(self, batch, context):
        """Calculate a few example quantities and accumulate counts per cut."""
        t0_values = []
        tof_values = []
        has_t0 = []
        for event_hits in context.corrected_tdc_hits:
            by_channel = {}
            for hit in event_hits:
                if hit.corrected_ns is not None:
                    by_channel.setdefault(hit.channel_id, hit.corrected_ns)
            good_t0 = all(channel in by_channel for channel in self.t0_channels)
            has_t0.append(good_t0)
            if good_t0:
                t0 = float(np.mean([by_channel[ch] for ch in self.t0_channels]))
                t0_values.append(t0)
                if all(ch in by_channel for ch in self.t2_channels):
                    t2 = float(np.mean([by_channel[ch] for ch in self.t2_channels]))
                    tof_values.append(t2 - t0)
                else:
                    tof_values.append(None)
            else:
                t0_values.append(None)
                tof_values.append(None)
        t0_selection = context.selections.add(
            "beamline_quality_and_T0", has_t0, parent="beamline_quality")
        self.selection_counts["beamline_quality"] = (
            self.selection_counts.get("beamline_quality", 0)
            + int(context.selections.mask("beamline_quality").sum()))
        self.selection_counts["beamline_quality_and_T0"] = (
            self.selection_counts.get("beamline_quality_and_T0", 0)
            + int(t0_selection.sum()))
        beamline_good = context.selections.mask("beamline_quality")
        self.t0_times.extend(value for value, keep in zip(t0_values, t0_selection)
                             if keep and value is not None)
        self.t2_minus_t0.extend(value for value, keep in zip(tof_values, beamline_good)
                                 if keep and value is not None)

    def finish(self):
        """Write basic PNGs and a machine-readable summary."""
        products = []
        for values, name, xlabel, bins, limits in (
            (self.t0_times, "t0_average.png", "Reference-corrected T0 average (ns)", 120, None),
            (self.t2_minus_t0, "t2_minus_t0.png", "T2 - T0 (ns)", 160, None),
        ):
            fig, axis = plt.subplots(figsize=(8, 5))
            if values:
                axis.hist(values, bins=bins, range=limits, histtype="step", color="C0")
            axis.set(xlabel=xlabel, ylabel="Readouts / bin", title=name.removesuffix(".png"))
            axis.grid(alpha=0.25)
            fig.tight_layout()
            path = self.output_dir / name
            fig.savefig(path, dpi=140)
            plt.close(fig)
            products.append(path.name)
        (self.output_dir / "study_summary.json").write_text(
            json.dumps({
                "selection_counts": self.selection_counts,
                "t0_values": len(self.t0_times),
                "t2_minus_t0_values": len(self.t2_minus_t0),
                "plots": products,
                "timing_note": "Beamline reference subtraction only; no WCTE clock alignment.",
            }, indent=2) + "\n",
            encoding="utf-8",
        )
