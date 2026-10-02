#!/usr/bin/env python3
"""Export quality-selected WCTE data in the WatChMaL WCSim NPZ hit convention.

The ROOT input is an already calibrated WCTEReadoutWindows production file.
DataLoader applies the WCTE window and hit quality masks. WCSimPMTMapping
converts (mPMT slot, PMT position) to zero-based WatChMaL digit PMT IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np

HIT_FIELDS = (
    "hit_pmt_calibrated_times",
    "hit_pmt_charges",
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
)
DQ_FIELDS = ("window_data_quality_mask", "hit_pmt_readout_mask")
OPTIONAL_FIELDS = ("run_id", "sub_run_id", "spill_counter", "readout_number", "window_time")
T5_FIELDS = ("T5_HasValidHit", "T5_HasMultipleScintillatorsHit", "T5_HasInTimeWindow")
VME_FIELDS = ("vme_digi_issues_bitmask", "vme_evt_quality_bitmask")
WINDOW_BITS = {1: "periodic_67_issue", 2: "slow_control_excluded", 4: "missing_waveforms",
               8: "mismatched_waveform_length", 16: "missing_trigger_signal"}
HIT_BITS = {1: "no_timing_constant", 2: "slow_control_excluded", 4: "manually_masked"}


def _bit_reasons(mask: int, definitions: dict[int, str]) -> list[str]:
    reasons = [name for bit, name in definitions.items() if mask & bit]
    unknown = mask & ~sum(definitions)
    if unknown:
        reasons.append(f"unknown_bits_0x{unknown:x}")
    return reasons


class CutDiagnostics:
    """Streaming, nonexclusive failure counts and sequential applied cut flow."""

    HIT_EDGES = np.array([0, 1, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, np.inf])

    def __init__(self, directory: Path, *, t5_quality: bool, vme_quality: bool,
                 available: set[str]):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.t5_quality = t5_quality
        self.vme_quality = vme_quality
        self.has_t5 = all(name in available for name in T5_FIELDS)
        self.has_vme = all(name in available for name in VME_FIELDS)
        self.cutflow: Counter[str] = Counter()
        self.window_reasons: Counter[str] = Counter()
        self.hit_reasons: Counter[str] = Counter()
        self.conditions: Counter[str] = Counter()
        self.hit_counts: Counter[str] = Counter()
        self.raw_hist = np.zeros(len(self.HIT_EDGES) - 1, dtype=np.int64)
        self.kept_hist = np.zeros(len(self.HIT_EDGES) - 1, dtype=np.int64)
        self.csv_path = directory / "rejected_windows.csv"
        self.csv_handle = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_handle, fieldnames=(
            "run_id", "sub_run_id", "event_number", "readout_number",
            "window_data_quality_mask", "applied_rejection_reasons",
        ))
        self.writer.writeheader()

    def record(self, batch) -> None:
        n = len(batch)
        window_masks = np.asarray(batch["window_data_quality_mask"], dtype=np.int64)
        window_good = window_masks == 0
        self.cutflow["raw_windows"] += n
        self.cutflow["after_window_quality"] += int(window_good.sum())

        hit_masks = ak.flatten(batch["hit_pmt_readout_mask"]).to_numpy().astype(np.int64)
        self.hit_counts["raw_hits"] += len(hit_masks)
        for bit, name in HIT_BITS.items():
            self.hit_reasons[name] += int(np.count_nonzero(hit_masks & bit))
        self.hit_reasons["any_nonzero_mask"] += int(np.count_nonzero(hit_masks))
        self.raw_hist += np.histogram(ak.num(batch["hit_pmt_readout_mask"]).to_numpy(),
                                      bins=self.HIT_EDGES)[0]
        good_window_hit_masks = batch["hit_pmt_readout_mask"][window_good]
        self.hit_counts["hits_in_good_windows"] += int(ak.sum(ak.num(good_window_hit_masks)))
        self.hit_counts["good_hits_in_good_windows"] += int(ak.sum(good_window_hit_masks == 0))

        vme_good = np.ones(n, dtype=bool)
        if self.has_vme:
            for field in VME_FIELDS:
                bad = np.asarray(batch[field]) != 0
                self.conditions[field] += int(bad.sum())
                vme_good &= ~bad
        t5_good = np.ones(n, dtype=bool)
        if self.has_t5:
            t5_failures = {
                "T5_no_valid_hit": ~np.asarray(batch["T5_HasValidHit"], dtype=bool),
                "T5_multiple_scintillators": np.asarray(batch["T5_HasMultipleScintillatorsHit"], dtype=bool),
                "T5_outside_time_window": ~np.asarray(batch["T5_HasInTimeWindow"], dtype=bool),
            }
            for name, bad in t5_failures.items():
                self.conditions[name] += int(bad.sum())
                t5_good &= ~bad
        after_vme = window_good & (vme_good if self.vme_quality else True)
        selected = after_vme & (t5_good if self.t5_quality else True)
        self.cutflow["after_vme_quality"] += int(after_vme.sum())
        self.cutflow["after_t5_quality"] += int(selected.sum())
        selected_good_hits = ak.sum(batch["hit_pmt_readout_mask"][selected] == 0, axis=1)
        self.kept_hist += np.histogram(np.asarray(selected_good_hits), bins=self.HIT_EDGES)[0]

        for i in range(n):
            reasons = [f"window:{name}" for name in _bit_reasons(int(window_masks[i]), WINDOW_BITS)]
            for name in reasons:
                self.window_reasons[name] += 1
            if self.vme_quality and not vme_good[i]:
                reasons.extend(f"vme:{field}" for field in VME_FIELDS if int(batch[field][i]) != 0)
            if self.t5_quality and not t5_good[i]:
                reasons.extend(name for name, bad in t5_failures.items() if bad[i])
            if not selected[i]:
                self.writer.writerow({
                    **{field: int(batch[field][i]) if field in batch.fields else ""
                       for field in ("run_id", "sub_run_id", "event_number", "readout_number")},
                    "window_data_quality_mask": int(window_masks[i]),
                    "applied_rejection_reasons": ";".join(reasons),
                })

    def finish(self) -> None:
        self.csv_handle.close()
        report = {
            "cutflow": dict(self.cutflow),
            "window_mask_reasons_nonexclusive": dict(self.window_reasons),
            "hit_mask_reasons_nonexclusive_on_raw_windows": dict(self.hit_reasons),
            "conditions_nonexclusive_on_raw_windows": dict(self.conditions),
            "hit_counts": dict(self.hit_counts),
            "applied_cuts": {"window_quality": True, "hit_quality": True,
                             "vme_quality": self.vme_quality, "t5_quality": self.t5_quality},
            "available_conditions": {"vme": self.has_vme, "t5": self.has_t5},
            "note": "Counts include complete ROOT batches scanned; with --max-events they may exceed exported windows. Failure reasons overlap.",
        }
        (self.directory / "cutflow.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        for axis, data, title in (
            (axes[0], self.window_reasons, "Window-mask reasons (overlapping)"),
            (axes[1], self.conditions, "Optional VME/T5 failures (overlapping)"),
        ):
            if data:
                axis.barh(list(data), list(data.values()))
            axis.set(xlabel="Raw windows", title=title)
        labels = [f"{int(lo)}–{int(hi) - 1}" if np.isfinite(hi) else f"{int(lo)}+"
                  for lo, hi in zip(self.HIT_EDGES[:-1], self.HIT_EDGES[1:])]
        x = np.arange(len(labels))
        axes[2].step(x, self.raw_hist, where="mid", label="Raw")
        axes[2].step(x, self.kept_hist, where="mid", label="After applied cuts + hit mask")
        axes[2].set_xticks(x, labels, rotation=65)
        axes[2].set(xlabel="Hits per window", ylabel="Windows", title="Hit multiplicity")
        axes[2].legend()
        fig.savefig(self.directory / "quality_diagnostics.png", dpi=160)
        plt.close(fig)
        print(f"Quality diagnostics: {self.directory}")


def _rows(values):
    """A one-dimensional object array, including when all rows have equal length."""
    out = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        out[i] = value
    return out


def _write_part(output: Path, part: int, columns: dict, single_part: bool):
    path = output if single_part else output.with_name(f"{output.stem}_part{part:05d}.npz")
    payload = {}
    for key, values in columns.items():
        payload[key] = _rows(values) if key.startswith("digi_hit_") else np.asarray(values)
    np.savez_compressed(path, **payload)
    print(f"{path}: {len(columns['event_id'])} windows", flush=True)
    return path


def export_root(
    input_path: Path,
    output_path: Path,
    *,
    max_events: int | None = None,
    max_input_windows: int | None = None,
    events_per_file: int = 5000,
    step_size: str = "100 MB",
    mapping_file: Path | None = None,
    t5_quality: bool = False,
    vme_quality: bool = False,
    diagnostics_dir: Path | None = None,
):
    from analysis_tools import DataLoader, WCSimPMTMapping

    if max_events is not None and max_events < 1:
        raise ValueError("--max-events must be positive")
    if max_input_windows is not None and max_input_windows < 1:
        raise ValueError("--max-input-windows must be positive")
    if events_per_file < 1:
        raise ValueError("--events-per-file must be positive")
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output path must end in .npz")

    mapping = WCSimPMTMapping(str(mapping_file) if mapping_file else None)
    with DataLoader(str(input_path), branches_to_load=[]) as loader:
        available = set(loader.file["WCTEReadoutWindows"].keys())
        required = (*HIT_FIELDS, *DQ_FIELDS, "event_number")
        if t5_quality:
            required += T5_FIELDS
        if vme_quality:
            required += VME_FIELDS
        missing = sorted(set(required) - available)
        if missing:
            raise ValueError("Missing required branches for requested cuts: " + ", ".join(missing))

        diagnostic_fields = (*T5_FIELDS, *VME_FIELDS) if diagnostics_dir is not None else ()
        loader.branches_to_load = list(dict.fromkeys(
            (*required, *(name for name in (*OPTIONAL_FIELDS, *diagnostic_fields) if name in available))
        ))
        loader.apply_mPMT_data_quality_cuts()
        if t5_quality:
            loader.apply_t5_event_quality_cuts()
        if vme_quality:
            loader.apply_vme_event_quality_cuts()

        names = ("digi_hit_time", "digi_hit_charge", "digi_hit_pmt",
                 "event_id", "source_file", *(
                     name for name in OPTIONAL_FIELDS if name in available
                 ))
        columns = {name: [] for name in names}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = (CutDiagnostics(diagnostics_dir, t5_quality=t5_quality,
                                      vme_quality=vme_quality, available=available)
                       if diagnostics_dir is not None else None)
        n_exported = n_unmapped = part = 0
        outputs = []
        raw_batches = loader.file["WCTEReadoutWindows"].iterate(
            expressions=loader.branches_to_load, step_size=step_size,
            entry_stop=max_input_windows, library="ak"
        ) if diagnostics is not None else None
        for raw_batch in raw_batches if raw_batches is not None else loader.iterate(
            step_size=step_size, entry_stop=max_input_windows
        ):
            if diagnostics is not None:
                diagnostics.record(raw_batch)
                batch = loader._apply_all_data_quality_cuts(raw_batch)
            else:
                batch = raw_batch
            for event in batch:
                times = np.asarray(event["hit_pmt_calibrated_times"], dtype=np.float64)
                charges = np.asarray(event["hit_pmt_charges"], dtype=np.float64)
                slots = np.asarray(event["hit_mpmt_slot_ids"], dtype=np.int64)
                positions = np.asarray(event["hit_pmt_position_ids"], dtype=np.int64)
                lengths = {len(x) for x in (times, charges, slots, positions)}
                if len(lengths) != 1:
                    raise ValueError(f"Unequal hit-array lengths at event {event['event_number']}")

                # A real PMT may be absent from a particular WCSim geometry.
                valid = ((slots >= 0) & (slots < mapping._valid_slotpos.shape[0])
                         & (positions >= 0) & (positions < mapping._valid_slotpos.shape[1]))
                in_range = valid.copy()
                valid[in_range] = mapping._valid_slotpos[slots[in_range], positions[in_range]]
                n_unmapped += int(np.count_nonzero(~valid))
                mapped = mapping.map_wcte_slot_pos_to_wcsim_tube_no(
                    slots[valid], positions[valid], use_watchmal_npz=True
                ) if np.any(valid) else np.empty(0, dtype=np.int64)
                columns["digi_hit_time"].append(times[valid])
                columns["digi_hit_charge"].append(charges[valid])
                columns["digi_hit_pmt"].append(np.atleast_1d(mapped).astype(np.int64))
                columns["event_id"].append(int(event["event_number"]))
                columns["source_file"].append(str(input_path))
                for name in OPTIONAL_FIELDS:
                    if name in columns:
                        columns[name].append(event[name])

                n_exported += 1
                if len(columns["event_id"]) == events_per_file:
                    outputs.append(_write_part(output_path, part, columns, single_part=False))
                    part += 1
                    columns = {name: [] for name in names}
                if max_events is not None and n_exported >= max_events:
                    break
            if max_events is not None and n_exported >= max_events:
                break

        if columns["event_id"]:
            outputs.append(_write_part(
                output_path, part, columns, single_part=(part == 0),
            ))
        print(f"Exported {n_exported} quality-selected windows; "
              f"excluded {n_unmapped} hits absent from WCSim geometry")
        if diagnostics is not None:
            diagnostics.finish()
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--max-events", type=int, help="Pilot limit after quality cuts")
    parser.add_argument("--max-input-windows", type=int,
                        help="Read only the first N ROOT windows, for matched cut comparisons")
    parser.add_argument("--events-per-file", type=int, default=5000)
    parser.add_argument("--step-size", default="100 MB")
    parser.add_argument("--mapping-file", type=Path,
                        help="WCSim geofile mapping; default is repository's v1.12.29 map")
    parser.add_argument("--t5-quality", action="store_true",
                        help="Also require the DataLoader T5 event quality selection")
    parser.add_argument("--vme-quality", action="store_true",
                        help="Also require the DataLoader VME event quality selection")
    parser.add_argument("--diagnostics-dir", type=Path,
                        help="Write cutflow.json, rejected_windows.csv, and quality_diagnostics.png")
    args = parser.parse_args()
    export_root(
        args.input_root, args.output_npz,
        max_events=args.max_events, max_input_windows=args.max_input_windows,
        events_per_file=args.events_per_file,
        step_size=args.step_size, mapping_file=args.mapping_file,
        t5_quality=args.t5_quality, vme_quality=args.vme_quality,
        diagnostics_dir=args.diagnostics_dir,
    )


if __name__ == "__main__":
    main()
