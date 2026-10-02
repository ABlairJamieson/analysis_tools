#!/usr/bin/env python3
"""Export quality-selected WCTE data in the WatChMaL WCSim NPZ hit convention.

The ROOT input is an already calibrated WCTEReadoutWindows production file.
DataLoader applies the WCTE window and hit quality masks. WCSimPMTMapping
converts (mPMT slot, PMT position) to zero-based WatChMaL digit PMT IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis_tools import DataLoader, WCSimPMTMapping

HIT_FIELDS = (
    "hit_pmt_calibrated_times",
    "hit_pmt_charges",
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
)
DQ_FIELDS = ("window_data_quality_mask", "hit_pmt_readout_mask")
OPTIONAL_FIELDS = ("run_id", "sub_run_id", "spill_counter", "readout_number", "window_time")


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
    events_per_file: int = 5000,
    step_size: str = "100 MB",
    mapping_file: Path | None = None,
    t5_quality: bool = False,
    vme_quality: bool = False,
):
    if max_events is not None and max_events < 1:
        raise ValueError("--max-events must be positive")
    if events_per_file < 1:
        raise ValueError("--events-per-file must be positive")
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output path must end in .npz")

    mapping = WCSimPMTMapping(str(mapping_file) if mapping_file else None)
    with DataLoader(str(input_path), branches_to_load=[]) as loader:
        available = set(loader.file["WCTEReadoutWindows"].keys())
        required = (*HIT_FIELDS, *DQ_FIELDS, "event_number")
        if t5_quality:
            required += ("T5_HasValidHit", "T5_HasMultipleScintillatorsHit", "T5_HasInTimeWindow")
        if vme_quality:
            required += ("vme_digi_issues_bitmask", "vme_evt_quality_bitmask")
        missing = sorted(set(required) - available)
        if missing:
            raise ValueError("Missing required branches for requested cuts: " + ", ".join(missing))

        loader.branches_to_load = list(dict.fromkeys(
            (*required, *(name for name in OPTIONAL_FIELDS if name in available))
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
        n_exported = n_unmapped = part = 0
        outputs = []
        for batch in loader.iterate(step_size=step_size):
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
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--max-events", type=int, help="Pilot limit after quality cuts")
    parser.add_argument("--events-per-file", type=int, default=5000)
    parser.add_argument("--step-size", default="100 MB")
    parser.add_argument("--mapping-file", type=Path,
                        help="WCSim geofile mapping; default is repository's v1.12.29 map")
    parser.add_argument("--t5-quality", action="store_true",
                        help="Also require the DataLoader T5 event quality selection")
    parser.add_argument("--vme-quality", action="store_true",
                        help="Also require the DataLoader VME event quality selection")
    args = parser.parse_args()
    export_root(
        args.input_root, args.output_npz,
        max_events=args.max_events, events_per_file=args.events_per_file,
        step_size=args.step_size, mapping_file=args.mapping_file,
        t5_quality=args.t5_quality, vme_quality=args.vme_quality,
    )


if __name__ == "__main__":
    main()
