import argparse
import csv
import json

import awkward as ak
import numpy as np
import pytest
import uproot

import analysis_tools
from scripts.export_wcte_npz import export_root
from scripts.compare_wcte_burst_exports import compare
from scripts.study_wcte_burst_gaps import run as run_root
from scripts.study_wcte_npz_burst_gaps import run as run_npz


class FakeMapping:
    def __init__(self, _path):
        self._valid_slotpos = np.ones((10, 100), dtype=bool)

    def map_wcte_slot_pos_to_wcsim_tube_no(self, slots, positions, *, use_watchmal_npz):
        assert use_watchmal_npz
        return 100 * slots + positions


def _sample_root(path):
    prompt = list(1700 + np.arange(8) * 2)
    delayed = list(4020 + np.arange(8) * 2)
    times = prompt + delayed
    with uproot.recreate(path) as root:
        tree = root.mktree("WCTEReadoutWindows", {
            "run_id": "int32", "event_number": "int32", "readout_number": "int32",
            "window_data_quality_mask": "int32", "hit_pmt_calibrated_times": "var * float64",
            "hit_pmt_charges": "var * float64", "hit_pmt_readout_mask": "var * int32",
            "hit_mpmt_slot_ids": "var * int32", "hit_pmt_position_ids": "var * int32",
            "beamline_pmt_tdc_ids": "var * int32", "beamline_pmt_tdc_times": "var * float64",
        })
        tree.extend({
            "run_id": [1827] * 3, "event_number": [40, 41, 42],
            "readout_number": [100, 101, 102],
            "window_data_quality_mask": [0, 4, 0],
            "hit_pmt_calibrated_times": ak.Array([times, times, times]),
            "hit_pmt_charges": ak.Array([[1.] * 16] * 3),
            "hit_pmt_readout_mask": ak.Array([[0] * 16, [0] * 16, [0] * 15 + [4]]),
            "hit_mpmt_slot_ids": ak.Array([[1] * 16] * 3),
            "hit_pmt_position_ids": ak.Array([list(range(8)) * 2] * 3),
            "beamline_pmt_tdc_ids": ak.Array([[31, 46, 0, 8, 32]] * 3),
            "beamline_pmt_tdc_times": ak.Array([[100., 200., 110., 120., 235.]] * 3),
        })


def _settings(**extra):
    base = dict(selection="all", entry_start=0, min_ns=0, max_ns=10000,
                bin_ns=10, width_ns=50, min_pmts=4, min_separation_ns=100,
                max_bursts=20, hist_bin_ns=50, tdc_time_mode="reference",
                prompt_min_ns=1500, prompt_max_ns=1900)
    return argparse.Namespace(**(base | extra))


def test_converter_preserves_entry_and_beamline_for_root_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_tools, "WCSimPMTMapping", FakeMapping)
    source = tmp_path / "pilot.root"
    _sample_root(source)
    converted = tmp_path / "converted_npz"
    converted.mkdir()
    outputs = export_root(source, converted / "R1827.npz", events_per_file=1,
                          step_size="1 KB")
    assert len(outputs) == 2
    manifest = json.loads((converted / "R1827_conversion_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["root_entry_preserved"]
    assert manifest["beamline_tdc_fields_preserved"]
    with pytest.raises(FileExistsError, match="fresh directory"):
        export_root(source, converted / "R1827.npz", events_per_file=1)
    with np.load(outputs[0], allow_pickle=True) as part:
        assert part["root_entry"].tolist() == [0]
        assert part["beamline_pmt_tdc_ids"][0].tolist() == [31, 46, 0, 8, 32]
        assert part["digi_hit_pmt"][0].tolist() == list(range(100, 108)) * 2
    with np.load(outputs[1], allow_pickle=True) as part:
        assert part["root_entry"].tolist() == [2]
        assert len(part["digi_hit_time"][0]) == 15

    root_dir = tmp_path / "root_gaps"
    npz_dir = tmp_path / "npz_gaps"
    root_summary = run_root(_settings(input_root=source, output_dir=root_dir,
                                      scan_windows=3, batch_windows=2))
    npz_summary = run_npz(_settings(input_dir=converted, output_dir=npz_dir,
                                    entry_stop=3))
    assert root_summary["counts"]["selected_windows"] == 2
    assert npz_summary["counts"]["selected_windows"] == 2
    for name in ("bursts.csv", "gaps.csv", "t0_group_gaps.csv", "alignment_candidates.csv"):
        with (root_dir / name).open(newline="") as handle:
            root_rows = list(csv.DictReader(handle))
        with (npz_dir / name).open(newline="") as handle:
            npz_rows = list(csv.DictReader(handle))
        assert root_rows == npz_rows
    assert compare(root_dir, npz_dir)["equivalent_within_tolerance"]
    assert (npz_dir / "burst_gap_histograms.png").is_file()
