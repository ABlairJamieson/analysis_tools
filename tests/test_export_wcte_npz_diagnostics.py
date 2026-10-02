import csv
import json

import awkward as ak

from scripts.export_wcte_npz import CutDiagnostics, _bit_reasons


def test_cut_diagnostics_reports_overlapping_reasons(tmp_path):
    batch = ak.Array({
        "run_id": [1827, 1827, 1827],
        "event_number": [10, 11, 12],
        "readout_number": [100, 101, 102],
        "window_data_quality_mask": [0, 4, 0],
        "hit_pmt_readout_mask": [[0, 1], [0], [0, 2, 4]],
        "vme_digi_issues_bitmask": [0, 0, 1],
        "vme_evt_quality_bitmask": [0, 0, 0],
        "T5_HasValidHit": [True, True, False],
        "T5_HasMultipleScintillatorsHit": [False, False, True],
        "T5_HasInTimeWindow": [True, True, False],
    })
    report = CutDiagnostics(tmp_path, t5_quality=True, vme_quality=True,
                            available=set(batch.fields))
    report.record(batch)
    report.finish()

    data = json.loads((tmp_path / "cutflow.json").read_text())
    assert data["cutflow"] == {
        "raw_windows": 3, "after_window_quality": 2,
        "after_vme_quality": 1, "after_t5_quality": 1,
    }
    assert data["window_mask_reasons_nonexclusive"] == {"window:missing_waveforms": 1}
    assert data["hit_mask_reasons_nonexclusive_on_raw_windows"]["any_nonzero_mask"] == 3
    assert data["conditions_nonexclusive_on_raw_windows"]["T5_multiple_scintillators"] == 1
    with (tmp_path / "rejected_windows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["event_number"] for row in rows] == ["11", "12"]
    assert "window:missing_waveforms" in rows[0]["applied_rejection_reasons"]
    assert "T5_no_valid_hit" in rows[1]["applied_rejection_reasons"]
    assert (tmp_path / "quality_diagnostics.png").stat().st_size > 0


def test_unknown_mask_bits_are_identified():
    assert _bit_reasons(36, {1: "a", 4: "b"}) == ["b", "unknown_bits_0x20"]
