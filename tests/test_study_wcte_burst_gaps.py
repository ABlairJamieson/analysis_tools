import argparse
import csv

import awkward as ak
import numpy as np
import uproot

from scripts.study_wcte_burst_gaps import find_bursts, run, t0_group_centers


def test_bursts_and_t0_groups():
    times = np.array([1700, 1702, 1704, 1706, 4020, 4022, 4024, 4026,
                      6650, 6652, 6654, 6656], dtype=float)
    bursts = find_bursts(times, np.ones(len(times)), np.arange(len(times)),
                         min_ns=0, max_ns=10000, bin_ns=10, width_ns=50,
                         min_pmts=4, min_separation_ns=100, max_bursts=20)
    assert len(bursts) == 3
    assert np.allclose(np.diff([burst.center_ns for burst in bursts]), [2320, 2630], atol=10)
    assert t0_group_centers([313, 0, 310, 3]) == [1.5, 311.5]


def test_root_burst_gap_outputs(tmp_path):
    root_path = tmp_path / "pilot.root"
    times = [1700, 1702, 1704, 1706, 4020, 4022, 4024, 4026]
    with uproot.recreate(root_path) as root:
        tree = root.mktree("WCTEReadoutWindows", {
            "run_id": "int32", "event_number": "int32", "readout_number": "int32",
            "window_data_quality_mask": "int32",
            "hit_pmt_calibrated_times": "var * float64",
            "hit_pmt_charges": "var * float64",
            "hit_pmt_readout_mask": "var * int32",
            "hit_mpmt_slot_ids": "var * int32",
            "hit_pmt_position_ids": "var * int32",
            "beamline_pmt_tdc_ids": "var * int32",
            "beamline_pmt_tdc_times": "var * float64",
        })
        tree.extend({
            "run_id": [1827, 1827], "event_number": [40, 41],
            "readout_number": [100, 101], "window_data_quality_mask": [0, 0],
            "hit_pmt_calibrated_times": ak.Array([times, times]),
            "hit_pmt_charges": ak.Array([[1.] * 8, [1.] * 8]),
            "hit_pmt_readout_mask": ak.Array([[0] * 8, [0] * 8]),
            "hit_mpmt_slot_ids": ak.Array([[1] * 8, [1] * 8]),
            "hit_pmt_position_ids": ak.Array([list(range(8)), list(range(8))]),
            "beamline_pmt_tdc_ids": ak.Array([[0, 0, 8, 32, 32], [0, 8, 11, 32]]),
            "beamline_pmt_tdc_times": ak.Array([[10, 310, 20, 35, 335], [10, 20, 25, 35]]),
        })
    output = tmp_path / "gaps"
    args = argparse.Namespace(input_root=root_path, output_dir=output, selection="tagged",
                              entry_start=0, scan_windows=2, batch_windows=2,
                              min_ns=0, max_ns=10000, bin_ns=10, width_ns=50,
                              min_pmts=4, min_separation_ns=100, max_bursts=20,
                              hist_bin_ns=50)
    summary = run(args)
    assert summary["counts"]["selected_windows"] == 1
    assert summary["counts"]["windows_with_multiple_bursts"] == 1
    with (output / "gaps.csv").open(newline="") as handle:
        gaps = list(csv.DictReader(handle))
    assert len(gaps) == 2
    assert {row["kind"] for row in gaps} == {"consecutive", "later_from_strongest"}
    assert np.isclose(float(gaps[0]["gap_ns"]), 2320, atol=10)
    with (output / "t0_group_gaps.csv").open(newline="") as handle:
        t0_gaps = list(csv.DictReader(handle))
    assert len(t0_gaps) == 1
    assert float(t0_gaps[0]["gap_ns"]) == 300
    assert (output / "burst_gap_histograms.png").stat().st_size > 0
    assert (output / "t0_vs_pmt_gap_histograms.png").stat().st_size > 0
