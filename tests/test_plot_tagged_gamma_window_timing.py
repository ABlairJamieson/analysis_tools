import argparse
import csv

import awkward as ak
import uproot

from scripts.plot_tagged_gamma_window_timing import peak_bins, run, tag_matches


def test_repeated_t0_and_hd_hits_are_preserved():
    hits = [(0, 10.0), (0, 310.0), (8, 20.0), (32, 35.0), (32, 335.0)]
    assert tag_matches(hits) == [(32, 35.0, 10.0), (32, 335.0, 310.0)]
    assert len(peak_bins(ak.to_numpy(ak.Array([1710, 1711, 2010, 2011])),
                         0, 3000, 10, 2)) == 2


def test_root_window_diagnostic_outputs(tmp_path):
    root_path = tmp_path / "pilot.root"
    with uproot.recreate(root_path) as root:
        tree = root.mktree("WCTEReadoutWindows", {
            "run_id": "int32", "event_number": "int32", "readout_number": "int32",
            "window_data_quality_mask": "int32",
            "hit_pmt_calibrated_times": "var * float64",
            "hit_pmt_charges": "var * float64",
            "hit_pmt_readout_mask": "var * int32",
            "beamline_pmt_tdc_ids": "var * int32",
            "beamline_pmt_tdc_times": "var * float64",
            "T5_hit_time": "var * float64",
        })
        tree.extend({
            "run_id": [1827, 1827], "event_number": [40, 41],
            "readout_number": [100, 101], "window_data_quality_mask": [0, 0],
            "hit_pmt_calibrated_times": ak.Array([[1710, 1711, 2010, 2011], [1710]]),
            "hit_pmt_charges": ak.Array([[1, 1, 1, 1], [1]]),
            "hit_pmt_readout_mask": ak.Array([[0, 0, 0, 0], [0]]),
            "beamline_pmt_tdc_ids": ak.Array([[0, 0, 8, 32, 32], [0, 8, 11, 32]]),
            "beamline_pmt_tdc_times": ak.Array([[10, 310, 20, 35, 335], [10, 20, 25, 35]]),
            "T5_hit_time": ak.Array([[], [4.0]]),
        })

    output = tmp_path / "diagnostics"
    args = argparse.Namespace(input_root=root_path, output_dir=output, entry_start=0,
                              scan_windows=2, batch_windows=2, max_plots=10,
                              readout_numbers=None, selection="tagged",
                              pmt_min_ns=0, pmt_max_ns=3000,
                              bin_ns=10, min_peak_hits=2)
    assert run(args) == (2, 1)
    with (output / "window_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["root_entry"] == "0"
    assert rows[0]["n_hd_t0_matches"] == "2"
    assert rows[0]["n_pmt_peaks"] == "2"
    assert (output / rows[0]["plot"]).stat().st_size > 0
    with (output / "beam_hits.csv").open(newline="") as handle:
        hits = list(csv.DictReader(handle))
    assert [row["channel_id"] for row in hits].count("0") == 2
    assert [row["matched_t0_times"] for row in hits if row["channel_id"] == "32"] == ["10.0", "310.0"]
