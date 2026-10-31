import argparse
import json

import awkward as ak
import numpy as np
import uproot

from scripts.study_wcte_beamline_prompt import main_peak, robust_group_mean, run


def test_main_peak_and_production_style_group_average():
    center, count, entries = main_peak([100.2, 100.3, 101.0, 500.0], bin_ns=2)
    assert abs(center - 100.0) < 2
    assert count >= 2
    assert entries == 4
    assert robust_group_mean({0: 10, 1: 11, 2: 12, 3: 310}, (0, 1, 2, 3)) == 11


def test_prompt_gated_beamline_outputs(tmp_path):
    root_path = tmp_path / "WCTE_merged_production_R1827.root"
    ids = [31, 46, *range(8), *range(48, 64), 32, 37, 17]
    corrected_times = [
        1000, 2000,
        1000, 1001, 1002, 1003, 1010, 1011, 1012, 1013,
        *[1030 + (ch - 48) * 0.1 for ch in range(48, 56)],
        *[1032 + (ch - 56) * 0.1 for ch in range(56, 64)],
        1040, 1047, 1045,
    ]
    raw_times = [corrected_times[0], corrected_times[1],
                 *[value + (1000 if channel <= 31 else 2000)
                   for channel, value in zip(ids[2:], corrected_times[2:])]]
    with uproot.recreate(root_path) as root:
        tree = root.mktree("WCTEReadoutWindows", {
            "beamline_pmt_tdc_ids": "var * int32",
            "beamline_pmt_tdc_times": "var * float64",
            "vme_digi_issues_bitmask": "int32",
        })
        tree.extend({
            "beamline_pmt_tdc_ids": ak.Array([ids, ids, ids]),
            "beamline_pmt_tdc_times": ak.Array([raw_times, raw_times, raw_times]),
            "vme_digi_issues_bitmask": np.array([0, 0, 0], dtype=np.int32),
        })
    out = tmp_path / "prompt"
    summary = run(argparse.Namespace(
        input_root=root_path, tree="WCTEReadoutWindows", output_dir=out,
        entry_start=0, max_events=10, prompt_half_width_ns=40,
        peak_bin_ns=2, end_diff_range_ns=40, t0_residual_range_ns=5,
    ))
    assert summary["run_number"] == 1827
    assert summary["beamline_quality_readouts"] == 3
    assert summary["prompt_gate_centers"]["T0 average (4 channels)"]["center_ns"] == 1001.5
    assert summary["t2_channels"] == [8]
    assert summary["tof_counts"]["T2-T0"] == 3
    assert summary["tof_counts"]["T0-T5"] == 24
    assert summary["tof_counts"]["T0-HD0"] == 3
    assert summary["tof_counts"]["T0-HD5"] == 3
    assert summary["tof_counts"]["T0-HD8"] == 3
    assert all(row == [0, 0, 3] for row in summary["t5_end_multiplicity"])
    assert (out / "beamline_prompt_peaks_t0_t2.png").stat().st_size > 0
    assert (out / "beamline_prompt_peaks_t5_bars_0-3.png").stat().st_size > 0
    assert (out / "beamline_prompt_peaks_t5_bars_4-7.png").stat().st_size > 0
    assert (out / "beamline_prompt_peaks_hd_0-3.png").stat().st_size > 0
    assert (out / "beamline_prompt_peaks_hd_4-7.png").stat().st_size > 0
    assert (out / "beamline_prompt_peaks_hd_8-11.png").stat().st_size > 0
    assert (out / "beamline_prompt_peaks_hd_12-14.png").stat().st_size > 0
    assert (out / "beamline_prompt_tof_page_1.png").stat().st_size > 0
    assert (out / "t0_channel_correction_comparison.png").stat().st_size > 0
    assert (out / "t0_channel_residuals_before_after_equalization.png").stat().st_size > 0
    assert (out / "t0_raw_vs_reference_subtraction.png").stat().st_size > 0
    assert (out / "t0_channels_vs_reference_2d.png").stat().st_size > 0
    assert (out / "t0_beam_spot_timing_proxy.png").stat().st_size > 0
    assert (out / "t0_beam_spot_timing_proxy_mm.png").stat().st_size > 0
    written_summary = json.loads((out / "summary.json").read_text())
    assert written_summary["prompt_gate_half_width_ns"] == 40
    diagnostic = written_summary["reference_subtraction_diagnostic"]
    assert set(diagnostic["per_channel"]) == {"0", "1", "2", "3"}
    assert diagnostic["per_channel"]["0"]["readouts"] == 3
    assert diagnostic["per_channel"]["0"]["raw_channel_median_ns"] == 2000
    assert diagnostic["per_channel"]["0"]["raw_channel31_median_ns"] == 1000
    assert diagnostic["per_channel"]["0"]["corrected_channel_median_ns"] == 1000
    assert diagnostic["per_channel"]["0"]["max_abs_difference_current_vs_direct_ns"] == 0
    zoom = written_summary["t2_t0_histogram_zoom"]
    assert zoom["all_pairs_total"] == 3
    assert zoom["all_pairs_in_view"] == 3
    beam_spot = written_summary["t0_beam_spot_timing_proxy"]
    assert beam_spot["events_with_all_four_t0_channels"] == 3
    assert beam_spot["vertical_top_minus_bottom_median_ns"] == -2
    assert beam_spot["horizontal_left_minus_right_median_ns"] == -1
    assert beam_spot["conversion_ns_per_m"] == 3
    assert beam_spot["scintillator_size_mm"] == [60, 60]
    assert beam_spot["events_inside_60x60mm"] == 3
