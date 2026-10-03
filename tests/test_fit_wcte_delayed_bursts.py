import argparse
import csv

import numpy as np
import pytest

from scripts.fit_wcte_delayed_bursts import fit_models, load_delays, run


FIELDS = ("root_entry", "run_id", "readout_number", "center_ns", "n_pmts")


def write_bursts(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_prompt_relative_selection(tmp_path):
    path = tmp_path / "bursts.csv"
    write_bursts(path, [
        dict(root_entry=0, run_id=1827, readout_number=45, center_ns=1700, n_pmts=100),
        dict(root_entry=0, run_id=1827, readout_number=45, center_ns=2030, n_pmts=50),
        dict(root_entry=0, run_id=1827, readout_number=45, center_ns=4020, n_pmts=20),
        dict(root_entry=1, run_id=1827, readout_number=57, center_ns=1700, n_pmts=100),
        dict(root_entry=1, run_id=1827, readout_number=57, center_ns=1750, n_pmts=100),
        dict(root_entry=1, run_id=1827, readout_number=57, center_ns=5000, n_pmts=20),
    ])
    delays, counts = load_delays(path, prompt_min_ns=1500, prompt_max_ns=1900,
                                  min_delay_ns=500, max_delay_ns=7500,
                                  min_prompt_pmts=10, min_delayed_pmts=10)
    assert delays.tolist() == [2320]
    assert counts["selected_windows"] == 1
    assert counts["windows_without_unique_prompt"] == 1


def test_rejects_unsorted_burst_export(tmp_path):
    path = tmp_path / "bursts.csv"
    write_bursts(path, [
        dict(root_entry=2, run_id=1827, readout_number=2, center_ns=1700, n_pmts=100),
        dict(root_entry=1, run_id=1827, readout_number=1, center_ns=1700, n_pmts=100),
    ])
    with pytest.raises(ValueError, match="ordered by ROOT entry"):
        load_delays(path, prompt_min_ns=1500, prompt_max_ns=1900,
                    min_delay_ns=500, max_delay_ns=7500,
                    min_prompt_pmts=10, min_delayed_pmts=10)


def test_lifetime_component_improves_synthetic_fit():
    edges = np.arange(500, 7501, 25)
    centers = (edges[:-1] + edges[1:]) / 2
    phase = np.array([1.7, 0.8, 0.5, 1.0])
    comb = phase[np.floor(np.mod(centers, 330) / 330 * len(phase)).astype(int)]
    observed = np.rint(20 + 35 * comb + 120 * np.exp(-centers / 2196.9811)).astype(int)
    fit = fit_models(observed, edges, phase, period_ns=330, lifetime_ns=2196.9811)
    assert fit["comb_plus_fixed_lifetime"]["coefficients"][2] > 50
    assert fit["twice_log_likelihood_improvement_descriptive"] > 100


def test_writes_summary_and_plot(tmp_path):
    path = tmp_path / "bursts.csv"
    rows = []
    for entry in range(40):
        rows.append(dict(root_entry=entry, run_id=1827, readout_number=entry,
                         center_ns=1700, n_pmts=100))
        for delay in (550, 880, 1210, 1540, 1870, 2200, 2530, 2860,
                      3190, 3520, 3850, 4180, 4510, 4840, 5170,
                      5500, 5830, 6160, 6490, 6820, 7150):
            rows.append(dict(root_entry=entry, run_id=1827, readout_number=entry,
                             center_ns=1700 + delay + entry % 5, n_pmts=20))
    write_bursts(path, rows)
    args = argparse.Namespace(bursts_csv=path, output_dir=tmp_path / "fit",
                              control_bursts_csv=None, prompt_min_ns=1500,
                              prompt_max_ns=1900, min_delay_ns=500,
                              max_delay_ns=7500, bin_ns=25, period_ns=330,
                              lifetime_ns=2196.9811, min_prompt_pmts=10,
                              min_delayed_pmts=10, phase_bins=12,
                              template_min_ns=5000, template_max_ns=7500,
                              min_template_bursts=50)
    summary = run(args)
    assert summary["counts"]["selected_windows"] == 40
    assert summary["template_source"] == "same_sample_late_sideband"
    assert (args.output_dir / "summary.json").stat().st_size > 0
    assert (args.output_dir / "histogram.csv").stat().st_size > 0
    assert (args.output_dir / "delayed_burst_lifetime_check.png").stat().st_size > 0
