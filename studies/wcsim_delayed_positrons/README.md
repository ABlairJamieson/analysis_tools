# WCSim delayed-positron candidate study

This folder contains the analysis-specific plugin only. Shared plugin loading,
selections, context, batching, progress reporting, and source adapters live in
`studies/framework/`.

The ROOT input is decoded directly with WatChMaL DataTools' `WCSimFile`
(`root_utils/root_file_utils.py`). The runner does not use the processed-data
`analysis_tools.DataLoader`, whose `WCTEReadoutWindows` schema is a different
kind of ROOT file, and it does not create an intermediate NPZ. Set `$DATATOOLS`
to the DataTools checkout, or pass `--datatools-path`. PyROOT/WCSim runtime
setup is required by DataTools.

Run from the `analysis_tools` repository root:

```bash
python3 studies/framework/runner.py \
  /path/to/mdt_wCDS_pi+_Uniform_0_800MeV_0001.root \
  /path/to/mdt_wCDS_pi-_Uniform_0_800MeV_0000.root \
  --study studies/wcsim_delayed_positrons/wcsim_delayed_positron_study.py \
  --output-dir outputs/delayed_pion_pilot \
  --max-events-per-file 1000
```

The default is a 100-event pilot. Use `--all-events` for each complete ROOT
file, `--step-events` to control plugin batch size, and `--progress-every` to
set progress frequency. The plugin writes `events.csv`, `clusters.csv`, and
`summary.json`.

Defaults preserve the previous scan: 50 ns sliding windows, at least 10
distinct PMTs for prompt and delayed candidates, and a 200–10,000 ns delayed
search relative to the prompt-cluster median time. `--time-mode auto` adds
`trigger_time[digi_hit_trigger]` to digit times when the DataTools ROOT event
provides both arrays; otherwise the summary records
`raw_missing_trigger_metadata`. `raw` and `trigger-plus` are explicit
alternatives. DataTools supplies the digit and trigger times; the analysis
adds the trigger offset at most once and does not apply another calibration.

Rank 1 is not simply the second cluster in time or the cluster with the most
digit hits. It is the delayed window with the most distinct PMTs, with total
charge as the tie-break; nearby overlapping maxima are suppressed. Read
`summary.json` and inspect candidate delays before interpreting them as Michel
positrons. A short observed digit span is a warning, not a readout-coverage
measure.

To add a study variation, copy `wcsim_delayed_positron_study.py` and select it
with `--study`. The plugin receives a list of event dictionaries with DataTools
fields `digi_hit_time`, `digi_hit_pmt`, `digi_hit_charge`, `digi_hit_trigger`,
`trigger_time`, `pid`, and `energy`. Add named cuts through
`context.selections`; do not duplicate ROOT reading or modify the shared
runner. For a new input format, add a source adapter in `studies/framework/`.
