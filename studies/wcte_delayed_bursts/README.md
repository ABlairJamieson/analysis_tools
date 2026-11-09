# WCTE delayed-burst study (real data)

This is the framework-based first rewrite of the real-data PMT burst study.
The shared `studies/wcte_beamline_prompt/runner.py` reads processed ROOT data
through `analysis_tools.DataLoader` and applies the common beamline TDC
reference correction once. This plugin then derives its own quality, beam, and
tagged selections and analyzes calibrated PMT hit times.

The observed beam-bunch structure (about 300 ns spacing) is intentionally left
in the sample. There is no bunch-phase veto, delayed-candidate veto, or cut
intended to remove those peaks in this first migration. The old
`scripts/study_wcte_burst_gaps.py` remains available for comparison.

Example from the repository root:

```bash
python3 studies/wcte_beamline_prompt/runner.py \
  /eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/1827/WCTE_merged_production_R1827.root \
  --study studies/wcte_delayed_bursts/wcte_delayed_bursts_study.py \
  --output-dir outputs/R1827_delayed_bursts_framework \
  --max-events 10000
```

Omit `--max-events` to process the full file. The plugin defaults to
`--selection tagged`, matching the earlier study's default. Use
`--selection all` to examine all readout-quality-good windows, or `--selection beam` for
T0+T2 windows without an HC2 hit. `--progress-every 1` gives a message after
each ROOT batch.

The output includes a run-stamped basic plot set, `summary.json`, and CSV tables
for per-readout burst multiplicity, burst properties, gap categories, and
delayed candidates with no veto applied. Raw-hit histograms are accumulated
batch-by-batch to keep memory use bounded on large runs.
