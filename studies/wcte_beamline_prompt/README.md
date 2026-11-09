# Extensible WCTE prompt-beamline studies

This package separates the processed-data ROOT reader from study-specific
calculations and plots. `runner.py` streams `DataLoader` batches and supplies
the common channel-31/channel-46 corrected TDC hits. A study plugin implements
`Study.process_batch(batch, context)` and `Study.finish()`.

## The two study files

- `example_study.py` is intentionally small and is the recommended student
  starting point. Copy it to a new file to try a study-specific observable.
- `wcte_beamline_prompt_study.py` is the migrated beamline prompt study. It
  produces prompt-peak pages for T0/T2, T5, and HD channels; prompt-gated TOF
  distributions; T5 end multiplicity/time differences; HD QDC cut progression;
  WCTE charge distributions by HD tag; CSV tables; and `summary.json`.
- `scripts/study_wcte_beamline_prompt.py` remains unchanged as the reference
  implementation while the DataLoader port is checked against its output.

Run a small example from the repository root:

```bash
python3 studies/wcte_beamline_prompt/runner.py \
  /path/to/WCTE_merged_production_R1827.root \
  --study studies/wcte_beamline_prompt/example_study.py \
  --output-dir outputs/R1827_prompt_example \
  --max-events 10000 --step-size "100 MB"
```

Run the migrated analysis:

```bash
python3 studies/wcte_beamline_prompt/runner.py \
  /path/to/WCTE_merged_production_R1827.root \
  --study studies/wcte_beamline_prompt/wcte_beamline_prompt_study.py \
  --output-dir outputs/R1827_prompt_study \
  --step-size "100 MB"
```

The runner prints progress every 10 batches by default. Lower `--step-size` if
memory is constrained. Study-specific settings (for example
`--prompt-half-width-ns`, `--peak-bin-ns`, and `--tof-first-peak-half-width-ns`)
are registered by the selected plugin; see `--help` after specifying
`--study`.

## Writing a plugin

Copy `example_study.py`, then:

1. Add non-standard ROOT input names to `required_branches` or
   `optional_branches` on `Study`.
2. In `process_batch(batch, context)`, calculate observables from that batch.
   `batch` is the original Awkward record; `context.corrected_tdc_hits[i]`
   contains corrected beamline TDC hits for readout `i`.
3. Add cuts with `context.selections.add(name, mask, parent="...")`. A new
   mask is ANDed with its parent; existing selections cannot be overwritten.
4. Accumulate only what is needed and write plots/tables in `finish()`.

`context.reference_times_ns[i]` contains the raw channel-31 and channel-46
times. The corrected TDC hits already have the relevant reference subtracted
once: channel 31 for IDs <=31 and channel 46 for IDs >31. Do not subtract a
reference again. This correction aligns beamline channels within their TDC
banks, not to the WCTE PMT clock.

The runner starts with `all` and a `beamline_quality` selection, requiring both
reference channels and, when available, a clear `vme_digi_issues_bitmask`.
WCTE PMT quality cuts are not implicitly applied. DataLoader's optional
quality/tagged-gamma filters also remain off unless explicitly enabled.

The migrated plugin preserves the main beamline timing and charge-cut products.
It is a framework port under validation, not yet asserted to be byte-for-byte
equivalent to every ancillary diagnostic from the legacy script. Compare its
`summary.json` and plots with the legacy output before relying on changed
selection behavior.
