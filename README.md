This repository contains some common WCTE analysis tools as a python package, the tools for processing the data are stored here as well as tools for analysing processed data. If you follow the installation instructions you can use these tools running import analysis_tools.

Examples for using some of these tools can be found in the analysis_examples directory and are also listed https://wcte.hyperk.ca/wg/simulation-and-analysis/data-analysis-1 .

The scripts used in the WCTE data production are included in the scripts repository. 

The layout of this repository is as follows:

```
analysis_tools/
├── analysis_examples/ #contains examples on using the analysis tools
├── analysis_tools/ #contains the python package for analysis - can be installed using instructions below
├── cpp_merger/ #contains the cpp merging code for the data production
├── scripts/ #contains the python scripts for running the production
├── data/ #contains data needed for the package functions
├── extern/ #contains external packages linked to this repository as submodules
├── include/ #contains data used by the beam analysis scripts 
├── notebooks/ #contains older example notebooks (from August 2025 workshop)
```

## Export calibrated real-data hits for NPZ analyses

The scripts/export_wcte_npz.py adapter reads a production ROOT file with
WCTEReadoutWindows. It uses analysis_tools.DataLoader to reject windows with
nonzero window_data_quality_mask and hits with nonzero hit_pmt_readout_mask.
It maps each WCTE (mPMT slot, PMT position) to the zero-based WCSim PMT ID
used by WatChMaL DataTools NPZ files, using WCSimPMTMapping. Hits absent
from the selected WCSim geometry are omitted and counted. The input must
already contain calibrated hit times and charges.

For a small check:

    python3 scripts/export_wcte_npz.py \
      /path/to/WCTE_merged_production_R1827.root \
      /path/to/output/R1827_test.npz --max-events 10

For a cut-flow pilot, add `--diagnostics-dir /path/to/output/R1827_diagnostics`.
This writes `cutflow.json`, a `rejected_windows.csv` listing every rejected
window and its applied rejection reasons, and `quality_diagnostics.png` with
window-mask reasons, optional VME/T5 failure counts, and before/after hit
multiplicity. Mask reasons overlap: a window or hit can fail multiple checks.
Optional VME/T5 failure counts are shown when those branches exist, even if
their cuts were not enabled; `applied_cuts` in JSON identifies what actually
removed windows. With `--max-events`, diagnostics cover complete ROOT batches
scanned and may count more windows than were exported. Use a small
`--step-size` for a tightly bounded pilot. The diagnostic plot requires
matplotlib; ordinary conversion does not.
For a matched comparison of default and T5-selected samples, run two pilot
exports with the same `--max-input-windows N` and different output paths,
adding `--t5-quality` to the second. This bounds the number of raw ROOT
windows, unlike `--max-events`, which counts only exported windows. In the
tagged-gamma setup T5 is a charged-particle veto/TOF detector: the
T5-selected export is a charged-particle control, **not** a cleaner
tagged-photon sample.

### Tagged-gamma readout-window timing diagnostic

`scripts/plot_tagged_gamma_window_timing.py` examines a bounded number of
ROOT readout windows before attempting bunch separation. For example, on
lxplus from this repository:

```bash
python3 scripts/plot_tagged_gamma_window_timing.py \
  /eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/1827/WCTE_merged_production_R1827.root \
  --output-dir outputs/R1827_timing_pilot --scan-windows 10000 --max-plots 20
```

The default `tagged` selection requires a quality-good window, T0 and T2
TDC hits, no HC2 hit, and at least one hodoscope hit in the published
T0-relative timing windows. It is a *window-level diagnostic*, not a
unique-particle or unique-bunch tag; T2 is not yet paired to a particular
T0 hit. The window/hit readout masks are applied, but the run's additional
good-PMT channel list is not. Pass `--selection beam` to omit the HD
requirement or `--selection all` to inspect every quality-good window.
Use `--entry-start` or `--readout-numbers` to inspect specific windows.

Each selected window gets a PNG with separate WCTE PMT, beamline TDC, and
T5 panels. The panels deliberately have independent clock origins: an
offset between PMT, beamline, and T5 systems has not been established.
By default the script subtracts beamline TDC references 31 and 46, following
`beam_monitors_pid.py`: IDs <=31 use reference 31, IDs >31 use reference 46.
Missing references leave the affected bank unavailable for tagging instead
of silently using raw times. `beam_hits.csv` stores both raw and corrected
times. Use `--tdc-time-mode raw` only to compare with older plots. After
checking a measured cross-system offset, use `--tdc-to-pmt-offset-ns VALUE`
to show reference-corrected TDC hits on the WCTE PMT axis. T5 stays separate.
`window_summary.csv` retains ROOT entry/event/readout identifiers;
`beam_hits.csv` retains *every* TDC hit, including repeated channel IDs,
with HD-to-T0 matches; `pmt_peaks.csv` records high-count histogram bins
as visual aids, not particle assignments. T5 is displayed as a possible
charged-particle indicator and is not required for tagged photons.

### Aggregate within-window burst-gap study

`scripts/study_wcte_burst_gaps.py` scans many ROOT readout windows and
histograms gaps between dense PMT-hit bursts. Start with a bounded Run 1827
sample on lxplus:

```bash
python3 scripts/study_wcte_burst_gaps.py \
  /eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/1827/WCTE_merged_production_R1827.root \
  --output-dir outputs/R1827_burst_gaps --scan-windows 100000
```

The default `tagged` selection is the same *window-level* selection as the
timing diagnostic above. Each PMT burst is a local maximum in a 50 ns
time-density window, requiring at least 10 distinct PMTs, with maxima
separated by 100 ns. These are exploratory, configurable thresholds, not
particle identifications. Try `--selection beam` for the broader sample or
`--min-pmts 20` to check threshold sensitivity. Runtime scales with the
number of ROOT windows scanned; increase `--scan-windows` after the pilot.

`burst_gap_histograms.png` shows consecutive-burst gaps and gaps from the
strongest burst to each later burst, split by the number of observed T0
time groups. The linear panels zoom to 0–1500 ns; the log panels show the
full range. `t0_vs_pmt_gap_histograms.png` compares the measured T0
group-gap and PMT burst-gap shapes. T0 hits less than 50 ns apart are
grouped; **50 ns is a grouping resolution, not an assumed T9 bunch period**.
`bursts.csv`, `gaps.csv`, and `t0_group_gaps.csv` preserve per-window values
and ROOT entry/readout IDs; `summary.json` records counts and thresholds.
The study now uses TDC references 31/46 by default, and writes
`alignment_candidates.csv` and `alignment_candidates.png` from windows with
one T0 group and one PMT burst in the configurable 1500–1900 ns prompt
range. Inspect that distribution and its stability across run segments
before using its median for `--tdc-to-pmt-offset-ns`. The empirical offset
includes beam-particle and light-propagation delays.
The beamline TDC and WCTE PMT clocks remain unaligned, and a missing later
T0 group does not prove that a later PMT burst was not beam-related. Gap
peaks alone cannot assign individual particles or distinguish delayed
physics from instrumental structure.

For a full run, omit --max-events. The script writes files such as
R1827_part00000.npz, with 5000 quality-selected windows per file by default.
Use --events-per-file to change that limit. The output contains
digi_hit_pmt, digi_hit_time, digi_hit_charge, and event_id, plus available
run/readout identifiers. window_time is retained separately as metadata;
the calibrated hit times are left in their original readout-window time
convention. The exporter does not invent WCSim trigger indices or truth.

The default WCSim map is the package's v1.12.29 geometry. Use
--mapping-file /path/to/wcsim_geofile.txt if the simulation analysis uses a
different geometry. Optional --t5-quality and --vme-quality flags apply
those existing DataLoader selections when the corresponding branches are
present. Check time and charge distributions before comparing real data
with simulation: this adapter does not perform detector-response calibration.

Every export writes `<output-stem>_conversion_manifest.json` beside the NPZ
parts, recording its input, output parts, cuts, mapping, and timing provenance.
The converter copies `hit_pmt_calibrated_times` and applies **no additional
PMT timing offsets**. Production obtains run-valid `timing_offsets` from the
WCTE calibration database and subtracts them upstream. If the merged ROOT
`Configuration` tree lacks the exact calibration revision or numerical
constants, the manifest explicitly says so. The batch worker moves this
manifest into `converted_npz` with the parts.

Do not subtract an external LED-offset JSON again from this calibrated branch.
Before using one to reprocess *raw* hit times, verify its run/time validity,
official status, PMT-ID convention, sign, and the upstream revision.

### Batch conversion on CERN

The batch helper has the 35 commissioning run numbers 1786-1842 supplied for
this analysis as its default list. It expects inputs at
/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/
production_v1_0/<run>/WCTE_merged_production_R<run>.root and writes NPZ
parts in each run's converted_npz directory. Run the helper from a checkout
on EOS. It reports missing inputs and unwritable output directories before
submission.

First prepare and submit a single-run pilot:

    python3 scripts/prepare_wcte_npz_batch.py --runs 1827
    module load lxbatch/eossubmit
    condor_submit outputs/wcte_npz_batch/convert_wcte_npz.sub

After inspecting the pilot output and conversion.done marker, prepare the
full list and submit it:

    python3 scripts/prepare_wcte_npz_batch.py
    wc -l outputs/wcte_npz_batch/jobs.txt
    condor_submit outputs/wcte_npz_batch/convert_wcte_npz.sub

One job processes each run. The worker sources the CERN LCG_110_swan view,
uses the quality-selected exporter, and writes a conversion.done marker
only when all NPZ parts for that run have completed. It refuses to overwrite
existing NPZ files without that marker. Inspect any failed run's logs and
.conversion_in_progress directory before resubmitting.

# Installation

Installation:

```
git clone <git repo location>
cd analysis_tools
pip install -e .
```

The `-e` flag allows you to edit the package 
If using on lxplus you will need to setup this in a python virtual environment 

Some of the analysis examples use external packages linked to this repository as submodules. To install the external packages run:
```
git submodule update --init --recursive
```
Since some of the modules are private you need to set up ssh access to github for this to work - see https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account . Note you can check your ssh is working by running ssh -T git@github.com. 

# Contribution Rules
Main branch on WCTE/analysis_tools is protected - please open a pull request (either from your own branch or fork) to push changes to main branch

# Package classes and functions

##  WaveformProcessing

Waveform processing contains a copy of the CFD used in the test stand repository

`WaveformProcessingTeststand.cfd_teststand_method()` processes the CFD using that method returning 
the charge and the time for a pulse in that waveform - including non-linearity corrections 
for both

Additionally the same CFD and charge calculation method used online by the mPMT is included 
in WaveformProcessingmPMT the versions to run on single waveforms and vectorised versions to run on arrays of waveforms are given 

## Pulse finding

Finds pulses in the waveforms using the same method as run online on the mPMT.

`do_pulse_finding` is a slower version processing one waveform at a time and returning a list of found pulses

`do_pulse_finding_vect` is a mostly vectorised version, returning a list of lists of found pulses for each waveform

`do_pulse_finding_fast` is a more optimised, fully-vectorised version that returns results as arrays of rows (waveforms)
and columns (samples) of all pulses found in all waveforms

## CalibrationDBInterface

Interfaces with the calibration database see more instruction here

https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation

Currently processed for the test database - to be updated when the production database 
is ready. The authentication requires a credential text file ./.wctecaldb.credential 
to be in the current working directory - more details in the database interface above

## PMTMapping 

PMTMapping is a class containing the mapping of the WCTE PMTs slot and position ids (position in the detector) to the channel and mPMT card ids (electronics numbering) and vice versa. This is generally not needed as processed data already has the mPMTs mapped to their position and slot numbers. 
Usage:
```
mapping = PMTMapping()
mapping.get_slot_pmt_pos_from_card_pmt_chan(card_id,pmt_channel)
```
 returns the slot and pmt position
and 
```
mapping.get_card_pmt_chan_from_slot_pmt_pos(slot_id,pmt_position)
```
returns the card and channel

The mapping json is located in the package

## DetectorGeometry

Class to load PMT positions, directions and calculate time of flight.

# Data Production Scripts

## production v_0
### add_timing_constants.py

The earliest version of production script for production v_0 which added timing constants to self-trigger data

## production v_0_5
### process_data_v0_5.py

Self-trigger data production v 0_5 (see https://wcte.hyperk.ca/wg/simulation-and-analysis/data-production-2) 
for self-trigger data which includes timing constants and data quality flags

## production v_1

Production v_1 processes both self-trigger and hardware-trigger data through a multi-step pipeline.
A top-level orchestrator script (`run_pipeline.py`) reads run metadata from the run info JSON,
determines the trigger type and beam analysis mode automatically, and calls the appropriate
pipeline scripts. Intermediate outputs are written to step-specific subdirectories under
`<output_base>/<run_number>/`.

### Top-level orchestrator

#### `run_pipeline.py`

The recommended entry point for production. For a given run it:
1. Reads trigger type and beam analysis mode from the run info JSON via `get_run_info()`.
2. Runs the appropriate trigger-specific pipeline (`run_hw_trigger_pipeline.py` or `run_self_trigger_pipeline.py`).
3. Conditionally runs beam monitor PID analysis (`WCTE_beam_analysis.py`) depending on the beam configuration.
4. Runs the T5 beam monitor analysis 

Usage:
```bash
python run_pipeline.py \
  -r <run_number> \
  -i <input_file(s)> \
  -o <output_base_dir> \
  [--steps wf calibrate dq beam] \
  [--debug]
```

**`--steps`** controls which parts of the pipeline run. If not provided all steps run by default.
If `wf` or `calibrate` is specified, the corresponding downstream steps are also re-run
(e.g. `--steps calibrate` also runs `dq`).

#### Beam analysis mode

The decision to run VME beam processing is determined automatically from the run info JSON.
If the run is in a tagged gamma configuration or downstream ACTs (act3–5) are missing,
the VME processing will not run.

### Pipeline runner scripts

These can also be run standalone (e.g. to reprocess a single step).

#### `run_self_trigger_pipeline.py`

Orchestrates self-trigger data processing through two steps:

```
Step 1 (calibrate) : calibrate_hits.py        → calibrated_hits/
Step 2 (dq)        : self_trigger_dq_flags.py → dq_flags/
```

Usage:
```bash
python run_self_trigger_pipeline.py \
  -i <input_file(s)> -r <run_number> -o <output_base_dir> [--steps calibrate dq] [--debug]
```

#### `run_hw_trigger_pipeline.py`

Orchestrates hardware-trigger data processing through three steps:

```
Step 1 (wf)        : hw_trigger_wf_processing.py → processed_waveforms/
Step 2 (calibrate) : calibrate_hits.py            → calibrated_hits/
Step 3 (dq)        : hw_trigger_dq_flags.py       → dq_flags/
```

Usage:
```bash
python run_hw_trigger_pipeline.py \
  -i <input_file(s)> -r <run_number> -o <output_base_dir> [--steps wf calibrate dq] [--debug]
```

Both pipeline scripts support `--steps` (see above). The dependency expansion is applied
automatically — requesting `calibrate` will also run `dq`.

### Intermediate scripts

#### `hw_trigger_wf_processing.py`

Runs pulse finding and charge/time determination on hardware-trigger raw waveform data.
Reads `WCTEReadoutWindows` trees and writes a `ProcessedWaveforms` ROOT tree.

#### `calibrate_hits.py`

Applies timing constants from the calibration database to hit times.
Works on both self-trigger (`WCTEReadoutWindows`) and waveform-processed (`ProcessedWaveforms`) input files.
Writes a `CalibratedHits` ROOT tree and a `Configuration` tree recording the git hash,
timing constant revision, and list of PMTs with timing constants.

#### `self_trigger_dq_flags.py`

Applies data quality flags for self-trigger runs. Determines the good channel list from the
intersection of slow-control stable channels and channels with calibration constants.
Applies trigger-level bitmask flags for slow-control excluded periods and the 67 ms periodic issue.
Applies hit-level bitmask flags for missing timing constants and slow-control unstable channels.

#### `hw_trigger_dq_flags.py`

Applies data quality flags for hardware-trigger runs. In addition to the channel-level flags above,
applies trigger-level flags for missing waveforms, missing trigger signals, and mismatched waveform lengths.

### Output structure

```
<output_base>/<run_number>/
  processed_waveforms/   (hw-trigger only)
    <base>_processed_waveforms.root
  calibrated_hits/
    <base>_calibrated_hits.root
  dq_flags/
    <base>_[self|hw]_trigger_dq_flags.root
    <base>_[self|hw]_trigger_dq_flags_status.json
  beam_data/             (normal beam runs only)
    beam_analysis_output_R<run_number>.root
```


### Shared utilities

Common functions used across all production scripts are in `analysis_tools/production_utils.py`:
- `get_run_info` — reads trigger type and beam analysis mode from the run info JSON
- `get_git_descriptor` — git provenance for output Configuration trees
- `file_sha256` — hash of the slow-control input file used
- `get_run_database_data`, `get_stable_mpmt_list_slow_control` — slow-control data access
- `get_slow_control_trigger_mask`, `get_67ms_mask` — trigger-level DQ masks
- `slot_pos_from_card_chan_list` — PMT channel mapping


# Beam monitor PID

`scripts/WCTE_beam_analysis.py` performs particle identification (PID) and momentum estimation for beamline runs, using the VME beam-monitor detectors (T0, T1, T4, T5/TOF, and the ACT Cherenkov counters). It calibrates the ACT PMTs (1 p.e. calibration), tags particle species event-by-event, and estimates the momentum of each particle species both just after the CERN beam pipe and at the WCTE tank window.

It is a **template** PID: reasonable defaults are used throughout, but the selection is not tuned or optimised for any specific physics analysis. Treat it as a starting point to build your own beam PID on top of, not as a final analysis tool.

The logic lives in `BeamAnalysis` (`analysis_tools/beam_monitors_pid.py`); the script is a thin CLI wrapper around it.

## Detectors used

| Detector | Channels | Role |
|---|---|---|
| T0 | 0–3 | Upstream timing reference (start of TOF) |
| T1 | 4–7 | Downstream-of-T0 timing (T0–T1 TOF used for proton/deuteron/helium-3/triton tagging) |
| Hole counter (HC) | 9–10 | Vetoes triggers with an off-axis/halo hit |
| ACT0–2 ("e-veto") | 12–17 | Cherenkov counters used to tag electrons/positrons |
| ACT3–5 ("tagger") | 18–23 | Cherenkov counters used to separate muons/pions (ACT5 optional — some runs don't have it) |
| T4 | 42–43 | Further downstream timing reference |
| T5 / TOF bars | 48–63 (8 bars × 2 SiPMs) | End-of-line scintillator bars; also used as a "reached the tank" requirement |
| Muon tagger | subset of ACT3–5 channels | Auxiliary muon tag, used when the ACT3-5 charge distribution doesn't show a clean muon/pion minimum |

Reference TDC channels 31 and 46 are used to correct all other times.

## Requirements / inputs

Run the script from the `scripts/` directory — it resolves two paths relative to its own location:
- `../analysis_tools` (the package, via `sys.path.append("../")`)
- `../include/1pe_calibration.json` (ACT PMT gain/pedestal calibration)

It also reads, from fixed EOS paths (not configurable via CLI):
- `/eos/experiment/wcte/configuration/run_info/google_sheet_beam_data.json` — run metadata (beam momentum, ACT refractive indices, beam configuration, whether ACT5 is installed, etc.), compiled by Laurence from the Google sheet
- `/eos/experiment/wcte/configuration/run_info/beamline_equipment_settings.csv` — collimator/acceptance jaw settings

Input ROOT file(s) must contain a `WCTEReadoutWindows` tree with the `beamline_pmt_tdc_times`, `beamline_pmt_tdc_ids`, `beamline_pmt_qdc_charges`, `beamline_pmt_qdc_ids`, and `spill_counter` branches — i.e. VME-merged offline data, e.g. from `/eos/experiment/wcte/data/2025_commissioning/offline_data_vme_match/`.

The run must satisfy preconditions read from the run-info JSON, or the script raises and exits:
- ACT0 must be in the beamline; lead glass must be out of the beamline.
- Unless `--no_acts` is passed: ACT0/1/2 must all share the same refractive index, ACT3 and ACT4 must be in the beamline, and ACT3/4(/5) must share the same refractive index.

## Usage

```bash
python3 WCTE_beam_analysis.py -r <run_number> -i <input_file(s)> -o <output_dir> [options]
```

| Flag | Required | Description |
|---|---|---|
| `-r`, `--run_number` | yes | Run number; must appear as `R<run_number>` in every input filename |
| `-i`, `--input_files` | yes | One or more `WCTEReadoutWindows` ROOT files (space-separated) |
| `-o`, `--output_dir` | yes | Directory for the output ROOT file and PDF (created if missing) |
| `--debug` | no | Only process the first 5000 events |
| `--no_acts` | no | Minimum-bias mode: relax the requirement that upstream/downstream ACTs share a refractive index (for runs without a full ACT lineup) |
| `--is_kaon_run` | no | Relabels identified muons as kaons, uses kaon dE/dx tables for their momentum estimate, and records the flag in `run_info` |
| `--use_emupi_TOF_separation` | no | Use time-of-flight instead of ACT3-5 charge to separate muons/pions (needed below the ~300 MeV/c muon Cherenkov threshold). Automatically forced on for runs before 1441 with `\|momentum\| < 300` MeV/c |

Example:
```bash
python3 WCTE_beam_analysis.py -r 2285 \
  -i /eos/experiment/wcte/data/2025_commissioning/offline_data_vme_match/WCTE_offline_R2285S0_VME_matched.root \
  -o ./beam_data_out
```

Multiple input files for the same run can be passed at once; each is processed independently and produces its own pair of output files.

## What it does

For each input file, in order:
1. **1 p.e. calibration** (`adjust_1pe_calibration`) — refines the ACT PMT gain calibration.
2. **Proton/deuteron/helium-3/triton tagging** (`tag_protons_TOF`) — via the T0–T1 time of flight, using theoretical TOF cuts computed from the nominal beam momentum. Done first to avoid double-counting these particles as something else. Protons are only tagged above 250 MeV/c; deuterons/helium-3/tritons only above 350 MeV/c.
3. **Electron/positron tagging** (`tag_electrons_ACT02`) — from a charge threshold on the ACT0–2 ("e-veto") sum.
4. **Sanity-check plots** (`plot_ACT35_left_vs_right`, `plot_ACT02_left_vs_right`) — visualise that the proton/electron removal looks correct.
5. **Muon/pion tagging** (`tag_muons_pions_ACT35`) — from the ACT3–5 ("tagger") charge, or from TOF instead if `--use_emupi_TOF_separation` applies. Falls back to an auxiliary muon-tagger cut if the ACT3-5 muon/pion populations don't show a clean minimum.
6. **TOF offset correction** (`measure_particle_TOF`) — corrects for cable-length-type offsets so later momentum estimates are meaningful.
7. **Momentum estimation** (`estimate_particle_momentum`) — mean momentum per particle species and per-trigger (not output to production 1), both just after the CERN beam pipe and at the WCTE tank window. The per-trigger error is taken from the electron TOF resolution (std of a Gaussian fit), so it is large for slow/heavy particles.
8. **POT-normalised yield plot** (`plot_number_particles_per_POT`).
9. **Event-quality plot** (`plot_event_quality_bitmask`).
10. **ROOT output** (`output_to_root`).

If no triggers pass the basic event-quality requirements, the analysis is skipped for that file (the PDF is still closed cleanly and a message is printed).

## Output

For each input file `<base>.root`, two files are written to `-o`:

- **`<base>_PID.pdf`** — every diagnostic plot produced above (ACT/TOF distributions, cut lines, per-particle momentum plots, event-quality histograms, etc.). Always check these to confirm the selection behaved sensibly for that run.
- **`<base>_beam_analysis.root`** — three trees:
  - **`beam_analysis`** — one entry per trigger: `event_id`, `spill_number`, per-channel TDC-corrected times and QDC charges/p.e. for T0/T1/T4/ACT0-5/muon tagger, `tof_t0t1` and other TOF combinations, `evt_quality_bitmask`, `digi_issues_bitmask`.
    - `evt_quality_bitmask` bits: `0` T0/T1 TDC missing, `1` T4 TDC missing, `2` T5 TDC missing, `3` hole-counter hit, `4` T4 QDC missing, `5` more than one T5 bar hit.
    - `digi_issues_bitmask` bits: `0` QDC failure, `1` missing digitiser times.
  - **`run_info`** — one entry: run number, nominal beam momentum, ACT refractive indices, whether ACT5 is present, jaw positions, `is_kaon_run` flag.
  - **`scalar_results`** — one entry: the cut values used (TOF/ACT cut lines), number of triggers by particle species, per-species TOF and momentum means/errors (near the CERN pipe and at the WCTE window), and nominal (theoretical) TOF per species.

  Per-trigger PID/momentum branches (`is_muon`, `is_electron`, ..., `final_momentum`, ...) are only written when `is_beam_paper_analysis=True`, which this CLI never sets — so only the aggregate PID counts and per-species momentum summaries in `scalar_results` are available from a standalone run. Note also that `is_kept` (whether a trigger passed the basic beam requirements) is used internally to count total triggers but is dropped before writing, so it is **not** present in the output `beam_analysis` tree.

## Caveats

- Helium-3 nuclei are tagged by TOF the same way as protons/deuterons/tritons, but the magnetic-rigidity/charge correction used for their theoretical TOF cut has not been as thoroughly validated — treat with care.
- No cut is applied on the T5 (TOF detector) total charge; it is saved in the output for reference only.
- Per-trigger momentum estimates carry large errors for slow/heavy particles, since the TOF resolution is derived from the (fast) electron peak.
- See the beam PID write-up and collaboration meeting slides below for the full physics rationale.
- The event bitmask is != 0 if:
    - The TOF is too short (< 10ns)
    - The particle does not reach T5 (though this is also checked more directly by Frantisek's BRB analysis)
    - The TDC or QDC failed

Email Alie if you have any questions.

