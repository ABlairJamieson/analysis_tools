# Historical timing calibrations

This directory contains calibration files retained for comparison studies.

`mohit_led_timing_offsets_nonofficial_20250401.json` is the user-provided
Mohit LED timing-offset export. Its metadata identifies it as
`calibration_name: timing_offsets`, `calibration_method: LED`, and
`official: false`, with timestamp/run identifier `20250401183945`.

It is **not** the active production calibration and must not be applied to
the already calibrated `hit_pmt_calibrated_times` branch or to converted NPZ
files containing those times. Use it only in a diagnostic path that reads
the original raw PMT times from a ROOT file, applies
`corrected_time = raw_time - timing_offset`, and records this file in the
output manifest. The purpose is to compare whether timing structures such
as the beam comb persist under an independent historical calibration.
