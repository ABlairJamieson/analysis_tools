# Shared study framework

The framework owns reusable mechanics: loading a `Study` plugin, passing
selection cut flow, iterating source batches, reporting progress, and closing
the source. Keep detector-specific calculations in a study folder.

## WCSim Monte Carlo ROOT

`runner.py` reads each ROOT event through WatChMaL DataTools'
`root_utils.root_file_utils.WCSimFile`. It needs the DataTools checkout in
`$DATATOOLS` (or `--datatools-path`) and the normal PyROOT/WCSim environment.
It does not invoke `event_dump.py`, create NPZ files, or use
`analysis_tools.DataLoader`; that loader reads processed WCTE data with a
different tree and branch schema.

The source adapter exposes a list of dictionaries per batch. Each event
contains digitized hit times, charges, PMT IDs, trigger indices/times, and
DataTools' primary PID and energy. DataTools' values are passed through as-is.
The delayed-positron plugin optionally adds a hit's matching trigger time once
when `--time-mode auto` is selected; it applies no extra channel calibration.

## Writing a plugin

Create a Python file that defines `Study` with `__init__(args)`,
`process_batch(batch, context)`, and `finish()`. Optionally define
`add_arguments(parser)`. Study-specific selections should be added through
`context.selections.add(name, mask, parent="...")`; masks are ANDed with their
parent, and names cannot be replaced. Source readers should adapt detector
data to documented event objects rather than being copied into each study.

For a WCSim ROOT plugin, run `python3 studies/framework/runner.py --study
PATH_TO_PLUGIN --help` to see the shared and plugin-specific options.
