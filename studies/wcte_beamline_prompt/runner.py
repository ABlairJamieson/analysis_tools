#!/usr/bin/env python3
"""Stream ROOT readouts through DataLoader and a small student-written study.

The runner owns file reading and the common channel-31/channel-46 reference
correction. A study plugin owns its observables, plots, and any extra cuts.
Selections are named masks; adding a child selection ANDs it with its parent,
so existing/base selections cannot be silently replaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import awkward as ak
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis_tools import DataLoader  # noqa: E402
from analysis_tools.beamline_timing import REFERENCE_IDS, TDCHit, correct_tdc_hits  # noqa: E402
from studies.framework.core import SelectionBook, StudyContext, load_study  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Merged WCTE production ROOT file")
    parser.add_argument("--study", type=Path, required=True,
                        help="Python file defining a Study class (see example_study.py)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree", default="WCTEReadoutWindows")
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--step-size", default="100 MB",
                        help="Uproot batch size, e.g. '100 MB' or 5000")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every N input batches")
    return parser


def _parse_args() -> tuple[argparse.Namespace, type]:
    """Load the selected plugin before parsing its optional arguments.

    Plugins may provide ``add_arguments(parser)``. This keeps the shared
    runner small while allowing a study to expose its own analysis settings.
    """
    parser = _build_parser()
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--study", type=Path, required=True)
    preliminary, _unknown = selector.parse_known_args()
    study_class = load_study(preliminary.study)
    add_arguments = getattr(study_class, "add_arguments", None)
    if add_arguments is not None:
        add_arguments(parser)
    return parser.parse_args(), study_class


def run(args: argparse.Namespace, study_class: type | None = None) -> None:
    if args.entry_start < 0 or (args.max_events is not None and args.max_events < 1):
        raise ValueError("entry-start must be nonnegative and max-events positive")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    if not args.input_root.is_file():
        raise FileNotFoundError(args.input_root)
    study_class = study_class or load_study(args.study)
    study = study_class(args)
    extra_branches = tuple(getattr(study, "required_branches", ()))
    optional_branches = tuple(getattr(study, "optional_branches", ()))
    branches = ["beamline_pmt_tdc_ids", "beamline_pmt_tdc_times"]
    branches.extend(name for name in extra_branches if name not in branches)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader streams batches. Quality cuts remain off unless the study
    # explicitly enables them; the base samples are visible as named masks.
    processed = 0
    passed_quality = 0
    batch_number = 0
    with DataLoader(args.input_root, branches_to_load=branches) as loader:
        tree = loader.file[args.tree]
        available = loader.available_branches
        for name in branches:
            if name not in available:
                raise ValueError(f"Study requires missing ROOT branch: {name}")
        branches.extend(name for name in optional_branches
                        if name in available and name not in branches)
        if loader.branches_to_load is not None:
            for name in branches:
                if name not in loader.branches_to_load:
                    loader.branches_to_load.append(name)
        stop = tree.num_entries
        if args.max_events is not None:
            stop = min(stop, args.entry_start + args.max_events)
        if args.entry_start > stop:
            raise ValueError("entry-start is beyond the end of the ROOT tree")

        for batch in loader.iterate(
            step_size=args.step_size,
            entry_start=args.entry_start,
            entry_stop=stop,
        ):
            ids_by_event = ak.to_list(batch["beamline_pmt_tdc_ids"])
            times_by_event = ak.to_list(batch["beamline_pmt_tdc_times"])
            corrected_by_event: list[list[TDCHit]] = []
            references: list[tuple[float | None, float | None]] = []
            quality = []
            for ids, times in zip(ids_by_event, times_by_event):
                valid_pair = ids is not None and times is not None and len(ids) == len(times)
                pairs = list(zip(ids, times)) if valid_pair else []
                corrected, refs = correct_tdc_hits(pairs)
                corrected_by_event.append(corrected)
                references.append(refs)
                good = (valid_pair and refs[0] is not None and refs[1] is not None)
                quality.append(bool(good))
            if "vme_digi_issues_bitmask" in batch.fields:
                digi_issues = ak.to_list(batch["vme_digi_issues_bitmask"])
                quality = [ok and value is not None and int(value) == 0
                           for ok, value in zip(quality, digi_issues)]

            selections = SelectionBook(len(batch))
            selections.add("beamline_quality", quality)
            context = StudyContext(
                input_path=args.input_root,
                output_dir=args.output_dir,
                entry_start=args.entry_start + processed,
                selections=selections,
                corrected_tdc_hits=corrected_by_event,
                reference_times_ns=references,
            )
            study.process_batch(batch, context)
            passed_quality += selections.mask("beamline_quality").sum().item()
            processed += len(batch)
            batch_number += 1
            if batch_number % args.progress_every == 0:
                print(f"Processed {processed:,}/{stop - args.entry_start:,} readouts; "
                      f"{passed_quality:,} pass beamline quality", flush=True)

    study.finish()
    print(f"Finished {processed:,} readouts ({passed_quality:,} beamline-quality); "
          f"results: {args.output_dir}", flush=True)


def main() -> None:
    args, study_class = _parse_args()
    run(args, study_class)


if __name__ == "__main__":
    main()
