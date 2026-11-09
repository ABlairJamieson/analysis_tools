#!/usr/bin/env python3
"""Run a study plugin directly over WCSim ROOT using WatChMaL DataTools.

The generic runner owns input iteration, batching, progress and the shared
plugin lifecycle. Source-specific ROOT decoding lives in
``datatools_wcsim_root.py``; analysis-specific calculations belong in the
plugin selected with ``--study``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from studies.framework.core import SelectionBook, StudyContext, load_study  # noqa: E402
from studies.framework.datatools_wcsim_root import DataToolsWCSimRootSource  # noqa: E402


def build_parser(study_class: type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="WCSim Monte Carlo ROOT files")
    parser.add_argument("--study", type=Path, required=True,
                        help="Python file defining a Study plugin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datatools-path", type=Path, default=None,
                        help="DataTools checkout; defaults to $DATATOOLS")
    parser.add_argument("--entry-start", type=int, default=0,
                        help="First ROOT entry to read in each input file")
    parser.add_argument("--max-events-per-file", type=int, default=100,
                        help="Pilot default; use --all-events for the full ROOT file")
    parser.add_argument("--all-events", action="store_true",
                        help="Read every ROOT entry, overriding the pilot cap")
    parser.add_argument("--step-events", type=int, default=100,
                        help="Number of events handed to the plugin per batch")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print progress after this many input events")
    add_arguments = getattr(study_class, "add_arguments", None)
    if add_arguments is not None:
        add_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--study", type=Path, required=True)
    selected, _unknown = selector.parse_known_args(argv)
    study_class = load_study(selected.study)
    args = build_parser(study_class).parse_args(argv)
    return args, study_class


def run(args: argparse.Namespace, study_class: type | None = None) -> None:
    if args.entry_start < 0:
        raise ValueError("entry-start must be nonnegative")
    if args.step_events < 1 or args.progress_every < 1:
        raise ValueError("step-events and progress-every must be positive")
    if args.all_events:
        if args.max_events_per_file != 100:
            raise ValueError("Use --all-events without --max-events-per-file")
    elif args.max_events_per_file < 1:
        raise ValueError("max-events-per-file must be positive")
    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    study_class = study_class or load_study(args.study)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study = study_class(args)
    total_processed = 0
    try:
        for path in args.inputs:
            file_processed = 0
            with DataToolsWCSimRootSource(path, args.datatools_path) as source:
                stop = source.entries
                if not args.all_events:
                    stop = min(stop, args.entry_start + args.max_events_per_file)
                if args.entry_start > stop:
                    raise ValueError(f"Entry start {args.entry_start} is past {path} ({stop} entries)")
                target = stop - args.entry_start
                print(f"Reading {path}: entries {args.entry_start:,}–{stop:,} "
                      f"of {source.entries:,} via DataTools", flush=True)
                for batch_start in range(args.entry_start, stop, args.step_events):
                    batch_stop = min(batch_start + args.step_events, stop)
                    batch = [source.read_event(entry) for entry in range(batch_start, batch_stop)]
                    context = StudyContext(
                        input_path=path,
                        output_dir=args.output_dir,
                        entry_start=batch_start,
                        selections=SelectionBook(len(batch)),
                    )
                    study.process_batch(batch, context)
                    file_processed += len(batch)
                    total_processed += len(batch)
                    if (file_processed % args.progress_every < len(batch)
                            or file_processed == target):
                        print(f"{path.name}: {file_processed:,}/{target:,} events processed",
                              flush=True)
            print(f"{path.name}: completed {file_processed:,} ROOT events", flush=True)
        study.finish()
    except Exception:
        abort = getattr(study, "abort", None)
        if abort is not None:
            abort()
        raise
    print(f"Finished {total_processed:,} events; results: {args.output_dir}", flush=True)


def main() -> None:
    args, study_class = parse_args()
    run(args, study_class)


if __name__ == "__main__":
    main()
