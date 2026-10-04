#!/usr/bin/env python3
"""Prepare one EOS-backed HTCondor conversion job per WCTE production run."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

DEFAULT_RUNS = (
    1786, 1788, 1790, 1792, 1794, 1796, 1798, 1800, 1802,
    1803, 1804, 1806, 1808, 1810, 1812, 1814, 1820, 1825,
    1816, 1819, 1817, 1818, 1827, 1829, 1831, 1833, 1834,
    1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842,
)
DEFAULT_BASE = Path(
    "/eos/experiment/wcte/data/2025_commissioning/"
    "processed_offline_data/production_v1_0"
)


def _eos_path(path: Path) -> str:
    # Do not resolve EOS symlinks: /eos/user/... can resolve to a host-specific
    # /eos/home-* path that another lxplus/worker host cannot open.
    absolute = os.path.abspath(path)
    if not absolute.startswith("/eos/"):
        raise ValueError(f"EosSubmit requires an /eos path: {absolute}")
    if any(char.isspace() for char in absolute):
        raise ValueError(f"HTCondor job-list paths cannot contain spaces: {absolute}")
    return absolute


def prepare(
    base_dir: Path,
    repo_dir: Path,
    submission_dir: Path,
    runs: tuple[int, ...],
    memory_gb: int,
    max_runtime_hours: int,
    events_per_file: int,
    output_subdir: str = "converted_npz",
):
    if memory_gb < 1 or max_runtime_hours < 1 or events_per_file < 1:
        raise ValueError("Memory, runtime, and events-per-file must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", output_subdir):
        raise ValueError("--output-subdir must be one shell-safe directory name")
    repo = _eos_path(repo_dir)
    submission = Path(_eos_path(submission_dir))
    worker = Path(repo) / "scripts" / "run_wcte_npz_one.sh"
    if not worker.is_file():
        raise FileNotFoundError(worker)
    if not worker.stat().st_mode & 0o111:
        raise PermissionError(f"Worker must be executable: chmod u+x {worker}")
    base = Path(_eos_path(base_dir))

    jobs = []
    missing = []
    completed = []
    blocked = []
    unwritable = []
    for run in dict.fromkeys(runs):
        source = base / str(run) / f"WCTE_merged_production_R{run}.root"
        destination = base / str(run) / output_subdir
        if not source.is_file():
            missing.append(str(source))
            continue
        if (destination / "conversion.done").is_file():
            completed.append(run)
            continue
        if destination.is_dir() and (
            list(destination.glob("*.npz")) or
            (destination / ".conversion_in_progress").exists()
        ):
            blocked.append(str(destination))
            continue
        writable_parent = destination if destination.is_dir() else source.parent
        if not os.access(writable_parent, os.W_OK):
            unwritable.append(str(destination))
            continue
        jobs.append((run, str(source), str(destination)))

    submission.mkdir(parents=True, exist_ok=True)
    logs = submission / "logs"
    logs.mkdir(exist_ok=True)
    jobs_path = submission / "jobs.txt"
    submit_path = submission / "convert_wcte_npz.sub"
    jobs_path.write_text(
        "".join(f"{run} {source} {dest}\n" for run, source, dest in jobs),
        encoding="utf-8",
    )
    submit_path.write_text(
        "\n".join((
            "universe = vanilla",
            f"initialdir = {repo}",
            f"executable = {worker}",
            f"arguments = $(run) $(input_path) $(output_dir) {repo} {events_per_file}",
            "should_transfer_files = NO",
            f"request_memory = {memory_gb} GB",
            "request_cpus = 1",
            f"+MaxRuntime = {max_runtime_hours * 3600}",
            f"output = {logs}/$(ClusterId).$(ProcId).out",
            f"error = {logs}/$(ClusterId).$(ProcId).err",
            f"log = {logs}/$(ClusterId).log",
            f"queue run,input_path,output_dir from {jobs_path}",
            "",
        )),
        encoding="utf-8",
    )
    print(f"Prepared {len(jobs)} jobs: {submit_path}")
    print(f"Already complete: {len(completed)}; missing inputs: {len(missing)}; "
          f"existing incomplete outputs: {len(blocked)}; unwritable: {len(unwritable)}")
    for label, values in (
        ("Missing input", missing),
        ("Needs inspection", blocked),
        ("Unwritable output", unwritable),
    ):
        for value in values:
            print(f"{label}: {value}")
    if jobs:
        print(f"Submit from EosSubmit lxplus: condor_submit {submit_path}")
    return submit_path, jobs_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--repo-dir", type=Path,
                        default=Path(__file__).absolute().parents[1])
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--runs", nargs="+", type=int, default=DEFAULT_RUNS,
                        help="Run numbers; default is the 35-run commissioning list")
    parser.add_argument("--memory-gb", type=int, default=32)
    parser.add_argument("--max-runtime-hours", type=int, default=72)
    parser.add_argument("--events-per-file", type=int, default=5000)
    parser.add_argument("--output-subdir", default="converted_npz",
                        help="Per-run NPZ directory; use converted_npz_v2 to preserve old parts")
    args = parser.parse_args()
    repo = args.repo_dir.absolute()
    submission = args.submission_dir or repo / "outputs" / "wcte_npz_batch"
    prepare(
        args.base_dir, repo, submission, tuple(args.runs),
        args.memory_gb, args.max_runtime_hours, args.events_per_file,
        args.output_subdir,
    )


if __name__ == "__main__":
    main()
