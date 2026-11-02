#!/usr/bin/env python3
"""Prepare one EOS-backed HTCondor job per WCTE beamline prompt-study run."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.prepare_wcte_npz_batch import DEFAULT_BASE, DEFAULT_RUNS  # noqa: E402


def eos_path(path: Path) -> str:
    """Return a stable absolute EOS path without resolving host-specific symlinks."""
    absolute = os.path.abspath(path)
    if not absolute.startswith("/eos/"):
        raise ValueError(f"EosSubmit requires an /eos path: {absolute}")
    if any(char.isspace() for char in absolute):
        raise ValueError(f"HTCondor job-list paths cannot contain spaces: {absolute}")
    return absolute


def prepare(base_dir: Path, repo_dir: Path, output_root: Path,
            submission_dir: Path, runs: tuple[int, ...],
            memory_gb: int, max_runtime_hours: int, max_events: int):
    if memory_gb < 1 or max_runtime_hours < 1 or max_events < 1:
        raise ValueError("Memory, runtime, and max-events must be positive")
    repo = Path(eos_path(repo_dir))
    base = Path(eos_path(base_dir))
    output_root = Path(eos_path(output_root))
    submission = Path(eos_path(submission_dir))
    worker = repo / "scripts" / "run_wcte_beamline_prompt_one.sh"
    study = repo / "scripts" / "study_wcte_beamline_prompt.py"
    if not worker.is_file():
        raise FileNotFoundError(worker)
    if not worker.stat().st_mode & 0o111:
        raise PermissionError(f"Worker must be executable on EOS; run: chmod u+x {worker}")
    if not study.is_file():
        raise FileNotFoundError(study)

    jobs = []
    completed = []
    blocked = []
    missing = []
    for run in dict.fromkeys(runs):
        source = base / str(run) / f"WCTE_merged_production_R{run}.root"
        destination = output_root / f"R{run}"
        if not source.is_file():
            missing.append(str(source))
            continue
        if ((destination / "summary.json").is_file()
                or (destination / ".beamline_prompt.done").is_file()):
            completed.append(run)
            continue
        if destination.is_dir() and any(destination.iterdir()):
            blocked.append(str(destination))
            continue
        jobs.append((run, str(source), str(destination)))

    output_root.mkdir(parents=True, exist_ok=True)
    submission.mkdir(parents=True, exist_ok=True)
    logs = submission / "logs"
    logs.mkdir(exist_ok=True)
    jobs_path = submission / "jobs.txt"
    submit_path = submission / "study_wcte_beamline_prompt.sub"
    jobs_path.write_text(
        "".join(f"{run} {source} {dest}\n" for run, source, dest in jobs),
        encoding="utf-8",
    )
    submit_path.write_text(
        "\n".join((
            "universe = vanilla",
            f"initialdir = {repo}",
            f"executable = {worker}",
            f"arguments = $(run) $(input_root) $(output_dir) {repo} {max_events}",
            "should_transfer_files = NO",
            f"request_memory = {memory_gb} GB",
            "request_cpus = 1",
            f"+MaxRuntime = {max_runtime_hours * 3600}",
            f"output = {logs}/$(ClusterId).$(ProcId).out",
            f"error = {logs}/$(ClusterId).$(ProcId).err",
            f"log = {logs}/$(ClusterId).log",
            f"queue run,input_root,output_dir from {jobs_path}",
            "",
        )),
        encoding="utf-8",
    )

    print(f"Prepared {len(jobs)} beamline prompt-study jobs: {submit_path}")
    print(f"Already complete: {len(completed)}; missing inputs: {len(missing)}; "
          f"existing incomplete outputs: {len(blocked)}")
    for label, values in (("Missing input", missing), ("Needs inspection", blocked)):
        for value in values:
            print(f"{label}: {value}")
    if jobs:
        print(f"Review {jobs_path}, then submit with: condor_submit {submit_path}")
    return submit_path, jobs_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="EOS directory for per-run output folders")
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--runs", nargs="+", type=int, default=DEFAULT_RUNS,
                        help="Run numbers; defaults to the commissioning list")
    parser.add_argument("--memory-gb", type=int, default=32)
    parser.add_argument("--max-runtime-hours", type=int, default=72)
    parser.add_argument("--max-events", type=int, default=1_000_000_000,
                        help="Upper bound; the study stops at the ROOT tree length")
    args = parser.parse_args()
    repo = args.repo_dir.absolute()
    output_root = args.output_root or repo / "outputs" / "wcte_beamline_prompt_by_hd_tdc"
    submission = args.submission_dir or repo / "outputs" / "wcte_beamline_prompt_batch"
    prepare(args.base_dir, repo, output_root, submission, tuple(args.runs),
            args.memory_gb, args.max_runtime_hours, args.max_events)


if __name__ == "__main__":
    main()
