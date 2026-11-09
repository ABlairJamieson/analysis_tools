"""Find prompt and delayed PMT clusters in WCSim Monte Carlo ROOT events.

The clustering/ranking behavior is migrated from LicketyFit's
``study_delayed_clusters.py`` and ``TimeClustering.py``. A delayed cluster is
an exploratory Michel-like candidate, not a truth tag or a measured decay.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TimeCluster:
    """One sliding-window burst, described by its selected digit hits."""

    start_ns: float
    end_ns: float
    center_ns: float
    n_hits: int
    n_pmts: int
    charge: float

    def to_dict(self):
        return asdict(self)


def event_hit_times(event: dict, mode: str = "auto") -> tuple[np.ndarray, str]:
    """Convert digit times to a common clock using trigger metadata if present."""
    if mode not in {"auto", "raw", "trigger-plus"}:
        raise ValueError("time mode must be auto, raw, or trigger-plus")
    times = np.asarray(event["digi_hit_time"], dtype=float).reshape(-1)
    if mode == "raw":
        return times, "raw"
    if "digi_hit_trigger" not in event or "trigger_time" not in event:
        if mode == "trigger-plus":
            raise ValueError("trigger-plus requires digi_hit_trigger and trigger_time")
        return times, "raw_missing_trigger_metadata"
    indices = np.asarray(event["digi_hit_trigger"], dtype=int).reshape(-1)
    offsets = np.asarray(event["trigger_time"], dtype=float).reshape(-1)
    if indices.size != times.size:
        raise ValueError("digi_hit_trigger and digi_hit_time have different lengths")
    if times.size and (offsets.size == 0 or np.any(indices < 0) or np.any(indices >= offsets.size)):
        raise ValueError("digit trigger index is outside trigger_time")
    return times + offsets[indices], "trigger_plus"


def _best_window(times, pmts, charges, width_ns):
    """Return the window with most distinct PMTs, then highest total charge."""
    if not times.size:
        return None
    order = np.argsort(times, kind="stable")
    t, p, q = times[order], pmts[order], charges[order]
    left = 0
    counts = {}
    best_left = best_right = 0
    best_pmts = -1
    best_charge = -1.0
    running_charge = 0.0
    for right in range(t.size):
        key = int(p[right])
        counts[key] = counts.get(key, 0) + 1
        running_charge += q[right]
        while t[right] - t[left] > width_ns:
            key = int(p[left])
            counts[key] -= 1
            if counts[key] == 0:
                del counts[key]
            running_charge -= q[left]
            left += 1
        if (len(counts), running_charge) > (best_pmts, best_charge):
            best_left, best_right = left, right + 1
            best_pmts, best_charge = len(counts), running_charge
    selected_t = t[best_left:best_right]
    return TimeCluster(float(selected_t[0]), float(selected_t[-1]),
                       float(np.median(selected_t)), int(selected_t.size),
                       int(best_pmts), float(np.sum(q[best_left:best_right])))


def _first_coincidence_end(times, pmts, width_ns, min_pmts):
    """Find the first sliding window containing the requested PMT count."""
    order = np.argsort(times, kind="stable")
    t, p = times[order], pmts[order]
    left = 0
    counts = {}
    for right in range(t.size):
        key = int(p[right])
        counts[key] = counts.get(key, 0) + 1
        while t[right] - t[left] > width_ns:
            key = int(p[left])
            counts[key] -= 1
            if counts[key] == 0:
                del counts[key]
            left += 1
        if len(counts) >= min_pmts:
            return float(t[right])
    return None


def find_delayed_clusters(times_ns, pmts, charges, *, width_ns=50.0,
                          min_pmts=10, prompt_min_pmts=10,
                          search_start_ns=200.0, search_end_ns=10_000.0,
                          prompt_search_ns=200.0, max_clusters=5):
    """Return the anchored prompt and ranked delayed burst candidates.

    Candidates are ranked by distinct PMT count, with total charge as the
    tie-break. Nearby overlapping windows are suppressed after each selection.
    """
    if width_ns <= 0 or min_pmts < 1 or prompt_min_pmts < 1 or max_clusters < 1:
        raise ValueError("window width, PMT thresholds, and max clusters must be positive")
    if search_start_ns < 0 or search_end_ns <= search_start_ns or prompt_search_ns <= 0:
        raise ValueError("invalid delayed search interval")
    times = np.asarray(times_ns, dtype=float).reshape(-1)
    ids = np.asarray(pmts, dtype=int).reshape(-1)
    charge = np.asarray(charges, dtype=float).reshape(-1)
    if not (times.size == ids.size == charge.size):
        raise ValueError("digit time, PMT ID, and charge arrays have different lengths")
    good = np.isfinite(times) & np.isfinite(charge) & (charge > 0)
    times, ids, charge = times[good], ids[good], charge[good]
    anchor = _first_coincidence_end(times, ids, width_ns, prompt_min_pmts)
    if anchor is None:
        return None, []
    early = (times >= anchor - width_ns) & (times <= anchor + prompt_search_ns)
    prompt = _best_window(times[early], ids[early], charge[early], width_ns)
    if prompt is None:
        return None, []
    delta = times - prompt.center_ns
    remaining = (delta >= search_start_ns) & (delta <= search_end_ns)
    candidates = []
    for _ in range(max_clusters):
        candidate = _best_window(times[remaining], ids[remaining], charge[remaining], width_ns)
        if candidate is None or candidate.n_pmts < min_pmts:
            break
        candidates.append(candidate)
        remaining &= ((times < candidate.start_ns - width_ns)
                      | (times > candidate.end_ns + width_ns))
    return prompt, candidates


class Study:
    """Study plugin producing the event and all-candidate tables."""

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("Delayed-cluster selection")
        group.add_argument("--width-ns", type=float, default=50.0)
        group.add_argument("--min-pmts", type=int, default=10)
        group.add_argument("--prompt-min-pmts", type=int, default=10)
        group.add_argument("--start-ns", type=float, default=200.0)
        group.add_argument("--end-ns", type=float, default=10_000.0)
        group.add_argument("--prompt-search-ns", type=float, default=200.0)
        group.add_argument("--min-time-ns", type=float, default=-1_000_000.0,
                           help="Reject digits earlier than this common-clock time")
        group.add_argument("--time-mode", choices=("auto", "raw", "trigger-plus"),
                           default="auto")

    def __init__(self, args):
        self.args = args
        self.output = args.output_dir
        self.events_path = self.output / "events.csv"
        self.clusters_path = self.output / "clusters.csv"
        self.events_file = self.events_path.open("w", newline="", encoding="utf-8")
        self.clusters_file = self.clusters_path.open("w", newline="", encoding="utf-8")
        self.event_fields = [
            "input_file", "event_index", "event_id", "primary_pid", "primary_energy",
            "time_mode", "n_digits", "n_rejected_early_digits", "n_triggers", "time_span_ns",
            "prompt_time_ns", "prompt_start_ns", "prompt_end_ns", "prompt_pmts", "prompt_n_hits",
            "prompt_charge", "n_delayed_clusters", "best_delayed_time_ns", "delta_t_ns",
            "best_delayed_pmts", "best_delayed_charge",
        ]
        self.cluster_fields = ["input_file", "event_index", "rank", "delta_t_ns", "start_ns",
                               "end_ns", "center_ns", "n_hits", "n_pmts", "charge"]
        self.events = csv.DictWriter(self.events_file, fieldnames=self.event_fields)
        self.clusters = csv.DictWriter(self.clusters_file, fieldnames=self.cluster_fields)
        self.events.writeheader()
        self.clusters.writeheader()
        self.total_events = 0
        self.tagged_events = 0
        self.rejected_digits = 0
        self.short_spans = 0
        self.time_modes = Counter()
        self.selection_counts = Counter()

    def process_batch(self, batch, context):
        """Process DataTools-adapted ROOT events without retaining hit arrays."""
        prompt_found = []
        candidate_found = []
        for offset, event in enumerate(batch):
            times, convention = event_hit_times(event, self.args.time_mode)
            self.time_modes[convention] += 1
            pmts = np.asarray(event["digi_hit_pmt"], dtype=int).reshape(-1)
            charges = np.asarray(event["digi_hit_charge"], dtype=float).reshape(-1)
            n_raw_digits = len(times)
            if len(pmts) != len(times) or len(charges) != len(times):
                raise ValueError(f"{context.input_path} entry {context.entry_start + offset}: "
                                 "digit time/PMT/charge array lengths differ")
            keep = np.isfinite(times) & (times >= self.args.min_time_ns)
            n_rejected = int(np.count_nonzero(~keep))
            self.rejected_digits += n_rejected
            times, pmts, charges = times[keep], pmts[keep], charges[keep]
            has_short_span = bool(len(times) and np.ptp(times) < self.args.end_ns)
            self.short_spans += has_short_span
            prompt, delayed = find_delayed_clusters(
                times, pmts, charges, width_ns=self.args.width_ns,
                min_pmts=self.args.min_pmts, prompt_min_pmts=self.args.prompt_min_pmts,
                search_start_ns=self.args.start_ns, search_end_ns=self.args.end_ns,
                prompt_search_ns=self.args.prompt_search_ns)
            best = delayed[0] if delayed else None
            event_id = event.get("event_id", context.entry_start + offset)
            pid = event.get("pid", "")
            row = {
                "input_file": str(context.input_path), "event_index": context.entry_start + offset,
                "event_id": _csv_scalar(event_id), "primary_pid": _csv_scalar(pid),
                "primary_energy": _csv_scalar(event.get("energy", "")),
                "time_mode": convention,
                "n_digits": n_raw_digits, "n_rejected_early_digits": n_rejected,
                "n_triggers": len(np.asarray(event.get("trigger_time", [])).reshape(-1)),
                "time_span_ns": float(np.ptp(times)) if len(times) else "",
                "prompt_time_ns": prompt.center_ns if prompt else "",
                "prompt_start_ns": prompt.start_ns if prompt else "",
                "prompt_end_ns": prompt.end_ns if prompt else "",
                "prompt_pmts": prompt.n_pmts if prompt else "",
                "prompt_n_hits": prompt.n_hits if prompt else "",
                "prompt_charge": prompt.charge if prompt else "",
                "n_delayed_clusters": len(delayed),
                "best_delayed_time_ns": best.center_ns if best else "",
                "delta_t_ns": best.center_ns - prompt.center_ns if best else "",
                "best_delayed_pmts": best.n_pmts if best else "",
                "best_delayed_charge": best.charge if best else "",
            }
            self.events.writerow(row)
            for rank, cluster in enumerate(delayed, 1):
                self.clusters.writerow({
                    "input_file": str(context.input_path),
                    "event_index": context.entry_start + offset,
                    "rank": rank,
                    "delta_t_ns": cluster.center_ns - prompt.center_ns,
                    **cluster.to_dict(),
                })
            prompt_found.append(prompt is not None)
            candidate_found.append(bool(delayed))
            self.total_events += 1
            self.tagged_events += bool(delayed)
        prompt_mask = context.selections.add("prompt_cluster_found", prompt_found)
        candidate_mask = context.selections.add("delayed_candidate_found", candidate_found,
                                               parent="prompt_cluster_found")
        self.selection_counts["all"] += len(prompt_found)
        self.selection_counts["prompt_cluster_found"] += int(prompt_mask.sum())
        self.selection_counts["delayed_candidate_found"] += int(candidate_mask.sum())

    def finish(self):
        """Close tables and save cut/configuration accounting."""
        self.events_file.close()
        self.clusters_file.close()
        summary = {
            "processed_events": self.total_events,
            "events_with_delayed_candidate": self.tagged_events,
            "candidate_event_fraction": (self.tagged_events / self.total_events
                                         if self.total_events else None),
            "selection_counts": dict(self.selection_counts),
            "time_conventions": dict(self.time_modes),
            "rejected_early_or_nonfinite_digits": self.rejected_digits,
            "observed_digit_span_shorter_than_search_end_events": self.short_spans,
            "settings": {
                "width_ns": self.args.width_ns, "min_pmts": self.args.min_pmts,
                "prompt_min_pmts": self.args.prompt_min_pmts,
                "search_start_ns": self.args.start_ns, "search_end_ns": self.args.end_ns,
                "prompt_search_ns": self.args.prompt_search_ns,
                "min_time_ns": self.args.min_time_ns, "time_mode": self.args.time_mode,
            },
            "interpretation": (
                "Rank-1 is the delayed sliding window with the most distinct PMTs, "
                "breaking ties by total charge; it is not simply the second-in-time "
                "or largest-hit cluster. Candidates are not truth tags. Digit-span "
                "shorter than the search window is a warning, not a readout-coverage measure."
            ),
            "outputs": [self.events_path.name, self.clusters_path.name],
        }
        (self.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def abort(self):
        """Close output streams if the ROOT scan fails before normal finish."""
        for handle in (self.events_file, self.clusters_file):
            if not handle.closed:
                handle.close()


def _csv_scalar(value):
    """Convert NumPy scalar/object values into CSV-friendly Python values."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return str(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return "" if value is None else value
