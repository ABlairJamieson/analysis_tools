"""Small, tested PMT time-density helpers for the real-data burst study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Burst:
    """A local PMT hit-density maximum and the hits assigned to its window."""

    center_ns: float
    start_ns: float
    end_ns: float
    n_hits: int
    n_pmts: int
    charge: float


def find_bursts(times_ns, charges, pmts, *, min_ns=0.0, max_ns=10_000.0,
                bin_ns=10.0, width_ns=50.0, min_pmts=10,
                min_separation_ns=100.0, max_bursts=20) -> list[Burst]:
    """Find separated, thresholded local maxima in calibrated PMT hit time.

    This is a time-density finder, not a particle or geometry clusterer.
    ``width_ns`` defines each burst's hit window; nearby maxima compete by
    smoothed hit count, and only windows with ``min_pmts`` distinct PMTs pass.
    """
    values = (min_ns, max_ns, bin_ns, width_ns, min_separation_ns)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Burst time settings must be finite")
    if (min_ns < 0 or max_ns <= min_ns or bin_ns <= 0 or width_ns <= 0
            or min_separation_ns <= 0 or min_pmts < 1 or max_bursts < 1):
        raise ValueError("Invalid burst search interval or threshold")

    times = np.asarray(times_ns, dtype=float).reshape(-1)
    charges = np.asarray(charges, dtype=float).reshape(-1)
    pmts = np.asarray(pmts, dtype=np.int64).reshape(-1)
    if not (len(times) == len(charges) == len(pmts)):
        raise ValueError("PMT time, charge, and ID arrays must have equal lengths")
    finite = np.isfinite(times) & np.isfinite(charges)
    times, charges, pmts = times[finite], charges[finite], pmts[finite]
    if not len(times):
        return []

    edges = np.arange(min_ns, max_ns + bin_ns, bin_ns)
    counts, _ = np.histogram(times, bins=edges)
    width_bins = max(1, int(round(width_ns / bin_ns)))
    smoothed = np.convolve(counts, np.ones(width_bins, dtype=int), mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    maxima = np.flatnonzero(
        (smoothed >= min_pmts)
        & (smoothed >= np.r_[0, smoothed[:-1]])
        & (smoothed > np.r_[smoothed[1:], 0])
    )

    chosen: list[Burst] = []
    for index in sorted(maxima, key=lambda candidate: smoothed[candidate], reverse=True):
        center = float(centers[index])
        if any(abs(center - burst.center_ns) < min_separation_ns for burst in chosen):
            continue
        start, end = center - width_ns / 2, center + width_ns / 2
        inside = (times >= start) & (times < end)
        distinct_pmts = int(np.unique(pmts[inside]).size)
        if distinct_pmts < min_pmts:
            continue
        chosen.append(Burst(center, start, end, int(inside.sum()), distinct_pmts,
                            float(charges[inside].sum())))
        if len(chosen) >= max_bursts:
            break
    return sorted(chosen, key=lambda burst: burst.center_ns)


def t0_group_centers(t0_times_ns, separation_ns=50.0) -> list[float]:
    """Collapse nearby corrected T0 PMT times into median beam-time groups."""
    if separation_ns <= 0 or not np.isfinite(separation_ns):
        raise ValueError("separation_ns must be a positive finite value")
    values = np.asarray(t0_times_ns, dtype=float).reshape(-1)
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return []
    split_at = np.flatnonzero(np.diff(values) > separation_ns) + 1
    return [float(np.median(group)) for group in np.split(values, split_at)]
