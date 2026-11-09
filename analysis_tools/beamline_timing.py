"""Shared beamline TDC reference-time correction.

The beam-monitor TDC banks use channel 31 for channels <= 31 and channel 46
for channels > 31. This is an intra-beamline correction only; it does not
align the beamline clock with the WCTE PMT clock. Relative detector offsets,
when required by a particular analysis, are a separate subsequent operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


REFERENCE_IDS = (31, 46)


@dataclass(frozen=True)
class TDCHit:
    """One finite raw TDC hit and its reference-corrected time, if available."""

    channel_id: int
    raw_ns: float
    corrected_ns: float | None


def correct_tdc_hits(
    hits: Iterable[tuple[int, float]], *, mode: str = "reference"
) -> tuple[list[TDCHit], tuple[float | None, float | None]]:
    """Correct hits using the first finite reference hit in each TDC bank.

    Repeated detector hits are retained and in input order. The reference
    channels themselves have no corrected detector time. If a bank reference
    is absent its detector hits receive ``None`` (never raw-time fallback).
    ``mode='raw'`` is an explicit diagnostic mode.
    """
    if mode not in ("reference", "raw"):
        raise ValueError("TDC time mode must be 'reference' or 'raw'")

    finite_hits = [(int(channel), float(time)) for channel, time in hits
                   if time is not None and isfinite(float(time))]
    refs: list[float | None] = [None, None]
    for channel, time in finite_hits:
        if channel == REFERENCE_IDS[0] and refs[0] is None:
            refs[0] = time
        elif channel == REFERENCE_IDS[1] and refs[1] is None:
            refs[1] = time

    corrected: list[TDCHit] = []
    for channel, time in finite_hits:
        if mode == "raw":
            value = time
        elif channel in REFERENCE_IDS:
            value = None
        else:
            reference = refs[0 if channel <= REFERENCE_IDS[0] else 1]
            value = time - reference if reference is not None else None
        corrected.append(TDCHit(channel, time, value))
    return corrected, (refs[0], refs[1])
