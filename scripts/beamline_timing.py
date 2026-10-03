"""Reference-channel correction for beamline TDC hit times.

The beam monitor analysis uses TDC channel 31 for IDs <= 31 and channel 46
for IDs > 31. Keep every repeated hit; do not turn channel IDs into a dict.
This correction does not align the TDC clock with WCTE PMT time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


REFERENCE_IDS = (31, 46)


@dataclass(frozen=True)
class TDCHit:
    channel_id: int
    raw_ns: float
    corrected_ns: float | None


def correct_tdc_hits(hits: list[tuple[int, float]], *, mode: str = "reference"
                     ) -> tuple[list[TDCHit], tuple[float | None, float | None]]:
    """Subtract the first hit on each reference channel, as beam_monitors_pid does.

    Missing references yield ``None`` corrected times for their channel bank.
    ``raw`` is an explicit diagnostic mode, not a fallback for missing refs.
    """
    if mode not in ("reference", "raw"):
        raise ValueError("TDC time mode must be 'reference' or 'raw'")
    refs: list[float | None] = [None, None]
    for channel, time in hits:
        if not isfinite(time):
            continue
        if channel == REFERENCE_IDS[0] and refs[0] is None:
            refs[0] = time
        elif channel == REFERENCE_IDS[1] and refs[1] is None:
            refs[1] = time
    corrected = []
    for channel, time in hits:
        if not isfinite(time):
            continue
        if mode == "raw":
            value = time
        elif channel in REFERENCE_IDS:
            value = None
        else:
            reference = refs[0 if channel <= REFERENCE_IDS[0] else 1]
            value = time - reference if reference is not None else None
        corrected.append(TDCHit(channel, time, value))
    return corrected, (refs[0], refs[1])
