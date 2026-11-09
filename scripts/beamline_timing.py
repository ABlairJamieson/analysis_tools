"""Backward-compatible imports for older scripts and external callers.

New code should import the shared implementation from
``analysis_tools.beamline_timing``. Keeping this module avoids breaking
commands and notebooks that used ``scripts.beamline_timing``.
"""

try:
    from analysis_tools.beamline_timing import REFERENCE_IDS, TDCHit, correct_tdc_hits
except ModuleNotFoundError as exc:
    # Direct execution sets sys.path to ``scripts/`` on some Python installs.
    if exc.name != "analysis_tools":
        raise
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis_tools.beamline_timing import REFERENCE_IDS, TDCHit, correct_tdc_hits

__all__ = ["REFERENCE_IDS", "TDCHit", "correct_tdc_hits"]
