"""Direct ROOT source adapter backed by WatChMaL DataTools' WCSimFile.

This reads WCSim ``wcsimT`` ROOT entries one at a time. It does not call
DataTools' ``event_dump.py`` and does not create an intermediate NPZ.
Set ``DATATOOLS`` to the DataTools checkout, or pass ``--datatools-path`` to
the shared runner. PyROOT/WCSim runtime setup is still required by DataTools.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def import_wcsim_file(datatools_path: Path | None = None):
    """Import DataTools' ROOT reader, with a helpful environment error."""
    candidates = [datatools_path]
    env_path = os.environ.get("DATATOOLS")
    if env_path:
        candidates.append(Path(env_path))
    for candidate in candidates:
        if candidate is not None:
            root = candidate.expanduser().resolve()
            if (root / "root_utils" / "root_file_utils.py").is_file():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                break
    else:
        raise RuntimeError(
            "Cannot locate WatChMaL DataTools. Set DATATOOLS to its checkout "
            "or provide --datatools-path. Also source the WCSim/PyROOT runtime."
        )
    try:
        from root_utils.root_file_utils import WCSimFile
    except ImportError as exc:
        raise RuntimeError(
            "Could not import DataTools WCSimFile; check DATATOOLS and the "
            "WCSim/PyROOT environment."
        ) from exc
    return WCSimFile


def event_from_wcsim_file(reader: Any, entry: int, input_path: Path) -> dict[str, Any]:
    """Adapt one DataTools WCSimFile entry to the stable plugin event schema."""
    reader.get_event(entry)
    truth = reader.get_event_info()
    digits = reader.get_digitized_hits()
    triggers = reader.get_triggers()
    return {
        "event_id": int(entry),
        "root_file": str(input_path),
        "pid": int(truth["pid"]),
        "energy": float(truth["energy"]),
        "digi_hit_time": digits["time"],
        "digi_hit_pmt": digits["pmt"],
        "digi_hit_charge": digits["charge"],
        "digi_hit_trigger": digits["trigger"],
        "trigger_time": triggers["time"],
    }


class DataToolsWCSimRootSource:
    """Context-managed iterator over DataTools-decoded WCSim ROOT events."""

    def __init__(self, path: Path, datatools_path: Path | None = None):
        self.path = path
        self._reader_type = import_wcsim_file(datatools_path)
        self.reader = None

    def __enter__(self):
        self.reader = self._reader_type(str(self.path))
        return self

    def __exit__(self, *_exc):
        if self.reader is not None:
            root_file = getattr(self.reader, "file", None)
            if root_file is not None and root_file.IsOpen():
                root_file.Close()
            self.reader = None

    @property
    def entries(self) -> int:
        if self.reader is None:
            raise RuntimeError("ROOT source is not open")
        return int(self.reader.nevent)

    def read_event(self, entry: int) -> dict[str, Any]:
        if self.reader is None:
            raise RuntimeError("ROOT source is not open")
        return event_from_wcsim_file(self.reader, entry, self.path)
