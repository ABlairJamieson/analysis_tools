"""Shared plugin lifecycle primitives used by study runners.

Keep source-specific reading and analysis-specific calculations out of this
module. A source adapter constructs :class:`StudyContext`; a study consumes
one batch and adds named selections without being able to overwrite parents.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class SelectionBook:
    """Per-batch AND-only named selections with parent-child cut flow."""

    def __init__(self, batch_size: int):
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        self._masks: dict[str, np.ndarray] = {"all": np.ones(batch_size, dtype=bool)}
        self._parents: dict[str, str | None] = {"all": None}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._masks)

    def add(self, name: str, mask: Any, *, parent: str = "all") -> np.ndarray:
        """Add a new mask ANDed with its parent; existing masks are immutable."""
        if not name or name in self._masks:
            raise ValueError(f"Selection name must be new and nonempty: {name!r}")
        if parent not in self._masks:
            raise KeyError(f"Unknown parent selection {parent!r}")
        local = np.asarray(mask, dtype=bool)
        if local.ndim != 1 or len(local) != len(self._masks["all"]):
            raise ValueError("Selection mask must be one boolean value per event")
        combined = self._masks[parent] & local
        self._masks[name] = combined
        self._parents[name] = parent
        return combined

    def mask(self, name: str) -> np.ndarray:
        return self._masks[name]

    def counts(self) -> dict[str, int]:
        return {name: int(mask.sum()) for name, mask in self._masks.items()}

    def parent(self, name: str) -> str | None:
        return self._parents[name]


@dataclass
class StudyContext:
    """Input metadata and shared selections for one plugin batch.

    Beamline-specific fields are optional so ROOT-source plugins can share the
    same lifecycle without pretending every source has those measurements.
    """

    input_path: Path
    output_dir: Path
    entry_start: int
    selections: SelectionBook
    corrected_tdc_hits: list[list[Any]] | None = None
    reference_times_ns: list[tuple[float | None, float | None]] | None = None


def load_study(path: Path) -> type:
    """Load a Python study file that defines a ``Study`` class."""
    spec = importlib.util.spec_from_file_location("analysis_user_study", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load study file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study_class = getattr(module, "Study", None)
    if study_class is None:
        raise TypeError(f"Study file {path} must define a Study class")
    return study_class
