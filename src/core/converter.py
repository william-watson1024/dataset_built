from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class Converter(ABC):
    """Small common interface for dataset converters."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    @abstractmethod
    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        """Yield raw records belonging to a canonical split."""

    @abstractmethod
    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        """Convert one raw record into one Canonical Sample."""

    def iter_samples(self, split: str) -> Iterator[dict[str, Any]]:
        for raw_sample in self.load_split(split):
            yield self.convert_sample(raw_sample, split)
