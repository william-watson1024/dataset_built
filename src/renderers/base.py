from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .table_renderer import RenderResult


class DatasetTableRenderer(Protocol):
    """Small common interface implemented by dataset-specific table adapters."""

    name: str
    dataset: str

    def render(self, table: Any, output_path: str | Path) -> RenderResult:
        ...

    @staticmethod
    def normalize(table: Any) -> list[list[str]]:
        ...

    @staticmethod
    def compute_hash(table: Any) -> str:
        ...
