from __future__ import annotations

from dataclasses import dataclass

from .table_renderer import Table


@dataclass(frozen=True)
class TableRecord:
    dataset: str
    split: str
    source_id: str
    table: Table
