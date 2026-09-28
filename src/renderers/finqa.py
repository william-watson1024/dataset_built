from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import TableRecord
from .table_renderer import RenderResult, TableRenderer, normalize_table


class FinQATableRenderer:
    """FinQA adapter; table extraction is separate from PNG rendering."""

    name = "finqa"
    dataset = "FinQA"

    def __init__(self, engine: TableRenderer | None = None) -> None:
        self.engine = engine or TableRenderer(browser="firefox_bidi")

    @staticmethod
    def normalize(table: Any):
        return normalize_table(table)

    @staticmethod
    def compute_hash(table: Any) -> str:
        from .table_renderer import table_hash
        return table_hash(table)

    def load_tables(self, datasets_root: Path) -> list[TableRecord]:
        records: list[TableRecord] = []
        files = [("train", datasets_root / "FinQA/dataset/train.json"),
                 ("validation", datasets_root / "FinQA/dataset/dev.json"),
                 ("test", datasets_root / "FinQA/dataset/test.json")]
        for split, path in files:
            items = json.loads(path.read_text(encoding="utf-8"))
            for index, item in enumerate(items):
                raw_table = item.get("table_ori") or item.get("table")
                if raw_table is None:
                    raise ValueError(f"FinQA record has no table: {path}:{index}")
                source_id = str(item.get("id") or item.get("filename") or index)
                records.append(TableRecord(self.dataset, split, source_id, normalize_table(raw_table)))
        return records

    def render(self, table: Any, output_path: str | Path) -> RenderResult:
        return self.engine.render(table, output_path)
