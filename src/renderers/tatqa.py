from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import TableRecord
from .table_renderer import RenderResult, TableRenderer, normalize_table, table_hash


class TATQATableRenderer:
    """TAT-QA adapter; table extraction is separate from PNG rendering."""

    name = "tatqa"
    dataset = "TAT-QA"

    def __init__(self, engine: TableRenderer | None = None) -> None:
        self.engine = engine or TableRenderer(browser="firefox_bidi")

    @staticmethod
    def normalize(table: Any) -> list[list[str]]:
        return normalize_table(table)

    @staticmethod
    def compute_hash(table: Any) -> str:
        return table_hash(table)

    def load_tables(self, datasets_root: Path) -> list[TableRecord]:
        records: list[TableRecord] = []
        files = [("train", datasets_root / "TAT-QA/tatqa_dataset_train.json"),
                 ("validation", datasets_root / "TAT-QA/tatqa_dataset_dev.json")]
        for split, path in files:
            items = json.loads(path.read_text(encoding="utf-8"))
            for index, item in enumerate(items):
                table_obj = item.get("table") or {}
                raw_table = table_obj.get("table") if isinstance(table_obj, dict) else None
                if raw_table is None:
                    raise ValueError(f"TAT-QA record has no table: {path}:{index}")
                source_id = str(table_obj.get("uid") or item.get("uid") or index)
                records.append(TableRecord(self.dataset, split, source_id, normalize_table(raw_table)))
        return records

    def render(self, table: Any, output_path: str | Path) -> RenderResult:
        return self.engine.render(table, output_path)
