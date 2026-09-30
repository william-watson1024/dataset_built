from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.core.converter import Converter
from src.renderers.finqa import FinQATableRenderer
from src.renderers.table_renderer import normalize_table, table_hash


class FinQAConverter(Converter):
    """Convert FinQA records and render each source table exactly once."""

    def __init__(self, config: dict[str, Any], project_root: Path, media_dir: Path) -> None:
        super().__init__(config, project_root)
        self.media_dir = media_dir
        self.renderer = FinQATableRenderer()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._prepared: dict[str, list[dict[str, Any]]] = {}

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    def _read_split(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"FinQA source file not found: {source_file}")
            data = json.loads(source_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"FinQA source file must contain a JSON array: {source_file}")
            for row_index, row in enumerate(data):
                if not isinstance(row, dict):
                    raise ValueError(f"FinQA record must be an object: {source_file}:{row_index}")
                raw = dict(row)
                raw["__source_file"] = str(source_file)
                raw["__row_index"] = row_index
                records.append(raw)
                if limit is not None and len(records) >= limit:
                    return records
        return records

    @staticmethod
    def _raw_table(raw_sample: dict[str, Any]) -> list[list[Any]]:
        table = raw_sample.get("table_ori") or raw_sample.get("table")
        if table is None:
            raise ValueError(f"FinQA record has no table: {raw_sample.get('id', '<missing>')}")
        return table

    def _render_table(self, table: list[list[Any]]) -> None:
        digest = table_hash(table)
        output = self.media_dir / f"{digest}.png"
        if not output.is_file() or output.stat().st_size == 0:
            self.renderer.render(table, output)

    def prepare_split(self, split: str, limit: int | None = None) -> None:
        records = self._read_split(split, limit)
        unique_tables: dict[str, list[list[Any]]] = {}
        for raw in records:
            table = self._raw_table(raw)
            unique_tables.setdefault(table_hash(table), table)
        with ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(self._render_table, unique_tables.values()))
        self._prepared[split] = records

    def close(self) -> None:
        self.renderer.engine.close()

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        if split in self._prepared:
            yield from self._prepared[split]
            return
        yield from self._read_split(split)

    @staticmethod
    def _context(raw_sample: dict[str, Any]) -> str:
        sections: list[str] = []
        for field in ("pre_text", "post_text"):
            value = raw_sample.get(field, [])
            if isinstance(value, list):
                text = "\n".join(str(item) for item in value)
            else:
                text = str(value)
            if text:
                sections.append(text)
        return "\n\n".join(sections)

    @staticmethod
    def _answer_text(answer: Any) -> str:
        if answer is None:
            return ""
        if isinstance(answer, (dict, list)):
            return json.dumps(answer, ensure_ascii=False)
        return str(answer)

    @staticmethod
    def _supervision(qa: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "program", "steps", "gold_inds", "exe_ans", "explanation",
            "program_re", "ann_table_rows", "ann_text_rows", "tfidftopn", "model_input",
        )
        return {field: qa.get(field) for field in fields}

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        qa = raw_sample.get("qa")
        if not isinstance(qa, dict):
            raise ValueError(f"FinQA record has no qa object: {raw_sample.get('id', '<missing>')}")
        raw_table = self._raw_table(raw_sample)
        digest = table_hash(raw_table)
        media_path = self.media_dir / f"{digest}.png"
        if not media_path.is_file() or media_path.stat().st_size == 0:
            self.renderer.render(raw_table, media_path)
        source_id = str(raw_sample.get("id") or raw_sample.get("filename") or raw_sample.get("__row_index"))
        filename = str(raw_sample.get("filename") or source_id)
        answer = qa.get("answer")
        # Some local FinQA records preserve the official execution answer in
        # ``exe_ans`` while leaving the textual ``answer`` field blank.  A
        # blank string is not a usable Canonical reference, so use the
        # execution answer as the explicit fallback and keep the raw QA
        # object unchanged in annotations.raw.
        if isinstance(answer, str) and not answer.strip():
            answer = qa.get("exe_ans")
        answer_text = self._answer_text(answer)
        references = [answer_text] if answer_text.strip() else []
        original_split = "dev" if split == "validation" else split
        transforms = ["finqa_table_to_png:v1"]
        if split == "validation":
            transforms.append("dev_to_validation:v1")
        return {
            "id": f"finqa:{split}:{source_id}",
            "source": "FinQA",
            "split": split,
            "task": "financial_table_reasoning",
            "language": "en",
            "input": {
                "images": [{"storage": "file", "path": f"media/{digest}.png"}],
                "context": self._context(raw_sample),
                "prompt": str(qa.get("question", "")),
                "history": [],
            },
            "output": {
                "text": answer_text,
                "format": "text",
                "references": references,
            },
            "annotations": {
                "supervision": self._supervision(qa),
                "evaluation": {
                    "answer": answer,
                    "exe_ans": qa.get("exe_ans"),
                    "has_answer": bool(references),
                },
                "raw": {
                    "id": raw_sample.get("id"),
                    "filename": raw_sample.get("filename"),
                    "table": raw_sample.get("table"),
                    "table_ori": raw_sample.get("table_ori"),
                    "qa": qa,
                    "source_file": raw_sample.get("__source_file"),
                    "row_index": raw_sample.get("__row_index"),
                },
            },
            "meta": {
                "source_id": source_id,
                "group_id": f"finqa:report:{filename}:table:{digest}",
                "subset": None,
                "original_split": original_split,
                "transforms": transforms,
            },
        }
