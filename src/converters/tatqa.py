from __future__ import annotations

import json
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.core.converter import Converter
from src.renderers.tatqa import TATQATableRenderer


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class TATQAConverter(Converter):
    """Convert TAT-QA report questions and render each source table once."""

    def __init__(self, config: dict[str, Any], project_root: Path, media_dir: Path) -> None:
        super().__init__(config, project_root)
        self.media_dir = media_dir
        self.renderer = TATQATableRenderer()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._prepared: dict[str, list[dict[str, Any]]] = {}

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    def _read_split(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"TAT-QA source file not found: {source_file}")
            data = json.loads(source_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"TAT-QA source file must contain a JSON array: {source_file}")
            for row_index, row in enumerate(data):
                if not isinstance(row, dict):
                    raise ValueError(f"TAT-QA report must be an object: {source_file}:{row_index}")
                questions = row.get("questions")
                if not isinstance(questions, list):
                    raise ValueError(f"TAT-QA report questions must be a list: {source_file}:{row_index}")
                for question_index, question in enumerate(questions):
                    if not isinstance(question, dict):
                        raise ValueError(
                            f"TAT-QA question must be an object: {source_file}:{row_index}:{question_index}"
                        )
                    raw = dict(row)
                    raw["questions"] = [question]
                    raw["__source_file"] = str(source_file)
                    raw["__row_index"] = row_index
                    raw["__question_index"] = question_index
                    records.append(raw)
                    if limit is not None and len(records) >= limit:
                        return records
        return records

    @staticmethod
    def _table(raw_report: dict[str, Any]) -> list[list[Any]]:
        table = raw_report.get("table")
        if not isinstance(table, dict) or not isinstance(table.get("table"), list):
            raise ValueError(f"TAT-QA report has no table.table: row {raw_report.get('__row_index', '<unknown>')}")
        return table["table"]

    def _render_table(self, table: list[list[Any]]) -> None:
        digest = self.renderer.compute_hash(table)
        output = self.media_dir / f"{digest}.png"
        if not output.is_file() or output.stat().st_size == 0:
            self.renderer.render(table, output)

    def prepare_split(self, split: str, limit: int | None = None) -> None:
        reports = self._read_split(split, limit)
        unique_tables: dict[str, list[list[Any]]] = {}
        for report in reports:
            table = self._table(report)
            unique_tables.setdefault(self.renderer.compute_hash(table), table)
        with ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(self._render_table, unique_tables.values()))
        self._prepared[split] = reports

    def close(self) -> None:
        self.renderer.engine.close()

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        if split in self._prepared:
            yield from self._prepared[split]
            return
        yield from self._read_split(split)

    @staticmethod
    def _context(raw_report: dict[str, Any]) -> str:
        paragraphs = raw_report.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            return str(paragraphs)
        ordered = sorted(
            (paragraph for paragraph in paragraphs if isinstance(paragraph, dict)),
            key=lambda paragraph: (paragraph.get("order", 0), str(paragraph.get("uid", ""))),
        )
        return "\n\n".join(str(paragraph.get("text", "")) for paragraph in ordered)

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @classmethod
    def _answer(cls, answer: Any) -> tuple[str, list[str], str]:
        if answer is None:
            return "", [], "text"
        if isinstance(answer, list):
            references = [cls._stringify(item) for item in answer]
            if len(answer) == 1:
                text = references[0]
                return text, references, "number" if _NUMBER_RE.fullmatch(text.strip()) else "text"
            return json.dumps(answer, ensure_ascii=False), references, "json"
        text = cls._stringify(answer)
        if isinstance(answer, dict):
            return text, [text], "json"
        return text, [text], "number" if _NUMBER_RE.fullmatch(text.strip()) else "text"

    @staticmethod
    def _supervision(question: dict[str, Any]) -> dict[str, Any]:
        return {
            "derivation": question.get("derivation", ""),
            "rel_paragraphs": question.get("rel_paragraphs", []),
            "req_comparison": question.get("req_comparison", False),
        }

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        questions = raw_sample.get("questions")
        if not isinstance(questions, list) or len(questions) != 1:
            raise ValueError("TAT-QA converter expects one question per raw sample")
        question = questions[0]
        if not isinstance(question, dict):
            raise ValueError("TAT-QA question must be an object")

        raw_table = self._table(raw_sample)
        digest = self.renderer.compute_hash(raw_table)
        media_path = self.media_dir / f"{digest}.png"
        if not media_path.is_file() or media_path.stat().st_size == 0:
            self.renderer.render(raw_table, media_path)

        table = raw_sample.get("table", {})
        table_uid = str(table.get("uid") or digest) if isinstance(table, dict) else digest
        source_id = str(question.get("uid") or f"{table_uid}:{question.get('order', 0)}")
        answer_text, references, output_format = self._answer(question.get("answer"))
        original_split = "dev" if split == "validation" else split
        transforms = ["tatqa_table_to_png:v1"]
        if split == "validation":
            transforms.append("dev_to_validation:v1")

        return {
            "id": f"tatqa:{split}:{source_id}",
            "source": "TAT-QA",
            "split": split,
            "task": "financial_table_reasoning",
            "language": "en",
            "input": {
                "images": [{"storage": "file", "path": f"media/{digest}.png"}],
                "context": self._context(raw_sample),
                "prompt": str(question.get("question", "")),
                "history": [],
            },
            "output": {
                "text": answer_text,
                "format": output_format,
                "references": references,
            },
            "annotations": {
                "supervision": self._supervision(question),
                "evaluation": {
                    "answer_type": question.get("answer_type"),
                    "answer_from": question.get("answer_from"),
                    "scale": question.get("scale", ""),
                    "has_answer": bool(references),
                },
                "raw": {
                    "table": raw_sample.get("table"),
                    "paragraphs": raw_sample.get("paragraphs"),
                    "question": question,
                    "source_file": Path(str(raw_sample.get("__source_file", ""))).name,
                    "row_index": raw_sample.get("__row_index"),
                },
            },
            "meta": {
                "source_id": source_id,
                "group_id": f"tatqa:report:{table_uid}:table:{digest}",
                "subset": None,
                "original_split": original_split,
                "transforms": transforms,
            },
        }
