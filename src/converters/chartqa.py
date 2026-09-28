from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.converter import Converter
from src.converters.docvqa import MediaStore


class ChartQAConverter(Converter):
    """Convert the local ChartQA JSON annotations and PNG charts."""

    def __init__(self, config: dict[str, Any], project_root: Path, media_dir: Path) -> None:
        super().__init__(config, project_root)
        self.media_store = MediaStore(media_dir)

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    @staticmethod
    def _subset(source_file: Path) -> str:
        if source_file.stem.endswith("_human"):
            return "human"
        if source_file.stem.endswith("_augmented"):
            return "machine"
        raise ValueError(f"Cannot determine ChartQA source type from filename: {source_file.name}")

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"ChartQA annotation file not found: {source_file}")
            data = json.loads(source_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"ChartQA annotation file must contain a JSON array: {source_file}")
            subset = self._subset(source_file)
            source_label = f"{source_file.parent.name}/{source_file.name}"
            image_dir = source_file.parent / "png"
            for row_index, row in enumerate(data):
                if not isinstance(row, dict):
                    raise ValueError(f"ChartQA row {row_index} is not an object: {source_file}")
                raw = dict(row)
                raw["__source_file"] = source_label
                raw["__source_path"] = str(source_file)
                raw["__image_dir"] = str(image_dir)
                raw["__row_index"] = row_index
                raw["__subset"] = subset
                yield raw

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        image_name = str(raw_sample["imgname"])
        image_path = Path(raw_sample["__image_dir"]) / image_name
        media_path = self.media_store.materialize_file(image_path)
        question = str(raw_sample["query"])
        answer = raw_sample.get("label")
        references = [] if answer is None else [str(answer)]
        subset = str(raw_sample["__subset"])
        row_index = int(raw_sample["__row_index"])
        source_id = f"{split}:{subset}:{row_index}"
        source_file = str(raw_sample["__source_file"])
        return {
            "id": f"chartqa:{source_id}",
            "source": "ChartQA",
            "split": split,
            "task": "chart_question_answering",
            "language": "en",
            "input": {
                "images": [{"storage": "file", "path": media_path}],
                "context": "",
                "prompt": question,
                "history": [],
            },
            "output": {
                "text": references[0] if references else "",
                "format": "text",
                "references": references,
            },
            "annotations": {
                "supervision": {"source_type": subset},
                "evaluation": {
                    "answer": answer,
                    "has_answers": bool(references),
                },
                "raw": {
                    "imgname": raw_sample["imgname"],
                    "query": raw_sample["query"],
                    "label": answer,
                    "source_type": subset,
                    "source_file": source_file,
                    "row_index": row_index,
                },
            },
            "meta": {
                "source_id": source_id,
                "group_id": f"chartqa:chart:{image_name}",
                "subset": subset,
                "original_split": "val" if split == "validation" else split,
                "transforms": ["chartqa_file_to_processed_media:v1"],
            },
        }
