from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.converter import Converter


class MediaStore:
    """Materialize embedded images once and return processed-relative paths."""

    def __init__(self, media_dir: Path) -> None:
        self.media_dir = media_dir
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._by_digest: dict[str, str] = {}

    def _materialize_bytes(self, data: bytes, original_name: str | None) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if digest in self._by_digest:
            return self._by_digest[digest]

        original_name = Path(original_name or "").name
        suffix = Path(original_name).suffix or ".bin"
        stem = Path(original_name).stem or "image"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"
        filename = f"{stem}{suffix}"
        candidate = self.media_dir / filename
        if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            filename = f"{stem}_{digest[:12]}{suffix}"
            candidate = self.media_dir / filename
        if not candidate.exists():
            candidate.write_bytes(data)
        media_path = f"media/{filename}"
        self._by_digest[digest] = media_path
        return media_path

    def materialize(self, image: dict[str, Any], image_path: str | None) -> str:
        image_bytes = image.get("bytes")
        if image_bytes is None:
            raise ValueError("DocVQA image resource has no embedded bytes")
        return self._materialize_bytes(bytes(image_bytes), image_path)

    def materialize_file(self, source_path: Path) -> str:
        if not source_path.is_file():
            raise FileNotFoundError(f"Image file not found: {source_path}")
        return self._materialize_bytes(source_path.read_bytes(), source_path.name)


class DocVQAConverter(Converter):
    """Streaming converter for local DocVQA Parquet shards."""

    def __init__(self, config: dict[str, Any], project_root: Path, media_dir: Path) -> None:
        super().__init__(config, project_root)
        self.media_store = MediaStore(media_dir)

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("DocVQA conversion requires pyarrow") from exc

        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"DocVQA source file not found: {source_file}")
            parquet_file = parquet.ParquetFile(source_file)
            row_index = 0
            for batch in parquet_file.iter_batches(batch_size=256):
                for raw in batch.to_pylist():
                    raw["__source_row"] = row_index
                    row_index += 1
                    yield raw

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        answers = raw_sample.get("answers")
        references = [str(answer) for answer in answers] if answers else []
        image = raw_sample.get("image") or {}
        image_path = image.get("path") if isinstance(image, dict) else None
        media_path = self.media_store.materialize(image, image_path)

        question_id = str(raw_sample["questionId"])
        document_id = str(raw_sample["ucsf_document_id"])
        page_no = str(raw_sample["ucsf_document_page_no"])
        original_split = str(raw_sample.get("data_split", split))
        return {
            "id": f"docvqa:{split}:{question_id}",
            "source": "DocVQA",
            "split": split,
            "task": "document_visual_question_answering",
            "language": "en",
            "input": {
                "images": [{"storage": "file", "path": media_path}],
                "context": "",
                "prompt": str(raw_sample["question"]),
                "history": [],
            },
            "output": {
                "text": references[0] if references else "",
                "format": "text",
                "references": references,
            },
            "annotations": {
                "supervision": {"question_types": raw_sample.get("question_types")},
                "evaluation": {
                    "answers": answers,
                    "answer_count": len(references),
                    "has_answers": bool(references),
                },
                "raw": {
                    "questionId": raw_sample["questionId"],
                    "docId": raw_sample.get("docId"),
                    "ucsf_document_id": raw_sample.get("ucsf_document_id"),
                    "ucsf_document_page_no": raw_sample.get("ucsf_document_page_no"),
                    "image_path": image_path,
                    "data_split": raw_sample.get("data_split"),
                },
            },
            "meta": {
                "source_id": question_id,
                "group_id": f"docvqa:document:{document_id}:page:{page_no}",
                "subset": None,
                "original_split": original_split,
                "transforms": ["parquet_embedded_image_to_file:v1"],
            },
        }
