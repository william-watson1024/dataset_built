from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.converters.docvqa import MediaStore
from src.core.converter import Converter


class STVQAConverter(Converter):
    """Convert ST-VQA image/question records into one sample per QA pair."""

    TASK = "scene_text_vqa"

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
            raise RuntimeError("ST-VQA conversion requires pyarrow") from exc

        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"ST-VQA source file not found: {source_file}")
            parquet_file = parquet.ParquetFile(source_file)
            fields = set(parquet_file.schema_arrow.names)
            if "image" not in fields:
                raise ValueError(f"ST-VQA shard must contain an image column: {source_file}")
            if "qas" not in fields and not ({"question", "query"} & fields):
                raise ValueError(f"ST-VQA shard must contain qas or question/query columns: {source_file}")

            row_index = 0
            for batch in parquet_file.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    if not isinstance(row, dict):
                        raise ValueError(f"ST-VQA row is not an object: {source_file}:{row_index}")
                    base = dict(row)
                    qas = base.get("qas")
                    if qas is None:
                        qas = [base]
                    if not isinstance(qas, list):
                        raise ValueError(f"ST-VQA qas must be an array: {source_file}:{row_index}")
                    for qa_index, qa in enumerate(qas):
                        if not isinstance(qa, dict):
                            raise ValueError(
                                f"ST-VQA QA must be an object: {source_file}:{row_index}:{qa_index}"
                            )
                        raw = dict(base)
                        raw["__qa"] = dict(qa)
                        raw["__source_file"] = source_file.name
                        raw["__source_path"] = str(source_file)
                        raw["__source_row"] = row_index
                        raw["__qa_index"] = qa_index
                        raw["__original_split"] = split
                        yield raw
                    row_index += 1

    @staticmethod
    def _first_present(*objects: dict[str, Any], names: tuple[str, ...]) -> Any:
        for obj in objects:
            for name in names:
                if name in obj and obj[name] is not None:
                    return obj[name]
        return None

    @staticmethod
    def _references(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            references: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    nested = STVQAConverter._first_present(item, names=("answer", "text", "value"))
                    references.extend(STVQAConverter._references(nested if nested is not None else item))
                else:
                    references.append(str(item))
            return references
        if isinstance(value, dict):
            nested = STVQAConverter._first_present(value, names=("answer", "text", "value"))
            return STVQAConverter._references(nested) if nested is not None else [json.dumps(value, ensure_ascii=False)]
        return [str(value)]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Keep raw metadata JSON-compatible without copying image bytes."""
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"__type__": "bytes", "size": len(value)}
        if isinstance(value, dict):
            return {str(key): STVQAConverter._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [STVQAConverter._json_safe(item) for item in value]
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return str(value)
        return value

    @staticmethod
    def _image_bytes(image: Any) -> bytes | None:
        if isinstance(image, (bytes, bytearray, memoryview)):
            return bytes(image)
        if isinstance(image, dict) and image.get("bytes") is not None:
            return bytes(image["bytes"])
        return None

    def _image_file(self, raw_sample: dict[str, Any], image: Any) -> Path:
        if isinstance(image, str):
            image_name = image
        elif isinstance(image, dict):
            image_name = image.get("path") or image.get("filename") or image.get("file_name")
        else:
            image_name = None
        if not image_name:
            raise ValueError(
                f"ST-VQA image has neither embedded bytes nor a path: "
                f"{raw_sample.get('__source_file')}:{raw_sample.get('__source_row')}"
            )
        image_path = Path(str(image_name))
        source_path = Path(str(raw_sample["__source_path"]))
        candidates = (
            image_path,
            source_path.parent / image_path,
            source_path.parent / "images" / image_path,
            self.project_root / image_path,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"ST-VQA image not found: {image_name} "
            f"(checked {', '.join(str(candidate) for candidate in candidates)})"
        )

    def _materialize_image(self, raw_sample: dict[str, Any]) -> tuple[str, str]:
        image = raw_sample.get("image")
        image_bytes = self._image_bytes(image)
        if image_bytes is not None:
            original_name = image.get("path") if isinstance(image, dict) else None
            digest = hashlib.sha256(image_bytes).hexdigest()
            return self.media_store.materialize_bytes(image_bytes, original_name), f"sha256:{digest}"

        image_path = self._image_file(raw_sample, image)
        image_bytes = image_path.read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        return self.media_store.materialize_file(image_path), f"sha256:{digest}"

    def _default_task_type(self) -> Any:
        rules = self.config.get("evaluation", {}).get("rules", {})
        return rules.get("default_task_type")

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        qa = raw_sample.get("__qa") or raw_sample
        if not isinstance(qa, dict):
            raise ValueError("ST-VQA QA record must be an object")

        question = self._first_present(qa, raw_sample, names=("question", "query"))
        if question is None:
            raise ValueError(
                f"ST-VQA QA has no question: {raw_sample.get('__source_file')}:{raw_sample.get('__source_row')}"
            )
        answer_value = self._first_present(qa, raw_sample, names=("answers", "answer"))
        references = self._references(answer_value)
        media_path, image_identity = self._materialize_image(raw_sample)

        question_id_value = self._first_present(
            qa, raw_sample, names=("question_id", "questionId", "qa_id", "id")
        )
        question_id = str(question_id_value) if question_id_value is not None else (
            f"{raw_sample['__source_file']}:{raw_sample['__source_row']}:{raw_sample['__qa_index']}"
        )
        image_id_value = self._first_present(
            qa, raw_sample, names=("image_id", "imageId", "img_id")
        )
        image = raw_sample.get("image")
        image_path_value = image.get("path") if isinstance(image, dict) else image if isinstance(image, str) else None
        image_id = str(image_id_value) if image_id_value is not None else str(image_path_value or image_identity)

        task_type_value = self._first_present(qa, raw_sample, names=("task_type", "task", "task_id"))
        task_type = str(task_type_value) if task_type_value is not None else self._default_task_type()
        subset = task_type

        evaluation: dict[str, Any] = {
            "answer_count": len(references),
            "has_answers": bool(references),
        }
        for key in ("task", "task_type", "task_id", "lexicon", "candidates"):
            value = self._first_present(qa, raw_sample, names=(key,))
            if value is not None:
                evaluation[key] = self._json_safe(value)
        if task_type is not None and "task_type" not in evaluation:
            evaluation["task_type"] = task_type
        for key in ("lexicon", "candidates"):
            if key in evaluation and isinstance(evaluation[key], (list, tuple, dict)):
                evaluation[f"{key}_count"] = len(evaluation[key])

        raw_without_internal = {
            key: value
            for key, value in raw_sample.items()
            if not key.startswith("__") and key != "image"
        }
        raw_without_internal["image"] = self._json_safe(image)
        raw_without_internal["qa"] = self._json_safe(qa)
        raw_without_internal.update(
            {
                "source_file": raw_sample["__source_file"],
                "source_row": raw_sample["__source_row"],
                "qa_index": raw_sample["__qa_index"],
                "question_id": question_id_value,
                "image_id": image_id_value,
            }
        )

        source_id = question_id
        return {
            "id": f"stvqa:{split}:{source_id}",
            "source": "ST-VQA",
            "split": split,
            "task": self.TASK,
            "language": "en",
            "input": {
                "images": [{"storage": "file", "path": media_path}],
                "context": "",
                "prompt": str(question),
                "history": [],
            },
            "output": {
                "text": references[0] if references else "",
                "format": "text",
                "references": references,
            },
            "annotations": {
                "supervision": {
                    "answers": references,
                    "answer_count": len(references),
                },
                "evaluation": evaluation,
                "raw": raw_without_internal,
            },
            "meta": {
                "source_id": source_id,
                "group_id": f"stvqa:image:{image_identity if image_identity else image_id}",
                "subset": str(subset) if subset is not None else None,
                "original_split": str(raw_sample.get("__original_split", split)),
                "transforms": [
                    "parquet_or_file_image_to_processed_media:v1",
                    "stvqa_qas_to_question_samples:v1",
                ],
                "source_file": raw_sample["__source_file"],
            },
        }
