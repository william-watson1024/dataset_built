from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.converter import Converter


class GovReportConverter(Converter):
    """Stream GovReport Parquet shards into report-level samples."""

    TASK = "government_report_summarization"
    PROMPT = "Summarize the following government report."

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("GovReport conversion requires pyarrow") from exc

        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"GovReport source file not found: {source_file}")
            parquet_file = parquet.ParquetFile(source_file)
            if "source" not in parquet_file.schema.names or "target" not in parquet_file.schema.names:
                raise ValueError(f"GovReport shard must contain source and target columns: {source_file}")
            row_index = 0
            for batch in parquet_file.iter_batches(batch_size=256, columns=["source", "target"]):
                for row in batch.to_pylist():
                    source = row.get("source")
                    target = row.get("target")
                    if not isinstance(source, str) or not isinstance(target, str):
                        raise ValueError(f"GovReport source/target must be strings: {source_file}:{row_index}")
                    yield {
                        "source": source,
                        "target": target,
                        "__source_file": source_file.name,
                        "__source_index": row_index,
                        "__original_split": split,
                    }
                    row_index += 1

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        source = raw_sample["source"]
        target = raw_sample["target"]
        source_ref = f"{raw_sample['__source_file']}:{raw_sample['__source_index']}"
        return {
            "id": f"govreport:{split}:{source_ref}",
            "source": "GovReport",
            "split": split,
            "task": self.TASK,
            "language": "en",
            "input": {
                "images": [],
                "context": source,
                "prompt": self.PROMPT,
                "history": [],
            },
            "output": {
                "text": target,
                "format": "text",
                "references": [target],
            },
            "annotations": {
                "supervision": {
                    "summary_available": True,
                    "source_field": "source",
                    "target_field": "target",
                },
                "evaluation": {
                    "summary_available": True,
                },
                "raw": {
                    "source_file": raw_sample["__source_file"],
                    "source_index": raw_sample["__source_index"],
                    "original_split": raw_sample["__original_split"],
                },
            },
            "meta": {
                "source_id": source_ref,
                "group_id": f"govreport:report:{source_ref}",
                "subset": None,
                "original_split": raw_sample["__original_split"],
                "transforms": ["govreport_parquet_to_canonical:v1"],
                "source_file": raw_sample["__source_file"],
            },
        }
