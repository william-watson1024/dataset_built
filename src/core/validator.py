from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class ValidationError(ValueError):
    """Validation error with sample/config location details."""


def _path(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return ".".join(parts) if parts else "$"


class SchemaValidator:
    def __init__(
        self,
        project_root: Path,
        sample_schema_path: str | Path = "assets/sample.schema.json",
        config_schema_path: str | Path = "assets/dataset_config.schema.json",
    ) -> None:
        self.project_root = project_root
        self.sample_schema = self._load_schema(sample_schema_path)
        self.config_schema = self._load_schema(config_schema_path)
        Draft202012Validator.check_schema(self.sample_schema)
        Draft202012Validator.check_schema(self.config_schema)
        self.sample_validator = Draft202012Validator(self.sample_schema)
        self.config_validator = Draft202012Validator(self.config_schema)
        self._parquet_metadata: dict[str, tuple[int, set[str]]] = {}

    def _load_schema(self, path: str | Path) -> dict[str, Any]:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.project_root / resolved
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValidationError(f"Schema not found: {resolved}") from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid Schema JSON: {resolved}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValidationError(f"Schema must be a JSON object: {resolved}")
        return data

    def validate_config(self, config: dict[str, Any]) -> None:
        errors = sorted(self.config_validator.iter_errors(config), key=lambda error: list(error.absolute_path))
        if errors:
            details = "; ".join(f"{_path(error)}: {error.message}" for error in errors)
            raise ValidationError(f"Dataset Config validation failed: {details}")

    def validate_sample(
        self,
        sample: dict[str, Any],
        check_resources: bool = False,
        resource_root: Path | None = None,
    ) -> None:
        sample_id = sample.get("id", "<missing>")
        errors = sorted(self.sample_validator.iter_errors(sample), key=lambda error: list(error.absolute_path))
        if errors:
            details = "; ".join(f"{_path(error)}: {error.message}" for error in errors)
            raise ValidationError(f"Sample {sample_id} validation failed: {details}")
        if check_resources:
            self.validate_resources(sample, resource_root=resource_root)

    def validate_resources(self, sample: dict[str, Any], resource_root: Path | None = None) -> None:
        base = resource_root or self.project_root
        for index, resource in enumerate(sample["input"]["images"]):
            storage = resource["storage"]
            path = Path(resource["path"])
            resolved = path if path.is_absolute() else base / path
            location = f"input.images[{index}]"
            if storage == "file":
                if not resolved.is_file():
                    raise ValidationError(f"Sample {sample['id']} {location}: file not found: {resolved}")
            elif storage == "zip":
                if not resolved.is_file():
                    raise ValidationError(f"Sample {sample['id']} {location}: ZIP not found: {resolved}")
                try:
                    with zipfile.ZipFile(resolved) as archive:
                        if resource["member"] not in archive.namelist():
                            raise ValidationError(f"Sample {sample['id']} {location}: ZIP member not found: {resource['member']}")
                except zipfile.BadZipFile as exc:
                    raise ValidationError(f"Sample {sample['id']} {location}: invalid ZIP: {resolved}") from exc
            elif storage == "parquet":
                if not resolved.is_file():
                    raise ValidationError(f"Sample {sample['id']} {location}: Parquet not found: {resolved}")
                try:
                    import pyarrow.parquet as parquet
                except ImportError as exc:
                    raise ValidationError("Parquet resource validation requires pyarrow") from exc
                cache_key = str(resolved)
                metadata = self._parquet_metadata.get(cache_key)
                if metadata is None:
                    parquet_file = parquet.ParquetFile(resolved)
                    metadata = (parquet_file.metadata.num_rows, set(parquet_file.schema_arrow.names))
                    self._parquet_metadata[cache_key] = metadata
                row_count, fields = metadata
                if resource["field"] not in fields:
                    raise ValidationError(f"Sample {sample['id']} {location}: Parquet field not found: {resource['field']}")
                if resource["row"] >= row_count:
                    raise ValidationError(f"Sample {sample['id']} {location}: row {resource['row']} out of range")
