from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


class ConfigError(ValueError):
    """Raised when a dataset config cannot be loaded or validated."""


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    datasets_root: Path
    processed_root: Path

    def variables(self) -> dict[str, str]:
        return {
            "PROJECT_ROOT": str(self.project_root),
            "DATASETS_ROOT": str(self.datasets_root),
            "PROCESSED_ROOT": str(self.processed_root),
        }


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_runtime_paths(project_root: Path | None = None) -> RuntimePaths:
    root_value = os.environ.get("PROJECT_ROOT")
    root = Path(root_value).expanduser().resolve() if root_value else (project_root or default_project_root()).resolve()
    datasets = Path(os.environ.get("DATASETS_ROOT", root.parent / "datasets")).expanduser().resolve()
    processed = Path(os.environ.get("PROCESSED_ROOT", root / "processed")).expanduser().resolve()
    return RuntimePaths(root, datasets, processed)


def resolve_project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_variables(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                raise ConfigError(f"Unknown config path variable: ${{{name}}}")
            return variables[name]
        return _VARIABLE.sub(replace, value)
    if isinstance(value, list):
        return [_expand_variables(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_variables(item, variables) for key, item in value.items()}
    return value


def _error_path(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return ".".join(parts) if parts else "$"


def _validate_config_object(config: Any, schema_path: Path) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Dataset Config Schema not found: {schema_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid Dataset Config Schema JSON: {schema_path}: {exc}") from exc

    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(f"{_error_path(error)}: {error.message}" for error in errors)
        raise ConfigError(f"Invalid dataset config: {details}")


def load_dataset_config(
    config_path: str | Path,
    project_root: Path | None = None,
    schema_path: str | Path | None = None,
    runtime_paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    paths = runtime_paths or get_runtime_paths(project_root)
    path = resolve_project_path(config_path, paths.project_root)
    if not path.is_file():
        raise ConfigError(f"Dataset config file not found: {path}")

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError(f"Dataset config must be a YAML object: {path}")

    config = _expand_variables(config, paths.variables())
    schema = resolve_project_path(schema_path or "assets/dataset_config.schema.json", paths.project_root)
    _validate_config_object(config, schema)
    return config
