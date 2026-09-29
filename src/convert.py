from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from src.converters.chartqa import ChartQAConverter
from src.converters.mmtab import MMTabConverter
from src.converters.docvqa import DocVQAConverter
from src.converters.finqa import FinQAConverter
from src.converters.tatqa import TATQAConverter
from src.converters.xfund import XFUNDConverter
from src.converters.doc2edag import Doc2EDAGConverter
from src.converters.govreport import GovReportConverter
from src.core.config import ConfigError, get_runtime_paths, load_dataset_config
from src.core.stats import Stats
from src.core.validator import SchemaValidator, ValidationError
from src.core.writer import JsonlWriter


CONVERTERS = {
    "docvqa": DocVQAConverter,
    "chartqa": ChartQAConverter,
    "mmtab": MMTabConverter,
    "finqa": FinQAConverter,
    "tatqa": TATQAConverter,
    "xfund": XFUNDConverter,
    "doc2edag": Doc2EDAGConverter,
    "govreport": GovReportConverter,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert datasets to Canonical JSONL")
    parser.add_argument("--config", required=True, help="Dataset YAML config path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum raw samples per split")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the configured output directory before conversion",
    )
    return parser.parse_args()


def _output_path(config: dict[str, Any], split: str) -> Path:
    paths = config.get("files", {}).get(split, [])
    if not paths:
        raise ValueError(f"No output file configured for split: {split}")
    return Path(paths[0])


def run(config_path: str, limit: int | None = None, clean_output: bool = False) -> int:
    runtime_paths = get_runtime_paths()
    config = load_dataset_config(config_path, runtime_paths=runtime_paths)
    root = runtime_paths.project_root
    validator = SchemaValidator(root)
    validator.validate_config(config)

    converter_name = config["conversion"]["converter"]
    converter_class = CONVERTERS.get(converter_name)
    if converter_class is None:
        raise ValueError(f"Unsupported converter {converter_name!r}")

    first_output = _output_path(config, "train")
    output_root = first_output.parent
    if clean_output and output_root.exists():
        if output_root == root or output_root == runtime_paths.processed_root:
            raise ValueError(f"Refusing to clean unsafe output directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if converter_class in (DocVQAConverter, ChartQAConverter, MMTabConverter, FinQAConverter, TATQAConverter, XFUNDConverter):
        converter = converter_class(config, root, output_root / "media")
    else:
        converter = converter_class(config, root)
    stats = Stats(config["dataset"]["name"])
    failures: list[dict[str, str]] = []
    summary: dict[str, dict[str, Any]] = {}

    for split in ("train", "validation", "test"):
        if not config.get("files", {}).get(split):
            continue
        output = _output_path(config, split)
        read_count = success_count = failure_count = 0
        prepare_split = getattr(converter, "prepare_split", None)
        if prepare_split is not None:
            prepare_split(split, limit=limit)
        with JsonlWriter(output) as writer:
            samples = iter(converter.iter_samples(split))
            while limit is None or read_count < limit:
                try:
                    sample = next(samples)
                except StopIteration:
                    break
                read_count += 1
                try:
                    validator.validate_sample(sample, check_resources=True, resource_root=output_root)
                    writer.write(sample)
                    stats.update(sample)
                    success_count += 1
                except (ValidationError, ValueError, OSError, RuntimeError) as exc:
                    failure_count += 1
                    failure = {
                        "split": split,
                        "sample_id": str(sample.get("id", "<missing>")),
                        "error": str(exc),
                    }
                    failures.append(failure)
                    print(
                        f"ERROR split={split} sample={failure['sample_id']}: {failure['error']}",
                        file=sys.stderr,
                    )
        summary[split] = {
            "read": read_count,
            "success": success_count,
            "failed": failure_count,
            "output": str(output),
        }

    close_converter = getattr(converter, "close", None)
    if close_converter is not None:
        close_converter()
    stats_path = output_root / "stats.json"
    stats.write(stats_path)
    print("Conversion summary:")
    for split, result in summary.items():
        print(
            f"  {split}: read={result['read']} success={result['success']} "
            f"failed={result['failed']} output={result['output']}"
        )
    print(f"  stats: {stats_path}")
    if failures:
        print(f"Conversion completed with {len(failures)} failure(s).", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return run(args.config, args.limit, args.clean_output)
    except (ConfigError, ValidationError, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
