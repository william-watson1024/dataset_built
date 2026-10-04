"""Discover converted training splits and write a baseline training manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_manifest(project_root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    processed_root = project_root / "processed"
    if not processed_root.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {processed_root}")

    datasets: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []
    for directory in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        files = {name: (directory / name).exists() for name in
                 ("train.jsonl", "validation.jsonl", "test.jsonl", "stats.json", "media")}
        discovered.append({"name": directory.name, "files": files})
        train_file = directory / "train.jsonl"
        if not train_file.is_file():
            continue

        tasks: Counter[str] = Counter()
        count = 0
        with train_file.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    sample = json.loads(line)
                    if sample["split"] != "train" or not isinstance(sample["task"], str):
                        raise ValueError("expected a train sample with a string task")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{train_file}:{line_number}: {exc}") from exc
                tasks[sample["task"]] += 1
                count += 1

        eval_sources = [
            {"split": split, "path": (directory / f"{split}.jsonl").relative_to(project_root).as_posix()}
            for split in ("validation", "test")
            if (directory / f"{split}.jsonl").is_file()
        ]
        datasets.append({
            "name": directory.name,
            "path": train_file.relative_to(project_root).as_posix(),
            "enabled": True,
            "sample_count": count,
            "task_distribution": dict(sorted(tasks.items())),
            "eval_sources": eval_sources,
        })

    manifest = {"version": "1.0", "datasets": datasets}
    summary = {
        "version": "1.0",
        "discovered_datasets": discovered,
        "train_dataset_count": len(datasets),
        "train_sample_count": sum(item["sample_count"] for item in datasets),
        "datasets": [{"name": item["name"], "sample_count": item["sample_count"],
                      "task_distribution": item["task_distribution"],
                      "eval_sources": item["eval_sources"]} for item in datasets],
        "eval_selection": "Labeled validation and test samples, in that order; Canonical files remain unchanged.",
        "notes": (["MMTab contains TAT-QA-derived samples; baseline retains them without cross-dataset deduplication."]
                  if any(item["name"] == "mmtab" for item in datasets) else []),
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (args.output_dir or root / "manifests").resolve()
    manifest, summary = build_manifest(root)
    write_json(output_dir / "train_manifest.json", manifest)
    write_json(output_dir / "train_manifest_summary.json", summary)
    print(f"Wrote {len(manifest['datasets'])} datasets and {summary['train_sample_count']} train samples to {output_dir}")


if __name__ == "__main__":
    main()
