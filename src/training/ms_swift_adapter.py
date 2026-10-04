"""Stream Canonical splits into separate ms-swift train and labeled eval JSONL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from .manifest import PROJECT_ROOT, write_json


IMAGE_TOKEN = "<image>"


class AdapterError(ValueError):
    """A Canonical sample cannot be safely represented as ms-swift data."""


def load_manifest(path: Path, project_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != "1.0" or not isinstance(manifest.get("datasets"), list):
        raise ValueError(f"Invalid training manifest: {path}")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in manifest["datasets"]:
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"Duplicate or invalid dataset name in manifest: {name!r}")
        seen.add(name)
        if entry.get("enabled") is False:
            continue
        if entry.get("enabled") is not True:
            raise ValueError(f"{name}: enabled must be a boolean")
        dataset_dir = (project_root / "processed" / name).resolve()
        train_path = (project_root / entry["path"]).resolve()
        if train_path != dataset_dir / "train.jsonl" or not train_path.is_file():
            raise ValueError(f"{name}: manifest must point to an existing processed/<name>/train.jsonl")
        sources = entry.get("eval_sources")
        if not isinstance(sources, list):
            raise ValueError(f"{name}: manifest needs an eval_sources list; regenerate it with src.training.manifest")
        eval_sources = []
        used_splits: set[str] = set()
        for source in sources:
            split = source.get("split")
            if split not in ("validation", "test") or split in used_splits:
                raise ValueError(f"{name}: invalid or repeated eval split: {split!r}")
            used_splits.add(split)
            source_path = (project_root / source["path"]).resolve()
            if source_path != dataset_dir / f"{split}.jsonl" or not source_path.is_file():
                raise ValueError(f"{name}: invalid or missing {split} eval source")
            eval_sources.append({"split": split, "path": source_path})
        entries.append({"name": name, "train_path": train_path,
                        "dataset_dir": dataset_dir, "eval_sources": eval_sources})
    return entries


def skip_reason(sample: dict[str, Any]) -> str | None:
    output = sample.get("output", {})
    target = output.get("text")
    if not isinstance(target, str):
        raise AdapterError("output.text must be a string")
    if not target.strip():
        return "empty_target"
    if output.get("references") == []:
        return "unsupervised_no_references"
    annotations = sample.get("annotations", {})
    for section in (sample, annotations.get("supervision", {}), annotations.get("evaluation", {})):
        if not isinstance(section, dict):
            continue
        if (section.get("unsupervised") is True or section.get("is_supervised") is False
                or section.get("supervised") is False or section.get("has_answers") is False):
            return "marked_unsupervised"
    return None


def convert_sample(sample: dict[str, Any], dataset_dir: Path,
                   expected_split: str = "train") -> dict[str, Any]:
    if sample.get("split") != expected_split:
        raise AdapterError(f"expected {expected_split} split, got {sample.get('split')!r}")
    inp = sample.get("input")
    if not isinstance(inp, dict):
        raise AdapterError("input must be an object")
    context, prompt, history, image_resources = (
        inp.get("context"), inp.get("prompt"), inp.get("history"), inp.get("images")
    )
    if not isinstance(context, str) or not isinstance(prompt, str):
        raise AdapterError("input.context and input.prompt must be strings")
    if not isinstance(history, list) or not isinstance(image_resources, list):
        raise AdapterError("input.history and input.images must be arrays")
    if not isinstance(sample["output"]["text"], str) or not sample["output"]["text"].strip():
        raise AdapterError("final assistant content must be nonempty")

    messages: list[dict[str, str]] = []
    for index, item in enumerate(history):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(item, dict) or item.get("role") != expected_role or not isinstance(item.get("content"), str):
            raise AdapterError(f"history[{index}] must be a {expected_role} message with string content")
        if IMAGE_TOKEN in item["content"]:
            raise AdapterError(f"history[{index}] contains an image token without a historical image resource")
        messages.append({"role": expected_role, "content": item["content"]})
    if len(history) % 2:
        raise AdapterError("history must contain complete user/assistant pairs")

    images: list[str] = []
    for index, resource in enumerate(image_resources):
        if not isinstance(resource, dict) or resource.get("storage") != "file":
            raise AdapterError(f"input.images[{index}] must be a converted file image")
        raw_path = resource.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise AdapterError(f"input.images[{index}].path must be nonempty")
        file_path = Path(raw_path)
        resolved = (file_path if file_path.is_absolute() else dataset_dir / file_path).resolve()
        if not resolved.is_file():
            raise AdapterError(f"input.images[{index}] file not found: {resolved}")
        images.append(str(resolved))

    user_content = context + "\n\n" + prompt if context else prompt
    user_content = IMAGE_TOKEN * len(images) + user_content
    if not user_content:
        raise AdapterError("current user content is empty")
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": sample["output"]["text"]})
    output: dict[str, Any] = {"messages": messages}
    if images:
        output["images"] = images
    if sum(message["content"].count(IMAGE_TOKEN) for message in messages) != len(images):
        raise AdapterError("image token count does not match images count")
    return output


def new_counts() -> dict[str, Any]:
    return {
        "samples": 0, "input_samples": 0, "skipped": Counter(), "tasks": Counter(),
        "languages": Counter(), "image_counts": Counter(), "turn_counts": Counter(),
        "image_files_checked": 0,
    }


def stream_source(source_path: Path, split: str, dataset_dir: Path, name: str,
                  dataset_file: TextIO, aggregate_file: TextIO,
                  counts: dict[str, Any], limit: int | None) -> dict[str, Any]:
    source_counts = {"input_samples": 0, "samples": 0, "skipped": Counter()}
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if limit is not None and source_counts["input_samples"] >= limit:
                break
            source_counts["input_samples"] += 1
            counts["input_samples"] += 1
            sample_id = "<unknown>"
            try:
                sample = json.loads(line)
                sample_id = sample.get("id", "<missing>")
                if sample.get("split") != split:
                    raise AdapterError(f"expected {split} split, got {sample.get('split')!r}")
                reason = skip_reason(sample)
                if reason:
                    counts["skipped"][reason] += 1
                    source_counts["skipped"][reason] += 1
                    continue
                output = convert_sample(sample, dataset_dir, expected_split=split)
                task, language = sample["task"], sample["language"]
                if not isinstance(task, str) or not isinstance(language, str):
                    raise AdapterError("task and language must be strings")
            except (json.JSONDecodeError, KeyError, TypeError, AdapterError) as exc:
                raise AdapterError(
                    f"dataset={name} split={split} sample_id={sample_id} line={line_number}: {exc}"
                ) from exc
            encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
            dataset_file.write(encoded)
            aggregate_file.write(encoded)
            counts["samples"] += 1
            source_counts["samples"] += 1
            counts["tasks"][task] += 1
            counts["languages"][language] += 1
            image_count = len(output.get("images", []))
            counts["image_files_checked"] += image_count
            counts["image_counts"]["zero" if image_count == 0 else "single" if image_count == 1 else "multiple"] += 1
            counts["turn_counts"]["single" if len(output["messages"]) == 2 else "multiple"] += 1
    source_counts["skipped"] = dict(sorted(source_counts["skipped"].items()))
    return source_counts


def serial_counts(counts: dict[str, Any]) -> dict[str, Any]:
    total = counts["samples"]
    image_counts = counts["image_counts"]
    return {
        "samples": total,
        "input_samples": counts["input_samples"],
        "skipped_samples": sum(counts["skipped"].values()),
        "skip_reasons": dict(sorted(counts["skipped"].items())),
        "tasks": dict(sorted(counts["tasks"].items())),
        "languages": dict(sorted(counts["languages"].items())),
        "image_samples": total - image_counts["zero"],
        "no_image_samples": image_counts["zero"],
        "image_count_distribution": {key: image_counts[key] for key in ("zero", "single", "multiple")},
        "turn_distribution": {key: counts["turn_counts"][key] for key in ("single", "multiple")},
        "image_files_checked": counts["image_files_checked"],
        "missing_image_files": 0,
    }


def prepare(manifest_path: Path, output_dir: Path, project_root: Path = PROJECT_ROOT,
            limit: int | None = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    output_dir = output_dir.resolve()
    processed_root = (project_root / "processed").resolve()
    if output_dir == processed_root or processed_root in output_dir.parents:
        raise ValueError("ms-swift output directory must be outside processed/ to preserve Canonical JSONL")
    entries = load_manifest(manifest_path, project_root)
    if not entries:
        raise ValueError("Manifest contains no enabled train datasets")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    train_counts, eval_counts = new_counts(), new_counts()
    datasets: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=".ms-swift-stage-", dir=output_dir.parent) as stage_name:
        stage = Path(stage_name)
        (stage / "datasets").mkdir()
        with (stage / "train.jsonl").open("w", encoding="utf-8") as train_all, (
            stage / "eval.jsonl"
        ).open("w", encoding="utf-8") as eval_all:
            for entry in entries:
                name, dataset_dir = entry["name"], entry["dataset_dir"]
                train_file = stage / "datasets" / f"{name}.train.jsonl"
                eval_file = stage / "datasets" / f"{name}.eval.jsonl"
                with train_file.open("w", encoding="utf-8") as train_target:
                    train_result = stream_source(
                        entry["train_path"], "train", dataset_dir, name,
                        train_target, train_all, train_counts, limit
                    )
                eval_results = []
                with eval_file.open("w", encoding="utf-8") as eval_target:
                    for source in entry["eval_sources"]:
                        result = stream_source(
                            source["path"], source["split"], dataset_dir, name,
                            eval_target, eval_all, eval_counts, limit
                        )
                        eval_results.append({"split": source["split"], **result})
                datasets[name] = {
                    "train": {"path": f"datasets/{name}.train.jsonl", **train_result},
                    "eval": {"path": f"datasets/{name}.eval.jsonl",
                             "samples": sum(item["samples"] for item in eval_results),
                             "sources": eval_results},
                }

        train_total, eval_total = train_counts["samples"], eval_counts["samples"]
        if sum(item["train"]["samples"] for item in datasets.values()) != train_total:
            raise AdapterError("train line count does not equal sum of dataset train lines")
        if sum(item["eval"]["samples"] for item in datasets.values()) != eval_total:
            raise AdapterError("eval line count does not equal sum of dataset eval lines")
        stats = {
            "total_training_samples": train_total,
            "total_evaluation_samples": eval_total,
            "datasets": {
                name: {
                    **item,
                    "train_share": item["train"]["samples"] / train_total if train_total else 0.0,
                    "eval_share": item["eval"]["samples"] / eval_total if eval_total else 0.0,
                }
                for name, item in datasets.items()
            },
            "train": serial_counts(train_counts),
            "eval": serial_counts(eval_counts),
            "train_lines": train_total,
            "eval_lines": eval_total,
            "limit_per_source": limit,
            "eval_source_policy": "validation then test; only samples with nonempty answers and references",
        }
        write_json(stage / "stats.json", stats)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "datasets").mkdir(exist_ok=True)
        for entry in entries:
            name = entry["name"]
            for split in ("train", "eval"):
                filename = f"{name}.{split}.jsonl"
                os.replace(stage / "datasets" / filename, output_dir / "datasets" / filename)
        for filename in ("train.jsonl", "eval.jsonl", "stats.json"):
            os.replace(stage / filename, output_dir / filename)
        legacy = output_dir / "all_train.jsonl"
        if legacy.exists():
            legacy.unlink()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/train_manifest.json")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum input samples per dataset and source split")
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest_path = (args.manifest if args.manifest.is_absolute() else root / args.manifest).resolve()
    output_dir = (args.output_dir if args.output_dir is not None else root / "training/ms_swift").resolve()
    stats = prepare(manifest_path, output_dir, root, args.limit)
    print(f"Wrote train={stats['total_training_samples']} eval={stats['total_evaluation_samples']} "
          f"to {output_dir}; skipped train={stats['train']['skipped_samples']} "
          f"eval={stats['eval']['skipped_samples']}")


if __name__ == "__main__":
    main()
