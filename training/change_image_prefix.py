"""Replace a path prefix in ms-swift JSONL image arrays.

The script changes only the top-level "images" list. It preserves Canonical
files under processed/ and leaves unmatched JSONL lines byte-for-byte intact.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent


def normalize_prefix(value: str) -> str:
    prefix = value.rstrip("/")
    if not prefix or prefix == "/":
        raise ValueError("prefix must be a non-root path")
    return prefix


def replace_prefix(path: str, old_prefix: str, new_prefix: str) -> str:
    if path == old_prefix:
        return new_prefix
    if path.startswith(old_prefix + "/"):
        return new_prefix + path[len(old_prefix):]
    return path


def list_jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"input file must be JSONL: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input file or directory not found: {input_path}")
    files = sorted(input_path.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"no JSONL files found under {input_path}")
    return files


def check_not_processed(path: Path) -> None:
    processed = (PROJECT_ROOT / "processed").resolve()
    resolved = path.resolve()
    if resolved == processed or processed in resolved.parents:
        raise ValueError(f"refusing to write inside Canonical processed/: {path}")


def rewrite_file(
    source: Path,
    destination: Path | None,
    old_prefix: str,
    new_prefix: str,
) -> dict[str, Any]:
    rows = changed_rows = changed_images = 0
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = destination.open("w", encoding="utf-8", newline="")
    else:
        target = None
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, 1):
                rows += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL row must be an object")
                    images = row.get("images", [])
                    if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
                        raise ValueError("images must be an array of strings")
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
                updated = [replace_prefix(path, old_prefix, new_prefix) for path in images]
                count = sum(before != after for before, after in zip(images, updated))
                if count:
                    row["images"] = updated
                    changed_rows += 1
                    changed_images += count
                    if target is not None:
                        target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                elif target is not None:
                    target.write(line)
    finally:
        if target is not None:
            target.close()
    return {"rows": rows, "changed_rows": changed_rows, "changed_images": changed_images}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRAINING_ROOT / "ms_swift",
                        help="ms-swift JSONL file or directory; defaults to training/ms_swift")
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument("--dry-run", action="store_true", help="count changes without writing")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--in-place", action="store_true", help="replace input JSONL files after validation")
    mode.add_argument("--output-dir", type=Path, help="write a separate directory with the same file layout")
    args = parser.parse_args()
    if not args.dry_run and not args.in_place and args.output_dir is None:
        parser.error("specify --in-place, --output-dir, or --dry-run")

    old_prefix = normalize_prefix(args.old_prefix)
    new_prefix = normalize_prefix(args.new_prefix)
    if old_prefix == new_prefix:
        parser.error("old and new prefixes are identical")
    input_path = args.input.resolve()
    files = list_jsonl_files(input_path)
    base = input_path if input_path.is_dir() else input_path.parent
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    if output_dir is not None:
        check_not_processed(output_dir)
        if output_dir == base or base in output_dir.parents or output_dir in base.parents:
            parser.error("--output-dir must be separate from the input directory")
    if args.in_place:
        for source in files:
            check_not_processed(source)

    totals = {"rows": 0, "changed_rows": 0, "changed_images": 0}
    if args.dry_run:
        for source in files:
            result = rewrite_file(source, None, old_prefix, new_prefix)
            for key in totals:
                totals[key] += result[key]
            print(f"{source}: {result['changed_images']} image paths in {result['changed_rows']} rows")
    else:
        stage_parent = output_dir.parent if output_dir is not None else base.parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".image-prefix-stage-", dir=stage_parent) as stage_name:
            stage = Path(stage_name)
            changed_files: list[tuple[Path, Path]] = []
            for source in files:
                relative = source.relative_to(base)
                staged = stage / relative
                result = rewrite_file(source, staged, old_prefix, new_prefix)
                for key in totals:
                    totals[key] += result[key]
                if output_dir is not None or result["changed_images"]:
                    destination = (output_dir / relative) if output_dir is not None else source
                    changed_files.append((staged, destination))
                else:
                    staged.unlink()
                print(f"{source}: {result['changed_images']} image paths in {result['changed_rows']} rows")
            for staged, destination in changed_files:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, destination)
    print(f"Total: {totals['changed_images']} image paths changed across "
          f"{totals['changed_rows']} of {totals['rows']} rows")


if __name__ == "__main__":
    main()
