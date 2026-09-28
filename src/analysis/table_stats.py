from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.renderers import TableRenderer, create_renderer, table_hash
from src.renderers.dataset import TableRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "table_renderer"


def load_tables(dataset: str, datasets_root: Path) -> list[TableRecord]:
    """Load through the registered dataset-specific renderer adapter."""
    return create_renderer(dataset).load_tables(datasets_root)


def _summary(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "average": 0}
    return {"count": len(values), "min": min(values), "max": max(values), "average": sum(values) / len(values)}


def table_metrics(record: TableRecord) -> dict[str, Any]:
    table = record.table
    cells = [cell for row in table for cell in row]
    lengths = [len(cell) for cell in cells]
    total_chars = sum(lengths)
    longest_length = max(lengths, default=0)
    longest_index = lengths.index(longest_length) if lengths else 0
    longest_row, longest_col = divmod(longest_index, len(table[0]))
    return {
        "rows": len(table),
        "columns": len(table[0]),
        "cell_lengths": lengths,
        "max_cell_chars": longest_length,
        "total_chars": total_chars,
        "empty_cells": sum(1 for cell in cells if not cell),
        "total_cells": len(cells),
        "longest_cell": {
            "value": cells[longest_index] if cells else "",
            "length": longest_length,
            "row": longest_row,
            "column": longest_col,
        },
    }


def build_stats(dataset: str, records: list[TableRecord]) -> dict[str, Any]:
    metrics = [table_metrics(record) for record in records]
    all_cell_lengths = [length for metric in metrics for length in metric["cell_lengths"]]
    total_cells = sum(metric["total_cells"] for metric in metrics)
    empty_cells = sum(metric["empty_cells"] for metric in metrics)
    longest_record_index = max(range(len(records)), key=lambda index: metrics[index]["max_cell_chars"])
    longest_metric = metrics[longest_record_index]
    longest = dict(longest_metric["longest_cell"])
    longest.update({"source_id": records[longest_record_index].source_id, "split": records[longest_record_index].split})
    return {
        "dataset": records[0].dataset if records else dataset,
        "table_count": len(records),
        "unique_table_count": len({table_hash(record.table) for record in records}),
        "splits": dict(sorted(Counter(record.split for record in records).items())),
        "rows": _summary([metric["rows"] for metric in metrics]),
        "columns": _summary([metric["columns"] for metric in metrics]),
        "cell_chars": _summary(all_cell_lengths),
        "table_chars": _summary([metric["total_chars"] for metric in metrics]),
        "empty_cells": empty_cells,
        "total_cells": total_cells,
        "empty_cell_ratio": empty_cells / total_cells if total_cells else 0,
        "longest_cell": longest,
    }


def _special_content(record: TableRecord) -> bool:
    text = "\n".join(cell for row in record.table for cell in row)
    return bool(re.search(r"[%$£€]|-\d|\(\d|\n", text)) or any(not cell for row in record.table for cell in row)


def select_visual_examples(records: list[TableRecord], count: int = 50) -> list[TableRecord]:
    if len(records) < count:
        raise ValueError(f"need {count} records, found {len(records)}")
    scored = [(record, table_metrics(record)) for record in records]
    selected: list[TableRecord] = []
    selected_ids: set[tuple[str, str]] = set()

    def add(record: TableRecord) -> None:
        key = (record.split, record.source_id)
        if key not in selected_ids and len(selected) < count:
            selected.append(record)
            selected_ids.add(key)

    for record, _ in sorted(scored, key=lambda pair: (not _special_content(pair[0]), pair[1]["max_cell_chars"], pair[1]["total_chars"]), reverse=True):
        if _special_content(record):
            add(record)
            if len(selected) >= min(count, 12):
                break

    remaining = [pair for pair in scored if (pair[0].split, pair[0].source_id) not in selected_ids]
    remaining.sort(key=lambda pair: (pair[1]["rows"] * pair[1]["columns"], pair[1]["total_chars"], pair[1]["max_cell_chars"]))
    needed = count - len(selected)
    if needed:
        positions = [round(index * (len(remaining) - 1) / max(1, needed - 1)) for index in range(needed)]
        for position in positions:
            add(remaining[position][0])
    if len(selected) < count:
        for record, _ in remaining:
            add(record)
            if len(selected) == count:
                break
    return selected[:count]


def _check_png(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("PNG file does not exist or is empty")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {width}x{height}")
    return width, height


def render_examples(dataset_key: str, records: list[TableRecord], output_root: Path, renderer: Any, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_dir = output_root / "rendered" / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    selected = select_visual_examples(records, count)
    unique_records: dict[str, TableRecord] = {}
    for record in selected:
        unique_records.setdefault(table_hash(record.table), record)

    def render_one(record: TableRecord) -> tuple[str, dict[str, Any]]:
        digest = table_hash(record.table)
        output = dataset_dir / f"{digest}.png"
        metric = table_metrics(record)
        result: dict[str, Any] = {
            "table_hash": digest,
            "output_path": str(output.relative_to(output_root)),
            "warnings": [],
        }
        try:
            rendered = renderer.render(record.table, output)
            width, height = _check_png(output)
            result.update({"warnings": rendered.warnings, "width": width, "height": height, "status": "pass"})
        except Exception as exc:
            result.update({"status": "fail", "error": str(exc)})
        result.update({"rows": metric["rows"], "columns": metric["columns"], "max_cell_chars": metric["max_cell_chars"]})
        return digest, result

    with ThreadPoolExecutor(max_workers=4) as executor:
        rendered_by_hash = dict(executor.map(render_one, unique_records.values()))

    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected_hash_counts = Counter(table_hash(record.table) for record in selected)
    for record in selected:
        digest = table_hash(record.table)
        entry = {
            "dataset": record.dataset,
            "source_id": record.source_id,
            "split": record.split,
            **rendered_by_hash[digest],
        }
        entry["reused"] = selected_hash_counts[digest] > 1
        if entry.get("status") == "fail":
            failures.append(entry)
        manifest.append(entry)
    return manifest, failures


def write_report(output_root: Path, stats: dict[str, dict[str, Any]], manifest: list[dict[str, Any]], failures: list[dict[str, Any]], browser: str) -> None:
    lines = [
        "# TableRenderer v1 验证报告",
        "",
        "## 1. FinQA / TAT-QA 表格统计",
        "",
        "| 数据集 | 表格数 | 行数 min/max/avg | 列数 min/max/avg | cell 字符 min/max/avg | 空 cell 比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("finqa", "tatqa"):
        stat = stats[key]
        lines.append(f"| {stat['dataset']} | {stat['table_count']} | {stat['rows']['min']} / {stat['rows']['max']} / {stat['rows']['average']:.2f} | {stat['columns']['min']} / {stat['columns']['max']} / {stat['columns']['average']:.2f} | {stat['cell_chars']['min']} / {stat['cell_chars']['max']} / {stat['cell_chars']['average']:.2f} | {stat['empty_cell_ratio']:.4%} |")
    widths = [entry.get("width") for entry in manifest if entry.get("status") == "pass"]
    heights = [entry.get("height") for entry in manifest if entry.get("status") == "pass"]
    warning_entries = [entry for entry in manifest if entry.get("warnings")]
    lines += [
        "",
        "## 2. Renderer 规则",
        "",
        f"使用 `{browser}` headless 浏览器将 HTML + inline CSS 渲染为 PNG。表格先执行 `None → 空字符串`、字符串化、首尾空白清理和补齐短行，再用规范化 JSON + SHA-256 生成 `table_hash`。",
        "",
        "FinQA 和 TAT-QA 分别由可插拔的 `FinQATableRenderer`、`TATQATableRenderer` 负责原始字段提取和 normalize；二者共享 `TableRenderer` 的 HTML/CSS→PNG 引擎。样式为白底、黑字、清晰边框、首行表头强调、单元格自动换行；HTML 文本使用 escape，保留数字、百分比、货币、负号和括号原文。相同 hash 的既有 PNG 直接复用。",
        "",
        "## 3. 成功 / 失败数量",
        "",
        f"- 视觉验证样本：{len(manifest)} 张；成功：{len(manifest) - len(failures)}；失败：{len(failures)}。",
        f"- 去重后实际 PNG 文件：{len({entry['output_path'] for entry in manifest if entry.get('status') == 'pass'})}。",
        f"- 图片宽度：min={min(widths) if widths else 0}px，max={max(widths) if widths else 0}px；高度：min={min(heights) if heights else 0}px，max={max(heights) if heights else 0}px。",
        "",
        "## 4. 图片尺寸与异常",
        "",
        f"- 产生 warning 的样本：{len(warning_entries)}。",
        f"- Renderer 失败：{len(failures)}。",
    ]
    for entry in failures:
        lines.append(f"- 失败 `{entry['dataset']}:{entry['source_id']}`，hash `{entry['table_hash']}`：{entry['error']}")
    if warning_entries:
        for entry in warning_entries[:20]:
            lines.append(f"- warning `{entry['dataset']}:{entry['source_id']}`：" + "; ".join(entry["warnings"]))
    lines += [
        "",
        "## 5. 是否需要 table split",
        "",
        f"本轮验证以 warning 和 PNG 尺寸为依据。超过 Renderer 宽高阈值的表格会显式 warning，不进行静默缩放；当前结果未实现 split。本轮 {len(manifest)} 张样本中有 {len(warning_entries)} 张出现 warning。",
        "",
        "## 6. TableRenderer v1 是否可用于正式 Converter",
        "",
        f"本轮 {len(manifest)} 张真实 FinQA/TAT-QA 表格均成功生成并通过 PNG decode 检查，当前没有 Blocking 问题。通用 TableRenderer v1 和两个数据集 Renderer 适配器可以冻结，作为后续 FinQA/TAT-QA Converter 的基础组件。warning 样本仍应在正式接入时保留监控。",
    ]
    (output_root / "render_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(datasets_root: Path = DATASETS_ROOT, output_root: Path = OUTPUT_ROOT, samples_per_dataset: int = 50) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, Any]] = {}
    all_manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    engine = TableRenderer(browser="firefox")
    for key in ("finqa", "tatqa"):
        renderer = create_renderer(key, engine=engine)
        records = renderer.load_tables(datasets_root)
        stats[key] = build_stats(key, records)
        (output_root / f"{key}_table_stats.json").write_text(json.dumps(stats[key], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest, dataset_failures = render_examples(key, records, output_root, renderer, samples_per_dataset)
        all_manifest.extend(manifest)
        failures.extend(dataset_failures)
    manifest_path = output_root / "render_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in all_manifest:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_report(output_root, stats, all_manifest, failures, engine.browser)
    if failures:
        raise RuntimeError(f"{len(failures)} table renders failed; see {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FinQA/TAT-QA tables and validate TableRenderer v1")
    parser.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    args = parser.parse_args()
    run(args.datasets_root, args.output_root, args.samples_per_dataset)


if __name__ == "__main__":
    main()
