from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.renderers.table_renderer import table_hash


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _json_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _path_image_id(raw: dict[str, Any], sample: dict[str, Any]) -> str:
    value = raw.get("image_id")
    if value:
        return str(value)
    image_path = sample.get("input", {}).get("images", [{}])[0].get("path", "")
    return Path(str(image_path)).stem


_ANSWER_OBJECT_RE = re.compile(r"\{\s*[\"']answer[\"']\s*:\s*\[[^{}]*\]\s*\}")


def _answer_values(sample: dict[str, Any]) -> tuple[set[str], str]:
    evaluation = sample.get("annotations", {}).get("evaluation", {})
    answer_list = evaluation.get("answer_list")
    if isinstance(answer_list, list):
        return {normalize_text(_json_text(value)) for value in answer_list}, "evaluation.answer_list"
    output_text = str(sample.get("output", {}).get("text", ""))
    for candidate in reversed(_ANSWER_OBJECT_RE.findall(output_text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), list):
            return {normalize_text(_json_text(value)) for value in parsed["answer"]}, "output.final_answer_json"
    return set(), "unavailable"


@dataclass
class Record:
    dataset: str
    sample_id: str
    source_id: str
    split: str
    subset: str | None
    task: str
    question: str
    answers: set[str]
    table_hash: str | None = None
    image_id: str = ""
    table_hash_method: str = "unresolved"
    raw_sample: dict[str, Any] = field(default_factory=dict)
    answer_source: str = "unavailable"


def load_tatqa(processed_root: Path) -> tuple[list[Record], dict[str, list[Record]], dict[str, list[Record]]]:
    records: list[Record] = []
    by_hash: dict[str, list[Record]] = defaultdict(list)
    by_prefix: dict[str, list[Record]] = defaultdict(list)
    for path in sorted(processed_root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                raw_table = sample.get("annotations", {}).get("raw", {}).get("table", {})
                rows = raw_table.get("table") if isinstance(raw_table, dict) else None
                if not isinstance(rows, list):
                    raise ValueError(f"TAT-QA sample has no structured raw table: {sample.get('id')}")
                digest = table_hash(rows)
                record = Record(
                    dataset="TAT-QA",
                    sample_id=str(sample["id"]),
                    source_id=str(sample.get("meta", {}).get("source_id", "")),
                    split=str(sample["split"]),
                    subset=sample.get("meta", {}).get("subset"),
                    task=str(sample.get("task", "")),
                    question=normalize_text(sample.get("input", {}).get("prompt", "")),
                    answers={normalize_text(value) for value in sample.get("output", {}).get("references", [])},
                    table_hash=digest,
                    answer_source="canonical.references",
                    image_id=str(raw_table.get("uid", "")) if isinstance(raw_table, dict) else "",
                    table_hash_method="canonical_raw_table",
                    raw_sample=sample,
                )
                records.append(record)
                by_hash[digest].append(record)
                by_prefix[record.question[:50]].append(record)
    return records, by_hash, by_prefix

def load_mmtab_eval_table_hashes(source_root: Path) -> dict[str, str]:
    path = source_root / "MMTab-eval_test_tables_23K.json"
    if not path.is_file():
        raise FileNotFoundError(f"MMTab structured table file not found: {path}")
    result: dict[str, str] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        if item.get("dataset_name") != "TAT-QA":
            continue
        rows = item.get("table_rows")
        if not isinstance(rows, list):
            continue
        result[str(item["image_id"])] = table_hash(rows)
    return result


def _question_candidates(prompt: str, by_prefix: dict[str, list[Record]]) -> list[Record]:
    normalized = normalize_text(prompt)
    candidates: dict[tuple[str, str], Record] = {}
    for index in range(max(0, len(normalized) - 50) + 1):
        for record in by_prefix.get(normalized[index:index + 50], []):
            if record.question and record.question in normalized:
                candidates[(record.sample_id, record.table_hash or "")] = record
    return list(candidates.values())


def load_mmtab(
    processed_root: Path,
    eval_hashes: dict[str, str],
    tatqa_prefix: dict[str, list[Record]],
) -> list[Record]:
    records: list[Record] = []
    pending_by_image: dict[str, list[Record]] = defaultdict(list)

    for path in sorted(processed_root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                supervision = sample.get("annotations", {}).get("supervision", {})
                raw = sample.get("annotations", {}).get("raw", {})
                dataset_name = str(supervision.get("dataset_name") or raw.get("dataset_name") or "")
                image_id = _path_image_id(raw, sample)
                is_tatqa = dataset_name.casefold() == "tat-qa" or image_id.casefold().startswith("tat-qa_")
                if not is_tatqa:
                    continue
                candidates = _question_candidates(sample.get("input", {}).get("prompt", ""), tatqa_prefix)
                answer_values, answer_source = _answer_values(sample)
                record = Record(
                    dataset="MMTab",
                    sample_id=str(sample["id"]),
                    source_id=str(sample.get("meta", {}).get("source_id", sample["id"])),
                    split=str(sample["split"]),
                    subset=sample.get("meta", {}).get("subset"),
                    task=str(sample.get("task", "")),
                    question=normalize_text(sample.get("input", {}).get("prompt", "")),
                    answers=answer_values,
                    image_id=image_id,
                    answer_source=answer_source,
                    raw_sample=sample,
                )
                records.append(record)
                pending_by_image[image_id].extend(candidates)

    image_hashes: dict[str, set[str]] = defaultdict(set)
    for image_id, candidates in pending_by_image.items():
        image_hashes[image_id].update(record.table_hash for record in candidates if record.table_hash)

    for record in records:
        if record.split == "test" and record.image_id in eval_hashes:
            record.table_hash = eval_hashes[record.image_id]
            record.table_hash_method = "mmtab_eval_structured_table"
        elif len(image_hashes[record.image_id]) == 1:
            record.table_hash = next(iter(image_hashes[record.image_id]))
            record.table_hash_method = "tatqa_question_anchor"

    return records

def _sorted_unique(values: set[str] | list[str]) -> list[str]:
    return sorted(set(values))


def _has_dataset_name_marker(record: Record) -> bool:
    supervision = record.raw_sample.get("annotations", {}).get("supervision", {})
    raw = record.raw_sample.get("annotations", {}).get("raw", {})
    value = supervision.get("dataset_name") or raw.get("dataset_name")
    return str(value).casefold() == "tat-qa"


def _record_dict(record: Record) -> dict[str, Any]:
    return {
        "dataset": record.dataset,
        "sample_id": record.sample_id,
        "source_id": record.source_id,
        "split": record.split,
        "subset": record.subset,
        "task": record.task,
        "image_id": record.image_id,
        "table_hash": record.table_hash,
        "table_hash_method": record.table_hash_method,
    }


def run(project_root: Path) -> dict[str, Any]:
    tatqa_root = project_root / "processed" / "tatqa"
    mmtab_root = project_root / "processed" / "mmtab"
    output_root = project_root / "tmp" / "mmtab_tatqa_overlap"
    output_root.mkdir(parents=True, exist_ok=True)

    tatqa_records, tatqa_by_hash, tatqa_prefix = load_tatqa(tatqa_root)
    eval_hashes = load_mmtab_eval_table_hashes(project_root.parent / "datasets" / "MMTab")
    mmtab_records = load_mmtab(mmtab_root, eval_hashes, tatqa_prefix)

    mmtab_by_hash: dict[str, list[Record]] = defaultdict(list)
    for record in mmtab_records:
        if record.table_hash:
            mmtab_by_hash[record.table_hash].append(record)
    overlap_hashes = sorted(set(mmtab_by_hash) & set(tatqa_by_hash))

    tables: list[dict[str, Any]] = []
    for digest in overlap_hashes:
        m_records = mmtab_by_hash[digest]
        t_records = tatqa_by_hash[digest]
        tables.append({
            "table_hash": digest,
            "mmtab_sample_ids": _sorted_unique([r.sample_id for r in m_records]),
            "tatqa_sample_ids": _sorted_unique([r.sample_id for r in t_records]),
            "mmtab_splits": _sorted_unique([r.split for r in m_records]),
            "tatqa_splits": _sorted_unique([r.split for r in t_records]),
            "mmtab_subsets": _sorted_unique([str(r.subset) for r in m_records if r.subset is not None]),
            "tatqa_answer_types": _sorted_unique([
                str(r.raw_sample.get("annotations", {}).get("evaluation", {}).get("answer_type", ""))
                for r in t_records
            ]),
            "mmtab_hash_methods": _sorted_unique([r.table_hash_method for r in m_records]),
        })

    qa_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for m_record in mmtab_records:
        if not m_record.table_hash or m_record.table_hash not in tatqa_by_hash:
            continue
        candidates = _question_candidates(m_record.raw_sample.get("input", {}).get("prompt", ""), tatqa_prefix)
        candidates = [r for r in candidates if r.table_hash == m_record.table_hash]
        for t_record in candidates:
            key = (m_record.table_hash, t_record.question)
            group = qa_groups.setdefault(key, {
                "table_hash": m_record.table_hash,
                "normalized_question": t_record.question,
                "mmtab_sample_ids": set(),
                "tatqa_sample_ids": set(),
                "mmtab_splits": set(),
                "tatqa_splits": set(),
                "mmtab_subsets": set(),
                "same_answer": False,
                "answer_comparable": False,
                "answer_comparisons": [],
            })
            group["mmtab_sample_ids"].add(m_record.sample_id)
            group["tatqa_sample_ids"].add(t_record.sample_id)
            group["mmtab_splits"].add(m_record.split)
            group["tatqa_splits"].add(t_record.split)
            if m_record.subset is not None:
                group["mmtab_subsets"].add(str(m_record.subset))
            comparable_answer = bool(m_record.answers and t_record.answers)
            exact_answer = comparable_answer and m_record.answers == t_record.answers
            group["answer_comparable"] = group["answer_comparable"] or comparable_answer
            group["same_answer"] = group["same_answer"] or exact_answer
            group["answer_comparisons"].append({
                "mmtab_sample_id": m_record.sample_id,
                "tatqa_sample_id": t_record.sample_id,
                "mmtab_answers": sorted(m_record.answers),
                "tatqa_answers": sorted(t_record.answers),
                "mmtab_answer_source": m_record.answer_source,
                "tatqa_answer_source": t_record.answer_source,
                "answer_comparable": comparable_answer,
                "same_answer": exact_answer,
            })

    qa_output: list[dict[str, Any]] = []
    for group in sorted(qa_groups.values(), key=lambda item: (item["table_hash"], item["normalized_question"])):
        qa_output.append({
            "table_hash": group["table_hash"],
            "normalized_question": group["normalized_question"],
            "mmtab_sample_ids": sorted(group["mmtab_sample_ids"]),
            "tatqa_sample_ids": sorted(group["tatqa_sample_ids"]),
            "mmtab_splits": sorted(group["mmtab_splits"]),
            "tatqa_splits": sorted(group["tatqa_splits"]),
            "mmtab_subsets": sorted(group["mmtab_subsets"]),
            "answer_comparable": group["answer_comparable"],
            "same_answer": group["same_answer"],
            "answer_comparisons": group["answer_comparisons"],
        })

    table_hashes_with_qa = {item["table_hash"] for item in qa_output}
    different_question_tables = []
    for digest in overlap_hashes:
        m_questions = {r.question for r in mmtab_by_hash[digest]}
        t_questions = {r.question for r in tatqa_by_hash[digest]}
        if any(question not in t_questions for question in m_questions) or any(question not in m_questions for question in t_questions):
            different_question_tables.append(digest)

    leakage: list[dict[str, Any]] = []
    risk_counts = Counter()

    def add_leakage(item: dict[str, Any], key: tuple[Any, ...]) -> None:
        if not any(existing.get("_key") == list(key) for existing in leakage):
            item["_key"] = list(key)
            leakage.append(item)

    for digest in overlap_hashes:
        m_records = mmtab_by_hash[digest]
        t_records = tatqa_by_hash[digest]
        m_train = [r for r in m_records if r.split == "train"]
        m_eval = [r for r in m_records if r.split != "train"]
        t_train = [r for r in t_records if r.split == "train"]
        t_eval = [r for r in t_records if r.split != "train"]

        for train_side, eval_side in ((m_train, t_eval), (t_train, m_eval)):
            if train_side and eval_side:
                risk_counts["P0"] += 1
                add_leakage({
                    "level": "table",
                    "risk": "P0",
                    "table_hash": digest,
                    "train_side": {
                        "dataset": train_side[0].dataset,
                        "split": train_side[0].split,
                        "sample_ids": sorted(r.sample_id for r in train_side),
                    },
                    "eval_side": {
                        "dataset": eval_side[0].dataset,
                        "split": eval_side[0].split,
                        "sample_ids": sorted(r.sample_id for r in eval_side),
                    },
                }, ("table", "P0", digest, train_side[0].dataset, eval_side[0].dataset))

        qa_for_table = [q for q in qa_output if q["table_hash"] == digest]
        for qa in qa_for_table:
            m_train_qa = any(split == "train" for split in qa["mmtab_splits"])
            m_eval_qa = any(split != "train" for split in qa["mmtab_splits"])
            t_train_qa = any(split == "train" for split in qa["tatqa_splits"])
            t_eval_qa = any(split != "train" for split in qa["tatqa_splits"])
            for train_dataset, eval_dataset, train_ids, eval_ids in (
                ("MMTab", "TAT-QA", qa["mmtab_sample_ids"], qa["tatqa_sample_ids"]),
            ):
                if m_train_qa and t_eval_qa:
                    risk_counts["P0"] += 1
                    add_leakage({
                        "level": "qa",
                        "risk": "P0",
                        "table_hash": digest,
                        "normalized_question": qa["normalized_question"],
                        "train_side": {"dataset": train_dataset, "split": "train", "sample_ids": train_ids},
                        "eval_side": {"dataset": eval_dataset, "split": "validation", "sample_ids": eval_ids},
                    }, ("qa", "P0", digest, qa["normalized_question"], "MMTab-train-TATQA-eval"))
                if t_train_qa and m_eval_qa:
                    risk_counts["P0"] += 1
                    add_leakage({
                        "level": "qa",
                        "risk": "P0",
                        "table_hash": digest,
                        "normalized_question": qa["normalized_question"],
                        "train_side": {"dataset": eval_dataset, "split": "train", "sample_ids": eval_ids},
                        "eval_side": {"dataset": train_dataset, "split": "test", "sample_ids": train_ids},
                    }, ("qa", "P0", digest, qa["normalized_question"], "TATQA-train-MMTab-eval"))

            if m_train_qa and t_train_qa:
                risk_counts["P1"] += 1
                add_leakage({
                    "level": "qa",
                    "risk": "P1",
                    "table_hash": digest,
                    "normalized_question": qa["normalized_question"],
                    "train_side": {"dataset": "MMTab", "split": "train", "sample_ids": qa["mmtab_sample_ids"]},
                    "comparison_side": {"dataset": "TAT-QA", "split": "train", "sample_ids": qa["tatqa_sample_ids"]},
                }, ("qa", "P1", digest, qa["normalized_question"]))

        if m_train and t_train:
            has_different_train_question = any(
                m_record.question != t_record.question
                for m_record in m_train
                for t_record in t_train
            )
            if has_different_train_question:
                risk_counts["P2"] += 1
                add_leakage({
                    "level": "table",
                    "risk": "P2",
                    "table_hash": digest,
                    "question_relation": "different_normalized_question",
                    "train_side": {"dataset": "MMTab", "split": "train", "sample_ids": sorted(r.sample_id for r in m_train)},
                    "comparison_side": {"dataset": "TAT-QA", "split": "train", "sample_ids": sorted(r.sample_id for r in t_train)},
                }, ("table", "P2", digest))

    for item in leakage:
        item.pop("_key", None)

    source_counts = Counter((r.split, r.task, r.subset) for r in mmtab_records)
    dataset_name_marker_samples = sum(1 for r in mmtab_records if _has_dataset_name_marker(r))
    image_marker_samples = sum(1 for r in mmtab_records if r.image_id.casefold().startswith("tat-qa_"))
    source_images = defaultdict(list)
    for record in mmtab_records:
        source_images[record.image_id].append(record)
    resolved_samples = Counter(r.split for r in mmtab_records if r.table_hash)
    resolved_images = Counter(r.split for image_id, rs in source_images.items() if any(r.table_hash for r in rs) for r in [rs[0]])
    overlap_samples = Counter(r.split for digest in overlap_hashes for r in mmtab_by_hash[digest])
    overlap_tat_samples = sum(len(tatqa_by_hash[digest]) for digest in overlap_hashes)
    unresolved_source_samples = sum(1 for r in mmtab_records if r.table_hash not in set(overlap_hashes))

    summary = {
        "method": {
            "text_normalization": "strip + collapse whitespace + lowercase",
            "table_hash": "src.renderers.table_renderer.table_hash(normalize_table(table))",
            "mmtab_train_table_source": "unique exact normalized question anchor to TAT-QA raw table",
            "mmtab_test_table_source": "MMTab-eval_test_tables_23K.json table_rows",
            "unresolved_records_are_not_counted_as_table_overlap": True,
        },
        "source_markers": {
            "mmtab_tatqa_samples_with_any_explicit_marker": len(mmtab_records),
            "mmtab_samples_dataset_name_exact_tatqa": dataset_name_marker_samples,
            "mmtab_samples_image_id_or_path_tatqa_marker": image_marker_samples,
            "mmtab_tatqa_unique_source_images": len(source_images),
            "by_split_task_subset": {
                f"{split}|{task}|{subset}": count
                for (split, task, subset), count in sorted(source_counts.items(), key=lambda item: str(item[0]))
            },
            "table_hash_recovered_samples": dict(sorted(resolved_samples.items())),
            "table_hash_recovered_unique_images": dict(sorted(resolved_images.items())),
            "unresolved_table_hash_samples": len(mmtab_records) - sum(resolved_samples.values()),
        },
        "table_hash_overlap": {
            "tatqa_unique_tables": len(tatqa_by_hash),
            "mmtab_unique_mapped_tables": len(mmtab_by_hash),
            "overlap_unique_tables": len(overlap_hashes),
            "mmtab_affected_samples": sum(overlap_samples.values()),
            "mmtab_affected_samples_by_split": dict(sorted(overlap_samples.items())),
            "tatqa_affected_samples": overlap_tat_samples,
        },
        "question_answer_overlap": {
            "same_table_same_question_unique_pairs": len(qa_output),
            "same_table_same_question_answer_comparable_unique_pairs": sum(1 for item in qa_output if item["answer_comparable"]),
            "same_table_same_question_same_answer_unique_pairs": sum(1 for item in qa_output if item["same_answer"]),
            "same_table_different_question_unique_tables": len(different_question_tables),
            "same_table_different_question_sample_pairs": sum(
                1 for digest in different_question_tables
                for m_record in mmtab_by_hash[digest]
                for t_record in tatqa_by_hash[digest]
                if m_record.question != t_record.question
            ),
        },
        "leakage": {
            "risk_counts": dict(sorted(risk_counts.items())),
            "confirmed_p0_only_over_resolved_table_hashes": True,
            "unresolved_mmtab_train_samples_outside_table_hash_check": len(mmtab_records) - sum(resolved_samples.values()),
            "table_level_p0_cases": sum(1 for item in leakage if item["level"] == "table" and item["risk"] == "P0"),
            "qa_level_p0_cases": sum(1 for item in leakage if item["level"] == "qa" and item["risk"] == "P0"),
            "p0_affected_mmtab_samples": len({sample_id for item in leakage if item["risk"] == "P0" for side in (item["train_side"], item["eval_side"]) if side["dataset"] == "MMTab" for sample_id in side["sample_ids"]}),
            "p0_affected_tatqa_samples": len({sample_id for item in leakage if item["risk"] == "P0" for side in (item["train_side"], item["eval_side"]) if side["dataset"] == "TAT-QA" for sample_id in side["sample_ids"]}),
        },
        "conclusion": (
            "存在评测泄漏，需要处理"
            if risk_counts.get("P0", 0)
            else "存在训练重复，但无已确认的评测泄漏；仍有 MMTab train 样本无法完成结构化 table_hash 判定"
            if risk_counts.get("P1", 0) or risk_counts.get("P2", 0)
            else "无已确认风险；仍有 MMTab train 样本无法完成结构化 table_hash 判定"
        ),
        "notes": [
            "MMTab processed train records do not retain structured table rows; unresolved train images are excluded from strict table_hash overlap.",
            "MMTab test includes table-level source data in MMTab-eval_test_tables_23K.json and is checked directly.",
            "No files under processed/ or datasets/ are modified by this check.",
        ],
    }

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    write_jsonl(output_root / "overlapping_tables.jsonl", tables)
    write_jsonl(output_root / "overlapping_qa.jsonl", qa_output)
    write_jsonl(output_root / "leakage_cases.jsonl", leakage)
    (output_root / "overlap_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# MMTab ↔ TAT-QA lineage / leakage check

## 1. 来源标识

- MMTab 中存在明确 TAT-QA 来源标记：{len(mmtab_records)} 条，{len(source_images)} 个 source image；其中 `dataset_name == TAT-QA` 的样本为 {dataset_name_marker_samples} 条。
- split/task/subset 分布见 `overlap_summary.json`。
- 严格恢复 table_hash 的 MMTab 样本：{sum(resolved_samples.values())} 条；未能确认 table_hash 的样本不计入 table overlap。

## 2. Table hash overlap

- TAT-QA unique table：{len(tatqa_by_hash)}。
- MMTab 可映射 unique table：{len(mmtab_by_hash)}。
- 重复 unique table：{len(overlap_hashes)}。
- 受影响 MMTab 样本：{sum(overlap_samples.values())}。
- 受影响 TAT-QA 样本：{overlap_tat_samples}。

## 3. Question / Answer overlap

- same table + same question：{len(qa_output)} 个规范化 QA 对。
- same table + same question + same answer：{sum(1 for item in qa_output if item['same_answer'])} 个规范化 QA 对（其中可严格比较答案的 QA 对为 {sum(1 for item in qa_output if item['answer_comparable'])} 个）。
- same table + different question：{len(different_question_tables)} 个 table；样本对数量 {summary['question_answer_overlap']['same_table_different_question_sample_pairs']}。

问题归一化仅使用 strip、连续空白归一和小写；没有使用 embedding 或语义判断。

## 4. Split leakage

- P0 table-level cases：{summary['leakage']['table_level_p0_cases']}。
- P0 QA-level cases：{summary['leakage']['qa_level_p0_cases']}。
- P0 影响 MMTab 样本：{summary['leakage']['p0_affected_mmtab_samples']}。
- P0 影响 TAT-QA 样本：{summary['leakage']['p0_affected_tatqa_samples']}。
- 风险分布：{json.dumps(summary['leakage']['risk_counts'], ensure_ascii=False)}。

## 5. 结论

**{summary['conclusion']}**

P0 定义为跨数据集 train ↔ validation/test 的严格 table_hash 或 QA 重叠。当前 P0 数字只覆盖已恢复 table_hash 的 MMTab 样本；仍有 {len(mmtab_records) - sum(resolved_samples.values())} 条 MMTab train 样本缺少结构化 table 或唯一 question anchor，不能据此证明这些样本不存在潜在 P0。当前结果建议后续优先按 `table_hash` 做跨数据集 lineage/去重，再在同表内部按规范化 question + answer 做 QA 级处理；本轮没有自动删除或修改任何样本。

## 6. 输出文件

- `overlap_summary.json`
- `overlapping_tables.jsonl`
- `overlapping_qa.jsonl`
- `leakage_cases.jsonl`
"""
    (output_root / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MMTab and TAT-QA table lineage and split leakage")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    summary = run(args.project_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
