from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class Stats:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.total = 0
        self.splits: Counter[str] = Counter()
        self.tasks: Counter[str] = Counter()
        self.languages: Counter[str] = Counter()
        self.formats: Counter[str] = Counter()
        self.subsets: Counter[str] = Counter()
        self.image_counts: Counter[str] = Counter()
        self.image_paths: set[str] = set()
        self.group_ids: set[str] = set()
        self.total_references = 0
        self.program_samples = 0
        self.steps_samples = 0
        self.gold_evidence_samples = 0
        self.answer_types: Counter[str] = Counter()
        self.answer_froms: Counter[str] = Counter()
        self.scales: Counter[str] = Counter()
        self.derivation_samples = 0
        self.supervised = 0
        self.unsupervised = 0
        self.with_images = 0
        self.without_images = 0
        self.lengths = {"context_chars": [], "prompt_chars": [], "output_chars": []}
        self.entity_counts: list[int] = []
        self.entity_labels: Counter[str] = Counter()
        self.relation_counts: list[int] = []
        self.samples_with_relations = 0
        self.samples_without_relations = 0
        self.bbox_samples = 0
        self.words_samples = 0

    def update(self, sample: dict[str, Any]) -> None:
        self.total += 1
        self.splits[sample["split"]] += 1
        self.tasks[sample["task"]] += 1
        self.group_ids.add(sample["meta"]["group_id"])
        self.languages[sample["language"]] += 1
        self.formats[sample["output"]["format"]] += 1
        supervision = sample.get("annotations", {}).get("supervision", {})
        evaluation = sample.get("annotations", {}).get("evaluation", {})
        for field, counter in (
            ("answer_type", self.answer_types),
            ("answer_from", self.answer_froms),
            ("scale", self.scales),
        ):
            if field not in evaluation:
                continue
            value = evaluation.get(field)
            if value is not None:
                counter[str(value)] += 1
        if supervision.get("derivation") not in (None, ""):
            self.derivation_samples += 1
        if supervision.get("program") not in (None, ""):
            self.program_samples += 1
        if supervision.get("steps") not in (None, [], ""):
            self.steps_samples += 1
        if supervision.get("gold_inds") not in (None, {}, ""):
            self.gold_evidence_samples += 1
        entities = supervision.get("entities")
        if isinstance(entities, list):
            self.entity_counts.append(len(entities))
            for entity in entities:
                if isinstance(entity, dict) and entity.get("label") is not None:
                    self.entity_labels[str(entity["label"])] += 1
        relations = supervision.get("relations")
        if isinstance(relations, list):
            relation_count = len(relations)
            self.relation_counts.append(relation_count)
            if relation_count:
                self.samples_with_relations += 1
            else:
                self.samples_without_relations += 1
        if isinstance(supervision.get("boxes"), list):
            self.bbox_samples += 1
        if isinstance(supervision.get("words"), list):
            self.words_samples += 1
        images = sample["input"]["images"]
        image_count = len(images)
        self.total_references += image_count
        self.image_paths.update(image["path"] for image in images)
        self.image_counts[str(image_count)] += 1
        if image_count:
            self.with_images += 1
        else:
            self.without_images += 1
        self.lengths["context_chars"].append(len(sample["input"]["context"]))
        self.lengths["prompt_chars"].append(len(sample["input"]["prompt"]))
        if sample["output"]["references"]:
            self.supervised += 1
            self.lengths["output_chars"].append(len(sample["output"]["text"]))
        else:
            self.unsupervised += 1
        subset = sample.get("meta", {}).get("subset")
        if subset:
            self.subsets[str(subset)] += 1

    @staticmethod
    def _summary(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "min": 0, "max": 0, "average": 0}
        return {"count": len(values), "min": min(values), "max": max(values), "average": sum(values) / len(values)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "total_samples": self.total,
            "splits": dict(sorted(self.splits.items())),
            "tasks": dict(sorted(self.tasks.items())),
            "languages": dict(sorted(self.languages.items())),
            "subsets": dict(sorted(self.subsets.items())),
            "supervised": self.supervised,
            "unsupervised": self.unsupervised,
            "samples_with_images": self.with_images,
            "samples_without_images": self.without_images,
            "with_images": self.with_images,
            "without_images": self.without_images,
            "total_references": self.total_references,
            "unique_images": len(self.image_paths),
            "table_count": self.total_references,
            "unique_tables": len(self.image_paths),
            "unique_groups": len(self.group_ids),
            "program_samples": self.program_samples,
            "steps_samples": self.steps_samples,
            "gold_evidence_samples": self.gold_evidence_samples,
            "answer_types": dict(sorted(self.answer_types.items())),
            "answer_froms": dict(sorted(self.answer_froms.items())),
            "scales": dict(sorted(self.scales.items())),
            "derivation_samples": self.derivation_samples,
            "without_derivation": self.total - self.derivation_samples,
            "image_count": {
                "distribution": dict(sorted(self.image_counts.items(), key=lambda item: int(item[0])))
            },
            "lengths": {key: self._summary(values) for key, values in self.lengths.items()},
            "output_formats": dict(sorted(self.formats.items())),
            "entity_total": sum(self.entity_counts),
            "entity_counts": self._summary(self.entity_counts),
            "entity_labels": dict(sorted(self.entity_labels.items())),
            "relation_total": sum(self.relation_counts),
            "relation_counts": self._summary(self.relation_counts),
            "samples_with_relations": self.samples_with_relations,
            "samples_without_relations": self.samples_without_relations,
            "bbox_coverage": {
                "samples": self.bbox_samples,
                "rate": self.bbox_samples / self.total if self.total else 0,
            },
            "words_coverage": {
                "samples": self.words_samples,
                "rate": self.words_samples / self.total if self.total else 0,
            },
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output
