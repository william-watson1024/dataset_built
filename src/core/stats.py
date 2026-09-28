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
        self.total_references = 0
        self.supervised = 0
        self.unsupervised = 0
        self.with_images = 0
        self.without_images = 0
        self.lengths = {"context_chars": [], "prompt_chars": [], "output_chars": []}

    def update(self, sample: dict[str, Any]) -> None:
        self.total += 1
        self.splits[sample["split"]] += 1
        self.tasks[sample["task"]] += 1
        self.languages[sample["language"]] += 1
        self.formats[sample["output"]["format"]] += 1
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
            "image_count": {
                "distribution": dict(sorted(self.image_counts.items(), key=lambda item: int(item[0])))
            },
            "lengths": {key: self._summary(values) for key, values in self.lengths.items()},
            "output_formats": dict(sorted(self.formats.items())),
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output
