from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.converter import Converter


class Doc2EDAGConverter(Converter):
    """Convert document-level ChFinAnn/Doc2EDAG event annotations."""

    TASK = "financial_event_extraction"
    PROMPT = "请从该金融公告中抽取所有事件及其论元，并以 JSON 输出。"

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"Doc2EDAG source file not found: {source_file}")
            data = json.loads(source_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Doc2EDAG source file must contain a JSON array: {source_file}")
            original_split = source_file.stem
            for index, record in enumerate(data):
                if not isinstance(record, list) or len(record) != 2 or not isinstance(record[1], dict):
                    raise ValueError(f"Doc2EDAG record must be [document_id, payload]: {source_file}:{index}")
                document_id, payload = record
                raw = dict(payload)
                raw["__document_id"] = str(document_id)
                raw["__source_file"] = source_file.name
                raw["__source_index"] = index
                raw["__original_split"] = original_split
                yield raw

    @staticmethod
    def _events(payload: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        raw_events = payload.get("recguid_eventname_eventdict_list") or []
        if not isinstance(raw_events, list):
            raise ValueError("Doc2EDAG event list must be an array")
        for index, event in enumerate(raw_events):
            if not isinstance(event, list) or len(event) < 3:
                raise ValueError(f"Doc2EDAG event {index} must contain record id, type, and arguments")
            arguments = event[2]
            if not isinstance(arguments, dict):
                raise ValueError(f"Doc2EDAG event {index} arguments must be an object")
            events.append({
                "event_type": str(event[1]),
                "arguments": {str(role): value for role, value in arguments.items()},
            })
        return events

    @staticmethod
    def _supervision(payload: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
        supervision: dict[str, Any] = {
            "events": events,
            "event_types": [event["event_type"] for event in events],
            "sentences": payload.get("sentences", []),
        }
        for field in (
            "ann_valid_mspans",
            "ann_valid_dranges",
            "ann_mspan2dranges",
            "ann_mspan2guess_field",
            "recguid_eventname_eventdict_list",
        ):
            if field in payload:
                supervision[field] = payload[field]
        return supervision

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        document_id = str(raw_sample["__document_id"])
        sentences = raw_sample.get("sentences")
        if not isinstance(sentences, list) or not all(isinstance(sentence, str) for sentence in sentences):
            raise ValueError(f"Doc2EDAG sentences must be a string array: {document_id}")
        events = self._events(raw_sample)
        answer_object = {"events": events}
        answer_text = json.dumps(answer_object, ensure_ascii=False, separators=(",", ":"))
        return {
            "id": f"doc2edag:{split}:{document_id}",
            "source": "Doc2EDAG",
            "split": split,
            "task": self.TASK,
            "language": "zh",
            "input": {
                "images": [],
                "context": "\n".join(sentences),
                "prompt": self.PROMPT,
                "history": [],
            },
            "output": {
                "text": answer_text,
                "format": "json",
                "references": [answer_text],
            },
            "annotations": {
                "supervision": self._supervision(raw_sample, events),
                "evaluation": {
                    "event_count": len(events),
                    "has_events": bool(events),
                },
                "raw": {
                    "document_id": document_id,
                    "source_file": raw_sample["__source_file"],
                    "source_index": raw_sample["__source_index"],
                    "original_split": raw_sample["__original_split"],
                },
            },
            "meta": {
                "source_id": document_id,
                "group_id": f"doc2edag:document:{document_id}",
                "subset": None,
                "original_split": raw_sample["__original_split"],
                "transforms": ["doc2edag_document_to_structured_json:v1"],
                "source_file": raw_sample["__source_file"],
            },
        }
