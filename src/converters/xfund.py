from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.converters.docvqa import MediaStore
from src.core.converter import Converter


class XFUNDConverter(Converter):
    """Convert XFUND form documents into document-level structured samples."""

    TASK = "form_information_extraction"
    PROMPT = "Extract the entities and key-value relations from this form."

    def __init__(self, config: dict[str, Any], project_root: Path, media_dir: Path) -> None:
        super().__init__(config, project_root)
        self.media_store = MediaStore(media_dir)

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        for source_file in self._source_files(split):
            if not source_file.is_file():
                raise FileNotFoundError(f"XFUND source file not found: {source_file}")
            payload = json.loads(source_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
                raise ValueError(f"XFUND source must contain a documents array: {source_file}")
            language = str(payload.get("lang") or source_file.name.split(".", 1)[0])
            original_split = str(payload.get("split") or source_file.name.split(".")[1])
            for index, document in enumerate(payload["documents"]):
                if not isinstance(document, dict):
                    raise ValueError(f"XFUND document {index} is not an object: {source_file}")
                raw = dict(document)
                raw["__source_file"] = source_file.name
                raw["__source_path"] = str(source_file)
                raw["__source_index"] = index
                raw["__language"] = language
                raw["__original_split"] = original_split
                yield raw

    @staticmethod
    def _image_name(raw_sample: dict[str, Any]) -> str:
        image = raw_sample.get("img")
        if not isinstance(image, dict) or not image.get("fname"):
            raise ValueError(f"XFUND sample has no image filename: {raw_sample.get('id', '<missing>')}")
        return str(image["fname"])

    def _image_path(self, raw_sample: dict[str, Any]) -> Path:
        image_name = self._image_name(raw_sample)
        source_path = Path(str(raw_sample["__source_path"]))
        candidates = (
            source_path.parent / image_name,
            source_path.parent / "images" / image_name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"XFUND image not found for {raw_sample.get('id', '<missing>')}: "
            f"{image_name} (checked {', '.join(str(path) for path in candidates)})"
        )

    @staticmethod
    def _document_annotations(document: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        entities: list[dict[str, Any]] = []
        boxes: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        seen_relations: set[tuple[int, int]] = set()
        entity_ids: set[int] = set()

        for entity in document:
            if not isinstance(entity, dict) or "id" not in entity:
                raise ValueError("XFUND entity must be an object with an id")
            entity_id = int(entity["id"])
            entity_ids.add(entity_id)
            entities.append({
                "id": entity_id,
                "text": str(entity.get("text", "")),
                "label": str(entity.get("label", "")),
            })
            boxes.append({"entity_id": entity_id, "box": entity.get("box")})
            for word in entity.get("words") or []:
                if not isinstance(word, dict):
                    raise ValueError(f"XFUND words for entity {entity_id} must be objects")
                word_record = dict(word)
                word_record["entity_id"] = entity_id
                words.append(word_record)

        for entity in document:
            source_id = int(entity["id"])
            for link in entity.get("linking") or []:
                if not isinstance(link, list) or len(link) != 2:
                    raise ValueError(f"XFUND linking for entity {source_id} must contain pairs")
                pair = (int(link[0]), int(link[1]))
                if pair[0] not in entity_ids or pair[1] not in entity_ids:
                    raise ValueError(f"XFUND relation references an unknown entity: {pair}")
                if pair not in seen_relations:
                    seen_relations.add(pair)
                    relations.append({"from": pair[0], "to": pair[1]})
        return entities, relations, boxes, words

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        document = raw_sample.get("document")
        if not isinstance(document, list):
            raise ValueError(f"XFUND sample has no document array: {raw_sample.get('id', '<missing>')}")
        entities, relations, boxes, words = self._document_annotations(document)
        image_name = self._image_name(raw_sample)
        media_path = self.media_store.materialize_file(self._image_path(raw_sample))
        source_id = str(raw_sample.get("id", raw_sample.get("uid", raw_sample["__source_index"])))
        uid = str(raw_sample.get("uid") or source_id)
        language = str(raw_sample["__language"])
        original_split = str(raw_sample["__original_split"])
        answer_object = {"entities": entities, "relations": relations}
        answer_text = json.dumps(answer_object, ensure_ascii=False, separators=(",", ":"))
        transforms = ["xfund_document_to_structured_json:v1", "image_to_processed_media:v1"]
        if original_split == "val" and split == "test":
            transforms.append("official_val_to_test:v1")
        return {
            "id": f"xfund:{split}:{language}:{source_id}",
            "source": "XFUND",
            "split": split,
            "task": self.TASK,
            "language": language,
            "input": {
                "images": [{"storage": "file", "path": media_path}],
                "context": "",
                "prompt": self.PROMPT,
                "history": [],
            },
            "output": {
                "text": answer_text,
                "format": "json",
                "references": [answer_text],
            },
            "annotations": {
                "supervision": {
                    "entities": entities,
                    "relations": relations,
                    "boxes": boxes,
                    "words": words,
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                },
                "evaluation": {
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "has_relations": bool(relations),
                },
                "raw": {
                    "id": raw_sample.get("id"),
                    "uid": raw_sample.get("uid"),
                    "img": raw_sample.get("img"),
                    "document": document,
                    "source_file": raw_sample["__source_file"],
                    "source_index": raw_sample["__source_index"],
                    "language": language,
                    "original_split": original_split,
                },
            },
            "meta": {
                "source_id": source_id,
                "group_id": f"xfund:document:{language}:{uid}",
                "subset": language,
                "original_split": original_split,
                "transforms": transforms,
                "source_file": raw_sample["__source_file"],
            },
        }
