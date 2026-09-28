from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from src.core.converter import Converter
from src.converters.docvqa import MediaStore


_TASK_NAMES = {
    "TQA": "table_question_answering",
    "TFV": "table_fact_verification",
    "T2T": "table_to_text",
    "TSD": "table_size_detection",
    "TCL": "table_cell_localization",
    "RCE": "row_column_extraction",
    "MCD": "merged_cell_detection",
    "TCE": "table_cell_extraction",
    "TCR": "table_cell_recognition",
    "TR": "table_recognition",
    "TSR": "table_structure_recognition",
    "TAT-QA": "table_question_answering",
    "AIT-QA": "table_question_answering",
    "WTQ": "table_question_answering",
    "TABMWP": "table_question_answering",
    "TabFact": "table_fact_verification",
    "InfoTabs": "table_question_answering",
    "HiTab": "table_question_answering",
    "FeTaQA": "table_question_answering",
    "PubHealthTab": "table_question_answering",
    "TabMCQ": "table_question_answering",
    "Rotowire": "table_to_text",
    "ToTTo": "table_to_text",
    "WikiBIO": "table_to_text",
    "table": "table_multitask",
    "table_recognition": "table_recognition",
}


def _task_name(raw_type: str, dataset_name: str | None = None) -> str:
    value = raw_type
    if value.startswith("OOD_"):
        value = value[4:]
    return _TASK_NAMES.get(value) or _TASK_NAMES.get(dataset_name or "") or "table_multitask"


def _format_name(prompt: str, output: str) -> str:
    combined = f"{prompt}\n{output}".lower()
    if "<table" in output.lower() or "html" in combined:
        return "html"
    if "json" in combined or ("{" in output and "}" in output) or ("[" in output and "]" in output):
        return "json"
    return "text"


def _clean_image_marker(value: Any) -> str:
    return re.sub(r"<image>", "", str(value), flags=re.IGNORECASE).strip()


def _role(value: str) -> str:
    if value == "human":
        return "user"
    if value == "gpt":
        return "assistant"
    raise ValueError(f"Unsupported MMTab conversation role: {value!r}")


class _JsonArrayReader:
    """Small streaming reader for the large top-level JSON arrays used by MMTab."""

    def __init__(self, path: Path, chunk_size: int = 1024 * 1024) -> None:
        self.path = path
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        decoder = json.JSONDecoder()
        with self.path.open("r", encoding="utf-8") as handle:
            buffer = handle.read(self.chunk_size)
            buffer = buffer.lstrip()
            if not buffer.startswith("["):
                raise ValueError(f"MMTab annotation file must contain a JSON array: {self.path}")
            buffer = buffer[1:]
            while True:
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                while True:
                    try:
                        item, end = decoder.raw_decode(buffer)
                        break
                    except json.JSONDecodeError as exc:
                        more = handle.read(self.chunk_size)
                        if not more:
                            raise ValueError(f"Invalid/truncated MMTab JSON: {self.path}") from exc
                        buffer += more
                if not isinstance(item, dict):
                    raise ValueError(f"MMTab record must be an object: {self.path}")
                yield item
                buffer = buffer[end:]
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                elif buffer.startswith("]"):
                    return
                else:
                    while not buffer:
                        more = handle.read(self.chunk_size)
                        if not more:
                            raise ValueError(f"Invalid/truncated MMTab JSON: {self.path}")
                        buffer += more
                    if not buffer.startswith(",") and not buffer.startswith("]"):
                        raise ValueError(f"Invalid separator in MMTab JSON: {self.path}")


class MMTabConverter(Converter):
    """Streaming converter for MMTab instruct JSON and eval JSON plus ZIP media."""

    def __init__(self, config: dict[str, Any], project_root: Path, media_dir: Path) -> None:
        super().__init__(config, project_root)
        self.media_store = MediaStore(media_dir)
        self._zip_archives: dict[Path, zipfile.ZipFile] = {}
        self._zip_exact: dict[Path, set[str]] = {}
        self._zip_by_basename: dict[Path, dict[str, list[str]]] = {}
        self._media_cache: dict[tuple[str, str], str] = {}

    def _source_files(self, split: str) -> list[Path]:
        paths = self.config.get("source_files", {}).get(split, [])
        return [Path(item) if Path(item).is_absolute() else self.project_root / Path(item) for item in paths]

    def _open_zip(self, path: Path) -> zipfile.ZipFile:
        if path not in self._zip_archives:
            if not path.is_file():
                raise FileNotFoundError(f"MMTab image ZIP not found: {path}")
            archive = zipfile.ZipFile(path)
            self._zip_archives[path] = archive
            exact = set()
            by_basename: dict[str, list[str]] = {}
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                exact.add(name)
                by_basename.setdefault(Path(name).name, []).append(name)
            self._zip_exact[path] = exact
            self._zip_by_basename[path] = by_basename
        return self._zip_archives[path]

    def _find_zip_member(self, image_ref: str, archives: list[Path]) -> tuple[Path, str] | None:
        names = [image_ref, Path(image_ref).name]
        if not Path(image_ref).suffix:
            names.extend([f"{image_ref}.jpg", f"{Path(image_ref).name}.jpg"])
        for archive_path in archives:
            self._open_zip(archive_path)
            exact = self._zip_exact[archive_path]
            for name in names:
                if name in exact:
                    return archive_path, name
            for name in names:
                candidates = self._zip_by_basename[archive_path].get(Path(name).name, [])
                if len(candidates) == 1:
                    return archive_path, candidates[0]
                if len(candidates) > 1:
                    raise ValueError(
                        f"Ambiguous MMTab image basename {Path(name).name!r} in {archive_path}"
                    )
        return None

    def load_split(self, split: str) -> Iterable[dict[str, Any]]:
        configured = self._source_files(split)
        annotation_files = [p for p in configured if p.suffix.lower() in {".json", ".jsonl"}]
        archives = [p for p in configured if p.suffix.lower() == ".zip"]
        if not annotation_files:
            raise ValueError(f"No MMTab annotation JSON configured for split: {split}")
        for annotation_file in annotation_files:
            if annotation_file.suffix.lower() != ".json":
                raise ValueError(f"MMTab currently expects JSON array annotations: {annotation_file}")
            for row_index, row in enumerate(_JsonArrayReader(annotation_file)):
                raw = dict(row)
                if "conversations" in raw:
                    image_ref = str(raw.get("image", ""))
                    record_kind = "instruct"
                else:
                    image_ref = f"{raw.get('image_id', '')}.jpg"
                    record_kind = "eval"
                if not image_ref or image_ref == ".jpg":
                    raise ValueError(f"MMTab record has no image reference: {annotation_file}:{row_index}")
                media_ref = self._find_zip_member(image_ref, archives)
                if media_ref is None:
                    candidate = annotation_file.parent / image_ref
                    if not candidate.is_file():
                        raise FileNotFoundError(
                            f"MMTab image not found for {image_ref!r}: {annotation_file}:{row_index}"
                        )
                    raw["__image_file"] = str(candidate)
                else:
                    raw["__image_zip"], raw["__image_member"] = map(str, media_ref)
                raw["__source_file"] = annotation_file.name
                raw["__row_index"] = row_index
                raw["__record_kind"] = record_kind
                yield raw

    def _materialize_image(self, raw: dict[str, Any]) -> str:
        if "__image_file" in raw:
            return self.media_store.materialize_file(Path(raw["__image_file"]))
        zip_path = str(raw["__image_zip"])
        member = str(raw["__image_member"])
        key = (zip_path, member)
        if key not in self._media_cache:
            data = self._open_zip(Path(zip_path)).read(member)
            self._media_cache[key] = self.media_store.materialize_bytes(data, Path(member).name)
        return self._media_cache[key]

    @staticmethod
    def _instruct_metadata(record_id: str, image_ref: str) -> tuple[str, str]:
        if record_id.startswith("table_recognition"):
            return "table_recognition", "MMTab"
        prefix = record_id.split("_")[0]
        dataset_name = prefix if prefix in _TASK_NAMES else "MMTab"
        return prefix, dataset_name

    @staticmethod
    def _conversation_fields(conversations: Any) -> tuple[str, list[dict[str, str]], str]:
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("MMTab instruct record has no conversations")
        final_gpt = next((i for i in range(len(conversations) - 1, -1, -1) if conversations[i].get("from") == "gpt"), None)
        if final_gpt is None:
            raise ValueError("MMTab instruct record has no assistant answer")
        user_indices = [i for i in range(final_gpt) if conversations[i].get("from") == "human"]
        if not user_indices:
            raise ValueError("MMTab instruct record has no user prompt")
        prompt_index = user_indices[-1]
        prompt = _clean_image_marker(conversations[prompt_index].get("value", ""))
        history = []
        for message in conversations[:prompt_index]:
            history.append({"role": _role(str(message.get("from"))), "content": _clean_image_marker(message.get("value", ""))})
        output = _clean_image_marker(conversations[final_gpt].get("value", ""))
        return prompt, history, output

    @staticmethod
    def _eval_references(answer_list: Any) -> list[str]:
        if not isinstance(answer_list, list):
            return [] if answer_list is None else [str(answer_list)]
        references = []
        for answer in answer_list:
            if isinstance(answer, (list, dict)):
                references.append(json.dumps(answer, ensure_ascii=False))
            else:
                references.append(str(answer))
        return references

    def convert_sample(self, raw_sample: dict[str, Any], split: str) -> dict[str, Any]:
        media_path = self._materialize_image(raw_sample)
        if raw_sample["__record_kind"] == "instruct":
            record_id = str(raw_sample["id"])
            image_ref = str(raw_sample["image"])
            prompt, history, output_text = self._conversation_fields(raw_sample["conversations"])
            raw_type, dataset_name = self._instruct_metadata(record_id, image_ref)
            task_type = _task_name(raw_type, dataset_name)
            references = [output_text] if output_text else []
            fmt = _format_name(prompt, output_text)
            source_id = record_id
            raw = {
                "id": raw_sample["id"],
                "image": raw_sample["image"],
                "conversations": raw_sample["conversations"],
                "task_type": raw_type,
                "dataset_name": dataset_name,
                "source_file": raw_sample["__source_file"],
                "row_index": raw_sample["__row_index"],
            }
            supervision = {
                "task_type": raw_type,
                "dataset_name": dataset_name,
                "output_format": fmt,
            }
            evaluation = {"has_answers": bool(references), "turn_count": len(raw_sample["conversations"]) // 2}
            subset = "instruct"
            original_split = split
        else:
            item_id = str(raw_sample["item_id"])
            image_ref = f"{raw_sample['image_id']}.jpg"
            prompt = str(raw_sample["input"])
            history = []
            output_text = str(raw_sample.get("output", ""))
            raw_type = str(raw_sample.get("task_type", "unknown"))
            dataset_name = str(raw_sample.get("dataset_name", "MMTab"))
            task_type = _task_name(raw_type, dataset_name)
            references = self._eval_references(raw_sample.get("answer_list"))
            fmt = _format_name(prompt, output_text)
            source_id = item_id
            raw = {
                "item_id": raw_sample["item_id"],
                "image_id": raw_sample["image_id"],
                "input": raw_sample["input"],
                "output": raw_sample["output"],
                "task_type": raw_sample["task_type"],
                "dataset_name": raw_sample["dataset_name"],
                "original_query": raw_sample["original_query"],
                "answer_list": raw_sample["answer_list"],
                "original_query_type": raw_sample["original_query_type"],
                "source_file": raw_sample["__source_file"],
                "row_index": raw_sample["__row_index"],
            }
            supervision = {
                "task_type": raw_type,
                "dataset_name": dataset_name,
                "original_query_type": raw_sample.get("original_query_type"),
                "output_format": fmt,
            }
            evaluation = {
                "answer_list": raw_sample.get("answer_list"),
                "answer_count": len(references),
                "has_answers": bool(references),
            }
            subset = "eval"
            original_split = split

        image_key = image_ref
        return {
            "id": f"mmtab:{split}:{source_id}",
            "source": "MMTab",
            "split": split,
            "task": f"mmtab_{task_type}",
            "language": "en",
            "input": {
                "images": [{"storage": "file", "path": media_path}],
                "context": "",
                "prompt": prompt,
                "history": history,
            },
            "output": {
                "text": output_text,
                "format": fmt,
                "references": references,
            },
            "annotations": {
                "supervision": supervision,
                "evaluation": evaluation,
                "raw": raw,
            },
            "meta": {
                "source_id": source_id,
                "group_id": f"mmtab:table:{image_key}",
                "subset": subset,
                "original_split": original_split,
                "transforms": ["mmtab_conversation_to_canonical:v1", "zip_image_to_file:v1"],
            },
        }
