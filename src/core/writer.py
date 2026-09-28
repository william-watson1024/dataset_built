from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None

    def __enter__(self) -> "JsonlWriter":
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def write(self, sample: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")
        self._handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
        self._handle.write("\n")

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
