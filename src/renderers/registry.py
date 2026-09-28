from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .finqa import FinQATableRenderer
from .table_renderer import TableRenderer
from .tatqa import TATQATableRenderer


RendererFactory = Callable[..., Any]
_RENDERERS: dict[str, RendererFactory] = {
    "finqa": FinQATableRenderer,
    "tatqa": TATQATableRenderer,
}


def register_renderer(name: str, factory: RendererFactory) -> None:
    """Register a dataset adapter without changing the common execution flow."""
    if not name or not callable(factory):
        raise ValueError("renderer name and callable factory are required")
    _RENDERERS[name] = factory


def create_renderer(name: str, engine: TableRenderer | None = None) -> Any:
    try:
        factory = _RENDERERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown table renderer {name!r}; available: {sorted(_RENDERERS)}") from exc
    return factory(engine=engine)


def available_renderers() -> tuple[str, ...]:
    return tuple(sorted(_RENDERERS))
