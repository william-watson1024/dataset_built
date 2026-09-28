"""Reusable table rendering utilities and dataset adapters."""

from .base import DatasetTableRenderer
from .finqa import FinQATableRenderer
from .registry import available_renderers, create_renderer, register_renderer
from .table_renderer import TableRenderer, normalize_table, table_hash
from .tatqa import TATQATableRenderer

__all__ = [
    "TableRenderer",
    "DatasetTableRenderer",
    "FinQATableRenderer",
    "TATQATableRenderer",
    "available_renderers",
    "create_renderer",
    "normalize_table",
    "register_renderer",
    "table_hash",
]
