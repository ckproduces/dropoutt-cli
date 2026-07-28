"""Report rendering."""

from __future__ import annotations

from . import html, terminal  # noqa: F401
from .escaping import escape_html, json_script_payload, safe_snippet  # noqa: F401

__all__ = ["html", "terminal", "safe_snippet", "escape_html", "json_script_payload"]
