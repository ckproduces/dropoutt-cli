"""Report rendering."""

from __future__ import annotations

from . import html, markdown, terminal
from .escaping import escape_html, json_script_payload, safe_snippet

__all__ = ["escape_html", "html", "json_script_payload", "markdown",
           "safe_snippet", "terminal"]
