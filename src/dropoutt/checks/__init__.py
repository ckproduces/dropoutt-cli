"""Check modules.

Imported explicitly rather than discovered, because import order determines
findings order and a scanner whose output shuffles between runs cannot be
diffed.
"""

from __future__ import annotations

from . import (  # noqa: F401
    tier0_corpus,
    tier0_generation,
    tier0_hygiene,
    tier0_structure,
    tier0_tokens,
    tier1_atlas,
    tier1_content,
    tier1_dedup,
    tier1_language,
)
from .base import REGISTRY, Check, Registry, make_finding, register

__all__ = ["REGISTRY", "Check", "Registry", "make_finding", "register"]
