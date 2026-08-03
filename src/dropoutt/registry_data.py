"""Loader for the shipped static reference data.

Everything here is read from JSON files under ``dropoutt/data`` via
``importlib.resources``, so it works from an installed wheel and from a zipapp,
not only from a source checkout.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import Any

from .compat import json_loads
from .regexgate import PatternGate, gate_for


def _load(*parts: str) -> dict[str, Any]:
    ref = resources.files("dropoutt.data")
    for part in parts:
        ref = ref / part
    return json_loads(ref.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def benchmarks() -> dict[str, Any]:
    return _load("benchmarks.json")


@lru_cache(maxsize=1)
def chat_templates() -> dict[str, Any]:
    return _load("chat_templates.json")


@lru_cache(maxsize=1)
def models() -> dict[str, Any]:
    return _load("models.json")


@lru_cache(maxsize=1)
def taxonomy() -> dict[str, Any]:
    return _load("taxonomy.json")


@lru_cache(maxsize=1)
def pii_patterns() -> dict[str, Any]:
    return _load("patterns", "pii.json")


@lru_cache(maxsize=1)
def identity_patterns() -> dict[str, Any]:
    return _load("patterns", "identity.json")


@lru_cache(maxsize=1)
def style_patterns() -> dict[str, Any]:
    return _load("patterns", "style.json")


# --------------------------------------------------------------------------
# Compiled views
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def compiled_pii() -> list[tuple[str, str, str, re.Pattern[str], str | None, PatternGate]]:
    """(id, label, severity, compiled regex, validator name, gate)."""
    out = []
    for p in pii_patterns()["patterns"]:
        out.append((p["id"], p["label"], p["severity"], re.compile(p["regex"]),
                    p.get("validator"), gate_for(p["regex"])))
    return out


@lru_cache(maxsize=1)
def compiled_identity() -> list[tuple[str, str, str, re.Pattern[str], PatternGate]]:
    """(group, id, lang, compiled regex, gate)."""
    data = identity_patterns()
    out = []
    for group in ("identity_leakage", "refusal_boilerplate"):
        for p in data[group]["patterns"]:
            out.append((group, p["id"], p["lang"], re.compile(p["regex"]),
                        gate_for(p["regex"])))
    return out


@lru_cache(maxsize=1)
def compiled_style_openers() -> list[tuple[str, str, re.Pattern[str], PatternGate]]:
    return [
        (p["id"], p["lang"], re.compile(p["regex"]), gate_for(p["regex"]))
        for p in style_patterns()["openers"]
    ]


@lru_cache(maxsize=1)
def template_index() -> list[dict[str, Any]]:
    return chat_templates()["families"]


def resolve_model_alias(name: str) -> str:
    """Map a shorthand such as ``qwen3`` to a full Hub id."""
    data = models()
    return data["aliases"].get(name.lower(), name)


def model_info(hf_id: str) -> dict[str, Any] | None:
    for m in models()["models"]:
        if m["hf_id"].lower() == hf_id.lower():
            return m
    return None


def template_family_for_model(hf_id: str) -> str | None:
    info = model_info(hf_id)
    return info.get("template") if info else None


def shippable_benchmarks() -> list[dict[str, Any]]:
    """Benchmarks whose licence permits us to distribute a hashed index."""
    return [b for b in benchmarks()["benchmarks"] if b.get("shippable")]


def detect_template_family(text: str) -> list[tuple[str, int]]:
    """Which chat template, if any, this text was already formatted with.

    Returns (family id, number of distinct delimiters matched), best first. The
    count matters: one stray ``[INST]`` in prose is not evidence, three
    delimiters from the same family is.
    """
    hits: list[tuple[str, int]] = []
    for family, delimiters, openers in _template_gates():
        if not any(opener in text for opener in openers):
            continue
        matched = sum(1 for d in delimiters if d in text)
        if matched:
            hits.append((family, matched))
    return sorted(hits, key=lambda kv: -kv[1])


@lru_cache(maxsize=1)
def _template_gates() -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    """Each family with its delimiters and the characters they can start with.

    A delimiter cannot be present unless its first character is, and a single
    character scan is memchr where a substring scan is not. Most records match
    no family and pay one character test per family instead of six substring
    searches. The gate is derived from the delimiters, never written by hand,
    so a new family cannot arrive with a stale gate that switches it off.
    """
    return tuple(
        (
            family["id"],
            tuple(family["delimiters"]),
            tuple({d[0] for d in family["delimiters"] if d}),
        )
        for family in template_index()
    )


# --------------------------------------------------------------------------
# Validators for PII patterns that have one. A regex without its checksum is
# worse than no check: an eleven-digit regex matches every phone number, every
# order id and every timestamp in the corpus.
# --------------------------------------------------------------------------


def tckn_checksum(value: str) -> bool:
    """Turkish national identification number checksum."""
    if len(value) != 11 or not value.isdigit() or value[0] == "0":
        return False
    d = [int(c) for c in value]
    odd = d[0] + d[2] + d[4] + d[6] + d[8]
    even = d[1] + d[3] + d[5] + d[7]
    if (odd * 7 - even) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def iban_mod97(value: str) -> bool:
    """IBAN mod-97 check."""
    v = value.replace(" ", "").upper()
    if len(v) < 15 or len(v) > 34:
        return False
    rearranged = v[4:] + v[:4]
    digits = ""
    for ch in rearranged:
        if ch.isdigit():
            digits += ch
        elif ch.isalpha():
            digits += str(ord(ch) - 55)
        else:
            return False
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def luhn(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 0:
            total += digit
            continue
        doubled = digit * 2
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


VALIDATORS = {
    "tckn_checksum": tckn_checksum,
    "iban_mod97": iban_mod97,
    "luhn": luhn,
}


def mask_value(value: str, style: str) -> str:
    """Produce a safe display form. The real value never leaves the scanner."""
    if style == "full":
        return "[redacted]"
    if style == "prefix6":
        return value[:6] + "…" + "*" * 6
    if style == "tail4":
        return "*" * max(0, len(value) - 4) + value[-4:]
    if style == "local_domain":
        local, _, domain = value.partition("@")
        if not domain:
            return "[redacted]"
        keep = local[:2] if len(local) > 2 else local[:1]
        dom, _, tld = domain.rpartition(".")
        return f"{keep}***@{dom[:1]}***.{tld}"
    return "[redacted]"
