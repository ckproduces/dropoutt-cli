"""Hugging Face dataset card front-matter parsing.

The spec says ``license`` is a scalar and ``language`` is a list. Real
repositories disagree constantly: ``cais/mmlu`` ships ``license: ["mit"]`` while
``TIGER-Lab/MMLU-Pro`` ships ``license: "mit"``, and ``language`` appears both
ways. So every field is coerced rather than trusted, and ``data_files`` may be a
bare string rather than a list of split mappings.

A minimal YAML subset is parsed by hand so the package does not need PyYAML in
its core dependency set. Front-matter is small and shallow, and the alternative
is adding a dependency to read ten lines.
"""

from __future__ import annotations

import re
from typing import Any

_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)


def parse_card(text: str) -> dict[str, Any]:
    """Extract the YAML front-matter of a dataset or model card."""
    match = _FM.match(text)
    if not match:
        return {}
    return _parse_block(match.group(1))


def _parse_block(block: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[str] = []

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if stripped.startswith("- ") and current_key is not None and indent > 0:
            current_list.append(_scalar(stripped[2:]))
            continue

        if ":" in stripped and indent == 0:
            if current_key is not None:
                out[current_key] = current_list if current_list else out.get(current_key)
                current_list = []
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            if value:
                out[key] = _value(value)
                current_key = None
            else:
                out[key] = None

    if current_key is not None and current_list:
        out[current_key] = current_list

    return _normalize(out)


def _scalar(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def _value(v: str) -> object:
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_scalar(p) for p in inner.split(",")]
    return _scalar(v)


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce the fields we care about into predictable shapes."""
    out: dict[str, Any] = {}

    lic = raw.get("license")
    if isinstance(lic, list):
        out["license"] = lic[0] if lic else None
    elif isinstance(lic, str):
        out["license"] = lic or None
    else:
        out["license"] = None

    lang = raw.get("language")
    if isinstance(lang, str):
        out["language"] = [lang]
    elif isinstance(lang, list):
        out["language"] = [str(x) for x in lang]
    else:
        out["language"] = []

    tasks = raw.get("task_categories")
    if isinstance(tasks, str):
        out["task_categories"] = [tasks]
    elif isinstance(tasks, list):
        out["task_categories"] = [str(x) for x in tasks]
    else:
        out["task_categories"] = []

    for key in ("pretty_name", "license_name", "license_link"):
        val = raw.get(key)
        if isinstance(val, str) and val:
            out[key] = val

    return out
