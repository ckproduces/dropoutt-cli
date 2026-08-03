"""Adapt raw records into the single internal :class:`Document` representation.

The important behaviour here is that nothing is silently coerced. When a role
string has to be mapped, the original is kept on the turn. When content is not
a string and has to be stringified, the turn is flagged. Both facts become
findings, because both are real ways a training set quietly loses data:

- A ShareGPT ``from: "gpt"`` turn is invisible to a trainer that masks on
  ``role == "assistant"``, so the record contributes no gradient at all.
- A tool-call dict that falls through to ``str(value)`` gets trained on as a
  Python repr, complete with single quotes and ``None``.
"""

from __future__ import annotations

from typing import Any

from .models import Document, Turn, content_hash
from .readers import RawRecord
from .schema_induction import ROLE_ALIASES

MAX_CONTENT_CHARS = 200_000

#: The keys each layout actually reads. Anything else in a record is carried by
#: the file but never reaches the model, and a key holding a real answer is a
#: silent data loss rather than a harmless extra column.
LAYOUT_KEYS: dict[str, tuple[str, ...]] = {
    "chatml": ("messages", "system", "system_prompt", "systemprompt"),
    "openai_chat": ("messages", "system", "system_prompt", "systemprompt"),
    "sharegpt": ("conversations", "system", "system_prompt", "systemprompt"),
    "alpaca": ("instruction", "input", "output", "response"),
    "prompt_completion": ("prompt", "completion", "response", "answer", "output"),
    "preference": ("prompt", "question", "instruction", "chosen", "rejected"),
    "qa": ("question", "answer", "answers", "output", "context", "passage"),
    "translation": ("translation",),
    "source_target": ("source", "target"),
}

#: The same table as a set per layout, built once. ``_unread_keys`` runs on
#: every record and used to rebuild this from the tuples above each time.
_LAYOUT_KEY_SETS: dict[str, frozenset[str]] = {
    layout: frozenset(keys) for layout, keys in LAYOUT_KEYS.items()
}

#: Bookkeeping columns that are supposed to be along for the ride. Flagging
#: these would bury the one key that matters under a dozen that never did.
BENIGN_KEYS = frozenset({
    "id", "uuid", "idx", "index", "source", "origin", "dataset", "split",
    "_split", "meta", "metadata", "lang", "language", "locale", "score",
    "rating", "quality", "category", "topic", "tags", "label", "labels",
    "model", "created", "created_at", "timestamp", "date", "url", "title",
    "license", "licence", "n_turns", "num_turns", "token_count", "length",
    "hash", "checksum", "version", "seed", "temperature", "difficulty",
})

#: An unread key has to hold at least this much text before it is worth a word.
#: Short leftovers are flags and enum values; a paragraph is an answer.
UNREAD_KEY_MIN_CHARS = 40


def _text_size(value: Any) -> int:
    """Roughly how much text a value carries, in characters."""
    if isinstance(value, str):
        return len(value.strip())
    if isinstance(value, list):
        return sum(_text_size(v) for v in value)
    if isinstance(value, dict):
        return sum(_text_size(v) for v in value.values())
    return 0


def _unread_keys(payload: dict, layout_id: str) -> list[tuple[str, int]]:
    """Keys carrying real text that this layout will never read.

    Measured on the Delta generation corpus: 26 records had their entire
    assistant answer sitting in a top-level ``assistant`` key beside ``messages``
    rather than inside it. The conversation then ends on a user turn, the record
    trains on nothing, and no existing check notices, because every key it does
    look at is perfectly well-formed.
    """
    known = _LAYOUT_KEY_SETS.get(layout_id)
    if known is None:
        return []
    out: list[tuple[str, int]] = []
    for key, value in payload.items():
        name = str(key).lower()
        if name in known or name in BENIGN_KEYS or name.startswith("_"):
            continue
        size = _text_size(value)
        if size >= UNREAD_KEY_MIN_CHARS:
            out.append((str(key), size))
    return sorted(out, key=lambda kv: -kv[1])


def _coerce_content(value: Any) -> tuple[str, bool]:
    """Return text and whether coercion was needed."""
    if isinstance(value, str):
        return value, False
    if value is None:
        return "", False
    if isinstance(value, list):
        # Anthropic-style content blocks: [{"type": "text", "text": "..."}]
        parts: list[str] = []
        structured = False
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                structured = True
        return "\n\n".join(parts), structured
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if isinstance(value.get(key), str):
                return value[key], True
        return str(value), True
    return str(value), True


def _map_role(raw: str) -> tuple[str, bool]:
    """Map a role string to the canonical vocabulary.

    Returns the mapped role and whether the source used a non-canonical name.
    """
    key = raw.strip().lower()
    mapped = ROLE_ALIASES.get(key)
    if mapped is None:
        return "unknown", True
    return mapped, key != mapped


def _turns_from_list(items: Any, role_key: str, content_key: str) -> list[Turn]:
    turns: list[Turn] = []
    if not isinstance(items, list):
        return turns
    for item in items:
        if not isinstance(item, dict):
            text, coerced = _coerce_content(item)
            turns.append(Turn("unknown", text[:MAX_CONTENT_CHARS], raw_role=None, coerced=True))
            continue
        raw_role = item.get(role_key)
        raw_role_s = raw_role if isinstance(raw_role, str) else ""
        role, renamed = _map_role(raw_role_s)
        text, coerced = _coerce_content(item.get(content_key))
        turns.append(
            Turn(
                role=role,
                content=text[:MAX_CONTENT_CHARS],
                raw_role=raw_role_s if (renamed or role == "unknown") else None,
                coerced=coerced,
            )
        )
    return turns


def to_document(
    rec: RawRecord, layout_id: str, dataset: str, *, index: int
) -> Document:
    """Build a Document from one raw record under an inferred layout."""
    payload = rec.payload

    if rec.error is not None or not isinstance(payload, dict):
        text = rec.raw_text or ""
        return Document(
            doc_id=content_hash(f"{rec.source_file}:{rec.source_index}:{text[:512]}"),
            text=text,
            turns=[],
            source_file=rec.source_file,
            source_index=rec.source_index,
            dataset=dataset,
            raw_keys=(),
            meta={"parse_error": rec.error} if rec.error else {},
        )

    keys = tuple(sorted(str(k) for k in payload))
    # Induction lower-cases key names before matching layouts; the lookups below
    # spell them out in lower case. A tabular export carries whatever the header
    # row said — "Question", "Result" — so a capitalised header could satisfy
    # induction and then produce a record with no turns at all. Building the
    # lower-case view once keeps the two halves reading the same record. The
    # original spellings stay in ``keys`` because that is what the report shows.
    lowered = {str(k).lower(): v for k, v in payload.items()}
    if len(lowered) == len(payload):
        payload = lowered
    turns: list[Turn] = []
    system: str | None = None

    if layout_id in ("chatml", "openai_chat"):
        turns = _turns_from_list(payload.get("messages"), "role", "content")
    elif layout_id == "sharegpt":
        turns = _turns_from_list(payload.get("conversations"), "from", "value")
    elif layout_id == "alpaca":
        instruction, _ = _coerce_content(payload.get("instruction"))
        extra, _ = _coerce_content(payload.get("input"))
        output, coerced = _coerce_content(
            payload.get("output") if payload.get("output") is not None else payload.get("response")
        )
        prompt = f"{instruction}\n\n{extra}".strip() if extra else instruction
        turns = [Turn("user", prompt), Turn("assistant", output, coerced=coerced)]
    elif layout_id == "prompt_completion":
        prompt, _ = _coerce_content(payload.get("prompt"))
        completion_val = next(
            (payload[k] for k in ("completion", "response", "answer", "output") if k in payload),
            None,
        )
        completion, coerced = _coerce_content(completion_val)
        turns = [Turn("user", prompt), Turn("assistant", completion, coerced=coerced)]
    elif layout_id == "preference":
        prompt_val = next(
            (payload[k] for k in ("prompt", "question", "instruction") if k in payload), None
        )
        prompt, _ = _coerce_content(prompt_val)
        chosen, coerced = _coerce_content(payload.get("chosen"))
        _rejected, _ = _coerce_content(payload.get("rejected"))
        turns = [Turn("user", prompt), Turn("assistant", chosen, coerced=coerced)]
    elif layout_id == "qa":
        question, _ = _coerce_content(payload.get("question"))
        answer_val = next(
            (payload[k] for k in ("answer", "answers", "output") if k in payload), None
        )
        answer, coerced = _coerce_content(answer_val)
        context, _ = _coerce_content(payload.get("context") or payload.get("passage"))
        prompt = f"{context}\n\n{question}".strip() if context else question
        turns = [Turn("user", prompt), Turn("assistant", answer, coerced=coerced)]
    elif layout_id in ("translation", "source_target"):
        if layout_id == "translation" and isinstance(payload.get("translation"), dict):
            pairs = list(payload["translation"].items())
            src = pairs[0][1] if pairs else ""
            tgt = pairs[1][1] if len(pairs) > 1 else ""
        else:
            src, _ = _coerce_content(payload.get("source"))
            tgt, _ = _coerce_content(payload.get("target"))
        turns = [Turn("user", str(src)), Turn("assistant", str(tgt))]
    else:
        body_val = next(
            (payload[k] for k in ("text", "content", "body", "raw") if k in payload), None
        )
        if body_val is not None:
            body, coerced = _coerce_content(body_val)
        else:
            # No text-bearing key. Joining the string values is honest; a
            # `str(dict)` repr would train the model on Python syntax.
            strings = [v for v in payload.values() if isinstance(v, str) and v.strip()]
            body = "\n".join(strings)
            coerced = True
        text = body[:MAX_CONTENT_CHARS]
        return Document(
            doc_id=content_hash(text) if text else content_hash(f"{rec.source_file}:{index}"),
            text=text,
            turns=[],
            source_file=rec.source_file,
            source_index=rec.source_index,
            dataset=dataset,
            raw_keys=keys,
            meta={"coerced": coerced} if coerced else {},
        )

    # A system prompt stored as a sibling column rather than a message.
    sys_val = payload.get("system") or payload.get("system_prompt") or payload.get("systemPrompt")
    if isinstance(sys_val, str) and sys_val.strip():
        system = sys_val
    else:
        first_system = next((t for t in turns if t.role == "system"), None)
        if first_system is not None:
            system = first_system.content

    flat = "\n".join(t.content for t in turns)
    meta: dict[str, Any] = {}
    unread = _unread_keys(payload, layout_id)
    if unread:
        meta["unread_keys"] = unread
    return Document(
        doc_id=content_hash(flat) if flat else content_hash(f"{rec.source_file}:{index}"),
        text=flat,
        turns=turns,
        system=system,
        source_file=rec.source_file,
        source_index=rec.source_index,
        dataset=dataset,
        raw_keys=keys,
        meta=meta,
    )


def to_messages(doc: Document) -> list[dict[str, str]]:
    """Render a Document back into chat-template input."""
    msgs: list[dict[str, str]] = []
    has_system_turn = any(t.role == "system" for t in doc.turns)
    if doc.system and not has_system_turn:
        msgs.append({"role": "system", "content": doc.system})
    for turn in doc.turns:
        role = turn.role if turn.role in ("system", "user", "assistant", "tool") else "user"
        msgs.append({"role": role, "content": turn.content})
    return msgs
