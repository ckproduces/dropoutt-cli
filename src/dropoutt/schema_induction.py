"""Schema induction: work out what shape these records are, without being told.

Layouts are scored rather than matched, because real files are messy and a
strict matcher would reject a file over one stray key. The distribution of
layouts across a folder is itself reported: three layouts in one directory is
almost always a collection bug, and it will silently break a preparation script
written for one of them.

This module also answers a question no other tool asks: *is this training data
at all?* Pointing a scanner at a project folder routinely turns up agent session
logs, telemetry and rollout traces. Forcing those into a chat layout produces
confident nonsense, so they are detected and reported as what they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .readers import RawRecord

# --------------------------------------------------------------------------
# Known layouts
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layout:
    layout_id: str
    label: str
    #: Keys that must be present for the layout to be considered at all.
    required: tuple[str, ...]
    #: Any-of groups: at least one key from each group must be present.
    any_of: tuple[tuple[str, ...], ...] = ()
    #: Keys that add confidence when present.
    optional: tuple[str, ...] = ()
    kind: str = "conversation"


LAYOUTS: tuple[Layout, ...] = (
    Layout("chatml", "ChatML messages", ("messages",), (), ("system", "source", "type"),
           "conversation"),
    Layout("sharegpt", "ShareGPT conversations", ("conversations",), (), ("system", "id"),
           "conversation"),
    Layout("openai_chat", "OpenAI chat completion", ("messages",), (), ("model", "choices"),
           "conversation"),
    Layout("alpaca", "Alpaca instruction", ("instruction",),
           (("output", "response"),), ("input", "text"), "completion"),
    Layout("prompt_completion", "Prompt/completion", ("prompt",),
           (("completion", "response", "answer", "output"),), (), "completion"),
    Layout("preference", "Preference triple", ("chosen", "rejected"),
           (), ("prompt", "question", "conversation"), "preference"),
    Layout("qa", "Question/answer", ("question",),
           (("answer", "answers", "output"),), ("context", "passage"), "qa"),
    Layout("translation", "Parallel translation", ("translation",), (), (), "translation"),
    Layout("source_target", "Source/target pair", ("source",), (("target",),), (), "translation"),
    Layout("text", "Bare text", ("text",), (), ("title", "url", "id", "meta"), "text"),
    Layout("content_only", "Bare content", ("content",), (), ("title", "url"), "text"),
)

#: Role key aliases seen in the wild. The mapping matters: a record whose role
#: is ``gpt`` produces zero trainable tokens in a trainer that only recognises
#: ``assistant``, and nothing warns you.
ROLE_ALIASES = {
    "user": "user", "human": "user", "prompter": "user", "client": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
    "model": "assistant", "chatgpt": "assistant", "ai": "assistant",
    "system": "system", "instruction": "system",
    "tool": "tool", "function": "tool", "observation": "tool",
}

#: Keys that hold the message list under each conversational layout.
TURN_KEYS = {"messages": ("role", "content"), "conversations": ("from", "value")}


# --------------------------------------------------------------------------
# Non-training-data detection
# --------------------------------------------------------------------------

#: Key signatures that indicate a log or trace rather than a dataset. Each entry
#: is a set of keys that, seen together, is decisive.
LOG_SIGNATURES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"type", "timestamp", "sessionid"}), "agent session log"),
    (frozenset({"type", "timestamp", "payload"}), "session trace"),
    (frozenset({"parentuuid", "sessionid"}), "agent session log"),
    (frozenset({"uuid", "parentuuid"}), "agent session log"),
    (frozenset({"level", "msg", "time"}), "application log"),
    (frozenset({"level", "message", "timestamp"}), "application log"),
    (frozenset({"event", "properties", "distinct_id"}), "product telemetry"),
    (frozenset({"span_id", "trace_id"}), "tracing span"),
    (frozenset({"operation", "timestamp", "sessionid"}), "queue operation log"),
)

#: Values of a ``type`` field that mark a trace record.
LOG_TYPE_VALUES = {
    "session_meta", "queue-operation", "turn_context", "event_msg",
    "response_item", "telemetry", "span", "heartbeat", "tool_result",
}


@dataclass(slots=True)
class SchemaVerdict:
    layout_id: str
    label: str
    confidence: float
    kind: str
    #: How many sampled records matched each layout.
    distribution: dict[str, int] = field(default_factory=dict)
    #: Set when the records look like logs rather than training data.
    not_training_data: str | None = None
    #: Set when induction failed and we fell back to raw text.
    degraded: str | None = None
    sample_size: int = 0
    unparseable: int = 0
    #: Role strings seen, before alias mapping. Used by the role-vocabulary check.
    raw_roles: dict[str, int] = field(default_factory=dict)

    @property
    def is_mixed(self) -> bool:
        real = {k: v for k, v in self.distribution.items() if k not in ("_unknown", "_unparseable")}
        if len(real) < 2:
            return False
        total = sum(real.values())
        # A layout appearing in fewer than 2% of records is noise, not a mix.
        significant = [v for v in real.values() if v / total >= 0.02]
        return len(significant) >= 2


def _keys_of(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        return {str(k).lower() for k in payload}
    return set()


def score_record(payload: Any) -> tuple[str | None, float]:
    """Score one record against every layout. Returns the best match."""
    if not isinstance(payload, dict):
        return (None, 0.0)
    keys = _keys_of(payload)
    best: tuple[str | None, float] = (None, 0.0)

    for layout in LAYOUTS:
        req = {k.lower() for k in layout.required}
        if not req.issubset(keys):
            continue
        ok = True
        for group in layout.any_of:
            if not any(g.lower() in keys for g in group):
                ok = False
                break
        if not ok:
            continue

        score = 0.6
        opt_hits = sum(1 for o in layout.optional if o.lower() in keys)
        if layout.optional:
            score += 0.2 * (opt_hits / len(layout.optional))
        # Prefer layouts whose required keys carry actual content.
        if _has_content(payload, layout):
            score += 0.2
        if score > best[1]:
            best = (layout.layout_id, score)

    return best


def _has_content(payload: dict, layout: Layout) -> bool:
    for key in layout.required:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return True
        if isinstance(val, (list, dict)) and val:
            return True
    return False


def detect_log(payload: Any) -> str | None:
    """Return a description when this record looks like a log, not training data."""
    if not isinstance(payload, dict):
        return None
    keys = _keys_of(payload)

    type_val = payload.get("type")
    if isinstance(type_val, str) and type_val.lower() in LOG_TYPE_VALUES:
        return f"records carry type={type_val!r}"

    for signature, description in LOG_SIGNATURES:
        if signature.issubset(keys):
            return description
    return None


def induce(records: list[RawRecord], *, sample: int = 10_000) -> SchemaVerdict:
    """Work out the layout of a set of records."""
    distribution: dict[str, int] = {}
    raw_roles: dict[str, int] = {}
    log_votes: dict[str, int] = {}
    unparseable = 0
    considered = 0

    for rec in records[:sample]:
        if rec.error is not None:
            unparseable += 1
            continue
        considered += 1

        log_reason = detect_log(rec.payload)
        if log_reason:
            log_votes[log_reason] = log_votes.get(log_reason, 0) + 1

        layout_id, _score = score_record(rec.payload)
        distribution[layout_id or "_unknown"] = distribution.get(layout_id or "_unknown", 0) + 1

        if isinstance(rec.payload, dict):
            _collect_roles(rec.payload, raw_roles)

    if unparseable:
        distribution["_unparseable"] = unparseable

    verdict = SchemaVerdict(
        layout_id="unknown", label="unknown", confidence=0.0, kind="text",
        distribution=distribution, sample_size=considered, unparseable=unparseable,
        raw_roles=raw_roles,
    )

    # Logs win over layout matching. A trace record can superficially resemble a
    # text layout because it has a `content` field, and calling it training data
    # would be worse than saying nothing.
    if considered and log_votes:
        reason, votes = max(log_votes.items(), key=lambda kv: kv[1])
        if votes / considered >= 0.5:
            verdict.not_training_data = reason
            verdict.confidence = votes / considered
            return verdict

    real = {k: v for k, v in distribution.items() if not k.startswith("_")}
    if not real:
        verdict.degraded = "no known layout matched; treating every record as raw text"
        verdict.layout_id = "raw"
        verdict.label = "raw text (fallback)"
        return verdict

    layout_id = max(real, key=lambda k: real[k])
    layout = next(x for x in LAYOUTS if x.layout_id == layout_id)
    verdict.layout_id = layout_id
    verdict.label = layout.label
    verdict.kind = layout.kind
    verdict.confidence = real[layout_id] / max(considered, 1)

    # A "best" layout that only covers a minority of records is not a verdict,
    # it is a guess. Say so rather than presenting a confident label.
    if verdict.confidence < 0.5:
        unknown = distribution.get("_unknown", 0)
        verdict.degraded = (
            f"best match {layout_id!r} covers only {verdict.confidence:.0%} of records"
            + (f"; {unknown} matched no known layout" if unknown else "")
        )
    return verdict


def _collect_roles(payload: dict, out: dict[str, int]) -> None:
    for turn_key, (role_key, _content_key) in TURN_KEYS.items():
        turns = payload.get(turn_key)
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if isinstance(turn, dict):
                role = turn.get(role_key)
                if isinstance(role, str):
                    out[role] = out.get(role, 0) + 1
