"""Chat template rendering and loss-mask derivation.

This is the highest-risk module in the package, because getting it subtly wrong
means reporting our own bug as the user's data problem.

Three things that are easy to get wrong and are handled explicitly here.

**`{% generation %}` is not standard Jinja.** It is a Hugging Face extension, and
a stock ``SandboxedEnvironment`` raises ``TemplateSyntaxError: unknown tag
'generation'`` on it. Qwen, Gemma and SmolLM templates all use it. We register a
real Jinja ``Extension`` that both parses the tag and records the character span
of everything it emits, which is exactly what the loss mask needs.

**The render context is larger than it looks.** Beyond ``raise_exception``,
``strftime_now`` and ``tojson``, real templates reach for ``bos_token``,
``eos_token``, ``tools``, ``documents``, ``add_generation_prompt``,
``date_string`` and Jinja's ``namespace``. A missing name produces a render
failure that looks like a data defect.

**Token offsets are not always into the input string.** Some normalizers and
pre-tokenizers report offsets into the *normalized* text. We self-check at load
time by reconstructing the input from offsets; when that fails, mask-dependent
checks are skipped with an honest reason rather than reported wrongly.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
from dataclasses import dataclass, field
from typing import Any

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment

#: Private sentinels wrapping generation blocks. Chosen from C0 controls that
#: cannot occur in sane template output, so stripping them is unambiguous.
_GEN_OPEN = "\x00\x11g\x00"
_GEN_CLOSE = "\x00\x11/g\x00"


class GenerationExtension(Extension):
    """Implements ``{% generation %} ... {% endgeneration %}``.

    The body is emitted wrapped in sentinels. A post-pass strips them and
    records the character spans, which become the trainable region.
    """

    #: Jinja reads this off the class to learn which tag the extension owns.
    #: RUF012 wants a ClassVar here and mypy forbids one: jinja2's Extension
    #: declares `tags` as an instance attribute, and a subclass may not narrow
    #: that to a class variable. The type checker is describing the base class,
    #: so it wins.
    tags: set[str] = {"generation"}  # noqa: RUF012

    def parse(self, parser):  # pragma: no cover - exercised via render
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(
            self.call_method("_wrap", []), [], [], body
        ).set_lineno(lineno)

    def _wrap(self, caller):  # pragma: no cover - exercised via render
        return f"{_GEN_OPEN}{caller()}{_GEN_CLOSE}"


def _strip_sentinels(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Remove sentinels, returning clean text and the spans they delimited."""
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    length = 0
    open_at: int | None = None
    while True:
        nxt_open = text.find(_GEN_OPEN, pos)
        nxt_close = text.find(_GEN_CLOSE, pos)
        candidates = [i for i in (nxt_open, nxt_close) if i != -1]
        if not candidates:
            break
        idx = min(candidates)
        out.append(text[pos:idx])
        length += idx - pos
        if idx == nxt_open and (nxt_close == -1 or nxt_open < nxt_close):
            open_at = length
            pos = idx + len(_GEN_OPEN)
        else:
            if open_at is not None:
                spans.append((open_at, length))
                open_at = None
            pos = idx + len(_GEN_CLOSE)
    out.append(text[pos:])
    return "".join(out), spans


def _raise_exception(message: str) -> None:
    raise TemplateRenderError(message)


def _strftime_now(fmt: str) -> str:
    return _dt.datetime.now().strftime(fmt)


def _tojson(obj: Any, indent: int | None = None) -> str:
    # Templates expect Python-compatible JSON, not Jinja's HTML-escaped variant.
    return _json.dumps(obj, ensure_ascii=False, indent=indent)


class TemplateRenderError(RuntimeError):
    """Raised by ``raise_exception`` inside a template."""


@dataclass(slots=True)
class RenderResult:
    text: str
    #: Character spans that the template marked as generated, i.e. trainable.
    generation_spans: list[tuple[int, int]] = field(default_factory=list)
    #: True when spans came from `{% generation %}` rather than the fallback.
    spans_from_tag: bool = False


@dataclass(slots=True)
class ChatTemplate:
    """A loaded chat template plus the special tokens it needs."""

    source: str
    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    unk_token: str | None = None
    additional_special_tokens: list[str] = field(default_factory=list)
    #: Which model this came from, for drift reporting.
    model_id: str | None = None
    template_hash: str = ""
    _env: Any = field(init=False, repr=False, default=None)
    _tpl: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        env = SandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[GenerationExtension],
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = _strftime_now
        env.policies["json.dumps_function"] = _tojson
        self._env = env
        self._tpl = env.from_string(self.source)

    @property
    def uses_generation_tag(self) -> bool:
        return "{% generation %}" in self.source or "{%- generation" in self.source

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: list[Any] | None = None,
    ) -> RenderResult:
        """Render messages. Raises TemplateRenderError on template failure."""
        try:
            raw = self._tpl.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                documents=None,
                bos_token=self.bos_token or "",
                eos_token=self.eos_token or "",
                pad_token=self.pad_token or "",
                unk_token=self.unk_token or "",
                additional_special_tokens=self.additional_special_tokens,
                date_string=_dt.date.today().strftime("%d %b %Y"),
            )
        except TemplateRenderError:
            raise
        except Exception as exc:
            raise TemplateRenderError(f"{type(exc).__name__}: {exc}") from exc

        clean, spans = _strip_sentinels(raw)
        if spans:
            return RenderResult(clean, spans, spans_from_tag=True)
        return RenderResult(clean, [], spans_from_tag=False)

    def spans_by_difference(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[tuple[int, int]]]:
        """Locate assistant content spans without a `{% generation %}` tag.

        Re-renders with each assistant message replaced by a unique sentinel and
        recovers the span by difference. Costs one extra render per assistant
        turn, and is only used for templates that lack the tag, such as Mistral.
        """
        full = self.render(messages).text
        spans: list[tuple[int, int]] = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            sentinel = f"\x00\x12S{i}\x00"
            alt_msgs = [dict(m) for m in messages]
            alt_msgs[i]["content"] = sentinel
            try:
                alt = self.render(alt_msgs).text
            except TemplateRenderError:
                continue
            start = alt.find(sentinel)
            if start < 0:
                continue
            tail = len(alt) - (start + len(sentinel))
            end = len(full) - tail
            if 0 <= start < end <= len(full):
                spans.append((start, end))
        return full, spans


def char_spans_to_token_mask(
    offsets: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> list[bool]:
    """Map character spans onto a token-level boolean mask."""
    mask = [False] * len(offsets)
    for start, end in spans:
        for k, (a, b) in enumerate(offsets):
            if a < end and b > start:
                mask[k] = True
    return mask


def offsets_are_reliable(text: str, offsets: list[tuple[int, int]]) -> bool:
    """Self-check that offsets index the input string rather than a normalized one.

    Some pre-tokenizers (Metaspace, Strip, Replace) report offsets into text we
    never see. Reconstructing from offsets and comparing catches that. When this
    returns False, mask-dependent checks must be skipped rather than trusted.
    """
    if not offsets:
        return False
    last_end = 0
    for a, b in offsets:
        if a < 0 or b > len(text) or a > b or a < last_end - 1:
            return False
        last_end = b
    # Reconstructed non-whitespace should match the source's non-whitespace.
    rebuilt = "".join(text[a:b] for a, b in offsets)
    return "".join(rebuilt.split()) == "".join(text.split())
