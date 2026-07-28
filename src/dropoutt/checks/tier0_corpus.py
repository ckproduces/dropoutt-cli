"""Tier 0 quality filters for the corpus profile.

Until now the corpus profile got encoding checks, duplicate detection and
little else, which meant that pointing dropoutt at a pretraining corpus produced
a report about its punctuation and not about whether the text was usable. These
are the three line-shape filters FineWeb publishes, at FineWeb's own thresholds.

They are reported, never applied. FineWeb's own ablations are the reason: the
filters were selected by measuring downstream effect on their corpus, and a
threshold that helped there is a hypothesis here, not a verdict. So each finding
states the share of documents that would be dropped and leaves the decision
where it belongs. Every one of them is ``UNVERIFIED`` for exactly this reason.

The shared per-document line statistics are computed once, in
:class:`LineShape`, and read by all three checks. Three passes over every line of
a pretraining corpus to answer three questions about the same lines would be a
waste of the streaming design.
"""

from __future__ import annotations

from ..context import ScanContext
from ..models import CostClass, Document, Evidence, Finding, Profile, Severity
from ..textutil import excerpt
from .base import Check, make_finding, register

#: FineWeb's published thresholds. A document failing one of these is dropped by
#: their pipeline; here it is only counted.
LINE_PUNCT_MIN = 0.12
SHORT_LINE_MAX = 0.67
SHORT_LINE_CHARS = 30
DUP_LINE_CHAR_MAX = 0.01

#: Characters that end a sentence in the scripts this tool targets. The Turkish
#: set adds nothing beyond Latin punctuation, but Arabic and CJK full stops are
#: distinct code points and a Latin-only test would mark those corpora as
#: entirely unpunctuated.
TERMINAL_PUNCT = ".!?…:;\"')]}۔،؟。！？」』"

#: Documents shorter than this are not worth a line-shape verdict: one line of
#: forty characters fails or passes on nothing.
MIN_DOC_CHARS = 200

#: A corpus needs this many documents before a share is worth reporting.
MIN_DOCS = 50


def line_shape(text: str) -> tuple[float, float, float] | None:
    """Return (punctuation share, short-line share, duplicated-line char share).

    ``None`` when the document is too short for the numbers to mean anything.
    """
    if len(text) < MIN_DOC_CHARS:
        return None
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None

    punct = sum(1 for ln in lines if ln[-1] in TERMINAL_PUNCT) / len(lines)
    short = sum(1 for ln in lines if len(ln) < SHORT_LINE_CHARS) / len(lines)

    seen: set[str] = set()
    dup_chars = 0
    for ln in lines:
        if ln in seen:
            dup_chars += len(ln)
        else:
            seen.add(ln)
    total_chars = sum(len(ln) for ln in lines) or 1
    return punct, short, dup_chars / total_chars


class _LineShapeCheck(Check):
    """Shared plumbing for the three filters. Not registered on its own."""

    tier = 0
    profiles = (Profile.CORPUS,)
    cost = CostClass.CHEAP
    severity = Severity.INFO

    #: Index into the line_shape tuple, and which direction fails.
    metric = 0
    threshold = 0.0
    fails_above = True

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        shape = doc.meta.get("_line_shape")
        if shape is None:
            if "_line_shape" in doc.meta:
                return
            shape = line_shape(doc.text)
            doc.meta["_line_shape"] = shape
            if shape is None:
                return
        self.total += 1
        value = shape[self.metric]
        failed = value > self.threshold if self.fails_above else value < self.threshold
        if not failed:
            return
        self.count += 1
        self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
        if len(self.evidence) < 4:
            self.evidence.append(
                Evidence(doc.doc_id, doc.source_file, doc.source_index,
                         f"{value:.2f} :: {excerpt(doc.text, 120)}")
            )

    def _finding(self, ctx: ScanContext, detail: str) -> list[Finding]:
        if not self.count or self.total < MIN_DOCS:
            return []
        return [
            make_finding(
                self, count=self.count, total=self.total, detail=detail,
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={"threshold": self.threshold, "share": round(self.count / self.total, 4)},
            )
        ]


@register
class UnpunctuatedLines(_LineShapeCheck):
    check_id = "T0-QUAL-001"
    title = "Documents whose lines mostly do not end in punctuation"
    metric = 0
    threshold = LINE_PUNCT_MIN
    fails_above = False
    fix = "Check the extractor: this is the shape of navigation, menus and link lists."
    rationale = (
        "Prose ends its lines with punctuation. Text that does not is usually not prose at "
        "all: it is a nav bar, a table of contents, a product grid or a tag cloud that the "
        "HTML extractor flattened into lines. FineWeb drops documents below 0.12 on this "
        "measure. dropoutt counts them and leaves the decision to you, because that threshold "
        "was tuned on FineWeb's corpus and not on yours."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        return self._finding(
            ctx,
            f"{self.count:,} of {self.total:,} documents "
            f"({self.count / max(self.total, 1):.1%}) have under {LINE_PUNCT_MIN:.0%} of "
            f"lines ending in punctuation",
        )


@register
class ShortLineDocuments(_LineShapeCheck):
    check_id = "T0-QUAL-002"
    title = "Documents built mostly from very short lines"
    metric = 1
    threshold = SHORT_LINE_MAX
    fails_above = True
    fix = "Check the extractor: short-line documents are usually lists rather than text."
    rationale = (
        f"A document where more than {SHORT_LINE_MAX:.0%} of lines are under "
        f"{SHORT_LINE_CHARS} characters is a list, a table or a menu. FineWeb drops these. "
        "The failure mode this catches is an extractor that produced structurally valid "
        "output from a page that never held an article."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        return self._finding(
            ctx,
            f"{self.count:,} of {self.total:,} documents "
            f"({self.count / max(self.total, 1):.1%}) are over {SHORT_LINE_MAX:.0%} short "
            f"lines (under {SHORT_LINE_CHARS} characters)",
        )


@register
class DuplicatedLines(_LineShapeCheck):
    check_id = "T0-QUAL-003"
    title = "Documents repeating their own lines"
    metric = 2
    threshold = DUP_LINE_CHAR_MAX
    fails_above = True
    fix = "Check the extractor for repeated boilerplate, and consider within-document dedup."
    rationale = (
        "Repetition inside a single document is a different problem from duplication across a "
        "corpus, and the corpus-level near-duplicate check will not find it. It is what a "
        "page footer repeated per section, a paginated comment thread, or a generation loop "
        "looks like. FineWeb drops documents above 1% of characters in repeated lines."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        return self._finding(
            ctx,
            f"{self.count:,} of {self.total:,} documents "
            f"({self.count / max(self.total, 1):.1%}) carry over "
            f"{DUP_LINE_CHAR_MAX:.0%} of their characters in repeated lines",
        )
