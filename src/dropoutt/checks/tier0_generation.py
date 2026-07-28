"""Tier 0 checks for data that came out of a generator.

Synthetic SFT data is now the common case, and it fails in ways that collected
data does not. The generator is a language model, so its mistakes are fluent:
the file parses, the records are well-formed, the text reads well, and something
is still wrong in a way that only shows up as a strange model months later.

Every threshold here was calibrated against a real 6,097-record Turkish
generation run rather than chosen for roundness, and the calibration is recorded
next to each one. Two of these checks are deliberately silent on that corpus —
that is the evidence they are not merely firing on everything.
"""

from __future__ import annotations

import re
from collections import Counter

from ..context import ScanContext
from ..models import CostClass, Document, Evidence, Finding, Profile, Severity
from ..textutil import excerpt
from .base import Check, make_finding, register

CONVERSATIONAL = (Profile.SFT, Profile.PREFERENCE)

#: Markers that wrap a model's private reasoning. Kept broad because the tag
#: varies by family, and the check is about consistency rather than about any
#: particular tag being right.
REASONING_TAGS = re.compile(
    r"<(think|thinking|thought|reasoning|scratchpad)>", re.IGNORECASE
)

#: A reasoning trace present in nearly all records, or nearly none, is a
#: decision. Between these bounds it is an accident: the generator was asked for
#: one thing and produced a mixture, and training on the mixture teaches the
#: model to reason unpredictably.
REASONING_MIXED_LOW = 0.02
REASONING_MIXED_HIGH = 0.98

#: Below this many records the share is too noisy to call a mixture.
REASONING_MIN_RECORDS = 200


@register
class EmbeddedRecords(Check):
    check_id = "T0-FORMAT-001"
    title = "Plain-text files are holding structured records"
    tier = 0
    profiles = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.FREE
    severity = Severity.INFO
    fix = "Rename these to .jsonl, or keep them and let dropoutt read the records."
    rationale = (
        "A .txt file holding JSON records read at face value becomes one document per file. "
        "The record count is then wrong by orders of magnitude, the profile is inferred from "
        "the wrong shape, and every conversational check is skipped without appearing in the "
        "skipped list, because as far as the scanner is concerned there were no conversations. "
        "dropoutt reads the records instead and reports that it had to."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        framings = ctx.stats.get("text_framing") or {}
        hits = {n: f for n, f in framings.items() if f.get("kind") != "prose"}
        if not hits:
            return []
        kinds = Counter(f["kind"] for f in hits.values())
        files = sum(int(f.get("files", 1)) for f in hits.values())
        shape = ", ".join(f"{k} in {n} dataset(s)" for k, n in kinds.most_common())
        return [
            make_finding(
                self,
                count=files,
                total=files,
                detail=(
                    f"{files} text file(s) hold JSON records rather than prose ({shape}); "
                    f"they were read as records, not as documents"
                ),
                by_dataset={n: int(f.get("files", 1)) for n, f in hits.items()},
                data={"framings": hits},
            )
        ]


@register
class GeneratorScaffolding(Check):
    check_id = "T0-GEN-001"
    title = "Generator scaffolding sits outside the records"
    tier = 0
    profiles = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.FREE
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Strip the text between records at generation time, and check whether it was meant to be inside them."
    rationale = (
        "Text between the records is whatever the generating model emitted around its output: "
        "a control tag it was told to honour and echoed instead, a preamble, a piece of "
        "reasoning that escaped the record. It is harmless where it sits, but it is evidence "
        "that the generator was not doing what the prompt asked, which usually means the "
        "records themselves are affected too."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        framings = ctx.stats.get("text_framing") or {}
        offenders = {
            name: f for name, f in framings.items()
            if f.get("kind") != "prose" and f.get("scaffolding")
        }
        if not offenders:
            return []
        samples: list[str] = []
        for f in offenders.values():
            for s in f["scaffolding"]:
                if s not in samples:
                    samples.append(s)
        shown = "; ".join(repr(excerpt(s, 60)) for s in samples[:3])
        return [
            make_finding(
                self,
                count=len(samples),
                total=len(framings),
                detail=f"{len(samples)} distinct run(s) of text found between records: {shown}",
                by_dataset={n: len(f["scaffolding"]) for n, f in offenders.items()},
                data={"samples": samples[:10]},
            )
        ]


@register
class UnreadKeys(Check):
    check_id = "T0-SCHEMA-005"
    title = "Record content sits in keys the layout never reads"
    tier = 0
    profiles = CONVERSATIONAL
    cost = CostClass.FREE
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Move the content into the layout's own fields, or confirm it is meant to be discarded."
    rationale = (
        "A record can be well-formed and still lose its answer. Seen in a real generation run: "
        "26 records carried a complete assistant turn in a top-level `assistant` key sitting "
        "beside `messages` rather than inside it. Every key the trainer reads was valid, so "
        "nothing complained; the conversation simply ended on a user turn and trained on "
        "nothing. Bookkeeping columns such as id, source and language are ignored here."
    )

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.keys: Counter[str] = Counter()
        self.chars = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        self.total += 1
        unread = doc.meta.get("unread_keys")
        if not unread:
            return
        self.count += 1
        self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
        for key, size in unread:
            self.keys[key] += 1
            self.chars += size
        if len(self.evidence) < 5:
            names = ", ".join(f"{k} ({n} chars)" for k, n in unread[:3])
            self.evidence.append(
                Evidence(doc.doc_id, doc.source_file, doc.source_index,
                         f"unread: {names} :: {excerpt(doc.text, 100)}")
            )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.count:
            return []
        named = ", ".join(f"{k!r} in {n}" for k, n in self.keys.most_common(3))
        return [
            make_finding(
                self,
                count=self.count,
                total=self.total,
                detail=(
                    f"{self.count} of {self.total} records carry text in keys this layout "
                    f"does not read ({named}); {self.chars:,} characters never reach the model"
                ),
                evidence=self.evidence,
                by_dataset=self.by_dataset,
                data={"keys": dict(self.keys.most_common(10))},
            )
        ]


@register
class InconsistentReasoning(Check):
    check_id = "T0-REASON-001"
    title = "Only some responses carry a reasoning trace"
    tier = 0
    profiles = CONVERSATIONAL
    cost = CostClass.FREE
    severity = Severity.WARNING
    blocking_in = ()
    fix = "Decide whether this run is a reasoning model or not, and make every record agree."
    rationale = (
        "Measured on a real generation run: 11.6% of assistant turns opened with a <think> "
        "block and 88.4% did not. Nothing in the pipeline noticed, because both shapes are "
        "valid records. Trained on as-is, the model learns to emit reasoning about one time "
        "in eight, unpredictably, and inference-time parsers that strip the block find nothing "
        "to strip in most responses. A dataset that is 100% reasoning or 0% reasoning is a "
        "decision; a dataset that is 12% reasoning is an accident."
    )

    def __init__(self) -> None:
        self.with_trace = 0
        self.total = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        answer = doc.assistant_text
        if not answer.strip():
            return
        self.total += 1
        if REASONING_TAGS.search(answer):
            self.with_trace += 1
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 3:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             excerpt(answer, 140))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if self.total < REASONING_MIN_RECORDS or not self.with_trace:
            return []
        share = self.with_trace / self.total
        if not (REASONING_MIXED_LOW <= share <= REASONING_MIXED_HIGH):
            return []
        minority = "with" if share < 0.5 else "without"
        n = self.with_trace if share < 0.5 else self.total - self.with_trace
        return [
            make_finding(
                self,
                count=n,
                total=self.total,
                detail=(
                    f"{share:.1%} of responses open a reasoning block and {1 - share:.1%} do "
                    f"not; the {minority}-reasoning group is {n:,} records"
                ),
                evidence=self.evidence,
                by_dataset=self.by_dataset,
                data={"share_with_reasoning": round(share, 4)},
            )
        ]


@register
class ResponseLengthCap(Check):
    check_id = "T0-TRUNC-002"
    title = "Responses stop at a generation length cap"
    tier = 0
    profiles = CONVERSATIONAL
    cost = CostClass.FREE
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Regenerate the affected records with a higher max_tokens, or drop them."
    rationale = (
        "When a generation run hits max_tokens the response stops mid-sentence, and the record "
        "still looks fine. The signature is not that responses are long: it is that an "
        "implausible number of them are the *same* length, because they all hit the same "
        "ceiling. A pile-up in the top length bucket is what this looks for, which is why it "
        "stays silent on data whose lengths are merely varied and long."
    )

    #: Share of responses landing in the top length bucket before a cap is
    #: suspected. A natural length distribution has a long thin tail; a capped
    #: one has a wall. Measured against an uncapped 5,968-response corpus, whose
    #: most common exact length held 1.2% of records.
    PILEUP_SHARE = 0.05
    #: Width of the bucket, in characters. Wide enough to absorb the last word
    #: varying, narrow enough that a real distribution does not fill it.
    BUCKET = 25
    #: How many times the bucket below it the top bucket must hold. Share alone
    #: is not enough: a corpus whose responses take only a dozen discrete
    #: lengths puts a large share in its top bucket without anything being
    #: truncated, and an earlier version of this check reported exactly that as
    #: a cap. A real ceiling is a *wall* — everything that would have been
    #: longer is piled against it, while the bucket below holds only the
    #: responses that naturally ended there.
    WALL_RATIO = 3.0
    #: How far below the ceiling to look for that comparison, in buckets.
    WINDOW_BUCKETS = 10
    #: A ceiling shorter than this is not a generation cap. A classification set
    #: whose every answer is one of three labels puts 100% of responses in its
    #: top bucket with nothing beneath, and passes both tests above — but nobody
    #: sets max_tokens to twelve characters, and calling that truncation would
    #: be wrong about a perfectly good dataset.
    MIN_CAP_CHARS = 200
    MIN_RECORDS = 200

    def __init__(self) -> None:
        self.lengths: list[int] = []
        self.evidence: list[Evidence] = []
        self.longest: list[tuple[int, Document]] = []

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        answer = doc.assistant_text.rstrip()
        if not answer:
            return
        self.lengths.append(len(answer))
        if len(self.longest) < 400:
            self.longest.append((len(answer), doc))

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if len(self.lengths) < self.MIN_RECORDS:
            return []
        top = max(self.lengths)
        if top < self.MIN_CAP_CHARS:
            return []
        # Only the ceiling matters. A pile-up in the middle of the distribution
        # is a template, not a cap, and belongs to a different check.
        in_bucket = sum(1 for n in self.lengths if n > top - self.BUCKET)
        share = in_bucket / len(self.lengths)
        if share < self.PILEUP_SHARE:
            return []

        # The wall test, measured over a window rather than against the single
        # adjacent bucket. Without it this fires on any distribution with few
        # distinct lengths, where the top bucket is large simply because every
        # bucket is; against only the adjacent bucket it still fires, because a
        # coarsely spaced distribution leaves that one empty. What distinguishes
        # a cap is that the top bucket towers over *everything* near it.
        below = [
            sum(1 for n in self.lengths
                if top - (k + 1) * self.BUCKET < n <= top - k * self.BUCKET)
            for k in range(1, self.WINDOW_BUCKETS + 1)
        ]
        tallest_below = max(below, default=0)
        if in_bucket < tallest_below * self.WALL_RATIO:
            return []
        for n, doc in sorted(self.longest, key=lambda kv: -kv[0])[:3]:
            if n > top - self.BUCKET:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             "ends: …" + excerpt(doc.assistant_text.rstrip()[-90:], 90))
                )
        return [
            make_finding(
                self,
                count=in_bucket,
                total=len(self.lengths),
                detail=(
                    f"{share:.1%} of responses end within {self.BUCKET} characters of the "
                    f"longest ({top:,}), against at most {tallest_below} in any bucket "
                    f"beneath it; that is a ceiling, not a distribution"
                ),
                evidence=self.evidence,
                data={"cap_chars": top, "pileup_share": round(share, 4),
                      "tallest_bucket_below": tallest_below},
            )
        ]
