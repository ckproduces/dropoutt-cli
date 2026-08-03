"""Tier 0 structural checks.

These are bugs, not quality judgements. A record whose trainable span is empty
contributes nothing to any gradient; a role string the trainer does not
recognise silently drops the record. Nothing here requires a model, and the
direction is never arguable.
"""

from __future__ import annotations

from ..context import ScanContext
from ..models import CostClass, Document, Evidence, Finding, Profile, Severity
from ..textutil import excerpt
from .base import Check, make_finding, register

CONVERSATIONAL = (Profile.SFT, Profile.PREFERENCE)


@register
class NotTrainingData(Check):
    check_id = "T0-SCHEMA-001"
    title = "Files are not training data"
    tier = 0
    unit = "dataset"
    profiles = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.FREE
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE)
    fix = "Point the scan at your dataset directory rather than a project or log directory."
    rationale = (
        "Agent session logs, telemetry and rollout traces are structurally similar enough to "
        "chat data that a naive importer will happily ingest them. Detecting them is more "
        "useful than forcing them into a layout and reporting confident nonsense about the result."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        offenders = dict(ctx.stats.get("not_training_data", {}))
        if not offenders:
            return []
        detail = "; ".join(f"{name}: {reason}" for name, reason in list(offenders.items())[:4])
        if len(offenders) > 4:
            detail += f", and {len(offenders) - 4} more"
        return [
            make_finding(
                self,
                count=len(offenders),
                total=len(ctx.datasets),
                detail=detail,
                by_dataset=dict.fromkeys(offenders, 1),
            )
        ]


@register
class MixedSchemas(Check):
    check_id = "T0-SCHEMA-002"
    title = "One folder contains several record layouts"
    tier = 0
    unit = "dataset"
    profiles = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.FREE
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Split the layouts into separate directories, or normalise them before training."
    rationale = (
        "A preparation script is written for one layout. When a folder holds three, the other "
        "two are usually dropped without a warning, because the loop that reads them just "
        "produces empty records and moves on."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        mixed = ctx.stats.get("mixed_schemas", {})
        if not mixed:
            return []
        parts = [f"{name}: {', '.join(f'{k} {v:.0%}' for k, v in dist.items())}"
                 for name, dist in list(mixed.items())[:3]]
        return [
            make_finding(
                self,
                count=len(mixed),
                total=len(ctx.datasets),
                detail="; ".join(parts),
                by_dataset=dict.fromkeys(mixed, 1),
            )
        ]


@register
class UnparseableRecords(Check):
    check_id = "T0-SCHEMA-003"
    title = "Records failed to parse"
    tier = 0
    profiles = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.FREE
    severity = Severity.WARNING
    blocking_in = (Profile.SFT, Profile.CORPUS)
    fix = "Repair or remove the malformed lines; they are silently skipped by most trainers."

    MERGE_SUM = ("count", "total")
    MERGE_COUNTS = ("by_dataset",)
    MERGE_EVIDENCE = ("evidence",)

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        self.total += 1
        if doc.meta.get("parse_error"):
            self.count += 1
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < self.EVIDENCE_CAP:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             f"{doc.meta['parse_error']} :: {excerpt(doc.text, 120)}")
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.count:
            return []
        return [
            make_finding(
                self, count=self.count, total=self.total,
                detail=f"{self.count:,} of {self.total:,} records could not be parsed as JSON",
                evidence=self.evidence, by_dataset=self.by_dataset,
            )
        ]


@register
class RoleValidity(Check):
    check_id = "T0-ROLE-001"
    title = "Conversations have a broken turn structure"
    tier = 0
    profiles = CONVERSATIONAL
    cost = CostClass.FREE
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT,)
    fix = "Drop or repair the affected records before training."
    rationale = (
        "Loss is computed on assistant spans. A record with no assistant turn contributes "
        "nothing. Consecutive same-role turns almost always indicate a flattening bug in "
        "whatever exported the data."
    )

    EVIDENCE_CAP = 6
    MERGE_SUM = ("total", "no_assistant", "consecutive", "empty_content", "ends_on_user")
    MERGE_COUNTS = ("by_dataset",)
    MERGE_EVIDENCE = ("evidence",)

    def __init__(self) -> None:
        self.total = 0
        self.no_assistant = 0
        self.consecutive = 0
        self.empty_content = 0
        self.ends_on_user = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if not doc.turns:
            return
        self.total += 1
        roles = [t.role for t in doc.turns]
        bad = False

        if "assistant" not in roles:
            self.no_assistant += 1
            bad = True
        for i in range(1, len(roles)):
            if roles[i] == roles[i - 1] and roles[i] in ("user", "assistant"):
                self.consecutive += 1
                bad = True
                break
        if any(not t.content.strip() for t in doc.turns):
            self.empty_content += 1
            bad = True
        if roles and roles[-1] == "user":
            self.ends_on_user += 1
            bad = True

        if bad:
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < self.EVIDENCE_CAP:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             f"roles={roles} :: {excerpt(doc.text, 140)}")
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        problems = []
        if self.no_assistant:
            problems.append(f"{self.no_assistant:,} with no assistant turn")
        if self.consecutive:
            problems.append(f"{self.consecutive:,} with consecutive same-role turns")
        if self.empty_content:
            problems.append(f"{self.empty_content:,} with an empty message")
        if self.ends_on_user:
            problems.append(f"{self.ends_on_user:,} ending on a user turn")
        if not problems:
            return []
        count = sum(self.by_dataset.values())
        return [
            make_finding(
                self, count=count, total=self.total,
                detail="; ".join(problems),
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={
                    "no_assistant": self.no_assistant,
                    "consecutive": self.consecutive,
                    "empty_content": self.empty_content,
                    "ends_on_user": self.ends_on_user,
                },
            )
        ]


@register
class RoleVocabulary(Check):
    check_id = "T0-ROLE-002"
    title = "Role names your trainer will not recognise"
    tier = 0
    profiles = CONVERSATIONAL
    cost = CostClass.FREE
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT,)
    fix = "Map roles to system/user/assistant before training, or confirm your trainer maps them."
    rationale = (
        "This is the quietest way to lose a whole dataset. A ShareGPT record uses "
        "from: 'gpt'. A trainer that masks on role == 'assistant' finds no assistant span, "
        "produces an all-ignored label vector, and drops the record. The counters that would "
        "have told you are frequently computed and then discarded."
    )

    MERGE_SUM = ("total", "affected")
    MERGE_COUNTS = ("seen", "by_dataset")
    MERGE_EVIDENCE = ("evidence",)

    def __init__(self) -> None:
        self.total = 0
        self.affected = 0
        self.seen: dict[str, int] = {}
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if not doc.turns:
            return
        self.total += 1
        nonstandard = [t for t in doc.turns if t.raw_role]
        if not nonstandard:
            return
        self.affected += 1
        self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
        for t in nonstandard:
            key = t.raw_role or "(missing)"
            self.seen[key] = self.seen.get(key, 0) + 1
        if len(self.evidence) < self.EVIDENCE_CAP:
            names = sorted({t.raw_role or "(missing)" for t in nonstandard})
            self.evidence.append(
                Evidence(doc.doc_id, doc.source_file, doc.source_index,
                         f"roles named {names} :: {excerpt(doc.text, 120)}")
            )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.affected:
            return []
        top = sorted(self.seen.items(), key=lambda kv: -kv[1])[:6]
        detail = "non-canonical role names: " + ", ".join(f"{k!r} ({v})" for k, v in top)
        return [
            make_finding(
                self, count=self.affected, total=self.total, detail=detail,
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={"roles": dict(top)},
            )
        ]


@register
class CoercedContent(Check):
    check_id = "T0-SCHEMA-004"
    title = "Message content was not text"
    tier = 0
    profiles = CONVERSATIONAL
    cost = CostClass.FREE
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Serialise structured content deliberately, or drop the affected records."
    rationale = (
        "When content is a dict and the pipeline falls through to str(value), the model is "
        "trained on a Python repr, single quotes and None included. It looks like data and "
        "trains like noise."
    )

    MERGE_SUM = ("total", "count")
    MERGE_COUNTS = ("by_dataset",)
    MERGE_EVIDENCE = ("evidence",)

    def __init__(self) -> None:
        self.total = 0
        self.count = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if not doc.turns:
            return
        self.total += 1
        if any(t.coerced for t in doc.turns):
            self.count += 1
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < self.EVIDENCE_CAP:
                bad = next(t for t in doc.turns if t.coerced)
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             excerpt(bad.content, 160))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.count:
            return []
        return [
            make_finding(
                self, count=self.count, total=self.total,
                detail=f"{self.count:,} records had non-string content coerced to text",
                evidence=self.evidence, by_dataset=self.by_dataset,
            )
        ]
