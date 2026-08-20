"""Check protocol and registry.

Every check is an object with two hooks:

``observe(doc, ctx)``
    Called once per record, in a single streaming pass. Checks accumulate
    whatever state they need here. This is what keeps 20-plus checks to one
    pass over the data rather than twenty.

``finalize(ctx)``
    Called once at the end, returning findings. Checks that need global state
    (near-duplicate clusters, the cross-dataset overlap matrix, contamination
    attribution) do their resolution here, which is the second phase.

A check declares what it needs via ``requires``. The runner uses that to decide
what can run, and to report everything else as skipped alongside the single flag
that would unlock it. That reporting is a feature, not an apology: it is the
progressive-disclosure ladder made visible.

``merge(other)``
    Fold a second instance of the same check, which observed a later contiguous
    slice of the same corpus, into this one. This is what lets the streaming
    pass be split across processes: each worker runs the real check over its
    shard and the parent combines the results, so the checks themselves need no
    knowledge that a shard exists.

    Almost every check is a bag of counters, count-maps and a capped evidence
    list, so it only has to *name* its state through the ``MERGE_*`` class
    attributes and the generic implementation below does the rest. Checks whose
    state does not reduce that way override ``merge``. Nothing is merged
    implicitly: a check that declares nothing and overrides nothing fails the
    coverage test in ``tests/test_merge.py``, because a silently unmerged
    counter would make a parallel scan quietly report smaller numbers than a
    serial one.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..models import (
    Confidence,
    CostClass,
    Document,
    Finding,
    Profile,
    Requirement,
    Severity,
    SkippedCheck,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..context import ScanContext


class Check:
    """Base class for all checks.

    Subclasses set the class attributes and override ``observe`` and/or
    ``finalize``.
    """

    #: Stable identifier, ``T{tier}-{GROUP}-{nnn}``. Never renumber these:
    #: users mute checks by id and those mutes live in version control.
    check_id: str = ""
    title: str = ""
    tier: int = 0
    profiles: tuple[Profile, ...] = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE)
    requires: tuple[Requirement, ...] = ()
    cost: CostClass = CostClass.FREE
    severity: Severity = Severity.WARNING
    #: v0.1 ships nothing verified. See models.Confidence for why that matters.
    confidence: Confidence = Confidence.UNVERIFIED
    #: One line telling the user what to do about it.
    fix: str = ""
    #: Longer explanation shown in `dropoutt checks show <id>`.
    rationale: str = ""
    #: Profiles under which a finding from this check would fail a run.
    blocking_in: tuple[Profile, ...] = ()

    #: What ``count`` counts, singular. Most checks count records, but some
    #: count datasets, files or areas of the atlas, and a report that sorts
    #: "1 of 1 datasets" against "49,872 of 50,000 records" by percentage puts
    #: the wrong thing first. ``total_unit`` differs only where the numerator
    #: and denominator are different things — distinct prompts out of records,
    #: for instance.
    unit: str = "record"
    total_unit: str = ""

    @classmethod
    def units(cls) -> tuple[str, str]:
        return cls.unit, cls.total_unit or cls.unit

    #: How many examples this check keeps. Read by ``observe`` and by the
    #: generic merge, so a sharded scan truncates at the same point a serial one
    #: does rather than at a multiple of it.
    EVIDENCE_CAP: int = 5

    #: Integer counters, summed.
    MERGE_SUM: tuple[str, ...] = ()
    #: ``dict[key, int]``, values summed.
    MERGE_COUNTS: tuple[str, ...] = ()
    #: ``dict[key, dict[key, int]]``, inner values summed.
    MERGE_NESTED: tuple[str, ...] = ()
    #: Lists concatenated in shard order and cut to ``EVIDENCE_CAP``.
    MERGE_EVIDENCE: tuple[str, ...] = ()
    #: Lists concatenated in shard order with no cap.
    MERGE_CONCAT: tuple[str, ...] = ()
    #: Dicts where the earlier shard's value wins, since it saw the earlier
    #: record. Optionally capped by ``MERGE_FIRST_CAP``.
    MERGE_FIRST: tuple[str, ...] = ()
    MERGE_FIRST_CAP: int | None = None
    #: Handled by an overriding ``merge``, because no per-attribute rule fits.
    MERGE_CUSTOM: tuple[str, ...] = ()
    #: Static configuration, or state derived in ``finalize`` from something
    #: that *is* merged. Named so the coverage test can tell "handled" from
    #: "forgotten".
    MERGE_IGNORE: tuple[str, ...] = ()

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        """Called once per record."""

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        """Called once after all records. Returns any findings."""
        return []

    def merge(self, other: Check) -> None:
        """Fold a later shard of the same corpus into this instance."""
        for name in self.MERGE_SUM:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in self.MERGE_COUNTS:
            target = getattr(self, name)
            for key, value in getattr(other, name).items():
                target[key] = target.get(key, 0) + value
        for name in self.MERGE_NESTED:
            target = getattr(self, name)
            for key, inner in getattr(other, name).items():
                bucket = target.setdefault(key, {})
                for sub, value in inner.items():
                    bucket[sub] = bucket.get(sub, 0) + value
        for name in self.MERGE_EVIDENCE:
            target = getattr(self, name)
            room = self.EVIDENCE_CAP - len(target)
            if room > 0:
                target.extend(getattr(other, name)[:room])
        for name in self.MERGE_CONCAT:
            getattr(self, name).extend(getattr(other, name))
        for name in self.MERGE_FIRST:
            target = getattr(self, name)
            cap = self.MERGE_FIRST_CAP
            for key, value in getattr(other, name).items():
                if key in target:
                    continue
                if cap is not None and len(target) >= cap:
                    break
                target[key] = value

    # -- helpers ---------------------------------------------------------

    @classmethod
    def applies_to(cls, profile: Profile) -> bool:
        return profile in cls.profiles

    @classmethod
    def missing_requirements(cls, ctx: ScanContext) -> list[Requirement]:
        return [r for r in cls.requires if not ctx.has(r)]


class Registry:
    """Holds every known check and resolves which ones can run."""

    def __init__(self) -> None:
        self._checks: dict[str, type[Check]] = {}

    def register(self, cls: type[Check]) -> type[Check]:
        if not cls.check_id:
            raise ValueError(f"{cls.__name__} has no check_id")
        if cls.check_id in self._checks:
            raise ValueError(f"duplicate check_id {cls.check_id}")
        self._checks[cls.check_id] = cls
        return cls

    def all(self) -> list[type[Check]]:
        return sorted(self._checks.values(), key=lambda c: c.check_id)

    def get(self, check_id: str) -> type[Check] | None:
        return self._checks.get(check_id)

    def resolve(
        self,
        ctx: ScanContext,
        *,
        max_tier: int = 1,
        muted: Iterable[str] = (),
    ) -> tuple[list[Check], list[SkippedCheck]]:
        """Split the catalog into what will run and what will not, with reasons."""
        muted_set = set(muted)
        active: list[Check] = []
        skipped: list[SkippedCheck] = []

        for cls in self.all():
            if cls.check_id in muted_set:
                skipped.append(
                    SkippedCheck(
                        cls.check_id, cls.title, "muted in dropoutt.toml", "remove from [mute]"
                    )
                )
                continue
            if cls.tier > max_tier:
                skipped.append(
                    SkippedCheck(
                        cls.check_id,
                        cls.title,
                        f"tier {cls.tier} not enabled",
                        f"--tier {cls.tier}",
                    )
                )
                continue
            if not cls.applies_to(ctx.profile) and ctx.profile is not Profile.UNKNOWN:
                continue
            missing = [r for r in cls.requires if not ctx.has(r)]
            if missing:
                reason, unlock = _explain(missing[0])
                skipped.append(SkippedCheck(cls.check_id, cls.title, reason, unlock))
                continue
            active.append(cls())

        return active, skipped


def _explain(req: Requirement) -> tuple[str, str]:
    """Map a missing requirement to a reason and the flag that fixes it."""
    return {
        Requirement.TOKENIZER: (
            "needs a tokenizer",
            "pass --model, e.g. --model Qwen/Qwen3-8B",
        ),
        Requirement.CHAT_TEMPLATE: (
            "needs the target model's chat template",
            "pass --model",
        ),
        Requirement.SEQ_LEN: (
            "needs a sequence length",
            "pass --seq-len 4096 (or --model to infer it)",
        ),
        Requirement.LANGID: (
            "language identification backend not installed",
            "reinstall dropoutt",
        ),
        Requirement.EMBEDDINGS: (
            "embedding backend not installed",
            "reinstall dropoutt",
        ),
        Requirement.ATLAS: (
            "no atlas available",
            "run dropoutt fetch, or reinstall dropoutt",
        ),
        Requirement.CONTAMINATION_INDEX: (
            "no benchmark indices found",
            "reinstall dropoutt; the benchmark indices ship inside the package",
        ),
        Requirement.MULTIPLE_DATASETS: (
            "only one dataset was discovered",
            "point at a folder containing several datasets",
        ),
        Requirement.NONE: ("", ""),
    }[req]


#: The global registry. Check modules register into this at import time.
REGISTRY = Registry()


def register(cls: type[Check]) -> type[Check]:
    return REGISTRY.register(cls)


def make_finding(
    check: Check,
    *,
    count: int,
    total: int,
    detail: str,
    evidence: list[Any] | None = None,
    wasted_tokens: int | None = None,
    by_dataset: dict[str, int] | None = None,
    data: dict[str, Any] | None = None,
    severity: Severity | None = None,
) -> Finding:
    """Build a finding from a check's own metadata, so ids and fixes stay in sync."""
    effective = severity or check.severity
    # Only advertise "would block under X" when this finding is actually at
    # blocking severity. A check that can block sometimes should not carry the
    # label when it downgraded itself for this particular result.
    blocks = tuple(p.value for p in check.blocking_in) if effective is Severity.BLOCKING else ()
    return Finding(
        check_id=check.check_id,
        title=check.title,
        severity=effective,
        confidence=check.confidence,
        count=count,
        total_considered=total,
        detail=detail,
        fix=check.fix,
        evidence=evidence or [],
        wasted_tokens=wasted_tokens,
        by_dataset=by_dataset or {},
        data=data or {},
        would_block_under=blocks,
    )
