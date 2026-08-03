"""One reading of a scan, shared by the terminal and the HTML report.

Both reports used to build their own view of the same result, which is how they
drifted: the terminal called something a gap and the page called it a coverage
area, and neither said what the reader should do about it. Everything either of
them shows is decided here, once. The templates lay it out; they do not decide
what it says.

Two editorial rules live in this module rather than in the templates.

**Failures lead.** A scan is read by someone deciding whether to start a
training run. The first thing on the page is what would go wrong, ordered by how
much of the corpus it touches, in a sentence that names the consequence rather
than the measurement.

**A number is never shown without its denominator or its meaning.** "15,523" is
not a finding. "One response in three repeats itself" is.

What the map says is decided in :mod:`dropoutt.report.atlas_story`, and how a
number becomes a phrase in :mod:`dropoutt.report.phrasing`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Evidence, Severity
from .atlas_story import AtlasStory, build_story
from .phrasing import count as _tokens
from .phrasing import in_words
from .phrasing import plural as _plural
from .phrasing import share as _share


@dataclass
class Problem:
    """One finding, rewritten for someone who has not read the check catalog."""

    check_id: str
    title: str
    severity: Severity
    is_blocking: bool
    would_block: tuple[str, ...]
    affected: int
    considered: int
    share: float
    tokens: int | None
    detail: str
    fix: str
    unit: str = "record"
    total_unit: str = "record"
    evidence: list[Evidence] = field(default_factory=list)
    by_dataset: dict[str, int] = field(default_factory=dict)

    @property
    def about_records(self) -> bool:
        """Whether this check's count is a share of the corpus.

        A finding that counts datasets or areas of the map has a percentage too,
        and it means something entirely different. Sorting "1 of 1 datasets have
        no licence" above "49,872 of 50,000 records are duplicates" because both
        read as 100% is exactly the kind of thing that makes a report ignored.
        """
        return self.unit in ("record", "response") and self.total_unit in ("record", "response")

    @property
    def scale(self) -> str:
        """The size of the problem, in the reader's units."""
        if not self.affected:
            return ""
        noun = _plural(self.unit if self.unit == self.total_unit else self.total_unit)
        if self.considered:
            head = f"{self.affected:,} of {self.considered:,} {noun}"
            if self.unit != self.total_unit:
                head = f"{self.affected:,} {_plural(self.unit)} across {self.considered:,} {noun}"
            elif self.about_records:
                head += f" · {_share(self.share)}"
            return head
        return f"{self.affected:,} {_plural(self.unit)}"

    @property
    def cost(self) -> str:
        if not self.tokens:
            return ""
        return f"{_tokens(self.tokens)} tokens"

    @property
    def rank(self) -> tuple[int, float, int]:
        """Reading order: most critical first, and size decides ties.

        Consequence is the outer key. What stops this run leads, then what would
        stop a run under some declared target, then warnings, then the rest — a
        page that puts a warning above five findings badged "would block" is
        read as an ordering mistake whatever the arithmetic behind it was, and
        the reader stops trusting the order.

        Size decides within a tier, and only there. A warning on a third of the
        corpus still leads the other warnings; it no longer leads the findings
        that fail the run.
        """
        share = self.share if self.about_records else 0.0
        if self.is_blocking:
            tier = 0
        elif self.severity is Severity.BLOCKING or self.would_block:
            tier = 1
        elif self.severity is Severity.WARNING:
            tier = 2
        else:
            tier = 3
        return (tier, -share, -(self.tokens or 0))



@dataclass
class Composition:
    """What is in the files, before any question of what is wrong with it.

    This is the half of a scan report that is not a complaint. A reader who has
    just been handed a folder wants to know what it *is* — what language, what
    shape, whether it is already formatted — and the previous report made them
    infer all of it from the findings, which only mention a property when it is
    broken.
    """

    #: (layout label, share of records, confidence) per distinct layout found.
    layouts: list[tuple[str, float, float]] = field(default_factory=list)
    #: Records that parsed into a recognised training layout, as a share.
    structured_share: float = 0.0
    structured_line: str = ""
    #: Chat template families already baked into the text, largest first.
    templates: list[tuple[str, int, float]] = field(default_factory=list)
    template_target: str = ""
    #: Records the layout matcher could not read at all.
    unparseable_share: float = 0.0
    #: Mean characters per record, and the corpus total.
    mean_chars: int = 0
    total_chars: int = 0
    #: Datasets carrying no licence, which is a publishing question not a bug.
    unlicensed: int = 0


@dataclass
class ScanSummary:
    verdict: str = ""
    tone: str = "clean"
    subtitle: str = ""
    records: int = 0
    datasets: int = 0
    files: int = 0
    profile: str = "unknown"
    languages: list[tuple[str, float]] = field(default_factory=list)
    language_line: str = ""
    tokens: int | None = None
    tokens_line: str = ""
    token_margin: float = 0.0
    composition: Composition = field(default_factory=Composition)
    problems: list[Problem] = field(default_factory=list)
    notes: list[Problem] = field(default_factory=list)
    atlas: AtlasStory | None = None
    blocking_enabled: bool = False
    elapsed: float = 0.0

    @property
    def worst(self) -> list[Problem]:
        return self.problems[:3]

    @property
    def blocking(self) -> list[Problem]:
        return [p for p in self.problems if p.is_blocking]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if not p.is_blocking]


def budget_rows(budget) -> list[dict]:
    """The tokenizer panel, cheapest first, as all three reports show it."""
    if budget is None or not budget.estimates:
        return []
    return [
        {
            "name": est.name,
            "total": est.total_tokens_est,
            "margin_share": est.margin_share,
            "premium": budget.premium_vs_cheapest(est),
        }
        for est in sorted(budget.estimates, key=lambda e: e.total_tokens_est)
        if not est.failed
    ]


def budget_method(budget, *, exact: bool) -> str:
    """How the token number was arrived at, in one sentence.

    Worth the space: it is the number most likely to be quoted out of the report
    and into a capacity plan, and "about 4 billion tokens" means something
    different depending on whether it was counted or inferred.
    """
    if exact:
        return (
            "Every record was tokenized with the model you passed, so these are "
            "counts rather than estimates."
        )
    if budget is None:
        return ""
    return (
        f"Estimated from {budget.sample_size:,} sampled records against an exact "
        f"character count. Each dataset is priced at its own measured "
        f"tokens-per-character and the totals added, rather than the corpus being "
        f"charged at one blended rate — on a corpus whose datasets differ in "
        f"language that distinction is worth tens of percent. The ± column is the "
        f"sampling error at 95% confidence."
    )




# --------------------------------------------------------------------------


def build(result, *, budget=None, include_evidence: bool = True) -> ScanSummary:
    """Read a scan result into the shape both reports render."""
    ctx = result.ctx
    summary = ScanSummary(
        records=result.records_scanned,
        datasets=len(result.discovery.datasets),
        files=len(result.discovery.files),
        profile=ctx.profile.value,
        blocking_enabled=ctx.blocking_enabled,
        elapsed=result.elapsed,
    )

    from ..checks.base import REGISTRY

    for finding in result.findings:
        check = REGISTRY.get(finding.check_id)
        unit, total_unit = check.units() if check is not None else ("record", "record")
        problem = Problem(
            check_id=finding.check_id,
            title=finding.title,
            severity=finding.severity,
            is_blocking=finding.is_blocking,
            would_block=finding.would_block_under,
            affected=finding.count,
            considered=finding.total_considered,
            share=finding.rate,
            tokens=finding.wasted_tokens,
            detail=finding.detail,
            fix=finding.fix,
            unit=unit,
            total_unit=total_unit,
            evidence=list(finding.evidence) if include_evidence else [],
            by_dataset=dict(finding.by_dataset),
        )
        if problem.severity is Severity.INFO and not problem.would_block:
            summary.notes.append(problem)
        else:
            summary.problems.append(problem)
    summary.problems.sort(key=lambda p: p.rank)
    summary.notes.sort(key=lambda p: p.rank)

    summary.languages = _languages(result)
    if summary.languages:
        summary.language_line = ", ".join(
            f"{code} {_share(share)}" for code, share in summary.languages[:3]
        )

    if budget is not None and budget.estimates:
        cheapest = budget.cheapest
        if cheapest is not None:
            summary.tokens = cheapest.total_tokens_est
            summary.token_margin = cheapest.margin_share
            spread = [e for e in budget.estimates if not e.failed]
            summary.tokens_line = (
                f"about {_tokens(cheapest.total_tokens_est)} tokens under "
                f"{cheapest.name}"
            )
            if len(spread) > 1:
                dearest = max(spread, key=lambda e: e.total_tokens_est)
                premium = budget.premium_vs_cheapest(dearest)
                if premium > 0.02:
                    summary.tokens_line += (
                        f", {premium:.0%} more under {dearest.name}"
                    )

    summary.composition = _composition(result)
    summary.atlas = build_story(result)
    summary.verdict, summary.tone, summary.subtitle = _verdict(summary)
    return summary


#: Layout kinds that mean "this record is a training example with structure the
#: tool understood", as opposed to a blob of text it will treat as one field.
STRUCTURED_KINDS = ("chat", "instruction", "preference")


def _composition(result) -> Composition:
    """What the files contain, read off the same pass that produced the findings."""
    comp = Composition()
    ctx = result.ctx
    comp.total_chars = int(ctx.stats.get("total_chars", 0))
    if result.records_scanned:
        comp.mean_chars = round(comp.total_chars / result.records_scanned)

    # -- layouts, weighted by how many records each dataset actually holds ---
    counts: dict[str, list[float]] = {}
    structured = 0
    unparseable = 0
    total = 0
    for dataset in ctx.datasets:
        verdict = result.verdicts.get(dataset.name)
        if verdict is None:
            continue
        n = dataset.record_count or 0
        total += n
        label = verdict.label or verdict.layout_id
        row = counts.setdefault(label, [0.0, 0.0])
        row[0] += n
        # Confidence is per dataset; the reported one is the weakest, because a
        # reader deciding whether to trust the layout column cares about the
        # dataset the tool was least sure of, not the average.
        row[1] = verdict.confidence if row[1] == 0.0 else min(row[1], verdict.confidence)
        if verdict.kind in STRUCTURED_KINDS:
            structured += n
        if verdict.sample_size:
            unparseable += n * (verdict.unparseable / verdict.sample_size)

    if total:
        comp.layouts = sorted(
            ((label, n / total, conf) for label, (n, conf) in counts.items()),
            key=lambda row: -row[1],
        )
        comp.structured_share = structured / total
        comp.unparseable_share = unparseable / total
    comp.structured_line = _structured_line(comp)

    # -- chat templates already baked into the text -------------------------
    for finding in result.findings:
        if finding.check_id != "T0-TMPL-001":
            continue
        families = finding.data.get("families") or {}
        seen = finding.total_considered or 1
        comp.templates = sorted(
            ((name, int(n), int(n) / seen) for name, n in families.items()),
            key=lambda row: -row[1],
        )
        comp.template_target = str(finding.data.get("target") or "")

    comp.unlicensed = sum(1 for d in ctx.datasets if not d.license)
    return comp


def _structured_line(comp: Composition) -> str:
    """One sentence on how much of the corpus had structure the tool could read."""
    if not comp.layouts:
        return ""
    share = comp.structured_share
    if share >= 0.995:
        line = "Every record parsed into a training layout."
    elif share <= 0.005:
        # Nothing to say. The layout table directly below already reads "bare
        # text", and a sentence restating it is a sentence the reader learns to
        # skip — taking the one above it, which is not a restatement, with it.
        line = ""
    else:
        line = (
            f"{_share(share)} of records parsed into a structured training "
            f"layout; the rest are treated as plain text."
        )
    if comp.unparseable_share >= 0.01:
        line += (
            f" {_share(comp.unparseable_share)} could not be read at all and "
            f"were skipped."
        )
    return line.strip()


def _languages(result) -> list[tuple[str, float]]:
    for finding in result.findings:
        if finding.check_id != "T1-LANG-001":
            continue
        composition = finding.data.get("composition", {})
        total = sum(composition.values()) or 1
        return [
            (code, count / total)
            for code, count in sorted(composition.items(), key=lambda kv: -kv[1])
        ]
    return []


def _verdict(summary: ScanSummary) -> tuple[str, str, str]:
    """One sentence at the top, and the tone it is said in."""
    blocking = [p for p in summary.problems if p.is_blocking]
    would = [p for p in summary.problems if p.would_block and not p.is_blocking]
    warnings = [p for p in summary.problems if not p.would_block]

    if blocking or would:
        serious = blocking or would
        noun = "problems" if len(serious) > 1 else "problem"
        verb = "will fail this run" if blocking else "would fail a fine-tuning run"
        return (
            f"{len(serious)} {noun} {verb}",
            "block",
            _lead(serious[0], [p for p in summary.problems if p not in serious]),
        )
    if warnings:
        noun = "things" if len(warnings) > 1 else "thing"
        return (
            f"Nothing blocking, {len(warnings)} {noun} worth fixing",
            "warn",
            _lead(warnings[0], warnings[1:]),
        )
    return ("Nothing to fix", "clean",
            "Every check that could run found nothing worth reporting.")


def _lead(first: Problem, rest: list[Problem]) -> str:
    """Name the problem the headline is about, then size up what is left.

    The headline counts what would stop a run; this names it. Pointing at the
    largest finding instead would be a non sequitur whenever the largest finding
    is not one of the blocking ones, which is most of the time.
    """
    lead = first.title
    if first.scale:
        lead += f" — {first.scale.split(' · ')[0]}"
    lead += "."
    if not rest:
        return lead
    biggest = min(rest, key=lambda p: p.rank)
    noun = "things" if len(rest) > 1 else "thing"
    tail = f" {len(rest)} other {noun} worth fixing"
    if biggest.about_records and biggest.share > 0:
        tail += (
            f"; the largest is “{biggest.title.lower()}”, "
            f"affecting {in_words(biggest.share)}"
        )
    return lead + tail + "."
