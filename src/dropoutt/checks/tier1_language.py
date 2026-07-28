"""Tier 1 language checks.

Most training data carries no language tag, so "wrong language label" is not the
useful question. These checks report what is measurable without a label:
detection confidence, deviation from a dataset's own modal language, script
disagreement, and Turkish that has lost its diacritics.

Two deliberate constraints. Nothing under ``MIN_CHARS`` produces a language
finding, because bag-of-n-gram identification is unreliable on short text and
this is where false positives come from. And when the reduced-accuracy fallback
backend is in use, every finding says so rather than presenting its guesses as
equivalent.
"""

from __future__ import annotations

from ..context import F_LANG, F_LANG_CONF, ScanContext
from ..langid import LangResult
from ..models import (
    CostClass,
    Document,
    Evidence,
    Finding,
    Profile,
    Requirement,
    Severity,
)
from ..textutil import excerpt, is_ascii_folded_turkish
from .base import Check, make_finding, register

ALL_PROFILES = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)

#: Below this many characters we do not emit a language finding at all.
MIN_CHARS = 40


@register
class LanguageComposition(Check):
    check_id = "T1-LANG-001"
    title = "Language composition and detection confidence"
    tier = 1
    profiles = ALL_PROFILES
    requires = (Requirement.LANGID,)
    cost = CostClass.CHEAP
    severity = Severity.INFO
    fix = "Informational. Compare the mix against what you intended to train."

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.low_conf = 0
        self.total = 0
        self.by_dataset_lang: dict[str, dict[str, int]] = {}
        self.evidence: list[Evidence] = []

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        lang = doc.meta.get(F_LANG)
        if lang is None:
            return
        self.total += 1
        self.counts[lang] = self.counts.get(lang, 0) + 1
        self.by_dataset_lang.setdefault(doc.dataset, {})
        self.by_dataset_lang[doc.dataset][lang] = (
            self.by_dataset_lang[doc.dataset].get(lang, 0) + 1
        )
        if lang == "unknown" and len(doc.text) >= MIN_CHARS:
            self.low_conf += 1
            if len(self.evidence) < 5:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             excerpt(doc.text, 140))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.total:
            return []
        ordered = sorted(self.counts.items(), key=lambda kv: -kv[1])
        mix = ", ".join(f"{k} {v / self.total:.0%}" for k, v in ordered[:6])
        detail = f"language mix: {mix}"
        backend = ctx.detector.backend if ctx.detector else "none"
        if ctx.detector is not None and ctx.detector.low_trust:
            detail += (
                f"; produced by the reduced-accuracy fallback backend ({backend}), "
                f"install 'dropoutt[lid]' for a real detector"
            )
        if self.low_conf:
            detail += f"; {self.low_conf} records of adequate length could not be identified"
        return [
            make_finding(
                self, count=self.low_conf, total=self.total, detail=detail,
                evidence=self.evidence,
                data={
                    "composition": dict(ordered),
                    "backend": backend,
                    "low_trust": bool(ctx.detector and ctx.detector.low_trust),
                    "per_dataset": self.by_dataset_lang,
                },
            )
        ]


@register
class LanguageOutliers(Check):
    check_id = "T1-LANG-002"
    title = "Records deviate from their dataset's main language"
    tier = 1
    profiles = ALL_PROFILES
    requires = (Requirement.LANGID,)
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    fix = "Inspect the outliers; they are often scraping accidents or misfiled files."
    rationale = (
        "This needs no declared label. A dataset detected as 97% Turkish with 3% Russian is "
        "telling you something regardless of what its card claims, and the 3% is where the "
        "collection bug is."
    )

    def __init__(self) -> None:
        self.per_dataset: dict[str, dict[str, int]] = {}
        self.samples: dict[tuple[str, str], Evidence] = {}
        self.total = 0

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        lang = doc.meta.get(F_LANG)
        conf = doc.meta.get(F_LANG_CONF) or 0.0
        if lang is None or lang == "unknown" or len(doc.text) < MIN_CHARS or conf < 0.7:
            return
        self.total += 1
        self.per_dataset.setdefault(doc.dataset, {})
        self.per_dataset[doc.dataset][lang] = self.per_dataset[doc.dataset].get(lang, 0) + 1
        key = (doc.dataset, lang)
        if key not in self.samples:
            self.samples[key] = Evidence(
                doc.doc_id, doc.source_file, doc.source_index,
                f"[{lang}] {excerpt(doc.text, 130)}"
            )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        outliers = 0
        evidence: list[Evidence] = []
        by_dataset: dict[str, int] = {}
        details: list[str] = []

        for dataset, counts in self.per_dataset.items():
            total = sum(counts.values())
            if total < 20:
                continue
            modal = max(counts, key=lambda k: counts[k])
            modal_share = counts[modal] / total
            if modal_share < 0.6:
                continue  # genuinely multilingual, not an outlier situation
            minority = {k: v for k, v in counts.items() if k != modal}
            n_minor = sum(minority.values())
            if not n_minor or n_minor / total > 0.25:
                continue
            outliers += n_minor
            by_dataset[dataset] = n_minor
            top = sorted(minority.items(), key=lambda kv: -kv[1])[:3]
            details.append(
                f"{dataset} is {modal_share:.0%} {modal} with "
                + ", ".join(f"{v} {k}" for k, v in top)
            )
            for lang, _ in top:
                ev = self.samples.get((dataset, lang))
                if ev is not None and len(evidence) < 6:
                    evidence.append(ev)

        if not outliers:
            return []
        return [
            make_finding(
                self, count=outliers, total=self.total,
                detail="; ".join(details[:4]),
                evidence=evidence, by_dataset=by_dataset,
            )
        ]


@register
class ScriptMismatch(Check):
    check_id = "T1-LANG-003"
    title = "Script does not match the detected language"
    tier = 1
    profiles = ALL_PROFILES
    requires = (Requirement.LANGID,)
    cost = CostClass.CHEAP
    severity = Severity.INFO
    fix = "Confirm this is intentional. Turkish in Arabic script is Ottoman, not a defect."

    def __init__(self) -> None:
        self.total = 0
        self.count = 0
        self.pairs: dict[str, int] = {}
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        result: LangResult | None = doc.meta.get("_lang_result")
        if result is None or ctx.detector is None:
            return
        if len(doc.text) < MIN_CHARS or result.is_unknown:
            return
        self.total += 1
        if ctx.detector.script_mismatch(result):
            self.count += 1
            key = f"{result.lang} in {result.script}"
            self.pairs[key] = self.pairs.get(key, 0) + 1
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 4:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             f"{key} :: {excerpt(doc.text, 120)}")
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.count:
            return []
        top = sorted(self.pairs.items(), key=lambda kv: -kv[1])[:4]
        return [
            make_finding(
                self, count=self.count, total=self.total,
                detail="; ".join(f"{k} ({v})" for k, v in top),
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={"pairs": dict(top)},
            )
        ]


@register
class AsciiFoldedTurkish(Check):
    check_id = "T1-LANG-004"
    title = "Turkish text has lost its diacritics"
    tier = 1
    profiles = ALL_PROFILES
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    fix = "Re-source the text with correct orthography; the diacritics cannot be restored reliably."
    rationale = (
        "Language identification will confidently call 'degil mi' Turkish, and it is, but it "
        "is damaged Turkish. A model trained on it learns the wrong orthography and will "
        "reproduce it. No general-purpose tool checks for this, and for a Turkish model it "
        "matters more than most things that are checked."
    )

    def __init__(self) -> None:
        self.total = 0
        self.count = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if len(doc.text) < MIN_CHARS:
            return
        self.total += 1
        if is_ascii_folded_turkish(doc.text):
            self.count += 1
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 5:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             excerpt(doc.text, 150))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.count:
            return []
        rate = self.count / self.total if self.total else 0
        return [
            make_finding(
                self, count=self.count, total=self.total,
                detail=(
                    f"{self.count} records ({rate:.1%}) read as Turkish but contain none of "
                    f"the Turkish-specific characters"
                ),
                evidence=self.evidence, by_dataset=self.by_dataset,
            )
        ]
