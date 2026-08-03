"""Tier 1 content checks: PII, identity leakage, style, licence, contamination."""

from __future__ import annotations

from ..contamination import NGRAM
from ..context import ScanContext, dedup_words_of
from ..models import (
    CostClass,
    Document,
    Evidence,
    Finding,
    Profile,
    Requirement,
    Severity,
)
from ..registry_data import (
    VALIDATORS,
    compiled_identity,
    compiled_pii,
    compiled_style_openers,
    mask_value,
    pii_patterns,
    style_patterns,
)
from ..textutil import eval_ngram_hashes, excerpt
from .base import Check, make_finding, register

ALL_PROFILES = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)


@register
class PIIAndSecrets(Check):
    check_id = "T1-PII-001"
    title = "Personal data and credentials in training text"
    tier = 1
    profiles = ALL_PROFILES
    cost = CostClass.CHEAP
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE)
    fix = "Redact or remove before training. A model will reproduce what it memorises."
    rationale = (
        "Patterns with a checksum use it. A bare eleven-digit regex matches every order id and "
        "timestamp in the corpus, which makes the check worse than useless. Matched values are "
        "never written into the findings file or the report: only a masked form and an offset, "
        "so the scan output stays safe to share."
    )

    EVIDENCE_CAP = 8
    MERGE_SUM = ("total", "affected")
    MERGE_COUNTS = ("hits", "by_dataset")
    MERGE_EVIDENCE = ("evidence",)
    #: The mask table is loaded from the shipped pattern file, not accumulated.
    MERGE_IGNORE = ("_masks",)

    def __init__(self) -> None:
        masks = {p["id"]: p["mask"] for p in pii_patterns()["patterns"]}
        self._masks = masks
        self.total = 0
        self.hits: dict[str, int] = {}
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}
        self.affected = 0

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        self.total += 1
        text = doc.text
        lowered = doc.text_lower
        found: list[tuple[str, str, str]] = []
        for pid, label, _sev, pattern, validator, gate in compiled_pii():
            if gate and not gate.passes(text, lowered):
                continue
            for match in pattern.finditer(text):
                value = match.group(0)
                if validator:
                    fn = VALIDATORS.get(validator)
                    if fn is not None and not fn(value):
                        continue
                found.append((pid, label, mask_value(value, self._masks.get(pid, "full"))))
                break
        if not found:
            return
        self.affected += 1
        self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
        for pid, _label, _masked in found:
            self.hits[pid] = self.hits.get(pid, 0) + 1
        if len(self.evidence) < self.EVIDENCE_CAP:
            kinds = ", ".join(f"{label}: {masked}" for _p, label, masked in found[:3])
            # Note: the excerpt is deliberately omitted. Showing surrounding
            # text would put the unmasked value back into the report.
            self.evidence.append(
                Evidence(doc.doc_id, doc.source_file, doc.source_index, kinds)
            )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.affected:
            return []
        labels = {p["id"]: p["label"] for p in pii_patterns()["patterns"]}
        top = sorted(self.hits.items(), key=lambda kv: -kv[1])
        detail = ", ".join(f"{labels.get(k, k)} ({v})" for k, v in top[:6])
        return [
            make_finding(
                self, count=self.affected, total=self.total, detail=detail,
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={"by_kind": dict(top), "values_redacted": True},
            )
        ]


@register
class IdentityLeakage(Check):
    check_id = "T1-IDENT-001"
    title = "Assistant identity leakage and refusal boilerplate"
    tier = 1
    unit = "response"
    profiles = (Profile.SFT, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Remove or rewrite; otherwise your model will claim to be someone else's product."

    EVIDENCE_CAP = 6
    MERGE_SUM = ("total", "identity", "refusal")
    MERGE_COUNTS = ("hits", "by_dataset")
    MERGE_EVIDENCE = ("evidence",)

    def __init__(self) -> None:
        self.total = 0
        self.identity = 0
        self.refusal = 0
        self.hits: dict[str, int] = {}
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        use_assistant = bool(doc.assistant_text)
        text = doc.assistant_text if use_assistant else doc.text
        if not text:
            return
        self.total += 1
        lowered = doc.assistant_lower if use_assistant else doc.text_lower
        found_group = None
        for group, pid, _lang, pattern, gate in compiled_identity():
            if gate and not gate.passes(text, lowered):
                continue
            if pattern.search(text):
                self.hits[pid] = self.hits.get(pid, 0) + 1
                found_group = found_group or group
                if group == "identity_leakage":
                    found_group = group
        if found_group is None:
            return
        if found_group == "identity_leakage":
            self.identity += 1
        else:
            self.refusal += 1
        self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
        if len(self.evidence) < self.EVIDENCE_CAP:
            self.evidence.append(
                Evidence(doc.doc_id, doc.source_file, doc.source_index, excerpt(text, 150))
            )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.identity and not self.refusal:
            return []
        parts = []
        if self.identity:
            parts.append(f"{self.identity:,} records claim to be another vendor's assistant")
        if self.refusal:
            rate = self.refusal / self.total if self.total else 0
            parts.append(f"{self.refusal:,} contain refusal boilerplate ({rate:.1%})")
        severity = Severity.BLOCKING if self.identity else Severity.INFO
        return [
            make_finding(
                self, count=self.identity + self.refusal, total=self.total,
                detail="; ".join(parts), evidence=self.evidence,
                by_dataset=self.by_dataset, severity=severity,
                data={"by_pattern": self.hits, "identity": self.identity,
                      "refusal": self.refusal},
            )
        ]


@register
class StyleTics(Check):
    check_id = "T1-STYLE-001"
    title = "Responses that all open the same way"
    tier = 1
    unit = "response"
    profiles = (Profile.SFT, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.CHEAP
    severity = Severity.INFO
    fix = "Informational. Consider diversifying if the rate is high enough to become a habit."
    rationale = (
        "Frequency is the finding, not presence. One 'Certainly!' is a sentence. 'Certainly!' "
        "opening forty percent of responses is a style your model will inherit and reproduce "
        "on every answer it gives."
    )

    EVIDENCE_CAP = 4
    MERGE_SUM = ("total", "affected")
    MERGE_COUNTS = ("hits",)
    MERGE_EVIDENCE = ("evidence",)
    #: Read from the shipped pattern file, identical in every shard.
    MERGE_IGNORE = ("threshold",)

    def __init__(self) -> None:
        self.threshold = style_patterns()["flag_when_rate_exceeds"]
        self.total = 0
        self.hits: dict[str, int] = {}
        self.affected = 0
        self.evidence: list[Evidence] = []

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        text = doc.assistant_text
        if not text:
            return
        self.total += 1
        lowered = doc.assistant_lower
        matched = False
        for pid, _lang, pattern, gate in compiled_style_openers():
            if gate and not gate.passes(text, lowered):
                continue
            if pattern.search(text):
                self.hits[pid] = self.hits.get(pid, 0) + 1
                matched = True
        if matched:
            self.affected += 1
            if len(self.evidence) < self.EVIDENCE_CAP:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index, excerpt(text, 130))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.total or not self.affected:
            return []
        rate = self.affected / self.total
        if rate < self.threshold:
            return []
        top = sorted(self.hits.items(), key=lambda kv: -kv[1])[:5]
        return [
            make_finding(
                self, count=self.affected, total=self.total,
                detail=(
                    f"{rate:.0%} of responses open with a formulaic phrase "
                    f"(threshold {self.threshold:.0%}): "
                    + ", ".join(f"{k} ({v})" for k, v in top)
                ),
                evidence=self.evidence,
                data={"rate": rate, "by_pattern": dict(top)},
            )
        ]


@register
class LicenceProvenance(Check):
    check_id = "T1-LIC-001"
    title = "Datasets have no recorded licence"
    tier = 1
    unit = "dataset"
    profiles = ALL_PROFILES
    cost = CostClass.FREE
    severity = Severity.WARNING
    fix = "Record the licence and source for each dataset before training on it."
    rationale = (
        "An audit of more than 1,800 public datasets found licence omission above 70% and "
        "licence error rates above 50% on popular hosting sites. Under EU AI Act Article "
        "53(1)(d) a general-purpose model provider has to publish a summary of training "
        "content, and that summary cannot be assembled from datasets whose origin was never "
        "recorded."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        missing = [d.name for d in ctx.datasets if not d.license]
        if not missing:
            return []
        return [
            make_finding(
                self, count=len(missing), total=len(ctx.datasets),
                # The count is carried by the finding, and every report prints
                # it beside the title. Repeating it here put "4 of 4 datasets"
                # on the page twice, one line apart, which reads as a rendering
                # bug rather than as emphasis.
                detail=(
                    "A dataset whose card records no licence cannot be cleared "
                    "for training: the terms the text was published under are "
                    "unknown, and so is whether they permit this use."
                ),
                by_dataset=dict.fromkeys(missing, 1),
                data={"missing": missing[:50]},
            )
        ]


@register
class BenchmarkContamination(Check):
    check_id = "T1-CONTAM-001"
    title = "Training data overlaps evaluation benchmarks"
    tier = 1
    unit = "benchmark question"
    profiles = ALL_PROFILES
    requires = (Requirement.CONTAMINATION_INDEX,)
    cost = CostClass.GLOBAL
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE)
    fix = "Remove the overlapping records, or stop reporting scores on the affected benchmark."
    rationale = (
        "Applies the Tulu 3 rule as published: an evaluation instance is contaminated when "
        "more than 50% of its tokens are covered by 8-gram matches against a single training "
        "instance, and a training set is contaminated when more than 2% of any evaluation's "
        "instances match. Note that removing contamination usually makes your reported score "
        "go DOWN. That is the point: the previous number was not real."
    )

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if ctx.contamination is None:
            return
        ctx.contamination.observe_hashes(
            eval_ngram_hashes(dedup_words_of(doc), NGRAM),
            doc.doc_id, doc.source_file, doc.source_index,
        )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if ctx.contamination is None:
            return []
        results = ctx.contamination.results()
        flagged = {k: v for k, v in results.items() if v["flagged"]}
        any_hit = {k: v for k, v in results.items() if v["n_contaminated"]}
        if not any_hit:
            return []

        parts = [
            f"{name} {v['rate']:.1%} ({v['n_contaminated']}/{v['n_instances']})"
            for name, v in sorted(any_hit.items(), key=lambda kv: -kv[1]["rate"])[:8]
        ]
        evidence: list[Evidence] = []
        for name, v in sorted(any_hit.items(), key=lambda kv: -kv[1]["rate"]):
            for w in v["witnesses"][:2]:
                rec = w.get("record")
                if rec:
                    evidence.append(
                        Evidence(rec[0], rec[1], rec[2],
                                 f"{name} instance {w['instance']} "
                                 f"{w['coverage']:.0%} token coverage")
                    )
            if len(evidence) >= 6:
                break

        return [
            make_finding(
                self,
                count=sum(v["n_contaminated"] for v in any_hit.values()),
                total=sum(v["n_instances"] for v in results.values()),
                detail="; ".join(parts),
                evidence=evidence,
                severity=Severity.BLOCKING if flagged else Severity.WARNING,
                data={
                    "rule": "Tulu 3: >50% token coverage per instance, >2% of instances",
                    "results": {k: {kk: vv for kk, vv in v.items() if kk != "witnesses"}
                                for k, v in results.items()},
                    "flagged": list(flagged),
                },
            )
        ]
