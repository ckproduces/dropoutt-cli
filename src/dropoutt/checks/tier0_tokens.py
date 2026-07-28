"""Tier 0 tokenizer and chat-template checks.

Everything here except template drift needs ``--model``. Without it these are
reported as skipped alongside the flag that unlocks them, rather than guessed at.
"""

from __future__ import annotations

from ..context import F_LOSS_MASK, F_RENDERED, F_TOKEN_COUNT, F_TOKEN_IDS, ScanContext
from ..models import (
    CostClass,
    Document,
    Evidence,
    Finding,
    Profile,
    Requirement,
    Severity,
)
from ..registry_data import detect_template_family, template_family_for_model
from ..textutil import excerpt
from .base import Check, make_finding, register

CONVERSATIONAL = (Profile.SFT, Profile.PREFERENCE)


@register
class TemplateDrift(Check):
    check_id = "T0-TMPL-001"
    title = "Data is already formatted with a chat template"
    tier = 0
    profiles = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Strip the embedded template, or confirm your trainer expects pre-formatted text."
    rationale = (
        "Message content that already contains turn delimiters gets those delimiters again "
        "when the trainer applies its own template. The model then learns a doubled format it "
        "will never see at inference. It is worse when the embedded family differs from the "
        "target model's, which is what this check reports."
    )

    def __init__(self) -> None:
        self.total = 0
        self.hits: dict[str, int] = {}
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        self.total += 1
        # Two distinct delimiters from one family is evidence; one stray token
        # in prose is not.
        families = [(fam, n) for fam, n in detect_template_family(doc.text) if n >= 2]
        if not families:
            return
        fam = families[0][0]
        self.hits[fam] = self.hits.get(fam, 0) + 1
        self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
        if len(self.evidence) < 5:
            self.evidence.append(
                Evidence(doc.doc_id, doc.source_file, doc.source_index,
                         f"looks like {fam} formatting :: {excerpt(doc.text, 140)}")
            )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.hits:
            return []
        target = template_family_for_model(ctx.model_id) if ctx.model_id else None
        parts = ", ".join(f"{fam} ({n})" for fam, n in sorted(self.hits.items(), key=lambda kv: -kv[1]))
        detail = f"embedded template markers found: {parts}"
        severity = self.severity
        if target and any(fam != target for fam in self.hits):
            mismatched = [f for f in self.hits if f != target]
            detail += f"; target model uses {target}, data carries {', '.join(mismatched)}"
            severity = Severity.BLOCKING
        return [
            make_finding(
                self, count=sum(self.hits.values()), total=self.total, detail=detail,
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={"families": self.hits, "target": target}, severity=severity,
            )
        ]


@register
class TemplateRender(Check):
    check_id = "T0-TMPL-002"
    title = "Records fail to render with the target chat template"
    tier = 0
    profiles = CONVERSATIONAL
    requires = (Requirement.CHAT_TEMPLATE,)
    cost = CostClass.TOKENIZER
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT,)
    fix = "Repair the affected records; the trainer will fail or silently drop them."

    def __init__(self) -> None:
        self.total = 0
        self.failed = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if not doc.turns:
            return
        self.total += 1
        err = doc.meta.get("render_error")
        if err:
            self.failed += 1
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 5:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             f"{err} :: {excerpt(doc.text, 120)}")
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.failed:
            return []
        return [
            make_finding(
                self, count=self.failed, total=self.total,
                detail=f"{self.failed} of {self.total} records could not be rendered",
                evidence=self.evidence, by_dataset=self.by_dataset,
            )
        ]


@register
class EmptyLossMask(Check):
    check_id = "T0-MASK-001"
    title = "Records contribute zero trainable tokens"
    tier = 0
    profiles = CONVERSATIONAL
    requires = (Requirement.CHAT_TEMPLATE, Requirement.TOKENIZER)
    cost = CostClass.TOKENIZER
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT,)
    fix = "Drop these records, or fix the role names so the assistant span is recognised."
    rationale = (
        "The most expensive silent failure in supervised fine-tuning. A record whose label "
        "vector is entirely ignore-index contributes nothing to any gradient, costs tokens in "
        "the packed block, and appears nowhere in the training logs. The usual cause is a role "
        "name the template does not recognise."
    )

    def __init__(self) -> None:
        self.total = 0
        self.empty = 0
        self.wasted = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        mask = doc.meta.get(F_LOSS_MASK)
        if mask is None:
            return
        self.total += 1
        if not any(mask):
            self.empty += 1
            self.wasted += len(mask)
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 6:
                roles = [t.raw_role or t.role for t in doc.turns]
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             f"roles={roles} :: {excerpt(doc.text, 120)}")
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.empty:
            return []
        pct = self.empty / self.total if self.total else 0
        return [
            make_finding(
                self, count=self.empty, total=self.total,
                detail=(
                    f"{self.empty} records ({pct:.1%}) have an entirely masked label vector "
                    f"and will train nothing"
                ),
                evidence=self.evidence, by_dataset=self.by_dataset,
                wasted_tokens=self.wasted,
            )
        ]


@register
class StopTokenConvention(Check):
    check_id = "T0-MASK-002"
    title = "Stop token is outside the trainable span"
    tier = 0
    profiles = CONVERSATIONAL
    requires = (Requirement.CHAT_TEMPLATE, Requirement.TOKENIZER)
    cost = CostClass.TOKENIZER
    severity = Severity.BLOCKING
    blocking_in = (Profile.SFT,)
    fix = "Include the end-of-turn token in the loss mask so the model learns to stop."
    rationale = (
        "If the end-of-turn token is never inside the trainable span, the model never learns "
        "to emit it and will not stop generating. Which convention applies depends on the "
        "template: a generation-tagged Qwen template puts the terminator inside the span, "
        "while a difference-derived span for a Mistral-style template does not. The check "
        "reports which convention was detected, because the number is meaningless without it."
    )

    def __init__(self) -> None:
        self.total = 0
        self.missing = 0
        self.convention: str | None = None

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        mask = doc.meta.get(F_LOSS_MASK)
        ids = doc.meta.get(F_TOKEN_IDS)
        if mask is None or ids is None or not any(mask):
            return
        eos_id = ctx.stats.get("eos_token_id")
        if eos_id is None:
            return
        self.total += 1
        trainable_ids = {i for i, m in zip(ids, mask) if m}
        if eos_id not in trainable_ids:
            self.missing += 1

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.total or self.missing < self.total * 0.9:
            return []

        tpl = ctx.chat_template
        tagged = tpl is not None and tpl.uses_generation_tag

        if tagged:
            # The template explicitly marked its generated region and left the
            # terminator outside it. That is a property of the template, and the
            # model really will not learn to stop.
            self.convention = "generation-tag"
            return [
                make_finding(
                    self, count=self.missing, total=self.total,
                    detail=(
                        f"the template marks its generated region explicitly and the "
                        f"end-of-turn token falls outside it in {self.missing}/{self.total} "
                        f"records, so the model will not learn to stop"
                    ),
                    data={"convention": self.convention},
                )
            ]

        # No generation tag, so the span came from our difference method, which
        # recovers the assistant *content* and by construction excludes whatever
        # the template appends after it. We cannot tell from this whether the
        # user's trainer includes the terminator, so reporting a defect here
        # would be reporting our own methodology as their bug.
        self.convention = "difference-derived"
        return [
            make_finding(
                self, count=0, total=self.total,
                detail=(
                    f"{ctx.model_id or 'this model'} ships no generation tag in its chat "
                    f"template, so trainable spans were recovered by difference and exclude "
                    f"the end-of-turn token by construction. Confirm your trainer includes "
                    f"it, or the model will not learn to stop. This is not a defect in your "
                    f"data."
                ),
                severity=Severity.INFO,
                data={"convention": self.convention, "verifiable": False},
            )
        ]


@register
class TruncationForecast(Check):
    check_id = "T0-TRUNC-001"
    title = "Records exceed the sequence length"
    tier = 0
    profiles = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE)
    requires = (Requirement.TOKENIZER, Requirement.SEQ_LEN)
    cost = CostClass.TOKENIZER
    severity = Severity.WARNING
    blocking_in = (Profile.SFT,)
    fix = "Raise the sequence length, split the records, or drop them deliberately."
    rationale = (
        "Two different bad outcomes hide behind one number. A record truncated from the end "
        "loses part of its answer. A record whose entire assistant span falls beyond the limit "
        "teaches nothing at all, and packing pipelines that concatenate rather than truncate "
        "will split it across blocks with no separator and full cross-document attention."
    )

    def __init__(self) -> None:
        self.total = 0
        self.over = 0
        self.answer_lost = 0
        self.tokens_lost = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        n = doc.meta.get(F_TOKEN_COUNT)
        if n is None or ctx.seq_len is None:
            return
        self.total += 1
        if n <= ctx.seq_len:
            return
        self.over += 1
        self.tokens_lost += n - ctx.seq_len
        self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1

        mask = doc.meta.get(F_LOSS_MASK)
        if mask is not None:
            kept = mask[: ctx.seq_len]
            if not any(kept):
                self.answer_lost += 1
                if len(self.evidence) < 5:
                    self.evidence.append(
                        Evidence(doc.doc_id, doc.source_file, doc.source_index,
                                 f"{n} tokens, entire assistant span beyond {ctx.seq_len}")
                    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.over:
            return []
        detail = (
            f"{self.over} of {self.total} records exceed seq_len={ctx.seq_len}, "
            f"losing {self.tokens_lost:,} tokens"
        )
        severity = self.severity
        if self.answer_lost:
            detail += f"; {self.answer_lost} lose their entire assistant span"
            severity = Severity.BLOCKING
        return [
            make_finding(
                self, count=self.over, total=self.total, detail=detail,
                evidence=self.evidence, by_dataset=self.by_dataset,
                wasted_tokens=self.tokens_lost, severity=severity,
                data={"answer_fully_lost": self.answer_lost, "seq_len": ctx.seq_len},
            )
        ]


@register
class PackingEfficiency(Check):
    check_id = "T0-PACK-001"
    title = "Packing efficiency under concat-and-chunk"
    tier = 0
    profiles = (Profile.SFT, Profile.CORPUS)
    requires = (Requirement.TOKENIZER, Requirement.SEQ_LEN)
    cost = CostClass.TOKENIZER
    severity = Severity.INFO
    fix = "Informational. Compare against your trainer's actual packing strategy."
    rationale = (
        "Reported for concat-and-chunk specifically, because the number differs by ten to "
        "twenty points between concat-and-chunk, first-fit-decreasing and best-fit. A packing "
        "efficiency with no algorithm attached is not comparable to anything."
    )

    def __init__(self) -> None:
        self.total_tokens = 0
        self.trainable_tokens = 0
        self.records = 0

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        n = doc.meta.get(F_TOKEN_COUNT)
        if n is None:
            return
        self.records += 1
        self.total_tokens += n
        mask = doc.meta.get(F_LOSS_MASK)
        if mask is not None:
            self.trainable_tokens += sum(1 for m in mask if m)

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.records or ctx.seq_len is None:
            return []
        blocks = self.total_tokens // ctx.seq_len
        residual = self.total_tokens - blocks * ctx.seq_len
        trainable_ratio = (
            self.trainable_tokens / self.total_tokens if self.total_tokens else 0.0
        )
        return [
            make_finding(
                self, count=0, total=self.records,
                detail=(
                    f"{self.total_tokens:,} tokens fill {blocks:,} blocks of {ctx.seq_len}, "
                    f"{residual:,} tokens land in a residual tail that concat-and-chunk "
                    f"pipelines usually discard; "
                    f"{trainable_ratio:.1%} of all tokens are trainable"
                ),
                wasted_tokens=residual,
                data={
                    "algorithm": "concat-and-chunk",
                    "seq_len": ctx.seq_len,
                    "blocks": blocks,
                    "residual_tokens": residual,
                    "trainable_ratio": trainable_ratio,
                },
            )
        ]
