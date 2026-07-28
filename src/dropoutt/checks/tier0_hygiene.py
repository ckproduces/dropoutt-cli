"""Tier 0 text hygiene, duplication and degeneracy checks.

None of these need a model. All of them are cheap enough to run on every commit.
"""

from __future__ import annotations

import re

from ..context import ScanContext
from ..models import CostClass, Document, Evidence, Finding, Profile, Severity
from ..registry_data import style_patterns
from ..textutil import (
    control_chars,
    excerpt,
    has_mojibake,
    needs_nfc,
    normalize_ws,
    repetition_ratio,
)
from .base import Check, make_finding, register

ALL_PROFILES = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)

# Turkish text that has been through a naive .lower() or .upper(): the dotted
# capital I and the dotless lowercase i are distinct letters, and the default
# locale mangles both. "ISTANBUL".lower() gives "istanbul" where Turkish wants
# "ıstanbul"; "iş".upper() gives "IŞ" where Turkish wants "İŞ".
_TR_CASE_DAMAGE = re.compile(r"\b(?:istanbul|izmir|ısparta|IsTANBUL)\b")


@register
class EncodingHygiene(Check):
    check_id = "T0-ENC-001"
    title = "Text encoding is damaged"
    tier = 0
    profiles = ALL_PROFILES
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    blocking_in = (Profile.SFT, Profile.CORPUS)
    fix = "Re-decode the source as UTF-8 and re-export; the damage is not recoverable by the model."
    rationale = (
        "Mojibake is a UTF-8 stream that was decoded as latin-1 somewhere upstream. It is "
        "especially destructive for Turkish, where it eats exactly the characters that carry "
        "meaning."
    )

    def __init__(self) -> None:
        self.total = 0
        self.mojibake = 0
        self.control = 0
        self.nfc = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        self.total += 1
        text = doc.text
        hit = False
        if has_mojibake(text):
            self.mojibake += 1
            hit = True
        ctrl = control_chars(text)
        if ctrl:
            self.control += 1
            hit = True
        if needs_nfc(text):
            self.nfc += 1
            hit = True
        if hit:
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 5:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index, excerpt(text, 160))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        parts = []
        if self.mojibake:
            parts.append(f"{self.mojibake} with mojibake")
        if self.control:
            parts.append(f"{self.control} with control characters")
        if self.nfc:
            parts.append(f"{self.nfc} not in Unicode NFC form")
        if not parts:
            return []
        count = sum(self.by_dataset.values())
        return [
            make_finding(
                self, count=count, total=self.total, detail="; ".join(parts),
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={"mojibake": self.mojibake, "control": self.control, "not_nfc": self.nfc},
            )
        ]


@register
class TurkishCaseHazard(Check):
    check_id = "T0-ENC-002"
    title = "Turkish dotted and dotless I damage"
    tier = 0
    profiles = ALL_PROFILES
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    fix = "Use a Turkish-aware casefold, or leave case untouched during preprocessing."
    rationale = (
        "Turkish has two distinct letter I pairs. A default-locale lower() turns 'I' into 'i' "
        "where Turkish wants 'ı', and upper() turns 'i' into 'I' where Turkish wants 'İ'. The "
        "result is fluent-looking text with the wrong orthography, which language "
        "identification will not flag."
    )

    def __init__(self) -> None:
        self.total = 0
        self.count = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        self.total += 1
        # Only meaningful if the text is plausibly Turkish at all.
        if "ı" not in doc.text and "İ" not in doc.text and "ş" not in doc.text:
            return
        if _TR_CASE_DAMAGE.search(doc.text):
            self.count += 1
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 4:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index, excerpt(doc.text, 160))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        if not self.count:
            return []
        return [
            make_finding(
                self, count=self.count, total=self.total,
                detail=f"{self.count} records show locale-incorrect Turkish case folding",
                evidence=self.evidence, by_dataset=self.by_dataset,
            )
        ]


@register
class ExactDuplicates(Check):
    check_id = "T0-DUP-001"
    title = "Exact and whitespace-identical duplicates"
    tier = 0
    profiles = ALL_PROFILES
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    fix = "Deduplicate before training, but see the note about cluster size below."
    rationale = (
        "Exact duplication wastes tokens outright. Note that removing duplicates is not "
        "automatically an improvement: FineWeb found the benefit comes from removing very "
        "large clusters, and that deduplicating harder than that made their corpus worse. "
        "This check reports; the decision is yours."
    )

    def __init__(self) -> None:
        self.total = 0
        self.exact: dict[str, int] = {}
        self.ws: dict[str, int] = {}
        self.first_seen: dict[str, tuple[str, str, int, str]] = {}
        self.by_dataset: dict[str, int] = {}
        self.dup_chars = 0

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if not doc.text.strip():
            return
        self.total += 1
        key = hash(doc.text)
        skey = hash(normalize_ws(doc.text).lower())
        prev = self.exact.get(key, 0)
        self.exact[key] = prev + 1
        self.ws[skey] = self.ws.get(skey, 0) + 1
        if prev == 0:
            if len(self.first_seen) < 50_000:
                self.first_seen[key] = (doc.doc_id, doc.source_file, doc.source_index,
                                        excerpt(doc.text, 160))
        else:
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            self.dup_chars += len(doc.text)

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        exact_extra = sum(v - 1 for v in self.exact.values() if v > 1)
        ws_extra = sum(v - 1 for v in self.ws.values() if v > 1)
        if not exact_extra and not ws_extra:
            return []
        clusters = sum(1 for v in self.exact.values() if v > 1)
        largest = max(self.exact.values()) if self.exact else 0
        evidence = [
            Evidence(d, f, i, t)
            for d, f, i, t in list(self.first_seen.values())[:4]
        ]
        detail = (
            f"{exact_extra} redundant copies across {clusters} exact clusters "
            f"(largest cluster {largest} copies)"
        )
        if ws_extra > exact_extra:
            detail += f"; {ws_extra} when whitespace and case are ignored"
        return [
            make_finding(
                self, count=exact_extra, total=self.total, detail=detail,
                evidence=evidence, by_dataset=self.by_dataset,
                wasted_tokens=int(self.dup_chars / 3.6),
                data={"clusters": clusters, "largest_cluster": largest,
                      "whitespace_extra": ws_extra},
            )
        ]


@register
class Degeneracy(Check):
    check_id = "T0-DEGEN-001"
    title = "Degenerate responses"
    tier = 0
    profiles = (Profile.SFT, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    fix = "Remove trivial, looping or prompt-copying responses."
    rationale = (
        "Three distinct failure shapes with one remedy: an answer too short to teach anything, "
        "a generation that got stuck in a loop, and a response that simply repeats the prompt. "
        "All three are common in scraped and synthetic data."
    )

    def __init__(self) -> None:
        cfg = style_patterns()["degeneracy"]
        self.min_chars = cfg["single_token_answer_max_chars"]
        self.rep_n = cfg["repetition_ngram"]
        self.rep_threshold = cfg["repetition_ratio_threshold"]
        self.copy_threshold = cfg["prompt_copy_similarity"]
        self.total = 0
        self.trivial = 0
        self.looping = 0
        self.copying = 0
        self.evidence: list[Evidence] = []
        self.by_dataset: dict[str, int] = {}

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        answer = doc.assistant_text.strip()
        if not doc.turns:
            answer = doc.text.strip()
        if not answer:
            return
        self.total += 1
        why = None

        if len(answer) <= self.min_chars:
            self.trivial += 1
            why = "response is a single token"
        elif repetition_ratio(answer, self.rep_n) >= self.rep_threshold:
            self.looping += 1
            why = "response repeats itself"
        else:
            prompt = doc.prompt_text.strip()
            if prompt and len(answer) > 40:
                a = normalize_ws(answer).lower()
                p = normalize_ws(prompt).lower()
                if a and (a in p or p in a):
                    self.copying += 1
                    why = "response copies the prompt"

        if why:
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < 6:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             f"{why} :: {excerpt(answer, 140)}")
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        parts = []
        if self.trivial:
            parts.append(f"{self.trivial} trivial")
        if self.looping:
            parts.append(f"{self.looping} looping")
        if self.copying:
            parts.append(f"{self.copying} copying the prompt")
        if not parts:
            return []
        count = sum(self.by_dataset.values())
        return [
            make_finding(
                self, count=count, total=self.total,
                detail=", ".join(parts) + " responses",
                evidence=self.evidence, by_dataset=self.by_dataset,
                data={"trivial": self.trivial, "looping": self.looping, "copying": self.copying},
            )
        ]
