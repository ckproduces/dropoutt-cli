"""Tier 0 text hygiene, duplication and degeneracy checks.

None of these need a model. All of them are cheap enough to run on every commit.
"""

from __future__ import annotations

from hashlib import blake2b as _blake
from typing import Any, NamedTuple

import numpy as _np

from ..context import ScanContext
from ..models import CostClass, Document, Evidence, Finding, Profile, Severity
from ..registry_data import style_patterns
from ..textutil import (
    excerpt,
    has_control_chars,
    has_mojibake,
    needs_nfc,
    normalize_ws,
    repetition_ratio,
)
from .base import Check, make_finding, register

ALL_PROFILES = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)

#: How many distinct records keep an example and a character count. Past this
#: the duplicate report is already complete in aggregate — counts come from the
#: unbounded tally — and the extra examples buy nothing.
FIRST_SEEN_CAP = 50_000


class _DigestTally:
    """A multiset of 64-bit digests, stored as an array and counted at the end.

    The obvious structure is ``dict[int, int]`` — digest to how many times it
    was seen — and that is what this was. It costs about 165 bytes per distinct
    record: a boxed key, a boxed count and a hash-table slot, to hold sixteen
    bytes of information. On a corpus with few duplicates almost every record is
    distinct, so it is 165 bytes per *record*, and the exact-duplicate check was
    the third-largest thing a scan held in memory after the near-duplicate index.

    Appending to a growable array costs eight. The counting that the dictionary
    was doing incrementally happens once, in ``counts``, as a sort — which is
    also faster than several million dictionary probes, and which the check
    already had to wait for because "is this a repeat?" is only answerable once
    every record has been seen.

    Only digests seen more than once come back out. Singletons are the answer to
    a question nobody asks, and there are millions of them.
    """

    def __init__(self) -> None:
        self._keys = _np.zeros(0, dtype=_np.uint64)
        self._size = 0

    def add(self, digest: int) -> None:
        if self._size >= self._keys.size:
            grown = _np.zeros(max(4096, self._keys.size * 2), dtype=_np.uint64)
            grown[: self._keys.size] = self._keys
            self._keys = grown
        self._keys[self._size] = digest
        self._size += 1

    def extend(self, other: _DigestTally) -> None:
        needed = self._size + other._size
        if needed > self._keys.size:
            grown = _np.zeros(max(4096, needed, self._keys.size * 2), dtype=_np.uint64)
            grown[: self._keys.size] = self._keys
            self._keys = grown
        self._keys[self._size : needed] = other._keys[: other._size]
        self._size = needed

    def counts(self) -> _Repeats:
        if not self._size:
            return _Repeats(_np.zeros(0, dtype=_np.uint64), _np.zeros(0, dtype=_np.int64))
        keys, counts = _np.unique(self._keys[: self._size], return_counts=True)
        keep = counts > 1
        return _Repeats(keys[keep], counts[keep].astype(_np.int64))


class _Repeats(NamedTuple):
    keys: Any
    counts: Any

    @property
    def size(self) -> int:
        return int(self.keys.size)


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

    MERGE_SUM = ("total", "mojibake", "control", "nfc")
    MERGE_COUNTS = ("by_dataset",)
    MERGE_EVIDENCE = ("evidence",)

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
        if has_control_chars(text):
            self.control += 1
            hit = True
        if needs_nfc(text):
            self.nfc += 1
            hit = True
        if hit:
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < self.EVIDENCE_CAP:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index, excerpt(text, 160))
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        parts = []
        if self.mojibake:
            parts.append(f"{self.mojibake:,} with mojibake")
        if self.control:
            parts.append(f"{self.control:,} with control characters")
        if self.nfc:
            parts.append(f"{self.nfc:,} not in Unicode NFC form")
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
class ExactDuplicates(Check):
    check_id = "T0-DUP-001"
    title = "The same record appears more than once"
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

    #: Records are keyed by a BLAKE2b digest rather than by ``hash()``. Python
    #: randomises string hashing per interpreter, so a sharded scan running in
    #: several processes would give the same text different keys in each one and
    #: count every duplicate as unique.
    @staticmethod
    def _key(text: str) -> int:
        return int.from_bytes(
            _blake(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big"
        )

    EVIDENCE_CAP = 4
    MERGE_SUM = ("total",)
    MERGE_FIRST = ("origin",)
    MERGE_FIRST_CAP = FIRST_SEEN_CAP
    #: Concatenated in ``merge`` — see :class:`_DigestTally`, which is not a
    #: shape any of the per-attribute rules describe.
    MERGE_CUSTOM = ("exact", "ws")
    #: Derived in finalize from the merged tables above.
    MERGE_IGNORE = ("by_dataset", "dup_chars")

    def __init__(self) -> None:
        self.total = 0
        self.exact = _DigestTally()
        self.ws = _DigestTally()
        #: key -> (dataset, characters, evidence) for the first copy seen.
        self.origin: dict[int, tuple[str, int, Evidence]] = {}
        self.by_dataset: dict[str, int] = {}
        self.dup_chars = 0

    def merge(self, other: Check) -> None:
        super().merge(other)
        if isinstance(other, ExactDuplicates):
            self.exact.extend(other.exact)
            self.ws.extend(other.ws)

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if not doc.text.strip():
            return
        self.total += 1
        key = self._key(doc.text)
        self.exact.add(key)
        self.ws.add(self._key(normalize_ws(doc.text).lower()))
        if key not in self.origin and len(self.origin) < FIRST_SEEN_CAP:
            self.origin[key] = (
                doc.dataset,
                len(doc.text),
                Evidence(doc.doc_id, doc.source_file, doc.source_index,
                         excerpt(doc.text, 160)),
            )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        exact_counts = self.exact.counts()
        exact_extra = int((exact_counts.counts - 1).sum()) if exact_counts.size else 0
        ws_counts = self.ws.counts()
        ws_extra = int((ws_counts.counts - 1).sum()) if ws_counts.size else 0
        if not exact_extra and not ws_extra:
            return []
        clusters = int(exact_counts.size)
        largest = int(exact_counts.counts.max()) if exact_counts.size else 0

        # Wasted characters and the per-dataset split are derived here rather
        # than counted as records arrive, because "is this a repeat?" depends on
        # everything seen so far and a shard has not seen everything. Redundant
        # copies are attributed to the dataset the text first appeared in, which
        # is also the only attribution that does not change with the order the
        # files happen to be read in.
        repeated = dict(
            zip(exact_counts.keys.tolist(), exact_counts.counts.tolist(), strict=True)
        )
        for key, count in repeated.items():
            entry = self.origin.get(key)
            if entry is None:
                continue
            dataset, chars, _ev = entry
            self.by_dataset[dataset] = self.by_dataset.get(dataset, 0) + (count - 1)
            self.dup_chars += chars * (count - 1)

        evidence = [
            entry[2] for key, entry in self.origin.items() if key in repeated
        ][: self.EVIDENCE_CAP]
        detail = (
            f"{exact_extra:,} redundant copies across {clusters:,} exact clusters "
            f"(largest cluster {largest:,} copies)"
        )
        if ws_extra > exact_extra:
            detail += f"; {ws_extra:,} when whitespace and case are ignored"
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
    title = "Responses that teach the model nothing"
    tier = 0
    unit = "response"
    profiles = (Profile.SFT, Profile.PREFERENCE, Profile.UNKNOWN)
    cost = CostClass.CHEAP
    severity = Severity.WARNING
    fix = "Remove trivial, looping or prompt-copying responses."
    rationale = (
        "Three distinct failure shapes with one remedy: an answer too short to teach anything, "
        "a generation that got stuck in a loop, and a response that simply repeats the prompt. "
        "All three are common in scraped and synthetic data."
    )

    EVIDENCE_CAP = 6
    MERGE_SUM = ("total", "trivial", "looping", "copying")
    MERGE_COUNTS = ("by_dataset",)
    MERGE_EVIDENCE = ("evidence",)
    #: Thresholds read from the shipped pattern file, identical in every shard.
    MERGE_IGNORE = ("min_chars", "rep_n", "rep_threshold", "copy_threshold")

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
                # Containment alone is not degeneracy, and treating it as such
                # misread 38 out of 38 flagged records on a real corpus. Every
                # one was either a multiple-choice answer — necessarily a
                # substring of the prompt that listed the options — or
                # extractive QA, where the instruction was literally "quote the
                # answer from the passage". Both are correct supervision, and
                # calling them degenerate teaches the user to ignore the check.
                #
                # What makes a copy degenerate is that it accounts for nearly
                # all of the other side: an answer that *is* the prompt, or a
                # prompt restated as the whole answer. Picking one option out of
                # five, or quoting one sentence from a passage, is neither.
                if a and a in p and len(a) >= len(p) * self.copy_threshold:
                    self.copying += 1
                    why = "response repeats the prompt back"
                elif a and p in a and len(p) >= len(a) * self.copy_threshold:
                    self.copying += 1
                    why = "response is the prompt with little added"

        if why:
            self.by_dataset[doc.dataset] = self.by_dataset.get(doc.dataset, 0) + 1
            if len(self.evidence) < self.EVIDENCE_CAP:
                self.evidence.append(
                    Evidence(doc.doc_id, doc.source_file, doc.source_index,
                             f"{why} :: {excerpt(answer, 140)}")
                )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        parts = []
        if self.trivial:
            parts.append(f"{self.trivial:,} trivial")
        if self.looping:
            parts.append(f"{self.looping:,} looping")
        if self.copying:
            parts.append(f"{self.copying:,} copying the prompt")
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
