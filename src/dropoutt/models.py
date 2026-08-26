"""Core data model.

Plain dataclasses rather than pydantic, to keep the core dependency set small
enough to install on a locked-down cluster.

The central abstraction is :class:`Document`. An SFT record is a document that
happens to carry extra structure, and a raw pretraining document is the same
object with ``turns`` empty. That is what lets one engine serve the ``sft``,
``corpus`` and ``preference`` profiles with only the structural checks differing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Bumped whenever a measurement changes, not merely when the package version
#: changes. It feeds the fingerprint id, so two fingerprints sharing an id have
#: to mean the same thing.
#:
#: 0.1.1 corrected the shape facet, which reported bytes on disk as
#: `total_chars` and a hardcoded 0 words. 0.1.2 added the full `region_counts`
#: histogram. 0.1.3 makes `content_hash` cover normalized record content and
#: structure rather than only file sizes and record counts, and hashes the
#: effective CLI configuration. 0.1.4 stops discarding the coverage facet above a
#: 10% off-atlas rate and counts `by_category` over placed records only, so both
#: the presence and the denominator of that facet changed. Fingerprints from
#: earlier versions describe something different and must not collide. 0.1.5
#: reads JSON records out of .txt and .md files instead of treating each file as
#: one document, and folds sharded siblings into one dataset — so record counts,
#: dataset counts and the inferred profile all change for those inputs — and adds
#: coverage gaps, effective region count and the per-dataset region signature to
#: the atlas facet. 0.2.0 makes the streaming pass shardable, and three
#: measurements move with it: the corpus digest is now a sum over records rather
#: than a chain, so it no longer depends on how the scan was divided; the atlas
#: and token-budget samples are a bottom-k over a positional hash rather than the
#: head of each dataset, so they cover the whole corpus instead of its
#: beginning; and near-duplicate shingles use a vectorised hash, which shifts
#: MinHash estimates by a fraction of a percent. Contamination hashes are
#: unchanged, because the shipped indices are made of them.
#: 0.3.0 changes three more. Language identification moved from
#: fasttext-lid.176 to py3langid, which is a different classifier over a
#: different label set, so every language share and every language-deviation
#: count moves — the backend is named in `provenance.langid_backend`, but a
#: version that did not move would let two incomparable language facets be
#: diffed. The near-duplicate index now stops at a memory-derived ceiling and
#: says so, so on a corpus above that ceiling its count is a floor over a prefix
#: rather than a rate over the corpus. And the per-shard token-budget cap was
#: sized against the total shard count rather than against how many shards a
#: dataset actually spans, which made a parallel scan's token estimate differ
#: from a serial one by 0.04%; they are identical now, and both differ slightly
#: from what a parallel 0.2.0 scan reported.
#: 0.4.0 moves the coverage map to `dropoutt atlas`, which changes the
#: fingerprint in two ways a reader would otherwise have to discover. The
#: `coverage` facet a scan writes is now the single value
#: `status: "not computed by scan (run dropoutt atlas)"` rather than a region
#: histogram — the same shape a 1.2 scan produced under `--no-atlas`, now the
#: only shape a scan produces. And `atlas_hash`, one of the six inputs to
#: `fingerprint_id`, is empty for every scan rather than naming the artifact, so
#: the id of an unchanged corpus moves. Two fingerprints of the same data across
#: that boundary must not be read as describing the same measurement, and a
#: version that did not move would let them be diffed as if they did.
PIPELINE_VERSION = "0.4.0"
FINGERPRINT_SCHEMA_VERSION = "fp-v0.1"


class Severity(str, Enum):
    """How much a finding matters.

    ``BLOCKING`` never causes a non-zero exit unless the user declared a target
    profile. Without a declared target there is no purpose against which
    something can be wrong, so the report says "would block under profile X"
    instead of failing the run.
    """

    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


class Confidence(str, Enum):
    """Whether acting on a finding has a measured effect.

    Every check in v0.1 is ``UNVERIFIED``. No calibration corpus exists yet, so
    the tool must not imply it has measured what removing this data does. This
    is the rule that came out of the FineWeb deduplication result, where the
    obvious action made models worse.
    """

    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class Profile(str, Enum):
    SFT = "sft"
    CORPUS = "corpus"
    PREFERENCE = "preference"
    UNKNOWN = "unknown"


class CostClass(str, Enum):
    """Roughly what a check costs, used to decide what runs by default."""

    FREE = "free"          # a few operations per record
    CHEAP = "cheap"        # regex or hashing per record
    TOKENIZER = "tokenizer"  # needs a tokenizer pass
    GLOBAL = "global"      # needs a second pass over accumulated state
    EMBEDDING = "embedding"  # needs vectors


class Requirement(str, Enum):
    """What a check needs before it can run.

    The runner uses these to decide which checks are possible given what the
    user supplied, and to report the rest as skipped with the exact flag that
    would unlock each one.
    """

    NONE = "none"
    TOKENIZER = "tokenizer"
    CHAT_TEMPLATE = "chat_template"
    SEQ_LEN = "seq_len"
    LANGID = "langid"
    EMBEDDINGS = "embeddings"
    ATLAS = "atlas"
    CONTAMINATION_INDEX = "contamination_index"
    MULTIPLE_DATASETS = "multiple_datasets"


@dataclass(slots=True)
class Turn:
    """One message in a conversation."""

    role: str
    content: str
    #: The role string exactly as it appeared in the source, before mapping.
    #: Kept because a mismatch between this and ``role`` is itself a finding:
    #: a ShareGPT ``from: gpt`` silently becomes zero trainable tokens in
    #: trainers that only look for ``assistant``.
    raw_role: str | None = None
    #: True when the source value was not a string and had to be coerced.
    coerced: bool = False


@dataclass(slots=True)
class Document:
    """One record, normalized.

    ``text`` is always populated and is what format-agnostic checks operate on.
    ``turns`` is populated only for conversational profiles.
    """

    doc_id: str
    text: str
    turns: list[Turn] = field(default_factory=list)
    system: str | None = None
    #: Which file this came from, and which line or row.
    source_file: str = ""
    source_index: int = 0
    #: Dataset this record belongs to. Derived from directory structure during
    #: discovery, never guessed at.
    dataset: str = ""
    #: Raw top-level keys of the source record, for schema reporting.
    raw_keys: tuple[str, ...] = ()
    #: Populated lazily by checks that need it.
    meta: dict[str, Any] = field(default_factory=dict)

    # Derived views, computed on first use and kept. Six checks ask for the
    # assistant text and three ask for a lowercased copy; recomputing a join or
    # a case fold of the whole record once per asker was, measurably, one of the
    # larger allocation sources in the scan.
    _assistant: str | None = None
    _prompt: str | None = None
    _text_lower: str | None = None
    _assistant_lower: str | None = None

    @property
    def assistant_text(self) -> str:
        value = self._assistant
        if value is None:
            value = "\n".join(t.content for t in self.turns if t.role == "assistant")
            self._assistant = value
        return value

    @property
    def prompt_text(self) -> str:
        value = self._prompt
        if value is None:
            value = "\n".join(t.content for t in self.turns if t.role != "assistant")
            self._prompt = value
        return value

    @property
    def text_lower(self) -> str:
        """Lowercased record text, for pattern gates. See ``regexgate``."""
        value = self._text_lower
        if value is None:
            value = self.text.lower()
            self._text_lower = value
        return value

    @property
    def assistant_lower(self) -> str:
        value = self._assistant_lower
        if value is None:
            value = self.assistant_text.lower()
            self._assistant_lower = value
        return value


@dataclass(slots=True)
class Evidence:
    """A concrete example backing a finding.

    Findings must be inspectable. A count with no records behind it is a
    statistic, not a finding.
    """

    doc_id: str
    source_file: str
    source_index: int
    excerpt: str
    #: For pair-shaped findings such as near-duplicates.
    partner_doc_id: str | None = None
    partner_excerpt: str | None = None
    score: float | None = None


@dataclass(slots=True)
class Finding:
    """One check violation, aggregated across the records it affects."""

    check_id: str
    title: str
    severity: Severity
    confidence: Confidence
    #: How many records are affected.
    count: int
    #: Out of how many considered. Lets the report show a rate honestly.
    total_considered: int
    #: One sentence stating what is wrong.
    detail: str
    #: What to do about it. A finding with no remediation is a complaint.
    fix: str
    evidence: list[Evidence] = field(default_factory=list)
    #: Cost of the finding in the user's own units, when computable.
    wasted_tokens: int | None = None
    #: Per-dataset breakdown, when the check is dataset-aware.
    by_dataset: dict[str, int] = field(default_factory=dict)
    #: Free-form structured payload for checks with richer output, such as the
    #: directional overlap matrix.
    data: dict[str, Any] = field(default_factory=dict)
    #: Profiles under which this finding would fail a run. Populated even when no
    #: target was declared, so the report can say "would block under sft" instead
    #: of failing a run whose purpose was never stated.
    would_block_under: tuple[str, ...] = ()
    #: True only when a target profile was declared and this finding matches it.
    is_blocking: bool = False

    @property
    def rate(self) -> float:
        return self.count / self.total_considered if self.total_considered else 0.0


@dataclass(slots=True)
class SkippedCheck:
    """A check that could not run, and the single thing that would unlock it."""

    check_id: str
    title: str
    reason: str
    unlock: str


@dataclass(slots=True)
class DatasetRef:
    """A dataset discovered on disk."""

    name: str
    root: str
    files: list[str] = field(default_factory=list)
    total_bytes: int = 0
    record_count: int = 0
    #: Detected schema layout id, or None when induction failed.
    schema_id: str | None = None
    #: Distribution of schema ids seen, so a mixed folder is itself reportable.
    schema_mix: dict[str, int] = field(default_factory=dict)
    #: Parsed from a dataset card if one was found next to the data.
    license: str | None = None
    declared_language: list[str] = field(default_factory=list)
    unparseable_files: list[str] = field(default_factory=list)


def content_hash(text: str) -> str:
    """Stable per-record identity, used for caching and for evidence refs."""
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


def hash_many(parts: list[str]) -> str:
    """Stable identity for a set of inputs, used for fingerprint provenance."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()
