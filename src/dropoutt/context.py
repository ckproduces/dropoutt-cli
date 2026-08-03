"""Scan context: everything shared across checks during one run.

The context also owns the per-record features that are expensive to compute and
needed by more than one check (token ids, the rendered chat template and its
loss mask, the detected language, the shingle set). The runner computes each of
those at most once per record and stashes it under a documented key in
``Document.meta``, so twenty checks still mean one tokenizer pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .models import DatasetRef, Profile, Requirement

if TYPE_CHECKING:  # pragma: no cover
    from .atlas.apply import Atlas
    from .chat_template import ChatTemplate
    from .contamination import ContaminationIndex
    from .langid import LanguageDetector
    from .tokenizer_panel import TokenizerHandle

# Documented keys written into Document.meta by the runner.
F_TOKEN_IDS = "token_ids"          # list[int] | None
F_TOKEN_COUNT = "token_count"      # int (exact if tokenizer, else estimated)
F_TOKEN_ESTIMATED = "token_estimated"  # bool
F_RENDERED = "rendered"            # str | None, chat template applied
F_LOSS_MASK = "loss_mask"          # list[bool] | None, aligned with token_ids
F_LANG = "lang"                    # str | None
F_LANG_CONF = "lang_conf"          # float | None
F_NORM_TEXT = "norm_text"          # str, whitespace/case normalized
F_SHINGLES = "shingles"            # list[int] hashed n-grams
F_DEDUP_WORDS = "dedup_words"      # list[str], normalized word sequence


def dedup_words_of(doc) -> list[str]:
    """The normalized word sequence for one record, computed at most once.

    Near-duplicate detection and contamination scanning both shingle the same
    string. Each used to run the full normalization itself — an NFC pass, a
    Turkish-aware case fold and two regex substitutions over every record,
    twice.
    """
    words = doc.meta.get(F_DEDUP_WORDS)
    if words is None:
        from .textutil import dedup_words

        words = dedup_words(doc.text)
        doc.meta[F_DEDUP_WORDS] = words
    return words


@dataclass
class ScanContext:
    """Shared state for one scan."""

    root: str
    profile: Profile = Profile.UNKNOWN
    #: Set only when the user declared what they are building. Without it, no
    #: finding may produce a non-zero exit code.
    target: str | None = None
    seq_len: int | None = None
    model_id: str | None = None

    datasets: list[DatasetRef] = field(default_factory=list)
    tokenizer: TokenizerHandle | None = None
    chat_template: ChatTemplate | None = None
    detector: LanguageDetector | None = None
    atlas: Atlas | None = None
    contamination: ContaminationIndex | None = None

    #: Total records seen, filled in by the runner.
    total_records: int = 0
    #: Anything the scan wants to tell the user that is not a finding.
    notes: list[str] = field(default_factory=list)
    #: Populated when a subsystem degraded rather than failed.
    degradations: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def has(self, req: Requirement) -> bool:
        if req is Requirement.NONE:
            return True
        if req is Requirement.TOKENIZER:
            return self.tokenizer is not None
        if req is Requirement.CHAT_TEMPLATE:
            return self.chat_template is not None
        if req is Requirement.SEQ_LEN:
            return self.seq_len is not None
        if req is Requirement.LANGID:
            return self.detector is not None
        if req is Requirement.EMBEDDINGS:
            return self.atlas is not None
        if req is Requirement.ATLAS:
            return self.atlas is not None
        if req is Requirement.CONTAMINATION_INDEX:
            return self.contamination is not None and not self.contamination.is_empty
        if req is Requirement.MULTIPLE_DATASETS:
            return len(self.datasets) > 1
        return False

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    def degraded(self, msg: str) -> None:
        """Record that something fell back rather than failing.

        Degrading quietly is worse than erroring, so everything recorded here is
        surfaced in the report.
        """
        if msg not in self.degradations:
            self.degradations.append(msg)

    @property
    def blocking_enabled(self) -> bool:
        """Whether a blocking finding may set a non-zero exit code.

        Blocking asserts that something is wrong *for a purpose*. With no target
        declared there is no purpose, so we report instead of failing.
        """
        return self.target is not None
