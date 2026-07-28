"""Token counting, and the cross-tokenizer budget comparison.

Two modes.

With ``--model``, one tokenizer is loaded and the corpus is tokenized exactly
once. Exact per-record counts are needed anyway for the truncation forecast and
packing efficiency, so nothing is estimated.

Without ``--model``, refusing to count tokens would be correct and useless.
Instead a panel of common tokenizers is run over a stratified sample and the
total is extrapolated from exact character counts. Tokens-per-character is
stable within a corpus, so a few hundred thousand sampled records puts the
standard error well under a percent, and the result is reported as an interval.

For non-English data this turns the missing model into the most useful output of
a zero-configuration run: the same Turkish corpus can cost meaningfully more
tokens under one tokenizer family than another, and that is a compute decision
worth making before any training run exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .compat import HAVE_TOKENIZERS

#: The comparison panel. Chosen to span the tokenizer families people actually
#: train on, not to be exhaustive.
PANEL = [
    ("Qwen3", "Qwen/Qwen3-8B"),
    ("Llama-3.1", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("Gemma-2", "unsloth/gemma-2-9b-it"),
    ("Mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("SmolLM2", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
]

#: Rough fallback when no tokenizer is installed at all. Deliberately crude, and
#: always reported as an estimate with the reason attached.
CHARS_PER_TOKEN_FALLBACK = 3.6


@dataclass(slots=True)
class TokenizerHandle:
    """A loaded tokenizer, or a character-ratio stand-in."""

    name: str
    model_id: str | None
    _tok: object | None = None
    #: Set when no real tokenizer could be loaded.
    estimated: bool = False
    tokenizer_hash: str = ""

    def count(self, text: str) -> int:
        if self._tok is None:
            return max(1, int(len(text) / CHARS_PER_TOKEN_FALLBACK))
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def encode(self, text: str) -> list[int]:
        if self._tok is None:
            return []
        return self._tok.encode(text, add_special_tokens=False).ids

    def count_batch(self, texts: list[str]) -> list[int]:
        if self._tok is None:
            return [max(1, int(len(t) / CHARS_PER_TOKEN_FALLBACK)) for t in texts]
        # encode_batch_fast skips offset tracking, which is pure overhead when
        # only the length is wanted.
        encs = self._tok.encode_batch_fast(texts, add_special_tokens=False)
        return [len(e.ids) for e in encs]


def load_tokenizer(model_id: str, *, offline: bool = False) -> TokenizerHandle | None:
    """Load a tokenizer by Hub id or local path. Returns None on failure."""
    if not HAVE_TOKENIZERS:
        return None
    from tokenizers import Tokenizer  # noqa: PLC0415

    try:
        import os  # noqa: PLC0415

        if os.path.isdir(model_id):
            tok = Tokenizer.from_file(os.path.join(model_id, "tokenizer.json"))
        elif os.path.isfile(model_id):
            tok = Tokenizer.from_file(model_id)
        else:
            if offline:
                return None
            tok = Tokenizer.from_pretrained(model_id)
    except Exception:
        return None

    # A tokenizer.json carrying a padding or truncation config would silently
    # change every count. Clear both.
    try:
        tok.no_padding()
        tok.no_truncation()
    except Exception:
        pass

    import hashlib  # noqa: PLC0415

    digest = hashlib.blake2b(tok.to_str().encode("utf-8"), digest_size=8).hexdigest()
    return TokenizerHandle(name=model_id.split("/")[-1], model_id=model_id, _tok=tok,
                           tokenizer_hash=digest)


@dataclass(slots=True)
class PanelEstimate:
    name: str
    model_id: str
    tokens_per_char: float
    total_tokens_est: int
    tokens_per_word: float
    #: Half-width of the reported interval, from the sample standard error.
    margin: int = 0
    failed: bool = False

    @property
    def interval(self) -> tuple[int, int]:
        return (self.total_tokens_est - self.margin, self.total_tokens_est + self.margin)


@dataclass(slots=True)
class BudgetReport:
    total_chars: int
    total_words: int
    sample_size: int
    estimates: list[PanelEstimate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def cheapest(self) -> PanelEstimate | None:
        ok = [e for e in self.estimates if not e.failed]
        return min(ok, key=lambda e: e.total_tokens_est) if ok else None

    def premium_vs_cheapest(self, est: PanelEstimate) -> float:
        base = self.cheapest
        if base is None or base.total_tokens_est == 0:
            return 0.0
        return (est.total_tokens_est / base.total_tokens_est) - 1.0


def estimate_budget(
    sample_texts: list[str],
    total_chars: int,
    total_words: int,
    *,
    offline: bool = False,
) -> BudgetReport:
    """Estimate total tokens under each panel tokenizer from a sample."""
    report = BudgetReport(total_chars=total_chars, total_words=total_words,
                          sample_size=len(sample_texts))
    if not sample_texts:
        return report

    if not HAVE_TOKENIZERS:
        report.notes.append(
            "No tokenizer backend installed, so token counts are a crude "
            "character-ratio estimate. Install 'dropoutt[tokenizer]' for real counts."
        )
        est = total_chars / CHARS_PER_TOKEN_FALLBACK
        report.estimates.append(
            PanelEstimate("character-ratio", "-", 1 / CHARS_PER_TOKEN_FALLBACK,
                          int(est), est / max(total_words, 1))
        )
        return report

    sample_chars = sum(len(t) for t in sample_texts)
    sample_words = sum(len(t.split()) for t in sample_texts)
    if sample_chars == 0:
        return report

    for name, model_id in PANEL:
        handle = load_tokenizer(model_id, offline=offline)
        if handle is None:
            report.estimates.append(PanelEstimate(name, model_id, 0.0, 0, 0.0, failed=True))
            continue
        counts = handle.count_batch(sample_texts)
        sample_tokens = sum(counts)
        tpc = sample_tokens / sample_chars
        total_est = int(total_chars * tpc)

        # Standard error of the per-record tokens-per-char ratio, scaled up.
        ratios = [c / max(len(t), 1) for c, t in zip(counts, sample_texts)]
        if len(ratios) > 1:
            mean = sum(ratios) / len(ratios)
            var = sum((r - mean) ** 2 for r in ratios) / (len(ratios) - 1)
            se = math.sqrt(var / len(ratios))
            margin = int(total_chars * 1.96 * se)
        else:
            margin = 0

        report.estimates.append(
            PanelEstimate(
                name=name,
                model_id=model_id,
                tokens_per_char=tpc,
                total_tokens_est=total_est,
                tokens_per_word=sample_tokens / max(sample_words, 1),
                margin=margin,
            )
        )

    return report
