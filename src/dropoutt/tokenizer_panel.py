"""Token counting, and the cross-tokenizer budget comparison.

Two modes.

With ``--model``, one tokenizer is loaded and the corpus is tokenized exactly
once. Exact per-record counts are needed anyway for the truncation forecast and
packing efficiency, so nothing is estimated.

Without ``--model``, refusing to count tokens would be correct and useless.
Instead a panel of common tokenizers is run over a sample and the total is
extrapolated from exact character counts.

The extrapolation is stratified by dataset, and that is the whole ballgame.
Tokens-per-character is stable *within* one body of text and varies enormously
between them: the same sentence costs about 0.19 tokens per character in
English and 0.28 in Turkish, and source code is different again. The scan's
sample is capped per dataset so that one huge dataset cannot swamp the atlas
histogram, which means the sample is not a miniature of the corpus — a corpus
that is 94% English by character can easily produce a sample that is only 60%
English. Pooling that sample into a single ratio therefore prices the whole
corpus at the blended rate of the *sample*, and measured against exact counts
on such a corpus it overstated the budget by 12% to 38% depending on the
tokenizer, while reporting a ±1% interval.

So each dataset's ratio is measured against that dataset's own character count
and the stratum totals are added. Same sample, same tokenizer calls; the error
on the corpus above drops to 0.05%. The interval is the stratified ratio
estimator's standard error with a finite-population correction, so a dataset
that was sampled in full contributes no uncertainty rather than inventing some.

For non-English data this turns the missing model into the most useful output of
a zero-configuration run: the same Turkish corpus can cost meaningfully more
tokens under one tokenizer family than another, and that is a compute decision
worth making before any training run exists.
"""

from __future__ import annotations

import contextlib
import math
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

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


def warm_panel(*, offline: bool = False) -> None:
    """Load every panel tokenizer into the cache.

    Called from a background thread while the scan reads records: it touches no
    scan state, and both the file read and the Rust parse release the GIL, so
    it is two seconds of the run that cost nothing.
    """
    if not HAVE_TOKENIZERS:
        return
    from concurrent.futures import ThreadPoolExecutor

    def _one(model_id: str) -> None:
        with contextlib.suppress(Exception):
            load_tokenizer(model_id, offline=offline)

    with ThreadPoolExecutor(max_workers=len(PANEL)) as pool:
        list(pool.map(_one, [model_id for _name, model_id in PANEL]))


@dataclass(slots=True)
class TokenizerHandle:
    """A loaded tokenizer, or a character-ratio stand-in."""

    name: str
    model_id: str | None
    #: A ``tokenizers.Tokenizer`` when one loaded. Typed loosely because
    #: ``tokenizers`` is an optional extra and must not be imported to describe
    #: the attribute that holds it.
    _tok: Any = None
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


#: One lock per tokenizer rather than one for all of them: the background
#: warm-up and the scan must not load the *same* tokenizer twice, but the five
#: panel members are meant to load at once.
_KEY_LOCKS: dict[tuple[str, bool], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def load_tokenizer(model_id: str, *, offline: bool = False) -> TokenizerHandle | None:
    """Load a tokenizer by Hub id or local path. Returns None on failure.

    Cached, so the CLI can warm the comparison panel in the background while the
    scan runs and find it already here afterwards. Loading five tokenizers is
    about two seconds and it does not depend on the data at all.
    """
    key = (model_id, offline)
    with _LOCKS_GUARD:
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        return _load_tokenizer(model_id, offline=offline)


@lru_cache(maxsize=16)
def _load_tokenizer(model_id: str, *, offline: bool = False) -> TokenizerHandle | None:
    if not HAVE_TOKENIZERS:
        return None
    from tokenizers import Tokenizer

    try:
        import os

        if os.path.isdir(model_id):
            tok = Tokenizer.from_file(os.path.join(model_id, "tokenizer.json"))
        elif os.path.isfile(model_id):
            tok = Tokenizer.from_file(model_id)
        elif offline:
            # Offline means "never reach the network", not "never use a
            # tokenizer". The point of running `dropoutt fetch` on a login node
            # is that the compute node can then work with no egress, so a
            # tokenizer already in the cache must still load. HF_HUB_OFFLINE
            # makes from_pretrained resolve purely from the local cache and
            # raise rather than connect.
            prior = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                tok = Tokenizer.from_pretrained(model_id)
            finally:
                if prior is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prior
        else:
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

    import hashlib

    digest = hashlib.blake2b(tok.to_str().encode("utf-8"), digest_size=8).hexdigest()
    return TokenizerHandle(name=model_id.rsplit("/", maxsplit=1)[-1], model_id=model_id, _tok=tok,
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

    @property
    def margin_share(self) -> float:
        if not self.total_tokens_est:
            return 0.0
        return self.margin / self.total_tokens_est


@dataclass(slots=True)
class BudgetReport:
    total_chars: int
    total_words: int
    sample_size: int
    estimates: list[PanelEstimate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: True when the corpus was tokenized rather than sampled.
    exact: bool = False
    #: Share of the corpus's characters that sat in a dataset the sample could
    #: price. Anything below 1.0 means some text was extrapolated at the
    #: corpus-wide rate instead of its own.
    covered_share: float = 1.0

    @property
    def cheapest(self) -> PanelEstimate | None:
        ok = [e for e in self.estimates if not e.failed]
        return min(ok, key=lambda e: e.total_tokens_est) if ok else None

    def premium_vs_cheapest(self, est: PanelEstimate) -> float:
        base = self.cheapest
        if base is None or base.total_tokens_est == 0:
            return 0.0
        return (est.total_tokens_est / base.total_tokens_est) - 1.0


#: Characters fed to the panel. Tokens per character is a corpus-level ratio, so
#: precision goes as the square root of the sample and there is nothing to buy
#: past a few million characters: at four million the standard error of the
#: estimate is already far below the difference between any two tokenizers in
#: the panel, which is what the table exists to show. Tokenizing forty million
#: characters five times was, measured, a quarter of the runtime of a
#: zero-configuration scan.
BUDGET_SAMPLE_CHARS = 4_000_000

#: Floor on any one dataset's slice of that budget. A dataset holding 0.1% of
#: the corpus still gets priced at its own rate rather than a neighbour's, and
#: at this size its ratio is already good to a fraction of a percent.
MIN_STRATUM_CHARS = 120_000


def _thin(texts: list[str], budget: int) -> list[str]:
    """Take an evenly spaced subsample under a character budget.

    Evenly spaced rather than random, so the same corpus gives the same number
    twice, and so the subsample keeps whatever ordering structure the sample
    already had.
    """
    total = sum(len(t) for t in texts)
    if total <= budget or not texts:
        return texts
    stride = max(2, (total + budget - 1) // budget)
    return texts[::stride]


@dataclass(slots=True)
class _Stratum:
    """One dataset: what was sampled from it, and how big it really is."""

    name: str
    texts: list[str]
    #: Characters of normalised text in the whole dataset, not the sample.
    total_chars: int
    #: Records in the whole dataset, for the finite-population correction.
    total_records: int


def _plan_strata(
    sample: dict[str, list[str]],
    chars_by_dataset: dict[str, int],
    records_by_dataset: dict[str, int],
    total_chars: int,
) -> tuple[list[_Stratum], int, float]:
    """Thin each dataset's sample, keeping the split, under one global budget.

    The budget is shared out in proportion to how much of the corpus each
    dataset is, because that is where precision is worth buying — but every
    dataset keeps a floor, since a dataset priced at another dataset's rate is
    the error this whole design exists to avoid.
    """
    names = [n for n in sorted(sample) if any(sample[n])]
    sampled_chars = {
        n: sum(len(t) for t in sample[n] if t) for n in names
    }
    # A dataset the scan never measured characters for still has to be given a
    # size. Splitting the corpus total by the strata's sampled character shares
    # is the only estimate available, and it is exactly right in the case that
    # actually produces a missing total: one dataset holding everything.
    unmeasured = sum(sampled_chars[n] for n in names if not chars_by_dataset.get(n))
    unclaimed = max(0, total_chars - sum(
        chars_by_dataset.get(n, 0) for n in names
    ))
    priced = {
        n: chars_by_dataset.get(n)
        or (int(unclaimed * sampled_chars[n] / unmeasured) if unmeasured else 0)
        for n in names
    }
    priced_total = sum(priced.values())
    strata: list[_Stratum] = []
    for name in names:
        weight = (priced[name] / priced_total) if priced_total else 0.0
        share = max(MIN_STRATUM_CHARS, int(BUDGET_SAMPLE_CHARS * weight))
        thinned = _thin([t for t in sample[name] if t], share)
        strata.append(
            _Stratum(
                name=name,
                texts=thinned,
                total_chars=priced[name],
                total_records=records_by_dataset.get(name, 0) or len(sample[name]),
            )
        )
    # Characters sitting in datasets that produced no sample at all. They still
    # have to be priced, and the only rate available is the corpus-wide one.
    covered = sum(s.total_chars for s in strata)
    unpriced = max(0, total_chars - covered)
    covered_share = (covered / total_chars) if total_chars else 1.0
    return strata, unpriced, covered_share


def estimate_budget(
    sample: dict[str, list[str]] | list[str],
    total_chars: int,
    total_words: int,
    *,
    chars_by_dataset: dict[str, int] | None = None,
    records_by_dataset: dict[str, int] | None = None,
    offline: bool = False,
) -> BudgetReport:
    """Estimate total tokens under each panel tokenizer from a sample.

    ``sample`` maps dataset name to that dataset's sampled texts. A bare list is
    accepted and treated as one stratum, which is the correct reading for a
    single-dataset corpus and the only thing older callers ever passed.
    """
    if not isinstance(sample, dict):
        sample = {"": list(sample)}
    chars_by_dataset = dict(chars_by_dataset or {})
    records_by_dataset = dict(records_by_dataset or {})

    strata, unpriced_chars, covered_share = _plan_strata(
        sample, chars_by_dataset, records_by_dataset, total_chars
    )
    sample_size = sum(len(s.texts) for s in strata)
    report = BudgetReport(total_chars=total_chars, total_words=total_words,
                          sample_size=sample_size, covered_share=covered_share)
    if not strata:
        return report
    if covered_share < 0.999:
        report.notes.append(
            f"{1 - covered_share:.0%} of the corpus sat in datasets the sample "
            f"could not price and was extrapolated at the corpus-wide rate."
        )

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

    flat = [t for s in strata for t in s.texts]

    # The five panel members are independent, and both halves of the work
    # release the GIL: loading parses a tokenizer.json in Rust, and
    # ``encode_batch_fast`` is a rayon-parallel Rust call. Running them in
    # threads turns five serial tokenizer passes into roughly one. Every
    # tokenizer sees one flat list so batching stays large; the stratum
    # boundaries are recovered by offset afterwards.
    from concurrent.futures import ThreadPoolExecutor

    def _run(entry: tuple[str, str]) -> tuple[str, str, list[int] | None]:
        name, model_id = entry
        handle = load_tokenizer(model_id, offline=offline)
        if handle is None:
            return name, model_id, None
        return name, model_id, handle.count_batch(flat)

    with ThreadPoolExecutor(max_workers=len(PANEL)) as pool:
        panel_results = list(pool.map(_run, PANEL))

    for name, model_id, counts in panel_results:
        if counts is None:
            report.estimates.append(PanelEstimate(name, model_id, 0.0, 0, 0.0, failed=True))
            continue
        report.estimates.append(
            _stratified(name, model_id, strata, counts, total_chars,
                        total_words, unpriced_chars)
        )
    return report


def _stratified(
    name: str,
    model_id: str,
    strata: list[_Stratum],
    counts: list[int],
    total_chars: int,
    total_words: int,
    unpriced_chars: int,
) -> PanelEstimate:
    """Add up per-dataset ratio estimates, and the variance that goes with them.

    Each dataset contributes ``C_d * (t_d / c_d)``: its own characters priced at
    its own measured rate. The variance is the textbook ratio-estimator one,

        Var(T_d) = (C_d / c̄_d)² · (1 − n_d/N_d) · s_d² / n_d,
        s_d² = Σ (t_i − R_d·c_i)² / (n_d − 1),

    where the residual ``t_i − R_d·c_i`` is what a record costs above or below
    what its length predicts. The finite-population correction matters here and
    not in the usual textbook setting: a small dataset is often sampled whole,
    and a whole dataset has been counted, not estimated.
    """
    total_est = 0.0
    variance = 0.0
    sample_tokens = 0
    sample_chars = 0
    sample_words = 0
    offset = 0

    for stratum in strata:
        n = len(stratum.texts)
        tok = counts[offset:offset + n]
        offset += n
        chars = [len(t) for t in stratum.texts]
        c_sum = sum(chars)
        t_sum = sum(tok)
        sample_tokens += t_sum
        sample_chars += c_sum
        sample_words += sum(t.count(" ") + 1 for t in stratum.texts)
        if c_sum == 0:
            continue
        ratio = t_sum / c_sum
        total_est += stratum.total_chars * ratio

        if n < 2:
            continue
        mean_chars = c_sum / n
        if mean_chars <= 0:
            continue
        residual = sum((t - ratio * c) ** 2 for t, c in zip(tok, chars, strict=True)) / (n - 1)
        fpc = max(0.0, 1.0 - (n / stratum.total_records)) if stratum.total_records else 1.0
        variance += (stratum.total_chars / mean_chars) ** 2 * fpc * residual / n

    # Datasets that produced no sample: priced at the blended rate, and the
    # uncertainty of doing so is not something a sample can measure, so it is
    # named in a note rather than folded into a number that would look precise.
    blended = (sample_tokens / sample_chars) if sample_chars else 0.0
    total_est += unpriced_chars * blended

    return PanelEstimate(
        name=name,
        model_id=model_id,
        tokens_per_char=(total_est / total_chars) if total_chars else blended,
        total_tokens_est=int(total_est),
        tokens_per_word=sample_tokens / max(sample_words, 1),
        margin=int(1.96 * math.sqrt(variance)),
    )
