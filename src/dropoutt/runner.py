"""The scan driver.

One streaming pass over the records. Features that are expensive and wanted by
more than one check (tokenization, template rendering, the loss mask, language
detection) are computed at most once per record and stashed on the document, so
twenty checks still mean one tokenizer pass.

Checks that need global state accumulate during the pass and resolve in
``finalize``. That is the second phase; it is not a second read of the data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import checks as _checks_pkg  # noqa: F401  (registers the catalog)
from .chat_template import (
    ChatTemplate,
    TemplateRenderError,
    char_spans_to_token_mask,
    offsets_are_reliable,
)
from .context import (
    F_LANG,
    F_LANG_CONF,
    F_LOSS_MASK,
    F_RENDERED,
    F_TOKEN_COUNT,
    F_TOKEN_IDS,
    ScanContext,
)
from .checks.base import REGISTRY
from .discovery import Discovery, discover
from .models import Document, Finding, Profile, SkippedCheck
from .normalize import to_document, to_messages
from .readers import RawRecord, read_file
from .schema_induction import SchemaVerdict, induce
from .tokenizer_panel import CHARS_PER_TOKEN_FALLBACK, TokenizerHandle

#: Minimum characters for a record to be placed on the atlas. Below this the
#: embedding is dominated by noise, for the same reason language identification
#: is gated on length.
ATLAS_MIN_CHARS = 80


@dataclass
class ScanResult:
    ctx: ScanContext
    discovery: Discovery
    findings: list[Finding] = field(default_factory=list)
    skipped: list[SkippedCheck] = field(default_factory=list)
    verdicts: dict[str, SchemaVerdict] = field(default_factory=dict)
    elapsed: float = 0.0
    records_scanned: int = 0

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.is_blocking]


def _iter_dataset_records(
    dataset_files: list[str], *, limit: int | None = None
) -> Iterator[RawRecord]:
    for path in dataset_files:
        suffix = Path(path).suffix
        if suffix in (".gz", ".zst", ".bz2", ".xz"):
            inner = Path(path).with_suffix("").suffix
            yield from read_file(path, inner, compressed=True, limit=limit)
        else:
            yield from read_file(path, suffix, limit=limit)


def scan(
    root: str,
    *,
    profile: Profile = Profile.UNKNOWN,
    target: str | None = None,
    model_id: str | None = None,
    seq_len: int | None = None,
    tokenizer: TokenizerHandle | None = None,
    chat_template: ChatTemplate | None = None,
    detector=None,
    contamination=None,
    atlas=None,
    max_tier: int = 1,
    muted: tuple[str, ...] = (),
    limit_per_file: int | None = None,
    induction_sample: int = 2000,
    progress: Callable[[str, int], None] | None = None,
    offline: bool = False,
) -> ScanResult:
    """Run a full scan and return findings plus context.

    ``offline`` must be threaded all the way to the atlas embedder. It is the
    only network call left inside a scan, and a promise of "never touches the
    network" that still downloads a 500 MB model is worse than no promise on a
    compute node with no egress.
    """
    started = time.time()
    disc = discover(root)

    ctx = ScanContext(
        root=disc.root,
        profile=profile,
        target=target,
        seq_len=seq_len,
        model_id=model_id,
        datasets=disc.datasets,
        tokenizer=tokenizer,
        chat_template=chat_template,
        detector=detector,
        contamination=contamination,
        atlas=atlas,
    )
    ctx.stats["not_training_data"] = {}
    ctx.stats["mixed_schemas"] = {}
    # Stratified sample for the cross-tokenizer budget estimate. Capped per
    # dataset so one huge dataset cannot dominate the ratio.
    ctx.stats["budget_sample"] = []
    ctx.stats["total_chars"] = 0
    ctx.stats["total_words"] = 0
    budget_sample: list[str] = ctx.stats["budget_sample"]
    atlas_sample: list[tuple[str, str]] = []
    per_dataset_cap = max(200, 20_000 // max(len(disc.datasets), 1))

    if tokenizer is not None and chat_template is not None:
        _probe_offsets(ctx, tokenizer, chat_template)
    if tokenizer is not None and chat_template is not None and chat_template.eos_token:
        ids = tokenizer.encode(chat_template.eos_token)
        if len(ids) == 1:
            ctx.stats["eos_token_id"] = ids[0]

    # ---- phase 1: induce the layout of each dataset -----------------------
    verdicts: dict[str, SchemaVerdict] = {}
    for ds in disc.datasets:
        sample = list(_iter_dataset_records(ds.files, limit=induction_sample))[:induction_sample]
        verdict = induce(sample)
        verdicts[ds.name] = verdict
        ds.schema_id = verdict.layout_id
        ds.schema_mix = verdict.distribution

        if verdict.not_training_data:
            ctx.stats["not_training_data"][ds.name] = verdict.not_training_data
        if verdict.is_mixed:
            total = sum(v for k, v in verdict.distribution.items() if not k.startswith("_")) or 1
            ctx.stats["mixed_schemas"][ds.name] = {
                k: v / total for k, v in verdict.distribution.items() if not k.startswith("_")
            }
        if verdict.degraded:
            ctx.degraded(f"{ds.name}: {verdict.degraded}")

    # If the profile was not declared, infer it from what the layouts imply.
    if profile is Profile.UNKNOWN:
        ctx.profile = _infer_profile(verdicts)

    # ---- resolve which checks can run ------------------------------------
    active, skipped = REGISTRY.resolve(ctx, max_tier=max_tier, muted=muted)

    # ---- phase 2: one streaming pass -------------------------------------
    scanned = 0
    for ds in disc.datasets:
        verdict = verdicts[ds.name]
        if verdict.not_training_data:
            # Still counted and reported, but not fed to content checks: forcing
            # log records through a chat layout produces confident nonsense.
            continue
        layout = verdict.layout_id
        sampled_here = 0
        for idx, rec in enumerate(_iter_dataset_records(ds.files, limit=limit_per_file)):
            doc = to_document(rec, layout, ds.name, index=idx)
            _compute_features(doc, ctx)
            ctx.stats["total_chars"] += len(doc.text)
            ctx.stats["total_words"] += doc.text.count(" ") + 1 if doc.text else 0
            if sampled_here < per_dataset_cap and len(doc.text) > 20:
                budget_sample.append(doc.text[:4000])
                sampled_here += 1
                if ctx.atlas is not None:
                    # Coverage comes from the same stratified sample: the atlas
                    # describes a distribution, and a sample describes it as
                    # well as every record at a fraction of the embedding cost.
                    #
                    # Records below ATLAS_MIN_CHARS are excluded rather than
                    # placed. A twenty-character record cannot be positioned on
                    # a topical map, and including it inflates the off-atlas
                    # rate with records that were never placeable. The count of
                    # exclusions is reported, not hidden.
                    if len(doc.text) >= ATLAS_MIN_CHARS:
                        atlas_sample.append((doc.text[:2000], doc.meta.get(F_LANG) or "unknown"))
                    else:
                        ctx.stats["atlas_too_short"] = ctx.stats.get("atlas_too_short", 0) + 1
            for check in active:
                try:
                    check.observe(doc, ctx)
                except Exception as exc:  # noqa: BLE001
                    ctx.degraded(f"check {check.check_id} errored: {type(exc).__name__}")
            scanned += 1
            ds.record_count += 1
            if progress is not None and scanned % 2000 == 0:
                progress(ds.name, scanned)

    ctx.total_records = scanned

    # ---- atlas coverage --------------------------------------------------
    if atlas is not None and atlas_sample:
        _compute_coverage(ctx, atlas_sample, offline=offline)

    # ---- phase 3: resolve ------------------------------------------------
    findings: list[Finding] = []
    for check in active:
        try:
            findings.extend(check.finalize(ctx))
        except Exception as exc:  # noqa: BLE001
            ctx.degraded(f"check {check.check_id} failed to finalize: {type(exc).__name__}")

    # Blocking only applies when a target was declared. Without one there is no
    # purpose against which something can be wrong.
    if ctx.blocking_enabled:
        for f in findings:
            f.is_blocking = (
                f.severity.value == "blocking" and ctx.profile.value in f.would_block_under
            )

    findings.sort(key=lambda f: (f.check_id,))
    return ScanResult(
        ctx=ctx, discovery=disc, findings=findings, skipped=skipped,
        verdicts=verdicts, elapsed=time.time() - started, records_scanned=scanned,
    )


def _infer_profile(verdicts: dict[str, SchemaVerdict]) -> Profile:
    """Guess what is being built. Presented as a hypothesis, never enforced."""
    kinds: dict[str, int] = {}
    for v in verdicts.values():
        if v.not_training_data:
            continue
        kinds[v.kind] = kinds.get(v.kind, 0) + 1
    if not kinds:
        return Profile.UNKNOWN
    if kinds.get("preference"):
        return Profile.PREFERENCE
    conversational = kinds.get("conversation", 0) + kinds.get("completion", 0) + kinds.get("qa", 0)
    if conversational >= kinds.get("text", 0):
        return Profile.SFT
    return Profile.CORPUS


def _probe_offsets(ctx: ScanContext, tok: TokenizerHandle, tpl: ChatTemplate) -> None:
    """Check that token offsets index the string we rendered.

    Some pre-tokenizers report offsets into a normalized string we never see.
    When that happens the loss mask cannot be derived correctly, so the
    mask-dependent checks are disabled with an honest reason instead of
    producing wrong numbers.
    """
    probe = [{"role": "user", "content": "merhaba dünya"},
             {"role": "assistant", "content": "selam!"}]
    try:
        rendered = tpl.render(probe).text
    except TemplateRenderError as exc:
        ctx.degraded(f"target chat template failed on a probe record: {exc}")
        ctx.chat_template = None
        return
    if tok._tok is None:  # noqa: SLF001
        return
    enc = tok._tok.encode(rendered, add_special_tokens=False)  # noqa: SLF001
    if not offsets_are_reliable(rendered, list(enc.offsets)):
        ctx.degraded(
            "this tokenizer reports offsets into a normalized string, so the loss mask "
            "cannot be derived reliably; mask-dependent checks were skipped"
        )
        ctx.stats["offsets_unreliable"] = True


def _compute_features(doc: Document, ctx: ScanContext) -> None:
    """Compute the shared per-record features exactly once."""
    text = doc.text

    if ctx.detector is not None and text:
        result = ctx.detector.detect(text)
        doc.meta[F_LANG] = result.lang
        doc.meta[F_LANG_CONF] = result.confidence
        doc.meta["_lang_result"] = result

    if ctx.tokenizer is None:
        doc.meta[F_TOKEN_COUNT] = max(1, int(len(text) / CHARS_PER_TOKEN_FALLBACK))
        return

    if ctx.chat_template is not None and doc.turns and not ctx.stats.get("offsets_unreliable"):
        try:
            messages = to_messages(doc)
            result = ctx.chat_template.render(messages)
            spans = result.generation_spans
            if not result.spans_from_tag:
                rendered, spans = ctx.chat_template.spans_by_difference(messages)
                result_text = rendered
            else:
                result_text = result.text
            doc.meta[F_RENDERED] = result_text
            enc = ctx.tokenizer._tok.encode(result_text, add_special_tokens=False)  # noqa: SLF001
            doc.meta[F_TOKEN_IDS] = list(enc.ids)
            doc.meta[F_TOKEN_COUNT] = len(enc.ids)
            doc.meta[F_LOSS_MASK] = char_spans_to_token_mask(list(enc.offsets), spans)
            return
        except TemplateRenderError as exc:
            doc.meta["render_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            doc.meta["render_error"] = f"{type(exc).__name__}: {exc}"

    doc.meta[F_TOKEN_COUNT] = ctx.tokenizer.count(text)


def _compute_coverage(
    ctx: ScanContext, sample: list[tuple[str, str]], *, offline: bool = False
) -> None:
    """Place a sample of records on the atlas and build the coverage report.

    Degrades rather than failing: if the embedder cannot be loaded, coverage is
    marked unavailable with the reason instead of the scan aborting.
    """
    from .atlas import load_embedder  # noqa: PLC0415

    embedder = load_embedder(ctx.atlas.embed_model, offline=offline)
    if embedder is None:
        ctx.degraded(
            "atlas is present but its embedding model could not be loaded; "
            "coverage was not computed"
        )
        return
    if embedder.dim != ctx.atlas.dim:
        ctx.degraded(
            f"atlas expects {ctx.atlas.dim}-dimensional embeddings but the loaded "
            f"model produces {embedder.dim}; coverage was not computed"
        )
        return

    texts = [t for t, _ in sample]
    langs = [lang for _, lang in sample]
    try:
        emb = embedder.encode(texts)
        regions, _scores = ctx.atlas.assign(emb)
        categories = ctx.atlas.categorize(emb)
    except Exception as exc:  # noqa: BLE001
        ctx.degraded(f"atlas assignment failed: {type(exc).__name__}")
        return

    coverage = ctx.atlas.coverage(regions, categories, langs)
    coverage["sampled_records"] = len(texts)
    too_short = ctx.stats.get("atlas_too_short", 0)
    if too_short:
        coverage["excluded_too_short"] = too_short
        coverage["min_chars"] = ATLAS_MIN_CHARS
    coverage["atlas_hash"] = ctx.atlas.artifact_hash
    ctx.stats["atlas_coverage"] = coverage
