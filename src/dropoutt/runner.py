"""The scan driver.

One streaming pass over the records. Features that are expensive and wanted by
more than one check (tokenization, template rendering, the loss mask, language
detection) are computed at most once per record and stashed on the document, so
twenty checks still mean one tokenizer pass.

Checks that need global state accumulate during the pass and resolve in
``finalize``. That is the second phase; it is not a second read of the data.
"""

from __future__ import annotations

import hashlib
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
    minhash_preset: str = "fineweb",
    muted: tuple[str, ...] = (),
    limit_per_file: int | None = None,
    induction_sample: int = 2000,
    progress: Callable[[str, int], None] | None = None,
    phase: Callable[[str], None] | None = None,
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
    ctx.stats["minhash_preset"] = minhash_preset
    # Stratified sample for the cross-tokenizer budget estimate. Capped per
    # dataset so one huge dataset cannot dominate the ratio.
    ctx.stats["budget_sample"] = []
    ctx.stats["total_chars"] = 0
    ctx.stats["total_words"] = 0
    content_hasher = hashlib.blake2b(digest_size=16)
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
    if phase is not None:
        phase("Inferring dataset layouts")
    verdicts: dict[str, SchemaVerdict] = {}
    ctx.stats["text_framing"] = _sniff_text_files(disc)
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
    if phase is not None:
        phase("Scanning records")
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
            _update_content_hash(content_hasher, doc)
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
                        # The embedder sees the first 2000 characters; the length
                        # kept here is the record's real one, because the
                        # off-atlas attribution is a statement about the data,
                        # not about what was fed to the model.
                        atlas_sample.append((
                            doc.text[:2000],
                            doc.meta.get(F_LANG) or "unknown",
                            ds.name,
                            len(doc.text),
                        ))
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
    ctx.stats["content_hash"] = content_hasher.hexdigest()

    # ---- atlas coverage --------------------------------------------------
    if atlas is not None and atlas_sample:
        if phase is not None:
            phase("Mapping atlas coverage")
        _compute_coverage(ctx, atlas_sample, offline=offline)

    # ---- phase 3: resolve ------------------------------------------------
    if phase is not None:
        phase("Finalizing checks")
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
                f.severity.value == "blocking" and ctx.target in f.would_block_under
            )

    findings.sort(key=lambda f: (f.check_id,))
    return ScanResult(
        ctx=ctx, discovery=disc, findings=findings, skipped=skipped,
        verdicts=verdicts, elapsed=time.time() - started, records_scanned=scanned,
    )


#: How many text files per dataset to sniff. The framing of a generation run is
#: a property of the run, not of the file, so a sample settles it; sniffing 250
#: shards to learn the same thing 250 times is wasted IO.
SNIFF_FILES_PER_DATASET = 12


def _sniff_text_files(disc: Discovery) -> dict[str, dict]:
    """Work out which datasets keep their records inside .txt or .md files.

    Runs before induction because it changes what induction sees: without it a
    folder of JSON-bearing .txt files induces as one bare-text document per
    file, and the profile is inferred from that.
    """
    from .sniff import sniff_file  # noqa: PLC0415

    out: dict[str, dict] = {}
    for ds in disc.datasets:
        candidates = [f for f in ds.files if Path(f).suffix.lower() in (".txt", ".md")]
        if not candidates:
            continue
        hits: dict[str, dict] = {}
        for path in candidates[:SNIFF_FILES_PER_DATASET]:
            framing = sniff_file(path)
            if not framing.is_records:
                continue
            entry = hits.setdefault(framing.kind, {
                "kind": framing.kind, "matched": 0, "scaffolding": [],
                "parse_rate": round(framing.parse_rate, 4),
            })
            entry["matched"] += 1
            for s in framing.scaffolding:
                if s not in entry["scaffolding"] and len(entry["scaffolding"]) < 5:
                    entry["scaffolding"].append(s)
        if not hits:
            continue
        # One entry per dataset, naming the framing that dominated. A dataset
        # mixing framings is unusual enough that the majority plus the sample
        # size is more useful than a breakdown nobody asked for.
        best = max(hits.values(), key=lambda h: h["matched"])
        best["files"] = len(candidates)
        best["sampled"] = min(len(candidates), SNIFF_FILES_PER_DATASET)
        out[ds.name] = best
    return out


def _update_content_hash(hasher, doc: Document) -> None:
    """Hash normalized content plus the structure that changes scan results."""
    parts = [
        doc.dataset,
        doc.text,
        doc.system or "",
        "\x1f".join(doc.raw_keys),
    ]
    for turn in doc.turns:
        parts.extend([
            turn.role,
            turn.raw_role or "",
            turn.content,
            "1" if turn.coerced else "0",
        ])
    for part in parts:
        raw = part.encode("utf-8", "surrogatepass")
        hasher.update(len(raw).to_bytes(8, "little"))
        hasher.update(raw)


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
    ctx: ScanContext,
    sample: list[tuple[str, str, str, int]],
    *,
    offline: bool = False,
    keep_examples: bool = True,
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

    texts = [row[0] for row in sample]
    langs = [row[1] for row in sample]
    datasets = [row[2] for row in sample]
    lengths = [row[3] for row in sample]
    try:
        emb = embedder.encode(texts)
        regions, scores, nearest = ctx.atlas.assign_full(emb)
        categories = ctx.atlas.categorize(emb)
    except Exception as exc:  # noqa: BLE001
        ctx.degraded(f"atlas assignment failed: {type(exc).__name__}")
        return

    coverage = ctx.atlas.coverage(
        regions, categories, langs,
        scores=scores, nearest=nearest, embeddings=emb,
        lengths=lengths, datasets=datasets, texts=texts,
    )
    coverage["sampled_records"] = len(texts)

    # Excerpts of the furthest off-atlas records, kept out of the coverage facet
    # on purpose. The facet is copied verbatim into fingerprint.json, which is
    # the artifact meant to be shareable, and record text is exactly what does
    # not belong there. The report reads these from ctx.stats instead.
    if keep_examples:
        # What a region actually contains, in the user's own words. The atlas
        # ships five frequency terms per region, which name the region in the
        # reference corpus and say nothing about what landed there from *this*
        # corpus. A region called "film, movie, films, filmi, best" is opaque
        # until you see the record of yours that sits closest to its centre.
        #
        # Held in ctx.stats rather than in the coverage facet, for the same
        # reason as the off-atlas excerpts: the facet is copied into
        # fingerprint.json, and record text must never reach a shareable file.
        by_region: dict[int, list[tuple[float, int]]] = {}
        for i, r in enumerate(regions):
            if r < 0:
                continue
            by_region.setdefault(int(r), []).append((float(scores[i]), i))
        ctx.stats["atlas_region_examples"] = {
            region: [
                {"score": round(s, 4), "excerpt": texts[i][:160], "dataset": datasets[i]}
                for s, i in sorted(rows, reverse=True)[:2]
            ]
            for region, rows in by_region.items()
        }
        # How alike the records inside each crowded region are. A region holding
        # a third of the corpus is not a problem; a region holding a third of
        # the corpus whose records are 0.9 alike is the same document written
        # out many times, and the near-duplicate check will miss it whenever the
        # wording varies more than the shingles tolerate.
        from .atlas.apply import _mean_pairwise  # noqa: PLC0415

        import numpy as _np

        unit = _np.asarray(emb, dtype=_np.float32)
        unit = unit / (_np.linalg.norm(unit, axis=1, keepdims=True) + 1e-9)
        ctx.stats["atlas_region_cohesion"] = {
            region: _mean_pairwise(unit[[i for _s, i in rows]])
            for region, rows in by_region.items()
            if len(rows) >= 20
        }

        off_idx = [i for i, r in enumerate(regions) if r < 0]
        off_idx.sort(key=lambda i: float(scores[i]))
        ctx.stats["atlas_off_examples"] = [
            {
                "score": round(float(scores[i]), 4),
                "chars": lengths[i],
                "dataset": datasets[i],
                "language": langs[i],
                "nearest_region": int(nearest[i]),
                "excerpt": texts[i][:160],
            }
            for i in off_idx[:5]
        ]
    too_short = ctx.stats.get("atlas_too_short", 0)
    if too_short:
        coverage["excluded_too_short"] = too_short
        coverage["min_chars"] = ATLAS_MIN_CHARS
    coverage["atlas_hash"] = ctx.atlas.artifact_hash
    ctx.stats["atlas_coverage"] = coverage
