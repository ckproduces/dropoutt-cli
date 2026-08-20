"""Running the streaming pass across processes.

The pass *is* the scan: one read of every record with every active check
watching it. It is pure Python and pure CPU, so on a machine with cores to spare
the only way to make it dramatically faster is to use them.

The design is deliberately boring. A shard is a contiguous slice of the corpus,
in the order a serial scan would read it. Each worker runs the *real* checks
over its slice — not a reduced version, not a sample — and hands its check
objects back. The parent folds them together in shard order through
``Check.merge``. Nothing inside a check knows that a shard exists, and a check
that does not say how it merges fails a test rather than quietly reporting a
smaller number.

The serial path is not a separate implementation. One shard covering everything
runs through exactly the same function in the calling process, so there is no
second code path to keep in agreement.

Four things elsewhere had to change to make the sharded result *identical*
rather than merely similar.

Record numbering. ``source_index`` reaches evidence, duplicate keys and the
fingerprint, so a shard has to number its records exactly as a serial read
would. That is why splitting a line-delimited file costs one sequential pass to
find the record index at each cut, rather than guessing from byte offsets.

Duplicate keys. ``ExactDuplicates`` keyed records by ``hash()``, which Python
randomises per interpreter. In separate processes the same text would hash
differently and every duplicate would count as unique.

The content digest, which was a chained hash over records in order. That would
have made the fingerprint id depend on how many shards the machine chose. It is
a sum now, so it depends on the records and nothing else.

Sampling. The atlas and token-budget samples used to be the first N records of
each dataset, which no shard can reproduce. They are now a bottom-k sample over
a per-record hash: the k smallest keys of a union are the k smallest keys of the
per-shard bottom-k, so the sample is identical however the corpus is divided.
It is also a better sample — the head of a dataset is not a random part of it
when the files are sorted by source, length or date, which they usually are.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Profile
from .readers import WHOLE_FILE, RawRecord, ReadSpan, plan_splits, read_file

#: Below this many bytes a scan finishes before workers would finish starting.
#: Process start-up plus handing state back costs roughly a second, and a serial
#: pass covers this much in about that time.
MIN_BYTES_FOR_PARALLEL = 24 << 20

#: Headroom on the per-shard sample cap. The parent keeps the globally smallest
#: ``target`` keys, so a shard only has to hold enough of its own smallest keys
#: to be sure it is not hiding one of them. Three times its expected share is
#: about forty standard deviations clear at the sizes involved.
SAMPLE_HEADROOM = 3

#: Characters of each sampled record kept in a shard's sample.
#:
#: This number is the single largest term in a scan's peak memory, and it used
#: to be 4000 for no reason anyone could name. Both consumers want less: the
#: atlas embedder truncates its input to 2000 characters before encoding, and
#: the token-budget estimator wants a tokens-per-character ratio, which a
#: two-thousand-character sample estimates as precisely as a four-thousand one.
#: So half of every sampled string was carried across a process boundary, held
#: in the parent, and thrown away.
SAMPLE_TEXT_CHARS = 2000

#: Bytes assumed per sampled record when sizing a cap against a memory budget.
#: A Python ``str`` of ``SAMPLE_TEXT_CHARS`` costs its length in bytes for
#: Latin-1 text and up to four times that for text the interpreter has to store
#: as UCS-4; the tuple, the heap slot and the pickle buffer add the rest. Set at
#: 2.5x the character count, which is above the measured cost of a Turkish or
#: Arabic corpus and well above an English one.
BYTES_PER_SAMPLE = int(SAMPLE_TEXT_CHARS * 2.5)

_M64 = (1 << 64) - 1


def sample_key(salt: int, index: int) -> int:
    """A uniform 64-bit key for one record, from where it sits and nothing else.

    Position rather than content, so that a record repeated ten thousand times
    does not put ten thousand copies of itself in the sample or none at all.
    """
    x = (salt ^ ((index + 1) * 0x9E3779B97F4A7C15)) & _M64
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _M64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _M64
    return x ^ (x >> 31)


def file_salt(path: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(path.encode("utf-8", "surrogatepass"), digest_size=8).digest(),
        "big",
    )


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One file, or one piece of one file, tagged with how to read it."""

    dataset: str
    layout: str
    path: str
    suffix: str
    compressed: bool
    span: ReadSpan
    salt: int
    approx_bytes: int


#: Shards per worker. More than one so a shard that turns out to be slow — a
#: file of long records, or one the page cache has not got — does not leave a
#: core idle while the rest wait, and so the progress bar moves more than once
#: per core.
SHARDS_PER_WORKER = 4


@dataclass
class ScanPlan:
    """How the corpus was divided, and what the division implies."""

    shards: list[list[WorkItem]] = field(default_factory=list)
    workers: int = 1
    total_bytes: int = 0
    #: Per-shard cap on the atlas sample. Corpus-wide within the shard, *not*
    #: per dataset — see :func:`run_shard`.
    atlas_cap: int = 20_000
    #: Per-shard cap on the token-budget sample, one entry per dataset. Sized
    #: individually because a dataset that lives in two shards needs each of
    #: them to keep more of it than a dataset spread across fifty.
    budget_caps: dict[str, int] = field(default_factory=dict)
    #: Cap for a dataset the planner did not see, which is every dataset on the
    #: serial path where there is one shard and it holds everything.
    budget_cap: int = 2_000
    #: Target atlas sample size once shards are merged.
    sample_target: int = 20_000
    #: Rough record count, for the progress bar. Never used in a measurement.
    estimated_records: int = 0
    #: Bytes the whole scan was allowed to spend on samples, and what the caps
    #: above came out at. Reported so a scan that sampled less than it wanted
    #: says why rather than being quietly less representative.
    memory_budget: int = 0
    sample_bound_by: str = "target"

    @property
    def parallel(self) -> bool:
        return len(self.shards) > 1 and self.workers > 1


def _suffix_of(path: str) -> tuple[str, bool]:
    from .discovery import effective_suffix

    return effective_suffix(path)


def worker_count(requested: int | None = None) -> int:
    """How many processes to use, honouring an explicit request and the machine.

    Always one inside a worker. Without that guard, a caller who runs a scan
    from an unguarded script under the ``spawn`` start method gets a process
    that re-imports the script, starts another scan, and repeats.

    Everything else is :func:`dropoutt.hardware.plan`, which reads CPU affinity,
    any cgroup quota, hyperthread topology and free memory. ``os.cpu_count()``
    was doing none of that: in a container limited to two cores on a 96-core
    host it reported 96, and the scan forked twelve workers into a quota that
    could run two of them.
    """
    import multiprocessing

    from .hardware import plan

    if multiprocessing.current_process().name != "MainProcess":
        return 1
    return plan(requested).workers


def pool_context(safe_to_fork: bool = True):
    """The start method to use.

    ``fork`` by default, because a forked worker inherits the benchmark index,
    the language model and every compiled pattern already in memory —
    copy-on-write, so twelve workers cost one copy rather than twelve, and none
    of them spends a fifth of a second rebuilding it.

    Not when a tokenizer is in play. ``tokenizers`` runs batches through rayon,
    whose thread pool does not survive a fork: the child inherits a pool whose
    threads do not exist and blocks the first time it uses one. A scan with
    ``--model`` therefore spawns, and the workers load the tokenizer themselves.
    """
    import multiprocessing

    available = multiprocessing.get_all_start_methods()
    order = ("fork", "spawn") if safe_to_fork else ("spawn", "fork")
    for method in order:
        if method in available:
            return multiprocessing.get_context(method)
    return multiprocessing.get_context()  # pragma: no cover


def plan_scan(
    discovery,
    layouts: dict[str, str],
    *,
    workers: int | None,
    limit_per_file: int | None,
    atlas_target: int,
    budget_target: int,
    datasets: int = 1,
    mean_record_bytes: float | None = None,
    memory_budget: int | None = None,
) -> ScanPlan:
    """Divide the corpus into contiguous shards, and size what each may keep."""
    wanted = worker_count(workers)
    items: list[WorkItem] = []
    total = 0
    for ds in discovery.datasets:
        layout = layouts.get(ds.name)
        if layout is None:
            continue
        for path in ds.files:
            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 1
            total += size
            suffix, compressed = _suffix_of(path)
            items.append(
                WorkItem(ds.name, layout, path, suffix, compressed, WHOLE_FILE,
                         file_salt(path), size)
            )

    plan = ScanPlan(total_bytes=total, sample_target=atlas_target, workers=wanted)
    plan.estimated_records = _estimate_records(total, mean_record_bytes)

    if wanted < 2 or total < MIN_BYTES_FOR_PARALLEL or not items:
        plan.shards = [items]
        plan.workers = 1
    else:
        pieces = wanted * SHARDS_PER_WORKER
        # A record limit is a debugging flag that counts per file. Splitting a
        # file would apply it per piece and silently read several times as much.
        if limit_per_file is None and len(items) < pieces:
            items = _split_files(items, pieces)
        plan.shards = _group(items, pieces)

    _size_samples(
        plan, atlas_target=atlas_target, budget_target=budget_target,
        datasets=max(1, datasets), memory_budget=memory_budget,
    )
    return plan


def _size_samples(
    plan: ScanPlan,
    *,
    atlas_target: int,
    budget_target: int,
    datasets: int,
    memory_budget: int | None,
) -> None:
    """Decide how many records each shard may keep, and prove it fits in memory.

    This is where a scan used to spend gigabytes. The old rule was one cap,
    applied *per dataset* inside every shard, derived from the corpus-wide atlas
    target of 200,000. On a forty-eight-shard scan that is 12,500 records per
    dataset per shard — so a shard covering ten datasets held 125,000 sampled
    strings, every worker held its own, and the parent then held all
    forty-eight results at once before merging any of them. A corpus with a few
    hundred datasets in it, which is exactly what a folder of mixed exports is,
    turned into tens of gigabytes of resident sample.

    Two things fix it. The atlas cap is now shard-wide rather than per dataset,
    because the parent's atlas heap is corpus-wide and never wanted a per-dataset
    split; the number of datasets a shard happens to span stops being a
    multiplier. And both caps are then checked against a real memory budget
    derived from this machine, so the answer on a 6 GB laptop is a smaller
    sample rather than the OOM reaper.

    The budget sample stays per dataset because it *is* stratified: each
    dataset's tokens-per-character is applied to that dataset's own character
    count. It is also two orders of magnitude smaller.
    """
    from .hardware import plan as hardware_plan

    shards = max(1, len(plan.shards))
    budget = memory_budget if memory_budget is not None else hardware_plan().memory_budget
    plan.memory_budget = budget

    # What the statistics want, before memory has an opinion. A shard keeps
    # SAMPLE_HEADROOM times its expected share of the global bottom-k, which is
    # what makes the merged sample identical however the corpus was divided.
    #
    # The atlas heap is corpus-wide, so every shard sees an equal expected share
    # of it and one number covers them all.
    want_atlas = max(256, -(-atlas_target * SAMPLE_HEADROOM // shards))

    # The budget heaps are per dataset, and a dataset is *not* spread over every
    # shard. Dividing its target by the total shard count is what made a
    # parallel scan's token estimate differ from a serial one by 0.04%: eight
    # datasets over fifty-two shards meant each dataset lived in about seven of
    # them, each shard was allowed a fifty-second of the target, and the merged
    # bottom-k was missing keys that a serial pass would have kept. The divisor
    # is how many shards actually hold that dataset, and the cap never exceeds
    # the target itself, because a dataset confined to one shard needs that
    # shard to keep all of it and no more.
    per_dataset = max(200, budget_target // max(1, datasets))
    spread: dict[str, int] = {}
    for shard in plan.shards:
        for name in {item.dataset for item in shard}:
            spread[name] = spread.get(name, 0) + 1
    plan.budget_caps = {
        name: min(per_dataset, max(64, -(-per_dataset * SAMPLE_HEADROOM // count)))
        for name, count in spread.items()
    }
    want_budget = max(plan.budget_caps.values(), default=per_dataset)

    # What memory allows. Live at once: every shard's atlas heap (workers run
    # concurrently and the parent holds one result at a time), the per-dataset
    # budget heaps inside them, and the parent's own merged heap of
    # `atlas_target`. Sized against the widest case rather than the average.
    live_shards = min(shards, max(1, plan.workers) + 1)
    parent_cost = atlas_target * BYTES_PER_SAMPLE
    per_shard_records = max(1, (budget - parent_cost) // (BYTES_PER_SAMPLE * live_shards))

    if per_shard_records >= want_atlas + want_budget * datasets:
        plan.atlas_cap, plan.budget_cap = want_atlas, want_budget
        plan.sample_bound_by = "target"
        return

    # Squeezed. The budget sample is protected first: it is the smaller of the
    # two by a wide margin, and losing it costs a token estimate that the report
    # leads with, while a smaller atlas sample costs resolution on a histogram.
    budget_cap = max(32, min(want_budget, per_shard_records // (2 * datasets)))
    atlas_cap = max(64, per_shard_records - budget_cap * datasets)
    plan.atlas_cap, plan.budget_cap = atlas_cap, budget_cap
    plan.budget_caps = {
        name: min(cap, budget_cap) for name, cap in plan.budget_caps.items()
    }
    plan.sample_bound_by = "memory"


#: Bytes per record when nothing better is known. Only ever moves a progress
#: bar; no measurement is derived from it.
FALLBACK_RECORD_BYTES = 900.0


def _estimate_records(total_bytes: int, mean_record_bytes: float | None) -> int:
    """A record count good enough for a progress bar, from bytes on disk.

    The mean comes from the records already read during schema induction, so on
    a line-delimited corpus the bar is usually within a few percent. Columnar
    formats give induction no byte count, and fall back to a flat average that
    is wrong by a factor rather than an order of magnitude.
    """
    per = mean_record_bytes if mean_record_bytes and mean_record_bytes > 1 else None
    return max(1, int(total_bytes / (per or FALLBACK_RECORD_BYTES)))


def _split_files(items: list[WorkItem], wanted: int) -> list[WorkItem]:
    """Cut the largest files up until there are enough pieces to go round."""
    parts_per_file = -(-wanted // max(len(items), 1))
    out: list[WorkItem] = []
    for item in items:
        spans = plan_splits(
            item.path, item.suffix, parts_per_file, compressed=item.compressed
        )
        if len(spans) < 2:
            out.append(item)
            continue
        share = max(1, item.approx_bytes // len(spans))
        for span in spans:
            out.append(
                WorkItem(item.dataset, item.layout, item.path, item.suffix,
                         item.compressed, span, item.salt, share)
            )
    return out


def _group(items: list[WorkItem], shards: int) -> list[list[WorkItem]]:
    """Group work items into contiguous shards of roughly equal size.

    Contiguity is the requirement, not balance: merging happens in shard order,
    and every "keep the first N examples" rule in the check catalog depends on
    that order being the order a serial read would have produced.
    """
    if shards <= 1 or len(items) <= 1:
        return [list(items)]
    total = sum(i.approx_bytes for i in items) or 1
    per_shard = total / shards
    out: list[list[WorkItem]] = []
    current: list[WorkItem] = []
    running = 0
    for item in items:
        current.append(item)
        running += item.approx_bytes
        if len(out) < shards - 1 and running >= per_shard * (len(out) + 1):
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------


@dataclass
class ShardConfig:
    """Everything a worker needs to reconstruct the scan for its slice.

    Deliberately small and made of plain data: it crosses a process boundary, so
    a loaded tokenizer or a merged benchmark index would be pickled with it.
    Workers rebuild those from disk, which they do concurrently.
    """

    root: str
    profile: Profile
    target: str | None
    seq_len: int | None
    model_id: str | None
    check_ids: list[str]
    minhash_preset: str
    limit_per_file: int | None
    #: Shard-wide cap on the atlas sample, and per-dataset cap on the budget
    #: sample. See :func:`_size_samples` for why they differ in that.
    atlas_cap: int
    budget_caps: dict[str, int]
    budget_cap: int
    want_atlas_sample: bool
    want_language: bool
    contamination_dirs: list[str]
    eval_sets: list[str]
    model_for_tokenizer: str | None
    offline: bool
    offsets_unreliable: bool
    eos_token_id: int | None


@dataclass
class ShardResult:
    """What one shard produces."""

    index: int = 0
    checks: dict[str, Any] = field(default_factory=dict)
    minhash: Any = None
    contamination: Any = None
    scanned: int = 0
    total_chars: int = 0
    total_words: int = 0
    content_total: int = 0
    #: (sample key, text, language, characters, dataset), shard-wide bottom-k.
    #: The parent merges these into one corpus-wide heap.
    atlas_samples: list[tuple[int, str, str, int, str]] = field(default_factory=list)
    #: dataset -> list of (sample key, text, characters), bottom-k per dataset.
    #: No language: the budget estimator prices characters, not languages, and
    #: carrying the code doubled the tuple for a field nothing read.
    budget_samples: dict[str, list[tuple[int, str, int]]] = field(default_factory=dict)
    record_counts: dict[str, int] = field(default_factory=dict)
    #: dataset -> characters of normalised text. The token budget extrapolates
    #: each dataset's own tokens-per-character against its own character count,
    #: so it needs the split rather than the corpus total.
    chars_by_dataset: dict[str, int] = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)


#: Built once per worker process and reused across the shards it handles.
_WORKER_STATE: dict[str, Any] = {}


def prime_worker(ctx) -> None:
    """Let the calling process reuse its own context for the serial path."""
    _WORKER_STATE["ctx"] = ctx


def _worker_context(config: ShardConfig):
    from .contamination import load_indices
    from .context import ScanContext
    from .langid import LanguageDetector

    cached = _WORKER_STATE.get("ctx")
    if cached is not None:
        return cached

    contamination = None
    if config.contamination_dirs:
        contamination = load_indices(*[Path(p) for p in config.contamination_dirs])
        if config.eval_sets:
            wanted = set(config.eval_sets)
            contamination.benchmarks = {
                name: index for name, index in contamination.benchmarks.items()
                if name in wanted
            }
        if contamination.is_empty:
            contamination = None

    tokenizer = chat_template = None
    if config.model_for_tokenizer:
        from .config import resolve_model

        resolved = resolve_model(config.model_for_tokenizer, offline=config.offline)
        tokenizer = resolved.tokenizer
        chat_template = resolved.chat_template

    ctx = ScanContext(
        root=config.root,
        profile=config.profile,
        target=config.target,
        seq_len=config.seq_len,
        model_id=config.model_id,
        tokenizer=tokenizer,
        chat_template=chat_template,
        detector=LanguageDetector() if config.want_language else None,
        contamination=contamination,
        atlas=None,
    )
    ctx.stats["minhash_preset"] = config.minhash_preset
    if config.offsets_unreliable:
        ctx.stats["offsets_unreliable"] = True
    if config.eos_token_id is not None:
        ctx.stats["eos_token_id"] = config.eos_token_id
    _WORKER_STATE["ctx"] = ctx
    return ctx


def iter_span_records(item: WorkItem, limit: int | None) -> Iterator[RawRecord]:
    return read_file(
        item.path, item.suffix, compressed=item.compressed, limit=limit, span=item.span
    )


def run_shard(
    payload: tuple[ShardConfig, int, list[WorkItem]],
    progress: Callable[[int], None] | None = None,
) -> ShardResult:
    """Scan one shard. Runs in a worker process, or in the caller for one shard."""
    import heapq

    from . import checks as _checks_pkg  # noqa: F401  (registers the catalog)
    from .checks.base import REGISTRY
    from .normalize import to_document
    from .runner import _compute_features, record_digest

    config, index, items = payload
    ctx = _worker_context(config)
    # A forked worker inherits whatever the parent had already recorded. Only
    # what this shard adds is sent home, or every shard would report the
    # parent's notes back to it.
    degraded_from = len(ctx.degradations)
    # Fresh check instances per shard: they accumulate, and a worker that
    # handled two shards would otherwise count the first one twice.
    active = [
        cls() for cls in (REGISTRY.get(cid) for cid in config.check_ids) if cls is not None
    ]

    result = ShardResult(index=index)
    ctx.stats.pop("_minhash_store", None)
    if ctx.contamination is not None:
        ctx.contamination.reset()

    # Bottom-k samples, held as max-heaps on the negated key so the worst kept
    # sample is the one that gets evicted. Two of them, for two different
    # questions: the atlas wants a corpus-wide sample and the token budget wants
    # a stratified one, and keeping a single per-dataset heap sized for the
    # larger of the two was what made a scan's memory scale with dataset count.
    #
    # A record that lands in both heaps is stored once. `text` is bound to one
    # string object and both tuples reference it, so the second heap costs a
    # pointer rather than a copy.
    from .context import F_LANG

    atlas_heap: list[tuple[int, str, str, int, str]] = []
    budget_heaps: dict[str, list[tuple[int, str, int]]] = {}
    atlas_cap = config.atlas_cap if config.want_atlas_sample else 0
    budget_caps = config.budget_caps
    default_budget_cap = config.budget_cap

    for item in items:
        salt = item.salt
        for rec in iter_span_records(item, config.limit_per_file):
            if rec.truncated:
                ctx.degraded(rec.truncated)
            doc = to_document(rec, item.layout, item.dataset, index=rec.source_index)
            result.content_total += record_digest(doc)
            _compute_features(doc, ctx)
            text = doc.text
            result.total_chars += len(text)
            result.total_words += text.count(" ") + 1 if text else 0
            result.chars_by_dataset[item.dataset] = (
                result.chars_by_dataset.get(item.dataset, 0) + len(text)
            )

            if len(text) > 20:
                key = -sample_key(salt, rec.source_index)
                kept = text[:SAMPLE_TEXT_CHARS]
                length = len(text)
                if atlas_cap:
                    entry = (
                        key, kept, doc.meta.get(F_LANG) or "unknown", length, item.dataset
                    )
                    if len(atlas_heap) < atlas_cap:
                        heapq.heappush(atlas_heap, entry)
                    elif key > atlas_heap[0][0]:
                        heapq.heapreplace(atlas_heap, entry)
                budget = budget_heaps.setdefault(item.dataset, [])
                cap = budget_caps.get(item.dataset, default_budget_cap)
                row = (key, kept, length)
                if len(budget) < cap:
                    heapq.heappush(budget, row)
                elif key > budget[0][0]:
                    heapq.heapreplace(budget, row)

            for check in active:
                try:
                    check.observe(doc, ctx)
                except Exception as exc:
                    ctx.degraded(f"check {check.check_id} errored: {type(exc).__name__}")
            result.scanned += 1
            result.record_counts[item.dataset] = (
                result.record_counts.get(item.dataset, 0) + 1
            )
            if progress is not None and result.scanned % 2000 == 0:
                progress(result.scanned)

    result.atlas_samples = [
        (-k, text, lang, chars, dataset)
        for k, text, lang, chars, dataset in sorted(atlas_heap, reverse=True)
    ]
    result.budget_samples = {
        name: [(-k, text, chars) for k, text, chars in sorted(heap, reverse=True)]
        for name, heap in budget_heaps.items()
    }
    result.checks = {check.check_id: check for check in active}
    result.minhash = ctx.stats.pop("_minhash_store", None)
    if ctx.contamination is not None:
        result.contamination = ctx.contamination.take_accumulator()
    result.degradations = ctx.degradations[degraded_from:]
    del ctx.degradations[degraded_from:]
    return result


def _run_shard_entry(payload):  # pragma: no cover - runs in a worker process
    return run_shard(payload)


def run_shards(
    config: ShardConfig,
    plan: ScanPlan,
    *,
    ctx,
    progress: Callable[[int, int], None] | None = None,
    phase: Callable[[str], None] | None = None,
    consume: Callable[[ShardResult], None] | None = None,
) -> list[ShardResult]:
    """Run every shard, in this process or across a pool.

    ``consume`` is called with each result **in shard order**, as soon as every
    shard before it has been consumed. Pass it and the return value is empty:
    the caller has already folded each result and this function drops its
    reference, so the parent holds one shard's samples rather than all of them.
    That was worth several gigabytes on a large corpus — the results used to
    accumulate in a dict until the last worker finished, and only then get
    merged.

    Order is not an optimisation detail here. Shards are contiguous slices in
    corpus order and every "keep the first N examples" rule in the check catalog
    depends on them being folded in that order, so a result that finishes early
    waits in the reorder buffer until its predecessors have been handed over.
    """
    ordered: list[ShardResult] = []
    delivered = 0

    def deliver(result: ShardResult) -> None:
        nonlocal delivered
        delivered += 1
        if consume is None:
            ordered.append(result)
        else:
            consume(result)

    if not plan.parallel:
        prime_worker(ctx)
        only = plan.shards[0] if plan.shards else []
        cb = None
        if progress is not None:
            cb = lambda done: progress(done, plan.estimated_records)  # noqa: E731
        deliver(run_shard((config, 0, only), cb))
        return ordered

    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Forked workers inherit this, so under ``fork`` they start with the
    # benchmark index and language model already in memory.
    prime_worker(ctx)
    payloads = [(config, i, shard) for i, shard in enumerate(plan.shards)]
    pending: dict[int, ShardResult] = {}
    next_index = 0
    done_records = 0
    if progress is not None:
        progress(0, plan.estimated_records)
    from .hardware import single_threaded_children

    try:
        # The limit is applied around pool *creation*: these libraries read the
        # environment when they load, so a child that sets it in its own entry
        # point is already too late. See `single_threaded_children`.
        with single_threaded_children(), ProcessPoolExecutor(
            max_workers=min(plan.workers, len(plan.shards)),
            mp_context=pool_context(config.model_for_tokenizer is None),
        ) as pool:
            futures = {pool.submit(_run_shard_entry, p): p[1] for p in payloads}
            for future in as_completed(futures):
                shard_result = future.result()
                pending[shard_result.index] = shard_result
                done_records += shard_result.scanned
                if progress is not None:
                    progress(done_records, plan.estimated_records)
                while next_index in pending:
                    deliver(pending.pop(next_index))
                    next_index += 1
    except Exception as exc:
        # A pool that cannot start is not a reason to fail a scan. Fall back to
        # doing the same work here, which produces the same answer more slowly.
        ctx.degraded(
            f"parallel scan unavailable ({type(exc).__name__}), ran on one core"
        )
        prime_worker(ctx)
        flat = [item for shard in plan.shards for item in shard]
        cb = None
        if progress is not None:
            cb = lambda done: progress(done, plan.estimated_records)  # noqa: E731
        ordered.clear()
        result = run_shard((config, 0, flat), cb)
        if consume is not None and delivered:
            # Shards were already folded into live check objects before the pool
            # died, and a check cannot be un-folded. The caller is told to throw
            # its state away and start from this one serial result rather than
            # being handed a second, overlapping set on top of a partial one.
            raise ShardRestart(result) from None
        if consume is not None:
            consume(result)
            return ordered
        return [result]
    for index in sorted(pending):
        deliver(pending[index])
    return ordered


class ShardRestart(Exception):
    """The pool failed mid-run; the enclosed serial result replaces everything.

    Raised only when ``run_shards`` was streaming results to a consumer, because
    that consumer has already folded some of them into live check objects and
    those cannot be un-folded. The caller catches this, rebuilds its checks and
    folds the single serial result instead.
    """

    def __init__(self, result: ShardResult) -> None:
        super().__init__("parallel scan failed after partial merge")
        self.result = result
