"""Command line interface.

Exit codes are three-valued on purpose. A checker that returns the same code for
"found problems" and "crashed" cannot be used in CI.

    0   completed, whether or not findings were reported
    1   internal error
    2   usage error
    10  blocking findings, and only when a target profile was declared
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .compat import (
    HAVE_MODEL2VEC,
    HAVE_TOKENIZERS,
    capability_report,
    json_dumps,
    json_loads,
)
from .config import CONFIG_NAME, Config, cache_dir, parse_profile, resolve_model
from .models import Profile
from .registry_data import resolve_model_alias

app = typer.Typer(
    name="dropoutt",
    help="Pre-flight checks and comparable fingerprints for LLM training datasets.",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 10


class _ProgressDisplay:
    """Spinner in a terminal, stable phase lines in redirected output."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._status = None
        self._last_phase = ""

    def __enter__(self) -> "_ProgressDisplay":
        if self.enabled and console.is_terminal:
            self._status = console.status("", spinner="dots")
            self._status.start()
        return self

    def phase(self, label: str) -> None:
        if not self.enabled or label == self._last_phase:
            return
        self._last_phase = label
        if self._status is not None:
            self._status.update(f"[cyan]working[/cyan] {escape(label)}")
        else:
            console.print(f"  [dim]{escape(label)}...[/dim]")

    def records(self, dataset: str, count: int) -> None:
        if not self.enabled:
            return
        label = f"Scanning {dataset} · {count:,} records"
        if self._status is not None:
            self._status.update(f"[cyan]working[/cyan] {escape(label)}")
        elif count == 2_000 or count % 20_000 == 0:
            console.print(f"  [dim]{escape(label)}[/dim]")

    def finish(self, label: str | None = None) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self.enabled and label:
            console.print(f"  [green]done[/green] {escape(label)}")

    def __exit__(self, *_args) -> None:
        self.finish()


@app.command(
    no_args_is_help=True,
    epilog="Example: dropoutt scan ./data --offline",
)
def scan(
    path: Path = typer.Argument(..., help="File or directory to scan."),
    model: Optional[str] = typer.Option(None, "--model", "-m",
                                        help="Target model id or local path. Unlocks token checks."),
    profile: str = typer.Option("auto", "--profile", "-p",
                                help="sft, corpus, preference, or auto."),
    target: Optional[str] = typer.Option(None, "--target",
                                         help="Declare what you are building. Enables blocking."),
    seq_len: Optional[int] = typer.Option(
        None, "--seq-len", min=1, help="Training sequence length."
    ),
    tier: Optional[int] = typer.Option(
        None, "--tier", min=0, help="Highest check tier to run. Defaults to config, then 1."
    ),
    out: Optional[Path] = typer.Option(None, "--out", "-o",
                                       help="Directory for findings, fingerprint and report."),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    limit: Optional[int] = typer.Option(
        None, "--limit", min=1, help="Max records per file."
    ),
    no_html: bool = typer.Option(False, "--no-html", help="Skip the HTML report."),
    no_atlas: bool = typer.Option(False, "--no-atlas", help="Skip atlas coverage."),
    no_evidence: bool = typer.Option(
        False,
        "--no-evidence",
        help="Omit record excerpts and source locations from terminal and output reports.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q",
                               help="Suppress the report; write output files and exit."),
) -> None:
    """Scan a dataset directory."""
    if not path.exists():
        console.print(f"[red]No such path:[/red] {escape(str(path))}")
        raise typer.Exit(EXIT_USAGE)

    from .atlas import load_bundled
    from .contamination import load_indices
    from .discovery import discover
    from .fingerprint import build as build_fingerprint
    from .langid import LanguageDetector
    from .report import terminal as term_report
    from .runner import scan as run_scan

    with _ProgressDisplay(enabled=not quiet) as activity:
        activity.phase("Discovering supported data files")
        preflight = discover(str(path))
        if not preflight.files or not any(file.readable for file in preflight.files):
            activity.finish()
            console.print(f"[red]No supported data files found in:[/red] {escape(str(path))}")
            console.print(
                "  [dim]Supported inputs: JSON, JSONL/NDJSON, Parquet, Arrow/Feather, "
                "ORC, CSV/TSV, TXT, and Markdown. Text formats may be compressed "
                "with gzip, bzip2, xz, or zstd.[/dim]"
            )
            raise typer.Exit(EXIT_USAGE)

        activity.phase("Reading configuration")
        try:
            cfg = Config.load(path)
        except ValueError as exc:
            activity.finish()
            console.print(f"[red]Invalid dropoutt.toml:[/red] {escape(str(exc))}")
            raise typer.Exit(EXIT_USAGE) from None
        model = model or cfg.model
        target = target or cfg.target
        seq_len = seq_len or cfg.seq_len
        tier = cfg.tier if tier is None else tier
        offline = offline or cfg.offline or _offline_from_environment()
        if model is not None and not isinstance(model, str):
            _invalid_config("model must be a string")
        if not isinstance(tier, int) or isinstance(tier, bool) or tier < 0:
            _invalid_config("tier must be a non-negative integer")
        if seq_len is not None and (
            not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1
        ):
            _invalid_config("seq_len must be a positive integer")
        from .minhash import PRESETS

        if cfg.minhash_preset not in PRESETS:
            _invalid_config(
                f"minhash_preset must be one of: {', '.join(sorted(PRESETS))}"
            )
        if profile == "auto":
            profile = cfg.profile
        _validate_profile(profile, option="profile", allow_auto=True)
        if target is not None:
            _validate_profile(target, option="target", allow_auto=False)

        tokenizer = chat_template = None
        model_notes: list[str] = []
        if model:
            activity.phase(f"Resolving model {model}")
            resolved = resolve_model(model, offline=offline)
            tokenizer = resolved.tokenizer
            chat_template = resolved.chat_template
            seq_len = seq_len or resolved.seq_len
            model = resolved.model_id
            model_notes = list(resolved.notes)

        activity.phase("Loading local checks, benchmark indexes, and atlas")
        detector = LanguageDetector()
        contamination = load_indices(*_contamination_dirs())
        if cfg.eval_sets:
            requested = set(cfg.eval_sets)
            missing = sorted(requested - set(contamination.benchmarks))
            if missing:
                activity.finish()
                console.print(
                    f"[red]Unknown eval_sets in dropoutt.toml:[/red] {', '.join(missing)}"
                )
                console.print(
                    "  [dim]Run `dropoutt benchmarks` and check the names created by "
                    "`dropoutt index-eval`.[/dim]"
                )
                raise typer.Exit(EXIT_USAGE)
            contamination.benchmarks = {
                name: index
                for name, index in contamination.benchmarks.items()
                if name in requested
            }
        atlas_obj = None if no_atlas else load_bundled()

        result = run_scan(
            str(path),
            profile=parse_profile(profile) if profile != "auto" else Profile.UNKNOWN,
            target=target,
            model_id=model,
            seq_len=seq_len,
            tokenizer=tokenizer,
            chat_template=chat_template,
            detector=detector,
            contamination=contamination if not contamination.is_empty else None,
            atlas=atlas_obj,
            max_tier=tier,
            minhash_preset=cfg.minhash_preset,
            muted=tuple(cfg.mute),
            limit_per_file=limit,
            progress=activity.records,
            phase=activity.phase,
            offline=offline,
        )

        # Count the text that would actually be trained on, not the bytes on disk.
        # `total_bytes` includes JSON syntax, keys and metadata columns, and in UTF-8
        # a Turkish or Arabic corpus costs more bytes per character than an English
        # one. Putting that number in a fingerprint would make the one artifact meant
        # to be comparable across datasets vary with language and file format. The
        # runner already accumulates real character and word counts over the
        # normalised text; use those, and fall back to bytes only if it ran no
        # records at all.
        activity.phase("Estimating token budget")
        scanned_chars = int(result.ctx.stats.get("total_chars", 0))
        total_chars = scanned_chars or sum(d.total_bytes for d in result.ctx.datasets)
        total_words = int(result.ctx.stats.get("total_words", 0))
        budget = _budget(result, total_chars, offline=offline)
        fp = build_fingerprint(
            result.ctx, result.findings,
            total_chars=total_chars, total_words=total_words,
            budget=budget,
            config_hash=Config(
                model=model,
                profile=result.ctx.profile.value,
                target=target,
                seq_len=seq_len,
                tier=tier,
                minhash_preset=cfg.minhash_preset,
                mute=list(cfg.mute),
                eval_sets=list(cfg.eval_sets),
            ).hash(),
        )

        activity.phase("Writing scan artifacts")
        out_dir = out or (path if path.is_dir() else path.parent) / ".dropoutt"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(
            out_dir,
            result,
            fp,
            budget,
            write_html=not no_html,
            include_evidence=not no_evidence,
        )
        activity.finish(f"Scanned {result.records_scanned:,} records")

    for note in model_notes:
        console.print(f"  [yellow]note[/yellow] {escape(note)}")
    if not quiet:
        term_report.render(console, result, budget=budget, show_evidence=not no_evidence)

    console.print(f"  [dim]wrote {escape(str(out_dir))}/fingerprint.json, findings.jsonl"
                  f"{', report.html' if not no_html else ''}[/dim]")
    if no_evidence:
        console.print("  [dim]record excerpts and source locations were omitted[/dim]")
    else:
        artifacts = "findings.jsonl" + (", and report.html" if not no_html else "")
        console.print(
            f"  [yellow]note[/yellow] {artifacts} may contain dataset excerpts and "
            "source paths; use --no-evidence before exporting them"
        )

    if result.ctx.blocking_enabled and result.blocking:
        console.print(f"\n  [bold red]{len(result.blocking)} blocking finding(s)[/bold red] "
                      f"under target profile {result.ctx.profile.value}")
        raise typer.Exit(EXIT_BLOCKED)
    raise typer.Exit(EXIT_OK)


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Directory to configure."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    offline: bool = typer.Option(
        False, "--offline", help="Resolve the model only from local files and caches."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Infer configuration from your data and write dropoutt.toml."""
    from .discovery import discover
    from .readers import read_file
    from .runner import _infer_profile
    from .schema_induction import induce

    if not path.exists():
        console.print(f"[red]No such path:[/red] {escape(str(path))}")
        console.print("  [dim]Pass an existing dataset file or directory.[/dim]")
        raise typer.Exit(EXIT_USAGE)

    target_file = (path if path.is_dir() else path.parent) / CONFIG_NAME
    if target_file.exists() and not force:
        console.print(
            f"[yellow]{escape(str(target_file))} already exists.[/yellow] "
            "Use --force to overwrite."
        )
        raise typer.Exit(EXIT_USAGE)

    resolved = None
    with _ProgressDisplay() as activity:
        activity.phase("Discovering supported data files")
        disc = discover(str(path))
        verdicts = {}
        activity.phase("Inferring dataset layouts")
        for ds in disc.datasets[:50]:
            sample = []
            for f in ds.files[:2]:
                sample.extend(list(read_file(f, Path(f).suffix, limit=300)))
            if sample:
                verdicts[ds.name] = induce(sample)

        profile = _infer_profile(verdicts)
        notes = {"profile": f"inferred from {len(verdicts)} dataset(s)"}
        offline = offline or _offline_from_environment()
        cfg = Config(model=model, profile=profile.value, offline=offline)

        if model:
            activity.phase(f"Resolving model {model}")
            resolved = resolve_model(model, offline=offline)
            cfg.model = resolved.model_id
            cfg.seq_len = resolved.seq_len
            notes["model"] = (
                "resolved from local files/cache" if offline else "resolved from the Hub"
            )
            if resolved.seq_len:
                notes["seq_len"] = "from the model config"
        activity.finish("Configuration inferred")

    if resolved is not None:
        for note in resolved.notes:
            console.print(f"  [yellow]note[/yellow] {escape(note)}")
        _render_confirmation(resolved)

    target_file.write_text(cfg.to_toml(inferred_notes=notes), encoding="utf-8")
    console.print(f"\n  Wrote [bold]{escape(str(target_file))}[/bold]")
    console.print(f"  Detected profile: [bold]{profile.value}[/bold] "
                  f"from {len(verdicts)} dataset(s)")
    console.print("  [dim]Edit the file to declare a target if you want findings to block.[/dim]")


@app.command(
    no_args_is_help=True,
    epilog=(
        "Example: dropoutt diff ./candidate/.dropoutt/fingerprint.json "
        "./mixture/.dropoutt/fingerprint.json"
    ),
)
def diff(
    left: Path = typer.Argument(..., help="Fingerprint of the dataset you are considering."),
    right: Path = typer.Argument(..., help="Fingerprint of what you already have."),
    full: bool = typer.Option(False, "--full", help="List every differing region."),
) -> None:
    """Compare two fingerprints against the shared atlas.

    Directional, and read left-against-right: "what does LEFT cover that RIGHT
    does not". That is the question worth asking before adding a dataset to a
    mixture, and it is not symmetric — a small specialised corpus can be wholly
    inside a large one while the large one is barely inside it.

    Both arguments are fingerprint.json files written by `dropoutt scan`.
    """
    from .atlas.compare import compare

    fps = [_read_fingerprint(left, "left"), _read_fingerprint(right, "right")]

    a, b = fps
    console.print()
    console.rule("[bold]dropoutt diff[/bold]", style="dim")
    console.print(f"  [dim]left [/dim] {escape(str(a.get('root', left)))}")
    console.print(f"  [dim]right[/dim] {escape(str(b.get('root', right)))}")

    if a.get("pipeline_version") != b.get("pipeline_version"):
        console.print(f"\n  [yellow]note[/yellow] different pipeline versions "
                      f"({escape(str(a.get('pipeline_version')))} against "
                      f"{escape(str(b.get('pipeline_version')))}); measurements may "
                      f"not mean the same thing")

    _diff_shape(a, b)

    cov_a = (a.get("facets", {}).get("coverage") or {}).get("values")
    cov_b = (b.get("facets", {}).get("coverage") or {}).get("values")
    result = compare(cov_a, cov_b)

    console.print("\n  [bold]Atlas comparison[/bold]")
    if not result.comparable:
        console.print(f"    [yellow]not comparable[/yellow] [dim]{escape(result.reason)}[/dim]")
        console.print("    [dim]Coverage is withheld rather than estimated when records "
                      "did not land on the atlas.[/dim]")
        raise typer.Exit(EXIT_OK)

    console.print(f"    Similarity   {result.similarity:.2f}  "
                  f"[dim](1.0 = same distribution over regions)[/dim]")
    console.print(f"    Shared       {result.shared_mass:.0%} of left sits in regions "
                  f"right also occupies")
    console.print(f"    New          [bold]{result.added_mass:.0%}[/bold] of left sits "
                  f"in regions right never reaches")

    if result.a_only:
        console.print("\n    [bold]Only in left[/bold] [dim]— what adding it would "
                      "bring[/dim]")
        for r, share, terms in (result.a_only if full else result.a_only[:8]):
            console.print(f"      {r:>3}  {share:>5.0%}  [dim]{escape(terms)}[/dim]")
        if not full and len(result.a_only) > 8:
            console.print(f"      [dim]and {len(result.a_only) - 8} more; --full to "
                          f"list them[/dim]")
    else:
        console.print("\n    [dim]Every region left occupies is already covered by "
                      "right.[/dim]")

    if result.b_only:
        console.print("\n    [bold]Only in right[/bold]")
        for r, share, terms in (result.b_only if full else result.b_only[:5]):
            console.print(f"      {r:>3}  {share:>5.0%}  [dim]{escape(terms)}[/dim]")

    shifts = [row for row in result.category_shift if abs(row[2] - row[3]) >= 0.02]
    if shifts:
        console.print("\n    [bold]Category mix[/bold]")
        table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2),
                      pad_edge=False)
        table.add_column("category")
        table.add_column("left", justify="right")
        table.add_column("right", justify="right")
        table.add_column("delta", justify="right")
        for _cid, key, sa, sb in (shifts if full else shifts[:8]):
            delta = sa - sb
            style = "green" if delta > 0 else "red"
            table.add_row(key, f"{sa:.0%}", f"{sb:.0%}",
                          f"[{style}]{delta:+.0%}[/{style}]")
        console.print(table)

    partial = min(result.a_head_coverage, result.b_head_coverage)
    if partial < 0.999:
        console.print(f"\n  [yellow]note[/yellow] one side predates the full region "
                      f"histogram, so shares are computed over the stored top regions: "
                      f"{result.a_head_coverage:.0%} of left, "
                      f"{result.b_head_coverage:.0%} of right. Re-scan for exact numbers.")
    console.print("  [dim]This is geometry, not a recommendation. Whether new coverage "
                  "helps depends on what you are training.[/dim]\n")
    raise typer.Exit(EXIT_OK)


def _diff_shape(a: dict, b: dict) -> None:
    """Size and redundancy, side by side."""
    sa = (a.get("facets", {}).get("shape") or {}).get("values", {})
    sb = (b.get("facets", {}).get("shape") or {}).get("values", {})
    ra = (a.get("facets", {}).get("redundancy") or {}).get("values", {})
    rb = (b.get("facets", {}).get("redundancy") or {}).get("values", {})
    if not sa and not sb:
        return

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2),
                  pad_edge=False)
    table.add_column("")
    table.add_column("left", justify="right")
    table.add_column("right", justify="right")
    rows = [
        ("records", f"{sa.get('records', 0):,}", f"{sb.get('records', 0):,}"),
        ("datasets", f"{sa.get('datasets', 0):,}", f"{sb.get('datasets', 0):,}"),
        ("characters", f"{sa.get('total_chars', 0):,}", f"{sb.get('total_chars', 0):,}"),
    ]
    for label, key in (("near-duplicate rate", "near_duplicate_rate"),
                       ("exact-duplicate rate", "exact_duplicate_rate")):
        if key in ra or key in rb:
            rows.append((label, f"{ra.get(key, 0):.1%}", f"{rb.get(key, 0):.1%}"))
    console.print("\n  [bold]Shape[/bold]")
    for label, x, y in rows:
        table.add_row(label, x, y)
    console.print(table)


@app.command(
    "index-eval",
    no_args_is_help=True,
    epilog=(
        "Example: dropoutt index-eval ./holdout.jsonl "
        "--name internal-eval --field question"
    ),
)
def index_eval(
    path: Path = typer.Argument(..., help="JSONL file holding your held-out evaluation set."),
    name: str = typer.Option(..., "--name", "-n", help="Name for this benchmark."),
    field: str = typer.Option("text", "--field", "-f", help="Field holding the eval text."),
    force: bool = typer.Option(False, "--force", help="Overwrite an index with the same name."),
) -> None:
    """Build a contamination index from your own evaluation set, locally.

    The index stores hashed 8-grams rather than raw text. Keep it inside the
    evaluation set's trust boundary: known phrases can still be tested against
    an unkeyed hash index.
    """
    from .contamination import BenchmarkIndex
    from .readers import read_file

    if not path.exists():
        console.print(f"[red]No such file:[/red] {escape(str(path))}")
        raise typer.Exit(EXIT_USAGE)
    if not path.is_file():
        console.print(f"[red]Not a file:[/red] {escape(str(path))}")
        console.print("  [dim]Pass one JSONL or NDJSON evaluation file.[/dim]")
        raise typer.Exit(EXIT_USAGE)
    if path.suffix.lower() not in {".jsonl", ".ndjson"}:
        console.print(f"[red]Unsupported evaluation file:[/red] {escape(str(path))}")
        console.print("  [dim]`index-eval` accepts .jsonl or .ndjson files.[/dim]")
        raise typer.Exit(EXIT_USAGE)
    if not name.strip() or Path(name).name != name or name in {".", ".."}:
        console.print(f"[red]Invalid benchmark name:[/red] {escape(repr(name))}")
        console.print("  [dim]Use a filename-safe name without path separators.[/dim]")
        raise typer.Exit(EXIT_USAGE)

    idx = BenchmarkIndex(name=name, n_instances=0, source=path.name)
    n = 0
    with _ProgressDisplay() as activity:
        activity.phase(f"Indexing {path.name}")
        for rec in read_file(str(path), path.suffix):
            if rec.error or not isinstance(rec.payload, dict):
                continue
            text = rec.payload.get(field)
            if not isinstance(text, str) or not text.strip():
                parts = [v for v in rec.payload.values() if isinstance(v, str)]
                text = " ".join(parts)
            if not text.strip():
                continue
            idx.add_instance(n, text)
            n += 1
            if n % 2_000 == 0:
                activity.records(path.name, n)
        activity.finish(f"Hashed {n:,} evaluation records" if n else None)
    idx.n_instances = n
    if n == 0:
        console.print(f"[red]No evaluation text found in:[/red] {escape(str(path))}")
        console.print(
            f"  [dim]Field {field!r} was empty or absent. Pass --field with the "
            "column containing the evaluation text.[/dim]"
        )
        raise typer.Exit(EXIT_USAGE)

    # Always the cache, never the install tree. Writing into the package would
    # put a user's private eval index inside site-packages and fail outright
    # wherever the install is read-only, which is the normal case on a shared
    # cluster. Both locations are searched at scan time, so this stays visible.
    out_dir = cache_dir() / "contamination"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.idx"
    if dest.exists() and not force:
        console.print(
            f"[yellow]{escape(str(dest))} already exists.[/yellow] Use --force to overwrite."
        )
        raise typer.Exit(EXIT_USAGE)
    idx.save(dest)
    console.print(f"  Indexed [bold]{n:,}[/bold] instances into {escape(str(dest))}")
    console.print(
        f"  [dim]{len(idx.postings):,} distinct 8-grams; no raw text was stored. "
        "Keep the index inside the evaluation set's trust boundary.[/dim]"
    )


@app.command()
def checks(
    check_id: Optional[str] = typer.Argument(None, help="Show one check in detail."),
) -> None:
    """List the check catalog."""
    from .checks.base import REGISTRY

    if check_id:
        cls = REGISTRY.get(check_id.upper())
        if cls is None:
            console.print(f"[red]No such check:[/red] {escape(check_id)}")
            raise typer.Exit(EXIT_USAGE)
        console.print(f"\n  [bold]{cls.check_id}[/bold]  {cls.title}")
        console.print(f"  tier {cls.tier} · {cls.cost.value} · {cls.severity.value} "
                      f"· {cls.confidence.value}")
        console.print(f"  profiles: {', '.join(p.value for p in cls.profiles)}")
        if cls.requires:
            console.print(f"  requires: {', '.join(r.value for r in cls.requires)}")
        if cls.blocking_in:
            console.print(f"  blocks under: {', '.join(p.value for p in cls.blocking_in)}")
        console.print(f"\n  [bold]Fix[/bold]  {cls.fix}")
        if cls.rationale:
            console.print(f"\n  {cls.rationale}\n")
        return

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("id", no_wrap=True)
    table.add_column("tier", justify="right")
    table.add_column("title")
    table.add_column("needs", style="dim")
    for cls in REGISTRY.all():
        table.add_row(
            cls.check_id, str(cls.tier), cls.title,
            ", ".join(r.value for r in cls.requires) or "-",
        )
    console.print(table)
    console.print(f"\n  [dim]{len(REGISTRY.all())} checks. "
                  f"All findings are unverified in this build.[/dim]\n")


@app.command()
def benchmarks() -> None:
    """List the benchmark registry used for contamination scanning."""
    from .contamination import load_indices
    from .registry_data import benchmarks as bench_data

    have = load_indices(*_contamination_dirs())
    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("id", no_wrap=True)
    table.add_column("eval split")
    table.add_column("licence")
    table.add_column("index")
    for b in bench_data()["benchmarks"]:
        status = "[green]built[/green]" if b["id"] in have.benchmarks else (
            "shippable" if b.get("shippable") else "[dim]not distributable[/dim]"
        )
        table.add_row(b["id"], f'{b["eval_split"]}', b.get("license") or "[dim]none[/dim]", status)
    console.print(table)
    console.print("\n  [dim]TurkBench has no downloadable dataset: its test set is private "
                  "and it is submission-only, so no index can exist.[/dim]\n")


@app.command()
def models() -> None:
    """List known models and shorthand aliases."""
    from .registry_data import models as model_data

    data = model_data()
    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("hf id", no_wrap=True)
    table.add_column("template")
    table.add_column("licence", style="dim")
    for m in data["models"]:
        table.add_row(m["hf_id"], m.get("template") or "-", m.get("license") or "-")
    console.print(table)
    console.print("\n  [dim]aliases: " + ", ".join(sorted(data["aliases"])) + "[/dim]\n")


@app.command()
def atlas() -> None:
    """Show the bundled atlas and its own quality numbers.

    The accuracy and purity figures are printed next to any coverage number for
    a reason: an atlas whose level-0 probe is weak produces coverage histograms
    that look precise and are not.
    """
    from .atlas import bundled_atlas_path, load_bundled

    path = bundled_atlas_path()
    if path is None:
        console.print("[yellow]No atlas bundled with this install.[/yellow]")
        raise typer.Exit(EXIT_OK)
    atl = load_bundled()
    if atl is None:
        console.print(f"[red]Atlas at {path} could not be loaded.[/red]")
        raise typer.Exit(EXIT_ERROR)

    meta = atl.meta
    console.print(f"\n  [bold]{meta.get('version')}[/bold]  {path}")
    console.print(f"  artifact hash      {atl.artifact_hash}")
    console.print(f"  embedding model    {atl.embed_model} ({atl.dim} dims)")
    console.print(f"  regions            {atl.n_regions}")
    console.print(f"  reference records  {meta.get('n_reference_records'):,}")
    console.print(f"  off-atlas cutoff   {atl.off_threshold:.3f} cosine")
    console.print("\n  [bold]Quality of this atlas[/bold]")
    console.print(f"    level-0 held-out accuracy    {meta.get('l0_holdout_accuracy'):.3f}")
    console.print(f"    region purity by taxonomy    {meta.get('region_purity_by_taxonomy'):.3f}")

    manifest = meta.get("manifest", [])
    ok = [m for m in manifest if m.get("collected")]
    bad = [m for m in manifest if not m.get("collected")]
    console.print(f"\n  built from {len(ok)} sources; {len(bad)} unavailable at build time")
    for m in bad[:6]:
        console.print(f"    [dim]skipped {m['hf_id']}[/dim]")

    console.print("\n  [bold]Largest regions[/bold]")
    terms = atl.region_terms
    for i in range(min(8, len(terms))):
        console.print(f"    {i:>3}  [dim]{terms[i]}[/dim]")
    console.print("\n  [dim]This is a coordinate system, not a quality reference. "
                  "It contains no notion of good or bad.[/dim]\n")


@app.command()
def fetch(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Also fetch this model's tokenizer."
    ),
    all_: bool = typer.Option(
        False, "--all", help="Fetch the whole comparison panel, not just one tokenizer."
    ),
) -> None:
    """Pre-download everything a later --offline run needs.

    Written for the cluster shape where the login node has network and the
    compute node has none. Run this there, point DROPOUTT_CACHE at shared
    storage, and every subsequent scan can use --offline.

    The atlas artifact and the contamination indices ship inside the package and
    are never downloaded. What this fetches is the atlas embedding model and
    tokenizers, which are too large to vendor.
    """
    from .atlas import embed as embed_mod

    failed = False
    console.print()
    console.print(f"  cache: [dim]{escape(str(cache_dir()))}[/dim]")

    wanted: list[tuple[str, str]] = []
    if model:
        # The alias table only, not resolve_model(): that loads the tokenizer as
        # a side effect and returns a dataclass, and here we want the id.
        wanted.append((model, resolve_model_alias(model)))
    if all_ or not model:
        from .tokenizer_panel import PANEL

        wanted.extend(PANEL)

    console.print("\n  [bold]Tokenizers[/bold]")
    if not HAVE_TOKENIZERS:
        failed = True
        hint = escape("pip install 'dropoutt[tokenizer]'")
        console.print("    [yellow]skipped[/yellow] "
                      f"[dim]tokenizers not installed; {hint}[/dim]")
    else:
        from .tokenizer_panel import load_tokenizer

        from .config import _load_tokenizer_config

        seen: set[str] = set()
        for name, model_id in wanted:
            if model_id in seen:
                continue
            seen.add(model_id)
            with _ProgressDisplay() as activity:
                activity.phase(f"Fetching tokenizer {name}")
                handle = load_tokenizer(model_id)
                # Cache tokenizer_config.json in the same pass. It carries the chat
                # template, and without it an --offline run counts raw text and
                # skips the loss-mask checks entirely.
                has_template = bool(_load_tokenizer_config(model_id).get("chat_template"))
            if handle and has_template:
                mark, extra = "[green]ok[/green]", ""
            elif handle:
                failed = True
                mark, extra = "[yellow]partial[/yellow]", "  [dim]no chat template[/dim]"
            else:
                failed = True
                mark, extra = "[red]failed[/red]", ""
            console.print(f"    {mark:<22} {escape(name)}  "
                          f"[dim]{escape(model_id)}[/dim]{extra}")

    console.print("\n  [bold]Atlas embedding model[/bold]")
    if not HAVE_MODEL2VEC:
        failed = True
        hint = escape("pip install 'dropoutt[atlas]'")
        console.print("    [yellow]skipped[/yellow] "
                      f"[dim]model2vec not installed; {hint}[/dim]")
    else:
        with _ProgressDisplay() as activity:
            activity.phase(f"Fetching atlas model {embed_mod.DEFAULT_MODEL}")
            emb = embed_mod.load()
        if emb is None:
            failed = True
            console.print("    [red]failed[/red] [dim]could not download "
                          f"{escape(embed_mod.DEFAULT_MODEL)}[/dim]")
        else:
            console.print(f"    [green]ok[/green]  [dim]{escape(emb.name)} "
                          f"({emb.dim} dims)[/dim]")

    console.print("\n  [dim]Bundled in the package, nothing to fetch: the atlas "
                  "artifact and the contamination indices.[/dim]")
    if failed:
        console.print(
            "  [red]Cache preparation is incomplete.[/red] Fix the failures above "
            "before running on a node without egress.\n"
        )
        raise typer.Exit(EXIT_ERROR)
    console.print("  [dim]Now run scans with --offline. Keep DROPOUTT_CACHE set to "
                  "this same path.[/dim]\n")


@app.command()
def doctor() -> None:
    """Show what is installed and what each missing piece costs."""
    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("component")
    table.add_column("status")
    table.add_column("without it")
    table.add_column("install", style="dim")
    for name, info in capability_report().items():
        status = "[green]yes[/green]" if info["available"] else "[yellow]no[/yellow]"
        # Escape the extras syntax: rich would read `[fast]` as a style tag
        # and silently drop it, turning the install hint into a wrong command.
        hint = "" if info["available"] else escape(str(info["install"]))
        table.add_row(name, status, str(info["impact"]), hint)
    console.print(table)
    console.print(f"\n  cache: [dim]{cache_dir()}[/dim]")
    console.print(f"  version: [dim]{__version__}[/dim]\n")


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit(EXIT_OK)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(EXIT_OK)


# --------------------------------------------------------------------------


def _contamination_dirs() -> tuple[Path, Path]:
    """Where contamination indices are read from, cache first.

    Both locations are always searched. The cache holds whatever the user built
    with `index-eval`; the package holds the ten shipped benchmarks. Returning
    only one of them meant that building a private index switched off every
    bundled benchmark without saying so. Cache comes first so a user index can
    intentionally shadow a shipped one of the same name.
    """
    from importlib import resources

    shipped = Path(str(resources.files("dropoutt.data") / "contamination"))
    return cache_dir() / "contamination", shipped


def _budget(result, total_chars: int, *, offline: bool):
    """Estimate the token budget from a stratified sample.

    Tokens-per-character is stable within a corpus, so extrapolating from a few
    hundred thousand sampled records against an exact character count gives an
    interval rather than a guess.
    """
    from .tokenizer_panel import estimate_budget

    sample_texts = result.ctx.stats.get("budget_sample", [])
    return estimate_budget(
        sample_texts,
        result.ctx.stats.get("total_chars", total_chars),
        result.ctx.stats.get("total_words", 0),
        offline=offline,
    )


def _write_outputs(
    out_dir: Path,
    result,
    fp,
    budget,
    *,
    write_html: bool,
    include_evidence: bool = True,
) -> None:
    from .report import html as html_report

    (out_dir / "fingerprint.json").write_text(
        json_dumps(fp.to_dict(), indent=True), encoding="utf-8"
    )
    with open(out_dir / "findings.jsonl", "w", encoding="utf-8") as fh:
        for f in result.findings:
            fh.write(json_dumps({
                "check_id": f.check_id,
                "title": f.title,
                "severity": f.severity.value,
                "confidence": f.confidence.value,
                "count": f.count,
                "total_considered": f.total_considered,
                "rate": f.rate,
                "detail": f.detail,
                "fix": f.fix,
                "wasted_tokens": f.wasted_tokens,
                "by_dataset": f.by_dataset,
                "would_block_under": list(f.would_block_under),
                "is_blocking": f.is_blocking,
                "data": f.data if include_evidence else _without_source_locations(f.data),
                "evidence": [
                    {
                        "doc_id": e.doc_id,
                        "source_file": e.source_file,
                        "source_index": e.source_index,
                        "excerpt": e.excerpt,
                        "partner_doc_id": e.partner_doc_id,
                        "score": e.score,
                    }
                    for e in f.evidence
                ] if include_evidence else [],
            }) + "\n")
    if write_html:
        (out_dir / "report.html").write_text(
            html_report.render(result, fp, budget, include_evidence=include_evidence),
            encoding="utf-8",
        )


def _read_fingerprint(path: Path, side: str) -> dict:
    """Read one diff operand and reject unrelated JSON or binary artifacts."""
    example = (
        "dropoutt diff ./candidate/.dropoutt/fingerprint.json "
        "./mixture/.dropoutt/fingerprint.json"
    )
    if not path.exists():
        console.print(f"[red]No such file:[/red] {escape(str(path))}")
        console.print(f"  [dim]Expected the {side} fingerprint written by `dropoutt scan`.[/dim]")
        console.print(f"  [dim]Example: {example}[/dim]")
        raise typer.Exit(EXIT_USAGE)
    if not path.is_file():
        console.print(f"[red]Not a file:[/red] {escape(str(path))}")
        console.print(f"  [dim]Pass a .dropoutt/fingerprint.json file. Example: {example}[/dim]")
        raise typer.Exit(EXIT_USAGE)
    if path.suffix.lower() != ".json":
        console.print(f"[red]Not a fingerprint JSON file:[/red] {escape(str(path))}")
        if path.suffix.lower() == ".npz":
            console.print(
                "  [dim]That is an atlas artifact. `dropoutt diff` compares two "
                "fingerprints generated by `dropoutt scan`; it does not compare a "
                "fingerprint with the bundled atlas file.[/dim]"
            )
        console.print(f"  [dim]Example: {example}[/dim]")
        raise typer.Exit(EXIT_USAGE)
    try:
        value = json_loads(path.read_text(encoding="utf-8"))
    except UnicodeError:
        console.print(f"[red]Not a UTF-8 fingerprint JSON file:[/red] {escape(str(path))}")
        console.print(f"  [dim]Example: {example}[/dim]")
        raise typer.Exit(EXIT_USAGE) from None
    except Exception:
        console.print(f"[red]Invalid JSON fingerprint:[/red] {escape(str(path))}")
        console.print(
            "  [dim]Pass the .dropoutt/fingerprint.json file written by "
            "`dropoutt scan`, without editing or converting it.[/dim]"
        )
        console.print(f"  [dim]Example: {example}[/dim]")
        raise typer.Exit(EXIT_USAGE) from None

    required = ("fingerprint_id", "schema_version", "pipeline_version", "facets")
    missing = [key for key in required if not isinstance(value, dict) or key not in value]
    if missing:
        console.print(f"[red]Not a dropoutt fingerprint:[/red] {escape(str(path))}")
        console.print(f"  [dim]Missing required field(s): {', '.join(missing)}.[/dim]")
        console.print(f"  [dim]Example: {example}[/dim]")
        raise typer.Exit(EXIT_USAGE)
    if not isinstance(value["facets"], dict):
        console.print(f"[red]Invalid dropoutt fingerprint:[/red] {escape(str(path))}")
        console.print("  [dim]The `facets` field must be a JSON object. Re-run `dropoutt scan`.[/dim]")
        raise typer.Exit(EXIT_USAGE)
    for name, facet in value["facets"].items():
        if not isinstance(facet, dict) or not isinstance(facet.get("values", {}), dict):
            console.print(f"[red]Invalid dropoutt fingerprint:[/red] {escape(str(path))}")
            console.print(
                f"  [dim]Facet {name!r} has an invalid structure. Re-run `dropoutt scan`.[/dim]"
            )
            raise typer.Exit(EXIT_USAGE)
    return value


def _without_source_locations(value):
    """Remove nested contamination witnesses from evidence-free output."""
    if isinstance(value, dict):
        return {
            key: _without_source_locations(item)
            for key, item in value.items()
            if key not in {"record", "source_file"}
        }
    if isinstance(value, list):
        return [_without_source_locations(item) for item in value]
    if isinstance(value, tuple):
        return [_without_source_locations(item) for item in value]
    return value


def _offline_from_environment() -> bool:
    """Honor both the product-specific and Hugging Face offline contracts."""
    truthy = {"1", "true", "yes", "on"}
    return any(
        os.environ.get(name, "").strip().lower() in truthy
        for name in ("DROPOUTT_OFFLINE", "HF_HUB_OFFLINE")
    )


def _validate_profile(value: str, *, option: str, allow_auto: bool) -> None:
    choices = {"sft", "corpus", "preference"}
    if allow_auto:
        choices.add("auto")
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        console.print(f"[red]Invalid --{option} value:[/red] {escape(str(value))}")
        console.print(f"  [dim]Choose one of: {allowed}.[/dim]")
        raise typer.Exit(EXIT_USAGE)


def _invalid_config(message: str) -> None:
    console.print(f"[red]Invalid dropoutt.toml:[/red] {escape(message)}")
    console.print("  [dim]Fix the [scan] value or override it on the command line.[/dim]")
    raise typer.Exit(EXIT_USAGE)


def _render_confirmation(resolved) -> None:
    """Render a couple of records and show the trainable span.

    Asserting that we understand the template is cheap; demonstrating it takes
    fifteen seconds and catches template and masking mismatches immediately.
    """
    if resolved.chat_template is None:
        return
    probe = [
        {"role": "user", "content": "Merhaba, nasılsın?"},
        {"role": "assistant", "content": "İyiyim, teşekkürler."},
    ]
    try:
        out = resolved.chat_template.render(probe)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]template failed to render:[/red] {escape(str(exc))}")
        return
    console.print(f"\n  [bold]Template check[/bold] ({escape(resolved.model_id)})")
    console.print(f"  [dim]{escape(repr(out.text))}[/dim]")
    spans = out.generation_spans
    if not spans:
        _, spans = resolved.chat_template.spans_by_difference(probe)
        console.print("  [dim]span source: difference (template has no generation tag)[/dim]")
    else:
        console.print("  [dim]span source: generation tag[/dim]")
    for a, b in spans:
        console.print(f"  trainable: [green]{escape(repr(out.text[a:b]))}[/green]")


def run() -> None:
    """Console entry point with concise handling for genuine internal failures."""
    try:
        app()
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("DROPOUTT_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            raise
        console.print(
            f"[red]dropoutt failed internally ({type(exc).__name__}).[/red] "
            "Re-run with DROPOUTT_DEBUG=1 for a traceback."
        )
        raise SystemExit(EXIT_ERROR) from None


if __name__ == "__main__":  # pragma: no cover
    run()
