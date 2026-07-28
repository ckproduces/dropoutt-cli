"""Command line interface.

Exit codes are three-valued on purpose. A checker that returns the same code for
"found problems" and "crashed" cannot be used in CI.

    0   completed, whether or not findings were reported
    1   internal error
    2   usage error
    10  blocking findings, and only when a target profile was declared
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .compat import HAVE_MODEL2VEC, HAVE_TOKENIZERS, capability_report, json_dumps
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


@app.command()
def scan(
    path: Path = typer.Argument(..., help="File or directory to scan."),
    model: Optional[str] = typer.Option(None, "--model", "-m",
                                        help="Target model id or local path. Unlocks token checks."),
    profile: str = typer.Option("auto", "--profile", "-p",
                                help="sft, corpus, preference, or auto."),
    target: Optional[str] = typer.Option(None, "--target",
                                         help="Declare what you are building. Enables blocking."),
    seq_len: Optional[int] = typer.Option(None, "--seq-len", help="Training sequence length."),
    tier: int = typer.Option(1, "--tier", help="Highest check tier to run."),
    out: Optional[Path] = typer.Option(None, "--out", "-o",
                                       help="Directory for findings, fingerprint and report."),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max records per file."),
    no_html: bool = typer.Option(False, "--no-html", help="Skip the HTML report."),
    no_atlas: bool = typer.Option(False, "--no-atlas", help="Skip atlas coverage."),
    quiet: bool = typer.Option(False, "--quiet", "-q",
                               help="Suppress the report; write output files and exit."),
) -> None:
    """Scan a dataset directory."""
    if not path.exists():
        console.print(f"[red]No such path:[/red] {path}")
        raise typer.Exit(EXIT_USAGE)

    from .atlas import load_bundled
    from .contamination import load_indices
    from .fingerprint import build as build_fingerprint
    from .langid import LanguageDetector
    from .report import html as html_report
    from .report import terminal as term_report
    from .runner import scan as run_scan
    from .tokenizer_panel import estimate_budget

    cfg = Config.load(path)
    model = model or cfg.model
    target = target or cfg.target
    seq_len = seq_len or cfg.seq_len
    offline = offline or cfg.offline
    if profile == "auto":
        profile = cfg.profile

    tokenizer = chat_template = None
    if model:
        resolved = resolve_model(model, offline=offline)
        tokenizer = resolved.tokenizer
        chat_template = resolved.chat_template
        seq_len = seq_len or resolved.seq_len
        model = resolved.model_id
        for note in resolved.notes:
            console.print(f"  [yellow]note[/yellow] {note}")

    detector = LanguageDetector()
    contamination = load_indices(*_contamination_dirs())
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
        muted=tuple(cfg.mute),
        limit_per_file=limit,
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
    scanned_chars = int(result.ctx.stats.get("total_chars", 0))
    total_chars = scanned_chars or sum(d.total_bytes for d in result.ctx.datasets)
    total_words = int(result.ctx.stats.get("total_words", 0))
    budget = _budget(result, total_chars, offline=offline)
    fp = build_fingerprint(
        result.ctx, result.findings,
        total_chars=total_chars, total_words=total_words,
        budget=budget, config_hash=cfg.hash(),
    )

    if not quiet:
        term_report.render(console, result, budget=budget)

    out_dir = out or (path if path.is_dir() else path.parent) / ".dropoutt"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(out_dir, result, fp, budget, write_html=not no_html)
    console.print(f"  [dim]wrote {out_dir}/fingerprint.json, findings.jsonl"
                  f"{', report.html' if not no_html else ''}[/dim]")

    if result.ctx.blocking_enabled and result.blocking:
        console.print(f"\n  [bold red]{len(result.blocking)} blocking finding(s)[/bold red] "
                      f"under target profile {result.ctx.profile.value}")
        raise typer.Exit(EXIT_BLOCKED)
    raise typer.Exit(EXIT_OK)


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Directory to configure."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Infer configuration from your data and write dropoutt.toml."""
    from .discovery import discover
    from .readers import read_file
    from .runner import _infer_profile
    from .schema_induction import induce

    target_file = (path if path.is_dir() else path.parent) / CONFIG_NAME
    if target_file.exists() and not force:
        console.print(f"[yellow]{target_file} already exists.[/yellow] Use --force to overwrite.")
        raise typer.Exit(EXIT_USAGE)

    disc = discover(str(path))
    verdicts = {}
    for ds in disc.datasets[:50]:
        sample = []
        for f in ds.files[:2]:
            sample.extend(list(read_file(f, Path(f).suffix, limit=300)))
        if sample:
            verdicts[ds.name] = induce(sample)

    profile = _infer_profile(verdicts)
    notes = {"profile": f"inferred from {len(verdicts)} dataset(s)"}
    cfg = Config(model=model, profile=profile.value)

    if model:
        resolved = resolve_model(model)
        cfg.model = resolved.model_id
        cfg.seq_len = resolved.seq_len
        notes["model"] = "resolved from the Hub"
        if resolved.seq_len:
            notes["seq_len"] = "from the model config"
        _render_confirmation(resolved)

    target_file.write_text(cfg.to_toml(inferred_notes=notes), encoding="utf-8")
    console.print(f"\n  Wrote [bold]{target_file}[/bold]")
    console.print(f"  Detected profile: [bold]{profile.value}[/bold] "
                  f"from {len(verdicts)} dataset(s)")
    console.print("  [dim]Edit the file to declare a target if you want findings to block.[/dim]")


@app.command("index-eval")
def index_eval(
    path: Path = typer.Argument(..., help="JSONL file holding your held-out evaluation set."),
    name: str = typer.Option(..., "--name", "-n", help="Name for this benchmark."),
    field: str = typer.Option("text", "--field", "-f", help="Field holding the eval text."),
) -> None:
    """Build a contamination index from your own evaluation set, locally.

    The index stores hashed 8-grams, never the text, so it is safe to keep
    alongside your data. Your held-out set never leaves this machine.
    """
    from .contamination import BenchmarkIndex
    from .readers import read_file

    if not path.exists():
        console.print(f"[red]No such file:[/red] {path}")
        raise typer.Exit(EXIT_USAGE)

    idx = BenchmarkIndex(name=name, n_instances=0, source=str(path))
    n = 0
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
    idx.n_instances = n

    # Always the cache, never the install tree. Writing into the package would
    # put a user's private eval index inside site-packages and fail outright
    # wherever the install is read-only, which is the normal case on a shared
    # cluster. Both locations are searched at scan time, so this stays visible.
    out_dir = cache_dir() / "contamination"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.idx"
    idx.save(dest)
    console.print(f"  Indexed [bold]{n:,}[/bold] instances into {dest}")
    console.print(f"  [dim]{len(idx.postings):,} distinct 8-grams; no text was stored[/dim]")


@app.command()
def checks(
    check_id: Optional[str] = typer.Argument(None, help="Show one check in detail."),
) -> None:
    """List the check catalog."""
    from .checks.base import REGISTRY

    if check_id:
        cls = REGISTRY.get(check_id.upper())
        if cls is None:
            console.print(f"[red]No such check:[/red] {check_id}")
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
    model: str = typer.Option(
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
            handle = load_tokenizer(model_id)
            # Cache tokenizer_config.json in the same pass. It carries the chat
            # template, and without it an --offline run counts raw text and
            # skips the loss-mask checks entirely.
            has_template = bool(_load_tokenizer_config(model_id).get("chat_template"))
            if handle and has_template:
                mark, extra = "[green]ok[/green]", ""
            elif handle:
                mark, extra = "[yellow]partial[/yellow]", "  [dim]no chat template[/dim]"
            else:
                mark, extra = "[red]failed[/red]", ""
            console.print(f"    {mark:<22} {escape(name)}  "
                          f"[dim]{escape(model_id)}[/dim]{extra}")

    console.print("\n  [bold]Atlas embedding model[/bold]")
    if not HAVE_MODEL2VEC:
        hint = escape("pip install 'dropoutt[atlas]'")
        console.print("    [yellow]skipped[/yellow] "
                      f"[dim]model2vec not installed; {hint}[/dim]")
    else:
        emb = embed_mod.load()
        if emb is None:
            console.print("    [red]failed[/red] [dim]could not download "
                          f"{escape(embed_mod.DEFAULT_MODEL)}[/dim]")
        else:
            console.print(f"    [green]ok[/green]  [dim]{escape(emb.name)} "
                          f"({emb.dim} dims)[/dim]")

    console.print("\n  [dim]Bundled in the package, nothing to fetch: the atlas "
                  "artifact and the contamination indices.[/dim]")
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


def _write_outputs(out_dir: Path, result, fp, budget, *, write_html: bool) -> None:
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
                "data": f.data,
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
                ],
            }) + "\n")
    if write_html:
        (out_dir / "report.html").write_text(
            html_report.render(result, fp, budget), encoding="utf-8"
        )


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
        console.print(f"  [red]template failed to render:[/red] {exc}")
        return
    console.print(f"\n  [bold]Template check[/bold] ({resolved.model_id})")
    console.print(f"  [dim]{out.text!r}[/dim]")
    spans = out.generation_spans
    if not spans:
        _, spans = resolved.chat_template.spans_by_difference(probe)
        console.print("  [dim]span source: difference (template has no generation tag)[/dim]")
    else:
        console.print("  [dim]span source: generation tag[/dim]")
    for a, b in spans:
        console.print(f"  trainable: [green]{out.text[a:b]!r}[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
