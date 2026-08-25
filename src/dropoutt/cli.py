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
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .compat import (
    HAVE_SCIPY,
    HAVE_TOKENIZERS,
    capability_report,
    json_dumps,
)
from .config import Config, cache_dir, parse_profile, resolve_model
from .models import Profile
from .progress import ProgressDisplay, start_warmup
from .registry_data import resolve_model_alias

app = typer.Typer(
    name="dropoutt",
    help=(
        "Pre-flight checks for LLM training data.\n\n"
        "Point dropoutt at a folder and it reports what would go wrong in a "
        "training run — broken loss masks, duplicates, contamination, language "
        "damage, token cost — plus where the corpus sits on a fixed map of "
        "public training data. Everything runs locally."
    ),
    add_completion=False,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)
console = Console()

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 10


@app.command(
    no_args_is_help=True,
    # Rich collapses single newlines in an epilog, so each example is its own
    # paragraph. Four of them is the most that reads as a list rather than a
    # wall.
    epilog=(
        "[b]Examples[/b]\n\n"
        "[dim]$[/dim] dropoutt scan ./data\n\n"
        "[dim]$[/dim] dropoutt scan ./data --model qwen3  "
        "[dim]# exact token counts[/dim]\n\n"
        "[dim]$[/dim] dropoutt scan ./data --target sft   "
        "[dim]# exit 10 on blocking findings[/dim]\n\n"
        "[dim]$[/dim] dropoutt scan ./data --offline      "
        "[dim]# never touch the network[/dim]"
    ),
)
def scan(
    path: Path = typer.Argument(..., help="File or directory to scan."),
    model: str | None = typer.Option(None, "--model", "-m",
                                     help="Target model id or local path. Unlocks token checks."),
    profile: str = typer.Option("auto", "--profile", "-p",
                                help="sft, corpus, preference, or auto."),
    target: str | None = typer.Option(None, "--target",
                                      help="Declare what you are building. Enables blocking."),
    seq_len: int | None = typer.Option(
        None, "--seq-len", min=1, help="Training sequence length."
    ),
    tier: int | None = typer.Option(
        None, "--tier", min=0, help="Highest check tier to run. Defaults to config, then 1."
    ),
    out: Path | None = typer.Option(None, "--out", "-o",
                                    help="Directory for findings, fingerprint and report."),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Max records per file."
    ),
    no_html: bool = typer.Option(False, "--no-html", help="Skip the HTML report."),
    no_open: bool = typer.Option(
        False, "--no-open", help="Do not open the report when the scan finishes."
    ),
    no_atlas: bool = typer.Option(False, "--no-atlas", help="Skip atlas coverage."),
    no_evidence: bool = typer.Option(
        False,
        "--no-evidence",
        help="Omit record excerpts and source locations from terminal and output reports.",
    ),
    workers: int | None = typer.Option(
        None, "--workers", "-j", min=1,
        help="Processes for the scan pass. Defaults to one per core, less one.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q",
                               help="Suppress the report; write output files and exit."),
    brief: bool = typer.Option(
        False, "--brief",
        help="Print the verdict and one line per finding instead of the full report.",
    ),
) -> None:
    """Check a folder of training data and write a report.

    Runs every check that this install can run, writes findings.jsonl, a
    comparable fingerprint and an HTML report, prints a summary, then opens the
    report if there is a desktop to open it on. No flags are required; each
    check that could not run names the one flag that would unlock it.
    """
    if not path.exists():
        console.print(f"[red]No such path:[/red] {escape(str(path))}")
        raise typer.Exit(EXIT_USAGE)

    from .atlas import load_bundled
    from .contamination import load_indices
    from .discovery import discover
    from .fingerprint import build as build_fingerprint
    from .langid import LanguageDetector
    from .parallel import MIN_BYTES_FOR_PARALLEL
    from .report import terminal as term_report
    from .runner import scan as run_scan

    with ProgressDisplay(enabled=not quiet) as activity:
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

        # The embedding model and the tokenizer comparison panel are three and a
        # half seconds of file reading that depend on nothing in the data. Start
        # them now, in threads, so they land while records are being read.
        #
        # They are joined before the scan pass if that pass is going to fork
        # workers, because forking from a process whose threads are inside a
        # Rust library is how you get a child that hangs on its first
        # allocation. Below the parallel threshold there is no fork and the
        # warm-up overlaps the whole scan.
        warmup = start_warmup(
            offline=offline, want_embedder=not no_atlas, want_panel=model is None
        )
        will_fork = (
            (workers is None or workers > 1)
            and preflight.total_bytes >= MIN_BYTES_FOR_PARALLEL
        )

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
                    "  [dim]Run `dropoutt benchmarks` for the names that exist.[/dim]"
                )
                raise typer.Exit(EXIT_USAGE)
            contamination.benchmarks = {
                name: index
                for name, index in contamination.benchmarks.items()
                if name in requested
            }
        atlas_obj = None if no_atlas else load_bundled()
        if will_fork:
            warmup.shutdown(wait=True)

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
            discovery=preflight,
            workers=workers,
            contamination_dirs=_contamination_dirs(),
            eval_sets=list(cfg.eval_sets),
        )

        # Count the text that would actually be trained on, not the bytes on disk.
        # `total_bytes` includes JSON syntax, keys and metadata columns, and in UTF-8
        # a Turkish or Arabic corpus costs more bytes per character than an English
        # one. Putting that number in a fingerprint would make the one artifact meant
        # to be comparable across datasets vary with language and file format. The
        # runner already accumulates real character and word counts over the
        # normalised text; use those, and fall back to bytes only if it ran no
        # records at all.
        warmup.shutdown(wait=True)
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
        # Read the result once. Both reports render the same reading, and
        # building it twice is how they used to disagree.
        from .report.summary import build as build_summary

        story = build_summary(result, budget=budget, include_evidence=not no_evidence)
        _write_outputs(
            out_dir,
            result,
            fp,
            budget,
            write_html=not no_html,
            include_evidence=not no_evidence,
            summary=story,
        )
        activity.finish(f"Scanned {result.records_scanned:,} records")

    for note in model_notes:
        console.print(f"  [yellow]note[/yellow] {escape(note)}")

    # Most-human first: the order is the order someone picks one in.
    written = (["report.html"] if not no_html else []) + [
        "report.md", "report.json", "findings.jsonl", "fingerprint.json",
    ]
    if not quiet:
        # The screen carries the whole report; the files carry the same thing in
        # other shapes, so it names them rather than the caller announcing them
        # separately. `--brief` returns to the one-line-per-finding view.
        term_report.render(
            console, result, budget=budget, show_evidence=not no_evidence,
            summary=story, out_dir=out_dir, written=written, brief=brief,
            fingerprint=fp,
        )
    else:
        console.print(f"  [dim]wrote {escape(str(out_dir))}/{', '.join(written)}[/dim]")
    if no_evidence:
        console.print("  [dim]record excerpts and source locations were omitted[/dim]")
    else:
        quoting = [name for name in written if name != "fingerprint.json"]
        console.print(
            f"  [yellow]note[/yellow] {', '.join(quoting)} may contain dataset "
            "excerpts and source paths; use --no-evidence before exporting them"
        )
    if not no_html and not no_open and not quiet:
        _show(out_dir / "report.html")

    if result.ctx.blocking_enabled and result.blocking:
        console.print(f"\n  [bold red]{len(result.blocking)} blocking finding(s)[/bold red] "
                      f"under target profile {result.ctx.profile.value}")
        raise typer.Exit(EXIT_BLOCKED)
    raise typer.Exit(EXIT_OK)


@app.command(epilog="Example: dropoutt checks T0-DUP-001")
def checks(
    check_id: str | None = typer.Argument(None, help="Show one check in detail."),
) -> None:
    """List every check, or explain one in detail."""
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
    """List the evaluation sets scanned for contamination."""
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
    """List the models --model understands, and their shorthands."""
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


@app.command(epilog="Example: dropoutt fetch --model qwen3")
def fetch(
    model: str | None = typer.Option(
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
        hint = escape("reinstall dropoutt")
        console.print("    [yellow]skipped[/yellow] "
                      f"[dim]tokenizers not installed; {hint}[/dim]")
    else:
        from .config import _load_tokenizer_config
        from .tokenizer_panel import load_tokenizer

        seen: set[str] = set()
        for name, model_id in wanted:
            if model_id in seen:
                continue
            seen.add(model_id)
            with ProgressDisplay() as activity:
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
    if not HAVE_SCIPY:
        failed = True
        hint = escape("reinstall dropoutt")
        console.print("    [yellow]skipped[/yellow] "
                      f"[dim]scipy not installed; {hint}[/dim]")
    else:
        with ProgressDisplay() as activity:
            activity.phase(f"Fetching atlas model {embed_mod.DEFAULT_MODEL}")
            emb = embed_mod.load()
        if emb is None:
            failed = True
            console.print("    [red]failed[/red] [dim]could not download "
                          f"{escape(embed_mod.DEFAULT_MODEL)}[/dim]")
        else:
            # The published weights are converted on the way in and the original
            # is deleted, so what is cached is not what was downloaded. Say so:
            # a user comparing the cache against the model card would otherwise
            # find four hundred megabytes missing and no explanation.
            console.print(f"    [green]ok[/green]  [dim]{escape(emb.name)} "
                          f"({emb.dim} dims, stored int8, "
                          f"{_dir_size(embed_mod.local_model_dir(emb.name, cache_dir()))})[/dim]")

    console.print("\n  [dim]Bundled in the package, nothing to fetch: the atlas "
                  "artifact and the contamination indices.[/dim]")

    _print_environment()
    if failed:
        console.print(
            "  [red]Cache preparation is incomplete.[/red] Fix the failures above "
            "before running on a node without egress.\n"
        )
        raise typer.Exit(EXIT_ERROR)
    console.print("  [dim]Now run scans with --offline. Keep DROPOUTT_CACHE set to "
                  "this same path.[/dim]\n")


def _dir_size(path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / (1 << 20):.0f} MB"


def _print_environment() -> None:
    """The interpreter, the cache, the machine, and anything that failed to import.

    This is what `dropoutt doctor` existed for. The command is gone — with no
    extras left there is no install decision for a capability table to inform,
    and a table that reads "yes" on every row six times is a command people stop
    running. What remains useful is the part that identifies *which* Python was
    probed, because the classic way to be confused by a missing import is to
    have installed it with a `pip` belonging to a different interpreter: a venv
    created by `uv` ships no pip, so an activated shell falls through to
    whichever pip is next on PATH and installs somewhere this process cannot
    see. That, plus the machine the scan would size itself against, is printed
    here — on the command whose whole job is preparing an environment.
    """
    from .hardware import plan

    console.print("\n  [bold]This environment[/bold]")
    console.print(f"    python   [dim]{escape(sys.executable)}[/dim]")
    console.print(f"    cache    [dim]{escape(str(cache_dir()))}[/dim]")
    console.print(f"    version  [dim]{escape(__version__)}[/dim]")
    console.print(f"    machine  [dim]{escape(plan().describe())}[/dim]")

    missing = [name for name, info in capability_report().items() if not info["available"]]
    if missing:
        console.print(
            f"\n  [yellow]{len(missing)} required package(s) could not be imported:[/yellow] "
            f"{escape(', '.join(missing))}"
        )
        console.print(
            "  [dim]Every dependency ships with dropoutt, so this means a broken "
            f"install rather than a missing extra. Repair it with:[/dim]\n"
            f"    [dim]{escape(_install_prefix())} --force-reinstall dropoutt[/dim]"
        )


def _install_prefix() -> str:
    """The installer command that targets *this* interpreter.

    `pip install` is wrong advice whenever the running interpreter has no pip,
    which is the default for venvs created by uv. In that case the shell's `pip`
    belongs to some other Python and the install silently lands out of reach.
    """
    import importlib.util

    if importlib.util.find_spec("pip") is not None:
        return f"{sys.executable} -m pip install"
    return "uv pip install"


@app.command("version")
def version_command() -> None:
    """Print the version and exit."""
    # highlight=False: rich colourises bare numbers, which would break the
    # version into differently-styled fragments and make it harder to copy.
    console.print(__version__, highlight=False)


@app.command("help")
def help_command(
    ctx: typer.Context,
    command: str | None = typer.Argument(None, help="Explain this command."),
) -> None:
    """Show this help, or the help for one command.

    `--help` is the convention and `help` is what people type. Both work, and
    both print the same text.
    """
    group_ctx = ctx.parent or ctx
    # The group, which owns the subcommands. Deliberately untyped: typer vendors
    # its own copy of click, so this is a TyperGroup whose base classes have
    # moved twice across click 8.x — under click 8.4 it does not derive from
    # click.Group at all, and an isinstance guard on that silently sent every
    # `help <command>` to the group's own help. Naming a class here would pin
    # this to a private typer module path that is expected to move again.
    group: Any = group_ctx.command
    if command is None:
        console.print(group_ctx.get_help())
        raise typer.Exit(EXIT_OK)

    sub = group.get_command(group_ctx, command)
    if sub is None:
        console.print(f"[red]No such command:[/red] {escape(command)}")
        known = ", ".join(sorted(group.list_commands(group_ctx)))
        console.print(f"  [dim]There is: {known}.[/dim]")
        raise typer.Exit(EXIT_USAGE)
    # The parent context contributes "dropoutt" to the usage line, so info_name
    # here is the bare command name.
    console.print(sub.get_help(typer.Context(sub, info_name=command, parent=group_ctx)))
    raise typer.Exit(EXIT_OK)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V",
                                 help="Show version and exit."),
) -> None:
    if version:
        console.print(__version__, highlight=False)
        raise typer.Exit(EXIT_OK)
    if ctx.invoked_subcommand is None:
        _greet()
        console.print(ctx.get_help())
        console.print(
            "  [dim]Start here:[/dim]  [bold]dropoutt scan ./data[/bold]\n"
            "  [dim]No flags needed. Add --model to unlock token checks, "
            "--target sft to fail CI on findings.[/dim]\n"
        )
        raise typer.Exit(EXIT_OK)


def _greet() -> None:
    """The mark, when there is a person and a terminal to show it to.

    Silent when stdout is redirected, when NO_COLOR is set, or when the console
    cannot encode the characters — see :mod:`dropoutt.branding`. A banner that
    lands in a CI log or crashes on a legacy Windows code page is worse than no
    banner.
    """
    from .branding import banner, supports_unicode, wants_decoration

    if not wants_decoration():
        return
    console.print()
    # highlight=False: rich colourises bare numbers by default, which turns the
    # version string into three differently-styled fragments.
    console.print(banner(__version__, width=console.width,
                         unicode_ok=supports_unicode()), highlight=False)
    console.print()


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
    """Estimate the token budget, stratified by dataset.

    Tokens-per-character is stable within one dataset and varies a lot between
    them, so each dataset's sampled ratio is applied to that dataset's own
    character count and the totals are added. Handing the estimator one pooled
    sample instead prices the corpus at the sample's blend of languages rather
    than its own, which is worth tens of percent on a mixed corpus.
    """
    from .tokenizer_panel import estimate_budget

    stats = result.ctx.stats
    return estimate_budget(
        stats.get("budget_sample", {}),
        stats.get("total_chars", total_chars),
        stats.get("total_words", 0),
        chars_by_dataset=stats.get("chars_by_dataset", {}),
        records_by_dataset={
            ds.name: ds.record_count for ds in result.ctx.datasets
        },
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
    summary=None,
) -> None:
    from .report import html as html_report
    from .report import json_report
    from .report import markdown as md_report

    (out_dir / "fingerprint.json").write_text(
        json_dumps(fp.to_dict(), indent=True), encoding="utf-8"
    )
    # These two are always written, and not behind --no-html. The flags mean
    # opposite things: --no-html is "do not build the thing I would open in a
    # browser", and these are the files for exactly that reader — a CI job
    # pasting a summary into a pull request, or a pipeline stage gating on
    # coverage. Both carry the same content as the page.
    (out_dir / "report.md").write_text(
        md_report.render(result, fp, budget, include_evidence=include_evidence,
                         summary=summary),
        encoding="utf-8",
    )
    (out_dir / "report.json").write_text(
        json_report.render(result, fp, budget, include_evidence=include_evidence,
                           summary=summary),
        encoding="utf-8",
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
            html_report.render(result, fp, budget, include_evidence=include_evidence,
                               summary=summary),
            encoding="utf-8",
        )


def _show(report: Path) -> None:
    """Open the finished report, when there is a screen in front of this process.

    Everything the scan promised is already on disk and already on screen by the
    time this runs, so a desktop that cannot open a file is worth one dim line
    and nothing more. See :mod:`dropoutt.desktop` for what counts as a reason
    not to.
    """
    from .desktop import open_report

    reason = open_report(report)
    if reason is None:
        console.print("  [dim]opened report.html[/dim]")


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



def run() -> None:
    """Console entry point with concise handling for genuine internal failures."""
    try:
        app()
    except Exception as exc:
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
