"""Terminal output.

The layout is deliberate. Discovery, then schema induction, then the hypothesis
about what is being built, then findings, then the token budget, then what could
not be checked and the single flag that unlocks each one.

That last section is a feature rather than an apology: it is the progressive
disclosure ladder made visible, and it is the reason the first run on a folder
with no flags is still worth reading.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..models import Confidence, Finding, Severity
from ..runner import ScanResult
from ..tokenizer_panel import BudgetReport
from .escaping import safe_snippet

SEV_STYLE = {
    Severity.BLOCKING: "bold red",
    Severity.WARNING: "yellow",
    Severity.INFO: "cyan",
}


def _m(value: object) -> str:
    """Escape a value before it is interpolated into a rich markup string.

    Two distinct things are handled by the same call. Install hints are written
    as `pip install 'dropoutt[tokenizer]'`, and rich reads `[tokenizer]` as a
    style tag and silently deletes it, so the user is told to install a package
    that does not exist. Separately, dataset names and reasons come from scanned
    data, and a folder named `[red]` would otherwise inject styling into our
    own output. Anything derived from data or carrying an extras name goes
    through here.
    """
    return escape(str(value))


def render(
    console: Console,
    result: ScanResult,
    *,
    budget: BudgetReport | None = None,
    show_evidence: bool = True,
) -> None:
    ctx = result.ctx
    disc = result.discovery

    console.print()
    console.rule("[bold]dropoutt scan[/bold]", style="dim")

    # -- discovery -------------------------------------------------------
    fmt = ", ".join(f"{k.lstrip('.')} {v}" for k, v in sorted(disc.format_counts.items()))
    console.print(f"  [dim]Discovered[/dim]  {len(disc.files):,} files, "
                  f"{len(disc.datasets):,} datasets, {disc.total_bytes / 1e6:,.1f} MB")
    console.print(f"  [dim]Formats[/dim]     {_m(fmt) or 'none'}")
    if disc.empty_files:
        console.print(f"  [dim]Empty files[/dim] {len(disc.empty_files)}")

    # -- schema ----------------------------------------------------------
    real = {n: v for n, v in result.verdicts.items() if not v.not_training_data}
    logs = {n: v for n, v in result.verdicts.items() if v.not_training_data}
    if real:
        console.print()
        console.print("  [bold]Schema induction[/bold]")
        counts: dict[str, int] = {}
        for v in real.values():
            counts[v.layout_id] = counts.get(v.layout_id, 0) + 1
        for layout, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
            console.print(f"    {_m(f'{layout:<22}')} {n} dataset(s)")
    if logs:
        console.print()
        console.print("  [bold yellow]Not training data[/bold yellow]")
        for name, v in list(logs.items())[:6]:
            console.print(f"    {_m(f'{name:<40}')} {_m(v.not_training_data)}")
        if len(logs) > 6:
            console.print(f"    [dim]and {len(logs) - 6} more[/dim]")

    # -- hypothesis ------------------------------------------------------
    console.print()
    console.print("  [bold]Best guess at what you are building[/bold]")
    console.print(f"    Stage      {_m(ctx.profile.value)}")
    lang = _language_summary(result)
    if lang:
        console.print(f"    Language   {_m(lang)}")
    console.print("    [dim]Confidence: medium. Confirm with `dropoutt init`.[/dim]")

    # -- findings --------------------------------------------------------
    console.print()
    if result.findings:
        console.print("  [bold]Findings[/bold]")
        table = Table(show_header=True, header_style="dim", box=None, pad_edge=False,
                      padding=(0, 2, 0, 2))
        table.add_column("check", style="dim", no_wrap=True)
        table.add_column("")
        table.add_column("count", justify="right")
        table.add_column("detail")
        for f in sorted(result.findings, key=_finding_order):
            marker = Text("●", style=SEV_STYLE.get(f.severity, ""))
            count = f"{f.count:,}" if f.count else "-"
            table.add_row(f.check_id, marker, count, _detail_text(f))
        console.print(table)
    else:
        console.print("  [green]No findings.[/green]")

    if show_evidence:
        _render_evidence(console, result.findings)

    # -- token budget ----------------------------------------------------
    if budget is not None and budget.estimates:
        console.print()
        title = ("Token budget" if ctx.tokenizer is not None
                 else "Token budget (estimated, no --model given)")
        console.print(f"  [bold]{title}[/bold]")
        cheapest = budget.cheapest
        for est in sorted(budget.estimates, key=lambda e: e.total_tokens_est):
            if est.failed:
                console.print(f"    {_m(f'{est.name:<14}')} [dim]tokenizer unavailable[/dim]")
                continue
            premium = budget.premium_vs_cheapest(est)
            extra = f"  (+{premium:.0%})" if cheapest and est is not cheapest and premium > 0 else ""
            console.print(
                f"    {_m(f'{est.name:<14}')} ~{est.total_tokens_est / 1e6:>8,.2f}M tokens   "
                f"{est.tokens_per_word:.2f} tok/word{extra}"
            )
        for note in budget.notes:
            console.print(f"    [dim]{_m(note)}[/dim]")

    # -- atlas coverage --------------------------------------------------
    atlas_findings = [f for f in result.findings if f.check_id.startswith("T1-ATLAS-")]
    _render_coverage(
        console,
        ctx.stats.get("atlas_coverage"),
        ctx.stats.get("atlas_off_examples"),
        show_evidence=show_evidence,
        region_examples=ctx.stats.get("atlas_region_examples"),
        cohesion=ctx.stats.get("atlas_region_cohesion"),
        atlas_findings=atlas_findings,
    )

    # -- skipped ---------------------------------------------------------
    if result.skipped:
        console.print()
        console.print("  [bold]Not checked, and why[/bold]")
        seen: set[str] = set()
        for s in result.skipped:
            if s.reason in seen:
                continue
            seen.add(s.reason)
            console.print(f"    {_m(f'{s.title:<46}')} [dim]{_m(s.reason)}[/dim]")
            if s.unlock:
                console.print(f"      [dim]→ {_m(s.unlock)}[/dim]")

    # -- degradations ----------------------------------------------------
    if ctx.degradations:
        console.print()
        console.print("  [bold]Degraded[/bold]")
        for d in ctx.degradations[:8]:
            console.print(f"    [dim]{_m(d)}[/dim]")

    # -- footer ----------------------------------------------------------
    console.print()
    unverified = sum(1 for f in result.findings if f.confidence is Confidence.UNVERIFIED)
    if unverified:
        console.print(
            "  [dim]All findings in this build are unverified: no measured effect size is "
            "attached to acting on them. They are structural observations, not predictions.[/dim]"
        )
    if not ctx.blocking_enabled:
        console.print("  [dim]No blocking verdict issued: no target declared. "
                      "Run `dropoutt init` or pass --target.[/dim]")
    console.print(f"  [dim]{result.records_scanned:,} records in {result.elapsed:.1f}s[/dim]")
    console.print()


def _render_coverage(
    console: Console,
    cov: dict | None,
    examples: list[dict] | None = None,
    *,
    show_evidence: bool = True,
    region_examples: dict | None = None,
    cohesion: dict | None = None,
    atlas_findings: list[Finding] | None = None,
) -> None:
    """Plot the corpus on a frozen map of professional training data.

    The atlas is a shared coordinate system compressed from high-quality public
    datasets. The decision questions it answers are what is present, what is
    missing, and what sits outside the map — not a quality score.
    """
    from ..atlas.compare import concentration

    if not cov:
        return

    console.print()
    console.print(f"  [bold]Atlas map[/bold] [dim]({_m(cov.get('atlas_version', '?'))})[/dim]")
    console.print("    [dim]Your corpus on a frozen map of professional training data — "
                  "what it covers, what it misses, what sits off the map.[/dim]")

    status = cov.get("status")
    if status not in ("ok", "suppressed", "none placed"):
        from ..atlas.compare import unusable_reason

        console.print(f"    [dim]{_m(unusable_reason(cov))}[/dim]")
        return

    records = int(cov.get("records", 0))
    placed_n = int(cov.get("placed", records - int(cov.get("off_atlas", 0))))
    short_n = int(cov.get("excluded_too_short", 0))
    min_chars = int(cov.get("min_chars", 80))

    if status == "ok":
        occupied = int(cov.get("regions_occupied", 0))
        total = int(cov.get("regions_total", 0))
        effective = float(cov.get("effective_regions", 0.0))
        conc = concentration(cov)
        console.print(f"    Placed       {placed_n:,} of {records:,} sampled "
                      f"[dim](every share below is over these {placed_n:,})[/dim]")
        if short_n:
            console.print(f"    Too short    {short_n:,} records under {min_chars} chars "
                          f"[dim](excluded from placement — embeddings need length)[/dim]")

        line = f"    Breadth      {occupied} of {total} regions touched"
        if effective:
            line += f", as spread out as {effective:.0f} even ones"
        console.print(line)
        if conc is not None:
            shape = "specialised" if conc < 0.45 else ("broad" if conc > 0.75 else "mixed")
            console.print(f"    Shape        {_m(_breadth_read(conc, shape, effective, total))}")

        if atlas_findings:
            for f in atlas_findings[:2]:
                console.print(f"    [yellow]→ {_m(f.check_id)}[/yellow] {_m(f.detail)}")

        # Absences first: the product reason for a frozen atlas.
        _render_gaps(console, cov)
        _render_categories(console, cov)
        _render_regions(
            console, cov, region_examples,
            show_evidence=show_evidence, cohesion=cohesion,
        )
        _render_dataset_signature(console, cov)

    _render_off_atlas(console, cov, examples, show_evidence=show_evidence)

    acc = cov.get("l0_holdout_accuracy")
    if acc is not None and status == "ok":
        console.print(f"    [dim]Subject-area labels are approximate "
                      f"(probe accuracy {acc:.3f}). Region placement is geometry, not labels.[/dim]")
    if status == "ok":
        console.print("    [dim]Ask what a candidate adds that you lack: "
                      "`dropoutt diff candidate.json have.json`[/dim]")


def _breadth_read(conc: float, shape: str, effective: float, total: int) -> str:
    """Say what the spread number means for the kind of corpus you might be building."""
    if shape == "specialised":
        return (f"{conc:.0%} of even coverage — specialised. Right for a single-task "
                f"set; too narrow for a pretraining mixture")
    if shape == "broad":
        return (f"{conc:.0%} of even coverage — broad. Right for a general corpus; "
                f"unusually scattered for a single-task set")
    return (f"{conc:.0%} of even coverage — mixed, covering roughly "
            f"{effective / total:.0%} of the atlas evenly")


def _render_regions(
    console: Console,
    cov: dict,
    region_examples: dict | None,
    *,
    show_evidence: bool,
    cohesion: dict | None = None,
) -> None:
    """The busiest neighbourhoods, named from the user's own records first."""
    tops = (cov.get("top_regions") or [])[:5]
    if not tops:
        return
    console.print("    [dim]Where yours sit[/dim] "
                  "[dim](your closest record, then the atlas caption)[/dim]")
    examples = region_examples or {}
    coh = cohesion or {}
    for r in tops:
        region = int(r["region"])
        share = float(r.get("share", 0.0))
        rows = examples.get(region) or examples.get(str(region)) or []
        if show_evidence and rows:
            snippet = str(rows[0].get("excerpt", "")).replace("\n", " ")[:72]
            console.print(f"      {share:>4.0%}  {int(r['records']):>5,}  "
                          f"[bold]{_m(snippet)}[/bold]")
            console.print(f"            [dim]atlas region {region}: "
                          f"{_m(str(r.get('terms', '')))}[/dim]")
        else:
            console.print(f"      {share:>4.0%}  {int(r['records']):>5,}  "
                          f"region {region}  [dim]{_m(str(r.get('terms', '')))}[/dim]")
        region_coh = coh.get(region, coh.get(str(region)))
        if region_coh is not None and float(region_coh) >= 0.85:
            console.print(f"            [yellow]near-identical writing here "
                          f"(cohesion {float(region_coh):.2f})[/yellow]")


def _render_categories(console: Console, cov: dict) -> None:
    from ..atlas.compare import category_labels  # noqa: PLC0415

    labels = category_labels()
    raw = cov.get("by_category") or {}
    denom = sum(int(v) for v in raw.values()) or 1
    ranked = sorted(raw.items(), key=lambda kv: -int(kv[1]))[:5]
    if not ranked:
        return
    console.print("    [dim]What you cover[/dim]")
    for cid, n in ranked:
        key = labels.get(int(cid), f"category {cid}")
        console.print(f"      {int(n) / denom:>5.0%}  {_m(key)}")


def _render_gaps(console: Console, cov: dict) -> None:
    """Subject areas the atlas knows and this corpus barely reaches.

    A histogram of your own data cannot say this. A frozen map can.
    """
    from ..atlas.compare import category_labels  # noqa: PLC0415

    gaps = cov.get("coverage_gaps") or []
    if not gaps:
        console.print("    [dim]Missing subject areas[/dim]  [green]none — "
                      "you touch every subject area the atlas covers[/green]")
        return
    labels = category_labels()
    total = int(cov.get("categories_total", 0)) or len(gaps)
    console.print(f"    [dim]Missing subject areas[/dim] [dim]({len(gaps)} of {total} "
                  f"empty or near-empty — listed, not judged)[/dim]")
    for g in gaps[:6]:
        name = labels.get(int(g["category"]), f"category {g['category']}")
        terms = str(g.get("terms") or "").strip()
        if terms:
            console.print(f"      {_m(name)}")
            console.print(f"            [dim]atlas knows this as: {_m(terms[:70])}[/dim]")
        else:
            note = f"{int(g['regions'])} regions, {int(g['records'])} records here"
            ref = g.get("reference_share")
            if ref:
                note += f"; {ref:.0%} of the reference corpus"
            console.print(f"      {_m(f'{name[:44]:<44}')} [dim]{note}[/dim]")
    if len(gaps) > 6:
        console.print(f"      [dim]and {len(gaps) - 6} more[/dim]")


def _render_dataset_signature(console: Console, cov: dict) -> None:
    """Which datasets occupy the same topical ground (text overlap cannot see this)."""
    block = cov.get("by_dataset_regions") or {}
    alike = block.get("most_alike") or []
    per = block.get("datasets") or {}
    if len(per) < 2:
        return
    console.print("    [dim]Same ground across datasets[/dim] "
                  "[dim](cosine of region histograms — text overlap cannot see this)[/dim]")
    for pair in alike[:3]:
        verdict = "same ground" if pair["similarity"] >= 0.8 else (
            "partly overlapping" if pair["similarity"] >= 0.5 else "distinct")
        console.print(f"      {pair['similarity']:>5.2f}  {_m(str(pair['a'])[:20])} vs "
                      f"{_m(str(pair['b'])[:20])}  [dim]({verdict})[/dim]")


#: How the off-atlas heading reads at each fit band. "poor" does not mean the
#: data is poor; it means the atlas describes little of it.
_FIT_NOTE = {
    "good": "the atlas describes this corpus well",
    "partial": "the atlas describes most of this corpus",
    "poor": "most of this corpus sits outside the atlas map",
}


def _render_off_atlas(
    console: Console,
    cov: dict,
    examples: list[dict] | None,
    *,
    show_evidence: bool = True,
) -> None:
    """Records unlike anything in the reference map — a finding, not a failure."""
    off = int(cov.get("off_atlas", 0))
    records = int(cov.get("records", 0)) or 1
    if not off:
        console.print("    Off the map  none")
        return

    rate = float(cov.get("off_atlas_rate", off / records))
    fit = str(cov.get("fit", ""))
    note = _FIT_NOTE.get(fit, "")
    tag = "yellow" if fit == "partial" else ("red" if fit == "poor" else "dim")
    console.print(f"    Off the map  [{tag}]{rate:.1%}[/{tag}]  {off:,} records "
                  f"[dim]({note})[/dim]")

    detail = cov.get("off_atlas_detail") or {}
    if not detail:
        return

    diagnosis = str(detail.get("diagnosis", ""))
    if diagnosis:
        console.print(f"      [dim]Why:[/dim] {_m(diagnosis)}")

    length = detail.get("length") or {}
    if length.get("off_median_chars") is not None:
        placed_len = length.get("placed_median_chars")
        line = f"      [dim]Length     off-map median {length['off_median_chars']} chars"
        if placed_len is not None:
            line += f", placed median {placed_len} chars"
        console.print(line + "[/dim]")

    score = detail.get("score") or {}
    if score.get("off_median") is not None:
        placed_med = score.get("placed_median")
        line = (f"      [dim]Distance   off-map median {score['off_median']:.2f}"
                f" similarity, cutoff {score.get('cutoff', 0):.2f}")
        if placed_med is not None:
            line += f", placed median {placed_med:.2f}"
        if score.get("near_miss_share"):
            line += f"; {score['near_miss_share']:.0%} near misses"
        console.print(line + "[/dim]")

    coh = detail.get("coherence") or {}
    surf = detail.get("surface") or {}
    if coh.get("off") is not None and coh.get("placed") is not None:
        console.print(f"      [dim]Alike      {coh['off']:.2f} within off-map set, "
                      f"{coh['placed']:.2f} within placed[/dim]")
    if surf.get("placed_whitespace") is not None:
        console.print(f"      [dim]Surface    {surf['off_whitespace']:.0%} whitespace "
                      f"and {surf['off_non_letter']:.0%} non-letter, against "
                      f"{surf['placed_whitespace']:.0%} and "
                      f"{surf['placed_non_letter']:.0%} placed[/dim]")

    near = detail.get("nearest_regions") or []
    if near:
        spread = detail.get("nearest_region_spread")
        extra = f", spread over {spread} regions" if spread else ""
        console.print(f"      [dim]Nearest atlas neighbours{extra}[/dim]")
        for r in near[:3]:
            console.print(f"        {int(r['region']):>3}  {int(r['records']):>6,}  "
                          f"[dim]{_m(str(r.get('terms', '')))}[/dim]")

    for key, label in (("by_dataset", "dataset"), ("by_language", "language")):
        groups = detail.get(key) or {}
        if len(groups) < 2:
            continue
        ranked = sorted(groups.items(), key=lambda kv: -kv[1]["rate"])[:3]
        console.print(f"      [dim]Off-map rate by {label}[/dim]")
        for name, st in ranked:
            console.print(f"        {_m(f'{str(name)[:22]:<22}')} {st['rate']:>5.0%} "
                          f"[dim]({int(st['off_atlas']):,} of "
                          f"{int(st['records']):,} records)[/dim]")

    if show_evidence and examples:
        console.print("      [dim]Furthest from the map[/dim]")
        for ex in examples[:3]:
            excerpt = str(ex.get("excerpt", "")).replace("\n", " ")[:64]
            console.print(f"        {float(ex.get('score', 0)):.2f}  "
                          f"[dim]{_m(excerpt)}[/dim]")


def _finding_order(f: Finding) -> tuple[int, str]:
    rank = {Severity.BLOCKING: 0, Severity.WARNING: 1, Severity.INFO: 2}
    return (rank.get(f.severity, 3), f.check_id)


def _detail_text(f: Finding) -> Text:
    text = Text(f.detail)
    if f.wasted_tokens:
        text.append(f"  [{f.wasted_tokens / 1e6:.2f}M tokens]", style="dim")
    if f.would_block_under and not f.is_blocking:
        text.append(f"  would block under {', '.join(f.would_block_under)}", style="dim")
    return text


def _render_evidence(console: Console, findings: list[Finding]) -> None:
    with_evidence = [f for f in findings if f.evidence]
    if not with_evidence:
        return
    console.print()
    console.print("  [bold]Examples[/bold]")
    for f in sorted(with_evidence, key=_finding_order)[:5]:
        console.print(f"    [dim]{f.check_id}[/dim] {_m(f.title)}")
        for ev in f.evidence[:2]:
            loc = f"{_short(ev.source_file)}:{ev.source_index}"
            console.print(f"      [dim]{_m(loc)}[/dim]  {_m(safe_snippet(ev.excerpt, 150))}")
            if ev.partner_excerpt:
                score = f" [{ev.score:.2f}]" if ev.score is not None else ""
                console.print(f"      [dim]matches{_m(score)}[/dim]  "
                              f"{_m(safe_snippet(ev.partner_excerpt, 150))}")


def _short(path: str, keep: int = 48) -> str:
    return path if len(path) <= keep else "…" + path[-(keep - 1):]


def _language_summary(result: ScanResult) -> str:
    for f in result.findings:
        if f.check_id == "T1-LANG-001":
            comp = f.data.get("composition", {})
            total = sum(comp.values()) or 1
            top = sorted(comp.items(), key=lambda kv: -kv[1])[:3]
            return ", ".join(f"{k} {v / total:.0%}" for k, v in top)
    return ""
