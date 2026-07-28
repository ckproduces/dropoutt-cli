"""Self-contained HTML report.

One file, no network, no CDN, no web fonts, opens from ``file://``. That matters
because the users this was built for work on clusters where standing up a server
and forwarding a port is awkward: a single file can be produced inside a batch
job and copied with ``scp`` when policy permits. By default it contains record
excerpts; ``--no-evidence`` removes excerpts and source locations.

Autoescaping is on unconditionally and every corpus string additionally passes
through :func:`safe_snippet`, because everything rendered here came from data we
did not write.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from jinja2 import Environment, StrictUndefined

from ..fingerprint import Fingerprint
from ..models import Confidence, Severity
from ..runner import ScanResult
from .escaping import safe_snippet

_CSS = """
:root{--bg:#fcfcfb;--fg:#1d1d1b;--muted:#666662;--line:#deded9;--soft:#f2f2ef;
--danger:#b42318;--warning:#8a5d00;--accent:#1769aa;--atlas:#1769aa;--empty:#d8d8d2}
@media(prefers-color-scheme:dark){:root{--bg:#151513;--fg:#e9e9e5;--muted:#aaa9a3;
--line:#353531;--soft:#20201d;--danger:#ff7b70;--warning:#e4b04d;
--accent:#76b8eb;--atlas:#76b8eb;--empty:#3d3d38}}
*{box-sizing:border-box}
html{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body{margin:0;padding:2.5rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font-family:inherit;font-size:15px;line-height:1.55}
.wrap{max-width:920px;margin:0 auto}
header{padding-bottom:1.5rem;border-bottom:1px solid var(--line)}
h1{font-size:1.55rem;font-weight:650;letter-spacing:-.02em;margin:0}
h2{font-size:1rem;font-weight:650;margin:0 0 1rem}
h3{font-size:.95rem;font-weight:650;margin:0}
p{margin:.45rem 0}
.path{color:var(--muted);font-size:.85rem;overflow-wrap:anywhere;margin-top:.2rem}
.summary{display:flex;flex-wrap:wrap;gap:.5rem 1.5rem;margin-top:1.4rem}
.summary div{min-width:92px}
.label{display:block;color:var(--muted);font-size:.72rem;text-transform:uppercase;
letter-spacing:.06em}
.value{display:block;font-size:1.05rem;font-weight:600;margin-top:.05rem}
.notice{border-left:2px solid var(--warning);padding:.1rem 0 .1rem .8rem;
color:var(--muted);font-size:.86rem;margin:1.25rem 0}
.section{padding:2rem 0;border-bottom:1px solid var(--line)}
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:1rem}
.section-note,.note{color:var(--muted);font-size:.84rem;max-width:72ch}
.finding{padding:1rem 0;border-top:1px solid var(--line)}
.finding:first-of-type{border-top:0;padding-top:0}
.finding-head{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
.check-id{font-size:.78rem;color:var(--muted)}
.count{margin-left:auto;color:var(--muted);font-size:.82rem}
.severity{display:inline-flex;align-items:center;gap:.35rem;font-size:.78rem;color:var(--muted)}
.severity::before{content:"";width:.45rem;height:.45rem;border-radius:50%;background:var(--accent)}
.severity.blocking::before{background:var(--danger)}
.severity.warning::before{background:var(--warning)}
.finding-detail{margin:.55rem 0 .25rem}
.fix{color:var(--muted);font-size:.86rem}
details{margin:.7rem 0}
summary{cursor:pointer;color:var(--muted);font-size:.84rem}
.evidence{margin:.6rem 0 0;padding:.65rem .75rem;background:var(--soft);
white-space:pre-wrap;overflow-wrap:anywhere;font-size:.82rem}
.location{display:block;color:var(--muted);font-size:.75rem;margin-bottom:.2rem;overflow-wrap:anywhere}
code,.mono{font-family:inherit;font-size:inherit}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.87rem}
th{text-align:left;color:var(--muted);font-weight:500;font-size:.72rem;text-transform:uppercase;
letter-spacing:.055em;padding:.45rem .55rem;border-bottom:1px solid var(--line)}
td{padding:.62rem .55rem;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.metrics{display:flex;flex-wrap:wrap;gap:.6rem 1.8rem;margin:1rem 0}
.metric{min-width:120px}
.atlas-map{display:block;width:100%;height:auto;margin:1rem 0 .6rem}
.atlas-point{fill:var(--empty)}
.atlas-point.occupied{fill:var(--hue,var(--atlas));fill-opacity:.78;stroke:var(--bg);stroke-width:1}
/* One hue per level-0 subject area, ordered by how much of the corpus sits in
each. Chosen for separability at 4px rather than for prettiness, and h6 is the
"empty here" grey that both the map and the legend fall back to. */
.h0{--hue:#1769aa}.h1{--hue:#0f8a6a}.h2{--hue:#a05a00}
.h3{--hue:#7a4fbf}.h4{--hue:#b0344e}.h5{--hue:#4a7a1f}.h6{--hue:var(--empty)}
.legend{display:flex;flex-wrap:wrap;gap:.5rem 1rem;color:var(--muted);font-size:.78rem}
.legend span{display:inline-flex;align-items:center;gap:.35rem}
.dot{width:.55rem;height:.55rem;border-radius:50%;background:var(--empty)}
.dot.occupied{background:var(--hue,var(--atlas))}
.coverage-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(230px,.75fr);
gap:1.75rem;margin-top:1.5rem}
.bar-row{display:grid;grid-template-columns:minmax(100px,1fr) 3fr 3.2rem;
gap:.65rem;align-items:center;margin:.55rem 0;font-size:.82rem}
.bar{height:.38rem;background:var(--soft);overflow:hidden}
.bar span{display:block;height:100%;background:var(--atlas)}
.region-list{list-style:none;margin:.35rem 0 0;padding:0}
.region-list li{display:grid;grid-template-columns:2.6rem 4.2rem 1fr;gap:.45rem;
padding:.4rem 0;border-bottom:1px solid var(--line);font-size:.82rem}
.region-list li:last-child{border-bottom:0}
.dim{color:var(--muted)}
ul.dim{padding-left:1.2rem}
footer{padding-top:1.5rem;color:var(--muted);font-size:.78rem}
@media(max-width:680px){
body{padding:1.5rem 1rem 3rem}
.coverage-grid{grid-template-columns:1fr}
.count{margin-left:0;width:100%}
th,td{padding-left:.35rem;padding-right:.35rem}
}
"""

_TEMPLATE = """<!-- generated by dropoutt {{ pipeline_version }} -->
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>dropoutt scan — {{ root_short }}</title>
<style>{{ css|safe }}</style>
<div class="wrap">
<header>
  <h1>dropoutt scan</h1>
  <div class="path">{{ root }}</div>
  <div class="summary" aria-label="Scan summary">
    <div><span class="label">Records</span><span class="value">{{ "{:,}".format(records) }}</span></div>
    <div><span class="label">Datasets</span><span class="value">{{ datasets|length }}</span></div>
    <div><span class="label">Findings</span><span class="value">{{ findings|length }}</span></div>
    <div><span class="label">Profile</span><span class="value">{{ profile }}</span></div>
  </div>
</header>

<p class="notice">
  Findings are unverified structural observations, not model-quality predictions.
  {% if not blocking_enabled %}No blocking verdict was issued because no target was declared.{% endif %}
  {% if includes_evidence %} This file contains dataset excerpts and source locations; keep it
  inside the dataset trust boundary.{% endif %}
</p>

{% if findings %}
<section class="section">
  <h2>Findings</h2>
  {% for f in findings %}
  <article class="finding">
    <div class="finding-head">
      <span class="severity {{ f.severity.value }}">{{ f.severity.value }}</span>
      <h3>{{ f.title }}</h3>
      <span class="check-id">{{ f.check_id }}</span>
      <span class="count">
        {{ "{:,}".format(f.count) if f.count else "—" }}
        {% if f.total_considered %} · {{ "%.2f%%"|format(f.rate * 100) }}{% endif %}
      </span>
    </div>
    <p class="finding-detail">{{ f.detail }}
      {% if f.wasted_tokens %}<span class="dim"> · {{ "{:,}".format(f.wasted_tokens) }} tokens</span>{% endif %}
    </p>
    <p class="fix">Fix: {{ f.fix }}
      {% if f.would_block_under and not f.is_blocking %}
      · Would block under {{ f.would_block_under|join(', ') }}
      {% endif %}
    </p>
    {% if includes_evidence and f.evidence %}
    <details>
      <summary>{{ f.evidence|length }} example{% if f.evidence|length != 1 %}s{% endif %}</summary>
      {% for ev in f.evidence %}
      <div class="evidence">
        <span class="location">{{ ev.source_file }}:{{ ev.source_index }}</span>{{ ev.excerpt }}
      </div>
      {% if ev.partner_excerpt %}
      <div class="evidence">
        <span class="location">matching record{% if ev.score %} · Jaccard {{ "%.2f"|format(ev.score) }}{% endif %}</span>{{ ev.partner_excerpt }}
      </div>
      {% endif %}
      {% endfor %}
    </details>
    {% endif %}
  </article>
  {% endfor %}
</section>
{% endif %}

{% if overlap %}
<section class="section">
  <h2>Cross-dataset overlap</h2>
  <p class="section-note">Directional: the share of “from” that also appears in “appears in”.</p>
  <div class="scroll">
  <table>
  <tr><th>from</th><th>appears in</th><th>matched</th><th>of</th><th>share</th></tr>
  {% for r in overlap %}
  <tr><td>{{ r["from"] }}</td><td>{{ r["to"] }}</td>
  <td>{{ "{:,}".format(r["matched"]) }}</td><td>{{ "{:,}".format(r["of"]) }}</td>
  <td>{{ "%.1f%%"|format(r["fraction"] * 100) }}</td></tr>
  {% endfor %}
  </table>
  </div>
</section>
{% endif %}

{% if budget %}
<section class="section">
  <h2>Token budget</h2>
  {% if not has_tokenizer %}
  <p class="section-note">Estimated from a stratified sample because no <code>--model</code> was given.</p>
  {% endif %}
  <div class="scroll">
  <table>
  <tr><th>tokenizer</th><th>total tokens</th><th>tokens per word</th><th>premium</th></tr>
  {% for e in budget %}
  <tr><td>{{ e.name }}</td><td>{{ "{:,}".format(e.total) }}</td>
  <td>{{ "%.2f"|format(e.tpw) }}</td>
  <td>{% if e.premium > 0 %}+{{ "%.0f%%"|format(e.premium * 100) }}{% else %}—{% endif %}</td></tr>
  {% endfor %}
  </table>
  </div>
</section>
{% endif %}

{% if coverage %}
<section class="section">
  <div class="section-head"><h2>Atlas coverage</h2><span class="section-note">{{ coverage.version or "" }}</span></div>
  {% if coverage.withheld %}
  <p class="note">{{ coverage.reason }}</p>
  {% else %}
  <p class="section-note">A shared coordinate system for comparing corpus coverage. It does not score quality.
  Every share below is over the {{ "{:,}".format(coverage.placed) }} records that landed on the atlas,
  out of {{ "{:,}".format(coverage.records) }} sampled.</p>
  <div class="metrics">
    <div class="metric"><span class="label">Placed</span><span class="value">{{ "{:,}".format(coverage.placed) }} / {{ "{:,}".format(coverage.records) }}</span></div>
    <div class="metric"><span class="label">Occupied</span><span class="value">{{ coverage.occupied }} / {{ coverage.total }}</span></div>
    <div class="metric"><span class="label">Effective</span><span class="value">{% if coverage.effective %}{{ "%.0f"|format(coverage.effective) }}{% else %}—{% endif %}</span></div>
    <div class="metric"><span class="label">Spread</span><span class="value">{% if coverage.concentration is not none %}{{ "%.0f%%"|format(coverage.concentration * 100) }}{% else %}—{% endif %}</span></div>
    <div class="metric"><span class="label">Off-atlas</span><span class="value">{{ "%.1f%%"|format(coverage.off_atlas_rate * 100) }}</span></div>
  </div>
  <p class="note">Occupied counts a region holding one record the same as one holding a third of the corpus.
  <strong>Effective</strong> is how many evenly-used regions this corpus is as spread out as, so the gap between
  the two is the size of the tail.</p>
  {% if coverage.points %}
  <svg class="atlas-map" viewBox="0 0 720 270" role="img" aria-labelledby="atlas-title atlas-desc">
    <title id="atlas-title">Where this corpus sits on the atlas</title>
    <desc id="atlas-desc">{{ coverage.occupied }} of {{ coverage.total }} regions are occupied. Circle size is the number of sampled records; colour is the subject area. Grey circles are regions the atlas covers and this corpus does not reach.</desc>
    <g aria-hidden="true">
    {% for p in coverage.points %}
    <circle class="atlas-point h{{ p.hue }}{% if p.records %} occupied{% endif %}" cx="{{ p.x }}" cy="{{ p.y }}" r="{{ p.radius }}">
      <title>Region {{ p.id }} — {{ p.category }}{% if p.records %}: {{ "{:,}".format(p.records) }} records{% if p.terms %}; {{ p.terms }}{% endif %}{% else %}: empty here{% endif %}</title>
    </circle>
    {% endfor %}
    </g>
  </svg>
  <div class="legend" aria-hidden="true">
    {% for c in coverage.categories[:6] %}
    <span><i class="dot occupied h{{ c.hue }}"></i>{{ c.name }}</span>
    {% endfor %}
    <span><i class="dot h6"></i>empty here</span>
  </div>
  {% endif %}
  <div class="coverage-grid">
    {% if coverage.categories %}
    <div>
      <h3>Category mix</h3>
      {% for c in coverage.categories %}
      <div class="bar-row"><span>{{ c.name }}</span><span class="bar"><span style="width:{{ "%.1f"|format(c.share * 100) }}%"></span></span><span>{{ "%.0f%%"|format(c.share * 100) }}</span></div>
      {% endfor %}
    </div>
    {% endif %}
    {% if coverage.regions %}
    <div>
      <h3>Largest regions</h3>
      <ul class="region-list">
      {% for r in coverage.regions %}
        <li><span>{{ r.id }}</span><span>{{ "{:,}".format(r.records) }}</span><span class="dim">{{ r.terms }}</span></li>
      {% endfor %}
      </ul>
    </div>
    {% endif %}
  </div>
  {% if coverage.gaps %}
  <h3>Not in this corpus</h3>
  <p class="section-note">{{ coverage.gaps|length }}{% if coverage.categories_total %} of {{ coverage.categories_total }}{% endif %}
  subject areas the atlas covers are empty or near-empty here. This is the question a histogram of your own
  data cannot answer: it takes a fixed coordinate system to tell you what is missing rather than what is present.
  Whether a gap matters depends on what you are building — it is listed, not judged.</p>
  <ul class="region-list">
  {% for g in coverage.gaps %}
    <li><span>{{ g.regions }} regions</span><span>{{ "{:,}".format(g.records) }} records</span><span class="dim">{{ g.name }}</span></li>
  {% endfor %}
  </ul>
  {% endif %}
  {% if coverage.dataset_overlap %}
  <h3>Topical overlap between datasets</h3>
  <p class="section-note">Cosine between each pair's region histograms. Two datasets sharing no wording can still
  occupy the same ground, which the near-duplicate matrix cannot see because it compares text.</p>
  <ul class="region-list">
  {% for p in coverage.dataset_overlap %}
    <li><span>{{ "%.2f"|format(p.similarity) }}</span><span>{{ p.a }}</span><span class="dim">{{ p.b }}</span></li>
  {% endfor %}
  </ul>
  {% endif %}
  {% if coverage.off_atlas.count %}
  <h3>Off-atlas records</h3>
  <p class="section-note">{{ "{:,}".format(coverage.off_atlas.count) }} records
  ({{ "%.1f%%"|format(coverage.off_atlas.rate * 100) }}) sat further from every region than the
  cutoff allows, so they are excluded from every share above.
  {% if coverage.off_atlas.diagnosis %}<strong>{{ coverage.off_atlas.diagnosis }}.</strong>{% endif %}</p>
  {% if coverage.off_atlas.score %}
  <p class="note">Median similarity {{ "%.2f"|format(coverage.off_atlas.score.off_median) }} against a cutoff of
  {{ "%.2f"|format(coverage.off_atlas.score.cutoff) }}{% if coverage.off_atlas.score.placed_median is not none %},
  where placed records median {{ "%.2f"|format(coverage.off_atlas.score.placed_median) }}{% endif %}.</p>
  {% endif %}
  <div class="coverage-grid">
    {% if coverage.off_atlas.nearest %}
    <div>
      <h3>Nearest regions, below the cutoff</h3>
      <ul class="region-list">
      {% for r in coverage.off_atlas.nearest %}
        <li><span>{{ r.region }}</span><span>{{ "{:,}".format(r.records) }}</span><span class="dim">{{ r.terms }}</span></li>
      {% endfor %}
      </ul>
    </div>
    {% endif %}
    {% for label, rows in [("language", coverage.off_atlas.by_language), ("dataset", coverage.off_atlas.by_dataset)] %}
    {% if rows|length > 1 %}
    <div>
      <h3>Off-atlas rate by {{ label }}</h3>
      {% for name, st in rows %}
      <div class="bar-row"><span>{{ name }}</span><span class="bar"><span style="width:{{ "%.1f"|format(st.rate * 100) }}%"></span></span><span>{{ "%.0f%%"|format(st.rate * 100) }}</span></div>
      {% endfor %}
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}
  <p class="note">Circle size encodes sampled record count. Label words are captions, not placement rules.
  {% if coverage.l0_accuracy %} Category labels are approximate; level-0 held-out accuracy is {{ "%.3f"|format(coverage.l0_accuracy) }}.{% endif %}
  Similarity to a region rises with record length, so a high off-atlas rate is usually a statement about
  how short the records are before it is a statement about their subject.
  Compare corpora with <code>dropoutt diff left.json right.json</code>.</p>
  {% endif %}
</section>
{% endif %}

<section class="section">
  <h2>Datasets</h2>
  <div class="scroll">
  <table>
  <tr><th>name</th><th>layout</th><th>files</th><th>records</th><th>licence</th></tr>
  {% for d in datasets %}
  <tr><td>{{ d.name }}</td><td>{{ d.layout or "—" }}</td>
  <td>{{ d.files }}</td><td>{{ "{:,}".format(d.records) }}</td>
  <td>{% if d.licence %}{{ d.licence }}{% else %}<span class="dim">not recorded</span>{% endif %}</td></tr>
  {% endfor %}
  </table>
  </div>
</section>

{% if skipped %}
<section class="section">
  <details>
    <summary>{{ skipped|length }} unavailable check{% if skipped|length != 1 %}s{% endif %}</summary>
    <div class="scroll">
    <table>
    <tr><th>check</th><th>why not</th><th>unlock</th></tr>
    {% for s in skipped %}
    <tr><td>{{ s.check_id }}<br><span class="dim">{{ s.title }}</span></td>
    <td>{{ s.reason }}</td><td class="dim">{{ s.unlock }}</td></tr>
    {% endfor %}
    </table>
    </div>
  </details>
</section>
{% endif %}

{% if degradations %}
<section class="section">
  <h2>Degraded</h2>
  <ul class="dim">{% for d in degradations %}<li>{{ d }}</li>{% endfor %}</ul>
</section>
{% endif %}

<section class="section">
  <details>
    <summary>Reproducibility metadata</summary>
    <div class="scroll">
    <table>
    {% for k, v in provenance %}
    <tr><td class="dim">{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
    </table>
    </div>
  </details>
</section>

<footer>
dropoutt {{ pipeline_version }} · fingerprint <span>{{ fingerprint_id }}</span> ·
{{ "{:,}".format(records) }} records in {{ "%.1f"|format(elapsed) }}s<br>
{% if includes_evidence %}
This report may contain excerpts of your data, including anything sensitive that the scan
found. Personal data matched by the PII check is masked, but surrounding text is not.
{% else %}
Record excerpts and source locations were omitted with <span class="mono">--no-evidence</span>.
Aggregate metadata, dataset names and the scan root remain.
{% endif %}
</footer>
</div>
"""


def _coverage_view(cov: dict | None) -> dict | None:
    """Flatten a coverage facet into what the template needs.

    Returns None when there is nothing to show, and a dict carrying only
    ``withheld`` plus a reason when coverage was suppressed. Category ids are
    resolved to names here, because "10: 59" in a shareable report is not
    information.
    """
    if not cov:
        return None

    from ..atlas.compare import (  # noqa: PLC0415
        category_labels,
        category_names,
        concentration,
        unusable_reason,
    )

    labels = category_labels()

    if cov.get("status") not in ("ok", "none placed"):
        return {"withheld": True, "reason": unusable_reason(cov),
                "version": cov.get("atlas_version")}

    off_detail = cov.get("off_atlas_detail") or {}
    off_view = {
        "count": int(cov.get("off_atlas", 0)),
        "rate": float(cov.get("off_atlas_rate", 0.0)),
        "fit": cov.get("fit"),
        "diagnosis": off_detail.get("diagnosis"),
        "score": off_detail.get("score"),
        "nearest": off_detail.get("nearest_regions") or [],
        "by_language": sorted(
            ((k, v) for k, v in (off_detail.get("by_language") or {}).items()),
            key=lambda kv: -kv[1]["rate"],
        )[:5],
        "by_dataset": sorted(
            ((k, v) for k, v in (off_detail.get("by_dataset") or {}).items()),
            key=lambda kv: -kv[1]["rate"],
        )[:5],
    }

    if cov.get("status") == "none placed":
        return {"withheld": False, "none_placed": True,
                "version": cov.get("atlas_version"),
                "records": int(cov.get("records", 0)),
                "placed": 0, "off_atlas": off_view,
                "off_atlas_rate": float(cov.get("off_atlas_rate", 0.0)),
                "occupied": 0, "total": cov.get("regions_total", 0),
                "concentration": None, "l0_accuracy": None,
                "points": [], "categories": [], "regions": []}

    names = category_names()
    raw = cov.get("by_category") or {}
    placed = sum(int(v) for v in raw.values()) or 1
    counts = {int(region): int(count) for region, count in
              (cov.get("region_counts") or {}).items()}
    points: list[dict[str, Any]] = []
    try:
        from ..atlas import load_bundled  # noqa: PLC0415

        atlas = load_bundled()
        if atlas is not None and atlas.meta.get("version") == cov.get("atlas_version"):
            xs = atlas.coords[:, 0]
            ys = atlas.coords[:, 1]
            x_span = float(xs.max() - xs.min()) or 1.0
            y_span = float(ys.max() - ys.min()) or 1.0
            max_count = max(counts.values(), default=1)
            # Occupied points are coloured by their level-0 subject area rather
            # than all rendered in one accent. A single-colour map answers only
            # "did anything land here", which the occupancy metric already says;
            # colouring by category turns the same picture into a statement
            # about which parts of the atlas this corpus lives in.
            ranked_cats = [int(c) for c, _ in
                           sorted(raw.items(), key=lambda kv: -int(kv[1]))]
            palette = {cid: i for i, cid in enumerate(ranked_cats[:6])}
            for region in range(atlas.n_regions):
                count = counts.get(region, 0)
                radius = 1.8 if not count else 3.0 + 8.0 * (count / max_count) ** 0.5
                cid = int(atlas.region_category[region])
                points.append({
                    "id": region,
                    "x": round(18.0 + 684.0 * float(xs[region] - xs.min()) / x_span, 1),
                    "y": round(18.0 + 234.0 * float(ys.max() - ys[region]) / y_span, 1),
                    "radius": round(radius, 1),
                    "records": count,
                    "hue": palette.get(cid, 6) if count else 6,
                    "category": names.get(cid, f"category {cid}"),
                    "terms": (
                        atlas.region_terms[region]
                        if region < len(atlas.region_terms) else ""
                    ),
                })
    except Exception:
        # The aggregate coverage remains useful if a report is rendered without
        # the matching atlas artifact. Omit the map rather than invent geometry.
        points = []
    return {
        "withheld": False,
        "none_placed": False,
        "version": cov.get("atlas_version"),
        "occupied": cov.get("regions_occupied", 0),
        "total": cov.get("regions_total", 0),
        "records": int(cov.get("records", 0)),
        "placed": int(cov.get("placed", placed)),
        "off_atlas_rate": cov.get("off_atlas_rate", 0.0),
        "off_atlas": off_view,
        "concentration": concentration(cov),
        "l0_accuracy": cov.get("l0_holdout_accuracy"),
        "points": points,
        "effective": cov.get("effective_regions"),
        "categories": [
            {"name": names.get(int(cid), f"category {cid}"),
             "share": int(n) / placed,
             "hue": i if i < 6 else 6}
            for i, (cid, n) in enumerate(
                sorted(raw.items(), key=lambda kv: -int(kv[1]))[:8])
        ],
        "categories_total": int(cov.get("categories_total", 0)),
        # What the atlas covers and this corpus does not. The one question a
        # histogram of your own data can never answer.
        "gaps": [
            {"name": labels.get(int(g["category"]), f"category {g['category']}"),
             "regions": int(g["regions"]), "records": int(g["records"])}
            for g in (cov.get("coverage_gaps") or [])[:10]
        ],
        "dataset_overlap": (cov.get("by_dataset_regions") or {}).get("most_alike", [])[:5],
        "regions": [
            {"id": int(r["region"]), "records": int(r["records"]),
             "share": float(r.get("share", 0.0)),
             "terms": str(r.get("terms", ""))}
            for r in (cov.get("top_regions") or [])[:10]
        ],
    }


def render(
    result: ScanResult,
    fp: Fingerprint,
    budget: Any = None,
    *,
    include_evidence: bool = True,
) -> str:
    env = Environment(autoescape=True, undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(_TEMPLATE)

    findings = deepcopy(sorted(
        result.findings,
        key=lambda f: ({Severity.BLOCKING: 0, Severity.WARNING: 1, Severity.INFO: 2}
                       .get(f.severity, 3), f.check_id),
    ))
    # Every excerpt goes through safe_snippet before Jinja sees it.
    for f in findings:
        if not include_evidence:
            f.evidence = []
            continue
        for ev in f.evidence:
            ev.excerpt = safe_snippet(ev.excerpt)
            if ev.partner_excerpt:
                ev.partner_excerpt = safe_snippet(ev.partner_excerpt)

    overlap = []
    for f in findings:
        if f.check_id == "T1-OVERLAP-001":
            overlap = f.data.get("matrix", [])[:40]

    budget_rows = []
    if budget is not None and budget.estimates:
        for e in sorted(budget.estimates, key=lambda x: x.total_tokens_est):
            if e.failed:
                continue
            budget_rows.append({
                "name": e.name,
                "total": e.total_tokens_est,
                "tpw": e.tokens_per_word,
                "premium": budget.premium_vs_cheapest(e),
            })

    seen: set[str] = set()
    skipped = []
    for s in result.skipped:
        if s.reason in seen:
            continue
        seen.add(s.reason)
        skipped.append(s)

    root = result.ctx.root
    return tpl.render(
        css=_CSS,
        includes_evidence=include_evidence,
        coverage=_coverage_view(result.ctx.stats.get("atlas_coverage")),
        root=root,
        root_short=root.split("/")[-1] or root,
        records=result.records_scanned,
        datasets=fp.datasets,
        findings=findings,
        overlap=overlap,
        budget=budget_rows,
        has_tokenizer=result.ctx.tokenizer is not None,
        skipped=skipped,
        degradations=result.ctx.degradations,
        profile=result.ctx.profile.value,
        blocking_enabled=result.ctx.blocking_enabled,
        provenance=sorted(fp.provenance.items()),
        fingerprint_id=fp.fingerprint_id,
        pipeline_version=fp.pipeline_version,
        elapsed=result.elapsed,
    )
