"""The dropoutt design system, as far as one offline HTML file can carry it.

The web product's tokens live in ``frontend/app/app/tokens/index.css`` and are
the source of truth: the primitive scales here are copied from it by hand and
should be re-copied rather than adjusted, so a report opened next to the app
does not look like a different product. What is *not* copied is the component
layer — there is no React here, and a stylesheet that inlines twenty component
modules to render six of them is worse than one that spells out six.

Three things are deliberately different from the web build.

No web fonts. The report has to open from ``file://`` on a machine with no
network — that is the whole reason it is one file — so the type stack falls
through to the system UI font. Inter is named first because a developer machine
often has it, and the metrics of the fallbacks are close enough that the layout
does not move.

One theme. The page used to answer to ``prefers-color-scheme``, which meant the
same file was two different documents depending on whose laptop opened it — and
the one thing a report is for is being forwarded. The density grid made that
concrete: its scale runs from an empty cell to a green one through a blue, and
a ramp tuned to read correctly on white does not read correctly on near-black.
It is light, always, on screen and on paper.

Print is a first-class target. A dataset report gets attached to a review, so
the print rules are here rather than bolted on: cards keep their borders and
lose their shadows, and nothing that carries information is hidden.
"""

from __future__ import annotations

from itertools import pairwise

#: Primitives, copied from the design system. Only the steps the report uses.
TOKENS = """
:root{
  color-scheme:light;
  --neutral-000:#fcfeff; --neutral-100:#f9fafc; --neutral-200:#f4f6f8;
  --neutral-300:#eff0f2; --neutral-400:#e5e6e8; --neutral-500:#d4d6d8;
  --neutral-600:#b6b7b9; --neutral-700:#97989a; --neutral-800:#797a7c;
  --neutral-900:#616467; --neutral-1000:#4f5154; --neutral-1100:#404345;
  --neutral-1200:#34373a; --neutral-1300:#2a2d2f; --neutral-1400:#222427;
  --neutral-1500:#1a1c1e; --neutral-1600:#131518; --neutral-1700:#0c0f11;
  --neutral-1900:#040507; --white:#ffffff;

  --brand-200:#004a9e; --brand-300:#005ebc; --brand-400:#0d70d4;
  --brand-500:#257fe5; --brand-600:#4c99f9; --brand-800:#b1d4ff;
  --brand-900:#d7eaff; --brand-1000:#f1f9ff;

  --green-300:#0c7521; --green-400:#228830; --green-500:#36973f;
  --green-800:#b7ddb7; --green-900:#daeeda; --green-1000:#f2fbf2;
  --orange-300:#ad3d00; --orange-400:#c44e00; --orange-500:#d55e00;
  --orange-800:#fdc7a9; --orange-900:#ffe3d3; --orange-1000:#fff5ee;
  --red-300:#b21235; --red-400:#ca2844; --red-500:#dc3c52;
  --red-800:#ffbcbd; --red-900:#ffdddd; --red-1000:#fff4f3;
  --teal-400:#008ca1; --teal-500:#009cb1; --teal-900:#d8f0f5; --teal-1000:#f1fdff;
  --pink-400:#984ea9; --pink-500:#a85db9;
  --yellow-300:#b17b00; --yellow-500:#e7ad00;

  --font: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
  --w-regular:400; --w-medium:500; --w-semibold:550; --w-bold:650;

  --s-1:1px; --s-2:2px; --s-4:4px; --s-6:6px; --s-8:8px; --s-10:10px;
  --s-12:12px; --s-16:16px; --s-20:20px; --s-24:24px; --s-32:32px;
  --s-40:40px; --s-48:48px; --s-64:64px; --s-80:80px;

  --r-sm:2px; --r-md:4px; --r-lg:6px; --r-xl:8px; --r-2xl:12px; --r-full:99999px;

  --t-11:11px; --t-12:12px; --t-13:13px; --t-14:14px; --t-15:15px; --t-16:16px;
  --t-18:18px; --t-20:20px; --t-24:24px; --t-28:28px; --t-36:36px;
  --lh-tight:1.2; --lh-snug:1.3; --lh-normal:1.5;
  --ls-tighter:-0.02em; --ls-tight:-0.01em; --ls-wide:0.02em;

  /* Semantics */
  --bg-canvas:var(--white); --bg-surface:var(--white); --bg-subtle:var(--neutral-100);
  --bg-inset:var(--neutral-200);
  --text:var(--neutral-1900); --text-muted:var(--neutral-1000);
  --text-faint:var(--neutral-800);
  --border:var(--neutral-400); --border-subtle:var(--neutral-300);
  --accent:var(--brand-400); --accent-soft:var(--brand-1000);
  --accent-border:var(--brand-900);
  --ok:var(--green-400); --ok-soft:var(--green-1000); --ok-border:var(--green-900);
  --warn:var(--orange-400); --warn-soft:var(--orange-1000); --warn-border:var(--orange-900);
  --bad:var(--red-400); --bad-soft:var(--red-1000); --bad-border:var(--red-900);
  --info:var(--teal-400); --info-soft:var(--teal-1000);
  --shadow-sm:0 1px 3px color-mix(in srgb, black 6%, transparent);
  --shadow-md:0 4px 16px color-mix(in srgb, black 10%, transparent);

  /* Chart series. Distinct at bar width first, pretty second. */
  --c0:var(--brand-500); --c1:var(--teal-500); --c2:var(--pink-500);
  --c3:var(--green-500); --c4:var(--yellow-500); --c5:var(--orange-500);
  --c6:var(--neutral-700);
}


"""

LAYOUT = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%}
body{background:var(--bg-canvas);color:var(--text);font-family:var(--font);
font-size:var(--t-15);line-height:var(--lh-normal);
-webkit-font-smoothing:antialiased;padding:0 var(--s-24) var(--s-80)}
.wrap{max-width:1060px;margin:0 auto}
a{color:var(--accent);text-underline-offset:2px}
code,.mono{font-family:var(--mono);font-size:.92em}
.num{font-variant-numeric:tabular-nums}
.muted{color:var(--text-muted)}
.faint{color:var(--text-faint)}
.t12{font-size:var(--t-12)}.t13{font-size:var(--t-13)}.t14{font-size:var(--t-14)}
h1,h2,h3,h4{font-weight:var(--w-medium);line-height:var(--lh-tight);
letter-spacing:var(--ls-tight)}
h2{font-size:var(--t-24);letter-spacing:var(--ls-tighter)}
h3{font-size:var(--t-16);font-weight:var(--w-semibold);letter-spacing:0}
h4{font-size:var(--t-13);font-weight:var(--w-semibold);letter-spacing:0}
p{margin:var(--s-8) 0}
.prose{max-width:74ch}

/* -- masthead --
   What the document is over whose product made it, and the folder it is about
   on the far side of the same line. The path is the one thing here that
   identifies the run, so it gets its own column rather than a third line: at
   the foot of a stack it read as a caption on the logo. */
.masthead{display:flex;align-items:flex-end;justify-content:space-between;
flex-wrap:wrap;gap:var(--s-12) var(--s-24);padding:var(--s-40) 0 var(--s-24);
border-bottom:1px solid var(--border-subtle)}
.masthead .brand{display:flex;flex-direction:column;align-items:flex-start;
gap:var(--s-8)}
.masthead .kind{color:var(--text-muted);font-size:var(--t-13);
font-weight:var(--w-medium);letter-spacing:var(--ls-wide);text-transform:uppercase}
.mark{display:block;height:34px;width:auto;color:var(--text)}
.masthead .where{color:var(--text-faint);font-size:var(--t-12);
overflow-wrap:anywhere;font-family:var(--mono);text-align:right;
max-width:52ch}

/* -- verdict strip --
   It sits at the head of the findings rather than at the head of the page. The
   sentence it carries is a count of what is below it and a name for the worst
   of them, which is a caption for that list; above the composition section it
   was a verdict on a corpus the reader had not been shown yet. */
.verdict{display:flex;align-items:center;gap:var(--s-12);flex-wrap:wrap;
padding:var(--s-16) var(--s-20);margin-bottom:var(--s-16);
border-radius:var(--r-2xl);
border:1px solid var(--border-subtle);background:var(--bg-subtle)}
.verdict.block{background:var(--bad-soft);border-color:var(--bad-border)}
.verdict.warn{background:var(--warn-soft);border-color:var(--warn-border)}
.verdict.clean{background:var(--ok-soft);border-color:var(--ok-border)}
.verdict .dot{width:8px;height:8px;border-radius:var(--r-full);flex:none;
background:var(--ok)}
.verdict.block .dot{background:var(--bad)}.verdict.warn .dot{background:var(--warn)}
.verdict .headline{font-weight:var(--w-semibold);font-size:var(--t-16)}
.verdict .lead{color:var(--text-muted);font-size:var(--t-14);flex-basis:100%}

/* -- sections --
   The number is part of the heading rather than a marginal note beside it. Set
   small it read as a footnote marker and was skipped; set at heading size it
   does the one thing a section number is for, which is telling a reader who
   scrolled past something how far past it they are.

   The gap below the heading belongs to the heading. It used to belong to the
   standfirst paragraph, so the one section without a standfirst had its card
   welded to its title. */
section{margin-top:var(--s-64)}
.sechead{display:flex;align-items:baseline;gap:var(--s-12);
margin-bottom:var(--s-24)}
.sechead h2{display:flex;align-items:baseline;gap:var(--s-12)}
.sechead .n{color:var(--text-faint);font-variant-numeric:tabular-nums;
font-weight:var(--w-medium)}

/* -- cards -- */
.card{background:var(--bg-surface);border:1px solid var(--border-subtle);
border-radius:var(--r-2xl);padding:var(--s-20);box-shadow:var(--shadow-sm)}
/* A heading owns the space under it wherever it appears. Tables and grids
   carry no top margin of their own, so without this the first header row sits
   on the baseline of the title above it. */
.card>h3+*,.card>h4+*{margin-top:var(--s-12)}
.grid{display:grid;gap:var(--s-12)}
.g2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(168px,1fr))}
.stat .k{display:block;color:var(--text-muted);font-size:var(--t-13);
font-weight:var(--w-medium)}
.stat .v{display:block;font-size:var(--t-28);font-weight:var(--w-medium);
letter-spacing:var(--ls-tight);margin-top:var(--s-8);font-variant-numeric:tabular-nums}
.stat .n{display:block;color:var(--text-muted);font-size:var(--t-13);margin-top:var(--s-4)}

/* -- badges -- */
.badge{display:inline-flex;align-items:center;gap:var(--s-4);padding:var(--s-1) var(--s-10);
border-radius:var(--r-full);font-size:var(--t-11);font-weight:var(--w-medium);
line-height:18px;white-space:nowrap;background:var(--bg-inset);color:var(--text-muted)}
.badge.bad{background:var(--bad-soft);color:var(--bad)}
.badge.warn{background:var(--warn-soft);color:var(--warn)}
.badge.ok{background:var(--ok-soft);color:var(--ok)}
.badge.info{background:var(--accent-soft);color:var(--accent)}

/* -- bars -- */
.bars{display:grid;gap:var(--s-10);margin-top:var(--s-12)}
/* The label column is sized for the atlas taxonomy, whose longest name is
   "Turkish literature, idiom and culture". Anything narrower truncates every
   subject to an ellipsis and the chart stops being readable. */
.bar{display:grid;grid-template-columns:minmax(84px,17rem) 1fr auto;gap:var(--s-12);
align-items:center;font-size:var(--t-13)}
.bar .label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar .track{height:6px;background:var(--bg-inset);border-radius:var(--r-full);
position:relative;min-width:40px}
.bar .fill{position:absolute;inset:0 auto 0 0;border-radius:var(--r-full);
background:var(--hue,var(--c0))}
.bar .ref{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--text-faint);
border-radius:var(--r-sm)}
.bar .v{color:var(--text-muted);font-variant-numeric:tabular-nums;min-width:3.2rem;
text-align:right}
.h0{--hue:var(--c0)}.h1{--hue:var(--c1)}.h2{--hue:var(--c2)}
.h3{--hue:var(--c3)}.h4{--hue:var(--c4)}.h5{--hue:var(--c5)}.h6{--hue:var(--c6)}

/* -- findings -- */
.finding{background:var(--bg-surface);border:1px solid var(--border-subtle);
border-left:3px solid var(--border);border-radius:var(--r-xl);padding:var(--s-16) var(--s-20);
margin-bottom:var(--s-12);box-shadow:var(--shadow-sm)}
.finding.bad{border-left-color:var(--bad)}
.finding.warn{border-left-color:var(--warn)}
.finding.info{border-left-color:var(--info)}
.ftop{display:flex;align-items:center;gap:var(--s-8);flex-wrap:wrap}
.ftop h3{flex:1 1 auto;min-width:0}
.fid{color:var(--text-faint);font-size:var(--t-11);font-family:var(--mono)}
.fscale{margin-top:var(--s-8);font-size:var(--t-15);font-weight:var(--w-medium);
font-variant-numeric:tabular-nums}
.fscale .cost{color:var(--bad)}
.fscale .sep{color:var(--text-faint);font-weight:var(--w-regular);padding:0 var(--s-6)}
.fdetail{color:var(--text-muted);font-size:var(--t-14);margin-top:var(--s-6);max-width:74ch}
.ffix{margin-top:var(--s-12);padding:var(--s-10) var(--s-12);background:var(--bg-subtle);
border-radius:var(--r-lg);font-size:var(--t-14)}
.ffix b{font-weight:var(--w-semibold)}
details{margin-top:var(--s-12)}
summary{cursor:pointer;color:var(--text-muted);font-size:var(--t-13);
font-weight:var(--w-medium)}
summary::marker{color:var(--text-faint)}
.excerpt{margin-top:var(--s-8);padding:var(--s-10) var(--s-12);background:var(--bg-inset);
border-radius:var(--r-lg);font-size:var(--t-13);line-height:var(--lh-normal);
white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--mono)}
.excerpt .loc{display:block;color:var(--text-faint);font-size:var(--t-11);
margin-bottom:var(--s-4);overflow-wrap:anywhere}
.allclear{border:1px solid var(--ok-border);background:var(--ok-soft);
border-radius:var(--r-2xl);padding:var(--s-20)}

/* -- insights --
   Each of these headlines is about a different quantity: a share of the corpus,
   a share of the map, a similarity between two datasets. Five of them stacked
   under one title read as one undifferentiated list, so each carries a badge
   naming which kind of claim it is, and the tinted panel keeps the group from
   dissolving into the card around it. */
.insights{display:grid;gap:var(--s-10)}
.insight{display:grid;grid-template-columns:3px 1fr;gap:var(--s-16);
padding:var(--s-16);border:1px solid var(--border-subtle);
border-radius:var(--r-xl);background:var(--bg-subtle)}
.insight .rule{border-radius:var(--r-full);background:var(--accent)}
.insight.warn .rule{background:var(--warn)}
.insight.warn{background:var(--warn-soft);border-color:var(--warn-border)}
.insight h3{margin:var(--s-6) 0 var(--s-4)}
.insight p{color:var(--text-muted);font-size:var(--t-14);margin:0;max-width:74ch}
.insight .quote{display:block;margin-top:var(--s-10);padding-left:var(--s-12);
border-left:2px solid var(--border);color:var(--text-faint);font-size:var(--t-13);
overflow-wrap:anywhere}

/* -- place lists -- */
.places{list-style:none}
.places li{display:grid;grid-template-columns:3.6rem 1fr;gap:var(--s-12);
padding:var(--s-12) 0;border-top:1px solid var(--border-subtle);align-items:start}
.places li:first-child{border-top:0;padding-top:var(--s-4)}
.places .pct{font-weight:var(--w-semibold);font-variant-numeric:tabular-nums;
text-align:right;font-size:var(--t-14)}
.places .yours{display:block;font-size:var(--t-14);overflow-wrap:anywhere}
.places .cap{display:block;color:var(--text-faint);font-size:var(--t-12);
margin-top:var(--s-4)}
.places .flag{display:block;color:var(--warn);font-size:var(--t-12);margin-top:var(--s-4)}

/* -- density grid --
   The map itself, drawn once: a row per subject area, a chip per fine cell,
   coloured *and labelled* by how densely this corpus sits in that cell against
   how densely the reference corpus does. Everything else in the section is a
   share of the corpus, which cannot answer "where is my data, against
   everything else".

   It is a real table, and that is the point. The header repeats across printed
   pages for free, which no flex column does; the ratio is in the cell rather
   than in a tooltip, so it survives paper, a screenshot pasted into a ticket,
   and reading by keyboard. Both of those were paid for by dropping the sort
   control, which could only ever have been CSS `order` on flex children. */
.agrid{width:100%;border-collapse:collapse;table-layout:auto;
font-size:var(--t-13)}
.agrid thead th{text-align:left;color:var(--text-faint);
font-weight:var(--w-semibold);font-size:var(--t-11);
letter-spacing:var(--ls-wide);text-transform:uppercase;white-space:nowrap;
padding:var(--s-8) 0;border-bottom:1px solid var(--border)}
.agrid td{padding:var(--s-4) 0;border-bottom:1px solid var(--border-subtle);
vertical-align:middle}
.agrid th.r,.agrid td.r{text-align:right;font-variant-numeric:tabular-nums;
padding-left:var(--s-16);white-space:nowrap}
.agrid td.r{font-weight:var(--w-medium)}
.agrid td.reach{color:var(--text-muted);font-weight:var(--w-regular)}
.agrid .reach-ok{color:var(--ok);font-size:var(--t-12)}
.agrid .reach-miss{color:var(--text-faint);font-size:var(--t-12)}
.agrid tr.empty td{color:var(--text-faint);font-weight:var(--w-regular)}
.imbalances{list-style:none}
.imbalances li{display:grid;grid-template-columns:4.2rem 1fr;gap:var(--s-12);
padding:var(--s-12) 0;border-top:1px solid var(--border-subtle);align-items:start}
.imbalances li:first-child{border-top:0;padding-top:var(--s-4)}
.imbalances .pct{font-weight:var(--w-semibold);font-variant-numeric:tabular-nums;
text-align:right;font-size:var(--t-14)}
.imbalances .pct.cut{color:var(--bad)}
.imbalances .pct.grow{color:var(--ok)}
.imbalances .yours{display:block;font-size:var(--t-14);overflow-wrap:anywhere}
.imbalances .cap{display:block;color:var(--text-faint);font-size:var(--t-12);
margin-top:var(--s-4)}
.imbalances .act{display:inline-block;margin-top:var(--s-4);font-size:var(--t-12);
font-weight:var(--w-medium)}
.imbalances .act.cut{color:var(--bad)}
.imbalances .act.grow{color:var(--ok)}

/* Name and cells are one column: the name is what the row of chips is about,
   and a gutter between them only invited the eye to read them as two tables. */
.area{display:grid;grid-template-columns:minmax(0,20rem) 1fr;
gap:var(--s-16);align-items:center}
.area .aname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.acells{display:flex;gap:var(--s-4);align-items:center;flex-wrap:wrap}

/* One fine cell, carrying its own ratio. Bold, because the label sits on a
   saturated fill at 11px and regular weight is the difference between a number
   and a smudge. The ink is chosen per step by measured contrast — see _ink.
   Every cell carries an inset border so white and near-white fills still read
   as boxes rather than empty gaps in the row. */
.cell{display:inline-flex;align-items:center;justify-content:center;
min-width:2.7rem;height:20px;padding:0 var(--s-4);flex:none;
border-radius:var(--r-md);background:var(--cell,var(--bg-inset));
box-shadow:inset 0 0 0 1px var(--neutral-500);
color:var(--ink,var(--text));font-size:var(--t-11);
font-weight:var(--w-bold);font-variant-numeric:tabular-nums;
letter-spacing:0;text-transform:none}
/* Never reached: white fill, same border, a zero instead of a blank. */
.cell:not(.on){background:var(--white);color:var(--text-faint);
font-weight:var(--w-medium)}

/* The scale is continuous, so the legend is a gradient rather than four
   swatches the reader has to interpolate between by eye. */
.legend{display:grid;grid-template-columns:1fr;gap:var(--s-6);
margin-top:var(--s-32);color:var(--text-muted);font-size:var(--t-12)}
.legend .ramp{height:10px;border-radius:var(--r-full);
border:1px solid var(--border-subtle)}
.legend .ends{display:flex;justify-content:space-between;gap:var(--s-12)}
.legend .mid{color:var(--text-faint)}

/* -- tables -- */
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:var(--t-13)}
th{text-align:left;color:var(--text-faint);font-weight:var(--w-semibold);
font-size:var(--t-11);letter-spacing:var(--ls-wide);text-transform:uppercase;
padding:var(--s-8) var(--s-10);border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:var(--s-10);border-bottom:1px solid var(--border-subtle);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:var(--s-6);margin-top:var(--s-8)}

footer{margin-top:var(--s-64);padding-top:var(--s-20);
border-top:1px solid var(--border-subtle);color:var(--text-faint);
font-size:var(--t-12);max-width:80ch}
footer p{margin:var(--s-6) 0}

/* -- narrow screens --
   The report is a file, and a file gets sent to someone. Where it is opened is
   not where it was written, so a phone is a first-class target and not a
   courtesy. Nothing is hidden at any width: wide content scrolls inside its own
   box, and everything else reflows. */
@media (max-width:900px){
  .masthead{padding-top:var(--s-32)}
  section{margin-top:var(--s-48)}
}

@media (max-width:640px){
  body{padding:0 var(--s-16) var(--s-48);font-size:var(--t-14)}
  section{margin-top:var(--s-40)}
  h2{font-size:var(--t-20)}
  .masthead{padding:var(--s-24) 0 var(--s-16);gap:var(--s-6)}
  .mark{height:28px}
  .verdict{padding:var(--s-12) var(--s-16);margin-top:var(--s-16)}
  .card{padding:var(--s-16)}
  .stat .v{font-size:var(--t-24)}
  .finding{padding:var(--s-12) var(--s-16)}
  .ftop{gap:var(--s-6)}
  .ftop h3{flex-basis:100%;order:3}
  /* The label goes above the bar rather than beside it. A seven-rem column
     truncates "Tool calls and agentic trajectories" to three words, and the
     name is the part worth reading. */
  .bar{grid-template-columns:1fr auto;gap:var(--s-4) var(--s-8)}
  .bar .label{grid-column:1/-1;white-space:normal;overflow:visible}
  .bar .v{min-width:2.8rem}
  .places li{grid-template-columns:2.8rem 1fr;gap:var(--s-8)}
  .insight{gap:var(--s-10);padding:var(--s-12)}
  /* The area name takes the full width and its cells wrap underneath. A
     twenty-rem name column beside a row of chips leaves three words of a
     subject label, and the label is what the row is about. */
  .area{grid-template-columns:1fr;gap:var(--s-6)}
  .agrid td{padding:var(--s-8) 0}
  /* A dataset name or an install hint is one long token; on a 320px screen it
     has to be allowed to break rather than push the page sideways. */
  code,.mono{overflow-wrap:anywhere}
  .badge{white-space:normal;text-align:left}
}

@media (max-width:400px){
  body{padding-inline:var(--s-12)}
  .g4{grid-template-columns:1fr}
  .stat .v{font-size:var(--t-20)}
}

@media print{
  body{background:#fff;color:#000;font-size:10.5pt;padding:0}
  .card,.finding,.verdict,.allclear{box-shadow:none;break-inside:avoid}
  section{break-before:auto;margin-top:22pt}
  .sechead,h3{break-after:avoid}
  details{display:block}
  details>summary{display:none}
  .insight,.places li,tr{break-inside:avoid}
  a{color:#000;text-decoration:none}
  .masthead .where{font-size:8pt}
  /* The grid is 48 rows and will cross a page boundary. A table header repeats
     itself across pages when it is told to; without this the second page is an
     unlabelled block of colour. Every square carries its own ratio, so nothing
     on this page depends on a pointer. */
  .agrid thead{display:table-header-group}
  .cell{print-color-adjust:exact;-webkit-print-color-adjust:exact}
}
"""

#: The ramp's anchor colours, as (level, hex). Copied from the token block
#: above and has to move with it. They are literals rather than
#: ``color-mix(var(--green-500) …)`` because each cell carries a number, and the
#: only way to know whether that number needs dark or light ink is to compute
#: the fill's luminance here rather than hand it to the browser.
#:
#: Four stops, and they are the traffic light this data already implies: green
#: is parity — as common in your data as on the map — then yellow, then red for
#: the most over-represented cell in the corpus. A single-hue ramp says "more"
#: and this says "more, and past a point that is a finding".
RAMP_ANCHORS = (
    (0.0, "#ffffff"),   # --white: reached, but nowhere near the map's density
    (1.0, "#36973f"),   # --green-500: as common here as on the map
    (2.0, "#e7ad00"),   # --yellow-500
    (3.0, "#dc3c52"),   # --red-500: the densest cell in this corpus
)

#: Steps per level unit across the whole ramp. Thirty-two keeps low-coverage
#: cells on a smooth white→green climb and gives saturated fills enough stops
#: that neighbouring ratios stay distinguishable.
RAMP_DIVISOR = 32
RAMP_STEPS = int(RAMP_ANCHORS[-1][0] * RAMP_DIVISOR)

#: Text colours the ink chooses between. Pure black and white.
INK_DARK = "#000000"
INK_LIGHT = "#ffffff"

#: Fills at or below this luminance get white ink. Pure WCAG contrast would
#: still pick black on green-500 and red-500 (they clear 4.5:1), but at 11px
#: on a saturated chip the dark digits smear; white is the readable choice.
#: Yellow-500 sits above the cut and keeps black.
INK_LIGHT_LUMINANCE = 0.40


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


def _luminance(colour: str) -> float:
    """WCAG relative luminance of a ``#rrggbb`` string."""
    channels = []
    for i in (1, 3, 5):
        c = int(colour[i:i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    lo, hi = min(la, lb), max(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _ink(fill: str) -> tuple[str, float]:
    """Black or white text on this fill, by relative luminance.

    White below :data:`INK_LIGHT_LUMINANCE`, black above. Returns the chosen
    ink and its WCAG contrast against the fill.
    """
    luminance = _luminance(fill)
    if luminance <= INK_LIGHT_LUMINANCE:
        return INK_LIGHT, 1.05 / (luminance + 0.05)
    return INK_DARK, (luminance + 0.05) / 0.05


def ramp_contrast() -> list[tuple[int, float]]:
    """Every step's chosen ink and the contrast it achieves. For the tests."""
    return [
        (step, _ink(_fill(step / RAMP_DIVISOR))[1])
        for step in range(RAMP_STEPS + 1)
    ]


def _fill(level: float) -> str:
    for (lo, low), (hi, high) in pairwise(RAMP_ANCHORS):
        if level <= hi or hi == RAMP_ANCHORS[-1][0]:
            return _mix(low, high, (level - lo) / (hi - lo) if hi > lo else 0.0)
    return RAMP_ANCHORS[-1][1]


def gradient() -> str:
    """The whole ramp as one CSS gradient, for the legend.

    A legend of four labelled swatches asked the reader to interpolate between
    them by eye. The scale is continuous, so the legend is too.
    """
    stops = ", ".join(
        f"{_fill(level)} {level / RAMP_ANCHORS[-1][0] * 100:.0f}%"
        for level, _ in RAMP_ANCHORS
    )
    return f"linear-gradient(90deg, {stops})"


def _ramp() -> str:
    """The density scale, as one class per quantised step."""
    rules = []
    for step in range(RAMP_STEPS + 1):
        fill = _fill(step / RAMP_DIVISOR)
        ink, _ = _ink(fill)
        rules.append(f".d{step}{{--cell:{fill};--ink:{ink};color:{ink}}}")
    return "\n".join(rules) + "\n"


def stylesheet() -> str:
    return TOKENS + LAYOUT + _ramp()


#: The dropoutt wordmark, from the product's own text-logo.svg, recoloured to
#: currentColor. The full lockup rather than the icon plus a `<span>dropoutt`:
#: the letterforms are drawn, not set, and a system font beside them was the
#: one place the report visibly stopped being the same product as the app.
TEXT_LOGO_SVG = (
    "<svg class=\"mark\" fill=\"currentColor\" role=\"img\" aria-label=\"dropoutt\" "
    "viewBox=\"0 0 693 204\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M54.9793"
    " 95.3351C58.4322 91.8823 64.0304 91.8822 67.4832 95.3351C70.9361 98.7879 "
    "70.936 104.386 67.4832 107.839C64.0304 111.292 58.4321 111.292 54.9793 "
    "107.839C51.5264 104.386 51.5264 98.788 54.9793 95.3351Z\"/><path "
    "fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M68.9488 49.5635C81.2257 "
    "37.2867 101.13 37.2868 113.407 49.5637C125.684 61.8406 125.684 81.7452 "
    "113.407 94.0221L105.766 101.663L113.407 109.305C125.684 121.581 125.684 "
    "141.486 113.407 153.763C101.226 165.944 81.5363 166.039 69.2379 "
    "154.049L68.9488 153.763L67.5922 152.407L76.2061 143.793L77.5627 "
    "145.149C85.0823 152.669 97.2739 152.669 104.793 145.149C112.313 137.63 "
    "112.313 125.438 104.793 117.918L97.1522 110.277L53.6663 153.763C41.3894 "
    "166.04 21.4846 166.04 9.2077 153.763C-2.97322 141.582 -3.06851 121.892 "
    "8.922 109.594L9.20751 109.305L16.8488 101.663L9.2077 94.0221C-3.06918 "
    "81.7452 -3.06918 61.8404 9.2077 49.5635C21.4846 37.2867 41.3892 37.2867 "
    "53.6661 49.5635L55.0219 50.9193L46.4082 59.5332L45.0524 58.1774C37.5328 "
    "50.6578 25.341 50.6578 17.8214 58.1774C10.3018 65.697 10.3019 77.8886 "
    "17.8214 85.4082L25.4626 93.0495L68.9488 49.5635ZM17.8216 117.919C10.302 "
    "125.438 10.302 137.63 17.8216 145.149C25.3412 152.669 37.5328 152.669 "
    "45.0524 145.149L52.6936 137.508L25.4628 110.277L17.8216 117.919ZM34.0767 "
    "101.663L61.3075 128.894L88.5383 101.663L61.3075 74.4325L34.0767 "
    "101.663ZM104.793 58.1774C97.2739 50.6579 85.0822 50.6578 77.5627 "
    "58.1774L69.9214 65.8186L97.1522 93.0495L104.793 85.4082C112.313 77.8886 "
    "112.313 65.697 104.793 58.1774Z\"/><path d=\"M205.374 "
    "37.2986H219.363V150H205.374V37.2986ZM179.281 151.572C168.488 151.572 "
    "160.052 148.061 153.974 141.04C147.896 134.02 144.858 123.75 144.858 "
    "110.232C144.858 101.011 146.272 93.3611 149.102 87.2833C151.931 81.1007 "
    "155.913 76.49 161.048 73.4511C166.182 70.3074 172.26 68.7356 179.281 "
    "68.7356C186.407 68.7356 192.275 70.517 196.886 74.0798C201.496 77.6427 "
    "204.902 82.5154 207.103 88.698C209.303 94.8806 210.404 102.059 210.404 "
    "110.232C210.404 118.301 209.303 125.479 207.103 131.767C204.902 137.949 "
    "201.496 142.822 196.886 146.385C192.275 149.843 186.407 151.572 179.281 "
    "151.572ZM182.582 139.783C190.546 139.783 196.309 137.216 199.872 "
    "132.081C203.54 126.946 205.374 119.663 205.374 110.232C205.374 100.801 "
    "203.54 93.5183 199.872 88.3836C196.309 83.1441 190.546 80.5244 182.582 "
    "80.5244C174.303 80.5244 168.278 83.1441 164.506 88.3836C160.733 93.5183 "
    "158.847 100.801 158.847 110.232C158.847 119.663 160.733 126.946 164.506 "
    "132.081C168.278 137.216 174.303 139.783 182.582 139.783ZM249.192 "
    "150H235.203V70.4646H249.192V84.2968C251.079 79.7909 253.751 76.3852 "
    "257.209 74.0798C260.667 71.6697 264.911 70.4646 269.941 "
    "70.4646H276.857V82.2534H267.897C261.715 82.2534 257.052 83.9825 253.908 "
    "87.4405C250.764 90.8986 249.192 96.0857 249.192 103.002V150ZM313.307 "
    "151.572C301.256 151.572 291.93 148.009 285.328 140.883C278.726 133.758 "
    "275.425 123.541 275.425 110.232C275.425 97.0288 278.726 86.8118 285.328 "
    "79.5813C291.93 72.3508 301.256 68.7356 313.307 68.7356C321.166 68.7356 "
    "327.925 70.3598 333.583 73.6083C339.242 76.752 343.538 81.4151 346.473 "
    "87.5977C349.511 93.7803 351.031 101.325 351.031 110.232C351.031 123.645 "
    "347.73 133.915 341.128 141.04C334.527 148.061 325.253 151.572 313.307 "
    "151.572ZM313.307 139.626C321.48 139.626 327.453 137.111 331.226 "
    "132.081C335.103 126.946 337.041 119.663 337.041 110.232C337.041 100.906 "
    "335.103 93.6755 331.226 88.5408C327.453 83.3013 321.48 80.6816 313.307 "
    "80.6816C305.028 80.6816 298.95 83.3013 295.073 88.5408C291.301 93.6755 "
    "289.415 100.906 289.415 110.232C289.415 119.663 291.301 126.946 295.073 "
    "132.081C298.95 137.111 305.028 139.626 313.307 139.626ZM376.862 "
    "70.4646V181.751H362.873V70.4646H376.862ZM402.797 151.572C395.777 151.572 "
    "389.961 149.843 385.35 146.385C380.739 142.822 377.334 137.949 375.133 "
    "131.767C373.037 125.479 371.989 118.353 371.989 110.389C371.989 102.216 "
    "373.037 95.0378 375.133 88.8552C377.334 82.6726 380.739 77.7999 385.35 "
    "74.237C389.961 70.6742 395.777 68.8927 402.797 68.8927C410.133 68.8927 "
    "416.368 70.4646 421.502 73.6083C426.637 76.752 430.514 81.4151 433.134 "
    "87.5977C435.859 93.6755 437.221 101.273 437.221 110.389C437.221 123.907 "
    "434.182 134.177 428.104 141.198C422.131 148.114 413.696 151.572 402.797 "
    "151.572ZM399.654 139.626C407.827 139.626 413.8 137.111 417.573 "
    "132.081C421.45 126.946 423.389 119.716 423.389 110.389C423.389 100.958 "
    "421.45 93.6755 417.573 88.5408C413.8 83.4061 407.827 80.8388 399.654 "
    "80.8388C391.795 80.8388 386.031 83.4061 382.363 88.5408C378.696 93.6755 "
    "376.862 100.958 376.862 110.389C376.862 119.716 378.696 126.946 382.363 "
    "132.081C386.031 137.111 391.795 139.626 399.654 139.626ZM481.021 "
    "151.572C468.971 151.572 459.644 148.009 453.043 140.883C446.441 133.758 "
    "443.14 123.541 443.14 110.232C443.14 97.0288 446.441 86.8118 453.043 "
    "79.5813C459.644 72.3508 468.971 68.7356 481.021 68.7356C488.881 68.7356 "
    "495.64 70.3598 501.298 73.6083C506.957 76.752 511.253 81.4151 514.187 "
    "87.5977C517.226 93.7803 518.746 101.325 518.746 110.232C518.746 123.645 "
    "515.445 133.915 508.843 141.04C502.241 148.061 492.967 151.572 481.021 "
    "151.572ZM481.021 139.626C489.195 139.626 495.168 137.111 498.94 "
    "132.081C502.818 126.946 504.756 119.663 504.756 110.232C504.756 100.906 "
    "502.818 93.6755 498.94 88.5408C495.168 83.3013 489.195 80.6816 481.021 "
    "80.6816C472.743 80.6816 466.665 83.3013 462.788 88.5408C459.016 93.6755 "
    "457.129 100.906 457.129 110.232C457.129 119.663 459.016 126.946 462.788 "
    "132.081C466.665 137.111 472.743 139.626 481.021 139.626ZM581.515 "
    "70.4646H595.505V150H581.515V138.054C579.105 141.931 576.066 145.18 572.398"
    " 147.799C568.731 150.314 563.753 151.572 557.466 151.572C548.873 151.572 "
    "542.062 149.162 537.032 144.341C532.002 139.416 529.487 131.347 529.487 "
    "120.135V70.4646H543.162V119.663C543.162 126.789 544.682 131.924 547.72 "
    "135.067C550.864 138.106 555.318 139.626 561.081 139.626C565.378 139.626 "
    "569.045 138.892 572.084 137.425C575.123 135.853 577.428 133.443 579 "
    "130.195C580.677 126.841 581.515 122.493 581.515 117.148V70.4646ZM646.661 "
    "82.2534H628.742V138.211H646.661V150H635.658C629.37 150 624.341 148.638 "
    "620.568 145.913C616.796 143.189 614.909 138.316 614.909 131.295V82.2534H60"
    "1.234V70.4646H614.909V46.7297H628.742V70.4646H646.661V82.2534ZM689.764 "
    "82.2534H671.845V138.211H689.764V150H678.761C672.473 150 667.444 148.638 "
    "663.671 145.913C659.899 143.189 658.012 138.316 658.012 131.295V82.2534H64"
    "4.337V70.4646H658.012V46.7297H671.845V70.4646H689.764V82.2534Z\"/></svg>"
)

#: The dropoutt mark, from the product's own icon-logo.svg, recoloured to
#: currentColor. Kept for the terminal-adjacent surfaces and for anywhere the
#: full lockup does not fit; the page header uses TEXT_LOGO_SVG.
LOGO_SVG = (
    '<svg class="mark" viewBox="0 0 625 625" fill="currentColor" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M279.82 279.82c17.574-17.573 46.066-17.574 63.64 0 17.573 17.573 '
    "17.573 46.066 0 63.639-17.574 17.574-46.066 17.574-63.64 0-17.573-17.573-"
    '17.573-46.066 0-63.639Z"/>'
    '<path fill-rule="evenodd" clip-rule="evenodd" d="M350.919 46.863c62.484-62.484 '
    "163.79-62.483 226.274-.001 62.483 62.484 62.484 163.79 0 226.274l-38.89 38.891 "
    "38.89 38.89c62.484 62.484 62.484 163.791 0 226.275-61.995 61.995-162.21 62.48-"
    "224.803 1.454l-1.471-1.454-6.905-6.904 43.841-43.84 6.905 6.903c38.271 38.271 "
    "100.321 38.271 138.592 0 38.271-38.271 38.272-100.322 0-138.593l-38.89-38.891"
    "-221.324 221.326c-62.484 62.483-163.791 62.483-226.275 0-61.9955-61.996-62.4804"
    "-162.21-1.4541-224.803l1.4532-1.473 38.8907-38.891-38.8907-38.89c-62.4839-62.484"
    "-62.4839-163.79 0-226.2744 62.4839-62.4835 163.7899-62.4835 226.2739 0l6.9 6.9004"
    "-43.84 43.8409-6.9-6.9004c-38.272-38.2714-100.323-38.2714-138.594 0-38.2712 38.2715"
    "-38.2711 100.3215 0 138.5925l38.891 38.891 221.325-221.3244ZM90.704 394.76c-38.2713 "
    "38.271-38.2713 100.322 0 138.593 38.271 38.271 100.322 38.271 138.593 0l38.89-38.892"
    "-138.592-138.593-38.891 38.892Zm82.731-82.732 138.593 138.592 138.593-138.592"
    "-138.593-138.593-138.593 138.593ZM533.352 90.704c-38.271-38.2712-100.321-38.2713"
    '-138.592 0l-38.891 38.89 138.593 138.593 38.89-38.891c38.271-38.271 38.271-100.321 '
    '0-138.592Z"/>'
    "</svg>"
)
