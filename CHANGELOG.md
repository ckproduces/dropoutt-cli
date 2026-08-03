# Changelog

## Unreleased

### The atlas places ten times as many records

The atlas and the token budget shared one 20,000-record ceiling, which put a
272,455-record corpus on the map with 6,852 of its records — few enough that a
subject area holding two percent of the corpus was decided by a hundred of
them. The two samples are now one bottom-k cut at two depths: 200,000 for the
atlas, still 20,000 for the panel. A prefix of a bottom-k over a uniform hash
is itself a uniform sample, so the budget keeps every property it had.

They are split because they cost different things. Placement is one forward
pass of a 128-dimension encoder per record — 30,000 records scan end to end in
under seven seconds — while pricing is five real tokenizers over the same text,
and tokens-per-character converges long before a histogram over 212 cells does.

Merging the shard samples now goes through a bounded heap. Shards may keep
three times their expected share so none of them discards a globally-small key,
which meant the parent was offered three times the target and held all of it to
throw two thirds away — tolerable at 20,000 and most of a gigabyte at 200,000.

### Atlas coverage is weighted by dataset size

The atlas sample is capped per dataset so that one huge dataset cannot dominate
it. That is the right sample and it was the wrong histogram: a dataset of ten
million records and one of ten thousand arrived at `Atlas.coverage` the same
size, so every coverage number described the average of your datasets rather
than your corpus. The token budget has always corrected for this in its own
way; coverage did not.

Each sampled record now carries how many corpus records it stands for, and the
histogram is weighted by it. Occupancy, entropy, effective regions, subject
shares and the gap list all derive from that histogram, so weighting it is the
whole of the fix. `placed` and `off_atlas` stay sample counts — they are honest
about being a sample — and `placed_estimated` is the weighted total beside them.

### Report: the map is drawn, not described

"Where your data sits" opened with four statistics and five sentences about a
map the reader had never seen. It now opens with the map itself: one row per
subject area, one chip per neighbourhood inside it, each chip carrying **and**
colouring your density there against the reference corpus's density in the same
place. All 48 areas are drawn whether or not the corpus touches them, because
the empty rows are half of what the picture says.

It is a real `<table>`, and the ratio is printed inside the chip rather than
revealed by hovering it. Both of those were bought by dropping the sort
control: `order` only applies to flex and grid children, so a sortable grid
could not be a table, and a table is what repeats its header across a printed
page. A number that exists only while a pointer is over it does not exist on
paper, in a screenshot pasted into a ticket, or for anyone reading by keyboard
— which is most of how this page is read.

The scale is four stops, and they are the traffic light the data already
implies: white, then green at parity, then yellow, then red for the densest
cell in the corpus. Below parity the curve is squared rather than linear — a
cell at 0.02x is one record where the map has fifty, and a linear ramp painted
it a comfortable-looking green two percent of the way along. Ink is chosen per
step by measured contrast against what actually renders; the previous version
compared each fill against pure black while the page rendered near-black, which
flattered the dark option and put 4.36:1 digits on the most saturated cells. A
test now fails if any step drops under AA.

A cell never reached is an outlined box with an `N` in it. An empty box at a
hairline border was ambiguous between "nothing here" and "nothing rendered",
and absence is what this section is for. A **Reach** column counts the
subregions touched in each row, and the legend is one gradient rather than four
swatches to interpolate between by eye.

The two place lists are ranked by density against the map instead of by record
count. Count asked "where is most of my data", which the grid above already
answers — and answers mostly with a fact about the map, since a region the
reference corpus spent forty cells on will hold more of anybody's data than one
it spent four on. They are now the two ends of one ordering: what you have most
of against the map, and what you have least of.

The density ratios are emitted by `Atlas.coverage` rather than derived in the
report, so `fingerprint.json` carries them and a CI job can assert on coverage
without loading the atlas artifact to divide by `region_size` itself.

The subject bar chart came out with it: it said the same thing one level
coarser, without the map's own resolution beside it. Its one irreplaceable line
— how much of the map this corpus never touches — is now a caption on the grid.

### Report: a Markdown rendering, and a terminal that stops repeating it

A scan runs in CI, and what a reviewer sees is a pull-request comment or a job
log. Attaching 60 KB of HTML to either produces something nobody opens. Every
scan now also writes `report.md`: the same reading of the same summary, as
pasteable Markdown, with the density grid as a table of ratios. It is written
whatever `--no-html` says, because that flag means "do not build the thing I
would open in a browser" and this is the file for exactly that reader.

The terminal is now triage. It answers *is anything wrong, and do I need to go
and look*, and then says where looking happens — each artifact named with a
word about what it is for. Everything it used to print in full was already in a
file written milliseconds earlier: the detail and fix for each finding, the
excerpts, the dataset table, the tokenizer panel, the off-map diagnosis. It is
about twenty lines whatever the corpus, down from sixty.

### Report: one theme, and the verdict caption sits with what it captions

The page answered to `prefers-color-scheme`, so the same file was two different
documents depending on whose laptop opened it — and the one thing a report is
for is being forwarded. A density ramp tuned to read on white does not read on
near-black. It is light, always, on screen and on paper.

The verdict strip moved from the head of the page to the head of the findings.
The sentence it carries is a count of what is directly below it and a name for
the worst of them, which is a caption for that list and was a verdict on a
corpus the reader had not been shown yet.

Findings are now ordered by consequence first and size second. Size used to be
promoted over nominal severity, which was defensible arithmetic and produced a
page where a badge reading "warning" stood above five badged "would block" — and
a reader who thinks the order is broken stops using it to decide what to read.

Every section standfirst is gone, along with the panels and captions that
restated a number already on the page: the licence finding printed "4 of 4
datasets" twice one line apart, "estimated from a sample" said a count was soft
without saying by how much, and a clean chat-template result took four lines to
report the absence of a problem the reader never had.

### Reader: a `.jsonl` holding a JSON array is read as records

The extension is a claim about the file, not a fact about it. A pretty-printed
array named `.jsonl` was read line by line, so every line was a parse error. On
one 1.37M-record corpus that was 1,256,362 failures, and the damage did not stop
at the parse count: the repeated `{` and `"license": ...` fragments became a
157,045-copy duplicate cluster, and they were too short to identify so 55% of
the corpus reported its language as unknown. One misread file, three wrong
findings. The incremental span scanner that `read_json` already used now handles
this case, and such a file is never line-split into shards.

### Reader: tabular headers are mapped to layout keys, and the delimiter is sniffed

A CSV carries whatever the author typed into the header row.
`Question,Activation-Feed,Result` matched no layout, fell back to raw text, and
had all three columns concatenated — which then produced "100% alike"
near-duplicate pairs, because the shared category column dominated the shingles.
Headers are now resolved against known layout keys, and the delimiter comes from
the header row rather than from the extension, so a semicolon-separated export
is no longer read as a single column.

### Near-duplicates: stricter, and no longer double-charging exact copies

The default MinHash preset is now `strict` (104 hashes, 8 bands of 13, Jaccard
0.85) rather than `fineweb` (0.75). At 0.75 the check fired on records that
share a template or a category column but say different things. Exact copies
inside a near-duplicate cluster are counted by `T0-DUP-001` and are now excluded
here, and no record is shown as its own near-copy. `fineweb` and `hf-neardedup`
remain available by name.

### Removed: two single-language checks

`T0-ENC-002` (Turkish dotted/dotless I damage) and `T1-LANG-004` (Turkish text
that lost its diacritics) are gone. The general language checks — composition,
outliers, script mismatch — are unchanged.

### Atlas: coarse and fine can no longer disagree

`categorize` derived the coarse region from a separate arg-max over L1
centroids while the fine cell came from an arg-max over L2 centroids. On the
shipped v2 artifact's own reference records those two answers differed for
**24.9%** of them, rising to 47.6% near the edge of a cell — drilling down could
flip the coarse answer, which is what "lite is an exact prefix of the hierarchy"
exists to rule out. The coarse label is now the parent of the assigned cell.

### Atlas: soft assignment is actually used

Top-5 soft assignment at T=0.08 had been implemented since v1 and called by
nothing but a test, so every coverage number was the top-1 answer presented as
the whole answer. Coverage now reports soft membership shares alongside the
hard histogram.

### Atlas: the shipped tables are read

Per-cell distance references, radial prototypes and the co-occurrence graph were
written into the artifact and never opened by any code path, and the
coarse-resolution correction the spec calls mandatory did not exist. The
correction table is now built and applied, and cells carry a flag saying whether
their siblings are distinct enough for a report to name one of them.

### Atlas: results name the encoder weights, and the version is pinned

Coverage results carry `encoder_weight_hash` and the normalization variant. The
bundled atlas is selected by a pinned version rather than "whichever file on
disk is newest", which was an implicit `atlas=latest`.

### Atlas build: collection is a separate, resumable, on-disk step

`tools/fetch_corpus.py` fetches the reference corpus once, in parallel, to a
gzipped shard per source plus a manifest, and skips anything already complete.
Collection was 87% of the v2 build wall clock and was paid again on every
re-cluster; nothing survived the process, so no two builds saw the same corpus.
`tools/build_atlas.py` reads that cache and never touches the network.

## 1.0.0

The first stable release. The command surface is frozen: six commands, and from
here they follow semantic versioning, so a flag will not change meaning inside a
major version.

### Atlas rebuilt as `atlas-lite-v1`

The coordinate system was redesigned against the architecture spec: shared
extract → chunk → dedup → SIF-embed (128-d) → frozen mean/PCA/L2 normalize →
two-level k-means (40 L1 / 480 L2), soft assignment, distance calibration, and
`pipeline_hash` on every coverage result. Reference sampling now spans math,
code, instruction/chat, legal/finance, scientific, dialogue, structured data,
and multilingual prose. Artifact budget raised to ≤ 5 MB; v1 ships at ~0.6 MB
with IDF table, norm constants, L1+L2 centroids, and reference mass. `atlas-lite-v0`
still loads; the package prefers v1.

### The token budget was wrong, by up to 38%

Measured against exact counts on a three-dataset corpus, the estimate printed
without `--model` came in **12% to 38% high** depending on the tokenizer — while
reporting a ±1% confidence interval that never contained the truth.

The cause was the sampling design, not the arithmetic. The scan caps its sample
per dataset so one huge dataset cannot swamp the atlas histogram, which means
the sample is not a miniature of the corpus: a corpus that is 94% English by
character produced a sample that was 60% English. Pooling that into a single
tokens-per-character ratio priced the whole corpus at the blend of the *sample*.
Tokens per character is about 0.19 in English and 0.28 in Turkish, so the gap
was worth tens of percent.

Each dataset is now priced at its own measured rate and the stratum totals
added. Same sample, same tokenizer calls. On the corpus above the error drops
from **+38% to −0.06%**, and the interval — now the stratified ratio estimator's
standard error, with a finite-population correction so a dataset that was
sampled in full contributes no uncertainty — contains the true value in every
case. The `± sampling` column in the report is that interval.

### The report describes before it complains

The page used to open with a list of findings, so a reader handed a folder
learned what was wrong with it before learning what it *was* — and learned what
it was only by inference, since a finding mentions a property of a corpus
exactly when that property is broken.

Composition now leads: language distribution, how much of the corpus parsed into
a structured training layout, which chat templates are already baked into the
text, record size, the dataset table. Then the findings, then the token budget,
then the map. The verdict stays above all of it as one line.

The page is now built on the product's own design tokens, so a report opened
next to the web app is recognisably the same product. Both themes ship, and
print is a first-class target rather than an afterthought.

### The atlas scatter plot is gone

It was honest and useless. The 258 dots sat at fixed positions, so two reports
could be held side by side — but the positions are a projection rather than
distances, which had to be disclaimed in a caption directly under the picture,
and having read the disclaimer there was nothing left to do with the dots.

In its place: sentences that clear two gates. A subject 7.6x denser here than
the map is built for. An area the map spends 11% of itself on that this corpus
barely reaches. One place holding 29% of everything, whose records are 0.88
alike — one template, not one topic. Both gates are needed and they fail in
opposite directions: on a 200,000-record corpus every difference is
statistically significant, so significance alone prints a page of 3% deviations;
on a 400-record sample a 5x difference is what noise looks like, so effect size
alone prints confident nonsense. Below 200 placed records no comparison is made
at all.

Two lists replace the single one: where the data piles up, and where it has only
a toehold. Reaching a place is not covering it, and an occupancy count cannot
tell the difference — which is how a narrow corpus comes to look broad.

### The report opens itself

A scan ends with an HTML file and the next thing anyone does is open it, so
`scan` opens it.

It is the default rather than a flag because the reasons not to are all
detectable. Over SSH the window would appear on the machine doing the work
rather than the one being looked at; in CI and under a batch scheduler there is
no session to open into; in a pipe the caller asked for text. Each of those is
checked before the default applies, along with `--quiet`, `--no-html`, and a
Linux box with no `DISPLAY`. `--no-open` and `DROPOUTT_OPEN=0` turn it off;
`DROPOUTT_OPEN=1` turns it on regardless, which is the X11-forwarding case where
the SSH check is wrong.

The report itself is now laid out for a phone as well as a monitor. That is less
strange than it sounds: the file gets sent to someone, and where it gets opened
is not where it was written.

### Four commands removed

`diff`, `index-eval`, `init` and `atlas` are gone. Each was useful and none was
finished enough to freeze for a decade of semantic versioning. What they did:

- `init` wrote a `dropoutt.toml`. The file is still read; write it by hand.
- `index-eval` built a contamination index from your own held-out set. The ten
  shipped benchmark indices are unaffected.
- `atlas` printed the artifact's own quality numbers. They travel in the
  `coverage` facet of every fingerprint.
- `diff` compared two fingerprints. The data it read is still written — the full
  region histogram is under `coverage.region_counts` precisely so two scans stay
  comparable.

### Speed, and one accelerator that was never actually running

`minhash.py` reported `"backend": "rensa"` in every near-duplicate finding, and
`dropoutt doctor` listed rensa as an installed accelerator with "speed only,
identical clusters". Nothing in the package ever called it — numpy did the work
on every machine, including the ones where the wheel was present.

Wiring it up was the wrong fix. A Rust MinHash seeds its permutations
differently, so the same corpus would produce different signatures, different
LSH buckets and a different duplicate count depending on which wheels happened
to be installed. The scan already guarantees its findings do not depend on how
many cores it was given. So the claim is gone, and so is the dependency: the
`fast` extra is now orjson alone.

What did get faster, measured on a 53,000-record three-language corpus:

- **Contamination hashing**, the single most expensive thing in a scan at about
  a quarter of it. The word list is joined and encoded once and each 8-gram is
  a `memoryview` slice of that one buffer, because the joined form of words
  *i..i+n* is a substring of the joined form of all of them. Digests are
  concatenated and read back with `frombuffer` instead of converted one at a
  time. 27 million list slices, joins and string encodes per scan, gone. The
  hashes are bit-identical — they have to be, the shipped indices are made of
  them — and `tests/test_fastpaths.py` asserts it against the definition.
- **Whitespace normalisation** is `str.split` rather than a regex. The two
  agree on every codepoint in Unicode, and split-then-join is four times
  faster. This runs three times per record.
- **The text-file sniffer** walked every character of every `.txt` in Python.
  Four characters can change its state, so numpy finds them and the loop visits
  only those — 7x on a half-megabyte probe. It also ran twice per file, once in
  discovery and once at read time; it is cached now.
- **The chat-template detector** ran 34 substring searches over every record.
  A delimiter cannot be present unless its first character is, and the gate is
  derived from the delimiters rather than written by hand. 7x on ordinary prose.
- Per-record set and dict rebuilds in schema induction and record normalisation
  now happen once at import.

Two latent bugs came out of the same pass. The incremental reader never trimmed
its buffer when a chunk contained no records, so a mislabelled dump with a long
records-free stretch was re-scanned from offset zero on every read — quadratic
in the file size, against a docstring promising bounded memory. And that reader
caught `Exception` around its JSON parse, so a `NameError` in the reader itself
would have been reported as a malformed record.

### Readable enough to open-source

`report/summary.py` was 1,151 lines. It is now three modules: the reading of a
scan, what the map says (`report/atlas_story.py`), and how a number becomes a
phrase (`report/phrasing.py`). The progress display moved out of `cli.py` into
`progress.py`; it was never about the command surface.

`pyproject.toml` now carries the lint configuration, and every rule that is
switched off says why in a comment. That deleted 56 identical `# noqa: PLC0415`
suppressions, each marking the same deliberate decision — imports live inside
functions so `--help` does not pay for numpy.

The bidi-override regex in `report/escaping.py` contained its characters
literally, which meant the source of the defence against direction-changing
text was itself unreadable and rendered wrongly in an editor. It is written as
escapes now.

### A CLI that looks like one

The product's mark, rasterised from its own icon, prints above the help. It is
dropped when stdout is redirected, when `NO_COLOR` or `TERM=dumb` is set, when
the terminal is narrower than 46 columns, and — the one that would otherwise
crash the program — when the output stream cannot encode a block character,
which is the default on a legacy Windows code page. A 7-bit rendering of the
same mark covers that case.

`-h` works as well as `--help` and `-V` as well as `--version`, and both have a
subcommand spelling too: `dropoutt help`, `dropoutt help scan`, and `dropoutt
version` all work, because guessing wrong about a tool's flag conventions should
not produce a usage error. `scan` prints four worked examples.

## 0.2.0

The scan got about thirteen times faster on a real corpus, and the report was
rewritten for the person who has to decide whether to start a training run.

### Speed

A 50,000-record SFT corpus went from **69 s to 13 s**, and a 200,000-record one
from about four and a half minutes to **21 s**. A scan of four small text files
did not get slower. Nothing was sampled away to get there; the checks see every
record they saw before.

Roughly half of it is the streaming pass now running across processes. A shard
is a contiguous slice of the corpus in the order a serial scan would read it,
each worker runs the real checks over its slice, and the parent folds the check
objects together afterwards. Findings, examples and the fingerprint id are
**identical** to a one-core run, which `tests/test_parallel.py` asserts. Under
the byte threshold, and with `-j 1`, the same code path runs in the calling
process, so there is no second implementation to keep in agreement.

Making that identity true forced three things to change, and each was already
a latent problem:

- The corpus digest was a chained hash over records in order, so it would have
  depended on how many shards a machine chose. It is a sum now, and depends on
  the records alone.
- `T0-DUP-001` keyed records by `hash()`, which Python randomises per
  interpreter — in separate processes the same text would have hashed
  differently and every duplicate would have counted as unique.
- The atlas and token-budget samples were the first N records of each dataset,
  which no shard can reproduce. They are a bottom-k sample over a positional
  hash now, so the sample is identical however the corpus is divided — and it
  covers the whole corpus rather than its beginning, which matters whenever the
  files are sorted by source, length or date.

The other half is arithmetic that was being done more times than necessary:

- Script detection was a Python loop over every character of every record. It
  is a 65,536-entry lookup table now, applied with numpy.
- Language identification calls fastText directly. The wrapper it went through
  re-collapsed whitespace with a regex over text that was already normalised,
  took a lock, and allocated a dict, for about 70 µs of overhead per record on
  top of a prediction that costs less.
- PII, identity and style patterns are gated by substrings derived from the
  patterns themselves (`dropoutt/regexgate.py`). A case-insensitive regex scans
  at 4–6 ns per character and a substring test at 0.36; sixteen of the former
  per record was the second largest line in the profile. The gates are derived,
  never hand-written, because a hand-written gate that drifts turns a check off
  silently.
- Near-duplicate and contamination scanning shared one normalisation pass
  instead of each running their own; MinHash shingles are hashed with a
  vectorised rolling hash; candidate pairs are verified as one matrix operation.
- The ten benchmark indices are merged into one sorted array with a compressed
  posting list. This removed 68 million dictionary lookups on the reference
  corpus and cut the memory tenfold, which is what makes running it in twelve
  processes affordable.
- The tokenizer panel loads in parallel and reads a bounded sample; the embedder
  and the panel warm up in the background while records are being read.

Contamination hashes are untouched — the shipped `.idx` files are made of them
and there is no text left anywhere to recompute them from.

`PIPELINE_VERSION` is `0.2.0`: the corpus digest, the sample and the MinHash
shingle hash all changed, so fingerprints from earlier versions describe
something slightly different and must not collide.

### The report

Both the terminal output and `report.html` were rewritten around one shared
reading of the scan (`dropoutt/report/summary.py`), so the two cannot drift.

- **Failures lead.** The page opens with a sentence — "1 problem would fail a
  fine-tuning run" — and then the problems, ordered by how much of the corpus
  each touches, each with its size, its cost in tokens, what to do, and examples.
- **Every number carries its unit.** Checks declare what they count. A finding
  about datasets no longer outranks one about records because both read as
  100%.
- **The map is the centrepiece**, drawn at full width with the legend and the
  dots agreeing on colour. Occupied ground is never drawn in the empty grey, and
  the caption says the layout is not a distance.
- **A place on the map is named by your own record** closest to its centre. The
  atlas's five-word captions are shown beside that, as captions. The two atlas
  findings no longer quote them at all: about forty percent of that text is
  function words shared with other regions, and a finding ending in
  "(such, used, other, also, some)" is noise presented as insight.
- The terminal report is roughly half its former length; the dataset table, the
  tokenizer comparison and the full off-map diagnosis live on the page, and the
  last line says where the page is.
- Several check titles were rewritten as problems rather than as topics —
  `T0-ROLE-001` read "Conversation role structure is valid", which is what a
  passing check would say.

### Progress

The scan shows a real progress bar with a record count, a percentage and a
remaining time, sized from a record-length estimate measured during schema
induction. Redirected output gets a line every 20,000 records instead.

### Packaging

- `py.typed`, Beta / OS-independent / Typing classifiers, sdist includes,
  GitHub Actions CI on Linux / macOS / Windows.
- Windows cache under `%LOCALAPPDATA%\dropoutt`.
- README rewritten with a full check-code catalog.

## 0.1.5

Three things, all of which came out of running the tool on real data and not
liking the answer.

### A folder of records read as a folder of documents

A scan of 248 `.txt` files reported **251 records** across **251 datasets**, at
one document each, and inferred the `corpus` profile. The files held **6,097 SFT
records**. Every conversational check was skipped — and not skipped visibly, in
the "not checked, and why" list, but never considered, because as far as the
scanner was concerned there were no conversations. Five findings came back where
there were fifteen.

Two independent causes, both fixed:

- **`.txt` and `.md` files are now sniffed rather than assumed.** A new brace-
  balanced scanner (`dropoutt.sniff`) finds every top-level JSON object in a
  file, which collapses line-delimited, blank-separated and ```json-fenced
  framings into one algorithm and makes the text *between* records fall out for
  free. Two rules keep it off real prose: most spans must parse as objects, and
  the spans must cover most of the file's non-whitespace bytes — so a blog post
  quoting one JSON snippet stays a blog post. Reading is incremental, so a
  mislabelled multi-gigabyte dump costs bounded memory.
- **Sharded siblings fold into one dataset.** `responses_0001.txt` … and
  `train-00000-of-00042.parquet` reduce to their family name when at least three
  siblings share it. Naming each shard as its own dataset made every per-dataset
  statistic a statistic about one shard, and made the overlap matrix compare a
  corpus against itself.

### The atlas produced a panel, not an answer

It printed an occupancy count, a spread percentage, and five frequency terms per
region — none of which a reader could act on, and none of which could ever reach
the findings table.

- **Effective region count.** Occupancy counts a region holding one record the
  same as one holding a third of the corpus. The exponential of the region
  entropy says how many *evenly used* regions the corpus is as spread out as, and
  the gap between the two numbers is the size of the tail.
- **Coverage gaps — the part that needed a frozen atlas to exist.** A histogram
  of your own data says what is present. Only a fixed coordinate system can say
  what is *absent*. Subject areas the atlas covers and the corpus does not are
  now listed by name. On the corpus above: no code generation, no reading
  comprehension, no arithmetic, no SQL, nothing on religion or ethics.
- **Regions named by your own records.** The five shipped terms label a region
  *in the reference corpus*. The scan now shows the user's own record sitting
  closest to each busy region's centre, which is what makes the label mean
  anything. Kept in `ctx.stats`, never in `fingerprint.json`.
- **Topical overlap between datasets.** Cosine between each pair's region
  histograms. Two datasets can share no wording and still occupy identical
  ground, which `T1-OVERLAP-001` compares text and cannot see.
- **The map is coloured by subject area** rather than by occupied-or-not, which
  the occupancy metric already said.
- **`tools/build_atlas.py` now records the reference distribution.** v0 computed
  it and threw it away, which is why gaps can only be reported as absence and not
  as under-representation. The loader reads it when present and says nothing when
  it is not; `atlas-lite-v0` is unchanged and still loads.

### Eleven checks added, taking the catalog from 27 to 38

Synthetic SFT data fails fluently: the file parses, the records are well-formed,
the text reads well, and something is still wrong. Every threshold below was
measured on a 6,097-record Turkish generation run before the check was written,
and two of them are deliberately silent on it.

| id | what it caught there |
| --- | --- |
| `T0-FORMAT-001` | 248 text files holding records |
| `T0-SCHEMA-005` | 37 records whose answer sat in a top-level key the layout never reads |
| `T0-REASON-001` | 17.9% of responses opened a `<think>` block and 82.1% did not |
| `T0-GEN-001` | 5 runs of generator scaffolding between records, including a leaked `<thinking_mode>off</thinking_mode>` |
| `T0-TRUNC-002` | nothing — correctly, the corpus is not capped |
| `T1-DUP-002` | nothing — correctly, no prompt is answered two ways |
| `T0-QUAL-001/002/003` | the three FineWeb line-shape filters, at FineWeb's thresholds, for the `corpus` profile |
| `T1-ATLAS-001/002` | topical narrowness and template-shaped crowding |

`T0-SCHEMA-005` is the one worth reading twice. A record can be well-formed and
still lose its answer: a complete assistant turn sitting in a top-level
`assistant` key *beside* `messages` rather than inside it. Every key the trainer
reads is valid, nothing complains, and the conversation ends on a user turn and
trains on nothing.

Both silent checks were made silent by evidence rather than by luck.
`T0-TRUNC-002` first fired on any distribution with few discrete response
lengths; it now requires the top length bucket to hold at least three times as
many records as any bucket within 250 characters beneath it, because a real cap
is a wall and a coarse distribution is not. `T1-ATLAS-001` first reported a
corpus touching 195 regions with an effective spread of 63 as concentrated,
which it is not; it now tests absolute narrowness instead of a ratio that every
long-tailed distribution satisfies.

### Existing checks made more precise

**`T0-DEGEN-001` no longer calls extraction degenerate.** It flagged a response
as copying the prompt whenever one contained the other. On a real corpus that
was wrong 38 times out of 38: every hit was either a multiple-choice answer —
necessarily a substring of the prompt that listed the options — or extractive QA,
where the instruction was literally "quote the answer from the passage". Both are
correct supervision. Containment now has to account for at least 90% of the other
side, so picking one option out of five, or quoting one sentence from a passage,
no longer counts, while a response that simply repeats the prompt still does.

### Compatibility

`PIPELINE_VERSION` moves to 0.1.5. Record counts, dataset counts and the inferred
profile all change for inputs holding records in text files, and the coverage
facet gains keys, so fingerprints from 0.1.4 describe something different and
must not be compared against these.

## 0.1.4

An off-atlas rate above 10% used to discard the entire coverage report and print
a sentence in its place:

```
    withheld 15% of records are off-atlas, above the 10% threshold. This atlas
    does not fit this corpus, so coverage numbers would be misleading and have
    been withheld.
```

That was wrong twice over. Off-atlas records are filtered out of the region
histogram *before* it is counted, so the histogram was never contaminated by them
and withholding it discarded a measurement that was correct for every record it
covered. And the off-atlas set is the most useful thing the atlas produces on a
corpus that does not fit it — it is a list of the records unlike anything in the
reference corpus — which the scan already knew and then threw away.

### Changed

- **Coverage is never withheld.** The histogram is reported whenever anything
  placed. Every share is over the **placed** records and the placed count is
  printed beside it, so the denominator is never implied. The only remaining
  no-numbers case is `none placed`, where the numbers genuinely do not exist.
- **Fit is graded, not passed or failed**: `good` at or below 10% off-atlas,
  `partial` to 35%, `poor` above. The underlying quantity is continuous and a
  corpus at 10.1% is not meaningfully different from one at 9.9%. Ten percent is
  five times the 2% an atlas-distributed corpus sits at by construction, since the
  cutoff is the 2nd percentile of the atlas's own reference records.
- **`dropoutt diff` no longer refuses a corpus with a high off-atlas rate.** It
  prints both placed shares and states which way the bias runs: a partial *right*
  side means regions it appears not to reach may be reached by records it could
  not place, so `New` is an upper bound. It still refuses across atlas versions,
  and for fingerprints written before this release whose histogram is already gone.

### Added

- **Off-atlas records are described rather than counted.** The report gives the
  similarity distribution against the cutoff, the regions they were nearest to
  anyway, the rate broken down by language and by dataset, the furthest excerpts,
  and a one-line reason.
- **The reason is attributed in measured order.** Similarity to a region rises
  steeply with record length: the same English paragraph scores **0.363 truncated
  to 20 characters and 0.787 at 2000**, landing in the same region throughout;
  across a real corpus the correlation between log length and similarity is about
  **0.49**, and the off-atlas rate falls from **33% under 80 characters to 0% above
  150**.

  The order of tests is: whether the records are written like prose at all, then
  length, then whether the off-atlas set is *coherent* rather than scattered, then
  concentration in one dataset or language, then whether they are simply near
  misses at the threshold.

  **Coherence deliberately does not claim a missing subject area.** It sounds like
  it should, and the measurements say otherwise: minified JavaScript scores 0.969
  mean pairwise cosine, HTML boilerplate 0.961, DNA 0.947, base64 0.871, against
  **0.277 for real prose** — and a genuinely missing topic (Ottoman
  endowment-deed vocabulary) scores 0.886, in the middle of the templates. High
  coherence means the records are alike, full stop. What separates a template from
  a subject is how the text is written, so the surface test runs first and the
  coherence sentence stops at what was measured.
- **Whitespace and non-letter share** of the off-atlas set against the placed set,
  which is what actually distinguishes markup, minified code, encoded blobs and
  log lines from a subject the atlas lacks. Measured shares: whitespace 0.158 for
  prose and 0.132 for the missing-topic case, against 0.000 for base64 and DNA and
  0.037 for HTML; non-letter 0.048 for prose and 0.000 for the missing topic,
  against 0.191 for base64 and 0.556 for hex logs.
- `Atlas.assign_full` returns the nearest region alongside the placement.
  `assign` computed it and threw it away, which cost the caller the one thing that
  makes an off-atlas record legible.

`docs/atlas.md` also now states plainly that **off-atlas is not the garbage
detector**. Machine formats usually place, confidently and wrongly, rather than
going off-atlas: on a corpus of 400 records where 100 were base64 and minified
JavaScript, the off-atlas count was zero and the blobs landed in a region labelled
`return, denklemin, array, tdrow, function`. Those records were caught by
`T1-LANG-001`, which flagged exactly 100 of 400. The checks are the instrument for
junk; a coordinate system will happily give nonsense a coordinate.

### Fixed

- **`by_category` counted every record while `region_counts` counted only placed
  ones**, putting two denominators in the same panel so a category share and a
  region share could not be read against each other. Both are now over placed
  records.
- Off-atlas excerpts are kept out of `fingerprint.json`, which is the artifact
  meant to be shareable, and respect `--no-evidence`.
- `docs/atlas.md` described a "33-word stoplist" for region labels. The literal
  has 38 entries and **only 16 of them do anything**: the other 22 are three
  characters or fewer and were already dropped by the length filter one step
  earlier. The labels also use the first 150 members in corpus order, not a
  random sample, which the docs did not say.

### Fixed (packaging and diagnostics)

- **`dropoutt doctor` now names the interpreter it probed.** Every status in that
  table is an import against one specific Python, and the common way to be
  confused by it is to install with a `pip` belonging to a different one. A venv
  created by `uv` ships no `pip`, so an activated shell falls through to whatever
  `pip` is next on `PATH` and installs somewhere the tool cannot see — the package
  installs successfully and the status stays `no`. `doctor` prints the interpreter
  path, and when anything is missing it prints an install command targeting that
  interpreter (`uv pip install` when the running Python has no `pip`).
- **The version had two sources and they drifted.** `pyproject.toml` and
  `src/dropoutt/__init__.py` each carried a literal, so 0.1.4 was tagged in one
  while `dropoutt doctor` reported 0.1.3 from the other. `pyproject.toml` now
  reads the version from the package via `[tool.hatch.version]`; bump
  `__init__.py` only.

`PIPELINE_VERSION` is bumped because both the presence and the denominator of the
coverage facet changed.

## 0.1.3

### Fixed

- `dropoutt diff` now rejects atlas `.npz` files, unrelated JSON, and malformed
  fingerprints with a command-specific explanation and example. Commands with
  required arguments print full help when run empty.
- Fingerprint `content_hash` now hashes normalized record content and structure.
  It previously hashed only dataset names, byte sizes, and record counts, so
  different same-sized corpora could receive the same fingerprint id.
- Fingerprint ids now include the effective profile, sequence length, tier,
  MinHash preset, and other runtime configuration after CLI overrides.
- Blocking now follows the declared `target`, not the inferred data profile.
- `tier` and `minhash_preset` from `dropoutt.toml` now reach the scanner.
- `eval_sets` now acts as a benchmark-name allowlist and rejects unknown names;
  it was previously parsed and then ignored.
- Malformed TOML is a usage error instead of being silently ignored. Python
  3.10 installs `tomli` so configuration works on every supported version.
- `init` no longer writes into the current directory when its requested path
  does not exist. `index-eval` rejects directories, unsupported extensions,
  empty inputs, and path traversal in benchmark names, and requires `--force`
  before overwriting an index.
- `scan` now rejects paths with no supported data files instead of returning a
  successful empty fingerprint.
- `.bz2` and `.xz` inputs are now actually decompressed. They were classified
  as supported but read as plain text. `.zst` input has an optional dependency
  and produces an install hint instead of gibberish when it is absent.
- Genuine internal failures exit 1 with a concise message. Set
  `DROPOUTT_DEBUG=1` to retain a traceback.
- `fetch` exits 1 when cache preparation is incomplete instead of printing a
  failure and returning success.

### Added

- `--no-evidence` omits record excerpts and source locations from terminal
  output, `findings.jsonl`, and `report.html`.
- The HTML report now uses a minimal, system-sans layout and plots occupied
  atlas regions on the frozen 2D atlas coordinates without scripts or network
  resources.
- Long CLI operations now show their active phase with a terminal spinner.
  Redirected batch output receives stable phase lines and periodic record
  counts.
- Arrow IPC, Feather, and ORC inputs are supported through the existing
  `pyarrow` extra. The model registry adds ten current model options and the
  benchmark registry adds MATH-500, AIME 2024/2025, BigCodeBench, and
  MMLU-Redux 2.0.
- `DROPOUTT_OFFLINE=1` and an existing `HF_HUB_OFFLINE=1` are honored by
  `scan` and `init`. `init` also accepts `--offline`.
- Reports and write-time output now state their confidentiality boundary.
- Fingerprints no longer carry contamination witness paths, and
  `--no-evidence` removes nested witness locations from structured finding data.
  Documentation no longer describes unkeyed private evaluation indices as safe
  to publish. New private indices store the input basename rather than its
  absolute path.

`PIPELINE_VERSION` is bumped because fingerprint identity changed.

## 0.1.2

The atlas was computed on every scan and then displayed nowhere. Coverage went
into `fingerprint.json` and never reached the terminal or the HTML report, and
there was no way to put two datasets side by side, which is the only thing a
shared coordinate system is for.

### Added

- **Atlas coverage in the scan output**, in the terminal and in `report.html`:
  regions occupied, spread against even coverage, off-atlas rate, top categories
  by name, and top regions with their label words. Suppressed coverage prints
  the reason rather than the numbers.
- **`dropoutt diff LEFT RIGHT`** compares two fingerprints across the shared
  atlas and answers "what does LEFT cover that RIGHT does not". Reports
  similarity, shared and new mass, the regions unique to each side, and the
  category mix with deltas.

  Directional, read left against right, for the same reason cross-dataset
  overlap is: a small specialised corpus can sit wholly inside a large one while
  the large one is barely inside it, and a symmetric score hides exactly the
  case worth acting on.

  It refuses rather than guesses. If either side had coverage suppressed, or the
  two fingerprints were built against different atlas versions, it says so and
  stops. It also does not rank datasets: whether new coverage helps depends on
  what you are training.
- **The full region histogram in the coverage facet.** `top_regions` is a
  display head capped at twelve, and comparing on it computed shares over a
  fraction of each corpus. On two real corpora the head covered 88% of one side
  and 36% of the other, overstating novelty as 100% where the true figure was
  62%. Only occupied regions are stored, so this costs a few hundred small
  integers.
- Category ids are resolved to names everywhere they are shown. `"10": 59` is
  not information.

`PIPELINE_VERSION` is bumped because the coverage facet changed.

## 0.1.1

Everything here was found by writing [docs/getting-started.md](docs/getting-started.md)
and running each documented command instead of assuming it worked.

### Fixed

- **The package could not be installed at all.** A `force-include` of
  `src/dropoutt/data` duplicated files that `packages` already carried, and
  hatchling refused to build: *"A second file is being added to the wheel
  archive at the same path"*. Removed; the wheel ships 4.4 MB of data as
  intended.
- **`--offline` reached the network.** `scan()` took no `offline` argument, so
  atlas coverage tried to download a ~500 MB embedding model on a node with no
  egress. `offline` is now threaded to the embedder, and a test fails on any
  socket connection during an offline scan.
- **`--offline` refused to use an already-populated cache.** The tokenizer and
  the chat template both returned empty rather than reading the cache, which
  silently changed every token number and disabled the loss-mask checks. Both
  now resolve locally, and an offline scan produces findings byte-identical to
  an online one.
- **`dropoutt fetch` did not exist**, although skipped-check messages told users
  to run it. Implemented: pre-downloads tokenizers, each model's
  `tokenizer_config.json`, and the atlas embedder into `DROPOUTT_CACHE`.
- **The fingerprint reported bytes on disk as `total_chars`, and `total_words`
  was hardcoded to 0.** Byte counts include JSON syntax and vary with language
  under UTF-8, so the one artifact meant to be comparable across datasets varied
  with file format. On the test fixture this reported 91,976 characters where
  there were 54,577.
- **`index-eval` wrote into the install tree**, failing on read-only
  site-packages, and building a private index silently switched off all ten
  bundled benchmarks. Indices are now written to the cache, and both locations
  are always searched.
- **Install hints lost their extras name.** rich parsed `[tokenizer]` as a style
  tag and deleted it, so users were told to `pip install 'dropoutt'`. All
  data-derived and hint strings are escaped at the render site; a dataset named
  `[red]` can no longer restyle the output either.
- `--quiet` help text described behaviour it does not have.

### Added

- `python -m dropoutt`, for the common case where the venv's `bin/` is not on
  `PATH`.
- [docs/getting-started.md](docs/getting-started.md).
- Honest documentation of what the atlas region labels are, why they read as
  random words, and what the level-0 accuracy figure does not mean. See
  [docs/atlas.md](docs/atlas.md).

## 0.1.0

First release. Tier 0 and Tier 1 checks, a fingerprint, and a first atlas.

### Added

- **27 checks** across two tiers. Tier 0 is structural and needs no model; Tier 1
  is statistical. `dropoutt checks` lists them; `dropoutt checks <id>` explains
  one.
- **Zero-configuration scan.** Discovery, schema induction against ten known
  layouts, detection of files that are not training data at all, hypothesis-based
  profile inference, and a cross-tokenizer token budget when no `--model` is given.
- **Contamination scanning** using the Tülu 3 rule, against 10 bundled hashed
  8-gram indices for permissively licensed benchmarks. `dropoutt index-eval`
  builds an index from your own held-out set locally.
- **`atlas-lite-v0`**: 258 regions over 152,622 reference records, level-0
  taxonomy accuracy 0.864, region purity 0.785. Coverage is suppressed when the
  off-atlas rate exceeds 10%, and is reported per language.
- **Self-contained HTML report**, one file, no server and no CDN.
- **Three-valued exit codes**: 0 completed, 1 internal error, 2 usage, 10
  blocking findings under a declared target.
- Verified static registries: 22 benchmarks with their genuinely scorable eval
  splits, 12 chat-template families with literal delimiters, 37 models including
  the Turkish set, PII patterns with checksums.

### Policies enforced in code and covered by tests

- Nothing blocks a run whose purpose was never declared.
- No check recommends deleting data without a measured effect; every finding in
  this release is `unverified`.
- Cross-dataset overlap is directional.
- PII never reaches the report; a test fails if a planted secret appears.
- Anything that degraded says so.

### Known issues

See [docs/limitations.md](docs/limitations.md). The largest are that language
identification is the main source of false positives, scale above roughly a
million records is unprofiled, and the scan is single-process.
