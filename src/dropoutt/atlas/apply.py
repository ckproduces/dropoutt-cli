"""Loading a frozen atlas and assigning records to regions.

The atlas is a coordinate system, not a quality reference. Nothing in this module
knows whether a region is good; it only knows which bin a record lands in, so
that two datasets scanned on different machines land in the *same* bins and can
be compared.

Two honesty rules are enforced here rather than left to callers.

Records too far from every centroid are **off-atlas**, and off-atlas rate is
reported per language as well as globally. For a language the embedding model
represents poorly, topical assignment is unreliable no matter how good the
clustering is, and averaging that away would hide it.

Off-atlas records are **described, never used as grounds for withholding the
rest**. Until 0.1.4 a rate above 10% discarded the whole coverage report and
printed a sentence in its place. That was wrong twice over. The region histogram
never contained off-atlas records to begin with — they are filtered out by
``regions >= 0`` before counting — so suppression threw away a measurement that
was correct for every record it covered. And the off-atlas set is itself the most
useful thing the atlas produces on an ill-fitting corpus: it is a list of the
records that do not look like anything in the reference corpus, which is either a
gap in the atlas or a problem in the data, and the tool can tell those apart.

What it cannot do is let the reader confuse the denominators. Every share over
regions or categories is a share of **placed** records, and the placed count is
printed beside it.

One caveat governs how the off-atlas number should be read, and it is measured,
not assumed: similarity to the nearest centroid rises steeply with record length.
The same English paragraph scores 0.363 truncated to 20 characters and 0.787 at
2000, landing in the same region throughout. Across a real corpus the correlation
between log length and similarity is about 0.49, and off-atlas rate falls from
33% for records under 80 characters to 0% above 150. A high off-atlas rate is
therefore first a statement about record length, second about language, and only
third about topic. ``_diagnose`` below attributes it in that order rather than
letting the reader assume the third.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Off-atlas rate bands. These grade how much of the corpus the atlas actually
#: describes; they never gate whether the numbers are shown.
#:
#: The lower bound is not arbitrary. The cutoff itself was set at the 2nd
#: percentile of the atlas's own reference records, so a corpus drawn from the
#: same distribution as the atlas sits near 2%. Ten percent is five times that
#: and is the point where "some records did not place" becomes worth a sentence.
#: Above 35% the placed half is a minority report about the corpus, and the
#: heading says so.
OFF_ATLAS_NOTABLE = 0.10
OFF_ATLAS_HIGH = 0.35

#: An off-atlas set whose records are this much shorter than the placed ones is
#: explained by length before anything else. Measured: off-atlas rate is 33% for
#: records under 80 characters and 0% above 150.
SHORT_RECORD_RATIO = 0.6

#: Mean pairwise cosine inside the off-atlas set has to beat the placed set by
#: this margin before it counts as one thing rather than scattered leftovers. On
#: a corpus whose off-atlas records were random filler the two came out at 0.383
#: against 0.434, i.e. scattered.
#:
#: High coherence means the records resemble each other and nothing more. It does
#: *not* identify a missing subject area, and the wording must not imply that it
#: does: measured against this atlas, minified JavaScript scores 0.969, HTML
#: boilerplate 0.961, DNA 0.947, hex logs 0.875 and base64 0.871, against 0.277
#: for real prose. A genuinely missing topic — Ottoman endowment-deed vocabulary
#: — scores 0.886, squarely inside that range. Coherence cannot tell a subject
#: from a template. The surface thresholds below are what separate those.
COHERENCE_MARGIN = 0.05

#: Machine formats are not written like prose, and this is what actually
#: distinguishes them from a subject the atlas happens to lack. Measured shares
#: over the same families: whitespace runs 0.158 for prose and 0.132 for the
#: missing-topic case, against 0.000 for base64 and DNA, 0.037 for HTML and 0.041
#: for minified JS. Non-letter characters run 0.048 for prose and 0.000 for the
#: missing topic, against 0.191 for base64, 0.395 for minified JS and 0.556 for
#: hex logs. Either test alone leaves a gap; together they caught all six machine
#: formats and neither of the two prose cases.
SURFACE_WHITESPACE_RATIO = 0.5
SURFACE_NON_LETTER_MARGIN = 0.15


@dataclass
class Atlas:
    """A frozen coordinate system."""

    centroids: np.ndarray            # (n_regions, dim), L2-normalised
    region_category: np.ndarray      # (n_regions,) taxonomy id per region
    coords: np.ndarray               # (n_regions, 2) for display
    probe_coef: np.ndarray
    probe_intercept: np.ndarray
    probe_classes: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    artifact_hash: str = ""
    #: How the reference corpus itself spread across these regions, when the
    #: artifact records it. ``atlas-lite-v0`` does not: the build computed the
    #: assignment and discarded it, so a gap can only be reported as absence and
    #: never as under-representation against a baseline. Optional rather than
    #: required so v0 keeps loading.
    region_size: np.ndarray | None = None

    @property
    def n_regions(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def embed_model(self) -> str:
        return str(self.meta.get("embed_model", "unknown"))

    @property
    def off_threshold(self) -> float:
        return float(self.meta.get("off_atlas_threshold", 0.35))

    @property
    def region_terms(self) -> list[str]:
        return list(self.meta.get("region_terms", []))

    @classmethod
    def load(cls, path: str | Path) -> "Atlas":
        import hashlib  # noqa: PLC0415

        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"][0]))
        return cls(
            centroids=data["centroids"],
            region_category=data["region_category"],
            coords=data["coords"],
            probe_coef=data["probe_coef"],
            probe_intercept=data["probe_intercept"],
            probe_classes=data["probe_classes"],
            meta=meta,
            artifact_hash=digest,
            region_size=(
                data["region_size"] if "region_size" in data.files else None
            ),
        )

    # -- assignment -------------------------------------------------------

    def assign(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (region index, cosine similarity) for each row.

        A region index of -1 means off-atlas. Use :meth:`assign_full` when the
        nearest region of an off-atlas record matters, which it usually does.
        """
        best, score, _ = self.assign_full(embeddings)
        return best, score

    def assign_full(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (region or -1, cosine similarity, nearest region regardless).

        The third value is the region a record would have landed in had the
        cutoff not applied. ``assign`` computes it and then throws it away, which
        costs the caller the one thing that makes an off-atlas record legible:
        a record sitting just under the cutoff next to a Python region is a
        different problem from one that is nearest to a region of Arabic
        Wikipedia and 0.2 away from it.
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sims = emb @ self.centroids.T
        nearest = sims.argmax(axis=1)
        score = sims.max(axis=1)
        best = np.where(score >= self.off_threshold, nearest, -1)
        return best, score, nearest

    def categorize(self, embeddings: np.ndarray) -> np.ndarray:
        """Level-0 taxonomy category per row, from the supervised probe."""
        emb = np.asarray(embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        logits = emb @ self.probe_coef.T + self.probe_intercept
        return self.probe_classes[logits.argmax(axis=1)]

    # -- coverage ---------------------------------------------------------

    def coverage(
        self,
        regions: np.ndarray,
        categories: np.ndarray,
        languages: list[str] | None = None,
        *,
        scores: np.ndarray | None = None,
        nearest: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
        lengths: list[int] | None = None,
        datasets: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the coverage report.

        Always returns a histogram when anything was placed. The keyword
        arguments are what turns an off-atlas count into an off-atlas
        description; each is optional so that a caller holding only regions and
        categories still gets a valid, if thinner, report.
        """
        total = int(len(regions))
        if total == 0:
            return {"status": "no records"}

        off_mask = np.asarray(regions) < 0
        off = int(off_mask.sum())
        off_rate = off / total
        placed = total - off

        result: dict[str, Any] = {
            "records": total,
            "placed": placed,
            "off_atlas": off,
            "off_atlas_rate": round(off_rate, 4),
            "fit": _fit_band(off_rate),
            "atlas_version": self.meta.get("version"),
            "embed_model": self.embed_model,
            "off_atlas_cutoff": round(self.off_threshold, 5),
            "l0_holdout_accuracy": self.meta.get("l0_holdout_accuracy"),
            "region_purity_by_taxonomy": self.meta.get("region_purity_by_taxonomy"),
        }

        if languages:
            per_lang: dict[str, dict[str, float]] = {}
            for lang in sorted(set(languages)):
                idx = [i for i, x in enumerate(languages) if x == lang]
                if len(idx) < 20:
                    continue
                lang_off = sum(1 for i in idx if regions[i] < 0) / len(idx)
                per_lang[lang] = {
                    "records": len(idx),
                    "off_atlas_rate": round(lang_off, 4),
                    "reliable": lang_off <= OFF_ATLAS_NOTABLE,
                }
            result["per_language"] = per_lang

        if off:
            result["off_atlas_detail"] = self._describe_off_atlas(
                off_mask, scores=scores, nearest=nearest, embeddings=embeddings,
                lengths=lengths, datasets=datasets, languages=languages,
                texts=texts,
            )

        if placed == 0:
            # Nothing landed anywhere, so there is no histogram to report. This
            # is the one case where the numbers genuinely do not exist, as
            # opposed to being judged unfit to show.
            result["status"] = "none placed"
            return result

        on = regions[regions >= 0]
        region_counts = np.bincount(on, minlength=self.n_regions)
        # Categories are counted over placed records only. Counting them over
        # every record while the region histogram covers only placed ones put
        # two different denominators in the same panel, so a category share and
        # a region share could not be read against each other.
        cat_counts: dict[str, int] = {}
        for c in np.asarray(categories)[~off_mask]:
            key = str(int(c))
            cat_counts[key] = cat_counts.get(key, 0) + 1

        nonzero = int((region_counts > 0).sum())
        mass = region_counts / max(region_counts.sum(), 1)
        entropy = float(-(mass[mass > 0] * np.log(mass[mass > 0])).sum())

        result.update({
            "status": "ok",
            "regions_occupied": nonzero,
            "regions_total": self.n_regions,
            "region_entropy": round(entropy, 4),
            "max_region_entropy": round(float(np.log(self.n_regions)), 4),
            # The number that makes "34 of 258 occupied" readable. Occupancy
            # counts a region the same whether it holds one record or a third of
            # the corpus; this is the count of *evenly used* regions the corpus
            # is as spread out as. A corpus touching 34 regions with 80% of its
            # mass in two of them has an effective count near 3, and the gap
            # between the two numbers is itself the finding.
            "effective_regions": round(float(np.exp(entropy)), 2),
            #: Shares over `placed`, not over `records`.
            "by_category": cat_counts,
            # The full histogram, sparse. `top_regions` below is a display
            # convenience capped at twelve, and comparing two fingerprints on
            # that head alone silently computes shares over a fraction of each
            # corpus. Only occupied regions are stored, so this costs a few
            # hundred small integers and makes `dropoutt diff` exact.
            "region_counts": {
                str(int(r)): int(region_counts[r])
                for r in np.nonzero(region_counts)[0]
            },
            "top_regions": [
                {"region": int(r), "records": int(region_counts[r]),
                 "share": round(float(mass[r]), 4),
                 "category": int(self.region_category[r]),
                 "terms": self.region_terms[r] if r < len(self.region_terms) else ""}
                for r in np.argsort(-region_counts)[:12]
                if region_counts[r] > 0
            ],
            "coverage_gaps": self._gaps(region_counts),
            # How many subject areas the atlas actually carries regions for.
            # Not the size of the taxonomy: 31 categories are defined and only
            # 20 drew enough reference data to be clustered, so a gap list
            # measured against 31 would count 11 areas the atlas cannot see
            # either.
            "categories_total": len({int(c) for c in self.region_category}),
        })
        if datasets is not None and len(datasets) == total:
            result["by_dataset_regions"] = self._per_dataset(regions, datasets)
        return result

    # -- what the corpus does *not* cover ---------------------------------

    #: A category holding less than this share of placed records is reported as
    #: a gap rather than as coverage. It is not zero, because the level-0 probe
    #: is 86% accurate on held-out reference data, so a category with a handful
    #: of records is as likely to be probe error as to be real presence.
    GAP_SHARE = 0.005

    def _gaps(self, region_counts: np.ndarray) -> list[dict[str, Any]]:
        """Regions the atlas knows about that this corpus never reaches.

        This is the question a coverage histogram cannot answer by itself. A
        histogram tells you what is in the corpus; it takes a fixed reference
        coordinate system to tell you what is *missing* from it, and that is the
        entire reason the atlas is a frozen artifact rather than a clustering of
        whatever was scanned.

        Grouped by level-0 category, because 258 empty region ids is a list and
        20 named subject areas is an answer. Each entry carries the atlas's own
        terms for the largest empty region in that category, so the reader can
        see what they are missing rather than being told a number.
        """
        total = max(int(region_counts.sum()), 1)
        cats = np.asarray(self.region_category)
        terms = self.region_terms
        out: list[dict[str, Any]] = []

        for cid in sorted({int(c) for c in cats}):
            members = np.where(cats == cid)[0]
            held = int(region_counts[members].sum())
            share = held / total
            if share > self.GAP_SHARE:
                continue
            empty = [int(r) for r in members if region_counts[r] == 0]
            # Name the gap by its largest region, which is a stand-in for the
            # part of the category the corpus is furthest from covering.
            sample = ""
            if empty:
                sample = terms[empty[0]] if empty[0] < len(terms) else ""
            entry: dict[str, Any] = {
                "category": cid,
                "regions": len(members),
                "regions_empty": len(empty),
                "records": held,
                "share": round(share, 5),
                "terms": sample,
            }
            # When the artifact records how the reference corpus was spread, an
            # absolute gap becomes a relative one: not merely "nothing of yours
            # is here" but "the reference corpus put 8% of itself here". v0
            # carries no such record, so the key is simply absent rather than
            # filled with a guess.
            if self.region_size is not None:
                ref = np.asarray(self.region_size, dtype=float)
                ref_total = float(ref.sum()) or 1.0
                entry["reference_share"] = round(float(ref[members].sum()) / ref_total, 5)
            out.append(entry)
        return sorted(out, key=lambda g: (-g.get("reference_share", 0.0), -g["regions"]))

    def _per_dataset(
        self, regions: np.ndarray, datasets: list[str]
    ) -> dict[str, Any]:
        """Each dataset's own region histogram, and how alike they are.

        Two datasets can look entirely different at the record level and occupy
        the same handful of regions, which is what "we merged three sources and
        got no new coverage" looks like from the outside. The overlap matrix
        cannot see it, because it compares text and these share none.
        """
        arr = np.asarray(datasets, dtype=object)
        reg = np.asarray(regions)
        out: dict[str, dict[str, Any]] = {}
        for name in sorted(set(datasets)):
            sel = (arr == name) & (reg >= 0)
            n = int(sel.sum())
            if n < 20:
                continue
            counts = np.bincount(reg[sel], minlength=self.n_regions)
            out[str(name)] = {
                "placed": n,
                "regions_occupied": int((counts > 0).sum()),
                "top_regions": [
                    {"region": int(r), "share": round(float(counts[r] / n), 4)}
                    for r in np.argsort(-counts)[:5] if counts[r] > 0
                ],
                "_mass": (counts / n).tolist(),
            }

        names = list(out)
        pairs: list[dict[str, Any]] = []
        for i, a in enumerate(names):
            va = np.asarray(out[a]["_mass"])
            for b in names[i + 1:]:
                vb = np.asarray(out[b]["_mass"])
                denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                if denom <= 0:
                    continue
                pairs.append({"a": a, "b": b,
                              "similarity": round(float(va @ vb / denom), 4)})
        for entry in out.values():
            del entry["_mass"]
        if pairs:
            pairs.sort(key=lambda p: -p["similarity"])
            return {"datasets": out, "most_alike": pairs[:5]}
        return {"datasets": out, "most_alike": []}

    # -- the off-atlas set ------------------------------------------------

    def _describe_off_atlas(
        self,
        off_mask: np.ndarray,
        *,
        scores: np.ndarray | None,
        nearest: np.ndarray | None,
        embeddings: np.ndarray | None,
        lengths: list[int] | None,
        datasets: list[str] | None,
        languages: list[str] | None,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Say what the off-atlas records are, not merely how many.

        Everything here is derived from arrays the scan already has in memory,
        so the description costs one pass over an existing sample and no extra
        embedding work.
        """
        detail: dict[str, Any] = {}
        on_mask = ~off_mask

        if scores is not None:
            s = np.asarray(scores, dtype=float)
            off_s = s[off_mask]
            detail["score"] = {
                "cutoff": round(self.off_threshold, 5),
                "off_median": round(float(np.median(off_s)), 4),
                "off_p10": round(float(np.percentile(off_s, 10)), 4),
                "off_max": round(float(off_s.max()), 4),
                "placed_median": (
                    round(float(np.median(s[on_mask])), 4) if on_mask.any() else None
                ),
                # Records within 0.05 of the cutoff were near misses. A set made
                # mostly of near misses is a threshold effect; one sitting far
                # below is genuinely unlike the reference corpus.
                "near_miss_share": round(
                    float((off_s >= self.off_threshold - 0.05).mean()), 4
                ),
            }

        if lengths is not None and len(lengths) == len(off_mask):
            ln = np.asarray(lengths, dtype=float)
            detail["length"] = {
                "off_median_chars": int(np.median(ln[off_mask])),
                "placed_median_chars": (
                    int(np.median(ln[on_mask])) if on_mask.any() else None
                ),
            }

        if embeddings is not None and int(off_mask.sum()) >= 2:
            emb = np.asarray(embeddings, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            detail["coherence"] = {
                "off": _mean_pairwise(emb[off_mask]),
                "placed": _mean_pairwise(emb[on_mask]) if on_mask.sum() >= 2 else None,
            }

        if texts is not None and len(texts) == len(off_mask):
            # Shares only, never the text. Whether a record is written like prose
            # or like a machine format is the signal coherence cannot supply.
            ws = np.array([_whitespace_share(t) for t in texts])
            nl = np.array([_non_letter_share(t) for t in texts])
            detail["surface"] = {
                "off_whitespace": round(float(np.mean(ws[off_mask])), 4),
                "placed_whitespace": (
                    round(float(np.mean(ws[on_mask])), 4) if on_mask.any() else None
                ),
                "off_non_letter": round(float(np.mean(nl[off_mask])), 4),
                "placed_non_letter": (
                    round(float(np.mean(nl[on_mask])), 4) if on_mask.any() else None
                ),
            }

        if nearest is not None:
            near = np.asarray(nearest)[off_mask]
            counts = np.bincount(near, minlength=self.n_regions)
            detail["nearest_regions"] = [
                {"region": int(r), "records": int(counts[r]),
                 "terms": self.region_terms[r] if r < len(self.region_terms) else ""}
                for r in np.argsort(-counts)[:6] if counts[r] > 0
            ]
            detail["nearest_region_spread"] = int((counts > 0).sum())

        for key, values in (("by_language", languages), ("by_dataset", datasets)):
            if not values or len(values) != len(off_mask):
                continue
            arr = np.asarray(values, dtype=object)
            share: dict[str, dict[str, Any]] = {}
            for name in sorted(set(values)):
                sel = arr == name
                n_off = int((sel & off_mask).sum())
                if not n_off:
                    continue
                share[str(name)] = {
                    "off_atlas": n_off,
                    "records": int(sel.sum()),
                    # Rate within this group, which is the number that says
                    # whether the group is responsible or merely large.
                    "rate": round(n_off / int(sel.sum()), 4),
                    "share_of_off_atlas": round(n_off / int(off_mask.sum()), 4),
                }
            if share:
                detail[key] = share

        detail["diagnosis"] = _diagnose(detail)
        return detail


def _fit_band(off_rate: float) -> str:
    """How much of the corpus the atlas describes, as a word.

    Graded rather than pass/fail, because the underlying quantity is continuous
    and a corpus at 10.1% is not meaningfully different from one at 9.9%.
    """
    if off_rate <= OFF_ATLAS_NOTABLE:
        return "good"
    if off_rate <= OFF_ATLAS_HIGH:
        return "partial"
    return "poor"


def _whitespace_share(text: str) -> float:
    return sum(c.isspace() for c in text) / len(text) if text else 0.0


def _non_letter_share(text: str) -> float:
    if not text:
        return 0.0
    return sum(not (c.isalpha() or c.isspace()) for c in text) / len(text)


def _mean_pairwise(vectors: np.ndarray) -> float:
    """Mean cosine between every pair of rows, for L2-normalised input.

    Computed from the sum of the vectors rather than an n-by-n matrix: for unit
    rows, ``sum(V @ V.T) == ||sum(V)||^2`` and the diagonal contributes exactly
    n, so the mean over ordered pairs is ``(||sum(V)||^2 - n) / (n(n-1))``. Exact,
    and linear in n instead of quadratic, which matters because the atlas sample
    can carry tens of thousands of rows.
    """
    n = int(len(vectors))
    if n < 2:
        return float("nan")
    total = float(np.linalg.norm(vectors.sum(axis=0)) ** 2)
    return round(max(-1.0, min(1.0, (total - n) / (n * (n - 1)))), 4)


def _diagnose(detail: dict[str, Any]) -> str:
    """One sentence naming the most likely reason records went off-atlas.

    Ordered by how strongly each cause was measured to drive the similarity
    score, not by how interesting it would be. Length dominates, so it is tested
    first; a corpus of short records reads as off-atlas whatever it is about.
    """
    # Surface first. A machine format is a more specific and more actionable
    # answer than "short", and short base64 is not usefully described as short.
    surface = detail.get("surface") or {}
    off_ws, placed_ws = surface.get("off_whitespace"), surface.get("placed_whitespace")
    off_nl, placed_nl = surface.get("off_non_letter"), surface.get("placed_non_letter")
    if placed_ws is not None and placed_nl is not None:
        thin = off_ws < placed_ws * SURFACE_WHITESPACE_RATIO
        symbolic = off_nl > placed_nl + SURFACE_NON_LETTER_MARGIN
        if thin or symbolic:
            how = []
            if thin:
                how.append(f"{off_ws:.0%} whitespace against {placed_ws:.0%}")
            if symbolic:
                how.append(f"{off_nl:.0%} non-letter characters against {placed_nl:.0%}")
            return (
                f"not written like prose: {' and '.join(how)}. This is the profile of "
                f"markup, minified code, encoded blobs or log lines rather than of a "
                f"subject the atlas is missing"
            )

    length = detail.get("length") or {}
    off_len = length.get("off_median_chars")
    placed_len = length.get("placed_median_chars")
    if off_len and placed_len and off_len < placed_len * SHORT_RECORD_RATIO:
        return (
            f"mostly short records: the off-atlas half has a median of {off_len} "
            f"characters against {placed_len} for the placed half. Similarity to a "
            f"region rises with length, so short records read as off-atlas whatever "
            f"they are about"
        )

    coh = detail.get("coherence") or {}
    off_coh, placed_coh = coh.get("off"), coh.get("placed")
    if (
        off_coh is not None and placed_coh is not None
        and off_coh > placed_coh + COHERENCE_MARGIN
    ):
        # Deliberately stops at what was measured. An earlier version of this
        # sentence called it "a real subject area missing from the reference
        # corpus", which the measurements do not support: templated junk scores
        # 0.87 to 0.97 here, above real prose at 0.28. Alike is all this means.
        return (
            f"one kind of thing, not scattered: the off-atlas records resemble each "
            f"other closely ({off_coh:.2f} mean pairwise cosine against "
            f"{placed_coh:.2f} for the placed records), and their surface looks like "
            f"prose. That is consistent with a subject the atlas does not cover; the "
            f"nearest-region words below are what tells you which subject"
        )

    for key, noun in (("by_dataset", "dataset"), ("by_language", "language")):
        groups = detail.get(key) or {}
        if not groups:
            continue
        name, stats = max(groups.items(), key=lambda kv: kv[1]["share_of_off_atlas"])
        if stats["share_of_off_atlas"] >= 0.6 and len(groups) > 1:
            return (
                f"concentrated in one {noun}: {stats['share_of_off_atlas']:.0%} of the "
                f"off-atlas records are {noun} {name!r}, where the rate is "
                f"{stats['rate']:.0%}"
            )

    score = detail.get("score") or {}
    near_miss = score.get("near_miss_share")
    if near_miss is not None and near_miss >= 0.5:
        return (
            f"near misses: {near_miss:.0%} of the off-atlas records sit within 0.05 of "
            f"the cutoff. This is a threshold effect more than a real gap"
        )

    if off_coh is not None and placed_coh is not None:
        return (
            f"scattered: the off-atlas records are no more alike ({off_coh:.2f}) than "
            f"the placed ones ({placed_coh:.2f}), so they share no single subject. "
            f"Usually filler, boilerplate or markup rather than a missing topic"
        )
    return "no single cause identified"


def bundled_atlas_path() -> Path | None:
    from importlib import resources  # noqa: PLC0415

    try:
        ref = resources.files("dropoutt.data") / "atlas" / "atlas-lite-v0.npz"
        path = Path(str(ref))
        return path if path.exists() else None
    except Exception:
        return None


def load_bundled() -> Atlas | None:
    path = bundled_atlas_path()
    if path is None:
        return None
    try:
        return Atlas.load(path)
    except Exception:
        return None
