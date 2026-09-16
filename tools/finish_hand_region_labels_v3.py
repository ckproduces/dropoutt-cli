#!/usr/bin/env python3
"""Merge hand-curated atlas-v3 fine-cell labels into region_labels_atlas-v3.json."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tools" / "atlas-data"
CURATE = Path("/tmp/atlas-v3-curate.json")
PARTIAL = DATA / "hand_region_labels_l1_0_57.json"
BATCH = DATA / "hand_region_labels_l1_58_255.json"
OUT = DATA / "region_labels_atlas-v3.json"
NPZ = ROOT / "src/dropoutt/data/atlas/atlas-v3.npz"

# Hand-curated fine-cell names keyed by L1 parent (cells 58–255).
# Each list length must match the number of fine cells under that L1.
HAND_L1: dict[int, list[str]] = {}

# fmt: off
HAND_L1.update({
58: [
    "Publishing rights deals and industry contact pages",
    "French newsletter signup and subscription contact",
    "Account registration and login contact forms",
    "Dutch and Portuguese email contact pages",
    "Email contact and correspondence boilerplate",
    "German event enquiry and Kontakt forms",
    "German FAQ and customer enquiry pages",
    "Press releases with incidental contact mentions",
    "Russian order enquiry and email contact",
    "US local news with newsletter signup",
    "Spanish phone contact and coronavirus notices",
    "Japanese membership and registration contact",
    "German medical practice contact and corona updates",
    "App support email and feedback contact pages",
    "Travel and lifestyle pages with contact sidebars",
    "Exam revision guides and publisher contact",
    "Commercial property lease enquiry contact",
    "German thank-you and enquiry confirmation pages",
    "Government administration contact forms",
    "Italian email signup and celebrity news pages",
],
59: [
    "Small business client marketing and services",
    "E-commerce lingerie and retail entrepreneurship",
    "Accounting services for entrepreneurs",
    "Holiday planning and seasonal small business",
    "French nonprofit association tax guidance",
    "Polish employment and startup workplace culture",
    "Sports loan news with business software mentions",
    "Protest organising and community business pages",
    "Teen drama media with entrepreneurship keywords",
    "Russian font licensing and company registration",
    "Indonesian SEO and online business guides",
    "UK family-run local business pages",
    "French enterprise consulting and client trust",
    "Post-church ministry and personal brand pages",
    "Small business coaching and priority advice",
    "Spanish property listings with business terms",
    "Dutch faith-leader career transition pages",
],
60: [
    "Middle Eastern and Asian footballer biographies",
    "Brazilian and Spanish football transfer news",
    "Real Madrid, Atlético and Champions League coverage",
    "German Champions League and Bundesliga commentary",
    "La Liga, Premier League and transfer gossip",
    "Argentine youth team and Copa América news",
    "Spanish-language football phrase and AI prompts",
    "Spanish La Liga midfielder player profiles",
    "Brazilian Santos and club footballer biographies",
    "FIFA World Cup and national team diplomacy news",
    "Argentine footballer club career profiles",
    "Portuguese football game and Premier League pages",
    "European stadium capacity and venue profiles",
    "La Liga, Premier League and FIFA governance disputes",
    "Ronaldo versus Messi social media rivalry",
    "Neymar, Barcelona and manager appointment news",
    "Messi, Barcelona and transfer speculation",
    "Serie A Juventus and Italian calcio results",
    "German football coach and player biographies",
],
61: [
    "Dot-com startup property and tenancy news",
    "German landlord and tenant law commentary",
    "Czech rental housing category listings",
    "Vietnamese property development articles",
    "French residential property price overviews",
    "Serbian municipal town and housing pages",
    "Dutch home improvement and rental blogs",
    "Australian do-not-contact and mortgage advice",
    "Real estate agent marketing and home fragrance",
    "Japanese rental apartment cost guides",
    "Portuguese industrial property demand news",
    "UK customer service and short-term rental",
    "Italian social housing and financial reconstruction",
    "Dutch housing policy and education investment",
    "US for-sale-by-owner home selling advice",
    "French rental agency and tenant services",
    "German landlord–tenant dispute guidance",
],
62: [
    "Promotional greeting card and marketing banners",
    "UK crime and domestic violence news",
    "Cosmic microwave background science news",
    "Vatican document and church press statements",
    "Hungarian push-notification signup prompts",
    "UK fashion industry acquisition news",
    "Celebrity cosmetic surgery news briefs",
    "US college sports subscription news",
    "US magazine media roundup pages",
    "Film crew accident and quality journalism",
    "Hungarian horoscope and software changelog pages",
    "Football career statistics news sidebars",
    "Thanksgiving catering promotional news",
    "New music release roundup pages",
],
63: [
    "Russian Google Ads and AdWords marketing",
    "Google Chrome and browser release news",
    "Affiliate marketing and Google platform comparisons",
    "Indonesian Gmail invitation and email guides",
    "US recruiting stories with search keywords",
    "Google Maps and abstract website mentions",
    "Persian Google and Android technology news",
    "SEO definition and search-engine optimisation guides",
    "French search-engine and SEO introduction pages",
],
})

# fmt: on


def load_hand_l1() -> dict[int, list[str]]:
    """Load hand lists from this module plus the L1 batch file."""
    result = dict(HAND_L1)
    if BATCH.is_file():
        extra = json.loads(BATCH.read_text(encoding="utf-8"))
        for k, v in extra.items():
            result[int(k)] = v
    return result


def merge_labels() -> list[str]:
    curate = json.loads(CURATE.read_text(encoding="utf-8"))
    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.is_file() else {}
    hand = load_hand_l1()

    labels: list[str | None] = [None] * 4096
    for key, name in partial.items():
        labels[int(key)] = name

    for group in curate:
        l1 = group["l1"]
        cells = [row[0] for row in group["cells"]]
        if l1 in hand:
            names = hand[l1]
            if len(names) != len(cells):
                raise ValueError(f"L1 {l1}: expected {len(cells)} names, got {len(names)}")
            for cid, name in zip(cells, names, strict=True):
                labels[cid] = name

    missing = [i for i, v in enumerate(labels) if v is None]
    if missing:
        raise SystemExit(f"Missing {len(missing)} labels; first gaps at cell ids {missing[:20]}")

    # Unique within each L1 parent.
    for group in curate:
        l1 = group["l1"]
        cells = [row[0] for row in group["cells"]]
        seen: set[str] = set()
        for cid in cells:
            name = labels[cid]
            assert name is not None
            if name in seen:
                raise ValueError(f"L1 {l1}: duplicate label '{name}'")
            seen.add(name)

    return [labels[i] for i in range(4096)]


def write_region_labels(labels: list[str], corpus_hash: str) -> None:
    payload = {
        "_comment": [
            "Curated human-readable labels for the 4096 atlas-v3 fine cells.",
            "Each name was hand-written against that cell's contrastive terms,",
            "language mix, and exemplar records drawn from reservoir samples.",
            "Names are chosen with respect to each other: no two cells under the same",
            "L1 parent share a name, and sibling cells are distinguished by subject.",
            "Declarative curation: builds and promotion re-apply these names.",
            "Subject names only; geography is named only when the cell itself is",
            "geographically specific.",
        ],
        "atlas_version": "atlas-v3",
        # Names are bound to the clustering they describe. The builder applies
        # this file only when the hash matches the corpus it is building on;
        # without it a rebuild would inherit 4,096 names written for cells
        # that no longer exist.
        "corpus_hash": corpus_hash,
        "labels": {str(i): labels[i] for i in range(len(labels))},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _artifact_corpus_hash() -> str:
    meta = np.load(NPZ, allow_pickle=True)["meta"].item()
    if isinstance(meta, str):
        meta = json.loads(meta)
    return str(meta.get("corpus_hash", ""))


def patch_npz(labels: list[str]) -> None:
    data = dict(np.load(NPZ, allow_pickle=True))
    meta = data["meta"].item()
    if isinstance(meta, str):
        meta = json.loads(meta)
    meta["region_labels"] = labels
    meta["region_labels_source"] = "curated:region_labels_atlas-v3.json"
    data["meta"] = np.array([json.dumps(meta)])
    tmp = NPZ.parent / "atlas-v3.npz.patched.npz"
    np.savez_compressed(tmp, **data)
    shutil.copy2(tmp, NPZ)
    tmp.unlink(missing_ok=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-hash", required=True,
        help=(
            "corpus_hash of the build these names were written against. Must "
            "match the packaged atlas-v3.npz; the stamp is the annotator's "
            "claim, so it is never inferred from whatever artifact is on disk."
        ),
    )
    args = parser.parse_args()
    actual = _artifact_corpus_hash()
    if args.corpus_hash != actual:
        raise SystemExit(
            f"refusing: names claim corpus {args.corpus_hash[:12]} but the packaged "
            f"artifact is corpus {actual[:12]}. Hand labels written for one "
            "clustering do not describe another; annotate the current build."
        )
    labels = merge_labels()
    write_region_labels(labels, args.corpus_hash)
    patch_npz(labels)
    print(f"Wrote {len(labels)} labels to {OUT} and patched {NPZ}")


if __name__ == "__main__":
    main()
