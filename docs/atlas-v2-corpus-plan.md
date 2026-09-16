# Atlas v2 corpus plan

Status: frozen for review. Fetching has not started.

## Corpus contract

- Retained UTF-8 text target: 150 GiB before JSON framing and gzip compression.
- Storage root: `/Volumes/ck512/dropoutt-atlas-v2`.
- General-web share: 90% from FineWeb and FineWeb2.
- Probe share: 10% from public, ungated, license-filtered natural-text datasets.
- Language quotas are byte quotas, not row quotas.
- Synthetic text, SFT/chat data, preference data, benchmarks, gated data, and private data are excluded.
- Probe bytes consume the English quota. They do not increase English above 49.5%.
- Cross-source exact and near-duplicate filtering remains required during model ingestion.

"Public web" means publicly retrievable web content. FineWeb and FineWeb2 are
released under ODC-By, but Common Crawl states that underlying page content can
remain subject to the original site owners' terms. This plan is not a claim
that 90% of the corpus is public-domain or uniformly open-licensed content.

## Domain shape

| Axis | Share | Retained target |
| --- | ---: | ---: |
| General web | 90.0% | 135.000 GiB |
| Encyclopedic | 3.0% | 4.500 GiB |
| Scientific | 2.0% | 3.000 GiB |
| Code | 2.0% | 3.000 GiB |
| Forum | 1.0% | 1.500 GiB |
| Legal and government | 1.0% | 1.500 GiB |
| Books | 0.5% | 0.750 GiB |
| Educational | 0.5% | 0.750 GiB |

## Source allocation

| Axis | Dataset | Retained target |
| --- | --- | ---: |
| English general web | `HuggingFaceFW/fineweb` | 59.250 GiB |
| Non-English general web | `HuggingFaceFW/fineweb-2` | 75.750 GiB |
| Encyclopedic | `wikimedia/wikipedia` | 4.500 GiB |
| Scientific | `common-pile/arxiv_papers_filtered` | 1.500 GiB |
| Scientific | `common-pile/peS2o_filtered` | 1.500 GiB |
| Code | `common-pile/stackv2_edu_filtered` | 3.000 GiB |
| Forum | `common-pile/stackexchange_filtered` | 1.500 GiB |
| Legal and government | `common-pile/regulations_filtered` | 0.750 GiB |
| Legal and government | `common-pile/caselaw_access_project_filtered` | 0.750 GiB |
| Books | `common-pile/project_gutenberg_filtered` | 0.375 GiB |
| Books | `common-pile/pre_1929_books_filtered` | 0.375 GiB |
| Educational | `common-pile/libretexts_filtered` | 0.750 GiB |

Hugging Face text-dataset downloads and likes were sampled on 2026-08-27 as a
discovery signal. Direct English Wikipedia was selected because it is popular,
natural text, public, ungated, and has declared redistribution licenses. Popular
benchmarks, synthetic datasets, SFT/chat sets, duplicates, noncommercial data,
and datasets without a declared redistribution basis were rejected.

## Language shape

The published W3Techs website-content-language snapshot from 2026-08-27 is the
frozen proxy. It measures websites, not traffic, tokens, or byte volume. Its
published numeric categories sum to 99.6%. The final 0.4% is an operational
long-tail proxy divided evenly across eight languages that W3Techs lists below
0.1%. Published categories are not renormalized.

| Language | Share | Retained target |
| --- | ---: | ---: |
| English | 49.5% | 74.250 GiB |
| Spanish | 6.0% | 9.000 GiB |
| German | 5.9% | 8.850 GiB |
| Japanese | 4.9% | 7.350 GiB |
| French | 4.5% | 6.750 GiB |
| Portuguese | 4.1% | 6.150 GiB |
| Russian | 3.4% | 5.100 GiB |
| Italian | 2.8% | 4.200 GiB |
| Dutch | 2.2% | 3.300 GiB |
| Polish | 1.8% | 2.700 GiB |
| Turkish | 1.6% | 2.400 GiB |
| Chinese | 1.3% | 1.950 GiB |
| Indonesian | 1.2% | 1.800 GiB |
| Czech, Persian, Vietnamese, Korean | 0.9% each | 1.350 GiB each |
| Ukrainian, Arabic, Hungarian | 0.6% each | 0.900 GiB each |
| Swedish, Romanian | 0.5% each | 0.750 GiB each |
| Greek, Danish, Finnish, Hebrew, Slovak | 0.4% each | 0.600 GiB each |
| Thai, Bulgarian, Norwegian | 0.3% each | 0.450 GiB each |
| Croatian, Serbian, Lithuanian | 0.2% each | 0.300 GiB each |
| Slovenian, Catalan, Estonian, Latvian, Bengali | 0.1% each | 0.150 GiB each |
| Hindi, Swahili, Tamil, Urdu, Georgian, Malay, Bosnian, Azerbaijani | 0.05% each | 0.075 GiB each |

## Progress and approval gate

The fetcher writes a live `progress.json` for each active source. The monitor
computes overall, axis, and language completion from retained UTF-8 bytes:

```bash
uv run python tools/atlas_fetch_progress.py --watch 5
```

After approval, the fetch command is:

```bash
uv run python tools/fetch_corpus.py
```

Do not run the fetch command before corpus-policy approval.
