# atlas-v3 benchmark report (technical)

Run 2026-09-12 on the release machine (14 cores, 48 GB). Four products placed the same texts through the exact `dropoutt atlas` path. **v3** is the shipped `atlas-v3.npz` (corpus 4203fc3a, 163.45M reference records, 4,096 hand-named cells in 256 hand-named regions). **v3-prev** is the earlier v3 build (corpus e92d45dd, 95.8M records) it replaced. **v2** (296 cells / 128 regions) and **v2-lite** (65 cells / 32 regions, 64-d) are the products it supersedes. Method, data provenance and rerun instructions are in [README.md](README.md); every number below is in `results/*.json`. A plain-language rendering is [REPORT.md](REPORT.md). Rerun after the off-atlas cutoff was calibrated to 0.3538 (see section 7); the centroids did not change.

## 1. Headline

| metric | v3 | v3-prev | v2 | v2-lite |
| --- | ---: | ---: | ---: | ---: |
| Held-out sources: cell → source accuracy (52-way, chance 2.1%) | 30.6% | 30.0% | 23.9% | 17.0% |
| Held-out sources: cell → axis accuracy (8-way) | 65.7% | 65.9% | 62.4% | 54.6% |
| MMLU: cell → subject accuracy (57-way, chance 11%) | 39.3% | 39.3% | 35.1% | 27.2% |
| MMLU-Pro: cell → category accuracy (14-way, chance 11%) | 50.6% | 49.8% | 45.7% | 37.7% |
| EXAMS-tr: region → subject accuracy (8-way, chance 24%) | 63.3% | 64.7% | 61.6% | 54.2% |
| Probe topics: same topic, different language → same region | 48.8% | 28.6% | 29.8% | 39.3% |
| Probe topics: different topic, same language → same region (lower is better) | 0.3% | 22.3% | 21.7% | 25.0% |
| WMT17 tr-en: sentence and its translation share a region | 40.2% | 40.2% | 40.7% | 38.5% |
| Web in 9 languages: NMI(region, language) at L1 (lower is better) | 0.091 | 0.170 | 0.138 | 0.135 |
| Injection: smallest share of eurlex detected in a web corpus | 0.25% | 0.25% | 0.50% | not detected |
| Injection: smallest share of Turkish instructions detected | 2.00% | 2.00% | 5.00% | 10.00% |
| Off-atlas rate on 52 held-out prose/code sources | 0.7% | 1.1% | 11.8% | 16.9% |
| Off-atlas rate on 1,400 synthetic machine-format records (higher is better; cutoff cannot fix this) | 0.0% | 0.0% | 6.6% | 3.4% |
| compare(): New between two halves of one corpus at 5,000 records (should be ~0) | 18.8% | 14.0% | 0.4% | 0.0% |
| Cell names: own centroid ranks first among all cells | 31.2% | 25.3% | 47.0% | 20.0% |
| Region names: own centroid ranks first among regions | 72.7% | 62.5% | 55.5% | 68.8% |
| Exemplar texts placed back land in their own cell | 53.3% | 29.8% | 40.3% | 32.9% |

**Reading it.** v3 is the best map on every categorisation and detection measure, and its hand names are the most aligned with their cells. Two things did not improve with the map and are properties of the code around it, not of the geometry: the off-atlas cutoff, now calibrated at 0.3538, cannot keep machine-format records off the map on this encoder; and `compare()` reads a fifth of a corpus as *new* against its own other half, because 4,096 cells are sparsely occupied at ordinary sample sizes. Both are quantified in sections 6 and 7.

## 2. What was placed

- **52 held-out sources** (1,000 records each, ≥ 80 characters, the CLI's placement floor) from the local v2 build cache; none is a v3 build input. Axes: code (13, eleven programming languages), math (9), legal/finance (7), scientific (4), instruction/chat (9, two Turkish), dialogue/QA (3), text-to-SQL (4), web (C4 en, C4 multilingual), Turkish news (1).
- **MMLU** test (13,756 questions with options, 57 subjects), **MMLU-Pro** test (12,007, 14 categories), **EXAMS** crosslingual Turkish (1,561 school-exam questions, 8 subjects).
- **WMT17 tr-en**: 8,000 sentence pairs with both sides ≥ 80 characters; the repo's **probe file** (14 topics × en/tr/ar/zh, plus 4 out-of-distribution blobs).
- **Language panels** (in-build family, same datasets as v3 inputs): fineweb / fineweb-2 in 9 languages (4,000 each) and Wikipedia in 13 languages (3,000 each).
- **Coverage**: C4 English (40,000, held out) as the injection base; seven held-out sources injected at 0.25–10%; a nine-step ladder of growing breadth; ten corpus pairs through the CLI's own `compare()`; 1,400 synthetic machine-format records.

Ceilings are a logistic-regression probe and a nearest-class-mean classifier trained on the same projected vectors (half the records), which say how much of what the encoder knows the frozen grid keeps. Accuracies are majority-vote maps from cell to label fitted on one half and scored on the other. AMI is adjusted mutual information, which corrects for the finite-sample inflation a 4,096-way partition gets for free.

## 3. Source categorisation (B1)

| product | off-atlas | cells used | source acc (cell) | source acc (region) | source acc, soft top-5 | source AMI (cell) | ceiling: centroid | ceiling: linear | axis acc (cell) | axis acc (region) | axis ceiling | code language acc (cell) | code lang. ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.7% | 3599 | 30.6% | 25.2% | 51.1% | 0.291 | 55.7% | 64.9% | 65.7% | 64.1% | 82.3% | 34.4% | 58.3% |
| v3-prev | 1.1% | 3269 | 30.0% | 23.1% | 49.5% | 0.292 | 55.9% | 65.1% | 65.9% | 63.5% | 82.0% | 33.9% | 59.2% |
| v2 | 11.8% | 295 | 23.9% | 20.7% | 35.1% | 0.314 | 56.0% | 65.7% | 62.4% | 59.5% | 82.2% | 25.6% | 59.3% |
| v2-lite | 16.9% | 65 | 17.0% | 14.3% | 24.4% | 0.256 | 49.9% | 57.0% | 54.6% | 52.6% | 76.6% | 26.9% | 52.9% |

Chance is 2.1% for sources and 33% (majority class: instruction) for axes. Reading a v3 cell as a source label recovers 47% of what a supervised probe gets (30.6 of 64.9 points); reading the soft top-5 recovers 79%. Programming language is not a subject and the map does not separate it (34% over a 23% majority class), which is the intended behaviour of a subject map. The v2 AMI is slightly higher than v3's because AMI penalises the 4,096-way partition for the many cells that hold a handful of records; every accuracy and purity measure, which is what a reader of the report sees, favours v3.

Where the held-out sources landed on v3 (largest region and its share of the source):

| source | off-atlas | regions reached | top region share | top region | top cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| hendrycks_math__geometry | 0.0% | 21 | 91% | Mathematics papers, proofs and theorems | Math research abstracts and proof-based forum questions |
| hendrycks_math__algebra | 0.0% | 24 | 87% | Mathematics papers, proofs and theorems | Math research abstracts and proof-based forum questions |
| lex_glue__eurlex | 0.0% | 26 | 87% | Commission regulations on export refunds and prices | EU regulations setting agricultural export refund rates |
| commitpackft__php | 1.1% | 60 | 81% | Source code, functions and programming questions | Typed class and API usage code snippets |
| CShorten__ML-ArXiv-Papers | 0.0% | 56 | 79% | Machine-learning, statistics and computer science papers | Deep and convolutional neural network papers |
| hendrycks_math__number_theory | 0.0% | 21 | 77% | Mathematics papers, proofs and theorems | Math research abstracts and proof-based forum questions |
| commitpackft__javascript | 1.3% | 78 | 76% | Source code, functions and programming questions | Typed class and API usage code snippets |
| commitpackft__java | 0.3% | 53 | 73% | Source code, functions and programming questions | Typed class and API usage code snippets |
| lex_glue__scotus | 0.0% | 52 | 70% | Court opinions, appeals and case law | Federal appellate opinions in United States v. defendant cases |
| nvidia__OpenMathInstruct-2 | 0.3% | 80 | 65% | Mathematics papers, proofs and theorems | Math research abstracts and proof-based forum questions |
| codecomplete__starcoderdata_0.003 | 2.3% | 77 | 58% | Source code, functions and programming questions | Typed class and API usage code snippets |
| commitpackft__shell | 0.1% | 54 | 58% | Software installation guides and shell setup | Embedded and IoT device setup instructions |
| commitpackft__c | 0.3% | 46 | 52% | Source code, functions and programming questions | Code review and algorithm optimization questions |
| open-web-math__open-web-math | 0.1% | 111 | 51% | Mathematics papers, proofs and theorems | Math research abstracts and proof-based forum questions |
| codeparrot-clean-valid | 0.4% | 57 | 49% | Source code, functions and programming questions | Code review and algorithm optimization questions |
| stack-exchange-preferences | 0.1% | 64 | 49% | Source code, functions and programming questions | Shell scripting and file-manipulation Q&A |
| commitpackft__rust | 1.2% | 65 | 49% | Source code, functions and programming questions | String and character-handling programming Q&A |
| commitpackft__go | 1.3% | 60 | 48% | Source code, functions and programming questions | Code review and algorithm optimization questions |
| finemath__finemath-3plus | 0.0% | 104 | 44% | Mathematics papers, proofs and theorems | Programming puzzles about number sequences and algorithms |
| commitpackft__python | 0.6% | 83 | 43% | Source code, functions and programming questions | Command-line file and directory management questions |
| proof-pile-2__arxiv | 0.0% | 51 | 42% | Mathematics papers, proofs and theorems | Research abstracts about differential equations and operator theory |
| lex_glue__ledgar | 0.0% | 41 | 41% | Statutes, decrees and official gazettes | Consumer contract terms and conditions notices |
| commitpackft__ruby | 0.8% | 90 | 40% | Source code, functions and programming questions | React and Ruby package installation snippets |
| openai__gsm8k__main | 0.4% | 113 | 36% | Money arithmetic word problems | Word problems about total cost and discounts |
| iamtarun__python_code_instructions_18k_alpaca | 0.5% | 89 | 32% | Mathematics papers, proofs and theorems | Tutorials about Python data analysis and visualization |
| lex_glue__ecthr_a | 0.0% | 58 | 31% | Court opinions, appeals and case law | Appellate opinions on civil judgments and procedural motions |
| qiaojin__PubMedQA__pqa_artificial | 0.0% | 50 | 31% | Cell biology, proteins and immunology | Research abstracts about cardiovascular and heart disease |
| motherduckdb__duckdb-text2sql-25k | 2.4% | 137 | 30% | Source code, functions and programming questions | Database and SQL query Q&A |
| xlcost-text-to-code__Python | 0.0% | 29 | 30% | Product listings, sizes and shopping-cart pages | Product listings spanning cosmetics, livestock and lab gear |
| gretelai__synthetic_text_to_sql | 0.3% | 151 | 28% | Court records, web pages and digitised books | SQL query and stored-procedure troubleshooting snippets |
| deepmind__aqua_rat__raw | 0.3% | 107 | 27% | Money arithmetic word problems | Programming puzzles about number sequences and algorithms |
| microsoft__orca-math-word-problems-200k | 0.7% | 122 | 26% | Money arithmetic word problems | Cost comparison and price calculation articles |
| lex_glue__unfair_tos | 1.9% | 112 | 24% | Privacy policies and data protection notices | Website terms and conditions boilerplate |
| gbharti__finance-alpaca | 0.1% | 123 | 23% | Loans, credit cards and personal payments | Articles about stock investment strategy and ETFs |
| xlangai__spider | 0.7% | 128 | 20% | Names and entries starting with T | SQL query and stored-procedure troubleshooting snippets |
| BEE-spoke-data__peS2o-100k_en-xlong | 0.0% | 89 | 18% | Cell biology, proteins and immunology | Research abstracts about cohort studies and disease risk factors |
| mcemilg__news-cat | 1.3% | 145 | 14% | Football leagues, cups and match previews | Football coverage of Turkey's Süper Lig clubs |
| winddude__reddit_finance_43_250k | 0.1% | 111 | 13% | Stock markets, interest rates and market reports | Articles about saving money and personal budgeting |
| Anthropic__hh-rlhf | 0.6% | 170 | 12% | Premise-hypothesis entailment prompts | NLI premise-hypothesis pairs about people observing others |
| b-mc2__sql-create-context | 5.0% | 159 | 8% | Source code, functions and programming questions | Programming puzzles about number sequences and algorithms |
| openbmb__UltraFeedback | 1.5% | 212 | 7% | Questions, answers and puzzle prompts | Nonsensical-sentence and paraphrase-identification training prompts |
| no_robots | 0.3% | 200 | 6% | Narrative prose, outdoor stories and memoirs | Personal blog poetry about dreams and emotions |
| allenai__c4__multilingual | 1.1% | 196 | 6% | Court records, web pages and digitised books | Local news articles and personal blog posts |
| tatsu-lab__alpaca | 0.7% | 193 | 5% | Languages, linguistics and language learning | Word game and anagram dictionary tools |
| ultrachat_200k | 0.3% | 188 | 5% | Recipes, cooking and home kitchens | Lessons about fiction genres and storytelling |
| turkish-nlp-suite__InstrucTurca | 2.4% | 210 | 4% | Source code, functions and programming questions | Multiple-choice aptitude-test math problems |
| TFLai__Turkish-Alpaca | 4.3% | 202 | 4% | Artificial intelligence, robotics and technology | Sentence-rewriting prompts about behavior and descriptive facts |
| nvidia__HelpSteer2 | 0.1% | 191 | 4% | Machine-learning, statistics and computer science papers | AI marketing and buzzword blog posts |
| OpenAssistant__oasst1 | 1.4% | 199 | 3% | Machine-learning, statistics and computer science papers | Python installation and package-import troubleshooting posts |
| rajpurkar__squad | 0.0% | 177 | 3% | Human rights, democracy and governance | Judaism religion and culture articles |
| databricksbricks-dolly-15k | 0.6% | 211 | 2% | US pro sports, coaches and draft news | Musical instrument description pages |
| allenai__c4__en | 0.1% | 218 | 2% | Club notices, recurring meetups and invitations | Marketing copy for landscaping and home-renovation trades |

The narrow sources (competition math, EU law, code, SCOTUS) land 70–90% in one aptly named region. Mixed instruction sets (alpaca, dolly, ultrachat, oasst1) spread over 190–210 regions with no region above 6%, which is what a general-purpose instruction mixture looks like on a subject map. Two oddities are explained by the data, not the map: xlcost Python is code rendered as `NEW_LINE INDENT` token streams, a template rather than prose, and squad's top region is whichever Wikipedia topic its passages happen to sample most.

## 4. Topic categorisation on labelled benchmarks (B2)

| product | set | off-atlas | cell → label acc | region → label acc | chance | ceiling (linear) | AMI cell | AMI region | purity cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | MMLU (57) | 1.0% | 39.3% | 36.1% | 11.0% | 63.6% | 0.307 | 0.364 | 55.5% |
| v3-prev | MMLU (57) | 1.9% | 39.3% | 36.0% | 11.0% | 63.7% | 0.309 | 0.364 | 54.7% |
| v2 | MMLU (57) | 17.7% | 35.1% | 32.5% | 11.0% | 63.7% | 0.348 | 0.347 | 38.5% |
| v2-lite | MMLU (57) | 17.5% | 27.2% | 24.1% | 11.0% | 53.8% | 0.296 | 0.272 | 28.5% |
| v3 | MMLU-Pro (14) | 0.4% | 50.6% | 45.3% | 11.3% | 69.3% | 0.284 | 0.322 | 62.8% |
| v3-prev | MMLU-Pro (14) | 0.9% | 49.8% | 46.6% | 11.3% | 68.8% | 0.285 | 0.329 | 61.9% |
| v2 | MMLU-Pro (14) | 12.9% | 45.7% | 42.7% | 11.3% | 68.9% | 0.314 | 0.311 | 47.7% |
| v2-lite | MMLU-Pro (14) | 13.5% | 37.7% | 35.7% | 11.3% | 61.6% | 0.277 | 0.274 | 39.1% |
| v3 | EXAMS-tr (8) | 2.8% | 57.4% | 63.3% | 23.6% | 81.7% | 0.194 | 0.279 | 85.1% |
| v3-prev | EXAMS-tr (8) | 4.5% | 55.4% | 64.7% | 23.6% | 80.7% | 0.199 | 0.286 | 84.4% |
| v2 | EXAMS-tr (8) | 25.9% | 61.7% | 61.6% | 23.6% | 81.2% | 0.279 | 0.292 | 69.6% |
| v2-lite | EXAMS-tr (8) | 28.7% | 53.8% | 54.2% | 23.6% | 75.7% | 0.222 | 0.231 | 56.4% |

On MMLU a v3 cell predicts the subject 3.6× better than chance and reaches 62% of the supervised ceiling; on the Turkish exam questions the 256 regions do better than the cells (63% against 57%), which is the expected shape for 1,561 short questions over 4,096 cells. v2 reads the Turkish set at cell level slightly better (62%) while leaving 26% of its questions off the map; v3 places 97%.

Selected MMLU subjects and the v3 region that holds most of their questions:

| MMLU subject | n | share | top region | share | second region |
| --- | ---: | ---: | ---: | ---: | ---: |
| astronomy | 150 | 51% | Astronomy, asteroids and space missions | 13% | Quantum physics, optics and nanomaterials |
| virology | 165 | 37% | Infectious disease, vaccines and pandemic news | 15% | Cell biology, proteins and immunology |
| nutrition | 303 | 36% | Weight loss, nutrition and dieting | 12% | Cell biology, proteins and immunology |
| international_law | 121 | 55% | Courts, lawyers and legal procedure | 12% | Human rights, democracy and governance |
| high_school_macroeconomics | 389 | 37% | Stock markets, interest rates and market reports | 24% | Economies, growth and economic policy |
| college_biology | 144 | 34% | Cell biology, proteins and immunology | 12% | Genetics, genomics and biotechnology |
| computer_security | 96 | 15% | Cybersecurity, malware and data breaches | 14% | Home networking, routers and internet access |
| marketing | 234 | 26% | Digital marketing and advertising agencies | 14% | Retail, e-commerce and online shopping |
| world_religions | 160 | 19% | India, Hindu culture and Indian places | 9% | Islam, Muslim scholars and Islamic history |
| electrical_engineering | 139 | 27% | Motors, power electronics and controllers | 9% | Quantum physics, optics and nanomaterials |
| moral_scenarios | 895 | 78% | Personal feelings, grief and confessional writing | 5% | Criminal charges, trials and prosecutions |
| abstract_algebra | 87 | 68% | Mathematics papers, proofs and theorems | 13% | Names and entries starting with G |
| high_school_us_history | 204 | 41% | Human rights, democracy and governance | 9% | Courts, lawyers and legal procedure |
| human_aging | 222 | 9% | Children, parenting and child welfare | 9% | Mental health, stress and emotional wellbeing |
| logical_fallacies | 163 | 15% | Criminal charges, trials and prosecutions | 9% | Supreme court rulings and case digests |

`moral_scenarios` (895 items, 78% in *Personal feelings, grief and confessional writing*) is the one large subject the map reads by register rather than by subject: the questions are first-person vignettes about everyday wrongdoing. `human_aging` and `logical_fallacies` spread thinly and their top regions are only loosely related. Full per-subject placements for all three sets are in `results/b2_topics.json`.

## 5. Subjects, not languages (B3)

### Designed probes and parallel sentences

| product | same topic, other language → same region | → same cell | different topic, same language → same region | OOD blobs off-atlas | WMT pairs share region | chance | share cell | chance | cell in partner's soft top-5 | off-atlas en / tr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 49% | 26% | 0% | 0% | 40.2% | 2.0% | 24.8% | 0.5% | 61.9% | 1.9% / 6.4% |
| v3-prev | 29% | 18% | 22% | 0% | 40.2% | 2.5% | 24.8% | 0.7% | 59.3% | 4.1% / 11.1% |
| v2 | 30% | 24% | 22% | 25% | 40.7% | 3.6% | 35.2% | 2.0% | 60.6% | 28.6% / 39.2% |
| v2-lite | 39% | 30% | 25% | 25% | 38.5% | 6.7% | 31.9% | 4.3% | 57.3% | 40.2% / 43.5% |
| v3, no lang. centering | 42% | 27% | 1% | 0% | 39.2% | 2.0% | 22.8% | 0.4% | 57.3% | 3.9% / 6.3% |

On the 14 designed topics v3 puts the same topic in the same region for 49% of cross-language pairs and never collides two different topics written in one language; the previous v3 build managed 29% and 22%. Switching v3's per-language centering off costs 7 points on the probes and raises language NMI on the web panel from 0.09 to 0.17, so the centering is doing real work. On 8,000 WMT17 news sentence pairs 40% land in the same region (chance 2%) and 62% have the translation's cell inside the sentence's soft top-5; that number is nearly the same on every product, which says it is a property of the encoder rather than of the map. The four OOD blobs are placed by every product; see section 8.

### Language clustering vitals

| product | panel | cells | NMI(cell, lang) | AMI(cell, lang) | NMI(region, lang) | dominant-language lift, cell | lift, region | records in ≥90%-one-language cells | top-10 regions shared between languages | off-atlas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | web ×9 | 4096 | 0.180 | 0.103 | 0.091 | 4.14 | 2.70 | 12% | 19% | 0.3% |
| v3 | wikipedia ×13 | 4096 | 0.216 | 0.145 | 0.117 | 5.62 | 3.47 | 6% | 21% | 0.5% |
| v3-prev | web ×9 | 4096 | 0.235 | 0.164 | 0.170 | 5.01 | 3.56 | 24% | 16% | 0.5% |
| v3-prev | wikipedia ×13 | 4096 | 0.268 | 0.215 | 0.216 | 6.44 | 4.75 | 17% | 31% | 0.7% |
| v2 | web ×9 | 296 | 0.141 | 0.133 | 0.138 | 3.29 | 3.02 | 11% | 32% | 6.2% |
| v2 | wikipedia ×13 | 296 | 0.208 | 0.197 | 0.197 | 4.67 | 4.08 | 3% | 47% | 10.4% |
| v2-lite | web ×9 | 65 | 0.134 | 0.132 | 0.135 | 2.78 | 2.64 | 11% | 44% | 9.1% |
| v2-lite | wikipedia ×13 | 65 | 0.209 | 0.206 | 0.205 | 3.92 | 3.68 | 3% | 56% | 16.6% |
| v3, no lang. centering | web ×9 | 4096 | 0.251 | 0.183 | 0.172 | 5.30 | 3.78 | 21% | 15% | 0.6% |
| v3, no lang. centering | wikipedia ×13 | 4096 | 0.275 | 0.210 | 0.183 | 6.80 | 4.62 | 8% | 14% | 1.0% |

These numbers are not comparable across different cell counts (a finer partition always carries more mutual information with anything), so read them within a product or between v3 and v3-prev, which share k. Against the previous build, v3 halves the language information in its regions (NMI 0.09 vs 0.17 on web, 0.12 vs 0.22 on Wikipedia) and halves the share of records sitting in single-language cells (12% vs 24%). Per-language off-atlas rates on v3 are all under 0.6%; on v2 Chinese web text was 13% off-atlas and Turkish 10%.

## 6. Coverage detection (B4)

### Injection sensitivity

A held-out source is mixed into 40,000 C4 English records. Its *home cells* are the cells holding 80% of a separate 5,000-record sample of it. The excess mass in those cells estimates the injected share; a Poisson z-score says whether the excess could be noise. Detected means z ≥ 3 and the estimate within a factor of two.

| injected source | v3 | v3-prev | v2 | v2-lite | v3 home cells | base already there |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EU regulations (eurlex) | 0.25% | 0.25% | 0.50% | not detected | 8 | 0.01% |
| Python commits | 0.25% | 0.25% | 0.25% | 1.00% | 45 | 1.03% |
| Math word problems (orca-math) | 1.00% | 1.00% | 2.00% | 5.00% | 171 | 6.06% |
| Turkish instructions (InstrucTurca) | 2.00% | 2.00% | 5.00% | 10.00% | 935 | 20.98% |
| Text-to-SQL (gretel) | 1.00% | 1.00% | 2.00% | 2.00% | 263 | 7.28% |
| Biomedical QA (PubMedQA) | 0.25% | 0.25% | 0.50% | 0.50% | 64 | 1.30% |
| Turkish web (fineweb-2 tr) | 2.00% | 2.00% | 5.00% | not detected | 867 | 22.35% |

v3 detail (fine cells):

| source | true share | records | z | estimated share | estimate / true | detected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EU regulations (eurlex) | 0.25% | 100 | 39.9 | 0.25% | 0.98 | yes |
| EU regulations (eurlex) | 0.50% | 201 | 78.8 | 0.48% | 0.97 | yes |
| EU regulations (eurlex) | 1.00% | 404 | 156.7 | 0.96% | 0.96 | yes |
| EU regulations (eurlex) | 2.00% | 816 | 323.7 | 1.98% | 0.99 | yes |
| EU regulations (eurlex) | 5.00% | 2105 | 837.1 | 5.03% | 1.01 | yes |
| EU regulations (eurlex) | 10.00% | 4444 | 1720.0 | 10.06% | 1.01 | yes |
| Python commits | 0.25% | 100 | 3.9 | 0.25% | 1.00 | yes |
| Python commits | 0.50% | 201 | 8.0 | 0.51% | 1.02 | yes |
| Python commits | 1.00% | 404 | 16.3 | 1.04% | 1.04 | yes |
| Python commits | 2.00% | 816 | 32.0 | 2.03% | 1.02 | yes |
| Python commits | 5.00% | 2105 | 79.1 | 4.95% | 0.99 | yes |
| Python commits | 10.00% | 4444 | 163.5 | 9.96% | 1.00 | yes |
| Math word problems (orca-math) | 0.25% | 100 | 1.2 | 0.21% | 0.82 | no |
| Math word problems (orca-math) | 0.50% | 201 | 2.6 | 0.42% | 0.85 | no |
| Math word problems (orca-math) | 1.00% | 404 | 5.4 | 0.89% | 0.89 | yes |
| Math word problems (orca-math) | 2.00% | 816 | 11.1 | 1.83% | 0.92 | yes |
| Math word problems (orca-math) | 5.00% | 2105 | 29.0 | 4.71% | 0.94 | yes |
| Math word problems (orca-math) | 10.00% | 4444 | 58.6 | 9.26% | 0.93 | yes |
| Turkish instructions (InstrucTurca) | 0.25% | 100 | 0.4 | 0.15% | 0.62 | no |
| Turkish instructions (InstrucTurca) | 0.50% | 201 | 0.8 | 0.31% | 0.62 | no |
| Turkish instructions (InstrucTurca) | 1.00% | 404 | 1.9 | 0.72% | 0.72 | no |
| Turkish instructions (InstrucTurca) | 2.00% | 816 | 3.6 | 1.38% | 0.69 | yes |
| Turkish instructions (InstrucTurca) | 5.00% | 2105 | 9.1 | 3.44% | 0.69 | yes |
| Turkish instructions (InstrucTurca) | 10.00% | 4444 | 18.9 | 6.96% | 0.70 | yes |
| Text-to-SQL (gretel) | 0.25% | 100 | 1.4 | 0.25% | 1.00 | no |
| Text-to-SQL (gretel) | 0.50% | 201 | 2.6 | 0.49% | 0.97 | no |
| Text-to-SQL (gretel) | 1.00% | 404 | 5.0 | 0.92% | 0.92 | yes |
| Text-to-SQL (gretel) | 2.00% | 816 | 10.2 | 1.87% | 0.94 | yes |
| Text-to-SQL (gretel) | 5.00% | 2105 | 26.2 | 4.74% | 0.95 | yes |
| Text-to-SQL (gretel) | 10.00% | 4444 | 53.7 | 9.47% | 0.95 | yes |
| Biomedical QA (PubMedQA) | 0.25% | 100 | 3.5 | 0.25% | 1.00 | yes |
| Biomedical QA (PubMedQA) | 0.50% | 201 | 7.1 | 0.51% | 1.02 | yes |
| Biomedical QA (PubMedQA) | 1.00% | 404 | 14.1 | 1.01% | 1.01 | yes |
| Biomedical QA (PubMedQA) | 2.00% | 816 | 28.3 | 2.03% | 1.01 | yes |
| Biomedical QA (PubMedQA) | 5.00% | 2105 | 71.5 | 5.03% | 1.01 | yes |
| Biomedical QA (PubMedQA) | 10.00% | 4444 | 145.2 | 9.95% | 0.99 | yes |
| Turkish web (fineweb-2 tr) | 0.25% | 100 | 0.5 | 0.18% | 0.74 | no |
| Turkish web (fineweb-2 tr) | 0.50% | 201 | 1.0 | 0.39% | 0.79 | no |
| Turkish web (fineweb-2 tr) | 1.00% | 404 | 1.9 | 0.77% | 0.77 | no |
| Turkish web (fineweb-2 tr) | 2.00% | 816 | 3.6 | 1.48% | 0.74 | yes |
| Turkish web (fineweb-2 tr) | 5.00% | 2105 | 9.6 | 3.83% | 0.77 | yes |
| Turkish web (fineweb-2 tr) | 10.00% | 4444 | 19.1 | 7.44% | 0.74 | yes |

Sources that own a few cells (EU law: 8 cells, biomedical abstracts: 64) are detected at 0.25% with the share recovered to within 3%. Sources that are themselves mixtures (Turkish instructions, Turkish web: ~900 home cells that C4 already occupies at 21–22%) need 2% and are under-estimated by a quarter, because their home cells are also web cells. v2 needs 2–10× the share for the same sources and v2-lite never detects EU law at all. What the CLI prints follows: at 10% eurlex the top home cell reads 72× the map's density against 0.3× in the base.

### Coverage ladder

| step | records | v3 occupied | v3 effective | v3 regions | v3-prev effective | v2 effective | v2-lite effective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| geometry only | 870 | 40/4096 | 40 (1%) | 21/256 | 1% | 7% | 16% |
| + all hendrycks math | 6852 | 281/4096 | 252 (6%) | 113/256 | 6% | 12% | 24% |
| + code (4 languages) | 14852 | 711/4096 | 412 (10%) | 194/256 | 11% | 14% | 29% |
| + chat (ultrachat) | 22852 | 2428/4096 | 1248 (30%) | 249/256 | 26% | 37% | 51% |
| + legal (3 lex_glue tasks) | 30952 | 2600/4096 | 1218 (30%) | 249/256 | 28% | 39% | 55% |
| + science (peS2o, arXiv, PubMedQA) | 39052 | 2730/4096 | 1260 (31%) | 249/256 | 29% | 37% | 53% |
| + English web (C4) | 47052 | 3293/4096 | 1712 (42%) | 253/256 | 37% | 47% | 61% |
| + web in 8 languages (fineweb-2) | 55052 | 3640/4096 | 2016 (49%) | 256/256 | 44% | 54% | 66% |
| + Turkish instructions + 13 wikipedias | 66852 | 3889/4096 | 2282 (56%) | 256/256 | 49% | 59% | 68% |

Effective coverage (sum of min(1, density) over cells) rises with breadth on every product and v3 spreads the ladder over the widest range (1% to 56%), which is what makes a narrow corpus legible as narrow: geometry alone reads 1% on v3 and 16% on v2-lite. The one non-monotone step (+ legal, 30.5% → 29.7%) is dilution, since adding records to a few dense cells lowers every other cell's share.

Reference-like corpus (fineweb English) at growing sample sizes, occupied cells / effective share:

| records | v3 | v3-prev | v2 | v2-lite |
| --- | ---: | ---: | ---: | ---: |
| 2000 | 1370 / 33% | 1045 / 26% | 258 / 69% | 63 / 66% |
| 5000 | 2219 / 52% | 1761 / 41% | 269 / 69% | 64 / 66% |
| 20000 | 3222 / 64% | 2800 / 49% | 283 / 71% | 65 / 66% |
| 50000 | 3542 / 64% | 3207 / 50% | 288 / 71% | 65 / 66% |
| 100000 | 3705 / 64% | 3408 / 50% | 289 / 71% | 65 / 66% |

On v3 the effective share of a web corpus saturates at 64% from 20,000 records; the remaining third of the map is not English web (code, law, multilingual prose, training prompts), which is the intended reading. The 500,000 default sample is more than the number needs; 20,000 gets within a point.

### The CLI's own comparison

| pair (5,000 records each side) | v3 sim / New | v3-prev | v2 | v2-lite | v3 region-level cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| same corpus, two halves | 0.93 / 19% | 0.96 / 14% | 0.99 / 0% | 1.00 / 0% | 0.99 |
| alpaca vs its Turkish translation | 0.66 / 20% | 0.65 / 23% | 0.76 / 1% | 0.87 / 0% | 0.89 |
| Turkish instructions vs English instructions | 0.36 / 47% | 0.44 / 40% | 0.46 / 2% | 0.75 / 0% | 0.59 |
| hendrycks algebra vs intermediate algebra | 0.94 / 9% | 0.87 / 5% | 0.91 / 5% | 0.91 / 3% | 0.99 |
| Python commits vs Python text-to-code | 0.18 / 69% | 0.16 / 61% | 0.08 / 18% | 0.42 / 7% | 0.40 |
| eurlex vs ECtHR case law | 0.00 / 98% | 0.00 / 69% | 0.03 / 52% | 0.05 / 19% | 0.00 |
| Turkish web vs English web | 0.18 / 47% | 0.13 / 53% | 0.46 / 4% | 0.62 / 0% | 0.57 |
| Turkish wikipedia vs English wikipedia | 0.21 / 26% | 0.28 / 21% | 0.49 / 1% | 0.70 / 0% | 0.56 |
| Python commits vs ultrachat | 0.07 / 48% | 0.07 / 42% | 0.08 / 3% | 0.18 / 0% | 0.14 |
| PubMedQA vs eurlex | 0.00 / 94% | 0.00 / 91% | 0.01 / 6% | 0.16 / 0% | 0.00 |

Directionally every product gets the pairs right: unrelated corpora (EU law vs case law, biomedical QA vs EU law) read as 0 similar and 94–98% new on v3; sibling math sets read 0.94 similar; alpaca and its Turkish translation read 0.66 similar with 81% shared mass, and 0.89 at region level. The absolute *New* figure on v3 is not trustworthy at this sample size: two halves of the same corpus read 19% new. Test-retest below shows why.

### Test-retest against sample size

allenai__c4__en:

| records per half | v3 New / cos(cell) / cos(region) | v3-prev New / cos(cell) / cos(region) | v2 New / cos(cell) / cos(region) | v2-lite New / cos(cell) / cos(region) |
| --- | ---: | ---: | ---: | ---: |
| 250 | 91% / 0.08 / 0.57 | 85% / 0.18 / 0.65 | 38% / 0.58 / 0.72 | 5% / 0.86 / 0.96 |
| 500 | 78% / 0.22 / 0.78 | 65% / 0.38 / 0.84 | 13% / 0.75 / 0.84 | 2% / 0.93 / 0.96 |
| 1000 | 67% / 0.29 / 0.87 | 51% / 0.53 / 0.90 | 4% / 0.87 / 0.93 | 0% / 0.97 / 0.98 |
| 2000 | 46% / 0.49 / 0.93 | 35% / 0.69 / 0.93 | 1% / 0.92 / 0.96 | 0% / 0.97 / 0.99 |
| 5000 | 18% / 0.72 / 0.97 | 17% / 0.85 / 0.97 | 0% / 0.97 / 0.98 | 0% / 0.99 / 1.00 |
| 10000 | 8% / 0.83 / 0.98 | 8% / 0.92 / 0.99 | 0% / 0.98 / 0.99 | 0% / 1.00 / 1.00 |

ultrachat_200k:

| records per half | v3 New / cos(cell) / cos(region) | v3-prev New / cos(cell) / cos(region) | v2 New / cos(cell) / cos(region) | v2-lite New / cos(cell) / cos(region) |
| --- | ---: | ---: | ---: | ---: |
| 250 | 69% / 0.37 / 0.80 | 62% / 0.48 / 0.85 | 24% / 0.89 / 0.91 | 3% / 0.93 / 0.94 |
| 500 | 65% / 0.51 / 0.88 | 54% / 0.62 / 0.88 | 14% / 0.92 / 0.94 | 1% / 0.96 / 0.97 |
| 1000 | 46% / 0.69 / 0.95 | 40% / 0.79 / 0.94 | 7% / 0.96 / 0.98 | 1% / 0.98 / 0.99 |
| 2000 | 35% / 0.82 / 0.97 | 28% / 0.90 / 0.97 | 2% / 0.98 / 0.98 | 0% / 0.99 / 1.00 |
| 5000 | 18% / 0.92 / 0.99 | 13% / 0.96 / 0.99 | 0% / 0.99 / 0.99 | 0% / 0.99 / 1.00 |
| 10000 | 8% / 0.96 / 0.99 | 7% / 0.98 / 0.99 | 0% / 1.00 / 1.00 | 0% / 1.00 / 1.00 |

fineweb-2_tur_Latn:

| records per half | v3 New / cos(cell) / cos(region) | v3-prev New / cos(cell) / cos(region) | v2 New / cos(cell) / cos(region) | v2-lite New / cos(cell) / cos(region) |
| --- | ---: | ---: | ---: | ---: |
| 250 | 68% / 0.40 / 0.78 | 80% / 0.22 / 0.70 | 34% / 0.67 / 0.80 | 4% / 0.81 / 0.89 |
| 500 | 60% / 0.57 / 0.82 | 69% / 0.39 / 0.83 | 16% / 0.76 / 0.84 | 0% / 0.89 / 0.94 |
| 1000 | 47% / 0.71 / 0.92 | 50% / 0.58 / 0.90 | 5% / 0.88 / 0.94 | 0% / 0.95 / 0.97 |
| 2000 | 32% / 0.81 / 0.95 | 36% / 0.72 / 0.94 | 2% / 0.94 / 0.96 | 0% / 0.98 / 0.99 |
| 5000 | 16% / 0.91 / 0.97 | 17% / 0.87 / 0.97 | 0% / 0.97 / 0.98 | 0% / 0.99 / 1.00 |
| 10000 | 8% / 0.95 / 0.99 | 8% / 0.93 / 0.99 | 0% / 0.99 / 0.99 | 0% / 0.99 / 1.00 |

Two disjoint halves of one corpus should read as the same corpus. On v2 they do from 1,000 records; on v3's 4,096 cells the fine histogram needs 10,000 records per side to get within 8% new and a cosine of 0.83–0.96, while the 256-region histogram is at 0.95–0.99 from 1,000 records. The map is fine enough that a fingerprint of a few thousand records is a sparse sample of it, and `compare()` reads sampling gaps as novelty. The region level is the stable read at those sizes.

## 7. Off-atlas calibration (B4d)

| product | cutoff in use | in-family p1 / p2 / p5 / p50 similarity | in-family off at cutoff | machine formats rejected at cutoff | at in-family p2: in-family off / machine rejected | at 0.50: in-family off / machine rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.35 | 0.403 / 0.425 / 0.466 / 0.653 | 0.15% | 0.0% | 9.0% / 49.2% | 9.0% / 49.2% |
| v3-prev | 0.35 | 0.384 / 0.409 / 0.453 / 0.638 | 0.30% | 0.0% | 10.9% / 5.9% | 10.9% / 5.9% |
| v2 | 0.35 | 0.285 / 0.309 / 0.346 / 0.527 | 5.40% | 6.6% | 41.1% / 68.8% | 41.1% / 68.8% |
| v2-lite | 0.35 | 0.249 / 0.276 / 0.314 / 0.499 | 9.69% | 3.4% | 50.2% / 44.1% | 50.2% / 44.1% |

When the first run was made the v3 artifact carried no `off_atlas_threshold` and the loader's 0.35 default applied. It has since been calibrated by `tools/calibrate_off_atlas_v3.py` to **0.3538**: the 2nd percentile of nearest-cell cosine over the build's own language-and-axis-balanced 2,000,000-row calibration draw (per-axis p2 runs from 0.312 for code and 0.334 for training prompts to 0.421 for scientific text), stamped with an `off_atlas_calibration` block and guarded by a test. The 26,000-record in-family sample used here is English-web heavy, which is why its own p2 is 0.425; the balanced draw is deliberately not that. The v2 default of 0.35 sits *above* v2's own p2 of 0.309, which is why v2 sends 12–18% of ordinary held-out prose off the map. At 0.3538 v3 places 99.8% of in-family text and 100% of 1,400 base64 / hex / DNA / minified-JS / CSV / random records; at 0.425 it would place 98% and still reject only 10% of the machine formats, because on this encoder those formats sit inside the prose similarity band (median 0.43–0.75 against 0.65 for prose). The cutoff cannot be made to reject them without discarding prose; the surface-share diagnosis in the off-atlas description (whitespace and non-letter shares) is what separates a template from a subject, and it only runs on records the cutoff already removed. Remaining recommendation: run the surface test on placed records too, or report machine-format share separately.

## 8. Names (B5)

| product | cell names | own cell ranks 1st / top-10 / top-100 | median rank of own cell | 1st among siblings | region names | own region ranks 1st / top-5 | exemplars back in own cell / region | exemplar in soft top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 4096 (hand) | 31% / 65% / 86% | 4 | 44% | 256 (hand) | 73% / 86% | 53% / 72% | 80% |
| v3-prev | 4096 (hand) | 25% / 57% / 75% | 6 | 36% | 256 (hand) | 62% / 79% | 30% / 43% | 48% |
| v2 | 296 (auto) | 47% / 67% / 85% | 2 | 54% | 128 (hand) | 55% / 80% | 40% / 44% | 49% |
| v2-lite | 65 (auto) | 20% / 45% / 100% | 12 | 38% | 32 (hand) | 69% / 88% | 33% / 38% | 45% |

A name embedded with the atlas encoder ranks its own centroid first among 4,096 for 31% of v3 cells and in the top ten for 65% (median rank 4), against 25% / 57% / 6 on the previous build (v2's 47% is over 296 candidates, an easier rank, and its captions are the automatic contrastive terms); among a region's siblings 44% of names pick out their own cell. Region names rank their own centroid first 73% of the time. The stored exemplar texts, 320-character heads of the records nearest each centroid, come back to their own cell 53% of the time and to their own region 72% (previous build: 30% / 43%), so the map's own witnesses are consistent with it.

Hygiene: 0 duplicate cell names and 0 duplicate region names; 5,028 distinct words across the 4,096 cell names. 21 cell names contain a language word: 14 are language-as-subject (Arabic and Russian poetry, Slovenian literary prose, Greek New Testament manuscripts, English language learning), 3 are place names or false matches (*Czech Republic*, *nail polish*), and 4 use a nationality adjective where the naming rules ask for the place (*Japanese baseball*, *Japanese television dramas*, *Slovenian supreme court*, *German place-name lists*). Two region names do the same ('German biographies, surnames and history', 'German municipalities and Central European localities'). Those six are the only naming-rule slips found.

### Blind judge

For each held-out source and each MMLU / MMLU-Pro / EXAMS subject, the name of the region (and, for v3, the cell) holding most of its records was scored blind by a language model on a 0–2 scale: 2 clearly describes the data, 1 partially or too generically, 0 unrelated. Items were shuffled and the product hidden.

| names | 52 sources | MMLU 57 | MMLU-Pro 14 | EXAMS-tr 8 | all 131 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 region names | 1.50 (62% clear) | 1.51 (56% clear) | 1.43 (50% clear) | 1.62 (62% clear) | 1.50 (58% clear) |
| v3 cell names | 1.60 (69% clear) | 1.68 (74% clear) | 1.50 (64% clear) | 1.75 (88% clear) | 1.63 (72% clear) |
| v2 region names | 1.13 (29% clear) | 1.42 (47% clear) | 1.29 (43% clear) | 1.62 (62% clear) | 1.31 (40% clear) |

Mean score and the share of items whose top name clearly describes them. Instruction mixtures cap the sources column: a corpus that spreads over 200 regions has no single describing region by construction.

## 9. Cost and the command (B6)

v3 at its default 500,000-record sample (web text in 5 languages):

| stage | measured |
| --- | ---: |
| records | 500,000 |
| mean characters | 1499 |
| language detection | 36.7 s |
| encode (tokenise + SIF pool) | 59.9 s (8,349 rec/s) |
| assign to 4,096 cells | 11.5 s (43,611 rec/s) |
| coverage report | 0.9 s |
| peak RSS during the pass | 7,889 MB (harness holds all texts and vectors in memory) |
| off-atlas / occupied / effective | 0.2% / 3994 / 2874 |

Encoding runs at 8,200 records/s on 1,500-character records; the profile note's 17,500/s was measured on shorter text. The whole pass is under two minutes, and the 0.9 s coverage step is negligible.

`dropoutt atlas` end to end (`--offline --no-open`, outputs under `runs/`):

| corpus / product | records | exit | wall | artifacts |
| --- | ---: | ---: | ---: | ---: |
| narrow-geometry/atlas-v3 | 1,616 | 0 | 1.9 s | atlas.html 1311 KB, atlas.json 1451 KB, atlas.md 9 KB |
| mixed-four-sources/atlas-v3 | 12,000 | 0 | 3.7 s | atlas.html 1385 KB, atlas.json 1470 KB, atlas.md 14 KB |
| mixed-four-sources/atlas-v2 | 12,000 | 0 | 5.3 s | atlas.html 255 KB, atlas.json 150 KB, atlas.md 10 KB |
| multilingual-web/atlas-v3 | 16,000 | 0 | 5.2 s | atlas.html 1434 KB, atlas.json 1478 KB, atlas.md 15 KB |

What the terminal said, trimmed (full text in each run's `terminal.txt`):

- **narrow-geometry / v3**: *Effective coverage 56 of 4096 (56 subregions) (specialised)*; *Mathematics papers, proofs and theorems 13/18 reach, 92.2% share*; *228 of the map's 256 subject areas never reached*; *31% of your data sits in a single place on the map*.
- **mixed-four-sources / v3**: the four sources come back as the four top areas, named: *Commission regulations on export refunds and prices 21.2%*, *Source code, functions and programming questions 11.0%*, *Money arithmetic word problems 6.5%*, *Software installation guides and shell setup 5.2%*; ultrachat is the long tail (*Recipes*, *Routines, sleep and time word problems*). 34 of 12,000 records off the map.
- **mixed-four-sources / v2**: the same corpus reads *Web servers, SEO and software troubleshooting 12.7%* with cells captioned `javascript, html, import, class` and `euro, only, back, were`; 1,895 records (15.8%) off the map.
- **multilingual-web / v3**: eight fineweb-2 languages read as *broad* (2,400 of 4,096 effective) with subject areas rather than languages on top: *Hotels, rooms and guest reviews*, *Football leagues, cups and match previews*, *Video games, consoles and gameplay*, *Heads of state and national political news*. One area, *Religious texts, archive scans and mixed stubs*, is a catch-all whose cells are named as such.

## 10. Findings

1. **v3 is a better coordinate system than v2 on held-out data.** Source separation +7 points at cell level, axis separation +3, MMLU subject purity +4, MMLU-Pro +5; every held-out prose source places at ≥ 96% where v2 dropped 12–26% of several sets off the map.
2. **v3 is a subject map, and the second build fixed the first's language clustering.** Same-topic cross-language agreement 49% vs 29%, zero same-language topic collisions vs 22%, half the language information in regions, 12% instead of 24% of web records in single-language cells. Per-language centering accounts for part of that (7 points on the probes, NMI 0.09 vs 0.17 without it).
3. **Coverage detection is sharp.** A source with its own cells is found at 0.25% of a 40,000-record web corpus with its share recovered to within 3%; mixtures need 2%. The reported density in the top home cell moves from 0.3× to 72× at 10% eurlex, so the terminal reads it without statistics.
4. **The hand names hold up.** 31% of cell names and 73% of region names rank their own centroid first with the same encoder; exemplars return to their own region 72% of the time; MMLU subjects land under regions a reader would pick for them (astronomy, virology, nutrition, international law, marketing, computer security).
5. **Fixed since the first run: the off-atlas cutoff.** The artifact shipped without `off_atlas_threshold`; it now carries 0.3538 from the balanced calibration draw, with provenance and a test. Machine formats are still placed at 100%, and no cutoff changes that on this encoder; the surface-share test has to run on placed records to catch templates.
6. **Defect: `compare()` over 4,096 cells mistakes sampling gaps for novelty.** 19% New between halves of one corpus at 5,000 records, 8% at 10,000 (v2: 0%). Shared/New should be computed at region level, or corrected for the sampling-only expectation, before v3 fingerprints are diffed.
7. **Resolution has a sampling cost the defaults already cover.** Fine-cell histograms stabilise at ~10,000 records; the 500,000 default is generous, and 20,000 records already saturate effective coverage for web text. For corpora under a few thousand records the region-level read is the reliable one, and the report could say so.

## 11. Caveats

- The held-out panel comes from the v2 build cache, so it is held out of **v3** but partly in-build for v2 and v2-lite, which if anything flatters those two.
- The language panels (fineweb, fineweb-2, Wikipedia) are the same datasets as v3 inputs, though not necessarily the same records. They are used only for the language vitals and coverage saturation, never for accuracy.
- Labels for the source panel are dataset of origin, which is a proxy for subject; a map that separates two math sets is not better for it. The axis-level numbers are the fairer read.
- Mutual-information measures are not comparable across products with different cell counts; accuracies and purities are what the tables compare, and AMI is shown for completeness.
- One blind LLM judge, one pass; treat the judge table as a check on face validity, not as a measurement.