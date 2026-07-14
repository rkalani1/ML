# Originality / Copyright Risk Audit — ML Web Edition

**Auditor role:** Originality / copyright risk (free educational GitHub Pages content)  
**Target:** `C:\Users\rkala\ML\docs\` especially `docs/curriculum/*.md`  
**Date:** 2026-07-13  
**Scope:** Full curriculum + site front-matter + assets  

---

## Executive summary

**Overall: PASS with residual MEDIUM syllabus-alignment notes. No clear HIGH-risk paste found; no in-place rewrites performed.**

The Web Edition curriculum is consistently framed as original teaching prose for neurologists/epidemiologists. Hard risk markers (ISBN, © boilerplate, “reproduced with permission,” commercial-book figure credits, paper abstracts, publisher image embeds) are **absent**. All 21 non-index curriculum files carry the Web Edition disclaimer banner and a clinical frame. `docs/assets/figures/` is empty; **0** markdown/HTML image embeds ship on the site (aligned with the ORIGINALITY manifest’s skip of 129 DOCX-embedded images).

---

## Chapters reviewed (21 + index)

| # | File | Title / role | Disclaimer | Clinical frame | Verdict |
|---|------|--------------|------------|----------------|---------|
| 1 | `00a-preface.md` | Preface | Yes | Yes | **pass** |
| 2 | `00b-how-to-use-this-textbook.md` | How to use | Yes | Yes | **pass** |
| 3 | `00-mathematical-foundations-for-machine-learning.md` | Ch 0 Math foundations | Yes | Yes | **pass** (long pure-math pedagogy; original voice) |
| 4 | `01-basic-concepts-of-machine-learning-and-artificial-intelligence.md` | Ch 1 Basics | Yes | Yes | **pass** |
| 5 | `02-visualization.md` | Ch 2 Visualization | Yes | Yes | **pass** |
| 6 | `03-probability-and-statistics.md` | Ch 3 Probability | Yes | Yes | **pass** |
| 7 | `04-clustering.md` | Ch 4 Clustering | Yes | Yes | **pass** |
| 8 | `05-frequent-itemset-mining-sequence-mining-and-information-retrieval.md` | Ch 5 Mining / IR | Yes | Yes | **pass** |
| 9 | `06-feature-engineering.md` | Ch 6 Features | Yes | Yes | **pass** |
| 10 | `07-dimensionality-reduction-and-data-decomposition.md` | Ch 7 Dim reduction | Yes | Yes | **pass** |
| 11 | `08-regression-analysis.md` | Ch 8 Regression | Yes | Yes | **pass** |
| 12 | `09-classification.md` | Ch 9 Classification | Yes | Yes | **pass** |
| 13 | `10-neural-networks-and-deep-learning.md` | Ch 10 Neural nets | Yes | Yes | **pass** |
| 14 | `11-self-supervised-deep-learning.md` | Ch 11 Self-supervised | Yes | Yes | **pass** |
| 15 | `12-deep-learning-models-and-applications-for-text-vision-and-audio.md` | Ch 12 Multimodal | Yes | Yes | **pass** |
| 16 | `13-reinforcement-learning.md` | Ch 13 RL | Yes | Yes | **pass** |
| 17 | `14-making-lighter-neural-network-and-machine-learning-models.md` | Ch 14 Efficient models | Yes | Yes | **pass** |
| 18 | `15-graph-mining-algorithms.md` | Ch 15 Graphs | Yes | Yes | **pass** |
| 19 | `16-concepts-and-challenges-of-working-with-data.md` | Ch 16 Data challenges | Yes | Yes | **pass** |
| 20 | `17-closing-synthesis-senior-practice.md` | Closing synthesis | Yes | Yes | **pass** |
| 21 | `18-selected-glossary.md` | Glossary | Yes | Yes | **pass** |
| — | `index.md` | TOC only | N/A | N/A | **pass** (navigation) |

Site pages also reviewed: `index.md`, `about.md`, `how-to-use.md`, `ORIGINALITY.md`, `update-protocol.md`.

---

## Risk findings

### HIGH (clear copyright / paste risk)

**Count: 0**

Searched and not found:

| Pattern | Result |
|---------|--------|
| ISBN / © / “All rights reserved” / permission lines | **0 hits** in curriculum |
| “adapted from / reproduced from / with permission” book figures | **0 hits** |
| Commercial book names (ESL, ISL, PRML, Goodfellow DL, Hands-On ML, Géron, Bishop, Hastie, Murphy textbooks as source text) | **0 attribution-as-source hits** |
| Paper-style abstracts (Background/Methods/Results/Conclusions blocks; arXiv/DOI dumps; “In this paper we propose…”) | **Not present as content blocks** |
| Long block quotes of third-party expressive prose | **None** |
| Publisher result tables / guideline instrument dumps | **None found** |
| Markdown/HTML image embeds of book/paper figures | **0 embeds site-wide** |

**Action:** No passage required in-place rewrite for HIGH paste risk.

---

### MEDIUM (structural / style residual risk)

**Count: 3**

#### M1 — Syllabus / TOC aligned to an external curriculum name (disclosed)

- **Where:** `00a-preface.md` states topical mastery aligned with the “full Rawassizadeh curriculum” and “published TOC.”
- **Chapter titles** form a comprehensive ML course outline (foundations → unsupervised → features → supervised → deep/SSL → multimodal → RL → efficient models → graphs → data challenges).
- **Risk type:** Outline similarity to a named commercial/academic syllabus is **topic coverage**, not wording plagiarism. Preface asserts “Exposition is original; the syllabus is comprehensive.”
- **Mitigation already present:** Explicit originality claim; clinical reframing throughout; no book prose attribution.
- **Recommendation:** Keep the disclosure. Optionally soft-name “a standard graduate ML topic sequence” in future editions if brand association is unwanted; **not** a paste fix.

#### M2 — Dense CS/algorithm pedagogy without clinical tokens in isolated paragraphs

- **Where (examples):** distance/k-means/DBSCAN algorithm blocks in Ch 4; Naive Bayes / CART / LightGBM blocks in Ch 9; Huffman/FLOPs/quantization exposition in Ch 14; data-structure survey (BST, B-tree, trie) in Ch 5; large pure-math development in Ch 0.
- **Risk type:** Long technical passages *feel* textbook-like because they teach standard methods. Spot-check shows **original teaching voice** (worked synthetic numbers, stroke vignettes at chapter level, original figure captions with clinical toy data). Not wholesale definitions lifted from ESL/PRML/Goodfellow.
- **Recommendation:** No rewrite required for copyright. Optional future polish: add one clinical “so what” sentence to the densest CS blocks for pedagogical consistency (not a legal necessity).

#### M3 — Figure captions without shipped figure files

- **Where:** ~111 caption lines of the form `Figure X.Y. …` across chapters; **0** `![...](...)` or `<img>` embeds; `docs/assets/figures/` empty.
- **Risk type:** **Low copyright risk** (no publisher images published). Residual product risk: readers see captions without art; captions describe **synthetic, curriculum-original** scenes (LVO screen PPV curves, toy k-means, EEG HMM, stroke dashboards)—not “adapted from Figure N of [book].”
- **Recommendation:** Keep omission until original code-drawn SVG/PNG are ready (matches `ORIGINALITY.md` figure backlog). Do **not** paste DOCX/publisher figures to fill gaps.

---

### LOW (expected / non-expressive or short standard definitions)

**Count: 4 categories**

| ID | Item | Notes |
|----|------|-------|
| L1 | Glossary one-liners | Short operational definitions of standard ML terms; not book-length expressive prose. |
| L2 | Shared formulas / algorithm names | Soft-margin SVM, ELBO, TF–IDF, PageRank, etc.—ideas/facts not protected as expression; exposition is re-taught with original worked examples. |
| L3 | “Figure X.Y” numbering | Internal pedagogy labels; not third-party figure credits. |
| L4 | Method family names in headings | e.g. REINFORCE/TRPO/PPO, XGBoost—standard field vocabulary. |

---

### PASS (positive controls confirmed)

| Control | Status |
|---------|--------|
| Web Edition disclaimer banner on every curriculum chapter (excl. index) | **Yes** |
| “Web Edition clinical frame” on every curriculum chapter (excl. index) | **Yes** |
| Stroke / neurology / epi framing dense in clinical notes sections | **Yes** |
| Site home + about originality bar + educational disclaimer | **Yes** (`index.md`, `about.md`) |
| No publisher-looking images under `docs/assets` | **Yes** (figures dir empty; no raster/SVG under docs) |
| DOCX image skip policy documented | **Yes** (`ORIGINALITY.md`: 129 DOCX images not published) |
| Synthetic vignettes / worked numbers | **Yes** (e.g. NIHSS–volume OLS, LVO Naive Bayes table, EEG HMM) |
| Closing synthesis is original senior clinical practice walkthrough | **Yes** |

---

## Images / assets status

| Check | Result |
|-------|--------|
| `docs/assets/figures/` | **Empty** (no files) |
| Image files under `docs/` (`png/jpg/svg/gif/webp`) | **0** |
| Markdown image embeds in curriculum | **0** |
| HTML `<img>` in docs | **0** |
| Figure *captions* (text only) | ~**111** — original synthetic descriptions |
| Evidence of lifted publisher figures | **None** |

**images_status:** `clean_no_publisher_assets` — skipped DOCX figures were **not** replaced by lifted images.

---

## Method notes (audit process)

1. Full file inventory of `docs/` and `curriculum/`.
2. Regex sweeps: ISBN, copyright, permission, adapted/reproduced, commercial textbook titles, abstract/paper markers, arXiv/DOI, publisher names as content sources.
3. Structural audit: disclaimer + clinical frame presence on all chapters.
4. Dense reading samples: openings, mid-chapter cores, clinical notes, chapter summaries, glossary, closing synthesis; Ch 0 sample of pure-math sections and practice set.
5. Clinical-keyword density on long paragraphs (residual non-clinical blocks are algorithm pseudocode / pure math—not foreign abstracts).
6. Assets scan: no image payloads.

**Limitation:** Automated + human review cannot mathematically prove global non-similarity to every commercial textbook. No HIGH expressive paste was identified. Prefer continued spot-checks before treating the site as final (as already noted in `ORIGINALITY.md`).

---

## Fixes applied this audit

**None.** No HIGH-risk paste passage required rewrite.

---

## Counts (for JSON consumers)

| Bucket | Count |
|--------|------:|
| chapters_reviewed | 21 |
| high | 0 |
| medium | 3 |
| low | 4 |
| pass | 22 (21 chapters + index) |
| rewrites | 0 |

---

## Top findings (ranked)

1. **PASS / images:** No publisher or DOCX-derived images published; captions-only figures reduce copyright exposure.  
2. **PASS / frames:** Universal Web Edition disclaimer + clinical frame across curriculum.  
3. **MEDIUM:** Explicit Rawassizadeh TOC/syllabus alignment—coverage not wording.  
4. **MEDIUM:** Isolated dense CS/math blocks read textbook-like but appear original.  
5. **HIGH:** None.
