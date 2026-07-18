# ML & AI for Neurologists — content, design, and accuracy review (2026-07-18)

Reviewer pass over the open-source MkDocs ebook (teal Material) for technical accuracy,
currency, minimalism, design, originality boundaries, and quiet machine-consumability.
Method: firsthand read of the landing, evidence register, preface, and high-traffic
chapters (01, 09, 16), plus a 22-agent fan-out (16 surface reviewers → adversarial
verification of every P0/P1 → 8 primary-source currency checks) whose findings were
re-verified before any edit. This note is an internal audit record; it is **not** in the
public nav.

## 1.1 Inventory

- **Nav / IA:** Introduction + Evidence register + Book (Preface, Ch 00–18). 20 curriculum
  files, ~8,800 lines. `use_directory_urls: false` → internal links are `.html`.
- **Shared patterns:** `## Opening` decision-hook + legacy figure pattern
  (`![alt]` + blank + `*caption*`) that `postprocess_site.py` converts to semantic
  `<figure>/<figcaption>`; teal palette; per-chapter `Clinical and Epidemiologic Notes`,
  `Chapter Summary`, `Practice and Reflection`.
- **Figures + manifest:** was 70 assets; now 69 after removing one redundant opening
  diagram. `manifest.json` records sha256/bytes/dims/software per asset with an explicit
  rights caveat.
- **Prior audits:** evidence register 2026-07-15/16 (M-01…M-12); math verifier is a
  bounded sample (18/19 chapters; Ch 17 qualitative); regression gates in
  `test_ebook_site.py` pin corrected claims and forbid known false ones.

## 1.2 Content-quality scores (1–5)

Dimensions: Accuracy · Currency · Clarity · Density/minimalism · Pedagogy · Clinical-ML honesty · Figure utility.

| Surface | Acc | Cur | Cla | Den | Ped | Hon | Fig |
|---|---|---|---|---|---|---|---|
| Landing + IA | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Evidence register | 5 | 5 | 5 | 5 | 5 | 5 | 4 |
| Preface | 4→5 | 5 | 4→5 | 3→4 | 4 | 5 | 3→4 |
| Ch 00 (first half) | 5 | 5 | 5 | 3 | 5 | 5 | 4 |
| Ch 00 (second half) | 4→5 | 5 | 5 | 4 | 5 | 5 | 4 |
| Ch 01 | 5 | 5 | 5 | 4 | 5 | 5 | 3→4 |
| Ch 02–03 | 5 | 5 | 5 | 4 | 5 | 5 | 4 |
| Ch 04–05 | 5 | 5 | 4→5 | 4 | 5 | 5 | 4 |
| Ch 06–07 | 5 | 4→5 | 5 | 4 | 5 | 5 | 4 |
| Ch 08–09 | 5 | 5 | 5 | 4 | 5 | 5 | 4 |
| Ch 10–11 | 5 | 5 | 5 | 4 | 5 | 5 | 4 |
| Ch 12–13 | 4→5 | 5 | 5 | 4 | 4→5 | 5 | 4 |
| Ch 14–15 | 5 | 5 | 5 | 5 | 5 | 5 | 3 |
| Ch 16 | 5 | 5 | 5 | 4 | 5 | 5 | 5 |
| Ch 17–18 | 5 | 5 | 5 | 4→5 | 5 | 5 | 5 |
| CSS/JS/print | 5 | 4 | 5 | 4 | 5 | 5 | 4 |

Headline: the book is already well-hardened. Across 16 surfaces the reviewers surfaced a
single genuine P0/P1 defect (one arithmetic slip). Clinical-ML honesty is uniformly 5/5:
prediction-vs-causation discipline, prevalence-transport worked examples, and "not an
empirically calibrated bedside estimate"-style hedges are consistent throughout.

## 1.3 Accuracy and currency (verified against primary sources)

Every substantive change below was reproduced by hand or confirmed against the primary
source before editing. No numbers, benchmarks, or citations were invented.

- **P1 (fixed).** Ch 00 gradient-descent answer stated `w₂ = 2.04`. With L′=2(w−3), η=0.2,
  w₀=0: w₁=1.2, then w₂ = 1.2 − 0.2·2·(1.2−3) = **1.92**. Internally inconsistent with the
  correct w₁ it prints. Not anchored in the math gate, so no script change needed. Fixed to
  1.92.
- **P1 currency (fixed).** Ch 07 stated "UMAP often preserves more global structure than
  t-SNE." That is the Becht et al. 2019 claim, rebutted by Kobak & Linderman 2021
  (*Nature Biotechnology* 39:156–157): the gap is largely an initialization artifact (UMAP
  defaults to spectral/Laplacian-eigenmaps init; t-SNE historically random). Reframed
  accordingly.
- **Currency (fixed).** Ch 12 Llama lineage stopped at Llama 2/3 → added Llama 4 (2025,
  first Llama mixture-of-experts, natively multimodal). Ch 12 SAM lineage stopped at SAM v2
  → added SAM 3 (2025, promptable concept segmentation via text/exemplar prompts) and the
  SAM 3D companion. Ch 12 "foundation models … will blur chapter boundaries" retensed to
  "now blur" (natively multimodal models are mainstream as of 2026-07). Ch 14 FlashAttention
  "-2/3" → "-2, -3, and -4" (FA-4, arXiv 2603.05451, Mar 2026, NVIDIA Blackwell).
- **Currency additions (fixed).** Ch 12 PEFT list gained one line on **QLoRA** (Dettmers
  et al. 2023: 4-bit frozen base + LoRA adapters, single-GPU fine-tuning) and the RLHF/DPO
  passage gained one hedged line on **GRPO** (DeepSeekMath, Shao et al. 2024: critic-free,
  group-normalized advantage; now common for reasoning-model training). No benchmark numbers.
- **KEEP (spot-verified, no change):** Adam bias correction (Kingma & Ba), LoRA count
  (65,536 adapter params at r=8 on a 4096² projection), RLHF/DPO framing, PSI as binned
  Jeffreys divergence, MCMC diagnostics, random-walk Metropolis tuning, Bloom linear-in-n,
  potential-based reward shaping, and all Ch 14 quantization/Huffman/CNN-sizing numbers.

## 1.4 Minimalism and design

- **Preface (fixed).** Cut the marketing overclaim ("This expanded edition … topical mastery
  a careful reader would gain from a full curriculum" / "Exposition is original; the syllabus
  is comprehensive"), keeping the concrete scope list. Moved the telestroke decision-hook
  above the 7-figure opening gallery so the clinical/ML decision lands first. Rewrote four
  name-only captions ("Boundaries of this open-source ebook.") into teaching captions.
- **Ch 01 (fixed).** Moved the resident decision-hook to lead the Opening; rewrote three
  name-only captions into teaching captions.
- **Ch 11 (fixed).** Removed one of two back-to-back near-identical pretrain→fine-tune
  diagrams (the non-standard `swarm3h_`-named asset), which also lets the decision-hook lead;
  improved the surviving caption; deleted the orphaned asset and regenerated the manifest.
- **Ch 04 (fixed).** Collapsed a doubled/confusing density line ("… exp(−4.5)≈0.0111 and
  exp(−4.5)≈0.0111 if means were symmetric—…") into one clean statement.
- **Ch 13 (fixed).** "Suppose three actions" → "two actions" (only Left and Right are
  specified; worked values untouched).
- **Ch 17 (fixed).** Dropped the redundant 5-row "Pre-deployment checklist" table, a strict
  subset of the 11-gate appraisal map.
- **Design (unchanged).** `extra.css` is coherent, teal, and passes all print/responsive/a11y
  contracts; no dead-rule fixes were worth the regression risk.

## 1.5 Quiet machine-readability

- Added `docs/llms.txt` (served at `/ML/llms.txt`): a technical, non-promotional index of
  the real `.html` chapter URLs plus the bounded evidence-register scope and the
  educational-only caveat. No agent-marketing language; not added to nav; a `.txt` (not
  `.md`) so `mkdocs --strict` raises no not-in-nav warning.
- Evidence register intro date widened to "15–16 July 2026" to match the M-10/11/12 rows.
- Headings, permalinks, key-numbers-in-text/tables, and the CRIT-APP cross-link were
  confirmed intact. Live spot-check of the deployed sites was not possible: this
  environment's network policy blocks `rkalani1.github.io`. Cross-links were verified
  structurally.

## 1.6 Findings register (ML-###)

| ID | Sev | Category | Surface | Disposition |
|---|---|---|---|---|
| ML-001 | P1 | accuracy | Ch 00 gradient-descent answer w₂ | FIXED (2.04→1.92; verified by recurrence) |
| ML-002 | P1 | currency | Ch 07 t-SNE/UMAP global structure | FIXED (Kobak & Linderman 2021) |
| ML-003 | P2 | currency | Ch 12 Llama lineage | FIXED (Llama 4, 2025) |
| ML-004 | P2 | currency | Ch 12 SAM lineage | FIXED (SAM 3 / SAM 3D, 2025) |
| ML-005 | P3 | currency | Ch 12 "will blur" multimodal tense | FIXED (retensed) |
| ML-006 | P3 | currency | Ch 14 FlashAttention version | FIXED (adds FA-4) |
| ML-007 | P2 | currency | Ch 12 PEFT list omits QLoRA | FIXED (one line) |
| ML-008 | P2 | currency | Ch 12 RLHF omits critic-free variants | FIXED (one GRPO line) |
| ML-009 | P2 | marketing-residue | Preface overclaim | FIXED (cut) |
| ML-010 | P2 | density | Preface hook buried behind gallery | FIXED (reordered) |
| ML-011 | P3 | figure-utility | Preface name-only captions | FIXED (4 rewritten) |
| ML-012 | P2 | figure-utility | Ch 01 name-only captions + hook order | FIXED (rewritten + reordered) |
| ML-013 | P2 | figure-redundancy | Ch 11 duplicate opening diagram | FIXED (removed asset + manifest) |
| ML-014 | P2 | clarity | Ch 04 doubled density line | FIXED (collapsed) |
| ML-015 | P2 | pedagogy | Ch 13 "three actions" mismatch | FIXED |
| ML-016 | P2 | density | Ch 17 redundant checklist table | FIXED (dropped) |
| ML-017 | P3 | consistency | Evidence-register intro date | FIXED (15–16 July) |
| ML-018 | P2 | machine-readability | No llms.txt | FIXED (added, non-promotional) |
| ML-019 | P2 | density | Ch 00 0.5 dot-product vs 0.10 overlap | NOT CHANGED — intentional labeled preview, load-bearing for cross-links; reference chapter tolerates it |
| ML-020 | P3 | IA | Landing CTA and Orientation card both → Ch 01 | NOT CHANGED — intentional dual entry into the book start |

## Gate results (all green)

`originality_scan` OK · `check_figure_coverage` 20/20 · `build_asset_manifest --check` OK ·
`verify_math_examples` ALL_PASS (96 checks, bounded 18/19) · `test_ebook_site` 21/21 ·
`mkdocs build --strict` OK · `postprocess_site` OK · `validate_rendered_site` OK
(23 HTML, no `<em>` leakage, figures balanced, alt/dimensions/a11y intact).

## Evidence-register deltas

No new rows were required (no new consequential quantitative claim was introduced; the two
one-line method mentions are qualitative and carry their primary sources inline). The intro
audit date was widened to 15–16 July 2026 for consistency. Owner may optionally add rows for
the t-SNE/UMAP initialization correction and the Llama 4 / SAM 3 currency notes in a future
pass.

## Not changed (deliberate)

- No Learning-Objectives sections reintroduced; no "score = care" framing; teal identity kept.
- Ch 00 length left as a reference chapter (density is an accepted tradeoff, not a defect).
- CSS/JS untouched — coherent and fully passing the tested contracts.
- No live redeploy; deployment happens only via the intentional green Actions workflow on main.
