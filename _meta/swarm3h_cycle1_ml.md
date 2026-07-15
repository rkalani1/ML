# Swarm 3h cycle-1 — ML open-source ebook

**Agent:** Grok Build cycle-1 implementer  
**When:** 2026-07-15  
**Repo:** https://github.com/rkalani1/ML  

## Goals addressed

1. **≥6 new original code-drawn matplotlib figures** under `docs/assets/figures/` via `scripts/generate_ebook_figures.py`.
2. Embedded especially in **preface / glossary / closing**; densified **ch03 probability** with real scientific plots.
3. **≥2 substantive teaching Markdown tables** added.
4. `verify_math_examples.py`, `check_figure_coverage.py`, `test_ebook_site.py`, `originality_scan.py`, `mkdocs build` — all green.
5. No meta nav tabs; no DOCX image imports; open-source ebook branding preserved.
6. Residuals written to temp path + this `_meta` note.

## New figures

| File | Where used |
|------|------------|
| `ml_fig_curriculum_map.png` | 00a preface |
| `ml_fig_glossary_families.png` | 18 glossary |
| `ml_fig_ppv_prevalence.png` | 18 glossary, 03 probability |
| `ml_fig_lifecycle_deploy.png` | 17 closing |
| `ml_fig_decision_curve.png` | 17 closing |
| `ml_fig_clt_sampling.png` | 03 probability (CLT densify) |
| `ml_fig_mle_bernoulli.png` | 03 probability (MLE densify) |
| `ml_fig_learning_curves.png` | 01 basic (reproducible generator) |
| `ml_fig_confusion_annotated.png` | 09 classification (reproducible generator) |

## New / retained teaching tables

- Preface: Curriculum blocks  
- Glossary: Prevalence → PPV quick reference  
- Closing: Pre-deployment checklist  

## Verification

```text
verify_math_examples.py     ALL_PASS
check_figure_coverage.py    OK 20 BAD 0
test_ebook_site.py          10 tests OK
originality_scan.py         OK
mkdocs build                exit 0
```

## FINAL residuals

Temp detail:  
`C:\Users\rkala\AppData\Local\Temp\grok-goal-e534d4870158\implementer\cycle1-ml-residuals.md`

**Required residuals: empty.**
