# Swarm 3h cycle-1 — ML open-source ebook (densify pass)

**Agent:** Grok Build cycle-1 implementer  
**When:** 2026-07-15  
**Repo:** https://github.com/rkalani1/ML  

## Goals addressed

1. **≥4 new original code-drawn matplotlib figures** under `docs/assets/figures/` via `scripts/generate_ebook_figures.py` (delivered **6**).
2. Embedded especially in **preface / glossary / ch12 deep apps / ch13 RL** (inventory low-figure chapters).
3. **1 substantive teaching Markdown table** (ch12 architecture family quick map).
4. `verify_math_examples.py`, `check_figure_coverage.py`, `test_ebook_site.py`, `originality_scan.py`, `mkdocs build` — all green.
5. No meta nav tabs; no DOCX image imports; open-source ebook branding preserved.
6. Residuals written to temp path + this `_meta` note.

## New figures

| File | Where used |
|------|------------|
| `ml_fig_claim_types.png` | 00a preface |
| `ml_fig_accuracy_trap.png` | 18 glossary |
| `ml_fig_causal_mask.png` | 12 deep learning apps |
| `ml_fig_dice_iou.png` | 12 deep learning apps |
| `ml_fig_discount_gamma.png` | 13 reinforcement learning |
| `ml_fig_bandit_explore.png` | 13 reinforcement learning |

## Teaching table

- Ch12: Architecture family quick map (encoder-only / decoder-only / enc-dec / CNN-UNet / ViT)

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
`C:\Users\rkala\AppData\Local\Temp\grok-goal-fa96971be2ac\implementer\cycle1-ml-residuals.md`

**Required residuals: empty.**
