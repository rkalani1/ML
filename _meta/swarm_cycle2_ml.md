# Swarm cycle-2 — ML open-source ebook (figure residual clear)

**Agent:** Grok Build cycle-2 implementer  
**When:** 2026-07-14  
**Repo:** https://github.com/rkalani1/ML  

## Goals addressed

1. **Caption-without-art debt cleared:** bare body `Figure x.y` captions either wired to original PNGs or converted to intentional `Figure concept (text diagram)` admonitions.  
2. **≥5 high-value original PNGs** across different chapters (actually **6**).  
3. **ch11 SSL and ch13 RL** each have **≥2** original embedded figures.  
4. **Legacy numbered assets** (`00_*.png` …) documented as frozen historical (present + linked).  
5. Verification gates green; **no open accuracy errors; no open P0/P1 figure items**.

## New figures (`docs/assets/figures/`)

| File | Chapter |
|------|---------|
| `ml_fig_core_functions.png` | 00 math foundations (Fig 0.2) |
| `ml_fig_bias_capacity.png` | 01 basic concepts (Fig 1.6) |
| `ml_fig_elbow_wss.png` | 04 clustering (Fig 4.5) |
| `ml_fig_activations.png` | 10 neural nets (Fig 10.2) |
| `ml_fig_triplet_ssl.png` | 11 SSL (Fig 11.3) — second figure |
| `ml_fig_value_iteration.png` | 13 RL (Fig 13.3) — second figure |

Generator: `scripts/generate_ebook_figures.py`  
Wiring / callouts: `scripts/cycle2_caption_art.py`  
Embed map updated: `scripts/embed_figures.py`

## Caption disposition

- Bare paragraph-start `Figure x.y.` count: **0**  
- Concept callouts: **105**  
- High-value PNG wires this cycle: **6**  
- Missing linked files: **0**

## Legacy numbered assets (frozen)

Present under `docs/assets/figures/` and listed as `LEGACY_NUMBERED_ASSETS` in the generator (not regenerated):

`00_vector_matrix`, `01_gradient_descent`, `02_viz_anatomy`, `03_bayes_update`, `04_kmeans`, `07_pca_projection`, `08_regression_fit`, `09_supervised_map`, `10_mlp_architecture`, `16_leakage_timeline`, `17_roc_curve`

## Verification

```text
python scripts/check_figure_coverage.py C:\Users\rkala\ML   # OK 20 BAD 0
python scripts/test_ebook_site.py C:\Users\rkala\ML         # 7/7 OK
python scripts/originality_scan.py C:\Users\rkala\ML        # OK
mkdocs build                                                # exit 0
```

## FINAL residuals

Full detail:  
`C:\Users\rkala\AppData\Local\Temp\grok-goal-d7de3228caab\implementer\cycle2-ml-final-residuals.md`

**Required residuals: empty.**  
Optional polish only: more concept→PNG upgrades, port legacy into generator, syllabus disclosure, MkDocs 2.0 platform warning.
