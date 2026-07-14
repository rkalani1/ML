# Swarm cycle-1 — ML open-source ebook

**Agent:** Grok Build cycle-1 implementer  
**When:** 2026-07-14  
**Repo:** https://github.com/rkalani1/ML  

## Goals addressed

1. Every curriculum chapter has ≥1 teaching visual (glossary may use tables; all now have images + tables where helpful).  
2. Original matplotlib figures only via `scripts/generate_ebook_figures.py` (extended).  
3. Broken image links fixed / none remaining.  
4. Teaching tables added (preface reader map; ch05 association metrics; glossary metric map retained).  
5. Synthetic metrics/math spot-checked (OLS, association rules, attention, Huffman) — all PASS.  
6. IA remains Introduction + Book only.

## Deliverables

### New / regenerated figures (`docs/assets/figures/`)

- `ml_fig_how_to_read.png`
- `ml_fig_association_rules.png`
- `ml_fig_feature_pipeline.png`
- `ml_fig_attention.png`
- `ml_fig_ols_fit.png`
- `ml_fig_metric_map.png`
- plus regenerable: `ml_fig_distill_prune.png`, `ml_fig_graph_toy.png`, `ml_fig_viz_hygiene.png`
- full prior `ml_fig_*` suite regenerated

### Chapters edited

`00a-preface`, `05`, `06`, `08`, `12`, `17`, `18` (+ scripts + optional CSS mobile/print)

### Verification

- `check_figure_coverage.py`: OK 20 BAD 0  
- `test_ebook_site.py`: 7/7 OK  
- `mkdocs build`: exit 0  

## REQUIRED residuals

See full detail:  
`C:\Users\rkala\AppData\Local\Temp\grok-goal-d7de3228caab\implementer\cycle1-ml-residuals.md`

Short list:

1. Many body `Figure x.y` captions still lack dedicated PNGs (polish).  
2. Legacy numbered `0x_*.png` not in generator (reproducibility).  
3. Optional second figure for ch11 SSL and ch13 RL.  
4. Prior originality syllabus-alignment residual unchanged.  
5. MkDocs 2.0 ecosystem warning (platform).
