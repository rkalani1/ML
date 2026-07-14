# Originality swarm verdict — ML Web Edition (short)

**Date:** 2026-07-13  
**Full opinion:** `C:\Users\rkala\OneDrive - UW\Codex\AI-Work-Sync\textbook-originality-swarm-VERDICT.md`

## Verdict

- **Absolute zero legal risk:** **Cannot be claimed.**
- **Practical copyright-claim risk (evidence-based):** **LOW.**
- **Public Pages:** **GO** — keep https://rkalani1.github.io/ML/ online.

## What was checked

- Swarm dense audit: `docs/_swarm_audit/originality_full.md` (+ JSON)
- Automated `scripts/originality_scan.py` + deep scan → **0 flags**
- LICENSE (ISC code + CC BY 4.0 content), `about.md`, `ORIGINALITY.md`
- 21 curriculum chapters; **0** published image embeds (129 DOCX images skipped)

## Key findings

| Severity | Item | Status |
|----------|------|--------|
| HIGH | Clear paste / abstracts / permission / book figures | **None found** |
| MEDIUM | Preface syllabus alignment to named external curriculum TOC | Disclosed; coverage ≠ wording paste |
| MEDIUM | Dense math/CS blocks feel textbook-like | Original voice + synthetic examples on spot-check |
| MEDIUM (product) | Captions without figure files | Intentional omission until code-redraw |
| SAFE | Universal Web Edition disclaimer + clinical frames | Present |

## Owner actions

1. Spot-check 3 random chapters yourself.  
2. **Code-redraw** key figures into `docs/assets/figures/` — do **not** re-import unverified DOCX images.  
3. Keep syllabus-vs-expression disclosure if external TOC name remains.  
4. Re-run originality CI after large edits.

## Honest limit

No scan certifies absolute legal immunity. Practical risk of a successful copyright claim on current published expression is low on present evidence.
