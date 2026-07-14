# Update protocol

## Canonical source

**This repository’s `docs/` Markdown** is the source of truth for the free Web Edition.
Do not treat a Desktop DOCX as canonical after this site is live.

## Safe edit workflow

1. Branch from `main`.  
2. Edit one chapter file under `docs/curriculum/`.  
3. If adding a figure: place original SVG/PNG under `docs/assets/figures/`, caption it,
   record `source: original` in `docs/ORIGINALITY.md`.  
4. Run `python scripts/originality_scan.py` and `mkdocs build --strict`.  
5. Open PR; merge after CI green.  

## Forbidden in updates

- Pasting paper abstracts  
- Embedding publisher PDFs  
- Copying commercial textbook paragraphs  
- Guideline table dumps of copyrighted instruments  
- Internal institutional phone lists or PHI  
- Third-party figures without a clear original-curriculum redraw pipeline  

## Versioning

Bump the curriculum note in `docs/index.md` or release tags when major parts change.
