# Update protocol

## Canonical source

The repository's `docs/` Markdown is the source of truth for the free Web
Edition. A desktop document is not canonical after publication.

## Required workflow

1. Branch from the current protected default branch.
2. Make a focused change and update the evidence register when a consequential
   factual or quantitative claim changes.
3. For every added or changed asset, first rebase onto the intended target and
   commit the asset plus its generator/source. Do not amend or rebase that
   content commit afterward. Run:

   ```bash
   python scripts/build_asset_rights_register.py --refresh-from-history
   python scripts/build_asset_manifest.py
   git diff -- _meta/asset-rights-register.json docs/assets/figures/manifest.json
   ```

   Review the regenerated origin, tool/source, license or publisher rights
   basis, AI assistance, human review, and exact hash, then commit the two
   inventories separately. Asset-changing pull requests must use a merge commit
   that preserves the recorded content commit; do not squash or rebase them. If
   a commit identity changes, refresh and review the inventories again.
4. Run the local source and rendered-site release suite:

   ```bash
   python -m pip install --require-hashes -r requirements.lock
   python scripts/originality_scan.py
   python scripts/check_figure_coverage.py
   python scripts/build_asset_rights_register.py --check
   python scripts/build_asset_manifest.py --check
   python scripts/check_publication_safety.py
   python scripts/test_release_gates.py
   python scripts/verify_math_examples.py
   python scripts/test_ebook_site.py .
   node scripts/test_mathjax_fallback.mjs
   node scripts/verify_mathjax_integrity.mjs
   mkdocs build --strict --clean
   python scripts/postprocess_site.py --site-dir site
   python scripts/validate_rendered_site.py --site-dir site
   python scripts/check_publication_safety.py --site-dir site
   ```

5. Treat the GitHub Actions `Release gates` job as required additional proof.
   It installs the wheel-only `requirements-audit-bootstrap.lock` before
   installing `requirements-audit.lock` without build isolation; runs
   `pip-audit` against all three locked dependency inventories, `cffconvert`
   against `CITATION.cff`,
   `scripts/verify_sbom_schema.py`, `scripts/test_gitleaks_controls.py`, and
   pinned Gitleaks against release-reachable history plus the current
   pull-request head and event merge result; and reruns the source and rendered-
   site checks on Ubuntu. Do not merge if any workflow gate is skipped or fails.
6. Open a pull request. Merge only after the non-deploying PR build succeeds
   and the factual/rights review appropriate to the change is complete.
7. Verify that GitHub Pages deployed the merged commit and reread the live
   release surface before recording completion.

## Never add

- pasted paper abstracts, publisher PDFs, commercial textbook passages, or
  copied guideline/clinical-instrument tables;
- a third-party figure or close redraw without compatible permission,
  attribution, and asset-level documentation;
- patient-identifiable or confidential institutional information;
- credentials, private keys, access tokens, or private datasets; or
- synthetic numbers presented as clinical thresholds, expected performance, or
  real-patient outcomes.

AI assistance must be disclosed and reviewed; it never substitutes for source,
rights, privacy, clinical, or security verification.

## Versioning

Update the citation metadata and release record for material editions. Use an
annotated release tag only after CI, review, and live exact-commit verification.
