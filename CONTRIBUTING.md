# Contributing

Contributions are welcome when they preserve the book's educational,
rights-aware, and reproducible-publication standards.

## Before opening a pull request

By contributing, you confirm that:

- you have the right to submit the contribution and agree that book content
  you submit may be distributed under CC BY 4.0 and software, configuration,
  and documentation tooling you submit may be distributed under the ISC
  License, unless an explicit, compatible file-specific exception is accepted;
- you wrote the contribution or have documented permission and a compatible
  license to submit it;
- any quotation, adaptation, dataset, figure, font, or other third-party
  material is necessary, proportionate, attributed, and recorded in
  `THIRD_PARTY_NOTICES.md` or the relevant source/provenance register;
- the contribution contains no patient-identifiable information, confidential
  institutional information, credentials, private keys, or non-public data;
- synthetic clinical examples are labeled synthetic and are not presented as
  patient-specific guidance;
- consequential medical, legal, regulatory, or performance claims cite a
  current primary or official source and state important limitations; and
- any material use of generative AI is disclosed in the pull request. AI
  assistance does not transfer responsibility for accuracy, originality,
  attribution, or licensing away from the contributor.

Do not paste article abstracts, publisher PDFs, commercial textbook prose,
licensed clinical instruments, guideline tables, or screenshots merely because
they are available online. Linking to a source does not grant reuse rights.

## Figures and other assets

Add only assets with a documented origin. Prefer reproducible code-generated
SVG or PNG files. The rights register records a Git commit containing the exact
asset bytes, so first rebase onto the intended target, commit the asset and its
generator/source, and do not amend or rebase that content commit afterward.
Then run:

```bash
python scripts/build_asset_rights_register.py --refresh-from-history
python scripts/build_asset_manifest.py
git diff -- _meta/asset-rights-register.json docs/assets/figures/manifest.json
```

Review every regenerated origin, rights basis, and hash before committing those
two inventories separately. An asset-changing pull request must be merged with
a merge commit that preserves the recorded content commit; do not squash or
rebase it. If commit identities change, rerun the refresh and review before
merge. Do not trace or closely imitate a third-party figure unless its license
clearly permits the adaptation and the adaptation is labeled and attributed.

## Local checks

Use Python 3.12. After installing the hash-locked dependencies, run:

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
mkdocs build --strict --clean
python scripts/postprocess_site.py --site-dir site
python scripts/validate_rendered_site.py --site-dir site
python scripts/check_publication_safety.py --site-dir site
node scripts/test_mathjax_fallback.mjs
node scripts/verify_mathjax_integrity.mjs
```

The required GitHub Actions `Release gates` job additionally installs the
wheel-only `requirements-audit-bootstrap.lock` before installing
`requirements-audit.lock` without build isolation; runs `pip-audit` against
all three locked dependency inventories, `cffconvert` against `CITATION.cff`,
`scripts/verify_sbom_schema.py`, `scripts/test_gitleaks_controls.py`, and pinned
Gitleaks against release-reachable history plus the current pull-request head
and event merge result; and reruns the source and rendered-site checks on
Ubuntu. These workflow-only checks are part of release readiness.

Open a pull request and wait for the complete build check. A passing automated
check is evidence about the checks it ran; it is not legal advice, peer review,
clinical validation, or a guarantee that no defect remains.
