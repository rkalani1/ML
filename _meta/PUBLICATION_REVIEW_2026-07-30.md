# Public-release candidate review record — ML

Reviewed: **30 July 2026**

Baseline reviewed against: `d7aab5815e3f9991453f23f893e7f452928d76d6`

## Disposition

No known release-blocking defect was identified in the reviewed current tree
after the checks recorded below. This is a bounded engineering, editorial, and
source review—not a legal opinion, a guarantee of non-infringement, independent
specialty peer review, clinical validation, regulatory clearance, or a promise
that publication cannot result in a claim.

The locally built, post-processed release-candidate artifact and the public Git
repository are different rights surfaces. The local artifact passed the stated
source and rendered gates. This record does not claim that the candidate commit
has been merged, deployed, or read back from live Pages; the workflow records
that exact-commit proof only after a successful main-branch deployment. The
repository as a whole is not represented as comprehensively rights-cleared
because thousands of superseded Matplotlib-tagged image blobs remain reachable
in history. Publishing from a clean-history repository is the conservative
route if repository-wide clearance is required.

The release uses scoped language throughout: it publishes only rights the
publisher controls, describes the book as educational, identifies selected
rather than exhaustive evidence review, and provides correction, rights, and
security reporting paths.

## Release surface reviewed

- 20 curriculum chapters and 23 Markdown pages under `docs/`
- 66 current teaching figures
- 29 hash-locked transitive Python build dependencies
- 24 generated HTML pages, 9.67 MiB after post-processing
- current source, rendered output, repository history, workflow configuration,
  citation metadata, third-party components, and public external links

## Rights, authorship, and licensing

- Site software and configuration are offered under ISC; publisher-controlled
  book material is offered under CC BY 4.0 only to the extent the publisher
  controls the relevant rights.
- Third-party icons, theme code, MathJax, trademarks, linked works, facts, and
  material outside the publisher's control are expressly excluded from the
  book-content grant and retain their own notices. The site uses system fonts
  rather than downloading external fonts.
- Every current figure has an exact hash and verified recorded-content Git
  record. A refresh selects the newest matching path-changing commit at refresh
  time; routine checks verify that the recorded commit is an ancestor of the
  current tree, changed the asset path, and contains the current bytes. The
  asset register reports 58 conservative generator-source candidates, eight
  embedded-metadata records, and no repository-content-history-only records.
  Three raster assets for which no honest generator or embedded provenance
  could be established were removed; their instructional content remains in
  semantic lists and a table. These records and replacements support
  traceability; they do not prove independent creation, copyrightability,
  identity, or ownership.
- AI assistance in drafting, editing, code, figures, testing, and review is
  disclosed. No AI output is treated as a factual authority.
- No current-tree publisher PDF, article attachment, copied clinical
  instrument, office document, archive, or media file was found. No long block
  quotation was found in the curriculum.

The review cannot determine the effect of an employment agreement, University
of Washington or other institutional intellectual-property policy, grant or
sponsor term, prior publication agreement, or contributor assignment. UW's
published policy generally leaves scholarly works with faculty authors but
identifies exceptions for commissioned work, certain funding or contractual
terms, and University-supported work. A creator who knows that one of those
facts applies should obtain the appropriate institutional or legal
determination before relying on the repository license.

## Technical, clinical, research, and public-safety review

- Every curriculum chapter received a full-file final review.
- Corrections addressed probability and statistical interpretation, validation
  and leakage, clustering and dimensionality reduction, regression and
  classification, neural-network training, deep-learning and foundation-model
  claims, reinforcement-learning updates and offline evaluation, compression
  arithmetic, graph algorithms, privacy, de-identification, fairness,
  calibration, monitoring, and model retirement.
- Universal clinical commands and categorical claims that architectures,
  metrics, checklists, validation sets, explanations, or governance documents
  prove safety or validity were removed. Model cards and lifecycle diagrams do
  not imply authorization to deploy.
- Worked numerical examples are synthetic unless a source is named. Current
  guidelines and local pathways, not the book, govern time-sensitive clinical
  decisions.
- Product and model names are educational examples, not endorsements. Product
  capabilities, availability, licensing, and regulatory status remain
  version-, jurisdiction-, and use-dependent.
- The evidence register is explicitly bounded to selected consequential claims
  and records its source and review date. It is not represented as full-book
  peer review.
- Exercises do not authorize collection of patient information or replace local
  privacy, security, IRB, regulatory, procurement, or clinical-governance
  determinations.

## Privacy, security, and software-supply-chain review

- The rendered site has no user accounts, uploads, data-submission forms,
  configured analytics, advertising, or publisher-set cookies. Theme search and
  palette forms operate locally.
- A reviewed Content Security Policy restricts connections and active content
  to the site itself, with the exact pinned MathJax bundle and its versioned
  math-font directory as the only configured remote runtime resources. Script
  authorization is limited to the exact MathJax URL, reviewed theme-inline
  hashes, and same-origin files; arbitrary inline script and host-wide CDN
  script authorization are not enabled.
- The release scanner checks the complete tracked current tree and the rendered
  Pages artifact. It rejects symlinks, unknown/NUL/non-UTF-8 repository or
  published files, unreviewed static suffixes, malformed or metadata-bearing
  PNGs outside a narrow reviewed chunk policy, and misassociated font hashes.
  It checks source and rendered HTML, Markdown, SVG active
  content after XML entity decoding, inline SVG, escaped CSS, duplicate HTML
  attributes, strict HTML document structure and policy placement, active
  document suffixes, remote-resource allowlists, browser-network primitives,
  tracker and font hosts, secret patterns, identifier patterns, and local
  macOS, Linux, and Windows paths. Each rendered figure/font path and digest
  must match its inventoried source counterpart. Exactly one reviewed
  `strict-origin-when-cross-origin` policy is required, and per-element
  referrer-policy overrides are rejected. Only a passive same-site sitemap XML file
  and byte-identical gzip copy are permitted. Raw authored script elements are
  rejected. Authored stylesheets are source-to-render hash-bound and may not
  embed `data:` payloads; rendered theme stylesheets must match the locked
  Material for MkDocs package byte-for-byte. Each authored JavaScript file is
  hash-inventoried, must match a
  separately maintained manually reviewed digest baseline, and must match its
  rendered copy byte-for-byte. Rendered theme JavaScript must match the
  hash-locked Material for MkDocs package. Every executable inline hash must be
  both allowlisted and observed. Every local rendered script source, including
  module suffixes and the configured `/ML/` 404-page path form, must resolve to
  an exact allowlisted path and digest; remote script markup is rejected.
- Pages with mathematical notation load only the exact MathJax 3.2.2 bundle
  from jsDelivr with Subresource Integrity, `crossorigin="anonymous"`, and a
  no-referrer policy. Failure removes the failed loader, leaves source notation
  visible, and produces an accessible status notice on each instant-navigation
  math page.
- Runtime, audit-bootstrap, and audit requirements are exact and hash-locked.
  All three inventories returned `No known vulnerabilities found` with
  pip-audit 2.10.1 on the review date.
- The 29-component CycloneDX 1.6 SBOM exactly matches the runtime/site-build
  lock, includes declared-license metadata for every component, and validates
  against the official CycloneDX 1.6 JSON schema. The separate 41-package CI
  audit/validation toolchain is completely and hash-locked in
  `requirements-audit.lock`. Its pip and setuptools build tools are installed
  first from wheel-only hashes in `requirements-audit-bootstrap.lock`, and the
  full lock is then installed without build isolation. It is not shipped in
  the site or represented as part of the runtime SBOM.
- GitHub Actions are commit-SHA pinned. Pull requests run all release gates;
  only a successful build of `main` can configure, upload, and deploy Pages.
  Each generated page is stamped with the exact validated Git revision, and the
  deployment job downloads the complete live response and structurally
  requires exactly one active direct-`head` marker with that revision.
- On pull-request events, CI explicitly fetches only the current pull-request
  head; the event checkout supplies its merge result. The blocking scan covers
  release-reachable refs plus those current-event refs, so a retained closed
  external pull request cannot permanently deny later `main` releases. CI
  downloads the exact Gitleaks 8.30.1 Linux archive only after verifying its
  published SHA-256 and scans the resulting reachable history as text with
  secret values redacted. Inline allow directives are ignored, repository
  configuration/ignore overrides are rejected or bypassed with a trusted empty
  ignore file, and integration mutations prove that both a base64-encoded
  secret hidden behind `-diff` plus `gitleaks:allow` and a secret introduced
  only in a merge result are still detected.
- GitHub private vulnerability reporting was enabled and confirmed by API
  readback for the private channel documented in `SECURITY.md`.

## Verification results

- originality/residue scan: pass
- figure coverage: 20 chapter checks, zero missing or bad
- current asset-rights register and public manifest recomputation: pass
- source publication-safety scan: 82 files and 66 figures, pass
- negative/mutation release-gate tests: 27 of 27 pass
- Gitleaks suppression and merge-patch integration mutations: pass
- recomputed numerical checks: 96 of 96 pass
- source structural/site tests: 24 of 24 pass
- MathJax failure test and live exact-bundle digest: pass
- MkDocs 1.6.1 strict clean build: pass
- rendered validation, exact revision-stamp check, and rendered publication-
  safety scan: pass
- GitHub Actions static validation with actionlint 1.7.12: pass
- citation metadata validation against CFF 1.2: pass
- shared live-link review across both books: 61 unique destinations, no
  confirmed broken destination; 18 publisher endpoints resisted automation with
  authentication, rate-limit, or anti-bot responses and remain subject to
  ordinary reader verification
- responsive browser review at desktop and 390 × 844 mobile widths, including
  dark mode and reduced motion: no broken local links or horizontal overflow
  found in the sampled high-risk pages

Before this hardening commit, 575 commits were reachable across fetched branch,
tag, and public pull-request refs. A post-commit Gitleaks 8.30.1 scan including
the candidate covered all 576 reachable commits as text (about 489 MB) and
found no leak. The pre-hardening all-object identifier/path/type screen covered
24,972 objects and 21,511 blobs and found no SSN, MRN, DOB, private-key marker,
or suspicious document/archive/database extension in the tested patterns.

This dated review deliberately included retained public pull-request refs.
Recurring blocking CI scans only release-reachable refs and the current
pull-request event; retained closed external pull requests remain part of
periodic history audit rather than an attacker-controlled permanent release
gate.

The all-ref image-object screen found 12,735 historical image paths and 11,378
unique reachable image blobs (11,376 PNG and 2 SVG). Sixty-six image blobs are
in the current release, leaving 11,312 non-current blobs reachable through
public history. Of the PNG blobs, 11,373 carry Matplotlib software metadata;
the other three carry no detected creator marker. No EXIF/IPTC
identity-or-location field, Adobe/other creator marker, textual
copyright/trademark/attribution marker, or active/remote SVG feature was found
by the tested metadata and text patterns. This automated screen is not raster
OCR, visual similarity review, source-data review, chain-of-title proof, or
legal clearance of every superseded image. History was not rewritten.

## Accessibility

The engineering target is WCAG 2.2 AA. Source and rendered checks cover
structure, alternative text, link and asset integrity, high-contrast color
schemes, keyboard/focus behavior, zoom/reflow, reduced motion, and responsive
layout. Automated and sampled manual review do not establish universal
conformance across every browser and assistive-technology combination.

## Sources controlling the review framework

- U.S. Copyright Office,
  [Copyright and Artificial Intelligence, Part 2](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf)
- [Creative Commons Attribution 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)
- [UW Copyright Policy summary](https://copyrightresource.uw.edu/uw-policies/uw-copyright-policies/)
  and [UW-owned works guidance](https://copyrightresource.uw.edu/uw-policies/uw-owned-works/)
- HHS,
  [HIPAA de-identification guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)
  and [OHRP quality-improvement FAQ](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/faq/quality-improvement-activities/index.html)
- FDA,
  [Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
- FTC,
  [Health Products Compliance Guidance](https://www.ftc.gov/business-guidance/resources/health-products-compliance-guidance)
- [WCAG 2.2](https://www.w3.org/TR/WCAG22/) and
  [ADA web-accessibility guidance](https://www.ada.gov/resources/web-guidance/)
- [GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service),
  [DMCA policy](https://docs.github.com/en/site-policy/content-removal-policies/dmca-takedown-policy),
  and [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)

## Residual human checks

Before treating the scoped license as a chain-of-title conclusion, the
maintainer should confirm personal facts unavailable to the repository review:

1. no commissioning, funding, institutional-resource, sponsored-research,
   employment, or prior-publication term gives another party rights in the
   release;
2. any human contributor whose protectable work remains has authorized the
   offered license;
3. any known source-derived excerpt, figure, instrument, or dataset is either
   removed, licensed, public domain, or used on a defensible basis with required
   attribution; and
4. counsel is consulted if a disputed asset, institutional ownership question,
   commercial use, registration, demand letter, or other material legal risk
   arises.

Passing this review means no concrete blocker was found within its stated
current-tree scope. It does not clear the public Git history as a whole and
does not mean that legal, clinical, privacy, security, or accessibility risk is
zero.
