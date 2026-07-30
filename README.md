# Machine Learning & AI for Neurologists

Free, open-source teaching ebook on ML/AI literacy in neurology and stroke.
It follows the path from cohort definition through honest validation,
decision-relevant evaluation, governance, monitoring, and retirement.

- [Read the book](https://rkalani1.github.io/ML/)
- [Publication standards and disclosures](https://rkalani1.github.io/ML/publication-standards.html)
- [Bounded evidence register](https://rkalani1.github.io/ML/evidence-register.html)
- [Critical-appraisal companion](https://rkalani1.github.io/CRIT-APP/)

This is educational material, not medical advice, patient-specific decision
support, device clearance, local policy, or institutional endorsement. Worked
numbers are synthetic unless a source is named. The evidence register records
selected primary/official-source checks; it is not full-book peer review.

## Build locally

Use Python 3.12:

```bash
python -m pip install --require-hashes -r requirements.lock
mkdocs serve
```

The full release gates and contribution standards are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Licensing and provenance

Publisher-controlled book material is offered under CC BY 4.0 only to the
extent the publisher controls the relevant rights. Site software is ISC.
Third-party components retain their own licenses. See [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[CITATION.cff](CITATION.cff).

AI tools assisted development; the human maintainer selected, revised, reviewed,
and decided what to publish. Current figure hashes and repository-origin
records are maintained in `_meta/asset-rights-register.json`. These measures
support traceability but are not a legal opinion or a guarantee of
non-infringement. The bounded checks and residual human confirmations are
recorded in
[`_meta/PUBLICATION_REVIEW_2026-07-30.md`](_meta/PUBLICATION_REVIEW_2026-07-30.md).
