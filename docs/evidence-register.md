# Evidence register

This register records consequential technical claims checked against primary papers or official references during the 15 July 2026 editorial audit. A checked row is narrow: it does **not** certify every sentence in the chapter or establish clinical utility.

| ID | Chapter and claim | Primary or official source | Verified conclusion | Status |
| --- | --- | --- | --- | --- |
| M-01 | Ch 0 · convexity, stationarity, and uniqueness | [Boyd & Vandenberghe, *Convex Optimization*](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) | A stationary point of a differentiable convex function is global; uniqueness and optimizer convergence require additional conditions | Checked 2026-07-15 |
| M-02 | Ch 0 · binary64 epsilon and unit roundoff | [Python `sys.float_info`](https://docs.python.org/3/library/sys.html#sys.float_info) | Gap above 1 is 2⁻⁵²; round-to-nearest unit roundoff is 2⁻⁵³ for normal results | Checked 2026-07-15 |
| M-03 | Ch 10 · Adam bias correction | [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980) | β₂=0.999 bias correction remains material for hundreds to thousands of steps | Checked 2026-07-15 |
| M-04 | Ch 12 · LoRA parameterization | [Hu et al., 2021](https://arxiv.org/abs/2106.09685) | ΔW=(α/r)BA; one 4096² projection at r=8 trains 65,536 adapter parameters | Checked 2026-07-15 |
| M-05 | Ch 12 · one common RLHF pipeline | [InstructGPT](https://arxiv.org/abs/2203.02155) | Supervised tuning, reward modeling, and policy optimization with a reference-policy constraint is one common pipeline, not the definition of all RLHF | Checked 2026-07-15 |
| M-06 | Ch 12 · DPO scope | [Rafailov et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) | Fits preferred/rejected pairs relative to a reference policy under pairwise-preference assumptions | Checked 2026-07-15 |
| M-07 | Ch 16 · random-walk Metropolis tuning | [Gelman, Roberts & Gilks](https://sites.stat.columbia.edu/gelman/research/published/baystat5.pdf) | Canonical spherical-normal tuning results are dimension- and target-specific guides, not universal acceptance criteria | Checked 2026-07-15 |
| M-08 | Ch 16 · MCMC convergence diagnostics | [Vehtari et al., 2021](https://arxiv.org/abs/1903.08008) | Fixed warmup alone does not establish convergence; inspect multiple chains, rank-normalized R-hat, ESS, and trace/rank plots | Checked 2026-07-15 |
| M-09 | Ch 16 · PSI threshold | [Potgieter et al., 2023](https://arxiv.org/abs/2307.11878) | Fixed PSI cutoffs are sample-size-insensitive heuristics; calibrate thresholds and use PSI as an investigation trigger | Checked 2026-07-15 |

## Review contract

For each consequential claim, preserve the source, assumptions, exact notation or data-generating setup, a reproducible calculation where applicable, and the review date. Distinguish a theorem from the assumptions needed by an algorithm, a synthetic teaching example from empirical evidence, and predictive performance from clinical utility.

## Known frontier

The 15 July 2026 pass corrected confirmed mathematical and reproducibility errors and added regression gates. It was not a line-by-line expert review of all 20 chapters. Unreviewed claims, benchmark currency, and clinical deployment advice remain **needs confirmation**. The [asset manifest](assets/figures/manifest.json) records hashes and software metadata; independent rights review remains incomplete.
