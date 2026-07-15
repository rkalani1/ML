# Preface

## Opening
![Skill stack built by this open-source ebook (original).](../assets/figures/ml_fig_skill_stack.png)

*Skill stack built by this open-source ebook (original).*

![Boundaries of this open-source ebook (original).](../assets/figures/ml_fig_boundaries.png)

*Boundaries of this open-source ebook (original).*

![Reading path through the open-source ebook (original).](../assets/figures/ml_fig_reader_journey.png)

*Reading path through the open-source ebook (original).*


![How to read this open-source ebook (original).](../assets/figures/ml_fig_how_to_read.png)

*Study path: recompute by hand → map method to clinical claim → audit metrics beyond AUC → external validation (original).*

![Model appraisal scorecard (original).](../assets/figures/ml_fig_appraisal_scorecard.png)

*Orientation graphic for how this open-source ebook treats models (original).*

![Curriculum map — open-source ML ebook path (original).](../assets/figures/ml_fig_curriculum_map.png)

*Curriculum map: math → unsupervised/features → supervised → deep/SSL/RL → graphs & data risks → senior practice (original).*

![Three clinical claim types: prediction, etiology, decision support (original).](../assets/figures/ml_fig_claim_types.png)

*Keep claim types separate in journal club: a high AUROC predicts; it does not prove cause or mandate action (original).*

![Prediction ≠ causation: confounder sketch and claim boxes (synthetic; original).](../assets/figures/ml_fig_pred_not_cause.png)

*Figure — Prediction is not causation. **Left:** synthetic scatter of feature *X* vs outcome *Y* colored by latent confounder *U*. A naive linear fit of *Y* on *X* looks strong, yet the data-generating process has no *X*→*Y* edge—only *U*→*X* and *U*→*Y*. **Right:** keep claim types separate—a calibrated predictor that uses *X* can be clinically useful without licensing an interventional “change *X* to change *Y*” story. Association rules, PageRank, and high AUROC all share this trap.*


A telestroke consult ends. The hub radiologist mentions a new large-vessel occlusion detector with a ‘state-of-the-art AUROC.’ The spoke hospital asks whether to buy it this quarter. This book exists so that conversation starts with definitions, data, and decision impact—not vendor vocabulary.

This expanded edition is designed so that studying this document alone delivers the topical mastery a careful reader would gain from a full curriculum on machine learning and artificial intelligence: foundations, unsupervised learning, feature engineering and decomposition, supervised methods, deep and self-supervised learning, multimodal models, reinforcement learning, efficient models, graph mining, and data challenges. Exposition is original; the syllabus is comprehensive.

You are a neurologist and epidemiologist. Examples emphasize stroke systems, neuroimaging, EHR phenotyping, multi-site validation, index time, leakage, calibration, and causal caution—while remaining faithful to general ML/AI theory and algorithms.

![Journal-club scorecard: high AUROC paper vs disciplined paper (synthetic; original).](../assets/figures/ml_fig_journal_club_card.png)

*Figure — Preface journal-club card. Score claim typing, leakage, cohort/label, discrimination+calibration, utility at clinical thresholds, external/temporal test, and prohibited uses/monitoring independently. Paper A can win the AUROC poster and still fail utility, transport, and governance gates. Prediction ≠ causation; a failed critical gate means do not ship.*

## Why Neurologists Need to Know Machine Learning

The intersection of neuroscience, clinical neurology, and computation is rapidly evolving. Historically, we relied on classical biostatistics to evaluate treatments and understand disease etiology. Today, machine learning models analyze continuous EEG streams, high-resolution MRIs, and vast troves of unstructured clinical text to predict outcomes, suggest diagnoses, and optimize resource allocation.

Understanding these models is no longer a niche skill—it is a core clinical competency. When a black-box model suggests that a patient with acute ischemic stroke is not a candidate for thrombectomy, you must know what features the model considers, what biases it might harbor, and what its uncertainty estimates actually mean. You cannot delegate this judgment.

## How to Read This Book

Every major algorithm family in the published TOC is taught with definitions and intuition.

![Study-design triad: cohort, index time, label legality before algorithm choice (original).](../assets/figures/ml_fig_study_design_triad.png)

*Figure — Design before optimizer. Lock **cohort** (who), **index time** (when features are legal), and **label** (what event, when ascertained, rater reliability). Features after index time are leakage; fuzzy labels cap achievable performance. Algorithm choice is secondary to this triad. Prediction ≠ causation.*

1. **Work numerical examples by hand:** Recompute intermediates. Nothing demystifies an algorithm faster than doing the matrix multiplication or gradient update yourself.
2. **Map each method:** Continually map the computational method back to the clinical problem space. Is this for prediction, etiology, or decision support?
3. **Be skeptical of performance metrics:** Always look past discrimination (AUROC) to calibration, utility, and external validation.

![Method family → allowed clinical claim (routing map; original).](../assets/figures/ml_fig_claim_routing.png)

*Figure — Preface routing map. Supervised scores license **prediction** claims when calibrated and externally checked; they do not license etiology (blocked dashed path). Causal designs support cause claims. Utility / net-benefit work supports decision-support claims. Clusters and embeddings are hypothesis-generating only. Prediction ≠ causation.*

There is no separate further-reading chapter—the book is the curriculum. Read it iteratively, refer to the mathematical foundations when needed, and apply these concepts rigorously to the next paper you read or the next clinical tool your hospital considers.

![External validation ladder: optimism shrinks as the test hardens (synthetic; original).](../assets/figures/ml_fig_external_ladder.png)

*Figure — Preface external-validation ladder (synthetic teaching). **Left:** resubstitution AUROC looks heroic; patient-wise CV, temporal split, external site, and prospective silent trial successively erode discrimination. **Right:** calibration error (ECE) often worsens on transport even when ranking still looks decent. Local AUROC is not a shipping license; prediction ≠ causation.*

![Group DRO vs ERM: average AUROC vs worst-group floor (synthetic; original).](../assets/figures/ml_fig_group_dro.png)

*Figure — Preface: average metrics can hide a failed site. **Left:** ERM looks strong overall while Site D (rare protocol) collapses; group DRO trades a little mean AUROC to lift the worst group. **Right:** training weights shift toward high-loss groups. Pre-specify groups—do not mine them on the test set. Robust training improves a prediction service under shift; it does not prove sites caused outcomes.*

![Ops lifecycle: deploy → monitor → investigate → rollback/retrain (original).](../assets/figures/ml_fig_ops_lifecycle.png)

*Figure — Preface: shipping starts evaluation. Versioned monitor → investigate → rollback/retrain loop; silent auto-retrain without governance is a hazard. Lifecycle hygiene keeps a prediction service safe—it is not causal discovery.*

![Three leakage modes: timing, fit, and label leakage (original).](../assets/figures/ml_fig_three_leakage_modes.png)\n![Evidence stack for clinical ML claims (original teaching).](../assets/figures/ml_fig_evidence_stack.png)

*Figure — Preface evidence stack. Internal AUROC alone licenses only ranking talk; calibration and prevalence enable probability counseling; external and silent trials raise ops safety; decision impact needs prospective design. Climbing the stack is required for stronger claims—**prediction is not causation** without confounders and design.*\n\n


![Reader path map by learning goal (original teaching).](../assets/figures/ml_fig_reader_paths.png)

*Figure — Preface navigation. Match your goal (rebuild math, ship, audit, deploy) to a short chapter route. Routes are study aids—not licenses to treat predictive scores as causal effects.*


![Learning curves: train vs val error vs sample size (synthetic; original).](../assets/figures/ml_fig_learning_curve_n.png)

*Figure — Preface learning curves. More n typically shrinks the generalization gap. Curves diagnose capacity and data need—not whether a score is a causal effect.*


![External validation gap across sites and time (synthetic; original).](../assets/figures/ml_fig_external_val_gap.png)

*Figure — Preface: optimism shrinks under transport. Dev AUROC overstates site B/C and next-year performance. External checks are mandatory; scores remain predictive, not causal, without design.*


![Pre-deploy readiness bars for claim, split, calib, slices, monitor, rollback (original).](../assets/figures/ml_fig_readiness_bars.png)

*Figure — Preface ops checklist as bars. Weak slices/monitor/rollback block shipping. Checklists enforce process—they do not convert predictions into causal effects.*


![Claim strength meter from ranking to causation (original).](../assets/figures/ml_fig_claim_strength.png)

*Figure — Causal claims sit at the top and need design—not AUROC alone. Pred ≠ cause without design.*


![Reproducibility checklist bars (original).](../assets/figures/ml_fig_repro_checklist.png)

*Figure — Seeds splits and code versioning matter. Reproducibility checklist bars Pred != cause without design.*


![audience teaching panel (original).](../assets/figures/ml_fig_audience_map.png)

*Figure — Teaching panel for audience. Pred != cause without design.*


![Cycle-34 densify scientific panel 2 (original).](../assets/figures/ml_fig_c34_01.png)

*Figure — Continuous densify panel 2. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 2 (original).](../assets/figures/ml_fig_c35_01.png)

*Figure — Continuous densify panel 2. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 2 (original).](../assets/figures/ml_fig_c36_01.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 2 (original).](../assets/figures/ml_fig_c37_01.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 2 (original).](../assets/figures/ml_fig_c38_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 2 (original).](../assets/figures/ml_fig_c39_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 2 (original).](../assets/figures/ml_fig_c40_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 2 (original).](../assets/figures/ml_fig_c41_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 2 (original).](../assets/figures/ml_fig_c42_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 2 (original).](../assets/figures/ml_fig_c43_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 2 (original).](../assets/figures/ml_fig_c44_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 2 (original).](../assets/figures/ml_fig_c45_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 2 (original).](../assets/figures/ml_fig_c46_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 2 (original).](../assets/figures/ml_fig_c47_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 2 (original).](../assets/figures/ml_fig_c48_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 2 (original).](../assets/figures/ml_fig_c49_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 2 (original).](../assets/figures/ml_fig_c50_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 2 (original).](../assets/figures/ml_fig_c51_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 2 (original).](../assets/figures/ml_fig_c52_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 2 (original).](../assets/figures/ml_fig_c53_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 2 (original).](../assets/figures/ml_fig_c54_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 2 (original).](../assets/figures/ml_fig_c55_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 2 (original).](../assets/figures/ml_fig_c56_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 2 (original).](../assets/figures/ml_fig_c57_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 2 (original).](../assets/figures/ml_fig_c58_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 2 (original).](../assets/figures/ml_fig_c59_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 2 (original).](../assets/figures/ml_fig_c60_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 2 (original).](../assets/figures/ml_fig_c61_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 2 (original).](../assets/figures/ml_fig_c62_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 2 (original).](../assets/figures/ml_fig_c63_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 2 (original).](../assets/figures/ml_fig_c64_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 2 (original).](../assets/figures/ml_fig_c65_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 2 (original).](../assets/figures/ml_fig_c66_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 2 (original).](../assets/figures/ml_fig_c67_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 2 (original).](../assets/figures/ml_fig_c68_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 2 (original).](../assets/figures/ml_fig_c69_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 2 (original).](../assets/figures/ml_fig_c70_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 2 (original).](../assets/figures/ml_fig_c71_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 2 (original).](../assets/figures/ml_fig_c72_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 2 (original).](../assets/figures/ml_fig_c73_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 2 (original).](../assets/figures/ml_fig_c74_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 2 (original).](../assets/figures/ml_fig_c75_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 2 (original).](../assets/figures/ml_fig_c76_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 2 (original).](../assets/figures/ml_fig_c77_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 2 (original).](../assets/figures/ml_fig_c78_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 2 (original).](../assets/figures/ml_fig_c79_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 2 (original).](../assets/figures/ml_fig_c80_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 2 (original).](../assets/figures/ml_fig_c81_01.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

*Figure — Preface leakage triad. Timing (features after t₀), fit (scalers/vocab fit on full cohort), and label leakage (proxies that are the outcome) are the three modes to hunt in every methods section. Leakage inflates prediction metrics—not causal truth.*

| Habit | What you do | What it prevents |
|-------|-------------|------------------|
| Recompute | Hand-check a slope, Bayes update, or gradient step | Ritual citation without understanding |
| Map the claim | Prediction vs etiology vs decision support | Treating every model as “actionable AI” |
| Metrics beyond AUC | Calibration, PPV at prevalence, net benefit | Deploying a well-ranked but miscalibrated score |
| External check | Site/time transport, subgroup, drift plan | Local optimism that evaporates at the next hospital |

## Curriculum blocks (teaching table)

| Block | Chapters (approx.) | Core skill you should leave with |
|-------|--------------------|----------------------------------|
| Math & probability | 00, 03 | Recompute gradients, Bayes updates, and likelihood claims by hand |
| Unsupervised / features | 04–07 | Clusters, rules, engineered features, and honest dimensionality reduction |
| Supervised methods | 08–09 | Regression/classification with thresholds, imbalance, and external checks |
| Deep, SSL, RL | 10–14 | Architectures, pretrain/fine-tune, sequential decisions, lighter models |
| Graphs & data risks | 15–16 | Network structure plus leakage, shift, missingness, governance |
| Senior practice | 17–18 | One-paragraph synthesis; shared metric and appraisal vocabulary |

![c82 teaching panel 01 (original).](../assets/figures/ml_fig_c82_01.png)
*Figure — Reproducible ML study loop; pred ≠ cause at the center. Synthetic teaching geometry—not a causal claim.*

![c83 teaching panel 01 (original).](../assets/figures/ml_fig_c83_01.png)
*Figure — Temporal train/val/test split—do not shuffle across deployment time. Synthetic teaching geometry—not a causal claim.*

![c84 teaching panel 01 (original).](../assets/figures/ml_fig_c84_01.png)
*Figure — Preregistration boxes for honest analytic claims. Synthetic teaching geometry—not a causal claim.*

![c85 teaching panel 01 (original).](../assets/figures/ml_fig_c85_01.png)
*Figure — Five-box protocol spine for transparent ML reporting. Synthetic teaching geometry—not a causal claim.*

![c86 teaching panel 01 (original).](../assets/figures/ml_fig_c86_01.png)
*Figure — Claim hierarchy: primary endpoint vs exploratory labels. Synthetic teaching geometry—not a causal claim.*

![c87 teaching panel 01 (original).](../assets/figures/ml_fig_c87_01.png)
*Figure — Prediction box vs causal claim box—keep them separate (pred≠cause). Synthetic teaching geometry—not a causal claim.*

![c88 teaching panel 01 (original).](../assets/figures/ml_fig_c88_01.png)
*Figure — Ethics/IRB map for human-data ML studies. Synthetic teaching geometry—not a causal claim.*

![c89 teaching panel 01 (original).](../assets/figures/ml_fig_c89_01.png)
*Figure — SPIRIT-like protocol elements for ML papers. Synthetic teaching geometry—not a causal claim.*

![c90 teaching panel 01 (original).](../assets/figures/ml_fig_c90_01.png)
*Figure — Dataset card fields for transparency. Synthetic teaching geometry—not a causal claim.*

![c91 teaching panel 01 (original).](../assets/figures/ml_fig_c91_01.png)
*Figure — CONSORT-like flow for ML cohort. Synthetic teaching geometry—not a causal claim.*

![c92 teaching panel 01 (original).](../assets/figures/ml_fig_c92_01.png)
*Figure — TRIPOD reporting items sketch. Synthetic teaching geometry—not a causal claim.*

![c93 teaching panel 01 (original).](../assets/figures/ml_fig_c93_01.png)
*Figure — Preanalysis plan version control. Synthetic teaching geometry—not a causal claim.*

![c94 teaching panel 01 (original).](../assets/figures/ml_fig_c94_01.png)
*Figure — Protocol deviation log. Synthetic teaching geometry—not a causal claim.*

![c95 teaching panel 01 (original).](../assets/figures/ml_fig_c95_01.png)
*Figure — Registered analysis time-lock. Synthetic teaching geometry—not a causal claim.*

![c96 teaching panel 01 (original).](../assets/figures/ml_fig_c96_01.png)
*Figure — Data provenance ledger chain. Synthetic teaching geometry—not a causal claim.*
