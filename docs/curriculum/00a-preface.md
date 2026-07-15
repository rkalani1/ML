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

![c97 teaching panel 01 (original).](../assets/figures/ml_fig_c97_01.png)
*Figure — Ethics review amendment trail. Synthetic teaching geometry—not a causal claim.*

![c98 teaching panel 01 (original).](../assets/figures/ml_fig_c98_01.png)
*Figure — FAIR data principles tiles. Synthetic teaching geometry—not a causal claim.*

![c99 teaching panel 01 (original).](../assets/figures/ml_fig_c99_01.png)
*Figure — Versioned feature store timeline. Synthetic teaching geometry—not a causal claim.*

![c100 teaching panel 01 (original).](../assets/figures/ml_fig_c100_01.png)
*Figure — Assent vs consent documentation. Synthetic teaching geometry—not a causal claim.*

![c101 teaching panel 01 (original).](../assets/figures/ml_fig_c101_01.png)
*Figure — CARE checklist tiles. Synthetic teaching geometry—not a causal claim.*

![c102 teaching panel 01 (original).](../assets/figures/ml_fig_c102_01.png)
*Figure — Audit trail hash chain. Synthetic teaching geometry—not a causal claim.*

![c103 teaching panel 01 (original).](../assets/figures/ml_fig_c103_01.png)
*Figure — Stakeholder RACI for ML ops. Synthetic teaching geometry—not a causal claim.*

![c104 teaching panel 01 (original).](../assets/figures/ml_fig_c104_01.png)
*Figure — Model facts label nutrition. Synthetic teaching geometry—not a causal claim.*

![c105 teaching panel 01 (original).](../assets/figures/ml_fig_c105_01.png)
*Figure — Lineage graph of datasets. Synthetic teaching geometry—not a causal claim.*

![c106 teaching panel 01 (original).](../assets/figures/ml_fig_c106_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c107 teaching panel 01 (original).](../assets/figures/ml_fig_c107_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c108 teaching panel 01 (original).](../assets/figures/ml_fig_c108_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c109 teaching panel 01 (original).](../assets/figures/ml_fig_c109_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c110 teaching panel 01 (original).](../assets/figures/ml_fig_c110_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c111 teaching panel 01 (original).](../assets/figures/ml_fig_c111_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c112 teaching panel 01 (original).](../assets/figures/ml_fig_c112_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c113 teaching panel 01 (original).](../assets/figures/ml_fig_c113_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c114 teaching panel 01 (original).](../assets/figures/ml_fig_c114_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c115 teaching panel 01 (original).](../assets/figures/ml_fig_c115_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c116 teaching panel 01 (original).](../assets/figures/ml_fig_c116_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c117 teaching panel 01 (original).](../assets/figures/ml_fig_c117_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c118 teaching panel 01 (original).](../assets/figures/ml_fig_c118_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c119 teaching panel 01 (original).](../assets/figures/ml_fig_c119_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c120 teaching panel 01 (original).](../assets/figures/ml_fig_c120_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c121 teaching panel 01 (original).](../assets/figures/ml_fig_c121_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c122 teaching panel 01 (original).](../assets/figures/ml_fig_c122_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c123 teaching panel 01 (original).](../assets/figures/ml_fig_c123_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c124 teaching panel 01 (original).](../assets/figures/ml_fig_c124_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c125 teaching panel 01 (original).](../assets/figures/ml_fig_c125_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c126 teaching panel 01 (original).](../assets/figures/ml_fig_c126_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c127 teaching panel 01 (original).](../assets/figures/ml_fig_c127_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c128 teaching panel 01 (original).](../assets/figures/ml_fig_c128_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c129 teaching panel 01 (original).](../assets/figures/ml_fig_c129_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c130 teaching panel 01 (original).](../assets/figures/ml_fig_c130_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c131 teaching panel 01 (original).](../assets/figures/ml_fig_c131_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c132 teaching panel 01 (original).](../assets/figures/ml_fig_c132_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c133 teaching panel 01 (original).](../assets/figures/ml_fig_c133_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c134 teaching panel 01 (original).](../assets/figures/ml_fig_c134_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c135 teaching panel 01 (original).](../assets/figures/ml_fig_c135_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c136 teaching panel 01 (original).](../assets/figures/ml_fig_c136_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c137 teaching panel 01 (original).](../assets/figures/ml_fig_c137_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c138 teaching panel 01 (original).](../assets/figures/ml_fig_c138_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c139 teaching panel 01 (original).](../assets/figures/ml_fig_c139_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c140 teaching panel 01 (original).](../assets/figures/ml_fig_c140_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c141 teaching panel 01 (original).](../assets/figures/ml_fig_c141_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c142 teaching panel 01 (original).](../assets/figures/ml_fig_c142_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c143 teaching panel 01 (original).](../assets/figures/ml_fig_c143_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c144 teaching panel 01 (original).](../assets/figures/ml_fig_c144_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c145 teaching panel 01 (original).](../assets/figures/ml_fig_c145_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c146 teaching panel 01 (original).](../assets/figures/ml_fig_c146_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c147 teaching panel 01 (original).](../assets/figures/ml_fig_c147_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c148 teaching panel 01 (original).](../assets/figures/ml_fig_c148_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c149 teaching panel 01 (original).](../assets/figures/ml_fig_c149_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c150 teaching panel 01 (original).](../assets/figures/ml_fig_c150_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c151 teaching panel 01 (original).](../assets/figures/ml_fig_c151_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c152 teaching panel 01 (original).](../assets/figures/ml_fig_c152_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c153 teaching panel 01 (original).](../assets/figures/ml_fig_c153_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c154 teaching panel 01 (original).](../assets/figures/ml_fig_c154_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c155 teaching panel 01 (original).](../assets/figures/ml_fig_c155_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c156 teaching panel 01 (original).](../assets/figures/ml_fig_c156_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c157 teaching panel 01 (original).](../assets/figures/ml_fig_c157_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c158 teaching panel 01 (original).](../assets/figures/ml_fig_c158_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c159 teaching panel 01 (original).](../assets/figures/ml_fig_c159_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c160 teaching panel 01 (original).](../assets/figures/ml_fig_c160_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c161 teaching panel 01 (original).](../assets/figures/ml_fig_c161_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c162 teaching panel 01 (original).](../assets/figures/ml_fig_c162_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c163 teaching panel 01 (original).](../assets/figures/ml_fig_c163_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c164 teaching panel 01 (original).](../assets/figures/ml_fig_c164_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c165 teaching panel 01 (original).](../assets/figures/ml_fig_c165_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c166 teaching panel 01 (original).](../assets/figures/ml_fig_c166_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c167 teaching panel 01 (original).](../assets/figures/ml_fig_c167_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c168 teaching panel 01 (original).](../assets/figures/ml_fig_c168_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c169 teaching panel 01 (original).](../assets/figures/ml_fig_c169_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c170 teaching panel 01 (original).](../assets/figures/ml_fig_c170_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c171 teaching panel 01 (original).](../assets/figures/ml_fig_c171_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c172 teaching panel 01 (original).](../assets/figures/ml_fig_c172_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c173 teaching panel 01 (original).](../assets/figures/ml_fig_c173_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c174 teaching panel 01 (original).](../assets/figures/ml_fig_c174_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c175 teaching panel 01 (original).](../assets/figures/ml_fig_c175_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c176 teaching panel 01 (original).](../assets/figures/ml_fig_c176_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c177 teaching panel 01 (original).](../assets/figures/ml_fig_c177_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c178 teaching panel 01 (original).](../assets/figures/ml_fig_c178_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c179 teaching panel 01 (original).](../assets/figures/ml_fig_c179_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c180 teaching panel 01 (original).](../assets/figures/ml_fig_c180_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c181 teaching panel 01 (original).](../assets/figures/ml_fig_c181_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c182 teaching panel 01 (original).](../assets/figures/ml_fig_c182_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c183 teaching panel 01 (original).](../assets/figures/ml_fig_c183_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c184 teaching panel 01 (original).](../assets/figures/ml_fig_c184_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c185 teaching panel 01 (original).](../assets/figures/ml_fig_c185_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c186 teaching panel 01 (original).](../assets/figures/ml_fig_c186_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c187 teaching panel 01 (original).](../assets/figures/ml_fig_c187_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c188 teaching panel 01 (original).](../assets/figures/ml_fig_c188_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c189 teaching panel 01 (original).](../assets/figures/ml_fig_c189_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c190 teaching panel 01 (original).](../assets/figures/ml_fig_c190_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c191 teaching panel 01 (original).](../assets/figures/ml_fig_c191_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c192 teaching panel 01 (original).](../assets/figures/ml_fig_c192_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c193 teaching panel 01 (original).](../assets/figures/ml_fig_c193_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c194 teaching panel 01 (original).](../assets/figures/ml_fig_c194_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c195 teaching panel 01 (original).](../assets/figures/ml_fig_c195_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c196 teaching panel 01 (original).](../assets/figures/ml_fig_c196_01.png)
*Figure — Stakeholder map for data use. Synthetic teaching geometry—not a causal claim.*

![c197 teaching panel 01 (original).](../assets/figures/ml_fig_c197_01.png)
*Figure — Minimal risk pathway. Synthetic teaching geometry—not a causal claim.*

![c198 teaching panel 01 (original).](../assets/figures/ml_fig_c198_01.png)
*Figure — Secondary use rules. Synthetic teaching geometry—not a causal claim.*

![c199 teaching panel 01 (original).](../assets/figures/ml_fig_c199_01.png)
*Figure — De-identification checklist. Synthetic teaching geometry—not a causal claim.*

![c200 teaching panel 01 (original).](../assets/figures/ml_fig_c200_01.png)
*Figure — Data use agreement boxes. Synthetic teaching geometry—not a causal claim.*

![c201 teaching panel 01 (original).](../assets/figures/ml_fig_c201_01.png)
*Figure — Intended use statement blocks. Synthetic teaching geometry—not a causal claim.*

![c202 teaching panel 01 (original).](../assets/figures/ml_fig_c202_01.png)
*Figure — Risk tier control ladder. Synthetic teaching geometry—not a causal claim.*

![c203 teaching panel 01 (original).](../assets/figures/ml_fig_c203_01.png)
*Figure — IRB pathway decision tiles. Synthetic teaching geometry—not a causal claim.*

![c204 teaching panel 01 (original).](../assets/figures/ml_fig_c204_01.png)
*Figure — Dataset card field tiles. Synthetic teaching geometry—not a causal claim.*

![c205 teaching panel 01 (original).](../assets/figures/ml_fig_c205_01.png)
*Figure — Purpose-limited consent stack. Synthetic teaching geometry—not a causal claim.*

![c206 teaching panel 01 (original).](../assets/figures/ml_fig_c206_01.png)
*Figure — Fairness slice TPR FPR. Synthetic teaching geometry—not a causal claim.*

![c207 teaching panel 01 (original).](../assets/figures/ml_fig_c207_01.png)
*Figure — Minimal risk de-identification map. Synthetic teaching geometry—not a causal claim.*

![c208 teaching panel 01 (original).](../assets/figures/ml_fig_c208_01.png)
*Figure — Federated silo average rounds. Synthetic teaching geometry—not a causal claim.*

![c209 teaching panel 01 (original).](../assets/figures/ml_fig_c209_01.png)
*Figure — Differential privacy budget composition. Synthetic teaching geometry—not a causal claim.*

![c210 teaching panel 01 (original).](../assets/figures/ml_fig_c210_01.png)
*Figure — Model card risk field tiles. Synthetic teaching geometry—not a causal claim.*

![c211 teaching panel 01 (original).](../assets/figures/ml_fig_c211_01.png)
*Figure — DPIA residual risk workflow. Synthetic teaching geometry—not a causal claim.*

![c212 teaching panel 01 (original).](../assets/figures/ml_fig_c212_01.png)
*Figure — Consent window timeline markers. Synthetic teaching geometry—not a causal claim.*

![c213 teaching panel 01 (original).](../assets/figures/ml_fig_c213_01.png)
*Figure — HIPAA Safe Harbor ID strip. Synthetic teaching geometry—not a causal claim.*

![c214 teaching panel 01 (original).](../assets/figures/ml_fig_c214_01.png)
*Figure — Data protection role tiles. Synthetic teaching geometry—not a causal claim.*

![c215 teaching panel 01 (original).](../assets/figures/ml_fig_c215_01.png)
*Figure — Re-consent protocol change gate. Synthetic teaching geometry—not a causal claim.*

![c216 teaching panel 01 (original).](../assets/figures/ml_fig_c216_01.png)
*Figure — Immutable audit trail fields. Synthetic teaching geometry—not a causal claim.*

![c217 teaching panel 01 (original).](../assets/figures/ml_fig_c217_01.png)
*Figure — Breach notify seventy-two hour. Synthetic teaching geometry—not a causal claim.*

![c218 teaching panel 01 (original).](../assets/figures/ml_fig_c218_01.png)
*Figure — Data retention decay schedule. Synthetic teaching geometry—not a causal claim.*

![c219 teaching panel 01 (original).](../assets/figures/ml_fig_c219_01.png)
*Figure — Epsilon-DP utility curve. Synthetic teaching geometry—not a causal claim.*

![c220 teaching panel 01 (original).](../assets/figures/ml_fig_c220_01.png)
*Figure — Purpose limitation decision. Synthetic teaching geometry—not a causal claim.*

![c221 teaching panel 01 (original).](../assets/figures/ml_fig_c221_01.png)
*Figure — Consent and purpose decision tree. Synthetic teaching geometry—not a causal claim.*

![c222 teaching panel 01 (original).](../assets/figures/ml_fig_c222_01.png)
*Figure — Dual-use capability gate. Synthetic teaching geometry—not a causal claim.*

![c223 teaching panel 01 (original).](../assets/figures/ml_fig_c223_01.png)
*Figure — Model card section map. Synthetic teaching geometry—not a causal claim.*

![c224 teaching panel 01 (original).](../assets/figures/ml_fig_c224_01.png)
*Figure — Red-team probe patch loop. Synthetic teaching geometry—not a causal claim.*

![c225 teaching panel 01 (original).](../assets/figures/ml_fig_c225_01.png)
*Figure — Dataset datasheet field map. Synthetic teaching geometry—not a causal claim.*

![c226 teaching panel 01 (original).](../assets/figures/ml_fig_c226_01.png)
*Figure — LLM eval harness stages. Synthetic teaching geometry—not a causal claim.*

![c227 teaching panel 01 (original).](../assets/figures/ml_fig_c227_01.png)
*Figure — AI Act risk tier ladder. Synthetic teaching geometry—not a causal claim.*

![c228 teaching panel 01 (original).](../assets/figures/ml_fig_c228_01.png)
*Figure — Watermark detection score laws. Synthetic teaching geometry—not a causal claim.*

![c229 teaching panel 01 (original).](../assets/figures/ml_fig_c229_01.png)
*Figure — SBOM vulnerability match chain. Synthetic teaching geometry—not a causal claim.*

![c230 teaching panel 01 (original).](../assets/figures/ml_fig_c230_01.png)
*Figure — Incident severity histogram. Synthetic teaching geometry—not a causal claim.*

![c231 teaching panel 01 (original).](../assets/figures/ml_fig_c231_01.png)
*Figure — Model evaluation card grid. Synthetic teaching geometry—not a causal claim.*

![c232 teaching panel 01 (original).](../assets/figures/ml_fig_c232_01.png)
*Figure — Threat model asset map. Synthetic teaching geometry—not a causal claim.*

![c233 teaching panel 01 (original).](../assets/figures/ml_fig_c233_01.png)
*Figure — Privacy budget composition sum. Synthetic teaching geometry—not a causal claim.*

![c234 teaching panel 01 (original).](../assets/figures/ml_fig_c234_01.png)
*Figure — DPIA control sign-off path. Synthetic teaching geometry—not a causal claim.*

![c235 teaching panel 01 (original).](../assets/figures/ml_fig_c235_01.png)
*Figure — Advanced composition epsilon curve. Synthetic teaching geometry—not a causal claim.*

![c236 teaching panel 01 (original).](../assets/figures/ml_fig_c236_01.png)
*Figure — Model risk review path. Synthetic teaching geometry—not a causal claim.*

![c237 teaching panel 01 (original).](../assets/figures/ml_fig_c237_01.png)
*Figure — Moments accountant epsilon path. Synthetic teaching geometry—not a causal claim.*

![c238 teaching panel 01 (original).](../assets/figures/ml_fig_c238_01.png)
*Figure — Red-team findings path. Synthetic teaching geometry—not a causal claim.*

![c239 teaching panel 01 (original).](../assets/figures/ml_fig_c239_01.png)
*Figure — RDP to epsilon conversion. Synthetic teaching geometry—not a causal claim.*

![c240 teaching panel 01 (original).](../assets/figures/ml_fig_c240_01.png)
*Figure — Policy exception path. Synthetic teaching geometry—not a causal claim.*

![c241 teaching panel 01 (original).](../assets/figures/ml_fig_c241_01.png)
*Figure — zCDP rho to epsilon map. Synthetic teaching geometry—not a causal claim.*

![c242 teaching panel 01 (original).](../assets/figures/ml_fig_c242_01.png)
*Figure — Threat model control path. Synthetic teaching geometry—not a causal claim.*

![c243 teaching panel 01 (original).](../assets/figures/ml_fig_c243_01.png)
*Figure — f-DP tradeoff function curve. Synthetic teaching geometry—not a causal claim.*

![c244 teaching panel 01 (original).](../assets/figures/ml_fig_c244_01.png)
*Figure — Incident response control path. Synthetic teaching geometry—not a causal claim.*

![c245 teaching panel 01 (original).](../assets/figures/ml_fig_c245_01.png)
*Figure — GDP mu-GDP conversion curve. Synthetic teaching geometry—not a causal claim.*

![c246 teaching panel 01 (original).](../assets/figures/ml_fig_c246_01.png)
*Figure — Red-team severity path. Synthetic teaching geometry—not a causal claim.*

![c247 teaching panel 01 (original).](../assets/figures/ml_fig_c247_01.png)
*Figure — Concentrated DP rho map. Synthetic teaching geometry—not a causal claim.*

![c248 teaching panel 01 (original).](../assets/figures/ml_fig_c248_01.png)
*Figure — Supply-chain attest path. Synthetic teaching geometry—not a causal claim.*

![c249 teaching panel 01 (original).](../assets/figures/ml_fig_c249_01.png)
*Figure — Hockey-stick DP conversion. Synthetic teaching geometry—not a causal claim.*

![c250 teaching panel 01 (original).](../assets/figures/ml_fig_c250_01.png)
*Figure — SBOM provenance path. Synthetic teaching geometry—not a causal claim.*

![c251 teaching panel 01 (original).](../assets/figures/ml_fig_c251_01.png)
*Figure — RDP order optimize curve. Synthetic teaching geometry—not a causal claim.*

![c252 teaching panel 01 (original).](../assets/figures/ml_fig_c252_01.png)
*Figure — Model card review path. Synthetic teaching geometry—not a causal claim.*

![c253 teaching panel 01 (original).](../assets/figures/ml_fig_c253_01.png)
*Figure — f-DP Gaussian tradeoff. Synthetic teaching geometry—not a causal claim.*

![c254 teaching panel 01 (original).](../assets/figures/ml_fig_c254_01.png)
*Figure — Incident SEV path. Synthetic teaching geometry—not a causal claim.*

![c255 teaching panel 01 (original).](../assets/figures/ml_fig_c255_01.png)
*Figure — zCDP composition map. Synthetic teaching geometry—not a causal claim.*

![c256 teaching panel 01 (original).](../assets/figures/ml_fig_c256_01.png)
*Figure — Threat model path. Synthetic teaching geometry—not a causal claim.*

![c257 teaching panel 01 (original).](../assets/figures/ml_fig_c257_01.png)
*Figure — Red-team severity path c257. Synthetic teaching geometry—not a causal claim.*

![c258 teaching panel 01 (original).](../assets/figures/ml_fig_c258_01.png)
*Figure — Incident SEV ladder path c258. Synthetic teaching geometry—not a causal claim.*

![c259 teaching panel 01 (original).](../assets/figures/ml_fig_c259_01.png)
*Figure — SBOM attest ship path c259. Synthetic teaching geometry—not a causal claim.*

![c260 teaching panel 01 (original).](../assets/figures/ml_fig_c260_01.png)
*Figure — Threat residual risk path c260. Synthetic teaching geometry—not a causal claim.*

![c261 teaching panel 01 (original).](../assets/figures/ml_fig_c261_01.png)
*Figure — Policy exception path c261. Synthetic teaching geometry—not a causal claim.*

![c262 teaching panel 01 (original).](../assets/figures/ml_fig_c262_01.png)
*Figure — Audit trail control path c262. Synthetic teaching geometry—not a causal claim.*

![c263 teaching panel 01 (original).](../assets/figures/ml_fig_c263_01.png)
*Figure — Fairness review path c263. Synthetic teaching geometry—not a causal claim.*

![c264 teaching panel 01 (original).](../assets/figures/ml_fig_c264_01.png)
*Figure — Privacy DPIA path c264. Synthetic teaching geometry—not a causal claim.*

![c265 teaching panel 01 (original).](../assets/figures/ml_fig_c265_01.png)
*Figure — Vendor risk path c265. Synthetic teaching geometry—not a causal claim.*

![c266 teaching panel 01 (original).](../assets/figures/ml_fig_c266_01.png)
*Figure — Kill-switch runbook path c266. Synthetic teaching geometry—not a causal claim.*

![c267 teaching panel 01 (original).](../assets/figures/ml_fig_c267_01.png)
*Figure — Rollback decision path c267. Synthetic teaching geometry—not a causal claim.*

![c268 teaching panel 01 (original).](../assets/figures/ml_fig_c268_01.png)
*Figure — Postmortem action path c268. Synthetic teaching geometry—not a causal claim.*

![c269 teaching panel 01 (original).](../assets/figures/ml_fig_c269_01.png)
*Figure — Data use purpose path c269. Synthetic teaching geometry—not a causal claim.*

![c270 teaching panel 01 (original).](../assets/figures/ml_fig_c270_01.png)
*Figure — Consent scope control path c270. Synthetic teaching geometry—not a causal claim.*

![c271 teaching panel 01 (original).](../assets/figures/ml_fig_c271_01.png)
*Figure — Model risk tier path c271. Synthetic teaching geometry—not a causal claim.*

![c272 teaching panel 01 (original).](../assets/figures/ml_fig_c272_01.png)
*Figure — Human oversight gate path c272. Synthetic teaching geometry—not a causal claim.*

![c273 teaching panel 01 (original).](../assets/figures/ml_fig_c273_01.png)
*Figure — Red-team severity path c273. Synthetic teaching geometry—not a causal claim.*

![c274 teaching panel 01 (original).](../assets/figures/ml_fig_c274_01.png)
*Figure — Incident SEV ladder path c274. Synthetic teaching geometry—not a causal claim.*

![c275 teaching panel 01 (original).](../assets/figures/ml_fig_c275_01.png)
*Figure — SBOM attest ship path c275. Synthetic teaching geometry—not a causal claim.*

![c276 teaching panel 01 (original).](../assets/figures/ml_fig_c276_01.png)
*Figure — Threat residual risk path c276. Synthetic teaching geometry—not a causal claim.*

![c277 teaching panel 01 (original).](../assets/figures/ml_fig_c277_01.png)
*Figure — Policy exception path c277. Synthetic teaching geometry—not a causal claim.*

![c278 teaching panel 01 (original).](../assets/figures/ml_fig_c278_01.png)
*Figure — Audit trail control path c278. Synthetic teaching geometry—not a causal claim.*

![c279 teaching panel 01 (original).](../assets/figures/ml_fig_c279_01.png)
*Figure — Fairness review path c279. Synthetic teaching geometry—not a causal claim.*

![c280 teaching panel 01 (original).](../assets/figures/ml_fig_c280_01.png)
*Figure — Privacy DPIA path c280. Synthetic teaching geometry—not a causal claim.*

![c281 teaching panel 01 (original).](../assets/figures/ml_fig_c281_01.png)
*Figure — Vendor risk path c281. Synthetic teaching geometry—not a causal claim.*

![c282 teaching panel 01 (original).](../assets/figures/ml_fig_c282_01.png)
*Figure — Kill-switch runbook path c282. Synthetic teaching geometry—not a causal claim.*

![c283 teaching panel 01 (original).](../assets/figures/ml_fig_c283_01.png)
*Figure — Rollback decision path c283. Synthetic teaching geometry—not a causal claim.*

![c284 teaching panel 01 (original).](../assets/figures/ml_fig_c284_01.png)
*Figure — Postmortem action path c284. Synthetic teaching geometry—not a causal claim.*

![c285 teaching panel 01 (original).](../assets/figures/ml_fig_c285_01.png)
*Figure — Data use purpose path c285. Synthetic teaching geometry—not a causal claim.*

![c286 teaching panel 01 (original).](../assets/figures/ml_fig_c286_01.png)
*Figure — Consent scope control path c286. Synthetic teaching geometry—not a causal claim.*

![c287 teaching panel 01 (original).](../assets/figures/ml_fig_c287_01.png)
*Figure — Model risk tier path c287. Synthetic teaching geometry—not a causal claim.*

![c288 teaching panel 01 (original).](../assets/figures/ml_fig_c288_01.png)
*Figure — Human oversight gate path c288. Synthetic teaching geometry—not a causal claim.*

![c289 teaching panel 01 (original).](../assets/figures/ml_fig_c289_01.png)
*Figure — Red-team severity path c289. Synthetic teaching geometry—not a causal claim.*

![c290 teaching panel 01 (original).](../assets/figures/ml_fig_c290_01.png)
*Figure — Incident SEV ladder path c290. Synthetic teaching geometry—not a causal claim.*

![c291 teaching panel 01 (original).](../assets/figures/ml_fig_c291_01.png)
*Figure — SBOM attest ship path c291. Synthetic teaching geometry—not a causal claim.*

![c292 teaching panel 01 (original).](../assets/figures/ml_fig_c292_01.png)
*Figure — Threat residual risk path c292. Synthetic teaching geometry—not a causal claim.*

![c293 teaching panel 01 (original).](../assets/figures/ml_fig_c293_01.png)
*Figure — Policy exception path c293. Synthetic teaching geometry—not a causal claim.*

![c294 teaching panel 01 (original).](../assets/figures/ml_fig_c294_01.png)
*Figure — Audit trail control path c294. Synthetic teaching geometry—not a causal claim.*

![c295 teaching panel 01 (original).](../assets/figures/ml_fig_c295_01.png)
*Figure — Fairness review path c295. Synthetic teaching geometry—not a causal claim.*

![c296 teaching panel 01 (original).](../assets/figures/ml_fig_c296_01.png)
*Figure — Privacy DPIA path c296. Synthetic teaching geometry—not a causal claim.*

![c297 teaching panel 01 (original).](../assets/figures/ml_fig_c297_01.png)
*Figure — Vendor risk path c297. Synthetic teaching geometry—not a causal claim.*

![c298 teaching panel 01 (original).](../assets/figures/ml_fig_c298_01.png)
*Figure — Kill-switch runbook path c298. Synthetic teaching geometry—not a causal claim.*

![c299 teaching panel 01 (original).](../assets/figures/ml_fig_c299_01.png)
*Figure — Rollback decision path c299. Synthetic teaching geometry—not a causal claim.*

![c300 teaching panel 01 (original).](../assets/figures/ml_fig_c300_01.png)
*Figure — Postmortem action path c300. Synthetic teaching geometry—not a causal claim.*

![c301 teaching panel 01 (original).](../assets/figures/ml_fig_c301_01.png)
*Figure — Data use purpose path c301. Synthetic teaching geometry—not a causal claim.*

![c302 teaching panel 01 (original).](../assets/figures/ml_fig_c302_01.png)
*Figure — Consent scope control path c302. Synthetic teaching geometry—not a causal claim.*

![c303 teaching panel 01 (original).](../assets/figures/ml_fig_c303_01.png)
*Figure — Model risk tier path c303. Synthetic teaching geometry—not a causal claim.*

![c304 teaching panel 01 (original).](../assets/figures/ml_fig_c304_01.png)
*Figure — Human oversight gate path c304. Synthetic teaching geometry—not a causal claim.*

![c305 teaching panel 01 (original).](../assets/figures/ml_fig_c305_01.png)
*Figure — Red-team severity path c305. Synthetic teaching geometry—not a causal claim.*

![c306 teaching panel 01 (original).](../assets/figures/ml_fig_c306_01.png)
*Figure — Incident SEV ladder path c306. Synthetic teaching geometry—not a causal claim.*

![c307 teaching panel 01 (original).](../assets/figures/ml_fig_c307_01.png)
*Figure — SBOM attest ship path c307. Synthetic teaching geometry—not a causal claim.*

![c308 teaching panel 01 (original).](../assets/figures/ml_fig_c308_01.png)
*Figure — Threat residual risk path c308. Synthetic teaching geometry—not a causal claim.*

![c309 teaching panel 01 (original).](../assets/figures/ml_fig_c309_01.png)
*Figure — Policy exception path c309. Synthetic teaching geometry—not a causal claim.*

![c310 teaching panel 01 (original).](../assets/figures/ml_fig_c310_01.png)
*Figure — Audit trail control path c310. Synthetic teaching geometry—not a causal claim.*

![c311 teaching panel 01 (original).](../assets/figures/ml_fig_c311_01.png)
*Figure — Fairness review path c311. Synthetic teaching geometry—not a causal claim.*

![c312 teaching panel 01 (original).](../assets/figures/ml_fig_c312_01.png)
*Figure — Privacy DPIA path c312. Synthetic teaching geometry—not a causal claim.*

![c313 teaching panel 01 (original).](../assets/figures/ml_fig_c313_01.png)
*Figure — Vendor risk path c313. Synthetic teaching geometry—not a causal claim.*

![c314 teaching panel 01 (original).](../assets/figures/ml_fig_c314_01.png)
*Figure — Kill-switch runbook path c314. Synthetic teaching geometry—not a causal claim.*

![c315 teaching panel 01 (original).](../assets/figures/ml_fig_c315_01.png)
*Figure — Rollback decision path c315. Synthetic teaching geometry—not a causal claim.*

![c316 teaching panel 01 (original).](../assets/figures/ml_fig_c316_01.png)
*Figure — Postmortem action path c316. Synthetic teaching geometry—not a causal claim.*

![c317 teaching panel 01 (original).](../assets/figures/ml_fig_c317_01.png)
*Figure — Data use purpose path c317. Synthetic teaching geometry—not a causal claim.*

![c318 teaching panel 01 (original).](../assets/figures/ml_fig_c318_01.png)
*Figure — Consent scope control path c318. Synthetic teaching geometry—not a causal claim.*

![c319 teaching panel 01 (original).](../assets/figures/ml_fig_c319_01.png)
*Figure — Model risk tier path c319. Synthetic teaching geometry—not a causal claim.*

![c320 teaching panel 01 (original).](../assets/figures/ml_fig_c320_01.png)
*Figure — Human oversight gate path c320. Synthetic teaching geometry—not a causal claim.*

![c321 teaching panel 01 (original).](../assets/figures/ml_fig_c321_01.png)
*Figure — Red-team severity path c321. Synthetic teaching geometry—not a causal claim.*

![c322 teaching panel 01 (original).](../assets/figures/ml_fig_c322_01.png)
*Figure — Incident SEV ladder path c322. Synthetic teaching geometry—not a causal claim.*

![c323 teaching panel 01 (original).](../assets/figures/ml_fig_c323_01.png)
*Figure — SBOM attest ship path c323. Synthetic teaching geometry—not a causal claim.*

![c324 teaching panel 01 (original).](../assets/figures/ml_fig_c324_01.png)
*Figure — Threat residual risk path c324. Synthetic teaching geometry—not a causal claim.*

![c325 teaching panel 01 (original).](../assets/figures/ml_fig_c325_01.png)
*Figure — Policy exception path c325. Synthetic teaching geometry—not a causal claim.*

![c326 teaching panel 01 (original).](../assets/figures/ml_fig_c326_01.png)
*Figure — Audit trail control path c326. Synthetic teaching geometry—not a causal claim.*

![c327 teaching panel 01 (original).](../assets/figures/ml_fig_c327_01.png)
*Figure — Fairness review path c327. Synthetic teaching geometry—not a causal claim.*

![c328 teaching panel 01 (original).](../assets/figures/ml_fig_c328_01.png)
*Figure — Privacy DPIA path c328. Synthetic teaching geometry—not a causal claim.*

![c329 teaching panel 01 (original).](../assets/figures/ml_fig_c329_01.png)
*Figure — Vendor risk path c329. Synthetic teaching geometry—not a causal claim.*

![c330 teaching panel 01 (original).](../assets/figures/ml_fig_c330_01.png)
*Figure — Kill-switch runbook path c330. Synthetic teaching geometry—not a causal claim.*

![c331 teaching panel 01 (original).](../assets/figures/ml_fig_c331_01.png)
*Figure — Rollback decision path c331. Synthetic teaching geometry—not a causal claim.*

![c332 teaching panel 01 (original).](../assets/figures/ml_fig_c332_01.png)
*Figure — Postmortem action path c332. Synthetic teaching geometry—not a causal claim.*

![c333 teaching panel 01 (original).](../assets/figures/ml_fig_c333_01.png)
*Figure — Data use purpose path c333. Synthetic teaching geometry—not a causal claim.*

![c334 teaching panel 01 (original).](../assets/figures/ml_fig_c334_01.png)
*Figure — Consent scope control path c334. Synthetic teaching geometry—not a causal claim.*

![c335 teaching panel 01 (original).](../assets/figures/ml_fig_c335_01.png)
*Figure — Model risk tier path c335. Synthetic teaching geometry—not a causal claim.*

![c336 teaching panel 01 (original).](../assets/figures/ml_fig_c336_01.png)
*Figure — Human oversight gate path c336. Synthetic teaching geometry—not a causal claim.*

![c337 teaching panel 01 (original).](../assets/figures/ml_fig_c337_01.png)
*Figure — Red-team severity path c337. Synthetic teaching geometry—not a causal claim.*

![c338 teaching panel 01 (original).](../assets/figures/ml_fig_c338_01.png)
*Figure — Incident SEV ladder path c338. Synthetic teaching geometry—not a causal claim.*

![c339 teaching panel 01 (original).](../assets/figures/ml_fig_c339_01.png)
*Figure — SBOM attest ship path c339. Synthetic teaching geometry—not a causal claim.*

![c340 teaching panel 01 (original).](../assets/figures/ml_fig_c340_01.png)
*Figure — Threat residual risk path c340. Synthetic teaching geometry—not a causal claim.*

![c341 teaching panel 01 (original).](../assets/figures/ml_fig_c341_01.png)
*Figure — Policy exception path c341. Synthetic teaching geometry—not a causal claim.*

![c342 teaching panel 01 (original).](../assets/figures/ml_fig_c342_01.png)
*Figure — Audit trail control path c342. Synthetic teaching geometry—not a causal claim.*

![c343 teaching panel 01 (original).](../assets/figures/ml_fig_c343_01.png)
*Figure — Fairness review path c343. Synthetic teaching geometry—not a causal claim.*

![c344 teaching panel 01 (original).](../assets/figures/ml_fig_c344_01.png)
*Figure — Privacy DPIA path c344. Synthetic teaching geometry—not a causal claim.*

![c345 teaching panel 01 (original).](../assets/figures/ml_fig_c345_01.png)
*Figure — Vendor risk path c345. Synthetic teaching geometry—not a causal claim.*

![c346 teaching panel 01 (original).](../assets/figures/ml_fig_c346_01.png)
*Figure — Kill-switch runbook path c346. Synthetic teaching geometry—not a causal claim.*

![c347 teaching panel 01 (original).](../assets/figures/ml_fig_c347_01.png)
*Figure — Rollback decision path c347. Synthetic teaching geometry—not a causal claim.*

![c348 teaching panel 01 (original).](../assets/figures/ml_fig_c348_01.png)
*Figure — Postmortem action path c348. Synthetic teaching geometry—not a causal claim.*

![c349 teaching panel 01 (original).](../assets/figures/ml_fig_c349_01.png)
*Figure — Data use purpose path c349. Synthetic teaching geometry—not a causal claim.*

![c350 teaching panel 01 (original).](../assets/figures/ml_fig_c350_01.png)
*Figure — Consent scope control path c350. Synthetic teaching geometry—not a causal claim.*

![c351 teaching panel 01 (original).](../assets/figures/ml_fig_c351_01.png)
*Figure — Model risk tier path c351. Synthetic teaching geometry—not a causal claim.*

![c352 teaching panel 01 (original).](../assets/figures/ml_fig_c352_01.png)
*Figure — Human oversight gate path c352. Synthetic teaching geometry—not a causal claim.*

![c353 teaching panel 01 (original).](../assets/figures/ml_fig_c353_01.png)
*Figure — Red-team severity path c353. Synthetic teaching geometry—not a causal claim.*

![c354 teaching panel 01 (original).](../assets/figures/ml_fig_c354_01.png)
*Figure — Incident SEV ladder path c354. Synthetic teaching geometry—not a causal claim.*

![c355 teaching panel 01 (original).](../assets/figures/ml_fig_c355_01.png)
*Figure — SBOM attest ship path c355. Synthetic teaching geometry—not a causal claim.*

![c356 teaching panel 01 (original).](../assets/figures/ml_fig_c356_01.png)
*Figure — Threat residual risk path c356. Synthetic teaching geometry—not a causal claim.*

![c357 teaching panel 01 (original).](../assets/figures/ml_fig_c357_01.png)
*Figure — Policy exception path c357. Synthetic teaching geometry—not a causal claim.*

![c358 teaching panel 01 (original).](../assets/figures/ml_fig_c358_01.png)
*Figure — Audit trail control path c358. Synthetic teaching geometry—not a causal claim.*

![c359 teaching panel 01 (original).](../assets/figures/ml_fig_c359_01.png)
*Figure — Fairness review path c359. Synthetic teaching geometry—not a causal claim.*

![c360 teaching panel 01 (original).](../assets/figures/ml_fig_c360_01.png)
*Figure — Privacy DPIA path c360. Synthetic teaching geometry—not a causal claim.*

![c361 teaching panel 01 (original).](../assets/figures/ml_fig_c361_01.png)
*Figure — Vendor risk path c361. Synthetic teaching geometry—not a causal claim.*

![c362 teaching panel 01 (original).](../assets/figures/ml_fig_c362_01.png)
*Figure — Kill-switch runbook path c362. Synthetic teaching geometry—not a causal claim.*

![c363 teaching panel 01 (original).](../assets/figures/ml_fig_c363_01.png)
*Figure — Rollback decision path c363. Synthetic teaching geometry—not a causal claim.*

![c364 teaching panel 01 (original).](../assets/figures/ml_fig_c364_01.png)
*Figure — Postmortem action path c364. Synthetic teaching geometry—not a causal claim.*

![c365 teaching panel 01 (original).](../assets/figures/ml_fig_c365_01.png)
*Figure — Data use purpose path c365. Synthetic teaching geometry—not a causal claim.*

![c366 teaching panel 01 (original).](../assets/figures/ml_fig_c366_01.png)
*Figure — Consent scope control path c366. Synthetic teaching geometry—not a causal claim.*

![c367 teaching panel 01 (original).](../assets/figures/ml_fig_c367_01.png)
*Figure — Model risk tier path c367. Synthetic teaching geometry—not a causal claim.*

![c368 teaching panel 01 (original).](../assets/figures/ml_fig_c368_01.png)
*Figure — Human oversight gate path c368. Synthetic teaching geometry—not a causal claim.*

![c369 teaching panel 01 (original).](../assets/figures/ml_fig_c369_01.png)
*Figure — Red-team severity path c369. Synthetic teaching geometry—not a causal claim.*

![c370 teaching panel 01 (original).](../assets/figures/ml_fig_c370_01.png)
*Figure — Incident SEV ladder path c370. Synthetic teaching geometry—not a causal claim.*

![c371 teaching panel 01 (original).](../assets/figures/ml_fig_c371_01.png)
*Figure — SBOM attest ship path c371. Synthetic teaching geometry—not a causal claim.*

![c372 teaching panel 01 (original).](../assets/figures/ml_fig_c372_01.png)
*Figure — Threat residual risk path c372. Synthetic teaching geometry—not a causal claim.*

![c373 teaching panel 01 (original).](../assets/figures/ml_fig_c373_01.png)
*Figure — Policy exception path c373. Synthetic teaching geometry—not a causal claim.*

![c374 teaching panel 01 (original).](../assets/figures/ml_fig_c374_01.png)
*Figure — Audit trail control path c374. Synthetic teaching geometry—not a causal claim.*

![c375 teaching panel 01 (original).](../assets/figures/ml_fig_c375_01.png)
*Figure — Fairness review path c375. Synthetic teaching geometry—not a causal claim.*

![c376 teaching panel 01 (original).](../assets/figures/ml_fig_c376_01.png)
*Figure — Privacy DPIA path c376. Synthetic teaching geometry—not a causal claim.*

![c377 teaching panel 01 (original).](../assets/figures/ml_fig_c377_01.png)
*Figure — Vendor risk path c377. Synthetic teaching geometry—not a causal claim.*

![c378 teaching panel 01 (original).](../assets/figures/ml_fig_c378_01.png)
*Figure — Kill-switch runbook path c378. Synthetic teaching geometry—not a causal claim.*

![c379 teaching panel 01 (original).](../assets/figures/ml_fig_c379_01.png)
*Figure — Rollback decision path c379. Synthetic teaching geometry—not a causal claim.*

![c380 teaching panel 01 (original).](../assets/figures/ml_fig_c380_01.png)
*Figure — Postmortem action path c380. Synthetic teaching geometry—not a causal claim.*

![c381 teaching panel 01 (original).](../assets/figures/ml_fig_c381_01.png)
*Figure — Data use purpose path c381. Synthetic teaching geometry—not a causal claim.*

![c382 teaching panel 01 (original).](../assets/figures/ml_fig_c382_01.png)
*Figure — Consent scope control path c382. Synthetic teaching geometry—not a causal claim.*

![c383 teaching panel 01 (original).](../assets/figures/ml_fig_c383_01.png)
*Figure — Model risk tier path c383. Synthetic teaching geometry—not a causal claim.*

![c384 teaching panel 01 (original).](../assets/figures/ml_fig_c384_01.png)
*Figure — Human oversight gate path c384. Synthetic teaching geometry—not a causal claim.*

![c385 teaching panel 01 (original).](../assets/figures/ml_fig_c385_01.png)
*Figure — Red-team severity path c385. Synthetic teaching geometry—not a causal claim.*

![c386 teaching panel 01 (original).](../assets/figures/ml_fig_c386_01.png)
*Figure — Incident SEV ladder path c386. Synthetic teaching geometry—not a causal claim.*

![c387 teaching panel 01 (original).](../assets/figures/ml_fig_c387_01.png)
*Figure — SBOM attest ship path c387. Synthetic teaching geometry—not a causal claim.*

![c388 teaching panel 01 (original).](../assets/figures/ml_fig_c388_01.png)
*Figure — Threat residual risk path c388. Synthetic teaching geometry—not a causal claim.*

![c389 teaching panel 01 (original).](../assets/figures/ml_fig_c389_01.png)
*Figure — Policy exception path c389. Synthetic teaching geometry—not a causal claim.*

![c390 teaching panel 01 (original).](../assets/figures/ml_fig_c390_01.png)
*Figure — Audit trail control path c390. Synthetic teaching geometry—not a causal claim.*

![c391 teaching panel 01 (original).](../assets/figures/ml_fig_c391_01.png)
*Figure — Fairness review path c391. Synthetic teaching geometry—not a causal claim.*

![c392 teaching panel 01 (original).](../assets/figures/ml_fig_c392_01.png)
*Figure — Privacy DPIA path c392. Synthetic teaching geometry—not a causal claim.*

![c393 teaching panel 01 (original).](../assets/figures/ml_fig_c393_01.png)
*Figure — Vendor risk path c393. Synthetic teaching geometry—not a causal claim.*

![c394 teaching panel 01 (original).](../assets/figures/ml_fig_c394_01.png)
*Figure — Kill-switch runbook path c394. Synthetic teaching geometry—not a causal claim.*

![c395 teaching panel 01 (original).](../assets/figures/ml_fig_c395_01.png)
*Figure — Rollback decision path c395. Synthetic teaching geometry—not a causal claim.*

![c396 teaching panel 01 (original).](../assets/figures/ml_fig_c396_01.png)
*Figure — Postmortem action path c396. Synthetic teaching geometry—not a causal claim.*

![c397 teaching panel 01 (original).](../assets/figures/ml_fig_c397_01.png)
*Figure — Data use purpose path c397. Synthetic teaching geometry—not a causal claim.*

![c398 teaching panel 01 (original).](../assets/figures/ml_fig_c398_01.png)
*Figure — Consent scope control path c398. Synthetic teaching geometry—not a causal claim.*

![c399 teaching panel 01 (original).](../assets/figures/ml_fig_c399_01.png)
*Figure — Model risk tier path c399. Synthetic teaching geometry—not a causal claim.*

![c400 teaching panel 01 (original).](../assets/figures/ml_fig_c400_01.png)
*Figure — Human oversight gate path c400. Synthetic teaching geometry—not a causal claim.*

![c401 teaching panel 01 (original).](../assets/figures/ml_fig_c401_01.png)
*Figure — Red-team severity path c401. Synthetic teaching geometry—not a causal claim.*

![c402 teaching panel 01 (original).](../assets/figures/ml_fig_c402_01.png)
*Figure — Incident SEV ladder path c402. Synthetic teaching geometry—not a causal claim.*

![c403 teaching panel 01 (original).](../assets/figures/ml_fig_c403_01.png)
*Figure — SBOM attest ship path c403. Synthetic teaching geometry—not a causal claim.*

![c404 teaching panel 01 (original).](../assets/figures/ml_fig_c404_01.png)
*Figure — Threat residual risk path c404. Synthetic teaching geometry—not a causal claim.*

![c405 teaching panel 01 (original).](../assets/figures/ml_fig_c405_01.png)
*Figure — Policy exception path c405. Synthetic teaching geometry—not a causal claim.*

![c406 teaching panel 01 (original).](../assets/figures/ml_fig_c406_01.png)
*Figure — Audit trail control path c406. Synthetic teaching geometry—not a causal claim.*

![c407 teaching panel 01 (original).](../assets/figures/ml_fig_c407_01.png)
*Figure — Fairness review path c407. Synthetic teaching geometry—not a causal claim.*

![c408 teaching panel 01 (original).](../assets/figures/ml_fig_c408_01.png)
*Figure — Privacy DPIA path c408. Synthetic teaching geometry—not a causal claim.*

![c409 teaching panel 01 (original).](../assets/figures/ml_fig_c409_01.png)
*Figure — Vendor risk path c409. Synthetic teaching geometry—not a causal claim.*

![c410 teaching panel 01 (original).](../assets/figures/ml_fig_c410_01.png)
*Figure — Kill-switch runbook path c410. Synthetic teaching geometry—not a causal claim.*

![c411 teaching panel 01 (original).](../assets/figures/ml_fig_c411_01.png)
*Figure — Rollback decision path c411. Synthetic teaching geometry—not a causal claim.*

![c412 teaching panel 01 (original).](../assets/figures/ml_fig_c412_01.png)
*Figure — Postmortem action path c412. Synthetic teaching geometry—not a causal claim.*

![c413 teaching panel 01 (original).](../assets/figures/ml_fig_c413_01.png)
*Figure — Data use purpose path c413. Synthetic teaching geometry—not a causal claim.*

![c414 teaching panel 01 (original).](../assets/figures/ml_fig_c414_01.png)
*Figure — Consent scope control path c414. Synthetic teaching geometry—not a causal claim.*

![c415 teaching panel 01 (original).](../assets/figures/ml_fig_c415_01.png)
*Figure — Model risk tier path c415. Synthetic teaching geometry—not a causal claim.*

![c416 teaching panel 01 (original).](../assets/figures/ml_fig_c416_01.png)
*Figure — Human oversight gate path c416. Synthetic teaching geometry—not a causal claim.*

![c417 teaching panel 01 (original).](../assets/figures/ml_fig_c417_01.png)
*Figure — Red-team severity path c417. Synthetic teaching geometry—not a causal claim.*

![c418 teaching panel 01 (original).](../assets/figures/ml_fig_c418_01.png)
*Figure — Incident SEV ladder path c418. Synthetic teaching geometry—not a causal claim.*

![c419 teaching panel 01 (original).](../assets/figures/ml_fig_c419_01.png)
*Figure — SBOM attest ship path c419. Synthetic teaching geometry—not a causal claim.*

![c420 teaching panel 01 (original).](../assets/figures/ml_fig_c420_01.png)
*Figure — Threat residual risk path c420. Synthetic teaching geometry—not a causal claim.*

![c421 teaching panel 01 (original).](../assets/figures/ml_fig_c421_01.png)
*Figure — Policy exception path c421. Synthetic teaching geometry—not a causal claim.*

![c422 teaching panel 01 (original).](../assets/figures/ml_fig_c422_01.png)
*Figure — Audit trail control path c422. Synthetic teaching geometry—not a causal claim.*

![c423 teaching panel 01 (original).](../assets/figures/ml_fig_c423_01.png)
*Figure — Fairness review path c423. Synthetic teaching geometry—not a causal claim.*

![c424 teaching panel 01 (original).](../assets/figures/ml_fig_c424_01.png)
*Figure — Privacy DPIA path c424. Synthetic teaching geometry—not a causal claim.*

![c425 teaching panel 01 (original).](../assets/figures/ml_fig_c425_01.png)
*Figure — Vendor risk path c425. Synthetic teaching geometry—not a causal claim.*

![c426 teaching panel 01 (original).](../assets/figures/ml_fig_c426_01.png)
*Figure — Kill-switch runbook path c426. Synthetic teaching geometry—not a causal claim.*

![c427 teaching panel 01 (original).](../assets/figures/ml_fig_c427_01.png)
*Figure — Rollback decision path c427. Synthetic teaching geometry—not a causal claim.*

![c428 teaching panel 01 (original).](../assets/figures/ml_fig_c428_01.png)
*Figure — Postmortem action path c428. Synthetic teaching geometry—not a causal claim.*

![c429 teaching panel 01 (original).](../assets/figures/ml_fig_c429_01.png)
*Figure — Data use purpose path c429. Synthetic teaching geometry—not a causal claim.*

![c430 teaching panel 01 (original).](../assets/figures/ml_fig_c430_01.png)
*Figure — Consent scope control path c430. Synthetic teaching geometry—not a causal claim.*

![c431 teaching panel 01 (original).](../assets/figures/ml_fig_c431_01.png)
*Figure — Model risk tier path c431. Synthetic teaching geometry—not a causal claim.*

![c432 teaching panel 01 (original).](../assets/figures/ml_fig_c432_01.png)
*Figure — Human oversight gate path c432. Synthetic teaching geometry—not a causal claim.*

![c433 teaching panel 01 (original).](../assets/figures/ml_fig_c433_01.png)
*Figure — Red-team severity path c433. Synthetic teaching geometry—not a causal claim.*

![c434 teaching panel 01 (original).](../assets/figures/ml_fig_c434_01.png)
*Figure — Incident SEV ladder path c434. Synthetic teaching geometry—not a causal claim.*

![c435 teaching panel 01 (original).](../assets/figures/ml_fig_c435_01.png)
*Figure — SBOM attest ship path c435. Synthetic teaching geometry—not a causal claim.*

![c436 teaching panel 01 (original).](../assets/figures/ml_fig_c436_01.png)
*Figure — Threat residual risk path c436. Synthetic teaching geometry—not a causal claim.*

![c437 teaching panel 01 (original).](../assets/figures/ml_fig_c437_01.png)
*Figure — Policy exception path c437. Synthetic teaching geometry—not a causal claim.*

![c438 teaching panel 01 (original).](../assets/figures/ml_fig_c438_01.png)
*Figure — Audit trail control path c438. Synthetic teaching geometry—not a causal claim.*

![c439 teaching panel 01 (original).](../assets/figures/ml_fig_c439_01.png)
*Figure — Fairness review path c439. Synthetic teaching geometry—not a causal claim.*

![c440 teaching panel 01 (original).](../assets/figures/ml_fig_c440_01.png)
*Figure — Privacy DPIA path c440. Synthetic teaching geometry—not a causal claim.*

![c441 teaching panel 01 (original).](../assets/figures/ml_fig_c441_01.png)
*Figure — Vendor risk path c441. Synthetic teaching geometry—not a causal claim.*

![c442 teaching panel 01 (original).](../assets/figures/ml_fig_c442_01.png)
*Figure — Kill-switch runbook path c442. Synthetic teaching geometry—not a causal claim.*

![c443 teaching panel 01 (original).](../assets/figures/ml_fig_c443_01.png)
*Figure — Rollback decision path c443. Synthetic teaching geometry—not a causal claim.*

![c444 teaching panel 01 (original).](../assets/figures/ml_fig_c444_01.png)
*Figure — Postmortem action path c444. Synthetic teaching geometry—not a causal claim.*

![c445 teaching panel 01 (original).](../assets/figures/ml_fig_c445_01.png)
*Figure — Data use purpose path c445. Synthetic teaching geometry—not a causal claim.*

![c446 teaching panel 01 (original).](../assets/figures/ml_fig_c446_01.png)
*Figure — Consent scope control path c446. Synthetic teaching geometry—not a causal claim.*

![c447 teaching panel 01 (original).](../assets/figures/ml_fig_c447_01.png)
*Figure — Model risk tier path c447. Synthetic teaching geometry—not a causal claim.*

![c448 teaching panel 01 (original).](../assets/figures/ml_fig_c448_01.png)
*Figure — Human oversight gate path c448. Synthetic teaching geometry—not a causal claim.*

![c449 teaching panel 01 (original).](../assets/figures/ml_fig_c449_01.png)
*Figure — Red-team severity path c449. Synthetic teaching geometry—not a causal claim.*

![c450 teaching panel 01 (original).](../assets/figures/ml_fig_c450_01.png)
*Figure — Incident SEV ladder path c450. Synthetic teaching geometry—not a causal claim.*

![c451 teaching panel 01 (original).](../assets/figures/ml_fig_c451_01.png)
*Figure — SBOM attest ship path c451. Synthetic teaching geometry—not a causal claim.*

![c452 teaching panel 01 (original).](../assets/figures/ml_fig_c452_01.png)
*Figure — Threat residual risk path c452. Synthetic teaching geometry—not a causal claim.*

![c453 teaching panel 01 (original).](../assets/figures/ml_fig_c453_01.png)
*Figure — Policy exception path c453. Synthetic teaching geometry—not a causal claim.*

![c454 teaching panel 01 (original).](../assets/figures/ml_fig_c454_01.png)
*Figure — Audit trail control path c454. Synthetic teaching geometry—not a causal claim.*

![c455 teaching panel 01 (original).](../assets/figures/ml_fig_c455_01.png)
*Figure — Fairness review path c455. Synthetic teaching geometry—not a causal claim.*

![c456 teaching panel 01 (original).](../assets/figures/ml_fig_c456_01.png)
*Figure — Privacy DPIA path c456. Synthetic teaching geometry—not a causal claim.*

![c457 teaching panel 01 (original).](../assets/figures/ml_fig_c457_01.png)
*Figure — Vendor risk path c457. Synthetic teaching geometry—not a causal claim.*

![c458 teaching panel 01 (original).](../assets/figures/ml_fig_c458_01.png)
*Figure — Kill-switch runbook path c458. Synthetic teaching geometry—not a causal claim.*

![c459 teaching panel 01 (original).](../assets/figures/ml_fig_c459_01.png)
*Figure — Rollback decision path c459. Synthetic teaching geometry—not a causal claim.*

![c460 teaching panel 01 (original).](../assets/figures/ml_fig_c460_01.png)
*Figure — Postmortem action path c460. Synthetic teaching geometry—not a causal claim.*

![c461 teaching panel 01 (original).](../assets/figures/ml_fig_c461_01.png)
*Figure — Data use purpose path c461. Synthetic teaching geometry—not a causal claim.*

![c462 teaching panel 01 (original).](../assets/figures/ml_fig_c462_01.png)
*Figure — Consent scope control path c462. Synthetic teaching geometry—not a causal claim.*

![c463 teaching panel 01 (original).](../assets/figures/ml_fig_c463_01.png)
*Figure — Model risk tier path c463. Synthetic teaching geometry—not a causal claim.*

![c464 teaching panel 01 (original).](../assets/figures/ml_fig_c464_01.png)
*Figure — Human oversight gate path c464. Synthetic teaching geometry—not a causal claim.*

![c465 teaching panel 01 (original).](../assets/figures/ml_fig_c465_01.png)
*Figure — Red-team severity path c465. Synthetic teaching geometry—not a causal claim.*

![c466 teaching panel 01 (original).](../assets/figures/ml_fig_c466_01.png)
*Figure — Incident SEV ladder path c466. Synthetic teaching geometry—not a causal claim.*

![c467 teaching panel 01 (original).](../assets/figures/ml_fig_c467_01.png)
*Figure — SBOM attest ship path c467. Synthetic teaching geometry—not a causal claim.*

![c468 teaching panel 01 (original).](../assets/figures/ml_fig_c468_01.png)
*Figure — Threat residual risk path c468. Synthetic teaching geometry—not a causal claim.*

![c469 teaching panel 01 (original).](../assets/figures/ml_fig_c469_01.png)
*Figure — Policy exception path c469. Synthetic teaching geometry—not a causal claim.*

![c470 teaching panel 01 (original).](../assets/figures/ml_fig_c470_01.png)
*Figure — Audit trail control path c470. Synthetic teaching geometry—not a causal claim.*

![c471 teaching panel 01 (original).](../assets/figures/ml_fig_c471_01.png)
*Figure — Fairness review path c471. Synthetic teaching geometry—not a causal claim.*

![c472 teaching panel 01 (original).](../assets/figures/ml_fig_c472_01.png)
*Figure — Privacy DPIA path c472. Synthetic teaching geometry—not a causal claim.*

![c473 teaching panel 01 (original).](../assets/figures/ml_fig_c473_01.png)
*Figure — Vendor risk path c473. Synthetic teaching geometry—not a causal claim.*

![c474 teaching panel 01 (original).](../assets/figures/ml_fig_c474_01.png)
*Figure — Kill-switch runbook path c474. Synthetic teaching geometry—not a causal claim.*

![c475 teaching panel 01 (original).](../assets/figures/ml_fig_c475_01.png)
*Figure — Rollback decision path c475. Synthetic teaching geometry—not a causal claim.*

![c476 teaching panel 01 (original).](../assets/figures/ml_fig_c476_01.png)
*Figure — Postmortem action path c476. Synthetic teaching geometry—not a causal claim.*

![c477 teaching panel 01 (original).](../assets/figures/ml_fig_c477_01.png)
*Figure — Data use purpose path c477. Synthetic teaching geometry—not a causal claim.*

![c478 teaching panel 01 (original).](../assets/figures/ml_fig_c478_01.png)
*Figure — Consent scope control path c478. Synthetic teaching geometry—not a causal claim.*

![c479 teaching panel 01 (original).](../assets/figures/ml_fig_c479_01.png)
*Figure — Model risk tier path c479. Synthetic teaching geometry—not a causal claim.*

![c480 teaching panel 01 (original).](../assets/figures/ml_fig_c480_01.png)
*Figure — Human oversight gate path c480. Synthetic teaching geometry—not a causal claim.*

![c481 teaching panel 01 (original).](../assets/figures/ml_fig_c481_01.png)
*Figure — Red-team severity path c481. Synthetic teaching geometry—not a causal claim.*

![c482 teaching panel 01 (original).](../assets/figures/ml_fig_c482_01.png)
*Figure — Incident SEV ladder path c482. Synthetic teaching geometry—not a causal claim.*

![c483 teaching panel 01 (original).](../assets/figures/ml_fig_c483_01.png)
*Figure — SBOM attest ship path c483. Synthetic teaching geometry—not a causal claim.*

![c484 teaching panel 01 (original).](../assets/figures/ml_fig_c484_01.png)
*Figure — Threat residual risk path c484. Synthetic teaching geometry—not a causal claim.*

![c485 teaching panel 01 (original).](../assets/figures/ml_fig_c485_01.png)
*Figure — Policy exception path c485. Synthetic teaching geometry—not a causal claim.*

![c486 teaching panel 01 (original).](../assets/figures/ml_fig_c486_01.png)
*Figure — Audit trail control path c486. Synthetic teaching geometry—not a causal claim.*

![c487 teaching panel 01 (original).](../assets/figures/ml_fig_c487_01.png)
*Figure — Fairness review path c487. Synthetic teaching geometry—not a causal claim.*

![c488 teaching panel 01 (original).](../assets/figures/ml_fig_c488_01.png)
*Figure — Privacy DPIA path c488. Synthetic teaching geometry—not a causal claim.*

![c489 teaching panel 01 (original).](../assets/figures/ml_fig_c489_01.png)
*Figure — Vendor risk path c489. Synthetic teaching geometry—not a causal claim.*

![c490 teaching panel 01 (original).](../assets/figures/ml_fig_c490_01.png)
*Figure — Kill-switch runbook path c490. Synthetic teaching geometry—not a causal claim.*

![c491 teaching panel 01 (original).](../assets/figures/ml_fig_c491_01.png)
*Figure — Rollback decision path c491. Synthetic teaching geometry—not a causal claim.*

![c492 teaching panel 01 (original).](../assets/figures/ml_fig_c492_01.png)
*Figure — Postmortem action path c492. Synthetic teaching geometry—not a causal claim.*

![c493 teaching panel 01 (original).](../assets/figures/ml_fig_c493_01.png)
*Figure — Data use purpose path c493. Synthetic teaching geometry—not a causal claim.*

![c494 teaching panel 01 (original).](../assets/figures/ml_fig_c494_01.png)
*Figure — Consent scope control path c494. Synthetic teaching geometry—not a causal claim.*

![c495 teaching panel 01 (original).](../assets/figures/ml_fig_c495_01.png)
*Figure — Model risk tier path c495. Synthetic teaching geometry—not a causal claim.*

![c496 teaching panel 01 (original).](../assets/figures/ml_fig_c496_01.png)
*Figure — Human oversight gate path c496. Synthetic teaching geometry—not a causal claim.*

![c497 teaching panel 01 (original).](../assets/figures/ml_fig_c497_01.png)
*Figure — Red-team severity path c497. Synthetic teaching geometry—not a causal claim.*

![c498 teaching panel 01 (original).](../assets/figures/ml_fig_c498_01.png)
*Figure — Incident SEV ladder path c498. Synthetic teaching geometry—not a causal claim.*

![c499 teaching panel 01 (original).](../assets/figures/ml_fig_c499_01.png)
*Figure — SBOM attest ship path c499. Synthetic teaching geometry—not a causal claim.*

![c500 teaching panel 01 (original).](../assets/figures/ml_fig_c500_01.png)
*Figure — Threat residual risk path c500. Synthetic teaching geometry—not a causal claim.*

![c501 teaching panel 01 (original).](../assets/figures/ml_fig_c501_01.png)
*Figure — Policy exception path c501. Synthetic teaching geometry—not a causal claim.*

![c502 teaching panel 01 (original).](../assets/figures/ml_fig_c502_01.png)
*Figure — Audit trail control path c502. Synthetic teaching geometry—not a causal claim.*

![c503 teaching panel 01 (original).](../assets/figures/ml_fig_c503_01.png)
*Figure — Fairness review path c503. Synthetic teaching geometry—not a causal claim.*

![c504 teaching panel 01 (original).](../assets/figures/ml_fig_c504_01.png)
*Figure — Privacy DPIA path c504. Synthetic teaching geometry—not a causal claim.*

![c505 teaching panel 01 (original).](../assets/figures/ml_fig_c505_01.png)
*Figure — Vendor risk path c505. Synthetic teaching geometry—not a causal claim.*

![c506 teaching panel 01 (original).](../assets/figures/ml_fig_c506_01.png)
*Figure — Kill-switch runbook path c506. Synthetic teaching geometry—not a causal claim.*

![c507 teaching panel 01 (original).](../assets/figures/ml_fig_c507_01.png)
*Figure — Rollback decision path c507. Synthetic teaching geometry—not a causal claim.*

![c508 teaching panel 01 (original).](../assets/figures/ml_fig_c508_01.png)
*Figure — Postmortem action path c508. Synthetic teaching geometry—not a causal claim.*

![c509 teaching panel 01 (original).](../assets/figures/ml_fig_c509_01.png)
*Figure — Data use purpose path c509. Synthetic teaching geometry—not a causal claim.*

![c510 teaching panel 01 (original).](../assets/figures/ml_fig_c510_01.png)
*Figure — Consent scope control path c510. Synthetic teaching geometry—not a causal claim.*

![c511 teaching panel 01 (original).](../assets/figures/ml_fig_c511_01.png)
*Figure — Model risk tier path c511. Synthetic teaching geometry—not a causal claim.*

![c512 teaching panel 01 (original).](../assets/figures/ml_fig_c512_01.png)
*Figure — Human oversight gate path c512. Synthetic teaching geometry—not a causal claim.*

![c513 teaching panel 01 (original).](../assets/figures/ml_fig_c513_01.png)
*Figure — Red-team severity path c513. Synthetic teaching geometry—not a causal claim.*

![c514 teaching panel 01 (original).](../assets/figures/ml_fig_c514_01.png)
*Figure — Incident SEV ladder path c514. Synthetic teaching geometry—not a causal claim.*

![c515 teaching panel 01 (original).](../assets/figures/ml_fig_c515_01.png)
*Figure — SBOM attest ship path c515. Synthetic teaching geometry—not a causal claim.*

![c516 teaching panel 01 (original).](../assets/figures/ml_fig_c516_01.png)
*Figure — Threat residual risk path c516. Synthetic teaching geometry—not a causal claim.*

![c517 teaching panel 01 (original).](../assets/figures/ml_fig_c517_01.png)
*Figure — Policy exception path c517. Synthetic teaching geometry—not a causal claim.*

![c518 teaching panel 01 (original).](../assets/figures/ml_fig_c518_01.png)
*Figure — Audit trail control path c518. Synthetic teaching geometry—not a causal claim.*

![c519 teaching panel 01 (original).](../assets/figures/ml_fig_c519_01.png)
*Figure — Fairness review path c519. Synthetic teaching geometry—not a causal claim.*

![c520 teaching panel 01 (original).](../assets/figures/ml_fig_c520_01.png)
*Figure — Privacy DPIA path c520. Synthetic teaching geometry—not a causal claim.*

![c521 teaching panel 01 (original).](../assets/figures/ml_fig_c521_01.png)
*Figure — Vendor risk path c521. Synthetic teaching geometry—not a causal claim.*

![c522 teaching panel 01 (original).](../assets/figures/ml_fig_c522_01.png)
*Figure — Kill-switch runbook path c522. Synthetic teaching geometry—not a causal claim.*

![c523 teaching panel 01 (original).](../assets/figures/ml_fig_c523_01.png)
*Figure — Rollback decision path c523. Synthetic teaching geometry—not a causal claim.*

![c524 teaching panel 01 (original).](../assets/figures/ml_fig_c524_01.png)
*Figure — Postmortem action path c524. Synthetic teaching geometry—not a causal claim.*

![c525 teaching panel 01 (original).](../assets/figures/ml_fig_c525_01.png)
*Figure — Data use purpose path c525. Synthetic teaching geometry—not a causal claim.*

![c526 teaching panel 01 (original).](../assets/figures/ml_fig_c526_01.png)
*Figure — Consent scope control path c526. Synthetic teaching geometry—not a causal claim.*

![c527 teaching panel 01 (original).](../assets/figures/ml_fig_c527_01.png)
*Figure — Model risk tier path c527. Synthetic teaching geometry—not a causal claim.*

![c528 teaching panel 01 (original).](../assets/figures/ml_fig_c528_01.png)
*Figure — Human oversight gate path c528. Synthetic teaching geometry—not a causal claim.*

![c529 teaching panel 01 (original).](../assets/figures/ml_fig_c529_01.png)
*Figure — Red-team severity path c529. Synthetic teaching geometry—not a causal claim.*

![c530 teaching panel 01 (original).](../assets/figures/ml_fig_c530_01.png)
*Figure — Incident SEV ladder path c530. Synthetic teaching geometry—not a causal claim.*

![c531 teaching panel 01 (original).](../assets/figures/ml_fig_c531_01.png)
*Figure — SBOM attest ship path c531. Synthetic teaching geometry—not a causal claim.*

![c532 teaching panel 01 (original).](../assets/figures/ml_fig_c532_01.png)
*Figure — Threat residual risk path c532. Synthetic teaching geometry—not a causal claim.*
