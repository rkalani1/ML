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

## Why Neurologists Need to Know Machine Learning

The intersection of neuroscience, clinical neurology, and computation is rapidly evolving. Historically, we relied on classical biostatistics to evaluate treatments and understand disease etiology. Today, machine learning models analyze continuous EEG streams, high-resolution MRIs, and vast troves of unstructured clinical text to predict outcomes, suggest diagnoses, and optimize resource allocation.

Understanding these models is no longer a niche skill—it is a core clinical competency. When a black-box model suggests that a patient with acute ischemic stroke is not a candidate for thrombectomy, you must know what features the model considers, what biases it might harbor, and what its uncertainty estimates actually mean. You cannot delegate this judgment.

## How to Read This Book

Every major algorithm family in the published TOC is taught with definitions and intuition.

1. **Work numerical examples by hand:** Recompute intermediates. Nothing demystifies an algorithm faster than doing the matrix multiplication or gradient update yourself.
2. **Map each method:** Continually map the computational method back to the clinical problem space. Is this for prediction, etiology, or decision support?
3. **Be skeptical of performance metrics:** Always look past discrimination (AUROC) to calibration, utility, and external validation.

![Method family → allowed clinical claim (routing map; original).](../assets/figures/ml_fig_claim_routing.png)

*Figure — Preface routing map. Supervised scores license **prediction** claims when calibrated and externally checked; they do not license etiology (blocked dashed path). Causal designs support cause claims. Utility / net-benefit work supports decision-support claims. Clusters and embeddings are hypothesis-generating only. Prediction ≠ causation.*

There is no separate further-reading chapter—the book is the curriculum. Read it iteratively, refer to the mathematical foundations when needed, and apply these concepts rigorously to the next paper you read or the next clinical tool your hospital considers.

![External validation ladder: optimism shrinks as the test hardens (synthetic; original).](../assets/figures/ml_fig_external_ladder.png)

*Figure — Preface external-validation ladder (synthetic teaching). **Left:** resubstitution AUROC looks heroic; patient-wise CV, temporal split, external site, and prospective silent trial successively erode discrimination. **Right:** calibration error (ECE) often worsens on transport even when ranking still looks decent. Local AUROC is not a shipping license; prediction ≠ causation.*

## Reader’s map (teaching table)

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
