# Preface

This textbook bridges machine learning theory and clinical appraisal for neurologists and epidemiologists. It covers core algorithms—foundations, unsupervised learning, feature engineering, supervised methods, deep learning, multimodal architectures, reinforcement learning, model compression, graph mining, and data challenges—emphasizing rigorous definitions, leakage prevention, calibration, and decision impact over vendor claims.

## Why Neurologists Need to Know Machine Learning

The intersection of neuroscience, clinical neurology, and computation is rapidly evolving. Historically, we relied on classical biostatistics to evaluate treatments and understand disease etiology. Today, machine learning models analyze continuous EEG streams, high-resolution MRIs, and vast troves of unstructured clinical text to predict outcomes, suggest diagnoses, and optimize resource allocation.

Understanding these models is no longer a niche skill—it is a core clinical competency. When a black-box model suggests that a patient with acute ischemic stroke is not a candidate for thrombectomy, you must know what features the model considers, what biases it might harbor, and what its uncertainty estimates actually mean. You cannot delegate this judgment.

## How to Read This Book

Every major algorithm family in the published TOC is taught with definitions and intuition.

1. **Work numerical examples by hand:** Recompute intermediates. Nothing demystifies an algorithm faster than doing the matrix multiplication or gradient update yourself.
2. **Map each method:** Continually map the computational method back to the clinical problem space. Is this for prediction, etiology, or decision support?
3. **Be skeptical of performance metrics:** Always look past discrimination (AUROC) to calibration, utility, and external validation.

There is no separate further-reading chapter—the book is the curriculum. Read it iteratively, refer to the mathematical foundations when needed, and apply these concepts rigorously to the next paper you read or the next clinical tool your hospital considers.

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
