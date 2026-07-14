# Preface

## Opening

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

There is no separate further-reading chapter—the book is the curriculum. Read it iteratively, refer to the mathematical foundations when needed, and apply these concepts rigorously to the next paper you read or the next clinical tool your hospital considers.
