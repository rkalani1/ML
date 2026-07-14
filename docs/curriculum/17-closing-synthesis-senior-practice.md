# Closing Synthesis: Senior Practice in Clinical Neurology and Epidemiology


![17 Roc Curve](../assets/figures/17_roc_curve.png)


## Opening

You finish a model paper and a methods appendix in one sitting. Synthesis means you can state the decision, the data generation process, the evaluation design, and the residual risks in one paragraph a charge nurse would respect.

The book has moved from describing data to deploying systems, but the through-argument was never really about algorithms. For a neurologist–epidemiologist, the machinery of gradient descent or attention is the tractable part; the demanding part is the reasoning that surrounds a model — what question it answers, when it is permitted to see each variable, and whether its numbers mean what a clinician assumes they mean. Senior competence is the habit of interrogating those surroundings before trusting any output. This chapter gathers the disciplines that outlived the chapters that introduced them, then walks a single clinical prediction study from raw data to post-deployment monitoring so the abstractions have somewhere concrete to land.

## Disciplines that recur across every chapter

**Index time and feature legality.** Every supervised problem carries a moment at which the prediction is made. A variable is legal only if it is knowable, for that patient, at or before that moment. The recurring failure is the retrospectively assembled table in which columns drawn from different points in a hospitalization sit side by side with no timestamp, so that a model quietly learns from the future. Fixing index time first — before touching a model — is the single most protective act in applied clinical ML.

**Leakage is the insidious cousin.** Beyond after-the-fact features, leakage hides in proxies of the outcome (a treatment ordered because the diagnosis was already obvious) and in contaminated splits (slices from one patient, or one scanner, appearing in both training and test). The defenses are grouped splits by patient and often by site, and suspicion whenever performance looks too good.

**Discrimination is not calibration, and neither is utility.** AUC measures ranking: the chance a random case outranks a random non-case. It says nothing about whether a predicted 0.20 corresponds to a 20% event rate, yet decisions are made at thresholds where calibration governs behavior. A modestly discriminating but well-calibrated model can outperform a sharper, miscalibrated one for a specific decision, which is why decision-analytic net benefit belongs beside the ROC curve.

**Prevalence, imbalance, and predictive value.** Identical sensitivity and specificity yield wildly different positive predictive value as the base rate moves. A model tuned on a case-enriched development sample will disappoint in a low-prevalence clinic unless its probabilities are mapped back to the deployment prevalence. Resampling to balance classes is a modeling convenience that silently rewrites the base rate; the calibration must be restored afterward.

**External validation and dataset shift.** Internal cross-validation estimates reproducibility, not transportability. Scanners, coding systems (an ICD transition, a local phenotype definition), and case mix all differ between sites, so a model can be internally excellent and externally useless. Geographic and temporal external validation, not a tighter internal fold, is the real examination.

**Label noise.** In clinical work the label is usually a human artifact — a chart-review phenotype, a billing code, an mRS obtained by telephone. Noisy or biased labels cap achievable performance and can quietly encode the practice patterns one hoped to improve.

**Prediction is not causation.** A model that predicts an outcome well says nothing about what happens if one intervenes. A DAG makes the distinction explicit, separating confounders from mediators and colliders, and warns against reading a predictor’s coefficient as a treatment effect or adjusting for a collider and manufacturing bias.

**Reproducibility and governance.** A model is not a result; it is an artifact with a version, a training set, a preprocessing pipeline, and a decision threshold. Unless each is recorded, the model can be neither trusted nor safely retired.

## One study, from data to drift: A Detailed Case Study

Consider building a model to estimate the probability of poor functional outcome — 90-day modified Rankin Scale 3–6 — for anterior-circulation large-vessel-occlusion patients treated with endovascular thrombectomy, intended to inform family counseling, never to dictate it. The estimand is prognostic, not causal.

### 1. Population and Index Time
Patients who completed thrombectomy at a comprehensive stroke center; index time is set at 24 hours after the procedure, so the model may use the early trajectory. Everything after hour 24 — day-5 exams, discharge disposition, rehabilitation notes — is illegal by construction.

### 2. Features
Legal inputs include age, pre-stroke mRS, admission NIHSS, baseline ASPECTS, occlusion site, onset-to-recanalization time, final reperfusion grade, the 24-hour NIHSS, and 24-hour imaging for hemorrhage. The dangerous temptations are specific: withdrawal-of-life-sustaining-therapy status is downstream of a poor-prognosis expectation and a determinant of death, so admitting it as a feature — or letting deaths after withdrawal dominate the label — closes a self-fulfilling loop. Reperfusion grade read by the treating operator may embed optimism, a quiet source of measurement noise.

### 3. Label
The 90-day mRS comes from a structured telephone interview with real inter-rater variability, and its missingness is not random: the sickest and the dead are hardest to reach. Blinded adjudication and sensitivity analyses under informative missingness are not decorations here; they bound what the model can honestly claim.

### 4. Model Selection and Tuning
Begin with penalized logistic regression (elastic net): interpretable coefficients, honest calibration, and few parameters for a few thousand patients, with gradient-boosted trees as a challenger. Resist deep networks at this sample size. Report discrimination and calibration together — a calibration plot with slope and intercept — plus net benefit across the threshold range clinicians would actually entertain. The math tells us: $$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta X)}} $$ where the regularization path for $\lambda$ is chosen via rigorous internal cross-validation.

### 5. Evaluation Strategy
Split grouped by patient, then split temporally (train on earlier years, test on later), then validate externally at partner centers with different CT scanners, different ASPECTS-reading habits, and a different case mix of late-window and higher-baseline-disability patients. Expect calibration to drift on transport; recalibrate the intercept and slope rather than assume the model travels intact.

### 6. Deployment and Monitoring
Ship with a model card stating population, index time, features, calibration, intended use, and — explicitly — prohibited uses, chief among them any use of the score as a sole trigger for withdrawal of care. Freeze the preprocessing pipeline to prevent train–serve skew. Monitor input distributions (age, NIHSS, ASPECTS), the prediction distribution, and, as outcomes mature, calibration over rolling windows. Predefine the triggers for recalibration and retraining — a new thrombectomy device, an expanded time window, a coding change, a replaced scanner — and keep a rollback path ready.

## Synthetic Teaching Table: Deployment Checklist

| Phase | Key Question | Method of Verification | Potential Failure Mode |
|---|---|---|---|
| **Design** | Is the index time strictly defined? | Map every feature timestamp against the index time. | Looking into the future (leakage). |
| **Data** | Are labels uniformly adjudicated? | Blinded chart review or standard interview tools. | Label noise masking true signal. |
| **Model** | Is it well-calibrated? | Calibration plot (intercept & slope). | High discrimination but clinically misleading risk estimates. |
| **Validate** | Does it transport? | External geographical & temporal testing. | Overfitting to local site characteristics or specific scanners. |
| **Deploy** | Who is watching it? | Automated drift monitoring and scheduled audits. | Silent failure as clinical practice or coding shifts. |

## What the discipline amounts to

The recurring lesson is not that models are dangerous or magical, but that a prediction is a claim conditioned on a time, a population, and a label, and that most clinical harm follows from forgetting one of those conditions. The competent researcher treats every number as provisional, states plainly what would falsify it, and keeps prediction and intervention in separate mental columns. If the book leaves a single reflex, let it be the pause before belief: what was known, and when; to whom this actually applies; and what, exactly, is being counted.
