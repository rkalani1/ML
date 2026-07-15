# Closing Synthesis: Senior Practice in Clinical Neurology and Epidemiology


![17 Roc Curve](../assets/figures/17_roc_curve.png)


## Opening
![One-paragraph senior synthesis habit (original).](../assets/figures/ml_fig_one_paragraph.png)

*One-paragraph senior synthesis habit (original).*

![Three poles of model evaluation (original).](../assets/figures/ml_fig_eval_triangle.png)

*Three poles of model evaluation (original).*


You finish a model paper and a methods appendix in one sitting. Synthesis means you can state the decision, the data generation process, the evaluation design, and the residual risks in one paragraph a charge nurse would respect.


![Teaching scorecard for model appraisal (original).](../assets/figures/ml_fig_appraisal_scorecard.png)

*Teaching scorecard for model appraisal (original).*

![Discrimination vs calibration vs utility (original).](../assets/figures/ml_fig_metric_map.png)

*Closing reminder: ranking, reliability, and decision value are distinct claims (original).*

![Senior practice lifecycle — design to drift (original).](../assets/figures/ml_fig_lifecycle_deploy.png)

*Lifecycle habit: design index time → fit → external validate → calibrate/utility → deploy with a model card → monitor drift (original).*

![Decision curve: net benefit vs threshold (synthetic; original).](../assets/figures/ml_fig_decision_curve.png)

*Net benefit is threshold-specific. A model can dominate treat-all/none on a range and still fail outside it (original scientific sketch).*

![Decision-curve net benefit and ΔNB vs defaults across thresholds (synthetic; original).](../assets/figures/ml_fig_decision_curve_nb.png)

*Figure — Closing utility panel. **Left:** model net benefit vs treat-all and treat-none across threshold probabilities. **Right:** ΔNB versus the better default highlights the clinical range where the model helps. Net benefit is not AUROC and not a causal treatment effect—choose pt with domain experts.*

![Model card minimum gates checklist (teaching; original).](../assets/figures/ml_fig_model_card_gates.png)

*Figure — Closing governance panel. Intended use, training data, metrics+slices, calibration, limits, and monitor plan are non-optional gates for shipping a prediction service. Cards document predictors—not causal claims.*

![Full senior ML appraisal checklist flowchart (original teaching graphic).](../assets/figures/ml_fig_appraisal_checklist.png)

*Figure — Eleven-gate appraisal flow for clinical prediction systems: question fit → index/legality → leakage → cohort/label → fit/capacity → discrimination → calibration → utility → external test → deploy card → drift monitor. Skip a gate only with a written reason. Prediction success never licenses a causal claim or sole-trigger withdrawal of care.*

![Prediction ≠ causation: confounder sketch and claim boxes (synthetic; original).](../assets/figures/ml_fig_pred_not_cause.png)

*Figure — Closing discipline for claim type. **Left:** synthetic data where confounder *U* drives both *X* and *Y*; a strong observational fit of *Y* on *X* coexists with no *X*→*Y* edge in the DGP. **Right:** prediction systems may use *X* lawfully when calibrated and useful; causal claims need design or identification assumptions. Senior practice states which claim is being made before acting on a score.*

The book has moved from describing data to deploying systems, but the through-argument was never really about algorithms. For a neurologist–epidemiologist, the machinery of gradient descent or attention is the tractable part; the demanding part is the reasoning that surrounds a model — what question it answers, when it is permitted to see each variable, and whether its numbers mean what a clinician assumes they mean. Senior competence is the habit of interrogating those surroundings before trusting any output. This chapter gathers the disciplines that outlived the chapters that introduced them, then walks a single clinical prediction study from raw data to post-deployment monitoring so the abstractions have somewhere concrete to land.

## Disciplines that recur across every chapter

**Index time and feature legality.** Every supervised problem carries a moment at which the prediction is made. A variable is legal only if it is knowable, for that patient, at or before that moment. The recurring failure is the retrospectively assembled table in which columns drawn from different points in a hospitalization sit side by side with no timestamp, so that a model quietly learns from the future. Fixing index time first — before touching a model — is the single most protective act in applied clinical ML.

**Leakage is the insidious cousin.** Beyond after-the-fact features, leakage hides in proxies of the outcome (a treatment ordered because the diagnosis was already obvious) and in contaminated splits (slices from one patient, or one scanner, appearing in both training and test). The defenses are grouped splits by patient and often by site, and suspicion whenever performance looks too good.

**Discrimination is not calibration, and neither is utility.** AUC measures ranking: the chance a random case outranks a random non-case. It says nothing about whether a predicted 0.20 corresponds to a 20% event rate, yet decisions are made at thresholds where calibration governs behavior. A modestly discriminating but well-calibrated model can outperform a sharper, miscalibrated one for a specific decision, which is why decision-analytic net benefit belongs beside the ROC curve.

![Bootstrap uncertainty for AUROC: percentile CI and width vs n (synthetic; original).](../assets/figures/ml_fig_bootstrap_auroc.png)

*Figure — Point AUROC is not enough. **Left:** bootstrap distribution of AUROC with percentile 95% CI around a single point estimate. **Right:** mean CI width shrinks with n—small cohorts license wide uncertainty. Bootstrap CIs still do not prove transportability; external validation remains the real examination. High AUROC is not causation.*

![Split conformal prediction: finite-sample band and empirical coverage vs 1−α (synthetic; original).](../assets/figures/ml_fig_conformal.png)

*Figure — Distribution-free intervals under exchangeability. **Left:** split-conformal band from calibration absolute residuals around a point predictor. **Right:** empirical coverage tracks target 1−α; tighter α yields wider intervals (annotated). Coverage guarantees break under dataset shift—re-estimate on local exchangeable cal data. Prediction intervals are not causal effects of intervening on x.*

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

![Multi-metric radar and gate scores for one synthetic LVO-style case study: high-AUC model vs balanced model (scientific; original).](../assets/figures/ml_fig_multimetric_radar.png)

*Figure — One study, many claims. **Left:** radar of goodness scores (AUROC, AUPRC, calibration slope, inverted Brier/ECE, net benefit, worst-subgroup AUROC). **Right:** Model A wins ranking metrics; Model B clears more calibration/utility gates against a teaching floor. Do not ship on AUROC alone. High predictive scores never license a causal claim or sole-trigger withdrawal of care.*

### 5. Evaluation Strategy
Split grouped by patient, then split temporally (train on earlier years, test on later), then validate externally at partner centers with different CT scanners, different ASPECTS-reading habits, and a different case mix of late-window and higher-baseline-disability patients. Expect calibration to drift on transport; recalibrate the intercept and slope rather than assume the model travels intact.

### 6. Deployment and Monitoring
Ship with a model card stating population, index time, features, calibration, intended use, and — explicitly — prohibited uses, chief among them any use of the score as a sole trigger for withdrawal of care. Freeze the preprocessing pipeline to prevent train–serve skew. Monitor input distributions (age, NIHSS, ASPECTS), the prediction distribution, and, as outcomes mature, calibration over rolling windows. Predefine the triggers for recalibration and retraining — a new thrombectomy device, an expanded time window, a coding change, a replaced scanner — and keep a rollback path ready.

![Silent-trial ops: PSI/ECE alarms and AUROC rollback floor (synthetic; original).](../assets/figures/ml_fig_rollback_triggers.png)\n![Shadow-mode gate funnel before actioning predictions (original teaching).](../assets/figures/ml_fig_shadow_mode.png)

*Figure — Senior practice funnel. Shadow mode logs scores without changing care; only systems that survive comparison and limited release advance. **Prediction ≠ causation**; decision impact requires prospective design and equity audits.*\n\n


![Slice evaluation matrix across subgroups and metrics (synthetic; original).](../assets/figures/ml_fig_slice_eval_matrix.png)

*Figure — Senior practice: never ship on overall AUROC alone. Site B and other slices can fail while averages look fine. Slice gates protect patients—**prediction still ≠ causation** without design.*


![Equity slice AUROC gaps vs a teaching equity floor (synthetic; original).](../assets/figures/ml_fig_equity_gaps.png)

*Figure — Overall metrics hide failed slices (language, insurance, rural site). Equity floors are deployment gates. Closing gaps needs design and data—not metric theater. Pred ≠ cause.*


![Incident response timeline from detect to retrain decision (original).](../assets/figures/ml_fig_incident_timeline.png)

*Figure — Senior ops: detect → page → mitigate → root cause → fix → optional retrain. Speed without learning repeats harm. Process fixes complement model fixes. Pred ≠ cause.*


![Pre-registration boxes: outcomes, splits, slices, stop rules, metrics (original).](../assets/figures/ml_fig_prereg_boxes.png)

*Figure — Senior practice: write the plan before peeking. Pre-reg reduces analytic flexibility theater. Process discipline supports trustworthy prediction claims—not causation by default.*


![Champion vs challenger metric snapshot (synthetic; original).](../assets/figures/ml_fig_champ_challenger.png)

*Figure — Promote only when multi-metric gates pass. Pred != cause without design.*


![Go/no-go deploy decision matrix (original).](../assets/figures/ml_fig_go_no_go.png)

*Figure — Multi-gate go-live decisions. Go/no-go deploy decision matrix Pred != cause without design.*


![raci teaching panel (original).](../assets/figures/ml_fig_raci_ops.png)

*Figure — Teaching panel for raci. Pred != cause without design.*


![Cycle-34 densify scientific panel 19 (original).](../assets/figures/ml_fig_c34_18.png)

*Figure — Continuous densify panel 19. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 19 (original).](../assets/figures/ml_fig_c35_18.png)

*Figure — Continuous densify panel 19. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 19 (original).](../assets/figures/ml_fig_c36_18.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 19 (original).](../assets/figures/ml_fig_c37_18.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 19 (original).](../assets/figures/ml_fig_c38_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 19 (original).](../assets/figures/ml_fig_c39_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 19 (original).](../assets/figures/ml_fig_c40_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 19 (original).](../assets/figures/ml_fig_c41_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 19 (original).](../assets/figures/ml_fig_c42_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 19 (original).](../assets/figures/ml_fig_c43_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 19 (original).](../assets/figures/ml_fig_c44_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 19 (original).](../assets/figures/ml_fig_c45_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 19 (original).](../assets/figures/ml_fig_c46_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 19 (original).](../assets/figures/ml_fig_c47_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 19 (original).](../assets/figures/ml_fig_c48_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 19 (original).](../assets/figures/ml_fig_c49_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 19 (original).](../assets/figures/ml_fig_c50_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 19 (original).](../assets/figures/ml_fig_c51_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 19 (original).](../assets/figures/ml_fig_c52_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 19 (original).](../assets/figures/ml_fig_c53_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 19 (original).](../assets/figures/ml_fig_c54_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 19 (original).](../assets/figures/ml_fig_c55_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 19 (original).](../assets/figures/ml_fig_c56_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 19 (original).](../assets/figures/ml_fig_c57_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 19 (original).](../assets/figures/ml_fig_c58_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 19 (original).](../assets/figures/ml_fig_c59_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 19 (original).](../assets/figures/ml_fig_c60_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 19 (original).](../assets/figures/ml_fig_c61_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 19 (original).](../assets/figures/ml_fig_c62_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 19 (original).](../assets/figures/ml_fig_c63_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 19 (original).](../assets/figures/ml_fig_c64_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 19 (original).](../assets/figures/ml_fig_c65_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 19 (original).](../assets/figures/ml_fig_c66_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 19 (original).](../assets/figures/ml_fig_c67_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 19 (original).](../assets/figures/ml_fig_c68_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 19 (original).](../assets/figures/ml_fig_c69_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 19 (original).](../assets/figures/ml_fig_c70_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 19 (original).](../assets/figures/ml_fig_c71_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 19 (original).](../assets/figures/ml_fig_c72_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 19 (original).](../assets/figures/ml_fig_c73_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 19 (original).](../assets/figures/ml_fig_c74_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 19 (original).](../assets/figures/ml_fig_c75_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 19 (original).](../assets/figures/ml_fig_c76_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 19 (original).](../assets/figures/ml_fig_c77_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 19 (original).](../assets/figures/ml_fig_c78_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 19 (original).](../assets/figures/ml_fig_c79_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 19 (original).](../assets/figures/ml_fig_c80_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 19 (original).](../assets/figures/ml_fig_c81_18.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

*Figure — Write triggers before go-live. **Left:** score PSI and ECE with pre-set alarm lines; a scanner swap at week 18 pushes PSI past 0.2. **Right:** live AUROC with a rollback floor at 0.80—recalibrate when calibration drifts, rollback when discrimination breaches the floor. Act on input/score monitors before lagged outcomes confirm failure. Prediction ≠ causation; ops monitoring is part of the science.*

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


## Pre-deployment checklist (teaching table)

| Gate | Pass looks like |
|------|-----------------|
| Question fit | Prediction vs action stated |
| Leakage audit | Index time fixed; no post-outcome features |
| Validation | External site or temporal holdout |
| Calibration | Plot + slope near 1 in target population |
| Governance | Version, threshold, monitoring owner |

## Full appraisal gate map (aligned to the flowchart)

| # | Gate | Pass criterion | Fail / stop rule |
|---|------|----------------|------------------|
| 1 | Question fit | Claim typed as prediction, etiology, or decision support | Marketing language blurs prediction with “what to do” |
| 2 | Index & legality | Every feature timestamped ≤ index time | Any post-outcome or post-decision column |
| 3 | Leakage audit | Patient- (and often site-) grouped splits; no outcome proxies | Random row split on longitudinal rows |
| 4 | Cohort & label | Inclusion, label source, missingness plan written | Convenience sample with silent label drift |
| 5 | Fit & capacity | Model class matched to n; regularization + early stop | Deep net on n ≪ parameters without strong pretrain |
| 6 | Discrimination | AUC and/or PR reported at deployment prevalence | Accuracy alone on imbalanced labels |
| 7 | Calibration | Reliability plot; slope/intercept in target mix | “AUC is high” used as probability honesty |
| 8 | Utility | Net benefit / decision curve over clinical thresholds | Threshold chosen only to maximize Youden on test |
| 9 | External test | Geographic or temporal holdout; expect recalibration | Internal CV sold as transportability |
| 10 | Deploy card | Version, threshold, intended and prohibited uses | Score used as sole withdrawal trigger |
| 11 | Monitor drift | Owner, metrics, rollback triggers | Ship-and-forget after a conference abstract |

![c82 teaching panel 18 (original).](../assets/figures/ml_fig_c82_18.png)
*Figure — Senior appraisal diamond: question, leakage, validation, utility. Synthetic teaching geometry—not a causal claim.*

![c83 teaching panel 18 (original).](../assets/figures/ml_fig_c83_18.png)
*Figure — Transportability: AUROC may hold while calibration error rises. Synthetic teaching geometry—not a causal claim.*

![c84 teaching panel 18 (original).](../assets/figures/ml_fig_c84_18.png)
*Figure — Decision curve net benefit vs threshold probability. Synthetic teaching geometry—not a causal claim.*

![c85 teaching panel 18 (original).](../assets/figures/ml_fig_c85_18.png)
*Figure — Senior go/no-go checklist tiles before sign-off. Synthetic teaching geometry—not a causal claim.*

![c86 teaching panel 18 (original).](../assets/figures/ml_fig_c86_18.png)
*Figure — Validation strength ladder from internal to prospective. Synthetic teaching geometry—not a causal claim.*

![c87 teaching panel 18 (original).](../assets/figures/ml_fig_c87_18.png)
*Figure — Deployment decision tree: monitor, revise, shadow. Synthetic teaching geometry—not a causal claim.*
