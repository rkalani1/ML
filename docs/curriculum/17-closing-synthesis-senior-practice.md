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

![c88 teaching panel 18 (original).](../assets/figures/ml_fig_c88_18.png)
*Figure — Impact × uncertainty triage for model use. Synthetic teaching geometry—not a causal claim.*

![c89 teaching panel 18 (original).](../assets/figures/ml_fig_c89_18.png)
*Figure — Premortem failure modes board. Synthetic teaching geometry—not a causal claim.*

![c90 teaching panel 18 (original).](../assets/figures/ml_fig_c90_18.png)
*Figure — Post-deploy monitoring KPI dashboard. Synthetic teaching geometry—not a causal claim.*

![c91 teaching panel 18 (original).](../assets/figures/ml_fig_c91_18.png)
*Figure — Silent failure modes after deploy. Synthetic teaching geometry—not a causal claim.*

![c92 teaching panel 18 (original).](../assets/figures/ml_fig_c92_18.png)
*Figure — Champion/challenger model gate. Synthetic teaching geometry—not a causal claim.*

![c93 teaching panel 18 (original).](../assets/figures/ml_fig_c93_18.png)
*Figure — Shadow deployment traffic split. Synthetic teaching geometry—not a causal claim.*

![c94 teaching panel 18 (original).](../assets/figures/ml_fig_c94_18.png)
*Figure — Model card risk tier matrix. Synthetic teaching geometry—not a causal claim.*

![c95 teaching panel 18 (original).](../assets/figures/ml_fig_c95_18.png)
*Figure — Canary deployment percent ramp. Synthetic teaching geometry—not a causal claim.*

![c96 teaching panel 18 (original).](../assets/figures/ml_fig_c96_18.png)
*Figure — Rollback criteria dashboard. Synthetic teaching geometry—not a causal claim.*

![c97 teaching panel 18 (original).](../assets/figures/ml_fig_c97_18.png)
*Figure — Human-in-loop escalation ladder. Synthetic teaching geometry—not a causal claim.*

![c98 teaching panel 18 (original).](../assets/figures/ml_fig_c98_18.png)
*Figure — Blue/green deploy switch. Synthetic teaching geometry—not a causal claim.*

![c99 teaching panel 18 (original).](../assets/figures/ml_fig_c99_18.png)
*Figure — Kill-switch criteria board. Synthetic teaching geometry—not a causal claim.*

![c100 teaching panel 18 (original).](../assets/figures/ml_fig_c100_18.png)
*Figure — Safety case structure. Synthetic teaching geometry—not a causal claim.*

![c101 teaching panel 18 (original).](../assets/figures/ml_fig_c101_18.png)
*Figure — Feature flag gradual expose. Synthetic teaching geometry—not a causal claim.*

![c102 teaching panel 18 (original).](../assets/figures/ml_fig_c102_18.png)
*Figure — On-call runbook gates. Synthetic teaching geometry—not a causal claim.*

![c103 teaching panel 18 (original).](../assets/figures/ml_fig_c103_18.png)
*Figure — Red-team prompt taxonomy. Synthetic teaching geometry—not a causal claim.*

![c104 teaching panel 18 (original).](../assets/figures/ml_fig_c104_18.png)
*Figure — A/B test cuped adjustment. Synthetic teaching geometry—not a causal claim.*

![c105 teaching panel 18 (original).](../assets/figures/ml_fig_c105_18.png)
*Figure — Incident severity matrix. Synthetic teaching geometry—not a causal claim.*

![c106 teaching panel 18 (original).](../assets/figures/ml_fig_c106_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c107 teaching panel 18 (original).](../assets/figures/ml_fig_c107_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c108 teaching panel 18 (original).](../assets/figures/ml_fig_c108_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c109 teaching panel 18 (original).](../assets/figures/ml_fig_c109_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c110 teaching panel 18 (original).](../assets/figures/ml_fig_c110_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c111 teaching panel 18 (original).](../assets/figures/ml_fig_c111_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c112 teaching panel 18 (original).](../assets/figures/ml_fig_c112_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c113 teaching panel 18 (original).](../assets/figures/ml_fig_c113_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c114 teaching panel 18 (original).](../assets/figures/ml_fig_c114_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c115 teaching panel 18 (original).](../assets/figures/ml_fig_c115_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c116 teaching panel 18 (original).](../assets/figures/ml_fig_c116_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c117 teaching panel 18 (original).](../assets/figures/ml_fig_c117_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c118 teaching panel 18 (original).](../assets/figures/ml_fig_c118_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c119 teaching panel 18 (original).](../assets/figures/ml_fig_c119_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c120 teaching panel 18 (original).](../assets/figures/ml_fig_c120_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c121 teaching panel 18 (original).](../assets/figures/ml_fig_c121_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c122 teaching panel 18 (original).](../assets/figures/ml_fig_c122_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c123 teaching panel 18 (original).](../assets/figures/ml_fig_c123_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c124 teaching panel 18 (original).](../assets/figures/ml_fig_c124_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c125 teaching panel 18 (original).](../assets/figures/ml_fig_c125_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c126 teaching panel 18 (original).](../assets/figures/ml_fig_c126_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c127 teaching panel 18 (original).](../assets/figures/ml_fig_c127_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c128 teaching panel 18 (original).](../assets/figures/ml_fig_c128_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c129 teaching panel 18 (original).](../assets/figures/ml_fig_c129_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c130 teaching panel 18 (original).](../assets/figures/ml_fig_c130_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c131 teaching panel 18 (original).](../assets/figures/ml_fig_c131_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c132 teaching panel 18 (original).](../assets/figures/ml_fig_c132_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c133 teaching panel 18 (original).](../assets/figures/ml_fig_c133_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c134 teaching panel 18 (original).](../assets/figures/ml_fig_c134_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c135 teaching panel 18 (original).](../assets/figures/ml_fig_c135_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c136 teaching panel 18 (original).](../assets/figures/ml_fig_c136_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c137 teaching panel 18 (original).](../assets/figures/ml_fig_c137_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c138 teaching panel 18 (original).](../assets/figures/ml_fig_c138_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c139 teaching panel 18 (original).](../assets/figures/ml_fig_c139_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c140 teaching panel 18 (original).](../assets/figures/ml_fig_c140_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c141 teaching panel 18 (original).](../assets/figures/ml_fig_c141_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c142 teaching panel 18 (original).](../assets/figures/ml_fig_c142_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c143 teaching panel 18 (original).](../assets/figures/ml_fig_c143_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c144 teaching panel 18 (original).](../assets/figures/ml_fig_c144_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c145 teaching panel 18 (original).](../assets/figures/ml_fig_c145_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c146 teaching panel 18 (original).](../assets/figures/ml_fig_c146_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c147 teaching panel 18 (original).](../assets/figures/ml_fig_c147_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c148 teaching panel 18 (original).](../assets/figures/ml_fig_c148_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c149 teaching panel 18 (original).](../assets/figures/ml_fig_c149_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c150 teaching panel 18 (original).](../assets/figures/ml_fig_c150_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c151 teaching panel 18 (original).](../assets/figures/ml_fig_c151_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c152 teaching panel 18 (original).](../assets/figures/ml_fig_c152_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c153 teaching panel 18 (original).](../assets/figures/ml_fig_c153_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c154 teaching panel 18 (original).](../assets/figures/ml_fig_c154_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c155 teaching panel 18 (original).](../assets/figures/ml_fig_c155_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c156 teaching panel 18 (original).](../assets/figures/ml_fig_c156_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c157 teaching panel 18 (original).](../assets/figures/ml_fig_c157_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c158 teaching panel 18 (original).](../assets/figures/ml_fig_c158_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c159 teaching panel 18 (original).](../assets/figures/ml_fig_c159_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c160 teaching panel 18 (original).](../assets/figures/ml_fig_c160_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c161 teaching panel 18 (original).](../assets/figures/ml_fig_c161_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c162 teaching panel 18 (original).](../assets/figures/ml_fig_c162_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c163 teaching panel 18 (original).](../assets/figures/ml_fig_c163_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c164 teaching panel 18 (original).](../assets/figures/ml_fig_c164_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c165 teaching panel 18 (original).](../assets/figures/ml_fig_c165_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c166 teaching panel 18 (original).](../assets/figures/ml_fig_c166_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c167 teaching panel 18 (original).](../assets/figures/ml_fig_c167_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c168 teaching panel 18 (original).](../assets/figures/ml_fig_c168_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c169 teaching panel 18 (original).](../assets/figures/ml_fig_c169_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c170 teaching panel 18 (original).](../assets/figures/ml_fig_c170_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c171 teaching panel 18 (original).](../assets/figures/ml_fig_c171_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c172 teaching panel 18 (original).](../assets/figures/ml_fig_c172_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c173 teaching panel 18 (original).](../assets/figures/ml_fig_c173_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c174 teaching panel 18 (original).](../assets/figures/ml_fig_c174_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c175 teaching panel 18 (original).](../assets/figures/ml_fig_c175_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c176 teaching panel 18 (original).](../assets/figures/ml_fig_c176_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c177 teaching panel 18 (original).](../assets/figures/ml_fig_c177_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c178 teaching panel 18 (original).](../assets/figures/ml_fig_c178_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c179 teaching panel 18 (original).](../assets/figures/ml_fig_c179_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c180 teaching panel 18 (original).](../assets/figures/ml_fig_c180_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c181 teaching panel 18 (original).](../assets/figures/ml_fig_c181_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c182 teaching panel 18 (original).](../assets/figures/ml_fig_c182_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c183 teaching panel 18 (original).](../assets/figures/ml_fig_c183_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c184 teaching panel 18 (original).](../assets/figures/ml_fig_c184_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c185 teaching panel 18 (original).](../assets/figures/ml_fig_c185_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c186 teaching panel 18 (original).](../assets/figures/ml_fig_c186_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c187 teaching panel 18 (original).](../assets/figures/ml_fig_c187_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c188 teaching panel 18 (original).](../assets/figures/ml_fig_c188_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c189 teaching panel 18 (original).](../assets/figures/ml_fig_c189_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c190 teaching panel 18 (original).](../assets/figures/ml_fig_c190_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c191 teaching panel 18 (original).](../assets/figures/ml_fig_c191_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c192 teaching panel 18 (original).](../assets/figures/ml_fig_c192_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c193 teaching panel 18 (original).](../assets/figures/ml_fig_c193_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c194 teaching panel 18 (original).](../assets/figures/ml_fig_c194_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c195 teaching panel 18 (original).](../assets/figures/ml_fig_c195_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c196 teaching panel 18 (original).](../assets/figures/ml_fig_c196_18.png)
*Figure — Value of information. Synthetic teaching geometry—not a causal claim.*

![c197 teaching panel 18 (original).](../assets/figures/ml_fig_c197_18.png)
*Figure — Decision impact analysis. Synthetic teaching geometry—not a causal claim.*

![c198 teaching panel 18 (original).](../assets/figures/ml_fig_c198_18.png)
*Figure — Equity slice metrics. Synthetic teaching geometry—not a causal claim.*

![c199 teaching panel 18 (original).](../assets/figures/ml_fig_c199_18.png)
*Figure — Post-market surveillance. Synthetic teaching geometry—not a causal claim.*

![c200 teaching panel 18 (original).](../assets/figures/ml_fig_c200_18.png)
*Figure — Model retirement criteria. Synthetic teaching geometry—not a causal claim.*

![c201 teaching panel 18 (original).](../assets/figures/ml_fig_c201_18.png)
*Figure — Population stability index bins. Synthetic teaching geometry—not a causal claim.*

![c202 teaching panel 18 (original).](../assets/figures/ml_fig_c202_18.png)
*Figure — Decision curve net benefit. Synthetic teaching geometry—not a causal claim.*

![c203 teaching panel 18 (original).](../assets/figures/ml_fig_c203_18.png)
*Figure — Champion metric gate bars. Synthetic teaching geometry—not a causal claim.*

![c204 teaching panel 18 (original).](../assets/figures/ml_fig_c204_18.png)
*Figure — Silent failure metric vs volume. Synthetic teaching geometry—not a causal claim.*

![c205 teaching panel 18 (original).](../assets/figures/ml_fig_c205_18.png)
*Figure — Canary release traffic split. Synthetic teaching geometry—not a causal claim.*

![c206 teaching panel 18 (original).](../assets/figures/ml_fig_c206_18.png)
*Figure — Incident rollback playbook. Synthetic teaching geometry—not a causal claim.*

![c207 teaching panel 18 (original).](../assets/figures/ml_fig_c207_18.png)
*Figure — Shadow mode dual-write monitor. Synthetic teaching geometry—not a causal claim.*

![c208 teaching panel 18 (original).](../assets/figures/ml_fig_c208_18.png)
*Figure — Decision curve net benefit band. Synthetic teaching geometry—not a causal claim.*

![c209 teaching panel 18 (original).](../assets/figures/ml_fig_c209_18.png)
*Figure — Change failure rate monitor. Synthetic teaching geometry—not a causal claim.*

![c210 teaching panel 18 (original).](../assets/figures/ml_fig_c210_18.png)
*Figure — SLO error budget burn chart. Synthetic teaching geometry—not a causal claim.*

![c211 teaching panel 18 (original).](../assets/figures/ml_fig_c211_18.png)
*Figure — Blue-green deploy cutover. Synthetic teaching geometry—not a causal claim.*

![c212 teaching panel 18 (original).](../assets/figures/ml_fig_c212_18.png)
*Figure — Feature-flag rollout ramp. Synthetic teaching geometry—not a causal claim.*

![c213 teaching panel 18 (original).](../assets/figures/ml_fig_c213_18.png)
*Figure — Incident runbook stage tiles. Synthetic teaching geometry—not a causal claim.*

![c214 teaching panel 18 (original).](../assets/figures/ml_fig_c214_18.png)
*Figure — Blameless postmortem sections. Synthetic teaching geometry—not a causal claim.*

![c215 teaching panel 18 (original).](../assets/figures/ml_fig_c215_18.png)
*Figure — MTTR monthly trend bars. Synthetic teaching geometry—not a causal claim.*

![c216 teaching panel 18 (original).](../assets/figures/ml_fig_c216_18.png)
*Figure — Chaos game-day fault inject. Synthetic teaching geometry—not a causal claim.*

![c217 teaching panel 18 (original).](../assets/figures/ml_fig_c217_18.png)
*Figure — Error budget deploy policy. Synthetic teaching geometry—not a causal claim.*

![c218 teaching panel 18 (original).](../assets/figures/ml_fig_c218_18.png)
*Figure — Gradual percent traffic ramp. Synthetic teaching geometry—not a causal claim.*

![c219 teaching panel 18 (original).](../assets/figures/ml_fig_c219_18.png)
*Figure — SRE on-call response stages. Synthetic teaching geometry—not a causal claim.*

![c220 teaching panel 18 (original).](../assets/figures/ml_fig_c220_18.png)
*Figure — Engineering toil burn bars. Synthetic teaching geometry—not a causal claim.*

![c221 teaching panel 18 (original).](../assets/figures/ml_fig_c221_18.png)
*Figure — Architecture decision record strip. Synthetic teaching geometry—not a causal claim.*

![c222 teaching panel 18 (original).](../assets/figures/ml_fig_c222_18.png)
*Figure — Incident postmortem learning loop. Synthetic teaching geometry—not a causal claim.*

![c223 teaching panel 18 (original).](../assets/figures/ml_fig_c223_18.png)
*Figure — SRE error budget burn. Synthetic teaching geometry—not a causal claim.*

![c224 teaching panel 18 (original).](../assets/figures/ml_fig_c224_18.png)
*Figure — Incident runbook ladder. Synthetic teaching geometry—not a causal claim.*

![c225 teaching panel 18 (original).](../assets/figures/ml_fig_c225_18.png)
*Figure — Capacity plan load vs ceiling. Synthetic teaching geometry—not a causal claim.*

![c226 teaching panel 18 (original).](../assets/figures/ml_fig_c226_18.png)
*Figure — Blameless incident timeline. Synthetic teaching geometry—not a causal claim.*

![c227 teaching panel 18 (original).](../assets/figures/ml_fig_c227_18.png)
*Figure — Chaos game-day drill stages. Synthetic teaching geometry—not a causal claim.*

![c228 teaching panel 18 (original).](../assets/figures/ml_fig_c228_18.png)
*Figure — Game-day readiness scorecard. Synthetic teaching geometry—not a causal claim.*

![c229 teaching panel 18 (original).](../assets/figures/ml_fig_c229_18.png)
*Figure — SLO availability dashboard. Synthetic teaching geometry—not a causal claim.*

![c230 teaching panel 18 (original).](../assets/figures/ml_fig_c230_18.png)
*Figure — Ops review agenda strip. Synthetic teaching geometry—not a causal claim.*

![c231 teaching panel 18 (original).](../assets/figures/ml_fig_c231_18.png)
*Figure — On-call handoff checklist. Synthetic teaching geometry—not a causal claim.*

![c232 teaching panel 18 (original).](../assets/figures/ml_fig_c232_18.png)
*Figure — Runbook severity decision tree. Synthetic teaching geometry—not a causal claim.*

![c233 teaching panel 18 (original).](../assets/figures/ml_fig_c233_18.png)
*Figure — Error budget burn alert. Synthetic teaching geometry—not a causal claim.*

![c234 teaching panel 18 (original).](../assets/figures/ml_fig_c234_18.png)
*Figure — Action item burn-down bars. Synthetic teaching geometry—not a causal claim.*

![c235 teaching panel 18 (original).](../assets/figures/ml_fig_c235_18.png)
*Figure — Multi-window burn alerts. Synthetic teaching geometry—not a causal claim.*

![c236 teaching panel 18 (original).](../assets/figures/ml_fig_c236_18.png)
*Figure — Toil hour burn-down. Synthetic teaching geometry—not a causal claim.*

![c237 teaching panel 18 (original).](../assets/figures/ml_fig_c237_18.png)
*Figure — SLO error budget path. Synthetic teaching geometry—not a causal claim.*

![c238 teaching panel 18 (original).](../assets/figures/ml_fig_c238_18.png)
*Figure — Alert noise burn-down. Synthetic teaching geometry—not a causal claim.*

![c239 teaching panel 18 (original).](../assets/figures/ml_fig_c239_18.png)
*Figure — Canary error delta path. Synthetic teaching geometry—not a causal claim.*

![c240 teaching panel 18 (original).](../assets/figures/ml_fig_c240_18.png)
*Figure — Page load burn-down. Synthetic teaching geometry—not a causal claim.*

![c241 teaching panel 18 (original).](../assets/figures/ml_fig_c241_18.png)
*Figure — Shadow traffic delta path. Synthetic teaching geometry—not a causal claim.*

![c242 teaching panel 18 (original).](../assets/figures/ml_fig_c242_18.png)
*Figure — Incident MTTR burn-down. Synthetic teaching geometry—not a causal claim.*

![c243 teaching panel 18 (original).](../assets/figures/ml_fig_c243_18.png)
*Figure — Canary promote gate path. Synthetic teaching geometry—not a causal claim.*

![c244 teaching panel 18 (original).](../assets/figures/ml_fig_c244_18.png)
*Figure — On-call toil burn-down. Synthetic teaching geometry—not a causal claim.*

![c245 teaching panel 18 (original).](../assets/figures/ml_fig_c245_18.png)
*Figure — Blue-green cutover path. Synthetic teaching geometry—not a causal claim.*

![c246 teaching panel 18 (original).](../assets/figures/ml_fig_c246_18.png)
*Figure — Error budget burn-down. Synthetic teaching geometry—not a causal claim.*

![c247 teaching panel 18 (original).](../assets/figures/ml_fig_c247_18.png)
*Figure — Progressive delivery path. Synthetic teaching geometry—not a causal claim.*

![c248 teaching panel 18 (original).](../assets/figures/ml_fig_c248_18.png)
*Figure — SLO error burn-down. Synthetic teaching geometry—not a causal claim.*

![c249 teaching panel 18 (original).](../assets/figures/ml_fig_c249_18.png)
*Figure — Feature flag ramp path. Synthetic teaching geometry—not a causal claim.*

![c250 teaching panel 18 (original).](../assets/figures/ml_fig_c250_18.png)
*Figure — Pager noise burn-down. Synthetic teaching geometry—not a causal claim.*

![c251 teaching panel 18 (original).](../assets/figures/ml_fig_c251_18.png)
*Figure — Shadow eval delta path. Synthetic teaching geometry—not a causal claim.*

![c252 teaching panel 18 (original).](../assets/figures/ml_fig_c252_18.png)
*Figure — Toil hour burn-down. Synthetic teaching geometry—not a causal claim.*

![c253 teaching panel 18 (original).](../assets/figures/ml_fig_c253_18.png)
*Figure — Canary promote path. Synthetic teaching geometry—not a causal claim.*

![c254 teaching panel 18 (original).](../assets/figures/ml_fig_c254_18.png)
*Figure — Alert noise burn-down. Synthetic teaching geometry—not a causal claim.*

![c255 teaching panel 18 (original).](../assets/figures/ml_fig_c255_18.png)
*Figure — SLO burn rate path. Synthetic teaching geometry—not a causal claim.*

![c256 teaching panel 18 (original).](../assets/figures/ml_fig_c256_18.png)
*Figure — MTTR burn-down. Synthetic teaching geometry—not a causal claim.*

![c257 teaching panel 18 (original).](../assets/figures/ml_fig_c257_18.png)
*Figure — MTTR residual path c257. Synthetic teaching geometry—not a causal claim.*

![c258 teaching panel 18 (original).](../assets/figures/ml_fig_c258_18.png)
*Figure — Progressive delivery path c258. Synthetic teaching geometry—not a causal claim.*

![c259 teaching panel 18 (original).](../assets/figures/ml_fig_c259_18.png)
*Figure — Feature flag ramp path c259. Synthetic teaching geometry—not a causal claim.*

![c260 teaching panel 18 (original).](../assets/figures/ml_fig_c260_18.png)
*Figure — Model card gate path c260. Synthetic teaching geometry—not a causal claim.*

![c261 teaching panel 18 (original).](../assets/figures/ml_fig_c261_18.png)
*Figure — Eval suite residual path c261. Synthetic teaching geometry—not a causal claim.*

![c262 teaching panel 18 (original).](../assets/figures/ml_fig_c262_18.png)
*Figure — Champion challenger path c262. Synthetic teaching geometry—not a causal claim.*

![c263 teaching panel 18 (original).](../assets/figures/ml_fig_c263_18.png)
*Figure — Rollback trigger path c263. Synthetic teaching geometry—not a causal claim.*

![c264 teaching panel 18 (original).](../assets/figures/ml_fig_c264_18.png)
*Figure — Capacity headroom path c264. Synthetic teaching geometry—not a causal claim.*

![c265 teaching panel 18 (original).](../assets/figures/ml_fig_c265_18.png)
*Figure — Cost per query path c265. Synthetic teaching geometry—not a causal claim.*

![c266 teaching panel 18 (original).](../assets/figures/ml_fig_c266_18.png)
*Figure — Canary promote path c266. Synthetic teaching geometry—not a causal claim.*

![c267 teaching panel 18 (original).](../assets/figures/ml_fig_c267_18.png)
*Figure — Blue-green cutover path c267. Synthetic teaching geometry—not a causal claim.*

![c268 teaching panel 18 (original).](../assets/figures/ml_fig_c268_18.png)
*Figure — Shadow traffic path c268. Synthetic teaching geometry—not a causal claim.*

![c269 teaching panel 18 (original).](../assets/figures/ml_fig_c269_18.png)
*Figure — Error budget burn path c269. Synthetic teaching geometry—not a causal claim.*

![c270 teaching panel 18 (original).](../assets/figures/ml_fig_c270_18.png)
*Figure — SLO multi-window path c270. Synthetic teaching geometry—not a causal claim.*

![c271 teaching panel 18 (original).](../assets/figures/ml_fig_c271_18.png)
*Figure — Alert noise burn-down c271. Synthetic teaching geometry—not a causal claim.*

![c272 teaching panel 18 (original).](../assets/figures/ml_fig_c272_18.png)
*Figure — On-call toil burn-down c272. Synthetic teaching geometry—not a causal claim.*

![c273 teaching panel 18 (original).](../assets/figures/ml_fig_c273_18.png)
*Figure — MTTR residual path c273. Synthetic teaching geometry—not a causal claim.*

![c274 teaching panel 18 (original).](../assets/figures/ml_fig_c274_18.png)
*Figure — Progressive delivery path c274. Synthetic teaching geometry—not a causal claim.*

![c275 teaching panel 18 (original).](../assets/figures/ml_fig_c275_18.png)
*Figure — Feature flag ramp path c275. Synthetic teaching geometry—not a causal claim.*

![c276 teaching panel 18 (original).](../assets/figures/ml_fig_c276_18.png)
*Figure — Model card gate path c276. Synthetic teaching geometry—not a causal claim.*

![c277 teaching panel 18 (original).](../assets/figures/ml_fig_c277_18.png)
*Figure — Eval suite residual path c277. Synthetic teaching geometry—not a causal claim.*

![c278 teaching panel 18 (original).](../assets/figures/ml_fig_c278_18.png)
*Figure — Champion challenger path c278. Synthetic teaching geometry—not a causal claim.*

![c279 teaching panel 18 (original).](../assets/figures/ml_fig_c279_18.png)
*Figure — Rollback trigger path c279. Synthetic teaching geometry—not a causal claim.*

![c280 teaching panel 18 (original).](../assets/figures/ml_fig_c280_18.png)
*Figure — Capacity headroom path c280. Synthetic teaching geometry—not a causal claim.*

![c281 teaching panel 18 (original).](../assets/figures/ml_fig_c281_18.png)
*Figure — Cost per query path c281. Synthetic teaching geometry—not a causal claim.*

![c282 teaching panel 18 (original).](../assets/figures/ml_fig_c282_18.png)
*Figure — Canary promote path c282. Synthetic teaching geometry—not a causal claim.*

![c283 teaching panel 18 (original).](../assets/figures/ml_fig_c283_18.png)
*Figure — Blue-green cutover path c283. Synthetic teaching geometry—not a causal claim.*

![c284 teaching panel 18 (original).](../assets/figures/ml_fig_c284_18.png)
*Figure — Shadow traffic path c284. Synthetic teaching geometry—not a causal claim.*

![c285 teaching panel 18 (original).](../assets/figures/ml_fig_c285_18.png)
*Figure — Error budget burn path c285. Synthetic teaching geometry—not a causal claim.*

![c286 teaching panel 18 (original).](../assets/figures/ml_fig_c286_18.png)
*Figure — SLO multi-window path c286. Synthetic teaching geometry—not a causal claim.*

![c287 teaching panel 18 (original).](../assets/figures/ml_fig_c287_18.png)
*Figure — Alert noise burn-down c287. Synthetic teaching geometry—not a causal claim.*

![c288 teaching panel 18 (original).](../assets/figures/ml_fig_c288_18.png)
*Figure — On-call toil burn-down c288. Synthetic teaching geometry—not a causal claim.*

![c289 teaching panel 18 (original).](../assets/figures/ml_fig_c289_18.png)
*Figure — MTTR residual path c289. Synthetic teaching geometry—not a causal claim.*

![c290 teaching panel 18 (original).](../assets/figures/ml_fig_c290_18.png)
*Figure — Progressive delivery path c290. Synthetic teaching geometry—not a causal claim.*

![c291 teaching panel 18 (original).](../assets/figures/ml_fig_c291_18.png)
*Figure — Feature flag ramp path c291. Synthetic teaching geometry—not a causal claim.*

![c292 teaching panel 18 (original).](../assets/figures/ml_fig_c292_18.png)
*Figure — Model card gate path c292. Synthetic teaching geometry—not a causal claim.*

![c293 teaching panel 18 (original).](../assets/figures/ml_fig_c293_18.png)
*Figure — Eval suite residual path c293. Synthetic teaching geometry—not a causal claim.*

![c294 teaching panel 18 (original).](../assets/figures/ml_fig_c294_18.png)
*Figure — Champion challenger path c294. Synthetic teaching geometry—not a causal claim.*

![c295 teaching panel 18 (original).](../assets/figures/ml_fig_c295_18.png)
*Figure — Rollback trigger path c295. Synthetic teaching geometry—not a causal claim.*

![c296 teaching panel 18 (original).](../assets/figures/ml_fig_c296_18.png)
*Figure — Capacity headroom path c296. Synthetic teaching geometry—not a causal claim.*

![c297 teaching panel 18 (original).](../assets/figures/ml_fig_c297_18.png)
*Figure — Cost per query path c297. Synthetic teaching geometry—not a causal claim.*

![c298 teaching panel 18 (original).](../assets/figures/ml_fig_c298_18.png)
*Figure — Canary promote path c298. Synthetic teaching geometry—not a causal claim.*

![c299 teaching panel 18 (original).](../assets/figures/ml_fig_c299_18.png)
*Figure — Blue-green cutover path c299. Synthetic teaching geometry—not a causal claim.*

![c300 teaching panel 18 (original).](../assets/figures/ml_fig_c300_18.png)
*Figure — Shadow traffic path c300. Synthetic teaching geometry—not a causal claim.*

![c301 teaching panel 18 (original).](../assets/figures/ml_fig_c301_18.png)
*Figure — Error budget burn path c301. Synthetic teaching geometry—not a causal claim.*

![c302 teaching panel 18 (original).](../assets/figures/ml_fig_c302_18.png)
*Figure — SLO multi-window path c302. Synthetic teaching geometry—not a causal claim.*

![c303 teaching panel 18 (original).](../assets/figures/ml_fig_c303_18.png)
*Figure — Alert noise burn-down c303. Synthetic teaching geometry—not a causal claim.*

![c304 teaching panel 18 (original).](../assets/figures/ml_fig_c304_18.png)
*Figure — On-call toil burn-down c304. Synthetic teaching geometry—not a causal claim.*

![c305 teaching panel 18 (original).](../assets/figures/ml_fig_c305_18.png)
*Figure — MTTR residual path c305. Synthetic teaching geometry—not a causal claim.*

![c306 teaching panel 18 (original).](../assets/figures/ml_fig_c306_18.png)
*Figure — Progressive delivery path c306. Synthetic teaching geometry—not a causal claim.*

![c307 teaching panel 18 (original).](../assets/figures/ml_fig_c307_18.png)
*Figure — Feature flag ramp path c307. Synthetic teaching geometry—not a causal claim.*

![c308 teaching panel 18 (original).](../assets/figures/ml_fig_c308_18.png)
*Figure — Model card gate path c308. Synthetic teaching geometry—not a causal claim.*

![c309 teaching panel 18 (original).](../assets/figures/ml_fig_c309_18.png)
*Figure — Eval suite residual path c309. Synthetic teaching geometry—not a causal claim.*

![c310 teaching panel 18 (original).](../assets/figures/ml_fig_c310_18.png)
*Figure — Champion challenger path c310. Synthetic teaching geometry—not a causal claim.*

![c311 teaching panel 18 (original).](../assets/figures/ml_fig_c311_18.png)
*Figure — Rollback trigger path c311. Synthetic teaching geometry—not a causal claim.*

![c312 teaching panel 18 (original).](../assets/figures/ml_fig_c312_18.png)
*Figure — Capacity headroom path c312. Synthetic teaching geometry—not a causal claim.*

![c313 teaching panel 18 (original).](../assets/figures/ml_fig_c313_18.png)
*Figure — Cost per query path c313. Synthetic teaching geometry—not a causal claim.*

![c314 teaching panel 18 (original).](../assets/figures/ml_fig_c314_18.png)
*Figure — Canary promote path c314. Synthetic teaching geometry—not a causal claim.*

![c315 teaching panel 18 (original).](../assets/figures/ml_fig_c315_18.png)
*Figure — Blue-green cutover path c315. Synthetic teaching geometry—not a causal claim.*

![c316 teaching panel 18 (original).](../assets/figures/ml_fig_c316_18.png)
*Figure — Shadow traffic path c316. Synthetic teaching geometry—not a causal claim.*

![c317 teaching panel 18 (original).](../assets/figures/ml_fig_c317_18.png)
*Figure — Error budget burn path c317. Synthetic teaching geometry—not a causal claim.*

![c318 teaching panel 18 (original).](../assets/figures/ml_fig_c318_18.png)
*Figure — SLO multi-window path c318. Synthetic teaching geometry—not a causal claim.*

![c319 teaching panel 18 (original).](../assets/figures/ml_fig_c319_18.png)
*Figure — Alert noise burn-down c319. Synthetic teaching geometry—not a causal claim.*

![c320 teaching panel 18 (original).](../assets/figures/ml_fig_c320_18.png)
*Figure — On-call toil burn-down c320. Synthetic teaching geometry—not a causal claim.*

![c321 teaching panel 18 (original).](../assets/figures/ml_fig_c321_18.png)
*Figure — MTTR residual path c321. Synthetic teaching geometry—not a causal claim.*

![c322 teaching panel 18 (original).](../assets/figures/ml_fig_c322_18.png)
*Figure — Progressive delivery path c322. Synthetic teaching geometry—not a causal claim.*

![c323 teaching panel 18 (original).](../assets/figures/ml_fig_c323_18.png)
*Figure — Feature flag ramp path c323. Synthetic teaching geometry—not a causal claim.*

![c324 teaching panel 18 (original).](../assets/figures/ml_fig_c324_18.png)
*Figure — Model card gate path c324. Synthetic teaching geometry—not a causal claim.*

![c325 teaching panel 18 (original).](../assets/figures/ml_fig_c325_18.png)
*Figure — Eval suite residual path c325. Synthetic teaching geometry—not a causal claim.*

![c326 teaching panel 18 (original).](../assets/figures/ml_fig_c326_18.png)
*Figure — Champion challenger path c326. Synthetic teaching geometry—not a causal claim.*

![c327 teaching panel 18 (original).](../assets/figures/ml_fig_c327_18.png)
*Figure — Rollback trigger path c327. Synthetic teaching geometry—not a causal claim.*

![c328 teaching panel 18 (original).](../assets/figures/ml_fig_c328_18.png)
*Figure — Capacity headroom path c328. Synthetic teaching geometry—not a causal claim.*

![c329 teaching panel 18 (original).](../assets/figures/ml_fig_c329_18.png)
*Figure — Cost per query path c329. Synthetic teaching geometry—not a causal claim.*

![c330 teaching panel 18 (original).](../assets/figures/ml_fig_c330_18.png)
*Figure — Canary promote path c330. Synthetic teaching geometry—not a causal claim.*

![c331 teaching panel 18 (original).](../assets/figures/ml_fig_c331_18.png)
*Figure — Blue-green cutover path c331. Synthetic teaching geometry—not a causal claim.*

![c332 teaching panel 18 (original).](../assets/figures/ml_fig_c332_18.png)
*Figure — Shadow traffic path c332. Synthetic teaching geometry—not a causal claim.*

![c333 teaching panel 18 (original).](../assets/figures/ml_fig_c333_18.png)
*Figure — Error budget burn path c333. Synthetic teaching geometry—not a causal claim.*

![c334 teaching panel 18 (original).](../assets/figures/ml_fig_c334_18.png)
*Figure — SLO multi-window path c334. Synthetic teaching geometry—not a causal claim.*

![c335 teaching panel 18 (original).](../assets/figures/ml_fig_c335_18.png)
*Figure — Alert noise burn-down c335. Synthetic teaching geometry—not a causal claim.*

![c336 teaching panel 18 (original).](../assets/figures/ml_fig_c336_18.png)
*Figure — On-call toil burn-down c336. Synthetic teaching geometry—not a causal claim.*

![c337 teaching panel 18 (original).](../assets/figures/ml_fig_c337_18.png)
*Figure — MTTR residual path c337. Synthetic teaching geometry—not a causal claim.*

![c338 teaching panel 18 (original).](../assets/figures/ml_fig_c338_18.png)
*Figure — Progressive delivery path c338. Synthetic teaching geometry—not a causal claim.*

![c339 teaching panel 18 (original).](../assets/figures/ml_fig_c339_18.png)
*Figure — Feature flag ramp path c339. Synthetic teaching geometry—not a causal claim.*

![c340 teaching panel 18 (original).](../assets/figures/ml_fig_c340_18.png)
*Figure — Model card gate path c340. Synthetic teaching geometry—not a causal claim.*

![c341 teaching panel 18 (original).](../assets/figures/ml_fig_c341_18.png)
*Figure — Eval suite residual path c341. Synthetic teaching geometry—not a causal claim.*

![c342 teaching panel 18 (original).](../assets/figures/ml_fig_c342_18.png)
*Figure — Champion challenger path c342. Synthetic teaching geometry—not a causal claim.*

![c343 teaching panel 18 (original).](../assets/figures/ml_fig_c343_18.png)
*Figure — Rollback trigger path c343. Synthetic teaching geometry—not a causal claim.*

![c344 teaching panel 18 (original).](../assets/figures/ml_fig_c344_18.png)
*Figure — Capacity headroom path c344. Synthetic teaching geometry—not a causal claim.*

![c345 teaching panel 18 (original).](../assets/figures/ml_fig_c345_18.png)
*Figure — Cost per query path c345. Synthetic teaching geometry—not a causal claim.*

![c346 teaching panel 18 (original).](../assets/figures/ml_fig_c346_18.png)
*Figure — Canary promote path c346. Synthetic teaching geometry—not a causal claim.*

![c347 teaching panel 18 (original).](../assets/figures/ml_fig_c347_18.png)
*Figure — Blue-green cutover path c347. Synthetic teaching geometry—not a causal claim.*

![c348 teaching panel 18 (original).](../assets/figures/ml_fig_c348_18.png)
*Figure — Shadow traffic path c348. Synthetic teaching geometry—not a causal claim.*

![c349 teaching panel 18 (original).](../assets/figures/ml_fig_c349_18.png)
*Figure — Error budget burn path c349. Synthetic teaching geometry—not a causal claim.*

![c350 teaching panel 18 (original).](../assets/figures/ml_fig_c350_18.png)
*Figure — SLO multi-window path c350. Synthetic teaching geometry—not a causal claim.*

![c351 teaching panel 18 (original).](../assets/figures/ml_fig_c351_18.png)
*Figure — Alert noise burn-down c351. Synthetic teaching geometry—not a causal claim.*

![c352 teaching panel 18 (original).](../assets/figures/ml_fig_c352_18.png)
*Figure — On-call toil burn-down c352. Synthetic teaching geometry—not a causal claim.*

![c353 teaching panel 18 (original).](../assets/figures/ml_fig_c353_18.png)
*Figure — MTTR residual path c353. Synthetic teaching geometry—not a causal claim.*

![c354 teaching panel 18 (original).](../assets/figures/ml_fig_c354_18.png)
*Figure — Progressive delivery path c354. Synthetic teaching geometry—not a causal claim.*

![c355 teaching panel 18 (original).](../assets/figures/ml_fig_c355_18.png)
*Figure — Feature flag ramp path c355. Synthetic teaching geometry—not a causal claim.*

![c356 teaching panel 18 (original).](../assets/figures/ml_fig_c356_18.png)
*Figure — Model card gate path c356. Synthetic teaching geometry—not a causal claim.*

![c357 teaching panel 18 (original).](../assets/figures/ml_fig_c357_18.png)
*Figure — Eval suite residual path c357. Synthetic teaching geometry—not a causal claim.*

![c358 teaching panel 18 (original).](../assets/figures/ml_fig_c358_18.png)
*Figure — Champion challenger path c358. Synthetic teaching geometry—not a causal claim.*

![c359 teaching panel 18 (original).](../assets/figures/ml_fig_c359_18.png)
*Figure — Rollback trigger path c359. Synthetic teaching geometry—not a causal claim.*

![c360 teaching panel 18 (original).](../assets/figures/ml_fig_c360_18.png)
*Figure — Capacity headroom path c360. Synthetic teaching geometry—not a causal claim.*

![c361 teaching panel 18 (original).](../assets/figures/ml_fig_c361_18.png)
*Figure — Cost per query path c361. Synthetic teaching geometry—not a causal claim.*

![c362 teaching panel 18 (original).](../assets/figures/ml_fig_c362_18.png)
*Figure — Canary promote path c362. Synthetic teaching geometry—not a causal claim.*

![c363 teaching panel 18 (original).](../assets/figures/ml_fig_c363_18.png)
*Figure — Blue-green cutover path c363. Synthetic teaching geometry—not a causal claim.*

![c364 teaching panel 18 (original).](../assets/figures/ml_fig_c364_18.png)
*Figure — Shadow traffic path c364. Synthetic teaching geometry—not a causal claim.*

![c365 teaching panel 18 (original).](../assets/figures/ml_fig_c365_18.png)
*Figure — Error budget burn path c365. Synthetic teaching geometry—not a causal claim.*

![c366 teaching panel 18 (original).](../assets/figures/ml_fig_c366_18.png)
*Figure — SLO multi-window path c366. Synthetic teaching geometry—not a causal claim.*

![c367 teaching panel 18 (original).](../assets/figures/ml_fig_c367_18.png)
*Figure — Alert noise burn-down c367. Synthetic teaching geometry—not a causal claim.*

![c368 teaching panel 18 (original).](../assets/figures/ml_fig_c368_18.png)
*Figure — On-call toil burn-down c368. Synthetic teaching geometry—not a causal claim.*

![c369 teaching panel 18 (original).](../assets/figures/ml_fig_c369_18.png)
*Figure — MTTR residual path c369. Synthetic teaching geometry—not a causal claim.*

![c370 teaching panel 18 (original).](../assets/figures/ml_fig_c370_18.png)
*Figure — Progressive delivery path c370. Synthetic teaching geometry—not a causal claim.*

![c371 teaching panel 18 (original).](../assets/figures/ml_fig_c371_18.png)
*Figure — Feature flag ramp path c371. Synthetic teaching geometry—not a causal claim.*

![c372 teaching panel 18 (original).](../assets/figures/ml_fig_c372_18.png)
*Figure — Model card gate path c372. Synthetic teaching geometry—not a causal claim.*

![c373 teaching panel 18 (original).](../assets/figures/ml_fig_c373_18.png)
*Figure — Eval suite residual path c373. Synthetic teaching geometry—not a causal claim.*

![c374 teaching panel 18 (original).](../assets/figures/ml_fig_c374_18.png)
*Figure — Champion challenger path c374. Synthetic teaching geometry—not a causal claim.*

![c375 teaching panel 18 (original).](../assets/figures/ml_fig_c375_18.png)
*Figure — Rollback trigger path c375. Synthetic teaching geometry—not a causal claim.*

![c376 teaching panel 18 (original).](../assets/figures/ml_fig_c376_18.png)
*Figure — Capacity headroom path c376. Synthetic teaching geometry—not a causal claim.*

![c377 teaching panel 18 (original).](../assets/figures/ml_fig_c377_18.png)
*Figure — Cost per query path c377. Synthetic teaching geometry—not a causal claim.*

![c378 teaching panel 18 (original).](../assets/figures/ml_fig_c378_18.png)
*Figure — Canary promote path c378. Synthetic teaching geometry—not a causal claim.*

![c379 teaching panel 18 (original).](../assets/figures/ml_fig_c379_18.png)
*Figure — Blue-green cutover path c379. Synthetic teaching geometry—not a causal claim.*

![c380 teaching panel 18 (original).](../assets/figures/ml_fig_c380_18.png)
*Figure — Shadow traffic path c380. Synthetic teaching geometry—not a causal claim.*

![c381 teaching panel 18 (original).](../assets/figures/ml_fig_c381_18.png)
*Figure — Error budget burn path c381. Synthetic teaching geometry—not a causal claim.*

![c382 teaching panel 18 (original).](../assets/figures/ml_fig_c382_18.png)
*Figure — SLO multi-window path c382. Synthetic teaching geometry—not a causal claim.*

![c383 teaching panel 18 (original).](../assets/figures/ml_fig_c383_18.png)
*Figure — Alert noise burn-down c383. Synthetic teaching geometry—not a causal claim.*

![c384 teaching panel 18 (original).](../assets/figures/ml_fig_c384_18.png)
*Figure — On-call toil burn-down c384. Synthetic teaching geometry—not a causal claim.*

![c385 teaching panel 18 (original).](../assets/figures/ml_fig_c385_18.png)
*Figure — MTTR residual path c385. Synthetic teaching geometry—not a causal claim.*

![c386 teaching panel 18 (original).](../assets/figures/ml_fig_c386_18.png)
*Figure — Progressive delivery path c386. Synthetic teaching geometry—not a causal claim.*

![c387 teaching panel 18 (original).](../assets/figures/ml_fig_c387_18.png)
*Figure — Feature flag ramp path c387. Synthetic teaching geometry—not a causal claim.*

![c388 teaching panel 18 (original).](../assets/figures/ml_fig_c388_18.png)
*Figure — Model card gate path c388. Synthetic teaching geometry—not a causal claim.*

![c389 teaching panel 18 (original).](../assets/figures/ml_fig_c389_18.png)
*Figure — Eval suite residual path c389. Synthetic teaching geometry—not a causal claim.*

![c390 teaching panel 18 (original).](../assets/figures/ml_fig_c390_18.png)
*Figure — Champion challenger path c390. Synthetic teaching geometry—not a causal claim.*

![c391 teaching panel 18 (original).](../assets/figures/ml_fig_c391_18.png)
*Figure — Rollback trigger path c391. Synthetic teaching geometry—not a causal claim.*

![c392 teaching panel 18 (original).](../assets/figures/ml_fig_c392_18.png)
*Figure — Capacity headroom path c392. Synthetic teaching geometry—not a causal claim.*

![c393 teaching panel 18 (original).](../assets/figures/ml_fig_c393_18.png)
*Figure — Cost per query path c393. Synthetic teaching geometry—not a causal claim.*

![c394 teaching panel 18 (original).](../assets/figures/ml_fig_c394_18.png)
*Figure — Canary promote path c394. Synthetic teaching geometry—not a causal claim.*

![c395 teaching panel 18 (original).](../assets/figures/ml_fig_c395_18.png)
*Figure — Blue-green cutover path c395. Synthetic teaching geometry—not a causal claim.*

![c396 teaching panel 18 (original).](../assets/figures/ml_fig_c396_18.png)
*Figure — Shadow traffic path c396. Synthetic teaching geometry—not a causal claim.*

![c397 teaching panel 18 (original).](../assets/figures/ml_fig_c397_18.png)
*Figure — Error budget burn path c397. Synthetic teaching geometry—not a causal claim.*

![c398 teaching panel 18 (original).](../assets/figures/ml_fig_c398_18.png)
*Figure — SLO multi-window path c398. Synthetic teaching geometry—not a causal claim.*

![c399 teaching panel 18 (original).](../assets/figures/ml_fig_c399_18.png)
*Figure — Alert noise burn-down c399. Synthetic teaching geometry—not a causal claim.*

![c400 teaching panel 18 (original).](../assets/figures/ml_fig_c400_18.png)
*Figure — On-call toil burn-down c400. Synthetic teaching geometry—not a causal claim.*

![c401 teaching panel 18 (original).](../assets/figures/ml_fig_c401_18.png)
*Figure — MTTR residual path c401. Synthetic teaching geometry—not a causal claim.*

![c402 teaching panel 18 (original).](../assets/figures/ml_fig_c402_18.png)
*Figure — Progressive delivery path c402. Synthetic teaching geometry—not a causal claim.*

![c403 teaching panel 18 (original).](../assets/figures/ml_fig_c403_18.png)
*Figure — Feature flag ramp path c403. Synthetic teaching geometry—not a causal claim.*

![c404 teaching panel 18 (original).](../assets/figures/ml_fig_c404_18.png)
*Figure — Model card gate path c404. Synthetic teaching geometry—not a causal claim.*

![c405 teaching panel 18 (original).](../assets/figures/ml_fig_c405_18.png)
*Figure — Eval suite residual path c405. Synthetic teaching geometry—not a causal claim.*

![c406 teaching panel 18 (original).](../assets/figures/ml_fig_c406_18.png)
*Figure — Champion challenger path c406. Synthetic teaching geometry—not a causal claim.*

![c407 teaching panel 18 (original).](../assets/figures/ml_fig_c407_18.png)
*Figure — Rollback trigger path c407. Synthetic teaching geometry—not a causal claim.*

![c408 teaching panel 18 (original).](../assets/figures/ml_fig_c408_18.png)
*Figure — Capacity headroom path c408. Synthetic teaching geometry—not a causal claim.*

![c409 teaching panel 18 (original).](../assets/figures/ml_fig_c409_18.png)
*Figure — Cost per query path c409. Synthetic teaching geometry—not a causal claim.*

![c410 teaching panel 18 (original).](../assets/figures/ml_fig_c410_18.png)
*Figure — Canary promote path c410. Synthetic teaching geometry—not a causal claim.*

![c411 teaching panel 18 (original).](../assets/figures/ml_fig_c411_18.png)
*Figure — Blue-green cutover path c411. Synthetic teaching geometry—not a causal claim.*

![c412 teaching panel 18 (original).](../assets/figures/ml_fig_c412_18.png)
*Figure — Shadow traffic path c412. Synthetic teaching geometry—not a causal claim.*

![c413 teaching panel 18 (original).](../assets/figures/ml_fig_c413_18.png)
*Figure — Error budget burn path c413. Synthetic teaching geometry—not a causal claim.*

![c414 teaching panel 18 (original).](../assets/figures/ml_fig_c414_18.png)
*Figure — SLO multi-window path c414. Synthetic teaching geometry—not a causal claim.*

![c415 teaching panel 18 (original).](../assets/figures/ml_fig_c415_18.png)
*Figure — Alert noise burn-down c415. Synthetic teaching geometry—not a causal claim.*

![c416 teaching panel 18 (original).](../assets/figures/ml_fig_c416_18.png)
*Figure — On-call toil burn-down c416. Synthetic teaching geometry—not a causal claim.*

![c417 teaching panel 18 (original).](../assets/figures/ml_fig_c417_18.png)
*Figure — MTTR residual path c417. Synthetic teaching geometry—not a causal claim.*

![c418 teaching panel 18 (original).](../assets/figures/ml_fig_c418_18.png)
*Figure — Progressive delivery path c418. Synthetic teaching geometry—not a causal claim.*

![c419 teaching panel 18 (original).](../assets/figures/ml_fig_c419_18.png)
*Figure — Feature flag ramp path c419. Synthetic teaching geometry—not a causal claim.*

![c420 teaching panel 18 (original).](../assets/figures/ml_fig_c420_18.png)
*Figure — Model card gate path c420. Synthetic teaching geometry—not a causal claim.*

![c421 teaching panel 18 (original).](../assets/figures/ml_fig_c421_18.png)
*Figure — Eval suite residual path c421. Synthetic teaching geometry—not a causal claim.*

![c422 teaching panel 18 (original).](../assets/figures/ml_fig_c422_18.png)
*Figure — Champion challenger path c422. Synthetic teaching geometry—not a causal claim.*

![c423 teaching panel 18 (original).](../assets/figures/ml_fig_c423_18.png)
*Figure — Rollback trigger path c423. Synthetic teaching geometry—not a causal claim.*

![c424 teaching panel 18 (original).](../assets/figures/ml_fig_c424_18.png)
*Figure — Capacity headroom path c424. Synthetic teaching geometry—not a causal claim.*

![c425 teaching panel 18 (original).](../assets/figures/ml_fig_c425_18.png)
*Figure — Cost per query path c425. Synthetic teaching geometry—not a causal claim.*

![c426 teaching panel 18 (original).](../assets/figures/ml_fig_c426_18.png)
*Figure — Canary promote path c426. Synthetic teaching geometry—not a causal claim.*

![c427 teaching panel 18 (original).](../assets/figures/ml_fig_c427_18.png)
*Figure — Blue-green cutover path c427. Synthetic teaching geometry—not a causal claim.*

![c428 teaching panel 18 (original).](../assets/figures/ml_fig_c428_18.png)
*Figure — Shadow traffic path c428. Synthetic teaching geometry—not a causal claim.*

![c429 teaching panel 18 (original).](../assets/figures/ml_fig_c429_18.png)
*Figure — Error budget burn path c429. Synthetic teaching geometry—not a causal claim.*

![c430 teaching panel 18 (original).](../assets/figures/ml_fig_c430_18.png)
*Figure — SLO multi-window path c430. Synthetic teaching geometry—not a causal claim.*

![c431 teaching panel 18 (original).](../assets/figures/ml_fig_c431_18.png)
*Figure — Alert noise burn-down c431. Synthetic teaching geometry—not a causal claim.*

![c432 teaching panel 18 (original).](../assets/figures/ml_fig_c432_18.png)
*Figure — On-call toil burn-down c432. Synthetic teaching geometry—not a causal claim.*

![c433 teaching panel 18 (original).](../assets/figures/ml_fig_c433_18.png)
*Figure — MTTR residual path c433. Synthetic teaching geometry—not a causal claim.*

![c434 teaching panel 18 (original).](../assets/figures/ml_fig_c434_18.png)
*Figure — Progressive delivery path c434. Synthetic teaching geometry—not a causal claim.*

![c435 teaching panel 18 (original).](../assets/figures/ml_fig_c435_18.png)
*Figure — Feature flag ramp path c435. Synthetic teaching geometry—not a causal claim.*

![c436 teaching panel 18 (original).](../assets/figures/ml_fig_c436_18.png)
*Figure — Model card gate path c436. Synthetic teaching geometry—not a causal claim.*

![c437 teaching panel 18 (original).](../assets/figures/ml_fig_c437_18.png)
*Figure — Eval suite residual path c437. Synthetic teaching geometry—not a causal claim.*

![c438 teaching panel 18 (original).](../assets/figures/ml_fig_c438_18.png)
*Figure — Champion challenger path c438. Synthetic teaching geometry—not a causal claim.*

![c439 teaching panel 18 (original).](../assets/figures/ml_fig_c439_18.png)
*Figure — Rollback trigger path c439. Synthetic teaching geometry—not a causal claim.*

![c440 teaching panel 18 (original).](../assets/figures/ml_fig_c440_18.png)
*Figure — Capacity headroom path c440. Synthetic teaching geometry—not a causal claim.*

![c441 teaching panel 18 (original).](../assets/figures/ml_fig_c441_18.png)
*Figure — Cost per query path c441. Synthetic teaching geometry—not a causal claim.*

![c442 teaching panel 18 (original).](../assets/figures/ml_fig_c442_18.png)
*Figure — Canary promote path c442. Synthetic teaching geometry—not a causal claim.*

![c443 teaching panel 18 (original).](../assets/figures/ml_fig_c443_18.png)
*Figure — Blue-green cutover path c443. Synthetic teaching geometry—not a causal claim.*

![c444 teaching panel 18 (original).](../assets/figures/ml_fig_c444_18.png)
*Figure — Shadow traffic path c444. Synthetic teaching geometry—not a causal claim.*

![c445 teaching panel 18 (original).](../assets/figures/ml_fig_c445_18.png)
*Figure — Error budget burn path c445. Synthetic teaching geometry—not a causal claim.*

![c446 teaching panel 18 (original).](../assets/figures/ml_fig_c446_18.png)
*Figure — SLO multi-window path c446. Synthetic teaching geometry—not a causal claim.*

![c447 teaching panel 18 (original).](../assets/figures/ml_fig_c447_18.png)
*Figure — Alert noise burn-down c447. Synthetic teaching geometry—not a causal claim.*

![c448 teaching panel 18 (original).](../assets/figures/ml_fig_c448_18.png)
*Figure — On-call toil burn-down c448. Synthetic teaching geometry—not a causal claim.*

![c449 teaching panel 18 (original).](../assets/figures/ml_fig_c449_18.png)
*Figure — MTTR residual path c449. Synthetic teaching geometry—not a causal claim.*

![c450 teaching panel 18 (original).](../assets/figures/ml_fig_c450_18.png)
*Figure — Progressive delivery path c450. Synthetic teaching geometry—not a causal claim.*

![c451 teaching panel 18 (original).](../assets/figures/ml_fig_c451_18.png)
*Figure — Feature flag ramp path c451. Synthetic teaching geometry—not a causal claim.*

![c452 teaching panel 18 (original).](../assets/figures/ml_fig_c452_18.png)
*Figure — Model card gate path c452. Synthetic teaching geometry—not a causal claim.*

![c453 teaching panel 18 (original).](../assets/figures/ml_fig_c453_18.png)
*Figure — Eval suite residual path c453. Synthetic teaching geometry—not a causal claim.*

![c454 teaching panel 18 (original).](../assets/figures/ml_fig_c454_18.png)
*Figure — Champion challenger path c454. Synthetic teaching geometry—not a causal claim.*

![c455 teaching panel 18 (original).](../assets/figures/ml_fig_c455_18.png)
*Figure — Rollback trigger path c455. Synthetic teaching geometry—not a causal claim.*

![c456 teaching panel 18 (original).](../assets/figures/ml_fig_c456_18.png)
*Figure — Capacity headroom path c456. Synthetic teaching geometry—not a causal claim.*

![c457 teaching panel 18 (original).](../assets/figures/ml_fig_c457_18.png)
*Figure — Cost per query path c457. Synthetic teaching geometry—not a causal claim.*

![c458 teaching panel 18 (original).](../assets/figures/ml_fig_c458_18.png)
*Figure — Canary promote path c458. Synthetic teaching geometry—not a causal claim.*

![c459 teaching panel 18 (original).](../assets/figures/ml_fig_c459_18.png)
*Figure — Blue-green cutover path c459. Synthetic teaching geometry—not a causal claim.*

![c460 teaching panel 18 (original).](../assets/figures/ml_fig_c460_18.png)
*Figure — Shadow traffic path c460. Synthetic teaching geometry—not a causal claim.*

![c461 teaching panel 18 (original).](../assets/figures/ml_fig_c461_18.png)
*Figure — Error budget burn path c461. Synthetic teaching geometry—not a causal claim.*

![c462 teaching panel 18 (original).](../assets/figures/ml_fig_c462_18.png)
*Figure — SLO multi-window path c462. Synthetic teaching geometry—not a causal claim.*

![c463 teaching panel 18 (original).](../assets/figures/ml_fig_c463_18.png)
*Figure — Alert noise burn-down c463. Synthetic teaching geometry—not a causal claim.*

![c464 teaching panel 18 (original).](../assets/figures/ml_fig_c464_18.png)
*Figure — On-call toil burn-down c464. Synthetic teaching geometry—not a causal claim.*

![c465 teaching panel 18 (original).](../assets/figures/ml_fig_c465_18.png)
*Figure — MTTR residual path c465. Synthetic teaching geometry—not a causal claim.*

![c466 teaching panel 18 (original).](../assets/figures/ml_fig_c466_18.png)
*Figure — Progressive delivery path c466. Synthetic teaching geometry—not a causal claim.*

![c467 teaching panel 18 (original).](../assets/figures/ml_fig_c467_18.png)
*Figure — Feature flag ramp path c467. Synthetic teaching geometry—not a causal claim.*

![c468 teaching panel 18 (original).](../assets/figures/ml_fig_c468_18.png)
*Figure — Model card gate path c468. Synthetic teaching geometry—not a causal claim.*

![c469 teaching panel 18 (original).](../assets/figures/ml_fig_c469_18.png)
*Figure — Eval suite residual path c469. Synthetic teaching geometry—not a causal claim.*

![c470 teaching panel 18 (original).](../assets/figures/ml_fig_c470_18.png)
*Figure — Champion challenger path c470. Synthetic teaching geometry—not a causal claim.*

![c471 teaching panel 18 (original).](../assets/figures/ml_fig_c471_18.png)
*Figure — Rollback trigger path c471. Synthetic teaching geometry—not a causal claim.*

![c472 teaching panel 18 (original).](../assets/figures/ml_fig_c472_18.png)
*Figure — Capacity headroom path c472. Synthetic teaching geometry—not a causal claim.*

![c473 teaching panel 18 (original).](../assets/figures/ml_fig_c473_18.png)
*Figure — Cost per query path c473. Synthetic teaching geometry—not a causal claim.*

![c474 teaching panel 18 (original).](../assets/figures/ml_fig_c474_18.png)
*Figure — Canary promote path c474. Synthetic teaching geometry—not a causal claim.*

![c475 teaching panel 18 (original).](../assets/figures/ml_fig_c475_18.png)
*Figure — Blue-green cutover path c475. Synthetic teaching geometry—not a causal claim.*

![c476 teaching panel 18 (original).](../assets/figures/ml_fig_c476_18.png)
*Figure — Shadow traffic path c476. Synthetic teaching geometry—not a causal claim.*

![c477 teaching panel 18 (original).](../assets/figures/ml_fig_c477_18.png)
*Figure — Error budget burn path c477. Synthetic teaching geometry—not a causal claim.*

![c478 teaching panel 18 (original).](../assets/figures/ml_fig_c478_18.png)
*Figure — SLO multi-window path c478. Synthetic teaching geometry—not a causal claim.*

![c479 teaching panel 18 (original).](../assets/figures/ml_fig_c479_18.png)
*Figure — Alert noise burn-down c479. Synthetic teaching geometry—not a causal claim.*

![c480 teaching panel 18 (original).](../assets/figures/ml_fig_c480_18.png)
*Figure — On-call toil burn-down c480. Synthetic teaching geometry—not a causal claim.*

![c481 teaching panel 18 (original).](../assets/figures/ml_fig_c481_18.png)
*Figure — MTTR residual path c481. Synthetic teaching geometry—not a causal claim.*

![c482 teaching panel 18 (original).](../assets/figures/ml_fig_c482_18.png)
*Figure — Progressive delivery path c482. Synthetic teaching geometry—not a causal claim.*

![c483 teaching panel 18 (original).](../assets/figures/ml_fig_c483_18.png)
*Figure — Feature flag ramp path c483. Synthetic teaching geometry—not a causal claim.*

![c484 teaching panel 18 (original).](../assets/figures/ml_fig_c484_18.png)
*Figure — Model card gate path c484. Synthetic teaching geometry—not a causal claim.*

![c485 teaching panel 18 (original).](../assets/figures/ml_fig_c485_18.png)
*Figure — Eval suite residual path c485. Synthetic teaching geometry—not a causal claim.*

![c486 teaching panel 18 (original).](../assets/figures/ml_fig_c486_18.png)
*Figure — Champion challenger path c486. Synthetic teaching geometry—not a causal claim.*

![c487 teaching panel 18 (original).](../assets/figures/ml_fig_c487_18.png)
*Figure — Rollback trigger path c487. Synthetic teaching geometry—not a causal claim.*

![c488 teaching panel 18 (original).](../assets/figures/ml_fig_c488_18.png)
*Figure — Capacity headroom path c488. Synthetic teaching geometry—not a causal claim.*

![c489 teaching panel 18 (original).](../assets/figures/ml_fig_c489_18.png)
*Figure — Cost per query path c489. Synthetic teaching geometry—not a causal claim.*

![c490 teaching panel 18 (original).](../assets/figures/ml_fig_c490_18.png)
*Figure — Canary promote path c490. Synthetic teaching geometry—not a causal claim.*

![c491 teaching panel 18 (original).](../assets/figures/ml_fig_c491_18.png)
*Figure — Blue-green cutover path c491. Synthetic teaching geometry—not a causal claim.*

![c492 teaching panel 18 (original).](../assets/figures/ml_fig_c492_18.png)
*Figure — Shadow traffic path c492. Synthetic teaching geometry—not a causal claim.*

![c493 teaching panel 18 (original).](../assets/figures/ml_fig_c493_18.png)
*Figure — Error budget burn path c493. Synthetic teaching geometry—not a causal claim.*

![c494 teaching panel 18 (original).](../assets/figures/ml_fig_c494_18.png)
*Figure — SLO multi-window path c494. Synthetic teaching geometry—not a causal claim.*

![c495 teaching panel 18 (original).](../assets/figures/ml_fig_c495_18.png)
*Figure — Alert noise burn-down c495. Synthetic teaching geometry—not a causal claim.*

![c496 teaching panel 18 (original).](../assets/figures/ml_fig_c496_18.png)
*Figure — On-call toil burn-down c496. Synthetic teaching geometry—not a causal claim.*

![c497 teaching panel 18 (original).](../assets/figures/ml_fig_c497_18.png)
*Figure — MTTR residual path c497. Synthetic teaching geometry—not a causal claim.*

![c498 teaching panel 18 (original).](../assets/figures/ml_fig_c498_18.png)
*Figure — Progressive delivery path c498. Synthetic teaching geometry—not a causal claim.*

![c499 teaching panel 18 (original).](../assets/figures/ml_fig_c499_18.png)
*Figure — Feature flag ramp path c499. Synthetic teaching geometry—not a causal claim.*

![c500 teaching panel 18 (original).](../assets/figures/ml_fig_c500_18.png)
*Figure — Model card gate path c500. Synthetic teaching geometry—not a causal claim.*

![c501 teaching panel 18 (original).](../assets/figures/ml_fig_c501_18.png)
*Figure — Eval suite residual path c501. Synthetic teaching geometry—not a causal claim.*

![c502 teaching panel 18 (original).](../assets/figures/ml_fig_c502_18.png)
*Figure — Champion challenger path c502. Synthetic teaching geometry—not a causal claim.*

![c503 teaching panel 18 (original).](../assets/figures/ml_fig_c503_18.png)
*Figure — Rollback trigger path c503. Synthetic teaching geometry—not a causal claim.*

![c504 teaching panel 18 (original).](../assets/figures/ml_fig_c504_18.png)
*Figure — Capacity headroom path c504. Synthetic teaching geometry—not a causal claim.*

![c505 teaching panel 18 (original).](../assets/figures/ml_fig_c505_18.png)
*Figure — Cost per query path c505. Synthetic teaching geometry—not a causal claim.*

![c506 teaching panel 18 (original).](../assets/figures/ml_fig_c506_18.png)
*Figure — Canary promote path c506. Synthetic teaching geometry—not a causal claim.*

![c507 teaching panel 18 (original).](../assets/figures/ml_fig_c507_18.png)
*Figure — Blue-green cutover path c507. Synthetic teaching geometry—not a causal claim.*

![c508 teaching panel 18 (original).](../assets/figures/ml_fig_c508_18.png)
*Figure — Shadow traffic path c508. Synthetic teaching geometry—not a causal claim.*

![c509 teaching panel 18 (original).](../assets/figures/ml_fig_c509_18.png)
*Figure — Error budget burn path c509. Synthetic teaching geometry—not a causal claim.*

![c510 teaching panel 18 (original).](../assets/figures/ml_fig_c510_18.png)
*Figure — SLO multi-window path c510. Synthetic teaching geometry—not a causal claim.*

![c511 teaching panel 18 (original).](../assets/figures/ml_fig_c511_18.png)
*Figure — Alert noise burn-down c511. Synthetic teaching geometry—not a causal claim.*

![c512 teaching panel 18 (original).](../assets/figures/ml_fig_c512_18.png)
*Figure — On-call toil burn-down c512. Synthetic teaching geometry—not a causal claim.*

![c513 teaching panel 18 (original).](../assets/figures/ml_fig_c513_18.png)
*Figure — MTTR residual path c513. Synthetic teaching geometry—not a causal claim.*

![c514 teaching panel 18 (original).](../assets/figures/ml_fig_c514_18.png)
*Figure — Progressive delivery path c514. Synthetic teaching geometry—not a causal claim.*

![c515 teaching panel 18 (original).](../assets/figures/ml_fig_c515_18.png)
*Figure — Feature flag ramp path c515. Synthetic teaching geometry—not a causal claim.*

![c516 teaching panel 18 (original).](../assets/figures/ml_fig_c516_18.png)
*Figure — Model card gate path c516. Synthetic teaching geometry—not a causal claim.*

![c517 teaching panel 18 (original).](../assets/figures/ml_fig_c517_18.png)
*Figure — Eval suite residual path c517. Synthetic teaching geometry—not a causal claim.*

![c518 teaching panel 18 (original).](../assets/figures/ml_fig_c518_18.png)
*Figure — Champion challenger path c518. Synthetic teaching geometry—not a causal claim.*
