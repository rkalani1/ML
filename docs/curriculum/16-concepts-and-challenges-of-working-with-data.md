# Chapter 16. Concepts and Challenges of Working with Data


![16 Leakage Timeline](../assets/figures/16_leakage_timeline.png)


## Opening

Missing NIHSS, duplicated MRNs, label drift after a documentation change—this is the real curriculum of clinical ML. Architecture papers rarely fail first; data pipelines do.


![Data challenges often reduce to time and shift (original).](../assets/figures/ml_fig_leakage_timeline.png)

*Data challenges often reduce to time and shift (original).*

![Distribution shift between cohorts (original).](../assets/figures/ml_fig_site_shift.png)

*Distribution shift between cohorts (original).*
## Learning Objectives

Map problem complexity and the clinical data lifecycle from cohort design through monitoring.

Apply stratified/cluster sampling and explain Monte Carlo, Metropolis-Hastings, Gibbs, and importance sampling.

Identify noise types and classical filters (Butterworth, Wiener, Kalman) alongside ML denoisers.

Handle imbalance and augmentation across tabular, image, text, and signal data; impute and interpolate missing values responsibly.

Detect anomalies with isolation forest, one-class SVM, LOF, and RANSAC; monitor drift and cold start.

Compute rater agreement (percent, Cohen, Fleiss, Krippendorff) with a worked kappa example.

Utilize LLMs via RAG and prompt patterns (zero/few-shot, CoT, self-consistency, ToT) under governance.

Report fairness metrics/mitigations and use SHAP/LIME for debugging with causal humility; work site-drift PPV numbers.

Institute governance for clinical ML: datasheet discipline, reproducibility, train-serve parity, and multi-site harmonization.

## 16.1 Why Data—Not Algorithms—Usually Dominate Outcomes

Across this textbook we have studied models: linear predictors, trees, neural networks, reinforcement learners, compressed edge models, and graph algorithms. In clinical and epidemiologic practice, the binding constraint is almost always the data. Labels are delayed, partial, and disputed; cohorts are selected by who arrives where with which insurance; scanners and coding systems change mid-study; the distribution at deployment is not the distribution in last year’s export. A modest model on carefully defined, well-labeled, prospectively monitored data routinely outperforms a fashionable architecture on a silent mess.

Treat data work as scientific measurement, not janitorial prelude. Every extract embodies inclusion criteria, timestamps, join keys, and operational quirks (a code that means “rule out” in one clinic and confirmed disease in another). Document those choices with the same care used for statistical estimators. This capstone chapter organizes problem complexity, sampling, noise, imbalance, missingness, anomalies, drift, rater agreement, LLM utilization, fairness, and interpretability—with worked numerical examples you can reuse in protocols.

## 16.2 Problem Complexity

Not every analytic task has the same inherent difficulty. Problem complexity combines statistical complexity (how much data are needed to estimate a target under noise), computational complexity (time and memory as functions of sample size and dimension), and scientific complexity (causal vs predictive goals, non-stationarity, feedback loops). A linear risk score with a clear index time and a frequent binary endpoint is low complexity relative to estimating individualized dynamic treatment regimes from confounded EHR streams.

Practical diagnostics of complexity include: effective sample size after rare outcomes and missingness; dimensionality relative to n; label noise ceilings (inter-rater kappa); shift between sites; and whether the decision requires counterfactuals. Match method ambition to complexity: deep sequence models on n=80 with noisy labels are usually the wrong response. Conversely, when n is huge and the signal is perceptual (imaging), expressive models may be justified if validation is rigorous.

Computational complexity still matters at scale: naive pairwise distances are O(n^2); full attention is O(N^2) in sequence length; exact betweenness is expensive on large care networks. Use the earlier chapters’ efficient algorithms when the data volume demands it, but never let big-O cleverness substitute for a well-posed cohort definition.

## 16.3 Sampling from Complex Data: Stratified, Cluster, Monte Carlo, and MCMC

Sampling determines who enters the analysis and which uncertainties we can claim to represent. Stratified sampling draws within predefined strata (site, age band, stroke subtype) to guarantee representation and reduce variance of stratified estimators. Cluster sampling draws groups (hospitals, households) then individuals within clusters—cheaper operationally but requires cluster-robust variance; treating clustered draws as i.i.d. understates uncertainty.

![16.1: Sampling a target distribution with Markov Chain Monte Carlo. Left: a Metropolis random-walk chain for a standard-normal](../assets/figures/ml_concept_16.1_189fb0a4.png)

*Figure 16.1 — original teaching graphic.*

Monte Carlo (MC) sampling estimates expectations by drawing independent samples from a distribution p and averaging f(x). The law of large numbers justifies convergence; variance of the estimator is Var(f(X))/n. Crude MC fails when p is hard to sample or when rare regions dominate f.

Markov Chain Monte Carlo (MCMC) constructs a Markov chain whose stationary distribution is the target posterior p(theta|data) when direct sampling is impossible. After burn-in, dependent samples approximate posterior expectations.

Metropolis-Hastings (MH). Propose theta’ ~ q(.|theta); accept with probability min(1, [p(theta’) q(theta|theta’)] / [p(theta) q(theta’|theta)]). Symmetric proposals (random walk) cancel q terms. Tune proposal scale for intermediate acceptance rates (rule of thumb often near 0.2-0.4 in moderate dimensions).

Gibbs sampling. When conditional distributions p(theta_j | theta_{-j}, data) are available, sample each coordinate from its full conditional. Gibbs is a special MH case with acceptance probability 1. Hierarchical Bayesian models in epidemiology often use Gibbs or Hamiltonian Monte Carlo (a more efficient advanced method).

Importance sampling. Draw from a proposal q, reweight by w = p/q, and form weighted averages. Effective sample size ESS = (sum w)^2 / sum w^2 collapses when q poorly matches p—the same pathology as off-policy RL importance ratios. Use for rare-event simulation and as a teaching bridge to sequential Monte Carlo.

Clinical sampling ethics: convenience samples of tertiary EHR data are not simple random samples of the disease population; report design and use design-based or model-based adjustments deliberately.

```
import math, random

# Metropolis-Hastings for a 1D standard-normal target (educational)
def mh_normal(n=5000, step=0.5, seed=0):
 random.seed(seed)
 x, xs = 0.0, []
 for _ in range(n):
 prop = x + random.gauss(0, step) # symmetric random-walk proposal
 log_alpha = -0.5 * (prop*prop - x*x) # log target-density ratio; q cancels
 if log_alpha >= 0 or random.random() < math.exp(log_alpha):
 x = prop # accept with prob min(1, exp(log_alpha))
 xs.append(x)
 return xs
```

## 16.4 Noise Types and Noise Reduction

Noise corrupts measurements, labels, and images. Types by statistical structure: white noise has flat spectrum and uncorrelated samples; Gaussian noise is additive normal (common model for sensor error); Poisson noise arises in photon-limited imaging; salt-and-pepper noise replaces random pixels with extremes; speckle noise is multiplicative in ultrasound and some MRI contexts; gradient noise in optimization is the stochasticity of minibatch gradients (sometimes intentionally injected).

![Gradient noise: mini-batch SGD jitter versus smooth batch GD on a synthetic loss valley (original).](../assets/figures/ml_fig_gradient_noise.png)

*Figure — Optimization noise is not measurement noise. Mini-batch gradient estimates inject controlled stochasticity into parameter updates; batch size and learning rate set the amplitude. Treat it as a training hyperparameter, not as sensor artifact to filter out of the data.*

Noise reduction. Machine learning denoisers learn mappings from noisy to clean examples (or self-supervised variants). Classical signal filters remain essential baselines and real-time tools:

Butterworth filter. A low-pass (or band-pass) design with maximally flat passband; order controls the sharpness of cutoff. Used to remove high-frequency interference from EEG/ECG while preserving signal band—always inspect phase effects and filter both directions (filtfilt) when offline.

Wiener filter. Optimal linear filter under stationary signal and noise spectra: minimizes mean squared error using power spectral densities. Adaptive Wiener variants estimate local statistics in images.

Kalman filter. Recursive Bayesian estimator for linear-Gaussian state-space models: predict step propagates state mean/covariance; update step assimilates the next measurement. Extended/unscented Kalman and particle filters handle nonlinearities. For tracking physiologic states or fusing multi-sensor streams, Kalman thinking—explicit process noise vs measurement noise—beats ad hoc smoothing.

Always separate measurement noise from biological variability and from label noise; each demands different remedies.

White/Gaussian/Poisson: probabilistic sensor models.

Salt-and-pepper and speckle: structured imaging artifacts.

Butterworth: frequency-domain band limiting.

Wiener: MSE-optimal linear filter under stationarity.

Kalman: sequential state estimation with process vs measurement noise.

## 16.5 Imbalanced Data and Augmentation

Class imbalance is the norm in serious neurologic outcomes: large-vessel occlusion among all ED headache presentations, aneurysm rupture among surveillance cohorts, rare adverse drug events. Metrics and training must adapt: use precision-recall, average precision, calibrated probabilities, and decision-curve analysis—not accuracy alone. Algorithmic responses include class weights, resampling (undersample majority, oversample minority), and threshold tuning on validation data that matches deployment prevalence.

![16.2: Correcting class imbalance with SMOTE-style oversampling. Left: a majority class (260 points, indigo) surrounds a rare m](../assets/figures/ml_concept_16.2_77c43ad6.png)

*Figure 16.2 — original teaching graphic.*

Data augmentation expands effective sample size by transforming examples without changing labels (when transformations preserve semantics).

Tabular: careful jitter of continuous features within clinical plausibility; SMOTE-style synthetic minorities (watch leakage and unrealistic combinations); mixup (training on convex combinations of two examples and their labels) on standardized features. Avoid random noise that creates impossible vitals.

Image: flips, rotations, elastic deformations, intensity shifts, simulated bias fields—matched to acquisition physics. For stroke CT, aggressive geometric warps may break anatomy; prefer validated medical augmentation libraries and radiologist spot checks.

Text: synonym replacement, back-translation, controlled paraphrasing; for clinical text, protect negation and dosages from corruption. LLM-based paraphrase needs PHI governance.

Signal/time series: time warping, channel dropout, additive noise at measured SNR, window slicing. Preserve temporal causality for forecasting tasks (no future peeking).

Augmentation is not a substitute for external data or better labels; it regularizes within the support of observed variation.

## 16.6 Reconstructing Missing Data: Imputation and Interpolation

Missingness mechanisms: MCAR (missing completely at random), MAR (missing at random given observed variables), MNAR (depends on unobserved values). Clinical data are saturated with MNAR: NIHSS more complete when stroke codes activate; advanced labs missing in comfort-care pathways; 90-day mRS missing when disabled patients are lost to follow-up.

![16.3: Missingness mechanisms and imputation in a patient × feature matrix. (a) Cells are shaded by value; dashed rose cells ma](../assets/figures/ml_concept_16.3_26dca7e7.png)

*Figure 16.3 — original teaching graphic.*

Imputation methods. Mean/median/mode: simple baselines; distort variances and correlations. k-NN imputation: fill from similar rows; needs a distance metric and careful scaling. Hot-deck: fill from a random similar donor record; cold-deck: from an external reference source. Regression imputation: predict missing feature from others; can be iterated (MICE). Multiple imputation propagates uncertainty for epidemiologic estimands. Griffin-Lim algorithm reconstructs signals (classic for spectrograms) from magnitude constraints via iterative projection—example of domain-specific reconstruction. Customized clinical rules (carry-forward last vitals within a short window only) encode workflow knowledge but can leak if windows cross index time improperly.

Interpolation methods (often for ordered domains). Linear and bilinear interpolation fill between grids (images, spatial maps). Polynomial interpolation fits higher-order curves—can overshoot (Runge). Splines use piecewise polynomials with smoothness constraints—workhorses for curves. Kriging (Gaussian process regression in geostatistics) provides spatial interpolation with uncertainty, used in environmental epi and some imaging contexts. Radial basis function interpolation (RBFI) reconstructs surfaces from scattered points using kernels centered at data locations.

Never impute the label with methods that leak outcome information. For prediction, compare strategies under realistic deployment missingness, not only on artificially complete rows. Sometimes the missingness indicator itself is predictive (a clinical decision not to measure)—interpret fairness implications carefully.

## 16.7 Anomaly and Outlier Detection

Anomalies are rare points that differ from a notion of normal. In clinical data they may be errors (wrong units), rare diseases, or adversarial inputs. Unsupervised methods learn normal structure without anomaly labels.

Isolation Forest (iForest). Randomly partitions features; anomalies are isolated in fewer splits on average (shorter path lengths in isolation trees). Scalable and effective for high-dimensional tabular outliers.

One-Class SVM. Learns a boundary around training data in a kernel feature space (or a half-space in that space), flagging points outside. Sensitive to kernel and nu parameters; can be heavy for large n.

Local Outlier Factor (LOF). Compares local density of a point to densities of neighbors; points in sparser regions than their neighbors score as outliers. Captures local anomalies missed by global methods.

RANSAC (Random Sample Consensus). Robust model fitting: repeatedly sample a minimal subset, fit a model (e.g., line, transformation), count inliers within a threshold, keep the model with most inliers. Classic for regression with contamination and for imaging registration with mismatched keypoints. Worked intuition: with 30% corrupted points, least squares collapses; RANSAC can still recover the dominant linear structure if enough clean minimal samples exist.

Use anomaly detectors as data-quality tools and as rare-event screens, not as automatic diagnoses. Tune alert rates to human review capacity.

```
# RANSAC line fit sketch (2D)
import random

def ransac_line(points, n_iter=200, thresh=1.0):
 best_inliers, best_model = [], None
 for _ in range(n_iter):
 (x1, y1), (x2, y2) = random.sample(points, 2)
 if x1 == x2:
 continue
 m = (y2 - y1) / (x2 - x1)
 b = y1 - m * x1
 inliers = [(x, y) for x, y in points if abs(y - (m * x + b)) <= thresh]
 if len(inliers) > len(best_inliers):
 best_inliers, best_model = inliers, (m, b)
 return best_model, len(best_inliers)
```

## 16.8 Drift, Concept Change, and the Cold Start Problem

Dataset shift occurs when training and target distributions differ. Covariate shift changes P(X); label shift changes P(Y); concept drift changes P(Y|X)—the relationship itself. Scanner upgrades, ICD-9 to ICD-10 transitions, new order sets, earlier thrombectomy eligibility, and tele-stroke routing all induce shift. Model performance can collapse silently if only average AUC on last year’s holdout is tracked.

![Temporal CV: random fold interleaving vs forward-chaining splits (original).](../assets/figures/ml_fig_temporal_cv.png)

*Figure — Time is not exchangeable. **Left:** shuffled K-fold paints future encounters into training folds—optimistic leakage on longitudinal EHR. **Right:** forward-chaining keeps train in the past and validation/test in the future. Prefer temporal or site-blocked splits whenever care pathways evolve; random CV estimates are not external validation and not causal proof.*

![Sinkhorn optimal transport: cost matrix and entropy-regularized coupling (original).](../assets/figures/ml_fig_sinkhorn_ot.png)

*Figure — Aligning distributions. **Left:** pairwise cost C between source and target bins. **Right:** Sinkhorn coupling P moves mass with entropy regularization. OT appears in domain adaptation and batch correction; the coupling is a transport plan—not a causal map between patients or sites.*

![1-Wasserstein shift via quantile functions (synthetic; original).](../assets/figures/ml_fig_wasserstein1.png)

*Figure — Distribution distance. **Left:** train vs serve histograms. **Right:** quantile functions; mean vertical gap approximates W₁. Useful for drift alarms—not a causal site effect size.*

![16.4: Monitoring for dataset shift across deployment windows. (a) Live AUROC is tracked monthly; concept drift (a change in P(](../assets/figures/ml_concept_16.4_6c511648.png)

*Figure 16.4 — original teaching graphic.*

![Deployment monitoring: score PSI and live AUROC floor with rollback (synthetic; original).](../assets/figures/ml_fig_drift_monitor.png)

*Figure — Monitoring is ops science, not a dashboard ornament. **Left:** reference vs current predicted-score histograms; population stability index PSI = Σᵢ (cᵢ − rᵢ)·ln(cᵢ/rᵢ) over bins quantifies mass shift (teaching rule of thumb: PSI ≳ 0.2 is a material alarm). Input and score drift often fire **before** labels arrive. **Right:** synthetic monthly live AUROC with a pre-set performance floor; when the metric breaches the floor, freeze thresholds, investigate train–serve skew / case-mix / scanner change, and execute a versioned rollback—do not silent-auto-retrain. Drift detection supports safe prediction service; it does not by itself prove causal pathways or authorize new treatment claims.*

Tackling drift: monitor input quantiles, embedding distances, prediction-score distributions, and calibrated outcome rates as labels arrive. Two common alarms are the population stability index—PSI = Σᵢ (cᵢ − rᵢ)·ln(cᵢ / rᵢ) summed over score bins, comparing current bin proportions cᵢ against a reference rᵢ, with PSI above ≈0.2 flagging a material shift—and, for high-dimensional imaging, the maximum mean discrepancy (MMD), a kernel distance between the reference and current embedding distributions. Retrain under version control with prospective evaluation, and prefer site-aware models when the multi-site case mix changes. Do not auto-retrain without governance—new inequitable workflows can be baked into “updated” models.

Cold start problem. New users, new hospitals, new devices, or new drugs lack history for personalized or site-adapted models. Mitigations: content-based features that do not need history; transfer from similar sites; hierarchical Bayesian pooling; explore under bandit constraints when ethical; collect minimum viable local labeled sets before trusting local fine-tunes. Cold start is acute when an MSU joins a network with different demographics and imaging vendors.

Worked site-drift PPV numbers. Suppose a large-vessel occlusion alert model uses a fixed score threshold chosen at Site A where prevalence p_A = 0.20 among alerted patients, sensitivity Se=0.90, specificity Sp=0.80. Positive predictive value PPV = Se*p / (Se*p + (1-Sp)*(1-p)).

At Site A: PPV_A = 0.90*0.20 / (0.90*0.20 + 0.20*0.80) = 0.18 / (0.18+0.16) = 0.18/0.34 ≈ 0.529 (52.9%).

At Site B with lower prevalence p_B = 0.05 among those scored (broader alerting), same Se/Sp: PPV_B = 0.90*0.05 / (0.90*0.05 + 0.20*0.95) = 0.045 / (0.045+0.19) = 0.045/0.235 ≈ 0.191 (19.1%).

The same model and threshold yields dramatically different clinical burden of false positives. If concept drift also reduces Sp to 0.70 at Site B (different scanners/artifacts), PPV_B’ = 0.045 / (0.045 + 0.30*0.95) = 0.045/0.330 ≈ 0.136 (13.6%). Always re-estimate calibration and PPV at each site; do not export thresholds blindly.

```
def ppv(se, sp, p):
 return se * p / (se * p + (1 - sp) * (1 - p))

print(round(ppv(0.90, 0.80, 0.20), 3)) # 0.529 (Site A)
print(round(ppv(0.90, 0.80, 0.05), 3)) # 0.191 (Site B, lower prevalence)
print(round(ppv(0.90, 0.70, 0.05), 3)) # 0.136 (Site B, specificity also drops)
```

## 16.9 Rater Agreement Methods

Labels in EHR and imaging research are rarely pure gold. Even expert review is noisy: two vascular neurologists may differ on TIA versus minor stroke or hemorrhagic transformation grades.

![16.5: Cohen's κ adjusts rater agreement for chance. Two reviewers independently label n = 100 non-contrast head CTs for intrac](../assets/figures/ml_concept_16.5_daaa898c.png)

*Figure 16.5 — original teaching graphic.*

Percentage agreement. Simply the fraction of items with identical ratings. Easy to communicate but inflated by chance when one class dominates.

Cohen’s κ. Chance-adjusted agreement for two raters on categorical labels: κ = (p_o − p_e) / (1 − p_e), where p_o is observed agreement and p_e is agreement expected from the marginal rates. Weighted κ handles ordinal scales (e.g., mRS) by penalizing distant disagreements more.

Worked Cohen’s κ. Two reviewers (A and B) independently label n = 100 non-contrast head CTs as ICH present or absent, giving this 2×2 agreement table:

| Rater A / Rater B | B present | B absent | A total |
| --- | --- | --- | --- |
| A present | 40 | 10 | 50 |
| A absent | 5 | 45 | 50 |
| B total | 45 | 55 | 100 |

The diagonal cells (40 both-present, 45 both-absent) are agreement. Observed agreement p_o = (40 + 45)/100 = 0.85. Expected (chance) agreement multiplies the two raters’ marginal proportions within each category and sums them: by chance both call “present” with probability (50/100)·(45/100) = 0.225, and both call “absent” with (50/100)·(55/100) = 0.275, so p_e = 0.225 + 0.275 = 0.50. Therefore

κ = (p_o − p_e)/(1 − p_e) = (0.85 − 0.50)/(1 − 0.50) = 0.35/0.50 = 0.70,

“substantial” on the usual verbal scale (0.61–0.80) but far from perfect: 85% raw agreement falls to 70% once the 50% agreement expected from the marginals alone is removed. This is a ceiling argument—if a model reports AUC 0.93 against these same labels, part of the residual error is irreducible reviewer noise, and discrimination claimed above the label’s reliability ceiling deserves suspicion.

Fleiss’ κ. Extends chance-adjusted agreement beyond two raters when a fixed number of raters scores each item—though not necessarily the same raters across items. Use it for multi-panel adjudication, say five neuroradiologists each grading every scan for hemorrhagic transformation. It assumes complete data and unordered categories.

Krippendorff’s α. The most general reliability coefficient: through a coincidence-matrix construction it tolerates any number of raters, missing ratings, and nominal, ordinal, interval, or ratio data, reducing to simpler coefficients as special cases. Reach for α when annotators overlap only partially (routine in crowd-labeled or multi-annotator ML datasets) or when the label is ordinal (mRS) so near-misses should count as partial agreement. When every item has exactly two complete nominal ratings, κ and α essentially coincide; the extra machinery earns its keep only under the raters or missingness that Cohen’s κ cannot represent.

Best practices: written label protocols, adjudication panels, periodic reliability audits, and modeling approaches robust to noise when appropriate.

```
# Cohen's kappa for 2x2 counts: a=both yes, b=A yes/B no, c=A no/B yes, d=both no
def cohen_kappa_2x2(a, b, c, d):
 n = a + b + c + d
 p_o = (a + d) / n
 p_yes = ((a + b) / n) * ((a + c) / n) # both say "yes" by chance
 p_no = ((c + d) / n) * ((b + d) / n) # both say "no" by chance
 p_e = p_yes + p_no
 return (p_o - p_e) / (1 - p_e)

print(round(cohen_kappa_2x2(40, 10, 5, 45), 3)) # 0.70
```

## 16.10 Utilizing LLMs: RAG and Prompt Engineering

Large language models (LLMs) enter clinical research workflows for drafting, coding assistance, literature triage, and retrieval—but they hallucinate, leak training biases, and mishandle PHI if misconfigured. Two pillars of safer utilization are retrieval-augmented generation and disciplined prompting.

![16.6: A retrieval-augmented generation (RAG) pipeline. Offline (bottom lane), source documents — guidelines and curated notes ](../assets/figures/ml_concept_16.6_3767a3ed.png)

*Figure 16.6 — original teaching graphic.*

Retrieval-Augmented Generation (RAG). Instead of relying only on parametric memory, retrieve external documents and condition generation on them. Pipeline: (1) index external data (guidelines, local protocols, curated notes) by chunking and embedding; (2) store vectors in an ANN index (e.g., HNSW from Chapter 15); (3) at query time retrieve top-k chunks; (4) prompt the LLM with the chunks plus the question; (5) cite sources for human verification. Fine-tuning the LLM on external data is an alternative or complement but is heavier to update; RAG keeps knowledge editable by re-indexing. Failure modes: retrieval misses the right chunk; contradictory chunks; prompt injection in retrieved text; outdated indexes. Evaluate RAG with answer faithfulness and retrieval recall, not chat fluency alone.

Prompt engineering. Zero-shot prompting asks the model to perform a task without examples. Few-shot prompting provides input-output exemplars in the prompt—often dramatically improving format adherence. Chain-of-Thought (CoT) encourages step-by-step reasoning (“think step by step”) for multi-hop problems; it can help arithmetic and logic but also elaborate confabulations—verify. Self-consistency samples multiple CoT rationales and majority-votes the final answer, improving robustness at higher compute cost. Tree-of-Thought (ToT) explores branching intermediate reasoning states with search/backtracking—more structured exploration than a single chain, useful for planning-style tasks, still not a substitute for validated clinical algorithms.

Governance: no PHI in commercial prompts without BAA and policy approval; prefer local or approved enterprise models for patient text; log prompts/outputs for audit; treat LLM output as draft under clinician responsibility.

RAG: editable external knowledge + citations; watch retrieval errors.

Zero/few-shot: control task format with exemplars.

CoT / self-consistency / ToT: structured reasoning trade compute for reliability.

Never confuse fluent prose with verified medical fact.

## 16.11 Fairness, Bias, and Transparency: Metrics, Mitigation, SHAP, and LIME

Fairness and transparency are not optional add-ons for clinical ML; they are part of validity. Bias can enter via selection, labels, features, optimization, and deployment thresholds. Concepts include disparate treatment vs disparate impact, group vs individual fairness, and the impossibility of satisfying all group parity metrics simultaneously when base rates differ (a formal tension to acknowledge, not a reason for apathy).

![16.7: A SHAP local explanation drawn as a waterfall. For one toy stroke case, the model's predicted probability of a poor 90-d](../assets/figures/ml_concept_16.7_40478392.png)

*Figure 16.7 — original teaching graphic.*

Fairness parity metrics (examples). Demographic parity: equal positive prediction rates across groups. Equalized odds: equal TPR and FPR across groups. Equal opportunity: equal TPR. Predictive parity: equal PPV. Calibration within groups: predicted probabilities match event rates inside each group. Different metrics encode different ethical choices; equalized odds and predictive parity can conflict. Report multiple metrics with clinical context (who is harmed by false negatives vs false positives in stroke triage?).

![Fairness–accuracy tradeoff under unequal group prevalence (synthetic; original).](../assets/figures/ml_fig_fairness_tradeoff.png)

*Figure — Fairness tradeoff. **Left:** sweeping a shared score threshold when groups differ in base rate (π₀=0.25, π₁=0.10) traces an accuracy vs |TPR gap| curve—the max-accuracy cut need not minimize disparity. **Right:** group-specific thresholds that match TPR (equal opportunity) can still leave an FPR gap. No single threshold erases all parity metrics when prevalences differ; document who is harmed by FN vs FP and prefer process fixes (equitable documentation) over cosmetic thresholding alone. Parity is not causation.*


![MCAR, MAR, and MNAR missingness mechanisms (synthetic scatter; original).](../assets/figures/ml_fig_missingness_mechanisms.png)

*Figure — Missingness mechanisms. Gold marks missing y under MCAR/MAR/MNAR cartoons. Wrong assumptions invent associations; imputation is not a causal identification strategy by itself.*


![Dual monitor: feature PSI and AUROC drift with alert floors (synthetic; original).](../assets/figures/ml_fig_drift_dual.png)

*Figure — Ops monitoring. Rising PSI and falling AUROC after week 16 trigger investigate/rollback paths. Alarms start inquiry—they do not by themselves assign causal blame to a site.*


![Feature legality timeline through decision and outcome (original).](../assets/figures/ml_fig_label_timeline.png)

*Figure — Only pre-decision features are legal for prediction at decision time. Post-decision fields in the feature set are leakage. Hygiene protects validity; it does not create causal identification.*


![Selection funnel from population into the analytic sample (original).](../assets/figures/ml_fig_selection_funnel.png)

*Figure — Who remains in complete-case modeling can distort associations. Selection is a validity threat; fixing it is not automatic causal identification.*


![Train/serve schema mismatch (original).](../assets/figures/ml_fig_schema_mismatch.png)

*Figure — Schema drift breaks pipelines; fix is engineering hygiene. Pred != cause without design.*


![Data unit-test gate funnel (original).](../assets/figures/ml_fig_unit_test_data.png)

*Figure — Tests catch leakage and schema bugs. Data unit-test gate funnel Pred != cause without design.*


![consent teaching panel (original).](../assets/figures/ml_fig_consent_scope.png)

*Figure — Teaching panel for consent. Pred != cause without design.*


![Cycle-34 densify scientific panel 18 (original).](../assets/figures/ml_fig_c34_17.png)

*Figure — Continuous densify panel 18. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 18 (original).](../assets/figures/ml_fig_c35_17.png)

*Figure — Continuous densify panel 18. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 18 (original).](../assets/figures/ml_fig_c36_17.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 18 (original).](../assets/figures/ml_fig_c37_17.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 18 (original).](../assets/figures/ml_fig_c38_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 18 (original).](../assets/figures/ml_fig_c39_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 18 (original).](../assets/figures/ml_fig_c40_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 18 (original).](../assets/figures/ml_fig_c41_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 18 (original).](../assets/figures/ml_fig_c42_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 18 (original).](../assets/figures/ml_fig_c43_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 18 (original).](../assets/figures/ml_fig_c44_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 18 (original).](../assets/figures/ml_fig_c45_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 18 (original).](../assets/figures/ml_fig_c46_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 18 (original).](../assets/figures/ml_fig_c47_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 18 (original).](../assets/figures/ml_fig_c48_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 18 (original).](../assets/figures/ml_fig_c49_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 18 (original).](../assets/figures/ml_fig_c50_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 18 (original).](../assets/figures/ml_fig_c51_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 18 (original).](../assets/figures/ml_fig_c52_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 18 (original).](../assets/figures/ml_fig_c53_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 18 (original).](../assets/figures/ml_fig_c54_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 18 (original).](../assets/figures/ml_fig_c55_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 18 (original).](../assets/figures/ml_fig_c56_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 18 (original).](../assets/figures/ml_fig_c57_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 18 (original).](../assets/figures/ml_fig_c58_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 18 (original).](../assets/figures/ml_fig_c59_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 18 (original).](../assets/figures/ml_fig_c60_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 18 (original).](../assets/figures/ml_fig_c61_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 18 (original).](../assets/figures/ml_fig_c62_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 18 (original).](../assets/figures/ml_fig_c63_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 18 (original).](../assets/figures/ml_fig_c64_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 18 (original).](../assets/figures/ml_fig_c65_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 18 (original).](../assets/figures/ml_fig_c66_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 18 (original).](../assets/figures/ml_fig_c67_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 18 (original).](../assets/figures/ml_fig_c68_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 18 (original).](../assets/figures/ml_fig_c69_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 18 (original).](../assets/figures/ml_fig_c70_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 18 (original).](../assets/figures/ml_fig_c71_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 18 (original).](../assets/figures/ml_fig_c72_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 18 (original).](../assets/figures/ml_fig_c73_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 18 (original).](../assets/figures/ml_fig_c74_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 18 (original).](../assets/figures/ml_fig_c75_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 18 (original).](../assets/figures/ml_fig_c76_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 18 (original).](../assets/figures/ml_fig_c77_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 18 (original).](../assets/figures/ml_fig_c78_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 18 (original).](../assets/figures/ml_fig_c79_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 18 (original).](../assets/figures/ml_fig_c80_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 18 (original).](../assets/figures/ml_fig_c81_17.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

Mitigation methods. Pre-processing: reweight or resample training data; remove or transform biased proxies carefully (proxy removal can fail). In-processing: fairness constraints or adversarial debiasing during training. Post-processing: group-specific thresholds. Process interventions: improve data collection for underrepresented groups, standardize imaging protocols, audit label quality by group. Technical patches without governance theater do not fix unjust care pathways.

Interpretability. Global methods summarize overall model behavior; local methods explain one prediction. SHAP (SHapley Additive exPlanations) borrows the Shapley value from cooperative game theory: it treats each feature as a “player” and distributes the credit for a single prediction so that the attributions sum exactly to the prediction, base value + Σ φᵢ = model output. A short list of axioms (efficiency, symmetry, null-player, additivity) makes this split of credit “fair” among features—which is fairness of accounting, not social fairness among patients. TreeSHAP computes the attributions exactly and cheaply for tree ensembles; KernelSHAP approximates them model-agnostically. Concretely, for one patient an LVO model might attribute a positive score to a high NIHSS and a hyperdense-vessel sign while a normal ASPECTS pulls the score down; a global SHAP summary (mean |φ| across patients) then ranks which features move predictions most. Three misuse traps recur. (1) Causal misreading: a large SHAP value for “prior thrombolysis” does not mean thrombolysis caused the outcome—the feature may be a marker of severity or of a care pathway, because SHAP explains the model, not the world. (2) Correlated features: when ASPECTS and NIHSS are collinear, credit is split in ways that depend on the background distribution and on whether interventional or conditional expectations are used, so a small SHAP value is not proof of irrelevance. (3) Leakage detection (the good use): if an in-hospital mortality model shows enormous SHAP for “ICU transfer” or a post-arrival procedure code, that is not insight but a post-outcome feature leaking the label—this debugging is SHAP’s strongest clinical use. Explaining a biased model does not debias it.

LIME (Local Interpretable Model-agnostic Explanations) takes a different route: to explain one instance it samples perturbations around that instance, weights them by proximity, records the black-box model’s outputs, and fits a simple sparse linear surrogate to that local neighborhood; the surrogate’s coefficients are the explanation. Concretely, for a note-based stroke-mimic classifier, LIME on one patient might highlight the tokens “hypoglycemia” and “witnessed seizure” as pushing the prediction toward mimic. Its meaning is strictly local and approximate: LIME says nothing about global behavior, and it is unstable—rerunning with a different random seed, kernel width, or superpixel segmentation (for images) can reorder the top features. Never present a single LIME map as ground truth; confirm the explanation is stable across seeds before trusting it. For imaging models, gradient saliency, Grad-CAM, and occlusion maps play the analogous local role and carry the same instability caveats. Both SHAP and LIME are debugging and audit instruments; neither licenses deploying an unfair or leaky model, and neither substitutes for a prospective causal study.

Transparency practice: model cards and datasheets spirit, documented intended use, versioned artifacts, and pathways for contesting automated recommendations.

## 16.12 Privacy, Integrity, Reproducibility, and Train-Serve Skew

Protected health information demands administrative, physical, and technical safeguards: minimum necessary fields, access control, encryption, de-identification when appropriate, and governance for multi-site sharing. Model artifacts can leak via membership inference; rare patients are most at risk.

Integrity failures include mis-joined IDs, clock skew, duplicated encounters, and data poisoning (injecting training points to manipulate parameters). Schema validation, provenance hashes, and anomaly detection on feature space are practical defenses.

Reproducibility requires fixed data versions, code commits, seeds where possible, and environment locks. The datasheet spirit is to travel a dataset with its provenance: who collected it and how, the sampling frame and inclusion criteria, label definitions and adjudication, known gaps, consent basis, and a maintenance plan—so a future reader can judge transportability. A stroke-imaging datasheet that discloses “posterior-circulation and wake-up strokes were excluded; scans are 3T Siemens only” tells a downstream user more about external validity than any single AUC. Model cards do the same for models: intended use, out-of-scope use, subgroup metrics, and failure modes.

Train-serve skew arises when the research feature pipeline differs from the production service, so the model silently sees different inputs at deployment than it saw in training. The failures are concrete in neurology. Research code may compute NIHSS from a curated structured field, while the live service parses it from free-text nursing notes with a different regex—systematically lower and more often missing—so the deployed score drifts from its trained behavior. A units mismatch is worse: if training glucose was in mmol/L but a post-upgrade production feed delivers mg/dL (about 18× larger), every downstream threshold is wrong until someone notices. Defenses: share one feature-engineering library between research and operations, add contract tests that pin units and encodings, and monitor live feature distributions against the training baseline so skew trips an alarm rather than a bad prediction.

## 16.13 Multi-Site Harmonization and Reporting Discipline

Multi-site studies amplify every data pathology: different EHRs, coding cultures, scanner fleets, and outcome ascertainment. Harmonization strategies include common data models, centralized data dictionaries, and imaging-protocol standardization. When features still carry a site signature, ComBat-style batch adjustment models each feature as a site-specific location-and-scale shift and empirical-Bayes shrinks those shifts toward a common mean before removing them. Concretely, cortical-thickness or DTI fractional-anisotropy values differ systematically between a 1.5T GE scanner and a 3T Siemens scanner; ComBat can strip that vendor/field-strength batch effect so a multiple-sclerosis-versus-control comparison is not confounded by which site imaged which patient. The essential caution: harmonize only after protecting the biological covariates of interest in the model, because when site is entangled with population (a specialty MS center scans sicker patients), naive batch correction will regress out real disease signal along with the scanner effect. Hierarchical models with random site intercepts are a gentler alternative that borrows strength across sites without hard-erasing site differences. Harmonize outcomes too: an mRS obtained by structured interview is not interchangeable with one abstracted from a chart note.

Reporting spirit (TRIPOD and related guidance): define the prediction goal, population, index time, features available at prediction, outcome definitions, handling of missing data, validation design (internal vs external), calibration, and clinical utility. Pre-register analysis plans when possible. Distinguish prediction from causal claims. A neurologist-epidemiologist who masters this chapter’s threats will reject more papers—and more vendor demos—for the right reasons.

## 16.14 Worked Capstone Scenario: Imbalance Meets Site Drift

Combine threads. You train a rare-outcome classifier (5% events) at Site A with balanced class weights, report PR-AUC 0.42 and PPV 0.53 at a high-sensitivity threshold on Site A validation (prevalence 20% in the enriched alerted population—see Section 16.8). You deploy at Site B without recalibration; prevalence among alerts is 5%, Sp slips due to scanner shift, PPV falls to ~14%. Label quality audit shows kappa 0.70 between chart reviewers, capping reliability. An isolation forest flags a burst of unit-conversion errors in glucose after an EHR update (data integrity). A RAG assistant for protocol Q&A is added but cites an outdated thrombolysis PDF until the index is refreshed (LLM ops). Fairness audit shows lower TPR in patients with limited English proficiency due to missing NIHSS documentation (MNAR missingness and care process bias).

Response plan: freeze deployment thresholds; re-estimate calibration at Site B; fix ETL unit bugs; re-index RAG; improve documentation equity interventions; retrain only under change control with prospective monitoring. This narrative is the real “full stack” of working with clinical data.

## 16.15 A Capstone Checklist for Neurologist-Epidemiologists

Before trusting a model-influenced decision pathway, verify: (1) question and decision context are explicit; (2) cohort and index time are defensible; (3) features are leakage-free; (4) labels have measured reliability; (5) missingness is profiled; (6) imbalance metrics match utility; (7) external or multi-site validation exists; (8) calibration and PPV are reported at target prevalence; (9) drift monitoring is funded; (10) fairness metrics and mitigations are documented; (11) explanations were used to debug, not to claim causality; (12) privacy and integrity controls are in place; (13) LLM components have retrieval evaluation and PHI policy; (14) train-serve parity is tested; (15) a human accountability path remains. Algorithms from earlier chapters are tools; this chapter is the professional standard for using them without self-deception.

Data definition beats model novelty for most clinical gains.

Measure label reliability (kappa/alpha) before celebrating AUC.

Export thresholds with prevalence and drift in mind.

Govern LLMs with RAG evaluation and PHI controls.

Fairness and interpretability are part of validation, not marketing.

## 16.16 Extended Worked Example: Kappa, Prevalence, and Threshold Choice

Three raters label 50 angiograms for LVO yes/no. Pairwise Cohen kappas are 0.62, 0.58, and 0.70; Fleiss’ kappa is 0.60. A model trained on majority-vote labels reports AUROC 0.88. Interpretation: disagreement is material; some “errors” are irresolvable ambiguity (borderline occlusions). Report human agreement beside model metrics. If deploying a high-sensitivity alert, compute PPV at each site’s alert prevalence as in Section 16.8 and choose thresholds per site to hold PPV above a minimum (e.g., 0.25) while monitoring missed true LVO rates via structured case review.

Add cold start: a new spoke hospital has only 40 historical alerts. Do not fine-tune a deep model on 40 cases. Prefer the shared model with site-specific threshold calibration using those 40 plus hierarchical shrinkage toward the network prior. Revisit after 200 labeled alerts.

## 16.17 MCMC Diagnostics and Importance Sampling Caution

After running Metropolis-Hastings or Gibbs, inspect trace plots, autocorrelation times, and Gelman-Rubin R-hat across chains. Poor mixing means reported posterior intervals are fiction. Reparameterize hierarchical models, use better proposals, or adopt Hamiltonian Monte Carlo when available. For importance sampling, always report ESS; if ESS is 12 after 10,000 draws, stop and redesign q. The parallel to offline RL is deliberate: reweighting historical data is fragile when the proposal/behavior distribution mismatches the target.

In epidemiologic Bayesian disease-mapping, MCMC is standard; in deep learning, stochastic optimization noise is not MCMC without careful theory. Do not claim “Bayesian uncertainty” from dropout samples alone without stating approximations.

## 16.18 Signal Filtering Worked Sketch

EEG sampled at 256 Hz contaminated by 60 Hz line noise and slow drift. A Butterworth band-pass 0.5-40 Hz (order 4), applied forward-backward, removes drift and high-frequency noise; a notch at 60 Hz may still be needed. Wiener filtering can further suppress stationary noise if spectra are estimated from quiet segments. Kalman filtering of a simple state (amplitude of a rhythm) can track nonstationary power for alerting—but clinical seizure detectors need validated algorithms, not ad hoc filters alone.

For blood-pressure streams with salt-and-pepper spikes from motion, median filters or RANSAC-like rejection of impossible deltas (e.g., 80 mmHg jump in one second) protect downstream models. Document every filter in the feature datasheet so train-serve skew does not reintroduce raw spikes in production.

## 16.19 Fairness Mitigation Walkthrough

Suppose TPR for LVO detection is 0.92 in one language group and 0.78 in another, driven partly by missing NIHSS text features for the second group. Mitigations in order of preferability: (1) improve equitable documentation and interpreter workflows (process); (2) add features available equally (imaging-only pathway); (3) reweight training or use fairness-constrained thresholds so that TPR gaps shrink without catastrophic PPV collapse; (4) document residual disparity and monitoring. SHAP may show “language” or “zip code” as drivers—treat as audit clues, not causal proof. LIME on individual false negatives can reveal missing severity features. Neither explanation method justifies deploying an unfair system without intervention.

## 16.20 RAG Evaluation Mini-Protocol

Build a 50-question gold set from stroke protocols. For each question, label relevant guideline chunks. Metrics: retrieval recall@5, answer faithfulness (does the generated answer stick to retrieved text?), and clinician usefulness scores. Test prompt variants: zero-shot vs three-shot; with/without CoT; self-consistency with five samples on the hardest 15 questions. Measure cost (tokens) and latency. Adversarial tests: contradictory retrieved PDFs; prompt injection in a malicious chunk; PHI leakage attempts. Only promote the RAG assistant after a target faithfulness threshold and an escalation path to human pharmacists/neurologists for dosing questions.

Tree-of-Thought may help multi-step logistics (“which center given time and symptoms”) but must call tools for real ETAs rather than inventing travel times. Tool use plus retrieval is more reliable than pure ToT prose.

## 16.21 Clinical and Epidemiologic Notes

Several threads in this chapter recur as clinical-epidemiologic principles worth stating on their own.

Know your denominator. EHR cohorts are prevalent-user, referral-filtered samples, not random draws from the disease population. A thrombectomy-outcome model trained at a comprehensive stroke center never observed the patients who died in transfer or were never referred; its estimates are conditional on surviving into that denominator. State the sampling frame explicitly and resist generalizing past it.

Guard the index time. The instant of prediction must precede every feature. “Last-known-well,” “door,” and “needle” are different clocks, and a feature measured after treatment (24-hour NIHSS, discharge disposition) leaks the outcome and inflates apparent discrimination. Leakage is the most common reason a dazzling internal AUC evaporates on external validation.

Respect the label ceiling. Reliability caps validity: when adjudicator κ is 0.70, an AUC reported near the reliability ceiling is partly fitting noise (Sections 16.9, 16.16). Report human agreement beside model metrics.

Prefer portable operating characteristics. Sensitivity and specificity travel across sites better than PPV and NPV, which move with prevalence. Recompute PPV at each site’s local prevalence before promising a false-alarm burden (Section 16.8).

Treat missingness as data. In neurology, missingness is rarely random: NIHSS is documented when a stroke code fires, advanced labs vanish on comfort-care pathways, and 90-day mRS is lost precisely for the most disabled patients (MNAR). The missingness indicator can encode acuity or a care disparity, so complete-case analysis both biases estimates and can hide inequity.

Separate prediction from causation. Nothing in a well-calibrated predictor licenses a causal or treatment claim; SHAP and LIME explain the model, not the disease. Causal questions need designs from the causal-inference toolkit, not attribution plots.

## 16.22 Connections

This capstone reuses the whole book. The sampling and MCMC methods (Metropolis-Hastings, Gibbs, Hamiltonian Monte Carlo) are the inference engines behind the Bayesian and probabilistic-modeling chapters; disease-mapping and hierarchical pooling for cold start are the same machinery applied to geography and to sites. The importance-sampling ESS collapse in Section 16.17 is the identical pathology as off-policy correction in reinforcement learning—reweighting data drawn from the wrong distribution is fragile everywhere it appears. RAG retrieval stands on the approximate-nearest-neighbor indexes (HNSW) from Chapter 15, and its embeddings are the representation-learning thread. Anomaly detection and drift monitoring are the operational face of the deployment and edge-model chapters, where compressed models meet nonstationary inputs. Big-O reasoning about O(n^2) distances and O(N^2) attention is the algorithmic-efficiency thread deciding what is computable at cohort scale. Fairness, calibration, and interpretability extend the evaluation chapter from average accuracy toward who is helped and who is harmed. The unifying move is the one this chapter argues throughout: the algorithm is rarely the bottleneck—the data, its provenance, and its governance are.


![c82 teaching panel 17 (original).](../assets/figures/ml_fig_c82_17.png)
*Figure — Missingness mechanisms MCAR / MAR / MNAR change recoverable structure. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 17 (original).](../assets/figures/ml_fig_c83_17.png)
*Figure — Cohort selection funnel—document every exclusion. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 17 (original).](../assets/figures/ml_fig_c84_17.png)
*Figure — Covariate, label, and concept shift taxonomy. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 17 (original).](../assets/figures/ml_fig_c85_17.png)
*Figure — Lifecycle risk heat: labeling and splitting are hot spots. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 17 (original).](../assets/figures/ml_fig_c86_17.png)
*Figure — Schema, unit/code, and semantic data drift modes. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 17 (original).](../assets/figures/ml_fig_c87_17.png)
*Figure — Label prevalence drift across calendar weeks. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 17 (original).](../assets/figures/ml_fig_c88_17.png)
*Figure — Label noise flips and decision boundary shift. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 17 (original).](../assets/figures/ml_fig_c89_17.png)
*Figure — Right-censoring timeline in outcomes. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 17 (original).](../assets/figures/ml_fig_c90_17.png)
*Figure — Batch/site effect in feature space. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 17 (original).](../assets/figures/ml_fig_c91_17.png)
*Figure — Train/serve skew checklist. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 17 (original).](../assets/figures/ml_fig_c92_17.png)
*Figure — Synthetic data privacy tradeoff. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 17 (original).](../assets/figures/ml_fig_c93_17.png)
*Figure — Differential privacy noise scale. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 17 (original).](../assets/figures/ml_fig_c94_17.png)
*Figure — Federated averaging rounds. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 17 (original).](../assets/figures/ml_fig_c95_17.png)
*Figure — k-anonymity quasi-identifiers. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 17 (original).](../assets/figures/ml_fig_c96_17.png)
*Figure — Membership inference risk. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 17 (original).](../assets/figures/ml_fig_c97_17.png)
*Figure — Concept drift detector alarm. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 17 (original).](../assets/figures/ml_fig_c98_17.png)
*Figure — Homomorphic encrypt latency tax. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 17 (original).](../assets/figures/ml_fig_c99_17.png)
*Figure — Reconstruction attack risk. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 17 (original).](../assets/figures/ml_fig_c100_17.png)
*Figure — Label delay in production. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 17 (original).](../assets/figures/ml_fig_c101_17.png)
*Figure — Secure aggregation federated. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 17 (original).](../assets/figures/ml_fig_c102_17.png)
*Figure — Model inversion attack. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 17 (original).](../assets/figures/ml_fig_c103_17.png)
*Figure — Schema evolution compatibility. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 17 (original).](../assets/figures/ml_fig_c104_17.png)
*Figure — DP-SGD clip-and-noise. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 17 (original).](../assets/figures/ml_fig_c105_17.png)
*Figure — Attribute inference risk. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 17 (original).](../assets/figures/ml_fig_c106_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 17 (original).](../assets/figures/ml_fig_c107_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 17 (original).](../assets/figures/ml_fig_c108_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 17 (original).](../assets/figures/ml_fig_c109_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 17 (original).](../assets/figures/ml_fig_c110_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 17 (original).](../assets/figures/ml_fig_c111_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 17 (original).](../assets/figures/ml_fig_c112_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 17 (original).](../assets/figures/ml_fig_c113_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 17 (original).](../assets/figures/ml_fig_c114_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 17 (original).](../assets/figures/ml_fig_c115_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 17 (original).](../assets/figures/ml_fig_c116_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 17 (original).](../assets/figures/ml_fig_c117_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 17 (original).](../assets/figures/ml_fig_c118_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 17 (original).](../assets/figures/ml_fig_c119_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 17 (original).](../assets/figures/ml_fig_c120_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 17 (original).](../assets/figures/ml_fig_c121_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 17 (original).](../assets/figures/ml_fig_c122_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 17 (original).](../assets/figures/ml_fig_c123_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 17 (original).](../assets/figures/ml_fig_c124_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 17 (original).](../assets/figures/ml_fig_c125_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 17 (original).](../assets/figures/ml_fig_c126_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 17 (original).](../assets/figures/ml_fig_c127_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 17 (original).](../assets/figures/ml_fig_c128_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 17 (original).](../assets/figures/ml_fig_c129_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 17 (original).](../assets/figures/ml_fig_c130_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 17 (original).](../assets/figures/ml_fig_c131_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 17 (original).](../assets/figures/ml_fig_c132_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 17 (original).](../assets/figures/ml_fig_c133_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 17 (original).](../assets/figures/ml_fig_c134_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 17 (original).](../assets/figures/ml_fig_c135_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 17 (original).](../assets/figures/ml_fig_c136_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 17 (original).](../assets/figures/ml_fig_c137_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 17 (original).](../assets/figures/ml_fig_c138_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 17 (original).](../assets/figures/ml_fig_c139_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 17 (original).](../assets/figures/ml_fig_c140_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 17 (original).](../assets/figures/ml_fig_c141_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 17 (original).](../assets/figures/ml_fig_c142_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 17 (original).](../assets/figures/ml_fig_c143_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 17 (original).](../assets/figures/ml_fig_c144_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 17 (original).](../assets/figures/ml_fig_c145_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 17 (original).](../assets/figures/ml_fig_c146_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 17 (original).](../assets/figures/ml_fig_c147_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 17 (original).](../assets/figures/ml_fig_c148_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 17 (original).](../assets/figures/ml_fig_c149_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 17 (original).](../assets/figures/ml_fig_c150_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 17 (original).](../assets/figures/ml_fig_c151_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 17 (original).](../assets/figures/ml_fig_c152_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 17 (original).](../assets/figures/ml_fig_c153_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 17 (original).](../assets/figures/ml_fig_c154_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 17 (original).](../assets/figures/ml_fig_c155_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 17 (original).](../assets/figures/ml_fig_c156_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 17 (original).](../assets/figures/ml_fig_c157_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 17 (original).](../assets/figures/ml_fig_c158_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 17 (original).](../assets/figures/ml_fig_c159_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 17 (original).](../assets/figures/ml_fig_c160_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 17 (original).](../assets/figures/ml_fig_c161_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 17 (original).](../assets/figures/ml_fig_c162_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 17 (original).](../assets/figures/ml_fig_c163_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 17 (original).](../assets/figures/ml_fig_c164_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 17 (original).](../assets/figures/ml_fig_c165_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 17 (original).](../assets/figures/ml_fig_c166_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 17 (original).](../assets/figures/ml_fig_c167_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 17 (original).](../assets/figures/ml_fig_c168_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 17 (original).](../assets/figures/ml_fig_c169_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 17 (original).](../assets/figures/ml_fig_c170_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 17 (original).](../assets/figures/ml_fig_c171_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 17 (original).](../assets/figures/ml_fig_c172_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 17 (original).](../assets/figures/ml_fig_c173_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 17 (original).](../assets/figures/ml_fig_c174_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 17 (original).](../assets/figures/ml_fig_c175_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 17 (original).](../assets/figures/ml_fig_c176_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 17 (original).](../assets/figures/ml_fig_c177_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 17 (original).](../assets/figures/ml_fig_c178_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 17 (original).](../assets/figures/ml_fig_c179_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 17 (original).](../assets/figures/ml_fig_c180_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 17 (original).](../assets/figures/ml_fig_c181_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 17 (original).](../assets/figures/ml_fig_c182_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 17 (original).](../assets/figures/ml_fig_c183_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 17 (original).](../assets/figures/ml_fig_c184_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 17 (original).](../assets/figures/ml_fig_c185_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 17 (original).](../assets/figures/ml_fig_c186_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 17 (original).](../assets/figures/ml_fig_c187_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 17 (original).](../assets/figures/ml_fig_c188_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 17 (original).](../assets/figures/ml_fig_c189_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 17 (original).](../assets/figures/ml_fig_c190_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 17 (original).](../assets/figures/ml_fig_c191_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 17 (original).](../assets/figures/ml_fig_c192_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 17 (original).](../assets/figures/ml_fig_c193_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 17 (original).](../assets/figures/ml_fig_c194_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 17 (original).](../assets/figures/ml_fig_c195_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 17 (original).](../assets/figures/ml_fig_c196_17.png)
*Figure — Prospective vs retrospective. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 17 (original).](../assets/figures/ml_fig_c197_17.png)
*Figure — Index date alignment. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 17 (original).](../assets/figures/ml_fig_c198_17.png)
*Figure — Immortal time bias. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 17 (original).](../assets/figures/ml_fig_c199_17.png)
*Figure — Competing event censor. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 17 (original).](../assets/figures/ml_fig_c200_17.png)
*Figure — Chart review gold labels. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 17 (original).](../assets/figures/ml_fig_c201_17.png)
*Figure — Selection collider bias path. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 17 (original).](../assets/figures/ml_fig_c202_17.png)
*Figure — Dataset shift taxonomy tiles. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 17 (original).](../assets/figures/ml_fig_c203_17.png)
*Figure — MCAR MAR MNAR mechanisms. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 17 (original).](../assets/figures/ml_fig_c204_17.png)
*Figure — Label shift prior mismatch. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 17 (original).](../assets/figures/ml_fig_c205_17.png)
*Figure — Concept drift in stream. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 17 (original).](../assets/figures/ml_fig_c206_17.png)
*Figure — Versioned code map freeze. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 17 (original).](../assets/figures/ml_fig_c207_17.png)
*Figure — Simpson paradox subgroup reverse. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 17 (original).](../assets/figures/ml_fig_c208_17.png)
*Figure — Multiplicity FDR vs FWER. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 17 (original).](../assets/figures/ml_fig_c209_17.png)
*Figure — Dataset shift type taxonomy. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 17 (original).](../assets/figures/ml_fig_c210_17.png)
*Figure — Feature timing leakage fence. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 17 (original).](../assets/figures/ml_fig_c211_17.png)
*Figure — Data unit test checklist. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 17 (original).](../assets/figures/ml_fig_c212_17.png)
*Figure — Synthetic data utility privacy. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 17 (original).](../assets/figures/ml_fig_c213_17.png)
*Figure — Batch effect site clusters. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 17 (original).](../assets/figures/ml_fig_c214_17.png)
*Figure — PSI feature drift bars. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 17 (original).](../assets/figures/ml_fig_c215_17.png)
*Figure — Label noise accuracy decay. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 17 (original).](../assets/figures/ml_fig_c216_17.png)
*Figure — Schema evolution migrate defaults. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 17 (original).](../assets/figures/ml_fig_c217_17.png)
*Figure — Imputation method bias bars. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 17 (original).](../assets/figures/ml_fig_c218_17.png)
*Figure — Data contract field checklist. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 17 (original).](../assets/figures/ml_fig_c219_17.png)
*Figure — Right-censoring time marks. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 17 (original).](../assets/figures/ml_fig_c220_17.png)
*Figure — Data freshness SLA lag. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 17 (original).](../assets/figures/ml_fig_c221_17.png)
*Figure — Schema evolution expand-contract. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 17 (original).](../assets/figures/ml_fig_c222_17.png)
*Figure — Data contract check SLI chain. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 17 (original).](../assets/figures/ml_fig_c223_17.png)
*Figure — PII redaction pipeline stages. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 17 (original).](../assets/figures/ml_fig_c224_17.png)
*Figure — Data lineage DAG stages. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 17 (original).](../assets/figures/ml_fig_c225_17.png)
*Figure — Feature freshness watermark SLA. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 17 (original).](../assets/figures/ml_fig_c226_17.png)
*Figure — Online feature store lookup. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 17 (original).](../assets/figures/ml_fig_c227_17.png)
*Figure — Population stability index bins. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 17 (original).](../assets/figures/ml_fig_c228_17.png)
*Figure — Canary deploy error watch. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 17 (original).](../assets/figures/ml_fig_c229_17.png)
*Figure — Data unit-test checklist. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 17 (original).](../assets/figures/ml_fig_c230_17.png)
*Figure — Data quality dimension scores. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 17 (original).](../assets/figures/ml_fig_c231_17.png)
*Figure — CDC replication lag trace. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 17 (original).](../assets/figures/ml_fig_c232_17.png)
*Figure — Feature flag metric step. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 17 (original).](../assets/figures/ml_fig_c233_17.png)
*Figure — Point-in-time feature join. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 17 (original).](../assets/figures/ml_fig_c234_17.png)
*Figure — Schema break rate trend. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 17 (original).](../assets/figures/ml_fig_c235_17.png)
*Figure — Training-serving skew path. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 17 (original).](../assets/figures/ml_fig_c236_17.png)
*Figure — Null-rate spike trend. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 17 (original).](../assets/figures/ml_fig_c237_17.png)
*Figure — Online offline store path. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 17 (original).](../assets/figures/ml_fig_c238_17.png)
*Figure — Freshness lag spike. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 17 (original).](../assets/figures/ml_fig_c239_17.png)
*Figure — Backfill materialize path. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 17 (original).](../assets/figures/ml_fig_c240_17.png)
*Figure — Duplicate rate spike. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 17 (original).](../assets/figures/ml_fig_c241_17.png)
*Figure — Feature store as-of path. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 17 (original).](../assets/figures/ml_fig_c242_17.png)
*Figure — Schema drift spike. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 17 (original).](../assets/figures/ml_fig_c243_17.png)
*Figure — Point-in-time join path. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 17 (original).](../assets/figures/ml_fig_c244_17.png)
*Figure — Late feature arrival spike. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 17 (original).](../assets/figures/ml_fig_c245_17.png)
*Figure — Entity resolution join path. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 17 (original).](../assets/figures/ml_fig_c246_17.png)
*Figure — Feature staleness spike. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 17 (original).](../assets/figures/ml_fig_c247_17.png)
*Figure — CDC change-data path. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 17 (original).](../assets/figures/ml_fig_c248_17.png)
*Figure — Null feature rate spike. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 17 (original).](../assets/figures/ml_fig_c249_17.png)
*Figure — Slowly changing dim path. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 17 (original).](../assets/figures/ml_fig_c250_17.png)
*Figure — Late join lag spike. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 17 (original).](../assets/figures/ml_fig_c251_17.png)
*Figure — Backfill watermark path. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 17 (original).](../assets/figures/ml_fig_c252_17.png)
*Figure — Schema version spike. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 17 (original).](../assets/figures/ml_fig_c253_17.png)
*Figure — PIT correct join path. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 17 (original).](../assets/figures/ml_fig_c254_17.png)
*Figure — Duplicate rate spike. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 17 (original).](../assets/figures/ml_fig_c255_17.png)
*Figure — Online offline store path. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 17 (original).](../assets/figures/ml_fig_c256_17.png)
*Figure — Freshness lag spike. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 17 (original).](../assets/figures/ml_fig_c257_17.png)
*Figure — Null rate spike c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 17 (original).](../assets/figures/ml_fig_c258_17.png)
*Figure — Duplicate spike path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 17 (original).](../assets/figures/ml_fig_c259_17.png)
*Figure — Label delay path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 17 (original).](../assets/figures/ml_fig_c260_17.png)
*Figure — PIT correctness path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 17 (original).](../assets/figures/ml_fig_c261_17.png)
*Figure — Backfill watermark path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 17 (original).](../assets/figures/ml_fig_c262_17.png)
*Figure — CDC stream path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 17 (original).](../assets/figures/ml_fig_c263_17.png)
*Figure — Entity resolve path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 17 (original).](../assets/figures/ml_fig_c264_17.png)
*Figure — Freshness SLO path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 17 (original).](../assets/figures/ml_fig_c265_17.png)
*Figure — Leakage audit path c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 17 (original).](../assets/figures/ml_fig_c266_17.png)
*Figure — Window aggregate path c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 17 (original).](../assets/figures/ml_fig_c267_17.png)
*Figure — Late arrival spike c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 17 (original).](../assets/figures/ml_fig_c268_17.png)
*Figure — Data contract path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 17 (original).](../assets/figures/ml_fig_c269_17.png)
*Figure — As-of join path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 17 (original).](../assets/figures/ml_fig_c270_17.png)
*Figure — Training-serving skew c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 17 (original).](../assets/figures/ml_fig_c271_17.png)
*Figure — Feature staleness path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 17 (original).](../assets/figures/ml_fig_c272_17.png)
*Figure — Schema drift spike c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 17 (original).](../assets/figures/ml_fig_c273_17.png)
*Figure — Null rate spike c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 17 (original).](../assets/figures/ml_fig_c274_17.png)
*Figure — Duplicate spike path c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 17 (original).](../assets/figures/ml_fig_c275_17.png)
*Figure — Label delay path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 17 (original).](../assets/figures/ml_fig_c276_17.png)
*Figure — PIT correctness path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 17 (original).](../assets/figures/ml_fig_c277_17.png)
*Figure — Backfill watermark path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 17 (original).](../assets/figures/ml_fig_c278_17.png)
*Figure — CDC stream path c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 17 (original).](../assets/figures/ml_fig_c279_17.png)
*Figure — Entity resolve path c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 17 (original).](../assets/figures/ml_fig_c280_17.png)
*Figure — Freshness SLO path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 17 (original).](../assets/figures/ml_fig_c281_17.png)
*Figure — Leakage audit path c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 17 (original).](../assets/figures/ml_fig_c282_17.png)
*Figure — Window aggregate path c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 17 (original).](../assets/figures/ml_fig_c283_17.png)
*Figure — Late arrival spike c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 17 (original).](../assets/figures/ml_fig_c284_17.png)
*Figure — Data contract path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 17 (original).](../assets/figures/ml_fig_c285_17.png)
*Figure — As-of join path c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 17 (original).](../assets/figures/ml_fig_c286_17.png)
*Figure — Training-serving skew c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 17 (original).](../assets/figures/ml_fig_c287_17.png)
*Figure — Feature staleness path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 17 (original).](../assets/figures/ml_fig_c288_17.png)
*Figure — Schema drift spike c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 17 (original).](../assets/figures/ml_fig_c289_17.png)
*Figure — Null rate spike c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 17 (original).](../assets/figures/ml_fig_c290_17.png)
*Figure — Duplicate spike path c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 17 (original).](../assets/figures/ml_fig_c291_17.png)
*Figure — Label delay path c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 17 (original).](../assets/figures/ml_fig_c292_17.png)
*Figure — PIT correctness path c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 17 (original).](../assets/figures/ml_fig_c293_17.png)
*Figure — Backfill watermark path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 17 (original).](../assets/figures/ml_fig_c294_17.png)
*Figure — CDC stream path c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 17 (original).](../assets/figures/ml_fig_c295_17.png)
*Figure — Entity resolve path c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 17 (original).](../assets/figures/ml_fig_c296_17.png)
*Figure — Freshness SLO path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 17 (original).](../assets/figures/ml_fig_c297_17.png)
*Figure — Leakage audit path c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 17 (original).](../assets/figures/ml_fig_c298_17.png)
*Figure — Window aggregate path c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 17 (original).](../assets/figures/ml_fig_c299_17.png)
*Figure — Late arrival spike c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 17 (original).](../assets/figures/ml_fig_c300_17.png)
*Figure — Data contract path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 17 (original).](../assets/figures/ml_fig_c301_17.png)
*Figure — As-of join path c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 17 (original).](../assets/figures/ml_fig_c302_17.png)
*Figure — Training-serving skew c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 17 (original).](../assets/figures/ml_fig_c303_17.png)
*Figure — Feature staleness path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 17 (original).](../assets/figures/ml_fig_c304_17.png)
*Figure — Schema drift spike c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 17 (original).](../assets/figures/ml_fig_c305_17.png)
*Figure — Null rate spike c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 17 (original).](../assets/figures/ml_fig_c306_17.png)
*Figure — Duplicate spike path c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 17 (original).](../assets/figures/ml_fig_c307_17.png)
*Figure — Label delay path c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 17 (original).](../assets/figures/ml_fig_c308_17.png)
*Figure — PIT correctness path c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 17 (original).](../assets/figures/ml_fig_c309_17.png)
*Figure — Backfill watermark path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 17 (original).](../assets/figures/ml_fig_c310_17.png)
*Figure — CDC stream path c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 17 (original).](../assets/figures/ml_fig_c311_17.png)
*Figure — Entity resolve path c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 17 (original).](../assets/figures/ml_fig_c312_17.png)
*Figure — Freshness SLO path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 17 (original).](../assets/figures/ml_fig_c313_17.png)
*Figure — Leakage audit path c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 17 (original).](../assets/figures/ml_fig_c314_17.png)
*Figure — Window aggregate path c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 17 (original).](../assets/figures/ml_fig_c315_17.png)
*Figure — Late arrival spike c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 17 (original).](../assets/figures/ml_fig_c316_17.png)
*Figure — Data contract path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 17 (original).](../assets/figures/ml_fig_c317_17.png)
*Figure — As-of join path c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 17 (original).](../assets/figures/ml_fig_c318_17.png)
*Figure — Training-serving skew c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 17 (original).](../assets/figures/ml_fig_c319_17.png)
*Figure — Feature staleness path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 17 (original).](../assets/figures/ml_fig_c320_17.png)
*Figure — Schema drift spike c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 17 (original).](../assets/figures/ml_fig_c321_17.png)
*Figure — Null rate spike c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 17 (original).](../assets/figures/ml_fig_c322_17.png)
*Figure — Duplicate spike path c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 17 (original).](../assets/figures/ml_fig_c323_17.png)
*Figure — Label delay path c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 17 (original).](../assets/figures/ml_fig_c324_17.png)
*Figure — PIT correctness path c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 17 (original).](../assets/figures/ml_fig_c325_17.png)
*Figure — Backfill watermark path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 17 (original).](../assets/figures/ml_fig_c326_17.png)
*Figure — CDC stream path c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 17 (original).](../assets/figures/ml_fig_c327_17.png)
*Figure — Entity resolve path c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 17 (original).](../assets/figures/ml_fig_c328_17.png)
*Figure — Freshness SLO path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 17 (original).](../assets/figures/ml_fig_c329_17.png)
*Figure — Leakage audit path c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 17 (original).](../assets/figures/ml_fig_c330_17.png)
*Figure — Window aggregate path c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 17 (original).](../assets/figures/ml_fig_c331_17.png)
*Figure — Late arrival spike c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 17 (original).](../assets/figures/ml_fig_c332_17.png)
*Figure — Data contract path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 17 (original).](../assets/figures/ml_fig_c333_17.png)
*Figure — As-of join path c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 17 (original).](../assets/figures/ml_fig_c334_17.png)
*Figure — Training-serving skew c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 17 (original).](../assets/figures/ml_fig_c335_17.png)
*Figure — Feature staleness path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 17 (original).](../assets/figures/ml_fig_c336_17.png)
*Figure — Schema drift spike c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 17 (original).](../assets/figures/ml_fig_c337_17.png)
*Figure — Null rate spike c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 17 (original).](../assets/figures/ml_fig_c338_17.png)
*Figure — Duplicate spike path c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 17 (original).](../assets/figures/ml_fig_c339_17.png)
*Figure — Label delay path c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 17 (original).](../assets/figures/ml_fig_c340_17.png)
*Figure — PIT correctness path c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 17 (original).](../assets/figures/ml_fig_c341_17.png)
*Figure — Backfill watermark path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 17 (original).](../assets/figures/ml_fig_c342_17.png)
*Figure — CDC stream path c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 17 (original).](../assets/figures/ml_fig_c343_17.png)
*Figure — Entity resolve path c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 17 (original).](../assets/figures/ml_fig_c344_17.png)
*Figure — Freshness SLO path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 17 (original).](../assets/figures/ml_fig_c345_17.png)
*Figure — Leakage audit path c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 17 (original).](../assets/figures/ml_fig_c346_17.png)
*Figure — Window aggregate path c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 17 (original).](../assets/figures/ml_fig_c347_17.png)
*Figure — Late arrival spike c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 17 (original).](../assets/figures/ml_fig_c348_17.png)
*Figure — Data contract path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 17 (original).](../assets/figures/ml_fig_c349_17.png)
*Figure — As-of join path c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 17 (original).](../assets/figures/ml_fig_c350_17.png)
*Figure — Training-serving skew c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 17 (original).](../assets/figures/ml_fig_c351_17.png)
*Figure — Feature staleness path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 17 (original).](../assets/figures/ml_fig_c352_17.png)
*Figure — Schema drift spike c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 17 (original).](../assets/figures/ml_fig_c353_17.png)
*Figure — Null rate spike c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 17 (original).](../assets/figures/ml_fig_c354_17.png)
*Figure — Duplicate spike path c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 17 (original).](../assets/figures/ml_fig_c355_17.png)
*Figure — Label delay path c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 17 (original).](../assets/figures/ml_fig_c356_17.png)
*Figure — PIT correctness path c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 17 (original).](../assets/figures/ml_fig_c357_17.png)
*Figure — Backfill watermark path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 17 (original).](../assets/figures/ml_fig_c358_17.png)
*Figure — CDC stream path c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 17 (original).](../assets/figures/ml_fig_c359_17.png)
*Figure — Entity resolve path c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 17 (original).](../assets/figures/ml_fig_c360_17.png)
*Figure — Freshness SLO path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 17 (original).](../assets/figures/ml_fig_c361_17.png)
*Figure — Leakage audit path c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 17 (original).](../assets/figures/ml_fig_c362_17.png)
*Figure — Window aggregate path c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 17 (original).](../assets/figures/ml_fig_c363_17.png)
*Figure — Late arrival spike c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 17 (original).](../assets/figures/ml_fig_c364_17.png)
*Figure — Data contract path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 17 (original).](../assets/figures/ml_fig_c365_17.png)
*Figure — As-of join path c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 17 (original).](../assets/figures/ml_fig_c366_17.png)
*Figure — Training-serving skew c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 17 (original).](../assets/figures/ml_fig_c367_17.png)
*Figure — Feature staleness path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 17 (original).](../assets/figures/ml_fig_c368_17.png)
*Figure — Schema drift spike c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 17 (original).](../assets/figures/ml_fig_c369_17.png)
*Figure — Null rate spike c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 17 (original).](../assets/figures/ml_fig_c370_17.png)
*Figure — Duplicate spike path c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 17 (original).](../assets/figures/ml_fig_c371_17.png)
*Figure — Label delay path c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 17 (original).](../assets/figures/ml_fig_c372_17.png)
*Figure — PIT correctness path c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 17 (original).](../assets/figures/ml_fig_c373_17.png)
*Figure — Backfill watermark path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 17 (original).](../assets/figures/ml_fig_c374_17.png)
*Figure — CDC stream path c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 17 (original).](../assets/figures/ml_fig_c375_17.png)
*Figure — Entity resolve path c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 17 (original).](../assets/figures/ml_fig_c376_17.png)
*Figure — Freshness SLO path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 17 (original).](../assets/figures/ml_fig_c377_17.png)
*Figure — Leakage audit path c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 17 (original).](../assets/figures/ml_fig_c378_17.png)
*Figure — Window aggregate path c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 17 (original).](../assets/figures/ml_fig_c379_17.png)
*Figure — Late arrival spike c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 17 (original).](../assets/figures/ml_fig_c380_17.png)
*Figure — Data contract path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 17 (original).](../assets/figures/ml_fig_c381_17.png)
*Figure — As-of join path c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 17 (original).](../assets/figures/ml_fig_c382_17.png)
*Figure — Training-serving skew c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 17 (original).](../assets/figures/ml_fig_c383_17.png)
*Figure — Feature staleness path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 17 (original).](../assets/figures/ml_fig_c384_17.png)
*Figure — Schema drift spike c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 17 (original).](../assets/figures/ml_fig_c385_17.png)
*Figure — Null rate spike c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 17 (original).](../assets/figures/ml_fig_c386_17.png)
*Figure — Duplicate spike path c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 17 (original).](../assets/figures/ml_fig_c387_17.png)
*Figure — Label delay path c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 17 (original).](../assets/figures/ml_fig_c388_17.png)
*Figure — PIT correctness path c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 17 (original).](../assets/figures/ml_fig_c389_17.png)
*Figure — Backfill watermark path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 17 (original).](../assets/figures/ml_fig_c390_17.png)
*Figure — CDC stream path c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 17 (original).](../assets/figures/ml_fig_c391_17.png)
*Figure — Entity resolve path c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 17 (original).](../assets/figures/ml_fig_c392_17.png)
*Figure — Freshness SLO path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 17 (original).](../assets/figures/ml_fig_c393_17.png)
*Figure — Leakage audit path c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 17 (original).](../assets/figures/ml_fig_c394_17.png)
*Figure — Window aggregate path c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 17 (original).](../assets/figures/ml_fig_c395_17.png)
*Figure — Late arrival spike c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 17 (original).](../assets/figures/ml_fig_c396_17.png)
*Figure — Data contract path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 17 (original).](../assets/figures/ml_fig_c397_17.png)
*Figure — As-of join path c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 17 (original).](../assets/figures/ml_fig_c398_17.png)
*Figure — Training-serving skew c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 17 (original).](../assets/figures/ml_fig_c399_17.png)
*Figure — Feature staleness path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 17 (original).](../assets/figures/ml_fig_c400_17.png)
*Figure — Schema drift spike c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 17 (original).](../assets/figures/ml_fig_c401_17.png)
*Figure — Null rate spike c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 17 (original).](../assets/figures/ml_fig_c402_17.png)
*Figure — Duplicate spike path c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 17 (original).](../assets/figures/ml_fig_c403_17.png)
*Figure — Label delay path c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 17 (original).](../assets/figures/ml_fig_c404_17.png)
*Figure — PIT correctness path c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 17 (original).](../assets/figures/ml_fig_c405_17.png)
*Figure — Backfill watermark path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 17 (original).](../assets/figures/ml_fig_c406_17.png)
*Figure — CDC stream path c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 17 (original).](../assets/figures/ml_fig_c407_17.png)
*Figure — Entity resolve path c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 17 (original).](../assets/figures/ml_fig_c408_17.png)
*Figure — Freshness SLO path c408. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Data quality and design dominate clinical ML outcomes. Problem complexity guides method choice. Sampling designs and MCMC methods (Metropolis-Hastings, Gibbs, importance sampling) support inference under complex distributions. Noise models and filters (Butterworth, Wiener, Kalman) clean signals; imbalance and modality-specific augmentation address skewed outcomes; imputation and interpolation repair missing structure without leakage. Anomaly methods (isolation forest, one-class SVM, LOF, RANSAC) protect integrity. Drift and cold start demand monitoring and recalibration—illustrated by site-level PPV collapse when prevalence and specificity change. Rater agreement metrics quantify label ceilings. LLMs require RAG and disciplined prompting under PHI governance. Fairness metrics, mitigation strategies, and SHAP/LIME explanations complete a professional validation stack for neurologist-epidemiologists.

## Practice and Reflection

(1) Recompute Cohen’s kappa if the off-diagonals double to 20 and 10 while keeping margins as feasible; interpret the change.

(2) Using Se=0.85, Sp=0.90, compute PPV at prevalence 0.30 and 0.02. What threshold strategy would you use across sites?

(3) Outline a Metropolis-Hastings sampler for a Beta-Binomial posterior; what proposal would you try first?

(4) Compare MAR vs MNAR for missing 90-day mRS. Which analyses remain credible under each assumption?

(5) Design tabular and imaging augmentations safe for ICH detection; list two unsafe transforms.

(6) When would you prefer RANSAC over least squares for a calibration curve fit?

(7) Write a few-shot prompt template for extracting NIHSS from note text; add a RAG constraint requiring guideline citations.

(8) Pick two fairness metrics for an LVO alert model and explain a clinical conflict between them.

(9) Use a SHAP summary plot narrative to diagnose potential label leakage from a post-arrival procedure code.

(10) Your LVO model was trained with NIHSS from a structured field, but production parses NIHSS from free-text notes. Describe two concrete ways train-serve skew could bias deployed predictions, and one contract test that would catch each.

(11) A multi-site DTI study spans 1.5T and 3T scanners, and the 3T site enrolls more severe patients. Explain why naive ComBat harmonization could erase real disease signal, and how covariate protection or a hierarchical site-intercept model avoids it.

(12) A mortality model shows a very large SHAP value for “ICU admission.” Argue why this is more likely evidence of label leakage than of a useful predictor, and state what you would check next.
