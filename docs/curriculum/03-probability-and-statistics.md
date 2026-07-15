# Chapter 3. Probability and Statistics


![03 Bayes Update](../assets/figures/03_bayes_update.png)


## Opening

A hemorrhage-risk model quotes sensitivity 0.92 and specificity 0.88. Without prevalence, Bayes, and calibration language, those numbers are theater. This chapter rebuilds the probability spine for clinical ML consumers.


![Predicted risk versus observed frequency (synthetic; original).](../assets/figures/ml_fig_calibration.png)

*Predicted risk versus observed frequency (synthetic; original).*
## Learning Objectives

Define random variables and distinguish continuous vs discrete types, dependent/independent/control variables, and independent trials.

Compute and interpret arithmetic, geometric, and harmonic means; median; mode; variance, SD, covariance; range, quartiles, and boxplots; degrees of freedom.

Apply joint and conditional probability and Bayes’ theorem with clinical base rates.

Relate PDF, PMF, and CDF and recognize Normal, Uniform, Beta, Dirichlet, Binomial, Bernoulli, Geometric, Poisson, Weibull, power-law/exponential/Zipf/Pareto, Chi-square, and Boltzmann distributions.

Use PP and QQ plots to assess distributional fit; define expectation and z-score normalization.

State the CLT and LLN, describe sampling bias, and construct basic confidence intervals.

Select conceptual families of hypothesis tests (t-tests, ANOVA/MANOVA/ANCOVA, chi-square, KS, Kruskal–Wallis, Mann–Whitney) and multiplicity corrections (Bonferroni, Tukey).

Report effect sizes (Cohen’s d, odds ratios), correlations, and information-theoretic quantities (entropy, IG, KL, cross-entropy, JS); sketch MLE and EM.

## Why Probability Underwrites Clinical Machine Learning

Every predictive model in neurology is a statement under uncertainty. Labels such as large-vessel occlusion on CTA, electrographic seizure on EEG, or 90-day modified Rankin Scale are noisy; future patients differ from historical registries; multiple models can fit the same finite sample. Probability supplies a coherent language for that uncertainty. Statistics supplies estimators, intervals, and tests that quantify how wrong finite-sample answers might be. Without this foundation, an AUC is a floating score, a calibrated probability is wishful labeling, and Bayesian updating—central to diagnosis and to many learning algorithms—cannot be stated precisely.

This chapter is applied and cumulative. We define variables and descriptive summaries; build joint and conditional probability through Bayes’ theorem with a fully worked LVO example; introduce PDFs, PMFs, CDFs and a catalog of distributions used in modeling; cover expectation, normalization, sampling laws, and confidence intervals; survey parametric and nonparametric tests with multiplicity control; and connect effect sizes, correlation, entropy-based divergences, maximum likelihood, and expectation–maximization. Clinical and epidemiologic notes keep the arithmetic tied to stroke and population reasoning.

## Concepts and Definitions: Variables

### Random (stochastic) variables; continuous and discrete

A random variable (RV) X is a numerical summary of an outcome of a random experiment—formally a measurable function from a sample space Ω to the real line. Discrete RVs take countable values (stroke counts per day, number of recurrences) and are described by a probability mass function p(x) = P(X = x). Continuous RVs take values on intervals (serum sodium, infarct volume on a continuum) and are described by a probability density function f such that P(a ≤ X ≤ b) = ∫_a^b f(x) dx; the density value f(x) is not itself a probability. Mixed types and distributions with atoms (point masses) appear when continuous labs are censored at detection limits.

Beyond the probabilistic continuous/discrete split, scientific data types include binary, nominal, ordinal, integer counts, and ratio-scale continuous measures. Legitimate operations depend on type: you may order mRS, but the gap from 1 to 2 is not the same clinical distance as from 5 to 6 in every decision context; you should not average nominal codes.

### Dependent, independent, and control variables

In experimental and regression language, independent variables (predictors, exposures, features) are inputs; the dependent variable is the response or outcome. Control variables are covariates included to adjust for confounding or precision—not ‘controls’ in the trial arm sense. In causal diagrams, the same clinical measurement might be a confounder, mediator, or collider depending on the estimand; calling a column ‘independent’ in software does not make it causally exogenous. ML feature lists should be designed with this vocabulary in mind when the goal is explanatory or policy-facing rather than pure prediction.

### Independent versus dependent trials

Independent trials mean that the outcome of one trial does not change the probability model of another: P(X_1, …, X_n) = Π_i P(X_i) under identical conditions (i.i.d. when the common distribution is shared). Dependent trials arise in clusters (patients in hospitals), repeated measures (days within ICU stay), and time series. Many textbook formulas (binomial variance, simple standard errors) assume independence. Violating it understates uncertainty—an endemic problem when hundreds of EEG windows from the same patient are treated as independent samples.

## Descriptive Statistics: Means, Spread, and Boxplots

### Arithmetic, geometric, and harmonic means

For positive or real observations x_1, …, x_n, the arithmetic mean is x̄ = (1/n) Σ_i x_i—the center of mass and the least-squares constant predictor. The geometric mean is (Π_i x_i)^{1/n} = exp((1/n) Σ log x_i) for x_i > 0, appropriate for average growth rates and log-normal-ish positive labs. The harmonic mean is n / Σ_i (1/x_i) for x_i > 0, natural for average rates when time is fixed and counts vary (and for F1 as a harmonic mean of precision and recall). Inequality: harmonic ≤ geometric ≤ arithmetic, with equality iff all values are equal.

### Median and mode

The median is a value that splits the ordered sample so that at least half the points lie on each side; for even n it is often the average of the two central order statistics. Medians resist extreme outliers better than means—prefer them for skewed length-of-stay and cost. The mode is the most frequent value (or a density peak in continuous data); multimodal clinical scores suggest mixture populations or coding artifacts.

### Variance, standard deviation, and covariance

Population variance is Var(X) = E[(X − μ)²]; standard deviation is its square root. The unbiased sample variance uses divisor n−1: s² = (1/(n−1)) Σ (x_i − x̄)². Covariance Cov(X,Y) = E[(X−μ_X)(Y−μ_Y)] measures joint linear co-movement; Corr(X,Y) = Cov(X,Y)/(σ_X σ_Y) standardizes to [−1,1]. Covariance matrices underpin multivariate Gaussians, PCA, and Mahalanobis distance.

### Range, quartiles, and boxplots

Range = max − min is simple and outlier-sensitive. Quartiles Q1, Q2 (median), Q3 divide the ordered sample into fourths; IQR = Q3 − Q1. Boxplots (whisker plots) draw the box from Q1 to Q3 with a median line and whiskers to fences such as [Q1 − 1.5·IQR, Q3 + 1.5·IQR], plotting exterior points individually. They compare groups quickly but hide multimodality—pair with violins or histograms when mixtures matter.

### Degrees of freedom

Degrees of freedom (df) count independent pieces of information available to estimate a parameter after constraints. Sample variance uses n−1 df because one df is spent estimating the mean. In regression, residual df are roughly n − p for p estimated coefficients. In chi-square tests, df depend on table size and estimated margins. Reporting df with test statistics is part of reproducible statistics.

## Probability: Joint, Conditional, and Bayes

Consider a sample space Ω of mutually exclusive, exhaustive outcomes. An event A is a measurable subset of Ω. Kolmogorov’s axioms: non-negativity P(A) ≥ 0; normalization P(Ω) = 1; countable additivity for disjoint unions. Then P(∅) = 0, P(A^c) = 1 − P(A), and the addition rule P(A ∪ B) = P(A) + P(B) − P(A ∩ B). Independence means P(A ∩ B) = P(A)P(B).

Joint probability P(A ∩ B) describes co-occurrence. Conditional probability P(A | B) = P(A ∩ B)/P(B) when P(B) > 0 restricts attention to the world where B is true. The law of total probability expands P(A) via a partition B_i: P(A) = Σ_i P(A | B_i) P(B_i). Bayes’ theorem reverses conditioning: P(H | E) = P(E | H) P(H) / P(E), with P(E) = Σ_j P(E | H_j) P(H_j) when hypotheses partition Ω.

### Worked numerical example: LVO screening

Suppose in an ED population of suspected acute ischemic stroke, P(LVO) = 0.20. A bedside scale has sensitivity P(+ | LVO) = 0.85 and specificity P(− | no LVO) = 0.70, so P(+ | no LVO) = 0.30. A patient screens positive. Then P(+) = 0.85·0.20 + 0.30·0.80 = 0.17 + 0.24 = 0.41, and P(LVO | +) = 0.17/0.41 ≈ 0.4146 ≈ 41.5%. Despite 85% sensitivity, PPV is only about 41.5% because false positives among the 80% without LVO dominate the marginal. At prevalence 0.05, the same sens/spec yield P(+) = 0.3275 and PPV ≈ 13.0%. Likelihood ratios LR+ = 0.85/0.30 ≈ 2.833 and LR− = 0.15/0.70 ≈ 0.214 travel across base rates; predictive values do not.

![3.1: Positive predictive value of the LVO bedside screen as a function of disease prevalence at fixed sensitivity 0.85 and sp](../assets/figures/ml_concept_3.1_1c7b7412.png)

*Figure 3.1 — original teaching graphic.*

![PPV vs prevalence with chapter LVO sens/spec (scientific recompute; original).](../assets/figures/ml_fig_ppv_prevalence.png)

*Scientific recompute of the same claim: at π=0.20, PPV≈0.41; at π=0.05, PPV≈0.13 (original matplotlib).*

```
prev, sens, spec = 0.20, 0.85, 0.70
p_pos = sens*prev + (1-spec)*(1-prev) # 0.41
ppv = (sens*prev)/p_pos # ~0.4146
lr_plus = sens/(1-spec) # ~2.833
print(p_pos, ppv, lr_plus)
```

## PDF, PMF, CDF, and a Catalog of Distributions

The PMF of a discrete RV assigns probabilities to points. The PDF of a continuous RV assigns density whose integrals over sets give probabilities. The cumulative distribution function F(x) = P(X ≤ x) is nondecreasing and right-continuous, with F(−∞) = 0 and F(∞) = 1; for continuous RVs, f = F′ almost everywhere. Survival functions 1 − F(x) appear throughout time-to-event analysis.

![3.2: A catalog of common distributions used in modeling: Normal (pdf), Poisson (pmf, lambda=4), Beta (three shapes, including](../assets/figures/ml_concept_3.2_72206d12.png)

*Figure 3.2 — original teaching graphic.*

### Normal (Gaussian)

X ~ N(μ, σ²) has support x ∈ (−∞, ∞), parameters mean μ and variance σ² > 0, with E[X] = μ and Var(X) = σ². Its density is (1/√(2πσ²)) exp(−(x−μ)²/(2σ²)). Linear combinations of independent Gaussians are Gaussian; the CLT explains approximate normality of many aggregates. Standardization Z = (X−μ)/σ ~ N(0,1). It models approximately symmetric measurement noise—for example, repeated volumetric readings of the same infarct; beware floors, ceilings, and skew in NIHSS-like scores.

### Uniform

Continuous Uniform(a,b) has support [a,b], parameters endpoints a < b, constant density 1/(b−a), with E[X] = (a+b)/2 and Var(X) = (b−a)²/12. Discrete uniform places equal mass 1/K on a finite set of K values. It models complete indifference over a bounded range. Uniforms appear as noninformative priors on bounded intervals, as random seeds for simulation, and as null models for fair lotteries—rarely as models of raw clinical labs.

### Beta and Dirichlet

Beta(α, β) has support (0,1), parameters shape α, β > 0, with E[X] = α/(α+β) and Var(X) = αβ/((α+β)²(α+β+1)). It is conjugate to Bernoulli/Binomial likelihoods—ideal for modeling an unknown proportion with prior pseudo-counts α−1 successes and β−1 failures. Clinically, a Beta(2, 8) prior encodes a plausible ~20% complication rate that observed events then update to a Beta posterior. Dirichlet is the multivariate generalization with support on the K-simplex (nonnegative vectors summing to 1), parameter vector α = (α_1, …, α_K), and mean component E[X_k] = α_k / Σ_j α_j; it is conjugate to categorical/multinomial models and central to topic models and mixed-membership clustering—for example, expressing a patient’s fractional membership across stroke etiologies. When α = β = 1, Beta is Uniform(0,1).

### Bernoulli, Binomial, Geometric

Bernoulli(p): support {0,1}, parameter success probability p ∈ [0,1], with E[X]=p and Var(X)=p(1−p); it models a single yes/no event such as thrombolysis given or not. Binomial(n,p): support {0,1,…,n}, the count of successes in n i.i.d. Bernoulli trials, P(Y=k)=C(n,k) p^k (1−p)^{n−k}, with E[Y]=np and Var(Y)=np(1−p); it models, say, the number of successful recanalizations in n independent thrombectomy attempts. Geometric(p): support {1,2,…} under the trials-until-first-success convention (definitions vary on whether the success trial is counted), with E[X]=1/p and Var(X)=(1−p)/p²; it models the number of independent trials to the first event and is memoryless.

### Poisson and Weibull

Poisson(λ): support {0,1,2,…}, parameter rate λ > 0, P(X=k)=e^{−λ} λ^k/k! for counts with E[X]=Var(X)=λ—stroke arrivals per day, rare adverse events. Overdispersion (variance > mean) pushes toward negative binomial. Weibull(k, λ): support (0,∞), shape k > 0 and scale λ > 0, with E[X]=λ·Γ(1+1/k); its hazard increases (k > 1), stays constant (k = 1, reducing to the Exponential), or decreases (k < 1). This flexible hazard makes it a workhorse of reliability and survival analysis—for example, a high early hazard after craniectomy that falls over time—when proportional hazards or exponential waiting times are too rigid.

### Power-law, exponential, Zipf, and Pareto

The Exponential(λ) has support (0,∞), rate parameter λ > 0, E[X]=1/λ and Var(X)=1/λ²; it is the memoryless continuous waiting time between events of a constant-hazard Poisson process (inter-arrival times of stroke admissions). Heavy-tailed laws are its opposite in spirit. The Pareto distribution has support x ≥ x_m with tail index α > 0 and mean αx_m/(α−1) only when α > 1—for α ≤ 1 the mean is infinite. Zipf is its discrete rank–frequency cousin: the r-th most common item has probability ∝ r^{−s}. These power laws appear in city sizes, word frequencies, and sometimes care utilization or network degree distributions. They imply that sample means can be unstable and that extreme events dominate totals. Always test fit carefully; many claimed power laws are merely heavy-tailed over a limited range.

### Chi-square and Boltzmann

Chi-square χ²(k) has support (0,∞), a single parameter—the degrees of freedom k—with E[X]=k and Var(X)=2k; it is the distribution of the sum of squares of k independent standard normals, a special Gamma, appearing in variance estimators and goodness-of-fit statistics. Boltzmann (and Gibbs) distributions in statistical physics have support over discrete states of energy E, assigning probability e^{−E/kT}/Z, where the partition function Z = Σ e^{−E/kT} normalizes and T is temperature; analogous softmax forms appear throughout ML as Gibbs distributions over labels or configurations, linking energy-based models to probabilistic classification.

### PP plots and QQ plots

Probability–probability (PP) plots graph empirical CDF values against theoretical CDF values; points near the diagonal support the theoretical model. Quantile–quantile (QQ) plots graph empirical quantiles against theoretical quantiles—more sensitive in the tails, hence preferred for checking Gaussian residuals or heavy tails. Systematic curvature diagnoses skewness; S-shapes diagnose tail weight.

![3.3: Normal quantile-quantile plots. A Gaussian sample falls along the reference line (left), while a right-skewed exponentia](../assets/figures/ml_concept_3.3_13979a8f.png)

*Figure 3.3 — original teaching graphic.*

## Expectation, Normalization, and How Much Data Is Enough

Expectation E[X] is the probability-weighted average (sum or integral). Linearity E[aX+bY]=aE[X]+bE[Y] holds without independence. Variance expands as E[X²]−(E[X])². Law of the unconscious statistician: E[g(X)] integrates g against the distribution of X without first finding the law of g(X).

![3.4: The central limit theorem in action. The sampling distribution of the mean of n Exponential(1) draws for n = 1, 5, 30 be](../assets/figures/ml_concept_3.4_b4d4fb23.png)

*Figure 3.4 — original teaching graphic.*

![Central limit theorem: sampling means of Exponential(1) for n=5 vs n=40 (scientific; original).](../assets/figures/ml_fig_clt_sampling.png)

*Parent density is skewed; means of n=5 remain skewed; means of n=40 are nearly Gaussian (original Monte Carlo panel).*

Z-score normalization transforms a value via z = (x − μ)/σ (or sample estimates s). Features on incommensurate scales become comparable; many distance-based learners (k-means, k-NN, PCA) require this discipline. Z-scores assume a roughly symmetric scale meaning; they do not fix heavy tails or coding errors. Robust alternatives use median and IQR.

The law of large numbers (LLN): sample averages converge to expectations under i.i.d. sampling with finite mean—the philosophical backbone of ‘more data helps.’ The central limit theorem (CLT): standardized sums become approximately normal for large n under mild conditions—justifying many Wald intervals and z-tests. Sampling bias (nonrandom selection, volunteer bias, collider stratification) breaks the link between sample and target population no matter how large n grows. Confidence intervals at level 1−α are random intervals that cover the true parameter with probability 1−α under the model in repeated sampling; approximate Wald interval for a proportion: p̂ ± z_{α/2} √(p̂(1−p̂)/n).

```
import math
p_hat, n, z = 0.17, 200, 1.96
se = math.sqrt(p_hat*(1-p_hat)/n) # ~0.0266
lo, hi = p_hat - z*se, p_hat + z*se # ~(0.118, 0.222)
print(se, lo, hi)
```

## Hypothesis Tests, Multiplicity, and Effect Size

Hypothesis testing formalizes conflict between data and a null H_0. A test statistic measures discrepancy; a p-value is the probability under H_0 of a result at least as extreme as observed—not the probability H_0 is true, and not clinical importance. Type I error is false rejection of H_0; Type II is false non-rejection. A/B tests in digital products are sequential or fixed-horizon significance tests on metrics; clinical analogues require pre-specification and care with peeking.

### Parametric tests: t-tests and ANOVA family

One-sample t-tests compare a mean to a null value; independent two-sample t-tests compare means of two groups; paired t-tests use within-pair differences. ANOVA compares means across more than two groups (one-way); repeated-measures ANOVA handles within-subject factors; factorial ANOVA examines multiple factors and interactions. MANOVA extends to multiple continuous outcomes jointly. ANCOVA blends ANOVA with regression adjustment for covariates. Assumptions include approximate normality of errors and, for classical variants, variance homogeneity; large n helps via CLT but does not fix bad estimands.

### Nonparametric tests

Chi-square tests assess independence in contingency tables or goodness-of-fit to a specified discrete distribution. Kolmogorov–Smirnov compares empirical and theoretical CDFs (or two empirical CDFs). Kruskal–Wallis is a rank-based multi-group alternative to one-way ANOVA; Mann–Whitney U (Wilcoxon rank-sum) compares two groups on stochastic dominance/ranks without requiring normality. Use these when ordinal outcomes or heavy tails make mean comparisons brittle—mRS comparisons often live here or in ordinal logistic models.

### Multiplicity: Bonferroni and Tukey

Testing many hypotheses inflates family-wise Type I error. Bonferroni adjusts by using α/m for m tests (conservative when tests are correlated). Tukey honest significant difference controls family-wise error for all pairwise mean comparisons after ANOVA. False discovery rate methods (Benjamini–Hochberg) are popular in high-dimensional screens. Pre-specify primary endpoints; exploratory subgroup p-values are hypothesis-generating without adjustment theater.

### Effect size: Cohen’s d and odds ratios

Statistical significance is not effect size. Cohen’s d = (μ_1 − μ_0)/s_pooled standardizes mean differences. Odds ratios compare odds of an event between groups; they arise naturally from logistic regression coefficients (exp(β)). Report effects with confidence intervals. Correlation coefficients (Pearson linear, Spearman rank) quantify association strength; they are not interventional effects.

![3.5: Effect size for two equal-variance Normal groups separated by Cohen's d = 0.80. The shaded region is the distributional ](../assets/figures/ml_concept_3.5_0843b4cd.png)

*Figure 3.5 — original teaching graphic.*

## Entropy, Divergences, Maximum Likelihood, and EM

Shannon entropy H(X) = −Σ_x p(x) log p(x) (or integral analogue) measures uncertainty in a distribution. Information gain in decision trees is the reduction in entropy (or impurity) from a split—feature selection by how much knowing X reduces uncertainty about Y. Kullback–Leibler divergence KL(p || q) = Σ p log(p/q) measures directed discrepancy from q to p (not symmetric, not a metric). Cross-entropy H(p,q) = −Σ p log q = H(p) + KL(p || q) is the workhorse classification training loss when p is a one-hot label and q is the model’s predicted distribution. Jensen–Shannon divergence symmetrizes and smooths KL, yielding a bounded score used in distribution comparison and GAN theory.

![3.6: Kullback-Leibler divergence between two discrete outcome (mRS) distributions p and q: paired probability masses (left) a](../assets/figures/ml_concept_3.6_9f5f31cf.png)

*Figure 3.6 — original teaching graphic.*

Maximum likelihood estimation chooses θ maximizing L(θ) = Π_i f(x_i | θ), usually via log-likelihood. For i.i.d. Bernoulli data with k successes in n trials, p̂_MLE = k/n. For i.i.d. Gaussians, μ̂ = x̄ and σ̂² uses divisor n (MLE) versus n−1 (unbiased). Regularized and Bayesian MAP estimates multiply by priors.

![Bernoulli log-likelihood surface with MLE at k/n (scientific; original).](../assets/figures/ml_fig_mle_bernoulli.png)

*For k=7 successes in n=10 trials, ℓ(p)=7 log p + 3 log(1−p) peaks at p̂=0.7 (original scientific plot).*

Expectation–maximization (EM) maximizes likelihood when some variables are latent (mixture memberships, missing labels). A concrete walkthrough: fit a two-component univariate Gaussian mixture to data x_1, …, x_n, where each point comes from component 1 ~ N(μ_1, σ_1²) with probability π or component 2 ~ N(μ_2, σ_2²) with probability 1−π, but the component label is unobserved. Maximizing the observed-data log-likelihood Σ_i log[π N(x_i | μ_1, σ_1²) + (1−π) N(x_i | μ_2, σ_2²)] directly is awkward because of the sum inside the log. EM sidesteps this by alternating two steps from an initial guess of (π, μ_1, σ_1², μ_2, σ_2²):

E-step: compute each point’s responsibility—the posterior probability it belongs to component 1—r_i = π N(x_i | μ_1, σ_1²) / [π N(x_i | μ_1, σ_1²) + (1−π) N(x_i | μ_2, σ_2²)]. These soft labels lie in [0,1] and use the current parameters.

M-step: re-estimate parameters as responsibility-weighted statistics: π ← (1/n) Σ_i r_i; μ_1 ← (Σ_i r_i x_i)/(Σ_i r_i); σ_1² ← (Σ_i r_i (x_i − μ_1)²)/(Σ_i r_i); and symmetrically for component 2 using weights (1 − r_i). This is a weighted version of the ordinary Gaussian MLE.

Each iteration provably does not decrease the observed-data likelihood, so EM climbs to a stationary point—usually a local, not global, maximum, which is why several random initializations and a best-likelihood pick are standard. EM underpins Gaussian mixture clustering, many HMM estimators (Baum–Welch), and missing-data algorithms. Clinically, a NIHSS histogram with a mild-stroke bulk and a severe-stroke tail can be fit as such a mixture, with responsibilities giving each patient a soft membership rather than a hard threshold cut; the same soft labels should never be reported as verified etiologic subtypes.

```
# Bernoulli MLE sketch
k, n = 34, 200
p_hat = k / n # 0.17
# loglik(p) = k*log(p) + (n-k)*log(1-p); critical point p = k/n
```

## From Distributions to Modeling Choices

Choosing a distribution is choosing a set of assumptions about support, tail weight, mean–variance relationships, and conjugacy. Binary endpoints invite Bernoulli or Binomial models and logistic links. Counts invite Poisson or negative binomial models; if the variance greatly exceeds the mean in stroke arrival or readmission counts, Poisson standard errors will be optimistically small. Symmetric continuous errors invite Gaussians; positive skewed labs may invite log-normal or Gamma models; proportions on (0,1) invite Beta regression; categorical labels with more than two levels invite multinomial models with Dirichlet priors in Bayesian settings.

Heavy-tailed behavior changes estimators as much as it changes plots. If length of stay or cost has a Pareto-like tail, the sample mean’s variance may be huge, and a few patients dominate totals—exactly when hospital contribution margins and outlier payment policies matter. Zipf-like ranks appear in token frequencies in clinical text: a handful of tokens carry most mass, which is why sublinear scaling and rare-token handling dominate NLP preprocessing. Weibull and other survival distributions remind us that censoring is part of the likelihood, not a nuisance to delete.

![Kaplan–Meier curves with right-censor marks and at-risk counts (synthetic; original).](../assets/figures/ml_fig_km_censor.png)

*Figure — Censoring is information. **Left:** two synthetic groups with Kaplan–Meier \(\hat S(t)\); “+” marks right-censored observations that leave the risk set without an event. **Right:** n still at risk at landmark times—late steps rest on few patients. Never drop censored rows as if they were missing completely; the likelihood must account for them. Group separation on a KM plot is association under the observed censoring process, not proof that group membership caused the survival difference.*

![Competing risks: CIF vs naive 1−KM treating other deaths as censor (synthetic; original).](../assets/figures/ml_fig_competing_risks.png)

*Figure — Two ways to fail. **Left:** cumulative incidence functions for event A and competing event B; their sum is the risk of either. **Right:** naive 1−KM that censors competing deaths can overstate cause-specific risk. Prefer CIF / Fine–Gray thinking when death other than stroke competes; curves are absolute risks under a regime—not automatic causal effects of covariates.*

![Hierarchical / empirical-Bayes shrinkage of small-site rates (synthetic; original).](../assets/figures/ml_fig_hierarchical_shrink.png)

*Figure — Small n sites shrink harder toward the global mean. Stabilizes noisy rates for surveillance dashboards; shrinkage is statistical borrowing—not proof that sites caused outcomes.*

Mixture distributions—finite Gaussian mixtures, zero-inflated counts, spike-and-slab priors—formalize the clinical intuition that one density cannot describe everyone. A NIHSS distribution with a spike at zero and a long right tail is not a failed Gaussian; it is a mixture of mild and severe regimes. EM and related algorithms estimate such mixtures; visualization (histograms, QQ plots) tells you when to try them. Boltzmann and softmax forms reappear when you turn energies or decision scores into probabilities: multiclass logistic regression is a Gibbs distribution over labels.

PP and QQ plots deserve routine use before parametric tests. A t-test on heavily skewed mRS differences may still be approximately valid for large n by the CLT applied to means, but effect-size interpretation and interval coverage can suffer; rank-based tests or ordinal models may match the estimand better. QQ plots of residuals after regression diagnose whether Gaussian-based intervals are decorative or trusted.

## Worked Example Extensions: Likelihood Ratios and Information

Return to the LVO screen with prevalence 0.20, sensitivity 0.85, and specificity 0.70. Prior odds of LVO are 0.20/0.80 = 0.25. After a positive screen, posterior odds = LR+ × prior odds ≈ 2.833 × 0.25 ≈ 0.708, so posterior probability ≈ 0.708/(1+0.708) ≈ 0.415, matching the earlier PPV. After a negative screen, posterior odds ≈ 0.214 × 0.25 ≈ 0.0535, posterior probability ≈ 0.051. This odds arithmetic is the everyday form of Bayes used on teaching rounds; ML systems that emit only uncalibrated scores force clinicians to invent a private, unstated likelihood ratio—which is how silent misuse begins.

![Likelihood ratios map prevalence to post-test probability (sens=0.85, spec=0.70; original).](../assets/figures/ml_fig_lr_prevalence.png)

![Bernoulli likelihood and likelihood-ratio intuition for coin-like events (original).](../assets/figures/ml_fig_likelihood_ratio_coin.png)

*Figure — Likelihood underwrites Bayes. **Left:** normalized Bernoulli likelihood for k events in n trials; MLE at k/n vs a point null at 0.5. **Right:** raw likelihood values under the null vs at the MLE (teaching LR). Formal Bayes factors need priors; LR updates odds—still not a causal claim.*


![Beta–Binomial posterior sharpening with more Bernoulli data (original).](../assets/figures/ml_fig_beta_binomial.png)

*Figure — Conjugate updating. Prior Beta(2,2) concentrates after successive observations. Posteriors quantify uncertainty for rates used in prediction counseling—**not** treatment effects without causal design.*


![Confidence band for the mean vs prediction interval for new y (synthetic OLS; original).](../assets/figures/ml_fig_ci_vs_pi.png)

*Figure — CI vs PI. The narrow band is uncertainty in the mean function; the wide band is for a new observation. Neither interval alone establishes causation.*


![Statistical power vs sample size for a fixed effect (synthetic; original).](../assets/figures/ml_fig_power_curve.png)

*Figure — Power increases with n. 80% is a planning guide, not clinical importance. Power calculations support study design—they do not alone identify causal effects.*


![Family-wise error vs number of tests at alpha=0.05 (original).](../assets/figures/ml_fig_multiple_testing.png)

*Figure — Multiplicity. Unadjusted testing inflates false positives as m grows. Correcting multiplicity is statistical hygiene—not by itself a causal design.*


![CI width versus sample size (teaching; original).](../assets/figures/ml_fig_ci_width_n.png)

*Figure — Narrow intervals are precision—not causal effects. Pred ≠ cause without design.*


![SE of mean vs n (original).](../assets/figures/ml_fig_sample_size_se.png)

*Figure — Standard errors shrink with sqrt(n). SE of mean vs n Pred != cause without design.*


![clt teaching panel (original).](../assets/figures/ml_fig_clt_hist.png)

*Figure — Teaching panel for clt. Pred != cause without design.*


![Cycle-34 densify scientific panel 5 (original).](../assets/figures/ml_fig_c34_04.png)

*Figure — Continuous densify panel 5. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 5 (original).](../assets/figures/ml_fig_c35_04.png)

*Figure — Continuous densify panel 5. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 5 (original).](../assets/figures/ml_fig_c36_04.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 5 (original).](../assets/figures/ml_fig_c37_04.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 5 (original).](../assets/figures/ml_fig_c38_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 5 (original).](../assets/figures/ml_fig_c39_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 5 (original).](../assets/figures/ml_fig_c40_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 5 (original).](../assets/figures/ml_fig_c41_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 5 (original).](../assets/figures/ml_fig_c42_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 5 (original).](../assets/figures/ml_fig_c43_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 5 (original).](../assets/figures/ml_fig_c44_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 5 (original).](../assets/figures/ml_fig_c45_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 5 (original).](../assets/figures/ml_fig_c46_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 5 (original).](../assets/figures/ml_fig_c47_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 5 (original).](../assets/figures/ml_fig_c48_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 5 (original).](../assets/figures/ml_fig_c49_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 5 (original).](../assets/figures/ml_fig_c50_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 5 (original).](../assets/figures/ml_fig_c51_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 5 (original).](../assets/figures/ml_fig_c52_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 5 (original).](../assets/figures/ml_fig_c53_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 5 (original).](../assets/figures/ml_fig_c54_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 5 (original).](../assets/figures/ml_fig_c55_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 5 (original).](../assets/figures/ml_fig_c56_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 5 (original).](../assets/figures/ml_fig_c57_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 5 (original).](../assets/figures/ml_fig_c58_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 5 (original).](../assets/figures/ml_fig_c59_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 5 (original).](../assets/figures/ml_fig_c60_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 5 (original).](../assets/figures/ml_fig_c61_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 5 (original).](../assets/figures/ml_fig_c62_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 5 (original).](../assets/figures/ml_fig_c63_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 5 (original).](../assets/figures/ml_fig_c64_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 5 (original).](../assets/figures/ml_fig_c65_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 5 (original).](../assets/figures/ml_fig_c66_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 5 (original).](../assets/figures/ml_fig_c67_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 5 (original).](../assets/figures/ml_fig_c68_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 5 (original).](../assets/figures/ml_fig_c69_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 5 (original).](../assets/figures/ml_fig_c70_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 5 (original).](../assets/figures/ml_fig_c71_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 5 (original).](../assets/figures/ml_fig_c72_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 5 (original).](../assets/figures/ml_fig_c73_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 5 (original).](../assets/figures/ml_fig_c74_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 5 (original).](../assets/figures/ml_fig_c75_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 5 (original).](../assets/figures/ml_fig_c76_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 5 (original).](../assets/figures/ml_fig_c77_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 5 (original).](../assets/figures/ml_fig_c78_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 5 (original).](../assets/figures/ml_fig_c79_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 5 (original).](../assets/figures/ml_fig_c80_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 5 (original).](../assets/figures/ml_fig_c81_04.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

*Figure — LR+ ≈ 2.83 and LR− ≈ 0.214 travel across base rates; PPV/NPV do not. **Left:** PPV after a positive screen as a function of pre-test prevalence π (vertical mark at the chapter’s π=0.20 ED example). **Right:** residual disease probability after a negative screen. Copying a paper’s PPV into a different prevalence is a base-rate error.*

Now connect to information theory with a tiny discrete example, using natural logs throughout so the units are nats. Suppose a binary outcome Y with P(Y=1)=0.2 and a binary feature X = screen result, with P(X=1)=0.41 (the positive-screen rate above) carrying the conditional probabilities from the screening table. The outcome’s entropy is H(Y) = −0.2 ln 0.2 − 0.8 ln 0.8 ≈ 0.500 nats. (Watch the units: the same quantity is 0.722 bits, because bits = nats / ln 2 = 0.500 / 0.6931; the numeral 0.7219 is the value in bits, not nats.) After observing X, the expected conditional entropy H(Y|X) is the screen-rate-weighted average of binary entropies at the posteriors 0.415 (screen +) and 0.051 (screen −): H(Y|X=1) ≈ 0.678 nats and H(Y|X=0) ≈ 0.201 nats; with P(X=1)=0.41 and P(X=0)=0.59, H(Y|X) ≈ 0.41·0.678 + 0.59·0.201 ≈ 0.397 nats. Information gain IG = H(Y) − H(Y|X) ≈ 0.500 − 0.397 ≈ 0.104 nats (about 0.15 bits). The screen cuts outcome uncertainty by roughly a fifth—a moderate, not decisive, reduction, consistent with a moderate LR+.

```
import math
def bernoulli_entropy(p):
 if p <= 0 or p >= 1:
 return 0.0
 return -p*math.log(p) - (1-p)*math.log(1-p)
H_y = bernoulli_entropy(0.20) # 0.500 nats
H_yx = 0.41*bernoulli_entropy(0.4146) + 0.59*bernoulli_entropy(0.0508) # 0.397 nats
IG = H_y - H_yx # 0.104 nats
print(H_y, H_yx, IG) # nats
```

Cross-entropy loss in classification is the same mathematical family: if the true label is a one-hot p and the model predicts q, the contribution −log q_true is minimized when q puts mass on the correct class. KL divergence adds the interpretation of extra bits needed if codes optimized for q are used when reality is p. Jensen–Shannon offers a symmetric, bounded alternative when comparing two models’ predictive distributions or two hospitals’ outcome distributions without treating either as absolute truth.

A small numeric micro-example makes these three quantities concrete. Take a true distribution over two classes p = (0.5, 0.5) and a model’s prediction q = (0.9, 0.1), with natural logs. Cross-entropy is H(p,q) = −Σ p log q = −0.5 ln 0.9 − 0.5 ln 0.1 ≈ 0.053 + 1.151 = 1.204 nats. The label entropy is H(p) = −0.5 ln 0.5 − 0.5 ln 0.5 = ln 2 ≈ 0.693 nats. Their difference is the KL divergence: KL(p || q) = H(p,q) − H(p) ≈ 1.204 − 0.693 = 0.511 nats, which also equals Σ p log(p/q) = 0.5 ln(0.5/0.9) + 0.5 ln(0.5/0.1) computed directly—confirming the identity H(p,q) = H(p) + KL(p || q). KL is asymmetric: reversing the arguments gives KL(q || p) = 0.9 ln(0.9/0.5) + 0.1 ln(0.1/0.5) ≈ 0.368 nats ≠ 0.511. Jensen–Shannon symmetrizes by averaging each distribution’s KL to the mixture m = (p+q)/2 = (0.7, 0.3): JS(p,q) = ½ KL(p || m) + ½ KL(q || m) ≈ ½(0.087) + ½(0.116) = 0.102 nats. JS stays finite and bounded (≤ ln 2 ≈ 0.693 nats) even when one distribution puts near-zero mass where the other does not—the regime where KL diverges to infinity—which is why JS is preferred for comparing two hospitals’ outcome distributions or two models’ predictions when neither is treated as absolute truth.

```
import math
p = [0.5, 0.5]; q = [0.9, 0.1]
m = [(pi+qi)/2 for pi, qi in zip(p, q)] # [0.7, 0.3]
H_p = -sum(pi*math.log(pi) for pi in p) # 0.693
H_pq = -sum(pi*math.log(qi) for pi, qi in zip(p, q)) # 1.204
KL_pq = sum(pi*math.log(pi/qi) for pi, qi in zip(p, q)) # 0.511
JS = (0.5*sum(pi*math.log(pi/mi) for pi, mi in zip(p, m))
 + 0.5*sum(qi*math.log(qi/mi) for qi, mi in zip(q, m))) # 0.102
print(H_p, H_pq, KL_pq, JS) # nats; check H_pq == H_p + KL_pq
```

## Hypothesis Testing in Registries and Trials: A Practical Map

Clinical ML projects inherit statistical testing cultures from both trials and observational epidemiology. A one-sample t-test might compare mean door-to-needle time to a guideline benchmark. An independent two-sample t-test might compare mean infarct volumes between device arms if approximate normality holds; otherwise Mann–Whitney is a common alternative on ranks. Paired t-tests fit before–after measurements on the same patients (for example, NIHSS at admission and at 24 hours), acknowledging dependence by analyzing differences.

When more than two groups appear—four hospital tiers, three device generations—one-way ANOVA tests equality of means under classical assumptions; Kruskal–Wallis is the rank analogue. Factorial ANOVA addresses two factors at once (for example, sex and treatment) and their interaction. Repeated-measures ANOVA handles multiple time points per patient but is increasingly replaced by mixed models that better tolerate missing visits. MANOVA addresses several continuous outcomes together (volume, shift, NIHSS) when a single combined null is scientifically meaningful; ANCOVA adjusts a group comparison for a baseline covariate (baseline severity), sitting close to linear regression with an indicator plus covariate.

Categorical outcomes use chi-square tests of independence (or Fisher’s exact in small tables) and goodness-of-fit tests against fixed theoretical proportions. Kolmogorov–Smirnov compares full distributions, useful when any difference in CDF—not only means—matters. Multiplicity is not optional theater: if you test twenty subgroups of a stroke trial for ‘signal,’ Bonferroni’s α/m is a blunt but transparent control of family-wise error; Tukey’s HSD is tailored to all pairwise mean comparisons after ANOVA; false discovery rate control is often preferable in high-dimensional biomarker screens where some false discoveries are tolerable.

Effect sizes keep significance tests from becoming idols. A tiny p-value for a 0.1-point NIHSS difference in a huge registry is not a quality revolution. Cohen’s d puts mean differences on a scale relative to noise; odds ratios from logistic models state multiplicative changes in odds—remembering that rare-event odds ratios approximate risk ratios, but common-event odds ratios do not. Always accompany tests with intervals and with a plain-language statement of the estimand: difference in means? difference in medians? odds ratio conditional on covariates?

## Clinical and Epidemiologic Notes

Diagnostic reasoning in neurology is Bayesian whether or not the formula is written down. Pre-test probability of LVO after a severe hemispheric syndrome differs from that after a transient monocular symptom; the same test language does not yield the same post-test beliefs. Screening tools with moderate specificity destroy PPV when prevalence falls—as the numerical example showed. Seizure-detection algorithms face extreme imbalance: hours of non-seizure for minutes of seizure. Reporting accuracy alone is malpractice-level evaluation design; precision at fixed recall, false alarms per hour, and time-to-detection dominate operational utility.

Study design constrains what probabilities mean. Incidence versus prevalence, competing risks for mRS and death, interval censoring of last-known-well, and informative missingness of 90-day outcomes change estimands. Models trained on complete cases silently condition on selection; the learned P(Y | X, observed) may not equal the population P(Y | X). Directed acyclic graphs and missing-data assumptions (MCAR, MAR, MNAR) determine whether your likelihood is even the right likelihood. Unsupervised clusters of ‘stroke phenotype’ are not etiology labels; association of a cluster with outcome is not proof of a causal subtype.

Population research still relies on Poisson and negative binomial models for counts, logistic models for dichotomized mRS, and ordinal models for full mRS. Flexible ML estimators of conditional means or distributions inherit the same identification problems as classical models. Transporting a model across sites may change P(X), P(Y|X), or both; probability literacy is what lets you name which piece shifted. Calibration plots and prevalence-aware metrics are the applied face of that literacy.

Always state population and prevalence when reporting PPV/NPV or precision.

Prefer likelihood ratios and calibrated probabilities for transfer across base rates.

Match metrics to imbalance and to clinical cost of false alarms.

Separate discrimination (AUC) from calibration; name the estimand explicitly.

Do not treat p < 0.05 as clinical importance or as P(H_0 false).

Use independence assumptions only when sampling design supports them.

## Sampling, Bias, and Confidence Intervals in Depth

The law of large numbers reassures us that averages settle; the central limit theorem reassures us that many averages look Gaussian when n is large enough. Neither theorem erases bias in how the sample was formed. Convenience samples of patients who complete 90-day follow-up systematically differ from those lost to follow-up; survivors who answer phones may be healthier, shifting estimated disability downward. Volunteer bias in app-based stroke diaries, referral bias into comprehensive centers, and immortal-time bias in poorly aligned exposures all create sampling distributions centered on the wrong target. More data of the wrong kind produces more confident wrongness—the opposite of what LLN intuition suggests if you forget the population.

Confidence intervals quantify uncertainty under a model and a sampling plan. For a mean with estimated standard error SE, an approximate 95% interval is estimate ± 1.96·SE when n is large and variance is finite. For proportions near 0 or 1, Wald intervals misbehave; Wilson or Jeffreys intervals are safer. Bootstrap intervals resample the dataset to approximate sampling variability when analytic SEs are hard—still assuming the observed sample represents the target process. Clustered data need cluster-robust or hierarchical SEs; ignoring hospital clustering is a common way to manufacture spurious precision in multi-center stroke analyses.

How much data is enough depends on effect size, variance, design effect from clustering, class imbalance, and the decision threshold for action. Power calculations for trials are the classical answer for hypothesis tests; for prediction models, learning curves and external validation matter more than a single p-value target. A model trained on 50 events with 100 features will overfit regardless of CLT slogans. Rules of thumb (events-per-variable) are crude but better than magical thinking that deep networks remove sample-size constraints.

## Connecting Classical Tests to Machine-Learning Metrics

Sensitivity is recall of the positive class; specificity is recall of the negative class; precision is positive predictive value on the evaluation distribution. F1 is the harmonic mean of precision and recall—hence its kinship with harmonic means discussed earlier. AUROC summarizes ranking quality across thresholds and is prevalence-invariant in a way that can hide poor PPV in low-prevalence screening. Average precision and precision–recall curves are often more informative under imbalance. Brier score and log-loss (cross-entropy) reward calibrated probabilities; they are scoring rules with deep roots in probability theory, not ad hoc leaderboard ornaments.

A/B tests in digital health products and classical two-group trials share sequential and multiplicity issues. Peeking at results daily without spending alpha inflates Type I error—the same sin as tuning on a test set until AUROC looks good. Pre-specification, locking analysis code, and using validation splits for model selection are the ML analogues of a statistical analysis plan. When both a classical endpoint test and an ML risk model appear in one paper, keep their estimands distinct: a significant mean difference in a trial arm is not automatic proof that a post-hoc cluster or a black-box score should guide individual care.

## Pitfalls and Senior Practice

Pitfalls to avoid: (1) Confusing P(D | +) with P(+ | D). (2) Applying sensitivity and specificity estimated in a severe cohort to a mild population without checking spectrum bias. (3) Interpreting a 95% CI as a 95% posterior probability without a Bayesian model. (4) Reporting only AUC for screening tasks with 1% prevalence. (5) Assuming independence across patients clustered in hospitals or across EEG windows from the same ICU stay. (6) Maximizing likelihood on leaked labels or on the test set. (7) Equating predictive feature importance with causal effect. (8) Multiple silent hypothesis tests across subgroups until a ‘significant’ stroke subtype appears. (9) Using arithmetic means alone for heavily skewed costs or lengths of stay. (10) Treating EM’s soft labels as ground-truth subtypes.

Senior practice is boring in the best way: define the sample space and estimand, write the likelihood you claim to be optimizing, report uncertainty, check calibration at deployment prevalence, and refuse to let software defaults substitute for probabilistic reasoning. Probability and statistics are not prerequisites you outgrow when deep learning arrives; they are the only language in which deep learning’s claims can be stated without self-deception.


![c82 teaching panel 04 (original).](../assets/figures/ml_fig_c82_04.png)
*Figure — Bayes update: prior × likelihood → posterior with MAP marker. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 04 (original).](../assets/figures/ml_fig_c83_04.png)
*Figure — Central limit theorem: sampling distribution of the mean. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 04 (original).](../assets/figures/ml_fig_c84_04.png)
*Figure — Confidence interval for the mean vs prediction interval for a new y. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 04 (original).](../assets/figures/ml_fig_c85_04.png)
*Figure — Gaussian location/scale family of densities. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 04 (original).](../assets/figures/ml_fig_c86_04.png)
*Figure — Log loss and entropy: cost of confident mistakes. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 04 (original).](../assets/figures/ml_fig_c87_04.png)
*Figure — Chi-square-like density family shapes. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 04 (original).](../assets/figures/ml_fig_c88_04.png)
*Figure — Bootstrap resampling distribution of a statistic. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 04 (original).](../assets/figures/ml_fig_c89_04.png)
*Figure — Statistical power vs sample size curve. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 04 (original).](../assets/figures/ml_fig_c90_04.png)
*Figure — Null sampling dist with critical region. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 04 (original).](../assets/figures/ml_fig_c91_04.png)
*Figure — p-value as tail probability under H0. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 04 (original).](../assets/figures/ml_fig_c92_04.png)
*Figure — Multiple testing Bonferroni wall. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 04 (original).](../assets/figures/ml_fig_c93_04.png)
*Figure — Likelihood ratio test idea. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 04 (original).](../assets/figures/ml_fig_c94_04.png)
*Figure — Type I vs Type II error regions. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 04 (original).](../assets/figures/ml_fig_c95_04.png)
*Figure — FDR Benjamini-Hochberg steps. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 04 (original).](../assets/figures/ml_fig_c96_04.png)
*Figure — Power vs effect size curves. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 04 (original).](../assets/figures/ml_fig_c97_04.png)
*Figure — Confidence interval coverage sim. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 04 (original).](../assets/figures/ml_fig_c98_04.png)
*Figure — Permutation test null histogram. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 04 (original).](../assets/figures/ml_fig_c99_04.png)
*Figure — Bootstrap CI percentile method. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 04 (original).](../assets/figures/ml_fig_c100_04.png)
*Figure — Exact vs asymptotic tests. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 04 (original).](../assets/figures/ml_fig_c101_04.png)
*Figure — TOST equivalence testing. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 04 (original).](../assets/figures/ml_fig_c102_04.png)
*Figure — Sequential testing alpha spend. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 04 (original).](../assets/figures/ml_fig_c103_04.png)
*Figure — Likelihood ratio confidence set. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 04 (original).](../assets/figures/ml_fig_c104_04.png)
*Figure — Wilson score interval. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 04 (original).](../assets/figures/ml_fig_c105_04.png)
*Figure — False discovery proportion. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 04 (original).](../assets/figures/ml_fig_c106_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 04 (original).](../assets/figures/ml_fig_c107_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 04 (original).](../assets/figures/ml_fig_c108_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 04 (original).](../assets/figures/ml_fig_c109_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 04 (original).](../assets/figures/ml_fig_c110_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 04 (original).](../assets/figures/ml_fig_c111_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 04 (original).](../assets/figures/ml_fig_c112_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 04 (original).](../assets/figures/ml_fig_c113_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 04 (original).](../assets/figures/ml_fig_c114_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 04 (original).](../assets/figures/ml_fig_c115_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 04 (original).](../assets/figures/ml_fig_c116_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 04 (original).](../assets/figures/ml_fig_c117_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 04 (original).](../assets/figures/ml_fig_c118_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 04 (original).](../assets/figures/ml_fig_c119_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 04 (original).](../assets/figures/ml_fig_c120_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 04 (original).](../assets/figures/ml_fig_c121_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 04 (original).](../assets/figures/ml_fig_c122_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 04 (original).](../assets/figures/ml_fig_c123_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 04 (original).](../assets/figures/ml_fig_c124_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 04 (original).](../assets/figures/ml_fig_c125_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 04 (original).](../assets/figures/ml_fig_c126_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 04 (original).](../assets/figures/ml_fig_c127_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 04 (original).](../assets/figures/ml_fig_c128_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 04 (original).](../assets/figures/ml_fig_c129_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 04 (original).](../assets/figures/ml_fig_c130_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 04 (original).](../assets/figures/ml_fig_c131_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 04 (original).](../assets/figures/ml_fig_c132_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 04 (original).](../assets/figures/ml_fig_c133_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 04 (original).](../assets/figures/ml_fig_c134_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 04 (original).](../assets/figures/ml_fig_c135_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 04 (original).](../assets/figures/ml_fig_c136_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 04 (original).](../assets/figures/ml_fig_c137_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 04 (original).](../assets/figures/ml_fig_c138_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 04 (original).](../assets/figures/ml_fig_c139_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 04 (original).](../assets/figures/ml_fig_c140_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 04 (original).](../assets/figures/ml_fig_c141_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 04 (original).](../assets/figures/ml_fig_c142_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 04 (original).](../assets/figures/ml_fig_c143_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 04 (original).](../assets/figures/ml_fig_c144_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 04 (original).](../assets/figures/ml_fig_c145_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 04 (original).](../assets/figures/ml_fig_c146_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 04 (original).](../assets/figures/ml_fig_c147_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 04 (original).](../assets/figures/ml_fig_c148_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 04 (original).](../assets/figures/ml_fig_c149_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 04 (original).](../assets/figures/ml_fig_c150_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 04 (original).](../assets/figures/ml_fig_c151_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 04 (original).](../assets/figures/ml_fig_c152_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 04 (original).](../assets/figures/ml_fig_c153_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 04 (original).](../assets/figures/ml_fig_c154_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 04 (original).](../assets/figures/ml_fig_c155_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 04 (original).](../assets/figures/ml_fig_c156_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 04 (original).](../assets/figures/ml_fig_c157_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 04 (original).](../assets/figures/ml_fig_c158_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 04 (original).](../assets/figures/ml_fig_c159_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 04 (original).](../assets/figures/ml_fig_c160_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 04 (original).](../assets/figures/ml_fig_c161_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 04 (original).](../assets/figures/ml_fig_c162_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 04 (original).](../assets/figures/ml_fig_c163_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 04 (original).](../assets/figures/ml_fig_c164_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 04 (original).](../assets/figures/ml_fig_c165_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 04 (original).](../assets/figures/ml_fig_c166_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 04 (original).](../assets/figures/ml_fig_c167_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 04 (original).](../assets/figures/ml_fig_c168_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 04 (original).](../assets/figures/ml_fig_c169_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 04 (original).](../assets/figures/ml_fig_c170_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 04 (original).](../assets/figures/ml_fig_c171_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 04 (original).](../assets/figures/ml_fig_c172_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 04 (original).](../assets/figures/ml_fig_c173_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 04 (original).](../assets/figures/ml_fig_c174_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 04 (original).](../assets/figures/ml_fig_c175_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 04 (original).](../assets/figures/ml_fig_c176_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 04 (original).](../assets/figures/ml_fig_c177_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 04 (original).](../assets/figures/ml_fig_c178_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 04 (original).](../assets/figures/ml_fig_c179_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 04 (original).](../assets/figures/ml_fig_c180_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 04 (original).](../assets/figures/ml_fig_c181_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 04 (original).](../assets/figures/ml_fig_c182_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 04 (original).](../assets/figures/ml_fig_c183_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 04 (original).](../assets/figures/ml_fig_c184_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 04 (original).](../assets/figures/ml_fig_c185_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 04 (original).](../assets/figures/ml_fig_c186_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 04 (original).](../assets/figures/ml_fig_c187_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 04 (original).](../assets/figures/ml_fig_c188_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 04 (original).](../assets/figures/ml_fig_c189_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 04 (original).](../assets/figures/ml_fig_c190_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 04 (original).](../assets/figures/ml_fig_c191_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 04 (original).](../assets/figures/ml_fig_c192_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 04 (original).](../assets/figures/ml_fig_c193_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 04 (original).](../assets/figures/ml_fig_c194_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 04 (original).](../assets/figures/ml_fig_c195_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 04 (original).](../assets/figures/ml_fig_c196_04.png)
*Figure — Beta-binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 04 (original).](../assets/figures/ml_fig_c197_04.png)
*Figure — Empirical Bayes shrink. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 04 (original).](../assets/figures/ml_fig_c198_04.png)
*Figure — Posterior predictive check. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 04 (original).](../assets/figures/ml_fig_c199_04.png)
*Figure — Credible vs confidence. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 04 (original).](../assets/figures/ml_fig_c200_04.png)
*Figure — Jeffreys prior sketch. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 04 (original).](../assets/figures/ml_fig_c201_04.png)
*Figure — Total variation distance masses. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 04 (original).](../assets/figures/ml_fig_c202_04.png)
*Figure — PIT histogram calibration check. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 04 (original).](../assets/figures/ml_fig_c203_04.png)
*Figure — One-dimensional Wasserstein transport. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 04 (original).](../assets/figures/ml_fig_c204_04.png)
*Figure — CRPS forecast CDF integrand. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 04 (original).](../assets/figures/ml_fig_c205_04.png)
*Figure — KS statistic ECDF gap. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 04 (original).](../assets/figures/ml_fig_c206_04.png)
*Figure — Bootstrap percentile CI. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 04 (original).](../assets/figures/ml_fig_c207_04.png)
*Figure — Jeffreys Beta prior vs uniform. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 04 (original).](../assets/figures/ml_fig_c208_04.png)
*Figure — Credible vs confidence interval. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 04 (original).](../assets/figures/ml_fig_c209_04.png)
*Figure — PIT histogram calibration check. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 04 (original).](../assets/figures/ml_fig_c210_04.png)
*Figure — Posterior predictive line bundle. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 04 (original).](../assets/figures/ml_fig_c211_04.png)
*Figure — Beta-binomial posterior update. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 04 (original).](../assets/figures/ml_fig_c212_04.png)
*Figure — Empirical null z-density fit. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 04 (original).](../assets/figures/ml_fig_c213_04.png)
*Figure — Normal-Normal conjugate update. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 04 (original).](../assets/figures/ml_fig_c214_04.png)
*Figure — ABC rejection sampling posterior. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 04 (original).](../assets/figures/ml_fig_c215_04.png)
*Figure — Dirichlet simplex sample cloud. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 04 (original).](../assets/figures/ml_fig_c216_04.png)
*Figure — Gaussian process posterior band. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 04 (original).](../assets/figures/ml_fig_c217_04.png)
*Figure — Conjugate likelihood prior map. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 04 (original).](../assets/figures/ml_fig_c218_04.png)
*Figure — Variational ELBO and KL path. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 04 (original).](../assets/figures/ml_fig_c219_04.png)
*Figure — Dirichlet process stick break. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 04 (original).](../assets/figures/ml_fig_c220_04.png)
*Figure — ABC-SMC epsilon annealing. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 04 (original).](../assets/figures/ml_fig_c221_04.png)
*Figure — Power posterior temperature path. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 04 (original).](../assets/figures/ml_fig_c222_04.png)
*Figure — Stein score discrepancy field. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 04 (original).](../assets/figures/ml_fig_c223_04.png)
*Figure — Variational free energy terms. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 04 (original).](../assets/figures/ml_fig_c224_04.png)
*Figure — Slice sampling under density. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 04 (original).](../assets/figures/ml_fig_c225_04.png)
*Figure — Highest posterior density region. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 04 (original).](../assets/figures/ml_fig_c226_04.png)
*Figure — PIT calibration histogram. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 04 (original).](../assets/figures/ml_fig_c227_04.png)
*Figure — Empirical Bayes group shrink. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 04 (original).](../assets/figures/ml_fig_c228_04.png)
*Figure — GP prior sample paths. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 04 (original).](../assets/figures/ml_fig_c229_04.png)
*Figure — Geweke early-late MCMC check. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 04 (original).](../assets/figures/ml_fig_c230_04.png)
*Figure — Horseshoe prior density spike. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 04 (original).](../assets/figures/ml_fig_c231_04.png)
*Figure — Posterior predictive draw fan. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 04 (original).](../assets/figures/ml_fig_c232_04.png)
*Figure — Variational dropout KL path. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 04 (original).](../assets/figures/ml_fig_c233_04.png)
*Figure — Gelman-Rubin R-hat approach. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 04 (original).](../assets/figures/ml_fig_c234_04.png)
*Figure — NUTS phase-space field. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 04 (original).](../assets/figures/ml_fig_c235_04.png)
*Figure — Rank-normalized R-hat path. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 04 (original).](../assets/figures/ml_fig_c236_04.png)
*Figure — HMC leapfrog field. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 04 (original).](../assets/figures/ml_fig_c237_04.png)
*Figure — ESS per chain path. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 04 (original).](../assets/figures/ml_fig_c238_04.png)
*Figure — Langevin dynamics field. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 04 (original).](../assets/figures/ml_fig_c239_04.png)
*Figure — Autocorr time path. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 04 (original).](../assets/figures/ml_fig_c240_04.png)
*Figure — Score-matching field. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 04 (original).](../assets/figures/ml_fig_c241_04.png)
*Figure — Gelman-Rubin multi-chain path. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 04 (original).](../assets/figures/ml_fig_c242_04.png)
*Figure — Score-based reverse field. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 04 (original).](../assets/figures/ml_fig_c243_04.png)
*Figure — Split-R-hat diagnostic path. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 04 (original).](../assets/figures/ml_fig_c244_04.png)
*Figure — Probability flow ODE field. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 04 (original).](../assets/figures/ml_fig_c245_04.png)
*Figure — Nested R-hat convergence path. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 04 (original).](../assets/figures/ml_fig_c246_04.png)
*Figure — Diffusion reverse SDE field. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 04 (original).](../assets/figures/ml_fig_c247_04.png)
*Figure — Bulk ESS multi-chain path. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 04 (original).](../assets/figures/ml_fig_c248_04.png)
*Figure — Score matching Langevin field. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 04 (original).](../assets/figures/ml_fig_c249_04.png)
*Figure — Rank-normalized R-hat path. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 04 (original).](../assets/figures/ml_fig_c250_04.png)
*Figure — Probability flow field. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 04 (original).](../assets/figures/ml_fig_c251_04.png)
*Figure — Multi-chain ESS path. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 04 (original).](../assets/figures/ml_fig_c252_04.png)
*Figure — Hamiltonian flow field. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 04 (original).](../assets/figures/ml_fig_c253_04.png)
*Figure — Split-R-hat path. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 04 (original).](../assets/figures/ml_fig_c254_04.png)
*Figure — Score-based Langevin field. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 04 (original).](../assets/figures/ml_fig_c255_04.png)
*Figure — Bulk ESS path. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 04 (original).](../assets/figures/ml_fig_c256_04.png)
*Figure — Reverse diffusion field. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 04 (original).](../assets/figures/ml_fig_c257_04.png)
*Figure — Importance weight ESS c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 04 (original).](../assets/figures/ml_fig_c258_04.png)
*Figure — ABC tolerance cool path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 04 (original).](../assets/figures/ml_fig_c259_04.png)
*Figure — Conjugate update path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 04 (original).](../assets/figures/ml_fig_c260_04.png)
*Figure — CLT sample mean path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 04 (original).](../assets/figures/ml_fig_c261_04.png)
*Figure — Bootstrap CI width path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 04 (original).](../assets/figures/ml_fig_c262_04.png)
*Figure — Jackknife bias path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 04 (original).](../assets/figures/ml_fig_c263_04.png)
*Figure — Posterior concentration path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 04 (original).](../assets/figures/ml_fig_c264_04.png)
*Figure — Prior sensitivity path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 04 (original).](../assets/figures/ml_fig_c265_04.png)
*Figure — Likelihood ratio path c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 04 (original).](../assets/figures/ml_fig_c266_04.png)
*Figure — Power vs n curve c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 04 (original).](../assets/figures/ml_fig_c267_04.png)
*Figure — Type I/II trade path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 04 (original).](../assets/figures/ml_fig_c268_04.png)
*Figure — FDR control path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 04 (original).](../assets/figures/ml_fig_c269_04.png)
*Figure — Permutation null bars c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 04 (original).](../assets/figures/ml_fig_c270_04.png)
*Figure — Credible interval path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 04 (original).](../assets/figures/ml_fig_c271_04.png)
*Figure — HMC acceptance path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 04 (original).](../assets/figures/ml_fig_c272_04.png)
*Figure — Variational ELBO path c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 04 (original).](../assets/figures/ml_fig_c273_04.png)
*Figure — Importance weight ESS c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 04 (original).](../assets/figures/ml_fig_c274_04.png)
*Figure — ABC tolerance cool path c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 04 (original).](../assets/figures/ml_fig_c275_04.png)
*Figure — Conjugate update path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 04 (original).](../assets/figures/ml_fig_c276_04.png)
*Figure — CLT sample mean path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 04 (original).](../assets/figures/ml_fig_c277_04.png)
*Figure — Bootstrap CI width path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 04 (original).](../assets/figures/ml_fig_c278_04.png)
*Figure — Jackknife bias path c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 04 (original).](../assets/figures/ml_fig_c279_04.png)
*Figure — Posterior concentration path c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 04 (original).](../assets/figures/ml_fig_c280_04.png)
*Figure — Prior sensitivity path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 04 (original).](../assets/figures/ml_fig_c281_04.png)
*Figure — Likelihood ratio path c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 04 (original).](../assets/figures/ml_fig_c282_04.png)
*Figure — Power vs n curve c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 04 (original).](../assets/figures/ml_fig_c283_04.png)
*Figure — Type I/II trade path c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 04 (original).](../assets/figures/ml_fig_c284_04.png)
*Figure — FDR control path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 04 (original).](../assets/figures/ml_fig_c285_04.png)
*Figure — Permutation null bars c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 04 (original).](../assets/figures/ml_fig_c286_04.png)
*Figure — Credible interval path c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 04 (original).](../assets/figures/ml_fig_c287_04.png)
*Figure — HMC acceptance path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 04 (original).](../assets/figures/ml_fig_c288_04.png)
*Figure — Variational ELBO path c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 04 (original).](../assets/figures/ml_fig_c289_04.png)
*Figure — Importance weight ESS c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 04 (original).](../assets/figures/ml_fig_c290_04.png)
*Figure — ABC tolerance cool path c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 04 (original).](../assets/figures/ml_fig_c291_04.png)
*Figure — Conjugate update path c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 04 (original).](../assets/figures/ml_fig_c292_04.png)
*Figure — CLT sample mean path c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 04 (original).](../assets/figures/ml_fig_c293_04.png)
*Figure — Bootstrap CI width path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 04 (original).](../assets/figures/ml_fig_c294_04.png)
*Figure — Jackknife bias path c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 04 (original).](../assets/figures/ml_fig_c295_04.png)
*Figure — Posterior concentration path c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 04 (original).](../assets/figures/ml_fig_c296_04.png)
*Figure — Prior sensitivity path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 04 (original).](../assets/figures/ml_fig_c297_04.png)
*Figure — Likelihood ratio path c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 04 (original).](../assets/figures/ml_fig_c298_04.png)
*Figure — Power vs n curve c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 04 (original).](../assets/figures/ml_fig_c299_04.png)
*Figure — Type I/II trade path c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 04 (original).](../assets/figures/ml_fig_c300_04.png)
*Figure — FDR control path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 04 (original).](../assets/figures/ml_fig_c301_04.png)
*Figure — Permutation null bars c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 04 (original).](../assets/figures/ml_fig_c302_04.png)
*Figure — Credible interval path c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 04 (original).](../assets/figures/ml_fig_c303_04.png)
*Figure — HMC acceptance path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 04 (original).](../assets/figures/ml_fig_c304_04.png)
*Figure — Variational ELBO path c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 04 (original).](../assets/figures/ml_fig_c305_04.png)
*Figure — Importance weight ESS c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 04 (original).](../assets/figures/ml_fig_c306_04.png)
*Figure — ABC tolerance cool path c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 04 (original).](../assets/figures/ml_fig_c307_04.png)
*Figure — Conjugate update path c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 04 (original).](../assets/figures/ml_fig_c308_04.png)
*Figure — CLT sample mean path c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 04 (original).](../assets/figures/ml_fig_c309_04.png)
*Figure — Bootstrap CI width path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 04 (original).](../assets/figures/ml_fig_c310_04.png)
*Figure — Jackknife bias path c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 04 (original).](../assets/figures/ml_fig_c311_04.png)
*Figure — Posterior concentration path c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 04 (original).](../assets/figures/ml_fig_c312_04.png)
*Figure — Prior sensitivity path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 04 (original).](../assets/figures/ml_fig_c313_04.png)
*Figure — Likelihood ratio path c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 04 (original).](../assets/figures/ml_fig_c314_04.png)
*Figure — Power vs n curve c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 04 (original).](../assets/figures/ml_fig_c315_04.png)
*Figure — Type I/II trade path c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 04 (original).](../assets/figures/ml_fig_c316_04.png)
*Figure — FDR control path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 04 (original).](../assets/figures/ml_fig_c317_04.png)
*Figure — Permutation null bars c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 04 (original).](../assets/figures/ml_fig_c318_04.png)
*Figure — Credible interval path c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 04 (original).](../assets/figures/ml_fig_c319_04.png)
*Figure — HMC acceptance path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 04 (original).](../assets/figures/ml_fig_c320_04.png)
*Figure — Variational ELBO path c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 04 (original).](../assets/figures/ml_fig_c321_04.png)
*Figure — Importance weight ESS c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 04 (original).](../assets/figures/ml_fig_c322_04.png)
*Figure — ABC tolerance cool path c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 04 (original).](../assets/figures/ml_fig_c323_04.png)
*Figure — Conjugate update path c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 04 (original).](../assets/figures/ml_fig_c324_04.png)
*Figure — CLT sample mean path c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 04 (original).](../assets/figures/ml_fig_c325_04.png)
*Figure — Bootstrap CI width path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 04 (original).](../assets/figures/ml_fig_c326_04.png)
*Figure — Jackknife bias path c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 04 (original).](../assets/figures/ml_fig_c327_04.png)
*Figure — Posterior concentration path c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 04 (original).](../assets/figures/ml_fig_c328_04.png)
*Figure — Prior sensitivity path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 04 (original).](../assets/figures/ml_fig_c329_04.png)
*Figure — Likelihood ratio path c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 04 (original).](../assets/figures/ml_fig_c330_04.png)
*Figure — Power vs n curve c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 04 (original).](../assets/figures/ml_fig_c331_04.png)
*Figure — Type I/II trade path c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 04 (original).](../assets/figures/ml_fig_c332_04.png)
*Figure — FDR control path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 04 (original).](../assets/figures/ml_fig_c333_04.png)
*Figure — Permutation null bars c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 04 (original).](../assets/figures/ml_fig_c334_04.png)
*Figure — Credible interval path c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 04 (original).](../assets/figures/ml_fig_c335_04.png)
*Figure — HMC acceptance path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 04 (original).](../assets/figures/ml_fig_c336_04.png)
*Figure — Variational ELBO path c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 04 (original).](../assets/figures/ml_fig_c337_04.png)
*Figure — Importance weight ESS c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 04 (original).](../assets/figures/ml_fig_c338_04.png)
*Figure — ABC tolerance cool path c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 04 (original).](../assets/figures/ml_fig_c339_04.png)
*Figure — Conjugate update path c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 04 (original).](../assets/figures/ml_fig_c340_04.png)
*Figure — CLT sample mean path c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 04 (original).](../assets/figures/ml_fig_c341_04.png)
*Figure — Bootstrap CI width path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 04 (original).](../assets/figures/ml_fig_c342_04.png)
*Figure — Jackknife bias path c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 04 (original).](../assets/figures/ml_fig_c343_04.png)
*Figure — Posterior concentration path c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 04 (original).](../assets/figures/ml_fig_c344_04.png)
*Figure — Prior sensitivity path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 04 (original).](../assets/figures/ml_fig_c345_04.png)
*Figure — Likelihood ratio path c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 04 (original).](../assets/figures/ml_fig_c346_04.png)
*Figure — Power vs n curve c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 04 (original).](../assets/figures/ml_fig_c347_04.png)
*Figure — Type I/II trade path c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 04 (original).](../assets/figures/ml_fig_c348_04.png)
*Figure — FDR control path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 04 (original).](../assets/figures/ml_fig_c349_04.png)
*Figure — Permutation null bars c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 04 (original).](../assets/figures/ml_fig_c350_04.png)
*Figure — Credible interval path c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 04 (original).](../assets/figures/ml_fig_c351_04.png)
*Figure — HMC acceptance path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 04 (original).](../assets/figures/ml_fig_c352_04.png)
*Figure — Variational ELBO path c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 04 (original).](../assets/figures/ml_fig_c353_04.png)
*Figure — Importance weight ESS c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 04 (original).](../assets/figures/ml_fig_c354_04.png)
*Figure — ABC tolerance cool path c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 04 (original).](../assets/figures/ml_fig_c355_04.png)
*Figure — Conjugate update path c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 04 (original).](../assets/figures/ml_fig_c356_04.png)
*Figure — CLT sample mean path c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 04 (original).](../assets/figures/ml_fig_c357_04.png)
*Figure — Bootstrap CI width path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 04 (original).](../assets/figures/ml_fig_c358_04.png)
*Figure — Jackknife bias path c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 04 (original).](../assets/figures/ml_fig_c359_04.png)
*Figure — Posterior concentration path c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 04 (original).](../assets/figures/ml_fig_c360_04.png)
*Figure — Prior sensitivity path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 04 (original).](../assets/figures/ml_fig_c361_04.png)
*Figure — Likelihood ratio path c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 04 (original).](../assets/figures/ml_fig_c362_04.png)
*Figure — Power vs n curve c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 04 (original).](../assets/figures/ml_fig_c363_04.png)
*Figure — Type I/II trade path c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 04 (original).](../assets/figures/ml_fig_c364_04.png)
*Figure — FDR control path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 04 (original).](../assets/figures/ml_fig_c365_04.png)
*Figure — Permutation null bars c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 04 (original).](../assets/figures/ml_fig_c366_04.png)
*Figure — Credible interval path c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 04 (original).](../assets/figures/ml_fig_c367_04.png)
*Figure — HMC acceptance path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 04 (original).](../assets/figures/ml_fig_c368_04.png)
*Figure — Variational ELBO path c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 04 (original).](../assets/figures/ml_fig_c369_04.png)
*Figure — Importance weight ESS c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 04 (original).](../assets/figures/ml_fig_c370_04.png)
*Figure — ABC tolerance cool path c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 04 (original).](../assets/figures/ml_fig_c371_04.png)
*Figure — Conjugate update path c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 04 (original).](../assets/figures/ml_fig_c372_04.png)
*Figure — CLT sample mean path c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 04 (original).](../assets/figures/ml_fig_c373_04.png)
*Figure — Bootstrap CI width path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 04 (original).](../assets/figures/ml_fig_c374_04.png)
*Figure — Jackknife bias path c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 04 (original).](../assets/figures/ml_fig_c375_04.png)
*Figure — Posterior concentration path c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 04 (original).](../assets/figures/ml_fig_c376_04.png)
*Figure — Prior sensitivity path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 04 (original).](../assets/figures/ml_fig_c377_04.png)
*Figure — Likelihood ratio path c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 04 (original).](../assets/figures/ml_fig_c378_04.png)
*Figure — Power vs n curve c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 04 (original).](../assets/figures/ml_fig_c379_04.png)
*Figure — Type I/II trade path c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 04 (original).](../assets/figures/ml_fig_c380_04.png)
*Figure — FDR control path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 04 (original).](../assets/figures/ml_fig_c381_04.png)
*Figure — Permutation null bars c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 04 (original).](../assets/figures/ml_fig_c382_04.png)
*Figure — Credible interval path c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 04 (original).](../assets/figures/ml_fig_c383_04.png)
*Figure — HMC acceptance path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 04 (original).](../assets/figures/ml_fig_c384_04.png)
*Figure — Variational ELBO path c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 04 (original).](../assets/figures/ml_fig_c385_04.png)
*Figure — Importance weight ESS c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 04 (original).](../assets/figures/ml_fig_c386_04.png)
*Figure — ABC tolerance cool path c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 04 (original).](../assets/figures/ml_fig_c387_04.png)
*Figure — Conjugate update path c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 04 (original).](../assets/figures/ml_fig_c388_04.png)
*Figure — CLT sample mean path c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 04 (original).](../assets/figures/ml_fig_c389_04.png)
*Figure — Bootstrap CI width path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 04 (original).](../assets/figures/ml_fig_c390_04.png)
*Figure — Jackknife bias path c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 04 (original).](../assets/figures/ml_fig_c391_04.png)
*Figure — Posterior concentration path c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 04 (original).](../assets/figures/ml_fig_c392_04.png)
*Figure — Prior sensitivity path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 04 (original).](../assets/figures/ml_fig_c393_04.png)
*Figure — Likelihood ratio path c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 04 (original).](../assets/figures/ml_fig_c394_04.png)
*Figure — Power vs n curve c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 04 (original).](../assets/figures/ml_fig_c395_04.png)
*Figure — Type I/II trade path c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 04 (original).](../assets/figures/ml_fig_c396_04.png)
*Figure — FDR control path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 04 (original).](../assets/figures/ml_fig_c397_04.png)
*Figure — Permutation null bars c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 04 (original).](../assets/figures/ml_fig_c398_04.png)
*Figure — Credible interval path c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 04 (original).](../assets/figures/ml_fig_c399_04.png)
*Figure — HMC acceptance path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 04 (original).](../assets/figures/ml_fig_c400_04.png)
*Figure — Variational ELBO path c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 04 (original).](../assets/figures/ml_fig_c401_04.png)
*Figure — Importance weight ESS c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 04 (original).](../assets/figures/ml_fig_c402_04.png)
*Figure — ABC tolerance cool path c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 04 (original).](../assets/figures/ml_fig_c403_04.png)
*Figure — Conjugate update path c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 04 (original).](../assets/figures/ml_fig_c404_04.png)
*Figure — CLT sample mean path c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 04 (original).](../assets/figures/ml_fig_c405_04.png)
*Figure — Bootstrap CI width path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 04 (original).](../assets/figures/ml_fig_c406_04.png)
*Figure — Jackknife bias path c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 04 (original).](../assets/figures/ml_fig_c407_04.png)
*Figure — Posterior concentration path c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 04 (original).](../assets/figures/ml_fig_c408_04.png)
*Figure — Prior sensitivity path c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 04 (original).](../assets/figures/ml_fig_c409_04.png)
*Figure — Likelihood ratio path c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 04 (original).](../assets/figures/ml_fig_c410_04.png)
*Figure — Power vs n curve c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 04 (original).](../assets/figures/ml_fig_c411_04.png)
*Figure — Type I/II trade path c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 04 (original).](../assets/figures/ml_fig_c412_04.png)
*Figure — FDR control path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 04 (original).](../assets/figures/ml_fig_c413_04.png)
*Figure — Permutation null bars c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 04 (original).](../assets/figures/ml_fig_c414_04.png)
*Figure — Credible interval path c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 04 (original).](../assets/figures/ml_fig_c415_04.png)
*Figure — HMC acceptance path c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 04 (original).](../assets/figures/ml_fig_c416_04.png)
*Figure — Variational ELBO path c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 04 (original).](../assets/figures/ml_fig_c417_04.png)
*Figure — Importance weight ESS c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 04 (original).](../assets/figures/ml_fig_c418_04.png)
*Figure — ABC tolerance cool path c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 04 (original).](../assets/figures/ml_fig_c419_04.png)
*Figure — Conjugate update path c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 04 (original).](../assets/figures/ml_fig_c420_04.png)
*Figure — CLT sample mean path c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 04 (original).](../assets/figures/ml_fig_c421_04.png)
*Figure — Bootstrap CI width path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 04 (original).](../assets/figures/ml_fig_c422_04.png)
*Figure — Jackknife bias path c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 04 (original).](../assets/figures/ml_fig_c423_04.png)
*Figure — Posterior concentration path c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 04 (original).](../assets/figures/ml_fig_c424_04.png)
*Figure — Prior sensitivity path c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 04 (original).](../assets/figures/ml_fig_c425_04.png)
*Figure — Likelihood ratio path c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 04 (original).](../assets/figures/ml_fig_c426_04.png)
*Figure — Power vs n curve c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 04 (original).](../assets/figures/ml_fig_c427_04.png)
*Figure — Type I/II trade path c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 04 (original).](../assets/figures/ml_fig_c428_04.png)
*Figure — FDR control path c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 04 (original).](../assets/figures/ml_fig_c429_04.png)
*Figure — Permutation null bars c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 04 (original).](../assets/figures/ml_fig_c430_04.png)
*Figure — Credible interval path c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 04 (original).](../assets/figures/ml_fig_c431_04.png)
*Figure — HMC acceptance path c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 04 (original).](../assets/figures/ml_fig_c432_04.png)
*Figure — Variational ELBO path c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 04 (original).](../assets/figures/ml_fig_c433_04.png)
*Figure — Importance weight ESS c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 04 (original).](../assets/figures/ml_fig_c434_04.png)
*Figure — ABC tolerance cool path c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 04 (original).](../assets/figures/ml_fig_c435_04.png)
*Figure — Conjugate update path c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 04 (original).](../assets/figures/ml_fig_c436_04.png)
*Figure — CLT sample mean path c436. Synthetic teaching geometry—not a causal claim.*


![c437 teaching panel 04 (original).](../assets/figures/ml_fig_c437_04.png)
*Figure — Bootstrap CI width path c437. Synthetic teaching geometry—not a causal claim.*


![c438 teaching panel 04 (original).](../assets/figures/ml_fig_c438_04.png)
*Figure — Jackknife bias path c438. Synthetic teaching geometry—not a causal claim.*


![c439 teaching panel 04 (original).](../assets/figures/ml_fig_c439_04.png)
*Figure — Posterior concentration path c439. Synthetic teaching geometry—not a causal claim.*


![c440 teaching panel 04 (original).](../assets/figures/ml_fig_c440_04.png)
*Figure — Prior sensitivity path c440. Synthetic teaching geometry—not a causal claim.*


![c441 teaching panel 04 (original).](../assets/figures/ml_fig_c441_04.png)
*Figure — Likelihood ratio path c441. Synthetic teaching geometry—not a causal claim.*


![c442 teaching panel 04 (original).](../assets/figures/ml_fig_c442_04.png)
*Figure — Power vs n curve c442. Synthetic teaching geometry—not a causal claim.*


![c443 teaching panel 04 (original).](../assets/figures/ml_fig_c443_04.png)
*Figure — Type I/II trade path c443. Synthetic teaching geometry—not a causal claim.*


![c444 teaching panel 04 (original).](../assets/figures/ml_fig_c444_04.png)
*Figure — FDR control path c444. Synthetic teaching geometry—not a causal claim.*


![c445 teaching panel 04 (original).](../assets/figures/ml_fig_c445_04.png)
*Figure — Permutation null bars c445. Synthetic teaching geometry—not a causal claim.*


![c446 teaching panel 04 (original).](../assets/figures/ml_fig_c446_04.png)
*Figure — Credible interval path c446. Synthetic teaching geometry—not a causal claim.*


![c447 teaching panel 04 (original).](../assets/figures/ml_fig_c447_04.png)
*Figure — HMC acceptance path c447. Synthetic teaching geometry—not a causal claim.*


![c448 teaching panel 04 (original).](../assets/figures/ml_fig_c448_04.png)
*Figure — Variational ELBO path c448. Synthetic teaching geometry—not a causal claim.*


![c449 teaching panel 04 (original).](../assets/figures/ml_fig_c449_04.png)
*Figure — Importance weight ESS c449. Synthetic teaching geometry—not a causal claim.*


![c450 teaching panel 04 (original).](../assets/figures/ml_fig_c450_04.png)
*Figure — ABC tolerance cool path c450. Synthetic teaching geometry—not a causal claim.*


![c451 teaching panel 04 (original).](../assets/figures/ml_fig_c451_04.png)
*Figure — Conjugate update path c451. Synthetic teaching geometry—not a causal claim.*


![c452 teaching panel 04 (original).](../assets/figures/ml_fig_c452_04.png)
*Figure — CLT sample mean path c452. Synthetic teaching geometry—not a causal claim.*


![c453 teaching panel 04 (original).](../assets/figures/ml_fig_c453_04.png)
*Figure — Bootstrap CI width path c453. Synthetic teaching geometry—not a causal claim.*


![c454 teaching panel 04 (original).](../assets/figures/ml_fig_c454_04.png)
*Figure — Jackknife bias path c454. Synthetic teaching geometry—not a causal claim.*


![c455 teaching panel 04 (original).](../assets/figures/ml_fig_c455_04.png)
*Figure — Posterior concentration path c455. Synthetic teaching geometry—not a causal claim.*


![c456 teaching panel 04 (original).](../assets/figures/ml_fig_c456_04.png)
*Figure — Prior sensitivity path c456. Synthetic teaching geometry—not a causal claim.*


![c457 teaching panel 04 (original).](../assets/figures/ml_fig_c457_04.png)
*Figure — Likelihood ratio path c457. Synthetic teaching geometry—not a causal claim.*


![c458 teaching panel 04 (original).](../assets/figures/ml_fig_c458_04.png)
*Figure — Power vs n curve c458. Synthetic teaching geometry—not a causal claim.*


![c459 teaching panel 04 (original).](../assets/figures/ml_fig_c459_04.png)
*Figure — Type I/II trade path c459. Synthetic teaching geometry—not a causal claim.*


![c460 teaching panel 04 (original).](../assets/figures/ml_fig_c460_04.png)
*Figure — FDR control path c460. Synthetic teaching geometry—not a causal claim.*


![c461 teaching panel 04 (original).](../assets/figures/ml_fig_c461_04.png)
*Figure — Permutation null bars c461. Synthetic teaching geometry—not a causal claim.*


![c462 teaching panel 04 (original).](../assets/figures/ml_fig_c462_04.png)
*Figure — Credible interval path c462. Synthetic teaching geometry—not a causal claim.*


![c463 teaching panel 04 (original).](../assets/figures/ml_fig_c463_04.png)
*Figure — HMC acceptance path c463. Synthetic teaching geometry—not a causal claim.*


![c464 teaching panel 04 (original).](../assets/figures/ml_fig_c464_04.png)
*Figure — Variational ELBO path c464. Synthetic teaching geometry—not a causal claim.*


![c465 teaching panel 04 (original).](../assets/figures/ml_fig_c465_04.png)
*Figure — Importance weight ESS c465. Synthetic teaching geometry—not a causal claim.*


![c466 teaching panel 04 (original).](../assets/figures/ml_fig_c466_04.png)
*Figure — ABC tolerance cool path c466. Synthetic teaching geometry—not a causal claim.*


![c467 teaching panel 04 (original).](../assets/figures/ml_fig_c467_04.png)
*Figure — Conjugate update path c467. Synthetic teaching geometry—not a causal claim.*


![c468 teaching panel 04 (original).](../assets/figures/ml_fig_c468_04.png)
*Figure — CLT sample mean path c468. Synthetic teaching geometry—not a causal claim.*


![c469 teaching panel 04 (original).](../assets/figures/ml_fig_c469_04.png)
*Figure — Bootstrap CI width path c469. Synthetic teaching geometry—not a causal claim.*


![c470 teaching panel 04 (original).](../assets/figures/ml_fig_c470_04.png)
*Figure — Jackknife bias path c470. Synthetic teaching geometry—not a causal claim.*


![c471 teaching panel 04 (original).](../assets/figures/ml_fig_c471_04.png)
*Figure — Posterior concentration path c471. Synthetic teaching geometry—not a causal claim.*


![c472 teaching panel 04 (original).](../assets/figures/ml_fig_c472_04.png)
*Figure — Prior sensitivity path c472. Synthetic teaching geometry—not a causal claim.*


![c473 teaching panel 04 (original).](../assets/figures/ml_fig_c473_04.png)
*Figure — Likelihood ratio path c473. Synthetic teaching geometry—not a causal claim.*


![c474 teaching panel 04 (original).](../assets/figures/ml_fig_c474_04.png)
*Figure — Power vs n curve c474. Synthetic teaching geometry—not a causal claim.*


![c475 teaching panel 04 (original).](../assets/figures/ml_fig_c475_04.png)
*Figure — Type I/II trade path c475. Synthetic teaching geometry—not a causal claim.*


![c476 teaching panel 04 (original).](../assets/figures/ml_fig_c476_04.png)
*Figure — FDR control path c476. Synthetic teaching geometry—not a causal claim.*


![c477 teaching panel 04 (original).](../assets/figures/ml_fig_c477_04.png)
*Figure — Permutation null bars c477. Synthetic teaching geometry—not a causal claim.*


![c478 teaching panel 04 (original).](../assets/figures/ml_fig_c478_04.png)
*Figure — Credible interval path c478. Synthetic teaching geometry—not a causal claim.*


![c479 teaching panel 04 (original).](../assets/figures/ml_fig_c479_04.png)
*Figure — HMC acceptance path c479. Synthetic teaching geometry—not a causal claim.*


![c480 teaching panel 04 (original).](../assets/figures/ml_fig_c480_04.png)
*Figure — Variational ELBO path c480. Synthetic teaching geometry—not a causal claim.*


![c481 teaching panel 04 (original).](../assets/figures/ml_fig_c481_04.png)
*Figure — Importance weight ESS c481. Synthetic teaching geometry—not a causal claim.*


![c482 teaching panel 04 (original).](../assets/figures/ml_fig_c482_04.png)
*Figure — ABC tolerance cool path c482. Synthetic teaching geometry—not a causal claim.*


![c483 teaching panel 04 (original).](../assets/figures/ml_fig_c483_04.png)
*Figure — Conjugate update path c483. Synthetic teaching geometry—not a causal claim.*


![c484 teaching panel 04 (original).](../assets/figures/ml_fig_c484_04.png)
*Figure — CLT sample mean path c484. Synthetic teaching geometry—not a causal claim.*


![c485 teaching panel 04 (original).](../assets/figures/ml_fig_c485_04.png)
*Figure — Bootstrap CI width path c485. Synthetic teaching geometry—not a causal claim.*


![c486 teaching panel 04 (original).](../assets/figures/ml_fig_c486_04.png)
*Figure — Jackknife bias path c486. Synthetic teaching geometry—not a causal claim.*


![c487 teaching panel 04 (original).](../assets/figures/ml_fig_c487_04.png)
*Figure — Posterior concentration path c487. Synthetic teaching geometry—not a causal claim.*


![c488 teaching panel 04 (original).](../assets/figures/ml_fig_c488_04.png)
*Figure — Prior sensitivity path c488. Synthetic teaching geometry—not a causal claim.*


![c489 teaching panel 04 (original).](../assets/figures/ml_fig_c489_04.png)
*Figure — Likelihood ratio path c489. Synthetic teaching geometry—not a causal claim.*


![c490 teaching panel 04 (original).](../assets/figures/ml_fig_c490_04.png)
*Figure — Power vs n curve c490. Synthetic teaching geometry—not a causal claim.*


![c491 teaching panel 04 (original).](../assets/figures/ml_fig_c491_04.png)
*Figure — Type I/II trade path c491. Synthetic teaching geometry—not a causal claim.*


![c492 teaching panel 04 (original).](../assets/figures/ml_fig_c492_04.png)
*Figure — FDR control path c492. Synthetic teaching geometry—not a causal claim.*


![c493 teaching panel 04 (original).](../assets/figures/ml_fig_c493_04.png)
*Figure — Permutation null bars c493. Synthetic teaching geometry—not a causal claim.*


![c494 teaching panel 04 (original).](../assets/figures/ml_fig_c494_04.png)
*Figure — Credible interval path c494. Synthetic teaching geometry—not a causal claim.*


![c495 teaching panel 04 (original).](../assets/figures/ml_fig_c495_04.png)
*Figure — HMC acceptance path c495. Synthetic teaching geometry—not a causal claim.*


![c496 teaching panel 04 (original).](../assets/figures/ml_fig_c496_04.png)
*Figure — Variational ELBO path c496. Synthetic teaching geometry—not a causal claim.*


![c497 teaching panel 04 (original).](../assets/figures/ml_fig_c497_04.png)
*Figure — Importance weight ESS c497. Synthetic teaching geometry—not a causal claim.*


![c498 teaching panel 04 (original).](../assets/figures/ml_fig_c498_04.png)
*Figure — ABC tolerance cool path c498. Synthetic teaching geometry—not a causal claim.*


![c499 teaching panel 04 (original).](../assets/figures/ml_fig_c499_04.png)
*Figure — Conjugate update path c499. Synthetic teaching geometry—not a causal claim.*


![c500 teaching panel 04 (original).](../assets/figures/ml_fig_c500_04.png)
*Figure — CLT sample mean path c500. Synthetic teaching geometry—not a causal claim.*


![c501 teaching panel 04 (original).](../assets/figures/ml_fig_c501_04.png)
*Figure — Bootstrap CI width path c501. Synthetic teaching geometry—not a causal claim.*


![c502 teaching panel 04 (original).](../assets/figures/ml_fig_c502_04.png)
*Figure — Jackknife bias path c502. Synthetic teaching geometry—not a causal claim.*


![c503 teaching panel 04 (original).](../assets/figures/ml_fig_c503_04.png)
*Figure — Posterior concentration path c503. Synthetic teaching geometry—not a causal claim.*


![c504 teaching panel 04 (original).](../assets/figures/ml_fig_c504_04.png)
*Figure — Prior sensitivity path c504. Synthetic teaching geometry—not a causal claim.*


![c505 teaching panel 04 (original).](../assets/figures/ml_fig_c505_04.png)
*Figure — Likelihood ratio path c505. Synthetic teaching geometry—not a causal claim.*


![c506 teaching panel 04 (original).](../assets/figures/ml_fig_c506_04.png)
*Figure — Power vs n curve c506. Synthetic teaching geometry—not a causal claim.*


![c507 teaching panel 04 (original).](../assets/figures/ml_fig_c507_04.png)
*Figure — Type I/II trade path c507. Synthetic teaching geometry—not a causal claim.*


![c508 teaching panel 04 (original).](../assets/figures/ml_fig_c508_04.png)
*Figure — FDR control path c508. Synthetic teaching geometry—not a causal claim.*


![c509 teaching panel 04 (original).](../assets/figures/ml_fig_c509_04.png)
*Figure — Permutation null bars c509. Synthetic teaching geometry—not a causal claim.*


![c510 teaching panel 04 (original).](../assets/figures/ml_fig_c510_04.png)
*Figure — Credible interval path c510. Synthetic teaching geometry—not a causal claim.*


![c511 teaching panel 04 (original).](../assets/figures/ml_fig_c511_04.png)
*Figure — HMC acceptance path c511. Synthetic teaching geometry—not a causal claim.*


![c512 teaching panel 04 (original).](../assets/figures/ml_fig_c512_04.png)
*Figure — Variational ELBO path c512. Synthetic teaching geometry—not a causal claim.*


![c513 teaching panel 04 (original).](../assets/figures/ml_fig_c513_04.png)
*Figure — Importance weight ESS c513. Synthetic teaching geometry—not a causal claim.*


![c514 teaching panel 04 (original).](../assets/figures/ml_fig_c514_04.png)
*Figure — ABC tolerance cool path c514. Synthetic teaching geometry—not a causal claim.*


![c515 teaching panel 04 (original).](../assets/figures/ml_fig_c515_04.png)
*Figure — Conjugate update path c515. Synthetic teaching geometry—not a causal claim.*


![c516 teaching panel 04 (original).](../assets/figures/ml_fig_c516_04.png)
*Figure — CLT sample mean path c516. Synthetic teaching geometry—not a causal claim.*


![c517 teaching panel 04 (original).](../assets/figures/ml_fig_c517_04.png)
*Figure — Bootstrap CI width path c517. Synthetic teaching geometry—not a causal claim.*


![c518 teaching panel 04 (original).](../assets/figures/ml_fig_c518_04.png)
*Figure — Jackknife bias path c518. Synthetic teaching geometry—not a causal claim.*


![c519 teaching panel 04 (original).](../assets/figures/ml_fig_c519_04.png)
*Figure — Posterior concentration path c519. Synthetic teaching geometry—not a causal claim.*


![c520 teaching panel 04 (original).](../assets/figures/ml_fig_c520_04.png)
*Figure — Prior sensitivity path c520. Synthetic teaching geometry—not a causal claim.*


![c521 teaching panel 04 (original).](../assets/figures/ml_fig_c521_04.png)
*Figure — Likelihood ratio path c521. Synthetic teaching geometry—not a causal claim.*


![c522 teaching panel 04 (original).](../assets/figures/ml_fig_c522_04.png)
*Figure — Power vs n curve c522. Synthetic teaching geometry—not a causal claim.*


![c523 teaching panel 04 (original).](../assets/figures/ml_fig_c523_04.png)
*Figure — Type I/II trade path c523. Synthetic teaching geometry—not a causal claim.*


![c524 teaching panel 04 (original).](../assets/figures/ml_fig_c524_04.png)
*Figure — FDR control path c524. Synthetic teaching geometry—not a causal claim.*


![c525 teaching panel 04 (original).](../assets/figures/ml_fig_c525_04.png)
*Figure — Permutation null bars c525. Synthetic teaching geometry—not a causal claim.*


![c526 teaching panel 04 (original).](../assets/figures/ml_fig_c526_04.png)
*Figure — Credible interval path c526. Synthetic teaching geometry—not a causal claim.*


![c527 teaching panel 04 (original).](../assets/figures/ml_fig_c527_04.png)
*Figure — HMC acceptance path c527. Synthetic teaching geometry—not a causal claim.*


![c528 teaching panel 04 (original).](../assets/figures/ml_fig_c528_04.png)
*Figure — Variational ELBO path c528. Synthetic teaching geometry—not a causal claim.*


![c529 teaching panel 04 (original).](../assets/figures/ml_fig_c529_04.png)
*Figure — Importance weight ESS c529. Synthetic teaching geometry—not a causal claim.*


![c530 teaching panel 04 (original).](../assets/figures/ml_fig_c530_04.png)
*Figure — ABC tolerance cool path c530. Synthetic teaching geometry—not a causal claim.*


![c531 teaching panel 04 (original).](../assets/figures/ml_fig_c531_04.png)
*Figure — Conjugate update path c531. Synthetic teaching geometry—not a causal claim.*


![c532 teaching panel 04 (original).](../assets/figures/ml_fig_c532_04.png)
*Figure — CLT sample mean path c532. Synthetic teaching geometry—not a causal claim.*


![c533 teaching panel 04 (original).](../assets/figures/ml_fig_c533_04.png)
*Figure — Bootstrap CI width path c533. Synthetic teaching geometry—not a causal claim.*


![c534 teaching panel 04 (original).](../assets/figures/ml_fig_c534_04.png)
*Figure — Jackknife bias path c534. Synthetic teaching geometry—not a causal claim.*


![c535 teaching panel 04 (original).](../assets/figures/ml_fig_c535_04.png)
*Figure — Posterior concentration path c535. Synthetic teaching geometry—not a causal claim.*


![c536 teaching panel 04 (original).](../assets/figures/ml_fig_c536_04.png)
*Figure — Prior sensitivity path c536. Synthetic teaching geometry—not a causal claim.*


![c537 teaching panel 04 (original).](../assets/figures/ml_fig_c537_04.png)
*Figure — Likelihood ratio path c537. Synthetic teaching geometry—not a causal claim.*


![c538 teaching panel 04 (original).](../assets/figures/ml_fig_c538_04.png)
*Figure — Power vs n curve c538. Synthetic teaching geometry—not a causal claim.*


![c539 teaching panel 04 (original).](../assets/figures/ml_fig_c539_04.png)
*Figure — Type I/II trade path c539. Synthetic teaching geometry—not a causal claim.*


![c540 teaching panel 04 (original).](../assets/figures/ml_fig_c540_04.png)
*Figure — FDR control path c540. Synthetic teaching geometry—not a causal claim.*


![c541 teaching panel 04 (original).](../assets/figures/ml_fig_c541_04.png)
*Figure — Permutation null bars c541. Synthetic teaching geometry—not a causal claim.*


![c542 teaching panel 04 (original).](../assets/figures/ml_fig_c542_04.png)
*Figure — Credible interval path c542. Synthetic teaching geometry—not a causal claim.*


![c543 teaching panel 04 (original).](../assets/figures/ml_fig_c543_04.png)
*Figure — HMC acceptance path c543. Synthetic teaching geometry—not a causal claim.*


![c544 teaching panel 04 (original).](../assets/figures/ml_fig_c544_04.png)
*Figure — Variational ELBO path c544. Synthetic teaching geometry—not a causal claim.*


![c545 teaching panel 04 (original).](../assets/figures/ml_fig_c545_04.png)
*Figure — Importance weight ESS c545. Synthetic teaching geometry—not a causal claim.*


![c546 teaching panel 04 (original).](../assets/figures/ml_fig_c546_04.png)
*Figure — ABC tolerance cool path c546. Synthetic teaching geometry—not a causal claim.*


![c547 teaching panel 04 (original).](../assets/figures/ml_fig_c547_04.png)
*Figure — Conjugate update path c547. Synthetic teaching geometry—not a causal claim.*


![c548 teaching panel 04 (original).](../assets/figures/ml_fig_c548_04.png)
*Figure — CLT sample mean path c548. Synthetic teaching geometry—not a causal claim.*


![c549 teaching panel 04 (original).](../assets/figures/ml_fig_c549_04.png)
*Figure — Bootstrap CI width path c549. Synthetic teaching geometry—not a causal claim.*


![c550 teaching panel 04 (original).](../assets/figures/ml_fig_c550_04.png)
*Figure — Jackknife bias path c550. Synthetic teaching geometry—not a causal claim.*


![c551 teaching panel 04 (original).](../assets/figures/ml_fig_c551_04.png)
*Figure — Posterior concentration path c551. Synthetic teaching geometry—not a causal claim.*


![c552 teaching panel 04 (original).](../assets/figures/ml_fig_c552_04.png)
*Figure — Prior sensitivity path c552. Synthetic teaching geometry—not a causal claim.*


![c553 teaching panel 04 (original).](../assets/figures/ml_fig_c553_04.png)
*Figure — Likelihood ratio path c553. Synthetic teaching geometry—not a causal claim.*


![c554 teaching panel 04 (original).](../assets/figures/ml_fig_c554_04.png)
*Figure — Power vs n curve c554. Synthetic teaching geometry—not a causal claim.*


![c555 teaching panel 04 (original).](../assets/figures/ml_fig_c555_04.png)
*Figure — Type I/II trade path c555. Synthetic teaching geometry—not a causal claim.*


![c556 teaching panel 04 (original).](../assets/figures/ml_fig_c556_04.png)
*Figure — FDR control path c556. Synthetic teaching geometry—not a causal claim.*


![c557 teaching panel 04 (original).](../assets/figures/ml_fig_c557_04.png)
*Figure — Permutation null bars c557. Synthetic teaching geometry—not a causal claim.*


![c558 teaching panel 04 (original).](../assets/figures/ml_fig_c558_04.png)
*Figure — Credible interval path c558. Synthetic teaching geometry—not a causal claim.*


![c559 teaching panel 04 (original).](../assets/figures/ml_fig_c559_04.png)
*Figure — HMC acceptance path c559. Synthetic teaching geometry—not a causal claim.*


![c560 teaching panel 04 (original).](../assets/figures/ml_fig_c560_04.png)
*Figure — Variational ELBO path c560. Synthetic teaching geometry—not a causal claim.*


![c561 teaching panel 04 (original).](../assets/figures/ml_fig_c561_04.png)
*Figure — Importance weight ESS c561. Synthetic teaching geometry—not a causal claim.*


![c562 teaching panel 04 (original).](../assets/figures/ml_fig_c562_04.png)
*Figure — ABC tolerance cool path c562. Synthetic teaching geometry—not a causal claim.*


![c563 teaching panel 04 (original).](../assets/figures/ml_fig_c563_04.png)
*Figure — Conjugate update path c563. Synthetic teaching geometry—not a causal claim.*


![c564 teaching panel 04 (original).](../assets/figures/ml_fig_c564_04.png)
*Figure — CLT sample mean path c564. Synthetic teaching geometry—not a causal claim.*


![c565 teaching panel 04 (original).](../assets/figures/ml_fig_c565_04.png)
*Figure — Bootstrap CI width path c565. Synthetic teaching geometry—not a causal claim.*


![c566 teaching panel 04 (original).](../assets/figures/ml_fig_c566_04.png)
*Figure — Jackknife bias path c566. Synthetic teaching geometry—not a causal claim.*


![c567 teaching panel 04 (original).](../assets/figures/ml_fig_c567_04.png)
*Figure — Posterior concentration path c567. Synthetic teaching geometry—not a causal claim.*


![c568 teaching panel 04 (original).](../assets/figures/ml_fig_c568_04.png)
*Figure — Prior sensitivity path c568. Synthetic teaching geometry—not a causal claim.*


![c569 teaching panel 04 (original).](../assets/figures/ml_fig_c569_04.png)
*Figure — Likelihood ratio path c569. Synthetic teaching geometry—not a causal claim.*


![c570 teaching panel 04 (original).](../assets/figures/ml_fig_c570_04.png)
*Figure — Power vs n curve c570. Synthetic teaching geometry—not a causal claim.*


![c571 teaching panel 04 (original).](../assets/figures/ml_fig_c571_04.png)
*Figure — Type I/II trade path c571. Synthetic teaching geometry—not a causal claim.*


![c572 teaching panel 04 (original).](../assets/figures/ml_fig_c572_04.png)
*Figure — FDR control path c572. Synthetic teaching geometry—not a causal claim.*


![c573 teaching panel 04 (original).](../assets/figures/ml_fig_c573_04.png)
*Figure — Permutation null bars c573. Synthetic teaching geometry—not a causal claim.*


![c574 teaching panel 04 (original).](../assets/figures/ml_fig_c574_04.png)
*Figure — Credible interval path c574. Synthetic teaching geometry—not a causal claim.*


![c575 teaching panel 04 (original).](../assets/figures/ml_fig_c575_04.png)
*Figure — HMC acceptance path c575. Synthetic teaching geometry—not a causal claim.*


![c576 teaching panel 04 (original).](../assets/figures/ml_fig_c576_04.png)
*Figure — Variational ELBO path c576. Synthetic teaching geometry—not a causal claim.*


![c577 teaching panel 04 (original).](../assets/figures/ml_fig_c577_04.png)
*Figure — Importance weight ESS c577. Synthetic teaching geometry—not a causal claim.*


![c578 teaching panel 04 (original).](../assets/figures/ml_fig_c578_04.png)
*Figure — ABC tolerance cool path c578. Synthetic teaching geometry—not a causal claim.*


![c579 teaching panel 04 (original).](../assets/figures/ml_fig_c579_04.png)
*Figure — Conjugate update path c579. Synthetic teaching geometry—not a causal claim.*


![c580 teaching panel 04 (original).](../assets/figures/ml_fig_c580_04.png)
*Figure — CLT sample mean path c580. Synthetic teaching geometry—not a causal claim.*


![c581 teaching panel 04 (original).](../assets/figures/ml_fig_c581_04.png)
*Figure — Bootstrap CI width path c581. Synthetic teaching geometry—not a causal claim.*


![c582 teaching panel 04 (original).](../assets/figures/ml_fig_c582_04.png)
*Figure — Jackknife bias path c582. Synthetic teaching geometry—not a causal claim.*


![c583 teaching panel 04 (original).](../assets/figures/ml_fig_c583_04.png)
*Figure — Posterior concentration path c583. Synthetic teaching geometry—not a causal claim.*


![c584 teaching panel 04 (original).](../assets/figures/ml_fig_c584_04.png)
*Figure — Prior sensitivity path c584. Synthetic teaching geometry—not a causal claim.*


![c585 teaching panel 04 (original).](../assets/figures/ml_fig_c585_04.png)
*Figure — Likelihood ratio path c585. Synthetic teaching geometry—not a causal claim.*


![c586 teaching panel 04 (original).](../assets/figures/ml_fig_c586_04.png)
*Figure — Power vs n curve c586. Synthetic teaching geometry—not a causal claim.*


![c587 teaching panel 04 (original).](../assets/figures/ml_fig_c587_04.png)
*Figure — Type I/II trade path c587. Synthetic teaching geometry—not a causal claim.*


![c588 teaching panel 04 (original).](../assets/figures/ml_fig_c588_04.png)
*Figure — FDR control path c588. Synthetic teaching geometry—not a causal claim.*


![c589 teaching panel 04 (original).](../assets/figures/ml_fig_c589_04.png)
*Figure — Permutation null bars c589. Synthetic teaching geometry—not a causal claim.*


![c590 teaching panel 04 (original).](../assets/figures/ml_fig_c590_04.png)
*Figure — Credible interval path c590. Synthetic teaching geometry—not a causal claim.*


![c591 teaching panel 04 (original).](../assets/figures/ml_fig_c591_04.png)
*Figure — HMC acceptance path c591. Synthetic teaching geometry—not a causal claim.*


![c592 teaching panel 04 (original).](../assets/figures/ml_fig_c592_04.png)
*Figure — Variational ELBO path c592. Synthetic teaching geometry—not a causal claim.*


![c593 teaching panel 04 (original).](../assets/figures/ml_fig_c593_04.png)
*Figure — Importance weight ESS c593. Synthetic teaching geometry—not a causal claim.*


![c594 teaching panel 04 (original).](../assets/figures/ml_fig_c594_04.png)
*Figure — ABC tolerance cool path c594. Synthetic teaching geometry—not a causal claim.*


![c595 teaching panel 04 (original).](../assets/figures/ml_fig_c595_04.png)
*Figure — Conjugate update path c595. Synthetic teaching geometry—not a causal claim.*


![c596 teaching panel 04 (original).](../assets/figures/ml_fig_c596_04.png)
*Figure — CLT sample mean path c596. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Random variables, data types, and independent trials structure probabilistic modeling. Descriptive statistics—means (arithmetic, geometric, harmonic), median, mode, variance/SD/covariance, range/quartiles/boxplots, and degrees of freedom—summarize samples. Joint and conditional probability and Bayes’ theorem convert priors and likelihoods into posteriors; a worked LVO example showed PPV ≈ 41.5% at 20% prevalence with sens 0.85 and spec 0.70, falling to ~13% at 5% prevalence. PDFs, PMFs, and CDFs frame distributions including Normal, Uniform, Beta, Dirichlet, Bernoulli/Binomial/Geometric, Poisson, Weibull, heavy-tailed/Zipf/Pareto families, Chi-square, and Boltzmann/softmax forms; PP/QQ plots check fit. Expectation, z-scores, CLT, LLN, sampling bias, and confidence intervals address estimation and sample size intuition. Hypothesis tests span t-tests, ANOVA-family methods, and nonparametric chi-square, KS, Kruskal–Wallis, and Mann–Whitney procedures, with Bonferroni and Tukey multiplicity control. Effect sizes, correlations, entropy, information gain, KL, cross-entropy, and JS divergences link statistics to modern ML losses. MLE and EM provide core estimation strategies for fully and partially observed models.

## Practice and Reflection

(1) Repeat the LVO Bayes calculation with prevalence 0.30, sensitivity 0.90, and specificity 0.60. Compute P(+), PPV, P(LVO|−), LR+, and LR−.

(2) In 10,000 patients with seizure prevalence 0.02, a detector has sensitivity 0.95 and specificity 0.90. How many true positives and false positives are expected among positives, and what is the PPV?

(3) Derive the MLE of p for i.i.d. Bernoulli observations with k successes in n trials by maximizing the log-likelihood.

(4) If stroke arrivals are Poisson with mean λ = 4 per day, compute P(X=0) and P(X≤2). What operational question does P(X≥8) address?

(5) Explain why a model with AUC 0.90 can still have poor PPV in low-prevalence LVO screening.

(6) A 95% CI for a 90-day death proportion is (0.11, 0.19). Give a correct frequentist interpretation and a common incorrect one.

(7) Why is independence violated for multiple EEG windows from the same patient, and what goes wrong if you treat them as i.i.d. for standard errors?

(8) Contrast Pearson correlation between NIHSS and door-to-needle time with a causal claim that reducing NIHSS would change door-to-needle time.

(9) Compute arithmetic, geometric, and harmonic means of {2, 4, 8}; verify the mean inequality.

(10) Sketch when you would prefer Mann–Whitney over a two-sample t-test for comparing mRS between two eras.

(11) Define KL(p||q) and cross-entropy H(p,q); show algebraically that H(p,q) = H(p) + KL(p||q) for discrete finite supports.

(12) In one paragraph each, explain what the E-step and M-step accomplish in a two-component univariate Gaussian mixture.
