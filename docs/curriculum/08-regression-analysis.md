# Chapter 8. Regression Analysis


![08 Regression Fit](../assets/figures/08_regression_fit.png)


## Opening

A lab wants to regress 90-day mRS from admission labs. Linear models still discipline thinking about targets, residuals, and collinearity before anyone reaches for a neural net.


![Reliability of numeric predictions matters as much as fit (original).](../assets/figures/ml_fig_calibration.png)

*Reliability of numeric predictions matters as much as fit (original).*
## Learning Objectives

Distinguish loss, cost, and objective functions and write them for squared error and log loss.

Derive and compute ordinary least squares for univariate and multiple linear regression with a full numerical example.

Build polynomial and piecewise regression models and evaluate with RSE, RMSE, R², and residual diagnostics.

Explain ARIMA(p,d,q) intuition for clinical and epidemiologic time series.

Formulate logistic and softmax regression, interpret coefficients, and evaluate with ROC, calibration, and information criteria.

Use cross-validation, learning curves, Wald tests, likelihood-ratio tests, and pseudo-R² appropriately.

Analyze overfitting, underfitting, and the bias–variance tradeoff; apply Ridge, Lasso, elastic net, and non-negative garrote.

Connect optimization mathematics (derivatives, gradients, Jacobian, Hessian, Taylor) to GD variants, Newton methods, and early stopping.

## Loss, Cost, and Objective Functions

Supervised learning fits a parametric mapping f(x; θ) from features x to a target y by optimizing a numerical criterion. Terminology varies across communities, but a useful distinction is: a loss ℓ(y, f(x; θ)) measures error on a single example; a cost (or empirical risk) averages loss over a training set, J(θ) = (1/n) ∑_i ℓ(y_i, f(x_i; θ)); an objective may add regularization, constraints, or priors, e.g. J(θ) + λ R(θ). Training solves θ̂ = argmin_θ of that objective (or a stochastic approximation thereof).

Squared error loss ℓ = (y − ŷ)² underpins classical linear regression and is differentiable and sensitive to outliers. Absolute error |y − ŷ| is more robust. Huber loss blends both—quadratic for residuals below a threshold δ and linear beyond it—so it stays smooth and differentiable near zero yet down-weights the leverage of gross outliers. For binary classification with probabilistic scores p ∈ (0,1), log loss (binary cross-entropy) ℓ = −[y log p + (1−y) log(1−p)] is the negative Bernoulli log-likelihood. Multi-class cross-entropy generalizes to softmax probabilities. Choosing a loss is choosing what “wrong” means clinically: large errors on huge ICH volumes may matter more than small errors near zero—or the reverse if threshold decisions dominate.

Objectives are not automatically causal estimands. Minimizing prediction error can use any feature known at prediction time; estimating a treatment effect requires a design and often a different target parameter. State the estimand before celebrating a low training loss. In multi-site stroke research, write the loss, the decision time, and the population on the same page of the analysis plan so optimization choices cannot silently redefine the scientific question.

## Univariate and Multiple Linear Regression

Univariate linear regression models E[y | x] = β₀ + β₁ x for a scalar predictor. Given n pairs (x_i, y_i), ordinary least squares (OLS) minimizes ∑_i (y_i − β₀ − β₁ x_i)². Closed form: β₁ = ∑ (x_i − x̄)(y_i − ȳ) / ∑ (x_i − x̄)², β₀ = ȳ − β₁ x̄. Residuals e_i = y_i − ŷ_i should look structureless if the linear mean and constant variance are plausible.

Multiple linear regression extends to a feature vector x ∈ R^p: y ≈ β₀ + xᵀ β, or in matrix form y = Xβ + ε with design matrix X including a column of ones for the intercept. If X has full column rank, the OLS estimator is β̂ = (Xᵀ X)⁻¹ Xᵀ y, equivalently solving the normal equations Xᵀ X β = Xᵀ y. Fitted values are ŷ = X β̂ = H y with hat matrix H = X(Xᵀ X)⁻¹ Xᵀ. Multicollinearity makes Xᵀ X ill-conditioned and coefficient variances large even when predictions remain usable.

### Worked Example: OLS by Hand

Four patients with a single predictor x = admission NIHSS and outcome y = infarct volume (simplified units):
(x,y): (4, 12), (8, 20), (10, 22), (14, 30).
Means: x̄ = (4+8+10+14)/4 = 9, ȳ = (12+20+22+30)/4 = 21.
Deviations (x−x̄): −5, −1, +1, +5. Deviations (y−ȳ): −9, −1, +1, +9.
∑ (x−x̄)(y−ȳ) = (−5)(−9)+(−1)(−1)+(1)(1)+(5)(9) = 45+1+1+45 = 92.
∑ (x−x̄)² = 25+1+1+25 = 52.
β₁ = 92/52 = 23/13 ≈ 1.769. β₀ = 21 − (23/13)·9 = 21 − 207/13 = (273 − 207)/13 = 66/13 ≈ 5.077.

![8.1: The worked ordinary-least-squares fit of infarct volume on admission NIHSS for the four points (4, 12), (8, 20), (10, 22](../assets/figures/ml_concept_8.1_f939d805.png)

*Figure 8.1 — original teaching graphic.*

![OLS fit for the four-point NIHSS–volume example (original).](../assets/figures/ml_fig_ols_fit.png)

*Exact OLS line ŷ = 66/13 + (23/13)x with residual segments; RSS ≈ 1.23, R² ≈ 0.992 (original).*

Fitted line: ŷ = 66/13 + (23/13) x. At x = 8, ŷ = (66 + 184)/13 = 250/13 ≈ 19.23 (residual 20 − 19.23 ≈ 0.77). At x = 14, ŷ = (66 + 322)/13 = 388/13 ≈ 29.85 (residual ≈ 0.15). RSS = ∑ e_i² can be computed from all four residuals for RSE below. The slope says that each additional NIHSS point associates with about 1.77 volume units higher expected infarct size in this toy sample—not a causal claim about NIHSS raising volume.

### Choosing Variables and Challenges

Variable inclusion may be pre-specified by protocol (preferred in epi), guided by domain knowledge, or data-driven (stepwise, penalization, wrappers from Chapter 6). Data-driven selection inflates type I error and optimistically biases performance unless nested in validation. Challenges include nonlinearity, interactions, heteroscedasticity, outliers, missingness, and endogeneity when coefficients are interpreted causally. Solutions include transforms, splines, robust SEs, mixed models for clustering, and explicit causal designs when effects—not forecasts—are the goal.

## Polynomial and Piecewise Regression

Polynomial regression remains linear in parameters but nonlinear in x: y ≈ β₀ + β₁ x + β₂ x² + ⋯ + β_p x^p. The OLS machinery applies to the expanded design matrix. Higher p fits wiggles that can interpolate noise—especially dangerous with small n. Regularization, low degree (p = 2 or 3), or domain transforms (log onset-to-arrival) are safer than high-degree global polynomials on clinical tables. Interactions are cross-basis terms: β_{12} x₁ x₂ allows the effect of x₁ to depend on x₂ (age × NIHSS).

Piecewise (segmented) regression allows different linear (or polynomial) regimes on intervals of x, joined or discontinuous at knots. Continuous piecewise linear models (linear splines) hinge at knots; cubic splines and restricted cubic splines are standard in epidemiology for flexible age and severity effects without the wild tail behavior of global high-degree polynomials. Generalized additive models (GAMs) sum smooth functions of individual predictors, g(E[y]) = β₀ + f₁(x₁) + f₂(x₂) + ⋯, staying additive across predictors so each effect remains a visualizable curve; interactions are added deliberately as joint smooths rather than assumed everywhere. Choose knot locations by quantiles or clinical thresholds (e.g., NIHSS cut-points used in trials), and penalize smoothness to avoid overfit.

## Evaluating Fit: RSE, RMSE, and Residuals

Residual Standard Error (RSE) estimates the standard deviation of the error term in a linear model: RSE = √(RSS / (n − p − 1)) for a model with intercept and p slopes (degrees of freedom n − p − 1). It is roughly the typical size of residuals in the units of y. Root Mean Square Error RMSE = √( (1/n) ∑ (y_i − ŷ_i)² ) (or with n−1) is closely related and widely reported for prediction; on held-out data it is a primary accuracy metric.

R² = 1 − RSS/TSS measures the fraction of sample variance in y linearly associated with the fitted mean. It never decreases when adding predictors; adjusted R² penalizes dimension slightly. R² is not a validation metric by itself. Report RMSE or MAE on held-out data in clinical units (mL, days). Residual-versus-fitted plots diagnose nonlinearity and heteroscedasticity; QQ plots assess approximate normality for small-sample inference.

## ARIMA: Autoregressive Integrated Moving Average

Time series regression forecasts future values from past values and past shocks. ARIMA(p,d,q) combines: Integration (d)—differencing the series d times to approach stationarity (d = 1 removes a random-walk trend); Autoregression AR(p)—linear dependence on p lagged values; Moving average MA(q)—linear dependence on q lagged forecast errors. The model is not magic: it assumes approximate linearity, stable dynamics, and adequate history.

Intuition for orders: d is chosen so the differenced series looks stationary (ADF tests, ACF decay). p is suggested by partial autocorrelation (PACF) cutoffs for AR-like series. q is suggested by autocorrelation (ACF) cutoffs for MA-like series. Information criteria (AIC/BIC) compare candidate (p,d,q) on training data; true forecast skill needs time-series cross-validation or rolling origins—not shuffled K-fold. Seasonal ARIMA adds seasonal AR, differencing, and MA terms for weekly ED volumes or annual cycles.

Extrapolation predicts beyond the last time point; interpolation fills gaps—ARIMA-style models can do both with care. Clinical uses include forecasting bed occupancy, stroke code volumes, and physiologic series. External shocks (pandemic, new EVT criteria) break stationarity; monitor forecast residuals and re-estimate when processes change.

## Logistic Regression and Softmax

Logistic regression models a binary outcome with the logit link: P(y = 1 | x) = σ(xᵀ β) where σ(z) = 1/(1+e^{−z}) and x includes 1 for the intercept. The log-odds equal the linear predictor: log(p/(1−p)) = xᵀ β. Coefficients are log-odds ratios: a unit increase in x_j multiplies odds by e^{β_j} holding other covariates fixed—interpret cautiously with collinearity and non-collapsibility of odds ratios.

Parameters are estimated by maximizing the Bernoulli likelihood (equivalently minimizing log loss), typically via Newton–Raphson / IRLS or gradient methods. There is no OLS-style simple closed form in general. Perfect separation makes MLE infinite; Firth penalty or L2 regularization stabilizes estimates—common in small clinical datasets.

Softmax (multinomial logistic) regression extends to K classes: P(y = k | x) = exp(xᵀ β_k) / ∑_{j=1}^K exp(xᵀ β_j), with an identifiability constraint (e.g., β_K = 0). Training minimizes multi-class cross-entropy. Softmax is the classification head of many neural nets; the classical version with linear predictors remains a strong baseline for multi-class stroke outcomes (e.g., mRS binned, TOAST categories) when n is modest and features are tabular.

### Worked Example: Logistic Regression Prediction and One Gradient Step

Consider one patient described by two standardized features, x = [x₁, x₂] = [1.0, 0.5], where x₁ is a scaled admission NIHSS and x₂ a scaled age, and a fitted model with coefficients β = [β₀, β₁, β₂] = [−1.0, 0.8, 0.4]; the intercept multiplies a constant x₀ = 1. The true label is y = 1 (the event occurred).

![8.2: The logistic sigmoid σ(z) = 1/(1 + e^−z), mapping the linear predictor z = xᵀβ (the log-odds) to a probability. At the c](../assets/figures/ml_concept_8.2_2853a7e2.png)

*Figure 8.2 — original teaching graphic.*

Linear predictor: z = β₀·1 + β₁·x₁ + β₂·x₂ = −1.0 + 0.8·(1.0) + 0.4·(0.5) = −1.0 + 0.8 + 0.2 = 0.0.

Probability: p = σ(z) = 1/(1 + e^−z) = 1/(1 + e^0) = 1/(1 + 1) = 0.5. At z = 0 the logistic curve is exactly at its midpoint, so this patient sits on the decision boundary and the model predicts a coin flip.

Log-loss for the true label y = 1: ℓ = −ln p = −ln(0.5) = ln 2 ≈ 0.693. That is the loss of an uninformative 50/50 guess—the reference value any useful model must beat.

Now take one gradient-descent step. The per-example gradient of log-loss with respect to βⱼ is (p − y)·xⱼ, with x₀ = 1. Here p − y = 0.5 − 1 = −0.5, so the three partials are (p−y)·x₀ = (−0.5)(1) = −0.5, (p−y)·x₁ = (−0.5)(1.0) = −0.5, and (p−y)·x₂ = (−0.5)(0.5) = −0.25. Because the prediction (0.5) undershoots the truth (1), every component is negative, so descent will raise the coefficients on the positive features and push z upward. With learning rate η = 0.5 and the rule βⱼ ← βⱼ − η·(p−y)·xⱼ:

β₀ ← −1.0 − 0.5·(−0.5) = −1.0 + 0.25 = −0.75

β₁ ← 0.8 − 0.5·(−0.5) = 0.8 + 0.25 = 1.05

β₂ ← 0.4 − 0.5·(−0.25) = 0.4 + 0.125 = 0.525

Recompute the linear predictor with the updated coefficients: z′ = −0.75 + 1.05·(1.0) + 0.525·(0.5) = −0.75 + 1.05 + 0.2625 = 0.5625. Using e^0.5625 ≈ 1.755 (so e^−0.5625 ≈ 1/1.755 ≈ 0.570; a coarser check is e^0.6 ≈ 1.822), the new probability is p′ = σ(0.5625) = 1/(1 + 0.570) = 1/1.570 ≈ 0.637. The prediction moved from 0.500 to 0.637—toward the true label y = 1—and the log-loss fell from 0.693 to −ln(0.637) ≈ 0.451. One step does not arrive at the target; the optimizer repeats this until the objective stops improving.

### Worked Example: Ridge Shrinkage on One Coefficient

The section above invoked L2 regularization to stabilize logistic coefficients under separation; ridge is that same L2 penalty, and its mechanics are clearest in one dimension with squared-error loss. The one-dimensional ridge estimator (no intercept) has the closed form β̂ = (Σ xᵢyᵢ)/(Σ xᵢ² + λ)—the least-squares slope with λ added to the denominator. Take three standardized toy observations (xᵢ, yᵢ) = (1, 2), (2, 4), (3, 6), chosen to lie exactly on the line y = 2x so that shrinkage is the only thing moving the estimate. The needed sums are Σ xᵢyᵢ = (1)(2) + (2)(4) + (3)(6) = 2 + 8 + 18 = 28 and Σ xᵢ² = 1² + 2² + 3² = 1 + 4 + 9 = 14.

At λ = 0 (no penalty) the estimator is ordinary least squares: β̂ = 28/14 = 2.00, exactly the true slope. At λ = 10, β̂ = 28/(14 + 10) = 28/24 = 7/6 ≈ 1.17. Equivalently, ridge multiplies the OLS slope by the shrinkage factor Σ xᵢ²/(Σ xᵢ² + λ) = 14/24 ≈ 0.583, so 2.00 × 0.583 ≈ 1.17.

The penalty pulled a perfectly-supported slope from 2.00 down to 1.17—a 42% reduction—trading a little bias for lower variance. As λ → 0 the estimate returns to OLS; as λ → ∞ the denominator dominates and β̂ → 0. A larger Σ xᵢ² (more spread, hence more signal, in the predictor) resists a fixed λ, which is precisely why predictors are standardized before a shared penalty is applied.

## Evaluating Regression and Classification Fitness

### Cross-Validation and Learning Curves

K-fold cross-validation partitions training data into K folds, repeatedly fitting on K−1 folds and evaluating on the held fold. Nested CV is required when selection and hyperparameter tuning occur. For temporal clinical data, prefer forward-chaining or rolling-origin validation over random folds that leak future practice patterns. Learning curves plot training and validation error against training set size: high bias shows both errors high and close; high variance shows a large gap with low training error and high validation error.

### ROC for Logistic Models

Receiver Operating Characteristic (ROC) curves plot true positive rate versus false positive rate as the classification threshold on predicted probability sweeps. AUC summarizes ranking quality. ROC ignores calibration: two models with identical AUC can have very different probability reliability. For clinical decisions, report calibration plots, the Brier score (the mean squared error between predicted probabilities and 0/1 outcomes, a joint measure of calibration and discrimination), and threshold-specific net benefit (from decision-curve analysis, which weighs true positives against false positives at a clinically chosen threshold) alongside AUC. Precision–recall curves are more informative under rare events (sICH).

![Brier score Murphy decomposition: reliability, resolution, uncertainty (synthetic; original).](../assets/figures/ml_fig_brier_decomp.png)

*Figure — Brier components. Murphy’s decomposition writes BS = REL − RES + UNC. **Left:** an overconfident model inflates reliability (miscalibration penalty); a near-constant predictor is “calibrated” to prevalence but has almost no resolution—low Brier for the wrong reason. **Right:** reliability diagrams for the same labels. Always pair Brier with a reliability plot and a discrimination metric; Brier is not a causal effect and is prevalence-dependent via UNC = π(1−π).*

![Residual histogram and QQ plot for heavy-tailed errors (synthetic; original).](../assets/figures/ml_fig_residual_qq.png)

*Figure — Gaussian residual assumptions are checkable. **Left:** heavy-tailed residual histogram. **Right:** QQ plot bows in the tails. Prefer robust SEs, transforms, or GLMs when tails dominate; diagnostics police intervals—not causation.*

![8.3: Discrimination is not calibration. (a) An ROC curve traced by sweeping the probability threshold, with the shaded area u](../assets/figures/ml_concept_8.3_6c89e5db.png)

*Figure 8.3 — original teaching graphic.*

### Wald Tests, Information Criteria, R² and Pseudo-R², LRT

Wald tests assess H₀: β_j = 0 using β̂_j / SE(β̂_j) compared to a normal or t reference (asymptotically). They are convenient in software output but can misbehave under separation and small samples. Likelihood-ratio tests (LRT) compare nested models via 2(ℓ_full − ℓ_reduced) ~ χ² with degrees of freedom equal to the difference in free parameters—often more reliable than Wald in logistic models.

Information criteria balance fit and complexity: AIC ≈ −2ℓ + 2k, BIC ≈ −2ℓ + k log n. Lower is better within a comparable likelihood family; they are not substitutes for external validation. Pseudo-R² measures summarize logistic fit relative to a null (intercept-only) model, but on different scales. McFadden’s is 1 − ℓ_full/ℓ_null, a ratio of log-likelihoods on which even strong models rarely exceed about 0.4; Nagelkerke rescales the Cox–Snell likelihood-ratio index by its maximum attainable value so the number can reach 1; Tjur’s coefficient of discrimination is the intuitive gap between mean predicted probability among events and among non-events; none is “percent variance explained” in the OLS sense—report them carefully and prefer predictive metrics on holdout data for model choice in clinical prediction.

## Overfitting, Underfitting, and the Bias–Variance Tradeoff

Underfitting (high bias) means the hypothesis class cannot represent the true regression function: a straight line through curved volume–NIHSS data, or too heavy regularization. Overfitting (high variance) means the model captures noise and sample idiosyncrasies: a high-degree polynomial through few points, unpenalized models with p ≈ n. Expected prediction error decomposes (for squared error) into bias² + variance + irreducible noise. Increasing model flexibility tends to decrease bias and increase variance; regularization and more data rebalance the tradeoff.

![8.4: Underfitting, good fit, and overfitting shown by fitting the same 13 noisy points with polynomials of degree 1, 3, and 1](../assets/figures/ml_concept_8.4_20d7c49c.png)

*Figure 8.4 — original teaching graphic.*

Clinical n is often small relative to candidate features. Prefer pre-specified predictors, penalization, dimensionality control, and honest temporal validation. A model that memorizes one comprehensive stroke center’s documentation style will not transport to a telestroke network.

## Regularization: Ridge, Lasso, Elastic Net, and Non-Negative Garrote

Regularization adds a penalty R(β) to the cost to stabilize estimates and implement inductive bias. Standardize predictors before applying a shared penalty weight λ so that units do not silently dominate.

Ridge (L2) minimizes ‖y − Xβ‖² + λ ‖β‖₂² (intercept typically unpenalized). Solutions shrink coefficients continuously toward zero but rarely to exact zero; multicollinearity is handled gracefully by stabilizing XᵀX + λI. Closed form: β̂_ridge = (XᵀX + λI)⁻¹ Xᵀ y.

Lasso (L1) minimizes ‖y − Xβ‖² + λ ‖β‖₁, promoting sparse solutions with exact zeros—built-in variable selection. Correlated feature groups may see arbitrary single-feature selection. Elastic net combines L1 and L2 penalties: λ₁ ‖β‖₁ + λ₂ ‖β‖₂², encouraging sparsity while grouping correlated predictors more stably—often preferable for correlated clinical labs and comorbidity indicators.

The non-negative garrote starts from an initial estimate (often OLS) β̃ and finds nonnegative shrinkage factors c_j ≥ 0 minimizing a least-squares criterion in the scaled coefficients c_j β̃_j with a penalty on ∑ c_j. It sparsifies and shrinks relative to a good initial fit. Historically important as a bridge between subset selection and modern L1 methods; still useful conceptually when coefficients must remain nonnegative after scaling an initial model.

Ridge: stable, dense solutions; good under multicollinearity.

Lasso: sparse solutions; variable selection; unstable under highly correlated clones.

Elastic net: compromise for grouped clinical features.

Non-negative garrote: shrink/select relative to an initial estimate with c_j ≥ 0.

Always scale features before a shared λ; choose λ by nested CV or temporal holdout.

## Optimization Mathematics for Model Fitting

Fitting differentiable models is numerical optimization of J(θ). The derivative dJ/dθ for scalar θ and the gradient ∇J(θ) for vector θ give first-order local linear approximations: J(θ + δ) ≈ J(θ) + ∇J(θ)ᵀ δ. Critical points satisfy ∇J = 0. The Jacobian matrix collects first partials of a vector-valued function; for a scalar objective the Jacobian of the residual vector appears in least-squares Gauss–Newton methods.

The Hessian H = ∇²J is the matrix of second partials. Taylor expansion to second order: J(θ + δ) ≈ J(θ) + ∇Jᵀ δ + (1/2) δᵀ H δ. Positive definite H at a critical point implies a local minimum for smooth objectives. Conditioning of H governs how elongated valleys are and how hard first-order methods struggle.

### Gradient Descent Variants

Batch gradient descent updates θ ← θ − η ∇J(θ) using the full training set gradient each step. It has low-noise gradients but is expensive for large n. Stochastic gradient descent (SGD) uses one example (or a random shuffle stream) per update: noisy but cheap and often better at escaping shallow local structure in nonconvex deep models. Mini-batch SGD uses b examples per step, the practical default: vectorized hardware efficiency with controlled noise. Learning rate η (and schedules, momentum, adaptive methods like Adam in later chapters) dominate empirical success.

![8.5: Batch gradient descent on a convex quadratic loss whose contours form an ill-conditioned, elongated bowl. Iterates follo](../assets/figures/ml_concept_8.5_e62f0349.png)

*Figure 8.5 — original teaching graphic.*

![Gradient noise: batch GD versus mini-batch SGD paths on a synthetic elongated quadratic (original).](../assets/figures/ml_fig_gradient_noise.png)

*Figure — Gradient noise. Full-batch steps follow a smooth trajectory; mini-batch gradients jitter with variance that scales roughly as 1/√batch size. That noise is both a cost (slower exact convergence) and a feature (exploration in nonconvex landscapes).*

### Newton Methods

Newton’s method uses curvature: θ ← θ − H⁻¹ ∇J. For smooth convex problems near the optimum, convergence is quadratic. Cost and need for Hessian inverses limit pure Newton on huge parameter spaces; quasi-Newton (BFGS/L-BFGS) and Gauss–Newton/IRLS for least squares and GLMs approximate curvature more cheaply. Logistic regression’s classical IRLS is a Newton-style algorithm on the log-likelihood.

### Early Stopping

Early stopping monitors validation loss during iterative optimization and stops when validation performance degrades while training loss still falls—an implicit regularizer. It is essential in neural training and also useful for path algorithms in penalized regression. Requires a validation stream that does not leak into final reported test metrics (use a three-way train/validation/test or nested scheme).

![Early stopping: validation loss bottoms while training loss continues to fall (synthetic; original).](../assets/figures/ml_fig_early_stopping.png)

*Figure — Early stopping. The dashed epoch is the first minimum of validation loss; continuing further improves training fit while generalization worsens. Patience is chosen on validation only—never by peeking at final test performance.*

**Early-stopping protocol (teaching table)**

| Step | Action | Honest practice |
|------|--------|-----------------|
| 1 | Split train / validation / test (or nested CV) | Group by patient and site; respect time |
| 2 | Track validation loss (or primary clinical metric) each epoch | Do not open the test set while choosing patience |
| 3 | Stop when val metric fails to improve for *p* epochs | Document *p*; common defaults 5–20 depending on schedule |
| 4 | Restore weights from best validation epoch | Or refit train+val with fixed step count—write it down |
| 5 | Report final metrics once on held-out test | One look; no further tuning |

## Extended OLS Worked Numbers and Prediction Intervals

Continuing the four-point NIHSS–volume example with β₀=66/13 and β₁=23/13, fitted values are:
x=4: ŷ=(66+92)/13=158/13≈12.15, e=12−12.15=−0.15
x=8: ŷ=250/13≈19.23, e=20−19.23=0.77
x=10: ŷ=(66+230)/13=296/13≈22.77, e=22−22.77=−0.77
x=14: ŷ=388/13≈29.85, e=30−29.85=0.15
RSS ≈ (−0.15)²+(0.77)²+(−0.77)²+(0.15)² ≈ 0.0225+0.5929+0.5929+0.0225 ≈ 1.231.
RSE = √(RSS/(4−2)) = √(1.231/2) ≈ √0.6155 ≈ 0.78 volume units.
TSS = ∑(y−ȳ)² = 81+1+1+81=164; R²=1−RSS/TSS≈1−1.231/164≈0.992—suspiciously high because n=4 and the design is nearly linear.

A mean prediction at a new x₀ is ŷ₀=β₀+β₁ x₀. Approximate confidence intervals for the mean response tighten near x̄ and widen as x₀ moves away (leverage). Prediction intervals for a new individual observation are wider still because they include residual noise σ². In clinical communication, distinguish “expected average volume for NIHSS=10” from “plausible volume for the next patient with NIHSS=10.” Tiny toy R² values do not license narrow clinical promises.

## Multiple Regression Diagnostics and Influence

Beyond RSE and R², competent regression practice inspects leverage and influence. Diagonal elements H_{ii} of the hat matrix measure leverage: high-leverage patients have unusual x patterns (extreme age–severity combinations) and pull the fit. Cook’s distance and DFBETAS quantify how much β̂ changes when a point is deleted. In stroke registries, influential points may be data errors (weight entered in pounds as kilograms) or the very severe cases you most need—review clinically before deleting.

Variance inflation factors (VIF) diagnose multicollinearity: VIF_j = 1 / (1 − R_j²) from regressing feature j on the others. Large VIF inflates coefficient SEs without necessarily destroying predictive RMSE. Remedies include combining features, penalization (Ridge), or accepting wide CIs for individual β_j while focusing on prediction. Partial residual plots help see whether a linear term in glucose is adequate or whether a spline is needed.

Heteroscedasticity—error variance growing with fitted volume or LOS—is common. OLS β̂ remains unbiased under correct mean specification, but classical SEs are wrong; use Huber–White robust SEs, weighted least squares, or transform y. For prediction, consider models that explicitly allow variance to depend on x (e.g., GLMs with gamma errors for LOS).

## From Linear Predictors to GLM Thinking

Generalized linear models unify many regression-like structures: a linear predictor η = xᵀ β, a link function relating η to the mean of y, and an exponential-family noise model. Linear regression uses identity link and Gaussian noise. Logistic regression uses logit link and Bernoulli noise. Poisson and negative binomial models suit counts (readmissions, seizure counts). Once you understand loss, linear predictors, regularization, and validation from this chapter, other GLMs are variations on likelihood and link—not an entirely new philosophy.

Survival and Cox models link to the same culture for time-to-event neurologic outcomes with censoring: a linear predictor sits inside a hazard h(t|x) = h₀(t) exp(xᵀ β). Partial likelihood yields log hazard ratios. Naive OLS on observed times that ignore censoring is biased. Modern ML extends Cox with neural hazards and random survival forests; the epidemiologic censoring discipline remains.

![Survival C-index: pairwise concordance under censoring (synthetic; original).](../assets/figures/ml_fig_cindex_pairs.png)

*Figure — C-index intuition. **Left:** higher predicted risk should co-occur with shorter observed times; censored points (squares) limit which pairs are comparable. **Right:** Harrell’s C counts concordant vs discordant comparable pairs (event *i* before *j*’s follow-up). C ranks risk, does not calibrate absolute survival probabilities, and is not a treatment-effect estimate—report number of events and censoring pattern with any C.*

## Regularization Paths and Choosing λ

As λ goes from large to small along a Lasso path, coefficients enter the model at values where their correlation with residuals overcomes the L1 penalty—producing a sequence of nested sparse models. Cross-validated λ_min minimizes average validation loss; λ_1se (one-standard-error rule) picks a sparser model within one SE of the minimum, often more stable for clinical reporting. Plot coefficient paths against log λ and mark selected values for transparency.

![8.6: Regularization coefficient paths for five standardized predictors as the penalty increases along log₁₀λ. Ridge (L2, left](../assets/figures/ml_concept_8.6_fdd46bf5.png)

*Figure 8.6 — original teaching graphic.*

![Lasso and Ridge regularization paths for five standardized predictors vs log₁₀λ (synthetic; original).](../assets/figures/ml_fig_regularization_path.png)

*Figure — Regularization path. Lasso (left) yields exact zeros and staggered feature entry; Ridge (right) shrinks coefficients continuously without hard selection. Mark λ_min and λ_1se on the path you report so readers can see the sparsity–fit tradeoff.*


![Cook's distance influence diagnostics for synthetic OLS (original).](../assets/figures/ml_fig_cooks_distance.png)

*Figure — Influence vs bulk fit. Left: golden points pull the line; right: Cook's D with a 4/n reference. High influence warrants investigation (data error, rare phenotype), not automatic deletion. **Influence ≠ causation**.*


![Partial dependence can look causal when features collinear (synthetic; original).](../assets/figures/ml_fig_pdp_collinearity.png)

*Figure — PDP caution. Left: x2 tracks x1; right: marginal PDP in x2 still slopes. Dependence confounds partial plots—**curves are not causal effects** without assumptions.*


![Normal QQ plot of synthetic OLS residuals (original).](../assets/figures/ml_fig_qq_normal.png)

*Figure — Residual normality check. Departures from the diagonal flag assumption stress for inference. Diagnostics improve modeling honesty—they do not prove causal effects.*


![GLM link functions: logit and log mean maps (original).](../assets/figures/ml_fig_glm_links.png)

*Figure — Link functions map linear predictors to means. Coefficient interpretation depends on the link. Modeling choices precede any careful causal contrast.*


![Heteroscedastic residual fan vs fitted values (synthetic; original).](../assets/figures/ml_fig_hetero_resid.png)

*Figure — Residual variance growing with fitted values flags OLS assumption stress. Consider WLS/robust SEs. Diagnostics improve honesty—not automatic causal ID.*


![VIF bars for multicollinearity (synthetic; original).](../assets/figures/ml_fig_vif_bars.png)

*Figure — Estimation fragility—not a causal DAG. Pred ≠ cause without design.*


![Ridge coefficient shrink vs lambda (original).](../assets/figures/ml_fig_ridge_trace.png)

*Figure — Shrinkage stabilizes OLS under collinearity. Ridge coefficient shrink vs lambda Pred != cause without design.*


![addedvar teaching panel (original).](../assets/figures/ml_fig_added_variable.png)

*Figure — Teaching panel for addedvar. Pred != cause without design.*


![Cycle-34 densify scientific panel 10 (original).](../assets/figures/ml_fig_c34_09.png)

*Figure — Continuous densify panel 10. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 10 (original).](../assets/figures/ml_fig_c35_09.png)

*Figure — Continuous densify panel 10. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 10 (original).](../assets/figures/ml_fig_c36_09.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 10 (original).](../assets/figures/ml_fig_c37_09.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 10 (original).](../assets/figures/ml_fig_c38_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 10 (original).](../assets/figures/ml_fig_c39_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 10 (original).](../assets/figures/ml_fig_c40_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 10 (original).](../assets/figures/ml_fig_c41_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 10 (original).](../assets/figures/ml_fig_c42_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 10 (original).](../assets/figures/ml_fig_c43_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 10 (original).](../assets/figures/ml_fig_c44_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 10 (original).](../assets/figures/ml_fig_c45_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 10 (original).](../assets/figures/ml_fig_c46_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 10 (original).](../assets/figures/ml_fig_c47_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 10 (original).](../assets/figures/ml_fig_c48_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 10 (original).](../assets/figures/ml_fig_c49_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 10 (original).](../assets/figures/ml_fig_c50_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 10 (original).](../assets/figures/ml_fig_c51_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 10 (original).](../assets/figures/ml_fig_c52_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 10 (original).](../assets/figures/ml_fig_c53_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 10 (original).](../assets/figures/ml_fig_c54_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 10 (original).](../assets/figures/ml_fig_c55_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 10 (original).](../assets/figures/ml_fig_c56_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 10 (original).](../assets/figures/ml_fig_c57_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 10 (original).](../assets/figures/ml_fig_c58_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 10 (original).](../assets/figures/ml_fig_c59_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 10 (original).](../assets/figures/ml_fig_c60_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 10 (original).](../assets/figures/ml_fig_c61_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 10 (original).](../assets/figures/ml_fig_c62_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 10 (original).](../assets/figures/ml_fig_c63_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 10 (original).](../assets/figures/ml_fig_c64_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 10 (original).](../assets/figures/ml_fig_c65_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 10 (original).](../assets/figures/ml_fig_c66_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 10 (original).](../assets/figures/ml_fig_c67_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 10 (original).](../assets/figures/ml_fig_c68_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 10 (original).](../assets/figures/ml_fig_c69_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 10 (original).](../assets/figures/ml_fig_c70_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 10 (original).](../assets/figures/ml_fig_c71_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 10 (original).](../assets/figures/ml_fig_c72_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 10 (original).](../assets/figures/ml_fig_c73_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 10 (original).](../assets/figures/ml_fig_c74_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 10 (original).](../assets/figures/ml_fig_c75_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 10 (original).](../assets/figures/ml_fig_c76_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 10 (original).](../assets/figures/ml_fig_c77_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 10 (original).](../assets/figures/ml_fig_c78_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 10 (original).](../assets/figures/ml_fig_c79_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 10 (original).](../assets/figures/ml_fig_c80_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 10 (original).](../assets/figures/ml_fig_c81_09.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

Elastic net’s mixing parameter α (weight on L1 versus L2) is a second hyperparameter; nest its selection. Grouped clinical features (multiple BP meds, multi-item NIHSS) sometimes use group Lasso variants so that whole groups enter together—beyond this chapter’s core, but aligned with elastic net’s motivation. The non-negative garrote’s c_j path similarly traces shrinkage of an initial fit; if the initial OLS is already nonsense due to p > n, start from Ridge or univariate screens instead.

## Optimization Practice Notes

Feature scaling is an optimization issue as well as a statistical one: without scaling, gradient steps zigzag in elongated valleys of J, requiring tiny η. Conditioning of XᵀX mirrors the Hessian of the least-squares objective. Mini-batch size trades noise for hardware efficiency; very small batches approximate SGD, very large batches approach batch GD. Shuffle clinical data carefully: random shuffles are fine for i.i.d. tables but wrong for time-ordered streaming ARIMA-style fits.

Numerical stability matters: log-sum-exp tricks for softmax, clipping probabilities away from 0/1 in log loss, and using solvers (coordinate descent for Lasso, L-BFGS for smooth logistic) rather than hand-rolled GD when libraries suffice. Early stopping patience (e.g., stop if validation loss fails to improve for p epochs) should be tuned on validation only; after selection, some teams refit on train+validation with fixed steps—document the protocol.

Mini-batch gradient descent on a logistic stroke model with n=5000 might use batch size 64, learning rate 0.05 with decay, and early stopping on a temporal validation year. Batch GD would compute exact gradients but waste epochs on redundant passes; pure SGD would be noisy and slower on vectorized hardware. Newton/IRLS may converge in tens of iterations for moderate p but needs care when p is large or features are collinear—hence L2 ridge inside the GLM.

Taylor intuition: near a minimum, J looks quadratic; gradient descent with small η crawls along the valley floor, while Newton jumps using curvature. Far from the minimum, the quadratic model is wrong—hence trust regions and line search in production optimizers. You need not implement BFGS to use it wisely: know that smooth convex logistic problems are well handled by second-order or quasi-Newton methods, while deep nonconvex nets (later chapters) live in the SGD world with early stopping as a first-class regularizer.

Putting optimization and statistics together: the same logistic likelihood can be maximized by IRLS, L-BFGS, or mini-batch SGD; the estimator’s statistical properties depend on the model and data, not on which convergent solver you picked. Solver failure (non-convergence, overflow) is a different problem from statistical bias. Always check convergence flags before interpreting coefficients or shipping probabilities to a bedside display.

A short ARIMA scenario: weekly ischemic stroke admissions y_t over two years. First differences (d=1) remove a slow level shift after a new thrombectomy service opens. ACF/PACF on differenced series suggest ARIMA(1,1,1). Fit on the first 18 months, forecast 8 weeks, and compare RMSE to a naive seasonal-naive baseline. If a pandemic shock hits, residuals explode—re-estimate or add exogenous regressors rather than trusting a frozen (p,d,q). Document the forecast origin and never shuffle weeks when scoring temporal models.

## Clinical and Epidemiologic Notes

Sample size and events-per-variable heuristics still discipline regression in registries: overfit models with dozens of free parameters on 80 ICH cases will not externally validate. Prefer pre-specified predictors, penalization, or dimensionality control. Multi-level structure (patients in hospitals) calls for mixed-effects models or cluster-robust variance when inference is the goal; pure prediction may use site fixed effects or site recalibration.

Confounding versus prediction: a predictive model may include any feature known at prediction time; an etiologic model requires causal design. Hospital site may be an excellent LOS predictor but a confounder in a multi-center drug study. When writing methods, state whether the regression is prognostic, descriptive, or causal.

Calibration of continuous and probabilistic predictions matters for decisions. Plot average y versus average ŷ in bins; for logistic models, calibration curves and intercept/slope refresh on new sites may be needed. For time-to-event endpoints with censoring, use survival methods (Cox and extensions) rather than naive OLS on observed times.

Targets such as infarct volume are right-skewed; log1p transforms and gamma-like GLMs often behave better than raw OLS. LOS is discrete-ish, skewed, and censored by death or transfer. Continuous biomarkers have measurement error that attenuates etiologic slopes (regression dilution). Publish predictive performance with confidence intervals from appropriate resampling or prospective evaluation, and avoid equating a significant β with clinical utility.

State decision time and estimand before choosing predictors and loss.

Report RMSE/MAE in clinical units and show calibration, not only R².

Use penalization or fewer features when n is modest; nest tuning in CV.

Do not treat OLS coefficients as causal without a supporting design.

Match ARIMA validation to time order; match logistic metrics to prevalence and decisions.

Inspect leverage and influence; do not drop severe cases casually.

Document λ selection (λ_min vs λ_1se) and early-stopping rules.

Separate solver convergence issues from statistical bias and confounding.


![c82 teaching panel 09 (original).](../assets/figures/ml_fig_c82_09.png)
*Figure — Residual diagnostics: constant scatter vs heteroscedastic fan. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 09 (original).](../assets/figures/ml_fig_c83_09.png)
*Figure — Regularization coefficient paths as λ grows. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 09 (original).](../assets/figures/ml_fig_c84_09.png)
*Figure — Partial residuals reveal leftover nonlinear curvature. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 09 (original).](../assets/figures/ml_fig_c85_09.png)
*Figure — OLS residuals are vertical distances to the fitted line. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 09 (original).](../assets/figures/ml_fig_c86_09.png)
*Figure — Null association cloud with flat fitted slope. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 09 (original).](../assets/figures/ml_fig_c87_09.png)
*Figure — Polynomial terms introduce mean-function curvature. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 09 (original).](../assets/figures/ml_fig_c88_09.png)
*Figure — Regression leverage / hat-matrix diagonal. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 09 (original).](../assets/figures/ml_fig_c89_09.png)
*Figure — Collinear predictors inflate coefficient SE. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 09 (original).](../assets/figures/ml_fig_c90_09.png)
*Figure — Cook's distance influence markers. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 09 (original).](../assets/figures/ml_fig_c91_09.png)
*Figure — Huber loss vs squared loss. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 09 (original).](../assets/figures/ml_fig_c92_09.png)
*Figure — Quantile regression pinball loss. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 09 (original).](../assets/figures/ml_fig_c93_09.png)
*Figure — GAM smooth component stack. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 09 (original).](../assets/figures/ml_fig_c94_09.png)
*Figure — Elastic net diamond+disk constraint. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 09 (original).](../assets/figures/ml_fig_c95_09.png)
*Figure — Poisson GLM log-link mean. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 09 (original).](../assets/figures/ml_fig_c96_09.png)
*Figure — Spline basis stack for GAM. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 09 (original).](../assets/figures/ml_fig_c97_09.png)
*Figure — Quantile loss fan of tau. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 09 (original).](../assets/figures/ml_fig_c98_09.png)
*Figure — Negative binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 09 (original).](../assets/figures/ml_fig_c99_09.png)
*Figure — Cox partial likelihood idea. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 09 (original).](../assets/figures/ml_fig_c100_09.png)
*Figure — Tweedie compound Poisson. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 09 (original).](../assets/figures/ml_fig_c101_09.png)
*Figure — Zero-inflated outcome mass. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 09 (original).](../assets/figures/ml_fig_c102_09.png)
*Figure — AFT survival acceleration. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 09 (original).](../assets/figures/ml_fig_c103_09.png)
*Figure — Student-t robust regression. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 09 (original).](../assets/figures/ml_fig_c104_09.png)
*Figure — Hurdle model two-part. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 09 (original).](../assets/figures/ml_fig_c105_09.png)
*Figure — Competing risks cumulative. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 09 (original).](../assets/figures/ml_fig_c106_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 09 (original).](../assets/figures/ml_fig_c107_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 09 (original).](../assets/figures/ml_fig_c108_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 09 (original).](../assets/figures/ml_fig_c109_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 09 (original).](../assets/figures/ml_fig_c110_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 09 (original).](../assets/figures/ml_fig_c111_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 09 (original).](../assets/figures/ml_fig_c112_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 09 (original).](../assets/figures/ml_fig_c113_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 09 (original).](../assets/figures/ml_fig_c114_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 09 (original).](../assets/figures/ml_fig_c115_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 09 (original).](../assets/figures/ml_fig_c116_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 09 (original).](../assets/figures/ml_fig_c117_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 09 (original).](../assets/figures/ml_fig_c118_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 09 (original).](../assets/figures/ml_fig_c119_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 09 (original).](../assets/figures/ml_fig_c120_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 09 (original).](../assets/figures/ml_fig_c121_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 09 (original).](../assets/figures/ml_fig_c122_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 09 (original).](../assets/figures/ml_fig_c123_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 09 (original).](../assets/figures/ml_fig_c124_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 09 (original).](../assets/figures/ml_fig_c125_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 09 (original).](../assets/figures/ml_fig_c126_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 09 (original).](../assets/figures/ml_fig_c127_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 09 (original).](../assets/figures/ml_fig_c128_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 09 (original).](../assets/figures/ml_fig_c129_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 09 (original).](../assets/figures/ml_fig_c130_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 09 (original).](../assets/figures/ml_fig_c131_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 09 (original).](../assets/figures/ml_fig_c132_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 09 (original).](../assets/figures/ml_fig_c133_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 09 (original).](../assets/figures/ml_fig_c134_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 09 (original).](../assets/figures/ml_fig_c135_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 09 (original).](../assets/figures/ml_fig_c136_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 09 (original).](../assets/figures/ml_fig_c137_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 09 (original).](../assets/figures/ml_fig_c138_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 09 (original).](../assets/figures/ml_fig_c139_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 09 (original).](../assets/figures/ml_fig_c140_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 09 (original).](../assets/figures/ml_fig_c141_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 09 (original).](../assets/figures/ml_fig_c142_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 09 (original).](../assets/figures/ml_fig_c143_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 09 (original).](../assets/figures/ml_fig_c144_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 09 (original).](../assets/figures/ml_fig_c145_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 09 (original).](../assets/figures/ml_fig_c146_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 09 (original).](../assets/figures/ml_fig_c147_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 09 (original).](../assets/figures/ml_fig_c148_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 09 (original).](../assets/figures/ml_fig_c149_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 09 (original).](../assets/figures/ml_fig_c150_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 09 (original).](../assets/figures/ml_fig_c151_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 09 (original).](../assets/figures/ml_fig_c152_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 09 (original).](../assets/figures/ml_fig_c153_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 09 (original).](../assets/figures/ml_fig_c154_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 09 (original).](../assets/figures/ml_fig_c155_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 09 (original).](../assets/figures/ml_fig_c156_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 09 (original).](../assets/figures/ml_fig_c157_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 09 (original).](../assets/figures/ml_fig_c158_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 09 (original).](../assets/figures/ml_fig_c159_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 09 (original).](../assets/figures/ml_fig_c160_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 09 (original).](../assets/figures/ml_fig_c161_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 09 (original).](../assets/figures/ml_fig_c162_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 09 (original).](../assets/figures/ml_fig_c163_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 09 (original).](../assets/figures/ml_fig_c164_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 09 (original).](../assets/figures/ml_fig_c165_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 09 (original).](../assets/figures/ml_fig_c166_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 09 (original).](../assets/figures/ml_fig_c167_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 09 (original).](../assets/figures/ml_fig_c168_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 09 (original).](../assets/figures/ml_fig_c169_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 09 (original).](../assets/figures/ml_fig_c170_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 09 (original).](../assets/figures/ml_fig_c171_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 09 (original).](../assets/figures/ml_fig_c172_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 09 (original).](../assets/figures/ml_fig_c173_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 09 (original).](../assets/figures/ml_fig_c174_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 09 (original).](../assets/figures/ml_fig_c175_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 09 (original).](../assets/figures/ml_fig_c176_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 09 (original).](../assets/figures/ml_fig_c177_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 09 (original).](../assets/figures/ml_fig_c178_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 09 (original).](../assets/figures/ml_fig_c179_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 09 (original).](../assets/figures/ml_fig_c180_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 09 (original).](../assets/figures/ml_fig_c181_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 09 (original).](../assets/figures/ml_fig_c182_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 09 (original).](../assets/figures/ml_fig_c183_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 09 (original).](../assets/figures/ml_fig_c184_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 09 (original).](../assets/figures/ml_fig_c185_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 09 (original).](../assets/figures/ml_fig_c186_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 09 (original).](../assets/figures/ml_fig_c187_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 09 (original).](../assets/figures/ml_fig_c188_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 09 (original).](../assets/figures/ml_fig_c189_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 09 (original).](../assets/figures/ml_fig_c190_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 09 (original).](../assets/figures/ml_fig_c191_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 09 (original).](../assets/figures/ml_fig_c192_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 09 (original).](../assets/figures/ml_fig_c193_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 09 (original).](../assets/figures/ml_fig_c194_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 09 (original).](../assets/figures/ml_fig_c195_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 09 (original).](../assets/figures/ml_fig_c196_09.png)
*Figure — MM algorithm majorization. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 09 (original).](../assets/figures/ml_fig_c197_09.png)
*Figure — IRLS weighted steps. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 09 (original).](../assets/figures/ml_fig_c198_09.png)
*Figure — GEE cluster robust SE. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 09 (original).](../assets/figures/ml_fig_c199_09.png)
*Figure — Mixed effects random slope. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 09 (original).](../assets/figures/ml_fig_c200_09.png)
*Figure — Partial residual smoother. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 09 (original).](../assets/figures/ml_fig_c201_09.png)
*Figure — Cook distance leverage residual. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 09 (original).](../assets/figures/ml_fig_c202_09.png)
*Figure — Partial residual nonlinearity. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 09 (original).](../assets/figures/ml_fig_c203_09.png)
*Figure — Ridge coefficient shrinkage path. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 09 (original).](../assets/figures/ml_fig_c204_09.png)
*Figure — Elastic-net constraint geometry. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 09 (original).](../assets/figures/ml_fig_c205_09.png)
*Figure — Variance inflation factors. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 09 (original).](../assets/figures/ml_fig_c206_09.png)
*Figure — Box-Cox log transform effect. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 09 (original).](../assets/figures/ml_fig_c207_09.png)
*Figure — Huber loss delta transition. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 09 (original).](../assets/figures/ml_fig_c208_09.png)
*Figure — Poisson GLM log-link mean. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 09 (original).](../assets/figures/ml_fig_c209_09.png)
*Figure — GAM partial smooth effect. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 09 (original).](../assets/figures/ml_fig_c210_09.png)
*Figure — Negative binomial overdispersion. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 09 (original).](../assets/figures/ml_fig_c211_09.png)
*Figure — Pinball quantile loss tau. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 09 (original).](../assets/figures/ml_fig_c212_09.png)
*Figure — Partial residual smooth check. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 09 (original).](../assets/figures/ml_fig_c213_09.png)
*Figure — IRLS coefficient convergence. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 09 (original).](../assets/figures/ml_fig_c214_09.png)
*Figure — Sandwich robust standard errors. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 09 (original).](../assets/figures/ml_fig_c215_09.png)
*Figure — Studentized residual outliers. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 09 (original).](../assets/figures/ml_fig_c216_09.png)
*Figure — Cook distance influence stem. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 09 (original).](../assets/figures/ml_fig_c217_09.png)
*Figure — Added-variable residual plot. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 09 (original).](../assets/figures/ml_fig_c218_09.png)
*Figure — Lasso CV one-SE rule. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 09 (original).](../assets/figures/ml_fig_c219_09.png)
*Figure — Quantile regression fan lines. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 09 (original).](../assets/figures/ml_fig_c220_09.png)
*Figure — Ridge leverage vs lambda. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 09 (original).](../assets/figures/ml_fig_c221_09.png)
*Figure — Partial residual nonlinearity check. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 09 (original).](../assets/figures/ml_fig_c222_09.png)
*Figure — Huber M-estimator rho curve. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 09 (original).](../assets/figures/ml_fig_c223_09.png)
*Figure — GAM smooth partial effect. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 09 (original).](../assets/figures/ml_fig_c224_09.png)
*Figure — QR upper-triangular R factor. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 09 (original).](../assets/figures/ml_fig_c225_09.png)
*Figure — LOESS local smooth sketch. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 09 (original).](../assets/figures/ml_fig_c226_09.png)
*Figure — Poisson GLM mean curve. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 09 (original).](../assets/figures/ml_fig_c227_09.png)
*Figure — Elastic-net coefficient path. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 09 (original).](../assets/figures/ml_fig_c228_09.png)
*Figure — Negative binomial count PMF. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 09 (original).](../assets/figures/ml_fig_c229_09.png)
*Figure — Box-Cox transform family. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 09 (original).](../assets/figures/ml_fig_c230_09.png)
*Figure — Robust vs OLS outlier pull. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 09 (original).](../assets/figures/ml_fig_c231_09.png)
*Figure — Cubic spline basis bundle. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 09 (original).](../assets/figures/ml_fig_c232_09.png)
*Figure — Quantile regression non-crossing. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 09 (original).](../assets/figures/ml_fig_c233_09.png)
*Figure — LAD vs squared residual cost. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 09 (original).](../assets/figures/ml_fig_c234_09.png)
*Figure — Pinball quantile loss shape. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 09 (original).](../assets/figures/ml_fig_c235_09.png)
*Figure — Huber transition residual cost. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 09 (original).](../assets/figures/ml_fig_c236_09.png)
*Figure — Expectile asymmetric loss. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 09 (original).](../assets/figures/ml_fig_c237_09.png)
*Figure — Tukey biweight residual cost. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 09 (original).](../assets/figures/ml_fig_c238_09.png)
*Figure — Poisson deviance residual. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 09 (original).](../assets/figures/ml_fig_c239_09.png)
*Figure — Fair loss residual cost. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 09 (original).](../assets/figures/ml_fig_c240_09.png)
*Figure — Tweedie deviance residual. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 09 (original).](../assets/figures/ml_fig_c241_09.png)
*Figure — Quantile pinball residual cost. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 09 (original).](../assets/figures/ml_fig_c242_09.png)
*Figure — Pinball tau residual map. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 09 (original).](../assets/figures/ml_fig_c243_09.png)
*Figure — Cauchy robust residual cost. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 09 (original).](../assets/figures/ml_fig_c244_09.png)
*Figure — Asymmetric Huber residual. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 09 (original).](../assets/figures/ml_fig_c245_09.png)
*Figure — Fairness equalized odds cost. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 09 (original).](../assets/figures/ml_fig_c246_09.png)
*Figure — Check loss tau residual. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 09 (original).](../assets/figures/ml_fig_c247_09.png)
*Figure — Tukey biweight residual cost. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 09 (original).](../assets/figures/ml_fig_c248_09.png)
*Figure — Expectile tau residual map. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 09 (original).](../assets/figures/ml_fig_c249_09.png)
*Figure — Welsch robust residual cost. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 09 (original).](../assets/figures/ml_fig_c250_09.png)
*Figure — Pinball asymmetric residual. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 09 (original).](../assets/figures/ml_fig_c251_09.png)
*Figure — Fairness demographic parity. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 09 (original).](../assets/figures/ml_fig_c252_09.png)
*Figure — Huber delta residual map. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 09 (original).](../assets/figures/ml_fig_c253_09.png)
*Figure — Quantile check residual. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 09 (original).](../assets/figures/ml_fig_c254_09.png)
*Figure — Expectile residual map. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 09 (original).](../assets/figures/ml_fig_c255_09.png)
*Figure — Tukey biweight cost. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 09 (original).](../assets/figures/ml_fig_c256_09.png)
*Figure — Cauchy residual map. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 09 (original).](../assets/figures/ml_fig_c257_09.png)
*Figure — Cross-val MSE path c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 09 (original).](../assets/figures/ml_fig_c258_09.png)
*Figure — Partial residual path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 09 (original).](../assets/figures/ml_fig_c259_09.png)
*Figure — Isotonic mono path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 09 (original).](../assets/figures/ml_fig_c260_09.png)
*Figure — Expectile residual path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 09 (original).](../assets/figures/ml_fig_c261_09.png)
*Figure — OLS residual QQ path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 09 (original).](../assets/figures/ml_fig_c262_09.png)
*Figure — Ridge path residual c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 09 (original).](../assets/figures/ml_fig_c263_09.png)
*Figure — Lasso soft-threshold path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 09 (original).](../assets/figures/ml_fig_c264_09.png)
*Figure — Elastic net path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 09 (original).](../assets/figures/ml_fig_c265_09.png)
*Figure — Huber residual cost c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 09 (original).](../assets/figures/ml_fig_c266_09.png)
*Figure — Quantile pinball path c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 09 (original).](../assets/figures/ml_fig_c267_09.png)
*Figure — Poisson deviance path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 09 (original).](../assets/figures/ml_fig_c268_09.png)
*Figure — Tweedie deviance path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 09 (original).](../assets/figures/ml_fig_c269_09.png)
*Figure — GAM smooth residual c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 09 (original).](../assets/figures/ml_fig_c270_09.png)
*Figure — Spline knot residual c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 09 (original).](../assets/figures/ml_fig_c271_09.png)
*Figure — Heteroscedasticity path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 09 (original).](../assets/figures/ml_fig_c272_09.png)
*Figure — Leverage cook bars c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 09 (original).](../assets/figures/ml_fig_c273_09.png)
*Figure — Cross-val MSE path c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 09 (original).](../assets/figures/ml_fig_c274_09.png)
*Figure — Partial residual path c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 09 (original).](../assets/figures/ml_fig_c275_09.png)
*Figure — Isotonic mono path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 09 (original).](../assets/figures/ml_fig_c276_09.png)
*Figure — Expectile residual path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 09 (original).](../assets/figures/ml_fig_c277_09.png)
*Figure — OLS residual QQ path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 09 (original).](../assets/figures/ml_fig_c278_09.png)
*Figure — Ridge path residual c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 09 (original).](../assets/figures/ml_fig_c279_09.png)
*Figure — Lasso soft-threshold path c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 09 (original).](../assets/figures/ml_fig_c280_09.png)
*Figure — Elastic net path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 09 (original).](../assets/figures/ml_fig_c281_09.png)
*Figure — Huber residual cost c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 09 (original).](../assets/figures/ml_fig_c282_09.png)
*Figure — Quantile pinball path c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 09 (original).](../assets/figures/ml_fig_c283_09.png)
*Figure — Poisson deviance path c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 09 (original).](../assets/figures/ml_fig_c284_09.png)
*Figure — Tweedie deviance path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 09 (original).](../assets/figures/ml_fig_c285_09.png)
*Figure — GAM smooth residual c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 09 (original).](../assets/figures/ml_fig_c286_09.png)
*Figure — Spline knot residual c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 09 (original).](../assets/figures/ml_fig_c287_09.png)
*Figure — Heteroscedasticity path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 09 (original).](../assets/figures/ml_fig_c288_09.png)
*Figure — Leverage cook bars c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 09 (original).](../assets/figures/ml_fig_c289_09.png)
*Figure — Cross-val MSE path c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 09 (original).](../assets/figures/ml_fig_c290_09.png)
*Figure — Partial residual path c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 09 (original).](../assets/figures/ml_fig_c291_09.png)
*Figure — Isotonic mono path c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 09 (original).](../assets/figures/ml_fig_c292_09.png)
*Figure — Expectile residual path c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 09 (original).](../assets/figures/ml_fig_c293_09.png)
*Figure — OLS residual QQ path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 09 (original).](../assets/figures/ml_fig_c294_09.png)
*Figure — Ridge path residual c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 09 (original).](../assets/figures/ml_fig_c295_09.png)
*Figure — Lasso soft-threshold path c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 09 (original).](../assets/figures/ml_fig_c296_09.png)
*Figure — Elastic net path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 09 (original).](../assets/figures/ml_fig_c297_09.png)
*Figure — Huber residual cost c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 09 (original).](../assets/figures/ml_fig_c298_09.png)
*Figure — Quantile pinball path c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 09 (original).](../assets/figures/ml_fig_c299_09.png)
*Figure — Poisson deviance path c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 09 (original).](../assets/figures/ml_fig_c300_09.png)
*Figure — Tweedie deviance path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 09 (original).](../assets/figures/ml_fig_c301_09.png)
*Figure — GAM smooth residual c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 09 (original).](../assets/figures/ml_fig_c302_09.png)
*Figure — Spline knot residual c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 09 (original).](../assets/figures/ml_fig_c303_09.png)
*Figure — Heteroscedasticity path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 09 (original).](../assets/figures/ml_fig_c304_09.png)
*Figure — Leverage cook bars c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 09 (original).](../assets/figures/ml_fig_c305_09.png)
*Figure — Cross-val MSE path c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 09 (original).](../assets/figures/ml_fig_c306_09.png)
*Figure — Partial residual path c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 09 (original).](../assets/figures/ml_fig_c307_09.png)
*Figure — Isotonic mono path c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 09 (original).](../assets/figures/ml_fig_c308_09.png)
*Figure — Expectile residual path c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 09 (original).](../assets/figures/ml_fig_c309_09.png)
*Figure — OLS residual QQ path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 09 (original).](../assets/figures/ml_fig_c310_09.png)
*Figure — Ridge path residual c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 09 (original).](../assets/figures/ml_fig_c311_09.png)
*Figure — Lasso soft-threshold path c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 09 (original).](../assets/figures/ml_fig_c312_09.png)
*Figure — Elastic net path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 09 (original).](../assets/figures/ml_fig_c313_09.png)
*Figure — Huber residual cost c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 09 (original).](../assets/figures/ml_fig_c314_09.png)
*Figure — Quantile pinball path c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 09 (original).](../assets/figures/ml_fig_c315_09.png)
*Figure — Poisson deviance path c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 09 (original).](../assets/figures/ml_fig_c316_09.png)
*Figure — Tweedie deviance path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 09 (original).](../assets/figures/ml_fig_c317_09.png)
*Figure — GAM smooth residual c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 09 (original).](../assets/figures/ml_fig_c318_09.png)
*Figure — Spline knot residual c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 09 (original).](../assets/figures/ml_fig_c319_09.png)
*Figure — Heteroscedasticity path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 09 (original).](../assets/figures/ml_fig_c320_09.png)
*Figure — Leverage cook bars c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 09 (original).](../assets/figures/ml_fig_c321_09.png)
*Figure — Cross-val MSE path c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 09 (original).](../assets/figures/ml_fig_c322_09.png)
*Figure — Partial residual path c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 09 (original).](../assets/figures/ml_fig_c323_09.png)
*Figure — Isotonic mono path c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 09 (original).](../assets/figures/ml_fig_c324_09.png)
*Figure — Expectile residual path c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 09 (original).](../assets/figures/ml_fig_c325_09.png)
*Figure — OLS residual QQ path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 09 (original).](../assets/figures/ml_fig_c326_09.png)
*Figure — Ridge path residual c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 09 (original).](../assets/figures/ml_fig_c327_09.png)
*Figure — Lasso soft-threshold path c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 09 (original).](../assets/figures/ml_fig_c328_09.png)
*Figure — Elastic net path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 09 (original).](../assets/figures/ml_fig_c329_09.png)
*Figure — Huber residual cost c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 09 (original).](../assets/figures/ml_fig_c330_09.png)
*Figure — Quantile pinball path c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 09 (original).](../assets/figures/ml_fig_c331_09.png)
*Figure — Poisson deviance path c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 09 (original).](../assets/figures/ml_fig_c332_09.png)
*Figure — Tweedie deviance path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 09 (original).](../assets/figures/ml_fig_c333_09.png)
*Figure — GAM smooth residual c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 09 (original).](../assets/figures/ml_fig_c334_09.png)
*Figure — Spline knot residual c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 09 (original).](../assets/figures/ml_fig_c335_09.png)
*Figure — Heteroscedasticity path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 09 (original).](../assets/figures/ml_fig_c336_09.png)
*Figure — Leverage cook bars c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 09 (original).](../assets/figures/ml_fig_c337_09.png)
*Figure — Cross-val MSE path c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 09 (original).](../assets/figures/ml_fig_c338_09.png)
*Figure — Partial residual path c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 09 (original).](../assets/figures/ml_fig_c339_09.png)
*Figure — Isotonic mono path c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 09 (original).](../assets/figures/ml_fig_c340_09.png)
*Figure — Expectile residual path c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 09 (original).](../assets/figures/ml_fig_c341_09.png)
*Figure — OLS residual QQ path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 09 (original).](../assets/figures/ml_fig_c342_09.png)
*Figure — Ridge path residual c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 09 (original).](../assets/figures/ml_fig_c343_09.png)
*Figure — Lasso soft-threshold path c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 09 (original).](../assets/figures/ml_fig_c344_09.png)
*Figure — Elastic net path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 09 (original).](../assets/figures/ml_fig_c345_09.png)
*Figure — Huber residual cost c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 09 (original).](../assets/figures/ml_fig_c346_09.png)
*Figure — Quantile pinball path c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 09 (original).](../assets/figures/ml_fig_c347_09.png)
*Figure — Poisson deviance path c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 09 (original).](../assets/figures/ml_fig_c348_09.png)
*Figure — Tweedie deviance path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 09 (original).](../assets/figures/ml_fig_c349_09.png)
*Figure — GAM smooth residual c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 09 (original).](../assets/figures/ml_fig_c350_09.png)
*Figure — Spline knot residual c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 09 (original).](../assets/figures/ml_fig_c351_09.png)
*Figure — Heteroscedasticity path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 09 (original).](../assets/figures/ml_fig_c352_09.png)
*Figure — Leverage cook bars c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 09 (original).](../assets/figures/ml_fig_c353_09.png)
*Figure — Cross-val MSE path c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 09 (original).](../assets/figures/ml_fig_c354_09.png)
*Figure — Partial residual path c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 09 (original).](../assets/figures/ml_fig_c355_09.png)
*Figure — Isotonic mono path c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 09 (original).](../assets/figures/ml_fig_c356_09.png)
*Figure — Expectile residual path c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 09 (original).](../assets/figures/ml_fig_c357_09.png)
*Figure — OLS residual QQ path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 09 (original).](../assets/figures/ml_fig_c358_09.png)
*Figure — Ridge path residual c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 09 (original).](../assets/figures/ml_fig_c359_09.png)
*Figure — Lasso soft-threshold path c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 09 (original).](../assets/figures/ml_fig_c360_09.png)
*Figure — Elastic net path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 09 (original).](../assets/figures/ml_fig_c361_09.png)
*Figure — Huber residual cost c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 09 (original).](../assets/figures/ml_fig_c362_09.png)
*Figure — Quantile pinball path c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 09 (original).](../assets/figures/ml_fig_c363_09.png)
*Figure — Poisson deviance path c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 09 (original).](../assets/figures/ml_fig_c364_09.png)
*Figure — Tweedie deviance path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 09 (original).](../assets/figures/ml_fig_c365_09.png)
*Figure — GAM smooth residual c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 09 (original).](../assets/figures/ml_fig_c366_09.png)
*Figure — Spline knot residual c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 09 (original).](../assets/figures/ml_fig_c367_09.png)
*Figure — Heteroscedasticity path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 09 (original).](../assets/figures/ml_fig_c368_09.png)
*Figure — Leverage cook bars c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 09 (original).](../assets/figures/ml_fig_c369_09.png)
*Figure — Cross-val MSE path c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 09 (original).](../assets/figures/ml_fig_c370_09.png)
*Figure — Partial residual path c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 09 (original).](../assets/figures/ml_fig_c371_09.png)
*Figure — Isotonic mono path c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 09 (original).](../assets/figures/ml_fig_c372_09.png)
*Figure — Expectile residual path c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 09 (original).](../assets/figures/ml_fig_c373_09.png)
*Figure — OLS residual QQ path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 09 (original).](../assets/figures/ml_fig_c374_09.png)
*Figure — Ridge path residual c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 09 (original).](../assets/figures/ml_fig_c375_09.png)
*Figure — Lasso soft-threshold path c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 09 (original).](../assets/figures/ml_fig_c376_09.png)
*Figure — Elastic net path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 09 (original).](../assets/figures/ml_fig_c377_09.png)
*Figure — Huber residual cost c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 09 (original).](../assets/figures/ml_fig_c378_09.png)
*Figure — Quantile pinball path c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 09 (original).](../assets/figures/ml_fig_c379_09.png)
*Figure — Poisson deviance path c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 09 (original).](../assets/figures/ml_fig_c380_09.png)
*Figure — Tweedie deviance path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 09 (original).](../assets/figures/ml_fig_c381_09.png)
*Figure — GAM smooth residual c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 09 (original).](../assets/figures/ml_fig_c382_09.png)
*Figure — Spline knot residual c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 09 (original).](../assets/figures/ml_fig_c383_09.png)
*Figure — Heteroscedasticity path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 09 (original).](../assets/figures/ml_fig_c384_09.png)
*Figure — Leverage cook bars c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 09 (original).](../assets/figures/ml_fig_c385_09.png)
*Figure — Cross-val MSE path c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 09 (original).](../assets/figures/ml_fig_c386_09.png)
*Figure — Partial residual path c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 09 (original).](../assets/figures/ml_fig_c387_09.png)
*Figure — Isotonic mono path c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 09 (original).](../assets/figures/ml_fig_c388_09.png)
*Figure — Expectile residual path c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 09 (original).](../assets/figures/ml_fig_c389_09.png)
*Figure — OLS residual QQ path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 09 (original).](../assets/figures/ml_fig_c390_09.png)
*Figure — Ridge path residual c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 09 (original).](../assets/figures/ml_fig_c391_09.png)
*Figure — Lasso soft-threshold path c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 09 (original).](../assets/figures/ml_fig_c392_09.png)
*Figure — Elastic net path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 09 (original).](../assets/figures/ml_fig_c393_09.png)
*Figure — Huber residual cost c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 09 (original).](../assets/figures/ml_fig_c394_09.png)
*Figure — Quantile pinball path c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 09 (original).](../assets/figures/ml_fig_c395_09.png)
*Figure — Poisson deviance path c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 09 (original).](../assets/figures/ml_fig_c396_09.png)
*Figure — Tweedie deviance path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 09 (original).](../assets/figures/ml_fig_c397_09.png)
*Figure — GAM smooth residual c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 09 (original).](../assets/figures/ml_fig_c398_09.png)
*Figure — Spline knot residual c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 09 (original).](../assets/figures/ml_fig_c399_09.png)
*Figure — Heteroscedasticity path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 09 (original).](../assets/figures/ml_fig_c400_09.png)
*Figure — Leverage cook bars c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 09 (original).](../assets/figures/ml_fig_c401_09.png)
*Figure — Cross-val MSE path c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 09 (original).](../assets/figures/ml_fig_c402_09.png)
*Figure — Partial residual path c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 09 (original).](../assets/figures/ml_fig_c403_09.png)
*Figure — Isotonic mono path c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 09 (original).](../assets/figures/ml_fig_c404_09.png)
*Figure — Expectile residual path c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 09 (original).](../assets/figures/ml_fig_c405_09.png)
*Figure — OLS residual QQ path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 09 (original).](../assets/figures/ml_fig_c406_09.png)
*Figure — Ridge path residual c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 09 (original).](../assets/figures/ml_fig_c407_09.png)
*Figure — Lasso soft-threshold path c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 09 (original).](../assets/figures/ml_fig_c408_09.png)
*Figure — Elastic net path c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 09 (original).](../assets/figures/ml_fig_c409_09.png)
*Figure — Huber residual cost c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 09 (original).](../assets/figures/ml_fig_c410_09.png)
*Figure — Quantile pinball path c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 09 (original).](../assets/figures/ml_fig_c411_09.png)
*Figure — Poisson deviance path c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 09 (original).](../assets/figures/ml_fig_c412_09.png)
*Figure — Tweedie deviance path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 09 (original).](../assets/figures/ml_fig_c413_09.png)
*Figure — GAM smooth residual c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 09 (original).](../assets/figures/ml_fig_c414_09.png)
*Figure — Spline knot residual c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 09 (original).](../assets/figures/ml_fig_c415_09.png)
*Figure — Heteroscedasticity path c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 09 (original).](../assets/figures/ml_fig_c416_09.png)
*Figure — Leverage cook bars c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 09 (original).](../assets/figures/ml_fig_c417_09.png)
*Figure — Cross-val MSE path c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 09 (original).](../assets/figures/ml_fig_c418_09.png)
*Figure — Partial residual path c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 09 (original).](../assets/figures/ml_fig_c419_09.png)
*Figure — Isotonic mono path c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 09 (original).](../assets/figures/ml_fig_c420_09.png)
*Figure — Expectile residual path c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 09 (original).](../assets/figures/ml_fig_c421_09.png)
*Figure — OLS residual QQ path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 09 (original).](../assets/figures/ml_fig_c422_09.png)
*Figure — Ridge path residual c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 09 (original).](../assets/figures/ml_fig_c423_09.png)
*Figure — Lasso soft-threshold path c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 09 (original).](../assets/figures/ml_fig_c424_09.png)
*Figure — Elastic net path c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 09 (original).](../assets/figures/ml_fig_c425_09.png)
*Figure — Huber residual cost c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 09 (original).](../assets/figures/ml_fig_c426_09.png)
*Figure — Quantile pinball path c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 09 (original).](../assets/figures/ml_fig_c427_09.png)
*Figure — Poisson deviance path c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 09 (original).](../assets/figures/ml_fig_c428_09.png)
*Figure — Tweedie deviance path c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 09 (original).](../assets/figures/ml_fig_c429_09.png)
*Figure — GAM smooth residual c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 09 (original).](../assets/figures/ml_fig_c430_09.png)
*Figure — Spline knot residual c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 09 (original).](../assets/figures/ml_fig_c431_09.png)
*Figure — Heteroscedasticity path c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 09 (original).](../assets/figures/ml_fig_c432_09.png)
*Figure — Leverage cook bars c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 09 (original).](../assets/figures/ml_fig_c433_09.png)
*Figure — Cross-val MSE path c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 09 (original).](../assets/figures/ml_fig_c434_09.png)
*Figure — Partial residual path c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 09 (original).](../assets/figures/ml_fig_c435_09.png)
*Figure — Isotonic mono path c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 09 (original).](../assets/figures/ml_fig_c436_09.png)
*Figure — Expectile residual path c436. Synthetic teaching geometry—not a causal claim.*


![c437 teaching panel 09 (original).](../assets/figures/ml_fig_c437_09.png)
*Figure — OLS residual QQ path c437. Synthetic teaching geometry—not a causal claim.*


![c438 teaching panel 09 (original).](../assets/figures/ml_fig_c438_09.png)
*Figure — Ridge path residual c438. Synthetic teaching geometry—not a causal claim.*


![c439 teaching panel 09 (original).](../assets/figures/ml_fig_c439_09.png)
*Figure — Lasso soft-threshold path c439. Synthetic teaching geometry—not a causal claim.*


![c440 teaching panel 09 (original).](../assets/figures/ml_fig_c440_09.png)
*Figure — Elastic net path c440. Synthetic teaching geometry—not a causal claim.*


![c441 teaching panel 09 (original).](../assets/figures/ml_fig_c441_09.png)
*Figure — Huber residual cost c441. Synthetic teaching geometry—not a causal claim.*


![c442 teaching panel 09 (original).](../assets/figures/ml_fig_c442_09.png)
*Figure — Quantile pinball path c442. Synthetic teaching geometry—not a causal claim.*


![c443 teaching panel 09 (original).](../assets/figures/ml_fig_c443_09.png)
*Figure — Poisson deviance path c443. Synthetic teaching geometry—not a causal claim.*


![c444 teaching panel 09 (original).](../assets/figures/ml_fig_c444_09.png)
*Figure — Tweedie deviance path c444. Synthetic teaching geometry—not a causal claim.*


![c445 teaching panel 09 (original).](../assets/figures/ml_fig_c445_09.png)
*Figure — GAM smooth residual c445. Synthetic teaching geometry—not a causal claim.*


![c446 teaching panel 09 (original).](../assets/figures/ml_fig_c446_09.png)
*Figure — Spline knot residual c446. Synthetic teaching geometry—not a causal claim.*


![c447 teaching panel 09 (original).](../assets/figures/ml_fig_c447_09.png)
*Figure — Heteroscedasticity path c447. Synthetic teaching geometry—not a causal claim.*


![c448 teaching panel 09 (original).](../assets/figures/ml_fig_c448_09.png)
*Figure — Leverage cook bars c448. Synthetic teaching geometry—not a causal claim.*


![c449 teaching panel 09 (original).](../assets/figures/ml_fig_c449_09.png)
*Figure — Cross-val MSE path c449. Synthetic teaching geometry—not a causal claim.*


![c450 teaching panel 09 (original).](../assets/figures/ml_fig_c450_09.png)
*Figure — Partial residual path c450. Synthetic teaching geometry—not a causal claim.*


![c451 teaching panel 09 (original).](../assets/figures/ml_fig_c451_09.png)
*Figure — Isotonic mono path c451. Synthetic teaching geometry—not a causal claim.*


![c452 teaching panel 09 (original).](../assets/figures/ml_fig_c452_09.png)
*Figure — Expectile residual path c452. Synthetic teaching geometry—not a causal claim.*


![c453 teaching panel 09 (original).](../assets/figures/ml_fig_c453_09.png)
*Figure — OLS residual QQ path c453. Synthetic teaching geometry—not a causal claim.*


![c454 teaching panel 09 (original).](../assets/figures/ml_fig_c454_09.png)
*Figure — Ridge path residual c454. Synthetic teaching geometry—not a causal claim.*


![c455 teaching panel 09 (original).](../assets/figures/ml_fig_c455_09.png)
*Figure — Lasso soft-threshold path c455. Synthetic teaching geometry—not a causal claim.*


![c456 teaching panel 09 (original).](../assets/figures/ml_fig_c456_09.png)
*Figure — Elastic net path c456. Synthetic teaching geometry—not a causal claim.*


![c457 teaching panel 09 (original).](../assets/figures/ml_fig_c457_09.png)
*Figure — Huber residual cost c457. Synthetic teaching geometry—not a causal claim.*


![c458 teaching panel 09 (original).](../assets/figures/ml_fig_c458_09.png)
*Figure — Quantile pinball path c458. Synthetic teaching geometry—not a causal claim.*


![c459 teaching panel 09 (original).](../assets/figures/ml_fig_c459_09.png)
*Figure — Poisson deviance path c459. Synthetic teaching geometry—not a causal claim.*


![c460 teaching panel 09 (original).](../assets/figures/ml_fig_c460_09.png)
*Figure — Tweedie deviance path c460. Synthetic teaching geometry—not a causal claim.*


![c461 teaching panel 09 (original).](../assets/figures/ml_fig_c461_09.png)
*Figure — GAM smooth residual c461. Synthetic teaching geometry—not a causal claim.*


![c462 teaching panel 09 (original).](../assets/figures/ml_fig_c462_09.png)
*Figure — Spline knot residual c462. Synthetic teaching geometry—not a causal claim.*


![c463 teaching panel 09 (original).](../assets/figures/ml_fig_c463_09.png)
*Figure — Heteroscedasticity path c463. Synthetic teaching geometry—not a causal claim.*


![c464 teaching panel 09 (original).](../assets/figures/ml_fig_c464_09.png)
*Figure — Leverage cook bars c464. Synthetic teaching geometry—not a causal claim.*


![c465 teaching panel 09 (original).](../assets/figures/ml_fig_c465_09.png)
*Figure — Cross-val MSE path c465. Synthetic teaching geometry—not a causal claim.*


![c466 teaching panel 09 (original).](../assets/figures/ml_fig_c466_09.png)
*Figure — Partial residual path c466. Synthetic teaching geometry—not a causal claim.*


![c467 teaching panel 09 (original).](../assets/figures/ml_fig_c467_09.png)
*Figure — Isotonic mono path c467. Synthetic teaching geometry—not a causal claim.*


![c468 teaching panel 09 (original).](../assets/figures/ml_fig_c468_09.png)
*Figure — Expectile residual path c468. Synthetic teaching geometry—not a causal claim.*


![c469 teaching panel 09 (original).](../assets/figures/ml_fig_c469_09.png)
*Figure — OLS residual QQ path c469. Synthetic teaching geometry—not a causal claim.*


![c470 teaching panel 09 (original).](../assets/figures/ml_fig_c470_09.png)
*Figure — Ridge path residual c470. Synthetic teaching geometry—not a causal claim.*


![c471 teaching panel 09 (original).](../assets/figures/ml_fig_c471_09.png)
*Figure — Lasso soft-threshold path c471. Synthetic teaching geometry—not a causal claim.*


![c472 teaching panel 09 (original).](../assets/figures/ml_fig_c472_09.png)
*Figure — Elastic net path c472. Synthetic teaching geometry—not a causal claim.*


![c473 teaching panel 09 (original).](../assets/figures/ml_fig_c473_09.png)
*Figure — Huber residual cost c473. Synthetic teaching geometry—not a causal claim.*


![c474 teaching panel 09 (original).](../assets/figures/ml_fig_c474_09.png)
*Figure — Quantile pinball path c474. Synthetic teaching geometry—not a causal claim.*


![c475 teaching panel 09 (original).](../assets/figures/ml_fig_c475_09.png)
*Figure — Poisson deviance path c475. Synthetic teaching geometry—not a causal claim.*


![c476 teaching panel 09 (original).](../assets/figures/ml_fig_c476_09.png)
*Figure — Tweedie deviance path c476. Synthetic teaching geometry—not a causal claim.*


![c477 teaching panel 09 (original).](../assets/figures/ml_fig_c477_09.png)
*Figure — GAM smooth residual c477. Synthetic teaching geometry—not a causal claim.*


![c478 teaching panel 09 (original).](../assets/figures/ml_fig_c478_09.png)
*Figure — Spline knot residual c478. Synthetic teaching geometry—not a causal claim.*


![c479 teaching panel 09 (original).](../assets/figures/ml_fig_c479_09.png)
*Figure — Heteroscedasticity path c479. Synthetic teaching geometry—not a causal claim.*


![c480 teaching panel 09 (original).](../assets/figures/ml_fig_c480_09.png)
*Figure — Leverage cook bars c480. Synthetic teaching geometry—not a causal claim.*


![c481 teaching panel 09 (original).](../assets/figures/ml_fig_c481_09.png)
*Figure — Cross-val MSE path c481. Synthetic teaching geometry—not a causal claim.*


![c482 teaching panel 09 (original).](../assets/figures/ml_fig_c482_09.png)
*Figure — Partial residual path c482. Synthetic teaching geometry—not a causal claim.*


![c483 teaching panel 09 (original).](../assets/figures/ml_fig_c483_09.png)
*Figure — Isotonic mono path c483. Synthetic teaching geometry—not a causal claim.*


![c484 teaching panel 09 (original).](../assets/figures/ml_fig_c484_09.png)
*Figure — Expectile residual path c484. Synthetic teaching geometry—not a causal claim.*


![c485 teaching panel 09 (original).](../assets/figures/ml_fig_c485_09.png)
*Figure — OLS residual QQ path c485. Synthetic teaching geometry—not a causal claim.*


![c486 teaching panel 09 (original).](../assets/figures/ml_fig_c486_09.png)
*Figure — Ridge path residual c486. Synthetic teaching geometry—not a causal claim.*


![c487 teaching panel 09 (original).](../assets/figures/ml_fig_c487_09.png)
*Figure — Lasso soft-threshold path c487. Synthetic teaching geometry—not a causal claim.*


![c488 teaching panel 09 (original).](../assets/figures/ml_fig_c488_09.png)
*Figure — Elastic net path c488. Synthetic teaching geometry—not a causal claim.*


![c489 teaching panel 09 (original).](../assets/figures/ml_fig_c489_09.png)
*Figure — Huber residual cost c489. Synthetic teaching geometry—not a causal claim.*


![c490 teaching panel 09 (original).](../assets/figures/ml_fig_c490_09.png)
*Figure — Quantile pinball path c490. Synthetic teaching geometry—not a causal claim.*


![c491 teaching panel 09 (original).](../assets/figures/ml_fig_c491_09.png)
*Figure — Poisson deviance path c491. Synthetic teaching geometry—not a causal claim.*


![c492 teaching panel 09 (original).](../assets/figures/ml_fig_c492_09.png)
*Figure — Tweedie deviance path c492. Synthetic teaching geometry—not a causal claim.*


![c493 teaching panel 09 (original).](../assets/figures/ml_fig_c493_09.png)
*Figure — GAM smooth residual c493. Synthetic teaching geometry—not a causal claim.*


![c494 teaching panel 09 (original).](../assets/figures/ml_fig_c494_09.png)
*Figure — Spline knot residual c494. Synthetic teaching geometry—not a causal claim.*


![c495 teaching panel 09 (original).](../assets/figures/ml_fig_c495_09.png)
*Figure — Heteroscedasticity path c495. Synthetic teaching geometry—not a causal claim.*


![c496 teaching panel 09 (original).](../assets/figures/ml_fig_c496_09.png)
*Figure — Leverage cook bars c496. Synthetic teaching geometry—not a causal claim.*


![c497 teaching panel 09 (original).](../assets/figures/ml_fig_c497_09.png)
*Figure — Cross-val MSE path c497. Synthetic teaching geometry—not a causal claim.*


![c498 teaching panel 09 (original).](../assets/figures/ml_fig_c498_09.png)
*Figure — Partial residual path c498. Synthetic teaching geometry—not a causal claim.*


![c499 teaching panel 09 (original).](../assets/figures/ml_fig_c499_09.png)
*Figure — Isotonic mono path c499. Synthetic teaching geometry—not a causal claim.*


![c500 teaching panel 09 (original).](../assets/figures/ml_fig_c500_09.png)
*Figure — Expectile residual path c500. Synthetic teaching geometry—not a causal claim.*


![c501 teaching panel 09 (original).](../assets/figures/ml_fig_c501_09.png)
*Figure — OLS residual QQ path c501. Synthetic teaching geometry—not a causal claim.*


![c502 teaching panel 09 (original).](../assets/figures/ml_fig_c502_09.png)
*Figure — Ridge path residual c502. Synthetic teaching geometry—not a causal claim.*


![c503 teaching panel 09 (original).](../assets/figures/ml_fig_c503_09.png)
*Figure — Lasso soft-threshold path c503. Synthetic teaching geometry—not a causal claim.*


![c504 teaching panel 09 (original).](../assets/figures/ml_fig_c504_09.png)
*Figure — Elastic net path c504. Synthetic teaching geometry—not a causal claim.*


![c505 teaching panel 09 (original).](../assets/figures/ml_fig_c505_09.png)
*Figure — Huber residual cost c505. Synthetic teaching geometry—not a causal claim.*


![c506 teaching panel 09 (original).](../assets/figures/ml_fig_c506_09.png)
*Figure — Quantile pinball path c506. Synthetic teaching geometry—not a causal claim.*


![c507 teaching panel 09 (original).](../assets/figures/ml_fig_c507_09.png)
*Figure — Poisson deviance path c507. Synthetic teaching geometry—not a causal claim.*


![c508 teaching panel 09 (original).](../assets/figures/ml_fig_c508_09.png)
*Figure — Tweedie deviance path c508. Synthetic teaching geometry—not a causal claim.*


![c509 teaching panel 09 (original).](../assets/figures/ml_fig_c509_09.png)
*Figure — GAM smooth residual c509. Synthetic teaching geometry—not a causal claim.*


![c510 teaching panel 09 (original).](../assets/figures/ml_fig_c510_09.png)
*Figure — Spline knot residual c510. Synthetic teaching geometry—not a causal claim.*


![c511 teaching panel 09 (original).](../assets/figures/ml_fig_c511_09.png)
*Figure — Heteroscedasticity path c511. Synthetic teaching geometry—not a causal claim.*


![c512 teaching panel 09 (original).](../assets/figures/ml_fig_c512_09.png)
*Figure — Leverage cook bars c512. Synthetic teaching geometry—not a causal claim.*


![c513 teaching panel 09 (original).](../assets/figures/ml_fig_c513_09.png)
*Figure — Cross-val MSE path c513. Synthetic teaching geometry—not a causal claim.*


![c514 teaching panel 09 (original).](../assets/figures/ml_fig_c514_09.png)
*Figure — Partial residual path c514. Synthetic teaching geometry—not a causal claim.*


![c515 teaching panel 09 (original).](../assets/figures/ml_fig_c515_09.png)
*Figure — Isotonic mono path c515. Synthetic teaching geometry—not a causal claim.*


![c516 teaching panel 09 (original).](../assets/figures/ml_fig_c516_09.png)
*Figure — Expectile residual path c516. Synthetic teaching geometry—not a causal claim.*


![c517 teaching panel 09 (original).](../assets/figures/ml_fig_c517_09.png)
*Figure — OLS residual QQ path c517. Synthetic teaching geometry—not a causal claim.*


![c518 teaching panel 09 (original).](../assets/figures/ml_fig_c518_09.png)
*Figure — Ridge path residual c518. Synthetic teaching geometry—not a causal claim.*


![c519 teaching panel 09 (original).](../assets/figures/ml_fig_c519_09.png)
*Figure — Lasso soft-threshold path c519. Synthetic teaching geometry—not a causal claim.*


![c520 teaching panel 09 (original).](../assets/figures/ml_fig_c520_09.png)
*Figure — Elastic net path c520. Synthetic teaching geometry—not a causal claim.*


![c521 teaching panel 09 (original).](../assets/figures/ml_fig_c521_09.png)
*Figure — Huber residual cost c521. Synthetic teaching geometry—not a causal claim.*


![c522 teaching panel 09 (original).](../assets/figures/ml_fig_c522_09.png)
*Figure — Quantile pinball path c522. Synthetic teaching geometry—not a causal claim.*


![c523 teaching panel 09 (original).](../assets/figures/ml_fig_c523_09.png)
*Figure — Poisson deviance path c523. Synthetic teaching geometry—not a causal claim.*


![c524 teaching panel 09 (original).](../assets/figures/ml_fig_c524_09.png)
*Figure — Tweedie deviance path c524. Synthetic teaching geometry—not a causal claim.*


![c525 teaching panel 09 (original).](../assets/figures/ml_fig_c525_09.png)
*Figure — GAM smooth residual c525. Synthetic teaching geometry—not a causal claim.*


![c526 teaching panel 09 (original).](../assets/figures/ml_fig_c526_09.png)
*Figure — Spline knot residual c526. Synthetic teaching geometry—not a causal claim.*


![c527 teaching panel 09 (original).](../assets/figures/ml_fig_c527_09.png)
*Figure — Heteroscedasticity path c527. Synthetic teaching geometry—not a causal claim.*


![c528 teaching panel 09 (original).](../assets/figures/ml_fig_c528_09.png)
*Figure — Leverage cook bars c528. Synthetic teaching geometry—not a causal claim.*


![c529 teaching panel 09 (original).](../assets/figures/ml_fig_c529_09.png)
*Figure — Cross-val MSE path c529. Synthetic teaching geometry—not a causal claim.*


![c530 teaching panel 09 (original).](../assets/figures/ml_fig_c530_09.png)
*Figure — Partial residual path c530. Synthetic teaching geometry—not a causal claim.*


![c531 teaching panel 09 (original).](../assets/figures/ml_fig_c531_09.png)
*Figure — Isotonic mono path c531. Synthetic teaching geometry—not a causal claim.*


![c532 teaching panel 09 (original).](../assets/figures/ml_fig_c532_09.png)
*Figure — Expectile residual path c532. Synthetic teaching geometry—not a causal claim.*


![c533 teaching panel 09 (original).](../assets/figures/ml_fig_c533_09.png)
*Figure — OLS residual QQ path c533. Synthetic teaching geometry—not a causal claim.*


![c534 teaching panel 09 (original).](../assets/figures/ml_fig_c534_09.png)
*Figure — Ridge path residual c534. Synthetic teaching geometry—not a causal claim.*


![c535 teaching panel 09 (original).](../assets/figures/ml_fig_c535_09.png)
*Figure — Lasso soft-threshold path c535. Synthetic teaching geometry—not a causal claim.*


![c536 teaching panel 09 (original).](../assets/figures/ml_fig_c536_09.png)
*Figure — Elastic net path c536. Synthetic teaching geometry—not a causal claim.*


![c537 teaching panel 09 (original).](../assets/figures/ml_fig_c537_09.png)
*Figure — Huber residual cost c537. Synthetic teaching geometry—not a causal claim.*


![c538 teaching panel 09 (original).](../assets/figures/ml_fig_c538_09.png)
*Figure — Quantile pinball path c538. Synthetic teaching geometry—not a causal claim.*


![c539 teaching panel 09 (original).](../assets/figures/ml_fig_c539_09.png)
*Figure — Poisson deviance path c539. Synthetic teaching geometry—not a causal claim.*


![c540 teaching panel 09 (original).](../assets/figures/ml_fig_c540_09.png)
*Figure — Tweedie deviance path c540. Synthetic teaching geometry—not a causal claim.*


![c541 teaching panel 09 (original).](../assets/figures/ml_fig_c541_09.png)
*Figure — GAM smooth residual c541. Synthetic teaching geometry—not a causal claim.*


![c542 teaching panel 09 (original).](../assets/figures/ml_fig_c542_09.png)
*Figure — Spline knot residual c542. Synthetic teaching geometry—not a causal claim.*


![c543 teaching panel 09 (original).](../assets/figures/ml_fig_c543_09.png)
*Figure — Heteroscedasticity path c543. Synthetic teaching geometry—not a causal claim.*


![c544 teaching panel 09 (original).](../assets/figures/ml_fig_c544_09.png)
*Figure — Leverage cook bars c544. Synthetic teaching geometry—not a causal claim.*


![c545 teaching panel 09 (original).](../assets/figures/ml_fig_c545_09.png)
*Figure — Cross-val MSE path c545. Synthetic teaching geometry—not a causal claim.*


![c546 teaching panel 09 (original).](../assets/figures/ml_fig_c546_09.png)
*Figure — Partial residual path c546. Synthetic teaching geometry—not a causal claim.*


![c547 teaching panel 09 (original).](../assets/figures/ml_fig_c547_09.png)
*Figure — Isotonic mono path c547. Synthetic teaching geometry—not a causal claim.*


![c548 teaching panel 09 (original).](../assets/figures/ml_fig_c548_09.png)
*Figure — Expectile residual path c548. Synthetic teaching geometry—not a causal claim.*


![c549 teaching panel 09 (original).](../assets/figures/ml_fig_c549_09.png)
*Figure — OLS residual QQ path c549. Synthetic teaching geometry—not a causal claim.*


![c550 teaching panel 09 (original).](../assets/figures/ml_fig_c550_09.png)
*Figure — Ridge path residual c550. Synthetic teaching geometry—not a causal claim.*


![c551 teaching panel 09 (original).](../assets/figures/ml_fig_c551_09.png)
*Figure — Lasso soft-threshold path c551. Synthetic teaching geometry—not a causal claim.*


![c552 teaching panel 09 (original).](../assets/figures/ml_fig_c552_09.png)
*Figure — Elastic net path c552. Synthetic teaching geometry—not a causal claim.*


![c553 teaching panel 09 (original).](../assets/figures/ml_fig_c553_09.png)
*Figure — Huber residual cost c553. Synthetic teaching geometry—not a causal claim.*


![c554 teaching panel 09 (original).](../assets/figures/ml_fig_c554_09.png)
*Figure — Quantile pinball path c554. Synthetic teaching geometry—not a causal claim.*


![c555 teaching panel 09 (original).](../assets/figures/ml_fig_c555_09.png)
*Figure — Poisson deviance path c555. Synthetic teaching geometry—not a causal claim.*


![c556 teaching panel 09 (original).](../assets/figures/ml_fig_c556_09.png)
*Figure — Tweedie deviance path c556. Synthetic teaching geometry—not a causal claim.*


![c557 teaching panel 09 (original).](../assets/figures/ml_fig_c557_09.png)
*Figure — GAM smooth residual c557. Synthetic teaching geometry—not a causal claim.*


![c558 teaching panel 09 (original).](../assets/figures/ml_fig_c558_09.png)
*Figure — Spline knot residual c558. Synthetic teaching geometry—not a causal claim.*


![c559 teaching panel 09 (original).](../assets/figures/ml_fig_c559_09.png)
*Figure — Heteroscedasticity path c559. Synthetic teaching geometry—not a causal claim.*


![c560 teaching panel 09 (original).](../assets/figures/ml_fig_c560_09.png)
*Figure — Leverage cook bars c560. Synthetic teaching geometry—not a causal claim.*


![c561 teaching panel 09 (original).](../assets/figures/ml_fig_c561_09.png)
*Figure — Cross-val MSE path c561. Synthetic teaching geometry—not a causal claim.*


![c562 teaching panel 09 (original).](../assets/figures/ml_fig_c562_09.png)
*Figure — Partial residual path c562. Synthetic teaching geometry—not a causal claim.*


![c563 teaching panel 09 (original).](../assets/figures/ml_fig_c563_09.png)
*Figure — Isotonic mono path c563. Synthetic teaching geometry—not a causal claim.*


![c564 teaching panel 09 (original).](../assets/figures/ml_fig_c564_09.png)
*Figure — Expectile residual path c564. Synthetic teaching geometry—not a causal claim.*


![c565 teaching panel 09 (original).](../assets/figures/ml_fig_c565_09.png)
*Figure — OLS residual QQ path c565. Synthetic teaching geometry—not a causal claim.*


![c566 teaching panel 09 (original).](../assets/figures/ml_fig_c566_09.png)
*Figure — Ridge path residual c566. Synthetic teaching geometry—not a causal claim.*


![c567 teaching panel 09 (original).](../assets/figures/ml_fig_c567_09.png)
*Figure — Lasso soft-threshold path c567. Synthetic teaching geometry—not a causal claim.*


![c568 teaching panel 09 (original).](../assets/figures/ml_fig_c568_09.png)
*Figure — Elastic net path c568. Synthetic teaching geometry—not a causal claim.*


![c569 teaching panel 09 (original).](../assets/figures/ml_fig_c569_09.png)
*Figure — Huber residual cost c569. Synthetic teaching geometry—not a causal claim.*


![c570 teaching panel 09 (original).](../assets/figures/ml_fig_c570_09.png)
*Figure — Quantile pinball path c570. Synthetic teaching geometry—not a causal claim.*


![c571 teaching panel 09 (original).](../assets/figures/ml_fig_c571_09.png)
*Figure — Poisson deviance path c571. Synthetic teaching geometry—not a causal claim.*


![c572 teaching panel 09 (original).](../assets/figures/ml_fig_c572_09.png)
*Figure — Tweedie deviance path c572. Synthetic teaching geometry—not a causal claim.*


![c573 teaching panel 09 (original).](../assets/figures/ml_fig_c573_09.png)
*Figure — GAM smooth residual c573. Synthetic teaching geometry—not a causal claim.*


![c574 teaching panel 09 (original).](../assets/figures/ml_fig_c574_09.png)
*Figure — Spline knot residual c574. Synthetic teaching geometry—not a causal claim.*


![c575 teaching panel 09 (original).](../assets/figures/ml_fig_c575_09.png)
*Figure — Heteroscedasticity path c575. Synthetic teaching geometry—not a causal claim.*


![c576 teaching panel 09 (original).](../assets/figures/ml_fig_c576_09.png)
*Figure — Leverage cook bars c576. Synthetic teaching geometry—not a causal claim.*


![c577 teaching panel 09 (original).](../assets/figures/ml_fig_c577_09.png)
*Figure — Cross-val MSE path c577. Synthetic teaching geometry—not a causal claim.*


![c578 teaching panel 09 (original).](../assets/figures/ml_fig_c578_09.png)
*Figure — Partial residual path c578. Synthetic teaching geometry—not a causal claim.*


![c579 teaching panel 09 (original).](../assets/figures/ml_fig_c579_09.png)
*Figure — Isotonic mono path c579. Synthetic teaching geometry—not a causal claim.*


![c580 teaching panel 09 (original).](../assets/figures/ml_fig_c580_09.png)
*Figure — Expectile residual path c580. Synthetic teaching geometry—not a causal claim.*


![c581 teaching panel 09 (original).](../assets/figures/ml_fig_c581_09.png)
*Figure — OLS residual QQ path c581. Synthetic teaching geometry—not a causal claim.*


![c582 teaching panel 09 (original).](../assets/figures/ml_fig_c582_09.png)
*Figure — Ridge path residual c582. Synthetic teaching geometry—not a causal claim.*


![c583 teaching panel 09 (original).](../assets/figures/ml_fig_c583_09.png)
*Figure — Lasso soft-threshold path c583. Synthetic teaching geometry—not a causal claim.*


![c584 teaching panel 09 (original).](../assets/figures/ml_fig_c584_09.png)
*Figure — Elastic net path c584. Synthetic teaching geometry—not a causal claim.*


![c585 teaching panel 09 (original).](../assets/figures/ml_fig_c585_09.png)
*Figure — Huber residual cost c585. Synthetic teaching geometry—not a causal claim.*


![c586 teaching panel 09 (original).](../assets/figures/ml_fig_c586_09.png)
*Figure — Quantile pinball path c586. Synthetic teaching geometry—not a causal claim.*


![c587 teaching panel 09 (original).](../assets/figures/ml_fig_c587_09.png)
*Figure — Poisson deviance path c587. Synthetic teaching geometry—not a causal claim.*


![c588 teaching panel 09 (original).](../assets/figures/ml_fig_c588_09.png)
*Figure — Tweedie deviance path c588. Synthetic teaching geometry—not a causal claim.*


![c589 teaching panel 09 (original).](../assets/figures/ml_fig_c589_09.png)
*Figure — GAM smooth residual c589. Synthetic teaching geometry—not a causal claim.*


![c590 teaching panel 09 (original).](../assets/figures/ml_fig_c590_09.png)
*Figure — Spline knot residual c590. Synthetic teaching geometry—not a causal claim.*


![c591 teaching panel 09 (original).](../assets/figures/ml_fig_c591_09.png)
*Figure — Heteroscedasticity path c591. Synthetic teaching geometry—not a causal claim.*


![c592 teaching panel 09 (original).](../assets/figures/ml_fig_c592_09.png)
*Figure — Leverage cook bars c592. Synthetic teaching geometry—not a causal claim.*


![c593 teaching panel 09 (original).](../assets/figures/ml_fig_c593_09.png)
*Figure — Cross-val MSE path c593. Synthetic teaching geometry—not a causal claim.*


![c594 teaching panel 09 (original).](../assets/figures/ml_fig_c594_09.png)
*Figure — Partial residual path c594. Synthetic teaching geometry—not a causal claim.*


![c595 teaching panel 09 (original).](../assets/figures/ml_fig_c595_09.png)
*Figure — Isotonic mono path c595. Synthetic teaching geometry—not a causal claim.*


![c596 teaching panel 09 (original).](../assets/figures/ml_fig_c596_09.png)
*Figure — Expectile residual path c596. Synthetic teaching geometry—not a causal claim.*


![c597 teaching panel 09 (original).](../assets/figures/ml_fig_c597_09.png)
*Figure — OLS residual QQ path c597. Synthetic teaching geometry—not a causal claim.*


![c598 teaching panel 09 (original).](../assets/figures/ml_fig_c598_09.png)
*Figure — Ridge path residual c598. Synthetic teaching geometry—not a causal claim.*


![c599 teaching panel 09 (original).](../assets/figures/ml_fig_c599_09.png)
*Figure — Lasso soft-threshold path c599. Synthetic teaching geometry—not a causal claim.*


![c600 teaching panel 09 (original).](../assets/figures/ml_fig_c600_09.png)
*Figure — Elastic net path c600. Synthetic teaching geometry—not a causal claim.*


![c601 teaching panel 09 (original).](../assets/figures/ml_fig_c601_09.png)
*Figure — Huber residual cost c601. Synthetic teaching geometry—not a causal claim.*


![c602 teaching panel 09 (original).](../assets/figures/ml_fig_c602_09.png)
*Figure — Quantile pinball path c602. Synthetic teaching geometry—not a causal claim.*


![c603 teaching panel 09 (original).](../assets/figures/ml_fig_c603_09.png)
*Figure — Poisson deviance path c603. Synthetic teaching geometry—not a causal claim.*


![c604 teaching panel 09 (original).](../assets/figures/ml_fig_c604_09.png)
*Figure — Tweedie deviance path c604. Synthetic teaching geometry—not a causal claim.*


![c605 teaching panel 09 (original).](../assets/figures/ml_fig_c605_09.png)
*Figure — GAM smooth residual c605. Synthetic teaching geometry—not a causal claim.*


![c606 teaching panel 09 (original).](../assets/figures/ml_fig_c606_09.png)
*Figure — Spline knot residual c606. Synthetic teaching geometry—not a causal claim.*


![c607 teaching panel 09 (original).](../assets/figures/ml_fig_c607_09.png)
*Figure — Heteroscedasticity path c607. Synthetic teaching geometry—not a causal claim.*


![c608 teaching panel 09 (original).](../assets/figures/ml_fig_c608_09.png)
*Figure — Leverage cook bars c608. Synthetic teaching geometry—not a causal claim.*


![c609 teaching panel 09 (original).](../assets/figures/ml_fig_c609_09.png)
*Figure — Cross-val MSE path c609. Synthetic teaching geometry—not a causal claim.*


![c610 teaching panel 09 (original).](../assets/figures/ml_fig_c610_09.png)
*Figure — Partial residual path c610. Synthetic teaching geometry—not a causal claim.*


![c611 teaching panel 09 (original).](../assets/figures/ml_fig_c611_09.png)
*Figure — Isotonic mono path c611. Synthetic teaching geometry—not a causal claim.*


![c612 teaching panel 09 (original).](../assets/figures/ml_fig_c612_09.png)
*Figure — Expectile residual path c612. Synthetic teaching geometry—not a causal claim.*


![c613 teaching panel 09 (original).](../assets/figures/ml_fig_c613_09.png)
*Figure — OLS residual QQ path c613. Synthetic teaching geometry—not a causal claim.*


![c614 teaching panel 09 (original).](../assets/figures/ml_fig_c614_09.png)
*Figure — Ridge path residual c614. Synthetic teaching geometry—not a causal claim.*


![c615 teaching panel 09 (original).](../assets/figures/ml_fig_c615_09.png)
*Figure — Lasso soft-threshold path c615. Synthetic teaching geometry—not a causal claim.*


![c616 teaching panel 09 (original).](../assets/figures/ml_fig_c616_09.png)
*Figure — Elastic net path c616. Synthetic teaching geometry—not a causal claim.*


![c617 teaching panel 09 (original).](../assets/figures/ml_fig_c617_09.png)
*Figure — Huber residual cost c617. Synthetic teaching geometry—not a causal claim.*


![c618 teaching panel 09 (original).](../assets/figures/ml_fig_c618_09.png)
*Figure — Quantile pinball path c618. Synthetic teaching geometry—not a causal claim.*


![c619 teaching panel 09 (original).](../assets/figures/ml_fig_c619_09.png)
*Figure — Poisson deviance path c619. Synthetic teaching geometry—not a causal claim.*


![c620 teaching panel 09 (original).](../assets/figures/ml_fig_c620_09.png)
*Figure — Tweedie deviance path c620. Synthetic teaching geometry—not a causal claim.*


![c621 teaching panel 09 (original).](../assets/figures/ml_fig_c621_09.png)
*Figure — GAM smooth residual c621. Synthetic teaching geometry—not a causal claim.*


![c622 teaching panel 09 (original).](../assets/figures/ml_fig_c622_09.png)
*Figure — Spline knot residual c622. Synthetic teaching geometry—not a causal claim.*


![c623 teaching panel 09 (original).](../assets/figures/ml_fig_c623_09.png)
*Figure — Heteroscedasticity path c623. Synthetic teaching geometry—not a causal claim.*


![c624 teaching panel 09 (original).](../assets/figures/ml_fig_c624_09.png)
*Figure — Leverage cook bars c624. Synthetic teaching geometry—not a causal claim.*


![c625 teaching panel 09 (original).](../assets/figures/ml_fig_c625_09.png)
*Figure — Cross-val MSE path c625. Synthetic teaching geometry—not a causal claim.*


![c626 teaching panel 09 (original).](../assets/figures/ml_fig_c626_09.png)
*Figure — Partial residual path c626. Synthetic teaching geometry—not a causal claim.*


![c627 teaching panel 09 (original).](../assets/figures/ml_fig_c627_09.png)
*Figure — Isotonic mono path c627. Synthetic teaching geometry—not a causal claim.*


![c628 teaching panel 09 (original).](../assets/figures/ml_fig_c628_09.png)
*Figure — Expectile residual path c628. Synthetic teaching geometry—not a causal claim.*


![c629 teaching panel 09 (original).](../assets/figures/ml_fig_c629_09.png)
*Figure — OLS residual QQ path c629. Synthetic teaching geometry—not a causal claim.*


![c630 teaching panel 09 (original).](../assets/figures/ml_fig_c630_09.png)
*Figure — Ridge path residual c630. Synthetic teaching geometry—not a causal claim.*


![c631 teaching panel 09 (original).](../assets/figures/ml_fig_c631_09.png)
*Figure — Lasso soft-threshold path c631. Synthetic teaching geometry—not a causal claim.*


![c632 teaching panel 09 (original).](../assets/figures/ml_fig_c632_09.png)
*Figure — Elastic net path c632. Synthetic teaching geometry—not a causal claim.*


![c633 teaching panel 09 (original).](../assets/figures/ml_fig_c633_09.png)
*Figure — Huber residual cost c633. Synthetic teaching geometry—not a causal claim.*


![c634 teaching panel 09 (original).](../assets/figures/ml_fig_c634_09.png)
*Figure — Quantile pinball path c634. Synthetic teaching geometry—not a causal claim.*


![c635 teaching panel 09 (original).](../assets/figures/ml_fig_c635_09.png)
*Figure — Poisson deviance path c635. Synthetic teaching geometry—not a causal claim.*


![c636 teaching panel 09 (original).](../assets/figures/ml_fig_c636_09.png)
*Figure — Tweedie deviance path c636. Synthetic teaching geometry—not a causal claim.*


![c637 teaching panel 09 (original).](../assets/figures/ml_fig_c637_09.png)
*Figure — GAM smooth residual c637. Synthetic teaching geometry—not a causal claim.*


![c638 teaching panel 09 (original).](../assets/figures/ml_fig_c638_09.png)
*Figure — Spline knot residual c638. Synthetic teaching geometry—not a causal claim.*


![c639 teaching panel 09 (original).](../assets/figures/ml_fig_c639_09.png)
*Figure — Heteroscedasticity path c639. Synthetic teaching geometry—not a causal claim.*


![c640 teaching panel 09 (original).](../assets/figures/ml_fig_c640_09.png)
*Figure — Leverage cook bars c640. Synthetic teaching geometry—not a causal claim.*


![c641 teaching panel 09 (original).](../assets/figures/ml_fig_c641_09.png)
*Figure — Cross-val MSE path c641. Synthetic teaching geometry—not a causal claim.*


![c642 teaching panel 09 (original).](../assets/figures/ml_fig_c642_09.png)
*Figure — Partial residual path c642. Synthetic teaching geometry—not a causal claim.*


![c643 teaching panel 09 (original).](../assets/figures/ml_fig_c643_09.png)
*Figure — Isotonic mono path c643. Synthetic teaching geometry—not a causal claim.*


![c644 teaching panel 09 (original).](../assets/figures/ml_fig_c644_09.png)
*Figure — Expectile residual path c644. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Loss measures per-example error; cost averages loss; objectives may add regularization. Univariate and multiple linear regression estimate conditional means by OLS, with closed-form solutions via normal equations; a full numerical NIHSS–volume example computes slopes, intercepts, and fitted values by hand. Polynomial and piecewise/spline models capture nonlinearity while remaining linear in parameters. RSE and RMSE quantify residual scale; residual plots and R² aid diagnosis but holdout metrics govern prediction claims. ARIMA(p,d,q) models temporal dependence after differencing. Logistic and softmax regression link linear predictors to probabilities via logit/softmax and train by likelihood. Evaluation uses CV, learning curves, ROC/AUC (with calibration), Wald and LRTs, information criteria, and pseudo-R². Overfitting and underfitting reflect the bias–variance tradeoff; Ridge, Lasso, elastic net, and the non-negative garrote implement shrinkage and selection. Optimization rests on gradients, Jacobians, Hessians, and Taylor expansions; batch/SGD/mini-batch gradient descent, Newton methods, and early stopping are the workhorse algorithms. Clinical use demands honest validation, calibration, and clear separation of prediction from causal inference.

## Practice and Reflection

(1) Using the four-point NIHSS–volume example, compute all residuals, RSS, RSE (with p=1), and R².

(2) Show that the OLS estimator satisfies the normal equations XᵀX β̂ = Xᵀy by setting ∇_β ‖y−Xβ‖² = 0.

(3) For ridge regression in one dimension (no intercept), derive β̂ = (∑ x_i y_i)/(∑ x_i² + λ) and interpret λ → 0 and λ → ∞.

(4) Explain why shuffled 10-fold CV is misleading for selecting ARIMA orders on monthly stroke admissions.

(5) A logistic model predicts sICH. Sketch axes of an ROC curve and state two reasons high AUC is insufficient for deployment.

(6) Compare Wald and likelihood-ratio tests for a single coefficient in logistic regression under near-separation.

(7) Sketch learning curves for (a) high bias and (b) high variance models and propose one remedy for each in a stroke registry.

(8) Contrast Lasso and elastic net when two labs are nearly collinear predictors of creatinine clearance.

(9) Write one gradient-descent update for minimizing (1/2)(β − 3)² starting at β=0 with η=0.2; iterate three steps.

(10) Define early stopping and describe a data-splitting scheme that keeps final test performance honest.

(11) You want the effect of door-to-needle time on discharge mRS. Why might OLS with every available EHR feature be the wrong tool?

(12) Softmax with K=3 outcome classes has how many free coefficient vectors under the usual identifiability constraint, and why is the constraint needed?
