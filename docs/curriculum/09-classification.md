# Chapter 9. Classification


![09 Supervised Map](../assets/figures/09_supervised_map.png)


## Opening
![Confusion matrix with synthetic counts (original).](../assets/figures/ml_fig_confusion_annotated.png)

*Confusion matrix with synthetic counts (original).*


A binary classifier flags LVO on CTA with impressive accuracy in the training center. Classification literacy means thresholds, class imbalance, costs of false negatives on the stroke pathway, and external validation—not leaderboard ego.


![Confusion matrix and ROC for a synthetic classifier (original).](../assets/figures/ml_fig_confusion_roc.png)

*Confusion matrix and ROC for a synthetic classifier (original).*
## Learning Objectives

Formulate binary, multiclass, and multilabel classification problems and interpret decision boundaries and margins.

Apply rule-based classifiers, Naive Bayes (including Gaussian with a worked prediction), k-NN with Voronoi geometry and KD-tree/LSH search, and soft-margin kernel SVMs.

Grow and prune decision trees with ID3, CHAID, C4.5, and CART criteria; state computational complexity.

Build ensembles: bagging, boosting, stacking, random forests, AdaBoost, GBDT for regression and classification, XGBoost, LightGBM, and CatBoost.

Select models with nested validation; compute accuracy, precision, recall, F1, ROC-AUC, PR metrics, and calibration under class imbalance.

Work numerical confusion-matrix and Naive Bayes examples end-to-end.

Map classifiers and metrics to neurologic prediction with phenotype labels, index-time features, and cost-sensitive thresholds.

## 9.1 The Classification Problem

Classification is supervised learning with a discrete target. Given training pairs (x_i, y_i) where x_i lives in a feature space X and y_i belongs to a finite label set Y = {1, …, K}, the goal is to learn a function f: X → Y that assigns labels to new inputs. When K = 2 we speak of binary classification; when K > 2 we speak of multiclass classification. Multilabel problems allow several labels to be active simultaneously and are usually reduced to collections of binary problems or modeled with structured outputs. Throughout this chapter we emphasize mutually exclusive classes unless multilabel is stated.

![9.1: Three classifiers induce three boundary geometries on one two-class toy set. k-NN (k = 3) carves piecewise regions that ](../assets/figures/ml_concept_9.1_ac880ae5.png)

*Figure 9.1 — original teaching graphic.*

A probabilistic view is often more useful than a hard assignment. Many algorithms estimate scores s_k(x) or class posteriors p(y = k | x). Hard predictions then arise by choosing argmax_k s_k(x), or by thresholding a single score for binary problems. Separating scoring from decision is crucial: the same model can support different operating points for screening versus confirmation, and evaluation should sometimes assess ranking quality rather than a single fixed threshold. Bayes-optimal classification minimizes expected risk under a loss. With 0–1 loss the optimal rule predicts the maximum posterior class; with unequal misclassification costs the optimal threshold shifts. Thinking in terms of risk clarifies why accuracy alone is not always the right objective and why class priors matter.

Inputs: feature vectors (tabular), tokens (text), pixels (images), graphs, or other structured objects.

Outputs: class labels or probability distributions over labels.

Training: minimize empirical risk with optional regularization on held-out validation data.

Inference: score, threshold or argmax, optionally calibrate probabilities.

A decision boundary is the set of points where the predicted class changes. Linear classifiers induce hyperplane boundaries w^T x + b = 0. The margin is the geometric distance from the hyperplane to the nearest training points of either class—an idea that reappears in support vector machines. Nonlinear boundaries arise by feature maps φ(x) so that a linear rule in φ-space becomes curved in the original space, or by models such as trees and nearest neighbors that define piecewise regions. Capacity and overfitting are geometric as well as statistical: a classifier that carves arbitrary Voronoi cells will eventually memorize labels. When classes overlap heavily, even the Bayes-optimal boundary leaves irreducible error; the right response is better features or acceptance of residual risk, not ever more flexible boundaries on the same inputs.

## 9.2 Rule-Based Classifiers

Rule-based classifiers express the decision as an ordered or unordered collection of if–then rules of the form IF condition THEN class. Conditions are conjunctions (and sometimes disjunctions) of tests on features: thresholds on continuous variables, membership in categorical sets, or presence of clinical codes. Classical inductive rule learners such as sequential covering algorithms grow a rule that covers many positives and few negatives, remove covered examples, and repeat until residual coverage is small. Decision lists order rules so that the first matching rule fires; decision sets may aggregate votes or resolve conflicts by rule specificity or accuracy.

Rules are attractive when interpretability and auditability dominate. A stroke-code pathway written as IF NIHSS ≥ 6 AND last-known-well ≤ 24 h AND no ICH on NCCT THEN activate CTA pathway is a human-authored rule classifier. Learned rules can approximate tree leaves flattened into logical form. Strengths include transparent logic, easy incorporation of hard clinical constraints, and straightforward override by domain experts. Weaknesses include brittleness at continuous boundaries, difficulty capturing smooth probability gradients, and combinatorial search cost when many candidate conditions exist. Hybrid systems use rules for eligibility gates and statistical models for risk scores inside the eligible set—an architecture that mirrors how stroke systems already separate inclusion criteria from prognostic estimation.

Evaluation of rule systems should still use held-out data. Apparent clarity of a rule does not guarantee transportability: thresholds optimized at one comprehensive center may misfire at a primary center with different case mix. Prefer reporting coverage (fraction of cases matched by any rule), accuracy among covered cases, and behavior of a default rule for unmatched cases. When rules are mined from data, prune using significance tests or minimum support/confidence thresholds analogous to association rule mining, and nest rule selection inside cross-validation to avoid selection bias.

## 9.3 Naive Bayes

Naive Bayes models the joint as p(x, y) = p(y) p(x | y) and makes the conditional independence assumption that features are independent given the class: p(x | y) = ∏_j p(x_j | y). Classification uses Bayes theorem: p(y | x) = p(x | y) p(y) / p(x), and since p(x) does not depend on y we rank classes by p(y) ∏_j p(x_j | y). Despite the often-false independence assumption, Naive Bayes is fast, robust to high-dimensional sparse data (classic in text classification with bag-of-words features), and produces surprisingly strong baselines when features are noisy and sample size is moderate.

### Bayes theorem and prediction

Bayes theorem formalizes updating beliefs about class given evidence. The prior p(y) encodes base rates—crucial in epidemiology when disease prevalence differs across settings. The likelihood p(x | y) encodes how features look under each class. The posterior p(y | x) is what we need for decision making under calibrated risk. In practice we compute in log space to avoid underflow: predict argmax_y [log p(y) + Σ_j log p(x_j | y)]. For binary or categorical features, store empirical frequencies with Laplace (add-one) or add-α smoothing so zero counts do not zero out the entire product. Multinomial Naive Bayes is the workhorse for count data such as word counts; Bernoulli Naive Bayes uses binary presence/absence features.

### Worked example: categorical Naive Bayes (toy table → posterior)

Consider a toy triage task: predict large-vessel occlusion (LVO, class L+) versus no LVO (L−) from two binary bedside features recorded at stroke-code activation—forced gaze deviation (Gaze) and severe arm weakness (Arm). A training set of 25 patients gives the counts below; each entry is the number of patients in that class with the feature present.

| Quantity | L+ (LVO) | L− (no LVO) |
| --- | --- | --- |
| Patients in class | 10 | 15 |
| Gaze deviation present | 8 | 3 |
| Severe arm weakness present | 7 | 3 |

The priors are P(L+) = 10/25 = 0.4 and P(L−) = 15/25 = 0.6. The class-conditional likelihoods read directly off the table: P(Gaze=yes | L+) = 8/10 = 0.8, P(Arm=yes | L+) = 7/10 = 0.7, P(Gaze=yes | L−) = 3/15 = 0.2, and P(Arm=yes | L−) = 3/15 = 0.2. For a query patient presenting with both signs (Gaze=yes, Arm=yes), form the unnormalized posterior score for each class as prior times the product of likelihoods—the naive independence assumption is exactly what licenses this multiplication:

score(L+) = P(L+) · P(Gaze=yes | L+) · P(Arm=yes | L+) = 0.4 · 0.8 · 0.7 = 0.224.

score(L−) = P(L−) · P(Gaze=yes | L−) · P(Arm=yes | L−) = 0.6 · 0.2 · 0.2 = 0.024.

Normalizing by their sum—this sum is the evidence term p(x) in Bayes theorem—gives the posterior P(L+ | x) = 0.224 / (0.224 + 0.024) = 0.224 / 0.248 ≈ 0.903, and P(L− | x) ≈ 0.097. A patient with both signs therefore carries roughly a 90% posterior probability of LVO. An odds-form cross-check verifies the arithmetic: prior odds are 0.4/0.6 = 0.667, the likelihood ratios are 0.8/0.2 = 4 for gaze and 0.7/0.2 = 3.5 for arm, so posterior odds = 0.667 · 4 · 3.5 = 9.33 and posterior probability = 9.33 / (1 + 9.33) ≈ 0.903—identical to the normalized computation.

If any cell were zero—say no L+ patient had ever shown some feature value—the product would collapse to zero regardless of all other evidence. Add-one (Laplace) smoothing prevents this veto by replacing a raw estimate c/n with (c + 1)/(n + v), where v is the number of values the feature can take. For a binary feature with a 0/10 cell the smoothed likelihood becomes (0 + 1)/(10 + 2) = 1/12 ≈ 0.083 rather than 0, so a single unseen combination no longer annihilates a class.

### Gaussian Naive Bayes

For continuous features, Gaussian Naive Bayes assumes x_j | y = k ~ N(μ_{k j}, σ²_{k j}). Parameters are estimated by class-conditional sample means and variances (with a floor on variance to prevent numerical blow-ups). The class-conditional density factors as a product of univariate Gaussians under the naive assumption. Alternatives include histograms, kernel density estimates, or semi-naive structures that allow a few feature dependencies. Gaussian NB is a strong first model for small tabular clinical datasets with roughly unimodal class-conditional features; highly skewed labs may benefit from log transforms before Gaussian modeling.

### Worked example: Gaussian Naive Bayes prediction

Suppose two classes, A and B, with equal priors p(A) = p(B) = 0.5. Feature x1 under A has mean 0 and variance 1; under B has mean 2 and variance 1. Feature x2 under A has mean 1 and variance 1; under B has mean 0 and variance 1. The Gaussian log-density (up to a shared constant c = −½ log(2πσ²)) is c − (x−μ)²/(2σ²). With σ² = 1 for all features, for query (x1, x2) = (1.0, 0.5): log p(x1 | A) = c − 0.5(1−0)² = c − 0.5; log p(x2 | A) = c − 0.5(0.5−1)² = c − 0.125; log p(x1 | B) = c − 0.5; log p(x2 | B) = c − 0.125. Class-conditional log-likelihoods match, so posteriors remain 0.5/0.5. Shift the query to (0.0, 1.0): log-likelihood score for A is 2c + 0; for B is 2c − 2.5. The score for A exceeds B by 2.5 in log space, so we predict A. Exponentiating and normalizing over the two classes (the two-class softmax) converts this log-score gap into an actual posterior: P(A | x) = 1/(1 + exp(−2.5)) ≈ 1/(1 + 0.082) ≈ 0.924. If we change priors to p(A) = 0.2, p(B) = 0.8, add log prior: score_A = 2c + log 0.2, score_B = 2c − 2.5 + log 0.8. Numerically log 0.2 ≈ −1.609 and log 0.8 ≈ −0.223, so score_A − score_B ≈ 2.5 − 1.609 + 0.223 ≈ 1.114 > 0 and A still wins, but less decisively. In posterior terms P(A | x) = 1/(1 + exp(−1.114)) ≈ 0.753, down from 0.924, so the informative prior pulls the decision toward B without flipping it. This arithmetic shows how means, variances, priors, and the product of likelihoods drive the decision under the naive independence assumption.

## 9.4 k-Nearest Neighbors

The k-nearest neighbor (k-NN) classifier is a nonparametric instance-based method. To label a query x, find the k training points closest to x under a distance d (often Euclidean after feature scaling), and predict by majority vote among their labels. Probability estimates can be class frequencies in the neighborhood, optionally distance-weighted. No parametric model is fit; computation is deferred to prediction time, which is costly for large n unless accelerated.

The hyperparameter k controls bias–variance trade-off. Small k yields highly local, low-bias, high-variance decisions; large k smooths estimates toward the global class prior. Distance metrics and feature scaling are as important as k: unnormalized features with large ranges dominate Euclidean distance. Asymptotically, under mild conditions, 1-NN error is at most twice the Bayes error in the large-sample limit for binary problems. High-dimensional spaces suffer distance concentration; dimensionality reduction or learned metrics can restore neighborhood structure.

### Voronoi tessellation

For 1-NN with Euclidean distance, the decision regions are exactly the Voronoi cells of the training points: each point owns the polyhedral region of space closer to it than to any other training point. Class labels paint these cells; the decision boundary consists of facets of the Voronoi diagram between differently labeled neighbors. This geometry makes clear why noisy labels create jagged islands of error and why editing rules that remove suspicious training points can smooth the boundary. For k > 1 the regions are more complex intersections of distance orderings, but the same spatial intuition applies.

### KD-trees

A KD-tree (k-dimensional tree) is a binary space-partitioning structure that recursively splits data along coordinate axes, cycling through dimensions or choosing the widest spread dimension. Nearest-neighbor queries traverse the tree, pruning branches whose bounding boxes cannot contain a closer point than the current best. In low-to-moderate dimensions (roughly d ≲ 20 with favorable structure) KD-trees yield sublinear query time on average. In high dimensions pruning fails and query cost approaches brute force—the curse of dimensionality in geometric form. Ball trees and cover trees are related structures that partition by hyperspheres rather than axis-aligned cuts.

### Locality-sensitive hashing (LSH)

Locality-sensitive hashing approximates nearest neighbors by hashing items so that similar points collide with high probability. A family of hash functions is locality-sensitive for a distance if P[h(x) = h(y)] is high when d(x, y) is small and low when d(x, y) is large. Classic constructions include random hyperplane hashes for cosine similarity and p-stable projections for Euclidean distance. Multiple hash tables and concatenated hash bits trade recall against speed. LSH is the right tool when n is huge, approximate neighbors suffice, and exact KD-tree search is too slow. In clinical retrieval—finding historically similar stroke cases for mortality review—approximate neighbors are often adequate if features are carefully standardized.

## 9.5 Support Vector Machines

A hard-margin support vector machine (SVM) finds the maximum-margin separating hyperplane between linearly separable classes. In canonical form the constraints are y_i (w^T x_i + b) ≥ 1 and the objective minimizes (1/2)‖w‖², which maximizes the geometric margin 2/‖w‖. Support vectors are the training points that lie on the margin boundaries and determine the solution; points farther away can often be removed without changing the hyperplane. Soft-margin SVM introduces slack variables ξ_i ≥ 0 allowing margin violations and misclassifications, with objective (1/2)‖w‖² + C Σ_i ξ_i. Large C insists on fitting training data tightly; small C prefers a wider margin with more training errors.

![9.2: The maximum-margin separating hyperplane w·x + b = 0 with its two margin boundaries (dashed, w·x + b = ±1). The four cir](../assets/figures/ml_concept_9.2_7aa2976a.png)

*Figure 9.2 — original teaching graphic.*

### Kernel trick and kernel functions

When classes are not linearly separable in input space, map inputs with φ(x) and learn a linear SVM in feature space. The dual formulation expresses the decision function as f(x) = Σ_i α_i y_i K(x_i, x) + b where K(x, x’) = ⟨φ(x), φ(x’)⟩. The kernel trick computes these inner products without materializing φ. Common kernels include linear K = x^T x’; polynomial K = (γ x^T x’ + r)^d; and radial basis function (RBF) K = exp(−γ ‖x − x’‖²). RBF kernels correspond to infinite-dimensional feature maps and can create highly flexible boundaries controlled by γ (length-scale) and C. Kernel methods trade interpretability for flexibility; hyperparameters require careful validation on held-out folds.

### Multilabel classification for SVM

Standard SVM is binary. Multiclass extensions use one-vs-rest (train K binary SVMs) or one-vs-one (train K(K−1)/2 pairwise SVMs and aggregate votes). Multilabel problems—for example simultaneous flags for atrial fibrillation, large-vessel occlusion, and hemorrhagic transformation risk—use a binary SVM per label (binary relevance), classifier chains that condition later labels on earlier predictions, or adapted ranking formulations. Label correlations matter: treating labels independently is simple but can produce incoherent combinations; chains and structured outputs capture dependence at higher cost.

![Multi-label metrics: Hamming loss vs exact-match vs micro-F1 (synthetic; original).](../assets/figures/ml_fig_hamming_multilabel.png)

*Figure — Multi-label scores are not interchangeable. **Left:** as the decision threshold moves, Hamming loss (mean per-label error), exact-match failure, and 1−micro-F1 peak in different places—exact match is harsh when many labels can fire. **Right:** per-label accuracy at the Hamming-best threshold still varies across clinical flags. Binary relevance ignores comorbidity structure; never treat co-predicted labels as a causal disease graph.*

### Prediction and computational complexity

Prediction for a kernel SVM sums kernel evaluations against support vectors: cost is O(n_sv · d_eff) per query where d_eff is the cost of one kernel evaluation (often O(d) for RBF). Training via quadratic programming is roughly between O(n²) and O(n³) depending on solver and cache, with linear SVMs scalable via specialized methods (LIBLINEAR, Pegasos-style stochastic subgradient) at near O(n d) per epoch. For very large datasets, linear SVMs or approximate kernel expansions (random Fourier features, Nyström) scale better than full kernel QP. Conceptually, SVM optimizes a hinge-loss style objective with margin regularization rather than log loss; probabilities require post-hoc calibration (Platt scaling) if needed.

## 9.6 Decision Trees: ID3, CHAID, C4.5, and CART

A decision tree recursively partitions the feature space. At each internal node a test on one or more features routes examples left or right (or into multiway branches). Leaves store class majorities or class probability estimates. Prediction follows the unique path from root to leaf consistent with the input. Trees handle mixed feature types, provide transparent decision rules, and capture interactions without manual cross-products—but single trees are unstable: small data changes can alter top splits substantially.

![9.3: A decision tree for bedside large-vessel-occlusion (LVO) triage. Each internal (indigo) node applies one condition—NIHSS](../assets/figures/ml_concept_9.3_f1a6bd23.png)

*Figure 9.3 — original teaching graphic.*

### ID3

Iterative Dichotomiser 3 (ID3) grows trees on categorical features using information gain: the decrease in Shannon entropy H = −Σ_k p_k log p_k from parent to weighted children. At each node ID3 chooses the feature that maximizes gain, branches on all values of that feature (multiway split), and recurses until pure leaves or no features remain. ID3 does not natively handle continuous features or missing values and tends to favor features with many levels because multiway splits can drive child entropy to zero. It is the conceptual ancestor of later algorithms that correct these limitations.

### CHAID

Chi-square Automatic Interaction Detector (CHAID) uses chi-square tests of independence between candidate predictors and the target to choose splits, with multiway partitions formed by merging category levels that are statistically homogeneous. Stopping is guided by significance thresholds (p-values) rather than pure impurity greed. CHAID is popular in marketing and survey analysis and can produce interpretable multi-branch trees. In clinical research it appears when investigators want association-test logic aligned with classical contingency-table thinking; care is required with multiple testing and with continuous predictors that must be binned.

### C4.5

C4.5 extends ID3 with gain ratio (information gain normalized by split intrinsic information) to reduce multi-valued feature bias, handles continuous features by evaluating binary thresholds, supports missing values via fractional instance weights, and applies error-based pruning after growth. It can also convert trees to rule sets. C4.5 remains a reference algorithm in textbooks; many production systems use CART-style binary trees for software simplicity while retaining C4.5 ideas about gain ratio and pruning philosophy.

### CART

Classification and Regression Trees (CART) grow binary trees. For classification, splits minimize weighted child impurity using Gini impurity G = 1 − Σ_k p_k² (or entropy). For regression, splits minimize weighted sum of squared errors in children. Continuous and ordered features are handled by threshold search; categorical features by subset splits (with computational shortcuts). CART uses cost-complexity pruning: grow a large tree, then prune subtrees trading training impurity against a penalty α times number of leaves, choosing α by cross-validation. Random forests and gradient boosting almost always use CART-style binary trees as base learners.

### Worked example: one Gini split

A node has 10 positives and 10 negatives (Gini = 1 − 0.5² − 0.5² = 0.5). A candidate split sends 8 positives and 2 negatives left, and 2 positives and 8 negatives right. Left Gini = 1 − 0.8² − 0.2² = 0.32; right Gini = 0.32. Weighted child impurity = 0.5·0.32 + 0.5·0.32 = 0.32. Impurity decrease = 0.5 − 0.32 = 0.18. A split into pure leaves (10/0 and 0/10) would give decrease 0.5 and be preferred. This is the arithmetic of greedy tree growth.

### Pruning and computational complexity

Unpruned trees overfit by isolating individual noisy points. Pre-pruning stops growth early (max depth, min samples per leaf, min impurity decrease). Post-pruning grows fully then removes branches that do not improve validation error (reduced-error pruning) or that fail cost-complexity tests. Depth limits and min-leaf sizes are the practical knobs in ensemble settings where individual trees are intentionally deep (forests) or shallow (boosting stumps). Training a binary tree by exhaustive threshold search costs roughly O(d n log n) with sorted features per node across the tree construction, though constants and histogram approximations vary. Prediction is O(depth), typically O(log n) for balanced trees but O(n) worst-case for pathological chains. Memory stores O(nodes) split parameters—small compared with kernel SVMs that store many support vectors.

## 9.7 Ensemble Learning Methods

### Bagging, boosting, and stacking

Bootstrap aggregating (bagging) trains base learners on bootstrap resamples of the training set and aggregates predictions by majority vote or averaging. Bagging primarily reduces variance of unstable learners such as deep trees. Boosting builds an additive ensemble sequentially, each new learner focusing on residual errors or reweighted hard examples, primarily reducing bias (and potentially variance with shrinkage). Stacking trains a meta-learner on out-of-fold predictions of heterogeneous base models so that the combiner learns when to trust each expert. Soft voting averages calibrated probabilities; hard voting averages labels. Ensembles help most when members make different errors.

![9.4: Three ensemble architectures. Bagging trains base learners in parallel on bootstrap resamples and votes/averages to redu](../assets/figures/ml_concept_9.4_d9210e0b.png)

*Figure 9.4 — original teaching graphic.*

### Random forests

Random forests bag many deep trees and, at each split, consider only a random subset of features (typically √d for classification). Feature randomness decorrelates trees so averaging reduces variance more effectively than bagging alone. Out-of-bag (OOB) samples provide a built-in estimate of generalization without a separate validation set, though careful practitioners still keep a final held-out test. Feature importance from impurity decrease or permutation is heuristic and can bias toward high-cardinality features. Random forests are strong off-the-shelf baselines for heterogeneous tabular data: little scaling needed, nonlinearities and interactions captured, relatively robust hyperparameters within a sensible range.

### AdaBoost

Adaptive Boosting (AdaBoost) maintains a distribution of weights over training examples. After each weak learner (often a depth-1 stump), increase weights of misclassified points and decrease weights of correctly classified ones. The weak learner’s weight in the final vote is α_t = ½ log((1−ε_t)/ε_t) where ε_t is its weighted error; final prediction is sign(Σ_t α_t h_t(x)). Under assumptions, training error decreases exponentially with rounds, though generalization still requires control of complexity and label noise. AdaBoost can over-emphasize outliers; robust variants and early stopping mitigate this.

### Gradient boosting decision trees (GBDT)

Gradient boosting machines cast boosting as functional gradient descent on a loss. At iteration t, compute pseudo-residuals r_i = −∂L(y_i, F_{t−1}(x_i))/∂F_{t−1}(x_i), fit a regression tree h_t to the residuals, and update F_t = F_{t−1} + ν h_t with shrinkage learning rate ν ∈ (0,1]. For squared error regression, residuals are ordinary prediction errors y − F. For logistic classification, residuals are y − p where p = σ(F), and trees fit these probability errors in logit space. Classification with GBDT thus reuses regression-tree machinery on gradient vectors, with leaf values chosen to minimize the logistic loss given the second-order structure of the loss when available.

Hyperparameters that matter most: number of trees, learning rate (smaller needs more trees), max depth or number of leaves, subsample row fraction, column subsample, and minimum child weight. Early stopping on a validation set is essential. GBDT is often state of the art on heterogeneous tabular data in health care and industry, but it overfits if trees are deep and learning rates large without regularization.

### XGBoost

eXtreme Gradient Boosting (XGBoost) popularized a regularized second-order boosting objective: for each leaf, approximate the loss with gradient g_i and Hessian h_i, and penalize leaf weights with L2 (and optionally L1) terms plus a cost per additional leaf. The optimal leaf weight and split gain have closed forms in terms of summed g and h. Systems engineering—histogram/approximate split finding, sparsity-aware algorithms, cache-aware access, out-of-core computation—made boosting practical at scale. Column subsampling, shrinkage, and max depth control complexity. XGBoost remains a default strong baseline for structured EHR tables predicting outcomes such as 90-day mRS.

### LightGBM

LightGBM grows trees leaf-wise (best-first) rather than level-wise, often lowering loss faster for the same number of leaves, with depth or leaf caps to limit overfitting. It uses histogram-based binning of continuous features for fast split finding and Gradient-based One-Side Sampling (GOSS) to keep large-gradient examples while randomly sampling small-gradient ones, plus Exclusive Feature Bundling (EFB) to merge sparse mutually exclusive features. The result is high accuracy with reduced training time and memory on large feature sets—useful for wide claims or NLP-derived bag-of-codes matrices.

### CatBoost

CatBoost focuses on categorical features and ordered boosting to reduce prediction shift. Categorical variables are handled with efficient encodings that use target statistics computed in a permutation-aware way to limit leakage of the label into encodings. Ordered boosting trains on prefixes of a random permutation so that residual estimates are less biased than classical gradient boosting’s use of the same points for residual and model fit. Symmetric (oblivious) trees can improve CPU inference speed. For clinical registries rich in hospital codes, payer categories, and free-text-derived tokens, CatBoost often reduces manual encoding labor while remaining competitive with XGBoost and LightGBM.

### Complexity, tradeoffs, and choosing a boosted-tree library

Random forests train T trees independently, each costing roughly O(n log n · m) where m is the number of features examined per split (about √d with feature subsampling), so training parallelizes trivially across trees and cores; inference is O(T · depth). They are variance-reduction machines—robust and low-tuning, but larger models that are rarely the single best on a carefully tuned tabular benchmark. Boosting is sequential: each tree depends on its predecessors, so it parallelizes within a tree (split finding) rather than across trees. Training cost scales as the number of rounds times per-tree cost, and inference is again O(T · depth), though here T is often large and the trees shallow. The governing tradeoff is bias versus variance—bagging and forests attack variance with deep decorrelated trees, while boosting attacks bias with shallow trees and shrinkage, paying for it in sequential training and greater sensitivity to label noise and hyperparameters.

Among gradient-boosting libraries the differences are mostly engineering and defaults, not a change in the underlying additive-model mathematics. XGBoost’s regularized second-order objective with level-wise histogram growth is a robust, well-documented default. LightGBM’s leaf-wise growth with GOSS and EFB is typically fastest and most memory-efficient on wide, sparse, or large-n data, but leaf-wise growth overfits small datasets unless leaf count and minimum child weight are capped. CatBoost’s ordered target statistics and ordered boosting shine when many high-cardinality categorical features would otherwise force manual encoding or leak the label. A workable protocol: pick one library, tune learning rate and tree size with early stopping on a validation fold, and switch only when categorical handling (favor CatBoost) or training speed on large data (favor LightGBM) is the binding constraint. Once tuned, the three usually finish within a small margin of one another.

## 9.8 Model Selection

No classifier dominates all problems (the practical ‘no free lunch’ message). For high-dimensional sparse text, linear models and Naive Bayes are strong. For heterogeneous tabular data with interactions, trees, forests, and gradient boosting excel. For natural hierarchical sensory inputs (images, audio), deep networks often win given enough data. Start simple, measure carefully, and add complexity only when validation justifies it.

Model selection should use nested or carefully separated validation: outer loops estimate generalization; inner loops tune hyperparameters and thresholds. Stratified k-fold preserves class proportions; for multi-site data, group by hospital; for longitudinal records, group by patient; for temporal drift, use forward-chaining splits. Keep a final test set untouched until the pipeline is frozen. Compare against simple baselines (majority class, logistic regression) before claiming victory for complex ensembles. Computational budget, latency, and interpretability may forbid large ensembles in production even when they win on a leaderboard.

Match model family to data modality and sample size.

Tune thresholds on validation data using clinical costs, not only default 0.5.

![Cost-sensitive threshold: expected cost vs t when FN costs 5× FP (synthetic; original).](../assets/figures/ml_fig_cost_threshold.png)

*Figure — Cost-aware operating point. **Left:** overlapping score densities for synthetic positives/negatives with default t=0.5 vs cost-minimizing t. **Right:** expected cost per case under \(c_{FN}=5\), \(c_{FP}=1\) as a function of threshold—the minimum need not sit at 0.5 or at Youden’s J. Tune t on validation with clinical costs; report the chosen operating point with precision/recall at that point.*

Report uncertainty: multiple seeds, bootstrap CIs, or cross-fit estimates.

Document the full pipeline: imputation, encoding, scaling, model, threshold.

## 9.9 Evaluation Metrics in Depth

Accuracy is (TP + TN) / (TP + TN + FP + FN). It is misleading under class imbalance: predicting always the majority class can score high while missing the minority entirely. Precision = TP / (TP + FP) answers: of predicted positives, how many are true? Recall (sensitivity) = TP / (TP + FN) answers: of actual positives, how many did we catch? Specificity = TN / (TN + FP). F1 = 2 · precision · recall / (precision + recall) balances the two when both matter. Balanced accuracy averages sensitivity and specificity.

### Worked example: confusion-matrix metrics

Consider a binary test set of 200 cases with TP = 40, FP = 10, FN = 20, TN = 130. Accuracy = (40 + 130) / 200 = 0.85. Precision = 40 / (40 + 10) = 0.80. Recall = 40 / (40 + 20) ≈ 0.667. F1 = 2 · 0.80 · 0.667 / (0.80 + 0.667) ≈ 0.727. Specificity = 130 / 140 ≈ 0.929. False positive rate = 1 − specificity ≈ 0.071. If each FN costs five times each FP in a triage setting, one may lower the decision threshold to raise recall at the expense of precision. Metrics must match the decision problem, not a default 0.5 threshold.

![Threshold trade-off: overlapping scores and sensitivity/specificity vs t (synthetic; original).](../assets/figures/ml_fig_threshold_sens_spec.png)

*Figure — Threshold as a clinical lever. **Left:** synthetic score densities for true negatives and true positives; overlap is why no single cut is perfect. Vertical lines mark default *t* = 0.50 and the Youden peak *t*\* = argmax(Se+Sp−1). **Right:** as *t* rises, sensitivity falls and specificity rises; Youden *J* peaks at an intermediate operating point. Default 0.5 is not sacred—match the cost of false negatives (missed LVO) versus false positives (unnecessary angiography/transfer). Ranking metrics (AUC) summarize the curve; bedside use still needs a chosen *t*, calibrated risk, and prevalence-aware PPV.*

![9.5: Worked confusion-matrix metrics for a binary test set of n = 200 (TP = 40, FN = 20, FP = 10, TN = 130). Reading off the ](../assets/figures/ml_concept_9.5_600c2868.png)

*Figure 9.5 — original teaching graphic.*

ROC curves plot true positive rate versus false positive rate as the threshold varies. AUC-ROC summarizes ranking quality: probability that a random positive scores higher than a random negative. ROC can look optimistic under severe imbalance. Precision–recall (PR) curves plot precision versus recall and are often more informative when positives are rare; average precision summarizes the PR curve. Calibration asks whether predicted probabilities match empirical frequencies (among cases scored 0.7, about 70% should be positive). Reliability diagrams and expected calibration error (ECE) diagnose miscalibration; Platt scaling or isotonic regression can recalibrate on validation data without changing ranking much. The Brier score BS = mean((p − y)²) jointly penalizes miscalibration and weak discrimination; Murphy’s decomposition BS = REL − RES + UNC separates reliability, resolution, and base-rate uncertainty (see also Chapter 8).

![Brier score components (reliability / resolution / uncertainty) for calibrated vs overconfident models (original).](../assets/figures/ml_fig_brier_decomp.png)

*Figure — Brier beyond a single number. Overconfidence inflates REL; a near-constant score near prevalence can look “calibrated” with tiny RES. Pair Brier with reliability diagrams and PR/ROC; do not treat a low Brier alone as clinical utility or causation.*

![Platt vs isotonic recalibration: reliability diagrams and ECE (synthetic; original).](../assets/figures/ml_fig_platt_isotonic.png)

*Figure — Post-hoc calibration. Fit a calibrator on a held-out set (never the final test). **Left:** raw overconfident scores pulled toward the diagonal by Platt (logistic on logit-p) and isotonic regression. **Right:** ECE drops after recalibration. Isotonic is flexible but can overfit tiny calibration sets; Platt is parametric and smoother. Recalibration changes probability honesty, not ranking much—and not causation.*

![Precision–recall and ROC for the same synthetic imbalanced cohort (prevalence ≈ 10%; original).](../assets/figures/ml_fig_precision_recall.png)

*Figure — Precision–recall under imbalance. The PR panel (left) starts near prevalence for a chance classifier and rewards high PPV at useful recall; the ROC panel (right) for the same scores can look deceptively strong because true negatives dominate the FPR denominator. Prefer PR-AUC and operating-point precision when events are rare.*

Class imbalance remedies include resampling (oversample minority, undersample majority, or synthetic methods like SMOTE used carefully inside folds only), class-weighted losses, threshold tuning, and metrics aligned with the minority class (recall, PR-AUC, balanced accuracy). Multiclass reduction: one-vs-rest trains K binary classifiers; one-vs-one trains K(K−1)/2 pairwise classifiers; softmax models handle multiclass natively. Always nest preprocessing in cross-validation; leakage—scaling with test statistics, using future information, or tuning on the test set—is the most common source of unrealistically good reported accuracy.

![Focal loss: modulating factor (1−p_t)^γ down-weights easy examples (original).](../assets/figures/ml_fig_focal_loss.png)

*Figure — Focal loss \(FL=(1-p_t)^\gamma(-\log p_t)\). **Left:** loss vs true-class probability for increasing \(\gamma\). **Right:** the modulating factor collapses toward zero for easy, high-\(p_t\) cases so hard minorities dominate the gradient. Still tune operating thresholds, report PR metrics, and recalibrate if absolute risk is used—loss design is not causation.*

![Calibration slope and intercept after logistic recalibration (synthetic; original).](../assets/figures/ml_fig_calibration_slope.png)

*Figure — Slope/intercept as a miscalibration diagnosis. **Left:** reliability before vs after fitting \(a + b\,\mathrm{logit}(p)\) on a held-out calibration set. **Right:** ideal \(a\approx 0\), \(b\approx 1\); \(b<1\) flags overconfident raw scores. Fit only on calibration data—never the final test—and remember recalibration improves probability honesty, not causal interpretability.*

![ECE bin contributions on a reliability diagram (synthetic; original).](../assets/figures/ml_fig_ece_bins.png)

*Figure — ECE = Σ mass_b · |acc_b − conf_b|. **Left:** equal-width reliability bars. **Right:** per-bin contributions—large mass in a miscalibrated bin dominates the scalar. Inspect tails; one ECE number can hide clinically critical high-risk bins. ECE is not a causal metric.*

![PR curve with chance baseline at prevalence (synthetic; original).](../assets/figures/ml_fig_pr_chance.png)

*Figure — Always draw the prevalence line on a PR plot. A random-ish ranker sits near π; models must beat that floor. PR-AUC is more honest than ROC under rare events. Still ranking quality—not causation.*

![SMOTE-style synthetic minority oversampling caution (synthetic; original).](../assets/figures/ml_fig_smote_caution.png)

*Figure — Synthetic minorities fill geometric gaps. Apply only inside training folds; recalibrate probabilities after resampling; synthetic points are not new patients and do not create causal balance.*


![Prior shift: same Se/Sp curves yield different PPV by prevalence (original).](../assets/figures/ml_fig_prior_shift_ppv.png)

*Figure — Prevalence transport. Fixed sensitivity/specificity still move PPV as π changes. Thresholds do not travel free across clinics. PPV is predictive counseling—not a causal effect size.*


![Hinge vs logistic surrogate losses vs margin (original).](../assets/figures/ml_fig_hinge_logloss.png)

*Figure — Surrogate losses enable convex/smooth optimization for classification. They are not clinical utilities and not causal contrasts.*


![Boosting rounds: train loss falls while test eventually rises (synthetic; original).](../assets/figures/ml_fig_boosting_stages.png)

*Figure — Stagewise fitting helps then overfits. Early stopping on validation is mandatory. More trees are not more truth—and not causal effects.*


![Youden-like threshold utility sketch (synthetic; original).](../assets/figures/ml_fig_threshold_j.png)

*Figure — Threshold choice needs costs and prevalence, not only a peak J on a training set. Operating points are decisions—not causal effects.*


![k-NN error versus k (synthetic; original).](../assets/figures/ml_fig_knn_k_curve.png)

*Figure — Neighborhood size is a hyperparameter—not etiology. Pred != cause without design.*


![costmat teaching panel (original).](../assets/figures/ml_fig_cost_matrix.png)

*Figure — Teaching panel for costmat. Pred != cause without design.*


![Cycle-34 densify scientific panel 11 (original).](../assets/figures/ml_fig_c34_10.png)

*Figure — Continuous densify panel 11. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 11 (original).](../assets/figures/ml_fig_c35_10.png)

*Figure — Continuous densify panel 11. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 11 (original).](../assets/figures/ml_fig_c36_10.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 11 (original).](../assets/figures/ml_fig_c37_10.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 11 (original).](../assets/figures/ml_fig_c38_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 11 (original).](../assets/figures/ml_fig_c39_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 11 (original).](../assets/figures/ml_fig_c40_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 11 (original).](../assets/figures/ml_fig_c41_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 11 (original).](../assets/figures/ml_fig_c42_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 11 (original).](../assets/figures/ml_fig_c43_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 11 (original).](../assets/figures/ml_fig_c44_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 11 (original).](../assets/figures/ml_fig_c45_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 11 (original).](../assets/figures/ml_fig_c46_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 11 (original).](../assets/figures/ml_fig_c47_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 11 (original).](../assets/figures/ml_fig_c48_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 11 (original).](../assets/figures/ml_fig_c49_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 11 (original).](../assets/figures/ml_fig_c50_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 11 (original).](../assets/figures/ml_fig_c51_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 11 (original).](../assets/figures/ml_fig_c52_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 11 (original).](../assets/figures/ml_fig_c53_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 11 (original).](../assets/figures/ml_fig_c54_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 11 (original).](../assets/figures/ml_fig_c55_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 11 (original).](../assets/figures/ml_fig_c56_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 11 (original).](../assets/figures/ml_fig_c57_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 11 (original).](../assets/figures/ml_fig_c58_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 11 (original).](../assets/figures/ml_fig_c59_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 11 (original).](../assets/figures/ml_fig_c60_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 11 (original).](../assets/figures/ml_fig_c61_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 11 (original).](../assets/figures/ml_fig_c62_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 11 (original).](../assets/figures/ml_fig_c63_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 11 (original).](../assets/figures/ml_fig_c64_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 11 (original).](../assets/figures/ml_fig_c65_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 11 (original).](../assets/figures/ml_fig_c66_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 11 (original).](../assets/figures/ml_fig_c67_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 11 (original).](../assets/figures/ml_fig_c68_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 11 (original).](../assets/figures/ml_fig_c69_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 11 (original).](../assets/figures/ml_fig_c70_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 11 (original).](../assets/figures/ml_fig_c71_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 11 (original).](../assets/figures/ml_fig_c72_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 11 (original).](../assets/figures/ml_fig_c73_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 11 (original).](../assets/figures/ml_fig_c74_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 11 (original).](../assets/figures/ml_fig_c75_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 11 (original).](../assets/figures/ml_fig_c76_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 11 (original).](../assets/figures/ml_fig_c77_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 11 (original).](../assets/figures/ml_fig_c78_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 11 (original).](../assets/figures/ml_fig_c79_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 11 (original).](../assets/figures/ml_fig_c80_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 11 (original).](../assets/figures/ml_fig_c81_10.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

## Clinical and Epidemiologic Notes

Classification is the workhorse of predictive modeling in neurology: large-vessel occlusion (LVO) versus not, hemorrhagic versus ischemic pathways, TOAST or CCS stroke subtype, malignant MCA edema risk, 90-day functional independence (mRS 0–2), or detection of atrial fibrillation on extended monitoring. Scientific quality depends as much on cohort design and metrics as on the choice among logistic regression, forests, or boosting.

![9.6: ROC and precision–recall views of the same strong model on an imbalanced problem (prevalence 10%). The ROC sits far abov](../assets/figures/ml_concept_9.6_ba0d6fb7.png)

*Figure 9.6 — original teaching graphic.*

Labels rarely fall from the sky. LVO may be defined by CTA/MRA adjudication, by thrombectomy performance (which confounds access and decision-making), or by proxy severity scores. Subtype labels from administrative codes disagree with expert consensus; training on codes and testing on codes can overstate clinical utility. Prefer reference-standard labels on a validation subset even if training uses a scalable computable phenotype. When labels are uncertain, model that uncertainty or perform sensitivity analyses across phenotype definitions.

A classifier that ‘predicts’ LVO using features that include the thrombectomy procedure code or the final interventional radiology report is not a triage model; it is a restatement of care already given. Features must be available at the decision time (prehospital scales and noncontrast CT findings at ED arrival, for example). Scaling, imputation, and feature selection must be fit inside each training fold. Group by patient and by stroke episode so serial encounters do not leak across splits.

Many neurologic targets are rare in unselected ED traffic yet common in enriched stroke-code cohorts. Prevalence shifts change positive predictive value even when sensitivity and specificity are stable—an elementary Bayes fact. Report precision–recall curves and prevalence-aware metrics for rare events. Choose thresholds with stroke neurologists: missing LVO may delay transfer, while false positives consume angiography and transport. AUROC summarizes ranking, but bedside use needs an operating point and calibrated absolute risk. Temporal and geographic external validation detect drift from evolving pathways (tenecteplase adoption, expanded EVT windows) and transportability across primary and comprehensive centers.

A worked prevalence calculation fixes the intuition. Suppose an LVO screening tool holds sensitivity 0.90 and specificity 0.90 in every setting. In an enriched stroke-code cohort with LVO prevalence 0.30, the positive predictive value is PPV = (0.90 · 0.30) / (0.90 · 0.30 + 0.10 · 0.70) = 0.27 / 0.34 ≈ 0.79. Move the identical tool to an unselected ED where LVO prevalence is 0.03 and PPV = (0.90 · 0.03) / (0.90 · 0.03 + 0.10 · 0.97) = 0.027 / 0.124 ≈ 0.22. Sensitivity and specificity are unchanged, yet fewer than one positive in four is now a true LVO—the arithmetic behind why a tool validated on a stroke service disappoints at a general front door, and why PPV and NPV must always be reported at the prevalence of the intended deployment rather than that of the development sample.

Calibration needs special care for rare adverse events such as symptomatic intracranial hemorrhage (sICH) after thrombolysis, seen in only a few percent of treated patients. Under such imbalance a model can post an excellent AUROC while systematically over- or under-stating absolute risk, and most predicted probabilities pile up near zero where reliability is hardest to estimate. Two traps recur. First, class-weighting or resampling (oversampling, SMOTE, undersampling) applied so the learner sees enough positives also inflates the apparent event rate, so probabilities trained under resampling are miscalibrated for the true prevalence and must be recalibrated—or the intercept prior-shift-corrected—back to the deployment base rate. Second, global calibration can look adequate while the high-risk tail, precisely the region that drives treat-or-withhold decisions, is badly calibrated; inspect reliability in the top risk decile, not only overall. For sICH-type targets, prefer PR-AUC to ROC-AUC, report calibration-in-the-large and the calibration slope, and set the operating threshold against the explicit cost of a missed hemorrhage risk versus withheld reperfusion rather than a reflexive 0.5.

Define phenotype labels and decision-time features before model shopping.

Prefer PR analysis and calibrated risks when events are rare or decisions use absolute risk.

Nest preprocessing in cross-validation; split by patient/episode/site as appropriate.

Require external or temporal validation before clinical deployment claims.

Keep interpretable baselines; match metrics to stroke-system costs of FP versus FN.


![c82 teaching panel 10 (original).](../assets/figures/ml_fig_c82_10.png)
*Figure — Decision threshold shifts when FN costs more than FP (synthetic costs). Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 10 (original).](../assets/figures/ml_fig_c83_10.png)
*Figure — Reliability diagram: overconfident vs calibrated probabilities. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 10 (original).](../assets/figures/ml_fig_c84_10.png)
*Figure — Precision–recall curves shift with prevalence. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 10 (original).](../assets/figures/ml_fig_c85_10.png)
*Figure — ROC curves compare ranking quality against chance. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 10 (original).](../assets/figures/ml_fig_c86_10.png)
*Figure — Sensitivity vs specificity as threshold moves. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 10 (original).](../assets/figures/ml_fig_c87_10.png)
*Figure — Confusion matrix counts at one operating threshold. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 10 (original).](../assets/figures/ml_fig_c88_10.png)
*Figure — One-vs-rest multilabel reduction sketch. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 10 (original).](../assets/figures/ml_fig_c89_10.png)
*Figure — Asymmetric misclassification cost matrix. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 10 (original).](../assets/figures/ml_fig_c90_10.png)
*Figure — Platt scaling sigmoid on scores. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 10 (original).](../assets/figures/ml_fig_c91_10.png)
*Figure — Margin geometry for linear separator. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 10 (original).](../assets/figures/ml_fig_c92_10.png)
*Figure — Isotonic calibration stepwise. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 10 (original).](../assets/figures/ml_fig_c93_10.png)
*Figure — Expected calibration error bins. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 10 (original).](../assets/figures/ml_fig_c94_10.png)
*Figure — Softmax multi-class simplex. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 10 (original).](../assets/figures/ml_fig_c95_10.png)
*Figure — Brier score calibration square. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 10 (original).](../assets/figures/ml_fig_c96_10.png)
*Figure — Proper scoring rule ranking. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 10 (original).](../assets/figures/ml_fig_c97_10.png)
*Figure — Decision stump axis split. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 10 (original).](../assets/figures/ml_fig_c98_10.png)
*Figure — Sharpness of probability peaks. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 10 (original).](../assets/figures/ml_fig_c99_10.png)
*Figure — Spiegelhalter calibration z. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 10 (original).](../assets/figures/ml_fig_c100_10.png)
*Figure — Macro vs micro F1. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 10 (original).](../assets/figures/ml_fig_c101_10.png)
*Figure — Class-wise calibration curves. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 10 (original).](../assets/figures/ml_fig_c102_10.png)
*Figure — Decision curve clinical net benefit. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 10 (original).](../assets/figures/ml_fig_c103_10.png)
*Figure — Error-correcting output codes. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 10 (original).](../assets/figures/ml_fig_c104_10.png)
*Figure — Threshold moving PR operating. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 10 (original).](../assets/figures/ml_fig_c105_10.png)
*Figure — Net reclassification index. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 10 (original).](../assets/figures/ml_fig_c106_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 10 (original).](../assets/figures/ml_fig_c107_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 10 (original).](../assets/figures/ml_fig_c108_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 10 (original).](../assets/figures/ml_fig_c109_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 10 (original).](../assets/figures/ml_fig_c110_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 10 (original).](../assets/figures/ml_fig_c111_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 10 (original).](../assets/figures/ml_fig_c112_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 10 (original).](../assets/figures/ml_fig_c113_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 10 (original).](../assets/figures/ml_fig_c114_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 10 (original).](../assets/figures/ml_fig_c115_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 10 (original).](../assets/figures/ml_fig_c116_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 10 (original).](../assets/figures/ml_fig_c117_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 10 (original).](../assets/figures/ml_fig_c118_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 10 (original).](../assets/figures/ml_fig_c119_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 10 (original).](../assets/figures/ml_fig_c120_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 10 (original).](../assets/figures/ml_fig_c121_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 10 (original).](../assets/figures/ml_fig_c122_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 10 (original).](../assets/figures/ml_fig_c123_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 10 (original).](../assets/figures/ml_fig_c124_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 10 (original).](../assets/figures/ml_fig_c125_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 10 (original).](../assets/figures/ml_fig_c126_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 10 (original).](../assets/figures/ml_fig_c127_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 10 (original).](../assets/figures/ml_fig_c128_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 10 (original).](../assets/figures/ml_fig_c129_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 10 (original).](../assets/figures/ml_fig_c130_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 10 (original).](../assets/figures/ml_fig_c131_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 10 (original).](../assets/figures/ml_fig_c132_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 10 (original).](../assets/figures/ml_fig_c133_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 10 (original).](../assets/figures/ml_fig_c134_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 10 (original).](../assets/figures/ml_fig_c135_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 10 (original).](../assets/figures/ml_fig_c136_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 10 (original).](../assets/figures/ml_fig_c137_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 10 (original).](../assets/figures/ml_fig_c138_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 10 (original).](../assets/figures/ml_fig_c139_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 10 (original).](../assets/figures/ml_fig_c140_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 10 (original).](../assets/figures/ml_fig_c141_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 10 (original).](../assets/figures/ml_fig_c142_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 10 (original).](../assets/figures/ml_fig_c143_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 10 (original).](../assets/figures/ml_fig_c144_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 10 (original).](../assets/figures/ml_fig_c145_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 10 (original).](../assets/figures/ml_fig_c146_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 10 (original).](../assets/figures/ml_fig_c147_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 10 (original).](../assets/figures/ml_fig_c148_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 10 (original).](../assets/figures/ml_fig_c149_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 10 (original).](../assets/figures/ml_fig_c150_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 10 (original).](../assets/figures/ml_fig_c151_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 10 (original).](../assets/figures/ml_fig_c152_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 10 (original).](../assets/figures/ml_fig_c153_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 10 (original).](../assets/figures/ml_fig_c154_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 10 (original).](../assets/figures/ml_fig_c155_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 10 (original).](../assets/figures/ml_fig_c156_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 10 (original).](../assets/figures/ml_fig_c157_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 10 (original).](../assets/figures/ml_fig_c158_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 10 (original).](../assets/figures/ml_fig_c159_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 10 (original).](../assets/figures/ml_fig_c160_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 10 (original).](../assets/figures/ml_fig_c161_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 10 (original).](../assets/figures/ml_fig_c162_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 10 (original).](../assets/figures/ml_fig_c163_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 10 (original).](../assets/figures/ml_fig_c164_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 10 (original).](../assets/figures/ml_fig_c165_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 10 (original).](../assets/figures/ml_fig_c166_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 10 (original).](../assets/figures/ml_fig_c167_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 10 (original).](../assets/figures/ml_fig_c168_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 10 (original).](../assets/figures/ml_fig_c169_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 10 (original).](../assets/figures/ml_fig_c170_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 10 (original).](../assets/figures/ml_fig_c171_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 10 (original).](../assets/figures/ml_fig_c172_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 10 (original).](../assets/figures/ml_fig_c173_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 10 (original).](../assets/figures/ml_fig_c174_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 10 (original).](../assets/figures/ml_fig_c175_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 10 (original).](../assets/figures/ml_fig_c176_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 10 (original).](../assets/figures/ml_fig_c177_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 10 (original).](../assets/figures/ml_fig_c178_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 10 (original).](../assets/figures/ml_fig_c179_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 10 (original).](../assets/figures/ml_fig_c180_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 10 (original).](../assets/figures/ml_fig_c181_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 10 (original).](../assets/figures/ml_fig_c182_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 10 (original).](../assets/figures/ml_fig_c183_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 10 (original).](../assets/figures/ml_fig_c184_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 10 (original).](../assets/figures/ml_fig_c185_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 10 (original).](../assets/figures/ml_fig_c186_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 10 (original).](../assets/figures/ml_fig_c187_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 10 (original).](../assets/figures/ml_fig_c188_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 10 (original).](../assets/figures/ml_fig_c189_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 10 (original).](../assets/figures/ml_fig_c190_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 10 (original).](../assets/figures/ml_fig_c191_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 10 (original).](../assets/figures/ml_fig_c192_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 10 (original).](../assets/figures/ml_fig_c193_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 10 (original).](../assets/figures/ml_fig_c194_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 10 (original).](../assets/figures/ml_fig_c195_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 10 (original).](../assets/figures/ml_fig_c196_10.png)
*Figure — Ordinal regression thresholds. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 10 (original).](../assets/figures/ml_fig_c197_10.png)
*Figure — Multilabel classifier chains. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 10 (original).](../assets/figures/ml_fig_c198_10.png)
*Figure — Cost-sensitive weighting. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 10 (original).](../assets/figures/ml_fig_c199_10.png)
*Figure — Label powerset reduce. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 10 (original).](../assets/figures/ml_fig_c200_10.png)
*Figure — Platt vs isotonic. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 10 (original).](../assets/figures/ml_fig_c201_10.png)
*Figure — PR curves shift with prevalence. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 10 (original).](../assets/figures/ml_fig_c202_10.png)
*Figure — Cost curves for class imbalance. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 10 (original).](../assets/figures/ml_fig_c203_10.png)
*Figure — Brier skill vs climatology. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 10 (original).](../assets/figures/ml_fig_c204_10.png)
*Figure — Threshold expected utility curve. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 10 (original).](../assets/figures/ml_fig_c205_10.png)
*Figure — MCC vs accuracy prevalence. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 10 (original).](../assets/figures/ml_fig_c206_10.png)
*Figure — Cohen kappa chance correction. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 10 (original).](../assets/figures/ml_fig_c207_10.png)
*Figure — Youden J ROC threshold. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 10 (original).](../assets/figures/ml_fig_c208_10.png)
*Figure — Cost-sensitive threshold shift. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 10 (original).](../assets/figures/ml_fig_c209_10.png)
*Figure — Precision-recall gain over baseline. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 10 (original).](../assets/figures/ml_fig_c210_10.png)
*Figure — Balanced accuracy vs prevalence. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 10 (original).](../assets/figures/ml_fig_c211_10.png)
*Figure — DET FPR-FNR tradeoff curve. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 10 (original).](../assets/figures/ml_fig_c212_10.png)
*Figure — Accuracy versus kappa agreement. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 10 (original).](../assets/figures/ml_fig_c213_10.png)
*Figure — Reliability diagram bins. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 10 (original).](../assets/figures/ml_fig_c214_10.png)
*Figure — NPV versus prevalence curve. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Classification maps features to discrete labels using scores or posteriors and a decision rule. Rule-based systems encode transparent logic; Naive Bayes multiplies class-conditional feature likelihoods under a conditional independence assumption, with Gaussian NB for continuous features and worked toy-table posterior and log-score arithmetic; k-NN assigns labels by neighborhood vote with Voronoi geometry and KD-tree or LSH acceleration; SVMs maximize soft margins and extend via kernels, multilabel reductions, and dual prediction over support vectors. Decision trees grow by impurity or statistical criteria in ID3, CHAID, C4.5, and CART lineages, then prune to control variance. Ensembles—bagging, boosting, stacking, random forests, AdaBoost, GBDT, XGBoost, LightGBM, and CatBoost—aggregate unstable or weak learners into strong predictors for tabular clinical data. Model selection needs nested validation and honest baselines. Evaluation must go beyond accuracy to precision, recall, F1, ROC/PR analysis, calibration, and imbalance-aware design. In neurology and epidemiology, labels are phenotypes, features must respect index time, prevalence shifts alter predictive values, and external validation plus calibration determine transportability.

## Practice and Reflection

(1) From TP=50, FP=30, FN=10, TN=110, compute accuracy, precision, recall, F1, and specificity. If each FN costs 5 units and each FP costs 1 unit, which threshold direction (more or less aggressive positive calling) is suggested qualitatively?

(2) Implement (by hand or code) 3-NN with Euclidean distance on a 2D toy set of 12 labeled points and classify three query points. How does standardization change neighbors?

(3) Complete the Gaussian Naive Bayes worked example with priors p(A)=0.1, p(B)=0.9 for query (0,1) and report which class wins.

(4) For a decision stump on a binary feature that splits 100 samples into groups of sizes 60 and 40 with given class counts, compute information gain versus Gini gain and compare.

(5) Sketch why gain ratio in C4.5 reduces the preference for high-arity categorical features relative to raw information gain in ID3.

(6) Design a stratified 5-fold CV protocol for an imbalanced medical dataset (5% positive). Where would SMOTE be applied relative to the fold splits, and why?

(7) Explain when you would prefer PR-AUC over ROC-AUC and when calibration matters more than ranking metrics.

(8) Propose features available at stroke-code activation for binary LVO classification. List three variables that would constitute leakage if included from the same encounter.

(9) Compare computational complexity of predicting with a linear SVM versus an RBF SVM with n_sv support vectors, and versus a random forest with T trees of depth d.

(10) A model predicts 90-day mRS 0–2 with AUROC 0.79 at a comprehensive center but PPV collapses at a spoke ED where good outcomes are less frequent. Explain using prevalence and suggest what to report besides AUROC.

(11) Using the categorical Naive Bayes toy table (priors 0.4/0.6; P(Gaze=yes | L+) = 0.8, P(Arm=yes | L+) = 0.7, P(Gaze=yes | L−) = 0.2, P(Arm=yes | L−) = 0.2), compute the posterior P(L+ | x) for a patient with Gaze=yes but Arm=no. Interpret the result, then redo the calculation with add-one smoothing assuming the L+ arm-weakness cell had been 0/10.

(12) A classifier has sensitivity 0.85 and specificity 0.92. Compute PPV at prevalence 0.40 and at prevalence 0.05. Explain to a colleague why the same model can be ‘trustworthy when positive’ on the stroke service yet not at a general front door.
