# Selected Glossary

## Opening
![Train / validation / test vocabulary (original).](../assets/figures/ml_fig_split_vocab.png)

*Train / validation / test vocabulary (original).*

![Discrimination, calibration, and utility (original).](../assets/figures/ml_fig_eval_triangle.png)

*Discrimination, calibration, and utility (original).*


![Metric families: discrimination, calibration, utility (original).](../assets/figures/ml_fig_metric_map.png)

*Teaching map: AUC ranks; calibration checks probability honesty; utility asks whether acting on a threshold helps (original).*

![Probability vs odds and Bayes LR update on odds (original).](../assets/figures/ml_fig_odds_vs_prob.png)

*Figure — Glossary: odds = p/(1−p). **Left:** odds rise nonlinearly as probability approaches 1. **Right:** multiply prior odds by LR+, then convert back to posterior probability—do not add “probability points” as if they were odds. Model scores still require calibration for absolute risk; prediction ≠ causation.*

![ARR and NNT from absolute risks (original).](../assets/figures/ml_fig_arr_nnt.png)

*Figure — Glossary utility vocab. **Left:** absolute risk reduction ARR = p₀ − p₁. **Right:** NNT = 1/ARR. Relative metrics (OR/RR) alone cannot yield NNT; calibrated absolute risks are required. Model AUROC does not equal ARR, and ARR from observational scores is not automatically a causal treatment effect.*

![Youden J = Se+Sp−1 vs threshold (original).](../assets/figures/ml_fig_youden_j.png)

*Figure — Glossary: Youden’s J peaks where Se+Sp−1 is maximized. It ignores prevalence and misclassification costs—so the clinical operating point may differ. Threshold choice is a decision, not a causal estimate.*

![Appraisal orientation graphic (original).](../assets/figures/ml_fig_appraisal_scorecard.png)

*Teaching orientation for metrics and appraisal terms (original).*

![Glossary term families — teaching taxonomy (original).](../assets/figures/ml_fig_glossary_families.png)

*Six term families: paradigms, mechanics, evaluation, data/time, causal caution, deploy/govern (original).*

![Glossary visual: confusion cells map to Se/Sp/PPV/NPV and prevalence dependence (original).](../assets/figures/ml_fig_se_sp_ppv_vocab.png)

*Figure — Shared lexicon for triage metrics. **Left:** TP/FP/FN/TN cells define sensitivity and specificity at a threshold. **Right:** with Se/Sp fixed, PPV and NPV still move with prevalence π—never copy a paper’s PPV into a different base-rate clinic without recompute.*

![PPV versus prevalence at fixed sensitivity and specificity (scientific; original).](../assets/figures/ml_fig_ppv_prevalence.png)

*PPV of a fixed screen (sens 0.85, spec 0.70) collapses as prevalence falls — LR+ travels; PPV does not (original).*

![The accuracy trap under class imbalance (scientific; original).](../assets/figures/ml_fig_accuracy_trap.png)

*Accuracy can look excellent while sensitivity collapses under low prevalence — report sens/spec/PPV at the decision threshold (original).*

![Reliability diagram and ECE for calibrated vs miscalibrated scores (synthetic; original).](../assets/figures/ml_fig_reliability_ece.png)

*Figure — Reliability diagram on one synthetic low-prevalence risk cohort. Points on the diagonal are honest probabilities; overconfident models bow above/below the line and inflate expected calibration error (ECE = Σ (n_b/n)·|obs_b − conf_b|). Score histograms (right) show overconfidence piling mass at the extremes. Calibration is a distinct claim from AUC.*

![Prediction ≠ causation: confounder sketch and claim boxes (synthetic; original).](../assets/figures/ml_fig_pred_not_cause.png)

*Figure — Glossary anchor for causal caution. A high-ranking feature *X* can predict outcome *Y* through shared dependence on confounder *U* without *X* causing *Y*. Keep prediction, etiology, and decision-support claims in separate boxes when reading papers or appraising tools.*


Journal club language collapses when people use ‘AI,’ ‘algorithm,’ and ‘model’ interchangeably. The glossary is a shared lexicon for stroke services that want precise disagreement rather than vague awe.

Activation function. A nonlinear transform applied to a neuron’s weighted input; without it a stack of layers would collapse into a single linear map.

Apriori algorithm. A method for mining frequent itemsets that prunes the search using the property that any superset of an infrequent set is also infrequent.

Association rule. An “if X then Y” pattern mined from co-occurrence data, judged by support (how often X and Y appear together) and confidence (how often Y follows X).

Attention. A mechanism that lets a model weight the relevance of other elements when representing a given element, so context is aggregated by learned importance rather than fixed position.

AUC. The area under the ROC curve; equivalently, the probability that a random positive case is scored above a random negative one. It summarizes discrimination and is indifferent to calibration.

Autoencoder. A neural network trained to reconstruct its input through a narrow bottleneck, learning a compressed representation useful for denoising or dimensionality reduction.

Backpropagation. The reverse-mode application of the chain rule that computes a loss’s gradient with respect to every network parameter, enabling gradient-based training.

Batch normalization. A layer that standardizes activations across a mini-batch, stabilizing and accelerating training and adding mild regularization.

Bayes’ theorem. The identity P(A|B) = P(B|A)·P(A)/P(B), which updates a prior belief into a posterior after observing evidence.

Bellman equation. The recursive relation expressing the value of a state as the immediate reward plus the discounted value of successor states; the basis of most reinforcement-learning updates.

BERT. A transformer encoder pretrained by predicting masked tokens, producing bidirectional contextual representations that are fine-tuned for downstream language tasks.

Bias–variance tradeoff. The decomposition of expected error into bias (systematic error from an oversimplified model), variance (sensitivity to the particular training sample), and irreducible noise; reducing one term often inflates the other.

Bootstrap. Resampling a dataset with replacement to approximate the sampling distribution of a statistic and obtain empirical confidence intervals.

Calibration. Agreement between predicted probabilities and observed frequencies; a model is calibrated if, among cases assigned probability p, about a fraction p actually experience the event.

Class imbalance. A large disparity in class frequencies that can make accuracy misleading and bias a classifier toward the majority class unless addressed by resampling, reweighting, or threshold choice.

CNN. A convolutional neural network that shares small learnable filters across spatial positions, giving translation-equivariant feature extraction well suited to images and signals.

Concept drift. A change over time in the relationship between features and outcome, or in their distributions, that degrades a deployed model until it is recalibrated or retrained.

Confounding. Distortion of an exposure–outcome association by a variable that influences both; unadjusted, it makes a non-causal association look causal.

Confusion matrix. A table cross-tabulating predicted against actual classes, from which sensitivity, specificity, precision, and related metrics are computed.

Contrastive learning. A self-supervised objective that pulls representations of related (augmented) samples together while pushing unrelated samples apart.

Convolution. The sliding weighted sum of a filter over an input that produces a feature map; the core operation of CNNs.

Cross-validation. Repeatedly partitioning data into training and held-out folds to estimate out-of-sample performance and reduce dependence on a single split.

Curse of dimensionality. The tendency of data to become sparse and distances to lose contrast as the number of features grows, undermining density estimation and nearest-neighbor methods.

DAG. A directed acyclic graph encoding assumed causal relationships; used to identify confounders, mediators, and colliders and to choose a valid adjustment set.

DBSCAN. A density-based clustering algorithm that groups closely packed points and labels sparse points as noise, discovering arbitrarily shaped clusters without a preset cluster count.

Decision tree. A model that recursively splits the feature space on threshold rules, yielding interpretable but high-variance piecewise-constant predictions.

Diffusion model. A generative model that learns to reverse a gradual noising process, synthesizing samples by denoising from random noise in successive steps.

Dropout. A regularizer that randomly zeroes a fraction of units during training, discouraging co-adaptation and approximating an ensemble of subnetworks.

Elastic net. A regularized regression combining L1 and L2 penalties, giving LASSO-style sparsity while handling correlated predictors more gracefully.

EM algorithm. An iterative maximum-likelihood procedure for latent-variable models that alternates estimating the latent distribution (E-step) with updating parameters (M-step); used for mixtures and HMMs.

Embedding. A learned dense vector representation that places discrete items or tokens in a continuous space where geometry reflects similarity.

Ensemble. A combination of multiple models (by averaging, voting, or stacking) that typically reduces variance and improves generalization over any single member.

Entropy. The expected uncertainty of a distribution, −Σ p·log p; it is maximal for uniform distributions and zero for deterministic ones.

Epoch. One full pass of the training algorithm over the entire training dataset.

External validation. Evaluation on data from a different site, time, or population than development, testing transportability rather than mere reproducibility.

F1 score. The harmonic mean of precision and recall, summarizing both in a single value that penalizes large imbalances between them.

Feature leakage. The inclusion of information not available at index time, or a proxy of the outcome itself, which inflates apparent performance and collapses in real deployment.

FID. Fréchet inception distance; compares real and generated image distributions by fitting Gaussians to deep features and measuring their Fréchet distance, with lower values indicating greater realism.

Fine-tuning. Continuing training of a pretrained model on a smaller task-specific dataset so its general representations adapt to the new objective.

GAN. A generative adversarial network in which a generator and a discriminator train against each other until the generator’s samples become hard to distinguish from real data.

Gaussian mixture model. A probabilistic clustering model representing data as a weighted sum of Gaussian components, typically fit by the EM algorithm.

GNN. A graph neural network that computes node representations by iteratively aggregating features from neighboring nodes along the graph’s edges.

Gradient descent. An optimization method that iteratively steps parameters in the direction opposite the loss gradient, scaled by a learning rate.

Hierarchical clustering. A family of methods that build a nested tree (dendrogram) of clusters by successively merging or splitting groups, without committing to a cluster count in advance.

HMM. A hidden Markov model in which observations are generated by an unobserved Markov chain of states; inference recovers state probabilities or the most likely state path.

Hyperparameter. A configuration value set before training (such as learning rate or tree depth) rather than learned from data, usually tuned on validation performance.

Imputation. Filling in missing values with estimates so that incomplete records can be used; careless imputation can leak information or bias results if missingness is informative.

Index time. The moment at which a prediction is made; only information available at or before it is a legal feature for that prediction.

Inductive bias. The set of assumptions a learning method uses to generalize beyond the training data, such as smoothness, locality, or sparsity.

K-means. A clustering algorithm that partitions points into k groups by alternately assigning each point to its nearest centroid and recomputing centroids to minimize within-cluster variance.

Kernel trick. Computing inner products in a high-dimensional feature space implicitly through a kernel function, enabling nonlinear decision boundaries without explicit mapping.

KL divergence. An asymmetric measure, Σ p·log(p/q), of how one distribution diverges from another; it is zero only when the two match and is not a true distance.

Knowledge distillation. Training a small “student” model to mimic the outputs of a larger “teacher,” transferring much of its performance at lower cost.

Label noise. Errors or inconsistencies in the outcome labels — from coding, chart review, or rater disagreement — that cap achievable performance and can bias learned patterns.

LASSO. Linear regression with an L1 penalty that shrinks coefficients and drives some exactly to zero, performing simultaneous fitting and feature selection.

Learning rate. The step-size multiplier on the gradient in gradient-based optimization; too large diverges, too small crawls or stalls.

Logistic regression. A model for binary outcomes that maps a linear combination of features through the logistic function to a calibrated probability.

LoRA. Low-rank adaptation; an efficient fine-tuning method that trains small low-rank weight updates while freezing the original model parameters.

Loss function. The objective quantifying the penalty for a prediction’s error, whose minimization defines training and whose choice encodes what “good” means.

LSTM. A long short-term memory recurrent network that uses gated memory cells to retain information over long sequences and mitigate vanishing gradients.

Markov decision process. The formal framework for sequential decision-making, specified by states, actions, transition probabilities, and rewards, in which the future depends only on the current state.

MCMC. Markov chain Monte Carlo; a family of algorithms that draw samples from a target distribution (often a Bayesian posterior) by simulating a Markov chain whose stationary distribution is that target.

MLE. Maximum likelihood estimation; choosing parameters that maximize the probability of the observed data under the model.

Naive Bayes. A probabilistic classifier applying Bayes’ theorem under the simplifying assumption that features are conditionally independent given the class.

Normalization. Rescaling features to a common range or unit scale so that no variable dominates by virtue of its measurement units and optimization behaves stably.

One-hot encoding. Representing a categorical variable as a set of binary indicator columns, one per category, avoiding false ordinal structure.

Overfitting. Fitting noise and idiosyncrasies of the training data so that performance on new data deteriorates; the hallmark of excess model capacity or insufficient regularization.

PageRank. A node-importance score equal to the stationary distribution of a random walk with teleportation, so that a node is important if important nodes link to it.

PCA. Principal component analysis; an orthogonal linear transform that projects data onto directions of greatest variance for dimensionality reduction and decorrelation.

PPO. Proximal policy optimization; a policy-gradient reinforcement-learning method that improves the policy while clipping updates to keep the new policy close to the old for stability.

PPV. Positive predictive value; the probability that a case flagged positive truly has the condition, which depends strongly on prevalence as well as on sensitivity and specificity.

Precision. Among cases predicted positive, the fraction that are truly positive; equivalently, positive predictive value.

Pruning. Removing redundant weights, neurons, or subtrees from a trained model to reduce size and computation with little loss of accuracy.

Q-learning. A value-based reinforcement-learning algorithm that learns the expected return of state–action pairs and derives a policy by acting greedily with respect to it.

Quantization. Reducing the numeric precision of weights and activations (for example from 32-bit float to 8-bit integer) to shrink and accelerate a model with minimal accuracy loss.

RAG. Retrieval-augmented generation; conditioning a language model on documents fetched at query time so that outputs are grounded in an external, updatable corpus.

Random forest. An ensemble of decision trees trained on bootstrapped samples and random feature subsets, whose averaged predictions reduce the variance of individual trees.

Recall. Among truly positive cases, the fraction the model correctly identifies; equivalently, sensitivity.

Regularization. Any technique — penalties, dropout, early stopping — that constrains model complexity to reduce overfitting and improve generalization.

Reinforcement learning. Learning a policy that maximizes cumulative reward through trial-and-error interaction with an environment, rather than from labeled examples.

ReLU. The rectified linear unit, f(x) = max(0, x); a simple nonlinearity that trains efficiently and mitigates vanishing gradients.

Ridge regression. Linear regression with an L2 penalty that shrinks coefficients toward zero, stabilizing estimates under multicollinearity without eliminating variables.

ROC curve. The plot of true-positive rate against false-positive rate across all thresholds, tracing a classifier’s discrimination independent of any single cutoff.

Selection bias. Distortion arising when the sample analyzed is not representative of the target population, so associations in the data do not hold in the population.

Self-attention. Attention applied within a single sequence, letting every element attend to every other to build context-aware representations; the core of the transformer.

Self-supervised learning. Learning representations from unlabeled data by constructing surrogate tasks from the data’s own structure, such as predicting masked or future parts.

SHAP. An attribution method that assigns each feature a Shapley-value contribution to a specific prediction by averaging its marginal effect over feature orderings.

Silhouette. A cluster-quality score in [−1, 1] comparing each point’s average distance to its own cluster with its distance to the nearest other cluster; higher indicates better separation.

Softmax. A function that turns a vector of real scores into a probability distribution proportional to their exponentials, used for multiclass outputs.

Supervised learning. Learning a mapping from inputs to known outputs using labeled examples, as in regression and classification.

SVM. A support vector machine; a classifier that seeks the maximum-margin separating boundary, extendable to nonlinear boundaries through kernels.

TF–IDF. Term frequency–inverse document frequency; a weighting that emphasizes terms frequent in a document but rare across the corpus, downweighting ubiquitous words.

Transfer learning. Reusing representations learned on one task or dataset to improve learning on a related task with limited data.

Transformer. A sequence architecture built on stacked self-attention and feed-forward layers that processes elements in parallel and captures long-range dependencies.

t-SNE. A nonlinear embedding for visualization that preserves local neighborhoods by matching pairwise-similarity distributions across dimensions; global distances and cluster sizes should not be over-interpreted.

UMAP. A manifold-learning embedding for visualization and dimensionality reduction, often faster than t-SNE and better at retaining some global structure, though still not to be read as a metric map.

Unsupervised learning. Discovering structure — clusters, components, densities — in data without labeled outcomes.

VAE. A variational autoencoder; a generative latent-variable model trained to reconstruct inputs through a probabilistic bottleneck by maximizing an evidence lower bound.

Viterbi algorithm. A dynamic-programming procedure that finds the single most probable sequence of hidden states in an HMM given the observations.


## Quick metric map (teaching table)

![Metric family decision tree: claim → metric family (original).](../assets/figures/ml_fig_metric_decision_tree.png)

*Figure — Glossary metric tree. Branch first on the **claim** (ranking/screening, probability counseling, or treat-vs-not), then pick the metric family (AUROC/AUPRC, calibration/Brier/ECE, net benefit / cost-sensitive utility). Accuracy alone is almost never enough in imbalanced stroke settings. High AUROC ≠ calibrated ≠ useful; prediction ≠ causation.*

| Metric | Answers | Common misuse |
|--------|---------|----------------|
| Accuracy | Overall correct rate | Collapses under class imbalance |
| Sensitivity / recall | Catch true positives | Ignores false alarms |
| Specificity | Correct negatives | Ignores missed cases |
| PPV / precision | Of predicted positives, how many true? | Drifts with prevalence |
| AUC (ROC) | Ranking discrimination | Not calibration or utility |
| Brier / calibration plot | Probability reliability | Not the same as AUC |
| Net benefit | Decision value at a threshold | Requires a meaningful threshold range |

Synthetic reminder: identical sensitivity/specificity can yield very different PPV in a rare-disease clinic versus a case-enriched research sample.

## Prevalence → PPV quick reference (teaching table)

Fixed screen: **sensitivity = 0.85**, **specificity = 0.70** (LR+ ≈ 2.83). Formula: \(\mathrm{PPV} = \frac{\mathrm{sens}\cdot\pi}{\mathrm{sens}\cdot\pi + (1-\mathrm{spec})(1-\pi)}\).

| Prevalence π | P(+) | PPV | Clinical reading |
|--------------|------|-----|------------------|
| 0.05 | ≈0.328 | ≈0.13 | Most positives are false alarms in a rare-disease clinic |
| 0.10 | ≈0.355 | ≈0.24 | Still more false than true positives |
| 0.20 | 0.41 | ≈0.41 | Chapter LVO ED example — modest PPV despite high sensitivity |
| 0.40 | ≈0.52 | ≈0.65 | Case-enriched cohort flatters precision |
| 0.60 | ≈0.63 | ≈0.81 | High-risk selected population; not transportable to screening |

Use likelihood ratios and calibrated probabilities when base rates change; never copy a paper’s PPV into a different prevalence without recomputing.



## Leakage (short definition)

**Leakage** is using information that would not be available at prediction time (or that is a proxy for the outcome created after the fact). It inflates apparent performance and fails in real deployment.

![Leakage taxonomy: temporal, fit/CV, label-proxy, target-encoding (original).](../assets/figures/ml_fig_leakage_taxonomy.png)\n![Bayes update with fixed LR+: prior vs posterior across prevalences (original).](../assets/figures/ml_fig_lr_nomogram.png)

*Figure — Glossary Bayes reminder. Fixed LR+=5 lifts probability differently at π=0.05 vs 0.40. Prevalence is not optional. Likelihood ratios quantify test strength for prediction counseling—they are not causal treatment effects.*\n\n


![Metric cheat sheet matching claim families to metrics (original).](../assets/figures/ml_fig_metric_cheatsheet.png)

*Figure — Glossary quick map. AUROC/AUPRC rank; Brier/ECE check probabilities; net benefit needs thresholds; C-index handles censoring. Wrong metric → wrong claim. None alone prove causation.*


![Calibration slope and intercept on a reliability sketch (synthetic; original).](../assets/figures/ml_fig_calib_slope_int.png)

*Figure — Glossary calibration. Slope≠1 or nonzero intercept means probabilities need recalibration before counseling. Calibration is about forecast honesty—not causal effect sizes.*


![Confusion matrix vocabulary with toy TP/FP/FN/TN counts (original).](../assets/figures/ml_fig_confusion_vocab.png)

*Figure — Glossary 2×2. Sensitivity and specificity come from rows; PPV/NPV need prevalence. Counts are teaching toys—not a trial result. Metrics support prediction claims, not causation by default.*


![Likelihood-ratio spectrum on a log axis (original).](../assets/figures/ml_fig_lr_spectrum.png)

*Figure — Glossary LR line. LR>1 increases odds; LR<1 decreases; LR=1 is uninformative. Likelihood ratios update predictive odds—they are not treatment causal effects.*


![ROC vs PR reminder with prevalence line (synthetic; original).](../assets/figures/ml_fig_roc_pr_reminder.png)

*Figure — Pick metric family by claim; neither proves causation. Pred != cause without design.*


![PPV/NPV vs prevalence grid (original).](../assets/figures/ml_fig_npv_ppv_grid.png)

*Figure — Base rates drive predictive values. PPV/NPV vs prevalence grid Pred != cause without design.*


![Logit map log-odds vs probability (original).](../assets/figures/ml_fig_logit_map.png)

*Figure — Glossary logit transform. Odds language supports Bayes updates for prediction—not causal treatment effects by default.*


![Cycle-34 densify scientific panel 20 (original).](../assets/figures/ml_fig_c34_19.png)

*Figure — Continuous densify panel 20. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 20 (original).](../assets/figures/ml_fig_c35_19.png)

*Figure — Continuous densify panel 20. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 20 (original).](../assets/figures/ml_fig_c36_19.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 20 (original).](../assets/figures/ml_fig_c37_19.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 20 (original).](../assets/figures/ml_fig_c38_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 20 (original).](../assets/figures/ml_fig_c39_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 20 (original).](../assets/figures/ml_fig_c40_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 20 (original).](../assets/figures/ml_fig_c41_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 20 (original).](../assets/figures/ml_fig_c42_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 20 (original).](../assets/figures/ml_fig_c43_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 20 (original).](../assets/figures/ml_fig_c44_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 20 (original).](../assets/figures/ml_fig_c45_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 20 (original).](../assets/figures/ml_fig_c46_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 20 (original).](../assets/figures/ml_fig_c47_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 20 (original).](../assets/figures/ml_fig_c48_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 20 (original).](../assets/figures/ml_fig_c49_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 20 (original).](../assets/figures/ml_fig_c50_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 20 (original).](../assets/figures/ml_fig_c51_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 20 (original).](../assets/figures/ml_fig_c52_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 20 (original).](../assets/figures/ml_fig_c53_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 20 (original).](../assets/figures/ml_fig_c54_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 20 (original).](../assets/figures/ml_fig_c55_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 20 (original).](../assets/figures/ml_fig_c56_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 20 (original).](../assets/figures/ml_fig_c57_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 20 (original).](../assets/figures/ml_fig_c58_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 20 (original).](../assets/figures/ml_fig_c59_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 20 (original).](../assets/figures/ml_fig_c60_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 20 (original).](../assets/figures/ml_fig_c61_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 20 (original).](../assets/figures/ml_fig_c62_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 20 (original).](../assets/figures/ml_fig_c63_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 20 (original).](../assets/figures/ml_fig_c64_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 20 (original).](../assets/figures/ml_fig_c65_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 20 (original).](../assets/figures/ml_fig_c66_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 20 (original).](../assets/figures/ml_fig_c67_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 20 (original).](../assets/figures/ml_fig_c68_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 20 (original).](../assets/figures/ml_fig_c69_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 20 (original).](../assets/figures/ml_fig_c70_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 20 (original).](../assets/figures/ml_fig_c71_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 20 (original).](../assets/figures/ml_fig_c72_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 20 (original).](../assets/figures/ml_fig_c73_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 20 (original).](../assets/figures/ml_fig_c74_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 20 (original).](../assets/figures/ml_fig_c75_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 20 (original).](../assets/figures/ml_fig_c76_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 20 (original).](../assets/figures/ml_fig_c77_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 20 (original).](../assets/figures/ml_fig_c78_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 20 (original).](../assets/figures/ml_fig_c79_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 20 (original).](../assets/figures/ml_fig_c80_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 20 (original).](../assets/figures/ml_fig_c81_19.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

*Figure — Glossary leakage map. Four common families: **temporal** (post-decision features), **fit/CV** (scaler/vocab/selector fit on the full cohort), **label proxy** (treatment or post-outcome codes as inputs), and **target-encoding** without LOO/OOF. All inflate apparent performance at train time and fail at true index time. Prediction ≠ causation.*

![c82 teaching panel 19 (original).](../assets/figures/ml_fig_c82_19.png)
*Figure — Glossary hub map: loss, capacity, validation, calibration, bias, generalization. Synthetic teaching geometry—not a causal claim.*

![c83 teaching panel 19 (original).](../assets/figures/ml_fig_c83_19.png)
*Figure — Core notation strip shared across the glossary. Synthetic teaching geometry—not a causal claim.*

![c84 teaching panel 19 (original).](../assets/figures/ml_fig_c84_19.png)
*Figure — DAG reminder: a confounder U can make association non-causal (pred ≠ cause). Synthetic teaching geometry—not a causal claim.*

![c85 teaching panel 19 (original).](../assets/figures/ml_fig_c85_19.png)
*Figure — Metric glossary strip: AUROC, ECE, PPV, F1, NLL. Synthetic teaching geometry—not a causal claim.*

![c86 teaching panel 19 (original).](../assets/figures/ml_fig_c86_19.png)
*Figure — Bayesian glossary chain: prior→likelihood→posterior. Synthetic teaching geometry—not a causal claim.*

![c87 teaching panel 19 (original).](../assets/figures/ml_fig_c87_19.png)
*Figure — Error decomposition glossary: bias, variance, noise, risk. Synthetic teaching geometry—not a causal claim.*

![c88 teaching panel 19 (original).](../assets/figures/ml_fig_c88_19.png)
*Figure — Symbols: E, Var, Cov, KL, H(X). Synthetic teaching geometry—not a causal claim.*

![c89 teaching panel 19 (original).](../assets/figures/ml_fig_c89_19.png)
*Figure — Abbrev strip: CV, OOD, SSL, RL, GNN. Synthetic teaching geometry—not a causal claim.*

![c90 teaching panel 19 (original).](../assets/figures/ml_fig_c90_19.png)
*Figure — Key identities: Bayes, chain, bias-var. Synthetic teaching geometry—not a causal claim.*

![c91 teaching panel 19 (original).](../assets/figures/ml_fig_c91_19.png)
*Figure — Acronym strip: AUC ECE Brier NLL. Synthetic teaching geometry—not a causal claim.*

![c92 teaching panel 19 (original).](../assets/figures/ml_fig_c92_19.png)
*Figure — Greek strip: alpha beta gamma lambda. Synthetic teaching geometry—not a causal claim.*

![c93 teaching panel 19 (original).](../assets/figures/ml_fig_c93_19.png)
*Figure — Inequality strip: Jensen, Hoeffding. Synthetic teaching geometry—not a causal claim.*

![c94 teaching panel 19 (original).](../assets/figures/ml_fig_c94_19.png)
*Figure — Loss zoo strip: CE MSE Huber. Synthetic teaching geometry—not a causal claim.*

![c95 teaching panel 19 (original).](../assets/figures/ml_fig_c95_19.png)
*Figure — Metric strip: recall F1 MCC. Synthetic teaching geometry—not a causal claim.*

![c96 teaching panel 19 (original).](../assets/figures/ml_fig_c96_19.png)
*Figure — Bound strip: VC, Rademacher. Synthetic teaching geometry—not a causal claim.*

![c97 teaching panel 19 (original).](../assets/figures/ml_fig_c97_19.png)
*Figure — Optimization strip: GD SGD Adam. Synthetic teaching geometry—not a causal claim.*

![c98 teaching panel 19 (original).](../assets/figures/ml_fig_c98_19.png)
*Figure — Prob strip: PDF CDF PMF HF. Synthetic teaching geometry—not a causal claim.*

![c99 teaching panel 19 (original).](../assets/figures/ml_fig_c99_19.png)
*Figure — Inequality strip II: Bernstein. Synthetic teaching geometry—not a causal claim.*

![c100 teaching panel 19 (original).](../assets/figures/ml_fig_c100_19.png)
*Figure — Eval strip: ID OOD stress. Synthetic teaching geometry—not a causal claim.*

![c101 teaching panel 19 (original).](../assets/figures/ml_fig_c101_19.png)
*Figure — Info strip: MI KL JS. Synthetic teaching geometry—not a causal claim.*

![c102 teaching panel 19 (original).](../assets/figures/ml_fig_c102_19.png)
*Figure — Concentration strip: Chernoff. Synthetic teaching geometry—not a causal claim.*

![c103 teaching panel 19 (original).](../assets/figures/ml_fig_c103_19.png)
*Figure — Robustness strip: noise shift. Synthetic teaching geometry—not a causal claim.*

![c104 teaching panel 19 (original).](../assets/figures/ml_fig_c104_19.png)
*Figure — Causal strip: do-calc caution. Synthetic teaching geometry—not a causal claim.*

![c105 teaching panel 19 (original).](../assets/figures/ml_fig_c105_19.png)
*Figure — PAC-Bayes bound strip. Synthetic teaching geometry—not a causal claim.*

![c106 teaching panel 19 (original).](../assets/figures/ml_fig_c106_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c107 teaching panel 19 (original).](../assets/figures/ml_fig_c107_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c108 teaching panel 19 (original).](../assets/figures/ml_fig_c108_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c109 teaching panel 19 (original).](../assets/figures/ml_fig_c109_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c110 teaching panel 19 (original).](../assets/figures/ml_fig_c110_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c111 teaching panel 19 (original).](../assets/figures/ml_fig_c111_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c112 teaching panel 19 (original).](../assets/figures/ml_fig_c112_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c113 teaching panel 19 (original).](../assets/figures/ml_fig_c113_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c114 teaching panel 19 (original).](../assets/figures/ml_fig_c114_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c115 teaching panel 19 (original).](../assets/figures/ml_fig_c115_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c116 teaching panel 19 (original).](../assets/figures/ml_fig_c116_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c117 teaching panel 19 (original).](../assets/figures/ml_fig_c117_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c118 teaching panel 19 (original).](../assets/figures/ml_fig_c118_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c119 teaching panel 19 (original).](../assets/figures/ml_fig_c119_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c120 teaching panel 19 (original).](../assets/figures/ml_fig_c120_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c121 teaching panel 19 (original).](../assets/figures/ml_fig_c121_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c122 teaching panel 19 (original).](../assets/figures/ml_fig_c122_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c123 teaching panel 19 (original).](../assets/figures/ml_fig_c123_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c124 teaching panel 19 (original).](../assets/figures/ml_fig_c124_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c125 teaching panel 19 (original).](../assets/figures/ml_fig_c125_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c126 teaching panel 19 (original).](../assets/figures/ml_fig_c126_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c127 teaching panel 19 (original).](../assets/figures/ml_fig_c127_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c128 teaching panel 19 (original).](../assets/figures/ml_fig_c128_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c129 teaching panel 19 (original).](../assets/figures/ml_fig_c129_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c130 teaching panel 19 (original).](../assets/figures/ml_fig_c130_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c131 teaching panel 19 (original).](../assets/figures/ml_fig_c131_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c132 teaching panel 19 (original).](../assets/figures/ml_fig_c132_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c133 teaching panel 19 (original).](../assets/figures/ml_fig_c133_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c134 teaching panel 19 (original).](../assets/figures/ml_fig_c134_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c135 teaching panel 19 (original).](../assets/figures/ml_fig_c135_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c136 teaching panel 19 (original).](../assets/figures/ml_fig_c136_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c137 teaching panel 19 (original).](../assets/figures/ml_fig_c137_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c138 teaching panel 19 (original).](../assets/figures/ml_fig_c138_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c139 teaching panel 19 (original).](../assets/figures/ml_fig_c139_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c140 teaching panel 19 (original).](../assets/figures/ml_fig_c140_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c141 teaching panel 19 (original).](../assets/figures/ml_fig_c141_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c142 teaching panel 19 (original).](../assets/figures/ml_fig_c142_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c143 teaching panel 19 (original).](../assets/figures/ml_fig_c143_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c144 teaching panel 19 (original).](../assets/figures/ml_fig_c144_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c145 teaching panel 19 (original).](../assets/figures/ml_fig_c145_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c146 teaching panel 19 (original).](../assets/figures/ml_fig_c146_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c147 teaching panel 19 (original).](../assets/figures/ml_fig_c147_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c148 teaching panel 19 (original).](../assets/figures/ml_fig_c148_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c149 teaching panel 19 (original).](../assets/figures/ml_fig_c149_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c150 teaching panel 19 (original).](../assets/figures/ml_fig_c150_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c151 teaching panel 19 (original).](../assets/figures/ml_fig_c151_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c152 teaching panel 19 (original).](../assets/figures/ml_fig_c152_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c153 teaching panel 19 (original).](../assets/figures/ml_fig_c153_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c154 teaching panel 19 (original).](../assets/figures/ml_fig_c154_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c155 teaching panel 19 (original).](../assets/figures/ml_fig_c155_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c156 teaching panel 19 (original).](../assets/figures/ml_fig_c156_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c157 teaching panel 19 (original).](../assets/figures/ml_fig_c157_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c158 teaching panel 19 (original).](../assets/figures/ml_fig_c158_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c159 teaching panel 19 (original).](../assets/figures/ml_fig_c159_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c160 teaching panel 19 (original).](../assets/figures/ml_fig_c160_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c161 teaching panel 19 (original).](../assets/figures/ml_fig_c161_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c162 teaching panel 19 (original).](../assets/figures/ml_fig_c162_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c163 teaching panel 19 (original).](../assets/figures/ml_fig_c163_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c164 teaching panel 19 (original).](../assets/figures/ml_fig_c164_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c165 teaching panel 19 (original).](../assets/figures/ml_fig_c165_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c166 teaching panel 19 (original).](../assets/figures/ml_fig_c166_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c167 teaching panel 19 (original).](../assets/figures/ml_fig_c167_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c168 teaching panel 19 (original).](../assets/figures/ml_fig_c168_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c169 teaching panel 19 (original).](../assets/figures/ml_fig_c169_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c170 teaching panel 19 (original).](../assets/figures/ml_fig_c170_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c171 teaching panel 19 (original).](../assets/figures/ml_fig_c171_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c172 teaching panel 19 (original).](../assets/figures/ml_fig_c172_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c173 teaching panel 19 (original).](../assets/figures/ml_fig_c173_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c174 teaching panel 19 (original).](../assets/figures/ml_fig_c174_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c175 teaching panel 19 (original).](../assets/figures/ml_fig_c175_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c176 teaching panel 19 (original).](../assets/figures/ml_fig_c176_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c177 teaching panel 19 (original).](../assets/figures/ml_fig_c177_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c178 teaching panel 19 (original).](../assets/figures/ml_fig_c178_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c179 teaching panel 19 (original).](../assets/figures/ml_fig_c179_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c180 teaching panel 19 (original).](../assets/figures/ml_fig_c180_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c181 teaching panel 19 (original).](../assets/figures/ml_fig_c181_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c182 teaching panel 19 (original).](../assets/figures/ml_fig_c182_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c183 teaching panel 19 (original).](../assets/figures/ml_fig_c183_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c184 teaching panel 19 (original).](../assets/figures/ml_fig_c184_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c185 teaching panel 19 (original).](../assets/figures/ml_fig_c185_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c186 teaching panel 19 (original).](../assets/figures/ml_fig_c186_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c187 teaching panel 19 (original).](../assets/figures/ml_fig_c187_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c188 teaching panel 19 (original).](../assets/figures/ml_fig_c188_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c189 teaching panel 19 (original).](../assets/figures/ml_fig_c189_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c190 teaching panel 19 (original).](../assets/figures/ml_fig_c190_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c191 teaching panel 19 (original).](../assets/figures/ml_fig_c191_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c192 teaching panel 19 (original).](../assets/figures/ml_fig_c192_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c193 teaching panel 19 (original).](../assets/figures/ml_fig_c193_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c194 teaching panel 19 (original).](../assets/figures/ml_fig_c194_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c195 teaching panel 19 (original).](../assets/figures/ml_fig_c195_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c196 teaching panel 19 (original).](../assets/figures/ml_fig_c196_19.png)
*Figure — Glossary support vector. Synthetic teaching geometry—not a causal claim.*

![c197 teaching panel 19 (original).](../assets/figures/ml_fig_c197_19.png)
*Figure — Glossary inductive bias. Synthetic teaching geometry—not a causal claim.*

![c198 teaching panel 19 (original).](../assets/figures/ml_fig_c198_19.png)
*Figure — Glossary conformal set. Synthetic teaching geometry—not a causal claim.*

![c199 teaching panel 19 (original).](../assets/figures/ml_fig_c199_19.png)
*Figure — Glossary aleatoric risk. Synthetic teaching geometry—not a causal claim.*

![c200 teaching panel 19 (original).](../assets/figures/ml_fig_c200_19.png)
*Figure — Glossary epistemic risk. Synthetic teaching geometry—not a causal claim.*

![c201 teaching panel 19 (original).](../assets/figures/ml_fig_c201_19.png)
*Figure — Bias variance irreducible strip. Synthetic teaching geometry—not a causal claim.*

![c202 teaching panel 19 (original).](../assets/figures/ml_fig_c202_19.png)
*Figure — Glossary margin support kernel. Synthetic teaching geometry—not a causal claim.*

![c203 teaching panel 19 (original).](../assets/figures/ml_fig_c203_19.png)
*Figure — Glossary entropy KL CE strip. Synthetic teaching geometry—not a causal claim.*

![c204 teaching panel 19 (original).](../assets/figures/ml_fig_c204_19.png)
*Figure — Glossary regularizer name strip. Synthetic teaching geometry—not a causal claim.*

![c205 teaching panel 19 (original).](../assets/figures/ml_fig_c205_19.png)
*Figure — Glossary Bayes term strip. Synthetic teaching geometry—not a causal claim.*

![c206 teaching panel 19 (original).](../assets/figures/ml_fig_c206_19.png)
*Figure — Glossary optimizer name strip. Synthetic teaching geometry—not a causal claim.*

![c207 teaching panel 19 (original).](../assets/figures/ml_fig_c207_19.png)
*Figure — Glossary loss function strip. Synthetic teaching geometry—not a causal claim.*

![c208 teaching panel 19 (original).](../assets/figures/ml_fig_c208_19.png)
*Figure — Glossary calibration metric strip. Synthetic teaching geometry—not a causal claim.*

![c209 teaching panel 19 (original).](../assets/figures/ml_fig_c209_19.png)
*Figure — Glossary discrimination metric strip. Synthetic teaching geometry—not a causal claim.*

![c210 teaching panel 19 (original).](../assets/figures/ml_fig_c210_19.png)
*Figure — Glossary self-supervised name strip. Synthetic teaching geometry—not a causal claim.*

![c211 teaching panel 19 (original).](../assets/figures/ml_fig_c211_19.png)
*Figure — Glossary graph mining strip. Synthetic teaching geometry—not a causal claim.*

![c212 teaching panel 19 (original).](../assets/figures/ml_fig_c212_19.png)
*Figure — Glossary deployment pattern strip. Synthetic teaching geometry—not a causal claim.*

![c213 teaching panel 19 (original).](../assets/figures/ml_fig_c213_19.png)
*Figure — Glossary core stats strip. Synthetic teaching geometry—not a causal claim.*

![c214 teaching panel 19 (original).](../assets/figures/ml_fig_c214_19.png)
*Figure — Glossary causal term strip. Synthetic teaching geometry—not a causal claim.*

![c215 teaching panel 19 (original).](../assets/figures/ml_fig_c215_19.png)
*Figure — Glossary network architecture strip. Synthetic teaching geometry—not a causal claim.*

![c216 teaching panel 19 (original).](../assets/figures/ml_fig_c216_19.png)
*Figure — Glossary RL core term strip. Synthetic teaching geometry—not a causal claim.*

![c217 teaching panel 19 (original).](../assets/figures/ml_fig_c217_19.png)
*Figure — Glossary feature engineering strip. Synthetic teaching geometry—not a causal claim.*

![c218 teaching panel 19 (original).](../assets/figures/ml_fig_c218_19.png)
*Figure — Glossary dim-reduction strip. Synthetic teaching geometry—not a causal claim.*

![c219 teaching panel 19 (original).](../assets/figures/ml_fig_c219_19.png)
*Figure — Glossary regression family strip. Synthetic teaching geometry—not a causal claim.*

![c220 teaching panel 19 (original).](../assets/figures/ml_fig_c220_19.png)
*Figure — Glossary classification strip. Synthetic teaching geometry—not a causal claim.*

![c221 teaching panel 19 (original).](../assets/figures/ml_fig_c221_19.png)
*Figure — Glossary probability term strip. Synthetic teaching geometry—not a causal claim.*

![c222 teaching panel 19 (original).](../assets/figures/ml_fig_c222_19.png)
*Figure — Glossary optimizer family strip. Synthetic teaching geometry—not a causal claim.*

![c223 teaching panel 19 (original).](../assets/figures/ml_fig_c223_19.png)
*Figure — Glossary clustering family strip. Synthetic teaching geometry—not a causal claim.*

![c224 teaching panel 19 (original).](../assets/figures/ml_fig_c224_19.png)
*Figure — Glossary self-supervised strip. Synthetic teaching geometry—not a causal claim.*

![c225 teaching panel 19 (original).](../assets/figures/ml_fig_c225_19.png)
*Figure — Glossary graph method strip. Synthetic teaching geometry—not a causal claim.*

![c226 teaching panel 19 (original).](../assets/figures/ml_fig_c226_19.png)
*Figure — Glossary RL algorithm strip. Synthetic teaching geometry—not a causal claim.*

![c227 teaching panel 19 (original).](../assets/figures/ml_fig_c227_19.png)
*Figure — Glossary dim-reduction strip. Synthetic teaching geometry—not a causal claim.*

![c228 teaching panel 19 (original).](../assets/figures/ml_fig_c228_19.png)
*Figure — Glossary compression strip. Synthetic teaching geometry—not a causal claim.*

![c229 teaching panel 19 (original).](../assets/figures/ml_fig_c229_19.png)
*Figure — Glossary feature-encoding strip. Synthetic teaching geometry—not a causal claim.*

![c230 teaching panel 19 (original).](../assets/figures/ml_fig_c230_19.png)
*Figure — Glossary evaluation metric strip. Synthetic teaching geometry—not a causal claim.*

![c231 teaching panel 19 (original).](../assets/figures/ml_fig_c231_19.png)
*Figure — Glossary neural architecture strip. Synthetic teaching geometry—not a causal claim.*

![c232 teaching panel 19 (original).](../assets/figures/ml_fig_c232_19.png)
*Figure — Glossary data-ops strip. Synthetic teaching geometry—not a causal claim.*

![c233 teaching panel 19 (original).](../assets/figures/ml_fig_c233_19.png)
*Figure — Glossary loss family strip. Synthetic teaching geometry—not a causal claim.*

![c234 teaching panel 19 (original).](../assets/figures/ml_fig_c234_19.png)
*Figure — Glossary SSL loss strip. Synthetic teaching geometry—not a causal claim.*

![c235 teaching panel 19 (original).](../assets/figures/ml_fig_c235_19.png)
*Figure — Glossary metric family strip. Synthetic teaching geometry—not a causal claim.*

![c236 teaching panel 19 (original).](../assets/figures/ml_fig_c236_19.png)
*Figure — Glossary RL objective strip. Synthetic teaching geometry—not a causal claim.*

![c237 teaching panel 19 (original).](../assets/figures/ml_fig_c237_19.png)
*Figure — Glossary calib metric strip. Synthetic teaching geometry—not a causal claim.*

![c238 teaching panel 19 (original).](../assets/figures/ml_fig_c238_19.png)
*Figure — Glossary graph algo strip. Synthetic teaching geometry—not a causal claim.*

![c239 teaching panel 19 (original).](../assets/figures/ml_fig_c239_19.png)
*Figure — Glossary deploy checklist strip. Synthetic teaching geometry—not a causal claim.*

![c240 teaching panel 19 (original).](../assets/figures/ml_fig_c240_19.png)
*Figure — Glossary privacy term strip. Synthetic teaching geometry—not a causal claim.*

![c241 teaching panel 19 (original).](../assets/figures/ml_fig_c241_19.png)
*Figure — Glossary optim term strip. Synthetic teaching geometry—not a causal claim.*

![c242 teaching panel 19 (original).](../assets/figures/ml_fig_c242_19.png)
*Figure — Glossary SSL objective strip. Synthetic teaching geometry—not a causal claim.*

![c243 teaching panel 19 (original).](../assets/figures/ml_fig_c243_19.png)
*Figure — Glossary regularizer strip. Synthetic teaching geometry—not a causal claim.*

![c244 teaching panel 19 (original).](../assets/figures/ml_fig_c244_19.png)
*Figure — Glossary gen model strip. Synthetic teaching geometry—not a causal claim.*

![c245 teaching panel 19 (original).](../assets/figures/ml_fig_c245_19.png)
*Figure — Glossary loss family strip. Synthetic teaching geometry—not a causal claim.*

![c246 teaching panel 19 (original).](../assets/figures/ml_fig_c246_19.png)
*Figure — Glossary embedding strip. Synthetic teaching geometry—not a causal claim.*

![c247 teaching panel 19 (original).](../assets/figures/ml_fig_c247_19.png)
*Figure — Glossary eval metric strip. Synthetic teaching geometry—not a causal claim.*

![c248 teaching panel 19 (original).](../assets/figures/ml_fig_c248_19.png)
*Figure — Glossary optim trick strip. Synthetic teaching geometry—not a causal claim.*

![c249 teaching panel 19 (original).](../assets/figures/ml_fig_c249_19.png)
*Figure — Glossary calib strip. Synthetic teaching geometry—not a causal claim.*

![c250 teaching panel 19 (original).](../assets/figures/ml_fig_c250_19.png)
*Figure — Glossary graph term strip. Synthetic teaching geometry—not a causal claim.*

![c251 teaching panel 19 (original).](../assets/figures/ml_fig_c251_19.png)
*Figure — Glossary SSL strip. Synthetic teaching geometry—not a causal claim.*

![c252 teaching panel 19 (original).](../assets/figures/ml_fig_c252_19.png)
*Figure — Glossary privacy strip. Synthetic teaching geometry—not a causal claim.*

![c253 teaching panel 19 (original).](../assets/figures/ml_fig_c253_19.png)
*Figure — Glossary regularize strip. Synthetic teaching geometry—not a causal claim.*

![c254 teaching panel 19 (original).](../assets/figures/ml_fig_c254_19.png)
*Figure — Glossary RL strip. Synthetic teaching geometry—not a causal claim.*

![c255 teaching panel 19 (original).](../assets/figures/ml_fig_c255_19.png)
*Figure — Glossary optim strip. Synthetic teaching geometry—not a causal claim.*

![c256 teaching panel 19 (original).](../assets/figures/ml_fig_c256_19.png)
*Figure — Glossary genAI strip. Synthetic teaching geometry—not a causal claim.*

![c257 teaching panel 19 (original).](../assets/figures/ml_fig_c257_19.png)
*Figure — Glossary embedding strip c257. Synthetic teaching geometry—not a causal claim.*

![c258 teaching panel 19 (original).](../assets/figures/ml_fig_c258_19.png)
*Figure — Glossary quant strip c258. Synthetic teaching geometry—not a causal claim.*

![c259 teaching panel 19 (original).](../assets/figures/ml_fig_c259_19.png)
*Figure — Glossary eval strip c259. Synthetic teaching geometry—not a causal claim.*

![c260 teaching panel 19 (original).](../assets/figures/ml_fig_c260_19.png)
*Figure — Glossary cluster strip c260. Synthetic teaching geometry—not a causal claim.*

![c261 teaching panel 19 (original).](../assets/figures/ml_fig_c261_19.png)
*Figure — Glossary IR strip c261. Synthetic teaching geometry—not a causal claim.*

![c262 teaching panel 19 (original).](../assets/figures/ml_fig_c262_19.png)
*Figure — Glossary deploy strip c262. Synthetic teaching geometry—not a causal claim.*

![c263 teaching panel 19 (original).](../assets/figures/ml_fig_c263_19.png)
*Figure — Glossary loss strip c263. Synthetic teaching geometry—not a causal claim.*

![c264 teaching panel 19 (original).](../assets/figures/ml_fig_c264_19.png)
*Figure — Glossary optim strip c264. Synthetic teaching geometry—not a causal claim.*

![c265 teaching panel 19 (original).](../assets/figures/ml_fig_c265_19.png)
*Figure — Glossary SSL strip c265. Synthetic teaching geometry—not a causal claim.*

![c266 teaching panel 19 (original).](../assets/figures/ml_fig_c266_19.png)
*Figure — Glossary RL strip c266. Synthetic teaching geometry—not a causal claim.*

![c267 teaching panel 19 (original).](../assets/figures/ml_fig_c267_19.png)
*Figure — Glossary privacy strip c267. Synthetic teaching geometry—not a causal claim.*

![c268 teaching panel 19 (original).](../assets/figures/ml_fig_c268_19.png)
*Figure — Glossary graph strip c268. Synthetic teaching geometry—not a causal claim.*

![c269 teaching panel 19 (original).](../assets/figures/ml_fig_c269_19.png)
*Figure — Glossary calib strip c269. Synthetic teaching geometry—not a causal claim.*

![c270 teaching panel 19 (original).](../assets/figures/ml_fig_c270_19.png)
*Figure — Glossary genAI strip c270. Synthetic teaching geometry—not a causal claim.*

![c271 teaching panel 19 (original).](../assets/figures/ml_fig_c271_19.png)
*Figure — Glossary metric strip c271. Synthetic teaching geometry—not a causal claim.*

![c272 teaching panel 19 (original).](../assets/figures/ml_fig_c272_19.png)
*Figure — Glossary regularize strip c272. Synthetic teaching geometry—not a causal claim.*

![c273 teaching panel 19 (original).](../assets/figures/ml_fig_c273_19.png)
*Figure — Glossary embedding strip c273. Synthetic teaching geometry—not a causal claim.*

![c274 teaching panel 19 (original).](../assets/figures/ml_fig_c274_19.png)
*Figure — Glossary quant strip c274. Synthetic teaching geometry—not a causal claim.*

![c275 teaching panel 19 (original).](../assets/figures/ml_fig_c275_19.png)
*Figure — Glossary eval strip c275. Synthetic teaching geometry—not a causal claim.*

![c276 teaching panel 19 (original).](../assets/figures/ml_fig_c276_19.png)
*Figure — Glossary cluster strip c276. Synthetic teaching geometry—not a causal claim.*

![c277 teaching panel 19 (original).](../assets/figures/ml_fig_c277_19.png)
*Figure — Glossary IR strip c277. Synthetic teaching geometry—not a causal claim.*

![c278 teaching panel 19 (original).](../assets/figures/ml_fig_c278_19.png)
*Figure — Glossary deploy strip c278. Synthetic teaching geometry—not a causal claim.*

![c279 teaching panel 19 (original).](../assets/figures/ml_fig_c279_19.png)
*Figure — Glossary loss strip c279. Synthetic teaching geometry—not a causal claim.*

![c280 teaching panel 19 (original).](../assets/figures/ml_fig_c280_19.png)
*Figure — Glossary optim strip c280. Synthetic teaching geometry—not a causal claim.*

![c281 teaching panel 19 (original).](../assets/figures/ml_fig_c281_19.png)
*Figure — Glossary SSL strip c281. Synthetic teaching geometry—not a causal claim.*

![c282 teaching panel 19 (original).](../assets/figures/ml_fig_c282_19.png)
*Figure — Glossary RL strip c282. Synthetic teaching geometry—not a causal claim.*

![c283 teaching panel 19 (original).](../assets/figures/ml_fig_c283_19.png)
*Figure — Glossary privacy strip c283. Synthetic teaching geometry—not a causal claim.*

![c284 teaching panel 19 (original).](../assets/figures/ml_fig_c284_19.png)
*Figure — Glossary graph strip c284. Synthetic teaching geometry—not a causal claim.*

![c285 teaching panel 19 (original).](../assets/figures/ml_fig_c285_19.png)
*Figure — Glossary calib strip c285. Synthetic teaching geometry—not a causal claim.*

![c286 teaching panel 19 (original).](../assets/figures/ml_fig_c286_19.png)
*Figure — Glossary genAI strip c286. Synthetic teaching geometry—not a causal claim.*

![c287 teaching panel 19 (original).](../assets/figures/ml_fig_c287_19.png)
*Figure — Glossary metric strip c287. Synthetic teaching geometry—not a causal claim.*

![c288 teaching panel 19 (original).](../assets/figures/ml_fig_c288_19.png)
*Figure — Glossary regularize strip c288. Synthetic teaching geometry—not a causal claim.*

![c289 teaching panel 19 (original).](../assets/figures/ml_fig_c289_19.png)
*Figure — Glossary embedding strip c289. Synthetic teaching geometry—not a causal claim.*

![c290 teaching panel 19 (original).](../assets/figures/ml_fig_c290_19.png)
*Figure — Glossary quant strip c290. Synthetic teaching geometry—not a causal claim.*

![c291 teaching panel 19 (original).](../assets/figures/ml_fig_c291_19.png)
*Figure — Glossary eval strip c291. Synthetic teaching geometry—not a causal claim.*

![c292 teaching panel 19 (original).](../assets/figures/ml_fig_c292_19.png)
*Figure — Glossary cluster strip c292. Synthetic teaching geometry—not a causal claim.*

![c293 teaching panel 19 (original).](../assets/figures/ml_fig_c293_19.png)
*Figure — Glossary IR strip c293. Synthetic teaching geometry—not a causal claim.*

![c294 teaching panel 19 (original).](../assets/figures/ml_fig_c294_19.png)
*Figure — Glossary deploy strip c294. Synthetic teaching geometry—not a causal claim.*

![c295 teaching panel 19 (original).](../assets/figures/ml_fig_c295_19.png)
*Figure — Glossary loss strip c295. Synthetic teaching geometry—not a causal claim.*

![c296 teaching panel 19 (original).](../assets/figures/ml_fig_c296_19.png)
*Figure — Glossary optim strip c296. Synthetic teaching geometry—not a causal claim.*

![c297 teaching panel 19 (original).](../assets/figures/ml_fig_c297_19.png)
*Figure — Glossary SSL strip c297. Synthetic teaching geometry—not a causal claim.*

![c298 teaching panel 19 (original).](../assets/figures/ml_fig_c298_19.png)
*Figure — Glossary RL strip c298. Synthetic teaching geometry—not a causal claim.*

![c299 teaching panel 19 (original).](../assets/figures/ml_fig_c299_19.png)
*Figure — Glossary privacy strip c299. Synthetic teaching geometry—not a causal claim.*

![c300 teaching panel 19 (original).](../assets/figures/ml_fig_c300_19.png)
*Figure — Glossary graph strip c300. Synthetic teaching geometry—not a causal claim.*

![c301 teaching panel 19 (original).](../assets/figures/ml_fig_c301_19.png)
*Figure — Glossary calib strip c301. Synthetic teaching geometry—not a causal claim.*

![c302 teaching panel 19 (original).](../assets/figures/ml_fig_c302_19.png)
*Figure — Glossary genAI strip c302. Synthetic teaching geometry—not a causal claim.*

![c303 teaching panel 19 (original).](../assets/figures/ml_fig_c303_19.png)
*Figure — Glossary metric strip c303. Synthetic teaching geometry—not a causal claim.*

![c304 teaching panel 19 (original).](../assets/figures/ml_fig_c304_19.png)
*Figure — Glossary regularize strip c304. Synthetic teaching geometry—not a causal claim.*

![c305 teaching panel 19 (original).](../assets/figures/ml_fig_c305_19.png)
*Figure — Glossary embedding strip c305. Synthetic teaching geometry—not a causal claim.*

![c306 teaching panel 19 (original).](../assets/figures/ml_fig_c306_19.png)
*Figure — Glossary quant strip c306. Synthetic teaching geometry—not a causal claim.*

![c307 teaching panel 19 (original).](../assets/figures/ml_fig_c307_19.png)
*Figure — Glossary eval strip c307. Synthetic teaching geometry—not a causal claim.*

![c308 teaching panel 19 (original).](../assets/figures/ml_fig_c308_19.png)
*Figure — Glossary cluster strip c308. Synthetic teaching geometry—not a causal claim.*

![c309 teaching panel 19 (original).](../assets/figures/ml_fig_c309_19.png)
*Figure — Glossary IR strip c309. Synthetic teaching geometry—not a causal claim.*

![c310 teaching panel 19 (original).](../assets/figures/ml_fig_c310_19.png)
*Figure — Glossary deploy strip c310. Synthetic teaching geometry—not a causal claim.*

![c311 teaching panel 19 (original).](../assets/figures/ml_fig_c311_19.png)
*Figure — Glossary loss strip c311. Synthetic teaching geometry—not a causal claim.*

![c312 teaching panel 19 (original).](../assets/figures/ml_fig_c312_19.png)
*Figure — Glossary optim strip c312. Synthetic teaching geometry—not a causal claim.*

![c313 teaching panel 19 (original).](../assets/figures/ml_fig_c313_19.png)
*Figure — Glossary SSL strip c313. Synthetic teaching geometry—not a causal claim.*

![c314 teaching panel 19 (original).](../assets/figures/ml_fig_c314_19.png)
*Figure — Glossary RL strip c314. Synthetic teaching geometry—not a causal claim.*

![c315 teaching panel 19 (original).](../assets/figures/ml_fig_c315_19.png)
*Figure — Glossary privacy strip c315. Synthetic teaching geometry—not a causal claim.*

![c316 teaching panel 19 (original).](../assets/figures/ml_fig_c316_19.png)
*Figure — Glossary graph strip c316. Synthetic teaching geometry—not a causal claim.*

![c317 teaching panel 19 (original).](../assets/figures/ml_fig_c317_19.png)
*Figure — Glossary calib strip c317. Synthetic teaching geometry—not a causal claim.*

![c318 teaching panel 19 (original).](../assets/figures/ml_fig_c318_19.png)
*Figure — Glossary genAI strip c318. Synthetic teaching geometry—not a causal claim.*

![c319 teaching panel 19 (original).](../assets/figures/ml_fig_c319_19.png)
*Figure — Glossary metric strip c319. Synthetic teaching geometry—not a causal claim.*

![c320 teaching panel 19 (original).](../assets/figures/ml_fig_c320_19.png)
*Figure — Glossary regularize strip c320. Synthetic teaching geometry—not a causal claim.*

![c321 teaching panel 19 (original).](../assets/figures/ml_fig_c321_19.png)
*Figure — Glossary embedding strip c321. Synthetic teaching geometry—not a causal claim.*

![c322 teaching panel 19 (original).](../assets/figures/ml_fig_c322_19.png)
*Figure — Glossary quant strip c322. Synthetic teaching geometry—not a causal claim.*

![c323 teaching panel 19 (original).](../assets/figures/ml_fig_c323_19.png)
*Figure — Glossary eval strip c323. Synthetic teaching geometry—not a causal claim.*

![c324 teaching panel 19 (original).](../assets/figures/ml_fig_c324_19.png)
*Figure — Glossary cluster strip c324. Synthetic teaching geometry—not a causal claim.*

![c325 teaching panel 19 (original).](../assets/figures/ml_fig_c325_19.png)
*Figure — Glossary IR strip c325. Synthetic teaching geometry—not a causal claim.*

![c326 teaching panel 19 (original).](../assets/figures/ml_fig_c326_19.png)
*Figure — Glossary deploy strip c326. Synthetic teaching geometry—not a causal claim.*

![c327 teaching panel 19 (original).](../assets/figures/ml_fig_c327_19.png)
*Figure — Glossary loss strip c327. Synthetic teaching geometry—not a causal claim.*

![c328 teaching panel 19 (original).](../assets/figures/ml_fig_c328_19.png)
*Figure — Glossary optim strip c328. Synthetic teaching geometry—not a causal claim.*

![c329 teaching panel 19 (original).](../assets/figures/ml_fig_c329_19.png)
*Figure — Glossary SSL strip c329. Synthetic teaching geometry—not a causal claim.*

![c330 teaching panel 19 (original).](../assets/figures/ml_fig_c330_19.png)
*Figure — Glossary RL strip c330. Synthetic teaching geometry—not a causal claim.*

![c331 teaching panel 19 (original).](../assets/figures/ml_fig_c331_19.png)
*Figure — Glossary privacy strip c331. Synthetic teaching geometry—not a causal claim.*

![c332 teaching panel 19 (original).](../assets/figures/ml_fig_c332_19.png)
*Figure — Glossary graph strip c332. Synthetic teaching geometry—not a causal claim.*

![c333 teaching panel 19 (original).](../assets/figures/ml_fig_c333_19.png)
*Figure — Glossary calib strip c333. Synthetic teaching geometry—not a causal claim.*

![c334 teaching panel 19 (original).](../assets/figures/ml_fig_c334_19.png)
*Figure — Glossary genAI strip c334. Synthetic teaching geometry—not a causal claim.*

![c335 teaching panel 19 (original).](../assets/figures/ml_fig_c335_19.png)
*Figure — Glossary metric strip c335. Synthetic teaching geometry—not a causal claim.*

![c336 teaching panel 19 (original).](../assets/figures/ml_fig_c336_19.png)
*Figure — Glossary regularize strip c336. Synthetic teaching geometry—not a causal claim.*

![c337 teaching panel 19 (original).](../assets/figures/ml_fig_c337_19.png)
*Figure — Glossary embedding strip c337. Synthetic teaching geometry—not a causal claim.*

![c338 teaching panel 19 (original).](../assets/figures/ml_fig_c338_19.png)
*Figure — Glossary quant strip c338. Synthetic teaching geometry—not a causal claim.*

![c339 teaching panel 19 (original).](../assets/figures/ml_fig_c339_19.png)
*Figure — Glossary eval strip c339. Synthetic teaching geometry—not a causal claim.*

![c340 teaching panel 19 (original).](../assets/figures/ml_fig_c340_19.png)
*Figure — Glossary cluster strip c340. Synthetic teaching geometry—not a causal claim.*

![c341 teaching panel 19 (original).](../assets/figures/ml_fig_c341_19.png)
*Figure — Glossary IR strip c341. Synthetic teaching geometry—not a causal claim.*

![c342 teaching panel 19 (original).](../assets/figures/ml_fig_c342_19.png)
*Figure — Glossary deploy strip c342. Synthetic teaching geometry—not a causal claim.*

![c343 teaching panel 19 (original).](../assets/figures/ml_fig_c343_19.png)
*Figure — Glossary loss strip c343. Synthetic teaching geometry—not a causal claim.*

![c344 teaching panel 19 (original).](../assets/figures/ml_fig_c344_19.png)
*Figure — Glossary optim strip c344. Synthetic teaching geometry—not a causal claim.*

![c345 teaching panel 19 (original).](../assets/figures/ml_fig_c345_19.png)
*Figure — Glossary SSL strip c345. Synthetic teaching geometry—not a causal claim.*

![c346 teaching panel 19 (original).](../assets/figures/ml_fig_c346_19.png)
*Figure — Glossary RL strip c346. Synthetic teaching geometry—not a causal claim.*

![c347 teaching panel 19 (original).](../assets/figures/ml_fig_c347_19.png)
*Figure — Glossary privacy strip c347. Synthetic teaching geometry—not a causal claim.*

![c348 teaching panel 19 (original).](../assets/figures/ml_fig_c348_19.png)
*Figure — Glossary graph strip c348. Synthetic teaching geometry—not a causal claim.*

![c349 teaching panel 19 (original).](../assets/figures/ml_fig_c349_19.png)
*Figure — Glossary calib strip c349. Synthetic teaching geometry—not a causal claim.*

![c350 teaching panel 19 (original).](../assets/figures/ml_fig_c350_19.png)
*Figure — Glossary genAI strip c350. Synthetic teaching geometry—not a causal claim.*

![c351 teaching panel 19 (original).](../assets/figures/ml_fig_c351_19.png)
*Figure — Glossary metric strip c351. Synthetic teaching geometry—not a causal claim.*

![c352 teaching panel 19 (original).](../assets/figures/ml_fig_c352_19.png)
*Figure — Glossary regularize strip c352. Synthetic teaching geometry—not a causal claim.*

![c353 teaching panel 19 (original).](../assets/figures/ml_fig_c353_19.png)
*Figure — Glossary embedding strip c353. Synthetic teaching geometry—not a causal claim.*

![c354 teaching panel 19 (original).](../assets/figures/ml_fig_c354_19.png)
*Figure — Glossary quant strip c354. Synthetic teaching geometry—not a causal claim.*

![c355 teaching panel 19 (original).](../assets/figures/ml_fig_c355_19.png)
*Figure — Glossary eval strip c355. Synthetic teaching geometry—not a causal claim.*

![c356 teaching panel 19 (original).](../assets/figures/ml_fig_c356_19.png)
*Figure — Glossary cluster strip c356. Synthetic teaching geometry—not a causal claim.*

![c357 teaching panel 19 (original).](../assets/figures/ml_fig_c357_19.png)
*Figure — Glossary IR strip c357. Synthetic teaching geometry—not a causal claim.*

![c358 teaching panel 19 (original).](../assets/figures/ml_fig_c358_19.png)
*Figure — Glossary deploy strip c358. Synthetic teaching geometry—not a causal claim.*

![c359 teaching panel 19 (original).](../assets/figures/ml_fig_c359_19.png)
*Figure — Glossary loss strip c359. Synthetic teaching geometry—not a causal claim.*

![c360 teaching panel 19 (original).](../assets/figures/ml_fig_c360_19.png)
*Figure — Glossary optim strip c360. Synthetic teaching geometry—not a causal claim.*

![c361 teaching panel 19 (original).](../assets/figures/ml_fig_c361_19.png)
*Figure — Glossary SSL strip c361. Synthetic teaching geometry—not a causal claim.*

![c362 teaching panel 19 (original).](../assets/figures/ml_fig_c362_19.png)
*Figure — Glossary RL strip c362. Synthetic teaching geometry—not a causal claim.*

![c363 teaching panel 19 (original).](../assets/figures/ml_fig_c363_19.png)
*Figure — Glossary privacy strip c363. Synthetic teaching geometry—not a causal claim.*

![c364 teaching panel 19 (original).](../assets/figures/ml_fig_c364_19.png)
*Figure — Glossary graph strip c364. Synthetic teaching geometry—not a causal claim.*

![c365 teaching panel 19 (original).](../assets/figures/ml_fig_c365_19.png)
*Figure — Glossary calib strip c365. Synthetic teaching geometry—not a causal claim.*

![c366 teaching panel 19 (original).](../assets/figures/ml_fig_c366_19.png)
*Figure — Glossary genAI strip c366. Synthetic teaching geometry—not a causal claim.*

![c367 teaching panel 19 (original).](../assets/figures/ml_fig_c367_19.png)
*Figure — Glossary metric strip c367. Synthetic teaching geometry—not a causal claim.*

![c368 teaching panel 19 (original).](../assets/figures/ml_fig_c368_19.png)
*Figure — Glossary regularize strip c368. Synthetic teaching geometry—not a causal claim.*

![c369 teaching panel 19 (original).](../assets/figures/ml_fig_c369_19.png)
*Figure — Glossary embedding strip c369. Synthetic teaching geometry—not a causal claim.*

![c370 teaching panel 19 (original).](../assets/figures/ml_fig_c370_19.png)
*Figure — Glossary quant strip c370. Synthetic teaching geometry—not a causal claim.*

![c371 teaching panel 19 (original).](../assets/figures/ml_fig_c371_19.png)
*Figure — Glossary eval strip c371. Synthetic teaching geometry—not a causal claim.*

![c372 teaching panel 19 (original).](../assets/figures/ml_fig_c372_19.png)
*Figure — Glossary cluster strip c372. Synthetic teaching geometry—not a causal claim.*

![c373 teaching panel 19 (original).](../assets/figures/ml_fig_c373_19.png)
*Figure — Glossary IR strip c373. Synthetic teaching geometry—not a causal claim.*

![c374 teaching panel 19 (original).](../assets/figures/ml_fig_c374_19.png)
*Figure — Glossary deploy strip c374. Synthetic teaching geometry—not a causal claim.*

![c375 teaching panel 19 (original).](../assets/figures/ml_fig_c375_19.png)
*Figure — Glossary loss strip c375. Synthetic teaching geometry—not a causal claim.*

![c376 teaching panel 19 (original).](../assets/figures/ml_fig_c376_19.png)
*Figure — Glossary optim strip c376. Synthetic teaching geometry—not a causal claim.*

![c377 teaching panel 19 (original).](../assets/figures/ml_fig_c377_19.png)
*Figure — Glossary SSL strip c377. Synthetic teaching geometry—not a causal claim.*

![c378 teaching panel 19 (original).](../assets/figures/ml_fig_c378_19.png)
*Figure — Glossary RL strip c378. Synthetic teaching geometry—not a causal claim.*

![c379 teaching panel 19 (original).](../assets/figures/ml_fig_c379_19.png)
*Figure — Glossary privacy strip c379. Synthetic teaching geometry—not a causal claim.*

![c380 teaching panel 19 (original).](../assets/figures/ml_fig_c380_19.png)
*Figure — Glossary graph strip c380. Synthetic teaching geometry—not a causal claim.*

![c381 teaching panel 19 (original).](../assets/figures/ml_fig_c381_19.png)
*Figure — Glossary calib strip c381. Synthetic teaching geometry—not a causal claim.*

![c382 teaching panel 19 (original).](../assets/figures/ml_fig_c382_19.png)
*Figure — Glossary genAI strip c382. Synthetic teaching geometry—not a causal claim.*

![c383 teaching panel 19 (original).](../assets/figures/ml_fig_c383_19.png)
*Figure — Glossary metric strip c383. Synthetic teaching geometry—not a causal claim.*

![c384 teaching panel 19 (original).](../assets/figures/ml_fig_c384_19.png)
*Figure — Glossary regularize strip c384. Synthetic teaching geometry—not a causal claim.*

![c385 teaching panel 19 (original).](../assets/figures/ml_fig_c385_19.png)
*Figure — Glossary embedding strip c385. Synthetic teaching geometry—not a causal claim.*

![c386 teaching panel 19 (original).](../assets/figures/ml_fig_c386_19.png)
*Figure — Glossary quant strip c386. Synthetic teaching geometry—not a causal claim.*

![c387 teaching panel 19 (original).](../assets/figures/ml_fig_c387_19.png)
*Figure — Glossary eval strip c387. Synthetic teaching geometry—not a causal claim.*

![c388 teaching panel 19 (original).](../assets/figures/ml_fig_c388_19.png)
*Figure — Glossary cluster strip c388. Synthetic teaching geometry—not a causal claim.*

![c389 teaching panel 19 (original).](../assets/figures/ml_fig_c389_19.png)
*Figure — Glossary IR strip c389. Synthetic teaching geometry—not a causal claim.*

![c390 teaching panel 19 (original).](../assets/figures/ml_fig_c390_19.png)
*Figure — Glossary deploy strip c390. Synthetic teaching geometry—not a causal claim.*

![c391 teaching panel 19 (original).](../assets/figures/ml_fig_c391_19.png)
*Figure — Glossary loss strip c391. Synthetic teaching geometry—not a causal claim.*

![c392 teaching panel 19 (original).](../assets/figures/ml_fig_c392_19.png)
*Figure — Glossary optim strip c392. Synthetic teaching geometry—not a causal claim.*

![c393 teaching panel 19 (original).](../assets/figures/ml_fig_c393_19.png)
*Figure — Glossary SSL strip c393. Synthetic teaching geometry—not a causal claim.*

![c394 teaching panel 19 (original).](../assets/figures/ml_fig_c394_19.png)
*Figure — Glossary RL strip c394. Synthetic teaching geometry—not a causal claim.*

![c395 teaching panel 19 (original).](../assets/figures/ml_fig_c395_19.png)
*Figure — Glossary privacy strip c395. Synthetic teaching geometry—not a causal claim.*

![c396 teaching panel 19 (original).](../assets/figures/ml_fig_c396_19.png)
*Figure — Glossary graph strip c396. Synthetic teaching geometry—not a causal claim.*

![c397 teaching panel 19 (original).](../assets/figures/ml_fig_c397_19.png)
*Figure — Glossary calib strip c397. Synthetic teaching geometry—not a causal claim.*

![c398 teaching panel 19 (original).](../assets/figures/ml_fig_c398_19.png)
*Figure — Glossary genAI strip c398. Synthetic teaching geometry—not a causal claim.*

![c399 teaching panel 19 (original).](../assets/figures/ml_fig_c399_19.png)
*Figure — Glossary metric strip c399. Synthetic teaching geometry—not a causal claim.*

![c400 teaching panel 19 (original).](../assets/figures/ml_fig_c400_19.png)
*Figure — Glossary regularize strip c400. Synthetic teaching geometry—not a causal claim.*

![c401 teaching panel 19 (original).](../assets/figures/ml_fig_c401_19.png)
*Figure — Glossary embedding strip c401. Synthetic teaching geometry—not a causal claim.*

![c402 teaching panel 19 (original).](../assets/figures/ml_fig_c402_19.png)
*Figure — Glossary quant strip c402. Synthetic teaching geometry—not a causal claim.*

![c403 teaching panel 19 (original).](../assets/figures/ml_fig_c403_19.png)
*Figure — Glossary eval strip c403. Synthetic teaching geometry—not a causal claim.*

![c404 teaching panel 19 (original).](../assets/figures/ml_fig_c404_19.png)
*Figure — Glossary cluster strip c404. Synthetic teaching geometry—not a causal claim.*

![c405 teaching panel 19 (original).](../assets/figures/ml_fig_c405_19.png)
*Figure — Glossary IR strip c405. Synthetic teaching geometry—not a causal claim.*

![c406 teaching panel 19 (original).](../assets/figures/ml_fig_c406_19.png)
*Figure — Glossary deploy strip c406. Synthetic teaching geometry—not a causal claim.*

![c407 teaching panel 19 (original).](../assets/figures/ml_fig_c407_19.png)
*Figure — Glossary loss strip c407. Synthetic teaching geometry—not a causal claim.*

![c408 teaching panel 19 (original).](../assets/figures/ml_fig_c408_19.png)
*Figure — Glossary optim strip c408. Synthetic teaching geometry—not a causal claim.*

![c409 teaching panel 19 (original).](../assets/figures/ml_fig_c409_19.png)
*Figure — Glossary SSL strip c409. Synthetic teaching geometry—not a causal claim.*

![c410 teaching panel 19 (original).](../assets/figures/ml_fig_c410_19.png)
*Figure — Glossary RL strip c410. Synthetic teaching geometry—not a causal claim.*

![c411 teaching panel 19 (original).](../assets/figures/ml_fig_c411_19.png)
*Figure — Glossary privacy strip c411. Synthetic teaching geometry—not a causal claim.*

![c412 teaching panel 19 (original).](../assets/figures/ml_fig_c412_19.png)
*Figure — Glossary graph strip c412. Synthetic teaching geometry—not a causal claim.*

![c413 teaching panel 19 (original).](../assets/figures/ml_fig_c413_19.png)
*Figure — Glossary calib strip c413. Synthetic teaching geometry—not a causal claim.*

![c414 teaching panel 19 (original).](../assets/figures/ml_fig_c414_19.png)
*Figure — Glossary genAI strip c414. Synthetic teaching geometry—not a causal claim.*

![c415 teaching panel 19 (original).](../assets/figures/ml_fig_c415_19.png)
*Figure — Glossary metric strip c415. Synthetic teaching geometry—not a causal claim.*

![c416 teaching panel 19 (original).](../assets/figures/ml_fig_c416_19.png)
*Figure — Glossary regularize strip c416. Synthetic teaching geometry—not a causal claim.*

![c417 teaching panel 19 (original).](../assets/figures/ml_fig_c417_19.png)
*Figure — Glossary embedding strip c417. Synthetic teaching geometry—not a causal claim.*

![c418 teaching panel 19 (original).](../assets/figures/ml_fig_c418_19.png)
*Figure — Glossary quant strip c418. Synthetic teaching geometry—not a causal claim.*

![c419 teaching panel 19 (original).](../assets/figures/ml_fig_c419_19.png)
*Figure — Glossary eval strip c419. Synthetic teaching geometry—not a causal claim.*

![c420 teaching panel 19 (original).](../assets/figures/ml_fig_c420_19.png)
*Figure — Glossary cluster strip c420. Synthetic teaching geometry—not a causal claim.*

![c421 teaching panel 19 (original).](../assets/figures/ml_fig_c421_19.png)
*Figure — Glossary IR strip c421. Synthetic teaching geometry—not a causal claim.*

![c422 teaching panel 19 (original).](../assets/figures/ml_fig_c422_19.png)
*Figure — Glossary deploy strip c422. Synthetic teaching geometry—not a causal claim.*

![c423 teaching panel 19 (original).](../assets/figures/ml_fig_c423_19.png)
*Figure — Glossary loss strip c423. Synthetic teaching geometry—not a causal claim.*

![c424 teaching panel 19 (original).](../assets/figures/ml_fig_c424_19.png)
*Figure — Glossary optim strip c424. Synthetic teaching geometry—not a causal claim.*

![c425 teaching panel 19 (original).](../assets/figures/ml_fig_c425_19.png)
*Figure — Glossary SSL strip c425. Synthetic teaching geometry—not a causal claim.*

![c426 teaching panel 19 (original).](../assets/figures/ml_fig_c426_19.png)
*Figure — Glossary RL strip c426. Synthetic teaching geometry—not a causal claim.*

![c427 teaching panel 19 (original).](../assets/figures/ml_fig_c427_19.png)
*Figure — Glossary privacy strip c427. Synthetic teaching geometry—not a causal claim.*

![c428 teaching panel 19 (original).](../assets/figures/ml_fig_c428_19.png)
*Figure — Glossary graph strip c428. Synthetic teaching geometry—not a causal claim.*

![c429 teaching panel 19 (original).](../assets/figures/ml_fig_c429_19.png)
*Figure — Glossary calib strip c429. Synthetic teaching geometry—not a causal claim.*

![c430 teaching panel 19 (original).](../assets/figures/ml_fig_c430_19.png)
*Figure — Glossary genAI strip c430. Synthetic teaching geometry—not a causal claim.*

![c431 teaching panel 19 (original).](../assets/figures/ml_fig_c431_19.png)
*Figure — Glossary metric strip c431. Synthetic teaching geometry—not a causal claim.*

![c432 teaching panel 19 (original).](../assets/figures/ml_fig_c432_19.png)
*Figure — Glossary regularize strip c432. Synthetic teaching geometry—not a causal claim.*

![c433 teaching panel 19 (original).](../assets/figures/ml_fig_c433_19.png)
*Figure — Glossary embedding strip c433. Synthetic teaching geometry—not a causal claim.*

![c434 teaching panel 19 (original).](../assets/figures/ml_fig_c434_19.png)
*Figure — Glossary quant strip c434. Synthetic teaching geometry—not a causal claim.*

![c435 teaching panel 19 (original).](../assets/figures/ml_fig_c435_19.png)
*Figure — Glossary eval strip c435. Synthetic teaching geometry—not a causal claim.*

![c436 teaching panel 19 (original).](../assets/figures/ml_fig_c436_19.png)
*Figure — Glossary cluster strip c436. Synthetic teaching geometry—not a causal claim.*

![c437 teaching panel 19 (original).](../assets/figures/ml_fig_c437_19.png)
*Figure — Glossary IR strip c437. Synthetic teaching geometry—not a causal claim.*

![c438 teaching panel 19 (original).](../assets/figures/ml_fig_c438_19.png)
*Figure — Glossary deploy strip c438. Synthetic teaching geometry—not a causal claim.*

![c439 teaching panel 19 (original).](../assets/figures/ml_fig_c439_19.png)
*Figure — Glossary loss strip c439. Synthetic teaching geometry—not a causal claim.*

![c440 teaching panel 19 (original).](../assets/figures/ml_fig_c440_19.png)
*Figure — Glossary optim strip c440. Synthetic teaching geometry—not a causal claim.*

![c441 teaching panel 19 (original).](../assets/figures/ml_fig_c441_19.png)
*Figure — Glossary SSL strip c441. Synthetic teaching geometry—not a causal claim.*

![c442 teaching panel 19 (original).](../assets/figures/ml_fig_c442_19.png)
*Figure — Glossary RL strip c442. Synthetic teaching geometry—not a causal claim.*

![c443 teaching panel 19 (original).](../assets/figures/ml_fig_c443_19.png)
*Figure — Glossary privacy strip c443. Synthetic teaching geometry—not a causal claim.*

![c444 teaching panel 19 (original).](../assets/figures/ml_fig_c444_19.png)
*Figure — Glossary graph strip c444. Synthetic teaching geometry—not a causal claim.*

![c445 teaching panel 19 (original).](../assets/figures/ml_fig_c445_19.png)
*Figure — Glossary calib strip c445. Synthetic teaching geometry—not a causal claim.*

![c446 teaching panel 19 (original).](../assets/figures/ml_fig_c446_19.png)
*Figure — Glossary genAI strip c446. Synthetic teaching geometry—not a causal claim.*

![c447 teaching panel 19 (original).](../assets/figures/ml_fig_c447_19.png)
*Figure — Glossary metric strip c447. Synthetic teaching geometry—not a causal claim.*

![c448 teaching panel 19 (original).](../assets/figures/ml_fig_c448_19.png)
*Figure — Glossary regularize strip c448. Synthetic teaching geometry—not a causal claim.*

![c449 teaching panel 19 (original).](../assets/figures/ml_fig_c449_19.png)
*Figure — Glossary embedding strip c449. Synthetic teaching geometry—not a causal claim.*

![c450 teaching panel 19 (original).](../assets/figures/ml_fig_c450_19.png)
*Figure — Glossary quant strip c450. Synthetic teaching geometry—not a causal claim.*

![c451 teaching panel 19 (original).](../assets/figures/ml_fig_c451_19.png)
*Figure — Glossary eval strip c451. Synthetic teaching geometry—not a causal claim.*

![c452 teaching panel 19 (original).](../assets/figures/ml_fig_c452_19.png)
*Figure — Glossary cluster strip c452. Synthetic teaching geometry—not a causal claim.*

![c453 teaching panel 19 (original).](../assets/figures/ml_fig_c453_19.png)
*Figure — Glossary IR strip c453. Synthetic teaching geometry—not a causal claim.*

![c454 teaching panel 19 (original).](../assets/figures/ml_fig_c454_19.png)
*Figure — Glossary deploy strip c454. Synthetic teaching geometry—not a causal claim.*

![c455 teaching panel 19 (original).](../assets/figures/ml_fig_c455_19.png)
*Figure — Glossary loss strip c455. Synthetic teaching geometry—not a causal claim.*

![c456 teaching panel 19 (original).](../assets/figures/ml_fig_c456_19.png)
*Figure — Glossary optim strip c456. Synthetic teaching geometry—not a causal claim.*

![c457 teaching panel 19 (original).](../assets/figures/ml_fig_c457_19.png)
*Figure — Glossary SSL strip c457. Synthetic teaching geometry—not a causal claim.*

![c458 teaching panel 19 (original).](../assets/figures/ml_fig_c458_19.png)
*Figure — Glossary RL strip c458. Synthetic teaching geometry—not a causal claim.*

![c459 teaching panel 19 (original).](../assets/figures/ml_fig_c459_19.png)
*Figure — Glossary privacy strip c459. Synthetic teaching geometry—not a causal claim.*

![c460 teaching panel 19 (original).](../assets/figures/ml_fig_c460_19.png)
*Figure — Glossary graph strip c460. Synthetic teaching geometry—not a causal claim.*

![c461 teaching panel 19 (original).](../assets/figures/ml_fig_c461_19.png)
*Figure — Glossary calib strip c461. Synthetic teaching geometry—not a causal claim.*

![c462 teaching panel 19 (original).](../assets/figures/ml_fig_c462_19.png)
*Figure — Glossary genAI strip c462. Synthetic teaching geometry—not a causal claim.*

![c463 teaching panel 19 (original).](../assets/figures/ml_fig_c463_19.png)
*Figure — Glossary metric strip c463. Synthetic teaching geometry—not a causal claim.*

![c464 teaching panel 19 (original).](../assets/figures/ml_fig_c464_19.png)
*Figure — Glossary regularize strip c464. Synthetic teaching geometry—not a causal claim.*

![c465 teaching panel 19 (original).](../assets/figures/ml_fig_c465_19.png)
*Figure — Glossary embedding strip c465. Synthetic teaching geometry—not a causal claim.*

![c466 teaching panel 19 (original).](../assets/figures/ml_fig_c466_19.png)
*Figure — Glossary quant strip c466. Synthetic teaching geometry—not a causal claim.*

![c467 teaching panel 19 (original).](../assets/figures/ml_fig_c467_19.png)
*Figure — Glossary eval strip c467. Synthetic teaching geometry—not a causal claim.*

![c468 teaching panel 19 (original).](../assets/figures/ml_fig_c468_19.png)
*Figure — Glossary cluster strip c468. Synthetic teaching geometry—not a causal claim.*

![c469 teaching panel 19 (original).](../assets/figures/ml_fig_c469_19.png)
*Figure — Glossary IR strip c469. Synthetic teaching geometry—not a causal claim.*

![c470 teaching panel 19 (original).](../assets/figures/ml_fig_c470_19.png)
*Figure — Glossary deploy strip c470. Synthetic teaching geometry—not a causal claim.*

![c471 teaching panel 19 (original).](../assets/figures/ml_fig_c471_19.png)
*Figure — Glossary loss strip c471. Synthetic teaching geometry—not a causal claim.*

![c472 teaching panel 19 (original).](../assets/figures/ml_fig_c472_19.png)
*Figure — Glossary optim strip c472. Synthetic teaching geometry—not a causal claim.*

![c473 teaching panel 19 (original).](../assets/figures/ml_fig_c473_19.png)
*Figure — Glossary SSL strip c473. Synthetic teaching geometry—not a causal claim.*

![c474 teaching panel 19 (original).](../assets/figures/ml_fig_c474_19.png)
*Figure — Glossary RL strip c474. Synthetic teaching geometry—not a causal claim.*

![c475 teaching panel 19 (original).](../assets/figures/ml_fig_c475_19.png)
*Figure — Glossary privacy strip c475. Synthetic teaching geometry—not a causal claim.*

![c476 teaching panel 19 (original).](../assets/figures/ml_fig_c476_19.png)
*Figure — Glossary graph strip c476. Synthetic teaching geometry—not a causal claim.*

![c477 teaching panel 19 (original).](../assets/figures/ml_fig_c477_19.png)
*Figure — Glossary calib strip c477. Synthetic teaching geometry—not a causal claim.*

![c478 teaching panel 19 (original).](../assets/figures/ml_fig_c478_19.png)
*Figure — Glossary genAI strip c478. Synthetic teaching geometry—not a causal claim.*

![c479 teaching panel 19 (original).](../assets/figures/ml_fig_c479_19.png)
*Figure — Glossary metric strip c479. Synthetic teaching geometry—not a causal claim.*

![c480 teaching panel 19 (original).](../assets/figures/ml_fig_c480_19.png)
*Figure — Glossary regularize strip c480. Synthetic teaching geometry—not a causal claim.*

![c481 teaching panel 19 (original).](../assets/figures/ml_fig_c481_19.png)
*Figure — Glossary embedding strip c481. Synthetic teaching geometry—not a causal claim.*

![c482 teaching panel 19 (original).](../assets/figures/ml_fig_c482_19.png)
*Figure — Glossary quant strip c482. Synthetic teaching geometry—not a causal claim.*

![c483 teaching panel 19 (original).](../assets/figures/ml_fig_c483_19.png)
*Figure — Glossary eval strip c483. Synthetic teaching geometry—not a causal claim.*

![c484 teaching panel 19 (original).](../assets/figures/ml_fig_c484_19.png)
*Figure — Glossary cluster strip c484. Synthetic teaching geometry—not a causal claim.*

![c485 teaching panel 19 (original).](../assets/figures/ml_fig_c485_19.png)
*Figure — Glossary IR strip c485. Synthetic teaching geometry—not a causal claim.*

![c486 teaching panel 19 (original).](../assets/figures/ml_fig_c486_19.png)
*Figure — Glossary deploy strip c486. Synthetic teaching geometry—not a causal claim.*

![c487 teaching panel 19 (original).](../assets/figures/ml_fig_c487_19.png)
*Figure — Glossary loss strip c487. Synthetic teaching geometry—not a causal claim.*

![c488 teaching panel 19 (original).](../assets/figures/ml_fig_c488_19.png)
*Figure — Glossary optim strip c488. Synthetic teaching geometry—not a causal claim.*

![c489 teaching panel 19 (original).](../assets/figures/ml_fig_c489_19.png)
*Figure — Glossary SSL strip c489. Synthetic teaching geometry—not a causal claim.*

![c490 teaching panel 19 (original).](../assets/figures/ml_fig_c490_19.png)
*Figure — Glossary RL strip c490. Synthetic teaching geometry—not a causal claim.*

![c491 teaching panel 19 (original).](../assets/figures/ml_fig_c491_19.png)
*Figure — Glossary privacy strip c491. Synthetic teaching geometry—not a causal claim.*

![c492 teaching panel 19 (original).](../assets/figures/ml_fig_c492_19.png)
*Figure — Glossary graph strip c492. Synthetic teaching geometry—not a causal claim.*

![c493 teaching panel 19 (original).](../assets/figures/ml_fig_c493_19.png)
*Figure — Glossary calib strip c493. Synthetic teaching geometry—not a causal claim.*

![c494 teaching panel 19 (original).](../assets/figures/ml_fig_c494_19.png)
*Figure — Glossary genAI strip c494. Synthetic teaching geometry—not a causal claim.*

![c495 teaching panel 19 (original).](../assets/figures/ml_fig_c495_19.png)
*Figure — Glossary metric strip c495. Synthetic teaching geometry—not a causal claim.*

![c496 teaching panel 19 (original).](../assets/figures/ml_fig_c496_19.png)
*Figure — Glossary regularize strip c496. Synthetic teaching geometry—not a causal claim.*

![c497 teaching panel 19 (original).](../assets/figures/ml_fig_c497_19.png)
*Figure — Glossary embedding strip c497. Synthetic teaching geometry—not a causal claim.*

![c498 teaching panel 19 (original).](../assets/figures/ml_fig_c498_19.png)
*Figure — Glossary quant strip c498. Synthetic teaching geometry—not a causal claim.*

![c499 teaching panel 19 (original).](../assets/figures/ml_fig_c499_19.png)
*Figure — Glossary eval strip c499. Synthetic teaching geometry—not a causal claim.*

![c500 teaching panel 19 (original).](../assets/figures/ml_fig_c500_19.png)
*Figure — Glossary cluster strip c500. Synthetic teaching geometry—not a causal claim.*

![c501 teaching panel 19 (original).](../assets/figures/ml_fig_c501_19.png)
*Figure — Glossary IR strip c501. Synthetic teaching geometry—not a causal claim.*

![c502 teaching panel 19 (original).](../assets/figures/ml_fig_c502_19.png)
*Figure — Glossary deploy strip c502. Synthetic teaching geometry—not a causal claim.*

![c503 teaching panel 19 (original).](../assets/figures/ml_fig_c503_19.png)
*Figure — Glossary loss strip c503. Synthetic teaching geometry—not a causal claim.*

![c504 teaching panel 19 (original).](../assets/figures/ml_fig_c504_19.png)
*Figure — Glossary optim strip c504. Synthetic teaching geometry—not a causal claim.*

![c505 teaching panel 19 (original).](../assets/figures/ml_fig_c505_19.png)
*Figure — Glossary SSL strip c505. Synthetic teaching geometry—not a causal claim.*

![c506 teaching panel 19 (original).](../assets/figures/ml_fig_c506_19.png)
*Figure — Glossary RL strip c506. Synthetic teaching geometry—not a causal claim.*

![c507 teaching panel 19 (original).](../assets/figures/ml_fig_c507_19.png)
*Figure — Glossary privacy strip c507. Synthetic teaching geometry—not a causal claim.*

![c508 teaching panel 19 (original).](../assets/figures/ml_fig_c508_19.png)
*Figure — Glossary graph strip c508. Synthetic teaching geometry—not a causal claim.*

![c509 teaching panel 19 (original).](../assets/figures/ml_fig_c509_19.png)
*Figure — Glossary calib strip c509. Synthetic teaching geometry—not a causal claim.*

![c510 teaching panel 19 (original).](../assets/figures/ml_fig_c510_19.png)
*Figure — Glossary genAI strip c510. Synthetic teaching geometry—not a causal claim.*

![c511 teaching panel 19 (original).](../assets/figures/ml_fig_c511_19.png)
*Figure — Glossary metric strip c511. Synthetic teaching geometry—not a causal claim.*

![c512 teaching panel 19 (original).](../assets/figures/ml_fig_c512_19.png)
*Figure — Glossary regularize strip c512. Synthetic teaching geometry—not a causal claim.*

![c513 teaching panel 19 (original).](../assets/figures/ml_fig_c513_19.png)
*Figure — Glossary embedding strip c513. Synthetic teaching geometry—not a causal claim.*

![c514 teaching panel 19 (original).](../assets/figures/ml_fig_c514_19.png)
*Figure — Glossary quant strip c514. Synthetic teaching geometry—not a causal claim.*

![c515 teaching panel 19 (original).](../assets/figures/ml_fig_c515_19.png)
*Figure — Glossary eval strip c515. Synthetic teaching geometry—not a causal claim.*

![c516 teaching panel 19 (original).](../assets/figures/ml_fig_c516_19.png)
*Figure — Glossary cluster strip c516. Synthetic teaching geometry—not a causal claim.*

![c517 teaching panel 19 (original).](../assets/figures/ml_fig_c517_19.png)
*Figure — Glossary IR strip c517. Synthetic teaching geometry—not a causal claim.*

![c518 teaching panel 19 (original).](../assets/figures/ml_fig_c518_19.png)
*Figure — Glossary deploy strip c518. Synthetic teaching geometry—not a causal claim.*

![c519 teaching panel 19 (original).](../assets/figures/ml_fig_c519_19.png)
*Figure — Glossary loss strip c519. Synthetic teaching geometry—not a causal claim.*

![c520 teaching panel 19 (original).](../assets/figures/ml_fig_c520_19.png)
*Figure — Glossary optim strip c520. Synthetic teaching geometry—not a causal claim.*

![c521 teaching panel 19 (original).](../assets/figures/ml_fig_c521_19.png)
*Figure — Glossary SSL strip c521. Synthetic teaching geometry—not a causal claim.*

![c522 teaching panel 19 (original).](../assets/figures/ml_fig_c522_19.png)
*Figure — Glossary RL strip c522. Synthetic teaching geometry—not a causal claim.*

![c523 teaching panel 19 (original).](../assets/figures/ml_fig_c523_19.png)
*Figure — Glossary privacy strip c523. Synthetic teaching geometry—not a causal claim.*

![c524 teaching panel 19 (original).](../assets/figures/ml_fig_c524_19.png)
*Figure — Glossary graph strip c524. Synthetic teaching geometry—not a causal claim.*

![c525 teaching panel 19 (original).](../assets/figures/ml_fig_c525_19.png)
*Figure — Glossary calib strip c525. Synthetic teaching geometry—not a causal claim.*

![c526 teaching panel 19 (original).](../assets/figures/ml_fig_c526_19.png)
*Figure — Glossary genAI strip c526. Synthetic teaching geometry—not a causal claim.*

![c527 teaching panel 19 (original).](../assets/figures/ml_fig_c527_19.png)
*Figure — Glossary metric strip c527. Synthetic teaching geometry—not a causal claim.*

![c528 teaching panel 19 (original).](../assets/figures/ml_fig_c528_19.png)
*Figure — Glossary regularize strip c528. Synthetic teaching geometry—not a causal claim.*

![c529 teaching panel 19 (original).](../assets/figures/ml_fig_c529_19.png)
*Figure — Glossary embedding strip c529. Synthetic teaching geometry—not a causal claim.*

![c530 teaching panel 19 (original).](../assets/figures/ml_fig_c530_19.png)
*Figure — Glossary quant strip c530. Synthetic teaching geometry—not a causal claim.*

![c531 teaching panel 19 (original).](../assets/figures/ml_fig_c531_19.png)
*Figure — Glossary eval strip c531. Synthetic teaching geometry—not a causal claim.*

![c532 teaching panel 19 (original).](../assets/figures/ml_fig_c532_19.png)
*Figure — Glossary cluster strip c532. Synthetic teaching geometry—not a causal claim.*

![c533 teaching panel 19 (original).](../assets/figures/ml_fig_c533_19.png)
*Figure — Glossary IR strip c533. Synthetic teaching geometry—not a causal claim.*

![c534 teaching panel 19 (original).](../assets/figures/ml_fig_c534_19.png)
*Figure — Glossary deploy strip c534. Synthetic teaching geometry—not a causal claim.*

![c535 teaching panel 19 (original).](../assets/figures/ml_fig_c535_19.png)
*Figure — Glossary loss strip c535. Synthetic teaching geometry—not a causal claim.*

![c536 teaching panel 19 (original).](../assets/figures/ml_fig_c536_19.png)
*Figure — Glossary optim strip c536. Synthetic teaching geometry—not a causal claim.*

![c537 teaching panel 19 (original).](../assets/figures/ml_fig_c537_19.png)
*Figure — Glossary SSL strip c537. Synthetic teaching geometry—not a causal claim.*

![c538 teaching panel 19 (original).](../assets/figures/ml_fig_c538_19.png)
*Figure — Glossary RL strip c538. Synthetic teaching geometry—not a causal claim.*

![c539 teaching panel 19 (original).](../assets/figures/ml_fig_c539_19.png)
*Figure — Glossary privacy strip c539. Synthetic teaching geometry—not a causal claim.*

![c540 teaching panel 19 (original).](../assets/figures/ml_fig_c540_19.png)
*Figure — Glossary graph strip c540. Synthetic teaching geometry—not a causal claim.*

![c541 teaching panel 19 (original).](../assets/figures/ml_fig_c541_19.png)
*Figure — Glossary calib strip c541. Synthetic teaching geometry—not a causal claim.*

![c542 teaching panel 19 (original).](../assets/figures/ml_fig_c542_19.png)
*Figure — Glossary genAI strip c542. Synthetic teaching geometry—not a causal claim.*

![c543 teaching panel 19 (original).](../assets/figures/ml_fig_c543_19.png)
*Figure — Glossary metric strip c543. Synthetic teaching geometry—not a causal claim.*

![c544 teaching panel 19 (original).](../assets/figures/ml_fig_c544_19.png)
*Figure — Glossary regularize strip c544. Synthetic teaching geometry—not a causal claim.*

![c545 teaching panel 19 (original).](../assets/figures/ml_fig_c545_19.png)
*Figure — Glossary embedding strip c545. Synthetic teaching geometry—not a causal claim.*

![c546 teaching panel 19 (original).](../assets/figures/ml_fig_c546_19.png)
*Figure — Glossary quant strip c546. Synthetic teaching geometry—not a causal claim.*

![c547 teaching panel 19 (original).](../assets/figures/ml_fig_c547_19.png)
*Figure — Glossary eval strip c547. Synthetic teaching geometry—not a causal claim.*

![c548 teaching panel 19 (original).](../assets/figures/ml_fig_c548_19.png)
*Figure — Glossary cluster strip c548. Synthetic teaching geometry—not a causal claim.*

![c549 teaching panel 19 (original).](../assets/figures/ml_fig_c549_19.png)
*Figure — Glossary IR strip c549. Synthetic teaching geometry—not a causal claim.*

![c550 teaching panel 19 (original).](../assets/figures/ml_fig_c550_19.png)
*Figure — Glossary deploy strip c550. Synthetic teaching geometry—not a causal claim.*

![c551 teaching panel 19 (original).](../assets/figures/ml_fig_c551_19.png)
*Figure — Glossary loss strip c551. Synthetic teaching geometry—not a causal claim.*

![c552 teaching panel 19 (original).](../assets/figures/ml_fig_c552_19.png)
*Figure — Glossary optim strip c552. Synthetic teaching geometry—not a causal claim.*

![c553 teaching panel 19 (original).](../assets/figures/ml_fig_c553_19.png)
*Figure — Glossary SSL strip c553. Synthetic teaching geometry—not a causal claim.*

![c554 teaching panel 19 (original).](../assets/figures/ml_fig_c554_19.png)
*Figure — Glossary RL strip c554. Synthetic teaching geometry—not a causal claim.*

![c555 teaching panel 19 (original).](../assets/figures/ml_fig_c555_19.png)
*Figure — Glossary privacy strip c555. Synthetic teaching geometry—not a causal claim.*

![c556 teaching panel 19 (original).](../assets/figures/ml_fig_c556_19.png)
*Figure — Glossary graph strip c556. Synthetic teaching geometry—not a causal claim.*

![c557 teaching panel 19 (original).](../assets/figures/ml_fig_c557_19.png)
*Figure — Glossary calib strip c557. Synthetic teaching geometry—not a causal claim.*

![c558 teaching panel 19 (original).](../assets/figures/ml_fig_c558_19.png)
*Figure — Glossary genAI strip c558. Synthetic teaching geometry—not a causal claim.*

![c559 teaching panel 19 (original).](../assets/figures/ml_fig_c559_19.png)
*Figure — Glossary metric strip c559. Synthetic teaching geometry—not a causal claim.*

![c560 teaching panel 19 (original).](../assets/figures/ml_fig_c560_19.png)
*Figure — Glossary regularize strip c560. Synthetic teaching geometry—not a causal claim.*

![c561 teaching panel 19 (original).](../assets/figures/ml_fig_c561_19.png)
*Figure — Glossary embedding strip c561. Synthetic teaching geometry—not a causal claim.*

![c562 teaching panel 19 (original).](../assets/figures/ml_fig_c562_19.png)
*Figure — Glossary quant strip c562. Synthetic teaching geometry—not a causal claim.*

![c563 teaching panel 19 (original).](../assets/figures/ml_fig_c563_19.png)
*Figure — Glossary eval strip c563. Synthetic teaching geometry—not a causal claim.*

![c564 teaching panel 19 (original).](../assets/figures/ml_fig_c564_19.png)
*Figure — Glossary cluster strip c564. Synthetic teaching geometry—not a causal claim.*

![c565 teaching panel 19 (original).](../assets/figures/ml_fig_c565_19.png)
*Figure — Glossary IR strip c565. Synthetic teaching geometry—not a causal claim.*

![c566 teaching panel 19 (original).](../assets/figures/ml_fig_c566_19.png)
*Figure — Glossary deploy strip c566. Synthetic teaching geometry—not a causal claim.*

![c567 teaching panel 19 (original).](../assets/figures/ml_fig_c567_19.png)
*Figure — Glossary loss strip c567. Synthetic teaching geometry—not a causal claim.*

![c568 teaching panel 19 (original).](../assets/figures/ml_fig_c568_19.png)
*Figure — Glossary optim strip c568. Synthetic teaching geometry—not a causal claim.*

![c569 teaching panel 19 (original).](../assets/figures/ml_fig_c569_19.png)
*Figure — Glossary SSL strip c569. Synthetic teaching geometry—not a causal claim.*

![c570 teaching panel 19 (original).](../assets/figures/ml_fig_c570_19.png)
*Figure — Glossary RL strip c570. Synthetic teaching geometry—not a causal claim.*

![c571 teaching panel 19 (original).](../assets/figures/ml_fig_c571_19.png)
*Figure — Glossary privacy strip c571. Synthetic teaching geometry—not a causal claim.*

![c572 teaching panel 19 (original).](../assets/figures/ml_fig_c572_19.png)
*Figure — Glossary graph strip c572. Synthetic teaching geometry—not a causal claim.*

![c573 teaching panel 19 (original).](../assets/figures/ml_fig_c573_19.png)
*Figure — Glossary calib strip c573. Synthetic teaching geometry—not a causal claim.*

![c574 teaching panel 19 (original).](../assets/figures/ml_fig_c574_19.png)
*Figure — Glossary genAI strip c574. Synthetic teaching geometry—not a causal claim.*

![c575 teaching panel 19 (original).](../assets/figures/ml_fig_c575_19.png)
*Figure — Glossary metric strip c575. Synthetic teaching geometry—not a causal claim.*

![c576 teaching panel 19 (original).](../assets/figures/ml_fig_c576_19.png)
*Figure — Glossary regularize strip c576. Synthetic teaching geometry—not a causal claim.*

![c577 teaching panel 19 (original).](../assets/figures/ml_fig_c577_19.png)
*Figure — Glossary embedding strip c577. Synthetic teaching geometry—not a causal claim.*

![c578 teaching panel 19 (original).](../assets/figures/ml_fig_c578_19.png)
*Figure — Glossary quant strip c578. Synthetic teaching geometry—not a causal claim.*

![c579 teaching panel 19 (original).](../assets/figures/ml_fig_c579_19.png)
*Figure — Glossary eval strip c579. Synthetic teaching geometry—not a causal claim.*

![c580 teaching panel 19 (original).](../assets/figures/ml_fig_c580_19.png)
*Figure — Glossary cluster strip c580. Synthetic teaching geometry—not a causal claim.*

![c581 teaching panel 19 (original).](../assets/figures/ml_fig_c581_19.png)
*Figure — Glossary IR strip c581. Synthetic teaching geometry—not a causal claim.*

![c582 teaching panel 19 (original).](../assets/figures/ml_fig_c582_19.png)
*Figure — Glossary deploy strip c582. Synthetic teaching geometry—not a causal claim.*

![c583 teaching panel 19 (original).](../assets/figures/ml_fig_c583_19.png)
*Figure — Glossary loss strip c583. Synthetic teaching geometry—not a causal claim.*

![c584 teaching panel 19 (original).](../assets/figures/ml_fig_c584_19.png)
*Figure — Glossary optim strip c584. Synthetic teaching geometry—not a causal claim.*

![c585 teaching panel 19 (original).](../assets/figures/ml_fig_c585_19.png)
*Figure — Glossary SSL strip c585. Synthetic teaching geometry—not a causal claim.*

![c586 teaching panel 19 (original).](../assets/figures/ml_fig_c586_19.png)
*Figure — Glossary RL strip c586. Synthetic teaching geometry—not a causal claim.*

![c587 teaching panel 19 (original).](../assets/figures/ml_fig_c587_19.png)
*Figure — Glossary privacy strip c587. Synthetic teaching geometry—not a causal claim.*

![c588 teaching panel 19 (original).](../assets/figures/ml_fig_c588_19.png)
*Figure — Glossary graph strip c588. Synthetic teaching geometry—not a causal claim.*

![c589 teaching panel 19 (original).](../assets/figures/ml_fig_c589_19.png)
*Figure — Glossary calib strip c589. Synthetic teaching geometry—not a causal claim.*

![c590 teaching panel 19 (original).](../assets/figures/ml_fig_c590_19.png)
*Figure — Glossary genAI strip c590. Synthetic teaching geometry—not a causal claim.*

![c591 teaching panel 19 (original).](../assets/figures/ml_fig_c591_19.png)
*Figure — Glossary metric strip c591. Synthetic teaching geometry—not a causal claim.*

![c592 teaching panel 19 (original).](../assets/figures/ml_fig_c592_19.png)
*Figure — Glossary regularize strip c592. Synthetic teaching geometry—not a causal claim.*

![c593 teaching panel 19 (original).](../assets/figures/ml_fig_c593_19.png)
*Figure — Glossary embedding strip c593. Synthetic teaching geometry—not a causal claim.*

![c594 teaching panel 19 (original).](../assets/figures/ml_fig_c594_19.png)
*Figure — Glossary quant strip c594. Synthetic teaching geometry—not a causal claim.*

![c595 teaching panel 19 (original).](../assets/figures/ml_fig_c595_19.png)
*Figure — Glossary eval strip c595. Synthetic teaching geometry—not a causal claim.*

![c596 teaching panel 19 (original).](../assets/figures/ml_fig_c596_19.png)
*Figure — Glossary cluster strip c596. Synthetic teaching geometry—not a causal claim.*

![c597 teaching panel 19 (original).](../assets/figures/ml_fig_c597_19.png)
*Figure — Glossary IR strip c597. Synthetic teaching geometry—not a causal claim.*

![c598 teaching panel 19 (original).](../assets/figures/ml_fig_c598_19.png)
*Figure — Glossary deploy strip c598. Synthetic teaching geometry—not a causal claim.*

![c599 teaching panel 19 (original).](../assets/figures/ml_fig_c599_19.png)
*Figure — Glossary loss strip c599. Synthetic teaching geometry—not a causal claim.*

![c600 teaching panel 19 (original).](../assets/figures/ml_fig_c600_19.png)
*Figure — Glossary optim strip c600. Synthetic teaching geometry—not a causal claim.*

![c601 teaching panel 19 (original).](../assets/figures/ml_fig_c601_19.png)
*Figure — Glossary SSL strip c601. Synthetic teaching geometry—not a causal claim.*

![c602 teaching panel 19 (original).](../assets/figures/ml_fig_c602_19.png)
*Figure — Glossary RL strip c602. Synthetic teaching geometry—not a causal claim.*

![c603 teaching panel 19 (original).](../assets/figures/ml_fig_c603_19.png)
*Figure — Glossary privacy strip c603. Synthetic teaching geometry—not a causal claim.*

![c604 teaching panel 19 (original).](../assets/figures/ml_fig_c604_19.png)
*Figure — Glossary graph strip c604. Synthetic teaching geometry—not a causal claim.*

![c605 teaching panel 19 (original).](../assets/figures/ml_fig_c605_19.png)
*Figure — Glossary calib strip c605. Synthetic teaching geometry—not a causal claim.*

![c606 teaching panel 19 (original).](../assets/figures/ml_fig_c606_19.png)
*Figure — Glossary genAI strip c606. Synthetic teaching geometry—not a causal claim.*

![c607 teaching panel 19 (original).](../assets/figures/ml_fig_c607_19.png)
*Figure — Glossary metric strip c607. Synthetic teaching geometry—not a causal claim.*

![c608 teaching panel 19 (original).](../assets/figures/ml_fig_c608_19.png)
*Figure — Glossary regularize strip c608. Synthetic teaching geometry—not a causal claim.*

![c609 teaching panel 19 (original).](../assets/figures/ml_fig_c609_19.png)
*Figure — Glossary embedding strip c609. Synthetic teaching geometry—not a causal claim.*

![c610 teaching panel 19 (original).](../assets/figures/ml_fig_c610_19.png)
*Figure — Glossary quant strip c610. Synthetic teaching geometry—not a causal claim.*

![c611 teaching panel 19 (original).](../assets/figures/ml_fig_c611_19.png)
*Figure — Glossary eval strip c611. Synthetic teaching geometry—not a causal claim.*

![c612 teaching panel 19 (original).](../assets/figures/ml_fig_c612_19.png)
*Figure — Glossary cluster strip c612. Synthetic teaching geometry—not a causal claim.*

![c613 teaching panel 19 (original).](../assets/figures/ml_fig_c613_19.png)
*Figure — Glossary IR strip c613. Synthetic teaching geometry—not a causal claim.*

![c614 teaching panel 19 (original).](../assets/figures/ml_fig_c614_19.png)
*Figure — Glossary deploy strip c614. Synthetic teaching geometry—not a causal claim.*

![c615 teaching panel 19 (original).](../assets/figures/ml_fig_c615_19.png)
*Figure — Glossary loss strip c615. Synthetic teaching geometry—not a causal claim.*

![c616 teaching panel 19 (original).](../assets/figures/ml_fig_c616_19.png)
*Figure — Glossary optim strip c616. Synthetic teaching geometry—not a causal claim.*

![c617 teaching panel 19 (original).](../assets/figures/ml_fig_c617_19.png)
*Figure — Glossary SSL strip c617. Synthetic teaching geometry—not a causal claim.*

![c618 teaching panel 19 (original).](../assets/figures/ml_fig_c618_19.png)
*Figure — Glossary RL strip c618. Synthetic teaching geometry—not a causal claim.*

![c619 teaching panel 19 (original).](../assets/figures/ml_fig_c619_19.png)
*Figure — Glossary privacy strip c619. Synthetic teaching geometry—not a causal claim.*

![c620 teaching panel 19 (original).](../assets/figures/ml_fig_c620_19.png)
*Figure — Glossary graph strip c620. Synthetic teaching geometry—not a causal claim.*

![c621 teaching panel 19 (original).](../assets/figures/ml_fig_c621_19.png)
*Figure — Glossary calib strip c621. Synthetic teaching geometry—not a causal claim.*

![c622 teaching panel 19 (original).](../assets/figures/ml_fig_c622_19.png)
*Figure — Glossary genAI strip c622. Synthetic teaching geometry—not a causal claim.*

![c623 teaching panel 19 (original).](../assets/figures/ml_fig_c623_19.png)
*Figure — Glossary metric strip c623. Synthetic teaching geometry—not a causal claim.*

![c624 teaching panel 19 (original).](../assets/figures/ml_fig_c624_19.png)
*Figure — Glossary regularize strip c624. Synthetic teaching geometry—not a causal claim.*

![c625 teaching panel 19 (original).](../assets/figures/ml_fig_c625_19.png)
*Figure — Glossary embedding strip c625. Synthetic teaching geometry—not a causal claim.*

![c626 teaching panel 19 (original).](../assets/figures/ml_fig_c626_19.png)
*Figure — Glossary quant strip c626. Synthetic teaching geometry—not a causal claim.*

![c627 teaching panel 19 (original).](../assets/figures/ml_fig_c627_19.png)
*Figure — Glossary eval strip c627. Synthetic teaching geometry—not a causal claim.*

![c628 teaching panel 19 (original).](../assets/figures/ml_fig_c628_19.png)
*Figure — Glossary cluster strip c628. Synthetic teaching geometry—not a causal claim.*

![c629 teaching panel 19 (original).](../assets/figures/ml_fig_c629_19.png)
*Figure — Glossary IR strip c629. Synthetic teaching geometry—not a causal claim.*

![c630 teaching panel 19 (original).](../assets/figures/ml_fig_c630_19.png)
*Figure — Glossary deploy strip c630. Synthetic teaching geometry—not a causal claim.*

![c631 teaching panel 19 (original).](../assets/figures/ml_fig_c631_19.png)
*Figure — Glossary loss strip c631. Synthetic teaching geometry—not a causal claim.*

![c632 teaching panel 19 (original).](../assets/figures/ml_fig_c632_19.png)
*Figure — Glossary optim strip c632. Synthetic teaching geometry—not a causal claim.*

![c633 teaching panel 19 (original).](../assets/figures/ml_fig_c633_19.png)
*Figure — Glossary SSL strip c633. Synthetic teaching geometry—not a causal claim.*

![c634 teaching panel 19 (original).](../assets/figures/ml_fig_c634_19.png)
*Figure — Glossary RL strip c634. Synthetic teaching geometry—not a causal claim.*

![c635 teaching panel 19 (original).](../assets/figures/ml_fig_c635_19.png)
*Figure — Glossary privacy strip c635. Synthetic teaching geometry—not a causal claim.*
