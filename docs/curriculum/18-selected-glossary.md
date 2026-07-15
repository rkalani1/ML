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
