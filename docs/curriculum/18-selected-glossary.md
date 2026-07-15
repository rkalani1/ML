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

*Figure — Glossary leakage map. Four common families: **temporal** (post-decision features), **fit/CV** (scaler/vocab/selector fit on the full cohort), **label proxy** (treatment or post-outcome codes as inputs), and **target-encoding** without LOO/OOF. All inflate apparent performance at train time and fail at true index time. Prediction ≠ causation.*
