# Chapter 6. Feature Engineering

## Opening
![Feature legality at index time (original).](../assets/figures/ml_fig_feature_legality.png)

*Feature legality at index time (original).*


Two sites train the same classifier for early neurologic deterioration. One encodes time-to-CT as minutes; the other as free-text. Feature engineering is where most ‘AI magic’ quietly lives—and where leakage hides.


![Feature pipeline: raw → impute → encode → scale → select → model (original).](../assets/figures/ml_fig_feature_pipeline.png)

*Every transform is fit on the training fold only, then frozen for validation/test (original).*

![Feature timing versus prediction time — leakage trap (original).](../assets/figures/ml_fig_leakage_timeline.png)

*Feature timing versus prediction time — leakage trap (original).*

![Preprocessing fit-on-train discipline vs full-cohort leakage (scientific split timeline; original).](../assets/figures/ml_fig_preprocess_fit_split.png)

*Figure — Honest preprocessing. **Wrong (top):** imputer, scaler, vocabulary, and selector statistics are fit on the entire cohort, so validation and test rows leak into the fitted transforms. **Right (bottom):** fit μ, σ, imputer, and vocabulary on the training segment only, freeze those objects, and apply transform-only to validation and test. Nested cross-validation re-fits the same objects inside each training fold. Timing leakage (post-decision codes) and fit leakage (global statistics) are different failure modes—both invalidate reported performance.*

## Learning Objectives

Classify feature types and state their modeling implications for neurologic and epidemiologic data.

Apply filter, wrapper (SFS/SBS, genetic algorithms), and embedded feature selection with honest nested validation.

Scale numerical features (min–max, z-score, L1/L2 norms) and transform with log and Box–Cox under fit-on-train discipline.

Encode categoricals with one-hot, dummy, effect, hashing, bin counting, and target encoding without leakage.

Engineer text features (BoW, subword, n-grams, POS, Word2Vec/GloVe/FastText, TF–IDF/TextRank/RAKE/YAKE).

Extract image and video features (HS corners, MSER, HOG, SIFT, watershed; motion vectors, optical flow, 3D CNN, GCN pointer).

Describe time-series and signal features: stationarity, seasonality, trend, motifs, lags, change points, and smoothing.

Distinguish MCAR, MAR, and MNAR missingness and translate each into legitimate imputation and missingness-indicator features that stay legal at index time.

Audit clinical pipelines for leakage and document feature availability at inference time.

## Basic Concepts of Feature Engineering

A learning algorithm never sees the patient, the stroke code, or the cohort protocol; it sees a matrix of features X and, in supervised settings, a target y. Feature engineering is the craft of transforming raw measurements—vital signs, NIH Stroke Scale (NIHSS) items, door-to-needle timestamps, imaging scores, comorbidity codes, laboratory panels, free-text notes, pixels, and waveforms—into representations that make the predictive or descriptive relationship easier to approximate.

![6.1: A feature-engineering pipeline: raw data flows through impute, encode, scale, and select stages into the model. Every tr](../assets/figures/ml_concept_6.1_4d270679.png)

*Figure 6.1 — original teaching graphic.*

In classical machine learning for clinical prediction, feature engineering is often the highest-leverage step. Deep learning can learn representations from images and text, yet tabular stroke registries, administrative claims, and multi-center epidemiologic cohorts still depend on deliberate design of covariates, careful handling of missingness, and honest separation of what is known at decision time from what is known only in retrospect. This chapter treats features as design decisions with clinical, statistical, and ethical consequences.

Feature generation creates new variables from raw inputs (ratios, encodings, embeddings). Feature selection chooses a subset of candidate variables. Both must sit inside validation loops so that choices are not tuned on the test set. Throughout, examples draw from acute ischemic stroke (AIS), intracerebral hemorrhage (ICH), TIA workups, and population surveillance—dense illustrations of time-critical features, ordinal scales, and leakage risk.

## Feature Selection: Filter, Wrapper, and Embedded Methods

Feature selection reduces dimensionality, lowers overfitting risk, speeds training and inference, and can improve interpretability. It is complementary to—but distinct from—dimensionality reduction (Chapter 7), which typically creates new coordinates rather than choosing original variables. Three classical families structure the literature: filters, wrappers, and embedded methods.

### Filtering Methods

Filters score each feature (or small groups) independently of a specific downstream model using statistical or information-theoretic criteria: variance thresholds, Pearson or Spearman correlation with y, mutual information, chi-square tests for categorical associations, ANOVA F-scores, and relief-style neighbor methods. Filters are fast and model-agnostic, making them good first screens on wide EHR or omics tables.

![Mutual-information filter scores vs downstream utility (synthetic; original).](../assets/figures/ml_fig_mi_filter.png)

*Figure — High MI is not a free pass. **Left:** synthetic MI bars—gold marks high MI but low task utility (zip/scanner-style proxies). **Right:** MI vs downstream ΔAUROC scatter. Nest filters inside CV; association with y is not causation and can encode site leakage.*

Limitations are real. Univariate filters miss synergistic pairs (two weak features that jointly predict). Correlation with y is not causation and can select mediators or colliders inappropriately for etiologic models. Highly redundant features may all pass a univariate threshold. Always re-evaluate selected sets inside cross-validation; selecting on the full sample before splitting leaks selection decisions into every fold.

### Wrapper Methods: SFS, SBS, and Genetic Algorithms

Wrappers treat the learning algorithm as a black box and search subsets to optimize a validation score (AUC, RMSE, log loss). Sequential Forward Selection (SFS) starts from the empty set and greedily adds the feature that most improves the score. Sequential Backward Selection (SBS) starts from all features and greedily removes the least harmful. Floating variants (SFFS/SBFS) allow limited backtracking. Wrappers capture model-specific interactions but are computationally expensive: each candidate subset requires training and evaluation.

Genetic algorithms (GAs) represent subsets as binary chromosomes and evolve populations with selection, crossover, and mutation under a fitness function equal to (or regularized by) validation performance and subset size. GAs explore more broadly than pure greedy SFS/SBS but introduce stochasticity and many hyperparameters (population size, mutation rate). In clinical n-small settings, aggressive wrappers overfit the validation fold; nested cross-validation or a fixed temporal outer holdout is mandatory.

![Nested CV for feature selection: outer holdout vs optimistic leaky scores (synthetic; original).](../assets/figures/ml_fig_nested_fs_cv.png)

*Figure — Nest the selector. **Left:** each outer fold freezes an untouched test slice while the outer-train block runs inner CV that chooses filters/wrappers/hyperparameters. **Right:** synthetic AUROC gap—reporting the same folds used to pick features inflates “skill”; nested scores sit near the true dashed line. Nested CV is expensive; a single temporal outer holdout is the common clinical compromise. Selection is not causal feature discovery.*

### Embedded Methods

Embedded methods perform selection during model training. L1-regularized linear and logistic regression (Lasso) drive coefficients to exact zero. Tree ensembles rank features by impurity decrease or permutation importance. Gradient-boosted trees similarly yield gain-based importances. Embedded methods are efficient relative to wrappers and often more accurate than pure filters, but importances are model-dependent and can be unstable under correlated clinical labs.

![SHAP / coefficient importance under collinearity: attribution mass swaps (synthetic; original).](../assets/figures/ml_fig_shap_collinearity.png)

*Figure — Collinearity breaks credit assignment. **Left:** two severity proxies share correlation ≈0.95 with each other; a third feature is nearly independent. **Right:** bootstrap |coefficients| for a linear risk model swing between the two collinear inputs even though both are proxies for one latent signal. SHAP values and impurity importances show the same credit-swapping. Report correlated groups, not a false unique “top feature.” Attribution ≠ causation.*

Filters: fast univariate/global scores; miss interactions; cheap screens.

Wrappers (SFS/SBS/GA): optimize a model’s validation metric; expensive; nest in CV.

Embedded: L1, tree importance; selection during training; still validate externally.

## Feature Engineering for Numerical Data

Numeric features take values on interval or ratio scales: age in years, systolic blood pressure in mmHg, serum glucose, infarct volume in mL, door-to-CT time in minutes. Many algorithms assume comparable scales. Euclidean distance in k-NN or k-means is dominated by large-range variables. L2-regularized linear models penalize large coefficients; unscaled features with large numeric ranges receive smaller coefficients for the same physical effect, distorting regularization paths. Tree-based models are largely invariant to monotonic rescaling of individual features.

![Box–Cox / log power transforms for skewed numeric features (original).](../assets/figures/ml_fig_boxcox.png)

*Figure — Power family. **Left:** Box–Cox curves for several \(\lambda\) (including log at \(\lambda=0\)). **Right:** a lognormal-like raw lab vs log-transformed density. Fit \(\lambda\) and scalers on the training fold only; transforms reshape geometry for learning—they do not create causal units of biology.*

![6.2: The same right-skewed glucose feature shown raw (mg/dL), min-max scaled to [0, 1], and z-standardized (mean 0, SD 1). Be](../assets/figures/ml_concept_6.2_36c46b0a.png)

*Figure 6.2 — original teaching graphic.*

### Min–Max Scaling

Min–max scaling maps a feature to [0, 1] (or a chosen interval) via (x − x_min)/(x_max − x_min). It is simple and preserves zero when x_min = 0, but is sensitive to extreme outliers: a single glucose of 1200 mg/dL stretches the scale for all patients. Estimate x_min and x_max on the training fold only, then apply to validation and test.

### Standardization and z-Score

Z-score standardization uses (x − μ)/σ so that the transformed feature has mean 0 and variance 1 on the data used to estimate μ and σ. It is the default for many linear models, PCA, and neural nets on tabular inputs. Robust alternatives replace μ and σ with median and IQR or MAD to resist lab errors. Again: fit on train only.

### L1 and L2 Norms

Feature-wise or sample-wise normalization by vector norms appears throughout ML. The L2 norm ‖x‖₂ = √(∑ x_j²) defines Euclidean length; dividing a row by ‖x‖₂ projects it onto the unit sphere (common in text). The L1 norm ‖x‖₁ = ∑ |x_j| relates to sparse geometry and Manhattan distance. For a two-feature row x = (3, 4): ‖x‖₂ = √(9 + 16) = 5, so the L2-normalized row is (0.6, 0.8) and lands on the unit circle; ‖x‖₁ = 3 + 4 = 7, so the L1-normalized row is (3/7, 4/7) ≈ (0.43, 0.57). Column-wise L2 scaling differs from row-wise unit-norm scaling: choose based on whether comparable feature units or comparable patient-vector lengths matter.

### Logarithmic and Box–Cox Transformations

Right-skewed positive variables—length of stay, infarct volume, costs, onset-to-arrival times—often benefit from log transforms. Use log1p(x) = log(1+x) when zeros occur. Logarithmic transforms compress multiplicative effects into additive ones and can stabilize variance for linear models.

The Box-Cox (Box–Cox) family is a parameterized power transform for strictly positive y or x: x(λ) = (x^λ − 1)/λ for λ ≠ 0, and log x for λ = 0. Maximum likelihood or profile likelihood on training data chooses λ. Yeo–Johnson extends Box–Cox to non-positive values. Transform parameters, like scalers, must be estimated inside the training pipeline. Over-transforming can hurt tree models that already split on raw thresholds clinicians understand.

## Feature Engineering for Categorical Data

Categorical features label classes without inherent order (hospital site, stroke mechanism treated as unordered labels, imaging modality). Ordinal features have ordered levels without guaranteed equal spacing (mRS 0–6, ASPECTS). Encoding choices invent geometry; bad choices invent false neighbors.

![6.3: One-hot versus target (mean) encoding of a hospital-site column. One-hot expands the category into indicator columns tha](../assets/figures/ml_concept_6.3_e34e911f.png)

*Figure 6.3 — original teaching graphic.*

### One-Hot Encoding and Dummy Coding

One-hot encoding creates a binary column per level. Dummy coding is the same idea with one reference level dropped to avoid perfect collinearity with an intercept in unregularized linear models. For a stroke mechanism field with levels {LAA, CE, SVS, Other, Undetermined}, dummy coding yields four indicators relative to a chosen baseline. One-hot/dummy works well with linear models and moderate cardinality; it explodes for hundreds of zip codes or thousands of drug strings.

### Effect Coding

Effect (sum-to-zero) coding contrasts each non-reference level with the grand mean rather than a single baseline category. Mechanically it uses the same k−1 columns as dummy coding, but the reference level is coded as −1 in every column instead of 0. For sites {Harbor, Metro, Clinic} with Clinic as reference, the two columns (is-Harbor, is-Metro) take Harbor → (1, 0), Metro → (0, 1), and Clinic → (−1, −1). Those −1 entries force the fitted level effects to sum to zero, so the intercept estimates the grand (unweighted) mean of the level means and each coefficient reads as that level’s deviation from the grand mean. The same information is reinterpreted: under dummy coding the Harbor coefficient means “Harbor versus Clinic,” whereas under effect coding it means “Harbor versus the average site.” Epidemiologists sometimes prefer effect coding for ANOVA-style interpretation; it remains a full-rank design for k levels with k−1 columns, differing from dummy coding only in the contrast matrix.

### Feature Hashing

The hashing trick maps category strings (or tokens) through a hash function into a fixed number of buckets m, typically with a sign hash to reduce bias. Collisions mix distinct categories into the same column—acceptable when m is large and models are regularized. Hashing enables streaming and memory-fixed pipelines for high-cardinality medication names without maintaining a growing vocabulary. Collisions are not clinically interpretable; do not use hashed columns as standalone audit trails for which drug drove a prediction.

![Feature hashing: collision probability and recoverable R² vs hash width m (synthetic; original).](../assets/figures/ml_fig_hash_collisions.png)

*Figure — Hash width is a capacity knob. **Left:** approximate birthday collision probability among k actives falls as m grows. **Right:** a signed-hash linear model’s recoverable signal (in-sample R²) collapses when m is too small. Choose m on validation; hashed columns are not causal feature selection and are poor audit trails for “which drug drove the score.”*

### Bin Counting

Bin counting replaces a categorical level with historical counts or rates: how often this hospital saw LVO, how often this code co-occurred with the label in past data. It compresses high cardinality into numeric summaries. Like all history-based features, bin counts must be computed with time-aware or out-of-fold discipline so that a row does not use its own future outcomes. Supervised bin counts that use the label are a short step from target encoding and inherit the same leakage risks.

### Target Encoding

Target (mean) encoding replaces a category with a smoothed estimate of the mean outcome in that category: e_c = (n_c · ȳ_c + m · ȳ) / (n_c + m), where ȳ_c is the category mean of y, ȳ is the global mean, n_c is the count, and m is a prior strength. Without out-of-fold or leave-one-out discipline, target encoding leaks the row’s own label into its features and produces fantastically optimistic validation scores—especially deadly in small stroke subgroups (rare mechanisms, rare hospitals). Frequency encoding (replace level by its prevalence) is weaker but safer as a baseline.

### Worked Example: Wrong Integer Encoding vs One-Hot

Consider four fictional TIA/minor stroke encounters predicting a continuous care-intensity score y. Features: age, unordered site (Harbor, Metro, Clinic), pathway (Standard vs Expedited).
Row1: age 62, Harbor, Standard, y=12
Row2: age 71, Metro, Standard, y=18
Row3: age 55, Clinic, Expedited, y=9
Row4: age 80, Harbor, Expedited, y=22

Wrong encoding: Harbor→1, Metro→2, Clinic→3 invents false geometry. Euclidean distance between Harbor Standard and Clinic Expedited becomes sqrt((62−55)²+(1−3)²+(0−1)²)=sqrt(54)≈7.35, while Harbor to Metro is sqrt((62−71)²+(1−2)²)≈9.06, so Clinic looks closer to Harbor than Metro does—driven by arbitrary codes, not care pattern. Correct approach: one-hot or dummy site indicators, pathway as 0/1, and z-score age using training μ and σ only. Neighbor geometry then mixes age with meaningful binary mismatches rather than fake ordinal hospital spacing.

### Worked Example: Smoothed Target Encoding with Out-of-Fold Discipline

Take a binary outcome—symptomatic intracranial hemorrhage (sICH) after thrombolysis—and encode hospital site. Suppose the training fold holds 20 patients with 4 events, so the global rate is ȳ = 4/20 = 0.20, with prior strength m = 10. Apply e_c = (n_c · ȳ_c + m · ȳ) / (n_c + m):

Site A: n_A = 5 patients, 2 events, so ȳ_A = 0.40. Then e_A = (5·0.40 + 10·0.20)/(5 + 10) = (2 + 2)/15 = 4/15 ≈ 0.267—the raw 0.40 shrunk toward the 0.20 prior.

Rare site B: n_B = 1 patient, 1 event, so the naive rate ȳ_B = 1.00, a single patient screaming “100% risk.” Smoothing tames it: e_B = (1·1.00 + 10·0.20)/(1 + 10) = 3/11 ≈ 0.273.

Smoothing alone does not stop leakage, because e_A was computed from the very rows it will encode. The out-of-fold (or leave-one-out) fix encodes each row from data that exclude that row. Leave-one-out for a site-A patient whose own y_i = 1 gives e = (5·0.40 − 1 + 10·0.20)/((5 − 1) + 10) = (2 − 1 + 2)/14 = 3/14 ≈ 0.214; for a site-A patient with y_i = 0 it gives e = (2 − 0 + 2)/14 = 4/14 ≈ 0.286. The encoding now moves with the patient’s own label—exactly the dependence leave-one-out removes so the label cannot leak into its own feature. For singleton site B, leave-one-out leaves no other B rows and the estimate collapses to the prior: e = (1·1.00 − 1 + 10·0.20)/((1 − 1) + 10) = 2/10 = 0.20. Without this discipline, site B would enter training as a perfect but fictitious predictor equal to its lone patient’s outcome—the classic small-subgroup leakage that inflates validation AUC and evaporates prospectively.

![Target encoding: naive leakage vs leave-one-out (chapter numbers; original).](../assets/figures/ml_fig_target_enc_loo.png)

*Figure — Target-encoding hygiene. **Left:** site A (n=5, \(\bar y_c=0.40\), prior strength m=10): naive smoothed mean ≈0.267 still mixes the row’s own label; LOO shifts with y_i (≈0.214 if y=1, ≈0.286 if y=0) so the label does not leak into its own feature. **Right:** singleton site B—naive encoding hugs the lone outcome; LOO collapses to the global prior. Small-subgroup naive target encoding manufactures AUROC that evaporates prospectively.*

## Feature Engineering for Textual Data

Clinical text—radiology impressions, ED notes, discharge summaries—is high-value and high-risk. Representations range from sparse counts to dense embeddings. Preprocessing typically includes section segmentation, de-identification, tokenization, and negation/uncertainty detection before any bag-of-words pipeline.

### Bag-of-Words and Subword Tokenization

Bag-of-words (BoW) represents a document as counts (or binary presence) of vocabulary terms, discarding order. It is simple, interpretable, and strong with linear models on short clinical snippets. Subword tokenization (BPE, WordPiece, Unigram LM) splits rare words into frequent pieces, reducing out-of-vocabulary rates for drug names and neologisms. Subwords feed both classical sparse models and neural language models.

### N-Grams and Part-of-Speech Tagging

N-grams extend BoW to contiguous token sequences (bigrams, trigrams), capturing short phrases such as “last known well” or “no acute infarct.” Dimensionality grows quickly; hashing or frequency thresholds help. Part-of-speech (POS) tagging labels tokens as nouns, verbs, adjectives, etc., enabling filters (keep nouns/adjectives for keyword features) and patterns for information extraction. POS features alone rarely beat strong embeddings but aid rule-based clinical NLP.

### Word Embeddings: Word2Vec, GloVe, and FastText

Word embeddings map tokens to dense vectors so that geometric proximity reflects distributional similarity. Word2Vec trains with skip-gram or CBOW objectives: predict context from word or word from context using shallow networks and negative sampling. GloVe factorizes a global word–word co-occurrence matrix with a weighted least-squares objective, blending count-based and prediction-based ideas. FastText extends Word2Vec with character n-gram vectors, improving morphology and rare words—useful for biomedical compounds.

![6.4: A two-dimensional feature embedding in which tokens sharing clinical context fall into labeled semantic neighborhoods (v](../assets/figures/ml_concept_6.4_44e273e7.png)

*Figure 6.4 — original teaching graphic.*

Document features can average word vectors, use TF–IDF-weighted averages, or feed sequences to recurrent/transformer encoders (later chapters). Domain-adapted embeddings (trained on clinical notes) often outperform generic web embeddings for EHR tasks, but still require leakage control: do not train embeddings on notes that contain the label narrative you are trying to predict if those notes are written after the outcome.

### Theme and Keyword Extraction: TF–IDF, TextRank, RAKE, YAKE

TF–IDF (Chapter 5) ranks terms that are frequent in a document but rare in the corpus—still a strong keyword baseline. TextRank builds a graph of terms or sentences with co-occurrence edges and applies a PageRank-style centrality score; top-ranked terms/sentences become keywords/summaries. RAKE (Rapid Automatic Keyword Extraction) splits text on stopwords and punctuation, scores candidate phrases by word co-occurrence statistics, and is fast unsupervised keywording. YAKE (Yet Another Keyword Extractor) uses statistical features (casing, position, frequency, relatedness) without external corpora, often strong on single documents. LLM-based keyword extraction is emerging but needs governance for PHI and reproducibility.

## Feature Engineering for Image and Video Data

Before deep end-to-end learning dominated vision, carefully designed local descriptors powered matching, detection, and recognition. Classical features remain useful for small medical datasets, interpretable pipelines, and hybrid systems that combine handcrafted descriptors with shallow classifiers.

### Image Processing Concepts

Digital images are arrays of intensities (grayscale) or channels (RGB, or multi-parametric MRI modalities). Preprocessing includes resampling to isotropic voxels, intensity normalization (z-score within brain mask, histogram matching), bias-field correction in MRI, and registration to templates. Features may be global (histograms, moments) or local (keypoints and descriptors). Medical imaging adds DICOM metadata, scanner/protocol batch effects, and strict train–test separation by patient—not by slice.

### Harris–Stephens Corner Detection

Harris–Stephens detects corners by examining the local autocorrelation of image intensities. Large intensity variation in two directions indicates a corner; edges vary in one direction; flat regions vary little. The Harris matrix of smoothed gradients yields a corner response score. Corners provide repeatable keypoints for registration and tracking.

### MSER, HOG, SIFT, and Watershed

Maximally Stable Extremal Regions (MSER) find connected components that remain stable across intensity thresholds—useful for blob-like structures and text-like regions. Histogram of Oriented Gradients (HOG) pools gradient orientations in spatial cells and normalizes blocks, capturing shape while tolerating modest illumination change; HOG famously powered pedestrian detection and still appears in medical texture pipelines.

Scale-Invariant Feature Transform (SIFT) detects scale-space extrema (typically Difference-of-Gaussians), assigns orientations, and builds 128-dimensional gradient histograms in rotated patches, achieving invariance to scale and rotation and robustness to moderate illumination change. SIFT descriptors support matching across images for registration and object recognition. Watershed transformation treats the image (or its gradient) as a topographic surface and floods basins from markers, segmenting regions; over-segmentation is common without careful marker choice—relevant for lesion or organ boundary sketches when combined with clinical priors.

### Video: Motion Vectors, Optical Flow, 3D CNNs, and Graphs

Video adds time. Motion vectors from block-matching or codec side information describe coarse displacement of patches between frames. Optical flow estimates dense per-pixel motion fields under brightness constancy and spatial smoothness assumptions (Horn–Schunck, Lucas–Kanade, and modern learning-based flows). Flow features capture gait, ultrasound probe motion, or seizure-related movement in monitoring videos.

3D convolutional neural networks extend 2D kernels across space and time to learn spatiotemporal filters directly from clip volumes—used in action recognition and increasingly in dynamic medical imaging. Graph convolutional networks (GCNs) process features on graph-structured data: skeleton joints for pose, electrode graphs for EEG-as-video-adjacent signals, or region-connectivity graphs from imaging parcellations. Treat GCNs as a pointer to graph feature learning (expanded in Chapter 15); the engineering message is that video and spatiotemporal clinical signals need features that respect both appearance and motion or topology.

## Feature Engineering for Signals and Time Series

Time series appear as vital-sign streams, continuous EEG, wearable accelerometry, daily case counts, and lab trajectories. Feature engineering extracts stationary summaries, seasonal patterns, trends, motifs, lags, and change points that downstream models can consume—or prepares series for dedicated forecasting models (ARIMA in Chapter 8).

### Stationarity, Seasonality, and Trend

A weakly stationary series has time-invariant mean and autocovariance structure. Many clinical series are non-stationary: means drift with disease progression, variance changes after interventions, and levels jump at care transitions. Differencing, detrending, and log transforms can approach stationarity. Seasonality is periodic structure (diurnal BP patterns, weekly ED volumes, academic calendar effects on staffing). Trends are long-run drifts (improving door-to-needle times after a quality program). Seasonal-trend decomposition (STL and relatives) separates components for feature use or residual modeling.

### Motifs, Lags, and Change Points

Motifs are recurring approximate subsequences—characteristic EEG graphoelements, stereotyped heart-rate patterns, or repeated mobility signatures. Discovery uses sliding windows with distance measures (Euclidean, DTW) and matrix-profile methods. Lag features are past values y_{t−k} or x_{t−k} used as predictors of the present; choosing lag sets encodes memory depth. Change-point detection finds times when distributional parameters shift (mean, variance, spectral content): CUSUM, PELT, Bayesian online change-point methods, and likelihood-ratio scans. Clinically, change points may mark deterioration, treatment effect onset, or artifactual sensor disconnects—always review before treating them as pure biology.

### Signal Concepts, Types, and Smoothing

Signals may be continuous-time concepts sampled discretely: amplitude, frequency content, phase, and noise characteristics matter. Biomedical signals include ECG, EEG, EMG, photoplethysmography, and intracranial pressure traces. Features include time-domain statistics (mean, variance, skewness, line length), frequency-domain band powers via Fourier methods, time–frequency features via wavelets (Chapter 7), and nonlinear measures (entropy, fractal indices) used carefully with small samples.

Smoothing reduces high-frequency noise: moving averages, exponential smoothing, Savitzky–Golay polynomial filters, and median filters (robust to spikes). Over-smoothing erases clinically sharp events (seizure onsets, BP drops). Choose kernels with time constants matched to the physiology and the decision latency. Fit smoother parameters on training segments; applying adaptive smoothers that peek at future samples inside a window is a subtle form of leakage for causal real-time prediction.

Concrete lag example: let MAP_t be mean arterial pressure at minute t. Features available to predict deterioration at t include MAP_t, MAP_{t−5}, MAP_{t−15}, the 15-minute slope (MAP_t − MAP_{t−15})/15, and a rolling standard deviation over [t−15, t] that uses only past samples (a causal window). Including MAP_{t+5} would be leakage for real-time alarms. Seasonality example: ED arrival counts often show weekday harmonics; features sin(2π d/7), cos(2π d/7) for day-of-week d capture weekly cycles without imposing an arbitrary Monday=0 integer geometry on tree and linear models alike.

## Leakage in Clinical Data and Pipeline Discipline

Leakage occurs when information unavailable at the intended prediction time enters training features. In stroke modeling the offenders are legion: discharge mRS used to predict discharge mRS; peak troponin during admission used to predict in-hospital mortality when the decision point was ED arrival; hospital-acquired pneumonia or hemicraniectomy flags used as inputs to a model that claims to risk-stratify at door; length of stay as a day-0 feature; radiology final reads finalized after the treatment decision; target encoding without out-of-fold discipline; scalers fit on the full cohort before splitting.

![6.5: An index-time legality timeline. Features knowable at the prediction time t0 (left, indigo) are legal inputs; variables ](../assets/figures/ml_concept_6.5_7e34f1ea.png)

*Figure 6.5 — original teaching graphic.*

A practical audit: draw the clinical timeline. Mark t₀ (for example, first ED blood pressure). Every feature must be knowable at t₀ for a t₀ model. If a field is only complete at discharge coding, it is not an arrival feature. Aggregate statistics computed with the test patient included leak. Identifiers that encode the label leak. If validation scores look “too good,” audit for leakage before celebrating—and before publishing AUCs that will not replicate prospectively.

A feature pipeline is an ordered list of transforms—type coercion → range checks → flag missingness → impute → encode → scale → model—with a single fit API on training data and transform/predict on new data. Document feature lineage: source system, units, allowed range, missing conventions, and whether the field is known at inference. Version feature definitions with the model; silent ICD coding updates are distribution shift.

```
# Conceptual clinical pipeline (pseudocode)
pipe = Pipeline([
 ('coerce', TypesAndUnits()),
 ('missing_flags', AddMissingnessIndicators()), # informative-presence features
 ('impute', SeparateNumCatImputers()), # fit on train only
 ('encode', OneHotAndOrdinalMaps()),
 ('scale', StandardizeNumericOnly()),
 ('model', Ridge(alpha=1.0)),
])
pipe.fit(X_train, y_train) # every column known at t0
y_hat = pipe.predict(X_test)
```

## Missingness Mechanisms as Feature Decisions

Blanks are not merely a nuisance to be silently filled; they are data whose reason for absence changes the correct engineering choice. Rubin’s taxonomy names three mechanisms, and each licenses a different legitimate handling—and a different leakage risk.

![Cyclic sin/cos encoding for hour-of-day vs ordinal trap (original).](../assets/figures/ml_fig_cyclic_encoding.png)

*Figure — Periodic features. Ordinal hour distance wrongly separates 23 from 0; circular distance and the unit-circle (sin, cos) embedding preserve midnight continuity. Encoding choices change models—they do not by themselves establish **causal** time-of-day effects.*


![Supervised bin rates and WoE-style log-odds lift (synthetic; original).](../assets/figures/ml_fig_woe_binning.png)

*Figure — Supervised binning caution. Event rates by age bin and a WoE-style lift can look decisive—but bin edges must live inside CV or they leak. Association with outcome is not a causal effect.*


![Target encoding: full-data leakage vs out-of-fold values (synthetic; original).](../assets/figures/ml_fig_target_encoding_cv.png)

*Figure — Target encoding hygiene. Full-data category means leak labels into features; out-of-fold encodings are safer. Encoded values are predictive transforms—not causal effects.*


![Statistical interaction surface for x1·x2 (synthetic; original).](../assets/figures/ml_fig_interaction_surface.png)

*Figure — Interaction contours. Product terms change predictions non-additively. Statistical interactions need design before biological causal stories.*


![Distance geometry before vs after standardization (synthetic; original).](../assets/figures/ml_fig_feature_scaling.png)

*Figure — Unscaled features let large-unit variables dominate distances and penalties. Scaling is preprocessing—not a causal adjustment by itself.*


![Polynomial feature expansion growth (original).](../assets/figures/ml_fig_poly_blowup.png)

*Figure — Feature maps expand capacity—not causal graphs. Pred ≠ cause without design.*


![One-hot vs embedding size trade-off (original).](../assets/figures/ml_fig_onehot_vs_embed.png)

*Figure — Dense embeddings compress high-card cats. One-hot vs embedding size trade-off Pred != cause without design.*


![binedges teaching panel (original).](../assets/figures/ml_fig_bin_edges_cv.png)

*Figure — Teaching panel for binedges. Pred != cause without design.*


![Cycle-34 densify scientific panel 8 (original).](../assets/figures/ml_fig_c34_07.png)

*Figure — Continuous densify panel 8. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 8 (original).](../assets/figures/ml_fig_c35_07.png)

*Figure — Continuous densify panel 8. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 8 (original).](../assets/figures/ml_fig_c36_07.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 8 (original).](../assets/figures/ml_fig_c37_07.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 8 (original).](../assets/figures/ml_fig_c38_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 8 (original).](../assets/figures/ml_fig_c39_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 8 (original).](../assets/figures/ml_fig_c40_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 8 (original).](../assets/figures/ml_fig_c41_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 8 (original).](../assets/figures/ml_fig_c42_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 8 (original).](../assets/figures/ml_fig_c43_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 8 (original).](../assets/figures/ml_fig_c44_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 8 (original).](../assets/figures/ml_fig_c45_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 8 (original).](../assets/figures/ml_fig_c46_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 8 (original).](../assets/figures/ml_fig_c47_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 8 (original).](../assets/figures/ml_fig_c48_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 8 (original).](../assets/figures/ml_fig_c49_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 8 (original).](../assets/figures/ml_fig_c50_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 8 (original).](../assets/figures/ml_fig_c51_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 8 (original).](../assets/figures/ml_fig_c52_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 8 (original).](../assets/figures/ml_fig_c53_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 8 (original).](../assets/figures/ml_fig_c54_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 8 (original).](../assets/figures/ml_fig_c55_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 8 (original).](../assets/figures/ml_fig_c56_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 8 (original).](../assets/figures/ml_fig_c57_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 8 (original).](../assets/figures/ml_fig_c58_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 8 (original).](../assets/figures/ml_fig_c59_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 8 (original).](../assets/figures/ml_fig_c60_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 8 (original).](../assets/figures/ml_fig_c61_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 8 (original).](../assets/figures/ml_fig_c62_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 8 (original).](../assets/figures/ml_fig_c63_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 8 (original).](../assets/figures/ml_fig_c64_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 8 (original).](../assets/figures/ml_fig_c65_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 8 (original).](../assets/figures/ml_fig_c66_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 8 (original).](../assets/figures/ml_fig_c67_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 8 (original).](../assets/figures/ml_fig_c68_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 8 (original).](../assets/figures/ml_fig_c69_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 8 (original).](../assets/figures/ml_fig_c70_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 8 (original).](../assets/figures/ml_fig_c71_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 8 (original).](../assets/figures/ml_fig_c72_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 8 (original).](../assets/figures/ml_fig_c73_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 8 (original).](../assets/figures/ml_fig_c74_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 8 (original).](../assets/figures/ml_fig_c75_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 8 (original).](../assets/figures/ml_fig_c76_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 8 (original).](../assets/figures/ml_fig_c77_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 8 (original).](../assets/figures/ml_fig_c78_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 8 (original).](../assets/figures/ml_fig_c79_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 8 (original).](../assets/figures/ml_fig_c80_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 8 (original).](../assets/figures/ml_fig_c81_07.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

\n![6.6: Rubin's missingness mechanisms. MCAR: absence is driven by chance, independent of all data. MAR: absence in x3 is explai](../assets/figures/ml_concept_6.6_59c3837d.png)

*Figure 6.6 — original teaching graphic.*

### Missing Completely At Random (MCAR)

Under MCAR the probability that a value is missing depends on nothing at all, observed or unobserved: P(missing) is effectively a constant. Intuition: a reagent lot lapses and a random afternoon of glucose panels is voided; a scanner is offline for maintenance on arbitrary days. Mechanism: the observed rows are a uniform random subsample, so complete-case analysis is unbiased (only less efficient) and simple mean or median imputation does not distort the marginal distribution. When to invoke it: rarely, and only with a documented, outcome-independent cause—MCAR is testable, not something to assume by default in EHR data.

### Missing At Random (MAR)

Under MAR, missingness depends only on observed variables, not on the unseen value itself once those observables are conditioned upon. Intuition: NIHSS is more often blank for rapidly improving TIA patients triaged as low-acuity—the blank is explained by observed acuity and care setting, not by the particular NIHSS score that was never recorded. Mechanism: conditioning on the observed covariates renders the missingness “ignorable,” which is exactly the assumption under which principled multiple imputation (MICE) or model-based imputation is valid. When to use it: the workhorse assumption for imputing covariates you will still model—provided the variables that drive the missingness are themselves present in the data.

### Missing Not At Random (MNAR)

Under MNAR, missingness depends on the unobserved value itself, even after conditioning on the observables. Intuition: a troponin is never drawn precisely because the patient looked well (the unmeasured value would likely have been normal); a cognitive subscore is blank because the very deficit being measured prevents the patient from completing it. Mechanism: the fact of missingness carries information about the missing value—and often about the outcome—so complete-case analysis and naive imputation are biased. When to model it: represent the missingness explicitly (indicators, pattern-mixture or selection models) rather than pretending a single fill value is the truth.

### Translating Mechanism into Features

The mechanism dictates which of three moves is honest: (a) impute and forget—defensible under MCAR or MAR when the value, not its absence, carries the signal; (b) impute and add a missingness indicator; or (c) treat “missing” as an informative level in its own right. In prediction—as opposed to etiologic inference—informative missingness is frequently among the strongest signals: that a CTA was ordered proxies for suspected large-vessel occlusion, and “NIHSS incomplete” co-varies with chaotic, high-severity arrivals. Exploiting this informative presence is legitimate.

It is also where leakage hides. A “troponin missing” flag is a lawful index-time feature only if the ordering decision and its resolution are settled by t₀; if troponin typically results hours after the ED disposition being modeled, then “troponin measured” smuggles the future workup into a t₀ model. Worse, a blank created by the outcome—a discharge field empty because the patient died before discharge coding—encodes the label as absence, the purest leakage disguised as informative missingness. Two disciplines follow. First, fit every imputer on the training fold only: imputing with the full-cohort mean leaks the test distribution into each row, the same error as fitting a scaler on all data. Second, certify each missingness indicator against the timeline exactly as you would any other feature. Prefer per-mechanism handling: MCAR tolerates simple imputation; MAR earns multiple imputation with the missingness drivers included; MNAR demands that the absence itself be modeled and, above all, checked for index-time legality.

## Feature Generation Beyond Selection

Feature generation creates new coordinates rather than merely choosing among raw columns. Domain-informed generation often outperforms generic polynomial explosions because it aligns with pathophysiology and operations: LKW-to-door minutes, door-to-CT, door-to-needle, wake-up/unknown-onset flags, NIHSS subscores, ASPECTS or automated core/penumbra volumes known pre-decision, BP variability over a fixed pre-decision window, and arrival mode (EMS versus private). Ratios (glucose relative to a site median computed on training data only) can stabilize multi-center panels.

Generic expansions still have a place: pairwise interactions among a short pre-specified list, restricted cubic splines of age and severity, and cyclic encodings of hour-of-day via sine and cosine so that 23:00 neighbors 00:00. Avoid crossing every ICD code with every lab—combinatorial generation without selection or penalization is an overfitting engine. Name features with temporal qualifiers (pre_EVT_ASPECTS versus discharge_mRS) so joins cannot silently attach the wrong timepoint.

Missingness indicators deserve explicit generation when MAR/MNAR structure is informative: “NIHSS incomplete,” “no CTA performed,” “transfer missing prehospital times.” Combined with careful imputation (mean/median only as baselines; model-based or MICE for estimands; native missing support in some tree boosters), indicators prevent the fiction that every blank is a random draw from the observed marginal. Document unit conversions (mmol/L versus mg/dL) as first-class generation steps; multi-site disasters often begin as silent unit mismatches labeled as “outliers.”

## Worked Scaling and Transform Walk-Through

Suppose training glucose values (mg/dL) are {90, 110, 130, 250}. For min–max to [0,1]: x_min=90, x_max=250, so scaled values are 0, 20/160=0.125, 40/160=0.25, 1.0. A test value of 80 yields (80−90)/(250−90) = −0.0625—outside [0,1], which is allowed and preferable to refitting min/max on the test set. For z-scores with training mean μ=(90+110+130+250)/4=145 and population σ for illustration: deviations −55, −35, −15, +105; variance (3025+1225+225+11025)/4 = 3875; σ≈62.25; z ≈ (−0.88, −0.56, −0.24, +1.69). The extreme 250 dominates both scales—robust scaling with median and IQR would compress its influence.

Log1p on the same vector: log(91), log(111), log(131), log(251) ≈ (4.51, 4.71, 4.88, 5.53), reducing right skew before standardization. A Box–Cox search on strictly positive training glucose might favor λ near 0 (log-like) when the likelihood surface peaks there; report λ with the model card. Never choose λ on the full dataset including test patients.

## Clinical and Epidemiologic Notes

Prediction is not causal inference. A feature that improves RMSE for infarct volume prediction may be a confounder, a mediator, or a collider artifact when the scientific goal is an exposure–outcome effect. Separate prognostic models (association, calibration, decision utility) from etiologic models (confounding control, not maximal R²).

External validity depends on case mix and coding. TOAST labels vary by workup intensity; NIHSS completeness varies by site. Multi-center training without site-aware evaluation can hide that the model mostly detects documentation style. For rare events (sICH after thrombolysis), aggressive target encoding of sparse categories is especially fragile.

Ethically, excellent predictors may be unacceptable decision inputs (proxies for socioeconomic status used to deny rehabilitation). Feature engineering is also governance: choose representations that match the intended use, the decision time, and the population that will bear the model’s errors. Imaging-derived embeddings and free-text vectors can encode site watermarks (scanner vendor, note templates); evaluate subgroup performance by site, language, and transfer status.

Operationalization fails when the warehouse used for training joins discharge facts absent from the real-time feature store. Mirror production availability during training. For signals and video, patient-level splits prevent slice/frame leakage across train and test. For keyword features from notes written after outcomes, either restrict to early sections timed before t₀ or reframe the task as retrospective retrieval rather than prospective prediction.

Define t₀ and list every feature’s availability relative to t₀.

Prefer clinically named durations over raw timestamps.

Treat imaging and ordinal scales as measurement instruments with error.

Nest selection, encoding, and scaling inside validation.

Audit text and imaging features for post-outcome documentation and reads.

Generate missingness indicators when blanks are informative clinical signals.

Match video/signal splits to patients, not frames, to avoid leakage.

## Connections

Feature engineering is the hinge between raw data and every later method. Selection here contrasts with dimensionality reduction (Chapter 7), which forges new axes (PCA, manifold methods) rather than choosing original, clinically nameable columns; the two are often stacked—filter first, then project. The scaling and norm discipline of this chapter is a prerequisite for distance- and gradient-based learners (k-NN, SVM, neural nets) and for the regularization paths of L1/L2 penalized models, where unstandardized units silently reweight the penalty. The text pipeline and TF–IDF extend the vector-space ideas of Chapter 5, and the embeddings sketched here become the input layers of the sequence models in later chapters. The time-series features—stationarity, lags, seasonality, change points—feed the forecasting models (ARIMA and state-space methods, Chapter 8), while wavelet and spectral signal features connect to the transforms in Chapter 7. Graph-structured features (GCNs) point forward to Chapter 15. Cutting across all of these is the leakage-and-index-time discipline: less a feature topic than an evaluation contract that every transform in the pipeline must honor.


![c82 teaching panel 07 (original).](../assets/figures/ml_fig_c82_07.png)
*Figure — Target encoding must be fit inside each CV fold to avoid label leakage. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 07 (original).](../assets/figures/ml_fig_c83_07.png)
*Figure — Feature interaction heatmap (schematic strengths). Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 07 (original).](../assets/figures/ml_fig_c84_07.png)
*Figure — Nested preprocessing pipeline fits only on training folds. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 07 (original).](../assets/figures/ml_fig_c85_07.png)
*Figure — Feature transforms can add signal or smuggle leakage. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 07 (original).](../assets/figures/ml_fig_c86_07.png)
*Figure — Scaling choices (raw/log/z) reshape feature geometry. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 07 (original).](../assets/figures/ml_fig_c87_07.png)
*Figure — Feature hashing redistributes mass via collisions. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 07 (original).](../assets/figures/ml_fig_c88_07.png)
*Figure — Weight-of-evidence binning for sparse categories. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 07 (original).](../assets/figures/ml_fig_c89_07.png)
*Figure — Cyclical time features sin/cos encodings. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 07 (original).](../assets/figures/ml_fig_c90_07.png)
*Figure — Time-aware feature availability grid. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 07 (original).](../assets/figures/ml_fig_c91_07.png)
*Figure — Binning vs keep-continuous tradeoff. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 07 (original).](../assets/figures/ml_fig_c92_07.png)
*Figure — Missing indicator features. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 07 (original).](../assets/figures/ml_fig_c93_07.png)
*Figure — Entity embeddings for categoricals. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 07 (original).](../assets/figures/ml_fig_c94_07.png)
*Figure — Hashing trick collision budget. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 07 (original).](../assets/figures/ml_fig_c95_07.png)
*Figure — Leave-one-group-out CV blocks. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 07 (original).](../assets/figures/ml_fig_c96_07.png)
*Figure — Frequency-rarity feature plot. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 07 (original).](../assets/figures/ml_fig_c97_07.png)
*Figure — Interaction feature crossed grid. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 07 (original).](../assets/figures/ml_fig_c98_07.png)
*Figure — CatBoost ordered target stats. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 07 (original).](../assets/figures/ml_fig_c99_07.png)
*Figure — Rare-category grouping ladder. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 07 (original).](../assets/figures/ml_fig_c100_07.png)
*Figure — Target leakage calendar map. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 07 (original).](../assets/figures/ml_fig_c101_07.png)
*Figure — Count encoding with CV. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 07 (original).](../assets/figures/ml_fig_c102_07.png)
*Figure — Hash embeddings for IDs. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 07 (original).](../assets/figures/ml_fig_c103_07.png)
*Figure — Group-time feature freeze. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 07 (original).](../assets/figures/ml_fig_c104_07.png)
*Figure — Smoothed target encoding. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 07 (original).](../assets/figures/ml_fig_c105_07.png)
*Figure — Collisions in ID hashing. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 07 (original).](../assets/figures/ml_fig_c106_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 07 (original).](../assets/figures/ml_fig_c107_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 07 (original).](../assets/figures/ml_fig_c108_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 07 (original).](../assets/figures/ml_fig_c109_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 07 (original).](../assets/figures/ml_fig_c110_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 07 (original).](../assets/figures/ml_fig_c111_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 07 (original).](../assets/figures/ml_fig_c112_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 07 (original).](../assets/figures/ml_fig_c113_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 07 (original).](../assets/figures/ml_fig_c114_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 07 (original).](../assets/figures/ml_fig_c115_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 07 (original).](../assets/figures/ml_fig_c116_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 07 (original).](../assets/figures/ml_fig_c117_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 07 (original).](../assets/figures/ml_fig_c118_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 07 (original).](../assets/figures/ml_fig_c119_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 07 (original).](../assets/figures/ml_fig_c120_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 07 (original).](../assets/figures/ml_fig_c121_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 07 (original).](../assets/figures/ml_fig_c122_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 07 (original).](../assets/figures/ml_fig_c123_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 07 (original).](../assets/figures/ml_fig_c124_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 07 (original).](../assets/figures/ml_fig_c125_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 07 (original).](../assets/figures/ml_fig_c126_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 07 (original).](../assets/figures/ml_fig_c127_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 07 (original).](../assets/figures/ml_fig_c128_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 07 (original).](../assets/figures/ml_fig_c129_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 07 (original).](../assets/figures/ml_fig_c130_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 07 (original).](../assets/figures/ml_fig_c131_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 07 (original).](../assets/figures/ml_fig_c132_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 07 (original).](../assets/figures/ml_fig_c133_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 07 (original).](../assets/figures/ml_fig_c134_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 07 (original).](../assets/figures/ml_fig_c135_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 07 (original).](../assets/figures/ml_fig_c136_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 07 (original).](../assets/figures/ml_fig_c137_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 07 (original).](../assets/figures/ml_fig_c138_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 07 (original).](../assets/figures/ml_fig_c139_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 07 (original).](../assets/figures/ml_fig_c140_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 07 (original).](../assets/figures/ml_fig_c141_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 07 (original).](../assets/figures/ml_fig_c142_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 07 (original).](../assets/figures/ml_fig_c143_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 07 (original).](../assets/figures/ml_fig_c144_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 07 (original).](../assets/figures/ml_fig_c145_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 07 (original).](../assets/figures/ml_fig_c146_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 07 (original).](../assets/figures/ml_fig_c147_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 07 (original).](../assets/figures/ml_fig_c148_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 07 (original).](../assets/figures/ml_fig_c149_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 07 (original).](../assets/figures/ml_fig_c150_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 07 (original).](../assets/figures/ml_fig_c151_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 07 (original).](../assets/figures/ml_fig_c152_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 07 (original).](../assets/figures/ml_fig_c153_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 07 (original).](../assets/figures/ml_fig_c154_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 07 (original).](../assets/figures/ml_fig_c155_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 07 (original).](../assets/figures/ml_fig_c156_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 07 (original).](../assets/figures/ml_fig_c157_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 07 (original).](../assets/figures/ml_fig_c158_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 07 (original).](../assets/figures/ml_fig_c159_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 07 (original).](../assets/figures/ml_fig_c160_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 07 (original).](../assets/figures/ml_fig_c161_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 07 (original).](../assets/figures/ml_fig_c162_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 07 (original).](../assets/figures/ml_fig_c163_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 07 (original).](../assets/figures/ml_fig_c164_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 07 (original).](../assets/figures/ml_fig_c165_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 07 (original).](../assets/figures/ml_fig_c166_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 07 (original).](../assets/figures/ml_fig_c167_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 07 (original).](../assets/figures/ml_fig_c168_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 07 (original).](../assets/figures/ml_fig_c169_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 07 (original).](../assets/figures/ml_fig_c170_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 07 (original).](../assets/figures/ml_fig_c171_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 07 (original).](../assets/figures/ml_fig_c172_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 07 (original).](../assets/figures/ml_fig_c173_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 07 (original).](../assets/figures/ml_fig_c174_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 07 (original).](../assets/figures/ml_fig_c175_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 07 (original).](../assets/figures/ml_fig_c176_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 07 (original).](../assets/figures/ml_fig_c177_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 07 (original).](../assets/figures/ml_fig_c178_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 07 (original).](../assets/figures/ml_fig_c179_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 07 (original).](../assets/figures/ml_fig_c180_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 07 (original).](../assets/figures/ml_fig_c181_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 07 (original).](../assets/figures/ml_fig_c182_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 07 (original).](../assets/figures/ml_fig_c183_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 07 (original).](../assets/figures/ml_fig_c184_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 07 (original).](../assets/figures/ml_fig_c185_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 07 (original).](../assets/figures/ml_fig_c186_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 07 (original).](../assets/figures/ml_fig_c187_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 07 (original).](../assets/figures/ml_fig_c188_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 07 (original).](../assets/figures/ml_fig_c189_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 07 (original).](../assets/figures/ml_fig_c190_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 07 (original).](../assets/figures/ml_fig_c191_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 07 (original).](../assets/figures/ml_fig_c192_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 07 (original).](../assets/figures/ml_fig_c193_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 07 (original).](../assets/figures/ml_fig_c194_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 07 (original).](../assets/figures/ml_fig_c195_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 07 (original).](../assets/figures/ml_fig_c196_07.png)
*Figure — Polynomial basis explosion. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 07 (original).](../assets/figures/ml_fig_c197_07.png)
*Figure — Spline knot placement. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 07 (original).](../assets/figures/ml_fig_c198_07.png)
*Figure — Feature crosses cardinality. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 07 (original).](../assets/figures/ml_fig_c199_07.png)
*Figure — Hashing vs embedding IDs. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 07 (original).](../assets/figures/ml_fig_c200_07.png)
*Figure — Ordinal encoding ranks. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 07 (original).](../assets/figures/ml_fig_c201_07.png)
*Figure — Temporal join leakage timeline. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 07 (original).](../assets/figures/ml_fig_c202_07.png)
*Figure — Out-of-fold target encoding. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 07 (original).](../assets/figures/ml_fig_c203_07.png)
*Figure — Mutual information feature ranking. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 07 (original).](../assets/figures/ml_fig_c204_07.png)
*Figure — Feature cross cardinality risk. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 07 (original).](../assets/figures/ml_fig_c205_07.png)
*Figure — Weight of evidence IV bins. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 07 (original).](../assets/figures/ml_fig_c206_07.png)
*Figure — Cyclic hour sin-cos features. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 07 (original).](../assets/figures/ml_fig_c207_07.png)
*Figure — Leave-one-out target encoding. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 07 (original).](../assets/figures/ml_fig_c208_07.png)
*Figure — Hashing trick collision rates. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 07 (original).](../assets/figures/ml_fig_c209_07.png)
*Figure — Ordered target statistics encoding. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 07 (original).](../assets/figures/ml_fig_c210_07.png)
*Figure — Recursive feature elimination path. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 07 (original).](../assets/figures/ml_fig_c211_07.png)
*Figure — Polynomial feature expansion map. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 07 (original).](../assets/figures/ml_fig_c212_07.png)
*Figure — Equal-width vs equal-frequency bins. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 07 (original).](../assets/figures/ml_fig_c213_07.png)
*Figure — Monotone WoE bin path. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 07 (original).](../assets/figures/ml_fig_c214_07.png)
*Figure — Frequency encoding tall head. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 07 (original).](../assets/figures/ml_fig_c215_07.png)
*Figure — Count vectorizer term-document. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 07 (original).](../assets/figures/ml_fig_c216_07.png)
*Figure — Helmert contrast coding matrix. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 07 (original).](../assets/figures/ml_fig_c217_07.png)
*Figure — Smoothed target mean encoding. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 07 (original).](../assets/figures/ml_fig_c218_07.png)
*Figure — Long-tail rare category ranks. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 07 (original).](../assets/figures/ml_fig_c219_07.png)
*Figure — Leave-group-out CV folds. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 07 (original).](../assets/figures/ml_fig_c220_07.png)
*Figure — Entity embedding scatter. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 07 (original).](../assets/figures/ml_fig_c221_07.png)
*Figure — Out-of-fold target encoding. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 07 (original).](../assets/figures/ml_fig_c222_07.png)
*Figure — Feature hashing collision bins. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 07 (original).](../assets/figures/ml_fig_c223_07.png)
*Figure — WoE and IV bin chart. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 07 (original).](../assets/figures/ml_fig_c224_07.png)
*Figure — Equal-width vs equal-freq bins. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 07 (original).](../assets/figures/ml_fig_c225_07.png)
*Figure — Mutual information feature matrix. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 07 (original).](../assets/figures/ml_fig_c226_07.png)
*Figure — CatBoost ordered target stats. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 07 (original).](../assets/figures/ml_fig_c227_07.png)
*Figure — Embedding table row lookup. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 07 (original).](../assets/figures/ml_fig_c228_07.png)
*Figure — Polynomial feature blow-up. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 07 (original).](../assets/figures/ml_fig_c229_07.png)
*Figure — Target leakage time boundary. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 07 (original).](../assets/figures/ml_fig_c230_07.png)
*Figure — Leave-one-group validation. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 07 (original).](../assets/figures/ml_fig_c231_07.png)
*Figure — Count encoding frequencies. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 07 (original).](../assets/figures/ml_fig_c232_07.png)
*Figure — Hash collision vs dimension. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 07 (original).](../assets/figures/ml_fig_c233_07.png)
*Figure — Hash dim collision tradeoff. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 07 (original).](../assets/figures/ml_fig_c234_07.png)
*Figure — Monotone WoE bin heights. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 07 (original).](../assets/figures/ml_fig_c235_07.png)
*Figure — Bloom filter false-positive rate. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 07 (original).](../assets/figures/ml_fig_c236_07.png)
*Figure — IV contribution bars. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 07 (original).](../assets/figures/ml_fig_c237_07.png)
*Figure — MinHash Jaccard estimate error. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 07 (original).](../assets/figures/ml_fig_c238_07.png)
*Figure — Chi-square bin bars. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 07 (original).](../assets/figures/ml_fig_c239_07.png)
*Figure — LSH band collision rate. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 07 (original).](../assets/figures/ml_fig_c240_07.png)
*Figure — KS statistic bars. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 07 (original).](../assets/figures/ml_fig_c241_07.png)
*Figure — SimHash Hamming collision rate. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 07 (original).](../assets/figures/ml_fig_c242_07.png)
*Figure — PSI drift bin bars. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 07 (original).](../assets/figures/ml_fig_c243_07.png)
*Figure — b-bit MinHash collision. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 07 (original).](../assets/figures/ml_fig_c244_07.png)
*Figure — KL divergence bin bars. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 07 (original).](../assets/figures/ml_fig_c245_07.png)
*Figure — Cross-polytope LSH rate. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 07 (original).](../assets/figures/ml_fig_c246_07.png)
*Figure — JSD distribution bin bars. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 07 (original).](../assets/figures/ml_fig_c247_07.png)
*Figure — Angular LSH collision rate. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 07 (original).](../assets/figures/ml_fig_c248_07.png)
*Figure — Hellinger distance bin bars. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 07 (original).](../assets/figures/ml_fig_c249_07.png)
*Figure — p-stable LSH collision rate. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 07 (original).](../assets/figures/ml_fig_c250_07.png)
*Figure — Wasserstein bin bars. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 07 (original).](../assets/figures/ml_fig_c251_07.png)
*Figure — OPH LSH collision rate. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 07 (original).](../assets/figures/ml_fig_c252_07.png)
*Figure — KS two-sample bars. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 07 (original).](../assets/figures/ml_fig_c253_07.png)
*Figure — Super-bit LSH rate. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 07 (original).](../assets/figures/ml_fig_c254_07.png)
*Figure — PSI population shift bars. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 07 (original).](../assets/figures/ml_fig_c255_07.png)
*Figure — Cross-polytope LSH rate. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 07 (original).](../assets/figures/ml_fig_c256_07.png)
*Figure — JSD bin shift bars. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 07 (original).](../assets/figures/ml_fig_c257_07.png)
*Figure — Missing indicator path c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 07 (original).](../assets/figures/ml_fig_c258_07.png)
*Figure — Scaler choice bars c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 07 (original).](../assets/figures/ml_fig_c259_07.png)
*Figure — Rare category group path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 07 (original).](../assets/figures/ml_fig_c260_07.png)
*Figure — Time feature lag path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 07 (original).](../assets/figures/ml_fig_c261_07.png)
*Figure — Cyclical encode path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 07 (original).](../assets/figures/ml_fig_c262_07.png)
*Figure — Text ngram DF bars c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 07 (original).](../assets/figures/ml_fig_c263_07.png)
*Figure — Image aug strength path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 07 (original).](../assets/figures/ml_fig_c264_07.png)
*Figure — Feature selection gain c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 07 (original).](../assets/figures/ml_fig_c265_07.png)
*Figure — Mutual info bars c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 07 (original).](../assets/figures/ml_fig_c266_07.png)
*Figure — Leakage audit path c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 07 (original).](../assets/figures/ml_fig_c267_07.png)
*Figure — Target encode leak path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 07 (original).](../assets/figures/ml_fig_c268_07.png)
*Figure — Hash trick collision path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 07 (original).](../assets/figures/ml_fig_c269_07.png)
*Figure — Polynomial degree path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 07 (original).](../assets/figures/ml_fig_c270_07.png)
*Figure — Binning residual path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 07 (original).](../assets/figures/ml_fig_c271_07.png)
*Figure — WOE transform path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 07 (original).](../assets/figures/ml_fig_c272_07.png)
*Figure — Interaction term bars c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 07 (original).](../assets/figures/ml_fig_c273_07.png)
*Figure — Missing indicator path c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 07 (original).](../assets/figures/ml_fig_c274_07.png)
*Figure — Scaler choice bars c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 07 (original).](../assets/figures/ml_fig_c275_07.png)
*Figure — Rare category group path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 07 (original).](../assets/figures/ml_fig_c276_07.png)
*Figure — Time feature lag path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 07 (original).](../assets/figures/ml_fig_c277_07.png)
*Figure — Cyclical encode path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 07 (original).](../assets/figures/ml_fig_c278_07.png)
*Figure — Text ngram DF bars c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 07 (original).](../assets/figures/ml_fig_c279_07.png)
*Figure — Image aug strength path c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 07 (original).](../assets/figures/ml_fig_c280_07.png)
*Figure — Feature selection gain c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 07 (original).](../assets/figures/ml_fig_c281_07.png)
*Figure — Mutual info bars c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 07 (original).](../assets/figures/ml_fig_c282_07.png)
*Figure — Leakage audit path c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 07 (original).](../assets/figures/ml_fig_c283_07.png)
*Figure — Target encode leak path c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 07 (original).](../assets/figures/ml_fig_c284_07.png)
*Figure — Hash trick collision path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 07 (original).](../assets/figures/ml_fig_c285_07.png)
*Figure — Polynomial degree path c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 07 (original).](../assets/figures/ml_fig_c286_07.png)
*Figure — Binning residual path c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 07 (original).](../assets/figures/ml_fig_c287_07.png)
*Figure — WOE transform path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 07 (original).](../assets/figures/ml_fig_c288_07.png)
*Figure — Interaction term bars c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 07 (original).](../assets/figures/ml_fig_c289_07.png)
*Figure — Missing indicator path c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 07 (original).](../assets/figures/ml_fig_c290_07.png)
*Figure — Scaler choice bars c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 07 (original).](../assets/figures/ml_fig_c291_07.png)
*Figure — Rare category group path c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 07 (original).](../assets/figures/ml_fig_c292_07.png)
*Figure — Time feature lag path c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 07 (original).](../assets/figures/ml_fig_c293_07.png)
*Figure — Cyclical encode path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 07 (original).](../assets/figures/ml_fig_c294_07.png)
*Figure — Text ngram DF bars c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 07 (original).](../assets/figures/ml_fig_c295_07.png)
*Figure — Image aug strength path c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 07 (original).](../assets/figures/ml_fig_c296_07.png)
*Figure — Feature selection gain c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 07 (original).](../assets/figures/ml_fig_c297_07.png)
*Figure — Mutual info bars c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 07 (original).](../assets/figures/ml_fig_c298_07.png)
*Figure — Leakage audit path c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 07 (original).](../assets/figures/ml_fig_c299_07.png)
*Figure — Target encode leak path c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 07 (original).](../assets/figures/ml_fig_c300_07.png)
*Figure — Hash trick collision path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 07 (original).](../assets/figures/ml_fig_c301_07.png)
*Figure — Polynomial degree path c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 07 (original).](../assets/figures/ml_fig_c302_07.png)
*Figure — Binning residual path c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 07 (original).](../assets/figures/ml_fig_c303_07.png)
*Figure — WOE transform path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 07 (original).](../assets/figures/ml_fig_c304_07.png)
*Figure — Interaction term bars c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 07 (original).](../assets/figures/ml_fig_c305_07.png)
*Figure — Missing indicator path c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 07 (original).](../assets/figures/ml_fig_c306_07.png)
*Figure — Scaler choice bars c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 07 (original).](../assets/figures/ml_fig_c307_07.png)
*Figure — Rare category group path c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 07 (original).](../assets/figures/ml_fig_c308_07.png)
*Figure — Time feature lag path c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 07 (original).](../assets/figures/ml_fig_c309_07.png)
*Figure — Cyclical encode path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 07 (original).](../assets/figures/ml_fig_c310_07.png)
*Figure — Text ngram DF bars c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 07 (original).](../assets/figures/ml_fig_c311_07.png)
*Figure — Image aug strength path c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 07 (original).](../assets/figures/ml_fig_c312_07.png)
*Figure — Feature selection gain c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 07 (original).](../assets/figures/ml_fig_c313_07.png)
*Figure — Mutual info bars c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 07 (original).](../assets/figures/ml_fig_c314_07.png)
*Figure — Leakage audit path c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 07 (original).](../assets/figures/ml_fig_c315_07.png)
*Figure — Target encode leak path c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 07 (original).](../assets/figures/ml_fig_c316_07.png)
*Figure — Hash trick collision path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 07 (original).](../assets/figures/ml_fig_c317_07.png)
*Figure — Polynomial degree path c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 07 (original).](../assets/figures/ml_fig_c318_07.png)
*Figure — Binning residual path c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 07 (original).](../assets/figures/ml_fig_c319_07.png)
*Figure — WOE transform path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 07 (original).](../assets/figures/ml_fig_c320_07.png)
*Figure — Interaction term bars c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 07 (original).](../assets/figures/ml_fig_c321_07.png)
*Figure — Missing indicator path c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 07 (original).](../assets/figures/ml_fig_c322_07.png)
*Figure — Scaler choice bars c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 07 (original).](../assets/figures/ml_fig_c323_07.png)
*Figure — Rare category group path c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 07 (original).](../assets/figures/ml_fig_c324_07.png)
*Figure — Time feature lag path c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 07 (original).](../assets/figures/ml_fig_c325_07.png)
*Figure — Cyclical encode path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 07 (original).](../assets/figures/ml_fig_c326_07.png)
*Figure — Text ngram DF bars c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 07 (original).](../assets/figures/ml_fig_c327_07.png)
*Figure — Image aug strength path c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 07 (original).](../assets/figures/ml_fig_c328_07.png)
*Figure — Feature selection gain c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 07 (original).](../assets/figures/ml_fig_c329_07.png)
*Figure — Mutual info bars c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 07 (original).](../assets/figures/ml_fig_c330_07.png)
*Figure — Leakage audit path c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 07 (original).](../assets/figures/ml_fig_c331_07.png)
*Figure — Target encode leak path c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 07 (original).](../assets/figures/ml_fig_c332_07.png)
*Figure — Hash trick collision path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 07 (original).](../assets/figures/ml_fig_c333_07.png)
*Figure — Polynomial degree path c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 07 (original).](../assets/figures/ml_fig_c334_07.png)
*Figure — Binning residual path c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 07 (original).](../assets/figures/ml_fig_c335_07.png)
*Figure — WOE transform path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 07 (original).](../assets/figures/ml_fig_c336_07.png)
*Figure — Interaction term bars c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 07 (original).](../assets/figures/ml_fig_c337_07.png)
*Figure — Missing indicator path c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 07 (original).](../assets/figures/ml_fig_c338_07.png)
*Figure — Scaler choice bars c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 07 (original).](../assets/figures/ml_fig_c339_07.png)
*Figure — Rare category group path c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 07 (original).](../assets/figures/ml_fig_c340_07.png)
*Figure — Time feature lag path c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 07 (original).](../assets/figures/ml_fig_c341_07.png)
*Figure — Cyclical encode path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 07 (original).](../assets/figures/ml_fig_c342_07.png)
*Figure — Text ngram DF bars c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 07 (original).](../assets/figures/ml_fig_c343_07.png)
*Figure — Image aug strength path c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 07 (original).](../assets/figures/ml_fig_c344_07.png)
*Figure — Feature selection gain c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 07 (original).](../assets/figures/ml_fig_c345_07.png)
*Figure — Mutual info bars c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 07 (original).](../assets/figures/ml_fig_c346_07.png)
*Figure — Leakage audit path c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 07 (original).](../assets/figures/ml_fig_c347_07.png)
*Figure — Target encode leak path c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 07 (original).](../assets/figures/ml_fig_c348_07.png)
*Figure — Hash trick collision path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 07 (original).](../assets/figures/ml_fig_c349_07.png)
*Figure — Polynomial degree path c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 07 (original).](../assets/figures/ml_fig_c350_07.png)
*Figure — Binning residual path c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 07 (original).](../assets/figures/ml_fig_c351_07.png)
*Figure — WOE transform path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 07 (original).](../assets/figures/ml_fig_c352_07.png)
*Figure — Interaction term bars c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 07 (original).](../assets/figures/ml_fig_c353_07.png)
*Figure — Missing indicator path c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 07 (original).](../assets/figures/ml_fig_c354_07.png)
*Figure — Scaler choice bars c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 07 (original).](../assets/figures/ml_fig_c355_07.png)
*Figure — Rare category group path c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 07 (original).](../assets/figures/ml_fig_c356_07.png)
*Figure — Time feature lag path c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 07 (original).](../assets/figures/ml_fig_c357_07.png)
*Figure — Cyclical encode path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 07 (original).](../assets/figures/ml_fig_c358_07.png)
*Figure — Text ngram DF bars c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 07 (original).](../assets/figures/ml_fig_c359_07.png)
*Figure — Image aug strength path c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 07 (original).](../assets/figures/ml_fig_c360_07.png)
*Figure — Feature selection gain c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 07 (original).](../assets/figures/ml_fig_c361_07.png)
*Figure — Mutual info bars c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 07 (original).](../assets/figures/ml_fig_c362_07.png)
*Figure — Leakage audit path c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 07 (original).](../assets/figures/ml_fig_c363_07.png)
*Figure — Target encode leak path c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 07 (original).](../assets/figures/ml_fig_c364_07.png)
*Figure — Hash trick collision path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 07 (original).](../assets/figures/ml_fig_c365_07.png)
*Figure — Polynomial degree path c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 07 (original).](../assets/figures/ml_fig_c366_07.png)
*Figure — Binning residual path c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 07 (original).](../assets/figures/ml_fig_c367_07.png)
*Figure — WOE transform path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 07 (original).](../assets/figures/ml_fig_c368_07.png)
*Figure — Interaction term bars c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 07 (original).](../assets/figures/ml_fig_c369_07.png)
*Figure — Missing indicator path c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 07 (original).](../assets/figures/ml_fig_c370_07.png)
*Figure — Scaler choice bars c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 07 (original).](../assets/figures/ml_fig_c371_07.png)
*Figure — Rare category group path c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 07 (original).](../assets/figures/ml_fig_c372_07.png)
*Figure — Time feature lag path c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 07 (original).](../assets/figures/ml_fig_c373_07.png)
*Figure — Cyclical encode path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 07 (original).](../assets/figures/ml_fig_c374_07.png)
*Figure — Text ngram DF bars c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 07 (original).](../assets/figures/ml_fig_c375_07.png)
*Figure — Image aug strength path c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 07 (original).](../assets/figures/ml_fig_c376_07.png)
*Figure — Feature selection gain c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 07 (original).](../assets/figures/ml_fig_c377_07.png)
*Figure — Mutual info bars c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 07 (original).](../assets/figures/ml_fig_c378_07.png)
*Figure — Leakage audit path c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 07 (original).](../assets/figures/ml_fig_c379_07.png)
*Figure — Target encode leak path c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 07 (original).](../assets/figures/ml_fig_c380_07.png)
*Figure — Hash trick collision path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 07 (original).](../assets/figures/ml_fig_c381_07.png)
*Figure — Polynomial degree path c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 07 (original).](../assets/figures/ml_fig_c382_07.png)
*Figure — Binning residual path c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 07 (original).](../assets/figures/ml_fig_c383_07.png)
*Figure — WOE transform path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 07 (original).](../assets/figures/ml_fig_c384_07.png)
*Figure — Interaction term bars c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 07 (original).](../assets/figures/ml_fig_c385_07.png)
*Figure — Missing indicator path c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 07 (original).](../assets/figures/ml_fig_c386_07.png)
*Figure — Scaler choice bars c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 07 (original).](../assets/figures/ml_fig_c387_07.png)
*Figure — Rare category group path c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 07 (original).](../assets/figures/ml_fig_c388_07.png)
*Figure — Time feature lag path c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 07 (original).](../assets/figures/ml_fig_c389_07.png)
*Figure — Cyclical encode path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 07 (original).](../assets/figures/ml_fig_c390_07.png)
*Figure — Text ngram DF bars c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 07 (original).](../assets/figures/ml_fig_c391_07.png)
*Figure — Image aug strength path c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 07 (original).](../assets/figures/ml_fig_c392_07.png)
*Figure — Feature selection gain c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 07 (original).](../assets/figures/ml_fig_c393_07.png)
*Figure — Mutual info bars c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 07 (original).](../assets/figures/ml_fig_c394_07.png)
*Figure — Leakage audit path c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 07 (original).](../assets/figures/ml_fig_c395_07.png)
*Figure — Target encode leak path c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 07 (original).](../assets/figures/ml_fig_c396_07.png)
*Figure — Hash trick collision path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 07 (original).](../assets/figures/ml_fig_c397_07.png)
*Figure — Polynomial degree path c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 07 (original).](../assets/figures/ml_fig_c398_07.png)
*Figure — Binning residual path c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 07 (original).](../assets/figures/ml_fig_c399_07.png)
*Figure — WOE transform path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 07 (original).](../assets/figures/ml_fig_c400_07.png)
*Figure — Interaction term bars c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 07 (original).](../assets/figures/ml_fig_c401_07.png)
*Figure — Missing indicator path c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 07 (original).](../assets/figures/ml_fig_c402_07.png)
*Figure — Scaler choice bars c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 07 (original).](../assets/figures/ml_fig_c403_07.png)
*Figure — Rare category group path c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 07 (original).](../assets/figures/ml_fig_c404_07.png)
*Figure — Time feature lag path c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 07 (original).](../assets/figures/ml_fig_c405_07.png)
*Figure — Cyclical encode path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 07 (original).](../assets/figures/ml_fig_c406_07.png)
*Figure — Text ngram DF bars c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 07 (original).](../assets/figures/ml_fig_c407_07.png)
*Figure — Image aug strength path c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 07 (original).](../assets/figures/ml_fig_c408_07.png)
*Figure — Feature selection gain c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 07 (original).](../assets/figures/ml_fig_c409_07.png)
*Figure — Mutual info bars c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 07 (original).](../assets/figures/ml_fig_c410_07.png)
*Figure — Leakage audit path c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 07 (original).](../assets/figures/ml_fig_c411_07.png)
*Figure — Target encode leak path c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 07 (original).](../assets/figures/ml_fig_c412_07.png)
*Figure — Hash trick collision path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 07 (original).](../assets/figures/ml_fig_c413_07.png)
*Figure — Polynomial degree path c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 07 (original).](../assets/figures/ml_fig_c414_07.png)
*Figure — Binning residual path c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 07 (original).](../assets/figures/ml_fig_c415_07.png)
*Figure — WOE transform path c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 07 (original).](../assets/figures/ml_fig_c416_07.png)
*Figure — Interaction term bars c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 07 (original).](../assets/figures/ml_fig_c417_07.png)
*Figure — Missing indicator path c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 07 (original).](../assets/figures/ml_fig_c418_07.png)
*Figure — Scaler choice bars c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 07 (original).](../assets/figures/ml_fig_c419_07.png)
*Figure — Rare category group path c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 07 (original).](../assets/figures/ml_fig_c420_07.png)
*Figure — Time feature lag path c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 07 (original).](../assets/figures/ml_fig_c421_07.png)
*Figure — Cyclical encode path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 07 (original).](../assets/figures/ml_fig_c422_07.png)
*Figure — Text ngram DF bars c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 07 (original).](../assets/figures/ml_fig_c423_07.png)
*Figure — Image aug strength path c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 07 (original).](../assets/figures/ml_fig_c424_07.png)
*Figure — Feature selection gain c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 07 (original).](../assets/figures/ml_fig_c425_07.png)
*Figure — Mutual info bars c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 07 (original).](../assets/figures/ml_fig_c426_07.png)
*Figure — Leakage audit path c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 07 (original).](../assets/figures/ml_fig_c427_07.png)
*Figure — Target encode leak path c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 07 (original).](../assets/figures/ml_fig_c428_07.png)
*Figure — Hash trick collision path c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 07 (original).](../assets/figures/ml_fig_c429_07.png)
*Figure — Polynomial degree path c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 07 (original).](../assets/figures/ml_fig_c430_07.png)
*Figure — Binning residual path c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 07 (original).](../assets/figures/ml_fig_c431_07.png)
*Figure — WOE transform path c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 07 (original).](../assets/figures/ml_fig_c432_07.png)
*Figure — Interaction term bars c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 07 (original).](../assets/figures/ml_fig_c433_07.png)
*Figure — Missing indicator path c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 07 (original).](../assets/figures/ml_fig_c434_07.png)
*Figure — Scaler choice bars c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 07 (original).](../assets/figures/ml_fig_c435_07.png)
*Figure — Rare category group path c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 07 (original).](../assets/figures/ml_fig_c436_07.png)
*Figure — Time feature lag path c436. Synthetic teaching geometry—not a causal claim.*


![c437 teaching panel 07 (original).](../assets/figures/ml_fig_c437_07.png)
*Figure — Cyclical encode path c437. Synthetic teaching geometry—not a causal claim.*


![c438 teaching panel 07 (original).](../assets/figures/ml_fig_c438_07.png)
*Figure — Text ngram DF bars c438. Synthetic teaching geometry—not a causal claim.*


![c439 teaching panel 07 (original).](../assets/figures/ml_fig_c439_07.png)
*Figure — Image aug strength path c439. Synthetic teaching geometry—not a causal claim.*


![c440 teaching panel 07 (original).](../assets/figures/ml_fig_c440_07.png)
*Figure — Feature selection gain c440. Synthetic teaching geometry—not a causal claim.*


![c441 teaching panel 07 (original).](../assets/figures/ml_fig_c441_07.png)
*Figure — Mutual info bars c441. Synthetic teaching geometry—not a causal claim.*


![c442 teaching panel 07 (original).](../assets/figures/ml_fig_c442_07.png)
*Figure — Leakage audit path c442. Synthetic teaching geometry—not a causal claim.*


![c443 teaching panel 07 (original).](../assets/figures/ml_fig_c443_07.png)
*Figure — Target encode leak path c443. Synthetic teaching geometry—not a causal claim.*


![c444 teaching panel 07 (original).](../assets/figures/ml_fig_c444_07.png)
*Figure — Hash trick collision path c444. Synthetic teaching geometry—not a causal claim.*


![c445 teaching panel 07 (original).](../assets/figures/ml_fig_c445_07.png)
*Figure — Polynomial degree path c445. Synthetic teaching geometry—not a causal claim.*


![c446 teaching panel 07 (original).](../assets/figures/ml_fig_c446_07.png)
*Figure — Binning residual path c446. Synthetic teaching geometry—not a causal claim.*


![c447 teaching panel 07 (original).](../assets/figures/ml_fig_c447_07.png)
*Figure — WOE transform path c447. Synthetic teaching geometry—not a causal claim.*


![c448 teaching panel 07 (original).](../assets/figures/ml_fig_c448_07.png)
*Figure — Interaction term bars c448. Synthetic teaching geometry—not a causal claim.*


![c449 teaching panel 07 (original).](../assets/figures/ml_fig_c449_07.png)
*Figure — Missing indicator path c449. Synthetic teaching geometry—not a causal claim.*


![c450 teaching panel 07 (original).](../assets/figures/ml_fig_c450_07.png)
*Figure — Scaler choice bars c450. Synthetic teaching geometry—not a causal claim.*


![c451 teaching panel 07 (original).](../assets/figures/ml_fig_c451_07.png)
*Figure — Rare category group path c451. Synthetic teaching geometry—not a causal claim.*


![c452 teaching panel 07 (original).](../assets/figures/ml_fig_c452_07.png)
*Figure — Time feature lag path c452. Synthetic teaching geometry—not a causal claim.*


![c453 teaching panel 07 (original).](../assets/figures/ml_fig_c453_07.png)
*Figure — Cyclical encode path c453. Synthetic teaching geometry—not a causal claim.*


![c454 teaching panel 07 (original).](../assets/figures/ml_fig_c454_07.png)
*Figure — Text ngram DF bars c454. Synthetic teaching geometry—not a causal claim.*


![c455 teaching panel 07 (original).](../assets/figures/ml_fig_c455_07.png)
*Figure — Image aug strength path c455. Synthetic teaching geometry—not a causal claim.*


![c456 teaching panel 07 (original).](../assets/figures/ml_fig_c456_07.png)
*Figure — Feature selection gain c456. Synthetic teaching geometry—not a causal claim.*


![c457 teaching panel 07 (original).](../assets/figures/ml_fig_c457_07.png)
*Figure — Mutual info bars c457. Synthetic teaching geometry—not a causal claim.*


![c458 teaching panel 07 (original).](../assets/figures/ml_fig_c458_07.png)
*Figure — Leakage audit path c458. Synthetic teaching geometry—not a causal claim.*


![c459 teaching panel 07 (original).](../assets/figures/ml_fig_c459_07.png)
*Figure — Target encode leak path c459. Synthetic teaching geometry—not a causal claim.*


![c460 teaching panel 07 (original).](../assets/figures/ml_fig_c460_07.png)
*Figure — Hash trick collision path c460. Synthetic teaching geometry—not a causal claim.*


![c461 teaching panel 07 (original).](../assets/figures/ml_fig_c461_07.png)
*Figure — Polynomial degree path c461. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Feature engineering designs the matrix models actually learn from. Selection methods include filters (fast statistical scores), wrappers (SFS, SBS, genetic search optimizing a model’s validation metric), and embedded approaches (L1, tree importance). Numerical pipelines use min–max scaling, z-score standardization, L1/L2 norms, and log/Box–Cox transforms—always fit on train. Categorical encodings include one-hot, dummy, effect coding, feature hashing, bin counting, and target encoding; each invents geometry and some leak labels if misused. Text features span bags-of-words, subwords, n-grams, POS tags, Word2Vec/GloVe/FastText embeddings, and keyword methods (TF–IDF, TextRank, RAKE, YAKE). Image descriptors include Harris corners, MSER, HOG, SIFT, and watershed segments; video adds motion vectors, optical flow, 3D CNNs, and graph-based pose/region models. Time series and signals contribute stationarity checks, seasonal and trend components, motifs, lags, change points, and carefully chosen smoothers. Missingness is itself a feature decision: whether a blank is MCAR, MAR, or MNAR dictates whether to impute, impute-and-flag, or model the absence, and an informative-missingness indicator is lawful only when it is knowable at t₀. In clinical data, leakage is the cardinal sin: features must be knowable at decision time, and pipelines must enforce fit-on-train discipline end to end.

## Practice and Reflection

(1) Compare filter, wrapper, and embedded selection for a 200-feature stroke registry with n = 800. Propose a nested CV scheme and justify computational trade-offs.

(2) Given ages {62, 71, 55, 80}, compute min–max scaled values to [0,1] and z-scores using population σ for illustration. Explain what changes if a new age 95 arrives at test time.

(3) Show algebraically why dummy coding with an intercept is full rank when one level is dropped, while full one-hot plus intercept is singular.

(4) Write a target-encoding formula with smoothing and describe an out-of-fold procedure that avoids using a row’s own label.

(5) Contrast Word2Vec skip-gram, GloVe, and FastText for rare medication strings in clinical notes.

(6) Explain how SIFT achieves scale and rotation robustness at a high level, and when a small labeled stroke CT dataset might still prefer a pretrained CNN embedding.

(7) For hourly mean arterial pressure series, define one lag feature set, one seasonal feature, and one change-point feature usable at time t without future leakage.

(8) Design a leakage audit checklist for a model that claims to predict 90-day mRS from ED arrival data at a comprehensive stroke center.

(9) When would you choose RAKE or YAKE over TF–IDF for keyword features from a single radiology report?

(10) Describe how optical flow features differ from 3D CNN features for detecting abnormal movement on bedside monitoring video.

(11) Classify each blank as MCAR, MAR, or MNAR and give its feature-engineering implication: (a) glucose panels voided one afternoon by a lab-analyzer outage; (b) troponin never ordered because the patient appeared clinically well; (c) 90-day mRS blank because the follow-up window has not yet elapsed. For each, state whether a missingness indicator is a legal feature for a model that predicts at ED arrival.

(12) Using target encoding with global rate ȳ = 0.20 and prior m = 10, compute the smoothed encoding for a site with 8 patients and 3 events, then give the leave-one-out encoding seen by one of its patients whose outcome is y = 1. Explain which number the model may legally train on.
