# Chapter 1. Basic Concepts of Machine Learning and Artificial Intelligence


![01 Gradient Descent](../assets/figures/01_gradient_descent.png)


## Opening
![Learning curves vs sample size (synthetic; original).](../assets/figures/ml_fig_learning_curves.png)

*Learning curves vs sample size (synthetic; original).*


An overnight resident asks whether ‘AI’ can rule out stroke on non-contrast CT. Before architecture debates begin, this chapter forces the supervised vs unsupervised vs reinforcement taxonomy and the prediction-versus-causation boundary that stroke teams violate most often.


![Supervised versus unsupervised learning paths (original teaching graphic).](../assets/figures/ml_fig_supervised_unsupervised_map.png)

*Supervised versus unsupervised learning paths (original teaching graphic).*

![Train / validation / test split along a clinical timeline (original).](../assets/figures/ml_fig_train_val_test.png)

*Train / validation / test split along a clinical timeline (original).*
## Learning Objectives

Place modern machine learning in a short historical arc from symbolic AI through statistical learning to large-scale generative systems.

Distinguish artificial intelligence, machine learning, data mining, and data science with operational project-scoping language.

Define an algorithm and evaluate learning systems along computational cost, predictive accuracy, and the accuracy–efficiency trade-off.

Recognize major dataset modalities—tabular, temporal, streams, graphs, text, images/video, audio—and common numerical types.

Map unsupervised, supervised, self-supervised, generative, and reinforcement tasks to clinical and epidemiologic problem statements.

Explain ground-truth construction and k-fold cross-validation as disciplines against optimistic evaluation.

Execute a step-by-step data workflow from question framing through monitoring under distribution shift.

Connect cohort design, index time, phenotype quality, and external validation to honest ML claims in neurology and population health.

Diagnose overfitting and underfitting, and reason about inductive bias, model capacity, and the bias–variance trade-off.

Treat fairness, privacy, and transparency as first-class design constraints in the ML workflow, previewed here and deepened in Chapter 16.

## What This Chapter Is About

Machine learning (ML) is the engineering practice of writing programs that improve task performance by processing data, rather than by hand-coding every decision rule. Artificial intelligence (AI) is the broader scientific and engineering project of building systems that perform tasks associated with intelligent behavior—perception, language, planning, and decision-making under uncertainty. Data science sits at the intersection of statistics, computing, and domain expertise, emphasizing measurement, data quality, analysis, and communication of uncertainty to decision-makers. Data mining historically names the discovery of patterns in large databases, often with an emphasis on scalable algorithms for association, clustering, and anomaly detection. These labels overlap heavily in practice, but confusing them produces bad project design: you may train a fashionable neural network when you needed a measurement plan, or you may build a dashboard when you needed a learning algorithm that generalizes.

This chapter builds a shared vocabulary for the rest of the book. We sketch a short history of AI and ML; separate data science, machine learning, data mining, and AI; define what an algorithm is; and show how to evaluate learning systems by computational complexity, runtime, accuracy, and the balance between accuracy and efficiency. We survey types of input data you will meet in clinics and registries, and the families of tasks ML can perform. We then treat ground-truth labels and k-fold cross-validation as the core of honest supervised evaluation, and close with a step-by-step data workflow. A fully worked numerical model-selection example and extended clinical–epidemiologic notes for stroke and population research anchor abstractions in practice.

If you remember only one principle from this chapter, let it be this: learning is not fitting the past—it is preparing to predict and act under uncertainty about the future, using assumptions you can name and evaluations you can defend, including the epidemiologic assumptions that define who is studied and when information is allowed to enter the model.

## A Short History of Artificial Intelligence and Machine Learning

The ambition to automate intelligent behavior is older than digital computers. Mechanical calculators, logical automata, and early cybernetics already framed control and feedback as computational problems. The mid-twentieth century crystallized AI as a research field: the 1956 Dartmouth workshop popularized the name ‘artificial intelligence,’ and the following decades explored symbolic reasoning, theorem proving, game-playing search, and knowledge representation. Early optimism about general intelligence met hard limits when combinatorial search exploded and hand-built knowledge bases proved brittle outside narrow domains. Those winters of funding and expectation did not erase progress; they redirected effort toward methods that scale with data rather than with hand-authored rules.

Statistical pattern recognition and adaptive filtering developed in parallel, often outside the AI brand. Linear discriminants, nearest-neighbor rules, and early neural networks (perceptrons and multilayer networks after backpropagation became practical) treated classification and regression as estimation problems. The 1980s and 1990s brought decision trees, ensemble methods, support vector machines, probabilistic graphical models, and a mature theory of generalization. Data mining emerged as organizations accumulated large transactional databases: association rules, clustering, and scalable mining algorithms became industrial tools. By the 2000s, internet-scale data, cheaper compute, and improved optimization made high-capacity models practical for speech, vision, and ranking.

The 2010s deep-learning wave combined multilayer neural networks, large labeled (and later unlabeled) corpora, GPUs, and careful engineering of architectures—convolutions for images, sequence models and then transformers for language. Self-supervised pretraining and generative models (including large language models and diffusion-based image generators) shifted practice again: systems learn rich representations from vast unlabeled text and media, then adapt with comparatively little task-specific supervision. Reinforcement learning matured in games and control, and hybrid systems now mix learned modules with retrieval, tools, and classical planners.

For clinical readers, the historical lesson is pragmatic. Symbolic AI still lives in clinical decision-support rules and guideline engines. Statistical ML underwrites risk scores and phenotyping models. Deep networks power imaging and waveform analysis. Generative systems draft notes and summarize literature—with new failure modes around hallucination and provenance. No single era abolished the previous ones. Choose methods for problem structure, data size, interpretability needs, and deployment constraints, not for fashion.

1950s–1970s: symbolic AI, search, knowledge representation; early adaptive systems.

1980s–1990s: statistical learning, trees, kernels, graphical models; industrial data mining.

2000s: web-scale data, ensembles, structured prediction, practical speech/vision pipelines.

2010s–present: deep learning, self-supervision, transformers, generative AI, hybrid tool-using agents.

Clinical implication: method choice should follow estimand, data, and safety—not hype cycles.

## Data Science, Machine Learning, Data Mining, and Artificial Intelligence

Artificial intelligence as a goal-oriented field asks how to build agents or systems that achieve goals in complex environments. Classical AI emphasized symbolic reasoning, search, knowledge representation, and planning. Modern AI practice also includes statistical learning, large-scale optimization, and systems that combine learned modules with hand-designed structure (for example, a retrieval component feeding a language model, or a rules engine that constrains unsafe model outputs). Saying a product is ‘AI-powered’ is marketing language; saying a system solves a task by combining search, learned perception, and a policy is engineering language. Prefer the latter when scoping work and when reviewing claims.

Machine learning is a subfield of AI—and of statistics and computer science—concerned with algorithms that adjust free parameters so that a performance measure improves on data. Formally, a common abstract setup is: given a hypothesis class H of candidate models, a loss function ℒ measuring error, and a dataset D drawn from some unknown process, find a hypothesis h in H that yields low expected loss on future samples from the same process. The key phrase is expected future performance. Fitting D alone is not enough; generalization is the point. Not all AI is ML: rule engines, constraint solvers, and pure search can be intelligent without learning. Not all ML is deep learning: linear models, trees, nearest neighbors, and clustering remain central tools.

Data mining traditionally emphasizes discovering previously unknown patterns in large databases—frequent itemsets, sequential patterns, clusters, and anomalies—often with a systems focus on scale, indexing, and disk- or stream-efficient algorithms. Machine learning and data mining share algorithms (clustering appears in both literatures), but mining often prioritizes exploratory discovery and descriptive patterns, whereas ML curricula often prioritize predictive generalization and formal risk minimization. In health systems, mining might surface co-prescription patterns; ML might predict 90-day functional outcome from admission features. Both require careful causal and privacy thinking before operational use.

Data science is the broader practice of defining questions, collecting and cleaning data, exploratory analysis, modeling (sometimes with ML), and communicating uncertainty to decision-makers. A data scientist may never train a neural network and still deliver decisive value by fixing a broken phenotype definition or by redesigning a sampling frame. Conversely, a modeler who ignores measurement error and cohort construction is not practicing mature data science. In short: AI names the ambition of intelligent systems; ML names a family of learning algorithms; data mining names scalable pattern discovery; data science names the measurement-to-decision pipeline that should govern them all.

### Why data science matters

Data science matters because modern decisions—clinical, operational, and scientific—are made under partial observation. Electronic health records, claims, registries, sensors, and imaging produce high volume and high messiness. Without disciplined measurement, the same database can support contradictory conclusions depending on inclusion criteria, missing-data handling, and metric choice. Data science is important precisely because raw data are not evidence until linked to a clear estimand: what population, what time zero, what outcome definition, what decision the number will change.

In stroke care and cerebrovascular epidemiology, data science underwrites surveillance of incidence and outcomes, quality improvement for door-to-needle and door-to-puncture times, risk adjustment for comparative reporting, and transport of models across hospitals with different coding intensity. A high-capacity predictor without a data-science backbone will encode documentation habits as if they were biology. A strong data-science process without any learning algorithm may still outperform a careless deep model by simply defining the right cohort and the right endpoint.

### What is an algorithm?

An algorithm is a finite, unambiguous procedure that transforms inputs into outputs through a sequence of well-defined steps. Algorithms must terminate (in theory, or with explicit stopping rules in practice), must specify control flow (branches, loops, recursion), and must operate on a defined data representation. Sorting a list, computing a sample mean, running gradient descent for a fixed number of epochs, and assigning each patient to the nearest cluster center are algorithms. ‘Use clinical judgment’ is not an algorithm until the judgment is operationalized into reproducible steps—or explicitly left as a human in the loop with documented authority.

Learning algorithms are algorithms whose outputs include estimated parameters, structures, or policies derived from data. The training procedure (for example, minimize average logistic loss plus L2 penalty via L-BFGS) is itself an algorithm; the fitted model is the artifact that will be used at inference. Distinguishing training-time algorithms from inference-time algorithms matters for latency, audit, and regulatory documentation: a hospital may retrain monthly offline while requiring millisecond inference at order entry.

AI: systems that act intelligently toward goals (may or may not learn from data).

ML: algorithms that improve task performance using data and a performance criterion.

Data mining: scalable discovery of patterns (associations, clusters, anomalies) in large stores.

Data science: measurement, analysis, modeling, and communication for decisions under uncertainty.

Algorithm: finite, unambiguous procedure from inputs to outputs; learning algorithms estimate models from data.

## Evaluating a Machine Learning Algorithm

Evaluating a learning system is multi-criteria. Stakeholders care whether predictions are accurate enough for the decision, whether the system runs within latency and cost budgets, whether it is stable under data shift, and whether its failures are detectable. This section focuses on computational complexity and efficiency, algorithms’ runtime in practice, predictive accuracy and related metrics, and the unavoidable trade-off between accuracy and efficiency.

### Computational complexity and efficiency

Computational complexity theory classifies algorithms by how resource use grows as input size grows. Common resources are time (number of elementary operations) and space (memory). Big-O notation hides constants and lower-order terms to expose asymptotic growth: an algorithm that takes roughly a·n² + b·n operations is O(n²). For ML, n often denotes the number of training examples, d the number of features, k the number of clusters or classes, and T the number of training iterations or epochs.

Training complexity and inference complexity differ. Training a linear model with closed-form least squares is dominated by forming and solving normal equations—roughly O(n d² + d³) in the naive dense case—while inference is a single dot product O(d) per example. Training a deep network may require many passes over n examples with costly gradient steps; inference is a single forward pass, still nontrivial for large models on edge devices. Complexity statements are models of cost, not stopwatches: cache effects, vectorization, GPU parallelism, and sparse structure can reorder practical winners.

### Algorithms’ runtime in practice

Runtime is the wall-clock or CPU time observed on particular hardware, software, and data. Profiling beats guesswork: I/O, data preprocessing, and evaluation loops often dominate the nominal learning update. For streaming or bedside deployment, tail latency (for example, 99th percentile response time) matters more than mean latency. For batch epidemiology on a registry extract, total job time and memory ceiling matter more than per-row microseconds.

Scalability questions you should ask explicitly: Does runtime grow linearly with n, or worse? Can the method use minibatches or distributed training? Does it require a full pairwise distance matrix (O(n²) memory)—fatal at n = 10⁶ without approximation? Can you precompute features once and reuse them across model candidates? In regulated clinical software, reproducibility of runtime environments (library versions, random seeds, hardware class) is part of validation, not an afterthought.

### Accuracy and related performance measures

Accuracy in the narrow sense is the fraction of correct predictions among all predictions. It is intuitive and often misleading under class imbalance: a model that always predicts ‘no seizure’ on an EEG stream with 1% seizure windows achieves 99% accuracy while missing every event. Prefer metrics aligned with decision costs. Classification commonly uses precision, recall (sensitivity), specificity, F1, Matthews correlation, AUROC, and average precision. Regression uses mean squared error, mean absolute error, and calibration of predicted means. Ranking uses NDCG or MAP. Probabilistic forecasts need calibration metrics (Brier score, reliability diagrams) in addition to discrimination.

Always separate the training loss (the objective optimized by the learning algorithm) from the reporting metric (what clinicians and quality committees care about). They need not match. Also report uncertainty when possible—confidence intervals, bootstrap ranges—and slice metrics by site, age band, and other prespecified subgroups. A single scalar on a mixed test set can hide catastrophic failure on posterior circulation strokes or on transferred patients.

### The balance between accuracy and efficiency

Higher capacity and heavier computation often buy accuracy—until they buy overfitting, fragility, or unaffordable latency. The accuracy–efficiency frontier is the set of models for which you cannot improve one axis without harming the other under fixed data and deployment constraints. In prehospital triage, a compact logistic score computable on paper may dominate a large ensemble that needs a GPU and a network round-trip. In offline research phenotyping, a slower but better-calibrated model may be fine.

Practical levers include: feature selection and dimensionality reduction; model compression and distillation; early-exit classifiers; approximate nearest neighbors; limiting context length; and cascades that run a cheap model first and a expensive model only on uncertain cases. Document the operating point you choose: ‘We accept a 1-point drop in AUROC to meet a 50 ms p95 latency budget’ is an engineering decision; pretending no trade-off exists is not.

![Train–serve skew: door-to-CT distribution shift and feature PSI (synthetic; original).](../assets/figures/ml_fig_train_serve_skew.png)

*Figure — Same weights, different input law. **Left:** training-window door-to-CT vs a faster serve window after a pathway change. **Right:** binned proportions and feature PSI quantifying the shift. Monitor features (and scores) in production—not only lagged outcomes. Train–serve skew is a first-class failure mode, not an afterthought.*

Complexity: asymptotic growth of time/memory with n, d, k, iterations.

Runtime: measured latency and throughput on real hardware, including preprocessing.

Accuracy: task metrics aligned with costs; not accuracy alone under imbalance.

Trade-off: choose operating points on the accuracy–efficiency frontier deliberately.

## Types of Input Datasets

Machine learning methods assume structure in their inputs. Matching method to data type is as important as matching method to task. This section surveys major dataset modalities you will encounter in clinical and epidemiologic work, and closes with numerical data types that cut across modalities.

![1.5: Six common data modalities in clinical machine learning: tabular records, temporal signals, graphs, text tokens (with a ](../assets/figures/ml_concept_1.5_afbc1aeb.png)

*Figure 1.5 — original teaching graphic.*

### Tabular datasets

Tabular data are rows (examples, often patients or encounters) and columns (features and labels). Registries, claims extracts, and many EHR analytic tables are tabular. Strengths include mature tools (SQL, data frames, gradient-boosted trees) and relatively transparent feature semantics. Challenges include missingness, mixed continuous/categorical types, leakage from future-dated fields, and hierarchical structure (multiple encounters per patient, multiple hospitals per system) that simple row-wise i.i.d. assumptions ignore. Tree ensembles and regularized linear models remain strong baselines on medium-sized clinical tables.

### Temporal datasets

Temporal data record ordered measurements over time: vital-sign trajectories, serial NIHSS, longitudinal mRS, claims spells, or clinic visits. Time introduces autocorrelation, irregular sampling, censoring, and competing risks. Methods range from classical time-series models and survival analysis to recurrent and attention-based sequence models. Critical design choices include index time, look-back windows, and whether prediction is at a fixed horizon (90-day death) or dynamic (risk updating each hospital day).

### Data streams

Streams are unbounded or continuously arriving sequences where storing everything forever is impossible or undesirable: bedside monitors, clickstreams, continuous EEG alerts, hospital ADT feeds. Stream learning emphasizes constant-time or logarithmic updates, concept drift detection, and sliding or fading windows. Evaluation must respect time order; random shuffles destroy causality and leak the future. Operational metrics include false alarms per hour and time-to-detection, not only offline AUC.

### Graph datasets

Graphs represent entities as nodes and relationships as edges: referral networks among hospitals, patient–provider bipartite graphs, molecular graphs, knowledge graphs of diseases and drugs, or brain connectivity graphs. Learning tasks include node classification, link prediction, community detection, and graph-level prediction. Graph structure encodes dependence that tabular flattening destroys; conversely, naive graph models can leak labels through neighborhood structure if splits are not designed carefully.

### Text

Clinical text includes notes, radiology reports, discharge summaries, and literature. Representation evolved from bag-of-words and TF–IDF to word embeddings and transformer language models. Tasks include classification (asserted stroke etiology in a note), named-entity recognition, relation extraction, summarization, and retrieval. De-identification, section structure, abbreviation sense, and copy-forward artifacts are health-specific hazards. Labels derived from notes can circularly encode the same text features a model uses—another leakage pathway.

### Image and video

Images and videos dominate radiology, pathology, ophthalmology, and procedural recordings. Pixel grids are high-dimensional and spatially correlated; convolutional networks and vision transformers exploit that structure. Clinical imaging adds DICOM metadata, scanner and protocol shift, 3-D volumes (CT/MRI), and video temporal context (angiography, ultrasound). Annotation is expensive; weak labels from reports and semi-supervised methods are common. Evaluation should include operating-point metrics relevant to triage versus definitive read, and subgroup performance by device vendor when possible.

### Audio

Audio includes speech, heart or lung sounds, and Doppler signals. Pipelines often transform waveforms to spectrograms and then apply vision-like or sequence models. Speech systems support dictation and conversational agents; signal audio supports diagnostic assist. Privacy is acute for voice. Noise, microphone variability, and language or accent shift degrade transportability—familiar themes under a different sensor.

### Numerical data types

Across modalities, atomic fields have types that constrain legitimate operations. Binary indicators (0/1) encode presence/absence. Nominal categoricals (TOAST subtype labels as names) have no order; ordinal categoricals (mRS 0–6) have order without equal intervals. Integer counts (arrivals per day) differ from continuous reals (serum creatinine). Intervals and ratios differ by whether zero is meaningful. Datetimes and durations need explicit time-zone and censoring rules. IDs are identifiers, not magnitudes—never average medical record numbers. Confusing types produces nonsense features (subtracting two nominal codes) and wrong models (linear regression on multi-level nominals without encoding).

Tabular: rows/columns; trees and linear models excel; watch leakage and hierarchy.

Temporal: ordered measures; define index time and horizons carefully.

Streams: continuous arrival; respect order; monitor drift and alarm rates.

Graphs: nodes/edges; dependence and split design are central.

Text/image/video/audio: high-dimensional sensors; annotation cost and shift dominate.

Numerical types: binary, nominal, ordinal, count, continuous, time—encode accordingly.

## Tasks That Machine Learning Can Perform

Learning problems differ by the kind of feedback available and by the form of the desired output. We organize tasks into unsupervised structure discovery, supervised prediction, self-supervised representation learning, generative modeling, and reinforcement learning. Hybrids are common; the taxonomy still helps you write a precise problem statement.

![1.2: A taxonomy of machine-learning tasks, splitting supervised learning into regression and classification and unsupervised ](../assets/figures/ml_concept_1.2_235226c2.png)

*Figure 1.2 — original teaching graphic.*

### Unsupervised learning: clustering

Clustering partitions unlabeled examples into groups of similar items under a distance or probabilistic model. Use clustering for phenotype exploration, cohort stratification, and anomaly review—not as automatic discovery of causal disease entities. Methods include k-means, hierarchical clustering, density-based clustering, and mixture models (Chapter 4). Evaluation is hard without labels; stability and external validation matter more than a pretty heatmap.

### Association rules and sequence mining

Association rule mining finds co-occurring itemsets (if A and B then C) with support and confidence statistics—classically in market baskets, also in co-prescription and comorbidity patterns. Sequence mining finds ordered patterns (A before B before C) in event streams such as care pathways. These methods are descriptive; high confidence is not causation. Multiple testing and clinically trivial rules are endemic; domain filters and prospective validation are required before process change.

### Dimensionality reduction

Dimensionality reduction maps high-dimensional x to a lower-dimensional z that preserves structure relevant for visualization, compression, noise reduction, or downstream modeling. Linear methods include PCA and factor analysis; nonlinear embeddings include kernel PCA, t-SNE, and UMAP (use the latter carefully—they can invent clusters). Autoencoders learn nonlinear compressions. In imaging and multi-omics, reduction is often mandatory before classical clustering or regression.

### Deviation and anomaly detection

Anomaly detection flags points or sequences that deviate from a notion of normal: rare imaging artifacts, impossible vital-sign combinations, fraud, or novel presentations. Methods include statistical thresholds, isolation forests, one-class SVMs, autoencoder reconstruction error, and density-based noise labels (as in DBSCAN). In clinical ops, precision of alerts and workload dominate; a detector with excellent retrospective AUC but 50 false pages per night will be disabled.

### Supervised learning: regression and correlation

Regression predicts a continuous (or conditionally continuous) target from features: length of stay, systolic blood pressure, infarct volume. Linear regression, generalized linear models, penalized regression, Gaussian processes, and neural regressors are options. Correlation quantifies association strength without necessarily producing a full predictive model; it is exploratory and vulnerable to confounding. Supervised regression estimates 𝔼[Y | X] or a full conditional distribution; correlation alone does not specify how to predict new cases under feature shift.

![Partial dependence is not a causal effect: observational PDP vs do(X) under confounding (synthetic; original).](../assets/figures/ml_fig_pdp_not_cause.png)

*Figure — PDP / ICE / SHAP follow the observational joint. **Left:** a confounder \(U\) drives both \(X\) and \(Y\); the binned observational mean of \(Y\) given \(X\) slopes strongly, while the approximate interventional \(E[Y\mid do(X)]\) is nearly flat. **Right:** solid arrows \(U\to X\), \(U\to Y\) explain the mirage; a weak dashed \(X\to Y\) does not license “change \(X\) to change \(Y\).” Prediction tools can still be useful—just keep claim types separate.*

![Bias–variance tradeoff vs model complexity (teaching sketch; original).](../assets/figures/ml_fig_bias_variance_curve.png)

*Figure — Expected test error ≈ bias² + variance + irreducible noise. Underfit sits left (high bias); overfit sits right (high variance). Regularization and more data move the curves; the “sweet spot” is task- and sample-size specific. Lower prediction error still does not establish causation.*

![Learning-curve signatures of underfit, overfit, and better capacity match (synthetic; original).](../assets/figures/ml_fig_learning_signatures.png)

*Figure — Read the train/val pair. **Left:** both errors high → underfit. **Middle:** large optimistic train–val gap → overfit. **Right:** both fall with n → capacity better matched. Curves diagnose fit—not causation.*

![Permutation importance ΔAUROC with correlated/leakage risks (synthetic; original).](../assets/figures/ml_fig_perm_importance.png)

*Figure — Drop in score when a feature is shuffled. Large drops flag useful *predictors* under the fitted model; correlated features share credit, and scanner/site can look “important” via leakage. Importance rankings are not causal effects.*

### Classification

Classification predicts discrete labels: stroke versus mimic, LVO versus no LVO, mRS 0–2 versus 3–6. Binary, multiclass, and multi-label settings differ in loss design and metrics. Logistic regression, linear discriminants, trees and forests, boosting, support vector machines, and deep networks are standard tools. Class imbalance, label noise, and calibration deserve first-class attention in medicine.

### Self-supervised learning

Self-supervised learning (SSL) creates supervisory signals from the structure of unlabeled data: predict a masked word, the next token, a held-out image patch, or whether two augmented views match. SSL pretrains representations that transfer to downstream tasks with fewer labels—crucial when expert annotation is scarce. Clinically, SSL on notes or imaging can improve sample efficiency, but still requires careful fine-tuning evaluation and bias auditing; unlabeled corpora inherit institutional bias.

### Generative AI

Generative models learn to sample from a distribution over data: text, images, tabular rows, or molecular structures. Families include autoregressive language models, variational autoencoders, generative adversarial networks, and diffusion models. Uses include data augmentation, synthetic controls under strict governance, draft documentation, and image reconstruction. Risks include hallucinated facts, memorization of private training records, and persuasive but wrong clinical prose. Treat generative outputs as proposals under human review unless a validated closed-loop task says otherwise.

### Reinforcement learning

Reinforcement learning (RL) trains an agent that observes states, takes actions, and receives rewards over time, aiming to maximize expected cumulative reward. Unlike supervised learning, the correct action is not given; exploration is required and feedback may be delayed. RL fits sequential decisions—ventilation weaning policies in simulation, game play, robotic control. In healthcare, offline RL from observational logs is attractive but hazardous: confounding by indication and unobserved severity can make harmful policies look optimal. Prefer cautious evaluation, conservative algorithms, and prospective trials before clinical autonomy.

Unsupervised: clustering, association/sequence mining, dimensionality reduction, anomaly detection.

Supervised: regression/correlation analyses and classification with labeled targets.

Self-supervised: pseudo-labels from data structure for representation learning.

Generative: sample or synthesize complex data; govern hallucinations and privacy.

Reinforcement: learn policies from interactive or offline reward signals.

## Training and Evaluation in Supervised Learning

### Training and inference

Supervised learning has two distinct phases, and conflating them is a frequent source of both software bugs and inflated claims. Training (also called fitting or estimation) is the phase in which a learning algorithm searches a hypothesis class H for parameters θ̂ that minimize an empirical objective—typically the average loss ℒ over the labeled training set, plus any regularization penalty. Training consumes the labels y, and it can be computationally heavy: many passes over the data, hyperparameter searches, and cross-validation all happen here. Inference (also called prediction or scoring) is the phase in which the single fitted model is applied to a new input x to produce an output ŷ = f(x; θ̂). Inference does not consult the label of the case being scored—if it did, there would be nothing left to predict.

The distinction is not pedantic; it drives three operational realities. First, cost profile: a hospital may retrain a phenotyping model monthly on a large offline cluster (training) while requiring millisecond scoring at order entry (inference), so the two phases carry different hardware and latency budgets. Second, information availability: at inference the model may use only data that exist at or before index time—the instant from which prediction is allowed to use past information and after which outcomes are counted—whereas the training set is assembled retrospectively with outcomes already known. A field that is trivially present during retrospective training but not yet recorded at bedside index time cannot be a legitimate inference feature; assuming otherwise is temporal leakage and inflates reported performance. Third, governance: auditors freeze and version the inference artifact and its input contract separately from the training procedure that produced it. Whenever you report a number, say which phase it describes—training error flatters the model, and only performance on data untouched during training estimates what inference will actually deliver.

### Ground-truth datasets

Ground truth denotes the reference labels treated as correct for training and evaluation. In handwritten digit recognition, ground truth is the digit identity. In stroke ML, ground truth is rarely metaphysical truth: it is a measurement process—adjudicated imaging reads, chart-reviewed etiology, 90-day mRS from structured follow-up, or a validated computable phenotype. Label noise, adjudication disagreement, and missing follow-up mean the ‘ground’ moves. Models learn the measurement process you provide, including its biases.

Constructing ground truth is a first-class scientific task. Define inclusion criteria, label schemas, adjudication rules for conflicts, and quality-control samples. Report inter-rater reliability when labels are human. When using billing codes as labels, cite validation statistics against chart review and consider sensitivity analyses with alternative phenotypes. For imaging, separate the train-time annotations from the evaluation standard if they differ in quality.

### k-fold cross-validation

If you fit parameters and claim performance on the same observations, you will systematically overstate quality. Hold-out validation splits data into training and validation (and ideally a final test) sets. When data are scarce, k-fold cross-validation (CV) rotates the validation role: partition the n examples into k disjoint folds of roughly equal size; for each i, train on all folds except i and evaluate on fold i; average the k scores. Common choices are k = 5 or k = 10. Leave-one-out CV is the extreme k = n, often expensive and high-variance for some metrics.

![1.4: Five-fold cross-validation rotates the held-out validation fold (amber) across five iterations inside the training parti](../assets/figures/ml_concept_1.4_87ae9aac.png)

*Figure 1.4 — original teaching graphic.*

Nested cross-validation uses an inner loop for hyperparameter selection and an outer loop for performance estimation, reducing optimistic bias from tuning. Grouped CV keeps all rows from the same patient (or site) inside the same fold to block leakage. Time-series CV trains only on the past and validates on the future. Stratified CV preserves class prevalence in each fold for classification. CV estimates internal reproducibility under exchangeability assumptions; it is not a substitute for external geographic or temporal validation.

# Pseudocode: grouped k-fold sketch
groups = patient_id # not row index
for train_idx, val_idx in GroupKFold(k=5).split(X, y, groups):
model.fit(X[train_idx], y[train_idx])
scores.append(metric(y[val_idx], model.predict(X[val_idx])))
report mean(scores), std(scores)

Hyperparameters (tree depth, regularization strength, network width) should be chosen using validation folds, not the final test set. The test set—if you have one—is for a rare, pre-specified final report. Repeated peeking converts the test set into a de facto validation set and burns its credibility.

## Working with Data Step by Step

A healthy ML workflow is linear in its accountability even when iterative in practice. The following sequence is a template for clinical and epidemiologic projects; adapt roles and documentation to your institution, but do not skip the scientific content of each step.

![1.1: The machine-learning lifecycle as an iterative pipeline running from problem framing through data, features, model, eval](../assets/figures/ml_concept_1.1_7405ed2f.png)

*Figure 1.1 — original teaching graphic.*

1. Frame the decision and estimand: who, when (index time), what action the score changes, what outcome horizon.

2. Define success metrics and constraints before model shopping (recall at fixed precision, calibration, latency).

3. Design the cohort: eligibility, exclusions, unit of analysis (patient vs encounter), site and era.

4. Inventory data sources and governance: EHR, imaging, device, registry linkage; IRB and privacy rules.

5. Build labeled datasets with explicit phenotype definitions and QC; quantify label reliability.

6. Exploratory data analysis: quality, missingness, shift across sites, Table-1 structure (Chapter 2).

7. Feature engineering under time discipline: only pre-index information; document look-back windows.

8. Split data with leakage control: grouped, temporal, or external site holds; freeze a test set.

9. Train baselines first (simple, interpretable), then richer models; tune only on validation/CV.

10. Evaluate discrimination, calibration, subgroups, and error slices; pre-specify primary analysis.

11. Stress-test: shift, missingness patterns, adversarial labels, operational workload of alerts.

12. Deploy with monitoring: data drift, performance drift, feedback loops; version models and code.

13. Update under governance: retraining criteria, change control, and communication to users.

Iteration is expected: EDA will rewrite features; external validation will humble a favorite model; monitoring will detect a coding-system change. What must remain stable is the honest mapping between claims and evidence. Skipping steps 1–5 and jumping to step 9 is the most expensive shortcut in clinical ML.

## Worked Example: Choosing a Model with Train and Validation Error

We walk through a small, fully numerical decision that mirrors everyday model selection. Suppose we predict a binary outcome from a single numeric feature using two candidate rule families on a fixed feature representation. We have 20 labeled examples. We hold out 6 as a validation set and train on 14. A real study would use more data and a separate test set; the logic is identical.

![Training vs validation error versus model capacity (original).](../assets/figures/ml_fig_bias_capacity.png)

*Figure 1.6. Training error falls monotonically with model capacity while validation error is U-shaped; the amber dashed line marks the sweet-spot capacity that minimizes validation error, near the selected Candidate A and far from the overfit Candidate B.*

Candidate A is a threshold rule on the feature: predict positive if x ≥ t, choosing t to minimize training classification error. Candidate B is a more flexible piecewise rule that can place two thresholds (a band classifier), also fit by minimizing training error. Higher flexibility means higher capacity.

After fitting, suppose we observe the following error counts (misclassifications). Candidate A: train errors = 3 out of 14 (train error rate 3/14 ≈ 0.214); validation errors = 2 out of 6 (val error rate 2/6 ≈ 0.333). Candidate B: train errors = 0 out of 14 (train error rate 0.000); validation errors = 3 out of 6 (val error rate 3/6 = 0.500).

Candidate B fits the training sample perfectly but generalizes worse on validation. This is the signature of overfitting: capacity absorbed noise or idiosyncrasies. Candidate A has higher training error but better validation error, suggesting a bias–variance compromise that is preferable for deployment under our metric (0–1 classification error). We select Candidate A based on validation performance. If we had peeked at a test set during this choice, we would need a fresh test set for final reporting. With only 6 validation points, the difference between 2/6 and 3/6 is noisy; in practice we would use cross-validation, more data, or uncertainty estimates before a high-stakes launch.

# Numerical summary of the selection rule
n_train, n_val = 14, 6
err_A = {‘train’: 3/14, ‘val’: 2/6} # ~0.214, ~0.333 -> SELECT
err_B = {‘train’: 0/14, ‘val’: 3/6} # 0.000, 0.500 -> reject (overfit)
assert err_A[‘val’] < err_B[‘val’]
# Optional: if FN costs 2x FP, recompute validation cost—not raw error.

Optional extension: if false negatives cost twice as much as false positives, recompute validation cost rather than raw error count. Metrics must reflect decision costs, not habit. The qualitative lesson stands regardless of sample size: never choose the model with the lowest training error alone.

## Clinical and Epidemiologic Notes

Neurologists and epidemiologists already practice a disciplined form of learning from data: they define populations, exposure and outcome windows, and threats to validity before interpreting associations. Machine learning does not replace that discipline; it amplifies whatever design you encode. A high-capacity model trained on a poorly timed electronic health record extract will confidently learn the wrong target.

Cohort design is model design. If you train a stroke outcome model only on patients who received comprehensive stroke center care, predictions for community hospitals may fail—not because the algorithm is weak, but because the cohort encodes a different care pathway, case mix, and documentation intensity. Inclusion and exclusion criteria should be written with the same care as a registry protocol: first-ever versus recurrent ischemic stroke, age bounds, imaging confirmation versus clinical diagnosis, and whether in-hospital events are allowed.

Index time (time zero) is the instant from which prediction is allowed to use past information and after which outcomes are counted. For a model that predicts 90-day recurrent stroke after discharge, a natural index time is discharge (or 24 hours after admission if the goal is early risk stratification). Features must be constructed only from data available at or before index time. Using discharge diagnosis codes finalized after the event, or imaging reports filed after the decision you claim to support, is temporal leakage: the model peeks into the future and reports inflated performance that cannot be realized at the bedside.

Supervised learning assumes a label y. In stroke research, y is often a phenotype: ischemic stroke subtype, symptomatic intracranial hemorrhage after thrombolysis, large-vessel occlusion on CTA, or modified Rankin Scale at 90 days. Phenotypes from billing codes alone have imperfect sensitivity and specificity relative to chart review. Label noise biases empirical risk minimization toward documentation habits. Prefer validated computable phenotypes and multi-source adjudication for critical endpoints. Missingness is informative: patients who lack follow-up imaging differ systematically from those who complete it.

Random splits of a single hospital system estimate internal reproducibility, not geographic or temporal generalizability. External validation on another health system, a later calendar period, or a different care tier is a stronger claim. Patient-level and encounter-level identifiers must not leak across splits. Report calibration and operating-point metrics aligned with neurologic decision costs; average AUROC alone is incomplete for rare events and for counseling conversations that need absolute risks.

Write eligibility, index time, look-back, and outcome windows before feature engineering.

Treat phenotypes as measurement processes with error rates, not as ground truth by default.

Use patient-level (and site-level) splits; reserve external/temporal validation for strong claims.

Report calibration and cost-aligned metrics; monitor subgroup error and documentation bias.

Match task type (classification, survival, RL) to the clinical decision, not to tool familiarity.

## Features, Labels, Capacity, and Inductive Bias

A feature is a measurable property used as model input. Features may be raw measurements or engineered quantities. A label is the supervised target. Training estimates parameters by optimizing an objective; inference applies a fixed model to new inputs. Inductive bias is the set of assumptions a learner uses to generalize beyond the training sample—linearity, local smoothness, tree-shaped interactions, convolutional locality. Every nontrivial learner has an inductive bias; without assumptions, no finite dataset determines future predictions uniquely.

![1.3: The same noisy sample fit three ways, showing an underfitting straight line, a good-fitting smooth curve tracking the tr](../assets/figures/ml_concept_1.3_afbe8cd4.png)

*Figure 1.3 — original teaching graphic.*

Model capacity informally measures how wide a range of functions a hypothesis class can represent. Underfitting: high train and test error—try richer features or capacity. Overfitting: low train error, high test error—try regularization, more data, simpler models, or early stopping. The bias–variance decomposition (for squared loss) separates irreducible noise, systematic wrong assumptions (bias), and sensitivity to the particular sample (variance). Use it as a diagnostic language, not a single number you always compute.

Capacity is the knob that trades bias for variance. A low-capacity learner—a constant, or a shallow linear rule—makes strong assumptions: it cannot chase every wiggle in the data, so its fitted function barely changes from one training sample to the next (low variance) but may be systematically off (high bias). A high-capacity learner—a deep tree, a wide network—can represent intricate functions, reducing systematic error (low bias) at the cost of a fit that swings with the particular sample it happened to see (high variance). For squared loss the expected error at a point decomposes as 𝔼[(ŷ − y)²] = (bias)² + variance + σ², where σ² is irreducible noise no model can remove. Good capacity selection minimizes the sum of the first two terms, which is almost never at either extreme.

Worked illustration (bias–variance). Let the true mean response at some input be f = 10 with irreducible noise variance σ² = 1, and compare two estimators over many hypothetical resamples of the training data. Model 1 predicts 8 in every resample: its expected prediction is 8, so bias = 8 − 10 = −2 and (bias)² = 4, and because it never varies, variance = 0; its structured error (everything but the irreducible term) is 4 + 0 = 4. Model 2 is flexible and predicts 7 in half the resamples and 13 in the other half: its expected prediction is (7 + 13)/2 = 10, so bias = 0, but variance = ((7 − 10)² + (13 − 10)²)/2 = (9 + 9)/2 = 9; its structured error is 0 + 9 = 9. Adding σ² = 1 gives a total expected squared error of 5 for Model 1 versus 10 for Model 2. The biased-but-stable model wins, 5 to 10—an explicit demonstration that lower bias does not guarantee lower error when it is purchased with variance. This is the same lesson the model-selection example taught under 0–1 loss: the more flexible rule fit its sample and generalized worse.

![Bias–variance decomposition of expected squared error versus model capacity (synthetic; original).](../assets/figures/ml_fig_bias_variance_decomp.png)

*Figure — Bias–variance sketch. As capacity rises, (bias)² falls while variance rises; irreducible noise is a floor. The vertical line marks the capacity that minimizes the sum—the teaching “sweet spot,” not a number you always compute from one fit.*


![Bias–variance–noise error stack vs capacity (synthetic; original).](../assets/figures/ml_fig_bias_variance_stack.png)

*Figure — Classic decomposition sketch. Bias falls then variance rises with capacity; irreducible noise sets a floor. The U-shaped total risk is about prediction error under a sampling model—not proof of a causal story.*


![Held-out accuracy vs label flip noise rate (synthetic; original).](../assets/figures/ml_fig_label_noise_acc.png)

*Figure — Label noise ceiling. As flips rise, achievable accuracy falls toward chance. Noisy supervision biases learning; it is related to but not identical with causal measurement-error designs.*


![Dataset shift taxonomy severity sketch: covariate, label, concept (original).](../assets/figures/ml_fig_shift_taxonomy.png)

*Figure — Transport difficulty rises from covariate/label shift to concept shift (P(y|x) changes). Shift diagnosis guides monitoring—**not moral blame** and not automatic causal attribution.*


![Aleatoric vs epistemic-style uncertainty bands (synthetic; original).](../assets/figures/ml_fig_uncertainty_bands.png)

*Figure — Uncertainty cartoons: irreducible noise-like width vs data-sparse expansion. Bands communicate forecast humility—they are not automatic causal identification tools.*


![OOD score vs number of training domains (synthetic; original).](../assets/figures/ml_fig_domain_breadth.png)

*Figure — Broader domain coverage often improves transport in teaching curves. Domain breadth helps generalization estimates—not automatic causal identification.*


![Smooth healthy training loss curve (synthetic; original).](../assets/figures/ml_fig_smooth_loss.png)

*Figure — Optimization trajectory is not a causal claim. Pred != cause without design.*


![Capacity vs approximation power toy curve (original).](../assets/figures/ml_fig_capacity_approx.png)

*Figure — Capacity enables approximation; not causation. Pred != cause without design.*


![inductive teaching panel (original).](../assets/figures/ml_fig_inductive_bias.png)

*Figure — Teaching panel for inductive. Pred != cause without design.*


![Cycle-34 densify scientific panel 3 (original).](../assets/figures/ml_fig_c34_02.png)

*Figure — Continuous densify panel 3. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 3 (original).](../assets/figures/ml_fig_c35_02.png)

*Figure — Continuous densify panel 3. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 3 (original).](../assets/figures/ml_fig_c36_02.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 3 (original).](../assets/figures/ml_fig_c37_02.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 3 (original).](../assets/figures/ml_fig_c38_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 3 (original).](../assets/figures/ml_fig_c39_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 3 (original).](../assets/figures/ml_fig_c40_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 3 (original).](../assets/figures/ml_fig_c41_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 3 (original).](../assets/figures/ml_fig_c42_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 3 (original).](../assets/figures/ml_fig_c43_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 3 (original).](../assets/figures/ml_fig_c44_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 3 (original).](../assets/figures/ml_fig_c45_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 3 (original).](../assets/figures/ml_fig_c46_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 3 (original).](../assets/figures/ml_fig_c47_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 3 (original).](../assets/figures/ml_fig_c48_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 3 (original).](../assets/figures/ml_fig_c49_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 3 (original).](../assets/figures/ml_fig_c50_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 3 (original).](../assets/figures/ml_fig_c51_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 3 (original).](../assets/figures/ml_fig_c52_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 3 (original).](../assets/figures/ml_fig_c53_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 3 (original).](../assets/figures/ml_fig_c54_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 3 (original).](../assets/figures/ml_fig_c55_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 3 (original).](../assets/figures/ml_fig_c56_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 3 (original).](../assets/figures/ml_fig_c57_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 3 (original).](../assets/figures/ml_fig_c58_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 3 (original).](../assets/figures/ml_fig_c59_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 3 (original).](../assets/figures/ml_fig_c60_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 3 (original).](../assets/figures/ml_fig_c61_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 3 (original).](../assets/figures/ml_fig_c62_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 3 (original).](../assets/figures/ml_fig_c63_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 3 (original).](../assets/figures/ml_fig_c64_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 3 (original).](../assets/figures/ml_fig_c65_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 3 (original).](../assets/figures/ml_fig_c66_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 3 (original).](../assets/figures/ml_fig_c67_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 3 (original).](../assets/figures/ml_fig_c68_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 3 (original).](../assets/figures/ml_fig_c69_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 3 (original).](../assets/figures/ml_fig_c70_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 3 (original).](../assets/figures/ml_fig_c71_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 3 (original).](../assets/figures/ml_fig_c72_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 3 (original).](../assets/figures/ml_fig_c73_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 3 (original).](../assets/figures/ml_fig_c74_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 3 (original).](../assets/figures/ml_fig_c75_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 3 (original).](../assets/figures/ml_fig_c76_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 3 (original).](../assets/figures/ml_fig_c77_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 3 (original).](../assets/figures/ml_fig_c78_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 3 (original).](../assets/figures/ml_fig_c79_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 3 (original).](../assets/figures/ml_fig_c80_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 3 (original).](../assets/figures/ml_fig_c81_02.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

## Common Pitfalls and Failure Modes

The failure modes below recur across projects and modalities. Each has a mechanism, a warning sign, and a discipline that prevents it. Treat this as a checklist to run against any pipeline before you believe its numbers.

Data leakage is the contamination of training or feature construction with information that will not be legitimately available at inference time, or that quietly encodes the label. Temporal leakage uses post-index information (a discharge code finalized after the event, an imaging report filed after the decision the model claims to support). Group leakage places rows from the same patient or site on both sides of a split, so the model memorizes individuals instead of learning a transferable rule. Target leakage admits a feature that is a near-proxy for the outcome (a “stroke-unit transfer” flag when predicting stroke). The warning sign is performance that looks too good; the discipline is to build features only from pre-index data and to split by patient and site, as in grouped and time-series cross-validation.

Distribution shift is any change between the data a model was trained on and the data it meets in deployment: different scanners, coding intensity, case mix, season, or care pathway. Because the i.i.d. assumption underlying most training silently fails, discrimination and especially calibration degrade. The discipline is external and temporal validation, subgroup reporting, and ongoing monitoring of inputs and outputs rather than a one-time launch.

Class imbalance makes accuracy misleading and can starve the optimizer of signal about the rare, usually important class (seizures, hemorrhages, deaths). Report precision, recall, and calibration; consider cost-sensitive losses or resampling; and choose the operating threshold on a validation set, never on the test set.

Label noise and shifting ground truth bias empirical risk minimization toward whatever measurement process produced the labels. Billing-code phenotypes with imperfect sensitivity and specificity teach the model documentation habits rather than biology. The discipline is validated computable phenotypes, adjudication for critical endpoints, and sensitivity analyses under alternative label definitions.

Overfitting the test set—multiplicity, or the “garden of forking paths”—occurs when the same held-out data guide many decisions (model families, thresholds, features), so the final estimate is optimistic. Repeated peeking converts a test set into a de facto validation set. The discipline is nested cross-validation for tuning, a pre-specified primary analysis, and a test set opened once.

Shortcut learning and spurious correlations arise when a high-capacity model latches onto a feature that predicts in the training distribution but is neither causal nor stable—a scanner watermark that co-occurs with disease, a laterality artifact, a hospital-specific token. Discrimination survives an internal split yet collapses on transfer. The discipline is stress-testing, subgroup and error-slice analysis, and causal reasoning about which features should matter.

Miscalibration means predicted probabilities do not match observed frequencies even when ranking (AUROC) is good. A model can discriminate well yet quote absolute risks that mislead a counseling conversation. Report reliability diagrams and Brier scores, and recalibrate (for example, by isotonic regression or Platt scaling) on data held apart from training whenever absolute risks drive decisions.

Leakage: allow no post-index or same-group information across splits; split by patient and site.

Shift: validate externally and temporally; monitor inputs and outputs after deployment.

Imbalance and label noise: use cost-aware metrics, validated phenotypes, and sensitivity analyses.

Multiplicity: pre-specify the primary analysis; open the test set once.

Shortcuts and miscalibration: stress-test error slices; report and repair calibration.

## Ethics, Fairness, and Privacy as Design Constraints

Fairness, privacy, and transparency are not compliance chores bolted on after modeling; they are design constraints that shape the estimand, the cohort, the features, the loss, and the evaluation. A model optimized only for average accuracy can be simultaneously state-of-the-art and unfit to deploy. This section previews the ideas as first-class requirements; Chapter 16 develops them formally.

Fairness concerns systematic differences in a model’s behavior across groups defined by attributes such as age, sex, race or ethnicity, language, insurance status, or geography. Two mechanisms produce unfairness even with well-intentioned engineering. First, label bias: if the ground truth itself reflects unequal access or documentation—say, under-coding of stroke in populations with poorer access to imaging—then a model trained to reproduce those labels inherits and can amplify the disparity. Second, representation bias: groups scarce in the training data receive high-variance, poorly calibrated predictions. Fairness is therefore measured by disaggregated performance—sensitivity, specificity, calibration, and error slices computed per subgroup—not by a single pooled number that can hide a subgroup catastrophe. Crucially, several intuitive fairness criteria (equal error rates versus equal calibration versus equal positive-prediction rates) are mathematically incompatible except in trivial cases, so fairness demands an explicit, defensible choice tied to the decision and its harms, not a checkbox.

Privacy is a first-class constraint because health data are identifying and sensitive. De-identification reduces but does not eliminate re-identification risk, especially with rich longitudinal or imaging data. High-capacity models can memorize training examples, exposing them to membership-inference attacks (an adversary infers whether a specific patient was in the training set) or, for generative models, verbatim regurgitation. Differential privacy adds calibrated noise to bound how much any single record can influence the model, trading a measurable amount of accuracy for a formal privacy guarantee; federated learning keeps records local and shares only model updates. These are deliberate design decisions with accuracy costs, not afterthoughts.

Transparency, accountability, and consent complete the picture. Users deserve to know a prediction’s basis and its uncertainty; institutions must assign responsibility for errors and provide a path to redress; and data use must respect the terms under which patients’ information was collected. The operational rule is to write fairness, privacy, and governance requirements into steps 1–5 of the workflow—alongside the estimand and cohort—so they constrain design rather than merely audit a finished model. When to relax a constraint is never a silent default: a deliberately less private or less interpretable model may be justified for a high-value task, but the trade-off must be named and approved, exactly as with the accuracy–efficiency frontier.

Fairness: disaggregate performance by subgroup; label and representation bias corrupt pooled metrics; fairness criteria can conflict and require an explicit choice.

Privacy: de-identification is not immunity; guard against memorization and membership inference; differential privacy and federation trade accuracy for guarantees.

Transparency and accountability: expose basis and uncertainty; assign responsibility; respect consent.

Encode ethics as a design input in workflow steps 1–5, and name any trade-off explicitly.

## Connections to Other Chapters

This chapter is scaffolding; later chapters supply the load-bearing detail.

Data understanding, missingness, and Table-1 structure are developed in Chapter 2, extending the workflow’s exploratory-analysis step.

Unsupervised methods previewed here—k-means, hierarchical clustering, and mixture models—are treated in Chapter 4, where the evaluation-without-labels problem is confronted directly.

Supervised regression and classification, the loss functions and optimization sketched under training and inference, and the bias–variance trade-off recur throughout the supervised-learning chapters, where regularization and capacity control become concrete algorithms.

Cross-validation discipline, calibration, and cost-aligned metrics introduced here are the backbone of the evaluation chapters, which formalize discrimination, calibration, and decision-analytic measures.

Fairness, privacy, and accountability, previewed as design constraints, are developed formally in Chapter 16, including differential privacy and subgroup evaluation.

The clinical and epidemiologic through-line—cohort design, index time, phenotype quality, leakage control, and external validation—connects to the applied neurology and population-health case studies later in the book.


![c82 teaching panel 02 (original).](../assets/figures/ml_fig_c82_02.png)
*Figure — Inductive bias spectrum from local k-NN to flexible deep models. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 02 (original).](../assets/figures/ml_fig_c83_02.png)
*Figure — Bias–variance tradeoff versus model capacity. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 02 (original).](../assets/figures/ml_fig_c84_02.png)
*Figure — Out-of-distribution points outside training support. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 02 (original).](../assets/figures/ml_fig_c85_02.png)
*Figure — Learning curves: more data typically shrinks the gen. gap. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 02 (original).](../assets/figures/ml_fig_c86_02.png)
*Figure — Version trajectories for underfit vs overfit errors. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 02 (original).](../assets/figures/ml_fig_c87_02.png)
*Figure — Interpretability versus flexibility across model families. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 02 (original).](../assets/figures/ml_fig_c88_02.png)
*Figure — IID assumption vs dependent clinical episodes. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 02 (original).](../assets/figures/ml_fig_c89_02.png)
*Figure — No free lunch: average over tasks is flat. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 02 (original).](../assets/figures/ml_fig_c90_02.png)
*Figure — k-fold CV partition diagram. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 02 (original).](../assets/figures/ml_fig_c91_02.png)
*Figure — Empirical risk vs true risk gap. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 02 (original).](../assets/figures/ml_fig_c92_02.png)
*Figure — PAC learning schematic bounds. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 02 (original).](../assets/figures/ml_fig_c93_02.png)
*Figure — Occams razor capacity control. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 02 (original).](../assets/figures/ml_fig_c94_02.png)
*Figure — VC-dimension capacity cartoon. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 02 (original).](../assets/figures/ml_fig_c95_02.png)
*Figure — Double descent test risk curve. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 02 (original).](../assets/figures/ml_fig_c96_02.png)
*Figure — Interpolation threshold n≈p. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 02 (original).](../assets/figures/ml_fig_c97_02.png)
*Figure — Bias-complexity tradeoff redraw. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 02 (original).](../assets/figures/ml_fig_c98_02.png)
*Figure — Interpolation vs extrapolation zones. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 02 (original).](../assets/figures/ml_fig_c99_02.png)
*Figure — Effective degrees of freedom. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 02 (original).](../assets/figures/ml_fig_c100_02.png)
*Figure — Approximation-estimation-optim split. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 02 (original).](../assets/figures/ml_fig_c101_02.png)
*Figure — Uniform convergence cartoon. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 02 (original).](../assets/figures/ml_fig_c102_02.png)
*Figure — Benign overfitting cartoon. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 02 (original).](../assets/figures/ml_fig_c103_02.png)
*Figure — Hypothesis class nested families. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 02 (original).](../assets/figures/ml_fig_c104_02.png)
*Figure — Rademacher complexity sketch. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 02 (original).](../assets/figures/ml_fig_c105_02.png)
*Figure — Implicit regularization GD. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 02 (original).](../assets/figures/ml_fig_c106_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 02 (original).](../assets/figures/ml_fig_c107_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 02 (original).](../assets/figures/ml_fig_c108_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 02 (original).](../assets/figures/ml_fig_c109_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 02 (original).](../assets/figures/ml_fig_c110_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 02 (original).](../assets/figures/ml_fig_c111_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 02 (original).](../assets/figures/ml_fig_c112_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 02 (original).](../assets/figures/ml_fig_c113_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 02 (original).](../assets/figures/ml_fig_c114_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 02 (original).](../assets/figures/ml_fig_c115_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 02 (original).](../assets/figures/ml_fig_c116_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 02 (original).](../assets/figures/ml_fig_c117_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 02 (original).](../assets/figures/ml_fig_c118_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 02 (original).](../assets/figures/ml_fig_c119_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 02 (original).](../assets/figures/ml_fig_c120_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 02 (original).](../assets/figures/ml_fig_c121_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 02 (original).](../assets/figures/ml_fig_c122_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 02 (original).](../assets/figures/ml_fig_c123_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 02 (original).](../assets/figures/ml_fig_c124_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 02 (original).](../assets/figures/ml_fig_c125_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 02 (original).](../assets/figures/ml_fig_c126_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 02 (original).](../assets/figures/ml_fig_c127_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 02 (original).](../assets/figures/ml_fig_c128_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 02 (original).](../assets/figures/ml_fig_c129_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 02 (original).](../assets/figures/ml_fig_c130_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 02 (original).](../assets/figures/ml_fig_c131_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 02 (original).](../assets/figures/ml_fig_c132_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 02 (original).](../assets/figures/ml_fig_c133_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 02 (original).](../assets/figures/ml_fig_c134_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 02 (original).](../assets/figures/ml_fig_c135_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 02 (original).](../assets/figures/ml_fig_c136_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 02 (original).](../assets/figures/ml_fig_c137_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 02 (original).](../assets/figures/ml_fig_c138_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 02 (original).](../assets/figures/ml_fig_c139_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 02 (original).](../assets/figures/ml_fig_c140_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 02 (original).](../assets/figures/ml_fig_c141_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 02 (original).](../assets/figures/ml_fig_c142_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 02 (original).](../assets/figures/ml_fig_c143_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 02 (original).](../assets/figures/ml_fig_c144_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 02 (original).](../assets/figures/ml_fig_c145_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 02 (original).](../assets/figures/ml_fig_c146_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 02 (original).](../assets/figures/ml_fig_c147_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 02 (original).](../assets/figures/ml_fig_c148_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 02 (original).](../assets/figures/ml_fig_c149_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 02 (original).](../assets/figures/ml_fig_c150_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 02 (original).](../assets/figures/ml_fig_c151_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 02 (original).](../assets/figures/ml_fig_c152_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 02 (original).](../assets/figures/ml_fig_c153_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 02 (original).](../assets/figures/ml_fig_c154_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 02 (original).](../assets/figures/ml_fig_c155_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 02 (original).](../assets/figures/ml_fig_c156_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 02 (original).](../assets/figures/ml_fig_c157_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 02 (original).](../assets/figures/ml_fig_c158_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 02 (original).](../assets/figures/ml_fig_c159_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 02 (original).](../assets/figures/ml_fig_c160_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 02 (original).](../assets/figures/ml_fig_c161_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 02 (original).](../assets/figures/ml_fig_c162_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 02 (original).](../assets/figures/ml_fig_c163_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 02 (original).](../assets/figures/ml_fig_c164_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 02 (original).](../assets/figures/ml_fig_c165_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 02 (original).](../assets/figures/ml_fig_c166_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 02 (original).](../assets/figures/ml_fig_c167_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 02 (original).](../assets/figures/ml_fig_c168_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 02 (original).](../assets/figures/ml_fig_c169_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 02 (original).](../assets/figures/ml_fig_c170_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 02 (original).](../assets/figures/ml_fig_c171_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 02 (original).](../assets/figures/ml_fig_c172_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 02 (original).](../assets/figures/ml_fig_c173_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 02 (original).](../assets/figures/ml_fig_c174_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 02 (original).](../assets/figures/ml_fig_c175_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 02 (original).](../assets/figures/ml_fig_c176_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 02 (original).](../assets/figures/ml_fig_c177_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 02 (original).](../assets/figures/ml_fig_c178_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 02 (original).](../assets/figures/ml_fig_c179_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 02 (original).](../assets/figures/ml_fig_c180_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 02 (original).](../assets/figures/ml_fig_c181_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 02 (original).](../assets/figures/ml_fig_c182_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 02 (original).](../assets/figures/ml_fig_c183_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 02 (original).](../assets/figures/ml_fig_c184_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 02 (original).](../assets/figures/ml_fig_c185_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 02 (original).](../assets/figures/ml_fig_c186_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 02 (original).](../assets/figures/ml_fig_c187_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 02 (original).](../assets/figures/ml_fig_c188_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 02 (original).](../assets/figures/ml_fig_c189_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 02 (original).](../assets/figures/ml_fig_c190_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 02 (original).](../assets/figures/ml_fig_c191_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 02 (original).](../assets/figures/ml_fig_c192_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 02 (original).](../assets/figures/ml_fig_c193_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 02 (original).](../assets/figures/ml_fig_c194_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 02 (original).](../assets/figures/ml_fig_c195_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 02 (original).](../assets/figures/ml_fig_c196_02.png)
*Figure — Surrogate loss calibration. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 02 (original).](../assets/figures/ml_fig_c197_02.png)
*Figure — Bayes consistency sketch. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 02 (original).](../assets/figures/ml_fig_c198_02.png)
*Figure — Agnostic learning setup. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 02 (original).](../assets/figures/ml_fig_c199_02.png)
*Figure — Realizable case bound. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 02 (original).](../assets/figures/ml_fig_c200_02.png)
*Figure — Excess risk decomposition. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 02 (original).](../assets/figures/ml_fig_c201_02.png)
*Figure — No free lunch risk averages. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 02 (original).](../assets/figures/ml_fig_c202_02.png)
*Figure — Excess risk three-way split. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 02 (original).](../assets/figures/ml_fig_c203_02.png)
*Figure — VC shattering of three points. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 02 (original).](../assets/figures/ml_fig_c204_02.png)
*Figure — Margin-based generalization sketch. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 02 (original).](../assets/figures/ml_fig_c205_02.png)
*Figure — Likelihood ratio density sketch. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 02 (original).](../assets/figures/ml_fig_c206_02.png)
*Figure — Sample complexity epsilon scale. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 02 (original).](../assets/figures/ml_fig_c207_02.png)
*Figure — Rademacher complexity sample ball. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 02 (original).](../assets/figures/ml_fig_c208_02.png)
*Figure — Bias variance noise triad. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 02 (original).](../assets/figures/ml_fig_c209_02.png)
*Figure — No free lunch average risk. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 02 (original).](../assets/figures/ml_fig_c210_02.png)
*Figure — Learning curve with CV band. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 02 (original).](../assets/figures/ml_fig_c211_02.png)
*Figure — Occam complexity sweet spot. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 02 (original).](../assets/figures/ml_fig_c212_02.png)
*Figure — Holm peel-off thresholds. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 02 (original).](../assets/figures/ml_fig_c213_02.png)
*Figure — Hoeffding bound sample size. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 02 (original).](../assets/figures/ml_fig_c214_02.png)
*Figure — Structural risk minimization band. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 02 (original).](../assets/figures/ml_fig_c215_02.png)
*Figure — Uniform convergence epsilon tube. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 02 (original).](../assets/figures/ml_fig_c216_02.png)
*Figure — PAC-Bayes KL complexity term. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 02 (original).](../assets/figures/ml_fig_c217_02.png)
*Figure — Rademacher sum concentration. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 02 (original).](../assets/figures/ml_fig_c218_02.png)
*Figure — Covering number epsilon scale. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 02 (original).](../assets/figures/ml_fig_c219_02.png)
*Figure — VC growth function sketch. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 02 (original).](../assets/figures/ml_fig_c220_02.png)
*Figure — Functional margin histogram. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 02 (original).](../assets/figures/ml_fig_c221_02.png)
*Figure — PAC-Bayes KL generalization sketch. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 02 (original).](../assets/figures/ml_fig_c222_02.png)
*Figure — Coreset compression error curve. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 02 (original).](../assets/figures/ml_fig_c223_02.png)
*Figure — Uniform convergence deviation. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 02 (original).](../assets/figures/ml_fig_c224_02.png)
*Figure — Geometric margin support vectors. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 02 (original).](../assets/figures/ml_fig_c225_02.png)
*Figure — Covering number vs epsilon. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 02 (original).](../assets/figures/ml_fig_c226_02.png)
*Figure — Rademacher complexity decay. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 02 (original).](../assets/figures/ml_fig_c227_02.png)
*Figure — Fat-shattering growth sketch. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 02 (original).](../assets/figures/ml_fig_c228_02.png)
*Figure — Algorithmic stability bound. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 02 (original).](../assets/figures/ml_fig_c229_02.png)
*Figure — MDL fit plus complexity trade. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 02 (original).](../assets/figures/ml_fig_c230_02.png)
*Figure — PAC-Bayes KL prior gap. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 02 (original).](../assets/figures/ml_fig_c231_02.png)
*Figure — Benign overfitting risk curves. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 02 (original).](../assets/figures/ml_fig_c232_02.png)
*Figure — Double descent risk curve. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 02 (original).](../assets/figures/ml_fig_c233_02.png)
*Figure — Natarajan dimension growth. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 02 (original).](../assets/figures/ml_fig_c234_02.png)
*Figure — Stability selection freqs. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 02 (original).](../assets/figures/ml_fig_c235_02.png)
*Figure — Pseudo-dimension growth sketch. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 02 (original).](../assets/figures/ml_fig_c236_02.png)
*Figure — Bootstrap inclusion freqs. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 02 (original).](../assets/figures/ml_fig_c237_02.png)
*Figure — Fat-shattering sample growth. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 02 (original).](../assets/figures/ml_fig_c238_02.png)
*Figure — Permutation importance bars. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 02 (original).](../assets/figures/ml_fig_c239_02.png)
*Figure — VC subgraph growth. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 02 (original).](../assets/figures/ml_fig_c240_02.png)
*Figure — SHAP mean abs bars. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 02 (original).](../assets/figures/ml_fig_c241_02.png)
*Figure — Natarajan dimension growth. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 02 (original).](../assets/figures/ml_fig_c242_02.png)
*Figure — Integrated gradients bars. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 02 (original).](../assets/figures/ml_fig_c243_02.png)
*Figure — Rademacher complexity bound. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 02 (original).](../assets/figures/ml_fig_c244_02.png)
*Figure — SHAP interaction bars. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 02 (original).](../assets/figures/ml_fig_c245_02.png)
*Figure — Covering number sample growth. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 02 (original).](../assets/figures/ml_fig_c246_02.png)
*Figure — DeepLIFT contribution bars. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 02 (original).](../assets/figures/ml_fig_c247_02.png)
*Figure — Shattering coefficient growth. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 02 (original).](../assets/figures/ml_fig_c248_02.png)
*Figure — Grad-CAM channel bars. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 02 (original).](../assets/figures/ml_fig_c249_02.png)
*Figure — Pseudo-dimension bound growth. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 02 (original).](../assets/figures/ml_fig_c250_02.png)
*Figure — Occlusion sensitivity bars. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 02 (original).](../assets/figures/ml_fig_c251_02.png)
*Figure — Fat-shattering dim growth. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 02 (original).](../assets/figures/ml_fig_c252_02.png)
*Figure — Integrated gradients bars. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 02 (original).](../assets/figures/ml_fig_c253_02.png)
*Figure — Natarajan class growth. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 02 (original).](../assets/figures/ml_fig_c254_02.png)
*Figure — SHAP beeswarm bars. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 02 (original).](../assets/figures/ml_fig_c255_02.png)
*Figure — VC subgraph growth. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 02 (original).](../assets/figures/ml_fig_c256_02.png)
*Figure — DeepLIFT channel bars. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 02 (original).](../assets/figures/ml_fig_c257_02.png)
*Figure — Occam razor trade curve c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 02 (original).](../assets/figures/ml_fig_c258_02.png)
*Figure — Bayes error floor path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 02 (original).](../assets/figures/ml_fig_c259_02.png)
*Figure — Agnostic learning path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 02 (original).](../assets/figures/ml_fig_c260_02.png)
*Figure — Online mistake bound c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 02 (original).](../assets/figures/ml_fig_c261_02.png)
*Figure — Active query gain path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 02 (original).](../assets/figures/ml_fig_c262_02.png)
*Figure — Semi-supervised mix path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 02 (original).](../assets/figures/ml_fig_c263_02.png)
*Figure — Multi-task transfer path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 02 (original).](../assets/figures/ml_fig_c264_02.png)
*Figure — Domain shift risk path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 02 (original).](../assets/figures/ml_fig_c265_02.png)
*Figure — Causality vs prediction strip c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 02 (original).](../assets/figures/ml_fig_c266_02.png)
*Figure — Bias-variance residual map c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 02 (original).](../assets/figures/ml_fig_c267_02.png)
*Figure — PAC bound sample growth c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 02 (original).](../assets/figures/ml_fig_c268_02.png)
*Figure — No-free-lunch sketch c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 02 (original).](../assets/figures/ml_fig_c269_02.png)
*Figure — Inductive bias path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 02 (original).](../assets/figures/ml_fig_c270_02.png)
*Figure — Hypothesis class growth c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 02 (original).](../assets/figures/ml_fig_c271_02.png)
*Figure — Empirical risk path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 02 (original).](../assets/figures/ml_fig_c272_02.png)
*Figure — Structural risk path c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 02 (original).](../assets/figures/ml_fig_c273_02.png)
*Figure — Occam razor trade curve c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 02 (original).](../assets/figures/ml_fig_c274_02.png)
*Figure — Bayes error floor path c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 02 (original).](../assets/figures/ml_fig_c275_02.png)
*Figure — Agnostic learning path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 02 (original).](../assets/figures/ml_fig_c276_02.png)
*Figure — Online mistake bound c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 02 (original).](../assets/figures/ml_fig_c277_02.png)
*Figure — Active query gain path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 02 (original).](../assets/figures/ml_fig_c278_02.png)
*Figure — Semi-supervised mix path c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 02 (original).](../assets/figures/ml_fig_c279_02.png)
*Figure — Multi-task transfer path c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 02 (original).](../assets/figures/ml_fig_c280_02.png)
*Figure — Domain shift risk path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 02 (original).](../assets/figures/ml_fig_c281_02.png)
*Figure — Causality vs prediction strip c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 02 (original).](../assets/figures/ml_fig_c282_02.png)
*Figure — Bias-variance residual map c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 02 (original).](../assets/figures/ml_fig_c283_02.png)
*Figure — PAC bound sample growth c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 02 (original).](../assets/figures/ml_fig_c284_02.png)
*Figure — No-free-lunch sketch c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 02 (original).](../assets/figures/ml_fig_c285_02.png)
*Figure — Inductive bias path c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 02 (original).](../assets/figures/ml_fig_c286_02.png)
*Figure — Hypothesis class growth c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 02 (original).](../assets/figures/ml_fig_c287_02.png)
*Figure — Empirical risk path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 02 (original).](../assets/figures/ml_fig_c288_02.png)
*Figure — Structural risk path c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 02 (original).](../assets/figures/ml_fig_c289_02.png)
*Figure — Occam razor trade curve c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 02 (original).](../assets/figures/ml_fig_c290_02.png)
*Figure — Bayes error floor path c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 02 (original).](../assets/figures/ml_fig_c291_02.png)
*Figure — Agnostic learning path c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 02 (original).](../assets/figures/ml_fig_c292_02.png)
*Figure — Online mistake bound c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 02 (original).](../assets/figures/ml_fig_c293_02.png)
*Figure — Active query gain path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 02 (original).](../assets/figures/ml_fig_c294_02.png)
*Figure — Semi-supervised mix path c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 02 (original).](../assets/figures/ml_fig_c295_02.png)
*Figure — Multi-task transfer path c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 02 (original).](../assets/figures/ml_fig_c296_02.png)
*Figure — Domain shift risk path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 02 (original).](../assets/figures/ml_fig_c297_02.png)
*Figure — Causality vs prediction strip c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 02 (original).](../assets/figures/ml_fig_c298_02.png)
*Figure — Bias-variance residual map c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 02 (original).](../assets/figures/ml_fig_c299_02.png)
*Figure — PAC bound sample growth c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 02 (original).](../assets/figures/ml_fig_c300_02.png)
*Figure — No-free-lunch sketch c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 02 (original).](../assets/figures/ml_fig_c301_02.png)
*Figure — Inductive bias path c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 02 (original).](../assets/figures/ml_fig_c302_02.png)
*Figure — Hypothesis class growth c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 02 (original).](../assets/figures/ml_fig_c303_02.png)
*Figure — Empirical risk path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 02 (original).](../assets/figures/ml_fig_c304_02.png)
*Figure — Structural risk path c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 02 (original).](../assets/figures/ml_fig_c305_02.png)
*Figure — Occam razor trade curve c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 02 (original).](../assets/figures/ml_fig_c306_02.png)
*Figure — Bayes error floor path c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 02 (original).](../assets/figures/ml_fig_c307_02.png)
*Figure — Agnostic learning path c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 02 (original).](../assets/figures/ml_fig_c308_02.png)
*Figure — Online mistake bound c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 02 (original).](../assets/figures/ml_fig_c309_02.png)
*Figure — Active query gain path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 02 (original).](../assets/figures/ml_fig_c310_02.png)
*Figure — Semi-supervised mix path c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 02 (original).](../assets/figures/ml_fig_c311_02.png)
*Figure — Multi-task transfer path c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 02 (original).](../assets/figures/ml_fig_c312_02.png)
*Figure — Domain shift risk path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 02 (original).](../assets/figures/ml_fig_c313_02.png)
*Figure — Causality vs prediction strip c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 02 (original).](../assets/figures/ml_fig_c314_02.png)
*Figure — Bias-variance residual map c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 02 (original).](../assets/figures/ml_fig_c315_02.png)
*Figure — PAC bound sample growth c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 02 (original).](../assets/figures/ml_fig_c316_02.png)
*Figure — No-free-lunch sketch c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 02 (original).](../assets/figures/ml_fig_c317_02.png)
*Figure — Inductive bias path c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 02 (original).](../assets/figures/ml_fig_c318_02.png)
*Figure — Hypothesis class growth c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 02 (original).](../assets/figures/ml_fig_c319_02.png)
*Figure — Empirical risk path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 02 (original).](../assets/figures/ml_fig_c320_02.png)
*Figure — Structural risk path c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 02 (original).](../assets/figures/ml_fig_c321_02.png)
*Figure — Occam razor trade curve c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 02 (original).](../assets/figures/ml_fig_c322_02.png)
*Figure — Bayes error floor path c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 02 (original).](../assets/figures/ml_fig_c323_02.png)
*Figure — Agnostic learning path c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 02 (original).](../assets/figures/ml_fig_c324_02.png)
*Figure — Online mistake bound c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 02 (original).](../assets/figures/ml_fig_c325_02.png)
*Figure — Active query gain path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 02 (original).](../assets/figures/ml_fig_c326_02.png)
*Figure — Semi-supervised mix path c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 02 (original).](../assets/figures/ml_fig_c327_02.png)
*Figure — Multi-task transfer path c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 02 (original).](../assets/figures/ml_fig_c328_02.png)
*Figure — Domain shift risk path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 02 (original).](../assets/figures/ml_fig_c329_02.png)
*Figure — Causality vs prediction strip c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 02 (original).](../assets/figures/ml_fig_c330_02.png)
*Figure — Bias-variance residual map c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 02 (original).](../assets/figures/ml_fig_c331_02.png)
*Figure — PAC bound sample growth c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 02 (original).](../assets/figures/ml_fig_c332_02.png)
*Figure — No-free-lunch sketch c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 02 (original).](../assets/figures/ml_fig_c333_02.png)
*Figure — Inductive bias path c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 02 (original).](../assets/figures/ml_fig_c334_02.png)
*Figure — Hypothesis class growth c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 02 (original).](../assets/figures/ml_fig_c335_02.png)
*Figure — Empirical risk path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 02 (original).](../assets/figures/ml_fig_c336_02.png)
*Figure — Structural risk path c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 02 (original).](../assets/figures/ml_fig_c337_02.png)
*Figure — Occam razor trade curve c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 02 (original).](../assets/figures/ml_fig_c338_02.png)
*Figure — Bayes error floor path c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 02 (original).](../assets/figures/ml_fig_c339_02.png)
*Figure — Agnostic learning path c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 02 (original).](../assets/figures/ml_fig_c340_02.png)
*Figure — Online mistake bound c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 02 (original).](../assets/figures/ml_fig_c341_02.png)
*Figure — Active query gain path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 02 (original).](../assets/figures/ml_fig_c342_02.png)
*Figure — Semi-supervised mix path c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 02 (original).](../assets/figures/ml_fig_c343_02.png)
*Figure — Multi-task transfer path c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 02 (original).](../assets/figures/ml_fig_c344_02.png)
*Figure — Domain shift risk path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 02 (original).](../assets/figures/ml_fig_c345_02.png)
*Figure — Causality vs prediction strip c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 02 (original).](../assets/figures/ml_fig_c346_02.png)
*Figure — Bias-variance residual map c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 02 (original).](../assets/figures/ml_fig_c347_02.png)
*Figure — PAC bound sample growth c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 02 (original).](../assets/figures/ml_fig_c348_02.png)
*Figure — No-free-lunch sketch c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 02 (original).](../assets/figures/ml_fig_c349_02.png)
*Figure — Inductive bias path c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 02 (original).](../assets/figures/ml_fig_c350_02.png)
*Figure — Hypothesis class growth c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 02 (original).](../assets/figures/ml_fig_c351_02.png)
*Figure — Empirical risk path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 02 (original).](../assets/figures/ml_fig_c352_02.png)
*Figure — Structural risk path c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 02 (original).](../assets/figures/ml_fig_c353_02.png)
*Figure — Occam razor trade curve c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 02 (original).](../assets/figures/ml_fig_c354_02.png)
*Figure — Bayes error floor path c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 02 (original).](../assets/figures/ml_fig_c355_02.png)
*Figure — Agnostic learning path c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 02 (original).](../assets/figures/ml_fig_c356_02.png)
*Figure — Online mistake bound c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 02 (original).](../assets/figures/ml_fig_c357_02.png)
*Figure — Active query gain path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 02 (original).](../assets/figures/ml_fig_c358_02.png)
*Figure — Semi-supervised mix path c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 02 (original).](../assets/figures/ml_fig_c359_02.png)
*Figure — Multi-task transfer path c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 02 (original).](../assets/figures/ml_fig_c360_02.png)
*Figure — Domain shift risk path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 02 (original).](../assets/figures/ml_fig_c361_02.png)
*Figure — Causality vs prediction strip c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 02 (original).](../assets/figures/ml_fig_c362_02.png)
*Figure — Bias-variance residual map c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 02 (original).](../assets/figures/ml_fig_c363_02.png)
*Figure — PAC bound sample growth c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 02 (original).](../assets/figures/ml_fig_c364_02.png)
*Figure — No-free-lunch sketch c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 02 (original).](../assets/figures/ml_fig_c365_02.png)
*Figure — Inductive bias path c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 02 (original).](../assets/figures/ml_fig_c366_02.png)
*Figure — Hypothesis class growth c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 02 (original).](../assets/figures/ml_fig_c367_02.png)
*Figure — Empirical risk path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 02 (original).](../assets/figures/ml_fig_c368_02.png)
*Figure — Structural risk path c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 02 (original).](../assets/figures/ml_fig_c369_02.png)
*Figure — Occam razor trade curve c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 02 (original).](../assets/figures/ml_fig_c370_02.png)
*Figure — Bayes error floor path c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 02 (original).](../assets/figures/ml_fig_c371_02.png)
*Figure — Agnostic learning path c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 02 (original).](../assets/figures/ml_fig_c372_02.png)
*Figure — Online mistake bound c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 02 (original).](../assets/figures/ml_fig_c373_02.png)
*Figure — Active query gain path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 02 (original).](../assets/figures/ml_fig_c374_02.png)
*Figure — Semi-supervised mix path c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 02 (original).](../assets/figures/ml_fig_c375_02.png)
*Figure — Multi-task transfer path c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 02 (original).](../assets/figures/ml_fig_c376_02.png)
*Figure — Domain shift risk path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 02 (original).](../assets/figures/ml_fig_c377_02.png)
*Figure — Causality vs prediction strip c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 02 (original).](../assets/figures/ml_fig_c378_02.png)
*Figure — Bias-variance residual map c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 02 (original).](../assets/figures/ml_fig_c379_02.png)
*Figure — PAC bound sample growth c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 02 (original).](../assets/figures/ml_fig_c380_02.png)
*Figure — No-free-lunch sketch c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 02 (original).](../assets/figures/ml_fig_c381_02.png)
*Figure — Inductive bias path c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 02 (original).](../assets/figures/ml_fig_c382_02.png)
*Figure — Hypothesis class growth c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 02 (original).](../assets/figures/ml_fig_c383_02.png)
*Figure — Empirical risk path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 02 (original).](../assets/figures/ml_fig_c384_02.png)
*Figure — Structural risk path c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 02 (original).](../assets/figures/ml_fig_c385_02.png)
*Figure — Occam razor trade curve c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 02 (original).](../assets/figures/ml_fig_c386_02.png)
*Figure — Bayes error floor path c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 02 (original).](../assets/figures/ml_fig_c387_02.png)
*Figure — Agnostic learning path c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 02 (original).](../assets/figures/ml_fig_c388_02.png)
*Figure — Online mistake bound c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 02 (original).](../assets/figures/ml_fig_c389_02.png)
*Figure — Active query gain path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 02 (original).](../assets/figures/ml_fig_c390_02.png)
*Figure — Semi-supervised mix path c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 02 (original).](../assets/figures/ml_fig_c391_02.png)
*Figure — Multi-task transfer path c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 02 (original).](../assets/figures/ml_fig_c392_02.png)
*Figure — Domain shift risk path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 02 (original).](../assets/figures/ml_fig_c393_02.png)
*Figure — Causality vs prediction strip c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 02 (original).](../assets/figures/ml_fig_c394_02.png)
*Figure — Bias-variance residual map c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 02 (original).](../assets/figures/ml_fig_c395_02.png)
*Figure — PAC bound sample growth c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 02 (original).](../assets/figures/ml_fig_c396_02.png)
*Figure — No-free-lunch sketch c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 02 (original).](../assets/figures/ml_fig_c397_02.png)
*Figure — Inductive bias path c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 02 (original).](../assets/figures/ml_fig_c398_02.png)
*Figure — Hypothesis class growth c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 02 (original).](../assets/figures/ml_fig_c399_02.png)
*Figure — Empirical risk path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 02 (original).](../assets/figures/ml_fig_c400_02.png)
*Figure — Structural risk path c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 02 (original).](../assets/figures/ml_fig_c401_02.png)
*Figure — Occam razor trade curve c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 02 (original).](../assets/figures/ml_fig_c402_02.png)
*Figure — Bayes error floor path c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 02 (original).](../assets/figures/ml_fig_c403_02.png)
*Figure — Agnostic learning path c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 02 (original).](../assets/figures/ml_fig_c404_02.png)
*Figure — Online mistake bound c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 02 (original).](../assets/figures/ml_fig_c405_02.png)
*Figure — Active query gain path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 02 (original).](../assets/figures/ml_fig_c406_02.png)
*Figure — Semi-supervised mix path c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 02 (original).](../assets/figures/ml_fig_c407_02.png)
*Figure — Multi-task transfer path c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 02 (original).](../assets/figures/ml_fig_c408_02.png)
*Figure — Domain shift risk path c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 02 (original).](../assets/figures/ml_fig_c409_02.png)
*Figure — Causality vs prediction strip c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 02 (original).](../assets/figures/ml_fig_c410_02.png)
*Figure — Bias-variance residual map c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 02 (original).](../assets/figures/ml_fig_c411_02.png)
*Figure — PAC bound sample growth c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 02 (original).](../assets/figures/ml_fig_c412_02.png)
*Figure — No-free-lunch sketch c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 02 (original).](../assets/figures/ml_fig_c413_02.png)
*Figure — Inductive bias path c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 02 (original).](../assets/figures/ml_fig_c414_02.png)
*Figure — Hypothesis class growth c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 02 (original).](../assets/figures/ml_fig_c415_02.png)
*Figure — Empirical risk path c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 02 (original).](../assets/figures/ml_fig_c416_02.png)
*Figure — Structural risk path c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 02 (original).](../assets/figures/ml_fig_c417_02.png)
*Figure — Occam razor trade curve c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 02 (original).](../assets/figures/ml_fig_c418_02.png)
*Figure — Bayes error floor path c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 02 (original).](../assets/figures/ml_fig_c419_02.png)
*Figure — Agnostic learning path c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 02 (original).](../assets/figures/ml_fig_c420_02.png)
*Figure — Online mistake bound c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 02 (original).](../assets/figures/ml_fig_c421_02.png)
*Figure — Active query gain path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 02 (original).](../assets/figures/ml_fig_c422_02.png)
*Figure — Semi-supervised mix path c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 02 (original).](../assets/figures/ml_fig_c423_02.png)
*Figure — Multi-task transfer path c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 02 (original).](../assets/figures/ml_fig_c424_02.png)
*Figure — Domain shift risk path c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 02 (original).](../assets/figures/ml_fig_c425_02.png)
*Figure — Causality vs prediction strip c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 02 (original).](../assets/figures/ml_fig_c426_02.png)
*Figure — Bias-variance residual map c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 02 (original).](../assets/figures/ml_fig_c427_02.png)
*Figure — PAC bound sample growth c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 02 (original).](../assets/figures/ml_fig_c428_02.png)
*Figure — No-free-lunch sketch c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 02 (original).](../assets/figures/ml_fig_c429_02.png)
*Figure — Inductive bias path c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 02 (original).](../assets/figures/ml_fig_c430_02.png)
*Figure — Hypothesis class growth c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 02 (original).](../assets/figures/ml_fig_c431_02.png)
*Figure — Empirical risk path c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 02 (original).](../assets/figures/ml_fig_c432_02.png)
*Figure — Structural risk path c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 02 (original).](../assets/figures/ml_fig_c433_02.png)
*Figure — Occam razor trade curve c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 02 (original).](../assets/figures/ml_fig_c434_02.png)
*Figure — Bayes error floor path c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 02 (original).](../assets/figures/ml_fig_c435_02.png)
*Figure — Agnostic learning path c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 02 (original).](../assets/figures/ml_fig_c436_02.png)
*Figure — Online mistake bound c436. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Artificial intelligence aims at systems that act intelligently; machine learning improves task performance by estimating models from data; data mining emphasizes scalable pattern discovery; data science governs measurement and decisions under uncertainty. A short history runs from symbolic AI through statistical learning and data mining to deep learning and generative systems—none fully obsolete. Algorithms are finite procedures; learning algorithms estimate models. Evaluation jointly considers computational complexity, runtime, task accuracy, and accuracy–efficiency trade-offs. Datasets arrive as tables, time series, streams, graphs, text, images/video, and audio, with numerical types that constrain encoding. Tasks include clustering, association and sequence mining, dimensionality reduction, anomaly detection, regression, classification, self-supervision, generative modeling, and reinforcement learning. Ground-truth labels are measurement processes; k-fold and grouped cross-validation support honest internal evaluation but do not replace external validation. A step-by-step workflow ties question framing to monitoring. The worked example selected a simpler model with better validation error over a flexible overfit rule. In neurology and epidemiology, cohort eligibility, index time, phenotype quality, leakage control, calibration, and external validation determine whether ML claims are scientifically meaningful. Recurring failure modes—data leakage, distribution shift, class imbalance, label noise, multiplicity, shortcut learning, and miscalibration—each carry a preventive discipline. Fairness, privacy, and transparency are design constraints to encode from the first workflow step, previewed here and developed in Chapter 16.

## Practice and Reflection

(1) Write a one-paragraph problem statement for 90-day stroke recurrence prediction. Identify whether it is primarily supervised, unsupervised, or RL, and justify the choice.

(2) In two sentences each, distinguish AI, ML, data mining, and data science for a hospital steering committee.

(3) Give big-O training and inference costs for naive dense linear regression (normal equations) in terms of n and d, and explain one practical factor big-O ignores.

(4) A seizure detector reports 99% accuracy on a stream with 1% seizure windows. Why is this insufficient? Propose two better metrics for operations.

(5) Classify each data source as tabular, temporal, stream, graph, text, image, or audio: (a) CTA slices, (b) continuous EEG alarm feed, (c) inter-hospital transfer network, (d) discharge summary, (e) registry row of comorbidities.

(6) Match tasks to methods at a high level: phenotype discovery without labels; next-token pretraining on notes; learning a titration policy in simulation; predicting dichotomized mRS.

(7) Define ground truth for large-vessel occlusion status in three different ways (imaging adjudication, NLP on reports, procedure codes) and state one bias of each.

(8) Sketch 5-fold grouped CV by patient for a dataset with 100 patients and 250 encounters. Why is row-wise CV unsafe here?

(9) List the workflow steps you would refuse to skip before deploying an ED triage model, and name one failure mode each step prevents.

(10) Using the worked example numbers, recompute which model wins if validation cost is 2·FN + 1·FP and Model A has 1 FN + 1 FP while Model B has 2 FN + 1 FP on the 6 validation points.

(11) Define index time, look-back, and outcome window for predicting symptomatic ICH after IV thrombolysis. List two features that would constitute temporal leakage.

(12) A stroke risk model has AUROC 0.82 internally but is poorly calibrated at a rural hospital. What does this mean clinically, and what steps would you take?

(13) A stroke early-warning model reports pooled AUROC 0.80 but was never evaluated by subgroup. Explain two distinct mechanisms—label bias and representation bias—by which it could still harm an under-served group, and name the disaggregated metrics you would require before deployment.

(14) A bedside model is scored 24 hours after admission (index time). For each candidate field, state whether it is a legitimate inference-time feature or temporal leakage, and justify using index time: (a) admission NIHSS, (b) discharge disposition, (c) first troponin drawn at 6 hours, (d) final adjudicated TOAST subtype.

(15) At an input where the true mean response is 12 and σ² = 2, Model P predicts 12 in every resample while Model Q predicts 8 and 16 with equal probability across resamples. Compute (bias)², variance, and total expected squared error for each, and state which you would prefer and why.
