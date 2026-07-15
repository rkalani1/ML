# Chapter 7. Dimensionality Reduction and Data Decomposition


![07 Pca Projection](../assets/figures/07_pca_projection.png)


## Opening

A multiparametric MRI radiomics pipeline has 1,200 features and 180 patients. Dimensionality reduction is not optional aesthetics; it is survival against overfitting and irreproducible stroke biomarkers.


![Dimensionality reduction intuition along a dominant axis (original).](../assets/figures/ml_fig_pca.png)

*Dimensionality reduction intuition along a dominant axis (original).*
## Learning Objectives

Explain the curse of dimensionality and how it motivates projection and decomposition in wide clinical and omics matrices.

Derive PCA from covariance eigenstructure, compute a small PCA by hand, and relate PCA to SVD and incremental PCA.

State the goals of LDA/Fisher discriminants versus unsupervised PCA for labeled neurologic phenotypes.

Contrast nonlinear embeddings LLE, t-SNE, and UMAP and list caveats for cluster interpretation.

Describe Fourier and wavelet decompositions and approximate aggregation for signals and time series.

Apply matrix factorizations: Cholesky, NMF, SVD; and topic models LSI and LDA on clinical text matrices.

Define tensors, mode-n unfolding, and CP, Tucker, and tensor-train decompositions at a conceptual level.

Decide when reduction helps versus when it erases rare but critical clinical signals.

## High Dimensions Are Not Just More Numbers

Modern neurologic datasets often describe each patient or sample with hundreds or millions of coordinates: CT perfusion maps vectorized into voxels, multiparametric MRI features, gene expression probes, proteomic peaks, continuous EEG channels, wearable accelerometry windows, or wide EHR panels joining every lab ordered during an admission. High dimensionality brings statistical and geometric pathologies collectively called the curse of dimensionality.

Distances concentrate: the difference between nearest and farthest neighbors shrinks relative to typical distances, weakening distance-based phenotyping. Sample complexity grows: covering a unit hypercube with ε-balls requires a number of samples exponential in dimension—fatal when n is a few hundred stroke patients and p is tens of thousands of omics features. Multicollinearity among labs and noise dimensions obscure signal.

Dimensionality reduction and matrix decomposition seek lower-dimensional structure—linear subspaces, nonnegative parts, independent sources, spectral components, or nonlinear manifolds—that preserve information needed for visualization, compression, denoising, or downstream learning. Reduction is not universally beneficial. Discarding dimensions can erase rare but critical signals, harm calibration, or violate interpretability constraints. This chapter develops PCA rigorously enough to compute by hand, surveys linear and nonlinear alternatives, signal decompositions, matrix and tensor factorizations, and topic models, with clinical–epidemiologic decision rules for when not to reduce.

## The Curse of Dimensionality

Consider points drawn uniformly in the d-dimensional unit hypercube. The volume of a thin shell near the boundary dominates as d grows; most mass sits near the surface, not the center. For Gaussian distributions, most probability mass concentrates in a thin annular region at a radius that grows like √d. Pairwise Euclidean distances between random points become relatively similar, so nearest-neighbor search for “similar stroke phenotypes” loses contrast when features are numerous, weakly scaled, and noisy.

![7.1: Two faces of the curse of dimensionality for points drawn uniformly in the unit hypercube. (a) The ratio of each query p](../assets/figures/ml_concept_7.1_1766b92d.png)

*Figure 7.1 — original teaching graphic.*

Statistically, estimating a full covariance matrix requires on the order of d² parameters; with n ≪ d, the sample covariance is singular and unstable—routine when a multi-lab panel plus comorbidity indicators exceeds the number of ICH admissions in a single-center year. Lipschitz concentration inequalities formalize that smooth functions of many independent variables concentrate tightly, which is both a blessing (stable averages) and a curse (little local structure for learning).

Geometry: nearest-neighbor contrast collapses as d grows with fixed n.

Statistics: covariance and density estimation need n that grow quickly with d.

Computation: distances and kernels cost O(n²d) or worse at scale.

Clinical reality: p ≫ n is normal in imaging-derived and omics stroke research.

## Linear Dimensionality Reduction: PCA, Incremental PCA, LDA, and Fisher

### Principal Component Analysis

Principal component analysis (PCA) finds orthogonal directions that capture maximal variance in a centered data matrix. Let X be an n × d matrix with rows as patients (or samples) and columns as features. Form the centered matrix X_c by subtracting the column means. The sample covariance may be written C = (1/n) X_cᵀ X_c (some texts use n−1). PCA solves for unit vectors u that maximize uᵀ C u, the variance of the projected scalar X_c u. The maximizer is the leading eigenvector of C; the maximal variance is the leading eigenvalue λ₁. Subsequent components maximize variance subject to orthogonality to previous directions.

![7.2: A scree plot of a covariance eigenvalue spectrum (bars, left axis) with the cumulative fraction of variance explained ov](../assets/figures/ml_concept_7.2_01c9aabd.png)

*Figure 7.2 — original teaching graphic.*

![PCA: cumulative variance explained and reconstruction error vs k (synthetic spectrum; original).](../assets/figures/ml_fig_pca_variance_recon.png)

*Figure — PCA compression curves on a synthetic eigenvalue spectrum. Left: per-component and cumulative fraction of variance; the 80% guideline marks a common default k, not a clinical warranty. Right: relative Frobenius reconstruction error falls as rank grows (Eckart–Young residual energy). Variance retained is a reconstruction metric—it does not guarantee better prediction of mRS, infarct growth, or any downstream label.*

![PCA scree vs parallel analysis for component retention (synthetic; original).](../assets/figures/ml_fig_pca_parallel.png)

*Figure — Parallel analysis as a retention check. **Left:** data eigenvalues vs mean eigenvalues of column-shuffled null matrices. **Right:** retain components above the null mean (teal bars). Still a geometric heuristic—PCA axes are not causal factors and need not be the best inputs for a clinical predictor.*

![NMF as nonnegative additive parts X≈WH (synthetic; original).](../assets/figures/ml_fig_nmf_parts_matrix.png)

*Figure — Nonnegative matrix factorization. **Left:** reconstructed data. **Middle/Right:** parts W and coefficients H. Additive parts aid interpretability for count-like panels but are still latent factors—not proven causal disease modules.*

If C = V Λ Vᵀ with eigenvalues λ₁ ≥ λ₂ ≥ ⋯ ≥ λ_d ≥ 0 and orthonormal eigenvectors as columns of V, then the k-dimensional PCA scores are Z = X_c V_k where V_k holds the top k eigenvectors. The fraction of variance explained by component j is λ_j / ∑_i λ_i. Cumulative explained variance guides the choice of k—but “80% variance retained” is not a guarantee of predictive utility for mRS or infarct growth. PCA assumes that large-variance directions are interesting, which fails when the signal is low-variance or when units of measurement arbitrarily inflate variance—hence the common practice of standardizing features before PCA when panels mix mg/dL, mmHg, and seconds.

### Teaching table: choosing k in PCA (compression vs claim)

| Criterion | What it optimizes | Safe clinical reading | Common failure |
|---|---|---|---|
| Cumulative variance (e.g. 80–95%) | Reconstruction energy of X | Good for denoising / visualization budgets | Treats large-variance noise as “signal” |
| Reconstruction error vs k (elbow) | Frobenius residual of rank-k map | Useful when the goal is compression | Elbow is subjective; not a p-value |
| Cross-validated downstream loss | Task error (AUC, RMSE, net benefit) | Correct when PCA is a **pipeline step** | Tuning k on the test set = leakage |
| Fixed k from prior study | Reproducibility of a published pipeline | OK only if features and units match | Silent shift when labs or scanners change |
| Supervised alternative (LDA / PLS) | Label separation, not variance | Prefer when phenotype labels exist and n/class is adequate | Unstable with rare subtypes |

Rule of thumb: pick k for the **claim you will make**. If the claim is “compressed representation for a predictor,” validate the full pipeline—including the choice of k—inside patient-grouped cross-validation, never on the locked test set.

Interpretation: loadings (entries of eigenvectors) show how original features contribute to each component. A first PC of a metabolic panel might load on renal function tests together; a first PC of voxelwise DWI intensities might reflect overall lesion burden. Loadings are not causal path coefficients.

### Worked Example: PCA on Three Points in 2D

We compute PCA fully by hand on n = 3 points in d = 2. Think of each point as a miniature two-feature patient: (x₁, x₂) might represent standardized (onset-to-arrival, NIHSS) for illustration—the algebra is what matters.
x₁ = (1, 2), x₂ = (3, 3), x₃ = (5, 4).
Mean μ = ((1+3+5)/3, (2+3+4)/3) = (3, 3). Centered data:
x₁_c = (−2, −1), x₂_c = (0, 0), x₃_c = (2, 1).

![7.3: The chapter's three-point PCA worked example, with x₂ nudged to (3, 4) so the second component is non-degenerate. The po](../assets/figures/ml_concept_7.3_a11e16f2.png)

*Figure 7.3 — original teaching graphic.*

Form X_c with rows as centered points. Then X_cᵀ X_c = [[8, 4], [4, 2]]. Using C = (1/n) X_cᵀ X_c = [[8/3, 4/3], [4/3, 2/3]]. Eigenvalues solve det(C − λI) = 0. C − λI = [[8/3−λ, 4/3], [4/3, 2/3−λ]]. Determinant: (8/3−λ)(2/3−λ) − (4/3)² = λ² − (10/3)λ. Thus λ(λ − 10/3) = 0. Eigenvalues: λ₁ = 10/3 ≈ 3.333, λ₂ = 0. Two independent identities confirm the algebra before we go further: eigenvalues must sum to the trace, λ₁ + λ₂ = 10/3 = 8/3 + 2/3 ✓, and multiply to the determinant, λ₁ · λ₂ = 0 = det(C) ✓—zero here because C is singular. Data are exactly one-dimensional after centering: all centered points lie on the line spanned by (2, 1).

For λ₁ = 10/3, solve (C − λ₁ I)v = 0 to get v₁ = (2, 1)/√5 ≈ (0.894, 0.447). Orthogonal v₂ = (−1, 2)/√5. Projected scores on the first PC: z₁ = −√5 ≈ −2.236, z₂ = 0, z₃ = √5 ≈ 2.236. Variance of scores (population form) matches λ₁: (1/3)((−√5)² + 0 + (√5)²) = 10/3. Variance explained by PC1 is 100% because λ₂ = 0. Reconstructing with one component recovers the centered data exactly. This hand computation shows the full pipeline: center, form covariance, eigen-decompose, project, and interpret eigenvalues as explained variance.

### SVD Connection and Incremental PCA

The singular value decomposition writes X_c = U Σ Wᵀ. The right singular vectors (columns of W) are principal directions; they equal the eigenvectors of X_cᵀ X_c. Singular values relate to eigenvalues by λ_j = σ_j² / n (with the 1/n covariance convention). PCA scores are proportional to left singular vectors scaled by singular values. Computing PCA via SVD of X_c is numerically stabler than forming C explicitly—especially when d is large or features are collinear.

Truncated SVD keeps the top k components and yields the best rank-k approximation to X_c in Frobenius norm (Eckart–Young). Incremental PCA updates low-rank approximations as mini-batches arrive, enabling PCA on datasets that do not fit in memory—streaming EHR feature dumps, multi-site federated summaries, or large imaging cohorts. Trade-offs include approximate components and sensitivity to batch order; still invaluable when classical full SVD is impossible.

### Worked Example: SVD of a 2×2 Matrix

The singular value decomposition sits behind truncated-SVD compression and the PCA–SVD link just described, so it repays grinding one small case entirely by hand. Take the 2×2 matrix A = [[1, 2], [2, 1]]. The recipe has four steps: form AᵀA, find its eigenvalues, take square roots for the singular values σ, then read off the best rank-1 approximation.

![7.4: Truncated-SVD low-rank approximation of a synthetic nonnegative image, the Eckart–Young optimum in Frobenius norm. Keepi](../assets/figures/ml_concept_7.4_61ba1117.png)

*Figure 7.4 — original teaching graphic.*

Because this A is symmetric, Aᵀ = A, so AᵀA = [[1, 2], [2, 1]]·[[1, 2], [2, 1]] = [[5, 4], [4, 5]] (top-left entry 1·1 + 2·2 = 5; each off-diagonal 1·2 + 2·1 = 4). Eigenvalues solve det(AᵀA − λI) = (5 − λ)² − 4² = 0, so (5 − λ)² = 16 and 5 − λ = ±4, giving λ₁ = 9 and λ₂ = 1. Two independent identities confirm the algebra before we continue: the eigenvalues sum to the trace, 9 + 1 = 10 = 5 + 5 ✓, and multiply to the determinant, 9 · 1 = 9 = 25 − 16 ✓. The singular values are the square roots of the eigenvalues: σ₁ = √9 = 3 and σ₂ = √1 = 1.

The right singular vectors are the eigenvectors of AᵀA. For λ₁ = 9, the matrix AᵀA − 9I = [[−4, 4], [4, −4]] forces −4a + 4b = 0, so v₁ ∝ (1, 1), i.e. v₁ = (1, 1)/√2. For λ₂ = 1, AᵀA − I = [[4, 4], [4, 4]] forces a = −b, so v₂ = (1, −1)/√2. The matching left singular vectors follow from u_i = A v_i / σ_i: u₁ = A·(1, 1)ᵀ/√2 ÷ 3 = (3, 3)ᵀ/√2 ÷ 3 = (1, 1)/√2, and u₂ = A·(1, −1)ᵀ/√2 ÷ 1 = (−1, 1)/√2. Thus A = U Σ Vᵀ with singular values 3 and 1.

The best rank-1 approximation (the Eckart–Young optimum in Frobenius norm) keeps only the largest singular triplet: A₁ = σ₁ u₁ v₁ᵀ = 3 · (1/√2, 1/√2)ᵀ (1/√2, 1/√2) = 3 · [[1/2, 1/2], [1/2, 1/2]] = [[1.5, 1.5], [1.5, 1.5]]. Verify through the residual: A − A₁ = [[−0.5, 0.5], [0.5, −0.5]], which is exactly the discarded triplet σ₂ u₂ v₂ᵀ = 1 · [[−1/2, 1/2], [1/2, −1/2]] ✓. The energy bookkeeping closes too: ‖A‖_F² = 1² + 2² + 2² + 1² = 10 = σ₁² + σ₂² = 9 + 1, and the rank-1 residual carries precisely ‖A − A₁‖_F² = σ₂² = 1. Retaining the top singular value therefore preserves 9/10 = 90% of the squared Frobenius energy—the matrix analog of PCA’s explained-variance fraction, and the exact mechanism behind truncated-SVD denoising of term–document and imaging matrices.

### LDA and Fisher Linear Discriminant

Linear Discriminant Analysis (LDA) for dimensionality reduction is supervised: it seeks projections that maximize between-class scatter relative to within-class scatter. For two classes, Fisher’s linear discriminant finds w maximizing (μ₁ − μ₀)² / (s₁² + s₀²) along the projected line, equivalently maximizing wᵀ S_B w / wᵀ S_W w with between- and within-class scatter matrices S_B and S_W. Multi-class LDA generalizes via generalized eigenvectors of S_W⁻¹ S_B, yielding at most C−1 discriminative directions for C classes.

Unlike PCA, LDA can discard high-variance directions that do not separate labels and amplify low-variance directions that do. It assumes roughly shared within-class covariances and needs enough samples per class to estimate S_W stably—problematic for rare stroke subtypes. Regularized or shrinkage LDA helps. Use LDA projections for visualization and as features for simple classifiers; do not confuse LDA-as-reduction with LDA-as-classifier, though they share mathematics.

![LDA vs PCA projection on elongated two-class data (synthetic; original).](../assets/figures/ml_fig_lda_vs_pca.png)

*Figure — Supervised vs unsupervised axes. **Left:** two classes share a high-variance diagonal; PC1 (deep) follows variance while Fisher/LDA (red) aims at class-mean separation. **Right:** 1-D projections—LDA often unmixes classes that PC1 still overlaps. LDA needs labels and enough per-class samples for S_W; PCA is not a classifier. Neither projection is a causal mechanism map.*

### Worked Example: Fisher’s Linear Discriminant on Two Small Classes

The exposition above states Fisher’s criterion abstractly; here we drive the whole computation on two tiny, clearly separable classes so every matrix is checkable by hand. Let Class A = {(1, 1), (2, 1), (1, 2)} and Class B = {(4, 4), (5, 4), (4, 5)}—two small triangular clouds sitting at opposite ends of the (1, 1) diagonal.

![7.5: Fisher's linear discriminant versus PCA on two classes that share an elongated within-class covariance. PCA (amber) poin](../assets/figures/ml_concept_7.5_37a7391b.png)

*Figure 7.5 — original teaching graphic.*

Class means. μ_A = ((1+2+1)/3, (1+1+2)/3) = (4/3, 4/3) ≈ (1.33, 1.33) and μ_B = ((4+5+4)/3, (4+4+5)/3) = (13/3, 13/3) ≈ (4.33, 4.33). Their difference is μ_A − μ_B = (4/3 − 13/3, 4/3 − 13/3) = (−3, −3).

Within-class scatter. Each within-class scatter matrix sums the outer products (x − μ)(x − μ)ᵀ over its class’s points. For Class A the three deviations from μ_A = (4/3, 4/3) are (−1/3, −1/3), (2/3, −1/3), and (−1/3, 2/3); their outer products are [[1/9, 1/9], [1/9, 1/9]], [[4/9, −2/9], [−2/9, 1/9]], and [[1/9, −2/9], [−2/9, 4/9]]. Summing entrywise, each diagonal term is 1/9 + 4/9 + 1/9 = 6/9 = 2/3 and each off-diagonal term is 1/9 − 2/9 − 2/9 = −3/9 = −1/3, so S_A = [[2/3, −1/3], [−1/3, 2/3]]. Class B is Class A translated by (3, 3), so its deviations—and hence its scatter—are identical: S_B = [[2/3, −1/3], [−1/3, 2/3]]. The pooled within-class scatter is S_W = S_A + S_B = [[4/3, −2/3], [−2/3, 4/3]].

Invert S_W. Using the 2×2 rule [[a, b], [c, d]]⁻¹ = (1/(ad − bc))·[[d, −b], [−c, a]], the determinant is (4/3)(4/3) − (−2/3)(−2/3) = 16/9 − 4/9 = 12/9 = 4/3. Hence S_W⁻¹ = (1/(4/3))·[[4/3, 2/3], [2/3, 4/3]] = (3/4)·[[4/3, 2/3], [2/3, 4/3]] = [[1, 1/2], [1/2, 1]]. Check S_W S_W⁻¹ = I: top-left (4/3)(1) + (−2/3)(1/2) = 4/3 − 1/3 = 1 ✓; top-right (4/3)(1/2) + (−2/3)(1) = 2/3 − 2/3 = 0 ✓; the other two entries mirror these to 0 and 1.

Fisher direction. w ∝ S_W⁻¹(μ_A − μ_B) = [[1, 1/2], [1/2, 1]]·(−3, −3) = ((1)(−3) + (1/2)(−3), (1/2)(−3) + (1)(−3)) = (−9/2, −9/2). Only direction matters, so w ∝ (1, 1) (equivalently (−1, −1)): the discriminant points straight along the diagonal that separates the clouds, orthogonal to the shared within-class spread.

Projected scores. Using w = (1, 1), project one representative from each class: for Class A’s (1, 1), wᵀx = 1 + 1 = 2; for Class B’s (4, 4), wᵀx = 4 + 4 = 8. Projecting every point gives Class A scores {2, 3, 3} and Class B scores {8, 9, 9}—two tight, non-overlapping intervals with a wide gap, so any threshold in (3, 8), for example 5.5, separates the classes perfectly. Here an unsupervised PCA of the pooled six points would also land near (1, 1), but only because the between-class shift happens to be the largest variance source; Fisher earns that axis deliberately, by maximizing the ratio wᵀS_B w / wᵀS_W w of between- to within-class scatter—which is what keeps it pointed at the discriminative direction even when that direction is not the highest-variance one.

## Nonlinear Dimensionality Reduction: LLE, t-SNE, and UMAP

Linear methods preserve global variance or linear discrimination but fail on curved manifolds: a Swiss roll needs unrolling, not a plane fit. Nonlinear neighbor embeddings aim to preserve local geometry in a low-dimensional map used primarily for visualization and exploratory phenotyping.

### Locally Linear Embedding (LLE)

LLE assumes each point lies on a locally linear patch of a manifold. For each point x_i, find k nearest neighbors and reconstruct x_i as a weighted combination of those neighbors, minimizing reconstruction error with weights summing to one. Then find low-dimensional coordinates y_i that respect the same weights, minimizing ∑ ‖y_i − ∑_j W_{ij} y_j‖² under centering and unit-covariance constraints. LLE can unroll smooth manifolds but is sensitive to k, noise, and gaps in sampling—common issues in small clinical cohorts.

### t-SNE

t-distributed Stochastic Neighbor Embedding (t-SNE) converts high-dimensional distances into conditional probabilities of neighborhood and finds low-dimensional points whose Student-t similarities match those probabilities by minimizing a KL divergence. The heavy-tailed t-distribution in the embedding space allows moderate distances to map farther apart, helping form tight visual clusters. Perplexity controls effective neighborhood size.

![t-SNE perplexity changes the picture (teaching caricature; original).](../assets/figures/ml_fig_tsne_perplexity.png)

*Figure — Perplexity teaching caution (caricature of geometry, **not** a real t-SNE optimization). **Left:** low effective neighborhood size tears a continuum (gray bridge) into noise islands and tightens local clumps. **Right:** broader neighborhoods compress gaps and keep the bridge—but inter-cluster distance still is not a metric effect size. Confirm candidate structure in original (or PCA) space; embeddings are hypothesis generators, not causal maps.*

Caveats are essential for scientific use. t-SNE does not preserve global distances or densities reliably; cluster sizes and between-cluster gaps are not trustworthy as effect sizes. Results depend on perplexity, learning rate, and random initialization. Never report “t-SNE clusters” as phenotypes without confirmatory analysis in the original feature space or with stable supervised labels. For multi-site imaging, batch effects can dominate the embedding.

### UMAP

Uniform Manifold Approximation and Projection (UMAP) constructs a fuzzy topological representation of the high-dimensional data and finds a low-dimensional layout that preserves that structure, grounded in Riemannian geometry and algebraic topology motivations. Empirically UMAP often preserves more global structure than t-SNE, scales better, and supports transform of new points more naturally. Hyperparameters (n_neighbors, min_dist, metric) still strongly affect plots. The same scientific cautions apply: embeddings are hypotheses generators, not automatic cluster proofs.

![n_neighbors caricature: local vs global neighbor-embedding layouts (teaching force sketch; original).](../assets/figures/ml_fig_umap_neighbors.png)

*Figure — n_neighbors tradeoff (force-layout caricature of the hyperparameter, **not** a production UMAP run). **Left:** original 2-D moons + blob. **Middle:** small neighborhoods emphasize local patches and can tear global arrangement. **Right:** large neighborhoods keep more global placement but blur fine structure. Always confirm candidate clusters in original/PCA space; map distances are not effect sizes and not causal subtypes.*

### How Neighbor Embeddings Mislead: A Reading Guide

Because t-SNE and UMAP optimize local neighborhoods while sacrificing global geometry, a handful of specific misreadings recur in imaging and omics work, each with a clinical failure mode. Cluster size is an artifact: both methods equalize local density, so a tight homogeneous group and a diffuse heterogeneous one can occupy similar map area—never read a cluster’s visual spread as biological heterogeneity or variance. Between-cluster distance is an artifact: two blobs far apart on the page may be no more dissimilar than two that touch, so do not rank subtypes by apparent separation. Point density is an artifact: crowding in the map does not track sample density in feature space. Most dangerous clinically, these methods can tear a genuine continuum into apparently discrete islands, or merge distinct groups into one—a smooth severity gradient (rising infarct volume, a titrated biomarker) can render as several disconnected clusters, inviting a false claim of discrete stroke subtypes when the biology is a continuum.

![7.6: Why neighbor-embedding maps must not be read literally. The identical three clusters (n = 130 each) are laid out two way](../assets/figures/ml_concept_7.6_4b3f1db0.png)

*Figure 7.6 — original teaching graphic.*

Two engineering realities sharpen the warning for p ≫ n biomedical data. Spurious clusters form easily: at small perplexity or n_neighbors, even isotropic Gaussian noise fragments into crisp-looking groups, so a compelling map is not evidence that structure exists. And technical variation dominates: in multi-site radiomics the leading axis of a UMAP is often scanner, sequence, or reconstruction kernel; in single-cell or bulk omics it is often library size, batch, or ancestry rather than cell type or disease—and it shifts again if highly-variable-feature selection or normalization changes. The discipline follows directly. Treat every embedding as a hypothesis generator, never proof. Confirm candidate clusters in the original (or PCA-reduced) feature space and quantify separation there—silhouette, or a held-out classifier—not by eye. Overlay known technical covariates (site, scanner, batch, sequencing run, ancestry PCs) and verify the clusters are not merely those. Regenerate the map across several random seeds and a range of perplexity/n_neighbors, keep only structure that persists, and report those settings. In a manuscript, a neighbor-embedding figure should illustrate a conclusion reached by other means, not carry the inferential weight of “natural clusters” by itself.

## Signal and Time Series Decomposition: Fourier, Wavelets, and Aggregation

### Fourier Transform

The Fourier transform decomposes a signal into sinusoidal frequency components. For discrete sampled series, the Discrete Fourier Transform (DFT) and its fast implementation (FFT) yield complex coefficients whose magnitudes form a power spectrum. Band powers (delta, theta, alpha, beta, gamma in EEG; respiratory and cardiac bands in physiology) become compact features. Fourier analysis assumes stationarity over the analysis window; sliding-window spectrograms (STFT) track time-varying spectra.

### Wavelet Transform

Wavelets provide simultaneous time and frequency localization by correlating the signal with scaled and translated wavelets (mother wavelet family). The Continuous Wavelet Transform (CWT) offers fine time–frequency maps; the Discrete Wavelet Transform (DWT) uses critically sampled filter banks for compact multi-resolution coefficients—approximation and detail coefficients at dyadic scales. Wavelets excel for transient events: seizure onsets, spikes, and non-stationary bursts that a global Fourier basis smears across time.

Choice of mother wavelet (Haar, Daubechies, Morlet, symlets) is a modeling decision. Thresholding wavelet coefficients implements denoising (wavelet shrinkage). For clinical ML, wavelet energy features at selected scales often feed classical classifiers when labeled EEG or sensor datasets are modest in size.

### Approximate Aggregation Methods

Long time series are often aggregated: piecewise aggregate approximation (PAA) replaces equal-length segments by their means; symbolic aggregate approximation (SAX) discretizes those means into symbols for motif and distance computations; histograms and sketches summarize distributions in windows. Aggregation reduces dimensionality and noise at the cost of fine temporal detail. Align segment lengths with clinical epochs (minutes around a code stroke, nightly sleep cycles) rather than arbitrary power-of-two convenience alone.

## Matrix Decomposition: Cholesky, NMF, SVD, and Topic Models

### Cholesky Decomposition

Any symmetric positive definite matrix A admits a Cholesky factorization A = L Lᵀ with L lower triangular and positive diagonal. Cholesky is the workhorse for solving A x = b (forward and back substitution), for sampling multivariate Gaussians (if Σ = L Lᵀ then L z with z ~ N(0,I) has covariance Σ), and for stable covariance manipulations in Gaussian processes and Kalman filters. In reduction pipelines, Cholesky appears inside whitening and in numerical linear algebra backends rather than as a “feature method” per se—but understanding it demystifies many library routines.

### Non-negative Matrix Factorization (NMF)

NMF approximates a nonnegative matrix X ≈ W H with W ≥ 0, H ≥ 0. Nonnegativity often yields parts-based, additive representations: topics as bags of words, imaging basis patterns as positive activations, or lab factors as co-elevated panels. Objectives include Frobenius loss or Kullback–Leibler divergence between X and W H, optimized by multiplicative updates or projected gradient methods. NMF is nonconvex; initializations matter. Rank r is a hyperparameter chosen by stability, held-out reconstruction, or domain interpretability.

![NMF parts-based factorization of a toy nonnegative image (original).](../assets/figures/ml_fig_nmf_parts.png)\n![PCA scree and cumulative explained variance (synthetic eigenvalues; original).](../assets/figures/ml_fig_pca_scree_cumvar.png)

*Figure — Rank choice heuristics. Bars are per-component variance fractions; gold curve is cumulative mass with a 90% guide line. Elbows and thresholds are teaching tools—principal axes are not automatically clinical factors without labels. **Geometry ≠ etiology**.*\n\n


![Random projection distance preservation at k=5,20,50 (synthetic; original).](../assets/figures/ml_fig_random_projection.png)

*Figure — JL-style intuition. Pairwise distances recover better as projection dimension grows. Preserving distances is a geometric guarantee—not semantic or causal meaning of axes.*


![ICA teaching sketch: mixed channels vs source-like recoveries (synthetic; original).](../assets/figures/ml_fig_ica_sources.png)

*Figure — Blind source separation cartoon. Recovered components can look like generators under independence assumptions—they are statistical, not automatic clinical mechanisms.*


![Kernel PCA-style unwrap of a circular manifold (synthetic; original).](../assets/figures/ml_fig_kpca_unwrap.png)

*Figure — Nonlinear dim-reduction cartoon. Input ring becomes an unwrapped coordinate plus noise. Coordinates are features for models—not causes.*


![Autoencoder reconstruction error vs bottleneck dimension (synthetic; original).](../assets/figures/ml_fig_ae_bottleneck.png)

*Figure — Larger latents reduce recon error with diminishing returns. Compression quality is not semantic or causal meaning.*


![Cumulative variance with 90 percent guide (synthetic; original).](../assets/figures/ml_fig_cumvar_guide.png)

*Figure — Heuristics for rank—not etiology. Pred ≠ cause without design.*


![t-SNE early exaggeration cartoon (original).](../assets/figures/ml_fig_tsne_early_exag.png)

*Figure — Hyperparams reshape maps without new biology. t-SNE early exaggeration cartoon Pred != cause without design.*


![screenoise teaching panel (original).](../assets/figures/ml_fig_scree_noise.png)

*Figure — Teaching panel for screenoise. Pred != cause without design.*


![Cycle-34 densify scientific panel 9 (original).](../assets/figures/ml_fig_c34_08.png)

*Figure — Continuous densify panel 9. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 9 (original).](../assets/figures/ml_fig_c35_08.png)

*Figure — Continuous densify panel 9. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 9 (original).](../assets/figures/ml_fig_c36_08.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 9 (original).](../assets/figures/ml_fig_c37_08.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 9 (original).](../assets/figures/ml_fig_c38_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 9 (original).](../assets/figures/ml_fig_c39_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 9 (original).](../assets/figures/ml_fig_c40_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 9 (original).](../assets/figures/ml_fig_c41_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 9 (original).](../assets/figures/ml_fig_c42_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 9 (original).](../assets/figures/ml_fig_c43_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 9 (original).](../assets/figures/ml_fig_c44_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 9 (original).](../assets/figures/ml_fig_c45_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 9 (original).](../assets/figures/ml_fig_c46_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 9 (original).](../assets/figures/ml_fig_c47_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 9 (original).](../assets/figures/ml_fig_c48_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 9 (original).](../assets/figures/ml_fig_c49_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 9 (original).](../assets/figures/ml_fig_c50_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 9 (original).](../assets/figures/ml_fig_c51_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 9 (original).](../assets/figures/ml_fig_c52_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 9 (original).](../assets/figures/ml_fig_c53_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 9 (original).](../assets/figures/ml_fig_c54_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 9 (original).](../assets/figures/ml_fig_c55_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 9 (original).](../assets/figures/ml_fig_c56_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 9 (original).](../assets/figures/ml_fig_c57_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 9 (original).](../assets/figures/ml_fig_c58_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 9 (original).](../assets/figures/ml_fig_c59_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 9 (original).](../assets/figures/ml_fig_c60_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 9 (original).](../assets/figures/ml_fig_c61_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 9 (original).](../assets/figures/ml_fig_c62_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 9 (original).](../assets/figures/ml_fig_c63_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 9 (original).](../assets/figures/ml_fig_c64_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 9 (original).](../assets/figures/ml_fig_c65_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 9 (original).](../assets/figures/ml_fig_c66_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 9 (original).](../assets/figures/ml_fig_c67_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 9 (original).](../assets/figures/ml_fig_c68_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 9 (original).](../assets/figures/ml_fig_c69_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 9 (original).](../assets/figures/ml_fig_c70_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 9 (original).](../assets/figures/ml_fig_c71_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 9 (original).](../assets/figures/ml_fig_c72_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 9 (original).](../assets/figures/ml_fig_c73_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 9 (original).](../assets/figures/ml_fig_c74_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 9 (original).](../assets/figures/ml_fig_c75_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 9 (original).](../assets/figures/ml_fig_c76_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 9 (original).](../assets/figures/ml_fig_c77_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 9 (original).](../assets/figures/ml_fig_c78_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 9 (original).](../assets/figures/ml_fig_c79_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 9 (original).](../assets/figures/ml_fig_c80_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 9 (original).](../assets/figures/ml_fig_c81_08.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

*Figure — NMF teaching panel. A synthetic 16×16 nonnegative “map” is a sum of three blob parts. Multiplicative-update NMF recovers additive parts \(H\) and sample loadings \(W\); reconstruction error falls with rank and plateaus near the true \(r=3\). Parts are ≥0 and additive—useful for territories or topics—but solutions are non-unique, depend on initialization, and do not license causal anatomy labels without external validation.*

### Singular Value Decomposition (SVD)

SVD X = U Σ Vᵀ is the Swiss army knife of matrix analysis: PCA, low-rank denoising, pseudoinverses, and latent semantic structure all flow from it. Truncated SVD compresses term–document matrices and imaging matrices. Compared with NMF, SVD allows negative loadings, which can be harder to interpret as “parts” but optimal for squared-error low-rank approximation.

### Topic Modeling: LSI and LDA

Latent Semantic Indexing (LSI), also called Latent Semantic Analysis (LSA), applies truncated SVD to a term–document matrix (often TF–IDF weighted). Documents and terms map into a latent semantic space where synonymy and polysemy are partially mitigated: documents that share themes but few exact terms can still be near. Query retrieval becomes cosine similarity in the reduced space. LSI is linear algebra, not a generative probabilistic model.

Latent Dirichlet Allocation (LDA—not to be confused with Linear Discriminant Analysis) is a generative Bayesian model: each document is a mixture of topics; each topic is a distribution over words; Dirichlet priors encourage sparse mixtures. Inference (variational EM or collapsed Gibbs sampling) estimates topic–word and document–topic distributions. LDA topics are often more interpretable than LSI components for literature corpora and note collections, but require choosing the number of topics and careful preprocessing (stopwords, domain terms). In clinical epi, topics can surface documentation themes; they are not automatic diagnosis codes and can reflect hospital templates rather than disease biology.

## Tensor Decompositions

Tensors generalize matrices to multi-way arrays. A third-order tensor X ∈ R^{I×J×K} might store patients × labs × time, or height × width × modality, or subjects × voxels × conditions. Multi-way structure is lost if one naively flattens everything into a matrix before PCA; tensor decompositions preserve modes.

### Mode-n Unfolding and n-Mode Products

Mode-n unfolding (matricization) rearranges a tensor into a matrix whose columns (or rows) are mode-n fibers—vectors obtained by fixing all indices except the n-th. The n-mode product multiplies a tensor by a matrix along mode n, transforming that mode’s dimension. These operations are the building blocks of multilinear algebra analogs of matrix products.

### CP, Tucker, and Tensor Train

Canonical Polyadic (CP) decomposition, also called PARAFAC, expresses a tensor as a sum of rank-1 tensors: X ≈ ∑_{r=1}^R a_r ∘ b_r ∘ c_r (for order 3), with factor matrices collecting the a_r, b_r, c_r. CP rank is the minimal R; computing it is hard, but alternating least squares often finds useful approximations. CP factors can be uniquely identifiable under mild conditions—unlike matrix SVD factors up to rotation—which aids interpretability.

Tucker decomposition writes X ≈ G ×₁ A ×₂ B ×₃ C with a core tensor G and factor matrices along each mode. It generalizes SVD to multi-way data (higher-order SVD is a common orthogonal Tucker form). Tucker is more flexible than CP but the core can be dense; multilinear rank is a tuple of ranks per mode.

Tensor Train (TT) decomposition factorizes a high-order tensor into a chain of third-order cores, achieving compression when TT-ranks are small. TT and related hierarchical formats scale to very high orders better than dense Tucker cores. In neuroimaging and multi-modal fusion, tensor methods remain research-active tools for structured dimensionality reduction.

Mini example of mode thinking: suppose a toy tensor stores 2 patients × 3 regions × 2 timepoints of a nonnegative perfusion summary. Mode-1 fibers are length-2 vectors across patients for fixed region–time; mode-2 fibers are length-3 regional profiles; mode-3 fibers are length-2 time pairs. A rank-1 CP term a ∘ b ∘ c says that a single patient loading vector a, region pattern b, and time pattern c combine multiplicatively. If true structure is approximately rank-1, reconstruction is compact; if each patient has idiosyncratic region–time interactions, you need higher rank or a Tucker core that allows interactions among factor dimensions. This vocabulary is enough to read neuroimaging tensor papers without implementing ALS from scratch on day one.

## Worked PCA Numerics: Scores, Loadings, and Stability

Return to the three-point example with an extra check. Centered matrix rows: r1=(−2,−1), r2=(0,0), r3=(2,1). Unit loading vector v₁=(2/√5, 1/√5). Scores z_i = r_i · v₁ give (−√5, 0, √5). The second loading v₂=(−1/√5, 2/√5) gives scores all zero, confirming zero variance on PC2. The projection matrix P = v₁ v₁ᵀ equals (1/5) [[4, 2], [2, 1]]. Applying P to r1: (1/5)(−8−2, −4−1)=(−2,−1), recovering r1 exactly—as expected for rank-1 data.

Now alter the data slightly so PC2 is nonzero: replace x₂ with (3, 4) instead of (3, 3). New mean μ=((1+3+5)/3,(2+4+4)/3)=(3, 10/3). Centered rows become (−2, −4/3), (0, 2/3), (2, 2/3). Forming C and solving eigenvalues is a good homework exercise; qualitatively, points no longer collinear, λ₂ > 0, and a one-component reconstruction incurs positive Frobenius error. This is the usual clinical regime: choose k so that residual structure is mostly noise relative to the downstream task, not so that λ₂ is mathematically zero.

Loadings for a standardized four-lab panel might read PC1 ≈ 0.55·creatinine + 0.52·BUN + 0.45·potassium + 0.47·phosphate, a “renal–electrolyte” direction. Reporting loadings with confidence requires bootstrap stability: if signs flip across resamples, do not over-narrate the biology. Site-stratified PCA can reveal that PC1 is “scanner A versus B” rather than pathophysiology—plot scores colored by site before publishing a “phenotype axis.”

Incremental PCA on successive weekly batches of the same labs should be monitored for loading drift. If week-12 loadings rotate relative to week-1 because a new assay entered the panel, either freeze the projection learned on a locked training window or retrain and version the component definitions. Unsupervised maps are part of the feature pipeline and inherit the same MLOps discipline as scalers and encoders.

## Whitening, Reconstruction, and When Not to Reduce

Whitening (sphering) transforms data so that the empirical covariance becomes the identity. One form uses the PCA basis: z_white = Λ_k^{−1/2} V_kᵀ x_c after optional truncation to k components. Whitened features have unit variance and zero correlation, which can help independent component analysis and some neural initializations. Whitening amplifies low-variance directions—including assay noise—if small eigenvalues are inverted without truncation or regularization. Prefer truncated or ridge-regularized whitening when eigenvalues span many orders of magnitude, as in multi-omics merged with vitals.

Reconstruction error ‖X_c − X_c V_k V_kᵀ‖_F measures how much variance a k-dimensional PCA keeps. Plotting cumulative explained variance helps pick k, but predictive tasks should choose k (or whether to use PCA at all) by nested validation of the downstream model. A component that explains little variance can still be highly predictive; a dominant component can be pure batch effect (scanner, site, shift).

Do not reduce when stakeholders require feature-level coefficients on original labs; when n is already tiny and reduction adds unstable maps; when rare binary indicators carry the clinical signal; or when reduction is fit on the full cohort before train/test split (leakage of test geometry into the projection). Sometimes the right “reduction” is domain-guided feature selection (Chapter 6), not an unsupervised map.

## Connecting Decompositions Across Modalities

A practical multi-modal stroke workup might: standardize and PCA-compress a wide lab panel to 10 components; extract wavelet band energies from a short EEG montage; run truncated SVD/LSI on TF–IDF of the ED note; leave ASPECTS and NIHSS unreduced as interpretable anchors; then feed the concatenation to a regularized model. Each decomposition is documented with fit-on-train parameters (means, loadings, IDF, wavelet settings).

NMF on nonnegative perfusion maps can yield parts resembling vascular territories; compare qualitatively with known anatomy rather than forcing a territory label onto every factor. Topic models on progress notes may rediscover template headings (“Neuro check,” “Plan”) as “topics”—filter boilerplate before claiming disease themes. Tensor CP on patients × regions × time can summarize longitudinal imaging if missingness and registration quality allow; otherwise collapse time with clinically defined epochs first.

Computational notes: randomized SVD and truncated iterative methods scale to tall thin or wide short matrices common in imaging. Incremental PCA suits nightly batch arrivals from a registry. For t-SNE/UMAP of 100k notes or cells, pre-reduce with PCA to 50 dimensions for speed and noise control—another place where linear and nonlinear methods compose.

Fourier features on a 256-sample EEG window might yield band powers that reduce 256 samples to five numbers (delta through gamma). Wavelet packet energies might yield twenty numbers with better transient sensitivity. Neither automatically beats the other on seizure detection—compare with nested validation and patient-wise splits. LSI on 10,000 notes with vocabulary 20,000 may keep 100 semantic dimensions for retrieval; LDA with K=50 topics yields document mixtures used as features for phenotype classifiers. In each case, the decomposition is a feature engineer’s tool (Chapter 6), not an end in itself.

Cholesky appears when sampling synthetic multivariate lab panels for stress tests: if a training covariance Σ is SPD, factor Σ=LLᵀ and map standard normals through L to match correlations. If Σ is singular because p>n, use a reduced-rank or diagonal-loaded covariance first. This closes the loop from decomposition numerics back to practical simulation for pipeline testing without moving real PHI off-site.

Finally, keep a written decision log: why PCA versus NMF, why k or rank r, which modes a tensor used, whether embeddings were supervised. Future you—and multi-site collaborators—will need that log more than a colorful UMAP. Record random seeds for t-SNE/UMAP and the training window used for any incremental PCA update.

## Clinical and Epidemiologic Notes

Use PCA and SVD to denoise and compress when variance aligns with signal—bulk lesion burden, correlated lab panels—but verify that rare critical features (a single pathogenic mutation, a sparse ASPECTS region) are not discarded. Standardize mixed units before PCA. Prefer SVD implementations over forming huge covariances explicitly.

Supervised LDA projections can improve class separation for visualization of stroke subtypes but need enough events per class and external validation. Nonlinear embeddings (t-SNE, UMAP) are excellent for talks and hypothesis generation; they are poor as sole evidence of “natural clusters” in manuscripts without confirmatory statistics. Always re-cluster or re-label in original space, and report stability across seeds and sites.

For signals, match Fourier versus wavelet features to stationarity and transient structure of the clinical phenomenon. For notes, LSI/LDA topics help explore corpora and build retrieval indices; they do not replace chart review for phenotype gold standards. Multi-way tensors fit multi-modal longitudinal designs conceptually; start simple (matrix methods) unless multi-way structure is clearly needed and n supports it.

Epidemiologic analyses that use principal components as covariates (for ancestry adjustment in genetics, or for lab summaries) should pre-specify how many components and from which samples they are estimated. Using components built with test outcomes in view is circular. When components become exposure proxies, interpretability and measurement error deserve the same scrutiny as any constructed index.

Curse of dimensionality: expect distance concentration and unstable covariances when p ≫ n.

PCA/SVD: variance-maximizing linear maps; worked eigenvalues tell reconstruction fidelity.

LDA/Fisher: supervised axes; limited to C−1 dimensions; shrinkage when n is small.

t-SNE/UMAP: visualize locally; do not over-interpret global geometry.

Fourier/wavelets: complementary spectral tools for EEG and physiology.

NMF/LSI/LDA topics: parts and themes; validate against clinical meaning.

Tensors: preserve multi-way structure (patient × feature × time) when justified.

Whitening and full-rank inversions amplify noise—truncate or regularize.

Fit projections on training data only; validate k by downstream performance.


![c82 teaching panel 08 (original).](../assets/figures/ml_fig_c82_08.png)
*Figure — PCA scree and cumulative variance for choosing component count. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 08 (original).](../assets/figures/ml_fig_c83_08.png)
*Figure — Neighbor/perplexity settings change embedding topology emphasis. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 08 (original).](../assets/figures/ml_fig_c84_08.png)
*Figure — PCA vs ICA teaching contrast on mixed sources. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 08 (original).](../assets/figures/ml_fig_c85_08.png)
*Figure — PCA axes track the covariance ellipse of the cloud. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 08 (original).](../assets/figures/ml_fig_c86_08.png)
*Figure — Reconstruction error versus retained components. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 08 (original).](../assets/figures/ml_fig_c87_08.png)
*Figure — Neighbor-preserving embeddings rearrange geometry. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 08 (original).](../assets/figures/ml_fig_c88_08.png)
*Figure — NMF parts-based non-negative factors. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 08 (original).](../assets/figures/ml_fig_c89_08.png)
*Figure — Sparse coding: few active dictionary atoms. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 08 (original).](../assets/figures/ml_fig_c90_08.png)
*Figure — Autoencoder bottleneck diagram. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 08 (original).](../assets/figures/ml_fig_c91_08.png)
*Figure — Random projection Johnson-Lindenstrauss. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 08 (original).](../assets/figures/ml_fig_c92_08.png)
*Figure — Kernel PCA feature map idea. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 08 (original).](../assets/figures/ml_fig_c93_08.png)
*Figure — UMAP neighbor graph sketch. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 08 (original).](../assets/figures/ml_fig_c94_08.png)
*Figure — Isomap geodesic distances. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 08 (original).](../assets/figures/ml_fig_c95_08.png)
*Figure — Multidimensional scaling stress. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 08 (original).](../assets/figures/ml_fig_c96_08.png)
*Figure — t-SNE early exaggeration phase. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 08 (original).](../assets/figures/ml_fig_c97_08.png)
*Figure — Locally linear embedding patch. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 08 (original).](../assets/figures/ml_fig_c98_08.png)
*Figure — Diffusion map eigenmodes. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 08 (original).](../assets/figures/ml_fig_c99_08.png)
*Figure — PHATE trajectory embedding. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 08 (original).](../assets/figures/ml_fig_c100_08.png)
*Figure — Laplacian eigenmaps. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 08 (original).](../assets/figures/ml_fig_c101_08.png)
*Figure — Autoencoder vs PCA axes. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 08 (original).](../assets/figures/ml_fig_c102_08.png)
*Figure — PaCMAP neighbor preservation. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 08 (original).](../assets/figures/ml_fig_c103_08.png)
*Figure — Hessian eigen spectrum loss. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 08 (original).](../assets/figures/ml_fig_c104_08.png)
*Figure — Factor analysis loadings. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 08 (original).](../assets/figures/ml_fig_c105_08.png)
*Figure — TriMAP triplet constraints. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 08 (original).](../assets/figures/ml_fig_c106_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 08 (original).](../assets/figures/ml_fig_c107_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 08 (original).](../assets/figures/ml_fig_c108_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 08 (original).](../assets/figures/ml_fig_c109_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 08 (original).](../assets/figures/ml_fig_c110_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 08 (original).](../assets/figures/ml_fig_c111_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 08 (original).](../assets/figures/ml_fig_c112_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 08 (original).](../assets/figures/ml_fig_c113_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 08 (original).](../assets/figures/ml_fig_c114_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 08 (original).](../assets/figures/ml_fig_c115_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 08 (original).](../assets/figures/ml_fig_c116_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 08 (original).](../assets/figures/ml_fig_c117_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 08 (original).](../assets/figures/ml_fig_c118_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 08 (original).](../assets/figures/ml_fig_c119_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 08 (original).](../assets/figures/ml_fig_c120_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 08 (original).](../assets/figures/ml_fig_c121_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 08 (original).](../assets/figures/ml_fig_c122_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 08 (original).](../assets/figures/ml_fig_c123_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 08 (original).](../assets/figures/ml_fig_c124_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 08 (original).](../assets/figures/ml_fig_c125_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 08 (original).](../assets/figures/ml_fig_c126_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 08 (original).](../assets/figures/ml_fig_c127_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 08 (original).](../assets/figures/ml_fig_c128_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 08 (original).](../assets/figures/ml_fig_c129_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 08 (original).](../assets/figures/ml_fig_c130_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 08 (original).](../assets/figures/ml_fig_c131_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 08 (original).](../assets/figures/ml_fig_c132_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 08 (original).](../assets/figures/ml_fig_c133_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 08 (original).](../assets/figures/ml_fig_c134_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 08 (original).](../assets/figures/ml_fig_c135_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 08 (original).](../assets/figures/ml_fig_c136_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 08 (original).](../assets/figures/ml_fig_c137_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 08 (original).](../assets/figures/ml_fig_c138_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 08 (original).](../assets/figures/ml_fig_c139_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 08 (original).](../assets/figures/ml_fig_c140_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 08 (original).](../assets/figures/ml_fig_c141_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 08 (original).](../assets/figures/ml_fig_c142_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 08 (original).](../assets/figures/ml_fig_c143_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 08 (original).](../assets/figures/ml_fig_c144_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 08 (original).](../assets/figures/ml_fig_c145_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 08 (original).](../assets/figures/ml_fig_c146_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 08 (original).](../assets/figures/ml_fig_c147_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 08 (original).](../assets/figures/ml_fig_c148_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 08 (original).](../assets/figures/ml_fig_c149_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 08 (original).](../assets/figures/ml_fig_c150_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 08 (original).](../assets/figures/ml_fig_c151_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 08 (original).](../assets/figures/ml_fig_c152_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 08 (original).](../assets/figures/ml_fig_c153_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 08 (original).](../assets/figures/ml_fig_c154_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 08 (original).](../assets/figures/ml_fig_c155_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 08 (original).](../assets/figures/ml_fig_c156_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 08 (original).](../assets/figures/ml_fig_c157_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 08 (original).](../assets/figures/ml_fig_c158_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 08 (original).](../assets/figures/ml_fig_c159_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 08 (original).](../assets/figures/ml_fig_c160_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 08 (original).](../assets/figures/ml_fig_c161_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 08 (original).](../assets/figures/ml_fig_c162_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 08 (original).](../assets/figures/ml_fig_c163_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 08 (original).](../assets/figures/ml_fig_c164_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 08 (original).](../assets/figures/ml_fig_c165_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 08 (original).](../assets/figures/ml_fig_c166_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 08 (original).](../assets/figures/ml_fig_c167_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 08 (original).](../assets/figures/ml_fig_c168_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 08 (original).](../assets/figures/ml_fig_c169_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 08 (original).](../assets/figures/ml_fig_c170_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 08 (original).](../assets/figures/ml_fig_c171_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 08 (original).](../assets/figures/ml_fig_c172_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 08 (original).](../assets/figures/ml_fig_c173_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 08 (original).](../assets/figures/ml_fig_c174_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 08 (original).](../assets/figures/ml_fig_c175_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 08 (original).](../assets/figures/ml_fig_c176_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 08 (original).](../assets/figures/ml_fig_c177_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 08 (original).](../assets/figures/ml_fig_c178_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 08 (original).](../assets/figures/ml_fig_c179_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 08 (original).](../assets/figures/ml_fig_c180_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 08 (original).](../assets/figures/ml_fig_c181_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 08 (original).](../assets/figures/ml_fig_c182_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 08 (original).](../assets/figures/ml_fig_c183_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 08 (original).](../assets/figures/ml_fig_c184_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 08 (original).](../assets/figures/ml_fig_c185_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 08 (original).](../assets/figures/ml_fig_c186_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 08 (original).](../assets/figures/ml_fig_c187_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 08 (original).](../assets/figures/ml_fig_c188_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 08 (original).](../assets/figures/ml_fig_c189_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 08 (original).](../assets/figures/ml_fig_c190_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 08 (original).](../assets/figures/ml_fig_c191_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 08 (original).](../assets/figures/ml_fig_c192_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 08 (original).](../assets/figures/ml_fig_c193_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 08 (original).](../assets/figures/ml_fig_c194_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 08 (original).](../assets/figures/ml_fig_c195_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 08 (original).](../assets/figures/ml_fig_c196_08.png)
*Figure — Nonnegative matrix parts. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 08 (original).](../assets/figures/ml_fig_c197_08.png)
*Figure — Dictionary learning atoms. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 08 (original).](../assets/figures/ml_fig_c198_08.png)
*Figure — Tensor CP decomposition. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 08 (original).](../assets/figures/ml_fig_c199_08.png)
*Figure — CCA shared views. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 08 (original).](../assets/figures/ml_fig_c200_08.png)
*Figure — ICA independence axes. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 08 (original).](../assets/figures/ml_fig_c201_08.png)
*Figure — Parallel analysis vs scree. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 08 (original).](../assets/figures/ml_fig_c202_08.png)
*Figure — Isomap geodesic vs chord. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 08 (original).](../assets/figures/ml_fig_c203_08.png)
*Figure — Sparse PCA cardinality path. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 08 (original).](../assets/figures/ml_fig_c204_08.png)
*Figure — Cattell scree elbow rule. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 08 (original).](../assets/figures/ml_fig_c205_08.png)
*Figure — MDS Kruskal stress curve. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 08 (original).](../assets/figures/ml_fig_c206_08.png)
*Figure — t-SNE early exaggeration. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 08 (original).](../assets/figures/ml_fig_c207_08.png)
*Figure — ICA unmixing independence axes. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 08 (original).](../assets/figures/ml_fig_c208_08.png)
*Figure — UMAP fuzzy simplicial sketch. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 08 (original).](../assets/figures/ml_fig_c209_08.png)
*Figure — Isomap geodesic vs Euclidean. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 08 (original).](../assets/figures/ml_fig_c210_08.png)
*Figure — LLE neighbor reconstruction weights. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 08 (original).](../assets/figures/ml_fig_c211_08.png)
*Figure — Kernel PCA ring manifold. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 08 (original).](../assets/figures/ml_fig_c212_08.png)
*Figure — Autoencoder bottleneck diagram. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 08 (original).](../assets/figures/ml_fig_c213_08.png)
*Figure — PCA biplot scores loadings. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 08 (original).](../assets/figures/ml_fig_c214_08.png)
*Figure — Factor analysis loading heat. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 08 (original).](../assets/figures/ml_fig_c215_08.png)
*Figure — Kernel ridge smooth fit. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 08 (original).](../assets/figures/ml_fig_c216_08.png)
*Figure — CCA canonical correlation decay. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 08 (original).](../assets/figures/ml_fig_c217_08.png)
*Figure — Johnson-Lindenstrauss projection k. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 08 (original).](../assets/figures/ml_fig_c218_08.png)
*Figure — NMF nonnegative basis heat. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 08 (original).](../assets/figures/ml_fig_c219_08.png)
*Figure — Procrustes shape alignment. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 08 (original).](../assets/figures/ml_fig_c220_08.png)
*Figure — Isomap k-neighbor stress. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 08 (original).](../assets/figures/ml_fig_c221_08.png)
*Figure — NMF non-negative parts factors. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 08 (original).](../assets/figures/ml_fig_c222_08.png)
*Figure — Canonical correlation view scores. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 08 (original).](../assets/figures/ml_fig_c223_08.png)
*Figure — Sparse PCA exact-zero loadings. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 08 (original).](../assets/figures/ml_fig_c224_08.png)
*Figure — ICA mix to independent sources. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 08 (original).](../assets/figures/ml_fig_c225_08.png)
*Figure — Kernel PCA radial feature cue. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 08 (original).](../assets/figures/ml_fig_c226_08.png)
*Figure — MDS stress vs dimension. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 08 (original).](../assets/figures/ml_fig_c227_08.png)
*Figure — UMAP fuzzy membership curves. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 08 (original).](../assets/figures/ml_fig_c228_08.png)
*Figure — CCA scree canonical corrs. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 08 (original).](../assets/figures/ml_fig_c229_08.png)
*Figure — Randomized SVD error vs rank. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 08 (original).](../assets/figures/ml_fig_c230_08.png)
*Figure — Factor analysis loading heat. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 08 (original).](../assets/figures/ml_fig_c231_08.png)
*Figure — Autoencoder bottleneck widths. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 08 (original).](../assets/figures/ml_fig_c232_08.png)
*Figure — Sparse coding dictionary atoms. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 08 (original).](../assets/figures/ml_fig_c233_08.png)
*Figure — t-SNE early exaggeration cool. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 08 (original).](../assets/figures/ml_fig_c234_08.png)
*Figure — NMF recon error vs rank. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 08 (original).](../assets/figures/ml_fig_c235_08.png)
*Figure — UMAP repulsion schedule. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 08 (original).](../assets/figures/ml_fig_c236_08.png)
*Figure — Dictionary size recon error. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 08 (original).](../assets/figures/ml_fig_c237_08.png)
*Figure — LargeVis negative sample cool. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 08 (original).](../assets/figures/ml_fig_c238_08.png)
*Figure — PCA energy vs components. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 08 (original).](../assets/figures/ml_fig_c239_08.png)
*Figure — PaCMAP neighbor graph cool. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 08 (original).](../assets/figures/ml_fig_c240_08.png)
*Figure — Randomized SVD error. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 08 (original).](../assets/figures/ml_fig_c241_08.png)
*Figure — t-SNE early exaggeration cool. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 08 (original).](../assets/figures/ml_fig_c242_08.png)
*Figure — Nyström rank approx error. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 08 (original).](../assets/figures/ml_fig_c243_08.png)
*Figure — TriMAP triplet cool path. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 08 (original).](../assets/figures/ml_fig_c244_08.png)
*Figure — Incremental PCA recon error. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 08 (original).](../assets/figures/ml_fig_c245_08.png)
*Figure — ForceAtlas2 layout cool. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 08 (original).](../assets/figures/ml_fig_c246_08.png)
*Figure — Kernel PCA recon error. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 08 (original).](../assets/figures/ml_fig_c247_08.png)
*Figure — LargeVis edge sample cool. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 08 (original).](../assets/figures/ml_fig_c248_08.png)
*Figure — Sparse PCA recon error. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 08 (original).](../assets/figures/ml_fig_c249_08.png)
*Figure — PaCMAP mid-near cool path. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 08 (original).](../assets/figures/ml_fig_c250_08.png)
*Figure — Randomized SVD error path. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 08 (original).](../assets/figures/ml_fig_c251_08.png)
*Figure — UMAP negative sample cool. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 08 (original).](../assets/figures/ml_fig_c252_08.png)
*Figure — Incremental PCA error. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 08 (original).](../assets/figures/ml_fig_c253_08.png)
*Figure — t-SNE late cool path. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 08 (original).](../assets/figures/ml_fig_c254_08.png)
*Figure — Nyström approx error. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 08 (original).](../assets/figures/ml_fig_c255_08.png)
*Figure — Force-directed cool path. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 08 (original).](../assets/figures/ml_fig_c256_08.png)
*Figure — Kernel PCA recon error. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 08 (original).](../assets/figures/ml_fig_c257_08.png)
*Figure — Autoencoder recon path c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 08 (original).](../assets/figures/ml_fig_c258_08.png)
*Figure — Sparse coding residual c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 08 (original).](../assets/figures/ml_fig_c259_08.png)
*Figure — Dictionary size error c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 08 (original).](../assets/figures/ml_fig_c260_08.png)
*Figure — Kernel PCA error c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 08 (original).](../assets/figures/ml_fig_c261_08.png)
*Figure — Incremental PCA path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 08 (original).](../assets/figures/ml_fig_c262_08.png)
*Figure — Truncated SVD path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 08 (original).](../assets/figures/ml_fig_c263_08.png)
*Figure — Manifold trust path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 08 (original).](../assets/figures/ml_fig_c264_08.png)
*Figure — PCA variance path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 08 (original).](../assets/figures/ml_fig_c265_08.png)
*Figure — Randomized SVD error c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 08 (original).](../assets/figures/ml_fig_c266_08.png)
*Figure — NMF recon residual c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 08 (original).](../assets/figures/ml_fig_c267_08.png)
*Figure — ICA independence path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 08 (original).](../assets/figures/ml_fig_c268_08.png)
*Figure — t-SNE cool path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 08 (original).](../assets/figures/ml_fig_c269_08.png)
*Figure — UMAP neighbor path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 08 (original).](../assets/figures/ml_fig_c270_08.png)
*Figure — PaCMAP mid-near path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 08 (original).](../assets/figures/ml_fig_c271_08.png)
*Figure — Isomap geodesic path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 08 (original).](../assets/figures/ml_fig_c272_08.png)
*Figure — LLE local residual c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 08 (original).](../assets/figures/ml_fig_c273_08.png)
*Figure — Autoencoder recon path c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 08 (original).](../assets/figures/ml_fig_c274_08.png)
*Figure — Sparse coding residual c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 08 (original).](../assets/figures/ml_fig_c275_08.png)
*Figure — Dictionary size error c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 08 (original).](../assets/figures/ml_fig_c276_08.png)
*Figure — Kernel PCA error c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 08 (original).](../assets/figures/ml_fig_c277_08.png)
*Figure — Incremental PCA path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 08 (original).](../assets/figures/ml_fig_c278_08.png)
*Figure — Truncated SVD path c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 08 (original).](../assets/figures/ml_fig_c279_08.png)
*Figure — Manifold trust path c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 08 (original).](../assets/figures/ml_fig_c280_08.png)
*Figure — PCA variance path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 08 (original).](../assets/figures/ml_fig_c281_08.png)
*Figure — Randomized SVD error c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 08 (original).](../assets/figures/ml_fig_c282_08.png)
*Figure — NMF recon residual c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 08 (original).](../assets/figures/ml_fig_c283_08.png)
*Figure — ICA independence path c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 08 (original).](../assets/figures/ml_fig_c284_08.png)
*Figure — t-SNE cool path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 08 (original).](../assets/figures/ml_fig_c285_08.png)
*Figure — UMAP neighbor path c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 08 (original).](../assets/figures/ml_fig_c286_08.png)
*Figure — PaCMAP mid-near path c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 08 (original).](../assets/figures/ml_fig_c287_08.png)
*Figure — Isomap geodesic path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 08 (original).](../assets/figures/ml_fig_c288_08.png)
*Figure — LLE local residual c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 08 (original).](../assets/figures/ml_fig_c289_08.png)
*Figure — Autoencoder recon path c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 08 (original).](../assets/figures/ml_fig_c290_08.png)
*Figure — Sparse coding residual c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 08 (original).](../assets/figures/ml_fig_c291_08.png)
*Figure — Dictionary size error c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 08 (original).](../assets/figures/ml_fig_c292_08.png)
*Figure — Kernel PCA error c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 08 (original).](../assets/figures/ml_fig_c293_08.png)
*Figure — Incremental PCA path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 08 (original).](../assets/figures/ml_fig_c294_08.png)
*Figure — Truncated SVD path c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 08 (original).](../assets/figures/ml_fig_c295_08.png)
*Figure — Manifold trust path c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 08 (original).](../assets/figures/ml_fig_c296_08.png)
*Figure — PCA variance path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 08 (original).](../assets/figures/ml_fig_c297_08.png)
*Figure — Randomized SVD error c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 08 (original).](../assets/figures/ml_fig_c298_08.png)
*Figure — NMF recon residual c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 08 (original).](../assets/figures/ml_fig_c299_08.png)
*Figure — ICA independence path c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 08 (original).](../assets/figures/ml_fig_c300_08.png)
*Figure — t-SNE cool path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 08 (original).](../assets/figures/ml_fig_c301_08.png)
*Figure — UMAP neighbor path c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 08 (original).](../assets/figures/ml_fig_c302_08.png)
*Figure — PaCMAP mid-near path c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 08 (original).](../assets/figures/ml_fig_c303_08.png)
*Figure — Isomap geodesic path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 08 (original).](../assets/figures/ml_fig_c304_08.png)
*Figure — LLE local residual c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 08 (original).](../assets/figures/ml_fig_c305_08.png)
*Figure — Autoencoder recon path c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 08 (original).](../assets/figures/ml_fig_c306_08.png)
*Figure — Sparse coding residual c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 08 (original).](../assets/figures/ml_fig_c307_08.png)
*Figure — Dictionary size error c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 08 (original).](../assets/figures/ml_fig_c308_08.png)
*Figure — Kernel PCA error c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 08 (original).](../assets/figures/ml_fig_c309_08.png)
*Figure — Incremental PCA path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 08 (original).](../assets/figures/ml_fig_c310_08.png)
*Figure — Truncated SVD path c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 08 (original).](../assets/figures/ml_fig_c311_08.png)
*Figure — Manifold trust path c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 08 (original).](../assets/figures/ml_fig_c312_08.png)
*Figure — PCA variance path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 08 (original).](../assets/figures/ml_fig_c313_08.png)
*Figure — Randomized SVD error c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 08 (original).](../assets/figures/ml_fig_c314_08.png)
*Figure — NMF recon residual c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 08 (original).](../assets/figures/ml_fig_c315_08.png)
*Figure — ICA independence path c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 08 (original).](../assets/figures/ml_fig_c316_08.png)
*Figure — t-SNE cool path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 08 (original).](../assets/figures/ml_fig_c317_08.png)
*Figure — UMAP neighbor path c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 08 (original).](../assets/figures/ml_fig_c318_08.png)
*Figure — PaCMAP mid-near path c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 08 (original).](../assets/figures/ml_fig_c319_08.png)
*Figure — Isomap geodesic path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 08 (original).](../assets/figures/ml_fig_c320_08.png)
*Figure — LLE local residual c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 08 (original).](../assets/figures/ml_fig_c321_08.png)
*Figure — Autoencoder recon path c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 08 (original).](../assets/figures/ml_fig_c322_08.png)
*Figure — Sparse coding residual c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 08 (original).](../assets/figures/ml_fig_c323_08.png)
*Figure — Dictionary size error c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 08 (original).](../assets/figures/ml_fig_c324_08.png)
*Figure — Kernel PCA error c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 08 (original).](../assets/figures/ml_fig_c325_08.png)
*Figure — Incremental PCA path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 08 (original).](../assets/figures/ml_fig_c326_08.png)
*Figure — Truncated SVD path c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 08 (original).](../assets/figures/ml_fig_c327_08.png)
*Figure — Manifold trust path c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 08 (original).](../assets/figures/ml_fig_c328_08.png)
*Figure — PCA variance path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 08 (original).](../assets/figures/ml_fig_c329_08.png)
*Figure — Randomized SVD error c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 08 (original).](../assets/figures/ml_fig_c330_08.png)
*Figure — NMF recon residual c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 08 (original).](../assets/figures/ml_fig_c331_08.png)
*Figure — ICA independence path c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 08 (original).](../assets/figures/ml_fig_c332_08.png)
*Figure — t-SNE cool path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 08 (original).](../assets/figures/ml_fig_c333_08.png)
*Figure — UMAP neighbor path c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 08 (original).](../assets/figures/ml_fig_c334_08.png)
*Figure — PaCMAP mid-near path c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 08 (original).](../assets/figures/ml_fig_c335_08.png)
*Figure — Isomap geodesic path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 08 (original).](../assets/figures/ml_fig_c336_08.png)
*Figure — LLE local residual c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 08 (original).](../assets/figures/ml_fig_c337_08.png)
*Figure — Autoencoder recon path c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 08 (original).](../assets/figures/ml_fig_c338_08.png)
*Figure — Sparse coding residual c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 08 (original).](../assets/figures/ml_fig_c339_08.png)
*Figure — Dictionary size error c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 08 (original).](../assets/figures/ml_fig_c340_08.png)
*Figure — Kernel PCA error c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 08 (original).](../assets/figures/ml_fig_c341_08.png)
*Figure — Incremental PCA path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 08 (original).](../assets/figures/ml_fig_c342_08.png)
*Figure — Truncated SVD path c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 08 (original).](../assets/figures/ml_fig_c343_08.png)
*Figure — Manifold trust path c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 08 (original).](../assets/figures/ml_fig_c344_08.png)
*Figure — PCA variance path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 08 (original).](../assets/figures/ml_fig_c345_08.png)
*Figure — Randomized SVD error c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 08 (original).](../assets/figures/ml_fig_c346_08.png)
*Figure — NMF recon residual c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 08 (original).](../assets/figures/ml_fig_c347_08.png)
*Figure — ICA independence path c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 08 (original).](../assets/figures/ml_fig_c348_08.png)
*Figure — t-SNE cool path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 08 (original).](../assets/figures/ml_fig_c349_08.png)
*Figure — UMAP neighbor path c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 08 (original).](../assets/figures/ml_fig_c350_08.png)
*Figure — PaCMAP mid-near path c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 08 (original).](../assets/figures/ml_fig_c351_08.png)
*Figure — Isomap geodesic path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 08 (original).](../assets/figures/ml_fig_c352_08.png)
*Figure — LLE local residual c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 08 (original).](../assets/figures/ml_fig_c353_08.png)
*Figure — Autoencoder recon path c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 08 (original).](../assets/figures/ml_fig_c354_08.png)
*Figure — Sparse coding residual c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 08 (original).](../assets/figures/ml_fig_c355_08.png)
*Figure — Dictionary size error c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 08 (original).](../assets/figures/ml_fig_c356_08.png)
*Figure — Kernel PCA error c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 08 (original).](../assets/figures/ml_fig_c357_08.png)
*Figure — Incremental PCA path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 08 (original).](../assets/figures/ml_fig_c358_08.png)
*Figure — Truncated SVD path c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 08 (original).](../assets/figures/ml_fig_c359_08.png)
*Figure — Manifold trust path c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 08 (original).](../assets/figures/ml_fig_c360_08.png)
*Figure — PCA variance path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 08 (original).](../assets/figures/ml_fig_c361_08.png)
*Figure — Randomized SVD error c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 08 (original).](../assets/figures/ml_fig_c362_08.png)
*Figure — NMF recon residual c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 08 (original).](../assets/figures/ml_fig_c363_08.png)
*Figure — ICA independence path c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 08 (original).](../assets/figures/ml_fig_c364_08.png)
*Figure — t-SNE cool path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 08 (original).](../assets/figures/ml_fig_c365_08.png)
*Figure — UMAP neighbor path c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 08 (original).](../assets/figures/ml_fig_c366_08.png)
*Figure — PaCMAP mid-near path c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 08 (original).](../assets/figures/ml_fig_c367_08.png)
*Figure — Isomap geodesic path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 08 (original).](../assets/figures/ml_fig_c368_08.png)
*Figure — LLE local residual c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 08 (original).](../assets/figures/ml_fig_c369_08.png)
*Figure — Autoencoder recon path c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 08 (original).](../assets/figures/ml_fig_c370_08.png)
*Figure — Sparse coding residual c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 08 (original).](../assets/figures/ml_fig_c371_08.png)
*Figure — Dictionary size error c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 08 (original).](../assets/figures/ml_fig_c372_08.png)
*Figure — Kernel PCA error c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 08 (original).](../assets/figures/ml_fig_c373_08.png)
*Figure — Incremental PCA path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 08 (original).](../assets/figures/ml_fig_c374_08.png)
*Figure — Truncated SVD path c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 08 (original).](../assets/figures/ml_fig_c375_08.png)
*Figure — Manifold trust path c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 08 (original).](../assets/figures/ml_fig_c376_08.png)
*Figure — PCA variance path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 08 (original).](../assets/figures/ml_fig_c377_08.png)
*Figure — Randomized SVD error c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 08 (original).](../assets/figures/ml_fig_c378_08.png)
*Figure — NMF recon residual c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 08 (original).](../assets/figures/ml_fig_c379_08.png)
*Figure — ICA independence path c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 08 (original).](../assets/figures/ml_fig_c380_08.png)
*Figure — t-SNE cool path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 08 (original).](../assets/figures/ml_fig_c381_08.png)
*Figure — UMAP neighbor path c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 08 (original).](../assets/figures/ml_fig_c382_08.png)
*Figure — PaCMAP mid-near path c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 08 (original).](../assets/figures/ml_fig_c383_08.png)
*Figure — Isomap geodesic path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 08 (original).](../assets/figures/ml_fig_c384_08.png)
*Figure — LLE local residual c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 08 (original).](../assets/figures/ml_fig_c385_08.png)
*Figure — Autoencoder recon path c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 08 (original).](../assets/figures/ml_fig_c386_08.png)
*Figure — Sparse coding residual c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 08 (original).](../assets/figures/ml_fig_c387_08.png)
*Figure — Dictionary size error c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 08 (original).](../assets/figures/ml_fig_c388_08.png)
*Figure — Kernel PCA error c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 08 (original).](../assets/figures/ml_fig_c389_08.png)
*Figure — Incremental PCA path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 08 (original).](../assets/figures/ml_fig_c390_08.png)
*Figure — Truncated SVD path c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 08 (original).](../assets/figures/ml_fig_c391_08.png)
*Figure — Manifold trust path c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 08 (original).](../assets/figures/ml_fig_c392_08.png)
*Figure — PCA variance path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 08 (original).](../assets/figures/ml_fig_c393_08.png)
*Figure — Randomized SVD error c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 08 (original).](../assets/figures/ml_fig_c394_08.png)
*Figure — NMF recon residual c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 08 (original).](../assets/figures/ml_fig_c395_08.png)
*Figure — ICA independence path c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 08 (original).](../assets/figures/ml_fig_c396_08.png)
*Figure — t-SNE cool path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 08 (original).](../assets/figures/ml_fig_c397_08.png)
*Figure — UMAP neighbor path c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 08 (original).](../assets/figures/ml_fig_c398_08.png)
*Figure — PaCMAP mid-near path c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 08 (original).](../assets/figures/ml_fig_c399_08.png)
*Figure — Isomap geodesic path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 08 (original).](../assets/figures/ml_fig_c400_08.png)
*Figure — LLE local residual c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 08 (original).](../assets/figures/ml_fig_c401_08.png)
*Figure — Autoencoder recon path c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 08 (original).](../assets/figures/ml_fig_c402_08.png)
*Figure — Sparse coding residual c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 08 (original).](../assets/figures/ml_fig_c403_08.png)
*Figure — Dictionary size error c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 08 (original).](../assets/figures/ml_fig_c404_08.png)
*Figure — Kernel PCA error c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 08 (original).](../assets/figures/ml_fig_c405_08.png)
*Figure — Incremental PCA path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 08 (original).](../assets/figures/ml_fig_c406_08.png)
*Figure — Truncated SVD path c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 08 (original).](../assets/figures/ml_fig_c407_08.png)
*Figure — Manifold trust path c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 08 (original).](../assets/figures/ml_fig_c408_08.png)
*Figure — PCA variance path c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 08 (original).](../assets/figures/ml_fig_c409_08.png)
*Figure — Randomized SVD error c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 08 (original).](../assets/figures/ml_fig_c410_08.png)
*Figure — NMF recon residual c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 08 (original).](../assets/figures/ml_fig_c411_08.png)
*Figure — ICA independence path c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 08 (original).](../assets/figures/ml_fig_c412_08.png)
*Figure — t-SNE cool path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 08 (original).](../assets/figures/ml_fig_c413_08.png)
*Figure — UMAP neighbor path c413. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

High-dimensional clinical data suffer geometric and statistical curses that motivate reduction and decomposition. PCA finds orthogonal maximum-variance directions; a full numerical example with three points shows centering, covariance eigen-decomposition, scores, and explained variance. SVD implements PCA stably; incremental PCA handles streaming or out-of-core data. LDA/Fisher seeks supervised discriminative projections. Nonlinear methods—LLE, t-SNE, UMAP—preserve local manifold structure for visualization with important interpretive caveats. Fourier and wavelet transforms decompose signals in frequency and time–frequency; aggregation methods compress long series. Matrix tools include Cholesky for SPD linear algebra, NMF for nonnegative parts-based factors, and SVD for optimal low-rank approximation. Topic models apply these ideas to text: LSI via truncated SVD and LDA as a Bayesian generative mixture of topics. Tensors extend factorizations multi-way via mode-n products with CP, Tucker, and tensor-train forms. Clinically, reduce when it denoises and clarifies; refuse reduction when it erases rare signals, harms calibration, or replaces needed interpretable covariates.

## Practice and Reflection

(1) Recompute the chapter’s 3-point PCA using divisor n−1 in the covariance. How do eigenvalues scale, and do principal directions change?

(2) Prove that for centered X, right singular vectors of X equal eigenvectors of XᵀX. Relate singular values to eigenvalues.

(3) For two-class Fisher discriminant in 2D with isotropic within-class variance, sketch why the optimal direction aligns with the mean difference.

(4) List three reasons a beautiful t-SNE plot of multi-center stroke radiomics might not imply transportable phenotypes.

(5) Compute by hand the 2-point DFT of the series [1, −1] and interpret the spectrum.

(6) Explain one clinical scenario favoring wavelets over a single global FFT window for EEG feature extraction.

(7) Contrast NMF and SVD on a nonnegative term–document matrix: reconstruction optimality versus interpretability of parts.

(8) Describe how LSI would rank documents for a query in the reduced space, and how LDA’s document–topic vector differs conceptually.

(9) Define mode-1 unfolding of a 2×3×2 tensor in words and state the shape of the resulting matrix.

(10) You have n = 120 ICH patients and p = 5000 gene-expression features. Propose a reduction pipeline for visualization and for supervised prediction, with leakage controls.
