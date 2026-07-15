# Chapter 4. Clustering


![04 Kmeans](../assets/figures/04_kmeans.png)


## Opening

A research coordinator proposes clustering ‘phenotypes’ of cryptogenic stroke from EHR labs and imaging codes. Clustering can discover structure—or invent comforting noise. This chapter teaches how to ask which one you just saw.


![Clustering sketch with centroids (synthetic data; original).](../assets/figures/ml_fig_clustering.png)

*Clustering sketch with centroids (synthetic data; original).*
## Learning Objectives

Define clustering as unsupervised partitioning and select similarity measures matched to feature type.

Compute Euclidean, Manhattan, Mahalanobis, Hamming, edit-based, cosine, Jaccard, DTW, and basic graph similarities.

Execute k-means, and contrast k-median and k-medoid objectives and robustness properties.

Explain DBSCAN and OPTICS density clustering via core points, reachability, and noise.

Describe hierarchical agglomerative single-linkage (SLINK) and divisive DIANA; outline BIRCH and CURE for large data.

Fit conceptual Gaussian mixture models and Fuzzy C-means as soft clustering alternatives.

Evaluate partitions with elbow/WSS, silhouette, Dunn, Davies–Bouldin, purity, and Rand index.

Apply clustering cautiously to stroke phenotype discovery without reifying clusters as causal etiologies.

## What Clustering Is and Is Not

Clustering algorithms group unlabeled observations so that items in the same group are more similar to each other than to items in other groups, under a stated similarity geometry. Formally, given points x_1, …, x_n in a feature space, a hard clustering is an assignment c: {1,…,n} → {1,…,k} (or a partition into nonempty sets C_1, …, C_k). Soft clustering returns responsibilities r_{ij} = P(cluster j | x_i), typically with Σ_j r_{ij} = 1.

Clustering is unsupervised: there is no ground-truth label inside the optimization unless you bring one in for external evaluation. That fact is easy to forget when a heatmap of three ‘stroke phenotypes’ looks medically narratable. Narratability is not validity. The algorithm optimizes a mathematical objective—compactness, density separation, mixture likelihood—not truth about TOAST etiology and not treatment-effect heterogeneity unless a separate design supports those claims.

This chapter develops similarity measures, partitioning methods (k-means/median/medoid), density-based DBSCAN and OPTICS, hierarchical SLINK and DIANA, large-scale BIRCH and CURE, probabilistic and fuzzy clustering, evaluation indices, a fully worked k-means example, and clinical–epidemiologic cautions.

## Similarity and Dissimilarity Measures

Most clustering methods need a distance d(x,y) or a similarity s(x,y). Distance choice is a modeling decision: it defines which patients are ‘near.’ Always standardize or otherwise scale continuous features before Euclidean geometry unless units are already commensurate. State the measure explicitly—results can flip under L1 versus L2, or under Jaccard versus Hamming for binary comorbidity vectors.

![4.1: Unit balls and iso-distance contours around the origin for the Euclidean (L2), Manhattan (L1), and Mahalanobis metrics (](../assets/figures/ml_concept_4.1_6c43a3ff.png)

*Figure 4.1 — original teaching graphic.*

### Euclidean and Manhattan distances

Euclidean (L2) distance d_2(x,y) = √(Σ_m (x_m − y_m)²) is the straight-line distance in R^p and the geometry assumed by classical k-means with means as centers. Manhattan (L1) distance d_1(x,y) = Σ_m |x_m − y_m| is more robust to single-coordinate spikes and corresponds to median-based geometry in one dimension. Both require comparable feature scales. Worked contrast on x = (1,1) and y = (6,5): d_2 = √((1−6)² + (1−5)²) = √(25 + 16) = √41 ≈ 6.40, whereas d_1 = |1−6| + |1−5| = 5 + 4 = 9. In general ‖v‖_1 ≥ ‖v‖_2, with equality only when the difference lies on a single axis; the L1/L2 gap widens as a difference spreads across many coordinates.

### Mahalanobis distance

Mahalanobis distance d_M(x,y) = √((x−y)^T Σ^{−1} (x−y)) accounts for covariance structure: directions of high variance and correlated features are down-weighted appropriately. When Σ = I, it reduces to Euclidean. Use when elliptical geometry is plausible and Σ can be estimated stably (or pooled within clusters). Unstable Σ in high dimensions requires shrinkage or diagonal approximations.

Worked example. Let Σ = [[1, 0.5], [0.5, 1]] (two positively correlated features), so det Σ = 1 − 0.25 = 0.75 and Σ^{−1} = (1/0.75)·[[1, −0.5], [−0.5, 1]] = [[4/3, −2/3], [−2/3, 4/3]]. Take two difference vectors of identical Euclidean length √2. With-grain Δ = (1, 1): Σ^{−1}Δ = (2/3, 2/3), so d_M² = Δ^T Σ^{−1} Δ = 2/3 + 2/3 = 4/3 and d_M = √(4/3) ≈ 1.15. Against-grain Δ = (1, −1): Σ^{−1}Δ = (2, −2), so d_M² = 2 + 2 = 4 and d_M = 2.0. The two pairs are the same straight-line distance apart (√2 ≈ 1.41), yet Mahalanobis rates the with-correlation pair closer (1.15) and the against-correlation pair farther (2.0), because moving against the covariance grain is statistically more surprising.

### Hamming, Levenshtein, and LCS distances

Hamming distance d_H(x,y) = Σ_m 1[x_m ≠ y_m] counts positions at which two equal-length strings or binary vectors differ—natural for fixed-length genotype masks or one-hot clinical indicator strings of common length. Levenshtein (edit) distance is the minimum number of insertions, deletions, and substitutions to transform one string into another—useful for noisy identifiers, sequences of care codes, and short text tokens. Longest common subsequence (LCS) related distances emphasize shared order without requiring contiguity; they appear in sequence comparison when gaps are semantically allowed.

### Cosine and Jaccard

Cosine similarity is (x·y)/(||x|| ||y||); cosine distance is often 1 − cosine similarity. It depends on orientation rather than magnitude—standard for bag-of-words, TF–IDF note vectors, and high-dimensional sparse embeddings. Jaccard similarity for sets is |A ∩ B|/|A ∪ B|; Jaccard distance is 1 − Jaccard. Use Jaccard for unordered sets of comorbidities, medications, or mutated gene sets without double-counting absences the way Hamming on sparse binaries might.

### Dynamic time warping (DTW)

DTW aligns two sequences by nonlinear time warping that minimizes cumulative pairwise costs, allowing similar shapes that are out of phase—vital-sign trajectories, serial NIHSS curves, gait signals. Classical DTW is O(T_1 T_2) in series lengths; constraints and approximations improve scale. DTW is a dissimilarity, not always a metric in the strict sense, but it is often more meaningful than pointwise Euclidean on raw time indices when sampling rates or event timing differ.

The alignment is computed by dynamic programming on an accumulated-cost matrix D, adding each local cost (here absolute difference) to the cheapest predecessor: D(i,j) = |a_i − b_j| + min{ D(i−1,j), D(i,j−1), D(i−1,j−1) }, seeded at D(1,1) = |a_1 − b_1|. Worked example on a = [1, 2, 3] and a faster b = [1, 3]. Local costs |a_i − b_j| are (1,1)=0, (1,2)=2, (2,1)=1, (2,2)=1, (3,1)=2, (3,2)=0. Accumulating: D(1,1)=0; D(2,1)=1+0=1; D(3,1)=2+1=3; D(1,2)=2+0=2; D(2,2)=1+min(2,1,0)=1; D(3,2)=0+min(1,3,1)=1. The DTW cost is D(3,2) = 1, achieved by the warping path (1,1)→(2,2)→(3,2), which stretches b’s value 3 to match both a_2 and a_3. Pointwise Euclidean cannot even be evaluated here without first forcing the unequal-length series onto a shared clock—the very distortion DTW is built to avoid.

### Graph similarities

When objects are graphs (brain connectivity, molecular graphs) or nodes in a network, similarity may use graph edit distance, shortest-path distances, diffusion kernels, or learned graph embeddings. Node similarities based on shared neighbors (for example, Jaccard on adjacency lists) support community detection: if node u has neighbor set {a,b,c} and node v has {b,c,d}, their shared-neighbor Jaccard similarity is |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5. Choose graph similarities that match the scientific claim: structural role equivalence differs from community co-membership.

Scale continuous features before Euclidean/Manhattan k-means-style clustering.

Use cosine/Jaccard for sparse high-dimensional or set-like clinical codes.

Use DTW for misaligned temporal trajectories; edit distances for sequences/strings.

Mahalanobis needs trustworthy covariance; shrink in high-p settings.

## Partitioning Methods: k-Means, k-Median, k-Medoid

Partitioning methods seek k groups that optimize a within-cluster dispersion criterion. They typically alternate assignment of points to representatives and update of representatives.

### k-means

K-means minimizes within-cluster sum of squares J = Σ_i Σ_j z_{ij} ||x_i − μ_j||² with hard assignments z_{ij} ∈ {0,1}. Lloyd’s algorithm: (1) initialize centers; (2) assign each point to the nearest center; (3) replace each center by the mean of its assigned points; (4) repeat until stable. J decreases monotonically to a local minimum; multiple restarts or k-means++ seeding are mandatory. Assumptions: roughly spherical, similar-sized clusters in Euclidean space; sensitivity to outliers because of squares.

```
Algorithm: k-means (Lloyd's algorithm)
Input: data X = {x_1, ..., x_n}, cluster count k
Output: centers μ_1, ..., μ_k and hard assignments z

1 initialize μ_1, ..., μ_k # prefer k-means++ over uniform-random seeds
2 repeat
3 # assignment step
4 for each point x_i:
5 z_i ← argmin_j ‖x_i − μ_j‖² # nearest center by squared L2
6 # update step
7 for each cluster j:
8 μ_j ← (1/|C_j|) · Σ_{i: z_i = j} x_i # mean of assigned points
9 until z stops changing (or a max-iteration cap is reached)
10 return μ, z
```

### k-median and k-medoid

K-median replaces squared Euclidean loss with L1 (or general distance) and updates centers toward coordinate-wise medians in R^p for L1—more robust to extreme coordinate values. K-medoid requires centers to be actual data points (medoids), minimizing total distance to medoids; PAM (partitioning around medoids) is a classic algorithm. Medoids are interpretable exemplars (‘this real patient is the prototype’) and work with arbitrary distance matrices, including DTW or Jaccard—unlike means, which need vector averaging.

## Full Worked 2-D k-Means Example

Cluster six points in R² with k = 2. Think of axis 1 as a standardized age-like coordinate and axis 2 as a standardized severity-like coordinate; numbers are chosen for clean arithmetic.

![4.2: Lloyd's algorithm for k-means (k = 2) on the six-point toy set: (a) initialize centroids at μ1 = (1, 1) and μ2 = (7, 6);](../assets/figures/ml_concept_4.2_d79fde27.png)

*Figure 4.2 — original teaching graphic.*

# Points and initialization
A,B,C = (1,1),(1,2),(2,1)
D,E,F = (6,5),(7,6),(6,7)
mu1, mu2 = (1,1), (7,6)
# Iteration 1 assigns {A,B,C}->{1}, {D,E,F}->{2}
# Updated means: mu1=(4/3,4/3), mu2=(19/3,6)
# Iteration 2: assignments unchanged -> converge, J=4

Iteration 1 — assignment with μ1=(1,1), μ2=(7,6). Squared distances: A to μ1 is 0, to μ2 is 61 → cluster 1. B: 1 vs 52 → 1. C: 1 vs 50 → 1. D: 41 vs 2 → 2. E: 61 vs 0 → 2. F: 61 vs 2 → 2. Clusters: C1={A,B,C}, C2={D,E,F}.

Iteration 1 — update: μ1 = mean(A,B,C) = (4/3, 4/3); μ2 = mean(D,E,F) = (19/3, 6). Iteration 2 recomputes assignments with these centers; all points remain in the same clusters. Means unchanged; stop. Within-cluster sum of squares: C1 contributes 4/3; C2 contributes 8/3; total J = 4.

Interpretation: the algorithm recovered the two visual clumps. Poor initialization can yield worse local minima—hence restarts. Forcing k=3 would reduce training WCSS without proving three clinical subtypes exist. Silhouette for point A: a(A)=1 (mean distance to B and C), b(A)≈7.341 (mean distance to C2), s(A)≈0.864—well clustered under this partition.

## Density-Based Methods: DBSCAN and OPTICS

![4.3: Two interleaving moons (generated with numpy) clustered two ways. (a) DBSCAN (ε = 0.20, minPts = 6) chains through dense](../assets/figures/ml_concept_4.3_a8c7db03.png)

*Figure 4.3 — original teaching graphic.*

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) defines clusters as dense regions separated by sparse regions. Parameters: radius ε > 0 and minPts. The ε-neighborhood of x is N_ε(x) = {y : d(x,y) ≤ ε}. Point x is a core point if |N_ε(x)| ≥ minPts. Border points are non-core points within ε of some core; noise points are neither. Clusters are maximal density-connected sets of points chained through core neighborhoods.

![DBSCAN core / border / noise and k-distance ε selection (synthetic; original).](../assets/figures/ml_fig_dbscan_density.png)

*Figure — DBSCAN density sketch. **Left:** core points (teal) have ≥ minPts neighbors inside ε; border points (gold) sit in a core neighborhood but are not dense enough themselves; noise (gray ×) is neither. Dashed circles show ε around example cores. **Right:** a k-distance plot (k = minPts − 1) sorted descending guides ε near the “knee.” Global ε fails when density varies; scale features before computing distances.*

```
Algorithm: DBSCAN
Input: data X, radius ε, minPts
Output: cluster labels, with some points marked NOISE

1 mark every point UNVISITED; C ← 0 # C counts clusters
2 for each point p in X:
3 if p is VISITED: continue
4 mark p VISITED
5 Nbr ← {q in X : d(p,q) ≤ ε} # ε-neighborhood (includes p)
6 if |Nbr| < minPts:
7 label p NOISE # may later be reclaimed as a border point
8 else:
9 C ← C + 1; add p to cluster C
10 Seeds ← Nbr \ {p}
11 while Seeds not empty:
12 pull q from Seeds
13 if q is labelled NOISE: add q to cluster C # border point
14 if q is VISITED: continue
15 mark q VISITED; add q to cluster C
16 qNbr ← {r in X : d(q,r) ≤ ε}
17 if |qNbr| ≥ minPts: # q is itself a core point
18 Seeds ← Seeds ∪ qNbr
19 return labels
```

DBSCAN does not require k a priori, finds non-spherical shapes, and labels noise—attractive when some cases are truly atypical. It is sensitive to ε, minPts, and scaling; a global ε struggles with varying density. High dimensions concentrate distances and weaken density thresholds. Tune ε with k-distance plots; treat defaults as provisional.

### OPTICS

OPTICS (Ordering Points To Identify the Clustering Structure) extends density clustering by producing an ordering of points with reachability distances that reveal cluster structure at multiple density levels. Rather than a single ε cut, the reachability plot exposes valleys (dense clusters) and peaks (separators). Extracting clusters from the ordering is a second step. OPTICS is valuable when density varies across phenotype clouds—common in mixed mild/severe clinical spaces—at higher conceptual and computational cost than basic DBSCAN.

## Hierarchical Methods: SLINK and DIANA

Agglomerative hierarchical clustering begins with each point alone and merges closest clusters until one remains; the history is a dendrogram. Divisive methods start with one cluster and recursively split.

![4.4: Agglomerative hierarchical clustering (average linkage, Euclidean) of the six toy points, drawn as a dendrogram. The com](../assets/figures/ml_concept_4.4_0c668308.png)

*Figure 4.4 — original teaching graphic.*

### Single-linkage (SLINK)

Single linkage defines intercluster distance as the minimum pairwise distance between points in the two clusters. SLINK is an efficient algorithm for single-linkage agglomerative clustering. Single linkage can recover elongated, non-elliptical chains—and can undesirably bridge distinct dense groups via thin noise chains (the chaining effect). Complete linkage uses maximum pairwise distance (preferring compact clusters); average linkage averages pairwise distances; Ward linkage merges pairs that least increase total within-cluster variance.

![Dendrogram cut height chooses k — single vs complete on P1=0…P4=11 (original).](../assets/figures/ml_fig_dendrogram_cut.png)

*Figure — Hierarchical cut. Chapter walk-through points on a line: P1=0, P2=1, P3=10, P4=11. Both linkages merge (P1,P2) and (P3,P4) at height 1; the **final** merge sits at 9 (single = min cross-distance) vs 11 (complete = max). A horizontal cut (here h=5) yields k=2. Cut height is a model-selection choice analogous to k-means k—not a discovery of natural clinical taxa.*

### DIANA

DIANA (Divisive Analysis) is a divisive hierarchical method: at each step it splits the cluster that is most heterogeneous, typically by selecting a seed point farthest from the cluster mean (or medoid) and reassigning points to form two subclusters that better separate diameter or average dissimilarity. Divisive methods can be more globally aware early on but are computationally heavier if implemented naively. Dendrogram cuts still require principled choice of height or k; bootstrap stability of branches should temper clinical naming of intermediate nodes.

## Large-Scale Hierarchical Clustering: BIRCH and CURE

### BIRCH

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) builds a CF-tree (clustering feature tree) that summarizes dense regions with sufficient statistics (count, linear sum, squared sum) under memory constraints. It scans large n efficiently, then clusters the compressed summaries. BIRCH assumes metric spaces friendly to spherical summaries and can struggle with non-spherical clusters and noise if parameters (branching, threshold) are poorly set. Use when n is large, d is moderate, and a single pass / limited memory profile matters—for example, multi-year registry extracts.

### CURE

CURE (Clustering Using Representatives) represents each cluster by a set of well-scattered representative points shrunk toward the centroid, enabling non-spherical shapes and relative outlier robustness compared with single-centroid methods. Hierarchical merging uses distances between representative sets. Sampling and partitioning strategies help CURE scale. It is a strong conceptual alternative when clusters are elongated or uneven but still compact enough for representative-point geometry.

## Probabilistic and Fuzzy Clustering: GMM and Fuzzy C-Means

### Gaussian mixture models

A Gaussian mixture model posits p(x) = Σ_{j=1}^k π_j N(x | μ_j, Σ_j) with mixing weights π_j. Soft responsibilities are r_{ij} ∝ π_j N(x_i | μ_j, Σ_j). EM alternates E-step responsibilities and M-step weighted updates of π_j, μ_j, Σ_j. Hard k-means resembles a spherical, hard-assignment limit. GMMs express boundary uncertainty (r ≈ 0.55/0.45) but still assume elliptical Gaussian components and can overfit full covariances when n is small relative to p.

### Fuzzy C-means

Fuzzy C-means (FCM) assigns each point a membership grade u_{ij} ∈ [0,1] with Σ_j u_{ij} = 1, minimizing a weighted within-cluster sum of distances with fuzzifier m > 1 (commonly m=2): J = Σ_i Σ_j u_{ij}^m ||x_i − c_j||². Updates alternate between memberships and centroids. Unlike GMM, classical FCM is not a full generative probabilistic model, but it similarly softens hard boundaries. Memberships are not posterior probabilities unless given additional calibration semantics; do not over-read u_{ij}=0.7 as ‘70% chance of disease subtype j.’

## Clustering Result Evaluation

### Intrinsic methods: elbow/WSS, silhouette, Dunn, Davies–Bouldin

Within-cluster sum of squares (WSS/WCSS) decreases as k grows; the elbow method looks for a diminishing-returns bend in WSS(k)—often subjective. Silhouette s(i) = (b(i) − a(i))/max{a(i),b(i)} with a(i) mean intra-cluster distance and b(i) nearest-cluster mean distance; average silhouette compares k. Dunn index rewards large intercluster separation relative to large intracluster diameter (higher is better). Davies–Bouldin index averages cluster similarity ratios of within-scatter to between-centroid separation (lower is better). All optimize geometry, not scientific usefulness.

![Elbow plot of WSS vs k on the six-point toy set (original).](../assets/figures/ml_fig_elbow_wss.png)

*Figure 4.5. Elbow plot of within-cluster sum of squares (WSS) against k for k-means on the six-point toy set. WSS drops sharply from 74.2 at k = 1 to 4.0 at k = 2, then flattens; the diminishing-returns bend (amber dashed line) marks the elbow at k = 2, matching the visual two-clump structure.*

![Mean silhouette vs k on synthetic three-blob data (scientific; original).](../assets/figures/ml_fig_silhouette_k.png)

*Figure — Silhouette as a k-choice companion to the elbow. **Left:** standardized two-feature synthetic data with three spherical blobs and k-means centroids (k = 3). **Right:** mean silhouette over k = 2…7 peaks at the generating structure (k = 3). Silhouette s(i) = (b(i) − a(i)) / max{a(i), b(i)} rewards compact, well-separated geometry; a high peak still does **not** prove clinical phenotypes, TOAST etiology, or treatment-effect strata—only that the partition fits the chosen distance. Use silhouette with stability (bootstrap / site holdout) and a pre-specified utility question, never as sole evidence for reifying clusters.*

![Gap statistic for choosing k vs reference null (synthetic; original).](../assets/figures/ml_fig_gap_statistic.png)

*Figure — Gap as another k heuristic. **Left:** three-blob synthetic geometry. **Right:** \(\mathrm{Gap}(k)=E^*[\log W_k]-\log W_k\) peaks near the generating \(k\). Compare observed within-cluster dispersion to a uniform reference; still a geometric score, not etiology. Pair gap/elbow/silhouette with stability and a pre-registered use case—clusters remain hypothesis generators, not causal subtypes.*

![Density clustering with noise points and stability scores (synthetic; original).](../assets/figures/ml_fig_density_stability.png)\n![Bootstrap co-association frequency heatmap (synthetic; original).](../assets/figures/ml_fig_coassociation.png)

*Figure — Co-association stability. Entry (i,j) is how often points co-cluster across bootstrap restarts. Block-diagonal structure signals geometric reproducibility, not named disease subtypes. **Clusters ≠ causal labels**.*\n\n


![1-D GMM density and soft responsibilities (synthetic; original).](../assets/figures/ml_fig_gmm_responsibilities.png)

*Figure — Soft clustering. Top: mixture and weighted components; bottom: E-step responsibilities. Probabilistic membership is geometric—not etiologic certainty. **Clusters ≠ causal subtypes**.*

*Figure — Not every point needs a cluster. **Left:** two dense groups plus noise. **Right:** teaching stability bars—prefer persistent clusters over ephemeral splits. Stability is geometric; noise points need review paths, not forced labels. Clusters ≠ etiologic subtypes.*

### Extrinsic methods: purity and Rand index

When reference labels exist for audit (for example, known TOAST codes), purity measures the fraction of points in each cluster belonging to its majority class, aggregated—high purity can coexist with over-fragmentation (many tiny pure clusters). The Rand index is the fraction of point pairs on which the clustering and the reference agree about being same-cluster vs different-cluster; adjusted Rand corrects for chance. External indices validate recovery of a reference ontology; they do not prove the reference is the right biology.

![Adjusted Rand Index vs k and bootstrap partition stability (synthetic; original).](../assets/figures/ml_fig_adjusted_rand.png)

*Figure — ARI and stability. **Left:** planted three-blob data with audit labels (not “discovered biology”). **Right:** ARI of k-means vs the planted partition peaks at true *k*; bootstrap ARI bands show when partitions are unstable under resampling. High ARI means recovery of a reference coding, not proof of clinical phenotypes or treatment-effect strata—pair with pre-specified utility and site holdouts.*

Stability under bootstrap resampling of patients, feature subsets, and site holdouts is often more informative than a single silhouette peak. Domain utility—whether clusters stratify outcomes or imaging patterns in pre-specified analyses—matters more than any internal score.

## Algorithm Selection: A Decision Guide

No clustering algorithm dominates all geometries. If you believe clusters are roughly spherical and of similar size in a standardized Euclidean space, k-means with many restarts is a strong, simple baseline. If outliers should not drag centers and you need actual exemplar patients as centers, prefer k-medoids with a domain-appropriate distance. If clusters may be elongated chains of density, DBSCAN or OPTICS are more natural than k-means—accepting the burden of density parameters and the possibility of many noise points. If you need a full hierarchy for taxonomy exploration at small n, agglomerative clustering with Ward or average linkage is interpretable; single linkage (SLINK) is the specialist tool when chaining is scientifically desired rather than feared.

When n is large and memory is tight, BIRCH-style summarization or mini-batch k-means become practical; CURE-style multiple representatives help when shapes are non-spherical but still compact enough to summarize. When uncertainty of assignment matters and elliptical blobs are plausible, GMMs with EM provide soft responsibilities and a likelihood for model comparison (AIC/BIC with caution). Fuzzy C-means offers soft memberships without a full generative story—useful as a robust soft partitioner, dangerous if memberships are misread as disease probabilities.

Hybrid workflows often outperform single-algorithm loyalty. Example: reduce imaging embeddings with PCA; run k-means with k suggested by silhouette and stability; compare to HDBSCAN/OPTICS noise labels as a sensitivity analysis; freeze a configuration; replicate on a second site before any manuscript names a phenotype. Another example: use hierarchical clustering on a subsample to propose k and structure, then refine with k-medoids on the full distance matrix. Document each choice as you would document a lab assay.

Spherical, similar size, Euclidean → k-means (+ restarts, k-means++).

Arbitrary distances / exemplars needed → k-medoids.

Non-spherical density pockets + noise → DBSCAN/OPTICS.

Taxonomy exploration at modest n → hierarchical (Ward/average; SLINK if chaining OK).

Large n, limited memory → BIRCH, mini-batch k-means, sampling + CURE ideas.

Soft elliptical blobs → GMM/EM; soft non-generative → Fuzzy C-means.

## Evaluation Deep Dive with Numerical Checks

Intrinsic indices answer: ‘How compact and separated is this partition under distance d?’ They do not answer: ‘Is this the disease taxonomy we should write into guidelines?’ Still, they are useful for rejecting absurd k and for comparing algorithms on the same feature space. Elbow plots of WSS versus k for k-means on the six-point toy data would show a large drop from k=1 to k=2 and smaller gains thereafter—consistent with the visual two-clump structure—but even a clear elbow can be wrong under label shift or feature leakage.

![4.6: Per-sample silhouette coefficients s(i) for a three-cluster dataset, sorted within each cluster. Longer bars indicate po](../assets/figures/ml_concept_4.6_ee0aedbb.png)

*Figure 4.6 — original teaching graphic.*

Silhouette analysis should be reported as a distribution, not only a mean. A mean silhouette of 0.5 with a mass of negative silhouettes in one purported cluster means that cluster is a dumping ground. In clinical data, negative silhouettes often flag patients with mixed physiology or with features dominated by a site effect. Dunn’s index is sensitive to the single worst cluster diameter and the single closest pair of clusters—high variance, but good at flagging near-merges. Davies–Bouldin averages similarity ratios; it is convenient and, like silhouette, prefers convex-ish clusters, so it can unfairly punish legitimate elongated DBSCAN structures.

Extrinsic scores require a reference labeling. Purity is intuitive for multi-class audit but rewards splitting each true class into many tiny pure clusters. Pair the Rand index (or adjusted Rand) with a statement of what the reference means: chart-adjudicated TOAST is not the same as billing-code TOAST. If adjusted Rand is high against billing codes but low against adjudication, you have learned documentation clusters—important for operations, misleading for etiology.

# Silhouette sketch for point A in the worked example
# C1={A,B,C}, C2={D,E,F}
# d(A,B)=1, d(A,C)=1 -> a(A)=1
# mean dist A to {D,E,F} ~ 7.341 -> b(A)
# s(A)=(b-a)/max(a,b)~0.864
a_A, b_A = 1.0, 7.341
s_A = (b_A - a_A) / max(a_A, b_A)
print(round(s_A, 3))

A practical evaluation battery for a stroke clustering paper: (1) pre-specify features and scaling; (2) report WSS elbow and silhouette distributions for a range of k; (3) report Dunn and Davies–Bouldin as secondary; (4) if labels exist for audit, report adjusted Rand and purity without claiming etiology; (5) bootstrap cluster stability (Jaccard of matched clusters across resamples); (6) site-holdout replication; (7) outcome association only as descriptive secondary analysis with multiplicity humility.

## Other Clustering Variants You Will Encounter

Beyond the TOC core, several variants appear in clinical papers. Spectral clustering uses eigenvectors of graph Laplacians built from similarity graphs—effective for non-convex clusters when similarities are well tuned, but sensitive to graph construction.

![Spectral clustering: moons in input space vs k-means on Laplacian eigenvectors (synthetic; original).](../assets/figures/ml_fig_spectral_clustering.png)

*Figure — Spectral sketch. **Left:** two interlocking moons recovered as clusters. **Right:** points in the embedding of the first nontrivial eigenvectors of the normalized Laplacian, where ordinary k-means separates them. Affinity kernel width is a sensitive hyperparameter; spectral partitions remain geometry, not etiology.*


![DBSCAN ε trade-off: cluster count vs noise count (synthetic; original).](../assets/figures/ml_fig_dbscan_eps.png)

*Figure — Density radius ε. Larger ε merges clusters and reduces noise labels. Parameter choice is geometric—not a clinical subtype control.*


![Agglomerative dendrogram schematic with a cut height (original).](../assets/figures/ml_fig_dendrogram_schematic.png)

*Figure — Hierarchical clustering tree. Cutting at a height chooses k. Dendrograms summarize distance geometry—not disease taxonomies without labels.*


![k-means initialization sensitivity on two blobs (synthetic; original).](../assets/figures/ml_fig_kmeans_init.png)

*Figure — Bad inits can stall; multi-start and k-means++ help. Init is optimization—not a window into true disease subtypes.*


![Uneven cluster sizes pie (synthetic; original).](../assets/figures/ml_fig_cluster_size_imbalance.png)

*Figure — Geometry of partitions—not etiologic prevalence. Pred ≠ cause without design.*


![k-means centroid migration paths (original).](../assets/figures/ml_fig_centroid_paths.png)

*Figure — Centroids move until assignment stabilizes. k-means centroid migration paths Pred != cause without design.*


![linkage teaching panel (original).](../assets/figures/ml_fig_linkage_compare.png)

*Figure — Teaching panel for linkage. Pred != cause without design.*


![Cycle-34 densify scientific panel 6 (original).](../assets/figures/ml_fig_c34_05.png)

*Figure — Continuous densify panel 6. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 6 (original).](../assets/figures/ml_fig_c35_05.png)

*Figure — Continuous densify panel 6. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 6 (original).](../assets/figures/ml_fig_c36_05.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 6 (original).](../assets/figures/ml_fig_c37_05.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 6 (original).](../assets/figures/ml_fig_c38_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 6 (original).](../assets/figures/ml_fig_c39_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 6 (original).](../assets/figures/ml_fig_c40_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 6 (original).](../assets/figures/ml_fig_c41_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 6 (original).](../assets/figures/ml_fig_c42_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 6 (original).](../assets/figures/ml_fig_c43_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 6 (original).](../assets/figures/ml_fig_c44_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 6 (original).](../assets/figures/ml_fig_c45_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 6 (original).](../assets/figures/ml_fig_c46_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 6 (original).](../assets/figures/ml_fig_c47_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 6 (original).](../assets/figures/ml_fig_c48_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 6 (original).](../assets/figures/ml_fig_c49_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 6 (original).](../assets/figures/ml_fig_c50_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 6 (original).](../assets/figures/ml_fig_c51_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 6 (original).](../assets/figures/ml_fig_c52_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 6 (original).](../assets/figures/ml_fig_c53_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 6 (original).](../assets/figures/ml_fig_c54_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 6 (original).](../assets/figures/ml_fig_c55_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 6 (original).](../assets/figures/ml_fig_c56_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 6 (original).](../assets/figures/ml_fig_c57_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 6 (original).](../assets/figures/ml_fig_c58_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 6 (original).](../assets/figures/ml_fig_c59_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 6 (original).](../assets/figures/ml_fig_c60_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 6 (original).](../assets/figures/ml_fig_c61_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 6 (original).](../assets/figures/ml_fig_c62_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 6 (original).](../assets/figures/ml_fig_c63_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 6 (original).](../assets/figures/ml_fig_c64_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 6 (original).](../assets/figures/ml_fig_c65_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 6 (original).](../assets/figures/ml_fig_c66_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 6 (original).](../assets/figures/ml_fig_c67_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 6 (original).](../assets/figures/ml_fig_c68_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 6 (original).](../assets/figures/ml_fig_c69_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 6 (original).](../assets/figures/ml_fig_c70_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 6 (original).](../assets/figures/ml_fig_c71_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 6 (original).](../assets/figures/ml_fig_c72_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 6 (original).](../assets/figures/ml_fig_c73_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 6 (original).](../assets/figures/ml_fig_c74_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 6 (original).](../assets/figures/ml_fig_c75_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 6 (original).](../assets/figures/ml_fig_c76_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 6 (original).](../assets/figures/ml_fig_c77_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 6 (original).](../assets/figures/ml_fig_c78_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 6 (original).](../assets/figures/ml_fig_c79_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 6 (original).](../assets/figures/ml_fig_c80_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 6 (original).](../assets/figures/ml_fig_c81_05.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

Mean-shift finds modes of a kernel density estimate without fixing k, related in spirit to density methods. Hierarchical density methods (HDBSCAN) combine ideas from OPTICS and hierarchy to extract stable clusters across density levels. Constrained clustering incorporates must-link and cannot-link hints from partial labels—semi-supervised structure discovery when experts can say ‘these two cases should not share a phenotype’ without full labeling.

Biclustering and co-clustering simultaneously cluster rows and columns of a matrix—patients and genes, or patients and sparse code sets—producing blocks of co-behavior. Topic models on clinical text are soft co-clustering cousins. Subspace clustering seeks clusters that exist only in feature subsets, relevant when different stroke mechanisms live in different biomarker panels. These methods multiply researcher degrees of freedom; pre-registration of analytic choices becomes even more important.

Online and streaming clustering update partitions as new encounters arrive—useful for monitoring whether a new care pathway creates a novel cluster in feature space. Concept drift in clusters may reflect true epidemiology (a new drug-use pattern in young stroke) or documentation drift. Treat streamed cluster IDs as operational features under governance, not as automatic alerts to clinicians without prospective evaluation.

## Clinical and Epidemiologic Notes

Stroke research has long used subtype taxonomies built from expert rules, imaging, and vascular workups. Unsupervised clustering of mixed clinical–imaging–omics features is sometimes proposed as a path to data-driven phenotypes. Methodological truth is narrower: clustering returns structure relative to features, distance, and algorithm. If features are dominated by severity and age, clusters may largely recreate severity strata already visible on a histogram of NIHSS. If features include hospital indicators, clusters may recreate referral networks. If features include post-treatment variables, clusters may encode care received rather than presenting biology—leaking the future into the phenotype label. Imaging-derived clusters deserve special suspicion: radiomic and deep-embedding features drift with scanner vendor, field strength, slice thickness, reconstruction kernel, and acquisition protocol, so an unsupervised partition of a multi-site imaging cohort can recover the scanner inventory rather than any neurobiology. A ‘cluster’ whose top discriminating features are texture statistics that happen to track a 1.5T-versus-3T split, or one site’s contrast timing, is a machine signature wearing a phenotype costume. Harmonization (for example, ComBat-style removal of site/scanner batch effects) and scanner-stratified or scanner-held-out replication are minimum defenses before any such cluster is called biological.

Reification is the central hazard: naming Cluster 2 ‘inflammatory LVO’ because mean CRP is higher does not establish an etiologic entity, a latent disease class, or a treatment target. Causal claims require designs that support them—trials, natural experiments, longitudinal processes with temporal order, or explicit causal models—not colorful dendrograms. Association of cluster labels with 90-day mRS shows prognostic separation under the selection of the cohort; it does not show that intervening to move a patient between clusters is possible or beneficial.

Epidemiologically, unsupervised is not etiology. Mixture models can still be useful as flexible density estimators, as tools for stratified descriptive Table 1, as anomaly detectors (DBSCAN noise as chart review candidates), or as pre-processing for downstream supervised models—if leakage is prevented and validation is external. Population studies should report how the analytic sample was selected, how missing vascular workup items were handled, and whether cluster structure replicates across sites and eras. Spectrum bias applies: clusters found in a comprehensive-center thrombectomy registry will not automatically appear in a community mild-stroke cohort.

For phenotype papers, demand: pre-specified feature sets (no outcome leakage), scaling recipes, multiple algorithms as sensitivity analyses, stability assessments, external replication, and clear language that partitions are operational groupings pending biological and clinical validation—not instant subtypes for guidelines. Used carefully, clustering supports description, anomaly review, and stratified analysis; used carelessly, it manufactures subtypes.

Exclude post-outcome and post-treatment features unless the estimand is care-path clustering.

Check that clusters are not proxies for site, era, or missing-data patterns.

Replicate partitions on external cohorts before clinical storytelling.

Use clusters as strata for description or prognosis only with appropriate validation language.

Never equate unsupervised separation with causal subtype discovery.

## Distance Choice Worked Micro-Examples

Consider two binary comorbidity vectors of length 4: u = (1,1,0,0) and v = (1,0,1,0). Hamming distance counts disagreeing coordinates: positions 2 and 3 differ, so d_H = 2. Jaccard similarity treats them as sets of active codes {1,2} vs {1,3}: intersection size 1, union size 3, so Jaccard = 1/3 and Jaccard distance = 2/3. Cosine similarity is (u·v)/(||u|| ||v||) = 1/(√2 · √2) = 0.5, so a common cosine distance is 0.5. Already three different ‘nearness’ numbers appear from the same pair; clustering them with different measures can reorder neighbors. For clinical code sets, Jaccard often matches the intuition of shared disease burden better than Hamming, which heavily weights shared zeros (the many diseases neither patient has).

For strings, compare care-sequence ‘ED-CT-tPA’ to ‘ED-CT-IR-tPA’. Levenshtein distance is 1 if we count a single insertion of ‘IR’ as one edit under a token alphabet, or more if computed at character level—token versus character granularity is part of the method definition. LCS length would be three shared ordered tokens (ED, CT, tPA) with a gap, yielding a high similarity despite an extra step. For vital-sign series that rise at different speeds, DTW can align a slow and a fast climb into a small cost, whereas pointwise Euclidean on clock time declares them far apart. Graph similarity on hospital-transfer networks might score two regional systems as similar if they share a hub-and-spoke pattern even when node labels (hospital names) differ—structure over identity.

Mahalanobis distance matters when features are correlated: admission NIHSS and GCS motor score move together, so double-counting their Euclidean contributions inflates severity separation along a single clinical axis. Estimating Σ on the training cohort and using d_M is a partial remedy, provided p is not too large relative to n. In high-dimensional embeddings from imaging networks, cosine similarity after L2 normalization is often more stable than Mahalanobis with a full free Σ.

## Hierarchical Linkage Walk-Through

Suppose four points on a line: P1=0, P2=1, P3=10, P4=11. Agglomerative clustering with single linkage first merges the closest pair—either (P1,P2) at distance 1 or (P3,P4) at distance 1 (ties broken by implementation). After merging (P1,P2) and (P3,P4), single-linkage distance between clusters is min distance across clusters = |1−10| = 9, so the dendrogram’s final merge sits at height 9. Complete linkage would use max cross distance = 11 at the final merge. The dendrogram heights differ even when the final bipartition matches. With a noise point at 5, single linkage may chain left and right through the bridge earlier than complete linkage, which resists chaining. That geometric story is why SLINK is powerful for elongated structures and hazardous for noisy clinical spaces with bridging outliers.

DIANA on the same four points would start with one cluster and seek a split that reduces heterogeneity: the natural first cut separates {P1,P2} from {P3,P4}. Divisive methods can recover large-scale separation first—sometimes aligning better with coarse clinical strata—before refining within arms. Computational cost grows when each split requires searching many bipartitions; practical DIANA implementations use heuristics rather than exhaustive search. In either hierarchical direction, cutting the tree is a model selection act analogous to choosing k in k-means; silhouette-on-cut and stability across bootstrap dendrograms are more defensible than aesthetic branch coloring.

## GMM Responsibilities: A One-Dimensional Numeric Sketch

Suppose a one-dimensional two-component mixture with equal weights π_1 = π_2 = 0.5, means μ_1 = 0, μ_2 = 6, and shared variance σ² = 1. For a point x = 2, the Gaussian densities are proportional to exp(−(2−0)²/2) = exp(−2) ≈ 0.1353 and exp(−(2−6)²/2) = exp(−8) ≈ 0.000335. Responsibilities: r_1 ∝ 0.5·0.1353, r_2 ∝ 0.5·0.000335, so after normalization r_1 ≈ 0.9975, r_2 ≈ 0.0025. The point is softly but almost certainly in component 1. For x = 3, densities exp(−4.5)≈0.0111 and exp(−4.5)≈0.0111 if means were symmetric—at x=3 with μ=(0,6), exp(−(3)²/2)=exp(−4.5)≈0.0111 and exp(−(3−6)²/2)=exp(−4.5)≈0.0111, so r ≈ (0.5, 0.5). Boundary patients get honest uncertainty; hard k-means would still force a single label.

Clinically, a soft weight of 0.5/0.5 means ‘under this feature geometry, the point sits between modes,’ not ‘50% chance of TOAST cardioembolic.’ Translating responsibilities into care still requires a supervised or rule-based layer, prospective validation, and human review. EM will also happily fit spurious extra components when k is too large; BIC penalties and held-out likelihood help, but scientific meaning remains external to the optimizer.

## From Unsupervised Structure to Downstream Use

Clustering rarely ends with a colored scatter plot. Legitimate downstream uses include: stratified descriptive statistics (Table 1 by cluster); sampling frames for chart review (oversample rare density regions or DBSCAN noise); initialization of mixture components for more elaborate models; and features for supervised learning, where cluster IDs or distances to medoids become inputs—constructed only from pre-index data if the supervised task is predictive. Illegitimate uses include: auto-assigning etiologic labels in the EHR, changing antithrombotic therapy solely because a patient falls near a cluster centroid, or claiming discovery of a new disease without biologic and prospective validation. The ethical line tracks the epidemiologic line: description and hypothesis generation versus claims that change individual care.

When cluster features enter a supervised model, evaluation must block leakage carefully. If clusters were fit on the full dataset including future test patients, test metrics are optimistic. Fit the clustering on training folds only, then assign test points to existing centers or density models (nearest medoid, highest posterior responsibility, or noise if outside density). That discipline mirrors how you would deploy the pipeline on tomorrow’s admissions.

## Pitfalls and Practical Recommendations

Pitfalls: (1) Not standardizing features before Euclidean k-means. (2) Treating a single random initialization as definitive. (3) Choosing k by maximizing a metric computed on the same data used to fit without stability checks. (4) Forcing every patient into a cluster when a noise category would be more honest. (5) Clustering on n ≈ p high-dimensional embeddings without dimensionality control—distances concentrate and clusters become unstable. (6) Leakage of labels or outcomes into features. (7) Interactive tweaking of ε until the figure matches a preferred narrative. (8) Publishing heatmaps of cluster × biomarker means as if they were confirmatory hypothesis tests without multiplicity discipline. (9) Using cosine distance on unnormalized sparse codes without understanding magnitude invariance. (10) Interpreting Fuzzy C-means memberships as calibrated subtype probabilities.

Practical recommendations: start with EDA projections and pairwise plots; standardize continuous features; run k-means with many restarts and k-means++; compare to hierarchical Ward and to a density method when non-spherical structure is plausible; report silhouette distributions and stability but lead with scientific estimands; keep a held-out site or temporal cohort for replication; and write limitations that a neurologist–epidemiologist reviewer would respect. Clustering is a tool for organizing heterogeneity, not a machine for minting disease entities.

When communicating to clinical audiences, lead with the question (‘Can we describe distinct admission profiles for staffing?’) rather than the algorithm name. Show a few exemplar patients (medoids) per cluster with readable timelines. Show what did not work: the k that collapsed under bootstrap, the site that would not replicate. Trust grows from transparent failure modes more than from a single polished t-SNE coloring.

## Connections

Clustering sits downstream of the preprocessing and distance choices developed earlier: standardization, encoding, and dimensionality reduction (PCA or nonlinear embeddings) reshape the geometry every algorithm here optimizes, so a partition is only as meaningful as the feature space feeding it. It is kin to the mixture-model and EM machinery of latent-variable estimation—k-means is a hard, spherical special case of the Gaussian mixture, and GMM responsibilities are its soft generalization. It connects forward to supervised learning twice over: cluster IDs or distances-to-medoids can become engineered features (only when built from pre-index data, to block leakage), and the same held-out and external-validation discipline used for classifiers decides whether a partition transports. DBSCAN noise ties clustering to anomaly and novelty detection, and DTW ties it to trajectory and time-series modeling. Finally, the chapter’s recurring caution—unsupervised separation is not etiology—connects to the causal-inference material: moving a patient between clusters is a counterfactual claim that clustering alone can never license.


![c82 teaching panel 05 (original).](../assets/figures/ml_fig_c82_05.png)
*Figure — Choosing k: inertia elbow plus silhouette separation check. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 05 (original).](../assets/figures/ml_fig_c83_05.png)
*Figure — DBSCAN idea: dense cores, borders, and noise points. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 05 (original).](../assets/figures/ml_fig_c84_05.png)
*Figure — Hierarchical structure sketch for multi-level data. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 05 (original).](../assets/figures/ml_fig_c85_05.png)
*Figure — Non-convex clusters break spherical k-means assumptions. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 05 (original).](../assets/figures/ml_fig_c86_05.png)
*Figure — Gap statistic sketch for selecting cluster count k. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 05 (original).](../assets/figures/ml_fig_c87_05.png)
*Figure — Dendrogram cut selects flat clusters from hierarchy. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 05 (original).](../assets/figures/ml_fig_c88_05.png)
*Figure — GMM soft assignment as responsibility weights. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 05 (original).](../assets/figures/ml_fig_c89_05.png)
*Figure — Hierarchical cut line → k clusters. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 05 (original).](../assets/figures/ml_fig_c90_05.png)
*Figure — Cluster stability across bootstraps. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 05 (original).](../assets/figures/ml_fig_c91_05.png)
*Figure — Spectral clustering Laplacian idea. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 05 (original).](../assets/figures/ml_fig_c92_05.png)
*Figure — Core-set / landmark sampling. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 05 (original).](../assets/figures/ml_fig_c93_05.png)
*Figure — HDBSCAN density hierarchy. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 05 (original).](../assets/figures/ml_fig_c94_05.png)
*Figure — Mean-shift mode seeking path. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 05 (original).](../assets/figures/ml_fig_c95_05.png)
*Figure — Affinity propagation exemplars. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 05 (original).](../assets/figures/ml_fig_c96_05.png)
*Figure — OPTICS reachability plot idea. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 05 (original).](../assets/figures/ml_fig_c97_05.png)
*Figure — BIRCH CF-tree clustering sketch. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 05 (original).](../assets/figures/ml_fig_c98_05.png)
*Figure — CURE representative points. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 05 (original).](../assets/figures/ml_fig_c99_05.png)
*Figure — CLIQUE subspace clusters. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 05 (original).](../assets/figures/ml_fig_c100_05.png)
*Figure — Streaming k-means centroids. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 05 (original).](../assets/figures/ml_fig_c101_05.png)
*Figure — Active learning query strategies. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 05 (original).](../assets/figures/ml_fig_c102_05.png)
*Figure — Deep embedded clustering. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 05 (original).](../assets/figures/ml_fig_c103_05.png)
*Figure — Mini-batch k-means speed path. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 05 (original).](../assets/figures/ml_fig_c104_05.png)
*Figure — Uncertainty sampling vs diversity. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 05 (original).](../assets/figures/ml_fig_c105_05.png)
*Figure — Contrastive clustering. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 05 (original).](../assets/figures/ml_fig_c106_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 05 (original).](../assets/figures/ml_fig_c107_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 05 (original).](../assets/figures/ml_fig_c108_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 05 (original).](../assets/figures/ml_fig_c109_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 05 (original).](../assets/figures/ml_fig_c110_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 05 (original).](../assets/figures/ml_fig_c111_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 05 (original).](../assets/figures/ml_fig_c112_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 05 (original).](../assets/figures/ml_fig_c113_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 05 (original).](../assets/figures/ml_fig_c114_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 05 (original).](../assets/figures/ml_fig_c115_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 05 (original).](../assets/figures/ml_fig_c116_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 05 (original).](../assets/figures/ml_fig_c117_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 05 (original).](../assets/figures/ml_fig_c118_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 05 (original).](../assets/figures/ml_fig_c119_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 05 (original).](../assets/figures/ml_fig_c120_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 05 (original).](../assets/figures/ml_fig_c121_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 05 (original).](../assets/figures/ml_fig_c122_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 05 (original).](../assets/figures/ml_fig_c123_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 05 (original).](../assets/figures/ml_fig_c124_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 05 (original).](../assets/figures/ml_fig_c125_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 05 (original).](../assets/figures/ml_fig_c126_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 05 (original).](../assets/figures/ml_fig_c127_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 05 (original).](../assets/figures/ml_fig_c128_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 05 (original).](../assets/figures/ml_fig_c129_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 05 (original).](../assets/figures/ml_fig_c130_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 05 (original).](../assets/figures/ml_fig_c131_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 05 (original).](../assets/figures/ml_fig_c132_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 05 (original).](../assets/figures/ml_fig_c133_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 05 (original).](../assets/figures/ml_fig_c134_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 05 (original).](../assets/figures/ml_fig_c135_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 05 (original).](../assets/figures/ml_fig_c136_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 05 (original).](../assets/figures/ml_fig_c137_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 05 (original).](../assets/figures/ml_fig_c138_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 05 (original).](../assets/figures/ml_fig_c139_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 05 (original).](../assets/figures/ml_fig_c140_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 05 (original).](../assets/figures/ml_fig_c141_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 05 (original).](../assets/figures/ml_fig_c142_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 05 (original).](../assets/figures/ml_fig_c143_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 05 (original).](../assets/figures/ml_fig_c144_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 05 (original).](../assets/figures/ml_fig_c145_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 05 (original).](../assets/figures/ml_fig_c146_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 05 (original).](../assets/figures/ml_fig_c147_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 05 (original).](../assets/figures/ml_fig_c148_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 05 (original).](../assets/figures/ml_fig_c149_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 05 (original).](../assets/figures/ml_fig_c150_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 05 (original).](../assets/figures/ml_fig_c151_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 05 (original).](../assets/figures/ml_fig_c152_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 05 (original).](../assets/figures/ml_fig_c153_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 05 (original).](../assets/figures/ml_fig_c154_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 05 (original).](../assets/figures/ml_fig_c155_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 05 (original).](../assets/figures/ml_fig_c156_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 05 (original).](../assets/figures/ml_fig_c157_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 05 (original).](../assets/figures/ml_fig_c158_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 05 (original).](../assets/figures/ml_fig_c159_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 05 (original).](../assets/figures/ml_fig_c160_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 05 (original).](../assets/figures/ml_fig_c161_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 05 (original).](../assets/figures/ml_fig_c162_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 05 (original).](../assets/figures/ml_fig_c163_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 05 (original).](../assets/figures/ml_fig_c164_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 05 (original).](../assets/figures/ml_fig_c165_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 05 (original).](../assets/figures/ml_fig_c166_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 05 (original).](../assets/figures/ml_fig_c167_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 05 (original).](../assets/figures/ml_fig_c168_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 05 (original).](../assets/figures/ml_fig_c169_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 05 (original).](../assets/figures/ml_fig_c170_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 05 (original).](../assets/figures/ml_fig_c171_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 05 (original).](../assets/figures/ml_fig_c172_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 05 (original).](../assets/figures/ml_fig_c173_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 05 (original).](../assets/figures/ml_fig_c174_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 05 (original).](../assets/figures/ml_fig_c175_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 05 (original).](../assets/figures/ml_fig_c176_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 05 (original).](../assets/figures/ml_fig_c177_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 05 (original).](../assets/figures/ml_fig_c178_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 05 (original).](../assets/figures/ml_fig_c179_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 05 (original).](../assets/figures/ml_fig_c180_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 05 (original).](../assets/figures/ml_fig_c181_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 05 (original).](../assets/figures/ml_fig_c182_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 05 (original).](../assets/figures/ml_fig_c183_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 05 (original).](../assets/figures/ml_fig_c184_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 05 (original).](../assets/figures/ml_fig_c185_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 05 (original).](../assets/figures/ml_fig_c186_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 05 (original).](../assets/figures/ml_fig_c187_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 05 (original).](../assets/figures/ml_fig_c188_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 05 (original).](../assets/figures/ml_fig_c189_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 05 (original).](../assets/figures/ml_fig_c190_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 05 (original).](../assets/figures/ml_fig_c191_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 05 (original).](../assets/figures/ml_fig_c192_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 05 (original).](../assets/figures/ml_fig_c193_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 05 (original).](../assets/figures/ml_fig_c194_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 05 (original).](../assets/figures/ml_fig_c195_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 05 (original).](../assets/figures/ml_fig_c196_05.png)
*Figure — Fuzzy c-means membership. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 05 (original).](../assets/figures/ml_fig_c197_05.png)
*Figure — Self-organizing map grid. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 05 (original).](../assets/figures/ml_fig_c198_05.png)
*Figure — Affinity matrix threshold. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 05 (original).](../assets/figures/ml_fig_c199_05.png)
*Figure — Consensus clustering. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 05 (original).](../assets/figures/ml_fig_c200_05.png)
*Figure — COP-kmeans constraints. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 05 (original).](../assets/figures/ml_fig_c201_05.png)
*Figure — Silhouette a-b geometry. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 05 (original).](../assets/figures/ml_fig_c202_05.png)
*Figure — Density core border validity. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 05 (original).](../assets/figures/ml_fig_c203_05.png)
*Figure — Gap statistic null reference. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 05 (original).](../assets/figures/ml_fig_c204_05.png)
*Figure — Bootstrap cluster stability ARI. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 05 (original).](../assets/figures/ml_fig_c205_05.png)
*Figure — Agglomerative dendrogram height. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 05 (original).](../assets/figures/ml_fig_c206_05.png)
*Figure — GMM soft responsibilities. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 05 (original).](../assets/figures/ml_fig_c207_05.png)
*Figure — HDBSCAN condensed tree cut. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 05 (original).](../assets/figures/ml_fig_c208_05.png)
*Figure — Mean-shift mode seeking path. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 05 (original).](../assets/figures/ml_fig_c209_05.png)
*Figure — OPTICS reachability valleys. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 05 (original).](../assets/figures/ml_fig_c210_05.png)
*Figure — Spectral Fiedler cluster split. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 05 (original).](../assets/figures/ml_fig_c211_05.png)
*Figure — Affinity propagation messages. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 05 (original).](../assets/figures/ml_fig_c212_05.png)
*Figure — BIRCH clustering feature tree. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 05 (original).](../assets/figures/ml_fig_c213_05.png)
*Figure — CURE representative scatter. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 05 (original).](../assets/figures/ml_fig_c214_05.png)
*Figure — Streaming k-means centroids. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 05 (original).](../assets/figures/ml_fig_c215_05.png)
*Figure — DENCLUE density hill climb. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 05 (original).](../assets/figures/ml_fig_c216_05.png)
*Figure — Mini-batch k-means WSS path. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 05 (original).](../assets/figures/ml_fig_c217_05.png)
*Figure — Clique percolation overlap. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 05 (original).](../assets/figures/ml_fig_c218_05.png)
*Figure — Spectral embedding cluster seeds. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 05 (original).](../assets/figures/ml_fig_c219_05.png)
*Figure — ROCK link agglomeration. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 05 (original).](../assets/figures/ml_fig_c220_05.png)
*Figure — CHAMELEON partition merge. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 05 (original).](../assets/figures/ml_fig_c221_05.png)
*Figure — BIRCH CF-tree hierarchy. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 05 (original).](../assets/figures/ml_fig_c222_05.png)
*Figure — HDBSCAN condensed lifetimes. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 05 (original).](../assets/figures/ml_fig_c223_05.png)
*Figure — OPTICS reachability valleys. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 05 (original).](../assets/figures/ml_fig_c224_05.png)
*Figure — Mean-shift mode seeking field. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 05 (original).](../assets/figures/ml_fig_c225_05.png)
*Figure — Spectral clustering Fiedler cut. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 05 (original).](../assets/figures/ml_fig_c226_05.png)
*Figure — Affinity propagation exemplars. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 05 (original).](../assets/figures/ml_fig_c227_05.png)
*Figure — CURE representative clustering. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 05 (original).](../assets/figures/ml_fig_c228_05.png)
*Figure — DENCLUE density attractors. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 05 (original).](../assets/figures/ml_fig_c229_05.png)
*Figure — Clique percolation overlap. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 05 (original).](../assets/figures/ml_fig_c230_05.png)
*Figure — Streaming k-means centers. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 05 (original).](../assets/figures/ml_fig_c231_05.png)
*Figure — DBSCAN core border noise. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 05 (original).](../assets/figures/ml_fig_c232_05.png)
*Figure — HDBSCAN cluster stability bars. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 05 (original).](../assets/figures/ml_fig_c233_05.png)
*Figure — Bandwidth multi-scale clusters. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 05 (original).](../assets/figures/ml_fig_c234_05.png)
*Figure — Spectral bicluster blocks. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 05 (original).](../assets/figures/ml_fig_c235_05.png)
*Figure — Variable-density cluster scatter. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 05 (original).](../assets/figures/ml_fig_c236_05.png)
*Figure — Co-cluster block heat. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 05 (original).](../assets/figures/ml_fig_c237_05.png)
*Figure — Noise vs dense modes scatter. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 05 (original).](../assets/figures/ml_fig_c238_05.png)
*Figure — Stochastic block heat. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 05 (original).](../assets/figures/ml_fig_c239_05.png)
*Figure — Core-distance k-dist scatter. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 05 (original).](../assets/figures/ml_fig_c240_05.png)
*Figure — Degree-corrected block heat. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Clustering groups unlabeled points under a stated distance and objective. Similarity choices—Euclidean, Manhattan, Mahalanobis, Hamming, Levenshtein, LCS, cosine, Jaccard, DTW, and graph-based measures—reshape results and must be reported with scaling recipes. K-means minimizes WCSS via assignment and mean updates; k-median and k-medoid offer robustness and exemplar centers. A worked six-point example converged to clusters {A,B,C} and {D,E,F} with J=4. DBSCAN and OPTICS find density-connected structure and noise; hierarchical SLINK (single linkage) and DIANA provide agglomerative and divisive trees; BIRCH and CURE scale hierarchical ideas to large n. GMMs and Fuzzy C-means supply soft memberships. Evaluation uses elbow/WSS, silhouette, Dunn, Davies–Bouldin, purity, and Rand-family indices, with stability and external replication outweighing any single geometric score. In stroke and population research, unsupervised structure is not etiology: control leakage, test transportability, and resist reification.

## Practice and Reflection

(1) Rerun the six-point k-means example with initial centers μ1=(2,1) and μ2=(6,5). Show the first assignment step and updated means.

(2) Compute the silhouette s(D) for point D in the converged partition of the worked example (Euclidean distance).

(3) Explain geometrically why single linkage can merge two dense clusters connected by a thin chain of points; give a clinical scenario where that is undesirable.

(4) For DBSCAN with minPts=3, describe qualitatively how increasing ε changes core points on data with one tight and one loose cloud.

(5) Write the E-step responsibility formula for a two-component 1-D GMM and interpret r_{i1}=0.9—what it does and does not mean clinically.

(6) You cluster a stroke registry and find three groups that differ mainly by hospital ID in the feature set. What went wrong, and how would you redesign features?

(7) Why does WCSS non-increase when k increases (at global optima for each k), and why then can lower WCSS not prove a larger k is scientifically better?

(8) Propose a validation plan to test whether a two-cluster solution based on admission features replicates at a second comprehensive stroke center.

(9) Contrast k-medoid with k-means when distances are Jaccard on medication sets: which is well-defined and why?

(10) Define Davies–Bouldin and Dunn indices in words; state whether higher or lower is better for each.

(11) Outline how BIRCH’s clustering features enable a single pass over n=5×10^6 registry rows and name one failure mode for non-spherical clusters.

(12) A manuscript names k-means clusters ‘athero’, ‘cardioembolic’, and ‘cryptogenic’ solely from mean feature patterns without vascular workup labels. Draft a five-sentence methods critique.
