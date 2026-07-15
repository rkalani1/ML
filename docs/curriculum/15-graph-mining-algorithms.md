# Chapter 15. Graph Mining Algorithms

## Opening
![Toy patient-similarity graph (original).](../assets/figures/ml_fig_graph_toy.png)

*Toy patient-similarity graph (original).*


A network analysis links hospitals, transfer patterns, and outcome codes. Graph methods can reveal systems structure; they can also launder confounding through edges. Literacy here protects both science and equity claims.


![Graph/embedding geometry can drift across sites (original).](../assets/figures/ml_fig_site_shift.png)

*Graph/embedding geometry can drift across sites (original).*
## Learning Objectives

Define graphs formally and represent them with adjacency matrices and lists.

Compute MSTs with Prim and Kruskal and shortest paths with Dijkstra and A*.

Apply bipartite matching ideas (Hungarian, Hopcroft-Karp) and interpret centrality measures.

Carry out a numerical PageRank iteration and explain HITS hubs and authorities.

Describe community detection with spectral clustering, Louvain, and Leiden.

Explain GNN challenges, message passing, pooling, and GCN/GAT/GraphSAGE; outline HNSW for ANN search.

Map methods to referral networks, comorbidity graphs, connectomics, and epidemiologic contact networks with causal caution.

## 15.1 What Is a Graph?

A graph G = (V, E) consists of a set of nodes (vertices) V and a set of edges E connecting pairs of nodes. Edges may be undirected {i, j} or directed (i -> j). They may be unweighted or carry weights w_ij (travel time, correlation, number of shared patients, synaptic density proxies). Multigraphs allow multiple edges; simple graphs do not. A path is a sequence of distinct nodes joined by edges; a cycle returns to its start; a connected component is a maximal set of nodes mutually reachable when ignoring direction (weak connectivity) or respecting direction (strong connectivity).

Graphs are the natural data type whenever relations—not only feature vectors—carry the signal. In clinical and epidemiologic work those relations include who refers to whom, which diagnoses co-occur, which brain regions connect, which papers cite which concepts, and who contacted whom during an outbreak investigation. Graph mining extracts structure: important nodes, dense communities, missing likely edges, and representations for downstream learning. This chapter builds from classical algorithms (spanning trees, shortest paths, matching, centrality, PageRank/HITS, community detection) to graph neural networks and high-dimensional graph search (HNSW), with clinical interpretations that respect privacy and causal limits.

Nodes: patients, physicians, hospitals, ICD codes, brain ROIs, papers, hosts.

Edges: referrals, co-occurrence, white-matter tracts, citations, contacts.

Attributes: node features (age, specialty), edge features (date, strength).

Tasks: ranking, clustering, matching, link prediction, graph classification, ANN search.

## 15.2 Adjacency and Algebraic Representations

Number nodes 1…n. The adjacency matrix A is n x n with A_ij = 1 (or w_ij) if an edge exists from i to j (convention: rows = sources, columns = targets for directed graphs; A is symmetric for undirected simple graphs). The degree of node i in an undirected graph is d_i = sum_j A_ij. The degree matrix D is diagonal with D_ii = d_i. The combinatorial Laplacian L = D - A appears in spectral clustering; random-walk and symmetric normalized Laplacians are common variants. Powers of A count walks: (A^k)_ij is the number of length-k walks from i to j in an unweighted graph.

Adjacency lists store, for each node, a list of neighbors—memory O(n + m) for m edges, far better than O(n^2) dense matrices when graphs are sparse (as most clinical networks are). Edge lists are convenient for I/O. Choose representation for algorithm and scale: spectral methods like sparse matrix ops on A or L; streaming algorithms may never materialize A.

edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
n = 4
adj = {i: set() for i in range(n)}
for u, v in edges:
adj[u].add(v)
adj[v].add(u)
A = [[0] * n for _ in range(n)]
for u, v in edges:
A[u][v] = A[v][u] = 1
degrees = [sum(row) for row in A]
print(adj, degrees)

## 15.3 Minimum Spanning Trees: Prim and Kruskal

A spanning tree of a connected undirected weighted graph is a subset of edges that connects all vertices without cycles. A minimum spanning tree (MST) minimizes the total edge weight. MSTs appear in network design (cheapest backbone linking hospitals), hierarchical clustering sketches, and approximation algorithms.

![15.1: A minimum spanning tree on a seven-node weighted transfer network, computed with Kruskal's algorithm. Edges are sorted b](../assets/figures/ml_concept_15.1_117485a0.png)

*Figure 15.1 — original teaching graphic.*

Kruskal’s algorithm. Sort all edges by increasing weight. Initialize a forest of single-node trees. Add the next lightest edge if it connects different components (union-find / disjoint set structure); skip if it would form a cycle. Runtime O(m log m) dominated by sorting (or better with sophisticated unions). Correctness follows from the cut property: the lightest edge across any cut is safe for some MST.

Prim’s algorithm. Grow a single tree from a start node. Repeatedly add the lightest edge that connects a tree node to a non-tree node (priority queue of candidate edges). With a binary heap, runtime is O(m log n); with Fibonacci heaps, O(m + n log n). Prim resembles Dijkstra but optimizes edge weight to the tree rather than path length from a source.

Worked Kruskal sketch. Nodes {A,B,C,D}, edges AB:1, BC:2, AC:3, BD:4, CD:5. Add AB (1), BC (2); skip AC (cycle); add BD (4); total weight 1+2+4=7. The MST connects all four nodes. Clinical analogy: linking regional stroke centers with minimum total transfer-agreement “cost” while avoiding redundant cycles in a backbone plan—though real systems optimize multi-criteria objectives beyond a single weight.

# Kruskal with union-find (educational)
def kruskal(n, edges):
# edges: list of (w, u, v)
parent = list(range(n))
def find(x):
while parent[x] != x:
parent[x] = parent[parent[x]]
x = parent[x]
return x
mst, total = [], 0
for w, u, v in sorted(edges):
ru, rv = find(u), find(v)
if ru != rv:
parent[ru] = rv
mst.append((u, v, w))
total += w
return mst, total

## 15.4 Shortest Paths: Dijkstra and A*

Shortest-path problems seek a minimum-weight path from a source s to a target t (or to all nodes). Edge weights must be carefully defined: travel time, risk, or 1 for hop count. Negative weights require Bellman-Ford; Dijkstra requires nonnegative weights.

![15.2: Dijkstra's shortest-path search on the chapter's five-hospital transfer graph, with undirected road-time weights in minu](../assets/figures/ml_concept_15.2_0e83ee90.png)

*Figure 15.2 — original teaching graphic.*

Dijkstra’s algorithm. Maintain tentative distances d(v), initially 0 at s and infinity elsewhere. Use a priority queue to repeatedly extract the node u with smallest d(u) and relax its outgoing edges: if d(u)+w(u,v) < d(v), update d(v) and predecessor. With a binary heap, time is O(m log n). Dijkstra is optimal for nonnegative weights and underpins routing in transfer networks and many map services.

A* search. A* augments Dijkstra with a heuristic h(v) estimating remaining cost to the goal. Nodes are prioritized by f(v)=g(v)+h(v), where g is the cost from the start. If h is admissible (never overestimates) and consistent, A* finds optimal paths while expanding fewer nodes than Dijkstra when the heuristic is informative. Euclidean distance is a classic admissible heuristic for spatial graphs; in hospital routing, estimated ambulance time from population centroids can serve as h if carefully calibrated not to overestimate.

Clinical mapping: prehospital routing to the nearest EVT-capable center is a shortest-path problem on a road network with time-dependent weights; do not confuse network distance with clinical appropriateness when bypass rules depend on last-known-well and severity.

import heapq

def dijkstra(adj, source):
# adj[u] = list of (v, weight)
dist = {source: 0.0}
pq = [(0.0, source)]
while pq:
d, u = heapq.heappop(pq)
if d != dist.get(u):
continue
for v, w in adj[u]:
nd = d + w
if nd < dist.get(v, float(‘inf’)):
dist[v] = nd
heapq.heappush(pq, (nd, v))
return dist

## 15.5 Matching Algorithms: Hungarian and Hopcroft-Karp

Matching problems select edges without shared endpoints. A maximum matching has maximum cardinality; a maximum weight matching maximizes sum of edge weights. Applications include assigning residents to clinics, pairing samples, and bipartite linking of records.

Hungarian algorithm (Kuhn-Munkres). Solves the assignment problem: bipartite matching with costs, finding a perfect matching of minimum total cost (or maximum weight via sign flip) in polynomial time—classically O(n^3) for n x n cost matrices. The algorithm maintains dual variables (labels) and explores equality subgraphs, adjusting labels until a perfect matching appears. For small n (tens of jobs and workers), Hungarian is a standard library call; for large sparse bipartite graphs, other solvers are preferred.

Hopcroft-Karp algorithm. Computes maximum cardinality matching in bipartite graphs in O(E sqrt(V)) time via layered BFS building multiple shortest augmenting paths, then DFS to augment along a maximal set of them. Faster than repeatedly finding one augmenting path. Use when you need maximum number of assignments without weights—e.g., matching as many patients as possible to available tele-neurology slots under bipartite constraints.

Worked assignment intuition. Three patients and three slots with cost matrix rows [2,3,3], [3,2,3], [3,3,2]. The diagonal assignment cost 2+2+2=6 is optimal. Hungarian systematically finds such assignments without brute force n! enumeration.

## 15.6 Centrality Measurement Algorithms

Centrality scores try to quantify importance. Degree centrality is simply d_i (or d_i/(n-1) normalized): nodes with many ties are locally busy. In a hospital referral network, high degree may mark a general neurologist who both sends and receives many consults—or a data artifact of a large clinic.

![15.3: Betweenness centrality on an eight-node referral network of two dense clusters joined by a single bridge. Each node's ar](../assets/figures/ml_concept_15.3_aec87e77.png)

*Figure 15.3 — original teaching graphic.*

Closeness centrality of i is often (n-1) / sum_j dist(i, j) for nodes in a connected component: how many hops on average to everyone else. High closeness means efficient reach across the network.

Betweenness centrality sums, over pairs s, t, the fraction of shortest paths that pass through i—identifying bridges between communities (e.g., a tertiary coordinator linking community EDs to endovascular services). Exact Brandes algorithm is O(n m) on unweighted graphs; large networks need approximation.

![Betweenness on a two-community graph joined by bridge path 3—4—5; Brandes scores (scientific; original).](../assets/figures/ml_fig_betweenness.png)

*Figure — Bridge structure drives betweenness. **Left:** two dense clusters linked by a single path; node size ∝ betweenness (gold = pure bridge). **Right:** Brandes scores rank the bridge and hubs far above leaf nodes. High betweenness marks structural brokerage, not clinical quality—edge definition and incomplete referral capture can invent or erase brokers.*

Eigenvector centrality scores nodes highly when they connect to other high-scoring nodes; it solves A x = lambda x for the leading eigenvector with nonnegative entries (Perron-Frobenius for strongly connected positive cases). Unlike degree, a node with few but elite neighbors can rank high.

Small numerical sketch. Undirected path 1—2—3 plus edge 2—4. Degrees: d1=1, d2=3, d3=1, d4=1. Node 2 is the unique degree center and has high betweenness: shortest paths among {1,3,4} pass through 2. Eigenvector centrality also peaks at 2. No single centrality is “correct”—choose based on the scientific question and report sensitivity to edge definition.

## 15.7 Link Analysis: PageRank (Worked) and HITS

PageRank models a random surfer who, with probability alpha (damping, typically 0.85), follows a random outgoing link, and with probability 1-alpha teleports to a random node. If P is the row-stochastic transition matrix, the PageRank vector r satisfies r = alpha P^T r + (1-alpha) v, where v is a personalization distribution (uniform by default). Damping guarantees a unique solution even if the graph is not strongly connected. Dangling nodes (out-degree 0) need a convention: usually distribute their mass uniformly.

### Worked Example: PageRank on a Tiny 4-Node Directed Graph

Take nodes {A, B, C, D} with directed edges A→B, A→C, B→C, C→A, D→C. Out-degrees are d(A)=2, d(B)=1, d(C)=1, d(D)=1, so no node is dangling (each has at least one out-link). Spreading each node’s out-mass equally over its targets gives the row-stochastic transition matrix P (rows = source), order A, B, C, D:

![15.4: PageRank on the worked four-node directed graph (edges A→B, A→C, B→C, C→A, D→C) with damping α=0.85 and uniform teleport](../assets/figures/ml_concept_15.4_0839aa12.png)

*Figure 15.4 — original teaching graphic.*

Row A: 0, 1/2, 1/2, 0
Row B: 0, 0, 1, 0
Row C: 1, 0, 0, 0
Row D: 0, 0, 1, 0

PageRank pushes mass backward along links, so the iteration uses the transpose P^T (row i of P^T lists who points into i):

Row A: 0, 0, 1, 0 (from C→A)
Row B: 1/2, 0, 0, 0 (from A→B, carrying half of A’s mass)
Row C: 1/2, 1, 0, 1 (from A→C, B→C, D→C)
Row D: 0, 0, 0, 0 (nothing points to D)

The all-zero D-row is diagnostic: no link feeds D, so its only score is teleport. Fix alpha = 0.85 and a uniform teleport v = (1/4, 1/4, 1/4, 1/4); the additive teleport term is (1 - alpha)v = 0.15 x 0.25 = 0.0375 for every node at every step. Iterate r_{k+1} = alpha P^T r_k + (1 - alpha)v starting from r_0 = (0.25, 0.25, 0.25, 0.25).

Iteration 1. Form P^T r_0 entry by entry:
A-entry = 1(0.25) = 0.2500
B-entry = 0.5(0.25) = 0.1250
C-entry = 0.5(0.25) + 1(0.25) + 1(0.25) = 0.6250
D-entry = 0

Scale each by alpha and add 0.0375:
r_1(A) = 0.85(0.2500) + 0.0375 = 0.25000
r_1(B) = 0.85(0.1250) + 0.0375 = 0.14375
r_1(C) = 0.85(0.6250) + 0.0375 = 0.56875
r_1(D) = 0.85(0) + 0.0375 = 0.03750

r_1 = (0.25000, 0.14375, 0.56875, 0.03750), and 0.25000 + 0.14375 + 0.56875 + 0.03750 = 1.00000. Mass is conserved because P is row-stochastic with no dangling nodes: the alpha-scaled flow retains exactly the fraction alpha, and teleport restores the remaining 1 - alpha.

![PageRank power iteration on the chapter four-node digraph with α=0.85 (scientific; original).](../assets/figures/ml_fig_pagerank_iter.png)

*Figure — PageRank as iterative mass flow. **Left:** the directed graph A→B, A→C, B→C, C→A, D→C; node size scales with the converged PageRank π (D is fed only by teleport, so its mass stays near (1−α)/n). **Right:** trajectories of r_k under r ← α Pᵀ r + (1−α)v from a uniform start; mass concentrates on well-linked nodes (especially C and A) within a few iterations. High PageRank ranks link structure—it is not causal importance, clinical priority, or equity of access.*

Iteration 2. P^T r_1:
A-entry = r_1(C) = 0.56875
B-entry = 0.5 r_1(A) = 0.12500
C-entry = 0.5 r_1(A) + r_1(B) + r_1(D) = 0.12500 + 0.14375 + 0.03750 = 0.30625
D-entry = 0

r_2(A) = 0.85(0.56875) + 0.0375 = 0.52094
r_2(B) = 0.85(0.12500) + 0.0375 = 0.14375
r_2(C) = 0.85(0.30625) + 0.0375 = 0.29781
r_2(D) = 0.03750

r_2 = (0.5209, 0.1438, 0.2978, 0.0375), sum = 1.0000.

Iteration 3. Using the unrounded r_2 values in P^T r_2:
A-entry = r_2(C) = 0.29781
B-entry = 0.5 r_2(A) = 0.26047
C-entry = 0.5 r_2(A) + r_2(B) + r_2(D) = 0.26047 + 0.14375 + 0.03750 = 0.44172
D-entry = 0

r_3(A) = 0.85(0.29781) + 0.0375 = 0.29064
r_3(B) = 0.85(0.26047) + 0.0375 = 0.25890
r_3(C) = 0.85(0.44172) + 0.0375 = 0.41296
r_3(D) = 0.03750

r_3 = (0.2906, 0.2589, 0.4130, 0.0375), sum = 1.0000.

Notice A and C trade places: r_1 and r_3 rank C > A, but r_2 ranks A > C. The swing is genuine, not an arithmetic error — the edges A→C and C→A form a length-2 cycle, so the undamped walk is nearly period-2 and damping shrinks the wobble by only a factor alpha per step. The practical lesson: never read final ranks off two or three iterations of a cyclic graph. Iterate until the L1 change ||r_{k+1} - r_k|| falls below a tolerance (say 1e-8), or solve the fixed point directly.

Exact fixed point. Because P^T’s D-row is zero, r(D) = alpha(0) + 0.0375 = 0.0375 exactly at any solution. Solving the linear system (I - alpha P^T) r = (1 - alpha)v (four equations in four unknowns) yields

r* = (0.3725, 0.1958, 0.3941, 0.0375), sum = 1.0000,

so the converged ranking is C > A > B > D. C leads because it is the only triple-target (A, B, and D all point to it) and it recycles mass back to A through C→A; D trails because no in-link ever reaches it and it survives on teleport alone. Running the power-iteration snippet below to convergence reproduces r* to four decimals — implement it and confirm both the ranking and sum(r) = 1.

HITS (Hyperlink-Induced Topic Search). HITS assigns each node an authority score (quality of content pointed to) and a hub score (quality as a pointer). Iteratively, authority a proportional to A^T h, hub h proportional to A a, with normalization. Authorities are nodes many good hubs point to; hubs point to many good authorities. On citation or referral graphs, authorities may be definitive guidelines or comprehensive stroke centers; hubs may be review articles or referring networks. HITS is query-dependent in its original web formulation (run on a subgraph); PageRank is typically global (or personalized).

import numpy as np

# PageRank power iteration on the 4-node example
P = np.array([
[0, 0.5, 0.5, 0],
[0, 0, 1, 0],
[1, 0, 0, 0],
[0, 0, 1, 0],
], dtype=float)
alpha, n = 0.85, 4
r = np.ones(n) / n
v = np.ones(n) / n
for _ in range(100):
r = alpha * P.T @ r + (1 - alpha) * v
print(np.round(r, 4), r.sum())

## 15.8 Community Detection: Spectral, Louvain, and Leiden

Communities are groups of nodes more densely connected internally than externally. Detecting them reveals modules in brain networks, clusters of co-morbid codes, or regional care patterns.

![15.5: Community structure in a twelve-node graph with three modules that are dense internally and sparse between. A modularity](../assets/figures/ml_concept_15.5_56fea22e.png)

*Figure 15.5 — original teaching graphic.*

Spectral clustering. Form Laplacian L (or normalized variant), compute the k eigenvectors with smallest eigenvalues, row-normalize embeddings, and run k-means. The spectral gap suggests the number of clusters. Works well for well-separated communities; costs depend on eigen-solves for large n.

Louvain algorithm. Greedy modularity maximization: modularity Q compares within-community edges to a null model with the same degrees. Louvain repeatedly (1) moves individual nodes to neighbor communities to raise Q, then (2) aggregates communities into super-nodes, iterating until Q stalls. It is fast on large networks and widely used, but can find arbitrarily poorly connected communities in some cases and is resolution-limit sensitive (may miss small communities).

![Modularity Q on a planted 3-community graph (synthetic; original).](../assets/figures/ml_fig_modularity_q.png)

*Figure — Community modularity. **Left:** synthetic graph with three dense modules and sparse between edges. **Right:** modularity \(Q\) for partitions with \(k=1\ldots6\) communities peaks near the true \(k=3\). Max \(Q\) is not a warranty of clinically meaningful modules—billing co-occurrence edges inject administrative artifacts.*

Leiden algorithm. Improves Louvain by adding a refinement phase that splits weakly connected communities, guaranteeing well-connected communities under common settings. Leiden is generally preferred over Louvain for modern practice when available. Always validate communities qualitatively: modularity optima need not equal clinically meaningful modules, and edge definitions (who co-occurs with whom in billing) inject administrative artifacts.

## 15.9 Graph Neural Networks: Challenges, Message Passing, Pooling, Spectral vs Spatial

Graph neural networks (GNNs) learn representations for nodes, edges, or whole graphs by propagating information along structure. Challenges of graphs for neural nets include: variable size and isomorphism (no canonical node order); heterogeneity of node/edge types; scalability to millions of edges; over-smoothing (node features become indistinguishable after many layers); over-squashing (long-range signals bottlenecked through narrow cuts); and distribution shift when deployment graphs differ from training graphs.

![GCN over-smoothing: feature variance and diversity collapse with depth (synthetic; original).](../assets/figures/ml_fig_oversmoothing.png)

*Figure — Over-smoothing as diffusion. Repeated linear message passing \(\hat{A}^L X\) drives node features toward a common signal: mean feature variance falls with depth and pairwise embedding diversity (1 − cosine) collapses. Residual/JK connections, attention (GAT), or simply shallower nets mitigate. More layers ≠ better on small clinical graphs.*

![15.6: One layer of graph neural-network message passing for a center node v. Each neighbor u ∈ N(v) sends a message m = MSG(h_](../assets/figures/ml_concept_15.6_5905fdcb.png)

*Figure 15.6 — original teaching graphic.*

Message passing layer (propagation). A generic layer updates node embeddings h_i by aggregating messages from neighbors:

m_i^{(l)} = AGGREGATE({ MSG(h_i^{(l)}, h_j^{(l)}, e_{ij}) : j in N(i) })

h_i^{(l+1)} = UPDATE(h_i^{(l)}, m_i^{(l)})

AGGREGATE may be sum, mean, max, or attention-weighted sum. Stacking L layers mixes information from L-hop neighborhoods.

Graph pooling coarsens graphs for graph-level prediction: cluster-based pooling (learn assignments), top-k node selection, or hierarchical coarsening. DiffPool and SAGPool are examples; naive global mean/sum pooling is often a strong baseline.

Spectral versus spatial. Spectral methods define convolution via graph Fourier transforms using Laplacian eigenvectors—theoretically neat but expensive and brittle across graphs of different sizes. Spatial methods define convolution as neighborhood aggregation in the vertex domain and dominate practice (GCN, GAT, GraphSAGE).

## 15.10 GCN, GAT, and GraphSAGE

Graph Convolutional Network (GCN). Kipf-style GCN uses a renormalized adjacency:

H^{(l+1)} = sigma( D_hat^{-1/2} A_hat D_hat^{-1/2} H^{(l)} W^{(l)} )

where A_hat = A + I (self-loops) and D_hat is the corresponding degree matrix. Each layer averages neighbor features (including self) then applies a linear map and nonlinearity. GCNs are strong baselines for node classification on citation-like graphs and for some biomedical graphs when labels are at nodes.

Graph Attention Network (GAT). GAT replaces uniform neighbor averaging with learned attention coefficients alpha_{ij}, computing weighted sums of neighbor transformations. Multi-head attention stabilizes training. Attention can highlight influential neighbors (important referring physicians, critical anatomical links) but adds parameters and can overfit small clinical graphs.

GraphSAGE. GraphSAGE samples a fixed-size neighborhood and aggregates with mean, LSTM, or pooling functions, enabling inductive learning on previously unseen nodes by using features. Sampling makes large graphs tractable. For multi-site hospital networks where new facilities appear, inductive models are preferable to purely transductive embeddings tied to a fixed node set.

![GraphSAGE neighborhood sampling fanout and cost vs depth (schematic; original).](../assets/figures/ml_fig_graphsage_sampling.png)

*Figure — Fixed fanout sampling. **Left:** target node samples \(k_1\) neighbors, then each samples \(k_2\)—a two-layer computational graph. **Right:** nodes touched grow roughly \(k+k^2+k^3\) with depth for fanout \(k=5\) (teaching counts). Inductive embeddings help new hospitals appear; graph proximity still is not causation or infection risk without a separate design.*

![Label propagation from few seeds on a similarity graph (synthetic; original).](../assets/figures/ml_fig_label_propagation.png)

*Figure — Semi-supervised graph labels. **Left:** two seeds per class amid unlabeled points. **Right:** soft scores after distance-weighted propagation caricature. Edges encode similarity you chose—not disease transmission. Sensitive to graph construction; not a causal infection model.*

Training tips: use early stopping against over-smoothing; try 2-3 layers before going deep; regularize; evaluate on held-out nodes or temporal splits; beware label leakage through edges constructed with future information.

# One GCN layer (dense educational form)
import numpy as np

def gcn_layer(A, H, W):
A_hat = A + np.eye(A.shape[0])
deg = A_hat.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-8, None)))
support = D_inv_sqrt @ A_hat @ D_inv_sqrt @ H @ W
return np.maximum(support, 0) # ReLU

## 15.11 High-Dimensional Search with Graphs: HNSW

Approximate nearest neighbor (ANN) search finds vectors close to a query in high dimension—central to embedding retrieval, RAG (Chapter 16), and image search. Hierarchical Navigable Small World (HNSW) graphs build multi-layer proximity graphs: upper layers are sparse for long-range greedy routing; the bottom layer is dense for refined search. Insertion connects each new point to M neighbors using a heuristic that maintains small-world navigability; search greedily walks toward the query starting from an entry point at the top layer, descending layers as it goes.

HNSW offers excellent recall-latency trade-offs in practice for embedding dimensions common in NLP and vision (hundreds to a few thousand). Parameters M and efConstruction/efSearch trade memory and recall for speed. Clinical embedding search (similar prior cases, guideline chunks, radiology report neighbors) often uses HNSW or related graph/quantization indexes (IVF-PQ, DiskANN). Always measure recall@k on a labeled neighbor set from your domain; generic benchmark numbers may not transfer to clinical embedding geometries.

## 15.12 Clinical and Epidemiologic Applications

Referral and care networks. Nodes are clinicians or facilities; edges are referrals or transfers. Centrality identifies brokers; communities reveal regional patterns; shortest paths inform access. Privacy: networks of named clinicians are sensitive; aggregate and audit re-identification risk.

Comorbidity graphs. Nodes are diagnoses or medications; edges are co-occurrence or partial correlation after adjustment. Communities suggest multi-morbidity modules. Causal caution: co-occurrence is not causation; billing intensity confounds edges.

Connectomics. Nodes are brain regions; edges are structural or functional connectivity. Graph metrics and GNNs explore disease-related reorganization (stroke disconnection syndromes, epilepsy networks). Reproducibility across scanners and parcellations is a major methodologic challenge; treat single-study “biomarker graphs” skeptically until external validation.

Epidemiologic contact and transmission networks. Matching and path algorithms support outbreak investigation, but missing edges (unreported contacts) dominate error. Prefer methods robust to incomplete observation and integrate with traditional epi models rather than replacing them.

Knowledge graphs for literature and guidelines link entities (diseases, drugs, trials). PageRank/HITS-like scores can surface authoritative nodes; GNN link prediction can suggest missing relations for curation—not for automatic clinical action without review.

Define edges with the same care as define labels—artifacts become communities.

Prefer longitudinal and multi-site validation for GNN claims.

Do not equate high centrality with clinical quality without outcomes data.

HNSW powers retrieval; retrieval quality still depends on embedding training.

### Privacy on Relational Data

Relational data resists the de-identification playbook that works for tabular records, and a neurologist-epidemiologist who publishes clinician or patient graphs must understand why.

Definition and why it is hard. In a table you can suppress or generalize identifiers until each record is indistinguishable from k - 1 others (k-anonymity). A graph leaks through its structure: even after stripping names, a node’s degree, its triangle count, or the shape of its 2-hop neighborhood can be near-unique. An adversary who knows a few edges around a target — a patient’s handful of known contacts, a physician’s known referral partners — can locate that target in a “de-identified” graph and then read off the edges they did not know. This is a structural re-identification attack. Two disclosure targets matter: node disclosure (learning which real entity a node is) and edge disclosure (learning that a specific relationship exists, e.g., that patient X was seen by a named HIV or psychiatric clinician).

Mechanisms for protection. (1) Aggregation and suppression: release community-level counts, degree distributions, or centrality summaries rather than the raw edge list, and suppress small cells. (2) Graph differential privacy: add calibrated noise so the output is nearly unchanged whether or not a given edge (edge-DP) or a given node with all of its edges (node-DP) is present. Edge-DP protects single relationships and is achievable for many statistics (degree sequences, subgraph counts, spectra); node-DP is much stronger but much harder, because one node can touch many edges, so the required noise is large. (3) Synthetic graphs: fit a generative model (a degree-corrected stochastic block model, or a DP graph generator) and release samples, never the real graph. (4) Compute without pooling: federated learning or secure multiparty computation lets sites jointly fit a GNN or compute centrality without any site exporting its patient-level edges.

When to use / when not. Use aggregation or edge-DP when the scientific claim is about population structure (module counts, connectivity gradients, ranking of specialties) rather than named individuals — you rarely need real edges to report that cardioembolic codes cluster apart from small-vessel codes. Reach for node-DP or secure computation when the nodes are people and the graph will leave a trusted enclave. Do not release a “de-identified” clinician referral graph or a patient contact graph as a raw edge list: degree and neighborhood signatures make re-identification straightforward, and the smallest subpopulations (rare diseases, small rural networks) are the most exposed. Pair any release with a re-identification risk audit on the actual data, not a generic assurance.

## 15.13 Synthesis

Classical graph algorithms answer precise structural questions: MSTs for cheap connectors, Dijkstra/A* for routes, Hungarian/Hopcroft-Karp for assignments, centrality and PageRank/HITS for importance, spectral/Louvain/Leiden for modules. GNNs extend learning to relational data via message passing, with GCN, GAT, and GraphSAGE as canonical models. HNSW brings graph ideas to high-dimensional search. For neurologist-epidemiologists, the highest value is often careful graph construction and classical metrics, with deep models reserved for settings with sufficient data, clear inductive tasks, and rigorous external validation.

## 15.14 Worked Dijkstra and A* on a Transfer Graph

Five hospitals: Community A, B, C; Primary Stroke Center P; Comprehensive Center Z. Undirected road-time weights (minutes): A-P 25, B-P 30, C-P 40, P-Z 35, A-Z 70, B-Z 80, C-Z 50, A-B 20, B-C 25. Patient at A needs Z. Dijkstra from A: initialize d(A)=0. Expand A: d(P)=25, d(Z)=70, d(B)=20. Expand B: d(P)=min(25,20+30)=25, d(C)=45, d(Z)=min(70,20+80)=70. Expand P: d(Z)=min(70,25+35)=60. Expand C: d(Z)=min(60,45+50)=60. Shortest A->Z is 60 minutes via A-P-Z (or ties). Routing through C instead costs d(C) + (C-Z) = 45 + 50 = 95, clearly worse — and note there is no direct A-C edge, so A reaches C only via B at cost 20 + 25 = 45.

A* with heuristic h = straight-line lower bound: suppose h(Z)=0, h(P)=30, h(C)=45, h(B)=55, h(A)=50, all admissible if never above true remaining time. f=g+h prioritizes expanding P earlier than exploring long detours toward B, reducing expansions on larger maps. If h overestimates (inadmissible), A* may return suboptimal routes—dangerous when used for clinical logistics recommendations. Always separate routing suggestion tools from clinical eligibility rules (time last known well, severity).

## 15.15 Worked Centrality and Community on a Small Referral Network

Directed referrals among six physicians {1..6}: edges 1->3, 2->3, 3->4, 3->5, 4->6, 5->6, 2->5, 1->4. Node 3 has high in-degree (hub of intermediate consults); node 6 is a sink authority for endovascular care. Betweenness peaks at 3 and possibly 4/5 as bridges. PageRank with damping will lift 6 and 4/5 relative to pure in-degree because importance flows through paths. Louvain on the undirected projection may find community {1,2,3} vs {4,5,6} if edges denser within those sets—interpretable as “front-line assessors” vs “intervention cluster,” but only if edge definitions are clean.

Add a comorbidity undirected graph on ICD nodes with weights as partial correlations after age/sex adjustment. Leiden communities might group cardioembolic-related codes separately from small-vessel codes. Validate against clinical taxonomy; administrative graphs can invent communities of coding convenience (same order set) rather than biology.

## 15.16 GNN Training Recipe and Pitfalls

Practical GNN recipe for node classification on a hospital graph: (1) define nodes/edges without future leakage; (2) split nodes by time or by site, not by random edges only; (3) start with 2-layer GraphSAGE mean aggregator and logistic head; (4) tune learning rate and dropout; (5) early-stop on validation macro-F1; (6) ablate graph vs features-only MLP—if MLP wins, the edges are not helping; (7) check over-smoothing by plotting neighbor embedding cosine vs depth; (8) external site validation.

Pitfalls: edges built from the label (connecting patients who share an outcome) leak; degree features can proxy hospital size and socioeconomic patterns; message passing can amplify majority site styles; explainability is harder than tabular SHAP. For connectomes, site effects and motion artifacts can dominate disease signal—graph metrics need the same harmonization discipline as any imaging biomarker.

HNSW ops note: when using embeddings of clinical text for nearest-neighbor case retrieval, rebuild indexes when the embedding model changes; mismatched spaces silently return nonsense neighbors. Track recall@10 on a hand-labeled similar-case set quarterly.

Ablate features-only baselines before claiming GNN value.

Prefer inductive GraphSAGE when nodes appear over time.

Guard against leakage in edge construction.

Re-evaluate communities and ranks after any edge definition change.

## 15.17 Matching in Operational Neurology

Hungarian assignment can schedule a limited number of same-day EEG slots against ordered studies with priority costs (status epilepticus rule-out gets low cost / high priority). Hopcroft-Karp can maximize the number of paired tele-stroke consult slots to waiting EDs under bipartite capacity constraints when weights are equal. These optimizations are operations research, not disease prediction—they still require fairness constraints (rural EDs not permanently deprioritized) and human override.

Record linkage matching between registry and claims is a different “matching” problem (probabilistic record linkage) sometimes confused with graph matching; use dedicated linkage methods and privacy-preserving join designs. Clarify vocabulary in multi-disciplinary teams.

## 15.18 Prim vs Kruskal Complexity and Implementation Notes

Prim and Kruskal compute the same MST weight on connected undirected graphs with unique edge weights; ties can yield different trees of equal cost. Kruskal shines when edges are easily sorted and the graph is sparse; union-find with path compression and union-by-rank makes cycle checks nearly O(1) amortized. Prim shines when you already have adjacency lists and a binary heap, especially if you stop early for a partial tree covering a subset of critical facilities.

Directed graphs do not have MSTs in the undirected sense; the related optimum branching problems (Edmonds’ algorithm) are beyond our scope but matter for directed transfer networks with one-way constraints. Multigraphs with parallel edges keep only the lightest edge between a pair before MST. Negative weights are allowed for MST (unlike Dijkstra) because there are no path-sum interpretations—only sum of selected edges.

Engineering checklist: validate connectivity first; handle disconnected graphs by computing a minimum spanning forest; store predecessor edges for reconstruction; unit-test on a triangle and on a path where the MST is unique.

## 15.19 HITS Iteration Detail and Comparison to PageRank

Initialize hub and authority vectors to 1 for each node (or 1/n). Repeat: a <- A^T h; h <- A a; normalize a and h (L2 or L1). On the four-node graph of Section 15.7, authorities concentrate on nodes with rich in-links from hubs; hubs concentrate on nodes that point to authorities. Unlike PageRank’s single score and teleport, HITS can surface complementary roles: a referring network (hub) versus a definitive intervention center (authority).

Stability: HITS on the full web without topic subgraphs can be dominated by tightly knit communities unrelated to the query—hence the historical use of query-based base sets. PageRank’s damping is more globally stable. Personalized PageRank (teleport concentrated on a seed set) recovers a middle ground useful for “importance relative to this hospital” rankings.

When publishing centrality results, always state the algorithm, damping/personalization, dangling-node policy, and edge definition. Rankings are not intrinsic properties of physicians independent of data construction.

## 15.20 Spectral Clustering Walk-Through

Given an undirected similarity graph, form the normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}. Compute the eigenvectors u1,…,uk corresponding to the k smallest eigenvalues; form matrix U with those columns; cluster rows of U with k-means. For k=2, the Fiedler vector (second-smallest eigenvalue’s eigenvector) already bipartitions the graph by sign pattern in simple cases.

![Fiedler vector sign cut and Laplacian spectrum on the same bridged graph (scientific; original).](../assets/figures/ml_fig_spectral_fiedler.png)

*Figure — Spectral bipartition intuition. **Left:** nodes colored by sign(u₂); Fiedler coordinates annotate each vertex; the bridge sits near the cut. **Right:** eigenvalues of L = D−A with λ₁≈0 and a spectral gap after λ₂; inset shows the 1-D Fiedler embedding. Communities from the spectrum reflect edge structure only—not causal modules of care quality.*


![Degree tails and assortativity caricature on synthetic graphs (original).](../assets/figures/ml_fig_degree_assort.png)

*Figure — Network structure diagnostics. Left: heavy-tailed degrees (log count). Right: assortative vs disassortative edge endpoints. Structure guides community and centrality analyses; it does not alone prove clinical **causation** without design and confounders.*


![Modularity Q versus number of communities k (synthetic; original).](../assets/figures/ml_fig_modularity_vs_k.png)

*Figure — Resolution choice. Q peaks at a teaching k; other k remain plausible under different scales. Communities are graph summaries—**not disease entities** without external labels and design.*


![Triadic closure dynamics: open triads vs closed triangles (cartoon; original).](../assets/figures/ml_fig_triadic_closure.png)

*Figure — Network evolution cartoon. Closure reduces open triads over time in generative stories. Growth models are not automatic maps of clinical referral **causation**.*


![Shortest path on a grid graph schematic (original).](../assets/figures/ml_fig_shortest_path.png)

*Figure — Path algorithms minimize edge cost on a graph. Without outcome-tied weights, shortest paths are geometry—not clinical priority rankings or causal pathways.*


![Mean random-walk hit times from A along a path graph (synthetic; original).](../assets/figures/ml_fig_hit_times.png)

*Figure — Hit times summarize diffusion distance on a graph. Without outcome-linked weights they are geometry—not care-pathway causation.*


![Ego-network 1-hop schematic (original).](../assets/figures/ml_fig_ego_network.png)

*Figure — Local graph view is structure—not automatic causal neighborhood. Pred != cause without design.*


![Community size distribution (original).](../assets/figures/ml_fig_community_sizes.png)

*Figure — Community sizes are graph summaries. Community size distribution Pred != cause without design.*


![edgebet teaching panel (original).](../assets/figures/ml_fig_edge_between.png)

*Figure — Teaching panel for edgebet. Pred != cause without design.*


![Cycle-34 densify scientific panel 17 (original).](../assets/figures/ml_fig_c34_16.png)

*Figure — Continuous densify panel 17. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 17 (original).](../assets/figures/ml_fig_c35_16.png)

*Figure — Continuous densify panel 17. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 17 (original).](../assets/figures/ml_fig_c36_16.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 17 (original).](../assets/figures/ml_fig_c37_16.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 17 (original).](../assets/figures/ml_fig_c38_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 17 (original).](../assets/figures/ml_fig_c39_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 17 (original).](../assets/figures/ml_fig_c40_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 17 (original).](../assets/figures/ml_fig_c41_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 17 (original).](../assets/figures/ml_fig_c42_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 17 (original).](../assets/figures/ml_fig_c43_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 17 (original).](../assets/figures/ml_fig_c44_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 17 (original).](../assets/figures/ml_fig_c45_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 17 (original).](../assets/figures/ml_fig_c46_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 17 (original).](../assets/figures/ml_fig_c47_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 17 (original).](../assets/figures/ml_fig_c48_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 17 (original).](../assets/figures/ml_fig_c49_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 17 (original).](../assets/figures/ml_fig_c50_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 17 (original).](../assets/figures/ml_fig_c51_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 17 (original).](../assets/figures/ml_fig_c52_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 17 (original).](../assets/figures/ml_fig_c53_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 17 (original).](../assets/figures/ml_fig_c54_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 17 (original).](../assets/figures/ml_fig_c55_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 17 (original).](../assets/figures/ml_fig_c56_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 17 (original).](../assets/figures/ml_fig_c57_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 17 (original).](../assets/figures/ml_fig_c58_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 17 (original).](../assets/figures/ml_fig_c59_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 17 (original).](../assets/figures/ml_fig_c60_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 17 (original).](../assets/figures/ml_fig_c61_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 17 (original).](../assets/figures/ml_fig_c62_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 17 (original).](../assets/figures/ml_fig_c63_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 17 (original).](../assets/figures/ml_fig_c64_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 17 (original).](../assets/figures/ml_fig_c65_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 17 (original).](../assets/figures/ml_fig_c66_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 17 (original).](../assets/figures/ml_fig_c67_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 17 (original).](../assets/figures/ml_fig_c68_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 17 (original).](../assets/figures/ml_fig_c69_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 17 (original).](../assets/figures/ml_fig_c70_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 17 (original).](../assets/figures/ml_fig_c71_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 17 (original).](../assets/figures/ml_fig_c72_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 17 (original).](../assets/figures/ml_fig_c73_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 17 (original).](../assets/figures/ml_fig_c74_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 17 (original).](../assets/figures/ml_fig_c75_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 17 (original).](../assets/figures/ml_fig_c76_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 17 (original).](../assets/figures/ml_fig_c77_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 17 (original).](../assets/figures/ml_fig_c78_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 17 (original).](../assets/figures/ml_fig_c79_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 17 (original).](../assets/figures/ml_fig_c80_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 17 (original).](../assets/figures/ml_fig_c81_16.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

Choosing k: inspect eigenvalue gaps; domain knowledge (expected number of care regions); stability across bootstrap edge subsamples. Spectral methods assume the similarity graph is meaningful—garbage k-NN graphs in high-dimensional noise yield garbage clusters. Compared with Louvain/Leiden, spectral clustering requires choosing k up front and costs eigen-decomposition, but connects cleanly to theoretical graph cuts (RatioCut, NCut).

Clinical connectomes often use partial correlation or streamline counts as weights; thresholding to build A can create or destroy communities. Report sensitivity to threshold and parcellation atlas.

## 15.21 Message Passing Expressive Power and Limitations

1-WL (Weisfeiler-Lehman) tests relate to the discriminative power of standard message-passing GNNs: if two nodes share the same computational tree of neighborhoods, basic MPNNs cannot separate them. Higher-order networks, graph transformers, or feature engineering (cycle counts, encoding identifiers carefully without leakage) may be needed for hard structural tasks. Most clinical node classification tasks are feature-rich (labs, demographics) where 2-3 layer MPNNs suffice if edges help.

Over-squashing: information from distant nodes compressed through narrow bottlenecks fails to arrive. Residual connections, jumping knowledge (concat multi-layer states), and graph rewiring are active research mitigations. For epidemiology contact networks with long chains, shallow models plus classical path analysis may beat deep GNNs.

Pooling for graph classification: global mean pool is a strong baseline; hierarchical pooling must preserve task-relevant subgraphs (e.g., seizure onset zones) rather than arbitrary clusters. Always compare against set models that ignore edges.

## Connections

Graph mining sits at the crossroads of several threads in this book; seeing the links keeps it from feeling like an isolated toolbox.

### Linear algebra and spectral methods

Adjacency and Laplacian matrices turn graph questions into eigenproblems. Eigenvector centrality is the leading eigenvector of A; PageRank is the stationary distribution of a damped Markov chain (the leading eigenvector of the “Google matrix”); spectral clustering partitions using the smallest Laplacian eigenvectors. The same eigen-decomposition machinery behind PCA and kernel methods reappears here — a graph is just another matrix to factor.

### Markov chains and probability

PageRank is a random walk with restart: the damping factor is the restart probability and the solution is a stationary distribution summing to 1. Personalized PageRank, random-walk graph kernels, and the neighbor-sampling of GraphSAGE are all Markov-chain constructions. If you are comfortable with stationary distributions and mixing, you already understand why damping guarantees a unique, stable ranking and why near-periodic graphs oscillate for a few iterations (as in the worked example above).

### Optimization

MST (Prim, Kruskal) is a greedy algorithm justified by matroid theory; the assignment problem (Hungarian) is a linear program whose constraint matrix is totally unimodular, so its optimum is automatically integral; matching and max-flow are duals. Community detection by modularity is a combinatorial objective, and spectral clustering is its continuous relaxation. Graph problems are a compact gallery of the optimization patterns — greedy, LP, relaxation — used throughout applied ML.

### Deep learning and embeddings

GNNs generalize the convolutions used for images from a fixed pixel grid to arbitrary neighborhoods: a 3x3 stencil becomes a variable neighbor set aggregated by message passing, which is why over-smoothing is the graph analogue of an over-deep CNN washing out detail. HNSW ties graph search to the embedding-and-retrieval pipeline: vectors from any model (including GNNs) are indexed as a navigable small-world graph to power the approximate nearest-neighbor lookups behind retrieval-augmented generation (Chapter 16).

### Causal inference and epidemiology

Networks break the independence assumption most models rest on: outcomes spill over along edges (interference), and shared network position confounds associations. Contact graphs feed transmission models; comorbidity graphs are descriptive, not causal. Here graph structure is often the mechanism of confounding and interference you must reason about explicitly, not a nuisance to average away — the same causal humility this book applies to observational tabular data.


![c82 teaching panel 16 (original).](../assets/figures/ml_fig_c82_16.png)
*Figure — Graphs: random-walk exploration vs one-shot message passing. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 16 (original).](../assets/figures/ml_fig_c83_16.png)
*Figure — PageRank mass on a small directed graph. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 16 (original).](../assets/figures/ml_fig_c84_16.png)
*Figure — Community structure: dense within, sparse between modules. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 16 (original).](../assets/figures/ml_fig_c85_16.png)
*Figure — Adjacency matrix view of an undirected graph. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 16 (original).](../assets/figures/ml_fig_c86_16.png)
*Figure — Ego-network: center node and one-hop neighbors. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 16 (original).](../assets/figures/ml_fig_c87_16.png)
*Figure — Heavy-tailed degree histogram in graphs. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 16 (original).](../assets/figures/ml_fig_c88_16.png)
*Figure — Betweenness centrality on a bridge edge. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 16 (original).](../assets/figures/ml_fig_c89_16.png)
*Figure — Modularity Q contribution sketch. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 16 (original).](../assets/figures/ml_fig_c90_16.png)
*Figure — Shortest path tree from source s. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 16 (original).](../assets/figures/ml_fig_c91_16.png)
*Figure — Louvain community agglomeration steps. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 16 (original).](../assets/figures/ml_fig_c92_16.png)
*Figure — Node2vec walk bias p/q. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 16 (original).](../assets/figures/ml_fig_c93_16.png)
*Figure — Graph attention coefficients. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 16 (original).](../assets/figures/ml_fig_c94_16.png)
*Figure — Label propagation on graph. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 16 (original).](../assets/figures/ml_fig_c95_16.png)
*Figure — LINE first/second-order prox. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 16 (original).](../assets/figures/ml_fig_c96_16.png)
*Figure — Heterophily graph caution. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 16 (original).](../assets/figures/ml_fig_c97_16.png)
*Figure — Weisfeiler-Lehman color refine. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 16 (original).](../assets/figures/ml_fig_c98_16.png)
*Figure — GraphSAGE neighbor sample. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 16 (original).](../assets/figures/ml_fig_c99_16.png)
*Figure — Temporal graph edge streams. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 16 (original).](../assets/figures/ml_fig_c100_16.png)
*Figure — Graph rewiring over-smoothing. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 16 (original).](../assets/figures/ml_fig_c101_16.png)
*Figure — SEAL self-adapting loop. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 16 (original).](../assets/figures/ml_fig_c102_16.png)
*Figure — Temporal GNN snapshots. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 16 (original).](../assets/figures/ml_fig_c103_16.png)
*Figure — Directed graph acyclicity check. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 16 (original).](../assets/figures/ml_fig_c104_16.png)
*Figure — PinSAGE bipartite walks. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 16 (original).](../assets/figures/ml_fig_c105_16.png)
*Figure — Dynamic graph snapshots. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 16 (original).](../assets/figures/ml_fig_c106_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 16 (original).](../assets/figures/ml_fig_c107_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 16 (original).](../assets/figures/ml_fig_c108_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 16 (original).](../assets/figures/ml_fig_c109_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 16 (original).](../assets/figures/ml_fig_c110_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 16 (original).](../assets/figures/ml_fig_c111_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 16 (original).](../assets/figures/ml_fig_c112_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 16 (original).](../assets/figures/ml_fig_c113_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 16 (original).](../assets/figures/ml_fig_c114_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 16 (original).](../assets/figures/ml_fig_c115_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 16 (original).](../assets/figures/ml_fig_c116_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 16 (original).](../assets/figures/ml_fig_c117_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 16 (original).](../assets/figures/ml_fig_c118_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 16 (original).](../assets/figures/ml_fig_c119_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 16 (original).](../assets/figures/ml_fig_c120_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 16 (original).](../assets/figures/ml_fig_c121_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 16 (original).](../assets/figures/ml_fig_c122_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 16 (original).](../assets/figures/ml_fig_c123_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 16 (original).](../assets/figures/ml_fig_c124_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 16 (original).](../assets/figures/ml_fig_c125_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 16 (original).](../assets/figures/ml_fig_c126_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 16 (original).](../assets/figures/ml_fig_c127_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 16 (original).](../assets/figures/ml_fig_c128_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 16 (original).](../assets/figures/ml_fig_c129_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 16 (original).](../assets/figures/ml_fig_c130_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 16 (original).](../assets/figures/ml_fig_c131_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 16 (original).](../assets/figures/ml_fig_c132_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 16 (original).](../assets/figures/ml_fig_c133_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 16 (original).](../assets/figures/ml_fig_c134_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 16 (original).](../assets/figures/ml_fig_c135_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 16 (original).](../assets/figures/ml_fig_c136_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 16 (original).](../assets/figures/ml_fig_c137_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 16 (original).](../assets/figures/ml_fig_c138_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 16 (original).](../assets/figures/ml_fig_c139_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 16 (original).](../assets/figures/ml_fig_c140_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 16 (original).](../assets/figures/ml_fig_c141_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 16 (original).](../assets/figures/ml_fig_c142_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 16 (original).](../assets/figures/ml_fig_c143_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 16 (original).](../assets/figures/ml_fig_c144_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 16 (original).](../assets/figures/ml_fig_c145_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 16 (original).](../assets/figures/ml_fig_c146_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 16 (original).](../assets/figures/ml_fig_c147_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 16 (original).](../assets/figures/ml_fig_c148_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 16 (original).](../assets/figures/ml_fig_c149_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 16 (original).](../assets/figures/ml_fig_c150_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 16 (original).](../assets/figures/ml_fig_c151_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 16 (original).](../assets/figures/ml_fig_c152_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 16 (original).](../assets/figures/ml_fig_c153_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 16 (original).](../assets/figures/ml_fig_c154_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 16 (original).](../assets/figures/ml_fig_c155_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 16 (original).](../assets/figures/ml_fig_c156_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 16 (original).](../assets/figures/ml_fig_c157_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 16 (original).](../assets/figures/ml_fig_c158_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 16 (original).](../assets/figures/ml_fig_c159_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 16 (original).](../assets/figures/ml_fig_c160_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 16 (original).](../assets/figures/ml_fig_c161_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 16 (original).](../assets/figures/ml_fig_c162_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 16 (original).](../assets/figures/ml_fig_c163_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 16 (original).](../assets/figures/ml_fig_c164_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 16 (original).](../assets/figures/ml_fig_c165_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 16 (original).](../assets/figures/ml_fig_c166_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 16 (original).](../assets/figures/ml_fig_c167_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 16 (original).](../assets/figures/ml_fig_c168_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 16 (original).](../assets/figures/ml_fig_c169_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 16 (original).](../assets/figures/ml_fig_c170_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 16 (original).](../assets/figures/ml_fig_c171_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 16 (original).](../assets/figures/ml_fig_c172_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 16 (original).](../assets/figures/ml_fig_c173_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 16 (original).](../assets/figures/ml_fig_c174_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 16 (original).](../assets/figures/ml_fig_c175_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 16 (original).](../assets/figures/ml_fig_c176_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 16 (original).](../assets/figures/ml_fig_c177_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 16 (original).](../assets/figures/ml_fig_c178_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 16 (original).](../assets/figures/ml_fig_c179_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 16 (original).](../assets/figures/ml_fig_c180_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 16 (original).](../assets/figures/ml_fig_c181_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 16 (original).](../assets/figures/ml_fig_c182_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 16 (original).](../assets/figures/ml_fig_c183_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 16 (original).](../assets/figures/ml_fig_c184_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 16 (original).](../assets/figures/ml_fig_c185_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 16 (original).](../assets/figures/ml_fig_c186_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 16 (original).](../assets/figures/ml_fig_c187_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 16 (original).](../assets/figures/ml_fig_c188_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 16 (original).](../assets/figures/ml_fig_c189_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 16 (original).](../assets/figures/ml_fig_c190_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 16 (original).](../assets/figures/ml_fig_c191_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 16 (original).](../assets/figures/ml_fig_c192_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 16 (original).](../assets/figures/ml_fig_c193_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 16 (original).](../assets/figures/ml_fig_c194_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 16 (original).](../assets/figures/ml_fig_c195_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 16 (original).](../assets/figures/ml_fig_c196_16.png)
*Figure — Motif counting graphs. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 16 (original).](../assets/figures/ml_fig_c197_16.png)
*Figure — Link prediction scores. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 16 (original).](../assets/figures/ml_fig_c198_16.png)
*Figure — Knowledge graph embeds. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 16 (original).](../assets/figures/ml_fig_c199_16.png)
*Figure — Hyperbolic graph space. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 16 (original).](../assets/figures/ml_fig_c200_16.png)
*Figure — Graph pooling DiffPool. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 16 (original).](../assets/figures/ml_fig_c201_16.png)
*Figure — Graph oversquashing bottleneck. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 16 (original).](../assets/figures/ml_fig_c202_16.png)
*Figure — Graph rewiring long-range links. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 16 (original).](../assets/figures/ml_fig_c203_16.png)
*Figure — PageRank walk stationary mass. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 16 (original).](../assets/figures/ml_fig_c204_16.png)
*Figure — Betweenness hub vs path bridge. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 16 (original).](../assets/figures/ml_fig_c205_16.png)
*Figure — Modularity block adjacency. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 16 (original).](../assets/figures/ml_fig_c206_16.png)
*Figure — Triangle graph motif. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 16 (original).](../assets/figures/ml_fig_c207_16.png)
*Figure — Louvain modularity climb passes. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 16 (original).](../assets/figures/ml_fig_c208_16.png)
*Figure — Motif Z-score randomization null. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 16 (original).](../assets/figures/ml_fig_c209_16.png)
*Figure — Katz path-attenuated centrality. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 16 (original).](../assets/figures/ml_fig_c210_16.png)
*Figure — HITS authority and hub scores. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 16 (original).](../assets/figures/ml_fig_c211_16.png)
*Figure — Graph label propagation seeds. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 16 (original).](../assets/figures/ml_fig_c212_16.png)
*Figure — Core-periphery adjacency block. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 16 (original).](../assets/figures/ml_fig_c213_16.png)
*Figure — Overlapping community sets. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 16 (original).](../assets/figures/ml_fig_c214_16.png)
*Figure — GraphSAGE neighborhood sample. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 16 (original).](../assets/figures/ml_fig_c215_16.png)
*Figure — node2vec biased walk path. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 16 (original).](../assets/figures/ml_fig_c216_16.png)
*Figure — Edge betweenness stroke map. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 16 (original).](../assets/figures/ml_fig_c217_16.png)
*Figure — Triadic closure edge predict. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 16 (original).](../assets/figures/ml_fig_c218_16.png)
*Figure — Structural role similarity heat. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 16 (original).](../assets/figures/ml_fig_c219_16.png)
*Figure — DeepWalk random path. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 16 (original).](../assets/figures/ml_fig_c220_16.png)
*Figure — Infomap description length. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 16 (original).](../assets/figures/ml_fig_c221_16.png)
*Figure — Modularity block community map. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 16 (original).](../assets/figures/ml_fig_c222_16.png)
*Figure — Line graph edge-to-vertex lift. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 16 (original).](../assets/figures/ml_fig_c223_16.png)
*Figure — Motif z-score significance. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 16 (original).](../assets/figures/ml_fig_c224_16.png)
*Figure — PageRank damping teleport. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 16 (original).](../assets/figures/ml_fig_c225_16.png)
*Figure — node2vec biased walk p-q. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 16 (original).](../assets/figures/ml_fig_c226_16.png)
*Figure — Harmonic label propagation path. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 16 (original).](../assets/figures/ml_fig_c227_16.png)
*Figure — Role2vec bridge vs community. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 16 (original).](../assets/figures/ml_fig_c228_16.png)
*Figure — Hyperbolic Poincaré embeddings. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 16 (original).](../assets/figures/ml_fig_c229_16.png)
*Figure — BIGCLAM overlapping affiliations. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 16 (original).](../assets/figures/ml_fig_c230_16.png)
*Figure — Random projection distance map. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 16 (original).](../assets/figures/ml_fig_c231_16.png)
*Figure — Graph attention weight matrix. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 16 (original).](../assets/figures/ml_fig_c232_16.png)
*Figure — metapath2vec typed walk. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 16 (original).](../assets/figures/ml_fig_c233_16.png)
*Figure — Louvain modularity gain path. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 16 (original).](../assets/figures/ml_fig_c234_16.png)
*Figure — Soft label diffusion path. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 16 (original).](../assets/figures/ml_fig_c235_16.png)
*Figure — Leiden refinement gain. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 16 (original).](../assets/figures/ml_fig_c236_16.png)
*Figure — Label smoothing schedule. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 16 (original).](../assets/figures/ml_fig_c237_16.png)
*Figure — Infomap map-equation drop. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 16 (original).](../assets/figures/ml_fig_c238_16.png)
*Figure — Mixup lambda schedule. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 16 (original).](../assets/figures/ml_fig_c239_16.png)
*Figure — Walktrap modularity path. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 16 (original).](../assets/figures/ml_fig_c240_16.png)
*Figure — CutMix region schedule. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 16 (original).](../assets/figures/ml_fig_c241_16.png)
*Figure — Label prop harmonic path. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 16 (original).](../assets/figures/ml_fig_c242_16.png)
*Figure — SpecAugment mask schedule. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 16 (original).](../assets/figures/ml_fig_c243_16.png)
*Figure — Spectral clustering cut path. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 16 (original).](../assets/figures/ml_fig_c244_16.png)
*Figure — RandAugment magnitude sched. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 16 (original).](../assets/figures/ml_fig_c245_16.png)
*Figure — Louvain modularity gain path. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 16 (original).](../assets/figures/ml_fig_c246_16.png)
*Figure — AutoAugment policy sched. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 16 (original).](../assets/figures/ml_fig_c247_16.png)
*Figure — Infomap flow compress path. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 16 (original).](../assets/figures/ml_fig_c248_16.png)
*Figure — TrivialAugment magnitude sched. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 16 (original).](../assets/figures/ml_fig_c249_16.png)
*Figure — Leiden multi-res path. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 16 (original).](../assets/figures/ml_fig_c250_16.png)
*Figure — Cutout region schedule. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 16 (original).](../assets/figures/ml_fig_c251_16.png)
*Figure — Walktrap community path. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 16 (original).](../assets/figures/ml_fig_c252_16.png)
*Figure — Mixup alpha schedule. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 16 (original).](../assets/figures/ml_fig_c253_16.png)
*Figure — Label prop harmonic path. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 16 (original).](../assets/figures/ml_fig_c254_16.png)
*Figure — RandAugment N schedule. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 16 (original).](../assets/figures/ml_fig_c255_16.png)
*Figure — Infomap compress path. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 16 (original).](../assets/figures/ml_fig_c256_16.png)
*Figure — SpecAugment time-mask sched. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 16 (original).](../assets/figures/ml_fig_c257_16.png)
*Figure — Louvain modularity path c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 16 (original).](../assets/figures/ml_fig_c258_16.png)
*Figure — Leiden refine path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 16 (original).](../assets/figures/ml_fig_c259_16.png)
*Figure — Label prop harmonic path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 16 (original).](../assets/figures/ml_fig_c260_16.png)
*Figure — GCN message path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 16 (original).](../assets/figures/ml_fig_c261_16.png)
*Figure — GAT attention heat c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 16 (original).](../assets/figures/ml_fig_c262_16.png)
*Figure — GraphSAGE sample path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 16 (original).](../assets/figures/ml_fig_c263_16.png)
*Figure — Node2vec walk path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 16 (original).](../assets/figures/ml_fig_c264_16.png)
*Figure — Link predict bars c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 16 (original).](../assets/figures/ml_fig_c265_16.png)
*Figure — Betweenness bars c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 16 (original).](../assets/figures/ml_fig_c266_16.png)
*Figure — Community cut path c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 16 (original).](../assets/figures/ml_fig_c267_16.png)
*Figure — Spectral cut path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 16 (original).](../assets/figures/ml_fig_c268_16.png)
*Figure — Motif count bars c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 16 (original).](../assets/figures/ml_fig_c269_16.png)
*Figure — Temporal edge path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 16 (original).](../assets/figures/ml_fig_c270_16.png)
*Figure — Heterograph type path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 16 (original).](../assets/figures/ml_fig_c271_16.png)
*Figure — Graph pool residual c271. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Graph mining extracts structure from relational data. Classical algorithms include minimum spanning trees (Prim, Kruskal), shortest paths (Dijkstra, A*), matching (Hungarian, Hopcroft-Karp), centrality measures, PageRank and HITS link analysis, and community detection (spectral, Louvain, Leiden). Graph neural networks address learning on graphs via message passing, with GCN, GAT, and GraphSAGE as core architectures, facing challenges of over-smoothing, scalability, and shift. HNSW enables fast approximate nearest-neighbor search on embedding graphs. Clinical applications span referral networks, comorbidity and connectomics, and outbreak contact graphs—always with careful edge definition, privacy, and causal humility.

## Practice and Reflection

(1) Run Kruskal by hand on a 5-node complete graph with distinct edge weights of your choosing; verify the MST weight with Prim from two different starts.

(2) Execute Dijkstra on a small directed graph; then design an admissible heuristic and discuss which nodes A* would skip.

(3) Implement the four-node PageRank example; report ranks for alpha in {0.5, 0.85, 0.99} and explain the trend.

(4) Compute degree, closeness, and betweenness on a path of five nodes; which node maximizes each?

(5) Explain over-smoothing in a 10-layer GCN and one mitigation strategy.

(6) Compare Louvain and Leiden: what failure mode of Louvain does Leiden address?

(7) Clinical design: define nodes/edges for a multi-hospital transfer network. Which centrality would you use to find bottleneck coordinators and why?

(8) Why might a comorbidity graph built from billing codes overstate edges in tertiary centers? Propose a normalization.

(9) For the four-node PageRank example, verify the exact fixed point by hand: write the four equations of (I - alpha P^T) r = (1 - alpha)v, solve them, and confirm r(D) = 0.0375 and sum(r) = 1. Explain in one sentence why r(D) never changes across iterations.

(10) Run two HITS iterations on the same four-node graph (initialize hubs and authorities to 1, alternate a <- A^T h and h <- A a with L1 normalization). Compare the top authority to the PageRank winner and explain why the two rankings can disagree.

(11) A colleague proposes publishing a “de-identified” physician referral graph as a raw edge list. Name two structural features that enable re-identification and give one differentially private or aggregated alternative that still answers a population-level question.

(12) In the Dijkstra transfer example, add a new edge P-C of weight 10 and recompute the shortest A->Z path. Does the given A* heuristic remain admissible after this change? Justify by comparing each h-value to the new true remaining cost.
