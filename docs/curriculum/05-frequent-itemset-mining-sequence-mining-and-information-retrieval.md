# Chapter 5. Frequent Itemset Mining, Sequence Mining, and Information Retrieval

## Opening

A claims-analysis team wants frequent co-prescription patterns after TIA and a simple retrieval system for similar prior cases. Itemsets and sequences are powerful; they are also experts at encoding practice fashion as ‘knowledge.’


![Support, confidence, and lift for the chapter’s five-transaction toy basket (original).](../assets/figures/ml_fig_association_rules.png)

*Association-rule metrics from the worked example: confidence is not lift (original).*

![Pattern mining sits on the unsupervised exploration path (original).](../assets/figures/ml_fig_supervised_unsupervised_map.png)

*Pattern mining sits on the unsupervised exploration path (original).*
## Learning Objectives

Define market-basket transactions and compute support, confidence, and lift for association rules with a full numerical example.

Explain information-retrieval concepts: documents, queries, relevance, inverted indexes, and TF–IDF ranking with cosine similarity.

Describe hash tables and MinHash for large-scale set similarity, and justify their use for near-duplicate clinical notes or phenotypes.

Compare tree structures (binary, 2-3, B/B+, red-black, trie/radix) and tree-search methods (BFS, DFS, beam search, MCTS) for indexing and planning.

Explain Bloom filters, sliding windows, and skip lists as memory-efficient structures for streams and approximate membership.

Run and contrast Apriori, FP-Growth, and ECLAT for frequent itemsets; explain GSP, SPADE, FreeSpan, and PrefixSpan for sequences.

Formulate HMMs on care sequences: forward likelihood, Viterbi decoding, and Baum–Welch training; place HMMs in a brief PGM context.

Apply co-occurrence, sequence, and retrieval methods to stroke pathways and evidence search without confusing support with causation.

## From Co-occurrence to Knowledge

Unsupervised learning is not only about partitioning continuous points. Many datasets are discrete collections: items purchased together, events ordered in time, or words appearing in documents. Frequent itemset mining, sequence mining, and information retrieval (IR) extract structure from such collections without requiring class labels. They answer: Which items tend to appear together? Which ordered patterns repeat? Which documents are relevant to a query?

These topics share a combinatorial flavor. The space of possible subsets, subsequences, or term combinations grows explosively with vocabulary size. Effective algorithms combine clever counting with monotonicity properties that allow aggressive pruning, and they rely on classical data structures—hashes, trees, filters, and indexes—to make counting and search feasible at scale. They also share evaluation challenges: a statistically frequent pattern need not be actionable, and a high cosine score need not match a clinician’s true information need.

This chapter builds the mathematical and algorithmic vocabulary needed to apply these methods carefully, with worked examples and clinical–epidemiologic notes throughout. We begin with basic transactional concepts, then cover IR systems and the supporting data structures, then frequent-pattern and sequence algorithms, and finally Hidden Markov Models for sequential prediction under partial observability.

## Basic Concepts: Transactions, Support, Confidence, and Lift

A market-basket (or transactional) dataset is a multiset of transactions. Each transaction t is a subset of a finite item universe I = {i₁, i₂, …, iₘ}. In retail, a transaction is a receipt; items are stock-keeping units. In web logs, a transaction may be the set of pages visited in a session. In clinical informatics, items might be ICD-coded diagnoses, medications, imaging orders, or procedures co-documented on an encounter. Order within a classical basket is ignored: {bread, milk} is the same as {milk, bread}. When order matters, we move to sequence mining later in the chapter.

Let D be a database of n transactions. The support count of an itemset X ⊆ I is the number of transactions that contain X as a subset: count(X) = |{t ∈ D : X ⊆ t}|. Relative support is s(X) = count(X) / n. An itemset is frequent if s(X) ≥ minsup for a user-chosen threshold minsup ∈ (0, 1]. The threshold is a modeling choice: too low yields an explosion of patterns; too high yields only trivial singletons. Domain experts often start near 1%–5% for large retail data and adjust.

Item universe I: all distinct products, tokens, codes, or symbols under study.

Transaction t ⊆ I: one co-occurrence bag (order ignored for itemset mining).

Support s(X): fraction of transactions containing every item in X.

Frequent itemset: s(X) ≥ minsup; the primary output of frequent pattern mining.

Association rules turn frequent itemsets into predictive statements. A rule is X → Y where X and Y are disjoint nonempty itemsets. Confidence of the rule is conf(X → Y) = s(X ∪ Y) / s(X), the empirical conditional probability that Y appears given X. Lift is lift(X → Y) = conf(X → Y) / s(Y) = s(X ∪ Y) / (s(X) s(Y)). Lift equals 1 under independence of X and Y; lift > 1 indicates positive association; lift < 1 indicates negative association. High confidence with lift near 1 can be misleading: if Y is almost always present, any antecedent may appear to “predict” Y.

![Association-rule support–confidence cloud colored by lift (synthetic; original).](../assets/figures/ml_fig_lift_frontier.png)

*Figure — Rule frontier. **Left:** support vs confidence with color = lift; minsup/minconf lines illustrate filtering. **Right:** high lift often lives at low support—interesting but unstable under small n and brutal multiple testing. Mine for hypotheses, not causation; still apply domain judgment after minsup/minconf/minlift filters.*

### Worked Example: Support, Confidence, and Lift

Consider a tiny store with n = 5 transactions over items {A, B, C, D}:
T1: {A, B, C}
T2: {A, B}
T3: {A, C, D}
T4: {B, C}
T5: {A, B, C, D}

![5.1: Support, confidence, and lift for three association rules mined from the five-transaction basket over items {A, B, C, D}](../assets/figures/ml_concept_5.1_a0d90a92.png)

*Figure 5.1 — original teaching graphic.*

Compute support for selected itemsets. count({A}) = 4 (T1–T3, T5), so s(A) = 4/5 = 0.80. count({B}) = 4, s(B) = 0.80. count({C}) = 4, s(C) = 0.80. count({D}) = 2, s(D) = 0.40. For pairs: count({A,B}) = 3, s(A,B) = 0.60; count({A,C}) = 3, s(A,C) = 0.60; count({B,C}) = 3, s(B,C) = 0.60; count({A,B,C}) = 2, s(A,B,C) = 0.40.

Rule A → B has conf(A → B) = s(A,B)/s(A) = 0.60/0.80 = 0.75. Lift(A → B) = 0.75/0.80 = 0.9375 ≈ 0.94. Because lift is slightly below 1, co-occurrence of A and B is slightly less than independence would predict, even though confidence looks decent. For A → D: s(A,D) = 0.40; conf(A → D) = 0.40/0.80 = 0.50; lift = 0.50/0.40 = 1.25. Confidence is lower than A → B, yet lift > 1 indicates a genuine positive association relative to the base rate of D.

| Rule | conf | lift | Teaching note |
|------|------|------|----------------|
| A → B | 0.75 | ≈0.94 | High conf, lift &lt; 1 (base-rate of B) |
| A → D | 0.50 | 1.25 | Lower conf, positive association vs s(D) |
| D → A | 1.00 | 1.25 | Perfect conf on small support of D; lift symmetric |

Three teaching points follow. First, with small n, one transaction swings support sharply. Second, confidence alone is not enough: popular consequents inflate confidence. Third, lift normalizes by the consequent’s base rate and can reverse the ranking of “interesting” rules. In practice one filters by minsup, minconf, and often minlift simultaneously, then still applies domain judgment.

## Information Retrieval: Concepts

Information retrieval systems accept a query q and return a ranked list of documents from a corpus C = {d₁, …, d_N}. Classical vector-space IR represents each document as a vector in a high-dimensional term space. Let V be the vocabulary of terms after tokenization, lowercasing, stopword removal, and optionally stemming or lemmatization. The term–document matrix A is |V| × N (or its transpose), with entry A_{t,d} equal to a weight for term t in document d.

![5.2: Cosine similarity of two document vectors in a two-term space. For d1 = (4, 2) and d2 = (1, 3), cos θ = (d1·d2) / (||d1|](../assets/figures/ml_concept_5.2_b3caf140.png)

*Figure 5.2 — original teaching graphic.*

Relevance is the ideal binary (or graded) judgment that a document satisfies an information need. Systems approximate relevance with scoring functions. Boolean retrieval returns documents that match logical combinations of terms. Ranked retrieval orders all documents by a score s(q, d). Evaluation against human judgments uses precision, recall, average precision (AP), mean average precision (MAP), and graded metrics such as nDCG.

Raw term frequency tf(t, d) counts occurrences of t in d. Rare terms that appear in few documents are more discriminative. Inverse document frequency is idf(t) = log(N / df(t)) (variants add smoothing), where df(t) is the number of documents containing t. The TF–IDF weight is typically w(t, d) = tf(t, d) · idf(t), sometimes with sublinear tf such as 1 + log tf(t, d). Documents and queries become vectors in R^{|V|}; similarity is often cosine similarity: cos(q, d) = (q · d) / (‖q‖ ‖d‖). Cosine is invariant to document length scaling, which helps long documents not dominate purely by having more terms.

### Worked Example: TF–IDF on a Tiny Corpus

Corpus of N = 3 documents (already tokenized):
d1: machine learning models
d2: machine learning data
d3: data mining models
Vocabulary (sorted): data, learning, machine, mining, models. Document frequencies: df(data)=2, df(learning)=2, df(machine)=2, df(mining)=1, df(models)=2. Using natural idf(t) = ln(N/df(t)): idf(data)=ln(3/2)≈0.405, idf(learning)≈0.405, idf(machine)≈0.405, idf(mining)=ln(3/1)≈1.099, idf(models)≈0.405.

Each document has term frequency 1 for its terms. TF–IDF vectors (order: data, learning, machine, mining, models):
d1: (0, 0.405, 0.405, 0, 0.405)
d2: (0.405, 0.405, 0.405, 0, 0)
d3: (0.405, 0, 0, 1.099, 0.405)
Query q = “machine learning” as TF–IDF: q = (0, 0.405, 0.405, 0, 0). ‖q‖ ≈ 0.573. ‖d1‖ ≈ 0.701; q·d1 ≈ 0.328; cos(q,d1) ≈ 0.816. Similarly cos(q,d2) ≈ 0.816; cos(q,d3) = 0. Ranking for q: d1 and d2 tie first, d3 last. The rare term mining in d3 does not help this query; it would dominate a query containing “mining.”

![TF–IDF coordinates and cosine ranking for the chapter three-document corpus (scientific; original).](../assets/figures/ml_fig_tfidf_cosine.png)

*Figure — Vector-space IR on the worked corpus. **Left:** TF–IDF weights for documents d1–d3 and query *q* = “machine learning” over vocabulary {data, learning, machine, mining, models}; rare *mining* receives the largest idf. **Right:** cosine similarity cos(*q*, *d*) = (*q*·*d*)/(‖*q*‖‖*d*‖) ranks d1 and d2 first (≈0.82) and d3 last (0). Cosine is a geometric score for ranking, not a clinical relevance judgment or a causal claim about terms.*

Precision at rank k is the fraction of the top-k returned documents that are relevant. Recall at rank k is the fraction of all relevant documents that appear in the top k. Average precision (AP) for one query integrates precision at each rank where a relevant document is retrieved; MAP averages AP across a query set. Offline IR evaluation requires a labeled test collection; online systems also use click models and A/B tests, which introduce position bias.

### The Inverted Index

Scanning every document for every query is impossible at web or enterprise scale. An inverted index maps each term t to a postings list of document identifiers (and often positions, fields, and payloads) where t occurs. Boolean queries intersect or union postings; ranked retrieval accumulates partial scores while traversing postings, using heuristics such as WAND and block-max indexes to skip low-scoring documents. Compression of postings (gap encoding, variable-byte or SIMD codecs) is essential.

![5.3: Constructing an inverted index from three tokenized documents: each term points to a postings list of document ids, and ](../assets/figures/ml_concept_5.3_66528c0e.png)

*Figure 5.3 — original teaching graphic.*

Dictionary: term → metadata and pointer to postings.

Postings: sorted doc-ids, optional term frequencies and positions.

Query processing: boolean algebra on lists or ranked accumulation of scores.

Skipping and compression: enable sublinear average query time on huge corpora.

## Hash Structures: Hash Tables and MinHash

A hash table maps keys to values using a hash function h that maps keys into a finite array of buckets. Ideal expected lookup, insert, and delete are O(1) when load factors are controlled and collisions are handled by chaining or open addressing. In IR and data mining, hash tables store the term dictionary, accumulate counts for itemsets, implement in-memory inverted indexes for moderate corpora, and back join maps for vertical mining algorithms.

Perfect hashing is rare for dynamic sets; collisions are inevitable. Cryptographic hashes (SHA-family) are overkill for bucket placement but useful for content-addressable storage of documents. Universal hashing families guarantee low expected collision rates independent of adversarial key patterns. In clinical pipelines, patient identifiers and code strings are often hashed for de-identified joins—never use reversible encodings of medical record numbers as “features.”

### Minwise Independent Permutations Hashing (MinHash)

Jaccard similarity between sets A and B is J(A,B) = |A ∩ B| / |A ∪ B|. Exact computation is expensive for large vocabularies or many document pairs. MinHash estimates Jaccard by applying random permutations π of the universe and recording min{π(x) : x ∈ A}. For a family of minwise independent permutations, P(min π(A) = min π(B)) = J(A,B). Repeating k independent hashes yields a signature; the fraction of equal signature coordinates estimates Jaccard.

![MinHash Jaccard estimate vs exact similarity as signature length k grows (scientific; original).](../assets/figures/ml_fig_minhash_jaccard.png)

*Figure — Approximate set similarity. **Left:** mean MinHash Ĵ over trials approaches exact J(A,B) as signature length k increases (error bands shrink). **Right:** one k=20 signature strip—teal cells are matching mins. High Ĵ flags near-duplicates; it is not clinical correctness and does not imply causal relatedness of the underlying patients or notes.*

In practice one uses k hash functions h₁, …, h_k from integers to integers and stores min_{x∈A} h_i(x) for each i. Locality-sensitive hashing (LSH) then buckets signatures so that similar sets collide more often, enabling subquadratic near-duplicate detection. Applications include near-duplicate web pages, plagiarism detection, and finding highly similar problem lists or medication sets across encounters without pairwise O(n²) Jaccard on full sets.

![MinHash LSH banding: candidate probability S-curves and threshold tradeoffs (original).](../assets/figures/ml_fig_minhash_banding.png)

*Figure — Banding controls the candidate explosion. **Left:** \(P(\text{candidate})=1-(1-s^r)^b\) as a function of true Jaccard \(s\) for different band/row pairs \((b,r)\)—S-curves place the steep region near your target similarity. **Right:** raising the Jaccard threshold trades recall of true near-duplicates against false-positive review load. MinHash approximates set overlap; it is not clinical correctness and not a causal graph of notes.*

![Association rules in support–confidence space colored by lift (synthetic; original).](../assets/figures/ml_fig_support_confidence.png)

*Figure — Rule mining plane. Points are candidate rules; dashed lines mark minsup/minconf; color encodes lift (confidence / P(consequent)). High lift with tiny support is often sampling noise. Co-occurrence in orders or notes is association—not a causal pathway.*

![IDF curve: common terms downweighted as document frequency rises (original).](../assets/figures/ml_fig_idf_curve.png)

*Figure — idf(t)=log(N/df(t)) shrinks ubiquitous tokens. TF–IDF is a retrieval weighting—not a clinical importance ranking and not a causal claim about disease terms.*

Clinical and epidemiologic note: MinHash and LSH can cluster near-duplicate discharge summaries or flag copy-forward notes, but similarity is not clinical correctness. Two notes with high Jaccard may both omit a critical negation (“no hemorrhage”). Always combine set-similarity retrieval with section-aware and negation-aware NLP when the use case is safety-critical case finding.

## Tree Data Structures for Search and Indexing

Trees organize ordered keys and hierarchical prefixes so that search, insert, and range queries are efficient. They underpin databases, file systems, and IR dictionaries. Understanding them clarifies why certain mining algorithms and indexes scale, and why others thrash memory.

### Binary Search Trees

A binary search tree (BST) stores keys so that every node’s left subtree holds smaller keys and the right subtree larger keys. Average search is O(log n) for random insertions; worst case degrades to O(n) if keys arrive sorted. Self-balancing variants keep height logarithmic. BSTs support ordered iteration and predecessor/successor queries that hash tables do not—for example, retrieving every lab result recorded between two timestamps from an in-memory ordered index of an encounter’s events.

### 2-3 Search Trees

A 2-3 tree is a balanced search tree where internal nodes have either one key and two children (2-node) or two keys and three children (3-node). All leaves sit at the same depth. Insertions may temporarily create 4-nodes that split and push a middle key upward, preserving balance, so search, insert, and delete all stay O(log n) with no worst-case degradation to a linear chain. 2-3 trees motivate the B-trees used in external storage—for instance, the on-disk index of an ICD-code dictionary that must remain balanced as new codes are added each fiscal year.

### B-Trees and B+ Trees

B-trees generalize multiway balanced trees for disk blocks: each node holds many keys so that tree height—and therefore the number of block reads per lookup—is O(log_b n) for branching factor b, tiny even for huge dictionaries, minimizing disk I/O. A B+ tree stores all records (or postings pointers) at the leaf level and links leaves for efficient range scans. Relational indexes and many inverted-index implementations use B+ trees or LSM-tree variants for on-disk durability. For IR, the dictionary may live in a B+ tree or finite-state transducer while postings sit in compressed contiguous files.

### Red-Black Trees

Red-black trees are binary trees with color invariants that keep the longest path at most twice the shortest, guaranteeing O(log n) operations. They are common in language standard libraries (ordered maps/sets). Relative to AVL trees they rebalance with fewer rotations on average, trading a slightly looser height bound for cheaper updates.

### Tries and Radix Trees

A trie (prefix tree) stores strings by sharing common prefixes: each edge is labeled by a character (or byte), and a path from the root spells a key. Search time is proportional to key length, independent of the number of keys when branching is dense. Radix trees (Patricia tries) compress unary paths into single edges labeled by substrings, reducing memory. Tries excel for autocomplete, IP routing, and token dictionaries. In clinical coding, a trie over ICD or medication strings supports fast prefix lookup during charting or phenotype queries.

![5.4: A trie (prefix tree) over the keys car, care, cart, cat, and dog. Shared prefixes collapse onto shared paths—car, care, ](../assets/figures/ml_concept_5.4_3b48a335.png)

*Figure 5.4 — original teaching graphic.*

## Tree Search Methods: BFS, DFS, Beam Search, and MCTS

Once a problem is cast as a tree or graph of states—candidate itemset lattice, parse tree, treatment plan, or game tree—search algorithms decide which nodes to expand. Breadth-first search (BFS) expands level by level using a queue; it finds shortest paths in unweighted graphs and enumerates itemsets by increasing size in level-wise mining. Depth-first search (DFS) uses a stack (or recursion), exploring a path to a leaf before backtracking; it uses less memory than BFS on deep sparse trees and is natural for recursive pattern-growth methods.

Beam search keeps only the top-b partial hypotheses at each expansion depth according to a scoring heuristic. It is a breadth-limited, greedy compromise: more efficient than full BFS, riskier than exact search. Beam search is ubiquitous in speech recognition, machine translation, and constrained generation; in pathway mining it can prune low-support prefixes early when an approximate ranking of patterns is acceptable.

Monte Carlo Tree Search (MCTS) builds a search tree by repeated simulations. Each iteration selects a leaf using a tree policy (often UCB1 balancing exploitation and exploration), expands a child, simulates a rollout to a terminal reward, and backpropagates the result. MCTS shines in large combinatorial spaces with cheap simulators—games, planning, and some experimental design settings—where exhaustive search is impossible. In clinical decision support research, MCTS-like planning appears in sequential treatment policies when a simulator of outcomes exists; it is not a substitute for trial evidence.

BFS: complete for shortest paths; memory-heavy on wide levels.

DFS: low memory; may dive deep into unproductive branches without heuristics.

Beam search: fixed beam width b; approximate; sensitive to scoring function.

MCTS: simulation-driven; needs a reward model; asymptotically improves with more rollouts.

## Bloom Filters, Sliding Windows, and Skip Lists

### Bloom Filters

A Bloom filter is a probabilistic set-membership structure: it can say “definitely not present” or “possibly present,” never “definitely present” without false positives. It uses a bit array of size m and k independent hash functions. To insert x, set bits h₁(x), …, h_k(x). To query x, check those bits; if any is zero, x is absent; if all are one, x may be present (or a false positive). There are no false negatives for standard Bloom filters. Each insert or query touches exactly k bits, so both run in O(k) time regardless of how many elements have been stored, and space is sublinear in the number of elements for a target false-positive rate.

Applications include web caches (avoid looking up missing keys), distributed databases (skip empty partitions), and streaming analytics. Counting Bloom filters and scalable variants handle deletions and growth. Clinical systems can use Bloom filters to screen whether a hashed patient token might belong to a registry before a privacy-preserving join—still requiring careful privacy review because false positives and side channels matter.

### Sliding Windows

Sliding windows restrict attention to recent data in a stream: the last W events, the last T time units, or a count-based window of the last n items. Fixed-size windows drop the oldest element when a new one arrives. Landmark windows grow from a fixed origin. Tilted or hierarchical time windows keep finer granularity for recent history and coarser aggregates for the past—useful for multi-scale monitoring.

Window aggregates (counts, sums, frequent items in the window) support real-time dashboards: door-to-needle times in the last 24 hours, medication administrations in a sliding shift, or streaming EEG features over a 10-second window. Choice of window width is a bias–variance decision: too short is noisy; too long dilutes acute changes. Align windows with clinical decision cycles (code stroke clock, nursing shift, billing day).

### Skip Lists

A SkipList (skip list) is a probabilistic layered linked list. The bottom layer holds all keys in sorted order; higher layers hold progressively sparser shortcuts. Search starts at the top layer and drops down when the next key would overshoot, achieving expected O(log n) search and update without complex rebalancing rotations. Skip lists are used in some in-memory databases and concurrent data structures because inserts are local. A concrete use is an ordered, concurrently updated in-memory index of streaming timestamped events—live medication-administration records that many writer threads append while dashboards issue range queries over the last shift—where expected O(log n) search and insert are achieved without lock-heavy tree rotations. Conceptually they offer an alternative to balanced trees for ordered dictionaries when simplicity and concurrency matter.

## Frequent Pattern Mining Algorithms: Apriori, FP-Growth, and ECLAT

Enumerating all 2^|I| − 1 nonempty itemsets is impossible for realistic inventories. The Apriori principle (monotonicity of support) states: if X ⊆ Y, then s(X) ≥ s(Y). Equivalently, if an itemset is infrequent, every superset is infrequent. This anti-monotonicity enables level-wise search and pruning across all major frequent-itemset algorithms.

### Apriori

Apriori generates candidates of size k from frequent itemsets of size k−1, prunes any candidate with an infrequent (k−1)-subset, then scans the database to count survivors. Pseudocode sketch:

![5.5: The subset lattice of itemsets over {A, B, C, D}. At minsup = 0.6 on the five-transaction basket, six itemsets are frequ](../assets/figures/ml_concept_5.5_a6a3912b.png)

*Figure 5.5 — original teaching graphic.*

![Apriori support lattice for {A,B,C} on the five-transaction basket with minsup=0.5 (scientific; original).](../assets/figures/ml_fig_apriori_lattice.png)

*Figure — Apriori pruning on the chapter basket (focus {A,B,C}). **Left:** itemset lattice with supports; teal nodes meet minsup = 0.50; the triple ABC (s = 0.40) is infrequent and dashed edges mark the downward closure cut. **Right:** horizontal support bars with the minsup line. Any superset of an infrequent set is infrequent—this is the Apriori principle that prunes candidates before counting. Frequent co-occurrence is not causation or protocol quality.*


![Association rules: confidence vs lift with support-sized bubbles (synthetic; original).](../assets/figures/ml_fig_conf_lift_scatter.png)

*Figure — Rule quality plane. High confidence with lift near 1 often reflects a prevalent item, not a discovery. Bubble size tracks support. **Association ≠ causation**; nest mining inside temporal splits when labels are clinical outcomes.*


![Sparse bag-of-words presence vs TF–IDF toy weights (original).](../assets/figures/ml_fig_tfidf_sparsity.png)

*Figure — Text matrices are sparse. IDF up-weights rare terms; high weight is not automatic clinical importance without a labeled task. Retrieval scores **rank**, they do not establish causation.*


![Precision@k for a synthetic ranked retrieval list (original).](../assets/figures/ml_fig_precision_at_k.png)

*Figure — Retrieval ranking quality as k grows. Always compare to prevalence baselines. High precision@k ranks documents—it does not prove causal importance of terms.*


![BM25-like term weights for a synthetic document (original).](../assets/figures/ml_fig_bm25_terms.png)

*Figure — IR weights up-rank rare informative tokens and down-weight stopwords. Retrieval relevance is not a causal claim about outcomes.*


![Recall@k saturation curve for synthetic retrieval (original).](../assets/figures/ml_fig_recall_at_k.png)

*Figure — Recall@k rises then plateaus as the ranked list lengthens. Pair with precision@k. Retrieval metrics rank content—they do not prove causal importance.*


![Sequence motif toy string ABACABA (original).](../assets/figures/ml_fig_sequence_motif.png)

*Figure — Pattern mining finds strings—not causes. Pred ≠ cause without design.*


![n-gram count decay with order (original).](../assets/figures/ml_fig_ngram_counts.png)

*Figure — Higher n-grams get sparser. n-gram count decay with order Pred != cause without design.*


![inverted teaching panel (original).](../assets/figures/ml_fig_inverted_index.png)

*Figure — Teaching panel for inverted. Pred != cause without design.*


![Cycle-34 densify scientific panel 7 (original).](../assets/figures/ml_fig_c34_06.png)

*Figure — Continuous densify panel 7. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 7 (original).](../assets/figures/ml_fig_c35_06.png)

*Figure — Continuous densify panel 7. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 7 (original).](../assets/figures/ml_fig_c36_06.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 7 (original).](../assets/figures/ml_fig_c37_06.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 7 (original).](../assets/figures/ml_fig_c38_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 7 (original).](../assets/figures/ml_fig_c39_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 7 (original).](../assets/figures/ml_fig_c40_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 7 (original).](../assets/figures/ml_fig_c41_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 7 (original).](../assets/figures/ml_fig_c42_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 7 (original).](../assets/figures/ml_fig_c43_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 7 (original).](../assets/figures/ml_fig_c44_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 7 (original).](../assets/figures/ml_fig_c45_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 7 (original).](../assets/figures/ml_fig_c46_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 7 (original).](../assets/figures/ml_fig_c47_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 7 (original).](../assets/figures/ml_fig_c48_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 7 (original).](../assets/figures/ml_fig_c49_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 7 (original).](../assets/figures/ml_fig_c50_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 7 (original).](../assets/figures/ml_fig_c51_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 7 (original).](../assets/figures/ml_fig_c52_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 7 (original).](../assets/figures/ml_fig_c53_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 7 (original).](../assets/figures/ml_fig_c54_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 7 (original).](../assets/figures/ml_fig_c55_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 7 (original).](../assets/figures/ml_fig_c56_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 7 (original).](../assets/figures/ml_fig_c57_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 7 (original).](../assets/figures/ml_fig_c58_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 7 (original).](../assets/figures/ml_fig_c59_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 7 (original).](../assets/figures/ml_fig_c60_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 7 (original).](../assets/figures/ml_fig_c61_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 7 (original).](../assets/figures/ml_fig_c62_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 7 (original).](../assets/figures/ml_fig_c63_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 7 (original).](../assets/figures/ml_fig_c64_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 7 (original).](../assets/figures/ml_fig_c65_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 7 (original).](../assets/figures/ml_fig_c66_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 7 (original).](../assets/figures/ml_fig_c67_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 7 (original).](../assets/figures/ml_fig_c68_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 7 (original).](../assets/figures/ml_fig_c69_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 7 (original).](../assets/figures/ml_fig_c70_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 7 (original).](../assets/figures/ml_fig_c71_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 7 (original).](../assets/figures/ml_fig_c72_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 7 (original).](../assets/figures/ml_fig_c73_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 7 (original).](../assets/figures/ml_fig_c74_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 7 (original).](../assets/figures/ml_fig_c75_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 7 (original).](../assets/figures/ml_fig_c76_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 7 (original).](../assets/figures/ml_fig_c77_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 7 (original).](../assets/figures/ml_fig_c78_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 7 (original).](../assets/figures/ml_fig_c79_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 7 (original).](../assets/figures/ml_fig_c80_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 7 (original).](../assets/figures/ml_fig_c81_06.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

```
Apriori(D, minsup):
 L1 ← frequent 1-itemsets in D
 k ← 2
 while L_{k−1} is nonempty:
 Ck ← join pairs of L_{k−1} that share first k−2 items
 prune c in Ck if any (k−1)-subset of c ∉ L_{k−1}
 for each transaction t in D: count candidates ⊆ t
 Lk ← {c in Ck : s(c) ≥ minsup}
 k ← k + 1
 return union of all Lk
```

The expensive phase is counting: each pass over D is costly when both |D| and |Ck| are large. Hash trees, bitmaps, and transaction projection reduce constant factors, but multi-scan design remains a bottleneck for dense data with low minsup. After frequent itemsets are known, rules are generated by splitting each frequent Z into X and Y = Z \ X and retaining rules with conf ≥ minconf.

### FP-Growth

Frequent-pattern growth (FP-growth) avoids explicit candidate generation. It compresses the database into a prefix tree (FP-tree) in which common prefixes of frequent items share nodes. Items are ordered by decreasing global frequency so that popular items appear near the root and maximize prefix sharing. Each node stores an item label and a count; header links chain all nodes of the same item.

Mining proceeds recursively by conditional pattern bases. For the least frequent remaining item i, collect all prefix paths ending at nodes labeled i, reweight them by those node counts, build a conditional FP-tree, and recurse. When a conditional tree is a single path, all combinations on that path can be enumerated directly. FP-growth shines when the tree fits in memory and transactions share structure; it struggles when the tree is bushy and dense.

### ECLAT

ECLAT (Equivalence Class Transformation) uses a vertical data layout: for each item (and later each itemset), store the tid-list of transaction identifiers containing it. Support equals the length of the tid-list (or the cardinality of its intersection for multi-item sets). Longer itemsets are formed by intersecting tid-lists of shorter ones within equivalence classes that share a common prefix. Diffsets (differences of tid-lists) can further reduce memory when lists are dense.

Vertical methods often reduce full database scans to a single initial pass plus combinatorial intersections. They are especially effective when transactions are sparse and tid-lists are short. Hybrid systems may use Apriori-style counting for sparse regimes and pattern growth or vertical joins for denser ones. Parallel variants partition transactions or equivalence classes across machines while preserving the same definitions of support and confidence.

## Sequence Mining Algorithms: GSP, SPADE, FreeSpan, and PrefixSpan

A sequence is an ordered list of itemsets (or of atomic events). Customer purchase histories, DNA base pairs, clickstreams, and care pathways are sequences. A sequence α is a subsequence of β if the elements of α appear in order (not necessarily contiguously) inside β. Support of a sequence pattern is the fraction of data sequences that contain it as a subsequence. Sequence mining seeks all patterns with support at least minsup, sometimes with constraints on gaps, windows, or shapes.

### Generalized Sequential Pattern (GSP)

GSP generalizes Apriori to sequences. It performs level-wise candidate generation of k-sequences from frequent (k−1)-sequences, then scans the sequence database to count candidates that occur as subsequences (respecting optional time constraints such as max-gap and min-gap). The Apriori principle still holds for subsequence support: any subsequence of a frequent sequence is frequent. GSP is conceptually clear but can generate many candidates and requires multiple scans.

### SPADE

SPADE (Sequential Pattern Discovery using Equivalence classes) represents data with vertical id-lists: for each event (or itemset), store the list of (sequence-id, timestamp) pairs where it occurs. Longer patterns are formed by joining id-lists of shorter patterns that share a common prefix equivalence class, carefully intersecting temporal order conditions. Vertical joins often reduce database scans to an initial pass plus operations on compact lists.

### FreeSpan

FreeSpan (Frequent pattern-projected Sequential pattern mining) projects the sequence database using frequent items as partitioning keys and grows patterns within projected databases. It reduces candidate testing relative to pure GSP by restricting search to projected subsets. FreeSpan still may generate projected databases that are large; it is an important conceptual bridge to more refined projection methods.

### PrefixSpan

PrefixSpan grows patterns in a horizontal projected-database fashion. Given a frequent prefix α, project the database to the suffixes that follow the first occurrence of α in each sequence. Mine frequent items (or itemsets) in that projected database; each becomes an extension of α. Recurse on the new prefixes. Like FP-growth, PrefixSpan avoids candidate explosion by only exploring extensions that appear in projected data. Pseudo-projection techniques store indices rather than copying suffixes to save memory.

Sequence rules and episode mining add predictive or temporal-window structure. Practitioners must choose whether gaps matter, whether items within the same basket are simultaneous events, and whether closed or maximal patterns should replace the full frequent set to reduce redundancy. Closed sequential patterns preserve support information for all subsequences while emitting fewer results.

## Sequential Prediction with Hidden Markov Models and a PGM Primer

### Probabilistic Graphical Models (Brief Introduction)

Probabilistic graphical models (PGMs) represent joint distributions with graphs: nodes are random variables; edges encode conditional dependence structure. Bayesian networks are directed acyclic graphs factorizing P(X₁,…,Xₙ) = ∏ P(Xᵢ | Pa(Xᵢ)). Markov random fields use undirected graphs. PGMs unify many classical models: naive Bayes, mixture models, Kalman filters, and Hidden Markov Models. Inference (computing marginals or conditionals) and learning (estimating parameters or structure) are the two core computational problems.

### Markov Models and Hidden Markov Models

A first-order Markov chain on discrete states S assumes P(s_t | s_{1:t−1}) = P(s_t | s_{t−1}). Transition matrix A with A_{ij} = P(s_t = j | s_{t−1} = i) and initial distribution π fully specify the process. In an HMM, states are latent; we observe emissions o_t with emission probabilities B_{j}(o) = P(o_t = o | s_t = j). The joint probability of a state path and observation sequence factors as π_{s₁} B_{s₁}(o₁) ∏_{t=2}^T A_{s_{t−1}s_t} B_{s_t}(o_t).

Three canonical HMM problems: (1) Likelihood—compute P(O | θ) for observations O and parameters θ = (π, A, B). (2) Decoding—find the most likely state path argmax_S P(S | O, θ). (3) Learning—estimate θ from data, usually with EM (Baum–Welch) when states are unobserved.

### Forward Algorithm (Likelihood)

Naive summation over all |S|^T paths is exponential. The forward algorithm defines α_t(j) = P(o₁,…,o_t, s_t = j | θ) and recurses: α₁(j) = π_j B_j(o₁), α_{t+1}(j) = [∑_i α_t(i) A_{ij}] B_j(o_{t+1}). Then P(O | θ) = ∑_j α_T(j). Time complexity is O(|S|² T). In practice one works in log-space or uses scaling to avoid underflow.

### Viterbi Algorithm (Decoding)

Viterbi replaces the sum in the forward recursion with a max and stores argmax backpointers: δ₁(j) = π_j B_j(o₁); δ_{t+1}(j) = [max_i δ_t(i) A_{ij}] B_j(o_{t+1}). The best path is recovered by tracing backpointers from argmax_j δ_T(j). Viterbi is dynamic programming for the maximum a posteriori path under the HMM factorization.

### Worked Example: Forward Likelihood and a Viterbi Trace

A tiny two-state HMM makes both algorithms concrete. Hidden states model EEG background: N (interictal/normal) and S (ictal/seizure). Each second a detector emits one symbol, q (quiet) or k (spike). Parameters θ = (π, A, B):

![5.6: Viterbi decoding of the two-state EEG HMM (N = normal, S = seizure) for the observation sequence O = (q, k, k). Each nod](../assets/figures/ml_concept_5.6_395ab0d8.png)

*Figure 5.6 — original teaching graphic.*

Initial: π_N = 0.8, π_S = 0.2.

Transitions: A_NN = 0.7, A_NS = 0.3, A_SN = 0.4, A_SS = 0.6.

Emissions: B_N(q) = 0.9, B_N(k) = 0.1; B_S(q) = 0.3, B_S(k) = 0.7.

Observe O = (q, k, k): one quiet second, then two spikes. The forward pass sums over all paths; α_t(j) is the joint probability of the observations through t and being in state j at t:

```
N (normal) S (seizure)
t=1 (q) α1(N) = 0.8·0.9 = 0.7200 α1(S) = 0.2·0.3 = 0.0600
t=2 (k) α2(N) = (0.72·0.7 + 0.06·0.4)·0.1 α2(S) = (0.72·0.3 + 0.06·0.6)·0.7
 = 0.528·0.1 = 0.0528 = 0.252·0.7 = 0.1764
t=3 (k) α3(N) = (0.0528·0.7 + 0.1764·0.4)·0.1 α3(S) = (0.0528·0.3 + 0.1764·0.6)·0.7
 = 0.10752·0.1 = 0.010752 = 0.12168·0.7 = 0.085176

P(O | θ) = α3(N) + α3(S) = 0.010752 + 0.085176 = 0.095928
```

Viterbi keeps the same recursion but replaces each sum with a max and records which predecessor won (the backpointer); δ_t(j) is the probability of the single best path ending in state j at t:

```
N (normal) S (seizure) argmax pred.
t=1 (q) δ1(N) = 0.8·0.9 = 0.7200 δ1(S) = 0.2·0.3 = 0.0600 — —
t=2 (k) δ2(N) = max(0.72·0.7, 0.06·0.4)·0.1 δ2(S) = max(0.72·0.3, 0.06·0.6)·0.7
 = 0.504·0.1 = 0.0504 = 0.216·0.7 = 0.1512 N→N N→S
t=3 (k) δ3(N) = max(0.0504·0.7, 0.1512·0.4)·0.1 δ3(S) = max(0.0504·0.3, 0.1512·0.6)·0.7
 = 0.06048·0.1 = 0.006048 = 0.09072·0.7 = 0.063504 S→N S→S

Termination: max(δ3(N)=0.006048, δ3(S)=0.063504) = 0.063504 → best final state S
Backtrace: t3 = S ←(S came from S)← t2 = S ←(S came from N)← t1 = N
Best path: (N, S, S)
```

The decoded path is (N, S, S): a normal second, seizure onset at t = 2, seizure sustained at t = 3—the trajectory intuition expects once spikes persist. Two lessons close the example. First, forward and Viterbi answer different questions from the same numbers: forward reports how probable the whole recording is under the model, P(O | θ) ≈ 0.0959, summing all 2³ = 8 paths; Viterbi reports the probability of the single best explanation, 0.063504. Their ratio, 0.063504 / 0.095928 ≈ 0.66, is the posterior probability of the Viterbi path—here the winner carries about two-thirds of the mass, so alternative stagings are not negligible. Second, Viterbi maximizes the joint probability of the entire path; naively concatenating the per-second most-probable state (marginal decoding from forward–backward) can produce a different or even transition-inconsistent sequence. When a coherent trajectory matters—staging a seizure, reconstructing a care pathway—decode with Viterbi, not with independent per-step argmaxes.

### Baum–Welch (Training)

Baum–Welch is EM for HMMs. The E-step uses forward–backward messages to compute expected transition and emission counts under the current parameters. The M-step re-estimates π, A, and B by normalizing those expected counts (with optional Dirichlet-style smoothing). The observed-data likelihood is nondecreasing each iteration but can converge to local maxima; multiple random initializations help. Supervised training with labeled state sequences reduces to counting transitions and emissions.

HMMs model care stages (ED → imaging → treatment → ward) when stages are imperfectly observed from sparse EHR events, or neurologic sequences such as seizure staging from EEG features. Limitations include the strong Markov assumption, sensitivity to emission design, and difficulty with long-range dependencies—motivating later RNN/transformer models while HMMs remain interpretable baselines.

## Clinical and Epidemiologic Notes

Itemset mining, sequence mining, and information retrieval appear throughout clinical informatics and stroke systems research, but they are easy to misinterpret if one confuses co-occurrence with causation or relevance ranking with evidence quality.

A hospital encounter can be treated as a transaction whose items are ICD-coded diagnoses, CPT procedures, administered medications, and imaging orders. Frequent itemsets surface combinations such as {atrial fibrillation, anticoagulation, ischemic stroke}. Support answers how common a combination is; confidence of AF → anticoagulation estimates how often anticoagulation accompanies AF documentation—not whether anticoagulation prevents stroke. Lift adjusts for base rates: a rule predicting “ischemic stroke” as the consequent looks confident in a stroke-service extract simply because stroke is nearly universal there.

If the basket is built from discharge codes finalized after the acute stay, patterns may reflect coding optimization or secondary complications rather than presenting phenotype. Define whether items are restricted to the first 24 hours (hyperacute pathway) or the full admission. Site-specific order sets create artifactual itemsets: a protocol that always co-orders CTA and CTP yields perfect co-occurrence without revealing biology.

Stroke care is sequential: last known well → ED arrival → imaging → thrombolysis or thrombectomy decision → ICU or ward → secondary prevention → rehabilitation. Sequence mining can discover frequent event orders. Patients who die early contribute shorter sequences; mining only survivors invents immortal-time-like bias. Transfer patients may have incomplete pre-arrival sequences. Align sequences on a clinically meaningful origin and report support conditional on still being observable. A further caution: the most frequent pathway describes habit, not quality. A high-support event order—or a high-confidence rule such as {admission} → {specific order set}—reflects what clinicians commonly do at that site, which may encode local convention, workflow constraints, or even guideline-discordant practice. Frequency and confidence quantify how prevalent a pattern is, not how concordant it is with evidence or how it affects outcomes; association is not protocol quality. Benchmark mined pathways against guideline-defined ideal paths and against outcomes before treating any discovered order as a standard.

Computable phenotypes refined by association rules must avoid leakage: do not use phenotype-defining codes as predictors of that same phenotype in prospective models. Multiple testing is severe; require minsup/lift, pre-specify families of interest, and validate on held-out time or external sites. TF–IDF and inverted indexes underwrite literature and note search; evaluate with precision@k and MAP, and apply negation detection so “no hemorrhage” is not retrieved as hemorrhage.

Support/confidence/lift describe co-occurrence, not causal treatment effects.

A frequent or high-confidence pathway reflects common practice, not validated protocol quality; benchmark against guidelines and outcomes.

Align baskets and sequences to index time; avoid post-outcome codes in pathway features.

Do not use phenotype-defining codes as predictors of that same phenotype.

Evaluate clinical search with precision/recall against expert relevance, including negation.

Re-validate frequent patterns and retrieval configurations across sites and eras.

## Putting the Pieces Together

Frequent itemset mining discovers co-occurrence structure in baskets; sequence mining extends that idea to ordered events; IR ranks textual objects by estimated relevance; HMMs add probabilistic prediction under latent states. All depend on counting, sparsity, and clever data structures—hashes, trees, filters, indexes, and dynamic programming tables. All can surface spurious patterns without careful thresholding and evaluation. In clinical and epidemiologic settings, the same tools illuminate care pathways, phenotype candidates, and evidence search—provided index time, base rates, and validation discipline are respected.


![c82 teaching panel 06 (original).](../assets/figures/ml_fig_c82_06.png)
*Figure — Support, confidence, and lift for association rules—filter, then do not claim cause. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 06 (original).](../assets/figures/ml_fig_c83_06.png)
*Figure — Apriori itemset lattice with support pruning. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 06 (original).](../assets/figures/ml_fig_c84_06.png)
*Figure — TF-IDF upweights informative rare terms. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 06 (original).](../assets/figures/ml_fig_c85_06.png)
*Figure — IR pipeline: documents → inverted index → ranked hits. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 06 (original).](../assets/figures/ml_fig_c86_06.png)
*Figure — Sequence mining over ordered event patterns. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 06 (original).](../assets/figures/ml_fig_c87_06.png)
*Figure — Lift versus confidence for association rules. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 06 (original).](../assets/figures/ml_fig_c88_06.png)
*Figure — BM25 score components: TF saturation + IDF. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 06 (original).](../assets/figures/ml_fig_c89_06.png)
*Figure — Query expansion with synonyms/related terms. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 06 (original).](../assets/figures/ml_fig_c90_06.png)
*Figure — Precision@k curve for ranked retrieval. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 06 (original).](../assets/figures/ml_fig_c91_06.png)
*Figure — n-gram language model backoff. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 06 (original).](../assets/figures/ml_fig_c92_06.png)
*Figure — Phrase query proximity bonus. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 06 (original).](../assets/figures/ml_fig_c93_06.png)
*Figure — Learning to rank pairwise hinge. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 06 (original).](../assets/figures/ml_fig_c94_06.png)
*Figure — Sliding window sequence motifs. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 06 (original).](../assets/figures/ml_fig_c95_06.png)
*Figure — Okapi BM25 IDF curve. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 06 (original).](../assets/figures/ml_fig_c96_06.png)
*Figure — Session-based next-item rank. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 06 (original).](../assets/figures/ml_fig_c97_06.png)
*Figure — Skip-gram context window. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 06 (original).](../assets/figures/ml_fig_c98_06.png)
*Figure — Learning-to-rank listwise softmax. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 06 (original).](../assets/figures/ml_fig_c99_06.png)
*Figure — ColBERT late interaction scores. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 06 (original).](../assets/figures/ml_fig_c100_06.png)
*Figure — Attention sink tokens. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 06 (original).](../assets/figures/ml_fig_c101_06.png)
*Figure — Dense retrieval dual encoders. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 06 (original).](../assets/figures/ml_fig_c102_06.png)
*Figure — Hybrid sparse-dense retrieval. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 06 (original).](../assets/figures/ml_fig_c103_06.png)
*Figure — Byte-level BPE vs wordpiece. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 06 (original).](../assets/figures/ml_fig_c104_06.png)
*Figure — Multi-vector retrieval ColBERTv2. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 06 (original).](../assets/figures/ml_fig_c105_06.png)
*Figure — Reranker cross-encoder stack. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 06 (original).](../assets/figures/ml_fig_c106_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 06 (original).](../assets/figures/ml_fig_c107_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 06 (original).](../assets/figures/ml_fig_c108_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 06 (original).](../assets/figures/ml_fig_c109_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 06 (original).](../assets/figures/ml_fig_c110_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 06 (original).](../assets/figures/ml_fig_c111_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 06 (original).](../assets/figures/ml_fig_c112_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 06 (original).](../assets/figures/ml_fig_c113_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 06 (original).](../assets/figures/ml_fig_c114_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 06 (original).](../assets/figures/ml_fig_c115_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 06 (original).](../assets/figures/ml_fig_c116_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 06 (original).](../assets/figures/ml_fig_c117_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 06 (original).](../assets/figures/ml_fig_c118_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 06 (original).](../assets/figures/ml_fig_c119_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 06 (original).](../assets/figures/ml_fig_c120_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 06 (original).](../assets/figures/ml_fig_c121_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 06 (original).](../assets/figures/ml_fig_c122_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 06 (original).](../assets/figures/ml_fig_c123_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 06 (original).](../assets/figures/ml_fig_c124_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 06 (original).](../assets/figures/ml_fig_c125_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 06 (original).](../assets/figures/ml_fig_c126_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 06 (original).](../assets/figures/ml_fig_c127_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 06 (original).](../assets/figures/ml_fig_c128_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 06 (original).](../assets/figures/ml_fig_c129_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 06 (original).](../assets/figures/ml_fig_c130_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 06 (original).](../assets/figures/ml_fig_c131_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 06 (original).](../assets/figures/ml_fig_c132_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 06 (original).](../assets/figures/ml_fig_c133_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 06 (original).](../assets/figures/ml_fig_c134_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 06 (original).](../assets/figures/ml_fig_c135_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 06 (original).](../assets/figures/ml_fig_c136_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 06 (original).](../assets/figures/ml_fig_c137_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 06 (original).](../assets/figures/ml_fig_c138_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 06 (original).](../assets/figures/ml_fig_c139_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 06 (original).](../assets/figures/ml_fig_c140_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 06 (original).](../assets/figures/ml_fig_c141_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 06 (original).](../assets/figures/ml_fig_c142_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 06 (original).](../assets/figures/ml_fig_c143_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 06 (original).](../assets/figures/ml_fig_c144_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 06 (original).](../assets/figures/ml_fig_c145_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 06 (original).](../assets/figures/ml_fig_c146_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 06 (original).](../assets/figures/ml_fig_c147_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 06 (original).](../assets/figures/ml_fig_c148_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 06 (original).](../assets/figures/ml_fig_c149_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 06 (original).](../assets/figures/ml_fig_c150_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 06 (original).](../assets/figures/ml_fig_c151_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 06 (original).](../assets/figures/ml_fig_c152_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 06 (original).](../assets/figures/ml_fig_c153_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 06 (original).](../assets/figures/ml_fig_c154_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 06 (original).](../assets/figures/ml_fig_c155_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 06 (original).](../assets/figures/ml_fig_c156_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 06 (original).](../assets/figures/ml_fig_c157_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 06 (original).](../assets/figures/ml_fig_c158_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 06 (original).](../assets/figures/ml_fig_c159_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 06 (original).](../assets/figures/ml_fig_c160_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 06 (original).](../assets/figures/ml_fig_c161_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 06 (original).](../assets/figures/ml_fig_c162_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 06 (original).](../assets/figures/ml_fig_c163_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 06 (original).](../assets/figures/ml_fig_c164_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 06 (original).](../assets/figures/ml_fig_c165_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 06 (original).](../assets/figures/ml_fig_c166_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 06 (original).](../assets/figures/ml_fig_c167_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 06 (original).](../assets/figures/ml_fig_c168_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 06 (original).](../assets/figures/ml_fig_c169_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 06 (original).](../assets/figures/ml_fig_c170_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 06 (original).](../assets/figures/ml_fig_c171_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 06 (original).](../assets/figures/ml_fig_c172_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 06 (original).](../assets/figures/ml_fig_c173_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 06 (original).](../assets/figures/ml_fig_c174_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 06 (original).](../assets/figures/ml_fig_c175_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 06 (original).](../assets/figures/ml_fig_c176_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 06 (original).](../assets/figures/ml_fig_c177_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 06 (original).](../assets/figures/ml_fig_c178_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 06 (original).](../assets/figures/ml_fig_c179_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 06 (original).](../assets/figures/ml_fig_c180_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 06 (original).](../assets/figures/ml_fig_c181_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 06 (original).](../assets/figures/ml_fig_c182_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 06 (original).](../assets/figures/ml_fig_c183_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 06 (original).](../assets/figures/ml_fig_c184_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 06 (original).](../assets/figures/ml_fig_c185_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 06 (original).](../assets/figures/ml_fig_c186_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 06 (original).](../assets/figures/ml_fig_c187_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 06 (original).](../assets/figures/ml_fig_c188_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 06 (original).](../assets/figures/ml_fig_c189_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 06 (original).](../assets/figures/ml_fig_c190_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 06 (original).](../assets/figures/ml_fig_c191_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 06 (original).](../assets/figures/ml_fig_c192_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 06 (original).](../assets/figures/ml_fig_c193_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 06 (original).](../assets/figures/ml_fig_c194_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 06 (original).](../assets/figures/ml_fig_c195_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 06 (original).](../assets/figures/ml_fig_c196_06.png)
*Figure — SpanBERT span masking. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 06 (original).](../assets/figures/ml_fig_c197_06.png)
*Figure — Dense passage retrieval. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 06 (original).](../assets/figures/ml_fig_c198_06.png)
*Figure — Query likelihood LM. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 06 (original).](../assets/figures/ml_fig_c199_06.png)
*Figure — PRF pseudo relevance. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 06 (original).](../assets/figures/ml_fig_c200_06.png)
*Figure — Learning sparse retrievers. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 06 (original).](../assets/figures/ml_fig_c201_06.png)
*Figure — Support confidence lift bars. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 06 (original).](../assets/figures/ml_fig_c202_06.png)
*Figure — PrefixSpan projection search. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 06 (original).](../assets/figures/ml_fig_c203_06.png)
*Figure — FP-Growth header and tree mine. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 06 (original).](../assets/figures/ml_fig_c204_06.png)
*Figure — BM25 term saturation parts. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 06 (original).](../assets/figures/ml_fig_c205_06.png)
*Figure — Apriori level-wise prune. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 06 (original).](../assets/figures/ml_fig_c206_06.png)
*Figure — TF-IDF cosine query space. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 06 (original).](../assets/figures/ml_fig_c207_06.png)
*Figure — PrefixSpan projected database grow. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 06 (original).](../assets/figures/ml_fig_c208_06.png)
*Figure — ECLAT vertical tidset intersect. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 06 (original).](../assets/figures/ml_fig_c209_06.png)
*Figure — SPADE sequence lattice join. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 06 (original).](../assets/figures/ml_fig_c210_06.png)
*Figure — GSP level-wise sequence candidates. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 06 (original).](../assets/figures/ml_fig_c211_06.png)
*Figure — Closed vs open sequence patterns. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 06 (original).](../assets/figures/ml_fig_c212_06.png)
*Figure — Maxgap sequence constraint. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 06 (original).](../assets/figures/ml_fig_c213_06.png)
*Figure — CloSpan closed sequence mine. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 06 (original).](../assets/figures/ml_fig_c214_06.png)
*Figure — Rule count vs confidence. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 06 (original).](../assets/figures/ml_fig_c215_06.png)
*Figure — Windowed sequence support count. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 06 (original).](../assets/figures/ml_fig_c216_06.png)
*Figure — Sequence motif information bars. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 06 (original).](../assets/figures/ml_fig_c217_06.png)
*Figure — Min-max gap sequence windows. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 06 (original).](../assets/figures/ml_fig_c218_06.png)
*Figure — Sequential pattern mine pipeline. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 06 (original).](../assets/figures/ml_fig_c219_06.png)
*Figure — SPADE id-list temporal join. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 06 (original).](../assets/figures/ml_fig_c220_06.png)
*Figure — Prefix projected grow closed. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 06 (original).](../assets/figures/ml_fig_c221_06.png)
*Figure — GSP candidate join-prune pipeline. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 06 (original).](../assets/figures/ml_fig_c222_06.png)
*Figure — CloSpan closed sequence pipeline. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 06 (original).](../assets/figures/ml_fig_c223_06.png)
*Figure — PrefixSpan projection grow. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 06 (original).](../assets/figures/ml_fig_c224_06.png)
*Figure — SPADE sequence pattern lattice. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 06 (original).](../assets/figures/ml_fig_c225_06.png)
*Figure — Apriori level-wise itemsets. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 06 (original).](../assets/figures/ml_fig_c226_06.png)
*Figure — FreeSpan projected sequence mine. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 06 (original).](../assets/figures/ml_fig_c227_06.png)
*Figure — CHARM closed itemset search. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 06 (original).](../assets/figures/ml_fig_c228_06.png)
*Figure — Sequence graph walk edges. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 06 (original).](../assets/figures/ml_fig_c229_06.png)
*Figure — SPADE sequence growth path. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 06 (original).](../assets/figures/ml_fig_c230_06.png)
*Figure — BIDE closed sequence extend. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 06 (original).](../assets/figures/ml_fig_c231_06.png)
*Figure — Eclat tidset intersections. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 06 (original).](../assets/figures/ml_fig_c232_06.png)
*Figure — CM-Span closed sequence path. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 06 (original).](../assets/figures/ml_fig_c233_06.png)
*Figure — GSP join-prune support path. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 06 (original).](../assets/figures/ml_fig_c234_06.png)
*Figure — PrefixSpan project grow. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 06 (original).](../assets/figures/ml_fig_c235_06.png)
*Figure — Candidate gen lattice path. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 06 (original).](../assets/figures/ml_fig_c236_06.png)
*Figure — CloSpan closed project. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 06 (original).](../assets/figures/ml_fig_c237_06.png)
*Figure — Apriori level-wise path. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 06 (original).](../assets/figures/ml_fig_c238_06.png)
*Figure — BIDE bi-directional path. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 06 (original).](../assets/figures/ml_fig_c239_06.png)
*Figure — Eclat vertical intersect path. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 06 (original).](../assets/figures/ml_fig_c240_06.png)
*Figure — Gap-BIDE sequence path. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 06 (original).](../assets/figures/ml_fig_c241_06.png)
*Figure — FP-growth tree project path. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 06 (original).](../assets/figures/ml_fig_c242_06.png)
*Figure — PrefixSpan projection path. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 06 (original).](../assets/figures/ml_fig_c243_06.png)
*Figure — ECLAT tidset intersect path. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 06 (original).](../assets/figures/ml_fig_c244_06.png)
*Figure — SPADE sequence lattice path. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 06 (original).](../assets/figures/ml_fig_c245_06.png)
*Figure — CHARM closed itemset path. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 06 (original).](../assets/figures/ml_fig_c246_06.png)
*Figure — CM-SPADE multi-dim path. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 06 (original).](../assets/figures/ml_fig_c247_06.png)
*Figure — LCM closed pattern path. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 06 (original).](../assets/figures/ml_fig_c248_06.png)
*Figure — GSP generalized sequence path. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 06 (original).](../assets/figures/ml_fig_c249_06.png)
*Figure — Diffset CHARM closed path. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 06 (original).](../assets/figures/ml_fig_c250_06.png)
*Figure — ClaSP closed sequence path. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 06 (original).](../assets/figures/ml_fig_c251_06.png)
*Figure — dEclat vertical mine path. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 06 (original).](../assets/figures/ml_fig_c252_06.png)
*Figure — GoKrimp sequential path. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 06 (original).](../assets/figures/ml_fig_c253_06.png)
*Figure — LCM-freq closed path. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 06 (original).](../assets/figures/ml_fig_c254_06.png)
*Figure — VMSP maximal sequence path. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 06 (original).](../assets/figures/ml_fig_c255_06.png)
*Figure — Apriori-TID mine path. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 06 (original).](../assets/figures/ml_fig_c256_06.png)
*Figure — SPADE+ sequence path. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 06 (original).](../assets/figures/ml_fig_c257_06.png)
*Figure — Charm closed path c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 06 (original).](../assets/figures/ml_fig_c258_06.png)
*Figure — PrefixSpan project path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 06 (original).](../assets/figures/ml_fig_c259_06.png)
*Figure — SPADE lattice path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 06 (original).](../assets/figures/ml_fig_c260_06.png)
*Figure — MinHash Jaccard path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 06 (original).](../assets/figures/ml_fig_c261_06.png)
*Figure — LSH band collision path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 06 (original).](../assets/figures/ml_fig_c262_06.png)
*Figure — TF-IDF weight bars c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 06 (original).](../assets/figures/ml_fig_c263_06.png)
*Figure — BM25 score bars c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 06 (original).](../assets/figures/ml_fig_c264_06.png)
*Figure — Inverted index path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 06 (original).](../assets/figures/ml_fig_c265_06.png)
*Figure — Query expand path c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 06 (original).](../assets/figures/ml_fig_c266_06.png)
*Figure — N-gram coverage bars c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 06 (original).](../assets/figures/ml_fig_c267_06.png)
*Figure — Sequence gap path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 06 (original).](../assets/figures/ml_fig_c268_06.png)
*Figure — Closed vs max path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 06 (original).](../assets/figures/ml_fig_c269_06.png)
*Figure — Support confidence path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 06 (original).](../assets/figures/ml_fig_c270_06.png)
*Figure — Apriori level-wise path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 06 (original).](../assets/figures/ml_fig_c271_06.png)
*Figure — FP-growth tree path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 06 (original).](../assets/figures/ml_fig_c272_06.png)
*Figure — Eclat tidset path c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 06 (original).](../assets/figures/ml_fig_c273_06.png)
*Figure — Charm closed path c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 06 (original).](../assets/figures/ml_fig_c274_06.png)
*Figure — PrefixSpan project path c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 06 (original).](../assets/figures/ml_fig_c275_06.png)
*Figure — SPADE lattice path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 06 (original).](../assets/figures/ml_fig_c276_06.png)
*Figure — MinHash Jaccard path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 06 (original).](../assets/figures/ml_fig_c277_06.png)
*Figure — LSH band collision path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 06 (original).](../assets/figures/ml_fig_c278_06.png)
*Figure — TF-IDF weight bars c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 06 (original).](../assets/figures/ml_fig_c279_06.png)
*Figure — BM25 score bars c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 06 (original).](../assets/figures/ml_fig_c280_06.png)
*Figure — Inverted index path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 06 (original).](../assets/figures/ml_fig_c281_06.png)
*Figure — Query expand path c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 06 (original).](../assets/figures/ml_fig_c282_06.png)
*Figure — N-gram coverage bars c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 06 (original).](../assets/figures/ml_fig_c283_06.png)
*Figure — Sequence gap path c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 06 (original).](../assets/figures/ml_fig_c284_06.png)
*Figure — Closed vs max path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 06 (original).](../assets/figures/ml_fig_c285_06.png)
*Figure — Support confidence path c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 06 (original).](../assets/figures/ml_fig_c286_06.png)
*Figure — Apriori level-wise path c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 06 (original).](../assets/figures/ml_fig_c287_06.png)
*Figure — FP-growth tree path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 06 (original).](../assets/figures/ml_fig_c288_06.png)
*Figure — Eclat tidset path c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 06 (original).](../assets/figures/ml_fig_c289_06.png)
*Figure — Charm closed path c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 06 (original).](../assets/figures/ml_fig_c290_06.png)
*Figure — PrefixSpan project path c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 06 (original).](../assets/figures/ml_fig_c291_06.png)
*Figure — SPADE lattice path c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 06 (original).](../assets/figures/ml_fig_c292_06.png)
*Figure — MinHash Jaccard path c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 06 (original).](../assets/figures/ml_fig_c293_06.png)
*Figure — LSH band collision path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 06 (original).](../assets/figures/ml_fig_c294_06.png)
*Figure — TF-IDF weight bars c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 06 (original).](../assets/figures/ml_fig_c295_06.png)
*Figure — BM25 score bars c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 06 (original).](../assets/figures/ml_fig_c296_06.png)
*Figure — Inverted index path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 06 (original).](../assets/figures/ml_fig_c297_06.png)
*Figure — Query expand path c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 06 (original).](../assets/figures/ml_fig_c298_06.png)
*Figure — N-gram coverage bars c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 06 (original).](../assets/figures/ml_fig_c299_06.png)
*Figure — Sequence gap path c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 06 (original).](../assets/figures/ml_fig_c300_06.png)
*Figure — Closed vs max path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 06 (original).](../assets/figures/ml_fig_c301_06.png)
*Figure — Support confidence path c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 06 (original).](../assets/figures/ml_fig_c302_06.png)
*Figure — Apriori level-wise path c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 06 (original).](../assets/figures/ml_fig_c303_06.png)
*Figure — FP-growth tree path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 06 (original).](../assets/figures/ml_fig_c304_06.png)
*Figure — Eclat tidset path c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 06 (original).](../assets/figures/ml_fig_c305_06.png)
*Figure — Charm closed path c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 06 (original).](../assets/figures/ml_fig_c306_06.png)
*Figure — PrefixSpan project path c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 06 (original).](../assets/figures/ml_fig_c307_06.png)
*Figure — SPADE lattice path c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 06 (original).](../assets/figures/ml_fig_c308_06.png)
*Figure — MinHash Jaccard path c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 06 (original).](../assets/figures/ml_fig_c309_06.png)
*Figure — LSH band collision path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 06 (original).](../assets/figures/ml_fig_c310_06.png)
*Figure — TF-IDF weight bars c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 06 (original).](../assets/figures/ml_fig_c311_06.png)
*Figure — BM25 score bars c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 06 (original).](../assets/figures/ml_fig_c312_06.png)
*Figure — Inverted index path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 06 (original).](../assets/figures/ml_fig_c313_06.png)
*Figure — Query expand path c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 06 (original).](../assets/figures/ml_fig_c314_06.png)
*Figure — N-gram coverage bars c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 06 (original).](../assets/figures/ml_fig_c315_06.png)
*Figure — Sequence gap path c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 06 (original).](../assets/figures/ml_fig_c316_06.png)
*Figure — Closed vs max path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 06 (original).](../assets/figures/ml_fig_c317_06.png)
*Figure — Support confidence path c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 06 (original).](../assets/figures/ml_fig_c318_06.png)
*Figure — Apriori level-wise path c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 06 (original).](../assets/figures/ml_fig_c319_06.png)
*Figure — FP-growth tree path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 06 (original).](../assets/figures/ml_fig_c320_06.png)
*Figure — Eclat tidset path c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 06 (original).](../assets/figures/ml_fig_c321_06.png)
*Figure — Charm closed path c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 06 (original).](../assets/figures/ml_fig_c322_06.png)
*Figure — PrefixSpan project path c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 06 (original).](../assets/figures/ml_fig_c323_06.png)
*Figure — SPADE lattice path c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 06 (original).](../assets/figures/ml_fig_c324_06.png)
*Figure — MinHash Jaccard path c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 06 (original).](../assets/figures/ml_fig_c325_06.png)
*Figure — LSH band collision path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 06 (original).](../assets/figures/ml_fig_c326_06.png)
*Figure — TF-IDF weight bars c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 06 (original).](../assets/figures/ml_fig_c327_06.png)
*Figure — BM25 score bars c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 06 (original).](../assets/figures/ml_fig_c328_06.png)
*Figure — Inverted index path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 06 (original).](../assets/figures/ml_fig_c329_06.png)
*Figure — Query expand path c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 06 (original).](../assets/figures/ml_fig_c330_06.png)
*Figure — N-gram coverage bars c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 06 (original).](../assets/figures/ml_fig_c331_06.png)
*Figure — Sequence gap path c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 06 (original).](../assets/figures/ml_fig_c332_06.png)
*Figure — Closed vs max path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 06 (original).](../assets/figures/ml_fig_c333_06.png)
*Figure — Support confidence path c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 06 (original).](../assets/figures/ml_fig_c334_06.png)
*Figure — Apriori level-wise path c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 06 (original).](../assets/figures/ml_fig_c335_06.png)
*Figure — FP-growth tree path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 06 (original).](../assets/figures/ml_fig_c336_06.png)
*Figure — Eclat tidset path c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 06 (original).](../assets/figures/ml_fig_c337_06.png)
*Figure — Charm closed path c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 06 (original).](../assets/figures/ml_fig_c338_06.png)
*Figure — PrefixSpan project path c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 06 (original).](../assets/figures/ml_fig_c339_06.png)
*Figure — SPADE lattice path c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 06 (original).](../assets/figures/ml_fig_c340_06.png)
*Figure — MinHash Jaccard path c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 06 (original).](../assets/figures/ml_fig_c341_06.png)
*Figure — LSH band collision path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 06 (original).](../assets/figures/ml_fig_c342_06.png)
*Figure — TF-IDF weight bars c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 06 (original).](../assets/figures/ml_fig_c343_06.png)
*Figure — BM25 score bars c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 06 (original).](../assets/figures/ml_fig_c344_06.png)
*Figure — Inverted index path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 06 (original).](../assets/figures/ml_fig_c345_06.png)
*Figure — Query expand path c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 06 (original).](../assets/figures/ml_fig_c346_06.png)
*Figure — N-gram coverage bars c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 06 (original).](../assets/figures/ml_fig_c347_06.png)
*Figure — Sequence gap path c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 06 (original).](../assets/figures/ml_fig_c348_06.png)
*Figure — Closed vs max path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 06 (original).](../assets/figures/ml_fig_c349_06.png)
*Figure — Support confidence path c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 06 (original).](../assets/figures/ml_fig_c350_06.png)
*Figure — Apriori level-wise path c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 06 (original).](../assets/figures/ml_fig_c351_06.png)
*Figure — FP-growth tree path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 06 (original).](../assets/figures/ml_fig_c352_06.png)
*Figure — Eclat tidset path c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 06 (original).](../assets/figures/ml_fig_c353_06.png)
*Figure — Charm closed path c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 06 (original).](../assets/figures/ml_fig_c354_06.png)
*Figure — PrefixSpan project path c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 06 (original).](../assets/figures/ml_fig_c355_06.png)
*Figure — SPADE lattice path c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 06 (original).](../assets/figures/ml_fig_c356_06.png)
*Figure — MinHash Jaccard path c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 06 (original).](../assets/figures/ml_fig_c357_06.png)
*Figure — LSH band collision path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 06 (original).](../assets/figures/ml_fig_c358_06.png)
*Figure — TF-IDF weight bars c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 06 (original).](../assets/figures/ml_fig_c359_06.png)
*Figure — BM25 score bars c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 06 (original).](../assets/figures/ml_fig_c360_06.png)
*Figure — Inverted index path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 06 (original).](../assets/figures/ml_fig_c361_06.png)
*Figure — Query expand path c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 06 (original).](../assets/figures/ml_fig_c362_06.png)
*Figure — N-gram coverage bars c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 06 (original).](../assets/figures/ml_fig_c363_06.png)
*Figure — Sequence gap path c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 06 (original).](../assets/figures/ml_fig_c364_06.png)
*Figure — Closed vs max path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 06 (original).](../assets/figures/ml_fig_c365_06.png)
*Figure — Support confidence path c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 06 (original).](../assets/figures/ml_fig_c366_06.png)
*Figure — Apriori level-wise path c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 06 (original).](../assets/figures/ml_fig_c367_06.png)
*Figure — FP-growth tree path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 06 (original).](../assets/figures/ml_fig_c368_06.png)
*Figure — Eclat tidset path c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 06 (original).](../assets/figures/ml_fig_c369_06.png)
*Figure — Charm closed path c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 06 (original).](../assets/figures/ml_fig_c370_06.png)
*Figure — PrefixSpan project path c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 06 (original).](../assets/figures/ml_fig_c371_06.png)
*Figure — SPADE lattice path c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 06 (original).](../assets/figures/ml_fig_c372_06.png)
*Figure — MinHash Jaccard path c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 06 (original).](../assets/figures/ml_fig_c373_06.png)
*Figure — LSH band collision path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 06 (original).](../assets/figures/ml_fig_c374_06.png)
*Figure — TF-IDF weight bars c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 06 (original).](../assets/figures/ml_fig_c375_06.png)
*Figure — BM25 score bars c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 06 (original).](../assets/figures/ml_fig_c376_06.png)
*Figure — Inverted index path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 06 (original).](../assets/figures/ml_fig_c377_06.png)
*Figure — Query expand path c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 06 (original).](../assets/figures/ml_fig_c378_06.png)
*Figure — N-gram coverage bars c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 06 (original).](../assets/figures/ml_fig_c379_06.png)
*Figure — Sequence gap path c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 06 (original).](../assets/figures/ml_fig_c380_06.png)
*Figure — Closed vs max path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 06 (original).](../assets/figures/ml_fig_c381_06.png)
*Figure — Support confidence path c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 06 (original).](../assets/figures/ml_fig_c382_06.png)
*Figure — Apriori level-wise path c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 06 (original).](../assets/figures/ml_fig_c383_06.png)
*Figure — FP-growth tree path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 06 (original).](../assets/figures/ml_fig_c384_06.png)
*Figure — Eclat tidset path c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 06 (original).](../assets/figures/ml_fig_c385_06.png)
*Figure — Charm closed path c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 06 (original).](../assets/figures/ml_fig_c386_06.png)
*Figure — PrefixSpan project path c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 06 (original).](../assets/figures/ml_fig_c387_06.png)
*Figure — SPADE lattice path c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 06 (original).](../assets/figures/ml_fig_c388_06.png)
*Figure — MinHash Jaccard path c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 06 (original).](../assets/figures/ml_fig_c389_06.png)
*Figure — LSH band collision path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 06 (original).](../assets/figures/ml_fig_c390_06.png)
*Figure — TF-IDF weight bars c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 06 (original).](../assets/figures/ml_fig_c391_06.png)
*Figure — BM25 score bars c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 06 (original).](../assets/figures/ml_fig_c392_06.png)
*Figure — Inverted index path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 06 (original).](../assets/figures/ml_fig_c393_06.png)
*Figure — Query expand path c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 06 (original).](../assets/figures/ml_fig_c394_06.png)
*Figure — N-gram coverage bars c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 06 (original).](../assets/figures/ml_fig_c395_06.png)
*Figure — Sequence gap path c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 06 (original).](../assets/figures/ml_fig_c396_06.png)
*Figure — Closed vs max path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 06 (original).](../assets/figures/ml_fig_c397_06.png)
*Figure — Support confidence path c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 06 (original).](../assets/figures/ml_fig_c398_06.png)
*Figure — Apriori level-wise path c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 06 (original).](../assets/figures/ml_fig_c399_06.png)
*Figure — FP-growth tree path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 06 (original).](../assets/figures/ml_fig_c400_06.png)
*Figure — Eclat tidset path c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 06 (original).](../assets/figures/ml_fig_c401_06.png)
*Figure — Charm closed path c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 06 (original).](../assets/figures/ml_fig_c402_06.png)
*Figure — PrefixSpan project path c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 06 (original).](../assets/figures/ml_fig_c403_06.png)
*Figure — SPADE lattice path c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 06 (original).](../assets/figures/ml_fig_c404_06.png)
*Figure — MinHash Jaccard path c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 06 (original).](../assets/figures/ml_fig_c405_06.png)
*Figure — LSH band collision path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 06 (original).](../assets/figures/ml_fig_c406_06.png)
*Figure — TF-IDF weight bars c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 06 (original).](../assets/figures/ml_fig_c407_06.png)
*Figure — BM25 score bars c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 06 (original).](../assets/figures/ml_fig_c408_06.png)
*Figure — Inverted index path c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 06 (original).](../assets/figures/ml_fig_c409_06.png)
*Figure — Query expand path c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 06 (original).](../assets/figures/ml_fig_c410_06.png)
*Figure — N-gram coverage bars c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 06 (original).](../assets/figures/ml_fig_c411_06.png)
*Figure — Sequence gap path c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 06 (original).](../assets/figures/ml_fig_c412_06.png)
*Figure — Closed vs max path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 06 (original).](../assets/figures/ml_fig_c413_06.png)
*Figure — Support confidence path c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 06 (original).](../assets/figures/ml_fig_c414_06.png)
*Figure — Apriori level-wise path c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 06 (original).](../assets/figures/ml_fig_c415_06.png)
*Figure — FP-growth tree path c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 06 (original).](../assets/figures/ml_fig_c416_06.png)
*Figure — Eclat tidset path c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 06 (original).](../assets/figures/ml_fig_c417_06.png)
*Figure — Charm closed path c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 06 (original).](../assets/figures/ml_fig_c418_06.png)
*Figure — PrefixSpan project path c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 06 (original).](../assets/figures/ml_fig_c419_06.png)
*Figure — SPADE lattice path c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 06 (original).](../assets/figures/ml_fig_c420_06.png)
*Figure — MinHash Jaccard path c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 06 (original).](../assets/figures/ml_fig_c421_06.png)
*Figure — LSH band collision path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 06 (original).](../assets/figures/ml_fig_c422_06.png)
*Figure — TF-IDF weight bars c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 06 (original).](../assets/figures/ml_fig_c423_06.png)
*Figure — BM25 score bars c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 06 (original).](../assets/figures/ml_fig_c424_06.png)
*Figure — Inverted index path c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 06 (original).](../assets/figures/ml_fig_c425_06.png)
*Figure — Query expand path c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 06 (original).](../assets/figures/ml_fig_c426_06.png)
*Figure — N-gram coverage bars c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 06 (original).](../assets/figures/ml_fig_c427_06.png)
*Figure — Sequence gap path c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 06 (original).](../assets/figures/ml_fig_c428_06.png)
*Figure — Closed vs max path c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 06 (original).](../assets/figures/ml_fig_c429_06.png)
*Figure — Support confidence path c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 06 (original).](../assets/figures/ml_fig_c430_06.png)
*Figure — Apriori level-wise path c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 06 (original).](../assets/figures/ml_fig_c431_06.png)
*Figure — FP-growth tree path c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 06 (original).](../assets/figures/ml_fig_c432_06.png)
*Figure — Eclat tidset path c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 06 (original).](../assets/figures/ml_fig_c433_06.png)
*Figure — Charm closed path c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 06 (original).](../assets/figures/ml_fig_c434_06.png)
*Figure — PrefixSpan project path c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 06 (original).](../assets/figures/ml_fig_c435_06.png)
*Figure — SPADE lattice path c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 06 (original).](../assets/figures/ml_fig_c436_06.png)
*Figure — MinHash Jaccard path c436. Synthetic teaching geometry—not a causal claim.*


![c437 teaching panel 06 (original).](../assets/figures/ml_fig_c437_06.png)
*Figure — LSH band collision path c437. Synthetic teaching geometry—not a causal claim.*


![c438 teaching panel 06 (original).](../assets/figures/ml_fig_c438_06.png)
*Figure — TF-IDF weight bars c438. Synthetic teaching geometry—not a causal claim.*


![c439 teaching panel 06 (original).](../assets/figures/ml_fig_c439_06.png)
*Figure — BM25 score bars c439. Synthetic teaching geometry—not a causal claim.*


![c440 teaching panel 06 (original).](../assets/figures/ml_fig_c440_06.png)
*Figure — Inverted index path c440. Synthetic teaching geometry—not a causal claim.*


![c441 teaching panel 06 (original).](../assets/figures/ml_fig_c441_06.png)
*Figure — Query expand path c441. Synthetic teaching geometry—not a causal claim.*


![c442 teaching panel 06 (original).](../assets/figures/ml_fig_c442_06.png)
*Figure — N-gram coverage bars c442. Synthetic teaching geometry—not a causal claim.*


![c443 teaching panel 06 (original).](../assets/figures/ml_fig_c443_06.png)
*Figure — Sequence gap path c443. Synthetic teaching geometry—not a causal claim.*


![c444 teaching panel 06 (original).](../assets/figures/ml_fig_c444_06.png)
*Figure — Closed vs max path c444. Synthetic teaching geometry—not a causal claim.*


![c445 teaching panel 06 (original).](../assets/figures/ml_fig_c445_06.png)
*Figure — Support confidence path c445. Synthetic teaching geometry—not a causal claim.*


![c446 teaching panel 06 (original).](../assets/figures/ml_fig_c446_06.png)
*Figure — Apriori level-wise path c446. Synthetic teaching geometry—not a causal claim.*


![c447 teaching panel 06 (original).](../assets/figures/ml_fig_c447_06.png)
*Figure — FP-growth tree path c447. Synthetic teaching geometry—not a causal claim.*


![c448 teaching panel 06 (original).](../assets/figures/ml_fig_c448_06.png)
*Figure — Eclat tidset path c448. Synthetic teaching geometry—not a causal claim.*


![c449 teaching panel 06 (original).](../assets/figures/ml_fig_c449_06.png)
*Figure — Charm closed path c449. Synthetic teaching geometry—not a causal claim.*


![c450 teaching panel 06 (original).](../assets/figures/ml_fig_c450_06.png)
*Figure — PrefixSpan project path c450. Synthetic teaching geometry—not a causal claim.*


![c451 teaching panel 06 (original).](../assets/figures/ml_fig_c451_06.png)
*Figure — SPADE lattice path c451. Synthetic teaching geometry—not a causal claim.*


![c452 teaching panel 06 (original).](../assets/figures/ml_fig_c452_06.png)
*Figure — MinHash Jaccard path c452. Synthetic teaching geometry—not a causal claim.*


![c453 teaching panel 06 (original).](../assets/figures/ml_fig_c453_06.png)
*Figure — LSH band collision path c453. Synthetic teaching geometry—not a causal claim.*


![c454 teaching panel 06 (original).](../assets/figures/ml_fig_c454_06.png)
*Figure — TF-IDF weight bars c454. Synthetic teaching geometry—not a causal claim.*


![c455 teaching panel 06 (original).](../assets/figures/ml_fig_c455_06.png)
*Figure — BM25 score bars c455. Synthetic teaching geometry—not a causal claim.*


![c456 teaching panel 06 (original).](../assets/figures/ml_fig_c456_06.png)
*Figure — Inverted index path c456. Synthetic teaching geometry—not a causal claim.*


![c457 teaching panel 06 (original).](../assets/figures/ml_fig_c457_06.png)
*Figure — Query expand path c457. Synthetic teaching geometry—not a causal claim.*


![c458 teaching panel 06 (original).](../assets/figures/ml_fig_c458_06.png)
*Figure — N-gram coverage bars c458. Synthetic teaching geometry—not a causal claim.*


![c459 teaching panel 06 (original).](../assets/figures/ml_fig_c459_06.png)
*Figure — Sequence gap path c459. Synthetic teaching geometry—not a causal claim.*


![c460 teaching panel 06 (original).](../assets/figures/ml_fig_c460_06.png)
*Figure — Closed vs max path c460. Synthetic teaching geometry—not a causal claim.*


![c461 teaching panel 06 (original).](../assets/figures/ml_fig_c461_06.png)
*Figure — Support confidence path c461. Synthetic teaching geometry—not a causal claim.*


![c462 teaching panel 06 (original).](../assets/figures/ml_fig_c462_06.png)
*Figure — Apriori level-wise path c462. Synthetic teaching geometry—not a causal claim.*


![c463 teaching panel 06 (original).](../assets/figures/ml_fig_c463_06.png)
*Figure — FP-growth tree path c463. Synthetic teaching geometry—not a causal claim.*


![c464 teaching panel 06 (original).](../assets/figures/ml_fig_c464_06.png)
*Figure — Eclat tidset path c464. Synthetic teaching geometry—not a causal claim.*


![c465 teaching panel 06 (original).](../assets/figures/ml_fig_c465_06.png)
*Figure — Charm closed path c465. Synthetic teaching geometry—not a causal claim.*


![c466 teaching panel 06 (original).](../assets/figures/ml_fig_c466_06.png)
*Figure — PrefixSpan project path c466. Synthetic teaching geometry—not a causal claim.*


![c467 teaching panel 06 (original).](../assets/figures/ml_fig_c467_06.png)
*Figure — SPADE lattice path c467. Synthetic teaching geometry—not a causal claim.*


![c468 teaching panel 06 (original).](../assets/figures/ml_fig_c468_06.png)
*Figure — MinHash Jaccard path c468. Synthetic teaching geometry—not a causal claim.*


![c469 teaching panel 06 (original).](../assets/figures/ml_fig_c469_06.png)
*Figure — LSH band collision path c469. Synthetic teaching geometry—not a causal claim.*


![c470 teaching panel 06 (original).](../assets/figures/ml_fig_c470_06.png)
*Figure — TF-IDF weight bars c470. Synthetic teaching geometry—not a causal claim.*


![c471 teaching panel 06 (original).](../assets/figures/ml_fig_c471_06.png)
*Figure — BM25 score bars c471. Synthetic teaching geometry—not a causal claim.*


![c472 teaching panel 06 (original).](../assets/figures/ml_fig_c472_06.png)
*Figure — Inverted index path c472. Synthetic teaching geometry—not a causal claim.*


![c473 teaching panel 06 (original).](../assets/figures/ml_fig_c473_06.png)
*Figure — Query expand path c473. Synthetic teaching geometry—not a causal claim.*


![c474 teaching panel 06 (original).](../assets/figures/ml_fig_c474_06.png)
*Figure — N-gram coverage bars c474. Synthetic teaching geometry—not a causal claim.*


![c475 teaching panel 06 (original).](../assets/figures/ml_fig_c475_06.png)
*Figure — Sequence gap path c475. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Transactional data model co-occurrence. Support measures how often an itemset appears; confidence estimates conditional probability for association rules; lift compares that probability to independence. IR represents documents and queries with TF–IDF (and related) weights, ranks by cosine or BM25-style scores, and evaluates with precision, recall, and MAP; inverted indexes make large-scale search feasible. Hash tables and MinHash support counting and approximate set similarity; trees (binary, 2-3, B/B+, red-black, trie/radix) organize ordered keys and prefixes; BFS, DFS, beam search, and MCTS explore combinatorial trees; Bloom filters, sliding windows, and skip lists handle membership, streams, and ordered maps efficiently. Apriori, FP-Growth, and ECLAT mine frequent itemsets under the Apriori principle; GSP, SPADE, FreeSpan, and PrefixSpan mine sequences. HMMs, as simple PGMs, use the forward algorithm for likelihood, Viterbi for decoding, and Baum–Welch for unsupervised training. Clinically, baskets and sequences describe encounters and pathways; rules must not be read as causal; phenotype mining must avoid label leakage; retrieval supports literature and note search with explicit evaluation. The central engineering tension is combinatorial explosion versus sparsity: good algorithms and good thresholds keep the useful patterns and leave the noise behind.

## Practice and Reflection

(1) Using the five-transaction example, list all frequent itemsets for minsup = 0.6 and generate all rules with minconf = 0.7. Report support, confidence, and lift for each rule.

(2) Prove formally that if X ⊆ Y then s(X) ≥ s(Y). Explain how this justifies discarding a candidate whose subset is infrequent.

(3) Contrast Apriori, FP-Growth, and ECLAT in terms of data layout (horizontal vs vertical), candidate generation, and number of database scans.

(4) Give a sequence database of four short sequences over {a,b,c}. Find all sequential patterns with minsup = 0.5 under the subsequence definition. Outline how PrefixSpan would project after prefix ⟨a⟩.

(5) For the three-document TF–IDF example, recompute rankings for the query “data models” using cosine similarity. How does the rare term mining affect scores if added to the query?

(6) A system returns 10 documents for a query; 4 of the top 10 are relevant, and there are 8 relevant documents in the collection. Compute precision@10 and recall@10. Sketch how AP would use the ranks of the relevant hits.

(7) Explain how MinHash estimates Jaccard similarity and why LSH enables subquadratic near-duplicate detection of clinical notes.

(8) Compare Bloom filters and hash tables for the task “is this ICD code in yesterday’s order set?” Include false-positive behavior and memory.

(9) Map a hyperacute ischemic stroke pathway to a sequence alphabet (at least six event types). Explain how early death or inter-facility transfer would bias naive support estimates of late-pathway patterns.

(10) For a two-state HMM of “pre-treatment” vs “post-treatment” with binary emission “imaging ordered,” write the forward recursion update and state what Viterbi would return that forward likelihood alone does not.

(11) You mine the rule {IV tPA} → {follow-up CT head} with high confidence in a stroke-center EHR. Give one causal interpretation that is unjustified and one process-of-care interpretation that might be justified after temporal checks.

(12) Design postings lists (doc-id only) for the three-document corpus and show how to answer the boolean query machine AND learning via list intersection.

(13) In the two-state EEG HMM of the worked example (states N, S; emissions q, k), extend the observation sequence to O = (q, k, k, q). Continue the Viterbi trellis one step to t = 4, report δ₄(N) and δ₄(S), and give the most probable state at t = 4 and the updated best path. Explain in one sentence why a quiet fourth second can flip the decoded state back to N.
