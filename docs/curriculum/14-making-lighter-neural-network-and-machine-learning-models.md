# Chapter 14. Making Lighter Neural Network and Machine Learning Models

## Opening

Your hospital’s edge device cannot run a 7-billion-parameter model during a code stroke. Compression, distillation, and efficient architectures are deployment medicine—not just engineering fashion.

## Learning Objectives

State why latency, energy, memory, and privacy constrain model deployment at the bedside and on the edge.

Explain classical compression (BPE, bitmap indexes, Huffman, LZW) and sparse coding as foundations for efficient data and features.

Construct a Huffman code from a symbol-frequency table, compute its average bits per symbol, and compare it against the fixed-length baseline and the entropy bound.

Apply quantization (vector, µ-law/A-law intuition, PTQ, QAT, channel-wise) and pruning (magnitude, OBD/OBS, activation, Taylor; structured/unstructured).

Describe LoRA, FlashAttention, MQA/GQA, sliding-window attention, gradient accumulation/checkpointing, and conjugate-gradient ideas for efficient training and inference.

Connect transfer, multitask, meta, curriculum, and federated learning plus AutoML/NAS to practical efficiency.

Explain knowledge distillation with temperature-scaled soft targets, and select the temperature T and mixing weight alpha for an edge student.

Reason about hardware-aware serving (KV-cache size, supported dtypes, joules per inference) when choosing which compression levers to pull.

Design evaluation protocols for compressed models under MSU/bedside constraints with clinical subgroup checks.

## 14.1 Why Lighter Models Matter

A model that is accurate in a data center can still fail as a clinical tool if it is too slow, too large, too power-hungry, or too dependent on continuous cloud connectivity. Lighter models address the practical constraints of deployment: milliseconds of latency in a stroke alert pathway, limited RAM on a tablet at the bedside, intermittent connectivity in a mobile stroke unit (MSU), battery life for wearable continuous monitoring, and institutional rules that prefer keeping identifiable data on-device rather than streaming raw waveforms or images off-site. Compression and efficient design are therefore not cosmetic optimizations; they are often the difference between a research prototype and a system that can operate under real clinical logistics.

Efficiency also has scientific and ethical dimensions. Training and repeatedly fine-tuning massive networks for multi-site consortium projects consumes substantial electricity and researcher time. Green AI asks us to report compute cost alongside accuracy and to prefer methods that achieve acceptable clinical utility at lower carbon and dollar cost. Privacy-preserving local inference reduces exposure of protected health information during routine prediction, complementing (but not replacing) institutional review, data-use agreements, and secure multi-site analytics. This chapter surveys classical data compression, sparse coding, quantization, pruning, LoRA, efficient attention, memory-efficient training, transfer and related training paradigms, knowledge distillation, and AutoML—always with evaluation that checks whether clinical performance survives the diet.

Latency: time from input ready to prediction (p50/p95 matter more than mean alone).

Throughput: predictions per second under batching; important for research jobs.

Memory footprint: parameters, activations, optimizer state, and runtime buffers.

Energy: joules per inference; critical for battery devices and green accounting.

Communication: bytes moved on and off device; costly in MSUs and rural tele-neurology.

## 14.2 Measuring Parameters, FLOPs, Latency, and Energy

Before compressing, measure. Parameter count is the number of stored trainable weights and biases. For a dense layer mapping R^{d_in} to R^{d_out}, parameters equal d_in * d_out + d_out if a bias is used. Convolutional layers store C_out * C_in * K_h * K_w weights for standard convolutions (plus biases). Parameter count predicts disk size under a given numeric format: roughly 4 bytes per parameter in float32, 2 in float16, 1 in int8, and less for structured sparse formats when zeros are not stored densely.

FLOPs (floating-point operations) approximate arithmetic work for a forward pass. For the same dense layer, multiply-adds are about d_in * d_out (often counted as 2 FLOPs each multiply-add; conventions differ—state yours). For a convolution with output spatial size H_out x W_out, a common estimate is 2 * H_out * W_out * C_out * C_in * K_h * K_w. FLOPs are not latency: memory bandwidth, kernel fusion, batch size, and hardware (CPU, GPU, NPU, microcontroller) dominate wall-clock time. Always benchmark on the target device with realistic input sizes and cold/warm cache conditions.

Worked numerical example: parameters and quantization memory. Consider a small multilayer network for tabular stroke risk scoring with layers 20 -> 64 -> 64 -> 2 (binary logits). Parameter counts: Layer 1: 20*64+64=1344; Layer 2: 64*64+64=4160; Layer 3: 64*2+2=130; total P=5634. Float32 storage: 5634*4=22536 bytes ≈ 22.0 KiB. Float16: 11268 bytes ≈ 11.0 KiB. Int8 weight-only: 5634 bytes ≈ 5.5 KiB plus tiny scale metadata. If we prune 40% of weights to exact zero and store a dense bitmask plus nonzero int8 values: nonzero ≈ 3380 bytes + 5634/8 ≈ 705 bytes mask ≈ 4.0 KiB before alignment. Structured pruning (removing whole neurons) avoids index overhead and better matches hardware.

Approximate dense FLOPs for one forward pass (counting multiply-adds as 2 FLOPs): 2*(20*64 + 64*64 + 64*2)=11008 FLOPs, ignoring activations. On a microcontroller that sustains 10 MFLOP/s effective, pure arithmetic would be ~1.1 ms, but memory loads dominate; measured latency might be several milliseconds. Use parameter and FLOP counts for design comparisons, then measure latency and energy on-device.

```
def mlp_params(layer_sizes, bias=True):
 p = 0
 for d_in, d_out in zip(layer_sizes, layer_sizes[1:]):
 p += d_in * d_out + (d_out if bias else 0)
 return p

P = mlp_params([20, 64, 64, 2])
assert P == 5634
print(P, P * 4, P * 1) # params, float32 bytes, int8 bytes
```

## 14.3 Data Compression Algorithms: BPE, Bitmap, Huffman, and LZW

Before neural compression, classical data compression reduces storage and communication of features, tokens, and indices. These algorithms appear inside modern ML pipelines (tokenizers, inverted indexes, feature stores) and teach information-theoretic intuition that also motivates quantization and sparse codes.

Byte Pair Encoding (BPE). BPE is a simple tokenization algorithm widely used in NLP and large language models. Start with a character-level vocabulary. Iteratively find the most frequent adjacent pair of symbols and merge them into a new symbol, until a target vocabulary size is reached. The result is a subword vocabulary that balances word-level efficiency with open-vocabulary coverage of rare words and morphology. BPE is not a general-purpose compressor for arbitrary binary files, but it compresses text into fewer tokens, which reduces sequence length and therefore attention compute. Clinical note corpora benefit because drug names, dosages, and abbreviations form reusable subwords.

Bitmap indexes. A bitmap index stores, for each distinct value of a categorical column (or each token in a set), a bit vector with a 1 in rows where that value occurs. Intersection and union of predicates become bitwise AND/OR—extremely fast for selective filters on large clinical tables (e.g., “ICD ischemic stroke AND age>=65 AND not comfort care”). Compression schemes such as run-length encoding and roaring bitmaps (which split the row space into fixed-size chunks and store each chunk as a sorted integer array, a dense bitset, or a run list, whichever is smallest) keep sparse bitmaps small. Bitmap indexes trade space for query speed and are foundational in analytic databases used for cohort construction.

Huffman coding. Huffman coding assigns shorter binary codes to more frequent symbols and longer codes to rare ones, producing a prefix-free code that is optimal among symbol codes for a known discrete distribution. Build a binary tree by repeatedly merging the two least frequent nodes; the code length of a symbol equals its depth in that tree, and no codeword is a prefix of another (so a stream decodes unambiguously left to right). In ML systems, Huffman (and related entropy codes) compress postings lists, sparse feature IDs, and sometimes quantized residual streams. The lesson for quantization: allocate fewer bits where mass concentrates.

Lempel-Ziv-Welch (LZW). LZW builds a dictionary of previously seen substrings on the fly and emits codes for dictionary entries—no need for a prior frequency table. It underpins classic formats (GIF, early compressors) and illustrates adaptive dictionary compression. For streaming telemetry from wearables or MSUs, dictionary methods can reduce bandwidth when signals or event logs repeat patterns. Modern learned compressors and neural codecs go further, but LZW remains a clean teaching algorithm: initialize dictionary with alphabet; while reading input, grow the longest match already in the dictionary and emit its code, then add the match-plus-next-symbol as a new entry.

BPE: subword tokens for text; reduces sequence length for transformers.

Bitmap indexes: fast set operations for cohort filters on categorical columns.

Huffman: shorter codes for frequent symbols; entropy-optimal symbol codes.

LZW: adaptive dictionary; good for repetitive streams without a prior model.

### Worked Example: Huffman Coding a Wearable Event Stream

A wearable cardiac patch classifies each detected beat into one of six event codes and streams that code sequence to a phone over a low-power radio. To cut bandwidth (and therefore battery drain) we entropy-code the stream. Over one monitoring window it logs 100 beats with the empirical counts below (toy data). We convert counts to probabilities by dividing by 100.

Figure 14.1. The Huffman code built for the wearable beat stream, whose six event codes N, S, V, A, P, Q occur 40, 25, 15, 10, 6, and 4 times per 100-beat window. Repeatedly merging the two lowest-weight nodes (P+Q=10, then +A=20, +V=35, +S=60, +N=100) yields a maximally skewed tree; labelling each branch 0/1 gives the prefix-free codewords 0, 10, 110, 1110, 11110, 11111. The average length L = sum_i p_i L_i = 2.25 bits/symbol beats the 3-bit fixed-length code (225 vs 300 bits per window, a 25% saving) and sits just above the entropy floor H = 2.20 bits, while the Kraft sum of 2^(-L_i) = 1 confirms a complete prefix code.

| Symbol | Meaning (toy) | Count | Probability p |
| --- | --- | --- | --- |
| N | normal | 40 | 0.40 |
| S | supraventricular | 25 | 0.25 |
| V | ventricular | 15 | 0.15 |
| A | artifact | 10 | 0.10 |
| P | paced | 6 | 0.06 |
| Q | unclassified | 4 | 0.04 |

Fixed-length baseline. Six distinct symbols need ceil(log2 6) = ceil(2.585) = 3 bits each. That is 3 bits/symbol, or 3 * 100 = 300 bits for the whole window, regardless of how skewed the frequencies are.

Build the tree. Maintain a pool of nodes, each carrying a weight; repeatedly remove the two smallest-weight nodes, join them under a new parent whose weight is their sum, and return the parent to the pool. Each row below shows the two nodes merged and the pool that results.

| Step | Merge (weights) | New node | Pool after step |
| --- | --- | --- | --- |
| 1 | P(6) + Q(4) | n1(10) | N40, S25, V15, A10, n1(10) |
| 2 | A(10) + n1(10) | n2(20) | N40, S25, V15, n2(20) |
| 3 | V(15) + n2(20) | n3(35) | N40, S25, n3(35) |
| 4 | S(25) + n3(35) | n4(60) | N40, n4(60) |
| 5 | N(40) + n4(60) | root(100) |  |

Assign codes. Walk from the root; label the branch toward the higher-weight child 0 and the branch toward the lower-weight child 1 (so frequent symbols sit near the root and receive short codes). The codeword of a leaf is the sequence of branch labels from root to leaf, and its length equals the leaf’s depth.

| Symbol | Codeword | Length L |
| --- | --- | --- |
| N | 0 | 1 |
| S | 10 | 2 |
| V | 110 | 3 |
| A | 1110 | 4 |
| P | 11110 | 5 |
| Q | 11111 | 5 |

This code is prefix-free: N begins with 0, every other codeword begins with 1, and the branching continues to separate S, V, A, P, Q, so no codeword is a prefix of another.

Average code length. Multiply each probability by its code length and sum:

N: 0.40 * 1 = 0.40

S: 0.25 * 2 = 0.50

V: 0.15 * 3 = 0.45

A: 0.10 * 4 = 0.40

P: 0.06 * 5 = 0.30

Q: 0.04 * 5 = 0.20

Sum L̄ = 0.40 + 0.50 + 0.45 + 0.40 + 0.30 + 0.20 = 2.25 bits/symbol. Over the 100-beat window that is 225 bits, versus 300 bits for the fixed-length code—a saving of 75 bits, or (3 − 2.25)/3 = 25%.

Compare to the entropy bound. The entropy H = −Σ p log2 p is the theoretical floor no lossless symbol code can beat. Computing term by term (using log2 0.40 = −1.322, log2 0.25 = −2.000, log2 0.15 = −2.737, log2 0.10 = −3.322, log2 0.06 = −4.059, log2 0.04 = −4.644):

N: 0.40 * 1.322 = 0.529

S: 0.25 * 2.000 = 0.500

V: 0.15 * 2.737 = 0.411

A: 0.10 * 3.322 = 0.332

P: 0.06 * 4.059 = 0.244

Q: 0.04 * 4.644 = 0.186

H = 0.529 + 0.500 + 0.411 + 0.332 + 0.244 + 0.186 ≈ 2.20 bits/symbol (the exact value is 2.2008; the displayed terms are each rounded to three decimals). So Huffman’s 2.25 bits sits only about 0.05 bits above the entropy floor of 2.20 and well below the naive 3-bit fixed code—the classic result that Huffman is optimal among integer-length symbol codes and never more than about one bit above H. A consistency check via the Kraft equality confirms a complete prefix code: 2⁻¹ + 2⁻² + 2⁻³ + 2⁻⁴ + 2⁻⁵ + 2⁻⁵ = 0.5 + 0.25 + 0.125 + 0.0625 + 0.03125 + 0.03125 = 1.0.

This particular distribution happens to produce a maximally skewed (“comb”) tree, so codeword lengths climb 1, 2, 3, 4, 5, 5; more uniform frequencies give bushier, more balanced trees with codes closer to the fixed length. Different tie-breaking rules when two nodes share a weight can yield different bit patterns but the same optimal average length. The takeaway that carries into quantization and pruning: spend bits where the probability mass lives, and starve the rare tail—but check first that the rare tail is not the clinically critical one (a run of ventricular beats is exactly the event you do not want to under-resolve).

```
# Huffman construction from symbol frequencies, and average code length.
import heapq

def build_huffman(freq): # freq: {symbol: weight}
 heap = [(w, i, sym) for i, (sym, w) in enumerate(freq.items())]
 heapq.heapify(heap) # min-heap keyed by weight
 k = len(freq) # tie-break id so nodes never compare
 while len(heap) > 1:
 w1, _, n1 = heapq.heappop(heap) # two smallest-weight nodes
 w2, _, n2 = heapq.heappop(heap)
 heapq.heappush(heap, (w1 + w2, k, (n1, n2)))
 k += 1
 root = heap[0][2]

 codes = {}
 def walk(node, prefix):
 if isinstance(node, str): # leaf
 codes[node] = prefix or "0" # lone-symbol edge case
 else:
 walk(node[0], prefix + "0") # higher-weight child -> 0
 walk(node[1], prefix + "1") # lower-weight child -> 1
 walk(root, "")
 return codes

def avg_bits(freq, codes):
 total = sum(freq.values())
 return sum(freq[s] * len(codes[s]) for s in freq) / total

freq = {"N": 40, "S": 25, "V": 15, "A": 10, "P": 6, "Q": 4}
codes = build_huffman(freq)
print(codes) # {'N': '0', 'S': '10', 'V': '110', ...}
print(avg_bits(freq, codes)) # 2.25
```

## 14.4 Sparse Coding

Sparse coding represents a vector x approximately as a linear combination of a few columns of a dictionary D: x ≈ D z with z sparse (many zeros). Learning D and z jointly is a classic unsupervised problem related to dictionary learning, compressed sensing, and efficient sensory coding hypotheses in neuroscience. Optimization typically alternates sparse inference for z (lasso; OMP, which greedily adds the dictionary atom most correlated with the current residual; ISTA/FISTA, which alternate a gradient step with soft-thresholding of the coefficients) with dictionary updates.

Why sparse codes lighten models: (1) storing and computing with few active coefficients reduces FLOPs when sparsity is exploited; (2) overcomplete dictionaries can capture structure with interpretable atoms (edge filters, motif detectors); (3) sparse linear models (L1 logistic regression) remain strong baselines for tabular clinical prediction with built-in feature selection. Sparse coding differs from neural ReLU sparsity: the former optimizes an explicit sparsity penalty on codes; the latter hopes architectures induce zeros. In practice, sparse linear models and tree ensembles are still the first “light” models to try on structured EHR tables before deep compression of neural nets.

Worked intuition. Suppose x in R^100 must be reconstructed. A dense code uses 100 coefficients; a 5-sparse code on a dictionary of 200 atoms uses 5 nonzero coefficients—20x fewer multiplies at inference if the dictionary multiplies are implemented sparsely. Dictionary quality determines reconstruction error; over-aggressive sparsity underfits rare but critical patterns (subtle ECG morphology, rare imaging signs).

## 14.5 Quantization: Vector, Companding, PTQ, QAT, and Channel-wise

Quantization represents continuous values with a smaller discrete set. Vector quantization (VQ) and cluster quantization map vectors to a codebook of centroids (k-means style); encoding stores the centroid index rather than the full vector. VQ-VAE and product quantization (which splits each vector into subvectors and quantizes each subvector with its own small codebook, so a handful of indices approximate a high-dimensional vector—the workhorse of billion-scale approximate nearest-neighbor search) in retrieval systems use related ideas. Scalar quantization maps each number independently.

Figure 14.2. Amplitude quantization of a smooth full-scale (plus/minus 1) signal onto a uniform 3-bit grid. Three bits give 2^3 = 8 levels spaced by step Delta = 2/7 (about 0.286), so the continuous curve (indigo) is replaced by the staircase (amber) that snaps each sample to its nearest level, encoded 000 to 111 on the right axis. The shaded band is the quantization error; the bit-width sets the resolution exponentially (n bits map to 2^n levels), so more bits mean a finer grid and smaller error at the cost of storage.

Non-uniform quantization and signal companding. Human perception and many signals have wide dynamic range. µ-law and A-law companding (classic telephony) apply a logarithmic-like transform before uniform quantization so that small amplitudes get finer effective resolution. The µ-law compressor is roughly F(x) = sign(x) ln(1+µ|x|)/ln(1+µ) for |x|<=1 (telephony uses µ=255); expand after transmission. The A-law variant is piecewise. These remind us that uniform int8 grids can waste levels on rarely used tails; learned or logarithmic quantizers sometimes help audio and sensor streams.

Quantization for neural networks. Fixed-point and integer formats replace float32 multiplies with cheaper int8/int4 operations on NPUs. Uniform affine quantization maps a real value r to an integer q by q = round(r / scale) + zero_point, with dequantization r_hat = scale * (q - zero_point). Per-tensor scales are simple; per-channel (channel-wise) scales for convolutional weights usually recover accuracy by adapting to each filter’s range. Dynamic quantization computes activation scales at runtime; static quantization calibrates scales on a representative dataset ahead of time for faster inference.

Post-Training Quantization (PTQ) converts a trained float model using calibration data without full retraining—fast but may lose accuracy on sensitive models. Quantization-Aware Training (QAT) simulates quantization during training (straight-through estimators for round operations) so weights adapt to discrete grids—more accurate, more expensive. When to quantize: after architecture and training are stable; validate calibration curves and subgroup metrics, not only top-1 accuracy. For clinical imaging, aggressive int4 weights can erase rare lesion cues; prefer mixed precision (int8 activations, higher precision sensitive layers) and pathology-stratified evaluation.

Worked scale example. A weight tensor with min=-1.2, max=1.8 quantized to int8 in [-128,127] might use scale = (1.8-(-1.2))/(127-(-128)) ≈ 3.0/255 ≈ 0.0118, zero_point chosen so that 0.0 maps near an integer. Channel-wise quantization computes such scales per output channel, reducing error when channels have very different magnitudes.

```
import numpy as np

def affine_quantize(x, n_bits=8):
 qmin, qmax = 0, 2**n_bits - 1
 xmin, xmax = float(np.min(x)), float(np.max(x))
 scale = (xmax - xmin) / (qmax - qmin) if xmax > xmin else 1.0
 zp = int(round(qmin - xmin / scale))
 q = np.clip(np.round(x / scale + zp), qmin, qmax).astype(np.int32)
 x_hat = scale * (q - zp)
 return q, scale, zp, x_hat
```

## 14.6 Pruning and Sparsification

Pruning sets selected weights or structures to zero (or removes them) so that computation and storage shrink. Key design choices: what to prune (weights, neurons, channels, heads, blocks), when to prune (one-shot after training, iterative prune-retrain, or during training), and structured versus unstructured sparsity.

Figure 14.3. Magnitude pruning of a small fully-connected network (4-6-3 units, 42 weights). On the left every weight is drawn with width and opacity proportional to its magnitude and coloured by sign; on the right the low-magnitude edges have been set to zero, removing 23 of 42 weights (54.8% sparsity) and leaving only the high-magnitude connections. This unstructured sparsity shrinks storage but needs sparse kernels to gain speed on dense hardware, which is why structured pruning of whole neurons or channels is usually preferred for edge CPUs.

Unstructured pruning can reach high sparsity in theory but needs sparse kernels or special hardware to gain latency. Structured pruning removes channels, filters, attention heads, or entire residual blocks, yielding dense smaller tensors that ordinary BLAS and conv libraries accelerate immediately.

How to identify pruning candidates. Magnitude-based pruning removes weights with smallest absolute value—simple and surprisingly strong with iterative fine-tuning. This connects to the lottery-ticket hypothesis: within a large trained network there often exists a small subnetwork that, retrained from its original initialization, matches the full model—so pruning is partly a search for that winning subnetwork rather than mere trimming. Activation-based pruning removes units that rarely activate on calibration data. Optimal Brain Damage (OBD) and Optimal Brain Surgeon (OBS) use second-order (Hessian) information to estimate the increase in loss from removing a parameter—more principled, more expensive; OBS accounts for parameter interactions via approximate inverse-Hessian updates. Taylor expansion based pruning scores the loss change using first-order gradients times weights (or higher-order approximations), ranking units by estimated impact.

A practical recipe: train to convergence; rank weights or structures by magnitude or saliency; remove a fraction s; fine-tune with a modest learning rate; validate not only average accuracy but calibration and subgroup performance (age bands, scanner vendors, transferring hospitals). In medical imaging, aggressive channel pruning can erase rare but critical features (subtle hyperdense MCA sign, small diffusion lesions); use class-wise and pathology-wise checks, not only global AUC.

Unstructured: fine-grained zeros; best compression ratio; hardest latency win on CPUs.

Structured: remove filters/neurons/heads; reliable speedups; coarser search.

OBD/OBS: Hessian-aware saliency; costly but classic theory.

Always fine-tune after pruning; one-shot pruning often over-destroys accuracy.

## 14.7 Low-Rank Adaptation (LoRA)

Full fine-tuning of large language or vision models updates all parameters and stores a separate copy per task—prohibitive when many clinical sites or tasks share a foundation model. Low-Rank Adaptation (LoRA) freezes the pretrained weights W0 and injects trainable low-rank updates: W = W0 + B A, where B is d x r, A is r x k, and rank r << min(d,k). Only A and B are trained (often with scaling alpha/r). At inference, BA can be merged into W0 so there is no extra latency, or kept separate for multi-tenant adapters.

Figure 14.4. Low-Rank Adaptation (LoRA). The pretrained weight matrix W0 (d x d) is frozen and the per-task update is factored as a product of two trainable low-rank matrices, W = W0 + B A, with B of shape d x r and A of shape r x d for rank r much smaller than d. For d = 1024 and r = 8, full fine-tuning would train d^2 = 1,048,576 parameters per task whereas LoRA trains only 2dr = 16,384 (a 64x reduction, about 1.6%); because B A can be merged back into W0 at inference, the adapter adds no extra latency.

LoRA is a parameter-efficient fine-tuning method, not a general compressor of the base model, but it makes specialization light: many small adapters can sit on one frozen backbone. Variants include QLoRA (LoRA on quantized bases), higher-rank or adaptive-rank schemes, and related adapters (prefix tuning, which prepends trainable key/value vectors to each attention layer; (IA)^3, which learns per-channel rescaling vectors). For multi-hospital NLP, train a shared clinical LLM backbone once, then LoRA-adapt to local note styles without shipping full model copies. Evaluate whether adapter merge preserves safety behaviors and whether low rank underfits rare local phenotypes.

## 14.8 Lighter Self-Attention: FlashAttention, MQA/GQA, Sliding Windows

Self-attention is the computational bottleneck of transformers: naive attention materializes an N x N score matrix for sequence length N, with memory and time quadratic in N. Long clinical notes, genomic sequences, and high-resolution imaging tokens make quadratic cost painful.

FlashAttention recomputes attention in tiles that fit in fast on-chip SRAM, avoiding materializing the full attention matrix in high-bandwidth memory. It is an exact attention algorithm (up to numerical details) with reduced memory traffic—often large wall-clock speedups and longer practical context windows. FlashAttention-2/3 further optimize parallelism and hardware utilization. For practitioners: prefer well-tested kernels rather than hand-rolled sparse patterns unless you need a specific structure.

Multi-Query Attention (MQA) and Grouped-Query Attention (GQA). Standard multi-head attention uses separate key/value projections per head. MQA shares a single key/value head across all query heads, dramatically reducing KV-cache size during autoregressive decoding—critical for serving LLMs. GQA is a middle ground: groups of query heads share KV heads, recovering quality while keeping caches smaller than full multi-head attention. For bedside chat-style assistants, decode-time memory often dominates; MQA/GQA are deployment levers as important as weight quantization.

Sliding window and dilated sliding window attention. Restrict each token to attend only to a local window of width w (and optionally dilated neighbors), reducing complexity from O(N^2) to O(N w). Stacking layers expands the receptive field. Dilated windows insert gaps to reach farther tokens without full density. These patterns suit long signals (EEG, continuous monitoring) where local context dominates but some long-range markers matter. Hybrid designs combine local windows with a few global tokens (CLS, summary tokens) for document-level information.

## 14.9 Memory-Efficient Gradient Descent: Accumulation, Checkpointing, Conjugate Gradients

Training memory, not only inference size, constrains who can fine-tune models. Gradient accumulation splits a large logical batch into micro-batches, computing and summing gradients before one optimizer step—simulating large batches under limited GPU memory at the cost of fewer synchronous updates per wall-clock time. It does not reduce activation memory within a micro-batch.

Gradient checkpointing (activation recomputation) stores only a subset of intermediate activations during the forward pass and recomputes the rest during backward. Memory falls roughly with the number of checkpoints at the cost of extra forward compute (commonly ~20-30% more time for large savings). A common design keeps about sqrt(L) checkpoints for L layers, giving roughly O(sqrt(L)) activation memory instead of O(L) at the price of one extra forward pass—essential for long clinical sequences and 3D medical volumes.

Conjugate gradient (CG) methods solve linear systems Ax=b using A-conjugate search directions, converging faster than steepest descent for quadratic objectives when A is SPD. In ML, CG appears inside Hessian-free optimization, natural gradient / Fisher-vector products (as in TRPO’s inner loop), and large least-squares problems. Computing the next CG step uses recurrence: residual r, search direction p, step size alpha = (r·r)/(p·A p), update x and r, then beta to mix the next direction—without forming A explicitly if matrix-vector products are available. For most deep nets, first-order methods (AdamW) dominate; know CG as the workhorse when second-order or trust-region subproblems appear.

## 14.10 Neural Network Training Paradigms for Efficiency

Transfer learning reuses a model pretrained on a large source task (ImageNet, web text, multi-hospital notes) and adapts it to a target task with less data and compute than training from scratch. Freeze early layers or fine-tune end-to-end with small learning rates. Multitask learning trains shared representations on several related tasks (e.g., ICH detection and midline shift regression), improving data efficiency when tasks share structure; watch for negative transfer when tasks conflict.

Meta-learning aims to learn how to learn: model-based methods adapt internal state quickly; metric-based methods (prototypical networks, matching networks) classify by similarity in an embedding space; optimization-based methods (MAML) learn initial parameters that fine-tune in a few gradient steps. Few-shot rare disease imaging is the natural clinical story—results are promising but brittle under domain shift.

Curriculum learning orders examples from easy to hard (or from common to rare phenotypes), sometimes stabilizing training and improving final performance. Federated learning trains across sites without pooling raw PHI: sites compute local updates, a server aggregates (e.g., FedAvg). Communication compression, differential privacy, and non-IID site distributions are first-class challenges. Federated methods lighten privacy risk of centralization but do not remove re-identification or membership risks entirely; governance remains essential.

## 14.11 Knowledge Distillation

Knowledge distillation trains a smaller student model to mimic a larger teacher. Hard targets are one-hot labels; soft targets are the teacher’s probability distribution over classes, which carry dark knowledge about similarities (e.g., confusing ischemic subtypes). Softmax temperature T flattens distributions: p_i = exp(z_i/T) / sum_j exp(z_j/T). Higher T reveals more inter-class structure; the distillation loss often scales by T^2 when using cross-entropy on soft targets (so that soft-target gradients keep a magnitude comparable to the hard-label term as T grows). A balancing factor alpha mixes soft distillation loss with hard label loss on the transfer set (data used for distillation—labeled or unlabeled): L = alpha * T^2 * CE(soft_student, soft_teacher) + (1 - alpha) * CE(student, hard_labels).

Figure 14.5. Knowledge distillation. A large teacher's logits are passed through a temperature-scaled softmax, p_i = exp(z_i/T) / sum_j exp(z_j/T), to produce soft targets that a small edge student mimics alongside the hard labels. Raising the temperature from T = 1 to T = 4 on the teacher logits z = [2.0, 1.0, 0.1] flattens the distribution from [0.66, 0.24, 0.10] to [0.42, 0.32, 0.26], exposing the inter-class similarities (the 'dark knowledge'). The student minimises L = alpha T^2 CE(soft) + (1 - alpha) CE(hard), where the T^2 factor keeps the soft-target gradient comparable in magnitude to the hard-label term as T grows.

Training recipe: train teacher well; choose student architecture that fits the edge budget; run distillation on a transfer set representative of deployment; tune T and alpha; validate student calibration and failure modes. Architectures include response-based distillation (logits), feature-based (match intermediate maps), and relation-based (match pairwise structures). For stroke imaging, a heavy ensemble teacher can distill into a single student deployable on MSU hardware—if the transfer set includes the MSU’s scanner characteristics.

```
import numpy as np

def softmax(z, T=1.0):
 z = np.asarray(z, dtype=float) / T
 z = z - z.max()
 e = np.exp(z)
 return e / e.sum()

teacher_logits = np.array([2.0, 1.0, 0.1])
print(softmax(teacher_logits, T=1.0)) # peaked -> ~[0.66, 0.24, 0.10]
print(softmax(teacher_logits, T=4.0)) # softer -> ~[0.42, 0.32, 0.26]
```

## 14.12 Automatic Machine Learning: Hyperparameters and NAS

AutoML automates model selection and hyperparameter search so that scarce expert time is spent on problem definition and validation design rather than manual grid fiddling. Grid search enumerates a Cartesian product of candidate values—simple, parallelizable, cursed by dimensionality. Random search samples combinations independently; empirically often finds good configurations faster than grids when only a few hyperparameters matter (because it does not waste trials re-testing the same value of an unimportant knob). Beyond these, three methods improve sample efficiency: Bayesian optimization fits a cheap surrogate model (often a Gaussian process or tree ensemble) to the observed score surface and picks the next configuration by an acquisition function that trades exploration against exploitation; Hyperband races many configurations on small budgets and repeatedly kills the worst, reallocating budget to survivors; population-based training evolves a pool of models, copying and perturbing the best mid-training.

Neural Architecture Search (NAS) searches over network structures (depth, width, cell motifs, connectivity). Approaches include reinforcement learning controllers, evolutionary search, and differentiable search (DARTS-style continuous relaxations that make the choice of operation a trainable weight). NAS can discover efficient architectures but is expensive and can overfit the validation set used for search; always hold out a final test and prefer search spaces that encode hardware constraints (latency on target device). For clinical teams, disciplined random search with early stopping and a clear validation protocol often beats poorly controlled NAS theater.

Report search budget (trials, GPU-hours) alongside accuracy so that “state of the art” claims are comparable under green AI norms.

## 14.13 Clinical and Epidemiologic Notes: Bedside, Edge, and MSU

Mobile stroke units, ED tablets, ICU bedside monitors, and rural tele-neurology endpoints share constraints: limited compute, limited bandwidth, strict latency, and high cost of failure. A 200 ms model that is 0.5 AUC points worse may beat a 3 s cloud model that arrives after the decision window. Quantization and structured pruning are usually the first levers; distillation helps when a large offline teacher is available; LoRA helps multi-site specialization; FlashAttention and GQA matter when language models enter the workflow.

Evaluation for compressed clinical models must include: (1) primary discrimination and calibration on target hardware; (2) latency p95 under concurrent load; (3) energy/battery impact if relevant; (4) subgroup performance (age, sex, race/ethnicity, scanner vendor, transfer status); (5) silent-failure modes under distribution shift (new protocol, new contrast timing); (6) privacy review of what leaves the device. Federated learning can train across MSUs without centralizing raw DICOM, but non-IID patient mixes and intermittent connectivity require robust aggregation and versioning.

Green AI and equity intersect: the institutions that can afford massive models are not always the ones that need bedside tools most. Lightweight models, openly reported compute costs, and transfer from well-resourced pretraining to under-resourced deployment sites are practical justice levers—not panaceas, but better than accuracy-only leaderboards.

Target-device benchmarks beat FLOP estimates for go/no-go decisions.

MSU/edge: prefer structured sparsity, int8, and offline-capable pipelines.

Validate pathology tails after every compression step.

Report compute and energy with accuracy for green and multi-site science.

## 14.14 Putting Compression Techniques Together

A sensible pipeline: (1) choose the smallest model family that can plausibly work (classical sparse models before deep nets when tabular); (2) train or transfer well in full precision; (3) distill if a large teacher helps; (4) prune structured units iteratively with fine-tuning; (5) quantize with PTQ, escalate to QAT if needed; (6) apply efficient attention and serving tricks for transformers; (7) use LoRA for site adapters; (8) measure on-device. Classical compressors (BPE, bitmaps, Huffman, LZW) optimize data movement around the model. AutoML can tune the pipeline but does not replace clinical validation.

complementary methods multiply gains: a distilled, pruned, int8 student with GQA decoding can be orders of magnitude cheaper than the original teacher ensemble while retaining most utility—if and only if evaluation is honest on the deployment distribution.

## 14.15 Worked Compression Stack on a Tiny CNN

Consider a pedagogical CNN for binary ICH-vs-no-ICH on downsampled slices: two convolutional blocks (16 then 32 filters, 3x3 kernels) and a 64-unit dense head. Approximate parameter counts: first conv 1*16*3*3 + 16 = 160; second 16*32*3*3 + 32 = 4640; dense if flattened spatial size is 8x8x32=2048 inputs: 2048*64+64=131136; logits 64*2+2=130; total roughly 136k parameters (~544 KB float32).

Figure 14.6. The accuracy-versus-size trade-off for the chapter's tiny ICH CNN and its compressed variants, plotted on a logarithmic size axis. Using the section's parameter counts and AUROC values, the fp32 baseline (about 544 KB, 0.91), int8 quantization (about 136 KB, 0.89), structured pruning (about 409 KB, 0.90), the distilled student (about 102 KB, 0.905), and a larger teacher ensemble (0.93) trace a Pareto frontier (dashed) through the non-dominated models. Distillation lifts the smallest model onto the frontier, while pruning or quantization applied on their own remain dominated by it.

Apply structured pruning of 25% of channels after the second conv (32->24 filters), reducing that layer’s parameters by 25% and shrinking the dense input from 2048 to 1536 features (8*8*24), cutting the dense layer from 131k to about 98k parameters—often the dominant saving. Fine-tune two epochs. Then PTQ to int8 on weights: storage falls ~4x on remaining weights if activations stay float16. Measure: suppose full float32 AUROC 0.91; pruned 0.90; int8 0.89 on a balanced research set. On an MSU tablet, latency might drop from 180 ms to 70 ms. Whether 0.89 is acceptable depends on decision utility, not leaderboard pride.

Add distillation: a larger teacher ensemble at AUROC 0.93 soft-labels the student during fine-tuning after pruning, recovering to 0.905. This storyboard—measure, prune structure, quantize, distill, re-measure on device and on rare positive tails—is the operational sequence to memorize.

Structure first: channel/head pruning before unstructured lottery tickets on edge CPUs.

Quantize after accuracy stabilizes; escalate PTQ to QAT only if tails fail.

Distill to recover student quality without shipping the teacher.

Always re-benchmark p95 latency and subgroup AUROC after each step.

## 14.16 Classical Feature Selection and Hashing as Non-Neural Lightening

Not every light model is a compressed neural net. For tabular stroke registries, L1-regularized logistic regression, tree ensembles with shallow depth, and mutual-information feature filters often dominate deep models after honest external validation. Feature hashing (hashing trick) maps high-cardinality categorical strings (med names, ICD strings) into a fixed-dimensional bag-of-features with collisions, enabling streaming and low memory. Bloom filters test set membership with controlled false-positive rates—useful for excluding known-non-events in retrieval pipelines.

Sparse linear models are already “pruned” by construction. Combined with BPE or dictionary-coded text features and bitmap indexes for cohort filters, a classical stack can serve bedside calculators without GPUs. The chapter’s neural compression tools matter when perceptual inputs (imaging, raw waveforms, long notes) force representation learning; otherwise prefer the lighter classical baseline and invest effort in labels and calibration.

## 14.17 Hardware-Aware Serving Notes

Compression interacts with hardware. GPUs prefer dense tensor cores—structured pruning and quantization to supported dtypes win; random 95% sparsity may not. NPUs and mobile DSPs love int8/int4 with operator fusion. CPUs benefit from smaller memory footprints that stay in cache. Batch size 1 bedside inference stresses latency differently than batch 32 research jobs. KV-cache size for LLM decoding often dominates RAM: GQA/MQA and shorter prompts via RAG chunk budgets are “light model” techniques as much as weight pruning.

Energy reporting: measure joules for a fixed workload (1000 inferences) under controlled screen/radio states when claiming edge suitability. Green AI multi-site papers should report training GPU-hours for the teacher and student, not only AUROC deltas.

## 14.18 Common Pitfalls Across the Compression Stack

Optimizing FLOPs instead of latency and energy. A model with half the FLOPs can be slower if memory bandwidth or unsupported operators dominate. Fix: benchmark wall-clock and joules on the actual target device.

Unstructured sparsity without sparse kernels. 90% zeros on a CPU running dense BLAS execute at dense speed (or slower, once indexing overhead is added). Fix: prefer structured pruning unless the runtime genuinely exploits sparsity.

Judging compression by global accuracy or AUROC alone. Aggregate metrics mask erosion of rare-but-critical tails (small diffusion lesions, hyperdense-MCA sign). Fix: pathology- and subgroup-stratified evaluation plus calibration after every step.

One-shot aggressive pruning or quantization. Cutting half the channels or jumping straight to int4 in a single pass often destroys accuracy irrecoverably. Fix: iterate prune/quantize with fine-tuning; escalate PTQ to QAT only if the tails fail.

Calibrating quantization on unrepresentative data. Static PTQ scales fit on the wrong distribution mis-scale activations at deployment. Fix: calibrate on data resembling the deployment scanner and site.

Letting NAS or AutoML score on the eventual test set. Search overfits whatever it is scored on, inflating reported performance. Fix: hold out a final untouched test set and report the search budget.

Equating on-device inference with privacy. Local prediction can still leak through model updates, logs, or model-inversion attacks; federated is not anonymous. Fix: pair local computation with differential privacy, secure aggregation, and governance.

Distilling or LoRA-adapting without re-checking behavior. A smaller student or a merged adapter can quietly lose calibration or safety behaviors. Fix: re-run the full calibration and safety suite on the final artifact, not just the base model.

## Connections

Information theory. Huffman coding, entropy, and the µ-law/quantization material all rest on one idea: spend bits where the uncertainty (probability mass) lives. The entropy H is the floor that no lossless symbol code can beat, and the worked example shows Huffman resting just above it. Lossy quantization deliberately dips below H by discarding precision the task can tolerate—a controlled version of the same accounting.

Optimization. Conjugate gradients, Hessian-aware pruning (OBD/OBS), and the straight-through estimator in QAT all reach back to the optimization material: curvature (the Hessian) tells you which parameters matter, even though first-order methods (AdamW) still run day-to-day training.

Transformers and attention. FlashAttention, MQA/GQA, and sliding windows presuppose the attention mechanism from the sequence-model chapters; they are its deployment-time counterpart, trading exactness or global context for memory and speed.

Representation reuse and generalization. Distillation, LoRA, and transfer/meta-learning all recycle learned representations, so they inherit the generalization and domain-shift cautions raised earlier; the calibration and evaluation chapters supply the subgroup and reliability checks that every compression step must survive.

Privacy and governance. Federated learning links to the data-governance material: keeping computation on-device is a technical lever, not a substitute for consent, data-use agreements, and membership-inference risk assessment.

## Chapter Summary

Lighter models enable bedside, edge, and mobile stroke unit deployment under latency, memory, energy, and privacy constraints. Classical compressors (BPE, bitmaps, Huffman, LZW) and sparse coding reduce data and feature cost; the Huffman worked example shows a prefix code reaching 2.25 bits/symbol against a 3-bit fixed baseline and a 2.20-bit entropy floor. Quantization (including companding intuition, PTQ, QAT, and channel-wise scales) and pruning (magnitude, Hessian-aware OBD/OBS, activation and Taylor scores; structured vs unstructured) shrink networks. LoRA specializes large models cheaply; FlashAttention, MQA/GQA, and sliding windows lighten attention; accumulation and checkpointing lighten training memory. Transfer, multitask, meta, curriculum, and federated learning improve data efficiency; distillation compresses teachers into students; AutoML and NAS search configurations under budget. Clinical notes emphasize on-device measurement, pathology-tail validation, and green AI reporting for equitable multi-site neurology.

## Practice and Reflection

(1) Recompute parameter counts and int8 storage for an MLP with sizes 40 -> 128 -> 128 -> 64 -> 3. Estimate dense FLOPs for one forward pass.

(2) Explain why structured channel pruning often yields better mobile latency than 90% unstructured sparsity on a CPU without sparse kernels.

(3) Compare PTQ and QAT for a CNN that must run on an MSU tablet: when is the extra training cost of QAT justified?

(4) Derive why BPE merge operations reduce average tokens per clinical note and how that reduces transformer attention cost.

(5) Sketch a LoRA adapter plan for three hospitals sharing one frozen LLM backbone. What do you store per site?

(6) Distill a 3-class teacher: write the soft-target cross-entropy with temperature T and balancing alpha against hard labels, and explain the T^2 factor.

(7) Design a random-search hyperparameter budget (20 trials) for a tabular stroke model; justify ranges for learning rate and weight decay.

(8) Clinical scenario: cloud AUC 0.91 with 2.5 s latency vs on-device AUC 0.88 with 120 ms. Which do you deploy for LVO alert support and why?

(9) Huffman practice: for the frequency table {a:50, b:20, c:15, d:10, e:5}, build the Huffman tree, list each codeword and its length, compute the average bits/symbol, compare it to the fixed-length code for five symbols, and compute the entropy H. Is your code within 0.1 bits of H?

(10) µ-law companding: explain why a logarithmic companding curve gives small-amplitude ECG or EEG samples finer effective quantization than a uniform int8 grid, and describe one situation where plain uniform int8 is adequate.

(11) Gradient checkpointing: a network stores L equally sized activation tensors. If you keep only about sqrt(L) checkpoints, how does peak activation memory scale, and what is the extra compute cost in forward passes?

(12) Federated learning: a consortium runs FedAvg across five non-IID sites. Name two failure modes that arise specifically from non-IID data, and one privacy risk that federation does NOT eliminate.
