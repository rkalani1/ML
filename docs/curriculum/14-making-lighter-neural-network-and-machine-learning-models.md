# Chapter 14. Making Lighter Neural Network and Machine Learning Models

## Opening
![Distill/prune teaching sketch (original).](../assets/figures/ml_fig_distill_prune.png)

*Distill/prune teaching sketch (original).*


Your hospital’s edge device cannot run a 7-billion-parameter model during a code stroke. Compression, distillation, and efficient architectures are deployment medicine—not just engineering fashion.


![Smaller deployed nets still need appraisal discipline (original).](../assets/figures/ml_fig_mlp.png)

*Smaller deployed nets still need appraisal discipline (original).*
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

![14.1: The Huffman code built for the wearable beat stream, whose six event codes N, S, V, A, P, Q occur 40, 25, 15, 10, 6, and](../assets/figures/ml_concept_14.1_50ca232a.png)

*Figure 14.1 — original teaching graphic.*

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

![14.2: Amplitude quantization of a smooth full-scale (plus/minus 1) signal onto a uniform 3-bit grid. Three bits give 2^3 = 8 l](../assets/figures/ml_concept_14.2_504238ed.png)

*Figure 14.2 — original teaching graphic.*

Non-uniform quantization and signal companding. Human perception and many signals have wide dynamic range. µ-law and A-law companding (classic telephony) apply a logarithmic-like transform before uniform quantization so that small amplitudes get finer effective resolution. The µ-law compressor is roughly F(x) = sign(x) ln(1+µ|x|)/ln(1+µ) for |x|<=1 (telephony uses µ=255); expand after transmission. The A-law variant is piecewise. These remind us that uniform int8 grids can waste levels on rarely used tails; learned or logarithmic quantizers sometimes help audio and sensor streams.

Quantization for neural networks. Fixed-point and integer formats replace float32 multiplies with cheaper int8/int4 operations on NPUs. Uniform affine quantization maps a real value r to an integer q by q = round(r / scale) + zero_point, with dequantization r_hat = scale * (q - zero_point). Per-tensor scales are simple; per-channel (channel-wise) scales for convolutional weights usually recover accuracy by adapting to each filter’s range. Dynamic quantization computes activation scales at runtime; static quantization calibrates scales on a representative dataset ahead of time for faster inference.

Post-Training Quantization (PTQ) converts a trained float model using calibration data without full retraining—fast but may lose accuracy on sensitive models. Quantization-Aware Training (QAT) simulates quantization during training (straight-through estimators for round operations) so weights adapt to discrete grids—more accurate, more expensive. When to quantize: after architecture and training are stable; validate calibration curves and subgroup metrics, not only top-1 accuracy. For clinical imaging, aggressive int4 weights can erase rare lesion cues; prefer mixed precision (int8 activations, higher precision sensitive layers) and pathology-stratified evaluation.

![Mixed precision: FP32/FP16/BF16 range vs digits and loss-scaling training sketch (original).](../assets/figures/ml_fig_mixed_precision.png)

*Figure — Automatic mixed precision (AMP) is not free lunch. **Left:** FP16 has a narrow dynamic range (~±6.5×10⁴) with ~3 decimal digits; BF16 keeps FP32-like range with coarser mantissa. **Right:** FP16 without loss scaling can overflow and diverge; loss-scaled FP16 or BF16 tracks FP32. Keep master weights in FP32, re-check rare-lesion metrics after AMP, and remember a faster predictor is still a predictor—not a causal mechanism.*

![Quantization bit-width vs relative size, latency proxy, and synthetic accuracy (scientific; original).](../assets/figures/ml_fig_quant_bits.png)

*Figure — Bits as a clinical constraint curve. **Left:** synthetic held-out accuracy versus relative weight storage (log scale) as bit-width falls from 32→2; a teaching clinical floor marks when the diet becomes unsafe. **Right:** the same bit-widths with relative size bars plus latency and accuracy proxies—FLOPs and byte counts are design tools, not wall-clock. Measure PTQ and QAT on the target edge device; subgroup AUROC can fall before global accuracy. Compression preserves a prediction service, not a causal claim.*

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

![Magnitude pruning: accuracy and realized FLOPs vs sparsity (synthetic; original).](../assets/figures/ml_fig_magnitude_prune.png)

*Figure — Unstructured vs structured pruning. **Left:** validation accuracy vs sparsity with a clinical floor at 0.80; structured channel pruning often loses accuracy earlier. **Right:** relative compute—unstructured zeros rarely deliver proportional FLOP/latency cuts without sparse kernels; structured pruning maps more honestly to hardware. Always measure latency on the target device and check rare pathology subgroups.*

![14.3: Magnitude pruning of a small fully-connected network (4-6-3 units, 42 weights). On the left every weight is drawn with w](../assets/figures/ml_concept_14.3_b3c32e6a.png)

*Figure 14.3 — original teaching graphic.*

Unstructured pruning can reach high sparsity in theory but needs sparse kernels or special hardware to gain latency. Structured pruning removes channels, filters, attention heads, or entire residual blocks, yielding dense smaller tensors that ordinary BLAS and conv libraries accelerate immediately.

How to identify pruning candidates. Magnitude-based pruning removes weights with smallest absolute value—simple and surprisingly strong with iterative fine-tuning. This connects to the lottery-ticket hypothesis: within a large trained network there often exists a small subnetwork that, retrained from its original initialization, matches the full model—so pruning is partly a search for that winning subnetwork rather than mere trimming. Activation-based pruning removes units that rarely activate on calibration data. Optimal Brain Damage (OBD) and Optimal Brain Surgeon (OBS) use second-order (Hessian) information to estimate the increase in loss from removing a parameter—more principled, more expensive; OBS accounts for parameter interactions via approximate inverse-Hessian updates. Taylor expansion based pruning scores the loss change using first-order gradients times weights (or higher-order approximations), ranking units by estimated impact.

A practical recipe: train to convergence; rank weights or structures by magnitude or saliency; remove a fraction s; fine-tune with a modest learning rate; validate not only average accuracy but calibration and subgroup performance (age bands, scanner vendors, transferring hospitals). In medical imaging, aggressive channel pruning can erase rare but critical features (subtle hyperdense MCA sign, small diffusion lesions); use class-wise and pathology-wise checks, not only global AUC.

Unstructured: fine-grained zeros; best compression ratio; hardest latency win on CPUs.

Structured: remove filters/neurons/heads; reliable speedups; coarser search.

OBD/OBS: Hessian-aware saliency; costly but classic theory.

Always fine-tune after pruning; one-shot pruning often over-destroys accuracy.

## 14.7 Low-Rank Adaptation (LoRA)

Full fine-tuning of large language or vision models updates all parameters and stores a separate copy per task—prohibitive when many clinical sites or tasks share a foundation model. Low-Rank Adaptation (LoRA) freezes the pretrained weights W0 and injects trainable low-rank updates: W = W0 + B A, where B is d x r, A is r x k, and rank r << min(d,k). Only A and B are trained (often with scaling alpha/r). At inference, BA can be merged into W0 so there is no extra latency, or kept separate for multi-tenant adapters.

![14.4: Low-Rank Adaptation (LoRA). The pretrained weight matrix W0 (d x d) is frozen and the per-task update is factored as a p](../assets/figures/ml_concept_14.4_07afc635.png)

*Figure 14.4 — original teaching graphic.*

![LoRA rank r vs trainable parameters and synthetic capacity for d=k=4096 (scientific; original).](../assets/figures/ml_fig_lora_rank.png)

*Figure — Rank is a capacity knob. **Left:** trainable params = r(d+k) versus full d×k; even r=64 is a tiny fraction of full fine-tune. **Right:** synthetic val loss falls then plateaus while % of full weights rises slowly. Low r can underfit rare local phenotypes; always re-check calibration and safety after adapter merge. LoRA specializes—it does not compress the frozen backbone.*

![LoRA vs full fine-tune: parameter thrift and diminishing returns in rank (synthetic; original).](../assets/figures/ml_fig_lora_vs_full.png)

*Figure — PEFT economics. **Left:** LoRA trainable fraction of one square layer vs rank r (log scale) versus 100% full FT. **Right:** synthetic task score rises then plateaus while full FT sits higher—validate rare phenotypes and safety after merge. Adapters specialize a predictor; they do not prove causal site effects.*

![KV-cache memory grows with context length while weights stay fixed (original).](../assets/figures/ml_fig_kv_cache.png)

*Figure — Decode-time memory. Weights are roughly fixed; KV cache scales with N (and layers/heads). MQA/GQA shrink caches. Long context is a systems lever—not automatic clinical benefit without retrieval hygiene.*

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

![14.5: Knowledge distillation. A large teacher's logits are passed through a temperature-scaled softmax, p_i = exp(z_i/T) / sum](../assets/figures/ml_concept_14.5_b3ee7a22.png)

*Figure 14.5 — original teaching graphic.*

![Distillation temperature T: soft-target class mass and entropy for a synthetic 3-class teacher (scientific; original).](../assets/figures/ml_fig_distill_temp.png)

*Figure — Soft targets carry dark knowledge. **Left:** softmax(z/T) at T=1 vs T=4 for logits [2.8, 1.1, −0.4]. **Right:** entropy H(p_T) rises and peak probability falls as T grows; the usual KD loss scales soft CE by T² so gradients stay comparable. Tune T and α on a validation set that matches the edge deployment distribution—not only average accuracy.*


![INT quantization: relative MSE vs synthetic accuracy by bit width (original).](../assets/figures/ml_fig_int8_quantize.png)

*Figure — Compression trade-offs. Coarser bits raise quantizer MSE while task accuracy often plateaus near INT8 on teaching curves. Recalibrate probabilities after quantize. Lighter models change **compute**, not the causal status of a prediction.*


![Unstructured vs structured sparsity patterns (synthetic weights; original).](../assets/figures/ml_fig_structured_prune.png)

*Figure — Pruning geometry. Unstructured magnitude zeros vs structured channel drops. Hardware prefers structure; always recalibrate. Compression changes compute—not the causal status of predictions.*


![Distilled student accuracy vs relative size under a fixed teacher (synthetic; original).](../assets/figures/ml_fig_distill_gap.png)

*Figure — Distillation recovers much of teacher accuracy at smaller size. Still recalibrate and re-slice. Compression is an engineering win—not a new causal claim.*


![Parameter-efficient fine-tuning: relative params vs accuracy (synthetic; original).](../assets/figures/ml_fig_peft_tradeoff.png)

*Figure — LoRA/adapters/prompts train far fewer parameters than full fine-tuning while recovering much accuracy. Still recalibrate and slice. PEFT is engineering—not a new causal claim.*


![Attention memory scaling: naive L^2 vs IO-aware sketch (original).](../assets/figures/ml_fig_attn_memory_scale.png)

*Figure — Sequence length drives memory. Engineering kernels change feasibility. Compute graphs are not disease mechanisms.*


![Tied/shared weights schematic (original).](../assets/figures/ml_fig_tied_weights.png)

*Figure — Parameter tying compresses models—not causal structure. Pred != cause without design.*


![Low-bit width vs relative error (synthetic; original).](../assets/figures/ml_fig_lowbit_error.png)

*Figure — Quantization precision trade-offs. Pred != cause without design.*


![kdsize teaching panel (original).](../assets/figures/ml_fig_kd_student_size.png)

*Figure — Teaching panel for kdsize. Pred != cause without design.*


![Cycle-34 densify scientific panel 16 (original).](../assets/figures/ml_fig_c34_15.png)

*Figure — Continuous densify panel 16. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 16 (original).](../assets/figures/ml_fig_c35_15.png)

*Figure — Continuous densify panel 16. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 16 (original).](../assets/figures/ml_fig_c36_15.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 16 (original).](../assets/figures/ml_fig_c37_15.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 16 (original).](../assets/figures/ml_fig_c38_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 16 (original).](../assets/figures/ml_fig_c39_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 16 (original).](../assets/figures/ml_fig_c40_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 16 (original).](../assets/figures/ml_fig_c41_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 16 (original).](../assets/figures/ml_fig_c42_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 16 (original).](../assets/figures/ml_fig_c43_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 16 (original).](../assets/figures/ml_fig_c44_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 16 (original).](../assets/figures/ml_fig_c45_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 16 (original).](../assets/figures/ml_fig_c46_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 16 (original).](../assets/figures/ml_fig_c47_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 16 (original).](../assets/figures/ml_fig_c48_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 16 (original).](../assets/figures/ml_fig_c49_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 16 (original).](../assets/figures/ml_fig_c50_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 16 (original).](../assets/figures/ml_fig_c51_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 16 (original).](../assets/figures/ml_fig_c52_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 16 (original).](../assets/figures/ml_fig_c53_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 16 (original).](../assets/figures/ml_fig_c54_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 16 (original).](../assets/figures/ml_fig_c55_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 16 (original).](../assets/figures/ml_fig_c56_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 16 (original).](../assets/figures/ml_fig_c57_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 16 (original).](../assets/figures/ml_fig_c58_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 16 (original).](../assets/figures/ml_fig_c59_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 16 (original).](../assets/figures/ml_fig_c60_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 16 (original).](../assets/figures/ml_fig_c61_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 16 (original).](../assets/figures/ml_fig_c62_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 16 (original).](../assets/figures/ml_fig_c63_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 16 (original).](../assets/figures/ml_fig_c64_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 16 (original).](../assets/figures/ml_fig_c65_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 16 (original).](../assets/figures/ml_fig_c66_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 16 (original).](../assets/figures/ml_fig_c67_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 16 (original).](../assets/figures/ml_fig_c68_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 16 (original).](../assets/figures/ml_fig_c69_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 16 (original).](../assets/figures/ml_fig_c70_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 16 (original).](../assets/figures/ml_fig_c71_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 16 (original).](../assets/figures/ml_fig_c72_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 16 (original).](../assets/figures/ml_fig_c73_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 16 (original).](../assets/figures/ml_fig_c74_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 16 (original).](../assets/figures/ml_fig_c75_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 16 (original).](../assets/figures/ml_fig_c76_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 16 (original).](../assets/figures/ml_fig_c77_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 16 (original).](../assets/figures/ml_fig_c78_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 16 (original).](../assets/figures/ml_fig_c79_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 16 (original).](../assets/figures/ml_fig_c80_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 16 (original).](../assets/figures/ml_fig_c81_15.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

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

![14.6: The accuracy-versus-size trade-off for the chapter's tiny ICH CNN and its compressed variants, plotted on a logarithmic ](../assets/figures/ml_concept_14.6_83a085ec.png)

*Figure 14.6 — original teaching graphic.*

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


![c82 teaching panel 15 (original).](../assets/figures/ml_fig_c82_15.png)
*Figure — Lighter models: magnitude pruning and teacher→student distillation. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 15 (original).](../assets/figures/ml_fig_c83_15.png)
*Figure — Softmax temperature softens teacher labels for distillation. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 15 (original).](../assets/figures/ml_fig_c84_15.png)
*Figure — Quantization maps continuous activations to discrete levels. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 15 (original).](../assets/figures/ml_fig_c85_15.png)
*Figure — Quantization bit-width trades size against accuracy. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 15 (original).](../assets/figures/ml_fig_c86_15.png)
*Figure — Structured pruning reduces parameters by layer. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 15 (original).](../assets/figures/ml_fig_c87_15.png)
*Figure — Compression path from full model to edge deploy. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 15 (original).](../assets/figures/ml_fig_c88_15.png)
*Figure — LoRA low-rank adapter update ΔW=BA. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 15 (original).](../assets/figures/ml_fig_c89_15.png)
*Figure — Weight sharing / tied embeddings. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 15 (original).](../assets/figures/ml_fig_c90_15.png)
*Figure — INT8 dynamic range clipping. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 15 (original).](../assets/figures/ml_fig_c91_15.png)
*Figure — Knowledge distillation temperature. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 15 (original).](../assets/figures/ml_fig_c92_15.png)
*Figure — Flash attention tiling idea. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 15 (original).](../assets/figures/ml_fig_c93_15.png)
*Figure — Speculative decoding draft-verify. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 15 (original).](../assets/figures/ml_fig_c94_15.png)
*Figure — Tokenizer BPE merge steps. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 15 (original).](../assets/figures/ml_fig_c95_15.png)
*Figure — KV-cache autoregressive reuse. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 15 (original).](../assets/figures/ml_fig_c96_15.png)
*Figure — Medusa multi-head drafting. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 15 (original).](../assets/figures/ml_fig_c97_15.png)
*Figure — Product quantization codebooks. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 15 (original).](../assets/figures/ml_fig_c98_15.png)
*Figure — Paged attention block tables. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 15 (original).](../assets/figures/ml_fig_c99_15.png)
*Figure — Lookahead decoding verify. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 15 (original).](../assets/figures/ml_fig_c100_15.png)
*Figure — GPTQ weight quantization. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 15 (original).](../assets/figures/ml_fig_c101_15.png)
*Figure — MLA multi-head latent attn. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 15 (original).](../assets/figures/ml_fig_c102_15.png)
*Figure — EAGLE draft head. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 15 (original).](../assets/figures/ml_fig_c103_15.png)
*Figure — AWQ activation-aware quant. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 15 (original).](../assets/figures/ml_fig_c104_15.png)
*Figure — GQA grouped query attention. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 15 (original).](../assets/figures/ml_fig_c105_15.png)
*Figure — Medusa-2 multi-token. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 15 (original).](../assets/figures/ml_fig_c106_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 15 (original).](../assets/figures/ml_fig_c107_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 15 (original).](../assets/figures/ml_fig_c108_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 15 (original).](../assets/figures/ml_fig_c109_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 15 (original).](../assets/figures/ml_fig_c110_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 15 (original).](../assets/figures/ml_fig_c111_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 15 (original).](../assets/figures/ml_fig_c112_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 15 (original).](../assets/figures/ml_fig_c113_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 15 (original).](../assets/figures/ml_fig_c114_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 15 (original).](../assets/figures/ml_fig_c115_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 15 (original).](../assets/figures/ml_fig_c116_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 15 (original).](../assets/figures/ml_fig_c117_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 15 (original).](../assets/figures/ml_fig_c118_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 15 (original).](../assets/figures/ml_fig_c119_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 15 (original).](../assets/figures/ml_fig_c120_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 15 (original).](../assets/figures/ml_fig_c121_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 15 (original).](../assets/figures/ml_fig_c122_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 15 (original).](../assets/figures/ml_fig_c123_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 15 (original).](../assets/figures/ml_fig_c124_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 15 (original).](../assets/figures/ml_fig_c125_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 15 (original).](../assets/figures/ml_fig_c126_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 15 (original).](../assets/figures/ml_fig_c127_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 15 (original).](../assets/figures/ml_fig_c128_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 15 (original).](../assets/figures/ml_fig_c129_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 15 (original).](../assets/figures/ml_fig_c130_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 15 (original).](../assets/figures/ml_fig_c131_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 15 (original).](../assets/figures/ml_fig_c132_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 15 (original).](../assets/figures/ml_fig_c133_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 15 (original).](../assets/figures/ml_fig_c134_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 15 (original).](../assets/figures/ml_fig_c135_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 15 (original).](../assets/figures/ml_fig_c136_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 15 (original).](../assets/figures/ml_fig_c137_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 15 (original).](../assets/figures/ml_fig_c138_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 15 (original).](../assets/figures/ml_fig_c139_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 15 (original).](../assets/figures/ml_fig_c140_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 15 (original).](../assets/figures/ml_fig_c141_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 15 (original).](../assets/figures/ml_fig_c142_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 15 (original).](../assets/figures/ml_fig_c143_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 15 (original).](../assets/figures/ml_fig_c144_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 15 (original).](../assets/figures/ml_fig_c145_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 15 (original).](../assets/figures/ml_fig_c146_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 15 (original).](../assets/figures/ml_fig_c147_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 15 (original).](../assets/figures/ml_fig_c148_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 15 (original).](../assets/figures/ml_fig_c149_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 15 (original).](../assets/figures/ml_fig_c150_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 15 (original).](../assets/figures/ml_fig_c151_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 15 (original).](../assets/figures/ml_fig_c152_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 15 (original).](../assets/figures/ml_fig_c153_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 15 (original).](../assets/figures/ml_fig_c154_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 15 (original).](../assets/figures/ml_fig_c155_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 15 (original).](../assets/figures/ml_fig_c156_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 15 (original).](../assets/figures/ml_fig_c157_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 15 (original).](../assets/figures/ml_fig_c158_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 15 (original).](../assets/figures/ml_fig_c159_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 15 (original).](../assets/figures/ml_fig_c160_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 15 (original).](../assets/figures/ml_fig_c161_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 15 (original).](../assets/figures/ml_fig_c162_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 15 (original).](../assets/figures/ml_fig_c163_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 15 (original).](../assets/figures/ml_fig_c164_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 15 (original).](../assets/figures/ml_fig_c165_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 15 (original).](../assets/figures/ml_fig_c166_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 15 (original).](../assets/figures/ml_fig_c167_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 15 (original).](../assets/figures/ml_fig_c168_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 15 (original).](../assets/figures/ml_fig_c169_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 15 (original).](../assets/figures/ml_fig_c170_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 15 (original).](../assets/figures/ml_fig_c171_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 15 (original).](../assets/figures/ml_fig_c172_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 15 (original).](../assets/figures/ml_fig_c173_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 15 (original).](../assets/figures/ml_fig_c174_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 15 (original).](../assets/figures/ml_fig_c175_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 15 (original).](../assets/figures/ml_fig_c176_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 15 (original).](../assets/figures/ml_fig_c177_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 15 (original).](../assets/figures/ml_fig_c178_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 15 (original).](../assets/figures/ml_fig_c179_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 15 (original).](../assets/figures/ml_fig_c180_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 15 (original).](../assets/figures/ml_fig_c181_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 15 (original).](../assets/figures/ml_fig_c182_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 15 (original).](../assets/figures/ml_fig_c183_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 15 (original).](../assets/figures/ml_fig_c184_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 15 (original).](../assets/figures/ml_fig_c185_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 15 (original).](../assets/figures/ml_fig_c186_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 15 (original).](../assets/figures/ml_fig_c187_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 15 (original).](../assets/figures/ml_fig_c188_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 15 (original).](../assets/figures/ml_fig_c189_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 15 (original).](../assets/figures/ml_fig_c190_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 15 (original).](../assets/figures/ml_fig_c191_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 15 (original).](../assets/figures/ml_fig_c192_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 15 (original).](../assets/figures/ml_fig_c193_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 15 (original).](../assets/figures/ml_fig_c194_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 15 (original).](../assets/figures/ml_fig_c195_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 15 (original).](../assets/figures/ml_fig_c196_15.png)
*Figure — SparseGPT pruning. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 15 (original).](../assets/figures/ml_fig_c197_15.png)
*Figure — Wanda prune metric. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 15 (original).](../assets/figures/ml_fig_c198_15.png)
*Figure — SmoothQuant scales. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 15 (original).](../assets/figures/ml_fig_c199_15.png)
*Figure — QLoRA NF4 storage. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 15 (original).](../assets/figures/ml_fig_c200_15.png)
*Figure — Speculative draft tree. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 15 (original).](../assets/figures/ml_fig_c201_15.png)
*Figure — Weight tying embed and head. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 15 (original).](../assets/figures/ml_fig_c202_15.png)
*Figure — KV cache decode memory trade. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 15 (original).](../assets/figures/ml_fig_c203_15.png)
*Figure — LoRA low-rank adapter insert. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 15 (original).](../assets/figures/ml_fig_c204_15.png)
*Figure — Structured channel prune mask. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 15 (original).](../assets/figures/ml_fig_c205_15.png)
*Figure — Uniform quantization bins. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 15 (original).](../assets/figures/ml_fig_c206_15.png)
*Figure — Distillation teacher to student. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 15 (original).](../assets/figures/ml_fig_c207_15.png)
*Figure — PTQ int8 percentile calibration. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 15 (original).](../assets/figures/ml_fig_c208_15.png)
*Figure — Low-rank SVD weight compress. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 15 (original).](../assets/figures/ml_fig_c209_15.png)
*Figure — SparseGPT weight mask pattern. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 15 (original).](../assets/figures/ml_fig_c210_15.png)
*Figure — GPTQ Hessian-aware quant error. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 15 (original).](../assets/figures/ml_fig_c211_15.png)
*Figure — AWQ activation-aware scales. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 15 (original).](../assets/figures/ml_fig_c212_15.png)
*Figure — Speculative decoding draft verify. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 15 (original).](../assets/figures/ml_fig_c213_15.png)
*Figure — Low-rank layer factorization. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 15 (original).](../assets/figures/ml_fig_c214_15.png)
*Figure — Transformer KV cache append. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 15 (original).](../assets/figures/ml_fig_c215_15.png)
*Figure — Lottery ticket prune rewind. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 15 (original).](../assets/figures/ml_fig_c216_15.png)
*Figure — SmoothQuant act-to-weight migrate. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 15 (original).](../assets/figures/ml_fig_c217_15.png)
*Figure — Depth pruning accuracy size. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 15 (original).](../assets/figures/ml_fig_c218_15.png)
*Figure — Bit-width size accuracy curve. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 15 (original).](../assets/figures/ml_fig_c219_15.png)
*Figure — HQQ half-quadratic quant. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 15 (original).](../assets/figures/ml_fig_c220_15.png)
*Figure — Sparse attention pattern mask. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 15 (original).](../assets/figures/ml_fig_c221_15.png)
*Figure — GPTQ Hessian-aware quant error. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 15 (original).](../assets/figures/ml_fig_c222_15.png)
*Figure — AWQ activation-salient channels. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 15 (original).](../assets/figures/ml_fig_c223_15.png)
*Figure — SmoothQuant activation migrate. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 15 (original).](../assets/figures/ml_fig_c224_15.png)
*Figure — QLoRA NF4 quantile bins. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 15 (original).](../assets/figures/ml_fig_c225_15.png)
*Figure — Speculative decoding verify path. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 15 (original).](../assets/figures/ml_fig_c226_15.png)
*Figure — GGUF quant perplexity curve. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 15 (original).](../assets/figures/ml_fig_c227_15.png)
*Figure — FlashAttention SRAM tiles. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 15 (original).](../assets/figures/ml_fig_c228_15.png)
*Figure — KV-cache quant memory trade. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 15 (original).](../assets/figures/ml_fig_c229_15.png)
*Figure — Mixture-of-Depths capacity bars. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 15 (original).](../assets/figures/ml_fig_c230_15.png)
*Figure — AWQ per-channel scales stem. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 15 (original).](../assets/figures/ml_fig_c231_15.png)
*Figure — LoRA rank quality tradeoff. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 15 (original).](../assets/figures/ml_fig_c232_15.png)
*Figure — bitsandbytes NF4 histogram. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 15 (original).](../assets/figures/ml_fig_c233_15.png)
*Figure — INT8 scale and zero-point. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 15 (original).](../assets/figures/ml_fig_c234_15.png)
*Figure — Column-wise quant error. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 15 (original).](../assets/figures/ml_fig_c235_15.png)
*Figure — Per-channel quant bars. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 15 (original).](../assets/figures/ml_fig_c236_15.png)
*Figure — Group-wise quant error bars. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 15 (original).](../assets/figures/ml_fig_c237_15.png)
*Figure — Group-wise bit allocation. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 15 (original).](../assets/figures/ml_fig_c238_15.png)
*Figure — AWQ protected channel bars. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 15 (original).](../assets/figures/ml_fig_c239_15.png)
*Figure — KV cache bit bars. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 15 (original).](../assets/figures/ml_fig_c240_15.png)
*Figure — SmoothQuant migrate bars. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 15 (original).](../assets/figures/ml_fig_c241_15.png)
*Figure — LoRA rank adapter bars. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 15 (original).](../assets/figures/ml_fig_c242_15.png)
*Figure — GPTQ layer error bars. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 15 (original).](../assets/figures/ml_fig_c243_15.png)
*Figure — QLoRA NF4 quant bars. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 15 (original).](../assets/figures/ml_fig_c244_15.png)
*Figure — AWQ scale protect bars. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 15 (original).](../assets/figures/ml_fig_c245_15.png)
*Figure — DoRA magnitude adapter bars. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 15 (original).](../assets/figures/ml_fig_c246_15.png)
*Figure — SmoothQuant migrate bars. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 15 (original).](../assets/figures/ml_fig_c247_15.png)
*Figure — GaLore low-rank proj bars. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 15 (original).](../assets/figures/ml_fig_c248_15.png)
*Figure — AWQ channel protect bars. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 15 (original).](../assets/figures/ml_fig_c249_15.png)
*Figure — VeRA shared adapter bars. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 15 (original).](../assets/figures/ml_fig_c250_15.png)
*Figure — GPTQ act-order bars. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 15 (original).](../assets/figures/ml_fig_c251_15.png)
*Figure — LoRA+ LR ratio bars. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 15 (original).](../assets/figures/ml_fig_c252_15.png)
*Figure — AWQ zero-point bars. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 15 (original).](../assets/figures/ml_fig_c253_15.png)
*Figure — QLoRA double-quant bars. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 15 (original).](../assets/figures/ml_fig_c254_15.png)
*Figure — SmoothQuant alpha bars. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 15 (original).](../assets/figures/ml_fig_c255_15.png)
*Figure — DoRA decompose bars. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 15 (original).](../assets/figures/ml_fig_c256_15.png)
*Figure — GPTQ groupsize bars. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 15 (original).](../assets/figures/ml_fig_c257_15.png)
*Figure — Group-query trade c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 15 (original).](../assets/figures/ml_fig_c258_15.png)
*Figure — SmoothQuant migrate bars c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 15 (original).](../assets/figures/ml_fig_c259_15.png)
*Figure — Prune magnitude bars c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 15 (original).](../assets/figures/ml_fig_c260_15.png)
*Figure — Quant int8 error bars c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 15 (original).](../assets/figures/ml_fig_c261_15.png)
*Figure — Distill temperature path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 15 (original).](../assets/figures/ml_fig_c262_15.png)
*Figure — Knowledge transfer path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 15 (original).](../assets/figures/ml_fig_c263_15.png)
*Figure — LoRA rank bars c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 15 (original).](../assets/figures/ml_fig_c264_15.png)
*Figure — QLoRA NF4 bars c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 15 (original).](../assets/figures/ml_fig_c265_15.png)
*Figure — Adapter insert path c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 15 (original).](../assets/figures/ml_fig_c266_15.png)
*Figure — Weight share residual c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 15 (original).](../assets/figures/ml_fig_c267_15.png)
*Figure — Early exit path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 15 (original).](../assets/figures/ml_fig_c268_15.png)
*Figure — Token prune path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 15 (original).](../assets/figures/ml_fig_c269_15.png)
*Figure — KV cache size bars c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 15 (original).](../assets/figures/ml_fig_c270_15.png)
*Figure — Speculative draft path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 15 (original).](../assets/figures/ml_fig_c271_15.png)
*Figure — FlashAttention trade c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 15 (original).](../assets/figures/ml_fig_c272_15.png)
*Figure — Activation ckpt trade c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 15 (original).](../assets/figures/ml_fig_c273_15.png)
*Figure — Group-query trade c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 15 (original).](../assets/figures/ml_fig_c274_15.png)
*Figure — SmoothQuant migrate bars c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 15 (original).](../assets/figures/ml_fig_c275_15.png)
*Figure — Prune magnitude bars c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 15 (original).](../assets/figures/ml_fig_c276_15.png)
*Figure — Quant int8 error bars c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 15 (original).](../assets/figures/ml_fig_c277_15.png)
*Figure — Distill temperature path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 15 (original).](../assets/figures/ml_fig_c278_15.png)
*Figure — Knowledge transfer path c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 15 (original).](../assets/figures/ml_fig_c279_15.png)
*Figure — LoRA rank bars c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 15 (original).](../assets/figures/ml_fig_c280_15.png)
*Figure — QLoRA NF4 bars c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 15 (original).](../assets/figures/ml_fig_c281_15.png)
*Figure — Adapter insert path c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 15 (original).](../assets/figures/ml_fig_c282_15.png)
*Figure — Weight share residual c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 15 (original).](../assets/figures/ml_fig_c283_15.png)
*Figure — Early exit path c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 15 (original).](../assets/figures/ml_fig_c284_15.png)
*Figure — Token prune path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 15 (original).](../assets/figures/ml_fig_c285_15.png)
*Figure — KV cache size bars c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 15 (original).](../assets/figures/ml_fig_c286_15.png)
*Figure — Speculative draft path c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 15 (original).](../assets/figures/ml_fig_c287_15.png)
*Figure — FlashAttention trade c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 15 (original).](../assets/figures/ml_fig_c288_15.png)
*Figure — Activation ckpt trade c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 15 (original).](../assets/figures/ml_fig_c289_15.png)
*Figure — Group-query trade c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 15 (original).](../assets/figures/ml_fig_c290_15.png)
*Figure — SmoothQuant migrate bars c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 15 (original).](../assets/figures/ml_fig_c291_15.png)
*Figure — Prune magnitude bars c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 15 (original).](../assets/figures/ml_fig_c292_15.png)
*Figure — Quant int8 error bars c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 15 (original).](../assets/figures/ml_fig_c293_15.png)
*Figure — Distill temperature path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 15 (original).](../assets/figures/ml_fig_c294_15.png)
*Figure — Knowledge transfer path c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 15 (original).](../assets/figures/ml_fig_c295_15.png)
*Figure — LoRA rank bars c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 15 (original).](../assets/figures/ml_fig_c296_15.png)
*Figure — QLoRA NF4 bars c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 15 (original).](../assets/figures/ml_fig_c297_15.png)
*Figure — Adapter insert path c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 15 (original).](../assets/figures/ml_fig_c298_15.png)
*Figure — Weight share residual c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 15 (original).](../assets/figures/ml_fig_c299_15.png)
*Figure — Early exit path c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 15 (original).](../assets/figures/ml_fig_c300_15.png)
*Figure — Token prune path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 15 (original).](../assets/figures/ml_fig_c301_15.png)
*Figure — KV cache size bars c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 15 (original).](../assets/figures/ml_fig_c302_15.png)
*Figure — Speculative draft path c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 15 (original).](../assets/figures/ml_fig_c303_15.png)
*Figure — FlashAttention trade c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 15 (original).](../assets/figures/ml_fig_c304_15.png)
*Figure — Activation ckpt trade c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 15 (original).](../assets/figures/ml_fig_c305_15.png)
*Figure — Group-query trade c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 15 (original).](../assets/figures/ml_fig_c306_15.png)
*Figure — SmoothQuant migrate bars c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 15 (original).](../assets/figures/ml_fig_c307_15.png)
*Figure — Prune magnitude bars c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 15 (original).](../assets/figures/ml_fig_c308_15.png)
*Figure — Quant int8 error bars c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 15 (original).](../assets/figures/ml_fig_c309_15.png)
*Figure — Distill temperature path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 15 (original).](../assets/figures/ml_fig_c310_15.png)
*Figure — Knowledge transfer path c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 15 (original).](../assets/figures/ml_fig_c311_15.png)
*Figure — LoRA rank bars c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 15 (original).](../assets/figures/ml_fig_c312_15.png)
*Figure — QLoRA NF4 bars c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 15 (original).](../assets/figures/ml_fig_c313_15.png)
*Figure — Adapter insert path c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 15 (original).](../assets/figures/ml_fig_c314_15.png)
*Figure — Weight share residual c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 15 (original).](../assets/figures/ml_fig_c315_15.png)
*Figure — Early exit path c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 15 (original).](../assets/figures/ml_fig_c316_15.png)
*Figure — Token prune path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 15 (original).](../assets/figures/ml_fig_c317_15.png)
*Figure — KV cache size bars c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 15 (original).](../assets/figures/ml_fig_c318_15.png)
*Figure — Speculative draft path c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 15 (original).](../assets/figures/ml_fig_c319_15.png)
*Figure — FlashAttention trade c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 15 (original).](../assets/figures/ml_fig_c320_15.png)
*Figure — Activation ckpt trade c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 15 (original).](../assets/figures/ml_fig_c321_15.png)
*Figure — Group-query trade c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 15 (original).](../assets/figures/ml_fig_c322_15.png)
*Figure — SmoothQuant migrate bars c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 15 (original).](../assets/figures/ml_fig_c323_15.png)
*Figure — Prune magnitude bars c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 15 (original).](../assets/figures/ml_fig_c324_15.png)
*Figure — Quant int8 error bars c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 15 (original).](../assets/figures/ml_fig_c325_15.png)
*Figure — Distill temperature path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 15 (original).](../assets/figures/ml_fig_c326_15.png)
*Figure — Knowledge transfer path c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 15 (original).](../assets/figures/ml_fig_c327_15.png)
*Figure — LoRA rank bars c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 15 (original).](../assets/figures/ml_fig_c328_15.png)
*Figure — QLoRA NF4 bars c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 15 (original).](../assets/figures/ml_fig_c329_15.png)
*Figure — Adapter insert path c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 15 (original).](../assets/figures/ml_fig_c330_15.png)
*Figure — Weight share residual c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 15 (original).](../assets/figures/ml_fig_c331_15.png)
*Figure — Early exit path c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 15 (original).](../assets/figures/ml_fig_c332_15.png)
*Figure — Token prune path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 15 (original).](../assets/figures/ml_fig_c333_15.png)
*Figure — KV cache size bars c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 15 (original).](../assets/figures/ml_fig_c334_15.png)
*Figure — Speculative draft path c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 15 (original).](../assets/figures/ml_fig_c335_15.png)
*Figure — FlashAttention trade c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 15 (original).](../assets/figures/ml_fig_c336_15.png)
*Figure — Activation ckpt trade c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 15 (original).](../assets/figures/ml_fig_c337_15.png)
*Figure — Group-query trade c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 15 (original).](../assets/figures/ml_fig_c338_15.png)
*Figure — SmoothQuant migrate bars c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 15 (original).](../assets/figures/ml_fig_c339_15.png)
*Figure — Prune magnitude bars c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 15 (original).](../assets/figures/ml_fig_c340_15.png)
*Figure — Quant int8 error bars c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 15 (original).](../assets/figures/ml_fig_c341_15.png)
*Figure — Distill temperature path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 15 (original).](../assets/figures/ml_fig_c342_15.png)
*Figure — Knowledge transfer path c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 15 (original).](../assets/figures/ml_fig_c343_15.png)
*Figure — LoRA rank bars c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 15 (original).](../assets/figures/ml_fig_c344_15.png)
*Figure — QLoRA NF4 bars c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 15 (original).](../assets/figures/ml_fig_c345_15.png)
*Figure — Adapter insert path c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 15 (original).](../assets/figures/ml_fig_c346_15.png)
*Figure — Weight share residual c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 15 (original).](../assets/figures/ml_fig_c347_15.png)
*Figure — Early exit path c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 15 (original).](../assets/figures/ml_fig_c348_15.png)
*Figure — Token prune path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 15 (original).](../assets/figures/ml_fig_c349_15.png)
*Figure — KV cache size bars c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 15 (original).](../assets/figures/ml_fig_c350_15.png)
*Figure — Speculative draft path c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 15 (original).](../assets/figures/ml_fig_c351_15.png)
*Figure — FlashAttention trade c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 15 (original).](../assets/figures/ml_fig_c352_15.png)
*Figure — Activation ckpt trade c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 15 (original).](../assets/figures/ml_fig_c353_15.png)
*Figure — Group-query trade c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 15 (original).](../assets/figures/ml_fig_c354_15.png)
*Figure — SmoothQuant migrate bars c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 15 (original).](../assets/figures/ml_fig_c355_15.png)
*Figure — Prune magnitude bars c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 15 (original).](../assets/figures/ml_fig_c356_15.png)
*Figure — Quant int8 error bars c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 15 (original).](../assets/figures/ml_fig_c357_15.png)
*Figure — Distill temperature path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 15 (original).](../assets/figures/ml_fig_c358_15.png)
*Figure — Knowledge transfer path c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 15 (original).](../assets/figures/ml_fig_c359_15.png)
*Figure — LoRA rank bars c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 15 (original).](../assets/figures/ml_fig_c360_15.png)
*Figure — QLoRA NF4 bars c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 15 (original).](../assets/figures/ml_fig_c361_15.png)
*Figure — Adapter insert path c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 15 (original).](../assets/figures/ml_fig_c362_15.png)
*Figure — Weight share residual c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 15 (original).](../assets/figures/ml_fig_c363_15.png)
*Figure — Early exit path c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 15 (original).](../assets/figures/ml_fig_c364_15.png)
*Figure — Token prune path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 15 (original).](../assets/figures/ml_fig_c365_15.png)
*Figure — KV cache size bars c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 15 (original).](../assets/figures/ml_fig_c366_15.png)
*Figure — Speculative draft path c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 15 (original).](../assets/figures/ml_fig_c367_15.png)
*Figure — FlashAttention trade c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 15 (original).](../assets/figures/ml_fig_c368_15.png)
*Figure — Activation ckpt trade c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 15 (original).](../assets/figures/ml_fig_c369_15.png)
*Figure — Group-query trade c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 15 (original).](../assets/figures/ml_fig_c370_15.png)
*Figure — SmoothQuant migrate bars c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 15 (original).](../assets/figures/ml_fig_c371_15.png)
*Figure — Prune magnitude bars c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 15 (original).](../assets/figures/ml_fig_c372_15.png)
*Figure — Quant int8 error bars c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 15 (original).](../assets/figures/ml_fig_c373_15.png)
*Figure — Distill temperature path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 15 (original).](../assets/figures/ml_fig_c374_15.png)
*Figure — Knowledge transfer path c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 15 (original).](../assets/figures/ml_fig_c375_15.png)
*Figure — LoRA rank bars c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 15 (original).](../assets/figures/ml_fig_c376_15.png)
*Figure — QLoRA NF4 bars c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 15 (original).](../assets/figures/ml_fig_c377_15.png)
*Figure — Adapter insert path c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 15 (original).](../assets/figures/ml_fig_c378_15.png)
*Figure — Weight share residual c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 15 (original).](../assets/figures/ml_fig_c379_15.png)
*Figure — Early exit path c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 15 (original).](../assets/figures/ml_fig_c380_15.png)
*Figure — Token prune path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 15 (original).](../assets/figures/ml_fig_c381_15.png)
*Figure — KV cache size bars c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 15 (original).](../assets/figures/ml_fig_c382_15.png)
*Figure — Speculative draft path c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 15 (original).](../assets/figures/ml_fig_c383_15.png)
*Figure — FlashAttention trade c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 15 (original).](../assets/figures/ml_fig_c384_15.png)
*Figure — Activation ckpt trade c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 15 (original).](../assets/figures/ml_fig_c385_15.png)
*Figure — Group-query trade c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 15 (original).](../assets/figures/ml_fig_c386_15.png)
*Figure — SmoothQuant migrate bars c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 15 (original).](../assets/figures/ml_fig_c387_15.png)
*Figure — Prune magnitude bars c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 15 (original).](../assets/figures/ml_fig_c388_15.png)
*Figure — Quant int8 error bars c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 15 (original).](../assets/figures/ml_fig_c389_15.png)
*Figure — Distill temperature path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 15 (original).](../assets/figures/ml_fig_c390_15.png)
*Figure — Knowledge transfer path c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 15 (original).](../assets/figures/ml_fig_c391_15.png)
*Figure — LoRA rank bars c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 15 (original).](../assets/figures/ml_fig_c392_15.png)
*Figure — QLoRA NF4 bars c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 15 (original).](../assets/figures/ml_fig_c393_15.png)
*Figure — Adapter insert path c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 15 (original).](../assets/figures/ml_fig_c394_15.png)
*Figure — Weight share residual c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 15 (original).](../assets/figures/ml_fig_c395_15.png)
*Figure — Early exit path c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 15 (original).](../assets/figures/ml_fig_c396_15.png)
*Figure — Token prune path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 15 (original).](../assets/figures/ml_fig_c397_15.png)
*Figure — KV cache size bars c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 15 (original).](../assets/figures/ml_fig_c398_15.png)
*Figure — Speculative draft path c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 15 (original).](../assets/figures/ml_fig_c399_15.png)
*Figure — FlashAttention trade c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 15 (original).](../assets/figures/ml_fig_c400_15.png)
*Figure — Activation ckpt trade c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 15 (original).](../assets/figures/ml_fig_c401_15.png)
*Figure — Group-query trade c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 15 (original).](../assets/figures/ml_fig_c402_15.png)
*Figure — SmoothQuant migrate bars c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 15 (original).](../assets/figures/ml_fig_c403_15.png)
*Figure — Prune magnitude bars c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 15 (original).](../assets/figures/ml_fig_c404_15.png)
*Figure — Quant int8 error bars c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 15 (original).](../assets/figures/ml_fig_c405_15.png)
*Figure — Distill temperature path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 15 (original).](../assets/figures/ml_fig_c406_15.png)
*Figure — Knowledge transfer path c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 15 (original).](../assets/figures/ml_fig_c407_15.png)
*Figure — LoRA rank bars c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 15 (original).](../assets/figures/ml_fig_c408_15.png)
*Figure — QLoRA NF4 bars c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 15 (original).](../assets/figures/ml_fig_c409_15.png)
*Figure — Adapter insert path c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 15 (original).](../assets/figures/ml_fig_c410_15.png)
*Figure — Weight share residual c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 15 (original).](../assets/figures/ml_fig_c411_15.png)
*Figure — Early exit path c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 15 (original).](../assets/figures/ml_fig_c412_15.png)
*Figure — Token prune path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 15 (original).](../assets/figures/ml_fig_c413_15.png)
*Figure — KV cache size bars c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 15 (original).](../assets/figures/ml_fig_c414_15.png)
*Figure — Speculative draft path c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 15 (original).](../assets/figures/ml_fig_c415_15.png)
*Figure — FlashAttention trade c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 15 (original).](../assets/figures/ml_fig_c416_15.png)
*Figure — Activation ckpt trade c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 15 (original).](../assets/figures/ml_fig_c417_15.png)
*Figure — Group-query trade c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 15 (original).](../assets/figures/ml_fig_c418_15.png)
*Figure — SmoothQuant migrate bars c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 15 (original).](../assets/figures/ml_fig_c419_15.png)
*Figure — Prune magnitude bars c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 15 (original).](../assets/figures/ml_fig_c420_15.png)
*Figure — Quant int8 error bars c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 15 (original).](../assets/figures/ml_fig_c421_15.png)
*Figure — Distill temperature path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 15 (original).](../assets/figures/ml_fig_c422_15.png)
*Figure — Knowledge transfer path c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 15 (original).](../assets/figures/ml_fig_c423_15.png)
*Figure — LoRA rank bars c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 15 (original).](../assets/figures/ml_fig_c424_15.png)
*Figure — QLoRA NF4 bars c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 15 (original).](../assets/figures/ml_fig_c425_15.png)
*Figure — Adapter insert path c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 15 (original).](../assets/figures/ml_fig_c426_15.png)
*Figure — Weight share residual c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 15 (original).](../assets/figures/ml_fig_c427_15.png)
*Figure — Early exit path c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 15 (original).](../assets/figures/ml_fig_c428_15.png)
*Figure — Token prune path c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 15 (original).](../assets/figures/ml_fig_c429_15.png)
*Figure — KV cache size bars c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 15 (original).](../assets/figures/ml_fig_c430_15.png)
*Figure — Speculative draft path c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 15 (original).](../assets/figures/ml_fig_c431_15.png)
*Figure — FlashAttention trade c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 15 (original).](../assets/figures/ml_fig_c432_15.png)
*Figure — Activation ckpt trade c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 15 (original).](../assets/figures/ml_fig_c433_15.png)
*Figure — Group-query trade c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 15 (original).](../assets/figures/ml_fig_c434_15.png)
*Figure — SmoothQuant migrate bars c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 15 (original).](../assets/figures/ml_fig_c435_15.png)
*Figure — Prune magnitude bars c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 15 (original).](../assets/figures/ml_fig_c436_15.png)
*Figure — Quant int8 error bars c436. Synthetic teaching geometry—not a causal claim.*


![c437 teaching panel 15 (original).](../assets/figures/ml_fig_c437_15.png)
*Figure — Distill temperature path c437. Synthetic teaching geometry—not a causal claim.*


![c438 teaching panel 15 (original).](../assets/figures/ml_fig_c438_15.png)
*Figure — Knowledge transfer path c438. Synthetic teaching geometry—not a causal claim.*


![c439 teaching panel 15 (original).](../assets/figures/ml_fig_c439_15.png)
*Figure — LoRA rank bars c439. Synthetic teaching geometry—not a causal claim.*


![c440 teaching panel 15 (original).](../assets/figures/ml_fig_c440_15.png)
*Figure — QLoRA NF4 bars c440. Synthetic teaching geometry—not a causal claim.*


![c441 teaching panel 15 (original).](../assets/figures/ml_fig_c441_15.png)
*Figure — Adapter insert path c441. Synthetic teaching geometry—not a causal claim.*


![c442 teaching panel 15 (original).](../assets/figures/ml_fig_c442_15.png)
*Figure — Weight share residual c442. Synthetic teaching geometry—not a causal claim.*


![c443 teaching panel 15 (original).](../assets/figures/ml_fig_c443_15.png)
*Figure — Early exit path c443. Synthetic teaching geometry—not a causal claim.*


![c444 teaching panel 15 (original).](../assets/figures/ml_fig_c444_15.png)
*Figure — Token prune path c444. Synthetic teaching geometry—not a causal claim.*


![c445 teaching panel 15 (original).](../assets/figures/ml_fig_c445_15.png)
*Figure — KV cache size bars c445. Synthetic teaching geometry—not a causal claim.*


![c446 teaching panel 15 (original).](../assets/figures/ml_fig_c446_15.png)
*Figure — Speculative draft path c446. Synthetic teaching geometry—not a causal claim.*


![c447 teaching panel 15 (original).](../assets/figures/ml_fig_c447_15.png)
*Figure — FlashAttention trade c447. Synthetic teaching geometry—not a causal claim.*


![c448 teaching panel 15 (original).](../assets/figures/ml_fig_c448_15.png)
*Figure — Activation ckpt trade c448. Synthetic teaching geometry—not a causal claim.*


![c449 teaching panel 15 (original).](../assets/figures/ml_fig_c449_15.png)
*Figure — Group-query trade c449. Synthetic teaching geometry—not a causal claim.*


![c450 teaching panel 15 (original).](../assets/figures/ml_fig_c450_15.png)
*Figure — SmoothQuant migrate bars c450. Synthetic teaching geometry—not a causal claim.*


![c451 teaching panel 15 (original).](../assets/figures/ml_fig_c451_15.png)
*Figure — Prune magnitude bars c451. Synthetic teaching geometry—not a causal claim.*


![c452 teaching panel 15 (original).](../assets/figures/ml_fig_c452_15.png)
*Figure — Quant int8 error bars c452. Synthetic teaching geometry—not a causal claim.*


![c453 teaching panel 15 (original).](../assets/figures/ml_fig_c453_15.png)
*Figure — Distill temperature path c453. Synthetic teaching geometry—not a causal claim.*


![c454 teaching panel 15 (original).](../assets/figures/ml_fig_c454_15.png)
*Figure — Knowledge transfer path c454. Synthetic teaching geometry—not a causal claim.*


![c455 teaching panel 15 (original).](../assets/figures/ml_fig_c455_15.png)
*Figure — LoRA rank bars c455. Synthetic teaching geometry—not a causal claim.*


![c456 teaching panel 15 (original).](../assets/figures/ml_fig_c456_15.png)
*Figure — QLoRA NF4 bars c456. Synthetic teaching geometry—not a causal claim.*


![c457 teaching panel 15 (original).](../assets/figures/ml_fig_c457_15.png)
*Figure — Adapter insert path c457. Synthetic teaching geometry—not a causal claim.*


![c458 teaching panel 15 (original).](../assets/figures/ml_fig_c458_15.png)
*Figure — Weight share residual c458. Synthetic teaching geometry—not a causal claim.*


![c459 teaching panel 15 (original).](../assets/figures/ml_fig_c459_15.png)
*Figure — Early exit path c459. Synthetic teaching geometry—not a causal claim.*


![c460 teaching panel 15 (original).](../assets/figures/ml_fig_c460_15.png)
*Figure — Token prune path c460. Synthetic teaching geometry—not a causal claim.*


![c461 teaching panel 15 (original).](../assets/figures/ml_fig_c461_15.png)
*Figure — KV cache size bars c461. Synthetic teaching geometry—not a causal claim.*


![c462 teaching panel 15 (original).](../assets/figures/ml_fig_c462_15.png)
*Figure — Speculative draft path c462. Synthetic teaching geometry—not a causal claim.*


![c463 teaching panel 15 (original).](../assets/figures/ml_fig_c463_15.png)
*Figure — FlashAttention trade c463. Synthetic teaching geometry—not a causal claim.*


![c464 teaching panel 15 (original).](../assets/figures/ml_fig_c464_15.png)
*Figure — Activation ckpt trade c464. Synthetic teaching geometry—not a causal claim.*


![c465 teaching panel 15 (original).](../assets/figures/ml_fig_c465_15.png)
*Figure — Group-query trade c465. Synthetic teaching geometry—not a causal claim.*


![c466 teaching panel 15 (original).](../assets/figures/ml_fig_c466_15.png)
*Figure — SmoothQuant migrate bars c466. Synthetic teaching geometry—not a causal claim.*


![c467 teaching panel 15 (original).](../assets/figures/ml_fig_c467_15.png)
*Figure — Prune magnitude bars c467. Synthetic teaching geometry—not a causal claim.*


![c468 teaching panel 15 (original).](../assets/figures/ml_fig_c468_15.png)
*Figure — Quant int8 error bars c468. Synthetic teaching geometry—not a causal claim.*


![c469 teaching panel 15 (original).](../assets/figures/ml_fig_c469_15.png)
*Figure — Distill temperature path c469. Synthetic teaching geometry—not a causal claim.*


![c470 teaching panel 15 (original).](../assets/figures/ml_fig_c470_15.png)
*Figure — Knowledge transfer path c470. Synthetic teaching geometry—not a causal claim.*


![c471 teaching panel 15 (original).](../assets/figures/ml_fig_c471_15.png)
*Figure — LoRA rank bars c471. Synthetic teaching geometry—not a causal claim.*


![c472 teaching panel 15 (original).](../assets/figures/ml_fig_c472_15.png)
*Figure — QLoRA NF4 bars c472. Synthetic teaching geometry—not a causal claim.*


![c473 teaching panel 15 (original).](../assets/figures/ml_fig_c473_15.png)
*Figure — Adapter insert path c473. Synthetic teaching geometry—not a causal claim.*


![c474 teaching panel 15 (original).](../assets/figures/ml_fig_c474_15.png)
*Figure — Weight share residual c474. Synthetic teaching geometry—not a causal claim.*


![c475 teaching panel 15 (original).](../assets/figures/ml_fig_c475_15.png)
*Figure — Early exit path c475. Synthetic teaching geometry—not a causal claim.*


![c476 teaching panel 15 (original).](../assets/figures/ml_fig_c476_15.png)
*Figure — Token prune path c476. Synthetic teaching geometry—not a causal claim.*


![c477 teaching panel 15 (original).](../assets/figures/ml_fig_c477_15.png)
*Figure — KV cache size bars c477. Synthetic teaching geometry—not a causal claim.*


![c478 teaching panel 15 (original).](../assets/figures/ml_fig_c478_15.png)
*Figure — Speculative draft path c478. Synthetic teaching geometry—not a causal claim.*


![c479 teaching panel 15 (original).](../assets/figures/ml_fig_c479_15.png)
*Figure — FlashAttention trade c479. Synthetic teaching geometry—not a causal claim.*


![c480 teaching panel 15 (original).](../assets/figures/ml_fig_c480_15.png)
*Figure — Activation ckpt trade c480. Synthetic teaching geometry—not a causal claim.*


![c481 teaching panel 15 (original).](../assets/figures/ml_fig_c481_15.png)
*Figure — Group-query trade c481. Synthetic teaching geometry—not a causal claim.*


![c482 teaching panel 15 (original).](../assets/figures/ml_fig_c482_15.png)
*Figure — SmoothQuant migrate bars c482. Synthetic teaching geometry—not a causal claim.*


![c483 teaching panel 15 (original).](../assets/figures/ml_fig_c483_15.png)
*Figure — Prune magnitude bars c483. Synthetic teaching geometry—not a causal claim.*


![c484 teaching panel 15 (original).](../assets/figures/ml_fig_c484_15.png)
*Figure — Quant int8 error bars c484. Synthetic teaching geometry—not a causal claim.*


![c485 teaching panel 15 (original).](../assets/figures/ml_fig_c485_15.png)
*Figure — Distill temperature path c485. Synthetic teaching geometry—not a causal claim.*


![c486 teaching panel 15 (original).](../assets/figures/ml_fig_c486_15.png)
*Figure — Knowledge transfer path c486. Synthetic teaching geometry—not a causal claim.*


![c487 teaching panel 15 (original).](../assets/figures/ml_fig_c487_15.png)
*Figure — LoRA rank bars c487. Synthetic teaching geometry—not a causal claim.*


![c488 teaching panel 15 (original).](../assets/figures/ml_fig_c488_15.png)
*Figure — QLoRA NF4 bars c488. Synthetic teaching geometry—not a causal claim.*


![c489 teaching panel 15 (original).](../assets/figures/ml_fig_c489_15.png)
*Figure — Adapter insert path c489. Synthetic teaching geometry—not a causal claim.*


![c490 teaching panel 15 (original).](../assets/figures/ml_fig_c490_15.png)
*Figure — Weight share residual c490. Synthetic teaching geometry—not a causal claim.*


![c491 teaching panel 15 (original).](../assets/figures/ml_fig_c491_15.png)
*Figure — Early exit path c491. Synthetic teaching geometry—not a causal claim.*


![c492 teaching panel 15 (original).](../assets/figures/ml_fig_c492_15.png)
*Figure — Token prune path c492. Synthetic teaching geometry—not a causal claim.*


![c493 teaching panel 15 (original).](../assets/figures/ml_fig_c493_15.png)
*Figure — KV cache size bars c493. Synthetic teaching geometry—not a causal claim.*


![c494 teaching panel 15 (original).](../assets/figures/ml_fig_c494_15.png)
*Figure — Speculative draft path c494. Synthetic teaching geometry—not a causal claim.*


![c495 teaching panel 15 (original).](../assets/figures/ml_fig_c495_15.png)
*Figure — FlashAttention trade c495. Synthetic teaching geometry—not a causal claim.*


![c496 teaching panel 15 (original).](../assets/figures/ml_fig_c496_15.png)
*Figure — Activation ckpt trade c496. Synthetic teaching geometry—not a causal claim.*


![c497 teaching panel 15 (original).](../assets/figures/ml_fig_c497_15.png)
*Figure — Group-query trade c497. Synthetic teaching geometry—not a causal claim.*


![c498 teaching panel 15 (original).](../assets/figures/ml_fig_c498_15.png)
*Figure — SmoothQuant migrate bars c498. Synthetic teaching geometry—not a causal claim.*


![c499 teaching panel 15 (original).](../assets/figures/ml_fig_c499_15.png)
*Figure — Prune magnitude bars c499. Synthetic teaching geometry—not a causal claim.*


![c500 teaching panel 15 (original).](../assets/figures/ml_fig_c500_15.png)
*Figure — Quant int8 error bars c500. Synthetic teaching geometry—not a causal claim.*


![c501 teaching panel 15 (original).](../assets/figures/ml_fig_c501_15.png)
*Figure — Distill temperature path c501. Synthetic teaching geometry—not a causal claim.*


![c502 teaching panel 15 (original).](../assets/figures/ml_fig_c502_15.png)
*Figure — Knowledge transfer path c502. Synthetic teaching geometry—not a causal claim.*


![c503 teaching panel 15 (original).](../assets/figures/ml_fig_c503_15.png)
*Figure — LoRA rank bars c503. Synthetic teaching geometry—not a causal claim.*


![c504 teaching panel 15 (original).](../assets/figures/ml_fig_c504_15.png)
*Figure — QLoRA NF4 bars c504. Synthetic teaching geometry—not a causal claim.*


![c505 teaching panel 15 (original).](../assets/figures/ml_fig_c505_15.png)
*Figure — Adapter insert path c505. Synthetic teaching geometry—not a causal claim.*


![c506 teaching panel 15 (original).](../assets/figures/ml_fig_c506_15.png)
*Figure — Weight share residual c506. Synthetic teaching geometry—not a causal claim.*


![c507 teaching panel 15 (original).](../assets/figures/ml_fig_c507_15.png)
*Figure — Early exit path c507. Synthetic teaching geometry—not a causal claim.*


![c508 teaching panel 15 (original).](../assets/figures/ml_fig_c508_15.png)
*Figure — Token prune path c508. Synthetic teaching geometry—not a causal claim.*


![c509 teaching panel 15 (original).](../assets/figures/ml_fig_c509_15.png)
*Figure — KV cache size bars c509. Synthetic teaching geometry—not a causal claim.*


![c510 teaching panel 15 (original).](../assets/figures/ml_fig_c510_15.png)
*Figure — Speculative draft path c510. Synthetic teaching geometry—not a causal claim.*


![c511 teaching panel 15 (original).](../assets/figures/ml_fig_c511_15.png)
*Figure — FlashAttention trade c511. Synthetic teaching geometry—not a causal claim.*


![c512 teaching panel 15 (original).](../assets/figures/ml_fig_c512_15.png)
*Figure — Activation ckpt trade c512. Synthetic teaching geometry—not a causal claim.*


![c513 teaching panel 15 (original).](../assets/figures/ml_fig_c513_15.png)
*Figure — Group-query trade c513. Synthetic teaching geometry—not a causal claim.*


![c514 teaching panel 15 (original).](../assets/figures/ml_fig_c514_15.png)
*Figure — SmoothQuant migrate bars c514. Synthetic teaching geometry—not a causal claim.*


![c515 teaching panel 15 (original).](../assets/figures/ml_fig_c515_15.png)
*Figure — Prune magnitude bars c515. Synthetic teaching geometry—not a causal claim.*


![c516 teaching panel 15 (original).](../assets/figures/ml_fig_c516_15.png)
*Figure — Quant int8 error bars c516. Synthetic teaching geometry—not a causal claim.*


![c517 teaching panel 15 (original).](../assets/figures/ml_fig_c517_15.png)
*Figure — Distill temperature path c517. Synthetic teaching geometry—not a causal claim.*


![c518 teaching panel 15 (original).](../assets/figures/ml_fig_c518_15.png)
*Figure — Knowledge transfer path c518. Synthetic teaching geometry—not a causal claim.*


![c519 teaching panel 15 (original).](../assets/figures/ml_fig_c519_15.png)
*Figure — LoRA rank bars c519. Synthetic teaching geometry—not a causal claim.*


![c520 teaching panel 15 (original).](../assets/figures/ml_fig_c520_15.png)
*Figure — QLoRA NF4 bars c520. Synthetic teaching geometry—not a causal claim.*


![c521 teaching panel 15 (original).](../assets/figures/ml_fig_c521_15.png)
*Figure — Adapter insert path c521. Synthetic teaching geometry—not a causal claim.*


![c522 teaching panel 15 (original).](../assets/figures/ml_fig_c522_15.png)
*Figure — Weight share residual c522. Synthetic teaching geometry—not a causal claim.*


![c523 teaching panel 15 (original).](../assets/figures/ml_fig_c523_15.png)
*Figure — Early exit path c523. Synthetic teaching geometry—not a causal claim.*


![c524 teaching panel 15 (original).](../assets/figures/ml_fig_c524_15.png)
*Figure — Token prune path c524. Synthetic teaching geometry—not a causal claim.*


![c525 teaching panel 15 (original).](../assets/figures/ml_fig_c525_15.png)
*Figure — KV cache size bars c525. Synthetic teaching geometry—not a causal claim.*


![c526 teaching panel 15 (original).](../assets/figures/ml_fig_c526_15.png)
*Figure — Speculative draft path c526. Synthetic teaching geometry—not a causal claim.*


![c527 teaching panel 15 (original).](../assets/figures/ml_fig_c527_15.png)
*Figure — FlashAttention trade c527. Synthetic teaching geometry—not a causal claim.*


![c528 teaching panel 15 (original).](../assets/figures/ml_fig_c528_15.png)
*Figure — Activation ckpt trade c528. Synthetic teaching geometry—not a causal claim.*


![c529 teaching panel 15 (original).](../assets/figures/ml_fig_c529_15.png)
*Figure — Group-query trade c529. Synthetic teaching geometry—not a causal claim.*


![c530 teaching panel 15 (original).](../assets/figures/ml_fig_c530_15.png)
*Figure — SmoothQuant migrate bars c530. Synthetic teaching geometry—not a causal claim.*


![c531 teaching panel 15 (original).](../assets/figures/ml_fig_c531_15.png)
*Figure — Prune magnitude bars c531. Synthetic teaching geometry—not a causal claim.*


![c532 teaching panel 15 (original).](../assets/figures/ml_fig_c532_15.png)
*Figure — Quant int8 error bars c532. Synthetic teaching geometry—not a causal claim.*


![c533 teaching panel 15 (original).](../assets/figures/ml_fig_c533_15.png)
*Figure — Distill temperature path c533. Synthetic teaching geometry—not a causal claim.*


![c534 teaching panel 15 (original).](../assets/figures/ml_fig_c534_15.png)
*Figure — Knowledge transfer path c534. Synthetic teaching geometry—not a causal claim.*


![c535 teaching panel 15 (original).](../assets/figures/ml_fig_c535_15.png)
*Figure — LoRA rank bars c535. Synthetic teaching geometry—not a causal claim.*


![c536 teaching panel 15 (original).](../assets/figures/ml_fig_c536_15.png)
*Figure — QLoRA NF4 bars c536. Synthetic teaching geometry—not a causal claim.*


![c537 teaching panel 15 (original).](../assets/figures/ml_fig_c537_15.png)
*Figure — Adapter insert path c537. Synthetic teaching geometry—not a causal claim.*


![c538 teaching panel 15 (original).](../assets/figures/ml_fig_c538_15.png)
*Figure — Weight share residual c538. Synthetic teaching geometry—not a causal claim.*


![c539 teaching panel 15 (original).](../assets/figures/ml_fig_c539_15.png)
*Figure — Early exit path c539. Synthetic teaching geometry—not a causal claim.*


![c540 teaching panel 15 (original).](../assets/figures/ml_fig_c540_15.png)
*Figure — Token prune path c540. Synthetic teaching geometry—not a causal claim.*


![c541 teaching panel 15 (original).](../assets/figures/ml_fig_c541_15.png)
*Figure — KV cache size bars c541. Synthetic teaching geometry—not a causal claim.*


![c542 teaching panel 15 (original).](../assets/figures/ml_fig_c542_15.png)
*Figure — Speculative draft path c542. Synthetic teaching geometry—not a causal claim.*


![c543 teaching panel 15 (original).](../assets/figures/ml_fig_c543_15.png)
*Figure — FlashAttention trade c543. Synthetic teaching geometry—not a causal claim.*


![c544 teaching panel 15 (original).](../assets/figures/ml_fig_c544_15.png)
*Figure — Activation ckpt trade c544. Synthetic teaching geometry—not a causal claim.*


![c545 teaching panel 15 (original).](../assets/figures/ml_fig_c545_15.png)
*Figure — Group-query trade c545. Synthetic teaching geometry—not a causal claim.*


![c546 teaching panel 15 (original).](../assets/figures/ml_fig_c546_15.png)
*Figure — SmoothQuant migrate bars c546. Synthetic teaching geometry—not a causal claim.*


![c547 teaching panel 15 (original).](../assets/figures/ml_fig_c547_15.png)
*Figure — Prune magnitude bars c547. Synthetic teaching geometry—not a causal claim.*


![c548 teaching panel 15 (original).](../assets/figures/ml_fig_c548_15.png)
*Figure — Quant int8 error bars c548. Synthetic teaching geometry—not a causal claim.*


![c549 teaching panel 15 (original).](../assets/figures/ml_fig_c549_15.png)
*Figure — Distill temperature path c549. Synthetic teaching geometry—not a causal claim.*


![c550 teaching panel 15 (original).](../assets/figures/ml_fig_c550_15.png)
*Figure — Knowledge transfer path c550. Synthetic teaching geometry—not a causal claim.*


![c551 teaching panel 15 (original).](../assets/figures/ml_fig_c551_15.png)
*Figure — LoRA rank bars c551. Synthetic teaching geometry—not a causal claim.*


![c552 teaching panel 15 (original).](../assets/figures/ml_fig_c552_15.png)
*Figure — QLoRA NF4 bars c552. Synthetic teaching geometry—not a causal claim.*


![c553 teaching panel 15 (original).](../assets/figures/ml_fig_c553_15.png)
*Figure — Adapter insert path c553. Synthetic teaching geometry—not a causal claim.*


![c554 teaching panel 15 (original).](../assets/figures/ml_fig_c554_15.png)
*Figure — Weight share residual c554. Synthetic teaching geometry—not a causal claim.*


![c555 teaching panel 15 (original).](../assets/figures/ml_fig_c555_15.png)
*Figure — Early exit path c555. Synthetic teaching geometry—not a causal claim.*


![c556 teaching panel 15 (original).](../assets/figures/ml_fig_c556_15.png)
*Figure — Token prune path c556. Synthetic teaching geometry—not a causal claim.*


![c557 teaching panel 15 (original).](../assets/figures/ml_fig_c557_15.png)
*Figure — KV cache size bars c557. Synthetic teaching geometry—not a causal claim.*


![c558 teaching panel 15 (original).](../assets/figures/ml_fig_c558_15.png)
*Figure — Speculative draft path c558. Synthetic teaching geometry—not a causal claim.*


![c559 teaching panel 15 (original).](../assets/figures/ml_fig_c559_15.png)
*Figure — FlashAttention trade c559. Synthetic teaching geometry—not a causal claim.*


![c560 teaching panel 15 (original).](../assets/figures/ml_fig_c560_15.png)
*Figure — Activation ckpt trade c560. Synthetic teaching geometry—not a causal claim.*


![c561 teaching panel 15 (original).](../assets/figures/ml_fig_c561_15.png)
*Figure — Group-query trade c561. Synthetic teaching geometry—not a causal claim.*


![c562 teaching panel 15 (original).](../assets/figures/ml_fig_c562_15.png)
*Figure — SmoothQuant migrate bars c562. Synthetic teaching geometry—not a causal claim.*


![c563 teaching panel 15 (original).](../assets/figures/ml_fig_c563_15.png)
*Figure — Prune magnitude bars c563. Synthetic teaching geometry—not a causal claim.*


![c564 teaching panel 15 (original).](../assets/figures/ml_fig_c564_15.png)
*Figure — Quant int8 error bars c564. Synthetic teaching geometry—not a causal claim.*


![c565 teaching panel 15 (original).](../assets/figures/ml_fig_c565_15.png)
*Figure — Distill temperature path c565. Synthetic teaching geometry—not a causal claim.*


![c566 teaching panel 15 (original).](../assets/figures/ml_fig_c566_15.png)
*Figure — Knowledge transfer path c566. Synthetic teaching geometry—not a causal claim.*


![c567 teaching panel 15 (original).](../assets/figures/ml_fig_c567_15.png)
*Figure — LoRA rank bars c567. Synthetic teaching geometry—not a causal claim.*


![c568 teaching panel 15 (original).](../assets/figures/ml_fig_c568_15.png)
*Figure — QLoRA NF4 bars c568. Synthetic teaching geometry—not a causal claim.*


![c569 teaching panel 15 (original).](../assets/figures/ml_fig_c569_15.png)
*Figure — Adapter insert path c569. Synthetic teaching geometry—not a causal claim.*


![c570 teaching panel 15 (original).](../assets/figures/ml_fig_c570_15.png)
*Figure — Weight share residual c570. Synthetic teaching geometry—not a causal claim.*

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
