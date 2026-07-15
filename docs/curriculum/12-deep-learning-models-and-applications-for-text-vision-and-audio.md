# Chapter 12. Deep Learning Models and Applications for Text, Vision, and Audio

## Opening

A multimodal model claims to fuse note text, DWI, and audio dysphagia screens. Cross-modal architectures are exciting; they also multiply failure modes. This chapter separates capability demos from clinically transportable systems.


![Self-attention weights for the three-token worked example (original).](../assets/figures/ml_fig_attention.png)

*Scaled dot-product self-attention α and context vector for q=t₁ (matches the chapter numerics; original).*

![Deep models compose layered representations (original diagram).](../assets/figures/ml_fig_mlp.png)

*Deep models compose layered representations (original diagram).*

![Site shift in embedding space (synthetic; original).](../assets/figures/ml_fig_site_shift.png)

*Site shift in embedding space (synthetic; original).*
## Learning Objectives

Implement sequence-to-sequence models with dot-product, Bahdanau, Luong, self-, and cross-attention.

Explain Transformer architecture: positional encoding, multi-head attention, encoder/decoder, masking.

Summarize structured state-space models (S4) and Mamba as efficient sequence alternatives.

Compare LLM families: BERT line, T5, GPT line, instruction-tuned chat models, Llama/Mistral MoE; fine-tune with RLHF, DPO, LoRA; cite evaluation benchmarks.

Trace vision classification from LeNet through ResNet and Inception to ViT; detection (R-CNN family, SSD, YOLO); segmentation (U-Net, FCN, Mask R-CNN, DeepLab, SAM).

Describe NeRF and 3D Gaussian splatting concepts for view synthesis.

Outline audio models: WaveNet, Tacotron, wav2vec, Whisper.

Apply these stacks to stroke imaging and clinical NLP with leakage, hallucination, and calibration awareness.

## 12.1 Sequence-to-Sequence and Attention

Sequence-to-sequence (seq2seq) models map an input sequence (tokens, frames, time points) to an output sequence of possibly different length—machine translation, summarization, speech recognition, discharge-summary generation. The classic encoder–decoder RNN compresses the input into a single fixed vector passed to the decoder. That bottleneck discards positional detail for long inputs and motivates attention: at each decoder step, compute a weighted average of encoder states rather than a single summary. Clinical sequence pairs include: EMS narrative → structured last-known-well and anticoagulant flags; radiology impressions → ICD-style codes; serial NIHSS plus vitals → free-text deterioration alerts. Length mismatch and domain vocabulary make attention and subword tokenization essential rather than optional.

![12.1: Sequence-to-sequence generation with attention. An encoder reads the source tokens (bottom) into hidden states h_s; as t](../assets/figures/ml_concept_12.1_7132e488.png)

*Figure 12.1 — original teaching graphic.*

Before Transformers, attention was a module bolted onto RNNs; after Transformers, attention is the primary computational motif and recurrence is optional. This chapter traces that shift, then surveys the application stacks that dominate text, vision, and audio practice—including efficient sequence models that try to escape quadratic attention cost. Each section ends with enough operational detail for a neurologist–epidemiologist to critique papers and protocols, not merely recognize acronyms. Where older curricula stopped at word2vec and simple RNNs, contemporary practice requires fluency with multi-head attention, instruction tuning, detection metrics, and speech foundation models because those are what appear in grant aims and vendor demos. Reading this chapter with a pencil: for each architecture family, write one stroke-system use case, one failure mode, and one metric that would convince a skeptical QI committee.

### Attention mechanisms

Dot-product attention scores compatibility of a query vector q with keys k_i by q^T k_i (optionally scaled by 1/√d), softmax-normalizes scores to weights α_i, and returns Σ_i α_i v_i for values v_i. Bahdanau (additive) attention computes scores with a small feedforward network a(s, h) = v^T tanh(W_s s + W_h h) combining decoder state s and encoder state h—historically first successful alignment for neural MT. Luong attention explores dot, general (q^T W k), and concat variants and couples them with input-feeding decoders. Self-attention sets queries, keys, and values from the same sequence so tokens contextualize each other. Cross-attention sets queries from one sequence (decoder) and keys/values from another (encoder or image tokens), fusing modalities or stages.

Scaled dot-product attention Attention(Q,K,V) = softmax(QK^T / √d_k) V is the workhorse of Transformers. Scaling prevents dot products from growing with dimension in a way that saturates softmax. Attention maps are sometimes inspected for ‘explanations,’ but they are not causal attributions; use them as weak diagnostic visualizations only.

![Attention / sampling temperature: sharp vs soft softmax mass (original).](../assets/figures/ml_fig_attention_temperature.png)

*Figure — Temperature T rescales logits before softmax. **Left:** fixed key scores become nearly one-hot at low T and nearly uniform at high T. **Right:** entropy rises and peak weight falls as T grows. Training often uses T=1 with 1/√d_k scaling; generation may anneal T for diversity. Softmax temperature is not a clinical “confidence dial,” and attention weights remain associative geometry—not causal pathways in the chart.*

A compact statement of the core algorithm—with an optional additive mask M holding 0 at allowed positions and −∞ at disallowed ones—makes the tensor shapes explicit:

```
# Scaled dot-product attention (one head)
# Q: n_q x d_k K: n_k x d_k V: n_k x d_v mask M: n_q x n_k (entries 0 or -inf)
function attention(Q, K, V, M=0):
 S = (Q @ transpose(K)) / sqrt(d_k) # scores, n_q x n_k
 A = softmax(S + M, axis=keys) # weights, each row sums to 1
 return A @ V # context, n_q x d_v

# Multi-head attention: query source X_q, key/value source X_kv, h heads
function multi_head(X_q, X_kv, M=0):
 for i in 1..h: # each head has d_k = d_model / h
 Q_i = X_q @ W_q[i]
 K_i = X_kv @ W_k[i]
 V_i = X_kv @ W_v[i]
 head_i = attention(Q_i, K_i, V_i, M)
 return concat(head_1, ..., head_h) @ W_o
```

Self-attention passes X_q = X_kv (one sequence contextualizes itself); cross-attention passes X_q from the decoder and X_kv from the encoder; a decoder’s masked self-attention supplies the lower-triangular M so no position attends to its future.

## 12.2 The Transformer

Transformers replace recurrence with stacked self-attention and position-wise feedforward layers, enabling full parallelization over sequence length during training. Residual (skip) connections around sublayers plus layer normalization stabilize deep stacks: output = LayerNorm(x + Sublayer(x)) (pre-norm variants reorder operations).

![12.2: The Transformer encoder–decoder. Inputs are token-embedded and summed with positional encodings, then passed through N e](../assets/figures/ml_concept_12.2_9dae343b.png)

*Figure 12.2 — original teaching graphic.*

### Positional encoding

Bare attention is permutation-equivariant: without position information, bag-of-tokens behavior emerges. Sinusoidal positional encodings add fixed functions of position index to embeddings; learned absolute position embeddings are alternative; relative position biases and rotary embeddings (RoPE) encode pairwise geometry more flexibly for long contexts. Medical notes and genomic sequences both stress long-context positional design. Extrapolation beyond training context lengths is imperfect: a model trained at 4k tokens may degrade at 32k without specialized positional schemes or continued training. For chart summarization, explicit retrieval of the relevant note sections often beats stuffing entire longitudinal records into a fragile long context.

![12.3: Sinusoidal positional encoding PE[pos, i]. Even dimensions use PE = sin(pos / 10000^{2i/d}) and odd dimensions use cos, ](../assets/figures/ml_concept_12.3_0a3122e9.png)

*Figure 12.3 — original teaching graphic.*

![Sinusoidal positional encoding heatmap and per-dimension waves (teaching recompute; original).](../assets/figures/ml_fig_pos_encoding.png)

*Figure — PE as a matrix. **Left:** PE[pos, *i*] for *d*=32, length 64—slow oscillations on low *i*, fast on high *i* (ωᵢ = 10000^(−2*i*/*d*)). **Right:** selected dimensions stacked; even *i* use sin, odd use cos. Without PE, self-attention is bag-of-tokens; RoPE/ALiBi are modern relatives for long EHR contexts. Positional geometry is not a clinical causal graph.*

### Multi-head attention and architecture

Multi-head attention runs h parallel attention heads with different learned projections, concatenates outputs, and projects again—allowing heads to specialize (syntax vs rare tokens, local vs long range). The original encoder stack applies multi-head self-attention and feedforward sublayers repeatedly. The decoder stack adds masked multi-head self-attention (causal mask so position t cannot attend to future tokens) and cross-attention over encoder outputs, then feedforward blocks. Encoder-only models (BERT-style) use bidirectional self-attention for understanding; decoder-only models (GPT-style) use causal masks for autoregressive generation; encoder–decoder models (T5-style) use both for conditional generation.

![Multi-head attention specialization caricature: local, prefix, sparse heads (synthetic; original).](../assets/figures/ml_fig_attention_heads.png)

*Figure — Heads are not interchangeable. Heatmaps show synthetic attention maps for three heads: local/positional band, prefix bias, and sparse content peaks. Specialization can help capacity, but inspecting one pretty head is not a clinical explanation and not a causal pathway through the note. Treat attention as associative geometry under a trained metric.*

![FlashAttention-style IO: tiled exact attention without full N×N materialization (original).](../assets/figures/ml_fig_flash_attention_io.png)

*Figure — Same softmax attention math, different memory traffic. **Left:** naive score-matrix storage grows ~N² while tiled IO grows much slower (teaching curves). **Right:** Q/K/V tiles → on-chip SRAM → accumulated output. Faster long-context serving still needs retrieval hygiene for longitudinal notes; speed is not causation.*

![RoPE: rotary positional embedding relative geometry (original).](../assets/figures/ml_fig_rope.png)

*Figure — Position as rotation. **Left:** queries/keys at positions t and t+Δ are rotated in a plane. **Right:** relative score contribution depends on Δ through cos(Δ·ω), aiding length extrapolation versus pure absolute PE. Positional geometry is still not a clinical causal graph.*

![Attention mask patterns: causal, bidirectional, sliding window (original).](../assets/figures/ml_fig_attention_masks.png)

*Figure — Allowed key positions for each query. Causal masks enable autoregressive decode; bidirectional suits encoder understanding; sliding windows bound cost for long notes. Mask pattern is inductive bias—not clinical causation.*

## 12.3 Structured State-Space Models and Mamba

Quadratic cost of self-attention in sequence length motivates efficient sequence layers. State-space models (SSMs) describe continuous-time latent dynamics h’(t) = A h(t) + B x(t), y(t) = C h(t) + D x(t), discretized for sequences. Linear time-invariant SSMs can be written as convolutions, enabling fast training with FFT-based methods while retaining recurrent inference.

Structured state space for sequence modeling (S4) imposes structure on A (e.g., diagonal plus low-rank) so long-range dependencies remain tractable and numerically stable. Mamba extends selective SSMs: parameters depend on the input so the model can ignore or retain information content-dependently, with hardware-aware parallel scan implementations. These architectures aim for transformer-level quality with near-linear scaling in length—attractive for long EHR timelines, continuous EEG, and high-resolution signals. As with any new backbone, clinical claims require the same validation hygiene as Transformers. A pragmatic evaluation for long EHR sequences is to hold the prediction task fixed (for example, 30-day readmission or delayed cerebral edema) and compare Transformer, S4/Mamba-style, and gradient-boosted summaries of the same features under identical splits—architecture wins only if the gain is stable across sites and calendar years.

## 12.4 Large Language Models

### BERT family

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only model pretrained with masked language modeling (predict randomly masked tokens from full bidirectional context) and next-sentence prediction (later shown less essential). Inputs pack token, segment, and position embeddings with special [CLS] and [SEP] tokens. Fine-tuning attaches task heads for classification, span extraction, or sequence labeling. Derivatives include RoBERTa (robustly optimized training: more data, dynamic masking, no NSP), DistilBERT (distilled smaller student), and XLM/XLM-R for multilingual settings. Clinical variants (domain-adaptive pretraining on notes) improve phenotyping and entity recognition when privacy-compliant corpora exist. Tokenization choices (WordPiece, BPE, SentencePiece) affect rare drug names and laterality phrases; always inspect unknown-token rates on local notes before declaring a backbone unfit.

![Subword tokenization and embedding nearest-neighbor geometry (teaching sketch; original).](../assets/figures/ml_fig_token_neighbors.png)

*Figure — Tokens and neighbors. **Left:** a clinical phrase is split into subword pieces (BPE / WordPiece style)—stems such as `throm` + `##bolysis` and `lateral` + `##ity` stay productive, while pure word-level vocabularies flood local notes with `[UNK]`. Always measure unknown-token rates on your own discharge summaries and med lists. **Right:** a synthetic two-dimensional sketch of static embedding neighborhoods (vascular events, scales, reperfusion drugs, language signs, hemorrhage terms). Proximity supports retrieval and transfer but is corpus-dependent: near ≠ synonymous, and embedding geometry is **not** a causal map of disease relationships.*

### T5 and the text-to-text framework

T5 casts every NLP problem as text-to-text: inputs and outputs are token sequences, with task prefixes (‘translate English to German:’, ‘cola sentence:’). An encoder–decoder Transformer is pretrained with span corruption objectives and fine-tuned per task. The unified interface simplifies multi-task learning and transfer—useful when a stroke group wants one stack for summarization, coding, and question answering over guidelines.

### GPT line and instruction-following chat models

GPT models are decoder-only Transformers trained to predict the next token. GPT-1 showed generative pretraining plus fine-tuning; GPT-2 scaled and emphasized zero-shot task transfer; GPT-3 demonstrated few-shot in-context learning at much larger scale. Instruct models and ChatGPT-style systems add instruction tuning and alignment so models follow user intents in dialogue rather than only continue web text. Capability jumps often come from scale, data mix, and alignment, not only architectural novelty.

### Llama, Mistral, and mixture-of-experts

Open-weight Llama generations popularized strong decoder-only models with careful data curation, rotary embeddings, and efficient attention implementations; Llama 2/3 improved scale, context, and chat variants. Mistral 7B delivered competitive quality at smaller size using grouped-query attention (GQA)—several query heads share one key/value head, shrinking the key/value cache that dominates memory during long-context generation—and sliding-window attention, in which each token attends only to a fixed window of recent tokens, bounding per-layer cost while still letting information propagate across stacked layers. Mixtral models use sparse mixture-of-experts (MoE): each token routes to a subset of expert FFNs (e.g., 8×7B with 2 experts active), increasing parameter capacity without fully proportional FLOPs per token. MoE complicates serving (memory for all experts) but offers a quality–compute trade-off relevant to on-prem hospital deployments.

### Fine-tuning: RLHF, DPO, LoRA, and domain adaptation

Full fine-tuning updates all weights on task data—costly and prone to catastrophic forgetting. LoRA (Low-Rank Adaptation) freezes each pretrained weight matrix W (shape d_out×d_in) and learns a low-rank update ΔW = B·A, where A is r×d_in and B is d_out×r with rank r ≪ min(d_in, d_out); the adapted layer computes (W + BA)x. Only A and B are trained, so trainable parameters fall from d_in·d_out to r·(d_in+d_out)—for a 4096×4096 projection at r=8 that is ≈65.5k versus ≈16.8M, roughly a 256× reduction. The tiny adapters cost little to store and can be merged back (W ← W + BA) for zero added inference latency, or kept separate and swapped per task. RLHF (reinforcement learning from human feedback) trains a reward model on human preference comparisons, then optimizes the policy (often with PPO) to raise reward while staying near a reference model via KL penalties. Direct Preference Optimization (DPO) reparameterizes preference learning into a supervised-style objective on preferred vs rejected responses, avoiding explicit RL loops. Domain-specific fine-tuning on de-identified notes or guidelines can improve terminology but risks privacy leakage, hallucination of citations, and overconfidence—mitigate with retrieval grounding and human review.

### Evaluation benchmarks

LLM evaluation mixes automatic and human protocols. General benchmarks include language understanding suites (GLUE/SuperGLUE historically), knowledge and reasoning sets (MMLU, BIG-bench subsets), reading comprehension, math (GSM8K), and coding tests. Chat models need preference win rates, safety evaluations, and instruction-following rubrics. Medical benchmarks (e.g., clinical QA, medical licensing-style questions) measure knowledge but not bedside utility, calibration, or site-specific workflow fit. Always evaluate on local notes for phenotype extraction: entity F1, document-level AUROC for computable phenotypes, and error analysis for negation and temporality. Latency, cost per 1k tokens, and hallucination rate under retrieval-augmented generation belong in the same report as accuracy.

## 12.5 Computer Vision: Classification Backbones

LeNet pioneered CNN digit recognition with conv–pool stacks and dense layers. AlexNet reignited deep vision at ImageNet scale with ReLU, dropout, and GPU training. VGG showed that deep stacks of small 3×3 convolutions work well. GoogLeNet/Inception introduced multi-branch modules mixing 1×1, 3×3, 5×5 convolutions and pooling in parallel, with 1×1 bottlenecks for efficiency; later Inception-v4 and Inception-ResNet hybridize residual links. ResNet introduced identity skip connections so deep nets train by learning residual functions F(x) with output x+F(x), enabling hundreds of layers and remaining a default medical imaging backbone.

![12.4: Vision Transformer (ViT) patch embedding. An image (here a head-CT-like scene) is split into a grid of non-overlapping p](../assets/figures/ml_concept_12.4_25ef3fd6.png)

*Figure 12.4 — original teaching graphic.*

Vision Transformers (ViT) split images into patch tokens, embed them linearly, add position encodings, and run Transformer encoders—treating vision as a sequence problem. With sufficient data or strong distillation/pretraining, ViTs match or exceed CNNs; hybrids combine convolutional stems with Transformer towers. For stroke CT with smaller labeled sets, CNN inductive bias (locality, translation equivariance) still helps unless domain-specific pretraining is strong. Transfer learning recipes should log which layers were frozen, what input normalization matched pretraining, and whether grayscale CT was replicated across RGB channels—an example of reproducibility detail that reviewers and regulators increasingly expect.

## 12.6 Object Detection

Detection outputs class labels and bounding boxes (and optionally confidence) for instances. Metrics include mean average precision (mAP) over IoU thresholds. Two-stage detectors propose regions then classify; one-stage detectors predict densely in one shot.

### R-CNN, Fast R-CNN, Faster R-CNN

R-CNN runs selective search proposals, warps each region through a CNN classifier, and refines boxes—accurate but slow due to per-region forward passes. Fast R-CNN shares a convolutional feature map and uses RoI pooling to extract per-proposal features with a multi-task loss for classification and box regression. Faster R-CNN replaces external proposals with a region proposal network (RPN) trained end-to-end with the detector, becoming a standard two-stage template for medical detection (e.g., aneurysm candidates) when latency allows.

### SSD and YOLO

Single Shot MultiBox Detector (SSD) predicts boxes and classes from multi-scale feature maps with default anchor boxes in one forward pass. YOLO (You Only Look Once) frames detection as grid-cell regression: v1 divides the image into cells predicting boxes and class probabilities; v2/YOLO9000 introduce anchor priors, batch norm, multi-scale training, and joint detection–classification training on large label spaces; v3 adds multi-scale predictions and deeper backbones. Later YOLO versions continue accuracy–speed engineering. Real-time YOLO-style detectors appeal to triage workflows (e.g., flagging potential ICH) if false-positive rates are managed in product design.

## 12.7 Semantic and Instance Segmentation

Semantic segmentation labels each pixel with a class (ischemic lesion vs background). Instance segmentation separates object instances (each hemorrhage component). U-Net’s encoder–decoder with skips is the clinical default for biomedical segmentation. Fully convolutional networks (FCN) replace dense layers with convolutions for spatial outputs and upsample coarse score maps. Mask R-CNN extends Faster R-CNN with a parallel mask head on RoI features for instance masks. DeepLab models use atrous (dilated) convolutions and atrous spatial pyramid pooling for multi-scale context; v3+ adds a decoder module for sharper boundaries.

![12.5: Detection versus segmentation on the same scene. Object detection (left) localizes each instance with a bounding box and](../assets/figures/ml_concept_12.5_fe60984f.png)

*Figure 12.5 — original teaching graphic.*

Segment Anything Model (SAM) is a promptable foundation model for segmentation: a heavy image encoder, a prompt encoder (points, boxes, masks), and a lightweight mask decoder produce masks zero-shot across domains. SAM v1 popularized interactive annotation acceleration; SAM v2 extends tracking/segmentation in videos. In stroke research, SAM-assisted labeling can cut mask time, but clinical deployment still needs task-specific validation—promptable generality is not the same as approved CAD.

## 12.8 3D View Synthesis: NeRF and Gaussian Splatting

Neural Radiance Fields (NeRF) represent a scene as a continuous function mapping 3D position and viewing direction to color and density, typically with an MLP. Rendering integrates colors along camera rays using volume rendering integrals; training minimizes error between rendered and observed pixels from multiple views. NeRFs enable novel-view synthesis and 3D-consistent reconstruction without explicit meshes, at the cost of slow training/rendering in classic form (many accelerations exist).

3D Gaussian splatting represents scenes as collections of anisotropic 3D Gaussians with opacity and appearance parameters, rasterized via fast splatting rather than dense ray marching. Optimization fits Gaussian parameters to multi-view images, often achieving real-time rendering with high visual quality. Concepts matter for surgical planning visualization, photogrammetric anatomy teaching, and research on 3D stroke anatomy from limited views—distinct from diagnostic voxel labeling in native CT/MRI grids.

## 12.9 Audio Models

Audio pipelines start from waveforms or spectrograms (STFT, mel-scale filterbanks). Modeling goals include generation (text-to-speech), enhancement, and recognition (speech-to-text). Domain features—pitch, MFCCs historically, learned filterbanks now—interact with architectural choice.

### WaveNet and Tacotron

WaveNet generates raw audio autoregressively with stacks of dilated causal convolutions, capturing long-range temporal structure for high-fidelity speech synthesis. Parallel WaveNet / probability density distillation accelerates sampling by training a parallel student. Tacotron systems map text to spectrograms with seq2seq (attention) models; Tacotron 2 pairs an improved spectrogram predictor with a neural vocoder (WaveNet-style) for natural TTS. Clinical relevance includes accessibility tools and standardized spoken prompts—not diagnostic auscultation replacements without targeted validation.

### wav2vec and Whisper

wav2vec models learn speech representations from unlabeled audio via contrastive self-supervision over latent convolutional encodings (v1) and, in wav2vec 2.0, masked latent prediction with quantization—enabling strong ASR with limited labeled speech. Whisper trains a sequence-to-sequence Transformer on large-scale weakly supervised audio–transcript pairs for robust multilingual speech recognition and translation, with multitask formatting (language ID, transcription, translation). Whisper-style models can transcribe clinical dictation or research interviews; privacy, domain accents, and medical vocabulary error rates need local measurement before chart integration. Combined audio–NLP pipelines (dictation → Whisper → LLM structuring → EHR fields) multiply error sources; insert human confirmation at medication and laterality fields even if free-text drafting is automated. Benchmarks for clinical ASR should report word error rate overall and on a medication-name subset, plus critical information omission rates judged by clinicians—not only generic WER on call-center corpora. For research interviews, store audio under IRB rules separate from model logs. Vendor speech engines and open Whisper checkpoints should be compared on the same held-out local sample before procurement decisions. Document microphone type and room noise when collecting that sample so results remain fully interpretable over time.

## 12.10 Worked Example: Attention Weights and a Causal Mask

Consider a decoder step with query q = [1.0, 0.0] and three encoder keys k1=[1.0,0.0], k2=[0.0,1.0], k3=[0.7,0.7], with values equal to keys for simplicity and d_k=2 so √d_k=√2≈1.414. The dot products q·k_i are 1.0, 0.0, and 0.7 (only the first component of q is nonzero, so each score reads off the first component of the key). Dividing by √2 gives scaled scores 0.707, 0.000, and 0.495. Exponentiate: exp(0.707)≈2.028, exp(0.000)=1.000, exp(0.495)≈1.640, which sum to 2.028+1.000+1.640=4.668. Softmax divides each exponential by that sum, giving weights α≈[2.028/4.668, 1.000/4.668, 1.640/4.668]=[0.434, 0.214, 0.351] (these sum to 1.000—a useful check). The attention output is the weighted sum 0.434·k1 + 0.214·k2 + 0.351·k3: the x-coordinate is 0.434·1.0 + 0.214·0.0 + 0.351·0.7 = 0.434 + 0.246 = 0.680, and the y-coordinate is 0.434·0.0 + 0.214·1.0 + 0.351·0.7 = 0.214 + 0.246 = 0.460, so the output ≈ [0.680, 0.460]. The query preferentially weights the aligned key k1 (largest α) while still mixing in context from k2 and k3, and the output lands between the keys, pulled toward k1 and k3. Had we skipped the 1/√d_k scaling, the ranking would be unchanged but the weights would be sharper (α≈[0.474, 0.174, 0.351], more mass on k1)—exactly the softmax saturation that scaling is designed to temper.

![12.6: Scaled dot-product self-attention weights for the chapter's three-token example. Tokens t_1 = [1,0], t_2 = [0,1], t_3 = ](../assets/figures/ml_concept_12.6_76dbd7a2.png)

*Figure 12.6 — original teaching graphic.*

Causal mask example for length 3: the allowed attention pattern is lower-triangular—each position attends to itself and earlier positions only. Position 0 attends to key 0; position 1 attends to keys 0 and 1, with key 2 (a future token) masked; position 2 attends to keys 0, 1, and 2. Masked entries are set to −∞ before the softmax so their weight α=0 (since exp(−∞)=0). During training of GPT-style models, this same triangular mask is applied at every layer and every position in parallel, so the prediction at position t never sees tokens after t—essential for valid next-token likelihoods, and the reason a single forward pass yields a training signal for all positions at once.

![Causal attention mask for decoder-only language models (scientific; original).](../assets/figures/ml_fig_causal_mask.png)

*Lower-triangular causal mask: query position t may attend only to keys ≤ t; future positions receive −∞ before softmax so their weight is zero (original).*

## 12.11 Systems, Context Length, Retrieval, and Hallucination Control

Production LLMs are systems, not only weight matrices. Context windows limit how many tokens of chart, guideline, and user text can be considered; hierarchical summarization or retrieval-augmented generation (RAG) pulls relevant passages from a vector index of notes or PDFs. RAG reduces some hallucinations when citations are required, but retrieved wrong notes can still mislead—retrieval precision/recall belong in evaluation. Chunking strategies, embedding model choice, and metadata filters (date, service line) dominate quality for stroke pathway Q&A bots.

Serving constraints: quantization (8-bit, 4-bit) shrinks memory; speculative decoding (a small draft model proposes several tokens that the large target model verifies in one pass, keeping the longest correct prefix) and FlashAttention-style kernels (IO-aware exact attention that avoids writing the full n×n score matrix to slow memory) improve throughput; MoE models need expert parallelism. On-prem deployment may be mandated for PHI. Latency budgets for triage NLP differ from offline phenotyping jobs. Log prompts and outputs under security policy for incident review, without creating a new unprotected PHI store.

Hallucination control patterns: force structured outputs (JSON schemas for med lists), require extractive spans for critical fields (last known well time), abstain when confidence or retrieval similarity is low, and route uncertain cases to humans. An example evaluation set should include adversarial notes with negation, family history (‘mother had ICH’), and hypotheticals that models often mis-attribute to the patient.

## 12.12 Detection and Segmentation Metrics in Clinical Context

Bounding-box mAP averages precision across recall levels and classes, often at IoU≥0.5 or a range (COCO-style). For tiny intracranial findings, IoU thresholds and center-distance criteria may better match clinical usefulness than strict overlap. Free-response ROC (FROC) plots lesion-level sensitivity against false positives per image—standard in CAD literature and preferable to image-level AUROC alone when multiple findings exist.

Segmentation Dice = 2|A∩B|/(|A|+|B|) emphasizes overlap; Hausdorff distance emphasizes worst-case boundary errors that matter for eloquent cortex. Volume error and relative volume difference matter for edema growth studies. Always pair geometric metrics with reader studies when claims enter care pathways. An example reporting table for DWI lesion models should include Dice, volume MAE, site-stratified results, and failure cases (small cortical dots, motion, craniotomy hardware).

![Dice versus IoU on synthetic lesion masks (scientific; original).](../assets/figures/ml_fig_dice_iou.png)


![Multi-head attention pattern schematics (causal, local, CLS, mixed; original).](../assets/figures/ml_fig_multihead_patterns.png)

*Figure — Different heads implement different dependency biases. Attention maps are computational, not clinical causal diagrams. Always re-check calibration and subgroups after fine-tuning.*


![Toy spectrogram / time–frequency map (synthetic; original).](../assets/figures/ml_fig_spectrogram_toy.png)

*Figure — Time–frequency features feed audio/vision models. Pretty maps still need validated labels and calibration before any care claim. Features ≠ diagnosis; pred ≠ cause.*


![Toy 2D token embedding neighborhoods (synthetic; original).](../assets/figures/ml_fig_token_embed_2d.png)

*Figure — Embedding clusters reflect corpus co-occurrence geometry. Neighbor tokens are not automatically clinical synonyms or causal partners.*

*Same predicted/GT pair: Dice = 2|A∩B|/(|A|+|B|) and IoU = |A∩B|/|A∪B| are monotone transforms of each other (Dice = 2·IoU/(1+IoU)); both punish boundary misses that matter near eloquent cortex (original).*

### Architecture family quick map (teaching table)

| Family | Core motif | Stroke-system use case | Failure mode to demand evidence on |
|--------|------------|------------------------|-------------------------------------|
| Encoder-only (BERT-line) | Bidirectional self-attention | Note phenotyping, NER, coding assist | Negation/temporality errors; site note-style shift |
| Decoder-only (GPT-line) | Causal mask + next-token | Draft summaries, guideline Q&A with RAG | Hallucinated meds/laterality; citation invention |
| Encoder–decoder (T5-line) | Cross-attention conditional gen | Impression → structured codes | Length/format drift; rare-entity collapse |
| CNN / U-Net | Local filters + skips | DWI lesion seg, ICH detection | Domain shift (scanner/protocol); tiny-lesion miss |
| ViT / multimodal | Patch or token fusion | Joint note+image demos | Data hunger; uncalibrated cross-modal confidence |

Class imbalance in detection: most anchors or pixels are background. Focal loss, online hard-example mining, and balanced sampling of positive slices (many CT slices have no lesion) are engineering necessities. Slice-level models need study-level aggregation rules defined a priori (max score, noisy-OR, learned pooling) so validation matches deployment.

## 12.13 Putting Modalities Together: Multimodal Stroke Decision Support

A realistic multimodal stack for comprehensive stroke centers might: (1) Whisper or vendor ASR for EMS radio; (2) BERT/LLM extraction of last-known-well and anticoagulant status from ED notes; (3) CNN/YOLO-style ICH flag on NCCT; (4) CTA LVO classifier; (5) U-Net core/penumbra estimates from CTP if protocols allow; (6) calibrated tabular model combining NIHSS, age, and imaging scores for outcome probabilities. Each stage needs independent validation; end-to-end ‘single network’ demos rarely survive multi-site operations without modular failure isolation.

Epidemiologic uses differ from bedside triage. NLP phenotype pipelines measure incidence and disparities across years; imaging AI measures care quality (door-to-imaging times with automated detection timestamps) if clocks are synchronized. Confounding is rampant: hospitals that buy AI may already have better processes. Study designs should separate algorithm accuracy papers from interrupted time-series or stepped-wedge evaluations of clinical impact—different claims, different methods.

Future-facing but grounded: foundation models that accept mixed image patches and text tokens will blur chapter boundaries between vision and language. Still, the invariants of this textbook remain—index time, leakage, calibration, external validation, and respect for uncertain labels. Mastery means choosing Transformers, SSMs, CNNs, or boosted trees deliberately, not defaulting to the largest model available.

## 12.14 Fine-Tuning Recipes and Parameter-Efficient Adaptation in Practice

Domain-adaptive pretraining continues a general LLM or vision backbone on unlabeled in-domain corpora (notes, guidelines, imaging) before task fine-tuning. Continual pretraining risks catastrophic forgetting of general abilities; replay mixtures and lower learning rates mitigate this. For imaging, continued contrastive or masked pretraining on head CT often outperforms naive ImageNet transfer for subtle findings.

Parameter-efficient fine-tuning (PEFT) families include LoRA, adapters inserted between layers, prefix/prompt tuning that learns virtual tokens, and (IA)³-style rescaling vectors. LoRA rank and which modules receive adapters (query/value projections vs all linear layers) trade quality for VRAM. Merge LoRA weights into base weights for simpler serving when multiple adapters are not required. An example hospital pattern: one base model, separate LoRA adapters for coding, summarization, and patient-education rewrite, swapped at inference under access control.

Instruction tuning datasets should reflect clinical tone and safety: refuse dosage changes without clinician review, avoid fabricating citations, and escalate uncertainty. Preference data for DPO/RLHF need dual clinician annotations with adjudication; noisy preferences encode annotator bias (verbosity bias, sycophancy). Measure not only win rates but factuality against source notes on a frozen audit set.

Vision fine-tuning recipes: linear probe, partial fine-tune last stages, full fine-tune with discriminative LR (smaller for early layers), and multi-task heads (segmentation plus classification) that share encoders. Strong augmentations help natural images; for CT, prefer mild geometric transforms and intensity noise within clinically plausible HU ranges. Early stopping on patient-grouped validation remains non-negotiable.

## 12.15 Benchmarks, Reporting Standards, and Common Failure Modes

Public vision benchmarks (ImageNet, COCO, Cityscapes) shaped architectures but do not measure stroke-care utility. Medical reporting standards (TRIPOD-AI, CLAIM, STARD-AI where applicable) emphasize data provenance, inclusion criteria, human comparison, and intended use. Align internal reports with these checklists even for unpublished quality-improvement models.

Common NLP failure modes: negation inversion, section confusion (past medical history vs assessment), temporal mis-ordering of events, unit errors, and copy-paste note artifacts that models treat as fresh observations. Common vision failure modes: slice selection bias, partial volume effects, postoperative cases excluded from training but present at deployment, and over-reliance on endotracheal tubes or other devices spuriously correlated with severity labels.

Example incident review: a CTA LVO model’s precision drops after a vendor software update changes reconstruction kernels. Response requires monitoring input feature drift (intensity histograms, noise estimates), not only outcome metrics, plus a rollback plan. MLOps for clinical models is part of the curriculum of responsible deployment—not an afterthought once the paper is accepted.

When comparing BERT vs GPT-style vs proprietary chat APIs for a phenotyping task, hold constant the label definition, split, and preprocessing; vary only the model and prompt or fine-tune recipe. Cost per thousand notes and PHI policy compliance can eliminate otherwise accurate systems. The ‘best’ model is the one that survives governance, budget, and external validation simultaneously.

## Clinical and Epidemiologic Notes: Stroke Imaging and Clinical NLP

Stroke imaging ML spans detection (ICH on NCCT), large-vessel occlusion classification on CTA, ASPECTS-like regional scoring, DWI/PWI lesion segmentation, and outcome prediction from mixed imaging–clinical features. CNN/ViT backbones and U-Net segmenters are tools; the science is in reference standards, slice selection, vendor shift, and whether outputs change door-to-needle or transfer decisions. Report operating points co-designed with stroke teams, not only mAP or Dice peaks.

Clinical NLP with BERT-family or LLM extractors can identify AF, anticoagulation mentions, last-known-well times, and mRS narratives from notes. Temporality and negation (‘no evidence of LVO’) remain primary error sources. LLMs summarizing charts risk hallucinated meds and omitted contradictions—use retrieval of source sentences, require human sign-off for documentation, and never auto-file unreviewed text into the EHR. De-identification failures are privacy incidents; prefer approved environments.

Multimodal systems that attend jointly over imaging tokens and report text can improve retrieval and second-read prioritization, but correlated training labels (report already states the finding) create leakage if the task is ‘predict report from image’ while evaluating against the same report. Define whether the use case is triage before reading or coding after reading. Epidemiologic phenotype pipelines should version models, track drift when note templates change, and validate against chart review samples each quarter—not only against a single academic test split.

### Clinical NLP de-identification

Clinical free text is saturated with protected health information (PHI): patient and provider names, dates, ages over 89, medical-record and account numbers, addresses, phone and fax numbers, and institution names. Before notes can leave a protected environment—for model training, benchmark release, or multi-site research—this PHI must be removed or replaced. De-identification is therefore itself a clinical NLP task, essentially named-entity recognition specialized for identifiers, and one of the oldest problems in the field. Under U.S. HIPAA there are two defensible routes: Safe Harbor (strip 18 enumerated identifier categories) and Expert Determination (a qualified expert certifies that re-identification risk is very small). Production pipelines usually combine deterministic components (regex for structured MRNs, phone and date formats) with sequence-labeling models (BERT-family taggers) that catch the names, locations, and misspellings that rules miss; detected spans are then deleted, replaced with a category tag such as ⟨NAME⟩, or surrogated—swapped for realistic fake values that preserve date intervals and token statistics so downstream models still train sensibly.

The two error types have sharply asymmetric costs. A false negative—leaked PHI—is a privacy breach and a reportable event, so pipelines are tuned for very high recall on identifiers even at the expense of precision. But over-redaction is not free: aggressive scrubbing can delete clinically load-bearing tokens (an eponymous syndrome such as Wallenberg, a drug or device named after a person, a place-name embedded in a stent model) and quietly degrade the phenotyping the corpus was built to support. Report de-identification recall by PHI category, and hand-audit a held-out sample; one missed identifier across tens of thousands of notes can still trigger a disclosure. Crucially, de-identified is not the same as anonymous: residual quasi-identifiers (rare diagnosis + rare date + small town) permit re-identification by linkage, so a scrubbed corpus is still not safe to paste into an external LLM API without governance review. And for prospective, in-workflow tools that must read the live chart—extracting last-known-well at the bedside—de-identification is simply the wrong tool: those systems retain PHI by necessity and must instead run inside an approved, access-controlled, logged environment.

### Site shift and distribution shift

A model trained at one hospital routinely underperforms at another because the input distribution has moved—dataset shift—and it helps to name the flavors. Covariate shift: the inputs themselves differ (a different CT scanner vendor, reconstruction kernel, slice thickness, field strength, or contrast timing; note templates, abbreviations, and section headings that vary by EHR and service line) while the true feature→label mapping is unchanged. Label (prior) shift: outcome prevalence differs—a comprehensive stroke center sees more large-vessel occlusions than a primary center—which mis-calibrates thresholds even when discrimination is preserved. Concept shift: the feature→label relationship itself changes (coding practice, a guideline update, a new thrombectomy pathway that alters who gets imaged). The intuition is that networks latch onto whatever predicts the label in the training set, including scanner- or site-specific artifacts—the ‘an AI can tell which hospital a scan came from’ phenomenon—so a model can look excellent internally and fail externally with no bug at all.

Because site shift surfaces first as input drift (intensity histograms, noise level, token distributions) and only later as an outcome-metric drop, monitoring must watch the inputs, not just AUROC or Dice. Mitigations map to the flavor: external, multi-site validation before deployment is the non-negotiable evidence; harmonization and matched preprocessing (intensity standardization, template-agnostic parsing, matching the normalization used in pretraining) reduce covariate shift; per-site recalibration or threshold re-tuning addresses label shift; domain-adaptive continued pretraining helps when unlabeled in-domain data exist; and prospective drift monitoring with a rollback plan catches the shifts nobody anticipated—the vendor kernel update in §12.15 is the canonical case. No amount of internal cross-validation substitutes for a held-out external site.

Match architecture class to modality: CNN/ViT/U-Net for images; Transformer/SSM for long text or signals; Whisper/wav2vec for speech.

For detection/segmentation, publish IoU/Dice with confidence intervals and external sites.

For LLMs, measure hallucination, calibration, and workflow time—not only exam-style accuracy.

Prevent leakage from reports into imaging labels and from future notes into admission-time prediction.

Govern synthetic media and auto-generated text under clinical documentation policy.

## Connections

Nothing here escapes the training machinery of earlier chapters: mini-batch SGD/Adam, weight decay, dropout, and batch/layer normalization drive these models too; attention, convolution, and state-space layers are just structured layers inside the same optimization loop, and early stopping on patient-grouped validation remains mandatory.

Representation learning links backward to static embeddings (word2vec) and forward to the contextual embeddings produced by BERT/GPT encoders, which feed the same downstream classifiers, survival models, and calibrated risk scores developed for tabular data.

Every honest claim in this chapter rests on the evaluation discipline established elsewhere in the book: patient-level splits, leakage control (report→image, future→admission-time), calibration assessment, and external validation. A Transformer or a foundation model earns no exemption from any of these.

Measuring clinical impact is a different inferential target than measuring model accuracy: whether automated detection shortens door-to-needle time, or an LLM summary changes decisions, belongs to the causal-inference and study-design chapters (interrupted time series, stepped-wedge), not to a held-out AUROC.

Model selection is a portfolio decision spanning this chapter and the tabular-methods chapter—boosted trees for structured features, CNN/ViT/U-Net for images, Transformer/SSM for long text and signals, Whisper/wav2vec for speech—chosen by data size, latency, interpretability, and governance rather than by novelty or parameter count.

## Chapter Summary

Modern multimodal deep learning rests on attention and its efficient cousins. Seq2seq models gain from dot-product, additive Bahdanau, Luong, self-, and cross-attention; Transformers industrialize multi-head attention with positional encodings, residual blocks, and causal masking—illustrated by a numerical attention-weight example and a causal mask sketch. S4 and Mamba offer state-space alternatives for long sequences with near-linear scaling ambitions. LLMs span BERT-style encoders and derivatives (RoBERTa, DistilBERT, XLM), T5 text-to-text models, GPT decoder-only generators through instruction-tuned chat systems, and open Llama/Mistral MoE lines, adapted with RLHF, DPO, LoRA, and other PEFT methods and judged on general, coding, medical, and local workflow benchmarks. Systems issues—context length, RAG, quantization, hallucination control—matter as much as weight counts. Vision progresses from LeNet–AlexNet–VGG–Inception–ResNet classifiers to ViTs; detectors from R-CNN through Faster R-CNN to SSD and YOLO; segmenters from FCN/U-Net through Mask R-CNN and DeepLab to promptable SAM, with Dice, Hausdorff, FROC, and mAP interpreted clinically. NeRF and 3D Gaussian splatting synthesize novel views. Audio stacks include WaveNet, Tacotron, wav2vec, and Whisper. Multimodal stroke decision support should remain modular and prospectively evaluated. In stroke imaging and clinical NLP, architecture choice is secondary to labels, leakage control, external validation, MLOps drift monitoring, and safe human–AI workflow design.

## Practice and Reflection

(1) Write the scaled dot-product attention formula and explain the role of 1/√d_k.

(2) Contrast encoder-only, decoder-only, and encoder–decoder Transformers with one clinical NLP task suited to each.

(3) Why does causal masking matter for GPT-style training but not for BERT-style masked LM?

(4) Sketch how LoRA modifies a weight matrix and estimate parameter savings for rank r ≪ min(d_in, d_out).

(5) Compare two-stage Faster R-CNN with one-stage YOLO for on-call ICH flagging: accuracy vs latency trade-offs.

(6) Design a U-Net evaluation plan for DWI lesion volume including inter-rater reliability and site shift.

(7) Explain how NeRF rendering differs from predicting a 3D CT voxel grid with a 3D CNN.

(8) Propose a Whisper deployment test for neurology clinic dictation including medication error analysis.

(9) List three leakage patterns when training a multimodal model on paired CTA images and radiology reports to ‘predict LVO.’

(10) Outline an RLHF vs DPO choice for aligning a hospital-local note-summarization model, including data needed for each.

(11) Recompute the worked attention example with q=[0,1] and interpret the new α weights.

(12) Draft a one-page monitoring plan for production CTA detection covering data drift, performance drift, and rollback.
