# Chapter 10. Neural Networks and Deep Learning


![10 Mlp Architecture](../assets/figures/10_mlp_architecture.png)


## Opening

A vendor demo animates a CNN highlighting an infarct core. Deep learning can extract hierarchical image features; it can also memorize scanner fingerprints. This chapter gives neurologists the vocabulary to interrogate both.


![Simple multilayer network diagram (original teaching graphic).](../assets/figures/ml_fig_mlp.png)

*Simple multilayer network diagram (original teaching graphic).*
## Learning Objectives

State universal approximation ideas and contrast biological neurons with artificial units.

Build perceptrons and multilayer perceptrons; select activations for hidden and output layers.

Choose cost functions (MSE, cross-entropy, KL, Hellinger) matched to tasks.

Apply optimizers: SGD with momentum, Nesterov, Adagrad, RMSprop, and Adam.

Execute a numerical forward and backward pass; explain vanishing/exploding gradients.

Use initialization, batch normalization, gradient clipping, dropout, and early stopping.

Describe CNN convolution and architectures; RNN, LSTM, GRU, bidirectional and deep RNNs.

Connect deep learning practice to neurologic imaging and sequential clinical data with leakage controls.

## 10.1 Universal Approximation and the Path to Deep Learning

Linear and generalized linear models map features through a single affine transformation before a link function. Neural networks deepen this idea: they compose many affine maps with elementwise nonlinearities, allowing hierarchical feature construction. Early layers often capture local or generic structure; later layers capture task-specific abstractions. Deep learning emphasizes learning these representations from data end-to-end with gradient-based optimization rather than handcrafting every feature. The price is data hunger, computational cost, and a larger surface area for overfitting, leakage, and brittle deployment.

Universal approximation theorems state that a feedforward network with a single hidden layer of sufficient width and a nonpolynomial activation can approximate continuous functions on compact subsets of R^d arbitrarily well in uniform norm (classical results associated with Cybenko, Hornik, and others). The theorem is existential: it does not guarantee that gradient descent finds the approximating weights, nor that the required width is practical. Depth can be exponentially more parameter-efficient for certain structured functions (hierarchical composition, long dependency chains), which motivates deep architectures in practice. In medicine, depth is not an automatic virtue: when tabular predictors are few and well understood, a regularized linear model may match a deep net while remaining easier to audit.

A practical reading of universal approximation is therefore dual: (1) do not fear that neural nets are theoretically incapable of representing complex neurologic risk functions; (2) do not assume that existence implies learnability from your finite, biased, multi-site sample. Optimization landscape geometry, implicit regularization of SGD, and architectural inductive bias jointly determine what is found. Empirical risk minimization on hospital A can approximate a function that fails on hospital B if the support of the input distribution shifts—for example, different CT kernels or NIHSS documentation habits. Theory licenses flexibility; epidemiology still demands transportability tests.

## 10.2 Biological versus Artificial Neural Networks

Biological neurons integrate synaptic inputs on dendrites, fire action potentials when membrane potential crosses threshold, and communicate via chemical synapses with plasticity rules (Hebbian learning, spike-timing-dependent plasticity) far richer than a single scalar weight update. Brains are recurrent, spiking, energy-efficient, and develop under evolutionary and developmental constraints. Artificial neural networks (ANNs) borrow loose metaphors—units, connections, activation—but implement continuous differentiable maps trained by global loss minimization. Conflating the two leads to bad neuroscience and bad engineering. Useful transfers are limited: distributed representations, hierarchy, and robustness through redundancy. Claims that a clinical CNN ‘works like a radiologist’s visual cortex’ are marketing, not mechanism.

An artificial unit computes a weighted sum of inputs plus bias, then applies a nonlinear activation. Layers arrange units in parallel; depth stacks layers. Learning adjusts weights to reduce a cost on a dataset. Unlike biological learning, batch training shuffles i.i.d. samples (or carefully blocked clinical samples), uses reverse-mode automatic differentiation, and often requires labeled supervision at scales biology does not receive as explicit class tags. Self-supervised objectives (Chapter 11) partially close that gap by manufacturing targets from structure in unlabeled signals.

## 10.3 Perceptron and Multilayer Perceptron

The classical perceptron (Rosenblatt) is a binary linear classifier with a hard threshold: predict sign(w^T x + b). The perceptron learning rule updates weights when a training example is misclassified: w ← w + η y x for label y ∈ {−1, +1}. If data are linearly separable, the perceptron converges to a separating hyperplane; if not, it cycles. With sigmoid or identity activations, a single unit recovers logistic or linear regression as special cases of a one-layer network.

![10.1: A multilayer perceptron with three input units, one hidden layer of four ReLU units, and two output units, fully connect](../assets/figures/ml_concept_10.1_3dca7021.png)

*Figure 10.1 — original teaching graphic.*

The multilayer perceptron (MLP), or fully connected feedforward network, stacks layers: h^{(0)} = x, h^{(ℓ)} = φ(W^{(ℓ)} h^{(ℓ−1)} + b^{(ℓ)}), and an output layer produces scores or probabilities. Hidden width, depth, and activation choice define capacity. Matrix notation keeps implementations efficient: a batch of inputs X ∈ R^{n×d} multiplies weight matrices so hardware accelerates dense linear algebra. Residual connections add a block’s input to its output to ease gradient flow in very deep stacks. For tabular EHR inputs, MLPs with modest width, strong regularization, and careful preprocessing often suffice; architecture search rarely substitutes for clean labels and honest validation. Width versus depth trade-offs are empirical: very wide shallow nets can approximate well but may overfit differently than deep thin nets. In practice, start with two or three hidden layers of moderate width (e.g., 64–512 units), ReLU, and AdamW with weight decay, then adjust only when learning curves demand it.

## 10.4 Activation Functions

Nonlinear activations prevent composition of layers from collapsing into a single affine map. Logistic sigmoid σ(z) = 1/(1+e^{−z}) maps to (0,1) but saturates for large |z|, yielding tiny gradients. Hyperbolic tangent tanh maps to (−1,1) and is zero-centered. Rectified linear unit ReLU(z) = max(0, z) is piecewise linear, sparse in activation, and efficient; units can ‘die’ if they receive only negative pre-activations. Variants include Leaky ReLU (small slope for z < 0), parametric ReLU, ELU, and GELU (common in transformers), which smooth the negative region.

![Activation functions: sigmoid, tanh, ReLU, leaky ReLU (original).](../assets/figures/ml_fig_activations.png)

*Figure 10.2. Four activation functions evaluated over the pre-activation z. Sigmoid and tanh are saturating: their flat tails push the gradient toward zero and starve deep stacks of learning signal (tanh is at least zero-centred). ReLU and GELU are non-saturating, holding a slope near one for positive z (ReLU is exactly linear there and can 'die' for z < 0; GELU is smooth and approaches the identity), which keeps gradients flowing.*

At the output, activations match the task: linear for unbounded regression, sigmoid for independent binary labels, softmax for mutually exclusive classes. Softmax(z)_k = exp(z_k)/Σ_j exp(z_j) produces a probability simplex. Numerically stable softmax subtracts max(z) before exponentiation. Matching activation and loss (softmax with categorical cross-entropy) yields clean gradients of the form (p − y) for one-hot targets. When class imbalance is extreme, probabilities can collapse toward the majority; class weights, focal loss, or resampling then matter as much as the activation itself.

## 10.5 Neural Network Cost Functions

### Quadratic cost (mean squared error)

Mean squared error for a scalar target is (ŷ − y)², averaged over dimensions and batch. MSE is natural for regression and arises as the negative log-likelihood under Gaussian noise with fixed variance. For classification, MSE on one-hot labels can work but is less aligned with probability geometry than cross-entropy and can saturate learning when sigmoid outputs are confidently wrong (small gradients). In continuous neurologic outcomes (predicted NIHSS or infarct volume), check residual structure: heteroscedasticity and outliers may favor Huber or quantile losses over plain MSE.

### Cross-entropy

Binary cross-entropy for label y ∈ {0,1} and prediction p ∈ (0,1) is −[y log p + (1−y) log(1−p)]. Categorical cross-entropy for one-hot y and softmax p is −Σ_k y_k log p_k, reducing to −log p_c for true class c. Cross-entropy heavily penalizes confident wrong predictions and pairs cleanly with sigmoid/softmax. Label smoothing replaces hard one-hots with slightly softer targets to reduce overconfidence.

### Kullback–Leibler divergence

KL divergence KL(p ‖ q) = Σ_k p_k log(p_k / q_k) measures how distribution q diverges from reference p. For one-hot p, KL reduces to cross-entropy up to the constant entropy of p. In soft-label settings (knowledge distillation, probabilistic clinician labels), minimizing KL(p_teacher ‖ p_student) transfers full distributional information. KL is asymmetric: KL(p ‖ q) ≠ KL(q ‖ p); mode-seeking versus mass-covering behavior depends on direction—critical in variational inference (Chapter 11).

### Hellinger (Bhattacharyya) distance

Hellinger distance between discrete distributions p and q is H(p, q) = (1/√2) ‖√p − √q‖_2, equivalently related to Bhattacharyya coefficient BC(p, q) = Σ_k √(p_k q_k) via H² = 1 − BC. Hellinger is a true metric (symmetric, triangle inequality) bounded in [0, 1], unlike KL. It appears in distributional learning, robust statistics, and some generative-model evaluations as an alternative discrepancy. As a training cost it is less common than cross-entropy but useful when one wants a bounded, symmetric measure of probability error—for example comparing predicted risk strata distributions across hospitals.

## 10.6 Optimizers

Stochastic gradient descent approximates the full-batch gradient with a mini-batch average: θ ← θ − η ∇_θ L_batch. Noise can help escape sharp minima. Learning rate η is the most sensitive hyperparameter. Batch size interacts with learning rate: larger batches yield stabler gradients but may require warmup and higher η.

![10.3: Optimizer trajectories on an elongated quadratic bowl f = (1/2)(theta_1^2 + 9 theta_2^2), whose ill-conditioning forms a](../assets/figures/ml_concept_10.3_69b59ac6.png)

*Figure 10.3 — original teaching graphic.*

### Exponentially weighted averages, momentum, and Nesterov

An exponentially weighted moving average of a sequence a_t is v_t = β v_{t−1} + (1−β) a_t (or without the (1−β) factor in some SGD notations). Momentum accumulates velocity v ← β v + ∇L (or v ← β v + (1−β) ∇L depending on convention) and steps θ ← θ − η v, damping oscillations in narrow valleys and accelerating along consistent directions. Nesterov accelerated gradient evaluates the gradient at a lookahead point θ − η β v (conceptually), then updates velocity—often yielding faster convergence on smooth convex problems and improved deep-network training in classical vision pipelines.

### Adagrad, RMSprop, and Adam

Adagrad accumulates squared gradients G_t = G_{t−1} + g_t ⊙ g_t and scales steps as η / (√G_t + ε) ⊙ g_t, giving larger effective steps to rare features—helpful for sparse text—but G_t grows monotonically so learning rates can shrink to zero. RMSprop replaces the cumulative sum with an exponential moving average of squared gradients, E[g²]_t = ρ E[g²]_{t−1} + (1−ρ) g_t², preventing indefinite decay. Adam combines a momentum-like first moment m_t = β1 m_{t−1} + (1−β1) g_t with an RMSprop-like second moment v_t = β2 v_{t−1} + (1−β2) g_t². Because both moments are initialized at zero, early estimates are biased toward zero; Adam therefore forms bias-corrected values m̂ = m_t / (1−β1^t) and v̂ = v_t / (1−β2^t) before stepping with η m̂ / (√v̂ + ε). The correction matters only for the first tens of steps—when t is small the denominators (1−β^t) are well below one and inflate the raw moments back to their intended scale—so that training does not crawl at the start. Default hyperparameters (β1=0.9, β2=0.999, ε small) work surprisingly often. AdamW decouples weight decay from the adaptive step, improving generalization in many deep models. Vision often uses SGD with momentum and schedules; transformers frequently use AdamW. On small medical datasets, aggressive adaptive optimizers can overfit quickly; early stopping on a patient-grouped validation set is mandatory.

SGD+momentum: simple, strong with tuning; good generalization tradition in CNNs.

Nesterov: lookahead momentum; classical acceleration.

Adagrad: sparse-friendly; may stall from shrinking rates.

RMSprop: EMA of squared grads; stable adaptive steps.

Adam/AdamW: robust defaults; watch weight-decay decoupling and overfitting on small n.

## 10.7 Backpropagation: Forward and Backward Pass

Training minimizes loss by gradient descent on parameters. Backpropagation applies the chain rule from the loss backward through the computational graph. If L depends on z and z = w · h, then ∂L/∂w = (∂L/∂z) · h and ∂L/∂h = (∂L/∂z) · w (scalars for intuition; tensors use Jacobians). Each module implements a forward map and a local gradient rule; automatic differentiation frameworks generate these from the forward program.

![10.4: The worked forward and backward pass for the chapter's tiny network (inputs x = [1, 2], label y = 1). The forward pass (](../assets/figures/ml_concept_10.4_9fe947e7.png)

*Figure 10.4 — original teaching graphic.*

### Worked numerical forward and backward sketch

Consider a tiny network: two inputs, one hidden layer of two ReLU units, single sigmoid output for binary classification. Parameters: W1 maps R²→R² with rows w1_1 = [0.5, −0.3], w1_2 = [0.2, 0.4], b1 = [0.1, −0.2]; W2 = [0.6, −0.5], b2 = 0.0. Input x = [1.0, 2.0], label y = 1.

FORWARD. Pre-activation z1 = W1 x + b1. Unit 1: 0.5·1 + (−0.3)·2 + 0.1 = 0.0; unit 2: 0.2·1 + 0.4·2 − 0.2 = 0.8. Hidden h = ReLU(z1) = [0.0, 0.8]. Output logit z2 = 0.6·0 + (−0.5)·0.8 + 0 = −0.4. Prediction p = σ(−0.4) ≈ 0.4013. Binary cross-entropy L = −log p ≈ 0.913.

BACKWARD. dL/dz2 = p − y ≈ 0.4013 − 1 = −0.5987. Gradients: dL/dW2 = (dL/dz2) · h^T gives [0, −0.5987·0.8] ≈ [0, −0.479]; dL/db2 = dL/dz2. Upstream to hidden: dL/dh = W2^T (dL/dz2) ≈ [0.6, −0.5]^T · (−0.5987) ≈ [−0.359, 0.299]. ReLU gate: dL/dz1 = dL/dh ⊙ 1[z1>0] = [0, 0.299] because first unit was zero. Then dL/dW1 = (dL/dz1) x^T and dL/db1 = dL/dz1. A gradient step W ← W − η ∇W decreases L for small η (e.g., 0.1). This sketch is the arithmetic automatic differentiation performs at scale. Verify shapes: if any dimension mismatches, the graph is wrong.

### The training loop as an algorithm

One backward pass yields gradients for a single mini-batch; training composes many such steps under an optimizer and a stopping rule. The canonical supervised loop makes the moving parts—and the clinical safeguards—explicit:

```
# Supervised training with backprop and early stopping
initialize θ # He/Glorot init matched to activation
best_val ← ∞ ; patience_left ← P
for epoch in 1..max_epochs:
 for each mini-batch (X, y) in shuffled training data:
 yhat ← forward(X, θ) # cache activations for the backward pass
 L ← loss(yhat, y) # e.g. cross-entropy
 g ← backward(L, θ) # reverse-mode autodiff gives ∂L/∂θ
 g ← clip_global_norm(g, c) # optional; standard for RNNs/Transformers
 θ ← optimizer_step(θ, g, η) # SGD+momentum / RMSprop / AdamW
 val ← loss(forward(X_val, θ), y_val) # patient- and site-grouped split
 if val < best_val − δ:
 best_val ← val ; θ* ← θ ; patience_left ← P
 else:
 patience_left ← patience_left − 1
 if patience_left == 0: break # early stopping
return θ* # checkpoint at best validation loss
```

Every clinical safeguard maps to a line here: the split that produces X_val must be grouped by patient and site rather than by random row, initialization is chosen for the activation, clipping guards recurrent stacks, and the returned parameters are the best validation checkpoint—not the last epoch, which has usually begun to overfit on a small cohort.

## 10.8 Regularization and Training Stability

### Vanishing and exploding gradients

Error signals flow backward multiplied by local Jacobians of activations and weight matrices. Vanishing gradients occur when many factors have magnitude less than one (deep sigmoid stacks, long unrolled RNNs), starving early layers of learning signal. Exploding gradients occur when factors exceed one, causing NaNs and divergent updates. Architectural choices—ReLU, residual links, normalization, gated RNNs, careful initialization—stabilize scales. Gradient clipping caps update magnitude before the optimizer consumes the gradient. Clipping by global norm rescales the entire gradient vector when it grows too long: if ‖g‖₂ > c, set g ← g · (c / ‖g‖₂), which preserves the descent direction while bounding the step; clipping by value instead clamps each component into [−c, c], which can distort direction. The global-norm form is standard in recurrent and language models, where a single exploding step could otherwise overwrite weights learned across many good steps. Choose c by inspecting the empirical distribution of gradient norms early in training rather than by folklore.

![Vanishing gradients vs residual skip highway (synthetic teaching; original).](../assets/figures/ml_fig_vanishing_residual.png)

*Figure — Left: relative backprop signal versus depth on a log scale. A deep sigmoid stack multiplies many factors ≪ 1, so early-layer gradients collapse; tanh decays more slowly; open ReLUs keep more signal. Right: a residual block adds an identity skip, so ∂L/∂h retains a direct +1 path and deep stacks can start near identity maps. Residual links do not license unlimited depth on a 200-patient MRI cohort—capacity still needs data (next figure).*

![Capacity vs sample size: validation error for low / medium / high capacity (synthetic; original).](../assets/figures/ml_fig_capacity_vs_n.png)

*Figure — Synthetic validation-error curves against training n (log scale). At small n, low-capacity regularized models win; high-capacity nets only pull ahead once n is large. Many stroke and single-center imaging cohorts live on the left half of this plot—depth is not free.*

### Weight initialization

Initialization sets signal scale at the start of training. Too-large weights explode activations; too-small vanish. Xavier/Glorot initialization scales variance from fan-in and fan-out for tanh-like activations; He/Kaiming initialization accounts for ReLU’s half-rectification. Residual branches are sometimes initialized near zero so deep nets start close to identity maps. When transferring ImageNet-pretrained weights to head CT, input-channel adaptation (replicating grayscale to three channels or reinitializing the first layer) must be intentional.

### Batch normalization and other normalizations

Batch normalization normalizes layer inputs across the mini-batch to zero mean and unit variance, then applies learned scale and shift. It stabilizes activation distributions, allows higher learning rates, and acts as mild regularization. At inference, running averages of mean and variance replace batch statistics. Layer normalization normalizes across features for each example—preferred in transformers. Related ideas include GroupNorm, InstanceNorm, and RMSNorm. In multi-site imaging, normalization layers can absorb scanner-specific shifts; this may help or harm transportability depending on whether site identity is entangled with the clinical label.

![BatchNorm: small-batch moment noise and train vs eval paths (original).](../assets/figures/ml_fig_batchnorm.png)

*Figure — BN discipline. **Left:** expected error of batch mean/variance vs population moments shrinks as batch size grows—tiny per-GPU batches make BN stats noisy and site-composition dependent. **Right:** schematic train path (batch μ,σ) vs eval path (running averages); serving must freeze running stats. Prefer GroupNorm/LayerNorm when batches are tiny or multi-site transport matters.*

### Dropout and early stopping

Dropout randomly zeroes activations during training with probability p, forcing redundant representations; at test time weights are scaled (or inverted dropout scales at train time) so expectations match. Weight decay adds (λ/2)‖θ‖² or multiplies weights by a factor slightly less than one each step. Early stopping monitors validation loss and halts when it stops improving—simple, effective, and mandatory on small clinical cohorts. Learning-rate schedules (step decay, cosine annealing, warmup) reshape η over time. Mixed precision, gradient accumulation, and distributed data parallelism address scale.

![Early stopping on train vs validation loss — mandatory regularizer for small clinical n (synthetic; original).](../assets/figures/ml_fig_early_stopping.png)

*Figure — Early stopping in deep training. Validation loss bottoms while training loss keeps falling; restore the best-validation checkpoint. On small multi-site MRI or tabular cohorts, patient-grouped validation and early stopping often beat architectural cleverness.*

Start with a simple MLP or linear baseline before complex architectures.

Monitor train versus validation loss to detect under/overfitting.

Tune learning rate first; then regularization strength.

Verify gradients on a tiny batch (loss decreases, no NaNs).

Split by patient (and site) before any normalization or augmentation fit.

## 10.9 Convolutional Neural Networks

Convolutional layers exploit spatial structure. A filter (kernel) of small spatial size slides over the input, computing local dot products and producing feature maps. Strictly, many frameworks implement cross-correlation (no kernel flip) but call it convolution; for learned filters the distinction is absorbed into the weights. Weight sharing means the same filter detects a pattern regardless of location—translation equivariance of feature maps. Multiple filters learn diverse patterns. Stacking convolutions with nonlinearities and pooling (or strided convolutions) builds larger receptive fields.

![10.5: A convolutional stage on a 6x6 input. A 3x3 kernel (the highlighted receptive field) slides with stride 1 to produce a 4](../assets/figures/ml_concept_10.5_626b3e54.png)

*Figure 10.5 — original teaching graphic.*

### CNN architecture

Typical vision stacks interleave conv → norm → activation → optional pooling, increasing channel depth while decreasing spatial resolution, then attach a classifier head (global pool + dense) or a dense prediction head (segmentation). Padding controls border effects; stride controls downsampling. Parameter counts grow with kernel size and channel product but remain far smaller than dense layers on flattened images. Classic stage patterns (more channels, less resolution) appear from LeNet through ResNet (Chapter 12). Residual blocks add skip connections around stacked convolutions.

### Types of convolutions

Standard 2D convolution operates on image planes. 3D convolution processes volumetric MRI/CT at higher memory cost; 2.5D multi-slice inputs are a practical stroke-imaging compromise. Depthwise separable convolutions (depthwise spatial filter per channel then 1×1 pointwise mix) reduce parameters (MobileNet-style). Dilated (atrous) convolutions insert gaps in kernels to expand receptive field without pooling—central to DeepLab segmentation. Transposed convolutions upsample feature maps for decoders. Grouped convolutions partition channels; 1×1 convolutions mix channels without spatial extent. Graph and spherical convolutions extend the idea beyond Euclidean grids.

## 10.10 Recurrent Neural Networks

Sequential data (text, time series, audio frames, EEG windows) invite models with temporal state. A basic RNN updates hidden state h_t = φ(W_h h_{t−1} + W_x x_t + b). The same weights apply at every time, sharing statistical strength across positions. Unrolling through time yields a deep computational graph; backpropagation through time (BPTT) trains it. Vanilla RNNs struggle with long-range dependencies due to vanishing gradients.

### LSTM and GRU

A vanilla RNN overwrites its whole hidden state each step, so information must survive repeated multiplication by W_h and repeated squashing by φ; across long horizons the product of Jacobians shrinks toward zero (vanishing) or blows up (exploding). Gated cells fix this by adding an explicitly protected memory and learned valves that decide what to keep, overwrite, and expose.

![10.6: An LSTM cell. The cell state c_t runs along the top as a protected memory highway, altered only by a forget-gate element](../assets/figures/ml_concept_10.6_82d22f93.png)

*Figure 10.6 — original teaching graphic.*

Long short-term memory (LSTM) carries a cell state c_t (protected long-term memory) alongside the hidden state h_t (the exposed working copy). Writing [h_{t−1}, x_t] for the concatenated previous hidden state and current input, each step computes three sigmoid gates in (0,1) and one tanh candidate:

forget gate f_t = σ(W_f·[h_{t−1}, x_t] + b_f): fraction of the old cell state to retain;

input gate i_t = σ(W_i·[h_{t−1}, x_t] + b_i): how much new content to write;

candidate c̃_t = tanh(W_c·[h_{t−1}, x_t] + b_c): the proposed new content;

output gate o_t = σ(W_o·[h_{t−1}, x_t] + b_o): how much of the cell to expose.

The cell updates additively, c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t, and the hidden state is h_t = o_t ⊙ tanh(c_t). The additive update is the mechanism that matters. Along the direct path ∂c_t/∂c_{t−1} ≈ f_t elementwise, so a forget gate held near 1 lets the gradient flow backward across hundreds of steps almost undamped—the ‘constant error carousel’—instead of decaying through the repeated W_h multiplications of a vanilla RNN. The network learns f_t ≈ 1, i_t ≈ 0 to preserve a marker (say, a seizure-onset feature seen early in an EEG window) and f_t ≈ 0 to reset when a new segment begins. Initializing the forget-gate bias positive (≈ +1) makes cells default to remembering, which noticeably speeds early training.

Gated recurrent units (GRU) merge cell and hidden state and use two gates: an update gate z_t that interpolates between the previous state and a candidate, h_t = (1−z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t, and a reset gate r_t controlling how much past state enters the candidate h̃_t = tanh(W·[r_t ⊙ h_{t−1}, x_t]). With three weight matrices instead of four, a GRU holds roughly three-quarters of an LSTM’s parameters, trains faster, and often matches LSTM accuracy on the smaller datasets typical of clinical work; LSTMs can edge ahead when very long, precise memory is required. Both were the workhorses of sequence modeling before large transformers and still appear in streaming monitors, small-footprint on-device models, and hybrids. Teacher forcing feeds ground-truth previous tokens during training of generative RNNs; exposure bias can arise at test time when the model instead consumes its own outputs.

### Bidirectional and deep RNNs

Bidirectional RNNs run forward and backward recurrences and concatenate states so each position sees left and right context—standard for offline sequence labeling (named entity recognition in notes, speech frame classification) but unsuitable when future context must not leak (true causal forecasting, real-time streaming). Deep RNNs stack multiple recurrent layers so the hidden state of layer ℓ is the input sequence to layer ℓ+1, increasing representational capacity at the cost of harder optimization; residual or dense connections between layers help. For irregularly sampled EHR sequences, time-aware variants, continuous-time models, or interpolation to a common grid are common engineering choices.

## 10.11 Training Schedules, Capacity, and Practical Diagnostics

Beyond choosing Adam or SGD, successful training depends on schedules and diagnostics. Learning-rate warmup starts η near zero and ramps over a few hundred or thousand steps so that randomly initialized layers do not take destructive early updates—especially important with large batches and LayerNorm/Transformer stacks. Cosine decay or step decay then reduces η so optimization settles. Plateau schedulers reduce η when validation loss stalls. Gradient accumulation simulates large batches on memory-limited GPUs by summing gradients over micro-batches before an optimizer step—critical for 3D CNNs on stroke CT.

Capacity control is not only dropout. Label smoothing softens one-hot targets. Mixup and CutMix interpolate inputs and labels for vision regularization. Stochastic depth randomly drops residual blocks during training. For small clinical n, the strongest regularizer is often architectural humility: fewer layers, pretrained encoders with frozen early stages, and heavy early stopping. Plot learning curves: if training loss is high, increase capacity or training time; if training loss is near zero while validation rises, reduce capacity or add regularization. A useful example diagnostic is the ‘small batch overfit test’: if the network cannot drive loss down on a 16-example subset, the implementation or learning rate is wrong before any clinical conclusion is drawn.

Reproducibility requires fixed seeds where possible, logged configs, dataset version hashes, and explicit documentation of windowing and resampling. Deterministic GPU kernels are not always available; report variance across seeds for small cohorts. When fine-tuning imaging models, unfreeze stages gradually: train the head, then last blocks, then full net with a lower η. This staged example procedure reduces catastrophic forgetting of pretrained filters on tiny labeled hemorrhage sets.

## 10.12 Worked Example: Mini-Batch SGD Update on Logistic Output

Consider a single linear unit with sigmoid output used as a one-layer network for binary classification—logistic regression as a neural special case. Parameters w = [0.5, −1.0], b = 0.0. Mini-batch of two examples: x1 = [1.0, 0.0], y1 = 1; x2 = [0.0, 1.0], y2 = 0. Learning rate η = 0.5.

Forward for x1: z1 = 0.5·1 + (−1)·0 + 0 = 0.5, p1 = σ(0.5) ≈ 0.6225, loss1 = −log p1 ≈ 0.474. Forward for x2: z2 = 0.5·0 + (−1)·1 = −1.0, p2 = σ(−1) ≈ 0.2689, loss2 = −log(1−p2) ≈ 0.313. Mean loss L ≈ 0.394. Gradients for logistic loss: ∂L/∂z = p − y, so g_z1 ≈ 0.6225 − 1 = −0.3775, g_z2 ≈ 0.2689 − 0 = 0.2689. Average gradient for w is (1/2)[g_z1 x1 + g_z2 x2] ≈ 0.5 · ([−0.3775, 0] + [0, 0.2689]) = [−0.1888, 0.1345]. Average gradient for b is 0.5·(−0.3775+0.2689) ≈ −0.0543. Update: w ← w − η ∇w ≈ [0.5, −1.0] − 0.5·[−0.1888, 0.1345] ≈ [0.5944, −1.0672], b ← 0 − 0.5·(−0.0543) ≈ 0.0272. Recompute losses with new parameters to verify L decreased—this is the same arithmetic deep networks perform layerwise via backpropagation.

## 10.13 From MLPs to Modern Blocks: Residuals, Depth, and Inductive Bias

Deep plain MLPs become hard to optimize as depth grows: signal and gradient scales drift. Residual connections change the hypothesis class to favor near-identity mappings, which empirically enables very deep vision and language models. Dense connections concatenate feature maps from earlier layers; highway networks gate the skip path. These motifs appear inside CNN stages and Transformer blocks (Chapter 12). Inductive bias—assumptions baked into architecture—explains why CNNs beat MLPs on small image datasets: local connectivity and weight sharing match the statistics of natural and medical images.

When is an MLP still preferred? Tabular registries with tens of engineered features, strong domain knowledge, and n in the low thousands often favor gradient-boosted trees (Chapter 9) or shallow MLPs over deep nets. Deep learning shines when raw high-dimensional signals contain hierarchical structure that feature engineering cannot cheaply capture. Hybrid pipelines—CNN embeddings of imaging plus boosted trees on tabular covariates—are common in multi-center stroke outcome models and should be validated as a single pipeline with joint leakage control.

Numerical stability checklist: subtract max before softmax; clamp probabilities before log; use float32 or mixed precision with loss scaling; clip gradients by global norm (e.g., 1.0 or 5.0) if loss spikes; verify that BatchNorm runs in eval mode at test time so running statistics are used. An example failure mode in multi-site CT is training with tiny per-GPU batches so BatchNorm statistics become noisy and encode site batch composition; GroupNorm or LayerNorm may transport better.

## Clinical and Epidemiologic Notes

Deep networks dominate when inputs are images, waveforms, or long free text; they are optional on small tabular registries. For noncontrast CT hemorrhage detection, DWI lesion segmentation, or CTA LVO classification, CNNs (and later vision transformers) learn filters that replace handcrafted radiomics—if labels are trustworthy and validation is site-aware. Prefer 3D or multi-slice context for intracranial pathology that is not confined to a single axial cut, but measure memory and slice-selection bias explicitly.

Preprocessing is half the model: windowing and leveling for CT, bias-field correction for MRI, resampling to common spacing, and skull stripping change difficulty more than swapping Adam for SGD. Augmentation (flips, small rotations, intensity jitter) must respect clinical laterality semantics: random horizontal flips can destroy left–right syndrome structure unless labels and metadata are transformed consistently. Never fit normalization statistics on a pool that includes test patients.

Sequential models for vitals, NIHSS trajectories, or EEG should define index time and prediction horizon before architecture choice. Leakage includes using future GCS to predict ‘deterioration’ labeled from the same future window. Report calibration of predicted probabilities for triage thresholds; high AUROC with miscalibrated risk still misleads capacity planning. External validation across scanner vendors and hospitals is the clinical analogue of domain generalization—batch norm and site-specific intensity shifts can silently encode hospital identity.

Interpretability requests from stroke committees often ask which pixels or time steps drove a score. Saliency maps and Grad-CAM on CNNs can highlight hemorrhage-density regions yet are sensitive to architecture and can highlight non-causal correlations (e.g., laterality markers of scanner positioning). Treat explanation tools as hypothesis generators for error analysis, not as legal justifications. Pair them with stratified error tables by age, sex, scanner, and transfer status.

Compute budgets shape science. A 3D CNN that barely fits on one GPU may force smaller validation cohorts or fewer seeds, underestimating uncertainty. Prefer pre-registered analysis plans that specify architecture family, not endless post-hoc search on the test set. When deep models and logistic baselines show similar AUROC, prefer the simpler model for deployment unless a clinically meaningful metric (time-to-decision, net benefit at a fixed sensitivity) favors the deep system in prospective evaluation.

Use deep models when representation learning beats hand features; otherwise prefer simpler baselines.

Align loss and output activation with the clinical decision (detection vs volume regression).

Group splits by patient and site; freeze preprocessing fit to training folds only.

Monitor gradient health (norms, NaNs) when training 3D CNNs on small stroke cohorts.

Document windowing, resampling, and augmentation as part of the scientific protocol.

Report calibration and site-stratified performance alongside discrimination metrics.

Pre-register primary metrics; limit test-set peeking during architecture search.

## Connections

To linear and generalized linear models: a single sigmoid or softmax unit is exactly logistic or multinomial regression. Neural networks keep the same likelihoods and links but insert learned nonlinear features beforehand, which is why cross-entropy and its (p − y) logit gradient reappear unchanged.

To tree ensembles (Chapter 9): boosting and MLPs are competing function approximators. Gradient-boosted trees usually win on small, heterogeneous tabular registries; deep nets win on raw images, waveforms, and text. Hybrid pipelines fuse CNN image embeddings with boosted trees on covariates and must be leakage-controlled as one pipeline.

To regularization and the bias–variance trade-off: dropout, weight decay, early stopping, and augmentation are the deep-learning realizations of classical capacity control. The small-n clinical regime turns them from optional tuning into decisive design choices.

To representation and generative learning (Chapter 11): self-supervised pretraining manufactures the labels biology never supplies as class tags, and the KL geometry introduced among the loss functions here becomes the engine of variational objectives.

To modern architectures (Chapter 12): the residual, normalization, and gating motifs taught here are the literal building blocks of ResNets and Transformers, where attention supplants recurrence for long-range dependencies.

To evaluation and epidemiology: universal approximation guarantees expressivity, never transportability. Calibration, site-stratified validation, and leakage control from the study-design chapters decide whether a high-capacity model survives a new scanner or hospital.

## Chapter Summary

Neural networks compose affine maps and nonlinear activations to learn hierarchical features; universal approximation justifies richness while depth often buys parameter efficiency. Artificial units are useful engineering abstractions, not faithful cortical simulations. Perceptrons and MLPs provide the feedforward backbone; activations (sigmoid, tanh, ReLU family, softmax) control expressivity and gradient flow. Costs include MSE, cross-entropy, KL divergence, and Hellinger distance, each with different geometry. Optimizers range from SGD with momentum and Nesterov acceleration to Adagrad, RMSprop, and Adam. Backpropagation implements the chain rule; a tiny numerical forward/backward example makes gradients concrete. A second worked mini-batch logistic update shows the same gradient arithmetic used inside deep stacks. Training stability uses careful initialization, batch/layer norm, clipping, dropout, schedules, and early stopping to fight vanishing and exploding gradients. CNNs exploit spatial weight sharing with 2D/3D, dilated, depthwise, and transposed convolutions; RNNs, LSTMs, GRUs, and bidirectional/deep stacks model sequences before Transformers take over long-range language tasks. Residual inductive bias and hybrid CNN-plus-tabular pipelines appear throughout clinical imaging. In neurologic imaging and longitudinal care, preprocessing, index time, site structure, calibration, and honest comparison to simple baselines determine whether deep learning helps patients or only leaderboards.

## Practice and Reflection

(1) Derive why a deep stack of linear layers without nonlinearities is equivalent to a single linear map.

(2) For the worked network, recompute the forward pass with x=[0.5, 0.5] and y=0; does L increase or decrease after one SGD step with η=0.1 on W2 only?

(3) Show that the gradient of binary cross-entropy with sigmoid output equals (p−y) with respect to the logit.

(4) Compare Adagrad and RMSprop update rules and explain when Adagrad’s monotonically growing denominator is harmful.

(5) Design a CNN input pipeline for NCCT ICH detection including windowing, split strategy, and augmentation constraints for laterality.

(6) Explain vanishing gradients in a vanilla RNN over T=100 steps with tanh activations, and how LSTM gates mitigate the issue.

(7) When would you prefer Hellinger distance to KL divergence for comparing two predicted risk distributions across hospitals?

(8) Outline an early-stopping protocol with patient-grouped validation for a small multi-site MRI dataset.

(9) Recompute the mini-batch logistic worked example with η=1.0 and comment on whether the mean loss is guaranteed to decrease.

(10) List three ways BatchNorm can harm multi-site transportability and propose a normalization alternative for each.

(11) Compare parameter counts order-of-magnitude for a dense layer on a 256×256 flattened image versus a 3×3 conv with 32 filters on the same spatial grid.
