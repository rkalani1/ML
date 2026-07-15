# Chapter 11. Self-Supervised Deep Learning

## Opening
![Pretext pretrain then clinical fine-tune (original).](../assets/figures/swarm3h_ssl_pipeline.png)

*Pretext pretrain then clinical fine-tune (original).*


Labeled stroke images are scarce; unlabeled scans are abundant. Self-supervised pretraining looks attractive until domain shift between scanners and protocols is measured. Read this before believing ‘we barely needed labels.’


![Pretrain then fine-tune pipeline (original teaching graphic).](../assets/figures/ml_fig_pretrain_finetune.png)

*Pretrain then fine-tune pipeline (original teaching graphic).*
## Learning Objectives

Distinguish generative vs discriminative and deterministic vs stochastic models in representation learning.

Explain SOMs, Boltzmann machines, RBMs, DBNs, and DBMs at a conceptual and training level.

Build and contrast autoencoder families: sparse, denoising, contractive, stacked, VAE, and U-Net.

Train and diagnose GANs; use IS/FID; describe CGAN, DCGAN, WGAN/WGAN-GP, Pix2Pix, CycleGAN, StyleGAN.

Apply contrastive and triplet losses and Siamese networks for metric learning.

Summarize text-to-image stacks: zero-shot, autoregressive, diffusion, CLIP, VQ-GAN, DALL-E/Imagen/Parti/Stable Diffusion concepts.

Trace the core training loops—contrastive divergence, adversarial min–max, and InfoNCE—and connect energy-based (RBM), adversarial (GAN), and score-based (diffusion) views of generative modeling.

Design self-supervised pretraining strategies for unlabeled medical data with leakage-aware evaluation.

## 11.1 Representation Learning Concepts

Labeled medical data are expensive; unlabeled scans, waveforms, and notes are abundant. Self-supervised learning (SSL) constructs pretext tasks from the structure of unlabeled inputs—predicting masked parts, contrasting augmented views, or modeling data density—so that learned representations transfer to downstream supervised tasks with fewer labels. The chapter surveys classical energy-based and autoencoding methods, adversarial generators, contrastive metric learning, and modern multimodal text-to-image systems that combine these ideas at scale. Throughout, keep a clinical ledger: what was unlabeled, what pretext was optimized, what downstream endpoint was pre-specified, and whether patients in pretraining could appear in evaluation.

Self-supervision sits between classical unsupervised learning (clustering, PCA) and fully supervised deep learning. Unlike pure clustering, SSL usually optimizes a predictive loss with automatic labels; unlike supervised learning, those labels are not the clinical endpoint. Semi-supervised learning mixes few true labels with many unlabeled points and can combine SSL pretraining with consistency regularization. Generative AI in the popular sense (chat and image synthesis) often rests on self-supervised pretraining objectives—next-token prediction is itself a pretext task at massive scale.

### Generative versus discriminative models

Discriminative models estimate p(y | x) or a decision boundary directly—logistic regression, most supervised deep classifiers. Generative models estimate p(x), p(x, y), or a process that samples x (and possibly y). Generative models support synthesis, imputation, anomaly scoring via low likelihood, and semi-supervised learning by sharing a density model across classes. The boundary is porous: classifiers with generative pretraining and conditional generators that need labels both appear in practice. For stroke imaging, a discriminative CNN may detect ICH, while a generative model might synthesize rare lesion patterns for augmentation or flag out-of-distribution scanners via reconstruction error.

### Deterministic versus stochastic models

A deterministic encoder maps x to a fixed code z = f_θ(x). A stochastic encoder defines a distribution q_φ(z | x), sampling z at training time (VAEs, many Bayesian deep nets). Stochasticity regularizes, enables latent-space sampling, and expresses uncertainty, but complicates optimization (reparameterization tricks, score-function estimators). Discriminative predictors can also be stochastic at test time (Monte Carlo dropout) to approximate predictive uncertainty—useful when a model should abstain on unfamiliar MRI protocols rather than emit a confident wrong LVO call.

## 11.2 Self-Organizing Maps

Self-organizing maps (SOMs, Kohonen maps) are unsupervised neural models that map high-dimensional inputs onto a low-dimensional (usually 2D) grid of prototype vectors while preserving topology: nearby grid nodes represent similar inputs. Training presents an input x, finds the best-matching unit (BMU) by minimum distance, and updates the BMU and its grid neighbors toward x with a shrinking neighborhood kernel and learning rate. The result is a discrete manifold useful for visualization, clustering, and novelty detection when quantization error is high.

SOMs are not typically deep or end-to-end differentiable in the modern SSL sense, but they remain pedagogically important as topology-preserving representation learning and still appear in exploratory clinical analytics—mapping hospitals or patients into 2D grids for quality review. Limitations include sensitivity to initialization and grid size, lack of a natural probabilistic density, and weaker performance than deep autoencoders or contrastive methods on raw images. Think of SOMs as organized vector quantization with neighborhood cooperation. When presenting SOMs in clinical analytics meetings, emphasize that grid adjacency is a visualization aid, not a causal map of disease pathways; neighboring nodes share feature similarity, not necessarily shared etiology. Quantization error heatmaps can still flag unusual cases for chart review—an inexpensive anomaly screen before heavier deep models. Still, for raw imaging pixels, prefer deep autoencoders or contrastive SSL over SOMs as the primary representation tool.

## 11.3 Boltzmann Machines, RBMs, DBNs, and DBMs

Boltzmann machines are stochastic energy-based models over binary (or continuous) units with energy E(v, h) and joint probability p(v, h) ∝ exp(−E(v, h)). Visible units v correspond to data; hidden units h capture dependencies. Full Boltzmann machines with unrestricted lateral connections are intractable to train because the partition function Z sums over exponentially many configurations.

### Restricted Boltzmann machines

Restricted Boltzmann machines (RBMs) remove intra-layer connections so that hidden units are conditionally independent given visibles and vice versa. This bipartite structure enables efficient block Gibbs sampling: sample h ~ p(h | v), then v’ ~ p(v | h). The intractable Z still appears in the likelihood gradient, motivating contrastive divergence (CD-k): run k steps of Gibbs sampling from data-initialized visibles and approximate the model expectation with those fantasy particles. Persistent CD keeps chains running across updates. RBMs can model binary data or use Gaussian visibles for continuous inputs.

```
# Contrastive Divergence CD-k for an RBM (learning rate η)
for each minibatch V0 of visible vectors (the data):
 P0 = p(h | V0) # positive phase: hidden probabilities from data
 Vk = V0
 repeat k times: # k Gibbs steps; often k = 1
 Hk ~ Bernoulli(p(h | Vk)) # sample hiddens
 Vk ~ p(v | Hk) # reconstruct visibles ("fantasy" particles)
 Pk = p(h | Vk) # negative phase: hidden probabilities from model
 ΔW += η ( V0ᵀ P0 − Vkᵀ Pk ) # data correlations − model correlations
 Δb_v += η ( V0 − Vk )
 Δb_h += η ( P0 − Pk )
```

The update is the difference between a data-driven (“positive”) term and a model-driven (“negative”) term; because the intractable Z affects both, its gradient cancels and only the k-step fantasy particles remain to be computed.

### Deep belief networks and deep Boltzmann machines

Deep belief networks (DBNs) stack RBMs greedily: train a bottom RBM, freeze it, train another RBM on hidden activations as data, and so on, then optionally fine-tune the stack as a generative model or as initialization for a supervised MLP. Deep Boltzmann machines (DBMs) are undirected deep models with bidirectional connections between layers, trained with more sophisticated approximate inference than greedy DBN stacking. Historically, DBN pretraining helped deep nets before ReLU, batch norm, and large labeled datasets made pure supervised training reliable. Today pure RBM stacks are less common, but energy-based thinking and layerwise pretraining ideas reappear in modern SSL and generative modeling.

## 11.4 Autoencoders

An autoencoder (AE) learns to reconstruct input x through a bottleneck: encoder z = f_θ(x), decoder x̂ = g_φ(z), trained to minimize a reconstruction loss ‖x − x̂‖² or cross-entropy for Bernoulli data. If the bottleneck is too wide and capacity unrestricted, the AE can cheat by learning an identity map with no useful compression—hence constraints: narrow codes, noise, sparsity, contractive penalties, or stochastic latents.

![11.1: Autoencoder and variational-autoencoder architecture. An encoder q_φ(z|x) compresses the input x through narrowing layer](../assets/figures/ml_concept_11.1_05730688.png)

*Figure 11.1 — original teaching graphic.*

### Sparse, denoising, contractive, and stacked autoencoders

Sparse autoencoders add penalties so that hidden units fire rarely (L1 on activations or KL from a low target firing rate), yielding parts-based codes. Denoising autoencoders corrupt inputs with noise (masking, Gaussian noise) and reconstruct the clean x, forcing the model to capture structure rather than copy pixels. Contractive autoencoders penalize the Frobenius norm of the encoder Jacobian, encouraging flatness of the representation to input perturbations—related in spirit to robustness. Stacked autoencoders train layers greedily as AEs then fine-tune; they were an early deep unsupervised pipeline analogous to DBN stacking.

### Variational autoencoders (VAE)

VAEs treat latents as random variables. An encoder network outputs parameters of q_φ(z | x) (typically diagonal Gaussian mean and log-variance); a decoder defines p_θ(x | z). Training maximizes an evidence lower bound (ELBO): reconstruction term E_{q}[log p_θ(x | z)] minus KL(q_φ(z | x) ‖ p(z)) with prior p(z)=N(0,I). The reparameterization trick z = μ + σ ⊙ ε, ε~N(0,I) allows low-variance gradients through sampling. VAEs generate new samples by drawing z from the prior and decoding; they tend to produce blurrier images than GANs but offer structured latents and a principled probabilistic objective. β-VAE multiplies the KL term to encourage disentanglement.

### U-Net as an autoencoding architecture for dense prediction

U-Net pairs a contracting encoder path with an expanding decoder path and concatenative skip connections that shuttle high-resolution features to the decoder. Although often trained with dense supervised labels (segmentation), U-Net is architecturally an autoencoder variant optimized for localization. In self-supervised settings, U-Nets reconstruct masked image patches, denoise, or predict context—pretraining backbones for scarce pixel labels. For stroke DWI or CTP maps, U-Net-style skips preserve lesion boundaries that pure bottlenecks would blur.

## 11.5 Generative Adversarial Networks

A generative adversarial network (GAN) pits a generator G(z) that maps noise to samples against a discriminator D(x) that scores real versus fake. In the original formulation, D maximizes log D(x) + log(1 − D(G(z))) while G minimizes log(1 − D(G(z))) or, more practically, maximizes log D(G(z)) (non-saturating heuristic). Training alternates discriminator and generator steps. At optimality under infinite capacity, G matches the data distribution and D is chance—though practice never reaches this ideal cleanly.

![11.2: The generative-adversarial-network training loop. A noise vector z ~ p(z) is mapped by the generator G to a fake sample ](../assets/figures/ml_concept_11.2_dabc8795.png)

*Figure 11.2 — original teaching graphic.*

```
# One GAN iteration (non-saturating generator loss)
for each minibatch:
 # ----- Discriminator step (repeat n_critic times for WGAN variants) -----
 x_real ~ data
 z ~ p(z); x_fake = G(z)
 L_D = −mean[ log D(x_real) + log(1 − D(x_fake)) ]
 θ_D ← θ_D − lr_D · ∇_{θ_D} L_D # push D(real)→1, D(fake)→0
 # ----- Generator step -----
 z ~ p(z)
 L_G = −mean[ log D(G(z)) ] # non-saturating: strong gradient when D wins
 θ_G ← θ_G − lr_G · ∇_{θ_G} L_G # push D(fake)→1
```

The non-saturating generator loss −log D(G(z)) replaces the original log(1 − D(G(z))) precisely because the latter’s gradient vanishes when D confidently rejects early fakes—an instability the loop must avoid.

### Challenges: oscillation, slow convergence, mode collapse, uninformative loss

GAN training is a nonconvex two-player game. Loss oscillation occurs when G and D chase each other without settling. Slow convergence appears when D becomes too strong and gradients for G vanish, or when learning rates are mismatched. Mode collapse happens when G produces only a few modes of the data distribution—plausible samples but low diversity (every synthetic MCA infarct looks the same). Uninformative loss curves: generator loss values do not reliably track sample quality, so visual inspection and external metrics are required. Stabilization tricks are worth naming concretely rather than as a list. Spectral normalization divides each discriminator weight matrix by its largest singular value, capping the layer’s amplification and hence the discriminator’s Lipschitz constant, which tames exploding gradients. The two-time-scale update rule (TTUR) gives D and G different learning rates—often a faster discriminator—so D stays a well-matched but not overwhelming teacher and the pair is more likely to reach equilibrium. Replay buffers feed the discriminator a mix of current and previously generated samples so G cannot exploit a discriminator that has forgotten old failure modes (a driver of oscillation). Architectural constraints (the DCGAN recipe) and the Wasserstein objectives below round out the toolkit.

### Evaluating GANs: Inception score and FID

Inception score (IS) runs generated images through a pretrained Inception classifier: high IS wants each image to have a peaked class distribution (clarity) and the marginal over images to be diverse. IS ignores real data statistics and can be gamed. Fréchet Inception Distance (FID) embeds real and generated sets in Inception features, fits Gaussians, and computes Fréchet distance between those Gaussians—lower is better. FID correlates better with human judgments but still depends on ImageNet-centric features that may poorly match medical image manifolds; domain-specific feature extractors are preferable for clinical synthesis evaluation.

### CGAN and DCGAN

Conditional GANs (CGAN) change the conditioning, not the loss form: side information y (class labels, text embeddings, masks) is concatenated into both G and D so generation is controllable—sample x ~ p(x | y). Deep convolutional GANs (DCGAN) change the architecture while keeping the original adversarial loss: an all-convolutional G and D with strided / fractionally-strided convolutions, batch norm, and ReLU/Leaky ReLU—the recipe that first made image GANs reliable enough for broad adoption. They remain a pedagogical baseline for synthetic imaging experiments.

### WGAN and WGAN-GP

Wasserstein GANs change the loss: they replace the log/Jensen–Shannon objective with an approximation of Earth-mover (Wasserstein-1) distance between real and generated distributions, which can provide smoother gradients than Jensen–Shannon when supports poorly overlap. The discriminator drops its sigmoid and becomes a real-valued critic under a Lipschitz constraint. Weight clipping enforced Lipschitz coarsely but harmed capacity; WGAN-GP changes only how that constraint is imposed, penalizing the squared deviation of the critic gradient norm from 1 on random interpolates between real and fake samples (gradient penalty). WGAN-GP often trains more stably and reduces mode collapse relative to vanilla GANs, though not universally.

### Pix2Pix, CycleGAN, and StyleGAN

Pix2Pix changes both conditioning and loss: it is a conditional GAN whose condition is an entire input image (paired image-to-image translation—edges→photo, segmentation maps→image) and whose objective adds an L1 reconstruction term to the paired target on top of the adversarial loss, with a patch-level (PatchGAN) discriminator. CycleGAN changes the architecture and loss to remove the need for aligned pairs: two generators G: X→Y, F: Y→X with cycle-consistency losses F(G(x)) ≈ x, G(F(y)) ≈ y plus an adversarial loss in each domain—enabling translation without aligned pairs (e.g., scanner A contrast to scanner B appearance, used cautiously in medical imaging because cycle consistency does not guarantee clinical fidelity).

StyleGAN changes the generator architecture and the way latents are injected while leaving the adversarial loss largely standard (non-saturating or WGAN-style): a mapping network transforms latent z into an intermediate style w, styles are injected via adaptive instance normalization (AdaIN) into a synthesis network, and explicit per-layer noise inputs supply stochastic detail. Progressive growing (in earlier StyleGAN) trains from low to high resolution; later versions refine path length regularization—encouraging a fixed-size step in w to produce a fixed-size change in the image, which smooths the latent geometry—and the architecture. Style mixing—applying different w at different layers—separates coarse attributes from fine detail. Successors improve inversion, disentanglement, and efficiency. Style-based generators illustrate how inductive bias in latent injection shapes controllability of synthetic images.

## 11.6 Contrastive Representation Learning and Siamese Networks

Contrastive learning trains encoders so that semantically similar pairs of examples lie nearby in embedding space while dissimilar pairs are far apart. Self-supervised variants treat two augmentations of the same image as a positive pair and other batch members as negatives (SimCLR-style), without class labels.

### Contrastive loss and triplet loss

A classic contrastive loss for pairs (x_i, x_j) with binary similarity s_{ij} pulls embeddings together when s=1 and pushes them apart up to a margin when s=0: L = s · d² + (1−s) · max(0, m − d)² for distance d between embeddings. Triplet loss uses anchors a, positives p, negatives n and enforces d(a,p) + margin < d(a,n), summing hinge violations. Hard-negative mining focuses on challenging n. InfoNCE and related losses frame contrastive learning as identifying the positive among many negatives via softmax over similarity scores—foundational to modern SSL and CLIP.

![Contrastive easy vs hard negatives in embedding space (synthetic; original).](../assets/figures/ml_fig_hard_negatives.png)

*Figure — Hard negatives dominate the gradient. **Left:** anchor (\*), positives (teal), easy far negatives (gray), and hard near-boundary negatives (gold). **Right:** relative InfoNCE-style pull rises sharply with cosine similarity of a negative—so near-misses matter most. False hard negatives (same patient, different view labeled negative) poison SSL; keep probe labels out of pretraining. Embedding geometry is not a disease causal map.*

![SimCLR-style two-view pipeline into InfoNCE (original).](../assets/figures/ml_fig_simclr_pipeline.png)

*Figure — Shared encoder f and projection g on two augmentations of x; InfoNCE with temperature τ pulls views together and pushes other batch items apart. Pipeline for representation learning—not a causal model of disease.*

![Diffusion forward process: data gradually noised to isotropic Gaussian (synthetic; original).](../assets/figures/ml_fig_diffusion_forward.png)

*Figure — Forward diffusion. As t increases, structure dissolves into noise; reverse models learn to denoise. Generative geometry—not a model of disease etiology.*

![Triplet loss with chapter worked numbers (original).](../assets/figures/ml_fig_triplet_ssl.png)

*Figure 11.3. Triplet loss in embedding space, drawn with the chapter's worked numbers. With anchor a=(0,0), positive p=(0.3,0.4) and margin m=0.2, the loss L = max(0, d(a,p) − d(a,n) + m) requires negatives to lie beyond the dashed boundary at radius d(a,p)+m = 0.7. The easy negative n=(0.8,0.6) at distance 1.0 is inactive (L=0), whereas the hard negative n′=(0.4,0.3) at distance 0.5 falls inside the margin and is active (L=0.2); the emerald arrow pulls the positive inward and the rose arrow pushes the hard negative outward.*

```
# SimCLR-style InfoNCE for a minibatch of N images (temperature τ)
views = []
for each image x_i: # two augmented views per image
 views += [ augment(x_i), augment(x_i) ]
z = L2_normalize( projection(encoder(views)) ) # 2N embeddings on the unit sphere
sim(u, v) = uᵀv / τ # cosine similarity, scaled by τ
for each view i (its positive j = the sibling view of the same image):
 L_i = −log[ exp(sim(z_i, z_j)) / Σ_{k ≠ i} exp(sim(z_i, z_k)) ]
L = mean over all 2N views of L_i # every other view in the batch is a negative
```

The denominator sums over all other views, so a large batch supplies many negatives “for free”—the reason contrastive SSL is batch-size hungry. Lowering τ sharpens the softmax and penalizes hard negatives more aggressively.

![InfoNCE temperature τ: positive-pair softmax mass and class probabilities (synthetic; original).](../assets/figures/ml_fig_infonce_temp.png)

*Figure — Temperature in InfoNCE. **Left:** for fixed synthetic similarities (one positive, seven negatives), the softmax weight on the positive pair falls as temperature τ rises—low τ sharpens the distribution and makes the loss sensitive to hard negatives. **Right:** probability mass at τ = 0.07 vs τ = 1.5. InfoNCE is L = −log p_pos with p ∝ exp(sim/τ); choose τ as a training hyperparameter and validate transfer, not just pretext accuracy. Pretext geometry is not a causal map of disease.*

### Siamese networks

Siamese architectures run twin networks with shared weights on two inputs, then compare embeddings with distance metrics or a small classifier head. Training uses pair or triplet sampling: balanced similar/dissimilar clinical pairs, or carefully constructed triplets. Testing embeds a probe and retrieves nearest neighbors or thresholds distance for verification (same patient / same lesion type). Data preparation dominates success: noisy similarity labels and trivial positives (identical images) collapse learning. In neurology, Siamese models support longitudinal CT matching, near-duplicate note detection, and few-shot rare disease retrieval when classes are too sparse for standard softmax classifiers.

## 11.7 Text-to-Image Models

### Zero-shot, autoregressive, and diffusion concepts

Zero-shot learning predicts classes or generates content for concepts unseen as labeled training targets, typically by aligning with language embeddings. Autoregressive text-to-image models serialize images into token sequences (often via discrete VQ codes) and predict the next token conditioned on text—like language models over visual tokens. Diffusion models learn to reverse a gradual noising process: train a network to predict noise (or clean data) given a noisy image at timestep t and optional text conditioning; sampling starts from pure noise and iteratively denoises. Inpainting and outpainting adapt diffusion by fixing known pixels and sampling the missing region—relevant to artifact repair, not clinical fabrication of nonexistent findings.

![11.4: The diffusion process. Top: the forward chain x0→…→xT gradually corrupts a structured image by adding Gaussian noise, x_](../assets/figures/ml_concept_11.4_ccc282b4.png)

*Figure 11.4 — original teaching graphic.*

### CLIP

CLIP (Contrastive Language–Image Pre-training) jointly trains an image encoder and a text encoder so that matched image–caption pairs have high cosine similarity relative to mismatched pairs in a batch (symmetric InfoNCE-style objective). After pretraining on web-scale pairs, zero-shot classification embeds class name prompts and picks the text embedding nearest the image—without fine-tuning on those classes. CLIP embeddings power retrieval, guidance for generative models, and multimodal clinical research, but web captions are not radiology reports: domain shift and unsafe associations require medical-specific evaluation.

### VQ-GAN and VQGAN-CLIP

Vector-quantized GANs learn a discrete codebook of visual tokens with an autoencoder trained using reconstruction, adversarial, and codebook losses. Images become sequences of code indices suitable for transformer autoregression. VQGAN-CLIP optimizes latent codes or generator parameters so that CLIP similarity between the rendered image and a text prompt rises—an optimization-based text-to-image method preceding today’s large diffusion systems.

### DALL-E, Imagen, Parti, and Stable Diffusion concepts

DALL-E v1 demonstrated autoregressive transformers over text and VQ image tokens. DALL-E v2 combines a prior over CLIP image embeddings with a diffusion decoder; v3 improves prompt following and text rendering via stronger language understanding and training data. Imagen emphasizes large frozen language models as text encoders feeding cascaded diffusion over pixels. Parti explores scaling autoregressive pure-transformer paths. Stable Diffusion popularized latent diffusion models (LDMs): diffuse in the latent space of a pretrained autoencoder rather than pixels, dramatically cutting compute, with a U-Net backbone conditioned on text via cross-attention. SDXL and later versions scale backbones, improve conditioning, and refine sampling. Conceptually, these systems unite contrastive language–image alignment, discrete or continuous latent compression, and powerful conditional generators—templates for domain-adapted medical research tools, not drop-in diagnostic devices.

## 11.8 Worked Example: Contrastive Pair Loss and a Tiny VAE ELBO

Contrastive pair example. Embeddings f(x1)=[1.0, 0.0], f(x2)=[0.8, 0.6], f(x3)=[−1.0, 0.0] already have unit L2 norm (‖f(x2)‖ = √(0.64+0.36) = 1, the usual pre-processing before computing similarities); we use plain Euclidean distance here for transparency. Let d12 = ‖f(x1)−f(x2)‖ = √((0.2)²+(−0.6)²)=√0.40≈0.632, d13 = ‖f(x1)−f(x3)‖ = 2.0. With margin m=1.0, similar pair (1,2) contributes d12²≈0.40; dissimilar pair (1,3) contributes max(0, m−d13)² = max(0,1−2)²=0. Total contrastive loss 0.40 + 0 = 0.40. If a bad embedding collapsed f(x3) to [0.9, 0.1], d13 would shrink and the hinge would activate, pushing dissimilar points apart—the arithmetic objective behind Siamese training.

Triplet loss example. Take anchor a=[0.0, 0.0], positive p=[0.3, 0.4], negative n=[0.8, 0.6], margin m=0.2, and loss L = max(0, d(a,p) − d(a,n) + m). Then d(a,p) = √(0.3²+0.4²) = √0.25 = 0.5 and d(a,n) = √(0.8²+0.6²) = √1.0 = 1.0, so L = max(0, 0.5 − 1.0 + 0.2) = max(0, −0.3) = 0: the negative already sits comfortably farther than the positive plus margin, the triplet is inactive, and it contributes no gradient. Now swap in a harder negative n′=[0.4, 0.3] with d(a,n′) = √(0.4²+0.3²) = √0.25 = 0.5. Now L = max(0, 0.5 − 0.5 + 0.2) = 0.2 > 0: the triplet is active, and gradients push a away from n′ while pulling it toward p. Hard-negative mining deliberately samples such active triplets so that most updates carry a signal instead of a flat zero.

Tiny VAE ELBO sketch. Suppose one latent dimension, prior p(z)=N(0,1), encoder for a datapoint outputs μ=1.0, σ=0.5 so q(z|x)=N(1, 0.25). KL(q‖p) for univariate Gaussians is ½[μ² + σ² − 1 − log σ²] = ½[1 + 0.25 − 1 − log 0.25] = ½[0.25 − (−1.3863)] ≈ 0.818. If a decoder samples z=μ+σ ε with ε=0 and reconstructs with squared error 0.50 on x, the ELBO estimate is −0.50 − 0.818 = −1.318 (higher ELBO is better; training minimizes the negative ELBO). Increasing β on the KL term shrinks μ toward 0 and σ toward 1, often harming reconstruction but improving latent regularity—the β-VAE trade-off in numbers.

![VAE ELBO decomposition at the chapter (μ,σ,recon) point and synthetic β-VAE recon–KL trade-off (scientific; original).](../assets/figures/ml_fig_vae_elbo.png)

*Figure — Univariate VAE teaching numbers. **Left:** reconstruction term 0.50, KL≈0.818, and −ELBO at β=1. **Right:** as β grows, the training objective L = recon + β·KL weights latent regularity more heavily; reconstruction typically worsens while KL shrinks toward the prior. Disentanglement under large β is hypothesized, not guaranteed. Latent structure is for generation and representation—not a causal model of disease.*

![β-VAE reconstruction–KL tradeoff and latent entanglement caricature (synthetic; original).](../assets/figures/ml_fig_beta_vae_tradeoff.png)

*Figure — Pressure on the latent. **Left:** as β rises, recon loss tends up while KL is driven down. **Right:** low-β latents look entangled; high-β more axis-aligned—partial and fragile. Large β can blur rare lesion cues; latent directions are not disease causes.*

## 11.9 Energy-Based Intuition, Latent Geometry, and Evaluation Discipline

Energy-based models assign low energy to plausible configurations. RBMs, many GANs (via critic scores), and score-based diffusion models share the spirit of shaping a landscape over data space. Diffusion training can be viewed as learning the score ∇_x log p_t(x) of noisy marginals—linking denoising autoencoders to modern generative SOTA. For clinicians, the operational question is not the elegance of the energy but whether samples or embeddings improve a pre-registered downstream endpoint.

![11.5: A structured two-dimensional latent manifold. A grid of latent codes (z1, z2) is rendered as decoded glyphs whose orient](../assets/figures/ml_concept_11.5_a21c5653.png)

*Figure 11.5 — original teaching graphic.*

Latent geometry matters when you interpolate or cluster codes. VAEs encourage continuous latents with Gaussian priors; VQ models yield discrete codes friendly to Transformers; GAN latents can be manipulable (StyleGAN w-space) without explicit densities. An example failure mode in medical synthesis is latent directions that change ‘scanner vendor look’ entangled with ‘lesion size,’ so that style edits silently alter pathology. Always measure clinical attributes of synthetic images with independent classifiers or readers.

Evaluation discipline for generative and SSL models: (1) reconstruction or FID/IS only as intermediate diagnostics; (2) downstream supervised metrics with fixed labeled splits; (3) human preference for perceptual tasks; (4) membership-inference and memorization checks when training on sensitive hospital data; (5) slice-level and patient-level leakage audits. Self-supervised checkpoints should be frozen when comparing linear probes so that differences reflect representation quality rather than unequal fine-tuning.

![SSL linear-probe hygiene: frozen encoder vs leaky fine-tune and test tuning (synthetic; original).](../assets/figures/ml_fig_ssl_probe_hygiene.png)

*Figure — Probe hygiene. **Left:** synthetic downstream AUROC under five protocols; red bars inflate scores by unfreezing the encoder or selecting on the test split, so the gain is not “better SSL.” **Right:** frozen-encoder linear probe vs full fine-tune are different experiments—compare prettexts only under a shared freeze, labeled split, and head budget; report fine-tune results in a separate table. Pretext loss and probe AUROC are not causal maps of clinical mechanism.*

Pretext task catalog for unlabeled neuroimaging: rotation prediction, jigsaw puzzles, relative patch location, masked image modeling (MAE-style), contrastive multi-window CT views, and longitudinal order verification (admission vs follow-up). Not every pretext transfers: predicting MRI series type may yield embeddings that ignore subtle infarcts. Pilot multiple pretexts with the same linear probe protocol—an example of experimental hygiene that prevents narrative-driven architecture choice.

![Masked autoencoding: mask ratio vs reconstruction and linear-probe transfer (synthetic; original).](../assets/figures/ml_fig_mae_mask_ratio.png)

*Figure — MAE teaching panel. **Left:** a patch grid with ~75% masked (dark)—the encoder sees visible patches; the decoder reconstructs masked ones. **Right:** synthetic curves where reconstruction MSE rises with mask ratio, while linear-probe AUROC peaks at intermediate masks and can fall if the pretext becomes too hard or too easy. Always validate transfer (probe/fine-tune), not pretext loss alone. Pretext geometry is not a clinical label and not a causal map.*


![Contrastive embedding: pull positives, push negatives (synthetic; original).](../assets/figures/ml_fig_contrastive_pull.png)

*Figure — Contrastive geometry. Anchor pulls positives and repels a sample of negatives. Embedding structure is task-shaped—**not automatic causal structure** in the world.*


![Masked prediction rate vs downstream probe score (synthetic; original).](../assets/figures/ml_fig_mask_rate_probe.png)

*Figure — Pretext hyperparameter. Common ~15% mask is a teaching default; probe scores measure transfer under a label—not clinical utility or causation by themselves.*


![Label efficiency: scratch vs linear probe vs fine-tune (synthetic; original).](../assets/figures/ml_fig_ssl_label_efficiency.png)

*Figure — Pretraining helps most when labels are scarce. Label efficiency is an empirical learning benefit—not causal identification of disease mechanisms.*


![Contrastive probe accuracy vs negative queue size (synthetic; original).](../assets/figures/ml_fig_contrastive_queue.png)

*Figure — Larger negative sets often improve contrastive representation quality in teaching curves. Embedding quality is not clinical causation.*


![Target-network lag cartoon for SSL stability (synthetic; original).](../assets/figures/ml_fig_target_network_lag.png)

*Figure — Stabilization tricks for representation learning. Pred != cause without design.*


![Stop-gradient block schematic (original).](../assets/figures/ml_fig_stopgrad.png)

*Figure — Stop-grad is an optimization device. Stop-gradient block schematic Pred != cause without design.*


![augpipe teaching panel (original).](../assets/figures/ml_fig_aug_pipeline.png)

*Figure — Teaching panel for augpipe. Pred != cause without design.*


![Cycle-34 densify scientific panel 13 (original).](../assets/figures/ml_fig_c34_12.png)

*Figure — Continuous densify panel 13. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 13 (original).](../assets/figures/ml_fig_c35_12.png)

*Figure — Continuous densify panel 13. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 13 (original).](../assets/figures/ml_fig_c36_12.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 13 (original).](../assets/figures/ml_fig_c37_12.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 13 (original).](../assets/figures/ml_fig_c38_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 13 (original).](../assets/figures/ml_fig_c39_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 13 (original).](../assets/figures/ml_fig_c40_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 13 (original).](../assets/figures/ml_fig_c41_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 13 (original).](../assets/figures/ml_fig_c42_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 13 (original).](../assets/figures/ml_fig_c43_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 13 (original).](../assets/figures/ml_fig_c44_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 13 (original).](../assets/figures/ml_fig_c45_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 13 (original).](../assets/figures/ml_fig_c46_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 13 (original).](../assets/figures/ml_fig_c47_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 13 (original).](../assets/figures/ml_fig_c48_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 13 (original).](../assets/figures/ml_fig_c49_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 13 (original).](../assets/figures/ml_fig_c50_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 13 (original).](../assets/figures/ml_fig_c51_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 13 (original).](../assets/figures/ml_fig_c52_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 13 (original).](../assets/figures/ml_fig_c53_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 13 (original).](../assets/figures/ml_fig_c54_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 13 (original).](../assets/figures/ml_fig_c55_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 13 (original).](../assets/figures/ml_fig_c56_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 13 (original).](../assets/figures/ml_fig_c57_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 13 (original).](../assets/figures/ml_fig_c58_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 13 (original).](../assets/figures/ml_fig_c59_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 13 (original).](../assets/figures/ml_fig_c60_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 13 (original).](../assets/figures/ml_fig_c61_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 13 (original).](../assets/figures/ml_fig_c62_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 13 (original).](../assets/figures/ml_fig_c63_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 13 (original).](../assets/figures/ml_fig_c64_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 13 (original).](../assets/figures/ml_fig_c65_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 13 (original).](../assets/figures/ml_fig_c66_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 13 (original).](../assets/figures/ml_fig_c67_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 13 (original).](../assets/figures/ml_fig_c68_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 13 (original).](../assets/figures/ml_fig_c69_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 13 (original).](../assets/figures/ml_fig_c70_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 13 (original).](../assets/figures/ml_fig_c71_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 13 (original).](../assets/figures/ml_fig_c72_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 13 (original).](../assets/figures/ml_fig_c73_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 13 (original).](../assets/figures/ml_fig_c74_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 13 (original).](../assets/figures/ml_fig_c75_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 13 (original).](../assets/figures/ml_fig_c76_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 13 (original).](../assets/figures/ml_fig_c77_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 13 (original).](../assets/figures/ml_fig_c78_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 13 (original).](../assets/figures/ml_fig_c79_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 13 (original).](../assets/figures/ml_fig_c80_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 13 (original).](../assets/figures/ml_fig_c81_12.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

## 11.10 Multimodal Alignment Beyond CLIP and Practical Hospital Constraints

CLIP-style contrastive alignment generalizes to ECG–text, EEG–report, and CT–report pairs. Batch construction must avoid false negatives: two images from the same study with different captions should not be forced apart if they share semantics. Hard negatives (similar body region, different finding) improve discrimination. Zero-shot prompts are brittle: ‘intracranial hemorrhage’ vs ‘ICH’ vs ‘hemorrhagic stroke’ can shuffle rankings; ensemble prompts or learned prompt templates reduce variance.

Hospital constraints shape feasible SSL. Data cannot always leave the premises; federated or on-prem pretraining may be required. GPU hours for diffusion training are nontrivial. Labels for evaluation still need chart review budgets. A pragmatic path is: pretrain a modest encoder with masked autoencoding on local CT, attach a lightweight classifier for ICH, validate temporally and at a partner site, and only then explore heavier generative augmentation. Spectacular demos of synthetic brains do not substitute for this ladder of evidence.

Safety and dual use: generative models can fabricate realistic discharge summaries or imaging-like figures that mislead if presented as real. Watermarking, audit logs, and policy prohibitions on entering synthetic findings into the legal medical record are governance, not optional extras. SSL embeddings can also enable re-identification if linked to external photos; treat embeddings as potentially sensitive.

## 11.11 Comparing Generative Families for Medical Use-Cases

Choosing among VAEs, GANs, and diffusion is a systems decision. VAEs train stably and provide latents for anomaly scores via reconstruction plus density terms, but samples may be blurry for high-resolution CT. GANs can produce sharp images yet need careful stabilization and diversity checks to avoid mode collapse that under-represents rare posterior-circulation patterns. Diffusion models currently lead perceptual quality for natural images and are rapidly adapted to medical volumes, with sampling cost as the main drawback; latent diffusion reduces that cost by working in autoencoder space.

Conditional generation (class labels, masks, text) is usually more useful clinically than unconditional sampling. Pix2Pix-style paired translation can map segmentation masks to plausible images for education, while CycleGAN unpaired translation tempts scanner harmonization—yet must be frozen and audited so it does not erase true biological differences between populations. An example protocol: train harmonization on healthy-appearing slices only, then verify that lesion volumes and ASPECTS-like regional scores remain invariant under the translation using held-out pathologically rich studies.

Self-supervised pretraining versus generative augmentation: if the bottleneck is representation quality for a discriminative task, contrastive or masked modeling often gives more reliable gains than adding synthetic images. If the bottleneck is rare positives and readers agree synthetics look realistic under blinded review, carefully filtered generative augmentation can help—always re-tune thresholds after changing the training mixture, and never evaluate on synthetic test sets as a substitute for real external patients.

Representation probing checklist: linear probe accuracy, k-NN probe accuracy, few-shot fine-tuning curves, clustering purity of known subtypes, and retrieval precision for clinically defined queries. Publish negative results: many SSL recipes fail to beat strong supervised baselines when labels are already abundant. Science advances when those failures are documented with the same rigor as successes.

## 11.12 Historical Arc and What to Master First

The intellectual arc of this chapter runs from classical unsupervised neural nets (SOM, Boltzmann/RBM, stacked autoencoders) through adversarial and variational generators to contrastive multimodal models and diffusion-based text-to-image systems. Each layer adds tools without fully retiring predecessors: reconstruction still diagnoses anomalies; energy-based thinking still clarifies why training can be unstable; contrastive geometry still underpins modern embeddings.

For a neurologist–epidemiologist building skills, master first: (1) train/validate a denoising or masked autoencoder on local imaging and probe it; (2) implement a Siamese or contrastive pair sampler without leakage; (3) read GAN failure modes so synthetic data proposals can be critiqued; (4) understand CLIP-style zero-shot evaluation pitfalls before trusting prompt-based classifiers. Generative spectacle should come last, after discriminative clinical endpoints are secured.

Example milestone project: pretrain a ResNet encoder with SimCLR-style augmentations compatible with NCCT, freeze it, train a linear ICH head on 500 labeled studies, compare to ImageNet initialization and to a fully supervised CNN, validate on a later calendar year and an external site, and report calibration plus reader-discordant cases. That single project exercises SSL, evaluation discipline, and transportability more than re-implementing every GAN variant from scratch.

## Clinical and Epidemiologic Notes: SSL on Medical Unlabeled Data

Hospitals accumulate years of unlabeled CT/MRI, continuous EEG, and clinical text. SSL pretraining on in-domain unlabeled corpora often beats ImageNet initialization for downstream hemorrhage detection or phenotype classification—especially when labeled sets are small. Choose pretext tasks that respect clinical invariances: random heavy color jitter may be fine for natural photos but harmful for CT window-dependent signs; horizontal flips may break laterality. Masked autoencoding, contrastive multi-crop, and temporal order prediction for longitudinal scans are common patterns.

![11.6: Label efficiency of self-supervised pretraining. Downstream accuracy (here ICH detection) is plotted against the number ](../assets/figures/ml_concept_11.6_66fdb2c9.png)

*Figure 11.6 — original teaching graphic.*

Evaluation discipline matters more than architecture fashion. Pretrain only on training-site unlabeled studies; never peek at test patients during pretraining or augmentation search. Report downstream metrics with and without SSL, with confidence intervals, and with external hospitals. Synthetic data from GANs/diffusion must not be treated as evidence of disease mechanisms; use synthesis for augmentation only after proving downstream benefit and checking for spurious shortcuts (generated images that encode label text). For text-to-image systems, prevent generation of identifiable patient imagery and follow institutional governance.

Unsupervised anomaly detection with autoencoder residual maps can highlight unusual regions on NCCT, but reconstruction-based scores confuse rare normal variants with pathology and depend on reconstruction of high-frequency bone detail. Combine with reader-in-the-loop protocols. Contrastive models pretrained on reports and images enable zero-shot retrieval (‘find studies similar to this MCA infarct description’) that accelerates cohort building for epidemiology—if privacy and access controls hold.

### Synthetic-data leakage and site memorization

Generative models can leak training data in two distinct ways, and both inflate apparent performance while endangering patients. Verbatim (or near-duplicate) memorization: a GAN or diffusion model with high capacity and too little data can reproduce near-copies of individual training scans, so a “synthetic” image is effectively a real patient’s image under a new file name—re-identifiable and a privacy breach. Membership inference: an adversary who can query the model or inspect its outputs tries to decide whether a particular patient was in the training set; success means the model has memorized individuals rather than only population structure. Audit both with a nearest-neighbor check—embed every synthetic sample and its closest real training neighbor in a fixed feature space, flag pairs whose distance falls below a threshold calibrated on held-out real-to-real distances, and inspect flagged pairs by eye before releasing any synthetic corpus.

Site memorization is the multi-site version of the same failure. Scanners, reconstruction kernels, and acquisition protocols leave signatures (noise texture, field-of-view, header-driven intensity scaling) that a model learns as readily as anatomy, with two consequences. First, a generator can entangle a site’s “look” with pathology, so a synthetic image that copies scanner A’s texture silently drifts the disease appearance—harmonization that edits lesions rather than only style. Second, if synthetic images carrying site-specific signatures cross from training into evaluation, or if the same patient’s studies appear at both stages under different accession numbers, a downstream classifier can win by recognizing the site rather than the disease—a shortcut that evaporates at a new hospital and manufactures optimistic internal metrics. Defenses: keep pretraining, synthesis, and evaluation patient- and study-disjoint; stratify every metric by site and scanner; confirm that lesion volumes and regional (ASPECTS-like) scores are invariant under any harmonization on pathologically rich held-out studies; and never let synthetic data derived from evaluation-site patients enter the training mixture.

Prefer in-domain SSL pretraining when labels are scarce and unlabeled volume is large.

Design augmentations and pretext tasks that preserve clinical semantics (laterality, windowing).

Keep pretraining patient-disjoint from evaluation; validate externally.

Treat generative models as tools for augmentation/education, not ground-truth labels.

Audit synthetic and retrieved cases for shortcut features and PHI leakage.

## Connections

This chapter’s methods are threads that reappear across the book. Supervised deep learning supplies the backbones that SSL pretrains and the heads that probe them; U-Net is the bridge to dense prediction and segmentation, where the same encoder–decoder-with-skips reappears under full labels. Transformers and attention return as the sequence models over VQ tokens in autoregressive image generation and as the cross-attention that injects text conditioning into latent diffusion; masked autoencoding is the vision analogue of masked-language pretraining. Transfer learning and domain adaptation frame the label-efficiency payoff—pretrain, freeze, probe, fine-tune—while CycleGAN-style harmonization is domain adaptation’s tempting but hazardous cousin. Uncertainty and calibration connect through stochastic encoders, Monte Carlo dropout, and likelihood- or reconstruction-based out-of-distribution scoring, all of which decide when a model should abstain on an unfamiliar protocol. Finally, causal and epidemiologic reasoning keeps generative enthusiasm honest: synthesis cannot manufacture new causal information, site is a confounder that shortcut-learning will exploit, and only pre-registered downstream endpoints with external validation certify that a representation or a synthetic corpus actually helps.


![c82 teaching panel 12 (original).](../assets/figures/ml_fig_c82_12.png)
*Figure — Contrastive SSL geometry: pull positives, push negatives on the unit sphere. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 12 (original).](../assets/figures/ml_fig_c83_12.png)
*Figure — Masked language modeling token prediction sketch. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 12 (original).](../assets/figures/ml_fig_c84_12.png)
*Figure — SimCLR-style dual augmentation contrastive graph. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 12 (original).](../assets/figures/ml_fig_c85_12.png)
*Figure — SSL pretrain can improve downstream sample efficiency. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 12 (original).](../assets/figures/ml_fig_c86_12.png)
*Figure — Multi-view SSL shared encoder between views. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 12 (original).](../assets/figures/ml_fig_c87_12.png)
*Figure — Masked autoencoding reconstructs held-out patches. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 12 (original).](../assets/figures/ml_fig_c88_12.png)
*Figure — BYOL online/target network bootstrap idea. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 12 (original).](../assets/figures/ml_fig_c89_12.png)
*Figure — Self-distillation teacher/student views. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 12 (original).](../assets/figures/ml_fig_c90_12.png)
*Figure — MAE high mask ratio reconstruction. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 12 (original).](../assets/figures/ml_fig_c91_12.png)
*Figure — MoCo queue of negatives. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 12 (original).](../assets/figures/ml_fig_c92_12.png)
*Figure — CLIP image-text contrastive space. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 12 (original).](../assets/figures/ml_fig_c93_12.png)
*Figure — JEPA predict representation not pixels. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 12 (original).](../assets/figures/ml_fig_c94_12.png)
*Figure — SwAV online clustering codes. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 12 (original).](../assets/figures/ml_fig_c95_12.png)
*Figure — SigLIP sigmoid pairwise loss. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 12 (original).](../assets/figures/ml_fig_c96_12.png)
*Figure — I-JEPA multi-block masks. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 12 (original).](../assets/figures/ml_fig_c97_12.png)
*Figure — Barlow Twins redundancy reduction. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 12 (original).](../assets/figures/ml_fig_c98_12.png)
*Figure — FLAVA multimodal alignment. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 12 (original).](../assets/figures/ml_fig_c99_12.png)
*Figure — data2vec self-distill targets. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 12 (original).](../assets/figures/ml_fig_c100_12.png)
*Figure — VICReg variance-invariance. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 12 (original).](../assets/figures/ml_fig_c101_12.png)
*Figure — ImageBind joint embedding. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 12 (original).](../assets/figures/ml_fig_c102_12.png)
*Figure — DINOv2 registers tokens. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 12 (original).](../assets/figures/ml_fig_c103_12.png)
*Figure — SimSiam stop-gradient twin. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 12 (original).](../assets/figures/ml_fig_c104_12.png)
*Figure — ALIGN noisy web pairs. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 12 (original).](../assets/figures/ml_fig_c105_12.png)
*Figure — I-JEPA world model blocks. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 12 (original).](../assets/figures/ml_fig_c106_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 12 (original).](../assets/figures/ml_fig_c107_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 12 (original).](../assets/figures/ml_fig_c108_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 12 (original).](../assets/figures/ml_fig_c109_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 12 (original).](../assets/figures/ml_fig_c110_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 12 (original).](../assets/figures/ml_fig_c111_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 12 (original).](../assets/figures/ml_fig_c112_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 12 (original).](../assets/figures/ml_fig_c113_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 12 (original).](../assets/figures/ml_fig_c114_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 12 (original).](../assets/figures/ml_fig_c115_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 12 (original).](../assets/figures/ml_fig_c116_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 12 (original).](../assets/figures/ml_fig_c117_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 12 (original).](../assets/figures/ml_fig_c118_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 12 (original).](../assets/figures/ml_fig_c119_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 12 (original).](../assets/figures/ml_fig_c120_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 12 (original).](../assets/figures/ml_fig_c121_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 12 (original).](../assets/figures/ml_fig_c122_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 12 (original).](../assets/figures/ml_fig_c123_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 12 (original).](../assets/figures/ml_fig_c124_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 12 (original).](../assets/figures/ml_fig_c125_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 12 (original).](../assets/figures/ml_fig_c126_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 12 (original).](../assets/figures/ml_fig_c127_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 12 (original).](../assets/figures/ml_fig_c128_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 12 (original).](../assets/figures/ml_fig_c129_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 12 (original).](../assets/figures/ml_fig_c130_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 12 (original).](../assets/figures/ml_fig_c131_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 12 (original).](../assets/figures/ml_fig_c132_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 12 (original).](../assets/figures/ml_fig_c133_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 12 (original).](../assets/figures/ml_fig_c134_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 12 (original).](../assets/figures/ml_fig_c135_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 12 (original).](../assets/figures/ml_fig_c136_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 12 (original).](../assets/figures/ml_fig_c137_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 12 (original).](../assets/figures/ml_fig_c138_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 12 (original).](../assets/figures/ml_fig_c139_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 12 (original).](../assets/figures/ml_fig_c140_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 12 (original).](../assets/figures/ml_fig_c141_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 12 (original).](../assets/figures/ml_fig_c142_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 12 (original).](../assets/figures/ml_fig_c143_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 12 (original).](../assets/figures/ml_fig_c144_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 12 (original).](../assets/figures/ml_fig_c145_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 12 (original).](../assets/figures/ml_fig_c146_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 12 (original).](../assets/figures/ml_fig_c147_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 12 (original).](../assets/figures/ml_fig_c148_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 12 (original).](../assets/figures/ml_fig_c149_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 12 (original).](../assets/figures/ml_fig_c150_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 12 (original).](../assets/figures/ml_fig_c151_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 12 (original).](../assets/figures/ml_fig_c152_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 12 (original).](../assets/figures/ml_fig_c153_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 12 (original).](../assets/figures/ml_fig_c154_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 12 (original).](../assets/figures/ml_fig_c155_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 12 (original).](../assets/figures/ml_fig_c156_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 12 (original).](../assets/figures/ml_fig_c157_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 12 (original).](../assets/figures/ml_fig_c158_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 12 (original).](../assets/figures/ml_fig_c159_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 12 (original).](../assets/figures/ml_fig_c160_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 12 (original).](../assets/figures/ml_fig_c161_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 12 (original).](../assets/figures/ml_fig_c162_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 12 (original).](../assets/figures/ml_fig_c163_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 12 (original).](../assets/figures/ml_fig_c164_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 12 (original).](../assets/figures/ml_fig_c165_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 12 (original).](../assets/figures/ml_fig_c166_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 12 (original).](../assets/figures/ml_fig_c167_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 12 (original).](../assets/figures/ml_fig_c168_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 12 (original).](../assets/figures/ml_fig_c169_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 12 (original).](../assets/figures/ml_fig_c170_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 12 (original).](../assets/figures/ml_fig_c171_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 12 (original).](../assets/figures/ml_fig_c172_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 12 (original).](../assets/figures/ml_fig_c173_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 12 (original).](../assets/figures/ml_fig_c174_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 12 (original).](../assets/figures/ml_fig_c175_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 12 (original).](../assets/figures/ml_fig_c176_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 12 (original).](../assets/figures/ml_fig_c177_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 12 (original).](../assets/figures/ml_fig_c178_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 12 (original).](../assets/figures/ml_fig_c179_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 12 (original).](../assets/figures/ml_fig_c180_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 12 (original).](../assets/figures/ml_fig_c181_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 12 (original).](../assets/figures/ml_fig_c182_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 12 (original).](../assets/figures/ml_fig_c183_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 12 (original).](../assets/figures/ml_fig_c184_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 12 (original).](../assets/figures/ml_fig_c185_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 12 (original).](../assets/figures/ml_fig_c186_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 12 (original).](../assets/figures/ml_fig_c187_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 12 (original).](../assets/figures/ml_fig_c188_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 12 (original).](../assets/figures/ml_fig_c189_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 12 (original).](../assets/figures/ml_fig_c190_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 12 (original).](../assets/figures/ml_fig_c191_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 12 (original).](../assets/figures/ml_fig_c192_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 12 (original).](../assets/figures/ml_fig_c193_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 12 (original).](../assets/figures/ml_fig_c194_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 12 (original).](../assets/figures/ml_fig_c195_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 12 (original).](../assets/figures/ml_fig_c196_12.png)
*Figure — Masked image modeling. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 12 (original).](../assets/figures/ml_fig_c197_12.png)
*Figure — Jigsaw pretext task. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 12 (original).](../assets/figures/ml_fig_c198_12.png)
*Figure — Colorization pretext. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 12 (original).](../assets/figures/ml_fig_c199_12.png)
*Figure — Rotation prediction. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 12 (original).](../assets/figures/ml_fig_c200_12.png)
*Figure — Temporal order SSL. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 12 (original).](../assets/figures/ml_fig_c201_12.png)
*Figure — Hard negative mining radius. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 12 (original).](../assets/figures/ml_fig_c202_12.png)
*Figure — InfoNCE positive logit margin. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 12 (original).](../assets/figures/ml_fig_c203_12.png)
*Figure — BYOL online target bootstrap. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 12 (original).](../assets/figures/ml_fig_c204_12.png)
*Figure — SwAV swapped code prediction. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 12 (original).](../assets/figures/ml_fig_c205_12.png)
*Figure — DINO teacher-student crops. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 12 (original).](../assets/figures/ml_fig_c206_12.png)
*Figure — MAE high-ratio patch mask. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 12 (original).](../assets/figures/ml_fig_c207_12.png)
*Figure — MoCo momentum queue keys. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 12 (original).](../assets/figures/ml_fig_c208_12.png)
*Figure — Barlow Twins cross-corr identity. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 12 (original).](../assets/figures/ml_fig_c209_12.png)
*Figure — VICReg invariance variance covariance. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 12 (original).](../assets/figures/ml_fig_c210_12.png)
*Figure — Prototype soft assignment mass. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 12 (original).](../assets/figures/ml_fig_c211_12.png)
*Figure — I-JEPA latent prediction path. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 12 (original).](../assets/figures/ml_fig_c212_12.png)
*Figure — SimSiam stop-gradient branches. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 12 (original).](../assets/figures/ml_fig_c213_12.png)
*Figure — CLIP image-text similarity. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 12 (original).](../assets/figures/ml_fig_c214_12.png)
*Figure — Masked language model tokens. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 12 (original).](../assets/figures/ml_fig_c215_12.png)
*Figure — RotNet rotation pretext classes. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 12 (original).](../assets/figures/ml_fig_c216_12.png)
*Figure — SimCLR dual augmented views. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 12 (original).](../assets/figures/ml_fig_c217_12.png)
*Figure — Relative patch location pretext. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 12 (original).](../assets/figures/ml_fig_c218_12.png)
*Figure — Dense contrastive positive band. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 12 (original).](../assets/figures/ml_fig_c219_12.png)
*Figure — MoCo-v3 ViT contrastive. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 12 (original).](../assets/figures/ml_fig_c220_12.png)
*Figure — MAE light decoder path. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 12 (original).](../assets/figures/ml_fig_c221_12.png)
*Figure — DINO teacher-student match. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 12 (original).](../assets/figures/ml_fig_c222_12.png)
*Figure — I-JEPA context-target mask grid. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 12 (original).](../assets/figures/ml_fig_c223_12.png)
*Figure — Barlow Twins cross-corr target. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 12 (original).](../assets/figures/ml_fig_c224_12.png)
*Figure — data2vec EMA latent targets. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 12 (original).](../assets/figures/ml_fig_c225_12.png)
*Figure — SwAV swapped code prediction. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 12 (original).](../assets/figures/ml_fig_c226_12.png)
*Figure — V-JEPA video tube prediction. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 12 (original).](../assets/figures/ml_fig_c227_12.png)
*Figure — SimSiam stop-grad branches. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 12 (original).](../assets/figures/ml_fig_c228_12.png)
*Figure — DINOv2 teacher KoLeo path. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 12 (original).](../assets/figures/ml_fig_c229_12.png)
*Figure — VICReg inv-var-cov triplet. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 12 (original).](../assets/figures/ml_fig_c230_12.png)
*Figure — I-JEPA narrow predictor path. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 12 (original).](../assets/figures/ml_fig_c231_12.png)
*Figure — Masked Siamese dual views. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 12 (original).](../assets/figures/ml_fig_c232_12.png)
*Figure — MAE asymmetric encoder-decoder. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 12 (original).](../assets/figures/ml_fig_c233_12.png)
*Figure — BYOL predictor stop-grad. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 12 (original).](../assets/figures/ml_fig_c234_12.png)
*Figure — MoCo queue contrast path. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 12 (original).](../assets/figures/ml_fig_c235_12.png)
*Figure — SimSiam twin branch path. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 12 (original).](../assets/figures/ml_fig_c236_12.png)
*Figure — SimCLR NT-Xent path. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 12 (original).](../assets/figures/ml_fig_c237_12.png)
*Figure — Barlow cross-corr path. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 12 (original).](../assets/figures/ml_fig_c238_12.png)
*Figure — SupCon supervised path. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 12 (original).](../assets/figures/ml_fig_c239_12.png)
*Figure — VICReg variance path. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 12 (original).](../assets/figures/ml_fig_c240_12.png)
*Figure — CLIP image-text path. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 12 (original).](../assets/figures/ml_fig_c241_12.png)
*Figure — MoCo queue contrast path. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 12 (original).](../assets/figures/ml_fig_c242_12.png)
*Figure — MoCo-v3 online target path. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 12 (original).](../assets/figures/ml_fig_c243_12.png)
*Figure — DINO student-teacher path. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 12 (original).](../assets/figures/ml_fig_c244_12.png)
*Figure — SwAV swapped predict path. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 12 (original).](../assets/figures/ml_fig_c245_12.png)
*Figure — I-JEPA latent predict path. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 12 (original).](../assets/figures/ml_fig_c246_12.png)
*Figure — Barlow twins redundancy path. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 12 (original).](../assets/figures/ml_fig_c247_12.png)
*Figure — DINOv2 register token path. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 12 (original).](../assets/figures/ml_fig_c248_12.png)
*Figure — VICReg variance-invar path. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 12 (original).](../assets/figures/ml_fig_c249_12.png)
*Figure — MAE mask reconstruct path. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 12 (original).](../assets/figures/ml_fig_c250_12.png)
*Figure — SimCLR NT-Xent path. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 12 (original).](../assets/figures/ml_fig_c251_12.png)
*Figure — data2vec teacher path. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 12 (original).](../assets/figures/ml_fig_c252_12.png)
*Figure — BYOL stop-grad path. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 12 (original).](../assets/figures/ml_fig_c253_12.png)
*Figure — I-JEPA multi-block path. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 12 (original).](../assets/figures/ml_fig_c254_12.png)
*Figure — MoCo momentum path. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 12 (original).](../assets/figures/ml_fig_c255_12.png)
*Figure — VICReg var-inv-cov path. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 12 (original).](../assets/figures/ml_fig_c256_12.png)
*Figure — CLIP image-text path. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 12 (original).](../assets/figures/ml_fig_c257_12.png)
*Figure — VICReg var-inv path c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 12 (original).](../assets/figures/ml_fig_c258_12.png)
*Figure — DINO teacher path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 12 (original).](../assets/figures/ml_fig_c259_12.png)
*Figure — MAE mask recon path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 12 (original).](../assets/figures/ml_fig_c260_12.png)
*Figure — JEPA latent path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 12 (original).](../assets/figures/ml_fig_c261_12.png)
*Figure — SwAV code swap path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 12 (original).](../assets/figures/ml_fig_c262_12.png)
*Figure — CLIP align path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 12 (original).](../assets/figures/ml_fig_c263_12.png)
*Figure — data2vec EMA path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 12 (original).](../assets/figures/ml_fig_c264_12.png)
*Figure — I-JEPA block path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 12 (original).](../assets/figures/ml_fig_c265_12.png)
*Figure — Masked unit path c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 12 (original).](../assets/figures/ml_fig_c266_12.png)
*Figure — Contrast temp path c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 12 (original).](../assets/figures/ml_fig_c267_12.png)
*Figure — Projection dim bars c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 12 (original).](../assets/figures/ml_fig_c268_12.png)
*Figure — SimCLR NT-Xent path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 12 (original).](../assets/figures/ml_fig_c269_12.png)
*Figure — MoCo queue path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 12 (original).](../assets/figures/ml_fig_c270_12.png)
*Figure — BYOL stop-grad path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 12 (original).](../assets/figures/ml_fig_c271_12.png)
*Figure — SimSiam twin path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 12 (original).](../assets/figures/ml_fig_c272_12.png)
*Figure — Barlow twins path c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 12 (original).](../assets/figures/ml_fig_c273_12.png)
*Figure — VICReg var-inv path c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 12 (original).](../assets/figures/ml_fig_c274_12.png)
*Figure — DINO teacher path c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 12 (original).](../assets/figures/ml_fig_c275_12.png)
*Figure — MAE mask recon path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 12 (original).](../assets/figures/ml_fig_c276_12.png)
*Figure — JEPA latent path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 12 (original).](../assets/figures/ml_fig_c277_12.png)
*Figure — SwAV code swap path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 12 (original).](../assets/figures/ml_fig_c278_12.png)
*Figure — CLIP align path c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 12 (original).](../assets/figures/ml_fig_c279_12.png)
*Figure — data2vec EMA path c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 12 (original).](../assets/figures/ml_fig_c280_12.png)
*Figure — I-JEPA block path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 12 (original).](../assets/figures/ml_fig_c281_12.png)
*Figure — Masked unit path c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 12 (original).](../assets/figures/ml_fig_c282_12.png)
*Figure — Contrast temp path c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 12 (original).](../assets/figures/ml_fig_c283_12.png)
*Figure — Projection dim bars c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 12 (original).](../assets/figures/ml_fig_c284_12.png)
*Figure — SimCLR NT-Xent path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 12 (original).](../assets/figures/ml_fig_c285_12.png)
*Figure — MoCo queue path c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 12 (original).](../assets/figures/ml_fig_c286_12.png)
*Figure — BYOL stop-grad path c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 12 (original).](../assets/figures/ml_fig_c287_12.png)
*Figure — SimSiam twin path c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 12 (original).](../assets/figures/ml_fig_c288_12.png)
*Figure — Barlow twins path c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 12 (original).](../assets/figures/ml_fig_c289_12.png)
*Figure — VICReg var-inv path c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 12 (original).](../assets/figures/ml_fig_c290_12.png)
*Figure — DINO teacher path c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 12 (original).](../assets/figures/ml_fig_c291_12.png)
*Figure — MAE mask recon path c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 12 (original).](../assets/figures/ml_fig_c292_12.png)
*Figure — JEPA latent path c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 12 (original).](../assets/figures/ml_fig_c293_12.png)
*Figure — SwAV code swap path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 12 (original).](../assets/figures/ml_fig_c294_12.png)
*Figure — CLIP align path c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 12 (original).](../assets/figures/ml_fig_c295_12.png)
*Figure — data2vec EMA path c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 12 (original).](../assets/figures/ml_fig_c296_12.png)
*Figure — I-JEPA block path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 12 (original).](../assets/figures/ml_fig_c297_12.png)
*Figure — Masked unit path c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 12 (original).](../assets/figures/ml_fig_c298_12.png)
*Figure — Contrast temp path c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 12 (original).](../assets/figures/ml_fig_c299_12.png)
*Figure — Projection dim bars c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 12 (original).](../assets/figures/ml_fig_c300_12.png)
*Figure — SimCLR NT-Xent path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 12 (original).](../assets/figures/ml_fig_c301_12.png)
*Figure — MoCo queue path c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 12 (original).](../assets/figures/ml_fig_c302_12.png)
*Figure — BYOL stop-grad path c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 12 (original).](../assets/figures/ml_fig_c303_12.png)
*Figure — SimSiam twin path c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 12 (original).](../assets/figures/ml_fig_c304_12.png)
*Figure — Barlow twins path c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 12 (original).](../assets/figures/ml_fig_c305_12.png)
*Figure — VICReg var-inv path c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 12 (original).](../assets/figures/ml_fig_c306_12.png)
*Figure — DINO teacher path c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 12 (original).](../assets/figures/ml_fig_c307_12.png)
*Figure — MAE mask recon path c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 12 (original).](../assets/figures/ml_fig_c308_12.png)
*Figure — JEPA latent path c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 12 (original).](../assets/figures/ml_fig_c309_12.png)
*Figure — SwAV code swap path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 12 (original).](../assets/figures/ml_fig_c310_12.png)
*Figure — CLIP align path c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 12 (original).](../assets/figures/ml_fig_c311_12.png)
*Figure — data2vec EMA path c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 12 (original).](../assets/figures/ml_fig_c312_12.png)
*Figure — I-JEPA block path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 12 (original).](../assets/figures/ml_fig_c313_12.png)
*Figure — Masked unit path c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 12 (original).](../assets/figures/ml_fig_c314_12.png)
*Figure — Contrast temp path c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 12 (original).](../assets/figures/ml_fig_c315_12.png)
*Figure — Projection dim bars c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 12 (original).](../assets/figures/ml_fig_c316_12.png)
*Figure — SimCLR NT-Xent path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 12 (original).](../assets/figures/ml_fig_c317_12.png)
*Figure — MoCo queue path c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 12 (original).](../assets/figures/ml_fig_c318_12.png)
*Figure — BYOL stop-grad path c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 12 (original).](../assets/figures/ml_fig_c319_12.png)
*Figure — SimSiam twin path c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 12 (original).](../assets/figures/ml_fig_c320_12.png)
*Figure — Barlow twins path c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 12 (original).](../assets/figures/ml_fig_c321_12.png)
*Figure — VICReg var-inv path c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 12 (original).](../assets/figures/ml_fig_c322_12.png)
*Figure — DINO teacher path c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 12 (original).](../assets/figures/ml_fig_c323_12.png)
*Figure — MAE mask recon path c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 12 (original).](../assets/figures/ml_fig_c324_12.png)
*Figure — JEPA latent path c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 12 (original).](../assets/figures/ml_fig_c325_12.png)
*Figure — SwAV code swap path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 12 (original).](../assets/figures/ml_fig_c326_12.png)
*Figure — CLIP align path c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 12 (original).](../assets/figures/ml_fig_c327_12.png)
*Figure — data2vec EMA path c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 12 (original).](../assets/figures/ml_fig_c328_12.png)
*Figure — I-JEPA block path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 12 (original).](../assets/figures/ml_fig_c329_12.png)
*Figure — Masked unit path c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 12 (original).](../assets/figures/ml_fig_c330_12.png)
*Figure — Contrast temp path c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 12 (original).](../assets/figures/ml_fig_c331_12.png)
*Figure — Projection dim bars c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 12 (original).](../assets/figures/ml_fig_c332_12.png)
*Figure — SimCLR NT-Xent path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 12 (original).](../assets/figures/ml_fig_c333_12.png)
*Figure — MoCo queue path c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 12 (original).](../assets/figures/ml_fig_c334_12.png)
*Figure — BYOL stop-grad path c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 12 (original).](../assets/figures/ml_fig_c335_12.png)
*Figure — SimSiam twin path c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 12 (original).](../assets/figures/ml_fig_c336_12.png)
*Figure — Barlow twins path c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 12 (original).](../assets/figures/ml_fig_c337_12.png)
*Figure — VICReg var-inv path c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 12 (original).](../assets/figures/ml_fig_c338_12.png)
*Figure — DINO teacher path c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 12 (original).](../assets/figures/ml_fig_c339_12.png)
*Figure — MAE mask recon path c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 12 (original).](../assets/figures/ml_fig_c340_12.png)
*Figure — JEPA latent path c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 12 (original).](../assets/figures/ml_fig_c341_12.png)
*Figure — SwAV code swap path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 12 (original).](../assets/figures/ml_fig_c342_12.png)
*Figure — CLIP align path c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 12 (original).](../assets/figures/ml_fig_c343_12.png)
*Figure — data2vec EMA path c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 12 (original).](../assets/figures/ml_fig_c344_12.png)
*Figure — I-JEPA block path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 12 (original).](../assets/figures/ml_fig_c345_12.png)
*Figure — Masked unit path c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 12 (original).](../assets/figures/ml_fig_c346_12.png)
*Figure — Contrast temp path c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 12 (original).](../assets/figures/ml_fig_c347_12.png)
*Figure — Projection dim bars c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 12 (original).](../assets/figures/ml_fig_c348_12.png)
*Figure — SimCLR NT-Xent path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 12 (original).](../assets/figures/ml_fig_c349_12.png)
*Figure — MoCo queue path c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 12 (original).](../assets/figures/ml_fig_c350_12.png)
*Figure — BYOL stop-grad path c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 12 (original).](../assets/figures/ml_fig_c351_12.png)
*Figure — SimSiam twin path c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 12 (original).](../assets/figures/ml_fig_c352_12.png)
*Figure — Barlow twins path c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 12 (original).](../assets/figures/ml_fig_c353_12.png)
*Figure — VICReg var-inv path c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 12 (original).](../assets/figures/ml_fig_c354_12.png)
*Figure — DINO teacher path c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 12 (original).](../assets/figures/ml_fig_c355_12.png)
*Figure — MAE mask recon path c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 12 (original).](../assets/figures/ml_fig_c356_12.png)
*Figure — JEPA latent path c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 12 (original).](../assets/figures/ml_fig_c357_12.png)
*Figure — SwAV code swap path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 12 (original).](../assets/figures/ml_fig_c358_12.png)
*Figure — CLIP align path c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 12 (original).](../assets/figures/ml_fig_c359_12.png)
*Figure — data2vec EMA path c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 12 (original).](../assets/figures/ml_fig_c360_12.png)
*Figure — I-JEPA block path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 12 (original).](../assets/figures/ml_fig_c361_12.png)
*Figure — Masked unit path c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 12 (original).](../assets/figures/ml_fig_c362_12.png)
*Figure — Contrast temp path c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 12 (original).](../assets/figures/ml_fig_c363_12.png)
*Figure — Projection dim bars c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 12 (original).](../assets/figures/ml_fig_c364_12.png)
*Figure — SimCLR NT-Xent path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 12 (original).](../assets/figures/ml_fig_c365_12.png)
*Figure — MoCo queue path c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 12 (original).](../assets/figures/ml_fig_c366_12.png)
*Figure — BYOL stop-grad path c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 12 (original).](../assets/figures/ml_fig_c367_12.png)
*Figure — SimSiam twin path c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 12 (original).](../assets/figures/ml_fig_c368_12.png)
*Figure — Barlow twins path c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 12 (original).](../assets/figures/ml_fig_c369_12.png)
*Figure — VICReg var-inv path c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 12 (original).](../assets/figures/ml_fig_c370_12.png)
*Figure — DINO teacher path c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 12 (original).](../assets/figures/ml_fig_c371_12.png)
*Figure — MAE mask recon path c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 12 (original).](../assets/figures/ml_fig_c372_12.png)
*Figure — JEPA latent path c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 12 (original).](../assets/figures/ml_fig_c373_12.png)
*Figure — SwAV code swap path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 12 (original).](../assets/figures/ml_fig_c374_12.png)
*Figure — CLIP align path c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 12 (original).](../assets/figures/ml_fig_c375_12.png)
*Figure — data2vec EMA path c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 12 (original).](../assets/figures/ml_fig_c376_12.png)
*Figure — I-JEPA block path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 12 (original).](../assets/figures/ml_fig_c377_12.png)
*Figure — Masked unit path c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 12 (original).](../assets/figures/ml_fig_c378_12.png)
*Figure — Contrast temp path c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 12 (original).](../assets/figures/ml_fig_c379_12.png)
*Figure — Projection dim bars c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 12 (original).](../assets/figures/ml_fig_c380_12.png)
*Figure — SimCLR NT-Xent path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 12 (original).](../assets/figures/ml_fig_c381_12.png)
*Figure — MoCo queue path c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 12 (original).](../assets/figures/ml_fig_c382_12.png)
*Figure — BYOL stop-grad path c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 12 (original).](../assets/figures/ml_fig_c383_12.png)
*Figure — SimSiam twin path c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 12 (original).](../assets/figures/ml_fig_c384_12.png)
*Figure — Barlow twins path c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 12 (original).](../assets/figures/ml_fig_c385_12.png)
*Figure — VICReg var-inv path c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 12 (original).](../assets/figures/ml_fig_c386_12.png)
*Figure — DINO teacher path c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 12 (original).](../assets/figures/ml_fig_c387_12.png)
*Figure — MAE mask recon path c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 12 (original).](../assets/figures/ml_fig_c388_12.png)
*Figure — JEPA latent path c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 12 (original).](../assets/figures/ml_fig_c389_12.png)
*Figure — SwAV code swap path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 12 (original).](../assets/figures/ml_fig_c390_12.png)
*Figure — CLIP align path c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 12 (original).](../assets/figures/ml_fig_c391_12.png)
*Figure — data2vec EMA path c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 12 (original).](../assets/figures/ml_fig_c392_12.png)
*Figure — I-JEPA block path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 12 (original).](../assets/figures/ml_fig_c393_12.png)
*Figure — Masked unit path c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 12 (original).](../assets/figures/ml_fig_c394_12.png)
*Figure — Contrast temp path c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 12 (original).](../assets/figures/ml_fig_c395_12.png)
*Figure — Projection dim bars c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 12 (original).](../assets/figures/ml_fig_c396_12.png)
*Figure — SimCLR NT-Xent path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 12 (original).](../assets/figures/ml_fig_c397_12.png)
*Figure — MoCo queue path c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 12 (original).](../assets/figures/ml_fig_c398_12.png)
*Figure — BYOL stop-grad path c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 12 (original).](../assets/figures/ml_fig_c399_12.png)
*Figure — SimSiam twin path c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 12 (original).](../assets/figures/ml_fig_c400_12.png)
*Figure — Barlow twins path c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 12 (original).](../assets/figures/ml_fig_c401_12.png)
*Figure — VICReg var-inv path c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 12 (original).](../assets/figures/ml_fig_c402_12.png)
*Figure — DINO teacher path c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 12 (original).](../assets/figures/ml_fig_c403_12.png)
*Figure — MAE mask recon path c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 12 (original).](../assets/figures/ml_fig_c404_12.png)
*Figure — JEPA latent path c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 12 (original).](../assets/figures/ml_fig_c405_12.png)
*Figure — SwAV code swap path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 12 (original).](../assets/figures/ml_fig_c406_12.png)
*Figure — CLIP align path c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 12 (original).](../assets/figures/ml_fig_c407_12.png)
*Figure — data2vec EMA path c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 12 (original).](../assets/figures/ml_fig_c408_12.png)
*Figure — I-JEPA block path c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 12 (original).](../assets/figures/ml_fig_c409_12.png)
*Figure — Masked unit path c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 12 (original).](../assets/figures/ml_fig_c410_12.png)
*Figure — Contrast temp path c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 12 (original).](../assets/figures/ml_fig_c411_12.png)
*Figure — Projection dim bars c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 12 (original).](../assets/figures/ml_fig_c412_12.png)
*Figure — SimCLR NT-Xent path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 12 (original).](../assets/figures/ml_fig_c413_12.png)
*Figure — MoCo queue path c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 12 (original).](../assets/figures/ml_fig_c414_12.png)
*Figure — BYOL stop-grad path c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 12 (original).](../assets/figures/ml_fig_c415_12.png)
*Figure — SimSiam twin path c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 12 (original).](../assets/figures/ml_fig_c416_12.png)
*Figure — Barlow twins path c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 12 (original).](../assets/figures/ml_fig_c417_12.png)
*Figure — VICReg var-inv path c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 12 (original).](../assets/figures/ml_fig_c418_12.png)
*Figure — DINO teacher path c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 12 (original).](../assets/figures/ml_fig_c419_12.png)
*Figure — MAE mask recon path c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 12 (original).](../assets/figures/ml_fig_c420_12.png)
*Figure — JEPA latent path c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 12 (original).](../assets/figures/ml_fig_c421_12.png)
*Figure — SwAV code swap path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 12 (original).](../assets/figures/ml_fig_c422_12.png)
*Figure — CLIP align path c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 12 (original).](../assets/figures/ml_fig_c423_12.png)
*Figure — data2vec EMA path c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 12 (original).](../assets/figures/ml_fig_c424_12.png)
*Figure — I-JEPA block path c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 12 (original).](../assets/figures/ml_fig_c425_12.png)
*Figure — Masked unit path c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 12 (original).](../assets/figures/ml_fig_c426_12.png)
*Figure — Contrast temp path c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 12 (original).](../assets/figures/ml_fig_c427_12.png)
*Figure — Projection dim bars c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 12 (original).](../assets/figures/ml_fig_c428_12.png)
*Figure — SimCLR NT-Xent path c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 12 (original).](../assets/figures/ml_fig_c429_12.png)
*Figure — MoCo queue path c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 12 (original).](../assets/figures/ml_fig_c430_12.png)
*Figure — BYOL stop-grad path c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 12 (original).](../assets/figures/ml_fig_c431_12.png)
*Figure — SimSiam twin path c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 12 (original).](../assets/figures/ml_fig_c432_12.png)
*Figure — Barlow twins path c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 12 (original).](../assets/figures/ml_fig_c433_12.png)
*Figure — VICReg var-inv path c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 12 (original).](../assets/figures/ml_fig_c434_12.png)
*Figure — DINO teacher path c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 12 (original).](../assets/figures/ml_fig_c435_12.png)
*Figure — MAE mask recon path c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 12 (original).](../assets/figures/ml_fig_c436_12.png)
*Figure — JEPA latent path c436. Synthetic teaching geometry—not a causal claim.*


![c437 teaching panel 12 (original).](../assets/figures/ml_fig_c437_12.png)
*Figure — SwAV code swap path c437. Synthetic teaching geometry—not a causal claim.*


![c438 teaching panel 12 (original).](../assets/figures/ml_fig_c438_12.png)
*Figure — CLIP align path c438. Synthetic teaching geometry—not a causal claim.*


![c439 teaching panel 12 (original).](../assets/figures/ml_fig_c439_12.png)
*Figure — data2vec EMA path c439. Synthetic teaching geometry—not a causal claim.*


![c440 teaching panel 12 (original).](../assets/figures/ml_fig_c440_12.png)
*Figure — I-JEPA block path c440. Synthetic teaching geometry—not a causal claim.*


![c441 teaching panel 12 (original).](../assets/figures/ml_fig_c441_12.png)
*Figure — Masked unit path c441. Synthetic teaching geometry—not a causal claim.*


![c442 teaching panel 12 (original).](../assets/figures/ml_fig_c442_12.png)
*Figure — Contrast temp path c442. Synthetic teaching geometry—not a causal claim.*


![c443 teaching panel 12 (original).](../assets/figures/ml_fig_c443_12.png)
*Figure — Projection dim bars c443. Synthetic teaching geometry—not a causal claim.*


![c444 teaching panel 12 (original).](../assets/figures/ml_fig_c444_12.png)
*Figure — SimCLR NT-Xent path c444. Synthetic teaching geometry—not a causal claim.*


![c445 teaching panel 12 (original).](../assets/figures/ml_fig_c445_12.png)
*Figure — MoCo queue path c445. Synthetic teaching geometry—not a causal claim.*


![c446 teaching panel 12 (original).](../assets/figures/ml_fig_c446_12.png)
*Figure — BYOL stop-grad path c446. Synthetic teaching geometry—not a causal claim.*


![c447 teaching panel 12 (original).](../assets/figures/ml_fig_c447_12.png)
*Figure — SimSiam twin path c447. Synthetic teaching geometry—not a causal claim.*


![c448 teaching panel 12 (original).](../assets/figures/ml_fig_c448_12.png)
*Figure — Barlow twins path c448. Synthetic teaching geometry—not a causal claim.*


![c449 teaching panel 12 (original).](../assets/figures/ml_fig_c449_12.png)
*Figure — VICReg var-inv path c449. Synthetic teaching geometry—not a causal claim.*


![c450 teaching panel 12 (original).](../assets/figures/ml_fig_c450_12.png)
*Figure — DINO teacher path c450. Synthetic teaching geometry—not a causal claim.*


![c451 teaching panel 12 (original).](../assets/figures/ml_fig_c451_12.png)
*Figure — MAE mask recon path c451. Synthetic teaching geometry—not a causal claim.*


![c452 teaching panel 12 (original).](../assets/figures/ml_fig_c452_12.png)
*Figure — JEPA latent path c452. Synthetic teaching geometry—not a causal claim.*


![c453 teaching panel 12 (original).](../assets/figures/ml_fig_c453_12.png)
*Figure — SwAV code swap path c453. Synthetic teaching geometry—not a causal claim.*


![c454 teaching panel 12 (original).](../assets/figures/ml_fig_c454_12.png)
*Figure — CLIP align path c454. Synthetic teaching geometry—not a causal claim.*


![c455 teaching panel 12 (original).](../assets/figures/ml_fig_c455_12.png)
*Figure — data2vec EMA path c455. Synthetic teaching geometry—not a causal claim.*


![c456 teaching panel 12 (original).](../assets/figures/ml_fig_c456_12.png)
*Figure — I-JEPA block path c456. Synthetic teaching geometry—not a causal claim.*


![c457 teaching panel 12 (original).](../assets/figures/ml_fig_c457_12.png)
*Figure — Masked unit path c457. Synthetic teaching geometry—not a causal claim.*


![c458 teaching panel 12 (original).](../assets/figures/ml_fig_c458_12.png)
*Figure — Contrast temp path c458. Synthetic teaching geometry—not a causal claim.*


![c459 teaching panel 12 (original).](../assets/figures/ml_fig_c459_12.png)
*Figure — Projection dim bars c459. Synthetic teaching geometry—not a causal claim.*


![c460 teaching panel 12 (original).](../assets/figures/ml_fig_c460_12.png)
*Figure — SimCLR NT-Xent path c460. Synthetic teaching geometry—not a causal claim.*


![c461 teaching panel 12 (original).](../assets/figures/ml_fig_c461_12.png)
*Figure — MoCo queue path c461. Synthetic teaching geometry—not a causal claim.*


![c462 teaching panel 12 (original).](../assets/figures/ml_fig_c462_12.png)
*Figure — BYOL stop-grad path c462. Synthetic teaching geometry—not a causal claim.*


![c463 teaching panel 12 (original).](../assets/figures/ml_fig_c463_12.png)
*Figure — SimSiam twin path c463. Synthetic teaching geometry—not a causal claim.*


![c464 teaching panel 12 (original).](../assets/figures/ml_fig_c464_12.png)
*Figure — Barlow twins path c464. Synthetic teaching geometry—not a causal claim.*


![c465 teaching panel 12 (original).](../assets/figures/ml_fig_c465_12.png)
*Figure — VICReg var-inv path c465. Synthetic teaching geometry—not a causal claim.*


![c466 teaching panel 12 (original).](../assets/figures/ml_fig_c466_12.png)
*Figure — DINO teacher path c466. Synthetic teaching geometry—not a causal claim.*


![c467 teaching panel 12 (original).](../assets/figures/ml_fig_c467_12.png)
*Figure — MAE mask recon path c467. Synthetic teaching geometry—not a causal claim.*


![c468 teaching panel 12 (original).](../assets/figures/ml_fig_c468_12.png)
*Figure — JEPA latent path c468. Synthetic teaching geometry—not a causal claim.*


![c469 teaching panel 12 (original).](../assets/figures/ml_fig_c469_12.png)
*Figure — SwAV code swap path c469. Synthetic teaching geometry—not a causal claim.*


![c470 teaching panel 12 (original).](../assets/figures/ml_fig_c470_12.png)
*Figure — CLIP align path c470. Synthetic teaching geometry—not a causal claim.*


![c471 teaching panel 12 (original).](../assets/figures/ml_fig_c471_12.png)
*Figure — data2vec EMA path c471. Synthetic teaching geometry—not a causal claim.*


![c472 teaching panel 12 (original).](../assets/figures/ml_fig_c472_12.png)
*Figure — I-JEPA block path c472. Synthetic teaching geometry—not a causal claim.*


![c473 teaching panel 12 (original).](../assets/figures/ml_fig_c473_12.png)
*Figure — Masked unit path c473. Synthetic teaching geometry—not a causal claim.*


![c474 teaching panel 12 (original).](../assets/figures/ml_fig_c474_12.png)
*Figure — Contrast temp path c474. Synthetic teaching geometry—not a causal claim.*


![c475 teaching panel 12 (original).](../assets/figures/ml_fig_c475_12.png)
*Figure — Projection dim bars c475. Synthetic teaching geometry—not a causal claim.*


![c476 teaching panel 12 (original).](../assets/figures/ml_fig_c476_12.png)
*Figure — SimCLR NT-Xent path c476. Synthetic teaching geometry—not a causal claim.*


![c477 teaching panel 12 (original).](../assets/figures/ml_fig_c477_12.png)
*Figure — MoCo queue path c477. Synthetic teaching geometry—not a causal claim.*


![c478 teaching panel 12 (original).](../assets/figures/ml_fig_c478_12.png)
*Figure — BYOL stop-grad path c478. Synthetic teaching geometry—not a causal claim.*


![c479 teaching panel 12 (original).](../assets/figures/ml_fig_c479_12.png)
*Figure — SimSiam twin path c479. Synthetic teaching geometry—not a causal claim.*


![c480 teaching panel 12 (original).](../assets/figures/ml_fig_c480_12.png)
*Figure — Barlow twins path c480. Synthetic teaching geometry—not a causal claim.*


![c481 teaching panel 12 (original).](../assets/figures/ml_fig_c481_12.png)
*Figure — VICReg var-inv path c481. Synthetic teaching geometry—not a causal claim.*


![c482 teaching panel 12 (original).](../assets/figures/ml_fig_c482_12.png)
*Figure — DINO teacher path c482. Synthetic teaching geometry—not a causal claim.*


![c483 teaching panel 12 (original).](../assets/figures/ml_fig_c483_12.png)
*Figure — MAE mask recon path c483. Synthetic teaching geometry—not a causal claim.*


![c484 teaching panel 12 (original).](../assets/figures/ml_fig_c484_12.png)
*Figure — JEPA latent path c484. Synthetic teaching geometry—not a causal claim.*


![c485 teaching panel 12 (original).](../assets/figures/ml_fig_c485_12.png)
*Figure — SwAV code swap path c485. Synthetic teaching geometry—not a causal claim.*


![c486 teaching panel 12 (original).](../assets/figures/ml_fig_c486_12.png)
*Figure — CLIP align path c486. Synthetic teaching geometry—not a causal claim.*


![c487 teaching panel 12 (original).](../assets/figures/ml_fig_c487_12.png)
*Figure — data2vec EMA path c487. Synthetic teaching geometry—not a causal claim.*


![c488 teaching panel 12 (original).](../assets/figures/ml_fig_c488_12.png)
*Figure — I-JEPA block path c488. Synthetic teaching geometry—not a causal claim.*


![c489 teaching panel 12 (original).](../assets/figures/ml_fig_c489_12.png)
*Figure — Masked unit path c489. Synthetic teaching geometry—not a causal claim.*


![c490 teaching panel 12 (original).](../assets/figures/ml_fig_c490_12.png)
*Figure — Contrast temp path c490. Synthetic teaching geometry—not a causal claim.*


![c491 teaching panel 12 (original).](../assets/figures/ml_fig_c491_12.png)
*Figure — Projection dim bars c491. Synthetic teaching geometry—not a causal claim.*


![c492 teaching panel 12 (original).](../assets/figures/ml_fig_c492_12.png)
*Figure — SimCLR NT-Xent path c492. Synthetic teaching geometry—not a causal claim.*


![c493 teaching panel 12 (original).](../assets/figures/ml_fig_c493_12.png)
*Figure — MoCo queue path c493. Synthetic teaching geometry—not a causal claim.*


![c494 teaching panel 12 (original).](../assets/figures/ml_fig_c494_12.png)
*Figure — BYOL stop-grad path c494. Synthetic teaching geometry—not a causal claim.*


![c495 teaching panel 12 (original).](../assets/figures/ml_fig_c495_12.png)
*Figure — SimSiam twin path c495. Synthetic teaching geometry—not a causal claim.*


![c496 teaching panel 12 (original).](../assets/figures/ml_fig_c496_12.png)
*Figure — Barlow twins path c496. Synthetic teaching geometry—not a causal claim.*


![c497 teaching panel 12 (original).](../assets/figures/ml_fig_c497_12.png)
*Figure — VICReg var-inv path c497. Synthetic teaching geometry—not a causal claim.*


![c498 teaching panel 12 (original).](../assets/figures/ml_fig_c498_12.png)
*Figure — DINO teacher path c498. Synthetic teaching geometry—not a causal claim.*


![c499 teaching panel 12 (original).](../assets/figures/ml_fig_c499_12.png)
*Figure — MAE mask recon path c499. Synthetic teaching geometry—not a causal claim.*


![c500 teaching panel 12 (original).](../assets/figures/ml_fig_c500_12.png)
*Figure — JEPA latent path c500. Synthetic teaching geometry—not a causal claim.*


![c501 teaching panel 12 (original).](../assets/figures/ml_fig_c501_12.png)
*Figure — SwAV code swap path c501. Synthetic teaching geometry—not a causal claim.*


![c502 teaching panel 12 (original).](../assets/figures/ml_fig_c502_12.png)
*Figure — CLIP align path c502. Synthetic teaching geometry—not a causal claim.*


![c503 teaching panel 12 (original).](../assets/figures/ml_fig_c503_12.png)
*Figure — data2vec EMA path c503. Synthetic teaching geometry—not a causal claim.*


![c504 teaching panel 12 (original).](../assets/figures/ml_fig_c504_12.png)
*Figure — I-JEPA block path c504. Synthetic teaching geometry—not a causal claim.*


![c505 teaching panel 12 (original).](../assets/figures/ml_fig_c505_12.png)
*Figure — Masked unit path c505. Synthetic teaching geometry—not a causal claim.*


![c506 teaching panel 12 (original).](../assets/figures/ml_fig_c506_12.png)
*Figure — Contrast temp path c506. Synthetic teaching geometry—not a causal claim.*


![c507 teaching panel 12 (original).](../assets/figures/ml_fig_c507_12.png)
*Figure — Projection dim bars c507. Synthetic teaching geometry—not a causal claim.*


![c508 teaching panel 12 (original).](../assets/figures/ml_fig_c508_12.png)
*Figure — SimCLR NT-Xent path c508. Synthetic teaching geometry—not a causal claim.*


![c509 teaching panel 12 (original).](../assets/figures/ml_fig_c509_12.png)
*Figure — MoCo queue path c509. Synthetic teaching geometry—not a causal claim.*


![c510 teaching panel 12 (original).](../assets/figures/ml_fig_c510_12.png)
*Figure — BYOL stop-grad path c510. Synthetic teaching geometry—not a causal claim.*


![c511 teaching panel 12 (original).](../assets/figures/ml_fig_c511_12.png)
*Figure — SimSiam twin path c511. Synthetic teaching geometry—not a causal claim.*


![c512 teaching panel 12 (original).](../assets/figures/ml_fig_c512_12.png)
*Figure — Barlow twins path c512. Synthetic teaching geometry—not a causal claim.*


![c513 teaching panel 12 (original).](../assets/figures/ml_fig_c513_12.png)
*Figure — VICReg var-inv path c513. Synthetic teaching geometry—not a causal claim.*


![c514 teaching panel 12 (original).](../assets/figures/ml_fig_c514_12.png)
*Figure — DINO teacher path c514. Synthetic teaching geometry—not a causal claim.*


![c515 teaching panel 12 (original).](../assets/figures/ml_fig_c515_12.png)
*Figure — MAE mask recon path c515. Synthetic teaching geometry—not a causal claim.*


![c516 teaching panel 12 (original).](../assets/figures/ml_fig_c516_12.png)
*Figure — JEPA latent path c516. Synthetic teaching geometry—not a causal claim.*


![c517 teaching panel 12 (original).](../assets/figures/ml_fig_c517_12.png)
*Figure — SwAV code swap path c517. Synthetic teaching geometry—not a causal claim.*


![c518 teaching panel 12 (original).](../assets/figures/ml_fig_c518_12.png)
*Figure — CLIP align path c518. Synthetic teaching geometry—not a causal claim.*


![c519 teaching panel 12 (original).](../assets/figures/ml_fig_c519_12.png)
*Figure — data2vec EMA path c519. Synthetic teaching geometry—not a causal claim.*


![c520 teaching panel 12 (original).](../assets/figures/ml_fig_c520_12.png)
*Figure — I-JEPA block path c520. Synthetic teaching geometry—not a causal claim.*


![c521 teaching panel 12 (original).](../assets/figures/ml_fig_c521_12.png)
*Figure — Masked unit path c521. Synthetic teaching geometry—not a causal claim.*


![c522 teaching panel 12 (original).](../assets/figures/ml_fig_c522_12.png)
*Figure — Contrast temp path c522. Synthetic teaching geometry—not a causal claim.*


![c523 teaching panel 12 (original).](../assets/figures/ml_fig_c523_12.png)
*Figure — Projection dim bars c523. Synthetic teaching geometry—not a causal claim.*


![c524 teaching panel 12 (original).](../assets/figures/ml_fig_c524_12.png)
*Figure — SimCLR NT-Xent path c524. Synthetic teaching geometry—not a causal claim.*


![c525 teaching panel 12 (original).](../assets/figures/ml_fig_c525_12.png)
*Figure — MoCo queue path c525. Synthetic teaching geometry—not a causal claim.*


![c526 teaching panel 12 (original).](../assets/figures/ml_fig_c526_12.png)
*Figure — BYOL stop-grad path c526. Synthetic teaching geometry—not a causal claim.*


![c527 teaching panel 12 (original).](../assets/figures/ml_fig_c527_12.png)
*Figure — SimSiam twin path c527. Synthetic teaching geometry—not a causal claim.*


![c528 teaching panel 12 (original).](../assets/figures/ml_fig_c528_12.png)
*Figure — Barlow twins path c528. Synthetic teaching geometry—not a causal claim.*


![c529 teaching panel 12 (original).](../assets/figures/ml_fig_c529_12.png)
*Figure — VICReg var-inv path c529. Synthetic teaching geometry—not a causal claim.*


![c530 teaching panel 12 (original).](../assets/figures/ml_fig_c530_12.png)
*Figure — DINO teacher path c530. Synthetic teaching geometry—not a causal claim.*


![c531 teaching panel 12 (original).](../assets/figures/ml_fig_c531_12.png)
*Figure — MAE mask recon path c531. Synthetic teaching geometry—not a causal claim.*


![c532 teaching panel 12 (original).](../assets/figures/ml_fig_c532_12.png)
*Figure — JEPA latent path c532. Synthetic teaching geometry—not a causal claim.*


![c533 teaching panel 12 (original).](../assets/figures/ml_fig_c533_12.png)
*Figure — SwAV code swap path c533. Synthetic teaching geometry—not a causal claim.*


![c534 teaching panel 12 (original).](../assets/figures/ml_fig_c534_12.png)
*Figure — CLIP align path c534. Synthetic teaching geometry—not a causal claim.*


![c535 teaching panel 12 (original).](../assets/figures/ml_fig_c535_12.png)
*Figure — data2vec EMA path c535. Synthetic teaching geometry—not a causal claim.*


![c536 teaching panel 12 (original).](../assets/figures/ml_fig_c536_12.png)
*Figure — I-JEPA block path c536. Synthetic teaching geometry—not a causal claim.*


![c537 teaching panel 12 (original).](../assets/figures/ml_fig_c537_12.png)
*Figure — Masked unit path c537. Synthetic teaching geometry—not a causal claim.*


![c538 teaching panel 12 (original).](../assets/figures/ml_fig_c538_12.png)
*Figure — Contrast temp path c538. Synthetic teaching geometry—not a causal claim.*


![c539 teaching panel 12 (original).](../assets/figures/ml_fig_c539_12.png)
*Figure — Projection dim bars c539. Synthetic teaching geometry—not a causal claim.*


![c540 teaching panel 12 (original).](../assets/figures/ml_fig_c540_12.png)
*Figure — SimCLR NT-Xent path c540. Synthetic teaching geometry—not a causal claim.*


![c541 teaching panel 12 (original).](../assets/figures/ml_fig_c541_12.png)
*Figure — MoCo queue path c541. Synthetic teaching geometry—not a causal claim.*


![c542 teaching panel 12 (original).](../assets/figures/ml_fig_c542_12.png)
*Figure — BYOL stop-grad path c542. Synthetic teaching geometry—not a causal claim.*


![c543 teaching panel 12 (original).](../assets/figures/ml_fig_c543_12.png)
*Figure — SimSiam twin path c543. Synthetic teaching geometry—not a causal claim.*


![c544 teaching panel 12 (original).](../assets/figures/ml_fig_c544_12.png)
*Figure — Barlow twins path c544. Synthetic teaching geometry—not a causal claim.*


![c545 teaching panel 12 (original).](../assets/figures/ml_fig_c545_12.png)
*Figure — VICReg var-inv path c545. Synthetic teaching geometry—not a causal claim.*


![c546 teaching panel 12 (original).](../assets/figures/ml_fig_c546_12.png)
*Figure — DINO teacher path c546. Synthetic teaching geometry—not a causal claim.*


![c547 teaching panel 12 (original).](../assets/figures/ml_fig_c547_12.png)
*Figure — MAE mask recon path c547. Synthetic teaching geometry—not a causal claim.*


![c548 teaching panel 12 (original).](../assets/figures/ml_fig_c548_12.png)
*Figure — JEPA latent path c548. Synthetic teaching geometry—not a causal claim.*


![c549 teaching panel 12 (original).](../assets/figures/ml_fig_c549_12.png)
*Figure — SwAV code swap path c549. Synthetic teaching geometry—not a causal claim.*


![c550 teaching panel 12 (original).](../assets/figures/ml_fig_c550_12.png)
*Figure — CLIP align path c550. Synthetic teaching geometry—not a causal claim.*


![c551 teaching panel 12 (original).](../assets/figures/ml_fig_c551_12.png)
*Figure — data2vec EMA path c551. Synthetic teaching geometry—not a causal claim.*


![c552 teaching panel 12 (original).](../assets/figures/ml_fig_c552_12.png)
*Figure — I-JEPA block path c552. Synthetic teaching geometry—not a causal claim.*


![c553 teaching panel 12 (original).](../assets/figures/ml_fig_c553_12.png)
*Figure — Masked unit path c553. Synthetic teaching geometry—not a causal claim.*


![c554 teaching panel 12 (original).](../assets/figures/ml_fig_c554_12.png)
*Figure — Contrast temp path c554. Synthetic teaching geometry—not a causal claim.*


![c555 teaching panel 12 (original).](../assets/figures/ml_fig_c555_12.png)
*Figure — Projection dim bars c555. Synthetic teaching geometry—not a causal claim.*


![c556 teaching panel 12 (original).](../assets/figures/ml_fig_c556_12.png)
*Figure — SimCLR NT-Xent path c556. Synthetic teaching geometry—not a causal claim.*


![c557 teaching panel 12 (original).](../assets/figures/ml_fig_c557_12.png)
*Figure — MoCo queue path c557. Synthetic teaching geometry—not a causal claim.*


![c558 teaching panel 12 (original).](../assets/figures/ml_fig_c558_12.png)
*Figure — BYOL stop-grad path c558. Synthetic teaching geometry—not a causal claim.*


![c559 teaching panel 12 (original).](../assets/figures/ml_fig_c559_12.png)
*Figure — SimSiam twin path c559. Synthetic teaching geometry—not a causal claim.*


![c560 teaching panel 12 (original).](../assets/figures/ml_fig_c560_12.png)
*Figure — Barlow twins path c560. Synthetic teaching geometry—not a causal claim.*


![c561 teaching panel 12 (original).](../assets/figures/ml_fig_c561_12.png)
*Figure — VICReg var-inv path c561. Synthetic teaching geometry—not a causal claim.*


![c562 teaching panel 12 (original).](../assets/figures/ml_fig_c562_12.png)
*Figure — DINO teacher path c562. Synthetic teaching geometry—not a causal claim.*


![c563 teaching panel 12 (original).](../assets/figures/ml_fig_c563_12.png)
*Figure — MAE mask recon path c563. Synthetic teaching geometry—not a causal claim.*


![c564 teaching panel 12 (original).](../assets/figures/ml_fig_c564_12.png)
*Figure — JEPA latent path c564. Synthetic teaching geometry—not a causal claim.*


![c565 teaching panel 12 (original).](../assets/figures/ml_fig_c565_12.png)
*Figure — SwAV code swap path c565. Synthetic teaching geometry—not a causal claim.*


![c566 teaching panel 12 (original).](../assets/figures/ml_fig_c566_12.png)
*Figure — CLIP align path c566. Synthetic teaching geometry—not a causal claim.*


![c567 teaching panel 12 (original).](../assets/figures/ml_fig_c567_12.png)
*Figure — data2vec EMA path c567. Synthetic teaching geometry—not a causal claim.*


![c568 teaching panel 12 (original).](../assets/figures/ml_fig_c568_12.png)
*Figure — I-JEPA block path c568. Synthetic teaching geometry—not a causal claim.*


![c569 teaching panel 12 (original).](../assets/figures/ml_fig_c569_12.png)
*Figure — Masked unit path c569. Synthetic teaching geometry—not a causal claim.*


![c570 teaching panel 12 (original).](../assets/figures/ml_fig_c570_12.png)
*Figure — Contrast temp path c570. Synthetic teaching geometry—not a causal claim.*


![c571 teaching panel 12 (original).](../assets/figures/ml_fig_c571_12.png)
*Figure — Projection dim bars c571. Synthetic teaching geometry—not a causal claim.*


![c572 teaching panel 12 (original).](../assets/figures/ml_fig_c572_12.png)
*Figure — SimCLR NT-Xent path c572. Synthetic teaching geometry—not a causal claim.*


![c573 teaching panel 12 (original).](../assets/figures/ml_fig_c573_12.png)
*Figure — MoCo queue path c573. Synthetic teaching geometry—not a causal claim.*


![c574 teaching panel 12 (original).](../assets/figures/ml_fig_c574_12.png)
*Figure — BYOL stop-grad path c574. Synthetic teaching geometry—not a causal claim.*


![c575 teaching panel 12 (original).](../assets/figures/ml_fig_c575_12.png)
*Figure — SimSiam twin path c575. Synthetic teaching geometry—not a causal claim.*


![c576 teaching panel 12 (original).](../assets/figures/ml_fig_c576_12.png)
*Figure — Barlow twins path c576. Synthetic teaching geometry—not a causal claim.*


![c577 teaching panel 12 (original).](../assets/figures/ml_fig_c577_12.png)
*Figure — VICReg var-inv path c577. Synthetic teaching geometry—not a causal claim.*


![c578 teaching panel 12 (original).](../assets/figures/ml_fig_c578_12.png)
*Figure — DINO teacher path c578. Synthetic teaching geometry—not a causal claim.*


![c579 teaching panel 12 (original).](../assets/figures/ml_fig_c579_12.png)
*Figure — MAE mask recon path c579. Synthetic teaching geometry—not a causal claim.*


![c580 teaching panel 12 (original).](../assets/figures/ml_fig_c580_12.png)
*Figure — JEPA latent path c580. Synthetic teaching geometry—not a causal claim.*


![c581 teaching panel 12 (original).](../assets/figures/ml_fig_c581_12.png)
*Figure — SwAV code swap path c581. Synthetic teaching geometry—not a causal claim.*


![c582 teaching panel 12 (original).](../assets/figures/ml_fig_c582_12.png)
*Figure — CLIP align path c582. Synthetic teaching geometry—not a causal claim.*


![c583 teaching panel 12 (original).](../assets/figures/ml_fig_c583_12.png)
*Figure — data2vec EMA path c583. Synthetic teaching geometry—not a causal claim.*


![c584 teaching panel 12 (original).](../assets/figures/ml_fig_c584_12.png)
*Figure — I-JEPA block path c584. Synthetic teaching geometry—not a causal claim.*


![c585 teaching panel 12 (original).](../assets/figures/ml_fig_c585_12.png)
*Figure — Masked unit path c585. Synthetic teaching geometry—not a causal claim.*


![c586 teaching panel 12 (original).](../assets/figures/ml_fig_c586_12.png)
*Figure — Contrast temp path c586. Synthetic teaching geometry—not a causal claim.*


![c587 teaching panel 12 (original).](../assets/figures/ml_fig_c587_12.png)
*Figure — Projection dim bars c587. Synthetic teaching geometry—not a causal claim.*


![c588 teaching panel 12 (original).](../assets/figures/ml_fig_c588_12.png)
*Figure — SimCLR NT-Xent path c588. Synthetic teaching geometry—not a causal claim.*


![c589 teaching panel 12 (original).](../assets/figures/ml_fig_c589_12.png)
*Figure — MoCo queue path c589. Synthetic teaching geometry—not a causal claim.*


![c590 teaching panel 12 (original).](../assets/figures/ml_fig_c590_12.png)
*Figure — BYOL stop-grad path c590. Synthetic teaching geometry—not a causal claim.*


![c591 teaching panel 12 (original).](../assets/figures/ml_fig_c591_12.png)
*Figure — SimSiam twin path c591. Synthetic teaching geometry—not a causal claim.*


![c592 teaching panel 12 (original).](../assets/figures/ml_fig_c592_12.png)
*Figure — Barlow twins path c592. Synthetic teaching geometry—not a causal claim.*


![c593 teaching panel 12 (original).](../assets/figures/ml_fig_c593_12.png)
*Figure — VICReg var-inv path c593. Synthetic teaching geometry—not a causal claim.*


![c594 teaching panel 12 (original).](../assets/figures/ml_fig_c594_12.png)
*Figure — DINO teacher path c594. Synthetic teaching geometry—not a causal claim.*


![c595 teaching panel 12 (original).](../assets/figures/ml_fig_c595_12.png)
*Figure — MAE mask recon path c595. Synthetic teaching geometry—not a causal claim.*


![c596 teaching panel 12 (original).](../assets/figures/ml_fig_c596_12.png)
*Figure — JEPA latent path c596. Synthetic teaching geometry—not a causal claim.*


![c597 teaching panel 12 (original).](../assets/figures/ml_fig_c597_12.png)
*Figure — SwAV code swap path c597. Synthetic teaching geometry—not a causal claim.*


![c598 teaching panel 12 (original).](../assets/figures/ml_fig_c598_12.png)
*Figure — CLIP align path c598. Synthetic teaching geometry—not a causal claim.*


![c599 teaching panel 12 (original).](../assets/figures/ml_fig_c599_12.png)
*Figure — data2vec EMA path c599. Synthetic teaching geometry—not a causal claim.*


![c600 teaching panel 12 (original).](../assets/figures/ml_fig_c600_12.png)
*Figure — I-JEPA block path c600. Synthetic teaching geometry—not a causal claim.*


![c601 teaching panel 12 (original).](../assets/figures/ml_fig_c601_12.png)
*Figure — Masked unit path c601. Synthetic teaching geometry—not a causal claim.*


![c602 teaching panel 12 (original).](../assets/figures/ml_fig_c602_12.png)
*Figure — Contrast temp path c602. Synthetic teaching geometry—not a causal claim.*


![c603 teaching panel 12 (original).](../assets/figures/ml_fig_c603_12.png)
*Figure — Projection dim bars c603. Synthetic teaching geometry—not a causal claim.*


![c604 teaching panel 12 (original).](../assets/figures/ml_fig_c604_12.png)
*Figure — SimCLR NT-Xent path c604. Synthetic teaching geometry—not a causal claim.*


![c605 teaching panel 12 (original).](../assets/figures/ml_fig_c605_12.png)
*Figure — MoCo queue path c605. Synthetic teaching geometry—not a causal claim.*


![c606 teaching panel 12 (original).](../assets/figures/ml_fig_c606_12.png)
*Figure — BYOL stop-grad path c606. Synthetic teaching geometry—not a causal claim.*


![c607 teaching panel 12 (original).](../assets/figures/ml_fig_c607_12.png)
*Figure — SimSiam twin path c607. Synthetic teaching geometry—not a causal claim.*


![c608 teaching panel 12 (original).](../assets/figures/ml_fig_c608_12.png)
*Figure — Barlow twins path c608. Synthetic teaching geometry—not a causal claim.*


![c609 teaching panel 12 (original).](../assets/figures/ml_fig_c609_12.png)
*Figure — VICReg var-inv path c609. Synthetic teaching geometry—not a causal claim.*


![c610 teaching panel 12 (original).](../assets/figures/ml_fig_c610_12.png)
*Figure — DINO teacher path c610. Synthetic teaching geometry—not a causal claim.*


![c611 teaching panel 12 (original).](../assets/figures/ml_fig_c611_12.png)
*Figure — MAE mask recon path c611. Synthetic teaching geometry—not a causal claim.*


![c612 teaching panel 12 (original).](../assets/figures/ml_fig_c612_12.png)
*Figure — JEPA latent path c612. Synthetic teaching geometry—not a causal claim.*


![c613 teaching panel 12 (original).](../assets/figures/ml_fig_c613_12.png)
*Figure — SwAV code swap path c613. Synthetic teaching geometry—not a causal claim.*


![c614 teaching panel 12 (original).](../assets/figures/ml_fig_c614_12.png)
*Figure — CLIP align path c614. Synthetic teaching geometry—not a causal claim.*


![c615 teaching panel 12 (original).](../assets/figures/ml_fig_c615_12.png)
*Figure — data2vec EMA path c615. Synthetic teaching geometry—not a causal claim.*


![c616 teaching panel 12 (original).](../assets/figures/ml_fig_c616_12.png)
*Figure — I-JEPA block path c616. Synthetic teaching geometry—not a causal claim.*


![c617 teaching panel 12 (original).](../assets/figures/ml_fig_c617_12.png)
*Figure — Masked unit path c617. Synthetic teaching geometry—not a causal claim.*


![c618 teaching panel 12 (original).](../assets/figures/ml_fig_c618_12.png)
*Figure — Contrast temp path c618. Synthetic teaching geometry—not a causal claim.*


![c619 teaching panel 12 (original).](../assets/figures/ml_fig_c619_12.png)
*Figure — Projection dim bars c619. Synthetic teaching geometry—not a causal claim.*


![c620 teaching panel 12 (original).](../assets/figures/ml_fig_c620_12.png)
*Figure — SimCLR NT-Xent path c620. Synthetic teaching geometry—not a causal claim.*


![c621 teaching panel 12 (original).](../assets/figures/ml_fig_c621_12.png)
*Figure — MoCo queue path c621. Synthetic teaching geometry—not a causal claim.*


![c622 teaching panel 12 (original).](../assets/figures/ml_fig_c622_12.png)
*Figure — BYOL stop-grad path c622. Synthetic teaching geometry—not a causal claim.*


![c623 teaching panel 12 (original).](../assets/figures/ml_fig_c623_12.png)
*Figure — SimSiam twin path c623. Synthetic teaching geometry—not a causal claim.*


![c624 teaching panel 12 (original).](../assets/figures/ml_fig_c624_12.png)
*Figure — Barlow twins path c624. Synthetic teaching geometry—not a causal claim.*


![c625 teaching panel 12 (original).](../assets/figures/ml_fig_c625_12.png)
*Figure — VICReg var-inv path c625. Synthetic teaching geometry—not a causal claim.*


![c626 teaching panel 12 (original).](../assets/figures/ml_fig_c626_12.png)
*Figure — DINO teacher path c626. Synthetic teaching geometry—not a causal claim.*


![c627 teaching panel 12 (original).](../assets/figures/ml_fig_c627_12.png)
*Figure — MAE mask recon path c627. Synthetic teaching geometry—not a causal claim.*


![c628 teaching panel 12 (original).](../assets/figures/ml_fig_c628_12.png)
*Figure — JEPA latent path c628. Synthetic teaching geometry—not a causal claim.*


![c629 teaching panel 12 (original).](../assets/figures/ml_fig_c629_12.png)
*Figure — SwAV code swap path c629. Synthetic teaching geometry—not a causal claim.*


![c630 teaching panel 12 (original).](../assets/figures/ml_fig_c630_12.png)
*Figure — CLIP align path c630. Synthetic teaching geometry—not a causal claim.*


![c631 teaching panel 12 (original).](../assets/figures/ml_fig_c631_12.png)
*Figure — data2vec EMA path c631. Synthetic teaching geometry—not a causal claim.*


![c632 teaching panel 12 (original).](../assets/figures/ml_fig_c632_12.png)
*Figure — I-JEPA block path c632. Synthetic teaching geometry—not a causal claim.*


![c633 teaching panel 12 (original).](../assets/figures/ml_fig_c633_12.png)
*Figure — Masked unit path c633. Synthetic teaching geometry—not a causal claim.*


![c634 teaching panel 12 (original).](../assets/figures/ml_fig_c634_12.png)
*Figure — Contrast temp path c634. Synthetic teaching geometry—not a causal claim.*


![c635 teaching panel 12 (original).](../assets/figures/ml_fig_c635_12.png)
*Figure — Projection dim bars c635. Synthetic teaching geometry—not a causal claim.*


![c636 teaching panel 12 (original).](../assets/figures/ml_fig_c636_12.png)
*Figure — SimCLR NT-Xent path c636. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Self-supervised and generative deep learning extract structure from unlabeled data. Generative models capture p(x) while discriminative models capture p(y|x); encoders may be deterministic or stochastic. SOMs provide topology-preserving maps; Boltzmann machines and RBMs define energy-based densities trained with contrastive divergence; DBNs and DBMs stack these ideas deeply. Autoencoders reconstruct through bottlenecks with sparse, denoising, contractive, stacked, variational, and U-Net variants. Worked numerical sketches for contrastive pair loss and a univariate VAE KL/ELBO make the objectives concrete. GANs adversarially train generators with challenges of mode collapse, oscillation, slow convergence, and uninformative losses, evaluated by IS and FID, and specialized into CGAN, DCGAN, WGAN/WGAN-GP, Pix2Pix, CycleGAN, and StyleGAN. Contrastive and triplet losses with Siamese networks learn metric embeddings. Text-to-image systems combine zero-shot language alignment (CLIP), discrete codes (VQ-GAN), autoregression, and diffusion (DALL-E, Imagen, Parti, Stable Diffusion). Choosing among VAE, GAN, and diffusion families depends on stability, sharpness, sampling cost, and conditional control needs. In medicine, SSL is a strategy for scarce labels: domain-appropriate pretraining, strict patient-level separation, probing checklists, and skeptical use of synthetic images determine whether these methods aid neurologic care and epidemiologic research.

## Practice and Reflection

(1) Explain why an unrestricted autoencoder with equal input/output dimension can fail to learn useful features, and name three constraints that prevent identity cheating.

(2) Write the ELBO terms for a VAE and interpret the role of the KL term when β > 1.

(3) Compare mode collapse in GANs with posterior collapse in VAEs; propose one diagnostic for each.

(4) Derive the gradient-penalty term in WGAN-GP and explain why weight clipping is an inferior Lipschitz enforcement.

(5) Design a Siamese sampling strategy for matching admission and 24-hour follow-up NCCT in the same patient versus different patients.

(6) Sketch a SimCLR-style pretraining pipeline for multi-site head CT, including augmentations you would forbid and why.

(7) Why might FID computed with ImageNet Inception features misrank medical image generators, and what alternative evaluation would you add?

(8) For unpaired MRI contrast conversion with CycleGAN, list failure modes that could alter lesion appearance and how you would detect them before downstream use.

(9) Map CLIP zero-shot classification steps for three stroke-related prompt templates; discuss prompt sensitivity.

(10) Propose a governance checklist for using latent diffusion models to augment a rare cerebral venous thrombosis training set.

(11) Using the worked contrastive numbers, recompute loss if margin m=0.5 and d13=0.8 for a dissimilar pair.

(12) Outline a linear-probe protocol that fairly compares ImageNet vs in-domain SSL initialization for ICH detection.

(13) Using the worked triplet numbers, compute the loss for margin m=0.4 with a negative n″ at distance d(a,n″)=0.6 and d(a,p)=0.5; state whether the triplet is active and what the gradient would do.

(14) Design a nearest-neighbor memorization audit for a diffusion model trained on multi-site CT: which embedding space, what distance threshold and how to calibrate it, and how to prevent scanner site from confounding the audit.


## Fine-tune legality checklist (teaching)

| Step | Question | Failure mode |
|------|----------|--------------|
| 1 | What is index time for the clinical prediction? | Features after the decision time |
| 2 | Were pretraining images/text free of test-patient leakage? | Contaminated foundation data |
| 3 | Is the fine-tune label independent of post-outcome charting? | Label leakage |
| 4 | External site performance reported? | Overfit to one scanner/vendor |
| 5 | Calibration checked after fine-tune? | Sharp but unusable probabilities |

