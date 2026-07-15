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

