#!/usr/bin/env python3
"""Cycle-16 densify — push remaining floor-13 chapters toward 14+."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#0d9488"
DEEP = "#0f766e"
INK = "#0f172a"
GOLD = "#c9a227"
SOFT = "#ecfeff"
SLATE = "#64748b"
GRAY = "#94a3b8"


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", path.name)
    return path


def style_ax(ax, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def fig_spectral_clustering():
    """Ch04/07: spectral clustering affinity → Laplacian → k-means on eigenvectors."""
    rng = np.random.default_rng(3)
    # two moons-ish via angle
    t1 = rng.uniform(0, np.pi, 80)
    t2 = rng.uniform(0, np.pi, 80)
    X1 = np.c_[np.cos(t1), np.sin(t1)] + rng.normal(0, 0.08, (80, 2))
    X2 = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)] + rng.normal(0, 0.08, (80, 2))
    X = np.vstack([X1, X2])
    # affinity
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    W = np.exp(-d2 / (2 * 0.25 ** 2))
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    # smallest eigenvectors of L (generalized would use D^{-1/2} L ...)
    # use normalized Laplacian
    d_inv_sqrt = 1 / np.sqrt(np.diag(D) + 1e-9)
    Ln = (d_inv_sqrt[:, None] * L) * d_inv_sqrt[None, :]
    evals, evecs = np.linalg.eigh(Ln)
    U = evecs[:, 1:3]  # skip trivial
    # kmeans on U
    cents = U[[0, 80]]
    for _ in range(20):
        lab = ((U[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for j in range(2):
            if (lab == j).any():
                cents[j] = U[lab == j].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.scatter(X[lab == 0, 0], X[lab == 0, 1], s=16, c=TEAL, alpha=0.8)
    ax.scatter(X[lab == 1, 0], X[lab == 1, 1], s=16, c=GOLD, alpha=0.8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    style_ax(ax, "Spectral clusters recover moons")

    ax = axes[1]
    ax.scatter(U[lab == 0, 0], U[lab == 0, 1], s=16, c=TEAL, alpha=0.8, label="cluster 0")
    ax.scatter(U[lab == 1, 0], U[lab == 1, 1], s=16, c=GOLD, alpha=0.8, label="cluster 1")
    ax.set_xlabel("eigenvector 2")
    ax.set_ylabel("eigenvector 3")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "k-means in spectral embedding")
    ax.text(0.98, 0.08, "Affinity geometry ≠ etiology.\nTune kernel width carefully.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Spectral clustering sketch (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_spectral_clustering.png")


def fig_scree_pca_parallel():
    """Ch07: scree vs parallel analysis for component retention."""
    rng = np.random.default_rng(9)
    n, p = 200, 12
    # low rank signal
    Z = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, p))
    X = Z @ W + 0.35 * rng.normal(size=(n, p))
    X = (X - X.mean(0)) / X.std(0)
    evals = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    # parallel: mean evals of shuffled
    nulls = []
    for b in range(40):
        Xs = X.copy()
        for j in range(p):
            rng.shuffle(Xs[:, j])
        nulls.append(np.linalg.eigvalsh(np.cov(Xs, rowvar=False))[::-1])
    nulls = np.mean(nulls, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    k = np.arange(1, p + 1)
    ax.plot(k, evals, "o-", color=TEAL, lw=2.2, label="data eigenvalues")
    ax.plot(k, nulls, "s--", color=GOLD, lw=2.0, label="parallel analysis mean")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Scree + parallel analysis")

    ax = axes[1]
    keep = evals > nulls
    ax.bar(k, evals, color=[TEAL if k_ else GRAY for k_ in keep], edgecolor="white")
    ax.plot(k, nulls, "o--", color=GOLD, lw=1.8, label="null mean")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Retain components above null")
    ax.text(0.98, 0.9, f"suggest keep first {int(keep.sum())}\nPCA axes ≠ causal factors",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("PCA retention: scree vs parallel analysis (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_pca_parallel.png")


def fig_vae_kl_beta_trade():
    """Ch11: β-VAE recon vs KL tradeoff (complement existing ELBO fig if any)."""
    betas = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    # synthetic tradeoff curves
    recon = 1.2 / (1 + 0.4 * betas) + 0.15
    kl = 0.05 + 0.35 * np.log1p(betas)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(betas, recon, "o-", color=TEAL, lw=2.4, label="recon term")
    ax.plot(betas, kl, "s-", color=GOLD, lw=2.4, label="KL term")
    ax.set_xlabel(r"$\beta$ weight on KL")
    ax.set_ylabel("loss component (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, r"$\beta$-VAE pressure on latent")

    ax = axes[1]
    # latent traversal caricature: low beta entangled, high beta factorized axes
    rng = np.random.default_rng(2)
    for i, (beta, title, col) in enumerate([(0.2, r"low $\beta$: entangled", GRAY),
                                            (4.0, r"high $\beta$: more factorized", TEAL)]):
        ax = axes[1] if i == 1 else axes[1]
    # two scatter panels reused as one with two clouds
    z1 = rng.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], 120)
    z2 = rng.multivariate_normal([0, 0], [[1, 0.05], [0.05, 1]], 120)
    ax.scatter(z1[:, 0], z1[:, 1], s=12, c=GRAY, alpha=0.5, label=r"low $\beta$ latent")
    ax.scatter(z2[:, 0], z2[:, 1], s=12, c=TEAL, alpha=0.6, label=r"high $\beta$ latent")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Disentanglement is partial & fragile")
    ax.text(0.02, 0.08, "Higher β can hurt recon of rare\nlesion cues. Latent axes ≠ causes.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.suptitle("β-VAE: reconstruction–KL tradeoff (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_beta_vae_tradeoff.png")


def fig_lora_vs_full():
    """Ch14: LoRA parameter count vs full FT quality caricature."""
    ranks = np.array([1, 2, 4, 8, 16, 32, 64])
    # params relative
    d = 4096
    lora_params = 2 * ranks * d / (d * d) * 100  # % of one square matrix
    full = 100
    # quality caricature
    qual = 0.72 + 0.18 * (1 - np.exp(-ranks / 10))
    full_q = 0.92

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(ranks, lora_params, "o-", color=TEAL, lw=2.4, label="LoRA % of one W")
    ax.axhline(full, color=GOLD, ls="--", lw=1.5, label="full FT = 100%")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("% trainable params (one square layer)")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Parameter thrift of low-rank adapters")

    ax = axes[1]
    ax.plot(ranks, qual, "o-", color=TEAL, lw=2.4, label="LoRA task score (synth)")
    ax.axhline(full_q, color=GOLD, ls="--", lw=1.5, label="full FT score")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("downstream score (synthetic)")
    ax.set_ylim(0.65, 1.0)
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Diminishing returns in r")
    ax.text(0.98, 0.08, "Validate rare phenotypes.\nAdapters are PEFT, not causation.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("LoRA vs full fine-tune: params and returns (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_lora_vs_full.png")


def fig_decision_curve_nb():
    """Ch17: net benefit decision curve."""
    pt = np.linspace(0.01, 0.6, 80)
    # synthetic model
    # NB = sens*prev - (1-spec)*(1-prev)*(pt/(1-pt))
    prev = 0.15
    # model with sens, spec depending on threshold ~ pt for well-calibrated
    # approximate: treat threshold = pt
    sens = np.clip(1 - 1.2 * pt, 0.2, 0.98)
    spec = np.clip(0.5 + 0.7 * pt, 0.4, 0.98)
    nb_model = sens * prev - (1 - spec) * (1 - prev) * (pt / (1 - pt))
    nb_all = prev - (1 - prev) * (pt / (1 - pt))
    nb_none = np.zeros_like(pt)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(pt, nb_model, color=TEAL, lw=2.5, label="model")
    ax.plot(pt, nb_all, color=GOLD, lw=2.0, ls="--", label="treat all")
    ax.plot(pt, nb_none, color=GRAY, lw=1.5, label="treat none")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("threshold probability pt")
    ax.set_ylabel("net benefit")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Decision curve (synthetic)")

    ax = axes[1]
    # delta NB vs treat-all
    dnb = nb_model - np.maximum(nb_all, 0)
    ax.fill_between(pt, 0, dnb, where=dnb > 0, color=TEAL, alpha=0.35, label="model better")
    ax.plot(pt, dnb, color=DEEP, lw=2.2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("threshold probability pt")
    ax.set_ylabel("Δ net benefit vs best default")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Utility depends on the decision threshold")
    ax.text(0.98, 0.9, "Net benefit ≠ AUROC.\nNot a causal treatment effect.\nUse clinically plausible pt.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Decision-curve net benefit vs threshold (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_decision_curve_nb.png")


def fig_bayes_factor_sketch():
    """Ch03: likelihood ratio path / Bayes factor teaching."""
    # two simple hypotheses for a coin
    k, n = 14, 20
    p = np.linspace(0.05, 0.95, 200)
    lik = p ** k * (1 - p) ** (n - k)
    lik /= lik.max()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(p, lik, color=TEAL, lw=2.4)
    ax.axvline(0.5, color=GRAY, ls="--", lw=1.2, label="H0: p=0.5")
    ax.axvline(k / n, color=GOLD, ls="--", lw=1.2, label=rf"MLE={k/n:.2f}")
    ax.set_xlabel("Bernoulli p")
    ax.set_ylabel("normalized likelihood")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, rf"Likelihood for k={k} events in n={n}")

    ax = axes[1]
    # BF approx lik(MLE)/lik(0.5) as crude teaching (not formal BF with priors)
    L0 = 0.5 ** k * 0.5 ** (n - k)
    Lm = (k / n) ** k * (1 - k / n) ** (n - k)
    ratio = Lm / L0
    ax.bar(["under H0\np=0.5", "at MLE"], [L0, Lm], color=[GRAY, TEAL], edgecolor="white")
    ax.set_ylabel("likelihood value")
    style_ax(ax, rf"Likelihood ratio ≈ {ratio:.1f} (teaching)")
    ax.text(0.5, 0.9, "Formal Bayes factors need priors.\nLR updates odds; still not causation.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Likelihood ratios underwrite Bayesian updates (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_likelihood_ratio_coin.png")


def fig_preface_ops_loop():
    """Preface: monitor → evaluate → rollback loop."""
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    nodes = [
        (1.5, 3.0, "Deploy\nv1", TEAL),
        (4.5, 3.0, "Monitor\nPSI / AUROC", GOLD),
        (7.5, 3.0, "Investigate\nshift / labels", DEEP),
        (10.2, 3.0, "Rollback\nor retrain", TEAL),
    ]
    for x, y, lab, c in nodes:
        ax.add_patch(FancyBboxPatch((x - 1.1, y - 0.9), 2.2, 1.8, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor=c, edgecolor="none", alpha=0.9))
        ax.text(x, y, lab, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    for x1, x2 in [(2.6, 3.4), (5.6, 6.4), (8.6, 9.1)]:
        ax.annotate("", xy=(x2, 3.0), xytext=(x1, 3.0),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(1.5, 1.5), xytext=(10.2, 1.5),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(6, 0.7, "versioned loop — silent auto-retrain without governance is a hazard",
            ha="center", fontsize=9, color=SLATE)
    ax.text(6, 5.3, "Preface ops: shipping is the start of evaluation",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(6, 4.75, "Prediction service lifecycle ≠ causal discovery",
            ha="center", fontsize=9, color="#475569")
    fig.tight_layout()
    save(fig, "ml_fig_ops_lifecycle.png")


def main():
    fig_spectral_clustering()
    fig_scree_pca_parallel()
    fig_vae_kl_beta_trade()
    fig_lora_vs_full()
    fig_decision_curve_nb()
    fig_bayes_factor_sketch()
    fig_preface_ops_loop()
    print("DONE cycle-16")


if __name__ == "__main__":
    main()
