#!/usr/bin/env python3
"""Cycle-26 densify — 12 chapters from floor-18 toward >=19 (teal; original)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, DEEP, INK, GOLD, GRAY, SLATE = (
    "#0d9488",
    "#0f766e",
    "#0f172a",
    "#c9a227",
    "#94a3b8",
    "#64748b",
)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", name)


def style_ax(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def fig_svd_spectrum():
    s = np.array([10, 6.5, 3.2, 1.1, 0.4, 0.15, 0.08, 0.03])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.semilogy(np.arange(1, len(s) + 1), s, color=TEAL, lw=2.4, marker="o")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ (log scale)")
    style_ax(ax, "SVD spectrum (synthetic matrix)")
    ax.text(0.98, 0.85, "Rank cues live in the drop.\nFactors ≠ clinical causes.", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_svd_spectrum.png")


def fig_external_val_gap():
    sites = ["dev", "site B", "site C", "next year"]
    auroc = [0.89, 0.81, 0.78, 0.76]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(sites, auroc, color=[TEAL, GOLD, GOLD, DEEP], edgecolor="white")
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("AUROC (synthetic)")
    style_ax(ax, "External validation shrinks optimism")
    ax.text(0.98, 0.9, "Transport is the real test.\nPred ≠ cause.", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_external_val_gap.png")


def fig_boxplot_vs_violin():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 200)
    b = np.concatenate([rng.normal(-0.5, 0.5, 100), rng.normal(1.2, 0.4, 100)])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].boxplot([a, b], tick_labels=["A", "B"], patch_artist=True,
                    boxprops=dict(facecolor=TEAL, alpha=0.6))
    style_ax(axes[0], "Boxplots hide multimodality")
    parts = axes[1].violinplot([a, b], showmeans=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(TEAL)
        pc.set_alpha(0.7)
    axes[1].set_xticks([1, 2], ["A", "B"])
    style_ax(axes[1], "Violins show density shape")
    fig.suptitle("Distribution displays (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, "Display choice changes interpretation—not etiology.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_box_vs_violin.png")


def fig_power_curve():
    n = np.array([20, 40, 80, 160, 320, 640])
    # power for fixed effect
    power = 1 - np.exp(-n / 120)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(n, power, color=TEAL, lw=2.4, marker="o")
    ax.axhline(0.8, color=GOLD, ls="--", lw=1.4, label="80% power guide")
    ax.set_xlabel("n per group (teaching)")
    ax.set_ylabel("power (synthetic)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Power grows with n (fixed effect size)")
    ax.text(0.98, 0.3, "Power ≠ clinical importance.\nNot a causal design alone.", transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_power_curve.png")


def fig_hierarchical_clustering_dendro():
    # simple handmade dendrogram-like linkage heights
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    # draw schematic dendrogram
    def link(x1, x2, h, y0=0):
        xm = 0.5 * (x1 + x2)
        ax.plot([x1, x1, x2, x2], [y0, h, h, y0], color=TEAL, lw=2)

    link(1, 2, 0.4)
    link(3, 4, 0.5)
    link(1.5, 3.5, 1.0, y0=0.4)  # approx
    ax.plot([1.5, 1.5], [0.4, 1.0], color=TEAL, lw=2)
    ax.plot([3.5, 3.5], [0.5, 1.0], color=TEAL, lw=2)
    ax.plot([1.5, 3.5], [1.0, 1.0], color=TEAL, lw=2)
    ax.plot([2.5, 5], [1.0, 1.6], color=GOLD, lw=2)
    ax.plot([5, 5], [0, 1.6], color=GOLD, lw=2)
    ax.plot([2.5, 2.5], [1.0, 1.6], color=TEAL, lw=2)
    ax.set_xticks([1, 2, 3, 4, 5], ["p1", "p2", "p3", "p4", "p5"])
    ax.set_ylabel("linkage distance")
    style_ax(ax, "Agglomerative dendrogram (schematic)")
    ax.text(0.02, 0.9, "Cut height chooses k.\nTree ≠ taxonomy of disease.", transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_dendrogram_schematic.png")


def fig_bm25_terms():
    terms = ["NIHSS", "LVO", "tPA", "the", "patient", "scanner"]
    score = [2.8, 2.4, 1.9, 0.05, 0.2, 1.1]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    colors = [TEAL if s > 1 else GRAY for s in score]
    ax.barh(terms[::-1], score[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("BM25-like term weight (synthetic doc)")
    style_ax(ax, "IR term weights favor rare informative tokens")
    ax.text(0.98, 0.15, "High weight ≠ causal role\nin outcomes.", transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_bm25_terms.png")


def fig_interaction_surface():
    x = np.linspace(-2, 2, 40)
    y = np.linspace(-2, 2, 40)
    X, Y = np.meshgrid(x, y)
    Z = 0.3 * X + 0.3 * Y + 0.6 * X * Y
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cs = ax.contourf(X, Y, Z, levels=12, cmap="YlGnBu")
    fig.colorbar(cs, ax=ax, fraction=0.046)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    style_ax(ax, "Interaction surface x1·x2 (synthetic)")
    ax.text(0.02, 0.92, "Interactions are statistical.\nNot automatic biology.", transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_interaction_surface.png")


def fig_kernel_pca_sketch():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 200)
    r = 1 + 0.1 * rng.normal(size=200)
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    # inner vs outer labels by radius noise
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].scatter(X[:, 0], X[:, 1], c=TEAL, s=12, alpha=0.7)
    style_ax(axes[0], "Nonlinear manifold (input)")
    # fake KPCA unwrap: angle vs small noise
    axes[1].scatter(t, rng.normal(0, 0.05, size=t.shape), c=TEAL, s=12, alpha=0.7)
    axes[1].set_xlabel("unwrapped coordinate")
    axes[1].set_ylabel("KPCA-2 (noise)")
    style_ax(axes[1], "Kernel PCA-style unwrap (cartoon)")
    fig.suptitle("Nonlinear dim reduction intuition (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, "Coordinates are features—not causes.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_kpca_unwrap.png")


def fig_glm_link():
    eta = np.linspace(-3, 3, 200)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(eta, 1 / (1 + np.exp(-eta)), color=TEAL, lw=2.4, label="logit link μ")
    ax.plot(eta, np.exp(eta), color=GOLD, lw=2.0, label="log link μ (Poisson-ish)")
    ax.set_ylim(0, 3)
    ax.set_xlabel("linear predictor η")
    ax.set_ylabel("mean μ")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "GLM link functions (teaching)")
    ax.text(0.02, 0.85, "Link choice is modeling.\nCoefficients need care\nbefore causal talk.", transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_glm_links.png")


def fig_dropout_train_eval():
    p = np.linspace(0, 0.8, 9)
    train_acc = 0.95 - 0.05 * p
    test_acc = 0.75 + 0.15 * p * (1 - p / 0.9) - 0.2 * (p > 0.5) * (p - 0.5)
    test_acc = np.clip(test_acc, 0.6, 0.95)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(p, train_acc, color=GOLD, lw=2.2, marker="o", label="train")
    ax.plot(p, test_acc, color=TEAL, lw=2.2, marker="s", label="test")
    ax.set_xlabel("dropout rate")
    ax.set_ylabel("accuracy (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Dropout regularizes—too much hurts")
    ax.text(0.98, 0.2, "Regularization ≠ causation.\nKeep eval without dropout.", transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_dropout_rate.png")


def fig_ssl_linear_probe():
    frac = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    from_scratch = 0.55 + 0.3 * np.log10(frac * 100) / 2
    probe = 0.72 + 0.15 * np.log10(frac * 100) / 2
    ft = 0.75 + 0.18 * np.log10(frac * 100) / 2
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(frac, from_scratch, color=GRAY, lw=2, marker="o", label="from scratch")
    ax.plot(frac, probe, color=GOLD, lw=2.2, marker="s", label="linear probe")
    ax.plot(frac, ft, color=TEAL, lw=2.4, marker="^", label="fine-tune")
    ax.set_xscale("log")
    ax.set_xlabel("labeled fraction")
    ax.set_ylabel("accuracy (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "SSL helps most when labels are scarce")
    ax.text(0.02, 0.15, "Label efficiency ≠ causal\nidentification.", transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_ssl_label_efficiency.png")


def fig_token_embed_2d():
    rng = np.random.default_rng(5)
    # clusters of tokens
    med = rng.normal([0, 0], 0.2, (8, 2))
    neuro = rng.normal([2, 1.5], 0.25, (8, 2))
    noise = rng.normal([1, -1], 0.5, (10, 2))
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter(med[:, 0], med[:, 1], c=TEAL, s=50, label="meds-ish")
    ax.scatter(neuro[:, 0], neuro[:, 1], c=GOLD, s=50, label="neuro-ish")
    ax.scatter(noise[:, 0], noise[:, 1], c=GRAY, s=40, label="other")
    for i, p in enumerate(med[:3]):
        ax.annotate(f"m{i}", p, textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Token embedding neighborhoods (toy 2D)")
    ax.text(0.02, 0.05, "Neighbors are corpus geometry—\nnot clinical synonymy proof.", transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_token_embed_2d.png")


def main() -> None:
    fig_svd_spectrum()
    fig_external_val_gap()
    fig_boxplot_vs_violin()
    fig_power_curve()
    fig_hierarchical_clustering_dendro()
    fig_bm25_terms()
    fig_interaction_surface()
    fig_kernel_pca_sketch()
    fig_glm_link()
    fig_dropout_train_eval()
    fig_ssl_linear_probe()
    fig_token_embed_2d()
    print("DONE cycle-26")


if __name__ == "__main__":
    main()
