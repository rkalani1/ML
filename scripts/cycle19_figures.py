#!/usr/bin/env python3
"""Cycle-19 densify — raise remaining floor-14 chapters to >=15."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, DEEP, INK, GOLD, GRAY, SLATE, SOFT = "#0d9488", "#0f766e", "#0f172a", "#c9a227", "#94a3b8", "#64748b", "#ecfeff"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", p.name)


def style_ax(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def fig_under_over_fit_curves():
    """Ch01 learning curves under/over fit."""
    n = np.array([50, 100, 200, 400, 800, 1600])
    # underfit: both high
    tr_u = 0.35 + 0.02 * np.log(n)
    te_u = 0.40 + 0.015 * np.log(n)
    # overfit: train low, test high then improves
    tr_o = 0.05 + 0.01 * np.log(n)
    te_o = 0.45 - 0.08 * np.log(n / 50)
    te_o = np.clip(te_o, 0.12, 1)
    # good
    tr_g = 0.18 - 0.03 * np.log(n / 50)
    te_g = 0.28 - 0.04 * np.log(n / 50)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    for ax, tr, te, title in [
        (axes[0], tr_u, te_u, "Underfit: both high"),
        (axes[1], tr_o, te_o, "Overfit: gap large"),
        (axes[2], tr_g, te_g, "Better capacity match"),
    ]:
        ax.plot(n, tr, "o-", color=TEAL, lw=2.2, label="train error")
        ax.plot(n, te, "s-", color=GOLD, lw=2.2, label="val error")
        ax.set_xscale("log")
        ax.set_xlabel("n train")
        ax.set_ylabel("error")
        ax.set_ylim(0, 0.55)
        ax.legend(frameon=False, fontsize=7)
        style_ax(ax, title)
    fig.suptitle("Learning-curve signatures of under/overfitting (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "ml_fig_learning_signatures.png")


def fig_small_multiples():
    """Ch02 small multiples vs dual axis."""
    x = np.arange(6)
    a = np.array([10, 12, 11, 14, 13, 15])
    b = np.array([100, 102, 98, 105, 101, 110])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(x, a, "o-", color=TEAL, lw=2)
    ax2.plot(x, b, "s-", color=GOLD, lw=2)
    ax.set_ylabel("rate A", color=TEAL)
    ax2.set_ylabel("count B", color=GOLD)
    style_ax(ax, "Dual axis — easy to misread")
    ax.text(0.5, 0.08, "Different scales invite false parallels", transform=ax.transAxes,
            ha="center", fontsize=8, color="#b45309")
    ax = axes[1]
    ax.plot(x, a / a[0], "o-", color=TEAL, lw=2, label="A / A0")
    ax.plot(x, b / b[0], "s-", color=GOLD, lw=2, label="B / B0")
    ax.set_ylabel("index (relative to start)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Indexed small-comparable series")
    fig.suptitle("Prefer comparable scales over dual axes (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_dual_vs_index.png")


def fig_hdbscan_stability():
    """Ch04 HDBSCAN stability caricature."""
    rng = np.random.default_rng(3)
    c1 = rng.normal([0, 0], 0.3, (50, 2))
    c2 = rng.normal([3, 0.2], 0.35, (50, 2))
    noise = rng.uniform(-1, 4, (30, 2))
    X = np.vstack([c1, c2, noise])
    lab = np.array([0] * 50 + [1] * 50 + [-1] * 30)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.scatter(X[lab == 0, 0], X[lab == 0, 1], c=TEAL, s=18, label="cluster")
    ax.scatter(X[lab == 1, 0], X[lab == 1, 1], c=GOLD, s=18, label="cluster")
    ax.scatter(X[lab < 0, 0], X[lab < 0, 1], c=GRAY, s=14, alpha=0.6, label="noise")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Density clusters + noise")
    ax = axes[1]
    # condensed tree stability bars teaching
    names = ["C1", "C2", "C1a", "C1b", "noise"]
    stab = [0.82, 0.78, 0.35, 0.30, 0.05]
    ax.barh(names, stab, color=[TEAL, GOLD, GRAY, GRAY, GRAY], edgecolor="white")
    ax.axvline(0.5, color=DEEP, ls="--", lw=1.2)
    ax.set_xlabel("cluster stability (teaching)")
    style_ax(ax, "Extract persistent clusters only")
    ax.text(0.98, 0.08, "Stability ≠ clinical subtype.\nNoise points need review paths.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Density clustering with noise and stability (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_density_stability.png")


def fig_tfidf_idf_curve():
    """Ch05 IDF curve."""
    df = np.arange(1, 500)
    N = 1000
    idf = np.log(N / df)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(df, idf, color=TEAL, lw=2.5)
    ax.set_xlabel("document frequency df(t)")
    ax.set_ylabel("idf(t) = log(N/df)")
    ax.axvline(50, color=GOLD, ls="--", lw=1.2)
    ax.text(55, idf[49], "common terms downweighted", fontsize=8, color=SLATE)
    style_ax(ax, "IDF downweights ubiquitous tokens")
    ax.text(0.98, 0.9, "TF–IDF is a retrieval weight,\nnot clinical importance or cause.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_idf_curve.png")


def fig_nmf_parts_add():
    """Ch07 NMF additive parts."""
    rng = np.random.default_rng(1)
    W = np.abs(rng.normal(size=(8, 3)))
    H = np.abs(rng.normal(size=(3, 10)))
    X = W @ H
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    axes[0].imshow(X, cmap="YlGnBu", aspect="auto")
    style_ax(axes[0], "X ≈ WH")
    axes[1].imshow(W, cmap="YlGnBu", aspect="auto")
    style_ax(axes[1], "W (parts)")
    axes[2].imshow(H, cmap="YlOrBr", aspect="auto")
    style_ax(axes[2], "H (coefficients)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("NMF: nonnegative additive parts (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.text(0.5, -0.02, "Parts are additive factors—not necessarily causal disease modules.",
             ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_nmf_parts_matrix.png")


def fig_residual_qq():
    """Ch08 residual QQ plot."""
    rng = np.random.default_rng(4)
    e = rng.standard_t(df=3, size=200)  # heavy tails
    e = (e - e.mean()) / e.std()
    e.sort()
    # theoretical normal quantiles via large normal sample order stats
    q = np.sort(rng.normal(size=100000))[np.linspace(0, 99999, len(e)).astype(int)]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.hist(e, bins=25, color=TEAL, edgecolor="white", density=True)
    style_ax(ax, "Residuals (heavy-tailed synth)")
    ax = axes[1]
    ax.plot(q, e, "o", color=TEAL, markersize=4, alpha=0.7)
    lim = max(abs(q).max(), abs(e).max())
    ax.plot([-lim, lim], [-lim, lim], ":", color=INK)
    ax.set_xlabel("theoretical normal quantiles")
    ax.set_ylabel("sample residual quantiles")
    style_ax(ax, "QQ plot: tails deviate")
    ax.text(0.02, 0.9, "Check before trusting\nGaussian intervals.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.suptitle("Residual diagnostics: histogram and QQ (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_residual_qq.png")


def fig_residual_connection():
    """Ch10 residual block sketch."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [(0.5, 1.8, "x"), (3.5, 1.8, "F(x)\n(layers)"), (7.0, 1.8, "x+F(x)")]
    cols = [TEAL, GOLD, DEEP]
    for (x, y, t), c in zip(boxes, cols):
        ax.add_patch(FancyBboxPatch((x, y), 2.4, 1.6, boxstyle="round,pad=0.04", facecolor=c, edgecolor="none"))
        ax.text(x + 1.2, y + 0.8, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(3.4, 2.6), xytext=(3.0, 2.6), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(6.9, 2.6), xytext=(6.0, 2.6), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    # skip
    ax.annotate("", xy=(7.0, 4.0), xytext=(1.7, 4.0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.0,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(4.2, 4.35, "identity skip", ha="center", fontsize=10, color=TEAL, fontweight="bold")
    ax.text(6, 0.7, "Eases gradient flow; still not a causal residual of biology", ha="center", fontsize=9, color=SLATE)
    ax.text(6, 0.2, "Residual learning: fit F, reuse x", ha="center", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ml_fig_residual_block.png")


def fig_simclr_pipeline():
    """Ch11 SimCLR two-view pipeline."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    steps = [(0.3, "image"), (2.5, "aug t"), (2.5, "aug t′"), (5.0, "encoder"), (7.5, "proj"), (10.0, "InfoNCE")]
    # custom layout
    ax.add_patch(FancyBboxPatch((0.3, 1.8), 1.8, 1.6, boxstyle="round", facecolor=GRAY, edgecolor="none"))
    ax.text(1.2, 2.6, "x", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.add_patch(FancyBboxPatch((2.8, 3.0), 1.8, 1.2, boxstyle="round", facecolor=TEAL, edgecolor="none"))
    ax.text(3.7, 3.6, "t(x)", ha="center", va="center", color="white", fontweight="bold")
    ax.add_patch(FancyBboxPatch((2.8, 1.0), 1.8, 1.2, boxstyle="round", facecolor=TEAL, edgecolor="none"))
    ax.text(3.7, 1.6, "t′(x)", ha="center", va="center", color="white", fontweight="bold")
    ax.add_patch(FancyBboxPatch((5.5, 1.5), 2.2, 2.2, boxstyle="round", facecolor=GOLD, edgecolor="none"))
    ax.text(6.6, 2.6, "f(·)\nshared", ha="center", va="center", color=INK, fontweight="bold")
    ax.add_patch(FancyBboxPatch((8.5, 1.5), 2.0, 2.2, boxstyle="round", facecolor=DEEP, edgecolor="none"))
    ax.text(9.5, 2.6, "g(·)\nproj", ha="center", va="center", color="white", fontweight="bold")
    ax.add_patch(FancyBboxPatch((11.2, 1.5), 2.4, 2.2, boxstyle="round", facecolor=TEAL, edgecolor="none"))
    ax.text(12.4, 2.6, "InfoNCE\nτ", ha="center", va="center", color="white", fontweight="bold")
    for x1, x2, y in [(2.1, 2.8, 3.5), (2.1, 2.8, 1.6), (4.6, 5.5, 3.2), (4.6, 5.5, 2.0), (7.7, 8.5, 2.6), (10.5, 11.2, 2.6)]:
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(7, 0.5, "Two views of one patient scan; other batch items are negatives — geometry ≠ causation",
            ha="center", fontsize=9, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_simclr_pipeline.png")


def fig_kv_cache():
    """Ch14 KV cache memory vs length."""
    N = np.array([512, 1024, 2048, 4096, 8192])
    # memory relative ~ N * layers * heads * dim
    kv = N / 512
    weights = np.ones_like(N) * 1.0
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(N, weights, "--", color=GRAY, lw=2, label="weights (fixed)")
    ax.plot(N, kv, "o-", color=TEAL, lw=2.4, label="KV-cache (∝ N)")
    ax.set_xlabel("context length N")
    ax.set_ylabel("relative memory units")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Decode memory often dominated by KV cache")
    ax.text(0.98, 0.1, "MQA/GQA shrink cache.\nLong context ≠ better care\nwithout retrieval discipline.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_kv_cache.png")


def fig_model_card_gates():
    """Ch17 model card gates checklist visual."""
    gates = ["Intended use", "Training data", "Metrics+slices", "Calibration", "Limits", "Monitor plan"]
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, g in enumerate(gates):
        x = 0.4 + (i % 3) * 3.9
        y = 2.2 if i < 3 else 0.4
        ax.add_patch(FancyBboxPatch((x, y), 3.5, 1.4, boxstyle="round,pad=0.04",
                                    facecolor=TEAL if i % 2 == 0 else DEEP, edgecolor="none", alpha=0.9))
        ax.text(x + 1.75, y + 0.7, g, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(6, 3.85, "Model card minimum gates (teaching)", ha="center", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ml_fig_model_card_gates.png")


def fig_glossary_youden():
    """Ch18 Youden J vs threshold."""
    t = np.linspace(0, 1, 100)
    se = 1 / (1 + np.exp(12 * (t - 0.45)))
    sp = 1 / (1 + np.exp(-12 * (t - 0.55)))
    J = se + sp - 1
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.plot(t, se, color=TEAL, lw=2.2, label="sensitivity")
    ax.plot(t, sp, color=GOLD, lw=2.2, label="specificity")
    ax.plot(t, J, color=DEEP, lw=2.5, label="Youden J=Se+Sp−1")
    tstar = t[J.argmax()]
    ax.axvline(tstar, color=GRAY, ls="--", lw=1.2)
    ax.set_xlabel("threshold")
    ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, rf"Youden peak near t*≈{tstar:.2f}")
    ax.text(0.98, 0.1, "Youden ignores prevalence & costs.\nNot always the clinical threshold.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_youden_j.png")


def fig_preface_leakage_icons():
    """Preface three leakage modes icons."""
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    items = [
        (1.5, "Timing\nleakage", "features after t₀"),
        (5.0, "Fit\nleakage", "scaler on all data"),
        (8.5, "Label\nleakage", "proxy = outcome"),
    ]
    for x, title, sub in items:
        ax.add_patch(FancyBboxPatch((x, 1.2), 2.8, 2.6, boxstyle="round,pad=0.05",
                                    facecolor=SOFT, edgecolor=TEAL, linewidth=2))
        ax.text(x + 1.4, 2.9, title, ha="center", va="center", fontsize=12, fontweight="bold", color=DEEP)
        ax.text(x + 1.4, 1.8, sub, ha="center", va="center", fontsize=9, color=SLATE)
    ax.text(6, 4.5, "Three leakage modes to hunt in every paper", ha="center", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ml_fig_three_leakage_modes.png")


def main():
    fig_under_over_fit_curves()
    fig_small_multiples()
    fig_hdbscan_stability()
    fig_tfidf_idf_curve()
    fig_nmf_parts_add()
    fig_residual_qq()
    fig_residual_connection()
    fig_simclr_pipeline()
    fig_kv_cache()
    fig_model_card_gates()
    fig_glossary_youden()
    fig_preface_leakage_icons()
    print("DONE cycle-19")


if __name__ == "__main__":
    main()
