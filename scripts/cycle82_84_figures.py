#!/usr/bin/env python3
"""Cycle-82/83/84: high-value novel scientific teal teaching figures for ML densify.

Prefer novel pedagogy over pure volume. Each panel teaches a distinct concept.
pred != cause throughout captions when embedded.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Arc, Wedge, Ellipse, Polygon
from matplotlib.collections import LineCollection

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#0d9488"
DEEP = "#0f766e"
INK = "#0f172a"
GOLD = "#c9a227"
SOFT = "#ecfeff"
SLATE = "#64748b"
ROSE = "#e11d48"
MINT = "#14b8a6"


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def style(ax, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def box(ax, x, y, w, h, text, fc=TEAL, ec="none", fs=9, tc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", wrap=True)


# ───────────────────────── Cycle 82: novel concepts ─────────────────────────

def c82_00_quadratic_bowl():
    """Math foundations: gradient on a quadratic loss bowl + contour."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    x = np.linspace(-2.5, 2.5, 200)
    for a, c in [(0.3, SLATE), (0.8, TEAL), (1.6, DEEP)]:
        ax1.plot(x, a * x ** 2, color=c, lw=2, label=f"a={a}")
    xs = np.array([2.0, 1.2, 0.5, 0.1])
    ax1.plot(xs, 0.8 * xs ** 2, "o-", color=GOLD, lw=2, ms=7, label="GD path")
    for i in range(len(xs) - 1):
        ax1.annotate("", xy=(xs[i + 1], 0.8 * xs[i + 1] ** 2),
                     xytext=(xs[i], 0.8 * xs[i] ** 2),
                     arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax1, "1D quadratic loss L(θ)=aθ²")
    ax1.set_xlabel("θ"); ax1.set_ylabel("L"); ax1.legend(fontsize=8, loc="upper center")
    ax1.grid(True, alpha=0.3)

    xx, yy = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(-2, 2, 80))
    zz = 1.2 * xx ** 2 + 0.4 * yy ** 2
    ax2.contour(xx, yy, zz, levels=12, colors=TEAL, linewidths=1.1)
    path = np.array([[1.8, 1.5], [1.1, 0.9], [0.55, 0.4], [0.15, 0.08], [0.02, 0.0]])
    ax2.plot(path[:, 0], path[:, 1], "o-", color=GOLD, lw=2, ms=6)
    ax2.plot(0, 0, "*", color=ROSE, ms=14)
    style(ax2, "2D contours + GD steps")
    ax2.set_xlabel("θ₁"); ax2.set_ylabel("θ₂"); ax2.set_aspect("equal")
    fig.suptitle("Gradient descent on a convex quadratic (math foundations)",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_00.png")


def c82_01_study_loop():
    """Preface: reproducible ML study loop."""
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    steps = [
        (0.4, 3.2, "Question &\nprotocol"),
        (3.5, 3.2, "Data &\nindex time"),
        (6.6, 3.2, "Model &\nvalidation"),
        (2.0, 0.7, "Report &\nlimits"),
        (5.0, 0.7, "External\ncheck"),
    ]
    cols = [TEAL, DEEP, GOLD, SLATE, MINT]
    for (x, y, t), c in zip(steps, cols):
        box(ax, x, y, 2.6, 1.4, t, fc=c, fs=10)
    arrows = [((3.0, 3.9), (3.5, 3.9)), ((6.1, 3.9), (6.6, 3.9)),
              ((7.9, 3.2), (6.3, 2.1)), ((5.0, 1.4), (3.0, 2.1)),
              ((2.0, 2.1), (1.7, 3.2))]
    for (a, b) in arrows:
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(5, 2.55, "pred ≠ cause", ha="center", fontsize=10,
            color=ROSE, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#fff1f2", edgecolor=ROSE))
    style(ax, "Reproducible ML study loop (preface)")
    return save(fig, "ml_fig_c82_01.png")


def c82_02_inductive_bias():
    """Basic concepts: inductive bias spectrum."""
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.plot([0.5, 9.5], [2.0, 2.0], color=SLATE, lw=3)
    points = [
        (1.2, "k-NN\nlocal"),
        (3.0, "linear\nmodels"),
        (5.0, "trees &\nensembles"),
        (7.0, "deep nets\nhierarchy"),
        (8.8, "very large\nmodels"),
    ]
    for x, lab in points:
        ax.plot(x, 2.0, "o", color=TEAL, ms=14)
        ax.text(x, 2.7, lab, ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.text(1.2, 0.9, "strong local\nassumptions", ha="center", fontsize=8, color=DEEP)
    ax.text(8.8, 0.9, "flexible\nfunction class", ha="center", fontsize=8, color=GOLD)
    ax.annotate("", xy=(9.2, 1.5), xytext=(0.8, 1.5),
                arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.5))
    ax.text(5, 1.2, "capacity / flexibility →", ha="center", fontsize=9, color=SLATE)
    style(ax, "Inductive bias spectrum (what the model assumes)")
    return save(fig, "ml_fig_c82_02.png")


def c82_03_anscombe_lesson():
    """Visualization: same stats, different geometry (Anscombe lesson)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0))
    rng = np.random.default_rng(7)
    # Set 1 linear
    x1 = np.linspace(4, 14, 11)
    y1 = 3 + 0.5 * x1 + rng.normal(0, 0.6, 11)
    # Set 2 curve
    x2 = x1.copy()
    y2 = 0.1 * (x2 - 9) ** 2 + 4 + rng.normal(0, 0.15, 11)
    # Set 3 outlier slope
    x3 = x1.copy()
    y3 = 3 + 0.5 * x3
    y3[10] = 12.5
    # Set 4 vertical cluster
    x4 = np.full(11, 8.0); x4[10] = 19
    y4 = 3 + 0.5 * np.linspace(5, 12, 11); y4[10] = 12.5
    sets = [(x1, y1, "linear-ish"), (x2, y2, "curved"),
            (x3, y3, "outlier slope"), (x4, y4, "x-outlier")]
    for ax, (x, y, title) in zip(axes.ravel(), sets):
        ax.scatter(x, y, c=TEAL, s=40, zorder=3)
        coef = np.polyfit(x, y, 1)
        xr = np.linspace(x.min() - 1, x.max() + 1, 50)
        ax.plot(xr, coef[0] * xr + coef[1], color=GOLD, lw=2)
        style(ax, title)
        ax.set_xlim(3, 20); ax.set_ylim(2, 14)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Same mean/var/corr can hide different structure — always plot",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_03.png")


def c82_04_likelihood_posterior():
    """Probability: likelihood vs posterior with prior."""
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    th = np.linspace(0, 1, 400)
    # Beta prior
    prior = th ** 1.5 * (1 - th) ** 3.5
    prior /= np.trapezoid(prior, th)
    # Binomial-like likelihood peak near 0.7
    like = th ** 12 * (1 - th) ** 5
    like /= np.trapezoid(like, th)
    post = prior * like
    post /= np.trapezoid(post, th)
    ax.plot(th, prior, color=SLATE, lw=2, label="prior p(θ)")
    ax.plot(th, like, color=GOLD, lw=2, label="likelihood p(D|θ)")
    ax.plot(th, post, color=TEAL, lw=2.5, label="posterior p(θ|D)")
    ax.fill_between(th, post, alpha=0.15, color=TEAL)
    ax.axvline(th[np.argmax(post)], color=DEEP, ls="--", lw=1.2, label="MAP")
    style(ax, "Prior × likelihood → posterior (Bayes update)")
    ax.set_xlabel("θ"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_04.png")


def c82_05_silhouette_elbow():
    """Clustering: elbow + silhouette dual panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    k = np.arange(1, 11)
    inertia = 120 * np.exp(-0.45 * (k - 1)) + 18 + np.array([0, 0, -2, 1, 0, 0.5, 0, 0, 0.2, 0])
    ax1.plot(k, inertia, "o-", color=TEAL, lw=2, ms=7)
    ax1.axvline(3, color=GOLD, ls="--", lw=1.5, label="elbow ~ k=3")
    style(ax1, "Within-cluster sum of squares")
    ax1.set_xlabel("k"); ax1.set_ylabel("inertia"); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    sil = np.array([0.0, 0.42, 0.61, 0.55, 0.48, 0.41, 0.35, 0.30, 0.27, 0.24])
    ax2.bar(k, sil, color=TEAL, edgecolor=DEEP, alpha=0.85)
    ax2.axhline(0.5, color=GOLD, ls="--", lw=1.2, label="rough quality line")
    style(ax2, "Mean silhouette by k")
    ax2.set_xlabel("k"); ax2.set_ylabel("silhouette"); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 0.8)
    fig.suptitle("Choosing k: elbow is a heuristic; silhouette checks separation",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_05.png")


def c82_06_support_conf_lift():
    """Mining: support–confidence–lift relationship."""
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.5, 4.0, 2.8, 1.4, "support\nP(A∩B)", fc=TEAL)
    box(ax, 3.6, 4.0, 2.8, 1.4, "confidence\nP(B|A)", fc=DEEP)
    box(ax, 6.7, 4.0, 2.8, 1.4, "lift\nP(B|A)/P(B)", fc=GOLD, tc=INK)
    box(ax, 2.0, 1.5, 6.0, 1.6,
        "High confidence can still be uninteresting\nif B is already common → check lift",
        fc=SOFT, tc=INK, fs=10)
    ax.annotate("", xy=(3.6, 4.7), xytext=(3.3, 4.7),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(6.7, 4.7), xytext=(6.4, 4.7),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(5, 0.6, "Association rules: filter on support, rank by lift, do not claim cause",
            ha="center", fontsize=9, color=ROSE, fontweight="bold")
    style(ax, "Support · confidence · lift (itemset mining)")
    return save(fig, "ml_fig_c82_06.png")


def c82_07_encoding_leakage():
    """Feature engineering: target encoding leakage risk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 6); ax1.axis("off")
    box(ax1, 0.5, 3.5, 4.0, 1.8, "WRONG\nencode on full Y\nthen split", fc=ROSE, fs=10)
    box(ax1, 5.2, 3.5, 4.3, 1.8, "RIGHT\nfit encoder\ninside each fold", fc=TEAL, fs=10)
    box(ax1, 1.5, 0.6, 7.0, 1.8, "Target mean encoding leaks label\ninformation into features",
        fc=SOFT, tc=INK, fs=10)
    style(ax1, "Target encoding protocol")

    cats = ["A", "B", "C", "D"]
    full = [0.62, 0.41, 0.55, 0.28]
    fold = [0.50, 0.45, 0.48, 0.35]
    x = np.arange(len(cats))
    ax2.bar(x - 0.18, full, 0.35, color=ROSE, label="fit on all Y", alpha=0.85)
    ax2.bar(x + 0.18, fold, 0.35, color=TEAL, label="fold-local", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(cats)
    style(ax2, "Encoded values can shift if Y leaks")
    ax2.set_ylabel("category → numeric"); ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Feature engineering: nest target statistics inside CV",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_07.png")


def c82_08_pca_scree():
    """Dimensionality: eigenvalue scree + cumulative variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ev = np.array([4.8, 2.1, 0.9, 0.45, 0.25, 0.15, 0.1, 0.08, 0.05, 0.02])
    ev = ev / ev.sum()
    cum = np.cumsum(ev)
    idx = np.arange(1, len(ev) + 1)
    ax1.bar(idx, ev, color=TEAL, edgecolor=DEEP)
    ax1.plot(idx, ev, "o-", color=GOLD, lw=1.5, ms=5)
    style(ax1, "Scree (fraction variance)")
    ax1.set_xlabel("component"); ax1.set_ylabel("prop. variance")
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.plot(idx, cum, "o-", color=TEAL, lw=2, ms=6)
    ax2.axhline(0.9, color=GOLD, ls="--", label="90% variance")
    ax2.axvline(idx[np.searchsorted(cum, 0.9)], color=ROSE, ls=":", lw=1.5)
    style(ax2, "Cumulative explained variance")
    ax2.set_xlabel("k components"); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.suptitle("PCA: how many components? use scree + cumulative variance",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_08.png")


def c82_09_residual_diagnostics():
    """Regression: residual fan + ideal band."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    rng = np.random.default_rng(3)
    fitted = np.linspace(0, 10, 80)
    res_good = rng.normal(0, 0.8, 80)
    res_het = rng.normal(0, 0.3 + 0.15 * fitted, 80)
    ax1.axhline(0, color=SLATE, lw=1)
    ax1.scatter(fitted, res_good, c=TEAL, s=22, alpha=0.75)
    style(ax1, "OK: mean-zero, constant spread")
    ax1.set_xlabel("fitted"); ax1.set_ylabel("residual")
    ax1.grid(True, alpha=0.3)
    ax2.axhline(0, color=SLATE, lw=1)
    ax2.scatter(fitted, res_het, c=GOLD, s=22, alpha=0.75)
    ax2.plot(fitted, 0.3 + 0.15 * fitted, color=ROSE, ls="--", lw=1.5)
    ax2.plot(fitted, -(0.3 + 0.15 * fitted), color=ROSE, ls="--", lw=1.5)
    style(ax2, "Problem: heteroscedastic fan")
    ax2.set_xlabel("fitted"); ax2.set_ylabel("residual")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Regression diagnostics: residual plots before claiming fit quality",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_09.png")


def c82_10_threshold_cost():
    """Classification: threshold vs FP/FN cost."""
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    t = np.linspace(0.05, 0.95, 100)
    # synthetic FP and FN rates vs threshold
    fp = 0.55 * (1 - t) ** 1.4
    fn = 0.45 * t ** 1.6
    cost_bal = 1 * fp + 1 * fn
    cost_fn = 1 * fp + 5 * fn
    ax.plot(t, cost_bal, color=TEAL, lw=2.2, label="cost = FP + FN")
    ax.plot(t, cost_fn, color=GOLD, lw=2.2, label="cost = FP + 5·FN")
    i1, i2 = np.argmin(cost_bal), np.argmin(cost_fn)
    ax.axvline(t[i1], color=TEAL, ls="--", alpha=0.7)
    ax.axvline(t[i2], color=GOLD, ls="--", alpha=0.7)
    ax.plot(t[i1], cost_bal[i1], "o", color=TEAL, ms=9)
    ax.plot(t[i2], cost_fn[i2], "o", color=GOLD, ms=9)
    style(ax, "Optimal threshold shifts with clinical cost weights")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("expected cost (synthetic)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_10.png")


def c82_11_vanishing_grad():
    """NN: signal decay with depth for saturating activations."""
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    depth = np.arange(1, 16)
    sigmoid_g = 0.25 ** depth * 5
    relu_g = np.full_like(depth, 1.0, dtype=float) * 0.9
    relu_g = 0.9 ** (0.15 * depth)
    residual = np.ones_like(depth, dtype=float) * 0.95
    ax.semilogy(depth, sigmoid_g, "o-", color=ROSE, lw=2, label="deep tanh/sigmoid path")
    ax.semilogy(depth, relu_g, "s-", color=TEAL, lw=2, label="ReLU-like path")
    ax.semilogy(depth, residual, "^-", color=GOLD, lw=2, label="residual shortcut")
    style(ax, "Gradient magnitude vs depth (schematic)")
    ax.set_xlabel("layer depth"); ax.set_ylabel("|∂L/∂h| (log scale)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_11.png")


def c82_12_contrastive_pairs():
    """SSL: contrastive positive/negative geometry."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    circ = Circle((0, 0), 1.0, fill=False, edgecolor=SLATE, lw=1.5, ls="--")
    ax.add_patch(circ)
    # anchor
    a = np.array([0.7, 0.4]); a = a / np.linalg.norm(a)
    p = np.array([0.65, 0.55]); p = p / np.linalg.norm(p)
    negs = np.array([[-0.8, 0.3], [0.2, -0.9], [-0.5, -0.7], [0.9, -0.2]])
    negs = negs / np.linalg.norm(negs, axis=1, keepdims=True)
    ax.scatter(*a, s=160, c=TEAL, zorder=5, label="anchor")
    ax.scatter(*p, s=160, c=GOLD, zorder=5, label="positive (augment)")
    ax.scatter(negs[:, 0], negs[:, 1], s=100, c=ROSE, zorder=5, label="negatives")
    ax.plot([0, a[0]], [0, a[1]], color=TEAL, lw=1.5)
    ax.plot([0, p[0]], [0, p[1]], color=GOLD, lw=1.5)
    for n in negs:
        ax.plot([0, n[0]], [0, n[1]], color=ROSE, lw=0.8, alpha=0.6)
    # arc for small angle
    ax.annotate("", xy=p, xytext=a,
                arrowprops=dict(arrowstyle="<->", color=DEEP, lw=1.5))
    ax.text(0.85, 0.75, "pull", color=DEEP, fontsize=9, fontweight="bold")
    ax.text(-0.95, -0.2, "push", color=ROSE, fontsize=9, fontweight="bold")
    style(ax, "Contrastive SSL: pull positives, push negatives on the unit sphere")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    fig.tight_layout()
    return save(fig, "ml_fig_c82_12.png")


def c82_13_early_late_fusion():
    """Multimodal: early vs late fusion."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax1, 0.4, 4.2, 2.2, 1.2, "text", fc=TEAL, fs=10)
    box(ax1, 0.4, 2.4, 2.2, 1.2, "image", fc=DEEP, fs=10)
    box(ax1, 0.4, 0.6, 2.2, 1.2, "audio", fc=GOLD, tc=INK, fs=10)
    box(ax1, 4.0, 2.2, 2.6, 1.6, "concat\nearly", fc=SLATE, fs=10)
    box(ax1, 7.4, 2.2, 2.2, 1.6, "joint\nmodel", fc=MINT, tc=INK, fs=10)
    for y in (4.8, 3.0, 1.2):
        ax1.annotate("", xy=(4.0, 3.0), xytext=(2.6, y),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax1.annotate("", xy=(7.4, 3.0), xytext=(6.6, 3.0),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    style(ax1, "Early fusion")

    box(ax2, 0.4, 4.2, 2.2, 1.2, "text→h_t", fc=TEAL, fs=9)
    box(ax2, 0.4, 2.4, 2.2, 1.2, "img→h_i", fc=DEEP, fs=9)
    box(ax2, 0.4, 0.6, 2.2, 1.2, "aud→h_a", fc=GOLD, tc=INK, fs=9)
    box(ax2, 4.0, 2.2, 2.6, 1.6, "score\nfusion", fc=SLATE, fs=10)
    box(ax2, 7.4, 2.2, 2.2, 1.6, "decision", fc=MINT, tc=INK, fs=10)
    for y in (4.8, 3.0, 1.2):
        ax2.annotate("", xy=(4.0, 3.0), xytext=(2.6, y),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax2.annotate("", xy=(7.4, 3.0), xytext=(6.6, 3.0),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    style(ax2, "Late fusion")
    fig.suptitle("Multimodal models: where modalities meet matters",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_13.png")


def c82_14_bandit_regret():
    """RL: cumulative regret of explore vs exploit."""
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    t = np.arange(1, 201)
    greedy = 0.35 * t  # linear high regret
    eps = 0.12 * t + 8 * np.sqrt(t)
    ucb = 6 * np.sqrt(t * np.log(t + 1))
    ax.plot(t, greedy, color=ROSE, lw=2, label="pure greedy (lock-in)")
    ax.plot(t, eps, color=GOLD, lw=2, label="ε-greedy")
    ax.plot(t, ucb, color=TEAL, lw=2.2, label="UCB-style (schematic)")
    style(ax, "Cumulative regret vs time (multi-armed bandit sketch)")
    ax.set_xlabel("round t"); ax.set_ylabel("cumulative regret")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_14.png")


def c82_15_prune_distill():
    """Compression: magnitude prune + distill sketch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    w = np.sort(np.abs(np.random.default_rng(1).normal(0, 1, 40)))[::-1]
    ax1.bar(np.arange(len(w)), w, color=TEAL, edgecolor=DEEP)
    ax1.axhline(0.55, color=GOLD, ls="--", lw=1.5, label="prune threshold")
    style(ax1, "Magnitude pruning")
    ax1.set_xlabel("weight rank"); ax1.set_ylabel("|w|"); ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.set_xlim(0, 10); ax2.set_ylim(0, 5); ax2.axis("off")
    box(ax2, 0.5, 2.5, 3.2, 1.8, "Teacher\nlarge, soft p", fc=TEAL, fs=10)
    box(ax2, 6.0, 2.5, 3.2, 1.8, "Student\nsmall, match p", fc=GOLD, tc=INK, fs=10)
    ax2.annotate("", xy=(6.0, 3.4), xytext=(3.7, 3.4),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=2))
    ax2.text(5, 1.5, "KL(soft_T || soft_S) + hard CE", ha="center",
             fontsize=9, color=DEEP, fontweight="bold")
    style(ax2, "Knowledge distillation")
    fig.suptitle("Making models lighter: prune sparse weights, distill soft labels",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_15.png")


def c82_16_random_walk_mp():
    """Graphs: random walk vs message passing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    # small graph coords
    pos = {0: (0.2, 0.5), 1: (0.5, 0.85), 2: (0.8, 0.5), 3: (0.5, 0.15), 4: (0.5, 0.5)}
    edges = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
    for ax, title in ((ax1, "Random walk trajectory"), (ax2, "One GNN message-pass")):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        for i, j in edges:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=SLATE, lw=1.5)
        for n, (x, y) in pos.items():
            ax.plot(x, y, "o", color=TEAL, ms=14)
            ax.text(x, y - 0.08, str(n), ha="center", fontsize=8, color=INK)
        style(ax, title)
        ax.axis("off")
    # walk path
    walk = [0, 4, 2, 3, 4, 1]
    for a, b in zip(walk, walk[1:]):
        ax1.annotate("", xy=pos[b], xytext=pos[a],
                     arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    # messages into center
    for n in (0, 1, 2, 3):
        ax2.annotate("", xy=pos[4], xytext=pos[n],
                     arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8,
                                     connectionstyle="arc3,rad=0.1"))
    ax2.plot(*pos[4], "o", color=ROSE, ms=16, zorder=5)
    fig.suptitle("Graph mining: walks explore paths; MP aggregates neighborhoods",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_16.png")


def c82_17_missingness():
    """Data challenges: MCAR / MAR / MNAR."""
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.4))
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 120)
    y = 0.6 * x + rng.normal(0, 0.7, 120)
    titles = ["MCAR: miss ⊥ data", "MAR: miss | observed", "MNAR: miss | missing"]
    for ax, title, mode in zip(axes, titles, ["mcar", "mar", "mnar"]):
        m = np.zeros(120, dtype=bool)
        if mode == "mcar":
            m = rng.random(120) < 0.25
        elif mode == "mar":
            m = (x > 0.5) & (rng.random(120) < 0.55)
        else:
            m = (y > 0.8) & (rng.random(120) < 0.7)
        ax.scatter(x[~m], y[~m], c=TEAL, s=18, alpha=0.8, label="observed")
        ax.scatter(x[m], y[m], c=ROSE, s=18, alpha=0.55, marker="x", label="missing Y")
        style(ax, title)
        ax.grid(True, alpha=0.25)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Missingness mechanism changes what imputation can recover",
                 fontsize=11, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    return save(fig, "ml_fig_c82_17.png")


def c82_18_appraisal_diamond():
    """Closing: senior appraisal diamond."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    verts = [(5, 9), (9, 5), (5, 1), (1, 5)]
    ax.add_patch(Polygon(verts, closed=True, facecolor=SOFT, edgecolor=TEAL, lw=2.5))
    labels = [(5, 8.3, "Question &\ncohort"), (7.8, 5, "Features &\nleakage"),
              (5, 1.7, "Validation &\ncalibration"), (2.2, 5, "Utility &\nlimits")]
    for x, y, t in labels:
        ax.text(x, y, t, ha="center", va="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.text(5, 5, "APPRAISE\nbefore deploy", ha="center", va="center",
            fontsize=11, color=DEEP, fontweight="bold")
    style(ax, "Senior practice: four-corner model appraisal")
    return save(fig, "ml_fig_c82_18.png")


def c82_19_glossary_map():
    """Glossary: hub concept map."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")
    hub = Circle((0, 0), 0.28, facecolor=TEAL, edgecolor=DEEP, lw=2)
    ax.add_patch(hub)
    ax.text(0, 0, "ML\ncore", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    nodes = [
        (0, 0.85, "loss"), (0.75, 0.45, "generalization"),
        (0.75, -0.45, "validation"), (0, -0.85, "calibration"),
        (-0.75, -0.45, "bias"), (-0.75, 0.45, "capacity"),
    ]
    for x, y, lab in nodes:
        ax.plot([0, x * 0.65], [0, y * 0.65], color=SLATE, lw=1.5)
        ax.add_patch(Circle((x, y), 0.22, facecolor=SOFT, edgecolor=TEAL, lw=1.5))
        ax.text(x, y, lab, ha="center", va="center", fontsize=8,
                color=INK, fontweight="bold")
    style(ax, "Glossary map: terms orbit the same decision core")
    return save(fig, "ml_fig_c82_19.png")


# ───────────────────────── Cycle 83 ─────────────────────────

def c83_00_svd_blocks():
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.3, 1.0, 2.0, 2.2, "A\nm×n", fc=TEAL, fs=11)
    ax.text(2.6, 2.0, "≈", fontsize=18, ha="center", va="center", color=INK)
    box(ax, 3.0, 1.0, 1.8, 2.2, "U\nm×k", fc=DEEP, fs=11)
    box(ax, 5.2, 1.4, 1.8, 1.4, "Σ\nk×k", fc=GOLD, tc=INK, fs=11)
    box(ax, 7.4, 1.0, 1.8, 2.2, "Vᵀ\nk×n", fc=MINT, tc=INK, fs=11)
    ax.text(10.2, 2.0, "rank-k\napprox", ha="center", va="center",
            fontsize=10, color=SLATE, fontweight="bold")
    style(ax, "SVD / low-rank factorization building blocks")
    return save(fig, "ml_fig_c83_00.png")


def c83_01_temporal_split():
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    segs = [(0.3, 6.5, "TRAIN (past)", TEAL),
            (7.0, 2.2, "VAL", GOLD),
            (9.4, 2.3, "TEST (future)", ROSE)]
    for x, w, lab, c in segs:
        box(ax, x, 1.0, w, 1.2, lab, fc=c, fs=10, tc="white" if c != GOLD else INK)
    ax.annotate("", xy=(11.5, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(6, 0.35, "time →  (no shuffle across time for deployment claims)",
            ha="center", fontsize=9, color=SLATE)
    style(ax, "Temporal train / validation / test split")
    return save(fig, "ml_fig_c83_01.png")


def c83_02_bias_variance():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    c = np.linspace(0.2, 5, 100)
    bias2 = 2.5 / c
    var = 0.15 * c
    noise = np.full_like(c, 0.4)
    total = bias2 + var + noise
    ax.plot(c, bias2, color=TEAL, lw=2, label="bias²")
    ax.plot(c, var, color=GOLD, lw=2, label="variance")
    ax.plot(c, noise, color=SLATE, lw=1.5, ls="--", label="irreducible")
    ax.plot(c, total, color=ROSE, lw=2.5, label="expected error")
    ax.axvline(c[np.argmin(total)], color=DEEP, ls=":", lw=1.5)
    style(ax, "Bias–variance tradeoff vs model capacity")
    ax.set_xlabel("capacity →"); ax.set_ylabel("error")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_02.png")


def c83_03_parallel_coords():
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    rng = np.random.default_rng(2)
    dims = 5
    for _ in range(40):
        y = rng.normal(0, 1, dims)
        y = (y - y.min()) / (y.max() - y.min() + 1e-9)
        ax.plot(range(dims), y, color=TEAL, alpha=0.25, lw=1)
    # highlight two clusters
    for mu, c in [(0.75, GOLD), (0.25, ROSE)]:
        for _ in range(8):
            y = rng.normal(mu, 0.08, dims)
            y = np.clip(y, 0, 1)
            ax.plot(range(dims), y, color=c, alpha=0.7, lw=1.5)
    style(ax, "Parallel coordinates: multi-feature pattern scanning")
    ax.set_xticks(range(dims)); ax.set_xticklabels([f"f{i+1}" for i in range(dims)])
    ax.set_ylabel("scaled value"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_03.png")


def c83_04_clt():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    rng = np.random.default_rng(5)
    raw = rng.exponential(1.5, 2000)
    ax1.hist(raw, bins=35, color=TEAL, edgecolor=DEEP, alpha=0.85, density=True)
    style(ax1, "Skewed raw population")
    ax1.set_xlabel("x"); ax1.grid(True, alpha=0.3)
    means = [raw[i * 30:(i + 1) * 30].mean() for i in range(60)]
    ax2.hist(means, bins=18, color=GOLD, edgecolor=DEEP, alpha=0.9, density=True)
    style(ax2, "Sampling distribution of mean (n=30)")
    ax2.set_xlabel("x̄"); ax2.grid(True, alpha=0.3)
    fig.suptitle("CLT: means of samples become approximately normal",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_04.png")


def c83_05_dbscan_cores():
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    rng = np.random.default_rng(4)
    c1 = rng.normal([-1, 0], 0.25, size=(60, 2))
    c2 = rng.normal([1.2, 0.2], 0.3, size=(70, 2))
    noise = rng.uniform(-2.5, 2.5, size=(15, 2))
    ax.scatter(c1[:, 0], c1[:, 1], c=TEAL, s=28, label="dense region A")
    ax.scatter(c2[:, 0], c2[:, 1], c=GOLD, s=28, label="dense region B")
    ax.scatter(noise[:, 0], noise[:, 1], c=ROSE, s=36, marker="x", label="noise/outlier")
    for center, col in [([-1, 0], TEAL), ([1.2, 0.2], GOLD)]:
        ax.add_patch(Circle(center, 0.55, fill=False, ls="--", edgecolor=col, lw=1.5))
    style(ax, "DBSCAN idea: density cores + border + noise")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_05.png")


def c83_06_apriori_lattice():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    levels = [
        [(5, 5.2, "{}")],
        [(2, 3.8, "{A}"), (5, 3.8, "{B}"), (8, 3.8, "{C}")],
        [(2, 2.2, "{A,B}"), (5, 2.2, "{A,C}"), (8, 2.2, "{B,C}")],
        [(5, 0.6, "{A,B,C}")],
    ]
    for lev in levels:
        for x, y, t in lev:
            box(ax, x - 0.9, y - 0.35, 1.8, 0.7, t, fc=TEAL if len(t) < 6 else DEEP, fs=9)
    for x in (2, 5, 8):
        ax.annotate("", xy=(x, 4.15), xytext=(5, 5.0),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1))
    ax.annotate("", xy=(2, 2.55), xytext=(2, 3.45),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1))
    ax.annotate("", xy=(5, 2.55), xytext=(2, 3.45),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1))
    ax.annotate("", xy=(5, 0.95), xytext=(5, 1.85),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    ax.text(9.3, 3.0, "prune\ninfrequent\nparents", ha="center", fontsize=8,
            color=ROSE, fontweight="bold")
    style(ax, "Apriori itemset lattice (bottom-up with support pruning)")
    return save(fig, "ml_fig_c83_06.png")


def c83_07_interaction_heat():
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    rng = np.random.default_rng(9)
    M = rng.normal(0, 1, (6, 6))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0)
    M[1, 3] = M[3, 1] = 2.4
    M[0, 2] = M[2, 0] = -1.8
    im = ax.imshow(M, cmap="coolwarm", vmin=-2.5, vmax=2.5)
    labs = [f"f{i+1}" for i in range(6)]
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(labs); ax.set_yticklabels(labs)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style(ax, "Feature interaction strength (schematic heat)")
    fig.tight_layout()
    return save(fig, "ml_fig_c83_07.png")


def c83_08_perplexity():
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2))
    rng = np.random.default_rng(11)
    for ax, perp, title in zip(axes, [2, 15, 60],
                               ["low perplexity", "mid", "high perplexity"]):
        # fake embeddings that "blur" more with higher perp
        a = rng.normal([-0.8, 0], 0.08 + perp / 400, (40, 2))
        b = rng.normal([0.8, 0], 0.08 + perp / 400, (40, 2))
        if perp > 40:
            a[:, 0] += 0.35; b[:, 0] -= 0.35
        ax.scatter(a[:, 0], a[:, 1], c=TEAL, s=16, alpha=0.8)
        ax.scatter(b[:, 0], b[:, 1], c=GOLD, s=16, alpha=0.8)
        style(ax, title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("t-SNE/UMAP perplexity/neighbors change topology emphasis",
                 fontsize=11, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_08.png")


def c83_09_regularization_paths():
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    lam = np.logspace(-2, 2, 50)
    # schematic ridge shrinks together; lasso hits zero
    for coef0, c in zip([2.0, -1.4, 0.9, 0.4], [TEAL, DEEP, GOLD, SLATE]):
        ridge = coef0 / (1 + lam)
        ax.plot(lam, ridge, color=c, lw=1.8, alpha=0.9)
    ax.set_xscale("log")
    style(ax, "Coefficient paths vs λ (ridge-like shrink; lasso would hit 0)")
    ax.set_xlabel("λ (log)"); ax.set_ylabel("β̂_j")
    ax.axhline(0, color=INK, lw=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_09.png")


def c83_10_reliability():
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    p = np.linspace(0.05, 0.95, 10)
    obs_bad = 0.15 + 0.55 * p
    obs_good = p + 0.03 * np.sin(2 * np.pi * p)
    ax.plot([0, 1], [0, 1], "--", color=SLATE, label="perfect")
    ax.plot(p, obs_bad, "o-", color=ROSE, lw=2, label="overconfident")
    ax.plot(p, obs_good, "s-", color=TEAL, lw=2, label="calibrated")
    style(ax, "Reliability diagram (calibration)")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_10.png")


def c83_11_attention_softmax():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    scores = np.array([2.1, 0.4, -0.5, 1.2, 0.1])
    ax1.bar(range(len(scores)), scores, color=GOLD, edgecolor=DEEP)
    style(ax1, "Raw attention scores")
    ax1.set_xlabel("key position"); ax1.set_ylabel("score")
    ex = np.exp(scores - scores.max())
    w = ex / ex.sum()
    ax2.bar(range(len(w)), w, color=TEAL, edgecolor=DEEP)
    style(ax2, "Softmax weights (sum=1)")
    ax2.set_xlabel("key position"); ax2.set_ylabel("α")
    ax2.set_ylim(0, 1)
    fig.suptitle("Attention: scores → softmax → weighted values",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_11.png")


def c83_12_mlm_mask():
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    toks = ["The", "MRI", "[MASK]", "shows", "DWI", "lesion"]
    cols = [TEAL, TEAL, ROSE, TEAL, TEAL, TEAL]
    for i, (t, c) in enumerate(zip(toks, cols)):
        box(ax, 0.4 + i * 1.9, 1.0, 1.7, 1.2, t, fc=c, fs=10)
    ax.text(6, 0.4, "MLM: predict masked token from bidirectional context",
            ha="center", fontsize=10, color=DEEP, fontweight="bold")
    style(ax, "Masked language modeling (self-supervised text)")
    return save(fig, "ml_fig_c83_12.png")


def c83_13_receptive_field():
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    widths = [0.4, 0.9, 1.6, 2.6]
    labels = ["pixel", "3×3", "stacked\n3×3", "deeper\nstack"]
    for i, (w, lab) in enumerate(zip(widths, labels)):
        x = 1.5 + i * 2.6
        ax.add_patch(Rectangle((x - w / 2, 1.5 - w / 2), w, w,
                               facecolor=TEAL, alpha=0.25 + 0.15 * i,
                               edgecolor=DEEP, lw=1.5))
        ax.text(x, 3.3, lab, ha="center", fontsize=9, color=INK, fontweight="bold")
    style(ax, "CNN receptive field grows with stacked local filters")
    return save(fig, "ml_fig_c83_13.png")


def c83_14_td_backup():
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.5, 1.3, 2.4, 1.4, "V(s_t)", fc=TEAL, fs=11)
    box(ax, 4.0, 1.3, 3.2, 1.4, "r_t + γ V(s_{t+1})", fc=GOLD, tc=INK, fs=10)
    box(ax, 8.5, 1.3, 3.0, 1.4, "TD error δ", fc=ROSE, fs=11)
    ax.annotate("", xy=(4.0, 2.0), xytext=(2.9, 2.0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(8.5, 2.0), xytext=(7.2, 2.0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(6, 0.5, "V ← V + α · δ   (bootstrap backup)",
            ha="center", fontsize=10, color=DEEP, fontweight="bold")
    style(ax, "Temporal-difference learning backup")
    return save(fig, "ml_fig_c83_14.png")


def c83_15_teacher_student_temp():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    logits = np.array([2.5, 1.0, 0.2, -0.5])
    for T, c, ls in [(1, TEAL, "-"), (3, GOLD, "--"), (8, SLATE, ":")]:
        e = np.exp(logits / T); p = e / e.sum()
        ax.plot(range(4), p, ls, marker="o", color=c, lw=2, label=f"T={T}")
    style(ax, "Softmax temperature softens teacher labels")
    ax.set_xlabel("class"); ax.set_ylabel("probability")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_15.png")


def c83_16_pagerank():
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    pos = {0: (0.2, 0.7), 1: (0.8, 0.75), 2: (0.75, 0.25), 3: (0.25, 0.2), 4: (0.5, 0.5)}
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (4, 2)]
    pr = {0: 0.15, 1: 0.22, 2: 0.28, 3: 0.12, 4: 0.35}
    for i, j in edges:
        ax.annotate("", xy=pos[j], xytext=pos[i],
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5,
                                    connectionstyle="arc3,rad=0.08"))
    for n, (x, y) in pos.items():
        r = 0.06 + 0.12 * pr[n]
        ax.add_patch(Circle((x, y), r, facecolor=TEAL, edgecolor=DEEP, lw=1.5, alpha=0.9))
        ax.text(x, y, f"{pr[n]:.2f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    style(ax, "PageRank mass concentrates on well-linked nodes")
    return save(fig, "ml_fig_c83_16.png")


def c83_17_cohort_funnel():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    layers = [
        (0.5, 4.6, 9.0, "All encounters N=10,000", TEAL),
        (1.2, 3.4, 7.6, "Adult + index stroke code  N=2,400", DEEP),
        (2.0, 2.2, 6.0, "Complete imaging + labs  N=1,100", GOLD),
        (2.8, 1.0, 4.4, "Analytic cohort  N=860", MINT),
    ]
    for x, y, w, lab, c in layers:
        box(ax, x, y, w, 0.9, lab, fc=c, fs=9, tc="white" if c != GOLD and c != MINT else INK)
    style(ax, "Cohort selection funnel (document every exclusion)")
    return save(fig, "ml_fig_c83_17.png")


def c83_18_transport():
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    sites = ["Develop\nsite A", "Temporal\nholdout A", "External\nsite B", "External\nsite C"]
    auroc = [0.84, 0.81, 0.72, 0.68]
    ece = [0.03, 0.04, 0.09, 0.12]
    x = np.arange(len(sites))
    ax.bar(x - 0.18, auroc, 0.35, color=TEAL, label="AUROC")
    ax.bar(x + 0.18, ece, 0.35, color=GOLD, label="ECE (calib. err)")
    ax.set_xticks(x); ax.set_xticklabels(sites, fontsize=8)
    style(ax, "Transportability: discrimination can hold while calibration drifts")
    ax.legend(fontsize=8); ax.set_ylim(0, 1); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c83_18.png")


def c83_19_notation_strip():
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    items = [
        (0.3, "x", "features"),
        (2.3, "y", "label/target"),
        (4.3, "ŷ", "prediction"),
        (6.3, "L", "loss"),
        (8.3, "θ", "params"),
        (10.1, "R", "risk/risk"),
    ]
    for x, sym, lab in items:
        box(ax, x, 1.6, 1.7, 1.5, f"{sym}\n{lab}", fc=TEAL, fs=10)
    ax.text(6, 0.6, "Shared notation reduces glossary thrash across chapters",
            ha="center", fontsize=10, color=DEEP, fontweight="bold")
    style(ax, "Core notation strip (glossary anchor)")
    return save(fig, "ml_fig_c83_19.png")


# ───────────────────────── Cycle 84 ─────────────────────────

def c84_00_norm_ball():
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    t = np.linspace(0, 2 * np.pi, 400)
    # L2
    ax.plot(np.cos(t), np.sin(t), color=TEAL, lw=2, label="L2 ball")
    # L1 diamond
    ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=GOLD, lw=2, label="L1 ball")
    # L∞ square
    ax.plot([1, 1, -1, -1, 1], [1, -1, -1, 1, 1], color=SLATE, lw=1.5, ls="--", label="L∞ ball")
    style(ax, "Unit balls: geometry of regularization")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_00.png")


def c84_01_pre_reg_boxes():
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.4, 2.8, 2.8, 1.6, "Hypothesis\npre-registered", fc=TEAL, fs=10)
    box(ax, 3.6, 2.8, 2.8, 1.6, "Analysis\nplan fixed", fc=DEEP, fs=10)
    box(ax, 6.8, 2.8, 2.8, 1.6, "Deviations\nlogged", fc=GOLD, tc=INK, fs=10)
    box(ax, 2.0, 0.5, 6.0, 1.5, "Optional exploratory analyses clearly labeled",
        fc=SOFT, tc=INK, fs=10)
    style(ax, "Preregistration boxes for honest ML claims")
    return save(fig, "ml_fig_c84_01.png")


def c84_02_ood_map():
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    rng = np.random.default_rng(8)
    id_ = rng.normal(0, 0.6, (200, 2))
    ood = rng.normal(2.5, 0.5, (40, 2))
    ax.scatter(id_[:, 0], id_[:, 1], c=TEAL, s=16, alpha=0.7, label="in-support")
    ax.scatter(ood[:, 0], ood[:, 1], c=ROSE, s=28, alpha=0.8, label="OOD")
    ax.add_patch(Ellipse((0, 0), 3.2, 3.2, fill=False, edgecolor=GOLD, lw=2, ls="--"))
    ax.text(0, 1.9, "support of training P(x)", ha="center", color=GOLD, fontsize=9)
    style(ax, "Out-of-distribution points sit outside training support")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_02.png")


def c84_03_color_scale_trap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    z = np.outer(np.linspace(0, 1, 20), np.ones(20))
    ax1.imshow(z, cmap="jet")
    style(ax1, "Rainbow scale (misleading)")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2.imshow(z, cmap="viridis")
    style(ax2, "Perceptually uniform scale")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle("Visualization: colormap choice changes perceived magnitude",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_03.png")


def c84_04_ci_vs_pi():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.linspace(0, 10, 50)
    mean = 1 + 0.4 * x
    ax.plot(x, mean, color=TEAL, lw=2.5, label="mean fit")
    ax.fill_between(x, mean - 0.3, mean + 0.3, color=TEAL, alpha=0.35, label="CI for mean")
    ax.fill_between(x, mean - 1.4, mean + 1.4, color=GOLD, alpha=0.2, label="PI for new y")
    style(ax, "Confidence interval ≠ prediction interval")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_04.png")


def c84_05_hierarchical():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 3.5, 4.8, 3.0, 0.9, "all patients", fc=TEAL, fs=10)
    box(ax, 0.8, 3.0, 3.0, 0.9, "site A", fc=DEEP, fs=10)
    box(ax, 6.2, 3.0, 3.0, 0.9, "site B", fc=DEEP, fs=10)
    box(ax, 0.3, 1.0, 1.8, 0.9, "ICU", fc=GOLD, tc=INK, fs=9)
    box(ax, 2.3, 1.0, 1.8, 0.9, "ward", fc=GOLD, tc=INK, fs=9)
    box(ax, 5.7, 1.0, 1.8, 0.9, "ICU", fc=GOLD, tc=INK, fs=9)
    box(ax, 7.7, 1.0, 1.8, 0.9, "ED", fc=GOLD, tc=INK, fs=9)
    for x in (2.3, 7.7):
        ax.annotate("", xy=(x, 3.9), xytext=(5, 4.8),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2))
    style(ax, "Hierarchical clustering / hierarchical structure sketch")
    return save(fig, "ml_fig_c84_05.png")


def c84_06_tfidf():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    terms = ["the", "stroke", "NIHSS", "and", "thrombectomy"]
    tf = np.array([0.12, 0.04, 0.03, 0.1, 0.02])
    idf = np.array([0.1, 2.4, 3.1, 0.15, 3.5])
    tfidf = tf * idf
    x = np.arange(len(terms))
    ax.bar(x - 0.2, tf, 0.2, color=SLATE, label="tf")
    ax.bar(x, idf / 5, 0.2, color=GOLD, label="idf (scaled)")
    ax.bar(x + 0.2, tfidf, 0.2, color=TEAL, label="tf-idf")
    ax.set_xticks(x); ax.set_xticklabels(terms, rotation=15)
    style(ax, "TF-IDF upweights informative rare clinical terms")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_06.png")


def c84_07_pipeline_nest():
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.3, 1.2, 11.2, 2.2, "", fc=SOFT, tc=INK, fs=1)
    box(ax, 0.6, 1.6, 2.4, 1.4, "impute", fc=TEAL, fs=10)
    box(ax, 3.3, 1.6, 2.4, 1.4, "scale", fc=DEEP, fs=10)
    box(ax, 6.0, 1.6, 2.4, 1.4, "select", fc=GOLD, tc=INK, fs=10)
    box(ax, 8.7, 1.6, 2.4, 1.4, "model", fc=MINT, tc=INK, fs=10)
    ax.text(6, 0.5, "Entire pipeline fits only on training folds of CV",
            ha="center", fontsize=10, color=ROSE, fontweight="bold")
    style(ax, "Nested preprocessing: no fit on validation/test rows")
    return save(fig, "ml_fig_c84_07.png")


def c84_08_ica_vs_pca():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    rng = np.random.default_rng(12)
    s1 = rng.uniform(-1, 1, 300)
    s2 = rng.uniform(-1, 1, 300)
    # mixed
    x = 0.8 * s1 + 0.5 * s2
    y = 0.3 * s1 + 0.9 * s2
    ax1.scatter(x, y, s=8, c=TEAL, alpha=0.6)
    style(ax1, "Observed mixture")
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.25)
    ax2.scatter(s1, s2, s=8, c=GOLD, alpha=0.6)
    style(ax2, "ICA aims at independent sources")
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.25)
    fig.suptitle("PCA finds variance axes; ICA seeks independence",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_08.png")


def c84_09_partial_residual():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    rng = np.random.default_rng(6)
    x = np.linspace(-2, 2, 80)
    y = 0.8 * x + 0.35 * x ** 2 + rng.normal(0, 0.25, 80)
    ax.scatter(x, y, c=TEAL, s=22, alpha=0.75)
    ax.plot(x, 0.8 * x, color=GOLD, lw=2, label="linear term")
    ax.plot(x, 0.8 * x + 0.35 * x ** 2, color=ROSE, lw=2, label="true nonlinear")
    style(ax, "Partial residual view: detect leftover curvature")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x_j"); ax.set_ylabel("partial residual")
    fig.tight_layout()
    return save(fig, "ml_fig_c84_09.png")


def c84_10_pr_curve():
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    r = np.linspace(0.05, 1, 50)
    p_prev_high = 0.9 - 0.35 * r + 0.05 * np.sin(5 * r)
    p_prev_low = 0.55 - 0.4 * r
    ax.plot(r, np.clip(p_prev_high, 0.05, 1), color=TEAL, lw=2.2, label="higher prevalence")
    ax.plot(r, np.clip(p_prev_low, 0.05, 1), color=GOLD, lw=2.2, label="lower prevalence")
    style(ax, "Precision–recall depends on prevalence")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_10.png")


def c84_11_batchnorm_place():
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    steps = [("Linear", TEAL), ("BatchNorm", GOLD), ("ReLU", DEEP), ("Dropout", SLATE), ("Linear", TEAL)]
    for i, (lab, c) in enumerate(steps):
        box(ax, 0.3 + i * 2.35, 0.9, 2.1, 1.3, lab, fc=c, fs=10, tc="white" if c != GOLD else INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.3 + (i + 1) * 2.35, 1.55),
                        xytext=(0.3 + i * 2.35 + 2.1, 1.55),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, "Common modern MLP block ordering (schematic)")
    return save(fig, "ml_fig_c84_11.png")


def c84_12_simclr_aug():
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.4, 1.4, 2.2, 1.4, "image x", fc=SLATE, fs=10)
    box(ax, 3.2, 2.4, 2.2, 1.2, "aug t", fc=TEAL, fs=10)
    box(ax, 3.2, 0.6, 2.2, 1.2, "aug t′", fc=TEAL, fs=10)
    box(ax, 6.2, 2.4, 1.6, 1.2, "z", fc=GOLD, tc=INK, fs=11)
    box(ax, 6.2, 0.6, 1.6, 1.2, "z′", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.4, 1.4, 1.4, 1.4, "NT-Xent", fc=ROSE, fs=9)
    ax.annotate("", xy=(3.2, 3.0), xytext=(2.6, 2.3),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(3.2, 1.2), xytext=(2.6, 1.8),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(6.2, 3.0), xytext=(5.4, 3.0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(6.2, 1.2), xytext=(5.4, 1.2),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(8.4, 2.3), xytext=(7.8, 3.0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(8.4, 1.8), xytext=(7.8, 1.2),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, "SimCLR-style contrastive augmentation graph")
    return save(fig, "ml_fig_c84_12.png")


def c84_13_spectrogram():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    t = np.linspace(0, 1, 200)
    f = np.linspace(0, 1, 80)
    T, F = np.meshgrid(t, f)
    S = np.exp(-((F - 0.2 - 0.5 * T) ** 2) / 0.01) + 0.4 * np.exp(-((F - 0.7) ** 2) / 0.02)
    ax.imshow(S, aspect="auto", origin="lower", cmap="mako" if "mako" in plt.colormaps() else "viridis",
              extent=[0, 1, 0, 1])
    style(ax, "Audio spectrogram: time–frequency teaching view")
    ax.set_xlabel("time"); ax.set_ylabel("frequency")
    fig.tight_layout()
    return save(fig, "ml_fig_c84_13.png")


def c84_14_policy_value():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    s = np.arange(6)
    v = np.array([0.2, 0.5, 0.9, 0.7, 0.4, 0.1])
    ax.bar(s, v, color=TEAL, edgecolor=DEEP)
    # policy arrows preferred actions
    for i, a in enumerate([1, 1, 0, -1, -1, 0]):
        if a != 0:
            ax.annotate("", xy=(i + 0.35 * a, v[i] + 0.08), xytext=(i, v[i] + 0.08),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    style(ax, "State values V(s) with greedy policy arrows")
    ax.set_xlabel("state"); ax.set_ylabel("V(s)"); ax.set_ylim(0, 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_14.png")


def c84_15_quantization():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.linspace(-1, 1, 500)
    y = np.tanh(2.5 * x)
    q = np.round(y * 4) / 4  # 3-bit-ish
    ax.plot(x, y, color=TEAL, lw=2.2, label="full precision")
    ax.plot(x, q, color=GOLD, lw=2, label="quantized levels")
    style(ax, "Quantization maps continuous weights/acts to discrete grids")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("input"); ax.set_ylabel("activation")
    fig.tight_layout()
    return save(fig, "ml_fig_c84_15.png")


def c84_16_modularity():
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    rng = np.random.default_rng(15)
    # two communities
    for center, col in [((-1, 0), TEAL), ((1, 0), GOLD)]:
        pts = rng.normal(center, 0.25, size=(25, 2))
        ax.scatter(pts[:, 0], pts[:, 1], c=col, s=40)
        # within edges
        for i in range(20):
            a, b = rng.integers(0, 25, 2)
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color=col, alpha=0.25, lw=0.8)
    # few between
    ax.plot([-0.7, 0.7], [0.1, -0.1], color=ROSE, lw=1.2, alpha=0.7)
    ax.plot([-0.6, 0.6], [-0.2, 0.15], color=ROSE, lw=1.2, alpha=0.7)
    style(ax, "Community structure: dense within, sparse between")
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    return save(fig, "ml_fig_c84_16.png")


def c84_17_shift_types():
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2))
    rng = np.random.default_rng(1)
    # covariate shift
    ax = axes[0]
    ax.hist(rng.normal(0, 1, 200), bins=20, alpha=0.6, color=TEAL, label="train P(x)")
    ax.hist(rng.normal(1.2, 1, 200), bins=20, alpha=0.6, color=GOLD, label="deploy P'(x)")
    style(ax, "Covariate shift")
    ax.legend(fontsize=6)
    # label shift
    ax = axes[1]
    ax.bar([0, 1], [0.7, 0.3], color=TEAL, alpha=0.8, label="train P(y)")
    ax.bar([0, 1], [0.4, 0.6], color=GOLD, alpha=0.5, label="deploy P'(y)")
    style(ax, "Label shift")
    ax.legend(fontsize=6)
    # concept shift
    ax = axes[2]
    x = np.linspace(-2, 2, 50)
    ax.plot(x, 1 / (1 + np.exp(-2 * x)), color=TEAL, lw=2, label="P(y|x)")
    ax.plot(x, 1 / (1 + np.exp(-2 * x + 2)), color=ROSE, lw=2, label="P'(y|x)")
    style(ax, "Concept shift")
    ax.legend(fontsize=6)
    fig.suptitle("Dataset shift taxonomy for deployment risk",
                 fontsize=11, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_17.png")


def c84_18_decision_curve():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    pt = np.linspace(0.01, 0.8, 100)
    # net benefit schematic
    nb_model = 0.25 - 0.4 * pt
    nb_all = 0.15 - 0.9 * pt
    nb_none = np.zeros_like(pt)
    ax.plot(pt, nb_model, color=TEAL, lw=2.2, label="model")
    ax.plot(pt, nb_all, color=GOLD, lw=2, label="treat all")
    ax.plot(pt, nb_none, color=SLATE, lw=1.5, ls="--", label="treat none")
    style(ax, "Decision curve: net benefit vs threshold probability")
    ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, "ml_fig_c84_18.png")


def c84_19_confounder_dag():
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 3.7, 4.4, 2.6, 1.1, "U confounder", fc=ROSE, fs=10)
    box(ax, 0.8, 1.5, 2.6, 1.1, "X treatment", fc=TEAL, fs=10)
    box(ax, 6.6, 1.5, 2.6, 1.1, "Y outcome", fc=GOLD, tc=INK, fs=10)
    ax.annotate("", xy=(2.1, 2.6), xytext=(4.5, 4.4),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(7.9, 2.6), xytext=(5.5, 4.4),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(6.6, 2.0), xytext=(3.4, 2.0),
                arrowprops=dict(arrowstyle="->", color=DEEP, lw=2))
    ax.text(5, 0.6, "Association X–Y may not be causal without addressing U",
            ha="center", fontsize=9, color=ROSE, fontweight="bold")
    style(ax, "DAG reminder: prediction ≠ causation")
    return save(fig, "ml_fig_c84_19.png")


CHAPTERS = [
    "00-mathematical-foundations-for-machine-learning.md",
    "00a-preface.md",
    "01-basic-concepts-of-machine-learning-and-artificial-intelligence.md",
    "02-visualization.md",
    "03-probability-and-statistics.md",
    "04-clustering.md",
    "05-frequent-itemset-mining-sequence-mining-and-information-retrieval.md",
    "06-feature-engineering.md",
    "07-dimensionality-reduction-and-data-decomposition.md",
    "08-regression-analysis.md",
    "09-classification.md",
    "10-neural-networks-and-deep-learning.md",
    "11-self-supervised-deep-learning.md",
    "12-deep-learning-models-and-applications-for-text-vision-and-audio.md",
    "13-reinforcement-learning.md",
    "14-making-lighter-neural-network-and-machine-learning-models.md",
    "15-graph-mining-algorithms.md",
    "16-concepts-and-challenges-of-working-with-data.md",
    "17-closing-synthesis-senior-practice.md",
    "18-selected-glossary.md",
]

CAPTIONS = {
    82: [
        "Gradient steps on a convex quadratic bowl and its contours—optimization geometry, not a clinical claim.",
        "Reproducible ML study loop; pred ≠ cause at the center.",
        "Inductive bias spectrum from local k-NN to flexible deep models.",
        "Anscombe lesson: identical summary stats can hide different structure—always plot.",
        "Bayes update: prior × likelihood → posterior with MAP marker.",
        "Choosing k: inertia elbow plus silhouette separation check.",
        "Support, confidence, and lift for association rules—filter, then do not claim cause.",
        "Target encoding must be fit inside each CV fold to avoid label leakage.",
        "PCA scree and cumulative variance for choosing component count.",
        "Residual diagnostics: constant scatter vs heteroscedastic fan.",
        "Decision threshold shifts when FN costs more than FP (synthetic costs).",
        "Vanishing gradients with depth; residual paths preserve signal (schematic).",
        "Contrastive SSL geometry: pull positives, push negatives on the unit sphere.",
        "Early vs late multimodal fusion: where modalities meet.",
        "Bandit cumulative regret: greedy lock-in vs ε-greedy vs UCB sketch.",
        "Lighter models: magnitude pruning and teacher→student distillation.",
        "Graphs: random-walk exploration vs one-shot message passing.",
        "Missingness mechanisms MCAR / MAR / MNAR change recoverable structure.",
        "Senior appraisal diamond: question, leakage, validation, utility.",
        "Glossary hub map: loss, capacity, validation, calibration, bias, generalization.",
    ],
    83: [
        "SVD building blocks for low-rank matrix approximation.",
        "Temporal train/val/test split—do not shuffle across deployment time.",
        "Bias–variance tradeoff versus model capacity.",
        "Parallel coordinates for multi-feature pattern scanning.",
        "Central limit theorem: sampling distribution of the mean.",
        "DBSCAN idea: dense cores, borders, and noise points.",
        "Apriori itemset lattice with support pruning.",
        "Feature interaction heatmap (schematic strengths).",
        "Neighbor/perplexity settings change embedding topology emphasis.",
        "Regularization coefficient paths as λ grows.",
        "Reliability diagram: overconfident vs calibrated probabilities.",
        "Attention scores become softmax weights that sum to one.",
        "Masked language modeling token prediction sketch.",
        "CNN receptive field growth with stacked local filters.",
        "Temporal-difference backup and TD error δ.",
        "Softmax temperature softens teacher labels for distillation.",
        "PageRank mass on a small directed graph.",
        "Cohort selection funnel—document every exclusion.",
        "Transportability: AUROC may hold while calibration error rises.",
        "Core notation strip shared across the glossary.",
    ],
    84: [
        "L1 / L2 / L∞ unit balls—geometry behind regularizers.",
        "Preregistration boxes for honest analytic claims.",
        "Out-of-distribution points outside training support.",
        "Colormap choice changes perceived magnitude (rainbow trap).",
        "Confidence interval for the mean vs prediction interval for a new y.",
        "Hierarchical structure sketch for multi-level data.",
        "TF-IDF upweights informative rare terms.",
        "Nested preprocessing pipeline fits only on training folds.",
        "PCA vs ICA teaching contrast on mixed sources.",
        "Partial residuals reveal leftover nonlinear curvature.",
        "Precision–recall curves shift with prevalence.",
        "Modern MLP block ordering with BatchNorm and Dropout.",
        "SimCLR-style dual augmentation contrastive graph.",
        "Audio spectrogram as a time–frequency teaching view.",
        "State values with greedy policy action arrows.",
        "Quantization maps continuous activations to discrete levels.",
        "Community structure: dense within, sparse between modules.",
        "Covariate, label, and concept shift taxonomy.",
        "Decision curve net benefit vs threshold probability.",
        "DAG reminder: a confounder U can make association non-causal (pred ≠ cause).",
    ],
}

GENERATORS = {
    82: [
        c82_00_quadratic_bowl, c82_01_study_loop, c82_02_inductive_bias, c82_03_anscombe_lesson,
        c82_04_likelihood_posterior, c82_05_silhouette_elbow, c82_06_support_conf_lift,
        c82_07_encoding_leakage, c82_08_pca_scree, c82_09_residual_diagnostics,
        c82_10_threshold_cost, c82_11_vanishing_grad, c82_12_contrastive_pairs,
        c82_13_early_late_fusion, c82_14_bandit_regret, c82_15_prune_distill,
        c82_16_random_walk_mp, c82_17_missingness, c82_18_appraisal_diamond, c82_19_glossary_map,
    ],
    83: [
        c83_00_svd_blocks, c83_01_temporal_split, c83_02_bias_variance, c83_03_parallel_coords,
        c83_04_clt, c83_05_dbscan_cores, c83_06_apriori_lattice, c83_07_interaction_heat,
        c83_08_perplexity, c83_09_regularization_paths, c83_10_reliability,
        c83_11_attention_softmax, c83_12_mlm_mask, c83_13_receptive_field, c83_14_td_backup,
        c83_15_teacher_student_temp, c83_16_pagerank, c83_17_cohort_funnel, c83_18_transport,
        c83_19_notation_strip,
    ],
    84: [
        c84_00_norm_ball, c84_01_pre_reg_boxes, c84_02_ood_map, c84_03_color_scale_trap,
        c84_04_ci_vs_pi, c84_05_hierarchical, c84_06_tfidf, c84_07_pipeline_nest,
        c84_08_ica_vs_pca, c84_09_partial_residual, c84_10_pr_curve, c84_11_batchnorm_place,
        c84_12_simclr_aug, c84_13_spectrogram, c84_14_policy_value, c84_15_quantization,
        c84_16_modularity, c84_17_shift_types, c84_18_decision_curve, c84_19_confounder_dag,
    ],
}


def embed_cycle(cycle: int) -> int:
    curr = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
    n = 0
    for i, ch in enumerate(CHAPTERS):
        p = curr / ch
        fig = f"ml_fig_c{cycle}_{i:02d}.png"
        cap = CAPTIONS[cycle][i]
        block = (
            f"\n![c{cycle} teaching panel {i:02d} (original).](../assets/figures/{fig})\n"
            f"*Figure — {cap} Synthetic teaching geometry—not a causal claim.*\n"
        )
        text = p.read_text(encoding="utf-8")
        if fig in text:
            continue
        # Insert before Chapter Summary if present, else append
        marker = "## Chapter Summary"
        if marker in text:
            text = text.replace(marker, block + "\n" + marker, 1)
        else:
            text = text.rstrip() + "\n" + block
        p.write_text(text, encoding="utf-8")
        n += 1
    return n


def main(cycles=None):
    cycles = cycles or [82, 83, 84]
    wrote = []
    for c in cycles:
        for fn in GENERATORS[c]:
            path = fn()
            wrote.append(path.name)
            print("WROTE", path.name)
        emb = embed_cycle(c)
        print(f"EMBEDDED cycle {c}: {emb} chapters")
    print(f"TOTAL_FIGURES {len(wrote)}")
    return wrote


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cyc = [int(x) for x in sys.argv[1].split(",")]
        main(cyc)
    else:
        main()
