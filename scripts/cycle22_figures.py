#!/usr/bin/env python3
"""Cycle-22 densify — push 12 chapters from floor-16 toward >=17 (teal; original)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, DEEP, INK, GOLD, GRAY, SLATE, SOFT = (
    "#0d9488",
    "#0f766e",
    "#0f172a",
    "#c9a227",
    "#94a3b8",
    "#64748b",
    "#ecfeff",
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


def fig_gradient_flow():
    """Ch00 math: vanishing vs residual gradient magnitude by depth."""
    depth = np.arange(1, 21)
    vanish = 0.9 ** depth
    residual = np.maximum(0.35, 0.95 ** (depth * 0.15))
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.semilogy(depth, vanish, color=GOLD, lw=2.4, label="plain deep stack")
    ax.semilogy(depth, residual, color=TEAL, lw=2.4, label="residual-style floor")
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("relative gradient magnitude")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Gradient flow vs depth (teaching curves)")
    ax.text(
        0.98,
        0.9,
        "Illustrative decay only.\nArchitecture eases optimization;\nnot a causal clinical path.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_gradient_flow_depth.png")


def fig_bias_variance_decomp():
    """Ch01: bias-variance toy decomposition vs model capacity."""
    cap = np.linspace(1, 10, 40)
    bias2 = 0.55 * np.exp(-0.35 * (cap - 1))
    var = 0.02 * (cap - 1) ** 1.4
    noise = 0.08
    tot = bias2 + var + noise
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.fill_between(cap, 0, bias2, color=TEAL, alpha=0.55, label="bias²")
    ax.fill_between(cap, bias2, bias2 + var, color=GOLD, alpha=0.55, label="variance")
    ax.fill_between(cap, bias2 + var, tot, color=GRAY, alpha=0.45, label="irreducible")
    ax.plot(cap, tot, color=INK, lw=2.0, label="expected risk")
    ax.set_xlabel("capacity (teaching scale)")
    ax.set_ylabel("error")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4)
    style_ax(ax, "Bias–variance sketch (synthetic)")
    ax.text(
        0.98,
        0.55,
        "Minimum risk ≠ truth of\nmechanism. Pred ≠ cause.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_bias_variance_stack.png")


def fig_ecdf_compare():
    """Ch02: ECDF comparison of two sites."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0.4, 1.15, 180)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for data, c, lab in [(a, TEAL, "site A"), (b, GOLD, "site B")]:
        xs = np.sort(data)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=c, lw=2.2, label=lab)
    ax.set_xlabel("feature value")
    ax.set_ylabel("ECDF")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Empirical CDFs detect shift without binning")
    ax.text(
        0.02,
        0.55,
        "Shift ≠ site effect\ncausally without design.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_ecdf_compare.png")


def fig_beta_binomial():
    """Ch03: Beta-Binomial posterior updating."""
    from math import comb

    # skip scipy; plot beta density via gamma formula
    def beta_pdf(x, a, b):
        # unnormalized then normalize numerically
        y = x ** (a - 1) * (1 - x) ** (b - 1)
        y = np.where((x <= 0) | (x >= 1), 0, y)
        return y / (np.trapezoid(y, x) + 1e-12)

    x = np.linspace(0.001, 0.999, 400)
    prior = beta_pdf(x, 2, 2)
    post1 = beta_pdf(x, 2 + 3, 2 + 7)  # 3/10
    post2 = beta_pdf(x, 2 + 3 + 12, 2 + 7 + 8)  # more data
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.plot(x, prior, color=GRAY, lw=2, label="prior Beta(2,2)")
    ax.plot(x, post1, color=GOLD, lw=2.2, label="after n=10")
    ax.plot(x, post2, color=TEAL, lw=2.4, label="after more data")
    ax.set_xlabel("θ (event rate)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Beta–Binomial posterior sharpening")
    ax.text(
        0.98,
        0.9,
        "Posteriors update beliefs.\nNot a causal treatment effect\nwithout design.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_beta_binomial.png")


def fig_gmm_responsibilities():
    """Ch04: 1D GMM soft responsibilities."""
    rng = np.random.default_rng(2)
    x = np.linspace(-4, 6, 400)
    # two components
    p1 = np.exp(-0.5 * ((x + 1) / 0.8) ** 2) / (0.8 * np.sqrt(2 * np.pi))
    p2 = np.exp(-0.5 * ((x - 2.5) / 1.0) ** 2) / (1.0 * np.sqrt(2 * np.pi))
    mix = 0.45 * p1 + 0.55 * p2
    r1 = 0.45 * p1 / mix
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.0), sharex=True)
    axes[0].plot(x, mix, color=INK, lw=2.2, label="mixture density")
    axes[0].plot(x, 0.45 * p1, color=TEAL, lw=1.6, ls="--", label="comp 1 weighted")
    axes[0].plot(x, 0.55 * p2, color=GOLD, lw=1.6, ls="--", label="comp 2 weighted")
    axes[0].legend(frameon=False, fontsize=8)
    style_ax(axes[0], "1-D GMM density")
    axes[1].plot(x, r1, color=TEAL, lw=2.3, label=r"$r_{i1}$")
    axes[1].plot(x, 1 - r1, color=GOLD, lw=2.3, label=r"$r_{i2}$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("responsibility")
    axes[1].legend(frameon=False, fontsize=8)
    style_ax(axes[1], "Soft assignment (E-step)")
    axes[1].text(
        0.02,
        0.15,
        "Soft labels are probabilistic geometry—not etiologic certainty.",
        transform=axes[1].transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.suptitle("GMM responsibilities (synthetic; original)", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "ml_fig_gmm_responsibilities.png")


def fig_tfidf_sparsity():
    """Ch05: bag-of-words / TF-IDF sparsity heatmap toy."""
    rng = np.random.default_rng(6)
    # docs x terms sparse
    M = rng.random((12, 20))
    M[M < 0.78] = 0
    # TF-IDF-ish scale
    df = (M > 0).sum(axis=0) + 1
    idf = np.log(1 + 12 / df)
    T = M * idf
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].imshow(M > 0, cmap="YlGnBu", aspect="auto")
    style_ax(axes[0], "Binary term presence")
    axes[0].set_xlabel("terms")
    axes[0].set_ylabel("documents")
    im = axes[1].imshow(T, cmap="YlGnBu", aspect="auto")
    style_ax(axes[1], "TF–IDF weights (toy)")
    axes[1].set_xlabel("terms")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle("Sparse text matrices (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, "High weight ≠ clinical importance without task labels.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_tfidf_sparsity.png")


def fig_woe_binning():
    """Ch06: weight-of-evidence style bin plot (teaching)."""
    bins = ["<50", "50-60", "60-70", "70-80", "80+"]
    # event rates
    rate = np.array([0.08, 0.12, 0.18, 0.28, 0.35])
    base = 0.18
    woe = np.log((rate / (1 - rate)) / (base / (1 - base)))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].bar(bins, rate, color=TEAL, edgecolor="white")
    axes[0].axhline(base, color=GOLD, ls="--", lw=1.5, label=f"base rate {base}")
    axes[0].set_ylabel("event rate")
    axes[0].legend(frameon=False, fontsize=8)
    style_ax(axes[0], "Binned event rates (synthetic)")
    colors = [TEAL if w > 0 else GOLD for w in woe]
    axes[1].bar(bins, woe, color=colors, edgecolor="white")
    axes[1].axhline(0, color=GRAY, lw=1)
    axes[1].set_ylabel("WoE-style log-odds lift")
    style_ax(axes[1], "Weight-of-evidence sketch")
    axes[1].text(
        0.02,
        0.9,
        "Bin edges leak if fit on full data.\nAssociation ≠ causation.",
        transform=axes[1].transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.suptitle("Supervised binning caution (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_woe_binning.png")


def fig_random_projection():
    """Ch07: Johnson-Lindenstrauss style distance preservation."""
    rng = np.random.default_rng(8)
    n, d = 80, 200
    X = rng.normal(size=(n, d))
    # pairwise distances sample
    idx = rng.choice(n, size=(120, 2), replace=True)
    idx = idx[idx[:, 0] != idx[:, 1]]
    def pdist(A, pairs):
        return np.linalg.norm(A[pairs[:, 0]] - A[pairs[:, 1]], axis=1)

    d_hi = pdist(X, idx)
    # project to k dims
    ks = [5, 20, 50]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for ax, k in zip(axes, ks):
        R = rng.normal(size=(d, k)) / np.sqrt(k)
        Y = X @ R
        d_lo = pdist(Y, idx)
        ax.scatter(d_hi, d_lo, s=12, c=TEAL, alpha=0.65)
        lim = max(d_hi.max(), d_lo.max())
        ax.plot([0, lim], [0, lim], color=GOLD, ls="--", lw=1.3)
        ax.set_xlabel("high-D dist")
        ax.set_ylabel("proj dist")
        style_ax(ax, f"k={k}")
    fig.suptitle("Random projection distance preservation (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.text(0.5, -0.04, "Distance preservation ≠ semantic or causal meaning.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_random_projection.png")


def fig_partial_dependence_caution():
    """Ch08: PDP can mislead under dependence — sketch."""
    rng = np.random.default_rng(1)
    x1 = rng.uniform(0, 1, 300)
    x2 = x1 + rng.normal(0, 0.08, 300)  # dependent
    # true y depends on x1 only-ish but correlated x2
    y = 2 * x1 + rng.normal(0, 0.15, 300)
    # fake PDP for x2: appears associated
    grid = np.linspace(0, 1, 25)
    # marginal smoother
    pdp = []
    for g in grid:
        w = np.exp(-((x2 - g) ** 2) / 0.02)
        pdp.append(np.average(y, weights=w + 1e-6))
    pdp = np.array(pdp)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].scatter(x1, x2, s=12, c=TEAL, alpha=0.55)
    axes[0].set_xlabel("x1 (driver)")
    axes[0].set_ylabel("x2 (correlated)")
    style_ax(axes[0], "Feature dependence")
    axes[1].plot(grid, pdp, color=GOLD, lw=2.4)
    axes[1].set_xlabel("x2")
    axes[1].set_ylabel("partial dependence (toy)")
    style_ax(axes[1], "Marginal PDP can look causal")
    axes[1].text(
        0.02,
        0.9,
        "Dependence confounds PDP.\nNot a causal effect curve.",
        transform=axes[1].transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.suptitle("Partial dependence caution under collinearity (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_pdp_collinearity.png")


def fig_softmax_temp():
    """Ch10: softmax temperature sharpens/softens."""
    logits = np.array([2.0, 1.0, 0.2, -0.5])
    labels = ["A", "B", "C", "D"]
    temps = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    for ax, T in zip(axes, temps):
        z = logits / T
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        ax.bar(labels, p, color=TEAL, edgecolor="white")
        ax.set_ylim(0, 1)
        style_ax(ax, f"T={T}")
        ax.set_ylabel("softmax mass" if T == 0.5 else "")
    fig.suptitle("Softmax temperature (synthetic logits; original)", color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.text(0.5, -0.04, "Temperature is a calibration/distill knob—not disease severity.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_softmax_temp.png")


def fig_contrastive_pos_neg():
    """Ch11: contrastive pull/push in embedding space."""
    rng = np.random.default_rng(4)
    anchor = np.array([0.0, 0.0])
    pos = rng.normal([0.6, 0.2], 0.08, size=(8, 2))
    neg = rng.normal([-0.8, 0.9], 0.25, size=(20, 2))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(neg[:, 0], neg[:, 1], c=GRAY, s=40, label="negatives", alpha=0.8)
    ax.scatter(pos[:, 0], pos[:, 1], c=TEAL, s=55, label="positives")
    ax.scatter(*anchor, c=GOLD, s=120, marker="*", label="anchor", zorder=3)
    for p in pos:
        ax.annotate("", xy=p, xytext=anchor, arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
    for n in neg[:6]:
        ax.annotate("", xy=n, xytext=anchor, arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.9, alpha=0.6))
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Contrastive: pull positives, push negatives")
    ax.text(
        0.02,
        0.05,
        "Embedding geometry is task-shaped.\nNot automatic causal structure.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_contrastive_pull.png")


def fig_multihead_attn_shapes():
    """Ch12: multi-head attention shape diagram-like bars."""
    # show heads attending different patterns
    n = 10
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.2))
    rng = np.random.default_rng(0)
    for h, ax in enumerate(axes):
        M = np.zeros((n, n))
        if h == 0:
            M = np.tril(np.ones((n, n)))
        elif h == 1:
            for i in range(n):
                M[i, max(0, i - 1) : i + 2] = 1
        elif h == 2:
            M[:, 0] = 1  # attend CLS-like
            M += np.eye(n) * 0.3
        else:
            M = rng.random((n, n))
            M = M / M.sum(axis=1, keepdims=True)
        ax.imshow(M, cmap="YlGnBu", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"head {h+1}", fontsize=11, fontweight="bold", color=INK)
        ax.set_facecolor("#fafafa")
    fig.suptitle("Multi-head attention patterns (schematic; original)", color=INK, fontsize=12, fontweight="bold", y=1.06)
    fig.text(0.5, -0.05, "Heads capture different dependencies; maps ≠ clinical causation.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_multihead_patterns.png")


def main() -> None:
    fig_gradient_flow()
    fig_bias_variance_decomp()
    fig_ecdf_compare()
    fig_beta_binomial()
    fig_gmm_responsibilities()
    fig_tfidf_sparsity()
    fig_woe_binning()
    fig_random_projection()
    fig_partial_dependence_caution()
    fig_softmax_temp()
    fig_contrastive_pos_neg()
    fig_multihead_attn_shapes()
    print("DONE cycle-22")


if __name__ == "__main__":
    main()
