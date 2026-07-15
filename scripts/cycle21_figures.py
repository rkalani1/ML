#!/usr/bin/env python3
"""Cycle-21 densify — lift 12 chapters at floor-15 to >=16 (teal; original)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

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


def fig_evidence_stack():
    """Ch preface: claim strength vs required evidence (not causal by default)."""
    levels = [
        "Internal AUROC\nonly",
        "Calibration\n+ prevalence",
        "Temporal /\nsite external",
        "Prospective\nsilent trial",
        "Decision impact\n(RCT / stepped)",
    ]
    height = np.array([1, 2, 3, 4, 5], dtype=float)
    claim = [
        "ranking only",
        "prob. counsel",
        "transport hope",
        "ops-safe deploy",
        "action claim",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = [GRAY, TEAL, TEAL, DEEP, GOLD]
    bars = ax.bar(np.arange(len(levels)), height, color=colors, edgecolor="white", width=0.72)
    ax.set_xticks(np.arange(len(levels)), levels, fontsize=8)
    ax.set_ylabel("evidence burden (teaching scale)")
    ax.set_ylim(0, 5.8)
    for i, c in enumerate(claim):
        ax.text(i, height[i] + 0.15, c, ha="center", fontsize=8, color=SLATE)
    style_ax(ax, "Evidence stack for clinical ML claims")
    ax.text(
        0.98,
        0.92,
        "Higher bars license stronger claims.\nPrediction ≠ causation\nwithout design + confounders.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_evidence_stack.png")


def fig_binning_artifact():
    """Ch02: histogram bin width can invent or hide structure."""
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.normal(-0.8, 0.35, 180), rng.normal(1.0, 0.4, 160)])
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    for ax, bins, title in [
        (axes[0], 6, "Too few bins"),
        (axes[1], 18, "Reasonable bins"),
        (axes[2], 80, "Too many bins"),
    ]:
        ax.hist(x, bins=bins, color=TEAL, edgecolor="white", alpha=0.9)
        ax.axvline(-0.8, color=GOLD, ls="--", lw=1.2)
        ax.axvline(1.0, color=GOLD, ls="--", lw=1.2)
        style_ax(ax, title)
        ax.set_xlabel("feature")
        ax.set_yticks([])
    fig.suptitle(
        "Histogram binning artifacts (synthetic mixture; original)",
        color=INK,
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.02,
        "Bin choice is a visualization hyperparameter—not evidence of subtypes or causal modes.",
        ha="center",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_binning_artifact.png")


def fig_coassociation():
    """Ch04: bootstrap co-association heatmap for cluster stability."""
    rng = np.random.default_rng(4)
    n = 24
    # three blocks + noise off-diagonal
    C = np.full((n, n), 0.12)
    for a, b in [(0, 8), (8, 16), (16, 24)]:
        C[a:b, a:b] = 0.75 + 0.15 * rng.random((b - a, b - a))
    C = np.clip((C + C.T) / 2, 0, 1)
    np.fill_diagonal(C, 1.0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(C, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
    ax.set_xlabel("point j")
    ax.set_ylabel("point i")
    style_ax(ax, "Bootstrap co-association frequency")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(same cluster)", fontsize=9)
    ax.text(
        0.02,
        -0.14,
        "Stable blocks = reproducible geometry. Not etiologic labels.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_coassociation.png")


def fig_conf_lift_scatter():
    """Ch05: association rules — confidence vs lift scatter."""
    rng = np.random.default_rng(7)
    conf = rng.uniform(0.2, 0.95, 60)
    # lift loosely related; some high conf low lift (prevalent item)
    lift = 0.6 + 1.8 * conf + rng.normal(0, 0.35, 60) - 0.8 * (conf > 0.85)
    lift = np.clip(lift, 0.4, 3.2)
    support = rng.uniform(0.02, 0.25, 60)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    sc = ax.scatter(conf, lift, s=400 * support, c=TEAL, alpha=0.7, edgecolors="white")
    ax.axhline(1.0, color=GOLD, ls="--", lw=1.5, label="lift = 1 (independence)")
    ax.set_xlabel("confidence")
    ax.set_ylabel("lift")
    style_ax(ax, "Association rules: confidence vs lift")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.98,
        0.08,
        "Bubble size ∝ support.\nHigh conf + lift≈1: prevalent item,\nnot a discovery.\nAssociation ≠ causation.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_conf_lift_scatter.png")


def fig_cyclic_encoding():
    """Ch06: hour-of-day cyclic sin/cos encoding vs ordinal trap."""
    h = np.arange(0, 24)
    # ordinal distance wrongly says 23 far from 0
    ord_d = np.abs(h - 0)
    # cyclic: angular distance
    ang = 2 * np.pi * h / 24
    cyc_d = np.minimum(np.abs(h - 0), 24 - np.abs(h - 0))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].plot(h, ord_d, color=GOLD, lw=2.2, marker="o", ms=4, label="|h−0| ordinal")
    axes[0].plot(h, cyc_d, color=TEAL, lw=2.2, marker="s", ms=4, label="circular hours")
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("distance to midnight")
    axes[0].legend(frameon=False, fontsize=8)
    style_ax(axes[0], "Ordinal hours break midnight")
    # unit circle embedding
    axes[1].scatter(np.cos(ang), np.sin(ang), c=h, cmap="YlGnBu", s=55, edgecolors="white")
    axes[1].plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)), color=GRAY, lw=1)
    for hh in [0, 6, 12, 18]:
        a = 2 * np.pi * hh / 24
        axes[1].annotate(str(hh), (np.cos(a) * 1.12, np.sin(a) * 1.12), ha="center", va="center", fontsize=8, color=INK)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-1.4, 1.4)
    axes[1].set_ylim(-1.4, 1.4)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    style_ax(axes[1], "sin/cos embedding on circle")
    fig.suptitle(
        "Cyclic feature encoding for periodic time (original)",
        color=INK,
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    save(fig, "ml_fig_cyclic_encoding.png")


def fig_pca_scree_cumvar():
    """Ch07: scree + cumulative variance for PCA rank choice."""
    eigs = np.array([4.2, 2.1, 1.1, 0.55, 0.35, 0.22, 0.15, 0.10, 0.08, 0.05])
    eigs = eigs / eigs.sum()
    cum = np.cumsum(eigs)
    k = np.arange(1, len(eigs) + 1)
    fig, ax1 = plt.subplots(figsize=(7.8, 4.1))
    ax1.bar(k, eigs, color=TEAL, edgecolor="white", label="explained var fraction")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("variance fraction", color=DEEP)
    ax1.set_xticks(k)
    ax2 = ax1.twinx()
    ax2.plot(k, cum, color=GOLD, lw=2.4, marker="o", label="cumulative")
    ax2.axhline(0.9, color=GRAY, ls="--", lw=1.2)
    ax2.set_ylabel("cumulative variance", color=GOLD)
    ax2.set_ylim(0, 1.05)
    style_ax(ax1, "PCA scree and cumulative variance")
    ax1.text(
        0.98,
        0.55,
        "Elbow / 90% are heuristics.\nPCs ≠ clinical factors\nwithout labels.",
        transform=ax1.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_pca_scree_cumvar.png")


def fig_cooks_distance():
    """Ch08: Cook's distance influence diagnostic."""
    rng = np.random.default_rng(9)
    n = 40
    x = rng.uniform(0, 10, n)
    y = 1.5 + 0.8 * x + rng.normal(0, 0.9, n)
    # influential points
    x = np.append(x, [9.5, 1.0])
    y = np.append(y, [2.0, 12.0])  # high leverage high residual-ish
    # OLS
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    p = 2
    mse = np.sum(resid**2) / (len(x) - p)
    cooks = (resid**2 / (p * mse)) * (h / (1 - h) ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    axes[0].scatter(x[:-2], y[:-2], c=TEAL, s=35, alpha=0.8, label="bulk")
    axes[0].scatter(x[-2:], y[-2:], c=GOLD, s=70, zorder=3, label="influential")
    xs = np.linspace(0, 10, 50)
    axes[0].plot(xs, beta[0] + beta[1] * xs, color=DEEP, lw=2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False, fontsize=8)
    style_ax(axes[0], "Fit with influential points")
    idx = np.arange(len(cooks))
    axes[1].stem(idx, cooks, linefmt=TEAL, markerfmt="o", basefmt=" ")
    axes[1].axhline(4 / len(cooks), color=GOLD, ls="--", lw=1.4, label="4/n rule of thumb")
    axes[1].set_xlabel("observation index")
    axes[1].set_ylabel("Cook's D")
    axes[1].legend(frameon=False, fontsize=8)
    style_ax(axes[1], "Cook's distance")
    axes[1].text(
        0.02,
        0.9,
        "Influence ≠ causation.\nInvestigate before deleting.",
        transform=axes[1].transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.suptitle(
        "Leverage/influence diagnostics (synthetic OLS; original)",
        color=INK,
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    save(fig, "ml_fig_cooks_distance.png")


def fig_epsilon_decay():
    """Ch13: ε-greedy exploration schedules."""
    t = np.arange(0, 200)
    eps_lin = np.clip(1.0 - t / 150, 0.05, 1.0)
    eps_exp = 0.05 + 0.95 * np.exp(-t / 40)
    eps_const = np.full_like(t, 0.1, dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(t, eps_lin, color=TEAL, lw=2.3, label="linear anneal → 0.05")
    ax.plot(t, eps_exp, color=DEEP, lw=2.3, label="exponential decay")
    ax.plot(t, eps_const, color=GOLD, ls="--", lw=1.8, label="constant ε=0.1")
    ax.set_xlabel("episode / step")
    ax.set_ylabel("ε (explore probability)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "ε-greedy exploration schedules")
    ax.text(
        0.98,
        0.55,
        "Exploration is policy design.\nNot a causal path in the\nclinical process.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_epsilon_decay.png")


def fig_int8_quantize():
    """Ch14: quantization error vs bit width teaching curves."""
    bits = np.array([2, 3, 4, 5, 6, 8, 16])
    # synthetic relative MSE of uniform quantizer on Gaussian-ish weights
    mse = 1.0 / (2 ** (2 * (bits - 1)))
    mse = mse / mse[0]
    acc = 0.55 + 0.35 * (1 - np.exp(-(bits - 2) / 2.2))
    fig, ax1 = plt.subplots(figsize=(7.8, 4.1))
    ax1.plot(bits, mse, color=GOLD, lw=2.4, marker="o", label="relative quant. MSE")
    ax1.set_xlabel("bit width")
    ax1.set_ylabel("relative quantization MSE", color=GOLD)
    ax1.set_xticks(bits)
    ax2 = ax1.twinx()
    ax2.plot(bits, acc, color=TEAL, lw=2.4, marker="s", label="task accuracy (synth)")
    ax2.set_ylabel("accuracy (synthetic)", color=TEAL)
    ax2.set_ylim(0.5, 1.0)
    style_ax(ax1, "INT quantization: error vs accuracy trade-off")
    ax1.text(
        0.02,
        0.9,
        "INT8 often preserves ranking.\nRecalibrate probs after quantize.\nCompression ≠ new science.",
        transform=ax1.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_int8_quantize.png")


def fig_degree_assort():
    """Ch15: degree distribution + assortativity sketch."""
    rng = np.random.default_rng(5)
    # power-law-ish degrees
    deg = rng.zipf(2.2, 400)
    deg = np.clip(deg, 1, 40)
    # assortativity caricature: same-degree edges
    d_u = rng.choice(np.arange(1, 21), size=80)
    d_v_assort = np.clip(d_u + rng.normal(0, 1.5, 80), 1, 25)
    d_v_dis = 26 - d_u + rng.normal(0, 1.5, 80)
    d_v_dis = np.clip(d_v_dis, 1, 25)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    axes[0].hist(deg, bins=np.arange(0.5, 41.5, 1), color=TEAL, edgecolor="white")
    axes[0].set_xlabel("degree")
    axes[0].set_ylabel("count")
    axes[0].set_yscale("log")
    style_ax(axes[0], "Heavy-tailed degree (synthetic)")
    axes[1].scatter(d_u, d_v_assort, c=TEAL, s=28, alpha=0.7, label="assortative")
    axes[1].scatter(d_u, d_v_dis, c=GOLD, s=28, alpha=0.55, label="disassortative")
    axes[1].set_xlabel("degree(u)")
    axes[1].set_ylabel("degree(v) on edge")
    axes[1].legend(frameon=False, fontsize=8)
    style_ax(axes[1], "Assortativity caricature")
    axes[1].text(
        0.02,
        0.08,
        "Network structure ≠ clinical\ncausation without design.",
        transform=axes[1].transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.suptitle(
        "Degree tails and assortativity (synthetic graphs; original)",
        color=INK,
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    save(fig, "ml_fig_degree_assort.png")


def fig_shadow_mode():
    """Ch17: silent/shadow deployment funnel."""
    stages = ["Train/\nvalidate", "Shadow\n(no action)", "Compare\nto SOC", "Limited\nrelease", "Full\ndeploy"]
    n = np.array([100, 100, 72, 40, 25], dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, n, color=[TEAL, TEAL, DEEP, GOLD, GRAY], edgecolor="white", height=0.65)
    ax.set_yticks(y, stages, fontsize=9)
    ax.set_xlabel("% of candidate systems still advancing (teaching)")
    ax.set_xlim(0, 110)
    for yi, ni in zip(y, n):
        ax.text(ni + 1.5, yi, f"{int(ni)}%", va="center", fontsize=9, color=INK)
    style_ax(ax, "Shadow-mode gate funnel before action")
    ax.text(
        0.98,
        0.12,
        "Shadow = log predictions,\nno care change.\nPred ≠ cause; impact needs\nprospective design.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_shadow_mode.png")


def fig_lr_nomogram():
    """Ch18 glossary: LR+ nomogram-style prior → posterior odds teaching."""
    prior = np.array([0.05, 0.10, 0.20, 0.40])
    lr_plus = 5.0
    prior_odds = prior / (1 - prior)
    post_odds = prior_odds * lr_plus
    post = post_odds / (1 + post_odds)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(prior))
    w = 0.35
    ax.bar(x - w / 2, prior, width=w, color=GRAY, edgecolor="white", label="prior π")
    ax.bar(x + w / 2, post, width=w, color=TEAL, edgecolor="white", label=f"posterior (LR+={lr_plus:g})")
    ax.set_xticks(x, [f"π={p:.2f}" for p in prior])
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Bayes update with fixed LR+ (prevalence matters)")
    ax.text(
        0.02,
        0.9,
        "Same test, different base rates\n→ different PPV.\nLR is not a causal effect.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    for i, (pr, po) in enumerate(zip(prior, post)):
        ax.annotate(
            "",
            xy=(i + w / 2, po),
            xytext=(i - w / 2, pr),
            arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.2),
        )
    fig.tight_layout()
    save(fig, "ml_fig_lr_nomogram.png")


def main() -> None:
    fig_evidence_stack()
    fig_binning_artifact()
    fig_coassociation()
    fig_conf_lift_scatter()
    fig_cyclic_encoding()
    fig_pca_scree_cumvar()
    fig_cooks_distance()
    fig_epsilon_decay()
    fig_int8_quantize()
    fig_degree_assort()
    fig_shadow_mode()
    fig_lr_nomogram()
    print("DONE cycle-21")


if __name__ == "__main__":
    main()
