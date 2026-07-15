#!/usr/bin/env python3
"""Cycle-20 densify — raise floor toward >=16 (batch of 8)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
TEAL, DEEP, INK, GOLD, GRAY, SLATE, SOFT = "#0d9488", "#0f766e", "#0f172a", "#c9a227", "#94a3b8", "#64748b", "#ecfeff"

def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig); print("WROTE", name)

def style_ax(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values(): s.set_color("#cbd5e1")

def fig_permutation_importance():
    rng = np.random.default_rng(2)
    names = ["NIHSS", "age", "glucose", "scanner", "noise"]
    base = 0.80
    drops = np.array([0.12, 0.05, 0.03, 0.09, 0.00])
    # bootstrap spread
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    y = np.arange(len(names))
    ax.barh(y, drops, color=[TEAL if d < 0.08 else GOLD for d in drops], edgecolor="white", xerr=0.015*rng.random(5)+0.01, ecolor=GRAY)
    ax.set_yticks(y, names)
    ax.set_xlabel("ΔAUROC when feature permuted (synthetic)")
    style_ax(ax, "Permutation importance (with caution)")
    ax.text(0.98, 0.1, "Correlated features share credit.\nImportance ≠ causation.\nScanner high = leakage risk.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout(); save(fig, "ml_fig_perm_importance.png")

def fig_pr_auc_chance():
    prev = 0.1
    r = np.linspace(0, 1, 100)
    # chance PR is horizontal at prevalence for random ranking-ish
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    # synthetic model
    prec = prev + (1-prev) * (1 - r) ** 1.5 * 0.7 + 0.05
    prec = np.clip(prec, prev, 1)
    ax.plot(r, prec, color=TEAL, lw=2.5, label="model")
    ax.axhline(prev, color=GOLD, ls="--", lw=1.8, label=f"chance ≈ prevalence ({prev})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "PR curve vs chance baseline")
    ax.text(0.98, 0.9, "Always draw the prevalence line.\nPR-AUC more honest when rare.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.tight_layout(); save(fig, "ml_fig_pr_chance.png")

def fig_wasserstein_hist():
    rng = np.random.default_rng(6)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0.8, 1.1, 500)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    axes[0].hist(a, bins=30, density=True, alpha=0.6, color=TEAL, label="train")
    axes[0].hist(b, bins=30, density=True, alpha=0.6, color=GOLD, label="serve")
    axes[0].legend(frameon=False, fontsize=8)
    style_ax(axes[0], "Shifted feature histograms")
    # 1D Wasserstein approx via sorted difference mean
    sa, sb = np.sort(a), np.sort(b)
    # resample equal
    qs = np.linspace(0, 1, 200)
    qa = np.quantile(a, qs); qb = np.quantile(b, qs)
    axes[1].plot(qs, qa, color=TEAL, lw=2, label="train quantiles")
    axes[1].plot(qs, qb, color=GOLD, lw=2, label="serve quantiles")
    axes[1].fill_between(qs, qa, qb, color=GRAY, alpha=0.3)
    w1 = np.mean(np.abs(qa - qb))
    axes[1].legend(frameon=False, fontsize=8)
    style_ax(axes[1], f"Quantile functions (W1≈{w1:.2f})")
    axes[1].text(0.02, 0.08, "W1 measures distribution shift.\nNot a causal site effect.",
                 transform=axes[1].transAxes, fontsize=8, color=SLATE)
    fig.suptitle("1-Wasserstein intuition via quantiles (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout(); save(fig, "ml_fig_wasserstein1.png")

def fig_diffusion_forward():
    t = np.linspace(0, 1, 6)
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(200, 2)) * 0.4 + np.array([0.0, 0.0])
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.4))
    for i, ax in enumerate(axes):
        noise = rng.normal(size=x0.shape)
        xt = np.sqrt(1 - t[i]) * x0 + np.sqrt(t[i]) * noise
        ax.scatter(xt[:, 0], xt[:, 1], s=6, c=TEAL, alpha=0.6)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t[i]:.1f}", fontsize=9, color=INK)
        ax.set_facecolor("#fafafa")
    fig.suptitle("Diffusion forward process: data → noise (synthetic; original)", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "ml_fig_diffusion_forward.png")

def fig_hierarchical_shrinkage():
    rng = np.random.default_rng(8)
    sites = np.array([5, 12, 20, 40, 80, 200])
    raw = 0.3 + rng.normal(0, 0.15, size=len(sites))
    # shrink toward grand mean 0.25
    m = 20
    shrunk = (sites * raw + m * 0.25) / (sites + m)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.scatter(sites, raw, s=70, c=GOLD, label="raw site rate", zorder=3)
    ax.scatter(sites, shrunk, s=70, c=TEAL, label="EB/hierarchical shrink", zorder=3)
    for i in range(len(sites)):
        ax.plot([sites[i], sites[i]], [raw[i], shrunk[i]], color=GRAY, lw=1.2)
    ax.axhline(0.25, color=DEEP, ls="--", lw=1.2, label="global mean")
    ax.set_xscale("log")
    ax.set_xlabel("site n")
    ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Small sites shrink harder")
    ax.text(0.98, 0.1, "Shrinkage stabilizes estimates.\nNot a causal site ranking.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout(); save(fig, "ml_fig_hierarchical_shrink.png")

def fig_attention_mask_types():
    n = 8
    causal = np.tril(np.ones((n, n)))
    bidir = np.ones((n, n))
    window = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - 2), min(n, i + 3)):
            window[i, j] = 1
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for ax, M, title in [(axes[0], causal, "causal"), (axes[1], bidir, "bidirectional"), (axes[2], window, "sliding window")]:
        ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Attention mask patterns (original)", color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.text(0.5, -0.02, "Mask = allowed positions. Pattern choice is inductive bias—not clinical causation.",
             ha="center", fontsize=8, color=SLATE)
    fig.tight_layout(); save(fig, "ml_fig_attention_masks.png")

def fig_smote_caution():
    rng = np.random.default_rng(3)
    maj = rng.normal([0, 0], 0.6, (80, 2))
    mino = rng.normal([1.5, 1.2], 0.25, (12, 2))
    # synthetic SMOTE-like
    synth = []
    for i in range(20):
        a, b = mino[rng.integers(0, len(mino))], mino[rng.integers(0, len(mino))]
        synth.append(a + rng.random() * (b - a))
    synth = np.array(synth)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    axes[0].scatter(maj[:, 0], maj[:, 1], c=GRAY, s=18, alpha=0.6, label="majority")
    axes[0].scatter(mino[:, 0], mino[:, 1], c=TEAL, s=40, label="minority")
    axes[0].legend(frameon=False, fontsize=8)
    style_ax(axes[0], "Before oversampling")
    axes[1].scatter(maj[:, 0], maj[:, 1], c=GRAY, s=18, alpha=0.5)
    axes[1].scatter(mino[:, 0], mino[:, 1], c=TEAL, s=40, label="real minority")
    axes[1].scatter(synth[:, 0], synth[:, 1], c=GOLD, s=30, marker="x", label="SMOTE-like")
    axes[1].legend(frameon=False, fontsize=8)
    style_ax(axes[1], "After synthetic minority")
    axes[1].text(0.02, 0.08, "Only inside train folds.\nRecalibrate probabilities.\nSynthetic ≠ new patients.",
                 transform=axes[1].transAxes, fontsize=8, color=SLATE)
    fig.suptitle("SMOTE-style oversampling caution (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout(); save(fig, "ml_fig_smote_caution.png")

def fig_early_stopping_patience():
    steps = np.arange(0, 80)
    tr = 1.5 * np.exp(-steps / 25) + 0.2
    va = 1.4 * np.exp(-steps / 28) + 0.28 + 0.002 * np.maximum(0, steps - 35)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(steps, tr, color=TEAL, lw=2.3, label="train loss")
    ax.plot(steps, va, color=GOLD, lw=2.3, label="val loss")
    best = int(np.argmin(va))
    ax.axvline(best, color=DEEP, ls="--", lw=1.3, label=f"best val @ {best}")
    ax.axvspan(best, best + 10, color=GRAY, alpha=0.2, label="patience window")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Early stopping restores the best checkpoint")
    ax.text(0.98, 0.9, "Stop when val stops improving.\nMandatory on small n.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.tight_layout(); save(fig, "ml_fig_early_stop_patience.png")

def main():
    fig_permutation_importance(); fig_pr_auc_chance(); fig_wasserstein_hist()
    fig_diffusion_forward(); fig_hierarchical_shrinkage(); fig_attention_mask_types()
    fig_smote_caution(); fig_early_stopping_patience()
    print("DONE cycle-20")
if __name__ == "__main__":
    main()
