#!/usr/bin/env python3
"""Cycle-17 densify — lift remaining 6 chapters to floor >=14."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


def fig_bias_variance_tradeoff_curve():
    """Ch01: classical bias-variance vs model complexity."""
    c = np.linspace(0.2, 5, 100)
    bias2 = 2.5 * np.exp(-c / 1.2)
    var = 0.15 * c ** 1.4
    noise = 0.35 * np.ones_like(c)
    total = bias2 + var + noise
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(c, bias2, color=GOLD, lw=2.3, label=r"bias$^2$")
    ax.plot(c, var, color=TEAL, lw=2.3, label="variance")
    ax.plot(c, noise, color=GRAY, lw=1.8, ls="--", label="irreducible")
    ax.plot(c, total, color=DEEP, lw=2.8, label="expected test error")
    i = total.argmin()
    ax.axvline(c[i], color=INK, ls=":", lw=1.2)
    ax.text(c[i] + 0.1, total.max() * 0.85, "sweet spot", fontsize=9, color=INK)
    ax.set_xlabel("model complexity")
    ax.set_ylabel("error")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Bias–variance tradeoff (teaching sketch)")
    ax.text(0.98, 0.08, "Complexity is not always 'more layers'.\nRegularization moves the curves.\nPrediction error ≠ causation.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_bias_variance_curve.png")


def fig_colorblind_palette():
    """Ch02: sequential vs rainbow colormap caution on clinical heatmap."""
    rng = np.random.default_rng(4)
    M = rng.normal(size=(12, 12))
    M = (M + M.T) / 2
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    im = ax.imshow(M, cmap="jet")
    ax.set_title("Rainbow (jet) — hard to order & not CB-safe", fontsize=11, fontweight="bold", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    im = ax.imshow(M, cmap="YlGnBu")
    ax.set_title("Sequential teal — ordered & print-safer", fontsize=11, fontweight="bold", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        a.set_facecolor("#fafafa")
    fig.suptitle("Colormap hygiene for clinical heatmaps (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "Prefer sequential/diverging perceptually uniform maps; rainbow can invent false boundaries.",
             ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_colormap_hygiene.png")


def fig_support_confidence_plane():
    """Ch05: support–confidence plane for rules with lift contours."""
    # synthetic rules
    rng = np.random.default_rng(8)
    n = 40
    sup = rng.uniform(0.02, 0.35, size=n)
    conf = rng.uniform(0.2, 0.95, size=n)
    # prevalence of consequent ~0.25
    prev = 0.25
    lift = conf / prev
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    sc = ax.scatter(sup, conf, c=lift, cmap="YlGnBu", s=55, edgecolors="white", vmin=0.5, vmax=3.5)
    ax.axhline(0.6, color=GOLD, ls="--", lw=1.2, label="min confidence")
    ax.axvline(0.08, color=TEAL, ls="--", lw=1.2, label="min support")
    # lift=1 line is conf=prev
    ax.axhline(prev, color=GRAY, ls=":", lw=1.3, label=f"lift=1 (conf=prev={prev})")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("lift")
    ax.set_xlabel("support")
    ax.set_ylabel("confidence")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_ax(ax, "Association rules in support–confidence space")
    ax.text(0.02, 0.92, "High lift + rare support may be noise.\nAssociation ≠ causation.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_support_confidence.png")


def fig_boxcox_family():
    """Ch06: Box-Cox / power transform family sketch."""
    x = np.linspace(0.2, 4, 200)
    lambdas = [-1, 0, 0.5, 1]
    colors = [GRAY, TEAL, GOLD, DEEP]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    for lam, c in zip(lambdas, colors):
        if abs(lam) < 1e-9:
            y = np.log(x)
            lab = r"$\lambda=0$ (log)"
        else:
            y = (x ** lam - 1) / lam
            lab = rf"$\lambda={lam}$"
        ax.plot(x, y, color=c, lw=2.2, label=lab)
    ax.set_xlabel("x > 0")
    ax.set_ylabel(r"Box–Cox $y^{(\lambda)}$")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Power transform family")

    ax = axes[1]
    rng = np.random.default_rng(1)
    raw = rng.lognormal(mean=1.0, sigma=0.6, size=400)
    # log
    logged = np.log(raw)
    ax.hist(raw, bins=30, density=True, color=GRAY, alpha=0.55, label="raw skewed")
    ax.hist(logged, bins=30, density=True, color=TEAL, alpha=0.65, label="log(x)")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Skew reduction (fit λ on train only)")
    ax.text(0.98, 0.9, "Never fit λ on full cohort.\nTransform is for modeling geometry,\nnot causal rescaling of biology.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Box–Cox / log transforms for numeric features (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_boxcox.png")


def fig_flash_attention_io():
    """Ch12/14-adjacent: attention memory IO teaching (FlashAttention idea)."""
    # bar: naive O(N^2) materialize vs tiled
    Ns = np.array([1, 2, 4, 8, 16]) * 1024
    # relative memory units for S matrix ~ N^2
    naive = (Ns / 1024) ** 2
    flash = Ns / 1024 * 0.8  # linear-ish IO caricature
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(Ns, naive, "o-", color=GOLD, lw=2.4, label="naive score matrix ~ N²")
    ax.plot(Ns, flash, "s-", color=TEAL, lw=2.4, label="tiled IO (teaching)")
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("relative memory / IO units")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Why exact attention needs careful IO")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.5, 2.0), 2.5, 2.2, boxstyle="round,pad=0.04", facecolor=GOLD, edgecolor="none"))
    ax.text(1.75, 3.1, "Q,K,V\ntiles", ha="center", va="center", color=INK, fontsize=11, fontweight="bold")
    ax.add_patch(FancyBboxPatch((3.8, 2.0), 2.5, 2.2, boxstyle="round,pad=0.04", facecolor=TEAL, edgecolor="none"))
    ax.text(5.05, 3.1, "SRAM\non-chip", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.1, 2.0), 2.5, 2.2, boxstyle="round,pad=0.04", facecolor=DEEP, edgecolor="none"))
    ax.text(8.35, 3.1, "Output\naccum", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(3.7, 3.1), xytext=(3.1, 3.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(7.0, 3.1), xytext=(6.4, 3.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.text(5, 1.2, "Same math as softmax attention; different memory traffic",
            ha="center", fontsize=9, color=SLATE)
    ax.text(5, 0.5, "Faster ≠ more causal. Long notes still need retrieval hygiene.",
            ha="center", fontsize=8, color=SLATE)
    style_ax(ax, "Tiling sketch (conceptual)")
    ax.set_title("Tiling sketch (conceptual)", fontsize=12, fontweight="bold", color=INK)
    fig.suptitle("FlashAttention-style IO: exact attention without full N×N materialization (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_flash_attention_io.png")


def fig_glossary_nnt():
    """Ch18: NNT / ARR from risk difference (decision utility vocab)."""
    p0 = 0.12
    p1 = np.linspace(0.02, 0.11, 80)
    arr = p0 - p1
    nnt = 1 / arr
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(p1, arr, color=TEAL, lw=2.5)
    ax.set_xlabel("risk with intervention p1")
    ax.set_ylabel(f"ARR = p0 − p1  (p0={p0})")
    style_ax(ax, "Absolute risk reduction")

    ax = axes[1]
    ax.plot(p1, nnt, color=DEEP, lw=2.5)
    ax.set_xlabel("risk with intervention p1")
    ax.set_ylabel("NNT = 1 / ARR")
    ax.set_ylim(0, 80)
    style_ax(ax, "Number needed to treat")
    ax.text(0.98, 0.9, "NNT needs absolute risks,\nnot OR/RR alone.\nModel scores need calibration\nbefore ARR claims.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Glossary: ARR and NNT from absolute risks (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_arr_nnt.png")


def main():
    fig_bias_variance_tradeoff_curve()
    fig_colorblind_palette()
    fig_support_confidence_plane()
    fig_boxcox_family()
    fig_flash_attention_io()
    fig_glossary_nnt()
    print("DONE cycle-17")


if __name__ == "__main__":
    main()
