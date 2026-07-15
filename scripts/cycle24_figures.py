#!/usr/bin/env python3
"""Cycle-24 densify — lift 12 chapters from floor-17 toward >=18 (teal; original)."""
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


def fig_matrix_cond():
    """Ch00: condition number vs collinearity."""
    conds = []
    rhos = np.linspace(0, 0.98, 40)
    for r in rhos:
        C = np.array([[1, r], [r, 1]], dtype=float)
        conds.append(np.linalg.cond(C))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.semilogy(rhos, conds, color=TEAL, lw=2.4)
    ax.set_xlabel("correlation ρ between features")
    ax.set_ylabel("cond(Σ)")
    style_ax(ax, "Collinearity inflates condition number")
    ax.text(
        0.02,
        0.85,
        "Ill-conditioning hurts OLS stability.\nNot a causal DAG by itself.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_condition_number.png")


def fig_train_val_learning():
    """Preface: learning curve train vs val vs n."""
    n = np.array([50, 100, 200, 400, 800, 1600])
    tr = 0.12 + 0.25 * np.exp(-n / 250)
    va = 0.22 + 0.35 * np.exp(-n / 400)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(n, tr, color=TEAL, lw=2.3, marker="o", label="train error")
    ax.plot(n, va, color=GOLD, lw=2.3, marker="s", label="val error")
    ax.fill_between(n, tr, va, color=GRAY, alpha=0.25, label="gap")
    ax.set_xscale("log")
    ax.set_xlabel("training size n")
    ax.set_ylabel("error")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Learning curves: more n shrinks the gap")
    ax.text(
        0.98,
        0.85,
        "Curves diagnose fit capacity.\nPred ≠ cause.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_learning_curve_n.png")


def fig_parallel_coords_caution():
    """Ch02: parallel coordinates can overplot."""
    rng = np.random.default_rng(2)
    n = 40
    X = rng.normal(size=(n, 5))
    X[:15] += np.array([1.2, 0.2, -0.5, 0.1, 0.8])
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    dims = np.arange(5)
    for i in range(n):
        c = TEAL if i < 15 else GRAY
        ax.plot(dims, X[i], color=c, alpha=0.35, lw=1.0)
    ax.set_xticks(dims, ["NIHSS", "age", "gluc", "SBP", "onset"])
    ax.set_ylabel("z-score (synthetic)")
    style_ax(ax, "Parallel coordinates (overplot risk)")
    ax.text(
        0.02,
        0.08,
        "Teal bundle = one cluster sketch.\nAxes order changes perception—not etiology.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_parallel_coords.png")


def fig_ci_vs_pi():
    """Ch03: confidence band vs prediction interval."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 30)
    y = 1 + 0.5 * x + rng.normal(0, 0.8, size=x.shape)
    # OLS
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    grid = np.linspace(0, 10, 100)
    G = np.column_stack([np.ones_like(grid), grid])
    mean = G @ beta
    resid = y - X @ beta
    s = np.sqrt(np.sum(resid**2) / (len(x) - 2))
    # crude bands
    ci = 1.96 * s * 0.25 * np.ones_like(grid)
    pi = 1.96 * s * np.sqrt(1 + 0.05) * np.ones_like(grid)
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.scatter(x, y, c=TEAL, s=28, alpha=0.8, label="data")
    ax.plot(grid, mean, color=DEEP, lw=2.2, label="mean fit")
    ax.fill_between(grid, mean - ci, mean + ci, color=TEAL, alpha=0.25, label="CI (mean)")
    ax.fill_between(grid, mean - pi, mean + pi, color=GOLD, alpha=0.15, label="PI (new y)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    style_ax(ax, "Confidence band ≠ prediction interval")
    ax.text(
        0.98,
        0.15,
        "CI: mean line uncertainty.\nPI: new observation.\nNeither is causal alone.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_ci_vs_pi.png")


def fig_dbscan_eps():
    """Ch04: DBSCAN eps effect on n clusters / noise."""
    eps = np.linspace(0.2, 1.5, 14)
    # synthetic response
    n_clust = np.clip(12 * np.exp(-eps * 1.8) + 1, 1, 15)
    n_noise = np.clip(80 * np.exp(-eps * 2.5), 0, 80)
    fig, ax1 = plt.subplots(figsize=(7.8, 4.0))
    ax1.plot(eps, n_clust, color=TEAL, lw=2.3, marker="o", label="# clusters")
    ax1.set_xlabel("ε")
    ax1.set_ylabel("# clusters", color=TEAL)
    ax2 = ax1.twinx()
    ax2.plot(eps, n_noise, color=GOLD, lw=2.3, marker="s", label="# noise")
    ax2.set_ylabel("# noise points", color=GOLD)
    style_ax(ax1, "DBSCAN ε trades clusters vs noise (synthetic)")
    ax1.text(
        0.98,
        0.85,
        "ε is geometric—not a\nclinical subtype knob.",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_dbscan_eps.png")


def fig_precision_at_k():
    """Ch05: precision@k curve for retrieval."""
    k = np.arange(1, 51)
    # decreasing precision
    p = 0.9 * (1 / (1 + 0.08 * (k - 1))) + 0.05
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(k, p, color=TEAL, lw=2.4)
    ax.axhline(0.2, color=GOLD, ls="--", lw=1.4, label="prevalence baseline")
    ax.set_xlabel("k (top ranks)")
    ax.set_ylabel("precision@k")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Retrieval precision@k (synthetic ranking)")
    ax.text(
        0.98,
        0.55,
        "Ranking quality ≠ causal\nrelevance of terms.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_precision_at_k.png")


def fig_target_encoding_cv():
    """Ch06: target encoding leakage vs nested CV."""
    cats = np.arange(8)
    # naive TE uses full-data means (inflated)
    naive = 0.4 + 0.15 * np.sin(cats) + 0.2
    nested = 0.4 + 0.08 * np.sin(cats)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    w = 0.35
    ax.bar(cats - w / 2, naive, width=w, color=GOLD, edgecolor="white", label="full-data TE (leaky)")
    ax.bar(cats + w / 2, nested, width=w, color=TEAL, edgecolor="white", label="out-of-fold TE")
    ax.set_xlabel("category id")
    ax.set_ylabel("encoded value (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Target encoding must be out-of-fold")
    ax.text(
        0.02,
        0.9,
        "Leakage inflates validation.\nEncodings ≠ causal effects.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_target_encoding_cv.png")


def fig_ica_sources():
    """Ch07: ICA unmixing sketch — mixed signals vs recovered."""
    t = np.linspace(0, 8, 400)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t + 0.4))
    # mix
    x1 = 0.7 * s1 + 0.5 * s2
    x2 = 0.3 * s1 - 0.8 * s2
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 4.4), sharex=True)
    axes[0, 0].plot(t, x1, color=GOLD, lw=1.5)
    style_ax(axes[0, 0], "mixed channel 1")
    axes[0, 1].plot(t, x2, color=GOLD, lw=1.5)
    style_ax(axes[0, 1], "mixed channel 2")
    axes[1, 0].plot(t, s1, color=TEAL, lw=1.5)
    style_ax(axes[1, 0], "source-like recovery A")
    axes[1, 1].plot(t, s2, color=TEAL, lw=1.5)
    style_ax(axes[1, 1], "source-like recovery B")
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle("ICA teaching: mixtures vs independent sources (synthetic; original)", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.02, "Recovered components are statistical—not automatic clinical generators.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_ica_sources.png")


def fig_qq_normal():
    """Ch08: residual QQ plot."""
    rng = np.random.default_rng(7)
    r = rng.normal(0, 1, 120)
    r = np.sort(r)
    # theoretical quantiles
    n = len(r)
    probs = (np.arange(1, n + 1) - 0.5) / n
    # approx inverse normal via erfinv
    from math import sqrt

    try:
        theo = np.sqrt(2) * erfinv(2 * probs - 1)  # type: ignore
    except NameError:
        # rational approximation for erfinv
        def erfinv_approx(x):
            a = 0.147
            ln = np.log(1 - x**2)
            t = 2 / (np.pi * a) + ln / 2
            return np.sign(x) * np.sqrt(np.sqrt(t**2 - ln / a) - t)

        theo = np.sqrt(2) * erfinv_approx(2 * probs - 1)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(theo, r, s=18, c=TEAL, alpha=0.8)
    lim = max(abs(theo).max(), abs(r).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=GOLD, lw=1.6, ls="--")
    ax.set_xlabel("theoretical normal quantiles")
    ax.set_ylabel("ordered residuals")
    style_ax(ax, "Normal QQ plot of residuals")
    ax.text(
        0.02,
        0.92,
        "Heavy tails warn about\nOLS assumptions.\nNot causal proof.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_qq_normal.png")


def fig_batchnorm_moments():
    """Ch10: batch norm train vs eval moments."""
    steps = np.arange(0, 80)
    batch_mean = 0.2 * np.sin(steps / 5) + rng_noise(steps)
    running = np.cumsum(batch_mean) / (np.arange(1, len(batch_mean) + 1))
    # smooth running
    running = 0.9 * np.concatenate([[0], running[:-1]]) + 0.1 * batch_mean
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(steps, batch_mean, color=GOLD, lw=1.5, alpha=0.8, label="batch mean")
    ax.plot(steps, running, color=TEAL, lw=2.3, label="running mean (eval)")
    ax.set_xlabel("step")
    ax.set_ylabel("activation mean (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "BatchNorm: batch stats vs running stats")
    ax.text(
        0.98,
        0.15,
        "Train/eval mismatch if frozen wrong.\nNorm layers ≠ causal mechanisms.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_batchnorm_moments.png")


def rng_noise(steps):
    rng = np.random.default_rng(1)
    return 0.05 * rng.normal(size=len(steps))


def fig_masked_prediction():
    """Ch11: masked language/time modeling schematic rates."""
    mask_rate = np.array([0.05, 0.10, 0.15, 0.25, 0.40, 0.60])
    # downstream probe acc synthetic
    probe = 0.55 + 0.3 * np.exp(-((mask_rate - 0.15) ** 2) / 0.02) - 0.15 * mask_rate
    probe = np.clip(probe, 0.5, 0.95)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(mask_rate, probe, color=TEAL, lw=2.4, marker="o")
    ax.axvline(0.15, color=GOLD, ls="--", lw=1.4, label="common 15% mask")
    ax.set_xlabel("mask rate")
    ax.set_ylabel("downstream probe score (synth)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Masked prediction rate vs probe (synthetic)")
    ax.text(
        0.98,
        0.2,
        "Pretext metrics ≠ clinical utility.\nSSL geometry ≠ causation.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_mask_rate_probe.png")


def fig_spectrogram_toy():
    """Ch12: toy spectrogram for audio/vision time-frequency."""
    rng = np.random.default_rng(4)
    t = np.linspace(0, 1, 64)
    f = np.linspace(0, 1, 32)
    T, F = np.meshgrid(t, f)
    S = np.exp(-((F - 0.3 - 0.4 * T) ** 2) / 0.01) + 0.15 * rng.random(T.shape)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    im = ax.imshow(S, origin="lower", aspect="auto", cmap="YlGnBu", extent=[0, 1, 0, 1])
    ax.set_xlabel("time")
    ax.set_ylabel("frequency")
    style_ax(ax, "Toy spectrogram / time–frequency map")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.text(
        0.02,
        0.9,
        "Features for models; maps ≠ diagnosis\nwithout validated labels.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_spectrogram_toy.png")


def main() -> None:
    fig_matrix_cond()
    fig_train_val_learning()
    fig_parallel_coords_caution()
    fig_ci_vs_pi()
    fig_dbscan_eps()
    fig_precision_at_k()
    fig_target_encoding_cv()
    fig_ica_sources()
    fig_qq_normal()
    fig_batchnorm_moments()
    fig_masked_prediction()
    fig_spectrogram_toy()
    print("DONE cycle-24")


if __name__ == "__main__":
    main()
