#!/usr/bin/env python3
"""Cycle-15 densify figures — optional floor raise toward >=14 (teal; original)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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


def fig_label_smoothing():
    """Ch10/09: label smoothing softens one-hot targets."""
    eps_list = [0.0, 0.1, 0.2]
    K = 5
    true = 2
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    x = np.arange(K)
    width = 0.25
    colors = [GRAY, TEAL, GOLD]
    for i, (eps, c) in enumerate(zip(eps_list, colors)):
        y = np.full(K, eps / (K - 1))
        y[true] = 1 - eps
        ax.bar(x + (i - 1) * width, y, width, color=c, edgecolor="white",
               label=rf"$\varepsilon$={eps}")
    ax.set_xticks(x, [f"class {k}" for k in range(K)], fontsize=8)
    ax.set_ylabel("target mass")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Soft targets from label smoothing")

    ax = axes[1]
    # synthetic CE vs smoothed CE for overconfident logit
    conf = np.linspace(0.5, 0.999, 80)
    # true class prob = conf; CE = -log(conf)
    ce = -np.log(conf)
    # smoothed: target (1-e) on true, e/(K-1) on others; loss = -(1-e)log p - sum others
    e = 0.1
    # assume wrong mass uniform on remaining (1-conf)
    # simplified: only true term + uniform remainder
    sm = -(1 - e) * np.log(conf) - e * np.log((1 - conf + 1e-9) / (K - 1) + 1e-12)
    ax.plot(conf, ce, color=GRAY, lw=2.2, label="hard CE")
    ax.plot(conf, sm, color=TEAL, lw=2.4, label=rf"smoothed CE ($\varepsilon$={e})")
    ax.set_xlabel("predicted p(true class)")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Smoothing reduces overconfidence pressure")
    ax.text(0.98, 0.9, "Does not fix label noise alone.\nNot a causal regularizer.\nRecheck calibration after.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Label smoothing: soft targets vs hard one-hot (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_label_smoothing.png")


def fig_focal_loss():
    """Ch09: focal loss down-weights easy examples."""
    p_t = np.linspace(0.01, 0.99, 200)
    gamma_list = [0, 1, 2, 5]
    colors = [GRAY, GOLD, TEAL, DEEP]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    for g, c in zip(gamma_list, colors):
        fl = -((1 - p_t) ** g) * np.log(p_t)
        ax.plot(p_t, fl, color=c, lw=2.2, label=rf"$\gamma$={g}")
    ax.set_xlabel(r"$p_t$ (prob of true class)")
    ax.set_ylabel("focal loss")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, r"$FL=(1-p_t)^\gamma(-\log p_t)$")

    ax = axes[1]
    # weight (1-pt)^gamma
    for g, c in zip(gamma_list, colors):
        ax.plot(p_t, (1 - p_t) ** g, color=c, lw=2.2, label=rf"$\gamma$={g}")
    ax.set_xlabel(r"$p_t$")
    ax.set_ylabel(r"modulating factor $(1-p_t)^\gamma$")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Easy examples get near-zero weight")
    ax.text(0.02, 0.08, "Helps imbalance / easy negatives.\nStill tune threshold & calibrate.\nLoss design ≠ causation.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.suptitle("Focal loss: focusing parameter γ (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_focal_loss.png")


def fig_gae_lambda():
    """Ch13: GAE λ bias-variance tradeoff sketch."""
    T = 20
    t = np.arange(T)
    # synthetic TD residuals
    rng = np.random.default_rng(6)
    delta = rng.normal(0, 0.4, size=T)
    delta[5] = 1.2
    delta[12] = -0.8
    gamma = 0.99

    def gae(deltas, lam):
        A = np.zeros(T)
        gae_v = 0.0
        for i in reversed(range(T)):
            gae_v = deltas[i] + gamma * lam * gae_v
            A[i] = gae_v
        return A

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.stem(t, delta, linefmt=GRAY, markerfmt="o", basefmt=" ")
    ax.plot(t, delta, color=GRAY, lw=1.0, alpha=0.5)
    ax.set_xlabel("time t")
    ax.set_ylabel(r"TD residual $\delta_t$")
    style_ax(ax, "Synthetic TD residuals")

    ax = axes[1]
    for lam, c, lab in [(0.0, GRAY, r"$\lambda$=0 (1-step)"), (0.5, GOLD, r"$\lambda$=0.5"),
                        (0.95, TEAL, r"$\lambda$=0.95"), (1.0, DEEP, r"$\lambda$=1 (MC-like)")]:
        ax.plot(t, gae(delta, lam), color=c, lw=2.2, label=lab)
    ax.set_xlabel("time t")
    ax.set_ylabel("GAE advantage A_t")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "GAE mixes multi-step TD residuals")
    ax.text(0.98, 0.08, "λ trades bias vs variance\nlike eligibility traces.\nNot a clinical policy license.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Generalized Advantage Estimation (GAE) λ (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_gae_lambda.png")


def fig_temporal_cv_leak():
    """Ch16/01: temporal CV vs random shuffle leakage."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # time axis
    ax.annotate("", xy=(11.5, 1.2), xytext=(0.5, 1.2),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.text(6, 0.6, "calendar time →", ha="center", fontsize=9, color=SLATE)
    # random folds mixed
    colors = [TEAL, GOLD, DEEP, GRAY]
    rng = np.random.default_rng(1)
    for i in range(24):
        x = 0.6 + i * 0.45
        c = colors[i % 4]
        ax.add_patch(FancyBboxPatch((x, 2.2), 0.35, 1.2, boxstyle="round,pad=0.01",
                                    facecolor=c, edgecolor="none"))
    ax.text(6, 4.0, "WRONG: random K-fold (future → train)", ha="center",
            fontsize=11, fontweight="bold", color="#b91c1c")
    ax.text(6, 3.55, "colors = folds interleaved in time", ha="center", fontsize=8, color=SLATE)
    style_ax(ax, "Shuffled CV on longitudinal data")
    ax.set_title("Shuffled CV on longitudinal data", fontsize=12, fontweight="bold", color=INK)

    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.annotate("", xy=(11.5, 1.2), xytext=(0.5, 1.2),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.text(6, 0.6, "calendar time →", ha="center", fontsize=9, color=SLATE)
    # expanding windows
    segs = [(0.5, 3.0, TEAL, "train"), (3.6, 1.5, GOLD, "val"), (5.2, 3.0, TEAL, "train"),
            (8.3, 1.5, GOLD, "val"), (9.9, 1.5, DEEP, "test")]
    # simpler: train | val blocks forward
    blocks = [
        (0.5, 4.5, TEAL, "TRAIN (past)"),
        (5.2, 2.0, GOLD, "VAL"),
        (7.4, 2.0, DEEP, "TEST (future)"),
    ]
    for x, w, c, lab in blocks:
        ax.add_patch(FancyBboxPatch((x, 2.2), w, 1.4, boxstyle="round,pad=0.02",
                                    facecolor=c, edgecolor="none", alpha=0.9))
        ax.text(x + w / 2, 2.9, lab, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.text(6, 4.2, "RIGHT: forward-chaining / temporal split", ha="center",
            fontsize=11, fontweight="bold", color=DEEP)
    ax.text(6, 3.7, "never train on times after validation", ha="center", fontsize=8, color=SLATE)
    style_ax(ax, "Time-respecting evaluation")
    ax.set_title("Time-respecting evaluation", fontsize=12, fontweight="bold", color=INK)
    fig.suptitle("Temporal leakage: random folds vs forward splits (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_temporal_cv.png")


def fig_calibration_slope():
    """Ch09/08: calibration slope and intercept."""
    rng = np.random.default_rng(15)
    n = 800
    z = rng.normal(0, 1, size=n)
    p_true = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p_true)
    # overconfident model
    p_raw = 1 / (1 + np.exp(-1.8 * z))
    p_raw = np.clip(p_raw, 1e-4, 1 - 1e-4)
    logit = np.log(p_raw / (1 - p_raw))
    # fit slope/intercept: y ~ bernoulli(logistic(a + b logit))
    X = np.c_[np.ones(n), logit]
    w = np.zeros(2)
    for _ in range(30):
        eta = X @ w
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-6
        grad = X.T @ (p - y)
        H = X.T @ (X * W[:, None])
        w = w - np.linalg.solve(H + 1e-4 * np.eye(2), grad)
    a, b = w
    p_cal = 1 / (1 + np.exp(-(a + b * logit)))

    def rel(p, y, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        cx, ox = [], []
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
            if m.sum() < 15:
                continue
            cx.append(p[m].mean())
            ox.append(y[m].mean())
        return np.array(cx), np.array(ox)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    cx, ox = rel(p_raw, y)
    ax.plot(cx, ox, "o-", color=GRAY, lw=2, label="raw")
    cx2, ox2 = rel(p_cal, y)
    ax.plot(cx2, ox2, "o-", color=TEAL, lw=2.2, label="after slope/intercept")
    ax.plot([0, 1], [0, 1], ":", color=INK, lw=1.2)
    ax.set_xlabel("predicted p")
    ax.set_ylabel("observed rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Reliability before/after logistic recal")

    ax = axes[1]
    ax.bar(["intercept a", "slope b"], [a, b], color=[GOLD, TEAL], edgecolor="white")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.axhline(1, color=GRAY, ls="--", lw=1.0)
    ax.text(0, a + 0.05 * np.sign(a + 1e-6), f"{a:.2f}", ha="center", fontweight="bold")
    ax.text(1, b + 0.05, f"{b:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("logistic recalibration coef")
    style_ax(ax, r"Ideal: $a\approx 0$, $b\approx 1$")
    ax.text(0.5, 0.08, f"b<1 ⇒ overconfident raw scores\nFit on held-out cal set only.",
            transform=ax.transAxes, ha="center", fontsize=8, color=SLATE)
    fig.suptitle("Calibration slope & intercept (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_calibration_slope.png")


def fig_graphsage_sampling():
    """Ch15: GraphSAGE neighborhood sampling fanout."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    # target node
    ax.scatter([5], [1.5], s=200, c=DEEP, zorder=5)
    ax.text(5, 1.0, "target v", ha="center", fontsize=9, color=INK)
    # layer 1 neighbors sampled
    l1 = [(2.5, 3.8), (5, 4.2), (7.5, 3.8)]
    for x, y in l1:
        ax.plot([5, x], [1.5, y], color=TEAL, lw=1.5)
        ax.scatter([x], [y], s=120, c=TEAL, zorder=5)
    ax.text(5, 5.0, "sample k1 neighbors (layer 1)", ha="center", fontsize=9, color=TEAL)
    # layer 2
    l2 = [(1.2, 6.5), (2.8, 6.8), (4.2, 6.5), (5.8, 6.8), (7.2, 6.5), (8.8, 6.8)]
    parents = [0, 0, 1, 1, 2, 2]
    for i, (x, y) in enumerate(l2):
        px, py = l1[parents[i]]
        ax.plot([px, x], [py, y], color=GOLD, lw=1.2)
        ax.scatter([x], [y], s=70, c=GOLD, zorder=5)
    ax.text(5, 7.5, "then k2 neighbors each (layer 2)", ha="center", fontsize=9, color=GOLD)
    style_ax(ax, "Fixed fanout sampling")
    ax.set_title("Fixed fanout sampling", fontsize=12, fontweight="bold", color=INK)

    ax = axes[1]
    depths = np.array([1, 2, 3])
    # nodes touched ≈ k + k^2 + k^3 for k=5
    k = 5
    touched = np.array([k, k + k**2, k + k**2 + k**3])
    ax.bar(depths, touched, color=TEAL, edgecolor="white")
    for d, v in zip(depths, touched):
        ax.text(d, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_xlabel("aggregator depth")
    ax.set_ylabel(f"nodes touched (fanout k={k}, teaching)")
    style_ax(ax, "Cost grows with depth × fanout")
    ax.text(0.98, 0.9, "Inductive: embed new nodes\nvia sampled neighbors.\nGraph proximity ≠ causation.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("GraphSAGE neighborhood sampling (schematic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_graphsage_sampling.png")


def main():
    fig_label_smoothing()
    fig_focal_loss()
    fig_gae_lambda()
    fig_temporal_cv_leak()
    fig_calibration_slope()
    fig_graphsage_sampling()
    print("DONE cycle-15 figures")


if __name__ == "__main__":
    main()
