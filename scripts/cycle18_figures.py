#!/usr/bin/env python3
"""Cycle-18 densify — push floor toward >=15 (teal; original)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

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


def fig_mutual_info_filter():
    """Ch06: MI filter scores vs spurious correlation."""
    rng = np.random.default_rng(10)
    names = ["NIHSS", "age", "glucose", "zip code", "scanner ID", "true noise"]
    # MI-like scores (synthetic)
    mi = np.array([0.22, 0.12, 0.09, 0.18, 0.16, 0.01])
    # downstream true utility (partial)
    util = np.array([0.20, 0.10, 0.07, 0.02, 0.01, 0.0])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    colors = [TEAL if u > 0.05 else GOLD if m > 0.1 else GRAY for m, u in zip(mi, util)]
    ax.barh(names, mi, color=colors, edgecolor="white")
    ax.set_xlabel("mutual information with y (synthetic)")
    style_ax(ax, "Filter scores can prefer proxies")
    ax.text(0.98, 0.08, "Gold bars: high MI, low utility\n(site/scanner leakage risk)",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)

    ax = axes[1]
    ax.scatter(mi, util, s=80, c=TEAL, edgecolors="white")
    for n, x, y in zip(names, mi, util):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("MI filter score")
    ax.set_ylabel("downstream ΔAUROC (synth)")
    style_ax(ax, "MI ≠ task utility")
    ax.text(0.02, 0.9, "Nest filters in CV.\nAssociation ≠ causation.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.suptitle("Mutual-information feature filter caution (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_mi_filter.png")


def fig_competing_risks():
    """Ch03/08: KM treats other deaths as censor vs CIF competing risks."""
    t = np.linspace(0, 365, 100)
    # event1 cumulative incidence rising
    cif1 = 1 - np.exp(-t / 400)
    cif2 = 1 - np.exp(-t / 550)
    # naive 1-KM treating other as censor overestimates
    km_naive = 1 - np.exp(-t / 280)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(t, cif1, color=TEAL, lw=2.4, label="CIF event A (stroke)")
    ax.plot(t, cif2, color=GOLD, lw=2.4, label="CIF event B (death other)")
    ax.plot(t, cif1 + cif2, color=DEEP, lw=2.0, ls="--", label="sum CIFs")
    ax.set_xlabel("days")
    ax.set_ylabel("cumulative incidence")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Competing risks: two causes")

    ax = axes[1]
    ax.plot(t, km_naive, color="#ef4444", lw=2.3, label="1−KM (other=censor) — optimistic")
    ax.plot(t, cif1, color=TEAL, lw=2.4, label="CIF event A")
    ax.set_xlabel("days")
    ax.set_ylabel("risk of A")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Why KM can overstate cause-specific risk")
    ax.text(0.98, 0.08, "Use CIF / Fine-Gray when\ncompeting deaths matter.\nCurves are risks, not effects.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Competing risks vs naive Kaplan–Meier (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_competing_risks.png")


def fig_label_propagation():
    """Ch15/04: label propagation on graph sketch."""
    rng = np.random.default_rng(2)
    # two blobs partially labeled
    A = rng.normal([-1.2, 0], 0.35, size=(40, 2))
    B = rng.normal([1.2, 0], 0.35, size=(40, 2))
    X = np.vstack([A, B])
    y = np.array([-1] * 80)
    y[0] = 0
    y[1] = 0
    y[40] = 1
    y[41] = 1
    # simple kNN soft labels after "propagation" caricature
    from collections import defaultdict
    # distance weighted vote from seeds
    seeds = np.where(y >= 0)[0]
    soft = np.zeros(80)
    for i in range(80):
        d = np.linalg.norm(X[i] - X[seeds], axis=1) + 1e-6
        w = 1 / d ** 2
        labs = y[seeds]
        soft[i] = (w * labs).sum() / w.sum()

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=28, c=GRAY, alpha=0.5, label="unlabeled")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=90, c=TEAL, edgecolors="white", label="seed class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=90, c=GOLD, edgecolors="white", label="seed class 1")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Few labels on a similarity graph")

    ax = axes[1]
    sc = ax.scatter(X[:, 0], X[:, 1], c=soft, cmap="YlGnBu", s=40, edgecolors="white")
    fig.colorbar(sc, ax=ax, label="propagated score")
    style_ax(ax, "After label propagation (caricature)")
    ax.text(0.98, 0.08, "Graph edges encode similarity,\nnot infection or causation.\nSensitive to graph construction.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Semi-supervised label propagation sketch (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_label_propagation.png")


def fig_sinkhorn_ot():
    """Ch16: Sinkhorn optimal transport coupling sketch."""
    rng = np.random.default_rng(5)
    n, m = 6, 6
    a = np.ones(n) / n
    b = np.ones(m) / m
    # cost matrix
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, m)
    C = (xs[:, None] - ys[None, :]) ** 2
    # Sinkhorn
    eps = 0.05
    K = np.exp(-C / eps)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(80):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = np.diag(u) @ K @ np.diag(v)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    im = ax.imshow(C, cmap="YlOrBr")
    ax.set_xlabel("target bin")
    ax.set_ylabel("source bin")
    fig.colorbar(im, ax=ax, fraction=0.046)
    style_ax(ax, "Cost matrix C (squared distance)")

    ax = axes[1]
    im = ax.imshow(P, cmap="YlGnBu")
    ax.set_xlabel("target bin")
    ax.set_ylabel("source bin")
    fig.colorbar(im, ax=ax, fraction=0.046)
    style_ax(ax, "Sinkhorn coupling P")
    ax.text(0.98, 0.08, "OT aligns distributions\n(domain adaptation).\nCoupling ≠ causal map.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Entropy-regularized optimal transport (Sinkhorn) (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_sinkhorn_ot.png")


def fig_rope_angles():
    """Ch12: RoPE rotation intuition."""
    theta = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.2)
    # two positions
    for pos, ang, c, lab in [(0, 0.4, TEAL, "pos t"), (1, 0.4 + 0.7, GOLD, "pos t+Δ")]:
        ax.arrow(0, 0, np.cos(ang), np.sin(ang), head_width=0.08, color=c, length_includes_head=True, lw=2)
        ax.scatter([np.cos(ang)], [np.sin(ang)], c=c, s=60, zorder=5, label=lab)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "RoPE rotates query/key by position")

    ax = axes[1]
    deltas = np.arange(0, 32)
    # relative score ~ cos(delta * omega)
    omega = 0.3
    ax.plot(deltas, np.cos(deltas * omega), color=TEAL, lw=2.4)
    ax.set_xlabel("relative distance Δ")
    ax.set_ylabel("relative rotation cos(Δ·ω)")
    style_ax(ax, "Relative geometry depends on Δ only")
    ax.text(0.98, 0.9, "Helps length extrapolation\nvs absolute PE.\nStill not clinical causality.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Rotary positional embeddings (RoPE) intuition (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_rope.png")


def fig_ece_reliability_bins():
    """Ch09/18: ECE bin contributions."""
    rng = np.random.default_rng(7)
    n = 1000
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-1.6 * z))
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    edges = np.linspace(0, 1, 11)
    conf, acc, mass = [], [], []
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf.append(p[m].mean())
        acc.append(y[m].mean())
        mass.append(m.mean())
    conf, acc, mass = map(np.array, (conf, acc, mass))
    contrib = mass * np.abs(acc - conf)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ":", color=INK)
    ax.bar(conf, acc, width=0.08, color=TEAL, edgecolor="white", alpha=0.85, label="bin accuracy")
    ax.plot(conf, conf, "o", color=GOLD, label="bin confidence")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Reliability diagram (equal-width bins)")

    ax = axes[1]
    ax.bar(np.arange(len(contrib)), contrib, color=DEEP, edgecolor="white")
    ax.set_xlabel("bin index")
    ax.set_ylabel("mass × |acc − conf|")
    ece = contrib.sum()
    ax.set_title(f"ECE contributions (ECE≈{ece:.3f})", fontsize=12, fontweight="bold", color=INK)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")
    ax.text(0.98, 0.9, "ECE is one scalar; inspect tails.\nRecalibrate on held-out data.\nNot a causal metric.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Expected calibration error (ECE) bin decomposition (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_ece_bins.png")


def main():
    fig_mutual_info_filter()
    fig_competing_risks()
    fig_label_propagation()
    fig_sinkhorn_ot()
    fig_rope_angles()
    fig_ece_reliability_bins()
    print("DONE cycle-18")


if __name__ == "__main__":
    main()
