#!/usr/bin/env python3
"""Cycle-85/86/87: further novel scientific teal teaching figures (ML densify)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch, Ellipse, Polygon, Arc

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, DEEP, INK, GOLD, SOFT, SLATE, ROSE, MINT = (
    "#0d9488", "#0f766e", "#0f172a", "#c9a227", "#ecfeff", "#64748b", "#e11d48", "#14b8a6"
)

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


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", name)
    return p


def style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def box(ax, x, y, w, h, text, fc=TEAL, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fc, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold")


# ─── Cycle 85 ───
def c85_00():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.linspace(-3, 3, 400)
    ax.plot(x, np.maximum(0, x), color=TEAL, lw=2.5, label="ReLU")
    ax.plot(x, np.where(x > 0, x, 0.1 * x), color=GOLD, lw=2, label="Leaky ReLU")
    ax.plot(x, 1 / (1 + np.exp(-x)), color=SLATE, lw=2, label="sigmoid")
    ax.plot(x, np.tanh(x), color=ROSE, lw=1.8, label="tanh")
    style(ax, "Activation functions (shape matters for gradients)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel("z"); ax.set_ylabel("a(z)")
    fig.tight_layout(); return save(fig, "ml_fig_c85_00.png")


def c85_01():
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    for i, (lab, c) in enumerate([("Aim", TEAL), ("Data map", DEEP), ("Analysis", GOLD),
                                   ("Report", MINT), ("Limits", SLATE)]):
        box(ax, 0.3 + i * 1.9, 1.3, 1.7, 1.4, lab, fc=c, fs=10, tc="white" if c not in (GOLD, MINT) else INK)
        if i < 4:
            ax.annotate("", xy=(0.3 + (i + 1) * 1.9, 2.0), xytext=(0.3 + i * 1.9 + 1.7, 2.0),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.text(5, 0.5, "Protocol spine for transparent ML reporting", ha="center",
            fontsize=10, color=DEEP, fontweight="bold")
    style(ax, "Preface: five-box study protocol spine")
    return save(fig, "ml_fig_c85_01.png")


def c85_02():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    n = np.array([50, 100, 200, 500, 1000, 2000])
    train = 0.05 + 0.0 * n
    test = 0.35 * np.exp(-n / 400) + 0.08
    ax.plot(n, train, "o-", color=TEAL, lw=2, label="train error")
    ax.plot(n, test, "s-", color=GOLD, lw=2, label="test error")
    style(ax, "Learning curves: more data shrinks generalization gap")
    ax.set_xlabel("n train"); ax.set_ylabel("error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c85_02.png")


def c85_03():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 80); y = 0.7 * x + rng.normal(0, 0.5, 80)
    ax1.scatter(x, y, c=TEAL, s=24, alpha=0.8)
    # overplot density with hex
    ax2.hexbin(x, y, gridsize=12, cmap="viridis", mincnt=1)
    style(ax1, "Scatter (overplot risk)"); style(ax2, "Hexbin density")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
    fig.suptitle("Visualization: density views when points overlap", fontsize=11,
                 fontweight="bold", color=INK, y=1.02)
    fig.tight_layout(); return save(fig, "ml_fig_c85_03.png")


def c85_04():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.linspace(0, 10, 200)
    for mu, s, c, lab in [(3, 0.8, TEAL, "N(3,0.8²)"), (6, 1.2, GOLD, "N(6,1.2²)"),
                          (5, 2.5, SLATE, "N(5,2.5²)")]:
        ax.plot(x, 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / s) ** 2),
                color=c, lw=2, label=lab)
    style(ax, "Gaussian family: location and scale control density shape")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel("x"); ax.set_ylabel("density")
    fig.tight_layout(); return save(fig, "ml_fig_c85_04.png")


def c85_05():
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    rng = np.random.default_rng(7)
    # two moons-ish
    t = np.linspace(0, np.pi, 80)
    a = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.08, (80, 2))
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)] + rng.normal(0, 0.08, (80, 2))
    ax.scatter(a[:, 0], a[:, 1], c=TEAL, s=18, label="cluster 1")
    ax.scatter(b[:, 0], b[:, 1], c=GOLD, s=18, label="cluster 2")
    # kmeans wrong centers
    ax.plot([0.5, 0.5], [0.2, 0.6], "X", color=ROSE, ms=12, label="k-means trap centers")
    style(ax, "Non-convex clusters defeat spherical k-means assumptions")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); return save(fig, "ml_fig_c85_05.png")


def c85_06():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.4, 3.0, 2.8, 1.4, "documents", fc=TEAL, fs=10)
    box(ax, 3.6, 3.0, 2.8, 1.4, "inverted\nindex", fc=DEEP, fs=10)
    box(ax, 6.8, 3.0, 2.8, 1.4, "ranked\nhits", fc=GOLD, tc=INK, fs=10)
    ax.annotate("", xy=(3.6, 3.7), xytext=(3.2, 3.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(6.8, 3.7), xytext=(6.4, 3.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(5, 1.5, "query → postings lists → score(tf-idf / BM25)", ha="center",
            fontsize=10, color=DEEP, fontweight="bold")
    style(ax, "Information retrieval pipeline sketch")
    return save(fig, "ml_fig_c85_06.png")


def c85_07():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    feats = ["raw age", "z-age", "log labs", "poly²", "interaction"]
    imp = [0.12, 0.18, 0.25, 0.09, 0.22]
    leak = [0, 0, 0, 0, 0.15]
    x = np.arange(len(feats))
    ax.bar(x, imp, color=TEAL, label="safe signal")
    ax.bar(x, leak, bottom=imp, color=ROSE, label="leakage risk slice")
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=15, fontsize=8)
    style(ax, "Feature transforms: some boost signal, some smuggle future info")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c85_07.png")


def c85_08():
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    rng = np.random.default_rng(2)
    pts = rng.multivariate_normal([0, 0], [[1, 0.85], [0.85, 1]], 120)
    ax.scatter(pts[:, 0], pts[:, 1], c=TEAL, s=16, alpha=0.7)
    # PCA axes
    ax.annotate("", xy=(2.2, 2.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.5))
    ax.annotate("", xy=(-0.9, 1.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    ax.text(2.0, 2.2, "PC1", color=GOLD, fontweight="bold")
    ax.text(-1.3, 1.1, "PC2", color=ROSE, fontweight="bold")
    style(ax, "PCA axes align with covariance ellipse")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); return save(fig, "ml_fig_c85_08.png")


def c85_09():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.linspace(0, 10, 40)
    y = 2 + 0.6 * x + np.sin(x)
    yhat = 2 + 0.6 * x
    ax.scatter(x, y, c=TEAL, s=28, label="data")
    ax.plot(x, yhat, color=GOLD, lw=2.2, label="linear fit")
    for xi, yi, yh in zip(x[::3], y[::3], yhat[::3]):
        ax.plot([xi, xi], [yi, yh], color=ROSE, lw=1)
    style(ax, "OLS residuals are vertical distances to the fit")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c85_09.png")


def c85_10():
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    fpr = np.linspace(0, 1, 50)
    tpr_good = 1 - (1 - fpr) ** 3
    tpr_weak = fpr ** 0.7 * 0.85
    ax.plot(fpr, tpr_good, color=TEAL, lw=2.2, label="strong ranker")
    ax.plot(fpr, tpr_weak, color=GOLD, lw=2, label="weak ranker")
    ax.plot([0, 1], [0, 1], "--", color=SLATE, label="chance")
    style(ax, "ROC curves compare ranking quality")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c85_10.png")


def c85_11():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
    layers = [(0.3, 3, TEAL), (2.5, 4, DEEP), (5.0, 5, GOLD), (7.7, 3, MINT), (10.0, 2, ROSE)]
    for x, n, c in layers:
        for i in range(n):
            y = 2.0 + (i - (n - 1) / 2) * 0.55
            ax.add_patch(Circle((x, y), 0.22, facecolor=c, edgecolor="none"))
    for a, b in zip(layers, layers[1:]):
        for i in range(a[1]):
            for j in range(min(2, b[1])):
                y1 = 2.0 + (i - (a[1] - 1) / 2) * 0.55
                y2 = 2.0 + (j - (b[1] - 1) / 2) * 0.55
                ax.plot([a[0], b[0]], [y1, y2], color=SLATE, alpha=0.25, lw=0.8)
    style(ax, "Feed-forward MLP width pattern (schematic)")
    return save(fig, "ml_fig_c85_11.png")


def c85_12():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    epochs = np.arange(1, 41)
    linear = 0.1 + 0.7 * (1 - np.exp(-epochs / 8))
    ssl = 0.35 + 0.55 * (1 - np.exp(-epochs / 5))
    ax.plot(epochs, linear, color=SLATE, lw=2, label="from scratch")
    ax.plot(epochs, ssl, color=TEAL, lw=2.2, label="pretrained encoder + probe")
    style(ax, "Self-supervised pretrain can lift sample efficiency")
    ax.set_xlabel("fine-tune epochs"); ax.set_ylabel("downstream accuracy (synth)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c85_12.png")


def c85_13():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.4, 1.4, 2.4, 1.6, "tokens", fc=TEAL, fs=10)
    box(ax, 3.5, 1.4, 2.8, 1.6, "self-attn\nblocks", fc=DEEP, fs=10)
    box(ax, 7.0, 1.4, 2.6, 1.6, "logits /\nheads", fc=GOLD, tc=INK, fs=10)
    ax.annotate("", xy=(3.5, 2.2), xytext=(2.8, 2.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(7.0, 2.2), xytext=(6.3, 2.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    style(ax, "Transformer encoder stack for text (high level)")
    return save(fig, "ml_fig_c85_13.png")


def c85_14():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    t = np.arange(0, 30)
    g = 0.9 ** t
    ax.bar(t, g, color=TEAL, edgecolor=DEEP)
    ax.axhline(0.1, color=GOLD, ls="--", label="≈0.1 remaining")
    style(ax, "Discount γᵗ shrinks distant rewards")
    ax.set_xlabel("steps ahead t"); ax.set_ylabel("γᵗ"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c85_14.png")


def c85_15():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bits = [32, 16, 8, 4]
    size = [1.0, 0.5, 0.25, 0.125]
    acc = [0.91, 0.905, 0.88, 0.82]
    ax.plot(bits, size, "o-", color=TEAL, lw=2, label="relative size")
    ax.plot(bits, acc, "s-", color=GOLD, lw=2, label="accuracy (synth)")
    style(ax, "Quantization bit-width: size vs accuracy tradeoff")
    ax.set_xlabel("bits"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout(); return save(fig, "ml_fig_c85_15.png")


def c85_16():
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    # adjacency heat
    A = np.zeros((8, 8))
    for i, j in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3), (5, 6), (6, 7)]:
        A[i, j] = A[j, i] = 1
    ax.imshow(A, cmap="Greens", vmin=0, vmax=1)
    style(ax, "Adjacency matrix encodes undirected edges")
    ax.set_xlabel("node j"); ax.set_ylabel("node i")
    fig.tight_layout(); return save(fig, "ml_fig_c85_16.png")


def c85_17():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    stages = ["collect", "clean", "label", "split", "train", "monitor"]
    risk = [0.4, 0.7, 0.9, 0.85, 0.5, 0.75]
    ax.barh(stages[::-1], risk[::-1], color=TEAL, edgecolor=DEEP)
    style(ax, "Data lifecycle risk heat (labeling & split often hottest)")
    ax.set_xlabel("relative leakage/quality risk (synth)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c85_17.png")


def c85_18():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    checks = [
        (0.5, 4.2, "External\nvalid?"),
        (3.6, 4.2, "Calibrated\nrisks?"),
        (6.7, 4.2, "Leakage\naudit?"),
        (2.0, 1.5, "Utility &\ncost?"),
        (5.5, 1.5, "Monitoring\nplan?"),
    ]
    for x, y, t in checks:
        box(ax, x, y, 2.6, 1.4, t, fc=TEAL, fs=10)
    ax.text(5, 0.5, "Closing checklist before senior sign-off", ha="center",
            fontsize=10, color=DEEP, fontweight="bold")
    style(ax, "Senior practice go/no-go checklist tiles")
    return save(fig, "ml_fig_c85_18.png")


def c85_19():
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    pairs = [("AUROC", "ranking"), ("ECE", "calibration"), ("PPV", "prevalence-\nsensitive"),
             ("F1", "thresholded"), ("NLL", "prob.\nquality")]
    for i, (a, b) in enumerate(pairs):
        box(ax, 0.3 + i * 2.35, 1.4, 2.15, 1.2, f"{a}\n{b}", fc=TEAL if i % 2 == 0 else DEEP, fs=8)
    style(ax, "Metric glossary strip: what each score answers")
    return save(fig, "ml_fig_c85_19.png")


# ─── Cycle 86 ───
def c86_00():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.linspace(-2, 2, 200)
    for p, c in [(1, GOLD), (2, TEAL), (4, DEEP)]:
        ax.plot(x, np.abs(x) ** p, color=c, lw=2, label=f"|x|^{p}")
    style(ax, "Lp penalties shape the geometry of solutions")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 4)
    fig.tight_layout(); return save(fig, "ml_fig_c86_00.png")


def c86_01():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.5, 1.3, 2.8, 1.6, "Primary\nendpoint", fc=TEAL, fs=10)
    box(ax, 3.6, 1.3, 2.8, 1.6, "Sensitivity\nanalyses", fc=GOLD, tc=INK, fs=10)
    box(ax, 6.7, 1.3, 2.8, 1.6, "Exploratory\n(labeled)", fc=SLATE, fs=10)
    style(ax, "Claim hierarchy: primary vs exploratory")
    return save(fig, "ml_fig_c86_01.png")


def c86_02():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    under = np.array([0.4, 0.38, 0.37, 0.36])
    over = np.array([0.05, 0.02, 0.01, 0.005])
    good = np.array([0.25, 0.18, 0.14, 0.12])
    x = np.arange(4)
    ax.plot(x, under, "o-", color=ROSE, label="underfit")
    ax.plot(x, over, "s-", color=GOLD, label="overfit train")
    ax.plot(x, good, "^-", color=TEAL, label="balanced test")
    ax.set_xticks(x); ax.set_xticklabels(["v1", "v2", "v3", "v4"])
    style(ax, "Model versioning: track under/overfit trajectories")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylabel("error")
    fig.tight_layout(); return save(fig, "ml_fig_c86_02.png")


def c86_03():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    rng = np.random.default_rng(5)
    for i, (mu, c) in enumerate([((0, 0), TEAL), ((3, 1), GOLD), ((1, 3), ROSE)]):
        pts = rng.normal(mu, 0.4, (40, 2))
        ax.scatter(pts[:, 0], pts[:, 1], c=c, s=20, alpha=0.75, label=f"class {i}")
    style(ax, "Class-conditional clouds (visualize before modeling)")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); return save(fig, "ml_fig_c86_03.png")


def c86_04():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    p = np.linspace(0.01, 0.99, 200)
    ax.plot(p, -np.log(p), color=TEAL, lw=2.2, label="-log p (true class)")
    ax.plot(p, -(p * np.log(p) + (1 - p) * np.log(1 - p)), color=GOLD, lw=2, label="binary entropy")
    style(ax, "Log loss penalizes confident wrong probabilities")
    ax.set_xlabel("p"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_04.png")


def c86_05():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    k = np.arange(2, 12)
    ch = np.array([0, 120, 80, 40, 22, 12, 8, 5, 3, 2])
    ax.bar(k, ch, color=TEAL, edgecolor=DEEP)
    ax.axvline(4, color=GOLD, ls="--", lw=1.5, label="gap peak ~4")
    style(ax, "Gap statistic sketch for choosing k")
    ax.set_xlabel("k"); ax.set_ylabel("gap"); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_05.png")


def c86_06():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.5, 3.2, 2.5, 1.2, "seq A", fc=TEAL, fs=10)
    box(ax, 3.7, 3.2, 2.5, 1.2, "seq B", fc=DEEP, fs=10)
    box(ax, 6.9, 3.2, 2.5, 1.2, "seq C", fc=GOLD, tc=INK, fs=10)
    box(ax, 2.5, 1.0, 5.0, 1.4, "frequent sequential patterns", fc=MINT, tc=INK, fs=10)
    for x in (1.7, 5.0, 8.1):
        ax.annotate("", xy=(5.0, 2.4), xytext=(x, 3.2),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2))
    style(ax, "Sequence mining aggregates ordered item patterns")
    return save(fig, "ml_fig_c86_06.png")


def c86_07():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    raw = np.array([2, 5, 8, 40, 100])
    log1p = np.log1p(raw)
    z = (raw - raw.mean()) / raw.std()
    x = np.arange(len(raw))
    ax.plot(x, raw / raw.max(), "o-", color=SLATE, label="raw scaled")
    ax.plot(x, log1p / log1p.max(), "s-", color=TEAL, label="log1p scaled")
    ax.plot(x, (z - z.min()) / (z.max() - z.min()), "^-", color=GOLD, label="z min-max")
    style(ax, "Scaling choices reshape feature geometry")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_07.png")


def c86_08():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    m = np.arange(1, 21)
    recon = np.exp(-0.25 * m) + 0.05
    ax.plot(m, recon, "o-", color=TEAL, lw=2)
    ax.axhline(0.1, color=GOLD, ls="--", label="error budget")
    style(ax, "Reconstruction error vs retained components")
    ax.set_xlabel("components kept"); ax.set_ylabel("recon error"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_08.png")


def c86_09():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 100)
    e = rng.normal(0, 1, 100)
    y = 0.0 * x + e
    ax.scatter(x, y, c=TEAL, s=22, alpha=0.75)
    ax.axhline(0, color=GOLD, lw=2, label="β̂≈0 fit")
    style(ax, "Null association: flat fit, residual noise only")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); return save(fig, "ml_fig_c86_09.png")


def c86_10():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    thr = np.linspace(0, 1, 50)
    sens = 1 - thr ** 1.5
    spec = thr ** 0.8
    ax.plot(thr, sens, color=TEAL, lw=2, label="sensitivity")
    ax.plot(thr, spec, color=GOLD, lw=2, label="specificity")
    style(ax, "Threshold moves sensitivity and specificity in opposite directions")
    ax.set_xlabel("threshold"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_10.png")


def c86_11():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    steps = np.arange(0, 60)
    loss = 2.0 * np.exp(-steps / 12) + 0.2 + 0.05 * np.sin(steps / 3)
    ax.plot(steps, loss, color=TEAL, lw=2.2)
    ax.axvline(35, color=GOLD, ls="--", label="early stop")
    style(ax, "Training loss trajectory with early-stopping mark")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_11.png")


def c86_12():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.4, 1.3, 2.6, 1.6, "view 1", fc=TEAL, fs=10)
    box(ax, 3.7, 1.3, 2.6, 1.6, "shared\nencoder", fc=DEEP, fs=10)
    box(ax, 7.0, 1.3, 2.6, 1.6, "view 2", fc=GOLD, tc=INK, fs=10)
    ax.annotate("", xy=(3.7, 2.1), xytext=(3.0, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(7.0, 2.1), xytext=(6.3, 2.1), arrowprops=dict(arrowstyle="<-", color=INK, lw=1.5))
    style(ax, "Multi-view SSL: shared representation across views")
    return save(fig, "ml_fig_c86_12.png")


def c86_13():
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    # patch grid
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j, 3 - i), 0.95, 0.95,
                                   facecolor=TEAL if (i + j) % 2 == 0 else DEEP,
                                   edgecolor="white", lw=2, alpha=0.85))
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-0.2, 4.2); ax.set_aspect("equal"); ax.axis("off")
    style(ax, "Vision transformer patch tokenization (grid)")
    return save(fig, "ml_fig_c86_13.png")


def c86_14():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    states = np.arange(5)
    q_a = np.array([0.2, 0.8, 0.5, 0.3, 0.6])
    q_b = np.array([0.5, 0.4, 0.7, 0.9, 0.2])
    w = 0.35
    ax.bar(states - w / 2, q_a, w, color=TEAL, label="Q(s,a1)")
    ax.bar(states + w / 2, q_b, w, color=GOLD, label="Q(s,a2)")
    style(ax, "Action-value table snapshot across states")
    ax.set_xlabel("state"); ax.set_ylabel("Q"); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_14.png")


def c86_15():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    layers = np.arange(1, 9)
    dense = 100 * np.ones_like(layers)
    pruned = 100 * (0.7 ** (layers - 1))
    ax.plot(layers, dense, "o-", color=SLATE, label="dense params %")
    ax.plot(layers, pruned, "s-", color=TEAL, label="after structured prune %")
    style(ax, "Structured pruning reduces parameters by layer")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel("layer")
    fig.tight_layout(); return save(fig, "ml_fig_c86_15.png")


def c86_16():
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    pos = {i: (np.cos(2 * np.pi * i / 6), np.sin(2 * np.pi * i / 6)) for i in range(6)}
    pos[6] = (0, 0)
    for i in range(6):
        ax.plot([pos[i][0], 0], [pos[i][1], 0], color=SLATE, lw=1.5)
        ax.plot([pos[i][0], pos[(i + 1) % 6][0]], [pos[i][1], pos[(i + 1) % 6][1]],
                color=SLATE, lw=1)
    for i, (x, y) in pos.items():
        ax.plot(x, y, "o", color=TEAL if i < 6 else GOLD, ms=14)
    style(ax, "Ego-network: center node and 1-hop neighborhood")
    ax.set_aspect("equal"); ax.axis("off")
    return save(fig, "ml_fig_c86_16.png")


def c86_17():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.4, 1.3, 2.8, 1.6, "schema\ndrift", fc=ROSE, fs=10)
    box(ax, 3.6, 1.3, 2.8, 1.6, "unit /\ncode drift", fc=GOLD, tc=INK, fs=10)
    box(ax, 6.8, 1.3, 2.8, 1.6, "semantic\ndrift", fc=TEAL, fs=10)
    style(ax, "Data quality failure modes beyond missingness")
    return save(fig, "ml_fig_c86_17.png")


def c86_18():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    labels = ["internal", "temporal", "geo\nexternal", "prospective"]
    strength = [0.4, 0.6, 0.8, 1.0]
    ax.barh(labels[::-1], strength[::-1], color=[SLATE, GOLD, TEAL, DEEP][::-1])
    style(ax, "Validation strength ladder (higher is more convincing)")
    ax.set_xlabel("relative evidential strength")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c86_18.png")


def c86_19():
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    terms = [("prior", TEAL), ("likelihood", DEEP), ("posterior", GOLD),
             ("MAP", MINT), ("MLE", SLATE)]
    for i, (t, c) in enumerate(terms):
        box(ax, 0.3 + i * 2.35, 1.0, 2.15, 1.3, t, fc=c, fs=10, tc="white" if c not in (GOLD, MINT) else INK)
    style(ax, "Bayesian glossary chain")
    return save(fig, "ml_fig_c86_19.png")


# ─── Cycle 87 ───
def c87_00():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=SLATE, lw=1)
    v = np.array([0.8, 0.4]); Mv = np.array([1.2, 0.2])
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.annotate("", xy=Mv, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    ax.text(0.85, 0.5, "v", color=TEAL, fontweight="bold")
    ax.text(1.25, 0.25, "Av", color=GOLD, fontweight="bold")
    style(ax, "Linear maps stretch and rotate vectors")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    fig.tight_layout(); return save(fig, "ml_fig_c87_00.png")


def c87_01():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.5, 1.3, 4.0, 1.6, "Prediction task\n(association OK)", fc=TEAL, fs=10)
    box(ax, 5.5, 1.3, 4.0, 1.6, "Causal claim\nneeds design/assumptions", fc=ROSE, fs=10)
    ax.text(5, 0.5, "pred ≠ cause — keep the boxes separate", ha="center",
            fontsize=10, color=INK, fontweight="bold")
    style(ax, "Preface boundary: prediction vs causal inference")
    return save(fig, "ml_fig_c87_01.png")


def c87_02():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = ["rules", "linear", "tree", "ensemble", "deep"]
    interpret = [0.95, 0.8, 0.55, 0.35, 0.2]
    flex = [0.15, 0.35, 0.55, 0.75, 0.95]
    ax.plot(xs, interpret, "o-", color=TEAL, lw=2, label="interpretability")
    ax.plot(xs, flex, "s-", color=GOLD, lw=2, label="flexibility")
    style(ax, "Model family tradeoff: interpretability vs flexibility")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout(); return save(fig, "ml_fig_c87_02.png")


def c87_03():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    cats = ["A", "B", "C", "D", "E"]
    vals = [40, 55, 48, 62, 50]
    ax.bar(cats, vals, color=TEAL, edgecolor=DEEP)
    ax.errorbar(cats, vals, yerr=[6, 4, 9, 5, 7], fmt="none", ecolor=GOLD, elinewidth=2, capsize=4)
    style(ax, "Always show uncertainty on summary charts")
    ax.set_ylabel("rate (%)"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_03.png")


def c87_04():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.linspace(0, 8, 200)
    import math
    for k, c in [(1, TEAL), (2, GOLD), (4, DEEP)]:
        # chi-square-ish shapes via gamma
        y = x ** (k / 2 - 1) * np.exp(-x / 2) / (2 ** (k / 2) * max(math.gamma(k / 2), 1e-9))
        y = np.nan_to_num(y)
        ax.plot(x, y, color=c, lw=2, label=f"df≈{k}")
    style(ax, "Chi-square family shapes used in many tests")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_04.png")


def c87_05():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    # dendrogram-like manual
    ax.plot([1, 1, 2, 2], [0, 1, 1, 0], color=TEAL, lw=2)
    ax.plot([1.5, 1.5, 3.5, 3.5], [1, 2, 2, 0], color=DEEP, lw=2)
    ax.plot([3, 3], [0, 0.6], color=TEAL, lw=2)
    ax.plot([4, 4, 5, 5], [0, 1.2, 1.2, 0], color=GOLD, lw=2)
    ax.plot([2.5, 2.5, 4.5, 4.5], [2, 3, 3, 1.2], color=ROSE, lw=2)
    ax.set_xlim(0, 6); ax.set_ylim(-0.2, 3.5); ax.axis("off")
    style(ax, "Agglomerative dendrogram cut chooses flat clusters")
    return save(fig, "ml_fig_c87_05.png")


def c87_06():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    conf = np.linspace(0.1, 1, 20)
    lift = 2.5 - 1.5 * conf + 0.1 * np.sin(5 * conf)
    ax.plot(conf, lift, "o-", color=TEAL, lw=2)
    ax.axhline(1.0, color=ROSE, ls="--", label="lift=1 independence")
    style(ax, "Lift vs confidence for mined rules")
    ax.set_xlabel("confidence"); ax.set_ylabel("lift"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_06.png")


def c87_07():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    # hashing collision sketch
    buckets = np.zeros(12)
    rng = np.random.default_rng(4)
    for _ in range(40):
        buckets[rng.integers(0, 12)] += 1
    ax.bar(range(12), buckets, color=TEAL, edgecolor=DEEP)
    style(ax, "Feature hashing: collisions redistribute mass across buckets")
    ax.set_xlabel("hash bucket"); ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_07.png")


def c87_08():
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    rng = np.random.default_rng(9)
    high = rng.normal(0, 1, (100, 2))
    # fake UMAP pull
    low = high * 0.3 + rng.normal(0, 0.05, (100, 2))
    ax.scatter(high[:, 0], high[:, 1], c=SLATE, s=12, alpha=0.4, label="high-D projection")
    ax.scatter(low[:, 0] + 3, low[:, 1], c=TEAL, s=16, alpha=0.8, label="neighbor-preserving embed")
    style(ax, "Neighbor-preserving embeddings rearrange geometry")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); return save(fig, "ml_fig_c87_08.png")


def c87_09():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.linspace(0, 5, 50)
    for a, c, lab in [(0.0, SLATE, "linear"), (0.4, TEAL, "mild curve"), (1.0, GOLD, "strong curve")]:
        ax.plot(x, 1 + 0.5 * x + a * x ** 2, color=c, lw=2, label=lab)
    style(ax, "Polynomial terms add curvature to regression mean")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_09.png")


def c87_10():
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    # confusion matrix
    cm = np.array([[80, 20], [10, 90]])
    im = ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color=INK, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred−", "pred+"]); ax.set_yticklabels(["true−", "true+"])
    style(ax, "Confusion matrix counts for a fixed threshold")
    fig.tight_layout(); return save(fig, "ml_fig_c87_10.png")


def c87_11():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    depth = np.arange(1, 21)
    train = 1 - 0.9 * (1 - np.exp(-depth / 3))
    val = 0.55 + 0.25 * np.exp(-((depth - 8) ** 2) / 20) - 0.05 * depth / 20
    # better: val peaks then drops
    val = 0.5 + 0.3 * (1 - np.exp(-depth / 4)) - 0.02 * np.maximum(0, depth - 10)
    ax.plot(depth, train, color=TEAL, lw=2, label="train acc")
    ax.plot(depth, val, color=GOLD, lw=2, label="val acc")
    style(ax, "Depth increases capacity; validation may peak then fall")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel("depth")
    fig.tight_layout(); return save(fig, "ml_fig_c87_11.png")


def c87_12():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.4, 1.3, 2.6, 1.6, "masked\npatch", fc=ROSE, fs=10)
    box(ax, 3.6, 1.3, 2.8, 1.6, "encoder\ncontext", fc=TEAL, fs=10)
    box(ax, 7.0, 1.3, 2.6, 1.6, "reconstruct", fc=GOLD, tc=INK, fs=10)
    ax.annotate("", xy=(3.6, 2.1), xytext=(3.0, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(7.0, 2.1), xytext=(6.4, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    style(ax, "Masked autoencoding for vision/audio SSL")
    return save(fig, "ml_fig_c87_12.png")


def c87_13():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(t, np.sin(3 * t) * np.exp(-0.15 * t), color=TEAL, lw=2)
    style(ax, "1D waveform before spectrogram / encoder front-ends")
    ax.set_xlabel("time"); ax.set_ylabel("amp"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_13.png")


def c87_14():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    episodes = np.arange(1, 81)
    ret = 10 * (1 - np.exp(-episodes / 25)) + np.sin(episodes / 3) * 0.5
    ax.plot(episodes, ret, color=TEAL, lw=2)
    ax.fill_between(episodes, ret - 1.5, ret + 1.5, color=TEAL, alpha=0.2)
    style(ax, "Episode return learning curve with variability band")
    ax.set_xlabel("episode"); ax.set_ylabel("return"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_14.png")


def c87_15():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.4, 1.3, 2.8, 1.6, "full model", fc=TEAL, fs=10)
    box(ax, 3.6, 1.3, 2.8, 1.6, "distill /\nprune", fc=DEEP, fs=10)
    box(ax, 6.8, 1.3, 2.8, 1.6, "edge\ndeploy", fc=GOLD, tc=INK, fs=10)
    ax.annotate("", xy=(3.6, 2.1), xytext=(3.2, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(6.8, 2.1), xytext=(6.4, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    style(ax, "Compression path from research model to edge device")
    return save(fig, "ml_fig_c87_15.png")


def c87_16():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    deg = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 8, 12])
    ax.hist(deg, bins=np.arange(0.5, 13.5, 1), color=TEAL, edgecolor=DEEP)
    style(ax, "Degree distribution often heavy-tailed in real graphs")
    ax.set_xlabel("degree"); ax.set_ylabel("count"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_16.png")


def c87_17():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    weeks = np.arange(1, 25)
    p = 0.55 + 0.08 * np.sin(weeks / 3) + 0.002 * weeks
    ax.plot(weeks, p, color=TEAL, lw=2.2)
    ax.axhline(0.55, color=GOLD, ls="--", label="baseline prevalence")
    style(ax, "Label prevalence can drift over calendar time")
    ax.set_xlabel("week"); ax.set_ylabel("P(y=1)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); return save(fig, "ml_fig_c87_17.png")


def c87_18():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 3.5, 4.5, 3.0, 1.1, "Deploy?", fc=DEEP, fs=11)
    box(ax, 0.8, 2.5, 3.0, 1.1, "Yes + monitor", fc=TEAL, fs=10)
    box(ax, 6.2, 2.5, 3.0, 1.1, "No / revise", fc=ROSE, fs=10)
    box(ax, 0.8, 0.6, 3.0, 1.1, "Shadow mode", fc=GOLD, tc=INK, fs=10)
    box(ax, 6.2, 0.6, 3.0, 1.1, "New data", fc=SLATE, fs=10)
    ax.annotate("", xy=(2.3, 3.6), xytext=(4.5, 4.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(7.7, 3.6), xytext=(5.5, 4.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(2.3, 1.7), xytext=(2.3, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(7.7, 1.7), xytext=(7.7, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, "Deployment decision tree for senior review")
    return save(fig, "ml_fig_c87_18.png")


def c87_19():
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    for i, (t, c) in enumerate([("bias", TEAL), ("variance", DEEP), ("noise", GOLD),
                                 ("risk", MINT), ("Bayes err", SLATE)]):
        box(ax, 0.3 + i * 2.35, 1.0, 2.15, 1.3, t, fc=c, fs=10,
            tc="white" if c not in (GOLD, MINT) else INK)
    style(ax, "Error decomposition glossary strip")
    return save(fig, "ml_fig_c87_19.png")


CAPTIONS = {
    85: [
        "Activation shapes control gradient flow through deep stacks.",
        "Five-box protocol spine for transparent ML reporting.",
        "Learning curves: more data typically shrinks the gen. gap.",
        "Hexbin density helps when scatterplots overplot.",
        "Gaussian location/scale family of densities.",
        "Non-convex clusters break spherical k-means assumptions.",
        "IR pipeline: documents → inverted index → ranked hits.",
        "Feature transforms can add signal or smuggle leakage.",
        "PCA axes track the covariance ellipse of the cloud.",
        "OLS residuals are vertical distances to the fitted line.",
        "ROC curves compare ranking quality against chance.",
        "Feed-forward MLP width pattern (schematic neurons).",
        "SSL pretrain can improve downstream sample efficiency.",
        "Transformer encoder stack for token sequences.",
        "Discount factors shrink the value of distant rewards.",
        "Quantization bit-width trades size against accuracy.",
        "Adjacency matrix view of an undirected graph.",
        "Lifecycle risk heat: labeling and splitting are hot spots.",
        "Senior go/no-go checklist tiles before sign-off.",
        "Metric glossary strip: AUROC, ECE, PPV, F1, NLL.",
    ],
    86: [
        "Lp penalty shapes (|x|, x², x⁴) alter solution geometry.",
        "Claim hierarchy: primary endpoint vs exploratory labels.",
        "Version trajectories for underfit vs overfit errors.",
        "Class-conditional clouds—visualize before modeling.",
        "Log loss and entropy: cost of confident mistakes.",
        "Gap statistic sketch for selecting cluster count k.",
        "Sequence mining over ordered event patterns.",
        "Scaling choices (raw/log/z) reshape feature geometry.",
        "Reconstruction error versus retained components.",
        "Null association cloud with flat fitted slope.",
        "Sensitivity vs specificity as threshold moves.",
        "Training loss curve with early-stopping marker.",
        "Multi-view SSL shared encoder between views.",
        "ViT-style patch grid tokenization for images.",
        "Q(s,a) table snapshot across discrete states.",
        "Structured pruning reduces parameters by layer.",
        "Ego-network: center node and one-hop neighbors.",
        "Schema, unit/code, and semantic data drift modes.",
        "Validation strength ladder from internal to prospective.",
        "Bayesian glossary chain: prior→likelihood→posterior.",
    ],
    87: [
        "Linear maps stretch and rotate vectors (Av vs v).",
        "Prediction box vs causal claim box—keep them separate (pred≠cause).",
        "Interpretability versus flexibility across model families.",
        "Bar charts need uncertainty intervals.",
        "Chi-square-like density family shapes.",
        "Dendrogram cut selects flat clusters from hierarchy.",
        "Lift versus confidence for association rules.",
        "Feature hashing redistributes mass via collisions.",
        "Neighbor-preserving embeddings rearrange geometry.",
        "Polynomial terms introduce mean-function curvature.",
        "Confusion matrix counts at one operating threshold.",
        "Depth vs train/val accuracy capacity curve.",
        "Masked autoencoding reconstructs held-out patches.",
        "Raw 1D waveform prior to spectral front-ends.",
        "RL episode return curve with variability band.",
        "Compression path from full model to edge deploy.",
        "Heavy-tailed degree histogram in graphs.",
        "Label prevalence drift across calendar weeks.",
        "Deployment decision tree: monitor, revise, shadow.",
        "Error decomposition glossary: bias, variance, noise, risk.",
    ],
}

GENERATORS = {
    85: [c85_00, c85_01, c85_02, c85_03, c85_04, c85_05, c85_06, c85_07, c85_08, c85_09,
         c85_10, c85_11, c85_12, c85_13, c85_14, c85_15, c85_16, c85_17, c85_18, c85_19],
    86: [c86_00, c86_01, c86_02, c86_03, c86_04, c86_05, c86_06, c86_07, c86_08, c86_09,
         c86_10, c86_11, c86_12, c86_13, c86_14, c86_15, c86_16, c86_17, c86_18, c86_19],
    87: [c87_00, c87_01, c87_02, c87_03, c87_04, c87_05, c87_06, c87_07, c87_08, c87_09,
         c87_10, c87_11, c87_12, c87_13, c87_14, c87_15, c87_16, c87_17, c87_18, c87_19],
}


def embed_cycle(cycle: int) -> int:
    n = 0
    for i, ch in enumerate(CHAPTERS):
        p = CURR / ch
        fig = f"ml_fig_c{cycle}_{i:02d}.png"
        cap = CAPTIONS[cycle][i]
        block = (
            f"\n![c{cycle} teaching panel {i:02d} (original).](../assets/figures/{fig})\n"
            f"*Figure — {cap} Synthetic teaching geometry—not a causal claim.*\n"
        )
        text = p.read_text(encoding="utf-8")
        if fig in text:
            continue
        marker = "## Chapter Summary"
        if marker in text:
            text = text.replace(marker, block + "\n" + marker, 1)
        else:
            text = text.rstrip() + "\n" + block
        p.write_text(text, encoding="utf-8")
        n += 1
    return n


def main(cycles=None):
    cycles = cycles or [85, 86, 87]
    total = 0
    for c in cycles:
        for fn in GENERATORS[c]:
            fn()
            total += 1
        print(f"EMBEDDED cycle {c}: {embed_cycle(c)}")
    print("TOTAL_FIGURES", total)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main([int(x) for x in sys.argv[1].split(",")])
    else:
        main()
