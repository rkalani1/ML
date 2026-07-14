#!/usr/bin/env python3
"""Generate original teaching figures for ML open-source ebook (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#0d9488"
DEEP = "#0f766e"
INK = "#0f172a"
GOLD = "#c9a227"
SOFT = "#ecfeff"


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", path.name)
    return path


def style_ax(ax, title: str):
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=10)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def fig_supervised_map():
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.4, 2.8, 2.4, 1.4, "Labeled\ndata", TEAL),
        (3.8, 2.8, 2.4, 1.4, "Model\nfit", DEEP),
        (7.2, 2.8, 2.4, 1.4, "Predict\nnew cases", GOLD),
        (0.4, 0.5, 2.4, 1.4, "Unlabeled\nstructure", "#64748b"),
        (3.8, 0.5, 2.4, 1.4, "Cluster /\nreduce", "#475569"),
        (7.2, 0.5, 2.4, 1.4, "Explore /\ngroup", "#334155"),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor=c, edgecolor="none", alpha=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(3.7, 3.5), xytext=(2.9, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(7.1, 3.5), xytext=(6.3, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(3.7, 1.2), xytext=(2.9, 1.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.annotate("", xy=(7.1, 1.2), xytext=(6.3, 1.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.text(5, 4.7, "Supervised path (top) vs unsupervised path (bottom)", ha="center", fontsize=12, color=INK)
    save(fig, "ml_fig_supervised_unsupervised_map.png")


def fig_gradient_descent():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.linspace(-3, 3, 400)
    y = 0.35 * (x - 0.3) ** 2 + 0.15 * np.sin(2.2 * x) + 0.4
    ax.plot(x, y, color=TEAL, lw=2.5)
    xs = np.array([-2.4, -1.6, -0.9, -0.3, 0.2, 0.55])
    ys = 0.35 * (xs - 0.3) ** 2 + 0.15 * np.sin(2.2 * xs) + 0.4
    ax.plot(xs, ys, "o-", color=GOLD, lw=1.8, markersize=7)
    ax.set_xlabel("parameter")
    ax.set_ylabel("loss")
    style_ax(ax, "Gradient descent intuition (synthetic loss surface)")
    ax.annotate("steps toward lower loss", xy=(0.2, ys[4]), xytext=(-1.2, 2.2), arrowprops=dict(arrowstyle="->", color=DEEP), color=DEEP)
    save(fig, "ml_fig_gradient_descent.png")


def fig_train_val_test():
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    segs = [(0.3, 7.2, "Train — fit parameters", TEAL), (7.6, 2.0, "Validation — tune", GOLD), (9.8, 1.9, "Test — once", DEEP)]
    for x, w, lab, c in segs:
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=c, edgecolor="none"))
        ax.text(x + w / 2, 1.55, lab, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(6, 2.5, "Time-respecting split (synthetic stroke cohort)", ha="center", fontsize=12, color=INK)
    ax.text(6, 0.45, "Never peek at test while choosing models", ha="center", fontsize=10, color="#64748b")
    save(fig, "ml_fig_train_val_test.png")


def fig_confusion_roc():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    ax = axes[0]
    mat = np.array([[82, 8], [11, 49]])
    im = ax.imshow(mat, cmap="YlGnBu")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1], ["Pred−", "Pred+"])
    ax.set_yticks([0, 1], ["True−", "True+"])
    style_ax(ax, "Confusion matrix (synthetic)")
    ax2 = axes[1]
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr) ** 2.2
    ax2.plot(fpr, tpr, color=TEAL, lw=2.5, label="model")
    ax2.plot([0, 1], [0, 1], "--", color="#94a3b8", label="chance")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(frameon=False, fontsize=9)
    style_ax(ax2, "ROC curve (synthetic)")
    fig.tight_layout()
    save(fig, "ml_fig_confusion_roc.png")


def fig_calibration():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    p = np.linspace(0.05, 0.95, 9)
    obs = p + 0.08 * np.sin(2 * np.pi * p) - 0.03
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="perfect")
    ax.plot(p, obs, "o-", color=TEAL, lw=2, label="model")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    style_ax(ax, "Calibration (synthetic neuro risk model)")
    save(fig, "ml_fig_calibration.png")


def fig_kmeans():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    centers = np.array([[0, 0], [3.2, 0.4], [1.2, 2.8]])
    colors = [TEAL, GOLD, DEEP]
    for i, c in enumerate(centers):
        pts = rng.normal(c, 0.45, size=(60, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.75, c=colors[i], edgecolors="none")
        ax.scatter(*c, s=160, c="white", edgecolors=colors[i], linewidths=2.5, zorder=5)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    style_ax(ax, "Clustering sketch (synthetic blobs + centroids)")
    save(fig, "ml_fig_clustering.png")


def fig_pca():
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    t = rng.normal(size=120)
    x = t * 2.2 + rng.normal(0, 0.25, 120)
    y = t * 1.1 + rng.normal(0, 0.35, 120)
    ax.scatter(x, y, s=16, c=TEAL, alpha=0.7)
    ax.arrow(-3, -1.5, 6.2, 3.1, head_width=0.2, color=GOLD, lw=2.2, length_includes_head=True)
    ax.text(2.2, 2.0, "PC1", color=GOLD, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    style_ax(ax, "Dimensionality reduction intuition (PCA sketch)")
    save(fig, "ml_fig_pca.png")


def fig_mlp():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    layers = [[1.2, [1.5, 3, 4.5]], [4, [1, 2.5, 4, 5]], [6.8, [2, 4]], [9, [3]]]
    labels = ["input", "hidden", "hidden", "out"]
    coords = []
    for li, (x, ys) in enumerate(layers):
        layer_pts = []
        for y in ys:
            ax.add_patch(Circle((x, y), 0.28, facecolor=TEAL if li < 3 else GOLD, edgecolor="none"))
            layer_pts.append((x, y))
        coords.append(layer_pts)
        ax.text(x, 0.45, labels[li], ha="center", color=INK, fontsize=10)
    for a, b in zip(coords, coords[1:]):
        for x1, y1 in a:
            for x2, y2 in b:
                ax.plot([x1 + 0.28, x2 - 0.28], [y1, y2], color="#94a3b8", lw=0.7, alpha=0.7)
    ax.text(5, 5.6, "Simple multilayer network (teaching diagram)", ha="center", fontsize=12, color=INK)
    save(fig, "ml_fig_mlp.png")


def fig_rl_loop():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 4.2), 3, 1.6, boxstyle="round,pad=0.05,rounding_size=0.2", facecolor=TEAL, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((6, 4.2), 3, 1.6, boxstyle="round,pad=0.05,rounding_size=0.2", facecolor=DEEP, edgecolor="none"))
    ax.text(2.5, 5, "Agent\n(policy)", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(7.5, 5, "Environment\n(state)", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.annotate("action", xy=(6, 5.3), xytext=(4.1, 5.3), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2), color=GOLD, fontsize=11)
    ax.annotate("reward + next state", xy=(4.1, 4.5), xytext=(6, 4.5), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2), color=GOLD, fontsize=10)
    ax.text(5, 2.2, "Reinforcement learning loop (teaching sketch)", ha="center", fontsize=12, color=INK)
    ax.text(5, 1.3, "Useful metaphor for sequential clinical decisions\n— not a claim that RL is ready for bedside control", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_rl_loop.png")


def fig_leakage():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.plot([0.5, 11.5], [2, 2], color="#cbd5e1", lw=6, solid_capstyle="round")
    for x, lab in [(2, "admit"), (5, "labs\navailable"), (8, "discharge\ncodes"), (10.5, "predict\nat door")]:
        ax.plot(x, 2, "o", color=TEAL, markersize=12)
        ax.text(x, 1.2, lab, ha="center", fontsize=9, color=INK)
    ax.annotate("LEAK if used\nat prediction time", xy=(8, 2.15), xytext=(7.2, 3.4), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8), color="#dc2626", fontsize=10, fontweight="bold")
    ax.text(6, 0.4, "Feature timing vs prediction time (synthetic stroke pathway)", ha="center", fontsize=11, color=INK)
    save(fig, "ml_fig_leakage_timeline.png")


def fig_pretrain_finetune():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for x, w, t, c in [(0.4, 4.2, "Pretrain on large\nrelated corpus", TEAL), (5.2, 4.4, "Fine-tune on local\nneuro labels", GOLD)]:
        ax.add_patch(FancyBboxPatch((x, 0.8), w, 1.5, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor=c, edgecolor="none"))
        ax.text(x + w / 2, 1.55, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(5.1, 1.55), xytext=(4.7, 1.55), arrowprops=dict(arrowstyle="->", lw=2, color=INK))
    ax.text(5, 2.7, "Self-supervised / transfer learning pipeline", ha="center", fontsize=12, color=INK)
    save(fig, "ml_fig_pretrain_finetune.png")


def fig_site_shift():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rng = np.random.default_rng(1)
    a = rng.normal([0, 0], 0.6, size=(80, 2))
    b = rng.normal([2.2, 0.3], 0.7, size=(80, 2))
    ax.scatter(a[:, 0], a[:, 1], s=16, c=TEAL, alpha=0.7, label="site A")
    ax.scatter(b[:, 0], b[:, 1], s=16, c=GOLD, alpha=0.7, label="site B")
    ax.legend(frameon=False)
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    style_ax(ax, "Dataset shift across sites (synthetic)")
    save(fig, "ml_fig_site_shift.png")


def fig_appraisal_card():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    items = [
        "1. Question fit (prediction vs action)",
        "2. Data provenance & leakage audit",
        "3. Validation design (site / time)",
        "4. Calibration + decision utility",
        "5. Equity / missingness / drift plan",
    ]
    ax.add_patch(FancyBboxPatch((0.6, 0.5), 8.8, 5, boxstyle="round,pad=0.05,rounding_size=0.2", facecolor=SOFT, edgecolor=TEAL, linewidth=2))
    ax.text(5, 5.1, "Model appraisal scorecard (teaching)", ha="center", fontsize=13, fontweight="bold", color=DEEP)
    for i, t in enumerate(items):
        ax.text(1.2, 4.2 - i * 0.7, t, fontsize=11, color=INK)
    save(fig, "ml_fig_appraisal_scorecard.png")


def main():
    fig_supervised_map()
    fig_gradient_descent()
    fig_train_val_test()
    fig_confusion_roc()
    fig_calibration()
    fig_kmeans()
    fig_pca()
    fig_mlp()
    fig_rl_loop()
    fig_leakage()
    fig_pretrain_finetune()
    fig_site_shift()
    fig_appraisal_card()
    print("DONE figures in", OUT)


if __name__ == "__main__":
    main()
