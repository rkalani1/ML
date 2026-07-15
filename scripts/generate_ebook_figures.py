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


def fig_association_rules():
    """Ch05: support / confidence / lift for the 5-transaction worked example."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    # Left: bar chart of conf vs lift for three rules
    ax = axes[0]
    rules = ["A→B", "A→D", "D→A"]
    conf = [0.75, 0.50, 1.00]
    lift = [0.94, 1.25, 1.25]
    x = np.arange(len(rules))
    w = 0.35
    ax.bar(x - w / 2, conf, w, label="confidence", color=TEAL)
    ax.bar(x + w / 2, lift, w, label="lift", color=GOLD)
    ax.axhline(1.0, color="#94a3b8", ls="--", lw=1.2, label="lift = 1 (indep.)")
    ax.set_xticks(x, rules)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax, "Association rules (n=5 toy basket)")
    # Right: support of itemsets
    ax2 = axes[1]
    items = ["A", "B", "C", "D", "AB", "AD"]
    sup = [0.80, 0.80, 0.80, 0.40, 0.60, 0.40]
    colors = [TEAL, TEAL, TEAL, DEEP, GOLD, GOLD]
    ax2.barh(items[::-1], sup[::-1], color=colors[::-1])
    ax2.set_xlabel("relative support s(X)")
    ax2.set_xlim(0, 1.0)
    style_ax(ax2, "Itemset support (same toy D)")
    fig.tight_layout()
    save(fig, "ml_fig_association_rules.png")


def fig_feature_pipeline():
    """Ch06: fit-on-train feature pipeline."""
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    stages = [
        (0.2, "Raw\ntable", "#64748b"),
        (2.4, "Impute", TEAL),
        (4.6, "Encode", DEEP),
        (6.8, "Scale", GOLD),
        (9.0, "Select", "#b45309"),
        (11.0, "Model", INK),
    ]
    for i, (x, lab, c) in enumerate(stages):
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.55, 1.0),
                1.3,
                1.2,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=c,
                edgecolor="none",
            )
        )
        ax.text(x + 0.1, 1.6, lab, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(stages[i + 1][0] - 0.6, 1.6),
                xytext=(x + 0.8, 1.6),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6),
            )
    ax.text(6, 2.65, "Feature pipeline — fit transforms on train fold only", ha="center", fontsize=12, color=INK)
    ax.text(6, 0.45, "Apply frozen transforms to validation / test (no leakage)", ha="center", fontsize=10, color="#64748b")
    save(fig, "ml_fig_feature_pipeline.png")


def fig_attention_toy():
    """Ch12: scaled-dot-product attention weight heatmap for 3-token toy."""
    # Tokens t1=[1,0], t2=[0,1], t3=[0.7,0.7]; d_k=2; scores = QK^T/sqrt(2)
    tokens = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    scale = np.sqrt(2.0)
    scores = tokens @ tokens.T / scale
    # softmax rows
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    alpha = exp / exp.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax = axes[0]
    im = ax.imshow(alpha, cmap="YlGnBu", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{alpha[i, j]:.3f}", ha="center", va="center", color="white" if alpha[i, j] > 0.35 else INK, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1, 2], ["t₁", "t₂", "t₃"])
    ax.set_yticks([0, 1, 2], ["q=t₁", "q=t₂", "q=t₃"])
    style_ax(ax, "Self-attention weights α (softmax row)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    # show first-row context
    ctx = alpha[0] @ tokens
    ax2.text(5, 5.3, "Worked row q=t₁", ha="center", fontsize=12, fontweight="bold", color=DEEP)
    ax2.text(5, 4.2, f"scaled scores ≈ [{scores[0,0]:.3f}, {scores[0,1]:.3f}, {scores[0,2]:.3f}]", ha="center", fontsize=10, color=INK)
    ax2.text(5, 3.3, f"α ≈ [{alpha[0,0]:.3f}, {alpha[0,1]:.3f}, {alpha[0,2]:.3f}]  (sums to 1)", ha="center", fontsize=10, color=INK)
    ax2.text(5, 2.3, f"context αV ≈ [{ctx[0]:.3f}, {ctx[1]:.3f}]", ha="center", fontsize=11, color=TEAL, fontweight="bold")
    ax2.text(5, 1.1, "d_k=2; scale 1/√d_k keeps softmax soft", ha="center", fontsize=9, color="#64748b")
    style_ax(ax2, "Attention numerics (synthetic tokens)")
    fig.tight_layout()
    save(fig, "ml_fig_attention.png")


def fig_distill_prune():
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # teacher
    ax.add_patch(FancyBboxPatch((0.5, 1.5), 3.2, 2.2, boxstyle="round,pad=0.05,rounding_size=0.15", facecolor=TEAL, edgecolor="none"))
    ax.text(2.1, 2.9, "Teacher\n(large net)", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    # student
    ax.add_patch(FancyBboxPatch((8.3, 1.8), 3.0, 1.6, boxstyle="round,pad=0.05,rounding_size=0.15", facecolor=GOLD, edgecolor="none"))
    ax.text(9.8, 2.6, "Student\n(edge device)", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.annotate("soft targets\n(temperature T)", xy=(8.2, 2.6), xytext=(4.0, 2.6), arrowprops=dict(arrowstyle="->", color=DEEP, lw=2), color=DEEP, fontsize=10, ha="center")
    ax.text(6, 4.4, "Knowledge distillation + pruning (teaching sketch)", ha="center", fontsize=12, color=INK)
    ax.text(6, 0.6, "Match teacher logits; then prune / quantize student for latency", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_distill_prune.png")


def fig_graph_toy():
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")
    # small patient-similarity graph
    pos = {
        "A": (0.2, 0.85),
        "B": (0.55, 0.95),
        "C": (0.9, 0.75),
        "D": (0.35, 0.45),
        "E": (0.75, 0.4),
        "F": (0.55, 0.1),
    }
    edges = [("A", "B"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"), ("D", "F"), ("E", "F")]
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", lw=1.8, zorder=1)
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.07, facecolor=TEAL if name in "ABD" else GOLD, edgecolor="white", lw=2, zorder=2))
        ax.text(x, y, name, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=3)
    ax.text(0.5, 1.08, "Toy patient-similarity graph", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(0.5, -0.08, "Nodes = patients; edges = shared pathway / phenotype link", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_graph_toy.png")


def fig_viz_hygiene():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    months = np.arange(1, 7)
    rates = np.array([48, 49, 50, 51, 50, 52])  # door-to-needle minutes
    ax = axes[0]
    ax.plot(months, rates, "o-", color=TEAL, lw=2.2)
    ax.set_ylim(0, 70)
    ax.set_xlabel("month")
    ax.set_ylabel("door-to-needle (min)")
    style_ax(ax, "Honest baseline (y from 0)")
    ax2 = axes[1]
    ax2.plot(months, rates, "o-", color="#dc2626", lw=2.2)
    ax2.set_ylim(47, 53)
    ax2.set_xlabel("month")
    ax2.set_ylabel("door-to-needle (min)")
    style_ax(ax2, "Truncated axis exaggerates change")
    fig.suptitle("Visualization hygiene (synthetic dashboard)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_viz_hygiene.png")


def fig_how_to_read():
    """Preface: how to study this ebook."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    steps = [
        (1.2, "Recompute\nby hand", TEAL),
        (4.5, "Map method\n→ clinical claim", DEEP),
        (7.8, "Audit metrics\nbeyond AUC", GOLD),
        (10.8, "External\nvalidation", "#b45309"),
    ]
    for i, (x, lab, c) in enumerate(steps):
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.1, 0.9),
                2.2,
                1.5,
                boxstyle="round,pad=0.04,rounding_size=0.15",
                facecolor=c,
                edgecolor="none",
            )
        )
        ax.text(x, 1.65, lab, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(steps[i + 1][0] - 1.15, 1.65),
                xytext=(x + 1.15, 1.65),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6),
            )
    ax.text(6, 2.9, "How to read this open-source ebook", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(6, 0.35, "Definitions → data → decision impact (not vendor vocabulary)", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_how_to_read.png")


def fig_metric_map():
    """Glossary: discrimination vs calibration vs utility."""
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.4, 2.2, 2.8, 2.0, "Discrimination\nAUC / ROC", TEAL),
        (3.6, 2.2, 2.8, 2.0, "Calibration\nreliability", GOLD),
        (6.8, 2.2, 2.8, 2.0, "Utility\nnet benefit", DEEP),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor=c, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(5, 1.2, "All three needed for clinical decisions — none alone is enough", ha="center", fontsize=11, color=INK)
    ax.text(5, 0.4, "Teaching map for glossary metrics (original)", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_metric_map.png")


def fig_ols_fit():
    """Ch08 companion: four-point NIHSS–volume OLS line (exact worked numbers)."""
    x = np.array([4.0, 8.0, 10.0, 14.0])
    y = np.array([12.0, 20.0, 22.0, 30.0])
    b1 = 23 / 13
    b0 = 66 / 13
    xs = np.linspace(3, 15, 50)
    ys = b0 + b1 * xs
    yhat = b0 + b1 * x
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(xs, ys, color=TEAL, lw=2.2, label=r"$\hat y = 66/13 + (23/13)x$")
    ax.scatter(x, y, s=60, c=GOLD, zorder=3, edgecolors=INK, linewidths=0.6, label="data")
    for xi, yi, yhi in zip(x, y, yhat):
        ax.plot([xi, xi], [yi, yhi], color="#e11d48", lw=1.4, alpha=0.85)
    ax.scatter([9], [21], s=40, c=DEEP, marker="x", zorder=4, label=r"$(\bar x,\bar y)$")
    ax.set_xlabel("admission NIHSS (toy)")
    ax.set_ylabel("infarct volume (toy units)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "OLS fit (chapter 8 worked example)")
    ax.text(0.98, 0.05, "RSS≈1.23  R²≈0.992", transform=ax.transAxes, ha="right", fontsize=9, color="#64748b")
    save(fig, "ml_fig_ols_fit.png")


def fig_core_functions():
    """Ch00 Fig 0.2: core ML function catalog."""
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4))
    x = np.linspace(-3, 3, 400)
    plots = [
        (axes[0, 0], x, 0.5 * x + 0.2, "Linear", TEAL),
        (axes[0, 1], x, 0.35 * x**2 - 0.4, "Quadratic", DEEP),
        (axes[0, 2], x, np.exp(0.7 * x) * 0.15, "Exponential", GOLD),
        (axes[1, 0], np.linspace(0.05, 4, 400), np.log(np.linspace(0.05, 4, 400)), "Logarithmic", "#b45309"),
        (axes[1, 1], x, 1 / (1 + np.exp(-x)), "Sigmoid", TEAL),
        (axes[1, 2], x, np.maximum(0, x), "ReLU", DEEP),
    ]
    for ax, xx, yy, title, c in plots:
        ax.plot(xx, yy, color=c, lw=2.2)
        ax.axhline(0, color="#e2e8f0", lw=1)
        ax.axvline(0, color="#e2e8f0", lw=1)
        style_ax(ax, title)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("f(x)", fontsize=9)
    fig.suptitle("Core functions of machine learning (teaching catalog)", color=INK, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "ml_fig_core_functions.png")


def fig_bias_capacity():
    """Ch01 Fig 1.6: train vs validation error vs capacity."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    cap = np.linspace(0.5, 10, 80)
    train = 0.55 * np.exp(-0.35 * cap) + 0.05
    val = 0.12 + 0.45 * np.exp(-0.55 * cap) + 0.018 * (cap - 3.2) ** 2
    ax.plot(cap, train, color=TEAL, lw=2.4, label="training error")
    ax.plot(cap, val, color=GOLD, lw=2.4, label="validation error")
    best = cap[np.argmin(val)]
    ax.axvline(best, color="#94a3b8", ls="--", lw=1.5, label="sweet-spot capacity")
    ax.set_xlabel("model capacity (synthetic axis)")
    ax.set_ylabel("error")
    ax.set_ylim(0, 0.7)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Bias–variance: capacity vs error (synthetic)")
    save(fig, "ml_fig_bias_capacity.png")


def fig_elbow_wss():
    """Ch04 Fig 4.5: elbow plot for six-point toy set WSS."""
    # Approximate chapter narrative: sharp drop 74.2→4.0 from k=1 to k=2 then flat
    ks = np.array([1, 2, 3, 4, 5, 6])
    wss = np.array([74.2, 4.0, 2.6, 1.5, 0.8, 0.2])
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(ks, wss, "o-", color=TEAL, lw=2.2, markersize=8)
    ax.annotate("elbow", xy=(2, 4.0), xytext=(3.2, 35), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6), color=GOLD, fontsize=11, fontweight="bold")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("within-cluster sum of squares")
    ax.set_xticks(ks)
    style_ax(ax, "Elbow plot of WSS vs k (six-point toy set)")
    save(fig, "ml_fig_elbow_wss.png")


def fig_activations():
    """Ch10 Fig 10.2: four activation functions."""
    z = np.linspace(-4, 4, 400)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6))
    acts = [
        (axes[0, 0], 1 / (1 + np.exp(-z)), "Sigmoid", TEAL),
        (axes[0, 1], np.tanh(z), "Tanh", DEEP),
        (axes[1, 0], np.maximum(0, z), "ReLU", GOLD),
        (axes[1, 1], np.where(z > 0, z, 0.1 * z), "Leaky ReLU (α=0.1)", "#b45309"),
    ]
    for ax, y, title, c in acts:
        ax.plot(z, y, color=c, lw=2.3)
        ax.axhline(0, color="#e2e8f0", lw=1)
        ax.axvline(0, color="#e2e8f0", lw=1)
        style_ax(ax, title)
        ax.set_xlabel("z")
        ax.set_ylabel("φ(z)")
    fig.suptitle("Activation functions (teaching panel)", color=INK, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "ml_fig_activations.png")


def fig_triplet_ssl():
    """Ch11 Fig 11.3: triplet loss with chapter worked numbers."""
    # a=(0,0), p=(0.3,0.4) d=0.5, m=0.2 → boundary radius 0.7
    # easy n=(0.8,0.6) d=1.0 L=0; hard n'=(0.4,0.3) d=0.5 L=0.2
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.set_aspect("equal")
    a = np.array([0.0, 0.0])
    p = np.array([0.3, 0.4])
    n_easy = np.array([0.8, 0.6])
    n_hard = np.array([0.4, 0.3])
    circ = plt.Circle(a, 0.7, fill=False, ls="--", color="#94a3b8", lw=1.6, label="margin boundary d(a,p)+m=0.7")
    ax.add_patch(circ)
    ax.scatter(*a, s=120, c=TEAL, zorder=3, label="anchor a")
    ax.scatter(*p, s=100, c=GOLD, zorder=3, label="positive p")
    ax.scatter(*n_easy, s=100, c="#64748b", zorder=3, label="easy neg n (L=0)")
    ax.scatter(*n_hard, s=100, c="#dc2626", zorder=3, label="hard neg n′ (L=0.2)")
    ax.annotate("", xy=p, xytext=a, arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate("pull +", xy=((a[0] + p[0]) / 2 - 0.05, (a[1] + p[1]) / 2 + 0.05), color=TEAL, fontsize=9)
    ax.annotate("", xy=(n_hard[0] * 1.35, n_hard[1] * 1.35), xytext=n_hard, arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.6))
    ax.annotate("push −", xy=(0.62, 0.55), color="#dc2626", fontsize=9)
    ax.set_xlim(-0.35, 1.15)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax, "Triplet loss (chapter worked numbers)")
    ax.text(0.5, -0.28, "L = max(0, d(a,p) − d(a,n) + m); m=0.2", transform=ax.transAxes, ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_triplet_ssl.png")


def fig_value_iteration():
    """Ch13 Fig 13.3: value iteration on two-state MDP."""
    gamma = 0.9
    # Simulate VI for plotting (matches chapter narrative directionally)
    V1, V2 = [0.0], [0.0]
    for _ in range(40):
        s1 = max(1 + gamma * V1[-1], 0 + gamma * V2[-1])  # Stay vs Go
        s2 = max(2 + gamma * V2[-1], 0 + gamma * V1[-1])
        V1.append(s1)
        V2.append(s2)
    it = np.arange(len(V1))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(it, V1, "o-", color=TEAL, lw=2, markersize=4, label=r"$V(s_1)$")
    ax.plot(it, V2, "s-", color=GOLD, lw=2, markersize=4, label=r"$V(s_2)$")
    ax.axhline(18, color=TEAL, ls="--", lw=1.2, alpha=0.7, label=r"$V^*(s_1)=18$")
    ax.axhline(20, color=GOLD, ls="--", lw=1.2, alpha=0.7, label=r"$V^*(s_2)=20$")
    ax.axvline(3, color="#94a3b8", ls=":", lw=1.4)
    ax.text(3.2, 8, "greedy flip\nGo in s₁", fontsize=9, color=DEEP)
    ax.set_xlabel("value-iteration sweep")
    ax.set_ylabel("state value")
    ax.set_xlim(0, 25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_ax(ax, "Value iteration on two-state MDP (γ=0.9)")
    save(fig, "ml_fig_value_iteration.png")


def fig_curriculum_map():
    """Preface: curriculum blocks of this open-source ebook."""
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    blocks = [
        (0.4, "Math &\nprob", TEAL),
        (2.7, "Unsup /\nfeatures", DEEP),
        (5.0, "Supervised\nmethods", GOLD),
        (7.3, "Deep &\nSSL / RL", "#b45309"),
        (9.6, "Graphs &\ndata risks", "#64748b"),
        (11.9, "Senior\npractice", INK),
    ]
    for i, (x, lab, c) in enumerate(blocks):
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.1),
                2.0,
                1.8,
                boxstyle="round,pad=0.04,rounding_size=0.14",
                facecolor=c,
                edgecolor="none",
            )
        )
        ax.text(x + 1.0, 2.0, lab, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if i < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(blocks[i + 1][0] - 0.05, 2.0),
                xytext=(x + 2.05, 2.0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5),
            )
    ax.text(7, 3.55, "Curriculum map — open-source ML ebook path", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(7, 0.45, "Foundations → methods → multimodal / RL → appraisal & deployment", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_curriculum_map.png")


def fig_glossary_families():
    """Glossary: families of terms used in clinical ML appraisal."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    families = [
        (0.4, 3.6, 2.9, 2.0, "Learning\nparadigms", TEAL, "supervised\nunsupervised\nRL / SSL"),
        (3.55, 3.6, 2.9, 2.0, "Model\nmechanics", DEEP, "loss, grad\nregularization\ncapacity"),
        (6.7, 3.6, 2.9, 2.0, "Evaluation\nlanguage", GOLD, "AUC, PPV\ncalibration\nnet benefit"),
        (0.4, 0.6, 2.9, 2.0, "Data &\ntime", "#64748b", "index time\nleakage\ndrift"),
        (3.55, 0.6, 2.9, 2.0, "Causal\ncaution", "#b45309", "confounding\nDAG / collider\ntransport"),
        (6.7, 0.6, 2.9, 2.0, "Deploy &\ngovern", INK, "model card\nmonitoring\nrollback"),
    ]
    for x, y, w, h, title, c, body in families:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14", facecolor=c, edgecolor="none", alpha=0.95))
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + 0.7, body, ha="center", va="center", color="white", fontsize=9)
    ax.text(5, 5.7, "Glossary term families (teaching taxonomy)", ha="center", fontsize=12, color=INK, fontweight="bold")
    save(fig, "ml_fig_glossary_families.png")


def fig_ppv_prevalence():
    """Scientific: PPV vs prevalence at fixed sens/spec (ch03 LVO numbers)."""
    sens, spec = 0.85, 0.70
    prev = np.linspace(0.01, 0.80, 200)
    ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(prev, ppv, color=TEAL, lw=2.6, label=f"sens={sens}, spec={spec}")
    for p, lab in [(0.05, "prev 5%"), (0.20, "prev 20%")]:
        y = (sens * p) / (sens * p + (1 - spec) * (1 - p))
        ax.plot(p, y, "o", color=GOLD, markersize=9, zorder=4)
        ax.annotate(f"{lab}\nPPV≈{y:.2f}", xy=(p, y), xytext=(p + 0.12, y - 0.12 if p < 0.15 else y + 0.08),
                    arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.3), color=DEEP, fontsize=9)
    ax.set_xlabel("disease prevalence P(D+)")
    ax.set_ylabel("positive predictive value P(D+|test+)")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_ax(ax, "PPV vs prevalence (LVO screen: sens 0.85, spec 0.70)")
    ax.text(0.98, 0.05, "LR+ = sens/(1−spec) ≈ 2.83 travels; PPV does not", transform=ax.transAxes,
            ha="right", fontsize=8, color="#64748b")
    save(fig, "ml_fig_ppv_prevalence.png")


def fig_lifecycle_deploy():
    """Closing: model lifecycle from design to drift monitoring."""
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    steps = [
        (0.5, "Design\nindex time", TEAL),
        (2.9, "Fit &\nregularize", DEEP),
        (5.3, "Validate\nexternal", GOLD),
        (7.7, "Calibrate\n& utility", "#b45309"),
        (10.1, "Deploy\ncard", "#64748b"),
        (12.5, "Monitor\n& drift", INK),
    ]
    for i, (x, lab, c) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 0.85), 2.0, 1.5, boxstyle="round,pad=0.03,rounding_size=0.12", facecolor=c, edgecolor="none"))
        ax.text(x + 1.0, 1.6, lab, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][0] - 0.05, 1.6), xytext=(x + 2.05, 1.6),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(7, 2.8, "Senior practice lifecycle (design → drift)", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(7, 0.3, "Every arrow is a place to stop and audit — not a pipeline to skip", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_lifecycle_deploy.png")


def fig_decision_curve():
    """Scientific: net-benefit decision curve (synthetic prognostic model)."""
    # Treat-all / treat-none / model net benefit (classic Vickers-style teaching sketch)
    pt = np.linspace(0.01, 0.60, 120)
    # Synthetic well-calibrated model with moderate discrimination
    prevalence = 0.25
    # Approximate: model NB from smooth ROC-like TPR/FPR vs threshold on risk
    # Map threshold to TPR/FPR with logistic-like response
    tpr = 1 / (1 + np.exp(12 * (pt - 0.28)))
    fpr = 1 / (1 + np.exp(10 * (pt - 0.18)))
    nb_model = tpr * prevalence - fpr * (1 - prevalence) * (pt / (1 - pt))
    nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
    nb_none = np.zeros_like(pt)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(pt, nb_model, color=TEAL, lw=2.5, label="model (synthetic)")
    ax.plot(pt, nb_all, color=GOLD, lw=1.8, ls="--", label="treat all")
    ax.plot(pt, nb_none, color="#94a3b8", lw=1.5, label="treat none")
    ax.axhline(0, color="#e2e8f0", lw=1)
    ax.set_xlabel("threshold probability (pt)")
    ax.set_ylabel("net benefit")
    ax.set_xlim(0, 0.60)
    ax.set_ylim(-0.05, 0.30)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Decision curve: net benefit vs threshold (synthetic)")
    ax.text(0.98, 0.05, "Utility claim is threshold-specific — not an AUC surrogate", transform=ax.transAxes,
            ha="right", fontsize=8, color="#64748b")
    save(fig, "ml_fig_decision_curve.png")


def fig_clt_sampling():
    """Scientific densify ch03: CLT — sampling distribution of the mean."""
    rng = np.random.default_rng(42)
    # Skewed parent: Exponential(λ=1) mean=1
    n_small, n_large = 5, 40
    n_rep = 4000
    means_small = rng.exponential(1.0, size=(n_rep, n_small)).mean(axis=1)
    means_large = rng.exponential(1.0, size=(n_rep, n_large)).mean(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.4))
    # Parent
    xs = np.linspace(0, 6, 200)
    axes[0].plot(xs, np.exp(-xs), color=TEAL, lw=2.2)
    axes[0].fill_between(xs, np.exp(-xs), color=TEAL, alpha=0.2)
    style_ax(axes[0], "Parent: Exp(1) skewed")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("density")
    # n=5
    axes[1].hist(means_small, bins=40, density=True, color=GOLD, alpha=0.85, edgecolor="none")
    mu, sd = means_small.mean(), means_small.std()
    zg = np.linspace(means_small.min(), means_small.max(), 200)
    axes[1].plot(zg, 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((zg - mu) / sd) ** 2), color=DEEP, lw=2)
    style_ax(axes[1], f"Mean of n={n_small} (still skewed)")
    axes[1].set_xlabel(r"$\bar x$")
    # n=40
    axes[2].hist(means_large, bins=40, density=True, color=TEAL, alpha=0.85, edgecolor="none")
    mu2, sd2 = means_large.mean(), means_large.std()
    zg2 = np.linspace(means_large.min(), means_large.max(), 200)
    axes[2].plot(zg2, 1 / (sd2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((zg2 - mu2) / sd2) ** 2), color=GOLD, lw=2)
    style_ax(axes[2], f"Mean of n={n_large} ≈ Normal")
    axes[2].set_xlabel(r"$\bar x$")
    fig.suptitle("Central limit theorem (synthetic Exp samples)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_clt_sampling.png")


def fig_mle_bernoulli():
    """Scientific densify ch03: Bernoulli log-likelihood surface for MLE."""
    # k=7 successes in n=10 trials → MLE p=0.7
    n, k = 10, 7
    p = np.linspace(0.02, 0.98, 300)
    ll = k * np.log(p) + (n - k) * np.log(1 - p)
    p_hat = k / n
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(p, ll, color=TEAL, lw=2.5)
    ax.axvline(p_hat, color=GOLD, ls="--", lw=1.8, label=rf"MLE $\hat p = k/n = {p_hat:.1f}$")
    ax.plot(p_hat, k * np.log(p_hat) + (n - k) * np.log(1 - p_hat), "o", color=GOLD, markersize=9, zorder=4)
    ax.set_xlabel(r"parameter $p$")
    ax.set_ylabel(r"log-likelihood $\ell(p)$")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, rf"Bernoulli MLE: {k} successes in {n} trials")
    ax.text(0.98, 0.08, r"$\ell(p)=k\log p+(n-k)\log(1-p)$", transform=ax.transAxes,
            ha="right", fontsize=9, color="#64748b")
    save(fig, "ml_fig_mle_bernoulli.png")


def fig_learning_curves():
    """Ch01: train/val error vs training set size (synthetic)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    n = np.linspace(50, 2000, 40)
    train = 0.08 + 0.12 * np.exp(-n / 400) + 0.01 * np.sin(n / 180)
    val = 0.18 + 0.35 * np.exp(-n / 550) + 0.012 * np.sin(n / 200 + 0.5)
    ax.plot(n, train, color=TEAL, lw=2.3, label="training error")
    ax.plot(n, val, color=GOLD, lw=2.3, label="validation error")
    ax.fill_between(n, train, val, color=SOFT, alpha=0.7)
    ax.set_xlabel("training set size (synthetic n)")
    ax.set_ylabel("error")
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Learning curves vs sample size (synthetic)")
    ax.text(0.98, 0.08, "Gap shrinks with more data — capacity fixed", transform=ax.transAxes,
            ha="right", fontsize=8, color="#64748b")
    save(fig, "ml_fig_learning_curves.png")


def fig_confusion_annotated():
    """Ch09: annotated confusion matrix with rates."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    # TN FP / FN TP
    mat = np.array([[82.0, 8.0], [11.0, 49.0]])
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=90)
    labels = [["TN 82", "FP 8"], ["FN 11", "TP 49"]]
    rates = [
        [f"spec≈{82/90:.2f}", f"FPR≈{8/90:.2f}"],
        [f"FNR≈{11/60:.2f}", f"sens≈{49/60:.2f}"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.12, labels[i][j], ha="center", va="center", color="white", fontsize=13, fontweight="bold")
            ax.text(j, i + 0.28, rates[i][j], ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks([0, 1], ["Pred −", "Pred +"])
    ax.set_yticks([0, 1], ["True −", "True +"])
    style_ax(ax, "Confusion matrix with rates (synthetic n=150)")
    ppv = 49 / 57
    npv = 82 / 93
    ax.text(0.5, -0.18, f"PPV≈{ppv:.2f}   NPV≈{npv:.2f}   Acc≈{(82+49)/150:.2f}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_confusion_annotated.png")


# ---------------------------------------------------------------------------
# Swarm cycle-2 scientific panels (bias–variance, PR, early stop, SGD noise, paths)
# ---------------------------------------------------------------------------


def fig_bias_variance_decomp():
    """Ch01 scientific: squared-error decomposition vs capacity (synthetic)."""
    cap = np.linspace(0.4, 10.0, 200)
    bias2 = 0.55 * np.exp(-0.42 * cap) + 0.02
    variance = 0.02 + 0.012 * (cap**1.55)
    noise = np.full_like(cap, 0.08)
    total = bias2 + variance + noise
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.fill_between(cap, 0, noise, color="#94a3b8", alpha=0.35, label=r"irreducible $\sigma^2$")
    ax.fill_between(cap, noise, noise + bias2, color=TEAL, alpha=0.45, label=r"(bias)$^2$")
    ax.fill_between(cap, noise + bias2, total, color=GOLD, alpha=0.50, label="variance")
    ax.plot(cap, total, color=INK, lw=2.4, label="expected error")
    best = cap[np.argmin(total)]
    ax.axvline(best, color=DEEP, ls="--", lw=1.6, label="min expected error")
    ax.set_xlabel("model capacity (synthetic axis)")
    ax.set_ylabel("squared error components")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    style_ax(ax, r"Bias–variance decomposition: $E=\mathrm{bias}^2+\mathrm{var}+\sigma^2$")
    ax.text(
        0.98,
        0.06,
        "Sweet spot balances falling bias against rising variance",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#64748b",
    )
    save(fig, "ml_fig_bias_variance_decomp.png")


def fig_precision_recall():
    """Ch09 scientific: precision–recall curve under class imbalance (synthetic)."""
    # Synthetic score distributions: positives ~ N(0.72, 0.18), negatives ~ N(0.32, 0.16)
    rng = np.random.default_rng(11)
    n_pos, n_neg = 80, 720  # prevalence ≈ 10%
    scores_pos = rng.normal(0.72, 0.18, n_pos)
    scores_neg = rng.normal(0.32, 0.16, n_neg)
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    thr = np.linspace(scores.max(), scores.min(), 200)
    prec, rec, fpr, tpr = [], [], [], []
    for t in thr:
        pred = scores >= t
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        tn = np.sum((pred == 0) & (labels == 0))
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        prec.append(p)
        rec.append(r)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
        tpr.append(r)
    prec, rec, fpr, tpr = map(np.array, (prec, rec, fpr, tpr))
    # Average precision (step-wise trapezoid on recall)
    order = np.argsort(rec)
    # numpy ≥2: trapezoid; older: trapz
    _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    ap = _trap(prec[order], rec[order])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    ax.plot(rec, prec, color=TEAL, lw=2.4, label=f"model AP≈{ap:.2f}")
    ax.axhline(n_pos / (n_pos + n_neg), color="#94a3b8", ls="--", lw=1.4, label="chance (prevalence)")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Precision–recall (prevalence ≈ 10%)")
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color=GOLD, lw=2.4, label="same model ROC")
    ax2.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.3, label="chance")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.legend(frameon=False, fontsize=8)
    style_ax(ax2, "ROC can look rosy under imbalance")
    fig.suptitle("PR vs ROC on the same imbalanced synthetic cohort", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_precision_recall.png")


def fig_early_stopping():
    """Ch08/10 scientific: train vs validation loss with early-stop epoch."""
    epochs = np.arange(0, 61)
    # Training loss keeps falling; validation bottoms then rises (overfit)
    train = 1.15 * np.exp(-0.055 * epochs) + 0.08 + 0.01 * np.sin(epochs / 3)
    val = 0.95 * np.exp(-0.048 * epochs) + 0.012 * np.maximum(0, epochs - 22) ** 1.35 / 40 + 0.22
    val += 0.008 * np.sin(epochs / 4 + 0.7)
    stop = int(np.argmin(val))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(epochs, train, color=TEAL, lw=2.4, label="training loss")
    ax.plot(epochs, val, color=GOLD, lw=2.4, label="validation loss")
    ax.axvline(stop, color=DEEP, ls="--", lw=1.8, label=f"early stop @ epoch {stop}")
    ax.plot(stop, val[stop], "o", color=DEEP, markersize=9, zorder=5)
    ax.annotate(
        "patience window\n(stop before overfit)",
        xy=(stop, val[stop]),
        xytext=(stop + 8, val[stop] + 0.28),
        arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.4),
        color=DEEP,
        fontsize=9,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.35)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Early stopping: validation bottoms while train still falls")
    ax.text(
        0.98,
        0.06,
        "Choose patience on validation only — never on final test",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#64748b",
    )
    save(fig, "ml_fig_early_stopping.png")


def fig_gradient_noise():
    """Ch08/16 scientific: batch GD vs mini-batch SGD paths (gradient noise)."""
    # Quadratic valley: J(w1,w2) = a*w1^2 + b*w2^2, a>>b elongated
    a, b = 4.0, 0.25
    # Contours
    w1 = np.linspace(-2.2, 2.2, 200)
    w2 = np.linspace(-2.2, 2.2, 200)
    W1, W2 = np.meshgrid(w1, w2)
    J = a * W1**2 + b * W2**2
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    cs = ax.contour(W1, W2, J, levels=12, colors="#cbd5e1", linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    # Batch GD (exact gradient)
    eta = 0.12
    p = np.array([2.0, 1.6], dtype=float)
    path_batch = [p.copy()]
    for _ in range(35):
        g = np.array([2 * a * p[0], 2 * b * p[1]])
        p = p - eta * g
        path_batch.append(p.copy())
    path_batch = np.array(path_batch)
    # SGD: exact grad + isotropic noise scaled by 1/sqrt(batch)
    rng = np.random.default_rng(5)
    p = np.array([2.0, 1.6], dtype=float)
    path_sgd = [p.copy()]
    noise_scale = 0.55
    for _ in range(55):
        g = np.array([2 * a * p[0], 2 * b * p[1]])
        g = g + rng.normal(0, noise_scale, size=2)
        p = p - eta * g
        path_sgd.append(p.copy())
    path_sgd = np.array(path_sgd)
    ax.plot(path_batch[:, 0], path_batch[:, 1], "o-", color=TEAL, lw=2.0, markersize=3.5, label="batch GD (low noise)")
    ax.plot(path_sgd[:, 0], path_sgd[:, 1], ".-", color=GOLD, lw=1.4, markersize=4, alpha=0.9, label="mini-batch SGD (noisy)")
    ax.plot(0, 0, "*", color=DEEP, markersize=14, label="optimum", zorder=6)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Gradient noise: batch vs mini-batch paths (synthetic quadratic)")
    ax.text(
        0.02,
        0.04,
        "Noise ∝ 1/√batch — helpful exploration, slower exact convergence",
        transform=ax.transAxes,
        fontsize=8,
        color="#64748b",
    )
    save(fig, "ml_fig_gradient_noise.png")


def fig_regularization_path():
    """Ch08 scientific: Lasso/Ridge coefficient paths vs log10(λ) (synthetic)."""
    # Synthetic path: five standardized predictors with staggered entry (Lasso) vs smooth shrink (Ridge)
    loglam = np.linspace(-3.0, 1.5, 160)
    lam = 10**loglam
    # True-ish soft-threshold style paths for Lasso
    beta_true = np.array([1.8, -1.2, 0.9, 0.45, -0.25])
    names = [r"$\beta_1$ NIHSS", r"$\beta_2$ age", r"$\beta_3$ glucose", r"$\beta_4$ SBP", r"$\beta_5$ sex"]
    colors = [TEAL, GOLD, DEEP, "#b45309", "#64748b"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    # Lasso: soft-threshold-ish with different critical λ
    crit = np.array([1.6, 1.1, 0.75, 0.40, 0.18])
    for i in range(5):
        # coefficient enters when λ < crit[i]; magnitude shrinks roughly (crit - λ)_+
        path = np.sign(beta_true[i]) * np.maximum(0.0, np.abs(beta_true[i]) * (1 - lam / crit[i]))
        axes[0].plot(loglam, path, color=colors[i], lw=2.0, label=names[i])
    axes[0].axvline(-0.4, color="#94a3b8", ls="--", lw=1.3)
    axes[0].text(-0.35, 1.55, r"$\lambda_{\min}$", fontsize=9, color="#64748b")
    axes[0].axvline(0.15, color="#94a3b8", ls=":", lw=1.3)
    axes[0].text(0.2, 1.55, r"$\lambda_{1se}$", fontsize=9, color="#64748b")
    axes[0].set_xlabel(r"$\log_{10}\lambda$")
    axes[0].set_ylabel("coefficient")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    style_ax(axes[0], "Lasso path (sparse entry)")
    # Ridge: continuous shrink, never exact zero
    for i in range(5):
        path = beta_true[i] / (1.0 + 2.2 * lam)
        axes[1].plot(loglam, path, color=colors[i], lw=2.0, label=names[i])
    axes[1].set_xlabel(r"$\log_{10}\lambda$")
    style_ax(axes[1], "Ridge path (dense shrink)")
    fig.suptitle("Regularization paths for five standardized predictors (synthetic)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_regularization_path.png")


# ---------------------------------------------------------------------------
# Legacy numbered PNGs (00_*.png … 17_*.png)
# These files already exist under docs/assets/figures/ and are linked from
# chapter openings. They are FROZEN historical assets — not regenerated here.
# Do not re-import from DOCX. Prefer ml_fig_* for all new art.
# ---------------------------------------------------------------------------
LEGACY_NUMBERED_ASSETS = (
    "00_vector_matrix.png",
    "01_gradient_descent.png",
    "02_viz_anatomy.png",
    "03_bayes_update.png",
    "04_kmeans.png",
    "07_pca_projection.png",
    "08_regression_fit.png",
    "09_supervised_map.png",
    "10_mlp_architecture.png",
    "16_leakage_timeline.png",
    "17_roc_curve.png",
)


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
    # Chapter-specific originals (cycle-1)
    fig_association_rules()
    fig_feature_pipeline()
    fig_attention_toy()
    fig_distill_prune()
    fig_graph_toy()
    fig_viz_hygiene()
    fig_how_to_read()
    fig_metric_map()
    fig_ols_fit()
    # High-value bare-caption panels (cycle-2)
    fig_core_functions()
    fig_bias_capacity()
    fig_elbow_wss()
    fig_activations()
    fig_triplet_ssl()
    fig_value_iteration()
    # Swarm cycle-1 densify (preface / glossary / closing + scientific plots)
    fig_curriculum_map()
    fig_glossary_families()
    fig_ppv_prevalence()
    fig_lifecycle_deploy()
    fig_decision_curve()
    fig_clt_sampling()
    fig_mle_bernoulli()
    fig_learning_curves()
    fig_confusion_annotated()
    # Swarm cycle-2 scientific panels
    fig_bias_variance_decomp()
    fig_precision_recall()
    fig_early_stopping()
    fig_gradient_noise()
    fig_regularization_path()
    print("DONE figures in", OUT)
    missing_legacy = [n for n in LEGACY_NUMBERED_ASSETS if not (OUT / n).exists()]
    if missing_legacy:
        print("WARN missing legacy assets:", missing_legacy)
    else:
        print("LEGACY numbered assets present:", len(LEGACY_NUMBERED_ASSETS))


if __name__ == "__main__":
    main()
