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
# Swarm 3h cycle-1 densify (preface / glossary / ch12 / ch13)
# ---------------------------------------------------------------------------


def fig_claim_types():
    """Preface: three clinical claim types readers must keep separate."""
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    boxes = [
        (0.35, 1.0, 3.5, 2.4, "Prediction", TEAL, "Will this event\nhappen?\n(score / risk)"),
        (4.25, 1.0, 3.5, 2.4, "Etiology", GOLD, "Why did it\nhappen?\n(cause / mechanism)"),
        (8.15, 1.0, 3.5, 2.4, "Decision support", DEEP, "What should we\ndo next?\n(action / utility)"),
    ]
    for x, y, w, h, title, c, body in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.16",
                facecolor=c,
                edgecolor="none",
                alpha=0.95,
            )
        )
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + 0.85, body, ha="center", va="center", color="white", fontsize=10)
    ax.text(6, 3.85, "Three claim types — do not swap them mid-journal-club", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(6, 0.35, "A high AUROC predicts; it does not prove cause or mandate action", ha="center", fontsize=9, color="#64748b")
    save(fig, "ml_fig_claim_types.png")


def fig_accuracy_trap():
    """Glossary scientific: accuracy looks fine while sensitivity collapses (imbalance)."""
    # Fixed sens/spec vs prevalence; accuracy = sens*p + spec*(1-p)
    sens_bad, spec_good = 0.40, 0.98  # majority-class-leaning classifier
    sens_bal, spec_bal = 0.85, 0.85
    prev = np.linspace(0.01, 0.50, 200)
    acc_bad = sens_bad * prev + spec_good * (1 - prev)
    acc_bal = sens_bal * prev + spec_bal * (1 - prev)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    ax = axes[0]
    ax.plot(prev, acc_bad, color=GOLD, lw=2.5, label=f"majority-leaning (sens={sens_bad}, spec={spec_good})")
    ax.plot(prev, acc_bal, color=TEAL, lw=2.5, label=f"balanced (sens=spec={sens_bal})")
    ax.axvline(0.08, color="#94a3b8", ls="--", lw=1.3)
    ax.text(0.085, 0.72, "stroke LVO-like\nprevalence ≈ 8%", fontsize=8, color="#64748b")
    ax.set_xlabel("prevalence")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.55, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    style_ax(ax, "Accuracy can flatter a weak screen")
    # Right: 2x2 counts at prev=8%, n=1000
    ax2 = axes[1]
    n, p = 1000, 0.08
    tp = int(round(sens_bad * n * p))
    fn = int(round(n * p)) - tp
    tn = int(round(spec_good * n * (1 - p)))
    fp = int(round(n * (1 - p))) - tn
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    im = ax2.imshow(mat, cmap="YlGnBu", vmin=0, vmax=mat.max())
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax2.text(
                j,
                i,
                f"{labels[i][j]}\n{int(mat[i, j])}",
                ha="center",
                va="center",
                color="white" if mat[i, j] > mat.max() * 0.45 else INK,
                fontsize=11,
                fontweight="bold",
            )
    ax2.set_xticks([0, 1], ["Pred−", "Pred+"])
    ax2.set_yticks([0, 1], ["True−", "True+"])
    acc = (tp + tn) / n
    style_ax(ax2, f"n=1000, prev=8%: acc≈{acc:.0%} but sens={sens_bad:.0%}")
    fig.suptitle("The accuracy trap under class imbalance (synthetic screen)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_accuracy_trap.png")


def fig_causal_mask():
    """Ch12 scientific: causal (lower-triangular) attention mask for decoder LM."""
    n = 6
    # Allowed = 0 (attend), blocked = -inf visualized as 1
    mask = np.triu(np.ones((n, n)), k=1)  # 1 = future (blocked)
    allowed = 1.0 - mask
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    im = ax.imshow(allowed, cmap="YlGnBu", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            lab = "✓" if allowed[i, j] > 0.5 else "−∞"
            ax.text(
                j,
                i,
                lab,
                ha="center",
                va="center",
                color="white" if allowed[i, j] > 0.5 else "#94a3b8",
                fontsize=11,
                fontweight="bold",
            )
    ax.set_xticks(range(n), [f"k{j}" for j in range(n)])
    ax.set_yticks(range(n), [f"q{i}" for i in range(n)])
    ax.set_xlabel("key position (past → future)")
    ax.set_ylabel("query position")
    style_ax(ax, "Causal mask: query t sees keys ≤ t only")
    # Right: softmax after mask on one row
    ax2 = axes[1]
    # toy scores for q3 attending to keys 0..5; future masked
    scores = np.array([0.4, 1.1, 0.2, 0.9, 2.5, 3.0])
    scores_masked = scores.copy()
    scores_masked[4:] = -1e9
    exp = np.exp(scores_masked - scores_masked.max())
    alpha = exp / exp.sum()
    x = np.arange(n)
    bars = ax2.bar(x, alpha, color=[TEAL if i <= 3 else "#cbd5e1" for i in range(n)], edgecolor="none")
    ax2.set_xticks(x, [f"t{i}" for i in range(n)])
    ax2.set_ylabel(r"attention weight $\alpha$")
    ax2.set_ylim(0, 1.05)
    ax2.axvline(3.5, color="#dc2626", ls="--", lw=1.5)
    ax2.text(3.6, 0.85, "future\nmasked", color="#dc2626", fontsize=9)
    style_ax(ax2, r"Softmax row for $q_3$ after causal mask")
    fig.suptitle("Decoder causal masking (GPT-style next-token training)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_causal_mask.png")


def fig_dice_iou():
    """Ch12 scientific: Dice vs IoU on two synthetic lesion masks."""
    # Build two binary masks on a small grid
    h, w = 48, 48
    yy, xx = np.mgrid[0:h, 0:w]
    # Ground truth: ellipse center-left
    gt = (((xx - 18) / 10) ** 2 + ((yy - 24) / 14) ** 2) <= 1.0
    # Prediction: shifted ellipse (partial overlap)
    pr = (((xx - 24) / 11) ** 2 + ((yy - 26) / 13) ** 2) <= 1.0
    inter = gt & pr
    union = gt | pr
    only_gt = gt & ~pr
    only_pr = pr & ~gt
    dice = 2 * inter.sum() / (gt.sum() + pr.sum())
    iou = inter.sum() / union.sum()
    # Panel image: color code regions
    rgb = np.ones((h, w, 3))
    rgb[only_gt] = np.array([13, 148, 136]) / 255.0  # teal GT only
    rgb[only_pr] = np.array([201, 162, 39]) / 255.0  # gold pred only
    rgb[inter] = np.array([15, 118, 110]) / 255.0  # deep overlap
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    ax = axes[0]
    ax.imshow(rgb, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    style_ax(ax, "Lesion masks: GT (teal) ∩ pred (gold)")
    ax.text(0.02, 0.05, "overlap = deep teal", transform=ax.transAxes, fontsize=8, color=INK)
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.text(5, 5.3, "Overlap metrics (same pair)", ha="center", fontsize=12, fontweight="bold", color=DEEP)
    ax2.text(5, 4.1, rf"IoU = $|A\cap B|/|A\cup B|$  ≈  {iou:.3f}", ha="center", fontsize=12, color=INK)
    ax2.text(5, 3.2, rf"Dice = $2|A\cap B|/(|A|+|B|)$  ≈  {dice:.3f}", ha="center", fontsize=12, color=INK)
    ax2.text(5, 2.2, rf"Identity: Dice = $2\,\mathrm{{IoU}}/(1+\mathrm{{IoU}})$", ha="center", fontsize=11, color=TEAL, fontweight="bold")
    check = 2 * iou / (1 + iou)
    ax2.text(5, 1.4, f"check: 2·IoU/(1+IoU) ≈ {check:.3f}", ha="center", fontsize=10, color="#64748b")
    ax2.text(5, 0.5, "Dice ≥ IoU always; both punish missed boundary", ha="center", fontsize=9, color="#64748b")
    style_ax(ax2, "Dice vs IoU numerics")
    fig.suptitle("Segmentation overlap metrics (synthetic DWI-like masks)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_dice_iou.png")


def fig_discount_gamma():
    """Ch13 scientific: present value of a delayed unit reward under γ."""
    t = np.arange(0, 41)
    gammas = [0.5, 0.9, 0.95, 0.99]
    colors = [GOLD, TEAL, DEEP, "#b45309"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    for g, c in zip(gammas, colors):
        ax.plot(t, g**t, color=c, lw=2.3, label=rf"$\gamma={g}$")
    ax.set_xlabel("delay t (steps)")
    ax.set_ylabel(r"discount factor $\gamma^t$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, r"Present value of +1 reward at time $t$")
    # Right: infinite-horizon constant reward R each step → R/(1-γ)
    ax2 = axes[1]
    g_grid = np.linspace(0.5, 0.995, 200)
    R = 1.0
    v = R / (1.0 - g_grid)
    ax2.plot(g_grid, v, color=TEAL, lw=2.5)
    for g, lab in [(0.9, r"$\gamma=0.9$ → 10"), (0.99, r"$\gamma=0.99$ → 100")]:
        ax2.plot(g, R / (1 - g), "o", color=GOLD, markersize=8, zorder=4)
        ax2.annotate(lab, xy=(g, R / (1 - g)), xytext=(g - 0.18, R / (1 - g) + 8 if g < 0.95 else R / (1 - g) - 25),
                     fontsize=8, color=DEEP, arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.2))
    ax2.set_xlabel(r"discount $\gamma$")
    ax2.set_ylabel(r"$V = R/(1-\gamma)$ for constant $R=1$")
    ax2.set_ylim(0, 120)
    style_ax(ax2, "Far-sighted agents value the stream more")
    fig.suptitle("Discounting in MDPs (why γ near 1 is far-sighted)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_discount_gamma.png")


def fig_bandit_explore():
    """Ch13 scientific: ε-greedy vs UCB cumulative regret on 5-arm Bernoulli bandit."""
    rng = np.random.default_rng(42)
    means = np.array([0.25, 0.40, 0.55, 0.50, 0.35])  # arm 2 is best
    K = len(means)
    T = 800
    n_runs = 80
    c_ucb = np.sqrt(2.0)

    def run_eps(eps: float) -> np.ndarray:
        regrets = np.zeros((n_runs, T))
        best = means.max()
        for r in range(n_runs):
            counts = np.zeros(K)
            values = np.zeros(K)
            cum = 0.0
            for t in range(1, T + 1):
                if rng.random() < eps or counts.min() == 0:
                    a = int(rng.integers(0, K)) if counts.min() > 0 else int(np.argmin(counts))
                else:
                    a = int(np.argmax(values))
                rew = float(rng.random() < means[a])
                counts[a] += 1
                values[a] += (rew - values[a]) / counts[a]
                cum += best - means[a]
                regrets[r, t - 1] = cum
        return regrets.mean(axis=0)

    def run_ucb() -> np.ndarray:
        regrets = np.zeros((n_runs, T))
        best = means.max()
        for r in range(n_runs):
            counts = np.zeros(K)
            values = np.zeros(K)
            cum = 0.0
            for t in range(1, T + 1):
                if counts.min() == 0:
                    a = int(np.argmin(counts))
                else:
                    bonus = c_ucb * np.sqrt(np.log(t) / counts)
                    a = int(np.argmax(values + bonus))
                rew = float(rng.random() < means[a])
                counts[a] += 1
                values[a] += (rew - values[a]) / counts[a]
                cum += best - means[a]
                regrets[r, t - 1] = cum
        return regrets.mean(axis=0)

    r_greedy = run_eps(0.0)
    r_eps = run_eps(0.10)
    r_ucb = run_ucb()
    steps = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    ax.plot(steps, r_greedy, color="#94a3b8", lw=2.0, label=r"greedy $\varepsilon=0$")
    ax.plot(steps, r_eps, color=GOLD, lw=2.2, label=r"$\varepsilon$-greedy $\varepsilon=0.1$")
    ax.plot(steps, r_ucb, color=TEAL, lw=2.2, label="UCB1")
    ax.set_xlabel("pull t")
    ax.set_ylabel("mean cumulative regret")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, f"5-arm Bernoulli bandit ({n_runs} runs)")
    # Right: true means
    ax2 = axes[1]
    x = np.arange(K)
    ax2.bar(x, means, color=[GOLD if m == means.max() else TEAL for m in means], edgecolor="none")
    ax2.set_xticks(x, [f"arm {i}" for i in range(K)])
    ax2.set_ylabel(r"true mean $\mu_a$")
    ax2.set_ylim(0, 0.75)
    for i, m in enumerate(means):
        ax2.text(i, m + 0.03, f"{m:.2f}", ha="center", fontsize=9, color=INK)
    style_ax(ax2, "True arm means (best = arm 2)")
    fig.suptitle("Exploration pays: ε-greedy and UCB vs pure greedy (synthetic)", color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_bandit_explore.png")


def fig_pca_variance_recon():
    """Ch07 scientific: cumulative variance explained + reconstruction error vs k (synthetic)."""
    # Synthetic eigenvalue spectrum (power-law-ish radiomics / lab panel)
    rng = np.random.default_rng(21)
    d = 20
    lam = 8.0 * np.exp(-0.35 * np.arange(d)) + 0.08 * rng.random(d)
    lam = np.sort(lam)[::-1]
    total = lam.sum()
    frac = lam / total
    cum = np.cumsum(frac)
    # Reconstruction MSE proxy = residual variance fraction (Eckart–Young / Frobenius)
    recon_err = 1.0 - cum
    k = np.arange(1, d + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    ax.bar(k, frac, color=TEAL, alpha=0.75, edgecolor="none", label="per-component fraction")
    ax.plot(k, cum, "o-", color=GOLD, lw=2.2, markersize=4, label="cumulative")
    ax.axhline(0.80, color=DEEP, ls="--", lw=1.4, label="80% guideline (not a guarantee)")
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    ax.axvline(k80, color=DEEP, ls=":", lw=1.3, alpha=0.8)
    ax.annotate(
        f"k≈{k80} for 80%",
        xy=(k80, 0.80),
        xytext=(k80 + 2.5, 0.55),
        fontsize=9,
        color=DEEP,
        arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.2),
    )
    ax.set_xlabel("number of components k")
    ax.set_ylabel("fraction of variance")
    ax.set_xlim(0.5, d + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    style_ax(ax, "Explained variance vs k (synthetic spectrum)")

    ax2 = axes[1]
    ax2.plot(k, recon_err, "o-", color=TEAL, lw=2.3, markersize=4.5, label=r"relative $\|X-X_k\|_F^2/\|X\|_F^2$")
    ax2.fill_between(k, recon_err, color=TEAL, alpha=0.15)
    ax2.axhline(0.20, color=GOLD, ls="--", lw=1.4, label="20% residual energy")
    ax2.set_xlabel("rank-k truncation")
    ax2.set_ylabel("relative reconstruction error")
    ax2.set_xlim(0.5, d + 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=False, fontsize=8)
    style_ax(ax2, "Reconstruction error falls as rank grows")
    fig.suptitle(
        "PCA: variance explained ≠ clinical utility; k is a compression choice (synthetic)",
        color=INK,
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "ml_fig_pca_variance_recon.png")


def fig_vanishing_residual():
    """Ch10 scientific: vanishing gradients in deep sigmoid stacks vs residual skip highway."""
    depth = np.arange(1, 21)
    # Product of |σ'(z)| factors: each layer multiplies by ~0.25 (sigmoid mid) or smaller
    g_sig = 0.22 ** (depth - 1)  # geometric decay of backprop signal
    g_tanh = 0.55 ** (depth - 1)
    g_relu = np.ones_like(depth, dtype=float) * 0.95 ** 0  # roughly unit through open ReLUs
    g_relu = 0.92 ** (0.15 * (depth - 1))  # mild decay from weight scale
    # Residual: additive path keeps a floor on gradient magnitude
    g_res = 0.35 + 0.55 * np.exp(-0.08 * (depth - 1))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    ax.semilogy(depth, g_sig, "o-", color=GOLD, lw=2.2, markersize=4, label=r"sigmoid stack ($\times\sim0.22$/layer)")
    ax.semilogy(depth, g_tanh, "s-", color="#b45309", lw=2.0, markersize=4, label=r"tanh stack ($\times\sim0.55$/layer)")
    ax.semilogy(depth, g_relu, "^-", color=TEAL, lw=2.0, markersize=4, label="ReLU (mild decay)")
    ax.set_xlabel("layer depth (from output backward)")
    ax.set_ylabel(r"relative $|\partial L / \partial h|$ (log scale)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Vanishing: product of Jacobians shrinks early-layer signal")
    ax.text(
        0.98,
        0.08,
        "Early layers starve when many factors < 1",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#64748b",
    )

    ax2 = axes[1]
    # Residual block schematic + gradient floor curve
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6.2)
    ax2.axis("off")
    # Main path boxes
    for x, lab in [(0.6, "h"), (3.6, "F(h)\n(weight layers)"), (7.0, "h+F(h)")]:
        ax2.add_patch(
            FancyBboxPatch(
                (x, 3.2),
                2.0 if "F" not in lab else 2.4,
                1.5,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=TEAL if "F" not in lab else DEEP,
                edgecolor="none",
                alpha=0.92,
            )
        )
        ax2.text(x + (1.0 if "F" not in lab else 1.2), 3.95, lab, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax2.annotate("", xy=(3.5, 3.95), xytext=(2.65, 3.95), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax2.annotate("", xy=(6.9, 3.95), xytext=(6.1, 3.95), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    # Skip arc
    ax2.annotate(
        "",
        xy=(7.2, 4.85),
        xytext=(1.6, 4.85),
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.4, connectionstyle="arc3,rad=-0.35"),
    )
    ax2.text(4.4, 5.55, "identity skip (+)", ha="center", color=GOLD, fontsize=11, fontweight="bold")
    ax2.text(5, 2.4, r"Backprop: $\partial L/\partial h$ keeps a direct $+1$ path", ha="center", fontsize=10, color=DEEP)
    ax2.text(5, 1.6, "Residual nets start near identity maps; deep stacks train", ha="center", fontsize=9, color="#64748b")
    ax2.text(5, 0.9, "without pure multiplicative collapse of gradient scale", ha="center", fontsize=9, color="#64748b")
    ax2.set_title("Residual skip = additive gradient highway", fontsize=12, fontweight="bold", color=INK, pad=8)
    # Mini inset: gradient floor
    inset = ax2.inset_axes([0.08, 0.02, 0.38, 0.28])
    inset.plot(depth, g_sig, color=GOLD, lw=1.5)
    inset.plot(depth, g_res, color=TEAL, lw=1.8)
    inset.set_yscale("log")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("sig vs residual floor", fontsize=7, color=INK)
    for s in inset.spines.values():
        s.set_color("#cbd5e1")
    fig.suptitle("Why depth needs architecture: vanishing gradients vs residual skips (teaching)", color=INK, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_vanishing_residual.png")


def fig_appraisal_checklist():
    """Ch17 closing: full ML appraisal checklist flowchart (teaching graphic)."""
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Two columns of sequential gates + final deploy row
    steps = [
        # (x, y, w, h, title, sub, color)
        (0.4, 8.2, 3.4, 1.3, "1. Question fit", "Prediction ≠ etiology\n≠ decision support", TEAL),
        (4.3, 8.2, 3.4, 1.3, "2. Index & legality", "Timestamp every feature\nvs prediction time", DEEP),
        (8.2, 8.2, 3.4, 1.3, "3. Leakage audit", "No post-outcome proxies\ngrouped splits", GOLD),
        (0.4, 6.2, 3.4, 1.3, "4. Cohort & label", "Who / when / how labeled\nmissingness plan", TEAL),
        (4.3, 6.2, 3.4, 1.3, "5. Fit & capacity", "Match n to model class\nregularize + early stop", DEEP),
        (8.2, 6.2, 3.4, 1.3, "6. Discrimination", "AUC / PR at prevalence\nnot accuracy alone", GOLD),
        (0.4, 4.2, 3.4, 1.3, "7. Calibration", "Reliability plot\nslope & intercept", TEAL),
        (4.3, 4.2, 3.4, 1.3, "8. Utility", "Net benefit / DCA\nat clinical thresholds", DEEP),
        (8.2, 4.2, 3.4, 1.3, "9. External test", "Site / temporal holdout\nexpect recalibration", GOLD),
        (2.0, 1.8, 3.6, 1.5, "10. Deploy card", "Version · threshold ·\nprohibited uses", TEAL),
        (6.4, 1.8, 3.6, 1.5, "11. Monitor drift", "Inputs · scores · labels\nrollback triggers", DEEP),
    ]
    for x, y, w, h, title, sub, c in steps:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", facecolor=c, edgecolor="none", alpha=0.92
            )
        )
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", color="white", fontsize=7.8, alpha=0.95)

    # Flow arrows row-wise
    for y in (8.85, 6.85, 4.85):
        for x0, x1 in ((3.85, 4.25), (7.75, 8.15)):
            ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    # Down from row ends to next row starts (simplified vertical cues)
    ax.annotate("", xy=(2.1, 7.55), xytext=(9.9, 8.15), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2, connectionstyle="arc3,rad=0.15"))
    ax.annotate("", xy=(2.1, 5.55), xytext=(9.9, 6.15), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2, connectionstyle="arc3,rad=0.15"))
    ax.annotate("", xy=(3.8, 3.35), xytext=(2.1, 4.15), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(6.35, 2.55), xytext=(5.65, 2.55), arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))

    ax.text(6, 9.7, "Senior ML appraisal checklist (clinical prediction systems)", ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(
        6,
        0.55,
        "Skip a gate → document why. Prediction success never licenses a causal or sole-care-withdrawal claim.",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    save(fig, "ml_fig_appraisal_checklist.png")


def fig_reliability_ece():
    """Glossary / evaluation: multi-model reliability diagram + ECE (synthetic)."""
    rng = np.random.default_rng(9)
    n = 2000
    # Latent risk then three score systems
    true_p = rng.beta(2.0, 5.0, n)  # low-prevalence-ish clinical risk
    y = rng.random(n) < true_p

    def bin_reliability(scores, y, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        centers, obs, confs, counts = [], [], [], []
        for i in range(n_bins):
            m = (scores >= edges[i]) & (scores < edges[i + 1] if i < n_bins - 1 else scores <= edges[i + 1])
            if m.sum() == 0:
                continue
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            confs.append(scores[m].mean())
            obs.append(y[m].mean())
            counts.append(m.sum())
        centers, obs, confs, counts = map(np.array, (centers, obs, confs, counts))
        ece = np.sum(counts / counts.sum() * np.abs(obs - confs))
        return centers, obs, confs, counts, ece

    # Well-calibrated: scores ≈ true_p with mild noise
    s_cal = np.clip(true_p + rng.normal(0, 0.04, n), 0.02, 0.98)
    # Overconfident: push toward 0/1
    s_over = np.clip(0.5 + 1.6 * (true_p - 0.5) + rng.normal(0, 0.03, n), 0.02, 0.98)
    # Underconfident: shrink toward 0.5
    s_under = np.clip(0.5 + 0.45 * (true_p - 0.5) + rng.normal(0, 0.03, n), 0.02, 0.98)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.5, label="perfect calibration")
    for scores, color, name in (
        (s_cal, TEAL, "well-calibrated"),
        (s_over, GOLD, "overconfident"),
        (s_under, DEEP, "underconfident"),
    ):
        centers, obs, confs, counts, ece = bin_reliability(scores, y.astype(float))
        ax.plot(confs, obs, "o-", color=color, lw=2.0, markersize=5, label=f"{name}  ECE≈{ece:.3f}")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed event frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax, "Reliability diagram (synthetic neuro risk cohort)")

    ax2 = axes[1]
    # Histogram of predicted scores for overconfident vs calibrated
    bins = np.linspace(0, 1, 21)
    ax2.hist(s_cal, bins=bins, density=True, alpha=0.55, color=TEAL, label="well-calibrated scores", edgecolor="none")
    ax2.hist(s_over, bins=bins, density=True, alpha=0.45, color=GOLD, label="overconfident scores", edgecolor="none")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("density")
    ax2.legend(frameon=False, fontsize=8)
    style_ax(ax2, "Score histograms: overconfidence piles at extremes")
    ax2.text(
        0.98,
        0.92,
        "ECE = Σ (n_b/n)·|obs_b − conf_b|",
        transform=ax2.transAxes,
        ha="right",
        fontsize=9,
        color=DEEP,
        fontweight="bold",
    )
    fig.suptitle(
        "Calibration is not AUC: reliability + ECE on one synthetic cohort",
        color=INK,
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "ml_fig_reliability_ece.png")


def fig_capacity_vs_n():
    """Ch10 scientific: validation error vs sample size for low / medium / high capacity."""
    n = np.array([50, 100, 200, 400, 800, 1600, 3200], dtype=float)
    # Synthetic generalization curves (teaching sketch, not a fit to real data)
    # Low capacity: high bias floor, little variance
    err_low = 0.28 + 0.12 * np.sqrt(50 / n) + 0.01 * np.sin(np.log(n))
    # Medium capacity: good tradeoff at moderate n
    err_med = 0.14 + 0.22 * np.sqrt(80 / n) + 0.04 * (50 / n)
    # High capacity: terrible at small n, best at large n
    err_high = 0.08 + 0.55 * np.sqrt(120 / n) + 0.15 * (80 / n) ** 0.9

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(n, err_low, "o-", color=GOLD, lw=2.3, markersize=7, label="low capacity (e.g. ridge)")
    ax.plot(n, err_med, "s-", color=TEAL, lw=2.3, markersize=7, label="medium capacity (e.g. modest MLP)")
    ax.plot(n, err_high, "^-", color=DEEP, lw=2.3, markersize=7, label="high capacity (deep / unregularized)")
    ax.set_xscale("log")
    ax.set_xlabel("training sample size n (log scale)")
    ax.set_ylabel("validation error (synthetic)")
    ax.set_ylim(0.05, 0.75)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Capacity must match data: deeper is not free")
    # annotate crossover
    ax.annotate(
        "small-n: simple wins",
        xy=(80, err_low[1]),
        xytext=(120, 0.62),
        fontsize=9,
        color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
    )
    ax.annotate(
        "large-n: capacity pays",
        xy=(2500, err_high[-1]),
        xytext=(500, 0.12),
        fontsize=9,
        color=DEEP,
        arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.2),
    )
    ax.text(
        0.98,
        0.06,
        "Clinical cohorts often live on the left half of this plot",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#64748b",
    )
    save(fig, "ml_fig_capacity_vs_n.png")


# ---------------------------------------------------------------------------
# Swarm 3h cycle-3 densify (ch06 / ch09 / ch04 / ch12 / ch16)
# ---------------------------------------------------------------------------


def fig_preprocess_fit_split():
    """Ch06 scientific: preprocessing must be fit on train only (leakage diagram)."""
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.0), gridspec_kw={"hspace": 0.55})

    # --- Top: WRONG — fit scaler / imputer / selector on full cohort ---
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.3, 0.9), 11.4, 1.5,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor="#fef2f2", edgecolor="#dc2626", linewidth=1.6,
        )
    )
    # Full cohort bar used for fit
    ax.add_patch(Rectangle((0.6, 1.2), 10.8, 0.7, facecolor="#fca5a5", edgecolor="none"))
    ax.text(6.0, 1.55, "FIT mean / SD / imputer / selector on ALL rows", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#7f1d1d")
    # Fake split labels still drawn under the fit
    for x, w, lab, c in [
        (0.6, 6.5, "Train", TEAL),
        (7.2, 2.0, "Val", GOLD),
        (9.4, 2.0, "Test", DEEP),
    ]:
        ax.add_patch(Rectangle((x, 0.35), w, 0.4, facecolor=c, edgecolor="none", alpha=0.85))
        ax.text(x + w / 2, 0.55, lab, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(0.3, 2.85, "WRONG — preprocessing leakage", fontsize=12, fontweight="bold", color="#dc2626")
    ax.text(11.7, 2.85, "test statistics contaminate fit", ha="right", fontsize=9, color="#b91c1c")

    # --- Bottom: RIGHT — fit on train, freeze, transform val/test ---
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.3, 0.55), 11.4, 2.15,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=SOFT, edgecolor=TEAL, linewidth=1.6,
        )
    )
    # Train fit zone
    ax.add_patch(Rectangle((0.6, 1.55), 6.5, 0.85, facecolor=TEAL, edgecolor="none"))
    ax.text(3.85, 1.98, "FIT on train only → freeze μ, σ, vocab, imputer", ha="center", va="center",
            color="white", fontsize=9.5, fontweight="bold")
    # Val / test transform only
    ax.add_patch(Rectangle((7.3, 1.55), 2.0, 0.85, facecolor=GOLD, edgecolor="none"))
    ax.text(8.3, 1.98, "TRANSFORM\nval", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((9.5, 1.55), 1.9, 0.85, facecolor=DEEP, edgecolor="none"))
    ax.text(10.45, 1.98, "TRANSFORM\ntest", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    # Timeline arrow
    ax.annotate("", xy=(11.3, 1.15), xytext=(0.7, 1.15),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(6.0, 0.85, "time-respecting cohort split  ·  never re-fit on val/test", ha="center",
            fontsize=9, color="#475569")
    ax.text(0.3, 2.9, "RIGHT — fit-transform discipline", fontsize=12, fontweight="bold", color=DEEP)
    ax.text(11.7, 2.9, "nested CV: re-fit inside each fold", ha="right", fontsize=9, color=DEEP)

    fig.suptitle("Feature preprocessing leakage vs honest split timeline", fontsize=13,
                 fontweight="bold", color=INK, y=0.98)
    save(fig, "ml_fig_preprocess_fit_split.png")


def fig_threshold_sens_spec():
    """Ch09 scientific: sensitivity / specificity trade-off vs decision threshold."""
    rng = np.random.default_rng(21)
    # Synthetic score distributions: negatives ~ Beta skew low, positives ~ Beta skew high
    n_neg, n_pos = 800, 200
    scores_neg = rng.beta(2.2, 5.5, size=n_neg)
    scores_pos = rng.beta(5.0, 2.4, size=n_pos)
    thresholds = np.linspace(0.02, 0.98, 120)
    sens, spec = [], []
    for t in thresholds:
        tp = np.sum(scores_pos >= t)
        fn = np.sum(scores_pos < t)
        tn = np.sum(scores_neg < t)
        fp = np.sum(scores_neg >= t)
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
    sens = np.asarray(sens)
    spec = np.asarray(spec)
    youden = sens + spec - 1.0
    t_star = thresholds[np.argmax(youden)]
    s_at = sens[np.argmax(youden)]
    sp_at = spec[np.argmax(youden)]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: score histograms
    ax = axes[0]
    bins = np.linspace(0, 1, 28)
    ax.hist(scores_neg, bins=bins, density=True, alpha=0.55, color="#94a3b8", label="true − (n=800)")
    ax.hist(scores_pos, bins=bins, density=True, alpha=0.65, color=TEAL, label="true + (n=200)")
    ax.axvline(0.50, color=GOLD, ls="--", lw=1.6, label="default t=0.50")
    ax.axvline(t_star, color=DEEP, ls="-", lw=1.8, label=f"Youden t*≈{t_star:.2f}")
    ax.set_xlabel("predicted score / risk")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Score overlap drives the trade-off")

    # Right: sens/spec vs threshold
    ax = axes[1]
    ax.plot(thresholds, sens, color=TEAL, lw=2.4, label="sensitivity (recall)")
    ax.plot(thresholds, spec, color=DEEP, lw=2.4, label="specificity")
    ax.plot(thresholds, youden, color=GOLD, lw=1.8, ls="--", label="Youden J = Se+Sp−1")
    ax.axvline(t_star, color="#64748b", ls=":", lw=1.3)
    ax.scatter([t_star], [s_at], color=TEAL, s=40, zorder=5)
    ax.scatter([t_star], [sp_at], color=DEEP, s=40, zorder=5)
    ax.set_xlabel("decision threshold t")
    ax.set_ylabel("rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style_ax(ax, "Operating point is a clinical choice")
    ax.text(
        0.98, 0.06,
        "Default 0.5 is not sacred — match cost of FN vs FP",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.tight_layout()
    save(fig, "ml_fig_threshold_sens_spec.png")


def fig_silhouette_k():
    """Ch04 scientific: silhouette vs k on synthetic 3-blob data (k-choice)."""
    rng = np.random.default_rng(11)
    # Three well-separated blobs in 2-D
    centers = np.array([[0.0, 0.0], [4.5, 0.3], [1.8, 3.8]])
    pts, true_lab = [], []
    for i, c in enumerate(centers):
        blob = rng.normal(c, 0.55, size=(70, 2))
        pts.append(blob)
        true_lab.append(np.full(70, i))
    X = np.vstack(pts)

    def lloyd(X, k, n_init=8, max_iter=40):
        best_inertia, best_labels, best_cents = np.inf, None, None
        n = X.shape[0]
        for seed in range(n_init):
            r = np.random.default_rng(seed + 100)
            cents = X[r.choice(n, size=k, replace=False)].copy()
            labels = np.zeros(n, dtype=int)
            for _ in range(max_iter):
                d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
                new_lab = d.argmin(axis=1)
                if np.array_equal(new_lab, labels):
                    break
                labels = new_lab
                for j in range(k):
                    mask = labels == j
                    if mask.any():
                        cents[j] = X[mask].mean(axis=0)
            inertia = ((X - cents[labels]) ** 2).sum()
            if inertia < best_inertia:
                best_inertia, best_labels, best_cents = inertia, labels.copy(), cents.copy()
        return best_labels, best_cents, best_inertia

    def mean_silhouette(X, labels):
        # Pairwise Euclidean distances (n small)
        n = X.shape[0]
        # squared then sqrt
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2) + 1e-12)
        sil = np.zeros(n)
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            if same.sum() == 0:
                sil[i] = 0.0
                continue
            a = D[i, same].mean()
            b = np.inf
            for cl in np.unique(labels):
                if cl == labels[i]:
                    continue
                other = labels == cl
                if other.any():
                    b = min(b, D[i, other].mean())
            sil[i] = (b - a) / max(a, b)
        return sil.mean()

    ks = np.arange(2, 8)
    sils, wss = [], []
    for k in ks:
        lab, _, inertia = lloyd(X, int(k))
        sils.append(mean_silhouette(X, lab))
        wss.append(inertia)
    sils = np.asarray(sils)
    wss = np.asarray(wss)
    k_best = int(ks[np.argmax(sils)])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # Left: data with k=3 partition
    lab3, cents3, _ = lloyd(X, 3)
    ax = axes[0]
    colors = [TEAL, GOLD, DEEP, "#b45309", "#64748b", "#a855f7"]
    for j in range(3):
        m = lab3 == j
        ax.scatter(X[m, 0], X[m, 1], s=18, c=colors[j], alpha=0.8, edgecolors="none")
        ax.scatter(cents3[j, 0], cents3[j, 1], s=140, c="white", edgecolors=colors[j],
                   linewidths=2.2, zorder=5, marker="D")
    ax.set_xlabel("feature 1 (standardized)")
    ax.set_ylabel("feature 2 (standardized)")
    style_ax(ax, "Synthetic 3-blob data (k-means, k=3)")

    # Right: silhouette vs k (+ light WSS twin for context)
    ax = axes[1]
    ax.plot(ks, sils, "o-", color=TEAL, lw=2.4, markersize=8, label="mean silhouette")
    ax.axvline(k_best, color=GOLD, ls="--", lw=1.5, label=f"peak at k={k_best}")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("mean silhouette")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style_ax(ax, "Silhouette peaks where geometry separates")
    # inset note
    ax.text(
        0.98, 0.92,
        "High silhouette ≠ clinical phenotype\n(geometry ≠ etiology)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.tight_layout()
    save(fig, "ml_fig_silhouette_k.png")


def fig_token_neighbors():
    """Ch12 scientific: subword tokenization + 2-D embedding nearest neighbors (teaching)."""
    fig = plt.figure(figsize=(9.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.2], wspace=0.28)

    # Left: subword segmentation of a clinical phrase
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(5, 5.5, "Subword tokenization (BPE / WordPiece sketch)", ha="center",
            fontsize=11, fontweight="bold", color=INK)
    ax.text(5, 4.85, 'phrase:  "thrombolysis laterality aphasia"', ha="center",
            fontsize=10, color="#475569", family="monospace")
    # token chips
    tokens = [
        (0.4, "throm", TEAL),
        (2.3, "##bolysis", DEEP),
        (4.6, "lateral", GOLD),
        (6.7, "##ity", "#b45309"),
        (8.3, "aphasia", TEAL),
    ]
    for x, tok, c in tokens:
        w = max(1.5, 0.22 * len(tok) + 0.9)
        ax.add_patch(FancyBboxPatch((x, 3.2), w, 1.0, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + w / 2, 3.7, tok, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", family="monospace")
    ax.annotate("", xy=(9.5, 2.7), xytext=(0.5, 2.7),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.3))
    ax.text(5, 2.35, "rare clinical strings split; common stems stay whole", ha="center",
            fontsize=9, color="#64748b")
    # UNK contrast
    ax.add_patch(FancyBboxPatch((1.2, 0.55), 7.6, 1.4, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2))
    ax.text(5, 1.55, "Word-level only → many [UNK] on local notes", ha="center",
            fontsize=9, color="#b91c1c")
    ax.text(5, 0.95, "Subword → lower OOV; still audit drug / laterality tokens", ha="center",
            fontsize=9, color=DEEP)

    # Right: 2-D nearest-neighbor embedding sketch (hand-placed for pedagogy)
    ax = fig.add_subplot(gs[0, 1])
    words = {
        "stroke": (0.2, 1.1),
        "CVA": (0.55, 0.85),
        "TIA": (0.9, 1.35),
        "NIHSS": (2.6, 2.4),
        "mRS": (3.0, 2.1),
        "thrombolysis": (1.8, 0.2),
        "tPA": (2.2, -0.15),
        "tenecteplase": (2.55, 0.35),
        "aphasia": (-1.2, 2.0),
        "dysarthria": (-0.85, 1.55),
        "hemorrhage": (-1.5, -0.8),
        "ICH": (-1.1, -1.15),
    }
    # neighborhoods
    groups = [
        (["stroke", "CVA", "TIA"], TEAL, "vascular events"),
        (["NIHSS", "mRS"], GOLD, "scales"),
        (["thrombolysis", "tPA", "tenecteplase"], DEEP, "reperfusion Rx"),
        (["aphasia", "dysarthria"], "#b45309", "language signs"),
        (["hemorrhage", "ICH"], "#64748b", "bleed terms"),
    ]
    for names, c, _ in groups:
        xs = [words[n][0] for n in names]
        ys = [words[n][1] for n in names]
        # soft hull as scatter cloud
        ax.scatter(xs, ys, s=220, c=c, alpha=0.18, edgecolors="none")
        for n in names:
            x, y = words[n]
            ax.scatter([x], [y], s=55, c=c, edgecolors="white", linewidths=0.8, zorder=3)
            ax.text(x + 0.12, y + 0.08, n, fontsize=8, color=INK, fontweight="bold")
    # query arrow: stroke → nearest neighbors
    ax.annotate(
        "", xy=words["CVA"], xytext=words["stroke"],
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
    )
    ax.annotate(
        "", xy=words["TIA"], xytext=words["stroke"],
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2, alpha=0.7),
    )
    ax.set_xlabel("embedding dim 1 (synthetic PCA sketch)")
    ax.set_ylabel("embedding dim 2")
    ax.set_xlim(-2.2, 3.7)
    ax.set_ylim(-1.7, 3.0)
    style_ax(ax, "Nearest neighbors in embedding space")
    ax.text(
        0.98, 0.04,
        "Near ≠ synonymous; geometry is corpus-dependent",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    save(fig, "ml_fig_token_neighbors.png")


def fig_drift_monitor():
    """Ch16 senior practice: deployment monitoring — score drift + PSI schematic."""
    rng = np.random.default_rng(5)
    # Reference (train/serve baseline) vs current window scores
    ref = rng.beta(2.5, 4.0, size=2000)
    # Drifted: mass shifts higher (case-mix / scanner / coding change)
    cur = rng.beta(3.8, 2.8, size=2000)

    bins = np.linspace(0, 1, 11)
    r_hist, _ = np.histogram(ref, bins=bins)
    c_hist, _ = np.histogram(cur, bins=bins)
    r_p = r_hist / r_hist.sum()
    c_p = c_hist / c_hist.sum()
    # PSI with floor to avoid log(0)
    eps = 1e-6
    psi_bins = (c_p + eps - (r_p + eps)) * np.log((c_p + eps) / (r_p + eps))
    psi = psi_bins.sum()
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Monthly AUROC synthetic series with a late drop
    months = np.arange(1, 13)
    auroc = np.array([0.84, 0.83, 0.845, 0.84, 0.835, 0.83, 0.828, 0.82, 0.81, 0.79, 0.77, 0.76])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: score distribution shift + PSI bars
    ax = axes[0]
    width = 0.035
    ax.bar(centers - width, r_p, width=width * 1.8, color=TEAL, alpha=0.85, label="reference (fit window)")
    ax.bar(centers + width, c_p, width=width * 1.8, color=GOLD, alpha=0.9, label="current deployment")
    ax.set_xlabel("predicted score bin")
    ax.set_ylabel("proportion")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, f"Score shift  ·  PSI ≈ {psi:.2f}")
    alarm = "ALARM (≈0.2+)" if psi >= 0.2 else "watch"
    ax.text(
        0.98, 0.55,
        f"PSI = Σ (c−r) ln(c/r)\n≈ {psi:.2f}  →  {alarm}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        color="#b91c1c" if psi >= 0.2 else DEEP,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ed", edgecolor=GOLD, alpha=0.95),
    )

    # Right: live AUROC trend with governance actions
    ax = axes[1]
    ax.plot(months, auroc, "o-", color=TEAL, lw=2.3, markersize=7)
    ax.axhline(0.80, color=GOLD, ls="--", lw=1.5, label="pre-set floor 0.80")
    ax.fill_between(months, 0.70, auroc, where=auroc < 0.80, color="#fecaca", alpha=0.55, interpolate=True)
    ax.annotate(
        "investigate /\nrollback plan",
        xy=(10, 0.79), xytext=(6.2, 0.74),
        fontsize=9, color="#b91c1c",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.3),
    )
    ax.set_xlabel("month after go-live")
    ax.set_ylabel("live AUROC (synthetic)")
    ax.set_ylim(0.70, 0.90)
    ax.set_xticks(months)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Outcome metric lag — monitor inputs too")
    ax.text(
        0.98, 0.06,
        "Prediction ≠ causation; drift is ops science",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.tight_layout()
    save(fig, "ml_fig_drift_monitor.png")


def fig_tfidf_cosine():
    """Ch05 scientific: TF–IDF vectors + cosine ranking from chapter toy corpus."""
    # Chapter worked example: N=3 docs, vocab order data, learning, machine, mining, models
    # idf = ln(N/df): data/learning/machine/models ≈0.405, mining ≈1.099
    terms = ["data", "learning", "machine", "mining", "models"]
    d1 = np.array([0.0, 0.405, 0.405, 0.0, 0.405])
    d2 = np.array([0.405, 0.405, 0.405, 0.0, 0.0])
    d3 = np.array([0.405, 0.0, 0.0, 1.099, 0.405])
    q = np.array([0.0, 0.405, 0.405, 0.0, 0.0])  # "machine learning"

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb + 1e-12))

    scores = [cos(q, d1), cos(q, d2), cos(q, d3)]
    docs = [d1, d2, d3]
    labels = ["d1: ML models", "d2: ML data", "d3: data mining models"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Left: grouped bar of TF–IDF weights + query overlay
    ax = axes[0]
    x = np.arange(len(terms))
    w = 0.18
    colors = [TEAL, DEEP, GOLD, "#64748b"]
    for i, (vec, lab, c) in enumerate(zip(docs + [q], labels + ["query q"], colors)):
        ax.bar(x + (i - 1.5) * w, vec, width=w, color=c, alpha=0.9, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=15, ha="right")
    ax.set_ylabel("TF–IDF weight")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style_ax(ax, "Chapter toy corpus · TF–IDF coordinates")

    # Right: cosine bars for ranking
    ax = axes[1]
    order = np.argsort(scores)[::-1]
    y = np.arange(3)
    ranked_scores = [scores[i] for i in order]
    ranked_labs = [labels[i] for i in order]
    cols = [TEAL if i < 2 else GOLD for i in range(3)]
    ax.barh(y, ranked_scores, color=cols, alpha=0.9, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(ranked_labs, fontsize=9)
    ax.set_xlabel("cos(q, d) = (q·d) / (‖q‖‖d‖)")
    ax.set_xlim(0, 1.05)
    for i, s in enumerate(ranked_scores):
        ax.text(s + 0.02, i, f"{s:.3f}", va="center", fontsize=10, fontweight="bold", color=INK)
    style_ax(ax, "Ranked retrieval for q = “machine learning”")
    ax.text(
        0.98, 0.08,
        "Rare term “mining” boosts d3 on other queries;\nnot this one. Cosine ≠ clinical relevance.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.tight_layout()
    save(fig, "ml_fig_tfidf_cosine.png")


def fig_infonce_temperature():
    """Ch11 scientific: InfoNCE / softmax weights vs temperature τ."""
    # Synthetic similarity logits: one positive, many negatives
    rng = np.random.default_rng(7)
    pos = 2.4
    neg = rng.normal(-0.3, 0.9, size=7)
    logits = np.concatenate([[pos], neg])  # index 0 = positive pair
    labels = ["pos"] + [f"n{i}" for i in range(1, 8)]
    taus = [0.07, 0.20, 0.50, 1.50]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: softmax mass on positive vs τ
    ax = axes[0]
    tau_grid = np.linspace(0.05, 2.0, 120)
    p_pos = []
    for t in tau_grid:
        z = logits / t
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        p_pos.append(p[0])
    p_pos = np.asarray(p_pos)
    ax.plot(tau_grid, p_pos, color=TEAL, lw=2.5)
    for t in taus:
        z = logits / t
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        ax.scatter([t], [p[0]], s=55, zorder=5, color=GOLD if t < 0.3 else DEEP)
        ax.annotate(f"τ={t:g}", (t, p[0]), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=INK)
    ax.set_xlabel("temperature τ")
    ax.set_ylabel("softmax weight on positive pair")
    ax.set_ylim(0, 1.05)
    style_ax(ax, "InfoNCE: low τ sharpens the positive")
    ax.text(
        0.98, 0.12,
        "L = −log  p_pos   (single positive)",
        transform=ax.transAxes, ha="right", fontsize=9, color="#475569",
        family="monospace",
    )

    # Right: bar of class probabilities at two temperatures
    ax = axes[1]
    x = np.arange(len(logits))
    w = 0.38
    for j, (t, c) in enumerate([(0.07, TEAL), (1.50, GOLD)]):
        z = logits / t
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        ax.bar(x + (j - 0.5) * w, p, width=w, color=c, alpha=0.9, label=f"τ={t:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("softmax probability")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Mass concentrates as τ → 0")
    ax.text(
        0.98, 0.90,
        "Hard negatives dominate gradients at low τ;\nhigh τ softens and can under-use hard pairs",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Contrastive InfoNCE temperature (synthetic similarities)", color=INK,
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_infonce_temp.png")


def fig_quant_bits_tradeoff():
    """Ch14 scientific: bit-width vs model size and synthetic accuracy (tiny ICH CNN stack)."""
    # Teaching curve inspired by chapter tiny CNN compression stack (not a real model dump)
    bits = np.array([32, 16, 8, 4, 2])
    # Relative size vs FP32 (idealized; int8 ≈ 1/4, etc.)
    size_rel = bits / 32.0
    # Synthetic held-out accuracy (percentage points) — mild drop then cliff at 2-bit
    acc = np.array([0.912, 0.910, 0.901, 0.868, 0.742])
    # Latency proxy (relative; lower bits often faster on supported HW)
    lat = np.array([1.00, 0.72, 0.48, 0.41, 0.38])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: accuracy vs relative size (log x)
    ax = axes[0]
    ax.plot(size_rel, acc, "o-", color=TEAL, lw=2.4, markersize=9, zorder=3)
    for b, s, a in zip(bits, size_rel, acc):
        ax.annotate(f"{b}-bit", (s, a), textcoords="offset points",
                    xytext=(6, 6 if b != 2 else -14), fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0.88, color=GOLD, ls="--", lw=1.4, label="clinical floor (teaching)")
    ax.set_xscale("log")
    ax.set_xlabel("relative weight storage (vs FP32)")
    ax.set_ylabel("held-out accuracy (synthetic)")
    ax.set_ylim(0.70, 0.95)
    ax.set_xlim(0.04, 1.3)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_ax(ax, "Accuracy–size trade-off under quantization")
    ax.text(
        0.02, 0.06,
        "PTQ vs QAT not shown — measure both;\nsubgroup AUROC can fall before global acc",
        transform=ax.transAxes, fontsize=8, color="#64748b",
    )

    # Right: size + latency dual view by bit-width
    ax = axes[1]
    x = np.arange(len(bits))
    bars = ax.bar(x - 0.18, size_rel, width=0.36, color=TEAL, alpha=0.9, label="size / FP32")
    ax2 = ax.twinx()
    ax2.plot(x, lat, "D-", color=GOLD, lw=2.2, markersize=8, label="latency proxy")
    ax2.plot(x, acc, "s--", color=DEEP, lw=1.8, markersize=7, label="accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}" for b in bits])
    ax.set_xlabel("weight bit-width")
    ax.set_ylabel("relative size", color=TEAL)
    ax2.set_ylabel("latency proxy / accuracy", color=INK)
    ax2.set_ylim(0.3, 1.05)
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    style_ax(ax, "Bits, bytes, and wall-clock (device-dependent)")
    ax.text(
        0.98, 0.08,
        "FLOPs ≠ latency; benchmark the edge device",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Making lighter models: quantization is a clinical constraint curve", color=INK,
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_quant_bits.png")


def fig_pagerank_iteration():
    """Ch15 scientific: PageRank power iteration on chapter 4-node digraph."""
    # Edges: A→B, A→C, B→C, C→A, D→C. Order A,B,C,D
    # Row-stochastic P:
    # A: 0, 1/2, 1/2, 0
    # B: 0, 0, 1, 0
    # C: 1, 0, 0, 0
    # D: 0, 0, 1, 0
    P = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    alpha = 0.85
    v = np.ones(4) / 4.0
    r = v.copy()
    hist = [r.copy()]
    for _ in range(25):
        r = alpha * (P.T @ r) + (1 - alpha) * v
        hist.append(r.copy())
    hist = np.asarray(hist)
    names = ["A", "B", "C", "D"]
    colors = [TEAL, GOLD, DEEP, "#64748b"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: graph schematic with final PR as node size
    ax = axes[0]
    pos = {"A": (0.2, 0.75), "B": (0.75, 0.85), "C": (0.55, 0.35), "D": (0.15, 0.2)}
    edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A"), ("D", "C")]
    for u, w in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[w]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.6,
                            connectionstyle="arc3,rad=0.08",
                            shrinkA=14, shrinkB=14),
        )
    r_final = hist[-1]
    for i, n in enumerate(names):
        x, y = pos[n]
        s = 900 + 4200 * r_final[i]
        ax.scatter([x], [y], s=s, c=colors[i], zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(x, y, n, ha="center", va="center", color="white", fontsize=12, fontweight="bold", zorder=4)
        ax.text(x, y - 0.12, f"π≈{r_final[i]:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.axis("off")
    ax.set_title("Directed graph · node size ∝ PageRank", fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.text(0.5, 0.02, "α=0.85 · uniform teleport · D only via teleport", ha="center",
            fontsize=8, color="#64748b", transform=ax.transAxes)

    # Right: convergence trajectories
    ax = axes[1]
    for i, n in enumerate(names):
        ax.plot(np.arange(hist.shape[0]), hist[:, i], "o-", color=colors[i],
                lw=2.0, markersize=3.5, label=n)
    # Mark chapter iteration 1 values approximately
    ax.axvline(1, color="#cbd5e1", ls=":", lw=1.2)
    ax.text(1.15, 0.62, "iter 1", fontsize=8, color="#64748b")
    ax.set_xlabel("power iteration k")
    ax.set_ylabel("r_k (PageRank mass)")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    style_ax(ax, "r ← α Pᵀ r + (1−α) v  converges")
    ax.text(
        0.98, 0.08,
        "High PR ≠ causal importance;\nedges encode link structure only",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("PageRank on the chapter four-node digraph (A→B,A→C,B→C,C→A,D→C)", color=INK,
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_pagerank_iter.png")


def fig_pred_not_cause():
    """Preface / closing / glossary: prediction scores ≠ causal structure (scientific teaching)."""
    rng = np.random.default_rng(42)
    # Confounder U drives both treatment-like exposure X and outcome Y;
    # a pure predictor of Y can use X without X causing Y.
    n = 400
    U = rng.normal(0, 1, size=n)
    X = 0.9 * U + rng.normal(0, 0.55, size=n)
    Y = 1.1 * U + rng.normal(0, 0.6, size=n)  # no direct X→Y
    # Fit naive linear association Y ~ X (observational)
    coef = np.polyfit(X, Y, 1)
    xline = np.linspace(X.min(), X.max(), 100)
    yhat = coef[0] * xline + coef[1]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: scatter with predictive fit (association ≠ causation)
    ax = axes[0]
    sc = ax.scatter(X, Y, c=U, cmap="viridis", s=16, alpha=0.75, edgecolors="none")
    ax.plot(xline, yhat, color=GOLD, lw=2.4, label=f"naive fit  Ŷ = {coef[0]:.2f}X + …")
    ax.set_xlabel("feature X (e.g., co-med / order code)")
    ax.set_ylabel("outcome Y")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_ax(ax, "Strong prediction from X — but Y ⊥ X | U")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("confounder U", fontsize=9)
    ax.text(
        0.98, 0.06,
        "Color = U. Conditioning on U removes X→Y association.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )

    # Right: two DAGs — predictive vs causal claim
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # Predictive system box
    ax.add_patch(FancyBboxPatch((0.4, 3.2), 4.2, 2.3, boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=SOFT, edgecolor=TEAL, linewidth=1.6))
    ax.text(2.5, 5.15, "Prediction claim", ha="center", fontsize=11, fontweight="bold", color=DEEP)
    for (x, y, t) in [(1.3, 4.2, "X"), (2.5, 4.2, "model"), (3.7, 4.2, "Ŷ")]:
        ax.add_patch(Circle((x, y), 0.38, facecolor=TEAL, edgecolor="none"))
        ax.text(x, y, t, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(2.05, 4.2), xytext=(1.7, 4.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(3.25, 4.2), xytext=(2.95, 4.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(2.5, 3.5, "OK if calibrated & useful\n(not a mechanism)", ha="center", fontsize=8, color="#475569")

    # Causal system box
    ax.add_patch(FancyBboxPatch((5.4, 3.2), 4.2, 2.3, boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor="#fff7ed", edgecolor=GOLD, linewidth=1.6))
    ax.text(7.5, 5.15, "Causal claim", ha="center", fontsize=11, fontweight="bold", color="#b45309")
    # U at top, X and Y below
    ax.add_patch(Circle((7.5, 4.55), 0.35, facecolor=GOLD, edgecolor="none"))
    ax.text(7.5, 4.55, "U", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.add_patch(Circle((6.4, 3.7), 0.35, facecolor=DEEP, edgecolor="none"))
    ax.text(6.4, 3.7, "X", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.add_patch(Circle((8.6, 3.7), 0.35, facecolor=DEEP, edgecolor="none"))
    ax.text(8.6, 3.7, "Y", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(6.55, 3.95), xytext=(7.25, 4.35), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(8.45, 3.95), xytext=(7.75, 4.35), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(7.5, 3.35, "No X→Y edge in DGP", ha="center", fontsize=8, color="#b45309")

    ax.text(5, 2.6, "Do not swap claims mid–journal club", ha="center", fontsize=12,
            fontweight="bold", color=INK)
    bullets = [
        "High AUROC / low MSE ⇒ ranking or fit quality",
        "Intervention effect needs design / identification",
        "Association rules & PR scores share this trap",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.6, 2.0 - i * 0.55, "•  " + b, fontsize=10, color="#334155", ha="left")
    ax.set_title("Prediction ≠ causation (synthetic confounder sketch)", fontsize=12,
                 fontweight="bold", color=INK, pad=6)
    fig.tight_layout()
    save(fig, "ml_fig_pred_not_cause.png")


def fig_apriori_support_lattice():
    """Ch05 scientific: support on itemset lattice + Apriori pruning (chapter basket)."""
    # n=5 transactions from chapter; supports for subsets of {A,B,C}
    # s(A)=0.8, s(B)=0.8, s(C)=0.8, s(AB)=0.6, s(AC)=0.6, s(BC)=0.6, s(ABC)=0.40
    minsup = 0.50
    nodes = {
        "∅": (0.5, 0.92, 1.0),
        "A": (0.18, 0.68, 0.80),
        "B": (0.50, 0.68, 0.80),
        "C": (0.82, 0.68, 0.80),
        "AB": (0.18, 0.40, 0.60),
        "AC": (0.50, 0.40, 0.60),
        "BC": (0.82, 0.40, 0.60),
        "ABC": (0.50, 0.12, 0.40),
    }
    edges = [
        ("∅", "A"), ("∅", "B"), ("∅", "C"),
        ("A", "AB"), ("A", "AC"), ("B", "AB"), ("B", "BC"), ("C", "AC"), ("C", "BC"),
        ("AB", "ABC"), ("AC", "ABC"), ("BC", "ABC"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for u, v in edges:
        x0, y0, s0 = nodes[u]
        x1, y1, s1 = nodes[v]
        # dashed if child would be pruned under minsup when parent rare (here all parents frequent)
        prune = s1 < minsup
        ax.plot([x0, x1], [y0, y1], color="#cbd5e1" if not prune else "#fca5a5",
                lw=1.4, ls="-" if not prune else "--", zorder=1)
    for name, (x, y, s) in nodes.items():
        frequent = s >= minsup
        fac = TEAL if frequent else "#fecaca"
        ec = DEEP if frequent else "#dc2626"
        ax.scatter([x], [y], s=1400 if name != "∅" else 900, c=fac, edgecolors=ec,
                   linewidths=1.8, zorder=2)
        ax.text(x, y + 0.01, name, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white" if frequent or name == "∅" else "#7f1d1d", zorder=3)
        if name != "∅":
            ax.text(x, y - 0.055, f"s={s:.2f}", ha="center", fontsize=8,
                    color="#134e4a" if frequent else "#991b1b", zorder=3)
    ax.text(0.5, 0.98, f"Itemset lattice · minsup = {minsup:.2f}", ha="center",
            fontsize=12, fontweight="bold", color=INK)
    ax.text(0.5, 0.01, "Apriori: if X is infrequent, all supersets of X are infrequent",
            ha="center", fontsize=8, color="#64748b")

    # Right: bar of supports with threshold
    ax = axes[1]
    order = ["A", "B", "C", "AB", "AC", "BC", "ABC"]
    sups = [nodes[k][2] for k in order]
    cols = [TEAL if s >= minsup else "#f87171" for s in sups]
    ax.barh(np.arange(len(order)), sups, color=cols, height=0.65)
    ax.axvline(minsup, color=GOLD, ls="--", lw=2, label=f"minsup={minsup}")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("support s(X) = count(X) / n   (n=5)")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_ax(ax, "Chapter basket supports (A,B,C focus)")
    ax.text(
        0.98, 0.92,
        "Frequent ≠ causal co-prescription\nLift still required for rules",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.tight_layout()
    save(fig, "ml_fig_apriori_lattice.png")


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
    # Swarm 3h cycle-1 densify (preface / glossary / ch12 / ch13)
    fig_claim_types()
    fig_accuracy_trap()
    fig_causal_mask()
    fig_dice_iou()
    fig_discount_gamma()
    fig_bandit_explore()
    # Swarm 3h cycle-2 densify (ch07 / ch10 / ch17 / glossary)
    fig_pca_variance_recon()
    fig_vanishing_residual()
    fig_appraisal_checklist()
    fig_reliability_ece()
    fig_capacity_vs_n()
    # Swarm 3h cycle-3 densify (ch06 / ch09 / ch04 / ch12 / ch16)
    fig_preprocess_fit_split()
    fig_threshold_sens_spec()
    fig_silhouette_k()
    fig_token_neighbors()
    fig_drift_monitor()
    # Continuous densify cycle-4 (preface / ch05 / ch11 / ch14 / ch15 / ch17 / glossary)
    fig_tfidf_cosine()
    fig_infonce_temperature()
    fig_quant_bits_tradeoff()
    fig_pagerank_iteration()
    fig_pred_not_cause()
    fig_apriori_support_lattice()
    print("DONE figures in", OUT)
    missing_legacy = [n for n in LEGACY_NUMBERED_ASSETS if not (OUT / n).exists()]
    if missing_legacy:
        print("WARN missing legacy assets:", missing_legacy)
    else:
        print("LEGACY numbered assets present:", len(LEGACY_NUMBERED_ASSETS))


if __name__ == "__main__":
    main()
