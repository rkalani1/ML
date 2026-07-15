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


def fig_vae_elbo_beta():
    """Ch11 scientific: VAE ELBO terms and β-VAE recon vs KL trade-off."""
    # Univariate Gaussian KL(q‖p) with p=N(0,1), q=N(μ,σ²): ½[μ²+σ²−1−log σ²]
    # Chapter sketch: μ=1, σ=0.5 → KL≈0.818; recon MSE=0.50 → −ELBO ≈ 1.318 at β=1
    mu, sig = 1.0, 0.5
    kl_unit = 0.5 * (mu**2 + sig**2 - 1.0 - np.log(sig**2))
    recon = 0.50  # E[−log p(x|z)] proxy (squared error)
    betas = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    # Synthetic teaching surfaces: higher β pulls encoder toward prior → KL↓, recon↑
    # Soft schedule (not a real training run): KL shrinks, recon rises with β
    kl_b = kl_unit / (1.0 + 0.55 * (betas - 1.0).clip(min=0)) * (1.0 + 0.15 * (1.0 - betas).clip(min=0))
    # Recenter so β=1 matches chapter numbers
    kl_b = kl_b * (kl_unit / kl_b[np.argmin(np.abs(betas - 1.0))])
    recon_b = recon * (1.0 + 0.22 * np.log1p(betas))
    recon_b = recon_b * (recon / recon_b[np.argmin(np.abs(betas - 1.0))])
    neg_elbo = recon_b + betas * kl_b  # training objective L = recon + β·KL

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: ELBO decomposition at chapter point + β scaling of objective
    ax = axes[0]
    x = np.arange(3)
    parts = [recon, kl_unit, recon + kl_unit]
    labs = ["recon\nE[−log p(x|z)]", "KL(q‖p)", "−ELBO\n(β=1)"]
    cols = [TEAL, GOLD, DEEP]
    ax.bar(x, parts, color=cols, width=0.62, alpha=0.92)
    for i, v in enumerate(parts):
        ax.text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("nats (teaching units)")
    ax.set_ylim(0, 1.55)
    style_ax(ax, "Chapter sketch: μ=1, σ=0.5, recon=0.50")
    ax.text(
        0.98, 0.92,
        "KL = ½[μ²+σ²−1−log σ²] ≈ 0.818\nELBO = −recon − KL  (higher better)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#475569",
        family="monospace",
    )

    # Right: β-VAE trade-off curves
    ax = axes[1]
    ax.plot(betas, recon_b, "o-", color=TEAL, lw=2.3, markersize=8, label="reconstruction term")
    ax.plot(betas, kl_b, "s-", color=GOLD, lw=2.3, markersize=7, label="KL(q‖p)")
    ax.plot(betas, neg_elbo, "D--", color=DEEP, lw=2.0, markersize=6, label="L = recon + β·KL")
    ax.axvline(1.0, color="#cbd5e1", ls=":", lw=1.3)
    ax.text(1.05, max(neg_elbo) * 0.92, "β=1 (ELBO)", fontsize=8, color="#64748b")
    ax.set_xlabel("β  (weight on KL)")
    ax.set_ylabel("loss contribution")
    ax.set_xscale("log", base=2)
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax, "β-VAE: regularity vs fidelity (synthetic schedule)")
    ax.text(
        0.98, 0.08,
        "Large β → latents near prior; recon often suffers\nDisentanglement is hypothesized, not guaranteed",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("VAE evidence lower bound and β trade-off (univariate Gaussian teaching)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_vae_elbo.png")


def fig_distill_temperature():
    """Ch14 scientific: distillation soft-target entropy vs temperature T."""
    # 3-class stroke-ish logits from a confident teacher
    z = np.array([2.8, 1.1, -0.4])  # classes: LVO, small-vessel, mimic (teaching labels)
    class_names = ["LVO", "SV", "mimic"]
    Ts = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    def softmax_T(logits, T):
        x = logits / T
        x = x - x.max()
        p = np.exp(x)
        return p / p.sum()

    entropies = []
    for T in Ts:
        p = softmax_T(z, T)
        entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
    entropies = np.asarray(entropies)
    # Softmax at T=1 and T=4 for side panel
    p1 = softmax_T(z, 1.0)
    p4 = softmax_T(z, 4.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: soft targets at two temperatures
    ax = axes[0]
    x = np.arange(len(z))
    w = 0.36
    ax.bar(x - w / 2, p1, width=w, color=TEAL, alpha=0.92, label="T=1 (harder soft)")
    ax.bar(x + w / 2, p4, width=w, color=GOLD, alpha=0.92, label="T=4 (darker knowledge)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel(r"teacher $p_i = \mathrm{softmax}(z_i/T)$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Temperature flattens teacher probabilities")
    ax.text(
        0.98, 0.90,
        "logits z = [2.8, 1.1, −0.4]\nL = α T² CE(p_s, p_t) + (1−α) CE(p_s, y)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#475569",
        family="monospace",
    )

    # Right: entropy of soft targets vs T + peak class mass
    ax = axes[1]
    peak = [float(softmax_T(z, T).max()) for T in Ts]
    ax.plot(Ts, entropies, "o-", color=TEAL, lw=2.4, markersize=8, label="H(p_T) entropy")
    ax2 = ax.twinx()
    ax2.plot(Ts, peak, "s--", color=GOLD, lw=2.0, markersize=7, label="max p_i")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("entropy H(p) [nats]", color=TEAL)
    ax2.set_ylabel("peak class probability", color="#b45309")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([f"{int(t) if t>=1 else t}" for t in Ts])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    style_ax(ax, "Soft-target entropy rises with T")
    ax.text(
        0.02, 0.08,
        "T² in the KD loss keeps soft gradients\ncomparable as T grows; tune T on val set",
        transform=ax.transAxes, fontsize=8, color="#64748b",
    )
    fig.suptitle("Knowledge distillation: temperature-scaled soft targets (synthetic 3-class teacher)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_distill_temp.png")


def fig_lora_rank():
    """Ch14 scientific: LoRA rank r vs trainable params and capacity sketch."""
    # Frozen W0 is d×k; trainable BA with B d×r, A r×k → params = r(d+k)
    d, k = 4096, 4096  # square attention-weight scale (teaching)
    full = d * k
    ranks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    lora_params = ranks * (d + k)
    frac = lora_params / full
    # Synthetic val loss: diminishing returns past moderate r (underfit → plateau)
    # Not a real run — teaching curve only
    loss = 0.42 + 0.28 * np.exp(-ranks / 10.0) + 0.01 * np.log1p(ranks / 8.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: parameter count vs rank
    ax = axes[0]
    ax.plot(ranks, lora_params / 1e6, "o-", color=TEAL, lw=2.4, markersize=8, label="LoRA (A,B)")
    ax.axhline(full / 1e6, color=GOLD, ls="--", lw=1.8, label=f"full d×k = {full/1e6:.0f}M")
    for r, p in zip(ranks[::2], (lora_params / 1e6)[::2]):
        ax.annotate(f"r={r}", (r, p), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("trainable parameters (millions)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_ax(ax, f"W = W₀ + BA · d=k={d} · params = r(d+k)")
    ax.text(
        0.98, 0.12,
        f"r=8 → {lora_params[ranks.tolist().index(8)]/1e6:.2f}M "
        f"({100*frac[ranks.tolist().index(8)]:.2f}% of full)\n"
        "Merge BA into W₀ at deploy → no extra latency",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )

    # Right: capacity vs rank (synthetic val loss) + % of full
    ax = axes[1]
    ax.plot(ranks, loss, "o-", color=DEEP, lw=2.4, markersize=8, label="val loss (synthetic)")
    ax2 = ax.twinx()
    ax2.plot(ranks, 100 * frac, "s--", color=GOLD, lw=2.0, markersize=6, label="% of full W")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("held-out loss (synthetic)", color=DEEP)
    ax2.set_ylabel("% of full fine-tune params", color="#b45309")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Rank is a capacity knob — validate rare phenotypes")
    ax.text(
        0.02, 0.08,
        "Low r underfits local note style / rare labels;\nhigh r → more params, still ≪ full FT",
        transform=ax.transAxes, fontsize=8, color="#64748b",
    )
    fig.suptitle("Low-Rank Adaptation: rank–parameter–capacity map (teaching, d=k=4096)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_lora_rank.png")


def fig_minhash_jaccard():
    """Ch05 scientific: MinHash estimate of Jaccard vs exact, vs signature length k."""
    rng = np.random.default_rng(11)
    # Two sets over universe {0..U-1}
    U = 200
    A = set(rng.choice(U, size=80, replace=False).tolist())
    # B overlaps ~40% Jaccard target: build from A and new elements
    n_inter = 35
    inter = set(rng.choice(list(A), size=n_inter, replace=False).tolist())
    rest_B = set(rng.choice([x for x in range(U) if x not in A], size=45, replace=False).tolist())
    B = inter | rest_B
    j_exact = len(A & B) / len(A | B)

    # MinHash with k independent hash functions: h_i(x) = (a_i * x + b_i) mod P
    P = 104729  # prime
    def minhash_sig(S, k, seed=0):
        rr = np.random.default_rng(seed)
        a = rr.integers(1, P, size=k)
        b = rr.integers(0, P, size=k)
        sig = np.empty(k, dtype=np.int64)
        for i in range(k):
            sig[i] = min(((a[i] * x + b[i]) % P) for x in S)
        return sig

    ks = np.array([5, 10, 20, 40, 80, 160, 320])
    # Multiple trials per k for error bars
    n_trials = 40
    means, stds = [], []
    for k in ks:
        ests = []
        for t in range(n_trials):
            sA = minhash_sig(A, int(k), seed=1000 + 17 * t + int(k))
            sB = minhash_sig(B, int(k), seed=1000 + 17 * t + int(k))  # same (a,b)
            # Same seed ⇒ same hash family for both sets
            ests.append(float(np.mean(sA == sB)))
        means.append(np.mean(ests))
        stds.append(np.std(ests))
    means, stds = np.asarray(means), np.asarray(stds)

    # One illustrative signature agreement strip at k=20
    k_show = 20
    sA = minhash_sig(A, k_show, seed=42)
    sB = minhash_sig(B, k_show, seed=42)
    agree = (sA == sB).astype(float)
    j_hat_show = agree.mean()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: estimate vs k with ±1 sd band
    ax = axes[0]
    ax.axhline(j_exact, color=GOLD, ls="--", lw=2.0, label=f"exact J(A,B)={j_exact:.3f}")
    ax.plot(ks, means, "o-", color=TEAL, lw=2.3, markersize=8, label="mean MinHash Ĵ")
    ax.fill_between(ks, means - stds, means + stds, color=TEAL, alpha=0.18, label="±1 sd (40 trials)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("signature length k")
    ax.set_ylabel("Jaccard estimate")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_ax(ax, r"P(min π(A)=min π(B)) = J(A,B)")
    ax.text(
        0.02, 0.92,
        f"|A|={len(A)}, |B|={len(B)}, |A∩B|={len(A & B)}\n"
        r"Ĵ = fraction of equal signature coords",
        transform=ax.transAxes, va="top", fontsize=8, color="#475569",
    )

    # Right: one signature match mask (rectangle strip)
    ax = axes[1]
    colors_row = [TEAL if a else "#e2e8f0" for a in agree]
    for i, c in enumerate(colors_row):
        ax.add_patch(Rectangle((i, 0.25), 0.92, 0.5, facecolor=c, edgecolor="white", linewidth=0.6))
    ax.set_xlim(0, k_show)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("hash function index i = 1…k")
    style_ax(ax, f"One trial k={k_show}: Ĵ={j_hat_show:.2f} vs exact {j_exact:.3f}")
    ax.text(
        0.5, 0.08,
        "Teal = matching mins (collision) · gray = mismatch · LSH buckets similar signatures",
        transform=ax.transAxes, ha="center", fontsize=8, color="#64748b",
    )
    ax.text(
        0.5, 0.88,
        "Near-duplicate notes: high Ĵ ≠ correct content\n(negation / omitted findings still matter)",
        transform=ax.transAxes, ha="center", fontsize=8, color="#64748b",
    )
    fig.suptitle("MinHash estimates Jaccard similarity (synthetic sets; original teaching)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_minhash_jaccard.png")


def fig_betweenness_bridge():
    """Ch15 scientific: betweenness on a two-cluster graph joined by a bridge."""
    # Graph: cluster L = {0,1,2,3} clique-ish; R = {5,6,7,8}; bridge 3—4—5 with 4 as pure bridge
    # Nodes: 0,1,2 dense left; 3 left-hub; 4 bridge; 5 right-hub; 6,7,8 dense right
    edges = [
        (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3),  # left
        (3, 4), (4, 5),  # bridge path
        (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8),  # right
    ]
    n = 9
    # Build adjacency
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Brandes-lite: all-pairs BFS for unweighted betweenness
    bet = np.zeros(n)
    for s in range(n):
        # BFS
        stack = []
        pred = {i: [] for i in range(n)}
        sigma = np.zeros(n)
        dist = np.full(n, -1)
        sigma[s] = 1.0
        dist[s] = 0
        from collections import deque
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bet[w] += delta[w]
    bet /= 2.0  # undirected

    # Layout positions
    pos = {
        0: (0.15, 0.75), 1: (0.08, 0.45), 2: (0.22, 0.35), 3: (0.32, 0.55),
        4: (0.50, 0.55),
        5: (0.68, 0.55), 6: (0.78, 0.75), 7: (0.92, 0.55), 8: (0.82, 0.35),
    }
    names = [str(i) for i in range(n)]
    bet_n = bet / (bet.max() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    ax = axes[0]
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#94a3b8", lw=1.6, zorder=1)
    for i in range(n):
        x, y = pos[i]
        size = 280 + 2200 * bet_n[i]
        # Bridge node gold; hubs teal; leaves deep
        if i == 4:
            c = GOLD
        elif i in (3, 5):
            c = TEAL
        else:
            c = DEEP
        ax.scatter([x], [y], s=size, c=c, zorder=3, edgecolors="white", linewidths=1.4)
        ax.text(x, y, names[i], ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.95)
    ax.axis("off")
    ax.set_title("Two clusters · single bridge path 3—4—5", fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.text(0.15, 0.22, "left community", ha="center", fontsize=9, color="#64748b", transform=ax.transAxes)
    ax.text(0.85, 0.22, "right community", ha="center", fontsize=9, color="#64748b", transform=ax.transAxes)
    ax.text(0.5, 0.05, "Node size ∝ betweenness · gold = pure bridge", ha="center",
            fontsize=8, color="#64748b", transform=ax.transAxes)

    ax = axes[1]
    order = np.argsort(bet)[::-1]
    y = np.arange(n)
    cols = [GOLD if i == 4 else (TEAL if i in (3, 5) else DEEP) for i in order]
    ax.barh(y, bet[order], color=cols, height=0.65, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([f"node {i}" for i in order])
    ax.set_xlabel("betweenness (Brandes, undirected)")
    style_ax(ax, "Bridge node dominates pair-flow scores")
    ax.text(
        0.98, 0.08,
        "High betweenness ≠ clinical quality;\nedge definition drives rankings",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Betweenness centrality on a bridged referral-style graph (scientific; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_betweenness.png")


def fig_spectral_fiedler():
    """Ch15 scientific: spectral clustering / Fiedler vector on bridged graph."""
    # Same 9-node bridge graph as betweenness for continuity
    edges = [
        (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3),
        (3, 4), (4, 5),
        (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8),
    ]
    n = 9
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0
    deg = A.sum(axis=1)
    D = np.diag(deg)
    L = D - A  # combinatorial Laplacian
    # Eigen-decomposition (symmetric)
    evals, evecs = np.linalg.eigh(L)
    # Smallest eigenvalue ~0 (connected); Fiedler = second smallest
    fiedler = evecs[:, 1]
    # Sign partition
    part = fiedler >= 0

    pos = {
        0: (0.15, 0.75), 1: (0.08, 0.45), 2: (0.22, 0.35), 3: (0.32, 0.55),
        4: (0.50, 0.55),
        5: (0.68, 0.55), 6: (0.78, 0.75), 7: (0.92, 0.55), 8: (0.82, 0.35),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Left: nodes colored by Fiedler sign; value as text
    ax = axes[0]
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#94a3b8", lw=1.5, zorder=1)
    for i in range(n):
        x, y = pos[i]
        c = TEAL if part[i] else GOLD
        ax.scatter([x], [y], s=520, c=c, zorder=3, edgecolors="white", linewidths=1.4)
        ax.text(x, y, str(i), ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=4)
        ax.text(x, y - 0.08, f"{fiedler[i]:+.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.18, 0.95)
    ax.axis("off")
    ax.set_title("Fiedler vector u₂ (sign ≈ bipartition)", fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.text(0.5, 0.04, "Teal vs gold = sign(u₂) · bridge node sits near cut",
            ha="center", fontsize=8, color="#64748b", transform=ax.transAxes)

    # Right: spectrum + Fiedler embedding 1-D
    ax = axes[1]
    kshow = min(8, n)
    ax.bar(np.arange(kshow), evals[:kshow], color=[TEAL if i > 0 else GOLD for i in range(kshow)],
           width=0.7, alpha=0.9)
    ax.axhline(0, color="#cbd5e1", lw=1)
    # Mark spectral gap
    if n > 2:
        ax.annotate(
            "spectral gap",
            xy=(1.5, (evals[1] + evals[2]) / 2),
            xytext=(3.2, evals[2] * 0.85 + 0.15),
            fontsize=8, color=DEEP,
            arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.2),
        )
    ax.set_xlabel("eigenvalue index (ascending)")
    ax.set_ylabel("λ of L = D − A")
    style_ax(ax, "Laplacian spectrum · λ₁≈0, λ₂ = algebraic connectivity")
    # Inset-like 1-D embedding strip
    y0 = ax.get_ylim()[1]
    # Plot Fiedler coords as secondary visual at bottom via twinx-less scatter in axes coords
    ax2 = ax.inset_axes([0.08, 0.12, 0.84, 0.22])
    order = np.argsort(fiedler)
    ax2.scatter(fiedler[order], np.zeros(n), c=[TEAL if part[i] else GOLD for i in order],
                s=60, zorder=3, edgecolors="white")
    for i in order:
        ax2.text(fiedler[i], 0.02, str(i), ha="center", va="bottom", fontsize=7, color=INK)
    ax2.axvline(0, color="#94a3b8", ls="--", lw=1)
    ax2.set_yticks([])
    ax2.set_xlabel("Fiedler coordinate (1-D spectral embedding)", fontsize=8)
    ax2.set_facecolor("#f8fafc")
    for s in ax2.spines.values():
        s.set_color("#cbd5e1")
    fig.suptitle("Spectral clustering intuition: Fiedler cut on the bridged graph (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_spectral_fiedler.png")


def fig_multimetric_radar():
    """Ch17 scientific: multi-metric radar for one senior case-study model."""
    # Synthetic LVO prediction case study — multiple claim types on one diagram
    metrics = [
        "AUROC",
        "AUPRC",
        "Calib.\nslope",
        "Brier\n(inv)",
        "Net ben.\n@0.15",
        "Subgroup\nAUROC min",
        "ECE\n(inv)",
    ]
    # Scores scaled to [0,1] "goodness" for radar (teaching only)
    # Model A: strong discrimination, weaker calibration / utility
    m_a = np.array([0.92, 0.71, 0.55, 0.62, 0.48, 0.68, 0.58])
    # Model B: slightly lower AUC, better calibration and net benefit
    m_b = np.array([0.86, 0.66, 0.88, 0.84, 0.78, 0.80, 0.86])
    # "Ship bar" minimum acceptable (not a real policy)
    m_floor = np.array([0.80, 0.50, 0.70, 0.60, 0.40, 0.70, 0.65])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles_c = np.concatenate([angles, angles[:1]])

    def close(v):
        return np.concatenate([v, v[:1]])

    fig = plt.figure(figsize=(9.2, 4.2))
    ax = fig.add_subplot(1, 2, 1, polar=True)
    ax.plot(angles_c, close(m_a), "o-", color=TEAL, lw=2.2, markersize=5, label="Model A · high AUC")
    ax.fill(angles_c, close(m_a), color=TEAL, alpha=0.12)
    ax.plot(angles_c, close(m_b), "s-", color=GOLD, lw=2.2, markersize=5, label="Model B · balanced")
    ax.fill(angles_c, close(m_b), color=GOLD, alpha=0.12)
    ax.plot(angles_c, close(m_floor), "--", color=DEEP, lw=1.5, label="teaching floor")
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=7, color="#64748b")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1.12))
    ax.set_title("One case study · multiple metrics\n(goodness scores, synthetic)", fontsize=12,
                 fontweight="bold", color=INK, pad=16)

    # Right: table-like bar of gaps to floor for model A vs B (net benefit focus)
    ax2 = fig.add_subplot(1, 2, 2)
    labels_short = ["AUROC", "AUPRC", "Cal slope", "Brier↑", "NB@0.15", "min AUC", "ECE↑"]
    x = np.arange(len(labels_short))
    w = 0.36
    ax2.bar(x - w / 2, m_a, width=w, color=TEAL, alpha=0.9, label="Model A")
    ax2.bar(x + w / 2, m_b, width=w, color=GOLD, alpha=0.9, label="Model B")
    ax2.plot(x, m_floor, "D--", color=DEEP, lw=1.6, markersize=6, label="floor")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("goodness score (0–1 teaching scale)")
    ax2.set_ylim(0, 1.08)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    style_ax(ax2, "A wins ranking; B clears more clinical gates")
    ax2.text(
        0.98, 0.92,
        "Do not ship on AUROC alone.\nPrediction ≠ causation; utility is threshold-bound.",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Senior appraisal: multi-metric view of one LVO-style case study (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_multimetric_radar.png")


def fig_claim_routing():
    """Preface scientific: map method family → allowed clinical claim type."""
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Source methods (left)
    methods = [
        (1.4, 5.2, "Supervised\nscore / AUC", TEAL),
        (1.4, 3.5, "Causal / IV\nRCT / DAG", DEEP),
        (1.4, 1.8, "Utility /\nnet benefit", GOLD),
        (1.4, 0.4, "Cluster /\nembedding map", "#64748b"),
    ]
    for x, y, lab, c in methods:
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.15, y - 0.45), 2.3, 1.0,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=c, edgecolor="none", alpha=0.95,
            )
        )
        ax.text(x, y, lab, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")

    # Claim boxes (right)
    claims = [
        (9.8, 5.0, "Prediction\nclaim", TEAL),
        (9.8, 3.3, "Etiology /\ncause claim", DEEP),
        (9.8, 1.6, "Decision-\nsupport claim", GOLD),
        (9.8, 0.3, "Hypothesis\nonly (EDA)", "#64748b"),
    ]
    for x, y, lab, c in claims:
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.25, y - 0.45), 2.5, 1.0,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=c, edgecolor="none", alpha=0.95,
            )
        )
        ax.text(x, y, lab, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")

    # Allowed edges (solid) and blocked (dashed red)
    # supervised → prediction (ok), not etiology
    ax.annotate("", xy=(8.5, 5.0), xytext=(2.6, 5.2),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.0))
    ax.annotate("", xy=(8.5, 3.3), xytext=(2.6, 5.05),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.4,
                                ls="--", connectionstyle="arc3,rad=-0.15"))
    ax.text(5.5, 5.55, "OK if calibrated + external", fontsize=7.5, color=DEEP, ha="center")
    ax.text(5.8, 4.35, "BLOCKED: AUROC ≠ cause", fontsize=7.5, color="#dc2626", ha="center")

    # causal → etiology
    ax.annotate("", xy=(8.5, 3.3), xytext=(2.6, 3.5),
                arrowprops=dict(arrowstyle="->", color=DEEP, lw=2.0))
    # utility → decision
    ax.annotate("", xy=(8.5, 1.6), xytext=(2.6, 1.8),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.0))
    # cluster → hypothesis only
    ax.annotate("", xy=(8.5, 0.3), xytext=(2.6, 0.4),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=2.0))
    # supervised can also feed decision IF threshold utility shown
    ax.annotate("", xy=(8.5, 1.75), xytext=(2.6, 4.9),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2,
                                ls=":", connectionstyle="arc3,rad=0.25"))
    ax.text(4.2, 2.55, "only with net-benefit\n+ threshold policy", fontsize=7, color="#b45309",
            ha="center")

    ax.text(6, 6.0, "Method family → allowed clinical claim (preface routing map)",
            ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(6, -0.15,
            "Prediction ≠ causation. Clusters and UMAP maps do not license etiology or action alone.",
            ha="center", fontsize=8.5, color="#64748b")
    fig.tight_layout()
    save(fig, "ml_fig_claim_routing.png")


def fig_dual_axis_caution():
    """Ch02 scientific: dual y-axis lie factor vs honest small multiples."""
    months = np.arange(1, 13)
    # Synthetic site metrics: admissions (volume) vs in-hospital mortality %
    # True correlation is weak; dual-axis scaling can fake alignment
    rng = np.random.default_rng(7)
    volume = 40 + 3 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 1.2, size=12)
    volume = np.clip(volume, 30, 55)
    mort = 8.5 + 0.15 * np.sin(2 * np.pi * (months + 3) / 12) + rng.normal(0, 0.25, size=12)
    mort = np.clip(mort, 7.0, 10.5)

    # "Lie factor" approximation: ratio of visual slope exaggeration
    # Dual-axis: normalize each series to [0,1] over its own range → slopes look comparable
    v_n = (volume - volume.min()) / (volume.max() - volume.min() + 1e-9)
    m_n = (mort - mort.min()) / (mort.max() - mort.min() + 1e-9)
    # Relative visual change over months 1→7
    vis_v = abs(v_n[6] - v_n[0])
    vis_m = abs(m_n[6] - m_n[0])
    true_v = abs(volume[6] - volume[0]) / volume[0]
    true_m = abs(mort[6] - mort[0]) / mort[0]
    # Tufte-style lie factor proxy: visual relative change / data relative change
    lf_v = vis_v / (true_v + 1e-9)
    lf_m = vis_m / (true_m + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # Left: dual axis (deceptive)
    ax = axes[0]
    ln1 = ax.plot(months, volume, "o-", color=TEAL, lw=2.2, markersize=6, label="admissions")
    ax.set_xlabel("month")
    ax.set_ylabel("admissions / month", color=TEAL)
    ax.tick_params(axis="y", labelcolor=TEAL)
    ax.set_ylim(volume.min() - 1, volume.max() + 1)  # truncated dual ranges
    ax2 = ax.twinx()
    ln2 = ax2.plot(months, mort, "s--", color="#dc2626", lw=2.2, markersize=5, label="mortality %")
    ax2.set_ylabel("mortality %", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_ylim(mort.min() - 0.2, mort.max() + 0.2)
    lns = ln1 + ln2
    ax.legend(lns, [l.get_label() for l in lns], frameon=False, fontsize=8, loc="upper left")
    style_ax(ax, "Dual y-axes (independent scales)")
    ax.text(
        0.98, 0.08,
        f"Visual co-movement is scale choice\nlie-factor proxy vol≈{lf_v:.1f}, mort≈{lf_m:.1f}",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )

    # Right: honest small multiples with free y but clear separate panels
    ax = axes[1]
    ax.plot(months, volume, "o-", color=TEAL, lw=2.2, markersize=6)
    ax.set_ylabel("admissions / month", color=TEAL)
    ax.set_xlabel("month")
    ax.set_ylim(0, 60)  # honest zero baseline for counts
    style_ax(ax, "Honest panel A: volume (y from 0)")
    # inset-like second series as twin but with annotation that scales differ
    # Actually use a second y with shared x but mark "separate claim"
    axb = ax.twinx()
    axb.plot(months, mort, "s--", color=GOLD, lw=2.0, markersize=5, alpha=0.85)
    axb.set_ylabel("mortality % (separate scale)", color=GOLD)
    axb.tick_params(axis="y", labelcolor=GOLD)
    axb.set_ylim(0, 12)
    axb.text(
        0.98, 0.92,
        "Prefer two stacked panels\nin print; twin only if\nscales declared loudly",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    style_ax(ax, "Declared dual scale + zero baselines")

    fig.suptitle("Dual-axis caution: fake alignment vs declared separate scales (synthetic)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_dual_axis_caution.png")


def fig_dbscan_density():
    """Ch04 scientific: DBSCAN core / border / noise with ε-neighborhoods."""
    rng = np.random.default_rng(11)
    # Two dense blobs + sparse noise + a bridge-thin pair
    c1 = rng.normal(loc=[-1.2, 0.0], scale=0.28, size=(28, 2))
    c2 = rng.normal(loc=[1.4, 0.3], scale=0.32, size=(24, 2))
    bridge = np.array([[0.1, 0.05], [0.35, 0.12], [0.55, 0.08]])  # may be border/noise
    noise = rng.uniform(low=[-2.8, -1.8], high=[2.8, 1.8], size=(10, 2))
    # Push some noise far
    noise[0] = [-2.5, 1.5]
    noise[1] = [2.4, -1.4]
    X = np.vstack([c1, c2, bridge, noise])

    eps = 0.55
    min_pts = 5
    # Manual DBSCAN labels for teaching clarity
    from collections import deque

    n = len(X)
    # pairwise distances
    dmat = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    neighbors = [np.where(dmat[i] <= eps + 1e-12)[0].tolist() for i in range(n)]
    n_count = np.array([len(nbr) for nbr in neighbors])
    is_core = n_count >= min_pts
    labels = -np.ones(n, dtype=int)  # -1 noise until assigned
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i] or not is_core[i]:
            continue
        # expand cluster
        cluster_id += 1
        q = deque([i])
        visited[i] = True
        labels[i] = cluster_id
        while q:
            p = q.popleft()
            for nb in neighbors[p]:
                if labels[nb] == -1:
                    labels[nb] = cluster_id  # border or core
                if not visited[nb]:
                    visited[nb] = True
                    if is_core[nb]:
                        q.append(nb)
                    labels[nb] = cluster_id

    # Classify roles for coloring
    roles = []
    for i in range(n):
        if labels[i] == -1:
            roles.append("noise")
        elif is_core[i]:
            roles.append("core")
        else:
            roles.append("border")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))

    # Left: scatter with ε circles on a few cores
    ax = axes[0]
    colmap = {"core": TEAL, "border": GOLD, "noise": "#94a3b8"}
    mark = {"core": "o", "border": "s", "noise": "x"}
    for role in ("core", "border", "noise"):
        idx = [i for i, r in enumerate(roles) if r == role]
        if not idx:
            continue
        ax.scatter(
            X[idx, 0], X[idx, 1],
            c=colmap[role], marker=mark[role], s=48 if role != "noise" else 55,
            linewidths=1.5 if role == "noise" else 0.5,
            edgecolors="white" if role != "noise" else colmap[role],
            label=f"{role} (n={len(idx)})", zorder=3,
        )
    # Draw ε for 3 representative cores
    core_idx = [i for i, r in enumerate(roles) if r == "core"]
    for j in core_idx[:3]:
        circ = Circle(X[j], eps, fill=False, ec=DEEP, ls="--", lw=1.2, alpha=0.7)
        ax.add_patch(circ)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, rf"DBSCAN roles  (ε={eps}, minPts={min_pts})")
    ax.text(
        0.02, 0.02,
        r"core: $|N_\varepsilon|\geq$ minPts; border: in some core ball; noise: else",
        transform=ax.transAxes, fontsize=7.5, color="#64748b", va="bottom",
    )

    # Right: k-distance sketch (sorted 4-NN distance ≈ minPts-1)
    ax = axes[1]
    k = min_pts - 1
    # k-th nearest neighbor distance (exclude self → sort and take [k])
    knn = np.sort(dmat, axis=1)[:, k]  # distance to k-th neighbor
    order = np.argsort(knn)[::-1]  # descending classic k-dist plot
    ax.plot(np.arange(n), knn[order], color=TEAL, lw=2.3)
    ax.axhline(eps, color=GOLD, ls="--", lw=1.8, label=rf"chosen ε={eps}")
    # knee annotation
    ax.annotate(
        "knee → pick ε near\nsteep rise of k-dist",
        xy=(int(0.15 * n), float(knn[order][int(0.15 * n)])),
        xytext=(0.45 * n, float(knn.max()) * 0.75),
        fontsize=8, color=DEEP,
        arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.2),
    )
    ax.set_xlabel("points sorted by k-distance (desc.)")
    ax.set_ylabel(rf"{k}-NN distance")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "k-distance plot guides ε (teaching)")
    ax.text(
        0.98, 0.08,
        "Global ε struggles when density varies\nscale features before distance",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("DBSCAN density: core / border / noise and ε selection (synthetic)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_dbscan_density.png")


def fig_nmf_parts():
    """Ch07 scientific: NMF parts-based reconstruction of a toy nonnegative image."""
    # Build a synthetic 16x16 "image" as sum of 3 nonnegative parts (blobs)
    yy, xx = np.mgrid[0:16, 0:16]
    def blob(cx, cy, sx, sy, amp=1.0):
        return amp * np.exp(-((xx - cx) ** 2) / (2 * sx ** 2) - ((yy - cy) ** 2) / (2 * sy ** 2))

    p1 = blob(4, 4, 2.2, 2.2, 1.0)      # "territory A"
    p2 = blob(11, 5, 2.0, 2.5, 0.9)     # "territory B"
    p3 = blob(8, 12, 3.0, 1.8, 0.85)    # "territory C"
    # Three "patients" / samples with different loadings
    H_true = np.array([
        [1.0, 0.2, 0.1],
        [0.15, 1.0, 0.25],
        [0.2, 0.15, 1.0],
        [0.7, 0.6, 0.3],
    ])  # 4 samples × 3 parts
    parts = np.stack([p1.ravel(), p2.ravel(), p3.ravel()], axis=0)  # 3 × 256
    X = H_true @ parts  # 4 × 256
    X = X + 0.02 * np.random.default_rng(0).random(X.shape)
    X = np.clip(X, 0, None)

    # Simple multiplicative-update NMF (Lee-Seung Frobenius) rank 3
    rng = np.random.default_rng(3)
    r = 3
    W = rng.random((X.shape[0], r)) + 0.1
    H = rng.random((r, X.shape[1])) + 0.1
    for _ in range(400):
        # H <- H * (W.T X) / (W.T W H)
        WH = W @ H
        H *= (W.T @ X) / (W.T @ WH + 1e-12)
        WH = W @ H
        W *= (X @ H.T) / (WH @ H.T + 1e-12)
    recon = W @ H
    rel_err = np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-12)

    fig = plt.figure(figsize=(9.4, 4.2))
    # Top row conceptually via gridspec: original sample 0, 3 parts, reconstruction
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(X[0].reshape(16, 16), cmap="YlGnBu", vmin=0)
    ax0.set_title("X sample 0", fontsize=9, color=INK)
    ax0.axis("off")

    for i in range(3):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(H[i].reshape(16, 16), cmap="YlGnBu", vmin=0)
        ax.set_title(f"part H[{i}]", fontsize=9, color=INK)
        ax.axis("off")

    axr = fig.add_subplot(gs[0, 4])
    axr.imshow(recon[0].reshape(16, 16), cmap="YlGnBu", vmin=0)
    axr.set_title("recon WH[0]", fontsize=9, color=INK)
    axr.axis("off")

    # Bottom: loadings W and residual energy vs rank
    axw = fig.add_subplot(gs[1, :2])
    im = axw.imshow(W, cmap="YlGnBu", aspect="auto")
    axw.set_xticks([0, 1, 2])
    axw.set_xticklabels(["p0", "p1", "p2"])
    axw.set_yticks(range(W.shape[0]))
    axw.set_yticklabels([f"s{i}" for i in range(W.shape[0])])
    axw.set_xlabel("parts")
    axw.set_ylabel("samples")
    style_ax(axw, r"Loadings $W$ (nonnegative)")
    fig.colorbar(im, ax=axw, fraction=0.046, pad=0.04)

    axb = fig.add_subplot(gs[1, 2:])
    ranks = [1, 2, 3, 4]
    errs = []
    for rr in ranks:
        Ww = rng.random((X.shape[0], rr)) + 0.1
        Hh = rng.random((rr, X.shape[1])) + 0.1
        for _ in range(350):
            WH = Ww @ Hh
            Hh *= (Ww.T @ X) / (Ww.T @ WH + 1e-12)
            WH = Ww @ Hh
            Ww *= (X @ Hh.T) / (WH @ Hh.T + 1e-12)
        errs.append(np.linalg.norm(X - Ww @ Hh) / (np.linalg.norm(X) + 1e-12))
    axb.plot(ranks, errs, "o-", color=TEAL, lw=2.4, markersize=8)
    axb.axvline(3, color=GOLD, ls="--", lw=1.5, label="true parts r=3")
    axb.set_xlabel("NMF rank r")
    axb.set_ylabel(r"relative Frobenius error $\|X-WH\|_F/\|X\|_F$")
    axb.set_xticks(ranks)
    axb.legend(frameon=False, fontsize=8)
    style_ax(axb, f"Reconstruction vs rank (err@3≈{rel_err:.3f})")
    axb.text(
        0.98, 0.92,
        "Parts are additive & ≥0\nNot unique; init matters\nInterpret ≠ causal territories",
        transform=axb.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("NMF: nonnegative parts-based factorization (toy 16×16; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    save(fig, "ml_fig_nmf_parts.png")


def fig_bellman_backup():
    """Ch13 scientific: single-state Bellman optimality backup (chapter numbers)."""
    # Chapter worked example: Left R=0 V=5 → 4.5; Right R=1 V=3 → 3.7; γ=0.9
    gamma = 0.9
    actions = [
        ("Left", 0.0, 5.0, "s_L"),
        ("Right", 1.0, 3.0, "s_R"),
        ("Stay", 0.5, 3.5, "s"),  # extra teaching action for 3-way max
    ]
    # Use chapter's two actions primarily; Stay is optional third for bar chart
    actions_ch = actions[:2]
    q_vals = [r + gamma * v for _, r, v, _ in actions_ch]
    q_all = [r + gamma * v for _, r, v, _ in actions]
    v_star = max(q_vals)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))

    # Left: diagram of one-step backup
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # state s
    ax.add_patch(FancyBboxPatch((3.7, 2.2), 2.6, 1.4,
                                boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=TEAL, edgecolor="none"))
    ax.text(5.0, 2.9, "s\n(backup)", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    # successors
    for (x, lab, col, q, r, v) in [
        (1.2, "s_L", DEEP, q_vals[0], 0.0, 5.0),
        (8.0, "s_R", GOLD, q_vals[1], 1.0, 3.0),
    ]:
        ax.add_patch(FancyBboxPatch((x - 1.0, 4.3), 2.0, 1.1,
                                    boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=col, edgecolor="none"))
        ax.text(x, 4.85, f"{lab}\nV={v:g}", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
        ax.annotate("", xy=(x, 4.25), xytext=(5.0 + (x - 5) * 0.15, 3.6),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
        ax.text((x + 5) / 2, 3.85, f"R={r:g}", ha="center", fontsize=8, color="#475569")
        ax.text(x, 3.55, f"Q={q:.1f}", ha="center", fontsize=9, color=col, fontweight="bold")

    ax.text(5.0, 1.5, rf"$Q(s,a)=R+\gamma V(s')$   $\gamma={gamma}$",
            ha="center", fontsize=10, color=INK, family="monospace")
    ax.text(5.0, 0.7, rf"$V^*(s)=\max_a Q(s,a)={v_star:.1f}$  (greedy: Left)",
            ha="center", fontsize=10, color=DEEP, fontweight="bold")
    ax.text(5.0, 0.15, "Chapter 13.6 single-state arithmetic = one VI sweep cell",
            ha="center", fontsize=8, color="#64748b")
    ax.set_title("Bellman optimality backup (one state)", fontsize=12,
                 fontweight="bold", color=INK, pad=8)

    # Right: bar of Q-values + policy improvement arrow
    ax = axes[1]
    names = [a[0] for a in actions_ch]
    cols = [TEAL, GOLD]
    bars = ax.bar(names, q_vals, color=cols, width=0.55, alpha=0.92)
    ax.axhline(v_star, color=DEEP, ls="--", lw=1.6, label=rf"$V^*(s)={v_star:.1f}$")
    for b, q in zip(bars, q_vals):
        ax.text(b.get_x() + b.get_width() / 2, q + 0.08, f"{q:.1f}",
                ha="center", fontsize=11, fontweight="bold", color=INK)
    # show components
    ax.text(0.02, 0.95,
            "Left:  0 + 0.9×5 = 4.5\nRight: 1 + 0.9×3 = 3.7",
            transform=ax.transAxes, va="top", fontsize=9, color="#475569",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=SOFT, edgecolor="#99f6e4"))
    ax.set_ylabel(r"$Q(s,a)$")
    ax.set_ylim(0, 5.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_ax(ax, "Greedy policy improvement: argmax Q")
    ax.text(
        0.98, 0.08,
        "Expectation form averages over P(s′|s,a);\nhere successors are deterministic.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Bellman backup grid cell: max over actions of R + γV (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_bellman_backup.png")


def fig_metric_decision_tree():
    """Ch18 / glossary: decision tree for choosing a metric family."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def node(x, y, w, h, text, color, fs=8.5):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                facecolor=color, edgecolor="none", alpha=0.95,
            )
        )
        ax.text(x, y, text, ha="center", va="center", color="white",
                fontsize=fs, fontweight="bold")

    def edge(x1, y1, x2, y2, lab=None, col=INK):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        if lab:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2 + 0.15, lab,
                    fontsize=7.5, color="#475569", ha="center")

    # Root
    node(7, 9.2, 4.2, 1.0, "What claim are you making?", DEEP, fs=10)
    # Level 1
    node(2.5, 7.0, 3.4, 1.0, "Ranking / screening\ncases", TEAL)
    node(7.0, 7.0, 3.4, 1.0, "Probabilities for\ncounseling", GOLD)
    node(11.5, 7.0, 3.4, 1.0, "Treat vs not\nat a threshold", "#b45309")
    edge(5.5, 8.7, 2.5, 7.55, "order")
    edge(7.0, 8.7, 7.0, 7.55, "risk #")
    edge(8.5, 8.7, 11.5, 7.55, "action")

    # Level 2 leaves
    node(1.3, 4.6, 2.6, 1.15, "AUROC /\nAUPRC\n(+ prevalence)", TEAL, fs=8)
    node(3.9, 4.6, 2.4, 1.15, "Sens @\nfixed Spec\n(operating pt)", TEAL, fs=8)
    node(7.0, 4.6, 3.2, 1.15, "Calibration plot\nBrier / ECE\nslope & intercept", GOLD, fs=8)
    node(10.5, 4.6, 2.6, 1.15, "Net benefit\ndecision curve", "#b45309", fs=8)
    node(13.0, 4.6, 2.2, 1.15, "Utility\nmatrix /\ncost-sens.", "#b45309", fs=8)

    edge(2.2, 6.5, 1.3, 5.25)
    edge(2.8, 6.5, 3.9, 5.25)
    edge(7.0, 6.5, 7.0, 5.25)
    edge(11.0, 6.5, 10.5, 5.25)
    edge(12.0, 6.5, 13.0, 5.25)

    # Bottom cautions
    node(3.5, 2.0, 5.5, 1.2, "Accuracy alone? Only if classes balanced\nand costs equal — rare in stroke.", "#64748b", fs=8)
    node(10.5, 2.0, 5.8, 1.2, "High AUROC ≠ calibrated ≠ useful.\nPrediction ≠ causation.", "#64748b", fs=8)
    edge(2.5, 4.0, 3.5, 2.65, col="#94a3b8")
    edge(7.0, 4.0, 10.5, 2.65, col="#94a3b8")

    ax.text(7, 0.55,
            "Glossary metric tree: pick the family that matches the claim, then the number.",
            ha="center", fontsize=9, color="#64748b")
    ax.text(7, 9.95, "Metric family decision tree (teaching; original)",
            ha="center", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ml_fig_metric_decision_tree.png")


def fig_external_ladder():
    """Preface: external validation ladder — optimism shrinks as transport hardens."""
    steps = [
        "Resubstitution\n(train=test)",
        "Random\nhold-out",
        "Patient-wise\nCV",
        "Temporal\nsplit",
        "External\nsite",
        "Prospective\nsilent trial",
    ]
    # Synthetic teaching AUROC optimism ladder (not a real study)
    auroc = np.array([0.94, 0.88, 0.86, 0.82, 0.78, 0.76])
    ece = np.array([0.02, 0.04, 0.05, 0.08, 0.11, 0.10])  # calibration often worsens first

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    x = np.arange(len(steps))
    ax.plot(x, auroc, "o-", color=TEAL, lw=2.4, markersize=9, label="AUROC (synthetic)")
    ax.fill_between(x, auroc, 0.70, color=TEAL, alpha=0.08)
    for i, v in enumerate(auroc):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=7.5)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("discrimination (AUROC)")
    ax.axhline(0.80, color="#cbd5e1", ls=":", lw=1.2)
    style_ax(ax, "Optimism shrinks as the test hardens")
    ax.text(0.02, 0.08, "Teaching ladder — not a meta-analysis",
            transform=ax.transAxes, fontsize=8, color="#64748b")

    ax = axes[1]
    ax.plot(x, ece, "s-", color=GOLD, lw=2.4, markersize=8, label="ECE (synthetic)")
    ax.plot(x, 1 - auroc, "o--", color=DEEP, lw=1.8, markersize=6, label="1 − AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("\n", " ") for s in steps], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("error-ish scale (teaching)")
    ax.set_ylim(0, 0.35)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Calibration & ranking both need transport checks")
    ax.text(
        0.98, 0.92,
        "Local AUROC is not a shipping license.\nPrediction ≠ causation; site shift is expected.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("External validation ladder (preface discipline; synthetic teaching)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_external_ladder.png")


def fig_window_cherry():
    """Ch02: cherry-picked time window manufactures a trend."""
    rng = np.random.default_rng(21)
    months = np.arange(1, 25)
    # Wandering door-to-needle with mild noise — no true secular trend
    base = 58 + 2.5 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 1.1, size=len(months))
    # Inject one bad spike month that cherry-pick start loves
    base[5] = 72  # month 6 worst
    base[17] = 54  # later dip

    # Cherry window: month 6 → 18 looks like big improvement
    i0, i1 = 5, 17
    full_slope = np.polyfit(months, base, 1)[0]
    cherry_slope = np.polyfit(months[i0:i1 + 1], base[i0:i1 + 1], 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    ax.plot(months, base, "o-", color=TEAL, lw=2.0, markersize=5)
    ax.axvspan(months[i0], months[i1], color=GOLD, alpha=0.18, label="cherry window")
    ax.plot([months[i0], months[i1]], [base[i0], base[i1]], "s--", color="#dc2626",
            lw=2.2, markersize=8, label="start→end narrative")
    ax.set_xlabel("month")
    ax.set_ylabel("door-to-needle (min)")
    ax.set_ylim(45, 80)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Same series, manufactured 'improvement'")
    ax.text(
        0.02, 0.08,
        f"cherry Δ = {base[i0] - base[i1]:.0f} min  |  full-series slope ≈ {full_slope:+.2f}/mo",
        transform=ax.transAxes, fontsize=8, color="#475569",
    )

    ax = axes[1]
    # Show many random windows' end-start change distribution
    deltas = []
    for _ in range(400):
        a, b = sorted(rng.choice(len(months), size=2, replace=False))
        if b - a < 3:
            continue
        deltas.append(base[a] - base[b])  # positive = "improvement" if lower is better
    deltas = np.array(deltas)
    ax.hist(deltas, bins=18, color=TEAL, alpha=0.85, edgecolor="white")
    cherry_d = base[i0] - base[i1]
    ax.axvline(cherry_d, color="#dc2626", lw=2.2, label=f"cherry Δ={cherry_d:.0f}")
    ax.axvline(0, color="#94a3b8", ls="--", lw=1.3)
    ax.set_xlabel("start − end minutes (positive = looks better)")
    ax.set_ylabel("count of random windows")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Many windows look like 'wins' by chance")
    ax.text(
        0.98, 0.92,
        "Pre-specify window before plotting.\nShow full series + uncertainty.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Cherry-picked time windows (synthetic DTN dashboard; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_window_cherry.png")


def fig_dendrogram_cut():
    """Ch04: agglomerative dendrogram with two cut heights → different k."""
    # Four points on a line: 0,1,10,11 from chapter walk-through
    # Build linkage heights for single vs complete
    # Single: merges (0,1)@1, (10,11)@1, then clusters @9
    # Complete: final merge @11
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))

    def draw_dendro(ax, final_h, title, cut_h, k_label):
        # Manual dendrogram for 4 leaves
        # leaf x positions
        xs = {"P1": 1, "P2": 2, "P3": 4, "P4": 5}
        # first merges
        # merge A: P1-P2 at h=1
        ax.plot([xs["P1"], xs["P1"]], [0, 1], color=TEAL, lw=2)
        ax.plot([xs["P2"], xs["P2"]], [0, 1], color=TEAL, lw=2)
        ax.plot([xs["P1"], xs["P2"]], [1, 1], color=TEAL, lw=2)
        mid12 = 1.5
        # merge B: P3-P4 at h=1
        ax.plot([xs["P3"], xs["P3"]], [0, 1], color=GOLD, lw=2)
        ax.plot([xs["P4"], xs["P4"]], [0, 1], color=GOLD, lw=2)
        ax.plot([xs["P3"], xs["P4"]], [1, 1], color=GOLD, lw=2)
        mid34 = 4.5
        # final merge
        ax.plot([mid12, mid12], [1, final_h], color=DEEP, lw=2)
        ax.plot([mid34, mid34], [1, final_h], color=DEEP, lw=2)
        ax.plot([mid12, mid34], [final_h, final_h], color=DEEP, lw=2)
        # cut line
        ax.axhline(cut_h, color="#dc2626", ls="--", lw=1.8, label=f"cut h={cut_h:g} → {k_label}")
        ax.set_xticks([1, 2, 4, 5])
        ax.set_xticklabels(["P1=0", "P2=1", "P3=10", "P4=11"])
        ax.set_ylabel("merge height (linkage distance)")
        ax.set_ylim(0, max(12, final_h + 1))
        ax.set_xlim(0.5, 5.5)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style_ax(ax, title)
        # annotate heights
        ax.text(1.5, 1.15, "1", ha="center", fontsize=8, color=TEAL)
        ax.text(4.5, 1.15, "1", ha="center", fontsize=8, color=GOLD)
        ax.text(3.0, final_h + 0.25, f"{final_h:g}", ha="center", fontsize=9, color=DEEP, fontweight="bold")

    draw_dendro(
        axes[0], final_h=9, title="Single linkage (ch. walk-through)",
        cut_h=5, k_label="k=2 clusters",
    )
    axes[0].text(
        0.98, 0.08,
        "min cross-distance final merge = 9\nSLINK chains elongated groups",
        transform=axes[0].transAxes, ha="right", fontsize=8, color="#64748b",
    )
    draw_dendro(
        axes[1], final_h=11, title="Complete linkage (same points)",
        cut_h=5, k_label="k=2 clusters",
    )
    axes[1].text(
        0.98, 0.08,
        "max cross-distance final merge = 11\ncut height is a model choice ≈ k",
        transform=axes[1].transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Dendrogram cut height chooses k (P1=0,P2=1,P3=10,P4=11; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_dendrogram_cut.png")


def fig_target_enc_loo():
    """Ch06: target encoding leakage vs leave-one-out (chapter site-A numbers)."""
    # Chapter sketch: site A n=5, ȳ_c=0.40, global ȳ=0.20, m=10 prior strength
    # Naive: e = (5*0.40 + 10*0.20)/(5+10) = (2+2)/15 = 0.267
    # LOO y=1: (2-1+2)/14 = 0.214; LOO y=0: (2-0+2)/14 = 0.286
    n_c, ybar_c, ybar, m = 5, 0.40, 0.20, 10.0
    naive = (n_c * ybar_c + m * ybar) / (n_c + m)
    loo_pos = (n_c * ybar_c - 1 + m * ybar) / (n_c - 1 + m)
    loo_neg = (n_c * ybar_c - 0 + m * ybar) / (n_c - 1 + m)
    # Singleton site B: naive = (1*1 + 10*0.2)/11 = 0.273; LOO collapses to prior 0.20
    naive_b = (1 * 1.0 + m * ybar) / (1 + m)
    loo_b = ybar  # prior

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    labs = ["naive\n(site A)", "LOO y=1", "LOO y=0", "prior ȳ"]
    vals = [naive, loo_pos, loo_neg, ybar]
    cols = ["#dc2626", TEAL, TEAL, "#94a3b8"]
    bars = ax.bar(labs, vals, color=cols, width=0.65, alpha=0.92)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("target encoding e_c")
    style_ax(ax, r"Site A (n=5, $\bar y_c$=0.40, m=10)")
    ax.text(
        0.98, 0.92,
        "Naive uses the row’s own y\nin the category mean → leakage",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )

    ax = axes[1]
    labs2 = ["naive\n(site B n=1)", "LOO / prior\n(site B)", "true\noutcome y"]
    vals2 = [naive_b, loo_b, 1.0]
    cols2 = ["#dc2626", TEAL, GOLD]
    bars = ax.bar(labs2, vals2, color=cols2, width=0.6, alpha=0.92)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("encoded feature / label")
    style_ax(ax, "Singleton site: naive ≈ label; LOO → prior")
    ax.text(
        0.98, 0.92,
        "Small subgroups + naive target enc.\n→ fictional AUROC that dies prospectively",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Target encoding: naive leakage vs leave-one-out (ch.06 numbers; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_target_enc_loo.png")


def fig_tsne_perplexity():
    """Ch07: teaching caution — perplexity changes map geometry (2-D toy)."""
    rng = np.random.default_rng(0)
    # Three isotropic blobs in 2D (already 2-D so 'embedding' is a neighbor graph sketch)
    # We simulate t-SNE-like effects: low perplexity fragments; high merges
    centers = np.array([[0.0, 0.0], [3.5, 0.2], [1.5, 3.0]])
    pts = []
    labs = []
    for i, c in enumerate(centers):
        pts.append(c + rng.normal(0, 0.35, size=(40, 2)))
        labs.append(np.full(40, i))
    # Add continuum bridge points between 0 and 1
    bridge = np.linspace(centers[0], centers[1], 12) + rng.normal(0, 0.08, size=(12, 2))
    X = np.vstack(pts + [bridge])
    y = np.concatenate(labs + [np.full(12, 3)])

    def fake_embed(perplexity_scale):
        # Teaching caricature: not real t-SNE — scales cluster separation & fragments continuum
        Z = X.copy()
        # pull toward local centers more when perplexity low
        for i in range(3):
            mask = y == i
            Z[mask] = centers[i] + (X[mask] - centers[i]) * (0.55 + 0.15 * perplexity_scale)
        # bridge: low perp tears to noise islands; high keeps chain
        bmask = y == 3
        if perplexity_scale < 0.5:
            Z[bmask] += rng.normal(0, 1.2, size=Z[bmask].shape)
            Z[bmask] *= 0.3
        else:
            Z[bmask] = X[bmask] * (0.9 + 0.1 * perplexity_scale)
        # global shrink of inter-cluster gap when high perplexity
        Z = Z - Z.mean(0)
        Z = Z * (0.7 + 0.5 * (1 - perplexity_scale))
        return Z

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    colors = [TEAL, GOLD, DEEP, "#94a3b8"]
    titles = [
        (0.15, "Low perplexity (local only)\n— continuum tears; noise islands"),
        (0.85, "Higher perplexity (broader neighbors)\n— gaps compress; not metric map"),
    ]
    for ax, (ps, title) in zip(axes, titles):
        Z = fake_embed(ps)
        for i, col in enumerate(colors):
            m = y == i
            lab = ["A", "B", "C", "bridge"][i]
            ax.scatter(Z[m, 0], Z[m, 1], c=col, s=28, alpha=0.85, label=lab, edgecolors="white", linewidths=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        style_ax(ax, title)
        ax.legend(frameon=False, fontsize=8, loc="best", markerscale=1.2)
    axes[0].text(
        0.02, 0.02,
        "Do not read cluster size/gap as effect size",
        transform=axes[0].transAxes, fontsize=8, color="#64748b",
    )
    axes[1].text(
        0.02, 0.02,
        "Confirm structure in original space; pred≠cause",
        transform=axes[1].transAxes, fontsize=8, color="#64748b",
    )
    fig.suptitle("t-SNE perplexity changes the picture (teaching caricature — not real t-SNE run)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_tsne_perplexity.png")


def fig_eligibility_trace():
    """Ch13: eligibility trace decay e_t = γλ e_{t-1} + 1{visit}; credit assignment."""
    gamma, lam = 0.9, 0.8
    T = 16
    # Visit schedule: state A at t=0,2,3; B at t=5; C at t=9; terminal reward at t=12 on C path
    visits = {0: "A", 2: "A", 3: "A", 5: "B", 9: "C", 12: "C"}
    states = ["A", "B", "C"]
    e = {s: np.zeros(T) for s in states}
    e_prev = {s: 0.0 for s in states}
    for t in range(T):
        for s in states:
            e_prev[s] = gamma * lam * e_prev[s]
            if visits.get(t) == s:
                e_prev[s] += 1.0  # accumulating trace
            e[s][t] = e_prev[s]
    # TD errors: mostly 0, spike at reward time t=12
    delta = np.zeros(T)
    delta[12] = 1.0
    # Cumulative credit to each state ∝ sum_t delta_t * e_t(s)
    credit = {s: float((delta * e[s]).sum()) for s in states}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    ax = axes[0]
    cols = {"A": TEAL, "B": GOLD, "C": DEEP}
    for s in states:
        ax.plot(np.arange(T), e[s], "-o", color=cols[s], lw=2.2, markersize=4, label=rf"$e({s})$")
    for t, s in visits.items():
        ax.axvline(t, color=cols[s], alpha=0.15, lw=6)
    ax.axvline(12, color="#dc2626", ls="--", lw=1.5, label=r"TD error $\delta_{12}=+1$")
    ax.set_xlabel("time t")
    ax.set_ylabel(r"eligibility $e_t(s)$")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_ax(ax, rf"Accumulating traces  ($\gamma={gamma}$, $\lambda={lam}$)")
    ax.text(
        0.98, 0.92,
        r"$e_t \leftarrow \gamma\lambda e_{t-1} + 1\{S_t=s\}$",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#475569",
        family="monospace",
    )

    ax = axes[1]
    names = list(credit.keys())
    vals = [credit[s] for s in names]
    bars = ax.bar(names, vals, color=[cols[s] for s in names], width=0.55, alpha=0.92)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel(r"credit $\sum_t \delta_t\, e_t(s)$")
    ax.set_ylim(0, max(vals) * 1.25 + 0.05)
    style_ax(ax, "Backward view: who gets the delayed reward?")
    ax.text(
        0.98, 0.92,
        "λ=0 → one-step TD\nλ→1 → MC-like credit\nGAE uses same trade-off",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Eligibility traces assign delayed TD credit (synthetic trajectory; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_eligibility_trace.png")


def fig_journal_club_card():
    """Preface: journal-club scorecard gates for an 'AI' paper."""
    gates = [
        "Claim typed\n(pred/etiology/action)",
        "Index time\n& leakage audit",
        "Cohort &\nlabel source",
        "Discrimination\n+ calibration",
        "Utility @\nclinical thresholds",
        "External /\ntemporal test",
        "Prohibited uses\n& monitoring",
    ]
    # Synthetic paper scores 0-1 for teaching
    paper_a = np.array([0.9, 0.85, 0.8, 0.9, 0.35, 0.4, 0.2])  # AUC-heavy vendor
    paper_b = np.array([0.85, 0.9, 0.85, 0.75, 0.8, 0.85, 0.8])  # disciplined

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(gates))
    w = 0.36
    ax.bar(x - w / 2, paper_a, width=w, color="#dc2626", alpha=0.85, label="Paper A · AUROC poster")
    ax.bar(x + w / 2, paper_b, width=w, color=TEAL, alpha=0.9, label="Paper B · disciplined")
    ax.axhline(0.7, color=GOLD, ls="--", lw=1.5, label="teaching pass ≥0.7")
    ax.set_xticks(x)
    ax.set_xticklabels(gates, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("gate score (teaching 0–1)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Journal-club scorecard: one high AUROC is not enough")
    ax.text(
        0.02, 0.92,
        "Score each gate independently.\nFail any critical gate → do not ship.\nPrediction ≠ causation.",
        transform=ax.transAxes, va="top", fontsize=8, color="#64748b",
    )
    fig.tight_layout()
    save(fig, "ml_fig_journal_club_card.png")


def fig_mae_mask_ratio():
    """Ch11: masked autoencoder — recon error vs mask ratio + patch sketch."""
    rng = np.random.default_rng(4)
    # Synthetic teaching curves: recon MSE rises with mask ratio; linear-probe AUROC peaks mid
    mask = np.array([0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90])
    recon = 0.08 + 0.55 * mask ** 1.4 + rng.normal(0, 0.01, size=len(mask))
    probe = 0.72 + 0.18 * np.exp(-((mask - 0.55) ** 2) / (2 * 0.12 ** 2)) - 0.08 * mask
    probe = np.clip(probe + rng.normal(0, 0.005, size=len(mask)), 0.55, 0.95)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    # 8x8 patch grid with 75% masked
    grid = np.ones((8, 8))
    flat = rng.choice(64, size=int(0.75 * 64), replace=False)
    g = grid.ravel().copy()
    g[flat] = 0.0
    ax.imshow(g.reshape(8, 8), cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
    # grid lines
    for i in range(9):
        ax.axhline(i - 0.5, color="white", lw=0.8)
        ax.axvline(i - 0.5, color="white", lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    style_ax(ax, "MAE patch mask (75% hidden → decoder reconstructs)")
    ax.text(
        0.5, -0.08,
        "Visible patches (teal) encode; masked (dark) are prediction targets",
        transform=ax.transAxes, ha="center", fontsize=8, color="#64748b",
    )

    ax = axes[1]
    ax.plot(mask, recon, "o-", color=GOLD, lw=2.3, markersize=7, label="pretext recon error")
    ax.set_xlabel("mask ratio")
    ax.set_ylabel("recon MSE (synthetic)", color=GOLD)
    ax.tick_params(axis="y", labelcolor=GOLD)
    ax2 = ax.twinx()
    ax2.plot(mask, probe, "s-", color=TEAL, lw=2.3, markersize=7, label="linear-probe AUROC")
    ax2.set_ylabel("downstream probe AUROC", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)
    ax2.set_ylim(0.55, 1.0)
    # peak marker
    ipeak = int(np.argmax(probe))
    ax2.axvline(mask[ipeak], color=DEEP, ls="--", lw=1.3)
    ax2.text(mask[ipeak] + 0.01, 0.97, f"peak≈{mask[ipeak]:.2f}", fontsize=8, color=DEEP)
    style_ax(ax, "Harder pretext ≠ better transfer (always)")
    ax.text(
        0.02, 0.08,
        "Validate with linear probe / fine-tune,\nnot recon loss alone. Pretext ≠ clinical label.",
        transform=ax.transAxes, fontsize=8, color="#64748b",
    )
    fig.suptitle("Masked autoencoding: mask ratio vs recon and transfer (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_mae_mask_ratio.png")


def fig_policy_iteration_loop():
    """Ch13: policy iteration — evaluate then improve until stable."""
    # Two-state MDP from chapter: track policy of s1 over outer iterations
    gamma = 0.9
    # Start with Stay everywhere
    # Show V after evaluation and greedy action flips
    # Outer loop records
    # Manual from chapter knowledge:
    # If pi = Stay,Stay: V_s2=2/(1-0.9)=20, V_s1=1/(1-0.9)=10  wait Stay in s1 gives 1+0.9 V_s1 so V=10
    # Q_go s1 = 0+0.9*20=18 > 10 so improve to Go
    # Then V* = 18, 20
    iters = [0, 1, 2]
    v_s1 = [0.0, 10.0, 18.0]
    v_s2 = [0.0, 20.0, 20.0]
    pi_s1 = ["Stay", "Stay", "Go"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    # flowchart style
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (5, 5.2, "Initialize π₀", TEAL),
        (5, 3.6, "Policy evaluation\nV ← solve (I−γP^π)⁻¹ r^π", DEEP),
        (5, 1.8, "Policy improvement\nπ' ← greedy(Q^π)", GOLD),
        (5, 0.4, "π' = π ?  → stop : π ← π'", "#b45309"),
    ]
    for x, y, lab, c in boxes:
        ax.add_patch(FancyBboxPatch((x - 2.2, y - 0.55), 4.4, 1.1,
                                    boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=c, edgecolor="none"))
        ax.text(x, y, lab, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    for y1, y2 in [(4.65, 4.15), (3.05, 2.35), (1.25, 0.95)]:
        ax.annotate("", xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    # loop back
    ax.annotate("", xy=(2.6, 3.6), xytext=(2.6, 0.4),
                arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.5,
                                connectionstyle="arc3,rad=0.0"))
    ax.plot([2.6, 2.8], [3.6, 3.6], color=DEEP, lw=1.5)
    ax.text(1.5, 2.0, "repeat", fontsize=8, color=DEEP, rotation=90, va="center")
    ax.set_title("Policy iteration loop", fontsize=12, fontweight="bold", color=INK)

    ax = axes[1]
    ax.plot(iters, v_s1, "o-", color=TEAL, lw=2.4, markersize=9, label=r"$V^{\pi}(s_1)$")
    ax.plot(iters, v_s2, "s-", color=GOLD, lw=2.4, markersize=8, label=r"$V^{\pi}(s_2)$")
    ax.axhline(18, color=TEAL, ls=":", lw=1.2)
    ax.axhline(20, color=GOLD, ls=":", lw=1.2)
    for i, p in enumerate(pi_s1):
        ax.text(i, v_s1[i] + 1.2, f"π(s1)={p}", ha="center", fontsize=8, color=DEEP)
    ax.set_xticks(iters)
    ax.set_xticklabels(["init", "after eval\nπ=Stay", "after improve\nπ=Go"])
    ax.set_ylabel("state value")
    ax.set_ylim(-1, 24)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Chapter two-state MDP: improve flips Stay→Go")
    ax.text(
        0.98, 0.08,
        "Policy improvement theorem: V^{π'} ≥ V^{π}\nExact on small known MDPs; samples approx.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Policy iteration: evaluate → improve until stable (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_policy_iteration.png")


def fig_rollback_triggers():
    """Ch17: pre-defined rollback / recalibration triggers on synthetic ops metrics."""
    weeks = np.arange(0, 26)
    # PSI, ECE, AUROC synthetic trajectories with events
    rng = np.random.default_rng(9)
    psi = 0.05 + 0.002 * weeks + rng.normal(0, 0.01, size=len(weeks))
    psi = np.clip(psi, 0, None)
    psi[18:] += 0.12  # scanner change
    ece = 0.04 + 0.001 * weeks + rng.normal(0, 0.005, size=len(weeks))
    ece[20:] += 0.06
    auroc = 0.84 - 0.001 * weeks + rng.normal(0, 0.005, size=len(weeks))
    auroc[22:] -= 0.05

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    ax = axes[0]
    ax.plot(weeks, psi, color=TEAL, lw=2.2, label="PSI (score)")
    ax.plot(weeks, ece, color=GOLD, lw=2.2, label="ECE")
    ax.axhline(0.20, color=TEAL, ls="--", lw=1.3, alpha=0.8)
    ax.axhline(0.10, color=GOLD, ls="--", lw=1.3, alpha=0.8)
    ax.axvline(18, color="#94a3b8", ls=":", lw=1.2)
    ax.text(18.2, 0.28, "scanner\nswap", fontsize=7.5, color="#64748b")
    ax.set_xlabel("week after go-live")
    ax.set_ylabel("monitoring statistic")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Pre-set alarm lines (PSI≥0.2, ECE≥0.1)")
    ax.fill_between(weeks, 0, 0.35, where=psi >= 0.20, color="#fecaca", alpha=0.35)

    ax = axes[1]
    ax.plot(weeks, auroc, "o-", color=DEEP, lw=2.2, markersize=4, label="live AUROC")
    ax.axhline(0.80, color="#dc2626", ls="--", lw=1.6, label="rollback floor 0.80")
    ax.fill_between(weeks, 0.70, auroc, where=auroc < 0.80, color="#fecaca", alpha=0.5, interpolate=True)
    # action annotations
    ax.annotate("recalibrate", xy=(20, ece[20] if False else auroc[20]), xytext=(12, 0.76),
                fontsize=8, color=GOLD, arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
    ax.annotate("rollback", xy=(23, auroc[23]), xytext=(15, 0.72),
                fontsize=8, color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
    ax.set_xlabel("week after go-live")
    ax.set_ylabel("live AUROC (synthetic)")
    ax.set_ylim(0.70, 0.90)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Outcome lag: act on inputs before AUROC dies")
    ax.text(
        0.98, 0.92,
        "Write triggers before go-live.\nPrediction ≠ causation; ops is science.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Silent-trial ops: recalibrate vs rollback triggers (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_rollback_triggers.png")


def fig_leakage_taxonomy():
    """Ch18 glossary: taxonomy of leakage types with clinical examples."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(6, 7.6, "Leakage taxonomy (glossary map)", ha="center", fontsize=12,
            fontweight="bold", color=INK)

    types = [
        (3, 5.5, "Temporal\nleakage", TEAL, "Post-decision labs,\nfinal reads, LOS"),
        (9, 5.5, "Fit / CV\nleakage", DEEP, "Scaler/vocab/select\nfit on full cohort"),
        (3, 2.2, "Label\nproxy leakage", GOLD, "Treatment codes\nas 'features'"),
        (9, 2.2, "Target-enc\nleakage", "#b45309", "Naive mean encoding\nwithout LOO/OOF"),
    ]
    for x, y, title, c, detail in types:
        ax.add_patch(FancyBboxPatch((x - 2.0, y - 1.2), 4.0, 2.4,
                                    boxstyle="round,pad=0.04,rounding_size=0.15",
                                    facecolor=c, edgecolor="none", alpha=0.92))
        ax.text(x, y + 0.45, title, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        ax.text(x, y - 0.55, detail, ha="center", va="center", color="white", fontsize=9)

    ax.text(6, 0.35,
            "Any leakage inflates apparent performance and fails at true index time. Prediction ≠ causation.",
            ha="center", fontsize=9, color="#64748b")
    fig.tight_layout()
    save(fig, "ml_fig_leakage_taxonomy.png")


def fig_cost_threshold_utility():
    """Ch09: cost-sensitive threshold — expected cost vs threshold with FN:FP costs."""
    # Synthetic score distributions
    rng = np.random.default_rng(2)
    pos = rng.beta(5, 2, size=3000)  # higher scores for positives
    neg = rng.beta(2, 5, size=7000)
    # Prevalence in this sample
    n_pos, n_neg = len(pos), len(neg)
    thresholds = np.linspace(0.05, 0.95, 37)
    # Costs: c_fn = 5, c_fp = 1 (chapter triage ratio)
    c_fn, c_fp = 5.0, 1.0
    exp_cost = []
    sens_list, spec_list = [], []
    for t in thresholds:
        tp = (pos >= t).sum()
        fn = (pos < t).sum()
        fp = (neg >= t).sum()
        tn = (neg < t).sum()
        # expected cost per patient
        cost = (c_fn * fn + c_fp * fp) / (n_pos + n_neg)
        exp_cost.append(cost)
        sens_list.append(tp / n_pos)
        spec_list.append(tn / n_neg)
    exp_cost = np.array(exp_cost)
    t_star = thresholds[int(np.argmin(exp_cost))]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    ax.hist(neg, bins=30, density=True, alpha=0.55, color="#94a3b8", label="neg scores")
    ax.hist(pos, bins=30, density=True, alpha=0.55, color=TEAL, label="pos scores")
    ax.axvline(0.5, color="#64748b", ls=":", lw=1.5, label="default t=0.5")
    ax.axvline(t_star, color="#dc2626", ls="--", lw=2.0, label=f"cost-min t≈{t_star:.2f}")
    ax.set_xlabel("model score")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Overlapping scores (synthetic LVO-ish)")

    ax = axes[1]
    ax.plot(thresholds, exp_cost, color=DEEP, lw=2.4, label=rf"E[cost], $c_{{FN}}$={c_fn:g}, $c_{{FP}}$={c_fp:g}")
    ax.axvline(t_star, color="#dc2626", ls="--", lw=1.8)
    ax.axvline(0.5, color="#64748b", ls=":", lw=1.5)
    ax.plot(t_star, exp_cost.min(), "o", color="#dc2626", markersize=9)
    ax.set_xlabel("threshold t")
    ax.set_ylabel("expected cost per case")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Cost-sensitive operating point ≠ 0.5")
    ax.text(
        0.98, 0.92,
        "Tune t on validation with clinical costs.\nYouden/0.5 can be the wrong policy.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Cost-sensitive threshold selection (FN cost 5× FP; synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_cost_threshold.png")


def fig_modularity_communities():
    """Ch15: modularity Q vs number of communities on a planted partition graph."""
    rng = np.random.default_rng(12)
    # Build simple SBM-like adjacency: 3 communities of 8 nodes
    n_c, c = 8, 3
    n = n_c * c
    A = np.zeros((n, n))
    for ci in range(c):
        nodes = np.arange(ci * n_c, (ci + 1) * n_c)
        for i in nodes:
            for j in nodes:
                if i < j and rng.random() < 0.55:
                    A[i, j] = A[j, i] = 1
    # sparse between
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and (i // n_c) != (j // n_c) and rng.random() < 0.05:
                A[i, j] = A[j, i] = 1
    m = A.sum() / 2
    k = A.sum(axis=1)
    # Modularity for a partition labels
    def modularity(labels):
        Q = 0.0
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += A[i, j] - (k[i] * k[j]) / (2 * m)
        return Q / (2 * m)

    # Partitions: all one; random; true 3; oversplit each community in half (k=6)
    lab1 = np.zeros(n, dtype=int)
    lab_true = np.repeat(np.arange(c), n_c)
    lab_rand = rng.integers(0, 3, size=n)
    lab6 = np.repeat(np.arange(6), n // 6)
    # k from 1..6 by merging true labels randomly for intermediate
    Qs = []
    ks = [1, 2, 3, 4, 5, 6]
    for kk in ks:
        if kk == 1:
            labs = lab1
        elif kk == 3:
            labs = lab_true
        elif kk == 6:
            labs = lab6
        elif kk == 2:
            labs = lab_true.copy()
            labs[labs == 2] = 1  # merge 2 into 1
        else:
            # random assignment into kk communities (noisy)
            labs = rng.integers(0, kk, size=n)
        Qs.append(modularity(labs))
    Qs = np.array(Qs)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    # spring-ish layout by community blocks
    pos = {}
    centers = [(-1.2, 0), (1.2, 0.3), (0, 1.4)]
    for i in range(n):
        ci = i // n_c
        ang = 2 * np.pi * (i % n_c) / n_c
        pos[i] = centers[ci] + 0.35 * np.array([np.cos(ang), np.sin(ang)])
    cols = [TEAL, GOLD, DEEP]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                same = (i // n_c) == (j // n_c)
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color="#cbd5e1" if not same else "#99f6e4", lw=0.7 if not same else 1.0, zorder=1)
    for i in range(n):
        ax.scatter(pos[i][0], pos[i][1], c=cols[i // n_c], s=55, zorder=2, edgecolors="white", linewidths=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    style_ax(ax, "Planted 3-community graph (synthetic)")
    ax.text(0.5, -0.05, "dense within · sparse between", transform=ax.transAxes,
            ha="center", fontsize=8, color="#64748b")

    ax = axes[1]
    ax.plot(ks, Qs, "o-", color=TEAL, lw=2.4, markersize=8)
    ax.axvline(3, color=GOLD, ls="--", lw=1.5, label="true k=3")
    ax.set_xlabel("number of communities in partition")
    ax.set_ylabel("modularity Q")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, r"$Q$ peaks near true modules (teaching)")
    ax.text(
        0.98, 0.08,
        "Max Q ≠ guaranteed clinical modules\nEdge definition injects artifacts",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Community modularity on a planted partition (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_modularity_q.png")


def fig_magnitude_prune():
    """Ch14: magnitude pruning — sparsity vs accuracy and FLOPs proxy."""
    sparsity = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # Synthetic teaching curves
    acc_unstruct = 0.86 - 0.02 * sparsity - 0.25 * np.maximum(sparsity - 0.7, 0) ** 2
    acc_struct = 0.86 - 0.05 * sparsity - 0.4 * np.maximum(sparsity - 0.5, 0) ** 2
    flops_proxy = 1.0 - 0.85 * sparsity  # unstructured rarely realizes full FLOP cut
    flops_struct = 1.0 - sparsity

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    ax.plot(sparsity, acc_unstruct, "o-", color=TEAL, lw=2.3, label="unstructured magnitude")
    ax.plot(sparsity, acc_struct, "s-", color=GOLD, lw=2.3, label="structured (channels)")
    ax.axhline(0.80, color="#dc2626", ls="--", lw=1.4, label="clinical floor 0.80")
    ax.set_xlabel("sparsity (fraction zeros)")
    ax.set_ylabel("validation accuracy (synthetic)")
    ax.set_ylim(0.55, 0.92)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Accuracy vs sparsity")
    ax.text(0.02, 0.08, "Rewind/fine-tune not shown; lottery ticket optional",
            transform=ax.transAxes, fontsize=8, color="#64748b")

    ax = axes[1]
    ax.plot(sparsity, flops_proxy, "o-", color=TEAL, lw=2.3, label="unstructured FLOPs (realized)")
    ax.plot(sparsity, flops_struct, "s-", color=GOLD, lw=2.3, label="structured FLOPs")
    ax.set_xlabel("sparsity")
    ax.set_ylabel("relative compute (1 = dense)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Hardware only loves structured zeros")
    ax.text(
        0.98, 0.92,
        "Sparse masks need sparse kernels.\nMeasure latency on target device.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Magnitude pruning: accuracy and compute vs sparsity (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_magnitude_prune.png")


def fig_lr_prevalence():
    """Ch03: likelihood ratios map pre-test odds → post-test odds across prevalence."""
    sens, spec = 0.85, 0.70  # match glossary teaching screen
    lr_pos = sens / (1 - spec)
    lr_neg = (1 - sens) / spec
    pi = np.linspace(0.02, 0.80, 80)
    # odds
    pre_odds = pi / (1 - pi)
    post_pos = pre_odds * lr_pos
    post_neg = pre_odds * lr_neg
    ppv = post_pos / (1 + post_pos)
    npv = 1 - post_neg / (1 + post_neg)  # 1 - P(Y=1|neg) = NPV if binary

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    ax.plot(pi, ppv, color=TEAL, lw=2.5, label="PPV after + test")
    ax.plot(pi, pi, color="#94a3b8", ls=":", lw=1.5, label="pre-test = prevalence")
    ax.axvline(0.20, color=GOLD, ls="--", lw=1.4)
    ax.text(0.21, 0.15, "π=0.20\n(ch. LVO ED)", fontsize=8, color=GOLD)
    ax.set_xlabel("pre-test prevalence π")
    ax.set_ylabel("post-test probability")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, rf"LR+ = {lr_pos:.2f}  (sens={sens}, spec={spec})")
    ax.text(0.98, 0.92, r"post odds = pre odds $\times$ LR+",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#475569",
            family="monospace")

    ax = axes[1]
    ax.plot(pi, 1 - npv, color="#dc2626", lw=2.3, label="P(Y=1 | −) = 1−NPV")
    ax.plot(pi, pi, color="#94a3b8", ls=":", lw=1.5, label="prevalence")
    ax.set_xlabel("pre-test prevalence π")
    ax.set_ylabel("post-test P(disease)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, rf"LR− = {lr_neg:.2f}  (negative test)")
    ax.text(
        0.98, 0.08,
        "Same sens/spec → different PPV/NPV\nwhen base rate moves. Copying a paper’s\nPPV is a prevalence error.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Likelihood ratios and prevalence (Bayes; teaching numbers; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_lr_prevalence.png")


def fig_batchnorm_train_eval():
    """Ch10: batch norm — batch stats vs running mean; train/eval mismatch."""
    rng = np.random.default_rng(6)
    # Population N(0,1); small batches estimate mean/var noisily
    pop = rng.normal(0, 1, size=5000)
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    mean_err = []
    var_err = []
    for B in batch_sizes:
        errs_m, errs_v = [], []
        for _ in range(200):
            b = rng.choice(pop, size=B, replace=False)
            errs_m.append(abs(b.mean() - 0.0))
            errs_v.append(abs(b.var(ddof=0) - 1.0))
        mean_err.append(np.mean(errs_m))
        var_err.append(np.mean(errs_v))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    ax.plot(batch_sizes, mean_err, "o-", color=TEAL, lw=2.3, label=r"E[|μ_B − μ|]")
    ax.plot(batch_sizes, var_err, "s-", color=GOLD, lw=2.3, label=r"E[|σ²_B − σ²|]")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("batch size B")
    ax.set_ylabel("mean absolute error of batch moments")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Small batches → noisy BN stats")

    ax = axes[1]
    # Train uses batch; eval uses running — show schematic distribution shift of normalized z
    x = np.linspace(-3, 3, 200)
    # train normalize with batch mean offset
    z_train = (x - 0.4) / 0.7
    z_eval = (x - 0.0) / 1.0
    ax.plot(x, np.exp(-0.5 * z_train ** 2) / np.sqrt(2 * np.pi), color=TEAL, lw=2.3, label="train (batch μ,σ)")
    ax.plot(x, np.exp(-0.5 * z_eval ** 2) / np.sqrt(2 * np.pi), color=GOLD, lw=2.3, label="eval (running μ,σ)")
    ax.set_xlabel("pre-BN activation (schematic)")
    ax.set_ylabel("density after normalize (schematic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Train/eval BN path must match serving")
    ax.text(
        0.98, 0.92,
        "Frozen running stats at deploy.\nTiny B or multi-GPU → sync BN or\nGroup/LayerNorm alternatives.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Batch normalization: batch moments vs running averages (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_batchnorm.png")


def fig_train_serve_skew():
    """Ch01/ch16: train-serve skew — feature distribution shift at deployment."""
    rng = np.random.default_rng(8)
    # Feature: door-to-CT minutes
    train = rng.normal(25, 8, size=2000)
    # Serve: pathway change → faster imaging
    serve = rng.normal(18, 7, size=2000)
    train = np.clip(train, 1, None)
    serve = np.clip(serve, 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    bins = np.linspace(0, 55, 25)
    ax.hist(train, bins=bins, density=True, alpha=0.65, color=TEAL, label="train window")
    ax.hist(serve, bins=bins, density=True, alpha=0.65, color=GOLD, label="serve window")
    ax.set_xlabel("door-to-CT (min)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Feature shift after pathway change")

    ax = axes[1]
    # PSI-like on feature
    bins2 = np.linspace(0, 55, 11)
    t_h, _ = np.histogram(train, bins=bins2)
    s_h, _ = np.histogram(serve, bins=bins2)
    t_p = t_h / t_h.sum() + 1e-6
    s_p = s_h / s_h.sum() + 1e-6
    psi = ((s_p - t_p) * np.log(s_p / t_p)).sum()
    centers = 0.5 * (bins2[:-1] + bins2[1:])
    w = (bins2[1] - bins2[0]) * 0.35
    ax.bar(centers - w / 2, t_p, width=w, color=TEAL, alpha=0.85, label="train")
    ax.bar(centers + w / 2, s_p, width=w, color=GOLD, alpha=0.85, label="serve")
    ax.set_xlabel("door-to-CT bin")
    ax.set_ylabel("proportion")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, f"Feature PSI ≈ {psi:.2f}")
    ax.text(
        0.98, 0.92,
        "Same model weights, different input law.\nMonitor features, not only labels.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Train–serve skew (synthetic door-to-CT shift; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_train_serve_skew.png")


def fig_oversmoothing_gcn():
    """Ch15/GNN: over-smoothing — node feature variance collapses with depth."""
    rng = np.random.default_rng(3)
    n, d = 40, 8
    # Random graph with communities
    A = (rng.random((n, n)) < 0.12).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0)
    # Add self-loops and normalize Â = D^{-1/2}(A+I)D^{-1/2}
    A_hat = A + np.eye(n)
    deg = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
    S = D_inv_sqrt @ A_hat @ D_inv_sqrt
    X = rng.normal(0, 1, size=(n, d))
    # Add community signal
    X[:20] += 1.5
    X[20:] -= 1.5
    depths = list(range(0, 16))
    var_mean = []
    H = X.copy()
    for L in depths:
        if L > 0:
            H = S @ H  # linear GCN-like diffusion without nonlinearity
        var_mean.append(H.var(axis=0).mean())
    var_mean = np.array(var_mean)
    # pairwise cosine diversity of rows
    def row_div(H):
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        C = Hn @ Hn.T
        return 1 - C[np.triu_indices(n, 1)].mean()
    divs = []
    H = X.copy()
    for L in depths:
        if L > 0:
            H = S @ H
        divs.append(row_div(H))
    divs = np.array(divs)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    ax.plot(depths, var_mean, "o-", color=TEAL, lw=2.4, markersize=6)
    ax.set_xlabel("message-passing depth L")
    ax.set_ylabel("mean feature variance across nodes")
    style_ax(ax, "Over-smoothing: features homogenize")
    ax.text(0.98, 0.92, r"linear $\hat{A}^L X$ diffusion",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#475569")

    ax = axes[1]
    ax.plot(depths, divs, "s-", color=GOLD, lw=2.4, markersize=6)
    ax.set_xlabel("depth L")
    ax.set_ylabel("mean pairwise diversity (1 − cos)")
    style_ax(ax, "Node embeddings become indistinguishable")
    ax.text(
        0.98, 0.08,
        "Residual/JK connections, GAT, or\nshallower nets mitigate. More layers\n≠ better on small clinical graphs.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("GCN over-smoothing with depth (synthetic diffusion; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_oversmoothing.png")


def fig_ssl_probe_hygiene():
    """Ch11: linear-probe hygiene — frozen encoder vs leaky fine-tune/probe protocol."""
    rng = np.random.default_rng(21)
    # Synthetic teaching: methods A..E with increasing protocol violations
    methods = [
        "Linear probe\n(frozen)",
        "k-NN probe\n(frozen)",
        "Probe + light\nhead FT",
        "Full FT\n(leaky)",
        "Tune on\ntest split",
    ]
    # True representation quality ≈ 0.78; protocol inflation grows
    true_rep = 0.78
    auroc = np.array([0.78, 0.76, 0.81, 0.89, 0.94])
    # Variance across seeds (honest protocols more stable)
    se = np.array([0.015, 0.018, 0.025, 0.04, 0.05])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    x = np.arange(len(methods))
    colors = [TEAL, TEAL, GOLD, "#dc2626", "#991b1b"]
    ax.bar(x, auroc, color=colors, alpha=0.9, edgecolor="white", width=0.7)
    ax.errorbar(x, auroc, yerr=1.96 * se, fmt="none", ecolor=INK, capsize=3, lw=1.2)
    ax.axhline(true_rep, color=DEEP, ls="--", lw=1.6, label=f"frozen-probe truth ≈ {true_rep}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("downstream AUROC (synthetic)")
    ax.set_ylim(0.65, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax, "Protocol inflation looks like better SSL")
    ax.text(
        0.98, 0.08,
        "Red bars mix representation\nquality with fine-tune budget\nand test leakage.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )

    ax = axes[1]
    # Two-panel schematic: frozen vs unfrozen during "probe"
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # Frozen path
    ax.add_patch(FancyBboxPatch((0.3, 3.5), 2.4, 1.6, boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=TEAL, edgecolor="none"))
    ax.text(1.5, 4.3, "Encoder\nFROZEN", ha="center", va="center", color="white",
            fontsize=9, fontweight="bold")
    ax.add_patch(FancyBboxPatch((3.5, 3.5), 2.4, 1.6, boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=GOLD, edgecolor="none"))
    ax.text(4.7, 4.3, "Linear\nprobe only", ha="center", va="center", color=INK,
            fontsize=9, fontweight="bold")
    ax.annotate("", xy=(3.4, 4.3), xytext=(2.8, 4.3),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.text(4.7, 3.15, "compares embeddings", ha="center", fontsize=8, color="#64748b")

    # Leaky path
    ax.add_patch(FancyBboxPatch((0.3, 0.6), 2.4, 1.6, boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor="#f87171", edgecolor="none"))
    ax.text(1.5, 1.4, "Encoder\nUNFROZEN", ha="center", va="center", color="white",
            fontsize=9, fontweight="bold")
    ax.add_patch(FancyBboxPatch((3.5, 0.6), 2.4, 1.6, boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor="#fecaca", edgecolor="#dc2626", linewidth=1.5))
    ax.text(4.7, 1.4, "Full model\n+ labels", ha="center", va="center", color=INK,
            fontsize=9, fontweight="bold")
    ax.annotate("", xy=(3.4, 1.4), xytext=(2.8, 1.4),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8))
    ax.text(4.7, 0.25, "confounds pretrain quality", ha="center", fontsize=8, color="#dc2626")

    ax.add_patch(FancyBboxPatch((6.6, 1.5), 3.0, 3.0, boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=SOFT, edgecolor=TEAL, linewidth=1.5))
    ax.text(8.1, 3.9, "Hygiene rules", ha="center", fontsize=10, fontweight="bold", color=DEEP)
    rules = [
        "1. Freeze encoder for probes",
        "2. Fixed labeled splits",
        "3. Same head / budget",
        "4. No test for selection",
        "5. Report FT separately",
    ]
    for i, r in enumerate(rules):
        ax.text(6.85, 3.35 - 0.4 * i, r, fontsize=8, color=INK, va="center")
    style_ax(ax, "Probe vs fine-tune (not interchangeable)")
    ax.set_title("Probe vs fine-tune (not interchangeable)", fontsize=13,
                 fontweight="bold", color=INK, pad=10)

    fig.suptitle("SSL linear-probe hygiene (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_ssl_probe_hygiene.png")


def fig_brier_components():
    """Ch08/09: Brier score decomposition — reliability, resolution, uncertainty."""
    rng = np.random.default_rng(7)
    # Three synthetic models on same prevalence π
    pi = 0.20
    n = 2000
    y = rng.binomial(1, pi, size=n)

    def brier(p, y):
        return float(np.mean((p - y) ** 2))

    def decomp(p, y, n_bins=10):
        # Murphy decomposition: BS = REL - RES + UNC
        # UNC = π(1-π); REL = E[(p_bin - o_bin)^2]; RES = E[(o_bin - π)^2]
        edges = np.linspace(0, 1, n_bins + 1)
        # soft bin by predicted p
        bins = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        rel = res = 0.0
        pi_hat = y.mean()
        unc = pi_hat * (1 - pi_hat)
        for b in range(n_bins):
            m = bins == b
            if not np.any(m):
                continue
            w = m.mean()
            p_bar = p[m].mean()
            o_bar = y[m].mean()
            rel += w * (p_bar - o_bar) ** 2
            res += w * (o_bar - pi_hat) ** 2
        return rel, res, unc, rel - res + unc

    # Model A: well calibrated, moderate discrimination
    logit = 1.2 * (y + rng.normal(0, 0.9, size=n) - 0.5)
    p_good = 1 / (1 + np.exp(-logit))
    # Platt-ish rescale toward base rate for nicer cal
    p_good = 0.7 * p_good + 0.3 * pi
    p_good = np.clip(p_good, 0.02, 0.98)

    # Model B: same ranking-ish but overconfident
    p_over = np.clip((p_good - 0.5) * 1.8 + 0.5, 0.01, 0.99)

    # Model C: under-dispersed (always near prevalence) — low RES
    p_flat = np.clip(pi + rng.normal(0, 0.04, size=n), 0.05, 0.4)

    models = [
        ("Calibrated", p_good, TEAL),
        ("Overconfident", p_over, GOLD),
        ("Near-constant", p_flat, "#64748b"),
    ]
    names, rels, ress, uncs, bss = [], [], [], [], []
    for name, p, _ in models:
        rel, res, unc, bs = decomp(p, y)
        names.append(name)
        rels.append(rel)
        ress.append(res)
        uncs.append(unc)
        bss.append(bs)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.22
    ax.bar(x - w, rels, w, label="reliability (↓ better)", color="#dc2626", alpha=0.85)
    ax.bar(x, ress, w, label="resolution (↑ better)", color=TEAL, alpha=0.9)
    ax.bar(x + w, uncs, w, label="uncertainty (base rate)", color="#94a3b8", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Brier component")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    style_ax(ax, r"Murphy: BS = REL − RES + UNC")
    ax.text(
        0.02, 0.92,
        f"UNC = π(1−π) ≈ {uncs[0]:.3f}\n(same cohort for all)",
        transform=ax.transAxes, fontsize=8, color="#64748b", va="top",
    )

    ax = axes[1]
    # Reliability diagram for calibrated vs overconfident
    def rel_curve(p, y, n_bins=8):
        edges = np.linspace(0, 1, n_bins + 1)
        centers, obs, conf = [], [], []
        for i in range(n_bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
            if m.sum() < 15:
                continue
            centers.append(p[m].mean())
            obs.append(y[m].mean())
            conf.append(p[m].mean())
        return np.array(centers), np.array(obs)

    for name, p, c in models[:2]:
        cx, ox = rel_curve(p, y)
        ax.plot(cx, ox, "o-", color=c, lw=2.2, markersize=6, label=f"{name} (BS={brier(p, y):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#94a3b8", lw=1.5)
    ax.set_xlabel("mean predicted probability in bin")
    ax.set_ylabel("observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Reliability diagram (same y)")
    ax.text(
        0.98, 0.08,
        "Low BS needs both good REL\nand high RES. Constant π is\ncalibrated but useless.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Brier score components: reliability, resolution, uncertainty (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_brier_decomp.png")


def fig_cindex_pairs():
    """Ch08: survival C-index intuition — pairwise concordance with censoring."""
    rng = np.random.default_rng(11)
    # Synthetic patients: risk score r, true time T, censor indicator
    n = 40
    risk = rng.normal(0, 1, size=n)
    # Higher risk → shorter time
    T = np.exp(1.2 - 0.7 * risk + rng.normal(0, 0.35, size=n))
    C = rng.uniform(0.5, T.max() * 0.9, size=n)
    observed = np.minimum(T, C)
    event = T <= C  # 1 if failure observed

    # All comparable pairs: i event and T_i < observed_j (standard Harrell-style comparable)
    conc = disc = tied = 0
    pair_xy = []  # for scatter of comparable pairs
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if observed[i] < observed[j]:
                # comparable: i failed before j's follow-up
                if risk[i] > risk[j]:
                    conc += 1
                    pair_xy.append((risk[i] - risk[j], 1))
                elif risk[i] < risk[j]:
                    disc += 1
                    pair_xy.append((risk[i] - risk[j], -1))
                else:
                    tied += 1
    total = conc + disc + tied
    c_index = (conc + 0.5 * tied) / total if total else 0.5

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    # Time vs risk: events vs censored
    ax.scatter(risk[event], observed[event], c=TEAL, s=40, alpha=0.85, label="event", zorder=3)
    ax.scatter(risk[~event], observed[~event], c=GOLD, s=40, alpha=0.85, marker="s", label="censored", zorder=3)
    # Show a few censor marks as upward ticks
    for ri, oi in zip(risk[~event][:8], observed[~event][:8]):
        ax.plot([ri, ri], [oi, oi + 0.15], color=GOLD, lw=1.2)
    ax.set_xlabel("predicted risk score (higher = worse)")
    ax.set_ylabel("observed time (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Risk should rank shorter times higher")
    ax.text(
        0.98, 0.92,
        "Censoring hides some pairs:\nonly comparable pairs enter C.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )

    ax = axes[1]
    # Stacked bar of pair outcomes
    cats = ["concordant", "discordant", "tied risk"]
    vals = [conc, disc, tied]
    cols = [TEAL, "#dc2626", "#94a3b8"]
    ax.bar(cats, vals, color=cols, edgecolor="white", width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("number of comparable pairs")
    style_ax(ax, rf"Harrell C ≈ {c_index:.2f}  (n={n} toy)")
    ax.text(
        0.5, 0.92,
        r"$C = \frac{\#\mathrm{conc} + 0.5\cdot\#\mathrm{ties}}{\#\mathrm{comparable}}$",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, color=DEEP,
    )
    ax.text(
        0.98, 0.08,
        "Not a calibration metric.\nNot causal treatment effect.\nNeeds enough uncensored pairs.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Survival C-index: pairwise concordance under censoring (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_cindex_pairs.png")


def fig_positional_encoding_heatmap():
    """Ch12: sinusoidal positional encoding PE[pos, dim] heatmap + one dimension."""
    d_model = 32
    max_len = 64
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    div = 10000 ** (i / d_model)
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("teal_pe", ["#ecfeff", TEAL, DEEP, INK])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    im = ax.imshow(pe.T, aspect="auto", cmap=cmap, origin="lower",
                   extent=[-0.5, max_len - 0.5, -0.5, d_model - 0.5], vmin=-1, vmax=1)
    ax.set_xlabel("position pos")
    ax.set_ylabel("dimension i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="PE value")
    style_ax(ax, r"Sinusoidal PE[pos, i]")
    ax.text(
        0.02, 0.95,
        r"even $i$: $\sin$, odd: $\cos$" + "\n" + r"$\omega_i = 10000^{-2i/d}$",
        transform=ax.transAxes, fontsize=8, color="white", va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=DEEP, alpha=0.75, edgecolor="none"),
    )

    ax = axes[1]
    dims_show = [0, 1, 4, 5, 14, 15]
    for di, d in enumerate(dims_show):
        ax.plot(np.arange(max_len), pe[:, d] + 0.15 * di, color=TEAL if d % 2 == 0 else GOLD,
                lw=1.8, label=f"i={d}")
    ax.set_xlabel("position pos")
    ax.set_ylabel("PE offset (stacked for display)")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    style_ax(ax, "Low-i: slow waves · high-i: fast")
    ax.text(
        0.02, 0.08,
        "Attention is permutation-equivariant\nwithout PE. RoPE/ALiBi are relatives\nfor long clinical notes.",
        transform=ax.transAxes, fontsize=8, color="#64748b",
    )
    fig.suptitle("Transformer positional encoding (sinusoidal teaching sketch; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_pos_encoding.png")


def fig_fairness_tradeoff():
    """Ch16: fairness–accuracy tradeoff under group base-rate differences."""
    rng = np.random.default_rng(4)
    # Two groups with different prevalence; same underlying score skill but shift
    n0, n1 = 800, 400
    pi0, pi1 = 0.25, 0.10  # base rates differ
    # Latent ability
    y0 = rng.binomial(1, pi0, size=n0)
    y1 = rng.binomial(1, pi1, size=n1)
    s0 = 0.9 * y0 + 0.35 * rng.normal(size=n0)
    s1 = 0.9 * y1 + 0.35 * rng.normal(size=n1) - 0.15  # slight measurement shift

    def metrics(s, y, t):
        pred = s >= t
        tp = ((pred) & (y == 1)).sum()
        fn = ((~pred) & (y == 1)).sum()
        fp = ((pred) & (y == 0)).sum()
        tn = ((~pred) & (y == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        acc = (tp + tn) / (tp + tn + fp + fn)
        return tpr, fpr, acc

    thresholds = np.linspace(s0.min(), s0.max(), 60)
    # Shared threshold policy
    gap_tpr, acc_shared = [], []
    for t in thresholds:
        tpr0, fpr0, _ = metrics(s0, y0, t)
        tpr1, fpr1, _ = metrics(s1, y1, t)
        # overall accuracy
        pred0 = s0 >= t
        pred1 = s1 >= t
        acc = (np.sum(pred0 == y0) + np.sum(pred1 == y1)) / (n0 + n1)
        gap_tpr.append(abs(tpr0 - tpr1))
        acc_shared.append(acc)
    gap_tpr = np.array(gap_tpr)
    acc_shared = np.array(acc_shared)

    # Equalized-odds style: pick per-group thresholds to match TPR≈0.80 and plot residual FPR gap vs overall acc
    target_tprs = np.linspace(0.5, 0.95, 25)
    fpr_gaps, accs_eo, tpr_levels = [], [], []
    for tgt in target_tprs:
        # find t per group for TPR closest to tgt
        def t_for_tpr(s, y, tgt):
            best_t, best = thresholds[0], 1e9
            for t in thresholds:
                tpr, _, _ = metrics(s, y, t)
                if abs(tpr - tgt) < best:
                    best, best_t = abs(tpr - tgt), t
            return best_t
        t0 = t_for_tpr(s0, y0, tgt)
        t1 = t_for_tpr(s1, y1, tgt)
        _, fpr0, _ = metrics(s0, y0, t0)
        _, fpr1, _ = metrics(s1, y1, t1)
        pred0 = s0 >= t0
        pred1 = s1 >= t1
        acc = (np.sum(pred0 == y0) + np.sum(pred1 == y1)) / (n0 + n1)
        fpr_gaps.append(abs(fpr0 - fpr1))
        accs_eo.append(acc)
        tpr_levels.append(tgt)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    ax.plot(acc_shared, gap_tpr, "o-", color=TEAL, lw=2.0, markersize=3.5, label="shared threshold path")
    # Mark max-acc shared
    i_best = int(np.argmax(acc_shared))
    ax.scatter([acc_shared[i_best]], [gap_tpr[i_best]], s=90, c=GOLD, zorder=4, label="max-acc shared t")
    ax.set_xlabel("overall accuracy")
    ax.set_ylabel("|TPR₀ − TPR₁| (equal opportunity gap)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Shared cut: accuracy vs TPR gap")
    ax.text(
        0.98, 0.92,
        f"Base rates π₀={pi0}, π₁={pi1}\nIdentical skill ≠ equal rates",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )

    ax = axes[1]
    ax.plot(accs_eo, fpr_gaps, "s-", color=GOLD, lw=2.0, markersize=5, label="group-specific t (match TPR)")
    ax.set_xlabel("overall accuracy")
    ax.set_ylabel("|FPR₀ − FPR₁| after matching TPR")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Equal opportunity can leave FPR gap")
    ax.text(
        0.98, 0.08,
        "No single threshold erases all\nparity gaps when base rates\ndiffer. Document who is harmed.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Fairness–accuracy tradeoff under unequal prevalence (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_fairness_tradeoff.png")


def fig_adjusted_rand_stability():
    """Ch04: Adjusted Rand Index vs k and bootstrap stability of partitions."""
    rng = np.random.default_rng(9)
    # Three true blobs
    n_per, k_true = 60, 3
    centers = np.array([[0, 0], [3.2, 0.2], [1.4, 2.8]])
    X = []
    y_true = []
    for j, c in enumerate(centers):
        X.append(c + rng.normal(0, 0.55, size=(n_per, 2)))
        y_true.append(np.full(n_per, j))
    X = np.vstack(X)
    y_true = np.concatenate(y_true)

    def kmeans(X, k, n_init=8, max_iter=40):
        best_inertia, best_lab = np.inf, None
        for _ in range(n_init):
            cents = X[rng.choice(len(X), size=k, replace=False)].copy()
            for _ in range(max_iter):
                d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
                lab = d.argmin(axis=1)
                new = np.array([
                    X[lab == j].mean(axis=0) if np.any(lab == j) else cents[j]
                    for j in range(k)
                ])
                if np.allclose(new, cents):
                    break
                cents = new
            inertia = ((X - cents[lab]) ** 2).sum()
            if inertia < best_inertia:
                best_inertia, best_lab = inertia, lab.copy()
        return best_lab

    def ari(a, b):
        # Adjusted Rand Index
        a = np.asarray(a)
        b = np.asarray(b)
        n = len(a)
        # contingency
        ka, kb = a.max() + 1, b.max() + 1
        # remap labels to 0..
        _, a = np.unique(a, return_inverse=True)
        _, b = np.unique(b, return_inverse=True)
        ka, kb = a.max() + 1, b.max() + 1
        cont = np.zeros((ka, kb), dtype=int)
        for i in range(n):
            cont[a[i], b[i]] += 1
        sum_comb_c = 0.0
        for i in range(ka):
            for j in range(kb):
                nij = cont[i, j]
                sum_comb_c += nij * (nij - 1) / 2
        sum_comb_a = sum(ni * (ni - 1) / 2 for ni in cont.sum(axis=1))
        sum_comb_b = sum(nj * (nj - 1) / 2 for nj in cont.sum(axis=0))
        comb_n = n * (n - 1) / 2
        expected = sum_comb_a * sum_comb_b / comb_n if comb_n else 0
        max_index = 0.5 * (sum_comb_a + sum_comb_b)
        return (sum_comb_c - expected) / (max_index - expected + 1e-12)

    ks = list(range(2, 8))
    ari_true = []
    ari_boot_mean = []
    ari_boot_lo = []
    ari_boot_hi = []
    for k in ks:
        lab = kmeans(X, k)
        ari_true.append(ari(y_true, lab))
        # bootstrap stability: ARI between full-data labeling and bootstrap refits
        boots = []
        for _ in range(25):
            idx = rng.integers(0, len(X), size=len(X))
            lab_b = kmeans(X[idx], k)
            # map bootstrap labels back is hard; instead ARI on bootstrap subset
            boots.append(ari(lab[idx], lab_b))
        boots = np.array(boots)
        ari_boot_mean.append(boots.mean())
        ari_boot_lo.append(np.percentile(boots, 10))
        ari_boot_hi.append(np.percentile(boots, 90))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    cols = [TEAL, GOLD, DEEP]
    for j in range(k_true):
        m = y_true == j
        ax.scatter(X[m, 0], X[m, 1], s=18, c=cols[j], alpha=0.8, edgecolors="none")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_xticks([])
    ax.set_yticks([])
    style_ax(ax, f"Planted k={k_true} blobs (audit labels)")
    ax.text(0.5, -0.08, "ARI needs a reference ontology — not pure discovery",
            transform=ax.transAxes, ha="center", fontsize=8, color="#64748b")

    ax = axes[1]
    ax.plot(ks, ari_true, "o-", color=TEAL, lw=2.4, markersize=7, label="ARI vs planted labels")
    ax.fill_between(ks, ari_boot_lo, ari_boot_hi, color=GOLD, alpha=0.25, label="bootstrap ARI 10–90%")
    ax.plot(ks, ari_boot_mean, "s--", color=GOLD, lw=1.8, markersize=6, label="mean bootstrap ARI")
    ax.axvline(k_true, color=DEEP, ls=":", lw=1.5, label="true k")
    ax.set_xlabel("k-means k")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    style_ax(ax, "Recovery & stability peak near true k")
    ax.text(
        0.98, 0.92,
        "High ARI ≠ clinical phenotype.\nStability ≠ treatment effect.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Clustering: Adjusted Rand and bootstrap stability (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_adjusted_rand.png")


def fig_feature_hash_collisions():
    """Ch06: feature hashing — collision rate and linear model SNR vs hash dimension."""
    rng = np.random.default_rng(15)
    # V true sparse features; hash to m bins
    V = 5000
    # active features per example ~ Poisson
    n = 3000
    true_w = rng.normal(0, 1, size=V)
    true_w[rng.random(V) > 0.02] = 0  # sparse signal
    dims = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    # Expected collision fraction among nonzeros (approx birthday) for k active
    k_active = 40
    # Prob two distinct active features share a bin (approx)
    coll_rate = 1 - np.exp(-k_active * (k_active - 1) / (2 * dims))

    # Simulate signed hashing + ridge-ish linear recovery of R^2
    r2s = []
    for m in dims:
        # generate design via hashing
        Xh = np.zeros((n, m))
        y = np.zeros(n)
        for i in range(n):
            feats = rng.choice(V, size=k_active, replace=False)
            signs = rng.choice([-1.0, 1.0], size=k_active)
            bins = feats % m
            for f, s, b in zip(feats, signs, bins):
                Xh[i, b] += s
            y[i] = true_w[feats] @ np.ones(k_active) * 0.15 + rng.normal(0, 0.5)
        # least squares with ridge
        XtX = Xh.T @ Xh + 1e-2 * np.eye(m)
        beta = np.linalg.solve(XtX, Xh.T @ y)
        yhat = Xh @ beta
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot)
    r2s = np.array(r2s)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    ax.plot(dims, coll_rate, "o-", color=TEAL, lw=2.4, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hash dimension m")
    ax.set_ylabel(f"approx collision prob (k={k_active} actives)")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], fontsize=8, rotation=30)
    style_ax(ax, "Birthday collisions fall as m grows")
    ax.text(
        0.98, 0.92,
        "Signed hash mitigates bias;\ncollisions still mix signals.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )

    ax = axes[1]
    ax.plot(dims, r2s, "s-", color=GOLD, lw=2.4, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hash dimension m")
    ax.set_ylabel(r"in-sample $R^2$ of hashed linear model")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], fontsize=8, rotation=30)
    style_ax(ax, "Too-small m destroys recoverable signal")
    ax.text(
        0.98, 0.08,
        "Pick m from validation, not folklore.\nHashing ≠ causal feature selection.",
        transform=ax.transAxes, ha="right", fontsize=8, color="#64748b",
    )
    fig.suptitle("Feature hashing: collisions vs hash width (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_hash_collisions.png")


def fig_lda_vs_pca():
    """Ch07: LDA (supervised) vs PCA (unsupervised) projection on two classes."""
    rng = np.random.default_rng(5)
    n = 120
    # Classes elongated along a direction that is NOT the best separator
    mean0 = np.array([-0.6, 0.0])
    mean1 = np.array([0.6, 0.0])
    # Shared cov: large variance along y=x-ish diagonal
    R = np.array([[1.0, 0.85], [0.85, 1.0]])
    X0 = rng.multivariate_normal(mean0, R, size=n)
    X1 = rng.multivariate_normal(mean1, R, size=n)
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)

    # PCA first component
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    w_pca = Vt[0]
    # LDA (Fisher) for 2-class equal cov: Σ^{-1}(μ1-μ0)
    Sw = R  # known generative
    w_lda = np.linalg.solve(Sw, mean1 - mean0)
    w_lda = w_lda / np.linalg.norm(w_lda)
    w_pca = w_pca / np.linalg.norm(w_pca)

    def project(w):
        return X @ w

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    ax.scatter(X0[:, 0], X0[:, 1], s=16, c=TEAL, alpha=0.75, label="class 0")
    ax.scatter(X1[:, 0], X1[:, 1], s=16, c=GOLD, alpha=0.75, label="class 1")
    # Draw direction lines through origin of data mean
    mu = X.mean(axis=0)
    for w, col, lab in [(w_pca, DEEP, "PC1"), (w_lda, "#dc2626", "LDA")]:
        t = np.linspace(-3.5, 3.5, 50)
        line = mu + t[:, None] * w
        ax.plot(line[:, 0], line[:, 1], color=col, lw=2.2, label=lab)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Same data; different axes")
    ax.text(0.02, 0.05, "PCA follows variance;\nLDA follows class means",
            transform=ax.transAxes, fontsize=8, color="#64748b")

    ax = axes[1]
    # 1D histograms of projections
    bins = np.linspace(-4, 4, 28)
    p_pca0 = project(w_pca)[:n]
    p_pca1 = project(w_pca)[n:]
    p_lda0 = project(w_lda)[:n]
    p_lda1 = project(w_lda)[n:]
    ax.hist(p_pca0, bins=bins, density=True, alpha=0.45, color=TEAL, label="PCA · c0")
    ax.hist(p_pca1, bins=bins, density=True, alpha=0.45, color=GOLD, label="PCA · c1")
    ax.hist(p_lda0, bins=bins, density=True, histtype="step", linewidth=2.0, color=DEEP, label="LDA · c0")
    ax.hist(p_lda1, bins=bins, density=True, histtype="step", linewidth=2.0, color="#dc2626", label="LDA · c1")
    ax.set_xlabel("projected coordinate")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    style_ax(ax, "LDA separates; PC1 can mix classes")
    ax.text(
        0.98, 0.92,
        "Need labels for LDA.\nPCA is not a classifier.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("LDA vs PCA projection (synthetic two-class; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_lda_vs_pca.png")


def fig_rule_lift_frontier():
    """Ch05: association-rule support–confidence–lift frontier (synthetic market basket)."""
    rng = np.random.default_rng(2)
    # Simulate many itemset rules with noise
    n_rules = 200
    support = rng.beta(2, 12, size=n_rules) * 0.35 + 0.01
    # confidence correlated loosely with support but with noise
    confidence = np.clip(0.35 + 0.9 * support + rng.normal(0, 0.12, size=n_rules), 0.05, 0.98)
    # base rate of consequent
    p_y = 0.22
    lift = confidence / p_y
    # A few "interesting" high-lift low-support rules
    support = support.copy()
    confidence = confidence.copy()
    support[:8] = rng.uniform(0.02, 0.06, size=8)
    confidence[:8] = rng.uniform(0.55, 0.85, size=8)
    lift = confidence / p_y

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    sc = ax.scatter(support, confidence, c=lift, cmap="viridis", s=28, alpha=0.85, edgecolors="none")
    ax.axhline(0.5, color="#94a3b8", ls="--", lw=1.2, label="conf min 0.50")
    ax.axvline(0.05, color=GOLD, ls="--", lw=1.2, label="sup min 0.05")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="lift")
    ax.set_xlabel("support")
    ax.set_ylabel("confidence")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_ax(ax, "Rule cloud colored by lift")

    ax = axes[1]
    # Pareto-ish: for each support bin max lift
    order = np.argsort(support)
    # Keep rules with conf>=0.5
    mask = confidence >= 0.5
    ax.scatter(support[mask], lift[mask], s=30, c=TEAL, alpha=0.75, label="conf ≥ 0.5")
    ax.scatter(support[~mask], lift[~mask], s=20, c="#cbd5e1", alpha=0.7, label="conf < 0.5")
    ax.axhline(1.0, color="#dc2626", ls=":", lw=1.5, label="lift = 1 (independence)")
    ax.set_xlabel("support")
    ax.set_ylabel("lift = conf / P(consequent)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "High lift often lives at low support")
    ax.text(
        0.98, 0.92,
        "Mine for hypotheses, not causality.\nMultiple testing is fierce.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Association rules: support–confidence–lift frontier (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_lift_frontier.png")


def fig_aspect_ratio_slope():
    """Ch02: aspect-ratio / banking-to-45° teaching — same series, misleading slopes."""
    t = np.arange(0, 24)
    # seasonal-ish rate
    y = 12 + 0.15 * t + 1.8 * np.sin(2 * np.pi * t / 12) + 0.3 * np.cos(2 * np.pi * t / 6)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    ax.plot(t, y, "o-", color=TEAL, lw=2.3, markersize=5)
    ax.set_ylim(0, 40)  # flattened
    ax.set_xlabel("month index")
    ax.set_ylabel("event rate (per 100)")
    style_ax(ax, "Tall y-span → slopes look flat")
    ax.text(0.5, 0.08, "Same data as right panel", transform=ax.transAxes,
            ha="center", fontsize=8, color="#64748b")

    ax = axes[1]
    ax.plot(t, y, "o-", color=TEAL, lw=2.3, markersize=5)
    pad = 0.5
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel("month index")
    ax.set_ylabel("event rate (per 100)")
    style_ax(ax, "Tight y-span → slopes look dramatic")
    ax.text(
        0.98, 0.92,
        "Banking to 45° (Cleveland) is a\nhygiene default—not deception.\nReport absolute change too.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Aspect ratio changes perceived trend (same series; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_aspect_ratio.png")


def fig_bootstrap_auroc_ci():
    """Ch17: bootstrap CI for AUROC — optimism of a single point estimate."""
    rng = np.random.default_rng(19)
    n = 180
    y = rng.binomial(1, 0.28, size=n)
    score = 0.9 * y + rng.normal(0, 0.7, size=n)

    def auroc(y, s):
        # Mann-Whitney form
        pos = s[y == 1]
        neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # P(score_pos > score_neg) + 0.5 P(tie)
        # vectorized
        # compare all pairs
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        return (gt + 0.5 * eq) / (len(pos) * len(neg))

    point = auroc(y, score)
    B = 400
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boots.append(auroc(y[idx], score[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    ax.hist(boots, bins=28, color=TEAL, alpha=0.85, edgecolor="white", density=True)
    ax.axvline(point, color=GOLD, lw=2.2, label=f"point AUROC={point:.3f}")
    ax.axvline(lo, color="#dc2626", ls="--", lw=1.6, label=f"2.5%={lo:.3f}")
    ax.axvline(hi, color="#dc2626", ls="--", lw=1.6, label=f"97.5%={hi:.3f}")
    ax.set_xlabel("bootstrap AUROC")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, f"Percentile CI width ≈ {hi - lo:.3f}")

    ax = axes[1]
    # Show how CI shrinks with n (synthetic curve)
    ns = np.array([40, 60, 80, 120, 180, 300, 500])
    widths = []
    for nn in ns:
        ws = []
        for _ in range(30):
            yy = rng.binomial(1, 0.28, size=nn)
            ss = 0.9 * yy + rng.normal(0, 0.7, size=nn)
            bb = []
            for _ in range(120):
                idx = rng.integers(0, nn, size=nn)
                bb.append(auroc(yy[idx], ss[idx]))
            bb = np.array(bb)
            ws.append(np.percentile(bb, 97.5) - np.percentile(bb, 2.5))
        widths.append(np.mean(ws))
    ax.plot(ns, widths, "o-", color=TEAL, lw=2.4, markersize=7)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("mean bootstrap 95% CI width")
    style_ax(ax, "Small n → wide AUROC uncertainty")
    ax.text(
        0.98, 0.92,
        "External validation still required.\nCI ≠ transportability.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#64748b",
    )
    fig.suptitle("Bootstrap uncertainty for AUROC (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ml_fig_bootstrap_auroc.png")


def fig_umap_neighbors_caricature():
    """Ch07: neighbor-embedding n_neighbors caricature (local vs global structure)."""
    rng = np.random.default_rng(8)
    # Two moons-ish + a distant cluster
    n = 100
    theta = np.linspace(0, np.pi, n)
    moon1 = np.c_[np.cos(theta), np.sin(theta)] + rng.normal(0, 0.06, size=(n, 2))
    moon2 = np.c_[1 - np.cos(theta), 0.5 - np.sin(theta)] + rng.normal(0, 0.06, size=(n, 2))
    blob = rng.normal(loc=[3.5, 1.5], scale=0.2, size=(40, 2))
    X = np.vstack([moon1, moon2, blob])
    labels = np.array([0] * n + [1] * n + [2] * 40)

    def knn_graph_layout(X, k, steps=80, lr=0.05):
        # Toy force layout: attract kNN, mild repel — caricature of neighbor embeddings
        n = len(X)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        knn = np.argsort(d2, axis=1)[:, :k]
        Y = X[:, :2].copy() + rng.normal(0, 0.05, size=(n, 2))
        for _ in range(steps):
            grad = np.zeros_like(Y)
            # attract neighbors toward a small target distance
            for i in range(n):
                for j in knn[i]:
                    diff = Y[i] - Y[j]
                    dist = float(np.linalg.norm(diff)) + 1e-3
                    # spring: pull if far, mild push if too close
                    grad[i] += 0.4 * (dist - 0.35) * (diff / dist)
            # global repel sample
            for i in range(n):
                js = rng.choice(n, size=min(15, n - 1), replace=False)
                for j in js:
                    if i == j:
                        continue
                    diff = Y[i] - Y[j]
                    dist = float(np.linalg.norm(diff)) + 1e-3
                    grad[i] -= 0.08 * (diff / dist) / dist  # repel
            # clip gradient for stability
            gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-9
            grad = grad * np.minimum(1.0, 2.0 / gn)
            Y -= lr * grad
            Y -= Y.mean(axis=0)
            # soft bound
            Y = np.clip(Y, -6, 6)
        return Y

    Y_local = knn_graph_layout(X, k=5)
    Y_global = knn_graph_layout(X, k=45)

    cols = [TEAL, GOLD, DEEP]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    ax = axes[0]
    for j in range(3):
        m = labels == j
        ax.scatter(X[m, 0], X[m, 1], s=14, c=cols[j], alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    style_ax(ax, "Original (2D toy)")

    ax = axes[1]
    for j in range(3):
        m = labels == j
        ax.scatter(Y_local[m, 0], Y_local[m, 1], s=14, c=cols[j], alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    style_ax(ax, "Small n_neighbors (local)")
    ax.text(0.5, -0.12, "can tear global layout", transform=ax.transAxes,
            ha="center", fontsize=8, color="#64748b")

    ax = axes[2]
    for j in range(3):
        m = labels == j
        ax.scatter(Y_global[m, 0], Y_global[m, 1], s=14, c=cols[j], alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    style_ax(ax, "Large n_neighbors (global-ish)")
    ax.text(0.5, -0.12, "may blur fine clusters", transform=ax.transAxes,
            ha="center", fontsize=8, color="#64748b")

    fig.suptitle("Neighbor embeddings: n_neighbors trades local vs global (caricature; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "ml_fig_umap_neighbors.png")


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
    # Continuous densify cycle-5 (ch05 / ch11 / ch14 / ch15 / ch17)
    fig_vae_elbo_beta()
    fig_distill_temperature()
    fig_lora_rank()
    fig_minhash_jaccard()
    fig_betweenness_bridge()
    fig_spectral_fiedler()
    fig_multimetric_radar()
    # Continuous densify cycle-6 (preface / ch02 / ch04 / ch07 / ch13 / ch18)
    fig_claim_routing()
    fig_dual_axis_caution()
    fig_dbscan_density()
    fig_nmf_parts()
    fig_bellman_backup()
    fig_metric_decision_tree()
    # Continuous densify cycle-7 (preface / ch02 / ch04 / ch06 / ch07 / ch13)
    fig_external_ladder()
    fig_window_cherry()
    fig_dendrogram_cut()
    fig_target_enc_loo()
    fig_tsne_perplexity()
    fig_eligibility_trace()
    # Continuous densify cycle-8 (preface / ch11 / ch13 / ch17 / ch18 / ch09)
    fig_journal_club_card()
    fig_mae_mask_ratio()
    fig_policy_iteration_loop()
    fig_rollback_triggers()
    fig_leakage_taxonomy()
    fig_cost_threshold_utility()
    # Continuous densify cycle-9 (ch15 / ch14 / ch03 / ch10 / ch01 / ch12)
    fig_modularity_communities()
    fig_magnitude_prune()
    fig_lr_prevalence()
    fig_batchnorm_train_eval()
    fig_train_serve_skew()
    fig_oversmoothing_gcn()
    # Continuous densify cycle-10 (ch11 / ch08 / ch12 / ch16 / ch04 / ch09-shared)
    fig_ssl_probe_hygiene()
    fig_brier_components()
    fig_cindex_pairs()
    fig_positional_encoding_heatmap()
    fig_fairness_tradeoff()
    fig_adjusted_rand_stability()
    # Continuous densify cycle-11 (ch06 / ch07 / ch05 / ch02 / ch17)
    fig_feature_hash_collisions()
    fig_lda_vs_pca()
    fig_rule_lift_frontier()
    fig_aspect_ratio_slope()
    fig_bootstrap_auroc_ci()
    fig_umap_neighbors_caricature()
    print("DONE figures in", OUT)
    missing_legacy = [n for n in LEGACY_NUMBERED_ASSETS if not (OUT / n).exists()]
    if missing_legacy:
        print("WARN missing legacy assets:", missing_legacy)
    else:
        print("LEGACY numbered assets present:", len(LEGACY_NUMBERED_ASSETS))


if __name__ == "__main__":
    main()
