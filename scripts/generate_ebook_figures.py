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
    print("DONE figures in", OUT)


if __name__ == "__main__":
    main()
