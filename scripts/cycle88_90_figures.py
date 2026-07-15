#!/usr/bin/env python3
"""Cycle-88/89/90: more novel scientific teal teaching figures (continuous densify)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch, Polygon

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


# Topics: each cycle maps 20 chapter-aligned novel concepts
TOPICS = {
    88: [
        ("dot_product", "Dot product as signed projection length"),
        ("irb_map", "Ethics/IRB map for human-data ML studies"),
        ("iid_vs_dep", "IID assumption vs dependent clinical episodes"),
        ("small_multiples", "Small multiples beat overstuffed single axes"),
        ("bootstrap", "Bootstrap resampling distribution of a statistic"),
        ("gmm", "GMM soft assignment as responsibility weights"),
        ("bm25", "BM25 score components: TF saturation + IDF"),
        ("woe", "Weight-of-evidence binning for sparse categories"),
        ("nmf", "NMF parts-based non-negative factors"),
        ("leverage", "Regression leverage / hat-matrix diagonal"),
        ("one_vs_rest", "One-vs-rest multilabel reduction sketch"),
        ("skip_conn", "Residual skip connection identity path"),
        ("byol", "BYOL online/target network bootstrap idea"),
        ("ctc", "CTC alignment lattice for speech (schematic)"),
        ("eligibility", "Eligibility trace decay in TD(λ)"),
        ("lora", "LoRA low-rank adapter update ΔW=BA"),
        ("betweenness", "Betweenness centrality on a bridge edge"),
        ("label_noise", "Label noise flips and decision boundary shift"),
        ("triage_matrix", "Impact × uncertainty triage for model use"),
        ("symbol_sheet", "Symbols: E, Var, Cov, KL, H(X)"),
    ],
    89: [
        ("condition_number", "Condition number: stretched level sets"),
        ("spirit_protocol", "SPIRIT-like protocol elements for ML papers"),
        ("no_free_lunch", "No free lunch: average over tasks is flat"),
        ("slopegraph", "Slopegraph for before/after group comparison"),
        ("power_curve", "Statistical power vs sample size curve"),
        ("hierarchical_k", "Hierarchical cut line → k clusters"),
        ("query_expansion", "Query expansion with synonyms/related terms"),
        ("time_features", "Cyclical time features sin/cos encodings"),
        ("sparse_coding", "Sparse coding: few active dictionary atoms"),
        ("collinearity", "Collinear predictors inflate coefficient SE"),
        ("cost_matrix", "Asymmetric misclassification cost matrix"),
        ("layer_norm", "LayerNorm across features per token"),
        ("dino", "Self-distillation teacher/student views"),
        ("bbox_iou", "IoU of predicted vs ground-truth boxes"),
        ("actor_critic", "Actor–critic dual network sketch"),
        ("weight_sharing", "Weight sharing / tied embeddings"),
        ("community_q", "Modularity Q contribution sketch"),
        ("censoring", "Right-censoring timeline in outcomes"),
        ("premortem", "Premortem failure modes board"),
        ("abbrev_strip", "Abbrev strip: CV, OOD, SSL, RL, GNN"),
    ],
    90: [
        ("jacobian", "Jacobian local linearization of f"),
        ("data_card", "Dataset card fields for transparency"),
        ("cross_val", "k-fold CV partition diagram"),
        ("raincloud", "Raincloud: density + jitter + box"),
        ("hypothesis_test", "Null sampling dist with critical region"),
        ("stability", "Cluster stability across bootstraps"),
        ("precision_at_k", "Precision@k curve for ranked retrieval"),
        ("target_leak_time", "Time-aware feature availability grid"),
        ("autoencoder", "Autoencoder bottleneck diagram"),
        ("influence", "Cook's distance influence markers"),
        ("platt", "Platt scaling sigmoid on scores"),
        ("dropout_mask", "Dropout random mask per forward pass"),
        ("mae_pretrain", "MAE high mask ratio reconstruction"),
        ("token_merge", "Vision/language token merge stages"),
        ("replay_buffer", "Experience replay buffer sampling"),
        ("int8_range", "INT8 dynamic range clipping"),
        ("shortest_path", "Shortest path tree from source s"),
        ("batch_effect", "Batch/site effect in feature space"),
        ("monitoring_kpi", "Post-deploy monitoring KPI dashboard"),
        ("formula_strip", "Key identities: Bayes, chain, bias-var"),
    ],
}


def draw_topic(ax, key: str, title: str):
    """Render a compact original teaching panel by topic key."""
    rng = np.random.default_rng(abs(hash(key)) % (2**32))

    if key in ("dot_product", "jacobian", "condition_number"):
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
        if key == "dot_product":
            ax.annotate("", xy=(1.5, 0.8), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
            ax.annotate("", xy=(1.2, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
            ax.plot([1.2, 1.2], [0, 0.64], color=ROSE, ls="--", lw=1.5)
            ax.text(1.55, 0.9, "a", color=TEAL, fontweight="bold")
            ax.text(1.25, -0.25, "b", color=GOLD, fontweight="bold")
            ax.text(0.2, -1.5, "a·b = ||a|| ||b|| cosθ", color=DEEP, fontsize=10, fontweight="bold")
        elif key == "jacobian":
            x = np.linspace(-1.5, 1.5, 20)
            y = np.linspace(-1.5, 1.5, 20)
            X, Y = np.meshgrid(x, y)
            U = 1 + 0.3 * X; V = 0.2 * Y
            ax.quiver(X, Y, U, V, color=TEAL, alpha=0.7)
            ax.plot(0, 0, "o", color=GOLD, ms=10)
            ax.text(-1.8, 1.7, "local linear map J_f", color=DEEP, fontsize=9, fontweight="bold")
        else:
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(2 * np.cos(t), 0.4 * np.sin(t), color=TEAL, lw=2)
            ax.plot(np.cos(t), np.sin(t), color=GOLD, lw=1.5, ls="--")
            ax.text(0, -1.7, "κ large → elongated level sets", ha="center", color=ROSE, fontsize=9)
        style(ax, title); ax.grid(True, alpha=0.25)
        return

    if key in ("irb_map", "spirit_protocol", "data_card", "premortem", "triage_matrix", "monitoring_kpi"):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        if key == "irb_map":
            items = [(0.4, 4, "consent"), (3.6, 4, "PHI min"), (6.8, 4, "purpose"),
                     (0.4, 1.5, "access"), (3.6, 1.5, "retention"), (6.8, 1.5, "share")]
            for x, y, t in items:
                box(ax, x, y, 2.6, 1.3, t, fc=TEAL, fs=10)
        elif key == "spirit_protocol":
            for i, t in enumerate(["aims", "design", "data", "analysis", "safety", "report"]):
                box(ax, 0.4 + (i % 3) * 3.1, 3.5 - (i // 3) * 2.5, 2.9, 1.5, t, fc=DEEP if i % 2 == 0 else TEAL, fs=10)
        elif key == "data_card":
            for i, t in enumerate(["provenance", "fields", "labels", "splits", "limits", "contact"]):
                box(ax, 0.4 + (i % 3) * 3.1, 3.5 - (i // 3) * 2.5, 2.9, 1.5, t,
                    fc=GOLD if i == 5 else TEAL, fs=10, tc=INK if i == 5 else "white")
        elif key == "premortem":
            for i, t in enumerate(["leakage", "shift", "label error", "metric mismatch", "workflow", "equity"]):
                box(ax, 0.4 + (i % 3) * 3.1, 3.5 - (i // 3) * 2.5, 2.9, 1.5, t, fc=ROSE if i < 2 else TEAL, fs=9)
        elif key == "triage_matrix":
            ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("on")
            ax.set_xticks([1, 2, 3, 4]); ax.set_yticks([1, 2, 3, 4])
            ax.set_xlabel("uncertainty"); ax.set_ylabel("clinical impact")
            ax.add_patch(Rectangle((0.5, 2.5), 2, 2, facecolor=ROSE, alpha=0.35))
            ax.add_patch(Rectangle((2.5, 2.5), 2, 2, facecolor=GOLD, alpha=0.35))
            ax.add_patch(Rectangle((0.5, 0.5), 2, 2, facecolor=TEAL, alpha=0.35))
            ax.add_patch(Rectangle((2.5, 0.5), 2, 2, facecolor=SLATE, alpha=0.25))
            ax.text(1.5, 3.5, "human\ngate", ha="center", fontsize=9, fontweight="bold")
            ax.text(3.5, 3.5, "caution", ha="center", fontsize=9, fontweight="bold")
            ax.text(1.5, 1.5, "auto\nOK?", ha="center", fontsize=9, fontweight="bold")
            ax.text(3.5, 1.5, "deprioritize", ha="center", fontsize=8)
        else:
            kpis = ["AUROC", "ECE", "alert rate", "override", "latency", "drift"]
            vals = [0.81, 0.06, 0.12, 0.18, 0.4, 0.22]
            ax.barh(kpis[::-1], vals[::-1], color=TEAL, edgecolor=DEEP)
            ax.set_xlim(0, 1); ax.axis("on")
            ax.grid(True, axis="x", alpha=0.3)
        style(ax, title)
        return

    if key in ("iid_vs_dep", "cross_val", "no_free_lunch", "label_noise", "batch_effect", "censoring"):
        if key == "iid_vs_dep":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            for i in range(6):
                box(ax, 0.3 + i * 1.55, 2.3, 1.4, 1.0, f"x{i}", fc=TEAL, fs=9)
            ax.text(5, 1.5, "IID: exchangeable rows", ha="center", color=DEEP, fontsize=9)
            for i in range(5):
                ax.annotate("", xy=(1.0 + (i + 1) * 1.55, 0.7), xytext=(1.0 + i * 1.55, 0.7),
                            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
            ax.text(5, 0.25, "dependent episodes: time/patient links", ha="center", color=ROSE, fontsize=9)
        elif key == "cross_val":
            folds = np.zeros((5, 10))
            for i in range(5):
                folds[i] = 1
                folds[i, i * 2:(i + 1) * 2] = 2
            ax.imshow(folds, cmap="YlGn", aspect="auto")
            ax.set_yticks(range(5)); ax.set_yticklabels([f"fold {i+1}" for i in range(5)])
            ax.set_xlabel("index order")
        elif key == "no_free_lunch":
            tasks = np.arange(12)
            ax.plot(tasks, 0.5 + 0.3 * np.sin(tasks), color=TEAL, lw=2, label="algo A")
            ax.plot(tasks, 0.5 + 0.3 * np.cos(tasks), color=GOLD, lw=2, label="algo B")
            ax.axhline(0.5, color=SLATE, ls="--", label="avg over all tasks")
            ax.legend(fontsize=8); ax.set_xlabel("task id"); ax.set_ylabel("risk")
            ax.grid(True, alpha=0.3)
        elif key == "label_noise":
            x = rng.normal(0, 1, (80, 2))
            y = (x[:, 0] + x[:, 1] > 0).astype(int)
            flip = rng.random(80) < 0.15
            y[flip] = 1 - y[flip]
            ax.scatter(x[y == 0, 0], x[y == 0, 1], c=TEAL, s=18, label="y=0")
            ax.scatter(x[y == 1, 0], x[y == 1, 1], c=GOLD, s=18, label="y=1")
            ax.scatter(x[flip, 0], x[flip, 1], facecolors="none", edgecolors=ROSE, s=80, label="flipped")
            ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        elif key == "batch_effect":
            a = rng.normal([0, 0], 0.5, (60, 2))
            b = rng.normal([2.2, 0.3], 0.5, (60, 2))
            ax.scatter(a[:, 0], a[:, 1], c=TEAL, s=16, label="site A")
            ax.scatter(b[:, 0], b[:, 1], c=GOLD, s=16, label="site B")
            ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        else:
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            for i, (lab, end) in enumerate([("event@6", 6), ("cens@8", 8), ("event@4", 4)]):
                y = 3 - i * 1.0
                ax.plot([0.5, end], [y, y], color=TEAL, lw=3)
                ax.plot(end, y, "o" if "event" in lab else ">", color=ROSE if "event" in lab else GOLD, ms=10)
                ax.text(end + 0.2, y, lab, va="center", fontsize=8)
        style(ax, title)
        return

    if key in ("small_multiples", "slopegraph", "raincloud", "power_curve", "bootstrap", "hypothesis_test"):
        if key == "small_multiples":
            fig = ax.figure
            ax.remove()
            axes = fig.subplots(2, 2)
            for a, c in zip(axes.ravel(), [TEAL, GOLD, DEEP, ROSE]):
                a.plot(np.linspace(0, 1, 20), rng.random(20).cumsum(), color=c, lw=1.5)
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_color("#cbd5e1")
            fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
            return
        if key == "slopegraph":
            for i, (a, b) in enumerate([(30, 45), (50, 40), (20, 55), (60, 62)]):
                ax.plot([0, 1], [a, b], "o-", color=TEAL, lw=2, ms=8)
                ax.text(-0.05, a, f"g{i}", ha="right", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["before", "after"])
            ax.set_xlim(-0.3, 1.3); ax.grid(True, axis="y", alpha=0.3)
        elif key == "raincloud":
            for i, mu in enumerate([0, 1.2, 0.4]):
                d = rng.normal(mu, 0.4, 40)
                ax.scatter(np.full(40, i) + rng.uniform(-0.08, 0.08, 40), d, c=TEAL, s=12, alpha=0.6)
                ax.boxplot([d], positions=[i], widths=0.25, patch_artist=True,
                           boxprops=dict(facecolor=SOFT), medianprops=dict(color=GOLD))
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["A", "B", "C"])
        elif key == "power_curve":
            n = np.linspace(10, 400, 50)
            power = 1 - np.exp(-n / 120)
            ax.plot(n, power, color=TEAL, lw=2.2)
            ax.axhline(0.8, color=GOLD, ls="--", label="80% power")
            ax.legend(fontsize=8); ax.set_xlabel("n"); ax.set_ylabel("power"); ax.grid(True, alpha=0.3)
        elif key == "bootstrap":
            raw = rng.exponential(1.0, 80)
            means = [rng.choice(raw, size=80, replace=True).mean() for _ in range(200)]
            ax.hist(means, bins=20, color=TEAL, edgecolor=DEEP, alpha=0.9)
            ax.axvline(np.mean(raw), color=GOLD, lw=2, label="point estimate")
            ax.legend(fontsize=8); ax.set_xlabel("bootstrap mean")
        else:
            z = np.linspace(-4, 4, 300)
            dens = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            ax.plot(z, dens, color=TEAL, lw=2)
            ax.fill_between(z[z > 1.96], dens[z > 1.96], color=ROSE, alpha=0.5, label="critical α")
            ax.legend(fontsize=8); ax.set_xlabel("test statistic"); ax.grid(True, alpha=0.3)
        style(ax, title)
        return

    if key in ("gmm", "hierarchical_k", "stability", "nmf", "sparse_coding", "autoencoder"):
        if key == "gmm":
            x = np.linspace(-4, 4, 200)
            y = 0.5 * np.exp(-0.5 * (x + 1.5) ** 2) + 0.5 * np.exp(-0.5 * (x - 1.2) ** 2)
            ax.plot(x, y, color=TEAL, lw=2.2)
            ax.fill_between(x, 0.5 * np.exp(-0.5 * (x + 1.5) ** 2), alpha=0.3, color=GOLD)
            ax.fill_between(x, 0.5 * np.exp(-0.5 * (x - 1.2) ** 2), alpha=0.3, color=DEEP)
            ax.set_xlabel("x"); ax.set_ylabel("density mix")
        elif key == "hierarchical_k":
            ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.axis("off")
            ax.plot([1, 1, 2, 2], [0, 1, 1, 0], TEAL, lw=2)
            ax.plot([1.5, 1.5, 3.5, 3.5], [1, 2.2, 2.2, 0], DEEP, lw=2)
            ax.plot([4, 4, 5, 5], [0, 1.2, 1.2, 0], GOLD, lw=2)
            ax.plot([2.5, 2.5, 4.5, 4.5], [2.2, 3.2, 3.2, 1.2], ROSE, lw=2)
            ax.axhline(1.6, color=INK, ls="--", lw=1.5)
            ax.text(5.2, 1.7, "cut → k", fontsize=9, color=INK, fontweight="bold")
        elif key == "stability":
            k = np.arange(2, 9)
            stab = np.array([0.4, 0.7, 0.85, 0.8, 0.6, 0.5, 0.45])
            ax.plot(k, stab, "o-", color=TEAL, lw=2)
            ax.set_xlabel("k"); ax.set_ylabel("stability"); ax.grid(True, alpha=0.3)
        elif key == "nmf":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            box(ax, 0.4, 1.2, 2.2, 1.8, "V≥0", fc=TEAL, fs=11)
            ax.text(3.0, 2.0, "≈", fontsize=16, ha="center")
            box(ax, 3.5, 1.2, 2.2, 1.8, "W≥0", fc=DEEP, fs=11)
            box(ax, 6.2, 1.2, 2.2, 1.8, "H≥0", fc=GOLD, tc=INK, fs=11)
        elif key == "sparse_coding":
            atoms = np.eye(6)
            coef = np.array([0.9, 0, 0.4, 0, 0, 0.2])
            ax.bar(range(6), coef, color=TEAL, edgecolor=DEEP)
            ax.set_xlabel("dictionary atom"); ax.set_ylabel("coefficient")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            box(ax, 0.4, 1.2, 2.0, 1.8, "x", fc=TEAL, fs=12)
            box(ax, 3.2, 1.5, 1.6, 1.2, "enc", fc=DEEP, fs=10)
            box(ax, 5.4, 1.7, 1.2, 0.8, "z", fc=ROSE, fs=11)
            box(ax, 7.2, 1.5, 1.6, 1.2, "dec", fc=GOLD, tc=INK, fs=10)
            ax.annotate("", xy=(3.2, 2.1), xytext=(2.4, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
            ax.annotate("", xy=(5.4, 2.1), xytext=(4.8, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
            ax.annotate("", xy=(7.2, 2.1), xytext=(6.6, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        style(ax, title)
        return

    if key in ("bm25", "query_expansion", "precision_at_k", "woe", "time_features", "target_leak_time"):
        if key == "bm25":
            tf = np.linspace(0, 20, 50)
            sat = tf / (1.2 + tf)
            ax.plot(tf, sat, color=TEAL, lw=2.2, label="TF saturation")
            ax.plot(tf, np.minimum(1, tf / 10), color=GOLD, ls="--", label="raw TF (clip)")
            ax.legend(fontsize=8); ax.set_xlabel("term freq"); ax.grid(True, alpha=0.3)
        elif key == "query_expansion":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            box(ax, 0.5, 1.4, 2.5, 1.4, "query", fc=TEAL, fs=10)
            box(ax, 3.7, 1.4, 2.5, 1.4, "+synonyms", fc=GOLD, tc=INK, fs=10)
            box(ax, 6.9, 1.4, 2.5, 1.4, "expanded\nquery", fc=DEEP, fs=10)
            ax.annotate("", xy=(3.7, 2.1), xytext=(3.0, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
            ax.annotate("", xy=(6.9, 2.1), xytext=(6.2, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
        elif key == "precision_at_k":
            k = np.arange(1, 21)
            p = np.maximum(0.2, 0.95 - 0.03 * k + 0.02 * np.sin(k))
            ax.plot(k, p, "o-", color=TEAL, lw=2)
            ax.set_xlabel("k"); ax.set_ylabel("precision@k"); ax.grid(True, alpha=0.3)
        elif key == "woe":
            bins = np.arange(5)
            woe = np.array([0.8, 0.3, -0.1, -0.5, -0.9])
            ax.bar(bins, woe, color=[TEAL if v > 0 else ROSE for v in woe], edgecolor=DEEP)
            ax.axhline(0, color=INK, lw=1); ax.set_xlabel("bin"); ax.set_ylabel("WoE")
        elif key == "time_features":
            h = np.arange(0, 24)
            ax.plot(h, np.sin(2 * np.pi * h / 24), color=TEAL, lw=2, label="sin hour")
            ax.plot(h, np.cos(2 * np.pi * h / 24), color=GOLD, lw=2, label="cos hour")
            ax.legend(fontsize=8); ax.set_xlabel("hour"); ax.grid(True, alpha=0.3)
        else:
            feats = ["NIHSS@t0", "labs@t0", "mRS@90d", "LOS"]
            avail = [1, 1, 0, 0]
            ax.barh(feats[::-1], [1] * 4, color=SOFT, edgecolor=SLATE)
            ax.barh(feats[::-1], avail[::-1], color=TEAL)
            ax.set_xlim(0, 1.2); ax.set_xlabel("available at prediction time")
            ax.text(0.5, -0.8, "red-zone features = leakage if used early", ha="center",
                    transform=ax.get_xaxis_transform(), color=ROSE, fontsize=9, fontweight="bold")
        style(ax, title)
        return

    if key in ("leverage", "collinearity", "influence", "one_vs_rest", "cost_matrix", "platt"):
        if key == "leverage":
            x = np.linspace(-2, 2, 30)
            y = 0.5 * x + rng.normal(0, 0.3, 30)
            x = np.append(x, 3.5); y = np.append(y, 1.2)
            lev = np.abs(x - x.mean())
            ax.scatter(x, y, c=lev, cmap="viridis", s=40)
            ax.plot(3.5, 1.2, "o", mfc="none", mec=ROSE, ms=14, mew=2)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True, alpha=0.3)
        elif key == "collinearity":
            xs = rng.normal(0, 1, 80)
            ax.scatter(xs, xs + rng.normal(0, 0.15, 80), c=TEAL, s=16, alpha=0.7)
            ax.set_xlabel("x1"); ax.set_ylabel("x2 ≈ x1"); ax.grid(True, alpha=0.3)
        elif key == "influence":
            x = np.linspace(0, 10, 25)
            y = 2 + 0.5 * x + rng.normal(0, 0.4, 25)
            y[20] = 12
            ax.scatter(x, y, c=TEAL, s=28)
            ax.plot(x[20], y[20], "o", mfc="none", mec=ROSE, ms=14, mew=2)
            coef = np.polyfit(x, y, 1)
            ax.plot(x, coef[0] * x + coef[1], color=GOLD, lw=2)
            ax.grid(True, alpha=0.3)
        elif key == "one_vs_rest":
            ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
            box(ax, 0.4, 3.2, 2.4, 1.2, "labels\nA,B,C", fc=SLATE, fs=9)
            for i, lab in enumerate(["A vs rest", "B vs rest", "C vs rest"]):
                box(ax, 3.6, 3.6 - i * 1.3, 2.8, 1.0, lab, fc=TEAL, fs=9)
            box(ax, 7.2, 1.8, 2.4, 1.4, "argmax\nscores", fc=GOLD, tc=INK, fs=9)
        elif key == "cost_matrix":
            cm = np.array([[0, 1], [5, 0]])
            ax.imshow(cm, cmap="YlOrRd")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred−", "pred+"]); ax.set_yticklabels(["true−", "true+"])
        else:
            s = np.linspace(-4, 4, 100)
            ax.plot(s, 1 / (1 + np.exp(-(1.2 * s - 0.3))), color=TEAL, lw=2.2, label="Platt σ(as+b)")
            ax.plot(s, 1 / (1 + np.exp(-s)), color=GOLD, ls="--", label="raw logistic")
            ax.legend(fontsize=8); ax.set_xlabel("score"); ax.set_ylabel("P(y=1)"); ax.grid(True, alpha=0.3)
        style(ax, title)
        return

    if key in ("skip_conn", "layer_norm", "dropout_mask", "byol", "dino", "mae_pretrain"):
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
        if key == "skip_conn":
            box(ax, 0.5, 1.8, 2.2, 1.4, "x", fc=TEAL, fs=11)
            box(ax, 3.5, 1.8, 2.4, 1.4, "F(x)", fc=DEEP, fs=11)
            box(ax, 7.0, 1.8, 2.4, 1.4, "x+F(x)", fc=GOLD, tc=INK, fs=11)
            ax.annotate("", xy=(3.5, 2.5), xytext=(2.7, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
            ax.annotate("", xy=(7.0, 2.5), xytext=(5.9, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
            ax.annotate("", xy=(8.0, 3.6), xytext=(1.6, 3.6),
                        arrowprops=dict(arrowstyle="->", color=ROSE, lw=2, connectionstyle="arc3,rad=-0.2"))
            ax.text(4.5, 4.2, "identity skip", color=ROSE, fontsize=9, fontweight="bold")
        elif key == "layer_norm":
            box(ax, 1, 1.5, 8, 2.2, "normalize across feature dim per token\n(μ,σ within token; then γ,β)", fc=TEAL, fs=11)
        elif key == "dropout_mask":
            m = rng.random((4, 8)) > 0.3
            ax.imshow(m, cmap="Greens", aspect="auto")
            ax.axis("on"); ax.set_xlabel("unit"); ax.set_ylabel("example")
        elif key == "byol":
            box(ax, 0.5, 3.0, 3.0, 1.4, "online net", fc=TEAL, fs=10)
            box(ax, 6.5, 3.0, 3.0, 1.4, "target net", fc=GOLD, tc=INK, fs=10)
            box(ax, 3.5, 0.8, 3.0, 1.4, "predict → stop-grad", fc=DEEP, fs=9)
            ax.annotate("", xy=(6.5, 3.7), xytext=(3.5, 3.7), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
            ax.text(5, 4.4, "EMA", ha="center", fontsize=9, color=SLATE)
        elif key == "dino":
            box(ax, 0.5, 1.8, 4.0, 1.6, "student views", fc=TEAL, fs=11)
            box(ax, 5.5, 1.8, 4.0, 1.6, "teacher views (EMA)", fc=GOLD, tc=INK, fs=11)
            ax.text(5, 0.7, "self-distill match distributions", ha="center", color=DEEP, fontsize=10, fontweight="bold")
        else:
            grid = np.ones((4, 4))
            mask = rng.random((4, 4)) < 0.75
            grid[mask] = 0
            ax.imshow(grid, cmap="Greens", vmin=0, vmax=1)
            ax.axis("on")
            ax.set_title("")  # style adds
            ax.text(1.5, -0.7, "high mask ratio patches", ha="center", color=ROSE, fontsize=9)
        style(ax, title)
        return

    if key in ("ctc", "bbox_iou", "token_merge", "actor_critic", "eligibility", "replay_buffer"):
        if key == "ctc":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            for i, t in enumerate(["-", "c", "a", "-", "t", "-"]):
                box(ax, 0.4 + i * 1.55, 2.0, 1.4, 1.2, t, fc=TEAL if t != "-" else SLATE, fs=11)
            ax.text(5, 0.8, "align audio frames to labels with blanks", ha="center", color=DEEP, fontsize=10)
        elif key == "bbox_iou":
            ax.add_patch(Rectangle((1, 1), 4, 3, fill=False, edgecolor=TEAL, lw=2, label="GT"))
            ax.add_patch(Rectangle((3, 2), 4, 3, fill=False, edgecolor=GOLD, lw=2, label="pred"))
            ax.add_patch(Rectangle((3, 2), 2, 2, facecolor=ROSE, alpha=0.3))
            ax.set_xlim(0, 8); ax.set_ylim(0, 6); ax.set_aspect("equal")
            ax.legend(fontsize=8); ax.text(4, 0.3, "IoU = inter / union", ha="center", fontsize=10, color=DEEP)
        elif key == "token_merge":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            for i in range(8):
                box(ax, 0.3 + i * 1.15, 2.4, 1.0, 0.9, str(i), fc=TEAL, fs=8)
            for i in range(4):
                box(ax, 1.0 + i * 2.0, 0.8, 1.5, 0.9, f"m{i}", fc=GOLD, tc=INK, fs=9)
            ax.text(5, 3.6, "merge similar tokens → fewer", ha="center", color=DEEP, fontsize=10)
        elif key == "actor_critic":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            box(ax, 0.5, 1.4, 3.5, 1.6, "Actor π(a|s)", fc=TEAL, fs=11)
            box(ax, 6.0, 1.4, 3.5, 1.6, "Critic V(s)/Q", fc=GOLD, tc=INK, fs=11)
            ax.text(5, 0.5, "advantage A = return − baseline", ha="center", color=DEEP, fontsize=10, fontweight="bold")
        elif key == "eligibility":
            t = np.arange(0, 20)
            ax.bar(t, 0.9 ** t, color=TEAL, edgecolor=DEEP)
            ax.set_xlabel("steps since visit"); ax.set_ylabel("eligibility e")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
            box(ax, 1, 1, 8, 3, "replay buffer D\nsample minibatch (s,a,r,s′)", fc=TEAL, fs=12)
        style(ax, title)
        return

    if key in ("lora", "weight_sharing", "int8_range", "betweenness", "community_q", "shortest_path"):
        if key == "lora":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            box(ax, 0.4, 1.3, 2.5, 1.6, "W frozen", fc=SLATE, fs=10)
            box(ax, 3.5, 1.3, 2.5, 1.6, "B A\nlow-rank", fc=TEAL, fs=10)
            box(ax, 6.8, 1.3, 2.7, 1.6, "W+BA", fc=GOLD, tc=INK, fs=11)
            ax.annotate("", xy=(3.5, 2.1), xytext=(2.9, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
            ax.annotate("", xy=(6.8, 2.1), xytext=(6.0, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
        elif key == "weight_sharing":
            ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
            box(ax, 1, 2.2, 3, 1.3, "embed E", fc=TEAL, fs=11)
            box(ax, 6, 2.2, 3, 1.3, "tied Eᵀ", fc=TEAL, fs=11)
            ax.annotate("", xy=(6, 2.8), xytext=(4, 2.8), arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2))
            ax.text(5, 1.2, "shared parameters", ha="center", color=DEEP, fontsize=10)
        elif key == "int8_range":
            x = np.linspace(-2, 2, 200)
            y = np.tanh(x)
            q = np.clip(np.round(y * 127), -128, 127) / 127
            ax.plot(x, y, color=TEAL, lw=2, label="fp")
            ax.plot(x, q, color=GOLD, lw=1.5, label="int8 grid")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        elif key == "betweenness":
            pos = {0: (0, 1), 1: (1, 2), 2: (1, 0), 3: (2, 1), 4: (3, 2), 5: (3, 0)}
            edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]
            for i, j in edges:
                c = ROSE if (i, j) == (1, 3) or (i, j) == (2, 3) else SLATE
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=c, lw=2 if c == ROSE else 1.5)
            for n, (x, y) in pos.items():
                ax.plot(x, y, "o", color=TEAL if n != 3 else GOLD, ms=14)
            ax.axis("off"); ax.set_aspect("equal")
        elif key == "community_q":
            ax.bar(["within", "between"], [0.7, 0.2], color=[TEAL, ROSE], edgecolor=DEEP)
            ax.set_ylabel("edge fraction"); ax.grid(True, axis="y", alpha=0.3)
        else:
            pos = {0: (0, 1), 1: (1, 2), 2: (1, 0), 3: (2, 1), 4: (3, 1)}
            edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
            path = {(0, 1), (1, 3), (3, 4)}
            for i, j in edges:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color=GOLD if (i, j) in path else SLATE, lw=2.5 if (i, j) in path else 1.2)
            for n, (x, y) in pos.items():
                ax.plot(x, y, "o", color=TEAL, ms=12)
                ax.text(x, y - 0.25, str(n), ha="center", fontsize=8)
            ax.axis("off")
        style(ax, title)
        return

    # glossary strips and formula sheets
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    if key == "symbol_sheet":
        for i, t in enumerate(["E[X]", "Var", "Cov", "KL", "H(X)"]):
            box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, t, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    elif key == "abbrev_strip":
        for i, t in enumerate(["CV", "OOD", "SSL", "RL", "GNN"]):
            box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, t, fc=GOLD if i % 2 else TEAL, fs=12,
                tc=INK if i % 2 else "white")
    elif key == "formula_strip":
        for i, t in enumerate(["Bayes", "chain\nrule", "bias-\nvar", "MLE", "MAP"]):
            box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, t, fc=TEAL if i % 2 == 0 else MINT, fs=10,
                tc="white" if i % 2 == 0 else INK)
    else:
        box(ax, 2, 1.2, 8, 1.6, key.replace("_", " "), fc=TEAL, fs=12)
    style(ax, title)


def gen_cycle(cycle: int):
    topics = TOPICS[cycle]
    for i, (key, title) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        draw_topic(ax, key, title)
        # if draw_topic removed ax for small multiples, figure already titled
        if fig.axes:
            pass
        save(fig, f"ml_fig_c{cycle}_{i:02d}.png")
    # embed
    n = 0
    for i, ch in enumerate(CHAPTERS):
        p = CURR / ch
        fig = f"ml_fig_c{cycle}_{i:02d}.png"
        cap = topics[i][1]
        block = (
            f"\n![c{cycle} teaching panel {i:02d} (original).](../assets/figures/{fig})\n"
            f"*Figure — {cap}. Synthetic teaching geometry—not a causal claim.*\n"
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
    print(f"EMBEDDED cycle {cycle}: {n}")


def main(cycles=None):
    for c in (cycles or [88, 89, 90]):
        gen_cycle(c)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main([int(x) for x in sys.argv[1].split(",")])
    else:
        main()
