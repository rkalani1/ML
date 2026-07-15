#!/usr/bin/env python3
"""Cycle-91/92/93: novel scientific teal teaching figures (continuous densify)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
TEAL, DEEP, INK, GOLD, SLATE, ROSE, MINT, SOFT = (
    "#0d9488", "#0f766e", "#0f172a", "#c9a227", "#64748b", "#e11d48", "#14b8a6", "#ecfeff"
)
CHS = sorted(p.name for p in CURR.glob("*.md"))

C91 = [
    ("Eigenvectors of a 2x2 stretch", "eig"),
    ("CONSORT-like flow for ML cohort", "consort"),
    ("Empirical risk vs true risk gap", "erm"),
    ("Violin plots for distribution shape", "violin"),
    ("p-value as tail probability under H0", "pval"),
    ("Spectral clustering Laplacian idea", "spectral"),
    ("n-gram language model backoff", "ngram"),
    ("Binning vs keep-continuous tradeoff", "bin"),
    ("Random projection Johnson-Lindenstrauss", "rp"),
    ("Huber loss vs squared loss", "huber"),
    ("Margin geometry for linear separator", "margin"),
    ("Learning rate schedules", "lr"),
    ("MoCo queue of negatives", "moco"),
    ("Beam search vs greedy decode", "beam"),
    ("Softmax policy in RL", "softp"),
    ("Knowledge distillation temperature", "kd"),
    ("Louvain community agglomeration steps", "louvain"),
    ("Train/serve skew checklist", "skew"),
    ("Silent failure modes after deploy", "silent"),
    ("Acronym strip: AUC ECE Brier NLL", "acr"),
]
C92 = [
    ("Matrix rank as dimension of column space", "rank"),
    ("TRIPOD reporting items sketch", "tripod"),
    ("PAC learning schematic bounds", "pac"),
    ("Alluvial-like cohort flow", "alluvial"),
    ("Multiple testing Bonferroni wall", "bonf"),
    ("Core-set / landmark sampling", "coreset"),
    ("Phrase query proximity bonus", "prox"),
    ("Missing indicator features", "missind"),
    ("Kernel PCA feature map idea", "kpca"),
    ("Quantile regression pinball loss", "pinball"),
    ("Isotonic calibration stepwise", "isotonic"),
    ("Adam moment estimates sketch", "adam"),
    ("CLIP image-text contrastive space", "clip"),
    ("Positional encoding sinusoids", "posenc"),
    ("Advantage normalization", "adv"),
    ("Flash attention tiling idea", "flash"),
    ("Node2vec walk bias p/q", "n2v"),
    ("Synthetic data privacy tradeoff", "synth"),
    ("Champion/challenger model gate", "champ"),
    ("Greek strip: alpha beta gamma lambda", "greek"),
]
C93 = [
    ("SVD singular values decay", "svdecay"),
    ("Preanalysis plan version control", "pap"),
    ("Occams razor capacity control", "occam"),
    ("Bland-Altman agreement plot", "ba"),
    ("Likelihood ratio test idea", "lrt"),
    ("HDBSCAN density hierarchy", "hdb"),
    ("Learning to rank pairwise hinge", "ltr"),
    ("Entity embeddings for categoricals", "entemb"),
    ("UMAP neighbor graph sketch", "umapg"),
    ("GAM smooth component stack", "gam"),
    ("Expected calibration error bins", "ece"),
    ("Gradient clipping by global norm", "clipg"),
    ("JEPA predict representation not pixels", "jepa"),
    ("Cross-attention query-key-value", "xattn"),
    ("Intrinsic motivation bonus", "intrin"),
    ("Speculative decoding draft-verify", "spec"),
    ("Graph attention coefficients", "gat"),
    ("Differential privacy noise scale", "dp"),
    ("Shadow deployment traffic split", "shadow"),
    ("Inequality strip: Jensen, Hoeffding", "ineq"),
]


def save(fig, n):
    fig.savefig(OUT / n, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", n)


def box(ax, x, y, w, h, t, fc=TEAL, fs=9, tc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc, edgecolor="none",
        )
    )
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def draw(ax, kind, title, seed):
    rng = np.random.default_rng(seed)
    if kind in ("eig", "rank", "svdecay", "rp", "kpca", "umapg"):
        if kind == "eig":
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(1.8 * np.cos(t), 0.6 * np.sin(t), color=TEAL, lw=2)
            ax.annotate("", xy=(1.8, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
            ax.annotate("", xy=(0, 0.6), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        elif kind == "rank":
            A = np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1]])
            ax.imshow(A, cmap="Greens")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(1, -0.7, "rank-deficient rows proportional", ha="center", color=ROSE, fontsize=9)
        elif kind == "svdecay":
            s = np.array([5, 3.2, 1.1, 0.4, 0.15, 0.05, 0.02, 0.01])
            ax.semilogy(range(1, 9), s, "o-", color=TEAL, lw=2)
            ax.set_xlabel("i")
            ax.set_ylabel("σ_i")
            ax.grid(True, alpha=0.3)
        elif kind == "rp":
            X = rng.normal(0, 1, (80, 2))
            R = rng.normal(0, 1, (2, 2))
            Y = X @ R
            ax.scatter(X[:, 0], X[:, 1], c=SLATE, s=10, alpha=0.5, label="orig")
            ax.scatter(Y[:, 0], Y[:, 1], c=TEAL, s=10, alpha=0.7, label="projected")
            ax.legend(fontsize=7)
            ax.set_aspect("equal")
        elif kind == "kpca":
            t = np.linspace(0, 2 * np.pi, 100)
            ax.scatter(np.cos(t), np.sin(t), c=TEAL, s=12)
            ax.scatter(0.3 * np.cos(t), 0.3 * np.sin(t), c=GOLD, s=12)
            ax.set_aspect("equal")
        else:
            for _ in range(30):
                a, b = rng.integers(0, 20, 2)
                ax.plot([a, b], [rng.random(), rng.random()], color=TEAL, alpha=0.3)
            ax.scatter(rng.random(20) * 20, rng.random(20), c=GOLD, s=20)
        style(ax, title)
        return

    if kind in ("consort", "tripod", "pap", "erm", "pac", "occam", "champ", "shadow", "silent", "skew", "alluvial", "gam", "isotonic", "ece"):
        if kind == "consort":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis("off")
            box(ax, 3, 4.5, 4, 1, "assessed N", fc=TEAL, fs=10)
            box(ax, 1, 2.5, 3, 1, "excluded", fc=ROSE, fs=10)
            box(ax, 6, 2.5, 3, 1, "analyzed", fc=GOLD, tc=INK, fs=10)
            ax.annotate("", xy=(2.5, 3.5), xytext=(5, 4.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
            ax.annotate("", xy=(7.5, 3.5), xytext=(5, 4.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        elif kind in ("tripod", "pap"):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis("off")
            labs = (
                ["title", "abstract", "methods", "results", "limits", "code"]
                if kind == "tripod"
                else ["v0.1 plan", "amend1", "freeze", "analyze", "report", "archive"]
            )
            for i, l in enumerate(labs):
                box(ax, 0.4 + (i % 3) * 3.1, 3.5 - (i // 3) * 2.5, 2.9, 1.5, l, fc=TEAL if i % 2 == 0 else DEEP, fs=9)
        elif kind == "alluvial":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 5)
            ax.axis("off")
            box(ax, 0.4, 1.5, 2.5, 2.2, "raw", fc=SLATE, fs=11)
            box(ax, 3.7, 1.8, 2.5, 1.6, "eligible", fc=TEAL, fs=11)
            box(ax, 7.0, 2.0, 2.5, 1.2, "analytic", fc=GOLD, tc=INK, fs=11)
        elif kind == "erm":
            n = np.linspace(20, 500, 40)
            ax.plot(n, 0.2 + 0.5 / np.sqrt(n), color=TEAL, lw=2, label="gen gap bound sketch")
            ax.plot(n, 0.15 + 0.02 * np.sin(n / 30), color=GOLD, lw=2, label="empirical risk")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("n")
        elif kind == "pac":
            eps = np.linspace(0.05, 0.5, 40)
            ax.plot(eps, 1 / eps ** 2, color=TEAL, lw=2)
            ax.set_xlabel("ε")
            ax.set_ylabel("n needed ~")
            ax.grid(True, alpha=0.3)
        elif kind == "occam":
            c = np.linspace(1, 10, 40)
            ax.plot(c, 2 / c + 0.05 * c, color=TEAL, lw=2)
            ax.set_xlabel("capacity")
            ax.set_ylabel("risk")
            ax.grid(True, alpha=0.3)
        elif kind == "champ":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 5)
            ax.axis("off")
            box(ax, 0.5, 1.5, 4, 2, "champion\nin prod", fc=TEAL, fs=11)
            box(ax, 5.5, 1.5, 4, 2, "challenger\nmust beat gates", fc=GOLD, tc=INK, fs=11)
        elif kind == "shadow":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 5)
            ax.axis("off")
            box(ax, 0.5, 1.5, 2.5, 2, "traffic", fc=SLATE, fs=10)
            box(ax, 4, 1.5, 2.5, 2, "prod", fc=TEAL, fs=10)
            box(ax, 7.2, 1.5, 2.5, 2, "shadow\nlog only", fc=GOLD, tc=INK, fs=10)
            ax.annotate("", xy=(4, 2.5), xytext=(3, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
            ax.annotate("", xy=(7.2, 2.5), xytext=(3, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
        elif kind == "silent":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis("off")
            for i, t in enumerate(["metric OK", "workflow break", "alert fatigue", "label drift"]):
                box(ax, 0.4 + (i % 2) * 4.8, 3.5 - (i // 2) * 2.5, 4.4, 1.8, t, fc=ROSE if i else TEAL, fs=10)
        elif kind == "skew":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 4)
            ax.axis("off")
            for i, t in enumerate(["schema", "defaults", "preproc", "feature store"]):
                box(ax, 0.4 + i * 2.35, 1.2, 2.2, 1.6, t, fc=TEAL if i % 2 == 0 else DEEP, fs=9)
        elif kind == "gam":
            x = np.linspace(0, 10, 100)
            ax.plot(x, np.sin(x) + 0.3 * x, color=TEAL, lw=2, label="f1+f2+…")
            ax.plot(x, np.sin(x), color=GOLD, ls="--", label="smooth f1")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        elif kind == "isotonic":
            x = np.sort(rng.uniform(0, 1, 15))
            y = np.minimum.accumulate(np.sort(rng.uniform(0, 1, 15))[::-1])[::-1]
            y = np.maximum.accumulate(0.2 + 0.7 * x + rng.normal(0, 0.05, 15) * 0)
            y = np.maximum.accumulate(0.1 + 0.8 * x)
            ax.step(x, y, where="post", color=TEAL, lw=2)
            ax.plot([0, 1], [0, 1], "--", color=SLATE)
            ax.grid(True, alpha=0.3)
        else:  # ece
            bins = np.arange(10)
            conf = bins / 10 + 0.05
            acc = conf - 0.08
            ax.bar(conf, acc, width=0.08, color=TEAL, edgecolor=DEEP, label="acc")
            ax.plot([0, 1], [0, 1], "--", color=GOLD)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        style(ax, title)
        return

    if kind in ("violin", "ba", "pval", "bonf", "lrt", "huber", "pinball", "margin"):
        if kind == "violin":
            d = [rng.normal(0, 1, 80), rng.normal(1, 0.7, 80), rng.normal(0.3, 1.2, 80)]
            ax.violinplot(d, showmeans=True)
            ax.set_xticks([1, 2, 3])
        elif kind == "ba":
            a = rng.normal(10, 2, 60)
            b = a + rng.normal(0, 0.8, 60)
            ax.scatter((a + b) / 2, a - b, c=TEAL, s=18)
            ax.axhline(0, color=GOLD, lw=1.5)
            sd = np.std(a - b)
            ax.axhline(1.96 * sd, color=ROSE, ls="--")
            ax.axhline(-1.96 * sd, color=ROSE, ls="--")
            ax.set_xlabel("mean")
            ax.set_ylabel("diff")
        elif kind == "pval":
            z = np.linspace(-4, 4, 300)
            d = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            ax.plot(z, d, color=TEAL, lw=2)
            ax.fill_between(z[z > 2.0], d[z > 2.0], color=ROSE, alpha=0.5)
            ax.axvline(2.0, color=GOLD, ls="--")
            ax.set_xlabel("statistic")
        elif kind == "bonf":
            m = np.arange(1, 50)
            ax.plot(m, 0.05 / m, color=TEAL, lw=2)
            ax.set_xlabel("tests m")
            ax.set_ylabel("α/m")
            ax.set_yscale("log")
        elif kind == "lrt":
            x = np.linspace(0, 10, 100)
            ax.plot(x, np.exp(-0.5 * x), color=TEAL, lw=2, label="χ² null approx")
            ax.axvline(3.84, color=GOLD, ls="--", label="crit 0.05 df1")
            ax.legend(fontsize=8)
        elif kind == "huber":
            r = np.linspace(-3, 3, 200)
            sq = 0.5 * r ** 2
            hub = np.where(np.abs(r) < 1, 0.5 * r ** 2, np.abs(r) - 0.5)
            ax.plot(r, sq, color=GOLD, lw=2, label="squared")
            ax.plot(r, hub, color=TEAL, lw=2, label="Huber")
            ax.legend(fontsize=8)
        elif kind == "pinball":
            u = np.linspace(-2, 2, 200)
            tau = 0.9
            loss = np.where(u >= 0, tau * u, (tau - 1) * u)
            ax.plot(u, loss, color=TEAL, lw=2)
            ax.set_xlabel("y-ŷ")
        else:
            ax.plot([-2, 2], [0, 0], color=SLATE, lw=2)
            ax.fill_between([-2, 2], [0.4, 0.4], [-0.4, -0.4], color=TEAL, alpha=0.2)
            ax.scatter(rng.normal(-1, 0.3, 30), rng.normal(0.8, 0.2, 30), c=TEAL, s=12)
            ax.scatter(rng.normal(1, 0.3, 30), rng.normal(-0.8, 0.2, 30), c=GOLD, s=12)
            ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        style(ax, title)
        return

    if kind in ("spectral", "coreset", "hdb", "ngram", "prox", "ltr", "bin", "missind", "entemb"):
        if kind in ("spectral", "coreset", "hdb"):
            a = rng.normal([-1, 0], 0.25, (40, 2))
            b = rng.normal([1, 0], 0.25, (40, 2))
            ax.scatter(a[:, 0], a[:, 1], c=TEAL, s=16)
            ax.scatter(b[:, 0], b[:, 1], c=GOLD, s=16)
            if kind == "coreset":
                ax.scatter([a[0, 0], b[0, 0]], [a[0, 1], b[0, 1]], c=ROSE, s=80, marker="*")
            if kind == "hdb":
                ax.add_patch(Circle((-1, 0), 0.6, fill=False, ls="--", edgecolor=TEAL))
                ax.add_patch(Circle((1, 0), 0.6, fill=False, ls="--", edgecolor=GOLD))
            ax.set_aspect("equal")
        elif kind == "ngram":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 4)
            ax.axis("off")
            box(ax, 0.4, 1.5, 2.5, 1.4, "unigram", fc=SLATE, fs=10)
            box(ax, 3.5, 1.5, 2.5, 1.4, "bigram", fc=TEAL, fs=10)
            box(ax, 6.6, 1.5, 2.5, 1.4, "backoff", fc=GOLD, tc=INK, fs=10)
        elif kind == "prox":
            ax.bar(["exact", "window3", "window10"], [1.0, 0.7, 0.4], color=TEAL, edgecolor=DEEP)
        elif kind == "ltr":
            ax.plot([0, 1], [0, 1], color=SLATE, ls="--")
            ax.scatter([0.2, 0.6], [0.7, 0.3], c=[TEAL, ROSE], s=60)
            ax.text(0.2, 0.8, "prefer i≻j", color=TEAL, fontsize=9)
            ax.set_xlabel("score i")
            ax.set_ylabel("score j")
        elif kind == "bin":
            x = rng.normal(0, 1, 200)
            ax.hist(x, bins=8, color=TEAL, edgecolor=DEEP, alpha=0.85)
        elif kind == "missind":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 4)
            ax.axis("off")
            box(ax, 0.5, 1.3, 4, 1.6, "x_j value", fc=TEAL, fs=11)
            box(ax, 5.5, 1.3, 4, 1.6, "1[x_j missing]", fc=GOLD, tc=INK, fs=11)
        else:
            emb = rng.normal(0, 1, (8, 2))
            ax.scatter(emb[:, 0], emb[:, 1], c=TEAL, s=40)
            for i in range(8):
                ax.text(emb[i, 0], emb[i, 1], f"c{i}", fontsize=8)
        style(ax, title)
        return

    if kind in (
        "lr", "adam", "clipg", "moco", "clip", "jepa", "posenc", "xattn", "beam", "softp",
        "adv", "intrin", "kd", "flash", "spec", "louvain", "n2v", "gat", "synth", "dp",
    ):
        if kind == "lr":
            s = np.arange(0, 100)
            ax.plot(s, 0.1 * (0.5 ** (s / 30)), color=TEAL, lw=2, label="exp decay")
            ax.plot(s, 0.1 * 0.5 * (1 + np.cos(np.pi * s / 100)), color=GOLD, lw=2, label="cosine")
            ax.legend(fontsize=8)
            ax.set_xlabel("step")
        elif kind == "adam":
            t = np.arange(1, 80)
            ax.plot(t, 1 - 0.9 ** t, color=TEAL, label="1-β1^t")
            ax.plot(t, 1 - 0.999 ** t, color=GOLD, label="1-β2^t")
            ax.legend(fontsize=8)
        elif kind == "clipg":
            g = np.linspace(0, 5, 50)
            ax.plot(g, np.minimum(g, 1.5), color=TEAL, lw=2)
            ax.axhline(1.5, color=GOLD, ls="--")
            ax.set_xlabel("||g||")
        elif kind in ("moco", "clip", "jepa"):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 4)
            ax.axis("off")
            if kind == "moco":
                box(ax, 0.4, 1.3, 3, 1.6, "query enc", fc=TEAL, fs=10)
                box(ax, 3.8, 1.3, 2.4, 1.6, "queue", fc=SLATE, fs=10)
                box(ax, 6.8, 1.3, 2.8, 1.6, "keys", fc=GOLD, tc=INK, fs=10)
            elif kind == "clip":
                box(ax, 0.5, 1.3, 4, 1.6, "image enc", fc=TEAL, fs=11)
                box(ax, 5.5, 1.3, 4, 1.6, "text enc", fc=GOLD, tc=INK, fs=11)
            else:
                box(ax, 0.5, 1.3, 4, 1.6, "context enc", fc=TEAL, fs=11)
                box(ax, 5.5, 1.3, 4, 1.6, "predict z_target", fc=DEEP, fs=11)
        elif kind == "posenc":
            p = np.arange(0, 50)
            ax.plot(p, np.sin(p / 5), color=TEAL)
            ax.plot(p, np.cos(p / 5), color=GOLD)
        elif kind == "xattn":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 4)
            ax.axis("off")
            box(ax, 0.4, 1.3, 2.5, 1.6, "Q from A", fc=TEAL, fs=10)
            box(ax, 3.7, 1.3, 2.5, 1.6, "K,V from B", fc=GOLD, tc=INK, fs=10)
            box(ax, 7, 1.3, 2.5, 1.6, "context", fc=DEEP, fs=10)
        elif kind == "beam":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 5)
            ax.axis("off")
            box(ax, 4, 3.5, 2, 1, "start", fc=TEAL, fs=10)
            for i, x in enumerate([1, 4, 7]):
                box(ax, x, 1.5, 2, 1, f"beam{i+1}", fc=GOLD, tc=INK, fs=9)
            for x in [2, 5, 8]:
                ax.annotate("", xy=(x, 2.5), xytext=(5, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        elif kind == "softp":
            a = np.array([1.0, 0.2, 0.5])
            p = np.exp(a) / np.exp(a).sum()
            ax.bar(range(3), p, color=TEAL)
        elif kind == "adv":
            ax.plot(np.cumsum(rng.normal(0, 1, 50)), color=TEAL, lw=1.5)
            ax.axhline(0, color=GOLD, ls="--")
            ax.set_ylabel("advantage path")
        elif kind == "intrin":
            ax.plot(np.exp(-np.linspace(0, 3, 50)), color=TEAL, lw=2, label="novelty bonus")
            ax.legend(fontsize=8)
        elif kind == "kd":
            logits = np.array([2.0, 1.0, 0.2])
            for T, c in [(1, TEAL), (4, GOLD)]:
                e = np.exp(logits / T)
                ax.plot(e / e.sum(), "o-", color=c, label=f"T={T}")
            ax.legend(fontsize=8)
        elif kind == "flash":
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 6)
            ax.axis("off")
            for i in range(3):
                for j in range(3):
                    box(ax, 0.5 + j * 2.4, 3.5 - i * 1.5, 2.1, 1.2, f"tile {i},{j}", fc=TEAL if (i + j) % 2 == 0 else DEEP, fs=8)
        elif kind == "spec":
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 4)
            ax.axis("off")
            box(ax, 0.5, 1.3, 4, 1.6, "draft model\nfast tokens", fc=TEAL, fs=10)
            box(ax, 5.5, 1.3, 4, 1.6, "verify model\naccept/reject", fc=GOLD, tc=INK, fs=10)
        elif kind in ("louvain", "n2v", "gat"):
            pos = {i: (np.cos(2 * np.pi * i / 6), np.sin(2 * np.pi * i / 6)) for i in range(6)}
            for i in range(6):
                ax.plot([pos[i][0], pos[(i + 1) % 6][0]], [pos[i][1], pos[(i + 1) % 6][1]], color=SLATE, lw=1)
                ax.plot(pos[i][0], pos[i][1], "o", color=TEAL, ms=12)
            ax.set_aspect("equal")
            ax.axis("off")
        elif kind == "synth":
            ax.plot([0, 1], [0.9, 0.4], color=TEAL, lw=2, label="utility")
            ax.plot([0, 1], [0.2, 0.85], color=ROSE, lw=2, label="privacy risk")
            ax.legend(fontsize=8)
            ax.set_xlabel("synth fidelity")
        else:
            eps = np.linspace(0.1, 5, 40)
            ax.plot(eps, 1 / eps, color=TEAL, lw=2)
            ax.set_xlabel("ε privacy")
            ax.set_ylabel("noise scale")
        if ax.axison:
            ax.grid(True, alpha=0.3)
        style(ax, title)
        return

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    labs = {
        "acr": ["AUC", "ECE", "Brier", "NLL", "κ"],
        "greek": ["α", "β", "γ", "λ", "η"],
        "ineq": ["Jensen", "Markov", "Cheby", "Hoeff", "McD"],
    }[kind]
    for i, t in enumerate(labs):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, t, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, title)


def run_cycle(cycle, topics):
    for i, (title, kind) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        draw(ax, kind, title, cycle * 100 + i)
        save(fig, f"ml_fig_c{cycle}_{i:02d}.png")
    for i, ch in enumerate(CHS):
        p = CURR / ch
        fig = f"ml_fig_c{cycle}_{i:02d}.png"
        cap = topics[i][0]
        block = (
            f"\n![c{cycle} teaching panel {i:02d} (original).](../assets/figures/{fig})\n"
            f"*Figure — {cap}. Synthetic teaching geometry—not a causal claim.*\n"
        )
        text = p.read_text(encoding="utf-8")
        if fig in text:
            continue
        if "## Chapter Summary" in text:
            text = text.replace("## Chapter Summary", block + "\n## Chapter Summary", 1)
        else:
            text = text.rstrip() + "\n" + block
        p.write_text(text, encoding="utf-8")
    print("EMBEDDED", cycle, 20)


def main(cycles=None):
    mapping = {91: C91, 92: C92, 93: C93}
    for c in cycles or [91, 92, 93]:
        run_cycle(c, mapping[c])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main([int(x) for x in sys.argv[1].split(",")])
    else:
        main()
