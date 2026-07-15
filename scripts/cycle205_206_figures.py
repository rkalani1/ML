#!/usr/bin/env python3
"""Cycle-205/206 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Arc, FancyArrowPatch, Ellipse

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
TEAL, DEEP, INK, GOLD, SLATE, ROSE, MINT = (
    "#0d9488", "#0f766e", "#0f172a", "#c9a227", "#64748b", "#e11d48", "#14b8a6"
)
CHS = sorted(p.name for p in CURR.glob("*.md"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", name)


def box(ax, x, y, w, h, t, fc=TEAL, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fc, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


# C205
def d_qr(ax, t, rng):
    # QR: columns of Q orthonormal, R upper triangular heatmap
    Q = np.array([[0.8, -0.6], [0.6, 0.8]])
    ax.annotate("", xy=Q[:, 0], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.annotate("", xy=Q[:, 1], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    ax.text(0.9, 0.55, "q1", color=TEAL, fontsize=10)
    ax.text(-0.75, 0.85, "q2", color=GOLD, fontsize=10)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ls="--", edgecolor=SLATE))
    R = np.array([[2.0, 1.1], [0.0, 1.4]])
    ax_ins = ax.inset_axes([0.62, 0.12, 0.32, 0.4])
    ax_ins.imshow(R, cmap="Greens")
    ax_ins.set_title("R upper", fontsize=8)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.2, 1.4)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_consent_stack(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, (lab, c) in enumerate([("collect", TEAL), ("use", DEEP), ("share", GOLD), ("retain", ROSE)]):
        tc = INK if c == GOLD else "white"
        box(ax, 0.5 + i * 2.9, 1.6, 2.7, 1.8, lab, fc=c, fs=12, tc=tc)
    ax.text(6, 4.2, "purpose-limited consent stack", ha="center", fontsize=11, color=INK)
    style(ax, t)


def d_radon_nikodym(ax, t, rng):
    x = np.linspace(0.01, 3, 200)
    p = np.exp(-x)
    q = 0.6 * np.exp(-0.6 * x)
    p, q = p / np.trapezoid(p, x), q / np.trapezoid(q, x)
    ax.plot(x, p, color=TEAL, lw=2, label="p")
    ax.plot(x, q, color=GOLD, lw=2, label="q")
    ax.plot(x, p / (q + 1e-9), color=ROSE, lw=1.8, label="dp/dq sketch")
    ax.set_ylim(0, 4)
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_qq_plot(ax, t, rng):
    data = np.sort(rng.standard_t(5, 120))
    u = (np.arange(1, 121) - 0.5) / 120
    z = np.quantile(rng.normal(0, 1, 8000), u)
    ax.scatter(z, data, c=TEAL, s=14, alpha=0.8)
    lim = max(abs(z).max(), abs(data).max())
    ax.plot([-lim, lim], [-lim, lim], "--", color=GOLD, lw=1.5)
    ax.set_xlabel("theoretical N(0,1) quantiles")
    ax.set_ylabel("sample quantiles")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_ks_test(ax, t, rng):
    x = np.sort(rng.normal(0, 1, 80))
    ecdf = np.arange(1, 81) / 80
    # logistic CDF as smooth H0 stand-in (no scipy)
    F = 1.0 / (1.0 + np.exp(-1.7 * x))
    ax.step(x, ecdf, where="post", color=TEAL, lw=2, label="ECDF")
    ax.plot(x, F, color=GOLD, lw=2, label="H0 CDF")
    i = int(np.argmax(np.abs(ecdf - F)))
    ax.annotate("", xy=(x[i], ecdf[i]), xytext=(x[i], F[i]), arrowprops=dict(arrowstyle="<->", color=ROSE, lw=2))
    ax.text(x[i] + 0.1, (ecdf[i] + F[i]) / 2, "D_n", color=ROSE, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dendrogram(ax, t, rng):
    # simple hierarchical merge sketch
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    leaves = [1, 2.5, 4, 5.5, 7, 8.5]
    for x in leaves:
        ax.plot([x, x], [0, 0.8], color=TEAL, lw=2)
        ax.plot(x, 0, "o", color=DEEP, ms=8)
    # merges
    merges = [((1, 2.5), 1.8), ((4, 5.5), 2.0), ((7, 8.5), 1.5), ((1.75, 4.75), 3.5), ((1.75, 7.75), 5.0)]
    # redo cleaner
    ax.plot([1, 1, 2.5, 2.5], [0.8, 1.8, 1.8, 0.8], color=TEAL, lw=1.8)
    ax.plot([4, 4, 5.5, 5.5], [0.8, 2.0, 2.0, 0.8], color=TEAL, lw=1.8)
    ax.plot([7, 7, 8.5, 8.5], [0.8, 1.5, 1.5, 0.8], color=TEAL, lw=1.8)
    ax.plot([1.75, 1.75, 4.75, 4.75], [1.8, 3.4, 3.4, 2.0], color=GOLD, lw=1.8)
    ax.plot([3.25, 3.25, 7.75, 7.75], [3.4, 5.0, 5.0, 1.5], color=ROSE, lw=1.8)
    ax.text(5, 5.5, "agglomerative height = merge cost", ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_apriori_levels(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    levels = [("L1 items", 5, TEAL), ("L2 pairs", 4, DEEP), ("L3 triples", 2, GOLD), ("L4", 1, ROSE)]
    x = 0.5
    for lab, n, c in levels:
        tc = INK if c == GOLD else "white"
        box(ax, x, 1.8, 2.6, 1.6, f"{lab}\n({n})", fc=c, fs=10, tc=tc)
        x += 2.9
    ax.text(6, 4.2, "level-wise candidate generation + prune", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_woe_iv(ax, t, rng):
    bins = np.arange(6)
    dist_g = np.array([0.05, 0.1, 0.2, 0.3, 0.25, 0.1])
    dist_b = np.array([0.2, 0.25, 0.25, 0.15, 0.1, 0.05])
    woe = np.log((dist_g + 1e-6) / (dist_b + 1e-6))
    iv = (dist_g - dist_b) * woe
    ax.bar(bins - 0.15, dist_g, 0.3, color=TEAL, label="good share")
    ax.bar(bins + 0.15, dist_b, 0.3, color=ROSE, label="bad share")
    ax2 = ax.twinx()
    ax2.plot(bins, woe, "o-", color=GOLD, lw=2, label="WoE")
    ax.set_xlabel("bin")
    ax.set_ylabel("share")
    ax2.set_ylabel("WoE")
    ax.text(0.02, 0.92, f"IV≈{iv.sum():.2f}", transform=ax.transAxes, fontsize=10, color=INK)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_mds_stress(ax, t, rng):
    dims = np.arange(1, 9)
    stress = 0.35 * np.exp(-0.4 * (dims - 1)) + 0.02
    ax.plot(dims, stress, "o-", color=TEAL, lw=2)
    ax.axhline(0.1, color=ROSE, ls="--", label="rule-of-thumb stress")
    ax.set_xlabel("embedding dim")
    ax.set_ylabel("Kruskal stress sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_vif(ax, t, rng):
    feats = [f"x{i}" for i in range(1, 7)]
    r2 = np.array([0.2, 0.55, 0.82, 0.91, 0.4, 0.15])
    vif = 1 / (1 - r2)
    colors = [ROSE if v > 5 else TEAL for v in vif]
    ax.bar(feats, vif, color=colors, edgecolor=DEEP)
    ax.axhline(5, color=GOLD, ls="--", label="VIF=5 caution")
    ax.axhline(10, color=ROSE, ls=":", label="VIF=10 high")
    ax.set_ylabel("VIF = 1/(1−R²_j)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_mcc(ax, t, rng):
    # MCC vs accuracy on imbalance
    prev = np.linspace(0.05, 0.5, 40)
    # fixed sens/spec
    sens, spec = 0.9, 0.85
    tp = sens * prev
    fn = (1 - sens) * prev
    tn = spec * (1 - prev)
    fp = (1 - spec) * (1 - prev)
    acc = tp + tn
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
    ax.plot(prev, acc, color=GOLD, lw=2, label="accuracy")
    ax.plot(prev, mcc, color=TEAL, lw=2, label="MCC")
    ax.set_xlabel("prevalence")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_resnet_stages(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    stages = ["stem", "stage1\nC2", "stage2\nC4", "stage3\nC8", "head"]
    for i, s in enumerate(stages):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.6, s, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
        if i < 4:
            ax.annotate("", xy=(2.5 + i * 2.35, 2.0), xytext=(2.35 + i * 2.35, 2.0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, t)


def d_dino(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.8, 3.5, 1.5, "student\nlocal crops", fc=TEAL, fs=11)
    box(ax, 4.5, 2.8, 3.5, 1.5, "teacher EMA\nglobal crop", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.5, 2.8, 3.0, 1.5, "CE match\nsharpen", fc=DEEP, fs=11)
    ax.annotate("", xy=(4.5, 3.5), xytext=(4.0, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.5, 3.5), xytext=(8.0, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.text(6, 1.3, "DINO self-distill without labels", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_patch_embed(ax, t, rng):
    # image grid -> tokens
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j, 3 - i), 0.95, 0.95, facecolor=TEAL if (i + j) % 2 == 0 else MINT, edgecolor=DEEP, alpha=0.8))
    ax.annotate("", xy=(5.5, 2), xytext=(4.2, 2), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    for k in range(8):
        ax.add_patch(Rectangle((6, 3.5 - k * 0.45), 2.8, 0.38, facecolor=GOLD if k == 0 else TEAL, edgecolor=DEEP, alpha=0.85))
        ax.text(7.4, 3.65 - k * 0.45, "CLS" if k == 0 else f"patch {k}", ha="center", va="center", fontsize=7, color=INK if k == 0 else "white")
    ax.set_xlim(-0.2, 9.5)
    ax.set_ylim(-0.5, 4.2)
    ax.axis("off")
    style(ax, t)


def d_eligibility_trace(ax, t, rng):
    steps = np.arange(0, 40)
    events = np.zeros(40)
    events[[5, 18, 30]] = 1
    e = 0
    trace = []
    for a in events:
        e = 0.9 * e + a
        trace.append(e)
    ax.stem(steps, events, linefmt=SLATE, markerfmt="ro", basefmt=" ")
    ax.plot(steps, trace, color=TEAL, lw=2, label="eligibility e_t")
    ax.set_xlabel("time")
    ax.set_ylabel("trace / event")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_quant_bins(ax, t, rng):
    w = rng.normal(0, 1, 200)
    ax.hist(w, bins=30, color=SLATE, alpha=0.4, density=True, label="fp32 weights")
    # fake quant levels
    levels = np.linspace(-2.5, 2.5, 9)
    for lv in levels:
        ax.axvline(lv, color=TEAL, alpha=0.7, lw=1)
    ax.hist(np.digitize(w, levels), bins=len(levels), density=True, alpha=0)  # no-op keep simple
    q = levels[np.clip(np.digitize(w, levels) - 1, 0, len(levels) - 1)]
    ax.hist(q, bins=levels, color=GOLD, alpha=0.55, density=True, label="uniform quant bins")
    ax.legend(fontsize=8)
    ax.set_xlabel("value")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_modularity(ax, t, rng):
    # two communities adjacency block
    A = np.zeros((10, 10))
    A[:5, :5] = rng.random((5, 5)) > 0.4
    A[5:, 5:] = rng.random((5, 5)) > 0.4
    A[:5, 5:] = rng.random((5, 5)) > 0.85
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    ax.imshow(A, cmap="Greens", interpolation="nearest")
    ax.axhline(4.5, color=GOLD, lw=2)
    ax.axvline(4.5, color=GOLD, lw=2)
    ax.set_xlabel("node")
    ax.set_ylabel("node")
    ax.text(4.5, -1.2, "block structure → high modularity", ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_concept_drift_stream(ax, t, rng):
    n = 200
    x = np.arange(n)
    y = (x < 100).astype(float) * (0.2 + 0.01 * rng.normal(size=n)) + (x >= 100) * (0.7 + 0.01 * rng.normal(size=n))
    ax.plot(x, y, color=TEAL, lw=1.5)
    ax.axvline(100, color=ROSE, ls="--", lw=2, label="concept change")
    ax.set_xlabel("stream index")
    ax.set_ylabel("label rate / error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_canary(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 1.8, 3, 1.8, "100%\nv1", fc=SLATE, fs=12)
    box(ax, 4.5, 1.8, 3, 1.8, "95% v1\n5% canary v2", fc=TEAL, fs=11)
    box(ax, 8.5, 1.8, 3, 1.8, "promote\nor rollback", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.5, 2.7), xytext=(3.5, 2.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("", xy=(8.5, 2.7), xytext=(7.5, 2.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    style(ax, t)


def d_glossary_likelihood(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["likelihood", "prior", "posterior", "MAP", "MLE"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


# C206
def d_cholesky(ax, t, rng):
    L = np.array([[2.0, 0, 0], [0.5, 1.5, 0], [0.3, 0.4, 1.2]])
    ax.imshow(L, cmap="Greens")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{L[i, j]:.1f}", ha="center", va="center", color=INK if L[i, j] else SLATE, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(1, 3.4, "L lower-triangular; A=LLᵀ", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_fairness_slice(ax, t, rng):
    groups = ["A", "B", "C", "D"]
    tpr = [0.88, 0.81, 0.76, 0.90]
    fpr = [0.12, 0.18, 0.22, 0.10]
    x = np.arange(len(groups))
    ax.bar(x - 0.15, tpr, 0.3, color=TEAL, label="TPR")
    ax.bar(x + 0.15, fpr, 0.3, color=ROSE, label="FPR")
    ax.axhline(np.mean(tpr), color=GOLD, ls="--", label="mean TPR")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_sample_complex(ax, t, rng):
    n = np.logspace(1, 4, 40)
    vc = 50
    # n ≳ (VC/ε²)log
    eps = np.sqrt(vc * np.log(n) / n)
    ax.loglog(n, eps, color=TEAL, lw=2)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("ε scale ~ √(d log n / n)")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_ridgeline(ax, t, rng):
    y0 = 0
    for i in range(6):
        x = np.linspace(-3, 3, 200)
        dens = np.exp(-0.5 * (x - 0.4 * i + 1) ** 2) / np.sqrt(2 * np.pi)
        ax.fill_between(x, y0, y0 + dens, color=TEAL if i % 2 == 0 else DEEP, alpha=0.7)
        ax.plot(x, y0 + dens, color=INK, lw=0.5, alpha=0.4)
        y0 += 0.35
    ax.set_yticks([])
    ax.set_xlabel("value")
    style(ax, t)


def d_bootstrap_ci(ax, t, rng):
    boots = rng.normal(0.55, 0.04, 500)
    ax.hist(boots, bins=30, color=TEAL, edgecolor=DEEP, alpha=0.85)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    ax.axvline(lo, color=ROSE, lw=2, label="2.5%")
    ax.axvline(hi, color=ROSE, lw=2, label="97.5%")
    ax.axvline(np.mean(boots), color=GOLD, lw=2, label="mean")
    ax.set_xlabel("bootstrap statistic")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gmm_respons(ax, t, rng):
    x = np.linspace(-4, 4, 300)
    r1 = np.exp(-0.5 * (x + 1.5) ** 2)
    r2 = np.exp(-0.5 * (x - 1.2) ** 2)
    s = r1 + r2 + 1e-9
    ax.plot(x, r1 / s, color=TEAL, lw=2, label="γ(z=1|x)")
    ax.plot(x, r2 / s, color=GOLD, lw=2, label="γ(z=2|x)")
    ax.set_xlabel("x")
    ax.set_ylabel("responsibility")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_tfidf_space(ax, t, rng):
    # 2d docs
    docs = rng.normal(0, 1, (25, 2))
    q = np.array([1.2, 0.8])
    sims = docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q) + 1e-9)
    ax.scatter(docs[:, 0], docs[:, 1], c=sims, cmap="YlGn", s=40, edgecolors=DEEP)
    ax.annotate("", xy=q, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    ax.text(q[0], q[1], " query", color=ROSE, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_cyclic_features(ax, t, rng):
    hour = np.arange(0, 24)
    ax.plot(hour, np.sin(2 * np.pi * hour / 24), color=TEAL, lw=2, label="sin(2πh/24)")
    ax.plot(hour, np.cos(2 * np.pi * hour / 24), color=GOLD, lw=2, label="cos(2πh/24)")
    ax.set_xlabel("hour")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_tsne_early_exag(ax, t, rng):
    steps = np.arange(0, 200)
    exag = np.where(steps < 100, 12.0, 1.0)
    kl = 3 * np.exp(-steps / 60) + 0.2
    ax.plot(steps, kl, color=TEAL, lw=2, label="KL loss sketch")
    ax2 = ax.twinx()
    ax2.plot(steps, exag, color=GOLD, lw=2, label="early exaggeration")
    ax.set_xlabel("iteration")
    ax.set_ylabel("KL")
    ax2.set_ylabel("exaggeration")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_box_cox(ax, t, rng):
    y = rng.lognormal(0, 0.7, 300)
    ax.hist(y, bins=30, color=SLATE, alpha=0.45, density=True, label="skewed y")
    # lambda=0 log
    ax.hist(np.log(y), bins=30, color=TEAL, alpha=0.55, density=True, label="log y (λ→0)")
    ax.legend(fontsize=8)
    ax.set_xlabel("value")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_cohen_kappa(ax, t, rng):
    # agreement vs chance
    po = np.linspace(0.5, 1.0, 40)
    pe = 0.4
    kappa = (po - pe) / (1 - pe)
    ax.plot(po, kappa, color=TEAL, lw=2)
    ax.axhline(0.6, color=GOLD, ls="--", label="substantial ~0.6")
    ax.set_xlabel("observed agreement p_o")
    ax.set_ylabel("κ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_prelu(ax, t, rng):
    x = np.linspace(-3, 3, 200)
    ax.plot(x, np.maximum(0, x), color=GOLD, lw=2, label="ReLU")
    ax.plot(x, np.where(x >= 0, x, 0.25 * x), color=TEAL, lw=2, label="PReLU a=0.25")
    ax.plot(x, np.where(x >= 0, x, 0.01 * x), color=SLATE, lw=1.5, ls="--", label="Leaky 0.01")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mae_pretrain(ax, t, rng):
    # mask patches
    for i in range(4):
        for j in range(6):
            masked = rng.random() < 0.6
            fc = SLATE if masked else TEAL
            ax.add_patch(Rectangle((j, 3 - i), 0.95, 0.95, facecolor=fc, edgecolor=DEEP, alpha=0.85))
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.text(3, -0.2, "MAE: high mask ratio, reconstruct pixels/tokens", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_spec_aug(ax, t, rng):
    S = rng.random((32, 80))
    S[10:16, :] *= 0.15  # time mask
    S[:, 30:40] *= 0.15  # freq mask
    ax.imshow(S, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("time")
    ax.set_ylabel("freq bin")
    ax.text(40, 34, "time/freq masks", ha="center", color="white", fontsize=10, fontweight="bold")
    style(ax, t)


def d_retrace(ax, t, rng):
    # multi-step returns importance
    k = np.arange(1, 12)
    c = np.cumprod(np.clip(rng.uniform(0.7, 1.0, 11), 0, 1))
    ax.bar(k, c, color=TEAL, edgecolor=DEEP)
    ax.set_xlabel("trace depth k")
    ax.set_ylabel("product of c_i (Retrace)")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_distil_student(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 4, 1.8, "teacher\nlarge logits", fc=GOLD, tc=INK, fs=12)
    box(ax, 7.5, 2.5, 4, 1.8, "student\ncompact", fc=TEAL, fs=12)
    ax.annotate("", xy=(7.5, 3.4), xytext=(4.5, 3.4), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    ax.text(6, 3.9, "T-soft CE + hard CE", ha="center", color=ROSE, fontsize=9, fontweight="bold")
    style(ax, t)


def d_triangle_motif(ax, t, rng):
    pos = {"a": (0, 0), "b": (1, 0), "c": (0.5, 0.87), "d": (2.2, 0.2)}
    for e in [("a", "b"), ("b", "c"), ("c", "a"), ("b", "d")]:
        ax.plot([pos[e[0]][0], pos[e[1]][0]], [pos[e[0]][1], pos[e[1]][1]], color=SLATE, lw=1.5)
    for n, p in pos.items():
        ax.plot(*p, "o", color=TEAL if n != "d" else GOLD, ms=14)
        ax.text(p[0], p[1] - 0.2, n, ha="center", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(1, 1.2, "triangle motif count", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_snomed_map(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.8, 3.2, 1.4, "local code", fc=SLATE, fs=11)
    box(ax, 4.4, 2.8, 3.2, 1.4, "map table\nversioned", fc=TEAL, fs=11)
    box(ax, 8.3, 2.8, 3.2, 1.4, "standard\nconcept", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.4, 3.5), xytext=(3.7, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.3, 3.5), xytext=(7.6, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.text(6, 1.3, "terminology mapping must freeze with model", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_rollback_plan(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    steps = ["detect", "page", "flip traffic", "verify", "postmortem"]
    for i, s in enumerate(steps):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, s, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_glossary_optim(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["SGD", "momentum", "Adam", "AdamW", "Lion"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C205 = [
    ("QR factorization geometry", d_qr),
    ("Purpose-limited consent stack", d_consent_stack),
    ("Likelihood ratio density sketch", d_radon_nikodym),
    ("Q-Q plot normality check", d_qq_plot),
    ("KS statistic ECDF gap", d_ks_test),
    ("Agglomerative dendrogram height", d_dendrogram),
    ("Apriori level-wise prune", d_apriori_levels),
    ("Weight of evidence IV bins", d_woe_iv),
    ("MDS Kruskal stress curve", d_mds_stress),
    ("Variance inflation factors", d_vif),
    ("MCC vs accuracy prevalence", d_mcc),
    ("ResNet stage width progression", d_resnet_stages),
    ("DINO teacher-student crops", d_dino),
    ("ViT patch embedding tokens", d_patch_embed),
    ("Eligibility trace decay events", d_eligibility_trace),
    ("Uniform quantization bins", d_quant_bins),
    ("Modularity block adjacency", d_modularity),
    ("Concept drift in stream", d_concept_drift_stream),
    ("Canary release traffic split", d_canary),
    ("Glossary Bayes term strip", d_glossary_likelihood),
]

C206 = [
    ("Cholesky factor lower triangle", d_cholesky),
    ("Fairness slice TPR FPR", d_fairness_slice),
    ("Sample complexity epsilon scale", d_sample_complex),
    ("Ridgeline density small multiples", d_ridgeline),
    ("Bootstrap percentile CI", d_bootstrap_ci),
    ("GMM soft responsibilities", d_gmm_respons),
    ("TF-IDF cosine query space", d_tfidf_space),
    ("Cyclic hour sin-cos features", d_cyclic_features),
    ("t-SNE early exaggeration", d_tsne_early_exag),
    ("Box-Cox log transform effect", d_box_cox),
    ("Cohen kappa chance correction", d_cohen_kappa),
    ("PReLU vs ReLU leak", d_prelu),
    ("MAE high-ratio patch mask", d_mae_pretrain),
    ("SpecAugment time-freq masks", d_spec_aug),
    ("Retrace truncated importance", d_retrace),
    ("Distillation teacher to student", d_distil_student),
    ("Triangle graph motif", d_triangle_motif),
    ("Versioned code map freeze", d_snomed_map),
    ("Incident rollback playbook", d_rollback_plan),
    ("Glossary optimizer name strip", d_glossary_optim),
]


def embed(cycle, topics):
    for i, (title, fn) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        rng = np.random.default_rng(cycle * 1000 + i * 23 + 5)
        try:
            fn(ax, title, rng)
        except Exception as e:
            # fallback if scipy missing in ks
            ax.clear()
            ax.text(0.5, 0.5, f"{title}\n(render fallback)", ha="center", va="center", transform=ax.transAxes)
            style(ax, title)
            print("FALLBACK", cycle, i, e)
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
    print("EMBEDDED", cycle, len(topics))


if __name__ == "__main__":
    import sys
    m = {205: C205, 206: C206}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [205, 206]
    for c in cycles:
        embed(c, m[c])
