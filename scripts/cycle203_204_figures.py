#!/usr/bin/env python3
"""Cycle-203/204 quality densify: more novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
from matplotlib.patches import Arc

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
TEAL, DEEP, INK, GOLD, SLATE, ROSE, MINT, SOFT = (
    "#0d9488",
    "#0f766e",
    "#0f172a",
    "#c9a227",
    "#64748b",
    "#e11d48",
    "#14b8a6",
    "#ecfeff",
)
CHS = sorted(p.name for p in CURR.glob("*.md"))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", name)


def box(ax, x, y, w, h, t, fc=TEAL, fs=9, tc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc, edgecolor="none",
        )
    )
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


# --- Cycle 203 draws ---

def d_det_volume(ax, title, rng):
    # parallelogram area = |det|
    O = np.array([0.0, 0.0])
    v1 = np.array([2.2, 0.4])
    v2 = np.array([0.6, 1.6])
    pts = np.array([O, v1, v1 + v2, v2])
    ax.add_patch(Polygon(pts, closed=True, facecolor=TEAL, alpha=0.25, edgecolor=DEEP, lw=2))
    ax.annotate("", xy=v1, xytext=O, arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    ax.annotate("", xy=v2, xytext=O, arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    det = abs(np.linalg.det(np.column_stack([v1, v2])))
    ax.text(1.2, 0.95, f"|det|=area≈{det:.2f}", fontsize=11, color=INK, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 3.2)
    ax.set_ylim(-0.3, 2.4)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_irb_tiles(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 4, 4.5, 4, 1.1, "human data?", fc=TEAL, fs=11)
    box(ax, 0.5, 2.5, 3.5, 1.1, "not research", fc=SLATE, fs=10)
    box(ax, 4.25, 2.5, 3.5, 1.1, "exempt?", fc=GOLD, tc=INK, fs=10)
    box(ax, 8, 2.5, 3.5, 1.1, "full board", fc=ROSE, fs=10)
    box(ax, 4.25, 0.6, 3.5, 1.1, "expedited", fc=DEEP, fs=10)
    ax.annotate("", xy=(2.2, 3.6), xytext=(5, 4.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(6, 3.6), xytext=(6, 4.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(9.8, 3.6), xytext=(7, 4.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(6, 1.7), xytext=(6, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, title)


def d_vc_shatter(ax, title, rng):
    # 3 points shattered by halfplanes vs 4 not always
    pts3 = np.array([[0, 0], [1.5, 0.2], [0.6, 1.2]])
    labels = [1, 0, 1]
    cols = [TEAL if y else GOLD for y in labels]
    ax.scatter(pts3[:, 0], pts3[:, 1], c=cols, s=90, edgecolors=DEEP, zorder=5)
    ax.plot([-0.3, 1.9], [1.0, -0.1], color=ROSE, lw=1.8, label="separating line")
    ax.text(0.6, -0.55, "3 points: all 2³ dichot. realizable (threshold)", ha="center", fontsize=9)
    ax.set_xlim(-0.6, 2.2)
    ax.set_ylim(-0.8, 1.7)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_horizon(ax, title, rng):
    t = np.linspace(0, 8 * np.pi, 300)
    for i, amp in enumerate([1.0, 0.7, 0.5, 0.35]):
        y = amp * (0.5 + 0.5 * np.sin(t + i) * np.cos(0.3 * t))
        base = i * 0.85
        ax.fill_between(t, base, base + y, color=TEAL if i % 2 == 0 else DEEP, alpha=0.7)
        ax.plot(t, base + y, color=INK, lw=0.4, alpha=0.4)
    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.set_ylabel("layered series")
    style(ax, title)


def d_wasserstein1d(ax, title, rng):
    x = np.linspace(-3, 3, 400)
    p = np.exp(-0.5 * (x + 1) ** 2)
    q = np.exp(-0.5 * (x - 1.2) ** 2)
    p, q = p / np.trapezoid(p, x), q / np.trapezoid(q, x)
    P = np.cumsum(p) / np.sum(p)
    Q = np.cumsum(q) / np.sum(q)
    ax.plot(x, p, color=TEAL, lw=2, label="p")
    ax.plot(x, q, color=GOLD, lw=2, label="q")
    # transport arrows at quantiles
    for a in [0.2, 0.5, 0.8]:
        ix = np.searchsorted(P, a)
        jx = np.searchsorted(Q, a)
        ax.annotate("", xy=(x[jx], 0.02), xytext=(x[ix], 0.02),
                    arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.4))
    ax.text(0, max(p.max(), q.max()) * 0.9, r"$W_1$~mean |F⁻¹−G⁻¹|", ha="center", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_gap_stat(ax, title, rng):
    k = np.arange(1, 9)
    logw = 4.5 - 0.55 * np.log(k) + rng.normal(0, 0.03, len(k))
    logw_null = 4.6 - 0.25 * np.log(k)
    gap = logw_null - logw
    ax.plot(k, logw, "o-", color=TEAL, lw=2, label="log W(k) data")
    ax.plot(k, logw_null, "s--", color=GOLD, lw=2, label="null reference")
    ax2 = ax.twinx()
    ax2.bar(k, gap, color=ROSE, alpha=0.35, label="Gap(k)")
    ax.set_xlabel("k clusters")
    ax.set_ylabel("log within SS")
    ax2.set_ylabel("gap")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_fpgrowth(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 3.0, 2.6, 1.3, "item header", fc=SLATE, fs=10)
    box(ax, 3.5, 3.0, 2.6, 1.3, "FP-tree", fc=TEAL, fs=10)
    box(ax, 6.6, 3.0, 2.6, 1.3, "cond. pattern", fc=DEEP, fs=10)
    box(ax, 9.2, 3.0, 2.4, 1.3, "mine", fc=GOLD, tc=INK, fs=10)
    for x in [3.0, 6.1, 9.2]:
        ax.annotate("", xy=(x + 0.3, 3.65), xytext=(x, 3.65), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(6, 1.3, "compress DB into tree; grow frequent itemsets", ha="center", fontsize=10, color=INK)
    style(ax, title)


def d_mi_rank(ax, title, rng):
    feats = [f"x{i}" for i in range(1, 9)]
    mi = np.array([0.42, 0.35, 0.28, 0.18, 0.12, 0.08, 0.05, 0.02])
    order = np.argsort(-mi)
    ax.barh(np.arange(len(feats)), mi[order], color=TEAL, edgecolor=DEEP)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels([feats[i] for i in order], fontsize=9)
    ax.axvline(0.15, color=ROSE, ls="--", label="keep threshold")
    ax.set_xlabel("I(X;Y) estimate")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, title)


def d_sparse_pca(ax, title, rng):
    k = np.arange(1, 21)
    load = np.exp(-0.15 * (k - 1)) * (1 + 0.1 * rng.normal(size=len(k)))
    card = np.minimum(k, 8)
    ax.plot(k, load, "o-", color=TEAL, lw=2, label="loading magnitude")
    ax.fill_between(k, 0, (card / card.max()) * load.max() * 0.9, color=GOLD, alpha=0.25, label="cardinality budget")
    ax.set_xlabel("feature index (sorted)")
    ax.set_ylabel("loading / budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_ridge_path(ax, title, rng):
    lam = np.logspace(-2, 2, 40)
    # synthetic coefficient paths
    for j, c in enumerate([TEAL, GOLD, ROSE, DEEP, SLATE]):
        beta = 1.5 / (1 + lam / (0.3 * (j + 1))) * ((-1) ** j)
        ax.semilogx(lam, beta, color=c, lw=1.8, label=f"β{j+1}")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xlabel("λ ridge")
    ax.set_ylabel("coefficient")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_brier_skill(ax, title, rng):
    # reliability diagram-ish + BSS
    conf = np.linspace(0.05, 0.95, 10)
    acc = conf - 0.08 * np.sin(3 * conf)
    ax.plot([0, 1], [0, 1], "--", color=SLATE, label="perfect")
    ax.plot(conf, acc, "o-", color=TEAL, lw=2, label="model reliability")
    clim = np.full_like(conf, 0.3)
    ax.axhline(0.3, color=GOLD, ls=":", label="climatology base rate")
    brier = np.mean((conf - acc) ** 2) + 0.04
    brier_clim = 0.3 * 0.7
    bss = 1 - brier / brier_clim
    ax.text(0.05, 0.9, f"BSS sketch≈{bss:.2f}", transform=ax.transAxes, fontsize=10, color=INK)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_bn_vs_ln(ax, title, rng):
    # batch: normalize across batch for each feature; layer: across features for each sample
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # feature columns
    for j in range(5):
        for i in range(4):
            ax.add_patch(Rectangle((1 + j * 0.7, 3.2 + i * 0.55), 0.55, 0.45, facecolor=TEAL, alpha=0.35 + 0.1 * i, edgecolor=DEEP))
    ax.add_patch(Rectangle((0.85, 3.1), 0.55 * 5 + 0.3, 0.45 * 4 + 0.25, fill=False, edgecolor=GOLD, lw=2, ls="--"))
    ax.text(2.6, 5.5, "BatchNorm: across batch (fixed feature)", ha="center", color=GOLD, fontsize=9)
    for j in range(5):
        for i in range(4):
            ax.add_patch(Rectangle((1 + j * 0.7, 0.4 + i * 0.55), 0.55, 0.45, facecolor=MINT, alpha=0.35 + 0.1 * j, edgecolor=DEEP))
    ax.add_patch(Rectangle((0.85, 0.85), 0.55 * 5 + 0.3, 0.45, fill=False, edgecolor=ROSE, lw=2, ls="--"))
    ax.text(2.6, 2.5, "LayerNorm: across features (fixed sample)", ha="center", color=ROSE, fontsize=9)
    style(ax, title)


def d_byol(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.8, 3.2, 1.4, "online\nenc+pred", fc=TEAL, fs=11)
    box(ax, 4.4, 2.8, 3.2, 1.4, "stop-grad\ntarget enc", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.3, 2.8, 3.2, 1.4, "MSE in\nproj space", fc=DEEP, fs=11)
    ax.annotate("", xy=(4.4, 3.5), xytext=(3.7, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("", xy=(8.3, 3.5), xytext=(7.6, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(6, 1.3, "BYOL: no negatives; EMA target bootstrap", ha="center", fontsize=10, color=INK)
    style(ax, title)


def d_mel_path(ax, title, rng):
    t = np.linspace(0, 4 * np.pi, 200)
    wave = np.sin(t) + 0.4 * np.sin(3 * t)
    ax.plot(t, wave, color=SLATE, lw=1, alpha=0.7, label="waveform")
    # mel filterbank cartoon as heat strip
    mel = np.abs(np.sin(np.linspace(0, 6, 24)[:, None] + t[None, :] * 0.3))
    ax.imshow(mel, aspect="auto", extent=[0, t[-1], -3.5, -1.2], cmap="YlGn", alpha=0.9)
    ax.text(t[-1] / 2, -3.9, "mel filterbank log-energy", ha="center", fontsize=9, color=INK)
    box_y = 1.8
    ax.text(0.5, 2.4, "→ CNN/Transformer encoder", fontsize=10, color=TEAL, fontweight="bold")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")
    style(ax, title)


def d_ppo_ratio(ax, title, rng):
    r = np.linspace(0.5, 1.5, 200)
    eps = 0.2
    adv = 1.0
    unclipped = r * adv
    clipped = np.clip(r, 1 - eps, 1 + eps) * adv
    obj = np.minimum(unclipped, clipped)
    ax.plot(r, unclipped, color=SLATE, ls="--", label="r·A")
    ax.plot(r, clipped, color=GOLD, ls=":", label="clip(r)A")
    ax.plot(r, obj, color=TEAL, lw=2.5, label="min (PPO)")
    ax.axvline(1 - eps, color=ROSE, ls="--", lw=1)
    ax.axvline(1 + eps, color=ROSE, ls="--", lw=1)
    ax.set_xlabel("probability ratio r(θ)")
    ax.set_ylabel("surrogate term")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_lora(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 1.8, 3.5, 1.6, "frozen W₀\n(d×d)", fc=SLATE, fs=11)
    box(ax, 4.5, 2.6, 2.5, 1.4, "B (d×r)", fc=TEAL, fs=11)
    box(ax, 4.5, 0.8, 2.5, 1.4, "A (r×d)", fc=DEEP, fs=11)
    box(ax, 8.2, 1.8, 3.3, 1.6, "W₀ + BA\ntrainable r≪d", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(8.2, 2.6), xytext=(7.0, 3.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.2, 2.4), xytext=(7.0, 1.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, title)


def d_pagerank(ax, title, rng):
    n = 6
    pos = {i: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)}
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3), (1, 4)]
    for a, b in edges:
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2,
                                    connectionstyle="arc3,rad=0.08"))
    pr = np.array([0.12, 0.18, 0.22, 0.2, 0.16, 0.12])
    for i in range(n):
        ax.plot(*pos[i], "o", color=TEAL, ms=12 + 40 * pr[i])
        ax.text(pos[i][0] * 1.25, pos[i][1] * 1.25, f"{pr[i]:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -1.55, "stationary mass of random walk + teleport", ha="center", fontsize=9)
    style(ax, title)


def d_missingness(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, (t, c) in enumerate([("MCAR\n⊥ all", TEAL), ("MAR\n⊥ | observed", GOLD), ("MNAR\ndepends on missing", ROSE)]):
        tc = INK if c == GOLD else "white"
        box(ax, 0.6 + i * 3.8, 1.5, 3.4, 2.0, t, fc=c, fs=11, tc=tc)
    style(ax, title)


def d_metric_gates(ax, title, rng):
    metrics = ["AUROC", "ECE", "Brier", "recall@5%", "latency"]
    scores = [0.91, 0.04, 0.12, 0.62, 0.88]
    floors = [0.85, 0.08, 0.18, 0.55, 0.80]
    # normalize for display: higher better except ECE/Brier already inverted in pass
    ax.barh(metrics, scores, color=TEAL, edgecolor=DEEP, alpha=0.85)
    for i, (s, f) in enumerate(zip(scores, floors)):
        ax.plot([f, f], [i - 0.4, i + 0.4], color=ROSE, lw=2)
    ax.set_xlabel("metric value (red = gate)")
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, title)


def d_entropy_strip(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, t in enumerate(["H(p)", "H(p,q)", "KL(p‖q)", "JS", "CE loss"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, t, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, title)


# --- Cycle 204 ---

def d_pseudoinverse(ax, title, rng):
    s = np.array([5, 2, 0.8, 0.0, 0.0])
    s_plus = np.array([1 / x if x > 1e-9 else 0 for x in s])
    x = np.arange(len(s))
    ax.bar(x - 0.15, s, 0.3, color=TEAL, label="σ")
    ax.bar(x + 0.15, s_plus, 0.3, color=GOLD, label="σ⁺ (1/σ or 0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"i={i+1}" for i in x])
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, title)


def d_data_card(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fields = ["provenance", "schema", "PII map", "label def", "splits", "limits"]
    for i, f in enumerate(fields):
        box(ax, 0.4 + (i % 3) * 3.8, 2.7 - (i // 3) * 2.0, 3.5, 1.5, f, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, title)


def d_margin_bound(ax, title, rng):
    gamma = np.linspace(0.05, 1.0, 40)
    # sketch: generalization ~ complexity/(n γ²)
    bound = 2.0 / (gamma ** 2) / 100 + 0.05
    ax.plot(gamma, bound, color=TEAL, lw=2.2)
    ax.set_xlabel("margin γ")
    ax.set_ylabel("gen. bound sketch")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_parallel_coords(ax, title, rng):
    n, d = 30, 5
    X = rng.normal(0, 1, (n, d))
    X[:10] += np.array([1.5, 0.5, -0.5, 0, 1])
    for i in range(n):
        ax.plot(range(d), X[i], color=TEAL if i < 10 else SLATE, alpha=0.45, lw=1)
    ax.set_xticks(range(d))
    ax.set_xticklabels([f"f{j}" for j in range(d)])
    ax.set_ylabel("scaled value")
    style(ax, title)


def d_crps(ax, title, rng):
    x = np.linspace(-3, 3, 300)
    # forecast CDF vs observation
    F = 1 / (1 + np.exp(-1.5 * (x + 0.2)))
    y = 0.5
    ax.plot(x, F, color=TEAL, lw=2, label="forecast CDF F")
    ax.axvline(y, color=ROSE, ls="--", label="observation y")
    ax.fill_between(x, F, np.where(x >= y, 1, 0), color=GOLD, alpha=0.3, label="CRPS integrand")
    ax.set_xlabel("x")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_cluster_stability(ax, title, rng):
    boots = np.arange(1, 21)
    ari = 0.75 + 0.08 * np.sin(boots) + rng.normal(0, 0.03, len(boots))
    ax.plot(boots, ari, "o-", color=TEAL, lw=1.8)
    ax.axhline(np.mean(ari), color=GOLD, ls="--", label=f"mean ARI={np.mean(ari):.2f}")
    ax.axhline(0.7, color=ROSE, ls=":", label="stability floor")
    ax.set_xlabel("bootstrap resample")
    ax.set_ylabel("ARI vs reference")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_bm25_terms(ax, title, rng):
    terms = ["tf sat", "IDF", "dl norm", "k1", "b"]
    vals = [0.9, 1.2, 0.7, 1.0, 0.75]
    ax.bar(terms, vals, color=[TEAL, DEEP, GOLD, MINT, ROSE], edgecolor=DEEP)
    ax.set_ylabel("relative weight sketch")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, title)


def d_interaction_cross(ax, title, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.8, 2.8, 1.3, "user_id", fc=TEAL, fs=11)
    box(ax, 3.6, 2.8, 2.8, 1.3, "item_id", fc=DEEP, fs=11)
    box(ax, 6.8, 2.8, 2.8, 1.3, "cross\nfeature", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(6.8, 3.45), xytext=(6.4, 3.45), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.5))
    ax.text(5, 1.3, "cartesian / hashed crosses explode cardinality", ha="center", fontsize=10, color=INK)
    style(ax, title)


def d_scree_cattell(ax, title, rng):
    k = np.arange(1, 15)
    ev = 8 * np.exp(-0.5 * (k - 1)) + 0.4
    ax.plot(k, ev, "o-", color=TEAL, lw=2)
    # elbow marker
    ax.axvline(4, color=ROSE, ls="--", label="Cattell elbow sketch")
    ax.fill_between(k, 0, ev, where=k <= 4, color=MINT, alpha=0.25)
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_elastic_net(ax, title, rng):
    t = np.linspace(-2, 2, 400)
    # L1 diamond vs L2 circle vs elastic
    ax.plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)), color=GOLD, lw=1.8, label="L2 ball")
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax.plot(diamond[:, 0], diamond[:, 1], color=TEAL, lw=1.8, label="L1 ball")
    # elastic approx superellipse
    p = 1.4
    th = np.linspace(0, 2 * np.pi, 300)
    r = (np.abs(np.cos(th)) ** p + np.abs(np.sin(th)) ** p) ** (-1 / p)
    ax.plot(r * np.cos(th), r * np.sin(th), color=ROSE, lw=2, label="elastic-net mix")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_threshold_utility(ax, title, rng):
    thr = np.linspace(0, 1, 100)
    # utility = TP benefit - FP cost tradeoff sketch
    tp = 1 - thr ** 0.7
    fp = (1 - thr) ** 1.2
    util = 2.0 * tp - 1.2 * fp
    ax.plot(thr, util, color=TEAL, lw=2.2, label="utility")
    i = int(np.argmax(util))
    ax.axvline(thr[i], color=GOLD, ls="--", label=f"argmax≈{thr[i]:.2f}")
    ax.set_xlabel("classification threshold")
    ax.set_ylabel("expected utility sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_attention_entropy(ax, title, rng):
    heads = np.arange(1, 9)
    H = rng.uniform(0.5, 2.8, size=8)
    H[2] = 0.35  # collapsed head
    H[5] = 2.9
    colors = [ROSE if h < 0.6 else TEAL for h in H]
    ax.bar(heads, H, color=colors, edgecolor=DEEP)
    ax.axhline(0.6, color=GOLD, ls="--", label="collapse floor")
    ax.set_xlabel("attention head")
    ax.set_ylabel("entropy of weights")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, title)


def d_swav(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.8, 3.0, 1.4, "view 1\ncodes", fc=TEAL, fs=11)
    box(ax, 4.5, 2.8, 3.0, 1.4, "view 2\ncodes", fc=DEEP, fs=11)
    box(ax, 8.5, 2.8, 3.0, 1.4, "swap\nprediction", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(8.5, 3.7), xytext=(3.5, 3.7), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.4))
    ax.annotate("", xy=(8.5, 3.3), xytext=(7.5, 3.3), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.4))
    ax.text(6, 1.3, "SwAV: swapped assignment prediction", ha="center", fontsize=10, color=INK)
    style(ax, title)


def d_relative_bias(ax, title, rng):
    # ALiBi linear bias vs distance
    d = np.arange(0, 40)
    for m, c, lab in [(-1, TEAL, "head steep"), (-0.5, GOLD, "head mid"), (-0.1, ROSE, "head flat")]:
        ax.plot(d, m * d, color=c, lw=2, label=lab)
    ax.set_xlabel("query-key distance")
    ax.set_ylabel("attention bias")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_ucb(ax, title, rng):
    t = np.arange(1, 80)
    mean = 0.4 + 0.1 * (1 - np.exp(-t / 20))
    bonus = 1.2 * np.sqrt(np.log(t + 1) / (t * 0.3 + 1))
    ax.plot(t, mean, color=TEAL, lw=2, label="μ̂")
    ax.plot(t, mean + bonus, color=GOLD, lw=2, label="UCB = μ̂ + bonus")
    ax.fill_between(t, mean, mean + bonus, color=GOLD, alpha=0.2)
    ax.set_xlabel("pulls")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def d_structured_prune(ax, title, rng):
    # channel prune mask
    W = rng.normal(0, 1, (8, 12))
    norms = np.linalg.norm(W, axis=1)
    keep = norms > np.median(norms)
    ax.imshow(W, cmap="coolwarm", aspect="auto")
    for i, k in enumerate(keep):
        ax.add_patch(Rectangle((-0.5, i - 0.5), 12, 1, fill=False, edgecolor=TEAL if k else ROSE, lw=2))
    ax.set_xlabel("weights in channel")
    ax.set_ylabel("channel")
    ax.text(6, -1.3, "green=keep channel  red=structured drop", ha="center", fontsize=9, color=INK)
    style(ax, title)


def d_betweenness(ax, title, rng):
    # star vs path betweenness idea
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 2)
    ax.axis("off")
    # star
    hub = (0, 0)
    ax.plot(*hub, "o", color=ROSE, ms=16)
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        p = (1.3 * np.cos(ang), 1.3 * np.sin(ang))
        ax.plot([hub[0], p[0]], [hub[1], p[1]], color=SLATE, lw=1)
        ax.plot(*p, "o", color=TEAL, ms=8)
    ax.text(0, -1.8, "high betweenness hub", ha="center", color=ROSE, fontsize=9)
    # path
    for i in range(5):
        ax.plot(4 + i * 0.7, 0, "o", color=TEAL if i != 2 else ROSE, ms=10 if i != 2 else 14)
        if i < 4:
            ax.plot([4 + i * 0.7, 4 + (i + 1) * 0.7], [0, 0], color=SLATE, lw=1.5)
    ax.text(5.4, -1.8, "path bridge node", ha="center", color=ROSE, fontsize=9)
    style(ax, title)


def d_label_shift(ax, title, rng):
    labs = ["neg", "pos"]
    train = [0.8, 0.2]
    serve = [0.55, 0.45]
    x = np.arange(2)
    ax.bar(x - 0.15, train, 0.3, color=TEAL, label="train π")
    ax.bar(x + 0.15, serve, 0.3, color=GOLD, label="serve π")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("class prior")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, title)


def d_silent_fail_metric(ax, title, rng):
    t = np.arange(0, 60)
    auc = 0.9 + rng.normal(0, 0.005, len(t))
    volume = 100 - np.clip(t - 20, 0, None) * 1.5
    ax.plot(t, auc, color=TEAL, lw=2, label="AUROC (looks fine)")
    ax2 = ax.twinx()
    ax2.plot(t, volume, color=ROSE, lw=2, label="eligible volume")
    ax.set_xlabel("week")
    ax.set_ylabel("AUROC")
    ax2.set_ylabel("N scored")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.grid(True, alpha=0.25)


def d_glossary_regularizers(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, t in enumerate(["L1", "L2", "elastic", "dropout", "early stop"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, t, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, title)


C203 = [
    ("Matrix determinant as parallelogram volume", d_det_volume),
    ("IRB pathway decision tiles", d_irb_tiles),
    ("VC shattering of three points", d_vc_shatter),
    ("Horizon chart layered density", d_horizon),
    ("One-dimensional Wasserstein transport", d_wasserstein1d),
    ("Gap statistic null reference", d_gap_stat),
    ("FP-Growth header and tree mine", d_fpgrowth),
    ("Mutual information feature ranking", d_mi_rank),
    ("Sparse PCA cardinality path", d_sparse_pca),
    ("Ridge coefficient shrinkage path", d_ridge_path),
    ("Brier skill vs climatology", d_brier_skill),
    ("BatchNorm vs LayerNorm axes", d_bn_vs_ln),
    ("BYOL online target bootstrap", d_byol),
    ("Mel filterbank to encoder path", d_mel_path),
    ("PPO clipped ratio objective", d_ppo_ratio),
    ("LoRA low-rank adapter insert", d_lora),
    ("PageRank walk stationary mass", d_pagerank),
    ("MCAR MAR MNAR mechanisms", d_missingness),
    ("Champion metric gate bars", d_metric_gates),
    ("Glossary entropy KL CE strip", d_entropy_strip),
]

C204 = [
    ("Moore-Penrose singular reciprocal", d_pseudoinverse),
    ("Dataset card field tiles", d_data_card),
    ("Margin-based generalization sketch", d_margin_bound),
    ("Parallel coordinates class traces", d_parallel_coords),
    ("CRPS forecast CDF integrand", d_crps),
    ("Bootstrap cluster stability ARI", d_cluster_stability),
    ("BM25 term saturation parts", d_bm25_terms),
    ("Feature cross cardinality risk", d_interaction_cross),
    ("Cattell scree elbow rule", d_scree_cattell),
    ("Elastic-net constraint geometry", d_elastic_net),
    ("Threshold expected utility curve", d_threshold_utility),
    ("Attention head entropy collapse", d_attention_entropy),
    ("SwAV swapped code prediction", d_swav),
    ("ALiBi distance bias slopes", d_relative_bias),
    ("UCB exploration bonus decay", d_ucb),
    ("Structured channel prune mask", d_structured_prune),
    ("Betweenness hub vs path bridge", d_betweenness),
    ("Label shift prior mismatch", d_label_shift),
    ("Silent failure metric vs volume", d_silent_fail_metric),
    ("Glossary regularizer name strip", d_glossary_regularizers),
]


def embed(cycle: int, topics: list) -> None:
    for i, (title, fn) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        rng = np.random.default_rng(cycle * 1000 + i * 19 + 77)
        fn(ax, title, rng)
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


def main(cycles=None):
    mapping = {203: C203, 204: C204}
    for c in cycles or [203, 204]:
        embed(c, mapping[c])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main([int(x) for x in sys.argv[1].split(",")])
    else:
        main()
