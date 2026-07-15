#!/usr/bin/env python3
"""Cycle-225/226 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch, Arc, Ellipse, Wedge, Polygon

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
TEAL, DEEP, INK, GOLD, SLATE, ROSE, MINT = (
    "#0d9488", "#0f766e", "#0f172a", "#c9a227", "#64748b", "#e11d48", "#14b8a6",
)
CHS = sorted(p.name for p in CURR.glob("*.md"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", name)


def box(ax, x, y, w, h, t, fc=TEAL, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fc, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def d_cholesky_factor(ax, t, rng):
    L = np.array([[2.0, 0, 0, 0], [0.8, 1.5, 0, 0], [0.3, 0.5, 1.2, 0], [0.1, 0.2, 0.4, 0.9]])
    ax.imshow(L, cmap="Greens")
    for i in range(4):
        for j in range(4):
            if L[i, j] != 0:
                ax.text(j, i, f"{L[i, j]:.1f}", ha="center", va="center", fontsize=10, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(1.5, 4.2, "L lower-tri  ·  A = L Lᵀ", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_datasheet_fields(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for i, lab in enumerate(["motivation", "composition", "collection", "preprocessing", "uses", "distribution"]):
        r, c = divmod(i, 3)
        box(ax, 0.4 + c * 4.0, 3.5 - r * 2.4, 3.6, 1.8, lab, fc=TEAL if (i % 2 == 0) else DEEP, fs=11)
    style(ax, t)


def d_covering_number(ax, t, rng):
    eps = np.logspace(-2, 0, 40)
    N = (1 / eps) ** 2.5
    ax.loglog(eps, N, color=TEAL, lw=2.2)
    ax.set_xlabel("ε")
    ax.set_ylabel("covering number N(ε) sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_splom_lower(ax, t, rng):
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.axis("off")
    for i in range(3):
        for j in range(3):
            x0, y0 = 0.4 + j * 2.9, 6.2 - i * 2.9
            ax.add_patch(Rectangle((x0, y0), 2.5, 2.5, fill=False, edgecolor=SLATE, lw=1))
            if i == j:
                xs = np.linspace(x0 + 0.2, x0 + 2.3, 30)
                dens = np.exp(-0.5 * ((xs - (x0 + 1.25)) / 0.4) ** 2)
                ax.fill_between(xs, y0 + 0.2, y0 + 0.2 + dens * 1.5, color=TEAL, alpha=0.7)
            elif i > j:
                ax.scatter(x0 + 1.25 + rng.normal(0, 0.4, 25), y0 + 1.25 + rng.normal(0, 0.4, 25), c=TEAL, s=8, alpha=0.7)
            else:
                ax.text(x0 + 1.25, y0 + 1.25, "ρ", ha="center", va="center", color=GOLD, fontsize=14, fontweight="bold")
    style(ax, t)


def d_hpd_interval(ax, t, rng):
    th = np.linspace(-4, 4, 400)
    p = 0.6 * np.exp(-0.5 * ((th + 1.2) / 0.5) ** 2) + 0.4 * np.exp(-0.5 * ((th - 1.5) / 0.6) ** 2)
    p = p / np.trapezoid(p, th)
    ax.fill_between(th, 0, p, color=TEAL, alpha=0.25)
    ax.plot(th, p, color=DEEP, lw=2)
    # HPD: highest density region above threshold
    thr = 0.12
    mask = p >= thr
    ax.fill_between(th, 0, p, where=mask, color=GOLD, alpha=0.55, label="HPD region")
    ax.axhline(thr, color=ROSE, ls="--", lw=1.2)
    ax.set_xlabel("θ")
    ax.set_ylabel("posterior")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_spectral_clustering_cut(ax, t, rng):
    # two moons-ish then fiedler sign
    th = np.linspace(0, np.pi, 40)
    a = np.c_[np.cos(th), np.sin(th)] + rng.normal(0, 0.05, (40, 2))
    b = np.c_[1 - np.cos(th), 1 - np.sin(th) - 0.5] + rng.normal(0, 0.05, (40, 2))
    ax.scatter(a[:, 0], a[:, 1], c=TEAL, s=28, label="cluster A")
    ax.scatter(b[:, 0], b[:, 1], c=GOLD, s=28, label="cluster B")
    ax.plot([0.5, 0.5], [-0.3, 1.3], "--", color=ROSE, lw=2, label="Fiedler cut")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_apriori_levels(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    levels = [
        (0.5, 4.5, ["A", "B", "C", "D", "E"], TEAL),
        (1.5, 2.8, ["AB", "AC", "BC", "BD"], DEEP),
        (3.0, 1.1, ["ABC", "ABD"], GOLD),
    ]
    for x0, y, items, c in levels:
        for i, lab in enumerate(items):
            box(ax, x0 + i * 2.2, y, 2.0, 1.0, lab, fc=c, tc=("white" if c != GOLD else INK), fs=10)
    ax.text(6, 5.5, "Apriori level-wise L1→L2→L3", ha="center", fontsize=10, color=SLATE)
    style(ax, t)


def d_mutual_info_heatmap(ax, t, rng):
    n = 7
    MI = np.abs(rng.normal(0, 0.3, (n, n)))
    MI = (MI + MI.T) / 2
    np.fill_diagonal(MI, 1.0)
    ax.imshow(MI, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("feature")
    ax.set_ylabel("feature")
    style(ax, t)


def d_kernel_pca_map(ax, t, rng):
    th = np.linspace(0, 2 * np.pi, 80)
    r = 1 + 0.3 * rng.normal(0, 1, 80)
    x = r * np.cos(th)
    y = r * np.sin(th)
    # "feature map" radial
    z = x**2 + y**2
    sc = ax.scatter(x, y, c=z, cmap="viridis", s=28)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(0, -1.8, "color ∝ φ radius (kernel PCA cue)", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_loess_smooth(ax, t, rng):
    x = np.sort(rng.uniform(0, 10, 60))
    y = np.sin(x / 1.5) + rng.normal(0, 0.25, 60)
    # simple moving local mean as LOESS sketch
    k = 8
    ys = np.array([y[max(0, i - k) : i + k + 1].mean() for i in range(len(x))])
    ax.scatter(x, y, c=TEAL, s=18, alpha=0.6)
    ax.plot(x, ys, color=GOLD, lw=2.2, label="LOESS sketch")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_focal_loss(ax, t, rng):
    p = np.linspace(0.01, 1, 100)
    for g, c in [(0, SLATE), (1, GOLD), (2, TEAL), (5, ROSE)]:
        fl = -((1 - p) ** g) * np.log(p)
        ax.plot(p, fl, color=c, lw=2, label=f"γ={g}")
    ax.set_xlabel("p_t")
    ax.set_ylabel("FL(p_t)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_batch_renorm(ax, t, rng):
    steps = np.arange(0, 50)
    r = 1 + 0.3 * np.sin(steps / 4) * np.exp(-steps / 30)
    d = 0.2 * np.cos(steps / 5) * np.exp(-steps / 25)
    ax.plot(steps, r, color=TEAL, lw=2, label="r clip gate")
    ax.plot(steps, d, color=GOLD, lw=2, label="d clip gate")
    ax.axhline(1, color=SLATE, ls=":", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("BatchRenorm correction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_swav_codes(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.2, 3.0, 1.8, "views\n(multi-crop)", fc=TEAL, fs=11)
    box(ax, 4.0, 2.2, 3.5, 1.8, "online codes\n(Sinkhorn)", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.0, 2.2, 3.5, 1.8, "swap predict\ncodes", fc=DEEP, fs=11)
    for x0, x1 in [(3.4, 4.0), (7.5, 8.0)]:
        ax.annotate("", xy=(x1, 3.1), xytext=(x0, 3.1), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_clip_contrast(ax, t, rng):
    n = 6
    S = rng.normal(0, 0.5, (n, n))
    np.fill_diagonal(S, 2.5)
    ax.imshow(S, cmap="coolwarm", vmin=-1, vmax=3)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=GOLD, lw=2))
    ax.set_xlabel("text")
    ax.set_ylabel("image")
    ax.text(2.5, -1.0, "CLIP: diagonal positives in batch", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_distributional_rl(ax, t, rng):
    z = np.linspace(-2, 8, 200)
    # two return distributions
    p1 = np.exp(-0.5 * ((z - 1) / 0.8) ** 2)
    p2 = np.exp(-0.5 * ((z - 4) / 1.2) ** 2)
    p1, p2 = p1 / p1.sum(), p2 / p2.sum()
    ax.fill_between(z, 0, p1, color=TEAL, alpha=0.5, label="Z(s,a1)")
    ax.fill_between(z, 0, p2, color=GOLD, alpha=0.5, label="Z(s,a2)")
    ax.set_xlabel("return")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_speculative_decoding(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.4, 3.5, 3.5, 1.8, "draft model\n(small, fast)", fc=TEAL, fs=11)
    box(ax, 4.4, 3.5, 3.5, 1.8, "propose K\ntokens", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.4, 3.5, 3.3, 1.8, "target verify\naccept/reject", fc=DEEP, fs=10)
    box(ax, 3.0, 1.0, 6, 1.5, "accepted prefix → speedup", fc=MINT, fs=12)
    for x0, x1 in [(3.9, 4.4), (7.9, 8.4)]:
        ax.annotate("", xy=(x1, 4.4), xytext=(x0, 4.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_node2vec_biased(ax, t, rng):
    # walk bias p,q schematic
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    pos = {0: (2, 3), 1: (5, 3), 2: (7.5, 4.5), 3: (7.5, 1.5), 4: (5, 5.2)}
    edges = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 4)]
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=SLATE, lw=1.5)
    for i, (x, y) in pos.items():
        ax.plot(x, y, "o", color=TEAL if i != 0 else GOLD, ms=14)
    ax.annotate("", xy=pos[1], xytext=pos[0], arrowprops=dict(arrowstyle="->", color=ROSE, lw=2.5))
    ax.text(3.3, 3.3, "t→v", color=ROSE, fontsize=10, fontweight="bold")
    ax.text(6.2, 3.8, "1/p return", fontsize=9, color=GOLD)
    ax.text(6.5, 2.2, "1/q outward", fontsize=9, color=DEEP)
    style(ax, t)


def d_freshness_watermark(ax, t, rng):
    t_ = np.arange(0, 48)
    age = np.maximum(0, t_ - 12) * 0.3 + rng.normal(0, 0.1, 48)
    ax.plot(t_, age, color=TEAL, lw=2)
    ax.axhline(2.0, color=GOLD, ls="--", label="max age SLA")
    ax.fill_between(t_, age, 2.0, where=age > 2.0, color=ROSE, alpha=0.3)
    ax.set_xlabel("hour")
    ax.set_ylabel("feature age")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_capacity_plan(ax, t, rng):
    weeks = np.arange(1, 13)
    demand = 50 + 3 * weeks + rng.normal(0, 2, 12)
    cap = np.full(12, 80.0)
    ax.plot(weeks, demand, "o-", color=TEAL, lw=2, label="load")
    ax.plot(weeks, cap, color=GOLD, ls="--", lw=2, label="capacity")
    ax.fill_between(weeks, demand, cap, where=demand < cap, color=MINT, alpha=0.2)
    ax.set_xlabel("week")
    ax.set_ylabel("QPS units")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gloss_graph_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["PageRank", "node2vec", "GCN", "GraphSAGE", "GAT"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_pseudoinverse(ax, t, rng):
    s = np.array([5.0, 2.0, 0.8, 1e-3, 1e-6])
    s_pinv = np.array([1 / x if x > 1e-2 else 0 for x in s])
    x = np.arange(len(s))
    w = 0.35
    ax.bar(x - w / 2, s, width=w, color=SLATE, label="σ")
    ax.bar(x + w / 2, s_pinv, width=w, color=TEAL, label="σ⁺ (thresh)")
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_eval_harness(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["prompts", "model\nunder test", "scorers", "report\ndash"]):
        box(ax, 0.4 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.1, 6.1, 9.1]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_rademacher_complexity(ax, t, rng):
    n = np.arange(20, 500, 10)
    rad = 2.5 / np.sqrt(n)
    ax.plot(n, rad, color=TEAL, lw=2.2)
    ax.set_xlabel("n")
    ax.set_ylabel("Rademacher complexity sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_violin_split(ax, t, rng):
    data = [rng.normal(0, 1, 150), rng.normal(0.5, 0.7, 150), rng.normal(-0.3, 1.2, 150)]
    parts = ax.violinplot(data, positions=[1, 2, 3], showmeans=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(TEAL)
        b.set_alpha(0.7)
    if "cmeans" in parts:
        parts["cmeans"].set_color(GOLD)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["A", "B", "C"])
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_pit_histogram(ax, t, rng):
    """Probability integral transform calibration hist."""
    # well-calibrated → flat; miscal → U-shape
    u_good = rng.uniform(0, 1, 400)
    ax.hist(u_good, bins=10, color=TEAL, edgecolor=DEEP, alpha=0.85, density=True)
    ax.axhline(1.0, color=GOLD, ls="--", label="ideal flat")
    ax.set_xlabel("PIT u = F(y)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_affinity_propagation(ax, t, rng):
    # exemplar messages schematic
    pts = rng.normal(0, 1, (20, 2))
    exemplars = [2, 9, 15]
    ax.scatter(pts[:, 0], pts[:, 1], c=TEAL, s=40, alpha=0.7)
    ax.scatter(pts[exemplars, 0], pts[exemplars, 1], c=GOLD, s=120, marker="s", label="exemplars", zorder=5)
    for i, p in enumerate(pts):
        e = exemplars[np.argmin([np.linalg.norm(p - pts[e]) for e in exemplars])]
        if i not in exemplars:
            ax.plot([p[0], pts[e][0]], [p[1], pts[e][1]], color=SLATE, lw=0.6, alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_freeSpan(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["seq DB", "projected\nDB", "annotate\nf-list", "FreeSpan\npatterns"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_catboost_ordered(ax, t, rng):
    """Ordered target statistics by permutation prefix."""
    n = 12
    y = rng.binomial(1, 0.4, n)
    # expanding mean excluding current (ordered TS)
    means = []
    for i in range(n):
        means.append(y[:i].mean() if i > 0 else 0.5)
    ax.step(np.arange(n), means, where="post", color=TEAL, lw=2.2, label="ordered TS")
    ax.scatter(np.arange(n), y, c=GOLD, s=40, zorder=3, label="y")
    ax.set_xlabel("permutation order")
    ax.set_ylabel("prior / target stat")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_multidimensional_scaling(ax, t, rng):
    # classic MDS stress vs dim
    d = np.arange(1, 10)
    stress = 0.5 * np.exp(-0.4 * (d - 1)) + 0.02
    ax.plot(d, stress, "o-", color=TEAL, lw=2.2)
    ax.set_xlabel("embedding dim")
    ax.set_ylabel("Kruskal stress sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_poisson_glm(ax, t, rng):
    x = np.linspace(0, 5, 40)
    mu = np.exp(0.3 + 0.4 * x)
    y = rng.poisson(mu)
    ax.scatter(x, y, c=TEAL, s=28, alpha=0.7)
    ax.plot(x, mu, color=GOLD, lw=2.2, label="μ=exp(xβ)")
    ax.set_xlabel("x")
    ax.set_ylabel("count y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_temperature_scaling(ax, t, rng):
    s = np.linspace(-4, 4, 100)
    for T, c in [(0.5, ROSE), (1.0, TEAL), (2.0, GOLD)]:
        p = 1 / (1 + np.exp(-s / T))
        ax.plot(s, p, color=c, lw=2, label=f"T={T}")
    ax.set_xlabel("logit")
    ax.set_ylabel("softmax (binary)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_muon_newton_schulz(ax, t, rng):
    """Newton–Schulz iterations toward orthogonal factor."""
    it = np.arange(0, 8)
    err = 0.8 * (0.35 ** it)
    ax.semilogy(it, err, "o-", color=TEAL, lw=2.2)
    ax.set_xlabel("Newton–Schulz iter")
    ax.set_ylabel("||XᵀX − I|| sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_vjepa2(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["video\ntubes", "context\nencoder", "predictor\n(masked)", "target\nEMA"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_musicgen(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 3.5, 1.8, "text / melody\ncondition", fc=TEAL, fs=11)
    box(ax, 4.5, 3.5, 3.5, 1.8, "transformer\non RVQ tokens", fc=GOLD, tc=INK, fs=10)
    box(ax, 8.5, 3.5, 3.2, 1.8, "audio\ndecode", fc=DEEP, fs=12)
    # multi-stream tokens
    for i in range(4):
        ax.plot([4.7, 7.7], [2.8 - i * 0.4, 2.8 - i * 0.4], color=MINT if i % 2 == 0 else TEAL, lw=3)
    ax.text(6.2, 1.0, "interleaved codebooks", ha="center", fontsize=9, color=SLATE)
    for x0, x1 in [(4.0, 4.5), (8.0, 8.5)]:
        ax.annotate("", xy=(x1, 4.4), xytext=(x0, 4.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_mcts_puct(ax, t, rng):
    """PUCT score: Q + U exploration term."""
    n = np.arange(1, 40)
    Q = 0.4 + 0.1 * np.sin(n / 3)
    U = 1.5 * np.sqrt(n.sum() if False else 100) * (0.2 / (1 + n))  # simplified
    U = 1.2 * np.sqrt(np.log(n + 1) + 1) / (1 + n) * 5
    ax.plot(n, Q, color=GOLD, lw=2, label="Q")
    ax.plot(n, U, color=SLATE, lw=1.8, label="U")
    ax.plot(n, Q + U, color=TEAL, lw=2.2, label="PUCT")
    ax.set_xlabel("N(s,a)")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gguf_quants(ax, t, rng):
    bits = [2, 3, 4, 5, 6, 8]
    ppl = [22, 16, 12.5, 11.2, 10.8, 10.5]
    ax.plot(bits, ppl, "o-", color=TEAL, lw=2.2)
    ax.set_xlabel("GGUF quant bits")
    ax.set_ylabel("perplexity sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_label_prop_graph(ax, t, rng):
    # harmonic function on path graph
    n = 12
    y = np.zeros(n)
    y[0] = 1
    y[-1] = 0
    # linear interpolate harmonic on path
    y = np.linspace(1, 0, n)
    ax.plot(np.arange(n), y, "o-", color=TEAL, lw=2)
    ax.scatter([0, n - 1], [1, 0], c=GOLD, s=100, zorder=5, label="clamped seeds")
    ax.set_xlabel("node along path")
    ax.set_ylabel("soft label")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_feature_store_online(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.2, 2.6, 1.8, "stream\nevents", fc=SLATE, fs=11)
    box(ax, 3.4, 2.2, 2.6, 1.8, "online\nstore", fc=TEAL, fs=12)
    box(ax, 6.4, 2.2, 2.6, 1.8, "model\nserving", fc=GOLD, tc=INK, fs=11)
    box(ax, 9.4, 2.2, 2.3, 1.8, "low-lat\nlookup", fc=DEEP, fs=11)
    for x in [3.0, 6.0, 9.0]:
        ax.annotate("", xy=(x + 0.4, 3.1), xytext=(x, 3.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    style(ax, t)


def d_blameless_timeline(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    times = [(1, "detect"), (3.5, "page"), (6, "mitigate"), (8.5, "resolve"), (10.5, "learn")]
    ax.plot([0.5, 11.5], [2.5, 2.5], color=SLATE, lw=3)
    for x, lab in times:
        ax.plot(x, 2.5, "o", color=TEAL, ms=14)
        box(ax, x - 0.9, 3.2, 1.8, 1.0, lab, fc=DEEP, fs=9)
    style(ax, t)


def d_gloss_rl_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["DQN", "PPO", "SAC", "MuZero", "Dreamer"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C225 = [
    ("Cholesky lower-triangular factor", d_cholesky_factor),
    ("Dataset datasheet field map", d_datasheet_fields),
    ("Covering number vs epsilon", d_covering_number),
    ("SPLOM lower-triangle collage", d_splom_lower),
    ("Highest posterior density region", d_hpd_interval),
    ("Spectral clustering Fiedler cut", d_spectral_clustering_cut),
    ("Apriori level-wise itemsets", d_apriori_levels),
    ("Mutual information feature matrix", d_mutual_info_heatmap),
    ("Kernel PCA radial feature cue", d_kernel_pca_map),
    ("LOESS local smooth sketch", d_loess_smooth),
    ("Focal loss gamma family", d_focal_loss),
    ("Batch renormalization gates", d_batch_renorm),
    ("SwAV swapped code prediction", d_swav_codes),
    ("CLIP batch contrastive diagonal", d_clip_contrast),
    ("Distributional RL return laws", d_distributional_rl),
    ("Speculative decoding verify path", d_speculative_decoding),
    ("node2vec biased walk p-q", d_node2vec_biased),
    ("Feature freshness watermark SLA", d_freshness_watermark),
    ("Capacity plan load vs ceiling", d_capacity_plan),
    ("Glossary graph method strip", d_gloss_graph_strip),
]

C226 = [
    ("Pseudoinverse singular threshold", d_pseudoinverse),
    ("LLM eval harness stages", d_eval_harness),
    ("Rademacher complexity decay", d_rademacher_complexity),
    ("Split violin group comparison", d_violin_split),
    ("PIT calibration histogram", d_pit_histogram),
    ("Affinity propagation exemplars", d_affinity_propagation),
    ("FreeSpan projected sequence mine", d_freeSpan),
    ("CatBoost ordered target stats", d_catboost_ordered),
    ("MDS stress vs dimension", d_multidimensional_scaling),
    ("Poisson GLM mean curve", d_poisson_glm),
    ("Temperature scaling logits", d_temperature_scaling),
    ("Newton-Schulz orthogonal iters", d_muon_newton_schulz),
    ("V-JEPA video tube prediction", d_vjepa2),
    ("MusicGen RVQ token streams", d_musicgen),
    ("PUCT exploration score terms", d_mcts_puct),
    ("GGUF quant perplexity curve", d_gguf_quants),
    ("Harmonic label propagation path", d_label_prop_graph),
    ("Online feature store lookup", d_feature_store_online),
    ("Blameless incident timeline", d_blameless_timeline),
    ("Glossary RL algorithm strip", d_gloss_rl_strip),
]


def embed(cycle: int, topics: list) -> None:
    assert len(topics) == len(CHS)
    for i, (title, fn) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        rng = np.random.default_rng(cycle * 1000 + i * 23 + 5)
        try:
            fn(ax, title, rng)
        except Exception as e:
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

    m = {225: C225, 226: C226}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [225, 226]
    for c in cycles:
        embed(c, m[c])
