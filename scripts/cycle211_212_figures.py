#!/usr/bin/env python3
"""Cycle-211/212 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Ellipse, Arc, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
TEAL, DEEP, INK, GOLD, SLATE, ROSE, MINT = (
    "#0d9488",
    "#0f766e",
    "#0f172a",
    "#c9a227",
    "#64748b",
    "#e11d48",
    "#14b8a6",
)
CHS = sorted(p.name for p in CURR.glob("*.md"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", name)


def box(ax, x, y, w, h, t, fc=TEAL, fs=9, tc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fc, edgecolor="none"
        )
    )
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


# ----- C211 -----


def d_frobenius(ax, t, rng):
    A = rng.normal(0, 1, (5, 5))
    B = A + rng.normal(0, 0.3, (5, 5))
    D = A - B
    im = ax.imshow(D, cmap="coolwarm", vmin=-1.5, vmax=1.5)
    frob = np.linalg.norm(D, "fro")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.12, f"‖A−B‖_F = {frob:.2f}", transform=ax.transAxes, ha="center", fontsize=11, color=INK)
    style(ax, t)


def d_dpia_flow(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    steps = ["data map", "risks", "controls", "residual", "sign-off"]
    for i, s in enumerate(steps):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, s, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_occam_razor(ax, t, rng):
    k = np.arange(1, 12)
    train = 1.2 * np.exp(-0.35 * k) + 0.05
    test = train + 0.02 * k**1.2
    ax.plot(k, train, "o-", color=GOLD, lw=2, label="train loss")
    ax.plot(k, test, "s-", color=TEAL, lw=2, label="test loss")
    i = int(np.argmin(test))
    ax.axvline(k[i], color=ROSE, ls="--", label=f"Occam pick k={k[i]}")
    ax.set_xlabel("model complexity k")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_connected_scatter(ax, t, rng):
    years = np.arange(2016, 2026)
    x = np.cumsum(rng.normal(0.3, 0.4, len(years)))
    y = np.cumsum(rng.normal(0.2, 0.5, len(years)))
    ax.plot(x, y, "-o", color=TEAL, lw=2, ms=8)
    for i, yr in enumerate(years):
        if i % 2 == 0:
            ax.text(x[i], y[i] + 0.15, str(yr), fontsize=7, color=INK, ha="center")
    ax.set_xlabel("metric A")
    ax.set_ylabel("metric B")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_beta_binomial(ax, t, rng):
    x = np.linspace(0.001, 0.999, 300)

    def beta_pdf(a, b):
        raw = x ** (a - 1) * (1 - x) ** (b - 1)
        return raw / np.trapezoid(raw, x)

    prior = beta_pdf(2, 8)
    post = beta_pdf(2 + 12, 8 + 28)  # 12 successes / 40 trials
    ax.plot(x, prior, color=SLATE, lw=1.8, label="Beta(2,8) prior")
    ax.plot(x, post, color=TEAL, lw=2.2, label="posterior after data")
    ax.axvline(12 / 40, color=GOLD, ls="--", label="MLE 12/40")
    ax.set_xlabel("θ")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_affinity_propagation(ax, t, rng):
    # responsibility/availability cartoon as message matrix heat
    R = rng.normal(0, 1, (8, 8))
    np.fill_diagonal(R, 2.5)
    ax.imshow(R, cmap="YlGn")
    ax.set_xlabel("exemplar k")
    ax.set_ylabel("point i")
    ax.text(0.5, 1.06, "affinity messages → exemplars", transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_closed_open_seq(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 5, 1.8, "closed: contiguous\nitemsets in window", fc=TEAL, fs=11)
    box(ax, 6.5, 2.5, 5, 1.8, "open: gapped\nsequences allowed", fc=GOLD, tc=INK, fs=11)
    ax.text(6, 1.2, "sequence pattern constraint styles", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_polynomial_features(ax, t, rng):
    x = np.linspace(-1.5, 1.5, 80)
    ax.plot(x, x, color=SLATE, lw=1.5, label="x")
    ax.plot(x, x**2, color=TEAL, lw=2, label="x²")
    ax.plot(x, x**3, color=GOLD, lw=2, label="x³")
    ax.set_xlabel("x")
    ax.set_ylabel("expanded feature")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_kernel_pca_rbf(ax, t, rng):
    # two rings in 2d
    th = np.linspace(0, 2 * np.pi, 60)
    ring1 = np.c_[np.cos(th), np.sin(th)] + rng.normal(0, 0.05, (60, 2))
    ring2 = 2.2 * np.c_[np.cos(th), np.sin(th)] + rng.normal(0, 0.05, (60, 2))
    ax.scatter(ring1[:, 0], ring1[:, 1], c=TEAL, s=18, label="class 0")
    ax.scatter(ring2[:, 0], ring2[:, 1], c=GOLD, s=18, label="class 1")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(0.5, -0.12, "RBF kernel PCA unfolds rings (intuition)", transform=ax.transAxes, ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_quantile_loss_pinball(ax, t, rng):
    u = np.linspace(-3, 3, 300)
    tau = 0.9
    loss = np.where(u >= 0, tau * u, (tau - 1) * u)
    ax.plot(u, loss, color=TEAL, lw=2.2, label=f"pinball τ={tau}")
    ax.plot(u, np.where(u >= 0, 0.5 * u, -0.5 * u), color=SLATE, ls="--", label="MAE (τ=0.5)")
    ax.set_xlabel("residual y − ŷ")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_detection_error_tradeoff(ax, t, rng):
    # DET curve: FNR vs FPR on normal deviate scale caricature
    fpr = np.linspace(0.01, 0.5, 40)
    fnr = 0.4 * np.exp(-3 * fpr) + 0.02
    ax.plot(fpr, fnr, color=TEAL, lw=2.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FPR")
    ax.set_ylabel("FNR")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_adamw_decouple(ax, t, rng):
    steps = np.arange(0, 60)
    # weight decay effect on ||w||
    w_l2 = 1.0 * np.exp(-0.01 * steps)
    w_adamw = 1.0 * (1 - 0.015) ** steps
    ax.plot(steps, w_l2, color=GOLD, lw=2, label="L2 in loss (coupled)")
    ax.plot(steps, w_adamw, color=TEAL, lw=2, label="AdamW decoupled decay")
    ax.set_xlabel("step")
    ax.set_ylabel("‖w‖ scale sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_jepa_predict(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.5, 3.2, 1.8, "context\nencoder", fc=TEAL, fs=11)
    box(ax, 4.4, 2.5, 3.2, 1.8, "predictor\nlatent", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.4, 2.5, 3.2, 1.8, "target\nEMA enc", fc=DEEP, fs=11)
    ax.annotate("", xy=(4.4, 3.4), xytext=(3.6, 3.4), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.4, 3.4), xytext=(7.6, 3.4), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.5))
    ax.text(6, 1.3, "I-JEPA: predict representations, not pixels", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_flash_attention(ax, t, rng):
    # tiled QK blocks
    for i in range(4):
        for j in range(4):
            fc = TEAL if (i + j) % 2 == 0 else MINT
            ax.add_patch(Rectangle((j, 3 - i), 0.95, 0.95, facecolor=fc, edgecolor=DEEP, alpha=0.85))
            ax.text(j + 0.47, 3 - i + 0.47, f"T{i}{j}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.text(2, -0.2, "FlashAttention: tiled SRAM-friendly matmuls", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_impala_vtrace(ax, t, rng):
    k = np.arange(1, 15)
    c = np.cumprod(np.clip(0.95 - 0.03 * rng.random(14), 0.5, 1.0))
    ax.plot(k, c, "o-", color=TEAL, lw=2, label="v-trace product c̄")
    ax.axhline(0.3, color=GOLD, ls="--", label="truncation floor")
    ax.set_xlabel("lookahead k")
    ax.set_ylabel("cumulative c")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_awq_scale(ax, t, rng):
    act = np.abs(rng.normal(0, 1, 32))
    w = rng.normal(0, 1, 32)
    # AWQ scales salient channels
    sal = act / (act.mean() + 1e-9)
    ax.bar(np.arange(32), np.abs(w), color=SLATE, alpha=0.5, label="|w|")
    ax.plot(np.arange(32), sal, color=TEAL, lw=2, label="activation scale")
    ax.set_xlabel("channel")
    ax.set_ylabel("magnitude / scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_label_prop_graph(ax, t, rng):
    # seed labels spread
    pos = {i: (np.cos(2 * np.pi * i / 8), np.sin(2 * np.pi * i / 8)) for i in range(8)}
    for i in range(8):
        j = (i + 1) % 8
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=SLATE, lw=1)
        ax.plot([pos[i][0], pos[(i + 2) % 8][0]], [pos[i][1], pos[(i + 2) % 8][1]], color=SLATE, lw=0.6, alpha=0.5)
    seeds = {0: TEAL, 4: GOLD}
    for i, p in pos.items():
        c = seeds.get(i, MINT)
        ax.plot(*p, "o", color=c, ms=14 if i in seeds else 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -1.4, "label propagation from seed nodes", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_unit_test_data(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["schema", "ranges", "leakage\nchecks", "slice\nmetrics", "CI"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_blue_green(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 1.8, 4.5, 2.2, "BLUE live v1\n100% traffic", fc=TEAL, fs=12)
    box(ax, 7, 1.8, 4.5, 2.2, "GREEN idle v2\nready flip", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(7, 2.9), xytext=(5, 2.9), arrowprops=dict(arrowstyle="<->", color=ROSE, lw=2))
    ax.text(6, 4.3, "atomic cutover / instant rollback", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_glossary_graphs(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["degree", "PageRank", "between.", "motif", "GNN"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


# ----- C212 -----


def d_trace_det_link(ax, t, rng):
    # for SPD: log det vs eigenvalues
    eigs = np.array([2.5, 1.8, 1.2, 0.7, 0.4])
    ax.bar(np.arange(1, 6), eigs, color=TEAL, edgecolor=DEEP)
    ax.axhline(np.exp(np.mean(np.log(eigs))), color=GOLD, ls="--", label="exp(mean log λ) geom mean")
    ax.text(0.02, 0.9, f"tr(A)={eigs.sum():.1f}  det(A)={np.prod(eigs):.2f}", transform=ax.transAxes, fontsize=10, color=INK)
    ax.set_xlabel("eigen-index")
    ax.set_ylabel("λ_i")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_consent_timeline(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.annotate("", xy=(11.5, 2), xytext=(0.5, 2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    for i, (lab, x) in enumerate([("enroll", 1.5), ("use window", 5), ("withdraw", 8.5), ("retain rule", 10.5)]):
        ax.plot(x, 2, "o", color=TEAL if i < 2 else ROSE, ms=12)
        ax.text(x, 2.6, lab, ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_peeling_off(ax, t, rng):
    # multiple testing peel-off significance
    ranks = np.arange(1, 21)
    p = np.sort(rng.uniform(0, 1, 20) ** 2)
    thr = 0.05 / (21 - ranks)
    ax.step(ranks, p, where="mid", color=TEAL, lw=2, label="ordered p")
    ax.plot(ranks, thr, color=GOLD, lw=1.8, label="Holm peel threshold")
    ax.set_xlabel("rank")
    ax.set_ylabel("p-value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_alluvial_flow(ax, t, rng):
    # simple alluvial-like flows
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    left = [("train", 4.5, TEAL), ("val", 1.5, GOLD), ("test", 1.0, ROSE)]
    right = [("fit", 3.5, TEAL), ("select", 2.0, GOLD), ("report", 1.5, DEEP)]
    y = 5.2
    for lab, h, c in left:
        ax.add_patch(Rectangle((0.5, y - h), 1.5, h, facecolor=c, edgecolor=DEEP, alpha=0.85))
        ax.text(1.25, y - h / 2, lab, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        y -= h + 0.15
    y = 5.2
    for lab, h, c in right:
        ax.add_patch(Rectangle((8, y - h), 1.5, h, facecolor=c, edgecolor=DEEP, alpha=0.85))
        ax.text(8.75, y - h / 2, lab, ha="center", va="center", color="white" if c != GOLD else INK, fontsize=9, fontweight="bold")
        y -= h + 0.15
    # ribbons
    ax.fill_between([2.2, 7.8], [4.5, 4.0], [1.0, 1.5], color=TEAL, alpha=0.2)
    ax.text(5, 5.5, "split → role alluvial", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_empirical_null(ax, t, rng):
    z = rng.normal(0, 1, 2000)
    # contamination
    z = np.concatenate([z, rng.normal(2.5, 0.5, 80)])
    ax.hist(z, bins=40, density=True, color=TEAL, alpha=0.7, edgecolor=DEEP)
    xs = np.linspace(-4, 5, 200)
    ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), color=GOLD, lw=2, label="N(0,1) theoretical")
    ax.plot(xs, np.exp(-0.5 * ((xs - 0.05) / 1.05) ** 2) / (1.05 * np.sqrt(2 * np.pi)), color=ROSE, lw=2, ls="--", label="empirical null fit")
    ax.legend(fontsize=8)
    ax.set_xlabel("z")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_birch_cf(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # CF tree nodes
    box(ax, 3.5, 4.5, 3, 1.0, "root CF", fc=GOLD, tc=INK, fs=11)
    for i, lab in enumerate(["CF1", "CF2", "CF3"]):
        box(ax, 0.8 + i * 3.1, 2.2, 2.6, 1.0, lab, fc=TEAL, fs=11)
        ax.plot([5, 2.1 + i * 3.1], [4.5, 3.2], color=SLATE, lw=1.2)
    for i in range(4):
        box(ax, 0.4 + i * 2.4, 0.4, 2.1, 0.9, f"leaf{i}", fc=DEEP, fs=9)
    ax.text(5, 5.8, "BIRCH clustering feature tree", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_maxgap_constraint(ax, t, rng):
    events = [(0, "A"), (1, "B"), (2, "C"), (5, "A"), (6, "B"), (10, "C")]
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(0, 3)
    for t0, e in events:
        ax.plot(t0, 1.5, "o", color=TEAL, ms=14)
        ax.text(t0, 1.9, e, ha="center", fontsize=11, fontweight="bold", color=INK)
    # maxgap=2 accepts first ABC, rejects sparse
    ax.annotate("", xy=(2, 1.2), xytext=(0, 1.2), arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2))
    ax.text(1, 0.7, "gaps≤2 ok", ha="center", color=GOLD, fontsize=9)
    ax.annotate("", xy=(10, 1.2), xytext=(5, 1.2), arrowprops=dict(arrowstyle="<->", color=ROSE, lw=2))
    ax.text(7.5, 0.7, "gap>2 reject", ha="center", color=ROSE, fontsize=9)
    ax.axis("off")
    style(ax, t)


def d_binning_strategies(ax, t, rng):
    x = rng.normal(0, 1, 300)
    ax.hist(x, bins=8, color=SLATE, alpha=0.4, density=True, label="equal width")
    qs = np.quantile(x, np.linspace(0, 1, 9))
    ax.hist(x, bins=qs, color=TEAL, alpha=0.45, density=True, label="equal frequency")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_autoencoder_bottleneck(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    widths = [3.0, 2.2, 1.2, 2.2, 3.0]
    labs = ["x", "h1", "z", "h2", "x̂"]
    x0 = 0.5
    for w, lab in zip(widths, labs):
        box(ax, x0, 2.5 - w / 2 + 1.2, 1.8, w * 0.7 + 0.8, lab, fc=TEAL if lab != "z" else GOLD, tc="white" if lab != "z" else INK, fs=11)
        x0 += 2.3
    style(ax, t)


def d_partial_residual(ax, t, rng):
    x = np.linspace(0, 10, 60)
    resid = 0.3 * np.sin(x) + rng.normal(0, 0.15, 60)
    ax.scatter(x, resid, c=TEAL, s=28, alpha=0.8)
    # lowess-like smooth
    ax.plot(x, 0.3 * np.sin(x), color=GOLD, lw=2.2, label="smooth of partial residual")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xlabel("x_j")
    ax.set_ylabel("r + β_j x_j")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mak_diagram(ax, t, rng):
    # multi-class accuracy vs kappa sketch
    po = np.linspace(0.4, 1.0, 40)
    pe = 0.25
    kappa = (po - pe) / (1 - pe)
    ax.plot(po, po, color=GOLD, lw=1.5, label="accuracy p_o")
    ax.plot(po, kappa, color=TEAL, lw=2.2, label="κ chance-corrected")
    ax.set_xlabel("observed agreement")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_se_block(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 1.8, 2.5, 1.8, "U\nfeatures", fc=SLATE, fs=11)
    box(ax, 3.5, 1.8, 2.5, 1.8, "squeeze\nglobal pool", fc=TEAL, fs=10)
    box(ax, 6.5, 1.8, 2.5, 1.8, "excite\nFC-σ", fc=DEEP, fs=11)
    box(ax, 9.5, 1.8, 2.2, 1.8, "scale\nchannels", fc=GOLD, tc=INK, fs=10)
    for i in range(3):
        ax.annotate("", xy=(3.5 + i * 3, 2.7), xytext=(3.0 + i * 3, 2.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, t)


def d_simsiam(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.6, 3.5, 1.6, "encoder+pred\nstopgrad branch", fc=TEAL, fs=11)
    box(ax, 8, 2.6, 3.5, 1.6, "encoder\nstop-grad", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(8, 3.4), xytext=(4, 3.4), arrowprops=dict(arrowstyle="<->", color=ROSE, lw=2))
    ax.text(6, 3.9, "cosine sym loss", ha="center", color=ROSE, fontsize=9, fontweight="bold")
    ax.text(6, 1.3, "SimSiam: collapse prevented by stop-grad", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_perceiver_latent(ax, t, rng):
    # many inputs, few latents
    for i in range(10):
        ax.plot(0.5, i * 0.35, "o", color=SLATE, ms=8)
    for j in range(4):
        ax.plot(3.5, 1.2 + j * 0.7, "o", color=TEAL, ms=12)
        for i in range(10):
            ax.plot([0.5, 3.5], [i * 0.35, 1.2 + j * 0.7], color=TEAL, alpha=0.08, lw=0.8)
    ax.text(0.5, -0.4, "inputs", ha="center", fontsize=9)
    ax.text(3.5, -0.4, "latents", ha="center", fontsize=9, color=TEAL)
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.8, 4)
    ax.axis("off")
    style(ax, t)


def d_muzero_model(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.5, 2.8, 1.6, "repr h", fc=TEAL, fs=12)
    box(ax, 4.0, 2.5, 2.8, 1.6, "dynamics g", fc=DEEP, fs=12)
    box(ax, 7.6, 2.5, 3.8, 1.6, "predict f → π,v", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(4.0, 3.3), xytext=(3.2, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(7.6, 3.3), xytext=(6.8, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.text(6, 1.3, "MuZero: learn model, plan in latent space", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_speculative_decode(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 4, 1.8, "draft model\nk tokens", fc=TEAL, fs=12)
    box(ax, 5.5, 2.5, 5.5, 1.8, "target verify\naccept/reject prefix", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(5.5, 3.4), xytext=(4.5, 3.4), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    style(ax, t)


def d_core_periphery(ax, t, rng):
    A = np.zeros((10, 10))
    A[:4, :4] = 1
    A[:4, 4:] = rng.random((4, 6)) > 0.6
    A[4:, :4] = A[:4, 4:].T
    np.fill_diagonal(A, 0)
    ax.imshow(A, cmap="Greens")
    ax.axhline(3.5, color=GOLD, lw=2)
    ax.axvline(3.5, color=GOLD, lw=2)
    ax.set_xlabel("node")
    ax.set_ylabel("node")
    ax.text(0.5, 1.06, "core–periphery block structure", transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_synthetic_data_risk(ax, t, rng):
    # utility vs privacy tradeoff
    eps = np.array([0.5, 1, 2, 4, 8])
    utility = 1 - np.exp(-0.4 * eps)
    risk = 0.1 + 0.08 * eps
    ax.plot(eps, utility, "o-", color=TEAL, lw=2, label="downstream utility")
    ax.plot(eps, risk, "s-", color=ROSE, lw=2, label="re-ID risk proxy")
    ax.set_xlabel("privacy ε (weaker →)")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_feature_flag(ax, t, rng):
    weeks = np.arange(0, 16)
    pct = np.clip(weeks * 8, 0, 100)
    ax.step(weeks, pct, where="post", color=TEAL, lw=2.2)
    ax.fill_between(weeks, 0, pct, step="post", color=TEAL, alpha=0.2)
    ax.set_xlabel("week")
    ax.set_ylabel("% cohort on new model")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_glossary_deploy(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["canary", "shadow", "blue/grn", "rollback", "SLO"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C211 = [
    ("Frobenius residual matrix heat", d_frobenius),
    ("DPIA residual risk workflow", d_dpia_flow),
    ("Occam complexity sweet spot", d_occam_razor),
    ("Connected scatter time path", d_connected_scatter),
    ("Beta-binomial posterior update", d_beta_binomial),
    ("Affinity propagation messages", d_affinity_propagation),
    ("Closed vs open sequence patterns", d_closed_open_seq),
    ("Polynomial feature expansion map", d_polynomial_features),
    ("Kernel PCA ring manifold", d_kernel_pca_rbf),
    ("Pinball quantile loss tau", d_quantile_loss_pinball),
    ("DET FPR-FNR tradeoff curve", d_detection_error_tradeoff),
    ("AdamW decoupled weight decay", d_adamw_decouple),
    ("I-JEPA latent prediction path", d_jepa_predict),
    ("FlashAttention tiled blocks", d_flash_attention),
    ("IMPALA v-trace truncation", d_impala_vtrace),
    ("AWQ activation-aware scales", d_awq_scale),
    ("Graph label propagation seeds", d_label_prop_graph),
    ("Data unit test checklist", d_unit_test_data),
    ("Blue-green deploy cutover", d_blue_green),
    ("Glossary graph mining strip", d_glossary_graphs),
]

C212 = [
    ("Trace and determinant of SPD", d_trace_det_link),
    ("Consent window timeline markers", d_consent_timeline),
    ("Holm peel-off thresholds", d_peeling_off),
    ("Train-val-test role alluvial", d_alluvial_flow),
    ("Empirical null z-density fit", d_empirical_null),
    ("BIRCH clustering feature tree", d_birch_cf),
    ("Maxgap sequence constraint", d_maxgap_constraint),
    ("Equal-width vs equal-frequency bins", d_binning_strategies),
    ("Autoencoder bottleneck diagram", d_autoencoder_bottleneck),
    ("Partial residual smooth check", d_partial_residual),
    ("Accuracy versus kappa agreement", d_mak_diagram),
    ("Squeeze-and-excitation channel scale", d_se_block),
    ("SimSiam stop-gradient branches", d_simsiam),
    ("Perceiver latent cross-attend", d_perceiver_latent),
    ("MuZero representation dynamics", d_muzero_model),
    ("Speculative decoding draft verify", d_speculative_decode),
    ("Core-periphery adjacency block", d_core_periphery),
    ("Synthetic data utility privacy", d_synthetic_data_risk),
    ("Feature-flag rollout ramp", d_feature_flag),
    ("Glossary deployment pattern strip", d_glossary_deploy),
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

    m = {211: C211, 212: C212}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [211, 212]
    for c in cycles:
        embed(c, m[c])
