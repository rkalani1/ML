#!/usr/bin/env python3
"""Cycle-221/222 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

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


# ----- C221: novel teaching geometry -----


def d_householder_reflect(ax, t, rng):
    """Householder reflection: v → H v flips across hyperplane."""
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.8, 2.2)
    ax.set_aspect("equal")
    # hyperplane (mirror)
    ax.plot([-2, 2], [0, 0], color=SLATE, lw=2, ls="--", label="H plane")
    # normal u
    ax.annotate("", xy=(0, 1.4), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    ax.text(0.15, 1.2, "u", color=GOLD, fontsize=11, fontweight="bold")
    # vector x and reflected Hx
    x = np.array([1.4, 1.1])
    hx = np.array([1.4, -1.1])
    ax.annotate("", xy=x, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5))
    ax.annotate("", xy=hx, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2.5))
    ax.text(x[0] + 0.1, x[1] + 0.1, "x", color=TEAL, fontsize=11, fontweight="bold")
    ax.text(hx[0] + 0.1, hx[1] - 0.25, "Hx", color=ROSE, fontsize=11, fontweight="bold")
    ax.plot([x[0], hx[0]], [x[1], hx[1]], ":", color=MINT, lw=1.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    style(ax, t)


def d_consent_purpose_tree(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    box(ax, 4, 5.5, 4, 1.1, "lawful basis?", fc=GOLD, tc=INK, fs=11)
    box(ax, 0.5, 3.2, 3.2, 1.1, "consent", fc=TEAL, fs=11)
    box(ax, 4.4, 3.2, 3.2, 1.1, "contract", fc=DEEP, fs=11)
    box(ax, 8.3, 3.2, 3.2, 1.1, "legitimate\ninterest", fc=SLATE, fs=10)
    box(ax, 0.5, 0.8, 3.2, 1.1, "granular\npurpose", fc=MINT, fs=10)
    box(ax, 4.4, 0.8, 3.2, 1.1, "scope limit", fc=TEAL, fs=11)
    box(ax, 8.3, 0.8, 3.2, 1.1, "LIA / opt-out", fc=ROSE, fs=10)
    for x in [2.1, 6.0, 9.9]:
        ax.plot([6, x], [5.5, 4.3], color=SLATE, lw=1.2)
    for x in [2.1, 6.0, 9.9]:
        ax.plot([x, x], [3.2, 1.9], color=SLATE, lw=1.2)
    style(ax, t)


def d_pac_bayes_kl(ax, t, rng):
    """PAC-Bayes: bound ∝ √((KL(Q||P)+log n)/n)."""
    n = np.logspace(1.5, 4, 80)
    kl = 2.0
    bound = np.sqrt((kl + np.log(n)) / n)
    ax.semilogx(n, bound, color=TEAL, lw=2.2, label="KL=2")
    ax.semilogx(n, np.sqrt((0.5 + np.log(n)) / n), color=GOLD, lw=2, label="KL=0.5")
    ax.set_xlabel("n")
    ax.set_ylabel("PAC-Bayes sketch")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_joyplot_ridges(ax, t, rng):
    """Ridge / joyplot density stack (teaching geometry)."""
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 6)
    xs = np.linspace(-4, 4, 200)
    for i, mu in enumerate([-1.2, -0.4, 0.3, 1.0, 1.6]):
        dens = np.exp(-0.5 * ((xs - mu) / 0.55) ** 2)
        y0 = 0.6 + i * 0.95
        ax.fill_between(xs, y0, y0 + dens * 0.9, color=TEAL, alpha=0.35 + 0.08 * i, zorder=i)
        ax.plot(xs, y0 + dens * 0.9, color=DEEP, lw=1.2, zorder=i + 1)
    ax.set_xlabel("value")
    ax.set_yticks([])
    ax.set_ylabel("group ridges →")
    style(ax, t)


def d_power_posterior(ax, t, rng):
    """Power posterior π_t ∝ L^t · prior; t cools likelihood."""
    th = np.linspace(-3, 3, 300)
    prior = np.exp(-0.5 * (th / 1.2) ** 2)
    lik = np.exp(-0.5 * ((th - 1.0) / 0.4) ** 2)
    for tpow, c, lab in [(0.0, SLATE, "t=0 prior"), (0.3, GOLD, "t=0.3"), (1.0, TEAL, "t=1 full")]:
        post = prior * (lik**tpow)
        post = post / np.trapezoid(post, th)
        ax.plot(th, post, color=c, lw=2, label=lab)
    ax.set_xlabel("θ")
    ax.set_ylabel("power posterior")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_birch_cf_tree(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    box(ax, 4.5, 5.6, 3, 1, "root CF", fc=GOLD, tc=INK, fs=11)
    for i, x in enumerate([1.2, 5.0, 8.8]):
        box(ax, x, 3.4, 2.4, 1, f"CF{i+1}", fc=TEAL, fs=11)
        ax.plot([6, x + 1.2], [5.6, 4.4], color=SLATE, lw=1.2)
    for j, x in enumerate([0.4, 2.4, 4.4, 6.4, 8.4, 10.0]):
        box(ax, x, 1.0, 1.6, 0.9, f"L{j+1}", fc=DEEP if j % 2 else MINT, fs=9)
        parent = [1.2, 1.2, 5.0, 5.0, 8.8, 8.8][j]
        ax.plot([parent + 1.2, x + 0.8], [3.4, 1.9], color=SLATE, lw=1)
    ax.text(6, 0.3, "BIRCH CF-tree: compact clustering features", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_gsp_candidate_gen(ax, t, rng):
    """GSP candidate generation: join Fk × Fk → Ck+1 with support filter."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.3, 2.2, 2.8, 1.6, "F_k\nfrequent", fc=TEAL, fs=11)
    box(ax, 3.5, 2.2, 2.8, 1.6, "join +\nprune", fc=GOLD, tc=INK, fs=11)
    box(ax, 6.7, 2.2, 2.4, 1.6, "C_{k+1}", fc=DEEP, fs=12)
    box(ax, 9.4, 2.2, 2.3, 1.6, "scan\nsupport", fc=ROSE, fs=11)
    for x0, x1 in [(3.1, 3.5), (6.3, 6.7), (9.1, 9.4)]:
        ax.annotate("", xy=(x1, 3), xytext=(x0, 3), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_target_encoding_cv(ax, t, rng):
    """Out-of-fold target encoding avoids leakage."""
    folds = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    means = {0: 0.2, 1: 0.5, 2: 0.8}
    x = np.arange(9)
    y = [means[f] + rng.normal(0, 0.05) for f in folds]
    colors = [TEAL if f != 1 else ROSE for f in folds]
    ax.bar(x, y, color=colors, edgecolor=DEEP)
    ax.axvspan(2.5, 5.5, color=ROSE, alpha=0.12)
    ax.text(4, max(y) * 0.95, "encode fold 1\nfrom 0∪2 only", ha="center", fontsize=9, color=ROSE)
    ax.set_xlabel("row")
    ax.set_ylabel("OOF target mean")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_nmf_parts(ax, t, rng):
    """NMF: V ≈ W H non-negative parts-based factors."""
    W = np.abs(rng.normal(0.3, 0.2, (8, 3)))
    H = np.abs(rng.normal(0.3, 0.2, (3, 6)))
    V = W @ H
    fig_mats = [V, W, H]
    titles = ["V", "W", "H"]
    # draw on single axes as triptych with imshow offsets via separate sub-axes? use main ax collage
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # V
    ax_v = ax.inset_axes([0.05, 0.15, 0.28, 0.7])
    ax_v.imshow(V, cmap="Greens", aspect="auto")
    ax_v.set_title("V", fontsize=10, color=INK)
    ax_v.set_xticks([])
    ax_v.set_yticks([])
    ax.text(4.0, 2.5, "≈", fontsize=22, ha="center", va="center", color=INK, transform=ax.transData)
    ax_w = ax.inset_axes([0.38, 0.15, 0.22, 0.7])
    ax_w.imshow(W, cmap="Greens", aspect="auto")
    ax_w.set_title("W≥0", fontsize=10, color=TEAL)
    ax_w.set_xticks([])
    ax_w.set_yticks([])
    ax.text(7.6, 2.5, "×", fontsize=18, ha="center", va="center", color=INK, transform=ax.transData)
    ax_h = ax.inset_axes([0.72, 0.25, 0.24, 0.5])
    ax_h.imshow(H, cmap="YlOrBr", aspect="auto")
    ax_h.set_title("H≥0", fontsize=10, color=GOLD)
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    style(ax, t)


def d_partial_residual(ax, t, rng):
    """Partial residual plot: residual + β_j x_j vs x_j."""
    x = np.linspace(0, 10, 60)
    true = 0.4 * x + 0.08 * x**2
    y = true + rng.normal(0, 0.8, 60)
    # linear fit residual + linear term
    beta = np.polyfit(x, y, 1)
    lin = np.polyval(beta, x)
    partial = (y - lin) + beta[0] * x
    ax.scatter(x, partial, c=TEAL, s=28, alpha=0.75, edgecolors=DEEP, linewidths=0.4)
    ax.plot(x, beta[0] * x + 0.05 * x**2, color=GOLD, lw=2, label="smooth hint")
    ax.set_xlabel("x_j")
    ax.set_ylabel("partial residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_label_propagation(ax, t, rng):
    """Semi-supervised label propagation on graph."""
    pos = {
        0: (1, 3),
        1: (3, 4.5),
        2: (3, 1.5),
        3: (5.5, 3.5),
        4: (5.5, 1.8),
        5: (8, 3),
        6: (10, 4),
        7: (10, 2),
    }
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6), (5, 7), (3, 4)]
    labels = {0: 0, 1: 0, 6: 1, 7: 1}  # seed labels
    ax.set_xlim(0, 11)
    ax.set_ylim(0.5, 5.5)
    ax.axis("off")
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=SLATE, lw=1.2, zorder=1)
    for i, (x, y) in pos.items():
        if i in labels:
            c = TEAL if labels[i] == 0 else ROSE
            ax.plot(x, y, "o", ms=16, color=c, zorder=2)
            ax.text(x, y, str(labels[i]), ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=3)
        else:
            ax.plot(x, y, "o", ms=14, color="white", markeredgecolor=GOLD, markeredgewidth=2, zorder=2)
            ax.text(x, y - 0.45, "?", ha="center", color=GOLD, fontsize=9)
    ax.text(5.5, 5.2, "seeds → smooth on graph → unlabeled", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_gradient_noise_scale(ax, t, rng):
    """Gradient noise scale B_noise ≈ ||g||² / tr(Σ) vs batch size."""
    B = np.logspace(0, 3, 50)
    g2 = 1.0
    tr = 0.02
    scale = g2 / (tr + 1e-9)
    # critical batch where B ~ B_noise
    util = 1 - np.exp(-B / scale)
    ax.semilogx(B, util, color=TEAL, lw=2.2)
    ax.axvline(scale, color=GOLD, ls="--", lw=1.8, label=f"B_noise≈{scale:.0f}")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("data-parallel util sketch")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_dino_teacher_student(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 3.5, 1.6, "student\n(local crops)", fc=TEAL, fs=11)
    box(ax, 4.5, 3.5, 3.5, 1.6, "teacher\n(EMA global)", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.5, 3.5, 3.2, 1.6, "sharpen +\ncenter", fc=DEEP, fs=11)
    box(ax, 3.5, 0.8, 5, 1.5, "cross-entropy match (no labels)", fc=MINT, fs=11)
    ax.annotate("", xy=(4.5, 4.3), xytext=(4.0, 4.3), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(8.5, 4.3), xytext=(8.0, 4.3), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(6, 2.3), xytext=(2.25, 3.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate("", xy=(6, 2.3), xytext=(10.1, 3.5), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_perceiver_latent(ax, t, rng):
    """Perceiver: latent array cross-attends to byte array."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # byte array
    for i in range(10):
        ax.add_patch(Rectangle((0.4 + i * 0.55, 4.2), 0.45, 0.9, facecolor=SLATE if i % 2 else TEAL, edgecolor="none"))
    ax.text(3.1, 5.3, "byte / pixel array (long)", ha="center", fontsize=10, color=INK)
    # latent
    for i in range(4):
        ax.add_patch(Rectangle((3.5 + i * 1.2, 1.5), 1.0, 1.2, facecolor=GOLD, edgecolor=DEEP, lw=1))
        ax.text(4.0 + i * 1.2, 2.1, f"L{i}", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(6, 0.7, "latent bottleneck (fixed size)", ha="center", fontsize=10, color=INK)
    ax.annotate(
        "",
        xy=(6, 2.7),
        xytext=(3.1, 4.2),
        arrowprops=dict(arrowstyle="->", color=ROSE, lw=2),
    )
    ax.text(5.2, 3.5, "cross-attn", color=ROSE, fontsize=10, fontweight="bold")
    style(ax, t)


def d_option_critic(ax, t, rng):
    """Option-critic: options with initiation, policy, termination."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 4, 3.2, 1.3, "initiation I_ω", fc=TEAL, fs=11)
    box(ax, 4.4, 4, 3.2, 1.3, "intra-option π_ω", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.3, 4, 3.2, 1.3, "termination β_ω", fc=ROSE, fs=11)
    # timeline
    tline = np.linspace(1, 11, 40)
    ax.plot(tline, np.ones_like(tline) * 2.2, color=SLATE, lw=2)
    # option segments
    ax.plot([1.5, 4.5], [2.2, 2.2], color=TEAL, lw=8, solid_capstyle="round")
    ax.plot([4.5, 8.0], [2.2, 2.2], color=GOLD, lw=8, solid_capstyle="round")
    ax.plot([8.0, 10.5], [2.2, 2.2], color=DEEP, lw=8, solid_capstyle="round")
    ax.text(3, 2.8, "ω1", ha="center", color=TEAL, fontweight="bold")
    ax.text(6.2, 2.8, "ω2", ha="center", color=GOLD, fontweight="bold")
    ax.text(9.2, 2.8, "ω3", ha="center", color=DEEP, fontweight="bold")
    ax.text(6, 1.2, "temporal abstraction over primitive actions", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_gptq_hessian(ax, t, rng):
    """GPTQ: layer-wise quant using Hessian / OBS-style updates."""
    bits = np.array([2, 3, 4, 8])
    mse_rt = np.array([0.45, 0.18, 0.07, 0.015])
    mse_gptq = np.array([0.28, 0.09, 0.035, 0.012])
    ax.plot(bits, mse_rt, "s--", color=SLATE, lw=1.8, label="round-to-nearest")
    ax.plot(bits, mse_gptq, "o-", color=TEAL, lw=2.2, label="GPTQ / OBS-ish")
    ax.set_xlabel("bits")
    ax.set_ylabel("layer recon MSE sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_modularity_q(ax, t, rng):
    """Modularity Q heatmap of community block structure."""
    n = 12
    A = np.zeros((n, n))
    for block in (0, 4, 8):
        A[block : block + 4, block : block + 4] = rng.uniform(0.5, 1.0, (4, 4))
        np.fill_diagonal(A[block : block + 4, block : block + 4], 0.0)
    # weak between
    A += rng.uniform(0, 0.15, (n, n))
    A = np.triu(A, 1)
    A = A + A.T
    ax.imshow(A, cmap="Greens")
    for b in [4, 8]:
        ax.axhline(b - 0.5, color=GOLD, lw=1.5)
        ax.axvline(b - 0.5, color=GOLD, lw=1.5)
    ax.set_xlabel("node")
    ax.set_ylabel("node")
    ax.text(6, -1.5, "high Q when dense within, sparse between", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_schema_evolution(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    versions = [
        (0.4, "v1\nid,x", TEAL),
        (3.3, "v2\n+y", DEEP),
        (6.2, "v3\nrename", GOLD),
        (9.1, "v4\ndrop z", ROSE),
    ]
    for x, lab, c in versions:
        box(ax, x, 2.5, 2.5, 2.2, lab, fc=c, tc=("white" if c != GOLD else INK), fs=11)
    for x in [2.9, 5.8, 8.7]:
        ax.annotate("", xy=(x + 0.4, 3.6), xytext=(x, 3.6), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))
    ax.text(6, 1.2, "compat rules: expand → migrate → contract", ha="center", fontsize=10, color=SLATE)
    style(ax, t)


def d_decision_record(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.4, 3.8, 2.6, 1.5, "context", fc=SLATE, fs=12)
    box(ax, 3.3, 3.8, 2.6, 1.5, "options", fc=TEAL, fs=12)
    box(ax, 6.2, 3.8, 2.6, 1.5, "decision", fc=GOLD, tc=INK, fs=12)
    box(ax, 9.1, 3.8, 2.6, 1.5, "consequences", fc=ROSE, fs=11)
    box(ax, 2.5, 1.0, 7, 1.6, "ADR / model card link + owner + review date", fc=DEEP, fs=11)
    for x0 in [3.0, 5.9, 8.8]:
        ax.annotate("", xy=(x0 + 0.3, 4.55), xytext=(x0, 4.55), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, t)


def d_gloss_prob_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["prior", "likelihood", "posterior", "MAP", "ELBO"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


# ----- C222 -----


def d_svd_truncation(ax, t, rng):
    """Energy retained vs rank in truncated SVD."""
    s = np.array([12.0, 7.5, 3.2, 1.1, 0.4, 0.15, 0.08, 0.03])
    energy = np.cumsum(s**2) / np.sum(s**2)
    k = np.arange(1, len(s) + 1)
    ax.plot(k, energy, "o-", color=TEAL, lw=2.2)
    ax.axhline(0.9, color=GOLD, ls="--", label="90% energy")
    ax.axvline(np.searchsorted(energy, 0.9) + 1, color=ROSE, ls=":", label="k*")
    ax.set_xlabel("rank k")
    ax.set_ylabel("cumulative energy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dual_use_gate(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 3.5, 1.8, "capability\nrequest", fc=TEAL, fs=12)
    box(ax, 4.5, 3.5, 3.5, 1.8, "dual-use\nrisk screen", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.5, 4.2, 3.2, 1.2, "allow + monitor", fc=MINT, fs=11)
    box(ax, 8.5, 2.5, 3.2, 1.2, "deny / sandbox", fc=ROSE, fs=11)
    ax.annotate("", xy=(4.5, 4.4), xytext=(4.0, 4.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(8.5, 4.8), xytext=(8.0, 4.6), arrowprops=dict(arrowstyle="->", color=MINT, lw=1.5))
    ax.annotate("", xy=(8.5, 3.1), xytext=(8.0, 4.0), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.5))
    style(ax, t)


def d_compression_set(ax, t, rng):
    """Coreset / compression set size vs approx quality."""
    m = np.arange(5, 80, 2)
    err = 1.2 / np.sqrt(m) + 0.02
    ax.plot(m, err, color=TEAL, lw=2.2)
    ax.fill_between(m, err * 0.85, err * 1.15, color=TEAL, alpha=0.2)
    ax.set_xlabel("|coreset| m")
    ax.set_ylabel("approx error sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_hexbin_density(ax, t, rng):
    x = rng.normal(0, 1, 800)
    y = 0.6 * x + rng.normal(0, 0.7, 800)
    hb = ax.hexbin(x, y, gridsize=18, cmap="Greens", mincnt=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    style(ax, t)


def d_stein_discrepancy(ax, t, rng):
    """KSD-style score residual scatter (teaching)."""
    z = rng.normal(0, 1, (80, 2))
    score = -z + 0.3 * rng.normal(0, 1, (80, 2))  # imperfect model score
    ax.quiver(z[:, 0], z[:, 1], score[:, 0], score[:, 1], color=TEAL, angles="xy", scale_units="xy", scale=4, width=0.004)
    ax.scatter(z[:, 0], z[:, 1], c=GOLD, s=18, zorder=3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_hdbscan_condense(ax, t, rng):
    """HDBSCAN condensed tree lifetime bars."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    # condensed clusters as horizontal lifespan
    spans = [(0.5, 8.5, 5, TEAL), (1, 4, 3.8, DEEP), (4.5, 8, 3.8, GOLD), (1.5, 3.2, 2.2, MINT), (5, 7.5, 2.2, ROSE)]
    for x0, x1, y, c in spans:
        ax.plot([x0, x1], [y, y], color=c, lw=10, solid_capstyle="round")
    ax.set_xlabel("λ (density scale)")
    ax.set_ylabel("cluster id")
    ax.set_yticks([5, 3.8, 2.2])
    ax.set_yticklabels(["A", "B/C", "leaves"])
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_clospan_closed(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.2, 2.6, 1.8, "sequences", fc=SLATE, fs=11)
    box(ax, 3.4, 2.2, 2.6, 1.8, "projected\nDB", fc=TEAL, fs=11)
    box(ax, 6.4, 2.2, 2.6, 1.8, "closed\ncheck", fc=GOLD, tc=INK, fs=11)
    box(ax, 9.4, 2.2, 2.3, 1.8, "CloSpan\nset", fc=DEEP, fs=11)
    for x0, x1 in [(3.0, 3.4), (6.0, 6.4), (9.0, 9.4)]:
        ax.annotate("", xy=(x1, 3.1), xytext=(x0, 3.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    style(ax, t)


def d_hashing_trick(ax, t, rng):
    """Feature hashing: many tokens → fixed dim with collisions."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    tokens = ["cat", "dog", "taxi", "apple", "ml", "gpu"]
    for i, tok in enumerate(tokens):
        box(ax, 0.3, 5.1 - i * 0.85, 2.2, 0.7, tok, fc=SLATE, fs=9)
        h = (hash(tok) % 5 + 5) % 5  # stable-ish within process
        # use deterministic from rng seed content
        h = [0, 2, 1, 4, 3, 2][i]
        ax.annotate(
            "",
            xy=(5.5, 0.9 + h * 0.95),
            xytext=(2.6, 5.45 - i * 0.85),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2, alpha=0.8),
        )
    for j in range(5):
        box(ax, 5.5, 0.6 + j * 0.95, 2.8, 0.75, f"bin {j}", fc=TEAL if j % 2 == 0 else DEEP, fs=10)
    box(ax, 9.0, 2.0, 2.7, 2.0, "dim D\nfixed", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(9.0, 3), xytext=(8.3, 3), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_canonical_corr(ax, t, rng):
    """CCA: max corr of linear projections of two views."""
    t_ = np.linspace(0, 2 * np.pi, 60)
    view_a = np.c_[np.cos(t_), 0.3 * np.sin(t_)] + rng.normal(0, 0.05, (60, 2))
    view_b = np.c_[np.cos(t_ + 0.4), 0.3 * np.sin(t_ + 0.4)] + rng.normal(0, 0.05, (60, 2))
    # project onto first axis for scatter of scores
    sa = view_a[:, 0]
    sb = view_b[:, 0]
    ax.scatter(sa, sb, c=TEAL, s=30, alpha=0.75, edgecolors=DEEP, linewidths=0.3)
    # best fit line
    m, b = np.polyfit(sa, sb, 1)
    xs = np.linspace(sa.min(), sa.max(), 50)
    ax.plot(xs, m * xs + b, color=GOLD, lw=2, label=f"corr≈{np.corrcoef(sa, sb)[0,1]:.2f}")
    ax.set_xlabel("view-A score")
    ax.set_ylabel("view-B score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_m_estimator(ax, t, rng):
    """Huber ρ function vs squared loss."""
    r = np.linspace(-4, 4, 200)
    delta = 1.5
    huber = np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
    ax.plot(r, 0.5 * r**2, color=SLATE, lw=1.8, label="½ r²")
    ax.plot(r, huber, color=TEAL, lw=2.2, label=f"Huber δ={delta}")
    ax.axvline(delta, color=GOLD, ls="--", alpha=0.7)
    ax.axvline(-delta, color=GOLD, ls="--", alpha=0.7)
    ax.set_xlabel("residual r")
    ax.set_ylabel("ρ(r)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_ecoc_codebook(ax, t, rng):
    """ECOC code matrix for multiclass."""
    # 5 classes × 8 dichotomies
    C = rng.choice([-1, 0, 1], size=(5, 8), p=[0.35, 0.1, 0.55])
    cmap = plt.cm.colors.ListedColormap(["#e11d48", "#f1f5f9", "#0d9488"])
    ax.imshow(C, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("binary classifier")
    ax.set_ylabel("class")
    ax.set_yticks(range(5))
    ax.set_xticks(range(8))
    for i in range(5):
        for j in range(8):
            ax.text(j, i, {1: "+", -1: "−", 0: "·"}[C[i, j]], ha="center", va="center", fontsize=10, color=INK)
    style(ax, t)


def d_muon_optimizer(ax, t, rng):
    """Muon / orthogonalized momentum sketch: singular values flattened."""
    s_before = np.array([3.2, 1.8, 0.9, 0.4, 0.15])
    s_after = np.ones_like(s_before) * np.mean(s_before) ** 0  # unit-ish after Newton-Schulz style
    s_after = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    x = np.arange(len(s_before))
    w = 0.35
    ax.bar(x - w / 2, s_before, width=w, color=SLATE, label="raw update σ")
    ax.bar(x + w / 2, s_after, width=w, color=TEAL, label="after ortho / NS")
    ax.set_xlabel("singular index")
    ax.set_ylabel("magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_ijepa_mask(ax, t, rng):
    """I-JEPA: context encodes, predict target block embeddings."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    # grid of patches
    for i in range(6):
        for j in range(6):
            x, y = 0.5 + j * 0.7, 3.5 + i * 0.65
            # context vs target blocks
            if 2 <= i <= 4 and 2 <= j <= 4:
                c = ROSE  # target
            elif (i + j) % 3 == 0:
                c = "#e2e8f0"  # masked unused
            else:
                c = TEAL  # context
            ax.add_patch(Rectangle((x, y), 0.62, 0.58, facecolor=c, edgecolor="white", lw=0.8))
    box(ax, 5.5, 5.5, 4, 1.4, "context encoder", fc=TEAL, fs=11)
    box(ax, 5.5, 3.5, 4, 1.4, "target encoder\n(EMA)", fc=GOLD, tc=INK, fs=11)
    box(ax, 5.5, 1.2, 4, 1.4, "predict in\nrepresentation space", fc=DEEP, fs=10)
    ax.text(2.6, 2.8, "teal=context  rose=target", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_whisper_encoder_decoder(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["mel\nspectrogram", "encoder\nblocks", "decoder +\nprefix", "text\ntokens"]):
        box(ax, 0.4 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.1, 6.1, 9.1]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
    style(ax, t)


def d_muzero_model(ax, t, rng):
    """MuZero: representation → dynamics → prediction."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.4, 3.5, 2.8, 1.8, "h: obs→s", fc=TEAL, fs=12)
    box(ax, 3.6, 3.5, 2.8, 1.8, "g: s,a→s',r", fc=GOLD, tc=INK, fs=11)
    box(ax, 6.8, 3.5, 2.8, 1.8, "f: s→p,v", fc=DEEP, fs=12)
    box(ax, 3.6, 1.0, 5.5, 1.6, "MCTS over learned model (not env)", fc=MINT, fs=11)
    ax.annotate("", xy=(3.6, 4.4), xytext=(3.2, 4.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(6.8, 4.4), xytext=(6.4, 4.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(6.35, 2.6), xytext=(5.0, 3.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    style(ax, t)


def d_awq_salience(ax, t, rng):
    """AWQ: protect salient weight channels by activation scale."""
    ch = np.arange(16)
    act_scale = np.abs(rng.normal(1, 0.5, 16))
    act_scale[5] = 3.5
    act_scale[11] = 2.8
    protect = act_scale > np.percentile(act_scale, 75)
    colors = [GOLD if p else TEAL for p in protect]
    ax.bar(ch, act_scale, color=colors, edgecolor=DEEP)
    ax.axhline(np.percentile(act_scale, 75), color=ROSE, ls="--", label="salience cut")
    ax.set_xlabel("output channel")
    ax.set_ylabel("activation scale")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_line_graph_lift(ax, t, rng):
    """Line graph: edges of G become vertices of L(G)."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # G
    gpos = {0: (1.5, 4.5), 1: (3.5, 5.2), 2: (3.5, 3.5), 3: (5.2, 4.5)}
    gedges = [(0, 1, "e1"), (0, 2, "e2"), (1, 2, "e3"), (1, 3, "e4"), (2, 3, "e5")]
    for a, b, _ in gedges:
        ax.plot([gpos[a][0], gpos[b][0]], [gpos[a][1], gpos[b][1]], color=SLATE, lw=1.5)
    for i, (x, y) in gpos.items():
        ax.plot(x, y, "o", color=TEAL, ms=12)
    ax.text(3.2, 5.6, "G", fontsize=12, fontweight="bold", color=INK)
    # L(G)
    lpos = {
        "e1": (7.5, 5.2),
        "e2": (7.5, 3.6),
        "e3": (9.0, 4.5),
        "e4": (10.5, 5.2),
        "e5": (10.5, 3.6),
    }
    ledges = [("e1", "e3"), ("e1", "e4"), ("e2", "e3"), ("e2", "e5"), ("e3", "e4"), ("e3", "e5"), ("e4", "e5")]
    for a, b in ledges:
        ax.plot([lpos[a][0], lpos[b][0]], [lpos[a][1], lpos[b][1]], color=GOLD, lw=1.2, alpha=0.8)
    for name, (x, y) in lpos.items():
        ax.plot(x, y, "s", color=DEEP, ms=11)
        ax.text(x, y - 0.35, name, ha="center", fontsize=8, color=INK)
    ax.text(9.0, 5.7, "L(G)", fontsize=12, fontweight="bold", color=INK)
    ax.annotate("", xy=(7.0, 4.5), xytext=(5.6, 4.5), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    style(ax, t)


def d_data_contracts(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.0, 2.6, 2.0, "schema\nchecks", fc=TEAL, fs=11)
    box(ax, 3.3, 2.0, 2.6, 2.0, "SLIs\nnull/vol", fc=DEEP, fs=11)
    box(ax, 6.2, 2.0, 2.6, 2.0, "owners\non-call", fc=GOLD, tc=INK, fs=11)
    box(ax, 9.1, 2.0, 2.5, 2.0, "break\n→ page", fc=ROSE, fs=11)
    style(ax, t)


def d_postmortem_loop(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    stages = [
        (5, 5.0, "detect"),
        (8, 3.5, "mitigate"),
        (6.5, 1.2, "blameless\nreview"),
        (3.5, 1.2, "actions"),
        (2, 3.5, "verify"),
    ]
    for x, y, lab in stages:
        box(ax, x - 1.1, y - 0.55, 2.2, 1.1, lab, fc=TEAL, fs=10)
    # cycle arrows
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for i, j in pairs:
        x0, y0, _ = stages[i]
        x1, y1, _ = stages[j]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5, connectionstyle="arc3,rad=0.25"),
        )
    style(ax, t)


def d_gloss_opt_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["SGD", "Adam", "Lion", "Muon", "SAM"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C221 = [
    ("Householder reflection geometry", d_householder_reflect),
    ("Consent and purpose decision tree", d_consent_purpose_tree),
    ("PAC-Bayes KL generalization sketch", d_pac_bayes_kl),
    ("Joyplot ridge density stack", d_joyplot_ridges),
    ("Power posterior temperature path", d_power_posterior),
    ("BIRCH CF-tree hierarchy", d_birch_cf_tree),
    ("GSP candidate join-prune pipeline", d_gsp_candidate_gen),
    ("Out-of-fold target encoding", d_target_encoding_cv),
    ("NMF non-negative parts factors", d_nmf_parts),
    ("Partial residual nonlinearity check", d_partial_residual),
    ("Graph label propagation seeds", d_label_propagation),
    ("Gradient noise scale vs batch", d_gradient_noise_scale),
    ("DINO teacher-student match", d_dino_teacher_student),
    ("Perceiver latent cross-attention", d_perceiver_latent),
    ("Option-critic temporal abstraction", d_option_critic),
    ("GPTQ Hessian-aware quant error", d_gptq_hessian),
    ("Modularity block community map", d_modularity_q),
    ("Schema evolution expand-contract", d_schema_evolution),
    ("Architecture decision record strip", d_decision_record),
    ("Glossary probability term strip", d_gloss_prob_strip),
]

C222 = [
    ("Truncated SVD energy vs rank", d_svd_truncation),
    ("Dual-use capability gate", d_dual_use_gate),
    ("Coreset compression error curve", d_compression_set),
    ("Hexbin bivariate density", d_hexbin_density),
    ("Stein score discrepancy field", d_stein_discrepancy),
    ("HDBSCAN condensed lifetimes", d_hdbscan_condense),
    ("CloSpan closed sequence pipeline", d_clospan_closed),
    ("Feature hashing collision bins", d_hashing_trick),
    ("Canonical correlation view scores", d_canonical_corr),
    ("Huber M-estimator rho curve", d_m_estimator),
    ("ECOC multiclass code matrix", d_ecoc_codebook),
    ("Muon orthogonalized singular values", d_muon_optimizer),
    ("I-JEPA context-target mask grid", d_ijepa_mask),
    ("Whisper encoder-decoder cascade", d_whisper_encoder_decoder),
    ("MuZero representation dynamics predict", d_muzero_model),
    ("AWQ activation-salient channels", d_awq_salience),
    ("Line graph edge-to-vertex lift", d_line_graph_lift),
    ("Data contract check SLI chain", d_data_contracts),
    ("Incident postmortem learning loop", d_postmortem_loop),
    ("Glossary optimizer family strip", d_gloss_opt_strip),
]


def embed(cycle: int, topics: list) -> None:
    assert len(topics) == len(CHS), f"{len(topics)} != {len(CHS)}"
    for i, (title, fn) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        rng = np.random.default_rng(cycle * 1000 + i * 23 + 5)
        try:
            fn(ax, title, rng)
        except Exception as e:
            ax.clear()
            ax.text(
                0.5,
                0.5,
                f"{title}\n(render fallback)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
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

    m = {221: C221, 222: C222}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [221, 222]
    for c in cycles:
        embed(c, m[c])
