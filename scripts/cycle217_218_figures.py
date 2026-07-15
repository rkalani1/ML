#!/usr/bin/env python3
"""Cycle-217/218 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch, Ellipse, Arc

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


# C217

def d_lu_pivot(ax, t, rng):
    A = np.array([[1e-10, 1.0], [1.0, 1.0]])
    ax.imshow(A, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{A[i,j]:.1e}" if A[i, j] < 1e-5 else f"{A[i,j]:.1f}", ha="center", va="center", fontsize=11, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.25, "partial pivoting swaps tiny pivot rows", transform=ax.transAxes, ha="center", fontsize=9, color=ROSE)
    style(ax, t)


def d_breach_notify(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["detect", "contain", "assess", "notify\n72h", "remediate"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else (ROSE if i == 3 else DEEP), fs=10)
    style(ax, t)


def d_rademacher_sum(ax, t, rng):
    n = np.arange(10, 400, 5)
    bound = np.sqrt(2 * np.log(2 * 50) / n)
    ax.plot(n, bound, color=TEAL, lw=2.2)
    ax.set_xlabel("n")
    ax.set_ylabel("Rademacher sum bound sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_pair_plot_sketch(ax, t, rng):
    # 2x2 pairs-plot collage on one axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(Rectangle((0.5, 5.5), 4, 4, fill=False, edgecolor=SLATE))
    xs = np.linspace(0.7, 4.3, 20)
    dens = np.exp(-0.5 * ((xs - 2.5) / 0.8) ** 2)
    ax.fill_between(xs, 5.6, 5.6 + dens * 2.5, color=TEAL, alpha=0.7)
    ax.add_patch(Rectangle((5.5, 5.5), 4, 4, fill=False, edgecolor=SLATE))
    ax.scatter(5.5 + 2 + rng.normal(0, 0.8, 40), 7.5 + rng.normal(0, 0.8, 40), c=TEAL, s=10, alpha=0.7)
    ax.add_patch(Rectangle((0.5, 0.5), 4, 4, fill=False, edgecolor=SLATE))
    ax.scatter(0.5 + 2 + rng.normal(0, 0.8, 40), 2.5 + rng.normal(0, 0.8, 40), c=GOLD, s=10, alpha=0.7)
    ax.add_patch(Rectangle((5.5, 0.5), 4, 4, fill=False, edgecolor=SLATE))
    ys = np.linspace(0.7, 4.3, 20)
    dens2 = np.exp(-0.5 * ((ys - 2.5) / 0.6) ** 2)
    ax.fill_betweenx(ys, 5.6, 5.6 + dens2 * 2.5, color=DEEP, alpha=0.7)
    style(ax, t)


def d_conjugate_prior_map(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    pairs = [("Bernoulli", "Beta"), ("Poisson", "Gamma"), ("Normal", "Normal"), ("Exp", "Gamma")]
    for i, (lik, pr) in enumerate(pairs):
        box(ax, 0.4 + i * 3.0, 2.8, 2.7, 1.2, lik, fc=TEAL, fs=10)
        box(ax, 0.4 + i * 3.0, 1.0, 2.7, 1.2, pr, fc=GOLD, tc=INK, fs=10)
        ax.annotate("", xy=(1.75 + i * 3.0, 2.2), xytext=(1.75 + i * 3.0, 2.8), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, t)


def d_clique_percolation(ax, t, rng):
    # overlapping k-cliques
    pos = {"a": (0, 0), "b": (1, 0), "c": (0.5, 0.87), "d": (1.8, 0.4), "e": (2.3, 1.2), "f": (2.8, 0.3)}
    edges = [("a", "b"), ("b", "c"), ("c", "a"), ("b", "d"), ("d", "e"), ("e", "b"), ("d", "f"), ("e", "f"), ("b", "f")]
    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=SLATE, lw=1.3)
    for n, p in pos.items():
        ax.plot(*p, "o", color=TEAL if n in "abc" else GOLD, ms=12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(1.2, -0.5, "clique percolation overlap communities", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_gap_constraint_seq(ax, t, rng):
    times = [0, 1, 2, 4, 7, 8]
    labs = ["A", "B", "C", "A", "B", "C"]
    for x, lab in zip(times, labs):
        ax.plot(x, 1, "o", color=TEAL, ms=14)
        ax.text(x, 1.25, lab, ha="center", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(2, 0.7), xytext=(0, 0.7), arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2))
    ax.text(1, 0.4, "mingap=0 maxgap=2", ha="center", color=GOLD, fontsize=9)
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(0, 2)
    ax.axis("off")
    style(ax, t)


def d_target_mean_smooth(ax, t, rng):
    n = np.arange(1, 40)
    global_m = 0.25
    raw = 0.6 * np.ones_like(n, dtype=float)
    m = 10
    smooth = (raw * n + global_m * m) / (n + m)
    ax.plot(n, raw, color=ROSE, ls="--", label="category mean")
    ax.plot(n, smooth, color=TEAL, lw=2, label="smoothed toward global")
    ax.axhline(global_m, color=GOLD, ls=":", label="global mean")
    ax.set_xlabel("category count n")
    ax.set_ylabel("encoded value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_random_projection(ax, t, rng):
    # JL: distances preserved
    d = np.linspace(5, 200, 40)
    eps = 0.1
    # required k ~ O(eps^-2 log n)
    k = (4 * np.log(100) / (eps**2 / 2 - eps**3 / 3)) * np.ones_like(d)
    ax.plot(d, k, color=TEAL, lw=2)
    ax.set_xlabel("ambient dim d")
    ax.set_ylabel("target k (JL sketch, n=100)")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_added_variable(ax, t, rng):
    x = rng.normal(0, 1, 50)
    r = 0.6 * x + rng.normal(0, 0.4, 50)
    ax.scatter(x, r, c=TEAL, s=28, alpha=0.8)
    b = np.polyfit(x, r, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=GOLD, lw=2)
    ax.set_xlabel("e(x_j | others)")
    ax.set_ylabel("e(y | others)")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_one_vs_rest(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["class A\nvs rest", "class B\nvs rest", "class C\nvs rest", "argmax\nscore"]):
        box(ax, 0.4 + i * 3.0, 1.6, 2.7, 1.8, lab, fc=TEAL if i < 3 else GOLD, tc="white" if i < 3 else INK, fs=11)
    style(ax, t)


def d_cosine_anneal(ax, t, rng):
    tgrid = np.arange(0, 100)
    T, eta_min, eta_max = 100, 0.01, 0.1
    eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * tgrid / T))
    ax.plot(tgrid, eta, color=TEAL, lw=2.2)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_relative_loc_pred(ax, t, rng):
    # two patches with relative offset vector
    ax.add_patch(Rectangle((1, 1), 2, 2, facecolor=TEAL, edgecolor=DEEP, alpha=0.8))
    ax.add_patch(Rectangle((5, 3), 2, 2, facecolor=GOLD, edgecolor=DEEP, alpha=0.8))
    ax.annotate("", xy=(6, 4), xytext=(2, 2), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    ax.text(3.5, 3.5, "Δ pos", color=ROSE, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(4.5, 0.4, "relative location prediction pretext", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_sentencepiece(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 3.5, 1.6, "raw text\n+ unigram/BPE", fc=SLATE, fs=11)
    box(ax, 4.5, 2.5, 3.5, 1.6, "SentencePiece\nvocab", fc=TEAL, fs=11)
    box(ax, 8.5, 2.5, 3.2, 1.6, "ids\nlossless", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.5, 3.3), xytext=(4.0, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.5, 3.3), xytext=(8.0, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, t)


def d_td_lambda(ax, t, rng):
    lam = np.linspace(0, 1, 40)
    # bias-variance caricature
    bias = (1 - lam) ** 2
    var = lam ** 2
    ax.plot(lam, bias, color=TEAL, lw=2, label="bias² sketch")
    ax.plot(lam, var, color=GOLD, lw=2, label="variance sketch")
    ax.plot(lam, bias + var, color=ROSE, lw=2, label="sum")
    ax.set_xlabel("λ")
    ax.set_ylabel("error component")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_depth_prune(ax, t, rng):
    depth = np.arange(1, 13)
    acc = 0.7 + 0.02 * depth - 0.0015 * depth**2
    size = depth * 10
    ax.plot(depth, acc, "o-", color=TEAL, lw=2, label="accuracy")
    ax2 = ax.twinx()
    ax2.plot(depth, size, "s--", color=GOLD, lw=1.8, label="params proxy")
    ax.set_xlabel("blocks kept")
    ax.set_ylabel("accuracy")
    ax2.set_ylabel("size")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_triadic_closure(ax, t, rng):
    pos = {"a": (0, 0), "b": (1, 0), "c": (0.5, 0.87)}
    ax.plot([0, 1], [0, 0], color=TEAL, lw=2)
    ax.plot([0, 0.5], [0, 0.87], color=TEAL, lw=2)
    ax.plot([1, 0.5], [0, 0.87], color=GOLD, lw=2.5, ls="--")
    for n, p in pos.items():
        ax.plot(*p, "o", color=DEEP, ms=14)
        ax.text(p[0], p[1] - 0.15, n, ha="center")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, 1.15, "open triad → close dashed edge", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_imputation_methods(ax, t, rng):
    methods = ["mean", "median", "knn", "mice", "model"]
    bias = [0.25, 0.18, 0.12, 0.08, 0.06]
    ax.bar(methods, bias, color=TEAL, edgecolor=DEEP)
    ax.set_ylabel("bias proxy (sketch)")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_error_budget_policy(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 1.8, 3.5, 2.0, "burn < 50%\nship freely", fc=TEAL, fs=11)
    box(ax, 4.3, 1.8, 3.5, 2.0, "50–100%\nslow + review", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.1, 1.8, 3.5, 2.0, "exhaust\nfreeze deploys", fc=ROSE, fs=11)
    style(ax, t)


def d_glossary_fe(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["scale", "encode", "select", "cross", "hash"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


# C218

def d_svd_economy(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 1.5, 2.2, 2.5, "A\nm×n", fc=SLATE, fs=12)
    box(ax, 3.2, 1.5, 2.0, 2.5, "U\nm×r", fc=TEAL, fs=12)
    box(ax, 5.8, 2.0, 2.0, 1.5, "Σ\nr×r", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.4, 1.5, 3.0, 2.5, "Vᵀ\nr×n", fc=DEEP, fs=12)
    ax.text(6, 4.4, "economy SVD keeps r = rank", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_data_retention(ax, t, rng):
    years = np.arange(0, 8)
    retain = np.array([1, 1, 1, 0.6, 0.3, 0.1, 0, 0])
    ax.step(years, retain, where="post", color=TEAL, lw=2.2)
    ax.fill_between(years, retain, step="post", color=TEAL, alpha=0.2)
    ax.set_xlabel("years since collection")
    ax.set_ylabel("fraction retained")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_covering_number(ax, t, rng):
    eps = np.logspace(-2, 0, 40)
    N = (1 / eps) ** 2
    ax.loglog(eps, N, color=TEAL, lw=2.2)
    ax.set_xlabel("ε")
    ax.set_ylabel("covering number N(ε) sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_beeswarm(ax, t, rng):
    for i, mu in enumerate([0, 0.5, -0.3, 1.0]):
        y = rng.normal(mu, 0.5, 40)
        x = i + rng.uniform(-0.15, 0.15, 40)
        ax.scatter(x, y, c=TEAL if i % 2 == 0 else DEEP, s=18, alpha=0.75)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["G1", "G2", "G3", "G4"])
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_variational_elbo(ax, t, rng):
    steps = np.arange(0, 60)
    elbo = -5 + 4 * (1 - np.exp(-steps / 15)) + rng.normal(0, 0.05, 60)
    kl = 2.0 * np.exp(-steps / 20) + 0.2
    ax.plot(steps, elbo, color=TEAL, lw=2, label="ELBO")
    ax.plot(steps, kl, color=GOLD, lw=2, label="KL(q‖p)")
    ax.legend(fontsize=8)
    ax.set_xlabel("VI iteration")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_spectral_init(ax, t, rng):
    # k-means++ style vs spectral embedding init
    th = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    for r, c, lab in [(1.0, TEAL, "spectral embed"), (0.3, GOLD, "random")]:
        pts = np.c_[r * np.cos(th + 0.2), r * np.sin(th + 0.2)]
        ax.scatter(pts[:, 0], pts[:, 1], c=c, s=80, label=lab, edgecolors=DEEP)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_spmf_pipeline(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["events", "sequences", "constraints", "patterns", "rules"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_rare_category(ax, t, rng):
    ranks = np.arange(1, 21)
    counts = 500 / ranks**1.2
    ax.bar(ranks, counts, color=TEAL, edgecolor=DEEP)
    ax.axhline(10, color=GOLD, ls="--", label="rare threshold")
    ax.set_xlabel("category rank")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_nmf_basis(ax, t, rng):
    W = np.abs(rng.normal(0, 1, (8, 3)))
    ax.imshow(W, cmap="YlGn", aspect="auto")
    ax.set_xlabel("basis k")
    ax.set_ylabel("feature")
    ax.text(0.5, 1.06, "NMF W≥0 parts-based factors", transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_lasso_cv_1se(ax, t, rng):
    loglam = np.linspace(-3, 1, 30)
    mse = 0.5 + 0.1 * (loglam + 1) ** 2 + 0.02 * rng.normal(size=30)
    se = 0.05 + 0.01 * np.abs(loglam)
    ax.plot(loglam, mse, color=TEAL, lw=2)
    ax.fill_between(loglam, mse - se, mse + se, color=TEAL, alpha=0.2)
    i = int(np.argmin(mse))
    thr = mse[i] + se[i]
    ax.axhline(thr, color=GOLD, ls="--", label="min + 1 SE")
    # 1se lambda larger
    j = i
    while j < len(mse) - 1 and mse[j] <= thr:
        j += 1
    ax.axvline(loglam[j], color=ROSE, ls=":", label="λ_1se")
    ax.axvline(loglam[i], color=DEEP, ls=":", label="λ_min")
    ax.set_xlabel("log λ")
    ax.set_ylabel("CV MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gmean_imbalance(ax, t, rng):
    prev = np.linspace(0.05, 0.5, 40)
    sens, spec = 0.85, 0.8
    gmean = np.sqrt(sens * spec) * np.ones_like(prev)
    acc = sens * prev + spec * (1 - prev)
    ax.plot(prev, acc, color=GOLD, lw=2, label="accuracy")
    ax.plot(prev, gmean, color=TEAL, lw=2, label="G-mean")
    ax.set_xlabel("prevalence")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_layer_drop(ax, t, rng):
    depth = np.arange(1, 13)
    p_drop = 0.05 * depth / depth.max()
    ax.bar(depth, 1 - p_drop, color=TEAL, edgecolor=DEEP, label="P(keep layer)")
    ax.plot(depth, p_drop, "o-", color=GOLD, label="drop prob")
    ax.set_xlabel("layer index")
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_dense_cl(ax, t, rng):
    # dense contrastive: many positives
    S = rng.normal(0, 0.2, (8, 8))
    for i in range(8):
        for j in range(8):
            if abs(i - j) <= 1:
                S[i, j] = 1.5
    ax.imshow(S, cmap="YlGn")
    ax.set_xlabel("key")
    ax.set_ylabel("query")
    ax.text(0.5, 1.06, "dense CL: local positive band", transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_conformer_conv(ax, t, rng):
    # depthwise conv receptive field growth
    layers = np.arange(1, 9)
    rf = 1 + 2 * layers  # caricature
    ax.step(layers, rf, where="mid", color=TEAL, lw=2.2)
    ax.set_xlabel("conv module depth")
    ax.set_ylabel("receptive field (frames)")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_iql_expectile(ax, t, rng):
    u = np.linspace(-3, 3, 200)
    tau = 0.8
    loss = np.abs(tau - (u < 0).astype(float)) * u**2
    ax.plot(u, loss, color=TEAL, lw=2.2, label=f"expectile τ={tau}")
    ax.plot(u, 0.5 * u**2, color=SLATE, ls="--", label="MSE")
    ax.legend(fontsize=8)
    ax.set_xlabel("residual")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_bit_sparsity(ax, t, rng):
    bits = np.array([16, 8, 4, 2])
    size = bits / 16
    acc = np.array([0.9, 0.88, 0.84, 0.7])
    ax.plot(size, acc, "o-", color=TEAL, lw=2)
    for b, s, a in zip(bits, size, acc):
        ax.text(s, a + 0.01, f"{b}-bit", ha="center", fontsize=8)
    ax.set_xlabel("relative model size")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_role_discovery(ax, t, rng):
    # structural equivalence heatmap
    R = rng.random((6, 6))
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1)
    ax.imshow(R, cmap="YlGn")
    ax.set_xlabel("node")
    ax.set_ylabel("node")
    ax.text(0.5, 1.06, "role similarity (structural equivalence)", transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_data_contract(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["schema", "nulls", "ranges", "freshness", "owners"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_gradual_rollout(ax, t, rng):
    hours = np.arange(0, 24)
    pct = np.clip(np.floor(hours / 4) * 20, 0, 100)
    ax.step(hours, pct, where="post", color=TEAL, lw=2.2)
    ax.set_xlabel("hours since start")
    ax.set_ylabel("% traffic")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_glossary_dimred(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["PCA", "t-SNE", "UMAP", "NMF", "AE"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C217 = [
    ("LU partial pivoting necessity", d_lu_pivot),
    ("Breach notify seventy-two hour", d_breach_notify),
    ("Rademacher sum concentration", d_rademacher_sum),
    ("Pairs plot matrix collage", d_pair_plot_sketch),
    ("Conjugate likelihood prior map", d_conjugate_prior_map),
    ("Clique percolation overlap", d_clique_percolation),
    ("Min-max gap sequence windows", d_gap_constraint_seq),
    ("Smoothed target mean encoding", d_target_mean_smooth),
    ("Johnson-Lindenstrauss projection k", d_random_projection),
    ("Added-variable residual plot", d_added_variable),
    ("One-vs-rest multiclass scores", d_one_vs_rest),
    ("Cosine annealing LR schedule", d_cosine_anneal),
    ("Relative patch location pretext", d_relative_loc_pred),
    ("SentencePiece vocab pipeline", d_sentencepiece),
    ("TD-lambda bias variance trade", d_td_lambda),
    ("Depth pruning accuracy size", d_depth_prune),
    ("Triadic closure edge predict", d_triadic_closure),
    ("Imputation method bias bars", d_imputation_methods),
    ("Error budget deploy policy", d_error_budget_policy),
    ("Glossary feature engineering strip", d_glossary_fe),
]

C218 = [
    ("Economy SVD factor shapes", d_svd_economy),
    ("Data retention decay schedule", d_data_retention),
    ("Covering number epsilon scale", d_covering_number),
    ("Beeswarm group comparison", d_beeswarm),
    ("Variational ELBO and KL path", d_variational_elbo),
    ("Spectral embedding cluster seeds", d_spectral_init),
    ("Sequential pattern mine pipeline", d_spmf_pipeline),
    ("Long-tail rare category ranks", d_rare_category),
    ("NMF nonnegative basis heat", d_nmf_basis),
    ("Lasso CV one-SE rule", d_lasso_cv_1se),
    ("G-mean vs accuracy imbalance", d_gmean_imbalance),
    ("LayerDrop keep probability", d_layer_drop),
    ("Dense contrastive positive band", d_dense_cl),
    ("Conformer depthwise RF growth", d_conformer_conv),
    ("IQL expectile regression loss", d_iql_expectile),
    ("Bit-width size accuracy curve", d_bit_sparsity),
    ("Structural role similarity heat", d_role_discovery),
    ("Data contract field checklist", d_data_contract),
    ("Gradual percent traffic ramp", d_gradual_rollout),
    ("Glossary dim-reduction strip", d_glossary_dimred),
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
    m = {217: C217, 218: C218}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [217, 218]
    for c in cycles:
        embed(c, m[c])
