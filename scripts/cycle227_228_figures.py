#!/usr/bin/env python3
"""Cycle-227/228 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc, Ellipse, Wedge

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


def d_condition_number(ax, t, rng):
    k = np.logspace(0, 6, 50)
    err = 1e-16 * k
    ax.loglog(k, err, color=TEAL, lw=2.2)
    ax.axhline(1e-8, color=GOLD, ls="--", label="tol sketch")
    ax.set_xlabel("κ(A)")
    ax.set_ylabel("relative error amp.")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_ai_act_risk(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for i, (lab, c) in enumerate([("minimal", MINT), ("limited", TEAL), ("high", GOLD), ("unacceptable", ROSE)]):
        box(ax, 0.5 + i * 3.0, 2.2, 2.7, 2.2, lab, fc=c, tc=("white" if c not in (GOLD, MINT) else INK), fs=12)
    ax.text(6, 1.0, "EU AI Act risk ladder (teaching sketch)", ha="center", fontsize=10, color=SLATE)
    style(ax, t)


def d_fat_shattering(ax, t, rng):
    n = np.arange(1, 30)
    # fat-shattering dim growth sketch
    g = np.minimum(2.0**n, (n.astype(float) ** 2) * 3)
    ax.semilogy(n, g, color=TEAL, lw=2.2)
    ax.set_xlabel("n")
    ax.set_ylabel("fat-shattering growth sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_beeswarm(ax, t, rng):
    groups = []
    for g, mu in enumerate([0, 0.8, -0.4]):
        y = rng.normal(mu, 0.5, 40)
        # simple swarm x jitter by density
        x = g + rng.uniform(-0.15, 0.15, 40)
        ax.scatter(x, y, c=TEAL if g != 1 else GOLD, s=22, alpha=0.75, edgecolors=DEEP, linewidths=0.3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["A", "B", "C"])
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_empirical_bayes(ax, t, rng):
    # group means shrunk toward grand mean
    true = rng.normal(0, 1.2, 12)
    noisy = true + rng.normal(0, 0.8, 12)
    grand = noisy.mean()
    shrink = 0.55 * noisy + 0.45 * grand
    x = np.arange(12)
    ax.scatter(x, noisy, c=SLATE, s=40, label="MLE")
    ax.scatter(x, shrink, c=TEAL, s=50, label="EB shrink")
    ax.axhline(grand, color=GOLD, ls="--", label="grand mean")
    ax.set_xlabel("group")
    ax.set_ylabel("estimate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_cure_clustering(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 2.8, 1.6, "sample\nrepresentatives", fc=TEAL, fs=10)
    box(ax, 3.7, 3.5, 2.8, 1.6, "partition\n(hier)", fc=GOLD, tc=INK, fs=11)
    box(ax, 6.9, 3.5, 2.8, 1.6, "label\nfull data", fc=DEEP, fs=11)
    ax.text(5, 1.5, "CURE: shrinking representatives toward centroid", ha="center", fontsize=10, color=SLATE)
    for x0, x1 in [(3.3, 3.7), (6.5, 6.9)]:
        ax.annotate("", xy=(x1, 4.3), xytext=(x0, 4.3), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_charm_closed(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["IT-tree", "diffset\nexplore", "closed\ncheck", "CHARM\nset"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_embeddings_lookup(ax, t, rng):
    # embedding matrix rows
    E = rng.normal(0, 1, (8, 6))
    ax.imshow(E, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
    ax.axhline(3.5, color=GOLD, lw=2)
    ax.text(5.5, 3.5, "row id → vector", color=GOLD, fontsize=9, va="center")
    ax.set_xlabel("dim")
    ax.set_ylabel("token / entity id")
    style(ax, t)


def d_umap_fuzzy(ax, t, rng):
    # fuzzy simplicial set strength vs distance
    d = np.linspace(0, 3, 100)
    for rho, c, lab in [(0.2, TEAL, "ρ=0.2"), (0.8, GOLD, "ρ=0.8")]:
        w = np.exp(-(np.maximum(0, d - rho) / 0.5))
        ax.plot(d, w, color=c, lw=2, label=lab)
    ax.set_xlabel("distance")
    ax.set_ylabel("fuzzy membership")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_elastic_net_path(ax, t, rng):
    lam = np.logspace(-2, 1, 40)
    # three coef paths
    for i, (b0, c) in enumerate([(1.2, TEAL), (0.8, GOLD), (0.4, ROSE)]):
        beta = b0 / (1 + lam) * np.exp(-0.1 * i * lam)
        ax.semilogx(lam, beta, color=c, lw=2, label=f"β{i+1}")
    ax.set_xlabel("λ")
    ax.set_ylabel("coefficient")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_error_correcting_output(ax, t, rng):
    # Hamming distance decode
    codes = np.array([[1, 1, 1, 0, 0], [1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]])
    pred = np.array([1, 1, 0, 0, 0])
    d = np.sum(codes != pred, axis=1)
    ax.bar(np.arange(4), d, color=TEAL, edgecolor=DEEP)
    ax.axhline(d.min(), color=GOLD, ls="--", label="min Hamming → class")
    ax.set_xlabel("class codeword")
    ax.set_ylabel("Hamming distance")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_layer_scale(ax, t, rng):
    depth = np.arange(1, 25)
    scale = 0.1 * np.ones_like(depth, dtype=float)
    # deeper layers often smaller init scale
    scale = 0.1 / np.sqrt(depth)
    ax.plot(depth, scale, "o-", color=TEAL, lw=2)
    ax.set_xlabel("layer index")
    ax.set_ylabel("LayerScale γ init sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_simsiam(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.5, 3.2, 1.8, "encoder\n+ projector", fc=TEAL, fs=11)
    box(ax, 4.2, 2.5, 3.2, 1.8, "predictor\n(one side)", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.0, 2.5, 3.5, 1.8, "stop-grad\nother branch", fc=DEEP, fs=11)
    ax.text(6, 1.2, "SimSiam: no negatives / no momentum encoder", ha="center", fontsize=10, color=SLATE)
    for x0, x1 in [(3.6, 4.2), (7.4, 8.0)]:
        ax.annotate("", xy=(x1, 3.4), xytext=(x0, 3.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_audiopalm(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["audio\ntokens", "text\ntokens", "shared\nPaLM", "gen\naudio/text"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_iql(ax, t, rng):
    """Implicit Q-learning: expectile regression."""
    u = np.linspace(-3, 3, 200)
    tau = 0.7
    loss = np.where(u < 0, (1 - tau) * u**2, tau * u**2)
    ax.plot(u, loss, color=TEAL, lw=2.2, label=f"expectile τ={tau}")
    ax.plot(u, 0.5 * u**2, color=SLATE, ls="--", lw=1.5, label="MSE")
    ax.set_xlabel("residual u")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_flash_attn_tiles(ax, t, rng):
    # block tiles of Q,K
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for i in range(4):
        for j in range(4):
            c = TEAL if (i + j) % 2 == 0 else DEEP
            ax.add_patch(Rectangle((1 + j * 1.2, 1 + i * 1.1), 1.05, 0.95, facecolor=c, edgecolor="white"))
    ax.text(3.4, 5.5, "FlashAttention SRAM tiles (Q/K/V blocks)", ha="center", fontsize=10, color=INK)
    box(ax, 6.5, 2.5, 3, 1.5, "O(N) HBM\ntraffic", fc=GOLD, tc=INK, fs=11)
    style(ax, t)


def d_role2vec(ax, t, rng):
    # structural roles vs communities
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # two communities with bridge roles
    for cx, cy, col in [(2.5, 3.5, TEAL), (7.5, 3.5, DEEP)]:
        for i in range(5):
            ang = 2 * np.pi * i / 5
            ax.plot(cx + 1.1 * np.cos(ang), cy + 1.1 * np.sin(ang), "o", color=col, ms=12)
        ax.plot(cx, cy, "o", color=GOLD, ms=14)
    ax.plot([3.6, 6.4], [3.5, 3.5], color=ROSE, lw=2)
    ax.text(5, 1.2, "role (bridge) ≠ community label", ha="center", fontsize=10, color=SLATE)
    style(ax, t)


def d_drift_psi(ax, t, rng):
    bins = np.arange(8)
    expected = np.array([0.15, 0.2, 0.18, 0.12, 0.1, 0.1, 0.08, 0.07])
    actual = np.array([0.08, 0.12, 0.15, 0.15, 0.15, 0.15, 0.12, 0.08])
    w = 0.35
    ax.bar(bins - w / 2, expected, width=w, color=SLATE, label="train")
    ax.bar(bins + w / 2, actual, width=w, color=TEAL, label="prod")
    psi = np.sum((actual - expected) * np.log((actual + 1e-9) / (expected + 1e-9)))
    ax.set_xlabel("bin")
    ax.set_ylabel("share")
    ax.legend(fontsize=8)
    ax.text(0.98, 0.95, f"PSI≈{psi:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=ROSE)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_chaos_game_day(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["inject\nfail", "observe", "mitigate", "retro", "fix"]):
        box(ax, 0.3 + i * 2.4, 1.6, 2.2, 1.9, lab, fc=TEAL if i % 2 == 0 else ROSE if i == 0 else DEEP, fs=11)
    style(ax, t)


def d_gloss_dimred_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["PCA", "t-SNE", "UMAP", "NMF", "AE"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_arnoldi(ax, t, rng):
    # Hessenberg structure
    n = 6
    H = np.triu(rng.uniform(0.2, 1.5, (n, n)), -1)
    ax.imshow(np.where(np.abs(H) > 0.05, H, 0), cmap="Greens")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(2.5, n + 0.15, "Arnoldi → upper Hessenberg", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_watermark_detect(ax, t, rng):
    z = np.linspace(-1, 6, 200)
    null = np.exp(-0.5 * ((z - 0) / 1) ** 2)
    alt = np.exp(-0.5 * ((z - 3) / 1) ** 2)
    ax.fill_between(z, 0, null / null.max(), color=SLATE, alpha=0.4, label="no watermark")
    ax.fill_between(z, 0, alt / alt.max(), color=TEAL, alpha=0.5, label="watermarked")
    ax.axvline(2.0, color=GOLD, ls="--", label="threshold")
    ax.set_xlabel("detection score")
    ax.set_ylabel("density sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_algorithmic_stability(ax, t, rng):
    n = np.logspace(1, 4, 40)
    stab = 1.0 / n
    gen = 2 * stab + np.sqrt(np.log(n) / n) * 0.3
    ax.loglog(n, stab, color=GOLD, lw=2, label="stability β")
    ax.loglog(n, gen, color=TEAL, lw=2.2, label="gen. bound sketch")
    ax.set_xlabel("n")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_contour_filled(ax, t, rng):
    x = np.linspace(-3, 3, 80)
    y = np.linspace(-3, 3, 80)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / 2) + 0.6 * np.exp(-((X + 1) ** 2 + (Y + 1) ** 2) / 1.5)
    cs = ax.contourf(X, Y, Z, levels=12, cmap="Greens")
    ax.contour(X, Y, Z, levels=6, colors=DEEP, linewidths=0.6, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    style(ax, t)


def d_gaussian_process(ax, t, rng):
    x = np.linspace(0, 10, 100)
    # prior samples
    K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 1.5**2)
    K += 1e-6 * np.eye(len(x))
    L = np.linalg.cholesky(K)
    for i in range(4):
        f = L @ rng.normal(0, 1, len(x))
        ax.plot(x, f, color=TEAL, alpha=0.5, lw=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x) prior draws")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_denclue(ax, t, rng):
    # density attractors
    x = np.linspace(-3, 3, 200)
    dens = 0.6 * np.exp(-0.5 * ((x + 1.2) / 0.5) ** 2) + 0.5 * np.exp(-0.5 * ((x - 1.0) / 0.6) ** 2)
    ax.fill_between(x, 0, dens, color=TEAL, alpha=0.4)
    ax.plot(x, dens, color=DEEP, lw=2)
    for m in [-1.2, 1.0]:
        ax.axvline(m, color=GOLD, ls="--", lw=1.5)
        ax.plot(m, np.interp(m, x, dens), "*", color=ROSE, ms=14)
    ax.set_xlabel("x")
    ax.set_ylabel("kernel density")
    ax.text(0, dens.max() * 0.9, "attractors", ha="center", color=GOLD, fontsize=9)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_sequence_graph(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    nodes = {"A": (1.5, 2.5), "B": (4, 4), "C": (4, 1), "D": (7, 2.5), "E": (9, 3.5)}
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E"), ("B", "E")]
    for a, b in edges:
        ax.annotate("", xy=nodes[b], xytext=nodes[a], arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    for name, (x, y) in nodes.items():
        ax.plot(x, y, "o", color=TEAL, ms=16)
        ax.text(x, y, name, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    style(ax, t)


def d_polynomial_features(ax, t, rng):
    # degree expansion size
    d = np.arange(1, 8)
    p = 5  # original dims
    # binom(p+d, d) sketch
    from math import comb
    sizes = [comb(p + k, k) for k in d]
    ax.semilogy(d, sizes, "o-", color=TEAL, lw=2.2)
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("# features (p=5)")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_cca_scree(ax, t, rng):
    can_corr = np.array([0.92, 0.71, 0.45, 0.22, 0.08, 0.03])
    ax.bar(np.arange(1, 7), can_corr, color=TEAL, edgecolor=DEEP)
    ax.set_xlabel("canonical component")
    ax.set_ylabel("canonical correlation")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_negative_binomial(ax, t, rng):
    k = np.arange(0, 30)
    # NB mean mu, dispersion alpha
    mu, alpha = 8.0, 0.3
    # r = 1/alpha, p = r/(r+mu)
    r = 1 / alpha
    p = r / (r + mu)
    # PMF via gamma
    from math import lgamma
    def nb_pmf(k):
        return np.exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1) + r * np.log(p) + k * np.log(1 - p))
    pmf = np.array([nb_pmf(int(i)) for i in k])
    ax.bar(k, pmf, color=TEAL, edgecolor=DEEP, width=0.9)
    ax.set_xlabel("count")
    ax.set_ylabel("NB PMF")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_isotonic(ax, t, rng):
    x = np.sort(rng.uniform(0, 1, 30))
    y = 0.2 + 0.7 * x + rng.normal(0, 0.08, 30)
    # pool adjacent violators sketch - monotone projection
    y_iso = np.maximum.accumulate(np.minimum.accumulate(y[::-1])[::-1] * 0 + y)
    # simple isotonic: cummean sort
    y_iso = np.array(y)
    for _ in range(20):
        for i in range(len(y_iso) - 1):
            if y_iso[i] > y_iso[i + 1]:
                m = 0.5 * (y_iso[i] + y_iso[i + 1])
                y_iso[i] = y_iso[i + 1] = m
    ax.scatter(x, y, c=SLATE, s=28, alpha=0.7, label="raw")
    ax.step(x, y_iso, where="mid", color=TEAL, lw=2.2, label="isotonic")
    ax.set_xlabel("score")
    ax.set_ylabel("calibrated p")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_schedule_free(ax, t, rng):
    steps = np.arange(0, 100)
    # schedule-free vs cosine
    cos = 0.5 * (1 + np.cos(np.pi * steps / 100))
    sf = 1 / np.sqrt(steps + 1)
    ax.plot(steps, cos, color=SLATE, lw=1.8, label="cosine LR")
    ax.plot(steps, sf / sf[0], color=TEAL, lw=2.2, label="schedule-free sketch")
    ax.set_xlabel("step")
    ax.set_ylabel("relative scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dino_v2(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.8, 3.5, 1.6, "ViT student", fc=TEAL, fs=12)
    box(ax, 4.5, 3.8, 3.5, 1.6, "ViT teacher\n(EMA)", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.5, 3.8, 3.2, 1.6, "KoLeo +\nSinkhorn", fc=DEEP, fs=11)
    box(ax, 2.5, 1.2, 7, 1.5, "high-res features / PCA maps (teaching)", fc=MINT, fs=11)
    for x0, x1 in [(4.0, 4.5), (8.0, 8.5)]:
        ax.annotate("", xy=(x1, 4.6), xytext=(x0, 4.6), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_seamless_m4t(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["speech\nin", "text\nin", "unitY /\nseq2seq", "speech\nout", "text\nout"]):
        box(ax, 0.25 + i * 2.4, 1.6, 2.2, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_awac(ax, t, rng):
    adv = np.linspace(-2, 2, 100)
    w = np.exp(np.clip(adv / 0.5, -10, 5))
    w = w / w.max()
    ax.plot(adv, w, color=TEAL, lw=2.2)
    ax.set_xlabel("advantage A")
    ax.set_ylabel("AWAC weight sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_kv_cache_quant(ax, t, rng):
    bits = np.array([2, 4, 8, 16])
    mem = 100 * bits / 16
    ppl_delta = np.array([1.5, 0.3, 0.05, 0.0])
    ax.plot(bits, mem, "o-", color=TEAL, lw=2, label="KV mem %")
    ax2 = ax.twinx()
    ax2.plot(bits, ppl_delta, "s--", color=GOLD, lw=2, label="Δppl")
    ax.set_xlabel("KV bits")
    ax.set_ylabel("memory % of FP16")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_hyperbolic_embed(ax, t, rng):
    # Poincaré disk
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=SLATE, lw=1.5)
    r = rng.uniform(0, 0.85, 40)
    ang = rng.uniform(0, 2 * np.pi, 40)
    ax.scatter(r * np.cos(ang), r * np.sin(ang), c=TEAL, s=28, alpha=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -1.25, "Poincaré disk embeddings", ha="center", fontsize=10, color=SLATE)
    style(ax, t)


def d_canary_deploy(ax, t, rng):
    t_ = np.arange(0, 30)
    err_base = 0.02 + rng.normal(0, 0.002, 30)
    err_can = 0.02 + np.where(t_ > 10, 0.01, 0) + rng.normal(0, 0.002, 30)
    ax.plot(t_, err_base, color=SLATE, lw=2, label="stable")
    ax.plot(t_, err_can, color=TEAL, lw=2, label="canary")
    ax.axvline(10, color=GOLD, ls="--", label="canary start")
    ax.set_xlabel("minute")
    ax.set_ylabel("error rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_game_day_scorecard(ax, t, rng):
    metrics = ["MTTD", "MTTR", "comms", "docs", "fix"]
    scores = [4, 3, 5, 4, 3]
    ax.barh(metrics, scores, color=TEAL, edgecolor=DEEP)
    ax.set_xlim(0, 5)
    ax.set_xlabel("score (1-5)")
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_gloss_compress_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["prune", "distill", "quantize", "LoRA", "KV-cache"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


C227 = [
    ("Matrix condition number error amp", d_condition_number),
    ("AI Act risk tier ladder", d_ai_act_risk),
    ("Fat-shattering growth sketch", d_fat_shattering),
    ("Beeswarm group comparison", d_beeswarm),
    ("Empirical Bayes group shrink", d_empirical_bayes),
    ("CURE representative clustering", d_cure_clustering),
    ("CHARM closed itemset search", d_charm_closed),
    ("Embedding table row lookup", d_embeddings_lookup),
    ("UMAP fuzzy membership curves", d_umap_fuzzy),
    ("Elastic-net coefficient path", d_elastic_net_path),
    ("ECOC Hamming decode distances", d_error_correcting_output),
    ("LayerScale depth init decay", d_layer_scale),
    ("SimSiam stop-grad branches", d_simsiam),
    ("AudioPaLM shared token path", d_audiopalm),
    ("IQL expectile residual loss", d_iql),
    ("FlashAttention SRAM tiles", d_flash_attn_tiles),
    ("Role2vec bridge vs community", d_role2vec),
    ("Population stability index bins", d_drift_psi),
    ("Chaos game-day drill stages", d_chaos_game_day),
    ("Glossary dim-reduction strip", d_gloss_dimred_strip),
]

C228 = [
    ("Arnoldi Hessenberg structure", d_arnoldi),
    ("Watermark detection score laws", d_watermark_detect),
    ("Algorithmic stability bound", d_algorithmic_stability),
    ("Filled contour density field", d_contour_filled),
    ("GP prior sample paths", d_gaussian_process),
    ("DENCLUE density attractors", d_denclue),
    ("Sequence graph walk edges", d_sequence_graph),
    ("Polynomial feature blow-up", d_polynomial_features),
    ("CCA scree canonical corrs", d_cca_scree),
    ("Negative binomial count PMF", d_negative_binomial),
    ("Isotonic calibration steps", d_isotonic),
    ("Schedule-free vs cosine LR", d_schedule_free),
    ("DINOv2 teacher KoLeo path", d_dino_v2),
    ("SeamlessM4T modality grid", d_seamless_m4t),
    ("AWAC advantage weights", d_awac),
    ("KV-cache quant memory trade", d_kv_cache_quant),
    ("Hyperbolic Poincaré embeddings", d_hyperbolic_embed),
    ("Canary deploy error watch", d_canary_deploy),
    ("Game-day readiness scorecard", d_game_day_scorecard),
    ("Glossary compression strip", d_gloss_compress_strip),
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

    m = {227: C227, 228: C228}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [227, 228]
    for c in cycles:
        embed(c, m[c])
