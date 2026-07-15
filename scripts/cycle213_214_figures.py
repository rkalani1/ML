#!/usr/bin/env python3
"""Cycle-213/214 quality densify: novel scientific teal teaching panels."""
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


# C213

def d_svd_truncation(ax, t, rng):
    k = np.arange(1, 25)
    s = 10 * np.exp(-0.28 * (k - 1)) + 0.2
    err = 1 - np.cumsum(s**2) / np.sum(s**2)
    ax.semilogy(k, err, "o-", color=TEAL, lw=2)
    ax.axhline(0.05, color=GOLD, ls="--", label="5% tail energy")
    ax.set_xlabel("rank r kept")
    ax.set_ylabel("relative tail ‖A−A_r‖_F²")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_hipaa_safe_harbor(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    items = ["names", "dates\n>year", "geo\n<small", "phones", "MRN", "faces"]
    for i, lab in enumerate(items):
        box(ax, 0.3 + (i % 6) * 1.95, 1.8, 1.85, 1.6, lab, fc=ROSE if i < 3 else TEAL, fs=9)
    ax.text(6, 4.2, "Safe Harbor identifiers to strip (teaching list)", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_hoeffding(ax, t, rng):
    n = np.logspace(1, 4, 50)
    eps = np.sqrt(np.log(2 / 0.05) / (2 * n))
    ax.loglog(n, eps, color=TEAL, lw=2.2)
    ax.set_xlabel("n")
    ax.set_ylabel("Hoeffding ε (δ=0.05)")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_slopegraph(ax, t, rng):
    cats = ["sens", "spec", "PPV", "NPV", "AUROC"]
    before = np.array([0.82, 0.75, 0.40, 0.95, 0.86])
    after = np.array([0.88, 0.71, 0.38, 0.96, 0.89])
    for i, c in enumerate(cats):
        col = TEAL if after[i] >= before[i] else ROSE
        ax.plot([0, 1], [before[i], after[i]], "-o", color=col, lw=2, ms=8)
        ax.text(-0.05, before[i], c, ha="right", va="center", fontsize=8, color=INK)
        ax.text(1.05, after[i], f"{after[i]:.2f}", ha="left", va="center", fontsize=8, color=INK)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v1", "v2"])
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_conjugate_update(ax, t, rng):
    # Normal-Normal mean update
    mu0, t0 = 0.0, 1.0
    data = rng.normal(1.2, 1, 12)
    n = len(data)
    mu_n = (t0 * mu0 + n * data.mean()) / (t0 + n)
    xs = np.linspace(-2, 4, 200)
    prior = np.exp(-0.5 * t0 * (xs - mu0) ** 2)
    prior /= np.trapezoid(prior, xs)
    post = np.exp(-0.5 * (t0 + n) * (xs - mu_n) ** 2)
    post /= np.trapezoid(post, xs)
    ax.plot(xs, prior, color=SLATE, lw=1.8, label="prior")
    ax.plot(xs, post, color=TEAL, lw=2.2, label="posterior")
    ax.axvline(data.mean(), color=GOLD, ls="--", label="sample mean")
    ax.legend(fontsize=8)
    ax.set_xlabel("μ")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_cure_reps(ax, t, rng):
    # hierarchical representatives
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 4)
    ax.axis("off")
    clusters = [np.array([[0, 0], [0.3, 0.2], [0.1, 0.4]]), np.array([[3, 1], [3.2, 1.3], [2.8, 0.8]]), np.array([[5, 2.5], [4.8, 2.2]])]
    cols = [TEAL, GOLD, DEEP]
    for pts, c in zip(clusters, cols):
        ax.scatter(pts[:, 0], pts[:, 1], c=c, s=40)
        rep = pts.mean(axis=0)
        ax.plot(*rep, "*", color=ROSE, ms=16)
    ax.text(2.5, 3.5, "CURE: scatter representatives per cluster", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_clospan(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 3.5, 1.6, "projected\nDB", fc=SLATE, fs=11)
    box(ax, 4.5, 2.5, 3.5, 1.6, "closed check\nno superseq", fc=TEAL, fs=11)
    box(ax, 8.5, 2.5, 3.2, 1.6, "CloSpan\noutput", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.5, 3.3), xytext=(4.0, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.5, 3.3), xytext=(8.0, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, t)


def d_woe_monotonic(ax, t, rng):
    bins = np.arange(6)
    woe = np.array([-0.8, -0.4, -0.1, 0.2, 0.5, 0.9])
    ax.step(bins, woe, where="mid", color=TEAL, lw=2.2)
    ax.plot(bins, woe, "o", color=DEEP, ms=8)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xlabel("ordered bin")
    ax.set_ylabel("WoE")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_pca_biplot(ax, t, rng):
    scores = rng.normal(0, 1, (40, 2))
    ax.scatter(scores[:, 0], scores[:, 1], c=TEAL, s=28, alpha=0.7)
    loadings = np.array([[0.8, 0.1], [0.2, 0.9], [-0.6, 0.4], [0.3, -0.7]])
    names = ["x1", "x2", "x3", "x4"]
    for v, n in zip(loadings, names):
        ax.annotate("", xy=1.8 * v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8))
        ax.text(1.9 * v[0], 1.9 * v[1], n, color=GOLD, fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_irls_path(ax, t, rng):
    it = np.arange(0, 12)
    beta = 1.5 * (1 - np.exp(-0.45 * it))
    ax.plot(it, beta, "o-", color=TEAL, lw=2)
    ax.axhline(1.5, color=GOLD, ls="--", label="converged β")
    ax.set_xlabel("IRLS iteration")
    ax.set_ylabel("coefficient")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_reliability_diagram(ax, t, rng):
    bins = np.linspace(0.05, 0.95, 10)
    obs = bins + rng.normal(0, 0.04, len(bins))
    obs = np.clip(obs, 0, 1)
    ax.plot([0, 1], [0, 1], "--", color=SLATE, lw=1.5)
    ax.plot(bins, obs, "o-", color=TEAL, lw=2, ms=8)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_grad_clip(ax, t, rng):
    g = np.linspace(0, 5, 100)
    clip = 1.5
    g_clip = np.minimum(g, clip)
    ax.plot(g, g, color=SLATE, ls="--", label="raw ‖g‖")
    ax.plot(g, g_clip, color=TEAL, lw=2.2, label=f"clipped at {clip}")
    ax.axhline(clip, color=GOLD, ls=":", label="threshold")
    ax.set_xlabel("pre-clip gradient norm")
    ax.set_ylabel("post-clip norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_clip_contrastive(ax, t, rng):
    # similarity matrix image-text
    S = rng.normal(0, 0.3, (6, 6))
    np.fill_diagonal(S, 2.0)
    ax.imshow(S, cmap="YlGn")
    for i in range(6):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEAL, lw=2))
    ax.set_xlabel("text j")
    ax.set_ylabel("image i")
    ax.text(0.5, 1.06, "CLIP: diagonal positives in batch", transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_relative_pos_bias(ax, t, rng):
    d = np.arange(-20, 21)
    bias = -np.abs(d) * 0.08
    ax.plot(d, bias, color=TEAL, lw=2)
    ax.set_xlabel("relative distance i−j")
    ax.set_ylabel("attention bias")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_qr_dagger(ax, t, rng):
    # Q-learning vs Double Q overestimation
    steps = np.arange(0, 80)
    q = 0.5 + 0.02 * steps + 0.3 * (1 - np.exp(-steps / 20))
    dq = 0.5 + 0.015 * steps + 0.15 * (1 - np.exp(-steps / 25))
    ax.plot(steps, q, color=ROSE, lw=2, label="max Q (overestimate)")
    ax.plot(steps, dq, color=TEAL, lw=2, label="Double Q")
    ax.set_xlabel("updates")
    ax.set_ylabel("value estimate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_tensor_decomp(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 1.5, 2.5, 2.5, "W\n[d×d]", fc=SLATE, fs=11)
    box(ax, 4, 2.5, 1.8, 1.5, "U", fc=TEAL, fs=12)
    box(ax, 6.2, 2.0, 1.5, 2.0, "S", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.0, 1.5, 1.8, 2.5, "V", fc=DEEP, fs=12)
    ax.text(5, 4.5, "low-rank factorize dense layer", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_community_overlap(ax, t, rng):
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(np.cos(th) - 0.5, np.sin(th), color=TEAL, alpha=0.4)
    ax.fill(np.cos(th) + 0.5, np.sin(th), color=GOLD, alpha=0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, 0, "overlap", ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(-0.5, -1.4, "community A", ha="center", color=TEAL)
    ax.text(0.5, -1.4, "community B", ha="center", color=GOLD)
    style(ax, t)


def d_batch_effect(ax, t, rng):
    for i, (m, c, lab) in enumerate([(-1, TEAL, "site A"), (1.2, GOLD, "site B"), (0.2, ROSE, "site C")]):
        x = rng.normal(m, 0.5, 80)
        y = rng.normal(i * 0.3, 0.6, 80)
        ax.scatter(x, y, c=c, s=20, alpha=0.7, label=lab)
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_runbook_pages(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["alert", "triage", "mitigate", "comms", "review"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_glossary_stats(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["mean", "var", "CI", "p-value", "power"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


# C214

def d_matrix_cond_solve(ax, t, rng):
    # residual growth with kappa
    kappa = np.logspace(0, 6, 40)
    resid = 1e-16 * kappa
    ax.loglog(kappa, resid, color=TEAL, lw=2.2)
    ax.set_xlabel("condition number κ")
    ax.set_ylabel("relative residual floor sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_dua_roles(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["controller", "processor", "subject", "DPO", "vendor"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_structural_risk(ax, t, rng):
    n = np.logspace(2, 5, 40)
    emp = 0.25 + 0.15 * np.exp(-n / 5000)
    pen = 2.0 * np.sqrt(np.log(n) / n)
    ax.semilogx(n, emp, color=GOLD, lw=2, label="empirical risk")
    ax.semilogx(n, emp + pen, color=TEAL, lw=2, label="SRM upper")
    ax.fill_between(n, emp, emp + pen, color=TEAL, alpha=0.15)
    ax.set_xlabel("n")
    ax.set_ylabel("risk")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_bullet_chart(ax, t, rng):
    metrics = ["AUROC", "ECE↓", "sens", "spec"]
    vals = [0.88, 0.06, 0.84, 0.79]
    targets = [0.85, 0.08, 0.80, 0.80]
    for i, (m, v, tgt) in enumerate(zip(metrics, vals, targets)):
        ax.barh(i, 1.0, color="#e2e8f0", height=0.5)
        ax.barh(i, v if m != "ECE↓" else v / 0.15, color=TEAL, height=0.25)
        ax.plot(tgt if m != "ECE↓" else tgt / 0.15, i, "|", color=GOLD, ms=18, mew=3)
        ax.text(-0.05, i, m, ha="right", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("normalized score")
    style(ax, t)


def d_abc_rejection(ax, t, rng):
    theta = rng.uniform(0, 5, 2000)
    # accept if sim near data
    dist = np.abs(theta - 2.3) + rng.normal(0, 0.3, 2000)
    acc = dist < 0.5
    ax.hist(theta[acc], bins=25, color=TEAL, alpha=0.85, density=True, label="accepted θ")
    ax.hist(theta, bins=25, color=SLATE, alpha=0.25, density=True, label="prior draws")
    ax.axvline(2.3, color=GOLD, lw=2, label="true-ish")
    ax.legend(fontsize=8)
    ax.set_xlabel("θ")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_stream_kmeans(ax, t, rng):
    tgrid = np.arange(0, 100)
    c1 = np.cumsum(rng.normal(0, 0.05, 100))
    c2 = 2 + np.cumsum(rng.normal(0, 0.05, 100))
    ax.plot(tgrid, c1, color=TEAL, lw=2, label="centroid 1")
    ax.plot(tgrid, c2, color=GOLD, lw=2, label="centroid 2")
    ax.set_xlabel("stream time")
    ax.set_ylabel("centroid coord")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_rule_growth(ax, t, rng):
    conf = np.linspace(0.5, 1.0, 30)
    n_rules = 200 * np.exp(-4 * (conf - 0.5)) + 5
    ax.plot(conf, n_rules, color=TEAL, lw=2.2)
    ax.set_xlabel("min confidence")
    ax.set_ylabel("# association rules")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_frequency_encode(ax, t, rng):
    cats = [f"c{i}" for i in range(8)]
    freq = np.array([400, 220, 150, 80, 40, 25, 15, 10])
    ax.bar(cats, freq, color=TEAL, edgecolor=DEEP)
    ax.set_ylabel("count → frequency feature")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_factor_analysis(ax, t, rng):
    # loadings heatmap
    L = rng.normal(0, 0.5, (8, 3))
    L[0:3, 0] = 0.9
    L[3:6, 1] = 0.85
    L[6:8, 2] = 0.8
    ax.imshow(L, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xlabel("latent factor")
    ax.set_ylabel("observed var")
    ax.set_xticks([0, 1, 2])
    style(ax, t)


def d_sandwich_se(ax, t, rng):
    # classic SE vs robust SE bars
    coefs = ["β0", "β1", "β2", "β3"]
    se_cl = [0.12, 0.08, 0.15, 0.05]
    se_rb = [0.18, 0.11, 0.22, 0.07]
    x = np.arange(len(coefs))
    ax.bar(x - 0.15, se_cl, 0.3, color=GOLD, label="model-based SE")
    ax.bar(x + 0.15, se_rb, 0.3, color=TEAL, label="sandwich SE")
    ax.set_xticks(x)
    ax.set_xticklabels(coefs)
    ax.set_ylabel("SE")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_npv_prevalence(ax, t, rng):
    prev = np.linspace(0.01, 0.5, 80)
    sens, spec = 0.9, 0.85
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    ax.plot(prev, npv, color=TEAL, lw=2.2)
    ax.set_xlabel("prevalence")
    ax.set_ylabel("NPV")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_swa(ax, t, rng):
    steps = np.arange(0, 100)
    w = np.sin(steps / 8) * np.exp(-steps / 80) + 0.5
    # SWA average tail
    swa = np.copy(w)
    for i in range(60, 100):
        swa[i] = w[60:i + 1].mean()
    ax.plot(steps, w, color=SLATE, lw=1.5, alpha=0.7, label="SGD iterate")
    ax.plot(steps[60:], swa[60:], color=TEAL, lw=2.2, label="SWA average")
    ax.axvline(60, color=GOLD, ls="--", label="SWA start")
    ax.legend(fontsize=8)
    ax.set_xlabel("step")
    ax.set_ylabel("weight coord")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_masked_lm(ax, t, rng):
    tokens = ["The", "[M]", "had", "a", "[M]", "stroke"]
    for i, tok in enumerate(tokens):
        fc = ROSE if tok == "[M]" else TEAL
        box(ax, 0.3 + i * 2.0, 1.5, 1.9, 1.5, tok, fc=fc, fs=10)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.text(6.2, 3.5, "MLM: reconstruct masked tokens", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_audio_spectrogram(ax, t, rng):
    S = np.abs(rng.normal(0, 1, (40, 100)))
    S *= np.linspace(1, 0.2, 40)[:, None]
    ax.imshow(np.log1p(S), aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel bin")
    style(ax, t)


def d_world_model_rollout(ax, t, rng):
    steps = np.arange(0, 20)
    true = np.sin(steps / 3)
    pred = true + rng.normal(0, 0.15, 20).cumsum() * 0.05
    ax.plot(steps, true, color=GOLD, lw=2, label="true latent")
    ax.plot(steps, pred, color=TEAL, lw=2, label="imagined rollout")
    ax.set_xlabel("imagination step")
    ax.set_ylabel("state coord")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_kv_cache(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i in range(6):
        ax.add_patch(Rectangle((0.5 + i * 1.4, 2), 1.2, 1.5, facecolor=TEAL if i < 4 else GOLD, edgecolor=DEEP, alpha=0.85))
        ax.text(1.1 + i * 1.4, 2.75, f"t{i}", ha="center", color="white", fontsize=10, fontweight="bold")
    ax.text(5, 4.2, "KV cache: reuse past keys/values", ha="center", fontsize=10, color=INK)
    ax.text(5, 1.2, "gold = new token append", ha="center", fontsize=9, color=GOLD)
    style(ax, t)


def d_graph_sage(ax, t, rng):
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis("off")
    ax.plot(0, 0, "o", color=ROSE, ms=18)
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        p = (1.2 * np.cos(ang), 1.2 * np.sin(ang))
        ax.plot([0, p[0]], [0, p[1]], color=SLATE, lw=1.2)
        ax.plot(*p, "o", color=TEAL, ms=12)
        for ang2 in np.linspace(0, 2 * np.pi, 3, endpoint=False):
            q = (p[0] + 0.45 * np.cos(ang2), p[1] + 0.45 * np.sin(ang2))
            ax.plot([p[0], q[0]], [p[1], q[1]], color=SLATE, lw=0.6, alpha=0.6)
            ax.plot(*q, "o", color=MINT, ms=6)
    ax.text(0, -1.8, "GraphSAGE: sample-and-aggregate neighborhood", ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_psi_drift(ax, t, rng):
    bins = np.arange(8)
    exp = np.array([0.15, 0.2, 0.18, 0.12, 0.1, 0.1, 0.08, 0.07])
    act = np.array([0.1, 0.12, 0.15, 0.18, 0.15, 0.12, 0.1, 0.08])
    psi = (act - exp) * np.log((act + 1e-9) / (exp + 1e-9))
    ax.bar(bins - 0.15, exp, 0.3, color=TEAL, label="expected")
    ax.bar(bins + 0.15, act, 0.3, color=GOLD, label="actual")
    ax.text(0.02, 0.92, f"PSI≈{psi.sum():.3f}", transform=ax.transAxes, fontsize=11, color=INK, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlabel("bin")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_postmortem_blameless(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["timeline", "impact", "causes\n(system)", "actions", "owners"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_glossary_causal(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["confound", "collider", "mediate", "DAG", "pred≠cause"]):
        box(ax, 0.25 + i * 2.35, 1.2, 2.25, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=9)
    style(ax, t)


C213 = [
    ("SVD truncation tail energy", d_svd_truncation),
    ("HIPAA Safe Harbor ID strip", d_hipaa_safe_harbor),
    ("Hoeffding bound sample size", d_hoeffding),
    ("Slopegraph metric version delta", d_slopegraph),
    ("Normal-Normal conjugate update", d_conjugate_update),
    ("CURE representative scatter", d_cure_reps),
    ("CloSpan closed sequence mine", d_clospan),
    ("Monotone WoE bin path", d_woe_monotonic),
    ("PCA biplot scores loadings", d_pca_biplot),
    ("IRLS coefficient convergence", d_irls_path),
    ("Reliability diagram bins", d_reliability_diagram),
    ("Gradient norm clipping map", d_grad_clip),
    ("CLIP image-text similarity", d_clip_contrastive),
    ("Relative position attention bias", d_relative_pos_bias),
    ("Double Q overestimation gap", d_qr_dagger),
    ("Low-rank layer factorization", d_tensor_decomp),
    ("Overlapping community sets", d_community_overlap),
    ("Batch effect site clusters", d_batch_effect),
    ("Incident runbook stage tiles", d_runbook_pages),
    ("Glossary core stats strip", d_glossary_stats),
]

C214 = [
    ("Condition number residual growth", d_matrix_cond_solve),
    ("Data protection role tiles", d_dua_roles),
    ("Structural risk minimization band", d_structural_risk),
    ("Bullet chart KPI targets", d_bullet_chart),
    ("ABC rejection sampling posterior", d_abc_rejection),
    ("Streaming k-means centroids", d_stream_kmeans),
    ("Rule count vs confidence", d_rule_growth),
    ("Frequency encoding tall head", d_frequency_encode),
    ("Factor analysis loading heat", d_factor_analysis),
    ("Sandwich robust standard errors", d_sandwich_se),
    ("NPV versus prevalence curve", d_npv_prevalence),
    ("SWA weight averaging tail", d_swa),
    ("Masked language model tokens", d_masked_lm),
    ("Mel spectrogram log energy", d_audio_spectrogram),
    ("World model imagination rollout", d_world_model_rollout),
    ("Transformer KV cache append", d_kv_cache),
    ("GraphSAGE neighborhood sample", d_graph_sage),
    ("PSI feature drift bars", d_psi_drift),
    ("Blameless postmortem sections", d_postmortem_blameless),
    ("Glossary causal term strip", d_glossary_causal),
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
    m = {213: C213, 214: C214}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [213, 214]
    for c in cycles:
        embed(c, m[c])
