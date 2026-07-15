#!/usr/bin/env python3
"""Cycle-229/230 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc, Ellipse, FancyBboxPatch as FBP

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


def d_gram_schmidt(ax, t, rng):
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect("equal")
    v1 = np.array([2.5, 0.4])
    v2 = np.array([1.2, 2.2])
    # project v2 on v1
    proj = (np.dot(v2, v1) / np.dot(v1, v1)) * v1
    u2 = v2 - proj
    ax.annotate("", xy=v1, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5))
    ax.annotate("", xy=v2, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.annotate("", xy=u2 + proj, xytext=proj, arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.5))
    ax.plot([v2[0], proj[0]], [v2[1], proj[1]], ":", color=ROSE, lw=1.5)
    ax.text(v1[0], v1[1] - 0.25, "u1", color=TEAL, fontsize=11)
    ax.text(v2[0] + 0.1, v2[1], "v2", color=SLATE, fontsize=11)
    ax.text(u2[0] / 2 + proj[0] + 0.15, u2[1] / 2 + proj[1], "u2", color=GOLD, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_sbom_chain(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["deps", "SBOM\nCycloneDX", "CVE\nmatch", "patch /\nwaive"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_compression_bounds(ax, t, rng):
    k = np.arange(1, 25)
    # Occam / MDL sketch
    err = np.exp(-0.15 * k) + 0.05
    comp = 0.02 * k
    ax.plot(k, err, color=TEAL, lw=2, label="fit term")
    ax.plot(k, comp, color=GOLD, lw=2, label="complexity")
    ax.plot(k, err + comp, color=ROSE, lw=2.2, label="total sketch")
    ax.set_xlabel("model size k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_alluvial(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for x in [1.5, 5, 8.5]:
        ax.plot([x, x], [0.8, 5.2], color=SLATE, lw=6, solid_capstyle="round")
    flows = [(1.2, 4.5, 3.5), (2.5, 2.0, 4.0), (3.8, 3.0, 1.5), (4.5, 1.2, 2.8)]
    for y0, y1, y2 in flows:
        ax.plot([1.5, 5, 8.5], [y0, y1, y2], color=TEAL, lw=4, alpha=0.45, solid_capstyle="round")
    ax.text(5, 5.6, "alluvial / Sankey category flows", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_geweke_diag(ax, t, rng):
    # MCMC chain early vs late mean
    chain = np.cumsum(rng.normal(0, 1, 400)) / np.sqrt(np.arange(1, 401))
    early = chain[:80]
    late = chain[-80:]
    ax.plot(chain, color=TEAL, lw=1.2, alpha=0.8)
    ax.axvspan(0, 80, color=GOLD, alpha=0.2, label="early")
    ax.axvspan(320, 400, color=ROSE, alpha=0.2, label="late")
    ax.axhline(early.mean(), color=GOLD, ls="--")
    ax.axhline(late.mean(), color=ROSE, ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("running mean")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_clique_percolation(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # two overlapping communities
    c1 = [(2, 3), (3, 4.2), (4, 3), (3, 1.8)]
    c2 = [(4, 3), (5.5, 4), (6.5, 2.8), (5.2, 1.8)]
    for a, b in zip(c1, c1[1:] + c1[:1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=TEAL, lw=1.5)
    for a, b in zip(c2, c2[1:] + c2[:1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GOLD, lw=1.5)
    for p in c1:
        ax.plot(*p, "o", color=TEAL, ms=12)
    for p in c2:
        ax.plot(*p, "o", color=GOLD, ms=12)
    ax.plot(4, 3, "o", color=ROSE, ms=14)
    ax.text(5, 5.3, "k-clique percolation: shared nodes", ha="center", fontsize=10, color=SLATE)
    style(ax, t)


def d_spam_growth(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["SDB", "prefix\ntree", "SPADE\njoin", "frequent\nseq"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_target_leakage_timeline(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.plot([0.5, 11.5], [2.5, 2.5], color=SLATE, lw=3)
    events = [(2, "feature\nbuild", TEAL), (5, "label\ntime", ROSE), (8, "train", GOLD), (10.5, "serve", DEEP)]
    for x, lab, c in events:
        ax.plot(x, 2.5, "o", color=c, ms=14)
        box(ax, x - 1, 3.2, 2.0, 1.2, lab, fc=c, tc=("white" if c != GOLD else INK), fs=9)
    ax.text(3.5, 1.2, "leak if features use post-label info", ha="center", fontsize=10, color=ROSE)
    style(ax, t)


def d_randomized_svd(ax, t, rng):
    k = np.arange(2, 30)
    err = 0.5 * np.exp(-0.12 * (k - 2)) + 0.02 + 0.01 * rng.normal(0, 1, len(k)).cumsum() * 0
    err = 0.5 * np.exp(-0.12 * (k - 2)) + 0.02
    ax.plot(k, err, color=TEAL, lw=2.2)
    ax.set_xlabel("target rank k")
    ax.set_ylabel("||A − QQBᵀ|| sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_box_cox(ax, t, rng):
    y = np.linspace(0.2, 4, 100)
    for lam, c in [(1, SLATE), (0.5, TEAL), (0, GOLD), (-1, ROSE)]:
        if abs(lam) < 1e-9:
            z = np.log(y)
        else:
            z = (y**lam - 1) / lam
        ax.plot(y, z, color=c, lw=2, label=f"λ={lam}")
    ax.set_xlabel("y > 0")
    ax.set_ylabel("Box–Cox z")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_precision_recall_curve(ax, t, rng):
    rec = np.linspace(0, 1, 50)
    prec = 1 - 0.6 * rec**1.5 + 0.05 * np.sin(8 * rec)
    prec = np.clip(prec, 0.2, 1)
    ax.plot(rec, prec, color=TEAL, lw=2.2)
    ax.fill_between(rec, 0, prec, color=TEAL, alpha=0.15)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_rmsprop(ax, t, rng):
    steps = np.arange(0, 80)
    g2 = np.abs(np.sin(steps / 5)) + 0.1
    v = np.zeros_like(g2, dtype=float)
    for i in range(1, len(g2)):
        v[i] = 0.9 * v[i - 1] + 0.1 * g2[i] ** 2
    ax.plot(steps, g2, color=SLATE, lw=1.5, label="|g|")
    ax.plot(steps, np.sqrt(v + 1e-8), color=TEAL, lw=2.2, label="RMS")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_vicreg(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 3.5, 1.8, "invariance\n(MSE views)", fc=TEAL, fs=11)
    box(ax, 4.3, 3.5, 3.5, 1.8, "variance\nhinge", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.1, 3.5, 3.5, 1.8, "covariance\noff-diag→0", fc=DEEP, fs=11)
    box(ax, 3, 1.0, 6, 1.5, "VICReg: no negatives, no stop-grad required", fc=MINT, fs=10)
    style(ax, t)


def d_rt_detr(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["backbone", "hybrid\nencoder", "IoU-aware\nquery", "boxes +\nclasses"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_conservative_q(ax, t, rng):
    q = np.linspace(-1, 3, 100)
    # CQL penalty pushes Q down on OOD
    pen = 0.3 * np.maximum(q - 1.0, 0) ** 2
    ax.plot(q, q, color=SLATE, ls="--", label="Q")
    ax.plot(q, q - pen, color=TEAL, lw=2.2, label="Q − CQL pen")
    ax.set_xlabel("Q value")
    ax.set_ylabel("regularized")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mixture_of_depths(ax, t, rng):
    layers = np.arange(1, 13)
    capacity = np.where(layers % 2 == 0, 1.0, 0.35)
    ax.bar(layers, capacity, color=[TEAL if c > 0.5 else GOLD for c in capacity], edgecolor=DEEP)
    ax.set_xlabel("layer")
    ax.set_ylabel("token capacity fraction")
    ax.set_ylim(0, 1.2)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_bigclam(ax, t, rng):
    # overlapping community affiliation
    F = np.abs(rng.normal(0.2, 0.3, (15, 3)))
    F[F < 0.35] = 0
    ax.imshow(F, cmap="Greens", aspect="auto")
    ax.set_xlabel("community")
    ax.set_ylabel("node")
    ax.set_xticks([0, 1, 2])
    ax.text(1, 15.5, "BIGCLAM affiliation F ≥ 0", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_unit_test_data(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["schema\ntest", "range\ntest", "null\nrate", "join\nkeys", "pass/\nfail"]):
        box(ax, 0.25 + i * 2.4, 1.6, 2.2, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_slo_dashboard(ax, t, rng):
    days = np.arange(1, 15)
    slo = 0.999
    avail = 0.998 + 0.0015 * np.sin(days / 2) + rng.normal(0, 0.0003, 14)
    ax.plot(days, avail, "o-", color=TEAL, lw=2)
    ax.axhline(slo, color=GOLD, ls="--", label="SLO 99.9%")
    ax.set_xlabel("day")
    ax.set_ylabel("availability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gloss_feat_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["OHE", "target", "hash", "WoE", "embed"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_woodbury(ax, t, rng):
    # low-rank update identity
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.0, 3.5, 2.0, "(A+UVᵀ)⁻¹", fc=TEAL, fs=14)
    box(ax, 4.5, 2.0, 7, 2.0, "A⁻¹ − A⁻¹U(I+VᵀA⁻¹U)⁻¹VᵀA⁻¹", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.5, 3), xytext=(3.9, 3), arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    style(ax, t)


def d_incident_sev(ax, t, rng):
    sevs = ["SEV1", "SEV2", "SEV3", "SEV4"]
    counts = [2, 5, 14, 28]
    colors = [ROSE, GOLD, TEAL, SLATE]
    ax.bar(sevs, counts, color=colors, edgecolor=DEEP)
    ax.set_ylabel("count / quarter")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_pac_bayes_kl_prior(ax, t, rng):
    kl = np.linspace(0.01, 10, 80)
    n = 1000
    bound = np.sqrt((kl + np.log(2 * np.sqrt(n))) / (2 * n))
    ax.plot(kl, bound, color=TEAL, lw=2.2)
    ax.set_xlabel("KL(Q || P)")
    ax.set_ylabel("gen. gap sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_radar_chart(ax, t, rng):
    cats = ["speed", "acc", "fair", "cost", "lat", "robust"]
    N = len(cats)
    vals = np.array([0.8, 0.7, 0.5, 0.6, 0.75, 0.55])
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals_c = vals.tolist() + [vals[0]]
    angles_c = angles + [angles[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles_c, vals_c, color=TEAL, lw=2)
    ax.fill(angles_c, vals_c, color=TEAL, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_yticklabels([])
    style(ax, t)


def d_horseshoe(ax, t, rng):
    beta = np.linspace(-3, 3, 200)
    # horseshoe-like density heavy tails + spike
    dens = 0.5 * np.exp(-0.5 * (beta / 0.15) ** 2) + 0.15 / (1 + beta**2)
    dens = dens / np.trapezoid(dens, beta)
    ax.plot(beta, dens, color=TEAL, lw=2.2)
    ax.set_xlabel("β")
    ax.set_ylabel("horseshoe-like density")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_stream_kmeans(ax, t, rng):
    t_ = np.arange(0, 100)
    c1 = np.sin(t_ / 10) + rng.normal(0, 0.1, 100)
    c2 = 2 + np.cos(t_ / 12) + rng.normal(0, 0.1, 100)
    ax.plot(t_, c1, color=TEAL, lw=1.8, label="center 1")
    ax.plot(t_, c2, color=GOLD, lw=1.8, label="center 2")
    ax.set_xlabel("stream time")
    ax.set_ylabel("centroid coord")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_bide_closed(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["seq", "backward\nextend", "forward\nextend", "closed\nBIDE"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_leave_one_group(ax, t, rng):
    groups = np.repeat(np.arange(5), 4)
    x = np.arange(len(groups))
    colors = [ROSE if g == 2 else TEAL for g in groups]
    ax.scatter(x, groups, c=colors, s=50)
    ax.axhspan(1.5, 2.5, color=ROSE, alpha=0.15)
    ax.set_xlabel("sample")
    ax.set_ylabel("group")
    ax.text(10, 2.7, "held-out group", color=ROSE, fontsize=9)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_factor_analysis(ax, t, rng):
    # loadings heatmap
    L = rng.normal(0, 0.5, (10, 3))
    L[np.abs(L) < 0.4] = 0
    ax.imshow(L, cmap="coolwarm", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xlabel("factor")
    ax.set_ylabel("variable")
    ax.set_xticks([0, 1, 2])
    style(ax, t)


def d_robust_regression(ax, t, rng):
    x = np.linspace(0, 10, 40)
    y = 0.5 * x + 1 + rng.normal(0, 0.4, 40)
    y[5] += 6
    y[20] -= 5
    ax.scatter(x, y, c=TEAL, s=30)
    # OLS vs robust sketch
    ax.plot(x, 0.5 * x + 1, color=GOLD, lw=2, label="robust")
    b = np.polyfit(x, y, 1)
    ax.plot(x, np.polyval(b, x), color=ROSE, ls="--", lw=1.8, label="OLS (outlier pull)")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_brier_skill(ax, t, rng):
    models = ["clim", "A", "B", "C"]
    bs = [0.25, 0.18, 0.12, 0.15]
    skill = 1 - np.array(bs) / bs[0]
    ax.bar(models, skill, color=[SLATE, TEAL, GOLD, DEEP], edgecolor=DEEP)
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylabel("Brier skill score")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_adafactor(ax, t, rng):
    steps = np.arange(0, 60)
    g = np.sin(steps / 4) * np.exp(-steps / 80)
    # second moment factored sketch
    v = np.cumsum(g**2) / (steps + 1)
    ax.plot(steps, g, color=SLATE, label="grad")
    ax.plot(steps, np.sqrt(v + 1e-8), color=TEAL, lw=2, label="factored RMS sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_ijepa_predictor(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.2, 3.5, 1.8, "context\ntokens", fc=TEAL, fs=12)
    box(ax, 4.3, 2.2, 3.5, 1.8, "predictor\n(narrow)", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.2, 2.2, 3.5, 1.8, "target\nrepr", fc=DEEP, fs=12)
    for x0, x1 in [(3.9, 4.3), (7.8, 8.2)]:
        ax.annotate("", xy=(x1, 3.1), xytext=(x0, 3.1), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_sam_audio(ax, t, rng):
    t_ = np.linspace(0, 4 * np.pi, 300)
    wave = np.sin(t_) + 0.3 * np.sin(3 * t_)
    ax.plot(t_, wave, color=TEAL, lw=1.5)
    # mask region
    ax.axvspan(4, 8, color=GOLD, alpha=0.25, label="prompt span")
    ax.set_xlabel("time")
    ax.set_ylabel("waveform")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_td3(ax, t, rng):
    steps = np.arange(0, 50)
    q1 = np.cumsum(rng.normal(0.05, 0.2, 50))
    q2 = q1 - np.abs(rng.normal(0.3, 0.1, 50))
    ax.plot(steps, q1, color=TEAL, lw=2, label="Qθ1")
    ax.plot(steps, q2, color=GOLD, lw=2, label="Qθ2")
    ax.plot(steps, np.minimum(q1, q2), color=ROSE, lw=2, ls="--", label="min (TD3)")
    ax.legend(fontsize=8)
    ax.set_xlabel("update")
    ax.set_ylabel("Q")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_awq_scales(ax, t, rng):
    ch = np.arange(20)
    s = np.abs(rng.normal(1, 0.4, 20))
    s[::5] *= 3
    ax.stem(ch, s, linefmt=TEAL, markerfmt="o", basefmt=" ")
    ax.set_xlabel("channel")
    ax.set_ylabel("AWQ scale s")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_fast_rp(ax, t, rng):
    # random projection distance preservation
    d_true = rng.uniform(0.5, 3, 40)
    d_rp = d_true + rng.normal(0, 0.15, 40)
    ax.scatter(d_true, d_rp, c=TEAL, s=30, alpha=0.75)
    lim = [0, 3.5]
    ax.plot(lim, lim, "--", color=GOLD, lw=1.5)
    ax.set_xlabel("original distance")
    ax.set_ylabel("RP distance")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_data_quality_score(ax, t, rng):
    dims = ["complete", "valid", "unique", "timely", "accurate"]
    scores = [0.92, 0.88, 0.95, 0.7, 0.85]
    ax.barh(dims, scores, color=TEAL, edgecolor=DEEP)
    ax.axvline(0.9, color=GOLD, ls="--", label="bar")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_ops_review(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["metrics", "alerts", "toil", "risks", "actions"]):
        box(ax, 0.3 + i * 2.4, 1.6, 2.2, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_gloss_eval_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["AUROC", "AUPRC", "ECE", "Brier", "F1"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C229 = [
    ("Gram-Schmidt orthogonalization", d_gram_schmidt),
    ("SBOM vulnerability match chain", d_sbom_chain),
    ("MDL fit plus complexity trade", d_compression_bounds),
    ("Alluvial multi-stage flows", d_alluvial),
    ("Geweke early-late MCMC check", d_geweke_diag),
    ("Clique percolation overlap", d_clique_percolation),
    ("SPADE sequence growth path", d_spam_growth),
    ("Target leakage time boundary", d_target_leakage_timeline),
    ("Randomized SVD error vs rank", d_randomized_svd),
    ("Box-Cox transform family", d_box_cox),
    ("Precision-recall operating curve", d_precision_recall_curve),
    ("RMSProp second-moment track", d_rmsprop),
    ("VICReg inv-var-cov triplet", d_vicreg),
    ("RT-DETR hybrid query cascade", d_rt_detr),
    ("Conservative Q offline penalty", d_conservative_q),
    ("Mixture-of-Depths capacity bars", d_mixture_of_depths),
    ("BIGCLAM overlapping affiliations", d_bigclam),
    ("Data unit-test checklist", d_unit_test_data),
    ("SLO availability dashboard", d_slo_dashboard),
    ("Glossary feature-encoding strip", d_gloss_feat_strip),
]

C230 = [
    ("Woodbury low-rank inverse", d_woodbury),
    ("Incident severity histogram", d_incident_sev),
    ("PAC-Bayes KL prior gap", d_pac_bayes_kl_prior),
    ("Radar multi-metric profile", d_radar_chart),
    ("Horseshoe prior density spike", d_horseshoe),
    ("Streaming k-means centers", d_stream_kmeans),
    ("BIDE closed sequence extend", d_bide_closed),
    ("Leave-one-group validation", d_leave_one_group),
    ("Factor analysis loading heat", d_factor_analysis),
    ("Robust vs OLS outlier pull", d_robust_regression),
    ("Brier skill score bars", d_brier_skill),
    ("Adafactor factored second moment", d_adafactor),
    ("I-JEPA narrow predictor path", d_ijepa_predictor),
    ("Audio prompt span mask", d_sam_audio),
    ("TD3 clipped double Q", d_td3),
    ("AWQ per-channel scales stem", d_awq_scales),
    ("Random projection distance map", d_fast_rp),
    ("Data quality dimension scores", d_data_quality_score),
    ("Ops review agenda strip", d_ops_review),
    ("Glossary evaluation metric strip", d_gloss_eval_strip),
]


def embed(cycle: int, topics: list) -> None:
    assert len(topics) == len(CHS)
    for i, (title, fn) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.8, 4.0), subplot_kw=dict(polar=(fn is d_radar_chart)))
        if fn is d_radar_chart:
            # polar axes already
            pass
        rng = np.random.default_rng(cycle * 1000 + i * 23 + 5)
        try:
            if fn is d_radar_chart:
                fig2, ax2 = plt.subplots(figsize=(7.8, 4.0), subplot_kw=dict(polar=True))
                plt.close(fig)
                fig = fig2
                ax = ax2
            fn(ax, title, rng)
        except Exception as e:
            ax.clear()
            if hasattr(ax, "set_theta_offset"):
                # recreate cartesian on failure for polar
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(7.8, 4.0))
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

    m = {229: C229, 230: C230}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [229, 230]
    for c in cycles:
        embed(c, m[c])
