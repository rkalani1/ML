#!/usr/bin/env python3
"""Cycle-207/208 quality densify: novel scientific teal teaching panels."""
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
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc,
            edgecolor="none",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        t,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        fontweight="bold",
    )


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


# ---------- C207 drawers (chapter-aligned, novel titles) ----------


def d_svd_energy(ax, t, rng):
    k = np.arange(1, 21)
    s = 12 * np.exp(-0.35 * (k - 1)) + 0.4
    energy = np.cumsum(s**2) / np.sum(s**2)
    ax.bar(k, s, color=TEAL, edgecolor=DEEP, alpha=0.9, label="σ_k")
    ax2 = ax.twinx()
    ax2.plot(k, energy, "o-", color=GOLD, lw=2, label="cum energy")
    ax.set_xlabel("component k")
    ax.set_ylabel("singular value")
    ax2.set_ylabel("cumulative energy")
    ax.legend(fontsize=7, loc="upper right")
    ax2.legend(fontsize=7, loc="center right")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_deid_map(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    steps = [
        ("direct\nIDs", ROSE),
        ("quasi-IDs", GOLD),
        ("suppress /\ngeneralize", TEAL),
        ("re-ID\nrisk check", DEEP),
    ]
    for i, (lab, c) in enumerate(steps):
        tc = INK if c == GOLD else "white"
        box(ax, 0.4 + i * 2.9, 1.6, 2.7, 1.8, lab, fc=c, fs=10, tc=tc)
        if i < 3:
            ax.annotate(
                "",
                xy=(3.2 + i * 2.9, 2.5),
                xytext=(3.0 + i * 2.9, 2.5),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3),
            )
    ax.text(6, 4.3, "minimal risk de-identification map", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_rademacher(ax, t, rng):
    n = np.logspace(1.5, 4, 40)
    # complexity ~ sqrt(log|H|/n) caricature
    rad = np.sqrt(np.log(50) / n) + 0.02
    ax.loglog(n, rad, color=TEAL, lw=2.2)
    ax.fill_between(n, 0, rad, color=TEAL, alpha=0.15)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("Rademacher complexity sketch")
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.55, 0.75, "smaller ball → tighter gen. gap", transform=ax.transAxes, fontsize=9, color=INK)
    style(ax, t)


def d_colorblind_palette(ax, t, rng):
    # Okabe-Ito inspired safe channels vs rainbow hazard
    safe = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    rainbow = plt.cm.jet(np.linspace(0, 1, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, c in enumerate(safe):
        ax.add_patch(Rectangle((0.5 + i * 1.5, 2.2), 1.3, 1.2, facecolor=c, edgecolor=INK, lw=0.5))
    for i, c in enumerate(rainbow):
        ax.add_patch(Rectangle((0.5 + i * 1.5, 0.5), 1.3, 1.2, facecolor=c, edgecolor=INK, lw=0.5))
    ax.text(5, 3.6, "colorblind-safe channels", ha="center", fontsize=10, color=TEAL, fontweight="bold")
    ax.text(5, 0.15, "rainbow hazard (avoid)", ha="center", fontsize=10, color=ROSE, fontweight="bold")
    style(ax, t)


def d_jeffreys_beta(ax, t, rng):
    x = np.linspace(0.001, 0.999, 300)
    # Beta(0.5,0.5) Jeffreys vs Beta(1,1) uniform vs Beta(2,2)
    def beta_pdf(a, b):
        # unnormalized then scale for teaching
        raw = x ** (a - 1) * (1 - x) ** (b - 1)
        return raw / np.trapezoid(raw, x)

    ax.plot(x, beta_pdf(0.5, 0.5), color=TEAL, lw=2, label="Jeffreys Beta(½,½)")
    ax.plot(x, beta_pdf(1, 1), color=GOLD, lw=2, label="uniform Beta(1,1)")
    ax.plot(x, beta_pdf(2, 2), color=SLATE, lw=1.8, ls="--", label="Beta(2,2)")
    ax.set_xlabel("θ")
    ax.set_ylabel("prior density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_hdbscan_tree(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # condensed tree sketch
    segs = [
        ((2, 5), (2, 3.5)),
        ((2, 3.5), (1, 2)),
        ((2, 3.5), (3.2, 2.2)),
        ((5.5, 5), (5.5, 2.8)),
        ((5.5, 2.8), (4.5, 1.5)),
        ((5.5, 2.8), (6.8, 1.2)),
        ((8, 4.5), (8, 1.8)),
    ]
    for a, b in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=TEAL, lw=2.2)
        ax.plot(*b, "o", color=DEEP, ms=8)
    ax.axhline(2.5, color=GOLD, ls="--", lw=1.8)
    ax.text(8.5, 2.6, "cut λ", color=GOLD, fontsize=10, fontweight="bold")
    ax.text(5, 5.6, "HDBSCAN condensed hierarchy", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_prefixspan(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.8, 3.2, 1.5, "sequences\nSDB", fc=SLATE, fs=11)
    box(ax, 4.3, 2.8, 3.4, 1.5, "prefix α\nproject", fc=TEAL, fs=11)
    box(ax, 8.3, 2.8, 3.2, 1.5, "grow if\nsupport ok", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.3, 3.5), xytext=(3.6, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("", xy=(8.3, 3.5), xytext=(7.7, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(6, 1.3, "PrefixSpan: projected DB, no candidate enumerate", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_loo_target_enc(ax, t, rng):
    cats = ["A", "B", "C", "D", "E"]
    naive = np.array([0.72, 0.41, 0.88, 0.55, 0.33])
    loo = naive - rng.uniform(0.05, 0.15, size=5)  # leave-one-out shrinks peaks
    x = np.arange(len(cats))
    ax.bar(x - 0.18, naive, 0.35, color=ROSE, label="in-fold mean (leak risk)")
    ax.bar(x + 0.18, loo, 0.35, color=TEAL, label="leave-one-out / CV")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("encoded P(y|cat)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_ica_unmix(ax, t, rng):
    tgrid = np.linspace(0, 4 * np.pi, 300)
    s1 = np.sin(tgrid)
    s2 = np.sign(np.sin(1.7 * tgrid + 0.3))
    # mixed observations
    x1 = 0.7 * s1 + 0.5 * s2
    x2 = -0.3 * s1 + 0.9 * s2
    ax.plot(tgrid, x1, color=SLATE, lw=1.2, alpha=0.7, label="mixed x1")
    ax.plot(tgrid, x2 - 2.5, color=SLATE, lw=1.2, alpha=0.7, label="mixed x2")
    ax.plot(tgrid, s1 - 5, color=TEAL, lw=1.6, label="source s1")
    ax.plot(tgrid, s2 - 7.5, color=GOLD, lw=1.6, label="source s2")
    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, axis="x", alpha=0.2)
    style(ax, t)


def d_huber_delta(ax, t, rng):
    r = np.linspace(-4, 4, 300)
    delta = 1.2
    quad = 0.5 * r**2
    huber = np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
    ax.plot(r, quad, color=SLATE, lw=1.5, ls="--", label="½ r²")
    ax.plot(r, huber, color=TEAL, lw=2.2, label=f"Huber δ={delta}")
    ax.axvline(delta, color=GOLD, ls=":", lw=1.5)
    ax.axvline(-delta, color=GOLD, ls=":", lw=1.5)
    ax.set_xlabel("residual r")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_youden_j(ax, t, rng):
    fpr = np.linspace(0, 1, 200)
    tpr = 1 - (1 - fpr) ** 2.2  # concave ROC
    j = tpr - fpr
    i = int(np.argmax(j))
    ax.plot(fpr, tpr, color=TEAL, lw=2.2, label="ROC")
    ax.plot([0, 1], [0, 1], "--", color=SLATE, lw=1)
    ax.plot([fpr[i], fpr[i]], [fpr[i], tpr[i]], color=ROSE, lw=2)
    ax.plot(fpr[i], tpr[i], "o", color=GOLD, ms=10)
    ax.text(fpr[i] + 0.04, (tpr[i] + fpr[i]) / 2, f"J={j[i]:.2f}", color=ROSE, fontsize=10, fontweight="bold")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gelu_swish(ax, t, rng):
    x = np.linspace(-4, 4, 300)
    # GELU approx and Swish
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    swish = x / (1 + np.exp(-x))
    relu = np.maximum(0, x)
    ax.plot(x, relu, color=SLATE, lw=1.5, ls="--", label="ReLU")
    ax.plot(x, gelu, color=TEAL, lw=2.2, label="GELU")
    ax.plot(x, swish, color=GOLD, lw=2, label="Swish/SiLU")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("φ(x)")
    style(ax, t)


def d_moco_queue(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.6, 3.0, 1.6, "query\nencoder", fc=TEAL, fs=11)
    box(ax, 4.0, 2.6, 3.2, 1.6, "momentum\nkey enc", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.0, 1.5, 3.5, 3.2, "queue K\nnegatives", fc=DEEP, fs=12)
    # queue slots
    for i in range(5):
        ax.add_patch(Rectangle((8.3, 1.8 + i * 0.5), 2.9, 0.4, facecolor=MINT if i % 2 == 0 else TEAL, alpha=0.5, edgecolor=DEEP))
    ax.annotate("", xy=(4.0, 3.4), xytext=(3.4, 3.4), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.0, 3.4), xytext=(7.2, 3.4), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.5))
    ax.text(6, 0.7, "MoCo: contrast query vs queued keys", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_ctc_paths(ax, t, rng):
    # time x labels lattice with blank
    labels = ["∅", "C", "A", "T"]
    T = 8
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    for i, lab in enumerate(labels):
        ax.text(-0.7, i, lab, ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
    # highlight one alignment path collapsing to CAT
    path = [(0, 0), (1, 1), (2, 1), (3, 0), (4, 2), (5, 0), (6, 3), (7, 0)]
    for ti in range(T):
        for li in range(len(labels)):
            ax.plot(ti, li, "o", color=SLATE, ms=6, alpha=0.35)
    xs, ys = zip(*path)
    ax.plot(xs, ys, "-o", color=TEAL, lw=2.4, ms=9)
    ax.set_xlabel("time step")
    ax.set_yticks([])
    ax.text(3.5, 3.6, "CTC paths → collapse repeats + blanks", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_dueling_q(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.0, 2.8, 1.8, "shared\ntrunk", fc=SLATE, fs=11)
    box(ax, 4.2, 3.1, 3.0, 1.4, "V(s)", fc=TEAL, fs=12)
    box(ax, 4.2, 1.2, 3.0, 1.4, "A(s,a)", fc=DEEP, fs=12)
    box(ax, 8.2, 2.0, 3.2, 1.8, "Q = V + A\n− mean A", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.2, 3.7), xytext=(3.3, 3.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(4.2, 1.9), xytext=(3.3, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(8.2, 2.9), xytext=(7.2, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(8.2, 2.7), xytext=(7.2, 2.0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, t)


def d_int8_calib(ax, t, rng):
    w = rng.normal(0, 1.1, 2000)
    ax.hist(w, bins=40, color=TEAL, alpha=0.75, density=True, edgecolor=DEEP, label="fp32 activations")
    # percentile clip range
    lo, hi = np.quantile(w, [0.005, 0.995])
    ax.axvline(lo, color=ROSE, lw=2, label="clip low")
    ax.axvline(hi, color=ROSE, lw=2, label="clip high")
    levels = np.linspace(lo, hi, 9)
    for lv in levels:
        ax.axvline(lv, color=GOLD, alpha=0.45, lw=0.9)
    ax.set_xlabel("activation value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(0.5, 0.9, "PTQ int8: percentile range → 256 bins", transform=ax.transAxes, ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_louvain_steps(ax, t, rng):
    steps = np.arange(1, 9)
    Q = np.array([0.12, 0.28, 0.41, 0.48, 0.52, 0.53, 0.53, 0.53])
    ax.step(steps, Q, where="mid", color=TEAL, lw=2.2)
    ax.plot(steps, Q, "o", color=DEEP, ms=8)
    ax.axvline(5, color=GOLD, ls="--", label="local max / coarsen")
    ax.set_xlabel("Louvain move / pass")
    ax.set_ylabel("modularity Q")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_simpson_paradox(ax, t, rng):
    # two groups reverse aggregate trend
    ax.scatter([1, 2, 3, 4], [1.0, 1.3, 1.6, 1.9], c=TEAL, s=60, label="group A")
    ax.plot([1, 4], [1.0, 1.9], color=TEAL, lw=1.5)
    ax.scatter([2.5, 3.5, 4.5, 5.5], [3.2, 3.5, 3.8, 4.1], c=GOLD, s=60, label="group B")
    ax.plot([2.5, 5.5], [3.2, 4.1], color=GOLD, lw=1.5)
    # aggregate negative slope illusion points
    agg_x = [1.5, 2.5, 3.5, 4.5]
    agg_y = [2.8, 2.4, 2.1, 1.7]
    ax.plot(agg_x, agg_y, "s--", color=ROSE, lw=2, ms=8, label="pooled (reversed)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_shadow_mode(ax, t, rng):
    weeks = np.arange(0, 24)
    live = 0.82 + 0.01 * np.sin(weeks / 3) + rng.normal(0, 0.005, len(weeks))
    shadow = 0.79 + 0.015 * np.sin(weeks / 3 + 0.5) + rng.normal(0, 0.006, len(weeks))
    ax.plot(weeks, live, color=TEAL, lw=2, label="live model metric")
    ax.plot(weeks, shadow, color=GOLD, lw=2, label="shadow challenger")
    ax.fill_between(weeks, live, shadow, color=SLATE, alpha=0.15)
    ax.set_xlabel("week (dual-write)")
    ax.set_ylabel("discrimination proxy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_glossary_loss(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["MSE", "MAE", "Huber", "CE", "focal"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


# ---------- C208 drawers ----------


def d_cond_number(ax, t, rng):
    # ellipse stretched by ill-conditioned matrix
    theta = np.linspace(0, 2 * np.pi, 200)
    # well-conditioned
    ax.plot(np.cos(theta), np.sin(theta), color=TEAL, lw=2, label="κ≈1 ball")
    # ill-conditioned stretch
    ax.plot(2.8 * np.cos(theta), 0.35 * np.sin(theta), color=ROSE, lw=2, label="large κ ellipse")
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(0, -1.3, "κ = σ_max/σ_min stretches level sets", ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_federated_avg(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 4.5, 3.5, 3, 1.2, "server w̄", fc=GOLD, tc=INK, fs=12)
    for i, lab in enumerate(["silo A", "silo B", "silo C", "silo D"]):
        x = 0.5 + i * 3.0
        box(ax, x, 0.8, 2.5, 1.3, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
        ax.annotate("", xy=(6, 3.5), xytext=(x + 1.25, 2.1), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.1))
        ax.annotate("", xy=(x + 1.25, 2.1), xytext=(6, 3.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9, alpha=0.5))
    ax.text(6, 0.25, "FedAvg: local steps → weighted average", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_bvn_triad(ax, t, rng):
    capacity = np.linspace(1, 20, 50)
    bias2 = 2.5 * np.exp(-0.25 * capacity)
    var = 0.02 * capacity**1.3
    noise = np.full_like(capacity, 0.15)
    total = bias2 + var + noise
    ax.plot(capacity, bias2, color=TEAL, lw=2, label="bias²")
    ax.plot(capacity, var, color=GOLD, lw=2, label="variance")
    ax.plot(capacity, noise, color=SLATE, lw=1.5, ls="--", label="irreducible")
    ax.plot(capacity, total, color=ROSE, lw=2.2, label="total risk")
    ax.set_xlabel("model capacity")
    ax.set_ylabel("error component")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_data_ink(ax, t, rng):
    # before/after chartjunk vs clean bars
    cats = ["A", "B", "C", "D"]
    vals = [3.2, 5.1, 2.4, 4.0]
    x = np.arange(len(cats))
    # left: chartjunk
    ax.bar(x - 0.15, vals, 0.3, color=ROSE, alpha=0.7, label="chartjunk heavy")
    for i, v in enumerate(vals):
        ax.plot([i - 0.15, i - 0.15], [0, v], color=SLATE, lw=4, alpha=0.3)
        ax.plot(i - 0.15, v, "*", color=GOLD, ms=14)
    # right: high data-ink
    ax.bar(x + 0.25, vals, 0.3, color=TEAL, edgecolor=DEEP, label="high data-ink")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    ax.text(0.5, 0.92, "Tufte data-ink: maximize informative ink", transform=ax.transAxes, ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_cred_vs_conf(ax, t, rng):
    # posterior density with credible interval vs fixed-width CI sketch
    x = np.linspace(-1, 3, 300)
    post = np.exp(-0.5 * ((x - 0.8) / 0.45) ** 2)
    post = post / np.trapezoid(post, x)
    ax.fill_between(x, post, color=TEAL, alpha=0.35, label="posterior")
    ax.plot(x, post, color=DEEP, lw=2)
    # HDI approx
    ax.axvline(0.1, color=GOLD, lw=2, label="95% credible HDI")
    ax.axvline(1.5, color=GOLD, lw=2)
    ax.axvspan(0.1, 1.5, color=GOLD, alpha=0.12)
    # frequentist CI mark
    ax.plot([0.25, 1.65], [-0.05, -0.05], color=ROSE, lw=3, solid_capstyle="butt", label="95% CI (procedure)")
    ax.set_xlabel("parameter")
    ax.set_ylabel("density")
    ax.set_ylim(-0.12, post.max() * 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_meanshift(ax, t, rng):
    # blob + mode-seeking path
    pts = np.vstack(
        [
            rng.normal([-1, 0], 0.35, size=(40, 2)),
            rng.normal([1.5, 0.8], 0.4, size=(40, 2)),
        ]
    )
    ax.scatter(pts[:, 0], pts[:, 1], c=SLATE, s=18, alpha=0.5)
    # path toward mode
    path = np.array([[-2.2, -1.2], [-1.6, -0.7], [-1.2, -0.3], [-1.0, 0.0], [-0.95, 0.05]])
    ax.plot(path[:, 0], path[:, 1], "-o", color=TEAL, lw=2.2, ms=7, label="mean-shift path")
    ax.plot(-1.0, 0.05, "*", color=GOLD, ms=16, label="mode")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_eclat_tidset(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    items = [("A", "{1,2,4,5}"), ("B", "{1,3,5}"), ("C", "{2,3,4}")]
    for i, (it, tid) in enumerate(items):
        box(ax, 0.5, 4.2 - i * 1.4, 1.8, 1.1, it, fc=TEAL, fs=12)
        ax.text(2.8, 4.6 - i * 1.4, f"tidset {tid}", fontsize=11, color=INK, va="center")
    box(ax, 5.5, 2.2, 4, 1.6, "A∩B = {1,5}\nsupport = 2", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(5.5, 3.0), xytext=(4.8, 4.5), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.5))
    ax.text(5, 5.5, "ECLAT vertical intersect", ha="center", fontsize=11, color=INK, fontweight="bold")
    style(ax, t)


def d_hashing_trick(ax, t, rng):
    # collision rate vs hash dim
    d = np.array([8, 16, 32, 64, 128, 256, 512])
    n_feat = 200
    # approx birthday collision prob caricature
    coll = 1 - np.exp(-n_feat * (n_feat - 1) / (2 * d))
    ax.plot(d, coll, "o-", color=TEAL, lw=2.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hash dimension d")
    ax.set_ylabel("approx collision probability")
    ax.axhline(0.1, color=GOLD, ls="--", label="10% collision")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_umap_fuzzy(ax, t, rng):
    # local fuzzy radius sketch
    pts = rng.normal(0, 1, size=(30, 2))
    ax.scatter(pts[:, 0], pts[:, 1], c=TEAL, s=40, edgecolors=DEEP)
    center = pts[0]
    ax.plot(*center, "o", color=ROSE, ms=12)
    for r, a in [(0.6, 0.35), (1.1, 0.18), (1.7, 0.08)]:
        ax.add_patch(Circle(center, r, fill=True, facecolor=GOLD, alpha=a, edgecolor=GOLD, lw=1))
    ax.set_aspect("equal")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.text(0.02, 0.92, "UMAP fuzzy simplicial neighborhood", transform=ax.transAxes, fontsize=9, color=INK)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_poisson_glm(ax, t, rng):
    x = np.linspace(-1, 2.5, 80)
    log_mu = -0.2 + 0.9 * x
    mu = np.exp(log_mu)
    y = rng.poisson(mu)
    ax.scatter(x, y, c=TEAL, s=22, alpha=0.7, label="counts")
    ax.plot(x, mu, color=GOLD, lw=2.2, label="μ = exp(Xβ)")
    ax.set_xlabel("covariate x")
    ax.set_ylabel("count y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_cost_sensitive(ax, t, rng):
    thr = np.linspace(0.05, 0.95, 50)
    # utility with asymmetric costs
    prev = 0.15
    sens = 1 - thr**1.5
    spec = thr**0.8
    # net benefit-ish: benefit TP - cost FP
    u_bal = sens * prev - (1 - spec) * (1 - prev)
    u_cost = 2.5 * sens * prev - 0.5 * (1 - spec) * (1 - prev)
    ax.plot(thr, u_bal, color=SLATE, lw=1.8, label="balanced utility")
    ax.plot(thr, u_cost, color=TEAL, lw=2.2, label="cost-sensitive utility")
    i = int(np.argmax(u_cost))
    ax.axvline(thr[i], color=GOLD, ls="--", label=f"opt thr≈{thr[i]:.2f}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected utility sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_group_instance_norm(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # batch of feature maps caricature
    for b in range(3):
        for c in range(4):
            ax.add_patch(
                Rectangle(
                    (0.5 + b * 1.6, 0.8 + c * 0.9),
                    1.4,
                    0.75,
                    facecolor=TEAL if c < 2 else MINT,
                    edgecolor=DEEP,
                    alpha=0.85,
                )
            )
    ax.text(2.5, 4.6, "GroupNorm: groups of channels / sample", ha="center", fontsize=9, color=INK)
    for b in range(3):
        for c in range(4):
            ax.add_patch(
                Rectangle(
                    (6.5 + b * 1.6, 0.8 + c * 0.9),
                    1.4,
                    0.75,
                    facecolor=GOLD if (b + c) % 2 == 0 else TEAL,
                    edgecolor=DEEP,
                    alpha=0.75,
                )
            )
    ax.text(8.5, 4.6, "InstanceNorm: per channel / sample", ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_barlow_twins(ax, t, rng):
    # cross-correlation matrix toward identity
    C = rng.normal(0, 0.15, size=(8, 8))
    np.fill_diagonal(C, 0.85 + rng.uniform(0, 0.1, 8))
    im = ax.imshow(C, cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    for i in range(8):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEAL, lw=1.5))
    ax.set_xlabel("embedding dim j")
    ax.set_ylabel("embedding dim i")
    ax.text(0.5, 1.08, "Barlow: on-diag→1, off-diag→0", transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_wav2vec(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    stages = ["waveform", "CNN\nencoder", "masked\ncontext", "contrastive\nunits"]
    cols = [SLATE, TEAL, DEEP, GOLD]
    for i, (s, c) in enumerate(zip(stages, cols)):
        tc = INK if c == GOLD else "white"
        box(ax, 0.4 + i * 3.0, 1.4, 2.7, 1.8, s, fc=c, fs=10, tc=tc)
        if i < 3:
            ax.annotate(
                "",
                xy=(3.2 + i * 3.0, 2.3),
                xytext=(3.0 + i * 3.0, 2.3),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3),
            )
    style(ax, t)


def d_cql_backup(ax, t, rng):
    a = np.arange(6)
    q_data = np.array([1.2, 2.0, 1.5, 0.8, 1.1, 0.5])
    q_ood = q_data + np.array([0.8, 0.2, 1.5, 1.2, 0.9, 1.8])  # overestimate OOD
    q_cql = q_ood - np.array([0.9, 0.15, 1.4, 1.1, 0.85, 1.6])
    ax.bar(a - 0.25, q_ood, 0.25, color=ROSE, label="naive Q (OOD↑)")
    ax.bar(a, q_data, 0.25, color=SLATE, label="data actions")
    ax.bar(a + 0.25, q_cql, 0.25, color=TEAL, label="CQL-regularized")
    ax.set_xlabel("action index")
    ax.set_ylabel("Q estimate")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_svd_compress(ax, t, rng):
    ranks = np.array([1, 2, 4, 8, 16, 32, 64])
    full = 64 * 64
    params = ranks * (64 + 64)
    err = 0.6 * np.exp(-0.08 * ranks) + 0.02
    ax.plot(ranks, params / full, "o-", color=TEAL, lw=2, label="param ratio")
    ax2 = ax.twinx()
    ax2.plot(ranks, err, "s-", color=GOLD, lw=2, label="recon error")
    ax.set_xlabel("rank r")
    ax.set_ylabel("parameters / full")
    ax2.set_ylabel("reconstruction error")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_motif_zscore(ax, t, rng):
    motifs = ["feed\nforward", "mutual\nedge", "3-cycle", "bi-fan", "bi-parallel"]
    z = np.array([3.2, 0.4, 2.5, -1.1, 1.8])
    colors = [TEAL if v > 1.5 else (ROSE if v < -1 else SLATE) for v in z]
    ax.barh(motifs, z, color=colors, edgecolor=DEEP)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(1.96, color=GOLD, ls="--", label="|z|≈1.96")
    ax.axvline(-1.96, color=GOLD, ls="--")
    ax.set_xlabel("motif Z-score vs randomized null")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_fdr_multiplicity(ax, t, rng):
    m = np.arange(1, 50)
    fwer = 1 - (1 - 0.05) ** m
    bh = 0.05 * m / m  # alpha flat for illustration of target
    ax.plot(m, fwer, color=ROSE, lw=2, label="FWER if uncorrected α=0.05 each")
    ax.axhline(0.05, color=TEAL, lw=2, label="target FDR/FWER 0.05")
    ax.plot(m, 0.05 * m / 20, color=GOLD, lw=1.8, ls="--", label="BH critical slope sketch")
    ax.set_xlabel("number of tests m")
    ax.set_ylabel("error rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_net_benefit_band(ax, t, rng):
    thr = np.linspace(0.05, 0.8, 60)
    # model net benefit caricature
    nb = 0.12 - 0.2 * (thr - 0.25) ** 2
    treat_all = 0.15 - thr * 0.35
    treat_none = np.zeros_like(thr)
    ax.plot(thr, nb, color=TEAL, lw=2.2, label="model")
    ax.plot(thr, treat_all, color=GOLD, lw=1.8, label="treat all")
    ax.plot(thr, treat_none, color=SLATE, lw=1.5, ls="--", label="treat none")
    ax.fill_between(thr, treat_none, nb, where=nb > 0, color=TEAL, alpha=0.15)
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_glossary_calib(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["ECE", "MCE", "Brier", "logloss", "reliab."]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C207 = [
    ("Singular value energy decay spectrum", d_svd_energy),
    ("Minimal risk de-identification map", d_deid_map),
    ("Rademacher complexity sample ball", d_rademacher),
    ("Colorblind-safe palette channels", d_colorblind_palette),
    ("Jeffreys Beta prior vs uniform", d_jeffreys_beta),
    ("HDBSCAN condensed tree cut", d_hdbscan_tree),
    ("PrefixSpan projected database grow", d_prefixspan),
    ("Leave-one-out target encoding", d_loo_target_enc),
    ("ICA unmixing independence axes", d_ica_unmix),
    ("Huber loss delta transition", d_huber_delta),
    ("Youden J ROC threshold", d_youden_j),
    ("GELU vs Swish activations", d_gelu_swish),
    ("MoCo momentum queue keys", d_moco_queue),
    ("CTC blank collapse alignment", d_ctc_paths),
    ("Dueling Q value advantage", d_dueling_q),
    ("PTQ int8 percentile calibration", d_int8_calib),
    ("Louvain modularity climb passes", d_louvain_steps),
    ("Simpson paradox subgroup reverse", d_simpson_paradox),
    ("Shadow mode dual-write monitor", d_shadow_mode),
    ("Glossary loss function strip", d_glossary_loss),
]

C208 = [
    ("Condition number stretch ellipse", d_cond_number),
    ("Federated silo average rounds", d_federated_avg),
    ("Bias variance noise triad", d_bvn_triad),
    ("Tufte data-ink ratio bars", d_data_ink),
    ("Credible vs confidence interval", d_cred_vs_conf),
    ("Mean-shift mode seeking path", d_meanshift),
    ("ECLAT vertical tidset intersect", d_eclat_tidset),
    ("Hashing trick collision rates", d_hashing_trick),
    ("UMAP fuzzy simplicial sketch", d_umap_fuzzy),
    ("Poisson GLM log-link mean", d_poisson_glm),
    ("Cost-sensitive threshold shift", d_cost_sensitive),
    ("GroupNorm vs InstanceNorm axes", d_group_instance_norm),
    ("Barlow Twins cross-corr identity", d_barlow_twins),
    ("Wav2vec contrastive unit path", d_wav2vec),
    ("CQL conservative Q backup", d_cql_backup),
    ("Low-rank SVD weight compress", d_svd_compress),
    ("Motif Z-score randomization null", d_motif_zscore),
    ("Multiplicity FDR vs FWER", d_fdr_multiplicity),
    ("Decision curve net benefit band", d_net_benefit_band),
    ("Glossary calibration metric strip", d_glossary_calib),
]


def embed(cycle: int, topics: list) -> None:
    assert len(topics) == len(CHS), f"need {len(CHS)} topics, got {len(topics)}"
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

    m = {207: C207, 208: C208}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [207, 208]
    for c in cycles:
        embed(c, m[c])
