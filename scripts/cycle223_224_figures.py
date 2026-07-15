#!/usr/bin/env python3
"""Cycle-223/224 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle, Arc, Ellipse, Wedge, Polygon

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


# ----- C223 -----


def d_givens_rotation(ax, t, rng):
    """Givens rotation zeros one entry in a plane."""
    th = np.deg2rad(35)
    c, s = np.cos(th), np.sin(th)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.8)
    ax.set_aspect("equal")
    # vector with both components
    v = np.array([1.0, 0.9])
    G = np.array([[c, s], [-s, c]])
    w = G @ v
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.annotate("", xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5))
    ax.text(v[0] + 0.05, v[1] + 0.05, "v", color=SLATE, fontsize=11)
    ax.text(w[0] + 0.05, w[1] + 0.08, "Gv (y≈0)", color=TEAL, fontsize=11, fontweight="bold")
    ax.plot([-1.4, 1.4], [0, 0], color=GOLD, ls="--", lw=1.2)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_model_card_sections(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    items = [
        (0.3, 3.5, "intended\nuse", TEAL),
        (3.2, 3.5, "training\ndata", DEEP),
        (6.1, 3.5, "metrics &\nlimits", GOLD),
        (9.0, 3.5, "ethical\nfactors", ROSE),
        (1.5, 1.0, "caveats", SLATE),
        (5.0, 1.0, "eval harness", MINT),
        (8.5, 1.0, "version", TEAL),
    ]
    for x, y, lab, c in items:
        box(ax, x, y, 2.6, 1.6, lab, fc=c, tc=("white" if c not in (GOLD, MINT) else INK), fs=10)
    style(ax, t)


def d_uniform_convergence(ax, t, rng):
    """Uniform deviation sup|Ê-E| shrinks with n (sketch)."""
    n = np.logspace(1, 4, 60)
    bound = np.sqrt(np.log(n) / n) * 1.5
    ax.semilogx(n, bound, color=TEAL, lw=2.2)
    ax.fill_between(n, 0, bound, color=TEAL, alpha=0.15)
    ax.set_xlabel("n")
    ax.set_ylabel("uniform dev. sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_streamgraph(ax, t, rng):
    """Theme-river / streamgraph stacked flows."""
    x = np.linspace(0, 10, 100)
    layers = []
    base = np.zeros_like(x)
    colors = [TEAL, DEEP, GOLD, MINT, ROSE]
    for i, c in enumerate(colors):
        amp = 0.4 + 0.15 * i
        y = amp * (1 + 0.5 * np.sin(x * (0.8 + 0.2 * i) + i)) * np.exp(-0.02 * (x - 5) ** 2)
        layers.append((base.copy(), base + y, c))
        base = base + y
    # center the stream
    mid = base / 2
    for lo, hi, c in layers:
        ax.fill_between(x, lo - mid, hi - mid, color=c, alpha=0.75)
    ax.set_xlabel("time")
    ax.set_yticks([])
    ax.set_ylabel("stacked themes")
    style(ax, t)


def d_variational_free_energy(ax, t, rng):
    """ELBO = E_q[log p] - KL(q||prior); free energy landscape sketch."""
    th = np.linspace(-3, 3, 200)
    nll = 0.5 * (th - 0.8) ** 2 + 0.3
    kl = 0.2 * th**2
    free = nll + kl
    ax.plot(th, nll, color=SLATE, lw=1.8, label="−E_q log p")
    ax.plot(th, kl, color=GOLD, lw=1.8, label="KL(q||p0)")
    ax.plot(th, free, color=TEAL, lw=2.4, label="free energy")
    ax.axvline(th[np.argmin(free)], color=ROSE, ls="--", alpha=0.7)
    ax.set_xlabel("variational param")
    ax.set_ylabel("energy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_optics_reachability(ax, t, rng):
    """OPTICS reachability plot: valleys = clusters."""
    n = 80
    x = np.arange(n)
    # synthetic reachability with two valleys
    r = (
        1.2
        + 0.15 * rng.normal(0, 1, n)
        + 1.5 * np.exp(-0.5 * ((x - 20) / 4) ** 2) * 0
        + np.where((x > 10) & (x < 30), 0.2, 0)
        + np.where((x > 45) & (x < 65), 0.25, 0)
    )
    r[10:30] = 0.25 + 0.08 * np.abs(np.sin(np.linspace(0, 3 * np.pi, 20)))
    r[45:65] = 0.3 + 0.1 * np.abs(np.sin(np.linspace(0, 2 * np.pi, 20)))
    r[0:10] = 1.0 + 0.2 * rng.random(10)
    r[30:45] = 1.1 + 0.15 * rng.random(15)
    r[65:] = 1.0 + 0.2 * rng.random(n - 65)
    ax.fill_between(x, 0, r, color=TEAL, alpha=0.5)
    ax.plot(x, r, color=DEEP, lw=1.5)
    ax.axhline(0.55, color=GOLD, ls="--", label="ξ cut")
    ax.set_xlabel("OPTICS order")
    ax.set_ylabel("reachability dist")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_prefixspan_project(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["SDB", "prefix α", "α-projected\nDB", "grow +\nfreq"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
    style(ax, t)


def d_woe_iv_bins(ax, t, rng):
    """Weight of Evidence / IV per bin."""
    bins = np.arange(6)
    woe = np.array([-0.8, -0.3, 0.1, 0.4, 0.7, 0.2])
    colors = [ROSE if w < 0 else TEAL for w in woe]
    ax.bar(bins, woe, color=colors, edgecolor=DEEP)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("bin")
    ax.set_ylabel("WoE")
    iv = np.sum(np.abs(woe) * 0.15)
    ax.text(0.98, 0.95, f"IV sketch≈{iv:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=SLATE)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_sparse_pca(ax, t, rng):
    """Sparse PCA loadings: many exact zeros."""
    load = rng.normal(0, 0.3, 16)
    load[np.abs(load) < 0.35] = 0
    load[2] = 0.9
    load[7] = -0.7
    load[12] = 0.55
    colors = [TEAL if v != 0 else SLATE for v in load]
    ax.barh(np.arange(16), load, color=colors, edgecolor=DEEP)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("loading")
    ax.set_ylabel("feature")
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_gam_partial(ax, t, rng):
    """GAM smooth partial effect with CI ribbon."""
    x = np.linspace(0, 10, 100)
    f = np.sin(x / 1.5) + 0.05 * x
    se = 0.15 + 0.05 * np.sin(x)
    ax.fill_between(x, f - 1.96 * se, f + 1.96 * se, color=TEAL, alpha=0.25)
    ax.plot(x, f, color=TEAL, lw=2.2)
    ax.axhline(0, color=SLATE, ls=":", lw=1)
    ax.set_xlabel("x_j")
    ax.set_ylabel("s_j(x_j)")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_cost_sensitive(ax, t, rng):
    """Cost matrix heat: FN vs FP asymmetric costs."""
    C = np.array([[0.0, 1.0], [10.0, 0.0]])
    im = ax.imshow(C, cmap="YlOrRd", vmin=0, vmax=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_xlabel("prediction (not causation)")
    style(ax, t)


def d_sharpness_aware(ax, t, rng):
    """Neighborhood max loss → SAM update direction."""
    xs = np.linspace(-2, 2, 200)
    sharp = xs**2 + 0.3 * np.sin(12 * xs)
    flat = 0.25 * xs**2
    ax.plot(xs, sharp, color=SLATE, lw=1.6, label="sharp basin")
    ax.plot(xs, flat, color=TEAL, lw=2.2, label="flat basin")
    # epsilon ball
    ax.axvspan(0.4, 0.7, color=GOLD, alpha=0.25, label="ε-ball")
    ax.annotate("", xy=(0.55, 0.55), xytext=(0.55, 0.15), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    ax.text(0.75, 0.4, "max loss\nin ball", color=ROSE, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlabel("weight")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_barlow_twins(ax, t, rng):
    """Barlow Twins: cross-correlation → identity."""
    C = np.eye(6) * 0.9 + rng.normal(0, 0.05, (6, 6))
    np.fill_diagonal(C, 1.0)
    C = (C + C.T) / 2
    # off-diag push toward 0, diag toward 1 shown as target
    ax.imshow(C, cmap="RdYlGn", vmin=-0.3, vmax=1.0)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{C[i, j]:.1f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlabel("embedding dim")
    ax.set_ylabel("embedding dim")
    ax.text(2.5, -0.9, "cross-corr C → I  (redundancy reduction)", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_flamingo_resampler(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.4, 3.5, 3.2, 1.8, "vision\nencoder", fc=TEAL, fs=12)
    box(ax, 4.2, 3.5, 3.4, 1.8, "Perceiver\nresampler", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.2, 3.5, 3.4, 1.8, "frozen\nLLM", fc=DEEP, fs=12)
    box(ax, 3.0, 1.0, 6, 1.5, "few visual tokens → text conditioning", fc=MINT, fs=11)
    for x0, x1 in [(3.6, 4.2), (7.6, 8.2)]:
        ax.annotate("", xy=(x1, 4.4), xytext=(x0, 4.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.6))
    style(ax, t)


def d_successor_features(ax, t, rng):
    """Successor features ψ: expected discounted future features."""
    t_ = np.arange(0, 25)
    gamma = 0.9
    phi = np.exp(-0.15 * t_)
    psi = np.array([np.sum((gamma ** np.arange(len(phi) - k)) * phi[k:]) for k in range(len(phi))])
    ax.plot(t_, phi, color=GOLD, lw=2, label="φ(s_t)")
    ax.plot(t_, psi, color=TEAL, lw=2.2, label="ψ^π (successor)")
    ax.set_xlabel("time")
    ax.set_ylabel("feature / SF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_smoothquant(ax, t, rng):
    """SmoothQuant: migrate activation outliers into weights."""
    ch = np.arange(12)
    act_out = np.abs(rng.normal(1, 0.3, 12))
    act_out[3] = 4.5
    act_out[9] = 3.8
    mig = np.sqrt(act_out)  # migration scales
    act_smooth = act_out / mig
    w = 0.35
    ax.bar(ch - w / 2, act_out, width=w, color=ROSE, label="act before")
    ax.bar(ch + w / 2, act_smooth, width=w, color=TEAL, label="act after smooth")
    ax.set_xlabel("channel")
    ax.set_ylabel("activation scale")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_motif_zscore(ax, t, rng):
    """Network motif significance vs random null."""
    motifs = ["fan-in", "fan-out", "cascade", "FFL", "cycle3"]
    z = np.array([2.1, 0.4, 3.5, 4.2, -1.1])
    colors = [TEAL if v > 1.96 else (ROSE if v < -1.96 else SLATE) for v in z]
    ax.barh(motifs, z, color=colors, edgecolor=DEEP)
    ax.axvline(1.96, color=GOLD, ls="--", lw=1.2)
    ax.axvline(-1.96, color=GOLD, ls="--", lw=1.2)
    ax.set_xlabel("z-score vs null")
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_pii_redaction(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.0, 2.6, 2.0, "raw\ntext", fc=SLATE, fs=12)
    box(ax, 3.4, 2.0, 2.6, 2.0, "NER /\nregex", fc=TEAL, fs=12)
    box(ax, 6.4, 2.0, 2.6, 2.0, "redact /\nhash", fc=GOLD, tc=INK, fs=11)
    box(ax, 9.4, 2.0, 2.3, 2.0, "safe\nexport", fc=MINT, fs=12)
    for x in [3.0, 6.0, 9.0]:
        ax.annotate("", xy=(x + 0.4, 3), xytext=(x, 3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    style(ax, t)


def d_sre_error_budget(ax, t, rng):
    days = np.arange(1, 31)
    burn = np.cumsum(rng.exponential(0.8, 30))
    budget = np.full_like(days, 20.0, dtype=float)
    ax.plot(days, burn, color=TEAL, lw=2.2, label="burn")
    ax.plot(days, budget, color=GOLD, ls="--", lw=2, label="budget")
    ax.fill_between(days, burn, budget, where=burn < budget, color=MINT, alpha=0.25)
    ax.fill_between(days, burn, budget, where=burn >= budget, color=ROSE, alpha=0.25)
    ax.set_xlabel("day")
    ax.set_ylabel("error budget units")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gloss_cluster_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["k-means", "GMM", "DBSCAN", "HDBSCAN", "spectral"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


# ----- C224 -----


def d_lanczos_tridiag(ax, t, rng):
    """Lanczos: A ≈ Q T Qᵀ with tridiagonal T."""
    n = 6
    T = np.zeros((n, n))
    alpha = rng.uniform(1, 3, n)
    beta = rng.uniform(0.2, 0.8, n - 1)
    np.fill_diagonal(T, alpha)
    for i in range(n - 1):
        T[i, i + 1] = beta[i]
        T[i + 1, i] = beta[i]
    ax.imshow(T, cmap="Greens")
    for i in range(n):
        for j in range(n):
            if abs(T[i, j]) > 1e-9:
                ax.text(j, i, f"{T[i, j]:.1f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(2.5, n + 0.2, "tridiagonal Krylov projection", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_red_team_loop(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    stages = [(1.5, 4.5, "probe"), (5, 5.0, "find\nfail"), (8, 3.5, "patch"), (5, 1.5, "retest"), (1.5, 2.5, "scope")]
    for x, y, lab in stages:
        box(ax, x - 1.1, y - 0.6, 2.2, 1.2, lab, fc=TEAL, fs=10)
    for i in range(len(stages)):
        j = (i + 1) % len(stages)
        ax.annotate(
            "",
            xy=(stages[j][0], stages[j][1]),
            xytext=(stages[i][0], stages[i][1]),
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5, connectionstyle="arc3,rad=0.2"),
        )
    style(ax, t)


def d_margin_theory(ax, t, rng):
    """Large-margin: geometric margin γ vs support vectors."""
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    # hyperplane
    ax.plot([-0.5, 4.5], [3.2, -0.5], color=INK, lw=2)
    # margins parallel
    ax.plot([-0.5, 4.5], [3.7, 0.0], color=TEAL, ls="--", lw=1.5)
    ax.plot([-0.5, 4.5], [2.7, -1.0], color=TEAL, ls="--", lw=1.5)
    # points
    pos = rng.normal([3.5, 3.0], 0.35, (8, 2))
    neg = rng.normal([0.8, 0.6], 0.35, (8, 2))
    ax.scatter(pos[:, 0], pos[:, 1], c=TEAL, s=50, edgecolors=DEEP)
    ax.scatter(neg[:, 0], neg[:, 1], c=ROSE, s=50, edgecolors=DEEP)
    # support vectors
    sv = np.array([[2.0, 1.55], [1.6, 1.9], [2.8, 1.0]])
    ax.scatter(sv[:, 0], sv[:, 1], facecolors="none", edgecolors=GOLD, s=160, linewidths=2, label="SV")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    style(ax, t)


def d_qq_plot(ax, t, rng):
    """Normal Q-Q plot teaching panel."""
    sample = np.sort(rng.normal(0, 1, 80))
    # theoretical quantiles
    probs = (np.arange(1, 81) - 0.5) / 80
    from math import erfc, sqrt  # noqa: keep simple approx via numpy
    theo = np.sqrt(2) * erfinv_approx(2 * probs - 1)
    ax.scatter(theo, sample, c=TEAL, s=22, alpha=0.8, edgecolors=DEEP, linewidths=0.3)
    lim = [-2.8, 2.8]
    ax.plot(lim, lim, "--", color=GOLD, lw=1.8)
    ax.set_xlabel("theoretical quantile")
    ax.set_ylabel("sample quantile")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def erfinv_approx(x):
    """Simple inverse erf approx for QQ (vectorized)."""
    x = np.asarray(x, dtype=float)
    # Beasley-Springer/Moro-ish low precision
    a = 0.147
    sign = np.sign(x)
    ln = np.log(1 - x**2)
    t = 2 / (np.pi * a) + ln / 2
    return sign * np.sqrt(np.sqrt(t**2 - ln / a) - t)


def d_slice_sampling(ax, t, rng):
    """Slice sampling: horizontal slice under density."""
    x = np.linspace(-4, 4, 400)
    p = np.exp(-0.5 * x**2) * (1 + 0.3 * np.cos(3 * x))
    p = p / np.trapezoid(p, x)
    ax.fill_between(x, 0, p, color=TEAL, alpha=0.35)
    ax.plot(x, p, color=DEEP, lw=2)
    y0 = 0.12
    # slice interval
    mask = p >= y0
    ax.axhline(y0, color=GOLD, ls="--", lw=1.5)
    xs = x[mask]
    if len(xs):
        ax.plot([xs.min(), xs.max()], [y0, y0], color=ROSE, lw=4, solid_capstyle="round", label="slice")
    ax.plot([0.5], [y0], "o", color=ROSE, ms=10)
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mean_shift(ax, t, rng):
    """Mean-shift: points drift toward density modes."""
    modes = np.array([[-1.5, 0], [1.5, 0.5]])
    pts = np.vstack(
        [
            rng.normal(modes[0], 0.35, (40, 2)),
            rng.normal(modes[1], 0.4, (40, 2)),
        ]
    )
    ax.scatter(pts[:, 0], pts[:, 1], c=TEAL, s=20, alpha=0.5)
    # arrows toward nearest mode
    for p in pts[::4]:
        m = modes[np.argmin(np.linalg.norm(modes - p, axis=1))]
        d = m - p
        ax.annotate("", xy=p + 0.35 * d, xytext=p, arrowprops=dict(arrowstyle="->", color=GOLD, lw=1))
    ax.scatter(modes[:, 0], modes[:, 1], c=ROSE, s=120, marker="*", zorder=5, label="modes")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_spade_lattice(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # lattice of sequential patterns
    nodes = {
        "<>": (5, 5.2),
        "<a>": (2.5, 3.6),
        "<b>": (5, 3.6),
        "<c>": (7.5, 3.6),
        "<ab>": (3.5, 1.6),
        "<bc>": (6.5, 1.6),
    }
    edges = [("<>", "<a>"), ("<>", "<b>"), ("<>", "<c>"), ("<a>", "<ab>"), ("<b>", "<ab>"), ("<b>", "<bc>"), ("<c>", "<bc>")]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=SLATE, lw=1.2)
    for name, (x, y) in nodes.items():
        box(ax, x - 0.9, y - 0.4, 1.8, 0.8, name, fc=TEAL if name != "<>" else GOLD, tc=("white" if name != "<>" else INK), fs=10)
    style(ax, t)


def d_binning_strategies(ax, t, rng):
    x = rng.normal(0, 1, 200)
    ax.hist(x, bins=8, color=TEAL, alpha=0.45, edgecolor=DEEP, label="equal-width")
    # equal-freq edges
    q = np.quantile(x, np.linspace(0, 1, 9))
    ax.hist(x, bins=q, color=GOLD, alpha=0.35, edgecolor=GOLD, label="equal-freq")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_ica_sources(ax, t, rng):
    """ICA: mixed signals → recovered independent sources."""
    t_ = np.linspace(0, 8, 300)
    s1 = np.sin(1.2 * t_)
    s2 = np.sign(np.sin(0.6 * t_))
    mix1 = 0.7 * s1 + 0.5 * s2
    mix2 = 0.3 * s1 - 0.8 * s2
    ax.plot(t_, mix1 + 3, color=SLATE, lw=1.2, label="mix")
    ax.plot(t_, mix2 + 1.5, color=SLATE, lw=1.2)
    ax.plot(t_, s1 - 0.5, color=TEAL, lw=1.5, label="source")
    ax.plot(t_, s2 - 2.5, color=GOLD, lw=1.5)
    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.legend(fontsize=8, loc="upper right")
    style(ax, t)


def d_qr_decomposition(ax, t, rng):
    """QR: columns of Q orthonormal, R upper triangular."""
    # show R structure
    R = np.triu(rng.uniform(0.5, 2, (5, 5)))
    ax.imshow(R, cmap="Greens")
    for i in range(5):
        for j in range(5):
            if j >= i:
                ax.text(j, i, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(2, 5.3, "R upper-tri  ·  QᵀQ = I (not shown)", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_platt_scaling(ax, t, rng):
    """Platt scaling: sigmoid on SVM scores."""
    s = np.linspace(-4, 4, 100)
    a, b = -1.2, 0.1
    p = 1 / (1 + np.exp(a * s + b))
    ax.plot(s, p, color=TEAL, lw=2.2, label="Platt σ(as+b)")
    raw = 1 / (1 + np.exp(-s))
    ax.plot(s, raw, color=SLATE, ls="--", lw=1.5, label="raw logistic")
    ax.set_xlabel("score s")
    ax.set_ylabel("P(y=1|s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_soap_optimizer(ax, t, rng):
    """SOAP / Shampoo-like preconditioner eigenvalue spectrum."""
    eig = np.sort(rng.lognormal(0, 1, 20))[::-1]
    ax.semilogy(eig, "o-", color=TEAL, lw=2)
    ax.set_xlabel("eigen index")
    ax.set_ylabel("precond. eigenvalue")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_data2vec(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.2, 3.0, 1.8, "masked\nstudent", fc=TEAL, fs=12)
    box(ax, 4.0, 2.2, 3.5, 1.8, "EMA teacher\nfull context", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.0, 2.2, 3.5, 1.8, "regress\nlatent targets", fc=DEEP, fs=11)
    for x0, x1 in [(3.4, 4.0), (7.5, 8.0)]:
        ax.annotate("", xy=(x1, 3.1), xytext=(x0, 3.1), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_encodec_rvq(ax, t, rng):
    """Residual vector quantization stack."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 4.0, 3, 1.4, "encoder z", fc=TEAL, fs=12)
    for i in range(4):
        box(ax, 4.0, 4.5 - i * 1.1, 3.5, 0.9, f"RVQ codebook {i+1}", fc=DEEP if i % 2 else GOLD, tc=("white" if i % 2 else INK), fs=10)
        if i == 0:
            ax.annotate("", xy=(4.0, 4.95), xytext=(3.5, 4.7), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.3))
        else:
            ax.annotate(
                "",
                xy=(5.75, 4.5 - i * 1.1 + 0.9),
                xytext=(5.75, 4.5 - (i - 1) * 1.1),
                arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.2),
            )
    box(ax, 8.5, 2.5, 3, 1.5, "residual\n↓", fc=MINT, fs=12)
    style(ax, t)


def d_dreamerv3(ax, t, rng):
    """World-model imagination returns."""
    steps = np.arange(0, 40)
    real = np.cumsum(rng.normal(0.05, 0.2, 40))
    imag = real + np.cumsum(rng.normal(0, 0.08, 40))
    ax.plot(steps, real, color=GOLD, lw=2, label="real return")
    ax.plot(steps, imag, color=TEAL, lw=2, label="imagined")
    ax.fill_between(steps, real, imag, color=TEAL, alpha=0.15)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_qlora_nf4(ax, t, rng):
    """NF4 quantile bins for QLoRA base weights."""
    # normal quantiles for 16 bins
    qs = np.linspace(0.01, 0.99, 16)
    edges = np.sqrt(2) * erfinv_approx(2 * qs - 1)
    ax.hist(rng.normal(0, 1, 5000), bins=edges, color=TEAL, edgecolor=DEEP, alpha=0.85)
    ax.set_xlabel("weight value")
    ax.set_ylabel("count in NF4 bins")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_pagerank_teleport(ax, t, rng):
    """PageRank: walk + teleport mixture."""
    damp = np.linspace(0.5, 0.99, 40)
    # entropy of stationary-ish as d increases (sketch)
    entropy = 1.5 * (1 - damp) + 0.3
    ax.plot(damp, entropy, color=TEAL, lw=2.2)
    ax.axvline(0.85, color=GOLD, ls="--", label="d≈0.85")
    ax.set_xlabel("damping d")
    ax.set_ylabel("stationary entropy sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_lineage_graph(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.4, 3.5, 2.5, 1.5, "raw", fc=SLATE, fs=12)
    box(ax, 3.5, 4.2, 2.5, 1.3, "clean", fc=TEAL, fs=12)
    box(ax, 3.5, 2.0, 2.5, 1.3, "feats", fc=DEEP, fs=12)
    box(ax, 7.0, 3.5, 2.5, 1.5, "train\nset", fc=GOLD, tc=INK, fs=12)
    box(ax, 10.0, 3.5, 1.8, 1.5, "model", fc=ROSE, fs=11)
    for a, b in [((2.9, 4.2), (3.5, 4.85)), ((2.9, 4.0), (3.5, 2.65)), ((6.0, 4.85), (7.0, 4.4)), ((6.0, 2.65), (7.0, 4.0)), ((9.5, 4.25), (10.0, 4.25))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, t)


def d_runbook_ladder(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for i, lab in enumerate(["alert", "triage", "mitigate", "fix", "document"]):
        box(ax, 0.5 + i * 0.3, 4.5 - i * 0.75, 8.5 - i * 0.4, 0.85, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_gloss_ssl_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["SimCLR", "BYOL", "DINO", "MAE", "JEPA"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C223 = [
    ("Givens plane rotation zeroing", d_givens_rotation),
    ("Model card section map", d_model_card_sections),
    ("Uniform convergence deviation", d_uniform_convergence),
    ("Streamgraph theme-river flows", d_streamgraph),
    ("Variational free energy terms", d_variational_free_energy),
    ("OPTICS reachability valleys", d_optics_reachability),
    ("PrefixSpan projection grow", d_prefixspan_project),
    ("WoE and IV bin chart", d_woe_iv_bins),
    ("Sparse PCA exact-zero loadings", d_sparse_pca),
    ("GAM smooth partial effect", d_gam_partial),
    ("Cost-sensitive confusion costs", d_cost_sensitive),
    ("SAM epsilon-ball sharpness", d_sharpness_aware),
    ("Barlow Twins cross-corr target", d_barlow_twins),
    ("Flamingo perceiver resampler", d_flamingo_resampler),
    ("Successor features discount sum", d_successor_features),
    ("SmoothQuant activation migrate", d_smoothquant),
    ("Motif z-score significance", d_motif_zscore),
    ("PII redaction pipeline stages", d_pii_redaction),
    ("SRE error budget burn", d_sre_error_budget),
    ("Glossary clustering family strip", d_gloss_cluster_strip),
]

C224 = [
    ("Lanczos tridiagonal projection", d_lanczos_tridiag),
    ("Red-team probe patch loop", d_red_team_loop),
    ("Geometric margin support vectors", d_margin_theory),
    ("Normal Q-Q diagnostic plot", d_qq_plot),
    ("Slice sampling under density", d_slice_sampling),
    ("Mean-shift mode seeking field", d_mean_shift),
    ("SPADE sequence pattern lattice", d_spade_lattice),
    ("Equal-width vs equal-freq bins", d_binning_strategies),
    ("ICA mix to independent sources", d_ica_sources),
    ("QR upper-triangular R factor", d_qr_decomposition),
    ("Platt scaling score map", d_platt_scaling),
    ("SOAP preconditioner spectrum", d_soap_optimizer),
    ("data2vec EMA latent targets", d_data2vec),
    ("EnCodec residual VQ stack", d_encodec_rvq),
    ("DreamerV3 imagination returns", d_dreamerv3),
    ("QLoRA NF4 quantile bins", d_qlora_nf4),
    ("PageRank damping teleport", d_pagerank_teleport),
    ("Data lineage DAG stages", d_lineage_graph),
    ("Incident runbook ladder", d_runbook_ladder),
    ("Glossary self-supervised strip", d_gloss_ssl_strip),
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

    m = {223: C223, 224: C224}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [223, 224]
    for c in cycles:
        embed(c, m[c])
