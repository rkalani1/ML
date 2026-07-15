#!/usr/bin/env python3
"""Cycle-209/210 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch, Ellipse, Arc, Polygon

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
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


# ----- C209 -----


def d_eigengap(ax, t, rng):
    lam = np.array([4.2, 3.1, 2.8, 0.9, 0.7, 0.5, 0.35, 0.2])
    ax.stem(np.arange(1, 9), lam, linefmt=TEAL, markerfmt="o", basefmt=" ")
    ax.plot([3.5, 3.5], [0, 3], color=GOLD, ls="--", lw=2)
    ax.annotate("eigengap", xy=(3.5, 1.8), xytext=(5.2, 3.5),
                arrowprops=dict(arrowstyle="->", color=GOLD), color=GOLD, fontsize=10, fontweight="bold")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dpa_budget(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    eps = [0.5, 1.0, 2.0, 4.0]
    for i, e in enumerate(eps):
        h = 0.6 + e * 0.5
        box(ax, 0.5 + i * 2.9, 1.2, 2.5, h, f"ε={e}", fc=TEAL if e <= 2 else ROSE, fs=12)
    ax.text(6, 4.3, "privacy budget spend (composition sketch)", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_no_free_lunch(ax, t, rng):
    # average risk equal for algorithms over all targets
    algs = ["A1", "A2", "A3", "A4"]
    # per-target risk varies but means equal
    data = np.array([[0.2, 0.8, 0.5, 0.4], [0.7, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.4], [0.55, 0.35, 0.45, 0.55]])
    x = np.arange(4)
    w = 0.18
    for i, a in enumerate(algs):
        ax.bar(x + (i - 1.5) * w, data[i], w, label=a, color=[TEAL, DEEP, GOLD, ROSE][i], edgecolor=DEEP)
    ax.axhline(0.475, color=SLATE, ls="--", lw=1.5, label="uniform avg risk")
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{j}" for j in range(4)])
    ax.set_ylabel("risk on target")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_hexbin_density(ax, t, rng):
    x = np.concatenate([rng.normal(-0.5, 0.6, 400), rng.normal(1.2, 0.5, 300)])
    y = np.concatenate([rng.normal(0.2, 0.7, 400), rng.normal(-0.8, 0.4, 300)])
    hb = ax.hexbin(x, y, gridsize=22, cmap="YlGn", mincnt=1)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="count")
    style(ax, t)


def d_pit_histogram(ax, t, rng):
    # probability integral transform under calibration
    u_cal = rng.uniform(0, 1, 800)
    u_mis = np.clip(rng.beta(2.5, 1.2, 800), 0, 1)
    ax.hist(u_cal, bins=12, density=True, alpha=0.55, color=TEAL, edgecolor=DEEP, label="calibrated PIT")
    ax.hist(u_mis, bins=12, density=True, alpha=0.45, color=ROSE, edgecolor=DEEP, label="miscalibrated")
    ax.axhline(1.0, color=GOLD, ls="--", lw=1.5, label="Uniform(0,1)")
    ax.set_xlabel("PIT value F(y)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_optics_reach(ax, t, rng):
    n = 80
    # synthetic reachability distances
    r = np.concatenate([
        np.linspace(0.8, 0.15, 20),
        np.linspace(0.2, 0.9, 8),
        np.linspace(0.85, 0.12, 25),
        np.linspace(0.15, 1.0, 10),
        np.linspace(0.9, 0.2, 17),
    ])
    ax.bar(np.arange(len(r)), r, color=TEAL, width=1.0, edgecolor="none")
    ax.axhline(0.35, color=GOLD, ls="--", lw=2, label="ξ steep cut")
    ax.set_xlabel("optics order")
    ax.set_ylabel("reachability distance")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_spade_lattice(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    nodes = {
        "<>": (5, 5.2),
        "<a>": (2, 3.5),
        "<b>": (5, 3.5),
        "<c>": (8, 3.5),
        "<ab>": (2, 1.5),
        "<ac>": (5, 1.5),
        "<bc>": (8, 1.5),
    }
    edges = [("<>", "<a>"), ("<>", "<b>"), ("<>", "<c>"), ("<a>", "<ab>"), ("<a>", "<ac>"), ("<b>", "<ab>"), ("<b>", "<bc>"), ("<c>", "<ac>"), ("<c>", "<bc>")]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=SLATE, lw=1.2)
    for lab, (x, y) in nodes.items():
        ax.plot(x, y, "o", color=TEAL, ms=16)
        ax.text(x, y, lab, ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.text(5, 0.4, "SPADE sequence lattice (id-list join)", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_catboost_ordered(ax, t, rng):
    idx = np.arange(1, 21)
    # ordered target stats converge
    true = 0.35
    running = true + 0.4 / np.sqrt(idx) * rng.normal(0, 1, 20).cumsum() / np.arange(1, 21)
    naive = np.full_like(idx, 0.55, dtype=float)
    ax.plot(idx, running, color=TEAL, lw=2, label="ordered TS")
    ax.axhline(true, color=GOLD, ls="--", label="true P(y|cat)")
    ax.axhline(0.55, color=ROSE, ls=":", label="full-fold mean (leak)")
    ax.set_xlabel("row order position")
    ax.set_ylabel("target statistic")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_isomap_geodesic(ax, t, rng):
    theta = np.linspace(0, 1.4 * np.pi, 80)
    r = 1 + 0.15 * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, color=SLATE, lw=1.5, alpha=0.5)
    ax.scatter(x[::3], y[::3], c=TEAL, s=28)
    # Euclidean chord vs geodesic arc
    i, j = 10, 55
    ax.plot([x[i], x[j]], [y[i], y[j]], "--", color=ROSE, lw=2, label="Euclidean chord")
    ax.plot(x[i:j], y[i:j], color=GOLD, lw=2.5, label="graph geodesic")
    ax.plot(x[i], y[i], "o", color=DEEP, ms=10)
    ax.plot(x[j], y[j], "o", color=DEEP, ms=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gam_partial(ax, t, rng):
    x = np.linspace(0, 10, 200)
    f = 0.4 * np.sin(x) + 0.05 * x
    se = 0.12 + 0.02 * np.sin(2 * x) ** 2
    ax.fill_between(x, f - 1.96 * se, f + 1.96 * se, color=TEAL, alpha=0.25)
    ax.plot(x, f, color=DEEP, lw=2.2, label="smooth f_j(x_j)")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xlabel("feature x_j")
    ax.set_ylabel("partial effect")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_pr_gain(ax, t, rng):
    rec = np.linspace(0, 1, 100)
    prev = 0.1
    prec_base = prev * np.ones_like(rec)
    prec_model = prev + (1 - prev) * (1 - rec) ** 0.7 * 0.55
    ax.plot(rec, prec_model, color=TEAL, lw=2.2, label="model PR")
    ax.plot(rec, prec_base, color=GOLD, ls="--", lw=1.8, label=f"baseline π={prev}")
    ax.fill_between(rec, prec_base, prec_model, where=prec_model >= prec_base, color=TEAL, alpha=0.2, label="PR gain area")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_rmsprop(ax, t, rng):
    steps = np.arange(0, 80)
    g2 = np.zeros(80)
    eg2 = 0
    for i in range(80):
        g = 1.2 * np.sin(i / 5) + rng.normal(0, 0.3)
        eg2 = 0.9 * eg2 + 0.1 * g * g
        g2[i] = eg2
    ax.plot(steps, g2, color=TEAL, lw=2, label="E[g²] (RMSProp cache)")
    ax.plot(steps, 0.01 / (np.sqrt(g2) + 1e-8), color=GOLD, lw=2, label="effective step scale")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_vicreg(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.5, 3.2, 1.8, "invariance\nMSE(z,z')", fc=TEAL, fs=11)
    box(ax, 4.4, 2.5, 3.2, 1.8, "variance\nhinge std", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.4, 2.5, 3.2, 1.8, "covariance\noff-diag→0", fc=DEEP, fs=11)
    ax.text(6, 1.2, "VICReg: no negatives, explicit var/cov terms", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_rope_rotary(ax, t, rng):
    # 2D rotation of query/key pairs by position
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for i, th in enumerate(theta):
        c, s = np.cos(th), np.sin(th)
        ax.annotate("", xy=(c, s), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=TEAL if i % 2 == 0 else GOLD, lw=1.5))
    ax.add_patch(Circle((0, 0), 1, fill=False, ls="--", edgecolor=SLATE))
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.text(0, -1.25, "RoPE: position-dependent planar rotations", ha="center", fontsize=9, color=INK)
    ax.grid(True, alpha=0.2)
    style(ax, t)


def d_priority_replay(ax, t, rng):
    n = 12
    td = np.abs(rng.normal(0, 1, n)) + 0.05
    alpha = 0.6
    p = td ** alpha
    p = p / p.sum()
    ax.bar(np.arange(n), p, color=TEAL, edgecolor=DEEP)
    ax2 = ax.twinx()
    ax2.plot(np.arange(n), td, "o-", color=GOLD, lw=1.5, label="|δ|")
    ax.set_xlabel("transition buffer index")
    ax.set_ylabel("sample probability")
    ax2.set_ylabel("|TD error|")
    ax2.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_sparse_gpt(ax, t, rng):
    W = rng.normal(0, 1, (10, 14))
    # mask smallest magnitude
    thr = np.quantile(np.abs(W), 0.5)
    mask = np.abs(W) >= thr
    show = np.where(mask, W, np.nan)
    ax.imshow(np.nan_to_num(show, nan=0), cmap="coolwarm", aspect="auto")
    for i in range(10):
        for j in range(14):
            if not mask[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#e2e8f0", edgecolor="none"))
    ax.set_xlabel("columns")
    ax.set_ylabel("rows")
    ax.text(0.5, 1.06, "SparseGPT: magnitude + Hessian-aware prune", transform=ax.transAxes, ha="center", fontsize=9, color=INK)
    style(ax, t)


def d_katz_centrality(ax, t, rng):
    # path-attenuated scores
    nodes = ["u", "v", "w", "x", "y"]
    score = np.array([1.8, 3.2, 1.1, 2.4, 0.9])
    ax.barh(nodes, score, color=TEAL, edgecolor=DEEP)
    ax.set_xlabel("Katz score (α-attenuated paths)")
    ax.axvline(score.mean(), color=GOLD, ls="--", label="mean")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_dataset_shift_types(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, (lab, c) in enumerate([
        ("covariate\nP(x)", TEAL),
        ("label\nP(y)", GOLD),
        ("concept\nP(y|x)", ROSE),
        ("prior\nπ(y)", DEEP),
    ]):
        tc = INK if c == GOLD else "white"
        box(ax, 0.5 + i * 2.9, 1.5, 2.7, 2.0, lab, fc=c, fs=11, tc=tc)
    ax.text(6, 4.3, "shift taxonomy (diagnose before 'fix')", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_change_failure(ax, t, rng):
    weeks = np.arange(1, 13)
    deploys = np.array([4, 5, 3, 6, 8, 7, 5, 9, 6, 4, 5, 7])
    fails = np.array([0, 1, 0, 1, 2, 1, 0, 3, 1, 0, 1, 1])
    rate = fails / deploys
    ax.bar(weeks, deploys, color=TEAL, alpha=0.5, label="deploys")
    ax.plot(weeks, fails, "o-", color=ROSE, lw=2, label="failed changes")
    ax2 = ax.twinx()
    ax2.plot(weeks, rate, "s--", color=GOLD, lw=1.5, label="change fail rate")
    ax.set_xlabel("week")
    ax.set_ylabel("count")
    ax2.set_ylabel("failure rate")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_glossary_metrics(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["AUROC", "AUPRC", "F1", "NLL", "NB"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


# ----- C210 -----


def d_power_iteration(ax, t, rng):
    A = np.array([[2.0, 0.8], [0.8, 1.2]])
    v = np.array([1.0, 0.2])
    v = v / np.linalg.norm(v)
    path = [v.copy()]
    for _ in range(8):
        v = A @ v
        v = v / np.linalg.norm(v)
        path.append(v.copy())
    path = np.array(path)
    # unit circle
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=SLATE, ls="--", lw=1)
    ax.plot(path[:, 0], path[:, 1], "o-", color=TEAL, lw=2, ms=7)
    # true dominant eigenvector direction
    w, V = np.linalg.eig(A)
    i = int(np.argmax(w.real))
    e = V[:, i].real
    e = e / np.linalg.norm(e)
    ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_model_card_risks(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fields = ["intended\nuse", "out of\nscope", "metrics\n& slices", "ethical\nrisks", "contact"]
    for i, f in enumerate(fields):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 2.0, f, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_learning_curve_cv(ax, t, rng):
    n = np.array([50, 100, 200, 400, 800, 1600])
    train = 0.95 - 0.08 * np.exp(-n / 300)
    val = 0.72 + 0.15 * (1 - np.exp(-n / 500))
    ax.plot(n, train, "o-", color=GOLD, lw=2, label="train score")
    ax.plot(n, val, "s-", color=TEAL, lw=2, label="CV val score")
    ax.fill_between(n, val - 0.03, val + 0.03, color=TEAL, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_raincloud(ax, t, rng):
    groups = [rng.normal(0, 1, 40), rng.normal(0.8, 0.9, 40), rng.normal(-0.3, 1.2, 40)]
    for i, g in enumerate(groups):
        # jitter strip
        x = i + rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(x, g, c=TEAL if i % 2 == 0 else DEEP, s=16, alpha=0.65)
        # half violin-ish density
        ys = np.linspace(g.min() - 0.3, g.max() + 0.3, 80)
        # kde approx
        dens = np.zeros_like(ys)
        for v in g:
            dens += np.exp(-0.5 * ((ys - v) / 0.35) ** 2)
        dens = dens / dens.max() * 0.35
        ax.fill_betweenx(ys, i, i + dens, color=GOLD, alpha=0.45)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["site A", "site B", "site C"])
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_posterior_predictive(ax, t, rng):
    x = np.linspace(0, 10, 50)
    # posterior draws of lines
    for _ in range(40):
        a = rng.normal(0.5, 0.15)
        b = rng.normal(1.0, 0.2)
        ax.plot(x, a + b * 0.1 * x, color=TEAL, alpha=0.15, lw=1)
    y_mean = 0.5 + 0.1 * x
    ax.plot(x, y_mean, color=GOLD, lw=2.5, label="posterior mean")
    ax.scatter(rng.uniform(0, 10, 25), 0.5 + 0.1 * rng.uniform(0, 10, 25) + rng.normal(0, 0.3, 25),
               c=ROSE, s=22, zorder=5, label="data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_spectral_gap_cluster(ax, t, rng):
    # affinity block + second eigenvector
    n = 12
    A = np.eye(n) * 0.1
    A[:6, :6] += rng.uniform(0.5, 1, (6, 6))
    A[6:, 6:] += rng.uniform(0.5, 1, (6, 6))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    w, V = np.linalg.eigh(L)
    fiedler = V[:, 1]
    ax.scatter(np.arange(n), fiedler, c=[TEAL if i < 6 else GOLD for i in range(n)], s=60, edgecolors=DEEP)
    ax.axhline(0, color=SLATE, ls="--")
    ax.set_xlabel("node index")
    ax.set_ylabel("Fiedler vector entry")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gsp_candidates(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    levels = [("1-seq", TEAL), ("join→2", DEEP), ("join→3", GOLD), ("prune", ROSE)]
    for i, (lab, c) in enumerate(levels):
        tc = INK if c in (GOLD,) else "white"
        box(ax, 0.5 + i * 2.9, 1.6, 2.7, 1.8, lab, fc=c, fs=12, tc=tc)
    ax.text(6, 4.2, "GSP: level-wise sequence candidate generation", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_rfe_path(ax, t, rng):
    k = np.arange(20, 0, -1)
    score = 0.55 + 0.02 * np.log(k) + rng.normal(0, 0.01, len(k))
    score = np.maximum.accumulate(score[::-1])[::-1] * 0 + 0.5 + 0.015 * k - 0.0004 * k**2
    ax.plot(k, score, "o-", color=TEAL, lw=2)
    best = int(np.argmax(score))
    ax.axvline(k[best], color=GOLD, ls="--", label=f"select k={k[best]}")
    ax.set_xlabel("features retained")
    ax.set_ylabel("CV score")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_lle_weights(ax, t, rng):
    # center point with neighbors and reconstruction weights
    c = np.array([0.0, 0.0])
    neigh = np.array([[0.8, 0.2], [0.3, 0.9], [-0.7, 0.4], [-0.2, -0.8], [0.6, -0.5]])
    w = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    ax.plot(*c, "o", color=ROSE, ms=14, label="point i")
    for p, wi in zip(neigh, w):
        ax.plot(*p, "o", color=TEAL, ms=10)
        ax.annotate("", xy=p, xytext=c, arrowprops=dict(arrowstyle="->", color=GOLD, lw=1 + 3 * wi))
        ax.text(p[0], p[1] + 0.12, f"{wi:.2f}", ha="center", fontsize=8, color=INK)
    recon = (neigh * w[:, None]).sum(axis=0)
    ax.plot(*recon, "s", color=DEEP, ms=10, label="Σ w_j x_j")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_negative_binomial(ax, t, rng):
    k = np.arange(0, 30)
    # NB vs Poisson
    mu = 6
    pois = np.exp(-mu) * mu**k / np.array([np.math.factorial(int(i)) for i in k])
    # NB overdispersed approx via formula
    r = 3.0
    p = r / (r + mu)
    from math import comb
    nb = np.array([comb(int(i + r - 1), int(i)) * (p**r) * ((1 - p) ** i) for i in k])
    ax.plot(k, pois, "o-", color=GOLD, lw=1.8, label="Poisson μ=6")
    ax.plot(k, nb, "s-", color=TEAL, lw=1.8, label="NB overdispersed")
    ax.set_xlabel("count")
    ax.set_ylabel("P(Y=k)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_balanced_accuracy(ax, t, rng):
    prev = np.linspace(0.05, 0.5, 40)
    sens, spec = 0.85, 0.80
    acc = sens * prev + spec * (1 - prev)
    bacc = 0.5 * (sens + spec) * np.ones_like(prev)
    ax.plot(prev, acc, color=GOLD, lw=2, label="accuracy")
    ax.plot(prev, bacc, color=TEAL, lw=2, label="balanced accuracy")
    ax.set_xlabel("prevalence")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_weight_norm(ax, t, rng):
    # direction vs magnitude
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=SLATE, ls="--")
    v = np.array([0.6, 0.8])
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5))
    ax.text(0.65, 0.9, "g · v/||v||", color=TEAL, fontsize=11, fontweight="bold")
    ax.annotate("", xy=1.5 * v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    ax.text(1.1, 1.35, "scale g", color=GOLD, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dino_prototypes(ax, t, rng):
    # soft assignment to prototypes
    x = np.linspace(-3, 3, 200)
    p1 = np.exp(-0.5 * (x + 1.2) ** 2)
    p2 = np.exp(-0.5 * (x - 0.5) ** 2)
    p3 = np.exp(-0.5 * (x - 1.8) ** 2)
    s = p1 + p2 + p3
    ax.stackplot(x, p1 / s, p2 / s, p3 / s, colors=[TEAL, GOLD, DEEP], labels=["proto1", "proto2", "proto3"], alpha=0.85)
    ax.set_xlabel("embedding coord")
    ax.set_ylabel("soft prototype mass")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    style(ax, t)


def d_conformer_block(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    stages = ["FFN\n½", "MHSA", "Conv\nmodule", "FFN\n½", "Layernorm"]
    for i, s in enumerate(stages):
        box(ax, 0.4 + i * 2.3, 2.0, 2.1, 2.0, s, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
        if i < 4:
            ax.annotate("", xy=(2.4 + i * 2.3, 3.0), xytext=(2.3 + i * 2.3, 3.0),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(6, 5.2, "Conformer block (speech/seq)", ha="center", fontsize=11, color=INK)
    style(ax, t)


def d_rnd_bonus(ax, t, rng):
    steps = np.arange(0, 100)
    err = 2.0 * np.exp(-steps / 35) + 0.05 + 0.05 * rng.normal(size=100)
    err = np.clip(err, 0.02, None)
    ax.plot(steps, err, color=TEAL, lw=2, label="||f̂(s)−f(s)||²")
    ax.fill_between(steps, 0, err, color=TEAL, alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("RND exploration bonus")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_gptq_hessian(ax, t, rng):
    # layer-wise quant error vs bits with Hessian awareness
    bits = np.array([2, 3, 4, 8])
    naive = np.array([0.45, 0.22, 0.10, 0.02])
    gptq = np.array([0.28, 0.12, 0.05, 0.015])
    x = np.arange(len(bits))
    ax.bar(x - 0.15, naive, 0.3, color=ROSE, label="round-to-nearest")
    ax.bar(x + 0.15, gptq, 0.3, color=TEAL, label="GPTQ Hessian")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bits])
    ax.set_xlabel("bits")
    ax.set_ylabel("layer output MSE (sketch)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_hits_auth_hub(ax, t, rng):
    labels = ["p1", "p2", "p3", "p4", "p5"]
    auth = np.array([0.9, 0.4, 0.7, 0.3, 0.55])
    hub = np.array([0.3, 0.85, 0.4, 0.75, 0.5])
    x = np.arange(len(labels))
    ax.bar(x - 0.15, auth, 0.3, color=TEAL, label="authority")
    ax.bar(x + 0.15, hub, 0.3, color=GOLD, label="hub")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_leakage_features(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # timeline
    ax.annotate("", xy=(11, 2.5), xytext=(0.5, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.axvline(6, color=GOLD, ls="--", lw=2)
    ax.text(6, 4.3, "prediction time", ha="center", color=GOLD, fontsize=10, fontweight="bold")
    box(ax, 1.0, 1.0, 3.5, 1.2, "OK: labs before t", fc=TEAL, fs=10)
    box(ax, 7.0, 1.0, 4.0, 1.2, "LEAK: post-outcome codes", fc=ROSE, fs=10)
    style(ax, t)


def d_slo_error_budget(ax, t, rng):
    days = np.arange(1, 31)
    burn = np.cumsum(rng.choice([0, 0, 0, 0.5, 1.2, 0], size=30))
    budget = 10.0
    ax.plot(days, burn, color=TEAL, lw=2, label="error budget burn")
    ax.axhline(budget, color=ROSE, ls="--", lw=2, label="monthly budget")
    ax.fill_between(days, burn, budget, where=burn < budget, color=TEAL, alpha=0.12)
    ax.set_xlabel("day of month")
    ax.set_ylabel("cumulative SLO burn")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_glossary_ssl(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["InfoNCE", "BYOL", "MAE", "DINO", "JEPA"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C209 = [
    ("Eigenvalue gap spectral clustering", d_eigengap),
    ("Differential privacy budget composition", d_dpa_budget),
    ("No free lunch average risk", d_no_free_lunch),
    ("Hexbin joint density view", d_hexbin_density),
    ("PIT histogram calibration check", d_pit_histogram),
    ("OPTICS reachability valleys", d_optics_reach),
    ("SPADE sequence lattice join", d_spade_lattice),
    ("Ordered target statistics encoding", d_catboost_ordered),
    ("Isomap geodesic vs Euclidean", d_isomap_geodesic),
    ("GAM partial smooth effect", d_gam_partial),
    ("Precision-recall gain over baseline", d_pr_gain),
    ("RMSProp second-moment cache", d_rmsprop),
    ("VICReg invariance variance covariance", d_vicreg),
    ("RoPE rotary position planes", d_rope_rotary),
    ("Prioritized experience replay", d_priority_replay),
    ("SparseGPT weight mask pattern", d_sparse_gpt),
    ("Katz path-attenuated centrality", d_katz_centrality),
    ("Dataset shift type taxonomy", d_dataset_shift_types),
    ("Change failure rate monitor", d_change_failure),
    ("Glossary discrimination metric strip", d_glossary_metrics),
]

C210 = [
    ("Power iteration dominant eigenvector", d_power_iteration),
    ("Model card risk field tiles", d_model_card_risks),
    ("Learning curve with CV band", d_learning_curve_cv),
    ("Raincloud distribution comparison", d_raincloud),
    ("Posterior predictive line bundle", d_posterior_predictive),
    ("Spectral Fiedler cluster split", d_spectral_gap_cluster),
    ("GSP level-wise sequence candidates", d_gsp_candidates),
    ("Recursive feature elimination path", d_rfe_path),
    ("LLE neighbor reconstruction weights", d_lle_weights),
    ("Negative binomial overdispersion", d_negative_binomial),
    ("Balanced accuracy vs prevalence", d_balanced_accuracy),
    ("Weight normalization direction scale", d_weight_norm),
    ("Prototype soft assignment mass", d_dino_prototypes),
    ("Conformer block stage stack", d_conformer_block),
    ("RND exploration bonus decay", d_rnd_bonus),
    ("GPTQ Hessian-aware quant error", d_gptq_hessian),
    ("HITS authority and hub scores", d_hits_auth_hub),
    ("Feature timing leakage fence", d_leakage_features),
    ("SLO error budget burn chart", d_slo_error_budget),
    ("Glossary self-supervised name strip", d_glossary_ssl),
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

    m = {209: C209, 210: C210}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [209, 210]
    for c in cycles:
        embed(c, m[c])
