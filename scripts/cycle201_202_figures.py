#!/usr/bin/env python3
"""Cycle-201/202 quality densify: novel scientific teal teaching panels.

Prefer QUALITY over volume: each panel uses custom geometry for a distinct
teaching point (not title-only reuse of prior kind keys).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.collections import LineCollection

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
TEAL, DEEP, INK, GOLD, SLATE, ROSE, MINT, SOFT = (
    "#0d9488",
    "#0f766e",
    "#0f172a",
    "#c9a227",
    "#64748b",
    "#e11d48",
    "#14b8a6",
    "#ecfeff",
)
CHS = sorted(p.name for p in CURR.glob("*.md"))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------------------
# Custom scientific draw functions (cycle 201)
# ---------------------------------------------------------------------------

def draw_condnum(ax, title, rng):
    """Condition number: small δA amplifies solution error."""
    t = np.linspace(0, 2 * np.pi, 400)
    # elongated unit ball under ill-conditioned map
    ax.plot(np.cos(t), 0.22 * np.sin(t), color=TEAL, lw=2.2, label="A·unit ball")
    ax.plot(0.55 * np.cos(t), 0.55 * np.sin(t), color=SLATE, lw=1.2, ls="--", label="well-cond. ball")
    ax.annotate("", xy=(1.0, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    ax.annotate("", xy=(0.0, 0.22), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    ax.text(1.05, 0.08, "σ_max", color=GOLD, fontsize=9)
    ax.text(0.08, 0.28, "σ_min", color=ROSE, fontsize=9)
    ax.text(0, -0.55, r"κ₂(A)=σ_max/σ_min  amplifies residual → solution error", ha="center", color=INK, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.75, 0.75)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_intended_use(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    items = [
        (0.3, 2.8, "population", TEAL),
        (3.2, 2.8, "setting", DEEP),
        (6.1, 2.8, "decision", GOLD),
        (9.0, 2.8, "action", ROSE),
        (1.5, 0.7, "out-of-scope\nexclusions", SLATE),
        (6.5, 0.7, "required\ninputs", MINT),
    ]
    for x, y, t, c in items:
        tc = INK if c in (GOLD, MINT) else "white"
        box(ax, x, y, 2.6, 1.5, t, fc=c, fs=9, tc=tc)
    ax.annotate("", xy=(4.5, 3.5), xytext=(2.9, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(7.4, 3.5), xytext=(5.8, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(10.3, 3.5), xytext=(8.7, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, title)


def draw_no_free_lunch(ax, title, rng):
    """Uniform average over all labelings: all learners equal."""
    x = np.arange(6)
    # over all f, risk equal; on a slice of natural f, structure wins
    avg = np.full(6, 0.5)
    natural = np.array([0.42, 0.35, 0.28, 0.33, 0.38, 0.45])
    w = 0.35
    ax.bar(x - w / 2, avg, width=w, color=SLATE, edgecolor=DEEP, label="avg over all f")
    ax.bar(x + w / 2, natural, width=w, color=TEAL, edgecolor=DEEP, label="structured f slice")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h{i}" for i in range(6)], fontsize=8)
    ax.set_ylabel("risk")
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    style(ax, title)


def draw_small_multiples(ax, title, rng):
    """Spaghetti overlay vs small-multiples readability."""
    t = np.linspace(0, 4 * np.pi, 120)
    colors = [TEAL, DEEP, GOLD, ROSE, SLATE, MINT]
    # left: spaghetti
    for i, c in enumerate(colors):
        ax.plot(t, np.sin(t + i * 0.4) + i * 0.05, color=c, alpha=0.55, lw=1.4)
    ax.axvline(6.5, color=INK, ls=":", lw=1)
    ax.text(3.0, 1.35, "spaghetti\noverlay", ha="center", fontsize=9, color=INK)
    # right: small multiples as offset windows
    for i, c in enumerate(colors[:3]):
        yy = -0.2 - i * 0.55 + 0.35 * np.sin(t[:40] + i)
        ax.plot(7.2 + t[:40] * 0.35, yy, color=c, lw=1.6)
        ax.text(11.3, yy.mean(), f"panel {i+1}", fontsize=7, color=c, va="center")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-2.0, 1.7)
    ax.set_xticks([])
    ax.set_yticks([])
    style(ax, title)


def draw_tv_distance(ax, title, rng):
    """Total variation: half L1 between densities."""
    x = np.linspace(-4, 4, 400)
    p = np.exp(-0.5 * (x + 0.8) ** 2) / np.sqrt(2 * np.pi)
    q = np.exp(-0.5 * (x - 0.9) ** 2) / np.sqrt(2 * np.pi) * 0.95
    ax.plot(x, p, color=TEAL, lw=2, label="p")
    ax.plot(x, q, color=GOLD, lw=2, label="q")
    ax.fill_between(x, p, q, where=(p > q), color=ROSE, alpha=0.35, label="½∫|p−q|")
    ax.fill_between(x, p, q, where=(q > p), color=ROSE, alpha=0.35)
    tv = 0.5 * np.trapezoid(np.abs(p - q), x)
    ax.text(0, max(p.max(), q.max()) * 0.92, f"TV(p,q)≈{tv:.2f}", ha="center", fontsize=10, color=INK)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_silhouette(ax, title, rng):
    """Silhouette: a vs b neighborhood cohesion/separation."""
    c1 = rng.normal([-1.2, 0], 0.28, (35, 2))
    c2 = rng.normal([1.3, 0.1], 0.32, (35, 2))
    ax.scatter(c1[:, 0], c1[:, 1], c=TEAL, s=22, alpha=0.85, label="cluster A")
    ax.scatter(c2[:, 0], c2[:, 1], c=GOLD, s=22, alpha=0.85, label="cluster B")
    pt = c1[0]
    ax.scatter([pt[0]], [pt[1]], c=ROSE, s=90, zorder=5, marker="*")
    ax.add_patch(Circle(pt, 0.55, fill=False, ls="--", edgecolor=TEAL, lw=1.5))
    ax.annotate("", xy=(c2.mean(0)), xytext=pt, arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.5))
    ax.text(pt[0] - 0.1, pt[1] + 0.7, "a: mean intra", color=TEAL, fontsize=8)
    ax.text(0.2, 0.7, "b: nearest other", color=ROSE, fontsize=8)
    ax.text(0, -1.15, r"s=(b−a)/max(a,b)  ∈ [−1,1]", ha="center", fontsize=9, color=INK)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_lift_surface(ax, title, rng):
    """Support / confidence / lift relationship (discrete bars)."""
    rules = ["A→B", "C→D", "E→F", "G→H"]
    support = np.array([0.18, 0.12, 0.08, 0.22])
    conf = np.array([0.72, 0.55, 0.91, 0.40])
    p_b = np.array([0.45, 0.50, 0.35, 0.55])
    lift = conf / p_b
    x = np.arange(len(rules))
    ax.bar(x - 0.2, support, 0.2, color=SLATE, label="support")
    ax.bar(x, conf, 0.2, color=TEAL, label="confidence")
    ax.bar(x + 0.2, lift / lift.max(), 0.2, color=GOLD, label="lift (scaled)")
    for i, L in enumerate(lift):
        ax.text(i + 0.2, (lift / lift.max())[i] + 0.03, f"{L:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(1.0 / lift.max(), color=ROSE, ls="--", lw=1, label="lift=1 baseline (scaled)")
    ax.set_xticks(x)
    ax.set_xticklabels(rules)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, title)


def draw_temporal_leakage(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # timeline
    ax.plot([0.5, 11.5], [2.5, 2.5], color=SLATE, lw=2)
    for x, lab, c in [
        (2.0, "index\nevent", TEAL),
        (5.5, "label\nwindow", GOLD),
        (9.0, "future\nlab", ROSE),
    ]:
        ax.plot(x, 2.5, "o", color=c, ms=12)
        ax.text(x, 1.5, lab, ha="center", fontsize=9, color=c, fontweight="bold")
    box(ax, 7.2, 3.3, 4.2, 1.2, "FORBIDDEN join:\nfeature uses future lab", fc=ROSE, fs=9)
    box(ax, 0.5, 3.3, 4.2, 1.2, "OK: features ≤ index", fc=TEAL, fs=9)
    ax.annotate("", xy=(9.0, 3.3), xytext=(9.0, 2.6), arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.3))
    style(ax, title)


def draw_parallel_analysis(ax, title, rng):
    """Scree vs parallel analysis null eigenvalues."""
    k = np.arange(1, 13)
    real = 6.5 * np.exp(-0.45 * (k - 1)) + 0.35 + rng.normal(0, 0.04, len(k))
    null = 1.1 * np.exp(-0.12 * (k - 1)) + 0.55
    ax.plot(k, real, "o-", color=TEAL, lw=2, label="data eigenvalues")
    ax.plot(k, null, "s--", color=GOLD, lw=2, label="parallel-analysis null")
    keep = real > null
    ax.fill_between(k, real, null, where=keep, color=MINT, alpha=0.25, label="retain components")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_cooks_distance(ax, title, rng):
    """Leverage vs residual; Cook's distance contours."""
    n = 80
    lev = rng.beta(2, 8, n) * 0.45
    res = rng.normal(0, 1, n)
    # inject influential points
    lev = np.append(lev, [0.38, 0.42])
    res = np.append(res, [2.8, -2.6])
    cook = (res ** 2) * lev / (1 - lev + 1e-6)
    sc = ax.scatter(lev, res, c=cook, cmap="YlOrRd", s=28, edgecolors=DEEP, linewidths=0.3)
    ax.axhline(0, color=SLATE, lw=1)
    ax.axvline(0.2, color=TEAL, ls="--", lw=1, label="high leverage")
    # rough Cook contour
    L = np.linspace(0.01, 0.45, 80)
    for thr, ls in [(0.5, ":"), (1.0, "--")]:
        R = np.sqrt(thr * (1 - L) / (L + 1e-9))
        ax.plot(L, R, color=ROSE, ls=ls, lw=1.2)
        ax.plot(L, -R, color=ROSE, ls=ls, lw=1.2)
    ax.set_xlabel("leverage h_ii")
    ax.set_ylabel("studentized residual")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Cook-like score", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_pr_prevalence(ax, title, rng):
    """Same ranking, different prevalence → PR changes, ROC stable idea."""
    scores = np.linspace(0, 1, 200)
    # synthetic ROC-like curve shared
    fpr = scores
    tpr = scores ** 0.55
    ax.plot(fpr, tpr, color=TEAL, lw=2.2, label="ROC (prevalence-robust sketch)")
    # PR curves shift with prevalence
    for pi, c, lab in [(0.5, GOLD, "π=0.50 PR"), (0.1, ROSE, "π=0.10 PR"), (0.02, SLATE, "π=0.02 PR")]:
        # crude PR from same scores assumption
        prec = (tpr * pi) / (tpr * pi + fpr * (1 - pi) + 1e-9)
        rec = tpr
        ax.plot(rec, prec, color=c, lw=1.8, label=lab)
    ax.set_xlabel("FPR or Recall")
    ax.set_ylabel("TPR or Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_vanishing_grad(ax, title, rng):
    """Product of Jacobians → vanishing with depth."""
    depth = np.arange(1, 21)
    # |λ|<1 contracts
    for lam, c, lab in [(0.95, TEAL, "|λ|=0.95"), (0.8, GOLD, "|λ|=0.80"), (1.05, ROSE, "|λ|=1.05 explode")]:
        ax.semilogy(depth, lam ** depth, "o-", color=c, lw=2, ms=3.5, label=lab)
    ax.axhline(1e-3, color=SLATE, ls=":", label="numerical floor")
    ax.set_xlabel("depth L")
    ax.set_ylabel(r"‖∂z_L/∂z_0‖ ~ |λ|^L")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    style(ax, title)


def draw_hard_negatives(ax, title, rng):
    """Contrastive ball: easy vs hard negatives."""
    anchor = np.array([0.0, 0.0])
    pos = np.array([0.55, 0.2])
    easy = rng.normal([2.2, 1.5], 0.25, (12, 2))
    hard = rng.normal([0.9, -0.15], 0.18, (10, 2))
    ax.scatter(easy[:, 0], easy[:, 1], c=SLATE, s=28, label="easy negatives", alpha=0.8)
    ax.scatter(hard[:, 0], hard[:, 1], c=ROSE, s=36, label="hard negatives", alpha=0.9)
    ax.scatter([pos[0]], [pos[1]], c=GOLD, s=90, marker="*", label="positive", zorder=5)
    ax.scatter([0], [0], c=TEAL, s=100, marker="X", label="anchor", zorder=5)
    ax.add_patch(Circle((0, 0), 1.15, fill=False, ls="--", edgecolor=TEAL, lw=1.5))
    ax.text(0.2, 1.25, "hard-negative radius", color=TEAL, fontsize=8)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_ctc_path(ax, title, rng):
    """CTC alignment lattice: blank-allowed path."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.5)
    labels = ["-", "C", "A", "T", "-"]
    for j, lab in enumerate(labels):
        for i in range(10):
            fc = SOFT if lab == "-" else "#ccfbf1"
            ax.add_patch(Rectangle((i, j), 0.9, 0.9, facecolor=fc, edgecolor=DEEP, lw=0.6))
            if i == 0:
                ax.text(-0.35, j + 0.45, lab, ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
    # one forced path
    path = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3), (8, 4), (9, 4)]
    xs = [p[0] + 0.45 for p in path]
    ys = [p[1] + 0.45 for p in path]
    ax.plot(xs, ys, color=TEAL, lw=2.5, marker="o", ms=5)
    ax.set_xticks(np.arange(10) + 0.45)
    ax.set_xticklabels([f"t{i}" for i in range(10)], fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("time frames")
    ax.text(5, 5.15, "one blank-aware alignment path (many sum in CTC)", ha="center", fontsize=8, color=INK)
    style(ax, title)


def draw_exploration_bonus(ax, title, rng):
    """Extrinsic reward vs decaying exploration bonus."""
    t = np.arange(0, 100)
    ext = 0.3 + 0.05 * np.sin(t / 8)
    bonus = 1.2 * np.exp(-t / 25)
    ax.plot(t, ext, color=GOLD, lw=2, label="extrinsic r")
    ax.plot(t, bonus, color=TEAL, lw=2, label="exploration bonus")
    ax.plot(t, ext + bonus, color=DEEP, lw=2.2, label="total shaping")
    ax.set_xlabel("episode / step")
    ax.set_ylabel("signal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_weight_tying(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.8, 3.5, 1.5, "token embed\nE ∈ R^{V×d}", fc=TEAL, fs=10)
    box(ax, 8.0, 2.8, 3.5, 1.5, "softmax head\nW = Eᵀ (tied)", fc=GOLD, tc=INK, fs=10)
    box(ax, 4.2, 0.6, 3.6, 1.5, "shared params\n↓ memory", fc=DEEP, fs=10)
    ax.annotate("", xy=(8.0, 3.5), xytext=(4.0, 3.5), arrowprops=dict(arrowstyle="<->", color=ROSE, lw=2))
    ax.text(6.0, 3.9, "tie", ha="center", color=ROSE, fontsize=10, fontweight="bold")
    style(ax, title)


def draw_oversquash(ax, title, rng):
    """Graph bottleneck: many leaves → one bridge → oversquashing."""
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.2, 2.5)
    ax.axis("off")
    # left leaves
    leaves_l = [(-2.5, y) for y in np.linspace(-1.6, 1.6, 7)]
    leaves_r = [(2.5, y) for y in np.linspace(-1.6, 1.6, 7)]
    bridge = [(-0.6, 0.0), (0.6, 0.0)]
    for p in leaves_l:
        ax.plot([p[0], bridge[0][0]], [p[1], bridge[0][1]], color=SLATE, lw=0.9, alpha=0.7)
        ax.plot(*p, "o", color=TEAL, ms=7)
    for p in leaves_r:
        ax.plot([p[0], bridge[1][0]], [p[1], bridge[1][1]], color=SLATE, lw=0.9, alpha=0.7)
        ax.plot(*p, "o", color=GOLD, ms=7)
    ax.plot([bridge[0][0], bridge[1][0]], [0, 0], color=ROSE, lw=3.5)
    ax.plot(*bridge[0], "o", color=ROSE, ms=12)
    ax.plot(*bridge[1], "o", color=ROSE, ms=12)
    ax.text(0, 0.45, "bottleneck\nedge", ha="center", color=ROSE, fontsize=9, fontweight="bold")
    ax.text(0, -2.0, "messages compressed → oversquashing", ha="center", fontsize=9, color=INK)
    style(ax, title)


def draw_selection_bias(ax, title, rng):
    """Collider / selection: conditioning opens non-causal path."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    nodes = {
        "U": (5, 5.0),
        "X": (2, 2.8),
        "Y": (8, 2.8),
        "S": (5, 1.0),
    }
    for name, (x, y) in nodes.items():
        fc = ROSE if name == "S" else TEAL
        box(ax, x - 0.7, y - 0.45, 1.4, 0.9, name, fc=fc, fs=11)
    # U→X, U→Y, X→S, Y→S
    def arr(a, b, c=INK):
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops=dict(arrowstyle="->", color=c, lw=1.6),
        )

    arr((5, 4.55), (2.3, 3.25))
    arr((5, 4.55), (7.7, 3.25))
    arr((2.3, 2.35), (4.5, 1.45), ROSE)
    arr((7.7, 2.35), (5.5, 1.45), ROSE)
    ax.text(5, 0.25, "condition on S (selection) → spurious X—Y association", ha="center", fontsize=9, color=ROSE)
    style(ax, title)


def draw_psi_drift(ax, title, rng):
    """Population stability index style bin mass shift."""
    bins = np.arange(8)
    train = np.array([0.08, 0.12, 0.18, 0.22, 0.16, 0.12, 0.08, 0.04])
    serve = np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.18, 0.14, 0.10])
    train = train / train.sum()
    serve = serve / serve.sum()
    w = 0.35
    ax.bar(bins - w / 2, train, width=w, color=TEAL, label="train mass")
    ax.bar(bins + w / 2, serve, width=w, color=GOLD, label="serve mass")
    psi_i = (serve - train) * np.log((serve + 1e-9) / (train + 1e-9))
    psi = psi_i.sum()
    ax2 = ax.twinx()
    ax2.plot(bins, psi_i, "o-", color=ROSE, lw=1.8, label="bin PSI term")
    ax2.set_ylabel("bin contribution", color=ROSE)
    ax.set_xlabel("score / feature bin")
    ax.set_ylabel("probability mass")
    ax.set_title(title + f"  (PSI≈{psi:.3f})", fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")
    ax.grid(True, axis="y", alpha=0.25)


def draw_bias_var_strip(ax, title, rng):
    """Bias–variance–noise decomposition bars."""
    complexity = np.linspace(1, 10, 40)
    bias2 = 2.5 / complexity
    var = 0.04 * complexity ** 1.3
    noise = np.full_like(complexity, 0.35)
    ax.fill_between(complexity, 0, noise, color=SLATE, alpha=0.35, label="irreducible")
    ax.fill_between(complexity, noise, noise + bias2, color=TEAL, alpha=0.55, label="bias²")
    ax.fill_between(complexity, noise + bias2, noise + bias2 + var, color=GOLD, alpha=0.55, label="variance")
    total = noise + bias2 + var
    ax.plot(complexity, total, color=ROSE, lw=2.2, label="expected MSE")
    i = int(np.argmin(total))
    ax.axvline(complexity[i], color=DEEP, ls="--", lw=1.2)
    ax.text(complexity[i] + 0.15, total.max() * 0.9, "sweet spot", color=DEEP, fontsize=9)
    ax.set_xlabel("model complexity")
    ax.set_ylabel("error")
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.grid(True, alpha=0.25)
    style(ax, title)


# ---------------------------------------------------------------------------
# Cycle 202 custom panels
# ---------------------------------------------------------------------------

def draw_schatten(ax, title, rng):
    """Schatten-p norms on singular values."""
    s = np.array([4.0, 2.5, 1.2, 0.6, 0.3, 0.15])
    pvals = [1, 2, 4]
    norms = [(np.sum(s ** p)) ** (1 / p) for p in pvals]
    ax.bar([f"p={p}" for p in pvals], norms, color=[TEAL, GOLD, ROSE], edgecolor=DEEP)
    ax.plot(range(len(s)), s, "o-", color=DEEP, label="σ_i")
    ax.set_ylabel("Schatten norm / σ")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, title)


def draw_risk_tiers(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    tiers = [("minimal", TEAL), ("limited", MINT), ("high", GOLD), ("unacceptable", ROSE)]
    for i, (t, c) in enumerate(tiers):
        tc = INK if c in (GOLD, MINT) else "white"
        box(ax, 0.4 + i * 2.9, 1.2, 2.7, 1.6, t, fc=c, fs=11, tc=tc)
    ax.text(6, 3.3, "risk-tiered controls scale with harm", ha="center", fontsize=10, color=INK)
    style(ax, title)


def draw_excess_risk(ax, title, rng):
    n = np.linspace(20, 2000, 80)
    approx = np.full_like(n, 0.12)
    estim = 1.5 / np.sqrt(n)
    optim = 0.4 / n ** 0.8
    ax.stackplot(n, approx, estim, optim, colors=[SLATE, TEAL, GOLD], labels=["approx", "estimation", "optimization"], alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("excess risk parts")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_ecdf_bands(ax, title, rng):
    x = np.sort(rng.normal(0, 1, 80))
    ecdf = np.arange(1, len(x) + 1) / len(x)
    # DKW-style band width sketch
    eps = 0.12
    ax.step(x, ecdf, where="post", color=TEAL, lw=2, label="ECDF")
    ax.fill_between(x, np.clip(ecdf - eps, 0, 1), np.clip(ecdf + eps, 0, 1), color=TEAL, alpha=0.2, label="DKW band sketch")
    ax.set_xlabel("x")
    ax.set_ylabel("F̂_n")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_pit_hist(ax, title, rng):
    """Probability integral transform under calibration."""
    u_cal = rng.uniform(0, 1, 400)
    u_mis = np.clip(rng.beta(2.5, 1.2, 400), 0, 1)
    ax.hist(u_cal, bins=12, density=True, alpha=0.55, color=TEAL, edgecolor=DEEP, label="calibrated PIT")
    ax.hist(u_mis, bins=12, density=True, alpha=0.45, color=ROSE, edgecolor=DEEP, label="miscalibrated PIT")
    ax.axhline(1.0, color=GOLD, ls="--", lw=1.5, label="Uniform(0,1)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_dbcv(ax, title, rng):
    """Density-based cluster validity: core vs border."""
    core = rng.normal([0, 0], 0.25, (40, 2))
    border = rng.normal([0.9, 0.2], 0.45, (25, 2))
    noise = rng.uniform(-2, 2, (15, 2))
    ax.scatter(noise[:, 0], noise[:, 1], c=SLATE, s=18, alpha=0.5, label="noise")
    ax.scatter(border[:, 0], border[:, 1], c=GOLD, s=28, label="border")
    ax.scatter(core[:, 0], core[:, 1], c=TEAL, s=36, label="core")
    ax.add_patch(Circle((0, 0), 0.7, fill=False, ls="--", edgecolor=TEAL, lw=1.5))
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_prefixspan(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 3.2, 2.4, 1.2, "seq DB", fc=SLATE, fs=10)
    box(ax, 3.5, 3.2, 2.4, 1.2, "prefix α", fc=TEAL, fs=10)
    box(ax, 6.6, 3.2, 2.4, 1.2, "project", fc=DEEP, fs=10)
    box(ax, 9.2, 3.2, 2.4, 1.2, "extend", fc=GOLD, tc=INK, fs=10)
    for x in [2.8, 5.9, 9.0]:
        ax.annotate("", xy=(x + 0.5, 3.8), xytext=(x, 3.8), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    box(ax, 3.5, 0.8, 5.5, 1.4, "depth-first frequent sequential patterns", fc=MINT, fs=10, tc=INK)
    style(ax, title)


def draw_target_enc_cv(ax, title, rng):
    """Out-of-fold target encoding prevents leakage."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    folds = ["F1", "F2", "F3", "F4"]
    for i, f in enumerate(folds):
        box(ax, 0.5 + i * 2.3, 3.0, 2.1, 1.2, f, fc=TEAL if i != 2 else GOLD, fs=10, tc="white" if i != 2 else INK)
    ax.text(5, 2.3, "encode F3 using means from F1∪F2∪F4 only", ha="center", fontsize=10, color=INK)
    box(ax, 2.0, 0.6, 6.0, 1.2, "never use full-data category mean in-train", fc=ROSE, fs=10)
    style(ax, title)


def draw_isomap_geodesic(ax, title, rng):
    """Swiss-roll style: Euclidean chord vs geodesic on manifold."""
    t = np.linspace(0.5 * np.pi, 3.8 * np.pi, 200)
    x = t * np.cos(t)
    y = t * np.sin(t)
    ax.plot(x, y, color=TEAL, lw=2, label="manifold")
    # two points
    i, j = 30, 150
    ax.plot([x[i], x[j]], [y[i], y[j]], color=ROSE, ls="--", lw=1.5, label="Euclidean chord")
    ax.plot(x[i:j], y[i:j], color=GOLD, lw=2.5, label="geodesic path")
    ax.scatter([x[i], x[j]], [y[i], y[j]], c=DEEP, s=50, zorder=5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_partial_residual(ax, title, rng):
    x = np.linspace(0, 10, 80)
    true = 0.15 * x ** 2 - 0.8 * x
    y = true + rng.normal(0, 1.2, size=len(x))
    # linear fit residual-ish
    coef = np.polyfit(x, y, 1)
    linear = np.polyval(coef, x)
    partial = y - linear + np.mean(linear)
    ax.scatter(x, partial, c=TEAL, s=16, alpha=0.75, label="partial residual")
    ax.plot(x, true - np.polyval(np.polyfit(x, true, 1), x) + np.mean(true), color=GOLD, lw=2, label="true nonlinear")
    ax.set_xlabel("x_j")
    ax.set_ylabel("partial residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_cost_curve(ax, title, rng):
    """Cost curve: expected cost vs probability cost."""
    pc = np.linspace(0, 1, 100)
    # two classifiers fixed FP/FN rates sketch
    for fpr, fnr, c, lab in [
        (0.1, 0.2, TEAL, "model A"),
        (0.25, 0.08, GOLD, "model B"),
        (0.0, 1.0, SLATE, "always neg"),
        (1.0, 0.0, ROSE, "always pos"),
    ]:
        cost = fnr * pc + fpr * (1 - pc)
        ax.plot(pc, cost, color=c, lw=2, label=lab)
    ax.set_xlabel("probability cost p_c(+)")
    ax.set_ylabel("normalized expected cost")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_skip_connection(ax, title, rng):
    """Residual path preserves gradient highway."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 1.8, 2.5, 1.4, "x", fc=SLATE, fs=12)
    box(ax, 4.0, 1.8, 2.5, 1.4, "F(x)", fc=TEAL, fs=12)
    box(ax, 8.5, 1.8, 2.5, 1.4, "x+F(x)", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(4.0, 2.5), xytext=(3.0, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("", xy=(8.5, 2.5), xytext=(6.5, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    # skip
    ax.annotate(
        "",
        xy=(9.0, 3.6),
        xytext=(1.7, 3.6),
        arrowprops=dict(arrowstyle="->", color=ROSE, lw=2, connectionstyle="arc3,rad=-0.2"),
    )
    ax.text(5.2, 4.1, "identity skip (gradient highway)", ha="center", color=ROSE, fontsize=9, fontweight="bold")
    style(ax, title)


def draw_info_nce(ax, title, rng):
    """InfoNCE logit margins: positive vs negatives."""
    logits = np.array([3.2] + list(rng.normal(0.2, 0.6, 7)))
    labels = ["pos"] + [f"n{i}" for i in range(1, 8)]
    colors = [GOLD] + [TEAL] * 7
    ax.barh(range(len(logits)), logits, color=colors, edgecolor=DEEP)
    ax.set_yticks(range(len(logits)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color=SLATE, lw=1)
    ax.set_xlabel("similarity logit")
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    ax.text(0.98, 0.05, f"p(pos)≈{p[0]:.2f}", transform=ax.transAxes, ha="right", fontsize=9, color=INK)
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, title)


def draw_rope_angles(ax, title, rng):
    """RoPE: rotate query/key by position angle."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=SLATE, lw=1, alpha=0.5)
    for m, c in [(0, TEAL), (1, GOLD), (3, ROSE)]:
        ang = 0.35 * m
        ax.annotate(
            "",
            xy=(np.cos(ang), np.sin(ang)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=c, lw=2),
        )
        ax.text(1.15 * np.cos(ang), 1.15 * np.sin(ang), f"m={m}", color=c, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_gae_lambda(ax, title, rng):
    """λ-return interpolates TD and MC."""
    T = 12
    t = np.arange(T)
    mc = np.cumsum(rng.normal(0.3, 0.4, T)[::-1])[::-1]
    td = np.full(T, mc.mean())
    for lam, c in [(0.0, SLATE), (0.5, TEAL), (0.9, GOLD), (1.0, ROSE)]:
        # convex blend sketch
        g = (1 - lam) * td + lam * mc if lam < 1 else mc
        if lam == 0:
            g = td
        ax.plot(t, g, "-o", color=c, lw=1.8, ms=3.5, label=f"λ={lam}")
    ax.set_xlabel("time t")
    ax.set_ylabel("λ-return target")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_kv_cache(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.8, 3.0, 1.5, "past K,V\ncache", fc=TEAL, fs=11)
    box(ax, 4.2, 2.8, 3.0, 1.5, "new token\nq_t", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.2, 2.8, 3.4, 1.5, "attn once\nover cache", fc=DEEP, fs=11)
    ax.annotate("", xy=(4.2, 3.5), xytext=(3.4, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("", xy=(8.2, 3.5), xytext=(7.2, 3.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(6, 1.2, "decode O(n) per step with cache vs recompute all", ha="center", fontsize=10, color=INK)
    style(ax, title)


def draw_graph_rewire(ax, title, rng):
    """Local edges + long-range rewires for mixing."""
    n = 12
    pos = {i: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)}
    # ring
    for i in range(n):
        j = (i + 1) % n
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=TEAL, lw=1.5)
        ax.plot(*pos[i], "o", color=DEEP, ms=8)
    # rewires
    for a, b in [(0, 5), (2, 8), (3, 9)]:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=GOLD, lw=1.8, ls="--")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -1.45, "local ring + long-range rewires", ha="center", fontsize=9, color=INK)
    style(ax, title)


def draw_dataset_shift_types(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    labs = [
        (0.3, "covariate\nshift", TEAL),
        (3.2, "label\nshift", GOLD),
        (6.1, "concept\ndrift", ROSE),
        (9.0, "domain\ngap", DEEP),
    ]
    for x, t, c in labs:
        tc = INK if c == GOLD else "white"
        box(ax, x, 1.5, 2.7, 2.0, t, fc=c, fs=10, tc=tc)
    style(ax, title)


def draw_decision_impact(ax, title, rng):
    """Net benefit vs threshold probability."""
    pt = np.linspace(0.01, 0.8, 100)
    sens, spec, prev = 0.85, 0.75, 0.2
    # simplified decision curve style
    nb_model = sens * prev - (1 - spec) * (1 - prev) * (pt / (1 - pt))
    nb_all = prev - (1 - prev) * (pt / (1 - pt))
    nb_none = np.zeros_like(pt)
    ax.plot(pt, nb_model, color=TEAL, lw=2.2, label="model")
    ax.plot(pt, nb_all, color=GOLD, lw=1.8, label="treat all")
    ax.plot(pt, nb_none, color=SLATE, lw=1.5, label="treat none")
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_ylim(-0.1, 0.35)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, title)


def draw_glossary_margins(ax, title, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    terms = ["margin", "support", "kernel", "dual", "slack"]
    for i, t in enumerate(terms):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, t, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, title)


C201 = [
    ("Condition number error magnification", draw_condnum),
    ("Intended use statement blocks", draw_intended_use),
    ("No free lunch risk averages", draw_no_free_lunch),
    ("Small multiples vs spaghetti", draw_small_multiples),
    ("Total variation distance masses", draw_tv_distance),
    ("Silhouette a-b geometry", draw_silhouette),
    ("Support confidence lift bars", draw_lift_surface),
    ("Temporal join leakage timeline", draw_temporal_leakage),
    ("Parallel analysis vs scree", draw_parallel_analysis),
    ("Cook distance leverage residual", draw_cooks_distance),
    ("PR curves shift with prevalence", draw_pr_prevalence),
    ("Vanishing gradient depth product", draw_vanishing_grad),
    ("Hard negative mining radius", draw_hard_negatives),
    ("CTC blank-aware alignment path", draw_ctc_path),
    ("Exploration bonus vs extrinsic", draw_exploration_bonus),
    ("Weight tying embed and head", draw_weight_tying),
    ("Graph oversquashing bottleneck", draw_oversquash),
    ("Selection collider bias path", draw_selection_bias),
    ("Population stability index bins", draw_psi_drift),
    ("Bias variance irreducible strip", draw_bias_var_strip),
]

C202 = [
    ("Schatten-p norms of singular values", draw_schatten),
    ("Risk tier control ladder", draw_risk_tiers),
    ("Excess risk three-way split", draw_excess_risk),
    ("ECDF with DKW band sketch", draw_ecdf_bands),
    ("PIT histogram calibration check", draw_pit_hist),
    ("Density core border validity", draw_dbcv),
    ("PrefixSpan projection search", draw_prefixspan),
    ("Out-of-fold target encoding", draw_target_enc_cv),
    ("Isomap geodesic vs chord", draw_isomap_geodesic),
    ("Partial residual nonlinearity", draw_partial_residual),
    ("Cost curves for class imbalance", draw_cost_curve),
    ("Residual skip gradient highway", draw_skip_connection),
    ("InfoNCE positive logit margin", draw_info_nce),
    ("RoPE positional rotation angles", draw_rope_angles),
    ("Lambda-return TD-MC blend", draw_gae_lambda),
    ("KV cache decode memory trade", draw_kv_cache),
    ("Graph rewiring long-range links", draw_graph_rewire),
    ("Dataset shift taxonomy tiles", draw_dataset_shift_types),
    ("Decision curve net benefit", draw_decision_impact),
    ("Glossary margin support kernel", draw_glossary_margins),
]


def embed(cycle: int, topics: list) -> None:
    for i, (title, fn) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        rng = np.random.default_rng(cycle * 1000 + i * 17 + 201)
        fn(ax, title, rng)
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


def main(cycles=None):
    mapping = {201: C201, 202: C202}
    for c in cycles or [201, 202]:
        embed(c, mapping[c])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main([int(x) for x in sys.argv[1].split(",")])
    else:
        main()
