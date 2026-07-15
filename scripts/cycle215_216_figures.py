#!/usr/bin/env python3
"""Cycle-215/216 quality densify: novel scientific teal teaching panels."""
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


def d_householder(ax, t, rng):
    v = np.array([1.0, 0.8])
    v = v / np.linalg.norm(v)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=SLATE, ls="--", lw=1)
    # reflector hyperplane
    n = v
    ax.plot([-1.2 * n[1], 1.2 * n[1]], [1.2 * n[0], -1.2 * n[0]], color=GOLD, lw=2, label="mirror")
    x = np.array([0.9, 0.2])
    # reflection Hx = x - 2 (v·x) v
    hx = x - 2 * np.dot(v, x) * v
    ax.annotate("", xy=x, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.annotate("", xy=hx, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ROSE, lw=2))
    ax.text(x[0], x[1], " x", color=TEAL, fontsize=10)
    ax.text(hx[0], hx[1], " Hx", color=ROSE, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_reconsent(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.0, 3.2, 1.8, "protocol\nchange", fc=SLATE, fs=11)
    box(ax, 4.4, 2.0, 3.2, 1.8, "re-consent\ngate", fc=TEAL, fs=11)
    box(ax, 8.3, 2.0, 3.2, 1.8, "continue\nor exit", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.4, 2.9), xytext=(3.7, 2.9), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.3, 2.9), xytext=(7.6, 2.9), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, t)


def d_uniform_convergence(ax, t, rng):
    x = np.linspace(0, 1, 100)
    for i in range(12):
        f = 0.5 + 0.2 * np.sin(2 * np.pi * (i + 1) * x) * rng.uniform(0.5, 1)
        ax.plot(x, f, color=TEAL, alpha=0.35, lw=1)
    ax.plot(x, 0.5 + 0.05 * np.sin(2 * np.pi * x), color=GOLD, lw=2.5, label="true f*")
    ax.fill_between(x, 0.35, 0.65, color=ROSE, alpha=0.1, label="ε-tube")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("f")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mosaic_plot(ax, t, rng):
    # 2x3 mosaic proportions
    widths = [0.4, 0.35, 0.25]
    heights = [[0.7, 0.3], [0.45, 0.55], [0.2, 0.8]]
    x0 = 0.1
    for w, hs in zip(widths, heights):
        y0 = 0.1
        for j, h in enumerate(hs):
            ax.add_patch(Rectangle((x0, y0), w * 0.85, h * 0.8, facecolor=TEAL if j == 0 else GOLD, edgecolor=DEEP, alpha=0.85))
            y0 += h * 0.8
        x0 += w * 0.9
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "mosaic: area ∝ joint count", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_dirichlet_simplex(ax, t, rng):
    # ternary sketch with samples
    def to_xy(p):
        # p3 barycentric
        a, b, c = p
        x = 0.5 * (2 * b + c) / (a + b + c)
        y = (np.sqrt(3) / 2) * c / (a + b + c)
        return x, y

    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    ax.plot(*np.vstack([tri, tri[0]]).T, color=SLATE, lw=1.5)
    samples = rng.dirichlet([2, 2, 2], 80)
    xy = np.array([to_xy(s) for s in samples])
    ax.scatter(xy[:, 0], xy[:, 1], c=TEAL, s=18, alpha=0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, -0.12, "Dirichlet draws on simplex", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_denclue(ax, t, rng):
    x = np.linspace(-3, 3, 200)
    dens = 0.6 * np.exp(-0.5 * (x + 1) ** 2) + 0.5 * np.exp(-0.5 * ((x - 1.2) / 0.7) ** 2)
    ax.fill_between(x, dens, color=TEAL, alpha=0.4)
    ax.plot(x, dens, color=DEEP, lw=2)
    # hill climbing paths
    for start in [-2.5, -0.5, 0.5, 2.2]:
        xs = [start]
        for _ in range(15):
            g = - (xs[-1] + 1) * 0.6 * np.exp(-0.5 * (xs[-1] + 1) ** 2) - (xs[-1] - 1.2) / 0.7 * 0.5 * np.exp(-0.5 * ((xs[-1] - 1.2) / 0.7) ** 2)
            # gradient of dens approx via finite
            i = int(np.argmin(np.abs(x - xs[-1])))
            if i < len(x) - 1:
                xs.append(xs[-1] + 0.15 * np.sign(dens[min(i + 1, 199)] - dens[max(i - 1, 0)]))
        ax.plot(xs, np.interp(xs, x, dens) + 0.02, color=GOLD, lw=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("kernel density")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_winnow_seq(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 3.5, 1.6, "candidate\nwindows", fc=SLATE, fs=11)
    box(ax, 4.5, 2.5, 3.5, 1.6, "support\ncount", fc=TEAL, fs=11)
    box(ax, 8.5, 2.5, 3.2, 1.6, "frequent\nseqs", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.5, 3.3), xytext=(4.0, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.5, 3.3), xytext=(8.0, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, t)


def d_count_vectorizer(ax, t, rng):
    terms = ["pain", "weak", "NIHSS", "MRI", "tPA"]
    docs = np.array([[2, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 1]])
    ax.imshow(docs, cmap="Greens", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(terms, rotation=30, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["d1", "d2", "d3"])
    for i in range(3):
        for j in range(5):
            ax.text(j, i, str(docs[i, j]), ha="center", va="center", color=INK, fontsize=10)
    style(ax, t)


def d_kernel_ridge(ax, t, rng):
    x = np.linspace(-3, 3, 80)
    y = np.sin(x) + rng.normal(0, 0.15, 80)
    # RBF-ish smooth
    pred = np.sin(x) * 0.9
    ax.scatter(x, y, c=TEAL, s=16, alpha=0.6, label="data")
    ax.plot(x, pred, color=GOLD, lw=2.2, label="kernel ridge fit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_studentized_resid(ax, t, rng):
    n = 40
    x = np.linspace(0, 10, n)
    r = rng.normal(0, 1, n)
    r[12] = 3.5
    r[30] = -3.2
    stud = r / (1.0 * np.sqrt(1 - 0.05))  # caricature
    ax.stem(x, stud, linefmt=TEAL, markerfmt="o", basefmt=" ")
    ax.axhline(2, color=GOLD, ls="--")
    ax.axhline(-2, color=GOLD, ls="--")
    ax.set_xlabel("case")
    ax.set_ylabel("studentized residual")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_micro_macro_f1(ax, t, rng):
    classes = ["A", "B", "C", "D"]
    f1 = [0.9, 0.4, 0.7, 0.55]
    support = np.array([200, 20, 80, 50])
    micro = np.average(f1, weights=support)
    macro = np.mean(f1)
    x = np.arange(len(classes))
    ax.bar(x, f1, color=TEAL, edgecolor=DEEP)
    ax.axhline(macro, color=GOLD, ls="--", lw=2, label=f"macro={macro:.2f}")
    ax.axhline(micro, color=ROSE, ls=":", lw=2, label=f"micro≈{micro:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("per-class F1")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_lookahead(ax, t, rng):
    steps = np.arange(0, 40)
    loss = np.exp(-steps / 12) + 0.1 * np.sin(steps)
    la = np.exp(-steps / 10) * 0.9 + 0.08 * np.sin(steps + 0.5)
    ax.plot(steps, loss, color=SLATE, lw=1.8, label="SGD")
    ax.plot(steps, la, color=TEAL, lw=2.2, label="Lookahead slow weights")
    ax.legend(fontsize=8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_rotation_pred(ax, t, rng):
    angles = [0, 90, 180, 270]
    for i, a in enumerate(angles):
        ax.add_patch(Rectangle((i * 2.5, 1.2), 2, 2, facecolor=TEAL, edgecolor=DEEP, alpha=0.8, transform=ax.transData))
        ax.text(i * 2.5 + 1, 2.2, f"{a}°", ha="center", va="center", color="white", fontsize=14, fontweight="bold", rotation=0)
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.text(4.5, 3.8, "RotNet: predict rotation class", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_byte_pair(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    stages = ["chars", "merges\nrank 1", "merges\nrank k", "subwords"]
    for i, s in enumerate(stages):
        box(ax, 0.4 + i * 3.0, 1.6, 2.7, 1.8, s, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
        if i < 3:
            ax.annotate("", xy=(3.2 + i * 3.0, 2.5), xytext=(3.0 + i * 3.0, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, t)


def d_sac_temperature(ax, t, rng):
    alpha = np.linspace(0.01, 2, 40)
    # entropy bonus tradeoff caricature
    ret = 1.2 - 0.3 * (alpha - 0.5) ** 2
    ent = 0.5 * np.log(1 + 5 * alpha)
    ax.plot(alpha, ret, color=TEAL, lw=2, label="return")
    ax.plot(alpha, ent, color=GOLD, lw=2, label="policy entropy")
    ax.set_xlabel("temperature α")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_pruning_lottery(ax, t, rng):
    sparsity = np.array([0, 0.2, 0.5, 0.8, 0.9, 0.95])
    acc = np.array([0.9, 0.89, 0.88, 0.84, 0.78, 0.65])
    rewind = np.array([0.9, 0.895, 0.89, 0.87, 0.84, 0.8])
    ax.plot(sparsity, acc, "o-", color=ROSE, lw=2, label="prune trained")
    ax.plot(sparsity, rewind, "s-", color=TEAL, lw=2, label="lottery rewind")
    ax.set_xlabel("sparsity")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_node2vec_walk(ax, t, rng):
    pos = {i: (np.cos(2 * np.pi * i / 7), np.sin(2 * np.pi * i / 7)) for i in range(7)}
    for i in range(7):
        j = (i + 1) % 7
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=SLATE, lw=1)
        ax.plot([pos[i][0], pos[(i + 2) % 7][0]], [pos[i][1], pos[(i + 2) % 7][1]], color=SLATE, lw=0.5, alpha=0.4)
    walk = [0, 1, 3, 4, 6, 5]
    for a, b in zip(walk[:-1], walk[1:]):
        ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    for i, p in pos.items():
        ax.plot(*p, "o", color=GOLD if i in walk else DEEP, ms=12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -1.5, "node2vec biased random walk", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_label_noise(ax, t, rng):
    noise = np.linspace(0, 0.4, 30)
    acc = 0.92 - 1.2 * noise**1.3
    ax.plot(noise, acc, color=TEAL, lw=2.2)
    ax.set_xlabel("label flip rate")
    ax.set_ylabel("test accuracy")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mttr_chart(ax, t, rng):
    months = np.arange(1, 9)
    mttr = np.array([6.5, 5.8, 5.2, 4.9, 3.8, 3.5, 3.2, 2.9])
    ax.bar(months, mttr, color=TEAL, edgecolor=DEEP)
    ax.axhline(4.0, color=GOLD, ls="--", label="SLO target 4h")
    ax.set_xlabel("month")
    ax.set_ylabel("MTTR (hours)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_glossary_nn(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["MLP", "CNN", "RNN", "Transformer", "GNN"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


# C216

def d_givens_rotate(ax, t, rng):
    th = np.linspace(0, np.pi / 3, 30)
    for i, a in enumerate(th[::5]):
        c, s = np.cos(a), np.sin(a)
        ax.annotate("", xy=(c, s), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=TEAL if i else GOLD, lw=1.8))
    ax.add_patch(Circle((0, 0), 1, fill=False, ls="--", edgecolor=SLATE))
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.2, 1.3)
    ax.text(0.6, -0.1, "Givens plane rotation", ha="center", fontsize=10, color=INK)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_audit_trail(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["who", "what", "when", "why", "hash"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_pac_bayes(ax, t, rng):
    kl = np.linspace(0.1, 8, 40)
    n = 1000
    bound = np.sqrt((kl + np.log(2 * np.sqrt(n) / 0.05)) / (2 * n))
    ax.plot(kl, bound, color=TEAL, lw=2.2)
    ax.set_xlabel("KL(Q‖P)")
    ax.set_ylabel("PAC-Bayes complexity term")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_diverging_bar(ax, t, rng):
    labs = ["feat A", "B", "C", "D", "E", "F"]
    vals = np.array([0.4, -0.2, 0.15, -0.35, 0.25, -0.1])
    colors = [TEAL if v >= 0 else ROSE for v in vals]
    ax.barh(labs, vals, color=colors, edgecolor=DEEP)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("signed effect / SHAP-like")
    ax.grid(True, axis="x", alpha=0.25)
    style(ax, t)


def d_gp_posterior(ax, t, rng):
    x = np.linspace(0, 10, 100)
    mu = np.sin(x / 2)
    std = 0.3 + 0.2 * np.sin(x / 3) ** 2
    ax.fill_between(x, mu - 2 * std, mu + 2 * std, color=TEAL, alpha=0.25, label="±2σ")
    ax.plot(x, mu, color=DEEP, lw=2, label="posterior mean")
    xs = np.array([1, 3, 6, 8])
    ax.scatter(xs, np.sin(xs / 2), c=GOLD, s=50, zorder=5, label="obs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mini_batch_kmeans(ax, t, rng):
    steps = np.arange(0, 50)
    wss = 12 * np.exp(-steps / 12) + 2 + rng.normal(0, 0.1, 50)
    ax.plot(steps, wss, color=TEAL, lw=2)
    ax.set_xlabel("mini-batch update")
    ax.set_ylabel("WSS proxy")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_sequence_logo(ax, t, rng):
    # simplified position weight
    pos = np.arange(1, 9)
    height = rng.uniform(0.3, 2.0, 8)
    ax.bar(pos, height, color=TEAL, edgecolor=DEEP)
    for i, h in enumerate(height):
        ax.text(pos[i], h + 0.05, rng.choice(list("ACGT")), ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("position")
    ax.set_ylabel("information (bits sketch)")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_helmert_coding(ax, t, rng):
    # contrast matrix heat
    H = np.array([[1, -1, 0, 0], [0.5, 0.5, -1, 0], [1 / 3, 1 / 3, 1 / 3, -1]])
    ax.imshow(H, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xlabel("level")
    ax.set_ylabel("contrast")
    style(ax, t)


def d_cca_corr(ax, t, rng):
    k = np.arange(1, 8)
    corr = np.array([0.92, 0.75, 0.4, 0.22, 0.1, 0.05, 0.02])
    ax.bar(k, corr, color=TEAL, edgecolor=DEEP)
    ax.set_xlabel("canonical component")
    ax.set_ylabel("canonical correlation")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_cooks_distance(ax, t, rng):
    n = 35
    D = rng.exponential(0.05, n)
    D[7] = 0.45
    D[22] = 0.32
    ax.stem(np.arange(n), D, linefmt=TEAL, markerfmt="o", basefmt=" ")
    ax.axhline(4 / n, color=GOLD, ls="--", label="4/n rule")
    ax.legend(fontsize=8)
    ax.set_xlabel("observation")
    ax.set_ylabel("Cook's D")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_top_k_accuracy(ax, t, rng):
    k = np.arange(1, 11)
    acc = 1 - 0.55 * np.exp(-0.35 * (k - 1))
    ax.plot(k, acc, "o-", color=TEAL, lw=2)
    ax.set_xlabel("k")
    ax.set_ylabel("top-k accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mixup(ax, t, rng):
    # two one-hots mixed
    ax.bar([0, 1, 2, 3], [1, 0, 0, 0], color=TEAL, alpha=0.5, label="y_i")
    ax.bar([0, 1, 2, 3], [0, 0, 1, 0], color=GOLD, alpha=0.5, label="y_j")
    ax.bar([0, 1, 2, 3], [0.6, 0, 0.4, 0], color=ROSE, alpha=0.7, label="λy_i+(1-λ)y_j")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["c0", "c1", "c2", "c3"])
    ax.legend(fontsize=8)
    ax.set_ylabel("label mass")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_simclr_views(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 3, 1.6, "image", fc=SLATE, fs=12)
    box(ax, 4.2, 3.2, 3, 1.2, "view t", fc=TEAL, fs=11)
    box(ax, 4.2, 1.4, 3, 1.2, "view t'", fc=DEEP, fs=11)
    box(ax, 8.2, 2.5, 3.3, 1.6, "NT-Xent\npositive pair", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.2, 3.7), xytext=(3.5, 3.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("", xy=(4.2, 2.0), xytext=(3.5, 2.7), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, t)


def d_whisper_task(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["audio", "encoder", "decoder\n+specials", "text"]):
        box(ax, 0.4 + i * 3.0, 1.6, 2.7, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
        if i < 3:
            ax.annotate("", xy=(3.2 + i * 3.0, 2.5), xytext=(3.0 + i * 3.0, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    style(ax, t)


def d_decision_transformer(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["R̂ return", "state s", "action a", "…", "a_t"]):
        box(ax, 0.3 + i * 2.35, 1.6, 2.2, 1.6, lab, fc=TEAL if i % 2 == 0 else GOLD, tc="white" if i % 2 == 0 else INK, fs=10)
    ax.text(6, 4.0, "Decision Transformer sequence", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_smoothquant(ax, t, rng):
    ch = np.arange(16)
    act = np.abs(rng.normal(0, 1, 16)) * np.linspace(0.5, 3, 16)
    w = np.abs(rng.normal(0, 1, 16))
    ax.plot(ch, act, "o-", color=ROSE, label="act outlier")
    ax.plot(ch, act / (act.max() / w.mean() + 1e-9) * w.mean() * 0 + w * (act / act.mean()), "s-", color=TEAL, label="migrated to weights")
    # simpler: show scaling
    ax.clear()
    ax.bar(ch - 0.15, act, 0.3, color=ROSE, label="|activation|")
    ax.bar(ch + 0.15, w * (act / (act.mean())), 0.3, color=TEAL, label="scaled |weight|")
    ax.legend(fontsize=8)
    ax.set_xlabel("channel")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_edge_between_flow(ax, t, rng):
    # edges with thickness = betweenness
    nodes = {"a": (0, 0), "b": (1, 1), "c": (2, 0), "d": (1, -1), "e": (3, 0.5)}
    edges = [("a", "b", 1), ("a", "d", 1), ("b", "c", 3), ("d", "c", 2), ("c", "e", 4), ("b", "e", 1)]
    for u, v, w in edges:
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color=TEAL, lw=0.8 + w * 0.8, alpha=0.8)
    for n, p in nodes.items():
        ax.plot(*p, "o", color=DEEP, ms=12)
        ax.text(p[0], p[1] + 0.15, n, ha="center", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(1.5, -1.5, "edge betweenness as stroke width", ha="center", fontsize=10, color=INK)
    style(ax, t)


def d_schema_evolution(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.0, 3.2, 1.8, "v1 schema", fc=SLATE, fs=12)
    box(ax, 4.4, 2.0, 3.2, 1.8, "migrate\n+ defaults", fc=TEAL, fs=11)
    box(ax, 8.3, 2.0, 3.2, 1.8, "v2 schema", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(4.4, 2.9), xytext=(3.7, 2.9), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("", xy=(8.3, 2.9), xytext=(7.6, 2.9), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    style(ax, t)


def d_chaos_game_day(ax, t, rng):
    weeks = np.arange(1, 11)
    # inject failures
    failures = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    ax.bar(weeks, failures, color=ROSE, edgecolor=DEEP)
    ax.set_xlabel("week")
    ax.set_ylabel("chaos experiment faults")
    ax.set_yticks([0, 1])
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_glossary_rl(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["policy", "value", "Q", "advantage", "return"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


C215 = [
    ("Householder reflector geometry", d_householder),
    ("Re-consent protocol change gate", d_reconsent),
    ("Uniform convergence epsilon tube", d_uniform_convergence),
    ("Mosaic plot joint proportions", d_mosaic_plot),
    ("Dirichlet simplex sample cloud", d_dirichlet_simplex),
    ("DENCLUE density hill climb", d_denclue),
    ("Windowed sequence support count", d_winnow_seq),
    ("Count vectorizer term-document", d_count_vectorizer),
    ("Kernel ridge smooth fit", d_kernel_ridge),
    ("Studentized residual outliers", d_studentized_resid),
    ("Micro vs macro F1 support", d_micro_macro_f1),
    ("Lookahead optimizer slow weights", d_lookahead),
    ("RotNet rotation pretext classes", d_rotation_pred),
    ("Byte-pair encoding merge ranks", d_byte_pair),
    ("SAC temperature entropy tradeoff", d_sac_temperature),
    ("Lottery ticket prune rewind", d_pruning_lottery),
    ("node2vec biased walk path", d_node2vec_walk),
    ("Label noise accuracy decay", d_label_noise),
    ("MTTR monthly trend bars", d_mttr_chart),
    ("Glossary network architecture strip", d_glossary_nn),
]

C216 = [
    ("Givens plane rotation steps", d_givens_rotate),
    ("Immutable audit trail fields", d_audit_trail),
    ("PAC-Bayes KL complexity term", d_pac_bayes),
    ("Diverging bar signed effects", d_diverging_bar),
    ("Gaussian process posterior band", d_gp_posterior),
    ("Mini-batch k-means WSS path", d_mini_batch_kmeans),
    ("Sequence motif information bars", d_sequence_logo),
    ("Helmert contrast coding matrix", d_helmert_coding),
    ("CCA canonical correlation decay", d_cca_corr),
    ("Cook distance influence stem", d_cooks_distance),
    ("Top-k accuracy saturation", d_top_k_accuracy),
    ("Mixup interpolated soft labels", d_mixup),
    ("SimCLR dual augmented views", d_simclr_views),
    ("Whisper encode-decode path", d_whisper_task),
    ("Decision Transformer RTG tokens", d_decision_transformer),
    ("SmoothQuant act-to-weight migrate", d_smoothquant),
    ("Edge betweenness stroke map", d_edge_between_flow),
    ("Schema evolution migrate defaults", d_schema_evolution),
    ("Chaos game-day fault inject", d_chaos_game_day),
    ("Glossary RL core term strip", d_glossary_rl),
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
    m = {215: C215, 216: C216}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [215, 216]
    for c in cycles:
        embed(c, m[c])
