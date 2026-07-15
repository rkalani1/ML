#!/usr/bin/env python3
"""Cycle-231/232 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc, Ellipse, Wedge, Polygon

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


def d_svd_economy(ax, t, rng):
    m, n, r = 8, 5, 3
    # economy SVD shapes as boxes
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.3, 1.5, 2.2, 3.5, f"A\n{m}×{n}", fc=SLATE, fs=12)
    box(ax, 3.2, 1.5, 2.2, 3.5, f"U\n{m}×{r}", fc=TEAL, fs=12)
    box(ax, 6.1, 2.2, 2.5, 2.2, f"Σ\n{r}×{r}", fc=GOLD, tc=INK, fs=12)
    box(ax, 9.3, 2.0, 2.5, 2.5, f"Vᵀ\n{r}×{n}", fc=DEEP, fs=12)
    ax.text(2.85, 3.2, "≈", fontsize=18, ha="center")
    ax.text(5.7, 3.2, "×", fontsize=16, ha="center")
    ax.text(8.9, 3.2, "×", fontsize=16, ha="center")
    style(ax, t)


def d_model_eval_card(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for i, lab in enumerate(["task\ndef", "splits", "metrics", "errors", "limits", "owner"]):
        r, c = divmod(i, 3)
        box(ax, 0.4 + c * 4.0, 3.5 - r * 2.4, 3.6, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=12)
    style(ax, t)


def d_benign_overfit(ax, t, rng):
    n = np.arange(10, 200, 5)
    train = 0.02 + 0.3 * np.exp(-n / 40)
    test = 0.15 + 0.1 * np.exp(-n / 60)  # still improves
    ax.plot(n, train, color=GOLD, lw=2, label="train risk")
    ax.plot(n, test, color=TEAL, lw=2.2, label="test risk")
    ax.set_xlabel("model capacity proxy")
    ax.set_ylabel("risk")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_horizon_chart(ax, t, rng):
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.3 * np.sin(3 * x)
    # horizon bands
    bands = 3
    for b in range(bands):
        lo, hi = b * 0.5, (b + 1) * 0.5
        yb = np.clip(np.abs(y) - lo, 0, 0.5)
        ax.fill_between(x, b, b + yb, color=TEAL, alpha=0.35 + 0.15 * b)
    ax.set_xlabel("time")
    ax.set_yticks([])
    ax.set_ylabel("horizon layers")
    style(ax, t)


def d_posterior_predictive(ax, t, rng):
    x = np.linspace(0, 10, 50)
    mu = 0.3 * x + 0.5
    draws = [mu + rng.normal(0, 0.4 + 0.05 * x) for _ in range(30)]
    for d in draws:
        ax.plot(x, d, color=TEAL, alpha=0.15, lw=1)
    ax.plot(x, mu, color=GOLD, lw=2.2, label="mean predictive")
    ax.set_xlabel("x")
    ax.set_ylabel("y rep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dbscan_core_border(ax, t, rng):
    core = rng.normal([0, 0], 0.3, (25, 2))
    border = rng.normal([0.9, 0.2], 0.25, (12, 2))
    noise = rng.uniform(-2, 2, (8, 2))
    ax.scatter(core[:, 0], core[:, 1], c=TEAL, s=50, label="core")
    ax.scatter(border[:, 0], border[:, 1], c=GOLD, s=50, label="border")
    ax.scatter(noise[:, 0], noise[:, 1], c=ROSE, s=40, marker="x", label="noise")
    ax.add_patch(Circle((0, 0), 0.6, fill=False, ec=DEEP, ls="--", lw=1.5))
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_eclat_tidset(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.4, 2.2, 3.5, 1.8, "vertical\ntidsets", fc=TEAL, fs=12)
    box(ax, 4.3, 2.2, 3.5, 1.8, "intersect\n↓", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.2, 2.2, 3.5, 1.8, "frequent\nitemsets", fc=DEEP, fs=12)
    for x0, x1 in [(3.9, 4.3), (7.8, 8.2)]:
        ax.annotate("", xy=(x1, 3.1), xytext=(x0, 3.1), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_count_encoding(ax, t, rng):
    cats = ["A", "B", "C", "D", "E"]
    counts = [120, 45, 8, 3, 1]
    ax.bar(cats, counts, color=TEAL, edgecolor=DEEP)
    ax.set_ylabel("frequency → count feature")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    style(ax, t)


def d_autoencoder_bottleneck(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    widths = [3.5, 2.5, 1.2, 2.5, 3.5]
    labels = ["in", "enc", "z", "dec", "out"]
    x = 0.5
    for w, lab, c in zip(widths, labels, [SLATE, TEAL, GOLD, TEAL, SLATE]):
        h = w
        box(ax, x, 3 - h / 2, 1.8, h, lab, fc=c, tc=("white" if c != GOLD else INK), fs=11)
        x += 2.3
    style(ax, t)


def d_spline_basis(ax, t, rng):
    x = np.linspace(0, 10, 200)
    knots = [2, 4, 6, 8]
    for i, k in enumerate(knots):
        # truncated power basis sketch
        b = np.maximum(x - k, 0) ** 3
        b = b / (b.max() + 1e-9)
        ax.plot(x, b + i * 0.05, color=TEAL if i % 2 == 0 else DEEP, lw=2)
    for k in knots:
        ax.axvline(k, color=GOLD, ls=":", lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("cubic spline basis")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_lift_chart(ax, t, rng):
    pct = np.linspace(0, 1, 50)
    lift = 1 + 2.5 * np.exp(-3 * pct)
    ax.plot(pct, lift, color=TEAL, lw=2.2)
    ax.axhline(1, color=SLATE, ls="--", label="baseline")
    ax.set_xlabel("population fraction")
    ax.set_ylabel("lift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_weight_decay_path(ax, t, rng):
    steps = np.arange(0, 100)
    w = 2.0 * (0.99 ** steps)
    ax.plot(steps, w, color=TEAL, lw=2.2, label="||w|| with decay")
    ax.plot(steps, 2.0 * np.ones_like(steps), color=SLATE, ls="--", label="no decay")
    ax.set_xlabel("step")
    ax.set_ylabel("weight norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_masked_siamese(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.2, 3.2, 1.8, "view A\n(masked)", fc=TEAL, fs=12)
    box(ax, 4.4, 2.2, 3.2, 1.8, "view B\n(masked)", fc=DEEP, fs=12)
    box(ax, 8.3, 2.2, 3.2, 1.8, "similarity\nloss", fc=GOLD, tc=INK, fs=12)
    ax.annotate("", xy=(8.3, 3.1), xytext=(7.6, 3.1), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(8.3, 3.3), xytext=(3.7, 3.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
    style(ax, t)


def d_detr_hungarian(ax, t, rng):
    C = rng.uniform(0.2, 2, (5, 5))
    # mark assignment diagonal-ish
    ax.imshow(C, cmap="YlOrRd")
    for i in range(5):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEAL, lw=2.5))
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground-truth")
    ax.text(2, -1.0, "Hungarian matching (teaching)", ha="center", fontsize=9, color=SLATE)
    style(ax, t)


def d_cql_ood(ax, t, rng):
    s = np.linspace(-2, 2, 40)
    a = np.linspace(-2, 2, 40)
    S, A = np.meshgrid(s, a)
    # high Q in data support only
    Q = np.exp(-0.5 * (S**2 + A**2) / 0.8) - 0.3 * (S**2 + A**2)
    cs = ax.contourf(S, A, Q, levels=15, cmap="Greens")
    ax.scatter(rng.normal(0, 0.5, 40), rng.normal(0, 0.5, 40), c=GOLD, s=12, alpha=0.7, label="data support")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("state")
    ax.set_ylabel("action")
    style(ax, t)


def d_lora_rank(ax, t, rng):
    r = np.array([1, 2, 4, 8, 16, 32, 64])
    adapt = 1 - np.exp(-r / 8)
    params = r / 64
    ax.plot(r, adapt, "o-", color=TEAL, lw=2, label="adapt quality sketch")
    ax.plot(r, params, "s--", color=GOLD, lw=2, label="param fraction")
    ax.set_xlabel("LoRA rank r")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_graph_attention(ax, t, rng):
    n = 6
    A = rng.uniform(0, 1, (n, n))
    A = A / A.sum(axis=1, keepdims=True)
    ax.imshow(A, cmap="Greens", vmin=0, vmax=A.max())
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlabel("key node")
    ax.set_ylabel("query node")
    style(ax, t)


def d_cdc_lag(ax, t, rng):
    t_ = np.arange(0, 60)
    lag = 0.5 + 0.02 * t_ + 0.3 * np.sin(t_ / 5)
    ax.plot(t_, lag, color=TEAL, lw=2)
    ax.axhline(2.0, color=GOLD, ls="--", label="max lag")
    ax.set_xlabel("minute")
    ax.set_ylabel("CDC lag (min)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_oncall_handoff(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["context", "open\nincidents", "risks", "watch\nitems", "ack"]):
        box(ax, 0.25 + i * 2.4, 1.6, 2.2, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_gloss_nn_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["MLP", "CNN", "RNN", "Transformer", "GNN"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
    style(ax, t)


def d_matrix_sketch(ax, t, rng):
    # CountSketch / sparse JL size
    eps = np.linspace(0.05, 0.5, 40)
    rows = 1 / eps**2
    ax.semilogy(eps, rows, color=TEAL, lw=2.2)
    ax.set_xlabel("ε")
    ax.set_ylabel("sketch rows sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_threat_model(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 3.5, 1.8, "assets", fc=TEAL, fs=12)
    box(ax, 4.3, 3.5, 3.5, 1.8, "adversaries", fc=ROSE, fs=12)
    box(ax, 8.1, 3.5, 3.5, 1.8, "controls", fc=GOLD, tc=INK, fs=12)
    box(ax, 2.5, 1.0, 7, 1.5, "assumptions + residual risk", fc=DEEP, fs=12)
    style(ax, t)


def d_double_descent(ax, t, rng):
    p = np.linspace(0.1, 5, 100)
    # classical U then descent
    risk = 0.3 + 0.5 * (p - 1) ** 2 * np.exp(-((p - 1) ** 2)) + 0.15 / (1 + 0.3 * p)
    # peak near interpolation
    risk = 0.2 + 0.8 * np.exp(-((p - 1.2) ** 2) / 0.15) + 0.15 * np.exp(-p / 3)
    ax.plot(p, risk, color=TEAL, lw=2.2)
    ax.axvline(1.2, color=GOLD, ls="--", label="interp. threshold")
    ax.set_xlabel("params / n")
    ax.set_ylabel("test risk sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_parallel_coords(ax, t, rng):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    dims = 5
    for _ in range(30):
        y = rng.uniform(0, 1, dims)
        ax.plot(range(dims), y, color=TEAL, alpha=0.35, lw=1)
    for i in range(dims):
        ax.plot([i, i], [0, 1], color=SLATE, lw=2)
    ax.set_xticks(range(dims))
    ax.set_xticklabels([f"f{i}" for i in range(dims)])
    ax.set_ylabel("normalized")
    style(ax, t)


def d_variational_dropout(ax, t, rng):
    alpha = np.logspace(-3, 1, 50)
    # sparsity inducing
    kl = 0.5 * np.log1p(1 / alpha)  # sketch
    ax.semilogx(alpha, kl, color=TEAL, lw=2.2)
    ax.set_xlabel("dropout variance α")
    ax.set_ylabel("KL sparsity sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_hdbscan_stability(ax, t, rng):
    clusters = np.arange(1, 8)
    stab = np.array([0.9, 0.75, 0.4, 0.85, 0.3, 0.6, 0.2])
    colors = [TEAL if s > 0.5 else SLATE for s in stab]
    ax.bar(clusters, stab, color=colors, edgecolor=DEEP)
    ax.axhline(0.5, color=GOLD, ls="--", label="keep threshold")
    ax.set_xlabel("cluster id")
    ax.set_ylabel("stability")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_cm_span(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["seq DB", "projected", "closed\ntest", "CM-Span"]):
        box(ax, 0.35 + i * 3.0, 1.6, 2.7, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    for x in [3.05, 6.05, 9.05]:
        ax.annotate("", xy=(x + 0.3, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    style(ax, t)


def d_hashing_collision(ax, t, rng):
    d = np.arange(10, 500, 10)
    # birthday collision prob sketch n=1000
    n = 1000
    p = 1 - np.exp(-n * (n - 1) / (2 * d))
    ax.plot(d, p, color=TEAL, lw=2.2)
    ax.set_xlabel("hash dim D")
    ax.set_ylabel("collision prob sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_sparse_coding(ax, t, rng):
    dict_atoms = rng.normal(0, 1, (20, 2))
    dict_atoms /= np.linalg.norm(dict_atoms, axis=1, keepdims=True)
    ax.scatter(dict_atoms[:, 0], dict_atoms[:, 1], c=TEAL, s=40)
    # sparse combo point
    x = 0.6 * dict_atoms[3] + 0.4 * dict_atoms[11]
    ax.plot(x[0], x[1], "*", color=GOLD, ms=16, label="sparse recon")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("atom x")
    ax.set_ylabel("atom y")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_quantile_crossing(ax, t, rng):
    x = np.linspace(0, 10, 50)
    q10 = 0.3 * x - 0.5
    q50 = 0.5 * x
    q90 = 0.4 * x + 1.5  # deliberate mild cross at end for teaching fix need
    ax.plot(x, q10, color=SLATE, lw=2, label="τ=0.1")
    ax.plot(x, q50, color=TEAL, lw=2, label="τ=0.5")
    ax.plot(x, q90, color=GOLD, lw=2, label="τ=0.9")
    ax.set_xlabel("x")
    ax.set_ylabel("quantile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_class_imbalance_weights(ax, t, rng):
    classes = np.arange(6)
    n = np.array([500, 200, 80, 30, 10, 5])
    w = n.sum() / (len(n) * n)
    ax.bar(classes - 0.2, n / n.max(), width=0.4, color=SLATE, label="freq (norm)")
    ax.bar(classes + 0.2, w / w.max(), width=0.4, color=TEAL, label="inv-freq weight")
    ax.set_xlabel("class")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_adamw_decouple(ax, t, rng):
    steps = np.arange(0, 50)
    adam = 1.5 * (0.995 ** steps)
    adamw = 1.5 * (0.99 ** steps)  # stronger explicit decay
    ax.plot(steps, adam, color=SLATE, lw=2, label="Adam L2-in-grad")
    ax.plot(steps, adamw, color=TEAL, lw=2.2, label="AdamW decoupled")
    ax.set_xlabel("step")
    ax.set_ylabel("||w|| sketch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mae_asymm(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 3.5, 1.8, "heavy encoder\nvisible 25%", fc=TEAL, fs=11)
    box(ax, 4.5, 3.5, 3.5, 1.8, "light decoder\n+ mask tokens", fc=GOLD, tc=INK, fs=11)
    box(ax, 8.5, 3.5, 3.2, 1.8, "pixel /\ntoken recon", fc=DEEP, fs=11)
    for x0, x1 in [(4.0, 4.5), (8.0, 8.5)]:
        ax.annotate("", xy=(x1, 4.4), xytext=(x0, 4.4), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    style(ax, t)


def d_conformer(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["FFN", "MHSA", "conv\nmodule", "FFN", "norm"]):
        box(ax, 0.25 + i * 2.4, 1.6, 2.2, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_soft_actor_critic(ax, t, rng):
    temp = np.linspace(0.05, 2, 40)
    ret = 10 - 2 * (temp - 0.5) ** 2
    ent = 3 * np.log(temp + 0.1)
    ax.plot(temp, ret, color=TEAL, lw=2, label="return sketch")
    ax.plot(temp, ent, color=GOLD, lw=2, label="entropy bonus")
    ax.set_xlabel("temperature α")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_bnb_nf4(ax, t, rng):
    x = rng.normal(0, 1, 5000)
    # show quant bins
    qs = np.linspace(-3, 3, 17)
    ax.hist(x, bins=qs, color=TEAL, edgecolor=DEEP, alpha=0.85)
    ax.set_xlabel("weight")
    ax.set_ylabel("count (NF4-ish bins)")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_metapath2vec(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["A", "P", "A", "P", "T"]):
        box(ax, 0.5 + i * 2.3, 2.0, 2.0, 1.5, lab, fc=TEAL if lab == "A" else (GOLD if lab == "P" else DEEP), tc=("white" if lab != "P" else INK), fs=14)
    ax.text(6, 1.0, "metapath A-P-A-P-T guided walk", ha="center", fontsize=10, color=SLATE)
    style(ax, t)


def d_feature_flag(ax, t, rng):
    t_ = np.arange(0, 20)
    on = (t_ >= 8).astype(float)
    metric = 0.8 + 0.05 * on + rng.normal(0, 0.01, 20)
    ax.step(t_, on, where="post", color=GOLD, lw=2, label="flag")
    ax.plot(t_, metric, color=TEAL, lw=2, label="metric")
    ax.set_xlabel("day")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_runbook_severity(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.5, 3.5, 3.5, 1.8, "symptoms", fc=SLATE, fs=12)
    box(ax, 4.3, 3.5, 3.5, 1.8, "severity\ntree", fc=TEAL, fs=12)
    box(ax, 8.1, 4.2, 3.5, 1.2, "SEV1 path", fc=ROSE, fs=11)
    box(ax, 8.1, 2.5, 3.5, 1.2, "SEV3 path", fc=GOLD, tc=INK, fs=11)
    ax.annotate("", xy=(4.3, 4.4), xytext=(4.0, 4.4), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    style(ax, t)


def d_gloss_data_strip(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["ETL", "CDC", "DQ", "lineage", "SLA"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C231 = [
    ("Economy SVD shape diagram", d_svd_economy),
    ("Model evaluation card grid", d_model_eval_card),
    ("Benign overfitting risk curves", d_benign_overfit),
    ("Horizon chart layered bands", d_horizon_chart),
    ("Posterior predictive draw fan", d_posterior_predictive),
    ("DBSCAN core border noise", d_dbscan_core_border),
    ("Eclat tidset intersections", d_eclat_tidset),
    ("Count encoding frequencies", d_count_encoding),
    ("Autoencoder bottleneck widths", d_autoencoder_bottleneck),
    ("Cubic spline basis bundle", d_spline_basis),
    ("Marketing lift curve", d_lift_chart),
    ("Weight decay norm path", d_weight_decay_path),
    ("Masked Siamese dual views", d_masked_siamese),
    ("DETR Hungarian cost assignment", d_detr_hungarian),
    ("CQL OOD Q landscape", d_cql_ood),
    ("LoRA rank quality tradeoff", d_lora_rank),
    ("Graph attention weight matrix", d_graph_attention),
    ("CDC replication lag trace", d_cdc_lag),
    ("On-call handoff checklist", d_oncall_handoff),
    ("Glossary neural architecture strip", d_gloss_nn_strip),
]

C232 = [
    ("Matrix sketch size vs epsilon", d_matrix_sketch),
    ("Threat model asset map", d_threat_model),
    ("Double descent risk curve", d_double_descent),
    ("Parallel coordinates profiles", d_parallel_coords),
    ("Variational dropout KL path", d_variational_dropout),
    ("HDBSCAN cluster stability bars", d_hdbscan_stability),
    ("CM-Span closed sequence path", d_cm_span),
    ("Hash collision vs dimension", d_hashing_collision),
    ("Sparse coding dictionary atoms", d_sparse_coding),
    ("Quantile regression non-crossing", d_quantile_crossing),
    ("Class imbalance inverse weights", d_class_imbalance_weights),
    ("AdamW decoupled decay norms", d_adamw_decouple),
    ("MAE asymmetric encoder-decoder", d_mae_asymm),
    ("Conformer block sandwich", d_conformer),
    ("SAC temperature entropy trade", d_soft_actor_critic),
    ("bitsandbytes NF4 histogram", d_bnb_nf4),
    ("metapath2vec typed walk", d_metapath2vec),
    ("Feature flag metric step", d_feature_flag),
    ("Runbook severity decision tree", d_runbook_severity),
    ("Glossary data-ops strip", d_gloss_data_strip),
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

    m = {231: C231, 232: C232}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [231, 232]
    for c in cycles:
        embed(c, m[c])
