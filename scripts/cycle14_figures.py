#!/usr/bin/env python3
"""Cycle-14 densify figures — raise ML floor to >=13 (teal; original)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#0d9488"
DEEP = "#0f766e"
INK = "#0f172a"
GOLD = "#c9a227"
SOFT = "#ecfeff"
SLATE = "#64748b"
GRAY = "#94a3b8"


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", path.name)
    return path


def style_ax(ax, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def fig_pdp_not_cause():
    """Ch01: partial dependence shows association geometry, not intervention effects."""
    rng = np.random.default_rng(21)
    n = 500
    # Confounder U drives both X and Y; X has little direct effect
    U = rng.normal(size=n)
    X = 0.9 * U + rng.normal(0, 0.35, size=n)
    Y = 1.1 * U + 0.05 * X + rng.normal(0, 0.4, size=n)

    # PDP-style: bin X and mean Y (observational)
    edges = np.linspace(X.min(), X.max(), 12)
    cx, my = [], []
    for i in range(len(edges) - 1):
        m = (X >= edges[i]) & (X < edges[i + 1])
        if m.sum() < 15:
            continue
        cx.append(0.5 * (edges[i] + edges[i + 1]))
        my.append(Y[m].mean())
    cx, my = np.array(cx), np.array(my)

    # True interventional E[Y|do(X=x)] ≈ 0.05*x (holding U distribution)
    xs = np.linspace(X.min(), X.max(), 50)
    # rough do: sample U from marginal, set X=x
    inter = []
    for x in xs:
        inter.append((1.1 * U + 0.05 * x).mean())
    inter = np.array(inter)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.scatter(X, Y, s=10, c=TEAL, alpha=0.35, edgecolors="none")
    ax.plot(cx, my, "o-", color=GOLD, lw=2.4, markersize=6, label="PDP / mean Y | X (obs.)")
    ax.plot(xs, inter, color=DEEP, lw=2.4, label=r"approx $E[Y\mid do(X)]$")
    ax.set_xlabel("feature X")
    ax.set_ylabel("outcome Y")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Observational curve ≠ intervention")
    ax.text(0.02, 0.08, "Confounder U → X and U → Y\nmakes PDP look causal.",
            transform=ax.transAxes, fontsize=8, color=SLATE)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    # DAG sketch
    for (x, y, lab, col) in [(5, 5.5, "U confounder", GOLD), (2, 2.2, "X", TEAL), (8, 2.2, "Y", DEEP)]:
        ax.add_patch(Circle((x, y), 0.85, facecolor=col, edgecolor="none", alpha=0.9))
        ax.text(x, y, lab, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(2.7, 2.8), xytext=(4.3, 4.9),
                arrowprops=dict(arrowstyle="->", color=INK, lw=2))
    ax.annotate("", xy=(7.3, 2.8), xytext=(5.7, 4.9),
                arrowprops=dict(arrowstyle="->", color=INK, lw=2))
    ax.annotate("", xy=(7.1, 2.2), xytext=(2.9, 2.2),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5, ls="--"))
    ax.text(5, 1.3, "dashed = weak/direct; solid = confounding paths", ha="center", fontsize=8, color=SLATE)
    ax.text(5, 0.5, "PDP / ICE / SHAP follow the observational joint — not do(·).",
            ha="center", fontsize=9, color=INK, fontweight="bold")
    style_ax(ax, "Why the curves disagree")
    ax.set_title("Why the curves disagree", fontsize=12, fontweight="bold", color=INK)
    fig.suptitle("Partial dependence is not a causal effect (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_pdp_not_cause.png")


def fig_attention_heads():
    """Ch12: multi-head specialization caricature."""
    rng = np.random.default_rng(4)
    n = 12  # tokens
    # Head 1: local banded
    H1 = np.eye(n)
    for i in range(n):
        for j in range(n):
            H1[i, j] = np.exp(-0.5 * ((i - j) / 1.2) ** 2)
        H1[i] /= H1[i].sum()
    # Head 2: prefers early tokens (CLS-ish)
    H2 = np.zeros((n, n))
    for i in range(n):
        w = np.array([3.0 if j < 2 else 0.3 for j in range(n)], dtype=float)
        w[i] += 0.5
        H2[i] = w / w.sum()
    # Head 3: sparse content peaks
    H3 = rng.dirichlet(np.ones(n) * 0.35, size=n)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, H, title, cmap in [
        (axes[0], H1, "Head A — local / positional", "YlGnBu"),
        (axes[1], H2, "Head B — prefix bias", "YlOrBr"),
        (axes[2], H3, "Head C — sparse content", "BuPu"),
    ]:
        im = ax.imshow(H, cmap=cmap, vmin=0, vmax=H.max())
        ax.set_xlabel("key pos")
        ax.set_ylabel("query pos")
        style_ax(ax, title)
        ax.set_xticks([0, n // 2, n - 1])
        ax.set_yticks([0, n // 2, n - 1])
    fig.suptitle("Multi-head attention: specialization caricature (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.text(0.5, -0.02,
             "Different heads can emphasize locality, prefixes, or sparse tokens — still not causal explanations of clinical text.",
             ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_attention_heads.png")


def fig_gap_statistic():
    """Ch04: gap statistic sketch for choosing k."""
    rng = np.random.default_rng(8)
    # True 3 blobs
    centers = np.array([[0, 0], [3.5, 0.2], [1.5, 3.0]])
    X = np.vstack([rng.normal(c, 0.45, size=(60, 2)) for c in centers])

    def wss(X, k, seed=0):
        r = np.random.default_rng(seed)
        # kmeans-ish
        cents = X[r.choice(len(X), size=k, replace=False)]
        for _ in range(15):
            d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            lab = d.argmin(axis=1)
            for j in range(k):
                if (lab == j).any():
                    cents[j] = X[lab == j].mean(axis=0)
        d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        return sum(((X[lab == j] - cents[j]) ** 2).sum() for j in range(k) if (lab == j).any())

    ks = np.arange(1, 8)
    logW = np.array([np.log(wss(X, k, seed=k) + 1e-9) for k in ks])
    # reference: uniform over bounding box
    lo, hi = X.min(axis=0), X.max(axis=0)
    logW_ref = []
    for k in ks:
        refs = []
        for b in range(12):
            Xu = rng.uniform(lo, hi, size=X.shape)
            refs.append(np.log(wss(Xu, k, seed=100 + b * k) + 1e-9))
        logW_ref.append(np.mean(refs))
    logW_ref = np.array(logW_ref)
    gap = logW_ref - logW

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=14, c=TEAL, alpha=0.7, edgecolors="none")
    ax.scatter(centers[:, 0], centers[:, 1], s=120, c=GOLD, edgecolors=INK, zorder=5, label="true centers")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Synthetic 3-cluster data")

    ax = axes[1]
    ax.plot(ks, gap, "o-", color=TEAL, lw=2.4, markersize=7, label="Gap(k)")
    k_star = ks[gap.argmax()]
    ax.axvline(k_star, color=GOLD, ls="--", lw=1.5, label=rf"argmax Gap = {k_star}")
    ax.set_xlabel("k")
    ax.set_ylabel("Gap statistic (teaching)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, r"Gap(k)=E*[log W_k] − log W_k")
    ax.text(0.98, 0.08, "Elbows & gaps are heuristics.\nStability and domain labels matter.\nClusters ≠ etiologic subtypes.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Gap statistic for choosing k (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_gap_statistic.png")


def fig_viz_truncation():
    """Ch02: truncated y-axis and aspect distortion caution."""
    cats = ["Site A", "Site B", "Site C", "Site D"]
    rates = np.array([12.1, 12.6, 13.0, 12.4])  # events per 100

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.bar(cats, rates, color=TEAL, edgecolor="white")
    ax.set_ylim(11.5, 13.2)
    ax.set_ylabel("rate per 100")
    style_ax(ax, "Truncated axis → exaggerated gaps")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.5, 0.08, "Same data, zoomed y-lim — differences look huge.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#b45309")

    ax = axes[1]
    ax.bar(cats, rates, color=DEEP, edgecolor="white")
    ax.set_ylim(0, 20)
    ax.set_ylabel("rate per 100")
    style_ax(ax, "Zero-baseline → honest magnitude")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(0.5, 0.08, "Start at 0 unless log scale is justified & labeled.",
            transform=ax.transAxes, ha="center", fontsize=8, color=SLATE)
    fig.suptitle("Visualization hygiene: y-axis truncation distorts clinical rates (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_axis_truncation.png")


def fig_minhash_banding():
    """Ch05: MinHash LSH banding recall vs false positives."""
    # Teaching curves: more bands → higher recall of similar pairs, more FP
    # s = Jaccard similarity
    s = np.linspace(0.05, 0.95, 80)
    # probability of becoming candidate: 1 - (1 - s^r)^b
    configs = [(10, 2, TEAL, "b=10,r=2"), (20, 5, GOLD, "b=20,r=5"), (40, 8, DEEP, "b=40,r=8")]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    for b, r, c, lab in configs:
        p = 1 - (1 - s ** r) ** b
        ax.plot(s, p, color=c, lw=2.3, label=lab)
    ax.set_xlabel("true Jaccard s")
    ax.set_ylabel("P(candidate pair)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "LSH banding S-curves")
    ax.axvline(0.7, color=GRAY, ls=":", lw=1.2)
    ax.text(0.72, 0.2, "target s", fontsize=8, color=SLATE)

    ax = axes[1]
    # tradeoff: expected candidates vs threshold
    thr = np.linspace(0.3, 0.9, 30)
    # synthetic: FP rate decreases with thr, recall decreases
    recall = 1 / (1 + np.exp(12 * (thr - 0.65)))
    fp = 0.35 * (1 - thr) ** 1.5
    ax.plot(thr, recall, color=TEAL, lw=2.4, label="recall of true near-dupes")
    ax.plot(thr, fp, color=GOLD, lw=2.4, label="relative FP load")
    ax.set_xlabel("Jaccard threshold for 'near duplicate'")
    ax.set_ylabel("rate (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Threshold trades recall vs review load")
    ax.text(0.98, 0.08, "MinHash approximates Jaccard;\nbanding controls candidate explosion.\nNot a causal graph of notes.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("MinHash LSH banding: candidate probability vs similarity (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_minhash_banding.png")


def fig_lr_schedule():
    """Ch10: LR schedules — step, cosine, warmup."""
    T = 200
    t = np.arange(T)
    base = 1e-3
    # step decay
    step = np.ones(T) * base
    step[t >= 80] = base * 0.1
    step[t >= 140] = base * 0.01
    # cosine
    cosine = base * 0.5 * (1 + np.cos(np.pi * t / T))
    # warmup + cosine
    wu = 20
    warm = np.zeros(T)
    for i in t:
        if i < wu:
            warm[i] = base * (i + 1) / wu
        else:
            warm[i] = base * 0.5 * (1 + np.cos(np.pi * (i - wu) / (T - wu)))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(t, step, color=GRAY, lw=2.0, label="step decay")
    ax.plot(t, cosine, color=GOLD, lw=2.2, label="cosine")
    ax.plot(t, warm, color=TEAL, lw=2.4, label="warmup + cosine")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Common LR schedules")

    ax = axes[1]
    rng = np.random.default_rng(3)
    # synthetic train loss under too-high LR vs warmup
    loss_hi = 2.5 * np.exp(-t / 90) + 0.4 + 0.15 * np.sin(t / 3) * (t < 40) + rng.normal(0, 0.03, T)
    loss_hi[:15] += np.linspace(0.8, 0, 15)  # unstable start
    loss_wu = 2.2 * np.exp(-t / 55) + 0.35 + rng.normal(0, 0.02, T)
    ax.plot(t, loss_hi, color="#ef4444", lw=1.8, alpha=0.85, label="LR too high early")
    ax.plot(t, loss_wu, color=TEAL, lw=2.2, label="warmup stabilizes")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Why warmup appears in deep nets")
    ax.text(0.98, 0.9, "Schedule is a knob, not magic.\nRetune when batch size / AMP changes.\nLow loss ≠ clinical validity.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Learning-rate schedules and warmup (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_lr_schedule.png")


def fig_hard_negatives():
    """Ch11: contrastive hard negatives vs easy."""
    rng = np.random.default_rng(12)
    # Anchor at origin; positives nearby; easy negatives far; hard near boundary
    pos = rng.normal([0.8, 0.1], 0.12, size=(25, 2))
    easy = rng.normal([-1.5, 1.2], 0.35, size=(40, 2))
    hard = rng.normal([0.55, 0.45], 0.18, size=(25, 2))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.scatter(easy[:, 0], easy[:, 1], s=22, c=GRAY, alpha=0.7, label="easy negatives")
    ax.scatter(hard[:, 0], hard[:, 1], s=28, c=GOLD, alpha=0.85, label="hard negatives")
    ax.scatter(pos[:, 0], pos[:, 1], s=28, c=TEAL, alpha=0.9, label="positives")
    ax.scatter([0], [0], s=140, c=DEEP, marker="*", edgecolors="white", linewidths=0.8, zorder=5, label="anchor")
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Contrastive batch geometry")

    ax = axes[1]
    # loss contribution caricature
    sims = np.linspace(-0.2, 0.95, 60)
    # InfoNCE-like weight ~ exp(sim/tau)
    tau = 0.1
    w = np.exp(sims / tau)
    w = w / w.max()
    ax.plot(sims, w, color=TEAL, lw=2.5)
    ax.axvspan(0.6, 0.95, color=GOLD, alpha=0.2, label="hard-negative band")
    ax.set_xlabel("cosine(sim) to negative")
    ax.set_ylabel("relative gradient pull (teaching)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Hard negatives dominate the loss")
    ax.text(0.02, 0.08, "Too-hard (false) negatives hurt SSL.\nProbe labels must stay out of pretraining.\nGeometry ≠ disease causation.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.suptitle("Contrastive learning: easy vs hard negatives (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_hard_negatives.png")


def fig_odds_prob_glossary():
    """Ch18: odds vs probability visual dictionary."""
    p = np.linspace(0.02, 0.95, 100)
    odds = p / (1 - p)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(p, odds, color=TEAL, lw=2.5)
    for pp in [0.1, 0.25, 0.5, 0.75]:
        oo = pp / (1 - pp)
        ax.plot([pp], [oo], "o", color=GOLD, markersize=8)
        ax.annotate(f"p={pp:.2f}\nodds={oo:.2f}", (pp, oo), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=DEEP)
    ax.set_xlabel("probability p")
    ax.set_ylabel("odds = p/(1-p)")
    ax.set_ylim(0, 8)
    style_ax(ax, "Odds grow nonlinearly with p")

    ax = axes[1]
    # LR update on odds
    prior_p = 0.1
    prior_odds = prior_p / (1 - prior_p)
    lr_plus = 4.0
    post_odds = prior_odds * lr_plus
    post_p = post_odds / (1 + post_odds)
    stages = ["prior\np", "prior\nodds", "× LR+", "post\nodds", "post\np"]
    vals = [prior_p, prior_odds, lr_plus, post_odds, post_p]
    colors = [GRAY, TEAL, GOLD, TEAL, DEEP]
    ax.bar(stages, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("value")
    style_ax(ax, "Bayes on odds: multiply LR, then map back")
    ax.text(0.5, 0.92, rf"example: p₀={prior_p}, LR+={lr_plus:.0f} → p₁={post_p:.2f}",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK)
    ax.text(0.98, 0.08, "Do not add probability points\nas if they were odds.\nModels output scores—not causes.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Glossary: probability vs odds and LR updates (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_odds_vs_prob.png")


def fig_group_dro():
    """Ch01/16: group DRO vs ERM worst-group accuracy."""
    groups = ["Site A\nmajority", "Site B", "Site C", "Site D\nrare protocol"]
    erm = np.array([0.91, 0.84, 0.79, 0.61])
    dro = np.array([0.87, 0.83, 0.81, 0.76])

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    x = np.arange(len(groups))
    w = 0.35
    ax.bar(x - w / 2, erm, w, color=GRAY, label="ERM (avg risk)", edgecolor="white")
    ax.bar(x + w / 2, dro, w, color=TEAL, label="group DRO emphasis", edgecolor="white")
    ax.set_xticks(x, groups, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Average vs worst-group performance")
    ax.axhline(erm.min(), color="#ef4444", ls=":", lw=1.2)
    ax.axhline(dro.min(), color=GOLD, ls="--", lw=1.2)

    ax = axes[1]
    # train objective sketch: upweight high-loss groups
    steps = np.arange(0, 80)
    wA = 0.7 * np.exp(-steps / 50) + 0.25
    wD = 0.15 + 0.55 * (1 - np.exp(-steps / 35))
    wB = 1 - wA - wD
    wB = np.clip(wB, 0.05, None)
    # renormalize
    S = wA + wB + wD
    wA, wB, wD = wA / S, wB / S, wD / S
    ax.stackplot(steps, wA, wB, wD, colors=[GRAY, GOLD, TEAL],
                 labels=["weight Site A", "Site B/C", "Site D (worst)"])
    ax.set_xlabel("training step")
    ax.set_ylabel("group weight in objective")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "DRO upweights high-loss groups")
    ax.text(0.02, 0.08, "Improving worst-group score can\ncost average AUROC. Groups must be\npre-specified—not mined on test.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.suptitle("Group distributionally robust optimization vs ERM (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_group_dro.png")


def main():
    fig_pdp_not_cause()
    fig_attention_heads()
    fig_gap_statistic()
    fig_viz_truncation()
    fig_minhash_banding()
    fig_lr_schedule()
    fig_hard_negatives()
    fig_odds_prob_glossary()
    fig_group_dro()
    print("DONE cycle-14 figures")


if __name__ == "__main__":
    main()
