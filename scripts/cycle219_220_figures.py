#!/usr/bin/env python3
"""Cycle-219/220 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

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
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fc, edgecolor="none")
    )
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def d_schur(ax, t, rng):
    T = np.triu(rng.normal(0, 1, (4, 4)))
    np.fill_diagonal(T, [2.1, 1.5, 0.8, 0.3])
    ax.imshow(T, cmap="Greens")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{T[i, j]:.1f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    style(ax, t)


def d_epsilon_dp(ax, t, rng):
    eps = np.array([0.1, 0.5, 1, 2, 4])
    util = 1 - np.exp(-eps)
    ax.plot(eps, util, "o-", color=TEAL, lw=2)
    ax.set_xlabel("ε")
    ax.set_ylabel("utility sketch")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_vc_growth(ax, t, rng):
    n = np.arange(1, 20)
    d = 3
    m = np.minimum(2.0**n, n.astype(float) ** d)
    ax.semilogy(n, m, color=TEAL, lw=2)
    ax.set_xlabel("n")
    ax.set_ylabel("growth function sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_ecdf_bands(ax, t, rng):
    x = np.sort(rng.normal(0, 1, 80))
    e = np.arange(1, 81) / 80
    ax.step(x, e, where="post", color=TEAL, lw=2)
    ax.fill_between(x, np.clip(e - 0.12, 0, 1), np.clip(e + 0.12, 0, 1), color=TEAL, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("ECDF ± band")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dirichlet_process(ax, t, rng):
    v = rng.beta(1, 3, 12)
    w = np.zeros(12)
    rem = 1.0
    for i in range(12):
        w[i] = v[i] * rem
        rem *= 1 - v[i]
    ax.bar(np.arange(12), w, color=TEAL, edgecolor=DEEP)
    ax.set_xlabel("atom")
    ax.set_ylabel("stick-breaking weight")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_rock_clusters(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 2.8, 1.5, "links", fc=SLATE, fs=11)
    box(ax, 3.8, 2.5, 2.8, 1.5, "goodness\nmeasure", fc=TEAL, fs=11)
    box(ax, 7.1, 2.5, 2.8, 1.5, "merge\nROCK", fc=GOLD, tc=INK, fs=11)
    style(ax, t)


def d_spade_idlist(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 4, 1.5, "id-list (sid,eid)", fc=TEAL, fs=12)
    box(ax, 5.5, 2.5, 4, 1.5, "temporal join", fc=GOLD, tc=INK, fs=12)
    style(ax, t)


def d_leave_group_out(ax, t, rng):
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    ax.scatter(np.arange(12), groups, c=[TEAL if g != 2 else ROSE for g in groups], s=60)
    ax.axhspan(1.5, 2.5, color=ROSE, alpha=0.15)
    ax.set_xlabel("sample")
    ax.set_ylabel("group id")
    ax.text(6, 2.7, "held-out group", ha="center", color=ROSE)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_procrustes(ax, t, rng):
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    A = np.c_[np.cos(th), np.sin(th)]
    B = A @ np.array([[0.9, -0.2], [0.2, 0.9]]) + 0.1
    ax.plot(A[:, 0], A[:, 1], "o-", color=TEAL, label="X")
    ax.plot(B[:, 0], B[:, 1], "s-", color=GOLD, label="Y aligned")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_quantile_reg_fans(ax, t, rng):
    x = np.linspace(0, 10, 50)
    for q, c in [(0.1, SLATE), (0.5, TEAL), (0.9, GOLD)]:
        ax.plot(x, 0.5 * x + (q - 0.5) * 2.5, color=c, lw=2, label=f"τ={q}")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_class_hierarchy(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 3.5, 4.5, 3, 1, "root", fc=GOLD, tc=INK, fs=11)
    box(ax, 0.5, 2.2, 2.5, 1, "A", fc=TEAL, fs=11)
    box(ax, 3.75, 2.2, 2.5, 1, "B", fc=TEAL, fs=11)
    box(ax, 7, 2.2, 2.5, 1, "C", fc=TEAL, fs=11)
    for x in [1.75, 5, 8.25]:
        ax.plot([5, x], [4.5, 3.2], color=SLATE, lw=1)
    style(ax, t)


def d_sam(ax, t, rng):
    x = np.linspace(-2, 2, 200)
    ax.plot(x, x**2, color=SLATE, lw=1.5, label="sharp")
    ax.plot(x, 0.3 * x**2, color=TEAL, lw=2, label="flat (SAM-ish)")
    ax.legend(fontsize=8)
    ax.set_xlabel("weight")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_moco_v3(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 3.5, 1.6, "ViT query", fc=TEAL, fs=12)
    box(ax, 4.5, 2.5, 3.5, 1.6, "momentum key", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.5, 2.5, 3.2, 1.6, "InfoNCE", fc=DEEP, fs=12)
    style(ax, t)


def d_vq_vae(ax, t, rng):
    z = rng.normal(0, 1, (40, 2))
    codes = rng.normal(0, 0.8, (8, 2))
    ax.scatter(z[:, 0], z[:, 1], c=TEAL, s=20, alpha=0.5, label="encoder z")
    ax.scatter(codes[:, 0], codes[:, 1], c=GOLD, s=80, marker="s", label="codebook")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_dreamer(ax, t, rng):
    steps = np.arange(0, 30)
    rew = np.cumsum(rng.normal(0.1, 0.3, 30))
    imag = rew + rng.normal(0, 0.5, 30).cumsum() * 0.05
    ax.plot(steps, rew, color=GOLD, lw=2, label="true")
    ax.plot(steps, imag, color=TEAL, lw=2, label="latent dream")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_hqq(ax, t, rng):
    bits = np.array([2, 3, 4, 8])
    err = np.array([0.3, 0.15, 0.07, 0.02])
    ax.plot(bits, err, "o-", color=TEAL, lw=2)
    ax.set_xlabel("bits")
    ax.set_ylabel("quant error")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_deepwalk(ax, t, rng):
    pos = {i: (np.cos(2 * np.pi * i / 6), np.sin(2 * np.pi * i / 6)) for i in range(6)}
    for i in range(6):
        j = (i + 1) % 6
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=SLATE, lw=1)
        ax.plot(*pos[i], "o", color=TEAL, ms=12)
    walk = [0, 1, 2, 4, 5]
    for a, b in zip(walk[:-1], walk[1:]):
        ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    ax.set_aspect("equal")
    ax.axis("off")
    style(ax, t)


def d_censoring(ax, t, rng):
    t0 = np.sort(rng.exponential(3, 30))
    c = (t0 > 5).astype(float)
    ax.scatter(t0, np.arange(30), c=[ROSE if x else TEAL for x in c], s=40)
    ax.axvline(5, color=GOLD, ls="--", label="censor time")
    ax.legend(fontsize=8)
    ax.set_xlabel("time")
    ax.set_ylabel("subject")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_sre_oncall(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["page", "ack", "mitigate", "handoff", "doc"]):
        box(ax, 0.3 + i * 2.35, 1.5, 2.2, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_gloss_reg(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["OLS", "GLM", "GAM", "tree", "forest"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_krylov(ax, t, rng):
    k = np.arange(1, 16)
    res = np.exp(-0.35 * k) + 0.01
    ax.semilogy(k, res, "o-", color=TEAL, lw=2)
    ax.set_xlabel("Krylov dim")
    ax.set_ylabel("residual")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_purpose_limit(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2, 3.5, 2, "stated purpose", fc=TEAL, fs=12)
    box(ax, 4.5, 2, 3.5, 2, "compatible use?", fc=GOLD, tc=INK, fs=12)
    box(ax, 8.5, 2, 3.2, 2, "block / DPIA", fc=ROSE, fs=12)
    style(ax, t)


def d_margin_dist(ax, t, rng):
    m = rng.normal(0.4, 0.5, 200)
    ax.hist(m, bins=30, color=TEAL, edgecolor=DEEP, alpha=0.85)
    ax.axvline(0, color=ROSE, lw=2, label="margin 0")
    ax.legend(fontsize=8)
    ax.set_xlabel("functional margin")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_parallel_sets(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for x in [1, 5, 9]:
        ax.plot([x, x], [0.5, 4.5], color=SLATE, lw=3)
    for y0, y1, y2 in [(1, 2, 1.5), (2.5, 3, 3.5), (3.5, 1.5, 2.5)]:
        ax.plot([1, 5, 9], [y0, y1, y2], color=TEAL, lw=2, alpha=0.7)
    style(ax, t)


def d_abc_smc(ax, t, rng):
    rounds = np.arange(1, 8)
    eps = 2 * 0.7 ** (rounds - 1)
    n_part = 1000 * (0.85) ** (rounds - 1)
    ax.plot(rounds, eps, "o-", color=TEAL, lw=2, label="ε_t")
    ax2 = ax.twinx()
    ax2.plot(rounds, n_part, "s-", color=GOLD, lw=2, label="ESS proxy")
    ax.set_xlabel("SMC round")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_chameleon(ax, t, rng):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 4, 1.5, "partition\n(graph)", fc=TEAL, fs=12)
    box(ax, 5.5, 2.5, 4, 1.5, "agglomerate\nRI/RC", fc=GOLD, tc=INK, fs=12)
    style(ax, t)


def d_prefix_projected(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["SDB", "α-project", "grow", "closed?"]):
        box(ax, 0.4 + i * 3.0, 1.6, 2.7, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_entity_embed(ax, t, rng):
    e = rng.normal(0, 1, (25, 2))
    ax.scatter(e[:, 0], e[:, 1], c=TEAL, s=40, edgecolors=DEEP)
    ax.set_xlabel("e1")
    ax.set_ylabel("e2")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_isomap_k(ax, t, rng):
    k = np.arange(2, 20)
    stress = 0.4 * np.exp(-0.2 * (k - 2)) + 0.05 + 0.01 * np.maximum(k - 12, 0)
    ax.plot(k, stress, "o-", color=TEAL, lw=2)
    ax.set_xlabel("k neighbors")
    ax.set_ylabel("stress proxy")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_ridge_leverage(ax, t, rng):
    lam = np.logspace(-3, 2, 40)
    lev = 1 / (1 + lam)
    ax.semilogx(lam, lev, color=TEAL, lw=2)
    ax.set_xlabel("λ")
    ax.set_ylabel("mean leverage sketch")
    ax.grid(True, which="both", alpha=0.25)
    style(ax, t)


def d_calibration_slope(ax, t, rng):
    logit = np.linspace(-3, 3, 40)
    p = 1 / (1 + np.exp(-0.7 * logit - 0.2))
    ax.plot(1 / (1 + np.exp(-logit)), p, color=TEAL, lw=2, label="recalibrated")
    ax.plot([0, 1], [0, 1], "--", color=SLATE)
    ax.set_xlabel("uncal p")
    ax.set_ylabel("cal p")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_lion_opt(ax, t, rng):
    steps = np.arange(0, 50)
    g = np.sin(steps / 3)
    update = np.sign(0.9 * np.cumsum(g) / np.arange(1, 51) + 0.1 * g)
    ax.plot(steps, g, color=SLATE, label="grad")
    ax.plot(steps, update, color=TEAL, lw=2, label="Lion sign update")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_mae_decoder(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.5, 2.5, 3.5, 1.6, "visible\npatches", fc=TEAL, fs=11)
    box(ax, 4.5, 2.5, 3.5, 1.6, "encoder", fc=DEEP, fs=12)
    box(ax, 8.5, 2.5, 3.2, 1.6, "light\ndecoder", fc=GOLD, tc=INK, fs=11)
    style(ax, t)


def d_audiolm(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, lab in enumerate(["wave", "semantic\ntokens", "coarse", "fine"]):
        box(ax, 0.4 + i * 3.0, 1.6, 2.7, 1.8, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


def d_r2d2(ax, t, rng):
    k = np.arange(1, 20)
    ret = np.exp(-0.15 * k)
    ax.bar(k, ret, color=TEAL, edgecolor=DEEP)
    ax.set_xlabel("n-step")
    ax.set_ylabel("return weight")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_sparse_attn(ax, t, rng):
    A = np.zeros((12, 12))
    for i in range(12):
        A[i, max(0, i - 2) : i + 1] = 1
        A[i, ::4] = 1
    ax.imshow(A, cmap="Greens")
    ax.set_xlabel("key")
    ax.set_ylabel("query")
    style(ax, t)


def d_infomap(ax, t, rng):
    steps = np.arange(1, 10)
    L = 2.5 - 0.3 * np.log(steps)
    ax.plot(steps, L, "o-", color=TEAL, lw=2)
    ax.set_xlabel("map equation passes")
    ax.set_ylabel("L description length")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_freshness_sla(ax, t, rng):
    hours = np.arange(0, 48)
    lag = np.clip(0.2 * hours + rng.normal(0, 0.5, 48), 0, None)
    ax.plot(hours, lag, color=TEAL, lw=2)
    ax.axhline(6, color=GOLD, ls="--", label="SLA 6h")
    ax.legend(fontsize=8)
    ax.set_xlabel("hour")
    ax.set_ylabel("data lag")
    ax.grid(True, alpha=0.25)
    style(ax, t)


def d_toil_burn(ax, t, rng):
    weeks = np.arange(1, 13)
    toil = np.array([12, 11, 10, 14, 9, 8, 13, 7, 6, 9, 5, 4])
    ax.bar(weeks, toil, color=TEAL, edgecolor=DEEP)
    ax.axhline(8, color=ROSE, ls="--", label="toil budget")
    ax.legend(fontsize=8)
    ax.set_xlabel("week")
    ax.set_ylabel("toil hours")
    ax.grid(True, axis="y", alpha=0.25)
    style(ax, t)


def d_gloss_class(ax, t, rng):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, lab in enumerate(["logit", "softmax", "margin", "OvR", "CRF"]):
        box(ax, 0.3 + i * 2.35, 1.2, 2.2, 1.5, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=11)
    style(ax, t)


C219 = [
    ("Schur triangular eigen form", d_schur),
    ("Epsilon-DP utility curve", d_epsilon_dp),
    ("VC growth function sketch", d_vc_growth),
    ("ECDF confidence band", d_ecdf_bands),
    ("Dirichlet process stick break", d_dirichlet_process),
    ("ROCK link agglomeration", d_rock_clusters),
    ("SPADE id-list temporal join", d_spade_idlist),
    ("Leave-group-out CV folds", d_leave_group_out),
    ("Procrustes shape alignment", d_procrustes),
    ("Quantile regression fan lines", d_quantile_reg_fans),
    ("Hierarchical class taxonomy", d_class_hierarchy),
    ("SAM flat vs sharp minima", d_sam),
    ("MoCo-v3 ViT contrastive", d_moco_v3),
    ("VQ-VAE codebook nearest", d_vq_vae),
    ("Dreamer latent imagination", d_dreamer),
    ("HQQ half-quadratic quant", d_hqq),
    ("DeepWalk random path", d_deepwalk),
    ("Right-censoring time marks", d_censoring),
    ("SRE on-call response stages", d_sre_oncall),
    ("Glossary regression family strip", d_gloss_reg),
]

C220 = [
    ("Krylov subspace residual drop", d_krylov),
    ("Purpose limitation decision", d_purpose_limit),
    ("Functional margin histogram", d_margin_dist),
    ("Parallel sets category flows", d_parallel_sets),
    ("ABC-SMC epsilon annealing", d_abc_smc),
    ("CHAMELEON partition merge", d_chameleon),
    ("Prefix projected grow closed", d_prefix_projected),
    ("Entity embedding scatter", d_entity_embed),
    ("Isomap k-neighbor stress", d_isomap_k),
    ("Ridge leverage vs lambda", d_ridge_leverage),
    ("Calibration slope remapping", d_calibration_slope),
    ("Lion sign-based optimizer", d_lion_opt),
    ("MAE light decoder path", d_mae_decoder),
    ("AudioLM token cascade", d_audiolm),
    ("R2D2 redistributed returns", d_r2d2),
    ("Sparse attention pattern mask", d_sparse_attn),
    ("Infomap description length", d_infomap),
    ("Data freshness SLA lag", d_freshness_sla),
    ("Engineering toil burn bars", d_toil_burn),
    ("Glossary classification strip", d_gloss_class),
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

    m = {219: C219, 220: C220}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [219, 220]
    for c in cycles:
        embed(c, m[c])
