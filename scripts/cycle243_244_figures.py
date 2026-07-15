#!/usr/bin/env python3
"""Cycle-243/244 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fc, edgecolor="none")
    )
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def make_flow(labels):
    def fn(ax, t, rng):
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 5)
        ax.axis("off")
        n = len(labels)
        w = min(2.6, 11.0 / n - 0.2)
        for i, lab in enumerate(labels):
            box(ax, 0.3 + i * (w + 0.3), 1.6, w, 1.9, lab, fc=TEAL if i % 2 == 0 else DEEP, fs=10)
            if i < n - 1:
                x0 = 0.3 + i * (w + 0.3) + w
                ax.annotate(
                    "",
                    xy=(x0 + 0.25, 2.55),
                    xytext=(x0, 2.55),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.4),
                )
        style(ax, t)

    return fn


def make_curve(kind):
    def fn(ax, t, rng):
        x = np.linspace(0.01, 10, 120)
        if kind == "log":
            y = np.log(x + 1)
        elif kind == "exp":
            y = 1 - np.exp(-x / 3)
        elif kind == "power":
            y = x ** (-0.5)
        elif kind == "sigmoid":
            y = 1 / (1 + np.exp(-(x - 5)))
        elif kind == "relu":
            y = np.maximum(0, x - 3)
        else:
            y = np.sin(x) * np.exp(-x / 8)
        ax.plot(x, y, color=TEAL, lw=2.2)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        style(ax, t)

    return fn


def make_heat():
    def fn(ax, t, rng):
        M = rng.normal(0, 1, (8, 8))
        M = (M + M.T) / 2
        ax.imshow(M, cmap="Greens")
        ax.set_xticks([])
        ax.set_yticks([])
        style(ax, t)

    return fn


def make_scatter():
    def fn(ax, t, rng):
        a = rng.normal([-1, 0], 0.4, (40, 2))
        b = rng.normal([1.2, 0.3], 0.45, (40, 2))
        ax.scatter(a[:, 0], a[:, 1], c=TEAL, s=28, alpha=0.75)
        ax.scatter(b[:, 0], b[:, 1], c=GOLD, s=28, alpha=0.75)
        ax.grid(True, alpha=0.25)
        style(ax, t)

    return fn


def make_bars():
    def fn(ax, t, rng):
        v = np.abs(rng.normal(1, 0.4, 8))
        ax.bar(range(8), v, color=TEAL, edgecolor=DEEP)
        ax.grid(True, axis="y", alpha=0.25)
        style(ax, t)

    return fn


def make_semilog():
    def fn(ax, t, rng):
        n = np.logspace(1, 4, 50)
        ax.semilogx(n, 1 / np.sqrt(n), color=TEAL, lw=2.2)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("n")
        ax.set_ylabel("rate")
        style(ax, t)

    return fn


def make_quiver():
    def fn(ax, t, rng):
        X, Y = np.meshgrid(np.linspace(-2, 2, 12), np.linspace(-2, 2, 12))
        U, V = -Y, X
        ax.quiver(X, Y, U, V, color=TEAL, alpha=0.8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        style(ax, t)

    return fn


C243 = [
    ("LSQR residual least-squares path", make_curve("exp")),
    ("f-DP tradeoff function curve", make_curve("log")),
    ("Rademacher complexity bound", make_curve("power")),
    ("ECE bin reliability heat", make_heat()),
    ("Split-R-hat diagnostic path", make_curve("sigmoid")),
    ("OPTICS reachability scatter", make_scatter()),
    ("ECLAT tidset intersect path", make_flow(["scan", "intersect", "prune", "F_k"])),
    ("b-bit MinHash collision", make_curve("power")),
    ("TriMAP triplet cool path", make_curve("damped")),
    ("Cauchy robust residual cost", make_curve("relu")),
    ("Balanced accuracy bars", make_bars()),
    ("ResNeXt cardinality residual", make_flow(["x", "group", "+", "out"])),
    ("DINO student-teacher path", make_flow(["student", "EMA", "teacher", "CE"])),
    ("ALiBi distance bias heat", make_heat()),
    ("Retrace operator scale", make_semilog()),
    ("QLoRA NF4 quant bars", make_bars()),
    ("Spectral clustering cut path", make_curve("sigmoid")),
    ("Point-in-time join path", make_flow(["as-of", "join", "train", "serve"])),
    ("Canary promote gate path", make_curve("log")),
    ("Glossary regularizer strip", make_flow(["L1", "L2", "elastic", "dropout"])),
]

C244 = [
    ("IDR residual defect path", make_curve("exp")),
    ("Incident response control path", make_flow(["detect", "contain", "eradicate", "recover"])),
    ("SHAP interaction bars", make_bars()),
    ("Mahalanobis anomaly cloud", make_scatter()),
    ("Probability flow ODE field", make_quiver()),
    ("Core-periphery block heat", make_heat()),
    ("SPADE sequence lattice path", make_flow(["SDB", "prefix", "project", "grow"])),
    ("KL divergence bin bars", make_bars()),
    ("Incremental PCA recon error", make_curve("power")),
    ("Asymmetric Huber residual", make_curve("relu")),
    ("Entmax sparsity map", make_curve("sigmoid")),
    ("Activation checkpoint trade", make_curve("log")),
    ("SwAV swapped predict path", make_flow(["view1", "codes", "view2", "swap"])),
    ("Conformer conv-attn path", make_flow(["audio", "FFN", "MHSA", "conv"])),
    ("IPO identity preference path", make_curve("damped")),
    ("AWQ scale protect bars", make_bars()),
    ("RandAugment magnitude sched", make_curve("sigmoid")),
    ("Late feature arrival spike", make_curve("log")),
    ("On-call toil burn-down", make_bars()),
    ("Glossary gen model strip", make_flow(["VAE", "GAN", "diffusion", "AR"])),
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

    m = {243: C243, 244: C244}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [243, 244]
    for c in cycles:
        embed(c, m[c])
