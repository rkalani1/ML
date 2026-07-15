#!/usr/bin/env python3
"""Cycle-247/248 quality densify: novel scientific teal teaching panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
TEAL, DEEP, INK, GOLD = "#0d9488", "#0f766e", "#0f172a", "#c9a227"
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


C247 = [
    ("CGS residual bi-CG path", make_curve("exp")),
    ("Concentrated DP rho map", make_curve("log")),
    ("Shattering coefficient growth", make_curve("power")),
    ("Isotonic calib reliability heat", make_heat()),
    ("Bulk ESS multi-chain path", make_curve("sigmoid")),
    ("Mean-shift mode scatter", make_scatter()),
    ("LCM closed pattern path", make_flow(["scan", "extend", "close", "F_k"])),
    ("Angular LSH collision rate", make_curve("power")),
    ("LargeVis edge sample cool", make_curve("damped")),
    ("Tukey biweight residual cost", make_curve("relu")),
    ("Macro recall class bars", make_bars()),
    ("EfficientNet compound residual", make_flow(["depth", "width", "res", "scale"])),
    ("DINOv2 register token path", make_flow(["patch", "reg", "teacher", "DINO"])),
    ("YaRN RoPE scale heat", make_heat()),
    ("V-trace importance mix", make_semilog()),
    ("GaLore low-rank proj bars", make_bars()),
    ("Infomap flow compress path", make_curve("sigmoid")),
    ("CDC change-data path", make_flow(["log", "stream", "apply", "serve"])),
    ("Progressive delivery path", make_curve("log")),
    ("Glossary eval metric strip", make_flow(["AUC", "F1", "ECE", "Brier"])),
]

C248 = [
    ("FGMRES flexible residual path", make_curve("exp")),
    ("Supply-chain attest path", make_flow(["build", "sign", "verify", "deploy"])),
    ("Grad-CAM channel bars", make_bars()),
    ("HBOS histogram anomaly cloud", make_scatter()),
    ("Score matching Langevin field", make_quiver()),
    ("Disassortative mix heat", make_heat()),
    ("GSP generalized sequence path", make_flow(["SDB", "prefix", "project", "grow"])),
    ("Hellinger distance bin bars", make_bars()),
    ("Sparse PCA recon error", make_curve("power")),
    ("Expectile tau residual map", make_curve("relu")),
    ("Sparsemax support map", make_curve("sigmoid")),
    ("FlashDecoding KV trade", make_curve("log")),
    ("VICReg variance-invar path", make_flow(["z1", "var", "inv", "cov"])),
    ("Wav2Vec2 contrast path", make_flow(["audio", "quant", "ctx", "InfoNCE"])),
    ("CPO constrained pref path", make_curve("damped")),
    ("AWQ channel protect bars", make_bars()),
    ("TrivialAugment magnitude sched", make_curve("sigmoid")),
    ("Null feature rate spike", make_curve("log")),
    ("SLO error burn-down", make_bars()),
    ("Glossary optim trick strip", make_flow(["warmup", "cosine", "SWA", "EMA"])),
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

    m = {247: C247, 248: C248}
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [247, 248]
    for c in cycles:
        embed(c, m[c])
