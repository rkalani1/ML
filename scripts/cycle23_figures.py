#!/usr/bin/env python3
"""Cycle-23 densify — lift remaining 7 floor-16 chapters to >=17 (+ extras)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, DEEP, INK, GOLD, GRAY, SLATE = (
    "#0d9488",
    "#0f766e",
    "#0f172a",
    "#c9a227",
    "#94a3b8",
    "#64748b",
)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", name)


def style_ax(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def fig_reader_path_map():
    """Preface: how to route reading by goal."""
    goals = ["Rebuild math", "Ship a model", "Audit a paper", "Deploy safely"]
    paths = [
        "00 → 03 → 08",
        "01 → 06 → 09 → 16",
        "17 + 18 + metrics",
        "16 → 17 → monitor",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for i, (g, p) in enumerate(zip(goals, paths)):
        y = 4.2 - i * 1.0
        ax.add_patch(
            FancyBboxPatch(
                (0.4, y - 0.35),
                3.2,
                0.7,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor="#ecfeff",
                edgecolor=TEAL,
                lw=1.5,
            )
        )
        ax.text(2.0, y, g, ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.annotate(
            "",
            xy=(7.2, y),
            xytext=(3.8, y),
            arrowprops=dict(arrowstyle="->", color=DEEP, lw=1.6),
        )
        ax.add_patch(
            FancyBboxPatch(
                (7.2, y - 0.35),
                2.4,
                0.7,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor="white",
                edgecolor=GOLD,
                lw=1.4,
            )
        )
        ax.text(8.4, y, p, ha="center", va="center", fontsize=9, color=SLATE)
    ax.set_title("Reader path map by goal (teaching)", fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.text(
        0.5,
        0.15,
        "Routes are study aids—not licenses for causal claims from predictive scores.",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_reader_paths.png")


def fig_q_learning_update():
    """Ch13: Q-learning backup diagram-like curves."""
    episodes = np.arange(0, 120)
    q_opt = 1.0
    q_on = q_opt * (1 - np.exp(-episodes / 35))
    q_off = q_opt * (1 - np.exp(-episodes / 22)) - 0.05 * np.sin(episodes / 8) * np.exp(-episodes / 50)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(episodes, q_on, color=TEAL, lw=2.3, label="on-policy (SARSA-like)")
    ax.plot(episodes, q_off, color=GOLD, lw=2.3, label="off-policy (Q-learning-like)")
    ax.axhline(q_opt, color=GRAY, ls="--", lw=1.3, label="optimal Q*")
    ax.set_xlabel("episodes")
    ax.set_ylabel("max_a Q(s0,a) (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "TD control learning curves (toy MDP)")
    ax.text(
        0.98,
        0.2,
        "Simulated returns ≠ license\nto automate care.\nPred/control ≠ proven causation.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_td_control_curves.png")


def fig_structured_prune():
    """Ch14: structured vs unstructured sparsity schematic."""
    rng = np.random.default_rng(2)
    W = rng.normal(size=(12, 16))
    # unstructured: small mag zeroed
    Wu = W.copy()
    thr = np.quantile(np.abs(Wu), 0.7)
    Wu[np.abs(Wu) < thr] = 0
    # structured: drop channels (columns)
    Ws = W.copy()
    drop = [1, 4, 7, 10, 13]
    Ws[:, drop] = 0
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].imshow(Wu, cmap="RdYlBu", aspect="auto", vmin=-2, vmax=2)
    style_ax(axes[0], "Unstructured magnitude prune")
    axes[0].set_xlabel("weight index")
    axes[0].set_ylabel("out channel")
    axes[1].imshow(Ws, cmap="RdYlBu", aspect="auto", vmin=-2, vmax=2)
    style_ax(axes[1], "Structured channel prune")
    axes[1].set_xlabel("weight index")
    fig.suptitle("Sparsity patterns for lighter models (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.04)
    fig.text(0.5, -0.03, "Sparsity changes compute; recalibrate—compression ≠ new clinical truth.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_structured_prune.png")


def fig_community_modularity_curve():
    """Ch15: modularity Q vs number of communities k."""
    k = np.arange(1, 12)
    # synthetic Q peaks mid
    Q = 0.05 + 0.55 * np.exp(-0.5 * ((k - 4) / 2.2) ** 2) - 0.01 * k
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(k, Q, color=TEAL, lw=2.4, marker="o")
    ax.axvline(4, color=GOLD, ls="--", lw=1.4, label="peak k (teaching)")
    ax.set_xlabel("number of communities k")
    ax.set_ylabel("modularity Q (synthetic)")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Modularity vs resolution (synthetic)")
    ax.text(
        0.98,
        0.85,
        "Resolution limits apply.\nCommunities ≠ disease entities.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_modularity_vs_k.png")


def fig_missingness_mech():
    """Ch16: MCAR / MAR / MNAR schematic panels."""
    rng = np.random.default_rng(5)
    n = 200
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(scale=0.7, size=n)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    # MCAR
    m = rng.random(n) < 0.25
    axes[0].scatter(x[~m], y[~m], c=TEAL, s=14, alpha=0.75)
    axes[0].scatter(x[m], y[m], c=GOLD, s=18, marker="x", label="missing y")
    style_ax(axes[0], "MCAR (missing ⊥ data)")
    axes[0].legend(frameon=False, fontsize=7)
    # MAR: missing depends on x
    m = x > 0.8
    axes[1].scatter(x[~m], y[~m], c=TEAL, s=14, alpha=0.75)
    axes[1].scatter(x[m], y[m], c=GOLD, s=18, marker="x")
    style_ax(axes[1], "MAR (depends on observed x)")
    # MNAR: depends on y
    m = y > 1.0
    axes[2].scatter(x[~m], y[~m], c=TEAL, s=14, alpha=0.75)
    axes[2].scatter(x[m], y[m], c=GOLD, s=18, marker="x")
    style_ax(axes[2], "MNAR (depends on y itself)")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Missingness mechanisms (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.05)
    fig.text(0.5, -0.04, "Imputation assumptions can invent associations—not causal proof.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_missingness_mechanisms.png")


def fig_slice_eval_matrix():
    """Ch17: subgroup x metric slice evaluation heatmap."""
    groups = ["overall", "age<60", "age>=60", "site A", "site B", "transfer", "direct"]
    metrics = ["AUROC", "AUPRC", "ECE↓", "Brier↓"]
    M = np.array(
        [
            [0.86, 0.42, 0.04, 0.14],
            [0.88, 0.38, 0.05, 0.13],
            [0.81, 0.45, 0.07, 0.16],
            [0.87, 0.40, 0.03, 0.12],
            [0.79, 0.33, 0.09, 0.18],
            [0.82, 0.48, 0.06, 0.15],
            [0.85, 0.39, 0.04, 0.14],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_yticks(range(len(groups)), groups)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=INK if M[i, j] < 0.7 else "white",
            )
    style_ax(ax, "Slice evaluation matrix (synthetic)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.text(
        0.0,
        -0.18,
        "Worst slices gate deploy—not overall AUROC alone. Pred ≠ cause.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_slice_eval_matrix.png")


def fig_metric_cheat_sheet():
    """Ch18: compact metric→question mapping bars."""
    metrics = ["AUROC", "AUPRC", "Brier", "ECE", "Net benefit", "C-index"]
    questions = [
        "ranking overall",
        "ranking @ rare +",
        "prob accuracy",
        "calib. gap",
        "decision value",
        "ranking + censor",
    ]
    scores = [0.9, 0.85, 0.7, 0.65, 0.8, 0.75]  # "fit for purpose" teaching
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    y = np.arange(len(metrics))
    ax.barh(y, scores, color=TEAL, edgecolor="white", height=0.7)
    ax.set_yticks(y, [f"{m} — {q}" for m, q in zip(metrics, questions)], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("teaching fitness for matching claim (not real data)")
    style_ax(ax, "Metric cheat sheet: match claim → metric")
    ax.text(
        0.98,
        0.08,
        "Wrong metric → wrong claim.\nNone of these alone prove causation.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_metric_cheatsheet.png")


def fig_class_prior_shift():
    """Ch09 extra: prior shift moves operating point."""
    thr = np.linspace(0, 1, 100)
    # synthetic sens/spec trade
    sens = 1 / (1 + np.exp(12 * (thr - 0.45)))
    spec = 1 / (1 + np.exp(-12 * (thr - 0.45)))
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    for prev, c, ls in [(0.05, GRAY, "--"), (0.15, GOLD, "-."), (0.35, TEAL, "-")]:
        ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev) + 1e-12)
        ax.plot(thr, ppv, color=c, lw=2.2, ls=ls, label=f"π={prev}")
    ax.set_xlabel("threshold on score")
    ax.set_ylabel("PPV")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, title="prevalence")
    style_ax(ax, "Prior shift: same Se/Sp, different PPV")
    ax.text(
        0.02,
        0.9,
        "Transport thresholds carefully.\nPPV is not a causal effect.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_prior_shift_ppv.png")


def fig_label_noise_curve():
    """Ch16 extra or ch01: accuracy vs label noise rate."""
    noise = np.linspace(0, 0.4, 9)
    acc_clean_cap = 0.92
    acc = acc_clean_cap - 1.1 * noise + 0.4 * noise**2
    acc = np.clip(acc, 0.5, 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(noise, acc, color=TEAL, lw=2.5, marker="o")
    ax.axhline(0.5, color=GRAY, ls="--", lw=1.2, label="chance (balanced)")
    ax.set_xlabel("label flip rate")
    ax.set_ylabel("held-out accuracy (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Label noise erodes achievable accuracy")
    ax.text(
        0.98,
        0.85,
        "Noisy labels bias learning.\nNot the same as causal\nmeasurement error design.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_label_noise_acc.png")


def main() -> None:
    fig_reader_path_map()
    fig_q_learning_update()
    fig_structured_prune()
    fig_community_modularity_curve()
    fig_missingness_mech()
    fig_slice_eval_matrix()
    fig_metric_cheat_sheet()
    fig_class_prior_shift()
    fig_label_noise_curve()
    print("DONE cycle-23")


if __name__ == "__main__":
    main()
