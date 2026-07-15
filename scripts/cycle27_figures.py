#!/usr/bin/env python3
"""Cycle-27 densify — lift remaining 6 floor-18 chapters to >=19 (+extras)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def fig_actor_critic():
    ep = np.arange(0, 80)
    actor = 1 - np.exp(-ep / 30)
    critic = 1 - np.exp(-ep / 18)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(ep, critic, color=TEAL, lw=2.3, label="critic value fit")
    ax.plot(ep, actor, color=GOLD, lw=2.3, label="actor return")
    ax.set_xlabel("episode")
    ax.set_ylabel("normalized metric (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Actor–critic co-training (toy curves)")
    ax.text(0.98, 0.2, "Simulated control ≠ bedside\npolicy license. Pred ≠ cause.", transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_actor_critic.png")


def fig_adapter_params():
    methods = ["full FT", "LoRA r=4", "LoRA r=16", "adapters", "prompt"]
    params = np.array([100, 0.8, 3.2, 2.0, 0.1])
    acc = np.array([0.91, 0.87, 0.89, 0.88, 0.84])
    fig, ax1 = plt.subplots(figsize=(8.0, 4.1))
    x = np.arange(len(methods))
    ax1.bar(x - 0.2, params, width=0.4, color=GOLD, label="% params (rel.)")
    ax2 = ax1.twinx()
    ax2.plot(x, acc, color=TEAL, lw=2.4, marker="o", label="accuracy")
    ax1.set_xticks(x, methods, fontsize=8)
    ax1.set_ylabel("% trainable (relative)")
    ax2.set_ylabel("accuracy (synthetic)")
    ax2.set_ylim(0.8, 0.95)
    style_ax(ax1, "Parameter-efficient fine-tuning trade-offs")
    ax1.text(0.02, 0.9, "Fewer params ≠ less validation.", transform=ax1.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_peft_tradeoff.png")


def fig_shortest_path():
    # grid path schematic
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    for i in range(5):
        for j in range(5):
            ax.add_patch(plt.Circle((i, j), 0.12, color=GRAY))
    # edges light
    for i in range(5):
        for j in range(5):
            if i < 4:
                ax.plot([i, i + 1], [j, j], color="#e2e8f0", lw=1)
            if j < 4:
                ax.plot([i, i], [j, j + 1], color="#e2e8f0", lw=1)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]
    xs, ys = zip(*path)
    ax.plot(xs, ys, color=TEAL, lw=3, marker="o", markersize=8)
    ax.scatter([0], [0], c=GOLD, s=120, zorder=3, label="start")
    ax.scatter([4], [4], c=DEEP, s=120, zorder=3, label="goal")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Shortest path on a grid graph (schematic)", fontsize=12, fontweight="bold", color=INK)
    ax.text(0.5, -0.05, "Path cost is graph geometry—not clinical priority without weights tied to outcomes.", transform=ax.transAxes, ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_shortest_path.png")


def fig_label_timeline():
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axhline(1.5, color=GRAY, lw=1.5)
    events = [(1, "admit"), (3, "labs"), (4.5, "scan"), (6, "decision"), (8, "outcome")]
    for x, lab in events:
        ax.plot(x, 1.5, "o", color=TEAL, ms=10)
        ax.text(x, 1.9, lab, ha="center", fontsize=9, color=INK)
    ax.axvspan(0, 6, color=TEAL, alpha=0.08)
    ax.text(3, 0.6, "features legal at decision time", ha="center", fontsize=9, color=DEEP)
    ax.axvspan(6, 10, color=GOLD, alpha=0.12)
    ax.text(8, 0.6, "post-decision / label window", ha="center", fontsize=9, color=GOLD)
    ax.axis("off")
    ax.set_title("Label and feature timeline hygiene", fontsize=12, fontweight="bold", color=INK)
    ax.text(0.5, 0.05, "Using post-decision features is leakage—not cleverness. Pred ≠ cause.", transform=ax.transAxes, ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_label_timeline.png")


def fig_incident_postmortem():
    stages = ["detect", "page", "mitigate", "root cause", "fix", "retrain?"]
    hours = [0, 0.5, 2, 12, 48, 120]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(hours, np.arange(len(stages)), color=TEAL, lw=2.3, marker="o")
    ax.set_yticks(range(len(stages)), stages)
    ax.set_xlabel("hours since alarm (teaching)")
    style_ax(ax, "Incident response timeline (ops)")
    ax.text(0.98, 0.15, "Speed without learning\nrepeats harm. Fix process,\nnot only the model.", transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_incident_timeline.png")


def fig_sensitivity_spec_vocab():
    # 2x2 confusion teaching with counts
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # boxes
    cells = [
        (1, 5.5, "TP\n82", TEAL),
        (5.5, 5.5, "FP\n11", GOLD),
        (1, 1.5, "FN\n8", GOLD),
        (5.5, 1.5, "TN\n49", TEAL),
    ]
    for x, y, t, c in cells:
        ax.add_patch(plt.Rectangle((x, y), 3.5, 3.2, facecolor=c, alpha=0.35, edgecolor=DEEP, lw=1.5))
        ax.text(x + 1.75, y + 1.6, t, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(5, 9.3, "Predicted +          Predicted −", ha="center", fontsize=11, color=SLATE)
    ax.text(0.3, 7.1, "Actual +", rotation=90, va="center", fontsize=11, color=SLATE)
    ax.text(0.3, 3.1, "Actual −", rotation=90, va="center", fontsize=11, color=SLATE)
    ax.text(5, 0.4, "Se=TP/(TP+FN)  Sp=TN/(TN+FP)  — counts are teaching toy", ha="center", fontsize=8, color=SLATE)
    ax.set_title("Confusion matrix vocabulary (toy counts)", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ml_fig_confusion_vocab.png")


def fig_boosting_stages():
    m = np.arange(1, 40)
    train = np.exp(-m / 12) * 0.8 + 0.05
    test = np.exp(-m / 15) * 0.75 + 0.12 + 0.002 * np.maximum(0, m - 25)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(m, train, color=TEAL, lw=2.3, label="train loss")
    ax.plot(m, test, color=GOLD, lw=2.3, label="test loss")
    ax.axvline(25, color=GRAY, ls="--", lw=1.3, label="early stop zone")
    ax.set_xlabel("boosting rounds")
    ax.set_ylabel("loss (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Boosting stages: fit then overfit")
    ax.text(0.98, 0.55, "More trees ≠ better.\nUse validation. Pred ≠ cause.", transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_boosting_stages.png")


def fig_aleatoric_epistemic():
    x = np.linspace(0, 10, 100)
    mean = np.sin(x / 2)
    ale = 0.15 + 0.05 * np.sin(x)
    epi = 0.05 + 0.25 * (x > 7) * (x - 7) / 3
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.plot(x, mean, color=DEEP, lw=2.2, label="mean pred")
    ax.fill_between(x, mean - ale, mean + ale, color=TEAL, alpha=0.35, label="aleatoric-ish")
    ax.fill_between(x, mean - ale - epi, mean + ale + epi, color=GOLD, alpha=0.2, label="+ epistemic-ish")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    style_ax(ax, "Uncertainty decomposition sketch")
    ax.text(0.02, 0.9, "Bands are teaching cartoons.\nUncertainty ≠ causal ID.", transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_uncertainty_bands.png")


def main() -> None:
    fig_actor_critic()
    fig_adapter_params()
    fig_shortest_path()
    fig_label_timeline()
    fig_incident_postmortem()
    fig_sensitivity_spec_vocab()
    fig_boosting_stages()
    fig_aleatoric_epistemic()
    print("DONE cycle-27")


if __name__ == "__main__":
    main()
