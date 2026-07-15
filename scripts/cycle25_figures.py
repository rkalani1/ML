#!/usr/bin/env python3
"""Cycle-25 densify — lift remaining 6 floor-17 chapters to >=18 (+extras)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

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


def fig_reward_shaping():
    """Ch13: sparse vs shaped reward learning speed."""
    ep = np.arange(0, 100)
    sparse = 1 - np.exp(-ep / 55)
    shaped = 1 - np.exp(-ep / 18)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(ep, shaped, color=TEAL, lw=2.4, label="shaped reward (faster)")
    ax.plot(ep, sparse, color=GOLD, lw=2.4, label="sparse terminal reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("normalized return (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Reward shaping speeds learning—and can hack")
    ax.text(
        0.98,
        0.2,
        "Shaping can create loopholes.\nSim return ≠ safe clinical policy.\nPred/control ≠ causation.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_reward_shaping.png")


def fig_knowledge_distill_gap():
    """Ch14: teacher-student accuracy vs size."""
    size = np.array([1, 2, 4, 8, 16, 32])
    teacher = np.full_like(size, 0.91, dtype=float)
    student = 0.70 + 0.18 * (1 - np.exp(-size / 6))
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(size, teacher, color=GOLD, lw=2.2, ls="--", label="teacher")
    ax.plot(size, student, color=TEAL, lw=2.4, marker="o", label="distilled student")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("relative student size")
    ax.set_ylabel("accuracy (synthetic)")
    ax.set_ylim(0.65, 1.0)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Distillation closes much of the teacher gap")
    ax.text(
        0.02,
        0.15,
        "Smaller ≠ less careful validation.\nRecalibrate after distill.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_distill_gap.png")


def fig_triadic_closure():
    """Ch15: triadic closure probability sketch."""
    # open triad count vs closed over time
    t = np.arange(0, 20)
    open_t = 50 * np.exp(-t / 8) + 5
    closed = 5 + 40 * (1 - np.exp(-t / 7))
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(t, open_t, color=GOLD, lw=2.3, label="open triads")
    ax.plot(t, closed, color=TEAL, lw=2.3, label="closed triangles")
    ax.set_xlabel("time (synthetic dynamics)")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Triadic closure dynamics (cartoon)")
    ax.text(
        0.98,
        0.55,
        "Network growth models ≠\nclinical referral causation\nwithout design.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_triadic_closure.png")


def fig_drift_monitor_panels():
    """Ch16: feature PSI and performance drift dual panel."""
    weeks = np.arange(0, 26)
    psi = 0.05 + 0.002 * weeks + 0.08 * (weeks > 16) * (weeks - 16) / 10
    auroc = 0.86 - 0.001 * weeks - 0.04 * (weeks > 16) * (weeks - 16) / 10
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].plot(weeks, psi, color=GOLD, lw=2.3)
    axes[0].axhline(0.2, color=GRAY, ls="--", lw=1.3, label="PSI alert")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_xlabel("week")
    axes[0].set_ylabel("PSI (synthetic)")
    style_ax(axes[0], "Input drift (PSI)")
    axes[1].plot(weeks, auroc, color=TEAL, lw=2.3)
    axes[1].axhline(0.80, color=GOLD, ls="--", lw=1.3, label="rollback floor")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("week")
    axes[1].set_ylabel("AUROC")
    style_ax(axes[1], "Performance drift")
    fig.suptitle("Monitor inputs and outcomes (synthetic; original)", color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.5, -0.03, "Drift alarms trigger investigation—not automatic causal blame.", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    save(fig, "ml_fig_drift_dual.png")


def fig_equity_gaps():
    """Ch17: subgroup metric gap bars."""
    groups = ["overall", "lang EN", "lang other", "insured", "underinsured", "site rural"]
    auroc = np.array([0.86, 0.87, 0.79, 0.86, 0.80, 0.78])
    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    colors = [TEAL if a >= 0.84 else GOLD for a in auroc]
    ax.barh(groups[::-1], auroc[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0.84, color=DEEP, ls="--", lw=1.3, label="equity floor (teaching)")
    ax.set_xlabel("AUROC (synthetic)")
    ax.set_xlim(0.7, 0.95)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Equity slice gaps vs overall")
    ax.text(
        0.98,
        0.12,
        "Overall hides failed slices.\nGaps need design fixes—not\nblame-the-metric theater.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_equity_gaps.png")


def fig_calibration_slope_int():
    """Ch18: calibration slope & intercept teaching."""
    rng = np.random.default_rng(3)
    p = np.linspace(0.05, 0.95, 12)
    # miscalibrated: slope<1, intercept>0
    obs = 0.08 + 0.75 * p + rng.normal(0, 0.02, size=p.shape)
    obs = np.clip(obs, 0, 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1.4, label="perfect")
    ax.scatter(p, obs, s=55, c=TEAL, zorder=3, label="binned reliability")
    # fit line
    X = np.column_stack([np.ones_like(p), p])
    beta, *_ = np.linalg.lstsq(X, obs, rcond=None)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, beta[0] + beta[1] * xs, color=GOLD, lw=2.2, label=f"slope={beta[1]:.2f}, int={beta[0]:.2f}")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Calibration slope & intercept")
    ax.text(
        0.02,
        0.9,
        "Slope≠1 → miscalibration.\nRecalibrate before counseling.\nNot a causal effect.",
        transform=ax.transAxes,
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_calib_slope_int.png")


def fig_hinge_vs_logloss():
    """Extra ch09: hinge vs logistic loss shapes."""
    z = np.linspace(-2, 2, 200)
    logloss = np.log1p(np.exp(-z))
    hinge = np.maximum(0, 1 - z)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(z, logloss, color=TEAL, lw=2.4, label="logistic loss")
    ax.plot(z, hinge, color=GOLD, lw=2.4, label="hinge loss")
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("margin y·f(x)")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "Classification surrogate losses")
    ax.text(
        0.98,
        0.85,
        "Surrogates enable optimization.\nThey are not utilities or\ncausal contrasts.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_hinge_logloss.png")


def fig_dataset_shift_types():
    """Extra: covariate vs label vs concept shift cartoon bars."""
    types = ["covariate\nshift", "label\nshift", "concept\nshift"]
    severity = [0.7, 0.55, 0.9]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(types, severity, color=[TEAL, GOLD, DEEP], edgecolor="white")
    ax.set_ylabel("transport difficulty (teaching)")
    ax.set_ylim(0, 1.1)
    style_ax(ax, "Dataset shift taxonomy (severity sketch)")
    ax.text(
        0.5,
        0.05,
        "Concept shift breaks P(y|x)—hardest. Shift ≠ moral failure of a site.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color=SLATE,
    )
    fig.tight_layout()
    save(fig, "ml_fig_shift_taxonomy.png")


def main() -> None:
    fig_reward_shaping()
    fig_knowledge_distill_gap()
    fig_triadic_closure()
    fig_drift_monitor_panels()
    fig_equity_gaps()
    fig_calibration_slope_int()
    fig_hinge_vs_logloss()
    fig_dataset_shift_types()
    print("DONE cycle-25")


if __name__ == "__main__":
    main()
