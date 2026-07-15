#!/usr/bin/env python3
"""Cycle-13 densify figures for ML ebook (teal brand; original scientific panels)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

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


def fig_nested_fs_cv():
    """Ch06: nested CV for feature selection — outer vs inner loops."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

    # Left: diagram of nested CV
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    # Outer folds
    for i, lab in enumerate(["Outer 1", "Outer 2", "Outer 3"]):
        y = 6.2 - i * 1.9
        ax.add_patch(FancyBboxPatch((0.4, y), 11.2, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=SOFT, edgecolor=TEAL, linewidth=1.5))
        # train blocks
        ax.add_patch(Rectangle((0.6, y + 0.25), 7.4, 1.0, facecolor=TEAL, edgecolor="none", alpha=0.85))
        ax.text(4.3, y + 0.75, "OUTER TRAIN\n(inner CV + FS)", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")
        # holdout
        ax.add_patch(Rectangle((8.2, y + 0.25), 3.2, 1.0, facecolor=GOLD, edgecolor="none", alpha=0.9))
        ax.text(9.8, y + 0.75, "OUTER\nTEST", ha="center", va="center",
                color=INK, fontsize=8, fontweight="bold")
        ax.text(0.2, y + 0.75, lab, ha="right", va="center", fontsize=8, color=DEEP, fontweight="bold",
                rotation=90)
    ax.text(6.0, 7.6, "Nested CV: selection never sees outer test", ha="center",
            fontsize=11, fontweight="bold", color=INK)
    ax.text(6.0, 0.35, "Inner loop chooses k / subset; outer reports honest score",
            ha="center", fontsize=8, color=SLATE)
    style_ax(ax, "Nested structure (schematic)")
    ax.set_title("Nested structure (schematic)", fontsize=12, fontweight="bold", color=INK)

    # Right: optimistic vs nested scores
    ax = axes[1]
    rng = np.random.default_rng(42)
    methods = ["Filter\n(no nest)", "Wrapper\n(inner only)", "Nested\nwrapper", "L1 embedded\nnested"]
    # synthetic AUCs: optimistic left, honest right
    opt = np.array([0.91, 0.93, 0.84, 0.83])
    hon = np.array([0.78, 0.76, 0.81, 0.80])
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w / 2, opt, w, color=GRAY, label="reported (leaky / optimistic)", edgecolor="white")
    ax.bar(x + w / 2, hon, w, color=TEAL, label="outer-fold honest", edgecolor="white")
    ax.set_xticks(x, methods, fontsize=8)
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("AUROC (synthetic stroke registry)")
    ax.axhline(0.80, color=GOLD, ls="--", lw=1.2, label="approx true skill")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    style_ax(ax, "Optimism gap without nesting")
    ax.text(0.02, 0.05, "Selecting features on the same\nfolds you score = double-dipping.",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    fig.suptitle("Feature selection must sit inside nested CV (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_nested_fs_cv.png")


def fig_attention_temperature():
    """Ch12: softmax temperature sharpens/softens attention."""
    rng = np.random.default_rng(7)
    # fixed logits for 8 keys
    logits = np.array([2.1, 1.4, 0.8, 0.3, -0.2, -0.5, -1.0, -1.4])
    temps = [0.3, 1.0, 2.5]
    colors = [DEEP, TEAL, GOLD]
    labels = [r"$T=0.3$ (sharp)", r"$T=1.0$ (std)", r"$T=2.5$ (soft)"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    x = np.arange(len(logits))
    width = 0.25
    for i, (T, c, lab) in enumerate(zip(temps, colors, labels)):
        p = np.exp(logits / T)
        p = p / p.sum()
        ax.bar(x + (i - 1) * width, p, width, color=c, edgecolor="white", label=lab)
    ax.set_xlabel("key index")
    ax.set_ylabel("attention weight α")
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, r"Softmax($\ell / T$) on fixed logits")

    ax = axes[1]
    # entropy vs T
    Ts = np.linspace(0.15, 4.0, 80)
    ents = []
    max_w = []
    for T in Ts:
        p = np.exp(logits / T)
        p = p / p.sum()
        ents.append(-(p * np.log(p + 1e-12)).sum())
        max_w.append(p.max())
    ax.plot(Ts, ents, color=TEAL, lw=2.4, label="entropy H(α)")
    ax2 = ax.twinx()
    ax2.plot(Ts, max_w, color=GOLD, lw=2.2, ls="--", label="max α")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("entropy (nats)", color=TEAL)
    ax2.set_ylabel("peak weight", color=GOLD)
    ax.axvline(1.0, color=GRAY, ls=":", lw=1.2)
    ax.text(1.05, 0.2, "T=1", fontsize=8, color=SLATE, transform=ax.get_xaxis_transform())
    # combined legend
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, fontsize=8, loc="center right")
    style_ax(ax, "Low T → sparse focus; high T → diffuse")
    ax.text(0.98, 0.08, "Sampling temperature ≠ training\nsoftmax scale; attention maps\nare not causal attributions.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Attention / sampling temperature sharpens or softens mass (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_attention_temperature.png")


def fig_ppo_clip():
    """Ch13: PPO clipped surrogate objective vs ratio r."""
    eps = 0.2
    r = np.linspace(0.4, 1.8, 400)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))

    # Positive advantage
    ax = axes[0]
    A = 1.0
    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    ppo = np.minimum(unclipped, clipped)
    ax.plot(r, unclipped, color=GRAY, lw=1.6, ls="--", label=r"$r\cdot A$ (unclipped)")
    ax.plot(r, clipped, color=GOLD, lw=1.6, ls=":", label=r"clip$(r)\cdot A$")
    ax.plot(r, ppo, color=TEAL, lw=2.6, label=r"PPO min$(rA,$ clip$\cdot A)$")
    ax.axvspan(1 - eps, 1 + eps, color=TEAL, alpha=0.08)
    ax.axvline(1 - eps, color=DEEP, ls=":", lw=1.0)
    ax.axvline(1 + eps, color=DEEP, ls=":", lw=1.0)
    ax.set_xlabel(r"probability ratio $r=\pi/\pi_{old}$")
    ax.set_ylabel("surrogate term")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, rf"Advantage $A>0$  (ε={eps})")
    ax.text(0.98, 0.12, "No gain for pushing r ≫ 1+ε\nwhen A>0 (prevents huge steps)",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)

    # Negative advantage
    ax = axes[1]
    A = -1.0
    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    ppo = np.minimum(unclipped, clipped)
    ax.plot(r, unclipped, color=GRAY, lw=1.6, ls="--", label=r"$r\cdot A$")
    ax.plot(r, clipped, color=GOLD, lw=1.6, ls=":", label=r"clip$(r)\cdot A$")
    ax.plot(r, ppo, color=TEAL, lw=2.6, label="PPO objective term")
    ax.axvspan(1 - eps, 1 + eps, color=TEAL, alpha=0.08)
    ax.axvline(1 - eps, color=DEEP, ls=":", lw=1.0)
    ax.axvline(1 + eps, color=DEEP, ls=":", lw=1.0)
    ax.set_xlabel(r"probability ratio $r=\pi/\pi_{old}$")
    ax.set_ylabel("surrogate term")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, rf"Advantage $A<0$  (ε={eps})")
    ax.text(0.02, 0.12, "When A<0, clipping stops\ncollapse of π(a|s) toward 0\nfrom one bad batch.",
            transform=ax.transAxes, ha="left", fontsize=8, color=SLATE)
    fig.suptitle("PPO clip: trust-region surrogate vs probability ratio (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_ppo_clip.png")


def fig_km_censor():
    """Ch03: Kaplan–Meier with right-censor marks."""
    # Event times and censoring for two groups
    # Group A (treatment-ish): longer survival
    # Group B: shorter
    # Build KM by hand
    def km_curve(times, events):
        # times: event or censor times; events: 1=event, 0=censor
        order = np.argsort(times)
        t = times[order]
        e = events[order]
        n = len(t)
        at_risk = n
        xs = [0.0]
        ys = [1.0]
        S = 1.0
        cens_t, cens_s = [], []
        i = 0
        while i < n:
            ti = t[i]
            # count events and censors at ti
            d = 0
            c = 0
            j = i
            while j < n and t[j] == ti:
                if e[j] == 1:
                    d += 1
                else:
                    c += 1
                j += 1
            if d > 0:
                S = S * (1 - d / at_risk)
                xs.append(ti)
                ys.append(S)
            # censors at this time (after event jump convention)
            for _ in range(c):
                cens_t.append(ti)
                cens_s.append(S)
            at_risk -= (d + c)
            i = j
        return np.array(xs), np.array(ys), np.array(cens_t), np.array(cens_s)

    # Synthetic times
    rng = np.random.default_rng(11)
    # Group A
    ta = np.array([30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 400, 450,
                   50, 100, 200, 280, 350], dtype=float)
    ea = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                   0, 1, 0, 1, 0])
    # Group B
    tb = np.array([20, 35, 40, 55, 70, 85, 100, 110, 130, 145, 160, 175, 190, 220, 250,
                   25, 65, 95, 140, 200], dtype=float)
    eb = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
                   1, 0, 1, 0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    for times, events, col, lab in [
        (ta, ea, TEAL, "Group A (n=20)"),
        (tb, eb, GOLD, "Group B (n=20)"),
    ]:
        xs, ys, ct, cs = km_curve(times, events)
        # step plot
        ax.step(xs, ys, where="post", color=col, lw=2.3, label=lab)
        ax.plot(ct, cs, marker="+", linestyle="none", color=col, markersize=10, markeredgewidth=1.6)
    ax.set_xlabel("days since index")
    ax.set_ylabel("Ŝ(t) Kaplan–Meier")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 480)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "KM with + right-censor marks")
    ax.text(0.02, 0.08, "Do not drop censored cases.\nCensoring is data, not missing noise.",
            transform=ax.transAxes, fontsize=8, color=SLATE)

    ax = axes[1]
    # At-risk table style bars at selected times
    times_grid = np.array([0, 90, 180, 270, 360])
    def at_risk(times, t):
        return (times >= t).sum()
    width = 12
    for offset, times, col, lab in [
        (-width / 2, ta, TEAL, "A"),
        (width / 2, tb, GOLD, "B"),
    ]:
        counts = [at_risk(times, t) for t in times_grid]
        ax.bar(times_grid + offset, counts, width=width, color=col, edgecolor="white", label=f"Group {lab}")
    ax.set_xlabel("days")
    ax.set_ylabel("n still at risk")
    ax.set_xticks(times_grid)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "At-risk counts underwrite the curve")
    ax.text(0.98, 0.92, "Late KM steps rest on few patients.\nPrediction of risk ≠ causal effect\nof group membership.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Kaplan–Meier survival with censor marks (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_km_censor.png")


def fig_shap_collinearity():
    """Ch06/09: SHAP / importance under collinearity is unstable."""
    rng = np.random.default_rng(5)
    n = 400
    # Two highly correlated labs + one independent
    z = rng.normal(size=n)
    x1 = z + rng.normal(0, 0.15, size=n)  # NIHSS-ish
    x2 = z + rng.normal(0, 0.15, size=n)  # correlated severity score
    x3 = rng.normal(size=n)               # age residual
    # True linear model uses only the latent z + x3
    logit = 1.2 * z + 0.6 * x3
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    # Fit OLS on (x1,x2,x3) — coefficients unstable under collinearity
    X = np.c_[np.ones(n), x1, x2, x3]
    beta, *_ = np.linalg.lstsq(X, y.astype(float), rcond=None)

    # Bootstrap coefficient absolute "importance"
    B = 80
    abs_coefs = np.zeros((B, 3))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        bb, *_ = np.linalg.lstsq(X[idx], y[idx].astype(float), rcond=None)
        abs_coefs[b] = np.abs(bb[1:])

    # Correlation matrix
    C = np.corrcoef(np.c_[x1, x2, x3].T)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    im = ax.imshow(C, cmap="YlGnBu", vmin=-1, vmax=1)
    names = ["x1 severity", "x2 correlated", "x3 other"]
    ax.set_xticks([0, 1, 2], names, fontsize=8, rotation=15)
    ax.set_yticks([0, 1, 2], names, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(C[i, j]) > 0.5 else INK, fontsize=10, fontweight="bold")
    style_ax(ax, "Feature correlation (synthetic labs)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    parts = ax.violinplot([abs_coefs[:, i] for i in range(3)], positions=[1, 2, 3],
                          showmeans=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(TEAL)
        b.set_alpha(0.75)
    parts["cmeans"].set_color(DEEP)
    ax.set_xticks([1, 2, 3], names, fontsize=8)
    ax.set_ylabel("|coefficient| across bootstrap")
    style_ax(ax, "Attribution mass swaps under collinearity")
    ax.text(0.5, 0.95, "True signal lives in shared latent z;\nx1/x2 fight for credit each resample.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=SLATE)
    ax.text(0.98, 0.08, "SHAP/importance ≠ causation.\nCorrelated features → unstable shares.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Collinearity caution for SHAP / coefficient importance (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_shap_collinearity.png")


def fig_mixed_precision():
    """Ch14: FP32 vs FP16 dynamic range and training sketch."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))

    ax = axes[0]
    # log-scale magnitude bands
    # FP16: ~5.96e-8 (subnormal) to 65504; FP32 much wider
    formats = ["FP32", "FP16", "BF16"]
    # approximate max exponent range in log10
    max_exp = [38, 4.5, 38]  # log10 of max finite
    min_norm = [-38, -4.5, -38]
    # significand bits (approx precision digits)
    digits = [7.2, 3.3, 2.4]
    x = np.arange(len(formats))
    ax.bar(x, max_exp, color=TEAL, edgecolor="white", label="≈ log10(max finite)")
    ax.bar(x, min_norm, color=GOLD, edgecolor="white", alpha=0.85, label="≈ log10(min normal)")
    for i, d in enumerate(digits):
        ax.text(i, max_exp[i] + 1.5, f"~{d:.1f} dec.\ndigits", ha="center", fontsize=8, color=DEEP)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, formats)
    ax.set_ylabel("decimal orders of magnitude")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_ax(ax, "Dynamic range vs mantissa precision")
    ax.text(0.02, 0.05, "FP16: small dynamic range → overflow.\nBF16: FP32 range, coarser mantissa.",
            transform=ax.transAxes, fontsize=8, color=SLATE)

    ax = axes[1]
    # Loss curves: FP16 without loss scale diverges; with loss scale tracks FP32
    steps = np.arange(0, 120)
    rng = np.random.default_rng(2)
    loss_fp32 = 1.8 * np.exp(-steps / 35) + 0.25 + rng.normal(0, 0.02, size=len(steps))
    loss_fp16_bad = loss_fp32.copy()
    loss_fp16_bad[40:] = loss_fp16_bad[39] + np.cumsum(rng.normal(0.03, 0.04, size=len(steps) - 40))
    loss_fp16_bad = np.clip(loss_fp16_bad, 0, 5)
    loss_fp16_ok = 1.8 * np.exp(-steps / 34) + 0.27 + rng.normal(0, 0.025, size=len(steps))
    ax.plot(steps, loss_fp32, color=DEEP, lw=2.2, label="FP32 reference")
    ax.plot(steps, loss_fp16_bad, color="#ef4444", lw=1.8, ls="--", label="FP16 no loss-scale (overflow)")
    ax.plot(steps, loss_fp16_ok, color=TEAL, lw=2.2, label="FP16 + loss scaling / BF16")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (synthetic)")
    ax.set_ylim(0, 3.5)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Why loss scaling exists")
    ax.text(0.98, 0.9, "Keep master weights in FP32.\nValidate clinical metrics after AMP—\nnot only training loss.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SLATE)
    fig.suptitle("Mixed precision: range, precision, and loss scaling (original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_mixed_precision.png")


def fig_hamming_multilabel():
    """Ch09: multi-label Hamming loss vs exact-match / micro-F1."""
    rng = np.random.default_rng(9)
    # 5 labels, synthetic predictions at different thresholds
    n, L = 200, 5
    # true labels with correlation
    z = rng.normal(size=(n, 2))
    W = rng.normal(size=(2, L))
    logits = z @ W + rng.normal(0, 0.3, size=(n, L))
    y = (logits > 0).astype(int)
    # model scores = logits + noise
    s = logits + rng.normal(0, 0.5, size=(n, L))
    thresholds = np.linspace(-1.5, 1.5, 25)

    hamming = []
    exact = []
    micro_f1 = []
    for t in thresholds:
        yhat = (s >= t).astype(int)
        # Hamming = fraction of wrong labels
        hamming.append(1 - (yhat == y).mean())
        exact.append((yhat == y).all(axis=1).mean())
        # micro F1
        tp = ((yhat == 1) & (y == 1)).sum()
        fp = ((yhat == 1) & (y == 0)).sum()
        fn = ((yhat == 0) & (y == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        micro_f1.append(2 * prec * rec / (prec + rec + 1e-9))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    ax.plot(thresholds, hamming, color=TEAL, lw=2.4, label="Hamming loss")
    ax.plot(thresholds, 1 - np.array(exact), color=GOLD, lw=2.2, label="1 − exact match")
    ax.plot(thresholds, 1 - np.array(micro_f1), color=DEEP, lw=2.0, ls="--", label="1 − micro-F1")
    ax.set_xlabel("decision threshold on score")
    ax.set_ylabel("error rate")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Multi-label metrics disagree")
    ax.text(0.02, 0.08, "Exact match is harsh when L is large.\nHamming averages per-label errors.",
            transform=ax.transAxes, fontsize=8, color=SLATE)

    ax = axes[1]
    # Confusion-style per-label accuracy at best Hamming threshold
    t_star = thresholds[int(np.argmin(hamming))]
    yhat = (s >= t_star).astype(int)
    per_label = (yhat == y).mean(axis=0)
    label_names = ["AF", "LVO", "HT risk", "Dysphagia", "Palliative"]
    ax.barh(label_names, per_label, color=TEAL, edgecolor="white")
    ax.axvline(per_label.mean(), color=GOLD, ls="--", lw=1.5, label=f"mean={per_label.mean():.2f}")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("per-label accuracy at t*")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, rf"Per-label accuracy @ best Hamming (t*={t_star:.2f})")
    ax.text(0.98, 0.08, "Binary relevance ignores label\ndependence; chains can help.\nScores ≠ causal comorbidity map.",
            transform=ax.transAxes, ha="right", fontsize=8, color=SLATE)
    fig.suptitle("Multi-label evaluation: Hamming, exact match, micro-F1 (synthetic; original)",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ml_fig_hamming_multilabel.png")


def main():
    fig_nested_fs_cv()
    fig_attention_temperature()
    fig_ppo_clip()
    fig_km_censor()
    fig_shap_collinearity()
    fig_mixed_precision()
    fig_hamming_multilabel()
    print("DONE cycle-13 figures")


if __name__ == "__main__":
    main()
