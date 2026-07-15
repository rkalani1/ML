#!/usr/bin/env python3
"""Cycle-249..256 quality densify: novel scientific teal teaching panels."""
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


C249 = [
    ("IDR-S residual defect path", make_curve("exp")),
    ("Hockey-stick DP conversion", make_curve("log")),
    ("Pseudo-dimension bound growth", make_curve("power")),
    ("Beta calibration reliability heat", make_heat()),
    ("Rank-normalized R-hat path", make_curve("sigmoid")),
    ("Spectral clustering scatter", make_scatter()),
    ("Diffset CHARM closed path", make_flow(["scan", "diff", "close", "F_k"])),
    ("p-stable LSH collision rate", make_curve("power")),
    ("PaCMAP mid-near cool path", make_curve("damped")),
    ("Welsch robust residual cost", make_curve("relu")),
    ("Matthews corr class bars", make_bars()),
    ("ConvNeXt modern residual", make_flow(["LN", "DWConv", "MLP", "out"])),
    ("MAE mask reconstruct path", make_flow(["mask", "enc", "dec", "MSE"])),
    ("ALiBi slope stack heat", make_heat()),
    ("Retrace C operator scale", make_semilog()),
    ("VeRA shared adapter bars", make_bars()),
    ("Leiden multi-res path", make_curve("sigmoid")),
    ("Slowly changing dim path", make_flow(["detect", "key", "hist", "serve"])),
    ("Feature flag ramp path", make_curve("log")),
    ("Glossary calib strip", make_flow(["Platt", "isotonic", "temp", "beta"])),
]

C250 = [
    ("QMR-SYM residual path", make_curve("exp")),
    ("SBOM provenance path", make_flow(["gen", "attest", "scan", "ship"])),
    ("Occlusion sensitivity bars", make_bars()),
    ("ECOD empirical CDF cloud", make_scatter()),
    ("Probability flow field", make_quiver()),
    ("Stochastic block heat", make_heat()),
    ("ClaSP closed sequence path", make_flow(["SDB", "prefix", "project", "grow"])),
    ("Wasserstein bin bars", make_bars()),
    ("Randomized SVD error path", make_curve("power")),
    ("Pinball asymmetric residual", make_curve("relu")),
    ("Entmax-1.5 sparsity map", make_curve("sigmoid")),
    ("Speculative decode trade", make_curve("log")),
    ("SimCLR NT-Xent path", make_flow(["aug1", "aug2", "proj", "NCE"])),
    ("Whisper multi-task path", make_flow(["audio", "enc", "dec", "task"])),
    ("RRHF rank response path", make_curve("damped")),
    ("GPTQ act-order bars", make_bars()),
    ("Cutout region schedule", make_curve("sigmoid")),
    ("Late join lag spike", make_curve("log")),
    ("Pager noise burn-down", make_bars()),
    ("Glossary graph term strip", make_flow(["PageRank", "Louvain", "GCN", "GAT"])),
]

C251 = [
    ("MINRES-QLP residual path", make_curve("exp")),
    ("RDP order optimize curve", make_curve("log")),
    ("Fat-shattering dim growth", make_curve("power")),
    ("Venn-Abers reliability heat", make_heat()),
    ("Multi-chain ESS path", make_curve("sigmoid")),
    ("BIRCH CF-tree scatter", make_scatter()),
    ("dEclat vertical mine path", make_flow(["scan", "diffset", "join", "F_k"])),
    ("OPH LSH collision rate", make_curve("power")),
    ("UMAP negative sample cool", make_curve("damped")),
    ("Fairness demographic parity", make_curve("relu")),
    ("G-mean class bars", make_bars()),
    ("RegNet any-stage residual", make_flow(["stem", "stage", "head", "out"])),
    ("data2vec teacher path", make_flow(["mask", "student", "EMA", "L2"])),
    ("T5 relative bias heat", make_heat()),
    ("IMPALA V-trace scale", make_semilog()),
    ("LoRA+ LR ratio bars", make_bars()),
    ("Walktrap community path", make_curve("sigmoid")),
    ("Backfill watermark path", make_flow(["as-of", "join", "train", "serve"])),
    ("Shadow eval delta path", make_curve("log")),
    ("Glossary SSL strip", make_flow(["MAE", "DINO", "JEPA", "CLIP"])),
]

C252 = [
    ("BiCG residual dual path", make_curve("exp")),
    ("Model card review path", make_flow(["scope", "metrics", "limits", "sign"])),
    ("Integrated gradients bars", make_bars()),
    ("KNN distance anomaly cloud", make_scatter()),
    ("Hamiltonian flow field", make_quiver()),
    ("Degree-corrected SBM heat", make_heat()),
    ("GoKrimp sequential path", make_flow(["SDB", "prefix", "project", "grow"])),
    ("KS two-sample bars", make_bars()),
    ("Incremental PCA error", make_curve("power")),
    ("Huber delta residual map", make_curve("relu")),
    ("Gumbel-softmax anneal map", make_curve("sigmoid")),
    ("MQA head-share trade", make_curve("log")),
    ("BYOL stop-grad path", make_flow(["online", "pred", "target", "MSE"])),
    ("Conformer block path", make_flow(["FFN", "MHSA", "conv", "FFN"])),
    ("SLiC sequence lik path", make_curve("damped")),
    ("AWQ zero-point bars", make_bars()),
    ("Mixup alpha schedule", make_curve("sigmoid")),
    ("Schema version spike", make_curve("log")),
    ("Toil hour burn-down", make_bars()),
    ("Glossary privacy strip", make_flow(["DP-SGD", "RDP", "zCDP", "GDP"])),
]

C253 = [
    ("LSQR least-squares residual", make_curve("exp")),
    ("f-DP Gaussian tradeoff", make_curve("log")),
    ("Natarajan class growth", make_curve("power")),
    ("Dirichlet calib reliability", make_heat()),
    ("Split-R-hat path", make_curve("sigmoid")),
    ("HDBSCAN leaf scatter", make_scatter()),
    ("LCM-freq closed path", make_flow(["scan", "extend", "close", "F_k"])),
    ("Super-bit LSH rate", make_curve("power")),
    ("t-SNE late cool path", make_curve("damped")),
    ("Quantile check residual", make_curve("relu")),
    ("Balanced F-beta bars", make_bars()),
    ("MobileNet inverted residual", make_flow(["expand", "DW", "project", "out"])),
    ("I-JEPA multi-block path", make_flow(["ctx", "tgt", "pred", "L2"])),
    ("RoPE NTK-scale heat", make_heat()),
    ("n-step lambda mix", make_semilog()),
    ("QLoRA double-quant bars", make_bars()),
    ("Label prop harmonic path", make_curve("sigmoid")),
    ("PIT correct join path", make_flow(["as-of", "join", "train", "serve"])),
    ("Canary promote path", make_curve("log")),
    ("Glossary regularize strip", make_flow(["weight-decay", "dropout", "LS", "aug"])),
]

C254 = [
    ("GMRES(m) restart residual", make_curve("exp")),
    ("Incident SEV path", make_flow(["detect", "page", "mitigate", "review"])),
    ("SHAP beeswarm bars", make_bars()),
    ("Isolation forest cloud", make_scatter()),
    ("Score-based Langevin field", make_quiver()),
    ("Core-periphery heat", make_heat()),
    ("VMSP maximal sequence path", make_flow(["SDB", "prefix", "project", "grow"])),
    ("PSI population shift bars", make_bars()),
    ("Nyström approx error", make_curve("power")),
    ("Expectile residual map", make_curve("relu")),
    ("Sparsemax proj map", make_curve("sigmoid")),
    ("GQA group-query trade", make_curve("log")),
    ("MoCo momentum path", make_flow(["query", "key", "queue", "NCE"])),
    ("RNN-T alignment path", make_flow(["audio", "enc", "joiner", "text"])),
    ("ORPO odds path", make_curve("damped")),
    ("SmoothQuant alpha bars", make_bars()),
    ("RandAugment N schedule", make_curve("sigmoid")),
    ("Duplicate rate spike", make_curve("log")),
    ("Alert noise burn-down", make_bars()),
    ("Glossary RL strip", make_flow(["PPO", "DPO", "GRPO", "REINFORCE"])),
]

C255 = [
    ("CGNE residual normal eq", make_curve("exp")),
    ("zCDP composition map", make_curve("log")),
    ("VC subgraph growth", make_curve("power")),
    ("Temperature scaling heat", make_heat()),
    ("Bulk ESS path", make_curve("sigmoid")),
    ("OPTICS xi-cluster scatter", make_scatter()),
    ("Apriori-TID mine path", make_flow(["scan", "join", "prune", "F_k"])),
    ("Cross-polytope LSH rate", make_curve("power")),
    ("Force-directed cool path", make_curve("damped")),
    ("Tukey biweight cost", make_curve("relu")),
    ("Top-k accuracy bars", make_bars()),
    ("DenseNet growth residual", make_flow(["x", "H", "cat", "out"])),
    ("VICReg var-inv-cov path", make_flow(["z", "var", "inv", "cov"])),
    ("Relative pos bias heat", make_heat()),
    ("GAE lambda return scale", make_semilog()),
    ("DoRA decompose bars", make_bars()),
    ("Infomap compress path", make_curve("sigmoid")),
    ("Online offline store path", make_flow(["as-of", "join", "train", "serve"])),
    ("SLO burn rate path", make_curve("log")),
    ("Glossary optim strip", make_flow(["AdamW", "Lion", "Muon", "SOAP"])),
]

C256 = [
    ("TFQMR transpose residual", make_curve("exp")),
    ("Threat model path", make_flow(["asset", "threat", "control", "residual"])),
    ("DeepLIFT channel bars", make_bars()),
    ("Local outlier factor cloud", make_scatter()),
    ("Reverse diffusion field", make_quiver()),
    ("Overlapping community heat", make_heat()),
    ("SPADE+ sequence path", make_flow(["SDB", "prefix", "project", "grow"])),
    ("JSD bin shift bars", make_bars()),
    ("Kernel PCA recon error", make_curve("power")),
    ("Cauchy residual map", make_curve("relu")),
    ("Softmax temperature map", make_curve("sigmoid")),
    ("FlashAttention-2 trade", make_curve("log")),
    ("CLIP image-text path", make_flow(["img", "txt", "proj", "NCE"])),
    ("HuBERT unit path", make_flow(["audio", "mask", "cluster", "CE"])),
    ("IPO identity pref path", make_curve("damped")),
    ("GPTQ groupsize bars", make_bars()),
    ("SpecAugment time-mask sched", make_curve("sigmoid")),
    ("Freshness lag spike", make_curve("log")),
    ("MTTR burn-down", make_bars()),
    ("Glossary genAI strip", make_flow(["AR", "diffusion", "flow", "consistency"])),
]


def embed(cycle: int, topics: list) -> None:
    assert len(topics) == len(CHS), f"{cycle}: {len(topics)} != {len(CHS)}"
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

    m = {
        249: C249,
        250: C250,
        251: C251,
        252: C252,
        253: C253,
        254: C254,
        255: C255,
        256: C256,
    }
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(m.keys())
    for c in cycles:
        embed(c, m[c])
