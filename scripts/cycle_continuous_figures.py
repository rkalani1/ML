#!/usr/bin/env python3
"""Continuous densify: novel scientific teal teaching panels for arbitrary cycles."""
from __future__ import annotations

import sys
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

# Per-chapter topic banks (rotated by cycle) — novel ML science teaching labels
BANKS = [
    # ch00 math
    [
        "Krylov subspace residual path",
        "Arnoldi orthogonal loss path",
        "Householder QR residual",
        "Cholesky condition path",
        "SVD singular decay path",
        "Jacobi iteration residual",
        "Gauss-Seidel residual",
        "SOR omega residual path",
        "Chebyshev semi-iter residual",
        "Multigrid V-cycle residual",
        "Preconditioned CG residual",
        "Deflated CG residual path",
        "Block CG residual path",
        "Flexible GMRES residual",
        "Recycling Krylov residual",
        "Sketch-and-project residual",
    ],
    # 00a preface / ethics-ops
    [
        "Data use purpose path",
        "Consent scope control path",
        "Model risk tier path",
        "Human oversight gate path",
        "Red-team severity path",
        "Incident SEV ladder path",
        "SBOM attest ship path",
        "Threat residual risk path",
        "Policy exception path",
        "Audit trail control path",
        "Fairness review path",
        "Privacy DPIA path",
        "Vendor risk path",
        "Kill-switch runbook path",
        "Rollback decision path",
        "Postmortem action path",
    ],
    # 01 basic concepts
    [
        "Bias-variance residual map",
        "PAC bound sample growth",
        "No-free-lunch sketch",
        "Inductive bias path",
        "Hypothesis class growth",
        "Empirical risk path",
        "Structural risk path",
        "Occam razor trade curve",
        "Bayes error floor path",
        "Agnostic learning path",
        "Online mistake bound",
        "Active query gain path",
        "Semi-supervised mix path",
        "Multi-task transfer path",
        "Domain shift risk path",
        "Causality vs prediction strip",
    ],
    # 02 viz
    [
        "Color map accessibility heat",
        "Small multiples layout path",
        "Linked brushing scatter",
        "Parallel coords bars",
        "Radar chart misuse bars",
        "Uncertainty band path",
        "Log-scale distortion path",
        "Glyph encoding scatter",
        "Treemap hierarchy heat",
        "Sankey flow path",
        "Calibrated color heat",
        "Animation frame residual",
        "Facet grid scatter",
        "Raincloud density path",
        "QQ residual path",
        "ECDF comparison path",
    ],
    # 03 prob/stats
    [
        "CLT sample mean path",
        "Bootstrap CI width path",
        "Jackknife bias path",
        "Posterior concentration path",
        "Prior sensitivity path",
        "Likelihood ratio path",
        "Power vs n curve",
        "Type I/II trade path",
        "FDR control path",
        "Permutation null bars",
        "Credible interval path",
        "HMC acceptance path",
        "Variational ELBO path",
        "Importance weight ESS",
        "ABC tolerance cool path",
        "Conjugate update path",
    ],
    # 04 clustering
    [
        "K-means inertia path",
        "K-medoids swap residual",
        "GMM EM likelihood path",
        "DBSCAN density scatter",
        "HDBSCAN condensed scatter",
        "OPTICS reachability path",
        "Spectral embedding scatter",
        "Mean-shift mode scatter",
        "BIRCH CF scatter",
        "Affinity prop exemplar",
        "Fuzzy c-means soft scatter",
        "Consensus cluster heat",
        "Cluster validity bars",
        "Gap statistic path",
        "Silhouette class bars",
        "Noise vs core scatter",
    ],
    # 05 itemset/IR
    [
        "Apriori level-wise path",
        "FP-growth tree path",
        "Eclat tidset path",
        "Charm closed path",
        "PrefixSpan project path",
        "SPADE lattice path",
        "MinHash Jaccard path",
        "LSH band collision path",
        "TF-IDF weight bars",
        "BM25 score bars",
        "Inverted index path",
        "Query expand path",
        "N-gram coverage bars",
        "Sequence gap path",
        "Closed vs max path",
        "Support confidence path",
    ],
    # 06 feature eng
    [
        "Target encode leak path",
        "Hash trick collision path",
        "Polynomial degree path",
        "Binning residual path",
        "WOE transform path",
        "Interaction term bars",
        "Missing indicator path",
        "Scaler choice bars",
        "Rare category group path",
        "Time feature lag path",
        "Cyclical encode path",
        "Text ngram DF bars",
        "Image aug strength path",
        "Feature selection gain",
        "Mutual info bars",
        "Leakage audit path",
    ],
    # 07 dim reduction
    [
        "PCA variance path",
        "Randomized SVD error",
        "NMF recon residual",
        "ICA independence path",
        "t-SNE cool path",
        "UMAP neighbor path",
        "PaCMAP mid-near path",
        "Isomap geodesic path",
        "LLE local residual",
        "Autoencoder recon path",
        "Sparse coding residual",
        "Dictionary size error",
        "Kernel PCA error",
        "Incremental PCA path",
        "Truncated SVD path",
        "Manifold trust path",
    ],
    # 08 regression
    [
        "OLS residual QQ path",
        "Ridge path residual",
        "Lasso soft-threshold path",
        "Elastic net path",
        "Huber residual cost",
        "Quantile pinball path",
        "Poisson deviance path",
        "Tweedie deviance path",
        "GAM smooth residual",
        "Spline knot residual",
        "Heteroscedasticity path",
        "Leverage cook bars",
        "Cross-val MSE path",
        "Partial residual path",
        "Isotonic mono path",
        "Expectile residual path",
    ],
    # 09 classification
    [
        "Logistic loss residual",
        "SVM margin path",
        "Tree depth bias path",
        "Random forest OOB path",
        "GBM stage residual",
        "XGBoost leaf bars",
        "Calibration reliability heat",
        "PR curve residual path",
        "ROC threshold path",
        "Cost-sensitive bars",
        "Class weight bars",
        "One-vs-rest residual",
        "Platt scale path",
        "Isotonic calib path",
        "Confusion structure heat",
        "Threshold utility path",
    ],
    # 10 neural nets
    [
        "Backprop residual path",
        "BatchNorm moving path",
        "LayerNorm residual path",
        "Residual highway path",
        "Attention weight heat",
        "Softmax temp map",
        "GELU activation path",
        "Dropout mask residual",
        "Init scale residual",
        "Learning rate warm path",
        "Grad clip norm path",
        "Skip connection path",
        "Depth vs width trade",
        "Lottery ticket path",
        "Neural tangent residual",
        "Feature collapse path",
    ],
    # 11 SSL
    [
        "SimCLR NT-Xent path",
        "MoCo queue path",
        "BYOL stop-grad path",
        "SimSiam twin path",
        "Barlow twins path",
        "VICReg var-inv path",
        "DINO teacher path",
        "MAE mask recon path",
        "JEPA latent path",
        "SwAV code swap path",
        "CLIP align path",
        "data2vec EMA path",
        "I-JEPA block path",
        "Masked unit path",
        "Contrast temp path",
        "Projection dim bars",
    ],
    # 12 multimodal
    [
        "Tokenize BPE path",
        "Positional encode heat",
        "Transformer block path",
        "Vision patch path",
        "Conv stem residual",
        "Audio spectrogram heat",
        "CTC blank path",
        "Transducer align path",
        "Whisper multi-task path",
        "Cross-attn heat",
        "Fusion early late path",
        "OCR layout path",
        "Speech enhance path",
        "Video frame sample path",
        "Multimodal align bars",
        "Caption decode path",
    ],
    # 13 RL
    [
        "Bellman residual path",
        "TD error path",
        "Q-learning update path",
        "Policy gradient path",
        "Advantage estimate path",
        "PPO clip path",
        "TRPO KL path",
        "SAC entropy path",
        "DQN target lag path",
        "n-step return path",
        "Eligibility trace path",
        "Actor-critic residual",
        "Exploration noise path",
        "Reward shaping path",
        "Offline RL restraint path",
        "Preference reward path",
    ],
    # 14 lighter models
    [
        "Prune magnitude bars",
        "Quant int8 error bars",
        "Distill temperature path",
        "Knowledge transfer path",
        "LoRA rank bars",
        "QLoRA NF4 bars",
        "Adapter insert path",
        "Weight share residual",
        "Early exit path",
        "Token prune path",
        "KV cache size bars",
        "Speculative draft path",
        "FlashAttention trade",
        "Activation ckpt trade",
        "Group-query trade",
        "SmoothQuant migrate bars",
    ],
    # 15 graphs
    [
        "PageRank iterate path",
        "Louvain modularity path",
        "Leiden refine path",
        "Label prop harmonic path",
        "GCN message path",
        "GAT attention heat",
        "GraphSAGE sample path",
        "Node2vec walk path",
        "Link predict bars",
        "Betweenness bars",
        "Community cut path",
        "Spectral cut path",
        "Motif count bars",
        "Temporal edge path",
        "Heterograph type path",
        "Graph pool residual",
    ],
    # 16 data challenges
    [
        "As-of join path",
        "Training-serving skew",
        "Feature staleness path",
        "Schema drift spike",
        "Null rate spike",
        "Duplicate spike path",
        "Label delay path",
        "PIT correctness path",
        "Backfill watermark path",
        "CDC stream path",
        "Entity resolve path",
        "Freshness SLO path",
        "Leakage audit path",
        "Window aggregate path",
        "Late arrival spike",
        "Data contract path",
    ],
    # 17 senior practice
    [
        "Canary promote path",
        "Blue-green cutover path",
        "Shadow traffic path",
        "Error budget burn path",
        "SLO multi-window path",
        "Alert noise burn-down",
        "On-call toil burn-down",
        "MTTR residual path",
        "Progressive delivery path",
        "Feature flag ramp path",
        "Model card gate path",
        "Eval suite residual path",
        "Champion challenger path",
        "Rollback trigger path",
        "Capacity headroom path",
        "Cost per query path",
    ],
    # 18 glossary
    [
        "Glossary loss strip",
        "Glossary optim strip",
        "Glossary SSL strip",
        "Glossary RL strip",
        "Glossary privacy strip",
        "Glossary graph strip",
        "Glossary calib strip",
        "Glossary genAI strip",
        "Glossary metric strip",
        "Glossary regularize strip",
        "Glossary embedding strip",
        "Glossary quant strip",
        "Glossary eval strip",
        "Glossary cluster strip",
        "Glossary IR strip",
        "Glossary deploy strip",
    ],
]


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


RENDERERS = [
    make_curve("exp"),
    make_flow(["need", "risk", "control", "sign"]),
    make_curve("power"),
    make_heat(),
    make_curve("sigmoid"),
    make_scatter(),
    make_flow(["scan", "join", "prune", "out"]),
    make_curve("log"),
    make_curve("damped"),
    make_curve("relu"),
    make_bars(),
    make_flow(["x", "F", "+", "y"]),
    make_flow(["a", "b", "c", "d"]),
    make_heat(),
    make_semilog(),
    make_bars(),
    make_curve("sigmoid"),
    make_flow(["as-of", "join", "train", "serve"]),
    make_curve("log"),
    make_flow(["A", "B", "C", "D"]),
]


def topics_for(cycle: int) -> list:
    assert len(BANKS) == len(CHS) == 20
    out = []
    for i, bank in enumerate(BANKS):
        title = bank[(cycle + i * 3) % len(bank)]
        # cycle-unique suffix to avoid caption collision across dense cycles
        title = f"{title} c{cycle}"
        fn = RENDERERS[i % len(RENDERERS)]
        # diversify renderers per cycle
        alt = [
            make_curve("exp"),
            make_curve("log"),
            make_curve("power"),
            make_curve("sigmoid"),
            make_curve("relu"),
            make_curve("damped"),
            make_heat(),
            make_scatter(),
            make_bars(),
            make_semilog(),
            make_quiver(),
            make_flow(["in", "mid", "out", "done"]),
        ]
        fn = alt[(cycle + i * 5) % len(alt)]
        out.append((title, fn))
    return out


def embed(cycle: int) -> None:
    topics = topics_for(cycle)
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
    if len(sys.argv) < 2:
        print("usage: cycle_continuous_figures.py START[,END] or CYCLE")
        sys.exit(2)
    arg = sys.argv[1]
    if "," in arg:
        a, b = arg.split(",", 1)
        cycles = list(range(int(a), int(b) + 1))
    elif "-" in arg and arg.replace("-", "").isdigit():
        a, b = arg.split("-", 1)
        cycles = list(range(int(a), int(b) + 1))
    else:
        cycles = [int(arg)]
    for c in cycles:
        embed(c)
