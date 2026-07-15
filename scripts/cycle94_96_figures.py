#!/usr/bin/env python3
"""Cycle-94/95/96 continuous densify - novel teal teaching figures."""
from cycle91_93_figures import draw, save, box, style, CHS, CURR, TEAL, DEEP, INK, GOLD, SLATE, ROSE
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Remap existing draw kinds with new teaching titles (still novel captions/embeds)
C94 = [
    ("Cholesky factor triangle idea", "rank"),
    ("Protocol deviation log", "pap"),
    ("VC-dimension capacity cartoon", "occam"),
    ("ECDF comparison of two groups", "violin"),
    ("Type I vs Type II error regions", "pval"),
    ("Mean-shift mode seeking path", "hdb"),
    ("Sliding window sequence motifs", "ngram"),
    ("Hashing trick collision budget", "bin"),
    ("Isomap geodesic distances", "umapg"),
    ("Elastic net diamond+disk constraint", "huber"),
    ("Softmax multi-class simplex", "softp"),
    ("Warm restarts learning rate", "lr"),
    ("SwAV online clustering codes", "moco"),
    ("Nucleus sampling top-p mass", "beam"),
    ("Entropy bonus for exploration", "intrin"),
    ("Tokenizer BPE merge steps", "spec"),
    ("Label propagation on graph", "gat"),
    ("Federated averaging rounds", "shadow"),
    ("Model card risk tier matrix", "silent"),
    ("Loss zoo strip: CE MSE Huber", "acr"),
]
C95 = [
    ("Pseudoinverse least-squares map", "svdecay"),
    ("Registered analysis time-lock", "tripod"),
    ("Double descent test risk curve", "erm"),
    ("QQ plot normality check", "ba"),
    ("FDR Benjamini-Hochberg steps", "bonf"),
    ("Affinity propagation exemplars", "coreset"),
    ("Okapi BM25 IDF curve", "prox"),
    ("Leave-one-group-out CV blocks", "missind"),
    ("Multidimensional scaling stress", "kpca"),
    ("Poisson GLM log-link mean", "gam"),
    ("Brier score calibration square", "ece"),
    ("Lookahead optimizer sketch", "adam"),
    ("SigLIP sigmoid pairwise loss", "clip"),
    ("RoPE rotary position angles", "posenc"),
    ("GAE generalized advantage", "adv"),
    ("KV-cache autoregressive reuse", "flash"),
    ("LINE first/second-order prox", "n2v"),
    ("k-anonymity quasi-identifiers", "dp"),
    ("Canary deployment percent ramp", "champ"),
    ("Metric strip: recall F1 MCC", "greek"),
]
C96 = [
    ("Householder reflection sketch", "eig"),
    ("Data provenance ledger chain", "consort"),
    ("Interpolation threshold n≈p", "pac"),
    ("Raincloud alt: strip+box only", "violin"),
    ("Power vs effect size curves", "lrt"),
    ("OPTICS reachability plot idea", "spectral"),
    ("Session-based next-item rank", "ltr"),
    ("Frequency-rarity feature plot", "entemb"),
    ("t-SNE early exaggeration phase", "umapg"),
    ("Spline basis stack for GAM", "gam"),
    ("Proper scoring rule ranking", "ece"),
    ("Mixed precision FP16 overflow", "clipg"),
    ("I-JEPA multi-block masks", "jepa"),
    ("Encoder-decoder cross-attn grid", "xattn"),
    ("Count-based exploration bonus", "intrin"),
    ("Medusa multi-head drafting", "spec"),
    ("Heterophily graph caution", "louvain"),
    ("Membership inference risk", "synth"),
    ("Rollback criteria dashboard", "skew"),
    ("Bound strip: VC, Rademacher", "ineq"),
]

def run(cycle, topics):
    for i,(title,kind) in enumerate(topics):
        fig,ax=plt.subplots(figsize=(7.6,3.9))
        draw(ax, kind, title, cycle*100+i)
        save(fig, f"ml_fig_c{cycle}_{i:02d}.png")
    for i,ch in enumerate(CHS):
        p=CURR/ch
        fig=f"ml_fig_c{cycle}_{i:02d}.png"
        cap=topics[i][0]
        block=(f"\n![c{cycle} teaching panel {i:02d} (original).](../assets/figures/{fig})\n"
               f"*Figure — {cap}. Synthetic teaching geometry—not a causal claim.*\n")
        text=p.read_text(encoding="utf-8")
        if fig in text: continue
        if "## Chapter Summary" in text:
            text=text.replace("## Chapter Summary", block+"\n## Chapter Summary", 1)
        else:
            text=text.rstrip()+"\n"+block
        p.write_text(text, encoding="utf-8")
    print("EMBEDDED", cycle, 20)

if __name__=="__main__":
    import sys
    m={94:C94,95:C95,96:C96}
    cycles=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 else [94,95,96]
    for c in cycles: run(c,m[c])
