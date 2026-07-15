#!/usr/bin/env python3
"""Cycle-97/98/99 continuous densify - novel teal teaching figures."""
from cycle91_93_figures import draw, save, CHS, CURR
import matplotlib.pyplot as plt

C97 = [
    ("QR decomposition thin/full forms", "rank"),
    ("Ethics review amendment trail", "pap"),
    ("Bias-complexity tradeoff redraw", "occam"),
    ("Ridgeline density facets", "violin"),
    ("Confidence interval coverage sim", "pval"),
    ("BIRCH CF-tree clustering sketch", "hdb"),
    ("Skip-gram context window", "ngram"),
    ("Interaction feature crossed grid", "bin"),
    ("Locally linear embedding patch", "umapg"),
    ("Quantile loss fan of tau", "pinball"),
    ("Decision stump axis split", "margin"),
    ("Cyclical learning rate triangles", "lr"),
    ("Barlow Twins redundancy reduction", "moco"),
    ("Contrastive decoding candidate set", "beam"),
    ("Max-entropy RL objective", "softp"),
    ("Product quantization codebooks", "flash"),
    ("Weisfeiler-Lehman color refine", "gat"),
    ("Concept drift detector alarm", "skew"),
    ("Human-in-loop escalation ladder", "silent"),
    ("Optimization strip: GD SGD Adam", "acr"),
]
C98 = [
    ("Determinant as parallelogram area", "eig"),
    ("FAIR data principles tiles", "tripod"),
    ("Interpolation vs extrapolation zones", "pac"),
    ("Parallel sets categorical flow", "alluvial"),
    ("Permutation test null histogram", "lrt"),
    ("CURE representative points", "coreset"),
    ("Learning-to-rank listwise softmax", "ltr"),
    ("CatBoost ordered target stats", "missind"),
    ("Diffusion map eigenmodes", "kpca"),
    ("Negative binomial overdispersion", "gam"),
    ("Sharpness of probability peaks", "ece"),
    ("SAM sharpness-aware min sketch", "adam"),
    ("FLAVA multimodal alignment", "clip"),
    ("ALiBi linear attention bias", "posenc"),
    ("PPO clip objective cartoon", "adv"),
    ("Paged attention block tables", "flash"),
    ("GraphSAGE neighbor sample", "n2v"),
    ("Homomorphic encrypt latency tax", "dp"),
    ("Blue/green deploy switch", "champ"),
    ("Prob strip: PDF CDF PMF HF", "greek"),
]
C99 = [
    ("Trace as sum of eigenvalues", "svdecay"),
    ("Versioned feature store timeline", "consort"),
    ("Effective degrees of freedom", "erm"),
    ("Hexbin vs scatter density choice", "ba"),
    ("Bootstrap CI percentile method", "bonf"),
    ("CLIQUE subspace clusters", "spectral"),
    ("ColBERT late interaction scores", "prox"),
    ("Rare-category grouping ladder", "entemb"),
    ("PHATE trajectory embedding", "umapg"),
    ("Cox partial likelihood idea", "gam"),
    ("Spiegelhalter calibration z", "ece"),
    ("Lion optimizer sign updates", "clipg"),
    ("data2vec self-distill targets", "jepa"),
    ("Perceiver latent bottleneck", "xattn"),
    ("RND prediction error bonus", "intrin"),
    ("Lookahead decoding verify", "spec"),
    ("Temporal graph edge streams", "louvain"),
    ("Reconstruction attack risk", "synth"),
    ("Kill-switch criteria board", "shadow"),
    ("Inequality strip II: Bernstein", "ineq"),
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
    m={97:C97,98:C98,99:C99}
    cycles=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 else [97,98,99]
    for c in cycles: run(c,m[c])
