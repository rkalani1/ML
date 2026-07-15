#!/usr/bin/env python3
"""Cycle-100/101/102 continuous densify."""
from cycle91_93_figures import draw, save, CHS, CURR
import matplotlib.pyplot as plt

C100=[
("Neumann series inverse sketch","rank"),
("Assent vs consent documentation","pap"),
("Approximation-estimation-optim split","occam"),
("Cumulative distribution ladders","violin"),
("Exact vs asymptotic tests","pval"),
("Streaming k-means centroids","hdb"),
("Attention sink tokens","ngram"),
("Target leakage calendar map","bin"),
("Laplacian eigenmaps","umapg"),
("Tweedie compound Poisson","gam"),
("Macro vs micro F1","margin"),
("One-cycle policy LR","lr"),
("VICReg variance-invariance","moco"),
("Speculative sampling tree","beam"),
("Successor features RL","softp"),
("GPTQ weight quantization","flash"),
("Graph rewiring over-smoothing","gat"),
("Label delay in production","skew"),
("Safety case structure","silent"),
("Eval strip: ID OOD stress","acr"),
]
C101=[
("Cramers rule geometric view","eig"),
("CARE checklist tiles","tripod"),
("Uniform convergence cartoon","pac"),
("Sankey exclusion cascade","alluvial"),
("TOST equivalence testing","lrt"),
("Active learning query strategies","coreset"),
("Dense retrieval dual encoders","ltr"),
("Count encoding with CV","missind"),
("Autoencoder vs PCA axes","kpca"),
("Zero-inflated outcome mass","gam"),
("Class-wise calibration curves","ece"),
("Muon optimizer sketch","adam"),
("ImageBind joint embedding","clip"),
("NoPE no position ablation","posenc"),
("GRPO group relative policy","adv"),
("MLA multi-head latent attn","flash"),
("SEAL self-adapting loop","n2v"),
("Secure aggregation federated","dp"),
("Feature flag gradual expose","champ"),
("Info strip: MI KL JS","greek"),
]
C102=[
("Moore-Penrose properties strip","svdecay"),
("Audit trail hash chain","consort"),
("Benign overfitting cartoon","erm"),
("Bland-Altman with proportional bias","ba"),
("Sequential testing alpha spend","bonf"),
("Deep embedded clustering","spectral"),
("Hybrid sparse-dense retrieval","prox"),
("Hash embeddings for IDs","entemb"),
("PaCMAP neighbor preservation","umapg"),
("AFT survival acceleration","gam"),
("Decision curve clinical net benefit","ece"),
("Schedule-free optimizer idea","clipg"),
("DINOv2 registers tokens","jepa"),
("MM-DiT multimodal diffusion","xattn"),
("ICM inverse curriculum","intrin"),
("EAGLE draft head","spec"),
("Temporal GNN snapshots","louvain"),
("Model inversion attack","synth"),
("On-call runbook gates","shadow"),
("Concentration strip: Chernoff","ineq"),
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
    m={100:C100,101:C101,102:C102}
    cycles=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 else [100,101,102]
    for c in cycles: run(c,m[c])
