#!/usr/bin/env python3
"""Cycle-103/104/105 continuous densify."""
from cycle91_93_figures import draw, save, CHS, CURR
import matplotlib.pyplot as plt

C103=[
("Singular vectors left/right roles","rank"),
("Stakeholder RACI for ML ops","pap"),
("Hypothesis class nested families","occam"),
("Letter-value plot sketch","violin"),
("Likelihood ratio confidence set","pval"),
("Mini-batch k-means speed path","hdb"),
("Byte-level BPE vs wordpiece","ngram"),
("Group-time feature freeze","bin"),
("Hessian eigen spectrum loss","umapg"),
("Student-t robust regression","huber"),
("Error-correcting output codes","margin"),
("Polyak averaging weights","lr"),
("SimSiam stop-gradient twin","moco"),
("Best-of-N sampling curve","beam"),
("Option-critic hierarchical RL","softp"),
("AWQ activation-aware quant","flash"),
("Directed graph acyclicity check","gat"),
("Schema evolution compatibility","skew"),
("Red-team prompt taxonomy","silent"),
("Robustness strip: noise shift","acr"),
]
C104=[
("Orthogonal Procrustes alignment","eig"),
("Model facts label nutrition","tripod"),
("Rademacher complexity sketch","pac"),
("Upset-plot set intersections","alluvial"),
("Wilson score interval","lrt"),
("Uncertainty sampling vs diversity","coreset"),
("Multi-vector retrieval ColBERTv2","ltr"),
("Smoothed target encoding","missind"),
("Factor analysis loadings","kpca"),
("Hurdle model two-part","gam"),
("Threshold moving PR operating","ece"),
("Sophia second-order sketch","adam"),
("ALIGN noisy web pairs","clip"),
("T5 relative bucket positions","posenc"),
("DPO preference optimization","adv"),
("GQA grouped query attention","flash"),
("PinSAGE bipartite walks","n2v"),
("DP-SGD clip-and-noise","dp"),
("A/B test cuped adjustment","champ"),
("Causal strip: do-calc caution","greek"),
]
C105=[
("Woodbury matrix identity cartoon","svdecay"),
("Lineage graph of datasets","consort"),
("Implicit regularization GD","erm"),
("Mean absolute scaled error","ba"),
("False discovery proportion","bonf"),
("Contrastive clustering","spectral"),
("Reranker cross-encoder stack","prox"),
("Collisions in ID hashing","entemb"),
("TriMAP triplet constraints","umapg"),
("Competing risks cumulative","gam"),
("Net reclassification index","ece"),
("Prodigy adaptive rates","clipg"),
("I-JEPA world model blocks","jepa"),
("Flamingo gated x-attn","xattn"),
("Go-Explore archive return","intrin"),
("Medusa-2 multi-token","spec"),
("Dynamic graph snapshots","louvain"),
("Attribute inference risk","synth"),
("Incident severity matrix","shadow"),
("PAC-Bayes bound strip","ineq"),
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
    m={103:C103,104:C104,105:C105}
    cycles=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 else [103,104,105]
    for c in cycles: run(c,m[c])
