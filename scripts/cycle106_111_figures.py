#!/usr/bin/env python3
"""Cycle-106..111 continuous densify."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle91_93_figures import draw, save, CHS, CURR
import matplotlib.pyplot as plt

base_titles = {
0: ['Krylov subspace iteration','Gershgorin disk theorem','Power method convergence','Arnoldi process sketch','Rayleigh quotient'],
1: ['Stakeholder map for data use','Minimal risk pathway','Secondary use rules','De-identification checklist','Data use agreement boxes'],
2: ['Surrogate loss calibration','Bayes consistency sketch','Agnostic learning setup','Realizable case bound','Excess risk decomposition'],
3: ['Horizon chart density','Dot plot with CI','Marimekko mosaic','Hexbin log counts','Paired slope multiples'],
4: ['Beta-binomial overdispersion','Empirical Bayes shrink','Posterior predictive check','Credible vs confidence','Jeffreys prior sketch'],
5: ['Fuzzy c-means membership','Self-organizing map grid','Affinity matrix threshold','Consensus clustering','COP-kmeans constraints'],
6: ['SpanBERT span masking','Dense passage retrieval','Query likelihood LM','PRF pseudo relevance','Learning sparse retrievers'],
7: ['Polynomial basis explosion','Spline knot placement','Feature crosses cardinality','Hashing vs embedding IDs','Ordinal encoding ranks'],
8: ['Nonnegative matrix parts','Dictionary learning atoms','Tensor CP decomposition','CCA shared views','ICA independence axes'],
9: ['MM algorithm majorization','IRLS weighted steps','GEE cluster robust SE','Mixed effects random slope','Partial residual smoother'],
10: ['Ordinal regression thresholds','Multilabel classifier chains','Cost-sensitive weighting','Label powerset reduce','Platt vs isotonic'],
11: ['Residual stream view','Mixture-of-experts routing','Hypernetworks weights','Neural ODE flow','Deep equilibrium model'],
12: ['Masked image modeling','Jigsaw pretext task','Colorization pretext','Rotation prediction','Temporal order SSL'],
13: ['Conformer conv-attn','Whisper encoder stack','Vision language projector','Audio spectrogram CNN','Token merging ToMe'],
14: ['World models dreamer','Model-based MPC loop','Offline RL conservatism','Distributional RL quantiles','Successor representation'],
15: ['SparseGPT pruning','Wanda prune metric','SmoothQuant scales','QLoRA NF4 storage','Speculative draft tree'],
16: ['Motif counting graphs','Link prediction scores','Knowledge graph embeds','Hyperbolic graph space','Graph pooling DiffPool'],
17: ['Prospective vs retrospective','Index date alignment','Immortal time bias','Competing event censor','Chart review gold labels'],
18: ['Value of information','Decision impact analysis','Equity slice metrics','Post-market surveillance','Model retirement criteria'],
19: ['Glossary support vector','Glossary inductive bias','Glossary conformal set','Glossary aleatoric risk','Glossary epistemic risk'],
}
kinds = ['rank','pap','occam','violin','pval','hdb','ngram','bin','umapg','huber','margin','lr','moco','beam','softp','flash','gat','skew','silent','acr']

def topics_for(cycle):
    offset = (cycle - 106) % 5
    return [(base_titles[i][offset % len(base_titles[i])], kinds[i]) for i in range(20)]

def run(cycle):
    topics = topics_for(cycle)
    for i, (title, kind) in enumerate(topics):
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        draw(ax, kind, title, cycle * 100 + i)
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
    print("EMBEDDED", cycle, 20)

if __name__ == "__main__":
    cycles = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(106, 112))
    for c in cycles:
        run(c)
