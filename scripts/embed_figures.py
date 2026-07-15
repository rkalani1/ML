#!/usr/bin/env python3
"""Embed original figures into ML curriculum chapters if missing."""
from __future__ import annotations

import re
from pathlib import Path

CURR = Path(__file__).resolve().parents[1] / "docs" / "curriculum"
# chapter file pattern -> list of (rel_path, caption)
MAP = {
    "00-mathematical": [
        ("../assets/figures/ml_fig_gradient_descent.png", "Gradient descent on a synthetic loss surface (original teaching graphic)."),
        ("../assets/figures/ml_fig_core_functions.png", "Core functions of machine learning (original teaching catalog)."),
    ],
    "00a-preface": [
        ("../assets/figures/ml_fig_how_to_read.png", "How to read this open-source ebook (original)."),
        ("../assets/figures/ml_fig_appraisal_scorecard.png", "Model appraisal scorecard (original)."),
        ("../assets/figures/ml_fig_curriculum_map.png", "Curriculum map — open-source ML ebook path (original)."),
    ],
    "01-basic": [
        ("../assets/figures/ml_fig_learning_curves.png", "Learning curves vs sample size (synthetic; original)."),
        ("../assets/figures/ml_fig_supervised_unsupervised_map.png", "Supervised versus unsupervised learning paths (original teaching graphic)."),
        ("../assets/figures/ml_fig_train_val_test.png", "Train / validation / test split along a clinical timeline (original)."),
        ("../assets/figures/ml_fig_bias_capacity.png", "Training vs validation error versus model capacity (original)."),
        ("../assets/figures/ml_fig_bias_variance_decomp.png", "Bias–variance decomposition of expected squared error versus capacity (synthetic; original)."),
    ],
    "02-visualization": [
        ("../assets/figures/ml_fig_viz_hygiene.png", "Visualization hygiene (original)."),
        ("../assets/figures/ml_fig_calibration.png", "Calibration view as a visual truth check (original teaching graphic)."),
    ],
    "03-probability": [
        ("../assets/figures/ml_fig_calibration.png", "Predicted risk versus observed frequency (synthetic; original)."),
        ("../assets/figures/ml_fig_ppv_prevalence.png", "PPV vs prevalence with chapter LVO sens/spec (scientific; original)."),
        ("../assets/figures/ml_fig_clt_sampling.png", "Central limit theorem sampling means (scientific; original)."),
        ("../assets/figures/ml_fig_mle_bernoulli.png", "Bernoulli log-likelihood MLE surface (scientific; original)."),
    ],
    "04-clustering": [
        ("../assets/figures/ml_fig_clustering.png", "Clustering sketch with centroids (synthetic data; original)."),
        ("../assets/figures/ml_fig_elbow_wss.png", "Elbow plot of WSS vs k on the six-point toy set (original)."),
    ],
    "05-frequent": [
        ("../assets/figures/ml_fig_association_rules.png", "Support, confidence, and lift for the chapter’s five-transaction toy basket (original)."),
        ("../assets/figures/ml_fig_supervised_unsupervised_map.png", "Pattern mining sits on the unsupervised exploration path (original)."),
    ],
    "06-feature": [
        ("../assets/figures/ml_fig_feature_pipeline.png", "Feature pipeline: raw → impute → encode → scale → select → model (original)."),
        ("../assets/figures/ml_fig_leakage_timeline.png", "Feature timing versus prediction time — leakage trap (original)."),
    ],
    "07-dimensionality": [
        ("../assets/figures/ml_fig_pca.png", "Dimensionality reduction intuition along a dominant axis (original)."),
    ],
    "08-regression": [
        ("../assets/figures/ml_fig_calibration.png", "Reliability of numeric predictions matters as much as fit (original)."),
        ("../assets/figures/ml_fig_ols_fit.png", "OLS fit for the four-point NIHSS–volume example (original)."),
        ("../assets/figures/ml_fig_early_stopping.png", "Early stopping: validation bottoms while train still falls (synthetic; original)."),
        ("../assets/figures/ml_fig_gradient_noise.png", "Gradient noise: batch GD vs mini-batch SGD paths (synthetic; original)."),
        ("../assets/figures/ml_fig_regularization_path.png", "Lasso and Ridge regularization paths vs log10 λ (synthetic; original)."),
    ],
    "09-classification": [
        ("../assets/figures/ml_fig_confusion_annotated.png", "Confusion matrix with synthetic counts (original)."),
        ("../assets/figures/ml_fig_confusion_roc.png", "Confusion matrix and ROC for a synthetic classifier (original)."),
        ("../assets/figures/ml_fig_precision_recall.png", "Precision–recall vs ROC under class imbalance (synthetic; original)."),
    ],
    "10-neural": [
        ("../assets/figures/ml_fig_mlp.png", "Simple multilayer network diagram (original teaching graphic)."),
        ("../assets/figures/ml_fig_activations.png", "Activation functions: sigmoid, tanh, ReLU, leaky ReLU (original)."),
        ("../assets/figures/ml_fig_early_stopping.png", "Early stopping on train vs validation loss (synthetic; original)."),
    ],
    "11-self-supervised": [
        ("../assets/figures/ml_fig_pretrain_finetune.png", "Pretrain then fine-tune pipeline (original teaching graphic)."),
        ("../assets/figures/ml_fig_triplet_ssl.png", "Triplet loss with chapter worked numbers (original)."),
    ],
    "12-deep-learning": [
        ("../assets/figures/ml_fig_attention.png", "Self-attention weights for the three-token worked example (original)."),
        ("../assets/figures/ml_fig_mlp.png", "Deep models compose layered representations (original diagram)."),
        ("../assets/figures/ml_fig_site_shift.png", "Site shift in embedding space (synthetic; original)."),
    ],
    "13-reinforcement": [
        ("../assets/figures/ml_fig_rl_loop.png", "Agent–environment loop for sequential decisions (original)."),
        ("../assets/figures/ml_fig_value_iteration.png", "Value iteration on the two-state MDP (original)."),
    ],
    "14-making-lighter": [
        ("../assets/figures/ml_fig_distill_prune.png", "Distill/prune teaching sketch (original)."),
        ("../assets/figures/ml_fig_mlp.png", "Smaller deployed nets still need appraisal discipline (original)."),
    ],
    "15-graph": [
        ("../assets/figures/ml_fig_graph_toy.png", "Toy patient-similarity graph (original)."),
        ("../assets/figures/ml_fig_site_shift.png", "Graph/embedding geometry can drift across sites (original)."),
    ],
    "16-concepts": [
        ("../assets/figures/ml_fig_leakage_timeline.png", "Data challenges often reduce to time and shift (original)."),
        ("../assets/figures/ml_fig_site_shift.png", "Distribution shift between cohorts (original)."),
        ("../assets/figures/ml_fig_gradient_noise.png", "Gradient noise in optimization (synthetic; original)."),
    ],
    "17-closing": [
        ("../assets/figures/ml_fig_appraisal_scorecard.png", "Teaching scorecard for model appraisal (original)."),
        ("../assets/figures/ml_fig_metric_map.png", "Discrimination vs calibration vs utility (original)."),
        ("../assets/figures/ml_fig_lifecycle_deploy.png", "Senior practice lifecycle — design to drift (original)."),
        ("../assets/figures/ml_fig_decision_curve.png", "Decision curve: net benefit vs threshold (synthetic; original)."),
    ],
    "18-selected": [
        ("../assets/figures/ml_fig_metric_map.png", "Metric families: discrimination, calibration, utility (original)."),
        ("../assets/figures/ml_fig_appraisal_scorecard.png", "Appraisal orientation graphic (original)."),
        ("../assets/figures/ml_fig_glossary_families.png", "Glossary term families — teaching taxonomy (original)."),
        ("../assets/figures/ml_fig_ppv_prevalence.png", "PPV versus prevalence at fixed sens/spec (scientific; original)."),
    ],
}


def already_has(path: str, text: str) -> bool:
    return Path(path).name in text


def main() -> None:
    for p in sorted(CURR.glob("*.md")):
        if p.name == "index.md":
            continue
        text = p.read_text(encoding="utf-8")
        key = None
        for k in MAP:
            if p.name.startswith(k) or k in p.name:
                # match prefix more carefully
                if p.name.startswith(k.split("-")[0]) or any(p.name.startswith(k)):
                    key = k
                    break
        # better match
        key = None
        for k in MAP:
            if p.name.startswith(k):
                key = k
                break
        if key is None:
            continue
        block = []
        for rel, cap in MAP[key]:
            if already_has(rel, text):
                continue
            block.append(f"\n![{cap}]({rel})\n\n*{cap}*\n")
        if not block:
            print("skip", p.name)
            continue
        # insert after first ## Opening section or after title
        insert = "".join(block)
        if re.search(r"^## Opening\s*$", text, re.M):
            # after opening paragraph block
            m = re.search(r"(## Opening\n\n.*?\n\n)", text, re.S)
            if m:
                text = text[: m.end()] + insert + text[m.end() :]
            else:
                text = text + "\n" + insert
        else:
            # after first H1 block
            m = re.search(r"(^# .+\n\n)", text, re.M)
            if m:
                text = text[: m.end()] + insert + text[m.end() :]
            else:
                text = insert + text
        p.write_text(text, encoding="utf-8")
        print("embedded", p.name, len(block), "figures")


if __name__ == "__main__":
    main()
