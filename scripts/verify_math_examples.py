#!/usr/bin/env python3
"""Reproducible checks for a bounded cross-chapter sample of ML ebook numerics.

This is deliberately not an exhaustive proof of every numerical or mathematical
claim. It recomputes representative high-risk worked examples in every numbered
quantitative chapter; Chapter 17 is a qualitative synthesis without worked
arithmetic. Source anchors keep the calculations coupled to the examples they
are intended to protect.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path


def near(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def add(
    checks: list[tuple[str, bool, str]],
    name: str,
    actual: float,
    expected: float,
    tol: float = 1e-3,
) -> None:
    checks.append(
        (name, near(actual, expected, tol), f"actual={actual:.8g} expected={expected:.8g}")
    )


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    return (sensitivity * prevalence) / (
        sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    )


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    # Ch 0: Bayes, multivariable Taylor arithmetic, and stable log-sum-exp.
    add(checks, "ch00_bayes_ppv", ppv(0.90, 0.80, 0.05), 0.19148936, 1e-8)
    x, y = 1.1, 1.9
    add(checks, "ch00_quadratic_taylor", x * x + 3 * y * y + x * y, 14.13, 1e-12)
    logits = [800.0, 801.0, 802.0]
    m = max(logits)
    log_sum_exp = m + math.log(sum(math.exp(item - m) for item in logits))
    add(checks, "ch00_log_sum_exp", log_sum_exp, 802.407606, 1e-6)

    # Ch 1: model selection and bias-variance decomposition.
    add(checks, "ch01_candidate_a_train_error", 3 / 14, 0.21428571, 1e-8)
    add(checks, "ch01_candidate_a_validation_error", 2 / 6, 1 / 3, 1e-12)
    add(checks, "ch01_candidate_b_validation_error", 3 / 6, 0.5, 1e-12)
    add(checks, "ch01_bias_variance_model1_total", (-2) ** 2 + 0 + 1, 5.0, 1e-12)
    add(checks, "ch01_bias_variance_model2_total", 0 + ((7 - 10) ** 2 + (13 - 10) ** 2) / 2 + 1, 10.0, 1e-12)

    # Ch 2: descriptive statistics and the worked NIHSS-mRS correlation.
    age = [58, 72, 65, 81, 49, 77, 61, 70]
    nihss = [4, 12, 8, 18, 3, 22, 6, 15]
    mrs90 = [1, 3, 2, 5, 0, 4, 1, 3]
    mean_age = sum(age) / len(age)
    add(checks, "ch02_mean_age", mean_age, 66.625, 1e-12)
    age_ss = sum((value - mean_age) ** 2 for value in age)
    add(checks, "ch02_age_ss", age_ss, 773.875, 1e-12)
    add(checks, "ch02_age_sample_sd", math.sqrt(age_ss / 7), 10.51445, 1e-5)
    mx, my = sum(nihss) / 8, sum(mrs90) / 8
    numerator = sum((a - mx) * (b - my) for a, b in zip(nihss, mrs90))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in nihss) * sum((b - my) ** 2 for b in mrs90)
    )
    add(checks, "ch02_nihss_mrs_pearson", numerator / denominator, 0.93279625, 1e-8)

    # Ch 3: Bayes at the chapter's reference prevalence.
    add(checks, "ch03_lvo_ppv", ppv(0.85, 0.70, 0.20), 0.41463415, 1e-8)

    # Ch 4: k-means WCSS and the point-A silhouette.
    c1 = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]
    c2 = [(6.0, 5.0), (7.0, 6.0), (6.0, 7.0)]
    mu1 = tuple(sum(p[i] for p in c1) / len(c1) for i in range(2))
    mu2 = tuple(sum(p[i] for p in c2) / len(c2) for i in range(2))
    wcss1 = sum(sum((p[i] - mu1[i]) ** 2 for i in range(2)) for p in c1)
    wcss2 = sum(sum((p[i] - mu2[i]) ** 2 for i in range(2)) for p in c2)
    add(checks, "ch04_wcss_cluster1", wcss1, 4 / 3, 1e-12)
    add(checks, "ch04_wcss_total", wcss1 + wcss2, 4.0, 1e-12)
    a = c1[0]
    within = sum(math.dist(a, point) for point in c1[1:]) / 2
    between = sum(math.dist(a, point) for point in c2) / 3
    add(checks, "ch04_silhouette_point_a", (between - within) / max(within, between), 0.86378, 1e-5)

    # Ch 5: association-rule arithmetic and the Bloom-filter scaling law.
    add(checks, "ch05_assoc_support", 0.40, 0.40, 1e-12)
    add(checks, "ch05_assoc_confidence", 0.40 / 0.80, 0.5, 1e-12)
    add(checks, "ch05_assoc_lift", (0.40 / 0.80) / 0.40, 1.25, 1e-12)
    epsilon = 0.01
    bits_per_item = -math.log(epsilon) / math.log(2) ** 2
    add(checks, "ch05_bloom_bits_per_item_1pct", bits_per_item, 9.585058, 1e-6)
    m_500 = 500 * bits_per_item
    m_1000 = 1000 * bits_per_item
    add(checks, "ch05_bloom_linear_scaling", m_1000 / m_500, 2.0, 1e-12)

    # Ch 6: fit-on-train scaling and transform arithmetic.
    glucose = [90.0, 110.0, 130.0, 250.0]
    add(checks, "ch06_minmax_110", (110 - 90) / (250 - 90), 0.125, 1e-12)
    add(checks, "ch06_minmax_test_80", (80 - 90) / (250 - 90), -0.0625, 1e-12)
    glucose_mean = sum(glucose) / len(glucose)
    glucose_sd = math.sqrt(sum((item - glucose_mean) ** 2 for item in glucose) / 4)
    add(checks, "ch06_population_sd", glucose_sd, math.sqrt(3875), 1e-12)
    add(checks, "ch06_zscore_250", (250 - glucose_mean) / glucose_sd, 1.6867606, 1e-7)

    # Ch 7: PCA scores and exact reconstruction of rank-one data.
    score = (-2) * (2 / math.sqrt(5)) + (-1) * (1 / math.sqrt(5))
    add(checks, "ch07_pc1_score", score, -math.sqrt(5), 1e-12)
    projected = (
        (4 * -2 + 2 * -1) / 5,
        (2 * -2 + 1 * -1) / 5,
    )
    add(checks, "ch07_projection_x", projected[0], -2.0, 1e-12)
    add(checks, "ch07_projection_y", projected[1], -1.0, 1e-12)

    # Ch 8: OLS coefficients, residual scale, and the logistic update.
    add(checks, "ch08_ols_beta1", 23 / 13, 1.76923077, 1e-8)
    rss = 208 / 169
    add(checks, "ch08_ols_rss", rss, 1.23076923, 1e-8)
    add(checks, "ch08_ols_rse", math.sqrt(rss / 2), 0.784465, 1e-6)
    add(checks, "ch08_logistic_updated_probability", sigmoid(0.5625), 0.637031, 1e-6)

    # Ch 9: confusion-matrix metrics, Naive Bayes, and prevalence transport.
    tp, fp, fn, tn = 40, 10, 20, 130
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    add(checks, "ch09_accuracy", (tp + tn) / (tp + fp + fn + tn), 0.85, 1e-12)
    add(checks, "ch09_precision", precision, 0.80, 1e-12)
    add(checks, "ch09_f1", 2 * precision * recall / (precision + recall), 0.7272727, 1e-7)
    add(checks, "ch09_categorical_nb_posterior", 0.224 / (0.224 + 0.024), 0.9032258, 1e-7)
    add(checks, "ch09_lvo_ppv_unselected_ed", ppv(0.90, 0.90, 0.03), 0.2177419, 1e-7)

    # Ch 10: the mini-batch sigmoid/BCE update and its requested loss check.
    p1, p2 = sigmoid(0.5), sigmoid(-1.0)
    old_loss = (-math.log(p1) - math.log(1 - p2)) / 2
    add(checks, "ch10_sigmoid_p1", p1, 0.6224593, 1e-7)
    add(checks, "ch10_sigmoid_p2", p2, 0.2689414, 1e-7)
    add(checks, "ch10_mean_bce", old_loss, 0.3936693, 1e-7)
    gz1, gz2 = p1 - 1, p2
    w_new = [0.5 - 0.5 * (gz1 / 2), -1.0 - 0.5 * (gz2 / 2)]
    b_new = 0.0 - 0.5 * ((gz1 + gz2) / 2)
    add(checks, "ch10_updated_w1", w_new[0], 0.5943852, 1e-7)
    add(checks, "ch10_updated_w2", w_new[1], -1.0672354, 1e-7)
    add(checks, "ch10_updated_bias", b_new, 0.027149812, 1e-9)
    new_p1, new_p2 = sigmoid(w_new[0] + b_new), sigmoid(w_new[1] + b_new)
    new_loss = (-math.log(new_p1) - math.log(1 - new_p2)) / 2
    checks.append(("ch10_recomputed_loss_decreases", new_loss < old_loss, f"old={old_loss:.8g} new={new_loss:.8g}"))

    # Ch 11: contrastive, triplet, and VAE ELBO arithmetic.
    d12_sq = (1.0 - 0.8) ** 2 + (0.0 - 0.6) ** 2
    add(checks, "ch11_contrastive_distance", math.sqrt(d12_sq), math.sqrt(0.4), 1e-12)
    add(checks, "ch11_contrastive_loss", d12_sq + max(0.0, 1.0 - 2.0) ** 2, 0.4, 1e-12)
    add(checks, "ch11_hard_triplet_loss", max(0.0, 0.5 - 0.5 + 0.2), 0.2, 1e-12)
    vae_kl = 0.5 * (1.0**2 + 0.5**2 - 1 - math.log(0.5**2))
    add(checks, "ch11_vae_kl", vae_kl, 0.8181472, 1e-7)
    add(checks, "ch11_vae_elbo", -0.5 - vae_kl, -1.3181472, 1e-7)

    # Ch 12: scaled attention weights and output coordinates.
    query = [1.0, 0.0]
    keys = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    scores = [sum(qi * ki for qi, ki in zip(query, key)) / math.sqrt(2) for key in keys]
    score_max = max(scores)
    exps = [math.exp(item - score_max) for item in scores]
    weights = [item / sum(exps) for item in exps]
    add(checks, "ch12_attention_w0", weights[0], 0.434, 0.001)
    add(checks, "ch12_attention_w1", weights[1], 0.214, 0.001)
    add(checks, "ch12_attention_w2", weights[2], 0.351, 0.001)
    output = [sum(weights[i] * keys[i][j] for i in range(3)) for j in range(2)]
    add(checks, "ch12_attention_output_x", output[0], 0.680, 0.001)
    add(checks, "ch12_attention_output_y", output[1], 0.460, 0.001)

    # Ch 13: Bellman backup and exact values in the two-state MDP.
    add(checks, "ch13_left_backup", 0 + 0.9 * 5, 4.5, 1e-12)
    add(checks, "ch13_right_backup", 1 + 0.9 * 3, 3.7, 1e-12)
    v2 = 2 / (1 - 0.9)
    v1 = 0.9 * v2
    add(checks, "ch13_two_state_v1", v1, 18.0, 1e-12)
    add(checks, "ch13_two_state_v2", v2, 20.0, 1e-12)

    # Ch 14: parameter counts, Huffman length/entropy, and CNN sizing.
    mlp_params = 20 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2
    add(checks, "ch14_mlp_parameters", mlp_params, 5634, 1e-12)
    add(checks, "ch14_mlp_float32_bytes", mlp_params * 4, 22536, 1e-12)
    probabilities = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]
    lengths = [1, 2, 3, 4, 5, 5]
    add(checks, "ch14_huffman_average_bits", sum(p * length for p, length in zip(probabilities, lengths)), 2.25, 1e-12)
    entropy = -sum(p * math.log2(p) for p in probabilities)
    add(checks, "ch14_huffman_entropy", entropy, 2.2007968, 1e-7)
    cnn_params = 160 + 4640 + 131136 + 130
    add(checks, "ch14_tiny_cnn_parameters", cnn_params, 136066, 1e-12)

    # Ch 15: converged PageRank and shortest transfer path.
    rank = [0.25] * 4
    alpha, teleport = 0.85, 0.0375
    for _ in range(1000):
        rank = [
            alpha * rank[2] + teleport,
            alpha * 0.5 * rank[0] + teleport,
            alpha * (0.5 * rank[0] + rank[1] + rank[3]) + teleport,
            teleport,
        ]
    add(checks, "ch15_pagerank_a", rank[0], 0.3725, 5e-5)
    add(checks, "ch15_pagerank_b", rank[1], 0.1958, 5e-5)
    add(checks, "ch15_pagerank_c", rank[2], 0.3941, 5e-5)
    add(checks, "ch15_pagerank_d", rank[3], 0.0375, 1e-12)
    add(checks, "ch15_pagerank_mass", sum(rank), 1.0, 1e-12)
    add(checks, "ch15_shortest_transfer_minutes", min(70, 25 + 35, 20 + 80, 20 + 25 + 50), 60, 1e-12)

    # Ch 16: site-drift PPV and chance-corrected agreement.
    add(checks, "ch16_site_a_ppv", ppv(0.90, 0.80, 0.20), 0.5294118, 1e-7)
    add(checks, "ch16_site_b_ppv", ppv(0.90, 0.80, 0.05), 0.1914894, 1e-7)
    add(checks, "ch16_site_b_shifted_specificity_ppv", ppv(0.90, 0.70, 0.05), 0.1363636, 1e-7)
    observed, expected = 0.85, 0.50
    add(checks, "ch16_cohen_kappa", (observed - expected) / (1 - expected), 0.70, 1e-12)

    # Ch 18: the prevalence-to-PPV reference table.
    for prevalence, expected_ppv in [
        (0.05, 0.129771),
        (0.10, 0.239437),
        (0.20, 0.414634),
        (0.40, 0.653846),
        (0.60, 0.809524),
    ]:
        add(
            checks,
            f"ch18_ppv_prevalence_{prevalence:.2f}",
            ppv(0.85, 0.70, prevalence),
            expected_ppv,
            1e-6,
        )

    # One file-local anchor per covered chapter prevents calculations from
    # silently outliving the examples they are intended to verify.
    root = Path(__file__).resolve().parents[1]
    source_expectations = {
        "ch00": ("00-mathematical-foundations-for-machine-learning.md", ("802.408", "14.13")),
        "ch01": ("01-basic-concepts-of-machine-learning-and-artificial-intelligence.md", ("3 / 14", "5 for Model 1 versus 10 for Model 2")),
        "ch02": ("02-visualization.md", ("0.933", "773.875")),
        "ch03": ("03-probability-and-statistics.md", ("0.4146",)),
        "ch04": ("04-clustering.md", ("total J = 4", "0.864")),
        "ch05": ("05-frequent-itemset-mining-sequence-mining-and-information-retrieval.md", ("m therefore grows linearly with n", "1.25")),
        "ch06": ("06-feature-engineering.md", ("−0.0625", "62.25")),
        "ch07": ("07-dimensionality-reduction-and-data-decomposition.md", ("−√5", "Between-cluster distance is not globally calibrated")),
        "ch08": ("08-regression-analysis.md", ("23/13", "0.637")),
        "ch09": ("09-classification.md", ("0.903", "0.727")),
        "ch10": ("10-neural-networks-and-deep-learning.md", ("0.5944", "0.0272")),
        "ch11": ("11-self-supervised-deep-learning.md", ("0.818", "−1.318")),
        "ch12": ("12-deep-learning-models-and-applications-for-text-vision-and-audio.md", ("[0.434, 0.214, 0.351]", "[0.680, 0.460]")),
        "ch13": ("13-reinforcement-learning.md", ("V^*(s_2)=2/(1-0.9)=20", "V^*(s_1)=0+0.9\\times20=18")),
        "ch14": ("14-making-lighter-neural-network-and-machine-learning-models.md", ("2.2008", "P=5634")),
        "ch15": ("15-graph-mining-algorithms.md", ("r* = (0.3725, 0.1958, 0.3941, 0.0375)", "60 minutes via A-P-Z")),
        "ch16": ("16-concepts-and-challenges-of-working-with-data.md", ("0.18/0.34 ≈ 0.529", "0.35/0.50 = 0.70")),
        "ch18": ("18-selected-glossary.md", ("| 0.60 | ≈0.63 | ≈0.81 |",)),
    }
    curriculum = root / "docs" / "curriculum"
    for chapter, (filename, anchors) in source_expectations.items():
        source = (curriculum / filename).read_text(encoding="utf-8")
        missing = [anchor for anchor in anchors if anchor not in source]
        checks.append(
            (
                f"{chapter}_source_anchor",
                not missing,
                "present" if not missing else f"missing={missing}",
            )
        )

    failed = 0
    for name, ok, detail in checks:
        print(("PASS" if ok else "FAIL"), name, detail)
        if not ok:
            failed += 1
    print(
        f"BOUNDED_SAMPLE chapters=18/19 checks={len(checks)} "
        "excluded=ch17_qualitative_synthesis"
    )
    if failed:
        print("FAILED", failed)
        return 1
    print("ALL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
