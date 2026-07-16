<div class="ebook-hero ebook-hero--split" markdown="1">
<div class="hero-copy" markdown="1">
<p class="eyebrow">Open-source clinical ML field guide</p>

# Machine Learning &amp; AI for Neurologists

Follow the full path from cohort definition to monitored clinical use—without confusing a polished model score with trustworthy evidence or causal benefit.

<p class="meta">CC BY 4.0 · Educational only — not medical advice · <a href="https://github.com/rkalani1/ML">Source on GitHub</a></p>
</div>
<div class="hero-art">
<picture>
  <source media="(max-width: 600px)" srcset="assets/figures/ml_clinical_model_path_mobile.png" width="600" height="950">
  <img src="assets/figures/ml_clinical_model_path.png" alt="Clinical model path from cohort definition through validation and decision" width="960" height="280" fetchpriority="high" decoding="async">
</picture>
</div>
</div>

<div class="ebook-start" markdown="1">
[Begin with the clinical ML map](curriculum/01-basic-concepts-of-machine-learning-and-artificial-intelligence.md)
<a class="secondary" href="evidence-register.html">Evidence register</a>
<a class="secondary" href="https://rkalani1.github.io/CRIT-APP/">Critical appraisal companion</a>
</div>

## Choose a route

<div class="route-grid">
<a href="curriculum/00-mathematical-foundations-for-machine-learning.html"><span class="route-kicker">Foundations</span><strong>Rebuild the minimum math</strong><span>Use notation, probability, calculus, linear algebra, and optimization as a reference.</span></a>
<a href="curriculum/01-basic-concepts-of-machine-learning-and-artificial-intelligence.html"><span class="route-kicker">Orientation</span><strong>See the whole ML lifecycle</strong><span>Frame the task, split honestly, compare baselines, validate, and monitor.</span></a>
<a href="curriculum/09-classification.html"><span class="route-kicker">Evaluation</span><strong>Read performance correctly</strong><span>Connect thresholds, calibration, prevalence, and decision consequences.</span></a>
<a href="curriculum/16-concepts-and-challenges-of-working-with-data.html"><span class="route-kicker">Clinical use</span><strong>Plan for real-world failure</strong><span>Interrogate missingness, shift, fairness, leakage, workflow, and drift.</span></a>
</div>

## Build trust in layers

<figure class="feature-figure">
<img src="assets/figures/ml_validation_layers.png" alt="Four validation layers: data validity, internal validity, external validity, and clinical utility" width="959" height="430" loading="lazy" decoding="async">
<figcaption>Validation is cumulative. Later-stage performance cannot repair a biased cohort, contaminated split, or unstable label.</figcaption>
</figure>

| Layer | Core question | Evidence to inspect |
| --- | --- | --- |
| Data validity | Are the cohort, time zero, predictors, and labels credible? | Sampling, label process, missingness, measurement, provenance |
| Internal validity | Does performance survive honest resampling? | Leakage-free split, simple baseline, calibration, uncertainty |
| External validity | Does the model travel across place, time, and prevalence? | Temporal and geographic validation, subgroup calibration, shift analysis |
| Clinical utility | Does using the model improve decisions or outcomes? | Thresholds, net benefit, workflow study, prospective impact, monitoring |

## Contents

<ul class="chapter-list">
<li class="part">I · Foundations</li>
<li><a href="curriculum/00a-preface.html"><span class="num">→</span><span>Preface: how to use this field guide</span></a></li>
<li><a href="curriculum/00-mathematical-foundations-for-machine-learning.html"><span class="num">00</span><span>Mathematical Foundations for Machine Learning</span></a></li>
<li><a href="curriculum/01-basic-concepts-of-machine-learning-and-artificial-intelligence.html"><span class="num">01</span><span>Basic Concepts of Machine Learning and Artificial Intelligence</span></a></li>
<li><a href="curriculum/02-visualization.html"><span class="num">02</span><span>Visualization</span></a></li>
<li><a href="curriculum/03-probability-and-statistics.html"><span class="num">03</span><span>Probability and Statistics</span></a></li>
<li class="part">II · Classical learning</li>
<li><a href="curriculum/04-clustering.html"><span class="num">04</span><span>Clustering</span></a></li>
<li><a href="curriculum/05-frequent-itemset-mining-sequence-mining-and-information-retrieval.html"><span class="num">05</span><span>Frequent Itemset Mining, Sequence Mining, and Information Retrieval</span></a></li>
<li><a href="curriculum/06-feature-engineering.html"><span class="num">06</span><span>Feature Engineering</span></a></li>
<li><a href="curriculum/07-dimensionality-reduction-and-data-decomposition.html"><span class="num">07</span><span>Dimensionality Reduction and Data Decomposition</span></a></li>
<li><a href="curriculum/08-regression-analysis.html"><span class="num">08</span><span>Regression Analysis</span></a></li>
<li><a href="curriculum/09-classification.html"><span class="num">09</span><span>Classification</span></a></li>
<li class="part">III · Deep &amp; representation learning</li>
<li><a href="curriculum/10-neural-networks-and-deep-learning.html"><span class="num">10</span><span>Neural Networks and Deep Learning</span></a></li>
<li><a href="curriculum/11-self-supervised-deep-learning.html"><span class="num">11</span><span>Self-Supervised Deep Learning</span></a></li>
<li><a href="curriculum/12-deep-learning-models-and-applications-for-text-vision-and-audio.html"><span class="num">12</span><span>Deep Learning Models and Applications for Text, Vision, and Audio</span></a></li>
<li class="part">IV · Advanced systems</li>
<li><a href="curriculum/13-reinforcement-learning.html"><span class="num">13</span><span>Reinforcement Learning</span></a></li>
<li><a href="curriculum/14-making-lighter-neural-network-and-machine-learning-models.html"><span class="num">14</span><span>Making Lighter Neural Network and Machine Learning Models</span></a></li>
<li><a href="curriculum/15-graph-mining-algorithms.html"><span class="num">15</span><span>Graph Mining Algorithms</span></a></li>
<li><a href="curriculum/16-concepts-and-challenges-of-working-with-data.html"><span class="num">16</span><span>Concepts and Challenges of Working with Data</span></a></li>
<li class="part">V · Synthesis &amp; reference</li>
<li><a href="curriculum/17-closing-synthesis-senior-practice.html"><span class="num">17</span><span>Closing Synthesis: Senior Practice in Clinical Neurology and Epidemiology</span></a></li>
<li><a href="curriculum/18-selected-glossary.html"><span class="num">18</span><span>Selected Glossary</span></a></li>
</ul>
