# Chapter 2. Visualization


![02 Viz Anatomy](../assets/figures/02_viz_anatomy.png)


## Opening

![Visualization hygiene (original).](../assets/figures/ml_fig_viz_hygiene.png)

*Clear baseline versus truncated axis (original teaching graphic).*


A quality dashboard shows a green line for door-to-needle. A density plot of NIHSS looks normal until you split by transfer status. Visualization is not decoration—it is the first place site shift and selection bias become visible.


![Calibration view as a visual truth check (original teaching graphic).](../assets/figures/ml_fig_calibration.png)

*Calibration view as a visual truth check (original teaching graphic).*
## Learning Objectives

State the historical purpose of statistical graphics and treat visualization as analytic method, not decoration.

Construct and interpret basic charts: pie, bar, histogram, line, scatter, and broken-axis displays—with caution where needed.

Use multidimensional views including surface, contour, area, radar, heatmap, calendar, timeline, box, and violin plots.

Read hierarchical visualizations (dendrogram, treemap, sunburst) and network views (node-link, arc, chord).

Apply deviation and change charts: divergent bars, dumbbells, waterfalls, and beeswarms.

Judge when 3-D graphics help versus when they distort, and design choropleth and bubble maps responsibly.

Explain multi-dimensional scaling as a visualization of similarity structure, name the misread risks of nonlinear embeddings, and choose encodings that match perception.

Build and read model-diagnostic displays—residual-versus-fitted plots, reliability/calibration diagrams, and row- and column-normalized confusion-matrix heatmaps—and state when each misleads.

Detect and avoid misleading visualizations; connect figures to epidemiologic Table 1 and model diagnostics in stroke research.

## Background and History

Visualization is the mapping of data to visual channels so that human perception can detect structure, anomaly, and comparison more efficiently than by scanning tables alone. Long before machine learning, scientists used graphics to argue about evidence: Playfair’s commercial and political charts in the late eighteenth century popularized line, bar, and pie forms; Nightingale’s polar area diagrams made sanitary reform vivid; Snow’s cholera map linked cases to a water pump; Minard’s flow map of Napoleon’s march combined geography, time, and attrition. Twentieth-century statisticians—notably Tukey—reframed graphics as tools for exploratory data analysis (EDA): the goal is to see what you did not expect, not only to decorate what you already believe.

Modern visualization inherits three intertwined purposes. First, exploration: surface data quality problems, distributional shape, and subgroup structure before any model is trained. Second, diagnosis: residual plots, calibration diagrams, and confusion-matrix heatmaps expose model failure modes that a single AUC hides. Third, communication: figures that a stroke attending, a data manager, and a methods reviewer can interrogate in the same meeting. For a neurologist–epidemiologist, the stakes are concrete. A skewed NIH Stroke Scale distribution with a pile of zeros is a mixture of mild and severe syndromes that will break a naive linear model. A scatter of age against door-to-needle time colored by hospital site can reveal a process problem long before an accuracy number does.

This chapter surveys a practical catalog of chart types aligned with common analytic needs, then treats encoding choice, multi-dimensional scaling, and graphical lies. Treat each chart as a claim about structure in data. Good charts force better questions; bad charts manufacture certainty.

Historical roots: Playfair, Nightingale, Snow, Minard; Tukey and EDA culture.

Three jobs: explore data, diagnose models, communicate uncertainty and comparisons.

Clinical ML stake: see quality, shift, imbalance, and error slices before trusting scores.

## Basic Visualizations

!!! note "Figure concept (text diagram) 2.1"

    A visual vocabulary of six core chart types on small toy stroke data: a bar chart of subtype counts, a histogram of a bimodal NIHSS distribution, a line chart of thrombolysis rate by year, a scatter of age versus NIHSS, side-by-side box plots of NIHSS by large-vessel-occlusion status, and violin plots of length of stay whose full shape reveals a bimodality a box would hide. Each chart type is matched to the question it answers.

### Pie chart

A pie chart partitions a whole into slices whose angles are proportional to category shares. Use pies only for a small number of mutually exclusive parts of a single total (ideally ≤ 5 slices) when the message is composition, not precise comparison. Humans compare angles and areas poorly relative to position along a common scale; ordered bar charts almost always support finer comparisons. In stroke reports, a pie of discharge disposition may be acceptable for a slide; a pie of twenty ICD chapters is not. Never use 3-D pies: perspective distorts slice area.

### Bar chart and histogram

Bar charts encode a quantitative value by bar length for discrete categories (counts of TOAST subtypes, mean door-to-needle by site). Sort bars by clinical taxonomy or by magnitude, not by accidental alphabetical order, unless the category axis is ordinal. Start the value axis at zero for count and most rate comparisons so that length remains proportional to magnitude.

A histogram partitions a continuous (or fine-grained ordinal) variable into bins and displays counts or densities per bin. Let x_1, …, x_n be observations. Choose bin edges covering the observed or clinically meaningful range. The count in bin j is the number of points falling in that interval; a density histogram scales so that bar areas sum to one. Bin width is a design choice: too few bins hide bimodality (for example, mild versus severe NIHSS mixtures); too many amplify noise in small cohorts. When comparing groups of unequal size, prefer density-normalized histograms or overlaid kernel densities so that sample size does not masquerade as shape.

### Line chart

Line charts connect ordered observations—usually over time—with line segments, emphasizing change and trend. They are the default for rates by year, learning curves by epoch, and vital-sign trajectories when sampling is regular enough. Multiple series should use distinguishable styles, not color alone. Beware irregular time gaps: connecting sparse points with straight lines implies linear interpolation that may be false. For event-time data with censoring, specialized survival curves (Kaplan–Meier) are more honest than naive mean trajectories.

### Scatter plot

A scatter plot places one continuous variable on the horizontal axis and another on the vertical axis, with each point an observation (or a jittered ordinal). Patterns to name explicitly include linear trend, curved relationship, heteroscedasticity (fan-shaped residual spread), clusters, and jointly extreme outliers. Encoding a third variable with color, shape, or small multiples often reveals that an apparent overall slope is a mixture of group-specific slopes—the visual form of confounding and of Simpson’s paradox. Scatter plots remain the most important bivariate EDA tool in clinical ML feature analysis.

### Broken axis chart

A broken (or discontinuous) axis omits a middle range of values, often to zoom on small differences while still showing a large outlier or a distant baseline. Broken axes are dangerous: viewers may miss the break and over-read differences. If you must break an axis, mark the discontinuity boldly, state it in the caption, and consider whether two panels (overview + zoom) would be clearer. For publication and quality dashboards, prefer intact axes or explicitly labeled insets over subtle breaks.

## More Than One-Dimensional Data Visualizations

### Surface plot

A surface plot displays a scalar function z = f(x, y) as a surface in three dimensions—useful for visualizing smooth response surfaces, kernel density estimates in 2-D, or loss landscapes in toy optimization examples. Surfaces help intuition when interaction between two continuous predictors matters, but viewpoint and occlusion can hide regions. Always rotate interactively when possible, or supplement with contours.

### Contour plot

Contour plots draw level sets of f(x, y)—curves where the function is constant—analogous to elevation maps. They avoid some 3-D occlusion problems while showing ridges, basins, and saddle structure. In ML, contours appear for 2-D density estimates, classifier decision regions (with probability contours), and optimization diagnostics. Label contour values; rainbow color maps can create false boundaries.

### Area plot

Area plots fill the region under a line (or between lines) to emphasize cumulative magnitude over an ordered axis. Stacked area charts show composition over time (for example, ischemic versus hemorrhagic admissions by month) but become hard to read for middle series whose baselines wander. Prefer stacked areas for a few categories with a meaningful total; otherwise use small multiples of line charts.

### Radar chart

Radar (spider) charts place multiple axes around a circle and connect scores into a polygon—common for profile comparison across a handful of scales (domain scores, multi-criterion model comparison). They struggle when axes are not comparable, when many variables clutter the polygon, or when area is misread as an overall score (area depends on axis order). Use sparingly; a parallel-coordinates plot or a simple table often communicates profiles more honestly.

### Heatmap

A heatmap maps a matrix of values to color. Correlation heatmaps survey redundancy among features; confusion-matrix heatmaps show class-wise error structure; calendar-style heatmaps (below) show intensity by day. Choose perceptually uniform sequential colormaps for ordered magnitudes; use diverging maps when a meaningful midpoint exists (zero correlation, zero residual). Annotate cells when the matrix is small enough to read. Heatmaps are overviews—drill into key pairs with scatter plots.

!!! note "Figure concept (text diagram) 2.2"

    Three-class confusion-matrix heatmap for a TOAST subtype classifier on an external test set (n = 130), shaded with a sequential indigo colormap built from SEQ_INDIGO. Each cell shows the count with its row-normalized recall in parentheses (diagonal recall 0.75, 0.70, 0.75); overall accuracy is 95/130 = 0.731 and column-normalized precision is LAA 0.68, CE 0.74, SVO 0.77. Raw-count shading tracks the largest classes, not the largest error rates.

### Calendar plot

Calendar plots arrange daily values on a calendar grid, coloring each day by a metric: arrivals, door-to-needle breaches, or model alert counts. They reveal weekly seasonality, holiday effects, and single-day shocks better than a plain time series when the calendar structure itself matters for staffing. Keep the color scale fixed when comparing months or sites.

### Timeline plot

Timelines place events along a temporal axis: symptom onset, last known well, ED arrival, imaging, thrombolysis, puncture, reperfusion. For a single patient, a timeline supports process review; for cohorts, aggregated timelines or Gantt-like swimmer plots show sequence heterogeneity. Align on index time when comparing pathways. Machine learning features derived from timestamps should be checked against timelines to catch leakage and impossible orders.

### Box plot and violin plot

A box plot summarizes a continuous variable with the five-number skeleton: lower whisker, first quartile Q1, median, third quartile Q3, and upper whisker. The interquartile range is IQR = Q3 − Q1; a common whisker rule extends to the most extreme points still inside [Q1 − 1.5·IQR, Q3 + 1.5·IQR], with more extreme points drawn individually. Box plots excel at side-by-side comparison across categories (NIHSS by LVO status; length of stay by site).

Box plots hide multimodality: two clumps can produce one wide box with a median in an empty middle. Violin plots add a kernel density shape to each group, revealing multimodality and skewness at the cost of slightly more visual complexity. When multimodality is clinically plausible—bimodal age mixing young dissection with older atherosclerosis—pair boxes with violins or faceted histograms. Report median and IQR in text when distributions are skewed; mean and SD only when symmetry makes them representative.

## Hierarchical Visualizations

### Dendrogram

A dendrogram displays the merge (or split) history of hierarchical clustering: leaves are observations or items; internal nodes are merges; join height encodes the linkage distance at which clusters fused. Cutting the tree at a height yields a partition. Dendrograms are excellent for small-to-moderate n and for taxonomy-like structure (lab panels, ICD hierarchies when pre-defined). They invite overinterpretation of unstable branches—bootstrap or multi-site replication before clinical storytelling (see Chapter 4).

### Treemap

Treemaps recursively partition a rectangle into nested rectangles whose areas encode a quantitative variable (volume, cost, patient counts) within a hierarchy (service line → diagnosis group → DRG). They use space efficiently for large hierarchies compared with node-link trees. Weaknesses include difficulty comparing non-aligned areas and poor encoding of small nodes. Use clear borders and hover/drill-down in interactive settings; for print, limit depth.

### Sunburst

A sunburst chart is a radial hierarchical display: concentric rings represent depth in the tree; arc length encodes magnitude. Sunbursts communicate root-to-leaf composition (for example, all stroke admissions → ischemic/hemorrhagic → subtypes). Like pies, they strain angular comparison; they work best interactively with filtering. Prefer sunbursts when hierarchical part-to-whole relationships are the primary message and category counts are not tiny.

## Graph and Network Visualizations

### Node-link diagram

Node-link diagrams draw entities as nodes and relations as edges—the default graph picture. Layout algorithms (force-directed, hierarchical, circular) strongly affect readability. Use them for referral networks, comorbidity co-occurrence graphs (with edge thresholds), and knowledge graphs. For large graphs, edge bundling, sampling, or aggregated community views prevent hairballs. Node size and color can encode degree, centrality, or attributes; keep legends explicit.

### Arc diagram

Arc diagrams place nodes along a line (often ordered by time, genome position, or a seriation of similarity) and draw edges as arcs above the line. They reduce clutter relative to 2-D force layouts when a meaningful linear order exists—for example, transitions among ordered care states, or correlations among ordered brain regions. Arc height or thickness can encode weight. They struggle when many long-range links cross.

### Chord diagram (circular link)

Chord diagrams arrange nodes on a circle and draw ribbons between pairs to encode flow or association strength—classic for origin–destination flows (transfer from spoke to hub hospitals) and for asymmetric confusion patterns among many classes. Ribbons can mislead about direction unless arrows or half-chords are used carefully. For accessibility and print clarity, a sorted origin–destination heatmap is often a better companion or substitute.

## Deviation and Change Visualizations

### Divergent bar plot

Divergent (diverging) bar plots extend bars left and right from a neutral baseline—ideal for showing improvement versus deterioration, surplus versus deficit, or Likert-style agreement. In quality improvement, a divergent bar of change in metric by site (post − pre) makes winners and laggards obvious. Align the neutral baseline; use a diverging color scale only if it adds redundancy, not confusion.

### Dumbbell plot

Dumbbell (connected dot) plots show two values per category as dots joined by a segment—before/after risk, male/female rates, model A versus model B AUROC by subgroup. They emphasize the difference while still showing absolute levels. Sort categories by difference or by baseline. Dumbbells often beat paired bars for change communication in clinical papers.

### Waterfall chart

Waterfall charts decompose a net change into sequential signed contributions: starting rate, then additions and subtractions, ending at the final rate. Finance popularized them; epidemiology can use them for stepwise risk adjustment or for explaining how a quality metric moved after a bundle of interventions. Keep the number of steps modest and the intermediate baselines visible.

### Beeswarm plot

Beeswarm plots place individual points along a categorical axis with repulsion so that points do not fully overlap—preserving raw data density better than a pure box plot while still showing groups. They are excellent for small-to-moderate n (trial secondary outcomes, single-center continuous scores). For very large n, switch to violins, sina plots with transparency, or histograms.

## Three-Dimensional Data Visualizations

Three-dimensional scatter plots, surfaces, and bars can display an extra quantitative axis, but perspective, occlusion, and display-medium limitations often destroy accurate comparison. Use 3-D when the third dimension is intrinsic (anatomical space, true volumetric imaging isosurfaces) or when interactive rotation is available to the reader. For static slides and papers, prefer 2-D projections, small multiples, color encodings, or facets over obligatory 3-D bar charts of categorical statistics—the latter almost always distort perceived height.

If you render 3-D clinical imaging-derived features for EDA, pair them with 2-D slices and quantitative summaries. Depth cues that look persuasive can still mislead about distances. When teaching loss landscapes or mixture densities, 3-D surfaces plus contour projections together work better than either alone.

## Geographical Map Visualizations

### Choropleth

Choropleth maps shade administrative regions by a statistic: age-adjusted stroke incidence, thrombolysis rate, or model coverage. They are powerful for spatial epidemiology and health-services research. Hazards include: large polygons dominating attention regardless of population; unstable rates in small areas; and arbitrary cut-points in color bins that manufacture clusters. Prefer normalized rates over raw counts, show uncertainty or suppress small cells, use perceptually uniform scales, and consider cartograms or inset tables when polygon area severely mismatches population.

### Bubble map

Bubble maps place scaled circles at geographic coordinates—facility locations sized by volume, colored by outcome. They preserve point location better than choropleths for hospital-level data. Overplotting in dense metro areas requires jitter, clustering markers, or zoom. Encode size to area carefully (radius scaled to sqrt of magnitude) so that perceptual area matches data. Provide legends with example bubble sizes.

## Multi-Dimensional Scaling and Other Multidimensional Views

When each observation is a high-dimensional vector, direct scatter plots of raw axes fail. Multi-dimensional scaling (MDS) finds low-dimensional coordinates (usually 2-D or 3-D) such that distances between embedded points approximate dissimilarities δ_ij in the original space. Classical MDS works from a squared distance matrix via eigendecomposition; metric and nonmetric variants optimize a stress criterion for different assumptions about how δ relates to embedded distance. Stress is a normalized measure of the mismatch between the original dissimilarities δ_ij and the embedded distances d_ij (smaller is better); report it, because a visually tidy 2-D map can still carry high stress when the data’s intrinsic dimensionality exceeds two, and readers will over-trust a clean-looking layout. A Shepard diagram—embedded distance plotted against original dissimilarity—shows directly whether the picture preserves the input distances or badly compresses them. Use MDS to visualize patient similarity, site similarity, or item structure when you already have a scientifically motivated distance (including Gower distances, which combine mixed continuous, ordinal, and binary clinical variables into a single metric).

!!! note "Figure concept (text diagram) 2.3"

    The same 2-D data rendered as two neighbor embeddings at different perplexity settings. At low perplexity (left) the three clusters appear far apart, with a tight blob and a diffuse blob of equal count; at high perplexity (right) the same points blend together. Apparent separation, gap width, and blob area are all artifacts of tuning, so distances in such maps must not be over-read.

PCA is a related linear projection maximizing variance rather than preserving arbitrary distances; t-SNE and UMAP emphasize local neighborhoods, which makes three of their features routinely misread. First, cluster sizes carry no meaning: a tight blob and a diffuse blob can hold equal counts, so relative area says nothing about prevalence. Second, between-cluster distances are not faithful: two clusters drawn side by side need not be more similar than two drawn far apart. Third, apparent gaps shift with tuning (the perplexity or n_neighbors setting) and with the random seed, so a separation seen in one run can dissolve in the next. Treat nonlinear embeddings as hypothesis generators: rerun several seeds and settings, color by batch or site to detect shift, color by known labels to audit leakage, and never claim that a pretty 2-D map alone validates a phenotype. Pair embeddings with quantitative stability checks and, when labels exist for audit, supervised metrics on held-out data.

Parallel coordinates, small-multiple sparklines, and linked brushing in interactive tools extend the toolbox for moderate d. For large n, prefer aggregated views: hexbin or 2-D density contours instead of overplotted scatters.

## Choosing Encodings and Avoiding Misleading Visualizations

Human perception ranks visual channels roughly as: position along a common scale, length, angle or slope, area, volume, then color saturation or hue (highly context-dependent). Encode the most important quantitative comparisons with position or length. Reserve color for categorical strata with a colorblind-safe palette and a modest number of levels. Size can carry an extra quantitative variable but collides under overplotting. Prefer small multiples over overloading a single panel with every channel.

!!! note "Figure concept (text diagram) 2.4"

    The truncated-baseline deception. The same four door-to-needle medians (60, 58, 59, 57 minutes) are drawn with a truncated y-axis starting at 55 minutes (left, misleading) and with an honest zero baseline (right). Only the axis differs, yet the truncated panel inflates a 3-minute spread into a dramatic difference; the printed values are identical in both panels.

Three deceptions deserve mechanical understanding, because they fool careful readers, not only careless ones. First, the truncated baseline: when a bar’s value axis does not start at zero, bar length no longer encodes magnitude, so a door-to-needle change from 60 to 58 minutes can be drawn to look like a near-halving simply by starting the axis at 55. A bar implies a zero; if you must zoom, switch to a dot or line that does not carry a length cue, and label the zoom. Second, dual y-axes: overlaying two series on independently chosen left and right scales invites the eye to read their crossings and co-movement as a relationship, but there is no canonical alignment—by rescaling one axis you can make stroke volume and mortality appear to track together or diverge at will, and the “crossover point” is a pure artifact of the chosen limits. Prefer two aligned panels sharing an x-axis. Third, cherry-picking the window: a metric that wanders will always contain some start-and-end pair that shows the story you want, so reporting door-to-needle “improvement” measured from the single worst month manufactures a trend that the full series would erase. State the whole period, fix the comparison window before looking, and show the surrounding context.

Start count/rate bars at zero unless you explicitly justify a zoomed scale and label it loudly.

Avoid dual y-axes that force false alignment between unrelated series; use aligned small multiples.

Declare log scales and transforms in the axis title, not only in a footnote.

Prefer perceptually uniform colormaps for sequential data; avoid rainbow scales for magnitudes.

Show uncertainty (intervals, bootstrap bands) when comparing estimates across groups.

Pair color with redundant non-color encodings (shape, pattern) for critical categories.

Do not cherry-pick time windows or y-limits to manufacture a policy narrative.

Refuse 3-D categorical bars and exploded 3-D pies in scientific communication.

Accessibility expands who can use your work correctly: sufficient luminance contrast, large fonts for projection and photocopy, descriptive captions and alt text, and tabular companions for key plotted values. Accessibility is quality control under time pressure, not only compliance.

## Worked Example: Tiny Stroke Dataset (Age, NIHSS, mRS)

We work a fully arithmetic EDA example on eight hypothetical acute ischemic stroke encounters. Features: age in years, admission NIHSS (higher is worse). Ordinal outcome: 90-day mRS on a 0–6 scale. The goal is not inference from n = 8; the goal is to practice summary computation, plot selection, and honest reading of structure.

```
# Toy cohort (hypothetical), id: 1..8
age = [58, 72, 65, 81, 49, 77, 61, 70]
nihss = [4, 12, 8, 18, 3, 22, 6, 15]
mrs90 = [1, 3, 2, 5, 0, 4, 1, 3]
n = 8
mean_age = sum(age)/n # 533/8 = 66.625
mean_nihss = sum(nihss)/n # 88/8 = 11.0
mean_mrs = sum(mrs90)/n # 19/8 = 2.375
```

Age values sorted: 49, 58, 61, 65, 70, 72, 77, 81. Sum = 533; mean = 66.625 years. Median is the average of the 4th and 5th order statistics: (65+70)/2 = 67.5. Range = 32 years. Using sample variance with divisor n−1 = 7, the sum of squared deviations from 66.625 is 773.875 (equivalently Σage² − n·mean² = 36285 − 8·66.625² = 36285 − 35511.125), so s² ≈ 110.55 and s ≈ 10.51 years.

NIHSS sorted: 3, 4, 6, 8, 12, 15, 18, 22. Mean = 11.0; median = 10.0. The mean slightly exceeds the median, consistent with modest right skew from the severe tail. Q1 = median of {3,4,6,8} = 5; Q3 = median of {12,15,18,22} = 16.5; IQR = 11.5. Upper fence Q3 + 1.5·IQR = 33.75, so maximum 22 is not flagged by the 1.5·IQR rule—a reminder that fence rules are heuristics and clinically extreme values can lie inside fences in a severe-enriched sample.

Primary EDA figure: scatter of age (x) versus NIHSS (y), with point label or color by mRS90. Compute Pearson r between NIHSS and mRS90. With means 11 and 2.375, the eight cross-products of centered values (NIHSS centered by 11, mRS by 2.375) are 9.625 + 0.625 + 1.125 + 18.375 + 19.0 + 17.875 + 6.875 + 2.5 = 76.0. Sum of squared NIHSS deviations = 334; sum of squared mRS deviations = 19.875. Then r = 76.0 / sqrt(334 × 19.875) = 76.0 / sqrt(6638.25) ≈ 76.0 / 81.48 ≈ 0.933. A strong linear association in this toy set matches clinical expectation that higher admission severity associates with worse 90-day function—but n = 8 forbids inferential swagger. Plot choice still matters: the scatter plus a side-by-side box of NIHSS by dichotomized mRS (0–2 vs 3–6) would be the communication package for a lab meeting.

```
import math
nihss = [4, 12, 8, 18, 3, 22, 6, 15]
mrs90 = [1, 3, 2, 5, 0, 4, 1, 3]
mx = sum(nihss)/8
my = sum(mrs90)/8
num = sum((x-mx)*(y-my) for x, y in zip(nihss, mrs90)) # 76.0
sxx = sum((x-mx)**2 for x in nihss) # 334.0
syy = sum((y-my)**2 for y in mrs90) # 19.875
r = num / math.sqrt(sxx*syy) # ~0.933
```

## Model-Diagnostic Visualization in Practice

Raw-data EDA is only half of a visualization practice for machine learning. After a model is fit, graphics become the fastest way to see whether the mathematical optimum is clinically usable. For regression and risk scores, define residuals e_i = y_i − ŷ_i (or deviance residuals for generalized linear models). A residual-versus-fitted plot should look like unstructured noise around zero. A smooth U-shape suggests missing nonlinearity or an omitted interaction—for example, age effects on outcome that differ by reperfusion status. A funnel shape, with residual spread growing as fitted values grow, indicates heteroscedasticity: prediction intervals that assume constant variance will be too narrow for severe patients and too wide for mild ones. Plotting residuals against individual predictors (door-to-needle time, glucose, systolic pressure) can reveal structure invisible in the marginal residual cloud.

!!! note "Figure concept (text diagram) 2.5"

    Reliability (calibration) diagram comparing a well-calibrated model, whose binned points track the 45° reference line, with an over-confident model, whose curve is too shallow and crosses the diagonal near 0.5. The marginal histogram shows the over-confident model piling predicted probabilities near 0 and 1. High discrimination (AUROC) can coexist with poor calibration after transport to a new site.

For binary and multiclass classifiers, the confusion matrix C is a K × K table where C_jk counts examples with true class j predicted as class k. As a heatmap, darker cells mark larger counts or rates. Row-normalization (divide each row by its sum) emphasizes recall patterns: among true class j, where do predictions go? Column-normalization emphasizes precision patterns: among predicted class k, what were the truths? Always state whether the matrix is computed on training, validation, or external test data. A perfect training heatmap with a chaotic external heatmap is a generalization failure, not a plotting bug. In stroke subtype models, off-diagonal mass between cardioembolic and cryptogenic labels often reflects label ontology more than algorithm choice—visualization makes that debate concrete.

Worked example — reading a 3×3 confusion-matrix heatmap. Suppose a subtype classifier is evaluated on a held-out test set of 130 encounters across three TOAST-style classes: large-artery atherosclerosis (LAA), cardioembolic (CE), and small-vessel occlusion (SVO). Rows are the true class, columns the predicted class:

```
Pred LAA Pred CE Pred SVO | row total
True LAA 30 6 4 | 40
True CE 10 35 5 | 50
True SVO 4 6 30 | 40
column total 44 47 39 | 130
```

Overall accuracy is the diagonal over the total: (30 + 35 + 30) / 130 = 95/130 ≈ 0.731. Now read the same table two ways. Row-normalizing divides each cell by its row total and answers “of the patients who truly have this subtype, where did predictions go?”—that is recall: LAA 30/40 = 0.75, CE 35/50 = 0.70, SVO 30/40 = 0.75. Column-normalizing divides by the column total and answers “of the patients predicted this subtype, how many truly had it?”—that is precision: LAA 30/44 ≈ 0.68, CE 35/47 ≈ 0.74, SVO 30/39 ≈ 0.77. The two normalizations tell different clinical stories from one matrix. LAA looks strong by recall (0.75), yet its precision is only 0.68, because ten cardioembolic patients were misread as LAA and inflate the predicted-LAA column; a clinician who trusts a predicted “LAA” label should know that roughly a third of such predictions are actually another subtype. This is exactly how a raw-count heatmap deceives under class imbalance: the darkest cells track the largest classes, not the largest error rates. Always state which normalization is shown and whether the counts are training, validation, or external-test—a crisp diagonal on training that dissolves on external data is a generalization failure, not a color-scale choice.

Reliability diagrams (calibration plots) bin predicted probabilities and compare mean predicted risk to observed event frequency within each bin. Perfect calibration lies on the diagonal. A model can have excellent discrimination (high AUROC) and still be systematically overconfident or underconfident after transport to a new site—a critical distinction when probabilities inform shared decision-making rather than mere ranking for a fixed-capacity bed. Learning curves plot training and validation scores against training set size and help separate high bias (both curves plateau poorly) from high variance (large gap that narrows with more data). Slice-wise metric bar charts by age band, sex, language preference, race and ethnicity, or hospital support equity and transportability audits that single average metrics erase.

Partial-dependence and accumulated-local-effect style plots show how a model’s prediction changes as one feature varies while others are held in a typical distribution. They are not causal effects, but they are useful sanity checks: a stroke mortality model whose partial dependence on age decreases after age 90 may be reflecting selection (only the robust very-old reach certain pathways) or a data artifact, not a biologic fountain of youth. Always pair such plots with domain review before celebrating ‘learned clinical knowledge.’

## A Grammar for Choosing Chart Types

Faced with a new table, analysts often open a plotting library and scroll through examples until something looks modern. A better discipline starts from the analytic question. If the question is part-to-whole composition with few categories, a pie or stacked bar may work; if the question is precise comparison of many categories, ordered bars win. If the question is change over a regularly sampled ordered axis, use lines; if the question is association between two continuous measures, use scatter (or hexbin when n is large). If the question is distribution shape and outliers within groups, use boxes, violins, or beeswarms. If the question is hierarchical contribution to a total, use treemap or sunburst. If the question is relational structure, use node-link or a matrix view of edges.

Multivariate pressure tempts radar charts and 3-D gadgets. Prefer small multiples: the same scatter faceted by site, or the same calibration curve faceted by age band. Faceting uses position—the strongest perceptual channel—repeatedly, rather than inventing a new channel that readers must decode under time pressure. When an extra continuous variable must appear on a bivariate scatter, color or size can help, but legends become cognitive load; direct labels on a few callout points often communicate more than a continuous color bar no one can read on a projector.

Interactive graphics in notebooks and dashboards add brushing, zooming, and tooltips. They are powerful for EDA and dangerous for publication if the only insight lives in a hover state the reader of a PDF cannot access. For durable scientific claims, ensure that a static export still carries the message. For operational dashboards, design for glanceability: one primary comparison per panel, consistent scales across related panels, and alerts that do not rely on hue alone.

Start from the question (composition, comparison, distribution, relationship, hierarchy, network).

Prefer position and length encodings for the primary quantitative comparison.

Use small multiples before exotic multivariate glyphs.

Export a static figure that still works without hover interactions.

Keep operational dashboards glanceable and colorblind-safe.

## Clinical and Epidemiologic Notes

Epidemiologic Table 1 is already a visualization plan: baseline characteristics by exposure or by cohort entry stratum. Extend that culture into ML projects. Before model training, plot missingness by site, outcome rates by calendar month (coding shifts), and severity distributions by transfer status. Selection bias often appears first as a figure—an empty cell in a continuum of care timeline, or a hospital whose NIHSS is systematically rounded to multiples of two, or a sudden step change in hemorrhage incidence the week a new coding system went live. Those patterns are not aesthetic; they are validity threats that no amount of gradient boosting will remove if ignored.

!!! note "Figure concept (text diagram) 2.6"

    Residual-versus-fitted diagnostics. Left: a healthy, structureless cloud centered on zero with a flat local-mean line. Right: a variance funnel (heteroscedasticity) whose spread grows with the fitted value, shown by the widening ±2σ envelope, warning that constant-variance prediction intervals will be too narrow for severe cases and too wide for mild ones.

Surveillance and quality improvement already use run charts and control charts; ML monitoring should inherit that culture. Plot weekly observed versus predicted event rates after deployment; plot the distribution of input features against the training reference (population stability indices are numeric summaries, not substitutes for pictures). When a new CT vendor arrives at one hub, embedding space scatter plots colored by vendor can reveal a batch effect before AUROC drops on a delayed outcome label.

For model reporting in manuscripts and governance packets, residual-versus-fitted plots, calibration diagrams, and row-normalized confusion heatmaps should be as standard as AUROC. A model can discriminate well and still miscalibrate after transport to a rural hospital; the calibration plot is the clinical honesty check for counseling-style probabilities. Slice-wise metric bar charts by age, sex, language preference, and race/ethnicity—when sample size allows—connect visualization to equity audit. Maps of stroke systems of care (choropleths of incidence, bubble maps of comprehensive stroke center volume) inform resource planning but must not reify unstable small-area rates. Network diagrams of transfers can identify fragile spokes—as hypothesis generators for operations research, not as automatic causal proof.

Finally, remember that figures travel. A dumbbell plot of disparity in thrombolysis rates may be screenshotted into a board presentation without your caption. Build titles and annotations that survive context collapse: population, years, numerator definition, and whether rates are adjusted. Visualization ethics is not only about avoiding truncated axes; it is about preventing motivated misreading when figures leave the methods section.

Make EDA plots a gate before training; keep a paper trail of plot-driven data fixes.

Standardize diagnostic plots for discrimination, calibration, and confusion structure.

Use maps and networks for systems questions; suppress unstable small-area estimates.

Design for color-vision deficiency, projection, photocopy, and caption-free reuse.

Monitor deployed models with the same graphical discipline used in quality improvement.

## Putting Visualization to Work: An End-to-End Mini Protocol

A reproducible visualization protocol for a new stroke ML project might run as follows. Day 1: draw univariate histograms or bar charts for every candidate feature and the outcome; flag impossible values and unexpected modes. Day 2: missingness heatmaps by site and by calendar month; decide whether missingness is a feature, a filter, or an imputation target. Day 3: pairwise scatters or a correlation heatmap for continuous blocks; note multicollinearity and nonlinear pairs for later modeling. Day 4: timeline audits on a random sample of encounters to verify index time and feature availability. Day 5: baseline model diagnostics—residuals or calibration and confusion—before any architecture search. Day 6: subgroup slices and, if geography matters, a simple map of sample size and outcome rate. Day 7: package a short visual appendix that freezes the data understanding on which modeling claims rest.

This protocol is deliberately boring. Boring graphics catch expensive errors early. Flashy embeddings and animated 3-D demos can wait until the histograms no longer surprise you. In clinical machine learning, the most valuable figure is often the one that forces you to fix the data, redefine the cohort, or abandon a leaked label—not the one that wins a poster layout award.

When teaching trainees, require that every modeling notebook open with a visualization section that would make sense to an epidemiologist coauthor. If the coauthor cannot state what population the histograms represent, the ML work is not yet ready for hyperparameter tuning. That social constraint is itself a quality tool: visualization is a shared language across specialties, and shared languages prevent private overfitting to a single analyst’s narrative.

## Connections

Visualization is not a standalone topic; it threads through the rest of this book. The univariate histograms and missingness heatmaps of this chapter are the front end of exploratory data analysis and feature engineering: the shape, skew, and impossible values you find here dictate transformations and imputation downstream. Dendrograms and MDS/PCA maps connect to unsupervised learning and clustering (Chapter 4), where the honest-reading cautions given here—unstable branches, meaningless t-SNE gaps, high-stress layouts—become validity criteria rather than aesthetics. Residual, calibration, and confusion-matrix displays are the visual face of model evaluation: ROC and precision–recall curves, calibration metrics, and the discrimination-versus-calibration distinction get their formal treatment alongside the numbers those plots summarize, and reliability diagrams plus slice-wise metric charts feed directly into fairness and transportability auditing. Control-chart-style monitoring of observed-versus-predicted rates connects to deployment and dataset-shift detection, where a drifting embedding scatter colored by site or scanner vendor is often the earliest warning that a model is degrading before the outcome labels arrive. Underlying all of it is one discipline shared with epidemiologic Table 1 and, later, causal diagrams: treat every figure as a checkable claim with a stated population, denominator, and data split. The picture is an argument, and it must survive scrutiny once it leaves the methods section.

## Chapter Summary

Visualization maps data to perception for exploration, model diagnosis, and communication. Its history runs from Playfair and Nightingale to Tukey’s EDA. Basic charts include pies (rare, few slices), bars and histograms, lines, scatters, and rarely broken axes. Multidimensional views include surfaces, contours, areas, radars, heatmaps, calendar and timeline plots, and box/violin summaries. Hierarchical structure uses dendrograms, treemaps, and sunbursts; networks use node-link, arc, and chord diagrams. Change is shown with divergent bars, dumbbells, waterfalls, and beeswarms. Three-dimensional graphics need interactive justification; choropleths and bubble maps serve spatial epidemiology with rate stability caveats. MDS and related embeddings visualize similarity in high dimensions but do not prove phenotypes. Encoding choice should follow perceptual rankings; misleading axes, dual scales, and decorative 3-D are scientific defects. A worked eight-patient example computed means, IQR, and Pearson r ≈ 0.93 between NIHSS and mRS, and a companion three-class example read recall and precision off a confusion-matrix heatmap—both as practice for honest EDA. Clinical ML inherits Table-1 discipline and adds calibration and error-slice graphics as first-class evidence.

## Practice and Reflection

(1) Redesign a 12-slice pie of discharge dispositions as a chart type better suited to comparison; justify the encoding.

(2) For NIHSS in a cohort of 500, explain how you would choose histogram bin width and when you would overlay a density.

(3) Sketch a scatter of age vs door-to-needle time faceted by transfer status; list three patterns that would change operations.

(4) When is a broken y-axis defensible in a quality report? Provide one acceptable and one unacceptable example.

(5) Compare heatmap vs scatter for diagnosing multicollinearity among five severity scores—what does each show better?

(6) Draw (schematically) a dendrogram cut yielding k=3 and state one stability check before naming the clusters clinically.

(7) A chord diagram of hospital transfers is unreadable at 40 nodes. Propose two alternative visualizations.

(8) Build a dumbbell plot concept for pre/post bundle change in median door-to-puncture by site; how will you sort sites?

(9) Critique a 3-D bar chart of quarterly mortality; rewrite the figure plan in 2-D with uncertainty.

(10) Choropleth of raw stroke death counts by county misleads—why? What statistic and annotations would you use instead?

(11) Explain classical MDS inputs and outputs in one paragraph; how does it differ from plotting the first two principal components?

(12) Using the toy data in the worked example, compute the median age and IQR of NIHSS; which plots would you present in a lab meeting?

(13) From the 3×3 confusion matrix in the diagnostic worked example, recompute LAA precision and recall if 5 of the 10 cardioembolic-as-LAA errors were instead classified correctly as cardioembolic. Which of the two metrics changes, which stays fixed, and why?

(14) A colleague shows a UMAP of stroke imaging features with two clean clusters and concludes there are two biological phenotypes. List three properties of the plot you would refuse to read literally, and one stability check you would run before believing the separation.
