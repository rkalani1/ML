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

![2.1: A visual vocabulary of six core chart types on small toy stroke data: a bar chart of subtype counts, a histogram of a bi](../assets/figures/ml_concept_2.1_78c12414.png)

*Figure 2.1 — original teaching graphic.*

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

![Y-axis truncation exaggerates small rate differences across sites (original).](../assets/figures/ml_fig_axis_truncation.png)

*Figure — Same four site rates, two stories. **Left:** truncated y-limits turn 0.5–1.0 point gaps into dramatic cliffs. **Right:** a zero baseline restores honest length encoding. Start quantitative axes at zero unless a labeled log scale or a clearly marked inset is justified. Misleading axes do not create causation—but they do create unjustified QI panic.*

![Colormap hygiene: rainbow jet vs sequential teal on the same matrix (original).](../assets/figures/ml_fig_colormap_hygiene.png)

*Figure — Heatmap color encodes magnitude. **Left:** rainbow (jet) invents false boundaries and is hard for color-vision deficiency. **Right:** a sequential teal map preserves order and prints more safely. Prefer sequential or diverging perceptually uniform maps for clinical matrices.*

![Dual axis vs indexed comparable series (original).](../assets/figures/ml_fig_dual_vs_index.png)

*Figure — Dual axes invite false parallels between series on different scales. Prefer indexing each series to a common baseline (or small multiples with shared axes) so slopes remain honest. Visual rhetoric is not evidence of causation.*

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

![2.2: Three-class confusion-matrix heatmap for a TOAST subtype classifier on an external test set (n = 130), shaded with a seq](../assets/figures/ml_concept_2.2_3e717523.png)

*Figure 2.2 — original teaching graphic.*

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

![2.3: The same 2-D data rendered as two neighbor embeddings at different perplexity settings. At low perplexity (left) the thr](../assets/figures/ml_concept_2.3_53e2258d.png)

*Figure 2.3 — original teaching graphic.*

PCA is a related linear projection maximizing variance rather than preserving arbitrary distances; t-SNE and UMAP emphasize local neighborhoods, which makes three of their features routinely misread. First, cluster sizes carry no meaning: a tight blob and a diffuse blob can hold equal counts, so relative area says nothing about prevalence. Second, between-cluster distances are not faithful: two clusters drawn side by side need not be more similar than two drawn far apart. Third, apparent gaps shift with tuning (the perplexity or n_neighbors setting) and with the random seed, so a separation seen in one run can dissolve in the next. Treat nonlinear embeddings as hypothesis generators: rerun several seeds and settings, color by batch or site to detect shift, color by known labels to audit leakage, and never claim that a pretty 2-D map alone validates a phenotype. Pair embeddings with quantitative stability checks and, when labels exist for audit, supervised metrics on held-out data.

Parallel coordinates, small-multiple sparklines, and linked brushing in interactive tools extend the toolbox for moderate d. For large n, prefer aggregated views: hexbin or 2-D density contours instead of overplotted scatters.

## Choosing Encodings and Avoiding Misleading Visualizations

Human perception ranks visual channels roughly as: position along a common scale, length, angle or slope, area, volume, then color saturation or hue (highly context-dependent). Encode the most important quantitative comparisons with position or length. Reserve color for categorical strata with a colorblind-safe palette and a modest number of levels. Size can carry an extra quantitative variable but collides under overplotting. Prefer small multiples over overloading a single panel with every channel.

![2.4: The truncated-baseline deception. The same four door-to-needle medians (60, 58, 59, 57 minutes) are drawn with a truncat](../assets/figures/ml_concept_2.4_960bbe38.png)

*Figure 2.4 — original teaching graphic.*

![Dual y-axis caution: fake alignment vs declared scales (synthetic; original).](../assets/figures/ml_fig_dual_axis_caution.png)

*Figure — Dual-axis lie factor. **Left:** admissions and mortality % share an x-axis but use independent truncated y-scales; the eye invents co-movement (and a “crossover”) that is mostly scale choice. **Right:** zero baselines and loudly declared separate scales reduce the visual lie; stacked small multiples are still safer for print. Tufte-style lie-factor proxies quantify how much the picture exaggerates relative change.*

Three deceptions deserve mechanical understanding, because they fool careful readers, not only careless ones. First, the truncated baseline: when a bar’s value axis does not start at zero, bar length no longer encodes magnitude, so a door-to-needle change from 60 to 58 minutes can be drawn to look like a near-halving simply by starting the axis at 55. A bar implies a zero; if you must zoom, switch to a dot or line that does not carry a length cue, and label the zoom. Second, dual y-axes: overlaying two series on independently chosen left and right scales invites the eye to read their crossings and co-movement as a relationship, but there is no canonical alignment—by rescaling one axis you can make stroke volume and mortality appear to track together or diverge at will, and the “crossover point” is a pure artifact of the chosen limits. Prefer two aligned panels sharing an x-axis. Third, cherry-picking the window: a metric that wanders will always contain some start-and-end pair that shows the story you want, so reporting door-to-needle “improvement” measured from the single worst month manufactures a trend that the full series would erase. State the whole period, fix the comparison window before looking, and show the surrounding context.

![Aspect ratio changes perceived trend slope (same series; original).](../assets/figures/ml_fig_aspect_ratio.png)

*Figure — Banking and aspect ratio. **Left:** a tall y-span flattens seasonal slopes. **Right:** a tight y-span makes the same series look dramatic. Cleveland’s banking-to-45° is a default hygiene for slope comparison—not a license to deceive. Always report absolute change and units alongside the picture.*

Start count/rate bars at zero unless you explicitly justify a zoomed scale and label it loudly.

Avoid dual y-axes that force false alignment between unrelated series; use aligned small multiples.

Declare log scales and transforms in the axis title, not only in a footnote.

Prefer perceptually uniform colormaps for sequential data; avoid rainbow scales for magnitudes.

Show uncertainty (intervals, bootstrap bands) when comparing estimates across groups.

Pair color with redundant non-color encodings (shape, pattern) for critical categories.

Do not cherry-pick time windows or y-limits to manufacture a policy narrative.

![Cherry-picked time windows manufacture a trend (synthetic DTN; original).](../assets/figures/ml_fig_window_cherry.png)

*Figure — Window cherry-picking. **Left:** a wandering door-to-needle series with one bad spike; the red start→end segment through a pre-chosen “improvement” window looks like a policy win. **Right:** random windows of the same series yield a wide distribution of start−end deltas—many “wins” by chance. Pre-specify the comparison window, show the full series, and report uncertainty.*


![Histogram binning artifacts on a synthetic two-mode mixture (original).](../assets/figures/ml_fig_binning_artifact.png)

*Figure — Visualization: bin width is a hyperparameter. Too few bins hide modes; too many invent noise spikes. Gold dashed lines mark generating centers. Bin choice is not evidence of clinical subtypes—**display ≠ etiology**.*


![ECDF comparison of two synthetic site distributions (original).](../assets/figures/ml_fig_ecdf_compare.png)

*Figure — ECDFs avoid arbitrary histogram bins. Vertical gaps flag location/scale shift between sites. Shift detection supports monitoring—not a causal claim about site quality without design.*


![Parallel coordinates with overplot risk on synthetic z-scores (original).](../assets/figures/ml_fig_parallel_coords.png)

*Figure — Parallel coordinates can reveal bundles but overplot quickly; axis order changes perception. Visual clusters remain geometry—not etiologic subtypes.*


![Boxplots vs violins for unimodal vs bimodal synthetic data (original).](../assets/figures/ml_fig_box_vs_violin.png)

*Figure — Boxplots can hide multimodality that violins reveal. Choose displays that match the question. Visual form is not etiology.*


![Same series with honest vs zoomed y-limits (original).](../assets/figures/ml_fig_axis_limits.png)

*Figure — Axis limits manufacture drama. Always inspect scale choices. Visualization ethics are part of scientific honesty—not etiology.*


![Dual-axis charts can mislead vs shared scale (original).](../assets/figures/ml_fig_dual_axis.png)

*Figure — Display ethics—not etiology. Pred ≠ cause without design.*


![Colormap rainbow vs sequential caution (original).](../assets/figures/ml_fig_colormap_abuse.png)

*Figure — Colormaps can invent false edges. Colormap rainbow vs sequential caution Pred != cause without design.*


![logscale teaching panel (original).](../assets/figures/ml_fig_log_scale.png)

*Figure — Teaching panel for logscale. Pred != cause without design.*


![Cycle-34 densify scientific panel 4 (original).](../assets/figures/ml_fig_c34_03.png)

*Figure — Continuous densify panel 4. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 4 (original).](../assets/figures/ml_fig_c35_03.png)

*Figure — Continuous densify panel 4. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 4 (original).](../assets/figures/ml_fig_c36_03.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 4 (original).](../assets/figures/ml_fig_c37_03.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 4 (original).](../assets/figures/ml_fig_c38_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 4 (original).](../assets/figures/ml_fig_c39_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 4 (original).](../assets/figures/ml_fig_c40_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 4 (original).](../assets/figures/ml_fig_c41_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 4 (original).](../assets/figures/ml_fig_c42_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 4 (original).](../assets/figures/ml_fig_c43_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 4 (original).](../assets/figures/ml_fig_c44_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 4 (original).](../assets/figures/ml_fig_c45_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 4 (original).](../assets/figures/ml_fig_c46_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 4 (original).](../assets/figures/ml_fig_c47_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 4 (original).](../assets/figures/ml_fig_c48_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 4 (original).](../assets/figures/ml_fig_c49_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 4 (original).](../assets/figures/ml_fig_c50_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 4 (original).](../assets/figures/ml_fig_c51_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 4 (original).](../assets/figures/ml_fig_c52_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 4 (original).](../assets/figures/ml_fig_c53_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 4 (original).](../assets/figures/ml_fig_c54_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 4 (original).](../assets/figures/ml_fig_c55_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 4 (original).](../assets/figures/ml_fig_c56_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 4 (original).](../assets/figures/ml_fig_c57_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 4 (original).](../assets/figures/ml_fig_c58_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 4 (original).](../assets/figures/ml_fig_c59_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 4 (original).](../assets/figures/ml_fig_c60_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 4 (original).](../assets/figures/ml_fig_c61_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 4 (original).](../assets/figures/ml_fig_c62_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 4 (original).](../assets/figures/ml_fig_c63_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 4 (original).](../assets/figures/ml_fig_c64_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 4 (original).](../assets/figures/ml_fig_c65_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 4 (original).](../assets/figures/ml_fig_c66_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 4 (original).](../assets/figures/ml_fig_c67_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 4 (original).](../assets/figures/ml_fig_c68_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 4 (original).](../assets/figures/ml_fig_c69_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 4 (original).](../assets/figures/ml_fig_c70_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 4 (original).](../assets/figures/ml_fig_c71_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 4 (original).](../assets/figures/ml_fig_c72_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 4 (original).](../assets/figures/ml_fig_c73_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 4 (original).](../assets/figures/ml_fig_c74_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 4 (original).](../assets/figures/ml_fig_c75_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 4 (original).](../assets/figures/ml_fig_c76_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 4 (original).](../assets/figures/ml_fig_c77_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 4 (original).](../assets/figures/ml_fig_c78_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 4 (original).](../assets/figures/ml_fig_c79_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 4 (original).](../assets/figures/ml_fig_c80_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 4 (original).](../assets/figures/ml_fig_c81_03.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

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

![2.5: Reliability (calibration) diagram comparing a well-calibrated model, whose binned points track the 45° reference line, w](../assets/figures/ml_concept_2.5_34de3e51.png)

*Figure 2.5 — original teaching graphic.*

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

![2.6: Residual-versus-fitted diagnostics. Left: a healthy, structureless cloud centered on zero with a flat local-mean line. R](../assets/figures/ml_concept_2.6_bdcc55f5.png)

*Figure 2.6 — original teaching graphic.*

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


![c82 teaching panel 03 (original).](../assets/figures/ml_fig_c82_03.png)
*Figure — Anscombe lesson: identical summary stats can hide different structure—always plot. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 03 (original).](../assets/figures/ml_fig_c83_03.png)
*Figure — Parallel coordinates for multi-feature pattern scanning. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 03 (original).](../assets/figures/ml_fig_c84_03.png)
*Figure — Colormap choice changes perceived magnitude (rainbow trap). Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 03 (original).](../assets/figures/ml_fig_c85_03.png)
*Figure — Hexbin density helps when scatterplots overplot. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 03 (original).](../assets/figures/ml_fig_c86_03.png)
*Figure — Class-conditional clouds—visualize before modeling. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 03 (original).](../assets/figures/ml_fig_c87_03.png)
*Figure — Bar charts need uncertainty intervals. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 03 (original).](../assets/figures/ml_fig_c88_03.png)
*Figure — Small multiples beat overstuffed single axes. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 03 (original).](../assets/figures/ml_fig_c89_03.png)
*Figure — Slopegraph for before/after group comparison. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 03 (original).](../assets/figures/ml_fig_c90_03.png)
*Figure — Raincloud: density + jitter + box. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 03 (original).](../assets/figures/ml_fig_c91_03.png)
*Figure — Violin plots for distribution shape. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 03 (original).](../assets/figures/ml_fig_c92_03.png)
*Figure — Alluvial-like cohort flow. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 03 (original).](../assets/figures/ml_fig_c93_03.png)
*Figure — Bland-Altman agreement plot. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 03 (original).](../assets/figures/ml_fig_c94_03.png)
*Figure — ECDF comparison of two groups. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 03 (original).](../assets/figures/ml_fig_c95_03.png)
*Figure — QQ plot normality check. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 03 (original).](../assets/figures/ml_fig_c96_03.png)
*Figure — Raincloud alt: strip+box only. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 03 (original).](../assets/figures/ml_fig_c97_03.png)
*Figure — Ridgeline density facets. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 03 (original).](../assets/figures/ml_fig_c98_03.png)
*Figure — Parallel sets categorical flow. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 03 (original).](../assets/figures/ml_fig_c99_03.png)
*Figure — Hexbin vs scatter density choice. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 03 (original).](../assets/figures/ml_fig_c100_03.png)
*Figure — Cumulative distribution ladders. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 03 (original).](../assets/figures/ml_fig_c101_03.png)
*Figure — Sankey exclusion cascade. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 03 (original).](../assets/figures/ml_fig_c102_03.png)
*Figure — Bland-Altman with proportional bias. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 03 (original).](../assets/figures/ml_fig_c103_03.png)
*Figure — Letter-value plot sketch. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 03 (original).](../assets/figures/ml_fig_c104_03.png)
*Figure — Upset-plot set intersections. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 03 (original).](../assets/figures/ml_fig_c105_03.png)
*Figure — Mean absolute scaled error. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 03 (original).](../assets/figures/ml_fig_c106_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 03 (original).](../assets/figures/ml_fig_c107_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 03 (original).](../assets/figures/ml_fig_c108_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 03 (original).](../assets/figures/ml_fig_c109_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 03 (original).](../assets/figures/ml_fig_c110_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 03 (original).](../assets/figures/ml_fig_c111_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 03 (original).](../assets/figures/ml_fig_c112_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 03 (original).](../assets/figures/ml_fig_c113_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 03 (original).](../assets/figures/ml_fig_c114_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 03 (original).](../assets/figures/ml_fig_c115_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 03 (original).](../assets/figures/ml_fig_c116_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 03 (original).](../assets/figures/ml_fig_c117_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 03 (original).](../assets/figures/ml_fig_c118_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 03 (original).](../assets/figures/ml_fig_c119_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 03 (original).](../assets/figures/ml_fig_c120_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 03 (original).](../assets/figures/ml_fig_c121_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 03 (original).](../assets/figures/ml_fig_c122_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 03 (original).](../assets/figures/ml_fig_c123_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 03 (original).](../assets/figures/ml_fig_c124_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 03 (original).](../assets/figures/ml_fig_c125_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 03 (original).](../assets/figures/ml_fig_c126_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 03 (original).](../assets/figures/ml_fig_c127_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 03 (original).](../assets/figures/ml_fig_c128_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 03 (original).](../assets/figures/ml_fig_c129_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 03 (original).](../assets/figures/ml_fig_c130_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 03 (original).](../assets/figures/ml_fig_c131_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 03 (original).](../assets/figures/ml_fig_c132_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 03 (original).](../assets/figures/ml_fig_c133_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 03 (original).](../assets/figures/ml_fig_c134_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 03 (original).](../assets/figures/ml_fig_c135_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 03 (original).](../assets/figures/ml_fig_c136_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 03 (original).](../assets/figures/ml_fig_c137_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 03 (original).](../assets/figures/ml_fig_c138_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 03 (original).](../assets/figures/ml_fig_c139_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 03 (original).](../assets/figures/ml_fig_c140_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 03 (original).](../assets/figures/ml_fig_c141_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 03 (original).](../assets/figures/ml_fig_c142_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 03 (original).](../assets/figures/ml_fig_c143_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 03 (original).](../assets/figures/ml_fig_c144_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 03 (original).](../assets/figures/ml_fig_c145_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 03 (original).](../assets/figures/ml_fig_c146_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 03 (original).](../assets/figures/ml_fig_c147_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 03 (original).](../assets/figures/ml_fig_c148_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 03 (original).](../assets/figures/ml_fig_c149_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 03 (original).](../assets/figures/ml_fig_c150_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 03 (original).](../assets/figures/ml_fig_c151_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 03 (original).](../assets/figures/ml_fig_c152_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 03 (original).](../assets/figures/ml_fig_c153_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 03 (original).](../assets/figures/ml_fig_c154_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 03 (original).](../assets/figures/ml_fig_c155_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 03 (original).](../assets/figures/ml_fig_c156_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 03 (original).](../assets/figures/ml_fig_c157_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 03 (original).](../assets/figures/ml_fig_c158_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 03 (original).](../assets/figures/ml_fig_c159_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 03 (original).](../assets/figures/ml_fig_c160_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 03 (original).](../assets/figures/ml_fig_c161_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 03 (original).](../assets/figures/ml_fig_c162_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 03 (original).](../assets/figures/ml_fig_c163_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 03 (original).](../assets/figures/ml_fig_c164_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 03 (original).](../assets/figures/ml_fig_c165_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 03 (original).](../assets/figures/ml_fig_c166_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 03 (original).](../assets/figures/ml_fig_c167_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 03 (original).](../assets/figures/ml_fig_c168_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 03 (original).](../assets/figures/ml_fig_c169_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 03 (original).](../assets/figures/ml_fig_c170_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 03 (original).](../assets/figures/ml_fig_c171_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 03 (original).](../assets/figures/ml_fig_c172_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 03 (original).](../assets/figures/ml_fig_c173_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 03 (original).](../assets/figures/ml_fig_c174_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 03 (original).](../assets/figures/ml_fig_c175_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 03 (original).](../assets/figures/ml_fig_c176_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 03 (original).](../assets/figures/ml_fig_c177_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 03 (original).](../assets/figures/ml_fig_c178_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 03 (original).](../assets/figures/ml_fig_c179_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 03 (original).](../assets/figures/ml_fig_c180_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 03 (original).](../assets/figures/ml_fig_c181_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 03 (original).](../assets/figures/ml_fig_c182_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 03 (original).](../assets/figures/ml_fig_c183_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 03 (original).](../assets/figures/ml_fig_c184_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 03 (original).](../assets/figures/ml_fig_c185_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 03 (original).](../assets/figures/ml_fig_c186_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 03 (original).](../assets/figures/ml_fig_c187_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 03 (original).](../assets/figures/ml_fig_c188_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 03 (original).](../assets/figures/ml_fig_c189_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 03 (original).](../assets/figures/ml_fig_c190_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 03 (original).](../assets/figures/ml_fig_c191_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 03 (original).](../assets/figures/ml_fig_c192_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 03 (original).](../assets/figures/ml_fig_c193_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 03 (original).](../assets/figures/ml_fig_c194_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 03 (original).](../assets/figures/ml_fig_c195_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 03 (original).](../assets/figures/ml_fig_c196_03.png)
*Figure — Horizon chart density. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 03 (original).](../assets/figures/ml_fig_c197_03.png)
*Figure — Dot plot with CI. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 03 (original).](../assets/figures/ml_fig_c198_03.png)
*Figure — Marimekko mosaic. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 03 (original).](../assets/figures/ml_fig_c199_03.png)
*Figure — Hexbin log counts. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 03 (original).](../assets/figures/ml_fig_c200_03.png)
*Figure — Paired slope multiples. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 03 (original).](../assets/figures/ml_fig_c201_03.png)
*Figure — Small multiples vs spaghetti. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 03 (original).](../assets/figures/ml_fig_c202_03.png)
*Figure — ECDF with DKW band sketch. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 03 (original).](../assets/figures/ml_fig_c203_03.png)
*Figure — Horizon chart layered density. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 03 (original).](../assets/figures/ml_fig_c204_03.png)
*Figure — Parallel coordinates class traces. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 03 (original).](../assets/figures/ml_fig_c205_03.png)
*Figure — Q-Q plot normality check. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 03 (original).](../assets/figures/ml_fig_c206_03.png)
*Figure — Ridgeline density small multiples. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 03 (original).](../assets/figures/ml_fig_c207_03.png)
*Figure — Colorblind-safe palette channels. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 03 (original).](../assets/figures/ml_fig_c208_03.png)
*Figure — Tufte data-ink ratio bars. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 03 (original).](../assets/figures/ml_fig_c209_03.png)
*Figure — Hexbin joint density view. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 03 (original).](../assets/figures/ml_fig_c210_03.png)
*Figure — Raincloud distribution comparison. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 03 (original).](../assets/figures/ml_fig_c211_03.png)
*Figure — Connected scatter time path. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 03 (original).](../assets/figures/ml_fig_c212_03.png)
*Figure — Train-val-test role alluvial. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 03 (original).](../assets/figures/ml_fig_c213_03.png)
*Figure — Slopegraph metric version delta. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 03 (original).](../assets/figures/ml_fig_c214_03.png)
*Figure — Bullet chart KPI targets. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 03 (original).](../assets/figures/ml_fig_c215_03.png)
*Figure — Mosaic plot joint proportions. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 03 (original).](../assets/figures/ml_fig_c216_03.png)
*Figure — Diverging bar signed effects. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 03 (original).](../assets/figures/ml_fig_c217_03.png)
*Figure — Pairs plot matrix collage. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 03 (original).](../assets/figures/ml_fig_c218_03.png)
*Figure — Beeswarm group comparison. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 03 (original).](../assets/figures/ml_fig_c219_03.png)
*Figure — ECDF confidence band. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 03 (original).](../assets/figures/ml_fig_c220_03.png)
*Figure — Parallel sets category flows. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 03 (original).](../assets/figures/ml_fig_c221_03.png)
*Figure — Joyplot ridge density stack. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 03 (original).](../assets/figures/ml_fig_c222_03.png)
*Figure — Hexbin bivariate density. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 03 (original).](../assets/figures/ml_fig_c223_03.png)
*Figure — Streamgraph theme-river flows. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 03 (original).](../assets/figures/ml_fig_c224_03.png)
*Figure — Normal Q-Q diagnostic plot. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 03 (original).](../assets/figures/ml_fig_c225_03.png)
*Figure — SPLOM lower-triangle collage. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 03 (original).](../assets/figures/ml_fig_c226_03.png)
*Figure — Split violin group comparison. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 03 (original).](../assets/figures/ml_fig_c227_03.png)
*Figure — Beeswarm group comparison. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 03 (original).](../assets/figures/ml_fig_c228_03.png)
*Figure — Filled contour density field. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 03 (original).](../assets/figures/ml_fig_c229_03.png)
*Figure — Alluvial multi-stage flows. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 03 (original).](../assets/figures/ml_fig_c230_03.png)
*Figure — Radar multi-metric profile. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 03 (original).](../assets/figures/ml_fig_c231_03.png)
*Figure — Horizon chart layered bands. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 03 (original).](../assets/figures/ml_fig_c232_03.png)
*Figure — Parallel coordinates profiles. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 03 (original).](../assets/figures/ml_fig_c233_03.png)
*Figure — Small-multiple correlation heat. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 03 (original).](../assets/figures/ml_fig_c234_03.png)
*Figure — Raincloud class comparison. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 03 (original).](../assets/figures/ml_fig_c235_03.png)
*Figure — Bicluster residual heat map. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 03 (original).](../assets/figures/ml_fig_c236_03.png)
*Figure — Pairwise class clouds. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 03 (original).](../assets/figures/ml_fig_c237_03.png)
*Figure — Confusion structure heat map. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 03 (original).](../assets/figures/ml_fig_c238_03.png)
*Figure — One-class support cloud. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 03 (original).](../assets/figures/ml_fig_c239_03.png)
*Figure — Calibration reliability heat. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 03 (original).](../assets/figures/ml_fig_c240_03.png)
*Figure — Isolation score cloud. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 03 (original).](../assets/figures/ml_fig_c241_03.png)
*Figure — Brier score reliability heat. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 03 (original).](../assets/figures/ml_fig_c242_03.png)
*Figure — Local outlier factor cloud. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 03 (original).](../assets/figures/ml_fig_c243_03.png)
*Figure — ECE bin reliability heat. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 03 (original).](../assets/figures/ml_fig_c244_03.png)
*Figure — Mahalanobis anomaly cloud. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 03 (original).](../assets/figures/ml_fig_c245_03.png)
*Figure — Platt scaling reliability heat. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 03 (original).](../assets/figures/ml_fig_c246_03.png)
*Figure — COPOD copula anomaly cloud. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 03 (original).](../assets/figures/ml_fig_c247_03.png)
*Figure — Isotonic calib reliability heat. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 03 (original).](../assets/figures/ml_fig_c248_03.png)
*Figure — HBOS histogram anomaly cloud. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 03 (original).](../assets/figures/ml_fig_c249_03.png)
*Figure — Beta calibration reliability heat. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 03 (original).](../assets/figures/ml_fig_c250_03.png)
*Figure — ECOD empirical CDF cloud. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 03 (original).](../assets/figures/ml_fig_c251_03.png)
*Figure — Venn-Abers reliability heat. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 03 (original).](../assets/figures/ml_fig_c252_03.png)
*Figure — KNN distance anomaly cloud. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 03 (original).](../assets/figures/ml_fig_c253_03.png)
*Figure — Dirichlet calib reliability. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 03 (original).](../assets/figures/ml_fig_c254_03.png)
*Figure — Isolation forest cloud. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 03 (original).](../assets/figures/ml_fig_c255_03.png)
*Figure — Temperature scaling heat. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 03 (original).](../assets/figures/ml_fig_c256_03.png)
*Figure — Local outlier factor cloud. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 03 (original).](../assets/figures/ml_fig_c257_03.png)
*Figure — Calibrated color heat c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 03 (original).](../assets/figures/ml_fig_c258_03.png)
*Figure — Animation frame residual c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 03 (original).](../assets/figures/ml_fig_c259_03.png)
*Figure — Facet grid scatter c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 03 (original).](../assets/figures/ml_fig_c260_03.png)
*Figure — Raincloud density path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 03 (original).](../assets/figures/ml_fig_c261_03.png)
*Figure — QQ residual path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 03 (original).](../assets/figures/ml_fig_c262_03.png)
*Figure — ECDF comparison path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 03 (original).](../assets/figures/ml_fig_c263_03.png)
*Figure — Color map accessibility heat c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 03 (original).](../assets/figures/ml_fig_c264_03.png)
*Figure — Small multiples layout path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 03 (original).](../assets/figures/ml_fig_c265_03.png)
*Figure — Linked brushing scatter c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 03 (original).](../assets/figures/ml_fig_c266_03.png)
*Figure — Parallel coords bars c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 03 (original).](../assets/figures/ml_fig_c267_03.png)
*Figure — Radar chart misuse bars c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 03 (original).](../assets/figures/ml_fig_c268_03.png)
*Figure — Uncertainty band path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 03 (original).](../assets/figures/ml_fig_c269_03.png)
*Figure — Log-scale distortion path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 03 (original).](../assets/figures/ml_fig_c270_03.png)
*Figure — Glyph encoding scatter c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 03 (original).](../assets/figures/ml_fig_c271_03.png)
*Figure — Treemap hierarchy heat c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 03 (original).](../assets/figures/ml_fig_c272_03.png)
*Figure — Sankey flow path c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 03 (original).](../assets/figures/ml_fig_c273_03.png)
*Figure — Calibrated color heat c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 03 (original).](../assets/figures/ml_fig_c274_03.png)
*Figure — Animation frame residual c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 03 (original).](../assets/figures/ml_fig_c275_03.png)
*Figure — Facet grid scatter c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 03 (original).](../assets/figures/ml_fig_c276_03.png)
*Figure — Raincloud density path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 03 (original).](../assets/figures/ml_fig_c277_03.png)
*Figure — QQ residual path c277. Synthetic teaching geometry—not a causal claim.*


![c278 teaching panel 03 (original).](../assets/figures/ml_fig_c278_03.png)
*Figure — ECDF comparison path c278. Synthetic teaching geometry—not a causal claim.*


![c279 teaching panel 03 (original).](../assets/figures/ml_fig_c279_03.png)
*Figure — Color map accessibility heat c279. Synthetic teaching geometry—not a causal claim.*


![c280 teaching panel 03 (original).](../assets/figures/ml_fig_c280_03.png)
*Figure — Small multiples layout path c280. Synthetic teaching geometry—not a causal claim.*


![c281 teaching panel 03 (original).](../assets/figures/ml_fig_c281_03.png)
*Figure — Linked brushing scatter c281. Synthetic teaching geometry—not a causal claim.*


![c282 teaching panel 03 (original).](../assets/figures/ml_fig_c282_03.png)
*Figure — Parallel coords bars c282. Synthetic teaching geometry—not a causal claim.*


![c283 teaching panel 03 (original).](../assets/figures/ml_fig_c283_03.png)
*Figure — Radar chart misuse bars c283. Synthetic teaching geometry—not a causal claim.*


![c284 teaching panel 03 (original).](../assets/figures/ml_fig_c284_03.png)
*Figure — Uncertainty band path c284. Synthetic teaching geometry—not a causal claim.*


![c285 teaching panel 03 (original).](../assets/figures/ml_fig_c285_03.png)
*Figure — Log-scale distortion path c285. Synthetic teaching geometry—not a causal claim.*


![c286 teaching panel 03 (original).](../assets/figures/ml_fig_c286_03.png)
*Figure — Glyph encoding scatter c286. Synthetic teaching geometry—not a causal claim.*


![c287 teaching panel 03 (original).](../assets/figures/ml_fig_c287_03.png)
*Figure — Treemap hierarchy heat c287. Synthetic teaching geometry—not a causal claim.*


![c288 teaching panel 03 (original).](../assets/figures/ml_fig_c288_03.png)
*Figure — Sankey flow path c288. Synthetic teaching geometry—not a causal claim.*


![c289 teaching panel 03 (original).](../assets/figures/ml_fig_c289_03.png)
*Figure — Calibrated color heat c289. Synthetic teaching geometry—not a causal claim.*


![c290 teaching panel 03 (original).](../assets/figures/ml_fig_c290_03.png)
*Figure — Animation frame residual c290. Synthetic teaching geometry—not a causal claim.*


![c291 teaching panel 03 (original).](../assets/figures/ml_fig_c291_03.png)
*Figure — Facet grid scatter c291. Synthetic teaching geometry—not a causal claim.*


![c292 teaching panel 03 (original).](../assets/figures/ml_fig_c292_03.png)
*Figure — Raincloud density path c292. Synthetic teaching geometry—not a causal claim.*


![c293 teaching panel 03 (original).](../assets/figures/ml_fig_c293_03.png)
*Figure — QQ residual path c293. Synthetic teaching geometry—not a causal claim.*


![c294 teaching panel 03 (original).](../assets/figures/ml_fig_c294_03.png)
*Figure — ECDF comparison path c294. Synthetic teaching geometry—not a causal claim.*


![c295 teaching panel 03 (original).](../assets/figures/ml_fig_c295_03.png)
*Figure — Color map accessibility heat c295. Synthetic teaching geometry—not a causal claim.*


![c296 teaching panel 03 (original).](../assets/figures/ml_fig_c296_03.png)
*Figure — Small multiples layout path c296. Synthetic teaching geometry—not a causal claim.*


![c297 teaching panel 03 (original).](../assets/figures/ml_fig_c297_03.png)
*Figure — Linked brushing scatter c297. Synthetic teaching geometry—not a causal claim.*


![c298 teaching panel 03 (original).](../assets/figures/ml_fig_c298_03.png)
*Figure — Parallel coords bars c298. Synthetic teaching geometry—not a causal claim.*


![c299 teaching panel 03 (original).](../assets/figures/ml_fig_c299_03.png)
*Figure — Radar chart misuse bars c299. Synthetic teaching geometry—not a causal claim.*


![c300 teaching panel 03 (original).](../assets/figures/ml_fig_c300_03.png)
*Figure — Uncertainty band path c300. Synthetic teaching geometry—not a causal claim.*


![c301 teaching panel 03 (original).](../assets/figures/ml_fig_c301_03.png)
*Figure — Log-scale distortion path c301. Synthetic teaching geometry—not a causal claim.*


![c302 teaching panel 03 (original).](../assets/figures/ml_fig_c302_03.png)
*Figure — Glyph encoding scatter c302. Synthetic teaching geometry—not a causal claim.*


![c303 teaching panel 03 (original).](../assets/figures/ml_fig_c303_03.png)
*Figure — Treemap hierarchy heat c303. Synthetic teaching geometry—not a causal claim.*


![c304 teaching panel 03 (original).](../assets/figures/ml_fig_c304_03.png)
*Figure — Sankey flow path c304. Synthetic teaching geometry—not a causal claim.*


![c305 teaching panel 03 (original).](../assets/figures/ml_fig_c305_03.png)
*Figure — Calibrated color heat c305. Synthetic teaching geometry—not a causal claim.*


![c306 teaching panel 03 (original).](../assets/figures/ml_fig_c306_03.png)
*Figure — Animation frame residual c306. Synthetic teaching geometry—not a causal claim.*


![c307 teaching panel 03 (original).](../assets/figures/ml_fig_c307_03.png)
*Figure — Facet grid scatter c307. Synthetic teaching geometry—not a causal claim.*


![c308 teaching panel 03 (original).](../assets/figures/ml_fig_c308_03.png)
*Figure — Raincloud density path c308. Synthetic teaching geometry—not a causal claim.*


![c309 teaching panel 03 (original).](../assets/figures/ml_fig_c309_03.png)
*Figure — QQ residual path c309. Synthetic teaching geometry—not a causal claim.*


![c310 teaching panel 03 (original).](../assets/figures/ml_fig_c310_03.png)
*Figure — ECDF comparison path c310. Synthetic teaching geometry—not a causal claim.*


![c311 teaching panel 03 (original).](../assets/figures/ml_fig_c311_03.png)
*Figure — Color map accessibility heat c311. Synthetic teaching geometry—not a causal claim.*


![c312 teaching panel 03 (original).](../assets/figures/ml_fig_c312_03.png)
*Figure — Small multiples layout path c312. Synthetic teaching geometry—not a causal claim.*


![c313 teaching panel 03 (original).](../assets/figures/ml_fig_c313_03.png)
*Figure — Linked brushing scatter c313. Synthetic teaching geometry—not a causal claim.*


![c314 teaching panel 03 (original).](../assets/figures/ml_fig_c314_03.png)
*Figure — Parallel coords bars c314. Synthetic teaching geometry—not a causal claim.*


![c315 teaching panel 03 (original).](../assets/figures/ml_fig_c315_03.png)
*Figure — Radar chart misuse bars c315. Synthetic teaching geometry—not a causal claim.*


![c316 teaching panel 03 (original).](../assets/figures/ml_fig_c316_03.png)
*Figure — Uncertainty band path c316. Synthetic teaching geometry—not a causal claim.*


![c317 teaching panel 03 (original).](../assets/figures/ml_fig_c317_03.png)
*Figure — Log-scale distortion path c317. Synthetic teaching geometry—not a causal claim.*


![c318 teaching panel 03 (original).](../assets/figures/ml_fig_c318_03.png)
*Figure — Glyph encoding scatter c318. Synthetic teaching geometry—not a causal claim.*


![c319 teaching panel 03 (original).](../assets/figures/ml_fig_c319_03.png)
*Figure — Treemap hierarchy heat c319. Synthetic teaching geometry—not a causal claim.*


![c320 teaching panel 03 (original).](../assets/figures/ml_fig_c320_03.png)
*Figure — Sankey flow path c320. Synthetic teaching geometry—not a causal claim.*


![c321 teaching panel 03 (original).](../assets/figures/ml_fig_c321_03.png)
*Figure — Calibrated color heat c321. Synthetic teaching geometry—not a causal claim.*


![c322 teaching panel 03 (original).](../assets/figures/ml_fig_c322_03.png)
*Figure — Animation frame residual c322. Synthetic teaching geometry—not a causal claim.*


![c323 teaching panel 03 (original).](../assets/figures/ml_fig_c323_03.png)
*Figure — Facet grid scatter c323. Synthetic teaching geometry—not a causal claim.*


![c324 teaching panel 03 (original).](../assets/figures/ml_fig_c324_03.png)
*Figure — Raincloud density path c324. Synthetic teaching geometry—not a causal claim.*


![c325 teaching panel 03 (original).](../assets/figures/ml_fig_c325_03.png)
*Figure — QQ residual path c325. Synthetic teaching geometry—not a causal claim.*


![c326 teaching panel 03 (original).](../assets/figures/ml_fig_c326_03.png)
*Figure — ECDF comparison path c326. Synthetic teaching geometry—not a causal claim.*


![c327 teaching panel 03 (original).](../assets/figures/ml_fig_c327_03.png)
*Figure — Color map accessibility heat c327. Synthetic teaching geometry—not a causal claim.*


![c328 teaching panel 03 (original).](../assets/figures/ml_fig_c328_03.png)
*Figure — Small multiples layout path c328. Synthetic teaching geometry—not a causal claim.*


![c329 teaching panel 03 (original).](../assets/figures/ml_fig_c329_03.png)
*Figure — Linked brushing scatter c329. Synthetic teaching geometry—not a causal claim.*


![c330 teaching panel 03 (original).](../assets/figures/ml_fig_c330_03.png)
*Figure — Parallel coords bars c330. Synthetic teaching geometry—not a causal claim.*


![c331 teaching panel 03 (original).](../assets/figures/ml_fig_c331_03.png)
*Figure — Radar chart misuse bars c331. Synthetic teaching geometry—not a causal claim.*


![c332 teaching panel 03 (original).](../assets/figures/ml_fig_c332_03.png)
*Figure — Uncertainty band path c332. Synthetic teaching geometry—not a causal claim.*


![c333 teaching panel 03 (original).](../assets/figures/ml_fig_c333_03.png)
*Figure — Log-scale distortion path c333. Synthetic teaching geometry—not a causal claim.*


![c334 teaching panel 03 (original).](../assets/figures/ml_fig_c334_03.png)
*Figure — Glyph encoding scatter c334. Synthetic teaching geometry—not a causal claim.*


![c335 teaching panel 03 (original).](../assets/figures/ml_fig_c335_03.png)
*Figure — Treemap hierarchy heat c335. Synthetic teaching geometry—not a causal claim.*


![c336 teaching panel 03 (original).](../assets/figures/ml_fig_c336_03.png)
*Figure — Sankey flow path c336. Synthetic teaching geometry—not a causal claim.*


![c337 teaching panel 03 (original).](../assets/figures/ml_fig_c337_03.png)
*Figure — Calibrated color heat c337. Synthetic teaching geometry—not a causal claim.*


![c338 teaching panel 03 (original).](../assets/figures/ml_fig_c338_03.png)
*Figure — Animation frame residual c338. Synthetic teaching geometry—not a causal claim.*


![c339 teaching panel 03 (original).](../assets/figures/ml_fig_c339_03.png)
*Figure — Facet grid scatter c339. Synthetic teaching geometry—not a causal claim.*


![c340 teaching panel 03 (original).](../assets/figures/ml_fig_c340_03.png)
*Figure — Raincloud density path c340. Synthetic teaching geometry—not a causal claim.*


![c341 teaching panel 03 (original).](../assets/figures/ml_fig_c341_03.png)
*Figure — QQ residual path c341. Synthetic teaching geometry—not a causal claim.*


![c342 teaching panel 03 (original).](../assets/figures/ml_fig_c342_03.png)
*Figure — ECDF comparison path c342. Synthetic teaching geometry—not a causal claim.*


![c343 teaching panel 03 (original).](../assets/figures/ml_fig_c343_03.png)
*Figure — Color map accessibility heat c343. Synthetic teaching geometry—not a causal claim.*


![c344 teaching panel 03 (original).](../assets/figures/ml_fig_c344_03.png)
*Figure — Small multiples layout path c344. Synthetic teaching geometry—not a causal claim.*


![c345 teaching panel 03 (original).](../assets/figures/ml_fig_c345_03.png)
*Figure — Linked brushing scatter c345. Synthetic teaching geometry—not a causal claim.*


![c346 teaching panel 03 (original).](../assets/figures/ml_fig_c346_03.png)
*Figure — Parallel coords bars c346. Synthetic teaching geometry—not a causal claim.*


![c347 teaching panel 03 (original).](../assets/figures/ml_fig_c347_03.png)
*Figure — Radar chart misuse bars c347. Synthetic teaching geometry—not a causal claim.*


![c348 teaching panel 03 (original).](../assets/figures/ml_fig_c348_03.png)
*Figure — Uncertainty band path c348. Synthetic teaching geometry—not a causal claim.*


![c349 teaching panel 03 (original).](../assets/figures/ml_fig_c349_03.png)
*Figure — Log-scale distortion path c349. Synthetic teaching geometry—not a causal claim.*


![c350 teaching panel 03 (original).](../assets/figures/ml_fig_c350_03.png)
*Figure — Glyph encoding scatter c350. Synthetic teaching geometry—not a causal claim.*


![c351 teaching panel 03 (original).](../assets/figures/ml_fig_c351_03.png)
*Figure — Treemap hierarchy heat c351. Synthetic teaching geometry—not a causal claim.*


![c352 teaching panel 03 (original).](../assets/figures/ml_fig_c352_03.png)
*Figure — Sankey flow path c352. Synthetic teaching geometry—not a causal claim.*


![c353 teaching panel 03 (original).](../assets/figures/ml_fig_c353_03.png)
*Figure — Calibrated color heat c353. Synthetic teaching geometry—not a causal claim.*


![c354 teaching panel 03 (original).](../assets/figures/ml_fig_c354_03.png)
*Figure — Animation frame residual c354. Synthetic teaching geometry—not a causal claim.*


![c355 teaching panel 03 (original).](../assets/figures/ml_fig_c355_03.png)
*Figure — Facet grid scatter c355. Synthetic teaching geometry—not a causal claim.*


![c356 teaching panel 03 (original).](../assets/figures/ml_fig_c356_03.png)
*Figure — Raincloud density path c356. Synthetic teaching geometry—not a causal claim.*


![c357 teaching panel 03 (original).](../assets/figures/ml_fig_c357_03.png)
*Figure — QQ residual path c357. Synthetic teaching geometry—not a causal claim.*


![c358 teaching panel 03 (original).](../assets/figures/ml_fig_c358_03.png)
*Figure — ECDF comparison path c358. Synthetic teaching geometry—not a causal claim.*


![c359 teaching panel 03 (original).](../assets/figures/ml_fig_c359_03.png)
*Figure — Color map accessibility heat c359. Synthetic teaching geometry—not a causal claim.*


![c360 teaching panel 03 (original).](../assets/figures/ml_fig_c360_03.png)
*Figure — Small multiples layout path c360. Synthetic teaching geometry—not a causal claim.*


![c361 teaching panel 03 (original).](../assets/figures/ml_fig_c361_03.png)
*Figure — Linked brushing scatter c361. Synthetic teaching geometry—not a causal claim.*


![c362 teaching panel 03 (original).](../assets/figures/ml_fig_c362_03.png)
*Figure — Parallel coords bars c362. Synthetic teaching geometry—not a causal claim.*


![c363 teaching panel 03 (original).](../assets/figures/ml_fig_c363_03.png)
*Figure — Radar chart misuse bars c363. Synthetic teaching geometry—not a causal claim.*


![c364 teaching panel 03 (original).](../assets/figures/ml_fig_c364_03.png)
*Figure — Uncertainty band path c364. Synthetic teaching geometry—not a causal claim.*


![c365 teaching panel 03 (original).](../assets/figures/ml_fig_c365_03.png)
*Figure — Log-scale distortion path c365. Synthetic teaching geometry—not a causal claim.*


![c366 teaching panel 03 (original).](../assets/figures/ml_fig_c366_03.png)
*Figure — Glyph encoding scatter c366. Synthetic teaching geometry—not a causal claim.*


![c367 teaching panel 03 (original).](../assets/figures/ml_fig_c367_03.png)
*Figure — Treemap hierarchy heat c367. Synthetic teaching geometry—not a causal claim.*


![c368 teaching panel 03 (original).](../assets/figures/ml_fig_c368_03.png)
*Figure — Sankey flow path c368. Synthetic teaching geometry—not a causal claim.*


![c369 teaching panel 03 (original).](../assets/figures/ml_fig_c369_03.png)
*Figure — Calibrated color heat c369. Synthetic teaching geometry—not a causal claim.*


![c370 teaching panel 03 (original).](../assets/figures/ml_fig_c370_03.png)
*Figure — Animation frame residual c370. Synthetic teaching geometry—not a causal claim.*


![c371 teaching panel 03 (original).](../assets/figures/ml_fig_c371_03.png)
*Figure — Facet grid scatter c371. Synthetic teaching geometry—not a causal claim.*


![c372 teaching panel 03 (original).](../assets/figures/ml_fig_c372_03.png)
*Figure — Raincloud density path c372. Synthetic teaching geometry—not a causal claim.*


![c373 teaching panel 03 (original).](../assets/figures/ml_fig_c373_03.png)
*Figure — QQ residual path c373. Synthetic teaching geometry—not a causal claim.*


![c374 teaching panel 03 (original).](../assets/figures/ml_fig_c374_03.png)
*Figure — ECDF comparison path c374. Synthetic teaching geometry—not a causal claim.*


![c375 teaching panel 03 (original).](../assets/figures/ml_fig_c375_03.png)
*Figure — Color map accessibility heat c375. Synthetic teaching geometry—not a causal claim.*


![c376 teaching panel 03 (original).](../assets/figures/ml_fig_c376_03.png)
*Figure — Small multiples layout path c376. Synthetic teaching geometry—not a causal claim.*


![c377 teaching panel 03 (original).](../assets/figures/ml_fig_c377_03.png)
*Figure — Linked brushing scatter c377. Synthetic teaching geometry—not a causal claim.*


![c378 teaching panel 03 (original).](../assets/figures/ml_fig_c378_03.png)
*Figure — Parallel coords bars c378. Synthetic teaching geometry—not a causal claim.*


![c379 teaching panel 03 (original).](../assets/figures/ml_fig_c379_03.png)
*Figure — Radar chart misuse bars c379. Synthetic teaching geometry—not a causal claim.*


![c380 teaching panel 03 (original).](../assets/figures/ml_fig_c380_03.png)
*Figure — Uncertainty band path c380. Synthetic teaching geometry—not a causal claim.*


![c381 teaching panel 03 (original).](../assets/figures/ml_fig_c381_03.png)
*Figure — Log-scale distortion path c381. Synthetic teaching geometry—not a causal claim.*


![c382 teaching panel 03 (original).](../assets/figures/ml_fig_c382_03.png)
*Figure — Glyph encoding scatter c382. Synthetic teaching geometry—not a causal claim.*


![c383 teaching panel 03 (original).](../assets/figures/ml_fig_c383_03.png)
*Figure — Treemap hierarchy heat c383. Synthetic teaching geometry—not a causal claim.*


![c384 teaching panel 03 (original).](../assets/figures/ml_fig_c384_03.png)
*Figure — Sankey flow path c384. Synthetic teaching geometry—not a causal claim.*


![c385 teaching panel 03 (original).](../assets/figures/ml_fig_c385_03.png)
*Figure — Calibrated color heat c385. Synthetic teaching geometry—not a causal claim.*


![c386 teaching panel 03 (original).](../assets/figures/ml_fig_c386_03.png)
*Figure — Animation frame residual c386. Synthetic teaching geometry—not a causal claim.*


![c387 teaching panel 03 (original).](../assets/figures/ml_fig_c387_03.png)
*Figure — Facet grid scatter c387. Synthetic teaching geometry—not a causal claim.*


![c388 teaching panel 03 (original).](../assets/figures/ml_fig_c388_03.png)
*Figure — Raincloud density path c388. Synthetic teaching geometry—not a causal claim.*


![c389 teaching panel 03 (original).](../assets/figures/ml_fig_c389_03.png)
*Figure — QQ residual path c389. Synthetic teaching geometry—not a causal claim.*


![c390 teaching panel 03 (original).](../assets/figures/ml_fig_c390_03.png)
*Figure — ECDF comparison path c390. Synthetic teaching geometry—not a causal claim.*


![c391 teaching panel 03 (original).](../assets/figures/ml_fig_c391_03.png)
*Figure — Color map accessibility heat c391. Synthetic teaching geometry—not a causal claim.*


![c392 teaching panel 03 (original).](../assets/figures/ml_fig_c392_03.png)
*Figure — Small multiples layout path c392. Synthetic teaching geometry—not a causal claim.*


![c393 teaching panel 03 (original).](../assets/figures/ml_fig_c393_03.png)
*Figure — Linked brushing scatter c393. Synthetic teaching geometry—not a causal claim.*


![c394 teaching panel 03 (original).](../assets/figures/ml_fig_c394_03.png)
*Figure — Parallel coords bars c394. Synthetic teaching geometry—not a causal claim.*


![c395 teaching panel 03 (original).](../assets/figures/ml_fig_c395_03.png)
*Figure — Radar chart misuse bars c395. Synthetic teaching geometry—not a causal claim.*


![c396 teaching panel 03 (original).](../assets/figures/ml_fig_c396_03.png)
*Figure — Uncertainty band path c396. Synthetic teaching geometry—not a causal claim.*


![c397 teaching panel 03 (original).](../assets/figures/ml_fig_c397_03.png)
*Figure — Log-scale distortion path c397. Synthetic teaching geometry—not a causal claim.*


![c398 teaching panel 03 (original).](../assets/figures/ml_fig_c398_03.png)
*Figure — Glyph encoding scatter c398. Synthetic teaching geometry—not a causal claim.*


![c399 teaching panel 03 (original).](../assets/figures/ml_fig_c399_03.png)
*Figure — Treemap hierarchy heat c399. Synthetic teaching geometry—not a causal claim.*


![c400 teaching panel 03 (original).](../assets/figures/ml_fig_c400_03.png)
*Figure — Sankey flow path c400. Synthetic teaching geometry—not a causal claim.*


![c401 teaching panel 03 (original).](../assets/figures/ml_fig_c401_03.png)
*Figure — Calibrated color heat c401. Synthetic teaching geometry—not a causal claim.*


![c402 teaching panel 03 (original).](../assets/figures/ml_fig_c402_03.png)
*Figure — Animation frame residual c402. Synthetic teaching geometry—not a causal claim.*


![c403 teaching panel 03 (original).](../assets/figures/ml_fig_c403_03.png)
*Figure — Facet grid scatter c403. Synthetic teaching geometry—not a causal claim.*


![c404 teaching panel 03 (original).](../assets/figures/ml_fig_c404_03.png)
*Figure — Raincloud density path c404. Synthetic teaching geometry—not a causal claim.*


![c405 teaching panel 03 (original).](../assets/figures/ml_fig_c405_03.png)
*Figure — QQ residual path c405. Synthetic teaching geometry—not a causal claim.*


![c406 teaching panel 03 (original).](../assets/figures/ml_fig_c406_03.png)
*Figure — ECDF comparison path c406. Synthetic teaching geometry—not a causal claim.*


![c407 teaching panel 03 (original).](../assets/figures/ml_fig_c407_03.png)
*Figure — Color map accessibility heat c407. Synthetic teaching geometry—not a causal claim.*


![c408 teaching panel 03 (original).](../assets/figures/ml_fig_c408_03.png)
*Figure — Small multiples layout path c408. Synthetic teaching geometry—not a causal claim.*


![c409 teaching panel 03 (original).](../assets/figures/ml_fig_c409_03.png)
*Figure — Linked brushing scatter c409. Synthetic teaching geometry—not a causal claim.*


![c410 teaching panel 03 (original).](../assets/figures/ml_fig_c410_03.png)
*Figure — Parallel coords bars c410. Synthetic teaching geometry—not a causal claim.*


![c411 teaching panel 03 (original).](../assets/figures/ml_fig_c411_03.png)
*Figure — Radar chart misuse bars c411. Synthetic teaching geometry—not a causal claim.*


![c412 teaching panel 03 (original).](../assets/figures/ml_fig_c412_03.png)
*Figure — Uncertainty band path c412. Synthetic teaching geometry—not a causal claim.*


![c413 teaching panel 03 (original).](../assets/figures/ml_fig_c413_03.png)
*Figure — Log-scale distortion path c413. Synthetic teaching geometry—not a causal claim.*


![c414 teaching panel 03 (original).](../assets/figures/ml_fig_c414_03.png)
*Figure — Glyph encoding scatter c414. Synthetic teaching geometry—not a causal claim.*


![c415 teaching panel 03 (original).](../assets/figures/ml_fig_c415_03.png)
*Figure — Treemap hierarchy heat c415. Synthetic teaching geometry—not a causal claim.*


![c416 teaching panel 03 (original).](../assets/figures/ml_fig_c416_03.png)
*Figure — Sankey flow path c416. Synthetic teaching geometry—not a causal claim.*


![c417 teaching panel 03 (original).](../assets/figures/ml_fig_c417_03.png)
*Figure — Calibrated color heat c417. Synthetic teaching geometry—not a causal claim.*


![c418 teaching panel 03 (original).](../assets/figures/ml_fig_c418_03.png)
*Figure — Animation frame residual c418. Synthetic teaching geometry—not a causal claim.*


![c419 teaching panel 03 (original).](../assets/figures/ml_fig_c419_03.png)
*Figure — Facet grid scatter c419. Synthetic teaching geometry—not a causal claim.*


![c420 teaching panel 03 (original).](../assets/figures/ml_fig_c420_03.png)
*Figure — Raincloud density path c420. Synthetic teaching geometry—not a causal claim.*


![c421 teaching panel 03 (original).](../assets/figures/ml_fig_c421_03.png)
*Figure — QQ residual path c421. Synthetic teaching geometry—not a causal claim.*


![c422 teaching panel 03 (original).](../assets/figures/ml_fig_c422_03.png)
*Figure — ECDF comparison path c422. Synthetic teaching geometry—not a causal claim.*


![c423 teaching panel 03 (original).](../assets/figures/ml_fig_c423_03.png)
*Figure — Color map accessibility heat c423. Synthetic teaching geometry—not a causal claim.*


![c424 teaching panel 03 (original).](../assets/figures/ml_fig_c424_03.png)
*Figure — Small multiples layout path c424. Synthetic teaching geometry—not a causal claim.*


![c425 teaching panel 03 (original).](../assets/figures/ml_fig_c425_03.png)
*Figure — Linked brushing scatter c425. Synthetic teaching geometry—not a causal claim.*


![c426 teaching panel 03 (original).](../assets/figures/ml_fig_c426_03.png)
*Figure — Parallel coords bars c426. Synthetic teaching geometry—not a causal claim.*


![c427 teaching panel 03 (original).](../assets/figures/ml_fig_c427_03.png)
*Figure — Radar chart misuse bars c427. Synthetic teaching geometry—not a causal claim.*


![c428 teaching panel 03 (original).](../assets/figures/ml_fig_c428_03.png)
*Figure — Uncertainty band path c428. Synthetic teaching geometry—not a causal claim.*


![c429 teaching panel 03 (original).](../assets/figures/ml_fig_c429_03.png)
*Figure — Log-scale distortion path c429. Synthetic teaching geometry—not a causal claim.*


![c430 teaching panel 03 (original).](../assets/figures/ml_fig_c430_03.png)
*Figure — Glyph encoding scatter c430. Synthetic teaching geometry—not a causal claim.*


![c431 teaching panel 03 (original).](../assets/figures/ml_fig_c431_03.png)
*Figure — Treemap hierarchy heat c431. Synthetic teaching geometry—not a causal claim.*


![c432 teaching panel 03 (original).](../assets/figures/ml_fig_c432_03.png)
*Figure — Sankey flow path c432. Synthetic teaching geometry—not a causal claim.*


![c433 teaching panel 03 (original).](../assets/figures/ml_fig_c433_03.png)
*Figure — Calibrated color heat c433. Synthetic teaching geometry—not a causal claim.*


![c434 teaching panel 03 (original).](../assets/figures/ml_fig_c434_03.png)
*Figure — Animation frame residual c434. Synthetic teaching geometry—not a causal claim.*


![c435 teaching panel 03 (original).](../assets/figures/ml_fig_c435_03.png)
*Figure — Facet grid scatter c435. Synthetic teaching geometry—not a causal claim.*


![c436 teaching panel 03 (original).](../assets/figures/ml_fig_c436_03.png)
*Figure — Raincloud density path c436. Synthetic teaching geometry—not a causal claim.*


![c437 teaching panel 03 (original).](../assets/figures/ml_fig_c437_03.png)
*Figure — QQ residual path c437. Synthetic teaching geometry—not a causal claim.*


![c438 teaching panel 03 (original).](../assets/figures/ml_fig_c438_03.png)
*Figure — ECDF comparison path c438. Synthetic teaching geometry—not a causal claim.*


![c439 teaching panel 03 (original).](../assets/figures/ml_fig_c439_03.png)
*Figure — Color map accessibility heat c439. Synthetic teaching geometry—not a causal claim.*


![c440 teaching panel 03 (original).](../assets/figures/ml_fig_c440_03.png)
*Figure — Small multiples layout path c440. Synthetic teaching geometry—not a causal claim.*


![c441 teaching panel 03 (original).](../assets/figures/ml_fig_c441_03.png)
*Figure — Linked brushing scatter c441. Synthetic teaching geometry—not a causal claim.*


![c442 teaching panel 03 (original).](../assets/figures/ml_fig_c442_03.png)
*Figure — Parallel coords bars c442. Synthetic teaching geometry—not a causal claim.*


![c443 teaching panel 03 (original).](../assets/figures/ml_fig_c443_03.png)
*Figure — Radar chart misuse bars c443. Synthetic teaching geometry—not a causal claim.*


![c444 teaching panel 03 (original).](../assets/figures/ml_fig_c444_03.png)
*Figure — Uncertainty band path c444. Synthetic teaching geometry—not a causal claim.*


![c445 teaching panel 03 (original).](../assets/figures/ml_fig_c445_03.png)
*Figure — Log-scale distortion path c445. Synthetic teaching geometry—not a causal claim.*


![c446 teaching panel 03 (original).](../assets/figures/ml_fig_c446_03.png)
*Figure — Glyph encoding scatter c446. Synthetic teaching geometry—not a causal claim.*


![c447 teaching panel 03 (original).](../assets/figures/ml_fig_c447_03.png)
*Figure — Treemap hierarchy heat c447. Synthetic teaching geometry—not a causal claim.*


![c448 teaching panel 03 (original).](../assets/figures/ml_fig_c448_03.png)
*Figure — Sankey flow path c448. Synthetic teaching geometry—not a causal claim.*


![c449 teaching panel 03 (original).](../assets/figures/ml_fig_c449_03.png)
*Figure — Calibrated color heat c449. Synthetic teaching geometry—not a causal claim.*


![c450 teaching panel 03 (original).](../assets/figures/ml_fig_c450_03.png)
*Figure — Animation frame residual c450. Synthetic teaching geometry—not a causal claim.*


![c451 teaching panel 03 (original).](../assets/figures/ml_fig_c451_03.png)
*Figure — Facet grid scatter c451. Synthetic teaching geometry—not a causal claim.*


![c452 teaching panel 03 (original).](../assets/figures/ml_fig_c452_03.png)
*Figure — Raincloud density path c452. Synthetic teaching geometry—not a causal claim.*


![c453 teaching panel 03 (original).](../assets/figures/ml_fig_c453_03.png)
*Figure — QQ residual path c453. Synthetic teaching geometry—not a causal claim.*


![c454 teaching panel 03 (original).](../assets/figures/ml_fig_c454_03.png)
*Figure — ECDF comparison path c454. Synthetic teaching geometry—not a causal claim.*


![c455 teaching panel 03 (original).](../assets/figures/ml_fig_c455_03.png)
*Figure — Color map accessibility heat c455. Synthetic teaching geometry—not a causal claim.*


![c456 teaching panel 03 (original).](../assets/figures/ml_fig_c456_03.png)
*Figure — Small multiples layout path c456. Synthetic teaching geometry—not a causal claim.*


![c457 teaching panel 03 (original).](../assets/figures/ml_fig_c457_03.png)
*Figure — Linked brushing scatter c457. Synthetic teaching geometry—not a causal claim.*


![c458 teaching panel 03 (original).](../assets/figures/ml_fig_c458_03.png)
*Figure — Parallel coords bars c458. Synthetic teaching geometry—not a causal claim.*


![c459 teaching panel 03 (original).](../assets/figures/ml_fig_c459_03.png)
*Figure — Radar chart misuse bars c459. Synthetic teaching geometry—not a causal claim.*


![c460 teaching panel 03 (original).](../assets/figures/ml_fig_c460_03.png)
*Figure — Uncertainty band path c460. Synthetic teaching geometry—not a causal claim.*


![c461 teaching panel 03 (original).](../assets/figures/ml_fig_c461_03.png)
*Figure — Log-scale distortion path c461. Synthetic teaching geometry—not a causal claim.*


![c462 teaching panel 03 (original).](../assets/figures/ml_fig_c462_03.png)
*Figure — Glyph encoding scatter c462. Synthetic teaching geometry—not a causal claim.*


![c463 teaching panel 03 (original).](../assets/figures/ml_fig_c463_03.png)
*Figure — Treemap hierarchy heat c463. Synthetic teaching geometry—not a causal claim.*


![c464 teaching panel 03 (original).](../assets/figures/ml_fig_c464_03.png)
*Figure — Sankey flow path c464. Synthetic teaching geometry—not a causal claim.*


![c465 teaching panel 03 (original).](../assets/figures/ml_fig_c465_03.png)
*Figure — Calibrated color heat c465. Synthetic teaching geometry—not a causal claim.*


![c466 teaching panel 03 (original).](../assets/figures/ml_fig_c466_03.png)
*Figure — Animation frame residual c466. Synthetic teaching geometry—not a causal claim.*


![c467 teaching panel 03 (original).](../assets/figures/ml_fig_c467_03.png)
*Figure — Facet grid scatter c467. Synthetic teaching geometry—not a causal claim.*


![c468 teaching panel 03 (original).](../assets/figures/ml_fig_c468_03.png)
*Figure — Raincloud density path c468. Synthetic teaching geometry—not a causal claim.*


![c469 teaching panel 03 (original).](../assets/figures/ml_fig_c469_03.png)
*Figure — QQ residual path c469. Synthetic teaching geometry—not a causal claim.*


![c470 teaching panel 03 (original).](../assets/figures/ml_fig_c470_03.png)
*Figure — ECDF comparison path c470. Synthetic teaching geometry—not a causal claim.*


![c471 teaching panel 03 (original).](../assets/figures/ml_fig_c471_03.png)
*Figure — Color map accessibility heat c471. Synthetic teaching geometry—not a causal claim.*


![c472 teaching panel 03 (original).](../assets/figures/ml_fig_c472_03.png)
*Figure — Small multiples layout path c472. Synthetic teaching geometry—not a causal claim.*


![c473 teaching panel 03 (original).](../assets/figures/ml_fig_c473_03.png)
*Figure — Linked brushing scatter c473. Synthetic teaching geometry—not a causal claim.*


![c474 teaching panel 03 (original).](../assets/figures/ml_fig_c474_03.png)
*Figure — Parallel coords bars c474. Synthetic teaching geometry—not a causal claim.*


![c475 teaching panel 03 (original).](../assets/figures/ml_fig_c475_03.png)
*Figure — Radar chart misuse bars c475. Synthetic teaching geometry—not a causal claim.*


![c476 teaching panel 03 (original).](../assets/figures/ml_fig_c476_03.png)
*Figure — Uncertainty band path c476. Synthetic teaching geometry—not a causal claim.*


![c477 teaching panel 03 (original).](../assets/figures/ml_fig_c477_03.png)
*Figure — Log-scale distortion path c477. Synthetic teaching geometry—not a causal claim.*


![c478 teaching panel 03 (original).](../assets/figures/ml_fig_c478_03.png)
*Figure — Glyph encoding scatter c478. Synthetic teaching geometry—not a causal claim.*


![c479 teaching panel 03 (original).](../assets/figures/ml_fig_c479_03.png)
*Figure — Treemap hierarchy heat c479. Synthetic teaching geometry—not a causal claim.*


![c480 teaching panel 03 (original).](../assets/figures/ml_fig_c480_03.png)
*Figure — Sankey flow path c480. Synthetic teaching geometry—not a causal claim.*


![c481 teaching panel 03 (original).](../assets/figures/ml_fig_c481_03.png)
*Figure — Calibrated color heat c481. Synthetic teaching geometry—not a causal claim.*


![c482 teaching panel 03 (original).](../assets/figures/ml_fig_c482_03.png)
*Figure — Animation frame residual c482. Synthetic teaching geometry—not a causal claim.*


![c483 teaching panel 03 (original).](../assets/figures/ml_fig_c483_03.png)
*Figure — Facet grid scatter c483. Synthetic teaching geometry—not a causal claim.*


![c484 teaching panel 03 (original).](../assets/figures/ml_fig_c484_03.png)
*Figure — Raincloud density path c484. Synthetic teaching geometry—not a causal claim.*


![c485 teaching panel 03 (original).](../assets/figures/ml_fig_c485_03.png)
*Figure — QQ residual path c485. Synthetic teaching geometry—not a causal claim.*


![c486 teaching panel 03 (original).](../assets/figures/ml_fig_c486_03.png)
*Figure — ECDF comparison path c486. Synthetic teaching geometry—not a causal claim.*


![c487 teaching panel 03 (original).](../assets/figures/ml_fig_c487_03.png)
*Figure — Color map accessibility heat c487. Synthetic teaching geometry—not a causal claim.*


![c488 teaching panel 03 (original).](../assets/figures/ml_fig_c488_03.png)
*Figure — Small multiples layout path c488. Synthetic teaching geometry—not a causal claim.*


![c489 teaching panel 03 (original).](../assets/figures/ml_fig_c489_03.png)
*Figure — Linked brushing scatter c489. Synthetic teaching geometry—not a causal claim.*


![c490 teaching panel 03 (original).](../assets/figures/ml_fig_c490_03.png)
*Figure — Parallel coords bars c490. Synthetic teaching geometry—not a causal claim.*


![c491 teaching panel 03 (original).](../assets/figures/ml_fig_c491_03.png)
*Figure — Radar chart misuse bars c491. Synthetic teaching geometry—not a causal claim.*


![c492 teaching panel 03 (original).](../assets/figures/ml_fig_c492_03.png)
*Figure — Uncertainty band path c492. Synthetic teaching geometry—not a causal claim.*


![c493 teaching panel 03 (original).](../assets/figures/ml_fig_c493_03.png)
*Figure — Log-scale distortion path c493. Synthetic teaching geometry—not a causal claim.*


![c494 teaching panel 03 (original).](../assets/figures/ml_fig_c494_03.png)
*Figure — Glyph encoding scatter c494. Synthetic teaching geometry—not a causal claim.*


![c495 teaching panel 03 (original).](../assets/figures/ml_fig_c495_03.png)
*Figure — Treemap hierarchy heat c495. Synthetic teaching geometry—not a causal claim.*


![c496 teaching panel 03 (original).](../assets/figures/ml_fig_c496_03.png)
*Figure — Sankey flow path c496. Synthetic teaching geometry—not a causal claim.*


![c497 teaching panel 03 (original).](../assets/figures/ml_fig_c497_03.png)
*Figure — Calibrated color heat c497. Synthetic teaching geometry—not a causal claim.*


![c498 teaching panel 03 (original).](../assets/figures/ml_fig_c498_03.png)
*Figure — Animation frame residual c498. Synthetic teaching geometry—not a causal claim.*


![c499 teaching panel 03 (original).](../assets/figures/ml_fig_c499_03.png)
*Figure — Facet grid scatter c499. Synthetic teaching geometry—not a causal claim.*


![c500 teaching panel 03 (original).](../assets/figures/ml_fig_c500_03.png)
*Figure — Raincloud density path c500. Synthetic teaching geometry—not a causal claim.*


![c501 teaching panel 03 (original).](../assets/figures/ml_fig_c501_03.png)
*Figure — QQ residual path c501. Synthetic teaching geometry—not a causal claim.*


![c502 teaching panel 03 (original).](../assets/figures/ml_fig_c502_03.png)
*Figure — ECDF comparison path c502. Synthetic teaching geometry—not a causal claim.*


![c503 teaching panel 03 (original).](../assets/figures/ml_fig_c503_03.png)
*Figure — Color map accessibility heat c503. Synthetic teaching geometry—not a causal claim.*


![c504 teaching panel 03 (original).](../assets/figures/ml_fig_c504_03.png)
*Figure — Small multiples layout path c504. Synthetic teaching geometry—not a causal claim.*


![c505 teaching panel 03 (original).](../assets/figures/ml_fig_c505_03.png)
*Figure — Linked brushing scatter c505. Synthetic teaching geometry—not a causal claim.*


![c506 teaching panel 03 (original).](../assets/figures/ml_fig_c506_03.png)
*Figure — Parallel coords bars c506. Synthetic teaching geometry—not a causal claim.*


![c507 teaching panel 03 (original).](../assets/figures/ml_fig_c507_03.png)
*Figure — Radar chart misuse bars c507. Synthetic teaching geometry—not a causal claim.*


![c508 teaching panel 03 (original).](../assets/figures/ml_fig_c508_03.png)
*Figure — Uncertainty band path c508. Synthetic teaching geometry—not a causal claim.*


![c509 teaching panel 03 (original).](../assets/figures/ml_fig_c509_03.png)
*Figure — Log-scale distortion path c509. Synthetic teaching geometry—not a causal claim.*


![c510 teaching panel 03 (original).](../assets/figures/ml_fig_c510_03.png)
*Figure — Glyph encoding scatter c510. Synthetic teaching geometry—not a causal claim.*


![c511 teaching panel 03 (original).](../assets/figures/ml_fig_c511_03.png)
*Figure — Treemap hierarchy heat c511. Synthetic teaching geometry—not a causal claim.*


![c512 teaching panel 03 (original).](../assets/figures/ml_fig_c512_03.png)
*Figure — Sankey flow path c512. Synthetic teaching geometry—not a causal claim.*


![c513 teaching panel 03 (original).](../assets/figures/ml_fig_c513_03.png)
*Figure — Calibrated color heat c513. Synthetic teaching geometry—not a causal claim.*


![c514 teaching panel 03 (original).](../assets/figures/ml_fig_c514_03.png)
*Figure — Animation frame residual c514. Synthetic teaching geometry—not a causal claim.*


![c515 teaching panel 03 (original).](../assets/figures/ml_fig_c515_03.png)
*Figure — Facet grid scatter c515. Synthetic teaching geometry—not a causal claim.*


![c516 teaching panel 03 (original).](../assets/figures/ml_fig_c516_03.png)
*Figure — Raincloud density path c516. Synthetic teaching geometry—not a causal claim.*


![c517 teaching panel 03 (original).](../assets/figures/ml_fig_c517_03.png)
*Figure — QQ residual path c517. Synthetic teaching geometry—not a causal claim.*


![c518 teaching panel 03 (original).](../assets/figures/ml_fig_c518_03.png)
*Figure — ECDF comparison path c518. Synthetic teaching geometry—not a causal claim.*


![c519 teaching panel 03 (original).](../assets/figures/ml_fig_c519_03.png)
*Figure — Color map accessibility heat c519. Synthetic teaching geometry—not a causal claim.*


![c520 teaching panel 03 (original).](../assets/figures/ml_fig_c520_03.png)
*Figure — Small multiples layout path c520. Synthetic teaching geometry—not a causal claim.*


![c521 teaching panel 03 (original).](../assets/figures/ml_fig_c521_03.png)
*Figure — Linked brushing scatter c521. Synthetic teaching geometry—not a causal claim.*


![c522 teaching panel 03 (original).](../assets/figures/ml_fig_c522_03.png)
*Figure — Parallel coords bars c522. Synthetic teaching geometry—not a causal claim.*


![c523 teaching panel 03 (original).](../assets/figures/ml_fig_c523_03.png)
*Figure — Radar chart misuse bars c523. Synthetic teaching geometry—not a causal claim.*


![c524 teaching panel 03 (original).](../assets/figures/ml_fig_c524_03.png)
*Figure — Uncertainty band path c524. Synthetic teaching geometry—not a causal claim.*


![c525 teaching panel 03 (original).](../assets/figures/ml_fig_c525_03.png)
*Figure — Log-scale distortion path c525. Synthetic teaching geometry—not a causal claim.*


![c526 teaching panel 03 (original).](../assets/figures/ml_fig_c526_03.png)
*Figure — Glyph encoding scatter c526. Synthetic teaching geometry—not a causal claim.*


![c527 teaching panel 03 (original).](../assets/figures/ml_fig_c527_03.png)
*Figure — Treemap hierarchy heat c527. Synthetic teaching geometry—not a causal claim.*


![c528 teaching panel 03 (original).](../assets/figures/ml_fig_c528_03.png)
*Figure — Sankey flow path c528. Synthetic teaching geometry—not a causal claim.*


![c529 teaching panel 03 (original).](../assets/figures/ml_fig_c529_03.png)
*Figure — Calibrated color heat c529. Synthetic teaching geometry—not a causal claim.*


![c530 teaching panel 03 (original).](../assets/figures/ml_fig_c530_03.png)
*Figure — Animation frame residual c530. Synthetic teaching geometry—not a causal claim.*


![c531 teaching panel 03 (original).](../assets/figures/ml_fig_c531_03.png)
*Figure — Facet grid scatter c531. Synthetic teaching geometry—not a causal claim.*


![c532 teaching panel 03 (original).](../assets/figures/ml_fig_c532_03.png)
*Figure — Raincloud density path c532. Synthetic teaching geometry—not a causal claim.*


![c533 teaching panel 03 (original).](../assets/figures/ml_fig_c533_03.png)
*Figure — QQ residual path c533. Synthetic teaching geometry—not a causal claim.*


![c534 teaching panel 03 (original).](../assets/figures/ml_fig_c534_03.png)
*Figure — ECDF comparison path c534. Synthetic teaching geometry—not a causal claim.*


![c535 teaching panel 03 (original).](../assets/figures/ml_fig_c535_03.png)
*Figure — Color map accessibility heat c535. Synthetic teaching geometry—not a causal claim.*


![c536 teaching panel 03 (original).](../assets/figures/ml_fig_c536_03.png)
*Figure — Small multiples layout path c536. Synthetic teaching geometry—not a causal claim.*


![c537 teaching panel 03 (original).](../assets/figures/ml_fig_c537_03.png)
*Figure — Linked brushing scatter c537. Synthetic teaching geometry—not a causal claim.*


![c538 teaching panel 03 (original).](../assets/figures/ml_fig_c538_03.png)
*Figure — Parallel coords bars c538. Synthetic teaching geometry—not a causal claim.*


![c539 teaching panel 03 (original).](../assets/figures/ml_fig_c539_03.png)
*Figure — Radar chart misuse bars c539. Synthetic teaching geometry—not a causal claim.*


![c540 teaching panel 03 (original).](../assets/figures/ml_fig_c540_03.png)
*Figure — Uncertainty band path c540. Synthetic teaching geometry—not a causal claim.*


![c541 teaching panel 03 (original).](../assets/figures/ml_fig_c541_03.png)
*Figure — Log-scale distortion path c541. Synthetic teaching geometry—not a causal claim.*


![c542 teaching panel 03 (original).](../assets/figures/ml_fig_c542_03.png)
*Figure — Glyph encoding scatter c542. Synthetic teaching geometry—not a causal claim.*


![c543 teaching panel 03 (original).](../assets/figures/ml_fig_c543_03.png)
*Figure — Treemap hierarchy heat c543. Synthetic teaching geometry—not a causal claim.*


![c544 teaching panel 03 (original).](../assets/figures/ml_fig_c544_03.png)
*Figure — Sankey flow path c544. Synthetic teaching geometry—not a causal claim.*


![c545 teaching panel 03 (original).](../assets/figures/ml_fig_c545_03.png)
*Figure — Calibrated color heat c545. Synthetic teaching geometry—not a causal claim.*


![c546 teaching panel 03 (original).](../assets/figures/ml_fig_c546_03.png)
*Figure — Animation frame residual c546. Synthetic teaching geometry—not a causal claim.*


![c547 teaching panel 03 (original).](../assets/figures/ml_fig_c547_03.png)
*Figure — Facet grid scatter c547. Synthetic teaching geometry—not a causal claim.*


![c548 teaching panel 03 (original).](../assets/figures/ml_fig_c548_03.png)
*Figure — Raincloud density path c548. Synthetic teaching geometry—not a causal claim.*


![c549 teaching panel 03 (original).](../assets/figures/ml_fig_c549_03.png)
*Figure — QQ residual path c549. Synthetic teaching geometry—not a causal claim.*


![c550 teaching panel 03 (original).](../assets/figures/ml_fig_c550_03.png)
*Figure — ECDF comparison path c550. Synthetic teaching geometry—not a causal claim.*


![c551 teaching panel 03 (original).](../assets/figures/ml_fig_c551_03.png)
*Figure — Color map accessibility heat c551. Synthetic teaching geometry—not a causal claim.*


![c552 teaching panel 03 (original).](../assets/figures/ml_fig_c552_03.png)
*Figure — Small multiples layout path c552. Synthetic teaching geometry—not a causal claim.*


![c553 teaching panel 03 (original).](../assets/figures/ml_fig_c553_03.png)
*Figure — Linked brushing scatter c553. Synthetic teaching geometry—not a causal claim.*


![c554 teaching panel 03 (original).](../assets/figures/ml_fig_c554_03.png)
*Figure — Parallel coords bars c554. Synthetic teaching geometry—not a causal claim.*


![c555 teaching panel 03 (original).](../assets/figures/ml_fig_c555_03.png)
*Figure — Radar chart misuse bars c555. Synthetic teaching geometry—not a causal claim.*


![c556 teaching panel 03 (original).](../assets/figures/ml_fig_c556_03.png)
*Figure — Uncertainty band path c556. Synthetic teaching geometry—not a causal claim.*


![c557 teaching panel 03 (original).](../assets/figures/ml_fig_c557_03.png)
*Figure — Log-scale distortion path c557. Synthetic teaching geometry—not a causal claim.*


![c558 teaching panel 03 (original).](../assets/figures/ml_fig_c558_03.png)
*Figure — Glyph encoding scatter c558. Synthetic teaching geometry—not a causal claim.*


![c559 teaching panel 03 (original).](../assets/figures/ml_fig_c559_03.png)
*Figure — Treemap hierarchy heat c559. Synthetic teaching geometry—not a causal claim.*


![c560 teaching panel 03 (original).](../assets/figures/ml_fig_c560_03.png)
*Figure — Sankey flow path c560. Synthetic teaching geometry—not a causal claim.*


![c561 teaching panel 03 (original).](../assets/figures/ml_fig_c561_03.png)
*Figure — Calibrated color heat c561. Synthetic teaching geometry—not a causal claim.*


![c562 teaching panel 03 (original).](../assets/figures/ml_fig_c562_03.png)
*Figure — Animation frame residual c562. Synthetic teaching geometry—not a causal claim.*


![c563 teaching panel 03 (original).](../assets/figures/ml_fig_c563_03.png)
*Figure — Facet grid scatter c563. Synthetic teaching geometry—not a causal claim.*


![c564 teaching panel 03 (original).](../assets/figures/ml_fig_c564_03.png)
*Figure — Raincloud density path c564. Synthetic teaching geometry—not a causal claim.*


![c565 teaching panel 03 (original).](../assets/figures/ml_fig_c565_03.png)
*Figure — QQ residual path c565. Synthetic teaching geometry—not a causal claim.*


![c566 teaching panel 03 (original).](../assets/figures/ml_fig_c566_03.png)
*Figure — ECDF comparison path c566. Synthetic teaching geometry—not a causal claim.*


![c567 teaching panel 03 (original).](../assets/figures/ml_fig_c567_03.png)
*Figure — Color map accessibility heat c567. Synthetic teaching geometry—not a causal claim.*


![c568 teaching panel 03 (original).](../assets/figures/ml_fig_c568_03.png)
*Figure — Small multiples layout path c568. Synthetic teaching geometry—not a causal claim.*


![c569 teaching panel 03 (original).](../assets/figures/ml_fig_c569_03.png)
*Figure — Linked brushing scatter c569. Synthetic teaching geometry—not a causal claim.*


![c570 teaching panel 03 (original).](../assets/figures/ml_fig_c570_03.png)
*Figure — Parallel coords bars c570. Synthetic teaching geometry—not a causal claim.*


![c571 teaching panel 03 (original).](../assets/figures/ml_fig_c571_03.png)
*Figure — Radar chart misuse bars c571. Synthetic teaching geometry—not a causal claim.*


![c572 teaching panel 03 (original).](../assets/figures/ml_fig_c572_03.png)
*Figure — Uncertainty band path c572. Synthetic teaching geometry—not a causal claim.*


![c573 teaching panel 03 (original).](../assets/figures/ml_fig_c573_03.png)
*Figure — Log-scale distortion path c573. Synthetic teaching geometry—not a causal claim.*


![c574 teaching panel 03 (original).](../assets/figures/ml_fig_c574_03.png)
*Figure — Glyph encoding scatter c574. Synthetic teaching geometry—not a causal claim.*


![c575 teaching panel 03 (original).](../assets/figures/ml_fig_c575_03.png)
*Figure — Treemap hierarchy heat c575. Synthetic teaching geometry—not a causal claim.*


![c576 teaching panel 03 (original).](../assets/figures/ml_fig_c576_03.png)
*Figure — Sankey flow path c576. Synthetic teaching geometry—not a causal claim.*


![c577 teaching panel 03 (original).](../assets/figures/ml_fig_c577_03.png)
*Figure — Calibrated color heat c577. Synthetic teaching geometry—not a causal claim.*


![c578 teaching panel 03 (original).](../assets/figures/ml_fig_c578_03.png)
*Figure — Animation frame residual c578. Synthetic teaching geometry—not a causal claim.*


![c579 teaching panel 03 (original).](../assets/figures/ml_fig_c579_03.png)
*Figure — Facet grid scatter c579. Synthetic teaching geometry—not a causal claim.*


![c580 teaching panel 03 (original).](../assets/figures/ml_fig_c580_03.png)
*Figure — Raincloud density path c580. Synthetic teaching geometry—not a causal claim.*


![c581 teaching panel 03 (original).](../assets/figures/ml_fig_c581_03.png)
*Figure — QQ residual path c581. Synthetic teaching geometry—not a causal claim.*


![c582 teaching panel 03 (original).](../assets/figures/ml_fig_c582_03.png)
*Figure — ECDF comparison path c582. Synthetic teaching geometry—not a causal claim.*


![c583 teaching panel 03 (original).](../assets/figures/ml_fig_c583_03.png)
*Figure — Color map accessibility heat c583. Synthetic teaching geometry—not a causal claim.*


![c584 teaching panel 03 (original).](../assets/figures/ml_fig_c584_03.png)
*Figure — Small multiples layout path c584. Synthetic teaching geometry—not a causal claim.*


![c585 teaching panel 03 (original).](../assets/figures/ml_fig_c585_03.png)
*Figure — Linked brushing scatter c585. Synthetic teaching geometry—not a causal claim.*


![c586 teaching panel 03 (original).](../assets/figures/ml_fig_c586_03.png)
*Figure — Parallel coords bars c586. Synthetic teaching geometry—not a causal claim.*


![c587 teaching panel 03 (original).](../assets/figures/ml_fig_c587_03.png)
*Figure — Radar chart misuse bars c587. Synthetic teaching geometry—not a causal claim.*


![c588 teaching panel 03 (original).](../assets/figures/ml_fig_c588_03.png)
*Figure — Uncertainty band path c588. Synthetic teaching geometry—not a causal claim.*


![c589 teaching panel 03 (original).](../assets/figures/ml_fig_c589_03.png)
*Figure — Log-scale distortion path c589. Synthetic teaching geometry—not a causal claim.*


![c590 teaching panel 03 (original).](../assets/figures/ml_fig_c590_03.png)
*Figure — Glyph encoding scatter c590. Synthetic teaching geometry—not a causal claim.*


![c591 teaching panel 03 (original).](../assets/figures/ml_fig_c591_03.png)
*Figure — Treemap hierarchy heat c591. Synthetic teaching geometry—not a causal claim.*


![c592 teaching panel 03 (original).](../assets/figures/ml_fig_c592_03.png)
*Figure — Sankey flow path c592. Synthetic teaching geometry—not a causal claim.*


![c593 teaching panel 03 (original).](../assets/figures/ml_fig_c593_03.png)
*Figure — Calibrated color heat c593. Synthetic teaching geometry—not a causal claim.*


![c594 teaching panel 03 (original).](../assets/figures/ml_fig_c594_03.png)
*Figure — Animation frame residual c594. Synthetic teaching geometry—not a causal claim.*


![c595 teaching panel 03 (original).](../assets/figures/ml_fig_c595_03.png)
*Figure — Facet grid scatter c595. Synthetic teaching geometry—not a causal claim.*


![c596 teaching panel 03 (original).](../assets/figures/ml_fig_c596_03.png)
*Figure — Raincloud density path c596. Synthetic teaching geometry—not a causal claim.*


![c597 teaching panel 03 (original).](../assets/figures/ml_fig_c597_03.png)
*Figure — QQ residual path c597. Synthetic teaching geometry—not a causal claim.*


![c598 teaching panel 03 (original).](../assets/figures/ml_fig_c598_03.png)
*Figure — ECDF comparison path c598. Synthetic teaching geometry—not a causal claim.*


![c599 teaching panel 03 (original).](../assets/figures/ml_fig_c599_03.png)
*Figure — Color map accessibility heat c599. Synthetic teaching geometry—not a causal claim.*


![c600 teaching panel 03 (original).](../assets/figures/ml_fig_c600_03.png)
*Figure — Small multiples layout path c600. Synthetic teaching geometry—not a causal claim.*


![c601 teaching panel 03 (original).](../assets/figures/ml_fig_c601_03.png)
*Figure — Linked brushing scatter c601. Synthetic teaching geometry—not a causal claim.*


![c602 teaching panel 03 (original).](../assets/figures/ml_fig_c602_03.png)
*Figure — Parallel coords bars c602. Synthetic teaching geometry—not a causal claim.*


![c603 teaching panel 03 (original).](../assets/figures/ml_fig_c603_03.png)
*Figure — Radar chart misuse bars c603. Synthetic teaching geometry—not a causal claim.*


![c604 teaching panel 03 (original).](../assets/figures/ml_fig_c604_03.png)
*Figure — Uncertainty band path c604. Synthetic teaching geometry—not a causal claim.*


![c605 teaching panel 03 (original).](../assets/figures/ml_fig_c605_03.png)
*Figure — Log-scale distortion path c605. Synthetic teaching geometry—not a causal claim.*


![c606 teaching panel 03 (original).](../assets/figures/ml_fig_c606_03.png)
*Figure — Glyph encoding scatter c606. Synthetic teaching geometry—not a causal claim.*


![c607 teaching panel 03 (original).](../assets/figures/ml_fig_c607_03.png)
*Figure — Treemap hierarchy heat c607. Synthetic teaching geometry—not a causal claim.*

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
