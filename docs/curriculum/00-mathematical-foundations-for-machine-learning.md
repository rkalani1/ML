# Chapter 0. Mathematical Foundations for Machine Learning


![00 Vector Matrix](../assets/figures/00_vector_matrix.png)


## Opening

A fellow freezes at a gradient step in a methods appendix for an ICH expansion model. The clinical question is still bedside-valid, but the math barrier is blocking appraisal. Chapter 0 rebuilds the minimum calculus and linear algebra needed to read ML without surrendering clinical judgment.


![Gradient descent on a synthetic loss surface (original teaching graphic).](../assets/figures/ml_fig_gradient_descent.png)

*Gradient descent on a synthetic loss surface (original teaching graphic).*
Machine learning looks intimidating from the outside mostly because of its notation. Strip away the symbols and the field rests on a compact stack of mathematics that a motivated reader can rebuild in a few focused sittings: the language of sets and functions; algebra and logarithms; the calculus of change (derivatives) and accumulation (integrals); the algebra of vectors and matrices; the logic of probability; and the discipline of optimization. This chapter teaches that stack from an elementary starting point and connects each piece to the exact place later in the book where it is used.

The chapter assumes only that you once learned high-school algebra and are willing to work examples by hand — nothing more. It does not assume you remember any of it. Concepts are introduced in the same order they build on one another: first the notation for reading equations aloud (0.1), then numbers, algebra, and logarithms (0.2) and the catalog of functions machine learning actually uses (0.3); sums and counting (0.4) and the trigonometry behind Fourier features and attention (0.5); single-variable calculus (0.6–0.8) and its multivariable extension — gradients, Jacobians, Hessians — that powers every optimizer and neural network (0.9); the linear algebra of vectors, matrices, and their eigen- and singular-value decompositions (0.10–0.12); the foundations of probability (0.13); optimization and gradient descent (0.14); the discrete mathematics and complexity behind algorithms (0.15); and the numerical realities of computing with finite-precision arithmetic (0.16). Section 0.17 collects a notation glossary and a table mapping each topic to the chapters that depend on it.

Two habits make this chapter pay off. First, work every numeric example with pen and paper; the intermediate steps are printed precisely so you can check yourself. Second, treat the chapter as a reference, not a gate — each concept carries a “→ Used in Chapter N” pointer, so when a later chapter invokes a gradient, an eigenvector, or Bayes’ theorem, you can return here for a full, self-contained treatment. You do not need to master all of it before Chapter 1; you need to know it is here.

## Learning Objectives

After working through this chapter, a reader will be able to:

Read, pronounce, and write standard mathematical notation, including set, function, summation, and logic symbols, and the Greek letters used throughout machine learning.

Manipulate algebraic expressions confidently; solve linear and quadratic equations and inequalities; and use the laws of exponents and logarithms fluently.

Identify and reason about the core functions of machine learning — linear, polynomial, exponential, logarithmic, sigmoid/softmax, and ReLU — from their formulas and graphs.

Evaluate sums and products, count with permutations and combinations, and expand binomials.

Use radians, the unit circle, and sinusoids, and explain why trigonometry underlies Fourier features, positional encodings, and cosine similarity.

Compute limits and derivatives, apply the chain rule, and locate maxima and minima of one-variable functions.

Interpret and compute definite integrals as areas and as probabilities and expectations.

Compute partial derivatives, gradients, Jacobians, and Hessians; apply the multivariable chain rule; and connect it explicitly to backpropagation and Taylor approximation.

Perform vector and matrix operations, including dot products, norms, matrix multiplication, inverses, determinants, and solving linear systems.

Find eigenvalues, eigenvectors, and singular values; test positive-definiteness; and explain how these underlie PCA and low-rank approximation.

State the axioms of probability, apply conditional probability and Bayes’ theorem, and define random variables, expectation, and variance.

Formulate an objective function, recognize convexity, and execute gradient descent by hand.

Analyze algorithmic cost with Big-O notation and anticipate the numerical pitfalls of finite-precision computation.

## How to Use This Chapter

Read it linearly the first time; the ordering is deliberate, and later sections lean on earlier ones (multivariable calculus in 0.9 assumes single-variable calculus from 0.6–0.8 and vectors from 0.10). After that, use it as a lookup: the symbol glossary and the topic-to-chapter map in 0.17 tell you exactly which foundation a given later chapter draws on. Each major section is self-contained and ends with practice problems whose answers are worked, so you can verify your understanding before moving on.

## 0.1 Reading Mathematics: Notation, Sets, and Logic

Mathematics is a language before it is a set of techniques. Much of the intimidation people feel comes not from hard ideas but from unfamiliar shorthand — the same way a lab report looks impenetrable until you know what “CBC” and “×10⁹/L” stand for. This section teaches you to read; once you can read fluently, most later formulas become almost conversational.

### Sets: collections of things

A set is simply a collection of distinct objects, called its elements. We write a set by listing its elements inside curly braces: A = {2, 4, 6, 8}. Order does not matter and repeats are ignored, so {2, 4} and {4, 2, 2} denote the same set.

Two symbols do most of the work:

∈ means “is an element of.” We read 4 ∈ A aloud as “4 is in A.” The negation is ∉: 5 ∉ A, “5 is not in A.”

⊆ means “is a subset of”: every element of the first set is also in the second. {2, 4} ⊆ A. Read it “is a subset of” or “is contained in.”

The empty set, written ∅ or {}, is the set with no elements at all — the mathematical equivalent of an empty test tube. It is a subset of every set.

Three operations combine sets. Let A = {1, 2, 3} and B = {2, 3, 4}.

Union ∪ — everything in either set: A ∪ B = {1, 2, 3, 4} (“A union B”).

Intersection ∩ — everything in both sets: A ∩ B = {2, 3} (“A intersect B”).

Complement — everything (within some agreed universe) not in the set. If our universe is {1, 2, 3, 4, 5}, then the complement of A, written Aᶜ, is {4, 5}. The difference A B (“A minus B”) keeps what is in A but not B: A B = {1}.

The number of elements in a finite set is its cardinality, written |A|. Here |A| = 3 and |A ∪ B| = 4.

### Set-builder notation

Listing elements fails when a set is infinite. Instead we state a rule:

{x ∈ ℝ : x > 0}

Read this as “the set of all real numbers x such that x is greater than 0” — i.e., the positive numbers. The colon (sometimes a vertical bar |) means “such that.” The part before it says what kind of object x is; the part after gives the condition it must satisfy. Another example: {n ∈ ℕ : n is even} is {2, 4, 6, …}.

### The standard number sets

Five collections appear so often they get dedicated blackboard-bold letters:

ℕ — the natural numbers: 0, 1, 2, 3, … (counting numbers).

ℤ — the integers: …, −2, −1, 0, 1, 2, … (naturals plus negatives; Z from German Zahlen).

ℚ — the rationals: every fraction p/q of integers with q ≠ 0, such as −3/4 or 5.

ℝ — the real numbers: every point on the continuous number line, including irrationals like √2 and π that no fraction captures.

ℝⁿ — ordered lists of n real numbers, such as (1.2, −0.5, 3.0) in ℝ³. A patient described by age, blood pressure, and weight is a point in ℝ³. Almost all data in this book lives in ℝⁿ.

These nest neatly: ℕ ⊆ ℤ ⊆ ℚ ⊆ ℝ.

### Intervals

An interval is a connected stretch of the real line. Square brackets include the endpoint; round brackets exclude it.

[a, b] — all x with a ≤ x ≤ b (closed, both ends included).

(a, b) — all x with a < x < b (open, both ends excluded).

[0, 1] — a workhorse: every probability lives here.

(0, ∞) — the positive reals; ∞ is always excluded because it is not a number.

→ Used in Chapter 3: events in probability are sets, and set operations become “and”/“or” for uncertain outcomes.

### The Greek alphabet you actually need

ML borrows Greek letters as a compact vocabulary. The letter is just a name; what matters is the convention attached to it. You will meet these constantly:

| Symbol | Name | Usually denotes |
| --- | --- | --- |
| α | alpha | learning rate; significance level |
| β | beta | regression coefficients |
| γ | gamma | discount factor (RL); a rate |
| δ | delta | a small change; an error term |
| ε | epsilon | a tiny positive quantity; noise |
| η | eta | learning rate (alternative) |
| θ | theta | a model’s parameters, generically |
| λ | lambda | regularization strength; an eigenvalue |
| μ | mu | a mean (average) |
| π | pi | the constant 3.14159…; also a policy (RL) |
| σ | sigma | standard deviation; the sigmoid function |
| φ | phi | a feature transformation |
| Σ | capital sigma | “sum of…”; also a covariance matrix |
| Π | capital pi | “product of…” |
| Δ | capital delta | a change or difference |
| ∇ | nabla | the gradient (Section 0.9) |

Do not memorize this table; refer back to it. Notice one trap: σ can mean a number (standard deviation) or a function (the sigmoid), and Σ can mean an instruction (add these up) or a matrix. Context always decides, and we will flag which is meant.

### Subscripts, superscripts, and indexing

When we have many related quantities we tag them with subscripts: x₁, x₂, x₃ are three different numbers, read “x-sub-one,” and so on. A generic one is xᵢ, “x-sub-i,” where the index i is a placeholder standing for “whichever one we mean.”

A grid of numbers needs two indices: xᵢⱼ (“x-sub-i-j”) is the entry in row i, column j. Superscripts usually mean powers (x², “x squared”), but a raised T is special: xᵀ means the transpose of x (turning a column into a row), not a power. We write vectors in bold lowercase (x) and matrices in bold uppercase (A); plain letters are ordinary numbers, called scalars.

→ Used in Chapter 7 and Chapter 10: every dataset is indexed this way, and xᵀ appears in nearly every matrix formula.

### Functions as machines

A function is a rule that takes an input and returns exactly one output. The notation

f : A → B

is read “f maps A to B” and says: f accepts inputs from set A (the domain) and produces outputs in set B (the codomain). The rule itself is written separately, e.g. f(x) = x². Keep two ideas distinct: f is the whole machine (the mapping); f(3) = 9 is a single output value. Confusing the machine with one of its outputs is the single most common reading error for beginners. Section 0.3 is devoted entirely to functions.

### Logic and quantifiers

Formal statements are glued together with a few connectives:

∧ “and,” ∨ “or” (inclusive: one or both), ¬ “not.”

⇒ “implies”: P ⇒ Q means “if P is true, then Q is true.”

⇔ “if and only if,” abbreviated iff: each side implies the other; they are logically equivalent.

Two quantifiers say how many:

∀ “for all” — ∀x ∈ ℝ, x² ≥ 0 reads “for all real x, x squared is at least 0” (true).

∃ “there exists” — ∃x ∈ ℝ such that x² = 2 reads “there is some real x whose square is 2” (true: x = √2).

Necessary vs. sufficient trips up even careful readers. If P ⇒ Q, then P is sufficient for Q (P alone guarantees Q) and Q is necessary for P (P cannot hold without Q). Example: “rain ⇒ clouds.” Rain is sufficient for clouds; clouds are necessary for rain — but clouds do not guarantee rain. A statement and its converse are different claims.

→ Used in Chapter 3 and Chapter 9: “iff” defines decision rules, and ∀/∃ make precise what a model must satisfy.

### Reading an equation aloud

You will understand formulas faster if you voice them. Take a preview from Chapter 8:

ŷ = wᵀx + b

Read: “y-hat equals w-transpose x, plus b.” The hat on ŷ marks a predicted value; w and x are vectors; wᵀx is a single number combining them; b is a scalar shift. Or the summation

∑ᵢ wᵢ xᵢ

reads “the sum over i of w-sub-i times x-sub-i” — multiply each pair and add the results. Saying it out loud converts a wall of symbols into a sentence.

### Proof by example, and disproof by counterexample

One worked case can illustrate a claim but never proves a universal (“∀”) statement — checking that 2 + 2 = 4 does not prove all sums. But a single counterexample disproves a universal outright: the claim “all prime numbers are odd” dies instantly at the prime 2. Throughout this book we reason with worked examples for intuition, while remembering that intuition and proof are not the same thing.

## 0.2 Numbers, Algebra, Exponents, and Logarithms

![0.1: Exponential and logarithm are inverse functions (reflected across y = x); a log scale turns exponential growth into a st](../assets/figures/ml_concept_0.1_b95475b9.png)

*Figure 0.1 — original teaching graphic.*

### The real line and basic arithmetic

Picture every real number as a point on an infinite horizontal line, zero in the middle, negatives left, positives right. Distance from zero is magnitude; side is sign. This mental image underlies almost everything later: data points, errors, and parameters are all positions on such lines.

A fraction p/q means p parts out of q. A ratio compares two quantities (a 3 : 1 ratio of controls to cases). A percentage is a fraction with denominator fixed at 100: 15% = 15/100 = 0.15, so 15% of 200 is 0.15 × 200 = 30. Epidemiology lives on such comparisons — if disease risk is 0.30 among the exposed and 0.10 among the unexposed, the risk ratio is 0.30 / 0.10 = 3.

### Order of operations

When several operations appear together, evaluate them in a fixed order, often abbreviated PEMDAS: Parentheses, Exponents, Multiplication and Division (left to right), Addition and Subtraction (left to right). For example:

2 + 3 × 4² = 2 + 3 × 16 = 2 + 48 = 50.

The exponent fires first, then the multiplication, then the addition. Ignoring this order is the arithmetic equivalent of a dosing error.

### Rearranging and solving linear equations

An equation asserts two expressions are equal; solving means finding the unknown that makes it true. The one rule: whatever you do to one side, do to the other, keeping the balance. A linear equation has the unknown only to the first power. Solve

3(x − 2) + 4 = 2x + 5.

Expand the left: 3x − 6 + 4 = 3x − 2, so 3x − 2 = 2x + 5. Subtract 2x: x − 2 = 5. Add 2: x = 7. Check by substituting back: left = 3(7 − 2) + 4 = 15 + 4 = 19; right = 2(7) + 5 = 19. ✓

→ Used in Chapter 8: fitting a straight line to data is solving equations like these at scale.

### Quadratic equations and the quadratic formula

A quadratic has the unknown squared: the general form is ax² + bx + c = 0 with a ≠ 0. Some factor by inspection, but one formula always works:

x = ( −b ± √(b² − 4ac) ) / (2a).

The ± means there are generally two solutions, one with +, one with −. The inside piece b² − 4ac is the discriminant: positive gives two real solutions, zero gives one, negative gives none on the real line. Solve 2x² + 3x − 2 = 0, so a = 2, b = 3, c = −2:

Discriminant: b² − 4ac = 3² − 4(2)(−2) = 9 + 16 = 25.

Square root: √25 = 5.

Solutions: x = (−3 ± 5) / (2·2) = (−3 ± 5)/4, giving x = 2/4 = 0.5 or x = −8/4 = −2.

Check x = 0.5: 2(0.25) + 3(0.5) − 2 = 0.5 + 1.5 − 2 = 0. ✓ Check x = −2: 2(4) + 3(−2) − 2 = 8 − 6 − 2 = 0. ✓

→ Used in Chapter 8: quadratic “bowl” shapes are the simplest error surfaces an optimizer descends.

### Inequalities and absolute value

An inequality uses ≤, <, ≥, or > instead of =. Solving works like equations with one twist: multiplying or dividing both sides by a negative number flips the direction. From −2x < 6, divide by −2 and flip: x > −3.

The absolute value |x| strips the sign, giving distance from zero: |−3| = 3, |3| = 3. So |a − b| is the distance between a and b, and the condition |x − μ| < 2 means “x lies within 2 of μ,” i.e. μ − 2 < x < μ + 2.

→ Used in Chapter 4: distances between data points are built from absolute values and their cousins.

### Exponents

An exponent counts repeated multiplication: aⁿ means a multiplied by itself n times, and a is the base. A handful of laws follow directly:

aᵐ · aⁿ = aᵐ⁺ⁿ — add exponents when multiplying: 2³ · 2⁴ = 2⁷ = 128 (check: 8 × 16 = 128). ✓

(aᵐ)ⁿ = aᵐⁿ — multiply exponents for a power of a power: (2³)² = 2⁶ = 64. ✓

a⁻ⁿ = 1 / aⁿ — a negative exponent means reciprocal: 2⁻³ = 1/8 = 0.125.

a⁰ = 1 for any a ≠ 0 — the “empty product.”

Fractional exponents are roots: a^(1/2) = √a and a^(1/n) is the n-th root. So 9^(1/2) = 3, 8^(1/3) = 2, and 8^(2/3) = (8^(1/3))² = 2² = 4.

### Scientific notation

Very large or small numbers are written as a number between 1 and 10 times a power of ten. Thus 6,700,000 = 6.7 × 10⁶ and 0.00042 = 4.2 × 10⁻⁴. To multiply, multiply the fronts and add the exponents: (3 × 10⁵)(2 × 10⁻²) = 6 × 10³ = 6000.

### The number e

Alongside π there is a second irrational constant, e ≈ 2.71828, that governs growth. It arises as the limit of (1 + 1/n)ⁿ as n grows without bound — the value of continuously compounded growth. It is the natural base for the exponential function eˣ, which we meet again in Section 0.3.

### Logarithms: the inverse of exponentiation

A logarithm answers the question hidden inside an exponent: to what power must I raise the base to get this number? By definition,

log_b(y) = x ⇔ bˣ = y.

So log₁₀(1000) = 3 because 10³ = 1000, and log₂(8) = 3 because 2³ = 8. The natural logarithm, ln, uses base e: ln(e²) = 2. Because logs and exponents undo each other, they are inverse functions — a relationship we will lean on constantly.

Three laws convert multiplication into addition — the reason logs are so beloved:

log(xy) = log x + log y. Example: log₁₀(200) = log₁₀(2 · 100) = log₁₀2 + log₁₀100 ≈ 0.301 + 2 = 2.301.

log(x / y) = log x − log y. Example: log₁₀(5) = log₁₀(10/2) = 1 − 0.301 = 0.699.

log(xⁿ) = n · log x. Example: log₁₀(2⁵) = 5 × 0.301 = 1.505.

To move between bases, use change of base: log_b(y) = ln(y) / ln(b). For instance log₂(10) = ln 10 / ln 2 ≈ 2.3026 / 0.6931 ≈ 3.322, and indeed 2^3.322 ≈ 10. ✓

### Exponential growth and decay

Exponentials model quantities that change by a proportion of their current size. Growth: N(t) = N₀ · e^(rt). An epidemic starting at N₀ = 100 cases with rate r = 0.2 per day reaches, at t = 10 days, N = 100 · e^(0.2·10) = 100 · e² ≈ 100 · 7.389 = 739 cases. Decay: a drug at concentration C(t) = C₀ · e^(−kt) with k = 0.1 per hour has half-life t½ = ln(2)/k ≈ 0.693 / 0.1 = 6.93 hours — the time for e^(−kt) to fall to exactly ½.

### Why logarithms pervade ML

Logs are not decoration; they are structural. First, models multiply many probabilities together, and tiny numbers like 0.001 × 0.002 × … underflow a computer to zero; taking logs turns that fragile product into a stable sum (the log-likelihood), the quantity most models actually maximize. Second, the log-odds or logit, ln(p / (1 − p)), stretches a probability trapped in [0, 1] onto the whole real line, which is exactly what a linear model needs to predict. Third, quantities spanning many orders of magnitude (gene expression, word counts) are tamed by plotting on a log scale, where equal steps mean equal ratios.

→ Used in Chapter 3 (log-likelihood), Chapter 9 (log-odds and logistic regression), and Chapter 14 (numerical stability).

## 0.3 Functions and Their Graphs (the ML “function zoo”)

![Core functions of machine learning (original teaching catalog).](../assets/figures/ml_fig_core_functions.png)

*Figure 0.2. The core functions of machine learning: linear, quadratic, exponential, logarithmic, sigmoid, and ReLU.*


![Gradient magnitude vs depth: plain stack vs residual floor (teaching; original).](../assets/figures/ml_fig_gradient_flow_depth.png)

*Figure — Optimization geometry. Plain deep stacks can drive gradients toward zero; residual-style paths keep a teaching floor. Curves are schematic—not a map of clinical mechanisms. **Architecture ≠ causation**.*


![Condition number vs feature correlation (synthetic; original).](../assets/figures/ml_fig_condition_number.png)

*Figure — Collinearity and conditioning. As ρ→1, cond(Σ) explodes and OLS becomes unstable. Numerical diagnostics are not causal graphs—they flag estimation fragility.*


![SVD spectrum on a synthetic matrix (original).](../assets/figures/ml_fig_svd_spectrum.png)

*Figure — Singular values on a log scale. Sharp drops hint at numerical rank. Spectral structure is linear algebra—not automatic clinical causation.*


![Rank-1 outer-product structure heatmap (synthetic; original).](../assets/figures/ml_fig_outer_product.png)

*Figure — Low-rank outer-product geometry. Useful for intuition about factor models and SVD truncations—not a claim about clinical causal factors.*


![Norm growth cartoon for matrix scales (teaching; original).](../assets/figures/ml_fig_norm_growth.png)

*Figure — Linear-algebra scale intuition—not clinical causation. Pred ≠ cause without design.*


![Gram-Schmidt orthogonalization sketch (original).](../assets/figures/ml_fig_gram_schmidt.png)

*Figure — Orthogonal bases aid numerics. Gram-Schmidt orthogonalization sketch Pred != cause without design.*


![jacobian teaching panel (original).](../assets/figures/ml_fig_jacobian_sketch.png)

*Figure — Teaching panel for jacobian. Pred != cause without design.*


![Cycle-34 densify scientific panel 1 (original).](../assets/figures/ml_fig_c34_00.png)

*Figure — Continuous densify panel 1. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 1 (original).](../assets/figures/ml_fig_c35_00.png)

*Figure — Continuous densify panel 1. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 1 (original).](../assets/figures/ml_fig_c36_00.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 1 (original).](../assets/figures/ml_fig_c37_00.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 1 (original).](../assets/figures/ml_fig_c38_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 1 (original).](../assets/figures/ml_fig_c39_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 1 (original).](../assets/figures/ml_fig_c40_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 1 (original).](../assets/figures/ml_fig_c41_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 1 (original).](../assets/figures/ml_fig_c42_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 1 (original).](../assets/figures/ml_fig_c43_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 1 (original).](../assets/figures/ml_fig_c44_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 1 (original).](../assets/figures/ml_fig_c45_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 1 (original).](../assets/figures/ml_fig_c46_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 1 (original).](../assets/figures/ml_fig_c47_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 1 (original).](../assets/figures/ml_fig_c48_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 1 (original).](../assets/figures/ml_fig_c49_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 1 (original).](../assets/figures/ml_fig_c50_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 1 (original).](../assets/figures/ml_fig_c51_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 1 (original).](../assets/figures/ml_fig_c52_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 1 (original).](../assets/figures/ml_fig_c53_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 1 (original).](../assets/figures/ml_fig_c54_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 1 (original).](../assets/figures/ml_fig_c55_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 1 (original).](../assets/figures/ml_fig_c56_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 1 (original).](../assets/figures/ml_fig_c57_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 1 (original).](../assets/figures/ml_fig_c58_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 1 (original).](../assets/figures/ml_fig_c59_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 1 (original).](../assets/figures/ml_fig_c60_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 1 (original).](../assets/figures/ml_fig_c61_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 1 (original).](../assets/figures/ml_fig_c62_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 1 (original).](../assets/figures/ml_fig_c63_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 1 (original).](../assets/figures/ml_fig_c64_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 1 (original).](../assets/figures/ml_fig_c65_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 1 (original).](../assets/figures/ml_fig_c66_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 1 (original).](../assets/figures/ml_fig_c67_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 1 (original).](../assets/figures/ml_fig_c68_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 1 (original).](../assets/figures/ml_fig_c69_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 1 (original).](../assets/figures/ml_fig_c70_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 1 (original).](../assets/figures/ml_fig_c71_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 1 (original).](../assets/figures/ml_fig_c72_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 1 (original).](../assets/figures/ml_fig_c73_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 1 (original).](../assets/figures/ml_fig_c74_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 1 (original).](../assets/figures/ml_fig_c75_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 1 (original).](../assets/figures/ml_fig_c76_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 1 (original).](../assets/figures/ml_fig_c77_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 1 (original).](../assets/figures/ml_fig_c78_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 1 (original).](../assets/figures/ml_fig_c79_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 1 (original).](../assets/figures/ml_fig_c80_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 1 (original).](../assets/figures/ml_fig_c81_00.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

### What a function is, and how to read its graph

Recall from Section 0.1 that a function maps each input to exactly one output. The domain is the set of allowed inputs; the range is the set of outputs actually produced. We picture a function by its graph: plot the input x horizontally and the output y = f(x) vertically, and mark every point (x, f(x)). Reading a graph is then physical: left-to-right is increasing input, height is output, and the “exactly one output” rule shows up as the vertical line test — any vertical line crosses the curve at most once.

ML uses a surprisingly small cast of functions over and over. Meet the zoo.

### Linear functions

f(x) = mx + b.

This is a straight line. m is the slope — the rise in y per unit step in x — and b is the y-intercept, the height where the line crosses x = 0. For f(x) = 2x + 1: at x = 0, y = 1; at x = 1, y = 3; each step right of 1 raises y by 2. Positive slope rises, negative slope falls, zero slope is flat. Linear functions are the backbone of regression and of every neuron’s pre-activation.

→ Used in Chapter 8: linear regression fits exactly this shape.

### Quadratics and polynomials

A quadratic f(x) = ax² + bx + c graphs as a parabola, a symmetric U-shape (opening up if a > 0, down if a < 0) with a single lowest or highest point called the vertex. Its bowl shape makes it the prototype of a well-behaved error surface. More generally a polynomial adds higher powers, a₀ + a₁x + a₂x² + … + aₙxⁿ; the highest power is the degree, and higher degree allows more wiggles.

→ Used in Chapter 8: the squared-error loss is a parabola in the parameters.

### Power, rational, and root functions

A power function is f(x) = xᵖ for a fixed exponent p: p = 2 gives the parabola, p = ½ gives the (sideways-opening) square root, p = −1 gives the reciprocal. A rational function is one polynomial divided by another, such as f(x) = 1/x, which shoots toward infinity near x = 0 and flattens toward 0 far out — behavior we call asymptotic.

### Exponential and logarithmic functions

The exponential f(x) = eˣ starts near 0 for very negative x, passes through (0, 1), and then climbs explosively — the mathematical signature of unchecked growth. Its inverse, the logarithm f(x) = ln(x), does the reverse: defined only for x > 0, it climbs steeply then flattens, compressing a huge range into a manageable one. Reflecting either curve across the diagonal line y = x produces the other, the visual fingerprint of inverse functions.

→ Used in Chapter 11: exponentials shape many generative and probability models.

### The logistic (sigmoid) function

Perhaps the single most important curve in this book:

σ(z) = 1 / (1 + e^(−z)).

It takes any real number and squashes it into the open interval (0, 1), making it perfect for turning an unbounded score into a probability. Its graph is a smooth S-curve: far left it hugs 0, far right it hugs 1, and it passes through the midpoint (0, 0.5). Compute a few values (using e^(−2) ≈ 0.1353, e² ≈ 7.389):

σ(0) = 1 / (1 + 1) = 0.5.

σ(2) = 1 / (1 + 0.1353) = 1 / 1.1353 ≈ 0.881.

σ(−2) = 1 / (1 + 7.389) = 1 / 8.389 ≈ 0.119.

Notice the elegant symmetry σ(−2) = 1 − σ(2), since 0.119 + 0.881 = 1. And its inverse is exactly the logit from Section 0.2: if p = σ(z), then z = ln(p / (1 − p)).

→ Used in Chapter 9: logistic regression bends a straight line into a probability with σ.

### Softmax: a preview

When there are several classes rather than two, the softmax generalizes the sigmoid: it takes a list of scores and returns positive numbers that sum to 1 — a probability distribution. Given raw scores (2, 1, 0), exponentiate each (e² ≈ 7.389, e¹ ≈ 2.718, e⁰ = 1), sum them (7.389 + 2.718 + 1 = 11.107), and divide:

(7.389, 2.718, 1) / 11.107 ≈ (0.665, 0.245, 0.090), which sums to 1.000. ✓

→ Used in Chapter 9 and Chapter 12: softmax produces class probabilities and attention weights.

### ReLU and piecewise functions

A piecewise function uses different rules on different stretches of input. The star example is the rectified linear unit:

ReLU(z) = max(0, z),

which returns z when z is positive and 0 otherwise — a flat floor that suddenly kinks upward at the origin. So ReLU(3) = 3 and ReLU(−3) = 0. Its very simplicity (and cheapness to compute) made it the default nonlinearity in modern neural networks.

→ Used in Chapter 10: ReLU is the workhorse activation between network layers.

### Properties worth naming

A few adjectives describe a function’s shape at a glance:

Monotonic — always heading one way: increasing (like eˣ) or decreasing (like e^(−x)), never reversing.

Even / odd — an even function is mirror-symmetric across the vertical axis, f(−x) = f(x), like x²; an odd function has rotational symmetry through the origin, f(−x) = −f(x), like x³.

Bounded — trapped between limits, as σ is confined to (0, 1).

Convex / concave — convex means bowl-shaped (holds water; a straight segment between any two points on the curve lies above it), like x²; concave is the upside-down cap, like ln(x). Convexity matters enormously because a convex bowl has a single lowest point, so an optimizer cannot get stuck in a false one.

→ Used in Chapter 8: convex loss functions guarantee optimization finds the true minimum.

### Composition and inverses

Composition feeds one function’s output into another. Written (f ∘ g)(x) = f(g(x)), read “f of g of x,” it means do g first, then f. With f(x) = 2x + 1 and g(x) = x², we get (f ∘ g)(x) = 2x² + 1, so (f ∘ g)(3) = 2·9 + 1 = 19, whereas (g ∘ f)(3) = (2·3 + 1)² = 7² = 49. Order matters. A deep neural network is nothing but a long composition of simple functions, layer after layer.

An inverse f⁻¹ undoes f: if f sends 3 to 7, then f⁻¹ sends 7 back to 3. To find it, swap roles and solve. For f(x) = 2x + 1, set y = 2x + 1 and solve for x: x = (y − 1)/2, so f⁻¹(x) = (x − 1)/2. Check: f(3) = 7 and f⁻¹(7) = (7 − 1)/2 = 3. ✓ (The logit/sigmoid pair is exactly this idea.)

→ Used in Chapter 10: the chain rule for backpropagation is composition made differentiable.

### Graph transformations

Small edits to a formula move its graph in predictable ways. Starting from y = f(x):

f(x) + c shifts the whole curve up by c (down if c is negative).

f(x − c) shifts it right by c — note the minus sign moves it the positive direction.

a · f(x) stretches it vertically by factor a (and flips it if a is negative).

f(−x) reflects it left-right across the vertical axis.

So (x − 2)² + 1 is the basic parabola x² slid 2 to the right and 1 upward, placing its vertex at (2, 1). Recognizing these moves lets you read an unfamiliar formula as a familiar shape in disguise.

→ Used in Chapter 6: rescaling and shifting features is exactly these transformations applied to data.

### Practice — 0.1–0.3

(Sets & logic) Let A = {1, 2, 3, 4} and B = {3, 4, 5}. Find A ∪ B, A ∩ B, and A B. Then decide whether the statement “∀x ∈ A, x < 5” is true.

(Linear equation) Solve 5(x − 3) = 2x + 6 for x, and verify your answer by substitution.

(Quadratic formula) Solve x² − 6x + 8 = 0 using the quadratic formula. State the discriminant, then both roots.

(Exponents & logs) (a) Simplify (2³ · 2⁵) / 2⁴ to a single power of 2, then a number. (b) Given log₁₀2 ≈ 0.301, compute log₁₀40. Hint: 40 = 4 × 10.

(Sigmoid) Using e⁻¹ ≈ 0.368 and e ≈ 2.718, compute σ(1) and σ(−1) for σ(z) = 1/(1 + e^(−z)), and confirm that σ(−1) = 1 − σ(1).

(Analyzing a function) For f(x) = (x − 2)² + 1: give the vertex, its minimum value, and whether it is convex. Describe it as a transformation of x². Then, with g(x) = x + 2 and h(x) = 3x, find (h ∘ g)(x).

Answers. 1. A ∪ B = {1, 2, 3, 4, 5}; A ∩ B = {3, 4}; A B = {1, 2}; the statement is true (1, 2, 3, 4 are all < 5). 2. 5x − 15 = 2x + 6 ⇒ 3x = 21 ⇒ x = 7 (check: both sides = 20). 3. Discriminant = 36 − 32 = 4, √4 = 2, so x = (6 ± 2)/2 = 4 or 2. 4. (a) 2^(3+5−4) = 2⁴ = 16; (b) log₁₀40 = log₁₀4 + log₁₀10 = 2(0.301) + 1 = 1.602. 5. σ(1) = 1/(1 + 0.368) = 1/1.368 ≈ 0.731; σ(−1) = 1/(1 + 2.718) = 1/3.718 ≈ 0.269; and 0.731 + 0.269 = 1. ✓ 6. Vertex (2, 1), minimum value 1, convex (opens upward); it is x² shifted right 2 and up 1; (h ∘ g)(x) = 3(x + 2) = 3x + 6.

## 0.4 Sums, Products, Factorials, and Counting

Machine learning is, at bottom, a great deal of adding and multiplying — the same operation repeated over thousands of data points, features, or parameters. Rather than write “add up all of these” in words, mathematics has a compact shorthand. This section teaches that shorthand and then the closely related art of counting: how many ways can something happen? Counting is the seed of probability (Chapter 3) and of the data-compression codes you will meet in Chapter 14.

![0.3: The geometric series Σ (1/2)ⁿ: terms shrink while the partial sums converge to 2.](../assets/figures/ml_concept_0.3_860b37f0.png)

*Figure 0.3 — original teaching graphic.*

### Sigma notation: the summation sign Σ

Intuition. Imagine you have five patients with ages 61, 47, 73, 58, and 66, and you want their total. You could write 61 + 47 + 73 + 58 + 66, but if there were 10,000 patients that would be hopeless. The Greek capital sigma, Σ, means “add up a whole list according to a rule.”

Formal definition. The expression

∑ᵢ₌₁ⁿ aᵢ

is read “the sum, as i goes from 1 to n, of aᵢ.” Here:

i is the index — a counter that ticks upward one integer at a time.

1 (below Σ) is the lower bound, where the counter starts.

n (above Σ) is the upper bound, where it stops.

aᵢ is the summand — the recipe telling you what to add at each step.

You substitute i = 1, 2, 3, …, n into the summand and add the results.

Worked example. Let the summand be i² (each index squared), summed from 1 to 4:

∑ᵢ₌₁⁴ i² = 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30.

Another, where the summand mixes the index with constants:

∑ₖ₌₂⁵ (2k + 1) = (2·2+1) + (2·3+1) + (2·4+1) + (2·5+1) = 5 + 7 + 9 + 11 = 32.

Notice the index letter (i, k, j, …) is arbitrary; it is just a label that disappears once the sum is written out.

### Properties of sums

Three rules let you rearrange sums safely. Each is just ordinary arithmetic seen from a height.

1. Linearity (constants factor out, sums split). For any constant c,

∑ (c·aᵢ) = c·∑ aᵢ and ∑ (aᵢ + bᵢ) = ∑ aᵢ + ∑ bᵢ.

Check: ∑ᵢ₌₁⁴ (3i) = 3+6+9+12 = 30, and 3·∑ᵢ₌₁⁴ i = 3·(1+2+3+4) = 3·10 = 30. ✓

2. Splitting the range. A sum can be cut into consecutive pieces:

∑ᵢ₌₁¹⁰ aᵢ = ∑ᵢ₌₁⁴ aᵢ + ∑ᵢ₌₅¹⁰ aᵢ.

Check with aᵢ = i: the whole is 1+2+…+10 = 55; the pieces are 10 and (5+6+7+8+9+10) = 45; and 10 + 45 = 55. ✓

3. Shifting the index. You may relabel the counter as long as you shift the bounds to match. Letting j = i − 2:

∑ᵢ₌₃⁶ (i − 2) = 1 + 2 + 3 + 4 = 10 = ∑ⱼ₌₁⁴ j.

Index shifting looks pedantic now but is the workhorse move behind convolutions and Fourier sums in Chapter 7.

### Double sums (a brief look)

When data comes in a grid — say a table with rows i and columns j — you sum over both. A double sum means “for each i, run through all j, and add everything”:

∑ᵢ₌₁² ∑ⱼ₌₁³ (i·j).

Work the inner sum first for each fixed i. For i = 1: (1·1)+(1·2)+(1·3) = 6. For i = 2: (2·1)+(2·2)+(2·3) = 12. Total = 6 + 12 = 18. (When the summand separates as here, the double sum equals the product of the two single sums: (1+2)·(1+2+3) = 3·6 = 18. ✓)

### Product notation Π and the factorial

Intuition. Just as Σ repeats addition, the capital Greek pi, Π, repeats multiplication.

Definition. ∏ᵢ₌₁ⁿ aᵢ means a₁ · a₂ · … · aₙ. For example ∏ᵢ₌₁⁴ i = 1·2·3·4 = 24.

That last product is so common it has its own name and symbol: the factorial.

Definition (factorial). For a positive integer n,

n! = n · (n−1) · (n−2) · … · 2 · 1,

and by convention 0! = 1 (an empty product, like an empty sum being 0, defaults to the multiplicative “do nothing” value). The first few:

0! = 1, 1! = 1, 2! = 2, 3! = 6, 4! = 24, 5! = 120, 6! = 720, 7! = 5040.

Factorials count arrangements: 5! = 120 is the number of distinct orders in which 5 patients could line up. They grow ferociously fast, which is exactly why brute-force counting becomes impossible and clever formulas are needed.

### Arithmetic series

Intuition. An arithmetic sequence adds a fixed step d each term: 2, 5, 8, 11, 14 (here d = 3). Summing such a sequence has a beautiful shortcut, discovered (legend says) by a young Gauss.

Formal result. If a₁ is the first term and aₙ the last, the sum of n equally spaced terms is

Sₙ = n · (a₁ + aₙ) / 2.

The idea: pair the smallest with the largest, second-smallest with second-largest — each pair has the same total, and there are n/2 pairs.

Worked example. Sum 2 + 5 + 8 + 11 + 14 (n = 5, a₁ = 2, a₅ = 14):

S₅ = 5·(2 + 14)/2 = 5·16/2 = 5·8 = 40.

Direct check: 2+5+8+11+14 = 40. ✓ And Gauss’s classic: 1 + 2 + … + 100 = 100·(1+100)/2 = 100·101/2 = 5050.

### Geometric series

Intuition. A geometric sequence multiplies by a fixed ratio r each term: 3, 6, 12, 24, 48 (here r = 2). These appear whenever a quantity is repeatedly scaled — compound interest, radioactive decay, and the “discount factors” of reinforcement learning.

Finite geometric sum. For n terms starting at a with ratio r ≠ 1,

∑ₖ₌₀ⁿ⁻¹ a·rᵏ = a·(1 − rⁿ) / (1 − r).

Worked example. 3 + 6 + 12 + 24 + 48 (a = 3, r = 2, n = 5):

= 3·(1 − 2⁵)/(1 − 2) = 3·(1 − 32)/(−1) = 3·(−31)/(−1) = 3·31 = 93.

Direct check: 3+6+12+24+48 = 93. ✓

Infinite geometric sum. If the ratio is small enough that terms shrink toward zero — precisely when |r| < 1 — the infinite sum settles on a finite value (it “converges”):

∑ₖ₌₀^∞ a·rᵏ = a / (1 − r), valid for |r| < 1.

Worked example. 1 + ½ + ¼ + ⅛ + … = 1/(1 − ½) = 1/(½) = 2. Each step covers half the remaining gap to 2, so the total approaches — but never exceeds — 2. If |r| ≥ 1 the terms do not shrink and the sum runs away to infinity; convergence is the whole point.

→ Used in Chapter 7: geometric and related sums underlie the infinite series behind Fourier analysis. Discounted-reward sums in later material are geometric series in disguise.

### The counting rules

Before probability, you must count outcomes. Two rules cover almost everything.

Product rule (AND). If one choice can be made in m ways and, independently, a second in n ways, the two together can be made in m·n ways. Three shirts and four pairs of trousers give 3·4 = 12 outfits.

Sum rule (OR). If you must pick one item from mutually exclusive groups of sizes m and n, there are m + n choices. Three novels or five textbooks give 3 + 5 = 8 single-book choices.

“AND multiplies, OR adds” — memorize that and most counting follows.

### Permutations: order matters

Intuition. How many ways can you fill k ranked slots from n distinct items, where being first differs from being second?

Definition. The number of permutations of n things taken k at a time is

P(n, k) = n! / (n − k)! = n·(n−1)·…·(n − k + 1).

Worked example. From 5 drugs, how many ways to choose a 1st-line and a 2nd-line therapy (order matters)?

P(5, 2) = 5!/3! = 120/6 = 20 (equivalently 5·4 = 20).

### Combinations: order does not matter

Intuition. Often only the group matters, not its internal order — a committee, a subset of features, a poker hand. Then we divide out the k! reorderings that we do not want to distinguish.

Definition. The number of combinations of n things taken k at a time is

C(n, k) = n! / (k!·(n − k)!).

This is read “n choose k” and is also written (n over k). It is always a whole number.

Worked example. From 5 drugs, how many unordered pairs?

C(5, 2) = 5!/(2!·3!) = 120/(2·6) = 120/12 = 10.

A larger one (why formulas beat brute force). The number of 5-card poker hands from a 52-card deck:

C(52, 5) = (52·51·50·49·48)/(5!) = 311,875,200 / 120 = 2,598,960.

No one enumerates 2.6 million hands by hand — the formula does it in one line.

### Pascal’s triangle and the binomial theorem

The numbers C(n, k) form a triangle in which each entry is the sum of the two above it — Pascal’s rule:

C(n, k) = C(n−1, k−1) + C(n−1, k).

```
n=0: 1
n=1: 1 1
n=2: 1 2 1
n=3: 1 3 3 1
n=4: 1 4 6 4 1
n=5: 1 5 10 10 5 1
```

Check: C(5, 2) = C(4, 1) + C(4, 2) = 4 + 6 = 10, matching the triangle. ✓

These same numbers are the coefficients when you expand a power of a sum — the binomial theorem:

(a + b)ⁿ = ∑ₖ₌₀ⁿ C(n, k) · a^(n−k) · bᵏ.

Worked example. Expand (a + b)³. The row for n = 3 is 1, 3, 3, 1:

(a + b)³ = a³ + 3a²b + 3ab² + b³.

Numeric check at a = b = 1: left side (1+1)³ = 8; right side 1+3+3+1 = 8. ✓ And a slightly harder one, (x + 2)³ = x³ + 3x²·2 + 3x·2² + 2³ = x³ + 6x² + 12x + 8; at x = 1 both sides give 27. ✓

### A probability preview: the binomial distribution

Here is where counting turns into probability. Flip a fair coin 4 times. What is the chance of exactly 2 heads? There are C(4, 2) = 6 arrangements of “which two flips are heads” (HHTT, HTHT, …), and each specific arrangement has probability (½)²·(½)² = 1/16. So

P(exactly 2 heads) = C(4, 2) · (½)² · (½)² = 6 · 1/16 = 6/16 = 3/8 = 0.375.

The general pattern — C(n, k) ways, each with probability pᵏ(1−p)^(n−k) — is the binomial distribution, the star of Chapter 3.

→ Used in Chapter 3: combinations C(n, k) are the counting engine of the binomial distribution. Chapter 14: factorials and counting arguments justify Huffman coding and the counting of code lengths. Throughout, “how many parameters does this model have?” is a product-rule question — a layer mapping 300 inputs to 200 outputs has 300·200 = 60,000 weights.

## 0.5 Trigonometry and the Unit Circle

Trigonometry began as the study of triangles, but its modern payload is the description of anything that repeats: waves, oscillations, rotations, and cycles. In machine learning, sine and cosine appear in three headline places — the sinusoidal position codes inside transformers (Chapter 12), the Fourier and wavelet features for signals (Chapter 7), and the angle-based similarity of vectors (Chapters 4 and 5). We build the ideas from a single circle.

![0.4: On the unit circle, cos θ and sin θ are the coordinates of the point at angle θ; tracing θ generates the sine wave.](../assets/figures/ml_concept_0.4_09e1caeb.png)

*Figure 0.4 — original teaching graphic.*

### Degrees versus radians (and why radians win)

Intuition. You know angles in degrees: a right angle is 90°, a full turn is 360°. Degrees are a human convention (360 is a nice, very divisible number). Mathematics prefers a unit tied to the circle itself.

Definition (radian). Draw a circle of radius 1. The radian measure of an angle is the length of arc it cuts on that circle. A full circle has circumference 2π, so a full turn is 2π radians. That gives the master conversion:

180° = π radians.

To convert, multiply by the appropriate form of 1:

radians = degrees · (π / 180), degrees = radians · (180 / π).

Worked conversions.

30° = 30·π/180 = π/6.

120° = 120·π/180 = 2π/3.

π/4 radians = (π/4)·(180/π) = 45°.

1 radian = 180/π ≈ 57.30°.

Radians win because they make calculus clean: with radians, the slope of sin at 0 is exactly 1 and no stray conversion factor of π/180 haunts every derivative. Assume radians everywhere unless a “°” is written.

### The unit circle and the definitions of sin, cos, tan

Intuition. Place a point on the unit circle and let a spoke from the center make angle θ with the positive x-axis, measured counterclockwise. As θ grows, the point travels around the rim. Its shadow on the horizontal axis and on the vertical axis are the cosine and sine.

Definition. For the point where the angle-θ spoke meets the unit circle:

cos θ = its x-coordinate,

sin θ = its y-coordinate,

tan θ = sin θ / cos θ (the slope of the spoke), undefined where cos θ = 0.

For a right triangle these reduce to the school ratios “sine = opposite/hypotenuse, cosine = adjacent/hypotenuse, tangent = opposite/adjacent,” because the hypotenuse here has length 1.

### Key values worth memorizing

Five angles cover most hand calculations. Reading coordinates off the unit circle:

| θ | 0 | π/6 (30°) | π/4 (45°) | π/3 (60°) | π/2 (90°) |
| --- | --- | --- | --- | --- | --- |
| sin θ | 0 | 1/2 | √2/2 | √3/2 | 1 |
| cos θ | 1 | √3/2 | √2/2 | 1/2 | 0 |
| tan θ | 0 | 1/√3 | 1 | √3 | undefined |

Numerically √2/2 ≈ 0.7071 and √3/2 ≈ 0.8660. Notice the sine column rising 0 → 1 while the cosine column falls 1 → 0: sine and cosine are the same shape, shifted by a quarter turn.

### Periodicity, amplitude, and phase

Because going once around the circle returns you to the start, sine and cosine repeat every 2π:

sin(θ + 2π) = sin θ, cos(θ + 2π) = cos θ.

We call 2π the period. Their values never leave [−1, 1]; the amplitude is how far they swing from center (here 1). A phase is a horizontal shift: cos θ = sin(θ + π/2), so cosine is just sine reported a quarter-turn early. Plotted against θ, both trace the familiar smooth wave — sine starting at 0 and climbing, cosine starting at 1 and falling — each completing one full ripple over an interval of length 2π.

### The Pythagorean identity and a couple of friends

Because (cos θ, sin θ) sits on a circle of radius 1, its coordinates obey x² + y² = 1. That is the single most-used identity in the subject:

sin²θ + cos²θ = 1.

Check at θ = π/6: (1/2)² + (√3/2)² = 1/4 + 3/4 = 1. ✓ At θ = π/4: (√2/2)² + (√2/2)² = 1/2 + 1/2 = 1. ✓

Two more that recur (the angle-addition formulas):

sin(α + β) = sin α cos β + cos α sin β, cos(α + β) = cos α cos β − sin α sin β.

Setting α = β gives the double-angle rule sin(2θ) = 2 sin θ cos θ. Also useful: sine is odd, sin(−θ) = −sin θ, while cosine is even, cos(−θ) = cos θ.

### Sinusoids: A·sin(ωt + φ)

Real signals are not the bare sin θ; they are stretched and shifted. The general sinusoid is

y(t) = A · sin(ω t + φ),

with three knobs: A the amplitude (height of the swing), ω the angular frequency (how fast it cycles, in radians per unit time), and φ the phase (where in the cycle it starts). The period is T = 2π/ω and the ordinary frequency is f = ω/(2π). For instance y = 3 sin(2t + π/2) swings between −3 and +3, has period 2π/2 = π, and is shifted a quarter-cycle early. This is the exact vocabulary Chapter 7 uses to decompose a signal into component waves, and Chapter 12 uses to build position codes of many different frequencies.

### The dot product and the angle between vectors

Intuition. Cosine measures alignment. Two arrows pointing the same way have cos θ = 1; perpendicular arrows have cos θ = 0; opposite arrows have cos θ = −1. This links trigonometry directly to the vectors of Chapter 0.10.

Definition (preview). For two vectors a and b, the dot product a·b (multiply matching components, then add) relates to the angle θ between them by

a·b = |a| · |b| · cos θ, so cos θ = (a·b) / (|a| |b|),

where |a| = √(a·a) is the vector’s length.

Worked example. Let a = (1, 2, 2) and b = (2, 2, 1). Then

a·b = 1·2 + 2·2 + 2·1 = 2 + 4 + 2 = 8,

|a| = √(1 + 4 + 4) = √9 = 3, and |b| = √(4 + 4 + 1) = √9 = 3,

cos θ = 8 / (3·3) = 8/9 ≈ 0.889, so θ = arccos(0.889) ≈ 27.3°.

The two vectors point in nearly the same direction, so their cosine similarity is high. This single number — cos θ, ignoring vector length — is how search engines and recommendation systems decide that two documents or two users are “alike,” the subject of Chapters 4 and 5.

### An honest preview of Euler’s formula

You will meet the symbol i, the imaginary unit, defined by i² = −1. A complex number a + bi is just a pair (a, b) that we have taught to multiply in a special way; it can be pictured as a point in the plane. The astonishing bridge between complex numbers and trigonometry is Euler’s formula:

e^{iθ} = cos θ + i sin θ.

Read it as: “traveling angle θ around the unit circle” and “the complex number e^{iθ}” are the same motion. Plugging in θ = π gives cos π + i sin π = −1 + 0 = −1, i.e. the celebrated e^{iπ} + 1 = 0, tying together e, i, π, 1, and 0. You do not need to manipulate complex numbers yet; simply know that e^{iθ} is shorthand for a cosine-plus-sine pair. That compactness is exactly why Chapter 7 writes the Fourier transform with e^{iθ} instead of juggling sin and cos separately.

→ Used in Chapter 4 / Chapter 5: cos θ between vectors is cosine similarity for embeddings and information retrieval. Chapter 7: sinusoids and e^{iθ} are the alphabet of Fourier and wavelet features. Chapter 12: sinusoidal positional encodings feed sequence position into transformers, and the attention score is a scaled dot product — cos θ wearing a different hat.

### Practice — 0.4–0.5

Work these by hand; a brief answer key follows.

Expand a sum. Write out and evaluate ∑ₖ₌₁⁴ (3k − 1).

Geometric series. A reward is worth 100 now, 100·(0.9) next step, 100·(0.9)² after that, and so on forever. Using a/(1 − r), find the total.

Combinations. A study enrolls 7 sites and you must pick 3 for a pilot (order irrelevant). Compute C(7, 3).

Binomial expansion. Use Pascal’s row 1, 4, 6, 4, 1 to expand (x + 1)⁴.

Radians and values. Convert 135° to radians, then give sin of that angle.

Cosine angle. For a = (1, 0, 1) and b = (0, 1, 1), compute cos θ and the angle θ.

Answer key.

Terms at k = 1,2,3,4 are 2, 5, 8, 11; sum = 2 + 5 + 8 + 11 = 26.

Here a = 100 and r = 0.9 (|r| < 1, so it converges): 100 / (1 − 0.9) = 100 / 0.1 = 1000.

C(7, 3) = 7!/(3!·4!) = 5040 / (6·24) = 5040 / 144 = 35.

(x + 1)⁴ = x⁴ + 4x³ + 6x² + 4x + 1 (check at x = 1: both sides 16).

135° = 135·π/180 = 3π/4; sin(3π/4) = √2/2 ≈ 0.707.

a·b = 1·0 + 0·1 + 1·1 = 1; |a| = √2, |b| = √2; cos θ = 1/(√2·√2) = 1/2, so θ = π/3 = 60°.

## 0.6 Limits, Continuity, and the Idea of the Derivative

Almost everything in machine learning comes down to one question: if I nudge this knob a tiny bit, how much does my error change? The mathematics that answers “how much does one thing change when another changes” is calculus. It rests on a single idea — the limit — so we start there, slowly.

![0.5: The derivative as a limit: as spacing h shrinks, secant slopes approach the tangent slope (2 at x = 1).](../assets/figures/ml_concept_0.5_b229f9ea.png)

*Figure 0.5 — original teaching graphic.*

### The intuition of a limit: approaching, not arriving

Imagine walking toward a wall, and each step covers half the remaining distance. After one step you are 1/2 of the way, then 3/4, then 7/8, then 15/16, … You never actually touch the wall, yet it is completely clear where you are heading. That destination — the value you get arbitrarily close to — is the limit.

A limit describes where a function is heading as its input approaches some value, regardless of what happens exactly at that value. This “regardless of the exact point” clause is the whole trick, and it is what lets us divide by something that is shrinking to zero without ever literally dividing by zero.

Consider the function

f(x) = (x² − 1) / (x − 1).

At x = 1 this is 0/0 — undefined, a genuine hole. But for every x other than 1, we can factor and cancel: x² − 1 = (x − 1)(x + 1), so f(x) = x + 1. Watch the value as x creeps toward 1:

| x | 0.9 | 0.99 | 1.01 | 1.1 |
| --- | --- | --- | --- | --- |
| f(x) | 1.9 | 1.99 | 2.01 | 2.1 |

From both sides the values home in on 2, even though f(1) does not exist. We write this as

lim_{x→1} f(x) = 2.

Read it aloud as “the limit, as x approaches 1, of f(x), equals 2.” The little arrow → means “approaches.”

### One-sided limits

Sometimes a function heads to different places depending on which direction you approach from. Take the sign-of-x function f(x) = x / |x|: for any positive x it equals +1, for any negative x it equals −1.

Approaching 0 from the right (x slightly above 0): lim_{x→0⁺} f(x) = +1.

Approaching 0 from the left (x slightly below 0): lim_{x→0⁻} f(x) = −1.

The superscripts ⁺ and ⁻ denote the two sides. Because the two one-sided limits disagree, the ordinary (two-sided) limit lim_{x→0} f(x) does not exist. A two-sided limit exists only when both sides agree.

### Continuity: no gaps, jumps, or holes

Intuitively, a function is continuous if you can draw its graph without lifting your pen — no sudden jumps, no holes. Formally, f is continuous at a point a when three things all hold:

f(a) is actually defined,

lim_{x→a} f(x) exists, and

the two agree: lim_{x→a} f(x) = f(a).

Our earlier f(x) = (x² − 1)/(x − 1) fails condition 1 at x = 1 (a removable hole — the limit exists, but the point is missing). The sign function fails condition 2 at 0 (a jump). Most functions you meet in ML — polynomials, exponentials, logarithms, the sigmoid — are continuous everywhere they are defined, which is exactly why the derivative machinery below works so smoothly.

### Average rate of change vs. instantaneous rate

Here is where calculus earns its keep. Suppose a toy epidemic has cumulative case count N(t) = t² (in thousands, with t in weeks). How fast are cases accumulating?

Over the interval from t = 2 to t = 4, the average rate of change is total change divided by elapsed time:

(N(4) − N(2)) / (4 − 2) = (16 − 4) / 2 = 12 / 2 = 6 thousand cases per week.

Geometrically this is the slope of the straight line — the secant line — connecting the two points (2, 4) and (4, 16) on the graph. The general formula for the secant slope between x = a and x = b is

(f(b) − f(a)) / (b − a).

But an average over two whole weeks blurs the detail. What is the rate at the single instant t = 2 — the instantaneous rate? To get it, we slide the second point closer and closer to the first and watch the secant slope settle down. Let the second point be t = 2 + h and let h shrink:

(N(2 + h) − N(2)) / h = ((2 + h)² − 4) / h = (4 + 4h + h² − 4) / h = (4h + h²) / h = 4 + h.

As h → 0, this approaches 4. The secant line has rotated into the tangent line — the straight line just grazing the curve at t = 2 — and its slope, 4, is the instantaneous rate of accumulation there. (Sanity check: the average rate 6 sits between the instantaneous rates at the two endpoints, 4 at t = 2 and 8 at t = 4, as it should.)

### The derivative as a limit

That limiting process — secant slope becoming tangent slope — is the derivative. For a function f at a point x, form the difference quotient

(f(x + h) − f(x)) / h,

which is the average rate of change over a step of size h. The derivative is its limit as the step shrinks to nothing:

f′(x) = lim_{h→0} (f(x + h) − f(x)) / h.

The derivative f′(x) is a new function: plug in any x, and it returns the slope of the tangent — the instantaneous rate of change — at that point.

### Worked example: the derivative of f(x) = x² from first principles

Every intermediate is checkable. Start with the difference quotient and expand (x + h)² = x² + 2xh + h²:

(f(x + h) − f(x)) / h = ((x + h)² − x²) / h = (x² + 2xh + h² − x²) / h = (2xh + h²) / h.

Now — and this is the pivotal move — because h is approaching 0 but is not yet 0, we may cancel it:

= 2x + h.

Finally take the limit. As h → 0 the leftover h vanishes:

f′(x) = lim_{h→0} (2x + h) = 2x.

Let us confirm numerically at x = 3, where the formula predicts f′(3) = 2·3 = 6. Using ever-smaller steps h:

| h | (f(3+h) − f(3)) / h | value |
| --- | --- | --- |
| 0.1 | (9.61 − 9)/0.1 | 6.1 |
| 0.01 | (9.0601 − 9)/0.01 | 6.01 |
| 0.001 | (9.006001 − 9)/0.001 | 6.001 |

The quotient marches straight toward 6. The limit is real; we do not need to reach h = 0 to know its destination.

### Notation

Two notations for the derivative appear throughout this book, and they mean the same thing:

Lagrange: f′(x), read “f prime of x.” Compact; good for stating rules.

Leibniz: dy/dx, read “d y d x,” where y = f(x). It literally evokes “an infinitesimal change in y divided by an infinitesimal change in x,” and it keeps track of which variable you are differentiating with respect to — invaluable once several variables are in play.

So for y = x² we may write f′(x) = 2x or dy/dx = 2x interchangeably.

→ Used in Chapter 8: the derivative is the engine of gradient descent — the slope tells the optimizer which way is downhill. → Used in Chapter 10: backpropagation is nothing but derivatives, chained together across a network.

## 0.7 Differential Calculus and One-Variable Optimization

Computing every derivative from the limit definition would be exhausting. Fortunately a small set of rules lets us differentiate almost any formula by inspection. We collect them, prove the one that matters most (the chain rule, which powers backpropagation), and then use them to find the bottom of a loss curve — the core task of training a model.

![0.6: Optima occur where f′ = 0. A convex function has one global minimum; a non-convex one can have several critical points.](../assets/figures/ml_concept_0.6_1492d114.png)

*Figure 0.6 — original teaching graphic.*

### The rules of differentiation

Throughout, c is a constant, n is a fixed power, and f and g are functions of x.

```
Constant rule: d/dx [c] = 0
Power rule: d/dx [xⁿ] = n · x^(n−1)
Constant-multiple rule: d/dx [c · f] = c · f′
Sum rule: d/dx [f + g] = f′ + g′
Product rule: d/dx [f · g] = f′·g + f·g′
Quotient rule: d/dx [f / g] = (f′·g − f·g′) / g²
Chain rule: d/dx [f(g(x))] = f′(g(x)) · g′(x)
```

Power + constant-multiple + sum, worked. Differentiate f(x) = 3x² + 2x − 5. Handle each term: the derivative of 3x² is 3·(2x) = 6x; of 2x is 2·(1) = 2; of the constant −5 is 0. So f′(x) = 6x + 2. At x = 1, f′(1) = 8 — the curve rises 8 units per unit of x there.

Product rule, worked. Differentiate f(x) = x²·eˣ. Let the two factors be u = x² (so u′ = 2x) and v = eˣ (so v′ = eˣ, see below). Then

f′(x) = u′v + uv′ = 2x·eˣ + x²·eˣ = eˣ(2x + x²).

Quotient rule, worked. Differentiate f(x) = x / (x + 1). Take u = x (u′ = 1) and v = x + 1 (v′ = 1):

f′(x) = (u′v − uv′) / v² = (1·(x + 1) − x·1) / (x + 1)² = (x + 1 − x) / (x + 1)² = 1 / (x + 1)².

### The chain rule — worked slowly, because it is backpropagation

The chain rule differentiates a composition: a function inside another function. The recipe: derivative of the outer function (evaluated at the inner), times derivative of the inner function. Think of it as a conversion chain — if y changes 3× as fast as u, and u changes 2× as fast as x, then y changes 3·2 = 6× as fast as x. Rates multiply.

Worked example. Differentiate f(x) = (3x² + 1)⁴.

Name the inner function: u = 3x² + 1, so the outer is u⁴.

Outer derivative (treat u as the variable): d/du [u⁴] = 4u³ = 4(3x² + 1)³.

Inner derivative: u′ = 6x.

Multiply: f′(x) = 4(3x² + 1)³ · 6x = 24x (3x² + 1)³.

A second, ML-flavored chain example. Differentiate f(x) = e^(−x²/2) (the shape of the bell curve). Inner: u = −x²/2, so u′ = −x. Outer: d/du[eᵘ] = eᵘ. Multiply:

f′(x) = e^(−x²/2) · (−x) = −x · e^(−x²/2).

A neural network is a deep stack of such compositions — linear step, then nonlinearity, then linear step, then nonlinearity, layer after layer. Backpropagation applies the chain rule from the output back to each weight, multiplying the local rates together. Master this one rule and you have understood the mathematical heart of deep learning.

### Derivatives of the functions ML actually uses

```
d/dx [eˣ] = eˣ (the exponential is its own derivative)
d/dx [ln x] = 1 / x
d/dx [sin x] = cos x
d/dx [cos x] = −sin x
```

Here e ≈ 2.71828 is Euler’s number and ln is the natural logarithm (base e). The fact that eˣ is its own slope is why it shows up everywhere growth or decay is proportional to size.

### Deriving the sigmoid’s derivative: σ′ = σ(1 − σ)

The sigmoid squashes any real number into the interval (0, 1), turning a score into something we can read as a probability:

σ(x) = 1 / (1 + e^(−x)).

It runs classifiers and neurons alike, and its derivative has a famously tidy form. Write σ(x) = (1 + e(−x))(−1) and apply the chain rule. The outer function is u^(−1) with derivative −u^(−2); the inner is u = 1 + e^(−x) with derivative u′ = −e^(−x) (itself a chain-rule result, since d/dx[e^(−x)] = e^(−x)·(−1)). Multiplying:

σ′(x) = −(1 + e(−x))(−2) · (−e^(−x)) = e^(−x) / (1 + e^(−x))².

Now the elegant part. Notice that

1 − σ(x) = 1 − 1/(1 + e^(−x)) = (1 + e^(−x) − 1)/(1 + e^(−x)) = e^(−x)/(1 + e^(−x)).

Therefore

σ(x)·(1 − σ(x)) = [1/(1 + e^(−x))] · [e^(−x)/(1 + e^(−x))] = e^(−x)/(1 + e^(−x))²,

which is exactly σ′(x). So

σ′(x) = σ(x)·(1 − σ(x)).

Numeric check at x = 0: σ(0) = 1/(1 + 1) = 0.5, so the formula gives σ′(0) = 0.5·(1 − 0.5) = 0.25. Direct substitution agrees: e⁰/(1 + e⁰)² = 1/2² = 1/4 = 0.25. The payoff is practical — once a network has computed σ(x) on the forward pass, it gets σ′ almost for free on the backward pass.

### Higher derivatives

Differentiating f′ again gives the second derivative f″(x) (or d²y/dx²), the rate at which the slope itself is changing — the “acceleration” of the function. For f(x) = x³: f′ = 3x², f″ = 6x, f‴ = 6, and f⁗ = 0. The second derivative is what tells maxima apart from minima, next.

### Increasing, decreasing, and critical points

The sign of f′ reveals the shape of f:

f′(x) > 0 on an interval → f is increasing (uphill) there.

f′(x) < 0 on an interval → f is decreasing (downhill) there.

f′(x) = 0 → a critical point: the tangent is flat. Peaks, valleys, and plateaus all live here.

To find where a function bottoms out (a minimum) or tops out (a maximum), we hunt for critical points by solving f′(x) = 0, then classify each one.

### First- and second-derivative tests

First-derivative test. Look at the sign of f′ just left and just right of a critical point c: - changes from + to − → local maximum (rising then falling: a peak), - changes from − to + → local minimum (falling then rising: a valley).

Second-derivative test. Often faster. At a critical point c where f′(c) = 0: - f″(c) > 0 → curve bends upward (concave up, ∪-shaped) → local minimum, - f″(c) < 0 → curve bends downward (concave down, ∩-shaped) → local maximum, - f″(c) = 0 → inconclusive; fall back on the first-derivative test.

### Convexity

A function is convex on an interval when f″(x) ≥ 0 throughout — it curves upward everywhere, like a bowl. Convexity is the property optimizers dream of: a convex function has no false valleys. Any critical point is automatically the global minimum, so gradient descent cannot get trapped in a lesser dip. Much of the design of loss functions is an effort to keep them convex, or nearly so.

![0.13: Gradient descent: stepping along the negative gradient walks the iterate down a convex loss surface to its minimum.](../assets/figures/ml_concept_0.13_6ef7569c.png)

*Figure 0.13 — original teaching graphic.*

### Worked minimization 1: the loss L(w) = (w − 3)² + 1

Read w as a single tunable weight and L(w) as the error it produces. We want the w that makes the error smallest.

Differentiate (chain rule on the squared term): L′(w) = 2(w − 3).

Set the derivative to zero: 2(w − 3) = 0 → w = 3.

Classify: L″(w) = 2 > 0 everywhere, so L is convex and w = 3 is the global minimum.

Minimum value: L(3) = (3 − 3)² + 1 = 1.

The best weight is 3 and the smallest achievable loss is 1. Now watch gradient descent discover this without being told the answer. The update rule is “step opposite the slope,” with a small step size η (the learning rate):

w ← w − η · L′(w).

Take η = 0.1 and start at w = 0. Since L′(w) = 2(w − 3), the update simplifies to w ← w − 0.2(w − 3) = 0.8w + 0.6:

| step | w (before) | L′(w) | w (after) |
| --- | --- | --- | --- |
| 1 | 0 | −6 | 0.6 |
| 2 | 0.6 | −4.8 | 1.08 |
| 3 | 1.08 | −3.84 | 1.464 |
| 4 | 1.464 | −3.072 | 1.7712 |

Each step nudges w toward 3, and the moves shrink as the slope flattens near the bottom. The fixed point of w = 0.8w + 0.6 is w = 3 — exactly the minimum calculus predicted.

```
w ← 0 # starting guess
η ← 0.1 # learning rate (step size)
repeat until w barely moves:
 g ← 2·(w − 3) # the derivative L′(w)
 w ← w − η·g # take one step downhill
```

### Worked minimization 2: a general quadratic

Minimize L(w) = 2w² − 8w + 3. Differentiate: L′(w) = 4w − 8. Set to zero: 4w − 8 = 0 → w = 2. Since L″(w) = 4 > 0, it is a minimum, with value L(2) = 2·4 − 8·2 + 3 = 8 − 16 + 3 = −5. (This matches the textbook shortcut that ax² + bx + c is minimized at w = −b/(2a) = 8/4 = 2 when a > 0.) Notice minima can be negative — “minimum” refers to the lowest output, not to any sign.

→ Used in Chapter 8: gradient descent minimizes regression and general loss functions this way. → Used in Chapter 10: backpropagation combines the chain rule with these tests to train networks. → Used in Chapter 3: maximum-likelihood estimation sets a derivative to zero to find the best-fitting parameter.

## 0.8 Integral Calculus and Areas

Differentiation breaks a total into its instantaneous rate. Integration runs the film backward: it accumulates a rate back into a total, and — read geometrically — it measures the area under a curve. For an epidemiologist this is the natural language of probability: the chance of an outcome is an area under a density curve.

![0.7: The definite integral is the area under a curve; under a probability density that area is a probability.](../assets/figures/ml_concept_0.7_a3194c90.png)

*Figure 0.7 — original teaching graphic.*

### The antiderivative (indefinite integral)

An antiderivative of f is any function F whose derivative is f — that is, F′ = f. We write

∫ f(x) dx = F(x) + C.

The elongated-S symbol ∫ means “integrate,” dx names the variable, and the constant of integration C is there because adding any constant to F does not change its slope (the derivative of a constant is 0), so antiderivatives come in a family shifted vertically. The basic reversals of the differentiation rules:

```
∫ xⁿ dx = x^(n+1) / (n + 1) + C (for n ≠ −1)
∫ eˣ dx = eˣ + C
∫ (1/x) dx = ln|x| + C (this covers the missing n = −1 case)
```

Quick check on the power rule: ∫ x² dx = x³/3 + C, and indeed d/dx[x³/3] = 3x²/3 = x². The rule undoes the power rule, as promised.

### The definite integral as signed area

The definite integral attaches limits a and b and returns a number — the signed area between the curve y = f(x) and the horizontal axis, from x = a to x = b:

∫ₐᵇ f(x) dx.

“Signed” means area above the axis counts as positive and area below counts as negative. For densities, which are never negative, all area is positive and this subtlety never bites.

### Riemann-sum intuition

How do you find the area under a curved top, where no simple geometry formula applies? Slice the region into many thin vertical rectangles, add up their areas, and refine. Split [a, b] into n strips each of width Δx = (b − a)/n; give strip i a height f(xᵢ) read off the curve; its area is f(xᵢ)·Δx. Summing gives a Riemann sum

Σ f(xᵢ) · Δx,

and the definite integral is the limit as the strips become infinitely thin (n → ∞).

Numeric example. Estimate the area under f(x) = x² from 0 to 1 with n = 4 strips (Δx = 0.25). Using the right edge of each strip, the heights at x = 0.25, 0.5, 0.75, 1.0 are 0.0625, 0.25, 0.5625, 1.0, which sum to 1.875; times Δx = 0.25 gives 0.46875 (an overestimate). Using the left edges (x = 0, 0.25, 0.5, 0.75) the heights sum to 0.875, giving 0.21875 (an underestimate). The true area is trapped between them, and their average, 0.34375, already hugs the exact answer 1/3 ≈ 0.3333. Finer slices would close the gap entirely.

### The Fundamental Theorem of Calculus

Adding up infinitely many rectangles by hand is hopeless. The Fundamental Theorem of Calculus (FTC) rescues us by revealing that integration and differentiation are inverse operations. Its evaluation form says: if F is any antiderivative of f (so F′ = f), then

∫ₐᵇ f(x) dx = F(b) − F(a).

Area collapses to a subtraction. For f(x) = x² an antiderivative is F(x) = x³/3, so

∫₀¹ x² dx = F(1) − F(0) = 1/3 − 0 = 1/3,

precisely the value our Riemann sums were converging to. The bracket notation [F(x)]ₐᵇ is shorthand for F(b) − F(a).

### Worked area example

Find the area under the line f(x) = 2x + 1 from x = 1 to x = 3. Antiderivative: F(x) = x² + x. Then

∫₁³ (2x + 1) dx = [x² + x]₁³ = (3² + 3) − (1² + 1) = (9 + 3) − (1 + 1) = 12 − 2 = 10.

Because the region here is a trapezoid, we can double-check by geometry: its parallel vertical sides have heights f(1) = 3 and f(3) = 7, and its width is 2, so the area is ½·(3 + 7)·2 = 10. The calculus and the geometry agree exactly.

### Substitution: reversing the chain rule (one worked case)

When an integrand contains a function and its derivative, substitution untangles it — it is the chain rule run backward. Compute ∫ 2x·(x² + 1)³ dx. Let u = x² + 1; then du/dx = 2x, i.e. du = 2x dx, which is exactly the 2x dx sitting in the integral. Substitute:

∫ 2x·(x² + 1)³ dx = ∫ u³ du = u⁴/4 + C = (x² + 1)⁴/4 + C.

Verify by differentiating the answer (chain rule): d/dx[(x² + 1)⁴/4] = 4(x² + 1)³·2x / 4 = 2x(x² + 1)³. It matches the integrand — and it is precisely the reverse of the chain-rule example (3x² + 1)⁴ we differentiated in §0.7.

### Integrals in probability

For a continuous random variable X — say a biomarker level, or a survival time — probability is described by a density function f(x). Densities obey two integral facts:

Total area is 1. A valid density is non-negative and encloses total area exactly one: ∫_{−∞}^{∞} f(x) dx = 1. Certainty corresponds to the whole area.

Probability is area. The chance that X lands between a and b is the area over that stretch: P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx.

Worked example. Let f(x) = 2x for 0 ≤ x ≤ 1 and 0 elsewhere. First confirm it is a legitimate density: it is non-negative on [0, 1], and

∫₀¹ 2x dx = [x²]₀¹ = 1 − 0 = 1. ✓

Now the probability that X falls in the lower half:

P(0 ≤ X ≤ 0.5) = ∫₀^{0.5} 2x dx = [x²]₀^{0.5} = 0.25 − 0 = 0.25.

So a quarter of the probability mass lies below 0.5 — sensible, since this density leans toward larger values.

### Expectation as an integral

The expected value E[X] — the long-run average of X — is the balancing point of the density, computed by integrating x weighted by f(x):

E[X] = ∫ x · f(x) dx.

For our f(x) = 2x on [0, 1]:

E[X] = ∫₀¹ x·(2x) dx = ∫₀¹ 2x² dx = [2x³/3]₀¹ = 2/3 − 0 = 2/3 ≈ 0.667.

The mean sits above 0.5, again reflecting the density’s rightward tilt.

### Normalization constants

What if a formula has the right shape but the wrong total area? We scale it by a normalization constant chosen to force the area to 1. Suppose we want a density proportional to x on [0, 2], written f(x) = c·x. Demand total area 1:

∫₀² c·x dx = c·[x²/2]₀² = c·(4/2) = 2c = 1 → c = 1/2.

So f(x) = x/2 on [0, 2] is the properly normalized density (check: ∫₀² (x/2) dx = ½·2 = 1 ✓). Every named distribution carries such a constant. The bell curve, whose kernel e^(−x²/2) we differentiated in §0.7, is normalized by 1/√(2π), because ∫_{−∞}^{∞} e^(−x²/2) dx works out to √(2π) — the constant that turns a bump into a probability density.

→ Used in Chapter 3: probability densities, expectation, and normalization constants are defined by exactly these integrals. → Used in Chapter 11: areas under curves reappear as model-evaluation scores such as the area under the ROC curve.

(Multivariable integration — volumes, joint densities, and integrals over several variables — is handled in §0.9.)

### Practice — 0.6–0.8

Work each by hand; the intermediate quantities are all checkable, and an answer key follows.

(Derivative from first principles.) Using the difference-quotient limit, show that if f(x) = 3x² then f′(x) = 6x. Expand f(x + h), simplify the quotient, then let h → 0.

(Differentiate, mixed rules.) Find the derivative of each:

f(x) = 4x³ − 5x² + 2x − 9;

g(x) = (x² + 1)⁶ (chain rule);

h(x) = x²·ln x (product rule);

k(x) = eˣ / (x + 1) (quotient rule).

(Find and classify an extremum.) For the loss L(w) = 3w² − 12w + 7, find the critical point, use the second-derivative test to classify it, and give the minimum value.

(Definite integral.) Evaluate ∫₀² (3x² + 2x) dx using the Fundamental Theorem of Calculus.

(Probability-area.) Let f(x) = 3x² for 0 ≤ x ≤ 1 and 0 elsewhere.

Verify that f is a valid density (total area 1).

Find P(X ≤ 0.5).

Find the expected value E[X].

(Chain rule / sigmoid.) The softplus function is s(x) = ln(1 + eˣ). Show that its derivative equals the sigmoid: s′(x) = σ(x) = 1/(1 + e^(−x)).

#### Answer key

f(x + h) = 3(x + h)² = 3x² + 6xh + 3h². The quotient (f(x + h) − f(x))/h = (6xh + 3h²)/h = 6x + 3h → 6x as h → 0.

12x² − 10x + 2. (b) 6(x² + 1)⁵·2x = 12x(x² + 1)⁵. (c) 2x·ln x + x²·(1/x) = 2x ln x + x. (d) (eˣ(x + 1) − eˣ·1)/(x + 1)² = x eˣ / (x + 1)².

L′(w) = 6w − 12 = 0 → w = 2. L″(w) = 6 > 0 → a minimum. Value: L(2) = 3·4 − 12·2 + 7 = 12 − 24 + 7 = −5.

∫₀² (3x² + 2x) dx = [x³ + x²]₀² = (8 + 4) − 0 = 12.

∫₀¹ 3x² dx = [x³]₀¹ = 1 ✓ (and 3x² ≥ 0 on [0, 1]). (b) P(X ≤ 0.5) = [x³]₀^{0.5} = 0.125. (c) E[X] = ∫₀¹ x·3x² dx = ∫₀¹ 3x³ dx = [3x⁴/4]₀¹ = 3/4 = 0.75.

By the chain rule, s′(x) = (1/(1 + eˣ))·eˣ = eˣ/(1 + eˣ). Multiply numerator and denominator by e^(−x): eˣ/(1 + eˣ) = 1/(e^(−x) + 1) = σ(x). (The derivative of softplus is exactly the sigmoid — a fact used to build smooth activations in Chapter 10.)

## 0.9 Multivariable Calculus: Gradients, Jacobians, Hessians, and Taylor

In sections 0.6–0.8 we learned to differentiate a function of one variable: given f(x), the derivative f′(x) tells us the slope, the rate at which f changes as we nudge x. But almost nothing in machine learning depends on a single number. A neural network’s loss depends on thousands or millions of weights at once. A logistic regression for stroke risk depends on age, blood pressure, glucose, and a dozen other inputs simultaneously. To train these models we must ask: if I nudge this input a little, holding all the others fixed, how does the output respond? And then: what is the single best direction to nudge everything at once?

![0.8: The gradient is perpendicular to the contour lines and points in the direction of steepest ascent; its negative points d](../assets/figures/ml_concept_0.8_a2f19287.png)

*Figure 0.8 — original teaching graphic.*

That is the subject of this section. It is the mathematical engine of every optimizer you will meet in this book. We build it up from the ground, leaning only on single-variable derivatives (0.6–0.8) and on vectors and matrices (0.10–0.11).

### Functions of several variables

A function of several variables takes in more than one number and returns one number. We write f(x, y) for two inputs, or, packing the inputs into a vector 𝐱 = (x₁, x₂, …, xₙ), we write f(𝐱) for n inputs. The output is still a single real number — we call such a function scalar-valued.

A running clinical analogy: imagine a risk score

f(age, ldl) = 0.03·age + 0.02·ldl

that returns a number. Two inputs go in, one score comes out. Later, f will be a loss — a single number measuring how wrong a model is — and its inputs will be the model’s parameters.

Geometrically, f(x, y) describes a surface. Over every point (x, y) on the flat floor, we raise the surface to height z = f(x, y). For two inputs we get a landscape of hills and valleys sitting above the plane. For n inputs we cannot picture the surface directly, but every idea below survives unchanged into n dimensions — that is the whole point of the vector notation.

### Level sets and contour maps

We cannot draw a surface in more than three dimensions, so we use a trick borrowed from topographic maps: the contour plot. A level set (or contour) is the set of all points where f takes one fixed value c:

{ (x, y) : f(x, y) = c }.

On a hiking map, each contour line connects points of equal elevation. Walk along a contour and your altitude never changes; walk across the contours and you climb or descend. Where contour lines bunch tightly together, the ground is steep; where they spread apart, it is gentle.

Hold on to two facts, because both return below:

Along a contour, f does not change.

The steepest way uphill is always perpendicular to the contour you are standing on.

The contour map is the flat “map” of the 3-D “mountain,” and reading it is exactly how we will reason about high-dimensional loss surfaces we cannot see.

### Partial derivatives

The partial derivative answers the one-variable question inside a many-variable world: hold every input fixed except one, and differentiate with respect to that one. We write ∂f/∂x (read “partial f, partial x”). The curved ∂ replaces the straight d to signal “there are other variables, and I am holding them constant.”

The mechanics are exactly the single-variable rules from 0.7 — you just treat every other variable as a constant number.

Formal definition. The partial derivative of f with respect to xᵢ is

∂f/∂xᵢ = limₕ→₀ [ f(…, xᵢ + h, …) − f(…, xᵢ, …) ] / h,

with all other inputs frozen. It measures the rate of change of f as you push xᵢ alone.

Worked example. Let f(x, y) = x²y.

To get ∂f/∂x, treat y as a constant coefficient. Then x²y is “y times x²,” whose x-derivative is y·(2x): so ∂f/∂x = 2xy.

To get ∂f/∂y, treat x as constant. Then x²y is “x² times y,” whose y-derivative is x²·1: so ∂f/∂y = x².

Evaluate at the point (x, y) = (3, 2):

∂f/∂x = 2·3·2 = 12, ∂f/∂y = 3² = 9.

Interpretation, in plain terms: standing at (3, 2), if we nudge x upward by a tiny amount ε (leaving y alone), f rises by about 12ε. If instead we nudge y up by ε, f rises by about 9ε. A quick sanity check confirms it: f(3, 2) = 18, and f(3.01, 2) = 3.01²·2 = 18.1202, a change of 0.1202 for a step of 0.01 — a rate of ≈ 12.02, matching ∂f/∂x = 12.

### The gradient: the direction of steepest ascent

The partials tell us the rate of change along each axis separately. Stack them into a single vector and we get the gradient, written ∇f (read “grad f” or “del f”):

∇f(𝐱) = ( ∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂xₙ ).

The gradient is a vector (bold, lowercase-style object) that lives in the same space as the inputs. It has two beautiful and central meanings:

Direction. ∇f points in the direction of steepest ascent — the compass bearing along which f increases fastest from your current point. Its negative, −∇f, points in the direction of steepest descent.

Magnitude. The length ‖∇f‖ is the rate of that fastest increase — how steep the steepest climb is.

And, connecting back to contours: ∇f is always perpendicular to the level set through your point. That is why the steepest path uphill crosses the contours at right angles.

This single fact — walk opposite the gradient to go downhill fastest — is the entire idea behind gradient descent, the algorithm that trains essentially every model in this book:

```
initialize 𝐱
repeat:
 𝐠 ← ∇f(𝐱) # gradient of the loss at the current point
 𝐱 ← 𝐱 − η·𝐠 # step downhill; η > 0 is the learning rate
until ‖𝐠‖ is small # near-flat ⇒ near a minimum
```

Worked example — gradient of a quadratic. Let

f(x, y) = x² + 3y² + xy.

Take the two partials:

∂f/∂x = 2x + y (∂/∂x of x² is 2x; of 3y² is 0; of xy is y), ∂f/∂y = 6y + x (∂/∂y of x² is 0; of 3y² is 6y; of xy is x).

So the gradient, as a vector, is ∇f(x, y) = (2x + y, 6y + x). At the point (1, 2):

∇f(1, 2) = ( 2·1 + 2 , 6·2 + 1 ) = (4, 13).

From (1, 2), the fastest way to increase f is to move in the direction (4, 13); to decrease f fastest — what an optimizer wants — move in −(4, 13) = (−4, −13). The steepness of that climb is ‖∇f‖ = √(4² + 13²) = √(16 + 169) = √185 ≈ 13.60.

→ Used in Chapter 8 (least squares and gradient-based fitting), Chapter 13 (policy-gradient methods, which ascend the gradient of expected reward), and Chapter 14 (optimization).

### Directional derivatives

The partials give the rate of change along the axis directions only. What if we want the rate of change in some arbitrary direction — say, northeast? That is the directional derivative. For a unit vector 𝐮 (a direction, length 1), the rate of change of f at 𝐱 in the direction 𝐮 is simply the dot product of the gradient with 𝐮:

D_𝐮 f = ∇f · 𝐮.

Recall from 0.10 that a dot product equals ‖∇f‖ ‖𝐮‖ cos θ, and since ‖𝐮‖ = 1,

D_𝐮 f = ‖∇f‖ cos θ,

where θ is the angle between 𝐮 and the gradient. This one line proves the claims above:

θ = 0 (𝐮 aligned with ∇f): cos θ = 1, the value is largest — steepest ascent.

θ = 180° (𝐮 opposite ∇f): cos θ = −1, the value is most negative — steepest descent.

θ = 90° (𝐮 perpendicular to ∇f): cos θ = 0, the value is zero — you are moving along a contour and f does not change.

Worked example. With ∇f(1, 2) = (4, 13), take the direction 𝐮 = (3, 4)/5 = (0.6, 0.8), which has length √(0.6² + 0.8²) = 1. Then

D_𝐮 f = ∇f · 𝐮 = 4·0.6 + 13·0.8 = 2.4 + 10.4 = 12.8.

As it must, 12.8 is less than the steepest possible rate ‖∇f‖ ≈ 13.60 — no direction beats the gradient itself.

### The multivariable chain rule (one step of backpropagation)

In 0.8 the single-variable chain rule let us differentiate a composition, f(g(x))′ = f′(g(x))·g′(x) — multiply the local rates along the chain. The multivariable version is the same idea, and it is the mathematical heart of backpropagation, the algorithm that trains neural networks.

The rule. If a quantity L depends on an intermediate a, which depends on z, which depends on a parameter w, then

∂L/∂w = (∂L/∂a) · (∂a/∂z) · (∂z/∂w).

You multiply the local derivatives along the path from w to L. When a variable feeds L through several paths, you sum the contributions of the paths — but our example is a single clean chain.

Fully worked backprop mini-example. Consider the tiniest possible neural network: one input x, one weight w, one bias b, a sigmoid activation, and a squared-error loss against a target y. This is the computation, broken into steps (the “forward pass”):

z = w·x + b (the pre-activation, a weighted input plus bias) a = σ(z) (the activation / prediction, with σ(z) = 1 / (1 + e⁻ᶻ)) L = (a − y)² (the squared-error loss)

We want ∂L/∂w and ∂L/∂b — how the loss responds to each parameter — so gradient descent knows how to adjust them. Compute the local derivatives:

∂L/∂a = 2(a − y), ∂a/∂z = σ′(z) = σ(z)(1 − σ(z)) = a(1 − a), (the sigmoid’s tidy derivative) ∂z/∂w = x, ∂z/∂b = 1.

Chain them together:

∂L/∂w = 2(a − y) · a(1 − a) · x, ∂L/∂b = 2(a − y) · a(1 − a) · 1.

Now put in numbers. Let w = 0.5, x = 2, b = −1, and target y = 1. Forward pass:

z = 0.5·2 + (−1) = 1 − 1 = 0, a = σ(0) = 1 / (1 + e⁰) = 1 / 2 = 0.5, L = (0.5 − 1)² = (−0.5)² = 0.25.

Backward pass (multiply the local rates, right to left):

∂L/∂a = 2(0.5 − 1) = 2·(−0.5) = −1, ∂a/∂z = a(1 − a) = 0.5·0.5 = 0.25, ∂L/∂w = (−1)·(0.25)·(x = 2) = −0.5, ∂L/∂b = (−1)·(0.25)·(1) = −0.25.

So ∇L = (∂L/∂w, ∂L/∂b) = (−0.5, −0.25). Because both partials are negative, increasing w and b would decrease the loss — and gradient descent does exactly that. With learning rate η = 0.1:

w ← 0.5 − 0.1·(−0.5) = 0.55, b ← −1 − 0.1·(−0.25) = −0.975.

Does the loss actually drop? New z = 0.55·2 − 0.975 = 0.125, so a = σ(0.125) ≈ 0.531, and L ≈ (0.531 − 1)² ≈ 0.220 — down from 0.25. The step worked. In pseudocode the whole thing is:

```
# forward pass
z = w*x + b
a = sigmoid(z)
L = (a - y)**2

# backward pass — the chain rule, right to left
dL_da = 2*(a - y)
da_dz = a*(1 - a)
dL_dw = dL_da * da_dz * x # ∂L/∂w
dL_db = dL_da * da_dz * 1 # ∂L/∂b
```

That is one step of backpropagation. A deep network simply has a much longer chain — dozens of layers — and backprop multiplies the local derivatives all the way from the loss back to each weight.

→ Used in Chapter 10 (backpropagation) — this is the single most important calculation in the book.

### The Jacobian: derivatives of vector-valued functions

So far f returned one number. But a neural-network layer takes a vector in and puts a vector out — it is vector-valued. Write such a function as 𝐟 : ℝⁿ → ℝᵐ, meaning n inputs go in and m outputs come out:

𝐟(𝐱) = ( f₁(𝐱), f₂(𝐱), …, fₘ(𝐱) ).

Each output component fᵢ has its own gradient. Stack those gradients as the rows of a matrix and you get the Jacobian 𝐉, an m×n matrix whose (i, j) entry is ∂fᵢ/∂xⱼ:

```
⎡ ∂f₁/∂x₁ ∂f₁/∂x₂ ⎤
 𝐉 = ⎢ ⎥ (here m = n = 2)
 ⎣ ∂f₂/∂x₁ ∂f₂/∂x₂ ⎦
```

The shape is worth memorizing: rows index outputs, columns index inputs. Row i is the gradient of output fᵢ. When there is only one output (m = 1), the Jacobian collapses to a single row — which is exactly the gradient (written as a row).

Worked example. Let 𝐟(x, y) = ( x² + y , 3xy ). The four partials are

∂f₁/∂x = 2x, ∂f₁/∂y = 1, ∂f₂/∂x = 3y, ∂f₂/∂y = 3x.

So the Jacobian is

```
⎡ 2x 1 ⎤
 𝐉 = ⎢ ⎥ , and at (x, y) = (1, 2):
 ⎣ 3y 3x ⎦
⎡ 2 1 ⎤
 𝐉(1,2) = ⎢ ⎥ .
 ⎣ 6 3 ⎦
```

A clean special case makes the connection to neural nets explicit. If a layer is linear, 𝐟(𝐱) = 𝐖𝐱, then ∂fᵢ/∂xⱼ = Wᵢⱼ, so the Jacobian is the weight matrix: 𝐉 = 𝐖. Backprop through a linear layer is therefore just multiplication by 𝐖.

→ Used in Chapter 10: backprop through a network chains Jacobians layer by layer.

### The Hessian: curvature and second derivatives

The gradient captures slope — a first-order, straight-line picture. But loss surfaces curve, and curvature is what distinguishes a bowl (a minimum) from a saddle. Curvature is second-order information, and for many variables it is collected in the Hessian matrix 𝐇, the matrix of all second partial derivatives. For two variables:

```
⎡ ∂²f/∂x² ∂²f/∂x∂y ⎤
 𝐇 = ⎢ ⎥ .
 ⎣ ∂²f/∂y∂x ∂²f/∂y² ⎦
```

The entry ∂²f/∂x∂y means “differentiate by x, then by y.” A fundamental result (Clairaut’s/Schwarz’s theorem) says that for the smooth functions we use, the order does not matter: ∂²f/∂x∂y = ∂²f/∂y∂x. Therefore the Hessian is symmetric — it equals its own transpose. That symmetry matters enormously in Chapter 7 (quadratic forms) and Chapter 12 (eigenvectors).

Worked example — Hessian of our quadratic. Return to f(x, y) = x² + 3y² + xy, whose gradient we found to be ∇f = (2x + y, 6y + x). Differentiate each partial again:

∂²f/∂x² = ∂/∂x (2x + y) = 2, ∂²f/∂y² = ∂/∂y (6y + x) = 6, ∂²f/∂x∂y = ∂/∂y (2x + y) = 1, ∂²f/∂y∂x = ∂/∂x (6y + x) = 1.

So

```
⎡ 2 1 ⎤
 𝐇 = ⎢ ⎥ .
 ⎣ 1 6 ⎦
```

Note two things. First, the off-diagonal entries agree (both 1) — symmetry, as promised. Second, this Hessian has no x or y in it: it is constant. That is special to quadratics, and it is precisely why quadratics are the model problem for optimization theory.

### Second-order Taylor expansion

In 0.8 we approximated a one-variable function near a point by a line (first-order Taylor) or a parabola (second-order). The multivariable version lets us approximate a whole loss surface near a point 𝐱 by a simple quadratic bowl. For a small step 𝐝,

f(𝐱 + 𝐝) ≈ f(𝐱) + ∇f(𝐱)ᵀ 𝐝 + ½ 𝐝ᵀ 𝐇(𝐱) 𝐝.

Read the three pieces left to right: the value at 𝐱, a linear correction from the gradient (the slope term, ∇fᵀ𝐝 is just the dot product ∇f · 𝐝), and a quadratic correction from the Hessian (the curvature term). The quantity 𝐝ᵀ𝐇𝐝 is a quadratic form — a matrix sandwiched between a vector and its transpose, producing a single number (Chapter 7).

Worked example. Use f(x, y) = x² + 3y² + xy at 𝐱 = (1, 2), where we already know f(1, 2) = 1 + 3·4 + 2 = 15, ∇f = (4, 13), and 𝐇 = [[2, 1], [1, 6]]. Take the step 𝐝 = (0.1, −0.1).

Linear term:

∇fᵀ𝐝 = 4·(0.1) + 13·(−0.1) = 0.4 − 1.3 = −0.9.

Curvature term — first compute 𝐇𝐝:

𝐇𝐝 = ( 2·0.1 + 1·(−0.1) , 1·0.1 + 6·(−0.1) ) = ( 0.1 , −0.5 ),

then

𝐝ᵀ𝐇𝐝 = 0.1·0.1 + (−0.1)·(−0.5) = 0.01 + 0.05 = 0.06, so ½ 𝐝ᵀ𝐇𝐝 = 0.03.

Taylor estimate:

f(1.1, 1.9) ≈ 15 + (−0.9) + 0.03 = 14.13.

Now the exact value: f(1.1, 1.9) = 1.1² + 3·1.9² + 1.1·1.9 = 1.21 + 10.83 + 2.09 = 14.13. Exact. For a quadratic function, the second-order Taylor expansion is not an approximation at all — it is the function itself, because a quadratic has no third derivatives. (Notice too that the linear estimate alone, 15 − 0.9 = 14.1, misses by 0.03; the curvature term supplies exactly that correction.)

→ Used in Chapter 8 (Newton’s method jumps straight to the minimum of this local quadratic) and Chapter 14 (second-order optimization).

### Critical points, the Hessian test, and convexity

A critical point (or stationary point) is where the gradient vanishes: ∇f = 𝟎. There the surface is momentarily flat, so it is a candidate for a minimum, a maximum, or a saddle (up in one direction, down in another). The gradient alone cannot tell these apart — it is zero for all three. Curvature decides, and curvature is the Hessian.

The Hessian (second-derivative) test. At a critical point:

𝐇 positive definite (curves up in every direction) ⇒ local minimum (a bowl).

𝐇 negative definite (curves down in every direction) ⇒ local maximum.

𝐇 indefinite (up some ways, down others) ⇒ saddle point.

“Positive definite” means 𝐝ᵀ𝐇𝐝 > 0 for every nonzero step 𝐝 — every direction curves upward. For a 2×2 symmetric matrix there is a quick test: both the top-left entry and the determinant must be positive.

Worked example. Find and classify the critical point of f(x, y) = x² + 3y² + xy. Set ∇f = 𝟎:

2x + y = 0 and x + 6y = 0.

From the first, y = −2x. Substituting into the second: x + 6(−2x) = x − 12x = −11x = 0, so x = 0, then y = 0. The only critical point is the origin (0, 0), and f(0, 0) = 0. Test it with the (constant) Hessian 𝐇 = [[2, 1], [1, 6]]:

top-left entry = 2 > 0, determinant = 2·6 − 1·1 = 12 − 1 = 11 > 0.

Both positive ⇒ 𝐇 is positive definite ⇒ (0, 0) is a minimum. Every direction curves upward, so this bowl has a single lowest point.

The tie to convexity. A function is convex if it curves upward everywhere — formally, if its Hessian is positive (semi-)definite at every point, not just at the critical one. Convexity is the property optimizers dream about: a convex function has no misleading saddles and no local minima to get stuck in — any critical point is the global minimum, and rolling downhill is guaranteed to find it. Our f is convex because its Hessian is positive definite everywhere. Least-squares loss (Chapter 8) is convex for the same reason; deep-network losses (Chapter 10) are generally not, which is exactly why training them is hard and why saddle points matter.

→ Used in Chapter 7 (quadratic forms and definiteness) and Chapter 8 (convexity guarantees a unique least-squares solution).

### Constrained optimization and Lagrange multipliers

Often we must minimize or maximize f while obeying a constraint — some equation g(𝐱) = 0 that our answer must satisfy (a fixed budget, a unit-length vector, a probability that sums to 1). The unconstrained rule “∇f = 𝟎” no longer applies, because the best point on the constraint curve is usually not flat.

The geometric insight is elegant. Picture the contours of f and the constraint curve g = 0 drawn on the same map. As you slide along the constraint curve, you cross contours of f — the value of f rises or falls — until you reach the point where the constraint curve just grazes a contour of f without crossing it. At that tangent point you cannot do any better while staying legal. And “tangent” means the two curves share the same perpendicular direction — so their gradients are parallel:

∇f = λ ∇g.

The scalar λ (lambda) is the Lagrange multiplier. Together with the constraint g = 0, this gives us exactly enough equations to solve.

Worked example. Maximize f(x, y) = xy subject to x + y = 10, i.e. g(x, y) = x + y − 10 = 0. The gradients are ∇f = (y, x) and ∇g = (1, 1). The condition ∇f = λ∇g gives

y = λ and x = λ, so x = y.

Feeding x = y into the constraint x + y = 10 gives 2x = 10, so x = 5, y = 5, and λ = 5. The maximum product is f(5, 5) = 25. (A quick check: 4·6 = 24 and 1·9 = 9 both fall short of 25.) This is the familiar fact that among all rectangles of fixed perimeter, the square encloses the most area — and it fell straight out of “set the gradients parallel.”

→ Used in Chapter 13 and in constrained formulations such as support vector machines and trust-region methods, where Lagrange multipliers turn a constrained problem into a solvable system.

### Practice — 0.9

Work these by hand; a compact answer key follows so you can self-check.

Partial derivatives. For f(x, y) = x³y + 4y², find ∂f/∂x and ∂f/∂y, and evaluate both at (1, 2).

Gradient and steepest ascent. For f(x, y) = 2x² + y² − xy, find ∇f and evaluate it at (2, 1). In which unit direction does f increase fastest there, and how fast (give ‖∇f‖)?

Chain rule / backprop. With z = wx + b, a = σ(z), and L = (a − y)², compute ∂L/∂w and ∂L/∂b at w = 1, x = 1, b = −1, y = 0. (Hint: first do the forward pass to get z, then a = σ(z); recall σ(0) = 0.5 and σ′(z) = a(1 − a).)

Hessian. For f(x, y) = x⁴ + y² + 2xy, find the Hessian 𝐇(x, y) and evaluate it at (1, 1). Is it positive definite there?

Taylor approximation. Using f(x, y) = x² + 3y² + xy with f(1, 2) = 15, ∇f(1, 2) = (4, 13), and 𝐇 = [[2, 1], [1, 6]], estimate f(1.2, 1.9) via the second-order Taylor expansion. Compare with the exact value.

Classify a critical point. For f(x, y) = x² − y², find the critical point and classify it (minimum, maximum, or saddle) using the Hessian.

Answer key.

∂f/∂x = 3x²y, ∂f/∂y = x³ + 8y. At (1, 2): ∂f/∂x = 3·1·2 = 6, ∂f/∂y = 1 + 16 = 17.

∂f/∂x = 4x − y, ∂f/∂y = 2y − x, so ∇f(2, 1) = (4·2 − 1, 2·1 − 2) = (7, 0). Fastest increase is in the direction (1, 0), at rate ‖∇f‖ = 7.

Forward: z = 1·1 + (−1) = 0, a = σ(0) = 0.5, L = (0.5 − 0)² = 0.25. Backward: ∂L/∂a = 2(0.5 − 0) = 1, ∂a/∂z = 0.5·0.5 = 0.25, so ∂L/∂w = 1·0.25·(x = 1) = 0.25 and ∂L/∂b = 1·0.25·1 = 0.25.

∂f/∂x = 4x³ + 2y, ∂f/∂y = 2y + 2x, so ∂²f/∂x² = 12x², ∂²f/∂y² = 2, ∂²f/∂x∂y = 2. At (1, 1): 𝐇 = [[12, 2], [2, 2]]. Top-left entry 12 > 0 and determinant 12·2 − 2·2 = 20 > 0 ⇒ positive definite (a local minimum).

Step 𝐝 = (0.2, −0.1). Linear term ∇fᵀ𝐝 = 4·0.2 + 13·(−0.1) = 0.8 − 1.3 = −0.5. Curvature: 𝐇𝐝 = (2·0.2 + 1·(−0.1), 1·0.2 + 6·(−0.1)) = (0.3, −0.4), so 𝐝ᵀ𝐇𝐝 = 0.2·0.3 + (−0.1)·(−0.4) = 0.06 + 0.04 = 0.10, and ½·0.10 = 0.05. Estimate = 15 − 0.5 + 0.05 = 14.55. Exact: f(1.2, 1.9) = 1.44 + 3·3.61 + 2.28 = 14.55 — an exact match, since f is quadratic.

∇f = (2x, −2y) = 𝟎 ⇒ (x, y) = (0, 0). Hessian 𝐇 = [[2, 0], [0, −2]]: it curves up along x (entry +2) but down along y (entry −2), so it is indefinite ⇒ (0, 0) is a saddle point — the prototype of the saddles that make deep-network training hard.

## 0.10 Vectors and Vector Spaces

Almost everything in machine learning begins by turning a real-world object — a patient, an image, a word, a day of case counts — into a list of numbers. That list is a vector, and the mathematics of vectors is the grammar that the rest of this book speaks. This section builds that grammar from nothing. If you can add, multiply, and take a square root, you have every prerequisite you need.

![0.9: Vector operations: addition by the parallelogram rule, projection onto another vector, and the angle encoded by the dot ](../assets/figures/ml_concept_0.9_9d68b349.png)

*Figure 0.9 — original teaching graphic.*

### Two pictures of a vector

There are two ways to look at a vector, and holding both in your head at once is the whole trick.

Picture 1 — a vector is an arrow. In the flat plane of a sheet of paper, an arrow that points 3 units to the right and 4 units up is a vector. It has a length (how long the arrow is) and a direction (where it points). Where you draw it does not matter; only its length and direction do. We write it as an ordered pair of its horizontal and vertical parts:

```
𝐚 = [3, 4]
```

Picture 2 — a vector is an ordered list of numbers. Forget arrows for a moment. A vector is simply a column of numbers stacked in a fixed order. This is the picture that matters for data. Suppose you record a patient’s age, systolic blood pressure, and LDL cholesterol:

```
𝐩 = [64, 138, 155]
```

This is a feature vector: each slot (each component) holds one measured quantity, and the order is fixed so that slot 1 always means “age,” slot 2 always means “blood pressure,” and so on. A vector with 3 components lives in “3-dimensional space.” A vector with 200 lab values lives in 200-dimensional space — impossible to draw, but the arithmetic is identical.

The magic is that these are the same object. The list [3, 4] is both the arrow and the data point. Geometry gives us intuition (length, angle, distance); the list gives us something a computer can store and crunch.

### The space ℝⁿ

We write ℝ for the set of all ordinary real numbers (−2, 0, 3.7, √2, …). We write ℝⁿ for the set of all vectors with exactly n real components. So [3, 4] ∈ ℝ² (read “is a member of R-two”), and the patient vector [64, 138, 155] ∈ ℝ³. The little superscript n is just the count of numbers in the list. A dataset of 500 patients, each with 3 features, is 500 separate points living in ℝ³.

We write vectors in bold lowercase (𝐚, 𝐱, 𝐩) and ordinary numbers, called scalars, in plain type (3, λ, −2).

### Addition and scalar multiplication

Two operations define everything else.

Adding two vectors means adding them slot by slot (they must have the same number of slots):

```
[3, 4] + [4, 3] = [3+4, 4+3] = [7, 7]
```

Geometrically, you place the tail of the second arrow at the head of the first; the sum is the arrow from the very start to the very end (the “tip-to-tail” rule).

Scalar multiplication means stretching or shrinking a vector by multiplying every component by the same scalar:

```
2 · [3, 4] = [6, 8] (twice as long, same direction)
−1 · [3, 4] = [−3, −4] (same length, flipped to the opposite direction)
0.5 · [3, 4] = [1.5, 2] (half as long, same direction)
```

A negative scalar reverses the arrow; a scalar between 0 and 1 shrinks it. That is the entire behaviour.

### The dot product

The dot product (also called the inner product) takes two vectors of the same size and returns a single scalar. You multiply matching components and add up the results:

For 𝐚 = [a₁, a₂, …, aₙ] and 𝐛 = [b₁, b₂, …, bₙ],

```
𝐚 · 𝐛 = a₁b₁ + a₂b₂ + ··· + aₙbₙ = Σ aᵢbᵢ
```

The big Σ (“sigma”) is just shorthand for “add up all the terms.” Worked, with 𝐚 = [3, 4] and 𝐛 = [4, 3]:

```
𝐚 · 𝐛 = (3)(4) + (4)(3) = 12 + 12 = 24
```

The dot product is the single most important operation in this book. Intuitively it measures how much two vectors point the same way. It is large and positive when they align, near zero when they are perpendicular, and negative when they point in opposing directions. We will make that precise with the cosine formula below.

→ Used in Chapter 12: the “attention” mechanism scores how relevant one token is to another as a dot product of their vectors.

### Length and norms

The length (or magnitude, or norm) of a vector measures how big it is. There is more than one sensible way to measure “big,” and each has a name.

The L2 norm (Euclidean length) is the ordinary straight-line length you would measure with a ruler. It comes straight from the Pythagorean theorem — square each component, add, take the square root:

```
‖𝐱‖₂ = √(x₁² + x₂² + ··· + xₙ²)
```

For 𝐚 = [3, 4]:

```
‖𝐚‖₂ = √(3² + 4²) = √(9 + 16) = √25 = 5
```

Notice a shortcut: ‖𝐱‖₂² = 𝐱 · 𝐱. A vector dotted with itself gives its squared length. (Check: 𝐚 · 𝐚 = 9 + 16 = 25 = 5².)

The L1 norm (Manhattan length) adds up the absolute values of the components — the distance you would walk on a city grid where you can only travel along blocks:

```
‖𝐱‖₁ = |x₁| + |x₂| + ··· + |xₙ|
```

For 𝐚 = [3, 4]: ‖𝐚‖₁ = |3| + |4| = 7.

The L∞ norm (max norm) is simply the largest absolute component:

```
‖𝐱‖∞ = max(|x₁|, |x₂|, …, |xₙ|)
```

For 𝐚 = [3, 4]: ‖𝐚‖∞ = max(3, 4) = 4.

The three norms answer three different questions — “how far as the crow flies?” (L2), “how far along the grid?” (L1), and “what is the single biggest coordinate?” (L∞) — and different ML methods choose different ones on purpose.

Unit vectors. A unit vector is any vector whose L2 norm equals exactly 1. To turn any (non-zero) vector into a unit vector pointing the same way, divide it by its own length — a move called normalizing:

```
𝐚̂ = 𝐚 / ‖𝐚‖₂ = [3, 4] / 5 = [0.6, 0.8]
```

Check: ‖[0.6, 0.8]‖₂ = √(0.36 + 0.64) = √1 = 1. ✓ The little hat (𝐚̂) is the customary mark for “this vector has been normalized to length 1.” Normalizing throws away magnitude and keeps only direction — exactly what you want when comparing the shape of two patients’ profiles regardless of overall scale.

→ Used in Chapter 6: feature scaling and normalization put every feature on a comparable footing before a model sees it.

### Euclidean distance

The distance between two points is the length of the vector connecting them: subtract one from the other, then take the L2 norm.

```
dist(𝐚, 𝐛) = ‖𝐚 − 𝐛‖₂
```

With 𝐚 = [3, 4] and 𝐛 = [4, 3]:

```
𝐚 − 𝐛 = [3−4, 4−3] = [−1, 1]
dist(𝐚, 𝐛) = √((−1)² + 1²) = √(1 + 1) = √2 ≈ 1.414
```

This is precisely how far apart two data points sit. When we cluster patients into subgroups, “similar patients” means “patients a small Euclidean distance apart.”

→ Used in Chapter 4: k-means clustering and k-nearest-neighbours are built entirely on Euclidean distance between feature vectors.

### The angle between vectors and cosine similarity

The dot product and the norms combine into a formula for the angle θ between two vectors:

```
cos θ = (𝐚 · 𝐛) / (‖𝐚‖₂ · ‖𝐛‖₂)
```

This quantity is called the cosine similarity. It ranges from −1 to +1:

+1 — the vectors point in exactly the same direction (θ = 0°).

0 — the vectors are perpendicular (θ = 90°); they share nothing.

−1 — the vectors point in exactly opposite directions (θ = 180°).

Worked, with 𝐚 = [3, 4] and 𝐛 = [4, 3]. We already have every piece: 𝐚 · 𝐛 = 24, ‖𝐚‖₂ = 5, ‖𝐛‖₂ = 5.

```
cos θ = 24 / (5 · 5) = 24 / 25 = 0.96
```

Taking the inverse cosine, θ = arccos(0.96) ≈ 16.3°. The two vectors point in nearly the same direction, so their cosine similarity is close to 1 — exactly what the picture of two arrows both heading up-and-to-the-right would suggest.

Cosine similarity cares only about direction, not length. A patient recorded in different units, or a document that is simply longer, is not penalized — only the pattern of the numbers matters. That property makes it the default similarity measure for text and high-dimensional data.

→ Used in Chapter 12: cosine similarity underlies how transformers and search systems judge two vectors as “close in meaning.”

### Orthogonality

Two vectors are orthogonal (a fancy word for perpendicular) when their dot product is exactly zero:

```
𝐚 · 𝐛 = 0 ⇒ 𝐚 and 𝐛 are orthogonal
```

Example: [2, 1] · [−1, 2] = (2)(−1) + (1)(2) = −2 + 2 = 0. These two arrows meet at a perfect right angle, and cos θ = 0, so θ = 90°. Orthogonal vectors carry completely independent information — knowing where you are along one tells you nothing about the other. This is the idea that PCA (Chapter 7) exploits to build a set of non-redundant “directions” through a dataset.

### Projection

Projection answers: if I shine a light straight down onto vector 𝐛, what shadow does vector 𝐚 cast along 𝐛? The shadow is the part of 𝐚 that lies in the direction of 𝐛. The formula is:

```
proj_𝐛(𝐚) = ( (𝐚 · 𝐛) / (𝐛 · 𝐛) ) 𝐛
```

The fraction (𝐚 · 𝐛)/(𝐛 · 𝐛) is a scalar that says “how many copies of 𝐛 to lay down”; multiplying it back by 𝐛 gives an actual vector pointing along 𝐛.

Worked, projecting 𝐚 = [3, 4] onto 𝐛 = [4, 3]. We have 𝐚 · 𝐛 = 24, and 𝐛 · 𝐛 = 4² + 3² = 25.

```
proj_𝐛(𝐚) = (24 / 25) · [4, 3] = 0.96 · [4, 3] = [3.84, 2.88]
```

The shadow [3.84, 2.88] points along 𝐛 and has length √(3.84² + 2.88²) = √(14.7456 + 8.2944) = √23.04 = 4.8. That length, 4.8, is the scalar projection — the size of 𝐚’s shadow — and equals (𝐚 · 𝐛)/‖𝐛‖₂ = 24/5 = 4.8. ✓

Projection is how a model decomposes a vector into a “part along a direction I care about” plus a leftover “part orthogonal to it.” Least-squares regression and PCA are, at heart, giant projection problems.

→ Used in Chapter 7: PCA projects every data point onto a handful of important directions to compress it.

### Linear combinations, span, independence, basis, dimension

These five words describe how vectors build a space, and they sound harder than they are.

A linear combination of some vectors is what you get by scaling each one and adding the results. From 𝐮 = [1, 0] and 𝐯 = [0, 1]:

```
3𝐮 + 4𝐯 = 3·[1, 0] + 4·[0, 1] = [3, 0] + [0, 4] = [3, 4]
```

Every point in the plane can be reached this way — 𝐮 handles the horizontal amount and 𝐯 the vertical.

The span of a set of vectors is the collection of all linear combinations you can form from them — every destination they can reach. The span of {[1, 0], [0, 1]} is the entire 2-D plane ℝ².

A set of vectors is linearly independent if none of them is a linear combination of the others — each one adds a genuinely new direction. If one can be built from the others, the set is linearly dependent and contains redundancy. Quick example: [1, 2] and [2, 4] are dependent, because [2, 4] = 2·[1, 2] — the second points the same way as the first and reaches nowhere new. In data terms, a feature that is just “twice another feature” is redundant in exactly this sense.

A basis is a minimal independent set that spans a space — just enough vectors to reach everywhere, with none wasted. The standard basis for ℝ² is {[1, 0], [0, 1]}. The dimension of a space is the number of vectors in any basis: ℝ² has dimension 2, ℝ³ has dimension 3, ℝⁿ has dimension n. Dimension is simply “how many independent directions the space contains.”

→ Used in Chapter 7: PCA searches for a small basis of independent directions that captures most of the variation in a high-dimensional dataset.

### Putting it all together

For 𝐚 = [3, 4] and 𝐛 = [4, 3] we found, using nothing beyond arithmetic and a square root:

| Quantity | Formula | Result |
| --- | --- | --- |
| Dot product | 𝐚 · 𝐛 | 24 |
| L2 norm of 𝐚 | √(3² + 4²) | 5 |
| L1 norm of 𝐚 | \|3\| + \|4\| | 7 |
| L2 norm of 𝐛 | √(4² + 3²) | 5 |
| Cosine similarity | 24 / (5·5) | 0.96 |
| Euclidean distance | ‖𝐚 − 𝐛‖₂ | √2 ≈ 1.414 |
| Projection of 𝐚 onto 𝐛 | (24/25)·[4, 3] | [3.84, 2.88] |

Every advanced idea later in the book is assembled from these seven bricks.

## 0.11 Matrices and Linear Transformations

A single vector describes one data point. But we never have just one patient — we have a whole cohort, and we want to transform all of them at once. The object that holds many vectors, and that acts on vectors to transform them, is the matrix.

![0.10: A matrix acts as a linear transformation, mapping the unit square to a rotated/scaled or sheared image.](../assets/figures/ml_concept_0.10_702284c9.png)

*Figure 0.10 — original teaching graphic.*

### Two pictures of a matrix

As with vectors, hold two images at once.

Picture 1 — a matrix is a table of numbers. Rows and columns of scalars in a rectangular grid. A cohort of 3 patients with 3 features each is naturally a 3-row, 3-column table. We write matrices in bold uppercase:

```
[ 1 2 ]
𝐀 = [ 3 4 ]
```

Picture 2 — a matrix is a machine that transforms vectors. Feed a vector in, get a (usually different) vector out. Rotations, stretches, and projections of space are all matrices. This is the “linear map” view, and it is why matrices matter beyond mere bookkeeping.

### Shape (dimensions)

A matrix with m rows and n columns is called an “m × n” matrix (say “m by n”). The 𝐀 above is 2 × 2. The entry in row i, column j is written aᵢⱼ; for our 𝐀, a₁₂ = 2 (row 1, column 2). Row count always comes first. A single column of m numbers is an m × 1 matrix — which is just a vector, tying the two objects together.

### Addition and scalar multiplication

These work exactly as they did for vectors — entry by entry. Two matrices must have the same shape to be added:

```
[ 1 2 ] [ 5 6 ] [ 1+5 2+6 ] [ 6 8 ]
[ 3 4 ] + [ 7 8 ] = [ 3+7 4+8 ] = [ 10 12 ]
```

Scalar multiplication multiplies every entry by the scalar:

```
[ 1 2 ] [ 2 4 ]
2 · [ 3 4 ] = [ 6 8 ]
```

### The matrix–vector product

Multiplying a matrix by a vector is where matrices earn their keep. There are two equivalent views, and both are worth knowing. Take

```
[ 1 2 ] [ 5 ]
𝐀 = [ 3 4 ] 𝐱 = [ 6 ]
```

View A — rows dotted with the vector. Each entry of the output is the dot product of one row of 𝐀 with 𝐱:

```
row 1 · 𝐱 = (1)(5) + (2)(6) = 5 + 12 = 17
row 2 · 𝐱 = (3)(5) + (4)(6) = 15 + 24 = 39

𝐀𝐱 = [17, 39]
```

View B — a linear combination of the columns. The output is 𝐱’s components used as weights on 𝐀’s columns:

```
𝐀𝐱 = 5·[1, 3] + 6·[2, 4] = [5, 15] + [12, 24] = [17, 39]
```

Same answer, [17, 39]. View A is how you compute; View B is what it means — a matrix times a vector is a weighted mixture of the matrix’s columns. For a rule to work, 𝐀’s column count must equal 𝐱’s length; the result has as many entries as 𝐀 has rows.

→ Used in Chapter 10: a neural-network layer is exactly this — a weight matrix multiplying an input vector, over and over.

### Matrix–matrix multiplication

Multiplying two matrices means applying one transformation after another. The rule: entry (i, j) of the product is row i of the left matrix dotted with column j of the right matrix. Fully worked with

```
[ 1 2 ] [ 5 6 ]
𝐀 = [ 3 4 ] 𝐁 = [ 7 8 ]
```

Compute each of the four entries of 𝐀𝐁:

```
(1,1): row1·col1 = (1)(5) + (2)(7) = 5 + 14 = 19
(1,2): row1·col2 = (1)(6) + (2)(8) = 6 + 16 = 22
(2,1): row2·col1 = (3)(5) + (4)(7) = 15 + 28 = 43
(2,2): row2·col2 = (3)(6) + (4)(8) = 18 + 32 = 50

 [ 19 22 ]
𝐀𝐁 = [ 43 50 ]
```

Conformability. You can multiply an (m × n) matrix by an (n × p) matrix only when the inner numbers match — the left matrix’s columns must equal the right matrix’s rows. The result is (m × p). “Two-by-two times two-by-two” works because the middle twos agree.

Non-commutativity. Order matters: in general 𝐀𝐁 ≠ 𝐁𝐀. Reversing our example:

```
(1,1): (5)(1) + (6)(3) = 5 + 18 = 23
(1,2): (5)(2) + (6)(4) = 10 + 24 = 34
(2,1): (7)(1) + (8)(3) = 7 + 24 = 31
(2,2): (7)(2) + (8)(4) = 14 + 32 = 46

 [ 23 34 ]
𝐁𝐀 = [ 31 46 ]
```

Different from 𝐀𝐁. Doing transformation 𝐁 then 𝐀 is not the same as 𝐀 then 𝐁 — just as “put on socks, then shoes” differs from “shoes, then socks.”

### The transpose

The transpose of a matrix, written 𝐀ᵀ, flips it across its diagonal: rows become columns. Entry (i, j) moves to (j, i).

```
[ 1 2 ] [ 1 3 ]
𝐀 = [ 3 4 ] 𝐀ᵀ = [ 2 4 ]
```

A handy rule about transposing a product — the order reverses:

```
(𝐀𝐁)ᵀ = 𝐁ᵀ𝐀ᵀ
```

Quick check with our matrices: (𝐀𝐁)ᵀ is [[19, 43], [22, 50]], and computing 𝐁ᵀ𝐀ᵀ gives the same [[19, 43], [22, 50]]. ✓ The transpose shows up constantly because the product 𝐗ᵀ𝐗 — a data matrix times its own transpose — produces the covariance-like structure at the heart of PCA and regression.

→ Used in Chapter 7 and Chapter 8: the matrix 𝐗ᵀ𝐗 encodes how features co-vary and drives both PCA and the regression normal equations.

### Special matrices

A few matrices are important enough to have names.

The identity matrix 𝐈 has 1s on the diagonal and 0s everywhere else. It is the “do nothing” matrix: 𝐈𝐱 = 𝐱 for every vector.

```
[ 1 0 ]
𝐈 = [ 0 1 ]
```

A diagonal matrix has non-zero entries only on the diagonal. Multiplying by it simply scales each coordinate independently:

```
[ 2 0 ]
[ 0 3 ] scales x by 2 and y by 3
```

A symmetric matrix equals its own transpose (𝐀 = 𝐀ᵀ) — it is a mirror image across the diagonal:

```
[ 1 2 ]
[ 2 5 ] entry (1,2) = entry (2,1) = 2
```

Covariance matrices and the adjacency matrices of undirected graphs are always symmetric, which is why symmetry is worth recognizing on sight.

→ Used in Chapter 15: an undirected graph’s adjacency matrix is symmetric — a 1 in entry (i, j) means nodes i and j are connected.

### The inverse

The inverse of a square matrix 𝐀, written 𝐀⁻¹, is the matrix that undoes it: 𝐀𝐀⁻¹ = 𝐈. It plays the role that “divide” plays for numbers (there is no matrix “division”; you multiply by the inverse instead).

For a 2 × 2 matrix there is a direct formula. Given

```
[ a b ]
𝐀 = [ c d ]
```

first compute the number ad − bc (the determinant, next section), then

```
1 [ d −b ]
𝐀⁻¹ = ───────── · [ −c a ]
 ad − bc
```

Worked, with

```
[ 2 1 ]
𝐀 = [ 1 1 ]
```

The determinant is ad − bc = (2)(1) − (1)(1) = 2 − 1 = 1. Swap a and d, negate b and c, divide by 1:

```
1 [ 1 −1 ] [ 1 −1 ]
𝐀⁻¹ = ─── · [ −1 2 ] = [ −1 2 ]
 1
```

Verify by multiplying back:

```
[ 2 1 ] [ 1 −1 ] [ (2)(1)+(1)(−1) (2)(−1)+(1)(2) ] [ 1 0 ]
𝐀𝐀⁻¹ = [ 1 1 ] [ −1 2 ] = [ (1)(1)+(1)(−1) (1)(−1)+(1)(2) ] = [ 0 1 ] = 𝐈 ✓
```

When the inverse fails. If ad − bc = 0, you would be dividing by zero: the inverse does not exist and the matrix is called singular. A determinant of zero always signals a matrix that cannot be undone — it has collapsed information that cannot be recovered.

### The determinant

The determinant of a square matrix is a single number, written det(𝐀), measuring how the transformation scales area (in 2-D) or volume (in 3-D). For a 2 × 2 matrix:

```
det(𝐀) = ad − bc
```

Take the pure scaling matrix that doubles x and triples y:

```
[ 2 0 ]
𝐒 = [ 0 3 ] det(𝐒) = (2)(3) − (0)(0) = 6
```

A unit square fed through 𝐒 comes out a 2-by-3 rectangle with area 6 — precisely the determinant. The interpretation is completely general:

|det| > 1 — the transformation expands areas.

|det| < 1 — it shrinks them.

det < 0 — it also flips orientation (like a mirror).

det = 0 — it squashes space flat onto a line or a point. Area becomes zero, information is lost, and (as we just saw) the matrix is singular and has no inverse.

That last line is the one to remember: det = 0 ⇒ singular ⇒ no inverse.

### Solving a linear system 𝐀𝐱 = 𝐛

A system of linear equations is a matrix equation in disguise. Consider

```
2x + y = 5
 x + y = 3
```

Stack the coefficients into a matrix and the unknowns into a vector, and this is exactly 𝐀𝐱 = 𝐛:

```
[ 2 1 ] [ x ] [ 5 ]
𝐀 = [ 1 1 ] 𝐱 = [ y ] 𝐛 = [ 3 ]
```

Method 1 — elimination. Subtract the second equation from the first. The y-terms cancel:

```
(2x + y) − (x + y) = 5 − 3 ⇒ x = 2
```

Substitute x = 2 into x + y = 3: 2 + y = 3, so y = 1. Solution: 𝐱 = [2, 1].

Method 2 — the inverse. Since 𝐀𝐱 = 𝐛, multiplying both sides by 𝐀⁻¹ gives 𝐱 = 𝐀⁻¹𝐛. We already found 𝐀⁻¹ = [[1, −1], [−1, 2]] for this very matrix, so:

```
[ 1 −1 ] [ 5 ] [ (1)(5)+(−1)(3) ] [ 5 − 3 ] [ 2 ]
𝐱 = [ −1 2 ] [ 3 ] = [ (−1)(5)+(2)(3) ] = [ −5 + 6 ] = [ 1 ]
```

Both methods give 𝐱 = [2, 1]. Check against the originals: 2(2) + 1 = 5 ✓ and 2 + 1 = 3 ✓.

→ Used in Chapter 8: linear regression fits its coefficients by solving the “normal equations” 𝐗ᵀ𝐗 𝛃 = 𝐗ᵀ𝐲 — the same 𝐀𝐱 = 𝐛 shape, just larger.

### Rank and singularity

The rank of a matrix is the number of genuinely independent directions among its columns (equivalently, its rows) — how much non-redundant information it holds. A 2 × 2 matrix whose two columns point in different directions has rank 2 and is called full-rank; it is invertible and its determinant is non-zero.

But look at

```
[ 1 2 ]
[ 2 4 ] det = (1)(4) − (2)(2) = 0
```

Its second column, [2, 4], is just twice the first, [1, 2] — no new direction. Rank is only 1, the determinant is 0, and the matrix is singular. Rank deficiency, zero determinant, and non-invertibility are three faces of the same phenomenon: redundant columns. In data, this is what happens when one feature is a copy or a linear combination of others, and it is why such features must be spotted and removed.

### Matrices as geometric transformations

Returning to Picture 2, here are the transformations you will meet most often, each a small matrix that reshapes space when it multiplies a vector.

Scaling stretches each axis. Applying [[2, 0], [0, 3]] to [1, 1] gives [2, 3].

Rotation turns vectors about the origin. The 90°-counterclockwise rotation is

```
[ 0 −1 ]
𝐑 = [ 1 0 ]
```

Applied to [1, 0]: (0·1 + (−1)·0, 1·1 + 0·0) = [0, 1]. The arrow that pointed right now points up — a quarter turn. Note det(𝐑) = (0)(0) − (−1)(1) = 1: rotations preserve area, as they must.

Projection flattens space onto a line. The projection onto the x-axis,

```
[ 1 0 ]
𝐏 = [ 0 0 ]
```

sends [3, 4] to [3, 0] — the height is discarded. Its determinant is 0 (it collapses the plane onto a line), so it is singular and cannot be undone; once you have thrown away the y-coordinate, you cannot recover it.

Seeing matrices as actions — stretch, turn, flatten — rather than static tables is the mental shift that makes deep learning, PCA, and graph methods click. Every layer of a neural network and every step of PCA is one of these geometric moves.

→ Used in Chapter 10: stacking many weight matrices, each a learned transformation, is exactly what gives a deep network its power.

### Practice — 0.10–0.11

Work these with pencil and paper; answers follow.

1. Dot product, norm, cosine. For 𝐮 = [1, 2, 2] and 𝐯 = [2, 0, 4], compute 𝐮 · 𝐯, ‖𝐮‖₂, ‖𝐯‖₂, and the cosine similarity cos θ.

2. Three norms. For 𝐱 = [3, −4, 1], compute the L1 norm ‖𝐱‖₁, the L2 norm ‖𝐱‖₂, and the L∞ norm ‖𝐱‖∞.

3. Projection. Project 𝐚 = [4, 2] onto 𝐛 = [3, 0]. What does the result tell you about projecting onto the x-axis?

4. Matrix multiplication. With 𝐀 = [[1, 0], [2, 1]] and 𝐁 = [[3, 4], [1, 2]], compute 𝐀𝐁.

5. Inverse of a 2 × 2. Find 𝐀⁻¹ for 𝐀 = [[3, 1], [2, 1]], and verify that 𝐀𝐀⁻¹ = 𝐈.

6. Solve a system. Solve 3x + y = 9 and 2x + y = 7 both by elimination and by using the inverse from Exercise 5.

Answers.

1. 𝐮 · 𝐯 = (1)(2) + (2)(0) + (2)(4) = 2 + 0 + 8 = 10. ‖𝐮‖₂ = √(1 + 4 + 4) = √9 = 3. ‖𝐯‖₂ = √(4 + 0 + 16) = √20 ≈ 4.472. cos θ = 10 / (3 × 4.472) ≈ 10 / 13.42 ≈ 0.745 (θ ≈ 41.8°).

2. ‖𝐱‖₁ = 3 + 4 + 1 = 8. ‖𝐱‖₂ = √(9 + 16 + 1) = √26 ≈ 5.10. ‖𝐱‖∞ = max(3, 4, 1) = 4.

3. 𝐚 · 𝐛 = (4)(3) + (2)(0) = 12; 𝐛 · 𝐛 = 9. proj = (12/9)·[3, 0] = (4/3)·[3, 0] = [4, 0]. Projecting onto the x-axis simply keeps the x-component (4) and zeroes the rest — the shadow on a horizontal line.

4.

```
[ (1)(3)+(0)(1) (1)(4)+(0)(2) ] [ 3 4 ]
𝐀𝐁 = [ (2)(3)+(1)(1) (2)(4)+(1)(2) ] = [ 7 10 ]
```

5. det = (3)(1) − (1)(2) = 1. 𝐀⁻¹ = [[1, −1], [−2, 3]]. Check: 𝐀𝐀⁻¹ = [[ (3)(1)+(1)(−2), (3)(−1)+(1)(3) ], [ (2)(1)+(1)(−2), (2)(−1)+(1)(3) ]] = [[1, 0], [0, 1]] = 𝐈 ✓.

6. Elimination: subtract the second from the first, (3x + y) − (2x + y) = 9 − 7, so x = 2; then 2(2) + y = 7 gives y = 3. Inverse: 𝐱 = 𝐀⁻¹𝐛 = [[1, −1], [−2, 3]]·[9, 7] = [9 − 7, −18 + 21] = [2, 3]. Both give 𝐱 = [2, 3]. Check: 3(2) + 3 = 9 ✓, 2(2) + 3 = 7 ✓.

## 0.12 Eigenvalues, Eigenvectors, and Matrix Decompositions

![0.11: A matrix turns the unit circle into an ellipse; eigenvectors are the directions only stretched, by the eigenvalues λ.](../assets/figures/ml_concept_0.11_2667d291.png)

*Figure 0.11 — original teaching graphic.*

### The idea: directions a matrix only stretches

In §0.11 a matrix became a transformation: feed it a vector and it rotates, stretches, and shears the whole plane. Most input arrows come out pointing somewhere new. But almost every matrix has a few special directions along which nothing rotates at all — the arrow that goes in comes out pointing the same way, only longer or shorter. Those privileged directions are the matrix’s eigenvectors, and the stretch factor along each one is its eigenvalue.

Think of a sheet of rubber pinned at the origin and stretched. Some fibres get dragged sideways as the sheet deforms; but a few fibres just get longer or shorter while staying on their own line. Find those fibres and you understand the deformation completely — everything else is a blend of them. For a data scientist the payoff is enormous: eigenvectors are the “natural axes” of a matrix, the coordinate system in which a complicated transformation becomes simple scaling.

### Definition and the characteristic equation

A nonzero vector 𝐯 is an eigenvector of a square matrix 𝐀 if

```
𝐀𝐯 = λ𝐯
```

for some scalar λ, its eigenvalue. In words: applying 𝐀 to 𝐯 does the same thing as multiplying 𝐯 by a single number. The direction survives; only the length (and possibly the sign) changes.

To find the λ’s, rewrite the definition as 𝐀𝐯 − λ𝐯 = 0, or (𝐀 − λ𝐈)𝐯 = 0. We want a nonzero 𝐯 solving this. From §0.11 we know a square matrix sends some nonzero vector to zero only when it is singular — that is, when its determinant vanishes. So the eigenvalues are exactly the numbers λ making

```
det(𝐀 − λ𝐈) = 0.
```

This is the characteristic equation. For a 2 × 2 matrix it is a quadratic in λ, so there are (at most) two eigenvalues; for an n × n matrix it is a degree-n polynomial with n roots (counted with repeats).

### Worked example: a 2 × 2 matrix

Take the symmetric matrix

```
[ 2 1 ]
𝐀 = [ 1 2 ]
```

Step 1 — subtract λ from the diagonal.

```
[ 2−λ 1 ]
𝐀 − λ𝐈 = [ 1 2−λ ]
```

Step 2 — set the determinant to zero. Using det = ad − bc:

```
det(𝐀 − λ𝐈) = (2−λ)(2−λ) − (1)(1) = (2−λ)² − 1 = 0.
```

So (2−λ)² = 1, giving 2−λ = ±1, hence λ₁ = 3 and λ₂ = 1.

A handy sanity check: the eigenvalues must sum to the trace (the diagonal sum) and multiply to the determinant. Here 3 + 1 = 4 = 2 + 2 ✓ and 3 × 1 = 3 = (2)(2) − (1)(1) ✓. Equivalently, the characteristic equation is always λ² − (trace)λ + (det) = λ² − 4λ + 3 = (λ − 3)(λ − 1) = 0.

Step 3 — find each eigenvector by solving (𝐀 − λ𝐈)𝐯 = 0.

For λ₁ = 3:

```
[ −1 1 ] [ v₁ ] 
(𝐀 − 3𝐈)𝐯 = [ 1 −1 ] [ v₂ ] = 0 ⇒ −v₁ + v₂ = 0 ⇒ v₂ = v₁.
```

Any multiple of 𝐯₁ = [1, 1] works. (Eigenvectors have no preferred length; we usually report a direction and often normalize it to unit length, here [1, 1]/√2.)

For λ₂ = 1:

```
[ 1 1 ] [ v₁ ] 
(𝐀 − 1𝐈)𝐯 = [ 1 1 ] [ v₂ ] = 0 ⇒ v₁ + v₂ = 0 ⇒ v₂ = −v₁,
```

giving 𝐯₂ = [1, −1]. Verify directly: 𝐀𝐯₁ = [2·1 + 1·1, 1·1 + 2·1] = [3, 3] = 3·[1, 1] ✓, and 𝐀𝐯₂ = [2 − 1, 1 − 2] = [1, −1] = 1·[1, −1] ✓.

Notice 𝐯₁ · 𝐯₂ = (1)(1) + (1)(−1) = 0: the two eigenvectors are orthogonal. That is not luck — it happens for every symmetric matrix, as we discuss below.

### Geometric meaning: invariant axes and stretch factors

Picture the transformation 𝐀 acting on the plane. The line through [1, 1] is an invariant axis: any arrow on it comes out three times longer, still on the line. The line through [1, −1] is a second invariant axis, and vectors on it are left completely unchanged (stretch factor 1). Every other vector is a mix of these two, so 𝐀 stretches it by 3 in one diagonal direction and by 1 in the perpendicular one. A circle of input arrows becomes an ellipse whose long axis points along [1, 1] with semi-length 3 and whose short axis points along [1, −1] with semi-length 1. Eigenvalues are the stretch factors; eigenvectors are the axes.

→ Used in Chapter 10: the eigenvalues of a network’s weight and Jacobian matrices govern whether repeated multiplication makes signals explode (λ > 1) or vanish (λ < 1) as they pass through many layers.

### Symmetric matrices: the spectral theorem

A matrix is symmetric if 𝐀ᵀ = 𝐀 (mirror-image across the diagonal), as our example is. Symmetric matrices are the friendliest in all of applied mathematics, because of the spectral theorem:

Spectral theorem (stated). Every real symmetric n × n matrix has n real eigenvalues and a set of n mutually orthogonal eigenvectors. Chosen to be unit length, these eigenvectors form an orthonormal basis for ℝⁿ.

Two guarantees matter. First, the eigenvalues are guaranteed real — no imaginary numbers sneak in (a general non-symmetric matrix, like a pure rotation, can have complex eigenvalues). Second, the eigenvectors are guaranteed orthogonal, so they define a clean, right-angled coordinate system. This is exactly why symmetric matrices — covariance matrices, correlation matrices, Hessians, graph Laplacians — sit at the heart of so many methods: they always come with a set of perpendicular natural axes.

### Quadratic forms and positive (semi)definiteness

Attach a symmetric matrix 𝐀 to a vector 𝐱 through the quadratic form

```
q(𝐱) = 𝐱ᵀ𝐀𝐱 (a single number).
```

For our example, with 𝐱 = [x₁, x₂],

```
𝐱ᵀ𝐀𝐱 = 2x₁² + 2x₁x₂ + 2x₂².
```

This is the multivariable analogue of “a·x²”: a bowl-shaped (or saddle-shaped) surface over the plane. Its shape is decided entirely by the eigenvalues of 𝐀. We classify:

𝐀 is positive definite if 𝐱ᵀ𝐀𝐱 > 0 for every 𝐱 ≠ 0 — equivalently, all eigenvalues are > 0. The surface is a genuine upward bowl.

positive semidefinite if 𝐱ᵀ𝐀𝐱 ≥ 0 (all eigenvalues ≥ 0) — a bowl that may be flat along some directions.

indefinite if some eigenvalues are positive and some negative — a saddle.

Our 𝐀 has eigenvalues 3 and 1, both positive, so it is positive definite. We can confirm it without eigenvalues by completing the square:

```
2x₁² + 2x₁x₂ + 2x₂² = 2(x₁ + ½x₂)² + (3/2)x₂²,
```

a sum of two non-negative terms that is zero only when x₁ = x₂ = 0. For a 2 × 2 symmetric matrix [[a, b], [b, c]] there is an even quicker test: it is positive definite exactly when a > 0 and det = ac − b² > 0. Here a = 2 > 0 and det = 4 − 1 = 3 > 0 ✓.

This is the same second-derivative test you met in §0.9. There, the Hessian — the matrix of second partial derivatives — decided whether a critical point was a minimum, maximum, or saddle. That decision is the definiteness of the Hessian: positive definite ⇒ local minimum (upward bowl), negative definite ⇒ local maximum, indefinite ⇒ saddle. And a function is convex precisely when its Hessian is positive semidefinite everywhere. Eigenvalues turn the vague word “curves upward” into an exact, checkable condition — the bridge we will cross again in §0.14.

→ Used in Chapter 8 and Chapter 14: positive-definiteness of 𝐗ᵀ𝐗 is what guarantees the least-squares problem has a unique solution and that its loss surface is a single convex bowl.

### Diagonalization: 𝐀 = 𝐐Λ𝐐⁻¹

Collect the eigenvectors as the columns of a matrix 𝐐 and the eigenvalues along the diagonal of a matrix Λ (capital lambda). Then the eigenvalue equation for all directions at once reads 𝐀𝐐 = 𝐐Λ, which rearranges to the diagonalization

```
𝐀 = 𝐐 Λ 𝐐⁻¹.
```

Read right to left, this is a story in three acts: 𝐐⁻¹ rewrites any vector in the eigenvector coordinate system; Λ simply scales each of those coordinates by its eigenvalue; 𝐐 translates back to the original coordinates. A messy matrix becomes “scale along the natural axes.”

For a symmetric matrix the eigenvectors are orthonormal, so 𝐐 is an orthogonal matrix and its inverse is just its transpose (𝐐⁻¹ = 𝐐ᵀ). Using the normalized eigenvectors of our example,

```
1 [ 1 1 ] [ 3 0 ]
𝐐 = ─── [ 1 −1 ] Λ = [ 0 1 ] and 𝐀 = 𝐐Λ𝐐ᵀ.
 √2
```

This special symmetric case, 𝐀 = 𝐐Λ𝐐ᵀ, is the spectral decomposition. One immediate bonus: powers become trivial, since 𝐀ᵏ = 𝐐Λᵏ𝐐⁻¹ — raise each eigenvalue to the k, leave the axes alone. Repeated application of a transformation (a Markov chain step, a layer of a network) is governed by the eigenvalues raised to a power.

### The Singular Value Decomposition (SVD)

Eigen-decomposition needs a square matrix, and even then a non-symmetric one may misbehave. Real data matrices are rectangular — say, patients × features. The singular value decomposition extends the whole idea to any m × n matrix 𝐀:

```
𝐀 = 𝐔 Σ 𝐕ᵀ.
```

The pieces:

𝐕 (n × n, orthonormal columns 𝐯₁, 𝐯₂, …) — the right singular vectors: a set of perpendicular input directions.

𝐔 (m × m, orthonormal columns 𝐮₁, 𝐮₂, …) — the left singular vectors: the perpendicular output directions those inputs map to.

Σ (m × n, diagonal, entries σ₁ ≥ σ₂ ≥ … ≥ 0) — the singular values: the non-negative stretch factors.

The geometry mirrors eigenvectors exactly: 𝐀 takes the orthonormal input direction 𝐯ᵢ, stretches it by σᵢ, and lays it down along the orthonormal output direction 𝐮ᵢ, i.e. 𝐀𝐯ᵢ = σᵢ𝐮ᵢ. Unlike eigenvectors, the input and output frames are allowed to differ (𝐔 ≠ 𝐕), which is what lets the matrix be rectangular.

Where do the pieces come from? Form the symmetric, positive-semidefinite matrix 𝐀ᵀ𝐀. Its eigenvectors are the columns of 𝐕, and its eigenvalues are the squares of the singular values, so σᵢ = √(eigenvalue of 𝐀ᵀ𝐀). (Likewise 𝐀𝐀ᵀ supplies 𝐔.) The SVD is thus “the eigen-decomposition of 𝐀ᵀ𝐀 and 𝐀𝐀ᵀ, stitched together.” In the special case where 𝐀 is itself symmetric and positive definite, the SVD and the eigen-decomposition coincide: 𝐔 = 𝐕 = 𝐐 and σᵢ = λᵢ. Our example matrix is exactly that case, with singular values σ₁ = 3, σ₂ = 1 and singular vectors [1, 1]/√2 and [1, −1]/√2.

### Low-rank approximation: keeping the top k

Write the SVD as a sum of rank-1 layers, ordered from most to least important:

```
𝐀 = σ₁ 𝐮₁𝐯₁ᵀ + σ₂ 𝐮₂𝐯₂ᵀ + …
```

Each term σᵢ𝐮ᵢ𝐯ᵢᵀ is a whole matrix (an outer product) weighted by its singular value. Because σ₁ ≥ σ₂ ≥ …, the first few terms carry most of the “energy.” Keeping only the top k gives the best possible rank-k approximation of 𝐀 — no other rank-k matrix comes closer (a fact called the Eckart–Young theorem). This is data compression with a guarantee.

Take our 𝐀 = [[2, 1], [1, 2]] and keep only the top term (k = 1). With 𝐮₁ = 𝐯₁ = [1, 1]/√2:

```
σ₁ [ 1 1 ] [ 1.5 1.5 ]
𝐀₁ = σ₁𝐮₁𝐯₁ᵀ = ─── · [ 1 1 ] · … = [ 1.5 1.5 ].
 2
```

The dropped part is 𝐀 − 𝐀₁ = [[0.5, −0.5], [−0.5, 0.5]], whose overall size (its Frobenius norm, √(0.5² + 0.5² + 0.5² + 0.5²) = 1) is exactly the discarded singular value σ₂ = 1. That is the general rule: throwing away small singular values costs you only as much error as those singular values are large. For a big matrix whose singular values decay quickly, a handful of terms reproduce it almost perfectly while storing a tiny fraction of the numbers.

### From SVD to PCA

Here is the connection that makes §0.12 worth the effort. In principal component analysis you have a cloud of data points and you compute their covariance matrix — a symmetric, positive-semidefinite matrix whose (i, j) entry is how feature i and feature j vary together. Its eigenvectors are the principal components: the orthogonal directions along which the data varies most, and its eigenvalues are the variances captured along each direction.

Suppose two standardized features have covariance matrix [[2, 1], [1, 2]] — our example again. The top principal component is [1, 1]/√2 (the features rise and fall together) with variance 3; the second is [1, −1]/√2 with variance 1. The total variance is the trace, 2 + 2 = 4, so the first component alone captures 3/4 = 75% of the variation. Projecting every data point onto that single axis compresses two features into one while keeping three-quarters of the signal. PCA is nothing more than “take the eigenvectors (equivalently, the singular vectors of the centered data matrix) and keep the top few.”

→ Used in Chapter 7: PCA, SVD, and low-rank approximation are the workhorses of dimensionality reduction, denoising, and compression — the single most important application of this section. → Used in Chapter 12: attention layers and modern model-compression schemes exploit the fact that large matrices are often close to low-rank, so a few singular directions carry most of the meaning.

## 0.13 Foundations of Probability

Probability is the mathematics of uncertainty. As a clinician you already reason probabilistically every day — a positive test raises your suspicion, a negative one lowers it — but you do so with intuition. This section gives you the formal machinery behind that intuition. We build only the foundations you need to reach Chapter 3, which develops statistics, estimation, and inference in depth. Here we cover the grammar: outcomes, events, the rules that combine them, Bayes’ theorem, and the two objects that dominate machine learning — random variables and their expectations.

![0.12: Bayes' theorem in natural frequencies: a 90%-accurate test still yields a low positive predictive value when disease is ](../assets/figures/ml_concept_0.12_59812f8d.png)

*Figure 0.12 — original teaching graphic.*

### Sample space, events, and the three axioms

Intuition. Before we can measure the chance of something, we must list everything that could happen. Rolling one die could produce a 1, 2, 3, 4, 5, or 6. That exhaustive list is the sample space.

Definition. The sample space Ω is the set of all possible outcomes of an experiment. An event is any subset of Ω — a collection of outcomes we care about. For the die, Ω = {1, 2, 3, 4, 5, 6}, and the event “roll is even” is the subset E = {2, 4, 6}.

A probability P assigns to every event a number obeying three rules, the Kolmogorov axioms:

Non-negativity: P(A) ≥ 0 for every event A. Chances are never negative.

Normalization: P(Ω) = 1. Something in the list must happen.

Additivity: if events A₁, A₂, … are mutually exclusive (no two can happen together), then P(A₁ ∪ A₂ ∪ …) = ∑ᵢ P(Aᵢ). Chances of non-overlapping events add.

Everything else in probability is a consequence of these three lines.

Equally-likely outcomes. When every outcome in a finite Ω is equally likely, probability reduces to counting:

P(A) = (number of outcomes in A) / (number of outcomes in Ω) = |A| / |Ω|.

Worked example. For the fair die, P(even) = |{2, 4, 6}| / |{1, 2, 3, 4, 5, 6}| = 3 / 6 = 0.5.

### Counting, complements, and the addition rule

Two consequences of the axioms are used constantly.

Complement rule. The complement Aᶜ is “A does not happen.” Since A and Aᶜ are mutually exclusive and together fill Ω, P(A) + P(Aᶜ) = 1, so

P(Aᶜ) = 1 − P(A).

This is the “at least one” shortcut: the chance of at least one event is often easiest as 1 minus the chance of none.

Addition rule. For any two events, overlapping or not,

P(A ∪ B) = P(A) + P(B) − P(A ∩ B).

We subtract the intersection because outcomes in both A and B were counted twice.

Worked example. Draw one card from 52. Let H = “heart” (13 cards) and F = “face card: J, Q, K” (12 cards). There are 3 heart face cards, so P(H ∩ F) = 3/52. Then

P(H ∪ F) = 13/52 + 12/52 − 3/52 = 22/52 = 11/26 ≈ 0.423.

### Conditional probability, independence, and the multiplication rule

Intuition. New information reshapes the sample space. Once you know event B occurred, only outcomes inside B remain possible, and we rescale probabilities to that smaller world.

Definition. The conditional probability of A given B, for P(B) > 0, is

P(A | B) = P(A ∩ B) / P(B).

Rearranging gives the multiplication rule:

P(A ∩ B) = P(A | B) · P(B) = P(B | A) · P(A).

Independence. Events A and B are independent when knowing one tells you nothing about the other: P(A | B) = P(A). Equivalently,

P(A ∩ B) = P(A) · P(B).

Worked example. Two fair coin flips are independent, so P(both heads) = P(H₁) · P(H₂) = (1/2)(1/2) = 1/4. Independence lets us multiply; it is the assumption behind the “naïve” in naïve Bayes classifiers. → Used in Chapter 9: independence assumptions in classifiers.

### The law of total probability and Bayes’ theorem

Law of total probability. Suppose events B₁, B₂, …, Bₙ partition Ω (exactly one of them happens). Then any event A can be reached through one of the pieces:

P(A) = ∑ᵢ P(A | Bᵢ) · P(Bᵢ).

You are averaging the conditional chances, weighted by how likely each piece is. This is the denominator you need for Bayes.

Bayes’ theorem reverses a conditional. From the multiplication rule, P(A ∩ B) can be written two ways, giving

P(A | B) = P(B | A) · P(A) / P(B).

Read it as: posterior = likelihood × prior / evidence. It updates a prior belief P(A) into a posterior P(A | B) after observing B.

Worked clinical example (this is exactly positive predictive value). A test has sensitivity P(+ | D) = 0.99 and specificity P(− | Dᶜ) = 0.95, so its false-positive rate is P(+ | Dᶜ) = 0.05. Disease prevalence is P(D) = 0.01, hence P(Dᶜ) = 0.99. A patient tests positive. What is P(D | +)?

First the evidence, by the law of total probability:

P(+) = P(+ | D)·P(D) + P(+ | Dᶜ)·P(Dᶜ) = (0.99)(0.01) + (0.05)(0.99) = 0.0099 + 0.0495 = 0.0594.

Then Bayes:

P(D | +) = P(+ | D)·P(D) / P(+) = 0.0099 / 0.0594 ≈ 0.167.

Even with a 99%-sensitive test, a positive result means only a 16.7% chance of disease — because the rare disease is swamped by false positives from the healthy 99%. This base-rate effect is the single most important lesson of the section. → Used in Chapter 9: Bayes-optimal classification; Chapter 16: class imbalance and rare events.

### Random variables: PMF, PDF, CDF

Intuition. We rarely care about raw outcomes; we care about numbers attached to them — a count, a lab value, a loss. A random variable is that number-valued summary.

Definition. A random variable X is a function from the sample space to the real numbers, X: Ω → ℝ. It is discrete if it takes countably many values (a coin count, number of seizures), and continuous if it takes values across an interval (blood pressure, reaction time).

Three functions describe a random variable:

PMF (discrete) — the probability mass function p(x) = P(X = x). It is a real probability at each value, and ∑ₓ p(x) = 1.

PDF (continuous) — the probability density function f(x) ≥ 0 with ∫ f(x) dx = 1. Here f(x) is not a probability; probability is area: P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx. For a continuous X, any single point has P(X = x) = 0.

CDF (both) — the cumulative distribution function F(x) = P(X ≤ x), rising from 0 to 1. It relates to the others by accumulation: discrete, F(x) = ∑{k ≤ x} p(k); continuous, F(x) = ∫{−∞}^{x} f(t) dt, and differentiating recovers the density, f(x) = F′(x).

So PMF/PDF and CDF are two views of the same information: sum or integrate to go “up” to the CDF, difference or differentiate to come back “down.”

### Expectation and variance

Expectation E[X] is the long-run average — the center of mass of the distribution:

discrete: E[X] = ∑ₓ x · p(x) continuous: E[X] = ∫ x · f(x) dx.

Variance measures spread around that center:

Var(X) = E[(X − μ)²] = E[X²] − (E[X])², where μ = E[X].

The right-hand shortcut (“mean of the square minus square of the mean”) is the one we usually compute. The standard deviation σ = √Var(X) restores the original units.

Key properties. Let a, b be constants.

Linearity of expectation: E[aX + b] = a·E[X] + b, and crucially E[X + Y] = E[X] + E[Y] for any X and Y, even dependent ones. This near-magical fact underlies expected-loss and bias–variance decompositions.

Scaling of variance: Var(aX + b) = a²·Var(X). Adding a constant shifts but does not spread; the constant b vanishes.

Sums: Var(X + Y) = Var(X) + Var(Y) only when X and Y are independent (or uncorrelated).

Worked example. For the fair die, E[X] = (1 + 2 + 3 + 4 + 5 + 6)/6 = 21/6 = 3.5. Then E[X²] = (1 + 4 + 9 + 16 + 25 + 36)/6 = 91/6 ≈ 15.167, so

Var(X) = 91/6 − (3.5)² = 15.167 − 12.25 = 2.917 = 35/12, σ ≈ 1.708.

By linearity, the expected total of two dice is E[X + Y] = 3.5 + 3.5 = 7 — no need to work out the distribution of the sum. → Used in Chapter 11: expectations define the objectives generative models optimize.

### Three distributions at a glance

You will meet these constantly; the full catalog waits in Chapter 3.

Bernoulli(p): one yes/no trial, X ∈ {0, 1}, P(X = 1) = p. E[X] = p, Var(X) = p(1 − p).

Binomial(n, p): number of successes in n independent Bernoulli trials. E[X] = np, Var(X) = np(1 − p).

Normal(μ, σ²): the continuous bell curve, symmetric about μ with spread σ. It is the workhorse of statistics and the default noise model in ML.

→ Used in Chapter 3 (statistics and inference), Chapter 9 (classification), Chapter 11 (generative models), Chapter 16 (data challenges).

## 0.14 Optimization: Objectives, Convexity, and Gradient Descent

### Objectives and loss functions

Nearly every machine-learning method reduces to the same sentence: choose the parameters that make some number as small as possible. That number is the objective function (when we minimize it we call it a loss or cost function). It measures how badly a model with parameters 𝐱 fits the data — mean squared error for regression, cross-entropy for classification, negative reward in reinforcement learning. Learning is optimization: turn the knobs 𝐱 until the loss bottoms out.

We write the goal as

```
𝐱* = argmin f(𝐱),
```

read “the argument that minimizes f” — not the smallest value of f, but the input 𝐱 that achieves it. Maximizing is the same problem in disguise: maximizing f is minimizing −f, so we can speak only of minimization without losing anything (maximizing a reward = minimizing its negative).

→ Used in Chapter 13: reinforcement learning maximizes expected reward, which the algorithms handle as minimizing its negative — the same argmin machinery.

### Minima, maxima, and saddle points

From §0.9 you know the terrain. A global minimum is the lowest point of the whole surface — the answer we truly want. A local minimum is merely lower than its immediate neighbours: the bottom of a side valley from which every small step leads uphill, even though a deeper valley exists elsewhere. A saddle point is flat in every direction yet is a minimum along some axes and a maximum along others — like a mountain pass, low between two peaks but high across the ridge.

All three share one feature: the ground is level, so the gradient is zero. Distinguishing them is exactly the Hessian (second-derivative) test from §0.9, now readable through §0.12: at a level point, a positive-definite Hessian (all eigenvalues > 0) means a local minimum, negative-definite means a maximum, and indefinite (mixed-sign eigenvalues) means a saddle. Saddles, not bad local minima, turn out to be the main obstacle in the vast parameter spaces of neural networks.

### Convex sets and convex functions

Some loss surfaces have no side valleys at all — a single bowl with one bottom. These are the convex functions, and they are the happy case.

A set is convex if, for any two points in it, the straight segment joining them stays entirely inside the set (a disk is convex; a crescent is not). A function f is convex if its graph never bulges above its own chords: for any two points 𝐱, 𝐲 and any blend fraction t between 0 and 1,

```
f( t𝐱 + (1−t)𝐲 ) ≤ t f(𝐱) + (1−t) f(𝐲).
```

The left side is the function’s value at a point between 𝐱 and 𝐲; the right side is the straight-line interpolation of the two heights. “Curve sits on or below the connecting line” is the whole idea. For a twice-differentiable function this is equivalent — tying §0.9 to §0.12 — to the Hessian being positive semidefinite everywhere (in one variable, f″(x) ≥ 0, the test you already saw).

Why do we care so much? Convexity guarantees no bad local minima. In a convex function every local minimum is automatically the global minimum, and there are no saddle points to stall on. So if we find any level-ground point, we are done — the optimizer cannot get trapped. Least-squares regression, ridge regression, logistic regression, and support-vector machines are all convex, which is why they are so reliable. Neural networks are not convex; training them is the art of doing well anyway.

→ Used in Chapter 8: linear and logistic regression are convex, so their training is guaranteed to reach the one true optimum.

### Stationarity: ∇f = 0

The multivariable version of “set the derivative to zero” is the stationarity condition

```
∇f(𝐱) = 0,
```

meaning every partial derivative vanishes at once — the ground is level in all directions simultaneously (§0.9). This is a necessary condition for a minimum: any minimum is a stationary point. For a convex function it is also sufficient — a stationary point is guaranteed to be the global minimum. Sometimes we can solve ∇f = 0 by hand (that is how linear regression’s “normal equations” arise in Chapter 8). Usually the equations are too tangled for that, and we descend toward the solution instead.

### Gradient descent

Recall from §0.9 that the gradient ∇f points in the direction of steepest increase. To go downhill, step the opposite way. That single instruction is gradient descent:

```
𝐱 ← 𝐱 − η ∇f(𝐱).
```

Starting from a guess, repeatedly nudge 𝐱 a little way down the local slope; the moves shrink as the ground flattens near a minimum, where ∇f → 0 and the updates stop. The knob η (eta) is the learning rate — how big a step to take:

η too small: each step barely moves; convergence is correct but painfully slow.

η too large: you overshoot the valley floor, landing higher on the far wall; steps can oscillate and even diverge, flinging you outward forever.

Picking η is the central practical skill of training, and the rest of this section is really about what makes it easy or hard.

### Worked example: descending a quadratic bowl

Minimize the two-variable bowl

```
f(𝐱) = x₁² + 4x₂², with gradient ∇f = [2x₁, 8x₂].
```

The minimum is obviously at the origin (0, 0), where f = 0; let us make gradient descent discover it. Start at 𝐱₀ = [1, 1] with learning rate η = 0.1. Each step applies x₁ ← x₁ − 0.1(2x₁) = 0.8 x₁ and x₂ ← x₂ − 0.1(8x₂) = 0.2 x₂:

| step k | 𝐱ₖ | ∇f(𝐱ₖ) | f(𝐱ₖ) |
| --- | --- | --- | --- |
| 0 | [1.000, 1.000] | [2.0, 8.0] | 5.000 |
| 1 | [0.800, 0.200] | [1.6, 1.6] | 0.800 |
| 2 | [0.640, 0.040] | [1.28, 0.32] | 0.416 |
| 3 | [0.512, 0.008] | [1.02, 0.06] | 0.262 |

The loss falls 5 → 0.8 → 0.416 → 0.262 and keeps shrinking. In closed form the coordinates are x₁ = 0.8ᵏ and x₂ = 0.2ᵏ, both marching to 0 — the true minimum — as k grows. Notice the second coordinate collapses far faster (factor 0.2 per step) than the first (factor 0.8). The two directions converge at different speeds because the bowl is steeper along x₂ than along x₁, and that imbalance is the theme of the next two subsections. In pseudocode:

```
x ← [1, 1] # starting guess
η ← 0.1 # learning rate
repeat until ∇f is tiny:
 g ← [2·x₁, 8·x₂] # the gradient at the current point
 x ← x − η·g # one step downhill
```

### Stochastic vs. batch (a preview)

In real training f is an average of the loss over many data points, so the exact gradient sums a contribution from every example — one batch (full) gradient step can mean touching millions of rows. Stochastic gradient descent (SGD) instead estimates the gradient from one example, or a small mini-batch, at a time. Each step is noisier but vastly cheaper, and the noise even helps jiggle the optimizer out of shallow traps. Almost all modern training is mini-batch SGD.

→ Used in Chapter 8 and Chapter 10: SGD and its adaptive descendants (momentum, RMSProp, Adam) are how regression models and deep networks are actually trained at scale.

### Ill-conditioning and the condition number

Why did our two coordinates converge at such different rates? Because the bowl is elongated. The curvature in each direction is set by the Hessian, here the constant matrix

```
[ 2 0 ]
𝐇 = [ 0 8 ],
```

with eigenvalues 2 and 8 (§0.12). Their ratio is the condition number

```
κ = λ_max / λ_min = 8 / 2 = 4.
```

When κ = 1 the bowl is a perfectly round basin and gradient descent heads straight to the bottom. When κ is large the bowl is a long, narrow valley: a single learning rate cannot suit both directions at once. Stability requires η < 2 / λ_max — here η < 2/8 = 0.25 — because in the steep direction the update multiplies the error by (1 − η·8), which blows up once |1 − 8η| > 1. (Try η = 0.3: the x₂ error is multiplied by 1 − 2.4 = −1.4 each step and explodes.) But an η small enough to keep the steep direction stable is too small for the shallow direction, which then crawls. Large κ means slow, zig-zagging descent down the length of the valley.

The cure is feature scaling. If we rescale the second coordinate so both directions have equal curvature — here substituting u = 2x₂ turns f into x₁² + u², a round bowl with κ = 1 — descent converges in almost a single step. This is exactly why we standardize features (subtract the mean, divide by the standard deviation) before training: it reshapes stretched valleys into round bowls that gradient descent handles easily.

→ Used in Chapter 8: feature standardization and well-conditioned design matrices are what make regression optimizers converge quickly and stably.

### Constrained optimization and Lagrange multipliers

Sometimes we must minimize f subject to a constraint g(𝐱) = 0 — stay on a surface while seeking the lowest point on it. At the constrained optimum you cannot improve f without stepping off the constraint. Geometrically that happens exactly when the two gradients are parallel:

```
∇f = λ ∇g,
```

where the scalar λ is the Lagrange multiplier. The intuition: ∇f is the downhill-blocking direction and ∇g is perpendicular to the constraint surface; when they align, every allowed move (along the surface, perpendicular to ∇g) is also perpendicular to ∇f, so no allowed step changes f to first order — you are stuck at the best feasible point.

A quick example: minimize f = x² + y² (squared distance to the origin) subject to x + y = 1. Here ∇f = [2x, 2y] and ∇g = [1, 1], so 2x = λ and 2y = λ force x = y; the constraint x + y = 1 then gives x = y = ½, with f = ½. The closest point on the line to the origin is its foot of perpendicular — precisely what the geometry predicts.

Regularization as a penalty. A softer cousin of a hard constraint is to add a penalty to the objective — for instance minimizing f(𝐱) + λ‖𝐱‖² instead of f alone. The extra term discourages large parameters, shrinking them toward zero; this is ridge regression (an L2 penalty) and, with ‖𝐱‖₁, lasso. Regularization both curbs overfitting and improves conditioning by adding curvature (it makes the Hessian “more positive definite”), which is why penalized problems are often easier to optimize than their raw counterparts.

→ Used in Chapter 8: ridge and lasso are penalized least-squares; the multiplier λ trades data-fit against model simplicity. → Used in Chapter 13: constrained and penalized objectives (trust regions, entropy bonuses) keep reinforcement-learning updates stable.

### Practice — 0.12, 0.14

Work these with pencil and paper; answers follow.

1. Eigenvalues and eigenvectors. For 𝐁 = [[4, 1], [2, 3]], find both eigenvalues and an eigenvector for each. (Hint: the characteristic equation is λ² − (trace)λ + det = 0.)

2. Test positive-definiteness. Using the eigenvalue test (or the shortcut “a > 0 and det > 0”), classify 𝐂 = [[3, 2], [2, 3]] and 𝐃 = [[1, 2], [2, 1]] as positive definite, indefinite, or neither.

3. Two gradient-descent steps by hand. For f(𝐱) = x₁² + 3x₂², start at 𝐱₀ = [2, 1] with η = 0.1 and compute 𝐱₁ and 𝐱₂. Is the loss decreasing?

4. Convex or not? Which of these are convex: (a) f(x) = x² + 3; (b) f(x, y) = x² − 4y²; (c) f(x) = eˣ; (d) f(x) = x³ (on all of ℝ)?

5. Low-rank / SVD (conceptual). A 1000 × 500 data matrix has singular values 50, 30, 10, then a long tail of tiny values. (i) What is the best rank-2 approximation made of? (ii) Roughly what fraction of the total “energy” (sum of squared singular values) does it capture? (iii) Why is storing the rank-2 factors far cheaper than the full matrix?

6. Condition number. The Hessian of a loss at its minimum is 𝐇 = [[10, 0], [0, 1]]. (i) Is this a genuine minimum? (ii) What is the condition number? (iii) Which direction forces the smaller learning rate, and what is the stability limit on η?

Answers.

1. Trace = 7, det = (4)(3) − (1)(2) = 10, so λ² − 7λ + 10 = (λ − 5)(λ − 2) = 0 ⇒ λ = 5, 2. For λ = 5: (𝐁 − 5𝐈) = [[−1, 1], [2, −2]] gives −v₁ + v₂ = 0, so 𝐯 = [1, 1]. For λ = 2: (𝐁 − 2𝐈) = [[2, 1], [2, 1]] gives 2v₁ + v₂ = 0, so 𝐯 = [1, −2]. Check: 𝐁[1, 1] = [5, 5] ✓ and 𝐁[1, −2] = [2, −4] ✓.

2. 𝐂: a = 3 > 0 and det = 9 − 4 = 5 > 0 ⇒ positive definite (eigenvalues 5 and 1). 𝐃: det = 1 − 4 = −3 < 0 ⇒ indefinite (eigenvalues 3 and −1, mixed signs) — a saddle, not positive definite.

3. ∇f = [2x₁, 6x₂]. Step 1: 𝐱₁ = [2, 1] − 0.1·[4, 6] = [1.6, 0.4]. Step 2: ∇f(𝐱₁) = [3.2, 2.4], so 𝐱₂ = [1.6, 0.4] − 0.1·[3.2, 2.4] = [1.28, 0.16]. Loss: f(𝐱₀) = 4 + 3 = 7, f(𝐱₁) = 2.56 + 0.48 = 3.04, f(𝐱₂) = 1.6384 + 0.0768 = 1.7152 — decreasing ✓.

4. (a) Convex (f″ = 2 > 0). (b) Not convex — Hessian [[2, 0], [0, −8]] is indefinite (a saddle). (c) Convex (f″ = eˣ > 0 everywhere). (d) Not convex on all of ℝ (f″ = 6x changes sign; it curves down for x < 0). Convex: (a) and (c).

5. (i) The best rank-2 approximation is σ₁𝐮₁𝐯₁ᵀ + σ₂𝐮₂𝐯₂ᵀ — the top two singular values with their singular-vector pairs. (ii) Captured energy = (50² + 30²)/(50² + 30² + 10² + tail) ≈ 3400/3500 ≈ 97%. (iii) The rank-2 factors need only 2 columns of 𝐔 (2 × 1000), 2 columns of 𝐕 (2 × 500), and 2 singular values — about 3000 numbers versus 1000 × 500 = 500,000 for the full matrix, roughly a 160-fold saving with almost no loss.

6. (i) Eigenvalues 10 and 1 are both > 0, so 𝐇 is positive definite ⇒ yes, a genuine local minimum. (ii) κ = 10/1 = 10 (an ill-conditioned, elongated bowl). (iii) The steep x₁ direction (curvature 10) forces the smaller learning rate; stability requires η < 2/λ_max = 2/10 = 0.2.

## 0.15 Discrete Mathematics, Graphs, and Algorithmic Complexity

Continuous mathematics (calculus, linear algebra) describes smooth quantities. Discrete mathematics describes countable, separated things: sets, relationships, networks, and the step-by-step cost of algorithms. This section equips you with the vocabulary of structure and the arithmetic of scale.

![0.14: Asymptotic growth of common complexity classes; the gap between O(n log n) and O(n²) or O(2ⁿ) decides what is computable](../assets/figures/ml_concept_0.14_93423239.png)

*Figure 0.14 — original teaching graphic.*

### Sets, relations, and functions

A set is an unordered collection of distinct elements; we write x ∈ A (“x is in A”), A ⊆ B (“A is contained in B”), and combine sets with union ∪, intersection ∩, and difference \. The Cartesian product A × B is the set of all ordered pairs (a, b) with a ∈ A and b ∈ B. A relation is simply a subset of A × B — a rule picking out which pairs are “connected” (patient–diagnosis, word–document).

A function f: A → B is a special relation that assigns to each element of A exactly one element of B. Three shapes matter:

Injective (one-to-one): different inputs give different outputs; nothing collides.

Surjective (onto): every element of B is hit by some input.

Bijective: both at once — a perfect pairing. Bijections are exactly the invertible functions, and they are how we say two sets “have the same size.”

Intuitively: injective wastes no output on two inputs, surjective leaves no output unused, bijective is a flawless dictionary between A and B.

### A little counting

Combinatorics answers “how many ways?”

Product rule: k independent choices followed by m choices give k · m combinations.

Permutations: n distinct items can be ordered in n! = n·(n−1)···1 ways.

Combinations: the number of ways to choose k items from n, order ignored, is

C(n, k) = n! / (k! (n − k)!).

Worked example. Choosing 2 features from 5: C(5, 2) = 5! / (2!·3!) = 120 / (2·6) = 120 / 12 = 10. Counting like this tells you how the size of a search space explodes as inputs grow — the seed of computational cost. → Used in Chapter 5: counting itemsets in pattern mining.

### Graphs: the language of connections

Intuition. Whenever objects relate to one another — neurons, brain regions, patients in a contact network, web pages — a graph captures the structure.

Definition. A graph G = (V, E) is a set of vertices (nodes) V and a set of edges E joining pairs of them. Variations:

Undirected: an edge {u, v} is a symmetric link. Directed: an edge (u, v) is an arrow from u to v.

Weighted: each edge carries a number (distance, cost, connection strength).

Degree: the number of edges touching a vertex; directed graphs split this into in-degree and out-degree.

Path: a sequence of vertices each joined to the next by an edge. A cycle is a path returning to its start.

Connectivity: a graph is connected if some path links every pair of vertices.

The adjacency matrix. A graph on n vertices can be stored as an n × n matrix A where A[i][j] = 1 if an edge joins i to j (or the edge’s weight), and 0 otherwise. Undirected graphs give a symmetric A. This is the bridge from graphs back to the linear algebra of §0.11: matrix powers count walks — the (i, j) entry of Aᵏ is the number of length-k walks from i to j.

Worked example. A triangle on vertices {1, 2, 3} with edges {1–2, 2–3, 1–3}: every vertex has degree 2, and

```
1 2 3
 1 0 1 1
 2 1 0 1
 3 1 1 0
```

The degrees sum to 2 + 2 + 2 = 6 = 2·|E| = 2·3. This is the handshake lemma: every edge contributes 2 to the total degree. → Used in Chapter 15: graph algorithms and representation learning.

### Growth of functions: Big-O, Θ, Ω

Intuition. Two algorithms may both “work,” yet one finishes in a blink on a million records while the other never finishes at all. What matters is not the exact operation count but how that count grows as the input size n grows. Asymptotic notation captures growth while ignoring constant factors and small-n noise.

Definition. We say f(n) = O(g(n)) — “f is order g” — if there are constants c > 0 and n₀ such that f(n) ≤ c·g(n) for all n ≥ n₀. Big-O is an upper bound: it promises f grows no faster than g. Two companions refine it:

Ω(g(n)) is a lower bound: f grows at least as fast as g.

Θ(g(n)) is a tight bound: f grows exactly like g (both O and Ω hold).

We keep only the dominant term and drop constants: 3n² + 50n + 200 is Θ(n²), because for large n the n² term rules.

### The complexity zoo

A handful of growth classes cover most algorithms. Here is what each feels like as n scales:

| Class | Name | Feel at scale | Example |
| --- | --- | --- | --- |
| O(1) | constant | instant, size-independent | array lookup |
| O(log n) | logarithmic | barely grows; doubling n adds one step | binary search |
| O(n) | linear | proportional; double n, double work | one pass over data |
| O(n log n) | linearithmic | slightly worse than linear | efficient sorting |
| O(n²) | quadratic | every pair; painful past ~10⁴ | all-pairs comparison |
| O(2ⁿ) | exponential | hopeless past ~40 | enumerate all subsets |

To make it concrete, approximate operation counts:

| n | log₂ n | n | n log₂ n | n² | 2ⁿ |
| --- | --- | --- | --- | --- | --- |
| 10 | ≈ 3 | 10 | ≈ 33 | 100 | 1,024 |
| 100 | ≈ 7 | 100 | ≈ 664 | 10⁴ | ≈ 1.3 × 10³⁰ |
| 1,000,000 | ≈ 20 | 10⁶ | ≈ 2 × 10⁷ | 10¹² | astronomically large |

### Worked example: analyzing a nested loop

Consider counting a constant-cost operation:

```
count = 0
for i in 1..n:
 for j in 1..n:
 count = count + 1 # constant work
```

The inner loop runs n times for each of the n outer passes, so the body executes n · n = n² times → O(n²).

Now a common variant that only looks at distinct pairs:

```
for i in 1..n:
 for j in i+1..n:
 compare(i, j) # constant work
```

The body runs (n−1) + (n−2) + … + 1 + 0 = n(n−1)/2 ≈ n²/2 times. The constant ½ drops out, so this is still O(n²) — half the work, but the same class, and it will scale just as badly.

### Why complexity governs algorithm choice

Suppose a machine does 10⁹ simple operations per second and n = 10⁶. An O(n log n) method needs ≈ 2 × 10⁷ operations — about 0.02 seconds. An O(n²) method on the same data needs ≈ 10¹² operations — about 1,000 seconds, roughly 17 minutes. Same problem, same computer: the algorithm’s class decided whether it took a blink or a coffee break, and an O(2ⁿ) approach would not finish before the universe cooled. This is why we analyze complexity before coding: at scale, the exponent beats every clever constant-factor trick. → Used in Chapter 1 (algorithm evaluation), Chapter 5 (why exhaustive mining is pruned), Chapter 15 (graph algorithm cost).

## 0.16 Numerical Computation and Practical Pitfalls

Mathematics on paper uses exact real numbers with infinite precision. Computers do not. Every ML system runs on approximate arithmetic, and a surprising number of “mysterious” bugs — silent zeros, NaNs, results that change between runs — are really numerical issues. This section shows how numbers are stored and where they break.

### How computers store real numbers

A computer stores a real number in floating point: a fixed budget of bits split into a sign, a fraction (mantissa), and an exponent, encoding sign × mantissa × 2^exponent — scientific notation in base 2. The standard 64-bit “double” gives about 15–16 significant decimal digits.

Because the budget is finite, most reals cannot be stored exactly. Famously, 0.1 has no finite binary expansion, so

0.1 + 0.2 → 0.30000000000000004, not exactly 0.3.

The gap between 1.0 and the next representable number is machine epsilon, ≈ 2.2 × 10⁻¹⁶ for doubles. Every stored value carries a relative rounding error of about this size. Usually harmless — until errors are amplified.

### Overflow and underflow

The exponent has limits too. A double can represent magnitudes up to ≈ 1.8 × 10³⁰⁸ and down to ≈ 2.2 × 10⁻³⁰⁸ (normalized).

Overflow: a result too large becomes ∞, poisoning everything downstream.

Underflow: a result too small collapses to 0, silently destroying information.

Underflow is the classic trap when multiplying many probabilities: 1,000 factors each around 0.01 give 10⁻²⁰⁰⁰, far below the smallest double, so the product rounds to exactly 0 — and any later logarithm returns −∞.

### Catastrophic cancellation

Intuition. Subtracting two nearly-equal numbers annihilates their shared leading digits and leaves only their uncertain trailing digits — so a tiny input error becomes a huge relative output error.

Worked tiny example. Work with 5 significant digits. Two true quantities, 12345.4 and 12343.6, arrive already rounded to 5 significant figures as 12345 and 12344.

computed difference = 12345 − 12344 = 1 true difference = 12345.4 − 12343.6 = 1.8

The relative error jumps from about 0.003% in each input to |1 − 1.8| / 1.8 = 0.8 / 1.8 ≈ 44% in the answer. The subtraction did not add error; it exposed the error that rounding had hidden. The lesson: avoid subtracting near-equal quantities. This is why the variance shortcut E[X²] − (E[X])² can lose precision when the mean is large relative to the spread.

### Working in log-space: the log-sum-exp trick

The cure for probability underflow is to compute in log-space, turning fragile products into stable sums:

log(p₁ · p₂ ··· pₙ) = ∑ᵢ log pᵢ.

This is exactly why we optimize the log-likelihood rather than the likelihood — the numbers stay in a sane range.

But sometimes we must add probabilities held as logs (for example, to normalize). Computing log ∑ᵢ exp(zᵢ) naïvely can overflow when the zᵢ are large. The log-sum-exp trick factors out the maximum m = maxᵢ zᵢ:

log ∑ᵢ exp(zᵢ) = m + log ∑ᵢ exp(zᵢ − m).

Now the largest term is exp(0) = 1, so nothing overflows and the rest are safely ≤ 1.

Worked example. Let z = [1000, 1001, 1002]. Directly, exp(1000) overflows a double. Take m = 1002:

= 1002 + log( exp(−2) + exp(−1) + exp(0) ) = 1002 + log( 0.1353 + 0.3679 + 1 ) = 1002 + log(1.5032) = 1002 + 0.4076 ≈ 1002.408.

Clipping. Because log(0) = −∞, code that takes log(p) — cross-entropy loss, for instance — first clips p into [ε, 1 − ε] with a tiny ε ≈ 10⁻¹⁵, keeping the logarithm finite.

### Numerical stability and conditioning

An algorithm is numerically stable if small input perturbations cause only small output changes. Some problems, though, are inherently ill-conditioned — they amplify any error, however good the algorithm. Solving a linear system Ax = b when A is nearly singular is the canonical case: the condition number κ(A) = σ₁ / σₙ (largest over smallest singular value, from the SVD of §0.12) measures the amplification. A large κ means a tiny wobble in b can swing x wildly. When an optimizer crawls or diverges, ill-conditioning is a prime suspect. → See §0.14 for how conditioning shapes optimization landscapes.

### Vectorization and the cost of matrix operations

Vectorization means expressing computation as operations on whole arrays rather than element-by-element loops. The math is identical, but array operations dispatch to hardware-optimized libraries (SIMD, BLAS) and run far faster than an interpreted loop. Prefer array expressions.

The cost of these operations follows the complexity classes of §0.15:

dot product of two length-n vectors: O(n)

matrix–vector product (n × n times n): O(n²)

matrix–matrix product (n × n times n × n): O(n³) with the naïve algorithm

That cubic term has teeth: doubling the dimension makes a matrix multiply 2³ = 8× slower. Knowing these costs tells you which reformulation of a model will actually be tractable.

### Determinism, seeds, and reproducibility

Computers cannot make truly random numbers; they run a pseudo-random number generator that produces a fixed, deterministic sequence from a starting seed. Setting the seed makes every random step — shuffles, weight initialization, sampling, train/test splits — repeat exactly, which is essential for debugging and for scientific reproducibility. Two cautions remain: floating-point addition is not associative, so summing numbers in a different order can give slightly different results, and parallel or GPU reductions may reorder those sums between runs. Full determinism therefore needs both a fixed seed and controlled execution order. → Used in Chapter 8 / Chapter 10 (reproducible training) and Chapter 16 (data and evaluation challenges).

### Practice — 0.13, 0.15, 0.16

(Bayes.) A screening test has sensitivity 0.90 and specificity 0.80. In a population with prevalence 0.05, a patient tests positive. Compute P(disease | positive). Comment on why it is low.

(Expectation and variance.) A random variable X is uniform on {1, 2, 3, 4}. (a) Find E[X] and Var(X). (b) If Y is an independent copy of X, use linearity to give E[X + Y], and give Var(X + Y).

(Big-O classification.) Give the tightest Big-O class for each: (a) binary search; (b) comparing every pair of n points; (c) summing a list once; (d) merge sort; (e) enumerating all subsets of n items.

(Analyze the loop.) Classify the running time in terms of n:

```
i = n
while i > 1:
 do_constant_work()
 i = i / 2
```

(Log-sum-exp / stability.) For logits z = [800, 801, 802], compute log ∑ exp(zᵢ) using the log-sum-exp trick. Why does the naïve computation fail on a 64-bit double?

(Cancellation.) Using 5-significant-digit arithmetic, the true values 12345.4 and 12343.6 are stored as 12345 and 12344. Compute the stored difference and the true difference, and report the relative error. What general rule does this illustrate?

Answers. 1. P(+) = 0.90·0.05 + 0.20·0.95 = 0.045 + 0.19 = 0.235; P(D|+) = 0.045/0.235 ≈ 0.191 (19.1%) — false positives from the healthy 95% dominate. 2. E[X] = 2.5; E[X²] = 30/4 = 7.5, so Var(X) = 7.5 − 6.25 = 1.25; E[X + Y] = 5, Var(X + Y) = 2.5. 3. (a) O(log n); (b) O(n²); (c) O(n); (d) O(n log n); (e) O(2ⁿ). 4. O(log n) — i halves each pass. 5. m = 802; 802 + log(exp(−2) + exp(−1) + 1) = 802 + log(1.5032) ≈ 802.408; naïve fails because exp(800) ≈ 10³⁴⁷ overflows the double’s ceiling of ≈ 1.8 × 10³⁰⁸. 6. Stored 12345 − 12344 = 1; true 1.8; relative error |1 − 1.8|/1.8 ≈ 44% — never subtract two nearly-equal numbers.

## 0.17 Notation Glossary and Map to the Book

This section is a reference. The first table lists the symbols used throughout the book with how to read them; the second maps each foundation in this chapter to the chapters that depend on it, so you can see precisely which mathematics a given topic requires.

### Symbol reference

| Symbol | Read as / meaning |
| --- | --- |
| ∈, ∉ | “is an element of” / “is not an element of” |
| ⊆, ∪, ∩, ∅ | subset; union; intersection; empty set |
| ℕ, ℤ, ℚ, ℝ, ℝⁿ | naturals; integers; rationals; reals; n-dimensional real vectors |
| ∀, ∃ | “for all”; “there exists” |
| ⇒, ⇔, iff | implies; if and only if; “if and only if” |
| ≈, ∝, ≜ | approximately equal; proportional to; defined as |
| f: A → B | a function f from set A to set B |
| f∘g, f⁻¹ | composition (“f after g”); inverse function |
| Σᵢ, Πᵢ | sum over index i; product over index i |
| n!, C(n,k) | n factorial; “n choose k” (combinations) |
| e, ln x, logₐ x | Euler’s number ≈ 2.718; natural log; log base a |
| σ(z) | logistic sigmoid 1/(1+e⁻ᶻ) |
| π, θ, φ, ω | pi; angle/parameter; phase; angular frequency |
| lim, ∞ | limit; infinity |
| f′(x), dy/dx | derivative of f; Leibniz derivative notation |
| ∂f/∂xᵢ, ∇f | partial derivative; gradient (vector of partials) |
| ∫, ∫ₐᵇ | integral; definite integral from a to b |
| 𝐉, 𝐇 | Jacobian matrix; Hessian matrix |
| 𝐱, 𝐀 | vector (bold lowercase); matrix (bold uppercase) |
| 𝐀ᵀ, 𝐀⁻¹ | transpose; inverse |
| 𝐈, det(𝐀) | identity matrix; determinant |
| 𝐚·𝐛, ‖𝐱‖ | dot (inner) product; norm (length) |
| ‖𝐱‖₁, ‖𝐱‖₂ | L1 norm (sum of \|components\|); L2 (Euclidean) norm |
| λ, 𝐯 | eigenvalue; eigenvector (𝐀𝐯 = λ𝐯) |
| 𝐔Σ𝐕ᵀ | singular value decomposition |
| P(A), P(A\|B) | probability of A; probability of A given B |
| E[X], Var(X), σ² | expectation; variance; variance |
| ~ | “is distributed as” |
| argmin, argmax | the input that minimizes / maximizes |
| η | learning rate (step size in gradient descent) |
| O(·), Θ(·), Ω(·) | asymptotic upper / tight / lower bounds (Big-O family) |
| ≪, ≫ | much less than; much greater than |

### Where each foundation is used

| Foundation (this chapter) | Chapters that rely on it |
| --- | --- |
| 0.1 Sets, functions, logic | Every chapter — the language of all definitions |
| 0.2 Exponents & logarithms | Ch3 (entropy, log-likelihood), Ch8 (log-odds/logistic), Ch10 (log-loss), Ch16 (log-sum-exp) |
| 0.3 The function zoo (sigmoid, softmax, ReLU) | Ch8 (logistic regression), Ch9 (classification), Ch10 (activations), Ch12 |
| 0.4 Sums, factorials, combinations | Ch3 (binomial, expectation), Ch5 (pattern counting), Ch14 (coding) |
| 0.5 Trigonometry & sinusoids | Ch4–Ch5 (cosine similarity), Ch7 (Fourier, wavelets), Ch12 (positional encoding) |
| 0.6–0.8 Single-variable calculus | Ch3 (densities, expectation as an integral), Ch8 (minimizing loss), Ch10 |
| 0.9 Gradients, Jacobian, Hessian, Taylor | Ch8 (optimization, Newton), Ch10 (backpropagation), Ch13 (policy gradients) |
| 0.10 Vectors, norms, dot product, cosine | Ch4 (distances), Ch6 (feature vectors), Ch7, Ch12 (attention) |
| 0.11 Matrices, inverse, determinant, systems | Ch7 (covariance, PCA), Ch8 (normal equations 𝐗ᵀ𝐗), Ch10 (weights), Ch15 (adjacency) |
| 0.12 Eigenvalues, eigenvectors, SVD | Ch7 (PCA/SVD), Ch12, Ch14 (low-rank, LoRA), Ch15 (spectral methods) |
| 0.13 Probability, Bayes, expectation | Ch3 (statistics), Ch9 (Naive Bayes, calibration), Ch11 (generative), Ch16 |
| 0.14 Optimization, convexity, gradient descent | Ch8 (regression, regularization), Ch10 (SGD/Adam), Ch13 |
| 0.15 Graphs & algorithmic complexity | Ch1 (evaluating algorithms), Ch5 (mining), Ch15 (graph algorithms) |
| 0.16 Numerical computation | Ch8/Ch10 (stable training), Ch16 (reproducibility, train–serve skew) |


![c82 teaching panel 00 (original).](../assets/figures/ml_fig_c82_00.png)
*Figure — Gradient steps on a convex quadratic bowl and its contours—optimization geometry, not a clinical claim. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 00 (original).](../assets/figures/ml_fig_c83_00.png)
*Figure — SVD building blocks for low-rank matrix approximation. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 00 (original).](../assets/figures/ml_fig_c84_00.png)
*Figure — L1 / L2 / L∞ unit balls—geometry behind regularizers. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 00 (original).](../assets/figures/ml_fig_c85_00.png)
*Figure — Activation shapes control gradient flow through deep stacks. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 00 (original).](../assets/figures/ml_fig_c86_00.png)
*Figure — Lp penalty shapes (|x|, x², x⁴) alter solution geometry. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 00 (original).](../assets/figures/ml_fig_c87_00.png)
*Figure — Linear maps stretch and rotate vectors (Av vs v). Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 00 (original).](../assets/figures/ml_fig_c88_00.png)
*Figure — Dot product as signed projection length. Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 00 (original).](../assets/figures/ml_fig_c89_00.png)
*Figure — Condition number: stretched level sets. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 00 (original).](../assets/figures/ml_fig_c90_00.png)
*Figure — Jacobian local linearization of f. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 00 (original).](../assets/figures/ml_fig_c91_00.png)
*Figure — Eigenvectors of a 2x2 stretch. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 00 (original).](../assets/figures/ml_fig_c92_00.png)
*Figure — Matrix rank as dimension of column space. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 00 (original).](../assets/figures/ml_fig_c93_00.png)
*Figure — SVD singular values decay. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 00 (original).](../assets/figures/ml_fig_c94_00.png)
*Figure — Cholesky factor triangle idea. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 00 (original).](../assets/figures/ml_fig_c95_00.png)
*Figure — Pseudoinverse least-squares map. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 00 (original).](../assets/figures/ml_fig_c96_00.png)
*Figure — Householder reflection sketch. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 00 (original).](../assets/figures/ml_fig_c97_00.png)
*Figure — QR decomposition thin/full forms. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 00 (original).](../assets/figures/ml_fig_c98_00.png)
*Figure — Determinant as parallelogram area. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 00 (original).](../assets/figures/ml_fig_c99_00.png)
*Figure — Trace as sum of eigenvalues. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 00 (original).](../assets/figures/ml_fig_c100_00.png)
*Figure — Neumann series inverse sketch. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 00 (original).](../assets/figures/ml_fig_c101_00.png)
*Figure — Cramers rule geometric view. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 00 (original).](../assets/figures/ml_fig_c102_00.png)
*Figure — Moore-Penrose properties strip. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 00 (original).](../assets/figures/ml_fig_c103_00.png)
*Figure — Singular vectors left/right roles. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 00 (original).](../assets/figures/ml_fig_c104_00.png)
*Figure — Orthogonal Procrustes alignment. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 00 (original).](../assets/figures/ml_fig_c105_00.png)
*Figure — Woodbury matrix identity cartoon. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 00 (original).](../assets/figures/ml_fig_c106_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 00 (original).](../assets/figures/ml_fig_c107_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 00 (original).](../assets/figures/ml_fig_c108_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 00 (original).](../assets/figures/ml_fig_c109_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 00 (original).](../assets/figures/ml_fig_c110_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 00 (original).](../assets/figures/ml_fig_c111_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 00 (original).](../assets/figures/ml_fig_c112_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 00 (original).](../assets/figures/ml_fig_c113_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 00 (original).](../assets/figures/ml_fig_c114_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 00 (original).](../assets/figures/ml_fig_c115_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 00 (original).](../assets/figures/ml_fig_c116_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 00 (original).](../assets/figures/ml_fig_c117_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 00 (original).](../assets/figures/ml_fig_c118_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 00 (original).](../assets/figures/ml_fig_c119_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 00 (original).](../assets/figures/ml_fig_c120_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 00 (original).](../assets/figures/ml_fig_c121_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 00 (original).](../assets/figures/ml_fig_c122_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 00 (original).](../assets/figures/ml_fig_c123_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 00 (original).](../assets/figures/ml_fig_c124_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 00 (original).](../assets/figures/ml_fig_c125_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 00 (original).](../assets/figures/ml_fig_c126_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 00 (original).](../assets/figures/ml_fig_c127_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 00 (original).](../assets/figures/ml_fig_c128_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 00 (original).](../assets/figures/ml_fig_c129_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 00 (original).](../assets/figures/ml_fig_c130_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 00 (original).](../assets/figures/ml_fig_c131_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 00 (original).](../assets/figures/ml_fig_c132_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 00 (original).](../assets/figures/ml_fig_c133_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 00 (original).](../assets/figures/ml_fig_c134_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 00 (original).](../assets/figures/ml_fig_c135_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 00 (original).](../assets/figures/ml_fig_c136_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 00 (original).](../assets/figures/ml_fig_c137_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 00 (original).](../assets/figures/ml_fig_c138_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 00 (original).](../assets/figures/ml_fig_c139_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 00 (original).](../assets/figures/ml_fig_c140_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 00 (original).](../assets/figures/ml_fig_c141_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 00 (original).](../assets/figures/ml_fig_c142_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 00 (original).](../assets/figures/ml_fig_c143_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 00 (original).](../assets/figures/ml_fig_c144_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 00 (original).](../assets/figures/ml_fig_c145_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 00 (original).](../assets/figures/ml_fig_c146_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 00 (original).](../assets/figures/ml_fig_c147_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 00 (original).](../assets/figures/ml_fig_c148_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 00 (original).](../assets/figures/ml_fig_c149_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 00 (original).](../assets/figures/ml_fig_c150_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 00 (original).](../assets/figures/ml_fig_c151_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 00 (original).](../assets/figures/ml_fig_c152_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 00 (original).](../assets/figures/ml_fig_c153_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 00 (original).](../assets/figures/ml_fig_c154_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 00 (original).](../assets/figures/ml_fig_c155_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 00 (original).](../assets/figures/ml_fig_c156_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 00 (original).](../assets/figures/ml_fig_c157_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 00 (original).](../assets/figures/ml_fig_c158_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 00 (original).](../assets/figures/ml_fig_c159_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 00 (original).](../assets/figures/ml_fig_c160_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 00 (original).](../assets/figures/ml_fig_c161_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 00 (original).](../assets/figures/ml_fig_c162_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 00 (original).](../assets/figures/ml_fig_c163_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 00 (original).](../assets/figures/ml_fig_c164_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 00 (original).](../assets/figures/ml_fig_c165_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 00 (original).](../assets/figures/ml_fig_c166_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 00 (original).](../assets/figures/ml_fig_c167_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 00 (original).](../assets/figures/ml_fig_c168_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 00 (original).](../assets/figures/ml_fig_c169_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 00 (original).](../assets/figures/ml_fig_c170_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 00 (original).](../assets/figures/ml_fig_c171_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 00 (original).](../assets/figures/ml_fig_c172_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 00 (original).](../assets/figures/ml_fig_c173_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 00 (original).](../assets/figures/ml_fig_c174_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 00 (original).](../assets/figures/ml_fig_c175_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 00 (original).](../assets/figures/ml_fig_c176_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 00 (original).](../assets/figures/ml_fig_c177_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 00 (original).](../assets/figures/ml_fig_c178_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 00 (original).](../assets/figures/ml_fig_c179_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 00 (original).](../assets/figures/ml_fig_c180_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 00 (original).](../assets/figures/ml_fig_c181_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 00 (original).](../assets/figures/ml_fig_c182_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 00 (original).](../assets/figures/ml_fig_c183_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 00 (original).](../assets/figures/ml_fig_c184_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 00 (original).](../assets/figures/ml_fig_c185_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 00 (original).](../assets/figures/ml_fig_c186_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 00 (original).](../assets/figures/ml_fig_c187_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 00 (original).](../assets/figures/ml_fig_c188_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 00 (original).](../assets/figures/ml_fig_c189_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 00 (original).](../assets/figures/ml_fig_c190_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 00 (original).](../assets/figures/ml_fig_c191_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 00 (original).](../assets/figures/ml_fig_c192_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 00 (original).](../assets/figures/ml_fig_c193_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 00 (original).](../assets/figures/ml_fig_c194_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 00 (original).](../assets/figures/ml_fig_c195_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 00 (original).](../assets/figures/ml_fig_c196_00.png)
*Figure — Krylov subspace iteration. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 00 (original).](../assets/figures/ml_fig_c197_00.png)
*Figure — Gershgorin disk theorem. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 00 (original).](../assets/figures/ml_fig_c198_00.png)
*Figure — Power method convergence. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 00 (original).](../assets/figures/ml_fig_c199_00.png)
*Figure — Arnoldi process sketch. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 00 (original).](../assets/figures/ml_fig_c200_00.png)
*Figure — Rayleigh quotient. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 00 (original).](../assets/figures/ml_fig_c201_00.png)
*Figure — Condition number error magnification. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 00 (original).](../assets/figures/ml_fig_c202_00.png)
*Figure — Schatten-p norms of singular values. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 00 (original).](../assets/figures/ml_fig_c203_00.png)
*Figure — Matrix determinant as parallelogram volume. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 00 (original).](../assets/figures/ml_fig_c204_00.png)
*Figure — Moore-Penrose singular reciprocal. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 00 (original).](../assets/figures/ml_fig_c205_00.png)
*Figure — QR factorization geometry. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 00 (original).](../assets/figures/ml_fig_c206_00.png)
*Figure — Cholesky factor lower triangle. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 00 (original).](../assets/figures/ml_fig_c207_00.png)
*Figure — Singular value energy decay spectrum. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 00 (original).](../assets/figures/ml_fig_c208_00.png)
*Figure — Condition number stretch ellipse. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 00 (original).](../assets/figures/ml_fig_c209_00.png)
*Figure — Eigenvalue gap spectral clustering. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 00 (original).](../assets/figures/ml_fig_c210_00.png)
*Figure — Power iteration dominant eigenvector. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 00 (original).](../assets/figures/ml_fig_c211_00.png)
*Figure — Frobenius residual matrix heat. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 00 (original).](../assets/figures/ml_fig_c212_00.png)
*Figure — Trace and determinant of SPD. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 00 (original).](../assets/figures/ml_fig_c213_00.png)
*Figure — SVD truncation tail energy. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 00 (original).](../assets/figures/ml_fig_c214_00.png)
*Figure — Condition number residual growth. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 00 (original).](../assets/figures/ml_fig_c215_00.png)
*Figure — Householder reflector geometry. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 00 (original).](../assets/figures/ml_fig_c216_00.png)
*Figure — Givens plane rotation steps. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 00 (original).](../assets/figures/ml_fig_c217_00.png)
*Figure — LU partial pivoting necessity. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 00 (original).](../assets/figures/ml_fig_c218_00.png)
*Figure — Economy SVD factor shapes. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 00 (original).](../assets/figures/ml_fig_c219_00.png)
*Figure — Schur triangular eigen form. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 00 (original).](../assets/figures/ml_fig_c220_00.png)
*Figure — Krylov subspace residual drop. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 00 (original).](../assets/figures/ml_fig_c221_00.png)
*Figure — Householder reflection geometry. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 00 (original).](../assets/figures/ml_fig_c222_00.png)
*Figure — Truncated SVD energy vs rank. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 00 (original).](../assets/figures/ml_fig_c223_00.png)
*Figure — Givens plane rotation zeroing. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 00 (original).](../assets/figures/ml_fig_c224_00.png)
*Figure — Lanczos tridiagonal projection. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 00 (original).](../assets/figures/ml_fig_c225_00.png)
*Figure — Cholesky lower-triangular factor. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 00 (original).](../assets/figures/ml_fig_c226_00.png)
*Figure — Pseudoinverse singular threshold. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 00 (original).](../assets/figures/ml_fig_c227_00.png)
*Figure — Matrix condition number error amp. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 00 (original).](../assets/figures/ml_fig_c228_00.png)
*Figure — Arnoldi Hessenberg structure. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 00 (original).](../assets/figures/ml_fig_c229_00.png)
*Figure — Gram-Schmidt orthogonalization. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 00 (original).](../assets/figures/ml_fig_c230_00.png)
*Figure — Woodbury low-rank inverse. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 00 (original).](../assets/figures/ml_fig_c231_00.png)
*Figure — Economy SVD shape diagram. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 00 (original).](../assets/figures/ml_fig_c232_00.png)
*Figure — Matrix sketch size vs epsilon. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 00 (original).](../assets/figures/ml_fig_c233_00.png)
*Figure — QR iteration residual decay. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 00 (original).](../assets/figures/ml_fig_c234_00.png)
*Figure — Conjugate gradient residual. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 00 (original).](../assets/figures/ml_fig_c235_00.png)
*Figure — Jacobi eigenvalue sweep residual. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 00 (original).](../assets/figures/ml_fig_c236_00.png)
*Figure — GMRES residual restart. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 00 (original).](../assets/figures/ml_fig_c237_00.png)
*Figure — Power iteration residual decay. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 00 (original).](../assets/figures/ml_fig_c238_00.png)
*Figure — BiCGSTAB residual path. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 00 (original).](../assets/figures/ml_fig_c239_00.png)
*Figure — Lanczos residual orthogonal loss. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 00 (original).](../assets/figures/ml_fig_c240_00.png)
*Figure — MINRES residual path. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 00 (original).](../assets/figures/ml_fig_c241_00.png)
*Figure — QMR residual bi-orthogonal path. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 00 (original).](../assets/figures/ml_fig_c242_00.png)
*Figure — CG residual A-norm path. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 00 (original).](../assets/figures/ml_fig_c243_00.png)
*Figure — LSQR residual least-squares path. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 00 (original).](../assets/figures/ml_fig_c244_00.png)
*Figure — IDR residual defect path. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 00 (original).](../assets/figures/ml_fig_c245_00.png)
*Figure — SYMMLQ residual Krylov path. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 00 (original).](../assets/figures/ml_fig_c246_00.png)
*Figure — TFQMR residual transpose path. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 00 (original).](../assets/figures/ml_fig_c247_00.png)
*Figure — CGS residual bi-CG path. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 00 (original).](../assets/figures/ml_fig_c248_00.png)
*Figure — FGMRES flexible residual path. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 00 (original).](../assets/figures/ml_fig_c249_00.png)
*Figure — IDR-S residual defect path. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 00 (original).](../assets/figures/ml_fig_c250_00.png)
*Figure — QMR-SYM residual path. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 00 (original).](../assets/figures/ml_fig_c251_00.png)
*Figure — MINRES-QLP residual path. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 00 (original).](../assets/figures/ml_fig_c252_00.png)
*Figure — BiCG residual dual path. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 00 (original).](../assets/figures/ml_fig_c253_00.png)
*Figure — LSQR least-squares residual. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 00 (original).](../assets/figures/ml_fig_c254_00.png)
*Figure — GMRES(m) restart residual. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 00 (original).](../assets/figures/ml_fig_c255_00.png)
*Figure — CGNE residual normal eq. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 00 (original).](../assets/figures/ml_fig_c256_00.png)
*Figure — TFQMR transpose residual. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 00 (original).](../assets/figures/ml_fig_c257_00.png)
*Figure — Arnoldi orthogonal loss path c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 00 (original).](../assets/figures/ml_fig_c258_00.png)
*Figure — Householder QR residual c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 00 (original).](../assets/figures/ml_fig_c259_00.png)
*Figure — Cholesky condition path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 00 (original).](../assets/figures/ml_fig_c260_00.png)
*Figure — SVD singular decay path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 00 (original).](../assets/figures/ml_fig_c261_00.png)
*Figure — Jacobi iteration residual c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 00 (original).](../assets/figures/ml_fig_c262_00.png)
*Figure — Gauss-Seidel residual c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 00 (original).](../assets/figures/ml_fig_c263_00.png)
*Figure — SOR omega residual path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 00 (original).](../assets/figures/ml_fig_c264_00.png)
*Figure — Chebyshev semi-iter residual c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 00 (original).](../assets/figures/ml_fig_c265_00.png)
*Figure — Multigrid V-cycle residual c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 00 (original).](../assets/figures/ml_fig_c266_00.png)
*Figure — Preconditioned CG residual c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 00 (original).](../assets/figures/ml_fig_c267_00.png)
*Figure — Deflated CG residual path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 00 (original).](../assets/figures/ml_fig_c268_00.png)
*Figure — Block CG residual path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 00 (original).](../assets/figures/ml_fig_c269_00.png)
*Figure — Flexible GMRES residual c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 00 (original).](../assets/figures/ml_fig_c270_00.png)
*Figure — Recycling Krylov residual c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 00 (original).](../assets/figures/ml_fig_c271_00.png)
*Figure — Sketch-and-project residual c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 00 (original).](../assets/figures/ml_fig_c272_00.png)
*Figure — Krylov subspace residual path c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 00 (original).](../assets/figures/ml_fig_c273_00.png)
*Figure — Arnoldi orthogonal loss path c273. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Mathematics is the compression format of machine learning: a page of symbols stands in for pages of prose, and fluency with the symbols is what makes the rest of the book legible. This chapter rebuilt that fluency from an elementary base. It began with the language itself — sets, functions, and logic — and the algebra of numbers, exponents, and logarithms, then cataloged the handful of functions (linear, polynomial, exponential, logarithmic, sigmoid, softmax, ReLU) that recur everywhere in modeling. Summation and counting supplied the combinatorics behind probability; trigonometry and the unit circle supplied the sinusoids behind Fourier features, positional encodings, and cosine similarity.

The calculus sequence is the analytic core. Single-variable derivatives measure change and locate optima; integrals measure accumulated area and, in probability, total mass and expectation. The multivariable extension — partial derivatives, the gradient as the direction of steepest ascent, the Jacobian and Hessian, and the second-order Taylor expansion — is exactly the machinery of backpropagation and of every optimizer in the book. Linear algebra supplied the other half: vectors and their norms, dot products, and cosine angles; matrices as data tables and as linear transformations, with multiplication, inverses, determinants, and linear systems; and the eigen- and singular-value decompositions that make PCA, low-rank approximation, and spectral methods possible. Probability contributed its axioms, conditional reasoning and Bayes’ theorem, random variables, expectation, and variance. Optimization tied calculus and linear algebra together through objective functions, convexity, and gradient descent. Finally, discrete mathematics and Big-O analysis governed which algorithms are affordable, and a short tour of floating-point arithmetic warned where exact mathematics and finite-precision computation diverge. With these tools in hand, no later chapter should be inaccessible; when one invokes a gradient, an eigenvector, or a posterior probability, the full treatment is here to return to.

## Practice and Reflection

Work these by hand; several deliberately combine two or three sections. Brief answers follow each.

(Logs turn products into sums.) Show that ln(p₁·p₂·p₃) = ln p₁ + ln p₂ + ln p₃, and explain why maximizing a log-likelihood is easier than maximizing a product of probabilities. (Answer: apply log(ab)=log a+log b twice; sums are numerically stable and differentiate term by term.)

(Sigmoid + log-loss.) A logistic model has z = −1.0 + 0.8(1.0) + 0.4(0.5). Compute z, the probability p = σ(z), and the log-loss −ln p for true label y = 1. (Answer: z = 0; p = 0.5; loss = ln 2 ≈ 0.693.)

(Gradient step.) For L(w) = (w − 3)² + 1, compute L′(w), and starting at w = 0 with learning rate η = 0.2, take two gradient-descent steps. (Answer: L′ = 2(w−3); w₁ = 1.2, w₂ = 2.04, approaching the minimum at w = 3.)

(Eigen/PCA.) Find the eigenvalues and eigenvectors of 𝐀 = [[2, 1], [1, 2]] and state which eigenvector is the first principal direction. (Answer: λ = 3 with 𝐯 = [1,1]/√2; λ = 1 with 𝐯 = [1,−1]/√2; the λ = 3 direction is first.)

(Vectors.) For 𝐚 = [3, 4] and 𝐛 = [4, 3], compute 𝐚·𝐛, ‖𝐚‖₂, ‖𝐛‖₂, and cos θ. (Answer: 24; 5; 5; cos θ = 24/25 = 0.96.)

(Bayes / PPV.) A test has sensitivity 0.90 and specificity 0.90; disease prevalence is 0.01. Compute the positive predictive value. (Answer: (0.90·0.01)/(0.90·0.01 + 0.10·0.99) = 0.009/0.108 ≈ 0.083.)

(Complexity.) Give the Big-O cost of a doubly nested loop over n items, and of sorting then scanning. (Answer: O(n²); O(n log n).)

(Integral as probability.) For the density f(x) = 2x on [0, 1], verify ∫₀¹ f dx = 1 and compute P(0 ≤ X ≤ 0.5). (Answer: ∫2x = x²; total = 1; P = 0.25.)

(Chain rule.) With h = σ(u) and u = wx + b, x = 2, w = 1, b = 0, compute ∂h/∂w. (Answer: ∂h/∂w = σ(u)(1−σ(u))·x; u = 2, σ(2) ≈ 0.881, so ≈ 0.881·0.119·2 ≈ 0.210.)

(Linear system.) Solve [[2, 1], [1, 3]]𝐱 = [3, 5] by any method. (Answer: x₁ = 0.8, x₂ = 1.4.)

(Counting.) Compute C(6, 2) and expand (a + b)³. (Answer: 15; a³ + 3a²b + 3ab² + b³.)

(Radians and cosine.) Convert 60° to radians and compute the cosine similarity of two unit vectors separated by that angle. (Answer: π/3; cos 60° = 0.5.)
