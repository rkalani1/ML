# Chapter 13. Reinforcement Learning

## Opening

A simulation lab proposes reinforcement learning for sequential BP targets after thrombolysis. RL requires a reward design that does not quietly optimize the wrong clinical trade-off. Stroke care is not Atari.


![Agent–environment loop for sequential decisions (original).](../assets/figures/ml_fig_rl_loop.png)

*Agent–environment loop for sequential decisions (original).*
## Learning Objectives

Formalize sequential decision making as a Markov decision process with states, actions, transitions, rewards, and discounting.

Define returns, policies, and value functions V and Q, and derive Bellman expectation and optimality equations with numerical backups.

Contrast exploitation versus exploration and apply multi-armed bandit algorithms (epsilon-greedy, UCB, Thompson sampling).

Implement dynamic programming (policy evaluation, policy iteration, value iteration) and sample-based methods (Monte Carlo, TD, SARSA, Q-learning, Dyna-Q) on small MDPs.

Explain n-step returns, TD(lambda) with eligibility traces, and function approximation for continuous or large state spaces.

Outline deep RL algorithms: DQN family (replay, target nets, DDQN, dueling, PER), policy gradients (REINFORCE, TRPO, PPO), actor-critic (A3C, DDPG, TD3, SAC), and world-model Dreamer variants.

Critique clinical reward misspecification, offline RL coverage failure, and ethical limits on exploration at the bedside.

## 13.1 From Supervised Labels to Sequential Decisions

Most of the methods in earlier chapters assume a fixed dataset of independent examples: given an input x, predict a label y or a continuous target. Reinforcement learning (RL) addresses a different problem. An agent repeatedly chooses actions that affect a dynamic environment. After each action the environment transitions to a new situation and emits a scalar reward signal. The agent’s objective is not to mimic a teacher’s labels on isolated examples, but to accumulate as much reward as possible over time—often over a long horizon in which early mistakes close off later opportunities.

This sequential structure creates credit assignment: a high reward received after many steps may depend on several earlier choices. It also creates a fundamental data problem. The distribution of states the agent visits depends on the policy the agent is currently using, so data collection and learning are intertwined. Finally, actions must balance exploration (trying uncertain options to learn) against exploitation (using the best known option). These three themes—credit assignment, policy-dependent data, and exploration—recur throughout the chapter.

RL appears in robot control, game playing, recommendation systems that optimize long-term engagement, inventory and pricing, dialogue management, and many industrial control loops. Even when a full RL solution is not deployed, the language of policies, value functions, and rewards is useful for analyzing any system that makes repeated decisions under uncertainty. For a neurologist-epidemiologist, the same language clarifies titration sequences, secondary prevention pathways, and adaptive trial designs—while also highlighting why casually applying game-playing RL recipes to irreversible clinical decisions is scientifically and ethically hazardous.

Credit assignment: which past actions caused a late outcome?

Non-i.i.d. data: the policy shapes the state distribution it learns from.

Exploration versus exploitation: learn more versus harvest known reward.

Delayed consequences: myopic greedy choices can destroy long-horizon value.

## 13.2 The Agent-Environment Loop and Exploitation versus Exploration

At discrete time steps t = 0, 1, 2, … the agent observes a state S_t drawn from a state space S (which may be discrete or continuous, fully or partially observed). It selects an action A_t from an action set A(S_t), or from a global action set A. The environment responds with a reward R_{t+1} and a next state S_{t+1}. The interaction produces a trajectory S_0, A_0, R_1, S_1, A_1, R_2, … . Conventionally the reward is indexed by the time at which it is received, so R_{t+1} is the immediate consequence of taking A_t in S_t.

![13.1: The agent-environment interaction loop. At each discrete step the agent, following its policy π(a|s), selects an action ](../assets/figures/ml_concept_13.1_48097877.png)

*Figure 13.1 — original teaching graphic.*

The agent’s behavior is summarized by a policy pi. A deterministic policy maps states to actions: a = pi(s). A stochastic policy specifies a distribution over actions, written pi(a|s) = P(A_t = a | S_t = s). Stochastic policies are essential for exploration and for domains with imperfect information.

Exploitation means selecting the action that currently appears best according to the agent’s estimates—for example, the action with highest estimated Q-value. Exploration means selecting a suboptimal or uncertain action to gather information that may improve future decisions. An agent that always exploits may lock onto a locally good action and never discover a superior one; an agent that explores forever sacrifices return. The optimal balance depends on horizon, non-stationarity, and the cost of mistakes. In clinical analogues, pure exploitation is “always follow the current best protocol estimate”; pure exploration is “randomize treatments without equipoise”—the second is usually unethical outside a trial.

Simple strategies include epsilon-greedy (choose a random action with probability epsilon, else greedy), Boltzmann or softmax exploration (sample actions with probabilities proportional to exp(Q(s,a)/tau) with temperature tau), and optimistic initialization (start with high Q values so untried actions look attractive). More sophisticated methods use uncertainty estimates: upper confidence bounds add a bonus proportional to the square root of log(t)/n(s,a); Thompson sampling draws a parameter from a posterior and acts optimally for that draw. In deep RL, intrinsic motivation bonuses (prediction error, count-based pseudo-counts, disagreement among ensembles) encourage visiting novel states when external rewards are sparse.

Observation: what the agent sees (may equal the true state or only a partial view).

Action: discrete choices (move left/right) or continuous controls (drug infusion rate).

Reward: a scalar teaching signal designed by the engineer; not automatically the true clinical goal.

Episode: a finite trajectory ending in a terminal state; continuing tasks never terminate.

Exploration schedule: often anneal randomness over training; freeze a safer policy at deployment.

## 13.3 Markov Decision Processes and Gridworld

A Markov decision process (MDP) is the standard mathematical model for fully observed RL. An MDP is a tuple (S, A, P, R, gamma). S is the set of states. A is the set of actions (or A(s) if actions depend on the state). The transition function P(s’ | s, a) gives the probability of moving to s’ when action a is taken in state s. The reward function can be written several equivalent ways: as an expected immediate reward R(s, a) = E[R_{t+1} | S_t = s, A_t = a], or more finely as R(s, a, s’) when the reward depends on the successor, or as a distribution over reward values. The discount factor gamma in [0, 1) scales future rewards relative to immediate ones.

The Markov property states that the future depends on the past only through the present state and action: P(S_{t+1}, R_{t+1} | history up to t) = P(S_{t+1}, R_{t+1} | S_t, A_t). In practice we engineer features or use recurrent architectures so that the representation we call “state” approximately satisfies this property. When only a partial observation is available, the model is a partially observable MDP (POMDP); that extension motivates belief states and history-based policies.

Discounting serves two purposes. First, it makes infinite-horizon returns finite when rewards are bounded: if |R| <= R_max then the return is at most R_max / (1 - gamma). Second, it encodes a preference for sooner rewards over later ones. Setting gamma close to 1 makes the agent far-sighted; gamma near 0 makes it myopic. For episodic tasks one may use gamma = 1 if every episode ends almost surely and total reward is bounded.

![Discount factor γ: present value of delayed rewards (scientific; original).](../assets/figures/ml_fig_discount_gamma.png)

*Left: γ^t decays a unit reward delayed t steps. Right: constant reward stream R=1 has value R/(1−γ), so γ=0.9 → 10 while γ=0.99 → 100 — far-sighted policies care about long horizons (original).*

### Gridworld as the Pedagogic MDP

A classic teaching environment is a small grid of cells. Each cell is a state; actions are the four compass moves {N, S, E, W}; some cells are walls; one cell is a terminal goal with reward +1; stepping into a pit yields -1 and terminates. A small living reward (for example -0.04 per step) encourages shorter paths. Transitions may be deterministic (intended move always succeeds) or stochastic (with probability p the agent slips sideways). With a known transition model, dynamic programming produces a value heat map and arrows for the greedy policy. Students should implement a 4x3 or 5x5 grid, run value iteration with gamma = 0.9, and verify that values decrease with distance from the goal and that the policy points along shortest safe paths. Stochastic slips create a risk-reward trade-off: a path hugging a cliff may be shorter in expectation under deterministic dynamics but catastrophic under slip noise—an analogy to aggressive versus conservative clinical pathways under outcome uncertainty.

![13.2: A deterministic 4×3 gridworld Markov decision process solved by value iteration (γ = 0.9, step cost −0.04). Each open ce](../assets/figures/ml_concept_13.2_70530ae2.png)

*Figure 13.2 — original teaching graphic.*

# Tiny deterministic gridworld sketch (educational)
# States: 0..n*m-1; actions: 0=N,1=S,2=E,3=W
GOAL, PIT = 3, 7 # example indices on a small grid
def step(s, a, n_cols=4):
if s in (GOAL, PIT):
return s, 0.0, True
r, c = divmod(s, n_cols)
dr, dc = [(-1, 0), (1, 0), (0, 1), (0, -1)][a]
nr, nc = r + dr, c + dc
if not (0 <= nr < 3 and 0 <= nc < n_cols):
nr, nc = r, c # bump into wall
ns = nr * n_cols + nc
if ns == GOAL:
return ns, 1.0, True
if ns == PIT:
return ns, -1.0, True
return ns, -0.04, False

## 13.4 Deterministic, Nondeterministic, and Stochastic Models

Environments differ in how actions map to outcomes. A deterministic model has, for each (s, a), a single next state s’ and a fixed reward. Planning reduces to search over a graph of state-action transitions. A stochastic (probabilistic) model specifies a distribution P(s’|s,a); the same action in the same state can yield different successors on different trials. “Nondeterministic” in classical AI often means a set of possible successors without probabilities; RL usually works with full probability kernels so that expectations are well-defined.

Stochasticity matters algorithmically: dynamic programming averages over P; sample-based methods must average over experience; variance of returns increases, so larger sample sizes or lower learning rates are needed. Clinically, “stochastic slips” model inter-patient variability and incomplete control (a prescribed dose does not deterministically produce the same physiologic response). When building simulators for offline evaluation, understating stochasticity produces brittle policies that look perfect in a deterministic toy world and fail under real noise.

State transition diagrams display states as nodes and actions as labeled arcs to successor states, often annotated with probabilities and rewards. Drawing the diagram for a small MDP is the best first debugging tool: if the diagram is wrong, no learning algorithm will save the model. For teaching, start with a two- or three-state diagram, write every P(s’|s,a) and R explicitly, and only then code value iteration. Many student bugs are silent diagram errors rather than coding errors.

## 13.5 Policy, Expected Return, Value Functions, and State Transitions

The return from time t is the discounted sum of future rewards:

G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + … = sum_{k=0}^{inf} gamma^k R_{t+1+k}.

Equivalently, G_t = R_{t+1} + gamma G_{t+1}. The agent seeks a policy that maximizes expected return from the start-state distribution, or more generally that maximizes the value of every state under the induced visitation.

The state-value function of a policy pi is the expected return when starting from s and thereafter following pi:

V^pi(s) = E_pi[G_t | S_t = s].

The action-value function (Q-function) is the expected return when starting from s, taking action a, and thereafter following pi:

Q^pi(s, a) = E_pi[G_t | S_t = s, A_t = a].

These are related by V^pi(s) = sum_a pi(a|s) Q^pi(s, a) for stochastic policies, and V^pi(s) = Q^pi(s, pi(s)) for deterministic ones.

An optimal state value is V*(s) = max_pi V^pi(s), and an optimal action value is Q*(s, a) = max_pi Q^pi(s, a). Any policy that achieves V* is optimal. In finite MDPs at least one deterministic optimal policy exists. Knowing Q* is especially convenient: a greedy policy pi*(s) in argmax_a Q*(s, a) is optimal without needing a separate model of P and R at decision time.

Value functions compress long-horizon consequences into a single number per state or state-action pair. Learning or computing values is therefore a central algorithmic strategy: estimate values well, then act greedily (or softly) with respect to them. Prediction problems estimate V^pi or Q^pi for a fixed policy; control problems seek an optimal or near-optimal policy. This distinction organizes tabular and deep methods alike: first evaluate, then improve, or fuse both into a single optimality backup as in value iteration and Q-learning.

## 13.6 Bellman Equations and Worked Examples

Because G_t = R_{t+1} + gamma G_{t+1}, taking expectations under pi yields the Bellman expectation equations:

V^pi(s) = sum_a pi(a|s) sum_{s’} P(s’|s,a) [ R(s,a,s’) + gamma V^pi(s’) ],

Q^pi(s,a) = sum_{s’} P(s’|s,a) [ R(s,a,s’) + gamma sum_{a’} pi(a’|s’) Q^pi(s’,a’) ].

In matrix form for finite S, V^pi = r^pi + gamma P^pi V^pi, so V^pi = (I - gamma P^pi)^{-1} r^pi. This linear system is the foundation of dynamic programming evaluation.

Optimal values satisfy the Bellman optimality equations:

V*(s) = max_a sum_{s’} P(s’|s,a) [ R(s,a,s’) + gamma V*(s’) ],

Q*(s,a) = sum_{s’} P(s’|s,a) [ R(s,a,s’) + gamma max_{a’} Q*(s’,a’) ].

These are nonlinear because of the max operator. They uniquely determine V* and Q* for gamma < 1 (or under mild conditions for episodic undiscounted problems). The optimality equations are both a characterization of the solution and a template for iterative algorithms: replace unknown values by current estimates and apply the right-hand side as an update.

### Worked Bellman Backup (Single State)

Suppose three actions from s with expected one-step returns already folded into R and known next values: action Left gives R=0 then V(s_L)=5; action Right gives R=1 then V(s_R)=3. With gamma=0.9, the action values are 0 + 0.9*5 = 4.5 and 1 + 0.9*3 = 3.7. The greedy choice is Left with backup 4.5. This single-state arithmetic is exactly what value iteration does for every state each sweep.

![Bellman optimality backup: Q = R + γV and V* = max Q (original).](../assets/figures/ml_fig_bellman_backup.png)

*Figure — One Bellman backup cell. **Left:** from state \(s\), action Left yields \(Q=0+0.9\times5=4.5\); Right yields \(Q=1+0.9\times3=3.7\). **Right:** the optimality backup takes \(\max_a Q(s,a)=4.5\), so the greedy improved policy picks Left. Value iteration applies this diagram to every state each sweep; the expectation form averages over \(P(s'|s,a)\) when transitions are stochastic.*

### Worked Two-State MDP (Deterministic)

States {s1, s2}, actions {Stay, Go}, gamma = 0.9. From s1, Stay stays with reward +1; Go moves to s2 with reward 0. From s2, Stay stays with reward +2; Go moves to s1 with reward 0. Initialize V_0(s1)=V_0(s2)=0.

![Value iteration on the two-state MDP (original).](../assets/figures/ml_fig_value_iteration.png)

*Figure 13.3. Value iteration on the worked two-state MDP (γ = 0.9). Starting from V_0 = 0, synchronous Bellman-optimality backups drive V(s_1) and V(s_2) to their exact fixed points V*(s_1) = 18 and V*(s_2) = 20 (dashed target lines). Early sweeps make Stay the greedy action in s_1, but once the estimated value of the richer state s_2 grows enough the greedy choice flips to Go at iteration 3—forgoing the immediate +1 reward for the more valuable state, which is the essence of long-horizon planning.*

Iteration 1: s1 Stay->1.0, Go->0 so V1(s1)=1.0; s2 Stay->2.0, Go->0 so V1(s2)=2.0.

Iteration 2: s1 Stay->1+0.9*1=1.9, Go->0+0.9*2=1.8 so V2(s1)=1.9; s2 Stay->2+0.9*2=3.8, Go->0+0.9*1=0.9 so V2(s2)=3.8.

Iteration 3: s1 Stay->1+0.9*1.9=2.71, Go->0+0.9*3.8=3.42 so V3(s1)=3.42 (greedy flips to Go); s2 Stay->5.42.

Exact optimum under pi*(s1)=Go, pi*(s2)=Stay: V*(s2)=2/(1-0.9)=20, V*(s1)=0+0.9*20=18. Check Stay in s1: 1+0.9*18=17.2 < 18; Go in s2: 0+0.9*18=16.2 < 20. Thus a short-term sacrifice (reward 0 when leaving s1) is optimal because s2 is far more valuable. This pattern—forgoing immediate reward for a better state—is the essence of sequential decision quality and the reason myopic supervised “next-step” predictors are not substitutes for value functions when horizons are long.

# Value iteration on the two-state MDP
gamma = 0.9
V = {“s1”: 0.0, “s2”: 0.0}
T = {
“s1”: {“Stay”: (1.0, “s1”), “Go”: (0.0, “s2”)},
“s2”: {“Stay”: (2.0, “s2”), “Go”: (0.0, “s1”)},
}
for it in range(50):
newV = {}
for s in V:
newV[s] = max(r + gamma * V[sp] for (r, sp) in T[s].values())
V = newV
pi = {s: max(T[s], key=lambda a: T[s][a][0] + gamma * V[T[s][a][1]]) for s in V}
print(V, pi) # ~ s1:18, s2:20; Go, Stay

## 13.7 Multi-Armed Bandits: Epsilon-Greedy, UCB, and Thompson Sampling

A multi-armed bandit is an MDP with a single state (or i.i.d. rounds): each action a (“arm”) yields a stochastic reward with unknown mean mu_a, and the goal is to maximize cumulative reward over T pulls. There is no long-term state transition, so the problem isolates exploration. Regret is the gap between the reward of always pulling the best arm and the reward actually obtained.

![13.4: Cumulative regret on a fixed six-armed Bernoulli bandit, averaged over 300 simulated runs. Pure greedy (ε = 0) frequentl](../assets/figures/ml_concept_13.4_ef666085.png)

*Figure 13.4 — original teaching graphic.*

### Epsilon-Greedy

With probability epsilon, choose a uniform random arm; otherwise choose argmax_a Q_hat(a), where Q_hat is the sample mean of rewards for arm a. Simple and robust; suboptimal logarithmic constants compared with UCB, but easy to anneal (decay epsilon over time). Constant epsilon wastes a fixed fraction of pulls forever; decaying schedules such as epsilon_t = c/t improve asymptotics but need tuning for finite horizons.

![ε-greedy vs UCB cumulative regret on a five-arm Bernoulli bandit (scientific; original).](../assets/figures/ml_fig_bandit_explore.png)

*Mean cumulative regret over 80 simulated runs: pure greedy locks onto a suboptimal arm; ε=0.1 and UCB1 keep exploring and lower long-run regret (synthetic means; original).*

### Upper Confidence Bound (UCB1)

Prefer arms with high estimated mean and high uncertainty:

UCB(a) = Q_hat(a) + c * sqrt( ln(t) / N(a) ),

where N(a) is the pull count and t is the total number of pulls. The bonus shrinks as N(a) grows, automatically shifting from exploration to exploitation. With appropriate c, UCB1 achieves logarithmic regret under standard bounded-reward assumptions. Intuitively, UCB is “optimism in the face of uncertainty”: untried arms look artificially good until data accumulate.

### Thompson Sampling

Maintain a posterior over each arm’s mean (e.g., Beta for Bernoulli rewards). Each round, sample a mean from each posterior and pull the arm with the best sample. This Bayesian approach explores arms that are plausible under the posterior and concentrates on the winner as posteriors shrink. For Bernoulli arms with Beta(alpha, beta) priors, observing reward 1 increments alpha; reward 0 increments beta. Thompson sampling often matches or beats UCB in practice and extends naturally to complex models when posterior sampling is feasible.

### Contextual Bandits and When Bandits Fit

Contextual bandits observe a feature vector each round and learn a mapping from context to arm values—halfway between bandits and full RL. Real-world bandit-like problems include ad placement, ranking variants, A/B testing with adaptive allocation, and dose-finding under equipoise. In stroke systems research, choosing among a small set of triage messaging templates to maximize door-to-CT compliance is closer to a contextual bandit than to deep sequential control—provided exploration is ethically constrained and primary safety outcomes are monitored.

Epsilon-greedy: simple; wastes fixed fraction on pure random pulls unless annealed.

UCB: optimism under uncertainty; no posterior model required.

Thompson: samples from posterior; natural randomization; needs a likelihood model.

Regret metrics: cumulative regret and simple regret (final arm quality) answer different questions.

import math
# UCB1 sketch for K arms
def ucb1_select(means, counts, t, c=math.sqrt(2)):
best, best_score = 0, -1e9
for a, (m, n) in enumerate(zip(means, counts)):
n = max(n, 1)
score = m + c * math.sqrt(math.log(max(t, 1)) / n)
if score > best_score:
best, best_score = a, score
return best

## 13.8 Optimal Policy, Control versus Prediction

An optimal policy pi* satisfies V^{pi*}(s) = V*(s) for all s (or almost all s under the relevant measure). In finite discounted MDPs, V* and Q* exist and are unique; at least one deterministic optimal policy exists even if many stochastic optima also exist. Softmax or entropy-regularized objectives (as in SAC later) intentionally prefer stochastic policies for robustness and exploration.

### Prediction versus Control

Prediction (policy evaluation) answers: “If we follow protocol pi, what is the expected long-run return from each state?” Control answers: “What policy maximizes return?” Monte Carlo and TD evaluation are prediction methods; Q-learning and policy iteration are control methods. In epidemiology, prediction resembles estimating outcomes under a fixed care pathway; control resembles finding a better pathway. Causal identification requirements differ: observational prediction of V^pi for the historical policy is closer to descriptive prognosis; claiming that a new pi’ would improve outcomes requires assumptions about counterfactual actions—exactly the offline RL problem revisited in the clinical notes.

Greedy policies with respect to Q* are optimal. Greedy policies with respect to approximate Q can be arbitrarily bad if approximation errors concentrate on critical state-action pairs—hence the importance of coverage, function-class capacity, and conservative updates in offline settings. Always separate the scientific claim (“we estimated value under historical care”) from the interventional claim (“we recommend a new automated policy”).

## 13.9 Tabular Monte Carlo: Policy Evaluation and Policy Improvement

Monte Carlo (MC) methods learn from complete episode returns without a model of P. After an episode ends, for each visited state (or state-action pair) one forms the return G_t and averages these returns. First-visit MC updates using only the first occurrence of a state in each episode; every-visit MC uses all occurrences. Under standard assumptions, averages converge to V^pi or Q^pi for the behavior policy that generated the data.

MC policy evaluation: generate episodes under pi; for each first visit to s, append G_t to a list; set V(s) to the mean of that list. MC control alternates improving the policy (epsilon-greedy with respect to Q) and generating new episodes. Because updates wait until the end of the episode, MC has high variance but is unbiased for the true return. It cannot bootstrap from other value estimates and struggles with very long episodes or continuing tasks unless artificial truncation is introduced.

### Challenges of Monte Carlo

Episodes must terminate; infinite-horizon tasks need discounting and truncation. (2) Variance of G_t grows with horizon. (3) Exploring starts or soft policies are required to visit all state-action pairs. (4) Off-policy MC uses importance sampling ratios product_k pi(a_k|s_k)/b(a_k|s_k), which can have huge variance when pi and behavior b differ.

### When to Use Monte Carlo

Prefer MC when a good simulator produces short episodes, when one wants unbiased targets, or when bootstrapping is unstable with function approximation. Prefer TD when episodes are long, partial trajectories are available, or online updates are required after each step. Hybrid n-step and lambda methods (later section) blend the two.

## 13.10 Dynamic Programming: Policy Iteration and Value Iteration

When a complete model P and R is known and S and A are finite and small, dynamic programming (DP) computes exact values and optimal policies. DP is rarely sufficient alone for large real-world problems, but it is the conceptual parent of almost every RL method: Monte Carlo and temporal-difference algorithms can be viewed as sample-based approximations to Bellman backups.

### Policy Evaluation

Given pi, iterative policy evaluation starts from an arbitrary V_0 and applies the Bellman expectation backup until convergence:

V_{k+1}(s) <- sum_a pi(a|s) sum_{s’} P(s’|s,a) [ R + gamma V_k(s’) ].

### Policy Improvement

Policy improvement constructs a greedy policy pi’(s) in argmax_a Q^pi(s,a), where Q^pi is computed from V^pi. The policy improvement theorem guarantees V^{pi’} >= V^pi, with strict improvement for some state if pi was not already optimal.

### Policy Iteration and Value Iteration

Policy iteration alternates complete evaluation and greedy improvement until the policy stabilizes. Value iteration merges the two: each sweep applies the optimality backup V_{k+1}(s) <- max_a sum_{s’} P(s’|s,a) [ R + gamma V_k(s’) ]. After convergence (or after enough sweeps), extract pi(s) in argmax_a Q(s,a) from the approximate V. Asynchronous and prioritized variants update states in flexible orders; the key requirement is that every state is updated infinitely often in the limit.

![Policy iteration: evaluate → improve until stable (chapter two-state MDP; original).](../assets/figures/ml_fig_policy_iteration.png)

*Figure — Policy iteration. **Left:** loop—initialize π, evaluate \(V^\pi\) under the Bellman expectation operator, greedy-improve, stop when π is unchanged. **Right:** on the chapter two-state MDP, evaluating Stay-everywhere yields \(V(s_1)=10\), \(V(s_2)=20\); improvement flips \(s_1\) to Go and reaches \(V^*=(18,20)\). Policy improvement theorem: \(V^{\pi'}\ge V^\pi\).*

Converting Bellman equations into iterative updates is simply replacing the equality V = T V with the assignment V <- T V, where T is the Bellman operator. For gamma < 1, T is a contraction in max-norm, so iterates converge to the unique fixed point regardless of initialization (for value iteration’s optimality operator).

### When to Use Dynamic Programming

Use DP for small known MDPs, for debugging reward design in simulators, and as a ground-truth oracle against which sample-based methods can be compared. Do not expect classical DP tables to scale to high-dimensional neurologic state spaces without abstraction or function approximation. Still, implementing policy and value iteration on a hand-specified 5-10 state clinical sketch (for example, triage stages with two actions) builds intuition no black-box deep RL tutorial replaces.

# Policy iteration sketch (finite MDP dicts)
# P[s][a] = list of (prob, next_s, reward)
def policy_evaluation(pi, P, gamma=0.9, theta=1e-8):
V = {s: 0.0 for s in P}
while True:
delta = 0.0
for s in P:
a = pi[s]
v = sum(p * (r + gamma * V[sp]) for p, sp, r in P[s][a])
delta = max(delta, abs(v - V[s]))
V[s] = v
if delta < theta:
return V

def policy_improvement(V, P, gamma=0.9):
pi = {}
for s in P:
pi[s] = max(
P[s],
key=lambda a: sum(p * (r + gamma * V[sp]) for p, sp, r in P[s][a]),
)
return pi

## 13.11 Temporal-Difference Learning, SARSA, Q-Learning, and Dyna-Q

Temporal-difference (TD) methods combine ideas from MC and DP: they learn from sampled experience (like MC) but bootstrap from current value estimates (like DP). The one-step TD target for policy evaluation is R_{t+1} + gamma V(S_{t+1}). The update is

![13.5: Backup diagrams contrasting how three families of methods estimate a value. Open circles are states, filled dots are act](../assets/figures/ml_concept_13.5_19e18b4d.png)

*Figure 13.5 — original teaching graphic.*

V(S_t) <- V(S_t) + alpha [ R_{t+1} + gamma V(S_{t+1}) - V(S_t) ],

where alpha is a step-size. The term in brackets is the TD error delta_t. Bootstrapping reduces variance relative to full returns but introduces bias that vanishes as V approaches V^pi.

![TD error δ = R + γV(s′) − V(s) and TD(0) convergence on a reward-at-end chain (original).](../assets/figures/ml_fig_td_error.png)

*Figure — One-step TD. **Left:** backup diagram for δ and the α-weighted update. **Right:** TD(0) value estimates on a five-state chain with terminal reward approach V* (dashed). TD bootstraps (lower variance than full Monte Carlo return; bias until V is accurate). Value estimates are not causal treatment effects.*

### SARSA (On-Policy Control)

After observing (S, A, R, S’, A’), update

Q(S,A) <- Q(S,A) + alpha [ R + gamma Q(S’,A’) - Q(S,A) ].

The next action A’ is chosen by the same epsilon-greedy policy used for behavior, so the method is on-policy. Soft policies that continue to explore are required for convergence results that guarantee optimality in the limit. On cliff-walking grids, SARSA often prefers safer paths because it learns Q for the exploratory policy that sometimes slips off the cliff.

### Q-Learning (Off-Policy Control)

Q-learning targets the optimal action value directly:

Q(S,A) <- Q(S,A) + alpha [ R + gamma max_{a’} Q(S’,a’) - Q(S,A) ].

Behavior may be epsilon-greedy for exploration, but the backup uses a greedy max, so the learned Q approaches Q* independent of the exploration policy (under tabular conditions: all pairs visited infinitely often, appropriate alpha). This off-policy character is powerful: one can learn about optimal behavior while acting suboptimally to explore.

Expected SARSA replaces Q(S’,A’) by the expectation under the current policy, reducing variance. Double Q-learning maintains two tables to reduce maximization bias from always taking max of noisy estimates.

### Dyna-Q

Model-based RL can mix real experience with simulated experience. Dyna-Q stores transitions in a simple model (dictionary of observed (s,a) -> (r,s’)), performs the usual Q-learning update on real experience, then for N planning steps samples previously seen (s,a) pairs and updates Q using the model. Even small N often accelerates learning dramatically in deterministic tabular domains. In stochastic domains the model should average outcomes; in large domains learned deep models (as in Dreamer) replace the tabular dictionary.

# Tabular Q-learning step
def q_learning_step(Q, s, a, r, sp, alpha=0.1, gamma=0.9, actions=range(4)):
td_target = r + gamma * max(Q[sp][a2] for a2 in actions)
Q[s][a] += alpha * (td_target - Q[s][a])
return Q

# Dyna-Q planning bonus after real update
def dyna_plan(Q, model, n_plan=10, alpha=0.1, gamma=0.9, actions=range(4)):
import random
keys = list(model.keys())
for _ in range(n_plan):
s, a = random.choice(keys)
r, sp = model[(s, a)]
q_learning_step(Q, s, a, r, sp, alpha, gamma, actions)
return Q

## 13.12 n-Step Methods, TD(lambda), and Eligibility Traces

One-step TD and Monte Carlo are endpoints of a spectrum. n-step returns use

G_t:t+n = R_{t+1} + gamma R_{t+2} + … + gamma^{n-1} R_{t+n} + gamma^n V(S_{t+n}),

trading bias and variance as n varies. TD(lambda) unifies these via lambda-returns, geometric mixtures of n-step returns. The forward view is conceptual; the backward view uses eligibility traces for efficient online updates.

### Eligibility Traces

Each state (or state-action pair) maintains a trace e(s) that spikes when visited and decays by gamma*lambda each step. The TD error delta updates all states in proportion to their traces: V(s) <- V(s) + alpha * delta * e(s). Accumulating traces add 1 on visits; replacing traces reset to 1. Traces implement multi-step credit assignment without waiting for episode end. Setting lambda=0 recovers one-step TD; lambda=1 approaches Monte Carlo (with appropriate implementations). Intermediate lambda often works best.

![Eligibility traces: decay, visits, and delayed TD credit (synthetic; original).](../assets/figures/ml_fig_eligibility_trace.png)

*Figure — Backward-view credit assignment. **Left:** accumulating traces \(e_t \leftarrow \gamma\lambda e_{t-1} + 1\{S_t=s\}\) for states A/B/C along a synthetic trajectory (\(\gamma=0.9\), \(\lambda=0.8\)); dashed line marks a delayed TD error \(\delta_{12}=+1\). **Right:** credit \(\sum_t \delta_t e_t(s)\) is largest for recently eligible states. \(\lambda=0\) recovers one-step TD; \(\lambda\to 1\) is MC-like; GAE in deep actor-critic uses the same bias–variance knob.*

In deep RL, classical eligibility traces are less common in pure form, but n-step returns and generalized advantage estimation (GAE) play related roles in actor-critic algorithms such as A3C and PPO. GAE computes advantages as an exponentially weighted mixture of multi-step TD residuals, controlled by a lambda-like parameter that again trades bias and variance.

![GAE λ mixes multi-step TD residuals (synthetic; original).](../assets/figures/ml_fig_gae_lambda.png)

*Figure — Generalized Advantage Estimation. **Left:** synthetic TD residuals \(\delta_t\). **Right:** GAE advantages for \(\lambda\in\{0,0.5,0.95,1\}\)—low \(\lambda\) is nearly one-step (higher bias, lower variance); \(\lambda\to 1\) is Monte-Carlo-like. Same bias–variance knob as eligibility traces; still not a license for bedside exploration with irreversible actions.*

## 13.13 Function Approximation and Continuous Spaces

Tabular methods store a number per state or state-action pair. In large or continuous spaces this is impossible. Function approximation represents V(s; w) or Q(s,a; w) with parameters w (linear features, trees, or neural networks) and updates w by gradient steps on a squared Bellman error or related loss. Linear TD with appropriate step-sizes has convergence theory; nonlinear approximators can diverge if care is not taken—the deadly triad of off-policy learning + bootstrapping + function approximation.

Feature design for linear approximators includes tile coding, radial basis functions, and Fourier bases. Neural networks free the engineer from manual features but require stabilization tricks discussed next. In continuous action spaces, argmax_a Q(s,a) is no longer a finite enumeration; policy gradient and actor-critic methods become natural. Semi-gradient methods treat the bootstrap target as a constant with respect to w, which is biased but often practical; true residual gradients have their own pathologies.

When deploying approximators, validate not only average return but failure modes: value overestimation, policy collapse, and poor coverage of rare but critical states (for example, hypotensive crises in a vital-sign MDP).

## 13.14 Deep Q-Networks: Replay, Target Networks, DDQN, Dueling, and PER

Deep Q-Networks (DQN) approximate Q(s,a; theta) with a deep network. Two core problems of naive deep RL are: (1) samples are temporally correlated, violating i.i.d. assumptions of stochastic gradient methods; (2) the bootstrap target r + gamma max_{a’} Q(s’,a’; theta) moves whenever theta moves, creating a non-stationary regression problem that can oscillate or diverge.

Experience replay stores transitions (s,a,r,s’,done) in a buffer and trains on random minibatches, breaking temporal correlation and reusing rare events. A target network theta^- is a lagged copy of theta, updated only periodically (or via slow Polyak averaging), used to form targets r + gamma max_{a’} Q(s’,a’; theta^-). Reward clipping and frame stacking help Atari-style inputs; convolutional towers process pixels.

### DQN Cost Function

Minimize the mean squared Bellman error on minibatches:

L(theta) = E[( r + gamma max_{a’} Q(s’,a’; theta^-) * (1-done) - Q(s,a; theta) )^2 ].

### Double DQN (DDQN)

Vanilla DQN’s max operator uses the same network to select and evaluate actions, causing maximization bias. DDQN selects a* = argmax_{a’} Q(s’,a’; theta) with the online net and evaluates Q(s’,a*; theta^-) with the target net, reducing overestimation.

### Dueling Networks

Separate streams estimate a state value V(s) and advantages A(s,a), recombined as Q(s,a) = V(s) + (A(s,a) - mean_{a’} A(s,a’)). This architecture helps when actions have similar values in many states and the main question is whether the state is good.

### Prioritized Experience Replay (PER)

Instead of sampling transitions uniformly, sample with probability proportional to TD-error magnitude (plus a small epsilon), so surprising transitions are replayed more often. Importance-sampling weights correct the bias introduced by non-uniform sampling. PER often speeds learning on sparse-reward tasks but needs tuning of priority exponent and IS annealing.

Later improvements (distributional RL, noisy nets, Rainbow combinations) refine return distributions and exploration. DQN-style methods suit discrete action spaces; continuous control usually needs policy-gradient or actor-critic methods.

## 13.15 Policy Gradient Methods: REINFORCE, TRPO, and PPO

Instead of learning Q and deriving a policy, policy gradient methods optimize J(theta) = E_{pi_theta}[G] directly. The REINFORCE identity gives

grad_theta J(theta) = E[ sum_t G_t grad_theta log pi_theta(A_t | S_t) ]

(or with a baseline b(S_t) subtracted from G_t to reduce variance without adding bias). REINFORCE is on-policy and high-variance but unbiased for the true gradient of expected return. It naturally handles continuous actions by parameterizing Gaussian or other densities. Softmax policies over discrete actions are equally natural.

### Advantages and Drawbacks of Policy Gradients

They optimize the quantity of interest directly; they can learn stochastic policies; they handle continuous high-dimensional actions; they avoid the hard argmax of Q-methods in continuous spaces. Disadvantages include high variance, sample inefficiency (on-policy), and sensitivity to step size—large updates can collapse the policy.

### Trust Region Policy Optimization (TRPO)

TRPO constrains policy updates so that the new policy does not move too far from the old one in KL divergence: maximize a surrogate advantage objective subject to E[KL(pi_old || pi_new)] <= delta. The theory uses minorization-maximization (MM) ideas and a surrogate loss that lower-bounds improvement when the trust region is respected. Practically, TRPO uses natural policy gradients with a Fisher-information matrix-vector product and conjugate gradient to solve the constrained problem, followed by a line search. Natural gradients precondition ordinary gradients by the Fisher metric so that steps are measured in distribution space rather than raw parameter space—important because two parameter vectors can induce very different or very similar policies depending on parameterization.

### Proximal Policy Optimization (PPO)

PPO approximates the trust-region idea with a simpler clipped surrogate objective. Let r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t) be the probability ratio. PPO maximizes

E[ min( r_t A_t , clip(r_t, 1-epsilon, 1+epsilon) A_t ) ],

which removes incentives to push r_t far outside [1-epsilon, 1+epsilon]. An alternative PPO formulation uses an adaptive KL penalty coefficient. PPO is typically implemented as an actor-critic model: a value head estimates V for advantage computation (often via GAE), and multiple epochs of minibatch SGD reuse each batch of on-policy data carefully. PPO became a default strong baseline for continuous and discrete control due to stability and implementation simplicity relative to TRPO.

![PPO clipped surrogate vs probability ratio r for A>0 and A<0 (original).](../assets/figures/ml_fig_ppo_clip.png)

*Figure — Why the clip exists. Horizontal axis is the importance ratio \(r=\pi_\theta(a|s)/\pi_{\mathrm{old}}(a|s)\); shaded band is \([1-\varepsilon,1+\varepsilon]\). **Left:** when advantage \(A>0\), the PPO min objective stops rewarding ever-larger \(r\) past \(1+\varepsilon\). **Right:** when \(A<0\), clipping limits how hard a single batch can drive probability mass toward zero. PPO is a trust-region heuristic for on-policy control—not a license for bedside exploration with irreversible actions.*


![ε-greedy exploration schedules: linear, exponential, constant (original).](../assets/figures/ml_fig_epsilon_decay.png)

*Figure — Exploration design. Linear anneal, exponential decay, and constant ε trade early search against later exploitation. Schedules are policy choices for simulated MDPs—not maps of clinical cause-effect paths. Constrain unsafe actions before any bedside trial.*


![TD control learning curves on a toy MDP (synthetic; original).](../assets/figures/ml_fig_td_control_curves.png)

*Figure — On- vs off-policy style learning curves toward Q*. Simulated value improvement is not a license to automate bedside policy. Constrain unsafe actions; **control algorithms ≠ proven clinical causation**.*


![Reward shaping vs sparse terminal reward learning curves (synthetic; original).](../assets/figures/ml_fig_reward_shaping.png)

*Figure — Shaping accelerates return curves but can open reward hacks. Simulated control performance is not a bedside license. **Policy learning ≠ proven clinical causation**.*


![Actor–critic co-training curves on a toy task (synthetic; original).](../assets/figures/ml_fig_actor_critic.png)

*Figure — Critic often stabilizes value estimates while actor return rises more slowly. Simulated curves are not a license for autonomous clinical control. **Pred/control ≠ causation**.*


![n-step return bias-variance teaching sketch (synthetic; original).](../assets/figures/ml_fig_nstep_bias_var.png)

*Figure — Longer n-step targets can reduce bias-like error while raising variance-like noise in teaching curves. TD design choices are algorithmic—not clinical causal pathways.*


![Policy entropy vs temperature (synthetic; original).](../assets/figures/ml_fig_policy_entropy.png)

*Figure — Exploration temperature changes entropy; not a clinical causal path. Pred != cause without design.*


![Replay buffer sample composition (original).](../assets/figures/ml_fig_replay_buffer.png)

*Figure — Off-policy reuse needs care for safety. Replay buffer sample composition Pred != cause without design.*


![discount teaching panel (original).](../assets/figures/ml_fig_discount_horizon.png)

*Figure — Teaching panel for discount. Pred != cause without design.*


![Cycle-34 densify scientific panel 15 (original).](../assets/figures/ml_fig_c34_14.png)

*Figure — Continuous densify panel 15. Synthetic teaching geometry—not a causal claim.*


![Cycle-35 densify scientific panel 15 (original).](../assets/figures/ml_fig_c35_14.png)

*Figure — Continuous densify panel 15. Synthetic teaching geometry—not a causal claim.*


![Cycle c36 densify panel 15 (original).](../assets/figures/ml_fig_c36_14.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![Cycle c37 densify panel 15 (original).](../assets/figures/ml_fig_c37_14.png)

*Figure — Continuous densify panel. Synthetic teaching geometry—not a causal claim.*


![c38 densify panel 15 (original).](../assets/figures/ml_fig_c38_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c39 densify panel 15 (original).](../assets/figures/ml_fig_c39_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c40 densify panel 15 (original).](../assets/figures/ml_fig_c40_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c41 densify panel 15 (original).](../assets/figures/ml_fig_c41_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c42 densify panel 15 (original).](../assets/figures/ml_fig_c42_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c43 densify panel 15 (original).](../assets/figures/ml_fig_c43_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c44 densify panel 15 (original).](../assets/figures/ml_fig_c44_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c45 densify panel 15 (original).](../assets/figures/ml_fig_c45_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c46 densify panel 15 (original).](../assets/figures/ml_fig_c46_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c47 densify panel 15 (original).](../assets/figures/ml_fig_c47_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c48 densify panel 15 (original).](../assets/figures/ml_fig_c48_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c49 densify panel 15 (original).](../assets/figures/ml_fig_c49_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c50 densify panel 15 (original).](../assets/figures/ml_fig_c50_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c51 densify panel 15 (original).](../assets/figures/ml_fig_c51_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c52 densify panel 15 (original).](../assets/figures/ml_fig_c52_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c53 densify panel 15 (original).](../assets/figures/ml_fig_c53_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c54 densify panel 15 (original).](../assets/figures/ml_fig_c54_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c55 densify panel 15 (original).](../assets/figures/ml_fig_c55_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c56 densify panel 15 (original).](../assets/figures/ml_fig_c56_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c57 densify panel 15 (original).](../assets/figures/ml_fig_c57_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c58 densify panel 15 (original).](../assets/figures/ml_fig_c58_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c59 densify panel 15 (original).](../assets/figures/ml_fig_c59_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c60 densify panel 15 (original).](../assets/figures/ml_fig_c60_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c61 densify panel 15 (original).](../assets/figures/ml_fig_c61_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c62 densify panel 15 (original).](../assets/figures/ml_fig_c62_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c63 densify panel 15 (original).](../assets/figures/ml_fig_c63_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c64 densify panel 15 (original).](../assets/figures/ml_fig_c64_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c65 densify panel 15 (original).](../assets/figures/ml_fig_c65_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c66 densify panel 15 (original).](../assets/figures/ml_fig_c66_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c67 densify panel 15 (original).](../assets/figures/ml_fig_c67_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c68 densify panel 15 (original).](../assets/figures/ml_fig_c68_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c69 densify panel 15 (original).](../assets/figures/ml_fig_c69_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c70 densify panel 15 (original).](../assets/figures/ml_fig_c70_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c71 densify panel 15 (original).](../assets/figures/ml_fig_c71_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c72 densify panel 15 (original).](../assets/figures/ml_fig_c72_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c73 densify panel 15 (original).](../assets/figures/ml_fig_c73_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c74 densify panel 15 (original).](../assets/figures/ml_fig_c74_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c75 densify panel 15 (original).](../assets/figures/ml_fig_c75_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c76 densify panel 15 (original).](../assets/figures/ml_fig_c76_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c77 densify panel 15 (original).](../assets/figures/ml_fig_c77_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c78 densify panel 15 (original).](../assets/figures/ml_fig_c78_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c79 densify panel 15 (original).](../assets/figures/ml_fig_c79_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c80 densify panel 15 (original).](../assets/figures/ml_fig_c80_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*


![c81 densify panel 15 (original).](../assets/figures/ml_fig_c81_14.png)

*Figure — Continuous densify. Synthetic teaching geometry—not a causal claim.*

# REINFORCE with baseline (conceptual)
# logp: log pi(a_t|s_t); returns: return-to-go; baseline: V(s_t)
def reinforce_loss(logp, returns, baseline):
adv = returns - baseline.detach()
return -(adv * logp).mean()

## 13.16 Actor-Critic Methods: A3C, DDPG, TD3, and SAC

Actor-critic methods maintain both a policy (actor) pi_theta and a value function (critic) V_w or Q_w. The critic supplies a low-variance target or advantage estimate A_t ≈ delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t) for the actor’s policy gradient. This hybrid often learns faster than pure REINFORCE.

### Asynchronous Advantage Actor-Critic (A3C)

Multiple workers each interact with their own environment copy, compute gradients of a shared actor-critic network asynchronously, and update shared parameters without a central replay buffer. Diversity of workers stabilizes learning; the advantage actor-critic (A2C) synchronous variant is often used in practice for simplicity. n-step returns are common in A3C implementations.

### Deep Deterministic Policy Gradient (DDPG)

For continuous actions, DDPG learns a deterministic actor mu_theta(s) and a critic Q_w(s,a) with replay and target networks (DQN-style). The actor is updated by gradient ascent on Q_w(s, mu_theta(s))—the deterministic policy gradient. Exploration adds noise (Ornstein-Uhlenbeck or Gaussian) to the actor’s action. DDPG can be fragile: overestimated Q values and brittle actor updates are common failure modes.

### Twin Delayed DDPG (TD3)

TD3 repairs DDPG with three ideas: (1) twin critics—train two Q networks and take the minimum for targets to curb overestimation; (2) delayed policy updates—update the actor less frequently than the critics; (3) target policy smoothing—add clipped noise to the target action when forming Bellman targets. These changes substantially stabilize continuous control.

### Soft Actor-Critic (SAC)

SAC is an off-policy maximum-entropy actor-critic. The objective maximizes expected return plus expected policy entropy: E[sum (r + alpha H(pi(.|s))) ]. Entropy regularization encourages exploration and robustness. Soft Q-values satisfy a soft Bellman equation with an entropy-augmented backup. SAC typically uses twin critics and target critics, a stochastic actor (Gaussian with reparameterization trick for low-variance gradients), and often automatic tuning of the temperature alpha. Reparameterization writes a = tanh(mu_theta(s) + sigma_theta(s) * epsilon), epsilon ~ N(0,I), so gradients flow through sampled actions. SAC is a strong default for many continuous-control benchmarks.

## 13.17 Dreamer Models: Learning Behaviors in Latent Imagination

Model-based RL learns a world model of transitions and rewards, then plans or optimizes a policy inside that model to reduce expensive real-environment interaction. Dreamer-style agents learn a latent dynamics model from pixels or sensory streams and train an actor-critic purely on imagined latent rollouts (“dreams”).

Straight-through gradients allow discrete latent variables to backpropagate approximate gradients. Dreamer v1 established the pattern of latent imagination for continuous control from pixels. Dreamer v2 improved discrete latents and world-model training for broader domains including Atari. Dreamer v3 aimed at robustness across domains with normalized returns, symlog predictions, and carefully scaled losses so that one set of hyperparameters works on many tasks.

Conceptually, Dreamer sits near Dyna: both leverage a learned model for additional updates. The difference is scale and representation—modern world models use recurrent state-space models and rich latent sequences rather than tabular (s,a)->(r,s’) dictionaries. Clinical caution: a world model trained on observational EHR dynamics will reproduce historical practice and confounding; planning inside such a model is not the same as identifying causal treatment effects. World models are promising for high-fidelity simulators and digital twins when validated, not for unsupervised autonomy over patients.

## 13.18 Reward Design Pitfalls

The reward function is a specification of desired behavior—and specifications go wrong. Reward hacking (specification gaming) occurs when the agent maximizes the literal reward in ways that violate the designer’s intent: a cleaning robot that hides dirt under a rug, a game agent that pauses forever to avoid dying if survival time is rewarded poorly, or a recommender that optimizes clicks by promoting outrage. Sparse rewards make learning hard; dense shaping rewards can speed learning but may change the optimal policy if not potential-based. Potential-based shaping restricts the added reward to the form F(s, s’) = gamma*Phi(s’) - Phi(s) for some potential function Phi over states. Because these bonus terms telescope along any trajectory, they cancel in every policy comparison and provably leave the set of optimal policies unchanged while still guiding early learning toward promising states. Shaping that is not of this form—for example, a flat bonus each time the agent moves nearer the goal—can silently install a new optimal policy that harvests the bonus (loitering near the goal) instead of solving the task; this is a common and easily missed source of reward hacking.

Misaligned proxies: optimizing an easy metric (time-on-site) instead of true utility (user wellbeing). Scale and shaping: poorly scaled rewards cause vanishing or exploding advantages in deep RL. Non-stationarity: if rewards depend on other agents or changing users, the MDP assumption weakens. Safety constraints: hard constraints (do not exceed torque limits) are often better as constrained MDPs or shields than as mild penalties the agent can trade off. Evaluation: always inspect trajectories qualitatively; high return is not sufficient evidence of desired behavior.

Good practice includes starting from the simplest reward that encodes the true objective, using constraints for safety, logging diverse rollouts, and treating reward design as an iterative engineering process with human review—not a one-shot hyperparameter.

Reward hacking: literal maximization without intended meaning.

Proxy failure: optimizing LOS, clicks, or billing codes instead of patient-centered outcomes.

Shaping risk: dense rewards that alter the optimal policy.

Constraint-first safety: hard shields beat soft penalties for irreversible harm.

## 13.19 Clinical and Epidemiologic Notes: Sequential Care Is High-Risk RL

Neurology and epidemiology confront sequential decisions daily: titration of antihypertensives after intracerebral hemorrhage, escalation of immunotherapy in neuroinflammatory disease, door-to-needle pathways for large-vessel occlusion, secondary-prevention medication sequences, and triage across emergency, intensive-care, and rehabilitation settings. These look like Markov decision processes—states, actions, delayed outcomes—and it is therefore tempting to apply reinforcement learning directly. That temptation is scientifically interesting and operationally dangerous. Unlike games or simulators, clinical trajectories are irreversible, sparsely labeled, confounded by indication, and ethically constrained: exploration means trying something uncertain on a real person.

### Reward Misspecification: The Primary Clinical Failure Mode

In supervised learning, a wrong label is a local error; in RL, a wrong reward rewrites the objective the agent optimizes everywhere. Examples that recur in neurologic care include rewarding short length of stay (which can encourage premature discharge), rewarding imaging utilization or procedure volume (which can inflate low-value care), rewarding binary survival at discharge while ignoring disability, cognitive outcome, caregiver burden, or equity of access, and rewarding protocol compliance as if compliance were identical to appropriateness for every phenotype.

### Worked Thought-Example of Misspecification

Suppose an intensive-care weaning policy is trained with reward +1 for each hour off mechanical ventilation and -10 for reintubation within 24 hours, with no term for delirium, aspiration, or long-term cognitive outcome. A greedy learner may learn aggressive early extubation that maximizes the +1 stream until the reintubation penalty is barely avoided on average in the training cohort. If reintubation is under-ascertained in the EHR (patients transferred, codes incomplete), the penalty is underestimated and the learned policy looks excellent offline while being unsafe. Reformulations include multi-objective returns, constrained MDPs with hard safety sets, potential-based shaping only when theory guarantees policy invariance, and human-in-the-loop vetoes.

### Offline RL Caution: Coverage and Confounding by Indication

Nearly all clinical data for sequential decisions are off-policy: they were generated by existing guidelines, individual clinician habits, bed availability, and unrecorded preferences—not by the target policy under study. Off-policy algorithms can in principle learn about alternative actions, but theory requires adequate coverage: every relevant state-action pair must appear often enough under behavior. In stroke and critical-care logs, rare but critical actions have near-zero support. Importance-sampling estimators become extremely high variance; naive fitted Q evaluation can be optimistically biased when the function approximator extrapolates to unsupported actions. Epidemiologic parallel: confounding by indication is the cohort-study cousin of off-policy bias. Patients who receive more aggressive therapy differ systematically in severity and goals of care. An RL agent that treats “action taken” as freely choosable without a credible model of why it was taken will rediscover indication, not invent better care.

### Ethics, Consent, and Exploration at the Bedside

Epsilon-greedy exploration is a textbook device; at the bedside it is an unconsented experiment unless embedded in an ethically approved learning health system, adaptive trial, or carefully monitored quality-improvement framework. Key ethical pressures include non-maleficence (unsafe random actions are unacceptable), justice (policies trained on tertiary-center data may under-serve community hospitals), autonomy (patient values differ), accountability (override authority and interpretability), and privacy (trajectories concatenate sensitive signals). Prefer supervised risk models, decision curves, and protocolized pathways when the decision is not truly sequential with long credit assignment. If sequential structure is essential, start with imitation of high-quality care or constrained contextual bandits under equipoise, not open-ended deep RL. Demand overlap diagnostics and conservative off-policy evaluation; plan prospective evaluation with safety monitoring before any automated action selection touches patients. Audit subgroup performance because policy regret is often concentrated in minorities of the state space.

Treat reward design as a multidisciplinary specification problem—document it like a trial endpoint.

Require coverage/overlap diagnostics before trusting offline value estimates.

Constrain exploration to safe action sets designed with domain experts.

Validate prospectively; retrospective return is not a license to automate care.

Audit equity: age, race/ethnicity, language, insurance, transferring hospital.

## 13.20 Putting the Pieces Together

A practical workflow for an RL problem is: (1) define observations, actions, episode termination, and reward; (2) build a simulator or logging pipeline; (3) choose a baseline that is not RL if possible (contextual bandit, model-predictive control with a model, imitation learning); (4) if sequential credit assignment is essential, start with a simple method—tabular Q-learning or DQN for discrete actions, PPO or SAC for continuous control; (5) monitor not only return but constraint violations and qualitative failure modes; (6) ablate exploration, discount, and network capacity; (7) for clinical domains, interrogate reward misspecification, off-policy coverage, and ethical constraints before any policy leaves the lab.

![13.6: A taxonomy of the reinforcement-learning algorithms surveyed in the chapter, organized into three families. Value-based ](../assets/figures/ml_concept_13.6_7c00d06e.png)

*Figure 13.6 — original teaching graphic.*

Understanding Bellman backups and on- versus off-policy learning is more valuable than memorizing every acronym: new algorithms almost always remix these ingredients. Relative to supervised learning, expect higher variance across seeds, greater sensitivity to implementation details, and a larger role for domain knowledge in reward and state design. Relative to planning with a perfect model, expect sample inefficiency when learning from scratch in high dimensions—hence the importance of simulation, demonstration data, and transfer.


![c82 teaching panel 14 (original).](../assets/figures/ml_fig_c82_14.png)
*Figure — Bandit cumulative regret: greedy lock-in vs ε-greedy vs UCB sketch. Synthetic teaching geometry—not a causal claim.*


![c83 teaching panel 14 (original).](../assets/figures/ml_fig_c83_14.png)
*Figure — Temporal-difference backup and TD error δ. Synthetic teaching geometry—not a causal claim.*


![c84 teaching panel 14 (original).](../assets/figures/ml_fig_c84_14.png)
*Figure — State values with greedy policy action arrows. Synthetic teaching geometry—not a causal claim.*


![c85 teaching panel 14 (original).](../assets/figures/ml_fig_c85_14.png)
*Figure — Discount factors shrink the value of distant rewards. Synthetic teaching geometry—not a causal claim.*


![c86 teaching panel 14 (original).](../assets/figures/ml_fig_c86_14.png)
*Figure — Q(s,a) table snapshot across discrete states. Synthetic teaching geometry—not a causal claim.*


![c87 teaching panel 14 (original).](../assets/figures/ml_fig_c87_14.png)
*Figure — RL episode return curve with variability band. Synthetic teaching geometry—not a causal claim.*


![c88 teaching panel 14 (original).](../assets/figures/ml_fig_c88_14.png)
*Figure — Eligibility trace decay in TD(λ). Synthetic teaching geometry—not a causal claim.*


![c89 teaching panel 14 (original).](../assets/figures/ml_fig_c89_14.png)
*Figure — Actor–critic dual network sketch. Synthetic teaching geometry—not a causal claim.*


![c90 teaching panel 14 (original).](../assets/figures/ml_fig_c90_14.png)
*Figure — Experience replay buffer sampling. Synthetic teaching geometry—not a causal claim.*


![c91 teaching panel 14 (original).](../assets/figures/ml_fig_c91_14.png)
*Figure — Softmax policy in RL. Synthetic teaching geometry—not a causal claim.*


![c92 teaching panel 14 (original).](../assets/figures/ml_fig_c92_14.png)
*Figure — Advantage normalization. Synthetic teaching geometry—not a causal claim.*


![c93 teaching panel 14 (original).](../assets/figures/ml_fig_c93_14.png)
*Figure — Intrinsic motivation bonus. Synthetic teaching geometry—not a causal claim.*


![c94 teaching panel 14 (original).](../assets/figures/ml_fig_c94_14.png)
*Figure — Entropy bonus for exploration. Synthetic teaching geometry—not a causal claim.*


![c95 teaching panel 14 (original).](../assets/figures/ml_fig_c95_14.png)
*Figure — GAE generalized advantage. Synthetic teaching geometry—not a causal claim.*


![c96 teaching panel 14 (original).](../assets/figures/ml_fig_c96_14.png)
*Figure — Count-based exploration bonus. Synthetic teaching geometry—not a causal claim.*


![c97 teaching panel 14 (original).](../assets/figures/ml_fig_c97_14.png)
*Figure — Max-entropy RL objective. Synthetic teaching geometry—not a causal claim.*


![c98 teaching panel 14 (original).](../assets/figures/ml_fig_c98_14.png)
*Figure — PPO clip objective cartoon. Synthetic teaching geometry—not a causal claim.*


![c99 teaching panel 14 (original).](../assets/figures/ml_fig_c99_14.png)
*Figure — RND prediction error bonus. Synthetic teaching geometry—not a causal claim.*


![c100 teaching panel 14 (original).](../assets/figures/ml_fig_c100_14.png)
*Figure — Successor features RL. Synthetic teaching geometry—not a causal claim.*


![c101 teaching panel 14 (original).](../assets/figures/ml_fig_c101_14.png)
*Figure — GRPO group relative policy. Synthetic teaching geometry—not a causal claim.*


![c102 teaching panel 14 (original).](../assets/figures/ml_fig_c102_14.png)
*Figure — ICM inverse curriculum. Synthetic teaching geometry—not a causal claim.*


![c103 teaching panel 14 (original).](../assets/figures/ml_fig_c103_14.png)
*Figure — Option-critic hierarchical RL. Synthetic teaching geometry—not a causal claim.*


![c104 teaching panel 14 (original).](../assets/figures/ml_fig_c104_14.png)
*Figure — DPO preference optimization. Synthetic teaching geometry—not a causal claim.*


![c105 teaching panel 14 (original).](../assets/figures/ml_fig_c105_14.png)
*Figure — Go-Explore archive return. Synthetic teaching geometry—not a causal claim.*


![c106 teaching panel 14 (original).](../assets/figures/ml_fig_c106_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c107 teaching panel 14 (original).](../assets/figures/ml_fig_c107_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c108 teaching panel 14 (original).](../assets/figures/ml_fig_c108_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c109 teaching panel 14 (original).](../assets/figures/ml_fig_c109_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c110 teaching panel 14 (original).](../assets/figures/ml_fig_c110_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c111 teaching panel 14 (original).](../assets/figures/ml_fig_c111_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c112 teaching panel 14 (original).](../assets/figures/ml_fig_c112_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c113 teaching panel 14 (original).](../assets/figures/ml_fig_c113_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c114 teaching panel 14 (original).](../assets/figures/ml_fig_c114_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c115 teaching panel 14 (original).](../assets/figures/ml_fig_c115_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c116 teaching panel 14 (original).](../assets/figures/ml_fig_c116_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c117 teaching panel 14 (original).](../assets/figures/ml_fig_c117_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c118 teaching panel 14 (original).](../assets/figures/ml_fig_c118_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c119 teaching panel 14 (original).](../assets/figures/ml_fig_c119_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c120 teaching panel 14 (original).](../assets/figures/ml_fig_c120_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c121 teaching panel 14 (original).](../assets/figures/ml_fig_c121_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c122 teaching panel 14 (original).](../assets/figures/ml_fig_c122_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c123 teaching panel 14 (original).](../assets/figures/ml_fig_c123_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c124 teaching panel 14 (original).](../assets/figures/ml_fig_c124_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c125 teaching panel 14 (original).](../assets/figures/ml_fig_c125_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c126 teaching panel 14 (original).](../assets/figures/ml_fig_c126_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c127 teaching panel 14 (original).](../assets/figures/ml_fig_c127_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c128 teaching panel 14 (original).](../assets/figures/ml_fig_c128_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c129 teaching panel 14 (original).](../assets/figures/ml_fig_c129_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c130 teaching panel 14 (original).](../assets/figures/ml_fig_c130_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c131 teaching panel 14 (original).](../assets/figures/ml_fig_c131_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c132 teaching panel 14 (original).](../assets/figures/ml_fig_c132_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c133 teaching panel 14 (original).](../assets/figures/ml_fig_c133_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c134 teaching panel 14 (original).](../assets/figures/ml_fig_c134_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c135 teaching panel 14 (original).](../assets/figures/ml_fig_c135_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c136 teaching panel 14 (original).](../assets/figures/ml_fig_c136_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c137 teaching panel 14 (original).](../assets/figures/ml_fig_c137_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c138 teaching panel 14 (original).](../assets/figures/ml_fig_c138_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c139 teaching panel 14 (original).](../assets/figures/ml_fig_c139_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c140 teaching panel 14 (original).](../assets/figures/ml_fig_c140_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c141 teaching panel 14 (original).](../assets/figures/ml_fig_c141_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c142 teaching panel 14 (original).](../assets/figures/ml_fig_c142_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c143 teaching panel 14 (original).](../assets/figures/ml_fig_c143_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c144 teaching panel 14 (original).](../assets/figures/ml_fig_c144_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c145 teaching panel 14 (original).](../assets/figures/ml_fig_c145_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c146 teaching panel 14 (original).](../assets/figures/ml_fig_c146_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c147 teaching panel 14 (original).](../assets/figures/ml_fig_c147_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c148 teaching panel 14 (original).](../assets/figures/ml_fig_c148_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c149 teaching panel 14 (original).](../assets/figures/ml_fig_c149_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c150 teaching panel 14 (original).](../assets/figures/ml_fig_c150_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c151 teaching panel 14 (original).](../assets/figures/ml_fig_c151_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c152 teaching panel 14 (original).](../assets/figures/ml_fig_c152_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c153 teaching panel 14 (original).](../assets/figures/ml_fig_c153_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c154 teaching panel 14 (original).](../assets/figures/ml_fig_c154_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c155 teaching panel 14 (original).](../assets/figures/ml_fig_c155_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c156 teaching panel 14 (original).](../assets/figures/ml_fig_c156_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c157 teaching panel 14 (original).](../assets/figures/ml_fig_c157_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c158 teaching panel 14 (original).](../assets/figures/ml_fig_c158_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c159 teaching panel 14 (original).](../assets/figures/ml_fig_c159_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c160 teaching panel 14 (original).](../assets/figures/ml_fig_c160_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c161 teaching panel 14 (original).](../assets/figures/ml_fig_c161_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c162 teaching panel 14 (original).](../assets/figures/ml_fig_c162_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c163 teaching panel 14 (original).](../assets/figures/ml_fig_c163_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c164 teaching panel 14 (original).](../assets/figures/ml_fig_c164_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c165 teaching panel 14 (original).](../assets/figures/ml_fig_c165_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c166 teaching panel 14 (original).](../assets/figures/ml_fig_c166_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c167 teaching panel 14 (original).](../assets/figures/ml_fig_c167_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c168 teaching panel 14 (original).](../assets/figures/ml_fig_c168_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c169 teaching panel 14 (original).](../assets/figures/ml_fig_c169_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c170 teaching panel 14 (original).](../assets/figures/ml_fig_c170_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c171 teaching panel 14 (original).](../assets/figures/ml_fig_c171_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c172 teaching panel 14 (original).](../assets/figures/ml_fig_c172_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c173 teaching panel 14 (original).](../assets/figures/ml_fig_c173_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c174 teaching panel 14 (original).](../assets/figures/ml_fig_c174_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c175 teaching panel 14 (original).](../assets/figures/ml_fig_c175_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c176 teaching panel 14 (original).](../assets/figures/ml_fig_c176_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c177 teaching panel 14 (original).](../assets/figures/ml_fig_c177_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c178 teaching panel 14 (original).](../assets/figures/ml_fig_c178_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c179 teaching panel 14 (original).](../assets/figures/ml_fig_c179_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c180 teaching panel 14 (original).](../assets/figures/ml_fig_c180_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c181 teaching panel 14 (original).](../assets/figures/ml_fig_c181_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c182 teaching panel 14 (original).](../assets/figures/ml_fig_c182_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c183 teaching panel 14 (original).](../assets/figures/ml_fig_c183_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c184 teaching panel 14 (original).](../assets/figures/ml_fig_c184_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c185 teaching panel 14 (original).](../assets/figures/ml_fig_c185_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c186 teaching panel 14 (original).](../assets/figures/ml_fig_c186_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c187 teaching panel 14 (original).](../assets/figures/ml_fig_c187_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c188 teaching panel 14 (original).](../assets/figures/ml_fig_c188_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c189 teaching panel 14 (original).](../assets/figures/ml_fig_c189_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c190 teaching panel 14 (original).](../assets/figures/ml_fig_c190_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c191 teaching panel 14 (original).](../assets/figures/ml_fig_c191_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c192 teaching panel 14 (original).](../assets/figures/ml_fig_c192_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c193 teaching panel 14 (original).](../assets/figures/ml_fig_c193_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c194 teaching panel 14 (original).](../assets/figures/ml_fig_c194_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c195 teaching panel 14 (original).](../assets/figures/ml_fig_c195_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c196 teaching panel 14 (original).](../assets/figures/ml_fig_c196_14.png)
*Figure — World models dreamer. Synthetic teaching geometry—not a causal claim.*


![c197 teaching panel 14 (original).](../assets/figures/ml_fig_c197_14.png)
*Figure — Model-based MPC loop. Synthetic teaching geometry—not a causal claim.*


![c198 teaching panel 14 (original).](../assets/figures/ml_fig_c198_14.png)
*Figure — Offline RL conservatism. Synthetic teaching geometry—not a causal claim.*


![c199 teaching panel 14 (original).](../assets/figures/ml_fig_c199_14.png)
*Figure — Distributional RL quantiles. Synthetic teaching geometry—not a causal claim.*


![c200 teaching panel 14 (original).](../assets/figures/ml_fig_c200_14.png)
*Figure — Successor representation. Synthetic teaching geometry—not a causal claim.*


![c201 teaching panel 14 (original).](../assets/figures/ml_fig_c201_14.png)
*Figure — Exploration bonus vs extrinsic. Synthetic teaching geometry—not a causal claim.*


![c202 teaching panel 14 (original).](../assets/figures/ml_fig_c202_14.png)
*Figure — Lambda-return TD-MC blend. Synthetic teaching geometry—not a causal claim.*


![c203 teaching panel 14 (original).](../assets/figures/ml_fig_c203_14.png)
*Figure — PPO clipped ratio objective. Synthetic teaching geometry—not a causal claim.*


![c204 teaching panel 14 (original).](../assets/figures/ml_fig_c204_14.png)
*Figure — UCB exploration bonus decay. Synthetic teaching geometry—not a causal claim.*


![c205 teaching panel 14 (original).](../assets/figures/ml_fig_c205_14.png)
*Figure — Eligibility trace decay events. Synthetic teaching geometry—not a causal claim.*


![c206 teaching panel 14 (original).](../assets/figures/ml_fig_c206_14.png)
*Figure — Retrace truncated importance. Synthetic teaching geometry—not a causal claim.*


![c207 teaching panel 14 (original).](../assets/figures/ml_fig_c207_14.png)
*Figure — Dueling Q value advantage. Synthetic teaching geometry—not a causal claim.*


![c208 teaching panel 14 (original).](../assets/figures/ml_fig_c208_14.png)
*Figure — CQL conservative Q backup. Synthetic teaching geometry—not a causal claim.*


![c209 teaching panel 14 (original).](../assets/figures/ml_fig_c209_14.png)
*Figure — Prioritized experience replay. Synthetic teaching geometry—not a causal claim.*


![c210 teaching panel 14 (original).](../assets/figures/ml_fig_c210_14.png)
*Figure — RND exploration bonus decay. Synthetic teaching geometry—not a causal claim.*


![c211 teaching panel 14 (original).](../assets/figures/ml_fig_c211_14.png)
*Figure — IMPALA v-trace truncation. Synthetic teaching geometry—not a causal claim.*


![c212 teaching panel 14 (original).](../assets/figures/ml_fig_c212_14.png)
*Figure — MuZero representation dynamics. Synthetic teaching geometry—not a causal claim.*


![c213 teaching panel 14 (original).](../assets/figures/ml_fig_c213_14.png)
*Figure — Double Q overestimation gap. Synthetic teaching geometry—not a causal claim.*


![c214 teaching panel 14 (original).](../assets/figures/ml_fig_c214_14.png)
*Figure — World model imagination rollout. Synthetic teaching geometry—not a causal claim.*


![c215 teaching panel 14 (original).](../assets/figures/ml_fig_c215_14.png)
*Figure — SAC temperature entropy tradeoff. Synthetic teaching geometry—not a causal claim.*


![c216 teaching panel 14 (original).](../assets/figures/ml_fig_c216_14.png)
*Figure — Decision Transformer RTG tokens. Synthetic teaching geometry—not a causal claim.*


![c217 teaching panel 14 (original).](../assets/figures/ml_fig_c217_14.png)
*Figure — TD-lambda bias variance trade. Synthetic teaching geometry—not a causal claim.*


![c218 teaching panel 14 (original).](../assets/figures/ml_fig_c218_14.png)
*Figure — IQL expectile regression loss. Synthetic teaching geometry—not a causal claim.*


![c219 teaching panel 14 (original).](../assets/figures/ml_fig_c219_14.png)
*Figure — Dreamer latent imagination. Synthetic teaching geometry—not a causal claim.*


![c220 teaching panel 14 (original).](../assets/figures/ml_fig_c220_14.png)
*Figure — R2D2 redistributed returns. Synthetic teaching geometry—not a causal claim.*


![c221 teaching panel 14 (original).](../assets/figures/ml_fig_c221_14.png)
*Figure — Option-critic temporal abstraction. Synthetic teaching geometry—not a causal claim.*


![c222 teaching panel 14 (original).](../assets/figures/ml_fig_c222_14.png)
*Figure — MuZero representation dynamics predict. Synthetic teaching geometry—not a causal claim.*


![c223 teaching panel 14 (original).](../assets/figures/ml_fig_c223_14.png)
*Figure — Successor features discount sum. Synthetic teaching geometry—not a causal claim.*


![c224 teaching panel 14 (original).](../assets/figures/ml_fig_c224_14.png)
*Figure — DreamerV3 imagination returns. Synthetic teaching geometry—not a causal claim.*


![c225 teaching panel 14 (original).](../assets/figures/ml_fig_c225_14.png)
*Figure — Distributional RL return laws. Synthetic teaching geometry—not a causal claim.*


![c226 teaching panel 14 (original).](../assets/figures/ml_fig_c226_14.png)
*Figure — PUCT exploration score terms. Synthetic teaching geometry—not a causal claim.*


![c227 teaching panel 14 (original).](../assets/figures/ml_fig_c227_14.png)
*Figure — IQL expectile residual loss. Synthetic teaching geometry—not a causal claim.*


![c228 teaching panel 14 (original).](../assets/figures/ml_fig_c228_14.png)
*Figure — AWAC advantage weights. Synthetic teaching geometry—not a causal claim.*


![c229 teaching panel 14 (original).](../assets/figures/ml_fig_c229_14.png)
*Figure — Conservative Q offline penalty. Synthetic teaching geometry—not a causal claim.*


![c230 teaching panel 14 (original).](../assets/figures/ml_fig_c230_14.png)
*Figure — TD3 clipped double Q. Synthetic teaching geometry—not a causal claim.*


![c231 teaching panel 14 (original).](../assets/figures/ml_fig_c231_14.png)
*Figure — CQL OOD Q landscape. Synthetic teaching geometry—not a causal claim.*


![c232 teaching panel 14 (original).](../assets/figures/ml_fig_c232_14.png)
*Figure — SAC temperature entropy trade. Synthetic teaching geometry—not a causal claim.*


![c233 teaching panel 14 (original).](../assets/figures/ml_fig_c233_14.png)
*Figure — REINFORCE baseline variance. Synthetic teaching geometry—not a causal claim.*


![c234 teaching panel 14 (original).](../assets/figures/ml_fig_c234_14.png)
*Figure — PPO clip ratio objective. Synthetic teaching geometry—not a causal claim.*


![c235 teaching panel 14 (original).](../assets/figures/ml_fig_c235_14.png)
*Figure — Advantage normalization scale. Synthetic teaching geometry—not a causal claim.*


![c236 teaching panel 14 (original).](../assets/figures/ml_fig_c236_14.png)
*Figure — TRPO KL trust region. Synthetic teaching geometry—not a causal claim.*


![c237 teaching panel 14 (original).](../assets/figures/ml_fig_c237_14.png)
*Figure — GAE lambda return scale. Synthetic teaching geometry—not a causal claim.*


![c238 teaching panel 14 (original).](../assets/figures/ml_fig_c238_14.png)
*Figure — DPO preference margin. Synthetic teaching geometry—not a causal claim.*


![c239 teaching panel 14 (original).](../assets/figures/ml_fig_c239_14.png)
*Figure — n-step return mix. Synthetic teaching geometry—not a causal claim.*


![c240 teaching panel 14 (original).](../assets/figures/ml_fig_c240_14.png)
*Figure — ORPO odds ratio path. Synthetic teaching geometry—not a causal claim.*


![c241 teaching panel 14 (original).](../assets/figures/ml_fig_c241_14.png)
*Figure — TD(lambda) eligibility mix. Synthetic teaching geometry—not a causal claim.*


![c242 teaching panel 14 (original).](../assets/figures/ml_fig_c242_14.png)
*Figure — KTO Kahneman-Tversky path. Synthetic teaching geometry—not a causal claim.*


![c243 teaching panel 14 (original).](../assets/figures/ml_fig_c243_14.png)
*Figure — Retrace operator scale. Synthetic teaching geometry—not a causal claim.*


![c244 teaching panel 14 (original).](../assets/figures/ml_fig_c244_14.png)
*Figure — IPO identity preference path. Synthetic teaching geometry—not a causal claim.*


![c245 teaching panel 14 (original).](../assets/figures/ml_fig_c245_14.png)
*Figure — Munchausen RL bonus scale. Synthetic teaching geometry—not a causal claim.*


![c246 teaching panel 14 (original).](../assets/figures/ml_fig_c246_14.png)
*Figure — SimPO simple preference path. Synthetic teaching geometry—not a causal claim.*


![c247 teaching panel 14 (original).](../assets/figures/ml_fig_c247_14.png)
*Figure — V-trace importance mix. Synthetic teaching geometry—not a causal claim.*


![c248 teaching panel 14 (original).](../assets/figures/ml_fig_c248_14.png)
*Figure — CPO constrained pref path. Synthetic teaching geometry—not a causal claim.*


![c249 teaching panel 14 (original).](../assets/figures/ml_fig_c249_14.png)
*Figure — Retrace C operator scale. Synthetic teaching geometry—not a causal claim.*


![c250 teaching panel 14 (original).](../assets/figures/ml_fig_c250_14.png)
*Figure — RRHF rank response path. Synthetic teaching geometry—not a causal claim.*


![c251 teaching panel 14 (original).](../assets/figures/ml_fig_c251_14.png)
*Figure — IMPALA V-trace scale. Synthetic teaching geometry—not a causal claim.*


![c252 teaching panel 14 (original).](../assets/figures/ml_fig_c252_14.png)
*Figure — SLiC sequence lik path. Synthetic teaching geometry—not a causal claim.*


![c253 teaching panel 14 (original).](../assets/figures/ml_fig_c253_14.png)
*Figure — n-step lambda mix. Synthetic teaching geometry—not a causal claim.*


![c254 teaching panel 14 (original).](../assets/figures/ml_fig_c254_14.png)
*Figure — ORPO odds path. Synthetic teaching geometry—not a causal claim.*


![c255 teaching panel 14 (original).](../assets/figures/ml_fig_c255_14.png)
*Figure — GAE lambda return scale. Synthetic teaching geometry—not a causal claim.*


![c256 teaching panel 14 (original).](../assets/figures/ml_fig_c256_14.png)
*Figure — IPO identity pref path. Synthetic teaching geometry—not a causal claim.*


![c257 teaching panel 14 (original).](../assets/figures/ml_fig_c257_14.png)
*Figure — Actor-critic residual c257. Synthetic teaching geometry—not a causal claim.*


![c258 teaching panel 14 (original).](../assets/figures/ml_fig_c258_14.png)
*Figure — Exploration noise path c258. Synthetic teaching geometry—not a causal claim.*


![c259 teaching panel 14 (original).](../assets/figures/ml_fig_c259_14.png)
*Figure — Reward shaping path c259. Synthetic teaching geometry—not a causal claim.*


![c260 teaching panel 14 (original).](../assets/figures/ml_fig_c260_14.png)
*Figure — Offline RL restraint path c260. Synthetic teaching geometry—not a causal claim.*


![c261 teaching panel 14 (original).](../assets/figures/ml_fig_c261_14.png)
*Figure — Preference reward path c261. Synthetic teaching geometry—not a causal claim.*


![c262 teaching panel 14 (original).](../assets/figures/ml_fig_c262_14.png)
*Figure — Bellman residual path c262. Synthetic teaching geometry—not a causal claim.*


![c263 teaching panel 14 (original).](../assets/figures/ml_fig_c263_14.png)
*Figure — TD error path c263. Synthetic teaching geometry—not a causal claim.*


![c264 teaching panel 14 (original).](../assets/figures/ml_fig_c264_14.png)
*Figure — Q-learning update path c264. Synthetic teaching geometry—not a causal claim.*


![c265 teaching panel 14 (original).](../assets/figures/ml_fig_c265_14.png)
*Figure — Policy gradient path c265. Synthetic teaching geometry—not a causal claim.*


![c266 teaching panel 14 (original).](../assets/figures/ml_fig_c266_14.png)
*Figure — Advantage estimate path c266. Synthetic teaching geometry—not a causal claim.*


![c267 teaching panel 14 (original).](../assets/figures/ml_fig_c267_14.png)
*Figure — PPO clip path c267. Synthetic teaching geometry—not a causal claim.*


![c268 teaching panel 14 (original).](../assets/figures/ml_fig_c268_14.png)
*Figure — TRPO KL path c268. Synthetic teaching geometry—not a causal claim.*


![c269 teaching panel 14 (original).](../assets/figures/ml_fig_c269_14.png)
*Figure — SAC entropy path c269. Synthetic teaching geometry—not a causal claim.*


![c270 teaching panel 14 (original).](../assets/figures/ml_fig_c270_14.png)
*Figure — DQN target lag path c270. Synthetic teaching geometry—not a causal claim.*


![c271 teaching panel 14 (original).](../assets/figures/ml_fig_c271_14.png)
*Figure — n-step return path c271. Synthetic teaching geometry—not a causal claim.*


![c272 teaching panel 14 (original).](../assets/figures/ml_fig_c272_14.png)
*Figure — Eligibility trace path c272. Synthetic teaching geometry—not a causal claim.*


![c273 teaching panel 14 (original).](../assets/figures/ml_fig_c273_14.png)
*Figure — Actor-critic residual c273. Synthetic teaching geometry—not a causal claim.*


![c274 teaching panel 14 (original).](../assets/figures/ml_fig_c274_14.png)
*Figure — Exploration noise path c274. Synthetic teaching geometry—not a causal claim.*


![c275 teaching panel 14 (original).](../assets/figures/ml_fig_c275_14.png)
*Figure — Reward shaping path c275. Synthetic teaching geometry—not a causal claim.*


![c276 teaching panel 14 (original).](../assets/figures/ml_fig_c276_14.png)
*Figure — Offline RL restraint path c276. Synthetic teaching geometry—not a causal claim.*


![c277 teaching panel 14 (original).](../assets/figures/ml_fig_c277_14.png)
*Figure — Preference reward path c277. Synthetic teaching geometry—not a causal claim.*

## Chapter Summary

Reinforcement learning studies agents that interact with environments to maximize cumulative reward. Markov decision processes formalize states, actions, transitions, rewards, and discounting. Policies induce value functions V and Q that satisfy Bellman equations; dynamic programming solves small known MDPs via policy and value iteration. Monte Carlo methods average full returns; temporal-difference methods bootstrap, yielding SARSA (on-policy), Q-learning (off-policy), and Dyna-Q planning. Multi-armed bandits isolate exploration via epsilon-greedy, UCB, and Thompson sampling. Function approximation and deep RL scale these ideas: DQN with replay and target networks, DDQN, dueling architectures, prioritized replay, REINFORCE, TRPO, PPO, and actor-critic methods including A3C, DDPG, TD3, and SAC, plus latent world-model Dreamer agents. Careful reward design is essential. Clinical and epidemiologic notes emphasize that sequential care is high-risk RL: reward misspecification rewrites the clinical objective, offline logs lack coverage and are confounded by indication, and ethical constraints sharply limit unsupervised exploration at the bedside.

## Practice and Reflection

(1) For the two-state MDP in Section 13.6 with gamma = 0.5, recompute V* and pi* analytically. How does the optimal action in s1 change compared with gamma = 0.9?

(2) Implement value iteration for a 4x4 gridworld with a single terminal goal (+1) and step cost -0.04. Plot V and the greedy policy for gamma in {0.5, 0.9, 0.99}.

(3) Prove that if pi’ is greedy with respect to Q^pi, then V^{pi’}(s) >= V^pi(s) for all s (policy improvement theorem sketch).

(4) Compare SARSA and Q-learning on a cliff-walking grid: explain why on-policy SARSA may prefer safer paths under epsilon-greedy behavior.

(5) Derive the REINFORCE gradient for a softmax policy over two actions. Show that subtracting a state-dependent baseline does not change the expected gradient.

(6) Explain how TD3’s twin critics and delayed policy updates address failure modes of DDPG.

(7) Design a reward for a recycling robot and then describe one plausible reward-hacking behavior. Propose a constraint that mitigates it.

(8) Clinical critique: propose a reward for post-stroke blood-pressure management in the first 72 hours. Identify two misspecification risks and an off-policy evaluation diagnostic you would require before trusting a learned policy on EHR trajectories.

(9) Ethics scenario: an offline RL system recommends a rarely used medication sequence supported by n = 12 historical patients. Argue for or against deployment using coverage, justice, and accountability criteria.
