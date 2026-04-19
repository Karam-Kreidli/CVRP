# Code Explanation

## 1. System Overview

Our system is an **RL-guided CVRP solver** built for the GECCO 2026 ML4VRP competition. Instead of solving the CVRP directly, we train a Reinforcement Learning agent to make **fleet-target strategy decisions** — choosing whether to push for fewer vehicles, lock the current fleet, or try a new random seed at each step of the search.

**Competition Objective:**
```
Score = 1000 * NV + TD    (lower is better)
```
- **NV** = Number of Vehicles (fleet size)
- **TD** = Total Distance traveled

Removing one vehicle saves 1000 distance units — fleet minimization dominates.

---

## 2. Architecture: Three Stages

The system has three main components that work together in a pipeline:

```
.vrp Instance  -->  Feature Extraction  -->  Fleet Manager  -->  HGS-CVRP Solver
            (12 hand-crafted)        (Stage 2)           (Stage 3)
                                                                      |
                                                                      v
                                                              PPO Training Loop
                                                                 (Stage 5)
```

### Feature Extraction (`Model/Solver_Engine.py`)

**What it does:** Computes 12 hand-crafted features from a CVRP instance that capture scale, demand structure, and spatial geometry. These replace the previous GNN encoder with low-cost, non-learned signals that are immediately useful to the policy (with stochastic customer subsampling for pairwise-distance terms on large instances).

| # | Feature | Formula | What it captures |
|---|---|---|---|
| 1 | size_norm | num_customers / 400 | Instance size prior |
| 2 | demand_fill_ratio | total_demand / (nv_min * capacity), nv_min = ceil(total_demand / capacity) | How close the instance is to capacity saturation |
| 3 | mean_dist_norm | mean(pairwise_customer_distances) / max_pairwise_distance | Typical customer spacing |
| 4 | std_dist_norm | std(pairwise_customer_distances) / max_pairwise_distance | Spatial variability (clustered vs dispersed) |
| 5 | depot_centrality | mean(depot_to_customer_distances) / max_pairwise_distance | Depot offset from customer mass |
| 6 | demand_cv | min((std(customer_demands) / max(mean(customer_demands), eps)), 2.0) / 2.0 | Demand heterogeneity with clipped normalization |
| 7 | capacity_tightness | max(customer_demands) / capacity | Largest single-stop load pressure |
| 8 | demand_minmax_ratio | min(customer_demands) / max(max(customer_demands), eps) | Demand imbalance from lightest to heaviest stop |
| 9 | top3_demand_share | sum(top3_customer_demands) / max(total_demand, eps) | Load concentration in a few heavy customers |
| 10 | depot_distance_cv | min((std(depot_distances) / max(mean(depot_distances), eps)), 2.0) / 2.0 | Radial spread irregularity |
| 11 | bbox_aspect_ratio | min(bbox_width, bbox_height) / max(max(bbox_width, bbox_height), eps) | Geometry anisotropy (elongated vs compact footprint) |
| 12 | radial_outlier_ratio | mean(depot_distances > (Q3 + 1.5 * IQR)), IQR = Q3 - Q1 | Fraction of customers that are radial outliers |

Computed once per instance at episode start, then reused for all 50 steps. Pairwise distance features use all customers for N <= 200 and a 200-customer subsample above that threshold.

---

### Stage 2 — Fleet Manager (`Model/Agent_Manager.py`)

**What it does:** The RL agent — the "brain" of the system. At each step, it looks at 12 instance features + 7 real-time solver statistics and chooses one of 10 fleet-target strategies.

**Observation (19-dim vector):**
```
obs = [instance_features (12) | solver_stats (7)]
```

The 7 solver statistics:
- **time_ratio** = steps_used / 50 — "How much budget is left?"
- **nv_ratio** = NV_current / NV_initial — "How much fleet reduction so far?"
- **score_ratio** = best_score / initial_score — "How much overall improvement?"
- **stagnation_ratio** = iters_no_improve / budget — "Am I stuck?"
- **nv_gap** = (best_nv - nv_min) / NV_initial — "How far from minimum fleet?"
- **last_reward** = clipped last reward / 10 — "Did the previous action help?"
- **last_action_norm** = last_action / NUM_ACTIONS — "What did I just try?"

**Action Space (10 discrete actions):**

Each action controls three dimensions: **fleet target** (how many vehicles to allow), **seed strategy** (same or new random seed), and **iteration budget** (how much compute to spend):

| # | Action | Fleet Target | Seed | Iters | Strategy |
|---|---|---|---|---|---|
| 0 | FREE_SAME | Unconstrained | Same | 500 | Quick solve, no fleet pressure |
| 1 | FREE_NEW | Unconstrained | New | 500 | Fresh start, no fleet pressure |
| 2 | LOCK_SAME | Lock best NV | Same | 500 | Optimize TD at current fleet size |
| 3 | LOCK_NEW | Lock best NV | New | 500 | Fresh start at current fleet size |
| 4 | PUSH_SAME | best_nv - 1 | Same | 1000 | Try to remove one vehicle |
| 5 | PUSH_NEW | best_nv - 1 | New | 1000 | Fresh start, remove one vehicle |
| 6 | FORCE_MIN | nv_min | New | 1500 | Force theoretical minimum fleet |
| 7 | FREE_DIVERSE_NEW | Unconstrained | New | 500 | Diversity-focused HGS settings without extra iteration cost |
| 8 | LOCK_AGGR_NEW | Lock best NV | New | 500 | Aggressive local-search settings at fixed fleet size |
| 9 | PUSH_BALANCED_NEW | best_nv - 1 | New | 1000 | Balanced HGS settings for push attempts |

The key insight: removing one vehicle saves 1000 in score, but forcing fewer vehicles can fail entirely. The agent must learn **when** pushing is worth the risk.

**Network:** Actor-Critic with shared trunk
```
obs (19) --> Linear(19->64) + ReLU --> Linear(64->64) + ReLU
                                     |
                        +---------------+---------------+
                        |                               |
                    Actor Head                     Critic Head
                   Linear(64->10)                 Linear(64->1)
                        |                               |
                 action_logits (10)              state_value (1)
```

- **Actor** outputs a probability distribution over the 10 actions (the policy)
- **Critic** outputs a single number estimating "how good is this state?" (the value function)
- **Parameters:** ~6,155 — intentionally tiny since it makes strategic decisions, not route computations

**Action Masking:** When the fleet is already at the theoretical minimum (NV <= NV_min = ceil(total_demand / capacity)), actions 4, 5, 6, 9 (PUSH_SAME, PUSH_NEW, FORCE_MIN, PUSH_BALANCED_NEW) are blocked by setting their logits to -10,000. After softmax, these become ~0 probability, preventing the agent from attempting impossible fleet reductions.

---

### Stage 3 — Solver Engine (`Model/Solver_Engine.py`)

**What it does:** The bridge between the RL world and the actual solver. Implements a Gymnasium environment that wraps HGS-CVRP (via the `hygese` Python package).

**Episode lifecycle:**
1. `reset()` — Pick a random .vrp instance, compute hand-crafted features, run an initial 500-iteration solve with default HGS parameters
2. `step(action)` — Translate the action to (fleet_target, seed, iteration_budget), run a **fresh** 500-1500 iteration solve, track the best solution found
3. Repeat `step()` 50 times = **~25,000-33,000 total HGS iterations** per episode

**Why 500 iterations per step?**

At 5,000 iterations (the previous design), HGS had essentially converged — the population stabilized and different action choices tended to collapse to similar outcomes. The agent got weak reward signal and couldn't learn. At 500 iterations, HGS is still actively searching, so strategy differences produce meaningfully different outcomes.

**How actions become solver parameters:**
The Fleet Manager doesn't touch routes directly. It controls two key levers:
- **num_vehicles** — Constrains how many vehicles HGS can use. Setting this to `best_nv - 1` forces HGS to find a solution with one fewer vehicle. If it succeeds, that's a 1000-point score improvement. If it fails, the result is discarded.
- **seed** — Same seed = reproducible search. New seed = different starting point (escape local optima).
- **iteration budget** — Harder tasks (PUSH, FORCE) get more iterations (1000-1500) since finding feasible solutions with fewer vehicles requires deeper search.
- **search bias knobs (for selected new actions)** — Additional HGS parameters (population/local-search settings) let the policy choose alternate search styles at the same iteration budget.

**Important: No warm starting.** HGS does not accept a previous solution as input. Each step runs a completely fresh solve. The environment tracks the best solution found across all steps in the episode.

**Reward:**
```python
if fleet_exploded:
    reward = -5.0                           # NV spiked by >2 or constrained solve failed
elif cand_score < best_score:
    pct = (best_score - cand_score) / best_score
    reward = pct * 100.0                    # percentage improvement over episode best
else:
    reward = -0.5                           # small cost for no improvement
```

Key properties:
- Compares against **episode best** (not the previous step's candidate) — reward is only positive when a new best solution is found, giving an unambiguous signal
- **Percentage-based** — normalizes across different instance sizes
- Small negative for no improvement teaches the agent to avoid wasteful actions

**Safety: Fleet Explosion / Failed-Solve Detection.** If an aggressive action causes NV to spike by more than 2 vehicles, or if the constrained solve fails/infeasible, the candidate is rejected and a penalty of -5.0 is returned.

---

### Stage 5 — PPO Training Loop (`Model/Train.py`)

**What it does:** Trains the Fleet Manager using Proximal Policy Optimization (PPO), one of the most popular and stable RL algorithms.

---

## 3. Reinforcement Learning Concepts

### What is RL?

Reinforcement Learning is a paradigm where an **agent** learns to make decisions by interacting with an **environment**. The agent observes a **state**, takes an **action**, receives a **reward**, and transitions to a new state. The goal is to learn a **policy** (a mapping from states to actions) that maximizes cumulative reward over time.

In our system:
- **Agent** = Fleet Manager
- **Environment** = CVRPEnv (HGS-CVRP solver wrapper)
- **State** = 19-dim observation vector (instance features + solver stats)
- **Action** = One of 10 fleet-target strategies (original 7 + 3 HGS-bias variants)
- **Reward** = Percentage improvement over episode best (or -0.5 / -5.0 penalties)

### Policy and Value Functions

**Policy (pi):** A probability distribution over actions given a state. The Actor head outputs this. During training, actions are *sampled* from this distribution (enabling exploration). During evaluation, we pick the *argmax* (greedy, no randomness).

**Value Function V(s):** The Critic head's estimate of "how much total future reward can we expect from this state?" This is used to compute *advantages* — how much better an action was compared to what we expected.

### Actor-Critic Architecture

Our Fleet Manager uses an **Actor-Critic** design with a **shared trunk**:

- The **shared trunk** (two 64-unit layers) extracts features common to both policy and value estimation. Sharing is efficient and keeps the critic's estimates aligned with the actor's policy.
- The **Actor** head outputs logits for each of the 10 actions. These are converted to probabilities by the Categorical distribution (softmax internally).
- The **Critic** head outputs a scalar V(s). PPO uses this to compute advantages.

### Advantage Estimation (GAE-lambda)

An **advantage** A(s, a) measures "how much better was action *a* in state *s* compared to the average action in that state?"
- Positive advantage = action was better than expected
- Negative advantage = action was worse than expected

We use **Generalized Advantage Estimation (GAE-lambda)** to compute advantages. There are two extremes:
- **Monte Carlo (lambda=1.0):** Use actual cumulative returns. Unbiased but high variance (noisy).
- **TD(0) (lambda=0.0):** Use one-step bootstrapped returns. Low variance but biased (depends on critic accuracy).

GAE-lambda interpolates between these. We use **lambda=0.90**, getting strong variance reduction while staying reasonably unbiased.

**The formula (working backwards from the last step):**
```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)     [TD error at step t]
A_t     = delta_t + gamma * lambda * A_{t+1}      [GAE accumulation]
```

The **discount factor gamma=0.95** controls how much future rewards matter. Over 50 steps, the last step's reward is worth 0.95^49 ~ 0.08 of its face value — appropriate since late-episode improvements are smaller and less predictable.

---

## 4. The Loss Function

PPO's total loss combines three components:

```
Loss = Policy Loss + 0.5 * Value Loss - 0.02 * Entropy Bonus
```

### Policy Loss (Clipped Surrogate Objective)

This is the core of PPO. The idea: we want to make good actions more likely and bad actions less likely, but we can't change the policy too much in one update or training becomes unstable.

**Step 1 — Compute the probability ratio:**
```
ratio = pi_new(a|s) / pi_old(a|s)
```
This measures how much more (or less) likely the current policy is to take this action compared to when the data was collected.

**Step 2 — Two surrogate objectives:**
```
surr1 = ratio * advantage              (unconstrained)
surr2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage  (constrained)
```
The clip (epsilon=0.2) bounds the ratio to [0.8, 1.2], limiting how much the policy can change.

**Step 3 — Take the minimum (pessimistic bound):**
```
policy_loss = -min(surr1, surr2)
```
This prevents the policy from changing too drastically in a single update. If the ratio drifts outside [0.8, 1.2], the clipped version takes over and blocks further change in that direction.

### Value Loss

```
value_loss = MSE(V_predicted(s), returns)
```
The critic learns to predict the actual returns (advantages + values). This is a standard mean squared error regression loss. It's weighted by **vf_coeff=0.5** in the total loss.

### Entropy Bonus

```
entropy_bonus = mean(entropy(policy))
```
Entropy measures how "spread out" the policy distribution is. High entropy = agent picks many different actions (exploring). Low entropy = agent always picks the same action (exploiting, but possibly stuck).

The entropy bonus (weighted by **ent_coeff=0.02**) is *subtracted* from the loss (since we minimize loss, subtracting encourages higher entropy). This prevents the policy from "collapsing" to always picking one action too early in training.

**What to watch:** If entropy drops to 0, the agent has stopped exploring — this is bad. Healthy training maintains entropy above ~0.5.

---

## 5. The Optimization Algorithm

### Adam Optimizer

We use **Adam** (Adaptive Moment Estimation), the standard optimizer for deep learning and RL. Adam maintains per-parameter running averages of:
- First moment (mean of gradients) — momentum
- Second moment (mean of squared gradients) — adaptive learning rate

This makes it more robust than vanilla SGD — parameters with noisy gradients get smaller updates, while parameters with consistent gradients get larger updates.

**Learning Rate:** Starts at **1e-4** and decays linearly to **5e-5** over training.

### Gradient Clipping

```
clip_grad_norm_(parameters, max_norm=0.5)
```
If the gradient norm exceeds 0.5, all gradients are scaled down proportionally. This prevents **exploding gradients**.

### FP16 Mixed Precision

PyTorch's Automatic Mixed Precision (AMP) runs the forward and backward passes in FP16 (half precision) on GPU. This roughly halves memory usage and can speed up computation. A **GradScaler** dynamically scales the loss to prevent underflow in FP16 gradients.

### KL Divergence Early Stopping

After each mini-epoch, we check how much the policy has changed from the old policy using KL divergence. If **KL > 1.5 * target_kl (0.015)**, we stop the PPO update early to prevent catastrophic policy changes.

---

## 6. Reward Processing

**Reward clipping:** Rewards are clipped to [-10, +10] before computing advantages. The percentage-based reward design keeps values in a reasonable range naturally:
- Typical improvement reward: +0.1 to +5.0
- No improvement: -0.5
- Fleet explosion: -5.0

Simple clipping is sufficient — no running statistics or normalization needed.

---

## 7. Curriculum Learning

Training starts with **easier instances** and gradually introduces harder ones:

- **Epochs 1-20:** Only small instances (N <= 100 customers)
- **Epochs 21+:** All instances (N up to 400 customers)

**Why?** The agent learns basic strategies (which configurations work when) faster on small, quick-to-solve instances. Once it has these fundamentals, it transfers this knowledge to larger, harder instances.

---

## 8. Training Loop Summary

Each epoch follows this cycle:

**Phase 1 — Collect Experience (8 episodes x 50 steps = 400 transitions)**
- Reset environment with a random instance
- Agent observes state, samples action (stochastic — with exploration)
- Environment executes action (fresh HGS solve, 500-1500 iters), returns reward
- Store all transitions in a rollout buffer

**Phase 2 — PPO Update (3 mini-epochs over the 400 transitions)**
- Normalize advantages to zero mean, unit variance
- Shuffle transitions into mini-batches of 128
- For each mini-batch: compute policy loss + value loss + entropy bonus, do gradient step
- Check KL divergence after each mini-epoch, stop early if too large

**Phase 3 — Evaluate (fixed eval set + optional holdout, greedy action selection)**
- No exploration — pure argmax action selection
- Same fixed eval instances each epoch for consistent progress tracking (default count = 5)
- Optional holdout evaluation every N epochs (if configured)
- Save the model using the selected tracking metric: `eval`, `holdout`, or `composite`

**Key metrics:**
- **Sel[metric]** (the real model-selection metric) — should trend downward over time
- **Eval score** — fixed-set progress signal (always useful for trend reading)
- **Holdout score** — out-of-sample signal when holdout is enabled
- **AvgScore** — noisy because of random instances, mostly ignore
- **Entropy** — should stay above ~0.5 (agent still exploring)
- **Policy loss** — small values are normal

---

## 9. Portfolio Baseline (`Baseline/Portfolio_Solver.py`)

A non-RL safety net. Systematically tries **11 HGS parameter configurations x N random seeds** per instance, keeping the best result. No learning involved — pure brute-force search over the config space.

Current defaults in `Baseline/Portfolio_Solver.py` are `num_seeds=5`, `nb_iter=10000`, and configurable `workers` for parallelism.

`Baseline/Baseline_Evaluation.py` complements this with two reporting modes:
- Single-solve baseline (fair RL comparison): one default and one large-pop solve per instance.
- Best-of-N baseline (upper bound): configurable `num_seeds x num_steps` (defaults `3 x 1`) with `nb_iter=25000`.

This guarantees competition-ready solutions regardless of whether RL training succeeds.

---

## 10. Competition-Oriented Inference Reporting (`Baseline/Infer.py`)

`Baseline/Infer.py` now supports a competition-facing evaluation mode when `--baseline` is enabled:

- Per-instance RL vs baseline rows with tie-aware outcomes (`win/loss/tie`) at **3-decimal score precision**.
- Pairwise Formula-1 surrogate points per instance: **10** for winner, **8** for second, tie -> **9/9** split.
- Aggregate summaries: win/loss/tie counts and rates, absolute/relative deltas, and point totals/margins.
- Hidden-subset robustness simulation via bootstrap sampling (`--subset_size`, `--bootstrap_trials`, `--bootstrap_seed`).
- Artifact export to `Results/` by default:
    - `<stem>_Instances.csv`
    - `<stem>_Summary.json`
    - `<stem>_Report.md`
- RL route export to `Solutions/HGS+RL/`:
    - `sol-Format/<instance>.sol`
    - `txt-Format/<instance>.txt`

Important interpretation note: this points view is a **pairwise RL-vs-baseline surrogate**, not the final multi-competitor competition ranking.

---

## 11. File Map

| File | Role | Parameters |
|------|------|-----------|
| `Model/Agent_Manager.py` | Fleet Manager — RL agent (Actor-Critic) | ~6,155 |
| `Model/Solver_Engine.py` | CVRPEnv — Gymnasium environment wrapping HGS-CVRP | — |
| `Model/Train.py` | PPO training loop, GAE, reward clipping | — |
| `Model/main.py` | Entry point, smoke tests, CLI | — |
| `Baseline/Infer.py` | RL inference + baseline comparison + competition-oriented reports | — |
| `Baseline/Baseline_Evaluation.py` | HGS baseline evaluation (default vs large-pop) | — |
| `Baseline/Portfolio_Solver.py` | Deterministic portfolio baseline (11 configs x N seeds) | — |

**Total trainable parameters: ~6,155.** The model is intentionally lightweight. The heavy lifting is done by HGS-CVRP's C++ solver — our RL agent just learns to steer it effectively.
