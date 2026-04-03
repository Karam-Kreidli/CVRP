# Code Explanation

## 1. System Overview

Our system is an **RL-guided CVRP solver** built for the GECCO 2026 ML4VRP competition. Instead of solving the CVRP directly, we train a Reinforcement Learning agent to **select HGS-CVRP algorithm configurations** — deciding which genetic algorithm parameters to use at each step of the search.

**Competition Objective:**
```
Score = 1000 * NV + TD    (lower is better)
```
- **NV** = Number of Vehicles (fleet size)
- **TD** = Total Distance traveled

Removing one vehicle saves 1000 distance units — fleet minimization dominates.

---

## 2. Architecture: Four Stages

The system has four main components that work together in a pipeline:

```
.vrp Instance  -->  GNN Encoder  -->  Fleet Manager  -->  HGS-CVRP Solver
                    (Stage 1)         (Stage 2)           (Stage 3)
                                                              |
                                                              v
                                                      PPO Training Loop
                                                         (Stage 5)
```

### Stage 1 — GNN Encoder (`model_vision.py`)

**What it does:** Reads a CVRP instance (customer locations + demands) and compresses it into a 128-dimensional vector — a "fingerprint" of the problem's spatial structure.

**Architecture:** Graph Attention Network (GAT)

1. **Input:** Each node (depot + customers) gets 3 features: `[x_normalized, y_normalized, demand/capacity]`
2. **Graph Construction:** k-NN graph with k=20 nearest neighbors (Euclidean distance, undirected). This keeps complexity at O(N*k) instead of O(N^2) for fully connected graphs.
3. **3 GAT Layers:** Each layer has 8 attention heads producing 16-dim outputs (concatenated to 128-dim). Each layer includes:
   - **GATConv** — multi-head attention over the k-NN graph. Each node aggregates information from its neighbors, weighted by learned attention scores. Different heads learn different spatial relationships.
   - **BatchNorm** — normalizes activations to stabilize training
   - **ELU activation** — non-linearity (like ReLU but smoother for negative values)
   - **Residual connection** — adds the layer's input back to its output (`x = x + x_residual`), preventing gradient degradation through deeper layers
4. **Mean Pooling:** Averages all node embeddings into a single 128-dim graph-level vector

**Why GAT?** Standard GNNs treat all neighbors equally. GAT uses attention to learn *which* neighbors matter most — e.g., a cluster of high-demand customers near capacity limits is more important than spread-out low-demand nodes.

**Parameters:** ~51,600. Computed once per instance, then reused for all 5 steps of an episode.

---

### Stage 2 — Fleet Manager (`agent_manager.py`)

**What it does:** The RL agent — the "brain" of the system. At each step, it looks at the graph embedding + 4 real-time solver statistics and chooses one of 7 HGS parameter configurations.

**Observation (132-dim vector):**
```
obs = [graph_embedding (128) | solver_stats (4)]
```

The 4 solver statistics:
- **time_ratio** = steps_used / 5 — "How much budget is left?"
- **nv_ratio** = NV_current / NV_initial — "How much fleet reduction so far?"
- **score_ratio** = best_score / initial_score — "How much overall improvement?"
- **stagnation_ratio** = iters_no_improve / budget — "Am I stuck?"

**Action Space (7 discrete actions):**

Each action selects a different genetic algorithm configuration. The key parameters controlled are:

- **mu** (min population size): Smaller = faster but less diverse. Larger = more diverse, slower.
- **lambda_** (offspring count): How many new solutions per generation.
- **nbGranular**: Local search neighborhood size. Higher = more thorough.
- **targetFeasible**: Fraction of feasible solutions. Lower = more aggressive exploration.
- **nbElite / nbClose**: Elite protection and diversity balance.

| # | Action | Strategy | Key Settings |
|---|--------|----------|------------|
| 0 | DEFAULT | Standard HGS defaults | mu=25, lambda=40, nbGranular=20, targetFeasible=0.2 |
| 1 | FAST_AGGRESSIVE | Small pop, speed + pressure | mu=15, lambda=20, nbGranular=15, targetFeasible=0.1 |
| 2 | LARGE_DIVERSE | Big pop, thorough search | mu=40, lambda=60, nbGranular=30, targetFeasible=0.3 |
| 3 | DEEP_SEARCH | Deep local search | mu=25, lambda=40, nbGranular=40, targetFeasible=0.2 |
| 4 | HIGH_TURNOVER | Max churn, high-risk | mu=10, lambda=80, nbGranular=20, targetFeasible=0.05 |
| 5 | STABLE_ELITE | Conservative refinement | mu=50, lambda=30, nbGranular=25, targetFeasible=0.4 |
| 6 | EXPLORE_NEW_SEED | Default + fresh random seed | Escapes local optima |

**Network:** Actor-Critic with shared trunk
```
obs (132) --> Linear(132->64) + ReLU --> Linear(64->64) + ReLU
                                              |
                              +---------------+---------------+
                              |                               |
                         Actor Head                     Critic Head
                       Linear(64->7)                   Linear(64->1)
                              |                               |
                     action_logits (7)                state_value (1)
```

- **Actor** outputs a probability distribution over the 7 actions (the policy)
- **Critic** outputs a single number estimating "how good is this state?" (the value function)
- **Parameters:** ~13,100 — intentionally tiny since it makes strategic decisions, not route computations

**Action Masking:** When the fleet is already at the theoretical minimum (NV = ceil(total_demand / capacity)), the most aggressive actions (1=FAST_AGGRESSIVE, 4=HIGH_TURNOVER) are blocked by setting their logits to -10,000. After softmax, these become ~0 probability, preventing the agent from wasting steps on impossible fleet reductions.

---

### Stage 3 — Solver Engine (`solver_engine.py`)

**What it does:** The bridge between the RL world and the actual solver. Implements a Gymnasium environment that wraps HGS-CVRP (via the `hygese` Python package).

**Episode lifecycle:**
1. `reset()` — Pick a random .vrp instance, encode it with the GNN, run an initial 5,000-iteration solve with default HGS parameters
2. `step(action)` — Translate the action to HGS AlgorithmParameters, run a **fresh** 5,000-iteration solve, track the best solution found
3. Repeat `step()` 5 times = **25,000 total HGS iterations** per episode

**How actions become solver parameters:**
The Fleet Manager doesn't touch routes directly. It controls HGS's **AlgorithmParameters** — the configuration of the genetic algorithm itself:
- **mu, lambda_** — Population size and offspring count. Controls the balance between exploration and exploitation in the genetic search.
- **nbGranular** — Neighborhood size for local search. Higher = deeper local optimization.
- **targetFeasible** — Fraction of population that must be feasible. Lower = solver explores more infeasible intermediate solutions.
- **nbElite, nbClose** — Elite protection and diversity measurement. Controls whether the GA favors exploitation or exploration.
- **seed** — Same seed = reproducible search. New seed = different starting point (escape local optima).

**Important: No warm starting.** Unlike some solvers, HGS does not accept a previous solution as input. Each step runs a completely fresh solve. The environment tracks the best solution found across all steps in the episode. This means the agent's value comes from finding configurations that consistently produce good results, not from iteratively improving a single solution.

**Reward:** `reward = previous_best_score - new_best_score` (positive when score improves)

**Safety: Fleet Explosion Detection.** If an aggressive action (FAST_AGGRESSIVE or HIGH_TURNOVER) causes NV to spike by more than 2 vehicles, the bad solution is rejected and a penalty of -5.0 is returned. This teaches the agent that aggressive configurations can backfire.

---

### Stage 5 — PPO Training Loop (`train.py`)

**What it does:** Trains the Fleet Manager using Proximal Policy Optimization (PPO), one of the most popular and stable RL algorithms.

---

## 3. Reinforcement Learning Concepts

### What is RL?

Reinforcement Learning is a paradigm where an **agent** learns to make decisions by interacting with an **environment**. The agent observes a **state**, takes an **action**, receives a **reward**, and transitions to a new state. The goal is to learn a **policy** (a mapping from states to actions) that maximizes cumulative reward over time.

In our system:
- **Agent** = Fleet Manager
- **Environment** = CVRPEnv (HGS-CVRP solver wrapper)
- **State** = 132-dim observation vector (graph embedding + solver stats)
- **Action** = One of 7 GA parameter configurations
- **Reward** = Score improvement (prev_score - new_score)

### Policy and Value Functions

**Policy (pi):** A probability distribution over actions given a state. The Actor head outputs this. During training, actions are *sampled* from this distribution (enabling exploration). During evaluation, we pick the *argmax* (greedy, no randomness).

**Value Function V(s):** The Critic head's estimate of "how much total future reward can we expect from this state?" This is used to compute *advantages* — how much better an action was compared to what we expected.

### Actor-Critic Architecture

Our Fleet Manager uses an **Actor-Critic** design with a **shared trunk**:

- The **shared trunk** (two 64-unit layers) extracts features common to both policy and value estimation. Sharing is efficient and keeps the critic's estimates aligned with the actor's policy.
- The **Actor** head outputs logits for each of the 7 actions. These are converted to probabilities by the Categorical distribution (softmax internally).
- The **Critic** head outputs a scalar V(s). PPO uses this to compute advantages.

### Advantage Estimation (GAE-lambda)

An **advantage** A(s, a) measures "how much better was action *a* in state *s* compared to the average action in that state?"
- Positive advantage = action was better than expected
- Negative advantage = action was worse than expected

We use **Generalized Advantage Estimation (GAE-lambda)** to compute advantages. There are two extremes:
- **Monte Carlo (lambda=1.0):** Use actual cumulative returns. Unbiased but high variance (noisy).
- **TD(0) (lambda=0.0):** Use one-step bootstrapped returns. Low variance but biased (depends on critic accuracy).

GAE-lambda interpolates between these. We use **lambda=0.95**, getting most of the variance reduction of TD while staying close to unbiased.

**The formula (working backwards from the last step):**
```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)     [TD error at step t]
A_t     = delta_t + gamma * lambda * A_{t+1}      [GAE accumulation]
```

The **discount factor gamma=0.99** controls how much future rewards matter. Over 5 steps, the last step's reward is worth 0.99^4 ~ 0.96 of its face value.

---

## 4. The Loss Function

PPO's total loss combines three components:

```
Loss = Policy Loss + 0.5 * Value Loss - 0.05 * Entropy Bonus
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

The entropy bonus (weighted by **ent_coeff=0.05**) is *subtracted* from the loss (since we minimize loss, subtracting encourages higher entropy). This prevents the policy from "collapsing" to always picking one action too early in training.

**What to watch:** If entropy drops to 0, the agent has stopped exploring — this is bad. Healthy training maintains entropy above ~0.5.

---

## 5. The Optimization Algorithm

### Adam Optimizer

We use **Adam** (Adaptive Moment Estimation), the standard optimizer for deep learning and RL. Adam maintains per-parameter running averages of:
- First moment (mean of gradients) — momentum
- Second moment (mean of squared gradients) — adaptive learning rate

This makes it more robust than vanilla SGD — parameters with noisy gradients get smaller updates, while parameters with consistent gradients get larger updates.

**Learning Rate:** Starts at **1e-4** and decays linearly to **5e-5** over training. This allows large, bold updates early (when the policy is random and needs to change a lot) and finer adjustments later (when the policy is maturing).

### Gradient Clipping

```
clip_grad_norm_(parameters, max_norm=0.5)
```
If the gradient norm exceeds 0.5, all gradients are scaled down proportionally. This prevents **exploding gradients** — a single bad transition producing a massive gradient that destroys the policy.

### FP16 Mixed Precision

PyTorch's Automatic Mixed Precision (AMP) runs the forward and backward passes in FP16 (half precision) on GPU. This roughly halves memory usage and can speed up computation. A **GradScaler** dynamically scales the loss to prevent underflow in FP16 gradients.

### KL Divergence Early Stopping

After each mini-epoch, we check how much the policy has changed from the old policy using KL divergence. If **KL > 1.5 * target_kl (0.015)**, we stop the PPO update early to prevent catastrophic policy changes. This is a safety mechanism on top of the clipping.

---

## 6. Reward Normalization (Welford's Algorithm)

**The problem:** Rewards in our system have a huge range:
- Removing a vehicle: ~+1000
- Good configuration finding lower TD: ~+5 to +50
- Fleet explosion penalty: -5.0

Without normalization, +1000 vehicle rewards would dominate gradients and destabilize training.

**The solution:** We track running mean and variance using **Welford's online algorithm** and normalize all rewards to roughly zero mean, unit variance before computing advantages.

Welford's algorithm is numerically stable and O(1) memory — it updates incrementally with each batch rather than storing all historical rewards. The statistics are saved in checkpoints so they persist across training resumes.

---

## 7. Curriculum Learning

Training starts with **easier instances** and gradually introduces harder ones:

- **Epochs 1-20:** Only small instances (N <= 100 customers)
- **Epochs 21+:** All instances (N up to 400 customers)

**Why?** The agent learns basic strategies (which configurations work when) faster on small, quick-to-solve instances. Once it has these fundamentals, it transfers this knowledge to larger, harder instances. Starting on 400-node instances from the beginning would make early learning very slow because each episode takes much longer and the solution space is much larger.

After epoch 20, the agent still sees small instances mixed with large ones — it doesn't forget what it learned.

---

## 8. Training Loop Summary

Each epoch follows this cycle:

**Phase 1 — Collect Experience (4 episodes x 5 steps = 20 transitions)**
- Reset environment with a random instance
- Agent observes state, samples action (stochastic — with exploration)
- Environment executes action (5,000 HGS iterations), returns reward
- Store all transitions in a rollout buffer

**Phase 2 — PPO Update (4 mini-epochs over the 20 transitions)**
- Normalize advantages to zero mean, unit variance
- Shuffle transitions into mini-batches of 64
- For each mini-batch: compute policy loss + value loss + entropy bonus, do gradient step
- Check KL divergence after each mini-epoch, stop early if too large

**Phase 3 — Evaluate (5 fixed instances, greedy action selection)**
- No exploration — pure argmax action selection
- Same 5 instances every epoch for consistent progress tracking
- Save the model if eval score improves (best model tracking)

**Key metrics:**
- **Eval score** (the real metric) — should trend downward over time
- **AvgScore** — noisy because of random instances, mostly ignore
- **Entropy** — should stay above ~0.5 (agent still exploring)
- **Policy loss** — small values are normal

---

## 9. Baseline Evaluation (`scripts/baseline_eval.py`)

To demonstrate that the RL agent adds value, we compare against HGS with fixed parameters:

**Single-Solve Baseline (fair comparison):**
One HGS solve with default parameters (seed=42) per instance — this is what HGS gets without any RL guidance. The RL agent must beat this.

**Best-of-N Baseline (upper bound):**
Multiple random seeds x 5 steps, keeping the best result — shows what HGS can achieve with brute-force repetition. This is an upper bound, not a fair comparison.

Both baselines use the same iteration budget (5,000 per step) as the RL agent.

---

## 10. File Map

| File | Role | Parameters |
|------|------|-----------|
| `model_vision.py` | GNN Encoder — spatial understanding | ~51,600 |
| `agent_manager.py` | Fleet Manager — RL agent (Actor-Critic) | ~13,100 |
| `solver_engine.py` | CVRPEnv — Gymnasium environment wrapping HGS-CVRP | — |
| `train.py` | PPO training loop, GAE, reward normalization | — |
| `main.py` | Entry point, smoke tests, CLI | — |
| `baseline_eval.py` | HGS baseline evaluation (default vs large-pop) | — |

**Total trainable parameters: ~64,700.** The model is intentionally lightweight. The heavy lifting is done by HGS-CVRP's C++ solver — our RL agent just learns to steer it effectively.
