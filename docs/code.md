# Code Explanation

## 1. System Overview

Our system is an **RL-guided CVRP solver** built for the GECCO 2026 ML4VRP competition. Instead of solving the CVRP directly, we train a Reinforcement Learning agent to **control how** the PyVRP solver runs — deciding when to push for fewer vehicles, when to polish routes, and when to explore new solution regions.

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
.vrp Instance  -->  GNN Encoder  -->  Fleet Manager  -->  PyVRP Solver
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

**Parameters:** ~51,600. Computed once per instance, then reused for all 20 steps of an episode.

---

### Stage 2 — Fleet Manager (`agent_manager.py`)

**What it does:** The RL agent — the "brain" of the system. At each step, it looks at the graph embedding + 4 real-time solver statistics and chooses one of 6 search strategies.

**Observation (132-dim vector):**
```
obs = [graph_embedding (128) | solver_stats (4)]
```

The 4 solver statistics:
- **time_ratio** = steps_used / 20 — "How much budget is left?"
- **nv_ratio** = NV_current / NV_initial — "How much fleet reduction so far?"
- **violation_ratio** = excess_load / total_demand — "Am I pushing too hard?"
- **stagnation_ratio** = iters_no_improve / budget — "Am I stuck?"

**Action Space (6 discrete actions):**

| # | Action | Penalty Increase | Use Case |
|---|--------|-----------------|----------|
| 0 | POLISH | Default params | Routes are good, just refine distances |
| 1 | MILD_PRESSURE | 2x | Gently nudge toward fewer vehicles |
| 2 | MODERATE_PRESSURE | 5x | Steady, reliable fleet reduction |
| 3 | AGGRESSIVE_PRESSURE | 10x | Force route merges (risky!) |
| 4 | EXPLORE_NEW_SEED | Default + new seed | Escape local optima |
| 5 | EXPLORE_PRESSURE | 5x + new seed | Escape and reduce simultaneously |

**Network:** Actor-Critic with shared trunk
```
obs (132) --> Linear(132->64) + ReLU --> Linear(64->64) + ReLU
                                              |
                              +---------------+---------------+
                              |                               |
                         Actor Head                     Critic Head
                       Linear(64->6)                   Linear(64->1)
                              |                               |
                     action_logits (6)                state_value (1)
```

- **Actor** outputs a probability distribution over the 6 actions (the policy)
- **Critic** outputs a single number estimating "how good is this state?" (the value function)
- **Parameters:** ~13,100 — intentionally tiny since it makes strategic decisions, not route computations

**Action Masking:** When the fleet is already at the theoretical minimum (NV = ceil(total_demand / capacity)), pressure actions (1, 2, 3, 5) are blocked by setting their logits to -10,000. After softmax, these become ~0 probability, preventing the agent from wasting steps on impossible fleet reductions.

---

### Stage 3 — Solver Engine (`solver_engine.py`)

**What it does:** The bridge between the RL world and the actual solver. Implements a Gymnasium environment that wraps PyVRP's Iterated Local Search (ILS) solver.

**Episode lifecycle:**
1. `reset()` — Pick a random .vrp instance, encode it with the GNN, run an initial 1000-iteration solve
2. `step(action)` — Translate the action to PyVRP parameters, run 1000 more iterations, compute reward
3. Repeat `step()` 20 times = 20,000 total ILS iterations per episode

**How actions become solver parameters:**
The Fleet Manager doesn't touch routes directly. It controls PyVRP's **penalty parameters**:
- **penalty_increase** — How much to raise the penalty for excess vehicles. Higher = more pressure to merge routes.
- **penalty_decrease** — How quickly the penalty relaxes. Lower = more sustained pressure.
- **target_feasible** — What fraction of the population should be feasible. Lower = solver explores more aggressively.
- **seed** — Same seed = continue search (exploitation). New seed = restart from different point (exploration).

**Reward:** `reward = previous_score - new_score` (positive when score improves)

**Safety: Fleet Explosion Detection.** If a pressure action causes NV to spike by more than 2 vehicles, the bad solution is rejected and a penalty of -5.0 is returned. This teaches the agent that aggressive pressure can backfire.

**Warm Starting:** Each step passes the current best solution as the starting point for PyVRP. The solver builds on what it already found rather than starting from scratch.

---

### Stage 5 — PPO Training Loop (`train.py`)

**What it does:** Trains the Fleet Manager using Proximal Policy Optimization (PPO), one of the most popular and stable RL algorithms.

---

## 3. Reinforcement Learning Concepts

### What is RL?

Reinforcement Learning is a paradigm where an **agent** learns to make decisions by interacting with an **environment**. The agent observes a **state**, takes an **action**, receives a **reward**, and transitions to a new state. The goal is to learn a **policy** (a mapping from states to actions) that maximizes cumulative reward over time.

In our system:
- **Agent** = Fleet Manager
- **Environment** = CVRPEnv (PyVRP solver wrapper)
- **State** = 132-dim observation vector (graph embedding + solver stats)
- **Action** = One of 6 solver strategies
- **Reward** = Score improvement (prev_score - new_score)

### Policy and Value Functions

**Policy (pi):** A probability distribution over actions given a state. The Actor head of our network outputs this. During training, actions are *sampled* from this distribution (enabling exploration). During evaluation, we pick the *argmax* (greedy, no randomness).

**Value Function V(s):** The Critic head's estimate of "how much total future reward can we expect from this state?" This is used to compute *advantages* — how much better an action was compared to what we expected.

### Actor-Critic Architecture

Our Fleet Manager uses an **Actor-Critic** design with a **shared trunk**:

- The **shared trunk** (two 64-unit layers) extracts features common to both policy and value estimation. Sharing is efficient and keeps the critic's estimates aligned with the actor's policy.
- The **Actor** head outputs logits for each of the 6 actions. These are converted to probabilities by the Categorical distribution (softmax internally).
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

The **discount factor gamma=0.99** controls how much future rewards matter. Over 20 steps, the last step's reward is worth 0.99^19 ~ 0.83 of its face value.

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
- Polishing routes: ~+5 to +50
- Fleet explosion penalty: -5.0

Without normalization, +1000 vehicle rewards would dominate gradients and destabilize training.

**The solution:** We track running mean and variance using **Welford's online algorithm** and normalize all rewards to roughly zero mean, unit variance before computing advantages.

Welford's algorithm is numerically stable and O(1) memory — it updates incrementally with each batch rather than storing all historical rewards. The statistics are saved in checkpoints so they persist across training resumes.

---

## 7. Curriculum Learning

Training starts with **easier instances** and gradually introduces harder ones:

- **Epochs 1-20:** Only small instances (N <= 100 customers)
- **Epochs 21+:** All instances (N up to 400 customers)

**Why?** The agent learns basic strategies (when to push vs. polish) faster on small, quick-to-solve instances. Once it has these fundamentals, it transfers this knowledge to larger, harder instances. Starting on 400-node instances from the beginning would make early learning very slow because each episode takes much longer and the solution space is much larger.

After epoch 20, the agent still sees small instances mixed with large ones — it doesn't forget what it learned.

---

## 8. Training Loop Summary

Each epoch follows this cycle:

**Phase 1 — Collect Experience (8 episodes x 20 steps = 160 transitions)**
- Reset environment with a random instance
- Agent observes state, samples action (stochastic — with exploration)
- Environment executes action (1000 PyVRP iterations), returns reward
- Store all transitions in a rollout buffer

**Phase 2 — PPO Update (4 mini-epochs over the 160 transitions)**
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

## 9. File Map

| File | Role | Parameters |
|------|------|-----------|
| `model_vision.py` | GNN Encoder — spatial understanding | ~51,600 |
| `agent_manager.py` | Fleet Manager — RL agent (Actor-Critic) | ~13,100 |
| `solver_engine.py` | CVRPEnv — Gymnasium environment wrapping PyVRP | — |
| `train.py` | PPO training loop, GAE, reward normalization | — |
| `main.py` | Entry point, smoke tests, CLI | — |
| `agent_driver.py` | UNUSED — kept for reference | — |

**Total trainable parameters: ~64,700.** The model is intentionally lightweight. The heavy lifting is done by PyVRP's C++ solver — our RL agent just learns to steer it effectively.
