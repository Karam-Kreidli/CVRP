# ML4VRP 2026 - System Architecture

## 1. Project Overview

We are building an **RL-guided CVRP solver** for the GECCO 2026 ML4VRP competition. The system uses a Reinforcement Learning agent to dynamically select **HGS-CVRP algorithm parameter configurations**, learning *which* genetic algorithm strategy works best for each instance and search phase.

**Competition Objective:**
```
Score = 1000 * NV + TD      (lower is better)
```
- **NV** = Number of Vehicles (fleet size)
- **TD** = Total Distance traveled

The 1000x multiplier on NV means eliminating one vehicle is worth 1000 distance units. Fleet minimization is the dominant factor.

**Dataset:** X-dataset (Uchoa et al., 2014) — 59 instances with 100-1001 customers.

**Solver:** HGS-CVRP via the `hygese` Python package (Hybrid Genetic Search — Vidal et al., 2012). A high-performance C++ genetic algorithm for CVRP, wrapped with a Python interface.

---

## 2. High-Level Architecture

```
                    CVRP Instance (.vrp file)
                            |
                            v
                +-----------------------+
                |  Hand-Crafted Feature |
                |  Extraction           |
                |                       |
                |  Input: node coords   |
                |         + demands     |
                |         + capacity    |
                |                       |
                |  Output:              |
                |    instance_features  |
                |    (1, 7)             |
                +-----------------------+
                            |
              instance_features (7-dim)
                            |
                            v
                +-----------------------+
                |   Fleet Manager       |
                |   (Actor-Critic RL    |
                |    Agent)             |
                |                       |
                |  Input: inst_feat     |
                |       + solver_stats  |
                |       = (1, 14)       |
                |                       |
                |  Output: action (0-6) |
                +-----------------------+
                            |
                   action (config choice)
                            |
                            v
                +-----------------------+
                |   HGS-CVRP Solver     |
                |   (C++ genetic alg)   |
                |                       |
                |  Runs 500 iterations  |
                |  with GA parameters   |
                |  set by the action    |
                |                       |
                |  Output: CVRP solution|
                +-----------------------+
                            |
                   reward = pct improvement
                            |
                            v
                +-----------------------+
                |   PPO Training Loop   |
                |                       |
                |  Updates Fleet Manager|
                |  policy to maximize   |
                |  cumulative reward    |
                +-----------------------+
```

---

## 3. Component Details

### 3.1 Instance Feature Extraction (`src/solver_engine.py`)

**Purpose:** Compute a fixed-size feature vector that captures the spatial and structural properties of a CVRP instance. These features let the RL agent distinguish between different types of instances and adapt its strategy accordingly.

**Features (7-dim vector):**

| # | Feature | Formula | What it captures |
|---|---------|---------|-----------------|
| 1 | size_norm | num_customers / 400 | Instance scale |
| 2 | demand_fill_ratio | total_demand / (nv_min * capacity) | How tightly vehicles are packed |
| 3 | mean_dist_norm | mean_distance / max_distance | Average inter-customer spacing |
| 4 | std_dist_norm | std_distance / max_distance | Distance variance (clustered vs spread) |
| 5 | depot_centrality | mean_depot_dist / max_distance | How central the depot is |
| 6 | demand_cv | std_demand / mean_demand | Demand heterogeneity |
| 7 | capacity_tightness | max_demand / capacity | How close any single customer is to filling a vehicle |

**Key details:**
- Computed **once per instance** at the start of each episode, then reused for all 50 steps
- All features are normalized to roughly [0, 1] range
- No learned parameters — these are deterministic computations

---

### 3.2 Fleet Manager (`src/agent_manager.py`)

**Purpose:** The RL agent. Decides which HGS parameter configuration to use at each step.

**Observation (14-dim):**
```
obs = [ instance_features (7) | solver_stats (7) ]
```

| Solver Stat | Formula | Meaning |
|---|---|---|
| time_ratio | steps_used / 50 | How far into the episode |
| nv_ratio | NV_current / NV_initial | Fleet reduction progress |
| score_ratio | best_score / initial_score | Overall improvement progress |
| stagnation_ratio | iters_no_improve / budget | Time since last improvement |
| nv_gap | (best_nv - nv_min) / NV_initial | How far from theoretical minimum fleet |
| last_reward | clipped last reward / 10 | Feedback from previous action |
| last_action_norm | last_action / NUM_ACTIONS | What was tried last |

**Action Space (7 discrete actions):**

Each action maps to a different HGS genetic algorithm configuration controlling population sizes, search granularity, feasibility targets, and elite management:

| # | Action | Strategy | Key Parameters |
|---|---|---|---|
| 0 | DEFAULT | Standard HGS defaults | mu=25, lambda=40, nbGranular=20 |
| 1 | FAST_AGGRESSIVE | Small pop, speed + pressure | mu=15, lambda=20, targetFeasible=0.1 |
| 2 | LARGE_DIVERSE | Big pop, thorough search | mu=40, lambda=60, nbGranular=30 |
| 3 | DEEP_SEARCH | Deep local search neighborhoods | mu=25, lambda=40, nbGranular=40 |
| 4 | HIGH_TURNOVER | Maximum churn, high-risk | mu=10, lambda=80, targetFeasible=0.05 |
| 5 | STABLE_ELITE | Conservative, large elite pool | mu=50, lambda=30, targetFeasible=0.4 |
| 6 | EXPLORE_NEW_SEED | Default params, fresh seed | Escapes local optima |

**Action Masking:** When NV = NV_min (theoretical minimum fleet = ceil(total_demand / capacity)), the most aggressive actions (1=FAST_AGGRESSIVE, 4=HIGH_TURNOVER) are blocked to prevent impossible fleet reductions.

**Network:**
```
obs (B, 14)
     |
     v
Linear(14 -> 64) + ReLU
     |
     v
Linear(64 -> 64) + ReLU
     |
     +-------------+
     |             |
     v             v
  Actor         Critic
Linear(64->7)  Linear(64->1)
     |             |
     v             v
action_logits  state_value
  (B, 7)        (B, 1)
```

- **Parameters:** ~5,700
- Intentionally small — the agent makes strategic decisions, not complex computations

---

### 3.3 HGS-CVRP Solver Engine (`src/solver_engine.py`)

**Purpose:** The actual CVRP solver. A high-performance C++ Hybrid Genetic Search engine wrapped as a Gymnasium environment.

**Episode structure:**
```
reset()
  |-> Load random instance
  |-> Compute hand-crafted instance features (once)
  |-> Initial solve (500 iterations, default params)
  |-> Return first observation
  |
  v
step(action) x 50 times
  |-> Map action to HGS AlgorithmParameters
  |-> Run FRESH solver for 500 iterations
  |-> Track best solution across all steps
  |-> Compute reward (pct improvement vs previous candidate)
  |-> Return (obs, reward, done, info)
  |
  v
Episode ends after 50 steps (= 25,000 total HGS iterations)
```

**Why 500 iterations per step (not 5,000)?**

At 5,000 iterations, HGS has essentially converged — the population has stabilized and different configs produce nearly identical results. At 500 iterations, HGS is still actively searching, so the choice of config genuinely affects the outcome. This gives the RL agent meaningful signal to learn from.

50 steps instead of 5 also means 10x more gradient signal per episode for PPO.

**How actions map to solver behavior:**

The Fleet Manager doesn't directly modify routes. Instead, it controls HGS's **genetic algorithm parameters**, which determine population dynamics, search depth, and feasibility pressure:

- **mu (min pop size):** Smaller = faster generations, less diversity. Larger = more diverse, slower convergence.
- **lambda_ (offspring size):** How many new solutions per generation. Higher = more exploration.
- **nbGranular:** Local search neighborhood size. Higher = more thorough but slower.
- **targetFeasible:** Fraction of population that should be feasible. Lower = more aggressive exploration.
- **nbElite / nbClose:** Elite protection and diversity balance.

**Key difference from warm-starting solvers:** Each step runs a **fresh** HGS solve (no warm starting). The environment tracks the best solution found across ALL steps in the episode. The agent's job is to find the parameter configuration that produces the best result.

**Reward design:**
```python
if fleet_exploded:
    reward = -5.0                    # penalty for NV spike
elif cand_score < prev_cand_score:
    pct = (prev_cand - cand) / prev_cand
    reward = pct * 100.0             # percentage improvement
else:
    reward = -0.5                    # small cost for no improvement
```

Key properties:
- Compares against **previous step's candidate** (not episode-best), so the agent gets signal every step
- **Percentage-based** — normalizes across different instance sizes
- Small negative for no improvement teaches the agent to avoid wasteful actions

**Safety mechanisms:**
- **Fleet explosion detection:** If an aggressive action causes NV to spike by >2, the bad solution is rejected and a penalty reward (-5.0) is returned
- **Best-of-N tracking:** The environment keeps the best solution found across all steps, so a bad step doesn't lose progress

---

### 3.4 PPO Training (`src/train.py`)

**Purpose:** Train the Fleet Manager to maximize cumulative reward using Proximal Policy Optimization.

**Training loop per epoch:**
```
For each of 8 episodes:
    1. Reset environment (random instance)
    2. For 50 steps:
        - Agent observes state
        - Agent picks action (with exploration)
        - Environment executes action (fresh HGS solve, 500 iters)
        - Store (state, action, reward, ...) in buffer
    3. Compute GAE advantages

PPO Update (3 mini-epochs):
    - Sample mini-batches from buffer
    - Update policy (clipped surrogate objective)
    - Update value function
    - Add entropy bonus (encourages exploration)
    - Early stop if KL divergence too large

Evaluate on 5 fixed instances (greedy, no exploration)
Save best model based on eval score
```

**Key hyperparameters:**

| Parameter | Value | Purpose |
|---|---|---|
| Learning rate | 1e-4 | Adam optimizer, with linear decay to 5e-5 |
| Gamma | 0.95 | Discount factor (shorter horizon for 50 steps) |
| Lambda (GAE) | 0.90 | More bias, less variance (appropriate for many-step episodes) |
| Epsilon clip | 0.2 | PPO policy update constraint |
| Entropy coeff | 0.02 | Exploration bonus (less needed with 50 steps of natural exploration) |
| PPO epochs | 3 | Update passes per rollout |
| Mini-batch size | 128 | Larger batches for stability |
| Target KL | 0.015 | Early stopping if policy changes too fast |
| FP16 | Yes | Mixed precision for GPU efficiency |

**Reward processing:** Rewards are clipped to [-10, +10] before computing advantages. The percentage-based reward design keeps values in a reasonable range naturally, so simple clipping is sufficient (no running statistics needed).

**Curriculum learning:**
- Epochs 1-20: Only train on small instances (N <= 100)
- Epochs 21+: Use all instances (N up to 1001)

This helps the agent learn basic strategies on easier problems before tackling harder ones.

---

## 4. Data Flow Summary

```
                  .vrp file
                     |
           +---------+---------+
           |                   |
           v                   v
     Feature Extraction    HGS reads
     (7 hand-crafted       problem data
      instance stats)      (distances,
           |                demands,
           v                capacity)
     instance_features         |
      (7-dim)                  |
           |                   |
           v                   |
     Fleet Manager             |
     observes:                 |
     [inst_feat | stats]       |
           |                   |
           v                   |
     Picks action (0-6)        |
           |                   |
           +-------------------+
           |
           v
     HGS solves with
     chosen GA parameters
     (500 iterations)
           |
           v
     New solution
     (NV, TD, score)
           |
           v
     reward = pct improvement
     over previous candidate
           |
           v
     PPO updates Fleet Manager
     to pick better actions
```

---

## 5. File Structure

```
src/
  model_vision.py    - GNN Encoder (legacy, not used in training pipeline)
  agent_manager.py   - Fleet Manager RL agent (Stage 2)
  solver_engine.py   - HGS-CVRP Gymnasium environment (Stage 3)
  train.py           - PPO training loop (Stage 5)
  main.py            - Entry point, smoke tests, CLI

scripts/
  setup_vm.sh        - Ubuntu GPU VM setup script
  baseline_eval.py   - HGS baseline evaluation (default vs large-pop)
  portfolio_solver.py - Deterministic portfolio baseline (11 configs x N seeds)

docs/
  architecture.md    - System architecture (this file)
  code.md            - Code explanation
  benchmark_reference.md - X-dataset BKS scores from Uchoa et al.

data/                - X-dataset .vrp instance files
logs/                - Training CSV metrics + best model
checkpoints/         - Periodic checkpoint .pth files
```

---

## 6. Training Metrics

When training runs, you'll see output like:
```
[  5/ 50] AvgScore: 63483 | Eval: 42230 *BEST* | Ent: 1.852 | 39m57s/epoch | ETA: 29h32m | Elapsed: 3h19m
```

| Metric | What it means |
|---|---|
| AvgScore | Average score across 8 training episodes (noisy due to random instances) |
| **Eval** | **Score on 5 fixed instances (the real progress metric)** |
| Ent | Policy entropy (>0.5 = still exploring, 0 = collapsed) |
| /epoch | Wall-clock time per epoch |
| ETA | Estimated time remaining |

**What to watch:** The Eval column should trend downward over time. AvgScore will be noisy — ignore it. Entropy should stay above ~0.5. A stagnation warning appears if eval hasn't improved in 15 epochs.

---

## 7. Model Sizes

| Component | Parameters | Purpose |
|---|---|---|
| Fleet Manager | ~5,700 | Strategy selection |

The model is intentionally lightweight. The heavy lifting is done by HGS-CVRP's C++ solver — our RL agent just learns to steer it effectively.

---

## 8. Baseline Comparison

The RL agent is compared against baselines:

| Baseline | Description | How it works |
|---|---|---|
| **Single Default** | One HGS solve with default params (seed=42) | Fair comparison — what HGS gets without RL |
| **Best-of-N** | Multiple seeds x 5 steps, keep best | Upper bound — brute-force repetition |
| **Portfolio** | 11 configs x 5 seeds, 10k iters each, keep best per instance | Strongest non-RL baseline |

The RL agent must beat the Single Default baseline to demonstrate that learned parameter tuning adds value over HGS's hand-tuned defaults.

Reference scores from the literature are in `docs/benchmark_reference.md` (Uchoa et al., 2014).

---

## 9. Why This Approach Works (Intuition)

### The Core Idea

The competition asks us to find the best CVRP solution (lowest `1000*NV + TD`). HGS-CVRP is already an excellent solver — but it uses **fixed parameters**. The same population sizes, granularity, and feasibility targets are applied regardless of whether the instance has tight clusters, spread-out customers, balanced demand, or skewed demand.

Our RL agent learns to **adapt the GA configuration to each instance and each moment in the search**. It's not solving the CVRP itself — it's learning to be a better operator of HGS than any fixed configuration could be.

### How Does the Agent Understand the Problem?

The agent builds understanding from two sources:

**Instance features (hand-crafted):**
Seven computed features capture the structural properties of each instance — size, demand distribution, spatial layout, depot position, and capacity tightness. Instances with different characteristics produce different feature vectors, allowing the agent to learn instance-specific strategies.

**Solver feedback (observation stats):**
At each step, the agent sees 7 real-time signals from the solver:
- **nv_ratio** — "How much fleet reduction has happened so far?"
- **score_ratio** — "How much overall improvement has been made?"
- **stagnation_ratio** — "Has progress stalled? Should I try something different?"
- **time_ratio** — "How much budget is left?"
- **nv_gap** — "How far from the theoretical minimum fleet?"
- **last_reward** — "Did the previous action help?"
- **last_action_norm** — "What did I just try?"

### The Learning Process

The agent doesn't know the right configuration in advance. It learns through trial and error over thousands of episodes:

1. Early in training, the agent picks configurations randomly
2. Sometimes FAST_AGGRESSIVE works and produces a lower score -> positive reward
3. Sometimes HIGH_TURNOVER backfires and the fleet explodes -> penalty (-5.0)
4. Sometimes LARGE_DIVERSE finds a better solution through thorough search -> reward
5. Sometimes EXPLORE_NEW_SEED escapes a local optimum -> breakthrough

Over time, PPO adjusts the policy so the agent picks configurations that led to positive outcomes more often.

### Analogy

Think of HGS as a race car and the Fleet Manager as the driver:
- The **car** (HGS) has all the power — it does the actual route optimization
- The **driver** (Fleet Manager) decides when to use aggressive settings (speed mode), when to use conservative settings (cruise mode), and when to restart from a different starting point
- The **track** (CVRP instance) is different every time — the features describe the track layout
- **Training** is the driver practicing on many different tracks until they develop good instincts
- A car on autopilot with fixed settings will finish the race, but a skilled driver adapts to each track and gets a better time
