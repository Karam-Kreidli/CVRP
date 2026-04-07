# ML4VRP 2026 - System Architecture

## 1. Project Overview

We are building an **RL-guided CVRP solver** for the GECCO 2026 ML4VRP competition. The system uses a Reinforcement Learning agent to make **fleet-target strategy decisions** — choosing whether to push for fewer vehicles, lock the current fleet, or try a new random seed at each step of the search.

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
                   action (strategy choice)
                            |
                            v
                +-----------------------+
                |   HGS-CVRP Solver     |
                |   (C++ genetic alg)   |
                |                       |
                |  Runs 500-1500 iters  |
                |  with fleet target +  |
                |  seed set by action   |
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

**Purpose:** The RL agent. Decides the fleet-target strategy at each step — whether to push for fewer vehicles, lock the current fleet size, or try a new random seed.

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

The key insight: removing one vehicle saves 1000 in score, but forcing fewer vehicles can fail entirely. The agent must learn **when** pushing is worth the risk — this is instance-dependent and changes as the search progresses.

**Action Masking:** When NV <= NV_min (theoretical minimum fleet = ceil(total_demand / capacity)), actions 4, 5, 6 (PUSH_SAME, PUSH_NEW, FORCE_MIN) are blocked since further fleet reduction is impossible.

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
  |-> Map action to (fleet_target, seed, iteration_budget)
  |-> Run FRESH solver for 500-1500 iterations
  |-> Track best solution across all steps
  |-> Compute reward (pct improvement vs previous candidate)
  |-> Return (obs, reward, done, info)
  |
  v
Episode ends after 50 steps (~25,000-33,000 total HGS iterations)
```

**How actions map to solver behavior:**

The Fleet Manager doesn't directly modify routes. It controls two key levers:

1. **Fleet target** (`num_vehicles` parameter in HGS): Constrains how many vehicles HGS can use. Setting this to `best_nv - 1` forces HGS to find a solution with one fewer vehicle — if it succeeds, that's a 1000-point score improvement. But if the instance is too tight, HGS fails entirely.

2. **Random seed**: Same seed reproduces the same search trajectory. A new seed explores a different region of the solution space, potentially escaping local optima.

3. **Iteration budget**: Harder tasks (PUSH, FORCE) get more iterations (1000-1500) since finding feasible solutions with fewer vehicles requires deeper search.

**Key difference from warm-starting solvers:** Each step runs a **fresh** HGS solve (no warm starting). The environment tracks the best solution found across ALL steps in the episode.

**Reward design:**
```python
if fleet_exploded:
    reward = -5.0                    # penalty for NV spike
elif cand_score < best_score:
    pct = (best_score - cand_score) / best_score
    reward = pct * 100.0             # percentage improvement over episode best
else:
    reward = -0.5                    # small cost for no improvement
```

Key properties:
- Compares against **episode best** (not the previous step's candidate) — reward is only positive when a new best is found, giving an unambiguous learning signal
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
        - Environment executes action (fresh HGS solve, 500-1500 iters)
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
     chosen fleet target + seed
     (500-1500 iterations)
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

The competition asks us to find the best CVRP solution (lowest `1000*NV + TD`). HGS-CVRP is already an excellent solver — but with default parameters it uses an unconstrained fleet. It doesn't strategically explore whether fewer vehicles could yield a better competition score.

Our RL agent learns **when to push for fewer vehicles and when to back off**. Since removing one vehicle saves 1000 points but can fail on tight instances, this is a genuinely learnable, instance-dependent decision. The agent isn't solving the CVRP itself — it's learning to be a strategic operator of HGS.

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

The agent doesn't know when to push for fewer vehicles in advance. It learns through trial and error over thousands of episodes:

1. Early in training, the agent picks strategies randomly
2. Sometimes PUSH succeeds and removes a vehicle -> reward of +1.97 (huge score improvement)
3. Sometimes PUSH fails and the fleet explodes -> penalty (-5.0)
4. Sometimes LOCK_SAME optimizes distance at the current fleet size -> small positive reward
5. Sometimes FREE_NEW with a fresh seed escapes a local optimum -> breakthrough

Over time, PPO adjusts the policy so the agent learns: "for this type of instance with this much slack, PUSH is worth the risk" vs. "this instance is too tight, just LOCK and optimize distance."

### Analogy

Think of HGS as a race car and the Fleet Manager as the driver:
- The **car** (HGS) has all the power — it does the actual route optimization
- The **driver** (Fleet Manager) decides when to push for fewer vehicles (aggressive overtake), when to lock the current fleet and optimize distance (cruise mode), and when to try a new seed (restart from a different position)
- The **track** (CVRP instance) is different every time — the features describe the track layout
- **Training** is the driver practicing on many different tracks until they develop good instincts
- A car on autopilot with fixed settings will finish the race, but a skilled driver adapts to each track and gets a better time
