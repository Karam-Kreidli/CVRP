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

**Dataset:** X-dataset (Uchoa et al., 2014) — 59 instances with 100-400 customers.

**Solver:** HGS-CVRP via the `hygese` Python package (Hybrid Genetic Search — Vidal et al., 2012). A high-performance C++ genetic algorithm for CVRP, wrapped with a Python interface.

---

## 2. High-Level Architecture

```
                    CVRP Instance (.vrp file)
                            |
                            v
                +-----------------------+
                |   GNN Encoder         |
                |   (Graph Attention    |
                |    Network)           |
                |                       |
                |  Input: node coords   |
                |         + demands     |
                |                       |
                |  Output:              |
                |    graph_embedding    |
                |    (1, 128)           |
                +-----------------------+
                            |
              graph_embedding (128-dim)
                            |
                            v
                +-----------------------+
                |   Fleet Manager       |
                |   (Actor-Critic RL    |
                |    Agent)             |
                |                       |
                |  Input: graph_emb     |
                |       + solver_stats  |
                |       = (1, 132)      |
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
                |  Runs 5,000 iterations|
                |  with GA parameters   |
                |  set by the action    |
                |                       |
                |  Output: CVRP solution|
                +-----------------------+
                            |
                   reward = prev_score - new_score
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

### 3.1 GNN Encoder (`src/model_vision.py`)

**Purpose:** Encode the spatial structure of a CVRP instance into a fixed-size vector that the RL agent can understand.

**Architecture:**
```
Node Features (N, 3)           Positions (N, 2)
[x_norm, y_norm, demand/Q]     [x_norm, y_norm]
         |                           |
         v                           v
  Linear(3 -> 128)            k-NN Graph (k=20)
         |                           |
         +---------------------------+
         |
         v
  +-----------------------------------+
  | GAT Layer 1 (8 heads, dim=16/head)|
  | BatchNorm -> ELU -> Residual      |
  +-----------------------------------+
         |
         v
  +-----------------------------------+
  | GAT Layer 2 (8 heads, dim=16/head)|
  | BatchNorm -> ELU -> Residual      |
  +-----------------------------------+
         |
         v
  +-----------------------------------+
  | GAT Layer 3 (8 heads, dim=16/head)|
  | BatchNorm -> ELU -> Residual      |
  +-----------------------------------+
         |
         +-----------------+
         |                 |
         v                 v
  node_embeddings    global_mean_pool
    (N, 128)              |
                          v
                  graph_embedding
                     (1, 128)
```

**Key details:**
- **Parameters:** ~51,600
- **Graph construction:** k-NN with k=20 nearest neighbors (Euclidean distance), undirected
- **Why GAT?** Attention lets the network learn which neighbor relationships matter most (e.g., tight clusters with high demand)
- **Residual connections** prevent gradient degradation through 3 layers
- Computed **once per instance** at the start of each episode, then reused

---

### 3.2 Fleet Manager (`src/agent_manager.py`)

**Purpose:** The RL agent. Decides which HGS parameter configuration to use at each step.

**Observation (132-dim):**
```
obs = [ graph_embedding (128) | solver_stats (4) ]
```

| Solver Stat | Formula | Meaning |
|---|---|---|
| time_ratio | steps_used / 5 | How far into the episode |
| nv_ratio | NV_current / NV_initial | Fleet reduction progress |
| score_ratio | best_score / initial_score | Overall improvement progress |
| stagnation_ratio | iters_no_improve / budget | Time since last improvement |

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
obs (B, 132)
     |
     v
Linear(132 -> 64) + ReLU
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

- **Parameters:** ~13,100
- Intentionally small — the agent makes strategic decisions, not complex computations

---

### 3.3 HGS-CVRP Solver Engine (`src/solver_engine.py`)

**Purpose:** The actual CVRP solver. A high-performance C++ Hybrid Genetic Search engine wrapped as a Gymnasium environment.

**Episode structure:**
```
reset()
  |-> Load random instance
  |-> Encode with GNN (once)
  |-> Initial solve (5,000 iterations, default params)
  |-> Return first observation
  |
  v
step(action) x 5 times
  |-> Map action to HGS AlgorithmParameters
  |-> Run FRESH solver for 5,000 iterations
  |-> Track best solution across all steps
  |-> Compute reward = prev_score - new_score
  |-> Return (obs, reward, done, info)
  |
  v
Episode ends after 5 steps (= 25,000 total HGS iterations)
```

**How actions map to solver behavior:**

The Fleet Manager doesn't directly modify routes. Instead, it controls HGS's **genetic algorithm parameters**, which determine population dynamics, search depth, and feasibility pressure:

- **mu (min pop size):** Smaller = faster generations, less diversity. Larger = more diverse, slower convergence.
- **lambda_ (offspring size):** How many new solutions per generation. Higher = more exploration.
- **nbGranular:** Local search neighborhood size. Higher = more thorough but slower.
- **targetFeasible:** Fraction of population that should be feasible. Lower = more aggressive exploration.
- **nbElite / nbClose:** Elite protection and diversity balance.

**Key difference from warm-starting solvers:** Each step runs a **fresh** HGS solve (no warm starting). The environment tracks the best solution found across ALL steps in the episode. The agent's job is to find the parameter configuration that produces the best result.

**Safety mechanisms:**
- **Fleet explosion detection:** If an aggressive action (FAST_AGGRESSIVE or HIGH_TURNOVER) causes NV to spike by >2, the bad solution is rejected and a penalty reward (-5.0) is returned
- **Best-of-N tracking:** The environment keeps the best solution found across all steps, so a bad step doesn't lose progress

---

### 3.4 PPO Training (`src/train.py`)

**Purpose:** Train the Fleet Manager to maximize cumulative reward using Proximal Policy Optimization.

**Training loop per epoch:**
```
For each of 4 episodes:
    1. Reset environment (random instance)
    2. For 5 steps:
        - Agent observes state
        - Agent picks action (with exploration)
        - Environment executes action (fresh HGS solve)
        - Store (state, action, reward, ...) in buffer
    3. Compute GAE advantages

PPO Update (4 mini-epochs):
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
| Gamma | 0.99 | Discount factor (future rewards matter) |
| Lambda (GAE) | 0.95 | Advantage estimation bias-variance tradeoff |
| Epsilon clip | 0.2 | PPO policy update constraint |
| Entropy coeff | 0.05 | Exploration bonus |
| PPO epochs | 4 | Update passes per rollout |
| Target KL | 0.015 | Early stopping if policy changes too fast |
| FP16 | Yes | Mixed precision for GPU efficiency |

**Reward normalization:** Uses Welford's online algorithm to track running mean/std of rewards. This prevents large rewards (e.g., +1000 from removing a vehicle) from causing training instability.

**Curriculum learning:**
- Epochs 1-20: Only train on small instances (N <= 100)
- Epochs 21+: Use all instances (N up to 400)

This helps the agent learn basic strategies on easier problems before tackling harder ones.

---

## 4. Data Flow Summary

```
                  .vrp file
                     |
           +---------+---------+
           |                   |
           v                   v
     GNN Encoder          HGS reads
     (spatial             problem data
      understanding)      (distances,
           |               demands,
           v               capacity)
     graph_embedding           |
      (128-dim)                |
           |                   |
           v                   |
     Fleet Manager             |
     observes:                 |
     [graph_emb | stats]       |
           |                   |
           v                   |
     Picks action (0-6)        |
           |                   |
           +-------------------+
           |
           v
     HGS solves with
     chosen GA parameters
     (5,000 iterations)
           |
           v
     New solution
     (NV, TD, score)
           |
           v
     reward = old_score - new_score
           |
           v
     PPO updates Fleet Manager
     to pick better actions
```

---

## 5. File Structure

```
src/
  model_vision.py    - GNN Encoder (Stage 1)
  agent_manager.py   - Fleet Manager RL agent (Stage 2)
  solver_engine.py   - HGS-CVRP Gymnasium environment (Stage 3)
  train.py           - PPO training loop (Stage 5)
  main.py            - Entry point, smoke tests, CLI
  agent_driver.py    - [UNUSED] Route Driver (kept for reference)

scripts/
  setup_vm.sh        - Ubuntu GPU VM setup script
  baseline_eval.py   - HGS baseline evaluation (default vs large-pop)

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
| AvgScore | Average score across 4 training episodes (noisy due to random instances) |
| **Eval** | **Score on 5 fixed instances (the real progress metric)** |
| Ent | Policy entropy (>0.5 = still exploring, 0 = collapsed) |
| /epoch | Wall-clock time per epoch |
| ETA | Estimated time remaining |

**What to watch:** The Eval column should trend downward over time. AvgScore will be noisy — ignore it. Entropy should stay above ~0.5. A stagnation warning appears if eval hasn't improved in 15 epochs.

---

## 7. Model Sizes

| Component | Parameters | Purpose |
|---|---|---|
| GNN Encoder | ~51,600 | Spatial understanding |
| Fleet Manager | ~13,100 | Strategy selection |
| **Total** | **~64,700** | Entire trainable model |

The model is intentionally lightweight. The heavy lifting is done by HGS-CVRP's C++ solver — our RL agent just learns to steer it effectively.

---

## 8. Baseline Comparison

The RL agent is compared against two HGS baselines (`scripts/baseline_eval.py`):

| Baseline | Description | How it works |
|---|---|---|
| **Single Default** | One HGS solve with default params (seed=42) | Fair comparison — what HGS gets without RL |
| **Best-of-N** | Multiple seeds x 5 steps, keep best | Upper bound — brute-force repetition |

The RL agent must beat the Single Default baseline to demonstrate that learned parameter tuning adds value over HGS's hand-tuned defaults.

Reference scores from the literature are in `docs/benchmark_reference.md` (Uchoa et al., 2014).

---

## 9. Why This Approach Works (Intuition)

### The Core Idea

The competition asks us to find the best CVRP solution (lowest `1000*NV + TD`). HGS-CVRP is already an excellent solver — but it uses **fixed parameters**. The same population sizes, granularity, and feasibility targets are applied regardless of whether the instance has tight clusters, spread-out customers, balanced demand, or skewed demand.

Our RL agent learns to **adapt the GA configuration to each instance and each moment in the search**. It's not solving the CVRP itself — it's learning to be a better operator of HGS than any fixed configuration could be.

### How Does the Agent Understand the Problem?

The agent never sees raw routes or calculates optimal fleet size. Instead, it builds understanding from two sources:

**Spatial understanding (GNN Encoder):**
The Graph Attention Network reads the customer locations and demands, producing a 128-dim embedding that captures the instance's spatial structure. Instances with different characteristics (tight clusters, spread-out nodes, high/low demand variance) produce different embeddings. Over training, the agent learns that certain spatial patterns respond better to certain configurations.

**Solver feedback (Observation Stats):**
At each step, the agent sees 4 real-time signals from the solver:
- **nv_ratio** — "How much fleet reduction has happened so far?"
- **score_ratio** — "How much overall improvement has been made?"
- **stagnation_ratio** — "Has progress stalled? Should I try something different?"
- **time_ratio** — "How much budget is left?"

### The Learning Process

The agent doesn't know the right configuration in advance. It learns through trial and error over thousands of episodes:

1. Early in training, the agent picks configurations randomly
2. Sometimes FAST_AGGRESSIVE works and produces a lower score -> positive reward
3. Sometimes HIGH_TURNOVER backfires and the fleet explodes -> penalty (-5.0)
4. Sometimes LARGE_DIVERSE finds a much better solution through thorough search -> large reward
5. Sometimes EXPLORE_NEW_SEED escapes a local optimum -> breakthrough

Over time, PPO adjusts the policy so the agent picks configurations that led to positive outcomes more often. It develops patterns like:
- "When stagnation is high -> try a new seed (stuck in local optimum)"
- "When nv_ratio is already near minimum -> use STABLE_ELITE or DEEP_SEARCH"
- "Instances with this spatial pattern -> LARGE_DIVERSE works well"
- "Early in episode -> try aggressive configs; late -> conservative ones"

### Why Not Just Tune Parameters Manually?

You could hand-tune a fixed strategy like "always use large population." But:
- **Different instances need different configurations.** A 100-customer instance with tight clusters behaves very differently from a 400-customer spread-out instance.
- **Different phases of the search need different configurations.** Early on, aggressive exploration is useful. Late in the search, stable refinement matters more.
- **The interaction is complex.** The best sequence of 5 configurations depends on how the solver responded to previous ones. No fixed rule captures this.

The RL agent learns a **dynamic, instance-adaptive policy** — something no hand-tuned configuration can match.

### Analogy

Think of HGS as a race car and the Fleet Manager as the driver:
- The **car** (HGS) has all the power — it does the actual route optimization
- The **driver** (Fleet Manager) decides when to use aggressive settings (speed mode), when to use conservative settings (cruise mode), and when to restart from a different starting point
- The **track** (CVRP instance) is different every time — the GNN reads the track layout
- **Training** is the driver practicing on many different tracks until they develop good instincts
- A car on autopilot with fixed settings will finish the race, but a skilled driver adapts to each track and gets a better time
