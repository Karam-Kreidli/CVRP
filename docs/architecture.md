# ML4VRP 2026 - System Architecture

## 1. Project Overview

We are building an **RL-guided CVRP solver** for the GECCO 2026 ML4VRP competition. The system uses a Reinforcement Learning agent to dynamically control **PyVRP's Iterated Local Search (ILS)** solver, learning *when* to push for fewer vehicles vs. *when* to optimize routes.

**Competition Objective:**
```
Score = 1000 * NV + TD      (lower is better)
```
- **NV** = Number of Vehicles (fleet size)
- **TD** = Total Distance traveled

The 1000x multiplier on NV means eliminating one vehicle is worth 1000 distance units. Fleet minimization is the dominant factor.

**Dataset:** X-dataset (Uchoa et al., 2014) — 59 instances with 100-400 customers.

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
                |  Output: action (0-5) |
                +-----------------------+
                            |
                   action (strategy choice)
                            |
                            v
                +-----------------------+
                |   PyVRP ILS Solver    |
                |   (C++ engine)        |
                |                       |
                |  Runs 1000 iterations |
                |  with parameters set  |
                |  by the action        |
                |                       |
                |  Output: improved     |
                |  CVRP solution        |
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

**Purpose:** The RL agent. Decides which search strategy PyVRP should use at each step.

**Observation (132-dim):**
```
obs = [ graph_embedding (128) | solver_stats (4) ]
```

| Solver Stat | Formula | Meaning |
|---|---|---|
| time_ratio | steps_used / 20 | How far into the episode |
| nv_ratio | NV_current / NV_initial | Fleet reduction progress |
| violation_ratio | excess_load / total_demand | Capacity constraint pressure |
| stagnation_ratio | iters_no_improve / budget | Time since last improvement |

**Action Space (6 discrete actions):**

| # | Action | What it does | When useful |
|---|---|---|---|
| 0 | POLISH | Default solver params, same seed | Routes are good, just refine |
| 1 | MILD_PRESSURE | Penalty increase=2.0 | Gently push to remove a vehicle |
| 2 | MODERATE_PRESSURE | Penalty increase=5.0 | Steady fleet reduction |
| 3 | AGGRESSIVE_PRESSURE | Penalty increase=10.0 | Force route merges |
| 4 | EXPLORE_NEW_SEED | New random seed, default params | Stuck in local optimum |
| 5 | EXPLORE+PRESSURE | New seed + moderate penalty | Escape + reduce simultaneously |

**Action Masking:** When NV = NV_min (theoretical minimum fleet = ceil(total_demand / capacity)), pressure actions (1, 2, 3, 5) are blocked to prevent impossible fleet reductions.

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
Linear(64->6)  Linear(64->1)
     |             |
     v             v
action_logits  state_value
  (B, 6)        (B, 1)
```

- **Parameters:** ~13,100
- Intentionally small — the agent makes strategic decisions, not complex computations

---

### 3.3 PyVRP Solver Engine (`src/solver_engine.py`)

**Purpose:** The actual CVRP solver. A high-performance C++ Iterated Local Search (ILS) engine wrapped as a Gymnasium environment.

**Episode structure:**
```
reset()
  |-> Load random instance
  |-> Encode with GNN (once)
  |-> Initial solve (1000 iterations)
  |-> Return first observation
  |
  v
step(action) x 20 times
  |-> Map action to PyVRP parameters
  |-> Run solver for 1000 iterations
  |-> Compute reward = prev_score - new_score
  |-> Return (obs, reward, done, info)
  |
  v
Episode ends after 20 steps (= 20,000 total ILS iterations)
```

**How actions map to solver behavior:**

The Fleet Manager doesn't directly modify routes. Instead, it controls PyVRP's **penalty parameters**, which determine how aggressively the solver penalizes solutions that use "too many" vehicles:

- **Higher penalty_increase** = solver is punished more for adding vehicles = more pressure to merge routes
- **Lower target_feasible** = solver tolerates more infeasible intermediate solutions = explores harder
- **New seed** = solver starts its randomized search from a different point = escapes local optima

**Safety mechanisms:**
- **Fleet explosion detection:** If a pressure action causes NV to spike by >2, the bad solution is rejected and a penalty reward (-5.0) is returned
- **Warm starting:** Each step passes the best solution so far as the starting point for the next solve

---

### 3.4 PPO Training (`src/train.py`)

**Purpose:** Train the Fleet Manager to maximize cumulative reward using Proximal Policy Optimization.

**Training loop per epoch:**
```
For each of 8 episodes:
    1. Reset environment (random instance)
    2. For 20 steps:
        - Agent observes state
        - Agent picks action (with exploration)
        - Environment executes action
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
| Learning rate | 1e-4 | Adam optimizer, with linear decay |
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
     GNN Encoder          PyVRP reads
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
     Picks action (0-5)        |
           |                   |
           +-------------------+
           |
           v
     PyVRP solves with
     chosen parameters
     (1000 ILS iterations)
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
  solver_engine.py   - PyVRP Gymnasium environment (Stage 3)
  train.py           - PPO training loop (Stage 5)
  main.py            - Entry point, smoke tests, CLI
  agent_driver.py    - [UNUSED] Route Driver (kept for reference)

notebooks/
  colab_bridge.ipynb - Google Colab training notebook

scripts/
  setup_vm.sh        - Ubuntu GPU VM setup script

data/                - X-dataset .vrp instance files
logs/                - Training CSV metrics + best model
checkpoints/         - Periodic checkpoint .pth files
docs/                - Documentation
```

---

## 6. Training Metrics

When training runs, you'll see output like:
```
[  5/200] AvgScore: 63483 | AvgNV: 28.5 | AvgTD: 34983 | Eval: 42880 | PL: 0.0016 | Ent: 0.928 | LR: 9.9e-05
```

| Metric | What it means |
|---|---|
| AvgScore | Average score across 8 training episodes (noisy due to random instances) |
| AvgNV | Average vehicles used in training |
| AvgTD | Average total distance in training |
| **Eval** | **Score on 5 fixed instances (the real progress metric)** |
| PL | PPO policy loss (small values normal) |
| Ent | Policy entropy (>0.5 = still exploring, 0 = collapsed) |
| LR | Current learning rate |

**What to watch:** The Eval column should trend downward over time. AvgScore will be noisy — ignore it.

---

## 7. Model Sizes

| Component | Parameters | Purpose |
|---|---|---|
| GNN Encoder | ~51,600 | Spatial understanding |
| Fleet Manager | ~13,100 | Strategy selection |
| **Total** | **~64,700** | Entire trainable model |

The model is intentionally lightweight. The heavy lifting is done by PyVRP's C++ solver — our RL agent just learns to steer it effectively.

---

## 8. Why This Approach Works (Intuition)

### The Core Idea

The competition asks us to find the best CVRP solution (lowest `1000*NV + TD`). PyVRP is already an excellent solver — but it uses **fixed parameters**. The same penalty settings are applied regardless of whether the instance has tight clusters, spread-out customers, balanced demand, or skewed demand.

Our RL agent learns to **adapt the solver strategy to each instance and each moment in the search**. It's not solving the CVRP itself — it's learning to be a better operator of the PyVRP solver than any fixed configuration could be.

### How Does the Agent Understand the Problem?

The agent never sees raw routes or calculates optimal fleet size. Instead, it builds understanding from two sources:

**Spatial understanding (GNN Encoder):**
The Graph Attention Network reads the customer locations and demands, producing a 128-dim embedding that captures the instance's spatial structure. Instances with different characteristics (tight clusters, spread-out nodes, high/low demand variance) produce different embeddings. Over training, the agent learns that certain spatial patterns respond better to certain strategies.

**Solver feedback (Observation Stats):**
At each step, the agent sees 4 real-time signals from the solver:
- **nv_ratio** — "How much fleet reduction has happened so far?"
- **violation_ratio** — "Am I pushing too hard? Are capacity constraints being violated?"
- **stagnation_ratio** — "Has progress stalled? Should I try something different?"
- **time_ratio** — "How much budget is left?"

### The Learning Process

The agent doesn't know the right strategy in advance. It learns through trial and error over thousands of episodes:

1. Early in training, the agent picks actions randomly
2. Sometimes AGGRESSIVE_PRESSURE works and removes a vehicle → big positive reward (+1000)
3. Sometimes it backfires and the fleet explodes → penalty (-5.0)
4. Sometimes POLISH steadily reduces distance → small positive rewards
5. Sometimes EXPLORE finds a much better solution from a new seed → large positive reward

Over time, PPO adjusts the policy so the agent picks actions that led to positive outcomes more often. It develops patterns like:
- "When nv_ratio is high and violation is low → push harder (room to reduce fleet)"
- "When stagnation is high → explore with a new seed (stuck in local optimum)"
- "When nv_ratio is already near minimum → just polish routes"
- "Instances with this spatial pattern → aggressive pressure works early, then polish"

### Why Not Just Tune Parameters Manually?

You could hand-tune a fixed strategy like "always use moderate pressure." But:
- **Different instances need different strategies.** A 100-customer instance with tight clusters behaves very differently from a 400-customer spread-out instance.
- **Different phases of the search need different strategies.** Early on, aggressive pressure is useful. Late in the search, polishing matters more. After stagnation, exploration is key.
- **The interaction is complex.** The best sequence of 20 actions depends on how the solver responded to previous actions. No fixed rule captures this.

The RL agent learns a **dynamic, instance-adaptive policy** — something no hand-tuned configuration can match.

### Analogy

Think of PyVRP as a race car and the Fleet Manager as the driver:
- The **car** (PyVRP) has all the power — it does the actual route optimization
- The **driver** (Fleet Manager) decides when to accelerate (pressure), when to cruise (polish), and when to take a different road (explore)
- The **track** (CVRP instance) is different every time — the GNN reads the track layout
- **Training** is the driver practicing on many different tracks until they develop good instincts
- A car on autopilot with fixed settings will finish the race, but a skilled driver adapts to each track and gets a better time
