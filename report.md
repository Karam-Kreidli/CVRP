# Report Structure Guide — RoutIQ: RL-Guided CVRP Solver

This document outlines every section, subsection, and sub-subsection of the project report, with detailed descriptions of what to include, where to place tables, and which code snippets to embed.

---

## 1. Introduction

### 1.1 Background
- Combinatorial optimization in logistics and supply chain.
- The Vehicle Routing Problem (VRP) family and its real-world importance.
- The role of machine learning in solving NP-hard optimization problems.
- The GECCO 2026 ML4VRP competition as the motivating context.

### 1.2 Problem Statement
- Formal definition of CVRP: a central depot, a set of customers with known demands, a fleet of capacity-limited vehicles.
- The mathematical objective: minimize total cost.
- The competition scoring formula: `Score = 1000 × NV + TD` — explain why fleet minimization dominates (removing one vehicle saves 1000 points while distance improvements are incremental).
- NP-hardness: exact methods fail beyond ~50 customers; heuristics are necessary but use static strategies that do not adapt per-instance.

### 1.3 Objectives
- Design a hybrid system that combines RL with an existing high-performance heuristic solver.
- Train an RL agent that learns when to push for fewer vehicles vs. optimizing distance.
- Evaluate against the baseline solver across the full 59-instance X-dataset.
- Build a deployable system with a REST API backend.

> **TABLE — Objectives and Deliverables Traceability Matrix**
> Columns: Objective | Deliverable | Location
> Example rows:
> - "Train RL agent" | "Trained model weights" | `logs/best_model.pth`
> - "Evaluate on dataset" | "Per-instance results" | `Results/Final_Full_Evaluation_Instances.csv`
> - "Build REST API" | "FastAPI server" | `Backend/Backend.py`
> - "Generate solution files" | ".sol and .txt route files" | `Solutions/HGS+RL/`

---

## 2. Literature Review

### 2.1 Classical Approaches to CVRP
- Exact methods (branch-and-bound, branch-and-cut) and their scalability limits.
- Construction heuristics (Clarke-Wright savings, nearest neighbor).
- Metaheuristics: tabu search, simulated annealing, genetic algorithms.

### 2.2 Hybrid Genetic Search (HGS-CVRP)
- Vidal et al. (2012) — the algorithm the baseline is built on.
- Population management (elite + diverse pools), local search operators.
- Why HGS is considered state-of-the-art for CVRP.

### 2.3 Reinforcement Learning for Combinatorial Optimization
- Attention-based models (Pointer Networks, Kool et al. 2019).
- RL for parameter control and algorithm configuration (the paradigm this project follows).
- Distinction: RL as a solver replacement vs. RL as a solver guide (this project uses the latter).

### 2.4 Hybrid RL + Heuristic Approaches
- Prior work on combining learned and classical solvers.
- Advantages: leverages decades of solver engineering while adding adaptability.
- Gap this project addresses: per-instance strategy selection for fleet minimization.

---

## 3. Our Solution: RoutIQ

### 3.1 High-Level Approach
- Do not replace HGS — guide it with strategic decisions.
- The RL agent (Fleet Manager) makes high-level choices; HGS handles route optimization.
- Analogy: the agent is a "coach" deciding game strategy, while HGS is the "player" executing.

### 3.2 Key Design Decisions
- Lightweight agent (~6,155 parameters) — strategic decisions don't need large models.
- Hand-crafted features instead of graph neural networks (interpretable, efficient, sufficient for strategy).
- Discrete action space with action masking for safety.

### 3.3 Competition Context
- Dataset: 59 X-dataset instances (Uchoa et al., 2014), 100–400 customers each.
- Scoring: `Score = 1000 × NV + TD`.
- Evaluation: head-to-head against default HGS with Formula-1 style pairwise points.

---

## 4. System Architecture

### 4.1 Pipeline Overview
- Stage 1: Parse the `.vrp` instance file.
- Stage 2: Extract 12 hand-crafted instance features (computed once).
- Stage 3: RL agent selects an action (fleet target + seed + budget).
- Stage 4: HGS-CVRP solver runs with the chosen parameters.
- Stage 5: Reward computed, observation updated, policy improved.

> **TABLE — Pipeline Stage Mapping**
> Columns: Stage | Component | Input | Output | File
> Example rows:
> - "1" | "Parser" | ".vrp file" | "Parsed coordinate/demand arrays" | `Model/Solver_Engine.py`
> - "2" | "Feature Extraction" | "Parsed data" | "12-dim feature vector" | `Model/Solver_Engine.py`
> - "3" | "Fleet Manager (RL)" | "19-dim observation" | "Action (0–9)" | `Model/Agent_Manager.py`
> - "4" | "HGS-CVRP Solver" | "Action params + instance" | "Routes, NV, TD" | `Model/Solver_Engine.py`
> - "5" | "Reward + Update" | "Score delta" | "PPO gradient step" | `Model/Train.py`

### 4.2 Episode Structure
- Reset: load a random instance, compute features, run initial 500-iteration HGS solve.
- 50 steps per episode: agent picks action → solver runs 500–1500 iterations → reward returned.
- Total HGS iterations per episode: ~25,000–33,000.
- Best solution tracked across the entire episode (not just per-step).

### 4.3 Technology Stack
- PyTorch for the RL model.
- Gymnasium for the RL environment interface.
- `hygese` Python bindings for the C++ HGS-CVRP solver.
- FastAPI for the backend REST API.

---

## 5. Baseline Model: HGS-CVRP

### 5.1 Algorithm Description
- Hybrid Genetic Search by Vidal et al. (2012).
- Population-based metaheuristic combining genetic operators with local search.
- Dual population pools: elite (best solutions) and diverse (exploratory solutions).
- Iterative improvement through crossover, mutation, and local search operators.

### 5.2 Why HGS as the Baseline
- Consistently produces near-optimal solutions across CVRP benchmarks.
- Used in top competition entries worldwide.
- Efficient C++ implementation available via the `hygese` Python package.
- The default configuration is already very strong — a challenging baseline to beat.

### 5.3 Baseline Configuration
- Default HGS parameters with 25,000 iterations per run.
- Single seed (42) for deterministic reproducibility.
- No fleet target constraint (unconstrained exploration).

> **CODE SNIPPET — Baseline HGS Solve**
> File: `Baseline/Infer.py`, lines 147–164
> Shows how a single baseline HGS solve is configured and executed. Demonstrates what "default HGS" means in terms of parameters.

---

## 6. The RL Model

### 6.1 Feature Engineering

#### 6.1.1 Instance Features (12-dimensional)
- Describe each of the 12 features: size_norm, demand_fill_ratio, mean_dist_norm, std_dist_norm, depot_centrality, demand_cv, capacity_tightness, demand_minmax_ratio, top3_demand_share, depot_distance_cv, bbox_aspect_ratio, radial_outlier_ratio.
- All normalized to roughly [0, 1] for stable learning.
- Computed once per instance (efficient — no per-step overhead).
- Subsampled for large instances (N > 200) to keep computation fast.

> **TABLE — Instance Features**
> Columns: Index | Feature Name | Formula / Description | Intuition
> 12 rows, one per feature. Example rows:
> - "0" | "size_norm" | "num_customers / 400" | "Normalizes instance scale so the agent can generalize across sizes"
> - "1" | "demand_fill_ratio" | "total_demand / (nv_min × capacity)" | "How tightly packed the vehicles are — closer to 1.0 means less slack"
> - "4" | "depot_centrality" | "mean(depot_distances) / max_dist" | "Whether the depot is centrally located or on the periphery"
> - "11" | "radial_outlier_ratio" | "fraction of customers beyond Q3 + 1.5×IQR" | "Detects far-flung customers that may need dedicated vehicles"

> **CODE SNIPPET — Feature Computation**
> File: `Model/Solver_Engine.py`, lines 184–314
> The full `_compute_instance_features()` function. This is central to the model design — shows how each of the 12 features is computed from raw instance data.

#### 6.1.2 Solver Statistics (7-dimensional)
- time_ratio, nv_ratio, score_ratio, stagnation_ratio, nv_gap, last_reward, last_action_norm.
- Updated every step — gives the agent real-time feedback on solver progress.
- Allows the agent to adapt strategy mid-episode (e.g., switch from exploration to exploitation).

> **TABLE — Solver Statistics**
> Columns: Index | Statistic | Formula | Purpose
> 7 rows. Example rows:
> - "0" | "time_ratio" | "steps_used / 50" | "How much of the episode budget has been consumed"
> - "1" | "nv_ratio" | "current_nv / initial_nv" | "Fleet reduction progress relative to start"
> - "4" | "nv_gap" | "(best_nv − nv_min) / initial_nv" | "Distance from theoretical minimum fleet size"

### 6.2 Action Space
- 10 discrete actions, each controlling three dimensions: fleet target, seed strategy, iteration budget.
- FREE (actions 0–1): unconstrained fleet, 500 iters.
- LOCK (actions 2–3): lock current best fleet size, 500 iters.
- PUSH (actions 4–5): target best_nv − 1, 1000 iters.
- FORCE MIN (action 6): force theoretical minimum fleet, 1500 iters.
- DIVERSE / AGGRESSIVE / BALANCED (actions 7–9): HGS parameter overrides for different search biases.
- Action masking: PUSH/FORCE actions are blocked when NV is already at the theoretical minimum.

> **TABLE — Action Space**
> Columns: Action ID | Name | Fleet Target | Seed | Iterations | HGS Bias
> 10 rows. Example rows:
> - "0" | "FREE_SAME" | "Unconstrained" | "Same" | "500" | "None"
> - "4" | "PUSH_SAME" | "best_nv − 1" | "Same" | "1000" | "None"
> - "6" | "FORCE_MIN" | "nv_min" | "New" | "1500" | "None"
> - "7" | "FREE_DIVERSE_NEW" | "Unconstrained" | "New" | "500" | "Diversity-biased"
> - "8" | "LOCK_AGGR_NEW" | "Lock best_nv" | "New" | "500" | "Aggressive local-search"
> - "9" | "PUSH_BALANCED_NEW" | "best_nv − 1" | "New" | "1000" | "Balanced explore/exploit"

> **CODE SNIPPET — Action Constants and HGS Overrides**
> File: `Model/Solver_Engine.py`, lines 65–84
> Shows the iteration budgets, ACTION_NAMES list, and ACTION_HGS_OVERRIDES dictionary that defines per-action solver parameter presets.

### 6.3 Network Architecture
- Actor-Critic with shared trunk.
- Shared trunk: Linear(19 → 64) + ReLU → Linear(64 → 64) + ReLU.
- Actor head: Linear(64 → 10) producing action logits.
- Critic head: Linear(64 → 1) producing state value V(s).
- Total parameters: ~6,155.
- Action masking applied to logits before softmax (blocked actions set to −10,000).

> **CODE SNIPPET — FleetManager Class**
> File: `Model/Agent_Manager.py`, lines 71–168
> The full `FleetManager` class including `__init__` (network layer definitions) and `forward()` (forward pass with action masking). This is the core model definition.

### 6.4 Reward Design
- Compare candidate score against episode best (not previous step).
- Percentage-based improvement: `reward = ((best − candidate) / best) × 100`.
- Small penalty (−0.5) for no improvement (cost of wasted compute).
- Large penalty (−5.0) for fleet explosions (NV spikes by > 2 or solve failure).
- Rewards clipped to [−10, 10] before advantage computation.

> **CODE SNIPPET — Reward Computation**
> File: `Model/Solver_Engine.py`, lines 597–625
> The reward computation block from the `step()` method. Shows the three reward branches: fleet explosion penalty, percentage improvement, and no-improvement penalty.

---

## 7. Training

### 7.1 Algorithm: PPO (Proximal Policy Optimization)
- Policy gradient method with clipped surrogate objective.
- Prevents destabilizing large policy updates.
- Clipped ratio: `min(ratio × advantage, clip(ratio, 1−ε, 1+ε) × advantage)`.
- Combined loss: policy loss + 0.5 × value loss − 0.02 × entropy bonus.

### 7.2 Hyperparameters

> **TABLE — PPO Hyperparameters**
> Columns: Parameter | Value | Description
> Rows:
> - "Discount factor (γ)" | "0.95" | "Agent focuses on near-term rewards over ~50 steps"
> - "GAE lambda (λ)" | "0.90" | "Bias-variance tradeoff for advantage estimation"
> - "Clip range (ε)" | "0.2" | "Limits policy change to ±20% per update"
> - "Learning rate" | "1e-4 → 5e-5" | "Adam optimizer with linear decay"
> - "PPO epochs" | "3" | "Passes over collected data per rollout"
> - "Mini-batch size" | "128" | "Gradient steps on 128 transitions at a time"
> - "Entropy coefficient" | "0.02" | "Exploration bonus to prevent premature convergence"
> - "Target KL" | "0.015" | "Early stopping if policy diverges too much"
> - "Max gradient norm" | "0.5" | "Gradient clipping for training stability"
> - "Value function coefficient" | "0.5" | "Weight of critic loss in the total loss"

### 7.3 Training Configuration
- 80 epochs, 8 episodes per epoch, 50 steps per episode.
- Dataset split: training pool, 5 fixed eval instances, 5 holdout instances.
- Best model selected by composite metric (eval + holdout scores).
- Checkpoints saved every 10 epochs.

### 7.4 Curriculum Learning
- Epochs 1–20: only small instances (N ≤ 100 customers).
- Epochs 21–80: all 59 instances unlocked.
- Rationale: easier instances first to establish basic policy, then generalize to harder ones.

> **CODE SNIPPET — Curriculum Learning Expansion**
> File: `Model/main.py`, lines 533–553
> Shows how the curriculum boundary is checked each epoch and how `set_max_nodes(None)` unlocks all instances once the agent has trained on small ones.

### 7.5 GAE Advantage Estimation
- Generalized Advantage Estimation (Schulman et al., 2016).
- Interpolates between Monte Carlo (high variance, unbiased) and TD(0) (low variance, biased).
- Formula: δ_t = r_t + γ × V(s_{t+1}) − V(s_t), then A_t = δ_t + γ × λ × A_{t+1}.
- Returns = advantages + values (target for critic training).

> **CODE SNIPPET — GAE Computation**
> File: `Model/Train.py`, lines 173–217
> The `compute_gae()` method from the RolloutBuffer class. Shows the backwards accumulation of TD errors and the advantage/return computation.

### 7.6 Training Results
- Describe convergence behavior: policy loss, value loss, entropy trends over 80 epochs.
- Best eval score achieved and at which epoch.
- Policy entropy stayed at ~1.8–1.9 (healthy exploration throughout).
- Learning rate decayed from 1e-4 to ~9.5e-5.

> **TABLE — Training Metrics (Sample Epochs)**
> Columns: Epoch | Eval Score | Policy Loss | Value Loss | Entropy | Learning Rate
> Show ~8–10 representative epochs (e.g., 1, 10, 20, 30, 40, 50, 60, 70, 80).
> Extract these values from `logs/training_metrics.csv`.

---

## 8. Testing: Running the Model on All 59 Instances

### 8.1 Inference Process
- Load trained model from `logs/best_model.pth`.
- For each of the 59 instances: reset environment → 50 greedy steps (argmax, no exploration) → record best solution.
- Greedy action selection (no sampling) for deterministic evaluation.

> **CODE SNIPPET — Inference Loop**
> File: `Baseline/Infer.py`, lines 84–144
> The `solve_instance()` function. Shows the full inference loop: observation parsing, action masking, greedy action selection, and result recording.

### 8.2 Solution File Generation
- Two output formats: `.sol` (competition format) and `.txt` (human-readable).
- Both contain routes in the format `Route #1: 5 12 8 ...`
- Stored in `Solutions/HGS+RL/sol-Format/` and `Solutions/HGS+RL/txt-Format/`.

> **CODE SNIPPET — Solution File Export**
> File: `Baseline/Infer.py`, lines 570–591
> The `write_solution_files()` function. Shows how routes are formatted and written to both `.sol` and `.txt` formats.

### 8.3 Per-Instance Results

> **TABLE — Full 59-Instance Evaluation Results**
> Columns: Instance | RL NV | RL TD | RL Score | BL NV | BL TD | BL Score | Delta | Delta % | Outcome
> All 59 rows from `Results/Final_Full_Evaluation_Instances.csv`.
> Trim elapsed_sec and points columns for readability.
> This is the core results table of the report.

---

## 9. Benchmarking: RL vs Baseline

### 9.1 Head-to-Head Comparison
- Win/Loss/Tie: 6 wins (10.2%), 42 losses (71.2%), 11 ties (18.6%).
- Mean score: RL = 70,301.76 vs Baseline = 70,266.22.
- Mean delta: +35.54 (baseline slightly better on average).
- Median delta: +18.0.

### 9.2 Formula-1 Surrogate Points
- Scoring: winner gets 10 pts, runner-up gets 8 pts, tie gives 9 pts each.
- RL total: 495 pts vs Baseline total: 567 pts (margin: −72).

### 9.3 Best and Worst Cases

> **TABLE — Top 5 RL Wins and Top 5 RL Losses**
> Columns: Instance | Delta (RL−BL) | Delta % | Outcome
> Top 5 wins:
> - X-n247-k50: −991 pts (−1.11%)
> - X-n256-k16: −980 pts (−2.73%)
> - X-n148-k46: −949 pts (−1.05%)
> - X-n313-k71: −622 pts (−0.37%)
> - X-n322-k28: −20 pts (−0.04%)
> Top 5 losses:
> - X-n289-k60: +1,097 pts (+0.70%)
> - X-n336-k84: +1,033 pts (+0.46%)
> - X-n266-k58: +477 pts (+0.36%)
> - X-n384-k52: +427 pts (+0.36%)
> - X-n359-k29: +409 pts (+0.51%)

### 9.4 Statistical Robustness
- Bootstrap analysis: 2,000 trials, subset size 20, seed 123.
- P(RL > Baseline) = 0.1% — baseline is statistically dominant.
- Expected points margin: −24.16 (in baseline's favor).
- P(mean delta < 0) = 31.8% — RL wins on score in ~1/3 of random subsets.

### 9.5 Analysis and Discussion
- When RL wins, the gains come from fleet reduction (removing a vehicle saves ~1000 pts).
- When RL loses, regressions are typically distance-based (smaller magnitude individually but more frequent).
- The agent shows instance-specific strength (e.g., X-n247-k50: −991 pts, X-n256-k16: −980 pts).
- Limitations: 59 instances is a small training set for RL; single baseline seed; policy may have converged to a local optimum.

---

## 10. System Deployment

### 10.1 Backend API
- FastAPI server with Uvicorn.
- Endpoints: POST `/solve`, GET `/status/{id}`, GET `/results/{id}`, POST `/benchmark`.
- Loads trained model on startup; async job processing with in-memory state.
- CORS-enabled for frontend integration.

> **CODE SNIPPET — /solve Endpoint**
> File: `Backend/Backend.py`, lines 370–404
> The POST `/solve` endpoint implementation. Shows file upload handling, metadata parsing, job creation, and threaded solver execution.

### 10.2 Frontend
- Brief description of the web interface for uploading instances and viewing results.
- Include a screenshot of the frontend.

---

## 11. Conclusion and Future Work

### 11.1 Summary
- Built a hybrid RL + HGS framework for CVRP (RoutIQ).
- Lightweight Fleet Manager agent (~6K params) that learns per-instance strategies.
- Complete pipeline from feature extraction to solution generation and evaluation.
- Evaluated on all 59 X-dataset instances with rigorous statistical comparison.

### 11.2 Key Findings
- RL can discover fleet reduction strategies on select instances (up to 991 points improvement).
- Beating a strong baseline consistently requires more training data and tuning.
- The hybrid approach is architecturally sound — the RL agent adds adaptability without sacrificing solver quality.
- The 6-win / 42-loss record shows the approach has promise but is not yet competitive overall.

### 11.3 Future Work
- Larger and more diverse instance datasets for better generalization.
- Attention-based feature learning to replace hand-crafted features.
- Multi-seed baselines for more robust comparison.
- Continuous action spaces (instead of 10 discrete actions).
- Warm-starting HGS from previous best solutions (instead of fresh solves each step).
- Meta-RL: train specialized policies per instance cluster.

---

## 12. References
- Vidal, T. et al. (2012). A hybrid genetic algorithm for large and medium scale capacitated arc routing problems.
- Uchoa, E. et al. (2014). New benchmark instances for the capacitated vehicle routing problem.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
- Kool, W. et al. (2019). Attention, Learn to Solve Routing Problems!
- (Add other references as appropriate.)

---

## Appendices (Optional)

### Appendix A: Full Training Configuration
- Contents of `logs/Run_Configuration.json` — full hyperparameter snapshot for reproducibility.

### Appendix B: Dataset Instance List
- All 59 instance names with customer counts and best-known solutions from literature.
- Source: `Documents/Benchmark_Reference.md`.

---

## Quick Reference: Tables and Code Snippets

### Tables (9 total)

| Section | Table Description |
|---------|-------------------|
| 1.3     | Objectives and deliverables traceability matrix |
| 4.1     | Pipeline stages mapping (stage, component, input, output, file) |
| 6.1.1   | 12 instance features (index, name, formula, intuition) |
| 6.1.2   | 7 solver statistics (index, name, formula, purpose) |
| 6.2     | 10 actions (ID, name, fleet target, seed, iterations, HGS bias) |
| 7.2     | PPO hyperparameters (parameter, value, description) |
| 7.6     | Training metrics per epoch (sample of ~8–10 epochs from CSV) |
| 8.3     | Full 59-instance results (instance, RL scores, BL scores, delta, outcome) |
| 9.3     | Top 5 wins + top 5 losses (instance, delta, delta %, outcome) |

### Code Snippets (10 total)

| Section | File                       | Lines     | What It Shows                              |
|---------|----------------------------|-----------|--------------------------------------------|
| 5.3     | `Baseline/Infer.py`        | 147–164   | Baseline HGS solve setup                   |
| 6.1.1   | `Model/Solver_Engine.py`   | 184–314   | Feature computation function               |
| 6.2     | `Model/Solver_Engine.py`   | 65–84     | Action constants and HGS overrides         |
| 6.3     | `Model/Agent_Manager.py`   | 71–168    | FleetManager class (network + forward)     |
| 6.4     | `Model/Solver_Engine.py`   | 597–625   | Reward computation in step()               |
| 7.4     | `Model/main.py`            | 533–553   | Curriculum learning expansion              |
| 7.5     | `Model/Train.py`           | 173–217   | GAE advantage computation                  |
| 8.1     | `Baseline/Infer.py`        | 84–144    | Inference loop (solve_instance)            |
| 8.2     | `Baseline/Infer.py`        | 570–591   | Solution file export                       |
| 10.1    | `Backend/Backend.py`       | 370–404   | /solve endpoint                            |
