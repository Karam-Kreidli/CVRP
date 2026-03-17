# Stage 3: Hybrid Genetic Search (HGS) via PyVRP
- **Framework:** PyVRP (high-performance C++ engine with Python wrapper).
- **Split Algorithm:** Mathematically optimal partitioning of giant tours into routes.
- **Crossover:** Selective Route Exchange (SREX).
- **Customization:** The HGS loop must be interruptible to allow Agent 2 to "educate" individuals after crossover using learned operator selection.

---

## Gymnasium Environment: `CVRPEnv`

**Note:** PyVRP v0.14+ uses Iterated Local Search (ILS), not a Genetic Algorithm with a population. Diversity is controlled via perturbation range and random seeds, not population parameters.

### Episode Structure
```
reset() ──→ Initial solve (1K iters) ──→ obs_0
  └──→ step(action_0) ──→ obs_1, reward_0
  └──→ step(action_1) ──→ obs_2, reward_1
  └──→ ...
  └──→ step(action_19) ──→ obs_20, reward_19, truncated=True
```
- **Budget:** 20 steps × 1,000 iterations = 20,000 total ILS iterations per episode
- **Warm start:** Each step passes `best_solution` as `initial_solution` for continuity

### Action → PyVRP Parameter Mapping

| Action | Name | PyVRP Behavior | Strategic Intent |
|--------|------|----------------|------------------|
| 0 | `INTENSIVE_POLISH` | Default `SolveParams()`, same seed | Exploit: optimize within current fleet |
| 1 | `ROUTE_ELIMINATION` | `PenaltyParams(penalty_increase=5.0, target_feasible=0.3, min_penalty=10.0)` | Aggressive: high penalties force route merging |
| 2 | `DIVERSITY_EXPLORE` | Default params, **new random seed** | Explore: escape local optima |

### Reward Function
```
reward = prev_score - new_score    (where score = 1000 * NV + TD)
```
Positive reward = score improved. Removing one vehicle yields up to +1000.

### Implementation
- File: `src/solver_engine.py`
- Class: `CVRPEnv(gym.Env)`
- Dependencies: `pyvrp`, `gymnasium`, `torch`, `src.model_vision.GNNEncoder`