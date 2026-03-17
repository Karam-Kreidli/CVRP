# Stage 2 & 4: Dual-Agent MARL Hierarchy
- **Agent 1 (Fleet Manager):** Operates at the Global/Strategic level.
  - Input: Graph Embedding + Current Solver Stats (Iteration, current NV).
  - Actions: Selects search modes (e.g., ROUTE_ELIMINATION, INTENSIVE_POLISH).
- **Agent 2 (Route Driver):** Operates at the Local/Tactical level.
  - Input: Node Embeddings of customers in a specific route.
  - Actions: Selects the best Neighborhood Operator (e.g., 2-Opt, SwapStar, Relocate).
- **Synergy:** Agent 1 sets the fleet goal; Agent 2 optimizes the distance within that goal.

---

## Agent 1: Fleet Manager — Detailed Architecture

### Observation Space (132-dim)
Concatenation of two vectors every K HGS iterations:

#### Graph Embedding (128-dim)
- Source: `GNNEncoder.forward()` → `graph_embedding` via global mean-pool
- Encodes global instance topology (spatial clusters, demand pressure)
- Computed once per instance, reused across solver iterations

#### Solver Statistics (4-dim)
Normalized runtime metrics from the HGS engine (Stage 3):

| Index | Name | Formula | Range | Semantics |
|-------|------|---------|-------|-----------|
| 0 | Time Ratio | t / T_max | [0, 1] | Progress through iteration budget |
| 1 | NV Ratio | NV_current / NV_initial | (0, 1] | Fleet reduction progress |
| 2 | Violation Ratio | total_capacity_overflow / total_demand | [0, ∞) | Feasibility pressure |
| 3 | Stagnation Ratio | iters_since_improvement / T_max | [0, 1] | Time since last BKS improvement |

### Action Space (Discrete, 3)
| Action | Name | Effect |
|--------|------|--------|
| 0 | KEEP | Maintain current fleet size |
| 1 | REMOVE | Attempt to reduce fleet by one vehicle |
| 2 | ADD | Restore one vehicle (recovery) |

### Network Architecture
```
obs (B, 132) = graph_embedding (B, 128) || solver_stats (B, 4)
                              ↓
                    Linear(132 → 64) + ReLU
                              ↓
                    Linear(64 → 64) + ReLU
                         ↙         ↘
              Actor: Linear(64 → 3)   Critic: Linear(64 → 1)
              action_logits (B, 3)    state_value (B, 1)
```

- **Parameters:** ~12,868 (lightweight — decisions are infrequent)
- **Training:** PPO with GAE(λ); reward = Δ(1000·NV + TD) between decisions
- **FP16 compatible:** All ops (Linear, ReLU, cat) are AMP-safe
- **Implementation:** `src/agent_manager.py` → `FleetManager(nn.Module)`

---

## Agent 2: Route Driver — Detailed Architecture

### Input: Node Embeddings (N × 128)
- Source: `GNNEncoder.forward()` → `node_embeddings` (before mean-pooling)
- Each row `h_i ∈ ℝ^128` encodes spatial position, demand, and local topology of customer `i`
- Variable-length: N ∈ [101, 401] depending on instance size

### Attention Pooling: (N, 128) → (1, 128)
Multi-Head Attention that learns which nodes are most informative for operator selection — focusing on "bottleneck" customers (tight clusters, near-capacity boundaries).

**Mathematical formulation** (H = 4 heads, d_k = 32 per head):

```
Given node embedding matrix  H ∈ ℝ^{N × 128}:

1. Query (instance summary):
   q = W_q · mean(H)                          q ∈ ℝ^{1 × 128}

2. Keys and Values (per-node):
   K = H · W_k^T                              K ∈ ℝ^{N × 128}
   V = H · W_v^T                              V ∈ ℝ^{N × 128}

3. Multi-head split (h = 1..4, d_k = 32):
   q_h ∈ ℝ^{1 × 32},  K_h ∈ ℝ^{N × 32},  V_h ∈ ℝ^{N × 32}

4. Scaled dot-product attention per head:
   α_h = softmax( q_h · K_h^T / √32 )        α_h ∈ ℝ^{1 × N}
   c_h = α_h · V_h                            c_h ∈ ℝ^{1 × 32}

5. Concatenate and project:
   context = [c_1 ‖ c_2 ‖ c_3 ‖ c_4] · W_o   context ∈ ℝ^{1 × 128}
```

**Complexity:** O(N · d) per head — linear in node count, handles 400 nodes in <1ms.

### Action Space (Discrete, 4)
| Action | Name | Effect | When Useful |
|--------|------|--------|-------------|
| 0 | TWO_OPT | Reverse a sub-sequence within a route | Untangling crossed route segments |
| 1 | SWAP | Exchange two customers between routes | Rebalancing load across routes |
| 2 | RELOCATE | Move a customer from one route to another | Reducing route count (NV penalty) |
| 3 | SWAP_STAR | Advanced cross-route exchange | Most powerful; expensive but high quality |

### Network Architecture
```
node_embeddings (N, 128)
           ↓
  AttentionPooling (4 heads, d_k=32)
           ↓
  tactical context (1, 128)
           ↓
  Linear(128 → 64) + ReLU
           ↓
  Linear(64 → 64) + ReLU
        ↙         ↘
Actor: Linear(64 → 4)   Critic: Linear(64 → 1)
operator_logits (1, 4)   state_value (1, 1)
```

- **Parameters:** ~75K (attention: ~65K, MLP: ~10K)
- **Trigger:** Called by CVRPEnv when Fleet Manager selects `INTENSIVE_POLISH` (action 0)
- **Training:** PPO with GAE(λ); reward = Δ(TD) within INTENSIVE_POLISH steps
- **FP16 compatible:** All ops (Linear, softmax, bmm) are AMP-safe
- **Implementation:** `src/agent_driver.py` → `RouteDriver(nn.Module)`