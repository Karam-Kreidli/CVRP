# Stage 5: Multi-Agent Training Loop

## Overview
Joint PPO optimization of the Fleet Manager and Route Driver using the
Multi-Agent Credit Assignment (MACA) framework. The competition objective
`1000 × NV + TD` is decomposed into agent-specific reward signals to enable
independent learning while maintaining cooperative behavior.

## Hyperparameters

| Parameter | Manager | Driver | Description |
|-----------|---------|--------|-------------|
| Learning Rate | 3×10⁻⁴ | 1×10⁻⁴ | Adam optimizer LR |
| γ (gamma) | 0.99 | 0.99 | Discount factor |
| λ (GAE-lambda) | 0.95 | 0.95 | GAE bias-variance tradeoff |
| ε (epsilon-clip) | 0.2 | 0.2 | PPO clipping range |
| c₁ (vf_coeff) | 0.5 | 0.5 | Value function loss coefficient |
| c₂ (ent_coeff) | 0.01 | 0.01 | Entropy bonus coefficient |
| PPO epochs | 4 | 4 | Update epochs per rollout |
| Mini-batch size | 64 | 1* | Mini-batch size |
| Max grad norm | 0.5 | 0.5 | Gradient clipping threshold |
| Target KL | 0.015 | 0.015 | Early stopping threshold |

\* Driver processes transitions individually due to variable-size node embeddings (N ∈ [101, 401]).

## MACA Credit Assignment

The competition objective `1000 × NV + TD` is decomposed into two reward streams:

### Manager Reward
```
R_manager = 1000 × ΔNV = 1000 × (NV_prev − NV_current)
```
- **+1000** for each vehicle eliminated
- **−1000** for each vehicle added
- Directly targets the dominant term in the objective (1000× multiplier)

### Driver Reward
```
R_driver = ΔTD = TD_prev − TD_current
```
- Positive when total distance decreases
- Focuses the Driver on route quality within the fleet size set by the Manager

### Reward Normalization
Running Mean/Std scaling (Welford's online algorithm) is applied to each
reward stream independently, preventing the Manager's ±1000 vehicle rewards
from causing gradient explosions:

```
R_normalized = (R − μ_running) / (σ_running + ε)
```

where μ and σ are updated incrementally after each rollout collection.

## PPO Clipped Objective

For each agent independently:

```
L = L_policy + c₁ · L_value − c₂ · H[π]

where:
  ratio     = exp(log π_θ(a|s) − log π_θ_old(a|s))
  L_policy  = −E[min(ratio · Â, clip(ratio, 1−ε, 1+ε) · Â)]
  L_value   = E[(V_θ(s) − R_target)²]
  H[π]      = −E[Σ_a π(a|s) log π(a|s)]           (entropy bonus)
  Â         = GAE(γ, λ) advantages, normalized per mini-batch
```

Advantages are computed via **Generalized Advantage Estimation**:
```
δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
Â_t = Σ_{l=0}^{T−t} (γλ)^l · δ_{t+l}
```

## Architecture Decisions

- **Two separate optimizers**: Different learning rates reflect the agents'
  different roles — the Manager makes infrequent high-impact decisions (3×10⁻⁴),
  while the Driver makes frequent tactical choices (1×10⁻⁴).
- **FP16 Mixed Precision**: `torch.amp.autocast("cuda")` + `GradScaler`
  for ~2× throughput on T4 GPU. All agent ops (Linear, ReLU, softmax, bmm)
  are AMP-safe.
- **Early KL stopping**: Halts PPO epochs if the approximate KL divergence
  exceeds 1.5 × target_kl, preventing destructive policy updates.
- **Separate RolloutBuffers**: Manager uses fixed 132-dim observations;
  Driver stores variable-length (N, 128) node embeddings.

## Implementation

| File | Contents |
|------|----------|
| `src/train.py` | `MARLTrainer`, `RolloutBuffer`, `DriverRolloutBuffer`, `RunningMeanStd`, `PPOConfig` |
| `src/main.py` | Training entry point with argparse, checkpoint saving, Google Drive push |

### Usage
```bash
# Smoke tests (all stages including training)
python -m src.main

# Full training
python -m src.main train --instance_path data/ --epochs 100 --fp16

# With Google Drive checkpoint backup
python -m src.main train --instance_path data/ --epochs 100 --fp16 \
    --gdrive_path /content/drive/MyDrive/ml4vrp
```