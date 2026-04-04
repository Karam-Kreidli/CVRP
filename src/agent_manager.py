"""
Stage 2 - Fleet Manager: The RL agent that controls the solver.

This is the "brain" of the system. It looks at hand-crafted instance features
and real-time solver statistics, then decides the solving strategy for the
next step: whether to push for fewer vehicles, try a new seed, or refine.

WHAT IT DOES:
  At each of the 50 steps in an episode, the Fleet Manager looks at:
    1. Instance features (7-dim — size, demand, distance stats, etc.)
    2. Seven real-time statistics from the solver ("how is the search going?")
  And chooses one of 7 strategic actions for the next solve.

OBSERVATION (what the agent sees):
  14-dim vector = [instance_features (7) | solver_stats (7)]

  Instance features (computed once per episode):
    - size_norm, demand_fill_ratio, mean_dist_norm, std_dist_norm,
      depot_centrality, demand_cv, capacity_tightness

  Solver stats (updated each step):
    - time_ratio, nv_ratio, score_ratio, stagnation_ratio,
      nv_gap, last_reward, last_action_norm

ACTION SPACE (what the agent can do):
  Each action controls (fleet_target, seed_strategy, iteration_budget):
  0 = FREE_SAME    — Free fleet, same seed, 500 iters
  1 = FREE_NEW     — Free fleet, new seed, 500 iters
  2 = LOCK_SAME    — Lock best NV, same seed, 500 iters
  3 = LOCK_NEW     — Lock best NV, new seed, 500 iters
  4 = PUSH_SAME    — Push best_nv-1, same seed, 1000 iters
  5 = PUSH_NEW     — Push best_nv-1, new seed, 1000 iters
  6 = FORCE_MIN    — Force nv_min, new seed, 1500 iters

ARCHITECTURE: Actor-Critic
  The network has two "heads" sharing a common trunk:
    - Actor: outputs a probability distribution over the 7 actions (policy)
    - Critic: outputs a single number estimating "how good is this state?" (value)
  PPO uses both during training — the actor learns better actions, the critic
  learns to evaluate states for computing advantages (see train.py).

Parameters: ~5,700 (intentionally tiny — it's making strategic decisions, not
computing routes)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


# Observation dimensions — must match solver_engine.py
INSTANCE_FEATURES_DIM = 7  # Hand-crafted instance features
SOLVER_STATS_DIM = 7       # [time_ratio, nv_ratio, score_ratio, stagnation, nv_gap, last_reward, last_action]

# Total number of discrete strategy choices available to the agent
NUM_FLEET_ACTIONS = 7

# Human-readable names for each action (used in logging and debugging)
ACTION_NAMES = [
    "FREE_SAME", "FREE_NEW", "LOCK_SAME",
    "LOCK_NEW", "PUSH_SAME", "PUSH_NEW", "FORCE_MIN",
]


class FleetManager(nn.Module):
    """Actor-Critic neural network for solver strategy selection.

    This is the trainable RL agent. It takes the current state (instance
    features + solver stats) and outputs:
      1. Action probabilities — which strategy to use next
      2. State value — how good is the current situation (for PPO training)

    The network is deliberately simple (2 hidden layers of 64 units) because
    the decisions are strategic, not computational. The solver does the heavy
    lifting; this network just learns to steer it.

    Args:
        embed_dim: Dimension of instance features (7).
        stats_dim: Dimension of solver statistics vector (7).
        hidden_dim: Width of hidden layers (64).
        num_actions: Number of discrete strategy choices (7).
    """

    def __init__(
        self,
        embed_dim: int = INSTANCE_FEATURES_DIM,
        stats_dim: int = SOLVER_STATS_DIM,
        hidden_dim: int = 64,
        num_actions: int = NUM_FLEET_ACTIONS,
    ):
        super().__init__()
        input_dim = embed_dim + stats_dim  # 7 + 7 = 14

        # --- Shared Feature Trunk ---
        # Both the actor and critic share these layers. This is efficient and
        # helps the critic's value estimates stay aligned with the actor's policy.
        # Two layers of 64 units with ReLU activation.
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 14 → 64
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 64 → 64
            nn.ReLU(),
        )

        # --- Actor Head (Policy) ---
        # Outputs raw logits (unnormalized scores) for each of the 7 actions.
        # These logits are converted to probabilities via softmax in Categorical().
        # Higher logit = higher probability of choosing that action.
        self.actor = nn.Linear(hidden_dim, num_actions)  # 64 → 7

        # --- Critic Head (Value Function) ---
        # Outputs a single scalar V(s) — the estimated cumulative future reward
        # from the current state. PPO uses this to compute "advantages" (how much
        # better an action was compared to the average action in that state).
        self.critic = nn.Linear(hidden_dim, 1)  # 64 → 1

    def forward(
        self,
        graph_embedding: torch.Tensor,
        solver_stats: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: observation → (action logits, state value).

        This is the core computation. Given the current state of the world
        (what the instance looks like + how the solver is doing), it outputs:
          - Which actions look promising (logits)
          - How good the current situation is overall (value)

        ACTION MASKING:
          When the fleet is already at the theoretical minimum (NV <= NV_min),
          fleet-reduction actions (4=PUSH_SAME, 5=PUSH_NEW, 6=FORCE_MIN) are
          blocked by setting their logits to -10000, making their probability
          essentially zero after softmax.

        Args:
            graph_embedding: (B, 7) hand-crafted instance features.
            solver_stats:    (B, 7) real-time solver feedback.
            action_mask:     (B, 7) bool tensor. True = action allowed, False = blocked.

        Returns:
            action_logits: (B, 7) raw scores for each action (masked ones set to -1e4).
            state_value:   (B, 1) estimated future reward from this state.
        """
        # Concatenate instance features + solver feedback into one observation
        obs = torch.cat([graph_embedding, solver_stats], dim=-1)  # (B, 14)

        # Shared feature extraction
        features = self.shared(obs)           # (B, 64)

        # Actor: score each possible action
        action_logits = self.actor(features)  # (B, 6)

        # Apply action mask: set blocked actions to -10000 so softmax gives ~0 probability
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, -1e4)

        # Critic: estimate state value
        state_value = self.critic(features)   # (B, 1)

        return action_logits, state_value

    def select_action(
        self,
        graph_embedding: torch.Tensor,
        solver_stats: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy (used during training rollouts).

        Unlike forward() which just returns logits, this method:
          1. Converts logits to a probability distribution (Categorical)
          2. SAMPLES an action (stochastic — enables exploration)
          3. Records the log-probability (needed for PPO's importance ratio)

        During evaluation, we use forward() + argmax instead (greedy, no randomness).

        Args:
            graph_embedding: (B, 7) hand-crafted instance features.
            solver_stats:    (B, 7) solver statistics.
            action_mask:     (B, 7) bool tensor. True = allowed, False = masked.

        Returns:
            action:      (B,) sampled action index (0-6).
            log_prob:    (B,) log-probability of the sampled action (for PPO).
            state_value: (B, 1) critic's value estimate (for advantage computation).
        """
        action_logits, state_value = self.forward(
            graph_embedding, solver_stats, action_mask=action_mask
        )
        # Create a categorical distribution from the logits
        # (softmax is applied internally by Categorical)
        dist = Categorical(logits=action_logits)

        # Sample an action — this is stochastic, so the agent explores
        # different strategies during training
        action = dist.sample()

        # Record log-probability — PPO needs this to compute the ratio
        # π_new(a|s) / π_old(a|s) for its clipped surrogate objective
        log_prob = dist.log_prob(action)

        return action, log_prob, state_value
