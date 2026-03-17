"""
Stage 2 - Fleet Manager: Actor-Critic agent for vehicle count decisions.

Consumes the global graph embedding from Stage 1 (GNNEncoder) concatenated
with a 4-dim solver statistics vector to make strategic fleet-sizing decisions.

The 1000x NV multiplier in the objective (1000*NV + TD) means that removing
even one vehicle is worth 1000 distance units — this agent's primary job is
to find the minimum feasible fleet size.

Observation space: (B, 132) = (B, 128) graph_embedding || (B, 4) solver_stats
  - graph_embedding: global topology from GNNEncoder mean-pool
  - solver_stats: [time_ratio, nv_ratio, violation_ratio, stagnation_ratio]

Action space (discrete, 3):
  0 = KEEP   — maintain current fleet size
  1 = REMOVE — attempt to remove one vehicle (aggressive)
  2 = ADD    — add one vehicle back (recovery)

Critic output: scalar state-value V(s) for PPO training.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


SOLVER_STATS_DIM = 4  # [time_ratio, nv_ratio, violation_ratio, stagnation_ratio]
NUM_FLEET_ACTIONS = 3  # KEEP, REMOVE, ADD


class FleetManager(nn.Module):
    """Actor-Critic network for fleet size management.

    Args:
        embed_dim: Dimension of the graph embedding from GNNEncoder (default: 128).
        stats_dim: Dimension of the solver statistics vector (default: 4).
        hidden_dim: Width of hidden layers (default: 64).
        num_actions: Discrete action space size (default: 3).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        stats_dim: int = SOLVER_STATS_DIM,
        hidden_dim: int = 64,
        num_actions: int = NUM_FLEET_ACTIONS,
    ):
        super().__init__()
        input_dim = embed_dim + stats_dim  # 132

        # Shared feature trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head — policy logits over {KEEP, REMOVE, ADD}
        self.actor = nn.Linear(hidden_dim, num_actions)

        # Critic head — scalar state value V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        graph_embedding: torch.Tensor,
        solver_stats: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing policy logits and state value.

        Args:
            graph_embedding: (B, 128) from GNNEncoder.
            solver_stats:    (B, 4)   [time_ratio, nv_ratio, violation_ratio, stagnation].
            action_mask:     (B, 3) bool tensor. True = allowed, False = masked.

        Returns:
            action_logits: (B, 3) raw logits over fleet actions (masked actions set to -1e9).
            state_value:   (B, 1) estimated state value V(s).
        """
        obs = torch.cat([graph_embedding, solver_stats], dim=-1)  # (B, 132)
        features = self.shared(obs)       # (B, 64)
        action_logits = self.actor(features)  # (B, 3)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, -1e4)
        state_value = self.critic(features)   # (B, 1)
        return action_logits, state_value

    def select_action(
        self,
        graph_embedding: torch.Tensor,
        solver_stats: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy for environment interaction.

        Args:
            graph_embedding: (B, 128) from GNNEncoder.
            solver_stats:    (B, 4) solver statistics.
            action_mask:     (B, 3) bool tensor. True = allowed, False = masked.

        Returns:
            action:    (B,) sampled action indices.
            log_prob:  (B,) log-probability of the sampled actions.
            state_value: (B, 1) critic estimate.
        """
        action_logits, state_value = self.forward(
            graph_embedding, solver_stats, action_mask=action_mask
        )
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, state_value
