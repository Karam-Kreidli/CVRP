"""
Stage 2 - Fleet Manager: The RL agent that controls the solver.

This is the "brain" of the system. While the GNN Encoder (Stage 1) understands
the spatial structure of a CVRP instance, the Fleet Manager makes DECISIONS
about how to run the solver.

WHAT IT DOES:
  At each of the 20 steps in an episode, the Fleet Manager looks at:
    1. The instance embedding (128-dim vector from the GNN — "what does this problem look like?")
    2. Four real-time statistics from the solver ("how is the search going?")
  And chooses one of 6 search strategies for the next 1000 solver iterations.

WHY THIS MATTERS:
  The competition score is 1000*NV + TD. Removing one vehicle saves 1000 distance
  units — so fleet reduction is the dominant factor. But pushing too hard for fewer
  vehicles can backfire (solver panics, routes explode). The Fleet Manager learns
  WHEN to push hard, WHEN to back off, and WHEN to try something completely different.

OBSERVATION (what the agent sees):
  132-dim vector = [graph_embedding (128) | solver_stats (4)]

  The 4 solver stats are:
    - time_ratio:       steps_used / 20         ("How much budget is left?")
    - nv_ratio:         NV_current / NV_initial  ("How much fleet reduction so far?")
    - violation_ratio:  excess_load / total_demand ("Am I pushing too hard?")
    - stagnation_ratio: iters_no_improve / budget  ("Am I stuck?")

ACTION SPACE (what the agent can do):
  0 = POLISH             — Default solver params, just optimize routes
  1 = MILD_PRESSURE      — Gentle penalty increase (2x) to nudge fewer vehicles
  2 = MODERATE_PRESSURE  — Steady penalty increase (5x) for reliable fleet reduction
  3 = AGGRESSIVE_PRESSURE — Heavy penalty (10x) to force route merges
  4 = EXPLORE_NEW_SEED   — New random seed, default params (escape local optima)
  5 = EXPLORE_PRESSURE   — New seed + moderate penalty (escape AND reduce)

ARCHITECTURE: Actor-Critic
  The network has two "heads" sharing a common trunk:
    - Actor: outputs a probability distribution over the 6 actions (policy)
    - Critic: outputs a single number estimating "how good is this state?" (value)
  PPO uses both during training — the actor learns better actions, the critic
  learns to evaluate states for computing advantages (see train.py).

Parameters: ~13,100 (intentionally tiny — it's making strategic decisions, not
computing routes)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


# The 4 real-time solver statistics that supplement the graph embedding
SOLVER_STATS_DIM = 4  # [time_ratio, nv_ratio, violation_ratio, stagnation_ratio]

# Total number of discrete strategy choices available to the agent
NUM_FLEET_ACTIONS = 6

# Human-readable names for each action (used in logging and debugging)
ACTION_NAMES = [
    "POLISH", "MILD_PRESSURE", "MODERATE_PRESSURE",
    "AGGRESSIVE_PRESSURE", "EXPLORE_NEW_SEED", "EXPLORE_PRESSURE",
]


class FleetManager(nn.Module):
    """Actor-Critic neural network for solver strategy selection.

    This is the trainable RL agent. It takes the current state (instance
    embedding + solver stats) and outputs:
      1. Action probabilities — which strategy to use next
      2. State value — how good is the current situation (for PPO training)

    The network is deliberately simple (2 hidden layers of 64 units) because
    the decisions are strategic, not computational. The solver does the heavy
    lifting; this network just learns to steer it.

    Args:
        embed_dim: Dimension of graph embedding from GNNEncoder (128).
        stats_dim: Dimension of solver statistics vector (4).
        hidden_dim: Width of hidden layers (64).
        num_actions: Number of discrete strategy choices (6).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        stats_dim: int = SOLVER_STATS_DIM,
        hidden_dim: int = 64,
        num_actions: int = NUM_FLEET_ACTIONS,
    ):
        super().__init__()
        input_dim = embed_dim + stats_dim  # 128 + 4 = 132

        # --- Shared Feature Trunk ---
        # Both the actor and critic share these layers. This is efficient and
        # helps the critic's value estimates stay aligned with the actor's policy.
        # Two layers of 64 units with ReLU activation.
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 132 → 64
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 64 → 64
            nn.ReLU(),
        )

        # --- Actor Head (Policy) ---
        # Outputs raw logits (unnormalized scores) for each of the 6 actions.
        # These logits are converted to probabilities via softmax in Categorical().
        # Higher logit = higher probability of choosing that action.
        self.actor = nn.Linear(hidden_dim, num_actions)  # 64 → 6

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
          When the fleet is already at the theoretical minimum (NV == NV_min),
          it's impossible to reduce further. We set the logits of pressure
          actions (1, 2, 3, 5) to -10000, which makes their probability
          essentially zero after softmax. This prevents the agent from wasting
          steps on impossible fleet reductions.

        Args:
            graph_embedding: (B, 128) instance summary from GNNEncoder.
            solver_stats:    (B, 4) real-time solver feedback.
            action_mask:     (B, 6) bool tensor. True = action allowed, False = blocked.

        Returns:
            action_logits: (B, 6) raw scores for each action (masked ones set to -1e4).
            state_value:   (B, 1) estimated future reward from this state.
        """
        # Concatenate instance understanding + solver feedback into one observation
        obs = torch.cat([graph_embedding, solver_stats], dim=-1)  # (B, 132)

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
            graph_embedding: (B, 128) from GNNEncoder.
            solver_stats:    (B, 4) solver statistics.
            action_mask:     (B, 6) bool tensor. True = allowed, False = masked.

        Returns:
            action:      (B,) sampled action index (0-5).
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
