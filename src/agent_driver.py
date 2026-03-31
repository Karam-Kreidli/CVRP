"""
Stage 4 - Route Driver: Tactical Actor-Critic agent for local search operator selection.

*** THIS MODULE IS NOT USED IN THE CURRENT IMPLEMENTATION. ***

It was originally designed to select which local search operators PyVRP should
apply at each step (TWO_OPT, SWAP, RELOCATE, SWAP_STAR). However, PyVRP's
solve() API runs its own internal operator selection and does not expose hooks
for injecting external operator choices mid-solve. The Route Driver would require
modifying PyVRP's C++ source code, which is outside the competition scope.

The current system uses only the Fleet Manager (agent_manager.py) to control
solver PARAMETERS (penalty levels, random seeds), not individual operators.

This file is kept for reference in case future PyVRP versions expose operator
selection APIs, or if we move to a custom solver implementation.

Original design:
  Consumes the (N, 128) node embeddings from Stage 1 (GNNEncoder) and produces
  a probability distribution over 4 local search operators.

  Architecture:
    1. Multi-Head Attention pooling: (N, 128) → (1, 128) tactical context vector
    2. Actor-Critic MLP: (1, 128) → operator logits (4) + state value (1)

  Action space (discrete, 4):
    0 = TWO_OPT      — Reverse a sub-sequence within a route
    1 = SWAP          — Exchange two customers between different routes
    2 = RELOCATE      — Move a customer from one route to another
    3 = SWAP_STAR     — Advanced cross-route exchange (most powerful operator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

NUM_OPERATORS = 4
OPERATOR_NAMES = ["TWO_OPT", "SWAP", "RELOCATE", "SWAP_STAR"]


class AttentionPooling(nn.Module):
    """Multi-Head Attention pooling over variable-length node embeddings.

    Computes a single tactical context vector from N node embeddings by
    learning which nodes are most informative for operator selection.

    Mathematically:
      score_i = (W_q · mean(H)) · (W_k · h_i)^T / sqrt(d_k)    for each head
      alpha_i = softmax(score_i)
      context = sum_i(alpha_i · W_v · h_i)                       per head
      output  = concat(context_1, ..., context_H) · W_o

    Where H is the node embedding matrix (N, embed_dim).

    Args:
        embed_dim: Input embedding dimension (default: 128).
        num_heads: Number of attention heads (default: 4).
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query is derived from the global mean (instance-level summary)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        # Key and value are per-node projections
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        # Output projection after concatenating heads
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool N node embeddings into a single context vector.

        Args:
            node_embeddings: (N, embed_dim) node-level representations.

        Returns:
            context: (1, embed_dim) tactical context vector.
        """
        N, D = node_embeddings.shape
        H = self.num_heads
        d_k = self.head_dim

        # Query: global mean as the "question" about instance structure
        global_mean = node_embeddings.mean(dim=0, keepdim=True)  # (1, D)
        q = self.W_q(global_mean).view(1, H, d_k).transpose(0, 1)  # (H, 1, d_k)

        # Keys and values: per-node projections
        k = self.W_k(node_embeddings).view(N, H, d_k).permute(1, 0, 2)  # (H, N, d_k)
        v = self.W_v(node_embeddings).view(N, H, d_k).permute(1, 0, 2)  # (H, N, d_k)

        # Scaled dot-product attention: (H, 1, d_k) × (H, d_k, N) → (H, 1, N)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (H, 1, N)

        # Weighted sum of values: (H, 1, N) × (H, N, d_k) → (H, 1, d_k)
        context = torch.bmm(attn_weights, v)  # (H, 1, d_k)

        # Concatenate heads and project
        context = context.transpose(0, 1).contiguous().view(1, D)  # (1, D)
        context = self.W_o(context)  # (1, D)

        return context


class RouteDriver(nn.Module):
    """Actor-Critic network for local search operator selection.

    Uses attention-based pooling to summarize node embeddings into a tactical
    context, then produces operator logits and a state value estimate.

    Args:
        embed_dim: Dimension of node embeddings from GNNEncoder (default: 128).
        num_heads: Attention heads for pooling (default: 4).
        hidden_dim: MLP hidden layer width (default: 64).
        num_operators: Number of local search operators (default: 4).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 64,
        num_operators: int = NUM_OPERATORS,
    ):
        super().__init__()

        # Attention pooling: (N, 128) → (1, 128)
        self.attention_pool = AttentionPooling(embed_dim, num_heads)

        # Shared feature trunk
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head — policy logits over operators
        self.actor = nn.Linear(hidden_dim, num_operators)

        # Critic head — scalar state value V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self, node_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing operator logits and state value.

        Args:
            node_embeddings: (N, 128) from GNNEncoder.

        Returns:
            operator_logits: (1, 4) raw logits over local search operators.
            state_value:     (1, 1) estimated state value V(s).
        """
        context = self.attention_pool(node_embeddings)  # (1, 128)
        features = self.shared(context)                 # (1, 64)
        operator_logits = self.actor(features)           # (1, 4)
        state_value = self.critic(features)              # (1, 1)
        return operator_logits, state_value

    def select_operator(
        self, node_embeddings: torch.Tensor
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an operator from the policy.

        Returns:
            operator: int index of selected operator.
            log_prob: (1,) log-probability of selected operator.
            state_value: (1, 1) critic estimate.
        """
        operator_logits, state_value = self.forward(node_embeddings)
        dist = Categorical(logits=operator_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, state_value
