"""
Stage 1 - Vision System: GNN Encoder for CVRP instances.

This is the "eyes" of our system. Before the RL agent can decide what strategy
to use on a CVRP instance, it needs to UNDERSTAND the instance — where are the
customers? How are they clustered? Where is demand concentrated?

This module takes raw instance data (customer coordinates + demands) and produces
a fixed-size vector (128 dimensions) that captures the spatial structure of the
problem. Think of it like compressing a map into a fingerprint that the RL agent
can read.

Outputs:
  - node_embeddings: (N, 128) — per-node representations (one vector per customer)
  - graph_embedding: (B, 128) — single vector summarizing the entire instance

Architecture: Linear projection -> 3x [GATConv + BatchNorm + ELU + Residual] -> mean pool
Graph construction: k-NN (k=20, Euclidean, undirected)

Why a Graph Neural Network?
  CVRP instances are naturally graphs — customers are nodes, and their spatial
  relationships (distances) define edges. A GNN can learn which spatial patterns
  (tight clusters, outliers, demand hotspots) matter for solver strategy selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    Graph Attention Network (GAT) encoder for CVRP instances.

    WHY GAT specifically?
      Standard GNNs treat all neighbors equally, but in CVRP, some neighbor
      relationships matter more than others. A cluster of high-demand customers
      near capacity limits is more important than spread-out low-demand nodes.
      GAT uses attention mechanisms to learn WHICH neighbors matter most.

    HOW IT WORKS (high level):
      1. Each customer becomes a node with 3 features: [x, y, demand/capacity]
      2. We connect each node to its 20 nearest neighbors (k-NN graph)
      3. Three GAT layers pass messages between neighbors, building up richer
         representations. Each layer has 8 attention heads that learn different
         aspects of the spatial structure.
      4. Mean-pooling aggregates all node vectors into one graph-level vector.

    COMPLEXITY:
      Using k-NN (k=20) keeps attention at O(N*20) instead of O(N^2),
      so even 400-node instances run efficiently on a T4 GPU.

    Parameters: ~51,600 (intentionally small — the heavy lifting is done by PyVRP)
    """

    def __init__(
        self,
        node_features: int = 3,      # Input features per node: [x_norm, y_norm, demand/Q]
        embed_dim: int = 128,         # Output embedding dimension for each node
        num_heads: int = 8,           # Number of attention heads per GAT layer
        num_layers: int = 3,          # Number of stacked GAT layers (depth of message passing)
        dropout: float = 0.1,         # Dropout rate in attention weights (regularization)
        k_neighbors: int = 20,        # Number of nearest neighbors in the k-NN graph
    ):
        super().__init__()

        # Each attention head produces (embed_dim / num_heads) features, then we
        # concatenate all heads. So embed_dim must be divisible by num_heads.
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.k_neighbors = k_neighbors
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads  # = 128/8 = 16 features per head

        # --- Input Projection ---
        # Maps raw 3-dim node features [x, y, demand/Q] up to 128-dim embedding space.
        # This gives the GAT layers a richer representation to work with.
        self.input_proj = nn.Linear(node_features, embed_dim)

        # --- GAT Layers (x3) ---
        # Each GATConv layer performs multi-head attention over the k-NN graph:
        #   - 8 attention heads, each producing 16-dim output
        #   - concat=True: outputs are concatenated → 8*16 = 128 dim (matches embed_dim)
        #   - add_self_loops=True: each node also attends to itself
        #   - dropout: randomly zeroes attention weights during training to prevent
        #     the network from relying too heavily on any single neighbor
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=embed_dim,
                out_channels=head_dim,     # Each head outputs 16 dims
                heads=num_heads,           # 8 heads × 16 = 128 total
                concat=True,              # Concatenate heads (not average)
                dropout=dropout,
                add_self_loops=True,       # Node attends to itself too
            )
            for _ in range(num_layers)
        ])

        # --- Batch Normalization (one per GAT layer) ---
        # Normalizes activations across the batch to stabilize training.
        # Without this, deeper layers can suffer from internal covariate shift,
        # where the distribution of inputs keeps changing as earlier layers update.
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(embed_dim)
            for _ in range(num_layers)
        ])

    def build_graph(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Construct an undirected k-NN graph from customer positions.

        WHY k-NN instead of fully connected?
          A fully connected graph on 400 nodes = 159,600 edges → O(N^2) attention.
          k-NN with k=20 gives only ~8,000 edges → O(N*k) attention. Much faster,
          and CVRP routes naturally connect nearby customers anyway.

        WHY undirected?
          If customer A is a neighbor of B, then B should also be a neighbor of A.
          We add edges in both directions to ensure symmetric message passing.

        Implementation note: Uses pure PyTorch (torch.cdist) instead of the
        torch-cluster C++ extension, so it works without compiling native code.

        Args:
            pos: (N_total, 2) normalized [x, y] coordinates for all nodes.
            batch: (N_total,) integer vector assigning each node to its graph
                   (needed when processing multiple instances in a batch).

        Returns:
            edge_index: (2, E) tensor where edge_index[0] = source nodes,
                        edge_index[1] = destination nodes.
        """
        k = self.k_neighbors  # 20
        device = pos.device
        unique_batches = batch.unique()
        rows, cols = [], []

        # Process each graph in the batch separately (they have different nodes)
        for b in unique_batches:
            mask = batch == b                    # Which nodes belong to this graph
            idx = mask.nonzero(as_tuple=True)[0] # Global indices of these nodes
            pos_b = pos[idx]                     # Positions of just this graph's nodes
            n = pos_b.size(0)
            actual_k = min(k, n - 1)             # Can't have more neighbors than nodes-1
            if actual_k <= 0:
                continue

            # Compute pairwise Euclidean distances between all nodes in this graph
            dist = torch.cdist(pos_b, pos_b)     # (n, n) distance matrix
            dist.fill_diagonal_(float("inf"))    # Don't connect a node to itself via k-NN

            # Find the k nearest neighbors for each node
            _, topk_idx = dist.topk(actual_k, dim=1, largest=False)  # (n, k) indices

            # Convert local indices back to global indices for the edge_index tensor
            src = idx.unsqueeze(1).expand_as(topk_idx).reshape(-1)  # Source nodes
            dst = idx[topk_idx.reshape(-1)]                          # Destination nodes

            # Add edges in BOTH directions (undirected graph)
            rows.append(src)
            cols.append(dst)
            rows.append(dst)   # Reverse direction
            cols.append(src)

        if rows:
            edge_index = torch.stack(
                [torch.cat(rows), torch.cat(cols)], dim=0
            )
            # Remove duplicate edges (can arise from bidirectional addition)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            # Edge case: no valid edges (e.g., single-node graphs)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        return edge_index

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a CVRP instance into embeddings.

        This is the main entry point. Called once per instance at the start
        of each episode — the resulting graph_embedding is then reused for
        all 20 steps of that episode (since the instance doesn't change).

        Args:
            x:     (N_total, 3) node features [x_norm, y_norm, demand/Q].
            pos:   (N_total, 2) node positions [x_norm, y_norm] for k-NN graph.
            batch: (N_total,)   batch vector mapping each node to its graph index.

        Returns:
            node_embeddings:  (N_total, 128) per-node representations.
            graph_embedding:  (B, 128) one vector per graph (instance summary).
        """
        # STEP 1: Build the k-NN graph structure from node positions.
        # We force FP32 here even if the rest of the model runs in FP16,
        # because distance calculations need full precision to get correct neighbors.
        with torch.amp.autocast("cuda", enabled=False):
            edge_index = self.build_graph(pos.float(), batch)

        # STEP 2: Project raw 3-dim features up to 128-dim embedding space.
        x = self.input_proj(x)  # (N, 3) → (N, 128)

        # STEP 3: Pass through 3 GAT layers with residual connections.
        # Each layer:
        #   1. GAT attention: each node aggregates information from its k=20 neighbors,
        #      weighted by learned attention scores. 8 heads capture different
        #      aspects of the spatial relationships.
        #   2. BatchNorm: stabilizes the distribution of activations.
        #   3. ELU activation: non-linearity (like ReLU but smoother for negatives).
        #   4. Residual connection (x + x_residual): adds the input back to the output.
        #      This prevents gradient degradation in deeper networks — gradients can
        #      flow directly through the skip connection during backpropagation.
        for gat, bn in zip(self.gat_layers, self.batch_norms):
            x_residual = x          # Save input for residual connection
            x = gat(x, edge_index)  # Multi-head attention over neighbors
            x = bn(x)               # Normalize activations
            x = F.elu(x)            # Non-linear activation
            x = x + x_residual      # Residual skip connection

        # STEP 4: Produce both outputs.
        node_embeddings = x  # (N, 128) — one vector per customer node

        # Global mean pooling: average all node embeddings to get one vector
        # for the entire instance. This is what the Fleet Manager sees.
        # It captures the "overall character" of the instance (clustered vs spread,
        # high demand vs low demand, etc.)
        graph_embedding = global_mean_pool(x, batch)  # (B, 128)

        return node_embeddings, graph_embedding
