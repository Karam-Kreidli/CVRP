"""
Stage 1 - Vision System: GNN Encoder for CVRP instances.

Encodes a CVRP instance (node coordinates + demands) into:
  - node_embeddings: (N, 128) per-node representations for Route Driver (Stage 4)
  - graph_embedding: (B, 128) global representation for Fleet Manager (Stage 2)

Architecture: Linear projection -> 3x [GATConv + BatchNorm + ELU + Residual] -> mean pool
Graph construction: k-NN (k=20, Euclidean, undirected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    Graph Attention Network encoder for CVRP instances.

    Produces two outputs consumed by downstream RL agents:
      - node_embeddings (N, embed_dim): per-customer features for Route Driver
      - graph_embedding (B, embed_dim): instance-level features for Fleet Manager

    Uses sparse k-NN graph construction to keep attention O(N*k) instead of O(N^2),
    enabling efficient processing of 400-node instances on a T4 GPU.
    """

    def __init__(
        self,
        node_features: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        k_neighbors: int = 20,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.k_neighbors = k_neighbors
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads

        self.input_proj = nn.Linear(node_features, embed_dim)

        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=embed_dim,
                out_channels=head_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(embed_dim)
            for _ in range(num_layers)
        ])

    def build_graph(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Construct undirected k-NN graph from node positions.

        Uses a pure-PyTorch implementation (torch.cdist) so that the optional
        torch-cluster C++ extension is not required.

        Args:
            pos: (N_total, 2) normalized [x, y] coordinates.
            batch: (N_total,) batch assignment vector.

        Returns:
            edge_index: (2, E) undirected edge index tensor.
        """
        k = self.k_neighbors
        device = pos.device
        unique_batches = batch.unique()
        rows, cols = [], []

        for b in unique_batches:
            mask = batch == b
            idx = mask.nonzero(as_tuple=True)[0]
            pos_b = pos[idx]
            n = pos_b.size(0)
            actual_k = min(k, n - 1)
            if actual_k <= 0:
                continue
            dist = torch.cdist(pos_b, pos_b)
            dist.fill_diagonal_(float("inf"))
            _, topk_idx = dist.topk(actual_k, dim=1, largest=False)
            src = idx.unsqueeze(1).expand_as(topk_idx).reshape(-1)
            dst = idx[topk_idx.reshape(-1)]
            rows.append(src)
            cols.append(dst)
            rows.append(dst)
            cols.append(src)

        if rows:
            edge_index = torch.stack(
                [torch.cat(rows), torch.cat(cols)], dim=0
            )
            # Remove duplicate edges
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        return edge_index

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode CVRP instance graph.

        Args:
            x:     (N_total, 3) node features [x_norm, y_norm, demand/Q].
            pos:   (N_total, 2) node positions [x_norm, y_norm] for k-NN.
            batch: (N_total,)   batch vector mapping nodes to graphs.

        Returns:
            node_embeddings:  (N_total, embed_dim) per-node representations.
            graph_embedding:  (B, embed_dim) per-graph global representation.
        """
        # k-NN always in FP32 for distance precision
        with torch.amp.autocast("cuda", enabled=False):
            edge_index = self.build_graph(pos.float(), batch)

        x = self.input_proj(x)

        for gat, bn in zip(self.gat_layers, self.batch_norms):
            x_residual = x
            x = gat(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = x + x_residual

        node_embeddings = x
        graph_embedding = global_mean_pool(x, batch)

        return node_embeddings, graph_embedding
