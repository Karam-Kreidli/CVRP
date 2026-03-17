# Stage 1: Graph Neural Network (GNN) Encoder
- **Architecture:** Graph Attention Network (GAT) using PyTorch Geometric or standard PyTorch.
- **Input:** Heterogeneous graph with node features (normalized x, y, and demand/Q).
- **Processing:** Use multiple Message Passing layers to capture spatial clusters.
- **Output:** 1. Node Embeddings: (N, 128) vector per customer.
  2. Graph Embedding: (1, 128) global vector representing the instance topology.
- **Purpose:** Provide a permutation-invariant spatial understanding to the RL agents.