---
name: ML4VRP 2026 Project Overview
description: GECCO 2026 ML4VRP competition entry - MARL+GNN hybrid solver for CVRP with 5-stage architecture
type: project
---

**ML4VRP 2026 — Hierarchical MARL-HGS Hybrid Solver** for GECCO 2026 competition (X-dataset track).

**Objective:** Minimize `1000 × NV + TD` for Capacitated Vehicle Routing Problem (CVRP) on Uchoa X-dataset (100-400 nodes).

**Why:** Competition entry — fleet minimization is primary goal (1000x multiplier on NV).

**How to apply:** All changes should preserve the 5-stage architecture and competition constraints. Performance matters — FP16, sparse graphs, efficient training.

## Five Stages
1. **Vision** (`src/model_vision.py`) — GAT encoder, k-NN(20), 3 layers, outputs (N,128) node + (1,128) graph embeddings
2. **Fleet Manager** (`src/agent_manager.py`) — Strategic agent, 3 actions (polish/eliminate/explore), 132-dim obs, MLP actor-critic
3. **Solver Engine** (`src/solver_engine.py`) — Gymnasium env wrapping PyVRP (ILS), action masking, warm restart
4. **Route Driver** (`src/agent_driver.py`) — Tactical agent, 4 local search operators, attention pooling over node embeddings
5. **Training** (`src/train.py`) — PPO+GAE, MACA credit assignment, FP16, curriculum learning, separate LRs

## Entry Points
- `python -m src.main` — smoke tests
- `python -m src.main train --instance_path data/ --epochs 100 --fp16` — training
- Colab notebooks in `notebooks/`

## Dependencies
PyTorch, PyTorch Geometric, PyVRP, Gymnasium, NumPy
