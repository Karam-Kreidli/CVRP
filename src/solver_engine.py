"""
Stage 3 - HGS Engine Wrapper: Gymnasium environment wrapping PyVRP's ILS solver.

Maps the Fleet Manager's discrete actions to PyVRP solver parameter regimes:
  0 = POLISH              — Default ILS, same seed (route optimization)
  1 = MILD_PRESSURE       — penalty_increase=2.0  (gentle fleet reduction)
  2 = MODERATE_PRESSURE   — penalty_increase=5.0  (steady fleet reduction)
  3 = AGGRESSIVE_PRESSURE — penalty_increase=10.0 (force route merges)
  4 = EXPLORE_NEW_SEED    — Fresh random seed, default params (escape local optima)
  5 = EXPLORE_PRESSURE    — Fresh seed + moderate penalty (escape + reduce)

Each step() runs the solver for a fixed iteration budget, then returns the
updated observation (132-dim), reward (change in competition score), and
termination status.

Competition score: 1000 * NV + TD  (lower is better)
"""

from __future__ import annotations

import math
import pathlib
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import pyvrp
from pyvrp import (
    Model,
    PenaltyParams,
    ProblemData,
    Solution,
    SolveParams,
    read,
)
from pyvrp.stop import MaxIterations

from src.model_vision import GNNEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM = 128
STATS_DIM = 4
OBS_DIM = EMBED_DIM + STATS_DIM  # 132
NUM_ACTIONS = 6
ITERS_PER_STEP = 1_000
MAX_STEPS = 20  # episode length: 20 steps × 1000 iters = 20K total iterations

ACTION_NAMES = [
    "POLISH", "MILD_PRESSURE", "MODERATE_PRESSURE",
    "AGGRESSIVE_PRESSURE", "EXPLORE_NEW_SEED", "EXPLORE_PRESSURE",
]


def competition_score(nv: int, td: float) -> float:
    """Compute the GECCO 2026 ML4VRP objective: 1000 * NV + TD."""
    return 1000.0 * nv + td


class CVRPEnv(gym.Env):
    """Gymnasium environment wrapping PyVRP for RL-guided CVRP solving.

    The agent (Fleet Manager) observes a 132-dim vector and selects one of
    six search strategies per step. Each step runs PyVRP's ILS for a fixed
    iteration budget under the chosen parameter regime.

    Args:
        instance_paths: List of paths to .vrp files (X-dataset format).
        encoder: Pre-trained GNNEncoder for computing graph embeddings.
        device: Torch device for encoder inference.
        iters_per_step: PyVRP iterations per environment step.
        max_steps: Maximum steps per episode before truncation.
        round_func: Rounding function for reading instances ("round" for X-dataset).
        max_nodes: Curriculum filter — only use instances with N ≤ max_nodes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_paths: list[str | pathlib.Path],
        encoder: GNNEncoder,
        device: torch.device = torch.device("cpu"),
        iters_per_step: int = ITERS_PER_STEP,
        max_steps: int = MAX_STEPS,
        round_func: str = "round",
        max_nodes: int | None = None,
    ):
        super().__init__()
        self._all_instance_paths = [pathlib.Path(p) for p in instance_paths]
        self.max_nodes = max_nodes
        self.instance_paths = self._filter_by_nodes(self._all_instance_paths)
        self.encoder = encoder
        self.device = device
        self.iters_per_step = iters_per_step
        self.max_steps = max_steps
        self.round_func = round_func

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # Episode state (initialized in reset)
        self._data: ProblemData | None = None
        self._best_solution: Solution | None = None
        self._graph_embedding: torch.Tensor | None = None
        self._node_embeddings: torch.Tensor | None = None
        self._nv_initial: int = 0
        self._nv_min: int = 1  # theoretical minimum feasible fleet
        self._prev_score: float = 0.0
        self._step_count: int = 0
        self._total_iters: int = 0
        self._iters_since_improvement: int = 0
        self._seed: int = 0

    def _compute_nv_min(self) -> int:
        """Compute theoretical minimum fleet: ceil(sum_demands / Q)."""
        data = self._data
        total_demand = sum(c.delivery[0] for c in data.clients())
        capacity = data.vehicle_type(0).capacity[0]
        return max(1, math.ceil(total_demand / capacity))

    def get_action_mask(self) -> np.ndarray:
        """Return a boolean mask over 6 actions. True = allowed, False = masked.

        Masks all pressure actions (1, 2, 3, 5) when current NV <= NV_min,
        since we can't reduce the fleet below the theoretical minimum.
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        if self._best_solution is not None:
            current_nv = self._best_solution.num_routes()
            if current_nv <= self._nv_min:
                mask[1] = False  # MILD_PRESSURE
                mask[2] = False  # MODERATE_PRESSURE
                mask[3] = False  # AGGRESSIVE_PRESSURE
                mask[5] = False  # EXPLORE_PRESSURE
        return mask

    def _filter_by_nodes(self, paths: list[pathlib.Path]) -> list[pathlib.Path]:
        """Filter instance paths by max_nodes (parsed from filename X-nNNN-kKK)."""
        if self.max_nodes is None:
            return list(paths)
        filtered = []
        for p in paths:
            try:
                n = int(p.stem.split("-")[1][1:])
                if n <= self.max_nodes:
                    filtered.append(p)
            except (IndexError, ValueError):
                filtered.append(p)  # keep non-standard names
        return filtered if filtered else list(paths)

    def set_max_nodes(self, max_nodes: int | None):
        """Update the curriculum: change max instance size and re-filter."""
        self.max_nodes = max_nodes
        self.instance_paths = self._filter_by_nodes(self._all_instance_paths)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Load a random instance, encode it, run an initial solve, return obs."""
        super().reset(seed=seed)

        # Pick random instance
        path = random.choice(self.instance_paths)
        self._data = read(str(path), round_func=self.round_func)
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._nv_min = self._compute_nv_min()

        # Encode instance with GNN (Stage 1)
        self._graph_embedding, self._node_embeddings = self._encode_instance()

        # Initial solve to get a starting solution
        res = pyvrp.solve(
            self._data,
            stop=MaxIterations(self.iters_per_step),
            seed=self._seed,
            display=False,
            collect_stats=False,
        )
        self._best_solution = res.best
        self._total_iters = res.num_iterations

        nv = self._best_solution.num_routes()
        td = self._best_solution.distance()
        self._nv_initial = nv
        self._prev_score = competition_score(nv, td)
        self._step_count = 0
        self._iters_since_improvement = 0

        obs = self._build_observation()
        info = {
            "instance": path.stem,
            "nv": nv,
            "td": td,
            "score": self._prev_score,
            "nv_min": self._nv_min,
            "action_mask": self.get_action_mask(),
        }
        return obs, info

    # Penalty returned when solver cannot satisfy hard fleet limit
    FAILURE_PENALTY = -5.0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one search strategy step.

        Args:
            action: 0-5, one of the six strategy actions.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self._data is not None, "Must call reset() before step()"
        self._step_count += 1

        # Map action to solver parameters and run
        params, step_seed = self._action_to_params(action)

        res = pyvrp.solve(
            self._data,
            stop=MaxIterations(self.iters_per_step),
            seed=step_seed,
            display=False,
            collect_stats=False,
            params=params,
            initial_solution=self._best_solution,
        )
        self._total_iters += res.num_iterations

        # Update best solution if improved
        candidate = res.best
        cand_nv = candidate.num_routes()
        cand_td = candidate.distance()
        cand_score = competition_score(cand_nv, cand_td)

        # Hard fleet limit: reject solutions that explode the fleet
        # If a pressure action was chosen and the solver panicked (NV spike),
        # apply a static failure penalty instead of using the bad solution.
        prev_nv = self._best_solution.num_routes()
        is_pressure = action in (1, 2, 3, 5)
        fleet_exploded = (is_pressure and cand_nv > prev_nv + 2)

        if fleet_exploded:
            # Solver panicked — don't update solution, return failure penalty
            self._iters_since_improvement += res.num_iterations
            reward = self.FAILURE_PENALTY
        elif candidate.is_feasible() and cand_score < self._prev_score:
            self._best_solution = candidate
            self._iters_since_improvement = 0
        else:
            self._iters_since_improvement += res.num_iterations

        # Compute reward (unless failure penalty already applied)
        new_nv = self._best_solution.num_routes()
        new_td = self._best_solution.distance()
        new_score = competition_score(new_nv, new_td)
        if not fleet_exploded:
            reward = self._prev_score - new_score  # positive if score decreased
        self._prev_score = new_score

        # Termination conditions
        terminated = False
        truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = {
            "action_name": ACTION_NAMES[action],
            "nv": new_nv,
            "td": new_td,
            "score": new_score,
            "nv_min": self._nv_min,
            "action_mask": self.get_action_mask(),
            "fleet_exploded": fleet_exploded,
            "step": self._step_count,
            "total_iters": self._total_iters,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GNNEncoder on the current instance, return (graph_emb, node_emb)."""
        data = self._data
        num_locs = data.num_locations  # depot + clients

        # Build node feature matrix: [x_norm, y_norm, demand/Q]
        coords = []
        demands = []
        for i in range(num_locs):
            loc = data.location(i)
            coords.append([loc.x, loc.y])
            if i == 0:
                demands.append(0.0)
            else:
                demands.append(float(data.clients()[i - 1].delivery[0]))

        coords = np.array(coords, dtype=np.float32)
        demands = np.array(demands, dtype=np.float32)

        # Normalize coordinates to [0, 1]
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        c_range = c_max - c_min
        c_range[c_range == 0] = 1.0  # avoid division by zero
        coords_norm = (coords - c_min) / c_range

        # Normalize demand by vehicle capacity
        capacity = float(data.vehicle_type(0).capacity[0])
        demand_norm = demands / capacity if capacity > 0 else demands

        # Assemble feature tensor [x_norm, y_norm, demand/Q]
        x = np.column_stack([coords_norm, demand_norm])
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        pos_t = torch.tensor(coords_norm, dtype=torch.float32, device=self.device)
        batch_t = torch.zeros(num_locs, dtype=torch.long, device=self.device)

        self.encoder.eval()
        with torch.no_grad():
            node_emb, graph_emb = self.encoder(x_t, pos_t, batch_t)

        return graph_emb, node_emb  # (1, 128), (N, 128)

    def _build_observation(self) -> np.ndarray:
        """Construct the 132-dim observation vector."""
        # Time ratio: steps used / max steps
        time_ratio = self._step_count / self.max_steps

        # NV ratio: current / initial
        nv_current = self._best_solution.num_routes()
        nv_ratio = nv_current / max(self._nv_initial, 1)

        # Violation ratio: excess load / total demand
        excess_loads = self._best_solution.excess_load()
        excess = float(excess_loads[0]) if isinstance(excess_loads, (list, tuple)) else float(excess_loads)
        total_demand = sum(
            c.delivery[0] for c in self._data.clients()
        )
        violation_ratio = excess / max(total_demand, 1.0)

        # Stagnation ratio: iters since improvement / total budget
        total_budget = self.max_steps * self.iters_per_step
        stagnation_ratio = self._iters_since_improvement / max(total_budget, 1)

        stats = torch.tensor(
            [[time_ratio, nv_ratio, violation_ratio, stagnation_ratio]],
            dtype=torch.float32,
            device=self.device,
        )

        # Concatenate graph embedding (1, 128) with stats (1, 4) → (1, 132)
        obs = torch.cat([self._graph_embedding, stats], dim=-1)
        return obs.squeeze(0).cpu().numpy()

    def _action_to_params(self, action: int) -> tuple[SolveParams, int]:
        """Map a discrete action to PyVRP SolveParams and seed.

        Returns:
            (params, seed) tuple for the pyvrp.solve() call.
        """
        if action == 0:
            # POLISH: default parameters, same seed for continuity
            return SolveParams(), self._seed

        elif action == 1:
            # MILD_PRESSURE: gentle penalty to nudge toward fewer vehicles
            penalty = PenaltyParams(
                penalty_increase=2.0,
                penalty_decrease=0.5,
                target_feasible=0.5,
                min_penalty=5.0,
                max_penalty=50_000.0,
            )
            return SolveParams(penalty=penalty), self._seed

        elif action == 2:
            # MODERATE_PRESSURE: steady penalty for reliable fleet reduction
            penalty = PenaltyParams(
                penalty_increase=5.0,
                penalty_decrease=0.5,
                target_feasible=0.3,
                min_penalty=10.0,
                max_penalty=100_000.0,
            )
            return SolveParams(penalty=penalty), self._seed

        elif action == 3:
            # AGGRESSIVE_PRESSURE: heavy penalty, force route merges
            penalty = PenaltyParams(
                penalty_increase=10.0,
                penalty_decrease=0.3,
                target_feasible=0.2,
                min_penalty=20.0,
                max_penalty=200_000.0,
            )
            return SolveParams(penalty=penalty), self._seed

        elif action == 4:
            # EXPLORE_NEW_SEED: fresh restart with default params
            self._seed = random.randint(0, 2**31)
            return SolveParams(), self._seed

        elif action == 5:
            # EXPLORE_PRESSURE: fresh seed + moderate penalty
            self._seed = random.randint(0, 2**31)
            penalty = PenaltyParams(
                penalty_increase=5.0,
                penalty_decrease=0.5,
                target_feasible=0.3,
                min_penalty=10.0,
                max_penalty=100_000.0,
            )
            return SolveParams(penalty=penalty), self._seed

        else:
            raise ValueError(f"Invalid action: {action}. Expected 0-5.")
