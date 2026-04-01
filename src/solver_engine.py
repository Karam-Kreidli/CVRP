"""
Stage 3 - Solver Engine: Gymnasium environment wrapping the HGS-CVRP solver.

This is the BRIDGE between the RL agent and the actual CVRP solver. It translates
the Fleet Manager's high-level strategy decisions into concrete HGS algorithm
parameters (population sizes, search granularity, feasibility targets, etc.).

HOW IT WORKS:
  The environment follows the standard Gymnasium (OpenAI Gym) interface:
    1. reset()  — Load a random CVRP instance, encode it with the GNN, do an initial solve
    2. step(action) — Apply the chosen parameter configuration for a fresh HGS solve
    3. Repeat step() MAX_STEPS times → episode ends

WHAT THE ACTIONS ACTUALLY DO:
  The Fleet Manager controls HGS's ALGORITHM PARAMETERS, which determine the
  genetic algorithm's population dynamics, search depth, and feasibility pressure:

  0 = DEFAULT            — HGS defaults (mu=25, lambda=40, nbGranular=20, targetFeasible=0.2)
  1 = FAST_AGGRESSIVE    — Small pop, low granularity, low feasibility target (speed + pressure)
  2 = LARGE_DIVERSE      — Big population, high granularity, relaxed feasibility (thorough search)
  3 = DEEP_SEARCH        — Default pop, very high granularity (deep local search neighborhoods)
  4 = HIGH_TURNOVER      — Tiny base pop, huge offspring, very low feasibility (maximum churn)
  5 = STABLE_ELITE       — Large base pop, fewer offspring, high feasibility (conservative)
  6 = EXPLORE_NEW_SEED   — Default params with a fresh random seed (escape local optima)

REWARD:
  reward = previous_score - new_score  (positive when the score IMPROVES)
  Since score = 1000*NV + TD, removing a vehicle gives reward ≈ +1000.

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

import hygese as hgs

from src.model_vision import GNNEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM = 128          # Graph embedding dimension (from GNN Encoder)
STATS_DIM = 4            # Solver statistics: [time_ratio, nv_ratio, score_ratio, stagnation]
OBS_DIM = EMBED_DIM + STATS_DIM  # 132 = total observation dimension for the agent
NUM_ACTIONS = 7          # Number of discrete strategy choices
ITERS_PER_STEP = 5_000   # HGS iterations (nbIter) per environment step
MAX_STEPS = 20           # Steps per episode
TIME_PER_STEP = 0.0      # HGS time limit per step (0 = use nbIter only)

ACTION_NAMES = [
    "DEFAULT", "FAST_AGGRESSIVE", "LARGE_DIVERSE",
    "DEEP_SEARCH", "HIGH_TURNOVER", "STABLE_ELITE", "EXPLORE_NEW_SEED",
]


def competition_score(nv: int, td: float) -> float:
    """Compute the GECCO 2026 ML4VRP objective: 1000 * NV + TD.

    The 1000x multiplier on NV means that fleet size dominates:
    removing just one vehicle saves 1000 distance units worth of score.
    """
    return 1000.0 * nv + td


def _parse_vrp_file(path: pathlib.Path) -> dict:
    """Parse a .vrp file (X-dataset TSPLIB format) into HGS data dict.

    HGS expects a dictionary with:
      - x_coordinates, y_coordinates: node positions (depot at index 0)
      - demands: demand per node (0 for depot)
      - vehicle_capacity: capacity per vehicle
      - num_vehicles: upper bound on fleet size
      - depot: index of the depot node
      - service_times: service time per node (0 for CVRP)

    Returns:
        dict ready for hgs_solver.solve_cvrp()
    """
    lines = path.read_text().splitlines()
    section = None
    dimension = 0
    capacity = 0
    coords = {}
    demands = {}
    depot_id = 1

    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue

        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
        elif line == "NODE_COORD_SECTION":
            section = "coord"
            continue
        elif line == "DEMAND_SECTION":
            section = "demand"
            continue
        elif line == "DEPOT_SECTION":
            section = "depot"
            continue
        elif line.startswith("NAME") or line.startswith("COMMENT") or \
             line.startswith("TYPE") or line.startswith("EDGE_WEIGHT_TYPE"):
            continue

        if section == "coord":
            parts = line.split()
            node_id = int(parts[0])
            coords[node_id] = (int(parts[1]), int(parts[2]))
        elif section == "demand":
            parts = line.split()
            node_id = int(parts[0])
            demands[node_id] = int(parts[1])
        elif section == "depot":
            val = int(line)
            if val >= 0:
                depot_id = val

    # Build arrays ordered by node ID (1-indexed in .vrp files)
    n = dimension
    x_coords = np.zeros(n, dtype=np.float64)
    y_coords = np.zeros(n, dtype=np.float64)
    demand_arr = np.zeros(n, dtype=np.float64)

    for i, node_id in enumerate(sorted(coords.keys())):
        x_coords[i] = coords[node_id][0]
        y_coords[i] = coords[node_id][1]
        demand_arr[i] = demands.get(node_id, 0)

    # Depot is index 0 (HGS convention); .vrp depot is usually node 1
    # The sorted order already puts node 1 at index 0

    # Upper bound on vehicles: ceil(total_demand / capacity)
    # Use a generous upper bound so HGS can explore freely
    total_demand = demand_arr.sum()
    nv_upper = max(1, int(math.ceil(total_demand / capacity))) * 2

    data = {
        "x_coordinates": x_coords,
        "y_coordinates": y_coords,
        "demands": demand_arr,
        "vehicle_capacity": capacity,
        "num_vehicles": nv_upper,
        "depot": 0,
        "service_times": np.zeros(n, dtype=np.float64),
    }
    return data


class CVRPEnv(gym.Env):
    """Gymnasium environment that wraps HGS-CVRP for RL-guided solving.

    This class is the interface between the RL world and the solver world.
    It follows the standard Gymnasium API so that any RL algorithm (PPO, A2C,
    DQN, etc.) can interact with it via reset() and step().

    EPISODE LIFECYCLE:
      1. reset() picks a random .vrp instance, encodes it with the GNN,
         runs an initial solve with default HGS parameters.
      2. step(action) maps the chosen action to HGS AlgorithmParameters,
         runs a fresh solve, and keeps the best solution found across all steps.
      3. After MAX_STEPS steps, the episode ends (truncated=True).

    NOTE ON WARM STARTING:
      Unlike PyVRP, HGS does not support warm starting (passing a previous
      solution). Each step runs a fresh solve. The environment tracks the best
      solution found across ALL steps in the episode. The agent's job is to
      find the parameter configuration that produces the best result for each
      instance.

    Args:
        instance_paths: List of paths to .vrp files (X-dataset format).
        encoder: GNNEncoder for computing graph embeddings (used at reset).
        device: Torch device for encoder inference.
        iters_per_step: HGS iterations per step (nbIter parameter).
        max_steps: Steps per episode.
        max_nodes: Curriculum filter — only use instances with <= N customers.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_paths: list[str | pathlib.Path],
        encoder: GNNEncoder,
        device: torch.device = torch.device("cpu"),
        iters_per_step: int = ITERS_PER_STEP,
        max_steps: int = MAX_STEPS,
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

        # Gymnasium space definitions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # --- Episode State ---
        self._hgs_data: dict | None = None          # Parsed instance for HGS
        self._vrp_path: pathlib.Path | None = None   # Current instance path
        self._best_nv: int = 0                       # Best NV found this episode
        self._best_td: float = 0.0                   # Best TD found this episode
        self._best_score: float = float("inf")       # Best score found this episode
        self._graph_embedding: torch.Tensor | None = None
        self._node_embeddings: torch.Tensor | None = None
        self._nv_initial: int = 0                    # Fleet size after initial solve
        self._nv_min: int = 1                        # Theoretical minimum fleet
        self._prev_score: float = 0.0                # Score from previous step
        self._step_count: int = 0
        self._iters_since_improvement: int = 0       # Stagnation counter
        self._seed: int = 0
        self._capacity: int = 1                      # Vehicle capacity (for nv_min)

    def _compute_nv_min(self) -> int:
        """Compute theoretical minimum fleet size: ceil(total_demand / capacity)."""
        total_demand = self._hgs_data["demands"].sum()
        capacity = self._hgs_data["vehicle_capacity"]
        return max(1, math.ceil(total_demand / capacity))

    def get_action_mask(self) -> np.ndarray:
        """Return a boolean mask over 7 actions. True = allowed, False = blocked.

        When the fleet is already at the theoretical minimum (NV == NV_min),
        block the high-pressure actions that try to reduce fleet further.
        Actions 1 (FAST_AGGRESSIVE) and 4 (HIGH_TURNOVER) are the most
        aggressive; block those when at minimum fleet.
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        if self._best_nv <= self._nv_min:
            mask[1] = False  # FAST_AGGRESSIVE
            mask[4] = False  # HIGH_TURNOVER
        return mask

    def _filter_by_nodes(self, paths: list[pathlib.Path]) -> list[pathlib.Path]:
        """Filter instance paths by customer count for curriculum learning."""
        if self.max_nodes is None:
            return list(paths)
        filtered = []
        for p in paths:
            try:
                n = int(p.stem.split("-")[1][1:])
                if n <= self.max_nodes:
                    filtered.append(p)
            except (IndexError, ValueError):
                filtered.append(p)
        return filtered if filtered else list(paths)

    def set_max_nodes(self, max_nodes: int | None):
        """Update the curriculum: change max instance size and re-filter."""
        self.max_nodes = max_nodes
        self.instance_paths = self._filter_by_nodes(self._all_instance_paths)

    # ------------------------------------------------------------------
    # HGS Solver
    # ------------------------------------------------------------------

    def _solve_hgs(self, params: hgs.AlgorithmParameters) -> dict:
        """Run HGS-CVRP with the given parameters and return result dict.

        Returns:
            dict with 'nv' (number of vehicles), 'td' (total distance),
            'score' (competition score), 'routes' (list of routes).
        """
        solver = hgs.Solver(parameters=params, verbose=False)
        result = solver.solve_cvrp(self._hgs_data, rounding=True)

        nv = len(result.routes)
        td = result.cost
        return {
            "nv": nv,
            "td": td,
            "score": competition_score(nv, td),
            "routes": result.routes,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode: load instance, encode it, do initial solve.

        Returns:
            obs: (132,) numpy array — the agent's first observation
            info: dict with instance name, initial NV/TD/score, action mask
        """
        super().reset(seed=seed)

        # Pick a random instance
        path = random.choice(self.instance_paths)
        self._vrp_path = path
        self._hgs_data = _parse_vrp_file(path)
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._nv_min = self._compute_nv_min()

        # Encode instance with GNN (done ONCE per episode)
        self._graph_embedding, self._node_embeddings = self._encode_instance()

        # Initial solve with default HGS parameters
        default_params = hgs.AlgorithmParameters(
            timeLimit=TIME_PER_STEP,
            nbIter=self.iters_per_step,
            seed=self._seed,
        )
        result = self._solve_hgs(default_params)

        self._best_nv = result["nv"]
        self._best_td = result["td"]
        self._best_score = result["score"]
        self._nv_initial = result["nv"]
        self._prev_score = result["score"]
        self._step_count = 0
        self._iters_since_improvement = 0

        obs = self._build_observation()
        info = {
            "instance": path.stem,
            "nv": result["nv"],
            "td": result["td"],
            "score": result["score"],
            "nv_min": self._nv_min,
            "action_mask": self.get_action_mask(),
        }
        return obs, info

    FAILURE_PENALTY = -5.0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one HGS solve with the chosen parameter configuration.

        Each step runs a FRESH HGS solve (no warm starting). The environment
        tracks the best solution found across all steps. The reward is based
        on whether this step's result improved the episode-best score.

        Args:
            action: 0-6, one of the seven strategy actions.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self._hgs_data is not None, "Must call reset() before step()"
        self._step_count += 1

        # Translate action to HGS parameters
        params = self._action_to_params(action)

        # Run fresh HGS solve
        result = self._solve_hgs(params)

        cand_nv = result["nv"]
        cand_td = result["td"]
        cand_score = result["score"]

        # Check for fleet explosion on aggressive actions
        is_aggressive = action in (1, 4)
        fleet_exploded = (is_aggressive and cand_nv > self._best_nv + 2)

        if fleet_exploded:
            self._iters_since_improvement += self.iters_per_step
            reward = self.FAILURE_PENALTY
        elif cand_score < self._best_score:
            # New best solution found!
            self._best_nv = cand_nv
            self._best_td = cand_td
            self._best_score = cand_score
            self._iters_since_improvement = 0
            reward = self._prev_score - cand_score
        else:
            self._iters_since_improvement += self.iters_per_step
            reward = self._prev_score - self._best_score  # 0 if no change

        self._prev_score = self._best_score

        terminated = False
        truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = {
            "action_name": ACTION_NAMES[action],
            "nv": self._best_nv,
            "td": self._best_td,
            "score": self._best_score,
            "nv_min": self._nv_min,
            "action_mask": self.get_action_mask(),
            "fleet_exploded": fleet_exploded,
            "step": self._step_count,
            "cand_score": cand_score,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GNN Encoder on the current instance to produce embeddings.

        Each node gets 3 features: [x_norm, y_norm, demand/capacity].
        """
        data = self._hgs_data
        x_coords = data["x_coordinates"]
        y_coords = data["y_coordinates"]
        demands = data["demands"]
        capacity = float(data["vehicle_capacity"])
        num_locs = len(x_coords)

        coords = np.column_stack([x_coords, y_coords]).astype(np.float32)

        # Normalize coordinates to [0, 1]
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        c_range = c_max - c_min
        c_range[c_range == 0] = 1.0
        coords_norm = (coords - c_min) / c_range

        # Normalize demand by capacity
        demand_norm = (demands / capacity if capacity > 0 else demands).astype(np.float32)

        # Assemble 3-feature input: [x_norm, y_norm, demand/Q]
        x = np.column_stack([coords_norm, demand_norm])
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        pos_t = torch.tensor(coords_norm, dtype=torch.float32, device=self.device)
        batch_t = torch.zeros(num_locs, dtype=torch.long, device=self.device)

        self.encoder.eval()
        with torch.no_grad():
            node_emb, graph_emb = self.encoder(x_t, pos_t, batch_t)

        return graph_emb, node_emb

    def _build_observation(self) -> np.ndarray:
        """Construct the 132-dim observation vector.

        graph_embedding (128) + solver_stats (4):
          - time_ratio: step progress through episode
          - nv_ratio: current best NV / initial NV
          - score_ratio: current best score / initial score (< 1 means improvement)
          - stagnation_ratio: steps without improvement / total budget
        """
        time_ratio = self._step_count / self.max_steps

        nv_ratio = self._best_nv / max(self._nv_initial, 1)

        initial_score = competition_score(self._nv_initial, self._best_td)
        score_ratio = self._best_score / max(initial_score, 1.0)

        total_budget = self.max_steps * self.iters_per_step
        stagnation_ratio = self._iters_since_improvement / max(total_budget, 1)

        stats = torch.tensor(
            [[time_ratio, nv_ratio, score_ratio, stagnation_ratio]],
            dtype=torch.float32,
            device=self.device,
        )

        obs = torch.cat([self._graph_embedding, stats], dim=-1)
        return obs.squeeze(0).cpu().numpy()

    def _action_to_params(self, action: int) -> hgs.AlgorithmParameters:
        """Map the agent's discrete action to HGS AlgorithmParameters.

        Each action configures a different genetic algorithm strategy:

        mu (min pop size): Smaller = faster generations, less diversity.
                           Larger = more diverse population, slower convergence.

        lambda_ (offspring size): Controls how many new solutions are generated
                                  before survivors are selected. Higher = more
                                  exploration per generation.

        nbGranular: Size of the local search neighborhood. Higher = more thorough
                    but slower local search. Key quality-vs-speed tradeoff.

        targetFeasible: Fraction of population that should be feasible. Lower =
                        more infeasible solutions allowed = explores harder but
                        riskier. Higher = conservative, more feasible solutions.

        nbElite: Number of elite (best) solutions protected from replacement.
                 More elites = more exploitation, fewer = more exploration.

        nbClose: Number of neighbors used for diversity calculation.
                 Affects the balance between cost and diversity in fitness.
        """
        if action == 0:
            # DEFAULT: Standard HGS parameters — the baseline configuration.
            return hgs.AlgorithmParameters(
                timeLimit=TIME_PER_STEP,
                nbIter=self.iters_per_step,
                seed=self._seed,
            )

        elif action == 1:
            # FAST_AGGRESSIVE: Small population, low granularity, low feasibility.
            # Fastest generations, maximum pressure to reduce vehicles.
            # Trades solution quality for speed and fleet reduction.
            return hgs.AlgorithmParameters(
                timeLimit=TIME_PER_STEP,
                nbIter=self.iters_per_step,
                seed=self._seed,
                mu=15,
                lambda_=20,
                nbGranular=15,
                targetFeasible=0.1,
                nbElite=2,
                nbClose=3,
            )

        elif action == 2:
            # LARGE_DIVERSE: Big population, high granularity, relaxed feasibility.
            # Thorough search with diverse solution pool — best for hard instances.
            return hgs.AlgorithmParameters(
                timeLimit=TIME_PER_STEP,
                nbIter=self.iters_per_step,
                seed=self._seed,
                mu=40,
                lambda_=60,
                nbGranular=30,
                targetFeasible=0.3,
                nbElite=6,
                nbClose=8,
            )

        elif action == 3:
            # DEEP_SEARCH: Default population, very high granularity.
            # Maximum local search depth — good for instances where route
            # structure is nearly optimal but distances can be polished.
            return hgs.AlgorithmParameters(
                timeLimit=TIME_PER_STEP,
                nbIter=self.iters_per_step,
                seed=self._seed,
                mu=25,
                lambda_=40,
                nbGranular=40,
                targetFeasible=0.2,
                nbElite=4,
                nbClose=5,
            )

        elif action == 4:
            # HIGH_TURNOVER: Tiny base pop, huge offspring count, very low feasibility.
            # Maximum churn — generates many solutions but keeps very few.
            # Aggressive exploration strategy that's high-risk, high-reward.
            return hgs.AlgorithmParameters(
                timeLimit=TIME_PER_STEP,
                nbIter=self.iters_per_step,
                seed=self._seed,
                mu=10,
                lambda_=80,
                nbGranular=20,
                targetFeasible=0.05,
                nbElite=2,
                nbClose=3,
            )

        elif action == 5:
            # STABLE_ELITE: Large base pop, fewer offspring, high feasibility.
            # Conservative strategy — maintains a large pool of good feasible
            # solutions and refines them slowly. Good when close to optimal.
            return hgs.AlgorithmParameters(
                timeLimit=TIME_PER_STEP,
                nbIter=self.iters_per_step,
                seed=self._seed,
                mu=50,
                lambda_=30,
                nbGranular=25,
                targetFeasible=0.4,
                nbElite=8,
                nbClose=7,
            )

        elif action == 6:
            # EXPLORE_NEW_SEED: Default params but with a fresh random seed.
            # Restarts the stochastic search from a different starting point.
            # Use when stagnated — a new seed can find completely different solutions.
            self._seed = random.randint(0, 2**31)
            return hgs.AlgorithmParameters(
                timeLimit=TIME_PER_STEP,
                nbIter=self.iters_per_step,
                seed=self._seed,
            )

        else:
            raise ValueError(f"Invalid action: {action}. Expected 0-6.")
