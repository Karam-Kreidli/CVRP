"""
Stage 3 - Solver Engine: Gymnasium environment wrapping the HGS-CVRP solver.

This is the BRIDGE between the RL agent and the actual CVRP solver. It translates
the Fleet Manager's high-level strategy decisions into concrete HGS algorithm
parameters (population sizes, search granularity, feasibility targets, etc.).

HOW IT WORKS:
  The environment follows the standard Gymnasium (OpenAI Gym) interface:
    1. reset()  — Load a random CVRP instance, compute hand-crafted features, do initial solve
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
  Percentage-based improvement over the previous step's candidate score.
  Positive when score improves, small negative (-0.5) when no improvement,
  larger penalty (-5.0) for fleet explosions.

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INSTANCE_FEATURES_DIM = 7    # Hand-crafted instance features (replaces GNN)
STATS_DIM = 7                # Solver stats: time, nv_ratio, score_ratio, stagnation, nv_gap, last_reward, last_action
OBS_DIM = INSTANCE_FEATURES_DIM + STATS_DIM  # 14 = total observation dimension
NUM_ACTIONS = 7              # Number of discrete strategy choices
ITERS_PER_STEP = 500         # HGS iterations (nbIter) per environment step
MAX_STEPS = 50               # Steps per episode (total budget: 500 * 50 = 25,000)
TIME_PER_STEP = 0.0          # HGS time limit per step (0 = use nbIter only)

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


def _compute_instance_features(data: dict) -> np.ndarray:
    """Compute hand-crafted instance features (replaces untrained GNN).

    Returns a 1-D numpy array of INSTANCE_FEATURES_DIM floats, all roughly
    in [0, 1] range for stable learning.

    Features:
      0. size_norm:          num_customers / 400
      1. demand_fill_ratio:  total_demand / (nv_min * capacity)
      2. mean_dist_norm:     mean inter-customer distance / max distance
      3. std_dist_norm:      std of inter-customer distances / max distance
      4. depot_centrality:   mean depot-to-customer distance / max distance
      5. demand_cv:          coefficient of variation of demands (std/mean)
      6. capacity_tightness: max single demand / capacity
    """
    x = data["x_coordinates"]
    y = data["y_coordinates"]
    demands = data["demands"]
    capacity = float(data["vehicle_capacity"])
    n = len(x)  # includes depot
    num_customers = n - 1

    # 0. Normalized instance size
    size_norm = num_customers / 400.0

    # 1. Demand fill ratio — how tightly packed the vehicles are
    total_demand = demands[1:].sum()  # exclude depot
    nv_min = max(1, math.ceil(total_demand / capacity))
    demand_fill_ratio = total_demand / (nv_min * capacity) if capacity > 0 else 1.0

    # Customer coordinates (exclude depot at index 0)
    cx = x[1:]
    cy = y[1:]

    # Pairwise distances (subsample if too large for memory)
    if num_customers <= 200:
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        # Exclude diagonal (self-distances)
        mask = ~np.eye(num_customers, dtype=bool)
        pairwise = dists[mask]
    else:
        # Subsample 200 customers for speed
        idx = np.random.choice(num_customers, 200, replace=False)
        sx, sy = cx[idx], cy[idx]
        dx = sx[:, None] - sx[None, :]
        dy = sy[:, None] - sy[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        mask = ~np.eye(200, dtype=bool)
        pairwise = dists[mask]

    max_dist = pairwise.max() if len(pairwise) > 0 else 1.0
    max_dist = max(max_dist, 1e-8)

    # 2. Mean distance normalized
    mean_dist_norm = pairwise.mean() / max_dist if len(pairwise) > 0 else 0.5

    # 3. Std distance normalized
    std_dist_norm = pairwise.std() / max_dist if len(pairwise) > 0 else 0.0

    # 4. Depot centrality — how central the depot is
    depot_dists = np.sqrt((cx - x[0])**2 + (cy - y[0])**2)
    depot_centrality = depot_dists.mean() / max_dist if max_dist > 0 else 0.5

    # 5. Demand coefficient of variation
    cust_demands = demands[1:]
    d_mean = cust_demands.mean() if num_customers > 0 else 1.0
    d_std = cust_demands.std() if num_customers > 0 else 0.0
    demand_cv = d_std / max(d_mean, 1e-8)
    demand_cv = min(demand_cv, 2.0) / 2.0  # Clip and normalize to [0, 1]

    # 6. Capacity tightness — largest single demand relative to vehicle capacity
    max_demand = cust_demands.max() if num_customers > 0 else 0.0
    capacity_tightness = max_demand / capacity if capacity > 0 else 1.0

    return np.array([
        size_norm,
        demand_fill_ratio,
        mean_dist_norm,
        std_dist_norm,
        depot_centrality,
        demand_cv,
        capacity_tightness,
    ], dtype=np.float32)


class CVRPEnv(gym.Env):
    """Gymnasium environment that wraps HGS-CVRP for RL-guided solving.

    This class is the interface between the RL world and the solver world.
    It follows the standard Gymnasium API so that any RL algorithm (PPO, A2C,
    DQN, etc.) can interact with it via reset() and step().

    EPISODE LIFECYCLE:
      1. reset() picks a random .vrp instance, computes hand-crafted features,
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
        device: Torch device (kept for compatibility, minimal GPU use now).
        iters_per_step: HGS iterations per step (nbIter parameter).
        max_steps: Steps per episode.
        max_nodes: Curriculum filter — only use instances with <= N customers.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_paths: list[str | pathlib.Path],
        device: torch.device = torch.device("cpu"),
        iters_per_step: int = ITERS_PER_STEP,
        max_steps: int = MAX_STEPS,
        max_nodes: int | None = None,
    ):
        super().__init__()

        self._all_instance_paths = [pathlib.Path(p) for p in instance_paths]
        self.max_nodes = max_nodes
        self.instance_paths = self._filter_by_nodes(self._all_instance_paths)
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
        self._instance_features: np.ndarray | None = None  # Hand-crafted features
        self._nv_initial: int = 0                    # Fleet size after initial solve
        self._nv_min: int = 1                        # Theoretical minimum fleet
        self._prev_cand_score: float = 0.0           # Previous step's candidate score
        self._step_count: int = 0
        self._iters_since_improvement: int = 0       # Stagnation counter
        self._seed: int = 0
        self._capacity: int = 1                      # Vehicle capacity (for nv_min)
        self._last_reward: float = 0.0               # Reward from previous step
        self._last_action: int = 0                   # Action from previous step

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
        """Start a new episode: load instance, compute features, do initial solve.

        Returns:
            obs: (OBS_DIM,) numpy array — the agent's first observation
            info: dict with instance name, initial NV/TD/score, action mask
        """
        super().reset(seed=seed)

        # Pick a random instance
        path = random.choice(self.instance_paths)
        self._vrp_path = path
        self._hgs_data = _parse_vrp_file(path)
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._nv_min = self._compute_nv_min()

        # Compute hand-crafted instance features (replaces GNN)
        self._instance_features = _compute_instance_features(self._hgs_data)

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
        self._prev_cand_score = result["score"]
        self._step_count = 0
        self._iters_since_improvement = 0
        self._last_reward = 0.0
        self._last_action = 0

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
        tracks the best solution found across all steps.

        Reward design:
          - Compare against PREVIOUS step's candidate (not episode-best)
          - Percentage-based improvement for instance-size normalization
          - Small negative (-0.5) for no improvement (cost of compute)
          - Larger penalty (-5.0) for fleet explosions

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

        # --- Reward: percentage improvement over previous candidate ---
        if fleet_exploded:
            self._iters_since_improvement += self.iters_per_step
            reward = self.FAILURE_PENALTY
        elif cand_score < self._prev_cand_score:
            # Improvement over previous step's candidate
            pct = (self._prev_cand_score - cand_score) / max(self._prev_cand_score, 1.0)
            reward = pct * 100.0
        else:
            # No improvement — small penalty for wasted compute
            self._iters_since_improvement += self.iters_per_step
            reward = -0.5

        # Update best-of-N tracking (still keep the best solution found)
        if cand_score < self._best_score:
            self._best_nv = cand_nv
            self._best_td = cand_td
            self._best_score = cand_score
            self._iters_since_improvement = 0

        # Track for next step's reward comparison
        self._prev_cand_score = cand_score
        self._last_reward = reward
        self._last_action = action

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

    def _build_observation(self) -> np.ndarray:
        """Construct the observation vector.

        instance_features (7) + solver_stats (7):
          Instance features (computed once per episode):
            - size_norm, demand_fill_ratio, mean_dist_norm, std_dist_norm,
              depot_centrality, demand_cv, capacity_tightness

          Solver stats (updated each step):
            - time_ratio: step progress through episode
            - nv_ratio: current best NV / initial NV
            - score_ratio: current best score / initial score (< 1 means improvement)
            - stagnation_ratio: steps without improvement / total budget
            - nv_gap: (best_nv - nv_min) / nv_initial — distance to fleet minimum
            - last_reward: reward from previous step (clipped to [-5, 10])
            - last_action_norm: previous action / NUM_ACTIONS
        """
        time_ratio = self._step_count / self.max_steps

        nv_ratio = self._best_nv / max(self._nv_initial, 1)

        initial_score = competition_score(self._nv_initial, self._best_td)
        score_ratio = self._best_score / max(initial_score, 1.0)

        total_budget = self.max_steps * self.iters_per_step
        stagnation_ratio = self._iters_since_improvement / max(total_budget, 1)

        nv_gap = (self._best_nv - self._nv_min) / max(self._nv_initial, 1)

        last_reward_clipped = np.clip(self._last_reward, -5.0, 10.0) / 10.0

        last_action_norm = self._last_action / NUM_ACTIONS

        stats = np.array([
            time_ratio, nv_ratio, score_ratio, stagnation_ratio,
            nv_gap, last_reward_clipped, last_action_norm,
        ], dtype=np.float32)

        obs = np.concatenate([self._instance_features, stats])
        return obs

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
