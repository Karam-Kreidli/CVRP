"""
Stage 3 - Solver Engine: Gymnasium environment wrapping the HGS-CVRP solver.

This is the BRIDGE between the RL agent and the actual CVRP solver. The agent
makes high-level strategic decisions — whether to push for fewer vehicles,
try a new random seed, or refine at the current fleet size — and the environment
translates those into HGS solver calls.

HOW IT WORKS:
  The environment follows the standard Gymnasium (OpenAI Gym) interface:
    1. reset()  — Load a random CVRP instance, compute hand-crafted features, do initial solve
    2. step(action) — Execute the chosen strategy (fleet target + seed + budget)
    3. Repeat step() MAX_STEPS times → episode ends

WHAT THE ACTIONS ACTUALLY DO:
  Each action is a (fleet_target, seed_strategy, iteration_budget) tuple.
  The agent's real decisions are:
    - Should I try to REDUCE the fleet? (huge score impact but risky)
    - Should I try a NEW SEED? (escape local optima)
    - How much COMPUTE to invest? (tighter fleet gets more budget)
    - Should I use a different HGS search bias? (for selected actions)

    0 = FREE_SAME        — Let HGS decide fleet, same seed, 500 iters
    1 = FREE_NEW         — Let HGS decide fleet, new seed, 500 iters
    2 = LOCK_SAME        — Lock at current best NV, same seed, 500 iters
    3 = LOCK_NEW         — Lock at current best NV, new seed, 500 iters
    4 = PUSH_SAME        — Try best_nv - 1, same seed, 1000 iters
    5 = PUSH_NEW         — Try best_nv - 1, new seed, 1000 iters
    6 = FORCE_MIN        — Force theoretical minimum NV, new seed, 1500 iters
    7 = FREE_DIVERSE_NEW — Free fleet, new seed, 500 iters, diversity-biased HGS params
    8 = LOCK_AGGR_NEW    — Lock best NV, new seed, 500 iters, aggressive local-search bias
    9 = PUSH_BALANCED_NEW — Push best_nv - 1, new seed, 1000 iters, balanced HGS params

REWARD:
    Percentage-based improvement over the episode-best score.
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
INSTANCE_FEATURES_DIM = 12   # Hand-crafted instance features
STATS_DIM = 7                # Solver stats: time, nv_ratio, score_ratio, stagnation, nv_gap, last_reward, last_action
OBS_DIM = INSTANCE_FEATURES_DIM + STATS_DIM  # 19 = total observation dimension
NUM_ACTIONS = 10             # Number of discrete strategy choices
MAX_STEPS = 50               # Steps per episode
TIME_PER_STEP = 0.0          # HGS time limit per step (0 = use nbIter only)

# Iteration budgets per action (tighter fleet targets get more compute)
ITERS_FREE = 500             # Free / lock actions — quick solves
ITERS_PUSH = 1000            # Push (best_nv - 1) — needs more search
ITERS_FORCE = 1500           # Force minimum fleet — hardest, gets most budget
ITERS_PER_STEP = ITERS_FREE  # Default for compatibility (used in train.py logging)

ACTION_NAMES = [
    "FREE_SAME", "FREE_NEW", "LOCK_SAME",
    "LOCK_NEW", "PUSH_SAME", "PUSH_NEW", "FORCE_MIN",
    "FREE_DIVERSE_NEW", "LOCK_AGGR_NEW", "PUSH_BALANCED_NEW",
]

# Action-specific HGS parameter overrides for the three newly added actions.
# These presets are intentionally low-cost (same iteration tiers) and only
# alter search bias, not fundamental solver behavior.
ACTION_HGS_OVERRIDES: dict[int, dict[str, float | int]] = {
    7: dict(mu=40, lambda_=60, nbGranular=30, targetFeasible=0.3, nbElite=6, nbClose=8),
    8: dict(mu=15, lambda_=20, nbGranular=15, targetFeasible=0.1, nbElite=2, nbClose=3),
    9: dict(mu=20, lambda_=60, nbGranular=25, targetFeasible=0.15, nbElite=3, nbClose=4),
}


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
            0.  size_norm:            num_customers / 400
            1.  demand_fill_ratio:    total_demand / (nv_min * capacity)
            2.  mean_dist_norm:       mean inter-customer distance / max distance
            3.  std_dist_norm:        std of inter-customer distances / max distance
            4.  depot_centrality:     mean depot-to-customer distance / max distance
            5.  demand_cv:            coefficient of variation of demands (std/mean)
            6.  capacity_tightness:   max single demand / capacity
            7.  demand_minmax_ratio:  min demand / max demand
            8.  top3_demand_share:    sum(top-3 demands) / total demand
            9.  depot_distance_cv:    std(depot distances) / mean(depot distances)
            10. bbox_aspect_ratio:    min(bbox width, bbox height) / max(bbox width, bbox height)
            11. radial_outlier_ratio: fraction of customers beyond Q3 + 1.5 * IQR in depot distances
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

    # 7. Demand min/max ratio — lower means stronger demand imbalance.
    min_demand = cust_demands.min() if num_customers > 0 else 0.0
    demand_minmax_ratio = min_demand / max(max_demand, 1e-8)

    # 8. Top-3 demand share — concentration of total load in a few customers.
    if num_customers > 0 and total_demand > 0:
        m = min(3, num_customers)
        top_m = np.partition(cust_demands, -m)[-m:]
        top3_demand_share = top_m.sum() / max(total_demand, 1e-8)
    else:
        top3_demand_share = 0.0

    # 9. Depot-distance CV — radial imbalance around depot.
    if len(depot_dists) > 0:
        depot_distance_cv = depot_dists.std() / max(depot_dists.mean(), 1e-8)
    else:
        depot_distance_cv = 0.0
    depot_distance_cv = min(depot_distance_cv, 2.0) / 2.0

    # 10. Bounding-box aspect ratio — elongated vs compact geometry.
    if num_customers > 1:
        bbox_w = float(cx.max() - cx.min())
        bbox_h = float(cy.max() - cy.min())
        bbox_aspect_ratio = min(bbox_w, bbox_h) / max(max(bbox_w, bbox_h), 1e-8)
    else:
        bbox_aspect_ratio = 1.0

    # 11. Radial outlier ratio — share of customers far from the main radial mass.
    if len(depot_dists) >= 4:
        q1, q3 = np.percentile(depot_dists, [25, 75])
        iqr = max(q3 - q1, 0.0)
        outlier_threshold = q3 + 1.5 * iqr
        radial_outlier_ratio = float(np.mean(depot_dists > outlier_threshold))
    else:
        radial_outlier_ratio = 0.0

    return np.array([
        size_norm,
        demand_fill_ratio,
        mean_dist_norm,
        std_dist_norm,
        depot_centrality,
        demand_cv,
        capacity_tightness,
        demand_minmax_ratio,
        top3_demand_share,
        depot_distance_cv,
        bbox_aspect_ratio,
        radial_outlier_ratio,
    ], dtype=np.float32)


class CVRPEnv(gym.Env):
    """Gymnasium environment that wraps HGS-CVRP for RL-guided solving.

    This class is the interface between the RL world and the solver world.
    It follows the standard Gymnasium API so that any RL algorithm (PPO, A2C,
    DQN, etc.) can interact with it via reset() and step().

    EPISODE LIFECYCLE:
      1. reset() picks a random .vrp instance, computes hand-crafted features,
         runs an initial solve with default HGS parameters.
      2. step(action) executes the chosen strategy: sets fleet target, seed,
         and iteration budget, then runs a fresh HGS solve.
      3. After MAX_STEPS steps, the episode ends (truncated=True).

    ACTION SPACE:
      Each action controls three levers:
        - Fleet target: free / lock at best_nv / push to best_nv-1 / force nv_min
        - Seed: same seed (reproducible) or new seed (escape local optima)
        - Iteration budget: 500 (quick) / 1000 (medium) / 1500 (thorough)
                - Optional HGS search-bias overrides for selected actions

    Args:
        instance_paths: List of paths to .vrp files (X-dataset format).
        device: Torch device (kept for compatibility, minimal GPU use now).
        max_steps: Steps per episode.
        max_nodes: Curriculum filter — only use instances with <= N customers.
    """

    metadata = {"render_modes": []}

    # Iteration budgets per action type
    ITERS_FREE = ITERS_FREE
    ITERS_PUSH = ITERS_PUSH
    ITERS_FORCE = ITERS_FORCE
    ITERS_PER_STEP = ITERS_PER_STEP  # For backward compat (logging)

    def __init__(
        self,
        instance_paths: list[str | pathlib.Path],
        device: torch.device = torch.device("cpu"),
        max_steps: int = MAX_STEPS,
        max_nodes: int | None = None,
    ):
        super().__init__()

        self._all_instance_paths = [pathlib.Path(p) for p in instance_paths]
        self.max_nodes = max_nodes
        self.instance_paths = self._filter_by_nodes(self._all_instance_paths)
        self.device = device
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
        self._prev_cand_score: float = 0.0           # Last candidate score (debug/info tracking)
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
        """Return a boolean mask over 10 actions. True = allowed, False = blocked.

        Block fleet-reduction actions when they can't possibly help:
        - PUSH (4,5,9) blocked when best_nv <= nv_min (can't go lower)
        - FORCE_MIN (6) blocked when best_nv <= nv_min (already there)
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        if self._best_nv <= self._nv_min:
            mask[4] = False  # PUSH_SAME
            mask[5] = False  # PUSH_NEW
            mask[6] = False  # FORCE_MIN
            mask[9] = False  # PUSH_BALANCED_NEW
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

    def _solve_hgs(
        self,
        params: hgs.AlgorithmParameters,
        num_vehicles: int | None = None,
    ) -> dict:
        """Run HGS-CVRP with the given parameters and return result dict.

        Args:
            params: HGS algorithm parameters (iterations, seed, etc.)
            num_vehicles: If set, force HGS to use at most this many vehicles.
                         None = use the default upper bound (free fleet).

        Returns:
            dict with 'nv' (number of vehicles), 'td' (total distance),
            'score' (competition score), 'routes' (list of routes).
        """
        # Temporarily override num_vehicles if a fleet target is specified
        data = self._hgs_data
        old_nv = data.get("num_vehicles")
        if num_vehicles is not None:
            data["num_vehicles"] = num_vehicles

        try:
            solver = hgs.Solver(parameters=params, verbose=False)
            result = solver.solve_cvrp(data, rounding=True)
        except Exception:
            # HGS can throw when fleet size is too tight.
            if num_vehicles is not None:
                data["num_vehicles"] = old_nv
            return {
                "nv": 9999,
                "td": float("inf"),
                "score": float("inf"),
                "routes": [],
                "failed": True,
            }

        # Restore original num_vehicles
        if num_vehicles is not None:
            data["num_vehicles"] = old_nv

        nv = len(result.routes)
        td = result.cost

        # HGS silently returns 0 routes / 0 cost when it can't find a
        # feasible solution with the constrained fleet size.
        if nv == 0 or td <= 0:
            return {
                "nv": 9999,
                "td": float("inf"),
                "score": float("inf"),
                "routes": [],
                "failed": True,
            }

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

        # Initial solve with default HGS parameters (free fleet, standard budget)
        default_params = hgs.AlgorithmParameters(
            timeLimit=TIME_PER_STEP,
            nbIter=ITERS_FREE,
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
        """Execute one HGS solve with the chosen strategy.

        Each action controls three levers:
          - Fleet target: free / lock best_nv / push best_nv-1 / force nv_min
          - Seed: same (reproducible) or new (escape local optima)
          - Iteration budget: 500 / 1000 / 1500 (tighter fleet = more compute)

        Reward design:
                    - Compare against EPISODE BEST score (not previous candidate)
          - Percentage-based improvement for instance-size normalization
          - Small negative (-0.5) for no improvement (cost of compute)
          - Larger penalty (-5.0) for fleet explosions (NV spike > 2)

        Args:
            action: 0-9, one of the ten strategy actions.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self._hgs_data is not None, "Must call reset() before step()"
        self._step_count += 1

        # Translate action to (HGS params, fleet target)
        params, num_vehicles, iters_used = self._action_to_params(action)

        # Run fresh HGS solve with optional fleet constraint
        result = self._solve_hgs(params, num_vehicles=num_vehicles)

        solve_failed = result.get("failed", False)
        cand_nv = result["nv"]
        cand_td = result["td"]
        cand_score = result["score"]

        # Check for fleet explosion (NV spiked by more than 2 vs best)
        # Also triggered by a failed solve (nv=9999)
        fleet_exploded = solve_failed or (cand_nv > self._best_nv + 2)

        # --- Reward: percentage improvement over episode best ---
        if fleet_exploded:
            self._iters_since_improvement += iters_used
            reward = self.FAILURE_PENALTY
            cand_score = self._best_score  # Don't let failed solve corrupt tracking
        elif cand_score < self._best_score:
            # Beat the episode best — this is a real improvement
            pct = (self._best_score - cand_score) / max(self._best_score, 1.0)
            reward = pct * 100.0
        else:
            # No improvement over best — small penalty for wasted compute
            self._iters_since_improvement += iters_used
            reward = -0.5

        # Update best-of-N tracking
        if cand_score < self._best_score:
            self._best_nv = cand_nv
            self._best_td = cand_td
            self._best_score = cand_score
            self._iters_since_improvement = 0

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

                instance_features (12) + solver_stats (7):
          Instance features (computed once per episode):
            - size_norm, demand_fill_ratio, mean_dist_norm, std_dist_norm,
                            depot_centrality, demand_cv, capacity_tightness,
                            demand_minmax_ratio, top3_demand_share, depot_distance_cv,
                            bbox_aspect_ratio, radial_outlier_ratio

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

        # Approximate total budget (using average per-step iters for normalization)
        avg_iters = (ITERS_FREE + ITERS_PUSH + ITERS_FORCE) / 3.0
        total_budget = self.max_steps * avg_iters
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

    def _action_to_params(
        self, action: int
    ) -> tuple[hgs.AlgorithmParameters, int | None, int]:
        """Map the agent's discrete action to (HGS params, fleet target, iters_used).

        Each action is a (fleet_target, seed_strategy, iteration_budget) combo:

          0 = FREE_SAME   — Free fleet, same seed, 500 iters
          1 = FREE_NEW    — Free fleet, new seed, 500 iters
          2 = LOCK_SAME   — Lock at best_nv, same seed, 500 iters
          3 = LOCK_NEW    — Lock at best_nv, new seed, 500 iters
          4 = PUSH_SAME   — Push to best_nv-1, same seed, 1000 iters
          5 = PUSH_NEW    — Push to best_nv-1, new seed, 1000 iters
          6 = FORCE_MIN   — Force nv_min, new seed, 1500 iters
          7 = FREE_DIVERSE_NEW  — Free fleet, new seed, 500 iters, diversity-biased params
          8 = LOCK_AGGR_NEW     — Lock at best_nv, new seed, 500 iters, aggressive params
          9 = PUSH_BALANCED_NEW — Push to best_nv-1, new seed, 1000 iters, balanced params

        Returns:
            params: HGS AlgorithmParameters
            num_vehicles: fleet target (None = free, int = forced)
            iters_used: iteration budget for this step
        """
        # Determine seed
        new_seed = action in (1, 3, 5, 6, 7, 8, 9)
        if new_seed:
            self._seed = random.randint(0, 2**31)

        # Determine fleet target and iteration budget
        if action in (0, 1):
            # FREE: let HGS decide fleet size
            num_vehicles = None
            iters = ITERS_FREE
        elif action in (2, 3):
            # LOCK: constrain to current best fleet size
            num_vehicles = self._best_nv
            iters = ITERS_FREE
        elif action in (4, 5):
            # PUSH: try one fewer vehicle than current best
            num_vehicles = max(self._nv_min, self._best_nv - 1)
            iters = ITERS_PUSH
        elif action == 6:
            # FORCE_MIN: force theoretical minimum fleet
            num_vehicles = self._nv_min
            iters = ITERS_FORCE
        elif action == 7:
            # FREE_DIVERSE_NEW: unconstrained fleet with diversity-focused search
            num_vehicles = None
            iters = ITERS_FREE
        elif action == 8:
            # LOCK_AGGR_NEW: lock fleet and use aggressive local-search settings
            num_vehicles = self._best_nv
            iters = ITERS_FREE
        elif action == 9:
            # PUSH_BALANCED_NEW: push one vehicle lower with balanced exploration
            num_vehicles = max(self._nv_min, self._best_nv - 1)
            iters = ITERS_PUSH
        else:
            raise ValueError(f"Invalid action: {action}. Expected 0-9.")

        # Apply optional action-specific HGS search-bias overrides.
        hgs_overrides = ACTION_HGS_OVERRIDES.get(action, {})

        params = hgs.AlgorithmParameters(
            timeLimit=TIME_PER_STEP,
            nbIter=iters,
            seed=self._seed,
            **hgs_overrides,
        )
        return params, num_vehicles, iters
