"""
Stage 3 - Solver Engine: Gymnasium environment wrapping PyVRP's ILS solver.

This is the BRIDGE between the RL agent and the actual CVRP solver. It translates
the Fleet Manager's high-level strategy decisions (e.g., "apply moderate pressure")
into concrete PyVRP solver parameters (e.g., penalty_increase=5.0, target_feasible=0.3).

HOW IT WORKS:
  The environment follows the standard Gymnasium (OpenAI Gym) interface:
    1. reset()  — Load a random CVRP instance, encode it with the GNN, do an initial solve
    2. step(action) — Apply the chosen strategy for 1000 ILS iterations, return reward
    3. Repeat step() 20 times → episode ends (20 steps × 1000 iters = 20,000 total)

WHAT THE ACTIONS ACTUALLY DO:
  The Fleet Manager doesn't touch routes directly. It controls PyVRP's PENALTY
  PARAMETERS, which determine how aggressively the solver penalizes solutions
  that use "too many" vehicles:

  0 = POLISH              — Default params, same seed (just optimize routes)
  1 = MILD_PRESSURE       — penalty_increase=2.0  (gentle nudge toward fewer vehicles)
  2 = MODERATE_PRESSURE   — penalty_increase=5.0  (steady fleet reduction)
  3 = AGGRESSIVE_PRESSURE — penalty_increase=10.0 (force route merges, risky!)
  4 = EXPLORE_NEW_SEED    — Fresh random seed, default params (escape local optima)
  5 = EXPLORE_PRESSURE    — New seed + moderate penalty (escape AND reduce)

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
EMBED_DIM = 128          # Graph embedding dimension (from GNN Encoder)
STATS_DIM = 4            # Solver statistics: [time_ratio, nv_ratio, violation_ratio, stagnation]
OBS_DIM = EMBED_DIM + STATS_DIM  # 132 = total observation dimension for the agent
NUM_ACTIONS = 6          # Number of discrete strategy choices
ITERS_PER_STEP = 1_000   # PyVRP ILS iterations per environment step
MAX_STEPS = 20           # Steps per episode → 20 × 1000 = 20,000 total iterations

ACTION_NAMES = [
    "POLISH", "MILD_PRESSURE", "MODERATE_PRESSURE",
    "AGGRESSIVE_PRESSURE", "EXPLORE_NEW_SEED", "EXPLORE_PRESSURE",
]


def competition_score(nv: int, td: float) -> float:
    """Compute the GECCO 2026 ML4VRP objective: 1000 * NV + TD.

    The 1000x multiplier on NV means that fleet size dominates:
    removing just one vehicle saves 1000 distance units worth of score.
    """
    return 1000.0 * nv + td


class CVRPEnv(gym.Env):
    """Gymnasium environment that wraps PyVRP for RL-guided CVRP solving.

    This class is the interface between the RL world and the solver world.
    It follows the standard Gymnasium API so that any RL algorithm (PPO, A2C,
    DQN, etc.) can interact with it via reset() and step().

    EPISODE LIFECYCLE:
      1. reset() picks a random .vrp instance, encodes it with the GNN,
         runs an initial 1000-iteration solve, and returns the first observation.
      2. step(action) maps the chosen action to PyVRP parameters, runs 1000
         more iterations, and returns (observation, reward, done, info).
      3. After 20 steps, the episode ends (truncated=True).

    The agent never sees raw routes or coordinates during decision-making —
    it only sees the 132-dim observation vector.

    Args:
        instance_paths: List of paths to .vrp files (X-dataset format).
        encoder: GNNEncoder for computing graph embeddings (used at reset).
        device: Torch device for encoder inference.
        iters_per_step: PyVRP iterations per step (default: 1000).
        max_steps: Steps per episode (default: 20 → 20,000 total iterations).
        round_func: Rounding for .vrp file parsing ("round" for X-dataset).
        max_nodes: Curriculum filter — only use instances with ≤ N customers.
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

        # Store all instance paths, then filter by curriculum constraint
        self._all_instance_paths = [pathlib.Path(p) for p in instance_paths]
        self.max_nodes = max_nodes
        self.instance_paths = self._filter_by_nodes(self._all_instance_paths)
        self.encoder = encoder
        self.device = device
        self.iters_per_step = iters_per_step
        self.max_steps = max_steps
        self.round_func = round_func

        # Gymnasium space definitions (required by the Gym API)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # --- Episode State ---
        # These are all reset at the start of each episode.
        self._data: ProblemData | None = None          # Parsed .vrp problem data
        self._best_solution: Solution | None = None    # Best solution found so far
        self._graph_embedding: torch.Tensor | None = None  # GNN output (reused all episode)
        self._node_embeddings: torch.Tensor | None = None  # Per-node embeddings
        self._nv_initial: int = 0                      # Fleet size after initial solve
        self._nv_min: int = 1                          # Theoretical minimum fleet (ceil(demand/capacity))
        self._prev_score: float = 0.0                  # Score from previous step (for reward calc)
        self._step_count: int = 0                      # Current step within episode
        self._total_iters: int = 0                     # Cumulative ILS iterations
        self._iters_since_improvement: int = 0         # Stagnation counter
        self._seed: int = 0                            # Random seed for PyVRP solver

    def _compute_nv_min(self) -> int:
        """Compute theoretical minimum fleet size: ceil(total_demand / vehicle_capacity).

        This is a LOWER BOUND — you can't serve all customers with fewer vehicles
        than this, because each vehicle can only carry 'capacity' units. In practice,
        the actual minimum may be higher due to distance/routing constraints, but
        this is a safe floor for action masking.
        """
        data = self._data
        total_demand = sum(c.delivery[0] for c in data.clients())
        capacity = data.vehicle_type(0).capacity[0]
        return max(1, math.ceil(total_demand / capacity))

    def get_action_mask(self) -> np.ndarray:
        """Return a boolean mask over 6 actions. True = allowed, False = blocked.

        ACTION MASKING prevents the agent from wasting steps on impossible actions.
        When the fleet is already at the theoretical minimum (NV == NV_min), there's
        no point applying pressure to reduce further. We block actions 1, 2, 3, 5
        (all the "pressure" variants) and only allow POLISH (0) and EXPLORE (4).

        This is important for training efficiency — without masking, the agent would
        waste many episodes learning by trial-and-error that pressure at NV_min is futile.
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)  # Start with all actions allowed
        if self._best_solution is not None:
            current_nv = self._best_solution.num_routes()
            if current_nv <= self._nv_min:
                # Can't reduce fleet below minimum — block all pressure actions
                mask[1] = False  # MILD_PRESSURE
                mask[2] = False  # MODERATE_PRESSURE
                mask[3] = False  # AGGRESSIVE_PRESSURE
                mask[5] = False  # EXPLORE_PRESSURE
        return mask

    def _filter_by_nodes(self, paths: list[pathlib.Path]) -> list[pathlib.Path]:
        """Filter instance paths by customer count for curriculum learning.

        X-dataset filenames follow the pattern "X-nNNN-kKK.vrp" where NNN is
        the number of customers. This parses the filename to extract N and
        filters to only include instances with N <= max_nodes.

        Used during early training to start with easier (smaller) instances
        before expanding to the full dataset.
        """
        if self.max_nodes is None:
            return list(paths)
        filtered = []
        for p in paths:
            try:
                # Parse "X-n101-k25.vrp" → n=101
                n = int(p.stem.split("-")[1][1:])
                if n <= self.max_nodes:
                    filtered.append(p)
            except (IndexError, ValueError):
                filtered.append(p)  # Keep non-standard filenames (safety)
        return filtered if filtered else list(paths)

    def set_max_nodes(self, max_nodes: int | None):
        """Update the curriculum: change max instance size and re-filter.

        Called by the training loop at the curriculum boundary (epoch 21)
        to unlock larger instances. Pass None to remove the filter entirely.
        """
        self.max_nodes = max_nodes
        self.instance_paths = self._filter_by_nodes(self._all_instance_paths)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode: load instance, encode it, do initial solve.

        This is called at the start of each training episode. It:
          1. Picks a random .vrp instance from the (curriculum-filtered) pool
          2. Runs the GNN Encoder ONCE to produce the graph embedding
             (this embedding is reused for all 20 steps — the instance doesn't change)
          3. Runs an initial 1000-iteration solve to get a starting solution
          4. Returns the first observation and metadata

        Returns:
            obs: (132,) numpy array — the agent's first observation
            info: dict with instance name, initial NV/TD/score, action mask
        """
        super().reset(seed=seed)

        # Pick a random instance from the available pool
        path = random.choice(self.instance_paths)
        self._data = read(str(path), round_func=self.round_func)
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._nv_min = self._compute_nv_min()

        # Encode instance with GNN (Stage 1) — done ONCE per episode
        # The graph_embedding captures the spatial structure of this instance
        self._graph_embedding, self._node_embeddings = self._encode_instance()

        # Initial solve with default parameters to get a starting solution
        # The agent will then try to IMPROVE this solution over 20 steps
        res = pyvrp.solve(
            self._data,
            stop=MaxIterations(self.iters_per_step),
            seed=self._seed,
            display=False,
            collect_stats=False,
        )
        self._best_solution = res.best
        self._total_iters = res.num_iterations

        # Record initial state for reward computation and observation building
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

    # Fixed penalty returned when a pressure action causes the fleet to explode.
    # This teaches the agent that aggressive pressure can backfire.
    FAILURE_PENALTY = -5.0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one search strategy step (the core RL interaction).

        This is where the RL loop happens:
          1. Agent chooses an action (0-5)
          2. We translate that action into PyVRP solver parameters
          3. PyVRP runs 1000 ILS iterations with those parameters
          4. We compute the reward (did the score improve?)
          5. Return the new observation for the next decision

        WARM STARTING: Each step passes the current best solution as the starting
        point for PyVRP. The solver doesn't start from scratch — it builds on
        the best solution found so far.

        FLEET EXPLOSION SAFETY: If a pressure action causes the fleet to spike
        by more than 2 vehicles, we REJECT the new solution (keep the old one)
        and return a penalty reward of -5.0. This prevents catastrophic regressions
        where the solver "panics" and adds vehicles to satisfy constraints.

        Args:
            action: 0-5, one of the six strategy actions.

        Returns:
            observation: (132,) numpy array for the next decision
            reward: float — positive if score improved, negative if worsened
            terminated: always False (no early termination)
            truncated: True after 20 steps (episode budget exhausted)
            info: dict with detailed step metadata for logging
        """
        assert self._data is not None, "Must call reset() before step()"
        self._step_count += 1

        # STEP 1: Translate the agent's action into solver parameters
        params, step_seed = self._action_to_params(action)

        # STEP 2: Run PyVRP's ILS solver for 1000 iterations
        # initial_solution = warm start from the best solution found so far
        res = pyvrp.solve(
            self._data,
            stop=MaxIterations(self.iters_per_step),
            seed=step_seed,
            display=False,
            collect_stats=False,
            params=params,
            initial_solution=self._best_solution,  # Warm start!
        )
        self._total_iters += res.num_iterations

        # STEP 3: Evaluate the candidate solution
        candidate = res.best
        cand_nv = candidate.num_routes()
        cand_td = candidate.distance()
        cand_score = competition_score(cand_nv, cand_td)

        # SAFETY CHECK: Fleet explosion detection
        # If a pressure action caused the solver to ADD vehicles (the opposite of
        # what we wanted), reject the solution and apply a penalty.
        # Threshold: NV increased by more than 2 from the previous best.
        prev_nv = self._best_solution.num_routes()
        is_pressure = action in (1, 2, 3, 5)
        fleet_exploded = (is_pressure and cand_nv > prev_nv + 2)

        if fleet_exploded:
            # Solver panicked — keep the old solution, penalize the agent
            self._iters_since_improvement += res.num_iterations
            reward = self.FAILURE_PENALTY
        elif candidate.is_feasible() and cand_score < self._prev_score:
            # Solution improved AND is feasible — accept it!
            self._best_solution = candidate
            self._iters_since_improvement = 0
        else:
            # Solution didn't improve or is infeasible — keep the old one
            self._iters_since_improvement += res.num_iterations

        # STEP 4: Compute reward = improvement in competition score
        # Positive reward when score goes DOWN (lower is better)
        new_nv = self._best_solution.num_routes()
        new_td = self._best_solution.distance()
        new_score = competition_score(new_nv, new_td)
        if not fleet_exploded:
            reward = self._prev_score - new_score  # positive = improvement
        self._prev_score = new_score

        # STEP 5: Check if episode is done
        terminated = False                              # Never early-terminate
        truncated = self._step_count >= self.max_steps  # Done after 20 steps

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
        """Run the GNN Encoder on the current instance to produce embeddings.

        This converts raw .vrp problem data into the 128-dim embedding that the
        Fleet Manager uses for decision-making. Called once per episode in reset().

        FEATURE ENGINEERING:
          Each node (depot + customers) gets 3 features:
            - x_norm: x-coordinate normalized to [0, 1]
            - y_norm: y-coordinate normalized to [0, 1]
            - demand/Q: demand divided by vehicle capacity (0 for depot)

          Normalization is critical because:
            - Different instances have different coordinate scales (0-100 vs 0-1000)
            - The GNN needs consistent input ranges to generalize across instances
            - demand/Q tells the network how "heavy" each customer is relative
              to vehicle capacity (0.5 = half a truck, 1.0 = full truck)

        Returns:
            graph_emb: (1, 128) — instance-level summary for the Fleet Manager
            node_emb: (N, 128) — per-node embeddings (not currently used by Fleet Manager)
        """
        data = self._data
        num_locs = data.num_locations  # depot (index 0) + all customers

        # Extract raw coordinates and demands from the parsed .vrp data
        coords = []
        demands = []
        for i in range(num_locs):
            loc = data.location(i)
            coords.append([loc.x, loc.y])
            if i == 0:
                demands.append(0.0)       # Depot has zero demand
            else:
                demands.append(float(data.clients()[i - 1].delivery[0]))

        coords = np.array(coords, dtype=np.float32)
        demands = np.array(demands, dtype=np.float32)

        # Normalize coordinates to [0, 1] range (min-max normalization)
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        c_range = c_max - c_min
        c_range[c_range == 0] = 1.0  # Avoid division by zero if all points are collinear
        coords_norm = (coords - c_min) / c_range

        # Normalize demand by vehicle capacity → fraction of a truckload
        capacity = float(data.vehicle_type(0).capacity[0])
        demand_norm = demands / capacity if capacity > 0 else demands

        # Assemble the 3-feature input tensor: [x_norm, y_norm, demand/Q]
        x = np.column_stack([coords_norm, demand_norm])
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        pos_t = torch.tensor(coords_norm, dtype=torch.float32, device=self.device)
        batch_t = torch.zeros(num_locs, dtype=torch.long, device=self.device)  # Single graph

        # Run the GNN (no gradients needed — the encoder is frozen during RL training)
        self.encoder.eval()
        with torch.no_grad():
            node_emb, graph_emb = self.encoder(x_t, pos_t, batch_t)

        return graph_emb, node_emb  # (1, 128), (N, 128)

    def _build_observation(self) -> np.ndarray:
        """Construct the 132-dim observation vector that the agent sees.

        The observation combines two information sources:
          1. graph_embedding (128 dims) — static spatial understanding of the instance
             (computed once at reset, reused every step)
          2. solver_stats (4 dims) — dynamic real-time feedback about the search progress
             (recomputed every step)

        The 4 solver statistics give the agent situational awareness:

          time_ratio (0→1):
            How far into the episode. 0.0 = just started, 1.0 = last step.
            Helps the agent learn to be aggressive early and polish late.

          nv_ratio (current_NV / initial_NV):
            Fleet reduction progress. 1.0 = no improvement, 0.5 = halved the fleet.
            Lower values mean less room for fleet reduction.

          violation_ratio (excess_load / total_demand):
            Capacity constraint pressure. 0.0 = all routes feasible,
            >0 = some routes are overloaded. High values suggest the agent
            is pushing too hard and should back off.

          stagnation_ratio (iters_no_improve / total_budget):
            How long since the last improvement. High values suggest the
            solver is stuck in a local optimum and the agent should EXPLORE
            (action 4 or 5) rather than keep polishing.
        """
        # Time ratio: what fraction of the episode budget has been used
        time_ratio = self._step_count / self.max_steps

        # NV ratio: current fleet size relative to the starting fleet
        nv_current = self._best_solution.num_routes()
        nv_ratio = nv_current / max(self._nv_initial, 1)

        # Violation ratio: how much demand exceeds vehicle capacities
        excess_loads = self._best_solution.excess_load()
        excess = float(excess_loads[0]) if isinstance(excess_loads, (list, tuple)) else float(excess_loads)
        total_demand = sum(
            c.delivery[0] for c in self._data.clients()
        )
        violation_ratio = excess / max(total_demand, 1.0)

        # Stagnation ratio: iterations without improvement / total budget
        total_budget = self.max_steps * self.iters_per_step
        stagnation_ratio = self._iters_since_improvement / max(total_budget, 1)

        stats = torch.tensor(
            [[time_ratio, nv_ratio, violation_ratio, stagnation_ratio]],
            dtype=torch.float32,
            device=self.device,
        )

        # Concatenate: [graph_embedding (128) | solver_stats (4)] → (1, 132)
        obs = torch.cat([self._graph_embedding, stats], dim=-1)
        return obs.squeeze(0).cpu().numpy()  # Return as numpy for Gymnasium compatibility

    def _action_to_params(self, action: int) -> tuple[SolveParams, int]:
        """Map the agent's discrete action to concrete PyVRP solver parameters.

        This is where the RL agent's decisions become real. Each action translates
        to a combination of:

          penalty_increase: How much to raise the penalty for excess vehicles.
            Higher values = solver is punished more for using "too many" vehicles,
            which pushes it to merge routes. But too high = solver can't find
            feasible solutions and may panic.

          penalty_decrease: How quickly the penalty relaxes when constraints are met.
            Lower values = penalty stays high longer = more sustained pressure.

          target_feasible: What fraction of the population should be feasible.
            Lower values = solver tolerates more infeasible intermediate solutions,
            which allows it to explore harder (useful for finding route merges).

          min/max_penalty: Bounds on the penalty value.

          seed: Random seed for PyVRP's stochastic search.
            Same seed = continue from where we left off (exploitation).
            New seed = start randomization from a fresh point (exploration).

        Returns:
            (params, seed) tuple for the pyvrp.solve() call.
        """
        if action == 0:
            # POLISH: Default parameters, keep the same seed.
            # Use case: routes are already good, just refine distances.
            # The solver continues its search trajectory without disruption.
            return SolveParams(), self._seed

        elif action == 1:
            # MILD_PRESSURE: Small penalty increase (2x).
            # Use case: gently encourage the solver to try solutions with one fewer vehicle.
            # Low risk, low reward — good when NV is close to optimal.
            penalty = PenaltyParams(
                penalty_increase=2.0,      # Mild: 2x multiplier on vehicle penalty
                penalty_decrease=0.5,      # Standard relaxation rate
                target_feasible=0.5,       # Half the population should be feasible
                min_penalty=5.0,
                max_penalty=50_000.0,
            )
            return SolveParams(penalty=penalty), self._seed

        elif action == 2:
            # MODERATE_PRESSURE: Medium penalty increase (5x).
            # Use case: steady, reliable fleet reduction. The workhorse action.
            # Balances exploration of fewer-vehicle solutions with feasibility.
            penalty = PenaltyParams(
                penalty_increase=5.0,      # Medium: 5x multiplier
                penalty_decrease=0.5,
                target_feasible=0.3,       # Allow more infeasible solutions (explores harder)
                min_penalty=10.0,
                max_penalty=100_000.0,
            )
            return SolveParams(penalty=penalty), self._seed

        elif action == 3:
            # AGGRESSIVE_PRESSURE: Heavy penalty increase (10x).
            # Use case: force route merges when the agent is confident NV can be reduced.
            # HIGH RISK: can cause fleet explosion if constraints are too tight.
            # The agent must learn when this is safe vs when it will backfire.
            penalty = PenaltyParams(
                penalty_increase=10.0,     # Heavy: 10x multiplier
                penalty_decrease=0.3,      # Slow relaxation (sustained pressure)
                target_feasible=0.2,       # Only 20% need to be feasible (aggressive)
                min_penalty=20.0,
                max_penalty=200_000.0,
            )
            return SolveParams(penalty=penalty), self._seed

        elif action == 4:
            # EXPLORE_NEW_SEED: Fresh random seed with default parameters.
            # Use case: solver is stuck in a local optimum (high stagnation_ratio).
            # A new seed restarts the randomized search from a different point,
            # potentially finding a completely different solution structure.
            self._seed = random.randint(0, 2**31)
            return SolveParams(), self._seed

        elif action == 5:
            # EXPLORE_PRESSURE: Fresh seed + moderate penalty.
            # Use case: escape a local optimum AND push for fewer vehicles simultaneously.
            # Combines the exploration benefit of a new seed with fleet reduction pressure.
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
