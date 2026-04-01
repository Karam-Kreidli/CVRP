"""
Baseline evaluation: Run HGS-CVRP with DEFAULT parameters on the same
5 eval instances and same iteration budget as the RL agent.

This script provides the baseline scores that the RL agent must beat to
demonstrate that learned parameter tuning outperforms the genetic algorithm's
hand-tuned defaults.

Two baselines are computed per instance:
  1. DEFAULT — HGS with all default AlgorithmParameters
  2. LARGE_POP — HGS with larger population (mu=50, lambda_=80) to show
     that simply increasing compute isn't enough; you need the RIGHT parameters.

Usage:
    python scripts/baseline_eval.py --instance_path data/
    python scripts/baseline_eval.py --instance_path data/ --nb_iter 5000 --num_seeds 5
"""

import argparse
import math
import pathlib

import numpy as np
import hygese as hgs


def competition_score(nv: int, td: float) -> float:
    """Competition objective: 1000 * NV + TD (lower is better)."""
    return 1000.0 * nv + td


def select_eval_instances(instance_paths: list[pathlib.Path], n: int = 5) -> list[pathlib.Path]:
    """Select n evenly-spaced instances for evaluation (same logic as main.py)."""
    sorted_paths = sorted(instance_paths, key=lambda p: p.stem)
    step = max(1, len(sorted_paths) // n)
    return [sorted_paths[i] for i in range(0, len(sorted_paths), step)][:n]


def parse_vrp_file(path: pathlib.Path) -> dict:
    """Parse a .vrp file into the dict format expected by HGS."""
    lines = path.read_text().splitlines()
    section = None
    dimension = 0
    capacity = 0
    coords = {}
    demands = {}

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
        elif line.startswith(("NAME", "COMMENT", "TYPE", "EDGE_WEIGHT_TYPE")):
            continue

        if section == "coord":
            parts = line.split()
            coords[int(parts[0])] = (int(parts[1]), int(parts[2]))
        elif section == "demand":
            parts = line.split()
            demands[int(parts[0])] = int(parts[1])

    n = dimension
    x_coords = np.zeros(n, dtype=np.float64)
    y_coords = np.zeros(n, dtype=np.float64)
    demand_arr = np.zeros(n, dtype=np.float64)

    for i, node_id in enumerate(sorted(coords.keys())):
        x_coords[i] = coords[node_id][0]
        y_coords[i] = coords[node_id][1]
        demand_arr[i] = demands.get(node_id, 0)

    total_demand = demand_arr.sum()
    nv_upper = max(1, int(math.ceil(total_demand / capacity))) * 2

    return {
        "x_coordinates": x_coords,
        "y_coordinates": y_coords,
        "demands": demand_arr,
        "vehicle_capacity": capacity,
        "num_vehicles": nv_upper,
        "depot": 0,
        "service_times": np.zeros(n, dtype=np.float64),
    }


def run_default_baseline(data: dict, nb_iter: int, seed: int) -> dict:
    """Run HGS with default parameters."""
    params = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=nb_iter, seed=seed)
    solver = hgs.Solver(parameters=params, verbose=False)
    result = solver.solve_cvrp(data, rounding=True)
    nv = len(result.routes)
    td = result.cost
    return {"nv": nv, "td": td, "score": competition_score(nv, td)}


def run_large_pop_baseline(data: dict, nb_iter: int, seed: int) -> dict:
    """Run HGS with larger population — tests if more compute alone helps."""
    params = hgs.AlgorithmParameters(
        timeLimit=0.0,
        nbIter=nb_iter,
        seed=seed,
        mu=50,
        lambda_=80,
        nbGranular=30,
        targetFeasible=0.2,
        nbElite=6,
        nbClose=7,
    )
    solver = hgs.Solver(parameters=params, verbose=False)
    result = solver.solve_cvrp(data, rounding=True)
    nv = len(result.routes)
    td = result.cost
    return {"nv": nv, "td": td, "score": competition_score(nv, td)}


def main():
    parser = argparse.ArgumentParser(description="HGS-CVRP Baseline Evaluation")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Directory containing .vrp files")
    parser.add_argument("--nb_iter", type=int, default=5000,
                        help="HGS iterations per solve (should match RL iters_per_step)")
    parser.add_argument("--num_seeds", type=int, default=3,
                        help="Number of random seeds to average over")
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Number of solves per instance (to match RL episode steps)")
    parser.add_argument("--all", action="store_true",
                        help="Run on ALL instances, not just the 5 eval ones")
    args = parser.parse_args()

    instance_dir = pathlib.Path(args.instance_path)
    instance_paths = sorted(instance_dir.glob("*.vrp"))
    assert len(instance_paths) > 0, f"No .vrp files in {args.instance_path}"

    if args.all:
        eval_instances = instance_paths
    else:
        eval_instances = select_eval_instances(instance_paths)

    print(f"Eval instances: {[p.stem for p in eval_instances]}")
    print(f"Config: nb_iter={args.nb_iter}, num_seeds={args.num_seeds}, num_steps={args.num_steps}")
    print(f"Total HGS iterations per instance per seed: {args.nb_iter * args.num_steps}")
    print()

    header = (
        f"{'Instance':<25s} "
        f"{'--- DEFAULT ---':^30s}   "
        f"{'--- LARGE POP ---':^30s}"
    )
    sub_header = (
        f"{'':25s} "
        f"{'NV':>5s} {'TD':>10s} {'Score':>10s}   "
        f"{'NV':>5s} {'TD':>10s} {'Score':>10s}"
    )
    sep = "-" * 95

    print("=" * 95)
    print(header)
    print(sub_header)
    print(sep)

    all_default_scores = []
    all_large_scores = []

    for inst_path in eval_instances:
        data = parse_vrp_file(inst_path)

        # Run multiple seeds, each with num_steps solves, keep the best
        best_default = {"score": float("inf")}
        best_large = {"score": float("inf")}

        for s in range(args.num_seeds):
            seed = 42 + s * 1000

            for step in range(args.num_steps):
                step_seed = seed + step
                res = run_default_baseline(data, args.nb_iter, step_seed)
                if res["score"] < best_default["score"]:
                    best_default = res

            for step in range(args.num_steps):
                step_seed = seed + step
                res = run_large_pop_baseline(data, args.nb_iter, step_seed)
                if res["score"] < best_large["score"]:
                    best_large = res

        all_default_scores.append(best_default["score"])
        all_large_scores.append(best_large["score"])

        print(
            f"{inst_path.stem:<25s} "
            f"{best_default['nv']:>5d} {best_default['td']:>10.0f} {best_default['score']:>10.0f}   "
            f"{best_large['nv']:>5d} {best_large['td']:>10.0f} {best_large['score']:>10.0f}"
        )

    print(sep)
    print(
        f"{'AVERAGE':<25s} "
        f"{'':>5s} {'':>10s} {np.mean(all_default_scores):>10.0f}   "
        f"{'':>5s} {'':>10s} {np.mean(all_large_scores):>10.0f}"
    )
    print("=" * 95)
    print()
    print("Compare these averages against your RL agent's Eval score.")
    print("If RL Eval < Default Average, the RL agent is adding value.")


if __name__ == "__main__":
    main()
