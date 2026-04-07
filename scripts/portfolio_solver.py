"""
Portfolio Solver: Run HGS-CVRP with multiple configs × multiple seeds per instance.

This is a non-RL baseline that systematically tries different HGS parameter
configurations and random seeds, keeping the best result per instance.
It guarantees competition-ready solutions without requiring a trained RL agent.

The portfolio approach works because:
  - Different configs occasionally find different NV (fleet sizes)
  - Different seeds explore different regions of the search space
  - Best-of-many eliminates unlucky solver runs

Usage:
    python scripts/portfolio_solver.py --instance_path data/
    python scripts/portfolio_solver.py --instance_path data/ --nb_iter 10000 --num_seeds 5
    python scripts/portfolio_solver.py --instance_path data/ --output_dir solutions/
"""

import argparse
import math
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import hygese as hgs


def competition_score(nv: int, td: float) -> float:
    """Competition objective: 1000 * NV + TD (lower is better)."""
    return 1000.0 * nv + td


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


# ---------------------------------------------------------------------------
# Parameter Configurations (portfolio of HGS strategies)
# ---------------------------------------------------------------------------
# Each config varies four main HGS knobs:
#   mu             — Min population size (larger = more diversity, slower per iter)
#   lambda_        — Max population size (controls how many solutions HGS keeps)
#   nbGranular     — Neighbourhood size for local search (larger = broader moves)
#   targetFeasible — Target fraction of feasible solutions in population
#   nbElite        — Number of elite solutions to preserve between generations
#   nbClose        — Number of "close" solutions considered in crossover
#
# The key insight: different configs work better on different instance types.
# DEFAULT is HGS's hand-tuned baseline; the others explore different trade-offs
# between population diversity, search depth, and feasibility pressure.
CONFIGS = {
    "DEFAULT":          {},  # HGS hand-tuned defaults — the baseline to beat
    "FAST_AGGRESSIVE":  dict(mu=15, lambda_=20,  nbGranular=15, targetFeasible=0.1,  nbElite=2,  nbClose=3),
    "LARGE_DIVERSE":    dict(mu=40, lambda_=60,  nbGranular=30, targetFeasible=0.3,  nbElite=6,  nbClose=8),
    "DEEP_SEARCH":      dict(mu=25, lambda_=40,  nbGranular=40, targetFeasible=0.2,  nbElite=4,  nbClose=5),
    "HIGH_TURNOVER":    dict(mu=10, lambda_=80,  nbGranular=20, targetFeasible=0.05, nbElite=2,  nbClose=3),
    "STABLE_ELITE":     dict(mu=50, lambda_=30,  nbGranular=25, targetFeasible=0.4,  nbElite=8,  nbClose=7),
    "LARGE_POP":        dict(mu=50, lambda_=80,  nbGranular=30, targetFeasible=0.2,  nbElite=6,  nbClose=7),
    "TINY_AGGRESSIVE":  dict(mu=5,  lambda_=15,  nbGranular=10, targetFeasible=0.05, nbElite=1,  nbClose=2),
    "WIDE_SEARCH":      dict(mu=30, lambda_=50,  nbGranular=50, targetFeasible=0.25, nbElite=5,  nbClose=6),
    "CONSERVATIVE":     dict(mu=60, lambda_=20,  nbGranular=20, targetFeasible=0.5,  nbElite=10, nbClose=8),
    "BALANCED_EXPLORE": dict(mu=20, lambda_=60,  nbGranular=25, targetFeasible=0.15, nbElite=3,  nbClose=4),
    "MAX_GRANULAR":     dict(mu=25, lambda_=40,  nbGranular=60, targetFeasible=0.2,  nbElite=4,  nbClose=5),
}


def solve_one(instance_path: str, config_name: str, config_params: dict,
              nb_iter: int, seed: int) -> dict:
    """Run a single HGS solve. Designed to be called in a process pool.

    Each call is one (instance, config, seed) combination. The process pool
    runs many of these in parallel across all available CPU cores.
    """
    path = pathlib.Path(instance_path)
    data = parse_vrp_file(path)

    params = hgs.AlgorithmParameters(
        timeLimit=0.0,
        nbIter=nb_iter,
        seed=seed,
        **config_params,
    )
    solver = hgs.Solver(parameters=params, verbose=False)
    result = solver.solve_cvrp(data, rounding=True)

    nv = len(result.routes)
    td = result.cost
    return {
        "instance": path.stem,
        "config": config_name,
        "seed": seed,
        "nv": nv,
        "td": td,
        "score": competition_score(nv, td),
        "routes": result.routes,
    }


def write_solution(output_dir: pathlib.Path, instance_name: str, routes: list):
    """Write solution in CVRPLIB format (DIMACS convention)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sol_path = output_dir / f"{instance_name}.sol"
    with open(sol_path, "w") as f:
        for i, route in enumerate(routes):
            # Routes in hygese are 0-indexed customer IDs
            # CVRPLIB format expects 1-indexed
            customers = " ".join(str(c + 1) for c in route)
            f.write(f"Route #{i+1}: {customers}\n")
    return sol_path


def main():
    parser = argparse.ArgumentParser(description="Portfolio Solver for ML4VRP")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Directory containing .vrp files")
    parser.add_argument("--nb_iter", type=int, default=10000,
                        help="HGS iterations per solve")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of random seeds per config")
    parser.add_argument("--output_dir", type=str, default="solutions",
                        help="Directory to write solution files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    instance_dir = pathlib.Path(args.instance_path)
    instance_paths = sorted(instance_dir.glob("*.vrp"))
    assert len(instance_paths) > 0, f"No .vrp files in {args.instance_path}"

    output_dir = pathlib.Path(args.output_dir)
    num_configs = len(CONFIGS)
    total_solves = len(instance_paths) * num_configs * args.num_seeds

    print(f"Portfolio Solver")
    print(f"  Instances: {len(instance_paths)}")
    print(f"  Configs: {num_configs} ({', '.join(CONFIGS.keys())})")
    print(f"  Seeds per config: {args.num_seeds}")
    print(f"  Iterations per solve: {args.nb_iter}")
    print(f"  Total solves: {total_solves}")
    print(f"  Workers: {args.workers}")
    print()

    # Submit all (instance × config × seed) combinations to the process pool.
    # Each future is one HGS solve. We collect results as they complete and
    # track only the best score per instance.
    best_per_instance = {}  # instance_name -> best result dict so far
    start_time = time.time()
    completed = 0

    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for inst_path in instance_paths:
            for config_name, config_params in CONFIGS.items():
                for seed_idx in range(args.num_seeds):
                    seed = 42 + seed_idx * 1000  # Deterministic seed sequence
                    fut = pool.submit(
                        solve_one, str(inst_path), config_name, config_params,
                        args.nb_iter, seed
                    )
                    futures.append(fut)

        for fut in as_completed(futures):
            result = fut.result()
            completed += 1
            inst = result["instance"]

            # Keep only the best-scoring result for each instance
            if inst not in best_per_instance or result["score"] < best_per_instance[inst]["score"]:
                best_per_instance[inst] = result

            if completed % 50 == 0 or completed == total_solves:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_solves - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total_solves}] {rate:.1f} solves/sec, ETA: {eta/60:.0f}m")

    # Print results
    elapsed = time.time() - start_time
    print()
    print("=" * 100)
    print("PORTFOLIO RESULTS")
    print("=" * 100)
    print(f"{'Instance':<25s} {'NV':>5s} {'TD':>10s} {'Score':>10s}   {'Best Config':<25s} {'Seed':>5s}")
    print("-" * 100)

    total_score = 0.0
    for inst_path in instance_paths:
        name = inst_path.stem
        if name in best_per_instance:
            r = best_per_instance[name]
            print(f"{name:<25s} {r['nv']:>5d} {r['td']:>10.0f} {r['score']:>10.0f}   {r['config']:<25s} {r['seed']:>5d}")
            total_score += r["score"]

            # Write solution file
            sol_path = write_solution(output_dir, name, r["routes"])

    print("-" * 100)
    avg_score = total_score / len(best_per_instance) if best_per_instance else 0
    print(f"{'AVERAGE':<25s} {'':>5s} {'':>10s} {avg_score:>10.0f}")
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Solutions written to: {output_dir}/")


if __name__ == "__main__":
    main()
