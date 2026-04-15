"""
Inference script: run the trained RL agent on one or more CVRP instances.

This script loads a trained FleetManager checkpoint and runs it greedily
(no exploration — pure argmax) on .vrp instances from the X-dataset. It
reports the RL agent's score (1000*NV + TD) and, optionally, compares it
against a single default HGS solve (the baseline).

===========================================================================
PREREQUISITES
===========================================================================
Run from the repo root with the virtual environment active:

    conda activate cvrp          # or: source venv/bin/activate
    cd /path/to/repo

The default checkpoint path is logs/best_model.pth — this is saved
automatically by train.py whenever the eval score improves.

===========================================================================
BASIC USAGE
===========================================================================

1. Single instance (unseen during validation):
       python -m scripts.infer --instance data/X-n200-k36.vrp

2. All instances in a directory:
       python -m scripts.infer --instance_dir data/

3. Compare RL agent vs HGS baseline (recommended):
       python -m scripts.infer --instance data/X-n200-k36.vrp --baseline

4. Compare all instances vs baseline:
       python -m scripts.infer --instance_dir data/ --baseline

5. See per-step action trace (which action at each of the 50 steps):
       python -m scripts.infer --instance data/X-n200-k36.vrp --verbose

===========================================================================
SWAPPING THE MODEL (how to use a different checkpoint or model class)
===========================================================================

Use a specific checkpoint (e.g., from a mid-training save):
    python -m scripts.infer --instance_dir data/ --model_path checkpoints/epoch_30.pth

To plug in a completely different model architecture:
  1. Import your new class at the top of this file.
  2. Add it to MODEL_REGISTRY:
         MODEL_REGISTRY = {
             "FleetManager": FleetManager,
             "MyNewModel":   MyNewModel,   # <-- add here
         }
  3. Pass it on the CLI:
         python -m scripts.infer --instance_dir data/ \\
             --model_class MyNewModel --model_path logs/my_model.pth

The new class must have the same forward() signature as FleetManager:
    forward(instance_features: Tensor[B,12], solver_stats: Tensor[B,7],
            action_mask: Tensor[B,10] | None) -> (logits: Tensor[B,10], value: Tensor[B,1])

===========================================================================
UNDERSTANDING THE OUTPUT
===========================================================================

Without --baseline:
    Instance                   NV        TD      Score  NV_min    Time
    X-n200-k36                 16     21453      37453      15    42.3s

    - NV:     number of vehicles in the best solution found
    - TD:     total distance traveled
    - Score:  competition objective = 1000*NV + TD  (lower is better)
    - NV_min: theoretical minimum fleet = ceil(total_demand / capacity)

With --baseline, extra columns show the single HGS default solve:
    Delta = RL Score - Baseline Score
    Negative delta = RL agent found a better solution than the baseline.

Validation instances (used during training eval, NOT unseen):
    X-n101-k25, X-n157-k13, X-n223-k34, X-n261-k13, X-n313-k71

All other instances in data/ were never seen during evaluation.

===========================================================================
ALL CLI FLAGS
===========================================================================
    --instance PATH         Single .vrp file to run on
    --instance_dir DIR      Directory of .vrp files (run on all)
    --model_path PATH       Checkpoint file  [default: logs/best_model.pth]
    --model_class NAME      Model class name [default: FleetManager]
    --device cpu|cuda       Torch device     [default: cpu]
    --seed INT              RNG seed for initial HGS solve [default: 42]
    --baseline              Also run single-default HGS for comparison
    --baseline_iters INT    HGS iters for baseline [default: 25000]
    --verbose               Print per-step action/score trace
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np
import torch

# Allow running from the repo root without installing as a package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.agent_manager import FleetManager, INSTANCE_FEATURES_DIM, ACTION_NAMES
from src.solver_engine import CVRPEnv


# ---------------------------------------------------------------------------
# Model registry — add new model classes here to make them selectable via CLI
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type] = {
    "FleetManager": FleetManager,
}


def load_model(
    model_path: str | pathlib.Path,
    model_class: str = "FleetManager",
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Load a trained model from a checkpoint file.

    Args:
        model_path:   Path to the .pth checkpoint saved by train.py.
        model_class:  Name of the model class (must be in MODEL_REGISTRY).
        device:       Torch device to load onto.

    Returns:
        Model in eval mode, ready for inference.
    """
    if model_class not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class '{model_class}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_class]
    model = cls()

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Support multiple checkpoint formats saved by different train.py versions
    if isinstance(checkpoint, dict):
        for key in ("manager_state_dict", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
        else:
            state = checkpoint  # assume it's a plain state_dict
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def solve_instance(
    instance_path: pathlib.Path,
    model: torch.nn.Module,
    device: torch.device,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run the RL agent on a single CVRP instance.

    Creates a one-instance CVRPEnv, resets it, then runs 50 greedy steps
    (argmax action selection — no exploration).

    Returns:
        dict with keys: instance, nv, td, score, nv_min, actions_taken,
                        action_counts, elapsed_sec
    """
    env = CVRPEnv(instance_paths=[instance_path], device=device)
    obs, info = env.reset(seed=seed)

    action_log = []
    t0 = time.time()
    done = False

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        inst_feat = obs_t[:, :INSTANCE_FEATURES_DIM]
        solver_stats = obs_t[:, INSTANCE_FEATURES_DIM:]

        action_mask = info.get("action_mask")
        mask_t = None
        if action_mask is not None:
            mask_t = torch.tensor(
                action_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(inst_feat, solver_stats, action_mask=mask_t)
            action_int = logits.argmax(dim=-1).item()

        obs, _, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated
        action_log.append(action_int)

        if verbose:
            print(
                f"  step {info['step']:>2d}  "
                f"action={ACTION_NAMES[action_int]:<12s}  "
                f"NV={info['nv']}  TD={info['td']:.0f}  "
                f"score={info['score']:.0f}"
                + ("  [EXPLODED]" if info.get("fleet_exploded") else "")
            )

    elapsed = time.time() - t0

    # Count how often each action was used
    action_counts = {name: 0 for name in ACTION_NAMES}
    for a in action_log:
        action_counts[ACTION_NAMES[a]] += 1

    return {
        "instance": instance_path.stem,
        "nv": info["nv"],
        "td": info["td"],
        "score": info["score"],
        "nv_min": info["nv_min"],
        "actions_taken": action_log,
        "action_counts": action_counts,
        "elapsed_sec": elapsed,
    }


def run_baseline(instance_path: pathlib.Path, nb_iter: int, seed: int = 42) -> dict:
    """Single HGS solve with default parameters — the no-RL baseline."""
    import hygese as hgs
    from src.solver_engine import _parse_vrp_file, competition_score

    data = _parse_vrp_file(instance_path)
    params = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=nb_iter, seed=seed)
    solver = hgs.Solver(parameters=params, verbose=False)
    result = solver.solve_cvrp(data, rounding=True)
    nv = len(result.routes)
    td = result.cost
    return {"nv": nv, "td": td, "score": competition_score(nv, td)}


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained RL agent on CVRP instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--instance", type=str,
        help="Path to a single .vrp instance file."
    )
    group.add_argument(
        "--instance_dir", type=str,
        help="Directory of .vrp files — run on all of them."
    )

    # --- Model ---
    parser.add_argument(
        "--model_path", type=str, default="logs/best_model.pth",
        help="Path to the trained model checkpoint (.pth)."
    )
    parser.add_argument(
        "--model_class", type=str, default="FleetManager",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model class to instantiate. Add new classes to MODEL_REGISTRY in this file."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the initial HGS solve."
    )

    # --- Baseline comparison ---
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also run single-default HGS baseline for comparison."
    )
    parser.add_argument(
        "--baseline_iters", type=int, default=25000,
        help="HGS iterations for the baseline solve (default matches RL budget: 50x500)."
    )

    # --- Output ---
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-step action/score trace for each instance."
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load model ---
    model_path = pathlib.Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model checkpoint not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}  (class={args.model_class})")
    model = load_model(model_path, model_class=args.model_class, device=device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # --- Collect instance paths ---
    if args.instance:
        instance_paths = [pathlib.Path(args.instance)]
    else:
        instance_paths = sorted(pathlib.Path(args.instance_dir).glob("*.vrp"))
        if not instance_paths:
            print(f"ERROR: no .vrp files found in {args.instance_dir}")
            sys.exit(1)

    print(f"Running on {len(instance_paths)} instance(s)...\n")

    # --- Print table header ---
    if args.baseline:
        header = (
            f"{'Instance':<25s}  "
            f"{'RL NV':>5s} {'RL TD':>10s} {'RL Score':>10s}  "
            f"{'BL NV':>5s} {'BL TD':>10s} {'BL Score':>10s}  "
            f"{'Delta':>8s}  {'Time':>6s}"
        )
    else:
        header = (
            f"{'Instance':<25s}  "
            f"{'NV':>5s} {'TD':>10s} {'Score':>10s}  "
            f"{'NV_min':>6s}  {'Time':>6s}"
        )

    sep = "-" * len(header)
    print(header)
    print(sep)

    # --- Run ---
    rl_scores = []
    bl_scores = []
    rl_wins = 0
    total_t0 = time.time()

    for inst_path in instance_paths:
        if args.verbose:
            print(f"\n[{inst_path.stem}]")

        result = solve_instance(
            inst_path, model, device, seed=args.seed, verbose=args.verbose
        )

        rl_scores.append(result["score"])
        elapsed_str = f"{result['elapsed_sec']:.1f}s"

        if args.baseline:
            bl = run_baseline(inst_path, nb_iter=args.baseline_iters, seed=args.seed)
            bl_scores.append(bl["score"])
            delta = result["score"] - bl["score"]
            delta_str = f"{delta:+.0f}"
            if delta < 0:
                rl_wins += 1

            print(
                f"{result['instance']:<25s}  "
                f"{result['nv']:>5d} {result['td']:>10.0f} {result['score']:>10.0f}  "
                f"{bl['nv']:>5d} {bl['td']:>10.0f} {bl['score']:>10.0f}  "
                f"{delta_str:>8s}  {elapsed_str:>6s}"
            )
        else:
            print(
                f"{result['instance']:<25s}  "
                f"{result['nv']:>5d} {result['td']:>10.0f} {result['score']:>10.0f}  "
                f"{result['nv_min']:>6d}  {elapsed_str:>6s}"
            )

        # Print action distribution if verbose
        if args.verbose:
            counts = result["action_counts"]
            dist = "  ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
            print(f"  Actions: {dist}")

    # --- Summary ---
    print(sep)
    total_elapsed = time.time() - total_t0
    tm, ts = divmod(int(total_elapsed), 60)

    if args.baseline:
        avg_rl = np.mean(rl_scores)
        avg_bl = np.mean(bl_scores)
        avg_delta = avg_rl - avg_bl
        print(
            f"{'AVERAGE':<25s}  "
            f"{'':>5s} {'':>10s} {avg_rl:>10.0f}  "
            f"{'':>5s} {'':>10s} {avg_bl:>10.0f}  "
            f"{avg_delta:>+8.0f}  {tm}m{ts:02d}s total"
        )
        print()
        print(f"RL wins: {rl_wins}/{len(instance_paths)} instances  "
              f"({'%.1f' % (100*rl_wins/len(instance_paths))}%)")
        if avg_delta < 0:
            print(f"RL is BETTER on average by {abs(avg_delta):.0f} points.")
        else:
            print(f"RL is WORSE on average by {avg_delta:.0f} points.")
    else:
        avg_rl = np.mean(rl_scores)
        print(
            f"{'AVERAGE':<25s}  "
            f"{'':>5s} {'':>10s} {avg_rl:>10.0f}  "
            f"{'':>6s}  {tm}m{ts:02d}s total"
        )


if __name__ == "__main__":
    main()