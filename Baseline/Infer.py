"""
Inference script: run the trained RL agent on one or more CVRP instances.

This script loads a trained FleetManager checkpoint and runs it greedily
(no exploration - pure argmax) on .vrp instances from the X-dataset.

When --baseline is enabled, it also produces competition-oriented reporting:
  - tie-aware win/loss/tie outcomes (ties at 3-decimal score precision)
  - pairwise Formula-1 surrogate points (10/8, tie -> 9/9)
  - aggregate metrics (counts, rates, deltas, points)
  - subset robustness via bootstrap simulation
  - CSV/JSON/Markdown artifacts in logs/ by default
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

# Allow running from the repo root without installing as a package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from Model.Agent_Manager import FleetManager, INSTANCE_FEATURES_DIM, ACTION_NAMES
from Model.Solver_Engine import CVRPEnv


# ---------------------------------------------------------------------------
# Model registry - add new model classes here to make them selectable via CLI
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type] = {
    "FleetManager": FleetManager,
}


TIE_DECIMALS = 3
PAIRWISE_POINTS_WINNER = 10.0
PAIRWISE_POINTS_RUNNER_UP = 8.0
PAIRWISE_POINTS_TIE = (PAIRWISE_POINTS_WINNER + PAIRWISE_POINTS_RUNNER_UP) / 2.0


def load_model(
    model_path: str | pathlib.Path,
    model_class: str = "FleetManager",
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Load a trained model from a checkpoint file."""
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
            state = checkpoint  # assume it is already a plain state_dict
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
    """Run the RL agent on a single CVRP instance."""
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
                f"action={ACTION_NAMES[action_int]:<18s}  "
                f"NV={info['nv']}  TD={info['td']:.0f}  "
                f"score={info['score']:.0f}"
                + ("  [EXPLODED]" if info.get("fleet_exploded") else "")
            )

    elapsed = time.time() - t0

    action_counts = {name: 0 for name in ACTION_NAMES}
    for action_id in action_log:
        action_counts[ACTION_NAMES[action_id]] += 1

    return {
        "instance": instance_path.stem,
        "nv": int(info["nv"]),
        "td": float(info["td"]),
        "score": float(info["score"]),
        "nv_min": int(info["nv_min"]),
        "routes": env.get_best_routes(),
        "actions_taken": action_log,
        "action_counts": action_counts,
        "elapsed_sec": float(elapsed),
    }


def run_baseline(instance_path: pathlib.Path, nb_iter: int, seed: int = 42) -> dict:
    """Single HGS solve with default parameters - the no-RL baseline."""
    import hygese as hgs
    from Model.Solver_Engine import _parse_vrp_file, competition_score

    data = _parse_vrp_file(instance_path)
    params = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=nb_iter, seed=seed)
    solver = hgs.Solver(parameters=params, verbose=False)
    result = solver.solve_cvrp(data, rounding=True)
    nv = len(result.routes)
    td = result.cost
    return {
        "nv": int(nv),
        "td": float(td),
        "score": float(competition_score(nv, td)),
        "seed": int(seed),
    }


def run_baseline_multi(
    instance_path: pathlib.Path,
    nb_iter: int,
    seeds: list[int],
    policy: str,
) -> dict:
    """Run baseline with one or multiple seeds and return the selected result.

    policy:
      - single: use the first seed result (default behavior)
      - best: use the best score among all provided seeds
    """
    runs = [run_baseline(instance_path, nb_iter=nb_iter, seed=s) for s in seeds]

    if policy == "single":
        chosen = runs[0]
    elif policy == "best":
        chosen = min(runs, key=lambda record: record["score"])
    else:
        raise ValueError(f"Unsupported baseline policy: {policy}")

    selected = dict(chosen)
    selected["num_runs"] = len(runs)
    return selected


def parse_seed_list(seed_text: str) -> list[int]:
    """Parse comma-separated integer seeds from CLI text."""
    seeds: list[int] = []
    for token in seed_text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Invalid seed '{token}' in --baseline_seeds. "
                "Use comma-separated integers, for example: 42,1042,2042"
            ) from exc

    if not seeds:
        raise ValueError(
            "No valid seeds found in --baseline_seeds. "
            "Provide at least one integer seed."
        )

    return seeds


def baseline_config_signature(nb_iter: int, policy: str, seeds: list[int]) -> str:
    """Stable signature used to keep cache entries consistent."""
    seed_part = ",".join(str(seed) for seed in seeds)
    return f"iters={nb_iter};policy={policy};seeds={seed_part}"


def load_baseline_cache(
    cache_path: pathlib.Path,
    expected_signature: str,
) -> dict[str, dict]:
    """Load baseline rows from CSV cache, filtered by config signature."""
    if not cache_path.exists():
        return {}

    cache: dict[str, dict] = {}
    skipped_signature_mismatch = 0
    accepted_legacy_rows = 0

    with cache_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            instance = (row.get("instance") or "").strip()
            if not instance:
                continue

            row_signature = (row.get("config_signature") or "").strip()
            if row_signature and row_signature != expected_signature:
                skipped_signature_mismatch += 1
                continue
            if not row_signature:
                accepted_legacy_rows += 1

            try:
                nv = int(float(row["nv"]))
                td = float(row["td"])
                score = float(row["score"])
                seed_raw = (row.get("seed") or "").strip()
                seed = int(float(seed_raw)) if seed_raw else None
            except (KeyError, TypeError, ValueError):
                continue

            cache[instance] = {
                "nv": nv,
                "td": td,
                "score": score,
                "seed": seed,
            }

    if skipped_signature_mismatch > 0:
        print(
            f"Skipped {skipped_signature_mismatch} cache row(s) with a different baseline config signature."
        )
    if accepted_legacy_rows > 0:
        print(
            f"Accepted {accepted_legacy_rows} legacy cache row(s) without signature. "
            "Use with care if baseline settings changed."
        )

    return cache


def write_baseline_cache(
    cache_path: pathlib.Path,
    cache: dict[str, dict],
    config_signature: str,
    baseline_iters: int,
    baseline_policy: str,
    baseline_seeds: list[int],
) -> None:
    """Write baseline cache rows for future runs with the same settings."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    seed_text = ",".join(str(seed) for seed in baseline_seeds)

    fieldnames = [
        "instance",
        "nv",
        "td",
        "score",
        "seed",
        "baseline_iters",
        "baseline_policy",
        "baseline_seeds",
        "config_signature",
        "updated_utc",
    ]

    with cache_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for instance in sorted(cache):
            record = cache[instance]
            writer.writerow(
                {
                    "instance": instance,
                    "nv": int(record["nv"]),
                    "td": float(record["td"]),
                    "score": float(record["score"]),
                    "seed": "" if record.get("seed") is None else int(record["seed"]),
                    "baseline_iters": baseline_iters,
                    "baseline_policy": baseline_policy,
                    "baseline_seeds": seed_text,
                    "config_signature": config_signature,
                    "updated_utc": timestamp_utc,
                }
            )


def round_score_for_tie(score: float, decimals: int = TIE_DECIMALS) -> float:
    """Round score using decimal formatting to apply tie-at-precision logic."""
    return float(f"{score:.{decimals}f}")


def classify_pairwise_outcome(
    rl_score: float,
    baseline_score: float,
    tie_decimals: int = TIE_DECIMALS,
) -> dict:
    """Classify win/loss/tie and assign pairwise points for one instance."""
    rl_rounded = round_score_for_tie(rl_score, tie_decimals)
    baseline_rounded = round_score_for_tie(baseline_score, tie_decimals)

    if rl_rounded < baseline_rounded:
        return {
            "outcome": "win",
            "rl_points": PAIRWISE_POINTS_WINNER,
            "bl_points": PAIRWISE_POINTS_RUNNER_UP,
            "rl_score_rounded": rl_rounded,
            "bl_score_rounded": baseline_rounded,
        }
    if rl_rounded > baseline_rounded:
        return {
            "outcome": "loss",
            "rl_points": PAIRWISE_POINTS_RUNNER_UP,
            "bl_points": PAIRWISE_POINTS_WINNER,
            "rl_score_rounded": rl_rounded,
            "bl_score_rounded": baseline_rounded,
        }

    return {
        "outcome": "tie",
        "rl_points": PAIRWISE_POINTS_TIE,
        "bl_points": PAIRWISE_POINTS_TIE,
        "rl_score_rounded": rl_rounded,
        "bl_score_rounded": baseline_rounded,
    }


def build_instance_record(
    rl_result: dict,
    baseline_result: dict,
    baseline_source: str,
    tie_decimals: int = TIE_DECIMALS,
) -> dict:
    """Build a structured per-instance comparison row."""
    delta = float(rl_result["score"] - baseline_result["score"])
    rel_delta_pct = (
        100.0 * delta / baseline_result["score"]
        if baseline_result["score"] != 0
        else math.nan
    )

    pairwise = classify_pairwise_outcome(
        rl_score=float(rl_result["score"]),
        baseline_score=float(baseline_result["score"]),
        tie_decimals=tie_decimals,
    )

    return {
        "instance": rl_result["instance"],
        "rl_nv": int(rl_result["nv"]),
        "rl_td": float(rl_result["td"]),
        "rl_score": float(rl_result["score"]),
        "bl_nv": int(baseline_result["nv"]),
        "bl_td": float(baseline_result["td"]),
        "bl_score": float(baseline_result["score"]),
        "delta": delta,
        "delta_pct": rel_delta_pct,
        "outcome": pairwise["outcome"],
        "rl_points": float(pairwise["rl_points"]),
        "bl_points": float(pairwise["bl_points"]),
        "rl_score_rounded": float(pairwise["rl_score_rounded"]),
        "bl_score_rounded": float(pairwise["bl_score_rounded"]),
        "tie_decimals": int(tie_decimals),
        "baseline_seed": baseline_result.get("seed"),
        "baseline_source": baseline_source,
        "elapsed_sec": float(rl_result["elapsed_sec"]),
    }


def _safe_float(value: float | None) -> float | None:
    """Convert NaN to None for JSON/report friendliness."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def summarize_comparison(records: list[dict]) -> dict:
    """Compute aggregate metrics for structured RL-vs-baseline records."""
    if not records:
        return {}

    deltas = np.array([record["delta"] for record in records], dtype=np.float64)
    rel_deltas = np.array([record["delta_pct"] for record in records], dtype=np.float64)
    rl_scores = np.array([record["rl_score"] for record in records], dtype=np.float64)
    bl_scores = np.array([record["bl_score"] for record in records], dtype=np.float64)
    rl_points = np.array([record["rl_points"] for record in records], dtype=np.float64)
    bl_points = np.array([record["bl_points"] for record in records], dtype=np.float64)

    n_instances = len(records)
    wins = sum(1 for record in records if record["outcome"] == "win")
    losses = sum(1 for record in records if record["outcome"] == "loss")
    ties = sum(1 for record in records if record["outcome"] == "tie")

    valid_rel_deltas = rel_deltas[~np.isnan(rel_deltas)]

    best_idx = int(np.argmin(deltas))
    worst_idx = int(np.argmax(deltas))

    return {
        "num_instances": n_instances,
        "counts": {
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(ties),
        },
        "rates_pct": {
            "wins": 100.0 * wins / n_instances,
            "losses": 100.0 * losses / n_instances,
            "ties": 100.0 * ties / n_instances,
        },
        "scores": {
            "rl_mean": float(np.mean(rl_scores)),
            "bl_mean": float(np.mean(bl_scores)),
        },
        "delta": {
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        },
        "delta_pct": {
            "mean": _safe_float(np.mean(valid_rel_deltas) if valid_rel_deltas.size else None),
            "median": _safe_float(np.median(valid_rel_deltas) if valid_rel_deltas.size else None),
            "min": _safe_float(np.min(valid_rel_deltas) if valid_rel_deltas.size else None),
            "max": _safe_float(np.max(valid_rel_deltas) if valid_rel_deltas.size else None),
        },
        "pairwise_points": {
            "rl_total": float(np.sum(rl_points)),
            "bl_total": float(np.sum(bl_points)),
            "rl_avg_per_instance": float(np.mean(rl_points)),
            "bl_avg_per_instance": float(np.mean(bl_points)),
            "margin_total": float(np.sum(rl_points) - np.sum(bl_points)),
            "margin_avg_per_instance": float(np.mean(rl_points - bl_points)),
        },
        "best_case_for_rl": {
            "instance": records[best_idx]["instance"],
            "delta": float(records[best_idx]["delta"]),
            "delta_pct": _safe_float(records[best_idx]["delta_pct"]),
        },
        "worst_case_for_rl": {
            "instance": records[worst_idx]["instance"],
            "delta": float(records[worst_idx]["delta"]),
            "delta_pct": _safe_float(records[worst_idx]["delta_pct"]),
        },
    }


def bootstrap_subset_points(
    records: list[dict],
    subset_size: int,
    num_trials: int,
    rng_seed: int,
) -> dict:
    """Bootstrap random subsets to estimate hidden-subset robustness."""
    if not records:
        return {}
    if subset_size <= 0:
        raise ValueError("subset_size must be >= 1")
    if num_trials <= 0:
        raise ValueError("num_trials must be >= 1")

    n_instances = len(records)
    rl_points = np.array([record["rl_points"] for record in records], dtype=np.float64)
    bl_points = np.array([record["bl_points"] for record in records], dtype=np.float64)
    deltas = np.array([record["delta"] for record in records], dtype=np.float64)

    rng = np.random.default_rng(rng_seed)
    sample_idx = rng.integers(0, n_instances, size=(num_trials, subset_size))

    sampled_rl_points = rl_points[sample_idx].sum(axis=1)
    sampled_bl_points = bl_points[sample_idx].sum(axis=1)
    sampled_points_margin = sampled_rl_points - sampled_bl_points
    sampled_avg_delta = deltas[sample_idx].mean(axis=1)

    return {
        "subset_size": int(subset_size),
        "num_trials": int(num_trials),
        "rng_seed": int(rng_seed),
        "prob_rl_beats_baseline_points": float(np.mean(sampled_points_margin > 0)),
        "prob_rl_loses_points": float(np.mean(sampled_points_margin < 0)),
        "prob_points_tie": float(np.mean(sampled_points_margin == 0)),
        "expected_points_margin": float(np.mean(sampled_points_margin)),
        "median_points_margin": float(np.median(sampled_points_margin)),
        "p05_points_margin": float(np.percentile(sampled_points_margin, 5)),
        "p95_points_margin": float(np.percentile(sampled_points_margin, 95)),
        "expected_rl_points": float(np.mean(sampled_rl_points)),
        "expected_bl_points": float(np.mean(sampled_bl_points)),
        "prob_avg_delta_lt_zero": float(np.mean(sampled_avg_delta < 0)),
        "expected_avg_delta": float(np.mean(sampled_avg_delta)),
    }


INSTANCE_REPORT_FIELDS = [
    "instance",
    "rl_nv",
    "rl_td",
    "rl_score",
    "bl_nv",
    "bl_td",
    "bl_score",
    "delta",
    "delta_pct",
    "outcome",
    "rl_points",
    "bl_points",
    "rl_score_rounded",
    "bl_score_rounded",
    "tie_decimals",
    "baseline_seed",
    "baseline_source",
    "elapsed_sec",
]


def write_instance_csv(path: pathlib.Path, records: list[dict]) -> None:
    """Write one per-instance row for downstream analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INSTANCE_REPORT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in INSTANCE_REPORT_FIELDS})


def write_summary_json(path: pathlib.Path, payload: dict) -> None:
    """Write aggregate and metadata in a machine-readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_solution_files(
    output_root: pathlib.Path,
    instance_name: str,
    routes: list,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write RL routes to both .sol and .txt formats."""
    sol_dir = output_root / "sol-format"
    txt_dir = output_root / "txt-format"
    sol_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i, route in enumerate(routes):
        customers = " ".join(str(customer) for customer in route)
        lines.append(f"Route #{i + 1}: {customers}")

    content = "\n".join(lines) + ("\n" if lines else "")
    sol_path = sol_dir / f"{instance_name}.sol"
    txt_path = txt_dir / f"{instance_name}.txt"
    sol_path.write_text(content, encoding="utf-8")
    txt_path.write_text(content, encoding="utf-8")
    return sol_path, txt_path


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}%"


def build_markdown_report(
    meta: dict,
    aggregate: dict,
    bootstrap: dict | None,
    records: list[dict],
) -> str:
    """Build a human-readable Markdown summary for quick review."""
    counts = aggregate["counts"]
    rates = aggregate["rates_pct"]
    scores = aggregate["scores"]
    deltas = aggregate["delta"]
    rel_deltas = aggregate["delta_pct"]
    points = aggregate["pairwise_points"]
    best_case = aggregate["best_case_for_rl"]
    worst_case = aggregate["worst_case_for_rl"]

    lines: list[str] = []
    lines.append("# RL vs Baseline Competition-Oriented Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Generated (UTC): {meta['generated_utc']}")
    lines.append(f"- Model: {meta['model_class']} @ {meta['model_path']}")
    lines.append(f"- Device: {meta['device']}")
    lines.append(f"- Instances evaluated: {meta['num_instances']}")
    lines.append(
        "- Baseline config: "
        f"iters={meta['baseline_iters']}, "
        f"policy={meta['baseline_policy']}, "
        f"seeds={meta['baseline_seeds']}"
    )
    lines.append("")

    lines.append("## Head-to-Head Outcomes")
    lines.append(
        f"- Wins/Losses/Ties (tie at {meta['tie_decimals']} decimals): "
        f"{counts['wins']}/{counts['losses']}/{counts['ties']}"
    )
    lines.append(
        f"- Win/Loss/Tie rates: "
        f"{rates['wins']:.1f}% / {rates['losses']:.1f}% / {rates['ties']:.1f}%"
    )
    lines.append(
        f"- Mean score: RL={scores['rl_mean']:.3f}, BL={scores['bl_mean']:.3f}"
    )
    lines.append(
        f"- Delta RL-BL: mean={deltas['mean']:+.3f}, median={deltas['median']:+.3f} "
        "(negative is better for RL)"
    )
    lines.append(
        f"- Relative delta: mean={_fmt_pct(rel_deltas['mean'])}, "
        f"median={_fmt_pct(rel_deltas['median'])}"
    )
    lines.append("")

    lines.append("## Pairwise Formula-1 Surrogate Points")
    lines.append("- Per-instance points: winner=10, runner-up=8, tie=9 each")
    lines.append(
        f"- Total points: RL={points['rl_total']:.1f}, BL={points['bl_total']:.1f}, "
        f"margin={points['margin_total']:+.1f}"
    )
    lines.append(
        f"- Average points per instance: RL={points['rl_avg_per_instance']:.3f}, "
        f"BL={points['bl_avg_per_instance']:.3f}"
    )
    lines.append("")

    lines.append("## Best/Worst Delta Cases")
    lines.append(
        f"- Best for RL: {best_case['instance']} "
        f"(delta={best_case['delta']:+.3f}, rel={_fmt_pct(best_case['delta_pct'])})"
    )
    lines.append(
        f"- Worst for RL: {worst_case['instance']} "
        f"(delta={worst_case['delta']:+.3f}, rel={_fmt_pct(worst_case['delta_pct'])})"
    )
    lines.append("")

    if bootstrap:
        lines.append("## Bootstrap Subset Robustness")
        lines.append(
            f"- Trials={bootstrap['num_trials']}, subset_size={bootstrap['subset_size']}, "
            f"seed={bootstrap['rng_seed']}"
        )
        lines.append(
            f"- P(RL points > BL points) = "
            f"{100.0 * bootstrap['prob_rl_beats_baseline_points']:.1f}%"
        )
        lines.append(
            f"- Expected points margin (RL-BL) = {bootstrap['expected_points_margin']:+.3f}"
        )
        lines.append(
            f"- Points margin p05/p95 = "
            f"{bootstrap['p05_points_margin']:+.3f} / {bootstrap['p95_points_margin']:+.3f}"
        )
        lines.append(
            f"- P(subset mean delta < 0) = "
            f"{100.0 * bootstrap['prob_avg_delta_lt_zero']:.1f}%"
        )
        lines.append("")

    top_improve = sorted(records, key=lambda row: row["delta"])[:5]
    top_regress = sorted(records, key=lambda row: row["delta"], reverse=True)[:5]

    lines.append("## Top 5 RL Improvements")
    lines.append("")
    lines.append("| Instance | Delta (RL-BL) | Delta % | Outcome | Points |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in top_improve:
        lines.append(
            f"| {row['instance']} | {row['delta']:+.3f} | {_fmt_pct(_safe_float(row['delta_pct']))} "
            f"| {row['outcome']} | {row['rl_points']:.0f}-{row['bl_points']:.0f} |"
        )
    lines.append("")

    lines.append("## Top 5 RL Regressions")
    lines.append("")
    lines.append("| Instance | Delta (RL-BL) | Delta % | Outcome | Points |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in top_regress:
        lines.append(
            f"| {row['instance']} | {row['delta']:+.3f} | {_fmt_pct(_safe_float(row['delta_pct']))} "
            f"| {row['outcome']} | {row['rl_points']:.0f}-{row['bl_points']:.0f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the trained RL agent on CVRP instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--instance",
        type=str,
        help="Path to a single .vrp instance file.",
    )
    group.add_argument(
        "--instance_dir",
        type=str,
        help="Directory of .vrp files - run on all of them.",
    )

    # --- Model ---
    parser.add_argument(
        "--model_path",
        type=str,
        default="logs/best_model.pth",
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="FleetManager",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model class to instantiate. Add new classes to MODEL_REGISTRY in this file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the initial HGS solve and default baseline seed.",
    )

    # --- Baseline comparison ---
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run HGS baseline for comparison.",
    )
    parser.add_argument(
        "--baseline_iters",
        type=int,
        default=25000,
        help="HGS iterations per baseline solve.",
    )
    parser.add_argument(
        "--baseline_seeds",
        type=str,
        default="",
        help="Comma-separated baseline seeds. Empty -> uses --seed.",
    )
    parser.add_argument(
        "--baseline_policy",
        type=str,
        default="single",
        choices=["single", "best"],
        help="How to pick baseline result when multiple seeds are provided.",
    )
    parser.add_argument(
        "--baseline_cache_csv",
        type=str,
        default="logs/competition_eval_baseline_cache.csv",
        help="CSV cache path for baseline results.",
    )
    parser.add_argument(
        "--no_baseline_cache",
        action="store_true",
        help="Disable baseline cache read/write.",
    )

    # --- Reporting and robustness ---
    parser.add_argument(
        "--report_dir",
        type=str,
        default="logs",
        help="Directory where comparison artifacts are written.",
    )
    parser.add_argument(
        "--report_stem",
        type=str,
        default="competition_eval",
        help="Output stem for report files.",
    )
    parser.add_argument(
        "--no_reports",
        action="store_true",
        help="Do not write CSV/JSON/Markdown report artifacts.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=20,
        help="Subset size used in bootstrap hidden-subset simulation.",
    )
    parser.add_argument(
        "--bootstrap_trials",
        type=int,
        default=2000,
        help="Number of bootstrap subset trials.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=123,
        help="RNG seed used for bootstrap subset simulation.",
    )
    parser.add_argument(
        "--skip_bootstrap",
        action="store_true",
        help="Skip bootstrap subset robustness simulation.",
    )

    # --- RL solution file export ---
    parser.add_argument(
        "--solution_output_dir",
        type=str,
        default="solutions/HGS+RL",
        help="Root directory for RL solution files (sol-format/ and txt-format).",
    )
    parser.add_argument(
        "--no_solution_files",
        action="store_true",
        help="Disable writing RL .sol and .txt solution files.",
    )

    # --- Verbosity ---
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step action/score trace for each instance.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # Parse baseline seeds early so CLI errors surface before expensive work.
    if args.baseline_seeds:
        try:
            baseline_seeds = parse_seed_list(args.baseline_seeds)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
    else:
        baseline_seeds = [int(args.seed)]

    if args.baseline_policy == "single":
        baseline_seeds = [baseline_seeds[0]]

    baseline_signature = baseline_config_signature(
        nb_iter=args.baseline_iters,
        policy=args.baseline_policy,
        seeds=baseline_seeds,
    )

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

    solution_output_root = pathlib.Path(args.solution_output_dir)
    solution_files_written = 0

    # --- Optional baseline cache ---
    baseline_cache_path = pathlib.Path(args.baseline_cache_csv)
    baseline_cache_enabled = args.baseline and (not args.no_baseline_cache)
    baseline_cache: dict[str, dict] = {}
    baseline_cache_updates = 0

    if baseline_cache_enabled:
        baseline_cache = load_baseline_cache(
            cache_path=baseline_cache_path,
            expected_signature=baseline_signature,
        )
        if baseline_cache:
            print(
                f"Loaded {len(baseline_cache)} baseline cache row(s) from {baseline_cache_path}."
            )

    # --- Print table header ---
    if args.baseline:
        header = (
            f"{'Instance':<25s}  "
            f"{'RL NV':>5s} {'RL TD':>10s} {'RL Score':>10s}  "
            f"{'BL NV':>5s} {'BL TD':>10s} {'BL Score':>10s}  "
            f"{'Delta':>9s} {'Out':>4s} {'Pts':>7s} {'Src':>5s}  {'Time':>6s}"
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

    rl_scores: list[float] = []
    comparison_records: list[dict] = []
    total_t0 = time.time()

    for instance_path in instance_paths:
        if args.verbose:
            print(f"\n[{instance_path.stem}]")

        rl_result = solve_instance(
            instance_path,
            model,
            device,
            seed=args.seed,
            verbose=args.verbose,
        )

        if not args.no_solution_files:
            write_solution_files(
                output_root=solution_output_root,
                instance_name=rl_result["instance"],
                routes=rl_result["routes"],
            )
            solution_files_written += 1

        rl_scores.append(float(rl_result["score"]))
        elapsed_str = f"{rl_result['elapsed_sec']:.1f}s"

        if args.baseline:
            cached = baseline_cache.get(rl_result["instance"]) if baseline_cache_enabled else None

            if cached is not None:
                baseline_result = dict(cached)
                baseline_source = "cache"
            else:
                baseline_result = run_baseline_multi(
                    instance_path=instance_path,
                    nb_iter=args.baseline_iters,
                    seeds=baseline_seeds,
                    policy=args.baseline_policy,
                )
                baseline_source = "solve"

                if baseline_cache_enabled:
                    baseline_cache[rl_result["instance"]] = {
                        "nv": int(baseline_result["nv"]),
                        "td": float(baseline_result["td"]),
                        "score": float(baseline_result["score"]),
                        "seed": baseline_result.get("seed"),
                    }
                    baseline_cache_updates += 1

            row = build_instance_record(
                rl_result=rl_result,
                baseline_result=baseline_result,
                baseline_source=baseline_source,
                tie_decimals=TIE_DECIMALS,
            )
            comparison_records.append(row)

            outcome_short = {"win": "W", "loss": "L", "tie": "T"}[row["outcome"]]
            points_text = f"{row['rl_points']:.0f}-{row['bl_points']:.0f}"

            print(
                f"{row['instance']:<25s}  "
                f"{row['rl_nv']:>5d} {row['rl_td']:>10.0f} {row['rl_score']:>10.0f}  "
                f"{row['bl_nv']:>5d} {row['bl_td']:>10.0f} {row['bl_score']:>10.0f}  "
                f"{row['delta']:>+9.0f} {outcome_short:>4s} {points_text:>7s} {baseline_source:>5s}  {elapsed_str:>6s}"
            )
        else:
            print(
                f"{rl_result['instance']:<25s}  "
                f"{rl_result['nv']:>5d} {rl_result['td']:>10.0f} {rl_result['score']:>10.0f}  "
                f"{rl_result['nv_min']:>6d}  {elapsed_str:>6s}"
            )

        if args.verbose:
            counts = rl_result["action_counts"]
            dist = "  ".join(
                f"{name}={count}" for name, count in counts.items() if count > 0
            )
            print(f"  Actions: {dist}")

    print(sep)
    total_elapsed = time.time() - total_t0
    tm, ts = divmod(int(total_elapsed), 60)

    if args.baseline:
        aggregate = summarize_comparison(comparison_records)

        print(
            f"{'AVERAGE':<25s}  "
            f"{'':>5s} {'':>10s} {aggregate['scores']['rl_mean']:>10.0f}  "
            f"{'':>5s} {'':>10s} {aggregate['scores']['bl_mean']:>10.0f}  "
            f"{aggregate['delta']['mean']:>+9.0f} {'':>4s} {'':>7s} {'':>5s}  {tm}m{ts:02d}s"
        )
        print()

        counts = aggregate["counts"]
        rates = aggregate["rates_pct"]
        points = aggregate["pairwise_points"]
        rel_delta = aggregate["delta_pct"]

        print(
            f"Outcomes (tie at {TIE_DECIMALS} decimals): "
            f"W={counts['wins']} ({rates['wins']:.1f}%), "
            f"L={counts['losses']} ({rates['losses']:.1f}%), "
            f"T={counts['ties']} ({rates['ties']:.1f}%)"
        )
        print(
            "Pairwise points (winner=10, runner-up=8, tie=9 each): "
            f"RL={points['rl_total']:.1f}, BL={points['bl_total']:.1f}, "
            f"margin={points['margin_total']:+.1f}"
        )
        print(
            f"Delta RL-BL (negative means RL better): "
            f"mean={aggregate['delta']['mean']:+.3f}, "
            f"median={aggregate['delta']['median']:+.3f}, "
            f"min={aggregate['delta']['min']:+.3f}, "
            f"max={aggregate['delta']['max']:+.3f}"
        )

        if rel_delta["mean"] is not None:
            print(
                f"Relative delta RL-BL: mean={rel_delta['mean']:+.3f}%, "
                f"median={rel_delta['median']:+.3f}%"
            )

        bootstrap = None
        if not args.skip_bootstrap:
            bootstrap = bootstrap_subset_points(
                records=comparison_records,
                subset_size=args.subset_size,
                num_trials=args.bootstrap_trials,
                rng_seed=args.bootstrap_seed,
            )
            print(
                f"Bootstrap subsets (size={bootstrap['subset_size']}, "
                f"trials={bootstrap['num_trials']}): "
                f"P(RL points > BL points)={100.0 * bootstrap['prob_rl_beats_baseline_points']:.1f}%, "
                f"expected points margin={bootstrap['expected_points_margin']:+.3f}"
            )

        if baseline_cache_enabled and baseline_cache_updates > 0:
            write_baseline_cache(
                cache_path=baseline_cache_path,
                cache=baseline_cache,
                config_signature=baseline_signature,
                baseline_iters=args.baseline_iters,
                baseline_policy=args.baseline_policy,
                baseline_seeds=baseline_seeds,
            )
            print(
                f"Baseline cache updated: {baseline_cache_updates} new row(s) -> {baseline_cache_path}"
            )

        if not args.no_reports:
            report_dir = pathlib.Path(args.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            instances_csv = report_dir / f"{args.report_stem}_instances.csv"
            summary_json = report_dir / f"{args.report_stem}_summary.json"
            report_md = report_dir / f"{args.report_stem}_report.md"

            meta = {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(model_path),
                "model_class": args.model_class,
                "device": str(device),
                "num_instances": len(instance_paths),
                "instance_scope": args.instance if args.instance else args.instance_dir,
                "seed": args.seed,
                "tie_decimals": TIE_DECIMALS,
                "baseline_iters": args.baseline_iters,
                "baseline_policy": args.baseline_policy,
                "baseline_seeds": baseline_seeds,
                "baseline_cache_csv": str(baseline_cache_path) if baseline_cache_enabled else None,
            }

            payload = {
                "meta": meta,
                "aggregate": aggregate,
                "bootstrap": bootstrap,
            }

            write_instance_csv(instances_csv, comparison_records)
            write_summary_json(summary_json, payload)
            report_md.write_text(
                build_markdown_report(meta, aggregate, bootstrap, comparison_records),
                encoding="utf-8",
            )

            print("Reports written:")
            print(f"  - {instances_csv}")
            print(f"  - {summary_json}")
            print(f"  - {report_md}")

        if not args.no_solution_files:
            print(
                "RL solution files written: "
                f"{solution_files_written} instance(s) -> "
                f"{solution_output_root / 'sol-format'} and {solution_output_root / 'txt-format'}"
            )
    else:
        avg_rl = np.mean(rl_scores)
        print(
            f"{'AVERAGE':<25s}  "
            f"{'':>5s} {'':>10s} {avg_rl:>10.0f}  "
            f"{'':>6s}  {tm}m{ts:02d}s total"
        )

        if not args.no_solution_files:
            print(
                "RL solution files written: "
                f"{solution_files_written} instance(s) -> "
                f"{solution_output_root / 'sol-format'} and {solution_output_root / 'txt-format'}"
            )


if __name__ == "__main__":
    main()
