# Model Folder

## Why This Folder Exists

`Model/` is the core algorithmic engine of the repository.

It exists to hold all trainable and environment logic in one place:

- policy network,
- environment wrapper around HGS-CVRP,
- PPO training implementation,
- command-line entrypoint for smoke tests and training.

## What This Folder Contains

```text
Model/
  __init__.py
  Agent_Manager.py
  Solver_Engine.py
  Train.py
  main.py
```

## File-by-File Explanation

## 1) `Agent_Manager.py`

Purpose:

- defines `FleetManager`, the actor-critic policy network.

Core behavior:

- observation split: 12 instance features + 7 solver stats,
- 10-action discrete policy head,
- value head for PPO critic,
- action masking support for impossible fleet-reduction moves.

## 2) `Solver_Engine.py`

Purpose:

- defines `CVRPEnv`, a Gymnasium environment that wraps HGS.

Core behavior:

- parses `.vrp` data,
- computes 12 hand-crafted instance features,
- maps each action to HGS parameters/fleet target/seed strategy,
- runs fresh HGS solve per step,
- tracks best solution across an episode,
- computes reward using percentage improvement over episode-best score,
- guards against failed/infeasible constrained solves and fleet explosion patterns.

## 3) `Train.py`

Purpose:

- PPO training loop implementation (`MARLTrainer`).

Core behavior:

- rollout collection,
- GAE computation,
- PPO update (clipped objective + entropy bonus),
- optional holdout evaluation,
- best-model selection by `eval`, `holdout`, or `composite`,
- logging to `Logs/Training_Metrics.csv`,
- run metadata to `Logs/Run_Configuration.json`,
- best checkpoint to `Logs/Best_Model.pth`.

## 4) `main.py`

Purpose:

- CLI entrypoint for smoke tests and training.

Modes:

- no subcommand: runs smoke-test suite,
- `train` subcommand: full PPO training loop.

Also handles:

- curriculum switch,
- resume from checkpoint,
- eval/holdout split selection,
- periodic/final checkpoint writing.

## 5) `__init__.py`

- package marker (currently empty).

## Running Mechanism from This Folder

1. `main.py` loads instance paths.
2. `CVRPEnv` handles reset/step and solver interaction.
3. `FleetManager` selects actions each step.
4. `MARLTrainer` aggregates rollouts and updates policy.
5. Artifacts are written to `Logs/` and `Checkpoints/`.

## Observation, Action, and Reward Summary

Observation (`19` dimensions):

- `12` instance features,
- `7` solver progress stats.

Action space (`10` actions):

- free/lock/push/force strategies,
- same/new seed variants,
- three search-bias variants:
  - `FREE_DIVERSE_NEW`,
  - `LOCK_AGGR_NEW`,
  - `PUSH_BALANCED_NEW`.

Reward:

- positive when a new episode-best score is found (`pct * 100`),
- small negative for no improvement,
- stronger negative for failed/infeasible/aggressive bad candidates.

## Important Training Defaults

From `Model/main.py` and `Model/Train.py`:

- epochs: `50`
- episodes per epoch: `8`
- max steps per episode: `50`
- PPO epochs: `3`
- mini-batch: `128`
- learning rate: `1e-4` with linear decay
- gamma/lambda: `0.95 / 0.90`
- clip epsilon: `0.2`
- target KL: `0.015`
- reward clip range: `[-10, 10]`

Curriculum default:

- start with `N <= 100` for first `20` epochs,
- then unlock all available instances.

## Commands

## Smoke tests

```bash
python -m Model.main
```

## Training

```bash
python -m Model.main train \
  --instance_path Data \
  --epochs 50 \
  --checkpoint_dir Checkpoints \
  --log_dir Logs
```

## Resume

```bash
python -m Model.main train \
  --instance_path Data \
  --resume Checkpoints/Checkpoint_Epoch40.pth \
  --start_epoch 41 \
  --checkpoint_dir Checkpoints \
  --log_dir Logs
```

## Why This Folder Is Separate from `Baseline/` and `Backend/`

- `Model/` focuses on algorithm/training internals.
- `Baseline/` focuses on evaluation and benchmark scripts.
- `Backend/` focuses on API serving and job orchestration.

This separation reduces coupling and makes debugging and experimentation easier.
