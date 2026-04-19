# Data Folder

## Why This Folder Exists

`Data/` is the canonical source of problem instances used by training, evaluation, baseline runs, and API testing.

Centralizing all `.vrp` files here ensures every workflow uses the same benchmark set.

## What This Folder Contains

- `59` CVRP instances in TSPLIB-like `.vrp` format.
- Current naming pattern: `X-n###-k##.vrp`.

Examples:

- `X-n101-k25.vrp`
- `X-n219-k73.vrp`
- `X-n401-k29.vrp`

## Naming Convention

`X-n###-k##.vrp` typically means:

- `n###`: total nodes in instance (depot + customers),
- `k##`: reference fleet hint from benchmark naming.

The actual best known fleet size can differ from `k` in some instances.

## How This Folder Is Used

- `Model/main.py` reads from this folder for training/evaluation splits.
- `Baseline/Infer.py` reads from this folder for RL inference and RL-vs-baseline reporting.
- `Baseline/Baseline_Evaluation.py` and `Baseline/Portfolio_Solver.py` consume the same files.
- `Backend/Backend.py` consumes uploaded `.vrp` files (not necessarily from this folder) but uses same parser logic.

## Expected Format Highlights

Each `.vrp` file should contain standard sections such as:

- `DIMENSION`
- `CAPACITY`
- `NODE_COORD_SECTION`
- `DEMAND_SECTION`
- `DEPOT_SECTION`
- `EOF`

The parser in `Model/Solver_Engine.py` expects these structures.

## Quick Inventory Check

From repo root:

```bash
ls Data/*.vrp | wc -l
```

Expected in current snapshot: `59`.

## If You Add New Instances

1. Keep `.vrp` extension.
2. Keep TSPLIB-style section formatting.
3. Verify parser compatibility by running:

```bash
python -m Model.main
```

4. For large additions, re-check training/eval split logic and runtime budget assumptions.
