# Solutions Folder

## Why This Folder Exists

`Solutions/` stores exported route files from baseline and RL-driven runs in competition-style text formats.

This folder exists to keep solver outputs separate from training logs and aggregate reports.

## What This Folder Contains

```text
Solutions/
  HGS/
    sol-Format/
    txt-Format/
  HGS+RL/
    sol-Format/
    txt-Format/
```

- `HGS/`: non-RL baseline exports (mainly from `Baseline/Portfolio_Solver.py`).
- `HGS+RL/`: RL-guided exports (mainly from `Baseline/Infer.py`).

## Current Snapshot Inventory

- `Solutions/HGS/sol-Format`: 59 `.sol` files
- `Solutions/HGS/txt-Format`: 59 `.txt` files
- `Solutions/HGS+RL/sol-Format`: 59 `.sol` files
- `Solutions/HGS+RL/txt-Format`: 59 `.txt` files

## Format Overview

Files are route-line based, for example:

```text
Route #1: 35 46 31
Route #2: 15 22 41 20
...
```

Each line represents one vehicle route as a sequence of customer IDs.

## Typical Producers

- `Baseline/Portfolio_Solver.py` -> `Solutions/HGS/`
- `Baseline/Infer.py` -> `Solutions/HGS+RL/`

Both write to parallel `.sol` and `.txt` structures for compatibility and convenience.

## Why `sol-Format` and `txt-Format` Both Exist

- `.sol` aligns with conventional solution-file naming.
- `.txt` provides plain text duplicate for tooling and quick inspection.

The content is intentionally equivalent.

## Folder Usage in Workflow

1. Run solver/inference scripts.
2. Inspect route exports in this folder.
3. Use `Results/` for aggregate comparisons.
4. Use `Solutions/` when route-level artifact sharing is needed.
