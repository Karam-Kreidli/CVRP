# Baseline Folder

## Why This Folder Exists

`Baseline/` exists to provide reproducible evaluation and comparison workflows independent of training.

It answers three practical needs:

- Evaluate RL policy outputs against non-RL references.
- Produce strong pure-HGS baselines.
- Export standardized reports and route artifacts for analysis.

## What This Folder Contains

```text
Baseline/
  Baseline_Evaluation.py
  Infer.py
  Portfolio_Solver.py
  Setup_VM.sh
```

## File-by-File Explanation

## 1) `Infer.py`

Purpose:

- run trained RL policy on one instance or a full directory,
- optionally run HGS baseline side-by-side,
- compute pairwise outcome metrics,
- write machine-readable and human-readable reports,
- export RL route files.

Key capabilities:

- single instance (`--instance`) or batch (`--instance_dir`),
- baseline on/off (`--baseline`),
- baseline seed policy (`single` or `best`),
- baseline cache with config signature,
- tie-aware scoring at 3 decimals,
- bootstrap subset robustness,
- artifact export to `Results/` and `Solutions/HGS+RL/`.

Default outputs when baseline mode is enabled:

- `Results/Competition_Eval_Instances.csv`
- `Results/Competition_Eval_Summary.json`
- `Results/Competition_Eval_Report.md`
- `Solutions/HGS+RL/sol-Format/*.sol`
- `Solutions/HGS+RL/txt-Format/*.txt`

## 2) `Baseline_Evaluation.py`

Purpose:

- evaluate classic HGS baselines without RL,
- print fair single-run comparison and upper-bound best-of-N comparison.

Modes in one script:

- Single-solve baseline:
  - default HGS and large-pop HGS,
  - one solve per instance.
- Best-of-N baseline:
  - configurable `num_seeds x num_steps`,
  - keeps best score found.

Useful for validating whether RL adds value beyond hand-tuned defaults.

## 3) `Portfolio_Solver.py`

Purpose:

- run a strong deterministic non-RL portfolio baseline.

Mechanism:

- tries 11 predefined HGS parameter configurations,
- runs each config across N seeds,
- keeps best result per instance,
- writes route files to `Solutions/HGS/` in both `.sol` and `.txt` formats.

Default profile:

- `nb_iter=10000`
- `num_seeds=5`
- `workers=4`

## 4) `Setup_VM.sh`

Purpose:

- helper script for GPU VM setup and quick bootstrap.

Important note:

- treat it as a setup template; it contains some legacy path strings and should be adjusted to current folder names/workflow when used.

## Running Mechanism in This Folder

Typical sequence:

1. Ensure `Logs/Best_Model.pth` exists.
2. Run `Infer.py` on Data for RL-only or RL-vs-baseline.
3. Run `Baseline_Evaluation.py` for scripted baseline tables.
4. Run `Portfolio_Solver.py` for strongest pure-HGS baseline set.

## Common Commands

## RL only

```bash
python Baseline/Infer.py \
  --instance_dir Data \
  --model_path Logs/Best_Model.pth \
  --solution_output_dir Solutions/HGS+RL
```

## RL vs baseline with reports

```bash
python Baseline/Infer.py \
  --instance_dir Data \
  --model_path Logs/Best_Model.pth \
  --baseline \
  --baseline_iters 25000 \
  --baseline_cache_csv Results/Competition_Eval_Baseline_Cache.csv \
  --report_dir Results \
  --report_stem Competition_Eval \
  --solution_output_dir Solutions/HGS+RL
```

## Default and large-pop baseline tables

```bash
python Baseline/Baseline_Evaluation.py --instance_path Data
```

## Portfolio baseline

```bash
python Baseline/Portfolio_Solver.py \
  --instance_path Data \
  --nb_iter 10000 \
  --num_seeds 5 \
  --workers 4 \
  --output_dir Solutions/HGS
```

## Why This Folder Is Separate from `Model/`

Separation keeps concerns clean:

- `Model/` focuses on training and RL internals.
- `Baseline/` focuses on evaluation protocols, benchmarking, and exported comparison artifacts.

This makes experiments easier to reason about and avoids mixing training logic with report-generation logic.
