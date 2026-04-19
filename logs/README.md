# Logs Folder

## Why This Folder Exists

`Logs/` stores persistent training artifacts that describe model state, run configuration, and learning dynamics over time.

It exists so training is inspectable, reproducible, and resumable with context.

## What This Folder Contains (Current Snapshot)

```text
Logs/
  Best_Model.pth
  Run_Configuration.json
  Training_Metrics.csv
```

## File Semantics

## 1) `Best_Model.pth`

- Best checkpoint selected by configured tracking metric in training.
- Saved by `Model/Train.py` whenever tracking score improves.
- Commonly used for inference:

```bash
python Baseline/Infer.py --instance_dir Data --model_path Logs/Best_Model.pth
```

## 2) `Run_Configuration.json`

- Captures PPO config and run metadata.
- Includes selected eval/holdout instances and best-model metric mode.
- Useful for reproducibility and experiment bookkeeping.

## 3) `Training_Metrics.csv`

- Epoch-level metrics emitted by trainer.
- Includes rollout stats, PPO losses, eval/holdout scores, and selection metric fields.

Important column groups include:

- rollout: `avg_nv`, `avg_td`, `avg_score`, `best_score`, `total_reward`, `total_steps`
- eval: `eval_score`, `eval_nv`, `eval_td`, `eval_instances_count`
- holdout: `holdout_eval_score`, `holdout_eval_nv`, `holdout_eval_td`, `holdout_instances_count`, `holdout_eval_interval`
- PPO/loss: `policy_loss`, `value_loss`, `entropy`
- optimizer: `lr`, `manager_lr_init`
- hyperparams snapshot: `gamma`, `lam`, `epsilon_clip`, `vf_coeff`, `ent_coeff`, `ppo_epochs`, `mini_batch_size`, `max_grad_norm`, `target_kl`
- reward controls: `reward_clip_min`, `reward_clip_max`, `failure_penalty`, `no_improvement_penalty`
- best-model tracking: `best_model_metric`, `selection_metric`, `tracking_score`
- metadata: `run_tag`

## How This Folder Is Produced

Produced by `Model/Train.py` called through `Model/main.py train`.

Typical command:

```bash
python -m Model.main train --instance_path Data --log_dir Logs
```

## Why This Folder Is Separate from `Checkpoints/`

- `Logs/` stores best-model + analytics/time-series.
- `Checkpoints/` stores periodic/final full-state snapshots for resume.

This separation keeps operational inference assets (`Best_Model.pth`) and recovery assets (`Checkpoint_Epoch*.pth`) conceptually clean.
