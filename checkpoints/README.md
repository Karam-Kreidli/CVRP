# Checkpoints Folder

## Why This Folder Exists

`Checkpoints/` stores full training-state snapshots so interrupted runs can be resumed exactly.

This folder exists for fault tolerance and long-running experiment continuity.

## What This Folder Contains (Current Snapshot)

```text
Checkpoints/
  Checkpoint_Epoch10.pth
  Checkpoint_Epoch20.pth
  Checkpoint_Epoch30.pth
  Checkpoint_Epoch40.pth
  Checkpoint_Epoch50.pth
  Checkpoint_Epoch60.pth
  Checkpoint_Epoch70.pth
  Checkpoint_Epoch80.pth
  Checkpoint_Final.pth
```

Current count in snapshot: `9` checkpoint files.

## Checkpoint Naming Convention

- Periodic checkpoint: `Checkpoint_Epoch{N}.pth`
- Final checkpoint: `Checkpoint_Final.pth`

## What a Checkpoint Includes

Saved by `Model/Train.py` (`save_checkpoint`):

- policy weights (`manager_state_dict`),
- optimizer state,
- scheduler state,
- AMP scaler state,
- epoch stats history,
- best tracking score state.

## Resume Mechanism

Use these files with `Model/main.py`:

```bash
python -m Model.main train \
  --instance_path Data \
  --resume Checkpoints/Checkpoint_Epoch40.pth \
  --start_epoch 41 \
  --checkpoint_dir Checkpoints \
  --log_dir Logs
```

## Relationship to Other Folders

- `Logs/Best_Model.pth`: best checkpoint for inference.
- `Checkpoints/*.pth`: resume snapshots for training continuation.

Both may point to different epochs depending on selection metric and training dynamics.

## Folder Maintenance Notes

- Keep at least one periodic and one final checkpoint for recovery safety.
- Remove stale checkpoints only after confirming best/final artifacts are preserved elsewhere.
