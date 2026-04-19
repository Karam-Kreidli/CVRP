"""
ML4VRP 2026 - RL-Guided CVRP Solver
Primary entry point: training loop and smoke tests.

This file has two modes:
  1. No arguments → runs smoke tests to verify the pipeline works end-to-end
  2. "train" subcommand → runs the full PPO training loop

SMOKE TESTS verify each stage of the pipeline independently:
  - Stage 2: Fleet Manager (forward pass, action sampling, FP16)
  - Stage 3: CVRPEnv (HGS-CVRP wrapper with real solver)
  - Stage 5: Training loop (2 PPO iterations on synthetic data)
  - Stage 6: Action masking (NV_min calculation and mask enforcement)

Usage:
    python -m src.main                                    # run all smoke tests
    python -m src.main train --instance_path data/ --epochs 100
    python -m src.main train --instance_path data/ --fp16 --gdrive_path /content/drive/MyDrive/ml4vrp
    python -m src.main train --instance_path data/ --resume checkpoints/checkpoint_epoch10.pth --start_epoch 11
"""

import argparse
import pathlib
import shutil
import time

import torch


def smoke_test_fleet_manager():
    """Verify FleetManager forward pass with 10 actions."""
    from Model.Agent_Manager import FleetManager, NUM_FLEET_ACTIONS, INSTANCE_FEATURES_DIM, SOLVER_STATS_DIM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = FleetManager().to(device)

    B = 4
    inst_feat = torch.rand(B, INSTANCE_FEATURES_DIM, device=device)
    solver_stats = torch.rand(B, SOLVER_STATS_DIM, device=device)

    action_logits, state_value = manager(inst_feat, solver_stats)

    assert action_logits.shape == (B, NUM_FLEET_ACTIONS), f"Expected ({B}, {NUM_FLEET_ACTIONS}), got {action_logits.shape}"
    assert state_value.shape == (B, 1), f"Expected ({B}, 1), got {state_value.shape}"

    # Verify select_action sampling
    action, log_prob, sv = manager.select_action(inst_feat, solver_stats)
    assert action.shape == (B,), f"Expected ({B},), got {action.shape}"
    assert log_prob.shape == (B,), f"Expected ({B},), got {log_prob.shape}"
    assert all(0 <= a < NUM_FLEET_ACTIONS for a in action.tolist())

    print(f"  Action logits: {action_logits.shape}")
    print(f"  State value:   {state_value.shape}")
    print(f"  Sampled actions: {action.tolist()}")
    print(f"  Parameters: {sum(p.numel() for p in manager.parameters()):,}")
    print("  PASSED")


def smoke_test_fleet_manager_fp16():
    """Verify FleetManager under AMP on GPU."""
    from Model.Agent_Manager import FleetManager, NUM_FLEET_ACTIONS, INSTANCE_FEATURES_DIM, SOLVER_STATS_DIM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  Skipped (no GPU)")
        return

    manager = FleetManager().to(device)
    B = 8
    inst_feat = torch.rand(B, INSTANCE_FEATURES_DIM, device=device)
    solver_stats = torch.rand(B, SOLVER_STATS_DIM, device=device)

    with torch.amp.autocast("cuda"):
        action_logits, state_value = manager(inst_feat, solver_stats)

    assert action_logits.shape == (B, NUM_FLEET_ACTIONS)
    assert state_value.shape == (B, 1)
    print(f"  Action logits dtype: {action_logits.dtype}")
    print(f"  State value dtype:   {state_value.dtype}")
    print("  PASSED")


def smoke_test_pipeline():
    """End-to-end: Instance features -> FleetManager on a synthetic observation."""
    from Model.Agent_Manager import FleetManager, ACTION_NAMES, INSTANCE_FEATURES_DIM, SOLVER_STATS_DIM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = FleetManager().to(device)

    # Simulate hand-crafted instance features + solver stats
    inst_feat = torch.rand(1, INSTANCE_FEATURES_DIM, device=device)
    solver_stats = torch.tensor([[0.5, 0.8, 0.95, 0.1, 0.2, 0.0, 0.0]], device=device)
    action, log_prob, value = manager.select_action(inst_feat, solver_stats)

    print(f"  Instance features: {inst_feat.shape}")
    print(f"  Fleet Manager -> action: {ACTION_NAMES[action.item()]}, value: {value.item():.4f}")
    print("  PASSED")


def smoke_test_cvrp_env():
    """Stage 3: Run CVRPEnv for one full step on a synthetic .vrp instance."""
    import tempfile
    import os
    from Model.Solver_Engine import CVRPEnv, ACTION_NAMES, OBS_DIM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vrp_content = """\
NAME : smoke-test-n10
COMMENT : Smoke test instance
TYPE : CVRP
DIMENSION : 11
CAPACITY : 50
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 50 50
2 20 70
3 80 70
4 30 30
5 70 30
6 50 80
7 10 50
8 90 50
9 40 60
10 60 40
11 50 20
DEMAND_SECTION
1 0
2 10
3 15
4 10
5 20
6 10
7 5
8 15
9 10
10 10
11 15
DEPOT_SECTION
1
-1
EOF
"""
    tmpdir = tempfile.mkdtemp()
    vrp_path = os.path.join(tmpdir, "smoke-test-n10.vrp")
    with open(vrp_path, "w") as f:
        f.write(vrp_content)

    env = CVRPEnv(
        instance_paths=[vrp_path],
        device=device,
        max_steps=5,
    )

    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
    print(f"  Reset -> NV={info['nv']}, TD={info['td']:.0f}, score={info['score']:.0f}")

    # Step through a mixed set including new actions.
    # These cover default behavior plus the three added action families.
    total_reward = 0.0
    for action in [0, 7, 8, 9, 6]:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (OBS_DIM,)
        print(f"  Step {info['step']}: {ACTION_NAMES[action]:>22s} -> "
              f"NV={info['nv']}, TD={info['td']:.0f}, "
              f"reward={reward:+.1f}, score={info['score']:.0f}")

    print(f"  Total reward over 5 steps: {total_reward:+.1f}")

    os.remove(vrp_path)
    os.rmdir(tmpdir)
    print("  PASSED")


def smoke_test_training():
    """Stage 5: Run 2 complete PPO iterations on a synthetic 100-node instance."""
    import tempfile
    import os
    import random as pyrandom
    from Model.Solver_Engine import CVRPEnv
    from Model.Train import MARLTrainer, PPOConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pyrandom.seed(42)
    lines = [
        "NAME : smoke-train-n100",
        "COMMENT : Training smoke test",
        "TYPE : CVRP",
        "DIMENSION : 100",
        "CAPACITY : 50",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for i in range(1, 101):
        lines.append(f"{i} {pyrandom.randint(0, 100)} {pyrandom.randint(0, 100)}")
    lines.append("DEMAND_SECTION")
    lines.append("1 0")
    for i in range(2, 101):
        lines.append(f"{i} {pyrandom.randint(1, 15)}")
    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF"])

    tmpdir = tempfile.mkdtemp()
    vrp_path = os.path.join(tmpdir, "smoke-train-n100.vrp")
    with open(vrp_path, "w") as f:
        f.write("\n".join(lines))

    env = CVRPEnv(
        instance_paths=[vrp_path],
        device=device,
        max_steps=3,
    )

    config = PPOConfig(
        mini_batch_size=4,
        ppo_epochs=2,
        use_fp16=(device.type == "cuda"),
    )

    trainer = MARLTrainer(env=env, config=config, device=device,
                          log_dir=tmpdir, gdrive_path=None)

    for i in range(2):
        stats = trainer.train_epoch(num_episodes=1, run_eval=False)
        print(
            f"  PPO Iteration {i + 1}: "
            f"score={stats['avg_score']:.0f}, "
            f"NV={stats['avg_nv']:.0f}, "
            f"loss={stats['policy_loss']:.4f}"
        )

    # Verify checkpoint save/load round-trip
    ckpt_path = os.path.join(tmpdir, "test_checkpoint.pth")
    trainer.save_checkpoint(ckpt_path)
    trainer.load_checkpoint(ckpt_path)
    print(f"  Checkpoint save/load: OK")

    import glob as globmod
    for f in globmod.glob(os.path.join(tmpdir, "*")):
        os.remove(f)
    os.rmdir(tmpdir)
    print("  PASSED")


def smoke_test_action_masking():
    """Verify NV_min calculation and action masking with 10-action space."""
    import tempfile
    import os
    from Model.Solver_Engine import CVRPEnv, NUM_ACTIONS, INSTANCE_FEATURES_DIM
    from Model.Agent_Manager import FleetManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 depot + 10 customers, each demand=10, capacity=50 -> NV_min = ceil(100/50) = 2
    vrp_content = """\
NAME : mask-test-n10
COMMENT : Action masking test
TYPE : CVRP
DIMENSION : 11
CAPACITY : 50
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 50 50
2 20 70
3 80 70
4 30 30
5 70 30
6 50 80
7 10 50
8 90 50
9 40 60
10 60 40
11 50 20
DEMAND_SECTION
1 0
2 10
3 10
4 10
5 10
6 10
7 10
8 10
9 10
10 10
11 10
DEPOT_SECTION
1
-1
EOF
"""
    tmpdir = tempfile.mkdtemp()
    vrp_path = os.path.join(tmpdir, "mask-test-n10.vrp")
    with open(vrp_path, "w") as f:
        f.write(vrp_content)

    env = CVRPEnv(
        instance_paths=[vrp_path],
        device=device,
        max_steps=5,
    )
    obs, info = env.reset(seed=42)
    nv_min = info["nv_min"]
    mask = info["action_mask"]
    current_nv = info["nv"]

    print(f"  NV_min = ceil(100/50) = {nv_min}")
    print(f"  Current NV = {current_nv}")
    print(f"  Action mask ({NUM_ACTIONS} actions): {mask.tolist()}")
    assert nv_min == 2, f"Expected NV_min=2, got {nv_min}"

    # Test that FleetManager respects the mask
    manager = FleetManager().to(device)
    manager.eval()

    inst_feat = torch.tensor(obs[:INSTANCE_FEATURES_DIM], dtype=torch.float32, device=device).unsqueeze(0)
    stats = torch.tensor(obs[INSTANCE_FEATURES_DIM:], dtype=torch.float32, device=device).unsqueeze(0)

    # Force mask: block all fleet-reduction actions
    # (4=PUSH_SAME, 5=PUSH_NEW, 6=FORCE_MIN, 9=PUSH_BALANCED_NEW)
    forced_mask = torch.tensor(
        [[True, True, True, True, False, False, False, True, True, False]],
        dtype=torch.bool,
        device=device,
    )
    with torch.no_grad():
        logits, _ = manager(inst_feat, stats, action_mask=forced_mask)

    for blocked in [4, 5, 6, 9]:
        assert logits[0, blocked].item() < -1e3, \
            f"Action {blocked} logit should be masked, got {logits[0, blocked].item()}"

    # Sample 100 actions with mask — none should be blocked actions
    actions = []
    for _ in range(100):
        with torch.no_grad():
            a, _, _ = manager.select_action(inst_feat, stats, action_mask=forced_mask)
        actions.append(a.item())
    blocked_set = {4, 5, 6, 9}
    sampled_blocked = blocked_set & set(actions)
    assert not sampled_blocked, f"Blocked actions {sampled_blocked} were sampled despite mask!"
    print(f"  100 masked samples: {set(actions)} (fleet-reduction actions never sampled)")

    os.remove(vrp_path)
    os.rmdir(tmpdir)
    print("  PASSED")


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def save_to_gdrive(src_path: pathlib.Path, gdrive_dir: str):
    """Copy a checkpoint file to Google Drive."""
    gdrive = pathlib.Path(gdrive_dir)
    gdrive.mkdir(parents=True, exist_ok=True)
    dst = gdrive / src_path.name
    shutil.copy2(src_path, dst)
    print(f"  Pushed to Google Drive: {dst}")


def train(args):
    """Run the PPO training loop for the Fleet Manager.

    This sets up all components and runs the main training loop:
      1. Load .vrp instance files from the dataset directory
      2. Create the CVRPEnv (Gymnasium environment wrapping HGS-CVRP)
      3. Select 5 fixed evaluation instances for progress tracking
      4. Create the MARLTrainer (PPO training engine)
      5. Run the epoch loop with curriculum learning and checkpointing
    """
    from Model.Solver_Engine import CVRPEnv
    from Model.Train import MARLTrainer, PPOConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Dataset ---
    # Supports both a single .vrp file and a directory of .vrp files
    instance_dir = pathlib.Path(args.instance_path)
    if instance_dir.is_file():
        instance_paths = [instance_dir]
    else:
        instance_paths = sorted(instance_dir.glob("*.vrp"))
    assert len(instance_paths) > 0, f"No .vrp files found at {args.instance_path}"
    print(f"Instances: {len(instance_paths)}")

    if args.reward_clip_min >= args.reward_clip_max:
        raise ValueError(
            f"Invalid reward clip range: min={args.reward_clip_min} must be < max={args.reward_clip_max}"
        )
    if args.eval_count < 0:
        raise ValueError(f"Invalid --eval_count={args.eval_count}. It must be >= 0.")
    if args.holdout_count < 0:
        raise ValueError(f"Invalid --holdout_count={args.holdout_count}. It must be >= 0.")
    if args.holdout_eval_interval <= 0:
        raise ValueError(
            f"Invalid --holdout_eval_interval={args.holdout_eval_interval}. It must be >= 1."
        )

    # --- Create Environment ---
    # CVRPEnv wraps HGS-CVRP and implements the Gymnasium interface.
    # Hand-crafted instance features replace the untrained GNN encoder.
    # Curriculum learning: start with small instances (N<=100) for the first
    # curriculum_epochs, then unlock all instances (up to 400 nodes).
    env = CVRPEnv(
        instance_paths=[str(p) for p in instance_paths],
        device=device,
        max_nodes=100 if args.curriculum_epochs > 0 else None,
        failure_penalty=args.failure_penalty,
        no_improvement_penalty=args.no_improvement_penalty,
    )
    if args.curriculum_epochs > 0:
        print(f"Curriculum: N<=100 for first {args.curriculum_epochs} epochs, then all")

    # --- Select Fixed Evaluation Instances ---
    # Pick N evenly-spaced instances from a sorted pool for consistent progress
    # tracking. The same fixed set is used every epoch.
    eval_instances = []
    if args.eval_count > 0:
        eval_pool = []
        if args.eval_instances:
            eval_path = pathlib.Path(args.eval_instances)
            if eval_path.is_file():
                eval_pool = [eval_path]
            else:
                eval_pool = sorted(eval_path.glob("*.vrp"), key=lambda p: p.stem)
        else:
            eval_pool = sorted(instance_paths, key=lambda p: p.stem)

        if eval_pool:
            step = max(1, len(eval_pool) // args.eval_count)
            eval_instances = [str(eval_pool[i]) for i in range(0, len(eval_pool), step)][:args.eval_count]

    if eval_instances:
        preview = ", ".join(pathlib.Path(p).stem for p in eval_instances[:5])
        print(f"Eval set: {len(eval_instances)} fixed instances (preview: {preview})")
    else:
        print("Eval set: disabled (eval_count=0)")

    # --- Select Holdout Evaluation Instances ---
    # Holdout split is non-overlapping with fixed eval set.
    holdout_instances = []
    if args.holdout_count > 0:
        holdout_pool = []
        if args.holdout_instances:
            holdout_path = pathlib.Path(args.holdout_instances)
            if holdout_path.is_file():
                holdout_pool = [holdout_path]
            else:
                holdout_pool = sorted(holdout_path.glob("*.vrp"), key=lambda p: p.stem)
        else:
            holdout_pool = sorted(instance_paths, key=lambda p: p.stem)

        eval_set = {pathlib.Path(p).resolve() for p in eval_instances}
        holdout_pool = [p for p in holdout_pool if p.resolve() not in eval_set]

        if holdout_pool:
            step = max(1, len(holdout_pool) // args.holdout_count)
            holdout_instances = [
                str(holdout_pool[i]) for i in range(0, len(holdout_pool), step)
            ][:args.holdout_count]

    if holdout_instances:
        preview = ", ".join(pathlib.Path(p).stem for p in holdout_instances[:5])
        print(
            f"Holdout set: {len(holdout_instances)} instances "
            f"(interval={args.holdout_eval_interval}, preview: {preview})"
        )
    elif args.holdout_count > 0:
        raise ValueError(
            "Holdout split requested but no holdout instances were selected. "
            "Check --holdout_instances path or reduce --holdout_count."
        )

    if args.best_model_metric in ("holdout", "composite") and not holdout_instances:
        raise ValueError(
            f"--best_model_metric={args.best_model_metric} requires a non-empty holdout set. "
            "Set --holdout_count > 0 (and optionally --holdout_instances)."
        )

    config = PPOConfig(
        gamma=args.gamma,
        lam=args.lam,
        epsilon_clip=args.epsilon_clip,
        vf_coeff=args.vf_coeff,
        manager_lr=args.manager_lr,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        use_fp16=args.fp16,
        ent_coeff=args.ent_coeff,
        target_kl=None if args.target_kl <= 0 else args.target_kl,
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
    )

    run_config = {
        "run_tag": args.run_tag,
        "cli_args": vars(args),
        "selected_eval_instances": eval_instances,
        "selected_holdout_instances": holdout_instances,
        "holdout_eval_interval": args.holdout_eval_interval,
        "best_model_metric": args.best_model_metric,
    }

    trainer = MARLTrainer(
        env=env, config=config, device=device,
        log_dir=args.log_dir, gdrive_path=args.gdrive_path,
        total_epochs=args.epochs,
        eval_instances=eval_instances,
        holdout_instances=holdout_instances,
        holdout_eval_interval=args.holdout_eval_interval,
        best_model_metric=args.best_model_metric,
        run_config=run_config,
    )

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume from Checkpoint (if provided) ---
    # Restores model weights, optimizer state, LR schedule, and best score.
    # Used after GPU quota exhaustion or accidental interruption.
    start_epoch = args.start_epoch
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  Loaded. best_score={trainer.best_score:.0f}, starting at epoch {start_epoch}")

    print(f"Training for {args.epochs} epochs (starting at {start_epoch}), {args.episodes_per_epoch} episodes/epoch...")

    # --- Curriculum Learning ---
    # If resuming past the curriculum boundary, immediately unlock all instances.
    # Otherwise, start with small instances and expand at curriculum_epochs+1.
    curriculum_expanded = start_epoch > args.curriculum_epochs
    if curriculum_expanded:
        env.set_max_nodes(None)

    # --- Main Training Loop ---
    epoch_times = []
    best_tracking_so_far = float("inf")
    epochs_since_improvement = 0
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()

        # Check if it's time to expand the curriculum (unlock larger instances)
        if not curriculum_expanded and epoch > args.curriculum_epochs and args.curriculum_epochs > 0:
            env.set_max_nodes(None)
            curriculum_expanded = True
            print(f"  [Epoch {epoch}] Curriculum expanded: now using all instances")

        stats = trainer.train_epoch(
            num_episodes=args.episodes_per_epoch,
            epoch_index=epoch,
        )

        # --- Timing ---
        epoch_elapsed = time.time() - epoch_t0
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_h, eta_rem = divmod(int(eta_seconds), 3600)
        eta_m = eta_rem // 60
        ep_m, ep_s = divmod(int(epoch_elapsed), 60)

        # --- Best tracking metric summary ---
        eval_str = ""
        holdout_str = ""
        tracking_str = ""
        is_new_best = False
        if "eval_score" in stats:
            eval_str = f" | Eval: {stats['eval_score']:>8.0f}"
        if "holdout_eval_score" in stats:
            holdout_str = f" | Holdout: {stats['holdout_eval_score']:>8.0f}"

        tracking_val_raw = stats.get("tracking_score", "")
        selection_metric = stats.get("selection_metric", "tracking")
        if tracking_val_raw != "":
            tracking_val = float(tracking_val_raw)
            if tracking_val < best_tracking_so_far:
                best_tracking_so_far = tracking_val
                epochs_since_improvement = 0
                is_new_best = True
            else:
                epochs_since_improvement += 1

            best_tag = " *BEST*" if is_new_best else ""
            tracking_str = (
                f" | Sel[{selection_metric}]: {tracking_val:>8.0f}{best_tag}"
            )
        else:
            tracking_str = f" | Sel[{selection_metric}]: {'n/a':>8s}"

        total_elapsed = time.time() - train_start
        te_h, te_rem = divmod(int(total_elapsed), 3600)
        te_m = te_rem // 60

        print(
            f"[{epoch:>4d}/{args.epochs}] "
            f"AvgScore: {stats['avg_score']:>8.0f}{eval_str}{holdout_str}{tracking_str} | "
            f"Ent: {stats['entropy']:.3f} | "
            f"{ep_m}m{ep_s:02d}s/epoch | "
            f"ETA: {eta_h}h{eta_m:02d}m | "
            f"Elapsed: {te_h}h{te_m:02d}m"
        )

        # --- Early warning ---
        if best_tracking_so_far < float("inf") and epochs_since_improvement >= 15:
            print(
                f"  WARNING: No tracked-metric improvement for {epochs_since_improvement} epochs. "
                f"Best tracked metric: {best_tracking_so_far:.0f}. Consider stopping if no progress soon."
            )

        if epoch % args.save_interval == 0:
            ckpt = checkpoint_dir / f"checkpoint_epoch{epoch}.pth"
            trainer.save_checkpoint(str(ckpt))
            print(f"  Saved: {ckpt}")
            if args.gdrive_path:
                save_to_gdrive(ckpt, args.gdrive_path)

    final = checkpoint_dir / "checkpoint_final.pth"
    trainer.save_checkpoint(str(final))
    total_time = time.time() - train_start
    tt_h, tt_rem = divmod(int(total_time), 3600)
    tt_m = tt_rem // 60
    print(f"Training complete in {tt_h}h{tt_m:02d}m. Final checkpoint: {final}")
    if args.gdrive_path:
        save_to_gdrive(final, args.gdrive_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML4VRP 2026 - RL-Guided CVRP Solver"
    )
    sub = parser.add_subparsers(dest="mode")

    tp = sub.add_parser("train", help="Run PPO training")
    tp.add_argument("--epochs", type=int, default=50)
    tp.add_argument("--batch_size", type=int, default=128)
    tp.add_argument("--instance_path", type=str, required=True,
                     help="Path to .vrp file or directory of .vrp files")
    tp.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    tp.add_argument("--save_interval", type=int, default=10)
    tp.add_argument("--episodes_per_epoch", type=int, default=8,
                     help="Episodes per epoch (more = stabler gradients)")
    tp.add_argument("--eval_count", type=int, default=5,
                     help="Number of fixed evaluation instances (0 disables eval)")
    tp.add_argument("--eval_instances", type=str, default=None,
                     help="Directory (or single file) used as evaluation pool")
    tp.add_argument("--holdout_count", type=int, default=0,
                     help="Number of holdout instances (0 disables holdout)")
    tp.add_argument("--holdout_instances", type=str, default=None,
                     help="Directory (or single file) used as holdout pool")
    tp.add_argument("--holdout_eval_interval", type=int, default=5,
                     help="Evaluate holdout every N epochs")
    tp.add_argument("--best_model_metric", type=str, default="eval",
                     choices=["eval", "holdout", "composite"],
                     help="Metric used to track/save best_model checkpoint")

    # PPO training knobs
    tp.add_argument("--gamma", type=float, default=0.95,
                     help="Reward discount factor")
    tp.add_argument("--lam", type=float, default=0.90,
                     help="GAE lambda")
    tp.add_argument("--epsilon_clip", type=float, default=0.2,
                     help="PPO clipping epsilon")
    tp.add_argument("--vf_coeff", type=float, default=0.5,
                     help="Value loss coefficient")
    tp.add_argument("--manager_lr", type=float, default=1e-4)
    tp.add_argument("--ppo_epochs", type=int, default=3,
                     help="PPO mini-epochs per training epoch")
    tp.add_argument("--max_grad_norm", type=float, default=0.5,
                     help="Gradient clipping norm")
    tp.add_argument("--target_kl", type=float, default=0.015,
                     help="KL early-stop threshold (<=0 disables)")

    # Reward/penalty controls
    tp.add_argument("--reward_clip_min", type=float, default=-10.0,
                     help="Lower reward clipping bound")
    tp.add_argument("--reward_clip_max", type=float, default=10.0,
                     help="Upper reward clipping bound")
    tp.add_argument("--failure_penalty", type=float, default=-5.0,
                     help="Penalty for fleet explosion / failed solve")
    tp.add_argument("--no_improvement_penalty", type=float, default=-0.5,
                     help="Penalty when a step does not beat episode-best score")

    tp.add_argument("--fp16", action="store_true",
                     help="Enable FP16 mixed precision (requires CUDA)")
    tp.add_argument("--ent_coeff", type=float, default=0.02,
                     help="Entropy bonus coefficient")
    tp.add_argument("--run_tag", type=str, default="",
                     help="Short run label stored in logs for traceability")
    tp.add_argument("--log_dir", type=str, default="logs",
                     help="Directory for CSV training metrics")
    tp.add_argument("--gdrive_path", type=str, default=None,
                     help="Google Drive directory for checkpoint backup")
    tp.add_argument("--curriculum_epochs", type=int, default=20,
                     help="Epochs to restrict to small instances before expanding")
    tp.add_argument("--resume", type=str, default=None,
                     help="Path to checkpoint .pth file to resume training from")
    tp.add_argument("--start_epoch", type=int, default=1,
                     help="Epoch number to start from (for resumed runs)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    else:
        print("=== Stage 2: Fleet Manager (10 actions) ===")
        smoke_test_fleet_manager()
        print("\n=== Stage 2: Fleet Manager FP16 ===")
        smoke_test_fleet_manager_fp16()
        print("\n=== Stage 2: Instance Features -> Fleet Manager Pipeline ===")
        smoke_test_pipeline()
        print("\n=== Stage 3: CVRPEnv (HGS-CVRP wrapper, 10 actions) ===")
        smoke_test_cvrp_env()
        print("\n=== Stage 5: Training (2 PPO iterations) ===")
        smoke_test_training()
        print("\n=== Stage 6: Action Masking & NV_min ===")
        smoke_test_action_masking()
        print("\nAll smoke tests passed.")
