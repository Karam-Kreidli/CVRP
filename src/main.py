"""
ML4VRP 2026 - Hierarchical MARL-HGS Hybrid Solver
Primary entry point: training loop (Stage 5) and smoke tests (Stages 1-4).

Usage:
    python -m src.main                                    # run all smoke tests
    python -m src.main train --instance_path data/ --epochs 100
    python -m src.main train --instance_path data/ --fp16 --gdrive_path /content/drive/MyDrive/ml4vrp
"""

import argparse
import pathlib
import shutil

import torch


def smoke_test():
    """Verify GNNEncoder runs on a synthetic CVRP instance."""
    from src.model_vision import GNNEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GNNEncoder().to(device)

    # Synthetic instance: 1 depot + 99 customers = 100 nodes
    N = 100
    coords = torch.rand(N, 2)
    demands = torch.rand(N, 1)
    demands[0] = 0.0  # depot has no demand

    x = torch.cat([coords, demands], dim=-1).to(device)
    pos = coords.to(device)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    node_emb, graph_emb = encoder(x, pos, batch)

    assert node_emb.shape == (100, 128), f"Expected (100, 128), got {node_emb.shape}"
    assert graph_emb.shape == (1, 128), f"Expected (1, 128), got {graph_emb.shape}"

    print(f"  Node embeddings: {node_emb.shape}")
    print(f"  Graph embedding: {graph_emb.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("  PASSED")


def smoke_test_fp16():
    """Verify AMP compatibility on GPU."""
    from src.model_vision import GNNEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  Skipped (no GPU)")
        return

    encoder = GNNEncoder().to(device)
    N = 400
    x = torch.rand(N, 3, device=device)
    x[0, 2] = 0.0
    pos = x[:, :2]
    batch = torch.zeros(N, dtype=torch.long, device=device)

    with torch.amp.autocast("cuda"):
        node_emb, graph_emb = encoder(x, pos, batch)

    assert node_emb.shape == (400, 128)
    assert graph_emb.shape == (1, 128)
    print(f"  Node embeddings dtype: {node_emb.dtype}")
    print(f"  Graph embedding dtype:  {graph_emb.dtype}")
    print("  PASSED")


def smoke_test_batched():
    """Verify batched processing with variable-size graphs."""
    from torch_geometric.data import Data, Batch
    from src.model_vision import GNNEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GNNEncoder().to(device)

    graphs = []
    total_nodes = 0
    for n in [101, 201, 301, 401]:
        coords = torch.rand(n, 2)
        demands = torch.rand(n, 1)
        demands[0] = 0.0
        x = torch.cat([coords, demands], dim=-1)
        graphs.append(Data(x=x, pos=coords))
        total_nodes += n

    batch = Batch.from_data_list(graphs).to(device)
    node_emb, graph_emb = encoder(batch.x, batch.pos, batch.batch)

    assert node_emb.shape == (total_nodes, 128), f"Expected ({total_nodes}, 128), got {node_emb.shape}"
    assert graph_emb.shape == (4, 128), f"Expected (4, 128), got {graph_emb.shape}"

    print(f"  Batched node embeddings: {node_emb.shape}")
    print(f"  Batched graph embeddings: {graph_emb.shape}")
    print("  PASSED")


def smoke_test_fleet_manager():
    """Verify FleetManager forward pass with dummy 132-dim input."""
    from src.agent_manager import FleetManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = FleetManager().to(device)

    B = 4
    graph_emb = torch.rand(B, 128, device=device)
    solver_stats = torch.rand(B, 4, device=device)

    action_logits, state_value = manager(graph_emb, solver_stats)

    assert action_logits.shape == (B, 3), f"Expected ({B}, 3), got {action_logits.shape}"
    assert state_value.shape == (B, 1), f"Expected ({B}, 1), got {state_value.shape}"

    # Verify select_action sampling
    action, log_prob, sv = manager.select_action(graph_emb, solver_stats)
    assert action.shape == (B,), f"Expected ({B},), got {action.shape}"
    assert log_prob.shape == (B,), f"Expected ({B},), got {log_prob.shape}"
    assert all(a in (0, 1, 2) for a in action.tolist())

    print(f"  Action logits: {action_logits.shape}")
    print(f"  State value:   {state_value.shape}")
    print(f"  Sampled actions: {action.tolist()}")
    print(f"  Parameters: {sum(p.numel() for p in manager.parameters()):,}")
    print("  PASSED")


def smoke_test_fleet_manager_fp16():
    """Verify FleetManager under AMP on GPU."""
    from src.agent_manager import FleetManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  Skipped (no GPU)")
        return

    manager = FleetManager().to(device)
    B = 8
    graph_emb = torch.rand(B, 128, device=device)
    solver_stats = torch.rand(B, 4, device=device)

    with torch.amp.autocast("cuda"):
        action_logits, state_value = manager(graph_emb, solver_stats)

    assert action_logits.shape == (B, 3)
    assert state_value.shape == (B, 1)
    print(f"  Action logits dtype: {action_logits.dtype}")
    print(f"  State value dtype:   {state_value.dtype}")
    print("  PASSED")


def smoke_test_pipeline():
    """End-to-end: GNNEncoder → FleetManager on a synthetic instance."""
    from src.model_vision import GNNEncoder
    from src.agent_manager import FleetManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GNNEncoder().to(device)
    manager = FleetManager().to(device)

    # Synthetic 400-node instance
    N = 401
    coords = torch.rand(N, 2)
    demands = torch.rand(N, 1)
    demands[0] = 0.0
    x = torch.cat([coords, demands], dim=-1).to(device)
    pos = coords.to(device)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # Stage 1: encode
    node_emb, graph_emb = encoder(x, pos, batch)

    # Stage 2: decide fleet action
    solver_stats = torch.tensor([[0.5, 0.8, 0.02, 0.1]], device=device)  # mid-run stats
    action, log_prob, value = manager.select_action(graph_emb, solver_stats)

    action_names = ["KEEP", "REMOVE", "ADD"]
    print(f"  GNN → graph_emb: {graph_emb.shape}")
    print(f"  Fleet Manager → action: {action_names[action.item()]}, value: {value.item():.4f}")
    print("  PASSED")


def smoke_test_cvrp_env():
    """Stage 3: Run CVRPEnv for one full step on a synthetic .vrp instance."""
    import tempfile
    import os
    from src.model_vision import GNNEncoder
    from src.solver_engine import CVRPEnv, ACTION_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GNNEncoder().to(device)
    encoder.eval()

    # Write a minimal CVRP instance in VRPLIB format (1 depot + 10 customers)
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
        encoder=encoder,
        device=device,
        iters_per_step=500,
        max_steps=5,
    )

    # Reset: initial solve
    obs, info = env.reset(seed=42)
    assert obs.shape == (132,), f"Expected (132,), got {obs.shape}"
    print(f"  Reset → NV={info['nv']}, TD={info['td']:.0f}, score={info['score']:.0f}")

    # Step through all 3 actions
    total_reward = 0.0
    for action in [0, 1, 2]:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (132,)
        print(f"  Step {info['step']}: {ACTION_NAMES[action]:>20s} → "
              f"NV={info['nv']}, TD={info['td']:.0f}, "
              f"reward={reward:+.1f}, score={info['score']:.0f}")

    print(f"  Total reward over 3 steps: {total_reward:+.1f}")

    # Cleanup
    os.remove(vrp_path)
    os.rmdir(tmpdir)
    print("  PASSED")


def smoke_test_route_driver():
    """Stage 4: Verify RouteDriver processes 400-node embeddings and selects an operator."""
    import time
    from src.agent_driver import RouteDriver, OPERATOR_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    driver = RouteDriver().to(device)
    driver.eval()

    # Simulate 400-node instance embeddings (as would come from GNNEncoder)
    N = 400
    node_embeddings = torch.randn(N, 128, device=device)

    # Forward pass — verify shapes
    with torch.no_grad():
        logits, value = driver(node_embeddings)
    assert logits.shape == (1, 4), f"Expected (1, 4), got {logits.shape}"
    assert value.shape == (1, 1), f"Expected (1, 1), got {value.shape}"

    # Timing: 100 forward passes, report average
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            driver(node_embeddings)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            driver(node_embeddings)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

    # Sampling
    with torch.no_grad():
        op, log_prob, sv = driver.select_operator(node_embeddings)
    assert 0 <= op <= 3, f"Invalid operator index: {op}"

    print(f"  Operator logits: {logits.squeeze().tolist()}")
    print(f"  Selected operator: {OPERATOR_NAMES[op]}")
    print(f"  State value: {sv.item():.4f}")
    print(f"  Avg inference: {elapsed_ms:.2f} ms ({N} nodes)")
    print(f"  Parameters: {sum(p.numel() for p in driver.parameters()):,}")
    if elapsed_ms > 10.0:
        print(f"  WARNING: {elapsed_ms:.2f} ms exceeds 10ms target for T4 GPU")
    print("  PASSED")


def smoke_test_route_driver_fp16():
    """Stage 4: Verify RouteDriver under AMP on GPU."""
    from src.agent_driver import RouteDriver, OPERATOR_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  Skipped (no GPU)")
        return

    driver = RouteDriver().to(device)
    driver.eval()
    node_embeddings = torch.randn(400, 128, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits, value = driver(node_embeddings)

    assert logits.shape == (1, 4)
    assert value.shape == (1, 1)
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Value dtype:  {value.dtype}")
    print("  PASSED")


def smoke_test_training():
    """Stage 5: Run 2 complete PPO iterations on a synthetic 100-node instance.

    Validates that the full training pipeline (rollout collection, MACA reward
    decomposition, reward normalization, GAE, PPO backward passes) runs
    without tensor shape or device errors.
    """
    import tempfile
    import os
    import random as pyrandom
    from src.model_vision import GNNEncoder
    from src.solver_engine import CVRPEnv
    from src.train import MARLTrainer, PPOConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GNNEncoder().to(device)
    encoder.eval()

    # Generate a synthetic 100-node CVRP instance (1 depot + 99 customers)
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
        encoder=encoder,
        device=device,
        iters_per_step=100,
        max_steps=3,
        route_driver=None,
    )

    config = PPOConfig(
        mini_batch_size=4,
        ppo_epochs=2,
        use_fp16=(device.type == "cuda"),
    )

    trainer = MARLTrainer(env=env, config=config, device=device,
                          log_dir=tmpdir, gdrive_path=None)

    for i in range(2):
        stats = trainer.train_epoch(num_episodes=1)
        print(
            f"  PPO Iteration {i + 1}: "
            f"score={stats['final_score']:.0f}, "
            f"NV={stats['final_nv']}, "
            f"mgr_loss={stats['mgr_policy_loss']:.4f}, "
            f"drv_loss={stats['drv_policy_loss']:.4f}, "
            f"mgr_norm={stats.get('mgr_norm_mag', 0):.3f}, "
            f"drv_norm={stats.get('drv_norm_mag', 0):.3f}"
        )

    # Verify checkpoint save/load round-trip
    ckpt_path = os.path.join(tmpdir, "test_checkpoint.pth")
    trainer.save_checkpoint(ckpt_path)
    trainer.load_checkpoint(ckpt_path)
    print(f"  Checkpoint save/load: OK")

    # Cleanup
    import glob as globmod
    for f in globmod.glob(os.path.join(tmpdir, "*")):
        os.remove(f)
    os.rmdir(tmpdir)
    print("  PASSED")


def smoke_test_action_masking():
    """Stage 6: Verify NV_min calculation and action masking.

    Creates a synthetic instance, computes NV_min, and confirms the Manager
    cannot choose ROUTE_ELIMINATION when NV <= NV_min.
    """
    import tempfile
    import os
    from src.model_vision import GNNEncoder
    from src.solver_engine import CVRPEnv
    from src.agent_manager import FleetManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GNNEncoder().to(device)
    encoder.eval()

    # 1 depot + 10 customers, each demand=10, capacity=50 → NV_min = ceil(100/50) = 2
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
        encoder=encoder,
        device=device,
        iters_per_step=500,
        max_steps=5,
    )
    obs, info = env.reset(seed=42)
    nv_min = info["nv_min"]
    mask = info["action_mask"]
    current_nv = info["nv"]

    print(f"  NV_min = ceil(100/50) = {nv_min}")
    print(f"  Current NV = {current_nv}")
    print(f"  Action mask: {mask.tolist()} (POLISH={mask[0]}, ELIM={mask[1]}, EXPLORE={mask[2]})")
    assert nv_min == 2, f"Expected NV_min=2, got {nv_min}"

    # Test that FleetManager respects the mask
    manager = FleetManager().to(device)
    manager.eval()

    graph_emb = torch.tensor(obs[:128], dtype=torch.float32, device=device).unsqueeze(0)
    stats = torch.tensor(obs[128:], dtype=torch.float32, device=device).unsqueeze(0)

    # Force mask: block ROUTE_ELIMINATION
    forced_mask = torch.tensor([[True, False, True]], dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, _ = manager(graph_emb, stats, action_mask=forced_mask)
    assert logits[0, 1].item() < -1e3, f"ROUTE_ELIMINATION logit should be masked, got {logits[0, 1].item()}"

    # Sample 100 actions with mask — none should be action 1
    actions = []
    for _ in range(100):
        with torch.no_grad():
            a, _, _ = manager.select_action(graph_emb, stats, action_mask=forced_mask)
        actions.append(a.item())
    assert 1 not in actions, f"ROUTE_ELIMINATION was sampled despite being masked!"
    print(f"  100 masked samples: {set(actions)} (action 1 never sampled) ✓")

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
    """Run the multi-agent PPO training loop."""
    from src.model_vision import GNNEncoder
    from src.solver_engine import CVRPEnv
    from src.train import MARLTrainer, PPOConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect instance files
    instance_dir = pathlib.Path(args.instance_path)
    if instance_dir.is_file():
        instance_paths = [instance_dir]
    else:
        instance_paths = sorted(instance_dir.glob("*.vrp"))
    assert len(instance_paths) > 0, f"No .vrp files found at {args.instance_path}"
    print(f"Instances: {len(instance_paths)}")

    encoder = GNNEncoder().to(device)
    encoder.eval()

    env = CVRPEnv(
        instance_paths=[str(p) for p in instance_paths],
        encoder=encoder,
        device=device,
        route_driver=None,
        max_nodes=100 if args.curriculum_epochs > 0 else None,
    )
    if args.curriculum_epochs > 0:
        print(f"Curriculum: N≤100 for first {args.curriculum_epochs} epochs, then N≤400")

    config = PPOConfig(
        manager_lr=args.manager_lr,
        driver_lr=args.driver_lr,
        mini_batch_size=args.batch_size,
        use_fp16=args.fp16,
        ent_coeff=args.ent_coeff,
    )

    trainer = MARLTrainer(
        env=env, config=config, device=device,
        log_dir=args.log_dir, gdrive_path=args.gdrive_path,
        total_epochs=args.epochs,
    )

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training for {args.epochs} epochs...")
    curriculum_expanded = False
    for epoch in range(1, args.epochs + 1):
        # Curriculum: expand to full dataset after curriculum_epochs
        if not curriculum_expanded and epoch > args.curriculum_epochs and args.curriculum_epochs > 0:
            env.set_max_nodes(None)
            curriculum_expanded = True
            print(f"  [Epoch {epoch}] Curriculum expanded: now using all instances (N≤400)")

        stats = trainer.train_epoch(num_episodes=args.episodes_per_epoch)
        print(
            f"[{epoch:>4d}/{args.epochs}] "
            f"Score: {stats['final_score']:>8.0f} | "
            f"NV: {stats['final_nv']:>2d} | "
            f"TD: {stats['final_td']:>8.0f} | "
            f"MgrPL: {stats['mgr_policy_loss']:>7.4f} | "
            f"DrvPL: {stats['drv_policy_loss']:>7.4f} | "
            f"MgrNrm: {stats.get('mgr_norm_mag', 0):.3f} | "
            f"DrvNrm: {stats.get('drv_norm_mag', 0):.3f} | "
            f"LR: {stats.get('mgr_lr', 0):.1e}/{stats.get('drv_lr', 0):.1e}"
        )

        if epoch % args.save_interval == 0:
            ckpt = checkpoint_dir / f"checkpoint_epoch{epoch}.pth"
            trainer.save_checkpoint(str(ckpt))
            print(f"  Saved: {ckpt}")
            if args.gdrive_path:
                save_to_gdrive(ckpt, args.gdrive_path)

    final = checkpoint_dir / "checkpoint_final.pth"
    trainer.save_checkpoint(str(final))
    print(f"Training complete. Final checkpoint: {final}")
    if args.gdrive_path:
        save_to_gdrive(final, args.gdrive_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML4VRP 2026 - Hierarchical MARL-HGS Hybrid Solver"
    )
    sub = parser.add_subparsers(dest="mode")

    # Train subcommand
    tp = sub.add_parser("train", help="Run multi-agent PPO training")
    tp.add_argument("--epochs", type=int, default=100)
    tp.add_argument("--batch_size", type=int, default=64)
    tp.add_argument("--instance_path", type=str, required=True,
                     help="Path to .vrp file or directory of .vrp files")
    tp.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    tp.add_argument("--save_interval", type=int, default=10)
    tp.add_argument("--episodes_per_epoch", type=int, default=1)
    tp.add_argument("--manager_lr", type=float, default=1e-4)
    tp.add_argument("--driver_lr", type=float, default=5e-4)
    tp.add_argument("--fp16", action="store_true",
                     help="Enable FP16 mixed precision (requires CUDA)")
    tp.add_argument("--ent_coeff", type=float, default=0.05,
                     help="Entropy bonus coefficient")
    tp.add_argument("--log_dir", type=str, default="logs",
                     help="Directory for CSV training metrics")
    tp.add_argument("--gdrive_path", type=str, default=None,
                     help="Google Drive directory for checkpoint backup")
    tp.add_argument("--curriculum_epochs", type=int, default=20,
                     help="Epochs to restrict to small instances (N<=100) before expanding")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    else:
        # Default: run all smoke tests
        print("=== Stage 1: Single Instance ===")
        smoke_test()
        print("\n=== Stage 1: FP16 (400 nodes) ===")
        smoke_test_fp16()
        print("\n=== Stage 1: Batched (variable sizes) ===")
        smoke_test_batched()
        print("\n=== Stage 2: Fleet Manager ===")
        smoke_test_fleet_manager()
        print("\n=== Stage 2: Fleet Manager FP16 ===")
        smoke_test_fleet_manager_fp16()
        print("\n=== Stage 1+2: End-to-End Pipeline ===")
        smoke_test_pipeline()
        print("\n=== Stage 3: CVRPEnv (PyVRP wrapper) ===")
        smoke_test_cvrp_env()
        print("\n=== Stage 4: Route Driver (400 nodes) ===")
        smoke_test_route_driver()
        print("\n=== Stage 4: Route Driver FP16 ===")
        smoke_test_route_driver_fp16()
        print("\n=== Stage 5: Training (2 PPO iterations) ===")
        smoke_test_training()
        print("\n=== Stage 6: Action Masking & NV_min ===")
        smoke_test_action_masking()
        print("\nAll smoke tests passed.")
